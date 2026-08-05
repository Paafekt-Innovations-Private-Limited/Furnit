import CoreML
import Dispatch
import Foundation
import TensorFlowLiteC
import TensorFlowLiteCMetal

enum RTMDetLiteRuntimeError: LocalizedError {
    case modelOpenFailed(String)
    case runtimeWorkerStopped
    case metalDelegateCreationFailed
    case metalDelegateApplicationFailed(Int32)
    case delegationAuditCreationFailed
    case delegationAuditApplicationFailed(Int32)
    case partialMetalDelegation(
        executionPlanNodes: Int,
        metalPartitions: Int,
        metalOriginalNodes: Int,
        remainingCPUNodes: Int,
        otherDelegateNodes: Int
    )
    case interpreterCreationFailed
    case signatureMissing([String])
    case signatureRunnerCreationFailed
    case tensorAllocationFailed
    case warmUpFailed
    case invalidTensor(String)
    case invocationFailed
    case outputCopyFailed(String)

    var errorDescription: String? {
        switch self {
        case .modelOpenFailed(let path):
            return "LiteRT could not open the RTMDet model at \(path)."
        case .runtimeWorkerStopped:
            return "The dedicated RTMDet LiteRT worker has stopped."
        case .metalDelegateCreationFailed:
            return "LiteRT could not create the iOS Metal delegate."
        case .metalDelegateApplicationFailed(let status):
            return "LiteRT could not apply the mandatory RTMDet Metal delegate (status \(status))."
        case .delegationAuditCreationFailed:
            return "LiteRT could not create the mandatory RTMDet Metal delegation audit."
        case .delegationAuditApplicationFailed(let status):
            return "LiteRT could not inspect the delegated RTMDet graph (status \(status))."
        case let .partialMetalDelegation(
            executionPlanNodes,
            metalPartitions,
            metalOriginalNodes,
            remainingCPUNodes,
            otherDelegateNodes
        ):
            return "RTMDet was not fully delegated to Metal "
                + "(plan=\(executionPlanNodes), Metal partitions=\(metalPartitions), "
                + "Metal ops=\(metalOriginalNodes), CPU nodes=\(remainingCPUNodes), "
                + "other delegate nodes=\(otherDelegateNodes))."
        case .interpreterCreationFailed:
            return "LiteRT could not create the Metal RTMDet interpreter."
        case .signatureMissing(let available):
            return "RTMDet serving_default signature is missing (available: \(available.joined(separator: ", ")))."
        case .signatureRunnerCreationFailed:
            return "LiteRT could not create the RTMDet signature runner."
        case .tensorAllocationFailed:
            return "LiteRT could not allocate RTMDet tensors on Metal."
        case .warmUpFailed:
            return "LiteRT could not warm up the RTMDet Metal graph."
        case .invalidTensor(let message):
            return "RTMDet tensor contract mismatch: \(message)"
        case .invocationFailed:
            return "LiteRT Metal RTMDet inference failed."
        case .outputCopyFailed(let name):
            return "LiteRT could not synchronize the Metal RTMDet output \(name)."
        }
    }
}

/// A persistent worker thread for the complete LiteRT/Metal lifecycle. A serial dispatch queue
/// can move between kernel threads; this object deliberately cannot. Android likewise creates,
/// warms, invokes, and closes its GPU runtime on one dedicated executor thread.
private final class RTMDetLiteRuntimeWorker: @unchecked Sendable {
    private final class State: @unchecked Sendable {
        let condition = NSCondition()
        let started = DispatchSemaphore(value: 0)
        let finished = DispatchSemaphore(value: 0)
        var jobs: [() -> Void] = []
        var stopping = false
    }

    private final class WorkItem<Value>: @unchecked Sendable {
        private let body: () throws -> Value
        private let completion = DispatchSemaphore(value: 0)
        private var result: Result<Value, Error>?

        init(body: @escaping () throws -> Value) {
            self.body = body
        }

        func execute() {
            result = Result { try body() }
            completion.signal()
        }

        func waitForResult() throws -> Value {
            completion.wait()
            guard let result else {
                throw RTMDetLiteRuntimeError.runtimeWorkerStopped
            }
            return try result.get()
        }
    }

    private let state: State
    private let thread: Thread

    init(name: String) {
        let state = State()
        self.state = state
        thread = Thread {
            state.started.signal()
            while true {
                state.condition.lock()
                while state.jobs.isEmpty && !state.stopping {
                    state.condition.wait()
                }
                if state.jobs.isEmpty && state.stopping {
                    state.condition.unlock()
                    break
                }
                let job = state.jobs.removeFirst()
                state.condition.unlock()
                autoreleasepool {
                    job()
                }
            }
            state.finished.signal()
        }
        thread.name = name
        thread.qualityOfService = .userInitiated
        thread.start()
        state.started.wait()
    }

    func sync<Value>(_ body: @escaping () throws -> Value) throws -> Value {
        if Thread.current === thread {
            return try body()
        }

        let item = WorkItem(body: body)
        state.condition.lock()
        guard !state.stopping else {
            state.condition.unlock()
            throw RTMDetLiteRuntimeError.runtimeWorkerStopped
        }
        state.jobs.append { item.execute() }
        state.condition.signal()
        state.condition.unlock()
        return try item.waitForResult()
    }

    func stop() {
        state.condition.lock()
        let wasAlreadyStopping = state.stopping
        state.stopping = true
        state.condition.broadcast()
        state.condition.unlock()

        if !wasAlreadyStopping && Thread.current !== thread {
            state.finished.wait()
        }
    }

    deinit {
        stop()
    }
}

/// The single iOS RTMDet backend. It runs a mathematically equivalent Metal-compatible
/// variant of Android's FP16 `.tflite` graph with the same `serving_default` tensor contract.
/// There is deliberately no Core ML or CPU runtime fallback hidden behind this object.
final class RTMDetLiteRuntime: @unchecked Sendable {
    static let modelName = "rtmdet-ins-m-raw-fp16"
    static let modelExtension = "tflite"
    static let signatureKey = "serving_default"
    static let inputName = "input"

    static let outputNames = [
        "cls_80", "cls_40", "cls_20",
        "bbox_80", "bbox_40", "bbox_20",
        "kernel_80", "kernel_40", "kernel_20",
        "mask_feat",
    ]

    static let expectedInputShape = [1, 640, 640, 3]
    static let expectedOutputShapes: [String: [Int]] = [
        "cls_80": [1, 80, 80, 80],
        "cls_40": [1, 40, 40, 80],
        "cls_20": [1, 20, 20, 80],
        "bbox_80": [1, 80, 80, 4],
        "bbox_40": [1, 40, 40, 4],
        "bbox_20": [1, 20, 20, 4],
        "kernel_80": [1, 80, 80, 169],
        "kernel_40": [1, 40, 40, 169],
        "kernel_20": [1, 20, 20, 169],
        "mask_feat": [1, 160, 160, 8],
    ]

    let inputWidth = 640
    let inputHeight = 640
    let runtimeVersion: String
    let delegationSummary: String

    /// Owns reusable NHWC extraction and NCHW decoder storage for one Metal-backed output.
    /// Android uses the same explicit NHWC -> NCHW conversion. Keeping that conversion literal
    /// here avoids relying on a custom-stride `MLMultiArray` view over delegate output memory.
    private final class OutputStorage {
        let byteCount: Int
        let spatialCount: Int
        let channelCount: Int
        let nhwcStore: NSMutableData
        let nchwStore: NSMutableData
        let array: MLMultiArray

        init(expectedShape: [Int]) throws {
            let n = expectedShape[0]
            let h = expectedShape[1]
            let w = expectedShape[2]
            let c = expectedShape[3]
            guard n == 1 else {
                throw RTMDetLiteRuntimeError.invalidTensor("only batch size 1 is supported")
            }
            spatialCount = h * w
            channelCount = c
            byteCount = expectedShape.reduce(1, *) * MemoryLayout<Float>.stride
            guard let nhwcStore = NSMutableData(length: byteCount),
                  let nchwStore = NSMutableData(length: byteCount) else {
                throw RTMDetLiteRuntimeError.invalidTensor("could not allocate output storage")
            }
            self.nhwcStore = nhwcStore
            self.nchwStore = nchwStore
            array = try MLMultiArray(
                dataPointer: nchwStore.mutableBytes,
                shape: [n, c, h, w].map(NSNumber.init(value:)),
                dataType: .float32,
                strides: [c * h * w, h * w, w, 1].map(NSNumber.init(value:)),
                deallocator: nil
            )
        }

        func copyAndTranspose(from tensor: UnsafePointer<TfLiteTensor>, name: String) throws {
            guard TfLiteTensorCopyToBuffer(
                tensor,
                nhwcStore.mutableBytes,
                byteCount
            ) == kTfLiteOk else {
                throw RTMDetLiteRuntimeError.outputCopyFailed(name)
            }

            let source = nhwcStore.bytes.assumingMemoryBound(to: Float.self)
            let destination = nchwStore.mutableBytes.assumingMemoryBound(to: Float.self)
            for spatialIndex in 0..<spatialCount {
                let sourceOffset = spatialIndex * channelCount
                for channel in 0..<channelCount {
                    destination[channel * spatialCount + spatialIndex] = source[sourceOffset + channel]
                }
            }
        }
    }

    private struct DelegationAudit {
        let executionPlanNodes: Int
        let metalPartitions: Int
        let metalOriginalNodes: Int
        let remainingCPUNodes: Int
        let otherDelegateNodes: Int

        var summary: String {
            "planNodes=\(executionPlanNodes) "
                + "metalPartitions=\(metalPartitions) "
                + "metalOps=\(metalOriginalNodes) "
                + "cpuNodes=\(remainingCPUNodes) "
                + "otherDelegateNodes=\(otherDelegateNodes)"
        }
    }

    private struct Handles {
        let model: OpaquePointer
        let interpreter: OpaquePointer
        let signatureRunner: OpaquePointer
        let metalDelegate: UnsafeMutablePointer<TfLiteDelegate>
        let delegationAuditHandle: OpaquePointer
        let delegationAudit: DelegationAudit
        let outputStorages: [String: OutputStorage]
    }

    private let worker: RTMDetLiteRuntimeWorker
    private let handles: Handles

    init(modelURL: URL) throws {
        let worker = RTMDetLiteRuntimeWorker(name: "com.paafekt.rtmdet.litert-metal")
        do {
            let handles = try worker.sync {
                try Self.makeHandles(modelURL: modelURL)
            }
            self.worker = worker
            self.handles = handles
            runtimeVersion = TfLiteVersion().map { String(cString: $0) } ?? "unknown"
            delegationSummary = handles.delegationAudit.summary
        } catch {
            worker.stop()
            throw error
        }
    }

    deinit {
        let handles = self.handles
        _ = try? worker.sync {
            Self.destroy(handles)
        }
        worker.stop()
    }

    /// Provides direct writable access to LiteRT's input tensor, then copies Metal-backed outputs
    /// into persistent storage and converts them from NHWC to contiguous NCHW exactly as Android
    /// does. The dedicated worker remains occupied through `consumeOutputs`; consumers must not
    /// retain the arrays.
    func invoke<Result>(
        prepareInput: @escaping (UnsafeMutableBufferPointer<Float>) throws -> Void,
        consumeOutputs: @escaping ([(name: String, array: MLMultiArray)]) throws -> Result
    ) throws -> Result {
        // `sync` takes an escaping closure, so bind the handles locally instead of capturing `self`.
        let handles = self.handles
        return try worker.sync {
            let inputTensor = try Self.tensor(
                named: Self.inputName,
                in: handles.signatureRunner,
                expectedShape: Self.expectedInputShape,
                mutable: true
            )
            guard let inputData = TfLiteTensorData(inputTensor) else {
                throw RTMDetLiteRuntimeError.invalidTensor("input data pointer is nil")
            }
            let inputElementCount = Self.expectedInputShape.reduce(1, *)
            guard TfLiteTensorByteSize(inputTensor) == inputElementCount * MemoryLayout<Float>.stride else {
                throw RTMDetLiteRuntimeError.invalidTensor(
                    "input byte count is not \(inputElementCount * MemoryLayout<Float>.stride)"
                )
            }

            try prepareInput(
                UnsafeMutableBufferPointer(
                    start: inputData.assumingMemoryBound(to: Float.self),
                    count: inputElementCount
                )
            )

            guard TfLiteSignatureRunnerInvoke(handles.signatureRunner) == kTfLiteOk else {
                throw RTMDetLiteRuntimeError.invocationFailed
            }

            var outputArrays: [(name: String, array: MLMultiArray)] = []
            outputArrays.reserveCapacity(Self.outputNames.count)
            for name in Self.outputNames {
                guard let expectedShape = Self.expectedOutputShapes[name] else { continue }
                guard let storage = handles.outputStorages[name] else {
                    throw RTMDetLiteRuntimeError.invalidTensor("missing output storage for \(name)")
                }
                let tensor = try Self.tensor(
                    named: name,
                    in: handles.signatureRunner,
                    expectedShape: expectedShape,
                    mutable: false
                )
                guard TfLiteTensorByteSize(tensor) == storage.byteCount else {
                    throw RTMDetLiteRuntimeError.invalidTensor(
                        "\(name) byte count is \(TfLiteTensorByteSize(tensor)), expected \(storage.byteCount)"
                    )
                }
                try storage.copyAndTranspose(from: tensor, name: name)
                outputArrays.append((name, storage.array))
            }

            return try consumeOutputs(outputArrays)
        }
    }

    private static func makeHandles(modelURL: URL) throws -> Handles {
        guard let model = modelURL.path.withCString({ TfLiteModelCreateFromFile($0) }) else {
            throw RTMDetLiteRuntimeError.modelOpenFailed(modelURL.path)
        }

        guard let options = TfLiteInterpreterOptionsCreate() else {
            TfLiteModelDelete(model)
            throw RTMDetLiteRuntimeError.interpreterCreationFailed
        }
        defer { TfLiteInterpreterOptionsDelete(options) }

        // Delegate fallback is different from partial delegation. Disable automatic fallback here;
        // the explicit post-Metal audit below separately rejects any residual CPU execution plan.
        TfLiteInterpreterOptionsSetEnableDelegateFallback(options, false)

        var interpreterToCleanUp: OpaquePointer?
        var runnerToCleanUp: OpaquePointer?
        var metalDelegateToCleanUp: UnsafeMutablePointer<TfLiteDelegate>?
        var auditToCleanUp: OpaquePointer?

        do {
            var delegateOptions = TFLGpuDelegateOptionsDefault()
            // Match Android's GPU semantics. FP16 constants enter through DEQUANTIZE, so disabling
            // quantization support silently strands most of this graph on CPU. Precision-loss mode
            // is also Android's explicit option and is the intended fast Metal arithmetic mode.
            delegateOptions.allow_precision_loss = true
            delegateOptions.wait_type = TFLGpuDelegateWaitTypePassive
            delegateOptions.enable_quantization = true
            guard let metalDelegate = withUnsafePointer(
                to: &delegateOptions,
                { TFLGpuDelegateCreate($0) }
            ) else {
                throw RTMDetLiteRuntimeError.metalDelegateCreationFailed
            }
            metalDelegateToCleanUp = metalDelegate

            // Create first, then apply Metal explicitly. This guarantees that the audit delegate's
            // Prepare callback sees the execution plan *after* Metal partitioning.
            guard let interpreter = TfLiteInterpreterCreate(model, options) else {
                throw RTMDetLiteRuntimeError.interpreterCreationFailed
            }
            interpreterToCleanUp = interpreter

            let metalStatus = TfLiteInterpreterModifyGraphWithDelegate(interpreter, metalDelegate)
            guard metalStatus == kTfLiteOk else {
                throw RTMDetLiteRuntimeError.metalDelegateApplicationFailed(
                    Int32(metalStatus.rawValue)
                )
            }

            guard let auditHandle = FurnitLiteRTDelegationAuditCreate(metalDelegate),
                  let auditDelegate = FurnitLiteRTDelegationAuditGetDelegate(auditHandle) else {
                throw RTMDetLiteRuntimeError.delegationAuditCreationFailed
            }
            auditToCleanUp = auditHandle

            let auditStatus = TfLiteInterpreterModifyGraphWithDelegate(interpreter, auditDelegate)
            guard auditStatus == kTfLiteOk else {
                throw RTMDetLiteRuntimeError.delegationAuditApplicationFailed(
                    Int32(auditStatus.rawValue)
                )
            }

            let delegationAudit = DelegationAudit(
                executionPlanNodes: Int(FurnitLiteRTDelegationAuditExecutionPlanNodeCount(auditHandle)),
                metalPartitions: Int(FurnitLiteRTDelegationAuditMetalPartitionCount(auditHandle)),
                metalOriginalNodes: Int(FurnitLiteRTDelegationAuditMetalOriginalNodeCount(auditHandle)),
                remainingCPUNodes: Int(FurnitLiteRTDelegationAuditRemainingCPUNodeCount(auditHandle)),
                otherDelegateNodes: Int(FurnitLiteRTDelegationAuditOtherDelegateNodeCount(auditHandle))
            )
            guard FurnitLiteRTDelegationAuditDidPrepare(auditHandle),
                  delegationAudit.executionPlanNodes > 0,
                  delegationAudit.metalPartitions > 0,
                  delegationAudit.metalOriginalNodes > 0,
                  delegationAudit.remainingCPUNodes == 0,
                  delegationAudit.otherDelegateNodes == 0 else {
                throw RTMDetLiteRuntimeError.partialMetalDelegation(
                    executionPlanNodes: delegationAudit.executionPlanNodes,
                    metalPartitions: delegationAudit.metalPartitions,
                    metalOriginalNodes: delegationAudit.metalOriginalNodes,
                    remainingCPUNodes: delegationAudit.remainingCPUNodes,
                    otherDelegateNodes: delegationAudit.otherDelegateNodes
                )
            }

            let signatureCount = TfLiteInterpreterGetSignatureCount(interpreter)
            let availableSignatures = (0..<signatureCount).compactMap { index -> String? in
                TfLiteInterpreterGetSignatureKey(interpreter, index).map(String.init(cString:))
            }
            guard availableSignatures.contains(signatureKey) else {
                throw RTMDetLiteRuntimeError.signatureMissing(availableSignatures)
            }

            guard let runner = signatureKey.withCString({
                TfLiteInterpreterGetSignatureRunner(interpreter, $0)
            }) else {
                throw RTMDetLiteRuntimeError.signatureRunnerCreationFailed
            }
            runnerToCleanUp = runner

            guard TfLiteSignatureRunnerAllocateTensors(runner) == kTfLiteOk else {
                throw RTMDetLiteRuntimeError.tensorAllocationFailed
            }

            let inputTensor = try tensor(
                named: inputName,
                in: runner,
                expectedShape: expectedInputShape,
                mutable: true
            )
            for name in outputNames {
                guard let expectedShape = expectedOutputShapes[name] else { continue }
                _ = try tensor(
                    named: name,
                    in: runner,
                    expectedShape: expectedShape,
                    mutable: false
                )
            }

            // Match Android's runtime preparation: compile/execute the delegated graph once at
            // load time so the first camera frame does not absorb Metal's one-time warm-up cost.
            guard let inputData = TfLiteTensorData(inputTensor) else {
                throw RTMDetLiteRuntimeError.invalidTensor("input data pointer is nil during warm-up")
            }
            inputData.initializeMemory(
                as: UInt8.self,
                repeating: 0,
                count: TfLiteTensorByteSize(inputTensor)
            )
            guard TfLiteSignatureRunnerInvoke(runner) == kTfLiteOk else {
                throw RTMDetLiteRuntimeError.warmUpFailed
            }

            var outputStorages: [String: OutputStorage] = [:]
            outputStorages.reserveCapacity(outputNames.count)
            for name in outputNames {
                guard let expectedShape = expectedOutputShapes[name] else { continue }
                outputStorages[name] = try OutputStorage(expectedShape: expectedShape)
            }

            return Handles(
                model: model,
                interpreter: interpreter,
                signatureRunner: runner,
                metalDelegate: metalDelegate,
                delegationAuditHandle: auditHandle,
                delegationAudit: delegationAudit,
                outputStorages: outputStorages
            )
        } catch {
            if let runnerToCleanUp {
                TfLiteSignatureRunnerDelete(runnerToCleanUp)
            }
            if let interpreterToCleanUp {
                TfLiteInterpreterDelete(interpreterToCleanUp)
            }
            if let auditToCleanUp {
                FurnitLiteRTDelegationAuditDelete(auditToCleanUp)
            }
            if let metalDelegateToCleanUp {
                TFLGpuDelegateDelete(metalDelegateToCleanUp)
            }
            TfLiteModelDelete(model)
            throw error
        }
    }

    private static func destroy(_ handles: Handles) {
        TfLiteSignatureRunnerDelete(handles.signatureRunner)
        TfLiteInterpreterDelete(handles.interpreter)
        FurnitLiteRTDelegationAuditDelete(handles.delegationAuditHandle)
        TFLGpuDelegateDelete(handles.metalDelegate)
        TfLiteModelDelete(handles.model)
    }

    private static func tensor(
        named name: String,
        in runner: OpaquePointer,
        expectedShape: [Int],
        mutable: Bool
    ) throws -> UnsafePointer<TfLiteTensor> {
        let tensor: UnsafePointer<TfLiteTensor>? = name.withCString { cName in
            if mutable {
                if let inputTensor = TfLiteSignatureRunnerGetInputTensor(runner, cName) {
                    return UnsafePointer(inputTensor)
                }
                return nil
            }
            return TfLiteSignatureRunnerGetOutputTensor(runner, cName)
        }
        guard let tensor else {
            throw RTMDetLiteRuntimeError.invalidTensor("missing \(name)")
        }
        guard TfLiteTensorType(tensor) == kTfLiteFloat32 else {
            throw RTMDetLiteRuntimeError.invalidTensor("\(name) is not float32")
        }
        let dimensionCount = Int(TfLiteTensorNumDims(tensor))
        let shape = (0..<dimensionCount).map { Int(TfLiteTensorDim(tensor, Int32($0))) }
        guard shape == expectedShape else {
            throw RTMDetLiteRuntimeError.invalidTensor("\(name) shape \(shape), expected \(expectedShape)")
        }
        return tensor
    }
}
