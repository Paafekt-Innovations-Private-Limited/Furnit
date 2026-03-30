import CoreML
import CoreVideo
import Foundation

/// Minimal ONNX Runtime wrapper for the iOS Furniture Fit path.
/// It mirrors the Android YOLO-E ONNX model I/O: `images` -> `output0` + `output1`.
final class YOLOEOnnxRuntime {

    static let shared = YOLOEOnnxRuntime()

    let modelInputSize = 640

    private let inputName = "images"
    private var detectionOutputName = "det_output"
    private var prototypeOutputName = "proto_output"
    private let inputShape: [NSNumber]
    private let lock = NSLock()

    private var env: ORTEnv?
    private var session: ORTSession?
    private var didAttemptLoad = false
    private var cachedInputTensorData: NSMutableData?

    private init() {
        inputShape = [1, 3, NSNumber(value: modelInputSize), NSNumber(value: modelInputSize)]
    }

    var isAvailable: Bool {
        ensureSession() != nil
    }

    func run(pixelBuffer: CVPixelBuffer) -> (det: MLMultiArray, proto: MLMultiArray)? {
        guard CVPixelBufferGetWidth(pixelBuffer) == modelInputSize,
              CVPixelBufferGetHeight(pixelBuffer) == modelInputSize else {
            return nil
        }

        guard let session = ensureSession() else {
            return nil
        }

        do {
            let inputValue = try inputValue(for: pixelBuffer)
            let outputs = try session.run(
                withInputs: [inputName: inputValue],
                outputNames: [detectionOutputName, prototypeOutputName],
                runOptions: nil
            )

            guard let detValue = outputs[detectionOutputName],
                  let protoValue = outputs[prototypeOutputName] else {
                return nil
            }

            let detArray = try multiArray(from: detValue)
            let protoArray = try multiArray(from: protoValue)
            return (det: detArray, proto: protoArray)
        } catch {
            logDebug("YOLOE ONNX: inference failed: \(error)")
            return nil
        }
    }

    private func ensureSession() -> ORTSession? {
        lock.lock()
        defer { lock.unlock() }

        if let session {
            return session
        }

        if didAttemptLoad {
            return nil
        }

        didAttemptLoad = true

        let onnxResourceNames = [
            "yoloe-26l-seg-pf_seg_o2m",
            "yoloe-11l-seg-pf",
        ]
        guard let modelURL = onnxResourceNames.compactMap({ name in
            Bundle.main.url(forResource: name, withExtension: "onnx")
        }).first else {
            logDebug("YOLOE ONNX: bundled model missing (tried \(onnxResourceNames.joined(separator: ", ")))")
            return nil
        }

        do {
            let env = try ORTEnv(loggingLevel: .warning)
            let options = try ORTSessionOptions()
            try options.setGraphOptimizationLevel(.all)
            try options.setIntraOpNumThreads(Int32(max(1, ProcessInfo.processInfo.activeProcessorCount - 1)))
            try options.setLogSeverityLevel(.warning)

            let session = try ORTSession(env: env, modelPath: modelURL.path, sessionOptions: options)
            self.env = env
            self.session = session

            let inputNames = try session.inputNames()
            let outputNameList = try session.outputNames()
            logDebug("YOLOE ONNX: loaded \(modelURL.lastPathComponent) inputs=\(inputNames) outputs=\(outputNameList)")

            let outputSet = Set(outputNameList)
            if outputSet.contains("det_output"), outputSet.contains("proto_output") {
                detectionOutputName = "det_output"
                prototypeOutputName = "proto_output"
            } else if outputSet.contains("output0"), outputSet.contains("output1") {
                detectionOutputName = "output0"
                prototypeOutputName = "output1"
            } else {
                logDebug("YOLOE ONNX: unexpected output names \(outputNameList); need det_output/proto_output or output0/output1")
                self.session = nil
                self.env = nil
                return nil
            }

            return session
        } catch {
            logDebug("YOLOE ONNX: session load failed: \(error)")
            return nil
        }
    }

    private func inputValue(for pixelBuffer: CVPixelBuffer) throws -> ORTValue {
        let tensorData = reusableInputTensorData()
        let floats = tensorData.mutableBytes.assumingMemoryBound(to: Float.self)
        fillNchwRgbTensor(from: pixelBuffer, destination: floats)
        return try ORTValue(tensorData: tensorData, elementType: .float, shape: inputShape)
    }

    private func reusableInputTensorData() -> NSMutableData {
        let requiredLength = modelInputSize * modelInputSize * 3 * MemoryLayout<Float>.size
        if let cachedInputTensorData, cachedInputTensorData.length == requiredLength {
            return cachedInputTensorData
        }

        let freshTensorData = NSMutableData(length: requiredLength) ?? NSMutableData()
        cachedInputTensorData = freshTensorData
        return freshTensorData
    }

    private func fillNchwRgbTensor(from pixelBuffer: CVPixelBuffer, destination: UnsafeMutablePointer<Float>) {
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            return
        }

        let width = modelInputSize
        let height = modelInputSize
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let hw = width * height
        let pixels = baseAddress.assumingMemoryBound(to: UInt8.self)

        let redPlane = destination
        let greenPlane = destination.advanced(by: hw)
        let bluePlane = destination.advanced(by: hw * 2)

        for y in 0..<height {
            let rowStart = pixels.advanced(by: y * bytesPerRow)
            for x in 0..<width {
                let pixelStart = rowStart.advanced(by: x * 4)
                let pixelIndex = y * width + x
                bluePlane[pixelIndex] = Float(pixelStart[0]) * (1.0 / 255.0)
                greenPlane[pixelIndex] = Float(pixelStart[1]) * (1.0 / 255.0)
                redPlane[pixelIndex] = Float(pixelStart[2]) * (1.0 / 255.0)
            }
        }
    }

    private func multiArray(from value: ORTValue) throws -> MLMultiArray {
        let tensorInfo = try value.tensorTypeAndShapeInfo()
        guard tensorInfo.elementType == .float else {
            throw NSError(
                domain: "YOLOEOnnxRuntime",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "Expected float tensor output"]
            )
        }

        let tensorData = try value.tensorData()
        let multiArray = try MLMultiArray(shape: tensorInfo.shape, dataType: .float32)
        let requiredBytes = multiArray.count * MemoryLayout<Float>.size
        guard tensorData.length >= requiredBytes else {
            throw NSError(
                domain: "YOLOEOnnxRuntime",
                code: 2,
                userInfo: [NSLocalizedDescriptionKey: "Unexpected ONNX tensor byte count"]
            )
        }

        memcpy(multiArray.dataPointer, tensorData.mutableBytes, requiredBytes)
        return multiArray
    }
}
