import CoreGraphics
@preconcurrency import CoreML
import CoreVideo
import Darwin
import Foundation
import UIKit

/// Runs a bundled calibration depth model and writes `depthpro_metric_depth.bin`
/// next to the generated room so `WallMeasurementEstimator` can consume a depth buffer on save/manual calibration.
@MainActor
final class DepthProMetricDepthService: ObservableObject {
    static let shared = DepthProMetricDepthService()

    @Published private(set) var isLoadingModel = false

    private let minimumSafeRemainingBytes: UInt64 = 2_500_000_000
    private let candidateBaseNames = [
        "DepthAnythingV2MetricIndoorSmall",
        "DepthAnythingV2MetricIndoorSmallF16",
        "depthanythingv2metricindoorsmall",
        "depthanythingv2metricindoorsmallf16",
        "DepthAnythingV2SmallF16INT8",
        "DepthAnythingV2SmallF16",
        "depthanythingv2smallf16int8",
        "depthanythingv2smallf16",
        "DepthProPrunedQuantized",
        "DepthProQuantized",
        "depthpro_pruned_quantized",
        "depthpro_quantized",
        "DepthProCanonical",
        "DepthProMetric",
        "depthpro_canonical",
        "depthpro_metric",
    ]

    private var model: MLModel?
    private var modelLoadTask: Task<Void, Never>?
    private var loadedModelName: String?

    private init() {}

    func ensureModelLoaded() {
        guard model == nil else { return }
        guard modelLoadTask == nil else { return }
        modelLoadTask = Task { [weak self] in
            guard let self else { return }
            defer { self.modelLoadTask = nil }
            await loadModel()
        }
    }

    func waitForModelReady(maxWaitSeconds: TimeInterval = 15) async {
        _ = maxWaitSeconds
        if model != nil { return }
        ensureModelLoaded()
        if let modelLoadTask {
            logDepthPro("WAITING FOR MODEL LOAD")
            await modelLoadTask.value
        }
    }

    func releaseResources() {
        guard model != nil || isLoadingModel else { return }
        model = nil
        loadedModelName = nil
        isLoadingModel = false
        logDepthPro("MODEL RELEASED")
    }

    func generateMetricDepthIfPossible(
        roomURL: URL,
        thumbnail: UIImage,
        referenceDepthMeters: Float? = nil
    ) async {
        let candidateURLs = candidateModelURLs()
        let packageSizeBytes = candidateURLs.map(Self.packageSizeOnDisk).max() ?? 0
        guard canSafelyLoadModel(packageSizeBytes: packageSizeBytes) else {
            logDepthPro(
                    "SKIPPED INFERENCE ROOM=\(roomURL.lastPathComponent) " +
                    "REASON=INSUFFICIENT_MEMORY PACKAGE_GB=\(String(format: "%.2f", Double(packageSizeBytes) / 1_000_000_000))"
            )
            return
        }

        await waitForModelReady()
        guard let model else {
            logDepthPro("MODEL UNAVAILABLE; SKIPPING METRIC DEPTH FOR \(roomURL.lastPathComponent)")
            return
        }
        let loadedModelName = self.loadedModelName ?? "UNKNOWN_MODEL"

        let sidecar = CameraExifSidecar.load(roomURL: roomURL)
        let start = CFAbsoluteTimeGetCurrent()

        do {
                    let result = try await runPredictionAsync(
                        model: model,
                        modelName: loadedModelName,
                        thumbnail: thumbnail,
                        sidecar: sidecar,
                        referenceDepthMeters: referenceDepthMeters
                    )
            try Self.writeDepthBuffer(
                width: result.width,
                height: result.height,
                channels: 1,
                values: result.depthMeters,
                to: Self.metricDepthURL(forRoomURL: roomURL)
            )

            var additions: [String: Double] = [
                "depthProMetricDepthWidthPx": Double(result.width),
                "depthProMetricDepthHeightPx": Double(result.height),
            ]
            if result.focalLengthPx > 0 {
                additions["focalLengthPx"] = Double(result.focalLengthPx)
            }
            if result.predictedFocalLengthPx > 0 {
                additions["depthProEstimatedFocalLengthPx"] = Double(result.predictedFocalLengthPx)
            }
            if result.fovDegrees > 0 {
                additions["depthProFovDegrees"] = Double(result.fovDegrees)
            }
            if sidecar["imageWidthPx"] == nil {
                additions["imageWidthPx"] = Double(result.sourceImageWidth)
            }
            if sidecar["imageHeightPx"] == nil {
                additions["imageHeightPx"] = Double(result.sourceImageHeight)
            }
            CameraExifSidecar.mergeDerivedValues(roomURL: roomURL, additions: additions)

            let elapsedMs = (CFAbsoluteTimeGetCurrent() - start) * 1000
            let anchorLog: String
            if let referenceDepthMeters, referenceDepthMeters > 0 {
                anchorLog = " REFERENCE_DEPTH_M=\(String(format: "%.3f", referenceDepthMeters))"
            } else {
                anchorLog = ""
            }
            logDepthPro(
                "METRIC DEPTH READY ROOM=\(roomURL.lastPathComponent) " +
                    "BUFFER=\(result.width)X\(result.height) " +
                    "SOURCE_IMAGE=\(result.sourceImageWidth)X\(result.sourceImageHeight) " +
                    "FOCAL_PX=\(String(format: "%.2f", result.focalLengthPx)) " +
                    "FOCAL_SOURCE=\(result.focalSource) " +
                    "FOV_DEG=\(String(format: "%.2f", result.fovDegrees))" +
                    anchorLog + " " +
                    "ELAPSED_MS=\(String(format: "%.1f", elapsedMs))"
            )
        } catch {
            logDepthPro(
                "METRIC DEPTH FAILED ROOM=\(roomURL.lastPathComponent) ERROR=\(error.localizedDescription)"
            )
        }

        releaseResources()
    }

    // MARK: - Model loading

    private func loadModel() async {
        guard model == nil, !isLoadingModel else { return }
        isLoadingModel = true
        defer { isLoadingModel = false }

        let urls = candidateModelURLs()
        guard !urls.isEmpty else {
            logDepthPro("NO CALIBRATION DEPTH COREML MODEL FOUND IN APP BUNDLE")
            return
        }

        let packageSizeBytes = urls.map(Self.packageSizeOnDisk).max() ?? 0
        let order = computeUnitLoadOrder(packageSizeBytes: packageSizeBytes)
        logDepthPro(
            "MODEL LOAD ORDER PHYSICAL_GB=\(String(format: "%.2f", physicalMemoryGB())) " +
                "PACKAGE_GB=\(String(format: "%.2f", Double(packageSizeBytes) / 1_000_000_000)) " +
                "ORDER=\(order.map(\.description).joined(separator: ","))"
        )

        for computeUnits in order {
            let config = MLModelConfiguration()
            config.computeUnits = computeUnits

            for url in urls {
                do {
                    let modelURL = try await Self.modelURLForLoad(sourceURL: url, packageSizeBytes: Self.packageSizeOnDisk(url))
                    model = try await Self.loadModel(contentsOf: modelURL, configuration: config)
                    loadedModelName = modelURL.deletingPathExtension().lastPathComponent
                    logDepthPro(
                        "MODEL LOADED NAME=\(modelURL.lastPathComponent) COMPUTE_UNITS=\(computeUnits.description)"
                    )
                    return
                } catch {
                    logDepthPro(
                        "MODEL LOAD FAILED NAME=\(url.lastPathComponent) COMPUTE_UNITS=\(computeUnits.description) ERROR=\(error.localizedDescription)"
                    )
                }
            }
        }
    }

    private func canSafelyLoadModel(packageSizeBytes: Int) -> Bool {
        guard let footprintBytes = Self.currentPhysicalFootprintBytes() else {
            logDepthPro("MEMORY GATE FOOTPRINT_UNAVAILABLE SAFE=true")
            return true
        }
        let deviceBytes = ProcessInfo.processInfo.physicalMemory
        let remainingBytes = deviceBytes > footprintBytes ? deviceBytes - footprintBytes : 0
        let packageBytes = UInt64(max(packageSizeBytes, 0))
        let requiredBytes = max(minimumSafeRemainingBytes, packageBytes * 2)
        let safe = remainingBytes >= requiredBytes
        logDepthPro(
            "MEMORY GATE AVAILABLE_MB=\(remainingBytes / 1_000_000) " +
                "REQUIRED_MB=\(requiredBytes / 1_000_000) " +
                "FOOTPRINT_MB=\(footprintBytes / 1_000_000) " +
                "SAFE=\(safe)"
        )
        return safe
    }

    private func candidateModelURLs() -> [URL] {
        guard let resourceRoot = Bundle.main.resourceURL else { return [] }
        guard let enumerator = FileManager.default.enumerator(
            at: resourceRoot,
            includingPropertiesForKeys: [.isRegularFileKey, .isDirectoryKey],
            options: [.skipsHiddenFiles]
        ) else {
            return []
        }

        var byBaseName: [String: URL] = [:]
        for case let url as URL in enumerator {
            let last = url.lastPathComponent
            if candidateBaseNames.contains(url.deletingPathExtension().lastPathComponent),
               last.hasSuffix(".mlmodelc") || last.hasSuffix(".mlpackage") {
                let base = url.deletingPathExtension().lastPathComponent
                if byBaseName[base] == nil {
                    byBaseName[base] = url
                }
            }
        }
        return candidateBaseNames.compactMap { byBaseName[$0] }
    }

    nonisolated private static func metricDepthURL(forRoomURL roomURL: URL) -> URL {
        let roomFolder = roomURL.deletingLastPathComponent()
        let stem = canonicalStem(forRoomURL: roomURL)
        return roomFolder.appendingPathComponent("\(stem)_depthpro_metric_depth.bin")
    }

    nonisolated private static func canonicalStem(forRoomURL roomURL: URL) -> String {
        var stem = roomURL.deletingPathExtension().lastPathComponent
        if stem.hasSuffix("_classic") {
            stem = String(stem.dropLast("_classic".count))
        }
        return stem
    }

    // MARK: - Prediction

    private func runPredictionAsync(
        model: MLModel,
        modelName: String,
        thumbnail: UIImage,
        sidecar: [String: Double],
        referenceDepthMeters: Float?
    ) async throws -> PredictionResult {
        return try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    let result = try Self.runPrediction(
                        model: model,
                        modelName: modelName,
                        thumbnail: thumbnail,
                        sidecar: sidecar,
                        referenceDepthMeters: referenceDepthMeters
                    )
                    continuation.resume(returning: result)
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    nonisolated private static func runPrediction(
        model: MLModel,
        modelName: String,
        thumbnail: UIImage,
        sidecar: [String: Double],
        referenceDepthMeters: Float?
    ) throws -> PredictionResult {
        let inputSize = modelInputSize(for: model)
        let sourceSize = sourceImagePixelSize(for: thumbnail)
        guard let preparedInput = makeLetterboxedPixelBuffer(
            from: thumbnail,
            sourceWidth: sourceSize.width,
            sourceHeight: sourceSize.height,
            inputWidth: inputSize.width,
            inputHeight: inputSize.height
        ) else {
            throw DepthProError.invalidImage("FAILED TO CREATE LETTERBOXED PIXEL BUFFER")
        }
        let provider = try makeInputProvider(
            model: model,
            pixelBuffer: preparedInput.pixelBuffer,
            sourceImageWidth: sourceSize.width
        )
        let output = try model.prediction(from: provider)

        if let imageBuffer = findDepthImageOutput(in: output) {
            let dense = try denseImage(from: imageBuffer)
            let remapped = remapDepthToSourceImage(
                values: dense.values,
                width: dense.width,
                height: dense.height,
                layout: preparedInput.layout,
                targetWidth: sourceSize.width,
                targetHeight: sourceSize.height
            )
            let exactFocal = exactFocalLengthPx(sidecar: sidecar, imageWidth: sourceSize.width)
            let focal = exactFocal?.value ?? 0
            let focalSource = exactFocal?.source ?? "MODEL_ONLY"
            let fovDegrees = exactFocal.map { focalLengthPxToFovDegrees(focalPx: $0.value, imageWidth: sourceSize.width) } ?? 0

            if modelOutputsMetricDepth(modelName: modelName) {
                logDepthPro(
                    "PREDICTION OUTPUTS MODE=IMAGE_METRIC_DIRECT " +
                        "MODEL=\(modelName.uppercased()) " +
                        "DEPTH=\(remapped.width)X\(remapped.height) " +
                        "MODEL_INPUT=\(dense.width)X\(dense.height) " +
                        "SOURCE_IMAGE=\(sourceSize.width)X\(sourceSize.height) " +
                        "FOCAL_PX=\(String(format: "%.2f", focal)) SOURCE=\(focalSource) " +
                        "FOV_DEG=\(String(format: "%.2f", fovDegrees))"
                )

                return PredictionResult(
                    width: remapped.width,
                    height: remapped.height,
                    sourceImageWidth: sourceSize.width,
                    sourceImageHeight: sourceSize.height,
                    depthMeters: remapped.values,
                    focalLengthPx: focal,
                    predictedFocalLengthPx: focal,
                    fovDegrees: fovDegrees,
                    focalSource: focalSource
                )
            }

            let scaled = try scaleRelativeDepthImage(
                remapped.values,
                width: remapped.width,
                height: remapped.height,
                referenceDepthMeters: referenceDepthMeters
            )

            logDepthPro(
                "PREDICTION OUTPUTS MODE=IMAGE_RELATIVE " +
                    "MODEL=\(modelName.uppercased()) " +
                    "DEPTH=\(remapped.width)X\(remapped.height) " +
                    "MODEL_INPUT=\(dense.width)X\(dense.height) " +
                    "SOURCE_IMAGE=\(sourceSize.width)X\(sourceSize.height) " +
                    "CENTER_RAW=\(String(format: "%.4f", scaled.centerRawDepth)) " +
                    "SCALE=\(String(format: "%.6f", scaled.scale)) " +
                    "REFERENCE_DEPTH_M=\(String(format: "%.4f", scaled.referenceDepthMeters)) " +
                    "FOCAL_PX=\(String(format: "%.2f", focal)) SOURCE=\(focalSource) " +
                    "FOV_DEG=\(String(format: "%.2f", fovDegrees))"
            )

            return PredictionResult(
                width: remapped.width,
                height: remapped.height,
                sourceImageWidth: sourceSize.width,
                sourceImageHeight: sourceSize.height,
                depthMeters: scaled.values,
                focalLengthPx: focal,
                predictedFocalLengthPx: focal,
                fovDegrees: fovDegrees,
                focalSource: focalSource
            )
        }

        if let metricFeature = findMetricDepthOutput(in: output) {
            let dense = try denseArray(from: metricFeature)
            guard dense.shape.count >= 2 else {
                throw DepthProError.invalidOutput("METRIC DEPTH SHAPE INVALID \(dense.shape)")
            }
            let depthHeight = dense.shape[dense.shape.count - 2]
            let depthWidth = dense.shape[dense.shape.count - 1]
            guard depthHeight > 0, depthWidth > 0 else {
                throw DepthProError.invalidOutput("METRIC DEPTH SIZE INVALID \(depthWidth)X\(depthHeight)")
            }
            let remapped = remapDepthToSourceImage(
                values: dense.values,
                width: depthWidth,
                height: depthHeight,
                layout: preparedInput.layout,
                targetWidth: sourceSize.width,
                targetHeight: sourceSize.height
            )

            let exactFocal = exactFocalLengthPx(sidecar: sidecar, imageWidth: sourceSize.width)
            let focal = exactFocal?.value ?? 0
            let focalSource = exactFocal?.source ?? "DEPTH_PRO_MODEL_ONLY"
            let fovDegrees = exactFocal.map { focalLengthPxToFovDegrees(focalPx: $0.value, imageWidth: sourceSize.width) } ?? 0

            logDepthPro(
                "PREDICTION OUTPUTS MODE=METRIC_DIRECT " +
                    "DEPTH=\(remapped.width)X\(remapped.height) " +
                    "MODEL_INPUT=\(depthWidth)X\(depthHeight) " +
                    "SOURCE_IMAGE=\(sourceSize.width)X\(sourceSize.height) " +
                    "FOCAL_PX=\(String(format: "%.2f", focal)) SOURCE=\(focalSource) " +
                    "FOV_DEG=\(String(format: "%.2f", fovDegrees))"
            )

            return PredictionResult(
                width: remapped.width,
                height: remapped.height,
                sourceImageWidth: sourceSize.width,
                sourceImageHeight: sourceSize.height,
                depthMeters: remapped.values,
                focalLengthPx: focal,
                predictedFocalLengthPx: focal,
                fovDegrees: fovDegrees,
                focalSource: focalSource
            )
        }

        guard let canonicalFeature = findCanonicalInverseDepthOutput(in: output) else {
            throw DepthProError.invalidOutput("CANONICAL OR METRIC DEPTH OUTPUT NOT FOUND")
        }
        guard let fovDegrees = findFovDegrees(in: output), fovDegrees.isFinite else {
            throw DepthProError.invalidOutput("FOV OUTPUT NOT FOUND")
        }

        let dense = try denseArray(from: canonicalFeature)
        guard dense.shape.count >= 2 else {
            throw DepthProError.invalidOutput("CANONICAL DEPTH SHAPE INVALID \(dense.shape)")
        }
        let depthHeight = dense.shape[dense.shape.count - 2]
        let depthWidth = dense.shape[dense.shape.count - 1]
        guard depthHeight > 0, depthWidth > 0 else {
            throw DepthProError.invalidOutput("CANONICAL DEPTH SIZE INVALID \(depthWidth)X\(depthHeight)")
        }
        let remappedCanonical = remapDepthToSourceImage(
            values: dense.values,
            width: depthWidth,
            height: depthHeight,
            layout: preparedInput.layout,
            targetWidth: sourceSize.width,
            targetHeight: sourceSize.height
        )

        let exactFocal = exactFocalLengthPx(sidecar: sidecar, imageWidth: sourceSize.width)
        let predictedFocal = 0.5 * Float(sourceSize.width) / tan(0.5 * fovDegrees * .pi / 180)
        let focal = exactFocal?.value ?? predictedFocal
        let focalSource = exactFocal?.source ?? "DEPTH_PRO_FOV"

        guard focal.isFinite, focal > 1e-3 else {
            throw DepthProError.invalidOutput("FOCAL LENGTH INVALID \(focal)")
        }

        let scale = Float(sourceSize.width) / focal
        var depthMeters = [Float](repeating: 0, count: remappedCanonical.width * remappedCanonical.height)
        for idx in 0..<depthMeters.count {
            let inverseDepth = remappedCanonical.values[idx] * scale
            let clampedInverseDepth = max(1e-4, min(1e4, inverseDepth))
            depthMeters[idx] = 1 / clampedInverseDepth
        }

        logDepthPro(
            "PREDICTION OUTPUTS DEPTH=\(depthWidth)X\(depthHeight) " +
                "REMAPPED=\(remappedCanonical.width)X\(remappedCanonical.height) " +
                "SOURCE_IMAGE=\(sourceSize.width)X\(sourceSize.height) " +
                "FOV_DEG=\(String(format: "%.2f", fovDegrees)) " +
                "PREDICTED_FOCAL_PX=\(String(format: "%.2f", predictedFocal)) " +
                "FINAL_FOCAL_PX=\(String(format: "%.2f", focal)) SOURCE=\(focalSource)"
        )

        return PredictionResult(
            width: remappedCanonical.width,
            height: remappedCanonical.height,
            sourceImageWidth: sourceSize.width,
            sourceImageHeight: sourceSize.height,
            depthMeters: depthMeters,
            focalLengthPx: focal,
            predictedFocalLengthPx: predictedFocal,
            fovDegrees: fovDegrees,
            focalSource: focalSource
        )
    }

    nonisolated private static func makeInputProvider(
        model: MLModel,
        pixelBuffer: CVPixelBuffer,
        sourceImageWidth: Int
    ) throws -> MLDictionaryFeatureProvider {
        let inputDescriptions = model.modelDescription.inputDescriptionsByName
        var dictionary: [String: MLFeatureValue] = [:]

        for (name, description) in inputDescriptions {
            switch description.type {
            case .image:
                dictionary[name] = MLFeatureValue(pixelBuffer: pixelBuffer)
            case .multiArray:
                let lowered = name.lowercased()
                if lowered.contains("originalwidth") || lowered == "original_width" {
                    dictionary[name] = try MLFeatureValue(
                        multiArray: scalarMultiArray(
                            constraint: description.multiArrayConstraint,
                            value: Float(sourceImageWidth)
                        )
                    )
                }
            default:
                break
            }
        }

        if dictionary.isEmpty {
            throw DepthProError.invalidInput("MODEL HAS NO BINDABLE INPUTS")
        }
        if dictionary["image"] == nil,
           inputDescriptions["image"] != nil {
            throw DepthProError.invalidInput("FEATURE IMAGE IS REQUIRED BUT NOT SPECIFIED")
        }

        let bound = dictionary.keys.sorted().joined(separator: ",")
        logDepthPro("INPUT BINDINGS \(bound)")
        return try MLDictionaryFeatureProvider(dictionary: dictionary)
    }

    nonisolated private static func modelInputSize(for model: MLModel) -> (width: Int, height: Int) {
        for (_, description) in model.modelDescription.inputDescriptionsByName {
            guard description.type == .image,
                  let constraint = description.imageConstraint
            else { continue }
            let width = max(1, Int(constraint.pixelsWide))
            let height = max(1, Int(constraint.pixelsHigh))
            return (width, height)
        }
        return (1536, 1536)
    }

    nonisolated private static func scalarMultiArray(
        constraint: MLMultiArrayConstraint?,
        value: Float
    ) throws -> MLMultiArray {
        let shape = constraint?.shape ?? [1, 1, 1, 1]
        let dataType = constraint?.dataType ?? .float32
        let array = try MLMultiArray(shape: shape, dataType: dataType)
        array[0] = NSNumber(value: value)
        return array
    }

    nonisolated private static func focalLengthPxToFovDegrees(focalPx: Float, imageWidth: Int) -> Float {
        guard focalPx.isFinite, focalPx > 1e-3, imageWidth > 0 else { return 0 }
        return 2 * atan(0.5 * Float(imageWidth) / focalPx) * 180 / .pi
    }

    nonisolated private static func packageSizeOnDisk(_ url: URL) -> Int {
        let fileManager = FileManager.default
        if !fileManager.fileExists(atPath: url.path, isDirectory: nil) {
            return 0
        }
        if let values = try? url.resourceValues(forKeys: [.totalFileAllocatedSizeKey, .fileAllocatedSizeKey, .fileSizeKey]),
           let size = values.totalFileAllocatedSize ?? values.fileAllocatedSize ?? values.fileSize {
            if url.pathExtension == "mlmodelc" || !url.hasDirectoryPath {
                return size
            }
        }
        guard let enumerator = fileManager.enumerator(
            at: url,
            includingPropertiesForKeys: [.totalFileAllocatedSizeKey, .fileAllocatedSizeKey, .fileSizeKey],
            options: [.skipsHiddenFiles]
        ) else {
            return 0
        }
        var total = 0
        for case let fileURL as URL in enumerator {
            let values = try? fileURL.resourceValues(forKeys: [.totalFileAllocatedSizeKey, .fileAllocatedSizeKey, .fileSizeKey])
            total += values?.totalFileAllocatedSize ?? values?.fileAllocatedSize ?? values?.fileSize ?? 0
        }
        return total
    }

    nonisolated private static func currentPhysicalFootprintBytes() -> UInt64? {
        var vmInfo = task_vm_info_data_t()
        var vmCount = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let result = withUnsafeMutablePointer(to: &vmInfo) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(vmCount)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &vmCount)
            }
        }
        guard result == KERN_SUCCESS else { return nil }
        return UInt64(vmInfo.phys_footprint)
    }

    nonisolated private static func modelURLForLoad(sourceURL: URL, packageSizeBytes: Int) async throws -> URL {
        if sourceURL.pathExtension == "mlmodelc" {
            return sourceURL
        }
        return try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .utility).async {
                do {
                    let fileManager = FileManager.default
                    let cachesRoot = try fileManager.url(for: .cachesDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
                    let compiledRoot = cachesRoot.appendingPathComponent("DepthProCompiled", isDirectory: true)
                    try fileManager.createDirectory(at: compiledRoot, withIntermediateDirectories: true, attributes: nil)

                    let baseName = sourceURL.deletingPathExtension().lastPathComponent
                    let compiledURL = compiledRoot.appendingPathComponent("\(baseName)_\(packageSizeBytes).mlmodelc", isDirectory: true)
                    if fileManager.fileExists(atPath: compiledURL.path) {
                        continuation.resume(returning: compiledURL)
                        return
                    }

                    let tempCompiledURL = try MLModel.compileModel(at: sourceURL)
                    if fileManager.fileExists(atPath: compiledURL.path) {
                        try? fileManager.removeItem(at: compiledURL)
                    }
                    try fileManager.moveItem(at: tempCompiledURL, to: compiledURL)
                    continuation.resume(returning: compiledURL)
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    nonisolated private static func loadModel(
        contentsOf url: URL,
        configuration: MLModelConfiguration
    ) async throws -> MLModel {
        try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .utility).async {
                do {
                    let loaded = try MLModel(contentsOf: url, configuration: configuration)
                    continuation.resume(returning: loaded)
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    private func computeUnitLoadOrder(packageSizeBytes: Int) -> [MLComputeUnits] {
        _ = packageSizeBytes
        return [.cpuOnly]
    }

    private func physicalMemoryGB() -> Double {
        Double(ProcessInfo.processInfo.physicalMemory) / 1_000_000_000
    }

    nonisolated private static func findMetricDepthOutput(in provider: MLFeatureProvider) -> MLMultiArray? {
        let preferredNames = [
            "depthMeters",
            "depth_meters",
            "metric_depth",
        ]
        for name in preferredNames {
            if let value = provider.featureValue(for: name)?.multiArrayValue {
                return value
            }
        }
        for name in provider.featureNames {
            guard let value = provider.featureValue(for: name)?.multiArrayValue else { continue }
            let lowered = name.lowercased()
            if lowered.contains("depth") && !lowered.contains("inverse") {
                return value
            }
        }
        return nil
    }

    nonisolated private static func modelOutputsMetricDepth(modelName: String) -> Bool {
        let lowered = modelName.lowercased()
        return lowered.contains("metric") && lowered.contains("depthanything")
    }

    nonisolated private static func findDepthImageOutput(in provider: MLFeatureProvider) -> CVPixelBuffer? {
        let preferredNames = [
            "depth",
            "depth_image",
            "predicted_depth",
        ]
        for name in preferredNames {
            if let value = provider.featureValue(for: name)?.imageBufferValue {
                return value
            }
        }
        for name in provider.featureNames {
            guard let value = provider.featureValue(for: name)?.imageBufferValue else { continue }
            if name.lowercased().contains("depth") {
                return value
            }
        }
        return nil
    }

    nonisolated private static func denseImage(from pixelBuffer: CVPixelBuffer) throws -> DenseImage {
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        guard width > 0, height > 0 else {
            throw DepthProError.invalidOutput("DEPTH IMAGE SIZE INVALID \(width)X\(height)")
        }

        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            throw DepthProError.invalidOutput("DEPTH IMAGE BASE ADDRESS MISSING")
        }

        let pixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        var values = [Float](repeating: 0, count: width * height)

        switch pixelFormat {
        case kCVPixelFormatType_OneComponent8:
            let ptr = baseAddress.assumingMemoryBound(to: UInt8.self)
            for y in 0..<height {
                let row = ptr.advanced(by: y * bytesPerRow)
                for x in 0..<width {
                    values[y * width + x] = Float(row[x]) / 255.0
                }
            }
        case kCVPixelFormatType_OneComponent16:
            let ptr = baseAddress.assumingMemoryBound(to: UInt16.self)
            let stride = bytesPerRow / MemoryLayout<UInt16>.size
            for y in 0..<height {
                let row = ptr.advanced(by: y * stride)
                for x in 0..<width {
                    values[y * width + x] = Float(row[x]) / 65535.0
                }
            }
        case kCVPixelFormatType_OneComponent16Half:
            let ptr = baseAddress.assumingMemoryBound(to: UInt16.self)
            let stride = bytesPerRow / MemoryLayout<UInt16>.size
            for y in 0..<height {
                let row = ptr.advanced(by: y * stride)
                for x in 0..<width {
                    values[y * width + x] = Float(Float16(bitPattern: row[x]))
                }
            }
        case kCVPixelFormatType_32BGRA:
            let ptr = baseAddress.assumingMemoryBound(to: UInt8.self)
            for y in 0..<height {
                let row = ptr.advanced(by: y * bytesPerRow)
                for x in 0..<width {
                    let offset = x * 4
                    let b = Float(row[offset]) / 255.0
                    let g = Float(row[offset + 1]) / 255.0
                    let r = Float(row[offset + 2]) / 255.0
                    values[y * width + x] = (r + g + b) / 3.0
                }
            }
        default:
            throw DepthProError.invalidOutput("UNSUPPORTED DEPTH IMAGE PIXEL FORMAT \(pixelFormat)")
        }

        return DenseImage(width: width, height: height, values: values)
    }

    nonisolated private static func scaleRelativeDepthImage(
        _ values: [Float],
        width: Int,
        height: Int,
        referenceDepthMeters: Float?
    ) throws -> (values: [Float], scale: Float, centerRawDepth: Float, referenceDepthMeters: Float) {
        guard let centerRawDepth = medianSample(values: values, width: width, height: height, centerX: width / 2, centerY: height / 2),
              centerRawDepth.isFinite, centerRawDepth > 1e-6 else {
            throw DepthProError.invalidOutput("RELATIVE DEPTH CENTER SAMPLE INVALID")
        }

        let anchorDepth = max(0.05, referenceDepthMeters ?? 1.0)
        let scale = anchorDepth / centerRawDepth
        let scaled = values.map { raw -> Float in
            guard raw.isFinite else { return 0 }
            return max(0, raw * scale)
        }
        return (scaled, scale, centerRawDepth, anchorDepth)
    }

    nonisolated private static func medianSample(
        values: [Float],
        width: Int,
        height: Int,
        centerX: Int,
        centerY: Int
    ) -> Float? {
        guard width > 0, height > 0, values.count == width * height else { return nil }
        var samples: [Float] = []
        samples.reserveCapacity(49)
        for dy in -3...3 {
            for dx in -3...3 {
                let x = min(max(centerX + dx, 0), width - 1)
                let y = min(max(centerY + dy, 0), height - 1)
                let value = values[y * width + x]
                if value.isFinite, value > 0 {
                    samples.append(value)
                }
            }
        }
        guard !samples.isEmpty else { return nil }
        samples.sort()
        return samples[samples.count / 2]
    }

    nonisolated private static func findCanonicalInverseDepthOutput(in provider: MLFeatureProvider) -> MLMultiArray? {
        let preferredNames = [
            "canonical_inverse_depth",
            "inverse_depth",
            "depth_logits",
            "var_0",
        ]
        for name in preferredNames {
            if let value = provider.featureValue(for: name)?.multiArrayValue {
                return value
            }
        }
        for name in provider.featureNames {
            guard let value = provider.featureValue(for: name)?.multiArrayValue else { continue }
            let lowered = name.lowercased()
            if lowered.contains("inverse") || lowered.contains("depth") {
                return value
            }
        }
        return nil
    }

    nonisolated private static func findFovDegrees(in provider: MLFeatureProvider) -> Float? {
        let preferredNames = ["fov_degrees", "fov_deg", "fov", "var_1"]
        for name in preferredNames {
            if let feature = provider.featureValue(for: name),
               let value = scalarFloat(from: feature) {
                return value
            }
        }
        for name in provider.featureNames {
            guard let feature = provider.featureValue(for: name) else { continue }
            if name.lowercased().contains("fov"), let value = scalarFloat(from: feature) {
                return value
            }
        }
        return nil
    }

    nonisolated private static func scalarFloat(from feature: MLFeatureValue) -> Float? {
        if let number = feature.multiArrayValue {
            if let dense = try? denseArray(from: number), let first = dense.values.first {
                return first
            }
        }
        if feature.type == .double { return Float(feature.doubleValue) }
        if feature.type == .int64 { return Float(feature.int64Value) }
        return nil
    }

    nonisolated private static func exactFocalLengthPx(
        sidecar: [String: Double],
        imageWidth: Int
    ) -> (value: Float, source: String)? {
        if let focalPx = sidecar["focalLengthPx"].map(Float.init), focalPx > 0.01 {
            return (focalPx, "SIDECAR_FOCAL_PX")
        }
        if let focalMM = sidecar["focalLengthMm"].map(Float.init), focalMM > 0.01 {
            let sensorWidthMM: Float?
            if let sensor = sidecar["sensorWidthMm"].map(Float.init), sensor > 0.01 {
                sensorWidthMM = sensor
            } else if let focal35 = sidecar["focalLength35mmEquivMm"].map(Float.init), focal35 > 0.01 {
                sensorWidthMM = 36.0 * focalMM / focal35
            } else {
                sensorWidthMM = nil
            }
            if let sensorWidthMM, sensorWidthMM > 0.01 {
                let focalPx = (focalMM / sensorWidthMM) * Float(imageWidth)
                return (focalPx, "EXIF_FOCAL_MM")
            }
        }
        return nil
    }

    nonisolated private static func sourceImagePixelSize(for image: UIImage) -> (width: Int, height: Int) {
        if let cg = image.cgImage {
            return (cg.width, cg.height)
        }
        let width = max(1, Int((image.size.width * image.scale).rounded()))
        let height = max(1, Int((image.size.height * image.scale).rounded()))
        return (width, height)
    }

    private struct LetterboxedInput {
        let pixelBuffer: CVPixelBuffer
        let layout: LetterboxLayout
    }

    private struct LetterboxLayout {
        let inputWidth: Int
        let inputHeight: Int
        let contentX: Float
        let contentY: Float
        let contentWidth: Float
        let contentHeight: Float
    }

    nonisolated private static func makeLetterboxedPixelBuffer(
        from image: UIImage,
        sourceWidth: Int,
        sourceHeight: Int,
        inputWidth: Int,
        inputHeight: Int
    ) -> LetterboxedInput? {
        guard sourceWidth > 0, sourceHeight > 0, inputWidth > 0, inputHeight > 0 else { return nil }
        guard let cgImage = image.cgImage else { return nil }
        let attrs: [CFString: Any] = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true,
        ]
        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            inputWidth,
            inputHeight,
            kCVPixelFormatType_32BGRA,
            attrs as CFDictionary,
            &pixelBuffer
        )
        guard status == kCVReturnSuccess, let pixelBuffer else { return nil }

        CVPixelBufferLockBaseAddress(pixelBuffer, [])
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, []) }

        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else { return nil }
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        guard let context = CGContext(
            data: baseAddress,
            width: inputWidth,
            height: inputHeight,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
        ) else {
            return nil
        }

        context.setFillColor(CGColor(gray: 0, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: inputWidth, height: inputHeight))

        let scale = min(Float(inputWidth) / Float(sourceWidth), Float(inputHeight) / Float(sourceHeight))
        let contentWidth = Float(sourceWidth) * scale
        let contentHeight = Float(sourceHeight) * scale
        let contentX = (Float(inputWidth) - contentWidth) * 0.5
        let contentY = (Float(inputHeight) - contentHeight) * 0.5

        context.interpolationQuality = CGInterpolationQuality.high
        context.draw(
            cgImage,
            in: CGRect(
                x: CGFloat(contentX),
                y: CGFloat(contentY),
                width: CGFloat(contentWidth),
                height: CGFloat(contentHeight)
            )
        )

        logDepthPro(
            "INPUT LETTERBOX SOURCE=\(sourceWidth)X\(sourceHeight) " +
                "MODEL_INPUT=\(inputWidth)X\(inputHeight) " +
                "CONTENT_RECT=(\(String(format: "%.2f", contentX)),\(String(format: "%.2f", contentY)),\(String(format: "%.2f", contentWidth)),\(String(format: "%.2f", contentHeight)))"
        )

        return LetterboxedInput(
            pixelBuffer: pixelBuffer,
            layout: LetterboxLayout(
                inputWidth: inputWidth,
                inputHeight: inputHeight,
                contentX: contentX,
                contentY: contentY,
                contentWidth: contentWidth,
                contentHeight: contentHeight
            )
        )
    }

    nonisolated private static func remapDepthToSourceImage(
        values: [Float],
        width: Int,
        height: Int,
        layout: LetterboxLayout,
        targetWidth: Int,
        targetHeight: Int
    ) -> DenseImage {
        guard width > 0, height > 0,
              targetWidth > 0, targetHeight > 0,
              values.count == width * height else {
            return DenseImage(width: width, height: height, values: values)
        }

        let scaleX = Float(width) / Float(layout.inputWidth)
        let scaleY = Float(height) / Float(layout.inputHeight)
        let contentX = layout.contentX * scaleX
        let contentY = layout.contentY * scaleY
        let contentWidth = max(1, layout.contentWidth * scaleX)
        let contentHeight = max(1, layout.contentHeight * scaleY)

        var remapped = [Float](repeating: 0, count: targetWidth * targetHeight)
        for y in 0..<targetHeight {
            let sourceY = contentY + ((Float(y) + 0.5) / Float(targetHeight)) * contentHeight - 0.5
            for x in 0..<targetWidth {
                let sourceX = contentX + ((Float(x) + 0.5) / Float(targetWidth)) * contentWidth - 0.5
                remapped[y * targetWidth + x] = bilinearSample(
                    values: values,
                    width: width,
                    height: height,
                    x: sourceX,
                    y: sourceY
                )
            }
        }

        return DenseImage(width: targetWidth, height: targetHeight, values: remapped)
    }

    nonisolated private static func bilinearSample(
        values: [Float],
        width: Int,
        height: Int,
        x: Float,
        y: Float
    ) -> Float {
        let clampedX = min(max(x, 0), Float(width - 1))
        let clampedY = min(max(y, 0), Float(height - 1))
        let x0 = Int(floor(clampedX))
        let y0 = Int(floor(clampedY))
        let x1 = min(x0 + 1, width - 1)
        let y1 = min(y0 + 1, height - 1)
        let tx = clampedX - Float(x0)
        let ty = clampedY - Float(y0)

        let v00 = values[y0 * width + x0]
        let v10 = values[y0 * width + x1]
        let v01 = values[y1 * width + x0]
        let v11 = values[y1 * width + x1]
        let top = v00 * (1 - tx) + v10 * tx
        let bottom = v01 * (1 - tx) + v11 * tx
        return top * (1 - ty) + bottom * ty
    }

    nonisolated private static func denseArray(from multiArray: MLMultiArray) throws -> DenseArray {
        let shape = multiArray.shape.map(\.intValue)
        let count = multiArray.count
        guard count > 0 else {
            throw DepthProError.invalidOutput("MULTIARRAY EMPTY")
        }

        if isRowMajorContiguous(multiArray) {
            switch multiArray.dataType {
            case .float32:
                let ptr = multiArray.dataPointer.bindMemory(to: Float.self, capacity: count)
                return DenseArray(shape: shape, values: Array(UnsafeBufferPointer(start: ptr, count: count)))
            case .double:
                let ptr = multiArray.dataPointer.bindMemory(to: Double.self, capacity: count)
                return DenseArray(shape: shape, values: UnsafeBufferPointer(start: ptr, count: count).map(Float.init))
            default:
                break
            }
        }

        var indices = [Int](repeating: 0, count: shape.count)
        var values = [Float]()
        values.reserveCapacity(count)

        func visit(_ dimension: Int) {
            if dimension == shape.count {
                let key = indices.map { NSNumber(value: $0) }
                values.append(multiArray[key].floatValue)
                return
            }
            for idx in 0..<shape[dimension] {
                indices[dimension] = idx
                visit(dimension + 1)
            }
        }

        visit(0)
        return DenseArray(shape: shape, values: values)
    }

    nonisolated private static func isRowMajorContiguous(_ multiArray: MLMultiArray) -> Bool {
        let shape = multiArray.shape.map(\.intValue)
        let strides = multiArray.strides.map(\.intValue)
        guard shape.count == strides.count else { return false }
        var expectedStride = 1
        for index in stride(from: shape.count - 1, through: 0, by: -1) {
            if strides[index] != expectedStride { return false }
            expectedStride *= max(shape[index], 1)
        }
        return true
    }

    nonisolated private static func writeDepthBuffer(
        width: Int,
        height: Int,
        channels: Int,
        values: [Float],
        to url: URL
    ) throws {
        guard values.count == width * height * channels else {
            throw DepthProError.invalidOutput(
                "DEPTH BUFFER COUNT MISMATCH VALUES=\(values.count) EXPECTED=\(width * height * channels)"
            )
        }

        var data = Data(capacity: 12 + values.count * MemoryLayout<Float>.size)
        for value in [Int32(width), Int32(height), Int32(channels)] {
            var little = value.littleEndian
            withUnsafeBytes(of: &little) { data.append(contentsOf: $0) }
        }
        for value in values {
            var bitPattern = value.bitPattern.littleEndian
            withUnsafeBytes(of: &bitPattern) { data.append(contentsOf: $0) }
        }
        try data.write(to: url, options: [.atomic])
    }
}

private extension MLComputeUnits {
    var description: String {
        switch self {
        case .all: return "ALL"
        case .cpuAndGPU: return "CPU_AND_GPU"
        case .cpuAndNeuralEngine: return "CPU_AND_NEURAL_ENGINE"
        case .cpuOnly: return "CPU_ONLY"
        @unknown default: return "UNKNOWN"
        }
    }
}

private struct DenseArray {
    let shape: [Int]
    let values: [Float]
}

private struct DenseImage {
    let width: Int
    let height: Int
    let values: [Float]
}

private struct PredictionResult {
    let width: Int
    let height: Int
    let sourceImageWidth: Int
    let sourceImageHeight: Int
    let depthMeters: [Float]
    let focalLengthPx: Float
    let predictedFocalLengthPx: Float
    let fovDegrees: Float
    let focalSource: String
}

private enum DepthProError: LocalizedError {
    case invalidImage(String)
    case invalidInput(String)
    case invalidOutput(String)

    var errorDescription: String? {
        switch self {
        case .invalidImage(let message), .invalidInput(let message), .invalidOutput(let message):
            return message
        }
    }
}
