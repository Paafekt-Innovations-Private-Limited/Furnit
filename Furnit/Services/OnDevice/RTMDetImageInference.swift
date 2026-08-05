import CoreGraphics
import CoreImage
import CoreML
import CoreVideo
import Accelerate
import Foundation
import UIKit

struct RTMDetInferenceResult {
    let detections: [FurnitureFitDetection]
    let overlayMaskImage: UIImage?
    let instanceMaskImages: [UIImage?]
    let maskAffinityGraph: RTMDetMaskAffinityGraph?
    let outputSummary: [String]
    let maskBuildCache: RTMDetMaskBuildCache?
}

struct RTMDetMaskAffinityGraph {
    fileprivate let neighbors: [[Int]]
    var nodeCount: Int { neighbors.count }

    func transitiveGroup(seedIndices: [Int]) -> [Int] {
        guard !seedIndices.isEmpty, !neighbors.isEmpty else { return [] }
        var inGroup = Set(seedIndices.filter { $0 >= 0 && $0 < neighbors.count })
        var frontier = Array(inGroup)
        while let current = frontier.popLast() {
            for neighbor in neighbors[current] where !inGroup.contains(neighbor) {
                inGroup.insert(neighbor)
                frontier.append(neighbor)
            }
        }
        return Array(inGroup).sorted()
    }
}

struct RTMDetCachedMaskResult {
    let image: UIImage
    /// Indices (in cache-detection-space) of detections whose masks were unioned
    /// into ``image`` via transitive affinity overlap.
    let affinityGroupIndices: [Int]
}

struct RTMDetMaskBuildCache {
    let sourceWidth: Int
    let sourceHeight: Int
    let detections: [FurnitureFitDetection]
    fileprivate let sourceBuffer: CVPixelBuffer
    fileprivate let candidates: [RTMDetImageInference.RawCandidate]
    fileprivate let mappedBoxes: [RTMDetImageInference.BoxRecord]
    fileprivate let maskFeatureMatrix: RTMDetImageInference.RawMaskFeatureMatrix
    fileprivate let mapping: RTMDetImageInference.ImageMapping
}

enum RTMDetImageInference {
    private static var lastInputTensorStats = "input=unavailable"
    private static let rawMaskModelSide = 640
    private static let rawMaskSide = 160
    private static let rawMaskAffinityThreshold: Float = 0.12
    private static let rawMaskAffinityBitThreshold: Float = 0.5
    private static let rawMaskRenderThreshold: Float = 0.30
    /// A narrow probability-space transition gives the final cutout subpixel coverage without a
    /// blur/filter pass. The midpoint remains the existing render threshold, so the visible contour
    /// does not move.
    private static let rawMaskRenderAntialiasHalfWidth: Float = 0.05

    private struct BilinearMaskAxisSample {
        let lowerIndex: Int
        let upperIndex: Int
        let upperWeight: Float
    }

    private enum PixelRGBLayout {
        case bgra
        case rgba
        case argb

        init(pixelFormat: OSType) {
            switch pixelFormat {
            case kCVPixelFormatType_32RGBA:
                self = .rgba
            case kCVPixelFormatType_32ARGB:
                self = .argb
            default:
                self = .bgra
            }
        }
    }

    private static func rgbBytes(
        from row: UnsafePointer<UInt8>,
        offset: Int,
        layout: PixelRGBLayout
    ) -> (r: UInt8, g: UInt8, b: UInt8) {
        switch layout {
        case .bgra:
            return (row[offset + 2], row[offset + 1], row[offset])
        case .rgba:
            return (row[offset], row[offset + 1], row[offset + 2])
        case .argb:
            return (row[offset + 1], row[offset + 2], row[offset + 3])
        }
    }

    /// Converts the already-computed mask probability into edge coverage. Most pixels take one of
    /// the two fast branches; only the narrow boundary band performs the interpolation.
    @inline(__always)
    static func rawMaskRenderAlpha(for probability: Float) -> UInt8 {
        guard probability.isFinite else { return 0 }
        let lower = rawMaskRenderThreshold - rawMaskRenderAntialiasHalfWidth
        let upper = rawMaskRenderThreshold + rawMaskRenderAntialiasHalfWidth
        if probability <= lower { return 0 }
        if probability >= upper { return 255 }
        let coverage = (probability - lower) / (upper - lower)
        return UInt8((coverage * 255).rounded())
    }

    @inline(__always)
    private static func premultiplied(_ component: UInt8, alpha: UInt8) -> UInt8 {
        UInt8((UInt16(component) * UInt16(alpha) + 127) / 255)
    }

    private static func isDirectRGBPixelFormat(_ pixelFormat: OSType) -> Bool {
        pixelFormat == kCVPixelFormatType_32BGRA ||
            pixelFormat == kCVPixelFormatType_32RGBA ||
            pixelFormat == kCVPixelFormatType_32ARGB
    }

    /// Shared GPU-backed Core Image context. Creating a `CIContext` allocates Metal/GPU state, so it
    /// must never be built per frame — reuse this one for every resize. `CIContext` rendering is
    /// thread-safe; inference runs on a single serial queue regardless.
    private static let sharedCIContext = CIContext()

    private static func rgbCompositingBuffer(
        for buffer: CVPixelBuffer,
        width: Int,
        height: Int
    ) -> CVPixelBuffer? {
        guard width > 0, height > 0 else { return nil }
        let pixelFormat = CVPixelBufferGetPixelFormatType(buffer)
        if isDirectRGBPixelFormat(pixelFormat) {
            return buffer
        }

        let attributes: [CFString: Any] = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true,
            kCVPixelBufferIOSurfacePropertiesKey: [:]
        ]
        var convertedBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            width,
            height,
            kCVPixelFormatType_32BGRA,
            attributes as CFDictionary,
            &convertedBuffer
        )
        guard status == kCVReturnSuccess, let convertedBuffer else { return nil }

        var image = CIImage(cvPixelBuffer: buffer)
        let extent = image.extent
        if extent.origin != .zero {
            image = image.transformed(by: CGAffineTransform(translationX: -extent.origin.x, y: -extent.origin.y))
        }
        if extent.width > 0,
           extent.height > 0,
           (abs(extent.width - CGFloat(width)) > 0.5 || abs(extent.height - CGFloat(height)) > 0.5) {
            image = image.transformed(
                by: CGAffineTransform(
                    scaleX: CGFloat(width) / extent.width,
                    y: CGFloat(height) / extent.height
                )
            )
        }

        sharedCIContext.render(
            image,
            to: convertedBuffer,
            bounds: CGRect(x: 0, y: 0, width: width, height: height),
            colorSpace: CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
        )
        return convertedBuffer
    }

    /// Reused pool of square model-input buffers so each frame recycles a buffer instead of calling
    /// `CVPixelBufferCreate`. Recreated only if the model input side changes. (Inference is serial.)
    private static var resizeBufferPool: CVPixelBufferPool?
    private static var resizeBufferPoolSize: Int = 0

    private static func squareInputBuffer(size: Int) -> CVPixelBuffer? {
        if resizeBufferPool == nil || resizeBufferPoolSize != size {
            let attrs: [CFString: Any] = [
                kCVPixelBufferIOSurfacePropertiesKey: [:],
                kCVPixelBufferMetalCompatibilityKey: true,
                kCVPixelBufferCGImageCompatibilityKey: true,
                kCVPixelBufferCGBitmapContextCompatibilityKey: true,
                kCVPixelBufferWidthKey: size,
                kCVPixelBufferHeightKey: size,
                kCVPixelBufferPixelFormatTypeKey: kCVPixelFormatType_32BGRA,
            ]
            var pool: CVPixelBufferPool?
            guard CVPixelBufferPoolCreate(kCFAllocatorDefault, nil, attrs as CFDictionary, &pool) == kCVReturnSuccess else {
                return nil
            }
            resizeBufferPool = pool
            resizeBufferPoolSize = size
        }
        guard let pool = resizeBufferPool else { return nil }
        var buffer: CVPixelBuffer?
        guard CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, pool, &buffer) == kCVReturnSuccess else {
            return nil
        }
        return buffer
    }

    // MARK: - Image helpers (relocated from the removed RTMDet path)

    /// The Android-parity LiteRT graph has a fixed 640×640 NHWC input contract.
    static func modelInputSize(for model: RTMDetLiteRuntime) -> Int {
        model.inputWidth
    }

    /// Loads a JPEG/PNG from disk into a BGRA `CVPixelBuffer`.
    static func pixelBufferFromImage(atPath path: String) -> CVPixelBuffer? {
        guard let img = UIImage(contentsOfFile: path) else { return nil }
        return pixelBufferFromImage(img)
    }

    /// Loads a `UIImage` into a BGRA `CVPixelBuffer` (same layout as the live camera path).
    static func pixelBufferFromImage(_ image: UIImage) -> CVPixelBuffer? {
        let format = UIGraphicsImageRendererFormat()
        format.scale = image.scale
        format.opaque = false
        let renderer = UIGraphicsImageRenderer(size: image.size, format: format)
        let drawn = renderer.image { _ in image.draw(in: CGRect(origin: .zero, size: image.size)) }
        guard let cgImage = drawn.cgImage else { return nil }
        let w = cgImage.width
        let h = cgImage.height
        var pb: CVPixelBuffer?
        let attrs: [CFString: Any] = [
            kCVPixelBufferIOSurfacePropertiesKey: [:],
            kCVPixelBufferMetalCompatibilityKey: true,
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true
        ]
        guard CVPixelBufferCreate(kCFAllocatorDefault, w, h, kCVPixelFormatType_32BGRA, attrs as CFDictionary, &pb) == kCVReturnSuccess,
              let buffer = pb else { return nil }
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard let ctx = CGContext(
            data: CVPixelBufferGetBaseAddress(buffer),
            width: w,
            height: h,
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
        ) else { return nil }
        ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: w, height: h))
        return buffer
    }

    private static func copyPixelBuffer(_ source: CVPixelBuffer) -> CVPixelBuffer? {
        CVPixelBufferLockBaseAddress(source, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(source, .readOnly) }
        let width = CVPixelBufferGetWidth(source)
        let height = CVPixelBufferGetHeight(source)
        let pixelFormat = CVPixelBufferGetPixelFormatType(source)
        var copy: CVPixelBuffer?
        let attrs: [CFString: Any] = [
            kCVPixelBufferIOSurfacePropertiesKey: [:],
            kCVPixelBufferMetalCompatibilityKey: true,
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true
        ]
        guard CVPixelBufferCreate(kCFAllocatorDefault, width, height, pixelFormat, attrs as CFDictionary, &copy) == kCVReturnSuccess,
              let copy else { return nil }
        CVPixelBufferLockBaseAddress(copy, [])
        defer { CVPixelBufferUnlockBaseAddress(copy, []) }
        guard let srcBase = CVPixelBufferGetBaseAddress(source),
              let dstBase = CVPixelBufferGetBaseAddress(copy) else { return nil }
        let srcRowBytes = CVPixelBufferGetBytesPerRow(source)
        let dstRowBytes = CVPixelBufferGetBytesPerRow(copy)
        let rowBytesToCopy = min(srcRowBytes, dstRowBytes)
        for y in 0..<height {
            memcpy(
                dstBase.advanced(by: y * dstRowBytes),
                srcBase.advanced(by: y * srcRowBytes),
                rowBytesToCopy
            )
        }
        return copy
    }

    fileprivate struct BoxRecord {
        let rowIndex: Int
        let x1: Float
        let y1: Float
        let x2: Float
        let y2: Float
        let score: Float
        let classIdx: Int?
    }

    private enum CoordinateSpace {
        case normalized
        case modelSquare
        case sourceImage
    }

    fileprivate struct ImageMapping {
        let modelSide: Int
        let sourceWidth: Int
        let sourceHeight: Int
        let usesLetterbox: Bool
    }

    private struct MaskPlaneInfo {
        let threshold: Float
        let maxValue: Float
        let minX: Int
        let minY: Int
        let maxX: Int
        let maxY: Int
    }

    fileprivate struct RawCandidate {
        let box: BoxRecord
        let kernel: [Float]
        let priorX: Float
        let priorY: Float
        let stride: Float
    }

    fileprivate struct RawMaskFeatureMatrix {
        let side: Int
        let pixelCount: Int
        /// Row-major `[channel][pixel]`, shape 8 x pixelCount.
        let features: [Float]
    }

    /// Per-stage wall-clock (ms) for live-frame profiling; carried into the postprocess so the whole
    /// breakdown is emitted on one debug line.
    private struct StageMillis {
        let resize: Double
        let input: Double
        let predict: Double
    }

    static func runInstanceSegmentation(
        image: UIImage,
        model: RTMDetLiteRuntime,
        confidenceThreshold: Float = 0.25,
        classBlacklist: Set<Int> = [],
        allowedClassIndices: Set<Int>? = nil,
        maxMaskCount: Int = 6,
        maxDetectionCount: Int? = nil,
        buildInstanceMasks: Bool = false,
        cacheMaskBuildInputs: Bool = false,
        restrictInstanceMasksToFrameCenter: Bool = false,
        debug: Bool = false
    ) throws -> RTMDetInferenceResult {
        guard let sourceBuffer = pixelBufferFromImage(image) else {
            throw NSError(
                domain: "RTMDetImageInference",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "UIImage -> CVPixelBuffer failed"],
            )
        }
        return try runInstanceSegmentation(
            pixelBuffer: sourceBuffer,
            model: model,
            confidenceThreshold: confidenceThreshold,
            classBlacklist: classBlacklist,
            allowedClassIndices: allowedClassIndices,
            maxMaskCount: maxMaskCount,
            maxDetectionCount: maxDetectionCount,
            buildInstanceMasks: buildInstanceMasks,
            cacheMaskBuildInputs: cacheMaskBuildInputs,
            restrictInstanceMasksToFrameCenter: restrictInstanceMasksToFrameCenter,
            debug: debug
        )
    }

    static func runInstanceSegmentation(
        pixelBuffer sourceBuffer: CVPixelBuffer,
        model: RTMDetLiteRuntime,
        confidenceThreshold: Float = 0.25,
        classBlacklist: Set<Int> = [],
        allowedClassIndices: Set<Int>? = nil,
        maxMaskCount: Int = 6,
        maxDetectionCount: Int? = nil,
        buildInstanceMasks: Bool = false,
        cacheMaskBuildInputs: Bool = false,
        restrictInstanceMasksToFrameCenter: Bool = false,
        debug: Bool = false
    ) throws -> RTMDetInferenceResult {
        let sourceWidth = CVPixelBufferGetWidth(sourceBuffer)
        let sourceHeight = CVPixelBufferGetHeight(sourceBuffer)
        let modelSide = modelInputSize(for: model)
        let usesLetterbox = false

        // Per-stage timing (cheap Date() marks) to locate the live-frame bottleneck; only the summary
        // string is built/logged in debug.
        let tStageStart = Date()
        guard let preparedBuffer = resizeStretchToSquare(src: sourceBuffer, size: modelSide) else {
            throw NSError(
                domain: "RTMDetImageInference",
                code: 2,
                userInfo: [NSLocalizedDescriptionKey: "Image preprocessing failed"],
            )
        }
        let tResized = Date()

        var tInputBuilt = tResized
        return try model.invoke(
            prepareInput: { inputBuffer in
                try writeRawBGRNHWC(
                    from: preparedBuffer,
                    to: inputBuffer,
                    collectStats: debug
                )
                tInputBuilt = Date()
            },
            consumeOutputs: { outputArrays in
                let tPredicted = Date()
                let preStageMillis = StageMillis(
                    resize: tResized.timeIntervalSince(tStageStart) * 1000,
                    input: tInputBuilt.timeIntervalSince(tResized) * 1000,
                    predict: tPredicted.timeIntervalSince(tInputBuilt) * 1000
                )
                let outputSummary = outputArrays.map { entry in
                    let shape = entry.array.shape.map(\.intValue).map(String.init).joined(separator: "x")
                    return "\(entry.name): \(shape) \(entry.array.dataType.debugName)"
                }
                guard isRawRTMDetOutput(outputArrays) else {
                    throw RTMDetLiteRuntimeError.invalidTensor(
                        "raw RTMDet outputs are incomplete: \(outputSummary.joined(separator: ", "))"
                    )
                }
                return try runRawHeadPostprocess(
                    outputArrays: outputArrays,
                    outputSummary: outputSummary,
                    sourceBuffer: sourceBuffer,
                    sourceWidth: sourceWidth,
                    sourceHeight: sourceHeight,
                    modelSide: modelSide,
                    usesLetterbox: usesLetterbox,
                    confidenceThreshold: confidenceThreshold,
                    classBlacklist: classBlacklist,
                    allowedClassIndices: allowedClassIndices,
                    maxMaskCount: maxMaskCount,
                    maxDetectionCount: maxDetectionCount,
                    buildInstanceMasks: buildInstanceMasks,
                    cacheMaskBuildInputs: cacheMaskBuildInputs,
                    preStageMillis: preStageMillis,
                    debug: debug
                )
            }
        )
    }

    private static func isRawRTMDetOutput(_ arrays: [(name: String, array: MLMultiArray)]) -> Bool {
        let names = Set(arrays.map(\.name))
        return names.contains("cls_80")
            && names.contains("bbox_80")
            && names.contains("kernel_80")
            && names.contains("mask_feat")
    }

    private static func runRawHeadPostprocess(
        outputArrays: [(name: String, array: MLMultiArray)],
        outputSummary: [String],
        sourceBuffer: CVPixelBuffer,
        sourceWidth: Int,
        sourceHeight: Int,
        modelSide: Int,
        usesLetterbox: Bool,
        confidenceThreshold: Float,
        classBlacklist: Set<Int>,
        allowedClassIndices: Set<Int>?,
        maxMaskCount: Int,
        maxDetectionCount: Int?,
        buildInstanceMasks: Bool,
        cacheMaskBuildInputs: Bool,
        preStageMillis: StageMillis,
        debug: Bool
    ) throws -> RTMDetInferenceResult {
        let arrays = Dictionary(uniqueKeysWithValues: outputArrays.map { ($0.name, $0.array) })
        guard let cls80 = arrays["cls_80"],
              let cls40 = arrays["cls_40"],
              let cls20 = arrays["cls_20"],
              let bbox80 = arrays["bbox_80"],
              let bbox40 = arrays["bbox_40"],
              let bbox20 = arrays["bbox_20"],
              let kernel80 = arrays["kernel_80"],
              let kernel40 = arrays["kernel_40"],
              let kernel20 = arrays["kernel_20"],
              let maskFeat = arrays["mask_feat"] else {
            throw NSError(
                domain: "RTMDetImageInference",
                code: 8,
                userInfo: [NSLocalizedDescriptionKey: "Raw RTMDet outputs are incomplete. Outputs: \(outputSummary.joined(separator: ", "))"],
            )
        }

        // A correctly delegated RTMDet head is mutually exclusive at an anchor: it may strongly
        // prefer one class, but it must not saturate many unrelated COCO classes simultaneously.
        // The former partial Metal/CPU execution produced exactly that signature (dozens of 1.00
        // scores and full-frame boxes). Refuse the frame before allocating/building any masks if
        // the runtime ever returns that impossible pattern again.
        try validateRawClassHeads([
            (array: cls80, side: 80),
            (array: cls40, side: 40),
            (array: cls20, side: 20),
        ])

        let classIndices: [Int]
        if let allowedClassIndices, !allowedClassIndices.isEmpty {
            classIndices = allowedClassIndices.sorted()
        } else {
            classIndices = Array(0..<80)
        }

        let tDecodeStart = Date()
        let rawCandidates = decodeRawCandidates(
            levels: [
                (cls: cls80, bbox: bbox80, kernel: kernel80, side: 80, stride: Float(modelSide) / 80),
                (cls: cls40, bbox: bbox40, kernel: kernel40, side: 40, stride: Float(modelSide) / 40),
                (cls: cls20, bbox: bbox20, kernel: kernel20, side: 20, stride: Float(modelSide) / 20),
            ],
            classIndices: classIndices,
            classBlacklist: classBlacklist,
            confidenceThreshold: confidenceThreshold,
            modelSide: modelSide,
            preNMSLimit: nil
        )
        let safetyLimit = maxDetectionCount ?? 200
        let selected = classAwareNMS(rawCandidates, iouThreshold: 0.5, limit: safetyLimit)
        let mapping = ImageMapping(
            modelSide: modelSide,
            sourceWidth: sourceWidth,
            sourceHeight: sourceHeight,
            usesLetterbox: usesLetterbox
        )
        let mappedBoxes = selected.map { mapBoxToSource(box: $0.box, mapping: mapping) }
        let mappedDetections = mappedBoxes.map { mapped in
            FurnitureFitDetection(
                x: (mapped.x1 + mapped.x2) * 0.5,
                y: (mapped.y1 + mapped.y2) * 0.5,
                w: max(1, mapped.x2 - mapped.x1),
                h: max(1, mapped.y2 - mapped.y1),
                confidence: mapped.score,
                classIdx: mapped.classIdx ?? -1
            )
        }

        let tDecoded = Date()
        let shouldPrepareMaskFeatures = maxMaskCount > 0 || buildInstanceMasks || cacheMaskBuildInputs
        let maskFeatureMatrix = shouldPrepareMaskFeatures ? rawMaskFeatureMatrix(from: maskFeat) : nil
        let rawMaskPlanes: [[Float]?]
        if shouldPrepareMaskFeatures {
            if let maskFeatureMatrix {
                rawMaskPlanes = selected.map { buildRawMaskPlane(candidate: $0, maskFeatureMatrix: maskFeatureMatrix) }
            } else {
                rawMaskPlanes = []
            }
        } else {
            rawMaskPlanes = []
        }
        let tMaskPlanes = Date()
        let maskBuildCache: RTMDetMaskBuildCache?
        if cacheMaskBuildInputs, let maskFeatureMatrix, let cachedSourceBuffer = copyPixelBuffer(sourceBuffer) {
            maskBuildCache = RTMDetMaskBuildCache(
                sourceWidth: sourceWidth,
                sourceHeight: sourceHeight,
                detections: mappedDetections,
                sourceBuffer: cachedSourceBuffer,
                candidates: selected,
                mappedBoxes: mappedBoxes,
                maskFeatureMatrix: maskFeatureMatrix,
                mapping: mapping
            )
        } else {
            maskBuildCache = nil
        }

        let combinedMask = maxMaskCount > 0
            ? buildCombinedRawMaskImage(
                rawMaskPlanes: rawMaskPlanes,
                boxes: mappedBoxes,
                maxMaskCount: maxMaskCount,
                sourceBuffer: sourceBuffer,
                mapping: mapping
            )
            : nil
        let tCombined = Date()
        let maskAffinityGraph = !rawMaskPlanes.isEmpty
            ? makeMaskAffinityGraph(rawMaskPlanes)
            : nil
        let instanceMaskImages: [UIImage?]
        if buildInstanceMasks {
            instanceMaskImages = mappedBoxes.indices.map { index in
                guard index < rawMaskPlanes.count else { return nil }
                let groupIndices = maskAffinityGraph?.transitiveGroup(seedIndices: [index]) ?? [index]
                let selectedIndices = groupIndices.isEmpty ? [index] : groupIndices
                return buildCombinedRawMaskImage(
                    rawMaskPlanes: selectedIndices.map { rawMaskPlanes[$0] },
                    boxes: selectedIndices.map { mappedBoxes[$0] },
                    maxMaskCount: selectedIndices.count,
                    sourceBuffer: sourceBuffer,
                    mapping: mapping
                )
            }
        } else {
            instanceMaskImages = []
        }
        let tInstances = Date()

        // Keep Debug timing useful without re-reading every tensor and mask only for logging.
        // Those diagnostic sweeps and their multi-kilobyte console output previously dominated
        // live-frame latency even after inference itself had completed.
        let debugSummary: [String] = debug
            ? [
                "stageMillis: resize=\(String(format: "%.1f", preStageMillis.resize)) "
                    + "input=\(String(format: "%.1f", preStageMillis.input)) "
                    + "predict=\(String(format: "%.1f", preStageMillis.predict)) "
                    + "decode=\(String(format: "%.1f", tDecoded.timeIntervalSince(tDecodeStart) * 1000)) "
                    + "maskPlanes=\(String(format: "%.1f", tMaskPlanes.timeIntervalSince(tDecoded) * 1000)) "
                    + "combined=\(String(format: "%.1f", tCombined.timeIntervalSince(tMaskPlanes) * 1000)) "
                    + "instances=\(String(format: "%.1f", tInstances.timeIntervalSince(tCombined) * 1000)) "
                    + "(planes=\(selected.count) buildInstanceMasks=\(buildInstanceMasks))",
                "rtmdetDims: sourceBuffer=\(sourceWidth)x\(sourceHeight) mapping=\(mapping.sourceWidth)x\(mapping.sourceHeight)",
                "rawSwiftDecode: candidates=\(rawCandidates.count) kept=\(selected.count)",
                "rtmdetInput: \(lastInputTensorStats)",
            ]
            : []
        return RTMDetInferenceResult(
            detections: mappedDetections,
            overlayMaskImage: combinedMask,
            instanceMaskImages: instanceMaskImages,
            maskAffinityGraph: maskAffinityGraph,
            outputSummary: outputSummary + debugSummary,
            maskBuildCache: maskBuildCache
        )
    }

    static func buildCachedMaskImage(
        from cache: RTMDetMaskBuildCache,
        detections requestedDetections: [FurnitureFitDetection],
        debug: Bool = false
    ) -> UIImage? {
        buildCachedMaskWithGroupInfo(from: cache, detections: requestedDetections, debug: debug)?.image
    }

    static func buildCachedMaskWithGroupInfo(
        from cache: RTMDetMaskBuildCache,
        detections requestedDetections: [FurnitureFitDetection],
        debug: Bool = false
    ) -> RTMDetCachedMaskResult? {
        guard !requestedDetections.isEmpty else { return nil }
        var selectedIndices: [Int] = []
        selectedIndices.reserveCapacity(requestedDetections.count)

        for requested in requestedDetections {
            guard let matchIndex = cache.detections.indices
                .filter({ !selectedIndices.contains($0) })
                .max(by: { lhs, rhs in
                    FurnitureFitIoU.calculate(cache.detections[lhs], requested) <
                        FurnitureFitIoU.calculate(cache.detections[rhs], requested)
                }) else { continue }
            let matched = cache.detections[matchIndex]
            guard matched.classIdx == requested.classIdx,
                  FurnitureFitIoU.calculate(matched, requested) >= 0.95 else { continue }
            selectedIndices.append(matchIndex)
        }

        guard !selectedIndices.isEmpty else { return nil }
        let allRawMaskPlanes = cache.candidates.map {
            buildRawMaskPlane(candidate: $0, maskFeatureMatrix: cache.maskFeatureMatrix)
        }
        let affinityGroup = makeMaskAffinityGraph(allRawMaskPlanes).transitiveGroup(seedIndices: selectedIndices)
        let expandedIndices = affinityGroup.isEmpty ? selectedIndices : affinityGroup
        let selectedPlanes: [[Float]?] = expandedIndices.map { index in
            buildRawMaskPlane(candidate: cache.candidates[index], maskFeatureMatrix: cache.maskFeatureMatrix)
        }
        let selectedBoxes = expandedIndices.map { cache.mappedBoxes[$0] }
        guard let image = buildCombinedRawMaskImage(
            rawMaskPlanes: selectedPlanes,
            boxes: selectedBoxes,
            maxMaskCount: selectedPlanes.count,
            sourceBuffer: cache.sourceBuffer,
            mapping: cache.mapping
        ) else { return nil }
        return RTMDetCachedMaskResult(image: image, affinityGroupIndices: expandedIndices)
    }

    private static func makeMaskAffinityGraph(_ planes: [[Float]?]) -> RTMDetMaskAffinityGraph {
        let bitsets = planes.map { maskBitset(for: $0) }
        var neighbors = Array(repeating: [Int](), count: planes.count)
        guard bitsets.count > 1 else { return RTMDetMaskAffinityGraph(neighbors: neighbors) }

        for leftIndex in 0..<(bitsets.count - 1) {
            let left = bitsets[leftIndex]
            guard left.onCount > 0 else { continue }
            for rightIndex in (leftIndex + 1)..<bitsets.count {
                let right = bitsets[rightIndex]
                guard right.onCount > 0 else { continue }
                let intersectionCount = bitsetIntersectionCount(left.words, right.words)
                let affinity = Float(intersectionCount) / Float(max(1, min(left.onCount, right.onCount)))
                if affinity >= rawMaskAffinityThreshold {
                    neighbors[leftIndex].append(rightIndex)
                    neighbors[rightIndex].append(leftIndex)
                }
            }
        }

        return RTMDetMaskAffinityGraph(neighbors: neighbors)
    }

    private static func maskBitset(for plane: [Float]?) -> (words: [UInt64], onCount: Int) {
        let pixelCount = rawMaskSide * rawMaskSide
        var words = [UInt64](repeating: 0, count: (pixelCount + 63) / 64)
        guard let plane, plane.count >= pixelCount else {
            return (words, 0)
        }
        var onCount = 0
        for index in 0..<pixelCount where plane[index].isFinite && plane[index] > rawMaskAffinityBitThreshold {
            words[index >> 6] |= UInt64(1) << UInt64(index & 63)
            onCount += 1
        }
        return (words, onCount)
    }

    private static func bitsetIntersectionCount(_ left: [UInt64], _ right: [UInt64]) -> Int {
        let count = min(left.count, right.count)
        var total = 0
        for index in 0..<count {
            total += Int((left[index] & right[index]).nonzeroBitCount)
        }
        return total
    }

    private static func decodeRawCandidates(
        levels: [(cls: MLMultiArray, bbox: MLMultiArray, kernel: MLMultiArray, side: Int, stride: Float)],
        classIndices: [Int],
        classBlacklist: Set<Int>,
        confidenceThreshold: Float,
        modelSide: Int,
        preNMSLimit: Int?
    ) -> [RawCandidate] {
        var candidates: [RawCandidate] = []
        candidates.reserveCapacity(preNMSLimit ?? 512)

        for level in levels {
            let clsAt = nchwReader(for: level.cls)
            let bboxAt = nchwReader(for: level.bbox)
            let kernelAt = nchwReader(for: level.kernel)
            let side = level.side
            let stride = level.stride

            for y in 0..<side {
                for x in 0..<side {
                    // sigmoid is monotonic, so argmax over raw logits == argmax over sigmoids: pick the
                    // best class by logit, then apply sigmoid once (instead of per class).
                    var bestClass = -1
                    var bestLogit = -Float.greatestFiniteMagnitude
                    for classIdx in classIndices where !classBlacklist.contains(classIdx) {
                        let logit = clsAt(0, classIdx, y, x)
                        if logit > bestLogit {
                            bestLogit = logit
                            bestClass = classIdx
                        }
                    }
                    guard bestClass >= 0 else { continue }
                    let bestScore = sigmoid(bestLogit)
                    guard bestScore >= confidenceThreshold else { continue }

                    let centerX = (Float(x) + 0.5) * stride
                    let centerY = (Float(y) + 0.5) * stride
                    let left = bboxAt(0, 0, y, x)
                    let top = bboxAt(0, 1, y, x)
                    let right = bboxAt(0, 2, y, x)
                    let bottom = bboxAt(0, 3, y, x)
                    guard left.isFinite, top.isFinite, right.isFinite, bottom.isFinite else { continue }

                    let maxSide = Float(modelSide)
                    let x1 = max(0, min(maxSide, centerX - left))
                    let y1 = max(0, min(maxSide, centerY - top))
                    let x2 = max(0, min(maxSide, centerX + right))
                    let y2 = max(0, min(maxSide, centerY + bottom))
                    guard x2 > x1 + 1, y2 > y1 + 1 else { continue }

                    var kernel = [Float](repeating: 0, count: 169)
                    for i in 0..<169 {
                        kernel[i] = kernelAt(0, i, y, x)
                    }
                    let rowIndex = candidates.count
                    let box = BoxRecord(
                        rowIndex: rowIndex,
                        x1: x1,
                        y1: y1,
                        x2: x2,
                        y2: y2,
                        score: bestScore,
                        classIdx: bestClass
                    )
                    candidates.append(RawCandidate(box: box, kernel: kernel, priorX: centerX, priorY: centerY, stride: stride))
                }
            }
        }

        let sortedCandidates = candidates.sorted { $0.box.score > $1.box.score }
        return preNMSLimit.map { Array(sortedCandidates.prefix(max(1, $0))) } ?? sortedCandidates
    }

    private static func validateRawClassHeads(
        _ levels: [(array: MLMultiArray, side: Int)]
    ) throws {
        let classCount = 80
        let saturationLogit: Float = 8
        let maximumSaturatedClassesAtOneAnchor = 7
        let maximumGloballySaturatedClasses = 19
        var classMaximums = [Float](repeating: -Float.greatestFiniteMagnitude, count: classCount)
        var nonFiniteCount = 0
        var worstAnchorSaturatedClassCount = 0

        for level in levels {
            let valueAt = nchwReader(for: level.array)
            for y in 0..<level.side {
                for x in 0..<level.side {
                    var anchorSaturatedClassCount = 0
                    for classIndex in 0..<classCount {
                        let value = valueAt(0, classIndex, y, x)
                        guard value.isFinite else {
                            nonFiniteCount += 1
                            continue
                        }
                        classMaximums[classIndex] = max(classMaximums[classIndex], value)
                        if value >= saturationLogit {
                            anchorSaturatedClassCount += 1
                        }
                    }
                    worstAnchorSaturatedClassCount = max(
                        worstAnchorSaturatedClassCount,
                        anchorSaturatedClassCount
                    )
                }
            }
        }

        let globallySaturatedClassCount = classMaximums.reduce(into: 0) { count, value in
            if value >= saturationLogit {
                count += 1
            }
        }
        guard nonFiniteCount == 0,
              worstAnchorSaturatedClassCount <= maximumSaturatedClassesAtOneAnchor,
              globallySaturatedClassCount <= maximumGloballySaturatedClasses else {
            throw RTMDetLiteRuntimeError.invalidTensor(
                "implausible class heads; refusing corrupted segmentation "
                    + "(nonFinite=\(nonFiniteCount), "
                    + "maxSaturatedClassesAtAnchor=\(worstAnchorSaturatedClassCount), "
                    + "globallySaturatedClasses=\(globallySaturatedClassCount))"
            )
        }
    }

    private static func classAwareNMS(_ candidates: [RawCandidate], iouThreshold: Float, limit: Int?) -> [RawCandidate] {
        // Standard class-aware NMS: confidence first, suppress same-class boxes whose IoU exceeds
        // the threshold. Area is only a deterministic tie-breaker, not the primary priority.
        let cappedLimit = limit.map { max(1, $0) }
        let sorted = candidates.sorted { lhs, rhs in
            if abs(lhs.box.score - rhs.box.score) > 1e-6 {
                return lhs.box.score > rhs.box.score
            }
            let lhsArea = max(0, lhs.box.x2 - lhs.box.x1) * max(0, lhs.box.y2 - lhs.box.y1)
            let rhsArea = max(0, rhs.box.x2 - rhs.box.x1) * max(0, rhs.box.y2 - rhs.box.y1)
            return lhsArea < rhsArea
        }
        var kept: [RawCandidate] = []
        kept.reserveCapacity(cappedLimit ?? candidates.count)

        for candidate in sorted {
            var suppressed = false
            for existing in kept where existing.box.classIdx == candidate.box.classIdx {
                if boxIoU(candidate.box, existing.box) > iouThreshold {
                    suppressed = true
                    break
                }
            }
            if !suppressed {
                let rowIndex = kept.count
                let box = BoxRecord(
                    rowIndex: rowIndex,
                    x1: candidate.box.x1,
                    y1: candidate.box.y1,
                    x2: candidate.box.x2,
                    y2: candidate.box.y2,
                    score: candidate.box.score,
                    classIdx: candidate.box.classIdx
                )
                kept.append(RawCandidate(box: box, kernel: candidate.kernel, priorX: candidate.priorX, priorY: candidate.priorY, stride: candidate.stride))
                if let cappedLimit, kept.count >= cappedLimit {
                    break
                }
            }
        }

        return kept
    }

    private static func boxIoU(_ a: BoxRecord, _ b: BoxRecord) -> Float {
        let ix = max(Float(0), min(a.x2, b.x2) - max(a.x1, b.x1))
        let iy = max(Float(0), min(a.y2, b.y2) - max(a.y1, b.y1))
        let intersection = ix * iy
        let areaA = max(Float(0), a.x2 - a.x1) * max(Float(0), a.y2 - a.y1)
        let areaB = max(Float(0), b.x2 - b.x1) * max(Float(0), b.y2 - b.y1)
        let union = areaA + areaB - intersection
        return union > 0 ? intersection / union : 0
    }

    private static func rawMaskFeatureMatrix(from maskFeat: MLMultiArray) -> RawMaskFeatureMatrix? {
        let side = rawMaskSide
        let pixelCount = side * side
        var features = [Float](repeating: 0, count: 8 * pixelCount)
        let dims = compactDims(maskFeat)
        let strides = maskFeat.strides.map(\.intValue)
        guard dims.count == 4,
              strides.count == 4,
              dims[0] >= 1,
              dims[1] >= 8,
              dims[2] >= side,
              dims[3] >= side else {
            return nil
        }

        let s1 = strides[1]
        let s2 = strides[2]
        let s3 = strides[3]
        let bound = storageSpan(for: maskFeat)

        switch maskFeat.dataType {
        case .float32:
            let ptr = maskFeat.dataPointer.assumingMemoryBound(to: Float.self)
            for c in 0..<8 {
                let channelOffset = c * pixelCount
                let sourceChannelOffset = c * s1
                for y in 0..<side {
                    let rowOffset = channelOffset + y * side
                    let sourceRowOffset = sourceChannelOffset + y * s2
                    for x in 0..<side {
                        let sourceIndex = sourceRowOffset + x * s3
                        features[rowOffset + x] = sourceIndex >= 0 && sourceIndex < bound ? ptr[sourceIndex] : 0
                    }
                }
            }
        case .float16:
            let ptr = maskFeat.dataPointer.assumingMemoryBound(to: UInt16.self)
            for c in 0..<8 {
                let channelOffset = c * pixelCount
                let sourceChannelOffset = c * s1
                for y in 0..<side {
                    let rowOffset = channelOffset + y * side
                    let sourceRowOffset = sourceChannelOffset + y * s2
                    for x in 0..<side {
                        let sourceIndex = sourceRowOffset + x * s3
                        features[rowOffset + x] = sourceIndex >= 0 && sourceIndex < bound
                            ? Float(Float16(bitPattern: ptr[sourceIndex]))
                            : 0
                    }
                }
            }
        default:
            let featAt = nchwReader(for: maskFeat)
            for c in 0..<8 {
                let channelOffset = c * pixelCount
                for y in 0..<side {
                    let rowOffset = channelOffset + y * side
                    for x in 0..<side {
                        features[rowOffset + x] = featAt(0, c, y, x)
                    }
                }
            }
        }
        return RawMaskFeatureMatrix(side: side, pixelCount: pixelCount, features: features)
    }

    private static func buildRawMaskPlane(candidate: RawCandidate, maskFeatureMatrix: RawMaskFeatureMatrix) -> [Float]? {
        guard candidate.kernel.count == 169 else { return nil }
        let maskSide = maskFeatureMatrix.side
        let pixelCount = maskFeatureMatrix.pixelCount
        let maskStride = Float(rawMaskModelSide) / Float(maskSide)

        let w1 = 0
        let w2 = w1 + 80
        let w3 = w2 + 64
        let b1 = w3 + 8
        let b2 = b1 + 8
        let b3 = b2 + 8
        let hiddenSize = Int32(8)
        let pixelCount32 = Int32(pixelCount)

        var w1Coord = [Float](repeating: 0, count: 8 * 2)
        var w1Feat = [Float](repeating: 0, count: 8 * 8)
        let w2Weights = Array(candidate.kernel[w2..<w3])
        let w3Weights = Array(candidate.kernel[w3..<b1])
        for outputChannel in 0..<8 {
            w1Coord[outputChannel * 2] = candidate.kernel[w1 + outputChannel * 10]
            w1Coord[outputChannel * 2 + 1] = candidate.kernel[w1 + outputChannel * 10 + 1]
            for featureChannel in 0..<8 {
                w1Feat[outputChannel * 8 + featureChannel] = candidate.kernel[w1 + outputChannel * 10 + 2 + featureChannel]
            }
        }

        var coord = [Float](repeating: 0, count: 2 * pixelCount)
        for y in 0..<maskSide {
            let rowOffset = y * maskSide
            for x in 0..<maskSide {
                let pixelOffset = rowOffset + x
                let gridX = (Float(x) + 0.5) * maskStride
                let gridY = (Float(y) + 0.5) * maskStride
                coord[pixelOffset] = (candidate.priorX - gridX) / max(1, candidate.stride * 8)
                coord[pixelCount + pixelOffset] = (candidate.priorY - gridY) / max(1, candidate.stride * 8)
            }
        }

        var h1 = [Float](repeating: 0, count: 8 * pixelCount)
        for outputChannel in 0..<8 {
            let bias = candidate.kernel[b1 + outputChannel]
            h1.replaceSubrange(
                outputChannel * pixelCount..<(outputChannel + 1) * pixelCount,
                with: repeatElement(bias, count: pixelCount)
            )
        }
        cblas_sgemm(
            CblasRowMajor,
            CblasNoTrans,
            CblasNoTrans,
            hiddenSize,
            pixelCount32,
            hiddenSize,
            1,
            w1Feat,
            hiddenSize,
            maskFeatureMatrix.features,
            pixelCount32,
            1,
            &h1,
            pixelCount32
        )
        cblas_sgemm(
            CblasRowMajor,
            CblasNoTrans,
            CblasNoTrans,
            hiddenSize,
            pixelCount32,
            2,
            1,
            w1Coord,
            2,
            coord,
            pixelCount32,
            1,
            &h1,
            pixelCount32
        )
        var zero: Float = 0
        vDSP_vthres(h1, 1, &zero, &h1, 1, vDSP_Length(h1.count))

        var h2 = [Float](repeating: 0, count: 8 * pixelCount)
        for outputChannel in 0..<8 {
            let bias = candidate.kernel[b2 + outputChannel]
            h2.replaceSubrange(
                outputChannel * pixelCount..<(outputChannel + 1) * pixelCount,
                with: repeatElement(bias, count: pixelCount)
            )
        }
        cblas_sgemm(
            CblasRowMajor,
            CblasNoTrans,
            CblasNoTrans,
            hiddenSize,
            pixelCount32,
            hiddenSize,
            1,
            w2Weights,
            hiddenSize,
            h1,
            pixelCount32,
            1,
            &h2,
            pixelCount32
        )
        vDSP_vthres(h2, 1, &zero, &h2, 1, vDSP_Length(h2.count))

        var logits = [Float](repeating: candidate.kernel[b3], count: pixelCount)
        cblas_sgemm(
            CblasRowMajor,
            CblasNoTrans,
            CblasNoTrans,
            1,
            pixelCount32,
            hiddenSize,
            1,
            w3Weights,
            hiddenSize,
            h2,
            pixelCount32,
            1,
            &logits,
            pixelCount32
        )

        var negLogits = [Float](repeating: 0, count: pixelCount)
        vDSP_vneg(logits, 1, &negLogits, 1, vDSP_Length(pixelCount))
        var expValues = [Float](repeating: 0, count: pixelCount)
        var expCount = Int32(pixelCount)
        vvexpf(&expValues, negLogits, &expCount)
        var one: Float = 1
        vDSP_vsadd(expValues, 1, &one, &expValues, 1, vDSP_Length(pixelCount))
        var out = [Float](repeating: 0, count: pixelCount)
        vDSP_svdiv(&one, expValues, 1, &out, 1, vDSP_Length(pixelCount))
        return out
    }

    private static func buildCombinedRawMaskImage(
        rawMaskPlanes: [[Float]?],
        boxes: [BoxRecord],
        maxMaskCount: Int,
        sourceBuffer: CVPixelBuffer,
        mapping: ImageMapping
    ) -> UIImage? {
        guard !rawMaskPlanes.isEmpty, !boxes.isEmpty else { return nil }
        let sourceWidth = mapping.sourceWidth
        let sourceHeight = mapping.sourceHeight
        guard sourceWidth > 0,
              sourceHeight > 0,
              CVPixelBufferGetWidth(sourceBuffer) == sourceWidth,
              CVPixelBufferGetHeight(sourceBuffer) == sourceHeight else {
            return nil
        }

        // The old path recomputed source-to-model divisions for every output pixel and selected one
        // 160x160 texel. Precompute each axis once, then sample four neighboring texels. This removes
        // the blocky nearest-neighbor edge while keeping rasterization in the existing single pass.
        let (xSamples, ySamples) = bilinearMaskSamplingAxes(
            maskWidth: rawMaskSide,
            maskHeight: rawMaskSide,
            mapping: mapping
        )

        var rgba = [UInt8](repeating: 0, count: sourceWidth * sourceHeight * 4)

        let compositingBuffer = rgbCompositingBuffer(for: sourceBuffer, width: sourceWidth, height: sourceHeight) ?? sourceBuffer
        CVPixelBufferLockBaseAddress(compositingBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(compositingBuffer, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(compositingBuffer) else { return nil }
        let rowBytes = CVPixelBufferGetBytesPerRow(compositingBuffer)
        let sourceLayout = PixelRGBLayout(pixelFormat: CVPixelBufferGetPixelFormatType(compositingBuffer))

        var paintedPixels = 0
        let count = min(max(1, maxMaskCount), boxes.count, rawMaskPlanes.count)
        for index in 0..<count {
            guard let plane = rawMaskPlanes[index], plane.count == rawMaskSide * rawMaskSide else { continue }
            let box = boxes[index]
            let boxWidth = max(0, box.x2 - box.x1)
            let boxHeight = max(0, box.y2 - box.y1)
            let padX = Int(ceil(Double(boxWidth) * 0.20))
            let padTop = Int(ceil(Double(boxHeight) * 1.00))
            let padBottom = Int(ceil(Double(boxHeight) * 0.25))
            let xMin = max(0, min(sourceWidth - 1, Int(floor(box.x1)) - padX))
            let yMin = max(0, min(sourceHeight - 1, Int(floor(box.y1)) - padTop))
            let xMax = max(0, min(sourceWidth - 1, Int(ceil(box.x2)) + padX))
            let yMax = max(0, min(sourceHeight - 1, Int(ceil(box.y2)) + padBottom))
            guard xMax >= xMin, yMax >= yMin else { continue }

            for y in yMin...yMax {
                let sourceRow = base.advanced(by: y * rowBytes).assumingMemoryBound(to: UInt8.self)
                let ySample = ySamples[y]
                let upperRowOffset = ySample.lowerIndex * rawMaskSide
                let lowerRowOffset = ySample.upperIndex * rawMaskSide
                for x in xMin...xMax {
                    let xSample = xSamples[x]
                    let upperLeft = plane[upperRowOffset + xSample.lowerIndex]
                    let upperRight = plane[upperRowOffset + xSample.upperIndex]
                    let lowerLeft = plane[lowerRowOffset + xSample.lowerIndex]
                    let lowerRight = plane[lowerRowOffset + xSample.upperIndex]
                    let upperValue = upperLeft + (upperRight - upperLeft) * xSample.upperWeight
                    let lowerValue = lowerLeft + (lowerRight - lowerLeft) * xSample.upperWeight
                    let value = upperValue + (lowerValue - upperValue) * ySample.upperWeight
                    let alpha = rawMaskRenderAlpha(for: value)
                    guard alpha > 0 else { continue }
                    let dest = (y * sourceWidth + x) * 4
                    let source = x * 4
                    let existingAlpha = rgba[dest + 3]
                    guard alpha > existingAlpha else { continue }
                    if existingAlpha == 0 { paintedPixels += 1 }
                    let rgb = rgbBytes(from: sourceRow, offset: source, layout: sourceLayout)
                    if alpha == 255 {
                        rgba[dest] = rgb.r
                        rgba[dest + 1] = rgb.g
                        rgba[dest + 2] = rgb.b
                    } else {
                        rgba[dest] = premultiplied(rgb.r, alpha: alpha)
                        rgba[dest + 1] = premultiplied(rgb.g, alpha: alpha)
                        rgba[dest + 2] = premultiplied(rgb.b, alpha: alpha)
                    }
                    rgba[dest + 3] = alpha
                }
            }
        }

        guard paintedPixels > 0 else { return nil }
        return rgbaImage(width: sourceWidth, height: sourceHeight, rgba: rgba)
    }

    private static func bilinearMaskSamplingAxes(
        maskWidth: Int,
        maskHeight: Int,
        mapping: ImageMapping
    ) -> (x: [BilinearMaskAxisSample], y: [BilinearMaskAxisSample]) {
        let modelSide = Float(mapping.modelSide)
        let sourceToModelX: Float
        let sourceToModelY: Float
        let modelPadX: Float
        let modelPadY: Float

        if mapping.usesLetterbox {
            let gain = min(
                modelSide / Float(mapping.sourceWidth),
                modelSide / Float(mapping.sourceHeight)
            )
            sourceToModelX = gain
            sourceToModelY = gain
            modelPadX = (modelSide - Float(mapping.sourceWidth) * gain) * 0.5
            modelPadY = (modelSide - Float(mapping.sourceHeight) * gain) * 0.5
        } else {
            sourceToModelX = modelSide / Float(mapping.sourceWidth)
            sourceToModelY = modelSide / Float(mapping.sourceHeight)
            modelPadX = 0
            modelPadY = 0
        }

        return (
            bilinearMaskAxisSamples(
                sourceCount: mapping.sourceWidth,
                maskCount: maskWidth,
                modelSide: modelSide,
                sourceToModelScale: sourceToModelX,
                modelPadding: modelPadX
            ),
            bilinearMaskAxisSamples(
                sourceCount: mapping.sourceHeight,
                maskCount: maskHeight,
                modelSide: modelSide,
                sourceToModelScale: sourceToModelY,
                modelPadding: modelPadY
            )
        )
    }

    private static func bilinearMaskAxisSamples(
        sourceCount: Int,
        maskCount: Int,
        modelSide: Float,
        sourceToModelScale: Float,
        modelPadding: Float
    ) -> [BilinearMaskAxisSample] {
        guard sourceCount > 0, maskCount > 0, modelSide > 0 else { return [] }
        let modelToMaskScale = Float(maskCount) / modelSide
        let step = sourceToModelScale * modelToMaskScale
        let firstCoordinate = (0.5 * sourceToModelScale + modelPadding) * modelToMaskScale - 0.5
        let lastMaskIndex = maskCount - 1

        return (0..<sourceCount).map { sourceIndex in
            let coordinate = firstCoordinate + Float(sourceIndex) * step
            let clamped = min(Float(lastMaskIndex), max(0, coordinate))
            let lowerIndex = Int(clamped.rounded(.down))
            let upperIndex = min(lastMaskIndex, lowerIndex + 1)
            return BilinearMaskAxisSample(
                lowerIndex: lowerIndex,
                upperIndex: upperIndex,
                upperWeight: clamped - Float(lowerIndex)
            )
        }
    }

    /// Writes the exact Android RTMDet contract directly into LiteRT's input allocation:
    /// NHWC float32, BGR channel order, raw 0...255 values. Normalization is part of the graph.
    private static func writeRawBGRNHWC(
        from pixelBuffer: CVPixelBuffer,
        to output: UnsafeMutableBufferPointer<Float>,
        collectStats: Bool
    ) throws {
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        guard width == 640, height == 640 else {
            throw NSError(
                domain: "RTMDetImageInference",
                code: 5,
                userInfo: [NSLocalizedDescriptionKey: "Invalid RTMDet input size \(width)x\(height); expected 640x640"],
            )
        }
        guard CVPixelBufferGetPixelFormatType(pixelBuffer) == kCVPixelFormatType_32BGRA else {
            throw NSError(
                domain: "RTMDetImageInference",
                code: 6,
                userInfo: [NSLocalizedDescriptionKey: "RTMDet preprocessing expected a BGRA pixel buffer"],
            )
        }
        let expectedCount = width * height * 3
        guard output.count == expectedCount else {
            throw RTMDetLiteRuntimeError.invalidTensor(
                "input element count \(output.count), expected \(expectedCount)"
            )
        }

        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

        guard let base = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            throw NSError(
                domain: "RTMDetImageInference",
                code: 7,
                userInfo: [NSLocalizedDescriptionKey: "Prepared pixel buffer has no base address"],
            )
        }

        let rowBytes = CVPixelBufferGetBytesPerRow(pixelBuffer)
        var minValue = Float.greatestFiniteMagnitude
        var maxValue = -Float.greatestFiniteMagnitude
        var sumValue: Double = 0

        for y in 0..<height {
            let row = base.advanced(by: y * rowBytes).assumingMemoryBound(to: UInt8.self)
            for x in 0..<width {
                let source = x * 4
                let destination = (y * width + x) * 3
                let b = Float(row[source])
                let g = Float(row[source + 1])
                let r = Float(row[source + 2])
                output[destination] = b
                output[destination + 1] = g
                output[destination + 2] = r
                if collectStats {
                    minValue = min(minValue, min(b, min(g, r)))
                    maxValue = max(maxValue, max(b, max(g, r)))
                    sumValue += Double(b + g + r)
                }
            }
        }

        if collectStats {
            let meanValue = sumValue / Double(expectedCount)
            lastInputTensorStats = "input=raw_bgr_nhwc[min=\(String(format: "%.1f", minValue)) max=\(String(format: "%.1f", maxValue)) mean=\(String(format: "%.1f", meanValue))]"
        }
    }

    private static func pickBoxesArray(from arrays: [(name: String, array: MLMultiArray)]) -> MLMultiArray? {
        let preferred = arrays.filter { entry in
            let dims = compactDims(entry.array)
            guard let last = dims.last else { return false }
            if entry.name.localizedCaseInsensitiveContains("bbox") || entry.name.localizedCaseInsensitiveContains("det") {
                return last == 5 || last == 6
            }
            return false
        }
        if let match = preferred.first?.array {
            return match
        }

        return arrays.first(where: { entry in
            let dims = compactDims(entry.array)
            guard let last = dims.last else { return false }
            return (dims.count == 2 || dims.count == 3) && (last == 5 || last == 6)
        })?.array
    }

    private static func pickLabelsArray(
        from arrays: [(name: String, array: MLMultiArray)],
        targetCountHint: Int
    ) -> MLMultiArray? {
        let preferred = arrays.filter { entry in
            let dims = compactDims(entry.array)
            let total = dims.reduce(1, *)
            guard total > 0 else { return false }
            if !(entry.name.localizedCaseInsensitiveContains("label") || entry.name.localizedCaseInsensitiveContains("class")) {
                return false
            }
            return dims.count <= 2 && (targetCountHint == 0 || total == targetCountHint)
        }
        if let match = preferred.first?.array {
            return match
        }

        return arrays.first(where: { entry in
            let dims = compactDims(entry.array)
            let total = dims.reduce(1, *)
            return dims.count <= 2 && total == targetCountHint
        })?.array
    }

    private static func pickMaskArray(
        from arrays: [(name: String, array: MLMultiArray)],
        targetCountHint: Int
    ) -> MLMultiArray? {
        let preferred = arrays.filter { entry in
            let dims = compactDims(entry.array)
            guard dims.count >= 3 else { return false }
            guard let maskHeight = dims.dropLast().last,
                  let maskWidth = dims.last else { return false }
            guard maskHeight > 1, maskWidth > 1 else { return false }
            if !(entry.name.localizedCaseInsensitiveContains("mask") || entry.name.localizedCaseInsensitiveContains("seg")) {
                return false
            }
            let count = maskCount(for: entry.array)
            return targetCountHint == 0 || count == targetCountHint
        }
        if let match = preferred.first?.array {
            return match
        }

        return arrays.first(where: { entry in
            let dims = compactDims(entry.array)
            guard dims.count >= 3 else { return false }
            guard let maskHeight = dims.dropLast().last,
                  let maskWidth = dims.last else { return false }
            guard maskHeight > 1, maskWidth > 1 else { return false }
            return true
        })?.array
    }

    private static func parseBoxes(
        from array: MLMultiArray,
        labelsArray: MLMultiArray?,
        confidenceThreshold: Float
    ) -> [BoxRecord] {
        let dims = compactDims(array)
        guard dims.count == 2 || dims.count == 3 else { return [] }
        let lastDim = dims.last ?? 0
        guard lastDim == 5 || lastDim == 6 else { return [] }

        let flat = toFloatArray(array)
        guard !flat.isEmpty else { return [] }

        let rows: Int
        let rowWidth = lastDim
        if dims.count == 2 {
            rows = dims[0]
        } else {
            rows = dims[0] == 1 ? dims[1] : dims[0]
        }

        let labels = labelsArray.map(toFloatArray)
        var parsed: [BoxRecord] = []
        parsed.reserveCapacity(rows)

        for row in 0..<rows {
            let base = row * rowWidth
            guard base + rowWidth <= flat.count else { break }

            let x1 = flat[base]
            let y1 = flat[base + 1]
            let x2 = flat[base + 2]
            let y2 = flat[base + 3]
            let score = flat[base + 4]
            let classIdx: Int?
            if rowWidth >= 6 {
                classIdx = Int(flat[base + 5].rounded())
            } else if let labels, row < labels.count {
                classIdx = Int(labels[row].rounded())
            } else {
                classIdx = nil
            }

            guard score.isFinite, score >= confidenceThreshold else { continue }
            guard x1.isFinite, y1.isFinite, x2.isFinite, y2.isFinite else { continue }
            guard x2 > x1, y2 > y1 else { continue }

            parsed.append(BoxRecord(rowIndex: row, x1: x1, y1: y1, x2: x2, y2: y2, score: score, classIdx: classIdx))
        }

        return parsed
    }

    private static func mapBoxToSource(box: BoxRecord, mapping: ImageMapping) -> BoxRecord {
        let coordinateSpace = inferCoordinateSpace(for: box, mapping: mapping)

        switch coordinateSpace {
        case .sourceImage:
            return clipBoxToSource(box, sourceWidth: mapping.sourceWidth, sourceHeight: mapping.sourceHeight)

        case .normalized:
            let sourceWidth = Float(mapping.sourceWidth)
            let sourceHeight = Float(mapping.sourceHeight)
            return BoxRecord(
                rowIndex: box.rowIndex,
                x1: max(0, min(sourceWidth, box.x1 * sourceWidth)),
                y1: max(0, min(sourceHeight, box.y1 * sourceHeight)),
                x2: max(0, min(sourceWidth, box.x2 * sourceWidth)),
                y2: max(0, min(sourceHeight, box.y2 * sourceHeight)),
                score: box.score,
                classIdx: box.classIdx
            )

        case .modelSquare:
            let mapped = mapModelRectToSource(
                x1: box.x1,
                y1: box.y1,
                x2: box.x2,
                y2: box.y2,
                mapping: mapping
            )
            return BoxRecord(
                rowIndex: box.rowIndex,
                x1: mapped.x1,
                y1: mapped.y1,
                x2: mapped.x2,
                y2: mapped.y2,
                score: box.score,
                classIdx: box.classIdx
            )
        }
    }

    private static func clipBoxToSource(_ box: BoxRecord, sourceWidth: Int, sourceHeight: Int) -> BoxRecord {
        let maxX = Float(sourceWidth)
        let maxY = Float(sourceHeight)
        return BoxRecord(
            rowIndex: box.rowIndex,
            x1: max(0, min(maxX, box.x1)),
            y1: max(0, min(maxY, box.y1)),
            x2: max(0, min(maxX, box.x2)),
            y2: max(0, min(maxY, box.y2)),
            score: box.score,
            classIdx: box.classIdx
        )
    }

    private static func inferCoordinateSpace(for box: BoxRecord, mapping: ImageMapping) -> CoordinateSpace {
        let maxCoord = max(box.x1, box.y1, box.x2, box.y2)
        if maxCoord <= 1.25 {
            return .normalized
        }
        if maxCoord <= Float(mapping.modelSide) * 1.1 {
            return .modelSquare
        }
        return .sourceImage
    }

    private static func mapModelRectToSource(
        x1: Float,
        y1: Float,
        x2: Float,
        y2: Float,
        mapping: ImageMapping
    ) -> (x1: Float, y1: Float, x2: Float, y2: Float) {
        let sourceWidth = Float(mapping.sourceWidth)
        let sourceHeight = Float(mapping.sourceHeight)

        if mapping.usesLetterbox {
            let gain = min(Float(mapping.modelSide) / sourceWidth, Float(mapping.modelSide) / sourceHeight)
            let padX = (Float(mapping.modelSide) - sourceWidth * gain) * 0.5
            let padY = (Float(mapping.modelSide) - sourceHeight * gain) * 0.5
            return (
                max(0, min(sourceWidth, (x1 - padX) / gain)),
                max(0, min(sourceHeight, (y1 - padY) / gain)),
                max(0, min(sourceWidth, (x2 - padX) / gain)),
                max(0, min(sourceHeight, (y2 - padY) / gain))
            )
        }

        let sx = sourceWidth / Float(mapping.modelSide)
        let sy = sourceHeight / Float(mapping.modelSide)
        return (
            max(0, min(sourceWidth, x1 * sx)),
            max(0, min(sourceHeight, y1 * sy)),
            max(0, min(sourceWidth, x2 * sx)),
            max(0, min(sourceHeight, y2 * sy))
        )
    }

    private static func buildCombinedMaskImage(
        from array: MLMultiArray?,
        boxes: [BoxRecord],
        maxMaskCount: Int,
        mapping: ImageMapping
    ) -> UIImage? {
        guard let array else { return nil }
        let dims = compactDims(array)
        guard dims.count >= 3 else { return nil }

        let availableMaskCount = maskCount(for: array)
        guard availableMaskCount > 0 else { return nil }
        let selectedBoxes = boxes.prefix(max(1, maxMaskCount)).filter { box in
            box.rowIndex >= 0 && box.rowIndex < availableMaskCount
        }
        guard !selectedBoxes.isEmpty else { return nil }

        guard let maskHeight = dims.dropLast().last,
              let maskWidth = dims.last else { return nil }
        guard maskHeight > 0, maskWidth > 0 else { return nil }

        let sourceWidth = mapping.sourceWidth
        let sourceHeight = mapping.sourceHeight
        var rgba = [UInt8](repeating: 0, count: sourceWidth * sourceHeight * 4)

        let valueAt = maskValueReader(for: array, maskWidth: maskWidth, maskHeight: maskHeight)
        let colors: [(UInt8, UInt8, UInt8)] = [
            (0, 200, 255),
            (0, 255, 120),
            (255, 180, 0),
            (255, 80, 140),
            (180, 120, 255),
            (255, 255, 0),
        ]

        var paintedPixels = 0
        for (drawIndex, box) in selectedBoxes.enumerated() {
            let maskIndex = box.rowIndex
            guard let maskInfo = maskPlaneInfo(
                maskIndex: maskIndex,
                maskWidth: maskWidth,
                maskHeight: maskHeight,
                valueAt: valueAt
            ) else { continue }
            let xMin = max(0, min(sourceWidth - 1, Int(floor(box.x1))))
            let yMin = max(0, min(sourceHeight - 1, Int(floor(box.y1))))
            let xMax = max(0, min(sourceWidth - 1, Int(ceil(box.x2))))
            let yMax = max(0, min(sourceHeight - 1, Int(ceil(box.y2))))
            guard xMax >= xMin, yMax >= yMin else { continue }

            let (r, g, b) = colors[drawIndex % colors.count]
            for y in yMin...yMax {
                for x in xMin...xMax {
                    let (sampleX, sampleY) = maskSampleCoordinate(
                        sourceX: x,
                        sourceY: y,
                        maskWidth: maskWidth,
                        maskHeight: maskHeight,
                        mapping: mapping
                    )
                    let maskValue = valueAt(maskIndex, sampleY, sampleX)
                    guard maskValue.isFinite, maskValue > maskInfo.threshold else { continue }
                    let confidence = normalizedMaskConfidence(value: maskValue, info: maskInfo)
                    guard confidence > 0.05 else { continue }
                    let dest = (y * sourceWidth + x) * 4
                    if rgba[dest + 3] == 0 {
                        paintedPixels += 1
                    }
                    let alpha = UInt8(max(28, min(150, Int((confidence * 150).rounded()))))
                    guard alpha >= rgba[dest + 3] else { continue }
                    let premultipliedAlpha = Float(alpha) / 255.0
                    rgba[dest] = UInt8((Float(r) * premultipliedAlpha).rounded())
                    rgba[dest + 1] = UInt8((Float(g) * premultipliedAlpha).rounded())
                    rgba[dest + 2] = UInt8((Float(b) * premultipliedAlpha).rounded())
                    rgba[dest + 3] = alpha
                }
            }
        }

        guard paintedPixels > 0 else { return nil }
        return rgbaImage(width: sourceWidth, height: sourceHeight, rgba: rgba)
    }

    private static func maskPlaneInfo(
        maskIndex: Int,
        maskWidth: Int,
        maskHeight: Int,
        valueAt: (Int, Int, Int) -> Float
    ) -> MaskPlaneInfo? {
        let threshold = localMaskForegroundThreshold(
            maskIndex: maskIndex,
            maskWidth: maskWidth,
            maskHeight: maskHeight,
            valueAt: valueAt
        )
        var maxValue = -Float.greatestFiniteMagnitude
        var minX = maskWidth
        var minY = maskHeight
        var maxX = -1
        var maxY = -1

        for y in 0..<maskHeight {
            for x in 0..<maskWidth {
                let value = valueAt(maskIndex, y, x)
                if value.isFinite {
                    maxValue = max(maxValue, value)
                }
                guard value.isFinite, value > threshold else { continue }
                minX = min(minX, x)
                minY = min(minY, y)
                maxX = max(maxX, x)
                maxY = max(maxY, y)
            }
        }

        guard maxX >= minX, maxY >= minY else { return nil }
        return MaskPlaneInfo(
            threshold: threshold,
            maxValue: maxValue,
            minX: minX,
            minY: minY,
            maxX: maxX,
            maxY: maxY
        )
    }

    private static func localMaskForegroundThreshold(
        maskIndex: Int,
        maskWidth: Int,
        maskHeight: Int,
        valueAt: (Int, Int, Int) -> Float
    ) -> Float {
        inferredMaskThreshold(
            maskIndex: maskIndex,
            maskWidth: maskWidth,
            maskHeight: maskHeight,
            valueAt: valueAt
        )
    }

    private static func normalizedMaskConfidence(value: Float, info: MaskPlaneInfo) -> Float {
        let denominator = info.maxValue - info.threshold
        guard denominator.isFinite, denominator > 1e-6 else { return 1 }
        return max(0, min(1, (value - info.threshold) / denominator))
    }

    private static func maskValueReader(
        for array: MLMultiArray,
        maskWidth: Int,
        maskHeight: Int
    ) -> (Int, Int, Int) -> Float {
        let valueAtOffset = floatReader(for: array)
        let dims = compactDims(array)
        let strides = array.strides.map(\.intValue)

        if dims.count == 4, strides.count == 4 {
            if dims[0] == 1 {
                return { maskIndex, y, x in
                    valueAtOffset(maskIndex * strides[1] + y * strides[2] + x * strides[3])
                }
            }
            if dims[1] == 1 {
                return { maskIndex, y, x in
                    valueAtOffset(maskIndex * strides[0] + y * strides[2] + x * strides[3])
                }
            }
        }

        if dims.count == 3, strides.count == 3 {
            return { maskIndex, y, x in
                valueAtOffset(maskIndex * strides[0] + y * strides[1] + x * strides[2])
            }
        }

        let pixelsPerMask = maskWidth * maskHeight
        return { maskIndex, y, x in
            valueAtOffset(maskIndex * pixelsPerMask + y * maskWidth + x)
        }
    }

    private static func floatReader(for array: MLMultiArray) -> (Int) -> Float {
        let bound = storageSpan(for: array)
        switch array.dataType {
        case .float32:
            let ptr = array.dataPointer.assumingMemoryBound(to: Float.self)
            return { index in
                guard index >= 0, index < bound else { return 0 }
                return ptr[index]
            }
        case .float16:
            let ptr = array.dataPointer.assumingMemoryBound(to: UInt16.self)
            return { index in
                guard index >= 0, index < bound else { return 0 }
                return Float(Float16(bitPattern: ptr[index]))
            }
        case .double:
            let ptr = array.dataPointer.assumingMemoryBound(to: Double.self)
            return { index in
                guard index >= 0, index < bound else { return 0 }
                return Float(ptr[index])
            }
        case .int32:
            let ptr = array.dataPointer.assumingMemoryBound(to: Int32.self)
            return { index in
                guard index >= 0, index < bound else { return 0 }
                return Float(ptr[index])
            }
        default:
            return { _ in 0 }
        }
    }

    private static func storageSpan(for array: MLMultiArray) -> Int {
        let shape = array.shape.map(\.intValue)
        let strides = array.strides.map(\.intValue)
        guard shape.count == strides.count, !shape.isEmpty else { return array.count }

        var span = 1
        for index in shape.indices {
            span += max(0, shape[index] - 1) * strides[index]
        }
        return max(array.count, span)
    }

    private static func nchwReader(for array: MLMultiArray) -> (Int, Int, Int, Int) -> Float {
        let valueAtOffset = floatReader(for: array)
        let strides = array.strides.map(\.intValue)
        guard strides.count == 4 else {
            return { _, _, _, _ in 0 }
        }
        // Capture strides as scalars so the hot per-element closure avoids an array subscript +
        // bounds check on every read.
        let s0 = strides[0], s1 = strides[1], s2 = strides[2], s3 = strides[3]
        return { n, c, y, x in
            valueAtOffset(n * s0 + c * s1 + y * s2 + x * s3)
        }
    }

    private static func sigmoid(_ value: Float) -> Float {
        if value >= 0 {
            let z = exp(-value)
            return 1 / (1 + z)
        }
        let z = exp(value)
        return z / (1 + z)
    }

    private static func maskSampleCoordinate(
        sourceX: Int,
        sourceY: Int,
        maskWidth: Int,
        maskHeight: Int,
        mapping: ImageMapping
    ) -> (Int, Int) {
        let modelX: Float
        let modelY: Float

        if mapping.usesLetterbox {
            let gain = min(Float(mapping.modelSide) / Float(mapping.sourceWidth), Float(mapping.modelSide) / Float(mapping.sourceHeight))
            let padX = (Float(mapping.modelSide) - Float(mapping.sourceWidth) * gain) * 0.5
            let padY = (Float(mapping.modelSide) - Float(mapping.sourceHeight) * gain) * 0.5
            modelX = (Float(sourceX) + 0.5) * gain + padX
            modelY = (Float(sourceY) + 0.5) * gain + padY
        } else {
            modelX = (Float(sourceX) + 0.5) * Float(mapping.modelSide) / Float(mapping.sourceWidth)
            modelY = (Float(sourceY) + 0.5) * Float(mapping.modelSide) / Float(mapping.sourceHeight)
        }

        let sx = min(maskWidth - 1, max(0, Int(modelX * Float(maskWidth) / Float(mapping.modelSide))))
        let sy = min(maskHeight - 1, max(0, Int(modelY * Float(maskHeight) / Float(mapping.modelSide))))
        return (sx, sy)
    }

    private static func detectionCount(for array: MLMultiArray) -> Int {
        let dims = compactDims(array)
        guard !dims.isEmpty else { return 0 }
        if dims.count == 2 {
            return dims[0]
        }
        if dims.count == 3 {
            return dims[0] == 1 ? dims[1] : dims[0]
        }
        return 0
    }

    private static func maskCount(for array: MLMultiArray) -> Int {
        let dims = compactDims(array)
        guard dims.count >= 3 else { return 0 }
        if dims.count == 3 {
            return dims[0]
        }
        if dims.count == 4 {
            if dims[0] == 1 {
                return dims[1]
            }
            if dims[1] == 1 {
                return dims[0]
            }
        }
        return dims[dims.count - 3]
    }

    private static func compactDims(_ array: MLMultiArray) -> [Int] {
        array.shape.map(\.intValue)
    }

    private static func toFloatArray(_ array: MLMultiArray) -> [Float] {
        let count = array.count
        guard count > 0 else { return [] }

        switch array.dataType {
        case .float32:
            let ptr = array.dataPointer.bindMemory(to: Float.self, capacity: count)
            return Array(UnsafeBufferPointer(start: ptr, count: count))
        case .float16:
            let ptr = array.dataPointer.bindMemory(to: UInt16.self, capacity: count)
            return (0..<count).map { Float(Float16(bitPattern: ptr[$0])) }
        case .double:
            let ptr = array.dataPointer.bindMemory(to: Double.self, capacity: count)
            return (0..<count).map { Float(ptr[$0]) }
        case .int32:
            let ptr = array.dataPointer.bindMemory(to: Int32.self, capacity: count)
            return (0..<count).map { Float(ptr[$0]) }
        default:
            return []
        }
    }

    private static func inferredMaskThreshold(
        maskIndex: Int,
        maskWidth: Int,
        maskHeight: Int,
        valueAt: (Int, Int, Int) -> Float
    ) -> Float {
        var minValue = Float.greatestFiniteMagnitude
        var maxValue = -Float.greatestFiniteMagnitude
        var sampledValues: [Float] = []
        let stepX = max(1, maskWidth / 128)
        let stepY = max(1, maskHeight / 128)
        sampledValues.reserveCapacity((maskWidth / stepX + 1) * (maskHeight / stepY + 1))

        for y in stride(from: 0, to: maskHeight, by: stepY) {
            for x in stride(from: 0, to: maskWidth, by: stepX) {
                let value = valueAt(maskIndex, y, x)
                guard value.isFinite else { continue }
                sampledValues.append(value)
                minValue = min(minValue, value)
                maxValue = max(maxValue, value)
            }
        }

        guard minValue.isFinite, maxValue.isFinite, maxValue > minValue else {
            return 0.5
        }

        let foregroundFloor = minValue + (maxValue - minValue) * 0.05
        let foregroundSamples = sampledValues
            .filter { $0 > foregroundFloor }
            .sorted()
        let percentileThreshold = percentile(foregroundSamples, fraction: 0.72)

        if minValue >= 0, maxValue <= 1 {
            if maxValue >= 0.5 {
                return max(0.5, percentileThreshold)
            }
            return percentileThreshold
        }
        return max(0, percentileThreshold)
    }

    private static func percentile(_ sortedValues: [Float], fraction: Float) -> Float {
        guard !sortedValues.isEmpty else { return 0.5 }
        let clampedFraction = max(0, min(1, fraction))
        let index = Int((Float(sortedValues.count - 1) * clampedFraction).rounded(.down))
        return sortedValues[index]
    }

    private static func rgbaImage(width: Int, height: Int, rgba: [UInt8]) -> UIImage? {
        guard rgba.count == width * height * 4 else { return nil }
        let data = Data(rgba)
        guard let provider = CGDataProvider(data: data as CFData) else { return nil }
        let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
        guard let cgImage = CGImage(
            width: width,
            height: height,
            bitsPerComponent: 8,
            bitsPerPixel: 32,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: CGBitmapInfo.byteOrder32Big.union(.init(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)),
            provider: provider,
            decode: nil,
            shouldInterpolate: true,
            intent: .defaultIntent
        ) else {
            return nil
        }
        return UIImage(cgImage: cgImage)
    }

    // Shared with the RTMDet image path, duplicated here so the RTMDet spike stays isolated.
    private static func resizeStretchToSquare(src: CVPixelBuffer, size: Int) -> CVPixelBuffer? {
        guard let dst = squareInputBuffer(size: size) else { return nil }
        sharedCIContext.render(CIImage(cvPixelBuffer: src).transformed(by: CGAffineTransform(scaleX: CGFloat(size) / CGFloat(CVPixelBufferGetWidth(src)), y: CGFloat(size) / CGFloat(CVPixelBufferGetHeight(src)))), to: dst)
        return dst
    }

    private static func resizeLetterboxToSquare(src: CVPixelBuffer, size: Int) -> CVPixelBuffer? {
        let srcW = CVPixelBufferGetWidth(src)
        let srcH = CVPixelBufferGetHeight(src)
        guard srcW > 0, srcH > 0 else { return nil }

        guard let dst = squareInputBuffer(size: size) else { return nil }

        CVPixelBufferLockBaseAddress(dst, [])
        if let base = CVPixelBufferGetBaseAddress(dst) {
            ImageLetterboxFill.fillOpaqueBGRA114(dstBase: base, totalByteCount: CVPixelBufferGetBytesPerRow(dst) * size)
        }
        CVPixelBufferUnlockBaseAddress(dst, [])

        let gain = min(CGFloat(size) / CGFloat(srcW), CGFloat(size) / CGFloat(srcH))
        let scaledW = CGFloat(srcW) * gain
        let scaledH = CGFloat(srcH) * gain
        let tx = (CGFloat(size) - scaledW) * 0.5
        let ty = (CGFloat(size) - scaledH) * 0.5
        let transform = CGAffineTransform(a: gain, b: 0, c: 0, d: gain, tx: tx, ty: ty)
        sharedCIContext.render(CIImage(cvPixelBuffer: src).transformed(by: transform), to: dst)
        return dst
    }
}

private extension MLMultiArrayDataType {
    var debugName: String {
        switch self {
        case .double:
            return "double"
        case .float16:
            return "float16"
        case .float32:
            return "float32"
        case .int32:
            return "int32"
        case .int8:
            return "int8"
        @unknown default:
            return "unknown"
        }
    }
}
