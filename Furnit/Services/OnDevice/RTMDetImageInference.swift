import CoreGraphics
import CoreImage
import CoreML
import CoreVideo
import Foundation
import UIKit

struct RTMDetInferenceResult {
    let detections: [FurnitureFitDetection]
    let overlayMaskImage: UIImage?
    let instanceMaskImages: [UIImage?]
    let outputSummary: [String]
}

enum RTMDetImageInference {

    private struct BoxRecord {
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

    private struct ImageMapping {
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

    private struct RawCandidate {
        let box: BoxRecord
        let kernel: [Float]
        let priorX: Float
        let priorY: Float
        let stride: Float
    }

    static func runInstanceSegmentation(
        image: UIImage,
        model: MLModel,
        confidenceThreshold: Float = 0.25,
        classBlacklist: Set<Int> = [],
        allowedClassIndices: Set<Int>? = nil,
        maxMaskCount: Int = 6,
        maxDetectionCount: Int = 12,
        buildInstanceMasks: Bool = false
    ) throws -> RTMDetInferenceResult {
        guard let sourceBuffer = YoloEImageInference.pixelBufferFromImage(image) else {
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
            buildInstanceMasks: buildInstanceMasks
        )
    }

    static func runInstanceSegmentation(
        pixelBuffer sourceBuffer: CVPixelBuffer,
        model: MLModel,
        confidenceThreshold: Float = 0.25,
        classBlacklist: Set<Int> = [],
        allowedClassIndices: Set<Int>? = nil,
        maxMaskCount: Int = 6,
        maxDetectionCount: Int = 12,
        buildInstanceMasks: Bool = false
    ) throws -> RTMDetInferenceResult {
        let sourceWidth = CVPixelBufferGetWidth(sourceBuffer)
        let sourceHeight = CVPixelBufferGetHeight(sourceBuffer)
        let modelSide = YoloEImageInference.modelInputSize(for: model)
        let usesLetterbox = modelSide >= 1280

        guard let preparedBuffer = usesLetterbox
            ? resizeLetterboxToSquare(src: sourceBuffer, size: modelSide)
            : resizeStretchToSquare(src: sourceBuffer, size: modelSide)
        else {
            throw NSError(
                domain: "RTMDetImageInference",
                code: 2,
                userInfo: [NSLocalizedDescriptionKey: "Image preprocessing failed"],
            )
        }

        let inputProvider = try inputProvider(for: preparedBuffer, model: model)
        let output = try model.prediction(from: inputProvider)

        let outputArrays = collectMultiArrays(from: output)
        let outputSummary = outputArrays.map { entry in
            let shape = entry.array.shape.map(\.intValue).map(String.init).joined(separator: "x")
            return "\(entry.name): \(shape) \(entry.array.dataType.debugName)"
        }

        if isRawRTMDetOutput(outputArrays) {
            return try runRawHeadPostprocess(
                outputArrays: outputArrays,
                outputSummary: outputSummary,
                sourceWidth: sourceWidth,
                sourceHeight: sourceHeight,
                modelSide: modelSide,
                usesLetterbox: usesLetterbox,
                confidenceThreshold: confidenceThreshold,
                classBlacklist: classBlacklist,
                allowedClassIndices: allowedClassIndices,
                maxMaskCount: maxMaskCount,
                maxDetectionCount: maxDetectionCount,
                buildInstanceMasks: buildInstanceMasks
            )
        }

        guard let boxesArray = pickBoxesArray(from: outputArrays) else {
            throw NSError(
                domain: "RTMDetImageInference",
                code: 3,
                userInfo: [NSLocalizedDescriptionKey: "Could not identify a detection boxes output. Outputs: \(outputSummary.joined(separator: ", "))"],
            )
        }

        let labelsArray = pickLabelsArray(from: outputArrays, targetCountHint: detectionCount(for: boxesArray))
        let maskArray = pickMaskArray(from: outputArrays, targetCountHint: detectionCount(for: boxesArray))

        let boxes = Array(parseBoxes(from: boxesArray, labelsArray: labelsArray, confidenceThreshold: confidenceThreshold)
            .filter { box in
                guard let classIdx = box.classIdx else { return true }
                if let allowedClassIndices, !allowedClassIndices.contains(classIdx) {
                    return false
                }
                return !classBlacklist.contains(classIdx)
            }
            .sorted { lhs, rhs in
                if lhs.score == rhs.score {
                    let lhsArea = (lhs.x2 - lhs.x1) * (lhs.y2 - lhs.y1)
                    let rhsArea = (rhs.x2 - rhs.x1) * (rhs.y2 - rhs.y1)
                    return lhsArea > rhsArea
                }
                return lhs.score > rhs.score
            }
            .prefix(max(1, maxDetectionCount)))
        let mapping = ImageMapping(
            modelSide: modelSide,
            sourceWidth: sourceWidth,
            sourceHeight: sourceHeight,
            usesLetterbox: usesLetterbox
        )

        let mappedBoxes = boxes.map { mapBoxToSource(box: $0, mapping: mapping) }
        let mappedDetections = mappedBoxes.map { mapped in
            return FurnitureFitDetection(
                x: (mapped.x1 + mapped.x2) * 0.5,
                y: (mapped.y1 + mapped.y2) * 0.5,
                w: max(1, mapped.x2 - mapped.x1),
                h: max(1, mapped.y2 - mapped.y1),
                confidence: mapped.score,
                classIdx: mapped.classIdx ?? -1
            )
        }

        let combinedMask = buildCombinedMaskImage(
            from: maskArray,
            boxes: mappedBoxes,
            maxMaskCount: maxMaskCount,
            mapping: mapping
        )
        let instanceMaskImages: [UIImage?]
        if buildInstanceMasks {
            instanceMaskImages = mappedBoxes.map { mappedBox in
                buildCombinedMaskImage(
                    from: maskArray,
                    boxes: [mappedBox],
                    maxMaskCount: 1,
                    mapping: mapping
                )
            }
        } else {
            instanceMaskImages = []
        }

        return RTMDetInferenceResult(
            detections: mappedDetections,
            overlayMaskImage: combinedMask,
            instanceMaskImages: instanceMaskImages,
            outputSummary: outputSummary
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
        sourceWidth: Int,
        sourceHeight: Int,
        modelSide: Int,
        usesLetterbox: Bool,
        confidenceThreshold: Float,
        classBlacklist: Set<Int>,
        allowedClassIndices: Set<Int>?,
        maxMaskCount: Int,
        maxDetectionCount: Int,
        buildInstanceMasks: Bool
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

        let classIndices: [Int]
        if let allowedClassIndices, !allowedClassIndices.isEmpty {
            classIndices = allowedClassIndices.sorted()
        } else {
            classIndices = Array(0..<80)
        }

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
            preNMSLimit: 500
        )
        let selected = classAwareNMS(rawCandidates, iouThreshold: 0.5, limit: max(1, maxDetectionCount))
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

        let rawMaskPlanes: [[Float]?]
        if buildInstanceMasks || maxMaskCount > 1 {
            rawMaskPlanes = selected.map { buildRawMaskPlane(candidate: $0, maskFeat: maskFeat) }
        } else {
            rawMaskPlanes = []
        }

        let combinedMask = buildCombinedRawMaskImage(
            rawMaskPlanes: rawMaskPlanes,
            boxes: mappedBoxes,
            maxMaskCount: maxMaskCount,
            mapping: mapping
        )
        let instanceMaskImages: [UIImage?]
        if buildInstanceMasks {
            instanceMaskImages = mappedBoxes.enumerated().map { index, mappedBox in
                guard index < rawMaskPlanes.count else { return nil }
                return buildCombinedRawMaskImage(
                    rawMaskPlanes: [rawMaskPlanes[index]],
                    boxes: [mappedBox],
                    maxMaskCount: 1,
                    mapping: mapping
                )
            }
        } else {
            instanceMaskImages = []
        }

        return RTMDetInferenceResult(
            detections: mappedDetections,
            overlayMaskImage: combinedMask,
            instanceMaskImages: instanceMaskImages,
            outputSummary: outputSummary + ["rawSwiftDecode: candidates=\(rawCandidates.count) kept=\(selected.count)"]
        )
    }

    private static func decodeRawCandidates(
        levels: [(cls: MLMultiArray, bbox: MLMultiArray, kernel: MLMultiArray, side: Int, stride: Float)],
        classIndices: [Int],
        classBlacklist: Set<Int>,
        confidenceThreshold: Float,
        modelSide: Int,
        preNMSLimit: Int
    ) -> [RawCandidate] {
        var candidates: [RawCandidate] = []
        candidates.reserveCapacity(preNMSLimit)

        for level in levels {
            let clsAt = nchwReader(for: level.cls)
            let bboxAt = nchwReader(for: level.bbox)
            let kernelAt = nchwReader(for: level.kernel)
            let side = level.side
            let stride = level.stride

            for y in 0..<side {
                for x in 0..<side {
                    var bestClass = -1
                    var bestScore: Float = 0
                    for classIdx in classIndices where !classBlacklist.contains(classIdx) {
                        let score = sigmoid(clsAt(0, classIdx, y, x))
                        if score > bestScore {
                            bestScore = score
                            bestClass = classIdx
                        }
                    }
                    guard bestClass >= 0, bestScore >= confidenceThreshold else { continue }

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

        return Array(candidates.sorted { $0.box.score > $1.box.score }.prefix(max(1, preNMSLimit)))
    }

    private static func classAwareNMS(_ candidates: [RawCandidate], iouThreshold: Float, limit: Int) -> [RawCandidate] {
        let sorted = candidates.sorted { $0.box.score > $1.box.score }
        var kept: [RawCandidate] = []
        kept.reserveCapacity(limit)

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
                if kept.count >= limit {
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

    private static func buildRawMaskPlane(candidate: RawCandidate, maskFeat: MLMultiArray) -> [Float]? {
        guard candidate.kernel.count == 169 else { return nil }
        let featAt = nchwReader(for: maskFeat)
        let maskSide = 80
        var out = [Float](repeating: 0, count: maskSide * maskSide)

        let w1 = 0
        let w2 = w1 + 80
        let w3 = w2 + 64
        let b1 = w3 + 8
        let b2 = b1 + 8
        let b3 = b2 + 8

        for y in 0..<maskSide {
            for x in 0..<maskSide {
                let gridX = (Float(x) + 0.5) * 8
                let gridY = (Float(y) + 0.5) * 8
                var input = [Float](repeating: 0, count: 10)
                input[0] = (candidate.priorX - gridX) / max(1, candidate.stride * 8)
                input[1] = (candidate.priorY - gridY) / max(1, candidate.stride * 8)
                for c in 0..<8 {
                    input[2 + c] = featAt(0, c, y, x)
                }

                var hidden1 = [Float](repeating: 0, count: 8)
                for o in 0..<8 {
                    var sum = candidate.kernel[b1 + o]
                    for i in 0..<10 {
                        sum += candidate.kernel[w1 + o * 10 + i] * input[i]
                    }
                    hidden1[o] = max(0, sum)
                }

                var hidden2 = [Float](repeating: 0, count: 8)
                for o in 0..<8 {
                    var sum = candidate.kernel[b2 + o]
                    for i in 0..<8 {
                        sum += candidate.kernel[w2 + o * 8 + i] * hidden1[i]
                    }
                    hidden2[o] = max(0, sum)
                }

                var logit = candidate.kernel[b3]
                for i in 0..<8 {
                    logit += candidate.kernel[w3 + i] * hidden2[i]
                }
                out[y * maskSide + x] = sigmoid(logit)
            }
        }
        return out
    }

    private static func buildCombinedRawMaskImage(
        rawMaskPlanes: [[Float]?],
        boxes: [BoxRecord],
        maxMaskCount: Int,
        mapping: ImageMapping
    ) -> UIImage? {
        guard !rawMaskPlanes.isEmpty, !boxes.isEmpty else { return nil }
        let sourceWidth = mapping.sourceWidth
        let sourceHeight = mapping.sourceHeight
        var rgba = [UInt8](repeating: 0, count: sourceWidth * sourceHeight * 4)
        let colors: [(UInt8, UInt8, UInt8)] = [
            (0, 200, 255),
            (0, 255, 120),
            (255, 180, 0),
            (255, 80, 140),
            (180, 120, 255),
            (255, 255, 0),
        ]

        var paintedPixels = 0
        let count = min(max(1, maxMaskCount), boxes.count, rawMaskPlanes.count)
        for index in 0..<count {
            guard let plane = rawMaskPlanes[index], plane.count == 80 * 80 else { continue }
            let box = boxes[index]
            let xMin = max(0, min(sourceWidth - 1, Int(floor(box.x1))))
            let yMin = max(0, min(sourceHeight - 1, Int(floor(box.y1))))
            let xMax = max(0, min(sourceWidth - 1, Int(ceil(box.x2))))
            let yMax = max(0, min(sourceHeight - 1, Int(ceil(box.y2))))
            guard xMax >= xMin, yMax >= yMin else { continue }

            let (r, g, b) = colors[index % colors.count]
            for y in yMin...yMax {
                for x in xMin...xMax {
                    let (sampleX, sampleY) = maskSampleCoordinate(
                        sourceX: x,
                        sourceY: y,
                        maskWidth: 80,
                        maskHeight: 80,
                        mapping: mapping
                    )
                    let value = plane[sampleY * 80 + sampleX]
                    guard value.isFinite, value > 0.5 else { continue }
                    let dest = (y * sourceWidth + x) * 4
                    if rgba[dest + 3] == 0 {
                        paintedPixels += 1
                    }
                    rgba[dest] = b
                    rgba[dest + 1] = g
                    rgba[dest + 2] = r
                    rgba[dest + 3] = max(rgba[dest + 3], UInt8(140))
                }
            }
        }

        guard paintedPixels > 0 else { return nil }
        return rgbaImage(width: sourceWidth, height: sourceHeight, rgba: rgba)
    }

    private static func inputProvider(for pixelBuffer: CVPixelBuffer, model: MLModel) throws -> MLFeatureProvider {
        let imageValue = MLFeatureValue(pixelBuffer: pixelBuffer)
        if model.modelDescription.inputDescriptionsByName["image"]?.type == .image {
            return try MLDictionaryFeatureProvider(dictionary: ["image": imageValue])
        }

        if let firstInputName = model.modelDescription.inputDescriptionsByName.first?.key,
           model.modelDescription.inputDescriptionsByName[firstInputName]?.type == .image {
            return try MLDictionaryFeatureProvider(dictionary: [firstInputName: imageValue])
        }

        if let multiArrayInput = model.modelDescription.inputDescriptionsByName.first(where: { $0.value.type == .multiArray }) {
            let inputName = multiArrayInput.key
            let constraint = multiArrayInput.value.multiArrayConstraint
            let shape = constraint?.shape.map(\.intValue) ?? []
            let dataType = constraint?.dataType ?? .float32
            let array = try rgbNCHWMultiArray(from: pixelBuffer, expectedShape: shape, dataType: dataType)
            return try MLDictionaryFeatureProvider(dictionary: [inputName: MLFeatureValue(multiArray: array)])
        }

        throw NSError(
            domain: "RTMDetImageInference",
            code: 4,
            userInfo: [NSLocalizedDescriptionKey: "Model does not expose a supported image or multi-array input"],
        )
    }

    private static func rgbNCHWMultiArray(
        from pixelBuffer: CVPixelBuffer,
        expectedShape: [Int],
        dataType: MLMultiArrayDataType
    ) throws -> MLMultiArray {
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        guard width > 0, height > 0 else {
            throw NSError(
                domain: "RTMDetImageInference",
                code: 5,
                userInfo: [NSLocalizedDescriptionKey: "Invalid prepared pixel buffer size"],
            )
        }

        let shape = expectedShape.isEmpty ? [1, 3, height, width] : expectedShape
        guard shape.count == 4, shape[0] == 1, shape[1] == 3, shape[2] == height, shape[3] == width else {
            throw NSError(
                domain: "RTMDetImageInference",
                code: 6,
                userInfo: [NSLocalizedDescriptionKey: "Unsupported RTMDet input shape \(shape); expected [1, 3, \(height), \(width)]"],
            )
        }

        let arrayDataType: MLMultiArrayDataType = dataType == .float16 ? .float16 : .float32
        let array = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: arrayDataType)
        let channelStride = width * height
        let float32Ptr = arrayDataType == .float32
            ? array.dataPointer.bindMemory(to: Float.self, capacity: array.count)
            : nil
        let float16Ptr = arrayDataType == .float16
            ? array.dataPointer.bindMemory(to: UInt16.self, capacity: array.count)
            : nil

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
        let meanB: Float = 103.53
        let meanG: Float = 116.28
        let meanR: Float = 123.675
        let stdB: Float = 57.375
        let stdG: Float = 57.12
        let stdR: Float = 58.395

        for y in 0..<height {
            let row = base.advanced(by: y * rowBytes).assumingMemoryBound(to: UInt8.self)
            for x in 0..<width {
                let source = x * 4
                let dest = y * width + x
                let b = Float(row[source])
                let g = Float(row[source + 1])
                let r = Float(row[source + 2])
                writeNormalizedInputValue((b - meanB) / stdB, offset: dest, float32Ptr: float32Ptr, float16Ptr: float16Ptr)
                writeNormalizedInputValue((g - meanG) / stdG, offset: channelStride + dest, float32Ptr: float32Ptr, float16Ptr: float16Ptr)
                writeNormalizedInputValue((r - meanR) / stdR, offset: channelStride * 2 + dest, float32Ptr: float32Ptr, float16Ptr: float16Ptr)
            }
        }

        return array
    }

    private static func writeNormalizedInputValue(
        _ value: Float,
        offset: Int,
        float32Ptr: UnsafeMutablePointer<Float>?,
        float16Ptr: UnsafeMutablePointer<UInt16>?
    ) {
        if let float32Ptr {
            float32Ptr[offset] = value
        } else if let float16Ptr {
            float16Ptr[offset] = Float16(value).bitPattern
        }
    }

    private static func collectMultiArrays(from output: MLFeatureProvider) -> [(name: String, array: MLMultiArray)] {
        output.featureNames.sorted().compactMap { name in
            guard let array = output.featureValue(for: name)?.multiArrayValue else { return nil }
            return (name, array)
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
                    rgba[dest] = b
                    rgba[dest + 1] = g
                    rgba[dest + 2] = r
                    let alpha = UInt8(max(28, min(150, Int((confidence * 150).rounded()))))
                    rgba[dest + 3] = max(rgba[dest + 3], alpha)
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
        let count = array.count
        switch array.dataType {
        case .float32:
            let ptr = array.dataPointer.bindMemory(to: Float.self, capacity: count)
            return { index in
                guard index >= 0, index < count else { return 0 }
                return ptr[index]
            }
        case .float16:
            let ptr = array.dataPointer.bindMemory(to: UInt16.self, capacity: count)
            return { index in
                guard index >= 0, index < count else { return 0 }
                return Float(Float16(bitPattern: ptr[index]))
            }
        case .double:
            let ptr = array.dataPointer.bindMemory(to: Double.self, capacity: count)
            return { index in
                guard index >= 0, index < count else { return 0 }
                return Float(ptr[index])
            }
        case .int32:
            let ptr = array.dataPointer.bindMemory(to: Int32.self, capacity: count)
            return { index in
                guard index >= 0, index < count else { return 0 }
                return Float(ptr[index])
            }
        default:
            return { _ in 0 }
        }
    }

    private static func nchwReader(for array: MLMultiArray) -> (Int, Int, Int, Int) -> Float {
        let valueAtOffset = floatReader(for: array)
        let strides = array.strides.map(\.intValue)
        guard strides.count == 4 else {
            return { _, _, _, _ in 0 }
        }
        return { n, c, y, x in
            valueAtOffset(n * strides[0] + c * strides[1] + y * strides[2] + x * strides[3])
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
        let colorSpace = CGColorSpaceCreateDeviceRGB()
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
            shouldInterpolate: false,
            intent: .defaultIntent
        ) else {
            return nil
        }
        return UIImage(cgImage: cgImage)
    }

    // Shared with the YOLOE image path, duplicated here so the RTMDet spike stays isolated.
    private static func resizeStretchToSquare(src: CVPixelBuffer, size: Int) -> CVPixelBuffer? {
        let attrs: [CFString: Any] = [
            kCVPixelBufferIOSurfacePropertiesKey: [:],
            kCVPixelBufferMetalCompatibilityKey: true,
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true,
        ]
        var out: CVPixelBuffer?
        guard CVPixelBufferCreate(
            kCFAllocatorDefault,
            size,
            size,
            kCVPixelFormatType_32BGRA,
            attrs as CFDictionary,
            &out
        ) == kCVReturnSuccess, let dst = out else { return nil }

        CIContext().render(CIImage(cvPixelBuffer: src).transformed(by: CGAffineTransform(scaleX: CGFloat(size) / CGFloat(CVPixelBufferGetWidth(src)), y: CGFloat(size) / CGFloat(CVPixelBufferGetHeight(src)))), to: dst)
        return dst
    }

    private static func resizeLetterboxToSquare(src: CVPixelBuffer, size: Int) -> CVPixelBuffer? {
        let srcW = CVPixelBufferGetWidth(src)
        let srcH = CVPixelBufferGetHeight(src)
        guard srcW > 0, srcH > 0 else { return nil }

        let attrs: [CFString: Any] = [
            kCVPixelBufferIOSurfacePropertiesKey: [:],
            kCVPixelBufferMetalCompatibilityKey: true,
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true,
        ]
        var out: CVPixelBuffer?
        guard CVPixelBufferCreate(
            kCFAllocatorDefault,
            size,
            size,
            kCVPixelFormatType_32BGRA,
            attrs as CFDictionary,
            &out
        ) == kCVReturnSuccess, let dst = out else { return nil }

        CVPixelBufferLockBaseAddress(dst, [])
        if let base = CVPixelBufferGetBaseAddress(dst) {
            YoloUltralyticsLetterboxFill.fillOpaqueBGRA114(dstBase: base, totalByteCount: CVPixelBufferGetBytesPerRow(dst) * size)
        }
        CVPixelBufferUnlockBaseAddress(dst, [])

        let gain = min(CGFloat(size) / CGFloat(srcW), CGFloat(size) / CGFloat(srcH))
        let scaledW = CGFloat(srcW) * gain
        let scaledH = CGFloat(srcH) * gain
        let tx = (CGFloat(size) - scaledW) * 0.5
        let ty = (CGFloat(size) - scaledH) * 0.5
        let transform = CGAffineTransform(a: gain, b: 0, c: 0, d: gain, tx: tx, ty: ty)
        CIContext().render(CIImage(cvPixelBuffer: src).transformed(by: transform), to: dst)
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
