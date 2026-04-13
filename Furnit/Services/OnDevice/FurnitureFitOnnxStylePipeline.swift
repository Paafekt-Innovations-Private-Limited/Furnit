import Foundation

/// Android `FurnitureFitManager` ONNX path parity: NMS → primary scoring → supporting-table heuristic → bbox-limited proto mask (max sigmoid).
enum FurnitureFitOnnxStylePipeline {

    static let confidenceThreshold: Float = 0.25
    static let iouThresholdNms: Float = 0.45
    static let maxDetectionsBeforeNms = 100
    /// Logit threshold for turning YOLOE mask logits into a binary proto mask.
    /// Tuned for Core ML float32 export so that weak positives in thin gaps
    /// (chair handles, bed rails) are suppressed while strong object pixels
    /// remain. ONNX path uses its own mask thresholding.
    static let maskLogitThreshold: Float = 1.0

    /// SAM-style stability score: IoU between masks at (threshold - offset)
    /// and (threshold + offset). Since the high-threshold mask is always a
    /// subset of the low-threshold mask, this reduces to:
    ///
    ///     stability = area(high) / area(low)
    ///
    /// Returns 0 if the low-threshold mask is empty.
    static func calculateMaskStabilityScore(
        logits: [Float],
        threshold: Float,
        offset: Float
    ) -> Float {
        guard !logits.isEmpty, offset > 0 else { return 0 }

        let highThresh = threshold + offset
        let lowThresh = threshold - offset
        var highCount = 0
        var lowCount = 0

        for v in logits {
            if v > highThresh { highCount += 1 }
            if v > lowThresh { lowCount += 1 }
        }

        guard lowCount > 0 else { return 0 }
        return Float(highCount) / Float(lowCount)
    }

    private static let includeSupportingTableForMonitorScene = true
    private static let monitorLikeClassIds: Set<Int> = [1063, 2675, 4105]
    private static let supportingTableClassIds: Set<Int> = [1061, 1301, 1325, 1503, 1885, 2324, 2836, 4564]

    private static func pickSupportingTableForMonitorScene(
        primaryDetection: FurnitureFitDetection,
        detections: [FurnitureFitDetection],
        primaryIndex: Int
    ) -> FurnitureFitDetection? {
        if !includeSupportingTableForMonitorScene { return nil }
        if !monitorLikeClassIds.contains(primaryDetection.classIdx) { return nil }

        let primaryLeft = primaryDetection.x - primaryDetection.w * 0.5
        let primaryRight = primaryDetection.x + primaryDetection.w * 0.5
        let primaryBottom = primaryDetection.y + primaryDetection.h * 0.5
        let primaryArea = max(1e-3, primaryDetection.w * primaryDetection.h)

        var best: FurnitureFitDetection?
        var bestScore: Float = -1

        for (idx, detection) in detections.enumerated() {
            if idx == primaryIndex { continue }
            if !supportingTableClassIds.contains(detection.classIdx) { continue }

            let candidateLeft = detection.x - detection.w * 0.5
            let candidateRight = detection.x + detection.w * 0.5
            let candidateTop = detection.y - detection.h * 0.5
            let overlapWidth = max(0, min(primaryRight, candidateRight) - max(primaryLeft, candidateLeft))
            let horizontalOverlapRatio = overlapWidth / max(1e-3, min(primaryDetection.w, detection.w))
            if horizontalOverlapRatio < 0.35 { continue }

            if detection.y <= primaryDetection.y { continue }

            let verticalGap = candidateTop - primaryBottom
            if verticalGap < -primaryDetection.h * 0.20 || verticalGap > primaryDetection.h * 0.60 { continue }

            let widthRatio = detection.w / max(1e-3, primaryDetection.w)
            if widthRatio < 0.75 || widthRatio > 5.0 { continue }

            let areaRatio = (detection.w * detection.h) / primaryArea
            if areaRatio < 0.50 || areaRatio > 12.0 { continue }

            let closenessTerm = 1 - min(1, abs(verticalGap) / max(primaryDetection.h * 0.60, 1e-3))
            let score = detection.confidence * horizontalOverlapRatio * max(0.1, closenessTerm)

            if score > bestScore {
                bestScore = score
                best = detection
            }
        }
        return best
    }

    /// Android `collectMaskDetections` (returns list for mask fusion, primary first).
    static func collectMaskDetections(
        primaryIndex: Int,
        detections: [FurnitureFitDetection]
    ) -> [FurnitureFitDetection] {
        guard primaryIndex >= 0, primaryIndex < detections.count else { return [] }
        let primaryDetection = detections[primaryIndex]

        let supportingTableDetection = pickSupportingTableForMonitorScene(
            primaryDetection: primaryDetection,
            detections: detections,
            primaryIndex: primaryIndex
        )

        let primaryLeft = primaryDetection.x - primaryDetection.w * 0.5
        let primaryTop = primaryDetection.y - primaryDetection.h * 0.5
        let primaryRight = primaryDetection.x + primaryDetection.w * 0.5
        let primaryBottom = primaryDetection.y + primaryDetection.h * 0.5
        let encompassTolerance: Float = 2
        let minimumCandidateConfidence: Float = 0.1
        let bboxDuplicateThreshold: Float = 0.7

        var bboxKept: [FurnitureFitDetection] = []

        for (idx, detection) in detections.enumerated() {
            if idx == primaryIndex || detection.confidence < minimumCandidateConfidence { continue }

            let candidateLeft = detection.x - detection.w * 0.5
            let candidateTop = detection.y - detection.h * 0.5
            let candidateRight = detection.x + detection.w * 0.5
            let candidateBottom = detection.y + detection.h * 0.5

            let encompassesPrimary =
                candidateLeft <= primaryLeft + encompassTolerance &&
                candidateTop <= primaryTop + encompassTolerance &&
                candidateRight >= primaryRight - encompassTolerance &&
                candidateBottom >= primaryBottom - encompassTolerance
            if encompassesPrimary { continue }

            let intersectsPrimary =
                !(candidateRight < primaryLeft || candidateLeft > primaryRight ||
                  candidateBottom < primaryTop || candidateTop > primaryBottom)
            if !intersectsPrimary { continue }

            let tooLarge =
                detection.w > primaryDetection.w * 1.5 &&
                detection.h > primaryDetection.h * 1.5
            if tooLarge { continue }

            if FurnitureFitIoU.calculate(detection, primaryDetection) > bboxDuplicateThreshold { continue }

            var shouldSkip = false
            var replaceIndex: Int?
            for (k, keptDetection) in bboxKept.enumerated() {
                let iou = FurnitureFitIoU.calculate(detection, keptDetection)
                if iou > bboxDuplicateThreshold {
                    if detection.confidence > keptDetection.confidence {
                        replaceIndex = k
                    } else {
                        shouldSkip = true
                    }
                    break
                }
            }
            if shouldSkip { continue }
            if let r = replaceIndex {
                bboxKept[r] = detection
            } else {
                bboxKept.append(detection)
            }
        }

        var maskDetections: [FurnitureFitDetection] = [primaryDetection]
        maskDetections.append(contentsOf: bboxKept)
        if let st = supportingTableDetection, !maskDetections.contains(where: { $0.classIdx == st.classIdx && $0.x == st.x && $0.y == st.y }) {
            maskDetections.append(st)
        }
        return maskDetections
    }

    /// Raw YOLOE prototype mask using per-pixel logits inside each detection's
    /// original bbox only. No sigmoid, no morphology, no heuristic expansion.
    static func buildBboxLimitedSigmoidMask(
        planes: [Float],
        protoW: Int,
        protoH: Int,
        modelSide: Float,
        detections: [FurnitureFitDetection]
    ) -> [UInt8] {
        return buildBboxLimitedSigmoidMaskWithLogits(
            planes: planes, protoW: protoW, protoH: protoH,
            modelSide: modelSide, detections: detections
        ).binary
    }

    /// Same as `buildBboxLimitedSigmoidMask` but also returns the raw float
    /// logits before thresholding, for debug visualization.
    static func buildBboxLimitedSigmoidMaskWithLogits(
        planes: [Float],
        protoW: Int,
        protoH: Int,
        modelSide: Float,
        detections: [FurnitureFitDetection]
    ) -> (binary: [UInt8], logits: [Float]) {
        let hwProto = protoW * protoH
        guard planes.count >= 32 * hwProto else {
            return ([UInt8](repeating: 0, count: hwProto),
                    [Float](repeating: 0, count: hwProto))
        }

        var maskProto = [Float](repeating: 0, count: hwProto)
        let protoScaleX = modelSide / Float(protoW)
        let protoScaleY = modelSide / Float(protoH)

        for detection in detections {
            guard detection.coeffs.count >= 32 else { continue }

            let bboxLeft = Int(floor((detection.x - detection.w * 0.5) / protoScaleX))
                .clamped(to: 0...(protoW - 1))
            let bboxTop = Int(floor((detection.y - detection.h * 0.5) / protoScaleY))
                .clamped(to: 0...(protoH - 1))
            let bboxRight = Int(floor((detection.x + detection.w * 0.5) / protoScaleX))
                .clamped(to: 0...(protoW - 1))
            let bboxBottom = Int(floor((detection.y + detection.h * 0.5) / protoScaleY))
                .clamped(to: 0...(protoH - 1))

            for py in bboxTop...bboxBottom {
                let rowBase = py * protoW
                for px in bboxLeft...bboxRight {
                    let protoPixelIndex = rowBase + px
                    var sum: Float = 0
                    var coeffIndex = 0
                    while coeffIndex < 32 {
                        let protoIdx = coeffIndex * hwProto + protoPixelIndex
                        sum += detection.coeffs[coeffIndex] * planes[protoIdx]
                        coeffIndex += 1
                    }
                    if sum > maskProto[protoPixelIndex] {
                        maskProto[protoPixelIndex] = sum
                    }
                }
            }
        }

        let threshold = maskLogitThreshold
        let binary = maskProto.map { $0 > threshold ? UInt8(255) : UInt8(0) }
        return (binary, maskProto)
    }

    // MARK: - ASCII mask visualization for debug logs

    /// Samples the proto-resolution mask down to a text grid and returns
    /// multi-line strings you can print to the console.
    ///
    /// - `logits`: raw float logits (protoW × protoH, row-major)
    /// - `binary`: thresholded UInt8 mask (same layout)
    /// - `protoW`, `protoH`: proto dimensions
    /// - `gridCols`, `gridRows`: target ASCII grid size (default 64×32)
    ///
    /// Returns two strings: (logitArt, binaryArt).
    /// Logit art uses ` ░▒▓█` to show intensity; binary art uses `·` / `█`.
    static func asciiMaskVisualization(
        logits: [Float],
        binary: [UInt8],
        protoW: Int,
        protoH: Int,
        gridCols: Int = 64,
        gridRows: Int = 32
    ) -> (logitArt: String, binaryArt: String) {
        guard protoW > 0, protoH > 0,
              logits.count >= protoW * protoH,
              binary.count >= protoW * protoH else {
            return ("(empty)", "(empty)")
        }

        let cols = min(gridCols, protoW)
        let rows = min(gridRows, protoH)

        var logitMin: Float = .greatestFiniteMagnitude
        var logitMax: Float = -.greatestFiniteMagnitude
        for v in logits {
            if v < logitMin { logitMin = v }
            if v > logitMax { logitMax = v }
        }
        let logitRange = logitMax - logitMin
        let logitChars: [Character] = [" ", "░", "▒", "▓", "█"]

        var logitLines = [String]()
        var binaryLines = [String]()

        for row in 0..<rows {
            let srcY = row * protoH / rows
            var logitRow = ""
            var binaryRow = ""
            for col in 0..<cols {
                let srcX = col * protoW / cols
                let idx = srcY * protoW + srcX

                let logitVal = logits[idx]
                if logitRange > 1e-6 {
                    let norm = (logitVal - logitMin) / logitRange
                    let ci = min(logitChars.count - 1, Int(norm * Float(logitChars.count)))
                    logitRow.append(logitChars[ci])
                } else {
                    logitRow.append(logitVal > 0 ? "█" : " ")
                }

                binaryRow.append(binary[idx] > 0 ? "█" : "·")
            }
            logitLines.append(logitRow)
            binaryLines.append(binaryRow)
        }

        let logitStats = String(format: "min=%.3f max=%.3f range=%.3f",
                                logitMin, logitMax, logitRange)
        let onCount = binary.prefix(protoW * protoH).filter { $0 > 0 }.count
        let totalPx = protoW * protoH
        let binaryStats = String(format: "on=%d/%d (%.1f%%)",
                                 onCount, totalPx,
                                 Float(onCount) / Float(max(1, totalPx)) * 100)

        let logitArt = "LOGIT HEATMAP (\(protoW)x\(protoH) → \(cols)x\(rows)) \(logitStats)\n"
            + logitLines.joined(separator: "\n")
        let binaryArt = "BINARY MASK (\(protoW)x\(protoH) → \(cols)x\(rows)) \(binaryStats)\n"
            + binaryLines.joined(separator: "\n")

        return (logitArt, binaryArt)
    }
}

private extension Float {
    func clamped(to range: ClosedRange<Float>) -> Float {
        min(max(self, range.lowerBound), range.upperBound)
    }
}

private extension BinaryInteger {
    func clamped(to range: ClosedRange<Self>) -> Self {
        min(max(self, range.lowerBound), range.upperBound)
    }
}
