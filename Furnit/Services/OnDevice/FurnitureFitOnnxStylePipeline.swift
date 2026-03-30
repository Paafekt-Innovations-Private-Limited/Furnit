import Foundation

/// Android `FurnitureFitManager` ONNX path parity: NMS → primary scoring → supporting-table heuristic → bbox-limited proto mask (max sigmoid).
enum FurnitureFitOnnxStylePipeline {

    static let confidenceThreshold: Float = 0.25
    static let iouThresholdNms: Float = 0.45
    static let maxDetectionsBeforeNms = 100

    private static let includeSupportingTableForMonitorScene = true
    private static let monitorLikeClassIds: Set<Int> = [1063, 2675, 4105]
    private static let supportingTableClassIds: Set<Int> = [1061, 1301, 1325, 1503, 1885, 2324, 2836, 4564]

    private struct PrimaryCandidateScore {
        let score: Float
        let isInteriorCandidate: Bool
    }

    private static func primaryDetectionScore(
        centerX: Float,
        centerY: Float,
        width: Float,
        height: Float,
        confidence: Float,
        frameWidth: Float,
        frameHeight: Float
    ) -> PrimaryCandidateScore {
        guard centerX.isFinite, centerY.isFinite, width.isFinite, height.isFinite, confidence.isFinite else {
            return PrimaryCandidateScore(score: -1, isInteriorCandidate: false)
        }
        if frameWidth <= 1 || frameHeight <= 1 || width <= 0 || height <= 0 {
            return PrimaryCandidateScore(score: -1, isInteriorCandidate: false)
        }

        let frameArea = frameWidth * frameHeight
        let areaNormalized = (width * height) / frameArea
        let minimumConfidence: Float = 0.15
        let minimumAreaNormalized: Float = 0.02
        if confidence < minimumConfidence || areaNormalized < minimumAreaNormalized {
            return PrimaryCandidateScore(score: -1, isInteriorCandidate: false)
        }

        let frameCenterX = frameWidth * 0.5
        let frameCenterY = frameHeight * 0.5
        let deltaX = (centerX - frameCenterX) / max(frameCenterX, 1)
        let deltaY = (centerY - frameCenterY) / max(frameCenterY, 1)
        let centerDistance = min(1, sqrt(deltaX * deltaX + deltaY * deltaY))
        let centerScore = 1 - centerDistance

        let boxLeft = centerX - width * 0.5
        let boxTop = centerY - height * 0.5
        let boxRight = centerX + width * 0.5
        let boxBottom = centerY + height * 0.5
        let edgeMarginX = max(frameWidth * 0.04, 1)
        let edgeMarginY = max(frameHeight * 0.04, 1)
        let leftClearance = (boxLeft / edgeMarginX).clamped(to: 0...1)
        let topClearance = (boxTop / edgeMarginY).clamped(to: 0...1)
        let rightClearance = ((frameWidth - boxRight) / edgeMarginX).clamped(to: 0...1)
        let bottomClearance = ((frameHeight - boxBottom) / edgeMarginY).clamped(to: 0...1)
        let edgeClearanceScore = max(0.1, min(min(leftClearance, topClearance), min(rightClearance, bottomClearance)))
        let isInteriorCandidate =
            leftClearance >= 1 && topClearance >= 1 && rightClearance >= 1 && bottomClearance >= 1

        let confidenceTerm = pow(confidence, 1.0)
        let areaTerm = pow(areaNormalized, 0.8)
        let centerTerm = pow(max(0, centerScore), 1.0)
        let edgeTerm = pow(edgeClearanceScore, 1.0)
        let score = confidenceTerm * areaTerm * centerTerm * edgeTerm
        return PrimaryCandidateScore(score: score, isInteriorCandidate: isInteriorCandidate)
    }

    /// Returns index into `detections` of the primary box (Android `pickPrimaryOnnxDetection`).
    static func pickPrimaryIndex(
        detections: [FurnitureFitDetection],
        frameWidth: Float,
        frameHeight: Float
    ) -> Int? {
        guard !detections.isEmpty else { return nil }

        var bestIndex: Int?
        var bestScore: Float = -1
        var bestEdgeIndex: Int?
        var bestEdgeScore: Float = -1

        for (idx, detection) in detections.enumerated() {
            let candidate = primaryDetectionScore(
                centerX: detection.x,
                centerY: detection.y,
                width: detection.w,
                height: detection.h,
                confidence: detection.confidence,
                frameWidth: frameWidth,
                frameHeight: frameHeight
            )
            if candidate.isInteriorCandidate && candidate.score > bestScore {
                bestScore = candidate.score
                bestIndex = idx
            } else if !candidate.isInteriorCandidate && candidate.score > bestEdgeScore {
                bestEdgeScore = candidate.score
                bestEdgeIndex = idx
            }
        }

        if let bestIndex { return bestIndex }
        if let bestEdgeIndex { return bestEdgeIndex }
        return detections.enumerated().max(by: { $0.element.confidence < $1.element.confidence })?.offset
    }

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

    /// Bbox-limited proto mask: per-pixel max sigmoid (Android ONNX loop).
    static func buildBboxLimitedSigmoidMask(
        planes: [Float],
        protoW: Int,
        protoH: Int,
        modelSide: Float,
        detections: [FurnitureFitDetection]
    ) -> [UInt8] {
        let hwProto = protoW * protoH
        guard planes.count >= 32 * hwProto else { return [UInt8](repeating: 0, count: hwProto) }

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
                    let p = rowBase + px
                    var sum: Float = 0
                    var c = 0
                    while c < 32 {
                        let protoIdx = c * hwProto + p
                        sum += detection.coeffs[c] * planes[protoIdx]
                        c += 1
                    }
                    let sigmoidVal = 1 / (1 + exp(-sum))
                    if sigmoidVal > maskProto[p] {
                        maskProto[p] = sigmoidVal
                    }
                }
            }
        }

        return maskProto.map { $0 > 0.5 ? UInt8(255) : UInt8(0) }
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
