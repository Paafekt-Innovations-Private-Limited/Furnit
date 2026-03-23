import Foundation
import CoreGraphics

// MARK: - Calibration box (image-space, pixel units)

/// Bounding box from YOLO in **image pixel coordinates** (origin top-left), center-size form.
public struct YoloCalibrationBox: Sendable {
    public var label: String
    public var centerX: CGFloat
    public var centerY: CGFloat
    public var width: CGFloat
    public var height: CGFloat
    public var confidence: Float

    public init(label: String, centerX: CGFloat, centerY: CGFloat, width: CGFloat, height: CGFloat, confidence: Float) {
        self.label = label
        self.centerX = centerX
        self.centerY = centerY
        self.width = width
        self.height = height
        self.confidence = confidence
    }
}

// MARK: - Ratio calibration (pure math, no CoreML)

/// Dimensionless YOLO height ratios for saved-room FurnitureFit (mirrors Android `YoloRatioCalibration`).
public enum YoloRatioCalibration {

    /// Core furniture categories used for per-class target height fractions.
    public static let furnitureCanonicalLabels: Set<String> = [
        "chair", "couch", "bed", "dining table", "table", "wardrobe", "desk", "bench", "ottoman"
    ]

    /// Map raw COCO / UI labels to canonical keys stored in metadata.
    public static func canonicalLabel(raw: String) -> String {
        let trimmed = raw.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty { return "unknown" }

        // Synonyms → canonical
        if trimmed.contains("sofa") || trimmed == "couch" { return "couch" }
        if trimmed.contains("chair") { return "chair" }
        if trimmed.contains("bed") { return "bed" }
        if trimmed.contains("dining") && trimmed.contains("table") { return "dining table" }
        if trimmed.contains("wardrobe") || trimmed.contains("closet") { return "wardrobe" }
        if trimmed.contains("desk") { return "desk" }
        if trimmed.contains("ottoman") { return "ottoman" }
        if trimmed.contains("bench") { return "bench" }
        if trimmed.contains("table") { return "table" }

        return trimmed
    }

    /// Wall-like boxes: very wide vs height, moderate vertical extent, often near horizontal edges.
    public static func wallHeightFractionOrFullFrame(imageSize: CGSize, boxes: [YoloCalibrationBox]) -> Float {
        let imageWidth = max(imageSize.width, 1)
        let imageHeight = max(imageSize.height, 1)

        var best: YoloCalibrationBox?
        var bestArea: CGFloat = 0

        for box in boxes {
            let labelCanon = canonicalLabel(raw: box.label)
            if furnitureCanonicalLabels.contains(labelCanon) { continue }

            let w = box.width
            let h = box.height
            guard w > 1, h > 1 else { continue }

            let widthFrac = Float(w / imageWidth)
            let heightFrac = Float(h / imageHeight)
            let aspect = w / max(h, 1)

            // Heuristic: wall segments are wide and not extremely tall
            let nearEdge = (box.centerX - w * 0.5) / imageWidth < 0.08 ||
                (box.centerX + w * 0.5) / imageWidth > 0.92
            let wideEnough = widthFrac > 0.35
            let moderateHeight = heightFrac > 0.08 && heightFrac < 0.85
            let flatEnough = aspect > 2.0

            if wideEnough && moderateHeight && flatEnough && (nearEdge || widthFrac > 0.55) {
                let area = w * h
                if area > bestArea {
                    bestArea = area
                    best = box
                }
            }
        }

        if let wallBox = best {
            let frac = Float(wallBox.height / imageHeight)
            return min(1.0, max(0.02, frac))
        }
        return 1.0
    }

    /// Per canonical furniture label, robust height / imageHeight in [0.01, 0.95].
    public static func furnitureHeightFractionsByLabel(imageHeight: CGFloat, boxes: [YoloCalibrationBox]) -> [String: Float] {
        let imageHeightSafe = max(imageHeight, 1)
        var byLabel: [String: [Float]] = [:]

        for box in boxes {
            let canon = canonicalLabel(raw: box.label)
            guard furnitureCanonicalLabels.contains(canon) else { continue }
            let frac = Float(box.height / imageHeightSafe)
            let clamped = min(0.95, max(0.01, frac))
            byLabel[canon, default: []].append(clamped)
        }

        var result: [String: Float] = [:]
        for (label, values) in byLabel {
            let sorted = values.sorted()
            guard !sorted.isEmpty else { continue }
            let idx = min(sorted.count - 1, Int(Double(sorted.count) * 0.75 - 0.001))
            result[label] = sorted[idx]
        }
        return result
    }
}

// MARK: - ROI geometry for live FurnitureFit second pass

public enum FurnitureFitRatioGeometry {

    /// Crop rect in **pixel buffer coordinates** (origin top-left) so that bbox height / cropHeight ≈ `rTarget`.
    public static func cropRectForTargetFurnitureFraction(
        frameWidth: Int,
        frameHeight: Int,
        bboxMinX: CGFloat,
        bboxMinY: CGFloat,
        bboxMaxX: CGFloat,
        bboxMaxY: CGFloat,
        rTarget: Float,
        minCropAreaFraction: Float = 0.12,
        maxCropAreaFraction: Float = 0.92
    ) -> CGRect? {
        guard frameWidth > 8, frameHeight > 8 else { return nil }
        let fw = CGFloat(frameWidth)
        let fh = CGFloat(frameHeight)

        let bh = max(1, bboxMaxY - bboxMinY)
        let cx = (bboxMinX + bboxMaxX) * 0.5
        let cy = (bboxMinY + bboxMaxY) * 0.5

        let rt = CGFloat(max(0.05, min(0.9, rTarget)))
        // Desired crop height so bh / cropH ≈ rt → cropH ≈ bh / rt
        var cropH = min(fh, max(bh * 1.15 / rt, bh * 1.25))
        cropH = min(cropH, fh)
        var cropW = min(fw, cropH * (fw / fh))
        cropH = min(cropH, fh)
        cropW = min(cropW, fw)

        let areaFrac = Float((cropW * cropH) / (fw * fh))

        // Reject only *too small* crops; for very large objects where the ideal crop
        // would exceed `maxCropAreaFraction`, fall back to using (almost) the whole
        // frame instead of aborting the ratio pass. This matches the UX expectation
        // of “zooming out as much as the camera allows” rather than silently
        // disabling the second pass.
        if areaFrac < minCropAreaFraction {
            return nil
        }

        let rect: CGRect
        if areaFrac > maxCropAreaFraction {
            // Fallback: ratio target is not achievable without exceeding the frame.
            // Use the full frame so downstream code can still run a second pass and
            // log what happened, even though r_curr will remain larger than r_target.
            rect = CGRect(x: 0, y: 0, width: fw, height: fh)
        } else {
            var originX = cx - cropW * 0.5
            var originY = cy - cropH * 0.5
            originX = max(0, min(fw - cropW, originX))
            originY = max(0, min(fh - cropH, originY))
            rect = CGRect(x: originX, y: originY, width: cropW, height: cropH)
        }

        if rect.width < 32 || rect.height < 32 { return nil }
        return rect
    }

    /// After second pass, verify height fraction in crop; used for debugging / optional rejection.
    public static func heightFractionInCrop(bboxHeightPx: CGFloat, cropHeightPx: CGFloat) -> Float {
        let ch = max(cropHeightPx, 1)
        return Float(min(1, max(0, bboxHeightPx / ch)))
    }
}
