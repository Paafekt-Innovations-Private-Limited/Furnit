import CoreGraphics
import Foundation

struct ObjectAnchor: ScaleAnchor {
    private struct Prior {
        let source: String
        let dimension: Dimension
        let meters: Double
        let tier: AnchorTier
    }

    private enum Dimension {
        case width
        case height
    }

    func candidates(ctx: SceneContext) -> [ScaleCandidate] {
        ctx.objectBoxes.compactMap { objectBox in
            guard let prior = prior(for: objectBox, ctx: ctx) else { return nil }
            guard let rawDepth = depthPercentile(
                depthMap: ctx.rawDepth,
                rect: objectBox.rect,
                fraction: 0.20
            ) else {
                return nil
            }
            let pixelSize: Double
            switch prior.dimension {
            case .width:
                pixelSize = objectBox.rect.width
            case .height:
                pixelSize = objectBox.rect.height
            }
            guard pixelSize > 8, ctx.focalPx > 1 else { return nil }
            let rawSize = pixelSize * rawDepth / ctx.focalPx
            guard rawSize.isFinite, rawSize > 0.05 else { return nil }

            let scale = prior.meters / rawSize
            guard let impliedHeight = ctx.impliedRoomHeightForScale(scale),
                  impliedHeight.isFinite else { return nil }
            guard scale.isFinite, scale > 0 else { return nil }
            return ScaleCandidate(
                source: prior.source,
                tier: prior.tier,
                depthScale: scale,
                detConf: Double(objectBox.confidence),
                impliedRoomHeight: impliedHeight,
                debug: String(
                    format: "cls=%d conf=%.2f px=%.0f rawSize=%.3fm prior=%.2fm tier=%d",
                    objectBox.classIdx,
                    objectBox.confidence,
                    pixelSize,
                    rawSize,
                    prior.meters,
                    prior.tier.rawValue
                )
            )
        }
    }

    private func prior(for objectBox: ScaleObjectBox, ctx: SceneContext) -> Prior? {
        switch objectBox.classIdx {
        case 56, 57, 58, 59:
            // Furniture is too variable to set room scale. It is still useful as an
            // exclusion mask, but never emits a scale observation.
            return nil
        case 61:
            let imageHeight = max(ctx.imageSize.height, 1)
            let bottomMargin = (imageHeight - objectBox.rect.maxY) / imageHeight
            let heightFraction = objectBox.rect.height / imageHeight
            if bottomMargin <= 0.05, heightFraction < 0.20 {
                return Prior(source: "toilet_seat", dimension: .height, meters: 0.40, tier: .codeFixture)
            }
            return Prior(source: "toilet_full", dimension: .height, meters: 0.78, tier: .codeFixture)
        default:
            return nil
        }
    }

    private func depthPercentile(depthMap: RoomDepthMap, rect: CGRect, fraction: Double) -> Double? {
        guard !depthMap.isEmpty, let width = depthMap.first?.count, width > 0 else { return nil }
        let height = depthMap.count
        let left = max(0, min(width - 1, Int(rect.minX.rounded(.down))))
        let right = max(0, min(width - 1, Int(rect.maxX.rounded(.up))))
        let top = max(0, min(height - 1, Int(rect.minY.rounded(.down))))
        let bottom = max(0, min(height - 1, Int(rect.maxY.rounded(.up))))
        guard left < right, top < bottom else { return nil }

        let maxSpan = max(right - left, bottom - top)
        let step = max(1, maxSpan / 120)
        var samples: [Float] = []
        for row in stride(from: top, through: bottom, by: step) {
            for column in stride(from: left, through: right, by: step) {
                let depth = depthMap[row][column]
                if depth.isFinite, depth > 0.1, depth < 50 {
                    samples.append(depth)
                }
            }
        }
        guard !samples.isEmpty else { return nil }
        samples.sort()
        let index = max(0, min(samples.count - 1, Int((fraction * Double(samples.count - 1)).rounded())))
        return Double(samples[index])
    }

}
