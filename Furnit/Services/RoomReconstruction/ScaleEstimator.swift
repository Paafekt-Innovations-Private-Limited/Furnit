import Foundation

struct ScaleEstimatorResult: Sendable {
    let depthScale: Double
    let confidence: Double
    let source: String
    let observations: [ScaleObservation]
    let clampedOut: [ScaleObservation]

    var debugSummary: String {
        let observationText = observations.map { observation in
            String(
                format: "%@/t%d=%.4f/conf%.2f/h%.2f/%@",
                observation.source,
                observation.tier.rawValue,
                observation.depthScale,
                observation.detConf,
                observation.impliedRoomHeight,
                observation.debug
            )
        }.joined(separator: "; ")
        let clampedText = clampedOut.map { observation in
            String(format: "%@=%.4f/h%.2f", observation.source, observation.depthScale, observation.impliedRoomHeight)
        }.joined(separator: "; ")
        return String(
            format: "source=%@ scale=%.4f conf=%.2f anchors=[%@] clamped_out=[%@]",
            source,
            depthScale,
            confidence,
            observationText,
            clampedText
        )
    }
}

struct ScaleEstimator {
    let anchors: [ScaleAnchor]

    init(anchors: [ScaleAnchor] = [
        TileAnchor(),
        ObjectAnchor(),
    ]) {
        self.anchors = anchors
    }

    func estimate(ctx: SceneContext) -> ScaleEstimatorResult {
        let observations = anchors.flatMap { $0.candidates(ctx: ctx) }
        let resolved = Self.resolveScale(
            anchors: observations,
            cameraHeightRawMeters: ctx.rawCameraHeightMeters,
            impliedRoomHeightForScale: ctx.impliedRoomHeightForScale
        )
        return ScaleEstimatorResult(
            depthScale: resolved.scale,
            confidence: resolved.confidence,
            source: resolved.source,
            observations: resolved.pool,
            clampedOut: resolved.clampedOut
        )
    }

    static func fuseScale(_ candidates: [ScaleCandidate], fallback: Double) -> (scale: Double, confidence: Double) {
        let resolved = resolveScale(
            anchors: candidates,
            cameraHeightRawMeters: 1.65 / max(fallback, 1e-6),
            impliedRoomHeightForScale: { _ in 2.7 }
        )
        return (resolved.scale, resolved.confidence)
    }

    static func resolveScale(
        anchors observations: [ScaleObservation],
        cameraHeightRawMeters: Double?,
        impliedRoomHeightForScale: (Double) -> Double?
    ) -> (scale: Double, confidence: Double, source: String, pool: [ScaleObservation], clampedOut: [ScaleObservation]) {
        let valid = observations.filter { observation in
            observation.depthScale.isFinite &&
                observation.depthScale > 0 &&
                (1.9...3.6).contains(observation.impliedRoomHeight)
        }
        let clampedOut = observations.filter { observation in
            !valid.contains { kept in
                kept.source == observation.source &&
                    kept.depthScale == observation.depthScale &&
                    kept.impliedRoomHeight == observation.impliedRoomHeight
            }
        }

        let tier1 = valid.filter { $0.tier == .architectural && $0.detConf > 0.5 }
        if !tier1.isEmpty {
            return (medianScale(tier1), 0.9, "tier1_architectural", tier1, clampedOut)
        }

        let tier2 = valid.filter { $0.tier == .codeFixture && $0.detConf > 0.5 }
        if !tier2.isEmpty {
            return (medianScale(tier2), 0.75, "tier2_fixture", tier2, clampedOut)
        }

        guard let cameraHeightRawMeters,
              cameraHeightRawMeters.isFinite,
              cameraHeightRawMeters > 0 else {
            return (1.0, 0.05, "tier3_camera_height_unavailable", [], clampedOut)
        }
        let cameraHeightMeters = min(1.75, max(1.55, 1.65))
        let scale = cameraHeightMeters / cameraHeightRawMeters
        let impliedHeight = impliedRoomHeightForScale(scale) ?? .nan
        let cameraObservation = ScaleObservation(
            source: "camera_height",
            tier: .cameraHeight,
            depthScale: scale,
            detConf: 1.0,
            impliedRoomHeight: impliedHeight,
            debug: String(format: "clampedH=%.2fm rawH=%.3fm", cameraHeightMeters, cameraHeightRawMeters)
        )
        return (scale, 0.5, "tier3_camera_height", [cameraObservation], clampedOut)
    }

    private static func medianScale(_ observations: [ScaleObservation]) -> Double {
        let scales = observations.map(\.depthScale).sorted()
        guard !scales.isEmpty else { return 1.0 }
        let middle = scales.count / 2
        if scales.count.isMultiple(of: 2) {
            return (scales[middle - 1] + scales[middle]) * 0.5
        }
        return scales[middle]
    }
}
