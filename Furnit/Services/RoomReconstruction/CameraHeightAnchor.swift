import Foundation

struct CameraHeightAnchor: ScaleAnchor {
    private let standingPriorMeters = 1.60

    func candidates(ctx: SceneContext) -> [ScaleCandidate] {
        guard let rawHeight = ctx.rawCameraHeightMeters,
              rawHeight.isFinite,
              rawHeight > 0.45,
              rawHeight < 5.0 else {
            return []
        }
        let scale = standingPriorMeters / rawHeight
        guard scale.isFinite, scale > 0,
              let impliedHeight = ctx.impliedRoomHeightForScale(scale),
              impliedHeight.isFinite else { return [] }
        return [
            ScaleCandidate(
                source: "camera_height",
                tier: .cameraHeight,
                depthScale: scale,
                detConf: 1.0,
                impliedRoomHeight: impliedHeight,
                debug: String(
                    format: "standing_prior=%.2fm rawH=%.3fm",
                    standingPriorMeters,
                    rawHeight
                )
            )
        ]
    }
}
