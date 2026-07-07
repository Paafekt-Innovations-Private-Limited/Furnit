import Foundation

struct ResolvedFocal: Sendable {
    let fx: Float
    let fy: Float
    let source: String
    let horizontalFOVDegrees: Float
    let clamped: Bool
}

enum FocalResolver {
    private static let minimumHorizontalFOVDegrees: Float = 55
    private static let maximumHorizontalFOVDegrees: Float = 88
    private static let defaultHorizontalFOVDegrees: Float = 70

    static func resolve(
        vpFocalPx: Float?,
        geoCalib: GeoCalibCalibrationResult?,
        imageWidth: Int,
        imageHeight: Int
    ) -> ResolvedFocal {
        let width = Float(max(imageWidth, 1))

        if let vpFocalPx,
           vpFocalPx.isFinite,
           vpFocalPx > 1 {
            let clamped = clampFocal(vpFocalPx, imageWidth: width)
            log(clamped, source: "vp")
            return ResolvedFocal(
                fx: clamped.focalPx,
                fy: clamped.focalPx,
                source: "vp",
                horizontalFOVDegrees: clamped.fovDegrees,
                clamped: clamped.clamped
            )
        }

        if let geoCalib, geoCalib.sourceWidth > 0 {
            let uniformScale = width / Float(max(geoCalib.sourceWidth, 1))
            let focal = geoCalib.focalLengthXPixels * uniformScale
            if focal.isFinite, focal > 1 {
                let clamped = clampFocal(focal, imageWidth: width)
                log(clamped, source: "geocalib")
                return ResolvedFocal(
                    fx: clamped.focalPx,
                    fy: clamped.focalPx,
                    source: clamped.clamped ? "geocalib_clamped" : "geocalib",
                    horizontalFOVDegrees: clamped.fovDegrees,
                    clamped: clamped.clamped
                )
            }
        }

        let defaultFocal = focalPixels(forHorizontalFOVDegrees: defaultHorizontalFOVDegrees, imageWidth: width)
        let clamped = clampFocal(defaultFocal, imageWidth: width)
        log(clamped, source: "default")
        return ResolvedFocal(
            fx: clamped.focalPx,
            fy: clamped.focalPx,
            source: "default_70deg",
            horizontalFOVDegrees: clamped.fovDegrees,
            clamped: clamped.clamped
        )
    }

    private static func clampFocal(_ focalPx: Float, imageWidth: Float) -> (focalPx: Float, fovDegrees: Float, clamped: Bool) {
        let minFocal = focalPixels(forHorizontalFOVDegrees: maximumHorizontalFOVDegrees, imageWidth: imageWidth)
        let maxFocal = focalPixels(forHorizontalFOVDegrees: minimumHorizontalFOVDegrees, imageWidth: imageWidth)
        let clampedFocal = min(max(focalPx, minFocal), maxFocal)
        return (
            focalPx: clampedFocal,
            fovDegrees: horizontalFOVDegrees(focalPx: clampedFocal, imageWidth: imageWidth),
            clamped: abs(clampedFocal - focalPx) > 0.5
        )
    }

    private static func focalPixels(forHorizontalFOVDegrees fovDegrees: Float, imageWidth: Float) -> Float {
        (imageWidth * 0.5) / tan((fovDegrees * .pi / 180) * 0.5)
    }

    private static func horizontalFOVDegrees(focalPx: Float, imageWidth: Float) -> Float {
        2 * atan((imageWidth * 0.5) / max(focalPx, 1)) * 180 / .pi
    }

    private static func log(_ resolved: (focalPx: Float, fovDegrees: Float, clamped: Bool), source: String) {
        logDebug(String(
            format: "[Focal] source=%@ focal_px=%.1f hFOV=%.1f clamped=%@",
            source,
            resolved.focalPx,
            resolved.fovDegrees,
            resolved.clamped ? "true" : "false"
        ))
    }
}
