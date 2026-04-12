import Foundation
import UIKit
import simd

// MARK: - Photo Orientation

/// Orientation of the source photo used for 3D reconstruction
enum PhotoOrientation: String, Codable, Hashable {
    case portrait
    case landscape
    case square

    /// Detect orientation from EXIF metadata
    /// iPhone camera: .up = landscape (sensor native), .left/.right = portrait (rotated)
    static func detect(from image: UIImage) -> PhotoOrientation {
        let exif = image.imageOrientation
        let uiWidth = image.size.width
        let uiHeight = image.size.height
        let cgWidth = image.cgImage?.width ?? 0
        let cgHeight = image.cgImage?.height ?? 0

        logDebug("📐 [Orientation] ========== DETECTION START ==========")
        logDebug("📐 [Orientation] EXIF raw value: \(exif.rawValue)")
        logDebug("📐 [Orientation] EXIF name: \(exif)")
        logDebug("📐 [Orientation] UIImage.size: \(uiWidth) x \(uiHeight)")
        logDebug("📐 [Orientation] CGImage size: \(cgWidth) x \(cgHeight)")

        switch exif {
        case .up, .upMirrored, .down, .downMirrored:
            logDebug("📐 [Orientation] EXIF .up/.down → landscape")
            logDebug("📐 [Orientation] ========== RESULT: LANDSCAPE ==========")
            return .landscape

        case .left, .leftMirrored, .right, .rightMirrored:
            logDebug("📐 [Orientation] EXIF .left/.right → portrait")
            logDebug("📐 [Orientation] ========== RESULT: PORTRAIT ==========")
            return .portrait

        @unknown default:
            logDebug("📐 [Orientation] EXIF unknown → portrait (default)")
            logDebug("📐 [Orientation] ========== RESULT: PORTRAIT (default) ==========")
            return .portrait
        }
    }

    /// For **saved** SHARP thumbnails and other upright JPEGs: orientation is usually baked to `.up`, so
    /// ``detect(from:)`` would wrongly return `.landscape` for portrait rooms (AR roll then stays landscape-native).
    ///
    /// Prefer **logical** width/height from ``UIImage/size`` (orientation is applied by UIKit) so landscape
    /// captures stay landscape even when on-disk EXIF still says `.left`/`.right` from the capture pipeline.
    /// Only for ~square images do we fall back to EXIF hints.
    static func detectFromStoredRoomThumbnail(_ image: UIImage) -> PhotoOrientation {
        let exif = image.imageOrientation
        let w = Float(image.size.width * image.scale)
        let h = Float(image.size.height * image.scale)

        guard w > 1, h > 1 else { return .portrait }

        if h > w * 1.05 {
            logDebug("📐 [Orientation] thumbnail logical \(w)×\(h) exif=\(exif) → portrait")
            return .portrait
        }
        if w > h * 1.05 {
            logDebug("📐 [Orientation] thumbnail logical \(w)×\(h) exif=\(exif) → landscape")
            return .landscape
        }

        // ~square: disambiguate with EXIF (legacy portrait phone storage quirks).
        switch exif {
        case .left, .leftMirrored, .right, .rightMirrored:
            logDebug("📐 [Orientation] thumbnail ~square exif .left/.right → portrait")
            return .portrait
        case .up, .upMirrored, .down, .downMirrored:
            logDebug("📐 [Orientation] thumbnail ~square exif .up/.down → square")
            return .square
        @unknown default:
            return .portrait
        }
    }

    /// User-friendly description
    var hint: String {
        switch self {
        case .landscape:
            return "Landscape – better for full wall width"
        case .portrait:
            return "Portrait – great for tall furniture"
        case .square:
            return "Square"
        }
    }
}

/// Actual min/max bounds of the room in 3D space
struct RoomBounds: Hashable {
    let minX: Float
    let maxX: Float
    let minY: Float
    let maxY: Float
    let minZ: Float
    let maxZ: Float

    var centerX: Float { (minX + maxX) / 2 }
    var centerY: Float { (minY + maxY) / 2 }
    var centerZ: Float { (minZ + maxZ) / 2 }
    var width: Float { maxX - minX }
    var height: Float { maxY - minY }
    var depth: Float { maxZ - minZ }
    /// Front wall Z (closest to camera, largest Z value)
    var frontZ: Float { maxZ }

    // MARK: - Default splat / list-room camera (Android `RoomBoundaryManager` parity)

    /// Small depth-adaptive inset from the back face, then an extra pull toward that back wall (see pull constants).
    private static func backCenterInsetFraction(depth: Float) -> Float {
        let t = min(1.0, max(0.0, depth / 6.0))
        return 0.035 + 0.065 * t
    }

    /// Nudge eye toward the back wall after inset (moves away from the front wall in +Z-forward framing).
    private static let pullTowardBackWallFraction: Float = 0.055
    private static let maxInsetFromBackAsDepthFraction: Float = 0.10
    private static let minClearanceFromBackPlane: Float = 0.06

    /// Eye and look-at target for MetalSplatter: **back center** of the room, looking at the **front wall** center
    /// (`minZ` → back face, `maxZ` → front in trimmed PLY bounds). Same rail for **portrait and landscape** captures:
    /// Android WebView applies a landscape π·Y mesh rotation; iOS Metal does not, so swapping Z rails for landscape
    /// put the camera on the wrong end (facing open/grey past the slab). ``photoOrientation`` is still passed for
    /// call-site clarity; AR roll uses it separately in ``GaussianSplatView``.
    func defaultSplatCameraEyeAndTarget(
        cameraPadding: Float = 0.05,
        photoOrientation: PhotoOrientation? = nil
    ) -> (eye: SIMD3<Float>, target: SIMD3<Float>) {
        _ = photoOrientation
        let d = max(depth, 1e-3)
        let fraction = Self.backCenterInsetFraction(depth: d)
        var insetFromBack = max(d * fraction, cameraPadding)
        insetFromBack = min(insetFromBack, d * Self.maxInsetFromBackAsDepthFraction)
        let pull = d * Self.pullTowardBackWallFraction
        let camY = centerY + 0.4
        var z = minZ + insetFromBack
        z -= pull
        z = max(z, minZ + Self.minClearanceFromBackPlane)
        let camZ = z
        let targetZ = maxZ
        let eye = SIMD3<Float>(centerX, camY, camZ)
        let target = SIMD3<Float>(centerX, centerY, targetZ)
        return (eye, target)
    }
}
