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

    /// Depth-adaptive inset from the back wall (18% for tiny rooms → 50% for deep rooms).
    private static func backCenterInsetFraction(depth: Float) -> Float {
        let t = min(1.0, max(0.0, depth / 6.0))
        return 0.18 + 0.32 * t
    }

    /// Eye and look-at target for MetalSplatter: **back center** of the room, looking at the **front wall** center.
    /// Matches ``RoomBoundaryManager/getCameraAtBackCenter()`` and Android list-room / thumbnail framing (not “in front of front wall” at centroid).
    func defaultSplatCameraEyeAndTarget(cameraPadding: Float = 0.3) -> (eye: SIMD3<Float>, target: SIMD3<Float>) {
        let fraction = Self.backCenterInsetFraction(depth: depth)
        let insetFromBack = max(depth * fraction, cameraPadding)
        let camZ = minZ + insetFromBack
        let camY = centerY + 0.4
        let eye = SIMD3<Float>(centerX, camY, camZ)
        let target = SIMD3<Float>(centerX, centerY, maxZ)
        return (eye, target)
    }
}
