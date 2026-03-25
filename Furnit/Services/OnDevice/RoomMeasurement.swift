import Foundation
import UIKit

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
}
