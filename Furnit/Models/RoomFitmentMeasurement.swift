import Foundation
import CoreGraphics
import simd

// MARK: - Room (splat / PLY scene units)

/// Width, height, and depth in **one consistent scene unit** (same as PLY / splat world space after raycast).
/// Ratios with ``FurnitureSceneSize`` are meaningful without converting to meters.
struct RoomRaycastDimensions: Codable, Equatable, Sendable {
    var width: Float
    var height: Float
    var depth: Float
}

/// Backward-friendly name used in the product spec.
typealias RoomMeasurement = RoomRaycastDimensions

// MARK: - Furniture (live camera, scene-relative units)

/// Furniture extent in the **same unit system** as ``RoomRaycastDimensions`` when depth is consistent.
struct FurnitureSceneSize: Equatable, Sendable {
    var width: Float
    var height: Float
    var depth: Float
}

// MARK: - Monocular furniture sizing (pinhole + depth)

/// Estimates furniture size from a 2D bbox and a **center depth** (meters or any unit, as long as room uses the same).
enum FurnitureMonocularMeasurer {

    /// Horizontal field of view in degrees → focal length in pixels (pinhole).
    static func focalLengthPixels(horizontalFieldOfViewDegrees: Float, imageWidth: Float) -> Float {
        guard horizontalFieldOfViewDegrees > 0.5, horizontalFieldOfViewDegrees < 179, imageWidth > 1 else { return 0 }
        let fov = horizontalFieldOfViewDegrees * (.pi / 180)
        return imageWidth / (2 * tanf(fov * 0.5))
    }

    /// Stand-in distance (meters) when there is no LiDAR / AR frame depth: scales ``roomDepthMeters`` by bbox vertical position.
    /// `midYTop` is bbox center Y ÷ image height with origin at the **top** (~0…1). Lower on screen (larger value) usually
    /// means floor furniture farther along the floor than a tabletop object high in frame — a single `roomDepth × 0.45` for
    /// everything cannot tell near vs far, which skews tall far chairs vs close glass.
    static func roomDepthProxyMeters(roomDepthMeters: Float, bboxCenterYNormalizedFromTop midYTop: Float) -> Float {
        guard roomDepthMeters > 0.01 else { return 0.5 }
        let y = min(max(midYTop, 0.08), 0.94)
        // Legacy flat factor was ~0.45 × room; match that when the bbox sits near mid-frame (y ≈ 0.5).
        let frac = 0.23 + 0.44 * y
        return roomDepthMeters * frac
    }

    /// `bbox` in **Vision** normalized space: origin bottom-left, Y up (same as `VNBoundingBox`).
    static func estimateSize(
        bbox: CGRect,
        imageWidth: Int,
        imageHeight: Int,
        focalLengthPixels: Float,
        centerDepth: Float
    ) -> FurnitureSceneSize? {
        guard imageWidth > 1, imageHeight > 1, focalLengthPixels > 1, centerDepth > 0.001 else { return nil }

        let pxW = Float(bbox.width) * Float(imageWidth)
        let pxH = Float(bbox.height) * Float(imageHeight)
        let w = (pxW * centerDepth) / focalLengthPixels
        let h = (pxH * centerDepth) / focalLengthPixels

        let nx = Float(bbox.midX)
        let nyTop = Float(bbox.maxY)
        let nyBot = Float(bbox.minY)
        let dTop = sampleNormalizedDepthProxy(nx: nx, ny: nyTop)
        let dBot = sampleNormalizedDepthProxy(nx: nx, ny: nyBot)
        let depthExtent = abs(dTop - dBot) * centerDepth

        return FurnitureSceneSize(width: w, height: h, depth: max(depthExtent, w * 0.15))
    }

    /// Maps furniture size in **meters** (pinhole + depth) into **scene units** matching ``RoomRaycastDimensions``,
    /// using per-axis scale `roomRaycast / roomMeters` from the Sharp nav / YOLO room box.
    static func furnitureMetersMappedToRaycastSceneUnits(
        furnitureMeters: FurnitureSceneSize,
        roomMetersWidth: Float,
        roomMetersHeight: Float,
        roomMetersDepth: Float,
        roomRaycastScene: RoomRaycastDimensions
    ) -> FurnitureSceneSize? {
        guard roomMetersWidth > 1e-5, roomMetersHeight > 1e-5, roomMetersDepth > 1e-5 else { return nil }
        let scaleW = roomRaycastScene.width / roomMetersWidth
        let scaleH = roomRaycastScene.height / roomMetersHeight
        let scaleD = roomRaycastScene.depth / roomMetersDepth
        return FurnitureSceneSize(
            width: furnitureMeters.width * scaleW,
            height: furnitureMeters.height * scaleH,
            depth: furnitureMeters.depth * scaleD
        )
    }

    /// Placeholder when no real per-pixel depth map is wired: weak vertical gradient so depth extent is non-zero.
    private static func sampleNormalizedDepthProxy(nx: Float, ny: Float) -> Float {
        0.5 + (ny - 0.5) * 0.08 + (nx - 0.5) * 0.02
    }
}

// MARK: - Fitment (pure ratios)

enum FitmentCheck {
    enum FitResult: Equatable {
        case fits
        case tooWide(ratio: Float)
        case tooTall(ratio: Float)
        case tooDeep(ratio: Float)
    }

    static func check(furniture: FurnitureSceneSize, room: RoomRaycastDimensions) -> [FitResult] {
        guard room.width > 1e-6, room.height > 1e-6, room.depth > 1e-6 else { return [.fits] }

        let widthRatio = furniture.width / room.width
        let heightRatio = furniture.height / room.height
        let depthRatio = furniture.depth / room.depth

        logDebug("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        logDebug("📐 [Fitment] Ratios vs raycast room (furniture ÷ room, same scene-unit space)")
        logDebug("📐 [Fitment] width ratio  \(String(format: "%.3f", widthRatio)) (\(String(format: "%.0f", widthRatio * 100))%)")
        logDebug("📐 [Fitment] height ratio \(String(format: "%.3f", heightRatio)) (\(String(format: "%.0f", heightRatio * 100))%)")
        logDebug("📐 [Fitment] depth ratio  \(String(format: "%.3f", depthRatio)) (\(String(format: "%.0f", depthRatio * 100))%) — furniture depth is a thickness proxy (pinhole + bbox), often ≪ room depth span; low % is normal")

        var results: [FitResult] = []
        if widthRatio > 1.0 {
            results.append(.tooWide(ratio: widthRatio))
            logDebug("❌ [Fitment] Too wide — \(String(format: "%.0f%%", (widthRatio - 1) * 100)) over room width")
        }
        if heightRatio > 1.0 {
            results.append(.tooTall(ratio: heightRatio))
            logDebug("❌ [Fitment] Too tall — \(String(format: "%.0f%%", (heightRatio - 1) * 100)) over room height")
        }
        if depthRatio > 1.0 {
            results.append(.tooDeep(ratio: depthRatio))
            logDebug("❌ [Fitment] Too deep — \(String(format: "%.0f%%", (depthRatio - 1) * 100)) over room depth")
        }
        if results.isEmpty {
            results.append(.fits)
            logDebug("✅ [Fitment] Within room extents (ratio space).")
        }
        logDebug("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        return results
    }
}

// MARK: - Overlay scale (room view)

struct OverlayScale: Equatable {
    var scaleX: Float
    var scaleY: Float

    func overlaySize(roomViewWidth: Float, roomViewHeight: Float) -> CGSize {
        CGSize(width: CGFloat(scaleX * roomViewWidth), height: CGFloat(scaleY * roomViewHeight))
    }

    /// Furniture ÷ room in **scene-unit** space (no log).
    static func ratios(furniture: FurnitureSceneSize, room: RoomRaycastDimensions) -> OverlayScale {
        guard room.width > 1e-6, room.height > 1e-6 else {
            return OverlayScale(scaleX: 0, scaleY: 0)
        }
        return OverlayScale(
            scaleX: furniture.width / room.width,
            scaleY: furniture.height / room.height
        )
    }

    static func compute(furniture: FurnitureSceneSize, room: RoomRaycastDimensions) -> OverlayScale {
        let r = ratios(furniture: furniture, room: room)
        logDebug("📐 [Overlay] scale \(String(format: "%.3f", r.scaleX)) × \(String(format: "%.3f", r.scaleY))")
        return r
    }
}
