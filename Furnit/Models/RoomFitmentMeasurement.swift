// RoomFitmentMeasurement.swift
// Furnit
//
// Downstream consumers of room dimensions for furniture sizing.
// RoomModel (defined in RoomModel.swift) is visible via same-module access — no import needed.

import Foundation
import CoreGraphics
import simd

// MARK: - RoomRaycastDimensions

/// Raw room dimensions in scene units (as measured by raycasting or AABB bounds).
struct RoomRaycastDimensions {
    var width: Float
    var height: Float
    var depth: Float
    /// Metres per scene unit (from ``RoomModel`` / splat calibration when known).
    var sceneToMeters: Float = 1.0

    init(width: Float, height: Float, depth: Float, sceneToMeters: Float = 1.0) {
        self.width = width
        self.height = height
        self.depth = depth
        self.sceneToMeters = sceneToMeters
    }
}

// MARK: - RoomMeasurement

typealias RoomMeasurement = RoomRaycastDimensions

// MARK: - FurnitureSceneSize

/// Furniture size expressed in scene units (or meters, depending on context).
struct FurnitureSceneSize {
    var width: Float
    var height: Float
    var depth: Float
}

// MARK: - FurnitureMonocularMeasurer

enum FurnitureMonocularMeasurer {

    /// Estimates the focal length in pixels from the image width and a typical horizontal FoV.
    /// - Parameters:
    ///   - imageWidth: Width of the captured image in pixels.
    ///   - hFovDegrees: Horizontal field of view in degrees (default 60°).
    /// - Returns: Focal length in pixels.
    static func focalLengthPixels(imageWidth: Float, hFovDegrees: Float = 60.0) -> Float {
        let hFovRad = hFovDegrees * (.pi / 180.0)
        return (imageWidth / 2.0) / tan(hFovRad / 2.0)
    }

    /// Produces a rough room-depth proxy in meters from scene-unit room dimensions.
    /// Uses the Z dimension (depth) scaled by `sceneToMeters`.
    /// - Parameters:
    ///   - room: Room dimensions in scene units.
    ///   - sceneToMeters: Conversion factor (meters per scene unit).
    /// - Returns: Depth proxy in meters.
    static func roomDepthProxyMeters(room: RoomRaycastDimensions, sceneToMeters: Float) -> Float {
        return room.depth * sceneToMeters
    }

    /// Estimates furniture size in scene units from a 2-D bounding box and monocular depth.
    /// - Parameters:
    ///   - boundingBoxNormalized: Normalized bounding box `(x, y, width, height)` in [0, 1].
    ///   - imageSize: Full image size in pixels.
    ///   - depthMeters: Estimated distance to the furniture in meters.
    ///   - sceneToMeters: Conversion factor (meters per scene unit).
    /// - Returns: Estimated `FurnitureSceneSize` in scene units.
    static func estimateSize(
        boundingBoxNormalized: CGRect,
        imageSize: CGSize,
        depthMeters: Float,
        sceneToMeters: Float
    ) -> FurnitureSceneSize {
        let focalPx = focalLengthPixels(imageWidth: Float(imageSize.width))
        let widthPx  = Float(boundingBoxNormalized.width)  * Float(imageSize.width)
        let heightPx = Float(boundingBoxNormalized.height) * Float(imageSize.height)

        guard focalPx > 0, sceneToMeters > 0.0001 else {
            logDebug("FurnitureMonocularMeasurer.estimateSize: invalid focal or scale, returning zeros")
            return FurnitureSceneSize(width: 0, height: 0, depth: 0)
        }

        let widthM  = (widthPx  * depthMeters) / focalPx
        let heightM = (heightPx * depthMeters) / focalPx
        // Depth of furniture approximated as average of width/height
        let depthM  = (widthM + heightM) / 2.0

        let toSU: (Float) -> Float = { $0 / sceneToMeters }
        return FurnitureSceneSize(width: toSU(widthM), height: toSU(heightM), depth: toSU(depthM))
    }

    /// Maps furniture meters directly to raycast scene units using the provided scale.
    /// - Parameters:
    ///   - furnitureMeters: Furniture dimensions in real-world meters.
    ///   - sceneToMeters: Conversion factor (meters per scene unit).
    /// - Returns: Furniture size expressed in scene units.
    static func furnitureMetersMappedToRaycastSceneUnits(
        furnitureMeters: FurnitureSceneSize,
        sceneToMeters: Float
    ) -> FurnitureSceneSize {
        guard sceneToMeters > 0.0001 else {
            logDebug("FurnitureMonocularMeasurer.furnitureMetersMappedToRaycastSceneUnits: sceneToMeters too small, returning input unchanged")
            return furnitureMeters
        }
        return FurnitureSceneSize(
            width:  furnitureMeters.width  / sceneToMeters,
            height: furnitureMeters.height / sceneToMeters,
            depth:  furnitureMeters.depth  / sceneToMeters
        )
    }

    /// Returns a normalized depth proxy in [0, 1] relative to the room depth.
    /// - Parameters:
    ///   - depthMeters: Absolute depth estimate in meters.
    ///   - roomDepthMeters: Total room depth in meters.
    /// - Returns: Clamped ratio of depth to room depth.
    static func sampleNormalizedDepthProxy(depthMeters: Float, roomDepthMeters: Float) -> Float {
        guard roomDepthMeters > 0.0001 else {
            logDebug("FurnitureMonocularMeasurer.sampleNormalizedDepthProxy: roomDepthMeters too small")
            return 0.0
        }
        return min(max(depthMeters / roomDepthMeters, 0.0), 1.0)
    }
}

// MARK: - FitmentCheck

enum FitmentCheck {

    /// Describes how well a piece of furniture fits along one room axis.
    enum FitResult {
        /// Furniture comfortably fits (ratio ≤ 0.7).
        case fits
        /// Furniture is a tight fit (ratio in (0.7, 0.9]).
        case tight
        /// Furniture does not fit (ratio > 0.9 or room dimension is zero).
        case doesNotFit
    }

    /// Checks whether `furniture` fits inside `room` along width, height, and depth.
    /// - Parameters:
    ///   - furniture: Furniture dimensions in scene units.
    ///   - room: Room dimensions in scene units.
    /// - Returns: Array of three `FitResult` values `[widthFit, heightFit, depthFit]`.
    static func check(
        furniture: FurnitureSceneSize,
        room: RoomRaycastDimensions
    ) -> [FitResult] {
        func evaluate(furnitureDim: Float, roomDim: Float, axis: String) -> FitResult {
            guard roomDim > 0.0001 else {
                logDebug("FitmentCheck.check: room \(axis) is near-zero, reporting doesNotFit")
                return .doesNotFit
            }
            let ratio = furnitureDim / roomDim
            logDebug("FitmentCheck.check: \(axis) ratio = \(ratio) (furniture \(furnitureDim) / room \(roomDim))")
            switch ratio {
            case ...0.7:         return .fits
            case 0.7...0.9:     return .tight
            default:             return .doesNotFit
            }
        }

        return [
            evaluate(furnitureDim: furniture.width,  roomDim: room.width,  axis: "width"),
            evaluate(furnitureDim: furniture.height, roomDim: room.height, axis: "height"),
            evaluate(furnitureDim: furniture.depth,  roomDim: room.depth,  axis: "depth")
        ]
    }
}

// MARK: - OverlayScale

/// Proportional overlay ratios for projecting furniture onto a room image.
struct OverlayScale {
    /// Width ratio: furniture width / room width.
    var widthRatio: Float
    /// Height ratio: furniture height / room height.
    var heightRatio: Float
    /// Depth ratio: furniture depth / room depth.
    var depthRatio: Float

    /// Computes overlay ratios from furniture and room scene-unit dimensions.
    /// - Parameters:
    ///   - furniture: Furniture dimensions in scene units.
    ///   - room: Room dimensions in scene units.
    /// - Returns: `OverlayScale` with clamped [0, 1] ratios.
    static func compute(
        furniture: FurnitureSceneSize,
        room: RoomRaycastDimensions
    ) -> OverlayScale {
        func ratio(_ f: Float, _ r: Float, axis: String) -> Float {
            guard r > 0.0001 else {
                logDebug("OverlayScale.compute: room \(axis) near-zero, ratio clamped to 0")
                return 0.0
            }
            let v = f / r
            logDebug("OverlayScale.compute: \(axis) ratio = \(v)")
            return min(max(v, 0.0), 1.0)
        }
        return OverlayScale(
            widthRatio:  ratio(furniture.width,  room.width,  axis: "width"),
            heightRatio: ratio(furniture.height, room.height, axis: "height"),
            depthRatio:  ratio(furniture.depth,  room.depth,  axis: "depth")
        )
    }

    /// Scales a canvas size by the overlay ratios.
    /// - Parameter canvasSize: The full canvas (room image) size in points/pixels.
    /// - Returns: Projected furniture size on the canvas.
    func overlaySize(for canvasSize: CGSize) -> CGSize {
        return CGSize(
            width:  CGFloat(widthRatio)  * canvasSize.width,
            height: CGFloat(heightRatio) * canvasSize.height
        )
    }
}

// MARK: - Internal logging
// Uses the global `logDebug` from `Utilities/Logger.swift`, which is gated by
// Settings → Debug Mode. The previous local shadow always logged in DEBUG builds
// regardless of the user toggle, which contributed to noise.

// MARK: - RoomModel Extensions
// Extensions below integrate with RoomModel (same module — no import required).

// MARK: RoomRaycastDimensions + RoomModel

extension RoomRaycastDimensions {
    /// Create from a ``RoomModel``'s AABB bounds (scene units).
    init(roomModel: RoomModel) {
        self.width = roomModel.roomBounds.size.x
        self.height = roomModel.roomBounds.size.y
        self.depth = roomModel.roomBounds.size.z
        self.sceneToMeters = roomModel.sceneToMeters
    }
}

// MARK: FitmentCheck + RoomModel

extension FitmentCheck {
    /// Fitment using ``RoomModel`` metric dimensions directly.
    /// Converts furniture meters → scene units via `roomModel.toScene()`, then checks ratios.
    static func check(
        furnitureMeters: FurnitureSceneSize,
        roomModel: RoomModel
    ) -> [FitResult] {
        let roomDims = RoomRaycastDimensions(roomModel: roomModel)
        let s = max(roomModel.sceneToMeters, 0.0001)
        let furnitureSU = FurnitureSceneSize(
            width: furnitureMeters.width / s,
            height: furnitureMeters.height / s,
            depth: furnitureMeters.depth / s
        )
        return check(furniture: furnitureSU, room: roomDims)
    }
}

// MARK: OverlayScale + RoomModel

extension OverlayScale {
    /// Overlay ratios from ``RoomModel`` (furniture in meters).
    static func compute(
        furnitureMeters: FurnitureSceneSize,
        roomModel: RoomModel
    ) -> OverlayScale {
        let roomDims = RoomRaycastDimensions(roomModel: roomModel)
        let s = max(roomModel.sceneToMeters, 0.0001)
        let furnitureSU = FurnitureSceneSize(
            width: furnitureMeters.width / s,
            height: furnitureMeters.height / s,
            depth: furnitureMeters.depth / s
        )
        return compute(furniture: furnitureSU, room: roomDims)
    }
}

// MARK: FurnitureMonocularMeasurer + RoomModel

extension FurnitureMonocularMeasurer {
    /// When ``RoomModel`` has a calibrated `sceneToMeters`, convert pinhole depth estimate
    /// to scene units and back to meters for better accuracy than raw `roomDepthMeters` proxy.
    static func calibratedDepthMeters(
        rawDepthMeters: Float,
        roomModel: RoomModel?
    ) -> Float {
        guard let rm = roomModel,
              rm.sceneToMeters > 0.0001,
              rm.sceneToMeters.isFinite else {
            return rawDepthMeters
        }
        // Clamp to room depth bounds in meters
        let maxDepthMeters = rm.depthMeters
        return min(max(rawDepthMeters, 0.2), max(maxDepthMeters, 0.5))
    }
}
