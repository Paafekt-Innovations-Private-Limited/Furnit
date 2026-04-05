import Foundation
import simd

// MARK: - AABB3

public struct AABB3: Equatable, Sendable {
    public let min: SIMD3<Float>
    public let max: SIMD3<Float>

    public init(min: SIMD3<Float>, max: SIMD3<Float>) {
        self.min = min
        self.max = max
    }

    public var center: SIMD3<Float> { (min + max) * 0.5 }
    public var size: SIMD3<Float> { max - min }

    public func contains(_ point: SIMD3<Float>) -> Bool {
        point.x >= min.x && point.x <= max.x &&
            point.y >= min.y && point.y <= max.y &&
            point.z >= min.z && point.z <= max.z
    }
}

// MARK: - DetectedPlane

public struct DetectedPlane: Equatable, Sendable {
    public enum PlaneType: Equatable, Sendable {
        case floor
        case ceiling
        case wall
    }

    public let type: PlaneType
    public let normal: SIMD3<Float>
    public let pointOnPlane: SIMD3<Float>

    public init(type: PlaneType, normal: SIMD3<Float>, pointOnPlane: SIMD3<Float>) {
        self.type = type
        self.normal = normal
        self.pointOnPlane = pointOnPlane
    }

    public func distance(to point: SIMD3<Float>) -> Float {
        simd_dot(normal, point - pointOnPlane)
    }

    public func project(_ point: SIMD3<Float>) -> SIMD3<Float> {
        point - distance(to: point) * normal
    }
}

// MARK: - FloorUVBounds

public struct FloorUVBounds: Equatable, Sendable {
    public let min: SIMD2<Float>
    public let max: SIMD2<Float>

    public init(min: SIMD2<Float>, max: SIMD2<Float>) {
        self.min = min
        self.max = max
    }
}

// MARK: - FreeFloorRegion

public struct FreeFloorRegion: Equatable, Sendable {
    public let polygon: [SIMD2<Float>]
    public let areaSqM: Float
    public let uvBounds: FloorUVBounds
    /// Fraction of the floor AABB occupied by obstacles (0 = all free, 1 = fully occupied).
    /// `nil` when computed via legacy AABB-only path.
    public let occupancyRatio: Float?

    public init(polygon: [SIMD2<Float>], areaSqM: Float, uvBounds: FloorUVBounds, occupancyRatio: Float? = nil) {
        self.polygon = polygon
        self.areaSqM = areaSqM
        self.uvBounds = uvBounds
        self.occupancyRatio = occupancyRatio
    }

    public func fits(width: Float, depth: Float) -> Bool {
        let regionWidth = uvBounds.max.x - uvBounds.min.x
        let regionDepth = uvBounds.max.y - uvBounds.min.y
        return (width <= regionWidth && depth <= regionDepth) ||
            (depth <= regionWidth && width <= regionDepth)
    }
}

// MARK: - RoomCorner

public struct RoomCorner: Equatable, Sendable {
    public let position: SIMD3<Float>
    public let uv: SIMD2<Float>

    public init(position: SIMD3<Float>, uv: SIMD2<Float>) {
        self.position = position
        self.uv = uv
    }
}

// MARK: - SurfacePalette

public struct SurfacePalette: Equatable, Sendable {
    public enum MaterialHint: String, Equatable, Sendable {
        case wood
        case tile
        case carpet
        case concrete
        case brick
        case plaster
        case marble
        case unknown
    }

    public struct SurfaceColors: Equatable, Sendable {
        public let primary: SIMD3<Float>
        public let secondary: SIMD3<Float>?
        public let hint: MaterialHint

        public init(primary: SIMD3<Float>, secondary: SIMD3<Float>? = nil, hint: MaterialHint = .unknown) {
            self.primary = primary
            self.secondary = secondary
            self.hint = hint
        }
    }

    public let floor: SurfaceColors?
    public let walls: SurfaceColors?
    public let ceiling: SurfaceColors?

    public init(floor: SurfaceColors?, walls: SurfaceColors?, ceiling: SurfaceColors?) {
        self.floor = floor
        self.walls = walls
        self.ceiling = ceiling
    }

    /// A palette with no surface colour information.
    public static let empty = SurfacePalette(floor: nil, walls: nil, ceiling: nil)
}

public extension SurfacePalette.SurfaceColors {
    /// Primary and secondary swatches for heuristics that expect a small color set.
    var dominantColors: [SIMD3<Float>] {
        if let secondary {
            return [primary, secondary]
        }
        return [primary]
    }
}

// MARK: - SourceCameraInfo

/// Optional EXIF / sidecar fields used for wall measurement, FurnitureFit pinhole sizing, and geometry fallbacks.
public struct SourceCameraInfo: Equatable, Sendable {
    public let focalLengthMM: Float?
    public let sensorWidthMM: Float?
    public let focalLength35mmEquivalentMM: Float?
    public let subjectDistanceMeters: Float?
    public let imageWidthPx: Int?
    public let imageHeightPx: Int?
    public let photoOrientation: Int?

    public init(
        focalLengthMM: Float? = nil,
        sensorWidthMM: Float? = nil,
        focalLength35mmEquivalentMM: Float? = nil,
        subjectDistanceMeters: Float? = nil,
        imageWidthPx: Int? = nil,
        imageHeightPx: Int? = nil,
        photoOrientation: Int? = nil
    ) {
        self.focalLengthMM = focalLengthMM
        self.sensorWidthMM = sensorWidthMM
        self.focalLength35mmEquivalentMM = focalLength35mmEquivalentMM
        self.subjectDistanceMeters = subjectDistanceMeters
        self.imageWidthPx = imageWidthPx
        self.imageHeightPx = imageHeightPx
        self.photoOrientation = photoOrientation
    }
}

// MARK: - RoomModel

public struct RoomModel: Equatable, Sendable {
    public let aabb: AABB3
    public let floor: DetectedPlane
    public let ceiling: DetectedPlane?
    public let walls: [DetectedPlane]
    public let corners: [RoomCorner]
    public let freeFloorRegions: [FreeFloorRegion]
    public let surfacePalette: SurfacePalette
    public let cameraInfo: SourceCameraInfo?
    /// Metres per one unit of scene-space AABB / splat coordinates (see ``RoomGeometryEngine``).
    public let sceneToMeters: Float

    public init(
        aabb: AABB3,
        floor: DetectedPlane,
        ceiling: DetectedPlane?,
        walls: [DetectedPlane],
        corners: [RoomCorner],
        freeFloorRegions: [FreeFloorRegion],
        surfacePalette: SurfacePalette,
        cameraInfo: SourceCameraInfo?,
        sceneToMeters: Float
    ) {
        self.aabb = aabb
        self.floor = floor
        self.ceiling = ceiling
        self.walls = walls
        self.corners = corners
        self.freeFloorRegions = freeFloorRegions
        self.surfacePalette = surfacePalette
        self.cameraInfo = cameraInfo
        self.sceneToMeters = sceneToMeters
    }

    /// Axis-aligned bounds in scene units (same space as ``aabb``).
    public var roomBounds: AABB3 { aabb }

    /// Returns a copy of the model with dimensions converted from centimetres to metres.
    public func toMeters() -> RoomModel {
        let coordScale: Float = 0.01
        let stmScale = 1.0 / coordScale
        func scaleAABB(_ b: AABB3) -> AABB3 { AABB3(min: b.min * coordScale, max: b.max * coordScale) }
        func scalePlane(_ p: DetectedPlane) -> DetectedPlane {
            DetectedPlane(type: p.type, normal: p.normal, pointOnPlane: p.pointOnPlane * coordScale)
        }
        func scaleCorner(_ c: RoomCorner) -> RoomCorner {
            RoomCorner(position: c.position * coordScale, uv: c.uv * coordScale)
        }
        func scaleRegion(_ r: FreeFloorRegion) -> FreeFloorRegion {
            let scaledPolygon = r.polygon.map { $0 * coordScale }
            let scaledBounds = FloorUVBounds(
                min: r.uvBounds.min * coordScale,
                max: r.uvBounds.max * coordScale
            )
            return FreeFloorRegion(
                polygon: scaledPolygon,
                areaSqM: r.areaSqM * coordScale * coordScale,
                uvBounds: scaledBounds,
                occupancyRatio: r.occupancyRatio
            )
        }
        return RoomModel(
            aabb: scaleAABB(aabb),
            floor: scalePlane(floor),
            ceiling: ceiling.map(scalePlane),
            walls: walls.map(scalePlane),
            corners: corners.map(scaleCorner),
            freeFloorRegions: freeFloorRegions.map(scaleRegion),
            surfacePalette: surfacePalette,
            cameraInfo: cameraInfo,
            sceneToMeters: sceneToMeters * stmScale
        )
    }

    /// Returns a copy of the model with dimensions converted from metres to scene units (centimetres).
    public func toScene() -> RoomModel {
        let coordScale: Float = 100.0
        let stmScale = 1.0 / coordScale
        func scaleAABB(_ b: AABB3) -> AABB3 { AABB3(min: b.min * coordScale, max: b.max * coordScale) }
        func scalePlane(_ p: DetectedPlane) -> DetectedPlane {
            DetectedPlane(type: p.type, normal: p.normal, pointOnPlane: p.pointOnPlane * coordScale)
        }
        func scaleCorner(_ c: RoomCorner) -> RoomCorner {
            RoomCorner(position: c.position * coordScale, uv: c.uv * coordScale)
        }
        func scaleRegion(_ r: FreeFloorRegion) -> FreeFloorRegion {
            let scaledPolygon = r.polygon.map { $0 * coordScale }
            let scaledBounds = FloorUVBounds(
                min: r.uvBounds.min * coordScale,
                max: r.uvBounds.max * coordScale
            )
            return FreeFloorRegion(
                polygon: scaledPolygon,
                areaSqM: r.areaSqM * coordScale * coordScale,
                uvBounds: scaledBounds,
                occupancyRatio: r.occupancyRatio
            )
        }
        return RoomModel(
            aabb: scaleAABB(aabb),
            floor: scalePlane(floor),
            ceiling: ceiling.map(scalePlane),
            walls: walls.map(scalePlane),
            corners: corners.map(scaleCorner),
            freeFloorRegions: freeFloorRegions.map(scaleRegion),
            surfacePalette: surfacePalette,
            cameraInfo: cameraInfo,
            sceneToMeters: sceneToMeters * stmScale
        )
    }

    public var centroid: SIMD3<Float> { aabb.center }

    /// Vertical span floor ↔ ceiling in scene units when both planes exist (avoids splat bleed on AABB Y).
    public var interiorHeightSceneUnits: Float? {
        guard let ceiling else { return nil }
        let h = abs(ceiling.distance(to: floor.pointOnPlane))
        return h > 0.001 ? h : nil
    }

    /// Planar footprint in scene units: uses **opposing** wall pairs (normals facing each other) when
    /// available; otherwise AABB XZ. Dot threshold matches wall diagnostics (``-0.45``), not a near-perfect
    /// ``-0.85``, so sparse RANSAC walls still yield a width/depth from the gap between them.
    public var interiorFootprintSceneUnits: (width: Float, depth: Float) {
        var widthSU = aabb.size.x
        var depthSU = aabb.size.z
        var bestWidthSpan: Float = 0
        var bestDepthSpan: Float = 0
        /// Normals roughly opposite (same wall pair from two sides).
        let opposingDotThreshold: Float = -0.45
        let minimumUsableSpanSU: Float = 0.05
        for i in 0..<walls.count {
            for j in (i + 1)..<walls.count {
                let w1 = walls[i], w2 = walls[j]
                guard simd_dot(w1.normal, w2.normal) < opposingDotThreshold else { continue }
                let sep = abs(w2.distance(to: w1.pointOnPlane))
                guard sep > 0.001 else { continue }
                let ax = abs(w1.normal.x)
                let az = abs(w1.normal.z)
                guard max(ax, az) > 0.05 else { continue }
                if ax >= az {
                    bestWidthSpan = max(bestWidthSpan, sep)
                } else {
                    bestDepthSpan = max(bestDepthSpan, sep)
                }
            }
        }
        if bestWidthSpan > minimumUsableSpanSU { widthSU = bestWidthSpan }
        if bestDepthSpan > minimumUsableSpanSU { depthSU = bestDepthSpan }
        return (widthSU, depthSU)
    }

    /// W×H×D in scene units aligned with plane interior (height from floor–ceiling).
    public var planeAwareSceneExtent: (width: Float, height: Float, depth: Float) {
        let fp = interiorFootprintSceneUnits
        let h = interiorHeightSceneUnits ?? aabb.size.y
        return (fp.width, h, fp.depth)
    }

    public var widthMeters: Float { interiorFootprintSceneUnits.width * sceneToMeters }
    public var depthMeters: Float { interiorFootprintSceneUnits.depth * sceneToMeters }

    public var heightMeters: Float {
        if let hSU = interiorHeightSceneUnits {
            return hSU * sceneToMeters
        }
        return aabb.size.y * sceneToMeters
    }

    /// Converts a metric depth into scene units using ``sceneToMeters``.
    public func toScene(depthMeters: Float) -> Float {
        depthMeters / max(sceneToMeters, 0.0001)
    }

    public func withSceneToMeters(_ newSceneToMeters: Float) -> RoomModel {
        RoomModel(
            aabb: aabb,
            floor: floor,
            ceiling: ceiling,
            walls: walls,
            corners: corners,
            freeFloorRegions: freeFloorRegions,
            surfacePalette: surfacePalette,
            cameraInfo: cameraInfo,
            sceneToMeters: newSceneToMeters
        )
    }
}
