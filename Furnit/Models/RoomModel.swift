import Foundation
import simd

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

public struct DetectedPlane: Equatable, Sendable {
    public enum PlaneType: String, Codable, Sendable {
        case floor
        case ceiling
        case wall
        case unknown
    }

    public let normal: SIMD3<Float>
    public let centroid: SIMD3<Float>
    public let d: Float
    public let type: PlaneType
    public let confidence: Float
    public let bounds: AABB3

    public init(
        normal: SIMD3<Float>,
        centroid: SIMD3<Float>,
        d: Float,
        type: PlaneType,
        confidence: Float,
        bounds: AABB3
    ) {
        self.normal = normal
        self.centroid = centroid
        self.d = d
        self.type = type
        self.confidence = confidence
        self.bounds = bounds
    }

    public func distance(to point: SIMD3<Float>) -> Float {
        dot(normal, point) + d
    }

    public func project(_ point: SIMD3<Float>) -> SIMD3<Float> {
        point - distance(to: point) * normal
    }
}

public struct FloorUVBounds: Equatable, Sendable {
    public let min: SIMD2<Float>
    public let max: SIMD2<Float>

    public init(min: SIMD2<Float>, max: SIMD2<Float>) {
        self.min = min
        self.max = max
    }
}

public struct FreeFloorRegion: Equatable, Sendable {
    public let polygon: [SIMD2<Float>]
    public let areaSqM: Float
    public let uvBounds: FloorUVBounds

    public init(polygon: [SIMD2<Float>], areaSqM: Float, uvBounds: FloorUVBounds) {
        self.polygon = polygon
        self.areaSqM = areaSqM
        self.uvBounds = uvBounds
    }

    public func fits(width: Float, depth: Float) -> Bool {
        let regionWidth = uvBounds.max.x - uvBounds.min.x
        let regionDepth = uvBounds.max.y - uvBounds.min.y
        return (width <= regionWidth && depth <= regionDepth) ||
            (depth <= regionWidth && width <= regionDepth)
    }
}

public struct RoomCorner: Equatable, Sendable {
    public let position: SIMD3<Float>
    public let wallIndexA: Int
    public let wallIndexB: Int
    public let bisector: SIMD3<Float>

    public init(position: SIMD3<Float>, wallIndexA: Int, wallIndexB: Int, bisector: SIMD3<Float>) {
        self.position = position
        self.wallIndexA = wallIndexA
        self.wallIndexB = wallIndexB
        self.bisector = bisector
    }
}

public struct SurfacePalette: Equatable, Sendable {
    public enum MaterialHint: String, Codable, Sendable {
        case wood
        case fabric
        case concrete
        case tile
        case plaster
        case carpet
        case glass
        case metal
        case unknown
    }

    public struct SurfaceColors: Equatable, Sendable {
        public let dominantColors: [SIMD3<Float>]
        public let materialHints: [MaterialHint]

        public init(dominantColors: [SIMD3<Float>], materialHints: [MaterialHint]) {
            self.dominantColors = dominantColors
            self.materialHints = materialHints
        }
    }

    public let floor: SurfaceColors?
    public let walls: SurfaceColors?
    public let ceiling: SurfaceColors?

    public init(
        floor: SurfaceColors?,
        walls: SurfaceColors?,
        ceiling: SurfaceColors?
    ) {
        self.floor = floor
        self.walls = walls
        self.ceiling = ceiling
    }
}

public struct SourceCameraInfo: Equatable, Sendable {
    public let focalLengthMM: Float?
    public let sensorWidthMM: Float?
    public let focalLength35mmEquivalentMM: Float?
    public let subjectDistanceMeters: Float?
    public let imageWidthPx: Int?
    public let imageHeightPx: Int?
    public let photoOrientation: Int?

    public init(
        focalLengthMM: Float?,
        sensorWidthMM: Float?,
        focalLength35mmEquivalentMM: Float?,
        subjectDistanceMeters: Float?,
        imageWidthPx: Int?,
        imageHeightPx: Int?,
        photoOrientation: Int?
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

public struct RoomModel: Equatable, Sendable {
    public let floorPlane: DetectedPlane?
    public let ceilingPlane: DetectedPlane?
    public let wallPlanes: [DetectedPlane]
    public let corners: [RoomCorner]
    public let freeFloorRegions: [FreeFloorRegion]
    public let roomBounds: AABB3
    public let sceneToMeters: Float
    public let surfacePalette: SurfacePalette
    public let cameraInfo: SourceCameraInfo
    public let extractedAt: Date

    public init(
        floorPlane: DetectedPlane?,
        ceilingPlane: DetectedPlane?,
        wallPlanes: [DetectedPlane],
        corners: [RoomCorner],
        freeFloorRegions: [FreeFloorRegion],
        roomBounds: AABB3,
        sceneToMeters: Float,
        surfacePalette: SurfacePalette,
        cameraInfo: SourceCameraInfo,
        extractedAt: Date
    ) {
        self.floorPlane = floorPlane
        self.ceilingPlane = ceilingPlane
        self.wallPlanes = wallPlanes
        self.corners = corners
        self.freeFloorRegions = freeFloorRegions
        self.roomBounds = roomBounds
        self.sceneToMeters = sceneToMeters
        self.surfacePalette = surfacePalette
        self.cameraInfo = cameraInfo
        self.extractedAt = extractedAt
    }

    public func toMeters(_ sceneLength: Float) -> Float {
        sceneLength * sceneToMeters
    }

    public func toScene(_ meters: Float) -> Float {
        meters / max(sceneToMeters, 0.0001)
    }

    public var centroid: SIMD3<Float> { roomBounds.center }
    public var widthMeters: Float { toMeters(roomBounds.size.x) }
    public var depthMeters: Float { toMeters(roomBounds.size.z) }
    public var heightMeters: Float {
        if let floorPlane, let ceilingPlane {
            return toMeters(abs(ceilingPlane.distance(to: floorPlane.centroid)))
        }
        return toMeters(roomBounds.size.y)
    }
}
