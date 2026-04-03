import Foundation
import simd

public struct CodableSIMD2: Codable, Equatable, Sendable {
    public let x: Float
    public let y: Float

    public init(_ value: SIMD2<Float>) {
        x = value.x
        y = value.y
    }

    public var value: SIMD2<Float> { SIMD2<Float>(x, y) }
}

public struct CodableSIMD3: Codable, Equatable, Sendable {
    public let x: Float
    public let y: Float
    public let z: Float

    public init(_ value: SIMD3<Float>) {
        x = value.x
        y = value.y
        z = value.z
    }

    public var value: SIMD3<Float> { SIMD3<Float>(x, y, z) }
}

public struct CodablePlane: Codable, Equatable, Sendable {
    public let normal: CodableSIMD3
    public let centroid: CodableSIMD3
    public let d: Float
    public let type: String
    public let confidence: Float
    public let boundsMin: CodableSIMD3
    public let boundsMax: CodableSIMD3

    public init(plane: DetectedPlane) {
        normal = CodableSIMD3(plane.normal)
        centroid = CodableSIMD3(plane.centroid)
        d = plane.d
        type = plane.type.rawValue
        confidence = plane.confidence
        boundsMin = CodableSIMD3(plane.bounds.min)
        boundsMax = CodableSIMD3(plane.bounds.max)
    }

    public func detectedPlane() -> DetectedPlane? {
        guard let type = DetectedPlane.PlaneType(rawValue: type) else { return nil }
        return DetectedPlane(
            normal: normal.value,
            centroid: centroid.value,
            d: d,
            type: type,
            confidence: confidence,
            bounds: AABB3(min: boundsMin.value, max: boundsMax.value)
        )
    }
}

public struct CodableCorner: Codable, Equatable, Sendable {
    public let position: CodableSIMD3
    public let wallIndexA: Int
    public let wallIndexB: Int
    public let bisector: CodableSIMD3

    public init(corner: RoomCorner) {
        position = CodableSIMD3(corner.position)
        wallIndexA = corner.wallIndexA
        wallIndexB = corner.wallIndexB
        bisector = CodableSIMD3(corner.bisector)
    }

    public func roomCorner() -> RoomCorner {
        RoomCorner(
            position: position.value,
            wallIndexA: wallIndexA,
            wallIndexB: wallIndexB,
            bisector: bisector.value
        )
    }
}

public struct CodableFreeRegion: Codable, Equatable, Sendable {
    public let polygon: [CodableSIMD2]
    public let areaSqM: Float
    public let uvBoundsMin: CodableSIMD2
    public let uvBoundsMax: CodableSIMD2

    public init(region: FreeFloorRegion) {
        polygon = region.polygon.map(CodableSIMD2.init)
        areaSqM = region.areaSqM
        uvBoundsMin = CodableSIMD2(region.uvBounds.min)
        uvBoundsMax = CodableSIMD2(region.uvBounds.max)
    }

    public func freeRegion() -> FreeFloorRegion {
        FreeFloorRegion(
            polygon: polygon.map(\.value),
            areaSqM: areaSqM,
            uvBounds: FloorUVBounds(min: uvBoundsMin.value, max: uvBoundsMax.value)
        )
    }
}

public struct CodableSurfacePalette: Codable, Equatable, Sendable {
    public struct SurfaceEntry: Codable, Equatable, Sendable {
        public let colors: [CodableSIMD3]
        public let materialHints: [String]

        public init(_ colors: SurfacePalette.SurfaceColors) {
            self.colors = colors.dominantColors.map(CodableSIMD3.init)
            self.materialHints = colors.materialHints.map(\.rawValue)
        }

        public func surfaceColors() -> SurfacePalette.SurfaceColors {
            SurfacePalette.SurfaceColors(
                dominantColors: colors.map(\.value),
                materialHints: materialHints.compactMap(SurfacePalette.MaterialHint.init(rawValue:))
            )
        }
    }

    public let floor: SurfaceEntry?
    public let walls: SurfaceEntry?
    public let ceiling: SurfaceEntry?

    public init(_ palette: SurfacePalette) {
        floor = palette.floor.map(SurfaceEntry.init)
        walls = palette.walls.map(SurfaceEntry.init)
        ceiling = palette.ceiling.map(SurfaceEntry.init)
    }

    public func surfacePalette() -> SurfacePalette {
        SurfacePalette(
            floor: floor?.surfaceColors(),
            walls: walls?.surfaceColors(),
            ceiling: ceiling?.surfaceColors()
        )
    }
}

public struct EnhancedRoomMetadata: Codable, Equatable, Sendable {
    public let version: Int
    public let roomWidth: Float?
    public let roomHeight: Float?
    public let roomDepth: Float?
    public let roomSceneWidth: Float?
    public let roomSceneHeight: Float?
    public let roomSceneDepth: Float?
    public let photoOrientation: Int?
    public let sceneToMeters: Float?
    public let roomBoundsMin: CodableSIMD3?
    public let roomBoundsMax: CodableSIMD3?
    public let floorPlane: CodablePlane?
    public let ceilingPlane: CodablePlane?
    public let wallPlanes: [CodablePlane]
    public let corners: [CodableCorner]
    public let freeFloorRegions: [CodableFreeRegion]
    public let surfacePalette: CodableSurfacePalette?
    public let cameraFocalLengthMM: Float?
    public let cameraSensorWidthMM: Float?
    public let cameraFocalLength35mmEquivalentMM: Float?
    public let cameraSubjectDistanceMeters: Float?
    public let cameraImageWidthPx: Int?
    public let cameraImageHeightPx: Int?
    public let extractedAt: Date?

    public init(
        version: Int = 2,
        roomWidth: Float?,
        roomHeight: Float?,
        roomDepth: Float?,
        roomSceneWidth: Float?,
        roomSceneHeight: Float?,
        roomSceneDepth: Float?,
        photoOrientation: Int?,
        sceneToMeters: Float?,
        roomBoundsMin: CodableSIMD3?,
        roomBoundsMax: CodableSIMD3?,
        floorPlane: CodablePlane?,
        ceilingPlane: CodablePlane?,
        wallPlanes: [CodablePlane],
        corners: [CodableCorner],
        freeFloorRegions: [CodableFreeRegion],
        surfacePalette: CodableSurfacePalette?,
        cameraFocalLengthMM: Float?,
        cameraSensorWidthMM: Float?,
        cameraFocalLength35mmEquivalentMM: Float?,
        cameraSubjectDistanceMeters: Float?,
        cameraImageWidthPx: Int?,
        cameraImageHeightPx: Int?,
        extractedAt: Date?
    ) {
        self.version = version
        self.roomWidth = roomWidth
        self.roomHeight = roomHeight
        self.roomDepth = roomDepth
        self.roomSceneWidth = roomSceneWidth
        self.roomSceneHeight = roomSceneHeight
        self.roomSceneDepth = roomSceneDepth
        self.photoOrientation = photoOrientation
        self.sceneToMeters = sceneToMeters
        self.roomBoundsMin = roomBoundsMin
        self.roomBoundsMax = roomBoundsMax
        self.floorPlane = floorPlane
        self.ceilingPlane = ceilingPlane
        self.wallPlanes = wallPlanes
        self.corners = corners
        self.freeFloorRegions = freeFloorRegions
        self.surfacePalette = surfacePalette
        self.cameraFocalLengthMM = cameraFocalLengthMM
        self.cameraSensorWidthMM = cameraSensorWidthMM
        self.cameraFocalLength35mmEquivalentMM = cameraFocalLength35mmEquivalentMM
        self.cameraSubjectDistanceMeters = cameraSubjectDistanceMeters
        self.cameraImageWidthPx = cameraImageWidthPx
        self.cameraImageHeightPx = cameraImageHeightPx
        self.extractedAt = extractedAt
    }

    public static func migrate(from legacy: [String: String]) -> EnhancedRoomMetadata {
        let orientation: Int?
        if let raw = legacy["photoOrientation"] {
            switch raw {
            case "portrait": orientation = 6
            case "landscape", "square": orientation = 1
            default: orientation = Int(raw)
            }
        } else {
            orientation = nil
        }

        return EnhancedRoomMetadata(
            roomWidth: legacy["roomWidth"].flatMap(Float.init),
            roomHeight: legacy["roomHeight"].flatMap(Float.init),
            roomDepth: legacy["roomDepth"].flatMap(Float.init),
            roomSceneWidth: legacy["roomSceneWidth"].flatMap(Float.init),
            roomSceneHeight: legacy["roomSceneHeight"].flatMap(Float.init),
            roomSceneDepth: legacy["roomSceneDepth"].flatMap(Float.init),
            photoOrientation: orientation,
            sceneToMeters: nil,
            roomBoundsMin: nil,
            roomBoundsMax: nil,
            floorPlane: nil,
            ceilingPlane: nil,
            wallPlanes: [],
            corners: [],
            freeFloorRegions: [],
            surfacePalette: nil,
            cameraFocalLengthMM: nil,
            cameraSensorWidthMM: nil,
            cameraFocalLength35mmEquivalentMM: nil,
            cameraSubjectDistanceMeters: nil,
            cameraImageWidthPx: nil,
            cameraImageHeightPx: nil,
            extractedAt: nil
        )
    }

    public static func from(
        roomModel: RoomModel,
        preserving legacy: EnhancedRoomMetadata? = nil
    ) -> EnhancedRoomMetadata {
        EnhancedRoomMetadata(
            roomWidth: roomModel.widthMeters,
            roomHeight: roomModel.heightMeters,
            roomDepth: roomModel.depthMeters,
            roomSceneWidth: roomModel.roomBounds.size.x,
            roomSceneHeight: roomModel.roomBounds.size.y,
            roomSceneDepth: roomModel.roomBounds.size.z,
            photoOrientation: legacy?.photoOrientation ?? roomModel.cameraInfo.photoOrientation,
            sceneToMeters: roomModel.sceneToMeters,
            roomBoundsMin: CodableSIMD3(roomModel.roomBounds.min),
            roomBoundsMax: CodableSIMD3(roomModel.roomBounds.max),
            floorPlane: roomModel.floorPlane.map(CodablePlane.init),
            ceilingPlane: roomModel.ceilingPlane.map(CodablePlane.init),
            wallPlanes: roomModel.wallPlanes.map(CodablePlane.init),
            corners: roomModel.corners.map(CodableCorner.init),
            freeFloorRegions: roomModel.freeFloorRegions.map(CodableFreeRegion.init),
            surfacePalette: CodableSurfacePalette(roomModel.surfacePalette),
            cameraFocalLengthMM: roomModel.cameraInfo.focalLengthMM,
            cameraSensorWidthMM: roomModel.cameraInfo.sensorWidthMM,
            cameraFocalLength35mmEquivalentMM: roomModel.cameraInfo.focalLength35mmEquivalentMM,
            cameraSubjectDistanceMeters: roomModel.cameraInfo.subjectDistanceMeters,
            cameraImageWidthPx: roomModel.cameraInfo.imageWidthPx,
            cameraImageHeightPx: roomModel.cameraInfo.imageHeightPx,
            extractedAt: roomModel.extractedAt
        )
    }

    public func roomModel() -> RoomModel? {
        guard let sceneToMeters,
              sceneToMeters.isFinite,
              sceneToMeters > 0.0001,
              let roomBoundsMin,
              let roomBoundsMax else { return nil }

        return RoomModel(
            floorPlane: floorPlane?.detectedPlane(),
            ceilingPlane: ceilingPlane?.detectedPlane(),
            wallPlanes: wallPlanes.compactMap { $0.detectedPlane() },
            corners: corners.map { $0.roomCorner() },
            freeFloorRegions: freeFloorRegions.map { $0.freeRegion() },
            roomBounds: AABB3(min: roomBoundsMin.value, max: roomBoundsMax.value),
            sceneToMeters: sceneToMeters,
            surfacePalette: surfacePalette?.surfacePalette() ?? SurfacePalette(floor: nil, walls: nil, ceiling: nil),
            cameraInfo: SourceCameraInfo(
                focalLengthMM: cameraFocalLengthMM,
                sensorWidthMM: cameraSensorWidthMM,
                focalLength35mmEquivalentMM: cameraFocalLength35mmEquivalentMM,
                subjectDistanceMeters: cameraSubjectDistanceMeters,
                imageWidthPx: cameraImageWidthPx,
                imageHeightPx: cameraImageHeightPx,
                photoOrientation: photoOrientation
            ),
            extractedAt: extractedAt ?? Date()
        )
    }
}
