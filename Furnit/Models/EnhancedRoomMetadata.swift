import Foundation
import simd

// MARK: - CodableSIMD2

public struct CodableSIMD2: Codable, Equatable, Sendable {
    public let x: Float
    public let y: Float

    public init(_ value: SIMD2<Float>) {
        x = value.x
        y = value.y
    }

    public var value: SIMD2<Float> { SIMD2<Float>(x, y) }
}

// MARK: - CodableSIMD3

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

// MARK: - CodablePlane

public struct CodablePlane: Codable, Equatable, Sendable {
    public let type: String
    public let normal: CodableSIMD3
    public let pointOnPlane: CodableSIMD3

    public init(plane: DetectedPlane) {
        switch plane.type {
        case .floor:   type = "floor"
        case .ceiling: type = "ceiling"
        case .wall:    type = "wall"
        }
        normal       = CodableSIMD3(plane.normal)
        pointOnPlane = CodableSIMD3(plane.pointOnPlane)
    }

    public func detectedPlane() -> DetectedPlane {
        let planeType: DetectedPlane.PlaneType
        switch type {
        case "floor":   planeType = .floor
        case "ceiling": planeType = .ceiling
        default:        planeType = .wall
        }
        return DetectedPlane(
            type: planeType,
            normal: normal.value,
            pointOnPlane: pointOnPlane.value
        )
    }
}

// MARK: - CodableCorner

public struct CodableCorner: Codable, Equatable, Sendable {
    public let position: CodableSIMD3
    public let uv: CodableSIMD2

    public init(corner: RoomCorner) {
        position = CodableSIMD3(corner.position)
        uv       = CodableSIMD2(corner.uv)
    }

    public func roomCorner() -> RoomCorner {
        RoomCorner(position: position.value, uv: uv.value)
    }
}

// MARK: - CodableFreeRegion

public struct CodableFreeRegion: Codable, Equatable, Sendable {
    public let polygon: [CodableSIMD2]
    public let areaSqM: Float
    public let uvBoundsMin: CodableSIMD2
    public let uvBoundsMax: CodableSIMD2
    /// `nil` when saved from legacy AABB-only freespace path.
    public let occupancyRatio: Float?

    public init(region: FreeFloorRegion) {
        polygon       = region.polygon.map(CodableSIMD2.init)
        areaSqM       = region.areaSqM
        uvBoundsMin   = CodableSIMD2(region.uvBounds.min)
        uvBoundsMax   = CodableSIMD2(region.uvBounds.max)
        occupancyRatio = region.occupancyRatio
    }

    public func freeRegion() -> FreeFloorRegion {
        FreeFloorRegion(
            polygon: polygon.map(\.value),
            areaSqM: areaSqM,
            uvBounds: FloorUVBounds(min: uvBoundsMin.value, max: uvBoundsMax.value),
            occupancyRatio: occupancyRatio
        )
    }
}

// MARK: - CodableSurfacePalette

public struct CodableSurfacePalette: Codable, Equatable, Sendable {

    public struct SurfaceEntry: Codable, Equatable, Sendable {
        public let primary: CodableSIMD3
        public let secondary: CodableSIMD3?
        public let hint: String

        public init(colors: SurfacePalette.SurfaceColors) {
            primary   = CodableSIMD3(colors.primary)
            secondary = colors.secondary.map(CodableSIMD3.init)
            hint      = colors.hint.rawValue
        }

        public func surfaceColors() -> SurfacePalette.SurfaceColors {
            SurfacePalette.SurfaceColors(
                primary:   primary.value,
                secondary: secondary?.value,
                hint:      SurfacePalette.MaterialHint(rawValue: hint) ?? .unknown
            )
        }
    }

    public let floor: SurfaceEntry?
    public let walls: SurfaceEntry?
    public let ceiling: SurfaceEntry?

    public init(palette: SurfacePalette) {
        floor   = palette.floor.map(SurfaceEntry.init)
        walls   = palette.walls.map(SurfaceEntry.init)
        ceiling = palette.ceiling.map(SurfaceEntry.init)
    }

    public func surfacePalette() -> SurfacePalette {
        SurfacePalette(
            floor:   floor?.surfaceColors(),
            walls:   walls?.surfaceColors(),
            ceiling: ceiling?.surfaceColors()
        )
    }
}

// MARK: - CodableSourceCameraInfo

public struct CodableSourceCameraInfo: Codable, Equatable, Sendable {
    public let focalLengthMM: Float?
    public let sensorWidthMM: Float?
    public let focalLength35mmEquivalentMM: Float?
    public let subjectDistanceMeters: Float?
    public let imageWidthPx: Int?
    public let imageHeightPx: Int?
    public let photoOrientation: Int?

    public init(info: SourceCameraInfo) {
        focalLengthMM = info.focalLengthMM
        sensorWidthMM = info.sensorWidthMM
        focalLength35mmEquivalentMM = info.focalLength35mmEquivalentMM
        subjectDistanceMeters = info.subjectDistanceMeters
        imageWidthPx = info.imageWidthPx
        imageHeightPx = info.imageHeightPx
        photoOrientation = info.photoOrientation
    }

    public func sourceCameraInfo() -> SourceCameraInfo {
        SourceCameraInfo(
            focalLengthMM: focalLengthMM,
            sensorWidthMM: sensorWidthMM,
            focalLength35mmEquivalentMM: focalLength35mmEquivalentMM,
            subjectDistanceMeters: subjectDistanceMeters,
            imageWidthPx: imageWidthPx,
            imageHeightPx: imageHeightPx,
            photoOrientation: photoOrientation
        )
    }
}

// MARK: - EnhancedRoomMetadata

public struct EnhancedRoomMetadata: Codable, Equatable, Sendable {

    // MARK: Schema versioning

    public let schemaVersion: Int
    public static let currentSchemaVersion = 2

    // MARK: Geometry

    public let aabbMin: CodableSIMD3
    public let aabbMax: CodableSIMD3
    public let floor: CodablePlane
    public let ceiling: CodablePlane?
    public let walls: [CodablePlane]
    public let corners: [CodableCorner]
    public let freeFloorRegions: [CodableFreeRegion]

    // MARK: Appearance

    public let surfacePalette: CodableSurfacePalette

    // MARK: Capture metrics

    public let capturedAt: Date
    public let sceneToMeters: Float
    public let sourceCameraInfo: CodableSourceCameraInfo?

    /// User-facing room size (m) persisted for lists / UI — **height × depth only** (width TBD).
    public let displayHeightMeters: Float?
    public let displayDepthMeters: Float?

    fileprivate init(
        schemaVersion: Int,
        aabbMin: CodableSIMD3,
        aabbMax: CodableSIMD3,
        floor: CodablePlane,
        ceiling: CodablePlane?,
        walls: [CodablePlane],
        corners: [CodableCorner],
        freeFloorRegions: [CodableFreeRegion],
        surfacePalette: CodableSurfacePalette,
        capturedAt: Date,
        sceneToMeters: Float,
        sourceCameraInfo: CodableSourceCameraInfo?,
        displayHeightMeters: Float?,
        displayDepthMeters: Float?
    ) {
        self.schemaVersion = schemaVersion
        self.aabbMin = aabbMin
        self.aabbMax = aabbMax
        self.floor = floor
        self.ceiling = ceiling
        self.walls = walls
        self.corners = corners
        self.freeFloorRegions = freeFloorRegions
        self.surfacePalette = surfacePalette
        self.capturedAt = capturedAt
        self.sceneToMeters = sceneToMeters
        self.sourceCameraInfo = sourceCameraInfo
        self.displayHeightMeters = displayHeightMeters
        self.displayDepthMeters = displayDepthMeters
    }

    // MARK: Init from RoomModel

    public init(roomModel: RoomModel, preserving previous: EnhancedRoomMetadata? = nil) {
        let h = roomModel.heightMeters
        let d = roomModel.depthMeters
        self.init(
            schemaVersion: Self.currentSchemaVersion,
            aabbMin: CodableSIMD3(roomModel.aabb.min),
            aabbMax: CodableSIMD3(roomModel.aabb.max),
            floor: CodablePlane(plane: roomModel.floor),
            ceiling: roomModel.ceiling.map(CodablePlane.init),
            walls: roomModel.walls.map(CodablePlane.init),
            corners: roomModel.corners.map(CodableCorner.init),
            freeFloorRegions: roomModel.freeFloorRegions.map(CodableFreeRegion.init),
            surfacePalette: CodableSurfacePalette(palette: roomModel.surfacePalette),
            capturedAt: previous?.capturedAt ?? Date(),
            sceneToMeters: roomModel.sceneToMeters,
            sourceCameraInfo: roomModel.cameraInfo.map(CodableSourceCameraInfo.init),
            displayHeightMeters: h.isFinite && h > 0.05 ? h : nil,
            displayDepthMeters: d.isFinite && d > 0.05 ? d : nil
        )
    }

    /// Legacy `*.ply.meta` flat string dictionary (room dimensions only, no room-intelligence JSON).
    public static func fromLegacyFlatMeta(_ legacy: [String: String]) -> EnhancedRoomMetadata {
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
        let src = CodableSourceCameraInfo(
            info: SourceCameraInfo(photoOrientation: orientation)
        )
        let sw = legacy["roomSceneWidth"].flatMap(Float.init)
        let sh = legacy["roomSceneHeight"].flatMap(Float.init)
        let sd = legacy["roomSceneDepth"].flatMap(Float.init)
        let sx = sw ?? 2
        let sy = sh ?? 2.5
        let sz = sd ?? 2
        let aabbMin = CodableSIMD3(SIMD3<Float>(0, 0, 0))
        let aabbMax = CodableSIMD3(SIMD3<Float>(sx, sy, sz))
        let floor = CodablePlane(plane: DetectedPlane(type: .floor, normal: SIMD3(0, 1, 0), pointOnPlane: .zero))
        let ceilingY = max(sy * 0.95, 0.1)
        let ceiling = CodablePlane(plane: DetectedPlane(
            type: .ceiling,
            normal: SIMD3(0, -1, 0),
            pointOnPlane: SIMD3(sx * 0.5, ceilingY, sz * 0.5)
        ))
        var stm: Float = 1.0
        if let hM = legacy["roomHeight"].flatMap(Float.init), let sceneH = sh, sceneH > 1e-4 {
            stm = hM / sceneH
        }
        let dh = legacy["roomHeight"].flatMap(Float.init)
        let dd = legacy["roomDepth"].flatMap(Float.init)
        return EnhancedRoomMetadata(
            schemaVersion: Self.currentSchemaVersion,
            aabbMin: aabbMin,
            aabbMax: aabbMax,
            floor: floor,
            ceiling: ceiling,
            walls: [],
            corners: [],
            freeFloorRegions: [],
            surfacePalette: CodableSurfacePalette(palette: .empty),
            capturedAt: Date(),
            sceneToMeters: stm,
            sourceCameraInfo: src,
            displayHeightMeters: dh.flatMap { $0.isFinite && $0 > 0.05 ? $0 : nil },
            displayDepthMeters: dd.flatMap { $0.isFinite && $0 > 0.05 ? $0 : nil }
        )
    }

    // MARK: - Schema Migration

    /// Migrates an older-version metadata payload to the current schema.
    /// No-op when `schemaVersion == currentSchemaVersion`.
    public static func migrate(from data: Data) throws -> EnhancedRoomMetadata {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601

        // Peek at the schema version
        let peek = try decoder.decode(VersionPeek.self, from: data)

        switch peek.schemaVersion {
        case Self.currentSchemaVersion:
            // Already current – decode directly
            return try decoder.decode(EnhancedRoomMetadata.self, from: data)

        case 1:
            // v1 → v2: freeFloorRegions gained occupancyRatio (decoded as nil automatically)
            return try decoder.decode(EnhancedRoomMetadata.self, from: data)

        default:
            throw DecodingError.dataCorrupted(
                DecodingError.Context(
                    codingPath: [],
                    debugDescription: "Unknown EnhancedRoomMetadata schemaVersion: \(peek.schemaVersion)"
                )
            )
        }
    }

    private struct VersionPeek: Decodable {
        let schemaVersion: Int
    }

    // MARK: - Factory

    /// Decodes persisted JSON and returns an `EnhancedRoomMetadata`, applying
    /// schema migration as needed.
    public static func from(data: Data) throws -> EnhancedRoomMetadata {
        try migrate(from: data)
    }

    /// Convenience factory that builds metadata from a `RoomModel`, optionally
    /// forwarding fields preserved from a prior save (e.g. `capturedAt`).
    public static func from(
        roomModel: RoomModel,
        preserving previous: EnhancedRoomMetadata? = nil
    ) -> EnhancedRoomMetadata {
        EnhancedRoomMetadata(roomModel: roomModel, preserving: previous)
    }

    // MARK: - Reconstruct RoomModel

    /// Reconstructs a `RoomModel` from the persisted metadata.
    public func roomModel() -> RoomModel {
        RoomModel(
            aabb: AABB3(min: aabbMin.value, max: aabbMax.value),
            floor: floor.detectedPlane(),
            ceiling: ceiling?.detectedPlane(),
            walls: walls.map { $0.detectedPlane() },
            corners: corners.map { $0.roomCorner() },
            freeFloorRegions: freeFloorRegions.map { $0.freeRegion() },
            surfacePalette: surfacePalette.surfacePalette(),
            cameraInfo: sourceCameraInfo?.sourceCameraInfo(),
            sceneToMeters: sceneToMeters
        )
    }

    // MARK: - Diagnostics

    public func printSaveDiagnostics() {
        print("[EnhancedRoomMetadata] schemaVersion   : \(schemaVersion)")
        print("[EnhancedRoomMetadata] capturedAt       : \(capturedAt)")
        print("[EnhancedRoomMetadata] sceneToMeters    : \(sceneToMeters)")
        if let dh = displayHeightMeters, let dd = displayDepthMeters {
            print("[EnhancedRoomMetadata] display H×D m  : \(dh) × \(dd)")
        } else {
            print("[EnhancedRoomMetadata] display H×D m  : nil")
        }
        print("[EnhancedRoomMetadata] walls            : \(walls.count)")
        print("[EnhancedRoomMetadata] corners          : \(corners.count)")
        print("[EnhancedRoomMetadata] freeFloorRegions : \(freeFloorRegions.count)")
        for (i, region) in freeFloorRegions.enumerated() {
            let ratioStr = region.occupancyRatio.map { String(format: "%.2f", $0) } ?? "nil (legacy)"
            print("[EnhancedRoomMetadata]   region[\(i)] areaSqM=\(region.areaSqM), occupancyRatio=\(ratioStr)")
        }
        print("[EnhancedRoomMetadata] surfacePalette   : floor=\(surfacePalette.floor != nil), walls=\(surfacePalette.walls != nil), ceiling=\(surfacePalette.ceiling != nil)")
    }
}
