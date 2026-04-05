import Foundation
import simd

public struct RoomFurnitureDimensions: Equatable, Sendable {
    public let widthM: Float
    public let heightM: Float
    public let depthM: Float

    public init(widthM: Float, heightM: Float, depthM: Float) {
        self.widthM = widthM
        self.heightM = heightM
        self.depthM = depthM
    }
}

public struct ClearanceReport: Equatable, Sendable {
    public static let minimumPassageM: Float = 0.6

    public let frontM: Float?
    public let backM: Float?
    public let leftM: Float?
    public let rightM: Float?

    public init(frontM: Float?, backM: Float?, leftM: Float?, rightM: Float?) {
        self.frontM = frontM
        self.backM = backM
        self.leftM = leftM
        self.rightM = rightM
    }

    public var warnings: [String] {
        var result: [String] = []
        if let frontM, frontM < Self.minimumPassageM { result.append("Front clearance below 0.6m.") }
        if let leftM, leftM < Self.minimumPassageM { result.append("Left clearance below 0.6m.") }
        if let rightM, rightM < Self.minimumPassageM { result.append("Right clearance below 0.6m.") }
        return result
    }
}

public struct FitLocation: Equatable, Sendable {
    public let centerScene: SIMD3<Float>
    public let yRotationRad: Float
    public let clearance: ClearanceReport
    public let regionIndex: Int

    public init(centerScene: SIMD3<Float>, yRotationRad: Float, clearance: ClearanceReport, regionIndex: Int) {
        self.centerScene = centerScene
        self.yRotationRad = yRotationRad
        self.clearance = clearance
        self.regionIndex = regionIndex
    }
}

public struct FitCheckResult: Equatable, Sendable {
    public let fitsInRoom: Bool
    public let fitLocations: [FitLocation]
    public let warnings: [String]

    public init(fitsInRoom: Bool, fitLocations: [FitLocation], warnings: [String]) {
        self.fitsInRoom = fitsInRoom
        self.fitLocations = fitLocations
        self.warnings = warnings
    }
}

public struct FitCheckEngine {
    private let roomModel: RoomModel

    public init(roomModel: RoomModel) {
        self.roomModel = roomModel
    }

    public func checkFit(furniture: RoomFurnitureDimensions) -> FitCheckResult {
        let roomWidth = roomModel.widthMeters
        let roomDepth = roomModel.depthMeters
        let roomHeight = roomModel.heightMeters
        let widthFits = furniture.widthM <= roomWidth || furniture.depthM <= roomWidth
        let depthFits = furniture.depthM <= roomDepth || furniture.widthM <= roomDepth
        let heightFits = furniture.heightM <= roomHeight

        guard widthFits, depthFits else {
            return FitCheckResult(fitsInRoom: false, fitLocations: [], warnings: ["Furniture footprint exceeds room extents."])
        }

        var warnings: [String] = []
        if !heightFits {
            warnings.append("Furniture height may exceed room height.")
        }

        let locations = roomModel.freeFloorRegions.enumerated().compactMap { index, region -> FitLocation? in
            let widthFitsRegion = region.fits(width: furniture.widthM, depth: furniture.depthM)
            let rotatedFitsRegion = region.fits(width: furniture.depthM, depth: furniture.widthM)
            guard widthFitsRegion || rotatedFitsRegion else { return nil }

            let regionCenterUV = (region.uvBounds.min + region.uvBounds.max) * 0.5
            let floorOrigin = roomModel.floor.pointOnPlane
            let centerScene = SIMD3<Float>(floorOrigin.x + regionCenterUV.x, floorOrigin.y, floorOrigin.z + regionCenterUV.y)
            let clearance = ClearanceReport(
                frontM: max(0, region.areaSqM.squareRoot() - furniture.depthM),
                backM: nil,
                leftM: max(0, region.areaSqM.squareRoot() - furniture.widthM),
                rightM: nil
            )
            return FitLocation(
                centerScene: centerScene,
                yRotationRad: widthFitsRegion ? 0 : (.pi / 2),
                clearance: clearance,
                regionIndex: index
            )
        }

        warnings.append(contentsOf: locations.flatMap { $0.clearance.warnings })
        return FitCheckResult(fitsInRoom: true, fitLocations: locations, warnings: warnings)
    }
}
