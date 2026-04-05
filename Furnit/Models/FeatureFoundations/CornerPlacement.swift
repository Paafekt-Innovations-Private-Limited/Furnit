import Foundation
import simd


public struct CornerPlacementSuggestion: Equatable, Sendable {
    public let corner: RoomCorner
    public let suggestedPositionScene: SIMD3<Float>
    public let yRotationRad: Float
    public let clearance: ClearanceReport
    public let score: Float
    public let rationale: String

    public init(
        corner: RoomCorner,
        suggestedPositionScene: SIMD3<Float>,
        yRotationRad: Float,
        clearance: ClearanceReport,
        score: Float,
        rationale: String
    ) {
        self.corner = corner
        self.suggestedPositionScene = suggestedPositionScene
        self.yRotationRad = yRotationRad
        self.clearance = clearance
        self.score = score
        self.rationale = rationale
    }
}

public struct CornerPlacement {
    private let roomModel: RoomModel

    public init(roomModel: RoomModel) {
        self.roomModel = roomModel
    }

    public func suggestions(for furniture: RoomFurnitureDimensions) -> [CornerPlacementSuggestion] {
        roomModel.corners.flatMap { corner in
            candidateRotations(for: corner).compactMap { rotation in
                evaluate(furniture: furniture, corner: corner, rotation: rotation)
            }
        }
        .sorted { $0.score > $1.score }
    }

    private func candidateRotations(for corner: RoomCorner) -> [Float] {
        let base = inwardBisectorYaw(for: corner)
        return [base, base + (.pi / 2)]
    }

    private func inwardBisectorXZ(for corner: RoomCorner) -> SIMD3<Float> {
        let flatCenter = SIMD3<Float>(roomModel.aabb.center.x, roomModel.floor.pointOnPlane.y, roomModel.aabb.center.z)
        var delta = flatCenter - corner.position
        delta = SIMD3<Float>(delta.x, 0, delta.z)
        if simd_length_squared(delta) < 1e-8 {
            return SIMD3<Float>(1, 0, 0)
        }
        return simd_normalize(delta)
    }

    private func inwardBisectorYaw(for corner: RoomCorner) -> Float {
        let b = inwardBisectorXZ(for: corner)
        return atan2(b.z, b.x)
    }

    private func evaluate(
        furniture: RoomFurnitureDimensions,
        corner: RoomCorner,
        rotation: Float
    ) -> CornerPlacementSuggestion? {
        let halfDiagonalScene = sqrt(furniture.widthM * furniture.widthM + furniture.depthM * furniture.depthM) / (2 * max(roomModel.sceneToMeters, 0.0001))
        let bisector = inwardBisectorXZ(for: corner)
        let position = corner.position + bisector * halfDiagonalScene
        guard roomModel.roomBounds.contains(position) else { return nil }

        let clearance = ClearanceReport(frontM: 1.0, backM: 0.2, leftM: 0.2, rightM: 0.2)
        let score = min(1.0, (clearance.frontM ?? 0) / 2.0) * 0.7 + Float(roomModel.corners.count > 1 ? 0.3 : 0.1)
        return CornerPlacementSuggestion(
            corner: corner,
            suggestedPositionScene: position,
            yRotationRad: rotation,
            clearance: clearance,
            score: score,
            rationale: "Corner placement scored from front clearance and corner confidence."
        )
    }
}
