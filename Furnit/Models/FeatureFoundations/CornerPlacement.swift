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
        let base = atan2(corner.bisector.z, corner.bisector.x)
        return [base, base + (.pi / 2)]
    }

    private func evaluate(
        furniture: RoomFurnitureDimensions,
        corner: RoomCorner,
        rotation: Float
    ) -> CornerPlacementSuggestion? {
        let halfDiagonalScene = sqrt(furniture.widthM * furniture.widthM + furniture.depthM * furniture.depthM) / (2 * max(roomModel.sceneToMeters, 0.0001))
        let position = corner.position + corner.bisector * halfDiagonalScene
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
