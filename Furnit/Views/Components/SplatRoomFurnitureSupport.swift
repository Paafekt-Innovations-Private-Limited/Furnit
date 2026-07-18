import Foundation
import UIKit
import simd

struct SplatRoomFurnitureItem: Identifiable, Equatable {
    let id = UUID()
    let category: String
    let dimensions: SIMD3<Float>
    let tint: UIColor
}

struct SplatRoomPlacedFurniture: Identifiable {
    let id: UUID
    let item: SplatRoomFurnitureItem
    var position: SIMD3<Float>
    var rotationY: Float
    var fits: Bool
    var clearanceMeters: Float
}
