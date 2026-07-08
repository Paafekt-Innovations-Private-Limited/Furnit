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

enum SplatRoomFurnitureCatalog {
    static let standardItems: [SplatRoomFurnitureItem] = [
        SplatRoomFurnitureItem(category: "Sofa", dimensions: SIMD3<Float>(1.85, 0.85, 0.90), tint: .systemGreen),
        SplatRoomFurnitureItem(category: "Bed", dimensions: SIMD3<Float>(2.00, 0.60, 1.50), tint: .systemBlue),
        SplatRoomFurnitureItem(category: "Table", dimensions: SIMD3<Float>(1.50, 0.75, 0.90), tint: .systemOrange),
        SplatRoomFurnitureItem(category: "Chair", dimensions: SIMD3<Float>(0.50, 0.85, 0.50), tint: .systemTeal),
        SplatRoomFurnitureItem(category: "Wardrobe", dimensions: SIMD3<Float>(1.50, 2.00, 0.60), tint: .systemPurple),
        SplatRoomFurnitureItem(category: "Desk", dimensions: SIMD3<Float>(1.20, 0.75, 0.60), tint: .systemPink),
    ]
}
