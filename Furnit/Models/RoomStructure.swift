//import Foundation
//import CoreGraphics
//
//// MARK: - Room Structure (with proper initialization)
//struct RoomStructure: Equatable {
//    var wallLines: [Line] = []
//    var floorRegion: CGRect?
//    var ceilingRegion: CGRect?
//    var vanishingPoint: CGPoint?
//    
//    // ✅ Boundary values from manual adjustment
//    var floorY: CGFloat = 0.85
//    var ceilingY: CGFloat = 0.15
//    var leftX: CGFloat = 0.12
//    var rightX: CGFloat = 0.88
//    var vanishingX: CGFloat = 0.5
//    var vanishingY: CGFloat = 0.45
//    
//    struct Line: Equatable {
//        var start: CGPoint
//        var end: CGPoint
//        var confidence: Float
//    }
//}
