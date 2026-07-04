import Foundation

enum RoomGenerationImplementation: String, CaseIterable, Codable, Identifiable, Sendable {
    case depthAnythingMetricUSDZ = "depth_anything_metric_usdz"
    case swiftSharpMath = "swift_sharp_math"
    case lidarSweepFusion = "lidar_sweep_fusion"
    case sharpCoreML = "sharp_core_ml"

    static let defaultImplementation: RoomGenerationImplementation = .depthAnythingMetricUSDZ

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .depthAnythingMetricUSDZ:
            return "Depth Anything Metric USDZ"
        case .swiftSharpMath:
            return "Swift SHARP Math"
        case .lidarSweepFusion:
            return "LiDAR Sweep Fusion"
        case .sharpCoreML:
            return "SHARP Core ML"
        }
    }
}
