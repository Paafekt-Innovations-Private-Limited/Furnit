import Foundation

enum RoomGenerationImplementation: String, CaseIterable, Codable, Identifiable, Sendable {
    case sharpCoreML = "sharp_core_ml"
    case qwenImageToRoom = "qwen_image_to_room"
    case depthAnythingMetricUSDZ = "depth_anything_metric_usdz"
    case lidarSweepFusion = "lidar_sweep_fusion"
    case swiftSharpMath = "swift_sharp_math"

    static let defaultImplementation: RoomGenerationImplementation = .depthAnythingMetricUSDZ

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .depthAnythingMetricUSDZ:
            return "Depth Anything"
        case .qwenImageToRoom:
            return "Qwen"
        case .swiftSharpMath:
            return "Swift SHARP Math"
        case .lidarSweepFusion:
            return "LiDAR Sweep"
        case .sharpCoreML:
            return "SHARP Room"
        }
    }

    var settingsDescription: String {
        switch self {
        case .depthAnythingMetricUSDZ:
            return "Single-photo metric depth mesh to USDZ."
        case .qwenImageToRoom:
            return "Qwen vision model via Ollama — room JSON to USDZ."
        case .swiftSharpMath:
            return "Local no-ML Swift math prototype for SHARP-style PLY testing."
        case .lidarSweepFusion:
            return "LiDAR posed-frame sweep flow for fit-grade room capture."
        case .sharpCoreML:
            return "Existing SHARP Core ML room generator."
        }
    }
}
