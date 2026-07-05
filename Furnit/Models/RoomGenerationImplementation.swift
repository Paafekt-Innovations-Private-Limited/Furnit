import Foundation

enum RoomGenerationImplementation: String, CaseIterable, Codable, Identifiable, Sendable {
    case depthAnythingMetricUSDZ = "depth_anything_metric_usdz"

    static let defaultImplementation: RoomGenerationImplementation = .depthAnythingMetricUSDZ

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .depthAnythingMetricUSDZ:
            return "Depth Anything"
        }
    }

    var settingsDescription: String {
        switch self {
        case .depthAnythingMetricUSDZ:
            return "Single-photo metric depth mesh to USDZ."
        }
    }
}
