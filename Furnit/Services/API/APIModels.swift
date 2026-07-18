import Foundation

// MARK: - Model File Type

/// Supported 3D model file types
enum ModelFileType: String, Codable {
    case usdz
    case ply
    case meshroom  // Manual setup rooms (image + metadata) - legacy
    case glb       // GLTF binary format - universal 3D format for WebGL

    /// File extension including the dot
    var fileExtension: String {
        return ".\(rawValue)"
    }

    /// Display name for the file type
    var displayName: String {
        switch self {
        case .usdz:
            return "3D Model"
        case .ply:
            return "3D Room"
        case .meshroom:
            return "Manual Room"
        case .glb:
            return "3D Room"
        }
    }

    /// Icon name for display
    var iconName: String {
        switch self {
        case .usdz:
            return "cube.fill"
        case .ply:
            return "circle.grid.3x3.fill"
        case .meshroom:
            return "square.grid.3x3.fill"
        case .glb:
            return "cube.transparent.fill"
        }
    }

    /// Icon color for display
    var iconColorName: String {
        switch self {
        case .usdz:
            return "green"
        case .ply:
            return "purple"
        case .meshroom:
            return "orange"
        case .glb:
            return "blue"
        }
    }
}
