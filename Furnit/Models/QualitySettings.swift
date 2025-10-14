import Foundation
import SwiftUI

// Movement speed levels for camera navigation
enum MovementSpeed: String, CaseIterable, Identifiable {
    case slow = "slow"
    case normal = "normal"
    case fast = "fast"

    var id: String { rawValue }

    // Display names for UI
    var displayName: String {
        switch self {
        case .slow:
            return "Slow"
        case .normal:
            return "Normal"
        case .fast:
            return "Fast"
        }
    }

    // Description for each speed level
    var description: String {
        switch self {
        case .slow:
            return "Precise control for detailed exploration"
        case .normal:
            return "Balanced speed for comfortable navigation"
        case .fast:
            return "Quick movement for faster exploration"
        }
    }

    // Icon for each speed level
    var icon: String {
        switch self {
        case .slow:
            return "tortoise.fill"
        case .normal:
            return "figure.walk"
        case .fast:
            return "hare.fill"
        }
    }

    // Actual speed value in units per frame
    var speedValue: Float {
        switch self {
        case .slow:
            return 0.001     // Extremely precise control for detailed exploration
        case .normal:
            return 0.0015    // Comfortable but controlled navigation
        case .fast:
            return 0.002     // Responsive but still very controllable
        }
    }
}

// Quality levels for 3D asset rendering
enum AssetQuality: String, CaseIterable, Identifiable {
    case standard = "standard"
    case high = "high" 
    case best = "best"
    
    var id: String { rawValue }
    
    // Display names for UI
    var displayName: String {
        switch self {
        case .standard:
            return "Standard"
        case .high:
            return "High"
        case .best:
            return "Best"
        }
    }
    
    // Description for each quality level
    var description: String {
        switch self {
        case .standard:
            return "Basic quality for faster performance"
        case .high:
            return "Enhanced quality with better details"
        case .best:
            return "Premium quality with maximum detail"
        }
    }
    
    // Icon for each quality level
    var icon: String {
        switch self {
        case .standard:
            return "speedometer"
        case .high:
            return "eye"
        case .best:
            return "crown"
        }
    }
    
    // Whether this quality level is available for selection
    var isAvailable: Bool {
        switch self {
        case .standard, .high:
            return true
        case .best:
            return false // Disabled - requires partner store visit
        }
    }
    
    // Special message for unavailable quality levels
    var unavailableMessage: String? {
        switch self {
        case .best:
            return "For best quality, please visit our partner stores"
        default:
            return nil
        }
    }
}

// ObservableObject to manage quality settings across the app
class QualitySettings: ObservableObject {
    // Current selected quality (published for UI updates)
    @Published var selectedQuality: AssetQuality {
        didSet {
            // Only save if the quality is actually available
            if selectedQuality.isAvailable {
                saveQuality()
            }
        }
    }

    // Current selected movement speed (published for UI updates)
    @Published var selectedMovementSpeed: MovementSpeed {
        didSet {
            saveMovementSpeed()
        }
    }

    // UserDefaults keys for persistence
    private let qualityKey = "selected_asset_quality"
    private let movementSpeedKey = "selected_movement_speed"
    
    // Initialize with saved quality or default to high
    init() {
        if let savedQuality = UserDefaults.standard.string(forKey: qualityKey),
           let quality = AssetQuality(rawValue: savedQuality),
           quality.isAvailable {
            self.selectedQuality = quality
        } else {
            // Default to standard quality if no saved preference
            self.selectedQuality = .standard
        }

        if let savedMovementSpeed = UserDefaults.standard.string(forKey: movementSpeedKey),
           let speed = MovementSpeed(rawValue: savedMovementSpeed) {
            self.selectedMovementSpeed = speed
        } else {
            // Default to normal speed if no saved preference
            self.selectedMovementSpeed = .normal
        }
    }
    
    // Save quality setting to UserDefaults
    private func saveQuality() {
        UserDefaults.standard.set(selectedQuality.rawValue, forKey: qualityKey)
        print("💾 Saved quality setting: \(selectedQuality.displayName)")
    }

    // Save movement speed setting to UserDefaults
    private func saveMovementSpeed() {
        UserDefaults.standard.set(selectedMovementSpeed.rawValue, forKey: movementSpeedKey)
        print("💾 Saved movement speed setting: \(selectedMovementSpeed.displayName)")
    }
    
    // Get all available quality options for UI
    var availableQualities: [AssetQuality] {
        return AssetQuality.allCases
    }
    
    // Check if a specific quality is selected
    func isSelected(_ quality: AssetQuality) -> Bool {
        return selectedQuality == quality
    }
    
    // Attempt to select a quality (only works if available)
    func selectQuality(_ quality: AssetQuality) {
        guard quality.isAvailable else {
            print("⚠️ Attempted to select unavailable quality: \(quality.displayName)")
            return
        }

        selectedQuality = quality
        print("🎨 Quality changed to: \(quality.displayName)")
    }

    // Select movement speed
    func selectMovementSpeed(_ speed: MovementSpeed) {
        selectedMovementSpeed = speed
        print("🏃 Movement speed changed to: \(speed.displayName)")
    }

    // Check if a specific movement speed is selected
    func isMovementSpeedSelected(_ speed: MovementSpeed) -> Bool {
        return selectedMovementSpeed == speed
    }
    
    // Reset to default quality
    func resetToDefault() {
        selectedQuality = .high
        selectedMovementSpeed = .normal
    }
}

// Extension for RealityKit rendering configuration
extension AssetQuality {
    // RealityKit rendering scale factor based on quality
    var renderScale: Float {
        switch self {
        case .standard:
            return 0.75
        case .high:
            return 1.0 
        case .best:
            return 1.25
        }
    }
    
    // Lighting intensity multiplier for RealityKit
    var lightingIntensity: Float {
        switch self {
        case .standard:
            return 0.7
        case .high:
            return 1.0
        case .best:
            return 1.3
        }
    }
    
    // Whether to enable high-resolution textures
    var useHighResTextures: Bool {
        switch self {
        case .standard:
            return false
        case .high:
            return true
        case .best:
            return true
        }
    }
    
    // Shadow quality level for RealityKit
    var shadowQuality: String {
        switch self {
        case .standard:
            return "low"
        case .high:
            return "medium"
        case .best:
            return "high"
        }
    }
    
    // LOD (Level of Detail) bias for model quality
    var lodBias: Float {
        switch self {
        case .standard:
            return 1.5 // Lower detail at distance
        case .high:
            return 1.0 // Normal detail
        case .best:
            return 0.5 // Higher detail at distance
        }
    }
}
