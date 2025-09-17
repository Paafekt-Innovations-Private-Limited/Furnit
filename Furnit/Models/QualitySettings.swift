import Foundation
import SwiftUI

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
    
    // UserDefaults key for persistence
    private let qualityKey = "selected_asset_quality"
    
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
    }
    
    // Save quality setting to UserDefaults
    private func saveQuality() {
        UserDefaults.standard.set(selectedQuality.rawValue, forKey: qualityKey)
        print("💾 Saved quality setting: \(selectedQuality.displayName)")
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
    
    // Reset to default quality
    func resetToDefault() {
        selectedQuality = .high
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