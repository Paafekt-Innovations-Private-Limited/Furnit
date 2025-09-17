import Foundation
import SwiftUI

// Global app state manager for sharing settings across views
class AppStateManager: ObservableObject {
    // Singleton instance for global access
    static let shared = AppStateManager()
    
    // Quality settings instance
    @Published var qualitySettings: QualitySettings
    
    // App version and build info
    let appVersion: String
    let buildNumber: String
    
    // Private initializer for singleton pattern
    private init() {
        self.qualitySettings = QualitySettings()
        
        // Get app version info from bundle
        self.appVersion = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "1.0"
        self.buildNumber = Bundle.main.infoDictionary?["CFBundleVersion"] as? String ?? "1"
        
        print("🚀 AppStateManager initialized")
        print("   App Version: \(appVersion) (\(buildNumber))")
        print("   Quality Setting: \(qualitySettings.selectedQuality.displayName)")
    }
    
    // Convenience method to get current quality
    var currentQuality: AssetQuality {
        return qualitySettings.selectedQuality
    }

    // Convenience method to get current movement speed
    var currentMovementSpeed: MovementSpeed {
        return qualitySettings.selectedMovementSpeed
    }
    
    // Update quality setting with validation
    func updateQuality(_ quality: AssetQuality) {
        // Ensure UI updates by sending change notification
        objectWillChange.send()
        qualitySettings.selectQuality(quality)
    }

    // Update movement speed setting with UI notifications
    func updateMovementSpeed(_ speed: MovementSpeed) {
        // Ensure UI updates by sending change notification
        objectWillChange.send()
        qualitySettings.selectMovementSpeed(speed)
    }
    
    // Get formatted app version string
    var formattedVersion: String {
        return "Version \(appVersion) (\(buildNumber))"
    }
    
    // Check if this is the first app launch
    private let hasLaunchedKey = "has_launched_before"
    
    var isFirstLaunch: Bool {
        return !UserDefaults.standard.bool(forKey: hasLaunchedKey)
    }
    
    // Mark app as launched
    func markAsLaunched() {
        UserDefaults.standard.set(true, forKey: hasLaunchedKey)
    }
    
    // Reset all settings to defaults
    func resetAllSettings() {
        qualitySettings.resetToDefault()
        UserDefaults.standard.removeObject(forKey: hasLaunchedKey)
        print("🔄 All settings reset to defaults")
    }
}

// SwiftUI Environment Key for dependency injection
struct AppStateManagerKey: EnvironmentKey {
    static let defaultValue = AppStateManager.shared
}

// Environment extension for easy access
extension EnvironmentValues {
    var appState: AppStateManager {
        get { self[AppStateManagerKey.self] }
        set { self[AppStateManagerKey.self] = newValue }
    }
}