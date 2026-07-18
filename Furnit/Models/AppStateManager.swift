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
        
        logDebug("🚀 AppStateManager initialized")
        logDebug("   App Version: \(appVersion) (\(buildNumber))")
    }
    
    // Convenience method to get current quality
    var currentQuality: AssetQuality {
        return qualitySettings.selectedQuality
    }

    // Get formatted app version string
    var formattedVersion: String {
        return "Version \(appVersion) (\(buildNumber))"
    }
    
    // Reset all settings to defaults
    func resetAllSettings() {
        qualitySettings.resetToDefault()
        logDebug("🔄 All settings reset to defaults")
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
