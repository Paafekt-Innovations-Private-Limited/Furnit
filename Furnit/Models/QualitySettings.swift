import ARKit
import Foundation
import SwiftUI

// Quality levels for 3D asset rendering
enum AssetQuality: String, CaseIterable, Identifiable {
    case standard = "standard"
    case high = "high"
    case best = "best"

    var id: String { rawValue }

    // Display names for UI (localized)
    var displayName: String {
        switch self {
        case .standard:
            return L10n.Quality.standard
        case .high:
            return L10n.Quality.high
        case .best:
            return L10n.Quality.best
        }
    }

}

// ObservableObject to manage quality settings across the app
class QualitySettings: ObservableObject {

    /// `true` when this device can supply scene/smoothed scene depth for Furniture Fit’s AR companion.
    static var supportsLiDARSceneDepth: Bool {
        ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth)
            || ARWorldTrackingConfiguration.supportsFrameSemantics(.smoothedSceneDepth)
    }

    /// Fixed rendering profile for the RealityKit/USDZ path.
    /// The Settings surface for asset quality has been removed because it did not
    /// apply consistently across the app.
    @Published var selectedQuality: AssetQuality = .standard

    // Debug mode toggle (published for UI updates)
    @Published var debugMode: Bool {
        didSet {
            saveDebugMode()
        }
    }

    // BBox-in-mask threshold (published for UI updates)
    // Controls how much of a bbox must be covered by the final mask
    @Published var bboxInMaskThreshold: Float {
        didSet {
            saveBboxInMaskThreshold()
        }
    }

    /// Effective companion support for Furniture Fit metric sizing.
    /// With the Settings toggle removed, this is now purely a hardware capability check.
    var furnitureFitARDepthCompanionRuntimeActive: Bool {
        Self.supportsLiDARSceneDepth
    }

    // UserDefaults keys for persistence
    private let debugModeKey = "debug_mode"
    private let bboxInMaskThresholdKey = "bbox_in_mask_threshold"

    // Initialize settings with stable defaults
    init() {
        #if DEBUG
        self.debugMode = UserDefaults.standard.bool(forKey: debugModeKey)
        #else
        self.debugMode = false
        UserDefaults.standard.set(false, forKey: debugModeKey)
        #endif

        // Load bbox-in-mask threshold setting, default to 0.30
        let savedBboxThreshold = UserDefaults.standard.float(forKey: bboxInMaskThresholdKey)
        self.bboxInMaskThreshold = savedBboxThreshold > 0 ? savedBboxThreshold : 0.30

    }

    // Save debug mode setting to UserDefaults
    private func saveDebugMode() {
        UserDefaults.standard.set(debugMode, forKey: debugModeKey)
        logDebug("💾 Saved debug mode setting: \(debugMode)")
    }

    // Save bbox-in-mask threshold setting to UserDefaults
    private func saveBboxInMaskThreshold() {
        UserDefaults.standard.set(bboxInMaskThreshold, forKey: bboxInMaskThresholdKey)
        logDebug("💾 Saved bbox-in-mask threshold setting: \(bboxInMaskThreshold)")
    }

    // Reset to default quality
    func resetToDefault() {
        selectedQuality = .standard
        debugMode = false
        bboxInMaskThreshold = 0.30
    }
}

// Extension for RealityKit rendering configuration
extension AssetQuality {
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
}
