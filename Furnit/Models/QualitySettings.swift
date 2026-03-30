import Foundation
import SwiftUI

// Movement speed levels for camera navigation
enum MovementSpeed: String, CaseIterable, Identifiable {
    case slow = "slow"
    case normal = "normal"
    case fast = "fast"

    var id: String { rawValue }

    // Display names for UI (localized)
    var displayName: String {
        switch self {
        case .slow:
            return L10n.Speed.slow
        case .normal:
            return L10n.Speed.normal
        case .fast:
            return L10n.Speed.fast
        }
    }

    // Description for each speed level (localized)
    var description: String {
        switch self {
        case .slow:
            return L10n.Speed.slowDescription
        case .normal:
            return L10n.Speed.normalDescription
        case .fast:
            return L10n.Speed.fastDescription
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

    // Description for each quality level (localized)
    var description: String {
        switch self {
        case .standard:
            return L10n.Quality.standardDescription
        case .high:
            return L10n.Quality.highDescription
        case .best:
            return L10n.Quality.bestDescription
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
    
    // Special message for unavailable quality levels (localized)
    var unavailableMessage: String? {
        switch self {
        case .best:
            return L10n.Quality.bestUnavailable
        default:
            return nil
        }
    }
}

// ObservableObject to manage quality settings across the app
class QualitySettings: ObservableObject {

    /// UserDefaults key for ``furnitureFitUseOnnxRuntime`` (read in ``FurnitureFitView`` to branch to Android-style stretch + postprocess).
    static let furnitureFitUseOnnxRuntimeStorageKey = "furniture_fit_use_onnx_runtime"
    /// UserDefaults key for optional Core ML GPU path (read by ``YOLOEModelService`` at model load).
    static let yoloeCoreMLAllowGPUKey = "yoloe_core_ml_allow_gpu"
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

    /// When `false` (default): Furniture Fit uses Core ML with letterbox preprocess and the standard mask pipeline. When `true`: same Core ML ``mlmodel`` with **stretch** preprocess and Android-aligned NMS / primary / mask (``FurnitureFitOnnxStylePipeline``). If the Core ML model is not loaded, falls back to bundled ONNX Runtime when a `.onnx` is present.
    @Published var furnitureFitUseOnnxRuntime: Bool {
        didSet {
            saveFurnitureFitUseOnnxRuntime()
        }
    }

    /// When `false` (default): YOLOE Core ML loads with CPU only (avoids SIGABRT on some 26L exports). When `true`: CPU+GPU (faster if stable on your device).
    @Published var yoloeCoreMLAllowGPU: Bool {
        didSet {
            saveYoloeCoreMLAllowGPU()
        }
    }

    // UserDefaults keys for persistence
    private let qualityKey = "selected_asset_quality"
    private let movementSpeedKey = "selected_movement_speed"
    private let debugModeKey = "debug_mode"
    private let bboxInMaskThresholdKey = "bbox_in_mask_threshold"
    private let furnitureFitUseOnnxRuntimeKey = QualitySettings.furnitureFitUseOnnxRuntimeStorageKey

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

        // Load debug mode setting, default to false
        self.debugMode = UserDefaults.standard.bool(forKey: debugModeKey)

        // Load bbox-in-mask threshold setting, default to 0.30
        let savedBboxThreshold = UserDefaults.standard.float(forKey: bboxInMaskThresholdKey)
        self.bboxInMaskThreshold = savedBboxThreshold > 0 ? savedBboxThreshold : 0.30

        // Default letterbox Core ML path (false); optional Android-style stretch + postprocess (same mlmodel when loaded)
        if UserDefaults.standard.object(forKey: furnitureFitUseOnnxRuntimeKey) != nil {
            self.furnitureFitUseOnnxRuntime = UserDefaults.standard.bool(forKey: furnitureFitUseOnnxRuntimeKey)
        } else {
            self.furnitureFitUseOnnxRuntime = false
        }

        if UserDefaults.standard.object(forKey: Self.yoloeCoreMLAllowGPUKey) != nil {
            self.yoloeCoreMLAllowGPU = UserDefaults.standard.bool(forKey: Self.yoloeCoreMLAllowGPUKey)
        } else {
            self.yoloeCoreMLAllowGPU = false
        }
    }
    
    // Save quality setting to UserDefaults
    private func saveQuality() {
        UserDefaults.standard.set(selectedQuality.rawValue, forKey: qualityKey)
        logDebug("💾 Saved quality setting: \(selectedQuality.displayName)")
    }

    // Save movement speed setting to UserDefaults
    private func saveMovementSpeed() {
        UserDefaults.standard.set(selectedMovementSpeed.rawValue, forKey: movementSpeedKey)
        logDebug("💾 Saved movement speed setting: \(selectedMovementSpeed.displayName)")
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

    private func saveFurnitureFitUseOnnxRuntime() {
        UserDefaults.standard.set(furnitureFitUseOnnxRuntime, forKey: furnitureFitUseOnnxRuntimeKey)
        logDebug("💾 Saved Furniture Fit ONNX preference: \(furnitureFitUseOnnxRuntime)")
    }

    private func saveYoloeCoreMLAllowGPU() {
        UserDefaults.standard.set(yoloeCoreMLAllowGPU, forKey: Self.yoloeCoreMLAllowGPUKey)
        logDebug("💾 Saved YOLOE Core ML GPU preference: \(yoloeCoreMLAllowGPU)")
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
            logDebug("⚠️ Attempted to select unavailable quality: \(quality.displayName)")
            return
        }

        selectedQuality = quality
        logDebug("🎨 Quality changed to: \(quality.displayName)")
    }

    // Select movement speed
    func selectMovementSpeed(_ speed: MovementSpeed) {
        selectedMovementSpeed = speed
        logDebug("🏃 Movement speed changed to: \(speed.displayName)")
    }

    // Check if a specific movement speed is selected
    func isMovementSpeedSelected(_ speed: MovementSpeed) -> Bool {
        return selectedMovementSpeed == speed
    }
    
    // Reset to default quality
    func resetToDefault() {
        selectedQuality = .high
        selectedMovementSpeed = .normal
        debugMode = false
        bboxInMaskThreshold = 0.30
        furnitureFitUseOnnxRuntime = false
        yoloeCoreMLAllowGPU = false
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
