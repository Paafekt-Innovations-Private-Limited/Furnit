import Foundation
import os.log

// MARK: - Runtime Debug Mode Check
/// Check if debug mode is enabled in Settings
/// Reads directly from UserDefaults to avoid circular dependency with AppStateManager
private func isDebugModeEnabled() -> Bool {
    // Read directly from UserDefaults (same key as QualitySettings uses)
    // This avoids triggering AppStateManager.shared initialization
    return UserDefaults.standard.bool(forKey: "debug_mode")
}

// MARK: - Global Debug Print Function
/// Use this instead of print() - only logs when debug mode is enabled in Settings
/// Named 'logDebug' to avoid conflict with Swift's debugPrint
@inline(__always)
func logDebug(_ items: Any..., separator: String = " ", terminator: String = "\n") {
    guard isDebugModeEnabled() else { return }
    let output = items.map { "\($0)" }.joined(separator: separator)
    print(output, terminator: terminator)
}

// MARK: - Always-on diagnostics (not gated by Settings debug mode)

/// YOLO wall measurement on save — always printed. Filter in Console: `WALL_MEAS` (matches Android `adb logcat | grep WALL_MEAS`).
func logWallMeasurement(_ message: String) {
    print("[WALL_MEAS] \(message)")
}

/// PLY bounds / navigation — always printed. Filter: `PLY_BOUNDS`.
func logPlyBoundsDiagnostic(_ message: String) {
    print("[PLY_BOUNDS] \(message)")
}

/// Centralized logging utility using os_log framework
/// - Debug/Info logs only appear when debug mode is enabled in Settings
/// - Error/Critical logs always appear for crash reporting
enum AppLogger {

    // MARK: - Log Categories
    private static let subsystem = Bundle.main.bundleIdentifier ?? "com.paafektinnovations.Paafekt"

    static let auth = Logger(subsystem: subsystem, category: "Authentication")
    static let ui = Logger(subsystem: subsystem, category: "UI")
    static let model = Logger(subsystem: subsystem, category: "ModelManager")
    static let scene = Logger(subsystem: subsystem, category: "SceneKit")
    static let room = Logger(subsystem: subsystem, category: "RoomProcessing")
    static let camera = Logger(subsystem: subsystem, category: "Camera")
    static let network = Logger(subsystem: subsystem, category: "Network")
    static let appCheck = Logger(subsystem: subsystem, category: "AppCheck")
    static let general = Logger(subsystem: subsystem, category: "General")

    // MARK: - Debug Logging (only when debug mode enabled in Settings)

    /// Debug log - only appears when debug mode is enabled in Settings
    static func debug(_ message: String, category: Logger = general) {
        guard isDebugModeEnabled() else { return }
        category.debug("\(message)")
    }

    /// Info log - only appears when debug mode is enabled in Settings
    static func info(_ message: String, category: Logger = general) {
        guard isDebugModeEnabled() else { return }
        category.info("\(message)")
    }

    /// Warning log - only appears when debug mode is enabled in Settings
    static func warning(_ message: String, category: Logger = general) {
        guard isDebugModeEnabled() else { return }
        category.warning("\(message)")
    }

    // MARK: - Production Logging (always logged for crash reporting)

    /// Error log - ALWAYS appears (for crash reporting)
    static func error(_ message: String, category: Logger = general) {
        category.error("\(message)")
    }

    /// Critical/Fault log - ALWAYS appears (for crash reporting)
    static func critical(_ message: String, category: Logger = general) {
        category.critical("\(message)")
    }
}

// MARK: - Convenience Extensions

extension AppLogger {
    /// Log authentication events
    static func authDebug(_ message: String) {
        debug(message, category: auth)
    }

    static func authError(_ message: String) {
        error(message, category: auth)
    }

    /// Log UI events
    static func uiDebug(_ message: String) {
        debug(message, category: ui)
    }

    /// Log model/file operations
    static func modelDebug(_ message: String) {
        debug(message, category: model)
    }

    static func modelError(_ message: String) {
        error(message, category: model)
    }

    /// Log scene/3D operations
    static func sceneDebug(_ message: String) {
        debug(message, category: scene)
    }

    /// Log room processing
    static func roomDebug(_ message: String) {
        debug(message, category: room)
    }

    static func roomError(_ message: String) {
        error(message, category: room)
    }

    /// Log camera operations
    static func cameraDebug(_ message: String) {
        debug(message, category: camera)
    }
}
