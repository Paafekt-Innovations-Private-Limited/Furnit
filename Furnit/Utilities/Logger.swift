import Foundation
import os.log

// MARK: - Global Debug Print Function
/// Use this instead of print() - only logs in DEBUG builds
/// Named 'logDebug' to avoid conflict with Swift's debugPrint
@inline(__always)
func logDebug(_ items: Any..., separator: String = " ", terminator: String = "\n") {
    #if DEBUG
    let output = items.map { "\($0)" }.joined(separator: separator)
    print(output, terminator: terminator)
    #endif
}

/// Centralized logging utility using os_log framework
/// - Debug logs only appear in DEBUG builds
/// - Error/Critical logs appear in all builds for crash reporting
enum AppLogger {

    // MARK: - Log Categories
    private static let subsystem = Bundle.main.bundleIdentifier ?? "com.pafekt.Furnit"

    static let auth = Logger(subsystem: subsystem, category: "Authentication")
    static let ui = Logger(subsystem: subsystem, category: "UI")
    static let model = Logger(subsystem: subsystem, category: "ModelManager")
    static let scene = Logger(subsystem: subsystem, category: "SceneKit")
    static let room = Logger(subsystem: subsystem, category: "RoomProcessing")
    static let camera = Logger(subsystem: subsystem, category: "Camera")
    static let network = Logger(subsystem: subsystem, category: "Network")
    static let appCheck = Logger(subsystem: subsystem, category: "AppCheck")
    static let general = Logger(subsystem: subsystem, category: "General")

    // MARK: - Debug Logging (only in DEBUG builds)

    /// Debug log - only appears in DEBUG builds
    static func debug(_ message: String, category: Logger = general) {
        #if DEBUG
        category.debug("\(message)")
        #endif
    }

    /// Info log - only appears in DEBUG builds
    static func info(_ message: String, category: Logger = general) {
        #if DEBUG
        category.info("\(message)")
        #endif
    }

    // MARK: - Production Logging (always logged)

    /// Warning log - appears in all builds
    static func warning(_ message: String, category: Logger = general) {
        category.warning("\(message)")
    }

    /// Error log - appears in all builds (for crash reporting)
    static func error(_ message: String, category: Logger = general) {
        category.error("\(message)")
    }

    /// Critical/Fault log - appears in all builds (for crash reporting)
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
