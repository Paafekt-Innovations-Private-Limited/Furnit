import Foundation
import os.log
import Darwin

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

// MARK: - Furniture Fit & PLY stdout (Settings → Debug mode only)
/// PLY bounds / navigation. Filter: `PLY_BOUNDS`.
func logPlyBoundsDiagnostic(_ message: String) {
    guard isDebugModeEnabled() else { return }
    print("[PLY_BOUNDS] \(message)")
}

/// AR-assisted FurnitureFit metrics. Filter: `FurnitureFitAR`.
func logFurnitureFitAR(_ message: String) {
    guard isDebugModeEnabled() else { return }
    print("[FurnitureFitAR] \(message)")
}

/// Furniture W×H estimate and pipeline tags. Filter: `FurnitureFitSize`.
func logFurnitureFitSize(_ message: String) {
    guard isDebugModeEnabled() else { return }
    print("[FurnitureFitSize] \(message)")
}

/// Overlay scale assist. Filter: `FurnitureFitOverlay`.
func logFurnitureFitOverlay(_ message: String) {
    guard isDebugModeEnabled() else { return }
    print("[FurnitureFitOverlay] \(message)")
}

// MARK: - Unified diagnostics (gated by Settings debug mode)
/// These diagnostics use Unified Logging (`Logger`) so they stream reliably in Xcode/Console,
/// but they are still gated by Settings → Debug mode to avoid noisy production logs.
private enum AlwaysOnOSLog {
    static let subsystem = Bundle.main.bundleIdentifier ?? "com.paafektinnovations.Paafekt"
    static let sharp = Logger(subsystem: subsystem, category: "SHARP")
    static let wallMeas = Logger(subsystem: subsystem, category: "WALL_MEAS")
    static let arRoom = Logger(subsystem: subsystem, category: "AR_ROOM")
    static let depthPro = Logger(subsystem: subsystem, category: "DEPTH_PRO")
    static let memory = Logger(subsystem: subsystem, category: "MEMORY")
}

/// YOLO wall measurement on save. Filter: `WALL_MEAS` (matches Android `adb logcat | grep WALL_MEAS`).
func logWallMeasurement(_ message: String) {
    guard isDebugModeEnabled() else { return }
    let line = "[WALL_MEAS] \(message)"
    AlwaysOnOSLog.wallMeas.notice("\(line, privacy: .public)")
}

/// SHARP Core ML / pipeline milestones. Filter: category `SHARP` or search `[SHARP]`.
func logSharpMilestone(_ message: String) {
    guard isDebugModeEnabled() else { return }
    let line = "[SHARP] \(message)"
    AlwaysOnOSLog.sharp.notice("\(line, privacy: .public)")
}

/// Sharp Room: SHARP-derived room W×H×D diagnostics (post-extract, pre-save). Filter: `AR_ROOM` or `[AR_ROOM_MEASURE]`.
func logARRoomMeasure(_ message: String) {
    guard isDebugModeEnabled() else { return }
    let line = "[AR_ROOM_MEASURE] \(message)"
    AlwaysOnOSLog.arRoom.notice("\(line, privacy: .public)")
}

/// Depth Pro metric-depth pipeline. Filter: `DEPTH_PRO` or `[DEPTH_PRO]`.
func logDepthPro(_ message: String) {
    guard isDebugModeEnabled() else { return }
    let line = "[DEPTH_PRO] \(message.uppercased())"
    AlwaysOnOSLog.depthPro.notice("\(line, privacy: .public)")
}

private struct AppMemorySnapshot {
    let residentBytes: UInt64
    let physicalFootprintBytes: UInt64
    let deviceMemoryBytes: UInt64

    var residentMB: Double { Double(residentBytes) / 1024.0 / 1024.0 }
    var footprintMB: Double { Double(physicalFootprintBytes) / 1024.0 / 1024.0 }
    var estimatedRemainingMB: Double {
        Double(max(0, Int64(deviceMemoryBytes) - Int64(physicalFootprintBytes))) / 1024.0 / 1024.0
    }
}

private func currentAppMemorySnapshot() -> AppMemorySnapshot? {
    var basicInfo = mach_task_basic_info()
    var basicCount = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
    let basicResult = withUnsafeMutablePointer(to: &basicInfo) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(basicCount)) {
            task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &basicCount)
        }
    }
    guard basicResult == KERN_SUCCESS else { return nil }

    var vmInfo = task_vm_info_data_t()
    var vmCount = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
    let vmResult = withUnsafeMutablePointer(to: &vmInfo) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(vmCount)) {
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &vmCount)
        }
    }
    let footprintBytes = vmResult == KERN_SUCCESS ? UInt64(vmInfo.phys_footprint) : UInt64(basicInfo.resident_size)

    return AppMemorySnapshot(
        residentBytes: UInt64(basicInfo.resident_size),
        physicalFootprintBytes: footprintBytes,
        deviceMemoryBytes: ProcessInfo.processInfo.physicalMemory
    )
}

/// Lightweight memory diagnostics for tracing which stage caused pressure.
/// Filter by category `MEMORY` or search `[MEMORY]`.
func logMemorySnapshot(_ source: String, details: String? = nil) {
    guard isDebugModeEnabled() else { return }
    guard let snapshot = currentAppMemorySnapshot() else {
        let line = "[MEMORY] source=\(source) snapshot=unavailable"
        AlwaysOnOSLog.memory.notice("\(line, privacy: .public)")
        return
    }
    var line =
        "[MEMORY] source=\(source) " +
        "resident_mb=\(String(format: "%.1f", snapshot.residentMB)) " +
        "footprint_mb=\(String(format: "%.1f", snapshot.footprintMB)) " +
        "remaining_est_mb=\(String(format: "%.1f", snapshot.estimatedRemainingMB))"
    if let details, !details.isEmpty {
        line += " details=\(details)"
    }
    AlwaysOnOSLog.memory.notice("\(line, privacy: .public)")
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
