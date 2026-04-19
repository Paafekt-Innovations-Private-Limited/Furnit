import Foundation
import os.log
import Darwin

// MARK: - Runtime Debug Mode Check (cached)
/// Cached mirror of `UserDefaults.standard.bool(forKey: "debug_mode")` so log gates
/// don't pay a UserDefaults dictionary lookup on every call (matters in per-frame paths).
///
/// The cache is refreshed:
///   * once at process start (cheap, lazy on first read)
///   * whenever any UserDefaults key changes (`UserDefaults.didChangeNotification`)
///
/// We deliberately avoid coupling to `AppStateManager` / `QualitySettings` so this stays
/// usable from initializers and background queues without circular dependencies.
private final class DebugModeWatcher: @unchecked Sendable {
    static let shared = DebugModeWatcher()

    private let lock = NSLock()
    private var cachedEnabled: Bool

    private init() {
        cachedEnabled = UserDefaults.standard.bool(forKey: "debug_mode")
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handleDefaultsChanged),
            name: UserDefaults.didChangeNotification,
            object: nil
        )
    }

    @objc private func handleDefaultsChanged() {
        let value = UserDefaults.standard.bool(forKey: "debug_mode")
        lock.lock()
        cachedEnabled = value
        lock.unlock()
    }

    var isEnabled: Bool {
        lock.lock()
        let value = cachedEnabled
        lock.unlock()
        return value
    }
}

@inline(__always)
private func isDebugModeEnabled() -> Bool {
    DebugModeWatcher.shared.isEnabled
}

// MARK: - Global Debug Print Function
/// Use this instead of `print()` — only logs when debug mode is enabled in Settings.
///
/// The message is `@autoclosure` so string interpolation is **not evaluated** when debug mode is off.
/// This makes calls in hot paths (per-frame logging) effectively free in release / when toggled off.
///
/// Named `logDebug` to avoid conflict with Swift's `debugPrint`.
@inline(__always)
func logDebug(_ message: @autoclosure () -> String) {
    guard isDebugModeEnabled() else { return }
    print(message())
}

// MARK: - Throttled debug logging
/// Per-key throttle state for `logDebugThrottled`. Lives outside the function so calls accumulate state.
private final class LogThrottleState: @unchecked Sendable {
    static let shared = LogThrottleState()
    private let lock = NSLock()
    private var lastEmittedAt: [String: CFAbsoluteTime] = [:]

    func shouldEmit(key: String, every interval: CFTimeInterval, now: CFAbsoluteTime) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        if let prev = lastEmittedAt[key], now - prev < interval {
            return false
        }
        lastEmittedAt[key] = now
        return true
    }
}

/// Emit a debug log at most once per `interval` seconds for a given `key`.
///
/// Use in hot loops where the same diagnostic would otherwise flood the console
/// (e.g. per-frame inference timings). Still gated by Settings → Debug Mode.
@inline(__always)
func logDebugThrottled(_ key: String, every interval: CFTimeInterval, _ message: @autoclosure () -> String) {
    guard isDebugModeEnabled() else { return }
    let now = CFAbsoluteTimeGetCurrent()
    guard LogThrottleState.shared.shouldEmit(key: key, every: interval, now: now) else { return }
    print(message())
}

// MARK: - Furniture Fit & PLY stdout (Settings → Debug mode only)
/// PLY bounds / navigation. Filter: `PLY_BOUNDS`.
@inline(__always)
func logPlyBoundsDiagnostic(_ message: @autoclosure () -> String) {
    guard isDebugModeEnabled() else { return }
    print("[PLY_BOUNDS] \(message())")
}

/// AR-assisted FurnitureFit metrics. Filter: `FurnitureFitAR`.
///
/// Permanently silenced (per-frame overhead): re-enable by uncommenting the body
/// or by switching to `logDebugThrottled("FurnitureFitAR", every: 1.0, ...)`.
@inline(__always)
func logFurnitureFitAR(_ message: @autoclosure () -> String) {
    _ = message  // intentionally unused
}

/// Furniture W×H estimate and pipeline tags. Filter: `FurnitureFitSize`.
@inline(__always)
func logFurnitureFitSize(_ message: @autoclosure () -> String) {
    _ = message  // intentionally unused (silenced; see logFurnitureFitAR)
}

/// Overlay scale assist. Filter: `FurnitureFitOverlay`.
@inline(__always)
func logFurnitureFitOverlay(_ message: @autoclosure () -> String) {
    _ = message  // intentionally unused (silenced; see logFurnitureFitAR)
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
@inline(__always)
func logWallMeasurement(_ message: @autoclosure () -> String) {
    guard isDebugModeEnabled() else { return }
    let line = "[WALL_MEAS] \(message())"
    AlwaysOnOSLog.wallMeas.notice("\(line, privacy: .public)")
}

/// SHARP Core ML / pipeline milestones. Filter: category `SHARP` or search `[SHARP]`.
@inline(__always)
func logSharpMilestone(_ message: @autoclosure () -> String) {
    guard isDebugModeEnabled() else { return }
    let line = "[SHARP] \(message())"
    AlwaysOnOSLog.sharp.notice("\(line, privacy: .public)")
}

/// Sharp Room: SHARP-derived room W×H×D diagnostics (post-extract, pre-save). Filter: `AR_ROOM` or `[AR_ROOM_MEASURE]`.
@inline(__always)
func logARRoomMeasure(_ message: @autoclosure () -> String) {
    guard isDebugModeEnabled() else { return }
    let line = "[AR_ROOM_MEASURE] \(message())"
    AlwaysOnOSLog.arRoom.notice("\(line, privacy: .public)")
}

/// Depth Pro metric-depth pipeline. Filter: `DEPTH_PRO` or `[DEPTH_PRO]`.
@inline(__always)
func logDepthPro(_ message: @autoclosure () -> String) {
    guard isDebugModeEnabled() else { return }
    let line = "[DEPTH_PRO] \(message().uppercased())"
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

    /// Debug log - only appears when debug mode is enabled in Settings.
    /// `message` is `@autoclosure` so interpolation is skipped when off.
    static func debug(_ message: @autoclosure () -> String, category: Logger = general) {
        guard isDebugModeEnabled() else { return }
        let rendered = message()
        category.debug("\(rendered, privacy: .public)")
    }

    /// Info log - only appears when debug mode is enabled in Settings.
    static func info(_ message: @autoclosure () -> String, category: Logger = general) {
        guard isDebugModeEnabled() else { return }
        let rendered = message()
        category.info("\(rendered, privacy: .public)")
    }

    /// Warning log - only appears when debug mode is enabled in Settings.
    static func warning(_ message: @autoclosure () -> String, category: Logger = general) {
        guard isDebugModeEnabled() else { return }
        let rendered = message()
        category.warning("\(rendered, privacy: .public)")
    }

    // MARK: - Production Logging (always logged for crash reporting)

    /// Error log - ALWAYS appears (for crash reporting).
    static func error(_ message: @autoclosure () -> String, category: Logger = general) {
        let rendered = message()
        category.error("\(rendered, privacy: .public)")
    }

    /// Critical/Fault log - ALWAYS appears (for crash reporting).
    static func critical(_ message: @autoclosure () -> String, category: Logger = general) {
        let rendered = message()
        category.critical("\(rendered, privacy: .public)")
    }
}

// MARK: - Convenience Extensions

extension AppLogger {
    /// Log authentication events
    static func authDebug(_ message: @autoclosure () -> String) {
        debug(message(), category: auth)
    }

    static func authError(_ message: @autoclosure () -> String) {
        error(message(), category: auth)
    }

    /// Log UI events
    static func uiDebug(_ message: @autoclosure () -> String) {
        debug(message(), category: ui)
    }

    /// Log model/file operations
    static func modelDebug(_ message: @autoclosure () -> String) {
        debug(message(), category: model)
    }

    static func modelError(_ message: @autoclosure () -> String) {
        error(message(), category: model)
    }

    /// Log scene/3D operations
    static func sceneDebug(_ message: @autoclosure () -> String) {
        debug(message(), category: scene)
    }

    /// Log room processing
    static func roomDebug(_ message: @autoclosure () -> String) {
        debug(message(), category: room)
    }

    static func roomError(_ message: @autoclosure () -> String) {
        error(message(), category: room)
    }

    /// Log camera operations
    static func cameraDebug(_ message: @autoclosure () -> String) {
        debug(message(), category: camera)
    }
}
