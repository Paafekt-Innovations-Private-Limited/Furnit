import Foundation
import ARKit
import UIKit
import simd

struct ARPlaneRoomMeasurement: Equatable, Sendable {
    let widthMeters: Float
    let heightMeters: Float
    let depthMeters: Float
    let floorAnchorCount: Int
    let wallAnchorCount: Int

    var sourceLabel: String { "ARKit_plane_detection_m_W×H×D" }
}

struct SharpRoomFurnitureItem: Identifiable, Equatable {
    let id = UUID()
    let category: String
    let dimensions: SIMD3<Float>
    let tint: UIColor
}

struct SharpRoomPlacedFurniture: Identifiable {
    let id: UUID
    let item: SharpRoomFurnitureItem
    var position: SIMD3<Float>
    var rotationY: Float
    var fits: Bool
    var clearanceMeters: Float
}

enum SharpRoomFurnitureCatalog {
    static let standardItems: [SharpRoomFurnitureItem] = [
        SharpRoomFurnitureItem(category: "Sofa", dimensions: SIMD3<Float>(1.85, 0.85, 0.90), tint: .systemGreen),
        SharpRoomFurnitureItem(category: "Bed", dimensions: SIMD3<Float>(2.00, 0.60, 1.50), tint: .systemBlue),
        SharpRoomFurnitureItem(category: "Table", dimensions: SIMD3<Float>(1.50, 0.75, 0.90), tint: .systemOrange),
        SharpRoomFurnitureItem(category: "Chair", dimensions: SIMD3<Float>(0.50, 0.85, 0.50), tint: .systemTeal),
        SharpRoomFurnitureItem(category: "Wardrobe", dimensions: SIMD3<Float>(1.50, 2.00, 0.60), tint: .systemPurple),
        SharpRoomFurnitureItem(category: "Desk", dimensions: SIMD3<Float>(1.20, 0.75, 0.60), tint: .systemPink),
    ]
}

final class ARMotionTracker: NSObject, ARSessionDelegate {
    let session = ARSession()
    private let sessionDelegateQueue = DispatchQueue(label: "com.furnit.sharproom.ar-motion", qos: .userInitiated)
    var initialTransform: simd_float4x4?
    var onRelativePoseUpdate: ((simd_float4x4) -> Void)?
    var onTrackingStatus: ((String) -> Void)?
    var onPlaneRoomMeasurementUpdate: ((ARPlaneRoomMeasurement?) -> Void)?
    /// Debug: log at most once per second while waiting for `.normal` before first reference.
    private var lastLimitedInitialLogTime: CFAbsoluteTime = 0
    private var planeAnchorsByID: [UUID: ARPlaneAnchor] = [:]
    private var lastPublishedPlaneMeasurement: ARPlaneRoomMeasurement?

    /// Sharp Live Room: 6DoF pose plus ARKit plane detection.
    static func makeLiveRoomConfiguration() -> ARWorldTrackingConfiguration {
        let config = ARWorldTrackingConfiguration()
        config.worldAlignment = .gravity
        config.planeDetection = [.horizontal, .vertical]
        config.isLightEstimationEnabled = false
        return config
    }

    func start() {
        guard ARWorldTrackingConfiguration.isSupported else {
            onTrackingStatus?("AR motion tracking unavailable on this device")
            logDebug("❌ [ARMotionTracker] world tracking unsupported")
            return
        }
        let config = Self.makeLiveRoomConfiguration()
        session.delegate = self
        session.delegateQueue = sessionDelegateQueue
        initialTransform = nil
        lastLimitedInitialLogTime = 0
        planeAnchorsByID.removeAll()
        publishPlaneRoomMeasurement(nil)
        CameraOwnershipDiagnostics.log(owner: "ARMotionTracker", event: "ar_run", details: "reason=start")
        logDebug("🚀 [ARMotionTracker] starting Live Room AR session worldAlignment=gravity planeDetection=horizontal+vertical frameSemantics=none")
        session.run(config, options: [.resetTracking, .removeExistingAnchors])
    }

    func stop() {
        CameraOwnershipDiagnostics.log(owner: "ARMotionTracker", event: "ar_pause", details: "reason=stop")
        logDebug("🛑 [ARMotionTracker] stopping tracking-only AR session")
        session.pause()
        initialTransform = nil
        planeAnchorsByID.removeAll()
        publishPlaneRoomMeasurement(nil)
    }

    /// Pauses frame delivery so ARKit stops flooding the main queue (SwiftUI alerts / `TextField` stay responsive).
    func pauseForModal() {
        CameraOwnershipDiagnostics.log(owner: "ARMotionTracker", event: "ar_pause", details: "reason=modal_or_camera_handoff")
        session.pause()
        logDebug("⏸️ [ARMotionTracker] session paused for modal UI")
    }

    /// Resumes after ``pauseForModal`` without resetting the world map (pair with Sharp Room modal dismiss).
    func resumeAfterModal() {
        guard ARWorldTrackingConfiguration.isSupported else { return }
        let config = Self.makeLiveRoomConfiguration()
        CameraOwnershipDiagnostics.log(owner: "ARMotionTracker", event: "ar_run", details: "reason=resume")
        session.run(config, options: [])
        logDebug("▶️ [ARMotionTracker] session resumed after modal UI")
    }

    deinit {
        CameraOwnershipDiagnostics.log(owner: "ARMotionTracker", event: "deinit")
    }

    /// Clears the stored reference pose so the next `didUpdate` uses the current device pose as identity (Sharp Room recenter).
    func resetReferencePose() {
        initialTransform = nil
        // Do **not** set `lastLimitedInitialLogTime = 0`: `now - 0` is huge, so the defer log throttle
        // would pass on every frame while waiting for `.normal`. Refresh the window from "now" instead.
        lastLimitedInitialLogTime = CFAbsoluteTimeGetCurrent()
        logDebug("📍 [ARMotionTracker] reference pose cleared — next frame becomes new origin (after .normal)")
    }

    private func isFloorPlane(_ plane: ARPlaneAnchor) -> Bool {
        switch plane.classification {
        case .floor:
            return true
        case .none:
            return plane.alignment == .horizontal
        default:
            return false
        }
    }

    private func isWallPlane(_ plane: ARPlaneAnchor) -> Bool {
        switch plane.classification {
        case .wall:
            return true
        case .none:
            return plane.alignment == .vertical
        default:
            return false
        }
    }

    private func publishPlaneRoomMeasurement(_ measurement: ARPlaneRoomMeasurement?) {
        guard lastPublishedPlaneMeasurement != measurement else { return }
        lastPublishedPlaneMeasurement = measurement
        DispatchQueue.main.async { [weak self] in
            self?.onPlaneRoomMeasurementUpdate?(measurement)
        }
    }

    private func recomputePlaneRoomMeasurement() {
        let floorPlanes = planeAnchorsByID.values.filter(isFloorPlane)
        let wallPlanes = planeAnchorsByID.values.filter(isWallPlane)

        guard let bestFloor = floorPlanes.max(by: {
            ($0.extent.x * $0.extent.z) < ($1.extent.x * $1.extent.z)
        }) else {
            publishPlaneRoomMeasurement(nil)
            return
        }

        let wallHeights = wallPlanes.map { $0.extent.y }.filter { $0.isFinite && $0 > 0.2 }
        guard let bestWallHeight = wallHeights.max(), bestWallHeight > 0.5 else {
            publishPlaneRoomMeasurement(nil)
            return
        }

        let measurement = ARPlaneRoomMeasurement(
            widthMeters: bestFloor.extent.x,
            heightMeters: bestWallHeight,
            depthMeters: bestFloor.extent.z,
            floorAnchorCount: floorPlanes.count,
            wallAnchorCount: wallPlanes.count
        )
        logDebug(
            "📐 [AR_MEASURE] authoritative room from planes W×H×D=" +
                "\(String(format: "%.3f", measurement.widthMeters))×" +
                "\(String(format: "%.3f", measurement.heightMeters))×" +
                "\(String(format: "%.3f", measurement.depthMeters))m " +
                "floors=\(measurement.floorAnchorCount) walls=\(measurement.wallAnchorCount)"
        )
        publishPlaneRoomMeasurement(measurement)
    }

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        // Copy pose state immediately; never retain `ARFrame` past this method (no async captures, no stored frame).
        let currentTransform = frame.camera.transform
        let trackingState = frame.camera.trackingState
        if initialTransform == nil {
            // Sharp Live Room uses pose as a motion hint over the splat; IMU-backed poses are still useful
            // when visual tracking is `.limited` (e.g. phone aimed at the on-screen room). Only `.notAvailable`
            // has no usable 6DoF.
            if case .notAvailable = trackingState {
                let now = CFAbsoluteTimeGetCurrent()
                if now - lastLimitedInitialLogTime > 1.0 {
                    lastLimitedInitialLogTime = now
                    logDebug(
                        "📍 [ARMotionTracker] waiting for pose — trackingState=\(trackingState)"
                    )
                }
                return
            }
            initialTransform = currentTransform
            logDebug("📍 [ARMotionTracker] captured initial camera transform (trackingState=\(trackingState))")
        }
        guard let initialTransform else { return }
        // Full device rotation (tilt to look at furniture). Position: floor height locked to reference Y,
        // and horizontal motion projected onto opening **forward** (XZ) so walking reads as pinch-style
        // dolly into/out of the scene — not lateral “one-finger pan” strafe from sideways steps.
        let p0 = SIMD3<Float>(
            initialTransform.columns.3.x,
            initialTransform.columns.3.y,
            initialTransform.columns.3.z
        )
        let p1 = SIMD3<Float>(
            currentTransform.columns.3.x,
            currentTransform.columns.3.y,
            currentTransform.columns.3.z
        )
        let deltaH = SIMD3<Float>(p1.x - p0.x, 0, p1.z - p0.z)
        var forwardXZ = SIMD3<Float>(
            -initialTransform.columns.2.x,
            0,
            -initialTransform.columns.2.z
        )
        let forwardLen = simd_length(forwardXZ)
        if forwardLen < 1e-4 {
            forwardXZ = SIMD3<Float>(0, 0, -1)
        } else {
            forwardXZ /= forwardLen
        }
        let alongForward = simd_dot(deltaH, forwardXZ)
        let dollyH = forwardXZ * alongForward
        let pSynth = SIMD3<Float>(p0.x + dollyH.x, p0.y, p0.z + dollyH.z)

        var floorPlaneCamera = currentTransform
        floorPlaneCamera.columns.3 = SIMD4<Float>(pSynth.x, pSynth.y, pSynth.z, 1)
        let relativeTransform = simd_mul(simd_inverse(initialTransform), floorPlaneCamera)
        // Always hop to main: never run consumer work synchronously in `didUpdate` — a slow callback
        // retains many `ARFrame`s and triggers "delegate is retaining N ARFrames" from ARKit.
        DispatchQueue.main.async { [weak self] in
            self?.onRelativePoseUpdate?(relativeTransform)
        }
    }

    func session(_ session: ARSession, cameraDidChangeTrackingState camera: ARCamera) {
        let status: String
        switch camera.trackingState {
        case .normal:
            status = "AR tracking normal"
        case .notAvailable:
            status = "AR tracking unavailable"
        case .limited:
            status = "AR tracking limited"
        }
        onTrackingStatus?(status)
        logDebug("📷 [ARMotionTracker] \(status) — trackingState=\(camera.trackingState)")
    }

    func session(_ session: ARSession, didAdd anchors: [ARAnchor]) {
        var changed = false
        for anchor in anchors {
            guard let plane = anchor as? ARPlaneAnchor else { continue }
            planeAnchorsByID[plane.identifier] = plane
            changed = true
        }
        if changed {
            recomputePlaneRoomMeasurement()
        }
    }

    func session(_ session: ARSession, didUpdate anchors: [ARAnchor]) {
        var changed = false
        for anchor in anchors {
            guard let plane = anchor as? ARPlaneAnchor else { continue }
            planeAnchorsByID[plane.identifier] = plane
            changed = true
        }
        if changed {
            recomputePlaneRoomMeasurement()
        }
    }

    func session(_ session: ARSession, didRemove anchors: [ARAnchor]) {
        var changed = false
        for anchor in anchors {
            guard anchor is ARPlaneAnchor else { continue }
            planeAnchorsByID.removeValue(forKey: anchor.identifier)
            changed = true
        }
        if changed {
            recomputePlaneRoomMeasurement()
        }
    }

    func session(_ session: ARSession, didFailWithError error: Error) {
        let message = "AR tracking failed: \(error.localizedDescription)"
        onTrackingStatus?(message)
        logDebug("❌ [ARMotionTracker] \(message)")
    }
}
