import Foundation
import ARKit
import UIKit
import simd

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
    var initialTransform: simd_float4x4?
    var onRelativePoseUpdate: ((simd_float4x4) -> Void)?
    var onTrackingStatus: ((String) -> Void)?
    /// Debug: log at most once per second while waiting for `.normal` before first reference.
    private var lastLimitedInitialLogTime: CFAbsoluteTime = 0

    func start() {
        guard ARWorldTrackingConfiguration.isSupported else {
            onTrackingStatus?("AR motion tracking unavailable on this device")
            logDebug("❌ [ARMotionTracker] world tracking unsupported")
            return
        }
        let config = ARWorldTrackingConfiguration()
        config.planeDetection = []
        config.isLightEstimationEnabled = false
        session.delegate = self
        initialTransform = nil
        lastLimitedInitialLogTime = 0
        logDebug("🚀 [ARMotionTracker] starting tracking-only AR session")
        session.run(config, options: [.resetTracking, .removeExistingAnchors])
    }

    func stop() {
        logDebug("🛑 [ARMotionTracker] stopping tracking-only AR session")
        session.pause()
        initialTransform = nil
    }

    /// Pauses frame delivery so ARKit stops flooding the main queue (SwiftUI alerts / `TextField` stay responsive).
    func pauseForModal() {
        session.pause()
        logDebug("⏸️ [ARMotionTracker] session paused for modal UI")
    }

    /// Resumes after ``pauseForModal`` without resetting the world map (pair with Sharp Room modal dismiss).
    func resumeAfterModal() {
        guard ARWorldTrackingConfiguration.isSupported else { return }
        let config = ARWorldTrackingConfiguration()
        config.planeDetection = []
        config.isLightEstimationEnabled = false
        session.run(config, options: [])
        logDebug("▶️ [ARMotionTracker] session resumed after modal UI")
    }

    /// Clears the stored reference pose so the next `didUpdate` uses the current device pose as identity (Sharp Room recenter).
    func resetReferencePose() {
        initialTransform = nil
        lastLimitedInitialLogTime = 0
        logDebug("📍 [ARMotionTracker] reference pose cleared — next frame becomes new origin (after .normal)")
    }

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        // Copy pose state immediately; never retain `ARFrame` past this method (no async captures, no stored frame).
        let currentTransform = frame.camera.transform
        let trackingState = frame.camera.trackingState
        if initialTransform == nil {
            // Capturing during `.limited(.initializing)` bakes a bad reference and the room stays tilted.
            guard case .normal = trackingState else {
                let now = CFAbsoluteTimeGetCurrent()
                if now - lastLimitedInitialLogTime > 1.0 {
                    lastLimitedInitialLogTime = now
                    logDebug(
                        "📍 [ARMotionTracker] deferring initial reference — trackingState=\(trackingState) " +
                            "(wait for .normal)"
                    )
                }
                return
            }
            initialTransform = currentTransform
            logDebug("📍 [ARMotionTracker] captured initial camera transform (trackingState=normal)")
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
        onRelativePoseUpdate?(relativeTransform)
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

    func session(_ session: ARSession, didFailWithError error: Error) {
        let message = "AR tracking failed: \(error.localizedDescription)"
        onTrackingStatus?(message)
        logDebug("❌ [ARMotionTracker] \(message)")
    }
}
