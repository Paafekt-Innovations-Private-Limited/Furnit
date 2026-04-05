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
        logDebug("🚀 [ARMotionTracker] starting tracking-only AR session")
        session.run(config, options: [.resetTracking, .removeExistingAnchors])
    }

    func stop() {
        logDebug("🛑 [ARMotionTracker] stopping tracking-only AR session")
        session.pause()
        initialTransform = nil
    }

    /// Clears the stored reference pose so the next `didUpdate` uses the current device pose as identity (Sharp Room recenter).
    func resetReferencePose() {
        initialTransform = nil
        logDebug("📍 [ARMotionTracker] reference pose cleared — next frame becomes new origin")
    }

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        let currentTransform = frame.camera.transform
        if initialTransform == nil {
            initialTransform = currentTransform
            logDebug("📍 [ARMotionTracker] captured initial camera transform")
        }
        guard let initialTransform else { return }
        let relativeTransform = simd_mul(simd_inverse(initialTransform), currentTransform)
        onRelativePoseUpdate?(relativeTransform)
    }

    func session(_ session: ARSession, cameraDidChangeTrackingState camera: ARCamera) {
        let status: String
        switch camera.trackingState {
        case .normal:
            status = "AR tracking normal"
        case .notAvailable:
            status = "AR tracking unavailable"
        case .limited(let reason):
            status = "AR tracking limited: \(reason)"
        }
        onTrackingStatus?(status)
        logDebug("📷 [ARMotionTracker] \(status)")
    }

    func session(_ session: ARSession, didFailWithError error: Error) {
        let message = "AR tracking failed: \(error.localizedDescription)"
        onTrackingStatus?(message)
        logDebug("❌ [ARMotionTracker] \(message)")
    }
}
