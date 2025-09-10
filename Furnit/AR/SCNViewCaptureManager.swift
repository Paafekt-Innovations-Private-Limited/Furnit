import SceneKit
import UIKit
import Combine

@MainActor
class SCNViewCaptureManager: NSObject, ObservableObject {
    // Published properties for reactive UI updates
    @Published var authorizationStatus: AuthorizationStatus = .ready
    @Published var isSessionRunning = false
    @Published var capturedImage: UIImage?
    @Published var errorMessage: String?
    
    // Authorization status for SCNView capture (always ready since no permissions needed)
    enum AuthorizationStatus {
        case ready
        case unavailable
    }
    
    // SCNView reference for capturing snapshots
    weak var sceneView: SCNView?
    
    override init() {
        super.init()
        // No authorization needed for SCNView snapshots
        authorizationStatus = .ready
        print("📷 SCNView capture manager initialized - ready for snapshots")
    }
    
    // Set the SCNView reference for snapshot capture
    func setSceneView(_ sceneView: SCNView) {
        self.sceneView = sceneView
        print("📷 SCNView reference set for snapshot capture")
    }
    
    // Check authorization status (always ready for SCNView snapshots)
    func checkAuthorizationStatus() {
        authorizationStatus = sceneView != nil ? .ready : .unavailable
        print("📷 Authorization status: \(authorizationStatus)")
    }
    
    // Start capture session (immediate for SCNView)
    func startSession() {
        guard let sceneView = sceneView,
              sceneView.scene != nil else {
            errorMessage = "Scene view not ready for capture"
            return
        }
        
        // Don't switch cameras - stay in room with current viewpoint
        isSessionRunning = true
        print("📷 SCNView capture session started - ready for snapshots from current viewpoint")
    }
    
    // Stop capture session
    func stopSession() {
        // No camera restoration needed since we don't switch cameras
        isSessionRunning = false
        capturedImage = nil
        errorMessage = nil
        print("📷 SCNView capture session stopped")
    }
    
    // Capture high-quality snapshot from SCNView
    func capturePhoto() async -> UIImage? {
        guard let sceneView = sceneView else {
            errorMessage = "Scene view not available for capture"
            return nil
        }
        
        guard sceneView.scene != nil else {
            errorMessage = "No scene loaded for capture"
            return nil
        }
        
        print("📷 Capturing snapshot from current room viewpoint...")
        
        // Capture snapshot synchronously from current camera position in room
        let capturedSnapshot = sceneView.snapshot()
        
        capturedImage = capturedSnapshot
        print("📷 Snapshot captured successfully from room view, size: \(capturedSnapshot.size)")
        return capturedSnapshot
    }
    
    // Support for multiple capture angles (future enhancement)
    func captureFromMultipleAngles() async -> [UIImage] {
        var capturedImages: [UIImage] = []
        
        // Capture from current position in room
        if let frontImage = await capturePhoto() {
            capturedImages.append(frontImage)
        }
        
        // Could add slight camera adjustments here for different angles
        // while keeping camera inside room boundaries
        
        return capturedImages
    }
}