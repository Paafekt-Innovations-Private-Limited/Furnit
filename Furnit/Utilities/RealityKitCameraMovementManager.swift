import RealityKit
import ARKit
import SwiftUI
import simd

class RealityKitCameraMovementManager: ObservableObject {
    // Camera movement properties
    weak var arView: ARView?
    private var boundaryManager: RealityKitBoundaryManager?
    private var displayLink: CADisplayLink?
    private var currentJoystickOffset: CGSize = .zero
    
    // Custom camera control for non-AR mode
    private weak var cameraAnchor: AnchorEntity?
    
    // Movement configuration
    private var movementSpeed: Float = 0.003 // Units per frame - much slower for precise control
    private let smoothingFactor: Float = 0.8 // Movement smoothing (0.0 = instant, 1.0 = no movement)
    
    // Callback for camera movement notifications
    var onCameraMove: (() -> Void)?
    
    init() {
        // Set up display link for smooth continuous movement
        displayLink = CADisplayLink(target: self, selector: #selector(updateCameraPosition))
        displayLink?.add(to: .main, forMode: .common)
    }
    
    deinit {
        // Clean up display link
        displayLink?.invalidate()
    }
    
    // Set the ARView reference for camera manipulation
    func setARView(_ arView: ARView) {
        self.arView = arView
        print("📱 Camera movement manager ARView set")
    }
    
    // Set camera anchor reference for direct camera control
    func setCameraAnchor(_ anchor: AnchorEntity) {
        self.cameraAnchor = anchor
        print("🎮 Camera movement manager now using camera anchor for direct control")
    }

    // Set boundary manager reference (shared from RealityKitView)
    func setBoundaryManager(_ manager: RealityKitBoundaryManager) {
        self.boundaryManager = manager
        print("🏠 Camera movement manager using shared boundary manager")
    }

    // Update movement speed from settings
    func updateMovementSpeed(_ speed: MovementSpeed) {
        movementSpeed = speed.speedValue
        print("🏃 Camera movement speed updated to: \(speed.displayName) (\(speed.speedValue))")
    }
    
    // Helper to find first model entity in anchor hierarchy
    private func findFirstModelEntity(in entity: Entity) -> Entity? {
        if entity.components.has(ModelComponent.self) {
            return entity
        }
        
        for child in entity.children {
            if let found = findFirstModelEntity(in: child) {
                return found
            }
        }
        
        return nil
    }
    
    // Method to recalculate boundaries when scene becomes available (now handled by shared boundary manager)
    func updateBoundaries() {
        print("🔄 Boundary updates now handled by shared boundary manager")
    }
    
    // Update joystick input from the virtual joystick
    func updateJoystickInput(_ offset: CGSize) {
        currentJoystickOffset = offset
    }
    
    // Continuous camera position updates based on joystick input (direct camera movement)
    @objc private func updateCameraPosition() {
        guard let _ = arView, let cameraAnchor = cameraAnchor else { return }
        
        // Skip if no joystick input
        guard abs(currentJoystickOffset.width) > 1 || abs(currentJoystickOffset.height) > 1 else { return }
        
        // Convert joystick input to movement vectors
        let forwardBackward = Float(-currentJoystickOffset.height) * movementSpeed // Negative for intuitive forward movement
        let leftRight = Float(currentJoystickOffset.width) * movementSpeed
        
        // Get camera's current transform for directional movement
        let cameraTransform = cameraAnchor.transform
        
        // Extract camera's forward and right vectors from camera's rotation
        // In RealityKit, the camera looks down the negative Z axis by default
        let forwardVector = SIMD3<Float>(
            cameraTransform.rotation.act(SIMD3<Float>(0, 0, -1)).x, // Forward X component
            0, // Keep movement horizontal (no Y movement)
            cameraTransform.rotation.act(SIMD3<Float>(0, 0, -1)).z  // Forward Z component
        )
        
        let rightVector = SIMD3<Float>(
            cameraTransform.rotation.act(SIMD3<Float>(1, 0, 0)).x, // Right X component  
            0, // Keep movement horizontal (no Y movement)
            cameraTransform.rotation.act(SIMD3<Float>(1, 0, 0)).z  // Right Z component
        )
        
        // Normalize vectors for consistent movement speed
        let normalizedForward = normalize(forwardVector)
        let normalizedRight = normalize(rightVector)
        
        // Calculate movement delta based on joystick input (direct camera movement)
        let cameraMovementDelta = SIMD3<Float>(
            normalizedForward.x * forwardBackward + normalizedRight.x * leftRight,
            0, // No vertical movement
            normalizedForward.z * forwardBackward + normalizedRight.z * leftRight
        )
        
        // Get current camera position
        let currentCameraPosition = cameraAnchor.transform.translation
        
        // Calculate proposed new camera position
        var proposedCameraPosition = SIMD3<Float>(
            currentCameraPosition.x + cameraMovementDelta.x,
            currentCameraPosition.y, // Keep same height
            currentCameraPosition.z + cameraMovementDelta.z
        )
        
        // Apply boundary constraints if boundary manager is available
        if let boundaryManager = boundaryManager {
            let originalProposedPosition = proposedCameraPosition
            proposedCameraPosition = boundaryManager.constrainCameraPosition(proposedCameraPosition)

            // Debug logging when movement is constrained
            if proposedCameraPosition.x != originalProposedPosition.x ||
               proposedCameraPosition.z != originalProposedPosition.z {
                print("🚧 Wall hit! Camera movement constrained from \(originalProposedPosition) to \(proposedCameraPosition)")
            }
        } else {
            print("⚠️ No boundary manager available - camera movement not constrained")
        }
        
        // Only update camera position if it's different (prevents unnecessary updates)
        if proposedCameraPosition.x != currentCameraPosition.x || 
           proposedCameraPosition.z != currentCameraPosition.z {
            
            // Create new transform preserving current rotation but updating position
            var newTransform = cameraAnchor.transform
            newTransform.translation = proposedCameraPosition
            
            // Apply the new camera transform (position only, rotation preserved)
            cameraAnchor.transform = newTransform
            
            // Notify that camera has moved
            onCameraMove?()
        }
    }
    
    // Reset camera to default position and orientation
    func resetCameraPosition() {
        guard let _ = arView, let cameraAnchor = cameraAnchor else { return }
        
        // Reset camera anchor to default transform with animation at eye level
        UIView.animate(withDuration: 0.5) {
            cameraAnchor.transform = Transform(rotation: simd_quatf(angle: 0, axis: SIMD3<Float>(0, 1, 0)), translation: SIMD3<Float>(0, 1.2, 3))
        }
        
        print("📷 Camera reset to default position and orientation")
    }
    
    // Enable/disable camera movement
    func setCameraMovementEnabled(_ enabled: Bool) {
        displayLink?.isPaused = !enabled
        print("🎮 Camera movement \(enabled ? "enabled" : "disabled")")
    }
}

// MARK: - Helper functions

private func normalize(_ vector: SIMD3<Float>) -> SIMD3<Float> {
    let length = sqrt(vector.x * vector.x + vector.y * vector.y + vector.z * vector.z)
    return length > 0 ? SIMD3<Float>(vector.x / length, vector.y / length, vector.z / length) : vector
}