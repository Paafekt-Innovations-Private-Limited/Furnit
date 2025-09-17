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
    
    // World anchor for gesture-based navigation
    private var worldAnchor: AnchorEntity?
    
    // Movement configuration
    private let movementSpeed: Float = 0.05 // Units per frame
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
        
        // Initialize boundary manager with the ARView
        boundaryManager = RealityKitBoundaryManager(arView: arView)
        
        // Don't create world anchor here - it will be shared via setWorldAnchor()
        // The world anchor will be created by gesture handlers first
        
        // Calculate room boundaries when scene is available
        if !arView.scene.anchors.isEmpty {
            print("🎬 Scene available, calculating boundaries...")
            if let firstAnchor = arView.scene.anchors.first,
               let modelEntity = findFirstModelEntity(in: firstAnchor) {
                boundaryManager?.calculateRoomBounds(from: modelEntity)
            }
        } else {
            print("⚠️ Scene not available yet when setting up camera movement manager")
        }
    }
    
    // Set up world anchor that will hold all content for transformation
    private func setupWorldAnchor() {
        guard let arView = arView else { return }
        
        // Create world anchor if it doesn't exist
        if worldAnchor == nil {
            worldAnchor = AnchorEntity(.world(transform: matrix_identity_float4x4))
            arView.scene.addAnchor(worldAnchor!)
            print("🌍 World anchor created for joystick-based navigation")
        }
    }
    
    // Add entity to world anchor for joystick control
    func addToWorld(_ entity: Entity) {
        // Ensure world anchor is set via setWorldAnchor() before using this method
        guard let worldAnchor = worldAnchor else {
            print("⚠️ World anchor not set - call setWorldAnchor() first")
            return
        }
        worldAnchor.addChild(entity)
    }
    
    // Set world anchor reference (to share with gesture handlers)
    func setWorldAnchor(_ anchor: AnchorEntity) {
        self.worldAnchor = anchor
        print("🎮 Camera movement manager now using shared world anchor")
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
    
    // Method to recalculate boundaries when scene becomes available
    func updateBoundaries() {
        guard let arView = arView else {
            print("⚠️ Cannot update boundaries - ARView not available")
            return
        }
        
        if !arView.scene.anchors.isEmpty {
            print("🔄 Updating camera movement boundaries...")
            if let firstAnchor = arView.scene.anchors.first,
               let modelEntity = findFirstModelEntity(in: firstAnchor) {
                boundaryManager?.calculateRoomBounds(from: modelEntity)
            }
        } else {
            print("⚠️ Cannot update boundaries - no anchors in scene")
        }
    }
    
    // Update joystick input from the virtual joystick
    func updateJoystickInput(_ offset: CGSize) {
        currentJoystickOffset = offset
    }
    
    // Continuous world position updates based on joystick input (inverse of camera movement)
    @objc private func updateCameraPosition() {
        guard let arView = arView, let worldAnchor = worldAnchor else { return }
        
        // Skip if no joystick input
        guard abs(currentJoystickOffset.width) > 1 || abs(currentJoystickOffset.height) > 1 else { return }
        
        // Convert joystick input to movement vectors
        let forwardBackward = Float(-currentJoystickOffset.height) * movementSpeed // Negative for intuitive forward movement
        let leftRight = Float(currentJoystickOffset.width) * movementSpeed
        
        // Get camera's current transform for directional movement
        let cameraTransform = arView.cameraTransform
        
        // Extract camera's forward and right vectors from transform
        // In RealityKit, the camera looks down the negative Z axis by default
        let forwardVector = SIMD3<Float>(
            -cameraTransform.rotation.act(SIMD3<Float>(0, 0, -1)).x, // Forward X component
            0, // Keep movement horizontal (no Y movement)
            -cameraTransform.rotation.act(SIMD3<Float>(0, 0, -1)).z  // Forward Z component
        )
        
        let rightVector = SIMD3<Float>(
            cameraTransform.rotation.act(SIMD3<Float>(1, 0, 0)).x, // Right X component  
            0, // Keep movement horizontal (no Y movement)
            cameraTransform.rotation.act(SIMD3<Float>(1, 0, 0)).z  // Right Z component
        )
        
        // Normalize vectors for consistent movement speed
        let normalizedForward = normalize(forwardVector)
        let normalizedRight = normalize(rightVector)
        
        // Calculate movement delta based on joystick input (inverse for world movement)
        let worldMovementDelta = SIMD3<Float>(
            -(normalizedForward.x * forwardBackward + normalizedRight.x * leftRight),
            0, // No vertical movement
            -(normalizedForward.z * forwardBackward + normalizedRight.z * leftRight)
        )
        
        // Get current world anchor position
        let currentWorldPosition = worldAnchor.transform.translation
        
        // Calculate proposed new world position
        let proposedWorldPosition = SIMD3<Float>(
            currentWorldPosition.x + worldMovementDelta.x,
            currentWorldPosition.y, // Keep same height
            currentWorldPosition.z + worldMovementDelta.z
        )
        
        // Apply boundary constraints (simulate camera constraint by checking effective camera position)
        let effectiveCameraPos = cameraTransform.translation - worldMovementDelta
        let constrainedCameraPos = boundaryManager?.constrainCameraPosition(effectiveCameraPos) ?? effectiveCameraPos
        let constraintDelta = constrainedCameraPos - cameraTransform.translation
        let constrainedWorldPosition = currentWorldPosition - constraintDelta
        
        // Debug logging to understand boundary constraints
        if proposedWorldPosition.x != constrainedWorldPosition.x || proposedWorldPosition.z != constrainedWorldPosition.z {
            print("🚧 Wall hit! World movement constrained")
        }
        
        // Only update world position if it's different (prevents unnecessary updates)
        if constrainedWorldPosition.x != currentWorldPosition.x || 
           constrainedWorldPosition.z != currentWorldPosition.z {
            
            // Create new transform with updated position
            var newTransform = worldAnchor.transform
            newTransform.translation = constrainedWorldPosition
            
            // Apply the new world transform
            worldAnchor.transform = newTransform
            
            // Notify that camera has moved
            onCameraMove?()
        }
    }
    
    // Reset world to default position (simulates camera reset)
    func resetCameraPosition() {
        guard let arView = arView, let worldAnchor = worldAnchor else { return }
        
        // Reset world anchor to identity transform with animation
        UIView.animate(withDuration: 0.5) {
            worldAnchor.transform = Transform(.identity)
        }
        
        print("📷 World reset to default position (camera view reset)")
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