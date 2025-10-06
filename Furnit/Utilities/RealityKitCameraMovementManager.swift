import RealityKit
import ARKit
import SwiftUI
import simd

class RealityKitCameraMovementManager: ObservableObject {
    // Define MovementSpeed enum locally with BALANCED values for smooth movement
    enum MovementSpeed: Float {
        case slow = 0.0016   // 2x original for gentle movement
        case normal = 0.003  // 2x original for comfortable cruising
        case fast = 0.006    // 2x original for quicker navigation
        case dollhouse = 0.008 // Special mode optimized for indoor dollhouse exploration
    }
    
    // Camera movement properties
    var arView: ARView?
    private var boundaryManager: RealityKitBoundaryManager?
    private var displayLink: CADisplayLink?
    var currentJoystickOffset: CGSize = .zero // Make public for simple movement system
    
    // Custom camera control for non-AR mode
    private var cameraAnchor: AnchorEntity?
    
    // Current speed setting
    @Published var currentSpeed: MovementSpeed = .normal
    
    // Ready state for joystick
    @Published var isReady: Bool = false
    
    // Movement configuration - BALANCED FOR SMOOTH MOVEMENT
    private var movementSpeed: Float = 0.04 // Moderate speed for smooth control
    private let joystickSensitivity: Float = 0.001 // Keep original sensitivity
    private let joystickDeadZone: Float = 5.0 // Dead zone threshold
    
    // Smooth movement
    private var targetVelocity: SIMD2<Float> = .zero
    private var currentVelocity: SIMD2<Float> = .zero
    private let smoothingFactor: Float = 0.18 // Balanced for smooth transitions
    
    // Callback for camera movement notifications
    var onCameraMove: (() -> Void)?
    
    init() {
        // Set up display link for smooth continuous movement
        displayLink = CADisplayLink(target: self, selector: #selector(updateCameraPosition))
        displayLink?.add(to: .main, forMode: .common)
        print("🎮 Camera movement manager initialized with enhanced joystick support")
    }
    
    deinit {
        // Clean up display link
        displayLink?.invalidate()
    }
    
    // Set the ARView reference for camera manipulation
    func setARView(_ arView: ARView) {
        self.arView = arView
        print("📱 [CameraMovementManager] ARView set: \(arView)")
        print("   ARView memory address: \(Unmanaged.passUnretained(arView).toOpaque())")
        print("   Current state - ARView: \(self.arView != nil), CameraAnchor: \(self.cameraAnchor != nil)")
        
        // Test immediate access
        print("   Immediate test: ARView is accessible: \(self.arView != nil)")
        
        checkReadiness()
        
        // Schedule a test to verify ARView persistence
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
            print("📱 [CameraMovementManager] ARView check after 1s: \(self.arView != nil)")
            if self.arView == nil {
                print("❌ ARView was deallocated! This is the problem.")
            } else {
                print("✅ ARView still accessible after 1s")
            }
        }
    }

    func setCameraAnchor(_ anchor: AnchorEntity) {
        print("🔥 [setCameraAnchor] CALLED with anchor: \(anchor)")
            
        self.cameraAnchor = anchor
        print("🎮 [CameraMovementManager] setCameraAnchor called")
        print("   Camera anchor position: \(anchor.position)")
        print("   Current state - ARView: \(self.arView != nil), CameraAnchor: \(self.cameraAnchor != nil)")
        
        print("🔥 [setCameraAnchor] Anchor stored, checking: \(self.cameraAnchor != nil)")
            
        checkReadiness()
    }
    
    
    
    // Helper method to check if everything is ready
    private func checkReadiness() {
        // Only set ready if BOTH are properly initialized
        let ready = arView != nil && cameraAnchor != nil
        
        if ready != isReady {
            // Set immediately on main thread since we know we're already on main
            if Thread.isMainThread {
                self.isReady = ready
                print("✅ Camera movement manager readiness changed to: \(ready) (sync)")
                if ready {
                    print("   ARView: \(self.arView != nil), CameraAnchor: \(self.cameraAnchor != nil)")
                }
            } else {
                DispatchQueue.main.async {
                    self.isReady = ready
                    print("✅ Camera movement manager readiness changed to: \(ready) (async)")
                    if ready {
                        print("   ARView: \(self.arView != nil), CameraAnchor: \(self.cameraAnchor != nil)")
                    }
                }
            }
        }
    }
    
    // Force ready state when we know both components are properly set
    func forceReady() {
        if Thread.isMainThread {
            self.isReady = true
            print("🔧 Camera movement manager forced to ready state (sync)")
        } else {
            DispatchQueue.main.async {
                self.isReady = true
                print("🔧 Camera movement manager forced to ready state (async)")
            }
        }
    }
    
    // Reset just the ready state without clearing references (DEPRECATED - use resetForNewView)
    func resetReadyState() {
        DispatchQueue.main.async {
            self.isReady = false
            print("🔄 Camera manager ready state reset (keeping ARView and anchor references)")
        }
    }
    
    // Reset ready state AND clear references for clean reinitialization
    func resetForNewView() {
        DispatchQueue.main.async {
            // Clear all references
            self.arView = nil
            self.cameraAnchor = nil
            self.boundaryManager = nil
            
            // Reset movement state
            self.currentVelocity = .zero
            self.targetVelocity = .zero
            self.currentJoystickOffset = .zero
            
            // Reset ready state
            self.isReady = false
            
            print("🔄 Camera manager fully reset for new view")
            print("   - Cleared ARView reference")
            print("   - Cleared camera anchor reference")
            print("   - Cleared boundary manager")
            print("   - Reset velocities")
        }
    }

    // Set boundary manager reference (shared from RealityKitView)
    func setBoundaryManager(_ manager: RealityKitBoundaryManager) {
        self.boundaryManager = manager
        print("🏠 Camera movement manager using shared boundary manager")
    }

    // Update movement speed from settings
    func updateMovementSpeed(_ speed: MovementSpeed) {
        setSpeed(speed)
    }
    
    func setSpeed(_ speed: MovementSpeed) {
        // Map enum to actual speed values with moderate scaling for smooth movement
        movementSpeed = speed.rawValue * 20.0  // Balanced multiplier for smooth control
        currentSpeed = speed
        print("🏃 Speed set to \(speed) (\(movementSpeed))")
    }
    
    // Update joystick input from the virtual joystick
    func updateJoystickInput(_ offset: CGSize) {
        // Print debug info for joystick input
        print("🎮 [CameraMovementManager] updateJoystickInput called with: \(offset)")
        
        // Calculate target velocity based on joystick position
        let x = Float(offset.width)
        let y = Float(offset.height)
        
        // Apply dead zone
        let magnitude = sqrt(x * x + y * y)
        print("🎮 [CameraMovementManager] Joystick magnitude: \(magnitude), deadZone: \(joystickDeadZone)")
        
        if magnitude > joystickDeadZone {
            // Normalize and apply sensitivity
            let normalizedX = x / 30.0  // 30 is max joystick distance
            let normalizedY = y / 30.0
            
            targetVelocity = SIMD2<Float>(normalizedX, normalizedY)
            print("🎮 [CameraMovementManager] Target velocity set: \(targetVelocity)")
            
            // Debug output for significant movements
            if abs(normalizedX) > 0.3 || abs(normalizedY) > 0.3 {
                print("🕹️ Joystick active: X=\(String(format: "%.2f", normalizedX)), Y=\(String(format: "%.2f", normalizedY))")
            }
        } else {
            targetVelocity = .zero
            print("🎮 [CameraMovementManager] Input below deadzone - target velocity set to zero")
        }
        
        currentJoystickOffset = offset
    }
    
    // Continuous camera position updates based on joystick input
    @objc private func updateCameraPosition() {
        // If display link is paused, don't process anything
        if displayLink?.isPaused == true {
            return
        }
        
        guard let arView = arView else {
            if targetVelocity != .zero {
                print("⚠️ [CameraMovementManager] updateCameraPosition: No ARView (target velocity: \(targetVelocity))")
                // Try to debug why ARView is missing
                print("   ARView reference lost during movement - this should not happen")
                print("   Consider using direct camera manipulation as fallback")
            }
            return
        }
        
        guard let cameraAnchor = cameraAnchor else {
            if targetVelocity != .zero {
                print("⚠️ [CameraMovementManager] updateCameraPosition: No camera anchor")
            }
            return
        }
        
        // Smooth velocity transition (use dollhouse-optimized smoothing)
        currentVelocity = mix(currentVelocity, targetVelocity, t: activeSmoothingFactor)
        
        // Skip if velocity is too small
        guard length(currentVelocity) > 0.01 else {
            // If we were moving and now stopped, call the callback
            if length(currentVelocity) < 0.01 && length(targetVelocity) < 0.01 {
                return
            }
            return
        }
        
        // Print current velocity for debugging (less frequent for dollhouse)
        if length(currentVelocity) > 0.1 { // Only print for significant movement
            print("🎮 [CameraMovementManager] Moving camera - currentVelocity: \(currentVelocity)")
        }
        
        // Convert joystick input to movement vectors
        let forwardBackward = -currentVelocity.y * movementSpeed  // Negative for intuitive movement
        let leftRight = currentVelocity.x * movementSpeed
        
        // Get camera's current transform for directional movement
        let cameraTransform = cameraAnchor.transform
        
        // Extract camera's forward and right vectors from camera's rotation
        let forwardVector = SIMD3<Float>(
            cameraTransform.rotation.act(SIMD3<Float>(0, 0, -1)).x,
            0, // Keep movement horizontal
            cameraTransform.rotation.act(SIMD3<Float>(0, 0, -1)).z
        )
        
        let rightVector = SIMD3<Float>(
            cameraTransform.rotation.act(SIMD3<Float>(1, 0, 0)).x,
            0, // Keep movement horizontal
            cameraTransform.rotation.act(SIMD3<Float>(1, 0, 0)).z
        )
        
        // Normalize vectors for consistent movement speed
        let normalizedForward = normalize(forwardVector)
        let normalizedRight = normalize(rightVector)
        
        // Calculate movement delta
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

            // Debug logging when movement is constrained (less frequent)
            if proposedCameraPosition.x != originalProposedPosition.x ||
               proposedCameraPosition.z != originalProposedPosition.z {
                if length(currentVelocity) > 0.2 { // Only log significant boundary hits
                    print("🚧 Camera position constrained: \(originalProposedPosition) -> \(proposedCameraPosition)")
                }
            }
        }
        
        // Only update camera position if it changed
        if proposedCameraPosition.x != currentCameraPosition.x ||
           proposedCameraPosition.z != currentCameraPosition.z {
            
            // Create new transform preserving rotation
            var newTransform = cameraAnchor.transform
            newTransform.translation = proposedCameraPosition
            
            // Apply the new camera transform
            cameraAnchor.transform = newTransform
            
            // Debug output for camera movement (less frequent)
            if length(currentVelocity) > 0.2 { // Only print for significant movements
                print("📍 [CameraMovementManager] Moving camera from \(currentCameraPosition) to \(proposedCameraPosition)")
            }
            
            // Notify that camera has moved
            onCameraMove?()
        }
    }
    
    // Reset camera to default position and orientation
    func resetCameraPosition() {
        guard arView != nil, let cameraAnchor = cameraAnchor else {
            print("⚠️ Cannot reset camera - missing references")
            return
        }
        
        // Reset camera anchor to default transform
        UIView.animate(withDuration: 0.5) {
            cameraAnchor.transform = Transform(
                rotation: simd_quatf(angle: 0, axis: SIMD3<Float>(0, 1, 0)),
                translation: SIMD3<Float>(0, 1.2, 3)
            )
        }
        
        // Reset velocities
        currentVelocity = .zero
        targetVelocity = .zero
        currentJoystickOffset = .zero
        
        print("📷 Camera reset to default position and orientation")
    }
    
    // Enable/disable camera movement
    func setCameraMovementEnabled(_ enabled: Bool) {
        if enabled {
            // Enable the display link
            displayLink?.isPaused = false
            print("🟢 Camera movement display link enabled")
        } else {
            // Disable the display link and reset state
            displayLink?.isPaused = true
            
            // Also reset velocities to prevent interference
            currentVelocity = .zero
            targetVelocity = .zero
            currentJoystickOffset = .zero
            
            print("🔴 Camera movement display link disabled and state cleared")
        }
    }
    
    // Get boundary manager for external use (renamed to avoid conflict)
    func getBoundaryManager() -> RealityKitBoundaryManager? {
        return self.boundaryManager
    }
    
    // Optimize movement settings specifically for dollhouse exploration
    func optimizeForDollhouse() {
        // Use the special dollhouse speed for optimal indoor navigation
        setSpeed(.dollhouse)
        
        // Significantly increase movement speed for responsive dollhouse navigation
        // Make it even faster than regular models for better indoor exploration
        movementSpeed = MovementSpeed.dollhouse.rawValue * 50.0 // Much higher multiplier for responsive movement
        
        // Reduce smoothing for more immediate response (override the private smoothingFactor)
        // We'll use a more direct approach by increasing the base speed instead
        
        print("🏠 Camera movement optimized for dollhouse exploration")
        print("   - Speed set to dollhouse mode")
        print("   - Movement speed boosted to \(movementSpeed) for responsive indoor navigation")
    }
    
    // Override smoothing for dollhouse to be more responsive
    private var activeSmoothingFactor: Float {
        // Use faster smoothing for dollhouse mode
        return currentSpeed == .dollhouse ? 0.35 : smoothingFactor // Much more responsive than 0.18
    }
    
    // Alternative movement method that works directly with passed references
    // This bypasses potential issues with stored weak references
    func updateCameraPositionWithDirectReferences(arView: ARView, cameraAnchor: AnchorEntity) {
        // Smooth velocity transition (use dollhouse-optimized smoothing)
        currentVelocity = mix(currentVelocity, targetVelocity, t: activeSmoothingFactor)
        
        // Skip if velocity is too small
        guard length(currentVelocity) > 0.01 else {
            return
        }
        
        // Print current velocity for debugging - SAME AS FIRST 2 ROOMS
        print("🎮 [CameraMovementManager] Moving camera - currentVelocity: \(currentVelocity)")
        
        // Convert joystick input to movement vectors
        let forwardBackward = -currentVelocity.y * movementSpeed
        let leftRight = currentVelocity.x * movementSpeed
        
        // Get camera's current transform for directional movement
        let cameraTransform = cameraAnchor.transform
        
        // Extract camera's forward and right vectors from camera's rotation
        let forwardVector = SIMD3<Float>(
            cameraTransform.rotation.act(SIMD3<Float>(0, 0, -1)).x,
            0, // Keep movement horizontal
            cameraTransform.rotation.act(SIMD3<Float>(0, 0, -1)).z
        )
        
        let rightVector = SIMD3<Float>(
            cameraTransform.rotation.act(SIMD3<Float>(1, 0, 0)).x,
            0, // Keep movement horizontal
            cameraTransform.rotation.act(SIMD3<Float>(1, 0, 0)).z
        )
        
        // Normalize vectors for consistent movement speed
        let normalizedForward = normalize(forwardVector)
        let normalizedRight = normalize(rightVector)
        
        // Calculate movement delta
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
            
            // Debug logging when movement is constrained - SAME AS FIRST 2 ROOMS  
            if proposedCameraPosition.x != originalProposedPosition.x ||
               proposedCameraPosition.z != originalProposedPosition.z {
                print("🚧 Camera position constrained: \(originalProposedPosition) -> \(proposedCameraPosition)")
                print("🚧 Boundary hit - movement constrained")
            }
        }
        
        // Only update camera position if it changed
        if proposedCameraPosition.x != currentCameraPosition.x ||
           proposedCameraPosition.z != currentCameraPosition.z {
            
            // Create new transform preserving rotation
            var newTransform = cameraAnchor.transform
            newTransform.translation = proposedCameraPosition
            
            // Apply the new camera transform
            cameraAnchor.transform = newTransform
            
            // Debug output for camera movement - SAME AS FIRST 2 ROOMS
            print("📍 [CameraMovementManager] Moving camera from \(currentCameraPosition) to \(proposedCameraPosition)")
            
            // Notify that camera has moved
            onCameraMove?()
        }
    }
}

// MARK: - Helper functions
private func normalize(_ vector: SIMD3<Float>) -> SIMD3<Float> {
    let length = sqrt(vector.x * vector.x + vector.y * vector.y + vector.z * vector.z)
    return length > 0 ? SIMD3<Float>(vector.x / length, vector.y / length, vector.z / length) : vector
}

private func length(_ vector: SIMD2<Float>) -> Float {
    return sqrt(vector.x * vector.x + vector.y * vector.y)
}

private func mix(_ a: SIMD2<Float>, _ b: SIMD2<Float>, t: Float) -> SIMD2<Float> {
    return a + (b - a) * t
}
