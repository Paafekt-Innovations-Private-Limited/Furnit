import RealityKit
import ARKit
import SwiftUI
import simd

class RealityKitCameraMovementManager: ObservableObject {
    // Define MovementSpeed enum locally with ENHANCED values for noticeable movement
    enum MovementSpeed: Float {
        case slow = 0.004     // 2.5x original for gentle but noticeable movement  
        case normal = 0.008   // 2.7x original for comfortable cruising
        case fast = 0.016     // 2.7x original for quicker navigation
    }
    
    // Camera movement properties
    weak var arView: ARView?
    private var boundaryManager: RealityKitBoundaryManager?
    private var displayLink: CADisplayLink?
    private var currentJoystickOffset: CGSize = .zero
    
    // Custom camera control for non-AR mode
    private weak var cameraAnchor: AnchorEntity?
    
    // Current speed setting
    @Published var currentSpeed: MovementSpeed = .normal
    
    // Movement configuration - ENHANCED FOR NOTICEABLE MOVEMENT
    private var movementSpeed: Float = 0.08  // 2x increased for more noticeable movement
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
        print("📱 Camera movement manager ARView set")
    }
    
    func setupARView(_ arView: ARView) {
        setARView(arView)
    }
    
    // Set camera anchor reference for direct camera control
    func setCameraAnchor(_ anchor: AnchorEntity) {
        self.cameraAnchor = anchor
        print("🎮 [CameraMovementManager] setCameraAnchor called - Camera anchor is now set!")
        print("🎮 [CameraMovementManager] Camera anchor position: \(anchor.transform.translation)")
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
        // Map enum to actual speed values with enhanced multiplier for noticeable movement
        movementSpeed = speed.rawValue * 30.0  // Increased from 20.0 to 30.0 for more noticeable movement
        currentSpeed = speed
        print("🏃 Speed set to \(speed) (\(movementSpeed))")
    }
    
    // Set speed specifically for dollhouse rooms (much faster)
    func setDollhouseSpeed() {
        // Use much higher speed for dollhouse rooms to make movement more noticeable
        movementSpeed = 0.3  // 5x faster than normal speed
        currentSpeed = .normal
        print("🏠 Dollhouse speed set to \(movementSpeed) (enhanced for better visibility)")
    }
    
    // Update joystick input from the virtual joystick
    func updateJoystickInput(_ offset: CGSize) {
        print("🎮 [CameraMovementManager] updateJoystickInput called with: \(offset)")
        currentJoystickOffset = offset
        
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
    }
    
    // Continuous camera position updates based on joystick input
    @objc private func updateCameraPosition() {
        guard let arView = arView, let cameraAnchor = cameraAnchor else { 
            if arView == nil {
                print("⚠️ [CameraMovementManager] updateCameraPosition: No ARView")
            }
            if cameraAnchor == nil {
                print("⚠️ [CameraMovementManager] updateCameraPosition: No camera anchor")
            }
            return 
        }
        
        // Smooth velocity transition
        currentVelocity = mix(currentVelocity, targetVelocity, t: smoothingFactor)
        
        // Skip if velocity is too small
        guard length(currentVelocity) > 0.01 else {
            // If we were moving and now stopped, call the callback
            if length(currentVelocity) < 0.01 && length(targetVelocity) < 0.01 {
                return
            }
            return
        }
        
        print("🎮 [CameraMovementManager] Moving camera - currentVelocity: \(currentVelocity)")
        
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

            // Debug logging when movement is constrained
            if proposedCameraPosition.x != originalProposedPosition.x ||
               proposedCameraPosition.z != originalProposedPosition.z {
                print("🚧 Boundary hit - movement constrained")
            }
        }
        
        // Only update camera position if it changed
        if proposedCameraPosition.x != currentCameraPosition.x ||
           proposedCameraPosition.z != currentCameraPosition.z {
            
            print("📍 [CameraMovementManager] Moving camera from \(currentCameraPosition) to \(proposedCameraPosition)")
            
            // Create new transform preserving rotation
            var newTransform = cameraAnchor.transform
            newTransform.translation = proposedCameraPosition
            
            // Apply the new camera transform
            cameraAnchor.transform = newTransform
            
            // Notify that camera has moved
            onCameraMove?()
        } else {
            print("📍 [CameraMovementManager] No position change - staying at \(currentCameraPosition)")
        }
    }
    
    // Reset camera to default position and orientation
    func resetCameraPosition() {
        guard let _ = arView, let cameraAnchor = cameraAnchor else { return }
        
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
        displayLink?.isPaused = !enabled
        if !enabled {
            // Reset velocities when disabling
            currentVelocity = .zero
            targetVelocity = .zero
        }
        print("🎮 Camera movement \(enabled ? "enabled" : "disabled")")
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
