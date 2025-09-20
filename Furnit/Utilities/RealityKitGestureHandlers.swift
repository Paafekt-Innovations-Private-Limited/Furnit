import RealityKit
import UIKit

// RealityKit-based gesture handlers to replace SceneKit gesture handlers
class RealityKitGestureHandlers {
    weak var arView: ARView?
    private var boundaryManager: RealityKitBoundaryManager?
    
    // Store gesture recognizers to prevent deallocation
    private var singlePanGesture: UIPanGestureRecognizer?
    private var doublePanGesture: UIPanGestureRecognizer?
    private var pinchGesture: UIPinchGestureRecognizer?
    private var rotationGesture: UIRotationGestureRecognizer?
    
    // Custom camera control for non-AR mode - direct camera manipulation
    private weak var cameraEntity: PerspectiveCamera?
    private weak var cameraAnchor: AnchorEntity?
    private var initialCameraTransform: Transform = Transform.identity
    private var lastPanTranslation: CGPoint = .zero
    private var initialTouchPoint: CGPoint?

    // Accumulated rotation state to prevent flickering and maintain smooth rotation
    private var accumulatedYaw: Float = 0.0    // Horizontal rotation around Y-axis
    private var accumulatedPitch: Float = 0.0  // Vertical rotation around X-axis
    
    // Note: Using total translation from gesture start instead of cumulative tracking for smoother rotation
    
    // Pan gesture configuration
    private let panSensitivity: Float = 0.005
    private let rotationSensitivity: Float = 0.01
    
    init(arView: ARView) {
        self.arView = arView
        setupGestureRecognizers()
    }
    
    // Set boundary manager for camera constraints
    func setBoundaryManager(_ manager: RealityKitBoundaryManager) {
        self.boundaryManager = manager
    }
    
    // Set camera references for direct camera control in non-AR mode
    func setCameraReferences(camera: PerspectiveCamera, cameraAnchor: AnchorEntity) {
        self.cameraEntity = camera
        self.cameraAnchor = cameraAnchor
        print("📷 Camera references set for direct camera control")
    }
    
    // Set up gesture recognizers for camera control
    private func setupGestureRecognizers() {
        guard let arView = arView else { return }
        
        // Single-finger pan gesture for rotation and forward/back movement
        singlePanGesture = UIPanGestureRecognizer(target: self, action: #selector(handlePanGesture(_:)))
        singlePanGesture?.maximumNumberOfTouches = 1
        singlePanGesture?.minimumNumberOfTouches = 1
        if let singlePan = singlePanGesture {
            arView.addGestureRecognizer(singlePan)
        }
        
        // Two-finger pan gesture for position movement (left/right/up/down)
        doublePanGesture = UIPanGestureRecognizer(target: self, action: #selector(handlePositionPanGesture(_:)))
        doublePanGesture?.minimumNumberOfTouches = 2
        doublePanGesture?.maximumNumberOfTouches = 2
        if let doublePan = doublePanGesture {
            arView.addGestureRecognizer(doublePan)
        }
        
        // Pinch gesture for zoom
        pinchGesture = UIPinchGestureRecognizer(target: self, action: #selector(handlePinchGesture(_:)))
        if let pinch = pinchGesture {
            arView.addGestureRecognizer(pinch)
        }
        
        // Keep rotation gesture as fallback for advanced users
        rotationGesture = UIRotationGestureRecognizer(target: self, action: #selector(handleRotationGesture(_:)))
        if let rotation = rotationGesture {
            arView.addGestureRecognizer(rotation)
        }
        
        // Remove the require(toFail:) that was preventing gestures from working
        // Allow both single and double pan gestures to work independently
        
        print("🎮 RealityKit gesture recognizers set up with intuitive controls")
        print("   Single finger: drag=look around (horizontal+vertical rotation)")
        print("   Two fingers: drag=position, pinch=zoom, rotate=turn")
        print("   Joystick: forward/backward/left/right movement")
        print("   Note: Very small single finger movements adjust height")
    }
    
    // Handle pan gesture with intuitive controls: drag to look around (horizontal + vertical rotation)
    @objc private func handlePanGesture(_ gesture: UIPanGestureRecognizer) {
        // print("🚨 PAN GESTURE CALLED - State: \(gesture.state.rawValue)")
        guard let arView = arView, let cameraAnchor = cameraAnchor else {
            print("⚠️ Pan gesture guard failed - arView: \(arView != nil), cameraAnchor: \(cameraAnchor != nil)")
            return
        }

        let translation = gesture.translation(in: arView)

        switch gesture.state {
        case .began:
            // Store initial position but don't reset accumulated rotation
            initialCameraTransform = cameraAnchor.transform
            initialTouchPoint = gesture.location(in: arView)
            lastPanTranslation = translation

        case .changed:
            // print("🔥 CAMERA GESTURE CHANGED STATE - translation: \(translation)")

            // Calculate incremental rotation delta since last update
            let deltaTranslation = CGPoint(
                x: translation.x - lastPanTranslation.x,
                y: translation.y - lastPanTranslation.y
            )

            // Convert delta to rotation increments
            let deltaYaw = Float(deltaTranslation.x) * rotationSensitivity * 0.5   // Horizontal rotation
            let deltaPitch = Float(deltaTranslation.y) * rotationSensitivity * 0.5 // Vertical rotation

            // Update accumulated rotation values
            accumulatedYaw += -deltaYaw  // Negative for natural direction
            accumulatedPitch += -deltaPitch // Negative for natural direction

            // Apply pitch limits to prevent over-rotation and tilting (45 degrees up/down max)
            let maxPitch: Float = Float.pi / 4.0 // 45 degrees
            accumulatedPitch = max(-maxPitch, min(maxPitch, accumulatedPitch))

            // Create rotation quaternions from accumulated values (prevents tilting by only using yaw and pitch)
            let yawRotation = simd_quatf(angle: accumulatedYaw, axis: SIMD3<Float>(0, 1, 0))     // Horizontal only
            let pitchRotation = simd_quatf(angle: accumulatedPitch, axis: SIMD3<Float>(1, 0, 0)) // Vertical only

            // Combine rotations: apply pitch first, then yaw (no roll component to prevent tilting)
            let combinedRotation = yawRotation * pitchRotation

            // Create new transform with accumulated rotation, preserving position
            var newTransform = Transform()
            newTransform.translation = initialCameraTransform.translation  // Lock position
            newTransform.rotation = combinedRotation  // Apply accumulated rotation (no roll/tilt)
            newTransform.scale = initialCameraTransform.scale
            cameraAnchor.transform = newTransform

            // print("📷 Accumulated rotation: Yaw=\(accumulatedYaw), Pitch=\(accumulatedPitch)")

            // Update last translation for next incremental calculation
            lastPanTranslation = translation

        case .ended, .cancelled:
            // Update initial transform to current state to preserve rotation for next gesture
            initialCameraTransform = cameraAnchor.transform
            initialTouchPoint = nil
            lastPanTranslation = .zero

        default:
            break
        }
    }
    
    // Handle two-finger pan gesture for position movement (strafe left/right/up/down)
    @objc private func handlePositionPanGesture(_ gesture: UIPanGestureRecognizer) {
        guard let _ = arView, let cameraAnchor = cameraAnchor else { return }
        
        let translation = gesture.translation(in: arView)
        
        switch gesture.state {
        case .began:
            initialCameraTransform = cameraAnchor.transform
            initialTouchPoint = gesture.location(in: arView)
            
        case .changed:
            // Two-finger pan for direct camera position movement (strafe)
            let deltaX = Float(translation.x) * panSensitivity
            let deltaY = Float(-translation.y) * panSensitivity // Invert Y for natural up/down
            
            // Get camera's current transform for directional reference
            let cameraTransform = cameraAnchor.transform
            
            // Calculate camera's right and up vectors in world space
            let cameraRight = normalize(SIMD3<Float>(
                cameraTransform.rotation.act(SIMD3<Float>(1, 0, 0)).x,
                0, // Keep horizontal movement
                cameraTransform.rotation.act(SIMD3<Float>(1, 0, 0)).z
            ))
            
            let cameraUp = SIMD3<Float>(0, 1, 0) // World up for vertical movement
            
            // Calculate camera movement (direct camera control)
            let cameraMovement = cameraRight * deltaX + cameraUp * deltaY
            var newPosition = initialCameraTransform.translation + cameraMovement
            
            // Apply boundary constraints if available
            if let boundaryManager = boundaryManager {
                newPosition = boundaryManager.constrainCameraPosition(newPosition)
            }
            
            // Apply the new camera position
            var newTransform = initialCameraTransform
            newTransform.translation = newPosition
            cameraAnchor.transform = newTransform

            // print("📷 Two-finger camera movement: (\(deltaX), \(deltaY))")

        case .ended, .cancelled:
            initialCameraTransform = cameraAnchor.transform
            initialTouchPoint = nil
            
        default:
            break
        }
    }
    
    // Handle pinch gesture for zoom (camera movement for zoom effect)
    @objc private func handlePinchGesture(_ gesture: UIPinchGestureRecognizer) {
        guard let _ = arView, let cameraAnchor = cameraAnchor else { return }
        
        switch gesture.state {
        case .began:
            initialCameraTransform = cameraAnchor.transform
            
        case .changed:
            // Calculate zoom factor (scale change from initial)
            let scale = gesture.scale
            let zoomFactor = (scale - 1.0) * 0.5 // Reduce sensitivity
            
            // Get camera's current transform for directional reference
            let cameraTransform = cameraAnchor.transform
            
            // Move camera forward/backward along its view direction for zoom effect
            let forward = normalize(SIMD3<Float>(
                cameraTransform.rotation.act(SIMD3<Float>(0, 0, -1)).x,
                cameraTransform.rotation.act(SIMD3<Float>(0, 0, -1)).y,
                cameraTransform.rotation.act(SIMD3<Float>(0, 0, -1)).z
            ))
            
            // Camera moves forward/backward for zoom effect
            let cameraMovement = forward * Float(zoomFactor)
            var newPosition = initialCameraTransform.translation + cameraMovement
            
            // Apply boundary constraints
            if let boundaryManager = boundaryManager {
                newPosition = boundaryManager.constrainCameraPosition(newPosition)
            }
            
            // Apply the new camera position
            var newTransform = initialCameraTransform
            newTransform.translation = newPosition
            cameraAnchor.transform = newTransform

            // print("📷 Camera zoom: \(zoomFactor)")

        case .ended, .cancelled:
            initialCameraTransform = cameraAnchor.transform
            
        default:
            break
        }
    }
    
    // Handle rotation gesture for direct camera rotation
    @objc private func handleRotationGesture(_ gesture: UIRotationGestureRecognizer) {
        guard let _ = arView, let cameraAnchor = cameraAnchor else { return }
        
        switch gesture.state {
        case .began:
            initialCameraTransform = cameraAnchor.transform
            
        case .changed:
            // Apply rotation around Y axis (horizontal rotation)
            let rotation = Float(gesture.rotation) * rotationSensitivity
            
            // Rotate camera directly for intuitive control
            let rotationQuat = simd_quatf(angle: rotation, axis: SIMD3<Float>(0, 1, 0))
            
            // Apply rotation to camera anchor
            var newTransform = initialCameraTransform
            newTransform.rotation = rotationQuat * initialCameraTransform.rotation
            cameraAnchor.transform = newTransform

            // print("📷 Camera rotation gesture: \(rotation) radians")

        case .ended, .cancelled:
            initialCameraTransform = cameraAnchor.transform
            
        default:
            break
        }
    }
    
    // Reset camera to default position and orientation
    func resetCameraPosition() {
        guard let _ = arView, let cameraAnchor = cameraAnchor else { return }

        // Reset accumulated rotation values
        accumulatedYaw = 0.0
        accumulatedPitch = 0.0

        // Reset camera anchor to default transform with animation at eye level
        UIView.animate(withDuration: 0.5) {
            cameraAnchor.transform = Transform(rotation: simd_quatf(angle: 0, axis: SIMD3<Float>(0, 1, 0)), translation: SIMD3<Float>(0, 1.2, 3))
        }

        // Update initial transform after reset
        initialCameraTransform = cameraAnchor.transform

        print("📷 Camera reset to default position and orientation with cleared rotation state")
    }
    
    // Enable/disable gestures based on AR state
    func setGesturesEnabled(_ enabled: Bool) {
        guard let arView = arView else { return }
        
        for gestureRecognizer in arView.gestureRecognizers ?? [] {
            gestureRecognizer.isEnabled = enabled
        }
        
        print("🎮 Gestures \(enabled ? "enabled" : "disabled")")
    }
    
}

// MARK: - Helper functions

private func normalize(_ vector: SIMD3<Float>) -> SIMD3<Float> {
    let length = sqrt(vector.x * vector.x + vector.y * vector.y + vector.z * vector.z)
    return length > 0 ? SIMD3<Float>(vector.x / length, vector.y / length, vector.z / length) : vector
}