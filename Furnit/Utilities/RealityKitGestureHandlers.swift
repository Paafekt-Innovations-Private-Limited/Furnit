import RealityKit
import UIKit

// RealityKit-based gesture handlers to replace SceneKit gesture handlers
class RealityKitGestureHandlers {
    weak var arView: ARView?
    private var boundaryManager: RealityKitBoundaryManager?
    
    // World transformation approach - since ARView.cameraTransform is read-only
    private(set) var worldAnchor: AnchorEntity?
    private var initialWorldTransform: Transform = Transform.identity
    private var lastPanTranslation: CGPoint = .zero
    private var initialTouchPoint: CGPoint?
    
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
    
    // Set up world anchor that will hold all content for transformation
    func setupWorldAnchor() {
        guard let arView = arView else { return }
        
        // Create world anchor if it doesn't exist
        if worldAnchor == nil {
            worldAnchor = AnchorEntity(.world(transform: matrix_identity_float4x4))
            arView.scene.addAnchor(worldAnchor!)
            print("🌍 World anchor created for gesture-based navigation")
        }
        
        // Move existing anchors to world anchor for unified transformation
        let existingAnchors = Array(arView.scene.anchors)
        for anchor in existingAnchors {
            if anchor !== worldAnchor {
                // Remove from scene and add to world anchor
                arView.scene.removeAnchor(anchor)
                worldAnchor?.addChild(anchor)
            }
        }
    }
    
    // Add entity to world anchor for gesture control
    func addToWorld(_ entity: Entity) {
        if worldAnchor == nil {
            setupWorldAnchor()
        }
        worldAnchor?.addChild(entity)
    }
    
    // Set up gesture recognizers for camera control
    private func setupGestureRecognizers() {
        guard let arView = arView else { return }
        
        // Single-finger pan gesture for rotation and forward/back movement
        let singlePanGesture = UIPanGestureRecognizer(target: self, action: #selector(handlePanGesture(_:)))
        singlePanGesture.maximumNumberOfTouches = 1
        singlePanGesture.minimumNumberOfTouches = 1
        arView.addGestureRecognizer(singlePanGesture)
        
        // Two-finger pan gesture for position movement (left/right/up/down)
        let doublePanGesture = UIPanGestureRecognizer(target: self, action: #selector(handlePositionPanGesture(_:)))
        doublePanGesture.minimumNumberOfTouches = 2
        doublePanGesture.maximumNumberOfTouches = 2
        arView.addGestureRecognizer(doublePanGesture)
        
        // Pinch gesture for zoom
        let pinchGesture = UIPinchGestureRecognizer(target: self, action: #selector(handlePinchGesture(_:)))
        arView.addGestureRecognizer(pinchGesture)
        
        // Keep rotation gesture as fallback for advanced users
        let rotationGesture = UIRotationGestureRecognizer(target: self, action: #selector(handleRotationGesture(_:)))
        arView.addGestureRecognizer(rotationGesture)
        
        // Ensure gestures work together properly
        singlePanGesture.require(toFail: doublePanGesture)
        
        print("🎮 RealityKit gesture recognizers set up with intuitive controls")
        print("   Single finger: horizontal=rotate, vertical=forward/back")
        print("   Two fingers: drag=position, pinch=zoom, rotate=turn")
    }
    
    // Handle pan gesture with intuitive controls: horizontal = rotation, vertical = movement
    @objc private func handlePanGesture(_ gesture: UIPanGestureRecognizer) {
        guard let arView = arView, let worldAnchor = worldAnchor else { return }
        
        let translation = gesture.translation(in: arView)
        
        switch gesture.state {
        case .began:
            initialWorldTransform = worldAnchor.transform
            initialTouchPoint = gesture.location(in: arView)
            lastPanTranslation = translation
            
        case .changed:
            // Calculate incremental change since last update
            let deltaTranslation = CGPoint(
                x: translation.x - lastPanTranslation.x,
                y: translation.y - lastPanTranslation.y
            )
            
            // Determine if gesture is primarily horizontal or vertical
            let deltaX = Float(translation.x)
            let deltaY = Float(translation.y)
            let isHorizontalGesture = abs(deltaX) > abs(deltaY)
            
            if isHorizontalGesture {
                // Horizontal swipe = Rotate around Y-axis (look left/right)
                // Use incremental change, not total translation
                let incrementalRotation = Float(deltaTranslation.x) * rotationSensitivity
                
                // Apply incremental rotation to current transform
                let rotationQuat = simd_quatf(angle: -incrementalRotation, axis: SIMD3<Float>(0, 1, 0))
                
                // Use current world transform, not initial
                var newTransform = worldAnchor.transform
                newTransform.rotation = rotationQuat * newTransform.rotation
                worldAnchor.transform = newTransform
                
                print("🔄 Horizontal rotation: \(incrementalRotation) radians (incremental)")
                
            } else {
                // Vertical swipe = Move forward/backward
                let movementDelta = deltaY * panSensitivity // No inversion needed for forward/back
                
                // Get camera's current transform for directional reference
                let cameraTransform = arView.cameraTransform
                
                // Calculate forward direction (camera looks down negative Z)
                let cameraForward = normalize(SIMD3<Float>(
                    cameraTransform.matrix.columns.2.x,
                    0, // Keep movement horizontal
                    cameraTransform.matrix.columns.2.z
                ))
                
                // Calculate world movement (opposite direction for intuitive control)
                let worldMovement = cameraForward * movementDelta
                var newPosition = initialWorldTransform.translation + worldMovement
                
                // Apply boundary constraints if available
                if let boundaryManager = boundaryManager {
                    // Calculate effective camera position for boundary checking
                    let effectiveCameraPos = cameraTransform.translation - worldMovement
                    let constrainedCameraPos = boundaryManager.constrainCameraPosition(effectiveCameraPos)
                    let constraintDelta = constrainedCameraPos - cameraTransform.translation
                    newPosition = initialWorldTransform.translation - constraintDelta
                }
                
                // Apply the new world transform
                var newTransform = initialWorldTransform
                newTransform.translation = newPosition
                worldAnchor.transform = newTransform
                
                print("⬆️ Vertical movement: \(movementDelta)")
            }
            
            // Update last translation for next incremental calculation
            lastPanTranslation = translation
            
        case .ended, .cancelled:
            initialWorldTransform = worldAnchor.transform
            initialTouchPoint = nil
            lastPanTranslation = .zero
            
        default:
            break
        }
    }
    
    // Handle two-finger pan gesture for position movement (strafe left/right/up/down)
    @objc private func handlePositionPanGesture(_ gesture: UIPanGestureRecognizer) {
        guard let arView = arView, let worldAnchor = worldAnchor else { return }
        
        let translation = gesture.translation(in: arView)
        
        switch gesture.state {
        case .began:
            initialWorldTransform = worldAnchor.transform
            initialTouchPoint = gesture.location(in: arView)
            
        case .changed:
            // Two-finger pan for direct position movement (strafe)
            let deltaX = Float(translation.x) * panSensitivity
            let deltaY = Float(-translation.y) * panSensitivity // Invert Y for natural up/down
            
            // Get camera's current transform for directional reference
            let cameraTransform = arView.cameraTransform
            
            // Calculate camera's right and up vectors
            let cameraRight = normalize(SIMD3<Float>(
                cameraTransform.matrix.columns.0.x,
                0, // Keep horizontal movement
                cameraTransform.matrix.columns.0.z
            ))
            
            let cameraUp = SIMD3<Float>(0, 1, 0) // World up for vertical movement
            
            // Calculate world movement (inverse for intuitive control)
            let worldMovement = -(cameraRight * deltaX + cameraUp * deltaY)
            var newPosition = initialWorldTransform.translation + worldMovement
            
            // Apply boundary constraints if available
            if let boundaryManager = boundaryManager {
                // Calculate effective camera position for boundary checking
                let effectiveCameraPos = cameraTransform.translation - worldMovement
                let constrainedCameraPos = boundaryManager.constrainCameraPosition(effectiveCameraPos)
                let constraintDelta = constrainedCameraPos - cameraTransform.translation
                newPosition = initialWorldTransform.translation - constraintDelta
            }
            
            // Apply the new world transform
            var newTransform = initialWorldTransform
            newTransform.translation = newPosition
            worldAnchor.transform = newTransform
            
            print("✋ Two-finger position movement: (\(deltaX), \(deltaY))")
            
        case .ended, .cancelled:
            initialWorldTransform = worldAnchor.transform
            initialTouchPoint = nil
            
        default:
            break
        }
    }
    
    // Handle pinch gesture for zoom (world scale/movement for zoom effect)
    @objc private func handlePinchGesture(_ gesture: UIPinchGestureRecognizer) {
        guard let arView = arView, let worldAnchor = worldAnchor else { return }
        
        switch gesture.state {
        case .began:
            initialWorldTransform = worldAnchor.transform
            
        case .changed:
            // Calculate zoom factor (scale change from initial)
            let scale = gesture.scale
            let zoomFactor = (scale - 1.0) * 0.5 // Reduce sensitivity
            
            // Get camera's current transform for directional reference
            let cameraTransform = arView.cameraTransform
            
            // Move world backward/forward along camera's view direction (inverse for zoom effect)
            let forward = normalize(SIMD3<Float>(
                cameraTransform.matrix.columns.2.x,
                cameraTransform.matrix.columns.2.y,
                cameraTransform.matrix.columns.2.z
            ))
            
            // World moves opposite to create zoom effect
            let worldMovement = -forward * Float(zoomFactor)
            var newPosition = initialWorldTransform.translation + worldMovement
            
            // Apply boundary constraints (simulate camera constraint)
            if let boundaryManager = boundaryManager {
                let effectiveCameraPos = cameraTransform.translation - worldMovement
                let constrainedCameraPos = boundaryManager.constrainCameraPosition(effectiveCameraPos)
                let constraintDelta = constrainedCameraPos - cameraTransform.translation
                newPosition = initialWorldTransform.translation - constraintDelta
            }
            
            // Apply the new world transform
            var newTransform = initialWorldTransform
            newTransform.translation = newPosition
            worldAnchor.transform = newTransform
            
        case .ended, .cancelled:
            initialWorldTransform = worldAnchor.transform
            
        default:
            break
        }
    }
    
    // Handle rotation gesture for world rotation (inverse of camera rotation)
    @objc private func handleRotationGesture(_ gesture: UIRotationGestureRecognizer) {
        guard let arView = arView, let worldAnchor = worldAnchor else { return }
        
        switch gesture.state {
        case .began:
            initialWorldTransform = worldAnchor.transform
            
        case .changed:
            // Apply rotation around Y axis (horizontal rotation)
            let rotation = Float(gesture.rotation) * rotationSensitivity
            
            // Rotate world opposite to camera rotation for intuitive control
            let rotationQuat = simd_quatf(angle: -rotation, axis: SIMD3<Float>(0, 1, 0))
            
            // Apply rotation to world anchor
            var newTransform = initialWorldTransform
            newTransform.rotation = rotationQuat * initialWorldTransform.rotation
            worldAnchor.transform = newTransform
            
        case .ended, .cancelled:
            initialWorldTransform = worldAnchor.transform
            
        default:
            break
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