import RealityKit
import UIKit

// RealityKit-based gesture handlers to replace SceneKit gesture handlers
// Inherits from NSObject for Objective-C gesture recognizer target-action compatibility
class RealityKitGestureHandlers: NSObject {
    weak var arView: ARView?
    private var boundaryManager: RealityKitBoundaryManager?

    // Object placement manager reference for object manipulation
    weak var objectPlacementManager: RealityKitObjectPlacementManager?

    // Store gesture recognizers to prevent deallocation
    private var singlePanGesture: UIPanGestureRecognizer?
    private var doublePanGesture: UIPanGestureRecognizer?
    private var pinchGesture: UIPinchGestureRecognizer?
    private var rotationGesture: UIRotationGestureRecognizer?
    private var longPressGesture: UILongPressGestureRecognizer?
    private var objectManipulationPanGesture: UIPanGestureRecognizer?

    // Haptic feedback generator for object selection
    private let hapticFeedbackGenerator = UIImpactFeedbackGenerator(style: .medium)
    
    // Custom camera control for non-AR mode - direct camera manipulation
    private weak var cameraEntity: PerspectiveCamera?
    private weak var cameraAnchor: AnchorEntity?
    private var initialCameraTransform: Transform = Transform.identity
    private var lastPanTranslation: CGPoint = .zero
    private var lastPositionPanTranslation: CGPoint = .zero  // For smooth two-finger position movement
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
        super.init()
        setupGestureRecognizers()

        // Prepare haptic feedback generator for lower latency
        hapticFeedbackGenerator.prepare()
    }
    
    // Set boundary manager for camera constraints
    func setBoundaryManager(_ manager: RealityKitBoundaryManager) {
        self.boundaryManager = manager
    }

    // Set object placement manager for object manipulation
    func setObjectPlacementManager(_ manager: RealityKitObjectPlacementManager) {
        self.objectPlacementManager = manager
        logDebug("🎯 Object placement manager set for manipulation handling")
    }

    // Set camera references for direct camera control in non-AR mode
    func setCameraReferences(camera: PerspectiveCamera, cameraAnchor: AnchorEntity) {
        self.cameraEntity = camera
        self.cameraAnchor = cameraAnchor

        // Initialize accumulated rotation from camera's current orientation
        initializeRotationFromCamera()

        logDebug("📷 Camera references set for direct camera control")
    }

    /// Re-sync rotation state after camera is repositioned (e.g., after model loads)
    /// Call this after setting camera position/rotation programmatically
    func syncRotationState() {
        initializeRotationFromCamera()
    }

    /// Initialize accumulated yaw/pitch from camera's current look direction
    /// This ensures rotation gestures work correctly for cameras with non-zero initial orientation
    /// Uses direct look direction calculation to avoid quaternion decomposition mismatches
    private func initializeRotationFromCamera() {
        guard let anchor = cameraAnchor else { return }

        // Get camera's forward direction (camera looks along -Z in local space)
        let rotation = anchor.transform.rotation
        let forward = rotation.act(SIMD3<Float>(0, 0, -1))  // Transform -Z by rotation

        // Calculate yaw from the horizontal component of forward direction
        // yaw = 0 means looking toward -Z (forward in world space)
        // Positive yaw means looking toward -X (left in world space)
        let yaw = atan2(-forward.x, -forward.z)

        // Calculate pitch from the vertical component
        // pitch = 0 means looking horizontal
        // Positive pitch means looking down
        // Using the length of horizontal component for proper angle calculation
        let horizontalLength = sqrt(forward.x * forward.x + forward.z * forward.z)
        let pitch = atan2(-forward.y, horizontalLength)

        accumulatedYaw = yaw
        accumulatedPitch = pitch

        logDebug("📷 Initialized rotation from look direction:")
        logDebug("   Forward: (\(forward.x), \(forward.y), \(forward.z))")
        logDebug("   Yaw: \(yaw) rad (\(yaw * 180 / Float.pi)°)")
        logDebug("   Pitch: \(pitch) rad (\(pitch * 180 / Float.pi)°)")
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

        // Note: Two-finger rotation gesture removed - single-finger pan handles rotation
        // This allows two-finger pan to work for strafe movement without conflicts

        // Long press gesture for object manipulation
        longPressGesture = UILongPressGestureRecognizer(target: self, action: #selector(handleLongPressGesture(_:)))
        longPressGesture?.minimumPressDuration = 0.8 // 800ms for long press
        if let longPress = longPressGesture {
            arView.addGestureRecognizer(longPress)
        }

        // Object manipulation pan gesture (separate from camera pan)
        objectManipulationPanGesture = UIPanGestureRecognizer(target: self, action: #selector(handleObjectManipulationPan(_:)))
        objectManipulationPanGesture?.maximumNumberOfTouches = 1
        objectManipulationPanGesture?.minimumNumberOfTouches = 1
        if let objPan = objectManipulationPanGesture {
            arView.addGestureRecognizer(objPan)
        }

        // Set up gesture priorities to prevent conflicts
        setupGesturePriorities()

        logDebug("🎮 RealityKit gesture recognizers set up with intuitive controls")
        logDebug("   Single finger: drag=look around (horizontal+vertical rotation)")
        logDebug("   Two fingers: drag=position, pinch=zoom, rotate=turn")
        logDebug("   Long press: select object for manipulation")
        logDebug("   During manipulation: horizontal swipe=rotate object")
        logDebug("   Joystick: forward/backward/left/right movement")
        logDebug("   Note: Very small single finger movements adjust height")

        // Log all gesture recognizers on the view
        if let gestures = arView.gestureRecognizers {
            logDebug("📋 Total gesture recognizers on ARView: \(gestures.count)")
            for (index, gesture) in gestures.enumerated() {
                logDebug("   [\(index)] \(type(of: gesture)) - enabled: \(gesture.isEnabled)")
            }
        }
    }
    
    // Set up gesture priorities to prevent conflicts
    private func setupGesturePriorities() {
        // Object manipulation pan should be disabled when not manipulating
        objectManipulationPanGesture?.isEnabled = false

        logDebug("🎯 Gesture priorities configured - object manipulation initially disabled")
    }

    // MARK: - Object Manipulation Gesture Handlers

    // Handle long press gesture for object selection
    @MainActor @objc private func handleLongPressGesture(_ gesture: UILongPressGestureRecognizer) {
        guard gesture.state == .began else { return }

        guard let arView = arView else {
            logDebug("⚠️ Long press gesture: no ARView available")
            return
        }

        let location = gesture.location(in: arView)

        // Try to select an object for manipulation
        if let placementManager = objectPlacementManager {
            let success = placementManager.handleLongPress(at: location)
            if success {
                // Trigger haptic feedback for successful object selection
                hapticFeedbackGenerator.impactOccurred()

                // Enable object manipulation gestures, disable camera gestures
                enableObjectManipulationMode(true)
                logDebug("🎯 Long press successful - object selected for manipulation")
                logDebug("📳 Haptic feedback triggered for object selection")
            } else {
                logDebug("📍 Long press found no objects to manipulate")
            }
        } else {
            logDebug("⚠️ No object placement manager available for long press handling")
        }
    }

    // Handle pan gesture for object rotation during manipulation
    @MainActor @objc private func handleObjectManipulationPan(_ gesture: UIPanGestureRecognizer) {
        guard let placementManager = objectPlacementManager,
              placementManager.isManipulatingObject,
              gesture.isEnabled else {
            // Only log warnings if gesture is enabled (unexpected behavior)
            // If gesture is disabled, this is expected during mode transitions
            if gesture.isEnabled {
                logDebug("⚠️ Object manipulation pan called but no object is being manipulated")
            }
            return
        }

        let translation = gesture.translation(in: arView)

        switch gesture.state {
        case .began:
            logDebug("🔄 Started object rotation gesture")

        case .changed:
            // Handle horizontal swipe for object rotation
            placementManager.handleObjectRotation(translation: translation)
            // Reset translation to get incremental changes
            gesture.setTranslation(.zero, in: arView)

        case .ended, .cancelled:
            // Keep manipulation mode active - user must explicitly cancel via buttons
            logDebug("🔄 Object manipulation gesture ended - staying in manipulation mode")

        default:
            break
        }
    }

    // Reset gesture recognizer states to prevent conflicts
    private func resetGestureStates() {
        // Cancel any active object manipulation gestures
        if let objectPan = objectManipulationPanGesture {
            if objectPan.state == .changed || objectPan.state == .began {
                logDebug("🔄 Resetting active object manipulation gesture state")
                objectPan.isEnabled = false
                // Small delay to ensure state is properly reset
                DispatchQueue.main.async {
                    objectPan.isEnabled = true
                }
            }
        }

        // Ensure camera gestures are in clean state
        [singlePanGesture, doublePanGesture, pinchGesture].forEach { gesture in
            gesture?.isEnabled = gesture?.isEnabled ?? true // Refresh state
        }

        logDebug("🔧 Gesture states reset for clean transitions")
    }

    // Enable/disable object manipulation mode
    private func enableObjectManipulationMode(_ enabled: Bool) {
        if enabled {
            // Enabling object manipulation - immediate switch
            objectManipulationPanGesture?.isEnabled = true
            singlePanGesture?.isEnabled = false  // Disable camera rotation during object manipulation
            doublePanGesture?.isEnabled = false // Disable camera movement during object manipulation
            pinchGesture?.isEnabled = false     // Disable camera zoom during object manipulation

            logDebug("🎯 Object manipulation mode: ENABLED")
            logDebug("   Camera gestures: DISABLED")
        } else {
            // Disabling object manipulation - clean transition back to camera control
            resetGestureStates()

            // Small delay to ensure clean gesture state transition
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.05) {
                self.objectManipulationPanGesture?.isEnabled = false
                self.singlePanGesture?.isEnabled = true   // Re-enable camera rotation
                self.doublePanGesture?.isEnabled = true  // Re-enable camera movement
                self.pinchGesture?.isEnabled = true      // Re-enable camera zoom

                logDebug("🎯 Object manipulation mode: DISABLED")
                logDebug("   Camera gestures: RE-ENABLED")
                logDebug("🎮 Camera rotation should now work normally")
            }
        }
    }

    // Public method to cancel object manipulation (called by Cancel button)
    @MainActor func cancelObjectManipulation() {
        guard let placementManager = objectPlacementManager else { return }

        placementManager.endObjectManipulation()
        enableObjectManipulationMode(false)

        logDebug("❌ Object manipulation cancelled by user")
    }

    // Public method to delete selected object (called by Delete button)
    @MainActor func deleteSelectedObject() {
        guard let placementManager = objectPlacementManager,
              let selectedObject = placementManager.selectedObject else {
            logDebug("⚠️ No object selected for deletion")
            return
        }

        logDebug("🗑️ Gesture handler: Starting object deletion...")

        // End manipulation mode FIRST to clean up gesture state
        placementManager.endObjectManipulation()
        enableObjectManipulationMode(false)

        // Small delay to ensure gesture state is cleaned up before removing object
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
            // Remove the object from the placement manager
            placementManager.removeObject(selectedObject.id)
            logDebug("🗑️ Gesture handler: Object deletion completed")
        }
    }

    // Handle pan gesture with intuitive controls: drag to look around (horizontal + vertical rotation)
    @objc private func handlePanGesture(_ gesture: UIPanGestureRecognizer) {
        logDebug("🚨 PAN GESTURE CALLED - State: \(gesture.state.rawValue)")
        guard let arView = arView, let cameraAnchor = cameraAnchor else {
            logDebug("⚠️ Pan gesture guard failed - arView: \(arView != nil), cameraAnchor: \(cameraAnchor != nil)")
            return
        }
        logDebug("✅ Pan gesture proceeding with arView and cameraAnchor")

        let translation = gesture.translation(in: arView)

        switch gesture.state {
        case .began:
            // Store initial position but don't reset accumulated rotation
            initialCameraTransform = cameraAnchor.transform
            initialTouchPoint = gesture.location(in: arView)
            lastPanTranslation = translation

        case .changed:
            logDebug("🔥 CAMERA GESTURE CHANGED STATE - translation: \(translation)")

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

            logDebug("📷 Accumulated rotation: Yaw=\(accumulatedYaw), Pitch=\(accumulatedPitch)")

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
        logDebug("🖐️ TWO-FINGER PAN GESTURE CALLED - State: \(gesture.state.rawValue)")
        guard let _ = arView, let cameraAnchor = cameraAnchor else {
            logDebug("⚠️ Two-finger pan guard failed - arView: \(arView != nil), cameraAnchor: \(cameraAnchor != nil)")
            return
        }
        logDebug("✅ Two-finger pan proceeding with cameraAnchor")

        let translation = gesture.translation(in: arView)

        switch gesture.state {
        case .began:
            initialCameraTransform = cameraAnchor.transform
            initialTouchPoint = gesture.location(in: arView)
            lastPositionPanTranslation = translation  // Initialize for delta calculation

        case .changed:
            // Calculate incremental delta since last update (same approach as single-finger pan)
            let deltaTranslation = CGPoint(
                x: translation.x - lastPositionPanTranslation.x,
                y: translation.y - lastPositionPanTranslation.y
            )

            // Convert delta to position movement
            let deltaX = Float(deltaTranslation.x) * panSensitivity
            let deltaY = Float(-deltaTranslation.y) * panSensitivity // Invert Y for natural up/down

            // Get camera's current transform for directional reference
            let currentTransform = cameraAnchor.transform

            // Calculate camera's right vector in world space (horizontal movement only)
            let cameraRight = normalize(SIMD3<Float>(
                currentTransform.rotation.act(SIMD3<Float>(1, 0, 0)).x,
                0, // Keep horizontal movement
                currentTransform.rotation.act(SIMD3<Float>(1, 0, 0)).z
            ))

            let cameraUp = SIMD3<Float>(0, 1, 0) // World up for vertical movement

            // Calculate incremental camera movement
            let cameraMovement = cameraRight * deltaX + cameraUp * deltaY
            var newPosition = currentTransform.translation + cameraMovement

            // Apply boundary constraints if available
            if let boundaryManager = boundaryManager {
                newPosition = boundaryManager.constrainCameraPosition(newPosition)
            }

            // Apply the new camera position (incremental update)
            var newTransform = currentTransform
            newTransform.translation = newPosition
            cameraAnchor.transform = newTransform

            // Update last translation for next delta calculation
            lastPositionPanTranslation = translation

            logDebug("📷 Two-finger camera movement delta: (\(deltaX), \(deltaY))")

        case .ended, .cancelled:
            initialCameraTransform = cameraAnchor.transform
            initialTouchPoint = nil
            lastPositionPanTranslation = .zero

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

            logDebug("📷 Camera zoom: \(zoomFactor)")

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

            logDebug("📷 Camera rotation gesture: \(rotation) radians")

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

        logDebug("📷 Camera reset to default position and orientation with cleared rotation state")
    }
    
    // Enable/disable gestures based on AR state
    func setGesturesEnabled(_ enabled: Bool) {
        guard let arView = arView else { return }

        for gestureRecognizer in arView.gestureRecognizers ?? [] {
            gestureRecognizer.isEnabled = enabled
        }

        logDebug("🎮 Gestures \(enabled ? "enabled" : "disabled")")
    }

}

// MARK: - Helper functions

private func normalize(_ vector: SIMD3<Float>) -> SIMD3<Float> {
    let length = sqrt(vector.x * vector.x + vector.y * vector.y + vector.z * vector.z)
    return length > 0 ? SIMD3<Float>(vector.x / length, vector.y / length, vector.z / length) : vector
}