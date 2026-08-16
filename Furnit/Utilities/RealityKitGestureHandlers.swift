import RealityKit
import UIKit

// RealityKit-based gesture handlers to replace SceneKit gesture handlers
// Inherits from NSObject for Objective-C gesture recognizer target-action compatibility
class RealityKitGestureHandlers: NSObject, UIGestureRecognizerDelegate {
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
    private var lastPinchScale: CGFloat = 1
    private var capturedFrustumPinchStartFieldOfView: Float = 60
    private var initialTouchPoint: CGPoint?
    private var infiniteZoomEnabled = false

    // Accumulated rotation state to prevent flickering and maintain smooth rotation
    private var accumulatedYaw: Float = 0.0    // Horizontal rotation around Y-axis
    private var accumulatedPitch: Float = 0.0  // Vertical rotation around X-axis
    /// The one camera origin from which a single-photo depth surface was authored. Unlike a
    /// volumetric room, this surface has no valid translated viewpoints: moving the eye exposes
    /// geometry that the source photograph never observed.
    private var capturedPhotoOpticalCenter: SIMD3<Float>?
    private var usesFlatPhotoNavigation = false
    private var layeredPhotoYawLimit: Float?
    private var layeredPhotoPitchLimit: Float?
    private var layeredPhotoBaseYaw: Float?
    private var layeredPhotoBasePitch: Float?
    private var layeredPhotoSourceHalfFovX: Float?
    private var layeredPhotoSourceHalfFovY: Float?
    private var layeredPhotoNearestDepth: Float?
    private var layeredPhotoCapturePosition: SIMD3<Float>?
    private var layeredPhotoCaptureForward: SIMD3<Float>?
    private var layeredPhotoCaptureRight: SIMD3<Float>?
    private var layeredPhotoCaptureUp: SIMD3<Float>?

    // Exterior inspection orbits around the framed model. Once the camera is within a genuine
    // room volume, single-finger drag becomes first-person yaw/pitch and keeps the eye fixed.
    // Single-photo depth surfaces turn in place while preserving their authored optical center.
    /// World-space point the viewer framed the room around, published by ``RealityKitView``.
    private var orbitTarget: SIMD3<Float>?
    /// Pivot and radius captured at gesture start so the swing never snaps on first movement.
    private var activeOrbitPivot: SIMD3<Float> = .zero
    private var activeOrbitRadius: Float = 0
    private var activeRotationTurnsInPlace = false
    private let minimumOrbitRadius: Float = 0.35
    private let fallbackOrbitRadius: Float = 3.0


    // Note: Using total translation from gesture start instead of cumulative tracking for smoother rotation
    
    // Pan gesture configuration
    private let panSensitivity = DepthAnythingPhotoCameraInteraction.positionTranslationMetersPerPoint
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

    /// Keeps the normal room/photo navigation contract unchanged when disabled. When enabled,
    /// pinch zoom may cross the camera boundary and uses the same wide camera range as Android.
    func setInfiniteZoomEnabled(_ enabled: Bool) {
        infiniteZoomEnabled = enabled
        cameraEntity?.camera.near = enabled ? 0.001 : 0.1
        cameraEntity?.camera.far = enabled ? 1000.0 : 100.0
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
        camera.camera.near = infiniteZoomEnabled ? 0.001 : 0.1
        camera.camera.far = infiniteZoomEnabled ? 1000.0 : 100.0

        // Initialize accumulated rotation from camera's current orientation
        initializeRotationFromCamera()

        logDebug("📷 Camera references set for direct camera control")
    }

    /// Re-sync rotation state after camera is repositioned (e.g., after model loads)
    /// Call this after setting camera position/rotation programmatically
    func syncRotationState() {
        initializeRotationFromCamera()
    }

    /// Installs or clears the navigation contract for a projective single-photo room.
    /// Every camera input path consults this value directly instead of inferring the contract from
    /// mutable room bounds, so gesture setup order cannot accidentally re-enable translation.
    func setCapturedPhotoOpticalCenter(_ opticalCenter: SIMD3<Float>?) {
        capturedPhotoOpticalCenter = opticalCenter
        if opticalCenter != nil {
            usesFlatPhotoNavigation = false
            layeredPhotoYawLimit = nil
            layeredPhotoPitchLimit = nil
            layeredPhotoBaseYaw = nil
            layeredPhotoBasePitch = nil
            clearLayeredPhotoCoverageContract()
        }
        if let opticalCenter, let cameraAnchor {
            var transform = cameraAnchor.transform
            transform.translation = opticalCenter
            cameraAnchor.transform = transform
        }
        initializeRotationFromCamera()
    }

    func setFlatPhotoNavigationEnabled(_ enabled: Bool) {
        usesFlatPhotoNavigation = enabled
        if enabled {
            capturedPhotoOpticalCenter = nil
            layeredPhotoYawLimit = nil
            layeredPhotoPitchLimit = nil
            layeredPhotoBaseYaw = nil
            layeredPhotoBasePitch = nil
            clearLayeredPhotoCoverageContract()
        }
        initializeRotationFromCamera()
    }

    func setLayeredPhotoLookLimits(
        maximumYaw: Float?,
        maximumPitch: Float?,
        sourceHorizontalFieldOfView: Float? = nil,
        sourceVerticalFieldOfView: Float? = nil,
        nearestReliableDepth: Float? = nil
    ) {
        layeredPhotoYawLimit = maximumYaw.flatMap { $0.isFinite && $0 > 0 ? min($0, .pi) : nil }
        layeredPhotoPitchLimit = maximumPitch.flatMap {
            $0.isFinite && $0 > 0 ? min($0, Float.pi / 2 - 0.05) : nil
        }
        if layeredPhotoYawLimit != nil { capturedPhotoOpticalCenter = nil }
        initializeRotationFromCamera()
        if layeredPhotoYawLimit != nil {
            // A Depth Anything room is authored looking toward +Z, which is world yaw π for a
            // RealityKit camera. Limits are an envelope around that captured heading, not around
            // world yaw zero; clamping around zero makes the first drag turn ~180° away.
            layeredPhotoBaseYaw = accumulatedYaw
            layeredPhotoBasePitch = accumulatedPitch
            layeredPhotoSourceHalfFovX = sourceHorizontalFieldOfView.map { $0 * .pi / 360 }
            layeredPhotoSourceHalfFovY = sourceVerticalFieldOfView.map { $0 * .pi / 360 }
            layeredPhotoNearestDepth = nearestReliableDepth
            if let cameraAnchor {
                let rotation = cameraAnchor.transform.rotation
                layeredPhotoCapturePosition = cameraAnchor.transform.translation
                layeredPhotoCaptureForward = rotation.act(SIMD3<Float>(0, 0, -1))
                layeredPhotoCaptureRight = rotation.act(SIMD3<Float>(1, 0, 0))
                layeredPhotoCaptureUp = rotation.act(SIMD3<Float>(0, 1, 0))
            }
        } else {
            layeredPhotoBaseYaw = nil
            layeredPhotoBasePitch = nil
            clearLayeredPhotoCoverageContract()
        }
    }

    private func clearLayeredPhotoCoverageContract() {
        layeredPhotoSourceHalfFovX = nil
        layeredPhotoSourceHalfFovY = nil
        layeredPhotoNearestDepth = nil
        layeredPhotoCapturePosition = nil
        layeredPhotoCaptureForward = nil
        layeredPhotoCaptureRight = nil
        layeredPhotoCaptureUp = nil
    }

    /// Reinterpret the D-pad as bounded look controls for captured-photo rooms so it follows the
    /// same authored heading and disocclusion envelope as touch rotation.
    @discardableResult
    func nudgeCapturedPhotoView(yawDelta: Float, pitchDelta: Float) -> Bool {
        guard let cameraAnchor,
              capturedPhotoOpticalCenter != nil || layeredPhotoYawLimit != nil else {
            return false
        }
        accumulatedYaw += yawDelta
        accumulatedPitch += pitchDelta
        constrainLayeredPhotoLookIfNeeded()
        if layeredPhotoPitchLimit == nil {
            let maxPitch = Float.pi / 2.0 - 0.05
            accumulatedPitch = max(-maxPitch, min(maxPitch, accumulatedPitch))
        }
        var transform = cameraAnchor.transform
        transform.translation = capturedPhotoOpticalCenter ?? transform.translation
        let yawRotation = simd_quatf(angle: accumulatedYaw, axis: SIMD3<Float>(0, 1, 0))
        let pitchRotation = simd_quatf(angle: accumulatedPitch, axis: SIMD3<Float>(1, 0, 0))
        transform.rotation = yawRotation * pitchRotation
        cameraAnchor.transform = transform
        return true
    }

    private func constrainLayeredPhotoLookIfNeeded() {
        let coverageLimits = currentLayeredPhotoCoverageLimits()
        if let authoredLimit = layeredPhotoYawLimit,
           let base = layeredPhotoBaseYaw {
            let limit = min(authoredLimit, coverageLimits?.yaw ?? authoredLimit)
            accumulatedYaw = Self.angleNearest(accumulatedYaw, to: base)
            accumulatedYaw = max(base - limit, min(base + limit, accumulatedYaw))
        }
        if let authoredLimit = layeredPhotoPitchLimit,
           let base = layeredPhotoBasePitch {
            let limit = min(authoredLimit, coverageLimits?.pitch ?? authoredLimit)
            accumulatedPitch = max(base - limit, min(base + limit, accumulatedPitch))
        }
    }

    /// Returns only the look margin actually covered by the captured image at the camera's current
    /// zoom position. Moving forward creates overscan; at Fit the margin is zero on the limiting
    /// axis. This prevents any camera action from revealing or stretching pixels outside the photo.
    private func currentLayeredPhotoCoverageLimits() -> (yaw: Float, pitch: Float)? {
        guard let cameraAnchor,
              let cameraEntity,
              let arView,
              let sourceHalfX = layeredPhotoSourceHalfFovX,
              let sourceHalfY = layeredPhotoSourceHalfFovY,
              let nearestDepth = layeredPhotoNearestDepth,
              let capturePosition = layeredPhotoCapturePosition,
              let captureForward = layeredPhotoCaptureForward,
              let captureRight = layeredPhotoCaptureRight,
              let captureUp = layeredPhotoCaptureUp,
              nearestDepth > 0.2,
              arView.bounds.width > 1,
              arView.bounds.height > 1 else { return nil }

        let displacement = cameraAnchor.transform.translation - capturePosition
        let forward = max(0, simd_dot(displacement, captureForward))
        let lateral = abs(simd_dot(displacement, captureRight))
        let vertical = abs(simd_dot(displacement, captureUp))
        let remainingDepth = max(nearestDepth - forward, nearestDepth * 0.25)
        let viewportAspect = Float(arView.bounds.width / arView.bounds.height)
        let cameraHalfY = cameraEntity.camera.fieldOfViewInDegrees * .pi / 360
        let cameraHalfX = atan(tan(cameraHalfY) * max(viewportAspect, 0.01))

        let sourceHalfWidth = tan(sourceHalfX) * nearestDepth
        let sourceHalfHeight = tan(sourceHalfY) * nearestDepth
        let coveredHalfX = atan(max(sourceHalfWidth - lateral, 0.001) / remainingDepth)
        let coveredHalfY = atan(max(sourceHalfHeight - vertical, 0.001) / remainingDepth)
        return (
            yaw: max(0, coveredHalfX - cameraHalfX - 0.01),
            pitch: max(0, coveredHalfY - cameraHalfY - 0.01)
        )
    }

    private static func angleNearest(_ angle: Float, to reference: Float) -> Float {
        var delta = angle - reference
        while delta > .pi { delta -= 2 * .pi }
        while delta < -.pi { delta += 2 * .pi }
        return reference + delta
    }

    /// Set the world-space point single-finger drag should swing the camera around.
    /// Passing `nil` falls back to a fixed radius ahead of the camera.
    func setOrbitTarget(_ target: SIMD3<Float>?) {
        orbitTarget = target
        // The viewer only republishes this after it has moved the camera, so the
        // accumulated angles must be re-derived from the new pose at the same time.
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
        // Positive pitch means looking up, matching the sign of the pitch quaternion
        // applied in `handlePanGesture`. Deriving the opposite sign here would flip the
        // camera vertically on the first drag after any programmatic reframe or D-pad step.
        // Using the length of horizontal component for proper angle calculation
        let horizontalLength = sqrt(forward.x * forward.x + forward.z * forward.z)
        let pitch = atan2(forward.y, horizontalLength)

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
            singlePan.delegate = self
            arView.addGestureRecognizer(singlePan)
        }
        
        // Two-finger pan gesture for position movement (left/right/up/down)
        doublePanGesture = UIPanGestureRecognizer(target: self, action: #selector(handlePositionPanGesture(_:)))
        doublePanGesture?.minimumNumberOfTouches = 2
        doublePanGesture?.maximumNumberOfTouches = 2
        if let doublePan = doublePanGesture {
            doublePan.delegate = self
            arView.addGestureRecognizer(doublePan)
        }

        // Pinch gesture for zoom
        pinchGesture = UIPinchGestureRecognizer(target: self, action: #selector(handlePinchGesture(_:)))
        if let pinch = pinchGesture {
            pinch.delegate = self
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

    func gestureRecognizer(
        _ gestureRecognizer: UIGestureRecognizer,
        shouldRecognizeSimultaneouslyWith otherGestureRecognizer: UIGestureRecognizer
    ) -> Bool {
        let pair = [gestureRecognizer, otherGestureRecognizer]
        return pair.contains { $0 === pinchGesture } && pair.contains { $0 === doublePanGesture }
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

    /// Capture the swing pivot and radius at gesture start. The pivot is placed along the
    /// camera's *current* forward direction at the room's distance, so the first movement
    /// continues from exactly where the camera already is instead of snapping onto the
    /// orbit sphere.
    private func beginOrbit(from cameraAnchor: AnchorEntity) {
        let position = cameraAnchor.transform.translation
        let forward = cameraAnchor.transform.rotation.act(SIMD3<Float>(0, 0, -1))

        let radius: Float
        if let orbitTarget {
            radius = max(simd_length(orbitTarget - position), minimumOrbitRadius)
        } else {
            radius = fallbackOrbitRadius
        }

        activeOrbitRadius = radius
        activeOrbitPivot = position + forward * radius
    }

    // Handle pan gesture with geometry-derived navigation: orbit outside, turn in place inside.
    @objc private func handlePanGesture(_ gesture: UIPanGestureRecognizer) {
        logDebug("🚨 PAN GESTURE CALLED - State: \(gesture.state.rawValue)")
        guard let arView = arView, let cameraAnchor = cameraAnchor else {
            logDebug("⚠️ Pan gesture guard failed - arView: \(arView != nil), cameraAnchor: \(cameraAnchor != nil)")
            return
        }
        logDebug("✅ Pan gesture proceeding with arView and cameraAnchor")

        let translation = gesture.translation(in: arView)

        if usesFlatPhotoNavigation {
            switch gesture.state {
            case .began:
                initialCameraTransform = cameraAnchor.transform
            case .changed:
                var transform = initialCameraTransform
                transform.translation.x = initialCameraTransform.translation.x
                    + Float(translation.x) * panSensitivity
                transform.translation.y = initialCameraTransform.translation.y
                    - Float(translation.y) * panSensitivity
                cameraAnchor.transform = transform
            case .ended, .cancelled:
                initialCameraTransform = cameraAnchor.transform
                lastPanTranslation = .zero
            default:
                break
            }
            return
        }

        switch gesture.state {
        case .began:
            if let opticalCenter = capturedPhotoOpticalCenter {
                var transform = cameraAnchor.transform
                transform.translation = opticalCenter
                cameraAnchor.transform = transform
            }
            // Store initial position but don't reset accumulated rotation
            initialCameraTransform = cameraAnchor.transform
            initialTouchPoint = gesture.location(in: arView)
            lastPanTranslation = translation
            activeRotationTurnsInPlace = boundaryManager?
                .isInsideNavigableInterior(cameraAnchor.transform.translation) == true ||
                capturedPhotoOpticalCenter != nil || layeredPhotoYawLimit != nil
            if !activeRotationTurnsInPlace {
                beginOrbit(from: cameraAnchor)
            }

        case .changed:
            logDebug("🔥 CAMERA GESTURE CHANGED STATE - translation: \(translation)")

            // Calculate incremental rotation delta since last update
            let deltaTranslation = CGPoint(
                x: translation.x - lastPanTranslation.x,
                y: translation.y - lastPanTranslation.y
            )

            let rotationDelta = DepthAnythingPhotoCameraInteraction.rotationDelta(
                for: deltaTranslation
            )

            // Update accumulated rotation values
            accumulatedYaw += rotationDelta.yaw
            accumulatedPitch += rotationDelta.pitch

            constrainLayeredPhotoLookIfNeeded()

            // Prevent vertical inversion. Captured rooms allow near-vertical looking; normal
            // room navigation retains its tighter 45-degree pitch limit.
            let usesCapturedPhotoFrustum = capturedPhotoOpticalCenter != nil
            if usesCapturedPhotoFrustum {
                let maxPitch = Float.pi / 2.0 - 0.05
                accumulatedPitch = max(-maxPitch, min(maxPitch, accumulatedPitch))
            } else if layeredPhotoPitchLimit != nil {
                constrainLayeredPhotoLookIfNeeded()
            } else {
                let maxPitch = Float.pi / 4.0
                accumulatedPitch = max(-maxPitch, min(maxPitch, accumulatedPitch))
            }

            // Create rotation quaternions from accumulated values (prevents tilting by only using yaw and pitch)
            let yawRotation = simd_quatf(angle: accumulatedYaw, axis: SIMD3<Float>(0, 1, 0))     // Horizontal only
            let pitchRotation = simd_quatf(angle: accumulatedPitch, axis: SIMD3<Float>(1, 0, 0)) // Vertical only

            // Combine rotations: apply pitch first, then yaw (no roll component to prevent tilting)
            let combinedRotation = yawRotation * pitchRotation

            var newTransform = Transform()
            if activeRotationTurnsInPlace {
                newTransform.translation = capturedPhotoOpticalCenter ?? initialCameraTransform.translation
            } else {
                // Exterior inspection keeps the room framed by swinging around its pivot.
                let newForward = combinedRotation.act(SIMD3<Float>(0, 0, -1))
                var orbitPosition = activeOrbitPivot - newForward * activeOrbitRadius
                if let boundaryManager = boundaryManager {
                    orbitPosition = boundaryManager.constrainCameraPosition(orbitPosition)
                }
                newTransform.translation = orbitPosition
            }
            newTransform.rotation = combinedRotation  // Apply accumulated rotation (no roll/tilt)
            newTransform.scale = initialCameraTransform.scale
            cameraAnchor.transform = newTransform

            logDebug(
                "📷 \(activeRotationTurnsInPlace ? "First person" : "Orbit"): " +
                    "Yaw=\(accumulatedYaw), Pitch=\(accumulatedPitch), radius=\(activeOrbitRadius)"
            )

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

        if usesFlatPhotoNavigation {
            if gesture.state == .ended || gesture.state == .cancelled {
                initialTouchPoint = nil
                lastPositionPanTranslation = .zero
            }
            return
        }

        if let opticalCenter = capturedPhotoOpticalCenter {
            var transform = cameraAnchor.transform
            transform.translation = opticalCenter
            cameraAnchor.transform = transform
            // The saved single-photo surface is defined around one optical center. Translation
            // reveals surfaces that were occluded in the only source view.
            if gesture.state == .ended || gesture.state == .cancelled {
                initialTouchPoint = nil
                lastPositionPanTranslation = .zero
            }
            return
        }

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
            lastPinchScale = 1
            if capturedPhotoOpticalCenter != nil || usesFlatPhotoNavigation,
               let cameraEntity {
                capturedFrustumPinchStartFieldOfView = cameraEntity.camera.fieldOfViewInDegrees
            }
            
        case .changed:
            // Match Android's non-layered photo path: keep the authored optical center and
            // expand projection zoom. Moving the eye for this room type distorts the source
            // photograph and does not behave like Android.
            if (capturedPhotoOpticalCenter != nil || usesFlatPhotoNavigation),
               let cameraEntity {
                if let opticalCenter = capturedPhotoOpticalCenter {
                    var transform = cameraAnchor.transform
                    transform.translation = opticalCenter
                    cameraAnchor.transform = transform
                }
                let scale = max(Float(gesture.scale), 0.01)
                let initialHalfFov = capturedFrustumPinchStartFieldOfView * .pi / 360
                let zoomedFieldOfView = 2 * atan(tan(initialHalfFov) / scale) * 180 / .pi
                let minimumFieldOfView: Float = infiniteZoomEnabled ? 0.05 : 12
                let maximumFieldOfView: Float = infiniteZoomEnabled ? 175 : 120
                cameraEntity.camera.fieldOfViewInDegrees = min(
                    max(zoomedFieldOfView, minimumFieldOfView),
                    maximumFieldOfView
                )
                lastPinchScale = gesture.scale
                return
            }
            // Apply only this event's scale delta so a simultaneous two-finger pan can compose
            // with pinch instead of each recognizer restoring its own gesture-start transform.
            let safePreviousScale = max(lastPinchScale, 0.001)
            let scaleDelta = gesture.scale / safePreviousScale
            let zoomFactor = Float(scaleDelta - 1.0)
                * DepthAnythingPhotoCameraInteraction.pinchDollyMetersPerScaleDelta
            
            // Get camera's current transform for directional reference
            let cameraTransform = cameraAnchor.transform
            
            // Move camera forward/backward along its view direction for zoom effect
            let forward = normalize(SIMD3<Float>(
                cameraTransform.rotation.act(SIMD3<Float>(0, 0, -1)).x,
                cameraTransform.rotation.act(SIMD3<Float>(0, 0, -1)).y,
                cameraTransform.rotation.act(SIMD3<Float>(0, 0, -1)).z
            ))
            
            // Camera moves forward/backward for zoom effect
            let cameraMovement = forward * zoomFactor
            var newPosition = cameraTransform.translation + cameraMovement
            
            // Apply boundary constraints
            if !infiniteZoomEnabled, let boundaryManager = boundaryManager {
                newPosition = boundaryManager.constrainCameraPosition(newPosition)
            }
            
            // Apply the new camera position
            var newTransform = cameraTransform
            newTransform.translation = newPosition
            cameraAnchor.transform = newTransform
            if layeredPhotoYawLimit != nil {
                constrainLayeredPhotoLookIfNeeded()
                var constrainedTransform = cameraAnchor.transform
                let yawRotation = simd_quatf(angle: accumulatedYaw, axis: SIMD3<Float>(0, 1, 0))
                let pitchRotation = simd_quatf(angle: accumulatedPitch, axis: SIMD3<Float>(1, 0, 0))
                constrainedTransform.rotation = yawRotation * pitchRotation
                cameraAnchor.transform = constrainedTransform
            }
            lastPinchScale = gesture.scale

        case .ended, .cancelled:
            initialCameraTransform = cameraAnchor.transform
            lastPinchScale = 1
            
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
