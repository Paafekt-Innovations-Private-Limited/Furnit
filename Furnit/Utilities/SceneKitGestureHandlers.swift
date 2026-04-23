import UIKit
import SwiftUI
import SceneKit

/// SceneKit-based gesture handlers - mirrors RealityKitGestureHandlers behavior
/// Provides consistent gesture behavior for SceneKit rooms (vintage/cozy/manual):
/// - Single finger: rotate (look around)
/// - Two fingers: strafe (move left/right/up/down)
/// - Pinch: zoom (move forward/backward)
class SceneKitGestureHandlers: NSObject {

    // View reference
    private weak var containerView: UIView?

    // Camera reference
    private weak var cameraNode: SCNNode?

    // Gesture recognizers
    private var singlePanGesture: UIPanGestureRecognizer?
    private var doublePanGesture: UIPanGestureRecognizer?
    private var pinchGesture: UIPinchGestureRecognizer?

    // Rotation state
    private var accumulatedYaw: Float = 0.0
    private var accumulatedPitch: Float = 0.0
    private var lastPanTranslation: CGPoint = .zero
    private var lastPositionPanTranslation: CGPoint = .zero
    private var initialCameraPosition: SCNVector3 = SCNVector3Zero

    // Sensitivity settings
    private let rotationSensitivity: Float = 0.005
    private let panSensitivity: Float = 0.005
    private let zoomSensitivity: Float = 2.5  // 5x faster zoom (was 0.5)
    private let maxPitch: Float = Float.pi / 4.0  // 45 degrees

    // MARK: - Initialization

    init(containerView: UIView) {
        self.containerView = containerView
        super.init()
        setupGestureRecognizers()
    }

    // MARK: - Camera Setup

    /// Set camera reference
    func setCameraNode(_ node: SCNNode) {
        self.cameraNode = node
        initializeRotationState()
        logDebug("📷 [SceneKitGesture] Camera node set")
    }

    private func initializeRotationState() {
        guard let camera = cameraNode else { return }
        // Extract current yaw from camera euler angles
        accumulatedYaw = camera.eulerAngles.y
        accumulatedPitch = camera.eulerAngles.x
        logDebug("📷 [SceneKitGesture] Initial rotation: yaw=\(accumulatedYaw), pitch=\(accumulatedPitch)")
    }

    // MARK: - Gesture Setup

    private func setupGestureRecognizers() {
        guard let view = containerView else { return }

        // Single-finger pan for rotation
        singlePanGesture = UIPanGestureRecognizer(target: self, action: #selector(handleRotationPan(_:)))
        singlePanGesture?.minimumNumberOfTouches = 1
        singlePanGesture?.maximumNumberOfTouches = 1
        if let gesture = singlePanGesture {
            view.addGestureRecognizer(gesture)
        }

        // Two-finger pan for strafe movement
        doublePanGesture = UIPanGestureRecognizer(target: self, action: #selector(handleStrafePan(_:)))
        doublePanGesture?.minimumNumberOfTouches = 2
        doublePanGesture?.maximumNumberOfTouches = 2
        if let gesture = doublePanGesture {
            view.addGestureRecognizer(gesture)
        }

        // Pinch for zoom (forward/backward)
        pinchGesture = UIPinchGestureRecognizer(target: self, action: #selector(handlePinch(_:)))
        if let gesture = pinchGesture {
            view.addGestureRecognizer(gesture)
        }

        logDebug("🎮 [SceneKitGesture] Gesture recognizers set up")
        logDebug("🎮 [SceneKitGesture] view frame=\(view.frame) bounds=\(view.bounds) interaction=\(view.isUserInteractionEnabled)")
        logDebug("   Single finger: rotate (look around)")
        logDebug("   Two fingers: strafe (move left/right/up/down)")
        logDebug("   Pinch: zoom (forward/backward)")
    }

    // MARK: - Gesture Handlers

    /// Handle single-finger pan for rotation (look around)
    @objc private func handleRotationPan(_ gesture: UIPanGestureRecognizer) {
        guard let camera = cameraNode else {
            logDebug("⚠️ [SceneKitGesture] Rotation pan - no camera")
            return
        }

        let translation = gesture.translation(in: gesture.view)

        switch gesture.state {
        case .began:
            lastPanTranslation = translation
            initialCameraPosition = camera.position
            logDebug("🚨 [SceneKitGesture] Rotation pan BEGAN touches=\(gesture.numberOfTouches) translation=\(translation) pos=\(camera.position) euler=\(camera.eulerAngles)")

        case .changed:
            // Calculate delta
            let deltaX = Float(translation.x - lastPanTranslation.x)
            let deltaY = Float(translation.y - lastPanTranslation.y)
            lastPanTranslation = translation

            // Update accumulated rotation
            accumulatedYaw += -deltaX * rotationSensitivity
            accumulatedPitch += -deltaY * rotationSensitivity

            // Clamp pitch to prevent over-rotation
            accumulatedPitch = max(-maxPitch, min(maxPitch, accumulatedPitch))

            // Apply rotation using euler angles (simpler for SceneKit)
            camera.eulerAngles = SCNVector3(accumulatedPitch, accumulatedYaw, 0)

            // Keep position locked during rotation
            camera.position = initialCameraPosition
            logDebug("🚨 [SceneKitGesture] Rotation CHANGED dx=\(deltaX) dy=\(deltaY) yaw=\(accumulatedYaw) pitch=\(accumulatedPitch)")

        case .ended, .cancelled:
            lastPanTranslation = .zero
            logDebug("🚨 [SceneKitGesture] Rotation pan ENDED pos=\(camera.position) euler=\(camera.eulerAngles)")

        default:
            break
        }
    }

    /// Handle two-finger pan for strafe movement
    @objc private func handleStrafePan(_ gesture: UIPanGestureRecognizer) {
        guard let camera = cameraNode else {
            logDebug("⚠️ [SceneKitGesture] Strafe pan - no camera")
            return
        }

        let translation = gesture.translation(in: gesture.view)

        switch gesture.state {
        case .began:
            lastPositionPanTranslation = translation
            logDebug("🖐️ [SceneKitGesture] Strafe pan BEGAN touches=\(gesture.numberOfTouches) translation=\(translation) pos=\(camera.position)")

        case .changed:
            // Calculate delta
            let deltaX = Float(translation.x - lastPositionPanTranslation.x)
            let deltaY = Float(translation.y - lastPositionPanTranslation.y)
            lastPositionPanTranslation = translation

            // Get camera's right vector (for horizontal strafe)
            let right = camera.worldRight

            // Calculate movement in XZ plane (horizontal strafe) and Y axis (vertical)
            let rightMovement = SCNVector3(right.x * deltaX * panSensitivity, 0, right.z * deltaX * panSensitivity)
            let upMovement = SCNVector3(0, -deltaY * panSensitivity, 0)

            // Apply movement
            camera.position = SCNVector3(
                camera.position.x + rightMovement.x + upMovement.x,
                camera.position.y + rightMovement.y + upMovement.y,
                camera.position.z + rightMovement.z + upMovement.z
            )
            logDebug("🖐️ [SceneKitGesture] Strafe CHANGED dx=\(deltaX) dy=\(deltaY) right=\(right) pos=\(camera.position)")

        case .ended, .cancelled:
            lastPositionPanTranslation = .zero
            logDebug("🖐️ [SceneKitGesture] Strafe pan ENDED pos=\(camera.position)")

        default:
            break
        }
    }

    /// Handle pinch for zoom (forward/backward movement)
    @objc private func handlePinch(_ gesture: UIPinchGestureRecognizer) {
        guard let camera = cameraNode else {
            logDebug("⚠️ [SceneKitGesture] Pinch - no camera")
            return
        }

        switch gesture.state {
        case .began:
            initialCameraPosition = camera.position
            logDebug("🔍 [SceneKitGesture] Pinch BEGAN touches=\(gesture.numberOfTouches) scale=\(gesture.scale) pos=\(camera.position)")

        case .changed:
            let scale = gesture.scale
            let zoomDelta = Float(scale - 1.0) * zoomSensitivity

            // Get forward direction (camera looks along -Z in local space)
            let forward = camera.worldFront

            // Move along forward direction (XZ plane only to stay at same height)
            let forwardXZ = normalize(SCNVector3(forward.x, 0, forward.z))

            camera.position = SCNVector3(
                initialCameraPosition.x + forwardXZ.x * zoomDelta,
                initialCameraPosition.y,
                initialCameraPosition.z + forwardXZ.z * zoomDelta
            )
            logDebug("🔍 [SceneKitGesture] Pinch CHANGED scale=\(scale) zoomDelta=\(zoomDelta) forward=\(forward) pos=\(camera.position)")

        case .ended, .cancelled:
            logDebug("🔍 [SceneKitGesture] Pinch ENDED pos=\(camera.position)")

        default:
            break
        }
    }

    // MARK: - Utility

    private func normalize(_ v: SCNVector3) -> SCNVector3 {
        let len = sqrt(v.x * v.x + v.y * v.y + v.z * v.z)
        if len > 0.001 {
            return SCNVector3(v.x / len, v.y / len, v.z / len)
        }
        return SCNVector3(0, 0, -1)
    }

    // MARK: - Cleanup

    func removeGestures() {
        guard let view = containerView else { return }

        if let gesture = singlePanGesture {
            view.removeGestureRecognizer(gesture)
        }
        if let gesture = doublePanGesture {
            view.removeGestureRecognizer(gesture)
        }
        if let gesture = pinchGesture {
            view.removeGestureRecognizer(gesture)
        }

        singlePanGesture = nil
        doublePanGesture = nil
        pinchGesture = nil

        logDebug("🎮 [SceneKitGesture] Gestures removed")
    }
}

// MARK: - SwiftUI Gesture Overlay for SceneKit

/// A transparent UIView overlay that captures gestures for SceneKit camera control
/// Use this on top of SceneView to enable gesture-based camera control
class SceneKitGestureOverlayView: UIView {
    private var gestureHandler: SceneKitGestureHandlers?

    override init(frame: CGRect) {
        super.init(frame: frame)
        setup()
    }

    required init?(coder: NSCoder) {
        super.init(coder: coder)
        setup()
    }

    private func setup() {
        backgroundColor = .clear
        isUserInteractionEnabled = true
        gestureHandler = SceneKitGestureHandlers(containerView: self)
        logDebug("🪟 [SceneKitGesture] Overlay setup frame=\(frame) bounds=\(bounds)")
    }

    /// Set the camera node to control
    func setCameraNode(_ node: SCNNode) {
        gestureHandler?.setCameraNode(node)
        logDebug("🪟 [SceneKitGesture] Overlay setCameraNode pos=\(node.position) euler=\(node.eulerAngles)")
    }

    override func layoutSubviews() {
        super.layoutSubviews()
        logDebug("🪟 [SceneKitGesture] Overlay layout frame=\(frame) bounds=\(bounds)")
    }

    override func touchesBegan(_ touches: Set<UITouch>, with event: UIEvent?) {
        super.touchesBegan(touches, with: event)
        logDebug("🪟 [SceneKitGesture] Overlay touchesBegan count=\(touches.count)")
    }

    deinit {
        gestureHandler?.removeGestures()
    }
}

/// SwiftUI wrapper for SceneKitGestureOverlayView
struct SceneKitGestureOverlay: UIViewRepresentable {
    let cameraNode: SCNNode?

    func makeUIView(context: Context) -> SceneKitGestureOverlayView {
        let view = SceneKitGestureOverlayView()
        if let camera = cameraNode {
            view.setCameraNode(camera)
        }
        return view
    }

    func updateUIView(_ uiView: SceneKitGestureOverlayView, context: Context) {
        if let camera = cameraNode {
            uiView.setCameraNode(camera)
        }
    }
}
