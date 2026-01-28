import SwiftUI
import RealityKit
import SceneKit

/// Single global controller for joystick AND camera movement
/// Views register their camera, this controller moves it
class GlobalCameraController {
    static let shared = GlobalCameraController()

    // Joystick state
    @Published var joystickOffset: CGSize = .zero

    // Camera references - only ONE is active at a time
    private weak var realityKitAnchor: AnchorEntity?
    private weak var realityKitCamera: PerspectiveCamera?  // Also store the camera entity
    private weak var sceneKitCamera: SCNNode?
    private weak var arView: ARView?  // Store ARView reference to update cameraTransform

    // Movement
    private var displayLink: CADisplayLink?
    private let moveSpeed: Float = 0.15  // Increased from 0.04 for more visible movement
    private let deadZone: Float = 5.0

    // Smooth movement
    private var currentVelocity: SIMD2<Float> = .zero
    private let smoothing: Float = 0.18

    // Battery optimization - track if displayLink is running
    private var isDisplayLinkActive = false

    private init() {
        // Create display link but DON'T start it - wait for camera registration
        displayLink = CADisplayLink(target: self, selector: #selector(update))
        displayLink?.isPaused = true  // Start paused to save battery
        displayLink?.add(to: .main, forMode: .common)
        logDebug("🎮 [GlobalCameraController] Initialized (displayLink PAUSED - battery saver)")
    }

    // MARK: - Battery Optimization

    /// Resume displayLink when a camera is active
    private func resumeDisplayLink() {
        guard !isDisplayLinkActive else { return }
        displayLink?.isPaused = false
        isDisplayLinkActive = true
        logDebug("🔋 [GlobalCameraController] DisplayLink RESUMED (camera active)")
    }

    /// Pause displayLink when no camera is active
    private func pauseDisplayLink() {
        guard isDisplayLinkActive else { return }
        displayLink?.isPaused = true
        isDisplayLinkActive = false
        currentVelocity = .zero  // Reset velocity
        logDebug("🔋 [GlobalCameraController] DisplayLink PAUSED (no camera - saving battery)")
    }

    /// Check if any camera is registered
    private var hasCameraRegistered: Bool {
        return realityKitAnchor != nil || sceneKitCamera != nil
    }

    // MARK: - Camera Registration

    /// Register RealityKit camera (for saved rooms)
    func registerRealityKitCamera(_ anchor: AnchorEntity, camera: PerspectiveCamera? = nil, arView: ARView? = nil) {
        realityKitAnchor = anchor
        realityKitCamera = camera
        self.arView = arView
        sceneKitCamera = nil  // Clear other type
        resumeDisplayLink()  // Start updates when camera is active
        logDebug("📷 [GlobalCameraController] RealityKit camera registered (anchor + camera entity + ARView)")
    }

    /// Register SceneKit camera (for room preview before saving)
    func registerSceneKitCamera(_ node: SCNNode) {
        sceneKitCamera = node
        realityKitAnchor = nil  // Clear other type
        resumeDisplayLink()  // Start updates when camera is active
        logDebug("📷 [GlobalCameraController] SceneKit camera registered")
    }

    /// Clear camera registration
    func clearCamera() {
        realityKitAnchor = nil
        realityKitCamera = nil
        sceneKitCamera = nil
        arView = nil
        pauseDisplayLink()  // Stop updates to save battery
        logDebug("📷 [GlobalCameraController] Camera cleared")
    }

    // MARK: - Joystick/Drag Input

    /// Last drag translation for delta calculation
    private var lastDragTranslation: CGSize = .zero

    func updateJoystick(_ offset: CGSize) {
        joystickOffset = offset
        if offset.width != 0 || offset.height != 0 {
            logDebug("🎮 [GlobalCameraController] Joystick update: \(offset)")
        }
    }

    /// Update from drag gesture - calculates delta from last position
    func updateFromDrag(_ translation: CGSize) {
        logDebug("🎮 [GlobalCameraController] updateFromDrag called: \(translation)")
        logDebug("   hasRealityKit: \(realityKitAnchor != nil), hasSceneKit: \(sceneKitCamera != nil)")

        // Calculate delta from last translation
        let deltaX = translation.width - lastDragTranslation.width
        let deltaY = translation.height - lastDragTranslation.height

        // Store for next delta calculation
        lastDragTranslation = translation

        // Convert delta to movement (scale for smooth control)
        let scaledOffset = CGSize(width: deltaX * 0.5, height: deltaY * 0.5)
        joystickOffset = scaledOffset

        // Also post notification for WebGL camera movement
        NotificationCenter.default.post(
            name: NSNotification.Name("WebGLJoystickMove"),
            object: nil,
            userInfo: ["offset": scaledOffset]
        )

        if abs(deltaX) > 0.1 || abs(deltaY) > 0.1 {
            logDebug("🎮 [GlobalCameraController] Drag delta: \(deltaX), \(deltaY)")
        }
    }

    /// Reset drag state when gesture ends
    func endDrag() {
        lastDragTranslation = .zero
        joystickOffset = .zero
        logDebug("🎮 [GlobalCameraController] Drag ended")
    }

    // MARK: - Movement Update (60fps)

    @objc private func update() {
        let x = Float(joystickOffset.width)
        let y = Float(joystickOffset.height)
        let magnitude = sqrt(x * x + y * y)

        // Debug: log when joystick is active
        if magnitude > deadZone {
            let hasRealityKit = realityKitAnchor != nil
            let hasSceneKit = sceneKitCamera != nil
            logDebug("🕹️ Joystick: \(magnitude) | RealityKit:\(hasRealityKit) SceneKit:\(hasSceneKit)")
        }

        // Calculate target velocity
        var targetVelocity: SIMD2<Float> = .zero
        if magnitude > deadZone {
            targetVelocity = SIMD2<Float>(x / 30.0, y / 30.0)
        }

        // Smooth velocity
        currentVelocity = currentVelocity + (targetVelocity - currentVelocity) * smoothing

        // Skip if too small
        let speed = sqrt(currentVelocity.x * currentVelocity.x + currentVelocity.y * currentVelocity.y)
        guard speed > 0.01 else { return }

        // Move whichever camera is registered
        if let anchor = realityKitAnchor {
            moveRealityKitCamera(anchor, camera: realityKitCamera)
        } else if let camera = sceneKitCamera {
            moveSceneKitCamera(camera)
        }
    }

    private func moveRealityKitCamera(_ anchor: AnchorEntity, camera: PerspectiveCamera?) {
        let forwardBackward = -currentVelocity.y * moveSpeed
        let leftRight = currentVelocity.x * moveSpeed

        let oldPos = anchor.transform.translation

        let transform = anchor.transform
        let forward = transform.rotation.act(SIMD3<Float>(0, 0, -1))
        let right = transform.rotation.act(SIMD3<Float>(1, 0, 0))

        let forwardXZ = SIMD3<Float>(forward.x, 0, forward.z)
        let rightXZ = SIMD3<Float>(right.x, 0, right.z)

        let normalizedForward = normalize(forwardXZ)
        let normalizedRight = normalize(rightXZ)

        let delta = SIMD3<Float>(
            normalizedForward.x * forwardBackward + normalizedRight.x * leftRight,
            0,
            normalizedForward.z * forwardBackward + normalizedRight.z * leftRight
        )

        // Move the anchor (camera is a child, so it moves with it)
        var newTransform = anchor.transform
        newTransform.translation.x += delta.x
        newTransform.translation.z += delta.z
        anchor.transform = newTransform

        let newPos = anchor.transform.translation
        let hasCameraEntity = camera != nil
        logDebug("📍 Move: \(oldPos) → \(newPos) delta:\(delta) cameraEntity:\(hasCameraEntity)")
        
        // ✅ Try forcing camera entity refresh instead of anchor manipulation
        if let cameraEntity = camera {
            cameraEntity.isEnabled = false
            cameraEntity.isEnabled = true
            logDebug("   🔄 Camera entity disabled/enabled to force update")
        }
    }

    private func moveSceneKitCamera(_ camera: SCNNode) {
        logDebug("📷 [SceneKit] moveSceneKitCamera CALLED - velocity: \(currentVelocity)")
        let normalizedX = currentVelocity.x
        let normalizedY = -currentVelocity.y  // Invert for intuitive control

        let forward = camera.worldFront
        let right = camera.worldRight

        logDebug("📷 [SceneKit] velocity: (\(normalizedX), \(normalizedY)) forward: \(forward) right: \(right)")

        let moveX = right.x * normalizedX * moveSpeed + forward.x * normalizedY * moveSpeed
        let moveZ = right.z * normalizedX * moveSpeed + forward.z * normalizedY * moveSpeed

        let oldPos = camera.position
        camera.position.x += moveX
        camera.position.z += moveZ
        logDebug("📍 [SceneKit] Move: (\(oldPos.x), \(oldPos.z)) → (\(camera.position.x), \(camera.position.z)) delta: (\(moveX), \(moveZ))")
    }

    private func normalize(_ v: SIMD3<Float>) -> SIMD3<Float> {
        let len = sqrt(v.x * v.x + v.y * v.y + v.z * v.z)
        return len > 0 ? v / len : v
    }
}

// MARK: - Touch-Anywhere Drag Overlay (replaces joystick)

struct TouchDragOverlay: View {
    var photoOrientation: PhotoOrientation = .portrait
    @State private var isDragging = false

    // Sensitivity for drag-to-camera movement conversion
    private let dragSensitivity: CGFloat = 0.8

    var body: some View {
        GeometryReader { geometry in
            ZStack {
                // Transparent drag area (full screen)
                Color.clear
                    .contentShape(Rectangle())
                    .highPriorityGesture(
                        DragGesture(minimumDistance: 1)
                            .onChanged { value in
                                logDebug("👆 [TouchDragOverlay] onChanged: \(value.translation)")
                                isDragging = true
                                // Use delta-based drag for smooth continuous movement
                                GlobalCameraController.shared.updateFromDrag(value.translation)
                            }
                            .onEnded { _ in
                                logDebug("👆 [TouchDragOverlay] onEnded")
                                isDragging = false
                                // Reset drag state
                                GlobalCameraController.shared.endDrag()
                            }
                    )

                // Bottom hint and orientation label
                VStack {
                    Spacer()
                    VStack(spacing: 8) {
                        // Drag hint
                        Text("Drag to move camera")
                            .font(.caption)
                            .foregroundColor(.white.opacity(0.7))
                            .shadow(color: .black, radius: 2, x: 1, y: 1)

                        // Orientation label
                        VStack(spacing: 1) {
                            Text(orientationSubtitle)
                                .font(.caption2)
                            Text(orientationTitle)
                                .font(.caption2)
                                .fontWeight(.medium)
                        }
                        .foregroundColor(.white.opacity(0.8))
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(Color.black.opacity(0.4))
                        .cornerRadius(6)
                    }
                    .padding(.bottom, 40)
                }
                .allowsHitTesting(false) // Let touches pass through to drag gesture
            }
        }
    }

    private var orientationTitle: String {
        switch photoOrientation {
        case .portrait, .square:
            return NSLocalizedString("orientation.portrait", comment: "")
        case .landscape:
            return NSLocalizedString("orientation.landscape", comment: "")
        }
    }

    private var orientationSubtitle: String {
        switch photoOrientation {
        case .portrait, .square:
            return NSLocalizedString("orientation.heldVertically", comment: "")
        case .landscape:
            return NSLocalizedString("orientation.heldHorizontally", comment: "")
        }
    }
}

// MARK: - Simple Joystick Overlay (legacy - use TouchDragOverlay instead)

struct SimpleJoystickOverlay: View {
    @State private var offset: CGSize = .zero
    var photoOrientation: PhotoOrientation = .portrait

    var body: some View {
        // Now just wraps TouchDragOverlay for backwards compatibility
        TouchDragOverlay(photoOrientation: photoOrientation)
    }
}
