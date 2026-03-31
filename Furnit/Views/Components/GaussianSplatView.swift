import SwiftUI
import MetalKit
import MetalSplatter
import simd

/// SwiftUI wrapper for MetalSplatter Gaussian splat rendering
/// Renders PLY files containing 3D Gaussian splat data using Metal
struct GaussianSplatView: UIViewRepresentable {

    // MARK: - Properties

    /// URL to the PLY file to render
    let plyURL: URL

    /// Binding to track loading state
    @Binding var isLoading: Bool

    /// Binding to report loading errors
    @Binding var loadError: String?

    /// Binding to control zoom level from parent view
    @Binding var zoomLevel: Float

    /// Optional callback for room bounds (MetalSplatter's `boundingBox` → `RoomBounds`).
    let onBoundsAvailable: ((RoomBounds) -> Void)?

    // MARK: - Constants

    /// Maximum number of simultaneous render operations
    private static let maxSimultaneousRenders = 3

    // MARK: - UIViewRepresentable

    func makeUIView(context: Context) -> MTKView {
        // Create Metal view for rendering
        let mtkView = MTKView()

        // Configure Metal device
        guard let device = MTLCreateSystemDefaultDevice() else {
            DispatchQueue.main.async {
                loadError = "Metal is not supported on this device"
            }
            return mtkView
        }

        mtkView.device = device

        // Match pixel formats from local MetalSplatter SimpleApp
        mtkView.colorPixelFormat = .bgra8Unorm_srgb
        mtkView.depthStencilPixelFormat = .depth32Float
        mtkView.sampleCount = 1

        // Clear color with alpha 0 (required for front-to-back rendering)
        mtkView.clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)

        // Set delegate for rendering callbacks
        mtkView.delegate = context.coordinator

        // On-demand rendering: only redraw when camera state changes to avoid 60fps idle drain.
        mtkView.enableSetNeedsDisplay = true
        mtkView.isPaused = true
        mtkView.preferredFramesPerSecond = 60

        // Add gesture recognizers for camera control
        let panGesture = UIPanGestureRecognizer(
            target: context.coordinator,
            action: #selector(Coordinator.handlePan(_:))
        )
        mtkView.addGestureRecognizer(panGesture)

        let pinchGesture = UIPinchGestureRecognizer(
            target: context.coordinator,
            action: #selector(Coordinator.handlePinch(_:))
        )
        mtkView.addGestureRecognizer(pinchGesture)

        // Initialize coordinator with Metal resources and start loading
        context.coordinator.setup(
            device: device,
            mtkView: mtkView,
            plyURL: plyURL
        )

        return mtkView
    }

    func updateUIView(_ uiView: MTKView, context: Context) {
        // Reload splat when SwiftUI reuses this view for a different PLY URL.
        if context.coordinator.currentURL != plyURL, let device = uiView.device {
            context.coordinator.setup(device: device, mtkView: uiView, plyURL: plyURL)
        }
        // Keep coordinator's zoom level in sync with external state (e.g. sliders).
        context.coordinator.zoomLevel = zoomLevel
    }

    func makeCoordinator() -> Coordinator {
        Coordinator(isLoading: $isLoading, loadError: $loadError, zoomLevel: $zoomLevel, onBoundsAvailable: onBoundsAvailable)
    }

    // MARK: - Coordinator

    /// Coordinator handles Metal rendering and gesture input
    class Coordinator: NSObject, MTKViewDelegate {

        // MARK: - Metal Resources

        private var device: MTLDevice?
        private var commandQueue: MTLCommandQueue?
        private weak var view: MTKView?

        // MetalSplatter renderer (protected by a small lock; setup runs on a Task, draw runs on MTKView's render loop).
        private let rendererLock = NSLock()
        private var _splatRenderer: SplatRenderer?
        private var splatRenderer: SplatRenderer? {
            get {
                rendererLock.lock()
                defer { rendererLock.unlock() }
                return _splatRenderer
            }
            set {
                rendererLock.lock()
                _splatRenderer = newValue
                rendererLock.unlock()
            }
        }

        /// Last loaded URL so `updateUIView` can trigger a reload on navigation changes.
        var currentURL: URL?

        // Semaphore for limiting concurrent renders
        private let inFlightSemaphore = DispatchSemaphore(value: maxSimultaneousRenders)

        // Current drawable size
        private var drawableSize: CGSize = .zero

        // MARK: - Camera State

        /// Camera yaw angle - initialized to 0 to face into the room
        private var cameraYaw: Float = 0

        /// Camera pitch angle (vertical rotation around X-axis)
        private var cameraPitch: Float = 0

        /// Camera position offset from centroid (for movement)
        private var cameraOffset: SIMD3<Float> = .zero

        /// XY scale applied in scene space (Furniture Fit / wall calibration via `WebGLScaleRoom` parity).
        private var sceneScale: SIMD2<Float> = SIMD2(1, 1)

        /// Zoom level binding from parent view
        @Binding var zoomLevel: Float

        private var didRegisterSharpRoomNotifications = false

        /// Callback back into SwiftUI for `RoomBounds` once MetalSplatter has loaded the PLY.
        private let onBoundsAvailable: ((RoomBounds) -> Void)?

        // MARK: - Camera Constants

        /// Maximum pitch angle (prevents camera flip) - ~80 degrees
        private let maxPitch: Float = 1.4

        /// Drag sensitivity - radians per point
        private let dragSensitivity: Float = 0.005

        /// Exterior camera mode distance (fallback when centroid unavailable)
        private let exteriorCameraZ: Float = -8.0

        /// Field of view in radians (~65 degrees)
        private let fovy: Float = 65.0 * .pi / 180.0

        // MARK: - Bindings

        @Binding var isLoading: Bool
        @Binding var loadError: String?

        // MARK: - Initialization

        init(isLoading: Binding<Bool>, loadError: Binding<String?>, zoomLevel: Binding<Float>, onBoundsAvailable: ((RoomBounds) -> Void)?) {
            self._isLoading = isLoading
            self._loadError = loadError
            self._zoomLevel = zoomLevel
            self.onBoundsAvailable = onBoundsAvailable
            super.init()
        }

        deinit {
            NotificationCenter.default.removeObserver(self)
        }

        // MARK: - Setup

        /// Initialize Metal resources and load PLY file
        func setup(device: MTLDevice, mtkView: MTKView, plyURL: URL) {
            self.device = device
            self.commandQueue = device.makeCommandQueue()
            self.drawableSize = mtkView.drawableSize
            self.view = mtkView
            self.currentURL = plyURL
            if !didRegisterSharpRoomNotifications {
                didRegisterSharpRoomNotifications = true
                registerSharpRoomParityNotifications()
            }

            logDebug("🎨 [GaussianSplatView] Starting to load PLY: \(plyURL.lastPathComponent)")

            // Load PLY asynchronously - local MetalSplatter uses async API
            Task { [weak self] in
                do {
                    // Initialize SplatRenderer with Metal configuration
                    let renderer = try await SplatRenderer(
                        device: device,
                        colorFormat: mtkView.colorPixelFormat,
                        depthFormat: mtkView.depthStencilPixelFormat,
                        sampleCount: mtkView.sampleCount,
                        maxViewCount: 1,
                        maxSimultaneousRenders: GaussianSplatView.maxSimultaneousRenders
                    )

                    // Load PLY file into renderer (async)
                    try await renderer.read(from: plyURL)

                    await MainActor.run { [weak self] in
                        guard let self = self else { return }

                        logDebug("✅ [GaussianSplatView] PLY loaded with \(renderer.splatCount) splats")

                        // Log centroid for interior camera positioning
                        if let centroid = renderer.centroid {
                            logDebug("📍 [GaussianSplatView] Splat centroid: \(centroid)")
                        }

                        if let bbox = renderer.boundingBox {
                            let bounds = RoomBounds(
                                minX: bbox.min.x, maxX: bbox.max.x,
                                minY: bbox.min.y, maxY: bbox.max.y,
                                minZ: bbox.min.z, maxZ: bbox.max.z
                            )
                            logDebug("📐 [GaussianSplatView] MetalSplatter bounds X[\(bounds.minX), \(bounds.maxX)] Y[\(bounds.minY), \(bounds.maxY)] Z[\(bounds.minZ), \(bounds.maxZ)]")
                            self.onBoundsAvailable?(bounds)
                        }

                        self.splatRenderer = renderer
                        self.isLoading = false
                        self.view?.setNeedsDisplay()
                    }
                } catch {
                    await MainActor.run { [weak self] in
                        logDebug("❌ [GaussianSplatView] PLY load failed: \(error)")
                        self?.loadError = "Failed to load PLY: \(error.localizedDescription)"
                        self?.isLoading = false
                    }
                }
            }
        }

        // MARK: - MTKViewDelegate

        func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
            drawableSize = size
        }

        func draw(in view: MTKView) {
            // Skip rendering if not ready
            guard let renderer = splatRenderer,
                  let commandQueue = commandQueue,
                  let drawable = view.currentDrawable else {
                return
            }

            // Wait for available render slot
            guard inFlightSemaphore.wait(timeout: .now() + 0.1) == .success else {
                return
            }

            guard let commandBuffer = commandQueue.makeCommandBuffer() else {
                inFlightSemaphore.signal()
                return
            }

            // Signal semaphore when command buffer completes
            let semaphore = inFlightSemaphore
            commandBuffer.addCompletedHandler { _ in
                semaphore.signal()
            }

            // Use the new render API - pass textures directly to commandBuffer
            do {
                try renderer.render(
                    viewports: [viewport],
                    colorTexture: view.multisampleColorTexture ?? drawable.texture,
                    colorStoreAction: view.multisampleColorTexture == nil ? .store : .multisampleResolve,
                    depthTexture: view.depthStencilTexture,
                    rasterizationRateMap: nil,
                    renderTargetArrayLength: 0,
                    to: commandBuffer
                )
            } catch {
                logDebug("❌ [GaussianSplatView] Render error: \(error)")
            }

            // Present drawable
            commandBuffer.present(drawable)
            commandBuffer.commit()
        }

        // MARK: - Viewport Calculation

        /// Calculate viewport descriptor for current frame
        private var viewport: SplatRenderer.ViewportDescriptor {
            let aspectRatio = Float(drawableSize.width / drawableSize.height)

            // Apply zoom by adjusting FOV (smaller FOV = zoom in, larger = zoom out)
            let zoomedFovy = fovy / zoomLevel
            let projectionMatrix = matrixPerspectiveRightHand(
                fovyRadians: zoomedFovy,
                aspectRatio: aspectRatio,
                nearZ: 0.01,
                farZ: 100.0
            )

            // User orbit rotation: Yaw (Y-axis) then Pitch (X-axis)
            let yawMatrix = matrix4x4Rotation(radians: cameraYaw, axis: SIMD3<Float>(0, 1, 0))
            let pitchMatrix = matrix4x4Rotation(radians: cameraPitch, axis: SIMD3<Float>(1, 0, 0))
            let userRotation = pitchMatrix * yawMatrix

            // Scene scale (e.g. wall calibration / Furniture Fit)
            let scaleMatrix = matrix4x4Scale(sceneScale.x, sceneScale.y, 1)

            // Camera position: use raw PLY bounds to place camera at FRONT WALL looking INTO the room.
            // Raw PLY: Z is negative; maxZ (least negative) ≈ front wall, minZ ≈ back wall.
            let cameraPos: SIMD3<Float>
            let lookTarget: SIMD3<Float>

            if let renderer = splatRenderer,
               let bbox = renderer.boundingBox,
               let centroid = renderer.centroid {
                let frontZ = bbox.max.z                  // e.g. -1.16 (front wall)
                let camZ = frontZ + 0.3                  // just in front of front wall (still negative)
                let camX = centroid.x                    // centered horizontally
                let camY = centroid.y + 0.2              // slight eye-level raise

                cameraPos = SIMD3<Float>(camX, camY, camZ) + cameraOffset
                lookTarget = centroid + cameraOffset
            } else if let centroid = splatRenderer?.centroid {
                // Fallback: position camera a bit in front of centroid along +Z looking back at centroid.
                cameraPos = SIMD3<Float>(centroid.x, centroid.y, centroid.z + 3) + cameraOffset
                lookTarget = centroid + cameraOffset
            } else {
                // Last resort: simple exterior camera looking at origin.
                cameraPos = SIMD3<Float>(0, 0, 3) + cameraOffset
                lookTarget = SIMD3<Float>(0, 0, 0) + cameraOffset
            }

            // Build look-at view matrix from eye/target/up.
            let viewBase = matrixLookAt(
                eye: cameraPos,
                target: lookTarget,
                up: SIMD3<Float>(0, 1, 0)
            )

            // Final view: user orbit around look-target, then apply scene scale.
            // We stay in raw PLY coordinates (no calibration flip).
            let viewMatrix = userRotation * viewBase * scaleMatrix

            // Create Metal viewport
            let mtlViewport = MTLViewport(
                originX: 0, originY: 0,
                width: drawableSize.width, height: drawableSize.height,
                znear: 0, zfar: 1
            )

            return SplatRenderer.ViewportDescriptor(
                viewport: mtlViewport,
                projectionMatrix: projectionMatrix,
                viewMatrix: viewMatrix,
                screenSize: SIMD2(x: Int(drawableSize.width), y: Int(drawableSize.height))
            )
        }

        // MARK: - Matrix Utilities

        /// Create perspective projection matrix (right-handed)
        private func matrixPerspectiveRightHand(fovyRadians fovy: Float, aspectRatio: Float, nearZ: Float, farZ: Float) -> simd_float4x4 {
            let ys = 1 / tanf(fovy * 0.5)
            let xs = ys / aspectRatio
            let zs = farZ / (nearZ - farZ)

            return simd_float4x4(columns: (
                SIMD4<Float>(xs, 0, 0, 0),
                SIMD4<Float>(0, ys, 0, 0),
                SIMD4<Float>(0, 0, zs, -1),
                SIMD4<Float>(0, 0, zs * nearZ, 0)
            ))
        }

        /// Create rotation matrix around arbitrary axis
        private func matrix4x4Rotation(radians: Float, axis: SIMD3<Float>) -> simd_float4x4 {
            let unitAxis = normalize(axis)
            let ct = cosf(radians)
            let st = sinf(radians)
            let ci = 1 - ct
            let x = unitAxis.x, y = unitAxis.y, z = unitAxis.z

            return simd_float4x4(columns: (
                SIMD4<Float>(ct + x * x * ci, y * x * ci + z * st, z * x * ci - y * st, 0),
                SIMD4<Float>(x * y * ci - z * st, ct + y * y * ci, z * y * ci + x * st, 0),
                SIMD4<Float>(x * z * ci + y * st, y * z * ci - x * st, ct + z * z * ci, 0),
                SIMD4<Float>(0, 0, 0, 1)
            ))
        }

        /// Look-at matrix (right-handed): positions camera at `eye` looking at `target` with `up` vector.
        private func matrixLookAt(eye: SIMD3<Float>, target: SIMD3<Float>, up: SIMD3<Float>) -> simd_float4x4 {
            let f = normalize(target - eye)           // Forward
            let r = normalize(cross(f, up))           // Right
            let u = cross(r, f)                       // True up

            return simd_float4x4(columns: (
                SIMD4<Float>(r.x, u.x, -f.x, 0),
                SIMD4<Float>(r.y, u.y, -f.y, 0),
                SIMD4<Float>(r.z, u.z, -f.z, 0),
                SIMD4<Float>(-dot(r, eye), -dot(u, eye), dot(f, eye), 1)
            ))
        }

        /// Create translation matrix
        private func matrix4x4Translation(_ x: Float, _ y: Float, _ z: Float) -> simd_float4x4 {
            return simd_float4x4(columns: (
                SIMD4<Float>(1, 0, 0, 0),
                SIMD4<Float>(0, 1, 0, 0),
                SIMD4<Float>(0, 0, 1, 0),
                SIMD4<Float>(x, y, z, 1)
            ))
        }

        private func matrix4x4Scale(_ x: Float, _ y: Float, _ z: Float) -> simd_float4x4 {
            simd_float4x4(columns: (
                SIMD4<Float>(x, 0, 0, 0),
                SIMD4<Float>(0, y, 0, 0),
                SIMD4<Float>(0, 0, z, 0),
                SIMD4<Float>(0, 0, 0, 1)
            ))
        }

        // MARK: - SharpRoomView notification parity (same names as WebGL viewer)

        private func registerSharpRoomParityNotifications() {
            let center = NotificationCenter.default
            center.addObserver(self, selector: #selector(onRecenter), name: NSNotification.Name("RecenterWebGLCamera"), object: nil)
            center.addObserver(self, selector: #selector(onJoystickMove(_:)), name: NSNotification.Name("WebGLJoystickMove"), object: nil)
            center.addObserver(self, selector: #selector(onScaleRoom(_:)), name: NSNotification.Name("WebGLScaleRoom"), object: nil)
            center.addObserver(self, selector: #selector(onCameraMoveUp), name: NSNotification.Name("WebGLCameraMoveUp"), object: nil)
            center.addObserver(self, selector: #selector(onCameraMoveDown), name: NSNotification.Name("WebGLCameraMoveDown"), object: nil)
            center.addObserver(self, selector: #selector(onCameraMoveLeft), name: NSNotification.Name("WebGLCameraMoveLeft"), object: nil)
            center.addObserver(self, selector: #selector(onCameraMoveRight), name: NSNotification.Name("WebGLCameraMoveRight"), object: nil)
        }

        @objc private func onRecenter() {
            DispatchQueue.main.async { [weak self] in
                guard let self else { return }
                self.cameraYaw = 0
                self.cameraPitch = 0
                self.cameraOffset = .zero
                self.view?.setNeedsDisplay()
            }
        }

        @objc private func onJoystickMove(_ notification: Notification) {
            guard let ui = notification.userInfo, let offset = ui["offset"] as? CGSize else { return }
            let dx = Float(offset.width) * 0.012
            let dy = Float(-offset.height) * 0.012
            DispatchQueue.main.async { [weak self] in
                guard let self else { return }
                self.cameraOffset.x += dx
                self.cameraOffset.z += dy
                self.view?.setNeedsDisplay()
            }
        }

        @objc private func onScaleRoom(_ notification: Notification) {
            guard let userInfo = notification.userInfo else { return }
            let scaleX: Double? = (userInfo["scaleX"] as? NSNumber)?.doubleValue ?? userInfo["scaleX"] as? Double
            let scaleY: Double? = (userInfo["scaleY"] as? NSNumber)?.doubleValue ?? userInfo["scaleY"] as? Double
            if let sx = scaleX, let sy = scaleY {
                DispatchQueue.main.async { [weak self] in
                    guard let self else { return }
                    self.sceneScale.x *= Float(sx)
                    self.sceneScale.y *= Float(sy)
                    self.view?.setNeedsDisplay()
                }
                return
            }
            let factor: Double?
            if let d = userInfo["factor"] as? Double {
                factor = d
            } else if let n = userInfo["factor"] as? NSNumber {
                factor = n.doubleValue
            } else {
                factor = nil
            }
            guard let f = factor else { return }
            DispatchQueue.main.async { [weak self] in
                guard let self else { return }
                let ff = Float(f)
                self.sceneScale.x *= ff
                self.sceneScale.y *= ff
                self.view?.setNeedsDisplay()
            }
        }

        @objc private func onCameraMoveUp() {
            DispatchQueue.main.async { [weak self] in
                guard let self else { return }
                self.cameraOffset.y += 0.15
                self.view?.setNeedsDisplay()
            }
        }

        @objc private func onCameraMoveDown() {
            DispatchQueue.main.async { [weak self] in
                guard let self else { return }
                self.cameraOffset.y -= 0.15
                self.view?.setNeedsDisplay()
            }
        }

        @objc private func onCameraMoveLeft() {
            DispatchQueue.main.async { [weak self] in
                guard let self else { return }
                self.cameraOffset.x -= 0.2
                self.view?.setNeedsDisplay()
            }
        }

        @objc private func onCameraMoveRight() {
            DispatchQueue.main.async { [weak self] in
                guard let self else { return }
                self.cameraOffset.x += 0.2
                self.view?.setNeedsDisplay()
            }
        }

        // MARK: - Gesture Handlers

        /// Handle pan gesture for orbital scene rotation
        @objc func handlePan(_ gesture: UIPanGestureRecognizer) {
            let translation = gesture.translation(in: gesture.view)

            // Horizontal drag rotates scene around Y-axis (reversed for orbital control)
            cameraYaw -= Float(translation.x) * dragSensitivity

            // Vertical drag rotates scene around X-axis, clamped to prevent flip
            cameraPitch -= Float(translation.y) * dragSensitivity
            cameraPitch = max(-maxPitch, min(maxPitch, cameraPitch))

            gesture.setTranslation(.zero, in: gesture.view)
            view?.setNeedsDisplay()
        }

        /// Handle pinch gesture for zoom
        @objc func handlePinch(_ gesture: UIPinchGestureRecognizer) {
            // Adjust zoom level via binding (syncs with slider)
            var newZoom = zoomLevel * Float(gesture.scale)

            // Clamp zoom range to match slider bounds
            newZoom = max(0.5, min(3.0, newZoom))
            zoomLevel = newZoom

            gesture.scale = 1.0
            view?.setNeedsDisplay()
        }
    }
}

// MARK: - Preview

#Preview {
    GaussianSplatView(
        plyURL: URL(fileURLWithPath: "/tmp/test.ply"),
        isLoading: .constant(true),
        loadError: .constant(nil),
        zoomLevel: .constant(1.0),
        onBoundsAvailable: nil
    )
}
