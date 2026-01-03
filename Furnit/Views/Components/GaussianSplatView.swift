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

        // Enable continuous rendering
        mtkView.enableSetNeedsDisplay = false
        mtkView.isPaused = false
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
        // Handle URL changes if needed
    }

    func makeCoordinator() -> Coordinator {
        Coordinator(isLoading: $isLoading, loadError: $loadError, zoomLevel: $zoomLevel)
    }

    // MARK: - Coordinator

    /// Coordinator handles Metal rendering and gesture input
    class Coordinator: NSObject, MTKViewDelegate {

        // MARK: - Metal Resources

        private var device: MTLDevice?
        private var commandQueue: MTLCommandQueue?

        // MetalSplatter renderer
        private var splatRenderer: SplatRenderer?

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

        /// Zoom level binding from parent view
        @Binding var zoomLevel: Float

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

        init(isLoading: Binding<Bool>, loadError: Binding<String?>, zoomLevel: Binding<Float>) {
            self._isLoading = isLoading
            self._loadError = loadError
            self._zoomLevel = zoomLevel
            super.init()
        }

        // MARK: - Setup

        /// Initialize Metal resources and load PLY file
        func setup(device: MTLDevice, mtkView: MTKView, plyURL: URL) {
            self.device = device
            self.commandQueue = device.makeCommandQueue()
            self.drawableSize = mtkView.drawableSize

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

                    await MainActor.run {
                        guard let self = self else { return }

                        logDebug("✅ [GaussianSplatView] PLY loaded with \(renderer.splatCount) splats")

                        // Log centroid for interior camera positioning
                        if let centroid = renderer.centroid {
                            logDebug("📍 [GaussianSplatView] Splat centroid: \(centroid)")
                        }

                        self.splatRenderer = renderer
                        self.isLoading = false
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
            _ = inFlightSemaphore.wait(timeout: .distantFuture)

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
                nearZ: 0.1,
                farZ: 100.0
            )

            // Rotation matrices: Yaw (Y-axis) then Pitch (X-axis)
            let yawMatrix = matrix4x4Rotation(radians: cameraYaw, axis: SIMD3<Float>(0, 1, 0))
            let pitchMatrix = matrix4x4Rotation(radians: cameraPitch, axis: SIMD3<Float>(1, 0, 0))
            let rotationMatrix = pitchMatrix * yawMatrix

            // Interior mode: use centroid to position camera inside the room
            let translationMatrix: simd_float4x4
            if let centroid = splatRenderer?.centroid {
                // Interior mode: camera at splat centroid, offset down for eye level
                let eyeLevelOffset = SIMD3<Float>(0, -0.3, 0)  // Lower camera for better view
                let cameraPos = centroid + cameraOffset + eyeLevelOffset
                translationMatrix = matrix4x4Translation(-cameraPos.x, -cameraPos.y, -cameraPos.z)
            } else {
                // Exterior mode: camera outside looking at origin
                translationMatrix = matrix4x4Translation(0.0, 0.0, exteriorCameraZ * zoomLevel)
            }

            // Calibration flip - turns model rightside-up (common for 3DGS PLY files)
            let calibrationMatrix = matrix4x4Rotation(radians: .pi, axis: SIMD3<Float>(0, 0, 1))

            // Orbital scene rotation: calibration first, then rotation (rotates scene around camera)
            let viewMatrix = translationMatrix * calibrationMatrix * rotationMatrix

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

        /// Create translation matrix
        private func matrix4x4Translation(_ x: Float, _ y: Float, _ z: Float) -> simd_float4x4 {
            return simd_float4x4(columns: (
                SIMD4<Float>(1, 0, 0, 0),
                SIMD4<Float>(0, 1, 0, 0),
                SIMD4<Float>(0, 0, 1, 0),
                SIMD4<Float>(x, y, z, 1)
            ))
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
        }

        /// Handle pinch gesture for zoom
        @objc func handlePinch(_ gesture: UIPinchGestureRecognizer) {
            // Adjust zoom level via binding (syncs with slider)
            var newZoom = zoomLevel / Float(gesture.scale)

            // Clamp zoom range to match slider bounds
            newZoom = max(0.5, min(3.0, newZoom))
            zoomLevel = newZoom

            gesture.scale = 1.0
        }
    }
}

// MARK: - Preview

#Preview {
    GaussianSplatView(
        plyURL: URL(fileURLWithPath: "/tmp/test.ply"),
        isLoading: .constant(true),
        loadError: .constant(nil),
        zoomLevel: .constant(1.0)
    )
}
