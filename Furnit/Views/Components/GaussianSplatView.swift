import SwiftUI
import MetalKit
import MetalSplatter
import SplatIO
import simd

private enum GaussianSplatLoadError: LocalizedError {
    case emptyScene

    var errorDescription: String? {
        switch self {
        case .emptyScene: return "The splat file contains no points."
        }
    }
}

// MARK: - GaussianSplatView

/// A SwiftUI view that renders a Gaussian Splat scene from a .ply file using MetalSplatter.
///
/// SHARP / SharpRoom: Metal loads `_classic.ply` (uchar RGB) when present. Camera: trimmed P3–P97 bounds, ``RoomBounds/defaultSplatCameraEyeAndTarget``; `_classic` uses view scale `(1,-1,-1)` and default yaw π so the opening view faces into the room.
struct GaussianSplatView: UIViewRepresentable {

    // MARK: Public interface

    /// URL of the .ply file to load and render.
    let plyURL: URL

    /// Binding updated on the main thread; true while the PLY is being read.
    @Binding var isLoading: Bool

    /// Binding updated on the main thread; non-nil when loading fails.
    @Binding var loadError: String?

    /// Zoom multiplier – clamped to 0.5 … 3.0 by the pinch gesture handler (or 0.1 … 50 when `infiniteZoom` is true).
    @Binding var zoomLevel: Float

    /// When true, use dolly-style pinch zoom with a much wider range and smaller near plane (matches the Room Viewer “Infinite Zoom” setting).
    let infiniteZoom: Bool

    /// Called once after a successful load with bounds derived from MetalSplatter’s AABB.
    var onBoundsAvailable: ((RoomBounds) -> Void)?

    // MARK: Concurrency guard

    /// Maximum number of frames that may be encoded simultaneously.
    nonisolated static let maxSimultaneousRenders = 3

    // MARK: UIViewRepresentable

    func makeCoordinator() -> Coordinator {
        Coordinator(
            isLoading: $isLoading,
            loadError: $loadError,
            zoomLevel: $zoomLevel,
            infiniteZoom: infiniteZoom,
            onBoundsAvailable: onBoundsAvailable
        )
    }

    func makeUIView(context: Context) -> MTKView {
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("[GaussianSplatView] No Metal device available on this hardware.")
        }

        let mtkView = MTKView(frame: .zero, device: device)

        // Pixel formats
        mtkView.colorPixelFormat          = .bgra8Unorm_srgb
        mtkView.depthStencilPixelFormat   = .depth32Float
        mtkView.sampleCount               = 1

        // Alpha 0 is required so that the compositor can blend splats over the
        // gray background using front-to-back order (pre-multiplied alpha).
        mtkView.clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)

        // On-demand rendering – only redraw when explicitly requested.
        mtkView.enableSetNeedsDisplay    = true
        mtkView.isPaused                  = true
        mtkView.preferredFramesPerSecond  = 60

        // Gesture recognisers
        let pan   = UIPanGestureRecognizer(target: context.coordinator,
                                           action: #selector(Coordinator.handlePan(_:)))
        let pinch = UIPinchGestureRecognizer(target: context.coordinator,
                                              action: #selector(Coordinator.handlePinch(_:)))
        mtkView.addGestureRecognizer(pan)
        mtkView.addGestureRecognizer(pinch)

        mtkView.delegate = context.coordinator
        context.coordinator.setup(view: mtkView, plyURL: plyURL)

        return mtkView
    }

    func updateUIView(_ mtkView: MTKView, context: Context) {
        let coordinator = context.coordinator

        // Reload when the URL changes
        if coordinator.currentURL != plyURL {
            coordinator.setup(view: mtkView, plyURL: plyURL)
        }

        // Sync zoom for rendering only — do **not** assign `coordinator.zoomLevel` (a `@Binding`)
        // here; that writes the parent `@State` during `updateUIView` and triggers
        // “Modifying state during view update”.
        coordinator.appliedZoomLevel = zoomLevel
        mtkView.setNeedsDisplay()
    }

    // MARK: - Coordinator

    final class Coordinator: NSObject, MTKViewDelegate {

        // ── Bindings ──────────────────────────────────────────────────────────
        @Binding var isLoading: Bool
        @Binding var loadError: String?
        @Binding var zoomLevel: Float
        /// Zoom used for projection (updated from parent in `updateUIView` without writing the binding).
        var appliedZoomLevel: Float = 1.0
        var onBoundsAvailable: ((RoomBounds) -> Void)?

        // ── Metal objects ─────────────────────────────────────────────────────
        var device: MTLDevice?
        var commandQueue: MTLCommandQueue?
        weak var view: MTKView?

        /// Pipeline that composites the splat scratch texture over a solid gray background.
        var compositePipeline: MTLRenderPipelineState?

        /// Intermediate texture – splats are rendered here first so we can composite over gray.
        var splatScratchTexture: MTLTexture?
        private var scratchWidth:  Int = 0
        private var scratchHeight: Int = 0

        // ── SplatRenderer (thread-safe) ───────────────────────────────────────
        private let rendererLock  = NSLock()
        private var _splatRenderer: SplatRenderer?
        var splatRenderer: SplatRenderer? {
            get { rendererLock.lock(); defer { rendererLock.unlock() }; return _splatRenderer }
            set { rendererLock.lock(); defer { rendererLock.unlock() }; _splatRenderer = newValue }
        }

        // ── State ─────────────────────────────────────────────────────────────
        var currentURL: URL?
        private let inFlightSemaphore = DispatchSemaphore(value: GaussianSplatView.maxSimultaneousRenders)
        var drawableSize: CGSize = .zero
        private var warmupEndTime: CFAbsoluteTime?

        /// Axis-aligned bounds from loaded `SplatPoint` positions (MetalSplatter 1.x `SplatRenderer` does not expose `boundingBox` / `centroid`).
        private var sceneBoundsMin: SIMD3<Float>?
        private var sceneBoundsMax: SIMD3<Float>?
        private var sceneCentroid: SIMD3<Float>?
        /// True when viewing a SHARP `_classic.ply` splat; used to undo the extra (y,z) → (-y,-z) flip applied at export so the room is not upside down.
        private var isSharpClassicPly: Bool = false

        // Camera orbit & pan state
        var cameraYaw:    Float = 0
        var cameraPitch:  Float = 0
        var cameraOffset: SIMD3<Float> = .zero
        var sceneScale:   SIMD2<Float> = SIMD2(1, 1)

        private var didRegisterSharpRoomNotifications = false

        // ── Camera constants ──────────────────────────────────────────────────
        private let maxPitch:        Float = 1.4
        private let dragSensitivity: Float = 0.005
        private let exteriorCameraZ: Float = -8.0
        private let fovy:            Float = 65 * (.pi / 180)   // 65° → radians

        /// Whether to use wide-range dolly-style pinch zoom and a smaller near plane (matches the “Infinite Zoom” setting).
        let infiniteZoom: Bool

        /// Linear RGB before S-curve composite (`BrightnessAdjust.metal`).
        var splatCompositeExposure: Float = 1.0
        /// Additive lift on dark tonemapped samples (smoothstep mask).
        var splatCompositeShadowLift: Float = 0.0

        // ── Init ──────────────────────────────────────────────────────────────
        init(
            isLoading: Binding<Bool>,
            loadError: Binding<String?>,
            zoomLevel:  Binding<Float>,
            infiniteZoom: Bool,
            onBoundsAvailable: ((RoomBounds) -> Void)?
        ) {
            _isLoading = isLoading
            _loadError = loadError
            _zoomLevel = zoomLevel
            self.infiniteZoom = infiniteZoom
            self.onBoundsAvailable = onBoundsAvailable
            super.init()
        }

        // MARK: Setup

        /// Defers work until after SwiftUI finishes the current update transaction (must run UIKit / bindings on the main actor).
        private func scheduleSwiftUIBindingUpdates(_ updates: @escaping @MainActor () -> Void) {
            Task { @MainActor in
                await Task.yield()
                updates()
            }
        }

        func setup(view mtkView: MTKView, plyURL: URL) {
            // ── Reset camera ─────────────────────────────────────────────────
            cameraPitch  = 0
            cameraOffset = .zero
            sceneScale   = SIMD2(1, 1)
            appliedZoomLevel = zoomLevel

            // ── Metal basics ─────────────────────────────────────────────────
            guard let device = mtkView.device ?? MTLCreateSystemDefaultDevice() else {
                print("❌ [GaussianSplatView] No Metal device")
                return
            }
            self.device       = device
            commandQueue      = device.makeCommandQueue()
            drawableSize      = mtkView.drawableSize
            self.view         = mtkView
            currentURL        = plyURL
            splatCompositeExposure = 1.0
            splatCompositeShadowLift = 0.0
            isSharpClassicPly = plyURL.lastPathComponent.contains("_classic")
            // Classic frame + `defaultSplatCameraEyeAndTarget` disagree on which Z is “into” the room — orbit 180° so you face the scene, not the wall behind you.
            cameraYaw = isSharpClassicPly ? Float.pi : 0

            // ── Composite pipeline ────────────────────────────────────────────
            if let library = device.makeDefaultLibrary(),
               let vertFn  = library.makeFunction(name: "compositeOverGrayVertex"),
               let fragFn  = library.makeFunction(name: "compositeOverGrayFragment") {
                let desc = MTLRenderPipelineDescriptor()
                desc.vertexFunction   = vertFn
                desc.fragmentFunction = fragFn
                desc.colorAttachments[0].pixelFormat = mtkView.colorPixelFormat

                // Standard over-compositing blend
                desc.colorAttachments[0].isBlendingEnabled             = true
                desc.colorAttachments[0].rgbBlendOperation             = .add
                desc.colorAttachments[0].alphaBlendOperation           = .add
                desc.colorAttachments[0].sourceRGBBlendFactor          = .sourceAlpha
                desc.colorAttachments[0].destinationRGBBlendFactor     = .oneMinusSourceAlpha
                desc.colorAttachments[0].sourceAlphaBlendFactor        = .one
                desc.colorAttachments[0].destinationAlphaBlendFactor   = .oneMinusSourceAlpha

                compositePipeline = try? device.makeRenderPipelineState(descriptor: desc)
            }
            logDebug("🔧 [GaussianSplatView] composite pipeline=\(compositePipeline != nil)")

            // ── Notifications ─────────────────────────────────────────────────
            if !didRegisterSharpRoomNotifications {
                registerSharpRoomParityNotifications()
                didRegisterSharpRoomNotifications = true
            }

            // ── Load PLY ──────────────────────────────────────────────────────
            logDebug("🎨 [GaussianSplatView] Starting to load PLY: \(plyURL.lastPathComponent)")

            // Capture view properties before async hop
            let clearColor   = mtkView.clearColor
            let colorFormat  = mtkView.colorPixelFormat
            let depthFormat  = mtkView.depthStencilPixelFormat
            let sampleCount  = mtkView.sampleCount

            splatRenderer = nil
            sceneBoundsMin = nil
            sceneBoundsMax = nil
            sceneCentroid = nil

            scheduleSwiftUIBindingUpdates {
                self.isLoading = true
                self.loadError = nil
                Task { [weak self] in
                    guard let self else { return }
                    do {
                    let points = try await AutodetectSceneReader(plyURL).readAll()
                    guard !points.isEmpty else { throw GaussianSplatLoadError.emptyScene }

                    // Trimmed P3–P97 per axis: camera framing. Full AABB: `onBoundsAvailable` (wall measurement).
                    let splatCount = points.count
                    let sortedX = points.map(\.position.x).sorted()
                    let sortedY = points.map(\.position.y).sorted()
                    let sortedZ = points.map(\.position.z).sorted()
                    let lo = min(splatCount - 1, max(0, Int(Float(splatCount) * 0.03)))
                    let hi = min(splatCount - 1, max(lo, Int(Float(splatCount) * 0.97)))
                    let trimmedMin = SIMD3<Float>(sortedX[lo], sortedY[lo], sortedZ[lo])
                    let trimmedMax = SIMD3<Float>(sortedX[hi], sortedY[hi], sortedZ[hi])

                    var fullMin = points[0].position
                    var fullMax = points[0].position
                    var positionSum = SIMD3<Float>.zero
                    for point in points {
                        let position = point.position
                        fullMin = simd_min(fullMin, position)
                        fullMax = simd_max(fullMax, position)
                        positionSum += position
                    }
                    let centroid = positionSum / Float(points.count)

                    #if DEBUG
                    logSphericalHarmonicsSanityCheck(points: points, plyFileName: plyURL.lastPathComponent)
                    #endif

                    let renderer = try SplatRenderer(
                        device: device,
                        colorFormat: colorFormat,
                        depthFormat: depthFormat,
                        sampleCount: sampleCount,
                        maxViewCount: 1,
                        maxSimultaneousRenders: GaussianSplatView.maxSimultaneousRenders,
                        highQualityDepth: true,
                        clearColor: clearColor
                    )
                    let chunk = try SplatChunk(device: device, from: points)
                    _ = await renderer.addChunk(chunk)

                    let bounds = RoomBounds(
                        minX: fullMin.x, maxX: fullMax.x,
                        minY: fullMin.y, maxY: fullMax.y,
                        minZ: fullMin.z, maxZ: fullMax.z
                    )
                    let loadedSplatCount = renderer.splatCount
                    self.scheduleSwiftUIBindingUpdates { [weak self] in
                        guard let self else { return }
                        logDebug("📐 [GaussianSplatView] Full bounds: X[\(fullMin.x),\(fullMax.x)] Y[\(fullMin.y),\(fullMax.y)] Z[\(fullMin.z),\(fullMax.z)]")
                        logDebug("📐 [GaussianSplatView] Trimmed P3–P97: X[\(trimmedMin.x),\(trimmedMax.x)] Y[\(trimmedMin.y),\(trimmedMax.y)] Z[\(trimmedMin.z),\(trimmedMax.z)]")
                        logDebug("✅ [GaussianSplatView] Loaded \(loadedSplatCount) splats centroid=\(centroid) (mean of all points; framing AABB trimmed P3–P97)")
                        self.onBoundsAvailable?(bounds)
                        self.sceneBoundsMin = trimmedMin
                        self.sceneBoundsMax = trimmedMax
                        self.sceneCentroid = centroid
                        self.splatRenderer = renderer
                        self.isLoading = false
                        mtkView.setNeedsDisplay()
                    }
                } catch {
                    self.scheduleSwiftUIBindingUpdates { [weak self] in
                        guard let self else { return }
                        logDebug("❌ [GaussianSplatView] Failed to load PLY: \(error)")
                        self.loadError = error.localizedDescription
                        self.isLoading = false
                    }
                }
                }
            }
        }

        // MARK: MTKViewDelegate

        func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
            drawableSize = size
            splatScratchTexture = nil
            scratchWidth = 0
            scratchHeight = 0
        }

        func draw(in view: MTKView) {
            guard let renderer = splatRenderer,
                  let commandQueue = commandQueue,
                  let drawable = view.currentDrawable,
                  let metalDevice = device
            else { return }

            let timeout = DispatchTime.now() + 0.1
            guard inFlightSemaphore.wait(timeout: timeout) == .success else { return }

            guard let commandBuffer = commandQueue.makeCommandBuffer() else {
                inFlightSemaphore.signal()
                return
            }
            commandBuffer.addCompletedHandler { [weak self] _ in
                self?.inFlightSemaphore.signal()
            }

            let vp = viewport

            if let pipeline = compositePipeline,
               let scratch = ensureSplatScratchTexture(
                   device: metalDevice,
                   drawable: drawable,
                   pixelFormat: view.colorPixelFormat
               ),
               let depthScratch = makeDepthTexture(device: metalDevice, width: scratch.width, height: scratch.height) {
                do {
                    try renderer.render(
                        viewports: [vp],
                        colorTexture: scratch,
                        colorStoreAction: .store,
                        depthTexture: depthScratch,
                        rasterizationRateMap: nil,
                        renderTargetArrayLength: 0,
                        to: commandBuffer
                    )
                } catch {
                    logDebug("❌ [GaussianSplatView] Render error: \(error)")
                }

                let compPassDesc = MTLRenderPassDescriptor()
                compPassDesc.colorAttachments[0].texture = drawable.texture
                compPassDesc.colorAttachments[0].loadAction = .dontCare
                compPassDesc.colorAttachments[0].storeAction = .store
                if let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: compPassDesc) {
                    encoder.setRenderPipelineState(pipeline)
                    encoder.setFragmentTexture(scratch, index: 0)
                    var compositeParams: [Float] = [splatCompositeExposure, splatCompositeShadowLift]
                    encoder.setFragmentBytes(&compositeParams, length: MemoryLayout<Float>.stride * 2, index: 0)
                    encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
                    encoder.endEncoding()
                }
            } else {
                do {
                    try renderer.render(
                        viewports: [vp],
                        colorTexture: drawable.texture,
                        colorStoreAction: .store,
                        depthTexture: view.depthStencilTexture,
                        rasterizationRateMap: nil,
                        renderTargetArrayLength: 0,
                        to: commandBuffer
                    )
                } catch {
                    logDebug("❌ [GaussianSplatView] Render error: \(error)")
                }
            }

            commandBuffer.present(drawable)
            commandBuffer.commit()
            // During warm-up, keep requesting new frames so the room sharpens quickly even when idle.
            if let warmupEndTime, CFAbsoluteTimeGetCurrent() < warmupEndTime {
                view.setNeedsDisplay()
            } else {
                warmupEndTime = nil
            }
        }

        // MARK: Viewport — camera & MetalSplatter matrix
        //
        // `_classic.ply`: view scale (1,-1,-1) aligns axes with `_3dgs`; default yaw π faces into the room vs back-wall camera math.

        private var viewport: SplatRenderer.ViewportDescriptor {
            let aspectRatio = Float(drawableSize.width / max(drawableSize.height, 1))
            let zoomedFovy  = infiniteZoom ? fovy : (fovy / appliedZoomLevel)

            let nearZ: Float = infiniteZoom ? 0.001 : 0.01
            let farZ: Float = infiniteZoom ? 250.0 : 100.0

            let projectionMatrix = matrixPerspectiveRightHand(
                fovyRadians:  zoomedFovy,
                aspectRatio:  aspectRatio,
                nearZ:        nearZ,
                farZ:         farZ
            )

            // SHARP `_classic.ply` stores positions in a frame rotated by 180° around Y and Z relative to `_3dgs` / base:
            // Pc = (x,  y,  z) = (x_a, -y_a, -z_a). To bring classic back into the canonical frame that `RoomBounds` / camera expects,
            // apply the same (1,-1,-1) again in view space when loading `_classic`.
            let flipY: Float = isSharpClassicPly ? -1 : 1
            let flipZ: Float = isSharpClassicPly ? -1 : 1
            let scaleMatrix = matrix4x4Scale(sceneScale.x, sceneScale.y * flipY, flipZ)

            let boundsMin = sceneBoundsMin
            let boundsMax = sceneBoundsMax
            let centroid = sceneCentroid

            // ── Camera: inside room, back → front wall (``RoomBounds`` / Android parity) ──
            var camPos: SIMD3<Float>
            var lookAt: SIMD3<Float>

            if let boundsMin, let boundsMax {
                let rb = RoomBounds(
                    minX: boundsMin.x, maxX: boundsMax.x,
                    minY: boundsMin.y, maxY: boundsMax.y,
                    minZ: boundsMin.z, maxZ: boundsMax.z
                )
                let (eye, target) = rb.defaultSplatCameraEyeAndTarget(cameraPadding: 0.3)
                camPos = eye + cameraOffset
                lookAt = target + cameraOffset
            } else if let centroid {
                camPos = SIMD3<Float>(centroid.x, centroid.y, centroid.z + 3) + cameraOffset
                lookAt = centroid + cameraOffset
            } else {
                camPos = SIMD3<Float>(0, 0, 3) + cameraOffset
                lookAt = SIMD3<Float>(0, 0, 0) + cameraOffset
            }

            // Infinite Zoom: convert pinch scaling into dolly along the view ray instead of narrowing the field of view.
            if infiniteZoom {
                let baseDir = camPos - lookAt
                let baseDistance = simd_length(baseDir)
                if baseDistance > 0.0001 {
                    let minZoom: Float = 0.1
                    let maxZoom: Float = 50.0
                    let clampedZoom = min(max(appliedZoomLevel, minZoom), maxZoom)
                    let distanceScale = 1.0 / clampedZoom
                    let newDistance = max(0.02, baseDistance * distanceScale)
                    let dirNorm = baseDir / baseDistance
                    camPos = lookAt + dirNorm * newDistance
                }
            }

            // ── Apply user orbit ─────────────────────────────────────────────
            let yawMatrix   = matrix4x4Rotation(radians: cameraYaw,
                                                 axis: SIMD3<Float>(0, 1, 0))
            let pitchMatrix = matrix4x4Rotation(radians: cameraPitch,
                                                 axis: SIMD3<Float>(1, 0, 0))
            let userRotation = pitchMatrix * yawMatrix
            let offset4      = userRotation * SIMD4<Float>(camPos - lookAt, 0)
            let rotatedEye   = lookAt + SIMD3<Float>(offset4.x, offset4.y, offset4.z)

            let viewBase = matrixLookAt(
                eye:    rotatedEye,
                target: lookAt,
                up:     SIMD3<Float>(0, 1, 0)
            )

            let viewMatrix = viewBase * scaleMatrix

            let mtlViewport = MTLViewport(
                originX: 0, originY: 0,
                width:   drawableSize.width,
                height:  drawableSize.height,
                znear:   0, zfar: 1
            )

            return SplatRenderer.ViewportDescriptor(
                viewport:         mtlViewport,
                projectionMatrix: projectionMatrix,
                viewMatrix:       viewMatrix,
                screenSize:       SIMD2(x: Int(drawableSize.width),
                                        y: Int(drawableSize.height))
            )
        }

        // MARK: Gesture Handlers

        @objc func handlePan(_ gesture: UIPanGestureRecognizer) {
            let translation = gesture.translation(in: gesture.view)
            cameraYaw   -= Float(translation.x) * dragSensitivity
            cameraPitch -= Float(translation.y) * dragSensitivity
            cameraPitch  = min(max(cameraPitch, -maxPitch), maxPitch)
            gesture.setTranslation(.zero, in: gesture.view)
            view?.setNeedsDisplay()
        }

        @objc func handlePinch(_ gesture: UIPinchGestureRecognizer) {
            let newZoom = appliedZoomLevel * Float(gesture.scale)
            let minZoom: Float = infiniteZoom ? 0.1 : 0.5
            let maxZoom: Float = infiniteZoom ? 50.0 : 3.0
            let clamped = min(max(newZoom, minZoom), maxZoom)
            appliedZoomLevel = clamped
            zoomLevel = clamped
            gesture.scale = 1.0
            view?.setNeedsDisplay()
        }

        // MARK: Scratch Texture

        /// Returns a cached MTLTexture suitable for the splat render pass.
        /// Re-creates the texture only when the drawable dimensions change.
        func ensureSplatScratchTexture(
            device:      MTLDevice,
            drawable:    CAMetalDrawable,
            pixelFormat: MTLPixelFormat
        ) -> MTLTexture? {
            let w = drawable.texture.width
            let h = drawable.texture.height
            if splatScratchTexture != nil, scratchWidth == w, scratchHeight == h {
                return splatScratchTexture
            }
            let desc = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: pixelFormat,
                width:       w,
                height:      h,
                mipmapped:   false
            )
            desc.usage        = [.renderTarget, .shaderRead]
            desc.storageMode  = .private
            splatScratchTexture = device.makeTexture(descriptor: desc)
            scratchWidth        = w
            scratchHeight       = h
            return splatScratchTexture
        }

        // MARK: Depth Texture Helper

        private func makeDepthTexture(device: MTLDevice, width: Int, height: Int) -> MTLTexture? {
            let desc = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: .depth32Float,
                width:       width,
                height:      height,
                mipmapped:   false
            )
            desc.usage       = .renderTarget
            desc.storageMode = .private
            return device.makeTexture(descriptor: desc)
        }

        // MARK: Notification Registration

        func registerSharpRoomParityNotifications() {
            let nc = NotificationCenter.default

            // Recenter: restore default camera orientation
            nc.addObserver(
                forName: NSNotification.Name("RecenterWebGLCamera"),
                object:  nil,
                queue:   nil
            ) { [weak self] _ in
                DispatchQueue.main.async { [weak self] in
                    guard let self else { return }
                    self.cameraYaw    = self.isSharpClassicPly ? Float.pi : 0
                    self.cameraPitch  = 0
                    self.cameraOffset = .zero
                    self.appliedZoomLevel = 1.0
                    self.zoomLevel = 1.0
                    self.view?.setNeedsDisplay()
                }
            }

            // Joystick pan: forward/back and left/right strafe
            nc.addObserver(
                forName: NSNotification.Name("WebGLJoystickMove"),
                object:  nil,
                queue:   nil
            ) { [weak self] note in
                DispatchQueue.main.async { [weak self] in
                    guard let self,
                          let offset = note.userInfo?["offset"] as? CGSize else { return }
                    let dx = Float(offset.width) * 0.012
                    let dy = Float(-offset.height) * 0.012
                    self.cameraOffset.x += dx
                    self.cameraOffset.z += dy
                    self.view?.setNeedsDisplay()
                }
            }

            // Scale: supports both (scaleX, scaleY) dict and single (factor) value
            nc.addObserver(
                forName: NSNotification.Name("WebGLScaleRoom"),
                object:  nil,
                queue:   nil
            ) { [weak self] note in
                DispatchQueue.main.async { [weak self] in
                    guard let self else { return }
                    if let sx = note.userInfo?["scaleX"] as? NSNumber,
                       let sy = note.userInfo?["scaleY"] as? NSNumber {
                        self.sceneScale.x *= Float(truncating: sx)
                        self.sceneScale.y *= Float(truncating: sy)
                    } else if let sx = note.userInfo?["scaleX"] as? Double,
                              let sy = note.userInfo?["scaleY"] as? Double {
                        self.sceneScale.x *= Float(sx)
                        self.sceneScale.y *= Float(sy)
                    } else if let f = note.userInfo?["factor"] as? NSNumber {
                        let ff = Float(truncating: f)
                        self.sceneScale.x *= ff
                        self.sceneScale.y *= ff
                    } else if let f = note.userInfo?["factor"] as? Double {
                        let ff = Float(f)
                        self.sceneScale.x *= ff
                        self.sceneScale.y *= ff
                    }
                    self.view?.setNeedsDisplay()
                }
            }

            // Vertical pan
            nc.addObserver(
                forName: NSNotification.Name("WebGLCameraMoveUp"),
                object:  nil,
                queue:   nil
            ) { [weak self] _ in
                DispatchQueue.main.async { [weak self] in
                    self?.cameraOffset.y += 0.15
                    self?.view?.setNeedsDisplay()
                }
            }

            nc.addObserver(
                forName: NSNotification.Name("WebGLCameraMoveDown"),
                object:  nil,
                queue:   nil
            ) { [weak self] _ in
                DispatchQueue.main.async { [weak self] in
                    self?.cameraOffset.y -= 0.15
                    self?.view?.setNeedsDisplay()
                }
            }

            // Horizontal pan
            nc.addObserver(
                forName: NSNotification.Name("WebGLCameraMoveLeft"),
                object:  nil,
                queue:   nil
            ) { [weak self] _ in
                DispatchQueue.main.async { [weak self] in
                    self?.cameraOffset.x -= 0.2
                    self?.view?.setNeedsDisplay()
                }
            }

            nc.addObserver(
                forName: NSNotification.Name("WebGLCameraMoveRight"),
                object:  nil,
                queue:   nil
            ) { [weak self] _ in
                DispatchQueue.main.async { [weak self] in
                    self?.cameraOffset.x += 0.2
                    self?.view?.setNeedsDisplay()
                }
            }
        }

        // MARK: - Matrix Utilities
        // All matrices are column-major simd_float4x4 matching Metal's convention.

        /// Standard right-handed perspective projection.
        private func matrixPerspectiveRightHand(
            fovyRadians: Float,
            aspectRatio: Float,
            nearZ:       Float,
            farZ:        Float
        ) -> simd_float4x4 {
            let ys  = 1 / tanf(fovyRadians * 0.5)
            let xs  = ys / aspectRatio
            let zs  = farZ / (nearZ - farZ)
            return simd_float4x4(columns: (
                SIMD4<Float>(xs,  0,       0,  0),
                SIMD4<Float>( 0, ys,       0,  0),
                SIMD4<Float>( 0,  0,      zs, -1),
                SIMD4<Float>( 0,  0, nearZ*zs,  0)
            ))
        }

        /// Rotation matrix around an arbitrary normalised axis.
        private func matrix4x4Rotation(radians: Float, axis: SIMD3<Float>) -> simd_float4x4 {
            let a = normalize(axis)
            let c = cosf(radians)
            let s = sinf(radians)
            let t = 1 - c
            return simd_float4x4(columns: (
                SIMD4<Float>(t*a.x*a.x + c,      t*a.x*a.y + s*a.z, t*a.x*a.z - s*a.y, 0),
                SIMD4<Float>(t*a.x*a.y - s*a.z,  t*a.y*a.y + c,     t*a.y*a.z + s*a.x, 0),
                SIMD4<Float>(t*a.x*a.z + s*a.y,  t*a.y*a.z - s*a.x, t*a.z*a.z + c,     0),
                SIMD4<Float>(0,                   0,                  0,                  1)
            ))
        }

        /// Standard look-at view matrix.
        private func matrixLookAt(
            eye:    SIMD3<Float>,
            target: SIMD3<Float>,
            up:     SIMD3<Float>
        ) -> simd_float4x4 {
            let z = normalize(eye - target)        // forward (from target to eye)
            let x = normalize(cross(up, z))        // right
            let y = cross(z, x)                    // recomputed up
            let t = SIMD3<Float>(-dot(x, eye), -dot(y, eye), -dot(z, eye))
            return simd_float4x4(columns: (
                SIMD4<Float>(x.x, y.x, z.x, 0),
                SIMD4<Float>(x.y, y.y, z.y, 0),
                SIMD4<Float>(x.z, y.z, z.z, 0),
                SIMD4<Float>(t.x, t.y, t.z, 1)
            ))
        }

        /// Translation matrix.
        private func matrix4x4Translation(_ tx: Float, _ ty: Float, _ tz: Float) -> simd_float4x4 {
            simd_float4x4(columns: (
                SIMD4<Float>(1, 0, 0, 0),
                SIMD4<Float>(0, 1, 0, 0),
                SIMD4<Float>(0, 0, 1, 0),
                SIMD4<Float>(tx, ty, tz, 1)
            ))
        }

        /// Non-uniform scale matrix.
        private func matrix4x4Scale(_ sx: Float, _ sy: Float, _ sz: Float) -> simd_float4x4 {
            simd_float4x4(columns: (
                SIMD4<Float>(sx, 0,  0,  0),
                SIMD4<Float>(0,  sy, 0,  0),
                SIMD4<Float>(0,  0,  sz, 0),
                SIMD4<Float>(0,  0,  0,  1)
            ))
        }
    }
}

#if DEBUG
/// Samples parsed SH0 (DC) coefficients — same slot as typical `f_dc` in 3DGS PLY — after load. See `SANITY CHECK` on `Coordinator`’s SH calibration comment block.
private func logSphericalHarmonicsSanityCheck(points: [SplatPoint], plyFileName: String) {
    let sampleCount = min(8, points.count)
    guard sampleCount > 0 else { return }
    let degreeLabel = points.first.map { String(describing: $0.color.shDegree) } ?? "?"
    var positiveDCCount = 0
    var parts: [String] = []
    parts.reserveCapacity(sampleCount)
    for index in 0..<sampleCount {
        let sh0 = points[index].color.sh0
        if sh0.x > 0 || sh0.y > 0 || sh0.z > 0 { positiveDCCount += 1 }
        parts.append("#\(index) sh0=\(String(format: "%.3f,%.3f,%.3f", sh0.x, sh0.y, sh0.z))")
    }
    logDebug("🔬 [GaussianSplatView][SH sanity] \(plyFileName): file SH degree≈\(degreeLabel) — \(parts.joined(separator: " | "))")
    logDebug("🔬 [GaussianSplatView][SH sanity] \(positiveDCCount)/\(sampleCount) sampled splats have any SH0 component >0 (negative SH0 is normal for dark surfaces; coarse hint only). For SH1+ PLYs, compare view-dependent colour vs WebGL if needed.")
    logDebug("🔬 [GaussianSplatView][SH sanity] Pass criteria vs WebGL/SparkJS: neutral walls within ~±10% brightness; colored view-dependent gradients oriented identically.")
}
#endif

// MARK: - Preview

#Preview {
    GaussianSplatView(
        plyURL: URL(fileURLWithPath: "/tmp/test.ply"),
        isLoading: .constant(true),
        loadError: .constant(nil as String?),
        zoomLevel: .constant(1.0),
        infiniteZoom: true,
        onBoundsAvailable: nil
    )
}
