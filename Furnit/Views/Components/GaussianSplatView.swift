import SwiftUI
import UIKit
import MetalKit
import Metal
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
/// SHARP / SharpRoom: some saved PLYs need classic SHARP orientation/rendering even when the file name no longer carries `_classic`.
/// For those loads, Y/Z are flipped to match view scale `(1,-1,-1)` after ``RoomBounds/defaultSplatCameraEyeAndTarget(photoOrientation:)`` (same min-Z back → max-Z front rail for portrait and landscape; ``photoOrientation`` is for AR roll).
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

    /// Source room/photo orientation.
    let arReferenceOrientation: PhotoOrientation

    /// When true, apply SHARP classic camera/orientation behavior even if the file name does not include `_classic`.
    var treatAsClassicPly: Bool = false

    /// Called once after a successful load with bounds derived from MetalSplatter’s AABB.
    var onBoundsAvailable: ((RoomBounds) -> Void)?

    /// Optional bridge so parents (e.g. ``SharpRoomView``) can call ``GaussianSplatMeasurementHost/measureRoom()`` after frames have rendered.
    var measurementHost: GaussianSplatMeasurementHost? = nil

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
            arReferenceOrientation: arReferenceOrientation,
            treatAsClassicPly: treatAsClassicPly,
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
        let tap = UITapGestureRecognizer(target: context.coordinator,
                                         action: #selector(Coordinator.handleTap(_:)))
        let rotation = UIRotationGestureRecognizer(target: context.coordinator,
                                                   action: #selector(Coordinator.handleRotation(_:)))
        mtkView.addGestureRecognizer(pan)
        mtkView.addGestureRecognizer(pinch)
        mtkView.addGestureRecognizer(tap)
        mtkView.addGestureRecognizer(rotation)

        mtkView.delegate = context.coordinator
        measurementHost?.coordinator = context.coordinator
        context.coordinator.measurementHost = measurementHost
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
        measurementHost?.coordinator = coordinator
        coordinator.measurementHost = measurementHost
        mtkView.setNeedsDisplay()
    }

    static func dismantleUIView(_ mtkView: MTKView, coordinator: Coordinator) {
        coordinator.teardown(view: mtkView)
    }

    // MARK: - Coordinator

    final class Coordinator: NSObject, MTKViewDelegate, SplatDepthQueryable, SplatColorReadable {

        // ── Bindings ──────────────────────────────────────────────────────────
        @Binding var isLoading: Bool
        @Binding var loadError: String?
        @Binding var zoomLevel: Float
        /// Zoom used for projection (updated from parent in `updateUIView` without writing the binding).
        var appliedZoomLevel: Float = 1.0
        var onBoundsAvailable: ((RoomBounds) -> Void)?
        weak var measurementHost: GaussianSplatMeasurementHost?
        var supportsColorReadback: Bool { false }

        /// Last camera matrices used for splat rendering (for depth raycast).
        private var lastProjectionMatrix: simd_float4x4 = matrix_identity_float4x4
        private var lastViewMatrix: simd_float4x4 = matrix_identity_float4x4

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
        private var depthScratchTexture: MTLTexture?
        private var setupGeneration: Int = 0
        private var hasRenderedDepthFrame = false
        #if DEBUG
        private var droppedFrameCount = 0
        #endif

        /// After loading a PLY we **subsample** splat centers with a stride so the list stays ~`/maxTrimmedSampleCount`
        /// (~60k). **Camera framing** `sceneBoundsMin` / `sceneBoundsMax` come from **per-axis P3–P97** on that
        /// subsample (≈3% / 97%: `lo` / `hi` indices on sorted X, Y, and Z marginals — not paired 3D points).
        /// This property is the **full** strided subsample passed to ``RoomGeometryEngine/measurePLYExtents``;
        /// percentile trimming is applied **there** (and for framing), **not** by slicing this array.
        private(set) var trimmedSplatPositions: [SIMD3<Float>]?

        /// Screenshot requests: one frame consumes one completion (Metal readback; `drawHierarchy` cannot capture `CAMetalLayer` reliably).
        private var pendingScreenshotCompletions: [(UIImage?) -> Void] = []
        /// Shared, CPU-readable target for an extra composite pass — never blit **from** the CAMetalLayer drawable (that aborts under validation).
        private var screenshotCaptureTexture: MTLTexture?

        /// Axis-aligned bounds from loaded `SplatPoint` positions (MetalSplatter 1.x `SplatRenderer` does not expose `boundingBox` / `centroid`).
        private var sceneBoundsMin: SIMD3<Float>?
        private var sceneBoundsMax: SIMD3<Float>?
        private var sceneCentroid: SIMD3<Float>?
        /// True when viewing a SHARP `_classic.ply` splat; used to undo the extra (y,z) → (-y,-z) flip applied at export so the room is not upside down.
        private var isSharpClassicPly: Bool = false
        private var pendingFurnitureItem: SharpRoomFurnitureItem?
        private var placedFurniture: [SharpRoomPlacedFurniture] = []
        private var selectedFurnitureID: UUID?
        private weak var overlayView: UIView?
        private var furnitureShapeLayers: [UUID: CAShapeLayer] = [:]
        private var projectedFurnitureHitRects: [UUID: CGRect] = [:]

        // Camera orbit & pan state
        var cameraYaw:    Float = 0
        var cameraPitch:  Float = 0
        var cameraOffset: SIMD3<Float> = .zero
        var sceneScale:   SIMD2<Float> = SIMD2(1, 1)

        private var didRegisterSharpRoomNotifications = false
        private var notificationTokens: [NSObjectProtocol] = []
        private var modalHeavyWorkPaused = false
        private var modalHeavyWorkPauseCount = 0

        // ── Camera constants ──────────────────────────────────────────────────
        private let maxPitch:        Float = 1.4
        private let dragSensitivity: Float = 0.005
        private let exteriorCameraZ: Float = -8.0
        private let fovy:            Float = 65 * (.pi / 180)   // 65° → radians

        /// Whether to use wide-range dolly-style pinch zoom and a smaller near plane (matches the “Infinite Zoom” setting).
        let infiniteZoom: Bool
        let arReferenceOrientation: PhotoOrientation
        let treatAsClassicPly: Bool

        /// Linear RGB before S-curve composite (`BrightnessAdjust.metal`).
        /// Slight lift so SHARP rooms do not look flatter/duller than the classic preview.
        var splatCompositeExposure: Float = 1.12
        /// Additive lift on dark tonemapped samples (smoothstep mask).
        var splatCompositeShadowLift: Float = 0.05

        private struct DepthReadbackSnapshot {
            let width: Int
            let height: Int
            let depths: [Float]
            let invProjectionMatrix: simd_float4x4
            let invViewMatrix: simd_float4x4
            let viewportSize: CGSize
        }

        // ── Init ──────────────────────────────────────────────────────────────
        init(
            isLoading: Binding<Bool>,
            loadError: Binding<String?>,
            zoomLevel:  Binding<Float>,
            infiniteZoom: Bool,
            arReferenceOrientation: PhotoOrientation,
            treatAsClassicPly: Bool,
            onBoundsAvailable: ((RoomBounds) -> Void)?
        ) {
            _isLoading = isLoading
            _loadError = loadError
            _zoomLevel = zoomLevel
            self.infiniteZoom = infiniteZoom
            self.arReferenceOrientation = arReferenceOrientation
            self.treatAsClassicPly = treatAsClassicPly
            self.onBoundsAvailable = onBoundsAvailable
            super.init()
        }

        deinit {
            let nc = NotificationCenter.default
            for token in notificationTokens {
                nc.removeObserver(token)
            }
            notificationTokens.removeAll()
        }

        // MARK: Setup

        /// Defers work until after SwiftUI finishes the current update transaction (must run UIKit / bindings on the main actor).
        private func scheduleSwiftUIBindingUpdates(_ updates: @escaping @MainActor () -> Void) {
            Task { @MainActor in
                await Task.yield()
                updates()
            }
        }

        func teardown(view targetView: MTKView?) {
            setupGeneration += 1

            let nc = NotificationCenter.default
            for token in notificationTokens {
                nc.removeObserver(token)
            }
            notificationTokens.removeAll()
            didRegisterSharpRoomNotifications = false

            if Thread.isMainThread {
                targetView?.delegate = nil
                targetView?.isPaused = true
                targetView?.enableSetNeedsDisplay = true
            }
            if self.view === targetView || targetView == nil {
                view = nil
            }

            // Drain the queue so in-flight completion handlers run before we release the coordinator.
            // (Swift’s `MTLCommandQueue` protocol does not expose `waitUntilAllCommandsCompleted`.)
            if let queue = commandQueue, let fence = queue.makeCommandBuffer() {
                fence.commit()
                fence.waitUntilCompleted()
            }

            rendererLock.lock()
            _splatRenderer = nil
            rendererLock.unlock()

            splatScratchTexture = nil
            depthScratchTexture = nil
            screenshotCaptureTexture = nil
            hasRenderedDepthFrame = false
            compositePipeline = nil
            commandQueue = nil
            device = nil
            flushAllPendingScreenshotRequestsWithNil()
            currentURL = nil
            sceneBoundsMin = nil
            sceneBoundsMax = nil
            sceneCentroid = nil
            warmupEndTime = nil
            if Thread.isMainThread {
                overlayView?.removeFromSuperview()
            }
            overlayView = nil
            furnitureShapeLayers.removeAll()
            projectedFurnitureHitRects.removeAll()
        }

        func setup(view mtkView: MTKView, plyURL: URL) {
            setupGeneration += 1
            let thisGeneration = setupGeneration

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
            splatCompositeExposure = 1.12
            splatCompositeShadowLift = 0.05
            isSharpClassicPly = treatAsClassicPly || plyURL.lastPathComponent.contains("_classic")
            // Classic: camera framing uses bounds flipped to canonical space (matches (1,-1,-1) view scale); no extra yaw offset.
            cameraYaw = 0

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

            if overlayView == nil {
                let overlay = UIView(frame: mtkView.bounds)
                overlay.backgroundColor = .clear
                overlay.isUserInteractionEnabled = false
                overlay.autoresizingMask = [.flexibleWidth, .flexibleHeight]
                mtkView.addSubview(overlay)
                overlayView = overlay
                logDebug("🧩 [GaussianSplatView] furniture overlay view attached")
            }

            // ── Notifications ─────────────────────────────────────────────────
            if !didRegisterSharpRoomNotifications {
                registerSharpRoomParityNotifications()
                didRegisterSharpRoomNotifications = true
            }

            // ── Load PLY ──────────────────────────────────────────────────────
            logDebug("🎨 [GaussianSplatView] Starting to load PLY: \(plyURL.lastPathComponent)")
            logMemorySnapshot("GaussianSplatView.setup", details: "phase=begin_load ply=\(plyURL.lastPathComponent)")

            // Capture view properties before async hop
            let clearColor   = mtkView.clearColor
            let colorFormat  = mtkView.colorPixelFormat
            let depthFormat  = mtkView.depthStencilPixelFormat
            let sampleCount  = mtkView.sampleCount

            splatRenderer = nil
            sceneBoundsMin = nil
            sceneBoundsMax = nil
            sceneCentroid = nil
            warmupEndTime = nil

            scheduleSwiftUIBindingUpdates {
                self.isLoading = true
                self.loadError = nil
                Task { [weak self] in
                    guard let self else { return }
                    guard self.setupGeneration == thisGeneration else { return }
                    do {
                    var points = try await AutodetectSceneReader(plyURL).readAll()
                    guard !points.isEmpty else { throw GaussianSplatLoadError.emptyScene }
                    logMemorySnapshot(
                        "GaussianSplatView.setup",
                        details: "phase=after_read_all ply=\(plyURL.lastPathComponent) splats=\(points.count)"
                    )

                    // Full AABB over all splats is exact (below). For camera framing we subsample (stride → ~60k),
                    // then build an AABB from per-axis P3–P97 on that subsample so we don’t sort millions of
                    // coordinates on every open. `trimmedSplatPositions` stores the whole subsample; P3–P97 is
                    // only used to set min/max framing bounds, not to shrink the array handed to geometry.
                    let splatCount = points.count
                    let maxTrimmedSampleCount = 60_000
                    let trimSampleStride = max(1, splatCount / maxTrimmedSampleCount)
                    var sampledPositions: [SIMD3<Float>] = []
                    sampledPositions.reserveCapacity(min(splatCount, maxTrimmedSampleCount))

                    var fullMin = points[0].position
                    var fullMax = points[0].position
                    var positionSum = SIMD3<Float>.zero
                    for (index, point) in points.enumerated() {
                        let position = point.position
                        fullMin = simd_min(fullMin, position)
                        fullMax = simd_max(fullMax, position)
                        positionSum += position
                        if index % trimSampleStride == 0 {
                            sampledPositions.append(position)
                        }
                    }
                    let centroid = positionSum / Float(points.count)

                    if sampledPositions.isEmpty {
                        sampledPositions.append(points[0].position)
                    }

                    // P3–P97 per axis on the **same** subsample (independent marginal percentiles).
                    // Sorting X/Y/Z separately then indexing with the same i was incorrect (mixed unrelated samples).
                    let xs = sampledPositions.map(\.x).sorted()
                    let ys = sampledPositions.map(\.y).sorted()
                    let zs = sampledPositions.map(\.z).sorted()
                    let trimmedSampleCount = sampledPositions.count
                    let lo = min(trimmedSampleCount - 1, max(0, Int(Float(trimmedSampleCount) * 0.03)))
                    let hi = min(trimmedSampleCount - 1, max(lo, Int(Float(trimmedSampleCount) * 0.97)))
                    let trimmedMin = SIMD3<Float>(xs[lo], ys[lo], zs[lo])
                    let trimmedMax = SIMD3<Float>(xs[hi], ys[hi], zs[hi])

                    let cameraBoundsMin = trimmedMin
                    let cameraBoundsMax = trimmedMax
                    let cameraCentroid  = centroid

                    #if DEBUG
                    if !isSharpClassicPly {
                        logSphericalHarmonicsSanityCheck(points: points, plyFileName: plyURL.lastPathComponent)
                    } else {
                        logDebug("🔬 [GaussianSplatView] Skipping SH check — classic PLY uses uchar RGB (no SH)")
                    }
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
                    points.removeAll(keepingCapacity: false)
                    logMemorySnapshot(
                        "GaussianSplatView.setup",
                        details: "phase=after_renderer_add_chunk ply=\(plyURL.lastPathComponent) splats=\(renderer.splatCount)"
                    )

                    let bounds = RoomBounds(
                        minX: fullMin.x, maxX: fullMax.x,
                        minY: fullMin.y, maxY: fullMax.y,
                        minZ: fullMin.z, maxZ: fullMax.z
                    )
                    let loadedSplatCount = renderer.splatCount
                    self.scheduleSwiftUIBindingUpdates { [weak self] in
                        guard let self, self.setupGeneration == thisGeneration else { return }
                        logDebug("📐 [GaussianSplatView] Full bounds: X[\(fullMin.x),\(fullMax.x)] Y[\(fullMin.y),\(fullMax.y)] Z[\(fullMin.z),\(fullMax.z)]")
                        logDebug("📐 [GaussianSplatView] Trimmed P3–P97 sample=\(trimmedSampleCount)/\(splatCount): X[\(trimmedMin.x),\(trimmedMax.x)] Y[\(trimmedMin.y),\(trimmedMax.y)] Z[\(trimmedMin.z),\(trimmedMax.z)]")
                        logDebug("📐 [GaussianSplatView] Camera framing AABB: X[\(cameraBoundsMin.x),\(cameraBoundsMax.x)] Y[\(cameraBoundsMin.y),\(cameraBoundsMax.y)] Z[\(cameraBoundsMin.z),\(cameraBoundsMax.z)]")
                        logDebug("✅ [GaussianSplatView] Loaded \(loadedSplatCount) splats centroid=\(centroid) cameraCentroid=\(cameraCentroid) (framing uses camera bounds)")
                        logMemorySnapshot(
                            "GaussianSplatView.setup",
                            details: "phase=ready ply=\(plyURL.lastPathComponent) splats=\(loadedSplatCount)"
                        )
                        self.onBoundsAvailable?(bounds)
                        self.sceneBoundsMin = cameraBoundsMin
                        self.sceneBoundsMax = cameraBoundsMax
                        self.sceneCentroid = cameraCentroid
                        self.trimmedSplatPositions = sampledPositions
                        self.splatRenderer = renderer
                        self.isLoading = false
                        self.warmupEndTime = CFAbsoluteTimeGetCurrent() + 3.0
                        mtkView.setNeedsDisplay()
                    }
                } catch {
                    self.scheduleSwiftUIBindingUpdates { [weak self] in
                        guard let self, self.setupGeneration == thisGeneration else { return }
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
            screenshotCaptureTexture = nil
        }

        /// Enqueues a one-shot capture of the next composite frame (offscreen shared texture → `UIImage` on the main queue).
        func scheduleScreenshotCapture(completion: @escaping (UIImage?) -> Void) {
            let enqueue = { [weak self] in
                self?.pendingScreenshotCompletions.append(completion)
                self?.view?.setNeedsDisplay()
            }
            if Thread.isMainThread {
                enqueue()
            } else {
                DispatchQueue.main.async(execute: enqueue)
            }
        }

        private func flushAllPendingScreenshotRequestsWithNil() {
            let completions = pendingScreenshotCompletions
            pendingScreenshotCompletions.removeAll()
            for completion in completions {
                DispatchQueue.main.async {
                    completion(nil)
                }
            }
        }

        private func ensureScreenshotCaptureTexture(device: MTLDevice, width: Int, height: Int, pixelFormat: MTLPixelFormat) -> MTLTexture? {
            if let existing = screenshotCaptureTexture,
               existing.width == width, existing.height == height, existing.pixelFormat == pixelFormat {
                return existing
            }
            let descriptor = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: pixelFormat,
                width: width,
                height: height,
                mipmapped: false
            )
            descriptor.storageMode = .shared
            descriptor.usage = [.renderTarget, .shaderRead]
            guard let texture = device.makeTexture(descriptor: descriptor) else { return nil }
            screenshotCaptureTexture = texture
            return texture
        }

        private static func makeUIImage(fromBGRA texture: MTLTexture, scale: CGFloat) -> UIImage? {
            let width = texture.width
            let height = texture.height
            let bytesPerPixel = 4
            let rowBytes = ((width * bytesPerPixel + 255) / 256) * 256
            var raw = [UInt8](repeating: 0, count: rowBytes * height)
            let region = MTLRegion(origin: MTLOrigin(x: 0, y: 0, z: 0), size: MTLSize(width: width, height: height, depth: 1))
            texture.getBytes(&raw, bytesPerRow: rowBytes, from: region, mipmapLevel: 0)
            guard let provider = CGDataProvider(data: Data(raw) as CFData) else { return nil }
            let bitmapInfo = CGBitmapInfo.byteOrder32Little.union(CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedFirst.rawValue))
            guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) else { return nil }
            guard let cgImage = CGImage(
                width: width,
                height: height,
                bitsPerComponent: 8,
                bitsPerPixel: 32,
                bytesPerRow: rowBytes,
                space: colorSpace,
                bitmapInfo: bitmapInfo,
                provider: provider,
                decode: nil,
                shouldInterpolate: true,
                intent: .defaultIntent
            ) else { return nil }
            return UIImage(cgImage: cgImage, scale: scale, orientation: .up)
        }

        func draw(in view: MTKView) {
            guard let commandQueue = commandQueue, let metalDevice = device else {
                flushAllPendingScreenshotRequestsWithNil()
                return
            }
            guard let renderer = splatRenderer else {
                flushAllPendingScreenshotRequestsWithNil()
                return
            }
            guard let drawable = view.currentDrawable else {
                if !pendingScreenshotCompletions.isEmpty {
                    view.setNeedsDisplay()
                }
                return
            }

            let timeout = DispatchTime.now() + 0.1
            if inFlightSemaphore.wait(timeout: timeout) != .success {
                #if DEBUG
                droppedFrameCount += 1
                if droppedFrameCount % 10 == 1 {
                    logDebug("⚠️ [GaussianSplatView] dropped \(droppedFrameCount) frames (GPU backpressure)")
                }
                #endif
                // If we are in warm-up, reschedule so we do not stall the sharpening loop.
                if warmupEndTime != nil {
                    view.setNeedsDisplay()
                }
                return
            }

            guard let commandBuffer = commandQueue.makeCommandBuffer() else {
                inFlightSemaphore.signal()
                flushAllPendingScreenshotRequestsWithNil()
                return
            }
            let inflightSemaphore = inFlightSemaphore
            commandBuffer.addCompletedHandler { _ in
                inflightSemaphore.signal()
            }

            let screenshotCompletion: ((UIImage?) -> Void)? = pendingScreenshotCompletions.isEmpty
                ? nil
                : pendingScreenshotCompletions.removeFirst()

            let vp = viewport
            lastProjectionMatrix = vp.projectionMatrix
            lastViewMatrix = vp.viewMatrix
            refreshFurnitureOverlay()

            if let pipeline = compositePipeline,
               let scratch = ensureSplatScratchTexture(
                   device: metalDevice,
                   drawable: drawable,
                   pixelFormat: view.colorPixelFormat
               ),
               let depthScratch = ensureDepthScratchTexture(device: metalDevice, width: scratch.width, height: scratch.height) {
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
                    if let completion = screenshotCompletion {
                        DispatchQueue.main.async {
                            completion(nil)
                        }
                    }
                    commandBuffer.present(drawable)
                    commandBuffer.commit()
                    return
                }

                if let completion = screenshotCompletion,
                   let captureTexture = ensureScreenshotCaptureTexture(
                       device: metalDevice,
                       width: scratch.width,
                       height: scratch.height,
                       pixelFormat: view.colorPixelFormat
                   ) {
                    let capturePassDesc = MTLRenderPassDescriptor()
                    capturePassDesc.colorAttachments[0].texture = captureTexture
                    capturePassDesc.colorAttachments[0].loadAction = .dontCare
                    capturePassDesc.colorAttachments[0].storeAction = .store
                    if let captureEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: capturePassDesc) {
                        captureEncoder.setRenderPipelineState(pipeline)
                        captureEncoder.setFragmentTexture(scratch, index: 0)
                        var compositeParams: [Float] = [splatCompositeExposure, splatCompositeShadowLift]
                        captureEncoder.setFragmentBytes(&compositeParams, length: MemoryLayout<Float>.stride * 2, index: 0)
                        captureEncoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
                        captureEncoder.endEncoding()
                        let contentScale = view.contentScaleFactor
                        commandBuffer.addCompletedHandler { _ in
                            let image = Self.makeUIImage(fromBGRA: captureTexture, scale: contentScale)
                            DispatchQueue.main.async {
                                completion(image)
                            }
                        }
                    } else {
                        DispatchQueue.main.async {
                            completion(nil)
                        }
                    }
                } else if let completion = screenshotCompletion {
                    DispatchQueue.main.async {
                        completion(nil)
                    }
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
                    if let completion = screenshotCompletion {
                        DispatchQueue.main.async {
                            completion(nil)
                        }
                    }
                    commandBuffer.present(drawable)
                    commandBuffer.commit()
                    return
                }
                if let completion = screenshotCompletion {
                    DispatchQueue.main.async {
                        completion(nil)
                    }
                }
            }

            hasRenderedDepthFrame = true

            commandBuffer.present(drawable)
            commandBuffer.commit()

            if !pendingScreenshotCompletions.isEmpty {
                view.setNeedsDisplay()
            }
            // During warm-up, keep requesting new frames so the room sharpens quickly even when idle.
            if let warmupEndTime, CFAbsoluteTimeGetCurrent() < warmupEndTime {
                view.setNeedsDisplay()
            } else {
                warmupEndTime = nil
            }
        }

        // MARK: Viewport — camera & MetalSplatter matrix
        //
        // `_classic.ply`: view scale (1,-1,-1); camera and geometry are both seen in the flipped space.

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
                // Back-center → front wall (``RoomBounds/defaultSplatCameraEyeAndTarget``). Same padding in AR
                // and touch; matches tighter back-wall standoff in ``RoomMeasurement``.
                let baseCameraPadding: Float = 0.05
                let (eye, target) = rb.defaultSplatCameraEyeAndTarget(
                    cameraPadding: baseCameraPadding,
                    photoOrientation: arReferenceOrientation
                )
                camPos = eye + cameraOffset
                lookAt = target + cameraOffset
            } else if let centroid {
                camPos = SIMD3<Float>(centroid.x, centroid.y, centroid.z + exteriorCameraZ) + cameraOffset
                lookAt = centroid + cameraOffset
            } else {
                camPos = SIMD3<Float>(0, 0, 3) + cameraOffset
                lookAt = SIMD3<Float>(0, 0, 0) + cameraOffset
            }

            // Classic PLY: apply the same Y/Z flip to the camera so it sees the room in the same space as the scaled geometry.
            if isSharpClassicPly {
                camPos = SIMD3<Float>(camPos.x, -camPos.y, -camPos.z)
                lookAt = SIMD3<Float>(lookAt.x, -lookAt.y, -lookAt.z)
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

            // ── Apply user orbit / zoom ────────────────────────────────────────────────────
            let yawMatrix   = matrix4x4Rotation(radians: cameraYaw,
                                                 axis: SIMD3<Float>(0, 1, 0))
            let pitchMatrix = matrix4x4Rotation(radians: cameraPitch,
                                                 axis: SIMD3<Float>(1, 0, 0))
            let userRotation = pitchMatrix * yawMatrix
            let offset4      = userRotation * SIMD4<Float>(camPos - lookAt, 0)
            let rotatedEye   = lookAt + SIMD3<Float>(offset4.x, offset4.y, offset4.z)

            let baseView = matrixLookAt(
                eye:    rotatedEye,
                target: lookAt,
                up:     SIMD3<Float>(0, 1, 0)
            )

            let viewMatrix = baseView * scaleMatrix

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

        // MARK: - Room measurement (depth buffer raycast)

        /// Queues one redraw so depth / view–proj matrices match the current camera before a CPU depth read.
        func requestRedrawForDepthMeasure() {
            view?.setNeedsDisplay()
        }

        var viewportSize: CGSize {
            if let view {
                return view.bounds.size
            }
            return drawableSize
        }

        /// Orbit, zoom, and scene scale back to defaults.
        func performSharpRoomRecenter() {
            cameraYaw = 0
            cameraPitch = 0
            cameraOffset = .zero
            appliedZoomLevel = 1.0
            zoomLevel = 1.0
            sceneScale = SIMD2(1, 1)
            logDebug("🎯 [GaussianSplatView] Sharp room recenter: orbit/zoom reset")
            view?.setNeedsDisplay()
        }

        func setModalHeavyWorkPaused(_ paused: Bool) {
            if paused {
                modalHeavyWorkPauseCount += 1
                guard !modalHeavyWorkPaused else { return }
                modalHeavyWorkPaused = true
                return
            }

            modalHeavyWorkPauseCount = max(0, modalHeavyWorkPauseCount - 1)
            guard modalHeavyWorkPauseCount == 0, modalHeavyWorkPaused else { return }
            modalHeavyWorkPaused = false
            view?.setNeedsDisplay()
        }

        func setPendingFurnitureItem(_ item: SharpRoomFurnitureItem?) {
            pendingFurnitureItem = item
            let text = item.map {
                String(
                    format: "Pending %@ %.2f×%.2f×%.2f m — tap floor to place",
                    $0.category, $0.dimensions.x, $0.dimensions.y, $0.dimensions.z
                )
            } ?? "Furniture: none"
            measurementHost?.updateFurnitureStatus(text, count: placedFurniture.count)
            logDebug("🪑 [GaussianSplatFurniture] pending furniture=\(item?.category ?? "nil")")
        }

        func clearPlacedFurniture() {
            placedFurniture.removeAll()
            selectedFurnitureID = nil
            pendingFurnitureItem = nil
            updateFurnitureStatusSummary("Furniture cleared")
            refreshFurnitureOverlay()
            view?.setNeedsDisplay()
            logDebug("🧹 [GaussianSplatFurniture] cleared placed furniture")
        }

        func rotateSelectedFurniture(by radians: Float) {
            guard let selectedFurnitureID,
                  let index = placedFurniture.firstIndex(where: { $0.id == selectedFurnitureID }) else {
                measurementHost?.updateFurnitureStatus("No selected furniture to rotate", count: placedFurniture.count)
                return
            }
            placedFurniture[index].rotationY += radians
            recheckFitment(at: index)
            updateFurnitureStatusSummary(
                String(
                    format: "Rotated %@ to %.1f°",
                    placedFurniture[index].item.category,
                    placedFurniture[index].rotationY * 180 / .pi
                )
            )
            refreshFurnitureOverlay()
            view?.setNeedsDisplay()
            logDebug("🔄 [GaussianSplatFurniture] rotated selected furniture by \(radians) rad")
        }

        func depthAt(screenPoint: CGPoint) -> Float? {
            guard let snapshot = makeDepthSnapshot(),
                  let worldPoint = worldPointAt(screenPoint: screenPoint, snapshot: snapshot) else {
                return nil
            }

            let cameraPosition = cameraWorldPosition(invView: snapshot.invViewMatrix)
            return simd_distance(worldPoint, cameraPosition)
        }

        func unproject(screenPoint: CGPoint, depth: Float) -> SIMD3<Float>? {
            guard depth.isFinite, depth > 0,
                  let snapshot = makeDepthSnapshot() else {
                return nil
            }

            let cameraPosition = cameraWorldPosition(invView: snapshot.invViewMatrix)
            guard let rayDirection = worldRayDirection(screenPoint: screenPoint, snapshot: snapshot) else {
                return nil
            }
            return cameraPosition + rayDirection * depth
        }

        func sampleDepthGrid(rows: Int, cols: Int) -> [[Float?]] {
            guard rows > 0, cols > 0,
                  let snapshot = makeDepthSnapshot() else { return [] }

            var grid = [[Float?]](repeating: [Float?](repeating: nil, count: cols), count: rows)
            let viewport = snapshot.viewportSize
            guard viewport.width > 0, viewport.height > 0 else { return [] }

            for row in 0..<rows {
                for col in 0..<cols {
                    let point = CGPoint(
                        x: (CGFloat(col) + 0.5) / CGFloat(cols) * viewport.width,
                        y: (CGFloat(row) + 0.5) / CGFloat(rows) * viewport.height
                    )
                    grid[row][col] = depthAt(screenPoint: point, snapshot: snapshot)
                }
            }

            DepthDiagnostics.logGridSummary(label: "GaussianSplatView.sampleDepthGrid", grid: grid)
            return grid
        }

        func buildPointCloud(rows: Int, cols: Int, maxDistance: Float) -> [SIMD3<Float>] {
            guard rows > 0, cols > 0,
                  let snapshot = makeDepthSnapshot() else { return [] }

            let viewport = snapshot.viewportSize
            guard viewport.width > 0, viewport.height > 0 else { return [] }

            var points: [SIMD3<Float>] = []
            points.reserveCapacity(rows * cols)

            for row in 0..<rows {
                for col in 0..<cols {
                    let point = CGPoint(
                        x: (CGFloat(col) + 0.5) / CGFloat(cols) * viewport.width,
                        y: (CGFloat(row) + 0.5) / CGFloat(rows) * viewport.height
                    )
                    guard let depth = depthAt(screenPoint: point, snapshot: snapshot),
                          depth.isFinite,
                          depth > 0,
                          depth < maxDistance,
                          let worldPoint = worldPointAt(screenPoint: point, snapshot: snapshot) else {
                        continue
                    }
                    points.append(worldPoint)
                }
            }

            logDebug(
                "🩺 [GaussianSplatView] buildPointCloud rows=\(rows) cols=\(cols) " +
                "points=\(points.count) snapshot=\(snapshot.width)x\(snapshot.height)"
            )
            return points
        }

        func colorAt(screenPoint _: CGPoint) -> SIMD3<Float>? {
            nil
        }

        /// Copies the last rendered splat depth texture to CPU and derives scene-unit **W×H×D** from a grid
        /// point cloud (**view-dependent** fallback vs full PLY).
        ///
        /// **Trimming:** ``trimmedSceneDimensions(points:trimFraction:)`` / ``trimmedRange(values:trimFraction:)``
        /// drop **the same fraction from both ends** of each axis’s sorted coordinates (this path passes
        /// **0.06** → ~**6%** dropped below the low quantile and ~**6%** above the high on X, Y, and Z
        /// independently). That yields a more robust span than raw min/max on depth noise / outliers.
        func measureRoomFromDepthBuffer() -> RoomRaycastDimensions? {
            guard let snapshot = makeDepthSnapshot() else { return nil }
            let sampledPoints = buildPointCloud(
                snapshot: snapshot,
                rows: 36,
                cols: 36,
                maxDistance: 12.0
            )
            guard sampledPoints.count >= 80 else {
                logDebug("❌ [Measure] Point-cloud room measure unavailable — only \(sampledPoints.count) valid samples")
                return nil
            }

            // Symmetric per-axis trim (6% each tail); see trimmedSceneDimensions / trimmedRange.
            let dims = trimmedSceneDimensions(points: sampledPoints, trimFraction: 0.06)
            let fname = currentURL?.lastPathComponent ?? "unknown"
            let plyKind = isSharpClassicPly ? "classic_ply" : "base_ply"
            logPlyBoundsDiagnostic(
                "Metal depth raycast room (\(plyKind) file=\(fname)) su: " +
                "W=\(String(format: "%.3f", dims.width)) H=\(String(format: "%.3f", dims.height)) D=\(String(format: "%.3f", dims.depth))"
            )
            logDebug(
                "🩺 [GaussianSplatView] measureRoomFromDepthBuffer samples=\(sampledPoints.count) " +
                "trimmed_su=\(String(format: "%.3f", dims.width))×\(String(format: "%.3f", dims.height))×\(String(format: "%.3f", dims.depth))"
            )
            return dims
        }

        private func buildPointCloud(
            snapshot: DepthReadbackSnapshot,
            rows: Int,
            cols: Int,
            maxDistance: Float
        ) -> [SIMD3<Float>] {
            let viewport = snapshot.viewportSize
            guard rows > 0, cols > 0,
                  viewport.width > 0, viewport.height > 0 else { return [] }

            var points: [SIMD3<Float>] = []
            points.reserveCapacity(rows * cols)

            for row in 0..<rows {
                for col in 0..<cols {
                    let point = CGPoint(
                        x: (CGFloat(col) + 0.5) / CGFloat(cols) * viewport.width,
                        y: (CGFloat(row) + 0.5) / CGFloat(rows) * viewport.height
                    )
                    guard let depth = depthAt(screenPoint: point, snapshot: snapshot),
                          depth.isFinite,
                          depth > 0,
                          depth < maxDistance,
                          let worldPoint = worldPointAt(screenPoint: point, snapshot: snapshot) else {
                        continue
                    }
                    points.append(worldPoint)
                }
            }

            return points
        }

        /// World-axis AABB span after **`trimmedRange`** on X, Y, and Z separately (marginal trim, not 3D peel).
        private func trimmedSceneDimensions(
            points: [SIMD3<Float>],
            trimFraction: Float
        ) -> RoomRaycastDimensions {
            let trimmedX = trimmedRange(values: points.map(\.x), trimFraction: trimFraction)
            let trimmedY = trimmedRange(values: points.map(\.y), trimFraction: trimFraction)
            let trimmedZ = trimmedRange(values: points.map(\.z), trimFraction: trimFraction)

            return RoomRaycastDimensions(
                width: max(0.001, trimmedX.upper - trimmedX.lower),
                height: max(0.001, trimmedY.upper - trimmedY.lower),
                depth: max(0.001, trimmedZ.upper - trimmedZ.lower)
            )
        }

        /// Sorted 1D values: drop **≈`trimFraction` × N** samples from the **low** end and the same count
        /// from the **high** end (capped so indices never cross). Used for depth-grid fallback sizing only.
        private func trimmedRange(values: [Float], trimFraction: Float) -> (lower: Float, upper: Float) {
            guard !values.isEmpty else { return (0, 0.001) }
            let sorted = values.sorted()
            let maxTrimIndex = max(0, (sorted.count - 1) / 2)
            let trimCount = min(Int(Float(sorted.count) * trimFraction), maxTrimIndex)
            let lowerIndex = min(trimCount, sorted.count - 1)
            let upperIndex = max(lowerIndex, sorted.count - 1 - trimCount)
            return (sorted[lowerIndex], sorted[upperIndex])
        }

        private func makeDepthSnapshot() -> DepthReadbackSnapshot? {
            guard hasRenderedDepthFrame,
                  let metalDevice = device,
                  let commandQueue = commandQueue,
                  let depthTexture = depthScratchTexture,
                  splatRenderer != nil else {
                logDebug("❌ [SplatDepth] Snapshot unavailable — frame/device/depth missing")
                return nil
            }

            let width = depthTexture.width
            let height = depthTexture.height
            guard width > 0, height > 0 else { return nil }

            let bytesPerRow = width * MemoryLayout<Float>.stride
            let totalBytes = bytesPerRow * height
            guard let buffer = metalDevice.makeBuffer(length: totalBytes, options: .storageModeShared),
                  let commandBuffer = commandQueue.makeCommandBuffer(),
                  let blit = commandBuffer.makeBlitCommandEncoder() else {
                logDebug("❌ [SplatDepth] Failed to allocate depth readback resources")
                return nil
            }

            blit.copy(
                from: depthTexture,
                sourceSlice: 0,
                sourceLevel: 0,
                sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                sourceSize: MTLSize(width: width, height: height, depth: 1),
                to: buffer,
                destinationOffset: 0,
                destinationBytesPerRow: bytesPerRow,
                destinationBytesPerImage: totalBytes
            )
            blit.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()

            let pointer = buffer.contents().bindMemory(to: Float.self, capacity: width * height)
            let depths = Array(UnsafeBufferPointer(start: pointer, count: width * height))
            return DepthReadbackSnapshot(
                width: width,
                height: height,
                depths: depths,
                invProjectionMatrix: simd_inverse(lastProjectionMatrix),
                invViewMatrix: simd_inverse(lastViewMatrix),
                viewportSize: viewportSize
            )
        }

        private func depthAt(screenPoint: CGPoint, snapshot: DepthReadbackSnapshot) -> Float? {
            guard let worldPoint = worldPointAt(screenPoint: screenPoint, snapshot: snapshot) else { return nil }
            let cameraPosition = cameraWorldPosition(invView: snapshot.invViewMatrix)
            return simd_distance(worldPoint, cameraPosition)
        }

        private func worldPointAt(screenPoint: CGPoint, snapshot: DepthReadbackSnapshot) -> SIMD3<Float>? {
            guard let rawDepth = rawDepthAt(screenPoint: screenPoint, snapshot: snapshot) else { return nil }
            return unprojectDepthSample(
                screenPoint: screenPoint,
                rawDepth: rawDepth,
                snapshot: snapshot
            )
        }

        private func rawDepthAt(screenPoint: CGPoint, snapshot: DepthReadbackSnapshot) -> Float? {
            let normalized = normalizedCoordinates(for: screenPoint, viewport: snapshot.viewportSize)
            let pixelX = min(max(Int(normalized.x * Float(snapshot.width - 1)), 0), snapshot.width - 1)
            let pixelY = min(max(Int(normalized.y * Float(snapshot.height - 1)), 0), snapshot.height - 1)
            let depth = snapshot.depths[pixelY * snapshot.width + pixelX]
            guard depth.isFinite, depth > 0.001, depth < 0.9999 else { return nil }
            return depth
        }

        private func worldRayDirection(screenPoint: CGPoint, snapshot: DepthReadbackSnapshot) -> SIMD3<Float>? {
            let normalized = normalizedCoordinates(for: screenPoint, viewport: snapshot.viewportSize)
            let xNdc = normalized.x * 2 - 1
            let yNdc = -(normalized.y * 2 - 1)
            let clipFar = SIMD4<Float>(xNdc, yNdc, 1, 1)
            var eyeFar = snapshot.invProjectionMatrix * clipFar
            guard abs(eyeFar.w) > 1e-6 else { return nil }
            eyeFar /= eyeFar.w
            let worldFar = snapshot.invViewMatrix * eyeFar
            let camera = cameraWorldPosition(invView: snapshot.invViewMatrix)
            let direction = SIMD3<Float>(worldFar.x, worldFar.y, worldFar.z) - camera
            let lengthSquared = simd_length_squared(direction)
            guard lengthSquared > 1e-8 else { return nil }
            return direction / sqrt(lengthSquared)
        }

        private func normalizedCoordinates(for screenPoint: CGPoint, viewport: CGSize) -> SIMD2<Float> {
            guard viewport.width > 0, viewport.height > 0 else {
                return SIMD2<Float>(0.5, 0.5)
            }
            let x = Float(min(max(screenPoint.x / viewport.width, 0), 1))
            let y = Float(min(max(screenPoint.y / viewport.height, 0), 1))
            return SIMD2<Float>(x, y)
        }

        private func cameraWorldPosition(invView: simd_float4x4) -> SIMD3<Float> {
            let camera = invView * SIMD4<Float>(0, 0, 0, 1)
            return SIMD3<Float>(camera.x, camera.y, camera.z)
        }

        private func unprojectDepthSample(
            screenPoint: CGPoint,
            rawDepth d: Float,
            snapshot: DepthReadbackSnapshot
        ) -> SIMD3<Float>? {
            let normalized = normalizedCoordinates(for: screenPoint, viewport: snapshot.viewportSize)
            let zNdc = d * 2 - 1
            let xNdc = normalized.x * 2 - 1
            let yNdc = -(normalized.y * 2 - 1)
            let clip = SIMD4<Float>(xNdc, yNdc, zNdc, 1)
            var eye = snapshot.invProjectionMatrix * clip
            guard abs(eye.w) > 1e-6 else { return nil }
            eye /= eye.w
            let world = snapshot.invViewMatrix * eye
            return SIMD3<Float>(world.x, world.y, world.z)
        }

        private func placePendingFurniture(at screenPoint: CGPoint, item: SharpRoomFurnitureItem) {
            guard let hitPoint = floorIntersection(screenPoint: screenPoint) else {
                measurementHost?.updateFurnitureStatus("No floor hit for \(item.category) tap", count: placedFurniture.count)
                logDebug("❌ [GaussianSplatFurniture] no floor hit for pending furniture \(item.category)")
                return
            }
            var piece = SharpRoomPlacedFurniture(
                id: UUID(),
                item: item,
                position: SIMD3<Float>(hitPoint.x, hitPoint.y + item.dimensions.y / 2, hitPoint.z),
                rotationY: 0,
                fits: true,
                clearanceMeters: 0
            )
            placedFurniture.append(piece)
            selectedFurnitureID = piece.id
            pendingFurnitureItem = nil
            recheckFitment(at: placedFurniture.count - 1)
            piece = placedFurniture[placedFurniture.count - 1]
            updateFurnitureStatusSummary(
                String(
                    format: "Placed %@ clearance %.0f cm",
                    piece.item.category,
                    piece.clearanceMeters * 100
                )
            )
            refreshFurnitureOverlay()
            view?.setNeedsDisplay()
            logDebug(
                "✅ [GaussianSplatFurniture] placed furniture category=\(piece.item.category) pos=\(piece.position) " +
                "dims=\(piece.item.dimensions) fits=\(piece.fits)"
            )
        }

        private func selectFurniture(near screenPoint: CGPoint) {
            guard !projectedFurnitureHitRects.isEmpty else {
                selectedFurnitureID = nil
                updateFurnitureStatusSummary("No furniture to select")
                return
            }
            if let containing = projectedFurnitureHitRects.first(where: { $0.value.insetBy(dx: -18, dy: -18).contains(screenPoint) }) {
                selectedFurnitureID = containing.key
            } else {
                let best = projectedFurnitureHitRects.min { lhs, rhs in
                    let lc = CGPoint(x: lhs.value.midX, y: lhs.value.midY)
                    let rc = CGPoint(x: rhs.value.midX, y: rhs.value.midY)
                    let ld = hypot(lc.x - screenPoint.x, lc.y - screenPoint.y)
                    let rd = hypot(rc.x - screenPoint.x, rc.y - screenPoint.y)
                    return ld < rd
                }
                selectedFurnitureID = best?.key
            }
            if let selectedFurnitureID,
               let selected = placedFurniture.first(where: { $0.id == selectedFurnitureID }) {
                updateFurnitureStatusSummary("Selected \(selected.item.category)")
                logDebug("🎯 [GaussianSplatFurniture] selected furniture \(selected.item.category)")
            }
            refreshFurnitureOverlay()
            view?.setNeedsDisplay()
        }

        private func moveSelectedFurniture(_ id: UUID, to screenPoint: CGPoint) {
            guard let index = placedFurniture.firstIndex(where: { $0.id == id }),
                  let hitPoint = floorIntersection(screenPoint: screenPoint) else { return }
            let halfHeight = placedFurniture[index].item.dimensions.y / 2
            placedFurniture[index].position = SIMD3<Float>(hitPoint.x, hitPoint.y + halfHeight, hitPoint.z)
            recheckFitment(at: index)
            updateFurnitureStatusSummary(
                String(
                    format: "Moved %@ clearance %.0f cm",
                    placedFurniture[index].item.category,
                    placedFurniture[index].clearanceMeters * 100
                )
            )
            refreshFurnitureOverlay()
            view?.setNeedsDisplay()
        }

        private func floorIntersection(screenPoint: CGPoint) -> SIMD3<Float>? {
            guard let ray = screenPointToWorldRay(screenPoint),
                  let boundsMin = sceneBoundsMin else { return nil }
            let planePoint = SIMD3<Float>(0, boundsMin.y, 0)
            let planeNormal = SIMD3<Float>(0, 1, 0)
            let denom = simd_dot(ray.direction, planeNormal)
            guard abs(denom) > 0.0001 else { return nil }
            let t = simd_dot(planePoint - ray.origin, planeNormal) / denom
            guard t > 0 else { return nil }
            return ray.origin + ray.direction * t
        }

        private func screenPointToWorldRay(_ point: CGPoint) -> (origin: SIMD3<Float>, direction: SIMD3<Float>)? {
            guard viewportSize.width > 1, viewportSize.height > 1 else { return nil }
            let ndcX = Float(point.x / viewportSize.width) * 2 - 1
            let ndcY = 1 - Float(point.y / viewportSize.height) * 2

            let invProjection = simd_inverse(lastProjectionMatrix)
            let invView = simd_inverse(lastViewMatrix)

            var nearPoint = invProjection * SIMD4<Float>(ndcX, ndcY, 0, 1)
            var farPoint = invProjection * SIMD4<Float>(ndcX, ndcY, 1, 1)
            guard abs(nearPoint.w) > 1e-6, abs(farPoint.w) > 1e-6 else { return nil }
            nearPoint /= nearPoint.w
            farPoint /= farPoint.w

            let worldNear = invView * nearPoint
            let worldFar = invView * farPoint
            let origin = SIMD3<Float>(worldNear.x, worldNear.y, worldNear.z)
            let direction = normalize(SIMD3<Float>(
                worldFar.x - worldNear.x,
                worldFar.y - worldNear.y,
                worldFar.z - worldNear.z
            ))
            return (origin, direction)
        }

        private func recheckFitment(at index: Int) {
            guard placedFurniture.indices.contains(index),
                  let boundsMin = sceneBoundsMin,
                  let boundsMax = sceneBoundsMax else { return }

            var piece = placedFurniture[index]
            let halfW = piece.item.dimensions.x / 2
            let halfD = piece.item.dimensions.z / 2
            let cosR = cos(piece.rotationY)
            let sinR = sin(piece.rotationY)
            let corners = [
                SIMD2<Float>( halfW * cosR - halfD * sinR,  halfW * sinR + halfD * cosR),
                SIMD2<Float>(-halfW * cosR - halfD * sinR, -halfW * sinR + halfD * cosR),
                SIMD2<Float>( halfW * cosR + halfD * sinR,  halfW * sinR - halfD * cosR),
                SIMD2<Float>(-halfW * cosR + halfD * sinR, -halfW * sinR - halfD * cosR),
            ]

            var fits = true
            var minimumClearance = Float.greatestFiniteMagnitude

            for corner in corners {
                let worldX = piece.position.x + corner.x
                let worldZ = piece.position.z + corner.y
                let clearX = min(worldX - boundsMin.x, boundsMax.x - worldX)
                let clearZ = min(worldZ - boundsMin.z, boundsMax.z - worldZ)
                let clearance = min(clearX, clearZ)
                minimumClearance = min(minimumClearance, clearance)
                if clearance < 0 { fits = false }
            }

            for otherIndex in placedFurniture.indices where otherIndex != index {
                let other = placedFurniture[otherIndex]
                let centerDistance = simd_distance(
                    SIMD2<Float>(piece.position.x, piece.position.z),
                    SIMD2<Float>(other.position.x, other.position.z)
                )
                let minimumSpacing = (piece.item.dimensions.x + other.item.dimensions.x) / 2
                minimumClearance = min(minimumClearance, centerDistance - minimumSpacing)
                if centerDistance < minimumSpacing { fits = false }
            }

            piece.fits = fits
            piece.clearanceMeters = minimumClearance.isFinite ? max(0, minimumClearance) : 0
            placedFurniture[index] = piece
            logDebug(
                "📏 [GaussianSplatFurniture] fitment category=\(piece.item.category) fits=\(fits) " +
                "clearance=\(String(format: "%.3f", piece.clearanceMeters))m"
            )
        }

        private func updateFurnitureStatusSummary(_ text: String) {
            measurementHost?.updateFurnitureStatus(text, count: placedFurniture.count)
        }

        private func refreshFurnitureOverlay() {
            guard let overlayView else { return }
            var nextRects: [UUID: CGRect] = [:]
            var usedIDs = Set<UUID>()

            for piece in placedFurniture {
                let points = projectedBoxPoints(for: piece)
                let layer = furnitureShapeLayers[piece.id] ?? {
                    let created = CAShapeLayer()
                    created.fillColor = UIColor.clear.cgColor
                    created.lineWidth = (selectedFurnitureID == piece.id) ? 3 : 2
                    overlayView.layer.addSublayer(created)
                    furnitureShapeLayers[piece.id] = created
                    return created
                }()

                usedIDs.insert(piece.id)
                let path = UIBezierPath()
                if points.count == 8 {
                    let edges = [(0,1),(1,3),(3,2),(2,0),(4,5),(5,7),(7,6),(6,4),(0,4),(1,5),(2,6),(3,7)]
                    for (edgeIndex, edge) in edges.enumerated() {
                        let start = points[edge.0]
                        let end = points[edge.1]
                        if edgeIndex == 0 { path.move(to: start) } else { path.move(to: start) }
                        path.addLine(to: end)
                    }
                    let bounds = points.reduce(into: CGRect.null) { partial, point in
                        partial = partial.union(CGRect(origin: point, size: .zero).insetBy(dx: -14, dy: -14))
                    }
                    nextRects[piece.id] = bounds
                }
                layer.path = path.cgPath
                layer.strokeColor = (piece.fits ? UIColor.systemGreen : UIColor.systemRed).cgColor
                layer.lineWidth = (selectedFurnitureID == piece.id) ? 3 : 2
                layer.isHidden = points.count != 8
            }

            for (id, layer) in furnitureShapeLayers where !usedIDs.contains(id) {
                layer.removeFromSuperlayer()
                furnitureShapeLayers.removeValue(forKey: id)
            }
            projectedFurnitureHitRects = nextRects
        }

        private func projectedBoxPoints(for piece: SharpRoomPlacedFurniture) -> [CGPoint] {
            let half = piece.item.dimensions * 0.5
            let localPoints = [
                SIMD3<Float>(-half.x, -half.y, -half.z),
                SIMD3<Float>( half.x, -half.y, -half.z),
                SIMD3<Float>(-half.x, -half.y,  half.z),
                SIMD3<Float>( half.x, -half.y,  half.z),
                SIMD3<Float>(-half.x,  half.y, -half.z),
                SIMD3<Float>( half.x,  half.y, -half.z),
                SIMD3<Float>(-half.x,  half.y,  half.z),
                SIMD3<Float>( half.x,  half.y,  half.z),
            ]
            let rotation = matrix4x4Rotation(radians: piece.rotationY, axis: SIMD3<Float>(0, 1, 0))
            var points: [CGPoint] = []
            points.reserveCapacity(8)
            for local in localPoints {
                let rotated = rotation * SIMD4<Float>(local.x, local.y, local.z, 1)
                let world = SIMD3<Float>(rotated.x, rotated.y, rotated.z) + piece.position
                guard let projected = projectWorldPoint(world) else { return [] }
                points.append(projected)
            }
            return points
        }

        private func projectWorldPoint(_ world: SIMD3<Float>) -> CGPoint? {
            let clip = lastProjectionMatrix * (lastViewMatrix * SIMD4<Float>(world.x, world.y, world.z, 1))
            guard abs(clip.w) > 1e-6 else { return nil }
            let ndc = clip / clip.w
            let x = CGFloat((ndc.x + 1) * 0.5) * viewportSize.width
            let y = CGFloat((1 - ndc.y) * 0.5) * viewportSize.height
            return CGPoint(x: x, y: y)
        }

        // MARK: Gesture Handlers

        @objc func handleTap(_ gesture: UITapGestureRecognizer) {
            let screenPoint = gesture.location(in: gesture.view)
            if let pendingFurnitureItem {
                placePendingFurniture(at: screenPoint, item: pendingFurnitureItem)
                return
            }
            selectFurniture(near: screenPoint)
        }

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

        @objc func handleRotation(_ gesture: UIRotationGestureRecognizer) {
            rotateSelectedFurniture(by: Float(gesture.rotation))
            gesture.rotation = 0
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

        private func ensureDepthScratchTexture(device: MTLDevice, width: Int, height: Int) -> MTLTexture? {
            if let existing = depthScratchTexture,
               existing.width == width,
               existing.height == height {
                return existing
            }
            let desc = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: .depth32Float,
                width:       width,
                height:      height,
                mipmapped:   false
            )
            desc.usage       = .renderTarget
            desc.storageMode = .private
            depthScratchTexture = device.makeTexture(descriptor: desc)
            return depthScratchTexture
        }

        // MARK: Notification Registration

        func registerSharpRoomParityNotifications() {
            let nc = NotificationCenter.default

            // Recenter: restore default camera orientation
            let recenterToken = nc.addObserver(
                forName: NSNotification.Name("RecenterWebGLCamera"),
                object:  nil,
                queue:   nil
            ) { [weak self] _ in
                DispatchQueue.main.async { [weak self] in
                    self?.performSharpRoomRecenter()
                }
            }
            notificationTokens.append(recenterToken)

            // Joystick pan: forward/back and left/right strafe
            let joystickToken = nc.addObserver(
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
            notificationTokens.append(joystickToken)

            // Scale: supports both (scaleX, scaleY) dict and single (factor) value
            let scaleToken = nc.addObserver(
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
            notificationTokens.append(scaleToken)

            // Vertical pan
            let moveUpToken = nc.addObserver(
                forName: NSNotification.Name("WebGLCameraMoveUp"),
                object:  nil,
                queue:   nil
            ) { [weak self] _ in
                DispatchQueue.main.async { [weak self] in
                    self?.cameraOffset.y += 0.15
                    self?.view?.setNeedsDisplay()
                }
            }
            notificationTokens.append(moveUpToken)

            let moveDownToken = nc.addObserver(
                forName: NSNotification.Name("WebGLCameraMoveDown"),
                object:  nil,
                queue:   nil
            ) { [weak self] _ in
                DispatchQueue.main.async { [weak self] in
                    self?.cameraOffset.y -= 0.15
                    self?.view?.setNeedsDisplay()
                }
            }
            notificationTokens.append(moveDownToken)

            // Horizontal pan
            let moveLeftToken = nc.addObserver(
                forName: NSNotification.Name("WebGLCameraMoveLeft"),
                object:  nil,
                queue:   nil
            ) { [weak self] _ in
                DispatchQueue.main.async { [weak self] in
                    self?.cameraOffset.x -= 0.2
                    self?.view?.setNeedsDisplay()
                }
            }
            notificationTokens.append(moveLeftToken)

            let moveRightToken = nc.addObserver(
                forName: NSNotification.Name("WebGLCameraMoveRight"),
                object:  nil,
                queue:   nil
            ) { [weak self] _ in
                DispatchQueue.main.async { [weak self] in
                    self?.cameraOffset.x += 0.2
                    self?.view?.setNeedsDisplay()
                }
            }
            notificationTokens.append(moveRightToken)

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
        arReferenceOrientation: .landscape,
        treatAsClassicPly: false,
        onBoundsAvailable: nil,
        measurementHost: nil
    )
}
