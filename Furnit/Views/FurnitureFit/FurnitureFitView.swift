// FurnitureFitView.swift
// Single-stage: detect → optional multi-candidate (5a–5b) → union mask → clip outside primary bbox (15d) → cutout
// With timing at every stage

import SwiftUI
import UIKit
import CoreML
import Accelerate
import AVFoundation
import ARKit
import CoreMotion
import SceneKit
import CoreText
import MetalKit
import simd

// BLAS helpers using C wrapper (BLASWrapper.m) to avoid Swift deprecation warnings
fileprivate typealias BLASInt = Int32

@inline(__always)
fileprivate func blas_scopy(_ n: BLASInt, _ x: UnsafePointer<Float>, _ incx: BLASInt, _ y: UnsafeMutablePointer<Float>, _ incy: BLASInt) {
    BlasScopy(n, x, incx, y, incy)
}

// MARK: - SwiftUI Wrapper
struct FurnitureFitViewSwiftUI: UIViewRepresentable {
    let mlModel: MLModel?
    var processInterval: TimeInterval = 0.07
    var confidenceThreshold: Float = 0.15
    var useBilinearUpscaling: Bool = true
    var active: Bool = false
    
    @ObservedObject private var appState = AppStateManager.shared

    func makeUIView(context: Context) -> FurnitureFitContainerView {
        let v = FurnitureFitContainerView()
        v.processInterval = processInterval
        v.confidenceThreshold = confidenceThreshold
        v.useBilinearUpscaling = useBilinearUpscaling
        v.setModel(mlModel)
        if active { v.startIfNeeded() }
        return v
    }

    func updateUIView(_ uiView: FurnitureFitContainerView, context: Context) {
        uiView.setModel(mlModel)
        uiView.processInterval = processInterval
        uiView.confidenceThreshold = confidenceThreshold
        uiView.useBilinearUpscaling = useBilinearUpscaling
        if active { uiView.startIfNeeded() } else { uiView.stop() }
    }

    static func dismantleUIView(_ uiView: FurnitureFitContainerView, coordinator: ()) {
        uiView.stop()
    }
}

// MARK: - Detection & Sizing Types
// Note: Using FurnitureFitDetection from FurnitureFitUtils.swift (single source of truth)
// typealias kept for minimal code changes
typealias UnionDet = FurnitureFitDetection

/// High-level furniture size estimate surfaced to SharpRoom / viewers.
struct FurnitureSizeEstimate {
    /// Horizontal extent in meters (pinhole + depth at bbox center when available; else legacy bbox×room).
    let widthMeters: Float
    /// Vertical extent in meters. Height-sensitive consumers should prefer ``arHeightMeters``
    /// and treat non-AR height as a non-canonical fallback only.
    let heightMeters: Float
    /// Canonical AR-based furniture height when available.
    let arHeightMeters: Float?
}

// MARK: - Main Container View
final class FurnitureFitContainerView: UIView, AVCaptureVideoDataOutputSampleBufferDelegate, ARSessionDelegate, UIGestureRecognizerDelegate {

    // MARK: Config
    var processInterval: TimeInterval = 0.07
    var confidenceThreshold: Float = 0.1
    /// After bbox clip, optional 3×3 morphological close on prototype mask before upscale; uses mask-texture composite path (not fused) when enabled.
    private let furnitureFitUseMorphologicalCloseMask: Bool = true
    var useBilinearUpscaling: Bool = true
    var lockedOrientation: PhotoOrientation = .portrait  // Locked orientation (no rotation needed when .landscape)

    // MARK: Room Dimensions (from SHARP output, in meters)
    var roomWidthMeters: Float = 4.0   // Default fallback
    var roomHeightMeters: Float = 3.0  // Default fallback
    /// Front-to-back span (m) for monocular depth fallback when AR depth is missing.
    var roomDepthMeters: Float = 4.0

    /// Optional splat depth-raycast room extents in **scene units** (same as saved `roomScene*` in `.meta`). Enables ratio fitment logs.
    var roomRaycastSceneDimensions: RoomRaycastDimensions?
    /// Canonical room intelligence model when available. Used to prefer persisted calibration and camera hints over heuristic sizing.
    var roomModel: RoomModel?
    /// Pinhole focal length in pixels; `0` uses back camera horizontal FOV when available.
    var cameraFocalLengthPixels: Float = 0

    private var captureBackCamera: AVCaptureDevice?
    private var lastSceneFitmentLogAt: CFAbsoluteTime = 0
    private var lastPinholeDepthAuditAt: CFAbsoluteTime = 0
    private let deviceMotionManager = CMMotionManager()
    private let deviceMotionQueue: OperationQueue = {
        let queue = OperationQueue()
        queue.name = "FurnitureFitDeviceMotionQueue"
        queue.qualityOfService = .userInitiated
        queue.maxConcurrentOperationCount = 1
        return queue
    }()
    private let devicePitchLock = NSLock()
    private var latestDevicePitchRadians: Float?

    /// Pinhole (and optional LiDAR snapshot) W×H in meters: first mask via ``finishFirstDetectionIfNeeded``, then each frame — all **after** the mask image is committed to UI.
    var onFurnitureSizeEstimated: ((FurnitureSizeEstimate) -> Void)?
    /// Mean straight sRGB (0…1) over opaque-enough pixels of the composited segmentation cutout; throttled (~4 Hz).
    var onSegmentationMaskMeanColorSRGB: ((SIMD3<Float>) -> Void)?

    private var lastSegmentationMeanColorPublishAt: CFAbsoluteTime = 0
    private let segmentationMeanColorMinPublishInterval: CFTimeInterval = 0.25

    // Sizing calculator (created when room dimensions are set)
    private var sizingCalculator: FurnitureSizingCalculator?

    // Debug mode - read from settings
    var debugMode: Bool {
        return AppStateManager.shared.qualitySettings.debugMode
    }

    // MARK: - Ignored Classes (loaded from blacklist.json)
    private var clsToIgnore: Set<Int> = []
    private var blacklistLoaded: Bool = false

    private func loadBlacklist() {
        guard let url = Bundle.main.url(forResource: "blacklist", withExtension: "json"),
              let data = try? Data(contentsOf: url),
              let dict = try? JSONSerialization.jsonObject(with: data) as? [String: String] else {
            if debugMode { logDebug("⚠️ Failed to load blacklist.json") }
            clsToIgnore = []
            return
        }
        clsToIgnore = Set(dict.keys.compactMap { Int($0) })
        if debugMode { logDebug("✅ Loaded \(clsToIgnore.count) blacklisted classes") }
    }

    private func loadBlacklistOnce() {
        if blacklistLoaded { return }
        loadBlacklist()
        blacklistLoaded = true
    }

    // MARK: Camera
    private let captureSession = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let sampleQueue = DispatchQueue(label: "com.furnit.sample", qos: .userInitiated)
    private let captureSessionControlQueue = DispatchQueue(label: "com.furnit.capture.control", qos: .userInitiated)

    // MARK: AR camera path (optional — when AR-assisted sizing enabled and ARKit supported)
    private let arSession = ARSession()
    private let arSCNView: ARSCNView = {
        let v = ARSCNView(frame: .zero)
        v.automaticallyUpdatesLighting = false
        if #available(iOS 15.0, *) {
            v.rendersCameraGrain = false
        }
        v.isHidden = true
        v.alpha = 0
        v.translatesAutoresizingMaskIntoConstraints = false
        return v
    }()
    /// Used only if `isUsingARCameraPath` (ARKit as camera); hybrid path uses AVCapture BGRA and skips CI YUV→BGRA.
    private var arBGRAReuse: CVPixelBuffer?
    private let arCIContext = CIContext(options: [.cacheIntermediates: false])
    /// View size for AR-assisted camera path (set in `layoutSubviews`).
    private var cachedARViewportSize: CGSize = .zero
    /// True only when ARKit supplies segmentation frames (`session(_:didUpdate:)` + `copyCapturedImageToBGRA`). Off for hybrid path.
    private var isUsingARCameraPath = false
    /// True when ARSession runs alongside AVCapture for depth / tracking while video frames come from `AVCaptureSession` (recommended).
    private var isARDepthCompanionSessionRunning = false

    /// True only when ARKit is feeding sizing (AR-as-camera or hybrid companion). Do **not** use plain world-tracking support here — that
    /// incorrectly made `startIfNeeded()` return before starting `AVCaptureSession`.
    private var hasARKitAssistedSizingPayload: Bool {
        isUsingARCameraPath || isARDepthCompanionSessionRunning
    }

    /// Not started while `AVCaptureSession` runs — a second `ARSession` still grabs the back camera (Fig -17281).
    private let asyncDepthSampler = FurnitureFitAsyncDepthSampler()
    /// Bumps on each `startClassicCameraPathIfNeeded` so a stale deferred AR `run` cannot fire after reconfigure.
    private var arCompanionStartGeneration: UInt64 = 0
    /// Set when AVCapture reports Fig-style contention so we drop AR for the rest of this container session (until `stop()`).
    private var suppressARDepthCompanionAfterCaptureFailure = false
    private var furnitureFitCameraStartupInitiated = false
    /// Log `[FurnitureFitSize]` policy once so Console filtering explains `wantAR=false` / nil depth snapshot.
    private static var didLogFurnitureFitSizingPolicy = false
    private var arAssistedScaleValid = false
    private var autoScaleFromAR: CGFloat = 1.0
    private var smoothedArOverlayScale: CGFloat = 1.0
    /// After primary class changes, apply first valid AR scale immediately instead of long EMA ramp (faster when panning between furniture).
    private var snapArOverlayScaleAfterPrimaryChange = false
    /// When depth/estimation drops for a few frames, keep last AR scale instead of snapping to 1× (avoids small/big flicker).
    private var arAssistedConsecutiveMisses: Int = 0
    private let maxArAssistedHoldMisses: Int = 18
    /// Latest AR-estimated furniture height in meters (UI only; does not affect segmentation).
    private var lastAREstimatedHeightMeters: Float?
    /// After first segmentation, publish one **refined** W×H (AR intrinsics + min depth in bbox) to parents — fixes huge first-frame numbers when YOLO boxes are loose.
    private var didPublishARRefinedFurnitureMetersEstimate = false

    // MARK: UI
    private let previewLayer = AVCaptureVideoPreviewLayer()
    private let maskImageView: UIImageView = {
        let iv = UIImageView()
        iv.contentMode = .scaleAspectFill  // Match camera preview's resizeAspectFill
        iv.backgroundColor = .clear
        iv.isOpaque = false
        iv.clipsToBounds = true
        iv.layer.minificationFilter = .linear
        iv.layer.magnificationFilter = .linear
        return iv
    }()
    
    // MARK: Progress UI
    private let progressContainer: UIView = {
        let v = UIView()
        v.translatesAutoresizingMaskIntoConstraints = false
        v.isHidden = true
        return v
    }()

    private let progressView: UIProgressView = {
        let pv = UIProgressView(progressViewStyle: .default)
        pv.translatesAutoresizingMaskIntoConstraints = false
        pv.tintColor = .systemGreen
        pv.trackTintColor = UIColor(white: 1.0, alpha: 0.3)
        return pv
    }()

    private let progressLabel: UILabel = {
        let l = UILabel()
        l.translatesAutoresizingMaskIntoConstraints = false
        l.textColor = .white
        l.font = .systemFont(ofSize: 14, weight: .medium)
        l.textAlignment = .center
        l.numberOfLines = 2
        l.adjustsFontSizeToFitWidth = true
        l.minimumScaleFactor = 0.72
        l.backgroundColor = UIColor.black.withAlphaComponent(0.4)
        l.layer.cornerRadius = 10
        l.clipsToBounds = true
        return l
    }()
    
    private var hasFirstDetection = false
    /// After the first successful segmentation in this container, skip startup progress on later `startIfNeeded()` (same UIView instance).
    private var segmentationCompletedOnceThisSession = false
    /// When true (e.g. Sharp Room already completed segmentation this session), hide startup progress even for a new UIView.
    var suppressStartupProgress = false
    /// Called once when the first valid segmentation mask is ready (parent can persist “no progress next time”).
    var onFirstSegmentationComplete: (() -> Void)?

    // MARK: - Overlay scale (room × AR when enabled × user pinch)
    /// From `FurnitureSizingCalculator` (currently neutral 1×; room meters come from SHARP / calibration).
    private var autoScaleFromRoom: CGFloat = 1.0
    /// User pinch multiplier (reset when primary class changes).
    private var userPinchScale: CGFloat = 1.0
    /// After user pinches, stop updating AR-assisted auto-scale until primary class changes.
    private var userLockedAssistedOverlayScale: Bool = false
    /// Last primary class used for overlay / pinch reset.
    private var lastOverlayPrimaryClassIdx: Int = -1

    private let minCombinedOverlayScale: CGFloat = 0.08
    private let maxCombinedOverlayScale: CGFloat = 3.0
    private enum OverlayPresentationMode {
        case deferredCentered
        case measuredPlacement
    }
    private var overlayPresentationMode: OverlayPresentationMode = .deferredCentered
    private var stableOverlayMeasurementFrameCount: Int = 0
    private var lastStableOverlayHeightMeters: Float?
    private var lastStableOverlayScale: CGFloat?
    private let requiredStableOverlayMeasurementFrames: Int = 3
    private let maxStableOverlayHeightDriftFraction: Float = 0.18
    private let maxStableOverlayScaleDrift: CGFloat = 0.18
    private let arOverlayResyncJumpFraction: CGFloat = 0.22
    private let arOverlayResyncHeightDriftFraction: Float = 0.22
    /// Primary scoring: penalize YOLO boxes that fill most of the model frame (desk / table / wrapping paper).
    /// Those steal `selectPrimary` from cup/glass and make AR overlay scale swing wildly when the camera moves.
    private let primarySelectionSoftAreaNormPivot: Float = 0.30
    private let primarySelectionLargeBoxPenaltyK: Float = 5.2

    /// Throttled logging for oscillation diagnosis (when `debugMode` is on).
    private var overlayDebugLastAssistedLabel: String = ""
    private var overlayDebugLastCombined: CGFloat = -1

    /// Throttled always-on AR metrics for cross-platform comparison (filter: `FurnitureFitAR`).
    private var lastFurnitureFitARFrameLogAt: TimeInterval = 0
    private let furnitureFitARLogInterval: TimeInterval = 0.5

    /// Throttled premultiplied RGBA sample at bbox center (filter: `FurnitureFit` + debugMode).
    private var lastCompositePremulRGBAlogAt: TimeInterval = 0
    private let compositePremulRGBAlogInterval: TimeInterval = 0.5

    /// Primary furniture bounding box in view coordinates (for gesture hit testing)
    private var primaryBboxInView: CGRect = .zero
    /// Smoothed **tight** primary bbox in image coords (model box, no margin). Used for mask clip so multi-candidate cannot spill outside the detection.
    private var smoothedTightPrimaryBbox: (x1: Float, y1: Float, x2: Float, y2: Float)?
    private let bboxSmoothingAlpha: Float = 0.35  // 0 = no smoothing, 1 = no memory
    private let bboxExpandMargin: Float = 0.08     // 8% expansion on each side

    /// STAGE 5a–5b: prune / bbox dedupe for extra furniture. Off = primary detection only in `kept2`.
    private var useMultiCandidateStage5 = true

    // MARK: - Metal (FIXED: stored properties instead of computed to prevent resource leak)
    private var metalDevice: MTLDevice?
    private var metalCommandQueue: MTLCommandQueue?  // FIXED: was computed property creating new queue on every access
    private var metalLibrary: MTLLibrary?            // FIXED: was computed property creating new library on every access
    private var compositePipeline: MTLComputePipelineState?
    private var fusedMaskCompositePipeline: MTLComputePipelineState?
    /// Fused compositing using Swift morphed prototype mask (same geometry as `sp_maxMaskAndComposite`, no logit loop).
    private var fusedMaskMorphedCompositePipeline: MTLComputePipelineState?

    // MARK: - Cached Metal buffers for fused compositing (prevents allocation per frame)
    private var cachedFusedPlanesBuf: MTLBuffer?
    private var cachedFusedCoeffBuf: MTLBuffer?
    private var cachedFusedPlanesCapacity: Int = 0
    private var cachedFusedCoeffCapacity: Int = 0
    private var cachedFusedProtoMaskBuf: MTLBuffer?
    private var cachedFusedProtoMaskCapacity: Int = 0

    // MARK: - CVMetalTextureCache (FIXED: was created per-frame causing memory leak)
    private var cvTextureCache: CVMetalTextureCache?

    // MARK: - Reusable prototype buffers (FIXED: prevents ~26MB allocation per frame)
    private var protoRawFloats: [Float] = []
    private var protoPlanes: [Float] = []

    // MARK: - Reusable CVPixelBuffer & MLMultiArray (prevents allocation per frame)
    /// Stretch path: full frame scaled to the model square (matches Android / ONNX-style preprocessing).
    private var cachedStretchBuffer: CVPixelBuffer?
    private var cachedStretchSize: Int = 0
    private var cachedMLArray: MLMultiArray?
    private var cachedMLArraySize: Int = 0

    /// Reused scratch for CPU fallback compositing (vDSP + BLAS); grows with frame width, never shrinks.
    private var compositeCpuScratchFloats: [Float] = []
    /// Proto mask upscaled to the active composite band via vImage; grows to the largest band seen and never shrinks.
    private var upscaledPlanarMaskScratch: [UInt8] = []
    // MARK: - Memory Logging
    private func logMemory(_ tag: String) {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        if result == KERN_SUCCESS {
            let usedMB = Double(info.resident_size) / 1024.0 / 1024.0
            logDebug("🧠 [\(tag)] Memory: \(String(format: "%.1f", usedMB)) MB")
        }
    }

    // MARK: Model & State
    private var mlModel: MLModel?  // yoloe-26l-seg-pf_seg_o2m (640 one-to-many) via YOLOEModelService
    private let detectionQueue = DispatchQueue(label: "com.furnit.detection", qos: .userInitiated)
    private var lastProcessTime = Date.distantPast
    private var isProcessing = false
    /// While inference is running, camera frames are dropped; when set, next completion clears `lastProcessTime` so the following frame is not delayed by `processInterval`.
    private var preferImmediateNextInference = false
    private let frameLock = NSLock() // Protects lastProcessTime, isProcessing, preferImmediateNextInference

    /// Serial queue for AR-assisted measurement (depth → height → overlay scale). This is
    /// coalesced rather than fully debounced so continuous segmentation frames do not starve
    /// AR sizing forever.
    private let measurementQueue = DispatchQueue(label: "com.furnit.furniturefit.measurement", qos: .utility)
    private let assistedMeasurementDebounceSeconds: TimeInterval = 0.22
    private struct PendingAssistedMeasurement {
        let primaryClassIdx: Int
        let bboxMinX: Int
        let bboxMinY: Int
        let bboxMaxX: Int
        let bboxMaxY: Int
        let bboxCenterImageX: CGFloat
        let bboxCenterImageY: CGFloat
        let bboxHeightImagePx: Float
        let imageWidth: Int
        let imageHeight: Int
        let arDepthSnapshot: FurnitureFitARDepthSnapshot?
    }
    private let pendingMeasurementLock = NSLock()
    private var pendingAssistedMeasurement: PendingAssistedMeasurement?
    private var assistedMeasurementDebounceWorkItem: DispatchWorkItem?
    /// Incremented on each new schedule and on invalidate so superseded debounce work is dropped.
    private var assistedMeasurementScheduleToken: UInt64 = 0

    // MARK: Orientation
    private var currentDeviceOrientation: UIDeviceOrientation = .portrait

    /// Thread-safe reset of isProcessing flag.
    private func resetProcessingFlag() {
        frameLock.lock()
        isProcessing = false
        if preferImmediateNextInference {
            lastProcessTime = .distantPast
            preferImmediateNextInference = false
        }
        frameLock.unlock()
    }

    private func invalidatePendingAssistedMeasurement() {
        assistedMeasurementDebounceWorkItem?.cancel()
        assistedMeasurementDebounceWorkItem = nil
        assistedMeasurementScheduleToken &+= 1
        pendingMeasurementLock.lock()
        pendingAssistedMeasurement = nil
        pendingMeasurementLock.unlock()
    }

    private func scheduleDebouncedAssistedMeasurement(
        primaryClassIdx: Int,
        bboxMinX: Int,
        bboxMinY: Int,
        bboxMaxX: Int,
        bboxMaxY: Int,
        bboxCenterImageX: CGFloat,
        bboxCenterImageY: CGFloat,
        bboxHeightImagePx: Float,
        imageWidth: Int,
        imageHeight: Int,
        arDepthSnapshot: FurnitureFitARDepthSnapshot?
    ) {
        pendingMeasurementLock.lock()
        pendingAssistedMeasurement = PendingAssistedMeasurement(
            primaryClassIdx: primaryClassIdx,
            bboxMinX: bboxMinX,
            bboxMinY: bboxMinY,
            bboxMaxX: bboxMaxX,
            bboxMaxY: bboxMaxY,
            bboxCenterImageX: bboxCenterImageX,
            bboxCenterImageY: bboxCenterImageY,
            bboxHeightImagePx: bboxHeightImagePx,
            imageWidth: imageWidth,
            imageHeight: imageHeight,
            arDepthSnapshot: arDepthSnapshot
        )
        pendingMeasurementLock.unlock()
        assistedMeasurementScheduleToken &+= 1
        let workToken = assistedMeasurementScheduleToken
        assistedMeasurementDebounceWorkItem?.cancel()
        let work = DispatchWorkItem { [weak self] in
            guard let self else { return }
            guard workToken == self.assistedMeasurementScheduleToken else { return }
            self.pendingMeasurementLock.lock()
            let p = self.pendingAssistedMeasurement
            self.pendingMeasurementLock.unlock()
            guard let p else { return }
            DispatchQueue.main.async {
                self.updateAssistedOverlayScale(
                    primaryClassIdx: p.primaryClassIdx,
                    bboxMinX: p.bboxMinX,
                    bboxMinY: p.bboxMinY,
                    bboxMaxX: p.bboxMaxX,
                    bboxMaxY: p.bboxMaxY,
                    bboxCenterImageX: p.bboxCenterImageX,
                    bboxCenterImageY: p.bboxCenterImageY,
                    bboxHeightImagePx: p.bboxHeightImagePx,
                    imageWidth: p.imageWidth,
                    imageHeight: p.imageHeight,
                    arDepthSnapshot: p.arDepthSnapshot
                )
            }
        }
        assistedMeasurementDebounceWorkItem = work
        measurementQueue.asyncAfter(deadline: .now() + assistedMeasurementDebounceSeconds, execute: work)
    }

    // MARK: Class Names (loaded from classes.json)
    internal lazy var classNames: [Int: String] = {
        guard let url = Bundle.main.url(forResource: "classes", withExtension: "json"),
              let data = try? Data(contentsOf: url),
              let dict = try? JSONSerialization.jsonObject(with: data) as? [String: String] else {
            if debugMode { logDebug("⚠️ Failed to load classes.json") }
            return [:]
        }
        var result: [Int: String] = [:]
        for (key, value) in dict {
            if let id = Int(key) {
                result[id] = value
            }
        }
        if debugMode { logDebug("✅ Loaded \(result.count) class names") }
        return result
    }()
    
    private func className(_ id: Int) -> String {
        let name = classNames[id] ?? "unknown"
        return "\u{001B}[1m::\(name) (id:\(id))\u{001B}[0m"
    }

    private func supportSurfaceHint(
        from candidates: [FurnitureFitDetection],
        imageWidth: Int,
        imageHeight: Int
    ) -> String? {
        guard imageWidth > 0, imageHeight > 0 else { return nil }

        let keywords = [
            "floor", "flooring", "ground", "carpet", "rug", "mat",
            "tile", "tiles", "pavement", "sidewalk", "concrete"
        ]
        let frameArea = Float(max(1, imageWidth * imageHeight))
        let lowerBandStart = Float(imageHeight) * 0.58

        for detection in candidates {
            let rawName = (classNames[detection.classIdx] ?? "unknown").lowercased()
            guard keywords.contains(where: { rawName.contains($0) }) else { continue }

            let areaNorm = (detection.w * detection.h) / frameArea
            let bottomY = detection.y + detection.h * 0.5
            let spansLowerBand = bottomY >= lowerBandStart
            let isLargeEnough = areaNorm >= 0.05
            guard spansLowerBand || isLargeEnough else { continue }

            logFurnitureFitSize(
                "phase=support_surface_hint matched=\(rawName) conf=\(String(format: "%.2f", detection.confidence)) " +
                "areaNorm=\(String(format: "%.3f", areaNorm)) bottomFrac=\(String(format: "%.3f", bottomY / Float(imageHeight)))"
            )
            return rawName
        }

        return nil
    }

    // MARK: - Init
    override init(frame: CGRect) {
        super.init(frame: frame)
        commonInit()
    }
    
    required init?(coder: NSCoder) {
        super.init(coder: coder)
        commonInit()
    }
    
    private func commonInit() {
        backgroundColor = .clear
        isUserInteractionEnabled = true
        
        previewLayer.session = captureSession
        previewLayer.videoGravity = .resizeAspectFill
        previewLayer.isHidden = true
        layer.addSublayer(previewLayer)
        
        maskImageView.isUserInteractionEnabled = true
        addSubview(arSCNView)
        addSubview(maskImageView)
        arSCNView.session = arSession
        maskImageView.translatesAutoresizingMaskIntoConstraints = false
        NSLayoutConstraint.activate([
            arSCNView.topAnchor.constraint(equalTo: topAnchor),
            arSCNView.bottomAnchor.constraint(equalTo: bottomAnchor),
            arSCNView.leadingAnchor.constraint(equalTo: leadingAnchor),
            arSCNView.trailingAnchor.constraint(equalTo: trailingAnchor),
            maskImageView.topAnchor.constraint(equalTo: topAnchor),
            maskImageView.bottomAnchor.constraint(equalTo: bottomAnchor),
            maskImageView.leadingAnchor.constraint(equalTo: leadingAnchor),
            maskImageView.trailingAnchor.constraint(equalTo: trailingAnchor)
        ])
        
        // Progress container holds both progress bar and label, rotates with device orientation
        addSubview(progressContainer)
        progressContainer.addSubview(progressView)
        progressContainer.addSubview(progressLabel)
        NSLayoutConstraint.activate([
            // Container centered horizontally, near top
            progressContainer.centerXAnchor.constraint(equalTo: centerXAnchor),
            progressContainer.topAnchor.constraint(equalTo: safeAreaLayoutGuide.topAnchor, constant: 40),
            progressContainer.widthAnchor.constraint(equalToConstant: 300),
            progressContainer.heightAnchor.constraint(equalToConstant: 62),

            // Progress bar inside container
            progressView.leadingAnchor.constraint(equalTo: progressContainer.leadingAnchor),
            progressView.trailingAnchor.constraint(equalTo: progressContainer.trailingAnchor),
            progressView.bottomAnchor.constraint(equalTo: progressContainer.bottomAnchor),

            // Label above progress bar
            progressLabel.centerXAnchor.constraint(equalTo: progressContainer.centerXAnchor),
            progressLabel.bottomAnchor.constraint(equalTo: progressView.topAnchor, constant: -6),
            progressLabel.heightAnchor.constraint(greaterThanOrEqualToConstant: 36),
            progressLabel.widthAnchor.constraint(equalTo: progressContainer.widthAnchor)
        ])
        
        let pinchGesture = UIPinchGestureRecognizer(target: self, action: #selector(handlePinch(_:)))
        pinchGesture.delegate = self
        pinchGesture.cancelsTouchesInView = false
        
        let panGesture = UIPanGestureRecognizer(target: self, action: #selector(handlePan(_:)))
        panGesture.delegate = self
        panGesture.cancelsTouchesInView = false
        panGesture.minimumNumberOfTouches = 1
        panGesture.maximumNumberOfTouches = 1
        
        addGestureRecognizer(pinchGesture)
        maskImageView.addGestureRecognizer(panGesture)

        // Listen for reset notification from SharpRoomView toolbar ("reset size" button).
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handleResetScaleTapped),
            name: NSNotification.Name("FurnitureFitResetOverlayScale"),
            object: nil
        )

        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handleCaptureSessionRuntimeError(_:)),
            name: AVCaptureSession.runtimeErrorNotification,
            object: captureSession
        )
        arSession.delegate = self
        
        setupCamera()
        setupMetal()
        if debugMode { logDebug("✅ FurnitureFitContainerView initialized") }
    }
    
    private func setupMetal() {
        // FIXED: Initialize Metal resources once as stored properties (not computed)
        // This prevents the OOM crash caused by creating new command queues per frame
        if metalDevice == nil {
            metalDevice = MTLCreateSystemDefaultDevice()
        }
        guard let device = metalDevice else {
            if debugMode { logDebug("❌ Metal not supported on this device") }
            return
        }

        // Create command queue ONCE and store it
        if metalCommandQueue == nil {
            metalCommandQueue = device.makeCommandQueue()
        }

        // Create library ONCE and store it
        if metalLibrary == nil {
            metalLibrary = device.makeDefaultLibrary()
        }

        guard let library = metalLibrary else {
            if debugMode { logDebug("❌ Failed to create Metal library") }
            return
        }

        // Create pipelines only once
        do {
            if compositePipeline == nil, let fn = library.makeFunction(name: "sp_compositeMask") {
                compositePipeline = try device.makeComputePipelineState(function: fn)
            }
            if fusedMaskCompositePipeline == nil, let fn2 = library.makeFunction(name: "sp_maxMaskAndComposite") {
                fusedMaskCompositePipeline = try device.makeComputePipelineState(function: fn2)
            }
            if fusedMaskMorphedCompositePipeline == nil, let fn3 = library.makeFunction(name: "sp_maxMaskAndCompositeMorphed") {
                fusedMaskMorphedCompositePipeline = try device.makeComputePipelineState(function: fn3)
            }
        } catch {
            if debugMode { logDebug("⚠️ Metal pipeline setup failed: \(error.localizedDescription)") }
            CrashReporter.shared.report(error, context: "Metal Pipeline Setup")
        }

        // FIXED: Create texture cache ONCE (was being created per-frame causing memory leak)
        if cvTextureCache == nil {
            CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, device, nil, &cvTextureCache)
        }
    }
    
    override func layoutSubviews() {
        super.layoutSubviews()
        previewLayer.frame = bounds
        cachedARViewportSize = bounds.size
        updateSizingCalculator()
    }

    /// Update sizing calculator with current view dimensions
    private func updateSizingCalculator() {
        guard bounds.width > 0, bounds.height > 0 else { return }
        sizingCalculator = FurnitureSizingCalculator(
            roomWidth: roomWidthMeters,
            roomHeight: roomHeightMeters,
            viewWidth: bounds.width,
            viewHeight: bounds.height
        )
    }

    /// Non-AR room scaling is disabled. Furniture measurement and overlay scale are AR-only.
    private func updateAutoScaleFromRoom(
        primaryBx1: Int,
        primaryBy1: Int,
        primaryBx2: Int,
        primaryBy2: Int,
        imageWidth: Int,
        imageHeight: Int,
        arDepthSnapshot: FurnitureFitARDepthSnapshot?,
        preferRoomRaycastSizing: Bool
    ) {
        _ = primaryBx1
        _ = primaryBy1
        _ = primaryBx2
        _ = primaryBy2
        _ = imageWidth
        _ = imageHeight
        _ = arDepthSnapshot
        _ = preferRoomRaycastSizing
        autoScaleFromRoom = 1.0
        applyCurrentOverlayScaleTransform()
    }

    private func updateOverlayPresentationMode(
        primaryClassIdx: Int,
        metric: PrimaryBboxMetersResult?
    ) {
        _ = metric
        if primaryClassIdx != lastOverlayPrimaryClassIdx {
            overlayPresentationMode = .deferredCentered
            stableOverlayMeasurementFrameCount = 0
            lastStableOverlayHeightMeters = nil
            lastStableOverlayScale = nil
        }

        let arSizingReady =
            hasARKitAssistedSizingPayload &&
            arAssistedScaleValid &&
            normalizedARFurnitureHeightMeters() != nil &&
            autoScaleFromAR.isFinite &&
            autoScaleFromAR > 0

        if arSizingReady, let arHeight = normalizedARFurnitureHeightMeters() {
            let scaleDrift = lastStableOverlayScale.map { abs(autoScaleFromAR - $0) } ?? 0
            let heightDriftFraction: Float
            if let lastHeight = lastStableOverlayHeightMeters, lastHeight > 0 {
                heightDriftFraction = abs(arHeight - lastHeight) / lastHeight
            } else {
                heightDriftFraction = 0
            }

            let isStable = scaleDrift <= maxStableOverlayScaleDrift &&
                heightDriftFraction <= maxStableOverlayHeightDriftFraction

            stableOverlayMeasurementFrameCount = isStable ? (stableOverlayMeasurementFrameCount + 1) : 1
            lastStableOverlayHeightMeters = arHeight
            lastStableOverlayScale = autoScaleFromAR

            if stableOverlayMeasurementFrameCount >= requiredStableOverlayMeasurementFrames {
                overlayPresentationMode = .measuredPlacement
            } else {
                overlayPresentationMode = .deferredCentered
            }
            return
        }

        // No usable AR metric (AVCapture-only, non-LiDAR, or depth/planes never validate): overlay stays 1× AR factor
        // but pinch/pan must still work. `handlePinch` / `handlePan` are gated on `.measuredPlacement` only.
        stableOverlayMeasurementFrameCount += 1
        if stableOverlayMeasurementFrameCount >= requiredStableOverlayMeasurementFrames {
            overlayPresentationMode = .measuredPlacement
        } else {
            overlayPresentationMode = .deferredCentered
        }
    }

    /// Combined overlay: AR metric scale × pinch. Non-AR room scaling is disabled.
    private func applyCurrentOverlayScaleTransform() {
        let arOn = hasARKitAssistedSizingPayload && arAssistedScaleValid
        let roomFactor: CGFloat = 1.0
        let assistedScale: CGFloat = arOn ? autoScaleFromAR : 1.0
        let product = roomFactor * assistedScale * userPinchScale
        let clamped = min(max(product, minCombinedOverlayScale), maxCombinedOverlayScale)
        let finalTransform: CGAffineTransform
        switch overlayPresentationMode {
        case .deferredCentered:
            let bboxCenter = CGPoint(x: primaryBboxInView.midX, y: primaryBboxInView.midY)
            let viewCenter = CGPoint(x: bounds.midX, y: bounds.midY)
            let translationX = primaryBboxInView.width > 0 ? (viewCenter.x - bboxCenter.x) : 0
            let translationY = primaryBboxInView.height > 0 ? (viewCenter.y - bboxCenter.y) : 0
            finalTransform = CGAffineTransform(translationX: translationX, y: translationY)
        case .measuredPlacement:
            finalTransform = CGAffineTransform(scaleX: clamped, y: clamped)
        }
        maskImageView.transform = finalTransform

        let wantAR = hasARKitAssistedSizingPayload && !userLockedAssistedOverlayScale
        let assistedLabel: String
        if arOn {
            assistedLabel = "AR"
        } else if wantAR {
            assistedLabel = "1x_AR_unavailable"
        } else {
            assistedLabel = "1x"
        }
        let jump = overlayDebugLastCombined < 0 || abs(clamped - overlayDebugLastCombined) > 0.02
        let labelChange = assistedLabel != overlayDebugLastAssistedLabel
        if jump || labelChange {
            overlayDebugLastAssistedLabel = assistedLabel
            overlayDebugLastCombined = clamped
            let modeLabel = overlayPresentationMode == .deferredCentered ? "centered_pending" : "measured"
            logFurnitureFitOverlay(
                "mode=\(modeLabel) assist=\(assistedLabel) roomStored=\(String(format: "%.3f", autoScaleFromRoom)) roomUsed=\(String(format: "%.3f", roomFactor)) ar=\(String(format: "%.3f", autoScaleFromAR)) pinch=\(String(format: "%.3f", userPinchScale)) → overlay=\(String(format: "%.3f", clamped)) wantAR=\(wantAR) arValid=\(arAssistedScaleValid)"
            )
        }
    }

    private func updateAssistedOverlayScale(
        primaryClassIdx: Int,
        bboxMinX: Int,
        bboxMinY: Int,
        bboxMaxX: Int,
        bboxMaxY: Int,
        bboxCenterImageX: CGFloat,
        bboxCenterImageY: CGFloat,
        bboxHeightImagePx: Float,
        imageWidth: Int,
        imageHeight: Int,
        arDepthSnapshot: FurnitureFitARDepthSnapshot?
    ) {
        if primaryClassIdx != lastOverlayPrimaryClassIdx {
            lastOverlayPrimaryClassIdx = primaryClassIdx
            userPinchScale = 1.0
            userLockedAssistedOverlayScale = false
            smoothedArOverlayScale = 1.0
            autoScaleFromAR = 1.0
            arAssistedScaleValid = false
            arAssistedConsecutiveMisses = 0
            snapArOverlayScaleAfterPrimaryChange = true
            overlayDebugLastAssistedLabel = ""
            overlayDebugLastCombined = -1
            didPublishARRefinedFurnitureMetersEstimate = false
            overlayPresentationMode = .deferredCentered
            stableOverlayMeasurementFrameCount = 0
            lastStableOverlayHeightMeters = nil
            lastStableOverlayScale = nil
        }

        let bboxWidthImagePx = Float(max(1, bboxMaxX - bboxMinX))

        let wantAR = hasARKitAssistedSizingPayload && !userLockedAssistedOverlayScale

        if wantAR, arDepthSnapshot != nil || arSession.currentFrame != nil {
            let nx = bboxCenterImageX / CGFloat(max(1, imageWidth))
            let ny = bboxCenterImageY / CGFloat(max(1, imageHeight))
            let nxMin = CGFloat(bboxMinX) / CGFloat(max(1, imageWidth))
            let nxMax = CGFloat(bboxMaxX) / CGFloat(max(1, imageWidth))
            let nyMinTop = CGFloat(bboxMinY) / CGFloat(max(1, imageHeight))
            let nyMaxTop = CGFloat(bboxMaxY) / CGFloat(max(1, imageHeight))

            var distM: Float?
            var fx: Float = 1
            var fy: Float = 1
            var distSource = "none"
            let trackingName: String = {
                guard let s = arSession.currentFrame?.camera.trackingState else { return "nil" }
                switch s {
                case .normal: return "normal"
                case .limited: return "limited"
                case .notAvailable: return "notAvailable"
                @unknown default: return "unknown"
                }
            }()

            if let snap = arDepthSnapshot {
                fy = snap.focalLengthY
                fx = snap.focalLengthX
                if snap.depthMap != nil,
                   let dRobust = FurnitureFitARSupport.depthMetersPercentileInNormalizedBgraRect(
                    snapshot: snap,
                    nxMin: nxMin,
                    nyMinTop: nyMinTop,
                    nxMax: nxMax,
                    nyMaxTop: nyMaxTop
                   ) {
                    distM = dRobust
                    distSource = "depth_snapshot_p20_rect"
                }
                if distM == nil {
                    distM = FurnitureFitARSupport.depthMeters(snapshot: snap, normalizedBgraNX: nx, normalizedBgraNY: ny)
                    distSource = distM != nil ? "depth_snapshot_center" : "depth_snapshot_miss"
                }
                // Snapshot exists but depth buffer missing/invalid this frame — align overlay with live `sceneDepth` (same as meter sizing fallback).
                if distM == nil, let frame = arSession.currentFrame {
                    let norm = CGPoint(x: nx, y: ny)
                    distM = FurnitureFitARSupport.sceneDepthMeters(frame: frame, normalizedImagePoint: norm)
                    if distM != nil { distSource = "scene_depth_snapshot_miss_fallback" }
                }
                // No plane raycast here: that would use `arSession.currentFrame` and desync from the YOLO frame.
            } else if let frame = arSession.currentFrame {
                let norm = CGPoint(x: nx, y: ny)
                distM = FurnitureFitARSupport.sceneDepthMeters(frame: frame, normalizedImagePoint: norm)
                if distM != nil {
                    distSource = "scene_depth"
                } else if #available(iOS 14.0, *) {
                    let sp = CGPoint(
                        x: nx * bounds.width,
                        y: ny * bounds.height
                    )
                    // Try any-plane raycast first (horizontal + vertical), fall back to horizontal only.
                    if let anyHit = FurnitureFitARSupport.distanceToAnyPlaneMeters(
                        session: arSession, frame: frame, screenPoint: sp, in: bounds
                    ) {
                        distM = anyHit.distance
                        let alignLabel: String
                        switch anyHit.alignment {
                        case .horizontal: alignLabel = "horiz"
                        case .vertical: alignLabel = "vert"
                        case .any: alignLabel = "any"
                        @unknown default: alignLabel = "unknown"
                        }
                        distSource = "plane_raycast_\(alignLabel)"
                    } else {
                        distSource = "no_metric_distance"
                    }
                } else {
                    distSource = "no_metric_distance"
                }
                let cap = frame.capturedImage
                let cw = CVPixelBufferGetWidth(cap)
                let ch = CVPixelBufferGetHeight(cap)
                let needsPortrait = (lockedOrientation == .portrait || lockedOrientation == .square)
                let bgraRotated = needsPortrait && cw > ch
                fy = FurnitureFitARSupport.focalLengthYForProcessedBGRA(
                    camera: frame.camera,
                    bgraHeight: imageHeight,
                    bgraIsRotatedFromCaptured: bgraRotated
                )
                fx = FurnitureFitARSupport.focalLengthXForProcessedBGRA(
                    camera: frame.camera,
                    bgraWidth: imageWidth,
                    bgraIsRotatedFromCaptured: bgraRotated
                )
            } else {
                distSource = "no_ar_frame"
            }

            // If we have a valid distance + focal length, update AR-assisted scale for this frame.
            if let d = distM,
               let estH = FurnitureFitARSupport.estimatedPhysicalHeightMeters(
                bboxHeightPixels: bboxHeightImagePx,
                distanceMeters: d,
                focalLengthYPixels: fy
               ) {
                let previousEstimatedHeightMeters = lastAREstimatedHeightMeters
                lastAREstimatedHeightMeters = estH
                let roomHeight = max(roomHeightMeters, 0.01)
                let currentImageHeightFraction = max(
                    CGFloat(bboxHeightImagePx) / CGFloat(max(1, imageHeight)),
                    0.01
                )
                let targetImageHeightFraction = CGFloat(
                    min(max(estH / roomHeight, 0.01), 0.98)
                )
                let raw = targetImageHeightFraction / currentImageHeightFraction
                if raw.isFinite, raw > 0 {
                    if hasFirstDetection,
                       !didPublishARRefinedFurnitureMetersEstimate,
                       fx > 1,
                       let estW = FurnitureFitARSupport.estimatedPhysicalWidthMeters(
                        bboxWidthPixels: bboxWidthImagePx,
                        distanceMeters: d,
                        focalLengthXPixels: fx
                       ) {
                        didPublishARRefinedFurnitureMetersEstimate = true
                        logFurnitureFitSize(
                            "phase=ar_overlay_refine distSource=\(distSource) dist_m=\(String(format: "%.3f", d)) W×H_m=\(String(format: "%.3f", estW))×\(String(format: "%.3f", estH)) metric_px=\(Int(bboxWidthImagePx))x\(Int(bboxHeightImagePx)) (requires AR companion + valid depth)"
                        )
                        onFurnitureSizeEstimated?(
                            FurnitureSizeEstimate(widthMeters: estW, heightMeters: estH, arHeightMeters: estH)
                        )
                    }
                    let target = min(max(raw, minCombinedOverlayScale), maxCombinedOverlayScale)
                    if snapArOverlayScaleAfterPrimaryChange {
                        // One shot after switching primary detection — avoids sluggish ramp from 1× while panning between pieces.
                        let clamped = min(max(target, minCombinedOverlayScale), maxCombinedOverlayScale)
                        smoothedArOverlayScale = clamped
                        snapArOverlayScaleAfterPrimaryChange = false
                    } else {
                        // AR-assisted scale can jitter due to depth noise; clamp per-frame change and EMA smooth.
                        // But when the measured object size materially changes, snap to the new ratio immediately
                        // instead of dragging the previous object's scale forward for several frames.
                        let base = smoothedArOverlayScale
                        let relativeJump: CGFloat = {
                            let denom = max(abs(base), abs(target), 0.001)
                            return abs(target - base) / denom
                        }()
                        let heightDriftFraction: Float = {
                            guard let previousEstimatedHeightMeters,
                                  previousEstimatedHeightMeters > 0 else { return 0 }
                            return abs(estH - previousEstimatedHeightMeters) / previousEstimatedHeightMeters
                        }()
                        if relativeJump >= arOverlayResyncJumpFraction ||
                            heightDriftFraction >= arOverlayResyncHeightDriftFraction {
                            smoothedArOverlayScale = target
                            if debugMode {
                                logDebug(
                                    "📐 [AR_RESYNC] target=\(String(format: "%.3f", target)) " +
                                    "base=\(String(format: "%.3f", base)) relJump=\(String(format: "%.3f", relativeJump)) " +
                                    "heightDrift=\(String(format: "%.3f", heightDriftFraction))"
                                )
                            }
                        } else {
                            // Slightly slower EMA when depth is noisy on small objects (glass / cup) to cut visible pulsing.
                            let maxStep: CGFloat = 0.06
                            let delta = max(-maxStep, min(maxStep, target - base))
                            let clampedTarget = base + delta
                            let alpha: CGFloat = 0.12
                            smoothedArOverlayScale = base * (1.0 - alpha) + clampedTarget * alpha
                        }
                    }
                    autoScaleFromAR = smoothedArOverlayScale
                    arAssistedScaleValid = true
                    arAssistedConsecutiveMisses = 0
                    overlayPresentationMode = .measuredPlacement
                    stableOverlayMeasurementFrameCount = max(stableOverlayMeasurementFrameCount, requiredStableOverlayMeasurementFrames)
                    lastStableOverlayHeightMeters = estH
                    lastStableOverlayScale = autoScaleFromAR
                    let estW = (bboxWidthImagePx / max(fx, 1)) * d
                    let nowLog = Date().timeIntervalSince1970
                    if nowLog - lastFurnitureFitARFrameLogAt >= furnitureFitARLogInterval {
                        lastFurnitureFitARFrameLogAt = nowLog
                        let arOnLog = hasARKitAssistedSizingPayload && arAssistedScaleValid
                        let roomUsedLog: CGFloat = arOnLog ? 1.0 : autoScaleFromRoom
                        let assistedLog: CGFloat = arOnLog ? autoScaleFromAR : 1.0
                        let combinedOverlay = roomUsedLog * assistedLog * userPinchScale
                        logFurnitureFitAR(
                            "distSource=\(distSource) dist=\(String(format: "%.2f", d))m furniture=\(String(format: "%.2f", estW))×\(String(format: "%.2f", estH))m roomH=\(String(format: "%.2f", roomHeight))m targetFrac=\(String(format: "%.3f", targetImageHeightFraction)) currentFrac=\(String(format: "%.3f", currentImageHeightFraction)) ratio=\(String(format: "%.3f", raw)) roomStored=\(String(format: "%.3f", autoScaleFromRoom)) roomUsed=\(String(format: "%.3f", roomUsedLog)) ar=\(String(format: "%.3f", autoScaleFromAR)) pinch=\(String(format: "%.3f", userPinchScale)) overlay=\(String(format: "%.3f", combinedOverlay)) tracking=\(trackingName) metricPx=\(Int(bboxWidthImagePx))×\(Int(bboxHeightImagePx)) fy=\(String(format: "%.1f", fy)) fx=\(String(format: "%.1f", fx))"
                        )
                    }
                    applyCurrentOverlayScaleTransform()
                    return
                }
            }
            // AR sizing is on but this frame had no depth / invalid estimate.
            // If we've *ever* had a valid AR scale, keep reusing it instead of
            // snapping back to 1× (that snap is what causes the big/small oscillation).
            if arAssistedScaleValid {
                if debugMode {
                    logDebug("📐 [AR_HOLD] no depth this frame; keeping last AR scale")
                }
                applyCurrentOverlayScaleTransform()
                return
            }
        } else if wantAR, arAssistedScaleValid {
            // AR is enabled but we didn't even get a frame callback this tick; keep last AR scale.
            if debugMode {
                logDebug("📐 [AR_HOLD] no AR frame; keeping last AR scale")
            }
            applyCurrentOverlayScaleTransform()
            return
        }

        // If we get here, either AR is off, or it's on but has NEVER produced a valid scale
        // in this session (`arAssistedScaleValid == false`). Use 1× assisted scale — no AR ↔ 1× flipping once AR was valid.
        arAssistedConsecutiveMisses = 0
        if !arAssistedScaleValid {
            autoScaleFromAR = 1.0
        }
        applyCurrentOverlayScaleTransform()
    }

    private func resetOverlayScalesForEmptyMask() {
        invalidatePendingAssistedMeasurement()
        autoScaleFromRoom = 1.0
        autoScaleFromAR = 1.0
        smoothedArOverlayScale = 1.0
        arAssistedConsecutiveMisses = 0
        arAssistedScaleValid = false
        overlayDebugLastAssistedLabel = ""
        overlayDebugLastCombined = -1
        lastAREstimatedHeightMeters = nil
        didPublishARRefinedFurnitureMetersEstimate = false
        userPinchScale = 1.0
        userLockedAssistedOverlayScale = false
        lastOverlayPrimaryClassIdx = -1
        overlayPresentationMode = .deferredCentered
        stableOverlayMeasurementFrameCount = 0
        lastStableOverlayHeightMeters = nil
        lastStableOverlayScale = nil
        smoothedTightPrimaryBbox = nil
        maskImageView.center = CGPoint(x: bounds.midX, y: bounds.midY)
        maskImageView.transform = .identity
    }
    
    override func didMoveToSuperview() {
        super.didMoveToSuperview()
        setupGestureConflictResolution()
    }
    
    private func setupGestureConflictResolution() {
        guard let panGesture = maskImageView.gestureRecognizers?.first(where: { $0 is UIPanGestureRecognizer }) as? UIPanGestureRecognizer else { return }
        
        if let vc = self.parentViewController,
           let navController = vc.navigationController,
           let interactivePopGesture = navController.interactivePopGestureRecognizer {
            panGesture.require(toFail: interactivePopGesture)
        }
    }

    // MARK: - Public
    func setModel(_ model: MLModel?) {
        // Must not `sync` onto `detectionQueue` from the main thread: SwiftUI calls this from
        // `updateUIView` while `processFrameInner` can run for hundreds of ms — main would freeze (“hung” UI).
        detectionQueue.async { [weak self] in
            guard let self else { return }
            if model === self.mlModel { return }
            self.mlModel = model
            if let model = model {
                let inputNames = model.modelDescription.inputDescriptionsByName.keys.joined(separator: ", ")
                let outputNames = model.modelDescription.outputDescriptionsByName.keys
                logDebug("🧠 [FurnitureFit] Model set - inputs: [\(inputNames)], outputs: [\(outputNames.joined(separator: ", "))]")
            }
        }
        // Note: loadBlacklist() is called in startIfNeeded() to avoid repeated calls from updateUIView
    }

    func startIfNeeded() {
        // SwiftUI calls this on many `updateUIView` passes; only run setup once until `stop()`.
        if captureSession.isRunning { return }
        if furnitureFitCameraStartupInitiated { return }
        furnitureFitCameraStartupInitiated = true
        logFurnitureFitSize(
            "phase=session_start suppressStartupProgress=\(suppressStartupProgress) segmentationCompletedOnceThisSession=\(segmentationCompletedOnceThisSession)"
        )

        if suppressStartupProgress || segmentationCompletedOnceThisSession {
            hasFirstDetection = false
            DispatchQueue.main.async {
                self.progressContainer.isHidden = true
                self.progressContainer.alpha = 1
            }
        } else {
            hasFirstDetection = false
            setProgress(0.05, text: "Starting camera…")
        }

        // Load blacklist once at start (not in setModel to avoid repeated calls from updateUIView)
        loadBlacklistOnce()

        // Setup orientation detection for landscape support
        currentDeviceOrientation = UIDevice.current.orientation.isValidInterfaceOrientation
            ? UIDevice.current.orientation : .portrait

        UIDevice.current.beginGeneratingDeviceOrientationNotifications()
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(deviceOrientationDidChange),
            name: UIDevice.orientationDidChangeNotification,
            object: nil
        )
        startDeviceMotionPitchUpdatesIfNeeded()

        requestCameraPermissionAndStart()

        // Set initial video rotation and progress orientation after camera starts
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) { [weak self] in
            guard let self = self else { return }
            self.updateVideoRotationForOrientation(self.currentDeviceOrientation)
            self.updateProgressOrientationOnMain(self.currentDeviceOrientation)
        }
    }

    @objc private func deviceOrientationDidChange() {
        let orientation = UIDevice.current.orientation
        if orientation.isValidInterfaceOrientation {
            currentDeviceOrientation = orientation
            // Update video rotation to match device orientation for full FOV
            updateVideoRotationForOrientation(orientation)
            // Rotate progress UI to align with device orientation
            updateProgressOrientationOnMain(orientation)
        }
    }

    private func updateProgressOrientationOnMain(_ orientation: UIDeviceOrientation) {
        DispatchQueue.main.async {
            UIView.animate(withDuration: 0.25) {
                // For landscape rooms, UI is already locked to landscape - don't rotate progress bar
                if self.lockedOrientation == .landscape {
                    self.progressContainer.transform = .identity
                    return
                }
                // For portrait rooms, rotate progress bar to match device orientation
                switch orientation {
                case .landscapeLeft:
                    self.progressContainer.transform = CGAffineTransform(rotationAngle: .pi / 2)
                case .landscapeRight:
                    self.progressContainer.transform = CGAffineTransform(rotationAngle: -.pi / 2)
                case .portraitUpsideDown:
                    self.progressContainer.transform = CGAffineTransform(rotationAngle: .pi)
                default:
                    self.progressContainer.transform = .identity
                }
            }
        }
    }

    private func updateVideoRotationForOrientation(_ orientation: UIDeviceOrientation) {
        guard !isUsingARCameraPath, let conn = videoOutput.connection(with: .video) else { return }

        // Keep rotation fixed based on locked orientation to ensure consistent buffer dimensions
        // This prevents segmentation misalignment when device rotates
        if lockedOrientation == .landscape {
            // Landscape room: always 0° for consistent 1280x720 landscape buffers
            conn.videoRotationAngle = 0
        } else {
            // Portrait room: always 90° for consistent 720x1280 portrait buffers
            conn.videoRotationAngle = 90
        }
    }

    func stop() {
        invalidatePendingAssistedMeasurement()
        frameLock.lock()
        isProcessing = false
        preferImmediateNextInference = false
        frameLock.unlock()
        // Synchronous reset: if UI/flags were only cleared in `main.async`, the next `startIfNeeded()` could run first and keep stale state.
        hasFirstDetection = false
        segmentationCompletedOnceThisSession = false
        lastSegmentationMeanColorPublishAt = 0
        maskImageView.image = nil
        resetOverlayScalesForEmptyMask()
        logFurnitureFitSize(
            "phase=session_stop cleared isProcessing + segmentation UI flags (fixes stuck frames + stale first-mask after brain off/on)"
        )
        asyncDepthSampler.stop()
        stopDeviceMotionPitchUpdates()
        furnitureFitCameraStartupInitiated = false
        arCompanionStartGeneration &+= 1
        suppressARDepthCompanionAfterCaptureFailure = false
        NotificationCenter.default.removeObserver(self, name: UIDevice.orientationDidChangeNotification, object: nil)
        UIDevice.current.endGeneratingDeviceOrientationNotifications()

        DispatchQueue.main.async {
            self.isUsingARCameraPath = false
            self.isARDepthCompanionSessionRunning = false
            self.arSession.pause()
        }
        captureSessionControlQueue.async {
            if self.captureSession.isRunning {
                self.captureSession.stopRunning()
            }
        }
    }

    private func startDeviceMotionPitchUpdatesIfNeeded() {
        guard deviceMotionManager.isDeviceMotionAvailable else { return }
        guard !deviceMotionManager.isDeviceMotionActive else { return }
        deviceMotionManager.deviceMotionUpdateInterval = 1.0 / 30.0
        deviceMotionManager.startDeviceMotionUpdates(
            using: .xArbitraryZVertical,
            to: deviceMotionQueue
        ) { [weak self] motion, _ in
            guard let self, let motion else { return }
            self.devicePitchLock.lock()
            self.latestDevicePitchRadians = Float(motion.attitude.pitch)
            self.devicePitchLock.unlock()
        }
    }

    private func stopDeviceMotionPitchUpdates() {
        deviceMotionManager.stopDeviceMotionUpdates()
        devicePitchLock.lock()
        latestDevicePitchRadians = nil
        devicePitchLock.unlock()
    }

    private func latestDownwardPitchDegrees() -> Float? {
        devicePitchLock.lock()
        let cachedPitchRadians = latestDevicePitchRadians
        devicePitchLock.unlock()
        let livePitchRadians = deviceMotionManager.deviceMotion.map { Float($0.attitude.pitch) }
        let pitchRadians = livePitchRadians ?? cachedPitchRadians
        guard let pitchRadians else { return nil }
        return abs(pitchRadians) * 180 / .pi
    }

    // MARK: - Camera Setup
    private func setupCamera() {
        captureSession.beginConfiguration()
        captureSession.sessionPreset = .hd1280x720
        captureSession.inputs.forEach { captureSession.removeInput($0) }
        captureSession.outputs.forEach { captureSession.removeOutput($0) }

        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
              let input = try? AVCaptureDeviceInput(device: device),
              captureSession.canAddInput(input) else {
            captureSession.commitConfiguration()
            return
        }
        
        captureSession.addInput(input)
        captureBackCamera = device
        videoOutput.setSampleBufferDelegate(self, queue: sampleQueue)
        videoOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        videoOutput.alwaysDiscardsLateVideoFrames = true
        
        if captureSession.canAddOutput(videoOutput) {
            captureSession.addOutput(videoOutput)
        }
        // Set video rotation based on locked orientation
        // For landscape rooms: use 0° (native landscape frames)
        // For portrait rooms: use 90° (rotated to portrait frames)
        if let conn = videoOutput.connection(with: .video) {
            if lockedOrientation == .landscape {
                conn.videoRotationAngle = 0
                logDebug("📷 [Camera] Landscape room: using 0° rotation (1280x720)")
            } else {
                conn.videoRotationAngle = 90
                logDebug("📷 [Camera] Portrait room: using 90° rotation (720x1280)")
            }
        }
        captureSession.commitConfiguration()
    }

    private func requestCameraPermissionAndStart() {
        // Camera permission still gates the AR-as-camera path because ARKit uses the same hardware.
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            startPreferredCameraPathIfNeeded()
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { granted in
                guard granted else { return }
                self.startPreferredCameraPathIfNeeded()
            }
        default:
            break
        }
    }

    private func startPreferredCameraPathIfNeeded() {
        let wantAR = AppStateManager.shared.qualitySettings.furnitureFitARDepthCompanionRuntimeActive
            && !suppressARDepthCompanionAfterCaptureFailure
        if wantAR {
            startARCameraPathIfNeeded()
        } else {
            startClassicCameraPathIfNeeded()
        }
    }

    /// Single camera owner: ARKit supplies both video (`capturedImage`) and depth, so there is no AVCapture + AR contention.
    private func startARCameraPathIfNeeded() {
        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            self.arCompanionStartGeneration &+= 1
            self.isUsingARCameraPath = true
            self.isARDepthCompanionSessionRunning = true

            self.captureSessionControlQueue.async { [weak self] in
                guard let self else { return }
                if self.captureSession.isRunning {
                    self.captureSession.stopRunning()
                }
                DispatchQueue.main.async { [weak self] in
                    guard let self, self.isUsingARCameraPath else { return }
                    let config = FurnitureFitARSupport.makeWorldTrackingConfiguration()
                    let hasLiDAR = QualitySettings.supportsLiDARSceneDepth
                    self.arSession.run(config, options: [.resetTracking, .removeExistingAnchors])
                    logFurnitureFitAR("AR session started — lidar=\(hasLiDAR) depthSource=\(hasLiDAR ? "sceneDepth" : "planeRaycast") (filter: FurnitureFitAR FurnitureFitOverlay)")

                    if !Self.didLogFurnitureFitSizingPolicy {
                        Self.didLogFurnitureFitSizingPolicy = true
                        logFurnitureFitSize(
                            "policy=ar_camera lidar=\(hasLiDAR) → segmentation from ARFrame.capturedImage; depth via \(hasLiDAR ? "sceneDepth" : "plane raycast"). Filters: FurnitureFitAR | FurnitureFitOverlay | FurnitureFitSize"
                        )
                    }
                }
            }
        }
    }

    /// Fig / AVFoundation codes often logged when `ARSession` + `AVCaptureSession` fight over the back camera.
    private static let figCameraContentionErrorCodes: Set<Int> = [-17281, -12784, -12710]

    private static func captureErrorIndicatesCameraContention(_ error: NSError) -> Bool {
        var visited = Set<ObjectIdentifier>()
        var current: NSError? = error
        while let e = current {
            let oid = ObjectIdentifier(e)
            if visited.contains(oid) { break }
            visited.insert(oid)
            if figCameraContentionErrorCodes.contains(e.code) { return true }
            current = e.userInfo[NSUnderlyingErrorKey] as? NSError
        }
        return false
    }

    @objc private func handleCaptureSessionRuntimeError(_ notification: Notification) {
        guard let err = notification.userInfo?[AVCaptureSessionErrorKey] as? NSError else { return }
        guard Self.captureErrorIndicatesCameraContention(err) else { return }
        logDebug("📷 [FurnitureFit] AVCaptureSession runtime error \(err.domain) code=\(err.code) — AR+camera contention; turning off AR companion for this Furniture Fit session.")
        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            let settingsOn = AppStateManager.shared.qualitySettings.furnitureFitARDepthCompanionRuntimeActive
            guard settingsOn || self.isARDepthCompanionSessionRunning else { return }
            self.suppressARDepthCompanionAfterCaptureFailure = true
            self.startPreferredCameraPathIfNeeded()
        }
    }

    /// AVCapture for video; optional ARSession **companion** (Settings) for metric depth on LiDAR **or** plane raycast on e.g. iPhone 12.
    private func startClassicCameraPathIfNeeded() {
        isUsingARCameraPath = false
        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            self.arCompanionStartGeneration &+= 1
            let companionLaunchToken = self.arCompanionStartGeneration
            self.isARDepthCompanionSessionRunning = false
            self.arSession.pause()

            self.captureSessionControlQueue.async { [weak self] in
                guard let self else { return }
                self.setupCamera()
                if !self.captureSession.isRunning {
                    self.captureSession.startRunning()
                }
                // Floor-contact depth estimation stays on the AVCapture path; no AR companion is started here.
                _ = companionLaunchToken
            }

            if self.debugMode {
                logDebug("📷 [FurnitureFit] AVCaptureSession (classic); floor-contact depth estimate active, AR path not selected")
            }

            if !Self.didLogFurnitureFitSizingPolicy {
                Self.didLogFurnitureFitSizingPolicy = true
                logFurnitureFitSize(
                    "policy=floor_contact_only: AVCapture video + floor-contact depth heuristic from bbox/mask bottom. AR path not selected."
                )
            }
        }
    }

    /*
    /// World-tracking AR beside `AVCaptureSession` (disabled for pinhole-only builds).
    private func scheduleARDepthCompanionAfterCaptureIfEligible(launchGeneration: UInt64) {
        let qs = AppStateManager.shared.qualitySettings
        guard qs.furnitureFitARDepthCompanionRuntimeActive,
              !suppressARDepthCompanionAfterCaptureFailure,
              FurnitureFitARSupport.isWorldTrackingSupported,
              QualitySettings.supportsLiDARSceneDepth else {
            if qs.furnitureFitARDepthCompanionEnabled,
               FurnitureFitARSupport.isWorldTrackingSupported,
               !QualitySettings.supportsLiDARSceneDepth {
                logFurnitureFitAR(
                    "event=ar_depth_companion_skipped reason=no_LiDAR parallel_AR_plus_AVCapture_causes_Fig_-17281 pinhole_uses_room_proxy_vy"
                )
            }
            return
        }
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.4) { [weak self] in
            guard let self else { return }
            guard launchGeneration == self.arCompanionStartGeneration else { return }
            guard !self.isUsingARCameraPath else { return }
            guard self.captureSession.isRunning else { return }
            let config = FurnitureFitARSupport.makeWorldTrackingConfiguration()
            self.arSession.run(config, options: [.resetTracking, .removeExistingAnchors])
            self.isARDepthCompanionSessionRunning = true
            logFurnitureFitAR("event=ar_depth_companion_started parallel_capture lidar=true sceneDepth_for_pinhole=true")
        }
    }
    */

    // MARK: - Capture Delegate
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        // Drop every frame while segmentation is in flight (do not queue work on `detectionQueue`).
        let now = Date()
        frameLock.lock()
        if isProcessing {
            preferImmediateNextInference = true
            frameLock.unlock()
            return
        }
        let shouldProcess = now.timeIntervalSince(lastProcessTime) >= processInterval
        if shouldProcess {
            isProcessing = true
            lastProcessTime = now
        }
        frameLock.unlock()

        guard shouldProcess else { return }
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            resetProcessingFlag()
            return
        }
        // Classic camera path feeds the non-AR floor-contact estimate only.
        let depthSnapshot: FurnitureFitARDepthSnapshot? = nil
        detectionQueue.async { [weak self] in
            self?.processFrame(pixelBuffer, arDepthSnapshot: depthSnapshot)
        }
    }

    // MARK: - Main Processing Pipeline
    private func processFrame(_ pixelBuffer: CVPixelBuffer, arDepthSnapshot: FurnitureFitARDepthSnapshot? = nil) {
        autoreleasepool {
            processFrameInner(pixelBuffer, arDepthSnapshot: arDepthSnapshot)
        }
    }

    /// Floor-contact size in **meters** from the primary bbox using a non-AR depth estimate from the object's floor contact.
    private typealias PrimaryBboxMetersResult = (size: FurnitureSceneSize, pipeline: String, distanceMeters: Float?)
    private typealias FloorContactSizingResult = (
        metric: PrimaryBboxMetersResult,
        contactImageX: Int,
        contactImageY: Int,
        widthPixels: Float,
        heightPixels: Float,
        focalPixels: Float,
        measuredPitchDeg: Float?,
        effectivePitchDeg: Float
    )

    private func estimatedPitchDegreesForFloorContact(
        contactImageY: Float,
        imageHeight: Int,
        focalPixels: Float
    ) -> (measuredPitchDeg: Float?, effectivePitchDeg: Float) {
        let measuredPitchDeg = latestDownwardPitchDegrees()
        let cy = Float(imageHeight) * 0.5
        let yCam = max(0, (contactImageY - cy) / max(focalPixels, 1))
        let minimumPitchDeg = atan(yCam) * 180 / .pi + 1.0
        let basePitchDeg = measuredPitchDeg ?? 35.0
        let effectivePitchDeg = min(80.0, max(max(10.0, basePitchDeg), minimumPitchDeg))
        return (measuredPitchDeg, effectivePitchDeg)
    }

    private func floorContactDepthEstimateMeters(
        contactImageY: Float,
        imageHeight: Int,
        focalPixels: Float
    ) -> (depthMeters: Float, measuredPitchDeg: Float?, effectivePitchDeg: Float) {
        let (measuredPitchDeg, effectivePitchDeg) = estimatedPitchDegreesForFloorContact(
            contactImageY: contactImageY,
            imageHeight: imageHeight,
            focalPixels: focalPixels
        )
        let cameraHeightMeters: Float = 1.2
        let cy = Float(imageHeight) * 0.5
        let yCam = (contactImageY - cy) / max(focalPixels, 1)
        let pitchRadians = effectivePitchDeg * .pi / 180
        let sinPitch = sin(pitchRadians)
        let cosPitch = cos(pitchRadians)
        let numerator = cameraHeightMeters * (yCam * sinPitch + cosPitch)
        let denominator = sinPitch - yCam * cosPitch
        guard denominator > 1e-4 else {
            return (0.3, measuredPitchDeg, effectivePitchDeg)
        }
        let depthMeters = max(0.3, min(10.0, numerator / denominator))
        return (depthMeters, measuredPitchDeg, effectivePitchDeg)
    }

    private func floorContactSizingResult(
        contactImageX: Float,
        contactImageY: Float,
        widthPixels: Float,
        heightPixels: Float,
        imageWidth: Int,
        imageHeight: Int,
        pipeline: String,
        preferRoomRaycastSizing: Bool
    ) -> FloorContactSizingResult? {
        guard imageWidth > 1, imageHeight > 1, widthPixels > 0.5, heightPixels > 0.5 else { return nil }

        let clampedImageX = min(max(contactImageX, 0), Float(max(1, imageWidth - 1)))
        let clampedImageY = min(max(contactImageY, 0), Float(max(1, imageHeight - 1)))
        var focalLengthX = cameraFocalLengthPixels
        if focalLengthX < 1, let camera = captureBackCamera {
            focalLengthX = FurnitureMonocularMeasurer.focalLengthPixels(
                imageWidth: Float(imageWidth),
                hFovDegrees: Float(camera.activeFormat.videoFieldOfView)
            )
        }
        if focalLengthX < 1 { focalLengthX = Float(imageWidth) * 0.73 }
        let focalLengthY = focalLengthX
        let depthEstimate = floorContactDepthEstimateMeters(
            contactImageY: clampedImageY,
            imageHeight: imageHeight,
            focalPixels: focalLengthY
        )
        let depthMeters = depthEstimate.depthMeters
        let preferredRoomMetricSize = preferredRoomModelMetricSize(
            contactImageX: clampedImageX,
            contactImageY: clampedImageY,
            widthPixels: widthPixels,
            heightPixels: heightPixels,
            imageWidth: imageWidth,
            imageHeight: imageHeight,
            depthMeters: depthMeters
        )

        let fallbackMetricSize: (widthMeters: Float, heightMeters: Float, pipeline: String)? = {
            guard let fallbackWidthMeters = FurnitureFitARSupport.estimatedPhysicalWidthMeters(
                bboxWidthPixels: widthPixels,
                distanceMeters: depthMeters,
                focalLengthXPixels: focalLengthX
            ),
            let fallbackHeightMeters = FurnitureFitARSupport.estimatedPhysicalHeightMeters(
                bboxHeightPixels: heightPixels,
                distanceMeters: depthMeters,
                focalLengthYPixels: focalLengthY
            ) else { return nil }
            return (fallbackWidthMeters, fallbackHeightMeters, pipeline)
        }()

        let selectedMetricSize: (widthMeters: Float, heightMeters: Float, pipeline: String)?
        if !preferRoomRaycastSizing {
            logFurnitureFitSize(
                "phase=metric_policy policy=prefer_pinhole reason=no_support_surface_hint depth_m=\(String(format: "%.3f", depthMeters))"
            )
            if let fallbackMetricSize,
               isReasonableFurnitureMetricEstimate(
                widthMeters: fallbackMetricSize.widthMeters,
                heightMeters: fallbackMetricSize.heightMeters,
                distanceMeters: depthMeters,
                pipeline: fallbackMetricSize.pipeline
               ) {
                selectedMetricSize = fallbackMetricSize
            } else {
                selectedMetricSize = nil
            }
        } else if let preferredRoomMetricSize,
                  isReasonableFurnitureMetricEstimate(
                    widthMeters: preferredRoomMetricSize.widthMeters,
                    heightMeters: preferredRoomMetricSize.heightMeters,
                    distanceMeters: depthMeters,
                    pipeline: preferredRoomMetricSize.pipeline
                  ) {
            logFurnitureFitSize(
                "phase=metric_policy policy=prefer_room_raycast reason=support_surface_hint depth_m=\(String(format: "%.3f", depthMeters)) " +
                "selected=\(preferredRoomMetricSize.pipeline)"
            )
            selectedMetricSize = preferredRoomMetricSize
        } else if let preferredRoomMetricSize {
            logFurnitureFitSize(
                "phase=metric_fallback rejected_pipeline=\(preferredRoomMetricSize.pipeline) " +
                "fallback_pipeline=\(fallbackMetricSize?.pipeline ?? "none") " +
                "depth_m=\(String(format: "%.3f", depthMeters))"
            )
            if let fallbackMetricSize,
               isReasonableFurnitureMetricEstimate(
                widthMeters: fallbackMetricSize.widthMeters,
                heightMeters: fallbackMetricSize.heightMeters,
                distanceMeters: depthMeters,
                pipeline: fallbackMetricSize.pipeline
               ) {
                selectedMetricSize = fallbackMetricSize
            } else {
                selectedMetricSize = nil
            }
        } else if let fallbackMetricSize,
                  isReasonableFurnitureMetricEstimate(
                    widthMeters: fallbackMetricSize.widthMeters,
                    heightMeters: fallbackMetricSize.heightMeters,
                    distanceMeters: depthMeters,
                    pipeline: fallbackMetricSize.pipeline
                  ) {
            logFurnitureFitSize(
                "phase=metric_policy policy=room_raycast_unavailable_using_pinhole depth_m=\(String(format: "%.3f", depthMeters))"
            )
            selectedMetricSize = fallbackMetricSize
        } else {
            selectedMetricSize = nil
        }

        guard let selectedMetricSize else { return nil }
        let widthMeters = selectedMetricSize.widthMeters
        let heightMeters = selectedMetricSize.heightMeters
        let sizingPipelinePrefix = selectedMetricSize.pipeline

        let depthThicknessMeters = max(0.04, min(widthMeters, heightMeters) * 0.12)
        return (
            metric: (
                size: FurnitureSceneSize(width: widthMeters, height: heightMeters, depth: depthThicknessMeters),
                pipeline: sizingPipelinePrefix +
                    "_pitch\(String(format: "%.1f", depthEstimate.effectivePitchDeg))" +
                    (depthEstimate.measuredPitchDeg != nil ? "_motion" : "_fallback"),
                distanceMeters: depthMeters
            ),
            contactImageX: Int(round(clampedImageX)),
            contactImageY: Int(round(clampedImageY)),
            widthPixels: widthPixels,
            heightPixels: heightPixels,
            focalPixels: focalLengthY,
            measuredPitchDeg: depthEstimate.measuredPitchDeg,
            effectivePitchDeg: depthEstimate.effectivePitchDeg
        )
    }

    private func isReasonableFurnitureMetricEstimate(
        widthMeters: Float,
        heightMeters: Float,
        distanceMeters: Float,
        pipeline: String
    ) -> Bool {
        guard widthMeters.isFinite,
              heightMeters.isFinite,
              distanceMeters.isFinite,
              widthMeters > 0.015,
              heightMeters > 0.015,
              distanceMeters > 0.2 else {
            logFurnitureFitSize(
                "phase=metric_reject pipeline=\(pipeline) reason=non_finite_or_too_small " +
                "W×H_m=\(String(format: "%.3f", widthMeters))×\(String(format: "%.3f", heightMeters)) " +
                "dist_m=\(String(format: "%.3f", distanceMeters))"
            )
            return false
        }

        let maxRoomSpan = max(roomWidthMeters, roomDepthMeters, 0.5)
        let widthLimit = max(6.0, maxRoomSpan * 1.35)
        let heightLimit = max(4.0, max(roomHeightMeters, 0.5) * 1.25)
        let distanceLimit = max(12.0, max(roomDepthMeters, 0.5) * 2.0)

        guard widthMeters <= widthLimit,
              heightMeters <= heightLimit,
              distanceMeters <= distanceLimit else {
            logFurnitureFitSize(
                "phase=metric_reject pipeline=\(pipeline) reason=exceeds_limits " +
                "W×H_m=\(String(format: "%.3f", widthMeters))×\(String(format: "%.3f", heightMeters)) " +
                "dist_m=\(String(format: "%.3f", distanceMeters)) " +
                "limits_w_h_d=\(String(format: "%.3f", widthLimit))×\(String(format: "%.3f", heightLimit))×\(String(format: "%.3f", distanceLimit)) " +
                "room_w_h_d=\(String(format: "%.3f", roomWidthMeters))×\(String(format: "%.3f", roomHeightMeters))×\(String(format: "%.3f", roomDepthMeters))"
            )
            return false
        }

        return true
    }

    private func preferredRoomModelMetricSize(
        contactImageX: Float,
        contactImageY: Float,
        widthPixels: Float,
        heightPixels: Float,
        imageWidth: Int,
        imageHeight: Int,
        depthMeters: Float
    ) -> (widthMeters: Float, heightMeters: Float, pipeline: String)? {
        guard imageWidth > 1, imageHeight > 1 else { return nil }

        let proportionalFallback: () -> (widthMeters: Float, heightMeters: Float, pipeline: String)? = {
            guard self.roomWidthMeters.isFinite, self.roomHeightMeters.isFinite,
                  self.roomWidthMeters > 0.05, self.roomHeightMeters > 0.05 else {
                return nil
            }
            let widthMeters = (widthPixels / Float(imageWidth)) * self.roomWidthMeters
            let heightMeters = (heightPixels / Float(imageHeight)) * self.roomHeightMeters
            guard widthMeters.isFinite, heightMeters.isFinite,
                  widthMeters > 0.01, heightMeters > 0.01 else {
                return nil
            }
            logFurnitureFitSize(
                "phase=room_model_sizing_fallback pipeline=room_bbox_proportion " +
                "W×H_m=\(String(format: "%.3f", widthMeters))×\(String(format: "%.3f", heightMeters)) " +
                "room_m=\(String(format: "%.3f", self.roomWidthMeters))×\(String(format: "%.3f", self.roomHeightMeters)) " +
                "bbox_px=\(String(format: "%.1f", widthPixels))×\(String(format: "%.1f", heightPixels))"
            )
            return (widthMeters, heightMeters, "room_bbox_proportion")
        }

        guard let roomModel,
              roomModel.sceneToMeters.isFinite,
              roomModel.sceneToMeters > 0.0001 else {
            logFurnitureFitSize(
                "phase=room_model_sizing_unavailable reason=missing_room_model_or_scale " +
                "roomModel=\(roomModel != nil) sceneToMeters=\(String(format: "%.4f", roomModel?.sceneToMeters ?? 0))"
            )
            return proportionalFallback()
        }

        guard let cam = roomModel.cameraInfo,
              let focalLengthMM = cam.focalLengthMM,
              focalLengthMM > 0,
              let sensorWidthMM = cam.sensorWidthMM,
              sensorWidthMM > 0 else {
            logFurnitureFitSize(
                "phase=room_model_sizing_unavailable reason=missing_intrinsics " +
                "sceneToMeters=\(String(format: "%.4f", roomModel.sceneToMeters)) " +
                "focal_mm=\(String(format: "%.2f", roomModel.cameraInfo?.focalLengthMM ?? 0)) " +
                "sensor_w_mm=\(String(format: "%.2f", roomModel.cameraInfo?.sensorWidthMM ?? 0))"
            )
            return proportionalFallback()
        }

        let intrinsics = DepthSizingService.CameraIntrinsics(
            focalLengthMM: focalLengthMM,
            sensorWidthMM: sensorWidthMM,
            imageWidthPx: Float(cam.imageWidthPx ?? imageWidth),
            imageHeightPx: Float(cam.imageHeightPx ?? imageHeight)
        )
        let sizingService = DepthSizingService(
            intrinsics: intrinsics,
            sceneToMeters: roomModel.sceneToMeters
        )

        let sceneDepth = roomModel.toScene(depthMeters: depthMeters)
        guard sceneDepth.isFinite, sceneDepth > 0 else {
            logFurnitureFitSize(
                "phase=room_model_sizing_unavailable reason=invalid_scene_depth " +
                "metric_depth=\(String(format: "%.3f", depthMeters)) scene_depth=\(String(format: "%.3f", sceneDepth))"
            )
            return proportionalFallback()
        }

        let halfWidth = widthPixels * 0.5
        let widthMeters = sizingService.estimateMetricWidth(
            leftScreenX: CGFloat(contactImageX - halfWidth),
            rightScreenX: CGFloat(contactImageX + halfWidth),
            depth: sceneDepth,
            viewportWidth: CGFloat(imageWidth)
        )
        let heightMeters = sizingService.estimateMetricHeight(
            topScreenY: CGFloat(contactImageY - heightPixels),
            bottomScreenY: CGFloat(contactImageY),
            depth: sceneDepth,
            viewportWidth: CGFloat(imageWidth)
        )

        guard widthMeters.isFinite, heightMeters.isFinite,
              widthMeters > 0.01, heightMeters > 0.01 else {
            logFurnitureFitSize(
                "phase=room_model_sizing_unavailable reason=invalid_intrinsics_output " +
                "W×H_m=\(String(format: "%.3f", widthMeters))×\(String(format: "%.3f", heightMeters))"
            )
            return proportionalFallback()
        }

        logFurnitureFitSize(
            "phase=room_model_sizing sceneToMeters=\(String(format: "%.4f", roomModel.sceneToMeters)) " +
            "scene_depth=\(String(format: "%.3f", sceneDepth)) metric_depth=\(String(format: "%.3f", depthMeters)) " +
            "focal_mm=\(String(format: "%.2f", focalLengthMM)) sensor_w_mm=\(String(format: "%.2f", sensorWidthMM)) " +
            "W×H_m=\(String(format: "%.3f", widthMeters))×\(String(format: "%.3f", heightMeters))"
        )

        return (widthMeters, heightMeters, "room_model_depth_intrinsics")
    }

    private func refinedMaskBoundsAndFloorContact(
        maskSmall: [UInt8],
        protoW: Int,
        protoH: Int,
        imageWidth: Int,
        imageHeight: Int,
        bboxCenterImageX: Int
    ) -> (contactImageX: Int, contactImageY: Int, widthPixels: Float, heightPixels: Float)? {
        guard protoW > 1, protoH > 1, imageWidth > 1, imageHeight > 1, !maskSmall.isEmpty else { return nil }

        var minProtoX = protoW
        var minProtoY = protoH
        var maxProtoX = -1
        var maxProtoY = -1

        for protoY in 0..<protoH {
            let rowOffset = protoY * protoW
            for protoX in 0..<protoW where maskSmall[rowOffset + protoX] > 0 {
                minProtoX = min(minProtoX, protoX)
                minProtoY = min(minProtoY, protoY)
                maxProtoX = max(maxProtoX, protoX)
                maxProtoY = max(maxProtoY, protoY)
            }
        }

        guard maxProtoX >= minProtoX, maxProtoY >= minProtoY else { return nil }

        let bboxCenterProtoX = Int(
            round((Float(bboxCenterImageX) / Float(max(1, imageWidth))) * Float(max(1, protoW - 1)))
        )
        var contactProtoX = bboxCenterProtoX
        var contactProtoY = maxProtoY
        var foundContact = false

        for protoY in stride(from: maxProtoY, through: minProtoY, by: -1) {
            let rowOffset = protoY * protoW
            var bestProtoXOnRow: Int?
            var bestCenterDelta = Int.max
            for protoX in minProtoX...maxProtoX where maskSmall[rowOffset + protoX] > 0 {
                let delta = abs(protoX - bboxCenterProtoX)
                if delta < bestCenterDelta {
                    bestCenterDelta = delta
                    bestProtoXOnRow = protoX
                }
            }
            if let rowX = bestProtoXOnRow {
                contactProtoX = rowX
                contactProtoY = protoY
                foundContact = true
                break
            }
        }

        guard foundContact else { return nil }

        let minImageX = max(0, Int(floor(Float(minProtoX) * Float(imageWidth) / Float(protoW))))
        let maxImageX = min(imageWidth, Int(ceil(Float(maxProtoX + 1) * Float(imageWidth) / Float(protoW))))
        let minImageY = max(0, Int(floor(Float(minProtoY) * Float(imageHeight) / Float(protoH))))
        let maxImageY = min(imageHeight, Int(ceil(Float(maxProtoY + 1) * Float(imageHeight) / Float(protoH))))
        let contactImageX = min(
            max(0, Int(round((Float(contactProtoX) + 0.5) * Float(imageWidth) / Float(protoW)))),
            max(0, imageWidth - 1)
        )
        let contactImageY = min(
            max(0, Int(round((Float(contactProtoY) + 0.5) * Float(imageHeight) / Float(protoH)))),
            max(0, imageHeight - 1)
        )

        return (
            contactImageX: contactImageX,
            contactImageY: contactImageY,
            widthPixels: Float(max(1, maxImageX - minImageX)),
            heightPixels: Float(max(1, maxImageY - minImageY))
        )
    }

    private func refinedMaskFloorContactSizeMeters(
        maskSmall: [UInt8],
        protoW: Int,
        protoH: Int,
        bboxCenterImageX: Int,
        imageWidth: Int,
        imageHeight: Int,
        preferRoomRaycastSizing: Bool
    ) -> FloorContactSizingResult? {
        guard let refinedMask = refinedMaskBoundsAndFloorContact(
            maskSmall: maskSmall,
            protoW: protoW,
            protoH: protoH,
            imageWidth: imageWidth,
            imageHeight: imageHeight,
            bboxCenterImageX: bboxCenterImageX
        ) else { return nil }

        return floorContactSizingResult(
            contactImageX: Float(refinedMask.contactImageX),
            contactImageY: Float(refinedMask.contactImageY),
            widthPixels: refinedMask.widthPixels,
            heightPixels: refinedMask.heightPixels,
            imageWidth: imageWidth,
            imageHeight: imageHeight,
            pipeline: "floor_contact_mask_contact",
            preferRoomRaycastSizing: preferRoomRaycastSizing
        )
    }

    /// Unlike ``FurnitureSizingCalculator``, this does **not** assume the full frame height equals ``roomHeightMeters``,
    /// so close-up shots do not blow up to ~room-sized readings.
    private func primaryBboxMonocularSizeMeters(
        primaryBx1: Int,
        primaryBy1: Int,
        primaryBx2: Int,
        primaryBy2: Int,
        imageWidth: Int,
        imageHeight: Int,
        arDepthSnapshot: FurnitureFitARDepthSnapshot?,
        preferRoomRaycastSizing: Bool
    ) -> PrimaryBboxMetersResult? {
        _ = arDepthSnapshot
        let bboxWpx = Float(max(1, primaryBx2 - primaryBx1))
        let bboxHpx = Float(max(1, primaryBy2 - primaryBy1))
        guard let floorMeasurement = floorContactSizingResult(
            contactImageX: Float(primaryBx1 + primaryBx2) * 0.5,
            contactImageY: Float(max(primaryBy1, primaryBy2 - 1)),
            widthPixels: bboxWpx,
            heightPixels: bboxHpx,
            imageWidth: imageWidth,
            imageHeight: imageHeight,
            pipeline: "floor_contact_bbox_bottom_center",
            preferRoomRaycastSizing: preferRoomRaycastSizing
        ) else { return nil }
        return floorMeasurement.metric
    }

    /// Throttled logs: furniture vs room **ratio** space when ``roomRaycastSceneDimensions`` is set (Sharp / saved `.meta`).
    private func maybeLogSceneFitmentFromPrimaryBbox(
        primaryBx1: Int,
        primaryBy1: Int,
        primaryBx2: Int,
        primaryBy2: Int,
        imageWidth: Int,
        imageHeight: Int,
        arDepthSnapshot: FurnitureFitARDepthSnapshot?,
        preferRoomRaycastSizing: Bool
    ) {
        guard let room = roomRaycastSceneDimensions,
              room.width > 1e-5, room.height > 1e-5, room.depth > 1e-5 else { return }
        let now = CFAbsoluteTimeGetCurrent()
        guard now - lastSceneFitmentLogAt > 2.0 else { return }
        lastSceneFitmentLogAt = now

        guard let metric = primaryBboxMonocularSizeMeters(
            primaryBx1: primaryBx1,
            primaryBy1: primaryBy1,
            primaryBx2: primaryBx2,
            primaryBy2: primaryBy2,
            imageWidth: imageWidth,
            imageHeight: imageHeight,
            arDepthSnapshot: arDepthSnapshot,
            preferRoomRaycastSizing: preferRoomRaycastSizing
        ) else { return }
        let furnMeters = metric.size

        // Pinhole path is in meters; room raycast is in splat scene units — align before ratio checks.
        let sceneToMeters: Float = {
            if let m = roomModel?.sceneToMeters, m > 1e-4 { return m }
            if room.depth > 1e-5 { return roomDepthMeters / room.depth }
            if room.width > 1e-5 { return roomWidthMeters / room.width }
            return 0
        }()
        guard sceneToMeters > 1e-4 else { return }
        let furnSu = FurnitureMonocularMeasurer.furnitureMetersMappedToRaycastSceneUnits(
            furnitureMeters: furnMeters,
            sceneToMeters: sceneToMeters
        )

        logFurnitureFitSize(
            "phase=fitment_abs furniture_su=\(String(format: "%.3f", furnSu.width))×\(String(format: "%.3f", furnSu.height))×\(String(format: "%.3f", furnSu.depth)) " +
            "room_su=\(String(format: "%.3f", room.width))×\(String(format: "%.3f", room.height))×\(String(format: "%.3f", room.depth)) " +
            "pinhole_m=\(String(format: "%.3f", furnMeters.width))×\(String(format: "%.3f", furnMeters.height))×\(String(format: "%.3f", furnMeters.depth)) " +
            "pipeline=\(metric.pipeline)"
        )

        _ = FitmentCheck.check(furniture: furnSu, room: room)
        _ = OverlayScale.compute(furniture: furnSu, room: room)
    }

    /// ONNX-style pipeline (stretch, NMS, primary, mask) using Core ML ``mlModel`` (parity with Android preprocessing).
    private func processFrameOnnxStyleCoreML(
        processBuffer: CVPixelBuffer,
        model: MLModel,
        arDepthSnapshot: FurnitureFitARDepthSnapshot?,
        frameStart: Date
    ) {
        let bufW = CVPixelBufferGetWidth(processBuffer)
        let bufH = CVPixelBufferGetHeight(processBuffer)
        let isLandscape = bufW > bufH

        let modelInputSize = YoloEImageInference.modelInputSize(for: model)

        if debugMode {
            logDebug("\n⏱️ ═══════════════════════════════════════════")
            logDebug("⏱️ ONNX-STYLE + CORE ML @ \(String(format: "%.3f", frameStart.timeIntervalSince1970))")
            logDebug("⏱️ Buffer: \(bufW)x\(bufH) → stretch \(modelInputSize)x\(modelInputSize), isLandscape: \(isLandscape)")
            logDebug("⏱️ ═══════════════════════════════════════════")
        }

        setProgress(0.15, text: "Preprocessing…")
        let t1 = Date()
        guard let stretched = resizeStretchToSquare(processBuffer, size: modelInputSize) else {
            if debugMode { logDebug("❌ ONNX-STYLE Core ML STAGE 1 FAILED: stretch resize") }
            resetProcessingFlag()
            return
        }
        if debugMode {
            logDebug("⏱️ ONNX-STYLE STAGE 1 - Preprocess (stretch): \(String(format: "%.2f", Date().timeIntervalSince(t1) * 1000)) ms")
        }

        let inputDesc = model.modelDescription.inputDescriptionsByName["image"]
        let expectsImage = inputDesc?.type == .image

        let inputProvider: MLFeatureProvider
        if expectsImage {
            guard let imageValue = MLFeatureValue(pixelBuffer: stretched) as MLFeatureValue?,
                  let provider = try? MLDictionaryFeatureProvider(dictionary: ["image": imageValue]) else {
                if debugMode { logDebug("❌ ONNX-STYLE Core ML STAGE 1 FAILED: Input prep") }
                resetProcessingFlag()
                return
            }
            inputProvider = provider
        } else {
            guard let inputArray = pixelBufferToMLMultiArray(stretched),
                  let provider = try? MLDictionaryFeatureProvider(dictionary: ["image": inputArray]) else {
                if debugMode { logDebug("❌ ONNX-STYLE Core ML STAGE 1 FAILED: Input prep (MultiArray)") }
                resetProcessingFlag()
                return
            }
            inputProvider = provider
        }

        setProgress(0.40, text: "Running model…")
        let t2 = Date()
        guard let output = try? model.prediction(from: inputProvider) else {
            if debugMode { logDebug("❌ ONNX-STYLE Core ML STAGE 2 FAILED: Model inference") }
            resetProcessingFlag()
            return
        }
        if debugMode {
            logDebug("⏱️ ONNX-STYLE STAGE 2 - Inference (Core ML): \(String(format: "%.2f", Date().timeIntervalSince(t2) * 1000)) ms")
            logMemory("AFTER ONNX-STYLE Core ML INFERENCE")
        }

        guard let pair = YoloEDetectionParser.extractDetectionAndProto(from: output) else {
            if debugMode {
                logDebug("❌ ONNX-STYLE Core ML: Missing output tensors")
                let availableOutputs = output.featureNames.joined(separator: ", ")
                logDebug("   Available outputs: \(availableOutputs)")
            }
            resetProcessingFlag()
            return
        }

        processFrameOnnxStyleCommon(
            processBuffer: processBuffer,
            arDepthSnapshot: arDepthSnapshot,
            frameStart: frameStart,
            detArray: pair.det,
            protoArray: pair.proto,
            modelSide: modelInputSize,
            stage2DebugLabel: "Core ML"
        )
    }

    /// Primary index: conf^1.5 × areaNorm^0.8 × center^0.5 with min conf/area gates.
    private func selectPrimaryIndexCoreFlow(candidates: [FurnitureFitDetection], modelSide: Int) -> Int? {
        var bestIdx: Int?
        var maxScore: Float = -1
        let minConf: Float = 0.15
        let minAreaNorm: Float = 0.02
        let frameArea = Float(modelSide * modelSide)
        let frameW = Float(modelSide)
        let frameH = Float(modelSide)
        let frameCx = frameW / 2
        let frameCy = frameH / 2

        for (i, d) in candidates.enumerated() {
            let areaNorm = (d.w * d.h) / frameArea
            guard d.confidence >= minConf, areaNorm >= minAreaNorm else { continue }

            let dx = (d.x - frameCx) / frameCx
            let dy = (d.y - frameCy) / frameCy
            let centerDist = min(1.0, sqrt(dx * dx + dy * dy))
            let centerScore = 1.0 - centerDist

            let confTerm = pow(d.confidence, 1.5)
            let areaTerm = pow(areaNorm, 0.8)
            let centerTerm = pow(max(0, centerScore), 0.5)
            let oversized = max(0, areaNorm - primarySelectionSoftAreaNormPivot)
            let largeBoxPenalty = exp(-primarySelectionLargeBoxPenaltyK * oversized)

            let score = confTerm * areaTerm * centerTerm * largeBoxPenalty
            if score > maxScore {
                maxScore = score
                bestIdx = i
            }
        }
        return bestIdx
    }

    /// Shared postprocess after stretch + inference: detection list, NMS, primary selection, and Android-style bbox-limited proto mask (``FurnitureFitOnnxStylePipeline``).
    private func processFrameOnnxStyleCommon(
        processBuffer: CVPixelBuffer,
        arDepthSnapshot: FurnitureFitARDepthSnapshot?,
        frameStart: Date,
        detArray: MLMultiArray,
        protoArray: MLMultiArray,
        modelSide: Int,
        stage2DebugLabel: String
    ) {
        let bufW = CVPixelBufferGetWidth(processBuffer)
        let bufH = CVPixelBufferGetHeight(processBuffer)
        let isLandscape = bufW > bufH
        let onnxSide = modelSide

        let t3 = Date()
        guard let protoInfo = parsePrototypes(protoArray) else {
            if debugMode { logDebug("❌ ONNX-STYLE (\(stage2DebugLabel)): parse prototypes failed") }
            resetProcessingFlag()
            return
        }
        let planes = protoInfo.planes
        let pW = protoInfo.width
        let pH = protoInfo.height

        // Parse the full detector output first. Blacklist filtering is only for primary object selection / mask fusion.
        // Support-surface hints (floor, carpet, tile, etc.) must inspect the unblacklisted detections.
        let rawDetections = YoloEDetectionParser.parseDetections(
            detArray: detArray,
            confidenceThreshold: confidenceThreshold,
            classBlacklist: []
        )
        YoloEDetectionParser.releaseF16Scratch()

        if rawDetections.isEmpty {
            if debugMode { logDebug("⚠️ ONNX-STYLE (\(stage2DebugLabel)): no detections after parse") }
            DispatchQueue.main.async {
                self.maskImageView.image = nil
                self.resetOverlayScalesForEmptyMask()
            }
            resetProcessingFlag()
            return
        }

        var candidates: [FurnitureFitDetection]
        if rawDetections.count > 1 {
            let sorted = rawDetections.sorted { $0.confidence > $1.confidence }
            let capped = Array(sorted.prefix(100))
            candidates = FurnitureFitNMS.apply(detections: capped, iouThreshold: 0.5)
        } else {
            candidates = rawDetections
        }

        let supportSurfaceHintName = supportSurfaceHint(
            from: candidates,
            imageWidth: onnxSide,
            imageHeight: onnxSide
        )
        let preferRoomRaycastSizing = supportSurfaceHintName != nil

        candidates = FurnitureFitFilter.excludingClasses(candidates, blacklist: clsToIgnore)
        if candidates.isEmpty {
            if debugMode { logDebug("⚠️ ONNX-STYLE (\(stage2DebugLabel)): no candidates after blacklist.json filter") }
            DispatchQueue.main.async {
                self.maskImageView.image = nil
                self.resetOverlayScalesForEmptyMask()
            }
            resetProcessingFlag()
            return
        }

        if debugMode {
            logDebug("📦 ONNX-STYLE (\(stage2DebugLabel)) candidates: \(candidates.count) (parse conf≥\(confidenceThreshold), NMS IoU≤0.5, blacklist.json \(clsToIgnore.count) ids)")
            for (i, d) in candidates.prefix(20).enumerated() {
                logDebug("   [\(i)] \(className(d.classIdx)) conf=\(String(format: "%.2f", d.confidence)) ctr=(\(Int(d.x)),\(Int(d.y))) sz=\(Int(d.w))x\(Int(d.h))")
            }
        }

        guard let primaryIdx = selectPrimaryIndexCoreFlow(candidates: candidates, modelSide: onnxSide) else {
            if debugMode { logDebug("⚠️ ONNX-STYLE (\(stage2DebugLabel)): no primary candidate (Core ML STAGE 4 gates)") }
            DispatchQueue.main.async {
                self.maskImageView.image = nil
                self.resetOverlayScalesForEmptyMask()
            }
            resetProcessingFlag()
            return
        }
        let primary = candidates[primaryIdx]
        let maskDetections = FurnitureFitOnnxStylePipeline.collectMaskDetections(primaryIndex: primaryIdx, detections: candidates)

        // Proto mask logits are only evaluated inside each detection bbox; expand **primary** only (explicit `primary`,
        // not positional `[0]`) so legs/wheels get logits. Drop the unexpanded primary from the fusion list via IoU.
        let expandedPrimaryForMask = onnxStyleExpandedPrimaryForMaskBuild(primary, onnxSide: onnxSide)
        let maskDetectionsForBuild = [expandedPrimaryForMask] + maskDetections.filter {
            FurnitureFitIoU.calculate($0, primary) < 0.999 && !clsToIgnore.contains($0.classIdx)
        }

        var maskSmall = FurnitureFitOnnxStylePipeline.buildBboxLimitedSigmoidMask(
            planes: planes,
            protoW: pW,
            protoH: pH,
            modelSide: Float(onnxSide),
            detections: maskDetectionsForBuild
        )
        if furnitureFitUseMorphologicalCloseMask {
            maskSmall = morphologicalBinaryClose3x3Planar8(mask: maskSmall, width: pW, height: pH)
        }

        // Core ML parity: clear proto mask outside **tight** primary rect (model space). Stops secondary-instance
        // fusion from painting into the expanded compositing band beyond the primary box.
        let protoScaleX = Float(pW) / Float(onnxSide)
        let protoScaleY = Float(pH) / Float(onnxSide)
        let protoPx1 = max(0, Int(floor((primary.x - primary.w * 0.5) * protoScaleX)))
        let protoPy1 = max(0, Int(floor((primary.y - primary.h * 0.5) * protoScaleY)))
        let protoPx2 = min(pW, Int(ceil((primary.x + primary.w * 0.5) * protoScaleX)))
        let protoPy2 = min(pH, Int(ceil((primary.y + primary.h * 0.5) * protoScaleY)))
        clipProtoPlanarMaskOutsideRect(
            mask: &maskSmall,
            protoW: pW,
            protoH: pH,
            clipPx1: protoPx1,
            clipPy1: protoPy1,
            clipPx2: protoPx2,
            clipPy2: protoPy2
        )

        let scaleX = Float(bufW) / Float(onnxSide)
        let scaleY = Float(bufH) / Float(onnxSide)
        let tightFx1 = (primary.x - primary.w * 0.5) * scaleX
        let tightFy1 = (primary.y - primary.h * 0.5) * scaleY
        let tightFx2 = (primary.x + primary.w * 0.5) * scaleX
        let tightFy2 = (primary.y + primary.h * 0.5) * scaleY
        let primaryBx1 = max(0, Int(tightFx1))
        let primaryBy1 = max(0, Int(tightFy1))
        let primaryBx2 = min(bufW, Int(tightFx2))
        let primaryBy2 = min(bufH, Int(tightFy2))

        // Compositing band: expand past the tight box. Outside the band we leave pixels transparent, so the live
        // camera shows through — chair bases / wheels often sit below the detector bbox and looked like "floor not masked".
        let bw = max(1, tightFx2 - tightFx1)
        let bh = max(1, tightFy2 - tightFy1)
        let compMarginW = bw * bboxExpandMargin
        let compMarginH = bh * bboxExpandMargin
        let compBx1 = max(0, Int(floor(tightFx1 - compMarginW)))
        let compBy1 = max(0, Int(floor(tightFy1 - compMarginH)))
        let compBx2 = min(bufW, Int(ceil(tightFx2 + compMarginW)))
        let compBy2 = min(bufH, Int(ceil(tightFy2 + compMarginH)))

        if debugMode {
            logDebug("⏱️ ONNX-STYLE (\(stage2DebugLabel)) STAGE 3–5 - parse/NMS/mask: \(String(format: "%.2f", Date().timeIntervalSince(t3) * 1000)) ms")
            logDebug("   🎯 PRIMARY: \(className(primary.classIdx)) bbox [\(primaryBx1),\(primaryBy1)]→[\(primaryBx2),\(primaryBy2)] mask dets=\(maskDetections.count)")
            logDebug("   🖼️ composite band (expanded): [\(compBx1),\(compBy1)]→[\(compBx2),\(compBy2)]")
            logDebug("   ✂️ proto mask clip (tight primary): [\(protoPx1),\(protoPy1)]→[\(protoPx2),\(protoPy2)] (proto \(pW)×\(pH))")
        }

        let normX1 = CGFloat(primaryBx1) / CGFloat(bufW)
        let normY1 = CGFloat(primaryBy1) / CGFloat(bufH)
        let normX2 = CGFloat(primaryBx2) / CGFloat(bufW)
        let normY2 = CGFloat(primaryBy2) / CGFloat(bufH)
        let bboxCenterImageX = CGFloat(primaryBx1 + primaryBx2) / 2
        let bboxCenterImageY = CGFloat(primaryBy1 + primaryBy2) / 2
        let bboxHeightImagePx = Float(max(1, primaryBy2 - primaryBy1))
        let bboxWidthImagePx = Float(max(1, primaryBx2 - primaryBx1))
        let finalMetricResult: PrimaryBboxMetersResult? = nil

        let arLabel: String
        let arTrackingLabel: String
        let planeAnchorCount: Int
        if let frame = arSession.currentFrame {
            switch frame.camera.trackingState {
            case .normal: arTrackingLabel = "normal"
            case .limited(let reason):
                switch reason {
                case .initializing: arTrackingLabel = "limited(init)"
                case .relocalizing: arTrackingLabel = "limited(reloc)"
                case .excessiveMotion: arTrackingLabel = "limited(motion)"
                case .insufficientFeatures: arTrackingLabel = "limited(features)"
                @unknown default: arTrackingLabel = "limited(unknown)"
                }
            case .notAvailable: arTrackingLabel = "notAvailable"
            @unknown default: arTrackingLabel = "unknown"
            }
            planeAnchorCount = frame.anchors.compactMap({ $0 as? ARPlaneAnchor }).count
        } else {
            arTrackingLabel = "noFrame"
            planeAnchorCount = 0
        }

        if hasARKitAssistedSizingPayload && arAssistedScaleValid {
            arLabel = "valid(\(String(format: "%.3f", autoScaleFromAR)))"
        } else if hasARKitAssistedSizingPayload {
            arLabel = "pending"
        } else {
            arLabel = "off"
        }

        let raycastSuStr: String
        if let r = roomRaycastSceneDimensions {
            raycastSuStr =
                "\(String(format: "%.3f", r.width))×\(String(format: "%.3f", r.height))×\(String(format: "%.3f", r.depth))"
        } else {
            raycastSuStr = "—"
        }

        logFurnitureFitSize(
            "phase=all " +
            "ar=\(arLabel) tracking=\(arTrackingLabel) planes=\(planeAnchorCount) " +
            "room_display_m=\(String(format: "%.2f", roomWidthMeters))×\(String(format: "%.2f", roomHeightMeters))×\(String(format: "%.2f", roomDepthMeters)) " +
            "raycast_su=\(raycastSuStr) " +
            "bbox=\(Int(bboxWidthImagePx))×\(Int(bboxHeightImagePx))px buf=\(bufW)×\(bufH) " +
            "ratioW=\(String(format: "%.3f", bboxWidthImagePx / Float(max(1, bufW)))) ratioH=\(String(format: "%.3f", bboxHeightImagePx / Float(max(1, bufH))))"
        )

        let auditNow = CFAbsoluteTimeGetCurrent()
        if auditNow - lastPinholeDepthAuditAt >= 1.75 {
            lastPinholeDepthAuditAt = auditNow
            let deviceLidar = QualitySettings.supportsLiDARSceneDepth
            let snapHasMap = arDepthSnapshot?.depthMap != nil
            let hasArFrame = arSession.currentFrame != nil
            let auditTracking: String = {
                guard let frame = arSession.currentFrame else { return "no_frame" }
                switch frame.camera.trackingState {
                case .normal: return "normal"
                case .limited: return "limited"
                case .notAvailable: return "notAvailable"
                @unknown default: return "unknown"
                }
            }()
            logFurnitureFitSize(
                "phase=floor_contact_audit device_has_LiDAR=\(deviceLidar) ar_depth_companion_running=\(isARDepthCompanionSessionRunning) " +
                "snapshot_depthMap_present=\(snapHasMap) ar_currentFrame=\(hasArFrame) tracking=\(auditTracking) " +
                "pipeline=disabled dist_used_m=— → non_AR_measurement_disabled"
            )
        }

        let maskHasForeground = maskSmall.contains(where: { $0 > 0 })
        let firstFrameMeters: (width: Float, height: Float, pipeline: String, dist: Float?)? = nil

        if isUsingARCameraPath {
            scheduleDebouncedAssistedMeasurement(
                primaryClassIdx: primary.classIdx,
                bboxMinX: primaryBx1,
                bboxMinY: primaryBy1,
                bboxMaxX: primaryBx2,
                bboxMaxY: primaryBy2,
                bboxCenterImageX: bboxCenterImageX,
                bboxCenterImageY: bboxCenterImageY,
                bboxHeightImagePx: bboxHeightImagePx,
                imageWidth: bufW,
                imageHeight: bufH,
                arDepthSnapshot: arDepthSnapshot
            )
        } else {
            DispatchQueue.main.async { [weak self] in
                self?.updateAssistedOverlayScale(
                    primaryClassIdx: primary.classIdx,
                    bboxMinX: primaryBx1,
                    bboxMinY: primaryBy1,
                    bboxMaxX: primaryBx2,
                    bboxMaxY: primaryBy2,
                    bboxCenterImageX: bboxCenterImageX,
                    bboxCenterImageY: bboxCenterImageY,
                    bboxHeightImagePx: bboxHeightImagePx,
                    imageWidth: bufW,
                    imageHeight: bufH,
                    arDepthSnapshot: arDepthSnapshot
                )
            }
        }
        DispatchQueue.main.async {
            let viewW = self.maskImageView.bounds.width
            let viewH = self.maskImageView.bounds.height
            self.primaryBboxInView = CGRect(
                x: normX1 * viewW,
                y: normY1 * viewH,
                width: (normX2 - normX1) * viewW,
                height: (normY2 - normY1) * viewH
            )
            self.updateAutoScaleFromRoom(
                primaryBx1: primaryBx1,
                primaryBy1: primaryBy1,
                primaryBx2: primaryBx2,
                primaryBy2: primaryBy2,
                imageWidth: bufW,
                imageHeight: bufH,
                arDepthSnapshot: arDepthSnapshot,
                preferRoomRaycastSizing: preferRoomRaycastSizing
            )
            self.updateOverlayPresentationMode(
                primaryClassIdx: primary.classIdx,
                metric: finalMetricResult
            )
            UIView.animate(withDuration: 0.18) {
                self.applyCurrentOverlayScaleTransform()
            }
        }

        setProgress(0.92, text: "Compositing…")
        let compStart = Date()
        // Proto mask → full frame: avoid `kvImageHighQualityResampling` (extra-soft mask → mushy cutout vs camera preview).
        let composedImage = compositeCpuBilinearProtoMaskCutout(
            processBuffer: processBuffer,
            maskProto: maskSmall,
            protoW: pW,
            protoH: pH,
            origW: bufW,
            origH: bufH,
            x0: compBx1,
            x1: compBx2,
            y0: compBy1,
            y1: compBy2,
            debugTag: "ONNX-style CPU composite"
        )

        let withDebugOverlay: CGImage? = {
            guard debugMode,
                  let base = composedImage else { return composedImage }
            return drawOnnxStyleDebugDetectionBboxesOnComposedImage(
                composed: base,
                candidates: candidates,
                primaryIndex: primaryIdx,
                origW: bufW,
                origH: bufH,
                scaleX: scaleX,
                scaleY: scaleY
            ) ?? base
        }()

        let bboxWidthPxForFinish = Int(max(1, primaryBx2 - primaryBx1))
        let bboxHeightPxForFinish = Int(max(1, primaryBy2 - primaryBy1))
        let arSnapAttached = arDepthSnapshot != nil

        if let finalImage = withDebugOverlay {
            let needsRotate = isLandscape && !self.isUsingARCameraPath && self.lockedOrientation != .landscape
            DispatchQueue.main.async { [weak self] in
                guard let self else { return }
                var out = finalImage
                if needsRotate {
                    if let r = self.rotateCGImage90(out, clockwise: true) { out = r }
                }
                let scale = self.window?.windowScene?.screen.scale ?? self.traitCollection.displayScale
                self.maskImageView.image = UIImage(cgImage: out, scale: scale, orientation: .up)
                self.scheduleSegmentationMeanColorPublishIfNeeded(compositedCgImage: out)
                self.commitPinholeFurnitureSizeAfterSegmentationMaskApplied(
                    maskHasForeground: maskHasForeground,
                    pinholeResult: finalMetricResult,
                    firstFrameMeters: firstFrameMeters,
                    imageWidth: bufW,
                    imageHeight: bufH,
                    bboxWidthPx: bboxWidthPxForFinish,
                    bboxHeightPx: bboxHeightPxForFinish,
                    arDepthSnapshotAttached: arSnapAttached
                )
            }
        } else if maskHasForeground {
            DispatchQueue.main.async { [weak self] in
                guard let self else { return }
                self.commitPinholeFurnitureSizeAfterSegmentationMaskApplied(
                    maskHasForeground: maskHasForeground,
                    pinholeResult: finalMetricResult,
                    firstFrameMeters: firstFrameMeters,
                    imageWidth: bufW,
                    imageHeight: bufH,
                    bboxWidthPx: bboxWidthPxForFinish,
                    bboxHeightPx: bboxHeightPxForFinish,
                    arDepthSnapshotAttached: arSnapAttached
                )
            }
        }

        resetProcessingFlag()
        if debugMode {
            let compMs = Date().timeIntervalSince(compStart) * 1000
            let frameTotalMs = Date().timeIntervalSince(frameStart) * 1000
            logDebug("🖼️ ONNX-STYLE (\(stage2DebugLabel)) compositing: \(String(format: "%.2f", compMs)) ms")
            logMemory("FRAME END")
            logDebug("⏱️ ONNX-STYLE (\(stage2DebugLabel)) FRAME TOTAL: \(String(format: "%.2f", frameTotalMs)) ms")
            logDebug("⏱️ ═══════════════════════════════════════════\n")
        }
    }

    /// Expands primary width/height for ONNX-style proto mask synthesis (legs/wheels), clamped to model square.
    private func onnxStyleExpandedPrimaryForMaskBuild(_ primary: FurnitureFitDetection, onnxSide: Int) -> FurnitureFitDetection {
        let side = Float(onnxSide)
        let maxHalfW = min(primary.x, side - primary.x)
        let maxHalfH = min(primary.y, side - primary.y)
        let capW = 2 * max(maxHalfW, 1)
        let capH = 2 * max(maxHalfH, 1)
        let ew = min(primary.w * (1 + 2 * bboxExpandMargin), capW)
        let eh = min(primary.h * (1 + 2 * bboxExpandMargin), capH)
        return FurnitureFitDetection(
            x: primary.x,
            y: primary.y,
            w: ew,
            h: eh,
            confidence: primary.confidence,
            classIdx: primary.classIdx,
            coeffs: primary.coeffs
        )
    }

    /// Zeros planar proto mask outside [clipPx1, clipPx2) × [clipPy1, clipPy2) (same idea as Core ML stage 15d clip in full-res).
    private func clipProtoPlanarMaskOutsideRect(
        mask: inout [UInt8],
        protoW: Int,
        protoH: Int,
        clipPx1: Int,
        clipPy1: Int,
        clipPx2: Int,
        clipPy2: Int
    ) {
        let clipX1 = max(0, clipPx1)
        let clipY1 = max(0, clipPy1)
        let clipX2 = min(protoW, clipPx2)
        let clipY2 = min(protoH, clipPy2)
        guard mask.count >= protoW * protoH else { return }
        guard clipX1 < clipX2, clipY1 < clipY2 else {
            mask.withUnsafeMutableBytes { raw in
                if let base = raw.baseAddress { memset(base, 0, protoW * protoH) }
            }
            return
        }
        if clipY1 > 0 {
            let topBytes = clipY1 * protoW
            mask.withUnsafeMutableBytes { raw in
                if let base = raw.baseAddress { memset(base, 0, topBytes) }
            }
        }
        if clipY2 < protoH {
            let bottomStart = clipY2 * protoW
            let bottomBytes = (protoH - clipY2) * protoW
            mask.withUnsafeMutableBytes { raw in
                if let base = raw.baseAddress {
                    memset(base.advanced(by: bottomStart), 0, bottomBytes)
                }
            }
        }
        if clipX1 > 0 || clipX2 < protoW {
            mask.withUnsafeMutableBytes { raw in
                guard let base = raw.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }
                for y in clipY1..<clipY2 {
                    let rowStart = y * protoW
                    if clipX1 > 0 {
                        memset(base.advanced(by: rowStart), 0, clipX1)
                    }
                    if clipX2 < protoW {
                        memset(base.advanced(by: rowStart + clipX2), 0, protoW - clipX2)
                    }
                }
            }
        }
    }

    private func clipFullMaskToPrimaryBbox(
        mask: inout [UInt8],
        origW: Int,
        origH: Int,
        px1: Int,
        py1: Int,
        px2: Int,
        py2: Int
    ) {
        let clipPx1 = max(0, px1)
        let clipPy1 = max(0, py1)
        let clipPx2 = min(origW, px2)
        let clipPy2 = min(origH, py2)
        if clipPy1 > 0 {
            let topBytes = clipPy1 * origW
            mask.withUnsafeMutableBytes { raw in
                if let base = raw.baseAddress { memset(base, 0, topBytes) }
            }
        }
        if clipPy2 < origH {
            let bottomStart = clipPy2 * origW
            let bottomBytes = (origH - clipPy2) * origW
            mask.withUnsafeMutableBytes { raw in
                if let base = raw.baseAddress {
                    memset(base.advanced(by: bottomStart), 0, bottomBytes)
                }
            }
        }
        if clipPx1 > 0 || clipPx2 < origW {
            mask.withUnsafeMutableBytes { raw in
                guard let base = raw.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }
                for y in clipPy1..<clipPy2 {
                    let rowStart = y * origW
                    if clipPx1 > 0 {
                        memset(base.advanced(by: rowStart), 0, clipPx1)
                    }
                    if clipPx2 < origW {
                        memset(base.advanced(by: rowStart + clipPx2), 0, origW - clipPx2)
                    }
                }
            }
        }
    }

    /// BGRA camera × (mask/255) → premultiplied RGBA inside the primary band using vDSP (`vfltu8`, `vsmul`, `vmul`, `vclip`, `vadd`, `vfixu8`).
    private func compositeCpuCameraBandAccelerated(
        outBase: UnsafeMutablePointer<UInt8>,
        origBase: UnsafePointer<UInt8>,
        maskBase: UnsafePointer<UInt8>,
        origW: Int,
        camRowBytes: Int,
        bytesPerRowOut: Int,
        x0: Int,
        x1: Int,
        y0: Int,
        y1: Int
    ) {
        let w = x1 - x0
        guard w > 0, y1 > y0 else { return }
        let need = 5 * w
        if compositeCpuScratchFloats.count < need {
            compositeCpuScratchFloats = [Float](repeating: 0, count: need)
        }
        compositeCpuScratchFloats.withUnsafeMutableBufferPointer { scratchBuf in
            guard let scratch = scratchBuf.baseAddress else { return }
            let alphaPtr = scratch
            let bPtr = scratch.advanced(by: w)
            let gPtr = scratch.advanced(by: 2 * w)
            let rPtr = scratch.advanced(by: 3 * w)
            let halfPtr = scratch.advanced(by: 4 * w)
            for hi in 0..<w {
                halfPtr[hi] = 0.5
            }
            var low: Float = 0
            var high: Float = 255
            var inv255: Float = 1.0 / 255.0
            let wl = vDSP_Length(w)
            for y in y0..<y1 {
                let outRowPtr = outBase.advanced(by: y * bytesPerRowOut)
                let origRowPtr = origBase.advanced(by: y * camRowBytes)
                let maskRowPtr = maskBase.advanced(by: y * origW + x0)
                vDSP_vfltu8(maskRowPtr, 1, alphaPtr, 1, wl)
                vDSP_vsmul(alphaPtr, 1, &inv255, alphaPtr, 1, wl)
                vDSP_vfltu8(origRowPtr.advanced(by: x0 * 4 + 0), 4, bPtr, 1, wl)
                vDSP_vfltu8(origRowPtr.advanced(by: x0 * 4 + 1), 4, gPtr, 1, wl)
                vDSP_vfltu8(origRowPtr.advanced(by: x0 * 4 + 2), 4, rPtr, 1, wl)
                vDSP_vmul(bPtr, 1, alphaPtr, 1, bPtr, 1, wl)
                vDSP_vmul(gPtr, 1, alphaPtr, 1, gPtr, 1, wl)
                vDSP_vmul(rPtr, 1, alphaPtr, 1, rPtr, 1, wl)
                vDSP_vclip(bPtr, 1, &low, &high, bPtr, 1, wl)
                vDSP_vclip(gPtr, 1, &low, &high, gPtr, 1, wl)
                vDSP_vclip(rPtr, 1, &low, &high, rPtr, 1, wl)
                vDSP_vadd(bPtr, 1, halfPtr, 1, bPtr, 1, wl)
                vDSP_vadd(gPtr, 1, halfPtr, 1, gPtr, 1, wl)
                vDSP_vadd(rPtr, 1, halfPtr, 1, rPtr, 1, wl)
                vDSP_vfixu8(rPtr, 1, outRowPtr.advanced(by: x0 * 4 + 0), 4, wl)
                vDSP_vfixu8(gPtr, 1, outRowPtr.advanced(by: x0 * 4 + 1), 4, wl)
                vDSP_vfixu8(bPtr, 1, outRowPtr.advanced(by: x0 * 4 + 2), 4, wl)
                var mx = 0
                while mx < w {
                    outRowPtr.advanced(by: (x0 + mx) * 4 + 3).pointee = maskRowPtr[mx]
                    mx += 1
                }
            }
        }
    }

    /// Logs premultiplied-last R, G, B, A at the center of the primary band (throttled when `debugMode` is on).
    private func logPremultipliedRGBASampleIfDebug(
        outBase: UnsafeMutablePointer<UInt8>,
        bytesPerRowOut: Int,
        width: Int,
        height: Int,
        x0: Int,
        x1: Int,
        y0: Int,
        y1: Int,
        tag: String
    ) {
        guard debugMode else { return }
        let now = Date().timeIntervalSince1970
        if now - lastCompositePremulRGBAlogAt < compositePremulRGBAlogInterval { return }
        lastCompositePremulRGBAlogAt = now
        guard width > 0, height > 0, x1 > x0, y1 > y0 else {
            logDebug("🎨 [RGBA \(tag)] invalid bbox; skip sample")
            return
        }
        let cx = (x0 + x1) / 2
        let cy = (y0 + y1) / 2
        guard cx >= 0, cx < width, cy >= 0, cy < height else { return }
        let offset = cy * bytesPerRowOut + cx * 4
        let red = outBase[offset]
        let green = outBase[offset + 1]
        let blue = outBase[offset + 2]
        let alpha = outBase[offset + 3]
        logDebug("🎨 [RGBA \(tag)] center (\(cx),\(cy)) premul R=\(red) G=\(green) B=\(blue) A=\(alpha)")
    }

    /// ONNX-style cutout: vImage **Planar8 scale** proto→full frame (one fast resize), then scan composite band only.
    /// Smoothstep on mask reduces bilinear “mush” and frame-to-frame edge flicker vs per-pixel bilinear in a huge band.
    private func compositeCpuBilinearProtoMaskCutout(
        processBuffer: CVPixelBuffer,
        maskProto: [UInt8],
        protoW: Int,
        protoH: Int,
        origW: Int,
        origH: Int,
        x0: Int,
        x1: Int,
        y0: Int,
        y1: Int,
        debugTag: String
    ) -> CGImage? {
        let pw = protoW
        let ph = protoH
        guard pw > 0, ph > 0, maskProto.count >= pw * ph, origW > 0, origH > 0 else { return nil }
        let xStart = max(0, min(origW, x0))
        let xEnd = max(0, min(origW, x1))
        let yStart = max(0, min(origH, y0))
        let yEnd = max(0, min(origH, y1))
        guard xStart < xEnd, yStart < yEnd else { return nil }

        let fullPixels = origW * origH
        if upscaledPlanarMaskScratch.count < fullPixels {
            upscaledPlanarMaskScratch = [UInt8](repeating: 0, count: fullPixels)
        }

        let scaleErr: vImage_Error = maskProto.withUnsafeBufferPointer { srcPtr in
            guard let srcBase = srcPtr.baseAddress else { return kvImageNullPointerArgument }
            var srcBuf = vImage_Buffer(
                data: UnsafeMutableRawPointer(mutating: srcBase),
                height: vImagePixelCount(ph),
                width: vImagePixelCount(pw),
                rowBytes: pw
            )
            return upscaledPlanarMaskScratch.withUnsafeMutableBufferPointer { dstPtr in
                guard let dstBase = dstPtr.baseAddress else { return kvImageNullPointerArgument }
                var dstBuf = vImage_Buffer(
                    data: dstBase,
                    height: vImagePixelCount(origH),
                    width: vImagePixelCount(origW),
                    rowBytes: origW
                )
                return vImageScale_Planar8(&srcBuf, &dstBuf, nil, vImage_Flags(0))
            }
        }
        if scaleErr != kvImageNoError {
            if debugMode {
                logDebug("⚠️ ONNX-style vImageScale_Planar8 failed (\(scaleErr)); compositing skipped")
            }
            return nil
        }

        guard let ctx = CGContext(
            data: nil,
            width: origW,
            height: origH,
            bitsPerComponent: 8,
            bytesPerRow: origW * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { return nil }

        CVPixelBufferLockBaseAddress(processBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(processBuffer, .readOnly) }
        guard let origBase = CVPixelBufferGetBaseAddress(processBuffer)?.assumingMemoryBound(to: UInt8.self) else { return nil }
        let camRowBytes = CVPixelBufferGetBytesPerRow(processBuffer)
        guard let outData = ctx.data else { return nil }
        let outBase = outData.assumingMemoryBound(to: UInt8.self)
        let bytesPerRowOut = origW * 4

        for y in 0..<origH {
            memset(outBase.advanced(by: y * bytesPerRowOut), 0, bytesPerRowOut)
        }

        let rgbDim: Float = 1.0
        let inv255: Float = 1.0 / 255.0
        // Tighter band than before: full opacity sooner inside the mask, thinner feather at the edge.
        let edgeLo: Float = 0.42
        let invEdgeWidth: Float = 1.0 / max(0.12, 0.58 - edgeLo)

        let maskFull = upscaledPlanarMaskScratch
        for y in yStart..<yEnd {
            let rowOff = y * origW
            let outRow = outBase.advanced(by: y * bytesPerRowOut)
            let camRow = origBase.advanced(by: y * camRowBytes)
            for x in xStart..<xEnd {
                let raw = Float(maskFull[rowOff + x]) * inv255
                let t = (raw - edgeLo) * invEdgeWidth
                let s = min(1, max(0, t))
                let sm = s * s * (3 - 2 * s)
                guard sm > 0.008 else { continue }
                let cx = x * 4
                let rf = Float(camRow[cx + 2]) * sm * rgbDim
                let gf = Float(camRow[cx + 1]) * sm * rgbDim
                let bf = Float(camRow[cx]) * sm * rgbDim
                outRow[cx] = UInt8(min(255, max(0, rf + 0.5)))
                outRow[cx + 1] = UInt8(min(255, max(0, gf + 0.5)))
                outRow[cx + 2] = UInt8(min(255, max(0, bf + 0.5)))
                outRow[cx + 3] = UInt8(min(255, max(0, sm * 255 + 0.5)))
            }
        }

        logPremultipliedRGBASampleIfDebug(
            outBase: outBase,
            bytesPerRowOut: bytesPerRowOut,
            width: origW,
            height: origH,
            x0: xStart,
            x1: xEnd,
            y0: yStart,
            y1: yEnd,
            tag: debugTag
        )
        return ctx.makeImage()
    }

    /// ONNX-style stretch coords → buffer pixels; CGContext stroke uses bottom-left origin, so flip Y like STAGE 7 letterbox debug.
    private func drawOnnxStyleDebugDetectionBboxesOnComposedImage(
        composed: CGImage,
        candidates: [FurnitureFitDetection],
        primaryIndex: Int,
        origW: Int,
        origH: Int,
        scaleX: Float,
        scaleY: Float
    ) -> CGImage? {
        let bytesPerRow = origW * 4
        guard let ctx = CGContext(
            data: nil,
            width: origW,
            height: origH,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { return nil }

        ctx.draw(composed, in: CGRect(x: 0, y: 0, width: origW, height: origH))

        func bufferRect(for d: FurnitureFitDetection) -> (bx1: Int, by1: Int, bw: Int, bh: Int) {
            let tightFx1 = (d.x - d.w * 0.5) * scaleX
            let tightFy1 = (d.y - d.h * 0.5) * scaleY
            let tightFx2 = (d.x + d.w * 0.5) * scaleX
            let tightFy2 = (d.y + d.h * 0.5) * scaleY
            let bx1 = max(0, Int(tightFx1))
            let by1 = max(0, Int(tightFy1))
            let bx2 = min(origW, Int(tightFx2))
            let by2 = min(origH, Int(tightFy2))
            let bw = max(1, bx2 - bx1)
            let bh = max(1, by2 - by1)
            return (bx1, by1, bw, bh)
        }

        let font = CTFontCreateWithName("Helvetica-Bold" as CFString, 36, nil)

        for (index, d) in candidates.enumerated() {
            let r = bufferRect(for: d)
            let cgRect = CGRect(x: r.bx1, y: origH - r.by1 - r.bh, width: r.bw, height: r.bh)
            ctx.setLineWidth(2.0)
            ctx.setStrokeColor(UIColor.cyan.cgColor)
            ctx.stroke(cgRect)

            guard index == primaryIndex else { continue }
            let plainName = classNames[d.classIdx] ?? "unknown"
            let confidence = String(format: "%.2f", d.confidence)
            let labelText = "\(plainName) (\(confidence))"
            let attributes: [NSAttributedString.Key: Any] = [
                .font: font,
                .foregroundColor: UIColor.white
            ]
            let attributedString = NSAttributedString(string: labelText, attributes: attributes)
            let line = CTLineCreateWithAttributedString(attributedString)
            let textBounds = CTLineGetBoundsWithOptions(line, .useOpticalBounds)
            let labelX = CGFloat(r.bx1)
            let labelY = CGFloat(origH - r.by1 + 4)
            let textBackgroundRect = CGRect(
                x: labelX - 2,
                y: labelY - textBounds.height - 2,
                width: textBounds.width + 4,
                height: textBounds.height + 4
            )
            ctx.setFillColor(UIColor.black.withAlphaComponent(0.7).cgColor)
            ctx.fill(textBackgroundRect)
            ctx.saveGState()
            ctx.textMatrix = .identity
            ctx.translateBy(x: labelX, y: labelY - textBounds.height)
            ctx.setFillColor(UIColor.white.cgColor)
            CTLineDraw(line, ctx)
            ctx.restoreGState()
        }

        if primaryIndex >= 0, primaryIndex < candidates.count {
            let d = candidates[primaryIndex]
            let r = bufferRect(for: d)
            let cgRect = CGRect(x: r.bx1, y: origH - r.by1 - r.bh, width: r.bw, height: r.bh)
            ctx.setStrokeColor(UIColor.red.cgColor)
            ctx.setLineWidth(4.0)
            ctx.stroke(cgRect)
        }

        return ctx.makeImage()
    }

    private func compositeCpuMaskOnly(
        processBuffer: CVPixelBuffer,
        maskFull: [UInt8],
        origW: Int,
        origH: Int,
        primaryBx1: Int,
        primaryBy1: Int,
        primaryBx2: Int,
        primaryBy2: Int
    ) -> CGImage? {
        let ctx = CGContext(
            data: nil,
            width: origW,
            height: origH,
            bitsPerComponent: 8,
            bytesPerRow: origW * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        )!
        CVPixelBufferLockBaseAddress(processBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(processBuffer, .readOnly) }
        guard let origBase = CVPixelBufferGetBaseAddress(processBuffer)?.assumingMemoryBound(to: UInt8.self) else { return nil }
        let bytesPerRowCam = CVPixelBufferGetBytesPerRow(processBuffer)
        let outBase = ctx.data!.assumingMemoryBound(to: UInt8.self)
        let bytesPerRowOut = origW * 4
        let x0 = max(0, primaryBx1)
        let x1 = min(origW, primaryBx2)
        let y0 = max(0, primaryBy1)
        let y1 = min(origH, primaryBy2)
        for y in 0..<origH {
            memset(outBase.advanced(by: y * bytesPerRowOut), 0, bytesPerRowOut)
        }
        maskFull.withUnsafeBufferPointer { maskBuf in
            guard let maskBase = maskBuf.baseAddress else { return }
            compositeCpuCameraBandAccelerated(
                outBase: outBase,
                origBase: origBase,
                maskBase: maskBase,
                origW: origW,
                camRowBytes: bytesPerRowCam,
                bytesPerRowOut: bytesPerRowOut,
                x0: x0,
                x1: x1,
                y0: y0,
                y1: y1
            )
        }
        logPremultipliedRGBASampleIfDebug(
            outBase: outBase,
            bytesPerRowOut: bytesPerRowOut,
            width: origW,
            height: origH,
            x0: x0,
            x1: x1,
            y0: y0,
            y1: y1,
            tag: "ONNX-style CPU composite"
        )
        return ctx.makeImage()
    }

    /// Stretch to fill `size`×`size` (matches Android ONNX `createScaledBitmap` preprocessing).
    private func resizeStretchToSquare(_ src: CVPixelBuffer, size: Int) -> CVPixelBuffer? {
        CVPixelBufferLockBaseAddress(src, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(src, .readOnly) }

        let srcW = CVPixelBufferGetWidth(src)
        let srcH = CVPixelBufferGetHeight(src)

        if cachedStretchSize != size || cachedStretchBuffer == nil {
            var newBuffer: CVPixelBuffer?
            guard CVPixelBufferCreate(nil, size, size, kCVPixelFormatType_32BGRA, nil, &newBuffer) == kCVReturnSuccess,
                  let buf = newBuffer else { return nil }
            cachedStretchBuffer = buf
            cachedStretchSize = size
        }

        guard let dst = cachedStretchBuffer else { return nil }
        CVPixelBufferLockBaseAddress(dst, [])
        defer { CVPixelBufferUnlockBaseAddress(dst, []) }

        guard let srcBase = CVPixelBufferGetBaseAddress(src),
              let dstBase = CVPixelBufferGetBaseAddress(dst) else { return nil }

        var srcBuffer = vImage_Buffer(
            data: srcBase,
            height: vImagePixelCount(srcH),
            width: vImagePixelCount(srcW),
            rowBytes: CVPixelBufferGetBytesPerRow(src)
        )
        var dstBuffer = vImage_Buffer(
            data: dstBase,
            height: vImagePixelCount(size),
            width: vImagePixelCount(size),
            rowBytes: CVPixelBufferGetBytesPerRow(dst)
        )
        guard vImageScale_ARGB8888(&srcBuffer, &dstBuffer, nil, vImage_Flags(kvImageHighQualityResampling)) == kvImageNoError else {
            return nil
        }
        return dst
    }

    private func processFrameInner(_ pixelBuffer: CVPixelBuffer, arDepthSnapshot: FurnitureFitARDepthSnapshot? = nil) {
        let frameStart = Date()
        if debugMode { logMemory("FRAME START") }
        loadBlacklistOnce()

        guard let model = mlModel else {
            resetProcessingFlag()
            return
        }
        processFrameOnnxStyleCoreML(
            processBuffer: pixelBuffer,
            model: model,
            arDepthSnapshot: arDepthSnapshot,
            frameStart: frameStart
        )
    }


    // MARK: - Parse Prototypes (FIXED: Accelerate for Float16)
    // Reuses instance-level buffers and uses Accelerate for FP16 → FP32 conversion.
    private func parsePrototypes(_ proto: MLMultiArray) -> (planes: [Float], count: Int, height: Int, width: Int)? {
        var shape = proto.shape.map { $0.intValue }
        if shape.count == 4 && shape[0] == 1 { shape.removeFirst() }
        guard shape.count == 3 else { return nil }

        let cIdx: Int
        if shape[0] == 32 { cIdx = 0 }
        else if shape[2] == 32 { cIdx = 2 }
        else { cIdx = shape.firstIndex(of: 32) ?? -1 }
        guard cIdx >= 0 else { return nil }

        let count = 32
        let h: Int, w: Int
        if cIdx == 0 { h = shape[1]; w = shape[2] }
        else { h = shape[0]; w = shape[1] }

        let planeSize = h * w
        let total = shape[0] * shape[1] * shape[2]

        guard proto.dataType == .float32 || proto.dataType == .float16 else { return nil }

        if protoRawFloats.count != total {
            protoRawFloats = [Float](repeating: 0, count: total)
        }
        if protoPlanes.count != count * planeSize {
            protoPlanes = [Float](repeating: 0, count: count * planeSize)
        }

        if proto.dataType == .float32 {
            protoRawFloats.withUnsafeMutableBufferPointer { dst in
                guard let dstBase = dst.baseAddress else { return }
                memcpy(dstBase, proto.dataPointer, total * MemoryLayout<Float>.size)
            }
        } else {
            // Float16 path: use Accelerate vImage for bulk conversion wherever strides are friendly.
            let strides = proto.strides.map { $0.intValue }
            let dims = proto.shape.map { $0.intValue }
            guard dims.count == strides.count else { return nil }

            let src16 = proto.dataPointer.bindMemory(to: UInt16.self, capacity: proto.count)
            let innerStride = strides.last ?? 1

            if innerStride == 1 {
                // Fast path: fully contiguous layout
                let isFullyContiguous: Bool = {
                    if dims.count == 3 {
                        return strides[2] == 1 &&
                               strides[1] == dims[2] &&
                               strides[0] == dims[1] * dims[2]
                    } else if dims.count == 4 {
                        return strides[3] == 1 &&
                               strides[2] == dims[3] &&
                               strides[1] == dims[2] * dims[3] &&
                               strides[0] == dims[1] * dims[2] * dims[3]
                    }
                    return false
                }()

                if isFullyContiguous {
                    // Single bulk FP16 → FP32 conversion.
                    protoRawFloats.withUnsafeMutableBufferPointer { dst in
                        guard let dstBase = dst.baseAddress else { return }
                        var srcBuf = vImage_Buffer(
                            data: UnsafeMutableRawPointer(mutating: src16),
                            height: 1,
                            width: vImagePixelCount(total),
                            rowBytes: total * MemoryLayout<UInt16>.size
                        )
                        var dstBuf = vImage_Buffer(
                            data: dstBase,
                            height: 1,
                            width: vImagePixelCount(total),
                            rowBytes: total * MemoryLayout<Float>.size
                        )
                        vImageConvert_Planar16FtoPlanarF(&srcBuf, &dstBuf, vImage_Flags(kvImageNoFlags))
                    }
                } else {
                    // Per-plane contiguous: each plane is packed even if outer dims are strided.
                    if dims.count == 4, dims[0] == 1 {
                        let cMax = dims[1], yMax = dims[2], xMax = dims[3]
                        let planePixels = yMax * xMax
                        for c in 0..<cMax {
                            let srcOffset = c * strides[1]
                            protoRawFloats.withUnsafeMutableBufferPointer { dst in
                                guard let dstBase = dst.baseAddress else { return }
                                let dstStart = dstBase.advanced(by: c * planePixels)
                                var srcBuf = vImage_Buffer(
                                    data: UnsafeMutableRawPointer(mutating: src16.advanced(by: srcOffset)),
                                    height: 1,
                                    width: vImagePixelCount(planePixels),
                                    rowBytes: planePixels * MemoryLayout<UInt16>.size
                                )
                                var dstBuf = vImage_Buffer(
                                    data: dstStart,
                                    height: 1,
                                    width: vImagePixelCount(planePixels),
                                    rowBytes: planePixels * MemoryLayout<Float>.size
                                )
                                vImageConvert_Planar16FtoPlanarF(&srcBuf, &dstBuf, vImage_Flags(kvImageNoFlags))
                            }
                        }
                    } else if dims.count == 3 {
                        let cMax = dims[0]
                        let planePixels = dims[1] * dims[2]
                        for c in 0..<cMax {
                            let srcOffset = c * strides[0]
                            protoRawFloats.withUnsafeMutableBufferPointer { dst in
                                guard let dstBase = dst.baseAddress else { return }
                                let dstStart = dstBase.advanced(by: c * planePixels)
                                var srcBuf = vImage_Buffer(
                                    data: UnsafeMutableRawPointer(mutating: src16.advanced(by: srcOffset)),
                                    height: 1,
                                    width: vImagePixelCount(planePixels),
                                    rowBytes: planePixels * MemoryLayout<UInt16>.size
                                )
                                var dstBuf = vImage_Buffer(
                                    data: dstStart,
                                    height: 1,
                                    width: vImagePixelCount(planePixels),
                                    rowBytes: planePixels * MemoryLayout<Float>.size
                                )
                                vImageConvert_Planar16FtoPlanarF(&srcBuf, &dstBuf, vImage_Flags(kvImageNoFlags))
                            }
                        }
                    } else {
                        return nil
                    }
                }
            } else {
                // Non-contiguous fallback: convert row-by-row when inner dim is contiguous,
                // fall back to scalar only when truly necessary (rare).
                if dims.count == 4, dims[0] == 1 {
                    let cMax = dims[1], yMax = dims[2], xMax = dims[3]
                    for c in 0..<cMax {
                        for y in 0..<yMax {
                            let srcOffset = c * strides[1] + y * strides[2]
                            let dstOffset = c * (yMax * xMax) + y * xMax
                            if strides.count > 3 && strides[3] == 1 {
                                protoRawFloats.withUnsafeMutableBufferPointer { dst in
                                    guard let dstBase = dst.baseAddress else { return }
                                    var srcBuf = vImage_Buffer(
                                        data: UnsafeMutableRawPointer(mutating: src16.advanced(by: srcOffset)),
                                        height: 1,
                                        width: vImagePixelCount(xMax),
                                        rowBytes: xMax * MemoryLayout<UInt16>.size
                                    )
                                    var dstBuf = vImage_Buffer(
                                        data: dstBase.advanced(by: dstOffset),
                                        height: 1,
                                        width: vImagePixelCount(xMax),
                                        rowBytes: xMax * MemoryLayout<Float>.size
                                    )
                                    vImageConvert_Planar16FtoPlanarF(&srcBuf, &dstBuf, vImage_Flags(kvImageNoFlags))
                                }
                            } else {
                                for x in 0..<xMax {
                                    let srcOff = srcOffset + x * strides[3]
                                    protoRawFloats[dstOffset + x] = Float(Float16(bitPattern: src16[srcOff]))
                                }
                            }
                        }
                    }
                } else if dims.count == 3 {
                    let d0 = dims[0], d1 = dims[1], d2 = dims[2]
                    for i in 0..<d0 {
                        for j in 0..<d1 {
                            let srcOffset = i * strides[0] + j * strides[1]
                            let dstOffset = i * (d1 * d2) + j * d2
                            if strides[2] == 1 {
                                protoRawFloats.withUnsafeMutableBufferPointer { dst in
                                    guard let dstBase = dst.baseAddress else { return }
                                    var srcBuf = vImage_Buffer(
                                        data: UnsafeMutableRawPointer(mutating: src16.advanced(by: srcOffset)),
                                        height: 1,
                                        width: vImagePixelCount(d2),
                                        rowBytes: d2 * MemoryLayout<UInt16>.size
                                    )
                                    var dstBuf = vImage_Buffer(
                                        data: dstBase.advanced(by: dstOffset),
                                        height: 1,
                                        width: vImagePixelCount(d2),
                                        rowBytes: d2 * MemoryLayout<Float>.size
                                    )
                                    vImageConvert_Planar16FtoPlanarF(&srcBuf, &dstBuf, vImage_Flags(kvImageNoFlags))
                                }
                            } else {
                                for k in 0..<d2 {
                                    let srcOff = srcOffset + k * strides[2]
                                    protoRawFloats[dstOffset + k] = Float(Float16(bitPattern: src16[srcOff]))
                                }
                            }
                        }
                    }
                } else {
                    return nil
                }
            }
        }

        if cIdx == 0 {
            protoPlanes.withUnsafeMutableBufferPointer { dst in
                guard let dstBase = dst.baseAddress else { return }
                protoRawFloats.withUnsafeBufferPointer { src in
                    guard let srcBase = src.baseAddress else { return }
                    memcpy(dstBase, srcBase, count * planeSize * MemoryLayout<Float>.size)
                }
            }
        } else if cIdx == 2 {
            // Channel-last → channel-first transpose using BLAS (stride copy).
            // Input: [H, W, 32] packed in protoRawFloats
            // Output: [32, H, W] packed in protoPlanes
            protoRawFloats.withUnsafeBufferPointer { src in
                protoPlanes.withUnsafeMutableBufferPointer { dst in
                    guard let srcBase = src.baseAddress, let dstBase = dst.baseAddress else { return }
                    for k in 0..<count {
                        // src: every 32nd element starting at k (stride=count)
                        // dst: contiguous plane of size planeSize
                        blas_scopy(
                            BLASInt(planeSize),
                            srcBase.advanced(by: k),
                            BLASInt(count),
                            dstBase.advanced(by: k * planeSize),
                            1
                        )
                    }
                }
            }
        }
        return (protoPlanes, count, h, w)
    }

    /// 3×3 binary closing (dilate ∘ erode) on a planar mask using vImage max/min — same semantics as 0/255 neighborhood ops, SIMD-friendly at prototype size.
    private func morphologicalBinaryClose3x3Planar8(mask: [UInt8], width: Int, height: Int) -> [UInt8] {
        let count = width * height
        guard count == mask.count, width >= 1, height >= 1 else { return mask }
        var dilated = [UInt8](repeating: 0, count: count)
        var closed = [UInt8](repeating: 0, count: count)
        let errMax: vImage_Error = mask.withUnsafeBufferPointer { srcPtr in
            dilated.withUnsafeMutableBufferPointer { dPtr in
                var s = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: srcPtr.baseAddress!), height: vImagePixelCount(height), width: vImagePixelCount(width), rowBytes: width)
                var d = vImage_Buffer(data: dPtr.baseAddress!, height: vImagePixelCount(height), width: vImagePixelCount(width), rowBytes: width)
                return vImageMax_Planar8(&s, &d, nil, 0, 0, 3, 3, vImage_Flags(kvImageNoFlags))
            }
        }
        guard errMax == kvImageNoError else {
            return morphologicalBinaryClose3x3(mask: mask, width: width, height: height)
        }
        let errMin: vImage_Error = dilated.withUnsafeBufferPointer { srcPtr in
            closed.withUnsafeMutableBufferPointer { dPtr in
                var s = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: srcPtr.baseAddress!), height: vImagePixelCount(height), width: vImagePixelCount(width), rowBytes: width)
                var d = vImage_Buffer(data: dPtr.baseAddress!, height: vImagePixelCount(height), width: vImagePixelCount(width), rowBytes: width)
                return vImageMin_Planar8(&s, &d, nil, 0, 0, 3, 3, vImage_Flags(kvImageNoFlags))
            }
        }
        guard errMin == kvImageNoError else {
            return morphologicalBinaryClose3x3(mask: mask, width: width, height: height)
        }
        return closed
    }

    /// 3×3 binary dilate (foreground = any > 0 in neighborhood).
    private func morphologicalBinaryDilate3x3(mask: [UInt8], width: Int, height: Int) -> [UInt8] {
        var out = [UInt8](repeating: 0, count: width * height)
        for y in 0..<height {
            for x in 0..<width {
                var hit = false
                outer: for dy in -1...1 {
                    for dx in -1...1 {
                        let nx = x + dx
                        let ny = y + dy
                        if nx < 0 || nx >= width || ny < 0 || ny >= height { continue }
                        if mask[ny * width + nx] > 0 {
                            hit = true
                            break outer
                        }
                    }
                }
                out[y * width + x] = hit ? 255 : 0
            }
        }
        return out
    }

    /// 3×3 binary erode (foreground only if all neighbors > 0).
    private func morphologicalBinaryErode3x3(mask: [UInt8], width: Int, height: Int) -> [UInt8] {
        var out = [UInt8](repeating: 0, count: width * height)
        for y in 0..<height {
            for x in 0..<width {
                var all = true
                outer: for dy in -1...1 {
                    for dx in -1...1 {
                        let nx = x + dx
                        let ny = y + dy
                        if nx < 0 || nx >= width || ny < 0 || ny >= height {
                            all = false
                            break outer
                        }
                        if mask[ny * width + nx] == 0 {
                            all = false
                            break outer
                        }
                    }
                }
                out[y * width + x] = all ? 255 : 0
            }
        }
        return out
    }

    /// Closing = dilate ∘ erode: fills small holes / smooths thin gaps without raising the logit threshold.
    private func morphologicalBinaryClose3x3(mask: [UInt8], width: Int, height: Int) -> [UInt8] {
        let dilated = morphologicalBinaryDilate3x3(mask: mask, width: width, height: height)
        return morphologicalBinaryErode3x3(mask: dilated, width: width, height: height)
    }

    // MARK: - MLMultiArray
    // NOTE: Consider exporting the model with built-in image preprocessing and FP16 inputs
    // to avoid this CPU conversion entirely. MLShapedArray<Float16> can also reduce bandwidth.
    private func pixelBufferToMLMultiArray(_ pixelBuffer: CVPixelBuffer) -> MLMultiArray? {
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        // Support any square size (640, 1280, etc.)
        guard width == height, width > 0 else { return nil }

        // Reuse cached MLMultiArray if size matches
        if cachedMLArraySize != width || cachedMLArray == nil {
            guard let newArray = try? MLMultiArray(shape: [1, 3, NSNumber(value: height), NSNumber(value: width)], dataType: .float32) else { return nil }
            cachedMLArray = newArray
            cachedMLArraySize = width
        }
        guard let array = cachedMLArray else { return nil }
        
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else { return nil }
        
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let pixelCount = width * height
        let floatSize = MemoryLayout<Float32>.size
        let planeStrideBytes = pixelCount * floatSize
        
        let rPtr = array.dataPointer.advanced(by: 0).assumingMemoryBound(to: Float32.self)
        let gPtr = array.dataPointer.advanced(by: planeStrideBytes).assumingMemoryBound(to: Float32.self)
        let bPtr = array.dataPointer.advanced(by: planeStrideBytes * 2).assumingMemoryBound(to: Float32.self)
        
        let src = baseAddress.assumingMemoryBound(to: UInt8.self)
        var rowU8 = [UInt8](repeating: 0, count: width * 4)
        var rowF = [Float](repeating: 0, count: width * 4)
        var scale: Float = 1.0 / 255.0
        
        var indicesR = [vDSP_Length](repeating: 0, count: width)
        var indicesG = [vDSP_Length](repeating: 0, count: width)
        var indicesB = [vDSP_Length](repeating: 0, count: width)
        for i in 0..<width {
            indicesR[i] = vDSP_Length(2 + i * 4)
            indicesG[i] = vDSP_Length(1 + i * 4)
            indicesB[i] = vDSP_Length(0 + i * 4)
        }
        
        for y in 0..<height {
            let rowStart = src.advanced(by: y * bytesPerRow)
            memcpy(&rowU8, rowStart, width * 4)
            
            rowU8.withUnsafeBufferPointer { u8 in
                rowF.withUnsafeMutableBufferPointer { f in
                    vDSP_vfltu8(u8.baseAddress!, 1, f.baseAddress!, 1, vDSP_Length(width * 4))
                    vDSP_vsmul(f.baseAddress!, 1, &scale, f.baseAddress!, 1, vDSP_Length(width * 4))
                }
            }
            
            rowF.withUnsafeBufferPointer { rf in
                vDSP_vgathr(rf.baseAddress!, indicesR, 1, rPtr.advanced(by: y * width), 1, vDSP_Length(width))
                vDSP_vgathr(rf.baseAddress!, indicesG, 1, gPtr.advanced(by: y * width), 1, vDSP_Length(width))
                vDSP_vgathr(rf.baseAddress!, indicesB, 1, bPtr.advanced(by: y * width), 1, vDSP_Length(width))
            }
        }
        return array
    }

    // MARK: - Rotation Helpers

    /// Rotates a CVPixelBuffer 90 degrees using vImage
    private func rotatePixelBuffer90(_ pixelBuffer: CVPixelBuffer, clockwise: Bool) -> CVPixelBuffer? {
        let srcWidth = CVPixelBufferGetWidth(pixelBuffer)
        let srcHeight = CVPixelBufferGetHeight(pixelBuffer)

        // After 90° rotation, dimensions swap
        let dstWidth = srcHeight
        let dstHeight = srcWidth

        // Create destination buffer with proper attributes
        let attrs: [String: Any] = [
            kCVPixelBufferCGImageCompatibilityKey as String: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey as String: true
        ]
        var dstBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(nil, dstWidth, dstHeight, kCVPixelFormatType_32BGRA, attrs as CFDictionary, &dstBuffer)
        guard status == kCVReturnSuccess, let dst = dstBuffer else { return nil }

        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        CVPixelBufferLockBaseAddress(dst, [])
        defer {
            CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly)
            CVPixelBufferUnlockBaseAddress(dst, [])
        }

        guard let srcBase = CVPixelBufferGetBaseAddress(pixelBuffer),
              let dstBase = CVPixelBufferGetBaseAddress(dst) else { return nil }

        let srcBytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let dstBytesPerRow = CVPixelBufferGetBytesPerRow(dst)

        // Use vImage for efficient rotation
        var srcBuffer = vImage_Buffer(
            data: srcBase,
            height: vImagePixelCount(srcHeight),
            width: vImagePixelCount(srcWidth),
            rowBytes: srcBytesPerRow
        )
        var dstBufferImg = vImage_Buffer(
            data: dstBase,
            height: vImagePixelCount(dstHeight),
            width: vImagePixelCount(dstWidth),
            rowBytes: dstBytesPerRow
        )

        // vImage rotation: kRotate90DegreesClockwise or kRotate90DegreesCounterClockwise
        let rotationConstant: UInt8 = clockwise ? UInt8(kRotate90DegreesClockwise) : UInt8(kRotate90DegreesCounterClockwise)
        let bgColor: [UInt8] = [0, 0, 0, 255]

        let error = vImageRotate90_ARGB8888(&srcBuffer, &dstBufferImg, rotationConstant, bgColor, vImage_Flags(kvImageNoFlags))
        guard error == kvImageNoError else { return nil }

        return dst
    }

    /// Rotates a CGImage 90 degrees
    private func rotateCGImage90(_ image: CGImage, clockwise: Bool) -> CGImage? {
        let srcWidth = image.width
        let srcHeight = image.height

        // After 90° rotation, dimensions swap
        let dstWidth = srcHeight
        let dstHeight = srcWidth

        // Explicitly use premultiplied alpha to preserve transparency after rotation
        let colorSpace = image.colorSpace ?? CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue

        guard let context = CGContext(
                  data: nil,
                  width: dstWidth,
                  height: dstHeight,
                  bitsPerComponent: 8,
                  bytesPerRow: dstWidth * 4,
                  space: colorSpace,
                  bitmapInfo: bitmapInfo
              ) else { return nil }

        // Apply rotation transform
        context.translateBy(x: CGFloat(dstWidth) / 2, y: CGFloat(dstHeight) / 2)
        context.rotate(by: clockwise ? -.pi / 2 : .pi / 2)
        context.translateBy(x: -CGFloat(srcWidth) / 2, y: -CGFloat(srcHeight) / 2)
        context.draw(image, in: CGRect(x: 0, y: 0, width: srcWidth, height: srcHeight))

        return context.makeImage()
    }

    // MARK: - Progress UI
    private func setProgress(_ value: Float, text: String) {
        guard !hasFirstDetection, !suppressStartupProgress else { return }
        DispatchQueue.main.async {
            self.progressContainer.isHidden = false
            self.progressView.progress = value
            self.progressLabel.text = "  \(text)  "
        }
    }

    /// Throttled mean sRGB of the composited cutout for room aesthetic / furniture profile (runs off the main thread).
    private func scheduleSegmentationMeanColorPublishIfNeeded(compositedCgImage: CGImage) {
        guard onSegmentationMaskMeanColorSRGB != nil else { return }
        let now = CFAbsoluteTimeGetCurrent()
        guard now - lastSegmentationMeanColorPublishAt >= segmentationMeanColorMinPublishInterval else { return }
        lastSegmentationMeanColorPublishAt = now

        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let mean = FurnitureSegmentationMeanColor.meanStraightSRGB(cgImage: compositedCgImage) else { return }
            DispatchQueue.main.async {
                guard let self, let callback = self.onSegmentationMaskMeanColorSRGB else { return }
                callback(mean)
            }
        }
    }

    /// Call on main **after** the composited segmentation mask is applied to `maskImageView` (or when compose failed but mask had foreground).
    /// Reads ``hasFirstDetection`` on main before ``finishFirstDetectionIfNeeded`` so per-frame pinhole is not skipped due to detection-queue races.
    private func commitPinholeFurnitureSizeAfterSegmentationMaskApplied(
        maskHasForeground: Bool,
        pinholeResult: PrimaryBboxMetersResult?,
        firstFrameMeters: (width: Float, height: Float, pipeline: String, dist: Float?)?,
        imageWidth: Int,
        imageHeight: Int,
        bboxWidthPx: Int,
        bboxHeightPx: Int,
        arDepthSnapshotAttached: Bool
    ) {
        guard maskHasForeground else { return }
        let hadSegmentationBeforeThisCommit = hasFirstDetection
        if !hadSegmentationBeforeThisCommit, let fm = firstFrameMeters {
            logFurnitureFitSize(
                "phase=first_mask_commit W×H_m=\(String(format: "%.3f", fm.width))×\(String(format: "%.3f", fm.height)) pipeline=\(fm.pipeline) dist_m=\(fm.dist.map { String(format: "%.3f", $0) } ?? "—") → finishFirst + parent callback"
            )
        } else if !hadSegmentationBeforeThisCommit {
            logFurnitureFitSize(
                "phase=first_mask_commit_no_metrics maskHasForeground=true bbox_px=\(bboxWidthPx)x\(bboxHeightPx) " +
                "buffer=\(imageWidth)x\(imageHeight) arDepthSnapshot=\(arDepthSnapshotAttached)"
            )
        }
        finishFirstDetectionIfNeeded(
            furnitureWidthMeters: firstFrameMeters?.width,
            furnitureHeightMeters: firstFrameMeters?.height,
            pipelineTag: firstFrameMeters?.pipeline ?? "mask_only_no_metric",
            distanceMeters: firstFrameMeters?.dist,
            arDepthSnapshotAttached: arDepthSnapshotAttached,
            imageWidth: imageWidth,
            imageHeight: imageHeight,
            bboxWidthPx: bboxWidthPx,
            bboxHeightPx: bboxHeightPx
        )
        if hadSegmentationBeforeThisCommit, let p = pinholeResult {
            let arHeight = normalizedARFurnitureHeightMeters()
            if arHeight == nil {
                logFurnitureFitSize(
                    "phase=height_publish_skip source=\(p.pipeline) reason=ar_only_height width_m=\(String(format: "%.3f", p.size.width)) non_ar_height_m=\(String(format: "%.3f", p.size.height))"
                )
            }
            let est = FurnitureSizeEstimate(
                widthMeters: p.size.width,
                heightMeters: arHeight ?? p.size.height,
                arHeightMeters: arHeight
            )
            onFurnitureSizeEstimated?(est)
        }
    }

    /// First successful mask: optional bbox×room-scale size in **meters** on the progress chip, and callback for parents (Sharp room UI).
    private func finishFirstDetectionIfNeeded(
        furnitureWidthMeters: Float?,
        furnitureHeightMeters: Float?,
        pipelineTag: String = "unknown",
        distanceMeters: Float? = nil,
        arDepthSnapshotAttached: Bool = false,
        imageWidth: Int = 0,
        imageHeight: Int = 0,
        bboxWidthPx: Int = 0,
        bboxHeightPx: Int = 0
    ) {
        guard !hasFirstDetection else { return }
        hasFirstDetection = true
        segmentationCompletedOnceThisSession = true
        onFirstSegmentationComplete?()

        if let fw = furnitureWidthMeters, let fh = furnitureHeightMeters {
            let distStr = distanceMeters.map { String(format: "%.3f", $0) } ?? "—"
            logFurnitureFitSize(
                "phase=first pipeline=\(pipelineTag) dist_m=\(distStr) W×H_m=\(String(format: "%.3f", fw))×\(String(format: "%.3f", fh)) buffer=\(imageWidth)x\(imageHeight) bbox_px=\(bboxWidthPx)x\(bboxHeightPx) arDepthSnapshot=\(arDepthSnapshotAttached)"
            )
            let arHeight = normalizedARFurnitureHeightMeters()
            if arHeight == nil {
                logFurnitureFitSize(
                    "phase=first_height_publish_skip source=\(pipelineTag) reason=ar_only_height width_m=\(String(format: "%.3f", fw)) non_ar_height_m=\(String(format: "%.3f", fh))"
                )
            }
            let est = FurnitureSizeEstimate(
                widthMeters: fw,
                heightMeters: arHeight ?? fh,
                arHeightMeters: arHeight
            )
            DispatchQueue.main.async {
                self.onFurnitureSizeEstimated?(est)
            }
        }

        let showMeterBanner = furnitureWidthMeters != nil &&
            normalizedARFurnitureHeightMeters() != nil &&
            !suppressStartupProgress
        let dismissDelay: TimeInterval = showMeterBanner ? 0.95 : 0

        DispatchQueue.main.async {
            if showMeterBanner,
               let fw = furnitureWidthMeters,
               let fh = self.normalizedARFurnitureHeightMeters() {
                self.progressContainer.isHidden = false
                self.progressContainer.alpha = 1
                self.progressView.progress = 1.0
                self.progressLabel.text = String(format: "  ~%.2f m × %.2f m  \n  (AR height)  ", fw, fh)
            }
            UIView.animate(withDuration: 0.25, delay: dismissDelay, options: []) {
                self.progressContainer.alpha = 0
            } completion: { _ in
                self.progressContainer.isHidden = true
                self.progressContainer.alpha = 1
            }
        }
    }

    private func normalizedARFurnitureHeightMeters() -> Float? {
        guard let value = lastAREstimatedHeightMeters,
              value.isFinite,
              value > 0.05 else { return nil }
        return value
    }

    // MARK: - Gestures
    @objc private func handlePinch(_ gesture: UIPinchGestureRecognizer) {
        guard maskImageView.image != nil else { return }
        guard overlayPresentationMode == .measuredPlacement else { return }

        switch gesture.state {
        case .began:
            userLockedAssistedOverlayScale = true
        case .changed:
            userLockedAssistedOverlayScale = true
            let newPinch = userPinchScale * gesture.scale
            userPinchScale = min(max(newPinch, 0.25), 4.0)
            applyCurrentOverlayScaleTransform()
            gesture.scale = 1.0
        case .ended, .cancelled:
            // Snap pinch back to neutral if user barely scaled (same spirit as old 0.9–1.1 total scale).
            if userPinchScale > 0.92 && userPinchScale < 1.08 {
                userPinchScale = 1.0
                userLockedAssistedOverlayScale = false
                UIView.animate(withDuration: 0.2) {
                    self.applyCurrentOverlayScaleTransform()
                }
            }
        default: break
        }
    }

    /// Reset overlay scale back to its default for the current detection: keep room and AR-assisted
    /// scaling, but remove user pinch so the furniture returns to its \"real\" size.
    @objc private func handleResetScaleTapped() {
        guard maskImageView.image != nil else { return }
        userPinchScale = 1.0
        userLockedAssistedOverlayScale = false
        if debugMode {
            logDebug("📐 [RESET] overlay scale reset to default (pinch=1.0)")
        }
        UIView.animate(withDuration: 0.2) {
            self.applyCurrentOverlayScaleTransform()
        }
    }

    /// Check if a point in image coordinates is on a non-transparent pixel
    private func isPointOnMask(point: CGPoint, in image: UIImage) -> Bool {
        guard let cgImage = image.cgImage else { return false }

        let width = cgImage.width
        let height = cgImage.height

        // Check bounds
        let x = Int(point.x)
        let y = Int(point.y)
        guard x >= 0 && x < width && y >= 0 && y < height else { return false }

        // Get pixel data
        guard let dataProvider = cgImage.dataProvider,
              let data = dataProvider.data,
              let bytes = CFDataGetBytePtr(data) else { return false }

        let bytesPerPixel = cgImage.bitsPerPixel / 8
        let bytesPerRow = cgImage.bytesPerRow
        let pixelOffset = y * bytesPerRow + x * bytesPerPixel

        // Check alpha channel (assuming RGBA or BGRA format)
        let alphaIndex = bytesPerPixel - 1  // Alpha is typically last byte
        let alpha = bytes[pixelOffset + alphaIndex]

        // Consider pixel "on mask" if alpha > 50
        return alpha > 50
    }
    
    @objc private func handlePan(_ gesture: UIPanGestureRecognizer) {
        guard maskImageView.image != nil else { return }
        guard overlayPresentationMode == .measuredPlacement else { return }

        let translation = gesture.translation(in: self)
        switch gesture.state {
        case .began, .changed:
            maskImageView.center = CGPoint(x: maskImageView.center.x + translation.x, y: maskImageView.center.y + translation.y)
            gesture.setTranslation(.zero, in: self)
        default: break
        }
    }
    
    func gestureRecognizer(_ gestureRecognizer: UIGestureRecognizer, shouldRecognizeSimultaneouslyWith other: UIGestureRecognizer) -> Bool {
        // Allow pinch and pan to work together
        if (gestureRecognizer is UIPinchGestureRecognizer && other is UIPanGestureRecognizer) ||
           (gestureRecognizer is UIPanGestureRecognizer && other is UIPinchGestureRecognizer) {
            return true
        }
        return false
    }

    func gestureRecognizer(_ gestureRecognizer: UIGestureRecognizer, shouldReceive touch: UITouch) -> Bool {
        // Only recognize gestures if touch is within the primary furniture bounding box
        guard maskImageView.image != nil else {
            logDebug("👆 [shouldReceive] No mask image")
            return false
        }
        guard primaryBboxInView.width > 0 && primaryBboxInView.height > 0 else {
            logDebug("👆 [shouldReceive] No valid bbox")
            return false
        }

        let touchPoint = touch.location(in: maskImageView)
        let contains = primaryBboxInView.contains(touchPoint)

        logDebug("👆 [shouldReceive] touchPoint=\(touchPoint) bbox=\(primaryBboxInView) contains=\(contains)")

        // Check if touch is within the primary bbox
        return contains
    }

    func gestureRecognizer(_ gestureRecognizer: UIGestureRecognizer, shouldBeRequiredToFailBy otherGestureRecognizer: UIGestureRecognizer) -> Bool {
        // Don't interfere with navigation gestures
        return false
    }

    // MARK: - Hit Testing
    override func hitTest(_ point: CGPoint, with event: UIEvent?) -> UIView? {
        // If no mask or no valid bbox, pass through all touches
        guard maskImageView.image != nil else {
            logDebug("👆 [hitTest] No mask image, passing through")
            return nil
        }

        guard primaryBboxInView.width > 0 && primaryBboxInView.height > 0 else {
            logDebug("👆 [hitTest] No valid bbox (\(primaryBboxInView)), passing through")
            return nil
        }

        // Convert point to maskImageView coordinates
        let pointInMask = convert(point, to: maskImageView)

        logDebug("👆 [hitTest] point=\(point) pointInMask=\(pointInMask) bbox=\(primaryBboxInView)")

        // Only handle touches inside the primary bbox
        if primaryBboxInView.contains(pointInMask) {
            logDebug("👆 [hitTest] INSIDE bbox - handling touch")
            return super.hitTest(point, with: event)
        }

        // Pass through touches outside bbox to underlying room view
        logDebug("👆 [hitTest] OUTSIDE bbox - passing through")
        return nil
    }
}

// MARK: - ARSessionDelegate (ARKit-as-camera path only; hybrid uses AVCapture + `makeDepthSnapshot` on capture thread)
extension FurnitureFitContainerView {
    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        guard isUsingARCameraPath else { return }
        isARDepthCompanionSessionRunning = true
        frameLock.lock()
        if isProcessing {
            preferImmediateNextInference = true
            frameLock.unlock()
            return
        }
        let now = Date()
        let shouldProcess = now.timeIntervalSince(lastProcessTime) >= processInterval
        if shouldProcess {
            isProcessing = true
            lastProcessTime = now
        }
        frameLock.unlock()
        guard shouldProcess else { return }
        guard let bgra = FurnitureFitARSupport.copyCapturedImageToBGRA(
            frame: frame,
            reuse: &arBGRAReuse,
            ciContext: arCIContext,
            lockedOrientation: lockedOrientation
        ) else {
            resetProcessingFlag()
            return
        }
        let depthSnapshot = FurnitureFitARSupport.makeDepthSnapshot(
            frame: frame,
            bgraWidth: CVPixelBufferGetWidth(bgra),
            bgraHeight: CVPixelBufferGetHeight(bgra),
            lockedOrientation: lockedOrientation
        )
        detectionQueue.async { [weak self] in
            self?.processFrame(bgra, arDepthSnapshot: depthSnapshot)
        }
    }

    func session(_ session: ARSession, didFailWithError error: Error) {
        let nsError = error as NSError
        logDebug("⚠️ [FurnitureFit] ARSession failed: \(nsError.domain) code=\(nsError.code) — falling back to AVCapture (AR companion off for this session)")
        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            suppressARDepthCompanionAfterCaptureFailure = true
            isUsingARCameraPath = false
            isARDepthCompanionSessionRunning = false
            arSession.pause()
            startClassicCameraPathIfNeeded()
        }
    }
}

extension UIView {
    var parentViewController: UIViewController? {
        var responder: UIResponder? = self
        while let r = responder {
            if let vc = r as? UIViewController { return vc }
            responder = r.next
        }
        return nil
    }
}

extension CGRect {
    var area: CGFloat { width * height }
}
