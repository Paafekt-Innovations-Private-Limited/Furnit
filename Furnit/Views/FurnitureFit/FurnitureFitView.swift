// FurnitureFitView.swift
// Single-stage: detect → optional multi-candidate (5a–5b) → union mask → clip outside primary bbox (15d) → cutout
// With timing at every stage

import SwiftUI
import UIKit
import CoreML
import Accelerate
import AVFoundation
import ARKit
import QuartzCore
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

    @AppStorage("furnitureFit.primaryDetectionMinConfidence") private var primaryDetectionMinConfidenceStorage: Double = 0.75
    @AppStorage("furnitureFit.primarySelectionByHighestConfidence") private var primarySelectionByHighestConfidence: Bool = false
    @AppStorage("furnitureFit.showFullVideoWithIdentifications") private var showFullVideoWithIdentifications: Bool = true

    @ObservedObject private var appState = AppStateManager.shared

    func makeUIView(context: Context) -> FurnitureFitContainerView {
        let v = FurnitureFitContainerView()
        v.processInterval = processInterval
        v.confidenceThreshold = confidenceThreshold
        v.primaryDetectionMinConfidence = clampPrimaryDetectionConfidence(primaryDetectionMinConfidenceStorage)
        v.primarySelectionByHighestConfidence = primarySelectionByHighestConfidence
        v.showFullVideoWithIdentifications = showFullVideoWithIdentifications
        v.useBilinearUpscaling = useBilinearUpscaling
        v.setModel(mlModel)
        if active { v.startIfNeeded() }
        return v
    }

    func updateUIView(_ uiView: FurnitureFitContainerView, context: Context) {
        uiView.setModel(mlModel)
        uiView.processInterval = processInterval
        uiView.confidenceThreshold = confidenceThreshold
        uiView.primaryDetectionMinConfidence = clampPrimaryDetectionConfidence(primaryDetectionMinConfidenceStorage)
        uiView.primarySelectionByHighestConfidence = primarySelectionByHighestConfidence
        uiView.showFullVideoWithIdentifications = showFullVideoWithIdentifications
        uiView.useBilinearUpscaling = useBilinearUpscaling
        if active { uiView.startIfNeeded() } else { uiView.stop() }
    }

    private func clampPrimaryDetectionConfidence(_ raw: Double) -> Float {
        Float(min(max(raw, 0.05), 0.99))
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
    /// Width from room-model intrinsics when available, else bbox × room width (no-LiDAR fallback).
    let widthMeters: Float
    /// Display height: AR when available, else bbox × room height (no-LiDAR fallback).
    let heightMeters: Float
    /// ARKit/LiDAR height when available.
    let arHeightMeters: Float?
}

enum FurnitureFitSegmentationMode: Equatable {
    case identifyOnly
    case segmentSelected
}

private struct DetectionOverlayItem {
    let rectInView: CGRect
    let label: String
    let confidence: Float
    let isSelected: Bool
}

private final class DetectionBBoxOverlayView: UIView {
    var items: [DetectionOverlayItem] = [] {
        didSet { setNeedsDisplay() }
    }

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
        isOpaque = false
        isUserInteractionEnabled = false
        contentMode = .redraw
    }

    override func draw(_ rect: CGRect) {
        guard let ctx = UIGraphicsGetCurrentContext() else { return }
        ctx.clear(rect)
        let paragraphStyle = NSMutableParagraphStyle()
        paragraphStyle.lineBreakMode = .byTruncatingTail
        for item in items {
            let strokeColor: UIColor = item.isSelected ? .systemYellow : UIColor.white.withAlphaComponent(0.88)
            let fillColor = UIColor.black.withAlphaComponent(item.isSelected ? 0.55 : 0.38)
            let lineWidth: CGFloat = item.isSelected ? 2.5 : 1.2
            let boxPath = UIBezierPath(roundedRect: item.rectInView, cornerRadius: 6)
            fillColor.setFill()
            strokeColor.setStroke()
            boxPath.lineWidth = lineWidth
            boxPath.stroke()

            let scoreText = String(format: "%.2f", item.confidence)
            let text = "\(item.label) \(scoreText)"
            let attributes: [NSAttributedString.Key: Any] = [
                .font: UIFont.systemFont(ofSize: item.isSelected ? 11 : 10, weight: item.isSelected ? .semibold : .medium),
                .foregroundColor: UIColor.white,
                .paragraphStyle: paragraphStyle
            ]
            let maxLabelWidth = min(max(item.rectInView.width, 56), 140)
            let textSize = (text as NSString).size(withAttributes: attributes)
            let labelRect = CGRect(
                x: item.rectInView.minX,
                y: max(0, item.rectInView.minY - textSize.height - 8),
                width: min(maxLabelWidth, textSize.width + 10),
                height: textSize.height + 6
            )
            let labelPath = UIBezierPath(roundedRect: labelRect, cornerRadius: 6)
            fillColor.setFill()
            labelPath.fill()
            (text as NSString).draw(
                in: labelRect.insetBy(dx: 5, dy: 3),
                withAttributes: attributes
            )
        }
    }
}

private extension UIButton.Configuration {
    static func furnitureSelectionChip() -> UIButton.Configuration {
        var config = UIButton.Configuration.plain()
        config.baseForegroundColor = .white
        config.background.backgroundColor = UIColor.black.withAlphaComponent(0.72)
        config.background.cornerRadius = 18
        config.contentInsets = NSDirectionalEdgeInsets(top: 8, leading: 14, bottom: 8, trailing: 14)
        config.image = UIImage(systemName: "xmark.circle.fill")
        config.imagePlacement = .trailing
        config.imagePadding = 8
        return config
    }
}

// MARK: - Main Container View
final class FurnitureFitContainerView: UIView, AVCaptureVideoDataOutputSampleBufferDelegate, ARSessionDelegate, UIGestureRecognizerDelegate {
    private static let fullVideoWithIdentificationsKey = "furnitureFit.showFullVideoWithIdentifications"
    private static let minimumARFurnitureHeightMeters: Float = 0.25
    private static let minimumARFurnitureWidthMeters: Float = 0.25

    // MARK: Config
    var processInterval: TimeInterval = 0.07
    var confidenceThreshold: Float = 0.1
    /// Minimum detector confidence (0…1) for **primary** furniture selection among qualifying boxes. Parsed candidates still use ``confidenceThreshold``.
    var primaryDetectionMinConfidence: Float = 0.75
    /// When `true`, primary is the **highest confidence** among boxes ≥ ``primaryDetectionMinConfidence`` (ties → larger area). When `false`, primary is the **largest area** among those boxes.
    var primarySelectionByHighestConfidence: Bool = false
    /// Keep the rendered mask raw from YOLOE logits. Do not morphologically alter it.
    private let furnitureFitUseMorphologicalCloseMask: Bool = false
    /// 3×3 binary closing on the composite band was filling thin gaps / “holes” in chair handles.
    /// Disabled so logits + threshold define the mask only (no morphology).
    private let furnitureFitNativeMaskMorphologicalClose: Bool = false
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
    /// Sharp Room: furniture depth can use the splat depth buffer instead of device-pitch floor heuristics.
    weak var sharpRoomSplatMeasurementHost: GaussianSplatMeasurementHost?
    var cameraFocalLengthPixels: Float = 0

    private var captureBackCamera: AVCaptureDevice?
    private var lastSceneFitmentLogAt: CFAbsoluteTime = 0
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

    var onFurnitureSizeEstimated: ((FurnitureSizeEstimate) -> Void)?
    /// Mean straight sRGB (0…1) over opaque-enough pixels of the composited segmentation cutout; throttled (~4 Hz).
    var onSegmentationMaskMeanColorSRGB: ((SIMD3<Float>) -> Void)?
    /// Sharp Room uses this to opt into AR sizing explicitly instead of defaulting to it.
    var arAssistedSizingEnabled: Bool = true

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
    private var captureSessionObserverTokens: [NSObjectProtocol] = []
    private let videoOutput = AVCaptureVideoDataOutput()
    private let sampleQueue = DispatchQueue(label: "com.furnit.sample", qos: .userInitiated)
    private let captureSessionControlQueue = DispatchQueue(label: "com.furnit.capture.control", qos: .userInitiated)

    // MARK: Camera Path
    private let arSession = ARSession()
    private let arSessionDelegateQueue = DispatchQueue(label: "com.furnit.furniturefit.arsession", qos: .userInitiated)
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
    /// Raw furniture estimate before user pinch is applied; used to republish the pill immediately on manual resize.
    private var latestEstimatedFurnitureBaseWidthMeters: Float?
    private var latestEstimatedFurnitureBaseHeightMeters: Float?
    private var latestEstimatedFurnitureBaseARHeightMeters: Float?
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
    private let detectionBBoxOverlayView = DetectionBBoxOverlayView()
    private let selectedObjectChipButton: UIButton = {
        let button = UIButton(type: .system)
        button.translatesAutoresizingMaskIntoConstraints = false
        button.configuration = .furnitureSelectionChip()
        button.isHidden = true
        button.alpha = 0
        return button
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
    /// Consecutive empty-detection frames since the last valid mask. Keeps the previous mask
    /// visible for up to `maskGraceFrameLimit` frames so brief tracking drops don't flicker.
    private var consecutiveEmptyMaskFrames = 0
    private let maskGraceFrameLimit = 3
    /// After the first successful segmentation in this container, skip startup progress on later `startIfNeeded()` (same UIView instance).
    private var segmentationCompletedOnceThisSession = false
    /// True only while the initial detection startup banner is allowed to update.
    private var startupProgressActive = false
    /// When true (e.g. Sharp Room already completed segmentation this session), hide startup progress even for a new UIView.
    var suppressStartupProgress = false
    /// Called once when the first valid segmentation mask is ready (parent can persist “no progress next time”).
    var onFirstSegmentationComplete: (() -> Void)?
    /// Reports the currently selected classes for multi-tap segmentation.
    var onSelectedClassLabelsChanged: (([String]) -> Void)?
    /// Controls whether identify-only mode should show the full live preview layer.
    var showIdentifyLivePreview: Bool = true

    // MARK: - Overlay scale (room × AR when enabled × user pinch)
    /// From `FurnitureSizingCalculator`; combined with a modest default baseline so large furniture
    /// like beds and sofas do not start too small before AR/proportion sizing kicks in.
    private var autoScaleFromRoom: CGFloat = 1.0
    /// User pinch multiplier (reset when primary class changes).
    private var userPinchScale: CGFloat = 1.0
    /// User pan translation in view points (reset when primary class changes).
    private var userPanOffset: CGPoint = .zero
    /// After user pinches, stop updating AR-assisted auto-scale until primary class changes.
    private var userLockedAssistedOverlayScale: Bool = false
    /// When true, skip per-frame AR/room/default overlay sizing so manual pinch/min–max is preserved across frames.
    private var shouldFreezeAutomaticOverlaySizing: Bool {
        userLockedAssistedOverlayScale || userPinchScale < 0.92 || userPinchScale > 1.08
    }
    /// Last primary class used for overlay / pinch reset.
    private var lastOverlayPrimaryClassIdx: Int = -1

    private let minCombinedOverlayScale: CGFloat = 0.08
    private let maxCombinedOverlayScale: CGFloat = 3.0
    private let defaultStaticOverlayScale: CGFloat = 1.28
    var showFullVideoWithIdentifications: Bool = {
        let defaults = UserDefaults.standard
        if defaults.object(forKey: FurnitureFitContainerView.fullVideoWithIdentificationsKey) != nil {
            return defaults.bool(forKey: FurnitureFitContainerView.fullVideoWithIdentificationsKey)
        }
        return true
    }() {
        didSet {
            guard oldValue != showFullVideoWithIdentifications else { return }
            updateVideoIdentificationPresentation()
        }
    }
    var segmentationMode: FurnitureFitSegmentationMode = .identifyOnly {
        didSet {
            guard oldValue != segmentationMode else { return }
            if segmentationMode == .identifyOnly {
                DispatchQueue.main.async {
                    self.resetOverlayScalesForEmptyMask(clearDetectedCandidates: false, clearSelections: false)
                    self.applyCurrentOverlayScaleTransform()
                }
            } else if segmentationMode == .segmentSelected {
                // Hide detection boxes immediately; mask-only presentation while segmenting.
                DispatchQueue.main.async {
                    self.detectionBBoxOverlayView.items = []
                    self.candidateBboxesInView = []
                }
            }
            updateVideoIdentificationPresentation()
        }
    }
    private var isShowingLiveVideoIdentifications: Bool {
        showFullVideoWithIdentifications && showIdentifyLivePreview && segmentationMode == .identifyOnly
    }
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
    /// Throttled logging for oscillation diagnosis (when `debugMode` is on).
    private var overlayDebugLastAssistedLabel: String = ""
    private var overlayDebugLastCombined: CGFloat = -1

    /// Throttled always-on AR metrics for cross-platform comparison (filter: `FurnitureFitAR`).
    private var lastFurnitureFitARFrameLogAt: TimeInterval = 0
    private let furnitureFitARLogInterval: TimeInterval = 0.5

    /// Throttled premultiplied RGBA sample at bbox center (filter: `FurnitureFit` + debugMode).
    private var lastCompositePremulRGBAlogAt: TimeInterval = 0
    private let compositePremulRGBAlogInterval: TimeInterval = 0.5
    /// Throttled sampled binary-mask shape log for the current composited object.
    private var lastCompositeMaskShapeLogAt: TimeInterval = 0
    private let compositeMaskShapeLogInterval: TimeInterval = 0.5
    /// Throttled pre-composite ASCII mask log for chair-like detections.
    private var lastChairMaskLogAt: TimeInterval = 0
    private let chairMaskLogInterval: TimeInterval = 0.75
    /// Throttled staged mask comparison for the primary chair bbox.
    private var lastChairStageComparisonLogAt: TimeInterval = 0
    private let chairStageComparisonLogInterval: TimeInterval = 0.75
    /// Throttled per-pixel contributor log for opaque chair-handle regions.
    private var lastChairContributorLogAt: TimeInterval = 0
    private let chairContributorLogInterval: TimeInterval = 0.75
    /// Throttled numeric logit grids (proto + bilinear) for chair structure / handle debugging.
    private var lastChairLogitGridLogAt: TimeInterval = 0
    private let chairLogitGridLogInterval: TimeInterval = 1.25

    /// Primary furniture bounding box in view coordinates (for gesture hit testing)
    private var primaryBboxInView: CGRect = .zero
    /// All current candidate bboxes in pre-transform overlay coordinates. Used for drawing and tap-selection.
    private var candidateBboxesInView: [CGRect] = []
    /// Latest displayed candidates aligned with ``candidateBboxesInView``. Main-thread only for tap-selection.
    private var latestDisplayedCandidates: [FurnitureFitDetection] = []
    private var latestDisplayedSelectedCandidateIndex: Int?
    /// Latest prototype mask state used to validate taps against per-candidate mask presence.
    private let tapMaskStateLock = NSLock()
    private var latestTapMaskPlanes: [Float] = []
    private var latestTapMaskProtoWidth: Int = 0
    private var latestTapMaskProtoHeight: Int = 0
    private var latestTapMaskModelSide: Int = 0
    private var latestTapMaskImageWidth: Int = 0
    private var latestTapMaskImageHeight: Int = 0
    private let selectedClassStateLock = NSLock()
    /// User-tapped instances (geometry snapshots). Segmentation matches these to current-frame boxes by IoU — not by class alone (avoids segmenting every chair when one is chosen).
    private var selectedDetectionPins: [FurnitureFitDetection] = []
    /// Minimum IoU between a live detection and a stored pin to treat as the same instance.
    private let pinMatchIoUThreshold: Float = 0.45
    /// Stored reference so `hitTest` can check whether a pinch is already in flight.
    private weak var overlayPinchGesture: UIPinchGestureRecognizer?
    /// Stored reference so `hitTest` can check whether a pan is already in flight.
    private weak var overlayPanGesture: UIPanGestureRecognizer?
    /// Stored reference so bbox taps can coexist with pan/pinch.
    private weak var overlayTapGesture: UITapGestureRecognizer?
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
    /// process_mask_native: bilinear upsample + threshold >0 on GPU.
    private var bilinearUpsamplePipeline: MTLComputePipelineState?

    // MARK: - Cached Metal buffers for fused compositing (prevents allocation per frame)
    private var cachedFusedPlanesBuf: MTLBuffer?
    private var cachedFusedCoeffBuf: MTLBuffer?
    private var cachedFusedPlanesCapacity: Int = 0
    private var cachedFusedCoeffCapacity: Int = 0
    private var cachedFusedProtoMaskBuf: MTLBuffer?
    private var cachedFusedProtoMaskCapacity: Int = 0

    // MARK: - Cached Metal buffers for bilinear upsample (process_mask_native)
    private var cachedLogitBuf: MTLBuffer?
    private var cachedLogitCapacity: Int = 0
    private var cachedBandMaskBuf: MTLBuffer?
    private var cachedBandMaskCapacity: Int = 0

    // MARK: - CVMetalTextureCache (FIXED: was created per-frame causing memory leak)
    private var cvTextureCache: CVMetalTextureCache?

    // MARK: - Reusable prototype buffers (FIXED: prevents ~26MB allocation per frame)
    private var protoRawFloats: [Float] = []
    private var protoPlanes: [Float] = []
    private var protoPlanesInterleaved: [Float] = []

    // MARK: - Reusable CVPixelBuffer & MLMultiArray (prevents allocation per frame)
    /// Stretch path: full frame scaled to the model square (matches Android / ONNX-style preprocessing).
    private var cachedStretchBuffer: CVPixelBuffer?
    private var cachedStretchSize: Int = 0
    /// Letterbox path for the legacy 1280 Core ML package (`yoloe-26l-seg-pf`).
    private var cachedLetterboxBuffer: CVPixelBuffer?
    private var cachedLetterboxSize: Int = 0
    private var cachedMLArray: MLMultiArray?
    private var cachedMLArraySize: Int = 0

    /// Reused scratch for CPU fallback compositing (vDSP + BLAS); grows with frame width, never shrinks.
    private var compositeCpuScratchFloats: [Float] = []
    /// Proto mask upscaled to the active composite band via vImage; grows to the largest band seen and never shrinks.
    private var upscaledPlanarMaskScratch: [UInt8] = []
    /// Whether the current model frame used Ultralytics-style letterbox instead of stretch.
    private var currentYoloUsesLetterbox: Bool = false
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
    private var mlModel: MLModel?  // yoloe-11l-seg-pf or yoloe-26l-seg-pf_* via YOLOEModelService
    private let detectionQueue = DispatchQueue(label: "com.furnit.detection", qos: .userInitiated)
    private var lastProcessTime = Date.distantPast
    private var isProcessing = false
    /// While inference is running, camera frames are dropped; when set, next completion clears `lastProcessTime` so the following frame is not delayed by `processInterval`.
    private var preferImmediateNextInference = false
    private let frameLock = NSLock() // Protects lastProcessTime, isProcessing, preferImmediateNextInference
    /// ARKit holds `ARFrame`s until `session(_:didUpdate:)` returns; CI BGRA + depth copy must stay well below camera FPS.
    private static let arSessionDelegateHeavyMinInterval: TimeInterval = 0.28
    /// Monotonic guard so we skip `didUpdate` **before** taking `frameLock` (cheap drops while inference is still running).
    private var lastARHeavyWorkFinishCAC: CFTimeInterval = 0

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

    // MARK: Class Names (localized: classes.json in each *.lproj; same keys as Android assets/classes.json)
    internal lazy var classNames: [Int: String] = {
        guard let url = Bundle.main.url(forResource: "classes", withExtension: "json"),
              let data = try? Data(contentsOf: url),
              let dict = try? JSONSerialization.jsonObject(with: data) as? [String: String] else {
            if debugMode { logDebug("⚠️ Failed to load localized classes.json (check xx.lproj)") }
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

    private func displayClassName(_ id: Int) -> String {
        classNames[id] ?? "unknown"
    }

    private func isChairLikeClass(_ id: Int) -> Bool {
        displayClassName(id).lowercased().contains("chair")
    }

    private enum ProtoDebugLayout: String {
        case channelFirst = "CHW"
        case interleavedPixelFirst = "PIXELxC"
    }

    private func buildSingleDetectionMaskForDebug(
        channelFirstPlanes: [Float],
        interleavedPlanes: [Float],
        protoW: Int,
        protoH: Int,
        modelSide: Int,
        detection: FurnitureFitDetection,
        layout: ProtoDebugLayout
    ) -> (logits: [Float], binary: [UInt8], left: Int, top: Int, right: Int, bottom: Int)? {
        guard detection.coeffs.count >= 32, protoW > 0, protoH > 0, modelSide > 0 else { return nil }

        let widthRatio = Float(protoW) / Float(modelSide)
        let heightRatio = Float(protoH) / Float(modelSide)
        let protoLeft = max(0, min(protoW - 1, Int(floor((detection.x - detection.w * 0.5) * widthRatio))))
        let protoTop = max(0, min(protoH - 1, Int(floor((detection.y - detection.h * 0.5) * heightRatio))))
        let protoRight = max(0, min(protoW - 1, Int(ceil((detection.x + detection.w * 0.5) * widthRatio))))
        let protoBottom = max(0, min(protoH - 1, Int(ceil((detection.y + detection.h * 0.5) * heightRatio))))
        guard protoRight >= protoLeft, protoBottom >= protoTop else { return nil }

        let cropWidth = protoRight - protoLeft + 1
        let cropHeight = protoBottom - protoTop + 1
        let cropCount = cropWidth * cropHeight
        var croppedLogits = [Float](repeating: 0, count: cropCount)
        var croppedBinary = [UInt8](repeating: 0, count: cropCount)
        let hwProto = protoW * protoH

        guard channelFirstPlanes.count >= 32 * hwProto,
              interleavedPlanes.count >= 32 * hwProto else { return nil }

        for protoY in protoTop...protoBottom {
            for protoX in protoLeft...protoRight {
                let protoPixelIndex = protoY * protoW + protoX
                var sum: Float = 0
                var coeffIndex = 0
                while coeffIndex < 32 {
                    switch layout {
                    case .channelFirst:
                        let planeIndex = coeffIndex * hwProto + protoPixelIndex
                        sum += detection.coeffs[coeffIndex] * channelFirstPlanes[planeIndex]
                    case .interleavedPixelFirst:
                        let planeIndex = protoPixelIndex * 32 + coeffIndex
                        sum += detection.coeffs[coeffIndex] * interleavedPlanes[planeIndex]
                    }
                    coeffIndex += 1
                }

                let cropIndex = (protoY - protoTop) * cropWidth + (protoX - protoLeft)
                croppedLogits[cropIndex] = sum
                croppedBinary[cropIndex] = sum > 0 ? 255 : 0
            }
        }

        return (croppedLogits, croppedBinary, protoLeft, protoTop, protoRight, protoBottom)
    }

    private func binaryMaskAscii(
        mask: [UInt8],
        width: Int,
        height: Int,
        gridCols: Int = 40,
        gridRows: Int = 20
    ) -> String {
        guard width > 0, height > 0, mask.count >= width * height else { return "(empty)" }
        let cols = min(gridCols, width)
        let rows = min(gridRows, height)
        var lines: [String] = []
        lines.reserveCapacity(rows)
        var onCount = 0

        for row in 0..<rows {
            let srcY = row * height / rows
            var line = ""
            line.reserveCapacity(cols)
            for col in 0..<cols {
                let srcX = col * width / cols
                let idx = srcY * width + srcX
                let isOn = mask[idx] != 0
                if isOn { onCount += 1 }
                line.append(isOn ? "█" : "·")
            }
            lines.append(line)
        }

        let totalSamples = max(1, cols * rows)
        let coveragePct = Float(onCount) / Float(totalSamples) * 100
        return "sampled \(cols)x\(rows) on=\(onCount)/\(totalSamples) (\(String(format: "%.1f", coveragePct))%)\n"
            + lines.joined(separator: "\n")
    }

    private func sampleBilinearLogit(
        maskLogits: [Float],
        protoW: Int,
        protoH: Int,
        imageX: Int,
        imageY: Int,
        imageW: Int,
        imageH: Int
    ) -> Float {
        FurnitureFitOnnxStylePipeline.bilinearUpsampledLogit(
            maskLogits: maskLogits,
            protoW: protoW,
            protoH: protoH,
            origW: imageW,
            origH: imageH,
            imageX: imageX,
            imageY: imageY
        )
    }

    private func logPrimaryChairContributorPixelsIfDebug(
        primary: FurnitureFitDetection,
        maskDetectionsForBuild: [FurnitureFitDetection],
        planes: [Float],
        fusedMaskLogits: [Float],
        protoW: Int,
        protoH: Int,
        modelSide: Int,
        origW: Int,
        origH: Int,
        primaryBx1: Int,
        primaryBy1: Int,
        primaryBx2: Int,
        primaryBy2: Int
    ) {
        guard debugMode, isChairLikeClass(primary.classIdx) else { return }
        // Disabled: noisy logit / per-pixel mask contributor debug
    }

    private func logPrimaryChairStageComparisonIfDebug(
        primary: FurnitureFitDetection,
        modelSide: Int,
        protoMask: [UInt8],
        protoW: Int,
        protoH: Int,
        fullResMask: [UInt8],
        outBase: UnsafeMutablePointer<UInt8>,
        bytesPerRowOut: Int,
        origW: Int,
        origH: Int,
        primaryBx1: Int,
        primaryBy1: Int,
        primaryBx2: Int,
        primaryBy2: Int
    ) {
        guard debugMode, isChairLikeClass(primary.classIdx) else { return }
        // Disabled: noisy mask-shape ASCII stage comparison
    }

    /// Prints throttled numeric grids of **fused** proto logits and bilinear image logits so you can
    /// compare raw model structure (handle gaps, sign) vs upsampled values at `maskUpsampleLogitBias`.
    private func logChairFusedLogitNumericGridsIfDebug(
        maskLogits: [Float],
        protoW: Int,
        protoH: Int,
        modelSide: Int,
        primary: FurnitureFitDetection,
        origW: Int,
        origH: Int,
        primaryBx1: Int,
        primaryBy1: Int,
        primaryBx2: Int,
        primaryBy2: Int
    ) {
        guard debugMode, isChairLikeClass(primary.classIdx) else { return }
        // Disabled: noisy fused proto / bilinear logit numeric grids
    }

    private func logChairDetectionMasksIfDebug(
        planes: [Float],
        interleavedPlanes: [Float],
        protoW: Int,
        protoH: Int,
        modelSide: Int,
        detections: [FurnitureFitDetection],
        protoShape: [Int],
        protoStrides: [Int],
        protoChannelAxis: Int
    ) {
        guard debugMode else { return }
        // Disabled: noisy chair ASCII logit/binary mask + protoShape lines
        _ = (planes, interleavedPlanes, protoW, protoH, modelSide, detections, protoShape, protoStrides, protoChannelAxis)
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
        CameraOwnershipDiagnostics.log(owner: "FurnitureFitContainerView", event: "init")
        
        previewLayer.session = captureSession
        previewLayer.videoGravity = .resizeAspectFill
        previewLayer.isHidden = !isShowingLiveVideoIdentifications
        layer.addSublayer(previewLayer)
        
        maskImageView.isUserInteractionEnabled = true
        addSubview(arSCNView)
        addSubview(maskImageView)
        addSubview(detectionBBoxOverlayView)
        addSubview(selectedObjectChipButton)
        arSCNView.session = arSession
        maskImageView.translatesAutoresizingMaskIntoConstraints = false
        detectionBBoxOverlayView.translatesAutoresizingMaskIntoConstraints = false
        NSLayoutConstraint.activate([
            arSCNView.topAnchor.constraint(equalTo: topAnchor),
            arSCNView.bottomAnchor.constraint(equalTo: bottomAnchor),
            arSCNView.leadingAnchor.constraint(equalTo: leadingAnchor),
            arSCNView.trailingAnchor.constraint(equalTo: trailingAnchor),
            maskImageView.topAnchor.constraint(equalTo: topAnchor),
            maskImageView.bottomAnchor.constraint(equalTo: bottomAnchor),
            maskImageView.leadingAnchor.constraint(equalTo: leadingAnchor),
            maskImageView.trailingAnchor.constraint(equalTo: trailingAnchor),
            detectionBBoxOverlayView.topAnchor.constraint(equalTo: topAnchor),
            detectionBBoxOverlayView.bottomAnchor.constraint(equalTo: bottomAnchor),
            detectionBBoxOverlayView.leadingAnchor.constraint(equalTo: leadingAnchor),
            detectionBBoxOverlayView.trailingAnchor.constraint(equalTo: trailingAnchor),
            selectedObjectChipButton.centerXAnchor.constraint(equalTo: centerXAnchor),
            selectedObjectChipButton.topAnchor.constraint(equalTo: safeAreaLayoutGuide.topAnchor, constant: 10)
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

        let tapGesture = UITapGestureRecognizer(target: self, action: #selector(handleBoundingBoxTap(_:)))
        tapGesture.delegate = self
        tapGesture.cancelsTouchesInView = false
        tapGesture.numberOfTapsRequired = 1
        
        addGestureRecognizer(pinchGesture)
        addGestureRecognizer(tapGesture)
        overlayPinchGesture = pinchGesture
        overlayTapGesture = tapGesture
        maskImageView.addGestureRecognizer(panGesture)
        overlayPanGesture = panGesture
        tapGesture.require(toFail: panGesture)
        selectedObjectChipButton.addTarget(self, action: #selector(handleClearSelectedObjectTapped), for: .touchUpInside)

        // Listen for reset notification from SharpRoomView toolbar ("reset size" button).
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handleResetScaleTapped),
            name: NSNotification.Name("FurnitureFitResetOverlayScale"),
            object: nil
        )
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handleClearSelectedObjectTapped),
            name: NSNotification.Name("FurnitureFitClearSelectedObjects"),
            object: nil
        )

        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handleCaptureSessionRuntimeError(_:)),
            name: AVCaptureSession.runtimeErrorNotification,
            object: captureSession
        )
        captureSessionObserverTokens = CameraOwnershipDiagnostics.makeCaptureSessionObservers(
            session: captureSession,
            owner: "FurnitureFitContainerView.AVCapture"
        )
        arSession.delegate = self
        arSession.delegateQueue = arSessionDelegateQueue
        
        setupCamera()
        setupMetal()
        if debugMode { logDebug("✅ FurnitureFitContainerView initialized") }
    }

    deinit {
        CameraOwnershipDiagnostics.removeObservers(captureSessionObserverTokens)
        NotificationCenter.default.removeObserver(self)
        CameraOwnershipDiagnostics.log(owner: "FurnitureFitContainerView", event: "deinit")
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
            if bilinearUpsamplePipeline == nil, let fn4 = library.makeFunction(name: "sp_bilinearUpsampleThreshold") {
                bilinearUpsamplePipeline = try device.makeComputePipelineState(function: fn4)
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
        if isShowingLiveVideoIdentifications {
            applyLockedOrientationVideoRotation()
        }
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

    /// When AR/LiDAR sizing is unavailable, fall back to the detected bbox pixel proportion
    /// relative to the room image so the overlay still scales with room dimensions.
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
        _ = arDepthSnapshot
        _ = preferRoomRaycastSizing

        if shouldFreezeAutomaticOverlaySizing {
            applyCurrentOverlayScaleTransform()
            return
        }

        let arSizingReady =
            arAssistedSizingEnabled &&
            hasARKitAssistedSizingPayload &&
            arAssistedScaleValid &&
            normalizedARFurnitureHeightMeters() != nil &&
            autoScaleFromAR.isFinite &&
            autoScaleFromAR > 0

        guard !arSizingReady else {
            autoScaleFromRoom = 1.0
            applyCurrentOverlayScaleTransform()
            return
        }

        guard arAssistedSizingEnabled, !QualitySettings.supportsLiDARSceneDepth else {
            autoScaleFromRoom = 1.0
            applyCurrentOverlayScaleTransform()
            return
        }

        guard imageWidth > 0, imageHeight > 0,
              bounds.width > 1, bounds.height > 1,
              primaryBboxInView.width > 1, primaryBboxInView.height > 1 else {
            autoScaleFromRoom = 1.0
            applyCurrentOverlayScaleTransform()
            return
        }

        let targetWidthFraction = CGFloat(max(1, primaryBx2 - primaryBx1)) / CGFloat(imageWidth)
        let targetHeightFraction = CGFloat(max(1, primaryBy2 - primaryBy1)) / CGFloat(imageHeight)
        let currentWidthFraction = primaryBboxInView.width / bounds.width
        let currentHeightFraction = primaryBboxInView.height / bounds.height

        var scaleCandidates: [CGFloat] = []
        if currentWidthFraction > 0.0001, targetWidthFraction.isFinite {
            let widthScale = targetWidthFraction / currentWidthFraction
            if widthScale.isFinite, widthScale > 0 {
                scaleCandidates.append(widthScale)
            }
        }
        if currentHeightFraction > 0.0001, targetHeightFraction.isFinite {
            let heightScale = targetHeightFraction / currentHeightFraction
            if heightScale.isFinite, heightScale > 0 {
                scaleCandidates.append(heightScale)
            }
        }

        if scaleCandidates.isEmpty {
            autoScaleFromRoom = 1.0
            applyCurrentOverlayScaleTransform()
            return
        }

        let proportionalScale = scaleCandidates.reduce(0, +) / CGFloat(scaleCandidates.count)
        autoScaleFromRoom = min(max(proportionalScale, 0.25), 2.5)
        applyCurrentOverlayScaleTransform()
    }

    private func updateOverlayPresentationMode(
        primaryClassIdx: Int,
        metric: PrimaryBboxMetersResult?
    ) {
        _ = metric
        if shouldFreezeAutomaticOverlaySizing {
            return
        }
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

        // No usable AR metric: never enter AR-style measured placement.
        // Non-AR stays on deferred/segmentation placement; pinch/pan are allowed separately.
        overlayPresentationMode = .deferredCentered
    }

    /// Combined overlay: AR metric scale when available, else room-proportion fallback, then user pinch.
    private func applyCurrentOverlayScaleTransform() {
        let arOn = arAssistedSizingEnabled && hasARKitAssistedSizingPayload && arAssistedScaleValid
        let allowRoomProportionFallback = arAssistedSizingEnabled && !QualitySettings.supportsLiDARSceneDepth
        let roomFactor: CGFloat = arOn ? 1.0 : (allowRoomProportionFallback ? autoScaleFromRoom : defaultStaticOverlayScale)
        let assistedScale: CGFloat = arOn ? autoScaleFromAR : 1.0
        let product = roomFactor * assistedScale * userPinchScale
        let clamped = min(max(product, minCombinedOverlayScale), maxCombinedOverlayScale)
        let finalTransform: CGAffineTransform
        if isShowingLiveVideoIdentifications {
            finalTransform = .identity
        } else {
            switch overlayPresentationMode {
            case .deferredCentered:
                // Scale around the maskImageView center, then translate so the
                // primary bbox center aligns with the view center (plus user pan
                // offset). Both room-proportion / static-default scaling AND user
                // pinch are included via `clamped`.
                let bboxCenter = CGPoint(x: primaryBboxInView.midX, y: primaryBboxInView.midY)
                let viewCenter = CGPoint(x: bounds.midX, y: bounds.midY)
                let autoTX = primaryBboxInView.width > 0 ? (viewCenter.x - bboxCenter.x) : 0
                let autoTY = primaryBboxInView.height > 0 ? (viewCenter.y - bboxCenter.y) : 0
                finalTransform = CGAffineTransform(scaleX: clamped, y: clamped)
                    .concatenating(CGAffineTransform(translationX: autoTX + userPanOffset.x,
                                                     y: autoTY + userPanOffset.y))
            case .measuredPlacement:
                finalTransform = CGAffineTransform(scaleX: clamped, y: clamped)
                    .concatenating(CGAffineTransform(translationX: userPanOffset.x,
                                                     y: userPanOffset.y))
            }
        }
        maskImageView.transform = finalTransform
        detectionBBoxOverlayView.transform = finalTransform

        let wantAR = arAssistedSizingEnabled && hasARKitAssistedSizingPayload && !userLockedAssistedOverlayScale
        let assistedLabel: String
        if arOn {
            assistedLabel = "AR"
        } else if wantAR {
            assistedLabel = "ROOM_PROP_AR_unavailable"
        } else if abs(autoScaleFromRoom - 1.0) > 0.02 {
            assistedLabel = "ROOM_PROP"
        } else {
            assistedLabel = "STATIC_DEFAULT"
        }
        let jump = overlayDebugLastCombined < 0 || abs(clamped - overlayDebugLastCombined) > 0.02
        let labelChange = assistedLabel != overlayDebugLastAssistedLabel
        if jump || labelChange {
            let loggedOverlayScale = isShowingLiveVideoIdentifications ? CGFloat(1.0) : clamped
            overlayDebugLastAssistedLabel = assistedLabel
            overlayDebugLastCombined = loggedOverlayScale
            let modeLabel: String
            if isShowingLiveVideoIdentifications {
                modeLabel = "full_video_identifications"
            } else {
                modeLabel = overlayPresentationMode == .deferredCentered ? "centered_pending" : "measured"
            }
            logFurnitureFitOverlay(
                "mode=\(modeLabel) assist=\(assistedLabel) roomStored=\(String(format: "%.3f", autoScaleFromRoom)) roomUsed=\(String(format: "%.3f", roomFactor)) ar=\(String(format: "%.3f", autoScaleFromAR)) pinch=\(String(format: "%.3f", userPinchScale)) → overlay=\(String(format: "%.3f", loggedOverlayScale)) wantAR=\(wantAR) arValid=\(arAssistedScaleValid)"
            )
        }
    }

    private func updateVideoIdentificationPresentation() {
        DispatchQueue.main.async {
            self.previewLayer.isHidden = !self.isShowingLiveVideoIdentifications
            self.applyLockedOrientationVideoRotation()
            self.applyCurrentOverlayScaleTransform()
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
            userPanOffset = .zero
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

        if shouldFreezeAutomaticOverlaySizing {
            applyCurrentOverlayScaleTransform()
            return
        }

        let bboxWidthImagePx = Float(max(1, bboxMaxX - bboxMinX))

        let wantAR = arAssistedSizingEnabled && hasARKitAssistedSizingPayload && !userLockedAssistedOverlayScale

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
                // Depth snapshot present but all depth lookups failed (non-LiDAR):
                // fall through to plane-raycast below.
            } else if let frame = arSession.currentFrame {
                let norm = CGPoint(x: nx, y: ny)
                distM = FurnitureFitARSupport.sceneDepthMeters(frame: frame, normalizedImagePoint: norm)
                if distM != nil {
                    distSource = "scene_depth"
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

            // Plane raycast fallback when no depth source has produced a distance yet
            // (non-LiDAR devices with snapshot but no depth map, or no scene depth).
            if distM == nil, #available(iOS 14.0, *), let frame = arSession.currentFrame {
                let sp = CGPoint(
                    x: nx * bounds.width,
                    y: ny * bounds.height
                )
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
                } else if distSource == "no_metric_distance" || distSource == "depth_snapshot_miss" || distSource.isEmpty {
                    distSource = "no_metric_distance"
                }
            }

            logFurnitureFitSize(
                "phase=ar_depth_resolve distSource=\(distSource) distM=\(distM.map { String(format: "%.3f", $0) } ?? "nil") " +
                "fx=\(String(format: "%.1f", fx)) fy=\(String(format: "%.1f", fy)) tracking=\(trackingName) " +
                "hasSnap=\(arDepthSnapshot != nil) snapDepth=\(arDepthSnapshot?.depthMap != nil) wantAR=\(wantAR)"
            )

            if let d = distM,
               let estH = FurnitureFitARSupport.estimatedPhysicalHeightMeters(
                bboxHeightPixels: bboxHeightImagePx,
                distanceMeters: d,
                focalLengthYPixels: fy
               ) {
                let effectiveARHeight = arAssistedSizingEnabled
                    ? max(estH, Self.minimumARFurnitureHeightMeters)
                    : estH
                let previousEstimatedHeightMeters = lastAREstimatedHeightMeters
                lastAREstimatedHeightMeters = effectiveARHeight
                let roomHeight = max(roomHeightMeters, 0.01)
                let currentImageHeightFraction = max(
                    CGFloat(bboxHeightImagePx) / CGFloat(max(1, imageHeight)),
                    0.01
                )
                let targetImageHeightFraction = CGFloat(
                    min(max(effectiveARHeight / roomHeight, 0.01), 0.98)
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
                        let effectiveARWidth = arAssistedSizingEnabled
                            ? max(estW, Self.minimumARFurnitureWidthMeters)
                            : estW
                        logFurnitureFitSize(
                            "phase=ar_overlay_refine distSource=\(distSource) dist_m=\(String(format: "%.3f", d)) W×H_m=\(String(format: "%.3f", effectiveARWidth))×\(String(format: "%.3f", effectiveARHeight)) metric_px=\(Int(bboxWidthImagePx))x\(Int(bboxHeightImagePx)) (requires AR companion + valid depth)"
                        )
                        onFurnitureSizeEstimated?(
                            makeFurnitureSizeEstimate(
                                widthMeters: effectiveARWidth,
                                heightMeters: effectiveARHeight,
                                arHeightMeters: effectiveARHeight
                            )
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
                            return abs(effectiveARHeight - previousEstimatedHeightMeters) / previousEstimatedHeightMeters
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
                    lastStableOverlayHeightMeters = effectiveARHeight
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
                            "distSource=\(distSource) dist=\(String(format: "%.2f", d))m furniture=\(String(format: "%.2f", estW))×\(String(format: "%.2f", effectiveARHeight))m roomH=\(String(format: "%.2f", roomHeight))m targetFrac=\(String(format: "%.3f", targetImageHeightFraction)) currentFrac=\(String(format: "%.3f", currentImageHeightFraction)) ratio=\(String(format: "%.3f", raw)) roomStored=\(String(format: "%.3f", autoScaleFromRoom)) roomUsed=\(String(format: "%.3f", roomUsedLog)) ar=\(String(format: "%.3f", autoScaleFromAR)) pinch=\(String(format: "%.3f", userPinchScale)) overlay=\(String(format: "%.3f", combinedOverlay)) tracking=\(trackingName) metricPx=\(Int(bboxWidthImagePx))×\(Int(bboxHeightImagePx)) fy=\(String(format: "%.1f", fy)) fx=\(String(format: "%.1f", fx))"
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

    private func resetOverlayScalesForEmptyMask(
        clearDetectedCandidates: Bool = true,
        clearSelections: Bool = false,
        resetUserOverlayScale: Bool = true
    ) {
        invalidatePendingAssistedMeasurement()
        autoScaleFromRoom = 1.0
        autoScaleFromAR = 1.0
        smoothedArOverlayScale = 1.0
        arAssistedConsecutiveMisses = 0
        arAssistedScaleValid = false
        overlayDebugLastAssistedLabel = ""
        overlayDebugLastCombined = -1
        lastAREstimatedHeightMeters = nil
        latestEstimatedFurnitureBaseWidthMeters = nil
        latestEstimatedFurnitureBaseHeightMeters = nil
        latestEstimatedFurnitureBaseARHeightMeters = nil
        didPublishARRefinedFurnitureMetersEstimate = false
        if resetUserOverlayScale {
            userPinchScale = 1.0
            userPanOffset = .zero
            userLockedAssistedOverlayScale = false
            lastOverlayPrimaryClassIdx = -1
            overlayPresentationMode = .deferredCentered
            stableOverlayMeasurementFrameCount = 0
            lastStableOverlayHeightMeters = nil
            lastStableOverlayScale = nil
        }
        smoothedTightPrimaryBbox = nil
        primaryBboxInView = .zero
        if clearDetectedCandidates {
            candidateBboxesInView = []
            latestDisplayedCandidates = []
            detectionBBoxOverlayView.items = []
            clearLatestTapMaskState()
        }
        latestDisplayedSelectedCandidateIndex = nil
        maskImageView.image = nil
        maskImageView.center = CGPoint(x: bounds.midX, y: bounds.midY)
        maskImageView.transform = .identity
        detectionBBoxOverlayView.transform = .identity
        if clearSelections {
            clearSelectedClassSelections()
        }
    }

    private func selectedPinsSnapshot() -> [FurnitureFitDetection] {
        selectedClassStateLock.lock()
        defer { selectedClassStateLock.unlock() }
        return selectedDetectionPins
    }

    private func hasSelectedClasses() -> Bool {
        !selectedPinsSnapshot().isEmpty
    }

    private func selectedClassLabelsSnapshot() -> [String] {
        selectedPinsSnapshot().map { displayClassName($0.classIdx) }
    }

    /// Maps each stored pin to the best-matching current-frame detection (same class, IoU ≥ threshold). Skips pins with no good match.
    private func matchedCandidatesForPins(
        candidates: [FurnitureFitDetection],
        pins: [FurnitureFitDetection]
    ) -> [FurnitureFitDetection] {
        guard !pins.isEmpty, !candidates.isEmpty else { return [] }
        var result: [FurnitureFitDetection] = []
        var usedCandidateIndices = Set<Int>()
        for pin in pins {
            var bestIdx: Int?
            var bestIoU: Float = 0
            for (i, cand) in candidates.enumerated() {
                guard cand.classIdx == pin.classIdx else { continue }
                let iou = FurnitureFitIoU.calculate(cand, pin)
                if iou > bestIoU {
                    bestIoU = iou
                    bestIdx = i
                }
            }
            guard let idx = bestIdx, bestIoU >= pinMatchIoUThreshold, !usedCandidateIndices.contains(idx) else { continue }
            usedCandidateIndices.insert(idx)
            result.append(candidates[idx])
        }
        return result
    }

    private func selectedClassChipTitle(from labels: [String]) -> String? {
        guard !labels.isEmpty else { return nil }
        if labels.count == 1 {
            return labels[0]
        }
        if labels.count == 2 {
            return "\(labels[0]), \(labels[1])"
        }
        return "\(labels.count) selected"
    }

    private func publishSelectedClassState() {
        let labels = selectedClassLabelsSnapshot()
        updateSelectedObjectChip(title: selectedClassChipTitle(from: labels))
        DispatchQueue.main.async {
            self.onSelectedClassLabelsChanged?(labels)
        }
    }

    private func updateSelectedObjectChip(title: String?) {
        DispatchQueue.main.async {
            self.selectedObjectChipButton.isHidden = true
            self.selectedObjectChipButton.alpha = 0
        }
    }

    private func updateDetectionOverlaySelectionHighlights() {
        let pins = selectedPinsSnapshot()
        detectionBBoxOverlayView.items = detectionBBoxOverlayView.items.enumerated().map { index, item in
            let isSel = index < latestDisplayedCandidates.count &&
                pins.contains { FurnitureFitIoU.calculate(latestDisplayedCandidates[index], $0) >= pinMatchIoUThreshold }
            return DetectionOverlayItem(
                rectInView: item.rectInView,
                label: item.label,
                confidence: item.confidence,
                isSelected: isSel
            )
        }
    }

    private func clearSelectedClassSelections() {
        selectedClassStateLock.lock()
        selectedDetectionPins.removeAll()
        selectedClassStateLock.unlock()
        publishSelectedClassState()
    }

    private func toggleSelectedDetection(_ detection: FurnitureFitDetection) {
        selectedClassStateLock.lock()
        if let idx = selectedDetectionPins.firstIndex(where: { FurnitureFitIoU.calculate($0, detection) >= 0.5 }) {
            selectedDetectionPins.remove(at: idx)
        } else {
            selectedDetectionPins.append(detection)
        }
        selectedClassStateLock.unlock()
        publishSelectedClassState()
    }

    /// Ultralytics-style `scale_boxes` mapping from model space back to the
    /// original image space. Supports both letterboxed-square inputs and the
    /// current stretch-to-square camera path.
    private func scaleBoxesFromModel(
        box: CGRect,
        modelShape: CGSize,
        imageShape: CGSize,
        usesLetterbox: Bool
    ) -> CGRect {
        let edgeOffset: CGFloat = 0.0  // edge bias disabled
        let x1: CGFloat
        let y1: CGFloat
        let x2: CGFloat
        let y2: CGFloat

        if usesLetterbox {
            // Ultralytics-style gain for square letterbox inputs.
            let gain = min(modelShape.width / imageShape.width, modelShape.height / imageShape.height)
            let padX = (modelShape.width - imageShape.width * gain) / 2.0
            let padY = (modelShape.height - imageShape.height * gain) / 2.0
            x1 = (box.minX - edgeOffset - padX) / gain
            y1 = (box.minY - edgeOffset - padY) / gain
            x2 = (box.maxX + edgeOffset - padX) / gain
            y2 = (box.maxY + edgeOffset - padY) / gain
        } else {
            // Stretch path: independent X/Y scaling, no padding.
            let gainX = imageShape.width / modelShape.width
            let gainY = imageShape.height / modelShape.height
            x1 = (box.minX - edgeOffset) * gainX
            y1 = (box.minY - edgeOffset) * gainY
            x2 = (box.maxX + edgeOffset) * gainX
            y2 = (box.maxY + edgeOffset) * gainY
        }

        let clippedX1 = max(0, x1)
        let clippedY1 = max(0, y1)
        let clippedX2 = min(imageShape.width, x2)
        let clippedY2 = min(imageShape.height, y2)

        return CGRect(
            x: clippedX1,
            y: clippedY1,
            width: max(1, clippedX2 - clippedX1),
            height: max(1, clippedY2 - clippedY1)
        )
    }

    private func bufferRect(
        for detection: FurnitureFitDetection,
        imageWidth: Int,
        imageHeight: Int,
        scaleX: Float,
        scaleY: Float
    ) -> CGRect {
        let modelShape = CGSize(
            width: CGFloat(Float(imageWidth) / max(scaleX, 1e-6)),
            height: CGFloat(Float(imageHeight) / max(scaleY, 1e-6))
        )
        let modelBox = CGRect(
            x: CGFloat(detection.x - detection.w * 0.5),
            y: CGFloat(detection.y - detection.h * 0.5),
            width: CGFloat(detection.w),
            height: CGFloat(detection.h)
        )
        let scaledBox = scaleBoxesFromModel(
            box: modelBox,
            modelShape: modelShape,
            imageShape: CGSize(width: imageWidth, height: imageHeight),
            usesLetterbox: currentYoloUsesLetterbox
        )
        let bx1 = max(0, Int(scaledBox.minX))
        let by1 = max(0, Int(scaledBox.minY))
        let bx2 = min(imageWidth, Int(scaledBox.maxX))
        let by2 = min(imageHeight, Int(scaledBox.maxY))
        return CGRect(
            x: bx1,
            y: by1,
            width: max(1, bx2 - bx1),
            height: max(1, by2 - by1)
        )
    }

    private func viewRect(
        for detection: FurnitureFitDetection,
        imageWidth: Int,
        imageHeight: Int,
        scaleX: Float,
        scaleY: Float
    ) -> CGRect {
        let bufferRect = bufferRect(
            for: detection,
            imageWidth: imageWidth,
            imageHeight: imageHeight,
            scaleX: scaleX,
            scaleY: scaleY
        )
        guard imageWidth > 0, imageHeight > 0 else { return .zero }
        let viewWidth = maskImageView.bounds.width
        let viewHeight = maskImageView.bounds.height
        return CGRect(
            x: bufferRect.minX / CGFloat(imageWidth) * viewWidth,
            y: bufferRect.minY / CGFloat(imageHeight) * viewHeight,
            width: bufferRect.width / CGFloat(imageWidth) * viewWidth,
            height: bufferRect.height / CGFloat(imageHeight) * viewHeight
        )
    }

    private func updateDetectionOverlay(
        candidates: [FurnitureFitDetection],
        selectedIndex: Int?,
        imageWidth: Int,
        imageHeight: Int,
        scaleX: Float,
        scaleY: Float
    ) {
        latestDisplayedCandidates = candidates
        latestDisplayedSelectedCandidateIndex = selectedIndex

        if segmentationMode == .segmentSelected || !showFullVideoWithIdentifications {
            candidateBboxesInView = []
            detectionBBoxOverlayView.items = []
            return
        }

        let rects = candidates.map {
            viewRect(
                for: $0,
                imageWidth: imageWidth,
                imageHeight: imageHeight,
                scaleX: scaleX,
                scaleY: scaleY
            )
        }
        let pins = selectedPinsSnapshot()
        candidateBboxesInView = rects
        detectionBBoxOverlayView.items = candidates.enumerated().map { index, detection in
            let isSel = pins.contains { FurnitureFitIoU.calculate(detection, $0) >= pinMatchIoUThreshold }
            return DetectionOverlayItem(
                rectInView: rects[index],
                label: displayClassName(detection.classIdx),
                confidence: detection.confidence,
                isSelected: isSel
            )
        }
    }

    private func updateLatestTapMaskState(
        planes: [Float],
        protoWidth: Int,
        protoHeight: Int,
        modelSide: Int,
        imageWidth: Int,
        imageHeight: Int
    ) {
        tapMaskStateLock.lock()
        latestTapMaskPlanes = planes
        latestTapMaskProtoWidth = protoWidth
        latestTapMaskProtoHeight = protoHeight
        latestTapMaskModelSide = modelSide
        latestTapMaskImageWidth = imageWidth
        latestTapMaskImageHeight = imageHeight
        tapMaskStateLock.unlock()
    }

    private func clearLatestTapMaskState() {
        tapMaskStateLock.lock()
        latestTapMaskPlanes = []
        latestTapMaskProtoWidth = 0
        latestTapMaskProtoHeight = 0
        latestTapMaskModelSide = 0
        latestTapMaskImageWidth = 0
        latestTapMaskImageHeight = 0
        tapMaskStateLock.unlock()
    }

    private func maskPresenceScoreForTap(
        detection: FurnitureFitDetection,
        pointInMaskView: CGPoint
    ) -> Float? {
        tapMaskStateLock.lock()
        let planes = latestTapMaskPlanes
        let protoWidth = latestTapMaskProtoWidth
        let protoHeight = latestTapMaskProtoHeight
        let modelSide = latestTapMaskModelSide
        let imageWidth = latestTapMaskImageWidth
        let imageHeight = latestTapMaskImageHeight
        tapMaskStateLock.unlock()

        guard !planes.isEmpty,
              protoWidth > 0,
              protoHeight > 0,
              modelSide > 0,
              imageWidth > 0,
              imageHeight > 0,
              detection.coeffs.count >= 32,
              maskImageView.bounds.width > 0,
              maskImageView.bounds.height > 0 else {
            return nil
        }

        let imageX = Float(pointInMaskView.x / maskImageView.bounds.width) * Float(imageWidth)
        let imageY = Float(pointInMaskView.y / maskImageView.bounds.height) * Float(imageHeight)
        let modelX = imageX * Float(modelSide) / Float(imageWidth)
        let modelY = imageY * Float(modelSide) / Float(imageHeight)

        let bboxHalfWidth = detection.w * 0.5
        let bboxHalfHeight = detection.h * 0.5
        let bboxMinX = detection.x - bboxHalfWidth
        let bboxMaxX = detection.x + bboxHalfWidth
        let bboxMinY = detection.y - bboxHalfHeight
        let bboxMaxY = detection.y + bboxHalfHeight
        guard modelX >= bboxMinX,
              modelX <= bboxMaxX,
              modelY >= bboxMinY,
              modelY <= bboxMaxY else {
            return nil
        }

        let protoX = min(
            protoWidth - 1,
            max(0, Int(floor(modelX * Float(protoWidth) / Float(modelSide))))
        )
        let protoY = min(
            protoHeight - 1,
            max(0, Int(floor(modelY * Float(protoHeight) / Float(modelSide))))
        )
        let hwProto = protoWidth * protoHeight
        let protoPixelIndex = protoY * protoWidth + protoX
        guard planes.count >= 32 * hwProto else { return nil }

        var sum: Float = 0
        var coeffIndex = 0
        while coeffIndex < 32 {
            let planeIndex = coeffIndex * hwProto + protoPixelIndex
            sum += detection.coeffs[coeffIndex] * planes[planeIndex]
            coeffIndex += 1
        }
        return sum > 0 ? sum : nil
    }

    private func candidateIndexForTap(_ pointInMaskView: CGPoint) -> Int? {
        guard !candidateBboxesInView.isEmpty, candidateBboxesInView.count == latestDisplayedCandidates.count else {
            return nil
        }
        tapMaskStateLock.lock()
        let hasTapMaskState =
            !latestTapMaskPlanes.isEmpty &&
            latestTapMaskProtoWidth > 0 &&
            latestTapMaskProtoHeight > 0 &&
            latestTapMaskModelSide > 0 &&
            latestTapMaskImageWidth > 0 &&
            latestTapMaskImageHeight > 0
        tapMaskStateLock.unlock()
        let paddedMatches = candidateBboxesInView.enumerated().filter { _, rect in
            rect.insetBy(dx: -10, dy: -10).contains(pointInMaskView)
        }
        guard !paddedMatches.isEmpty else { return nil }
        let maskMatches = paddedMatches.compactMap { match -> (offset: Int, rect: CGRect, score: Float)? in
            guard match.offset < latestDisplayedCandidates.count else { return nil }
            guard let score = maskPresenceScoreForTap(
                detection: latestDisplayedCandidates[match.offset],
                pointInMaskView: pointInMaskView
            ) else {
                return nil
            }
            return (offset: match.offset, rect: match.element, score: score)
        }
        if hasTapMaskState, maskMatches.isEmpty {
            return nil
        }
        let candidatesForSelection = maskMatches.isEmpty
            ? paddedMatches.map { (offset: $0.offset, rect: $0.element, score: Float.leastNormalMagnitude) }
            : maskMatches
        return candidatesForSelection.max { lhs, rhs in
            if abs(lhs.score - rhs.score) > 0.0001 {
                return lhs.score < rhs.score
            }
            let leftArea = lhs.rect.width * lhs.rect.height
            let rightArea = rhs.rect.width * rhs.rect.height
            if abs(leftArea - rightArea) > 1 {
                return leftArea > rightArea
            }
            return latestDisplayedCandidates[lhs.offset].confidence < latestDisplayedCandidates[rhs.offset].confidence
        }?.offset
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
            startupProgressActive = false
            DispatchQueue.main.async {
                self.progressContainer.isHidden = true
                self.progressContainer.alpha = 1
            }
        } else {
            hasFirstDetection = false
            startupProgressActive = true
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
        applyLockedOrientationVideoRotation()
    }

    /// Applies the same `videoRotationAngle` to ``AVCaptureVideoDataOutput`` and ``AVCaptureVideoPreviewLayer``
    /// so the live identify-mode feed matches the pixel buffers used for segmentation. Without syncing the
    /// preview layer, landscape-locked Sharp rooms could show a skewed or misaligned camera feed vs overlays.
    private func applyLockedOrientationVideoRotation() {
        guard !isUsingARCameraPath else { return }
        let angle: CGFloat = lockedOrientation == .landscape ? 0 : 90

        let applyVideoOutput: () -> Void = { [weak self] in
            guard let self else { return }
            guard let conn = self.videoOutput.connection(with: .video), conn.isVideoRotationAngleSupported(angle) else { return }
            conn.videoRotationAngle = angle
        }
        let applyPreview: () -> Void = { [weak self] in
            guard let self else { return }
            guard let previewConn = self.previewLayer.connection, previewConn.isVideoRotationAngleSupported(angle) else { return }
            previewConn.videoRotationAngle = angle
        }

        applyVideoOutput()
        if Thread.isMainThread {
            applyPreview()
        } else {
            DispatchQueue.main.async(execute: applyPreview)
        }
    }

    func stop() {
        invalidatePendingAssistedMeasurement()
        consecutiveEmptyMaskFrames = 0
        frameLock.lock()
        isProcessing = false
        preferImmediateNextInference = false
        frameLock.unlock()
        lastARHeavyWorkFinishCAC = 0
        // Synchronous reset: if UI/flags were only cleared in `main.async`, the next `startIfNeeded()` could run first and keep stale state.
        hasFirstDetection = false
        segmentationCompletedOnceThisSession = false
        startupProgressActive = false
        lastSegmentationMeanColorPublishAt = 0
        maskImageView.image = nil
        resetOverlayScalesForEmptyMask(clearDetectedCandidates: true, clearSelections: true)
        logFurnitureFitSize(
            "phase=session_stop cleared isProcessing + segmentation UI flags (fixes stuck frames + stale first-mask after brain off/on)"
        )
        asyncDepthSampler.stop()
        stopDeviceMotionPitchUpdates()
        furnitureFitCameraStartupInitiated = false
        arCompanionStartGeneration &+= 1
        suppressARDepthCompanionAfterCaptureFailure = false
        sharpRoomSplatMeasurementHost = nil
        NotificationCenter.default.removeObserver(self, name: UIDevice.orientationDidChangeNotification, object: nil)
        UIDevice.current.endGeneratingDeviceOrientationNotifications()

        DispatchQueue.main.async {
            self.isUsingARCameraPath = false
            self.isARDepthCompanionSessionRunning = false
            CameraOwnershipDiagnostics.log(owner: "FurnitureFitContainerView.ARSession", event: "ar_pause", details: "reason=stop")
            self.arSession.pause()
        }
        captureSessionControlQueue.async {
            if self.captureSession.isRunning {
                CameraOwnershipDiagnostics.log(owner: "FurnitureFitContainerView.AVCapture", event: "capture_stopRequested")
                self.captureSession.stopRunning()
            }
        }
    }

    /// Switch AR-assisted sizing mode without fully tearing down the Furniture Fit container.
    /// Full `stop()` resets segmentation UI state and can leave Sharp Room feeling frozen when the
    /// user taps the AR sizing icon after brain mode is already active.
    func reconfigureAssistedSizingModeIfNeeded() {
        guard furnitureFitCameraStartupInitiated || captureSession.isRunning || isUsingARCameraPath || isARDepthCompanionSessionRunning else {
            return
        }
        invalidatePendingAssistedMeasurement()
        if !arAssistedSizingEnabled {
            autoScaleFromAR = 1.0
            smoothedArOverlayScale = 1.0
            arAssistedScaleValid = false
            lastAREstimatedHeightMeters = nil
        }
        startPreferredCameraPathIfNeeded()
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
        captureSession.commitConfiguration()
        if lockedOrientation == .landscape {
            logDebug("📷 [Camera] Landscape room: using 0° rotation (1280x720)")
        } else {
            logDebug("📷 [Camera] Portrait room: using 90° rotation (720x1280)")
        }
        applyLockedOrientationVideoRotation()
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

    /// Sharp Room already supplies splat-surface depth.
    /// Keep Furniture Fit on the classic path there unless the viewer explicitly opts into AR sizing.
    private var shouldForceClassicCameraPath: Bool {
        sharpRoomSplatMeasurementHost != nil && !arAssistedSizingEnabled
    }

    private func startPreferredCameraPathIfNeeded() {
        let wantAR = arAssistedSizingEnabled
            && !shouldForceClassicCameraPath
            && AppStateManager.shared.qualitySettings.furnitureFitARDepthCompanionRuntimeActive
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
                    CameraOwnershipDiagnostics.log(owner: "FurnitureFitContainerView.ARSession", event: "ar_run", details: "reason=startARCameraPath lidar=\(hasLiDAR)")
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
            CameraOwnershipDiagnostics.log(owner: "FurnitureFitContainerView.ARSession", event: "ar_pause", details: "reason=startClassicCameraPath")
            self.arSession.pause()

            self.captureSessionControlQueue.async { [weak self] in
                guard let self else { return }
                self.setupCamera()
                if !self.captureSession.isRunning {
                    CameraOwnershipDiagnostics.log(owner: "FurnitureFitContainerView.AVCapture", event: "capture_startRequested")
                    self.captureSession.startRunning()
                }
                // Floor-contact depth estimation stays on the AVCapture path; no AR companion is started here.
                _ = companionLaunchToken
            }

            if self.debugMode {
                let reason = self.shouldForceClassicCameraPath
                    ? "sharp_room_uses_splat_depth"
                    : "ar_path_not_selected"
                logDebug("📷 [FurnitureFit] AVCaptureSession (classic); floor-contact depth estimate active reason=\(reason)")
            }

            if !Self.didLogFurnitureFitSizingPolicy {
                Self.didLogFurnitureFitSizingPolicy = true
                if self.shouldForceClassicCameraPath {
                    logFurnitureFitSize(
                        "policy=sharp_room_classic_camera: AVCapture video + SHARP/splat-aware sizing. No separate Furniture Fit ARSession when Sharp Room host is active."
                    )
                } else {
                    logFurnitureFitSize(
                        "policy=floor_contact_only: AVCapture video + floor-contact depth heuristic from bbox/mask bottom. AR path not selected."
                    )
                }
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

    /// Sharp Room: depth to the splat surface at the floor-contact pixel (matches the rendered splat camera). Caller must be on the main thread.
    private func liveRoomSplatDepthMetersForFloorContactIfAvailable(
        contactImageX: Float,
        contactImageY: Float,
        imageWidth: Int,
        imageHeight: Int
    ) -> Float? {
        guard let host = sharpRoomSplatMeasurementHost,
              let rm = roomModel,
              rm.sceneToMeters > 1e-5,
              window != nil else { return nil }

        let dest = maskImageView.convert(maskImageView.bounds, to: nil)
        guard dest.width > 2, dest.height > 2 else { return nil }

        let iw = CGFloat(imageWidth)
        let ih = CGFloat(imageHeight)
        let scale = max(dest.width / iw, dest.height / ih)
        let dispW = iw * scale
        let dispH = ih * scale
        let ox = dest.midX - dispW * 0.5
        let oy = dest.midY - dispH * 0.5
        let px = (CGFloat(contactImageX) + 0.5) / iw
        let py = (CGFloat(contactImageY) + 0.5) / ih
        let windowPoint = CGPoint(x: ox + px * dispW, y: oy + py * dispH)

        guard let sceneDist = host.splatCameraToSurfaceDistanceSceneUnits(atWindowPoint: windowPoint),
              sceneDist.isFinite,
              sceneDist > 0.02,
              sceneDist < 500 else { return nil }

        let meters = sceneDist * rm.sceneToMeters
        guard meters.isFinite, meters > 0.15, meters < 40 else { return nil }
        return meters
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
        _ = preferRoomRaycastSizing
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

        let liveRoomSplatDepthMeters: Float? = {
            if Thread.isMainThread {
                return liveRoomSplatDepthMetersForFloorContactIfAvailable(
                    contactImageX: clampedImageX,
                    contactImageY: clampedImageY,
                    imageWidth: imageWidth,
                    imageHeight: imageHeight
                )
            }
            var out: Float?
            DispatchQueue.main.sync {
                out = self.liveRoomSplatDepthMetersForFloorContactIfAvailable(
                    contactImageX: clampedImageX,
                    contactImageY: clampedImageY,
                    imageWidth: imageWidth,
                    imageHeight: imageHeight
                )
            }
            return out
        }()

        let depthMeters: Float
        let measuredPitchDeg: Float?
        let effectivePitchDeg: Float
        let depthPipelineSuffix: String
        if let dm = liveRoomSplatDepthMeters {
            depthMeters = dm
            measuredPitchDeg = nil
            effectivePitchDeg = 0
            depthPipelineSuffix = "_splat"
            logFurnitureFitSize(
                "phase=metric_depth source=splat depth_m=\(String(format: "%.3f", dm)) " +
                    "contact_px=(\(Int(clampedImageX)),\(Int(clampedImageY))) buf=\(imageWidth)×\(imageHeight)"
            )
        } else {
            let depthEstimate = floorContactDepthEstimateMeters(
                contactImageY: clampedImageY,
                imageHeight: imageHeight,
                focalPixels: focalLengthY
            )
            depthMeters = depthEstimate.depthMeters
            measuredPitchDeg = depthEstimate.measuredPitchDeg
            effectivePitchDeg = depthEstimate.effectivePitchDeg
            depthPipelineSuffix =
                "_pitch\(String(format: "%.1f", depthEstimate.effectivePitchDeg))" +
                (depthEstimate.measuredPitchDeg != nil ? "_motion" : "_fallback")
        }

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
            guard roomWidthMeters.isFinite, roomHeightMeters.isFinite,
                  roomWidthMeters > 0.05, roomHeightMeters > 0.05 else { return nil }
            let w = (widthPixels / Float(imageWidth)) * roomWidthMeters
            let h = (heightPixels / Float(imageHeight)) * roomHeightMeters
            guard w.isFinite, h.isFinite, w > 0.01, h > 0.01 else { return nil }
            return (w, h, "room_bbox_proportion")
        }()

        let selectedMetricSize: (widthMeters: Float, heightMeters: Float, pipeline: String)?
        if let preferredRoomMetricSize,
           isReasonableFurnitureMetricEstimate(
            widthMeters: preferredRoomMetricSize.widthMeters,
            heightMeters: preferredRoomMetricSize.heightMeters,
            distanceMeters: depthMeters,
            pipeline: preferredRoomMetricSize.pipeline
           ) {
            logFurnitureFitSize(
                "phase=metric_policy policy=room_model_width_prop_height depth_m=\(String(format: "%.3f", depthMeters)) " +
                "selected=\(preferredRoomMetricSize.pipeline)"
            )
            selectedMetricSize = preferredRoomMetricSize
        } else if let fallbackMetricSize,
                  isReasonableFurnitureMetricEstimate(
                    widthMeters: fallbackMetricSize.widthMeters,
                    heightMeters: fallbackMetricSize.heightMeters,
                    distanceMeters: depthMeters,
                    pipeline: fallbackMetricSize.pipeline
                  ) {
            logFurnitureFitSize(
                "phase=metric_policy policy=room_bbox_proportion depth_m=\(String(format: "%.3f", depthMeters))"
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
                pipeline: sizingPipelinePrefix + depthPipelineSuffix,
                distanceMeters: depthMeters
            ),
            contactImageX: Int(round(clampedImageX)),
            contactImageY: Int(round(clampedImageY)),
            widthPixels: widthPixels,
            heightPixels: heightPixels,
            focalPixels: focalLengthY,
            measuredPitchDeg: measuredPitchDeg,
            effectivePitchDeg: effectivePitchDeg
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
        let heightMeters = (heightPixels / Float(imageHeight)) * roomHeightMeters

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
            "W_m=\(String(format: "%.3f", widthMeters)) H_m_prop=\(String(format: "%.3f", heightMeters))"
        )

        return (widthMeters, heightMeters, "room_model_intrinsics_width_prop_height")
    }

    private func imageBoundsForMask(
        maskSmall: [UInt8],
        protoW: Int,
        protoH: Int,
        imageWidth: Int,
        imageHeight: Int
    ) -> CGRect? {
        guard protoW > 0,
              protoH > 0,
              imageWidth > 0,
              imageHeight > 0,
              maskSmall.count >= protoW * protoH else {
            return nil
        }

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

        let minImageX = max(0, Int(floor(Float(minProtoX) * Float(imageWidth) / Float(protoW))))
        let minImageY = max(0, Int(floor(Float(minProtoY) * Float(imageHeight) / Float(protoH))))
        let maxImageX = min(imageWidth, Int(ceil(Float(maxProtoX + 1) * Float(imageWidth) / Float(protoW))))
        let maxImageY = min(imageHeight, Int(ceil(Float(maxProtoY + 1) * Float(imageHeight) / Float(protoH))))

        guard maxImageX > minImageX, maxImageY > minImageY else { return nil }
        return CGRect(
            x: minImageX,
            y: minImageY,
            width: maxImageX - minImageX,
            height: maxImageY - minImageY
        )
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

    /// ONNX-style pipeline using Core ML ``mlModel``. The legacy 1280 package
    /// (`yoloe-26l-seg-pf`) expects Ultralytics-style letterbox; the newer
    /// `_seg_o2m` package keeps the Android-parity stretch path.
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

        let usesLetterboxPreprocess = modelInputSize >= 1280
        currentYoloUsesLetterbox = usesLetterboxPreprocess

        if debugMode {
            logDebug("\n⏱️ ═══════════════════════════════════════════")
            logDebug("⏱️ ONNX-STYLE + CORE ML @ \(String(format: "%.3f", frameStart.timeIntervalSince1970))")
            logDebug(
                "⏱️ Buffer: \(bufW)x\(bufH) → \(usesLetterboxPreprocess ? "letterbox" : "stretch") " +
                "\(modelInputSize)x\(modelInputSize), isLandscape: \(isLandscape)"
            )
            logDebug("⏱️ ═══════════════════════════════════════════")
        }

        setProgress(0.15, text: "Preprocessing…")
        let t1 = Date()
        let preparedBuffer = usesLetterboxPreprocess
            ? resizeLetterboxToSquare(processBuffer, size: modelInputSize)
            : resizeStretchToSquare(processBuffer, size: modelInputSize)
        guard let modelInputBuffer = preparedBuffer else {
            if debugMode {
                logDebug("❌ ONNX-STYLE Core ML STAGE 1 FAILED: \(usesLetterboxPreprocess ? "letterbox" : "stretch") resize")
            }
            resetProcessingFlag()
            return
        }
        if debugMode {
            logDebug(
                "⏱️ ONNX-STYLE STAGE 1 - Preprocess (\(usesLetterboxPreprocess ? "letterbox" : "stretch")): " +
                "\(String(format: "%.2f", Date().timeIntervalSince(t1) * 1000)) ms"
            )
        }

        let inputDesc = model.modelDescription.inputDescriptionsByName["image"]
        let expectsImage = inputDesc?.type == .image

        let inputProvider: MLFeatureProvider
        if expectsImage {
            guard let imageValue = MLFeatureValue(pixelBuffer: modelInputBuffer) as MLFeatureValue?,
                  let provider = try? MLDictionaryFeatureProvider(dictionary: ["image": imageValue]) else {
                if debugMode { logDebug("❌ ONNX-STYLE Core ML STAGE 1 FAILED: Input prep") }
                resetProcessingFlag()
                return
            }
            inputProvider = provider
        } else {
            guard let inputArray = pixelBufferToMLMultiArray(modelInputBuffer),
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

    /// Primary index: among candidates with confidence ≥ ``primaryDetectionMinConfidence``, either highest confidence or largest bbox area (see ``primarySelectionByHighestConfidence``).
    /// (`modelSide` unused; kept for call-site stability with the ONNX-style pipeline.)
    private func selectPrimaryIndexCoreFlow(candidates: [FurnitureFitDetection], modelSide _: Int) -> Int? {
        let minConfidence = min(max(primaryDetectionMinConfidence, 0.05), 0.99)
        if primarySelectionByHighestConfidence {
            var bestIdx: Int?
            var bestConfidence: Float = -1
            var bestAreaAtBestConfidence: Float = 0
            for (i, d) in candidates.enumerated() {
                guard d.confidence >= minConfidence else { continue }
                let area = d.w * d.h
                if d.confidence > bestConfidence {
                    bestConfidence = d.confidence
                    bestAreaAtBestConfidence = area
                    bestIdx = i
                } else if abs(d.confidence - bestConfidence) <= 1e-6, area > bestAreaAtBestConfidence {
                    bestAreaAtBestConfidence = area
                    bestIdx = i
                }
            }
            return bestIdx
        }
        var bestIdx: Int?
        var maxArea: Float = 0
        for (i, d) in candidates.enumerated() {
            guard d.confidence >= minConfidence else { continue }
            let area = d.w * d.h
            if area > maxArea {
                maxArea = area
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
        let scaleX = Float(bufW) / Float(onnxSide)
        let scaleY = Float(bufH) / Float(onnxSide)

        let t3 = Date()
        guard let protoInfo = parsePrototypes(protoArray) else {
            if debugMode { logDebug("❌ ONNX-STYLE (\(stage2DebugLabel)): parse prototypes failed") }
            resetProcessingFlag()
            return
        }
        let planes = protoInfo.planes
        let pW = protoInfo.width
        let pH = protoInfo.height
        updateLatestTapMaskState(
            planes: planes,
            protoWidth: pW,
            protoHeight: pH,
            modelSide: onnxSide,
            imageWidth: bufW,
            imageHeight: bufH
        )

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
            consecutiveEmptyMaskFrames += 1
            if consecutiveEmptyMaskFrames > maskGraceFrameLimit {
                DispatchQueue.main.async {
                    self.resetOverlayScalesForEmptyMask(clearDetectedCandidates: true, clearSelections: false)
                    self.finishStartupProgressIfNeeded()
                }
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
            consecutiveEmptyMaskFrames += 1
            if consecutiveEmptyMaskFrames > maskGraceFrameLimit {
                DispatchQueue.main.async {
                    self.resetOverlayScalesForEmptyMask(clearDetectedCandidates: true, clearSelections: false)
                }
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
        logChairDetectionMasksIfDebug(
            planes: planes,
            interleavedPlanes: protoInfo.interleavedPlanes,
            protoW: pW,
            protoH: pH,
            modelSide: onnxSide,
            detections: candidates,
            protoShape: protoInfo.shape,
            protoStrides: protoInfo.strides,
            protoChannelAxis: protoInfo.channelAxis
        )

        if segmentationMode == .identifyOnly && showFullVideoWithIdentifications {
            // Full-video identifications ON: show bbox overlays only, no mask compositing.
            let primaryIdxOpt = selectPrimaryIndexCoreFlow(candidates: candidates, modelSide: onnxSide)
            DispatchQueue.main.async {
                self.resetOverlayScalesForEmptyMask(
                    clearDetectedCandidates: false,
                    clearSelections: false,
                    resetUserOverlayScale: false
                )
                self.updateDetectionOverlay(
                    candidates: candidates,
                    selectedIndex: primaryIdxOpt,
                    imageWidth: bufW,
                    imageHeight: bufH,
                    scaleX: scaleX,
                    scaleY: scaleY
                )
                if let p = primaryIdxOpt {
                    self.primaryBboxInView = self.viewRect(
                        for: candidates[p],
                        imageWidth: bufW,
                        imageHeight: bufH,
                        scaleX: scaleX,
                        scaleY: scaleY
                    )
                } else {
                    self.primaryBboxInView = .zero
                }
                self.applyCurrentOverlayScaleTransform()
                self.finishStartupProgressIfNeeded()
            }
            resetProcessingFlag()
            return
        }

        // Determine selected candidates + primary for mask compositing.
        let selectedCandidates: [FurnitureFitDetection]
        let primary: FurnitureFitDetection
        let primaryIdx: Int

        if segmentationMode == .identifyOnly {
            // Full-video OFF: auto-select single primary by confidence/area (old behavior).
            guard let autoIdx = selectPrimaryIndexCoreFlow(candidates: candidates, modelSide: onnxSide) else {
                if debugMode { logDebug("⚠️ ONNX-STYLE (\(stage2DebugLabel)): no auto-primary among candidates") }
                consecutiveEmptyMaskFrames += 1
                if consecutiveEmptyMaskFrames > maskGraceFrameLimit {
                    DispatchQueue.main.async {
                        self.resetOverlayScalesForEmptyMask(clearDetectedCandidates: false, clearSelections: false, resetUserOverlayScale: false)
                    }
                }
                resetProcessingFlag()
                return
            }
            primary = candidates[autoIdx]
            primaryIdx = autoIdx
            selectedCandidates = [primary]
            DispatchQueue.main.async {
                self.updateDetectionOverlay(
                    candidates: candidates,
                    selectedIndex: autoIdx,
                    imageWidth: bufW,
                    imageHeight: bufH,
                    scaleX: scaleX,
                    scaleY: scaleY
                )
            }
        } else {
            // segmentSelected: user-pinned selections.
            DispatchQueue.main.async {
                self.updateDetectionOverlay(
                    candidates: candidates,
                    selectedIndex: nil,
                    imageWidth: bufW,
                    imageHeight: bufH,
                    scaleX: scaleX,
                    scaleY: scaleY
                )
            }

            let pins = selectedPinsSnapshot()
            let matchedCandidates = matchedCandidatesForPins(candidates: candidates, pins: pins)
            guard !matchedCandidates.isEmpty else {
                if debugMode { logDebug("⚠️ ONNX-STYLE (\(stage2DebugLabel)): segment mode active but no selected instances match current frame (IoU≥\(pinMatchIoUThreshold))") }
                consecutiveEmptyMaskFrames += 1
                if consecutiveEmptyMaskFrames > maskGraceFrameLimit {
                    DispatchQueue.main.async {
                        self.resetOverlayScalesForEmptyMask(
                            clearDetectedCandidates: false,
                            clearSelections: false,
                            resetUserOverlayScale: false
                        )
                        self.updateDetectionOverlay(
                            candidates: candidates,
                            selectedIndex: nil,
                            imageWidth: bufW,
                            imageHeight: bufH,
                            scaleX: scaleX,
                            scaleY: scaleY
                        )
                    }
                }
                resetProcessingFlag()
                return
            }

            guard let selectedPrimaryRelativeIndex = selectPrimaryIndexCoreFlow(candidates: matchedCandidates, modelSide: onnxSide) else {
                if debugMode { logDebug("⚠️ ONNX-STYLE (\(stage2DebugLabel)): no primary candidate among matched instances") }
                consecutiveEmptyMaskFrames += 1
                if consecutiveEmptyMaskFrames > maskGraceFrameLimit {
                    DispatchQueue.main.async {
                        self.resetOverlayScalesForEmptyMask(
                            clearDetectedCandidates: false,
                            clearSelections: false,
                            resetUserOverlayScale: false
                        )
                        self.updateDetectionOverlay(
                            candidates: candidates,
                            selectedIndex: nil,
                            imageWidth: bufW,
                            imageHeight: bufH,
                            scaleX: scaleX,
                            scaleY: scaleY
                        )
                    }
                }
                resetProcessingFlag()
                return
            }
            let matched = matchedCandidates[selectedPrimaryRelativeIndex]
            guard let pIdx = candidates.firstIndex(where: { FurnitureFitIoU.calculate($0, matched) >= 0.99 }) else {
                if debugMode { logDebug("⚠️ ONNX-STYLE (\(stage2DebugLabel)): primary not found in full candidate list") }
                resetProcessingFlag()
                return
            }
            primary = matched
            primaryIdx = pIdx
            selectedCandidates = matchedCandidates
        }

        // Build the proto-space mask from a fusion list of detections.
        // - When running in identify-only mode (no manual selection), mirror the
        //   Android path by fusing the primary with nearby overlapping detections
        //   via `collectMaskDetections`, and expanding the primary bbox slightly
        //   for legs/wheels before synthesizing logits.
        // - When running in segmentSelected mode, restrict fusion to the
        //   user-pinned instances, but still apply the primary expansion per pin.
        let maskDetectionsForBuild: [FurnitureFitDetection]
        if segmentationMode == .segmentSelected {
            // Manual selection: expand each matched pin and build a mask over the
            // union of those expanded boxes.
            maskDetectionsForBuild = selectedCandidates.map {
                onnxStyleExpandedPrimaryForMaskBuild($0, onnxSide: onnxSide)
            }
        } else {
            // Auto primary: Android-style fusion of primary + supporting
            // detections that intersect it (monitor/table, overlapped props,
            // etc.), with an expanded primary bbox for mask synthesis.
            let maskSource = [primary]  // Temporary debug isolation: primary only
            let expandedPrimary = onnxStyleExpandedPrimaryForMaskBuild(primary, onnxSide: onnxSide)
            var fusion: [FurnitureFitDetection] = [expandedPrimary]
            for det in maskSource {
                // Drop the unexpanded primary (near-duplicate) while keeping
                // genuinely distinct overlapping detections.
                if FurnitureFitIoU.calculate(det, primary) < 0.999 {
                    fusion.append(det)
                }
            }
            maskDetectionsForBuild = fusion
        }

        let maskResult = FurnitureFitOnnxStylePipeline.buildFullFieldLogitMask(
            planes: planes,
            protoW: pW,
            protoH: pH,
            detections: maskDetectionsForBuild
        )
        var maskSmall = maskResult.binary
        if furnitureFitUseMorphologicalCloseMask {
            maskSmall = morphologicalBinaryClose3x3Planar8(mask: maskSmall, width: pW, height: pH)
        }

        let primaryBufferRect = bufferRect(
            for: primary,
            imageWidth: bufW,
            imageHeight: bufH,
            scaleX: scaleX,
            scaleY: scaleY
        )
        let primaryBx1 = Int(primaryBufferRect.minX)
        let primaryBy1 = Int(primaryBufferRect.minY)
        let primaryBx2 = Int(primaryBufferRect.maxX)
        let primaryBy2 = Int(primaryBufferRect.maxY)

        // Build the composite band from the fused detection boxes rather than the
        // thresholded small mask. This avoids cropping away weak thin parts (for
        // example the left handle of a chair) before full-res logits are sampled.
        let clipCandidates = segmentationMode == .segmentSelected ? selectedCandidates : maskDetectionsForBuild
        let clipLeftModel = clipCandidates.map { $0.x - $0.w * 0.5 }.min() ?? (primary.x - primary.w * 0.5)
        let clipTopModel = clipCandidates.map { $0.y - $0.h * 0.5 }.min() ?? (primary.y - primary.h * 0.5)
        let clipRightModel = clipCandidates.map { $0.x + $0.w * 0.5 }.max() ?? (primary.x + primary.w * 0.5)
        let clipBottomModel = clipCandidates.map { $0.y + $0.h * 0.5 }.max() ?? (primary.y + primary.h * 0.5)
        var maskSmallForComposite = maskSmall
        var maskLogitsForComposite = maskResult.logits

        let clipModelRect = CGRect(
            x: CGFloat(clipLeftModel),
            y: CGFloat(clipTopModel),
            width: CGFloat(max(clipRightModel - clipLeftModel, 1)),
            height: CGFloat(max(clipBottomModel - clipTopModel, 1))
        )
        let clipBufferRect = scaleBoxesFromModel(
            box: clipModelRect,
            modelShape: CGSize(width: CGFloat(onnxSide), height: CGFloat(onnxSide)),
            imageShape: CGSize(width: bufW, height: bufH),
            usesLetterbox: currentYoloUsesLetterbox
        )
        let bandMarginW: Float = 0  // edge band expansion disabled
        let bandMarginH: Float = 0  // edge band expansion disabled
        let cropMarginPx = FurnitureFitOnnxStylePipeline.nativeCompositeBandMarginPx
        let compBx1 = max(0, Int(floor(Float(clipBufferRect.minX) - bandMarginW)) - cropMarginPx)
        let compBy1 = max(0, Int(floor(Float(clipBufferRect.minY) - bandMarginH)) - cropMarginPx)
        let compBx2 = min(bufW, Int(ceil(Float(clipBufferRect.maxX) + bandMarginW)) + cropMarginPx)
        let compBy2 = min(bufH, Int(ceil(Float(clipBufferRect.maxY) + bandMarginH)) + cropMarginPx)

        if debugMode {
            let stage35Ms = Date().timeIntervalSince(t3) * 1000
            logDebug("⏱️ ONNX-STYLE (\(stage2DebugLabel)) STAGE 3–5 - parse/NMS/mask: \(String(format: "%.2f", stage35Ms)) ms")
        }

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

        if arAssistedSizingEnabled && hasARKitAssistedSizingPayload && arAssistedScaleValid {
            arLabel = "valid(\(String(format: "%.3f", autoScaleFromAR)))"
        } else if arAssistedSizingEnabled && hasARKitAssistedSizingPayload {
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
            self.primaryBboxInView = self.viewRect(
                for: primary,
                imageWidth: bufW,
                imageHeight: bufH,
                scaleX: scaleX,
                scaleY: scaleY
            )
            self.updateDetectionOverlay(
                candidates: candidates,
                selectedIndex: primaryIdx,
                imageWidth: bufW,
                imageHeight: bufH,
                scaleX: scaleX,
                scaleY: scaleY
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
        logChairFusedLogitNumericGridsIfDebug(
            maskLogits: maskLogitsForComposite,
            protoW: pW,
            protoH: pH,
            modelSide: onnxSide,
            primary: primary,
            origW: bufW,
            origH: bufH,
            primaryBx1: primaryBx1,
            primaryBy1: primaryBy1,
            primaryBx2: primaryBx2,
            primaryBy2: primaryBy2
        )
        let composedImage: CGImage? = currentYoloUsesLetterbox
            ? compositeLegacy1280UnionMaskCutout(
                processBuffer: processBuffer,
                planes: planes,
                protoW: pW,
                protoH: pH,
                modelSide: onnxSide,
                origW: bufW,
                origH: bufH,
                maskDetectionsForBuild: maskDetectionsForBuild
            )
            : (compositeGpuNativeMaskCutout(
                processBuffer: processBuffer,
                maskProto: maskSmallForComposite,
                maskLogits: maskLogitsForComposite,
                protoW: pW,
                protoH: pH,
                origW: bufW,
                origH: bufH,
                x0: compBx1,
                x1: compBx2,
                y0: compBy1,
                y1: compBy2,
                primary: primary,
                primaryBx1: primaryBx1,
                primaryBy1: primaryBy1,
                primaryBx2: primaryBx2,
                primaryBy2: primaryBy2,
                debugTag: "process_mask_native GPU"
            ) ?? compositeCpuBilinearProtoMaskCutoutFromLogits(
                processBuffer: processBuffer,
                maskProto: maskSmallForComposite,
                maskLogits: maskLogitsForComposite,
                planes: planes,
                protoW: pW,
                protoH: pH,
                modelSide: onnxSide,
                origW: bufW,
                origH: bufH,
                x0: compBx1,
                x1: compBx2,
                y0: compBy1,
                y1: compBy2,
                primary: primary,
                primaryBx1: primaryBx1,
                primaryBy1: primaryBy1,
                primaryBx2: primaryBx2,
                primaryBy2: primaryBy2,
                maskDetectionsForBuild: maskDetectionsForBuild,
                debugTag: "ONNX-style CPU fallback"
            ))

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
            consecutiveEmptyMaskFrames = 0
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
                self.commitFurnitureSizeAfterSegmentationMaskApplied(
                    maskHasForeground: maskHasForeground,
                    primaryMetricResult: finalMetricResult,
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
                self.commitFurnitureSizeAfterSegmentationMaskApplied(
                    maskHasForeground: maskHasForeground,
                    primaryMetricResult: finalMetricResult,
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
        let ew = min(primary.w, capW)  // edge expansion disabled
        let eh = min(primary.h, capH)  // edge expansion disabled
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

    private func clipProtoFloatMaskOutsideRect(
        mask: inout [Float],
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
            mask = [Float](repeating: 0, count: protoW * protoH)
            return
        }
        if clipY1 > 0 {
            for index in 0..<(clipY1 * protoW) {
                mask[index] = 0
            }
        }
        if clipY2 < protoH {
            for index in (clipY2 * protoW)..<(protoH * protoW) {
                mask[index] = 0
            }
        }
        if clipX1 > 0 || clipX2 < protoW {
            for y in clipY1..<clipY2 {
                let rowStart = y * protoW
                if clipX1 > 0 {
                    for x in 0..<clipX1 {
                        mask[rowStart + x] = 0
                    }
                }
                if clipX2 < protoW {
                    for x in clipX2..<protoW {
                        mask[rowStart + x] = 0
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

    // MARK: - Small-region post-processing (holes / islands)

    /// Fills small background "holes" (0-valued regions fully enclosed by 255) in a
    /// band-local binary mask. Mirrors SAM's `remove_small_regions(..., mode="holes")`.
    private func fillSmallHolesInMaskBand(
        mask: inout [UInt8],
        width: Int,
        height: Int,
        areaThreshold: Int
    ) {
        guard width > 0, height > 0, mask.count >= width * height, areaThreshold > 0 else { return }

        let w = width
        let h = height
        var visited = [Bool](repeating: false, count: w * h)
        var queue = [(x: Int, y: Int)]()
        queue.reserveCapacity(128)

        func idx(_ x: Int, _ y: Int) -> Int { y * w + x }

        // 8-neighbour BFS on background (0) regions (matches SAM connectivity)
        for startY in 0..<h {
            for startX in 0..<w {
                let startIndex = idx(startX, startY)
                if visited[startIndex] || mask[startIndex] != 0 {
                    continue
                }

                queue.removeAll(keepingCapacity: true)
                queue.append((startX, startY))
                visited[startIndex] = true

                var component: [Int] = [startIndex]
                var touchesBorder = (startX == 0 || startX == w - 1 || startY == 0 || startY == h - 1)

                while !queue.isEmpty {
                    let (x, y) = queue.removeLast()
                    // 8-connected neighbours: N, S, E, W, and 4 diagonals.
                    let neighbours = [
                        (x: x - 1, y: y),
                        (x: x + 1, y: y),
                        (x: x,     y: y - 1),
                        (x: x,     y: y + 1),
                        (x: x - 1, y: y - 1),
                        (x: x + 1, y: y - 1),
                        (x: x - 1, y: y + 1),
                        (x: x + 1, y: y + 1),
                    ]
                    for n in neighbours {
                        guard n.x >= 0, n.x < w, n.y >= 0, n.y < h else { continue }
                        let nIndex = idx(n.x, n.y)
                        if visited[nIndex] || mask[nIndex] != 0 { continue }
                        visited[nIndex] = true
                        queue.append((n.x, n.y))
                        component.append(nIndex)
                        if n.x == 0 || n.x == w - 1 || n.y == 0 || n.y == h - 1 {
                            touchesBorder = true
                        }
                    }
                }

                // If this background component does NOT touch the band border and
                // is smaller than the threshold, treat it as a "hole" and fill it.
                if !touchesBorder && component.count < areaThreshold {
                    for i in component {
                        mask[i] = 255
                    }
                }
            }
        }
    }

    /// Removes small foreground "islands" (255-valued blobs) in a band-local
    /// binary mask. Mirrors SAM's `remove_small_regions(..., mode="islands")`.
    private func removeSmallIslandsInMaskBand(
        mask: inout [UInt8],
        width: Int,
        height: Int,
        areaThreshold: Int
    ) {
        guard width > 0, height > 0, mask.count >= width * height, areaThreshold > 0 else { return }

        let w = width
        let h = height
        var visited = [Bool](repeating: false, count: w * h)
        var queue = [(x: Int, y: Int)]()
        queue.reserveCapacity(128)

        func idx(_ x: Int, _ y: Int) -> Int { y * w + x }

        // 8-neighbour BFS on foreground (255) regions (matches SAM connectivity)
        for startY in 0..<h {
            for startX in 0..<w {
                let startIndex = idx(startX, startY)
                if visited[startIndex] || mask[startIndex] == 0 {
                    continue
                }

                queue.removeAll(keepingCapacity: true)
                queue.append((startX, startY))
                visited[startIndex] = true

                var component: [Int] = [startIndex]

                while !queue.isEmpty {
                    let (x, y) = queue.removeLast()
                    // 8-connected neighbours: N, S, E, W, and 4 diagonals.
                    let neighbours = [
                        (x: x - 1, y: y),
                        (x: x + 1, y: y),
                        (x: x,     y: y - 1),
                        (x: x,     y: y + 1),
                        (x: x - 1, y: y - 1),
                        (x: x + 1, y: y - 1),
                        (x: x - 1, y: y + 1),
                        (x: x + 1, y: y + 1),
                    ]
                    for n in neighbours {
                        guard n.x >= 0, n.x < w, n.y >= 0, n.y < h else { continue }
                        let nIndex = idx(n.x, n.y)
                        if visited[nIndex] || mask[nIndex] == 0 { continue }
                        visited[nIndex] = true
                        queue.append((n.x, n.y))
                        component.append(nIndex)
                    }
                }

                // Foreground island smaller than the threshold → remove it.
                if component.count < areaThreshold {
                    for i in component {
                        mask[i] = 0
                    }
                }
            }
        }
    }

    /// Simple binary dilation on a band-local mask (3×3 structuring element).
    private func dilateBandMask(
        mask: inout [UInt8],
        width: Int,
        height: Int,
        iterations: Int
    ) {
        guard width > 0, height > 0, mask.count >= width * height, iterations > 0 else { return }

        let w = width
        let h = height
        var scratch = mask

        for _ in 0..<iterations {
            scratch = mask
            for y in 0..<h {
                for x in 0..<w {
                    let idx = y * w + x
                    if scratch[idx] == 255 {
                        mask[idx] = 255
                        continue
                    }
                    var found = false
                    for ny in max(0, y - 1)...min(h - 1, y + 1) {
                        for nx in max(0, x - 1)...min(w - 1, x + 1) {
                            if scratch[ny * w + nx] == 255 {
                                found = true
                                break
                            }
                        }
                        if found { break }
                    }
                    if found {
                        mask[idx] = 255
                    }
                }
            }
        }
    }

    /// Simple binary erosion on a band-local mask (3×3 structuring element).
    private func erodeBandMask(
        mask: inout [UInt8],
        width: Int,
        height: Int,
        iterations: Int
    ) {
        guard width > 0, height > 0, mask.count >= width * height, iterations > 0 else { return }

        let w = width
        let h = height
        var scratch = mask

        for _ in 0..<iterations {
            scratch = mask
            for y in 0..<h {
                for x in 0..<w {
                    let idx = y * w + x
                    if scratch[idx] == 0 {
                        mask[idx] = 0
                        continue
                    }
                    var allOn = true
                    for ny in max(0, y - 1)...min(h - 1, y + 1) {
                        for nx in max(0, x - 1)...min(w - 1, x + 1) {
                            if scratch[ny * w + nx] == 0 {
                                allOn = false
                                break
                            }
                        }
                        if !allOn { break }
                    }
                    mask[idx] = allOn ? 255 : 0
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
                // Normalize alpha to 0...1 before premultiplying RGB. The output
                // image itself still stores alpha as 0...255 below.
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

    /// Bytes per row actually allocated by Core Graphics (often **>** `width * 4`); using the tight
    /// stride for row pointers causes **horizontal shearing / zebra banding** in the final `CGImage`.
    private func cgBitmapAllocatedBytesPerRow(_ context: CGContext) -> Int {
        context.bytesPerRow
    }

    /// Clears the entire RGBA output, including Core Graphics row padding, so pixels outside the
    /// segmentation mask stay fully transparent and the live preview shows through underneath.
    private func fillCompositeBufferTransparent(
        outBase: UnsafeMutablePointer<UInt8>,
        height: Int,
        bytesPerRowOut: Int
    ) {
        guard height > 0, bytesPerRowOut > 0 else { return }
        for y in 0..<height {
            memset(outBase.advanced(by: y * bytesPerRowOut), 0, bytesPerRowOut)
        }
    }

    /// Reference CPU compositor. Expects `outBase` to already be transparent outside the band.
    private func compositeCpuCameraBandReference(
        outBase: UnsafeMutablePointer<UInt8>,
        origBase: UnsafePointer<UInt8>,
        maskBase: UnsafePointer<UInt8>,
        camRowBytes: Int,
        maskRowBytes: Int,
        bytesPerRowOut: Int,
        x0: Int,
        x1: Int,
        y0: Int,
        y1: Int
    ) {
        guard x1 > x0, y1 > y0 else { return }

        for y in y0..<y1 {
            let outRowPtr = outBase.advanced(by: y * bytesPerRowOut)
            let origRowPtr = origBase.advanced(by: y * camRowBytes)
            let maskRowPtr = maskBase.advanced(by: y * maskRowBytes)

            for x in x0..<x1 {
                let maskValue = maskRowPtr[x]
                let outputOffset = x * 4
                let cameraOffset = x * 4

                guard maskValue != 0 else { continue }

                outRowPtr[outputOffset + 0] = origRowPtr[cameraOffset + 2]
                outRowPtr[outputOffset + 1] = origRowPtr[cameraOffset + 1]
                outRowPtr[outputOffset + 2] = origRowPtr[cameraOffset + 0]
                outRowPtr[outputOffset + 3] = maskValue
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
        // Disabled: premultiplied RGBA sample logging
        _ = (outBase, bytesPerRowOut, width, height, x0, x1, y0, y1, tag)
    }

    /// Logs a compact sampled binary mask silhouette for the current object band.
    private func logSampledMaskShapeIfDebug(
        mask: [UInt8],
        width: Int,
        height: Int,
        x0: Int,
        x1: Int,
        y0: Int,
        y1: Int,
        tag: String,
        gridCols: Int = 48,
        gridRows: Int = 24
    ) {
        guard debugMode else { return }
        // Disabled: MASK SHAPE ASCII grid
        _ = (mask, width, height, x0, x1, y0, y1, tag, gridCols, gridRows)
    }

    /// ONNX-style cutout: hard binary mask compositing.
    /// Each output pixel samples the proto mask with nearest-neighbor lookup and is
    /// either fully opaque or fully transparent. This avoids semi-transparent
    /// upscaled edges that can leak floor or wall through the segmented furniture.
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

        guard let ctx = CGContext(
            data: nil,
            width: origW,
            height: origH,
            bitsPerComponent: 8,
            bytesPerRow: 0,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { return nil }
        let bytesPerRowOut = cgBitmapAllocatedBytesPerRow(ctx)

        CVPixelBufferLockBaseAddress(processBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(processBuffer, .readOnly) }
        guard let origBase = CVPixelBufferGetBaseAddress(processBuffer)?.assumingMemoryBound(to: UInt8.self) else { return nil }
        let camRowBytes = CVPixelBufferGetBytesPerRow(processBuffer)
        guard let outData = ctx.data else { return nil }
        let outBase = outData.assumingMemoryBound(to: UInt8.self)

        fillCompositeBufferTransparent(
            outBase: outBase,
            height: origH,
            bytesPerRowOut: bytesPerRowOut
        )
        let protoScaleX = Float(pw) / Float(origW)
        let protoScaleY = Float(ph) / Float(origH)

        for y in yStart..<yEnd {
            let outRow = outBase.advanced(by: y * bytesPerRowOut)
            let camRow = origBase.advanced(by: y * camRowBytes)
            let protoY = min(ph - 1, max(0, Int(Float(y) * protoScaleY)))
            let protoRowOff = protoY * pw
            for x in xStart..<xEnd {
                let protoX = min(pw - 1, max(0, Int(Float(x) * protoScaleX)))
                guard maskProto[protoRowOff + protoX] != 0 else { continue }
                let cx = x * 4
                outRow[cx] = camRow[cx + 2]
                outRow[cx + 1] = camRow[cx + 1]
                outRow[cx + 2] = camRow[cx]
                outRow[cx + 3] = 255
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

    /// ONNX-style cutout driven by bilinearly upsampled YOLOE logits.
    /// We upsample the 160×160 proto logits (align_corners=false semantics) to the
    /// full camera resolution, apply a positive threshold to shrink the mask away
    /// from thin gaps, and write fully opaque pixels wherever the upsampled logit
    /// exceeds that threshold.
    private func compositeCpuBilinearProtoMaskCutoutFromLogitsLegacy(
        processBuffer: CVPixelBuffer,
        maskProto: [UInt8],
        maskLogits: [Float],
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
        guard pw > 0,
              ph > 0,
              maskProto.count >= pw * ph,
              maskLogits.count >= pw * ph,
              origW > 0, origH > 0 else { return nil }

        let xStart = max(0, min(origW, x0))
        let xEnd = max(0, min(origW, x1))
        let yStart = max(0, min(origH, y0))
        let yEnd = max(0, min(origH, y1))
        guard xStart < xEnd, yStart < yEnd else { return nil }

        guard let ctx = CGContext(
            data: nil,
            width: origW,
            height: origH,
            bitsPerComponent: 8,
            bytesPerRow: 0,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { return nil }
        let bytesPerRowOut = cgBitmapAllocatedBytesPerRow(ctx)
        CVPixelBufferLockBaseAddress(processBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(processBuffer, .readOnly) }
        guard let origBase = CVPixelBufferGetBaseAddress(processBuffer)?.assumingMemoryBound(to: UInt8.self) else { return nil }
        let camRowBytes = CVPixelBufferGetBytesPerRow(processBuffer)
        guard let outData = ctx.data else { return nil }
        let outBase = outData.assumingMemoryBound(to: UInt8.self)

        fillCompositeBufferTransparent(
            outBase: outBase,
            height: origH,
            bytesPerRowOut: bytesPerRowOut
        )

        let protoScaleX = Float(pw) / Float(origW)
        let protoScaleY = Float(ph) / Float(origH)
        let maxPx = pw - 1
        let maxPy = ph - 1

        // Ultralytics-style mask native: upsample full-res logits, then apply
        // a fixed bias before a zero threshold, i.e. (logit - 1.2) > 0.
        // Temporarily relax bias to 0.0 to test for left-handle erosion.
        let logitBias: Float = 0.0

        for y in yStart..<yEnd {
            let outRow = outBase.advanced(by: y * bytesPerRowOut)
            let camRow = origBase.advanced(by: y * camRowBytes)

            // align_corners = false mapping for Y
            let fy = (Float(y) + 0.5) * protoScaleY - 0.5
            let py0 = max(0, min(maxPy, Int(floor(fy))))
            let py1 = max(0, min(maxPy, py0 + 1))
            let ty = fy - Float(py0)

            for x in xStart..<xEnd {
                // align_corners = false mapping for X
                let fx = (Float(x) + 0.5) * protoScaleX - 0.5
                let px0 = max(0, min(maxPx, Int(floor(fx))))
                let px1 = max(0, min(maxPx, px0 + 1))
                let tx = fx - Float(px0)

                let idx00 = py0 * pw + px0
                let idx10 = py0 * pw + px1
                let idx01 = py1 * pw + px0
                let idx11 = py1 * pw + px1

                let v00 = maskLogits[idx00]
                let v10 = maskLogits[idx10]
                let v01 = maskLogits[idx01]
                let v11 = maskLogits[idx11]

                let rawLogit =
                    v00 * (1 - tx) * (1 - ty) +
                    v10 * tx * (1 - ty) +
                    v01 * (1 - tx) * ty +
                    v11 * tx * ty

                // Apply negative bias as suggested, then require the biased
                // logit to be strictly positive. This forces all logits in
                // (0, logitBias] to become transparent, which helps clear the
                // "grey wall" inside thin gaps like chair handles.
                let biasedLogit = rawLogit - logitBias
                guard biasedLogit > 0 else { continue }

                let cx = x * 4
                outRow[cx] = camRow[cx + 2]
                outRow[cx + 1] = camRow[cx + 1]
                outRow[cx + 2] = camRow[cx]
                outRow[cx + 3] = 255
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

    /// ONNX / Ultralytics `process_mask` order on the composite path: fused proto
    /// logits → bilinear upsample to full camera size (`align_corners=False`) →
    /// write only the image-space crop band → threshold last (`logit > nativeCompositeUpsampleLogitThreshold()`).
    private func compositeCpuBilinearProtoMaskCutoutFromLogits(
        processBuffer: CVPixelBuffer,
        maskProto: [UInt8],
        maskLogits: [Float],
        planes: [Float],
        protoW: Int,
        protoH: Int,
        modelSide: Int,
        origW: Int,
        origH: Int,
        x0: Int,
        x1: Int,
        y0: Int,
        y1: Int,
        primary: FurnitureFitDetection,
        primaryBx1: Int,
        primaryBy1: Int,
        primaryBx2: Int,
        primaryBy2: Int,
        maskDetectionsForBuild: [FurnitureFitDetection],
        debugTag: String
    ) -> CGImage? {
        let pw = protoW
        let ph = protoH
        guard pw > 0,
              ph > 0,
              maskProto.count >= pw * ph,
              maskLogits.count >= pw * ph,
              origW > 0, origH > 0 else { return nil }

        let xStart = max(0, min(origW, x0))
        let xEnd = max(0, min(origW, x1))
        let yStart = max(0, min(origH, y0))
        let yEnd = max(0, min(origH, y1))
        guard xStart < xEnd, yStart < yEnd else { return nil }

        let bandW = xEnd - xStart
        let bandH = yEnd - yStart
        guard bandW > 0, bandH > 0 else { return nil }

        guard let ctx = CGContext(
            data: nil,
            width: origW,
            height: origH,
            bitsPerComponent: 8,
            bytesPerRow: 0,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { return nil }
        let bytesPerRowOut = cgBitmapAllocatedBytesPerRow(ctx)

        CVPixelBufferLockBaseAddress(processBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(processBuffer, .readOnly) }
        guard let origBase = CVPixelBufferGetBaseAddress(processBuffer)?.assumingMemoryBound(to: UInt8.self) else { return nil }
        let camRowBytes = CVPixelBufferGetBytesPerRow(processBuffer)
        guard let outData = ctx.data else { return nil }
        let outBase = outData.assumingMemoryBound(to: UInt8.self)

        fillCompositeBufferTransparent(
            outBase: outBase,
            height: origH,
            bytesPerRowOut: bytesPerRowOut
        )

        let protoScaleX = Float(pw) / Float(origW)
        let protoScaleY = Float(ph) / Float(origH)
        let maxPx = pw - 1
        let maxPy = ph - 1
        let fullCount = origW * origH
        if upscaledPlanarMaskScratch.count < fullCount {
            upscaledPlanarMaskScratch = [UInt8](repeating: 0, count: fullCount)
        } else {
            upscaledPlanarMaskScratch.withUnsafeMutableBytes { raw in
                if let base = raw.baseAddress { memset(base, 0, fullCount) }
            }
        }

        // Ultralytics `process_mask`: upsample float logits (bilinear, align_corners=False),
        // then crop in image space (here: only fill the composite band), then threshold.
        let upsampleThreshold = FurnitureFitOnnxStylePipeline.nativeCompositeUpsampleLogitThreshold()
        for by in 0..<bandH {
            let y = yStart + by
            let fy = (Float(y) + 0.5) * protoScaleY - 0.5
            let py0 = max(0, min(maxPy, Int(floor(fy))))
            let py1 = max(0, min(maxPy, py0 + 1))
            let ty = fy - Float(py0)

            for bx in 0..<bandW {
                let x = xStart + bx
                let fx = (Float(x) + 0.5) * protoScaleX - 0.5
                let px0 = max(0, min(maxPx, Int(floor(fx))))
                let px1 = max(0, min(maxPx, px0 + 1))
                let tx = fx - Float(px0)

                let idx00 = py0 * pw + px0
                let idx10 = py0 * pw + px1
                let idx01 = py1 * pw + px0
                let idx11 = py1 * pw + px1

                let v00 = maskLogits[idx00]
                let v10 = maskLogits[idx10]
                let v01 = maskLogits[idx01]
                let v11 = maskLogits[idx11]

                let rawLogit =
                    v00 * (1 - tx) * (1 - ty) +
                    v10 * tx * (1 - ty) +
                    v01 * (1 - tx) * ty +
                    v11 * tx * ty

                let maskValue: UInt8 = rawLogit > upsampleThreshold ? 255 : 0
                upscaledPlanarMaskScratch[y * origW + x] = maskValue
            }
        }

        if furnitureFitNativeMaskMorphologicalClose, bandW >= 3, bandH >= 3 {
            var bandCompact = [UInt8](repeating: 0, count: bandW * bandH)
            for by in 0..<bandH {
                let y = yStart + by
                for bx in 0..<bandW {
                    let x = xStart + bx
                    bandCompact[by * bandW + bx] = upscaledPlanarMaskScratch[y * origW + x]
                }
            }
            let closed = morphologicalBinaryClose3x3Planar8(mask: bandCompact, width: bandW, height: bandH)
            for by in 0..<bandH {
                let y = yStart + by
                for bx in 0..<bandW {
                    let x = xStart + bx
                    upscaledPlanarMaskScratch[y * origW + x] = closed[by * bandW + bx]
                }
            }
        }

        logSampledMaskShapeIfDebug(
            mask: upscaledPlanarMaskScratch,
            width: origW,
            height: origH,
            x0: xStart,
            x1: xEnd,
            y0: yStart,
            y1: yEnd,
            tag: debugTag
        )

        upscaledPlanarMaskScratch.withUnsafeBufferPointer { maskPtr in
            guard let maskBase = maskPtr.baseAddress else { return }
            compositeCpuCameraBandReference(
                outBase: outBase,
                origBase: origBase,
                maskBase: maskBase,
                camRowBytes: camRowBytes,
                maskRowBytes: origW,
                bytesPerRowOut: bytesPerRowOut,
                x0: xStart,
                x1: xEnd,
                y0: yStart,
                y1: yEnd
            )
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
        logPrimaryChairStageComparisonIfDebug(
            primary: primary,
            modelSide: modelSide,
            protoMask: maskProto,
            protoW: pw,
            protoH: ph,
            fullResMask: upscaledPlanarMaskScratch,
            outBase: outBase,
            bytesPerRowOut: bytesPerRowOut,
            origW: origW,
            origH: origH,
            primaryBx1: primaryBx1,
            primaryBy1: primaryBy1,
            primaryBx2: primaryBx2,
            primaryBy2: primaryBy2
        )
        logPrimaryChairContributorPixelsIfDebug(
            primary: primary,
            maskDetectionsForBuild: maskDetectionsForBuild,
            planes: planes,
            fusedMaskLogits: maskLogits,
            protoW: pw,
            protoH: ph,
            modelSide: modelSide,
            origW: origW,
            origH: origH,
            primaryBx1: primaryBx1,
            primaryBy1: primaryBy1,
            primaryBx2: primaryBx2,
            primaryBy2: primaryBy2
        )
        return ctx.makeImage()
    }

    /// Legacy 1280 path transplanted from the old `SmartyPants` implementation:
    /// build a full-field union mask, undo letterbox, then composite once.
    private func compositeLegacy1280UnionMaskCutout(
        processBuffer: CVPixelBuffer,
        planes: [Float],
        protoW: Int,
        protoH: Int,
        modelSide: Int,
        origW: Int,
        origH: Int,
        maskDetectionsForBuild: [FurnitureFitDetection]
    ) -> CGImage? {
        guard currentYoloUsesLetterbox,
              protoW > 0,
              protoH > 0,
              modelSide > 0,
              planes.count >= 32 * protoW * protoH else { return nil }

        let planeSize = protoW * protoH
        let validDetections = maskDetectionsForBuild.filter { $0.coeffs.count >= 32 }
        guard !validDetections.isEmpty else { return nil }

        var prototypeMatrix = [Float](repeating: 0, count: planeSize * 32)
        var zero: Float = 0
        prototypeMatrix.withUnsafeMutableBufferPointer { dstPtr in
            planes.withUnsafeBufferPointer { srcPtr in
                guard let dstBase = dstPtr.baseAddress, let srcBase = srcPtr.baseAddress else { return }
                for channelIndex in 0..<32 {
                    let srcStart = srcBase.advanced(by: channelIndex * planeSize)
                    let dstStart = dstBase.advanced(by: channelIndex)
                    vDSP_vsadd(srcStart, 1, &zero, dstStart, 32, vDSP_Length(planeSize))
                }
            }
        }

        var maximumLogits = [Float](repeating: -Float.greatestFiniteMagnitude, count: planeSize)
        let batchSize = 64
        let matrixHeight = Int32(planeSize)
        let matrixDepth = Int32(32)
        let alpha: Float = 1
        let beta: Float = 0
        var batchStart = 0

        while batchStart < validDetections.count {
            let batchEnd = min(validDetections.count, batchStart + batchSize)
            let batchCount = batchEnd - batchStart
            let matrixWidth = Int32(batchCount)
            var coefficientMatrix = [Float](repeating: 0, count: 32 * batchCount)

            for detectionOffset in 0..<batchCount {
                let coeffs = validDetections[batchStart + detectionOffset].coeffs
                for channelIndex in 0..<32 {
                    coefficientMatrix[channelIndex * batchCount + detectionOffset] = coeffs[channelIndex]
                }
            }

            var logitsBatch = [Float](repeating: 0, count: planeSize * batchCount)
            prototypeMatrix.withUnsafeBufferPointer { protoPtr in
                coefficientMatrix.withUnsafeBufferPointer { coeffPtr in
                    logitsBatch.withUnsafeMutableBufferPointer { batchPtr in
                        guard let protoBase = protoPtr.baseAddress,
                              let coeffBase = coeffPtr.baseAddress,
                              let batchBase = batchPtr.baseAddress else { return }
                        cblas_sgemm(
                            CblasRowMajor,
                            CblasNoTrans,
                            CblasNoTrans,
                            matrixHeight,
                            matrixWidth,
                            matrixDepth,
                            alpha,
                            protoBase,
                            matrixDepth,
                            coeffBase,
                            matrixWidth,
                            beta,
                            batchBase,
                            matrixWidth
                        )
                    }
                }
            }

            logitsBatch.withUnsafeBufferPointer { batchPtr in
                maximumLogits.withUnsafeMutableBufferPointer { maxPtr in
                    guard let batchBase = batchPtr.baseAddress, let maxBase = maxPtr.baseAddress else { return }
                    for pixelIndex in 0..<planeSize {
                        var rowMaximum: Float = 0
                        vDSP_maxv(
                            batchBase.advanced(by: pixelIndex * batchCount),
                            1,
                            &rowMaximum,
                            vDSP_Length(batchCount)
                        )
                        if rowMaximum > maxBase[pixelIndex] {
                            maxBase[pixelIndex] = rowMaximum
                        }
                    }
                }
            }

            batchStart = batchEnd
        }

        var protoBinaryMask = [UInt8](repeating: 0, count: planeSize)
        for pixelIndex in 0..<planeSize {
            protoBinaryMask[pixelIndex] = maximumLogits[pixelIndex] > 0 ? 255 : 0
        }

        let fullMask = makeLegacy1280FullMaskFromProtoWithLetterboxFix(
            maskSmall: protoBinaryMask,
            protoW: protoW,
            protoH: protoH,
            modelInput: modelSide,
            origW: origW,
            origH: origH
        )

        guard let ctx = CGContext(
            data: nil,
            width: origW,
            height: origH,
            bitsPerComponent: 8,
            bytesPerRow: 0,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { return nil }
        let outBytesPerRow = cgBitmapAllocatedBytesPerRow(ctx)

        CVPixelBufferLockBaseAddress(processBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(processBuffer, .readOnly) }
        guard let origBase = CVPixelBufferGetBaseAddress(processBuffer)?.assumingMemoryBound(to: UInt8.self),
              let outBase = ctx.data?.assumingMemoryBound(to: UInt8.self) else { return nil }
        let origBytesPerRow = CVPixelBufferGetBytesPerRow(processBuffer)

        fillCompositeBufferTransparent(
            outBase: outBase,
            height: origH,
            bytesPerRowOut: outBytesPerRow
        )

        for y in 0..<origH {
            let origRow = y * origBytesPerRow
            let outRow = y * outBytesPerRow
            let maskRow = y * origW
            for x in 0..<origW {
                guard fullMask[maskRow + x] != 0 else { continue }
                let outOffset = outRow + x * 4
                let origOffset = origRow + x * 4
                outBase[outOffset + 0] = origBase[origOffset + 2]
                outBase[outOffset + 1] = origBase[origOffset + 1]
                outBase[outOffset + 2] = origBase[origOffset + 0]
                outBase[outOffset + 3] = 255
            }
        }

        return ctx.makeImage()
    }

    private func makeLegacy1280FullMaskFromProtoWithLetterboxFix(
        maskSmall: [UInt8],
        protoW: Int,
        protoH: Int,
        modelInput: Int,
        origW: Int,
        origH: Int
    ) -> [UInt8] {
        guard protoW > 0, protoH > 0, modelInput > 0, origW > 0, origH > 0 else { return [] }

        var maskModel = [UInt8](repeating: 0, count: modelInput * modelInput)
        maskModel.withUnsafeMutableBufferPointer { dstPtr in
            maskSmall.withUnsafeBufferPointer { srcPtr in
                guard let dstBase = dstPtr.baseAddress, let srcBase = srcPtr.baseAddress else { return }
                var source = vImage_Buffer(
                    data: UnsafeMutableRawPointer(mutating: srcBase),
                    height: vImagePixelCount(protoH),
                    width: vImagePixelCount(protoW),
                    rowBytes: protoW
                )
                var destination = vImage_Buffer(
                    data: dstBase,
                    height: vImagePixelCount(modelInput),
                    width: vImagePixelCount(modelInput),
                    rowBytes: modelInput
                )
                _ = vImageScale_Planar8(&source, &destination, nil, vImage_Flags(kvImageHighQualityResampling))
            }
        }

        let gain = min(Float(modelInput) / Float(origW), Float(modelInput) / Float(origH))
        let contentWidth = Int(round(Float(origW) * gain))
        let contentHeight = Int(round(Float(origH) * gain))
        let padX = Int(round((Float(modelInput) - Float(contentWidth)) * 0.5))
        let padY = Int(round((Float(modelInput) - Float(contentHeight)) * 0.5))
        let cropX = max(0, min(modelInput - 1, padX))
        let cropY = max(0, min(modelInput - 1, padY))
        let cropWidth = max(1, min(modelInput - cropX, contentWidth))
        let cropHeight = max(1, min(modelInput - cropY, contentHeight))

        var croppedMask = [UInt8](repeating: 0, count: cropWidth * cropHeight)
        for y in 0..<cropHeight {
            let sourceRow = (cropY + y) * modelInput + cropX
            let destinationRow = y * cropWidth
            for x in 0..<cropWidth {
                croppedMask[destinationRow + x] = maskModel[sourceRow + x]
            }
        }

        var fullMask = [UInt8](repeating: 0, count: origW * origH)
        fullMask.withUnsafeMutableBufferPointer { dstPtr in
            croppedMask.withUnsafeBufferPointer { srcPtr in
                guard let dstBase = dstPtr.baseAddress, let srcBase = srcPtr.baseAddress else { return }
                var source = vImage_Buffer(
                    data: UnsafeMutableRawPointer(mutating: srcBase),
                    height: vImagePixelCount(cropHeight),
                    width: vImagePixelCount(cropWidth),
                    rowBytes: cropWidth
                )
                var destination = vImage_Buffer(
                    data: dstBase,
                    height: vImagePixelCount(origH),
                    width: vImagePixelCount(origW),
                    rowBytes: origW
                )
                _ = vImageScale_Planar8(&source, &destination, nil, vImage_Flags(kvImageHighQualityResampling))
            }
        }
        return fullMask
    }

    // MARK: - process_mask_native (GPU bilinear upsample + threshold)

    /// Matches Metal `BilinearUpsampleParams` layout (8×UInt32 + Float).
    private struct BilinearUpsampleParams {
        var protoW:  UInt32
        var protoH:  UInt32
        var origW:   UInt32
        var origH:   UInt32
        var xStart:  UInt32
        var yStart:  UInt32
        var bandW:   UInt32
        var bandH:   UInt32
        var logitThreshold: Float
    }

    /// GPU-accelerated bilinear upsample of full-field logits + threshold >0,
    /// returning a UInt8 band mask.  Falls back to CPU if Metal is unavailable.
    private func gpuBilinearUpsampleAndThreshold(
        logits: [Float],
        protoW: Int,
        protoH: Int,
        origW: Int,
        origH: Int,
        xStart: Int,
        yStart: Int,
        bandW: Int,
        bandH: Int,
        logitThreshold: Float
    ) -> [UInt8]? {
        guard let device = metalDevice,
              let queue = metalCommandQueue,
              let pipeline = bilinearUpsamplePipeline else { return nil }

        let logitByteCount = logits.count * MemoryLayout<Float>.stride
        let maskByteCount  = bandW * bandH

        if cachedLogitBuf == nil || cachedLogitCapacity < logits.count {
            cachedLogitBuf = device.makeBuffer(length: logitByteCount, options: .storageModeShared)
            cachedLogitCapacity = logits.count
        }
        if cachedBandMaskBuf == nil || cachedBandMaskCapacity < maskByteCount {
            cachedBandMaskBuf = device.makeBuffer(length: maskByteCount, options: .storageModeShared)
            cachedBandMaskCapacity = maskByteCount
        }
        guard let logitBuf = cachedLogitBuf,
              let maskBuf  = cachedBandMaskBuf else { return nil }

        logits.withUnsafeBufferPointer { ptr in
            memcpy(logitBuf.contents(), ptr.baseAddress!, logitByteCount)
        }
        memset(maskBuf.contents(), 0, maskByteCount)

        guard let cmdBuf  = queue.makeCommandBuffer(),
              let encoder = cmdBuf.makeComputeCommandEncoder() else { return nil }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(logitBuf, offset: 0, index: 0)
        encoder.setBuffer(maskBuf,  offset: 0, index: 1)

        var params = BilinearUpsampleParams(
            protoW: UInt32(protoW),
            protoH: UInt32(protoH),
            origW:  UInt32(origW),
            origH:  UInt32(origH),
            xStart: UInt32(xStart),
            yStart: UInt32(yStart),
            bandW:  UInt32(bandW),
            bandH:  UInt32(bandH),
            logitThreshold: logitThreshold
        )
        encoder.setBytes(&params, length: MemoryLayout<BilinearUpsampleParams>.stride, index: 2)

        let threadGroupW = min(16, bandW)
        let threadGroupH = min(16, bandH)
        let threadGroupSize = MTLSize(width: threadGroupW, height: threadGroupH, depth: 1)
        let gridSize = MTLSize(width: bandW, height: bandH, depth: 1)
        if device.supportsFamily(.apple4) {
            encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupSize)
        } else {
            let tgCountW = (bandW + threadGroupW - 1) / threadGroupW
            let tgCountH = (bandH + threadGroupH - 1) / threadGroupH
            encoder.dispatchThreadgroups(
                MTLSize(width: tgCountW, height: tgCountH, depth: 1),
                threadsPerThreadgroup: threadGroupSize
            )
        }

        encoder.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        var result = [UInt8](repeating: 0, count: maskByteCount)
        memcpy(&result, maskBuf.contents(), maskByteCount)
        return result
    }

    /// `process_mask_native` composite: full-field matmul → GPU bilinear upsample →
    /// crop + threshold → optional morph close → CPU camera composite.
    private func compositeGpuNativeMaskCutout(
        processBuffer: CVPixelBuffer,
        maskProto: [UInt8],
        maskLogits: [Float],
        protoW: Int,
        protoH: Int,
        origW: Int,
        origH: Int,
        x0: Int,
        x1: Int,
        y0: Int,
        y1: Int,
        primary: FurnitureFitDetection,
        primaryBx1: Int,
        primaryBy1: Int,
        primaryBx2: Int,
        primaryBy2: Int,
        debugTag: String
    ) -> CGImage? {
        let pw = protoW
        let ph = protoH
        guard pw > 0, ph > 0,
              maskLogits.count >= pw * ph,
              origW > 0, origH > 0 else { return nil }

        let xStart = max(0, min(origW, x0))
        let xEnd   = max(0, min(origW, x1))
        let yStart = max(0, min(origH, y0))
        let yEnd   = max(0, min(origH, y1))
        guard xStart < xEnd, yStart < yEnd else { return nil }

        let bandW = xEnd - xStart
        let bandH = yEnd - yStart
        guard bandW > 0, bandH > 0 else { return nil }

        let upsampleThreshold = FurnitureFitOnnxStylePipeline.nativeCompositeUpsampleLogitThreshold()

        guard var bandMask = gpuBilinearUpsampleAndThreshold(
            logits: maskLogits,
            protoW: pw,
            protoH: ph,
            origW: origW,
            origH: origH,
            xStart: xStart,
            yStart: yStart,
            bandW: bandW,
            bandH: bandH,
            logitThreshold: upsampleThreshold
        ) else {
            if debugMode { logDebug("⚠️ GPU bilinear upsample failed, no fallback") }
            return nil
        }

        if furnitureFitNativeMaskMorphologicalClose, bandW >= 3, bandH >= 3 {
            bandMask = morphologicalBinaryClose3x3Planar8(mask: bandMask, width: bandW, height: bandH)
        }

        guard let ctx = CGContext(
            data: nil,
            width: origW,
            height: origH,
            bitsPerComponent: 8,
            bytesPerRow: 0,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { return nil }
        let bytesPerRowOut = cgBitmapAllocatedBytesPerRow(ctx)

        CVPixelBufferLockBaseAddress(processBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(processBuffer, .readOnly) }
        guard let origBase = CVPixelBufferGetBaseAddress(processBuffer)?
                .assumingMemoryBound(to: UInt8.self) else { return nil }
        let camRowBytes = CVPixelBufferGetBytesPerRow(processBuffer)
        guard let outData = ctx.data else { return nil }
        let outBase = outData.assumingMemoryBound(to: UInt8.self)

        fillCompositeBufferTransparent(
            outBase: outBase,
            height: origH,
            bytesPerRowOut: bytesPerRowOut
        )

        for by in 0..<bandH {
            let y = yStart + by
            let outRowPtr  = outBase.advanced(by: y * bytesPerRowOut)
            let origRowPtr = origBase.advanced(by: y * camRowBytes)
            for bx in 0..<bandW {
                let maskVal = bandMask[by * bandW + bx]
                guard maskVal != 0 else { continue }
                let x = xStart + bx
                let outOff = x * 4
                let camOff = x * 4
                outRowPtr[outOff + 0] = origRowPtr[camOff + 2]  // B → R
                outRowPtr[outOff + 1] = origRowPtr[camOff + 1]  // G → G
                outRowPtr[outOff + 2] = origRowPtr[camOff + 0]  // R → B
                outRowPtr[outOff + 3] = 255
            }
        }

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
        guard let ctx = CGContext(
            data: nil,
            width: origW,
            height: origH,
            bitsPerComponent: 8,
            bytesPerRow: 0,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { return nil }

        ctx.draw(composed, in: CGRect(x: 0, y: 0, width: origW, height: origH))

        func bufferRect(for d: FurnitureFitDetection) -> (bx1: Int, by1: Int, bw: Int, bh: Int) {
            let scaledRect = self.bufferRect(
                for: d,
                imageWidth: origW,
                imageHeight: origH,
                scaleX: scaleX,
                scaleY: scaleY
            )
            let bx1 = Int(scaledRect.minX)
            let by1 = Int(scaledRect.minY)
            let bx2 = Int(scaledRect.maxX)
            let by2 = Int(scaledRect.maxY)
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
        guard let ctx = CGContext(
            data: nil,
            width: origW,
            height: origH,
            bitsPerComponent: 8,
            bytesPerRow: 0,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { return nil }
        let bytesPerRowOut = cgBitmapAllocatedBytesPerRow(ctx)
        CVPixelBufferLockBaseAddress(processBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(processBuffer, .readOnly) }
        guard let origBase = CVPixelBufferGetBaseAddress(processBuffer)?.assumingMemoryBound(to: UInt8.self) else { return nil }
        let bytesPerRowCam = CVPixelBufferGetBytesPerRow(processBuffer)
        guard let outBase = ctx.data?.assumingMemoryBound(to: UInt8.self) else { return nil }
        let x0 = max(0, primaryBx1)
        let x1 = min(origW, primaryBx2)
        let y0 = max(0, primaryBy1)
        let y1 = min(origH, primaryBy2)
        fillCompositeBufferTransparent(
            outBase: outBase,
            height: origH,
            bytesPerRowOut: bytesPerRowOut
        )
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

    /// Ultralytics-style letterbox into a square model input with 114 padding.
    private func resizeLetterboxToSquare(_ src: CVPixelBuffer, size: Int) -> CVPixelBuffer? {
        CVPixelBufferLockBaseAddress(src, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(src, .readOnly) }

        let srcW = CVPixelBufferGetWidth(src)
        let srcH = CVPixelBufferGetHeight(src)
        guard srcW > 0, srcH > 0 else { return nil }

        if cachedLetterboxSize != size || cachedLetterboxBuffer == nil {
            var newBuffer: CVPixelBuffer?
            guard CVPixelBufferCreate(nil, size, size, kCVPixelFormatType_32BGRA, nil, &newBuffer) == kCVReturnSuccess,
                  let buf = newBuffer else { return nil }
            cachedLetterboxBuffer = buf
            cachedLetterboxSize = size
        }

        guard let dst = cachedLetterboxBuffer else { return nil }
        CVPixelBufferLockBaseAddress(dst, [])
        defer { CVPixelBufferUnlockBaseAddress(dst, []) }

        guard let srcBase = CVPixelBufferGetBaseAddress(src),
              let dstBase = CVPixelBufferGetBaseAddress(dst) else { return nil }

        let dstRowBytes = CVPixelBufferGetBytesPerRow(dst)
        YoloUltralyticsLetterboxFill.fillOpaqueBGRA114(
            dstBase: dstBase,
            totalByteCount: dstRowBytes * size
        )

        let scale = min(Float(size) / Float(srcW), Float(size) / Float(srcH))
        let scaledWidth = max(1, min(size, Int(round(Float(srcW) * scale))))
        let scaledHeight = max(1, min(size, Int(round(Float(srcH) * scale))))
        let padX = (size - scaledWidth) / 2
        let padY = (size - scaledHeight) / 2

        var srcBuffer = vImage_Buffer(
            data: srcBase,
            height: vImagePixelCount(srcH),
            width: vImagePixelCount(srcW),
            rowBytes: CVPixelBufferGetBytesPerRow(src)
        )
        var dstRegion = vImage_Buffer(
            data: dstBase.advanced(by: padY * dstRowBytes + padX * 4),
            height: vImagePixelCount(scaledHeight),
            width: vImagePixelCount(scaledWidth),
            rowBytes: dstRowBytes
        )
        guard vImageScale_ARGB8888(&srcBuffer, &dstRegion, nil, vImage_Flags(kvImageHighQualityResampling)) == kvImageNoError else {
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
    private func parsePrototypes(_ proto: MLMultiArray) -> (planes: [Float], interleavedPlanes: [Float], count: Int, height: Int, width: Int, shape: [Int], strides: [Int], channelAxis: Int)? {
        var normalizedStrides = proto.strides.map { $0.intValue }
        var shape = proto.shape.map { $0.intValue }
        if shape.count == 4 && shape[0] == 1 {
            shape.removeFirst()
            if normalizedStrides.count == 4 {
                normalizedStrides.removeFirst()
            }
        }
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
        if protoPlanesInterleaved.count != count * planeSize {
            protoPlanesInterleaved = [Float](repeating: 0, count: count * planeSize)
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

        // GPU fused kernels prefer [pixel][32] so one pixel's prototype vector is contiguous.
        protoPlanes.withUnsafeBufferPointer { src in
            protoPlanesInterleaved.withUnsafeMutableBufferPointer { dst in
                guard let srcBase = src.baseAddress, let dstBase = dst.baseAddress else { return }
                for protoIndex in 0..<planeSize {
                    let dstRowBase = protoIndex * count
                    for channel in 0..<count {
                        dstBase[dstRowBase + channel] = srcBase[channel * planeSize + protoIndex]
                    }
                }
            }
        }

        return (protoPlanes, protoPlanesInterleaved, count, h, w, shape, normalizedStrides, cIdx)
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

    /// Fill interior holes using horizontal ∩ vertical run-length fill.
    /// A background pixel is filled when there is foreground both left and right
    /// on the same row **and** both above and below in the same column. Bridges
    /// structural gaps (armrest-to-backrest, slats, etc.) at proto resolution
    /// without expanding the outer silhouette.
    private func fillInteriorHolesHVIntersection(mask: inout [UInt8], width: Int, height: Int) {
        let count = width * height
        guard count == mask.count, width > 1, height > 1 else { return }

        var hFill = [Bool](repeating: false, count: count)
        var vFill = [Bool](repeating: false, count: count)

        for y in 0..<height {
            let row = y * width
            var left = -1
            var right = -1
            for x in 0..<width {
                if mask[row + x] > 0 {
                    if left < 0 { left = x }
                    right = x
                }
            }
            if left >= 0 {
                for x in left...right { hFill[row + x] = true }
            }
        }

        for x in 0..<width {
            var top = -1
            var bottom = -1
            for y in 0..<height {
                if mask[y * width + x] > 0 {
                    if top < 0 { top = y }
                    bottom = y
                }
            }
            if top >= 0 {
                for y in top...bottom { vFill[y * width + x] = true }
            }
        }

        for i in 0..<count where mask[i] == 0 {
            if hFill[i] && vFill[i] { mask[i] = 255 }
        }
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
                  bytesPerRow: 0,
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
        guard startupProgressActive, !hasFirstDetection, !suppressStartupProgress else { return }
        DispatchQueue.main.async {
            self.progressContainer.isHidden = false
            self.progressView.progress = value
            self.progressLabel.text = "  \(text)  "
        }
    }

    private func finishStartupProgressIfNeeded() {
        guard startupProgressActive || !segmentationCompletedOnceThisSession else { return }
        startupProgressActive = false
        let shouldNotify = !segmentationCompletedOnceThisSession
        segmentationCompletedOnceThisSession = true
        if shouldNotify {
            onFirstSegmentationComplete?()
        }
        progressContainer.alpha = 0
        progressContainer.isHidden = true
        progressContainer.alpha = 1
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

    /// Bbox height × room height when AR depth is unavailable (non-LiDAR / no AR height yet).
    private func proportionalFurnitureHeightMeters(bboxHeightPx: Int, imageHeight: Int) -> Float? {
        guard roomHeightMeters > 0.05, imageHeight > 0 else { return nil }
        let h = (Float(bboxHeightPx) / Float(imageHeight)) * roomHeightMeters
        guard h.isFinite, h > 0.01 else { return nil }
        return h
    }

    private func makeFurnitureSizeEstimate(
        widthMeters: Float,
        heightMeters: Float,
        arHeightMeters: Float?
    ) -> FurnitureSizeEstimate {
        var finalWidth = widthMeters
        var finalHeight = heightMeters
        var finalARHeight = arHeightMeters

        if arAssistedSizingEnabled, let arHeightMeters, arHeightMeters.isFinite, arHeightMeters > 0 {
            let clampedARHeight = max(arHeightMeters, Self.minimumARFurnitureHeightMeters)
            finalARHeight = clampedARHeight
            finalHeight = max(heightMeters, clampedARHeight)
            finalWidth = max(widthMeters, Self.minimumARFurnitureWidthMeters)
        }

        latestEstimatedFurnitureBaseWidthMeters = finalWidth
        latestEstimatedFurnitureBaseHeightMeters = finalHeight
        latestEstimatedFurnitureBaseARHeightMeters = finalARHeight

        if arAssistedSizingEnabled,
           let arHeight = finalARHeight,
           arHeight.isFinite,
           arHeight > 0 {
            let pinch = Float(max(userPinchScale, 0.01))
            finalWidth *= pinch
            finalHeight *= pinch
            finalARHeight = arHeight * pinch
        }

        return FurnitureSizeEstimate(
            widthMeters: finalWidth,
            heightMeters: finalHeight,
            arHeightMeters: finalARHeight
        )
    }

    private func publishLatestFurnitureSizeEstimateForCurrentPinchIfNeeded() {
        guard let callback = onFurnitureSizeEstimated,
              let width = latestEstimatedFurnitureBaseWidthMeters,
              let height = latestEstimatedFurnitureBaseHeightMeters,
              width.isFinite,
              height.isFinite,
              width > 0,
              height > 0 else { return }

        let arHeight = latestEstimatedFurnitureBaseARHeightMeters
        callback(
            makeFurnitureSizeEstimate(
                widthMeters: width,
                heightMeters: height,
                arHeightMeters: arHeight
            )
        )
    }

    /// Call on main **after** the composited segmentation mask is applied to `maskImageView` (or when compose failed but mask had foreground).
    private func commitFurnitureSizeAfterSegmentationMaskApplied(
        maskHasForeground: Bool,
        primaryMetricResult: PrimaryBboxMetersResult?,
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
        if hadSegmentationBeforeThisCommit {
            let arHeight = normalizedARFurnitureHeightMeters()
            let propH = proportionalFurnitureHeightMeters(bboxHeightPx: bboxHeightPx, imageHeight: imageHeight)

            if let p = primaryMetricResult {
                let heightMeters = arHeight ?? propH ?? p.size.height
                if arHeight == nil {
                    logFurnitureFitSize(
                        "phase=height_fallback source=\(p.pipeline) width_m=\(String(format: "%.3f", p.size.width)) " +
                        "height_m=\(String(format: "%.3f", heightMeters)) (prop_h=\(propH.map { String(format: "%.3f", $0) } ?? "—"))"
                    )
                }
                onFurnitureSizeEstimated?(makeFurnitureSizeEstimate(
                    widthMeters: p.size.width,
                    heightMeters: heightMeters,
                    arHeightMeters: arHeight
                ))
            } else if arHeight != nil || propH != nil {
                let bboxWRatio = Float(bboxWidthPx) / Float(max(1, imageWidth))
                let estimatedWidth = bboxWRatio * roomWidthMeters
                let heightMeters = arHeight ?? propH ?? 0
                logFurnitureFitSize(
                    "phase=commit_ar_publish arH=\(arHeight.map { String(format: "%.3f", $0) } ?? "nil") " +
                    "propH=\(propH.map { String(format: "%.3f", $0) } ?? "nil") estW=\(String(format: "%.3f", estimatedWidth))"
                )
                onFurnitureSizeEstimated?(makeFurnitureSizeEstimate(
                    widthMeters: estimatedWidth > 0.01 ? estimatedWidth : (latestEstimatedFurnitureBaseWidthMeters ?? 0.3),
                    heightMeters: heightMeters,
                    arHeightMeters: arHeight
                ))
            }
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
        let canShowStartupCompletionBanner = startupProgressActive
        hasFirstDetection = true
        startupProgressActive = false
        let shouldNotify = !segmentationCompletedOnceThisSession
        segmentationCompletedOnceThisSession = true
        if shouldNotify {
            onFirstSegmentationComplete?()
        }

        if let fw = furnitureWidthMeters, let fh = furnitureHeightMeters {
            let distStr = distanceMeters.map { String(format: "%.3f", $0) } ?? "—"
            logFurnitureFitSize(
                "phase=first pipeline=\(pipelineTag) dist_m=\(distStr) W×H_m=\(String(format: "%.3f", fw))×\(String(format: "%.3f", fh)) buffer=\(imageWidth)x\(imageHeight) bbox_px=\(bboxWidthPx)x\(bboxHeightPx) arDepthSnapshot=\(arDepthSnapshotAttached)"
            )
            let arHeight = normalizedARFurnitureHeightMeters()
            let propH = proportionalFurnitureHeightMeters(bboxHeightPx: bboxHeightPx, imageHeight: imageHeight)
            let heightMeters = arHeight ?? propH ?? fh
            if arHeight == nil {
                logFurnitureFitSize(
                    "phase=first_height_fallback source=\(pipelineTag) width_m=\(String(format: "%.3f", fw)) height_m=\(String(format: "%.3f", heightMeters))"
                )
            }
            let est = FurnitureSizeEstimate(
                widthMeters: fw,
                heightMeters: heightMeters,
                arHeightMeters: arHeight
            )
            DispatchQueue.main.async {
                self.onFurnitureSizeEstimated?(self.makeFurnitureSizeEstimate(
                    widthMeters: est.widthMeters,
                    heightMeters: est.heightMeters,
                    arHeightMeters: est.arHeightMeters
                ))
            }
        }

        let showMeterBanner = canShowStartupCompletionBanner &&
            furnitureWidthMeters != nil &&
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
        return arAssistedSizingEnabled ? max(value, Self.minimumARFurnitureHeightMeters) : value
    }

    // MARK: - Gestures
    @objc private func handlePinch(_ gesture: UIPinchGestureRecognizer) {
        guard maskImageView.image != nil else {
            logDebug("📐 [PINCH] ignored – no mask image")
            return
        }
        guard !isShowingLiveVideoIdentifications else { return }

        switch gesture.state {
        case .began:
            userLockedAssistedOverlayScale = true
            logDebug("📐 [PINCH] began  mode=\(overlayPresentationMode) pinchScale=\(String(format: "%.3f", userPinchScale))")
        case .changed:
            userLockedAssistedOverlayScale = true
            let newPinch = userPinchScale * gesture.scale
            userPinchScale = min(max(newPinch, 0.25), 4.0)
            applyCurrentOverlayScaleTransform()
            publishLatestFurnitureSizeEstimateForCurrentPinchIfNeeded()
            gesture.scale = 1.0
        case .ended, .cancelled:
            logDebug("📐 [PINCH] ended  pinchScale=\(String(format: "%.3f", userPinchScale))")
            if userPinchScale > 0.92 && userPinchScale < 1.08 {
                userPinchScale = 1.0
                userLockedAssistedOverlayScale = false
                UIView.animate(withDuration: 0.2) {
                    self.applyCurrentOverlayScaleTransform()
                }
                publishLatestFurnitureSizeEstimateForCurrentPinchIfNeeded()
            } else {
                userLockedAssistedOverlayScale = true
                UIView.animate(withDuration: 0.2) {
                    self.applyCurrentOverlayScaleTransform()
                }
                publishLatestFurnitureSizeEstimateForCurrentPinchIfNeeded()
            }
        default: break
        }
    }

    /// Reset overlay scale back to its default for the current detection: keep room and AR-assisted
    /// scaling, but remove user pinch so the furniture returns to its \"real\" size.
    @objc private func handleResetScaleTapped() {
        guard maskImageView.image != nil else { return }
        userPinchScale = 1.0
        userPanOffset = .zero
        userLockedAssistedOverlayScale = false
        if debugMode {
            logDebug("📐 [RESET] overlay scale reset to default (pinch=1.0, pan=zero)")
        }
        UIView.animate(withDuration: 0.2) {
            self.applyCurrentOverlayScaleTransform()
        }
        publishLatestFurnitureSizeEstimateForCurrentPinchIfNeeded()
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
        guard !isShowingLiveVideoIdentifications else { return }

        let translation = gesture.translation(in: self)
        switch gesture.state {
        case .began:
            logDebug("📐 [PAN] began  mode=\(overlayPresentationMode) offset=(\(String(format: "%.1f", userPanOffset.x)), \(String(format: "%.1f", userPanOffset.y)))")
        case .changed:
            userPanOffset.x += translation.x
            userPanOffset.y += translation.y
            applyCurrentOverlayScaleTransform()
            gesture.setTranslation(.zero, in: self)
        case .ended, .cancelled:
            logDebug("📐 [PAN] ended  offset=(\(String(format: "%.1f", userPanOffset.x)), \(String(format: "%.1f", userPanOffset.y)))")
        default: break
        }
    }

    @objc private func handleBoundingBoxTap(_ gesture: UITapGestureRecognizer) {
        guard gesture.state == .ended else { return }
        let pointInMask = gesture.location(in: maskImageView)
        guard let tappedIndex = candidateIndexForTap(pointInMask) else { return }
        guard tappedIndex < latestDisplayedCandidates.count else { return }
        let tappedDetection = latestDisplayedCandidates[tappedIndex]
        toggleSelectedDetection(tappedDetection)
        latestDisplayedSelectedCandidateIndex = tappedIndex
        let pins = selectedPinsSnapshot()
        detectionBBoxOverlayView.items = detectionBBoxOverlayView.items.enumerated().map { index, item in
            let isSel = index < latestDisplayedCandidates.count &&
                pins.contains { FurnitureFitIoU.calculate(latestDisplayedCandidates[index], $0) >= pinMatchIoUThreshold }
            return DetectionOverlayItem(
                rectInView: item.rectInView,
                label: item.label,
                confidence: item.confidence,
                isSelected: isSel
            )
        }
        logDebug(
            "👆 [TAP_SELECT] class=\(displayClassName(tappedDetection.classIdx)) conf=\(String(format: "%.2f", tappedDetection.confidence)) index=\(tappedIndex)"
        )
    }

    @objc private func handleClearSelectedObjectTapped() {
        clearSelectedClassSelections()
        if segmentationMode == .segmentSelected {
            resetOverlayScalesForEmptyMask(clearDetectedCandidates: false, clearSelections: false)
        }
        updateDetectionOverlaySelectionHighlights()
        logDebug("👆 [TAP_SELECT] cleared selected classes; returning to live identification feed")
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
        if gestureRecognizer is UITapGestureRecognizer {
            let touchPoint = touch.location(in: maskImageView)
            return candidateIndexForTap(touchPoint) != nil
        }

        if isShowingLiveVideoIdentifications &&
            (gestureRecognizer is UIPinchGestureRecognizer || gestureRecognizer is UIPanGestureRecognizer) {
            return false
        }

        // Pinch should be easy to start anywhere over the active segmented overlay.
        // Don't over-filter it touch-by-touch or the second finger frequently gets rejected.
        if gestureRecognizer is UIPinchGestureRecognizer {
            guard maskImageView.image != nil else {
                logDebug("👆 [shouldReceive pinch] No mask image")
                return false
            }
            guard primaryBboxInView.width > 0 && primaryBboxInView.height > 0 else {
                logDebug("👆 [shouldReceive pinch] No valid bbox")
                return false
            }
            return true
        }

        // Pan: accept touches inside or near the primary bbox so the overlay is easy to grab.
        guard maskImageView.image != nil else {
            logDebug("👆 [shouldReceive pan] No mask image")
            return false
        }
        guard primaryBboxInView.width > 0 && primaryBboxInView.height > 0 else {
            logDebug("👆 [shouldReceive pan] No valid bbox")
            return false
        }

        let touchPoint = touch.location(in: maskImageView)
        let expandedBbox = primaryBboxInView.insetBy(dx: -Self.pinchHitTestPadding,
                                                     dy: -Self.pinchHitTestPadding)
        let contains = expandedBbox.contains(touchPoint)

        logDebug("👆 [shouldReceive pan] touchPoint=\(touchPoint) bbox=\(primaryBboxInView) expanded=\(expandedBbox) contains=\(contains)")
        return contains
    }

    func gestureRecognizer(_ gestureRecognizer: UIGestureRecognizer, shouldBeRequiredToFailBy otherGestureRecognizer: UIGestureRecognizer) -> Bool {
        // Don't interfere with navigation gestures
        return false
    }

    // MARK: - Hit Testing

    /// Extra padding around `primaryBboxInView` so that the second pinch finger
    /// (which typically lands outside the tight detection box) still hit-tests to
    /// this view, allowing `UIPinchGestureRecognizer` to receive both touches.
    private static let pinchHitTestPadding: CGFloat = 100

    override func hitTest(_ point: CGPoint, with event: UIEvent?) -> UIView? {
        if !selectedObjectChipButton.isHidden {
            let pointInChip = convert(point, to: selectedObjectChipButton)
            if selectedObjectChipButton.point(inside: pointInChip, with: event) {
                return selectedObjectChipButton
            }
        }
        let pointInMask = convert(point, to: maskImageView)
        let candidateContains = candidateBboxesInView.contains { $0.insetBy(dx: -10, dy: -10).contains(pointInMask) }

        if candidateContains && maskImageView.image == nil {
            return super.hitTest(point, with: event)
        }

        if isShowingLiveVideoIdentifications {
            return candidateContains ? super.hitTest(point, with: event) : nil
        }

        guard maskImageView.image != nil else {
            logDebug("👆 [hitTest] No mask image, passing through")
            return nil
        }

        let hasActivePrimary = primaryBboxInView.width > 0 && primaryBboxInView.height > 0

        guard hasActivePrimary || candidateContains else {
            logDebug("👆 [hitTest] No valid bbox / candidate at point, passing through")
            return nil
        }

        logDebug("👆 [hitTest] point=\(point) pointInMask=\(pointInMask) bbox=\(primaryBboxInView)")

        // If a pinch or pan is already in flight, accept touches anywhere so the
        // gesture isn't interrupted when a finger drifts outside the box.
        if let pinch = overlayPinchGesture,
           pinch.state == .began || pinch.state == .changed {
            logDebug("👆 [hitTest] Pinch in progress – accepting")
            return super.hitTest(point, with: event)
        }
        if let pan = overlayPanGesture,
           pan.state == .began || pan.state == .changed {
            logDebug("👆 [hitTest] Pan in progress – accepting")
            return super.hitTest(point, with: event)
        }

        // Tight bbox: pan and tap
        if hasActivePrimary && primaryBboxInView.contains(pointInMask) {
            logDebug("👆 [hitTest] INSIDE bbox - handling touch")
            return super.hitTest(point, with: event)
        }

        if candidateContains {
            logDebug("👆 [hitTest] INSIDE candidate bbox - handling touch")
            return super.hitTest(point, with: event)
        }

        // Expanded bbox: allow pinch initiation when fingers straddle the edge
        let expanded = primaryBboxInView.insetBy(dx: -Self.pinchHitTestPadding, dy: -Self.pinchHitTestPadding)
        if hasActivePrimary && expanded.contains(pointInMask) {
            logDebug("👆 [hitTest] INSIDE expanded bbox (pinch margin) - handling touch")
            return super.hitTest(point, with: event)
        }

        logDebug("👆 [hitTest] OUTSIDE expanded bbox - passing through")
        return nil
    }
}

// MARK: - ARSessionDelegate (ARKit-as-camera path only; hybrid uses AVCapture + `makeDepthSnapshot` on capture thread)
extension FurnitureFitContainerView {
    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        guard isUsingARCameraPath else { return }
        isARDepthCompanionSessionRunning = true
        let t = CACurrentMediaTime()
        if lastARHeavyWorkFinishCAC > 0, t - lastARHeavyWorkFinishCAC < Self.arSessionDelegateHeavyMinInterval {
            return
        }
        frameLock.lock()
        if isProcessing {
            preferImmediateNextInference = true
            frameLock.unlock()
            return
        }
        isProcessing = true
        lastProcessTime = Date()
        frameLock.unlock()
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
        lastARHeavyWorkFinishCAC = CACurrentMediaTime()
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
