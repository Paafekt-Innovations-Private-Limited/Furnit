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
import SceneKit
import CoreText
import MetalKit
import simd

// MARK: - Main Container View
final class FurnitureFitContainerView: UIView, AVCaptureVideoDataOutputSampleBufferDelegate, ARSessionDelegate, UIGestureRecognizerDelegate {
    private static let fullVideoWithIdentificationsKey = "furnitureFit.showFullVideoWithIdentifications"
    private static let minimumARFurnitureHeightMeters: Float = 0.25
    private static let minimumARFurnitureWidthMeters: Float = 0.25

    private var oneImageRunAwaitingSave = false
    private var oneImageRunFinished = false

    // MARK: Config
    var processInterval: TimeInterval = 0.07
    var confidenceThreshold: Float = 0.10
    /// Minimum detector confidence (0…1) for **primary** furniture selection among qualifying boxes. Parsed candidates still use ``confidenceThreshold``.
    var primaryDetectionMinConfidence: Float = 0.57
    /// When `true`, primary is the **highest confidence** among boxes ≥ ``primaryDetectionMinConfidence`` (ties → larger area).
    /// When `false`, primary comes from the **3 largest** qualifying boxes by area, then picks the **highest confidence** within that shortlist.
    var primarySelectionByHighestConfidence: Bool = false
    /// Lower parse threshold used only for low-confidence secondary contributors.
    private let furnitureFitInsidePrimaryContributorConfidenceThreshold: Float = 0.10
    /// 3×3 binary closing on the composite band was filling thin gaps / “holes” in chair handles.
    /// Disabled so logits + threshold define the mask only (no morphology).
    private let furnitureFitNativeMaskMorphologicalClose: Bool = false
    // var useBilinearUpscaling: Bool = true
    var useBilinearUpscaling: Bool = false
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
    var onFurnitureSizeEstimated: ((FurnitureSizeEstimate) -> Void)?
    /// Mean straight sRGB (0…1) over opaque-enough pixels of the composited segmentation cutout; throttled (~4 Hz).
    var onSegmentationMaskMeanColorSRGB: ((SIMD3<Float>) -> Void)?
    /// Sharp Room uses this to opt into AR sizing explicitly instead of defaulting to it.
    var arAssistedSizingEnabled: Bool = true
    /// Optional manual furniture height from the room viewers' calibrate sheet. When set, the overlay
    /// uses this value for AR-assisted scaling instead of the raw depth-derived height.
    var manualFurnitureHeightOverrideMeters: Float?

    private var lastSegmentationMeanColorPublishAt: CFAbsoluteTime = 0
    private let segmentationMeanColorMinPublishInterval: CFTimeInterval = 0.25

    // Debug mode - read from settings
    var debugMode: Bool {
        return AppStateManager.shared.qualitySettings.debugMode
    }

    // MARK: - Ignored Classes (loaded from blacklist.json)
    private let classBlacklist = FurnitureFitClassBlacklist()

    func submitStillImageForScanning(_ image: UIImage, requestID: UUID) {
        pendingStillImageScanRequest = (image, requestID)
        if processedStillImageScanRequestID == requestID { return }
        DispatchQueue.main.async { [weak self] in
            self?.setProgress(0.05, text: "Scanning selected photo…")
        }
        detectionQueue.async { [weak self] in
            self?.runPendingStillImageScanIfNeeded()
        }
    }

    private func runPendingStillImageScanIfNeeded() {
        guard stillImageScanModeEnabled,
              let pendingRequest = pendingStillImageScanRequest,
              processedStillImageScanRequestID != pendingRequest.requestID else { return }
        guard mlModel != nil else { return }
        guard let pixelBuffer = YoloEImageInference.pixelBufferFromImage(pendingRequest.image) else {
            DispatchQueue.main.async { [weak self] in
                self?.setProgress(1.0, text: "Photo load failed")
                self?.finishStartupProgressIfNeeded()
            }
            return
        }

        frameLock.lock()
        if isProcessing {
            frameLock.unlock()
            return
        }
        isProcessing = true
        frameLock.unlock()

        DispatchQueue.main.async { [weak self] in
            self?.setProgress(0.12, text: "Scanning selected photo…")
        }
        processFrameInner(pixelBuffer, arDepthSnapshot: nil)
        processedStillImageScanRequestID = pendingRequest.requestID

        if let latestRequest = pendingStillImageScanRequest,
           latestRequest.requestID != processedStillImageScanRequestID {
            detectionQueue.async { [weak self] in
                self?.runPendingStillImageScanIfNeeded()
            }
        }
    }

    // MARK: Camera
    private let captureSession = AVCaptureSession()
    private var captureSessionObserverTokens: [NSObjectProtocol] = []
    private let videoOutput = AVCaptureVideoDataOutput()
    private let sampleQueue = DispatchQueue(label: "com.furnit.sample", qos: .userInitiated)
    private let captureSessionControlQueue = DispatchQueue(label: "com.furnit.capture.control", qos: .userInitiated)
    private var coreMLInferenceQueueGeneration: UInt = 0
    private var coreMLInferenceQueue = DispatchQueue(label: "com.furnit.coreml.inference.0", qos: .userInitiated)
    private var coreMLInferenceBackoffUntil: Date?
    private let coreMLInferenceWarningSeconds: TimeInterval = 2.0
    private let coreMLInferenceTimeoutSeconds: TimeInterval = 12.0

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
    /// When enabled, skip camera startup and run the selected still image through the same segmentation pipeline.
    var stillImageScanModeEnabled: Bool = false
    private var pendingStillImageScanRequest: (image: UIImage, requestID: UUID)?
    private var processedStillImageScanRequestID: UUID?

    // MARK: - Overlay scale (room × AR when enabled × user pinch)
    /// Room/AR-derived scale factor before the user's pinch multiplier.
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
        return false
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
    /// Auto-primary hysteresis state for identify-only mode.
    private var stableAutoPrimaryDetection: FurnitureFitDetection?
    private var pendingAutoPrimaryDetection: FurnitureFitDetection?
    private var pendingAutoPrimaryFrameCount: Int = 0
    private let autoPrimaryPersistenceIoUThreshold: Float = 0.45
    private let autoPrimarySwitchRequiredFrames: Int = 3
    private let autoPrimaryConfidenceSwitchGain: Float = 1.08
    private let autoPrimaryConfidenceSwitchMargin: Float = 0.03
    private let autoPrimaryAreaShortlistCount: Int = 3

    // MARK: - Metal (FIXED: stored properties instead of computed to prevent resource leak)
    private var metalDevice: MTLDevice?
    private var metalCommandQueue: MTLCommandQueue?  // FIXED: was computed property creating new queue on every access
    private var metalLibrary: MTLLibrary?            // FIXED: was computed property creating new library on every access
    /// process_mask_native: bilinear upsample + threshold >0 on GPU.
    private var bilinearUpsamplePipeline: MTLComputePipelineState?

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

    // Once-per-session diagnostic logs to verify proto/det tensor layouts hit the Accelerate fast paths.
    // Reset on session stop so a new model/shape emits fresh diagnostics.
    private var didLogProtoLayoutDiagnostic = false
    private var didLogDetLayoutDiagnostic = false

    // Per-frame ANE→CPU materialization split out of the proto/parse timers.
    // Measured by forcing a `.dataPointer` touch before the memcpy/parse work begins.
    private var lastProtoDataPointerSyncMs: Double = 0
    private var lastDetDataPointerSyncMs: Double = 0

    // MARK: - Reusable CVPixelBuffer & MLMultiArray (prevents allocation per frame)
    /// Shared square model-input buffer reused by both stretch and letterbox preprocessing.
    private var cachedSquareBuffer: CVPixelBuffer?
    private var cachedSquareSize: Int = 0
    private var cachedMLArray: MLMultiArray?
    private var cachedMLArraySize: Int = 0

    /// Reused scratch for CPU fallback compositing (vDSP + BLAS); preallocated for common widths and grows if needed.
    private var compositeCpuScratchFloats: [Float] = [Float](repeating: 0, count: 5 * 2048)
    /// Proto mask upscaled to the active composite band via vImage; grows to the largest band seen and never shrinks.
    private var upscaledPlanarMaskScratch: [UInt8] = []
    /// Whether the current model frame used Ultralytics-style letterbox instead of stretch.
    private var currentYoloUsesLetterbox: Bool = false
    /// Shared vertical-flip activation for live Furniture Fit and still-image inference.
    private var furnitureFitVerticalFlipModelInput: Bool { YoloEImageInference.verticalFlipModelInputEnabled }
    /// Set per frame when ``furnitureFitVerticalFlipModelInput`` is enabled and the flip succeeds.
    private var currentYoloInputVerticallyFlipped: Bool = false
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
    private var mlModel: MLModel?  // yoloe-11l-seg-pf via YOLOEModelService
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
        if oneImageRunAwaitingSave {
            oneImageRunAwaitingSave = false
            oneImageRunFinished = true
            logDebug("🖼️ oneImageRun: finished without writing PNG (early exit or no composed image)")
        }
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
            if self.isShowingLiveVideoIdentifications {
                self.maskImageView.image = nil
            }
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
                let effectiveARHeight = resolvedAssistedFurnitureHeightMeters(
                    estimatedHeightMeters: estH
                ) ?? estH
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
        resetAutoPrimarySelectionStability()
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
        let scaledBox = FurnitureFitGeometry.scaleBoxesFromModel(
            box: modelBox,
            modelShape: modelShape,
            imageShape: CGSize(width: imageWidth, height: imageHeight),
            usesLetterbox: currentYoloUsesLetterbox,
            inputVerticallyFlipped: currentYoloInputVerticallyFlipped
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
        let tapHitPadding: CGFloat = isShowingLiveVideoIdentifications ? 22 : 10
        let paddedMatches = candidateBboxesInView.enumerated().filter { _, rect in
            rect.insetBy(dx: -tapHitPadding, dy: -tapHitPadding).contains(pointInMaskView)
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
        let shouldRequireMaskHit = hasTapMaskState && !isShowingLiveVideoIdentifications
        if shouldRequireMaskHit, maskMatches.isEmpty {
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
        // Note: class blacklist is loaded in startIfNeeded() to avoid repeated calls from updateUIView
    }

    func startIfNeeded() {
        // SwiftUI calls this on many `updateUIView` passes; only run setup once until `stop()`.
        if captureSession.isRunning { return }
        if stillImageScanModeEnabled, furnitureFitCameraStartupInitiated {
            detectionQueue.async { [weak self] in
                self?.runPendingStillImageScanIfNeeded()
            }
            return
        }
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
        classBlacklist.loadBlacklistOnce(debugMode: debugMode) { logDebug($0) }

        if stillImageScanModeEnabled {
            setProgress(0.08, text: "Still image scan")
            detectionQueue.async { [weak self] in
                self?.runPendingStillImageScanIfNeeded()
            }
            return
        }

        if FurnitureFitOneImageDebugSupport.runEnabled {
            setProgress(0.08, text: "One-image debug…")
            detectionQueue.asyncAfter(deadline: .now() + 0.2) { [weak self] in
                self?.runSingleImageDebugPipelineIfNeeded()
            }
            return
        }

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
        requestCameraPermissionAndStart()

        // Set initial video rotation and progress orientation after camera starts
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) { [weak self] in
            guard let self = self else { return }
            self.updateVideoRotationForOrientation(self.currentDeviceOrientation)
            self.updateProgressOrientationOnMain(self.currentDeviceOrientation)
        }
    }

    /// Runs one resolved still image on ``detectionQueue`` (full ONNX-style composite). Skips camera; writes PNG on success.
    private func runSingleImageDebugPipelineIfNeeded() {
        guard FurnitureFitOneImageDebugSupport.runEnabled, !oneImageRunFinished else { return }
        for _ in 0..<80 {
            if mlModel != nil { break }
            Thread.sleep(forTimeInterval: 0.05)
        }
        guard mlModel != nil else {
            logDebug("🖼️ oneImageRun: no Core ML model after wait — abort")
            oneImageRunFinished = true
            DispatchQueue.main.async { [weak self] in
                self?.setProgress(1.0, text: "One-image: no model")
                self?.finishStartupProgressIfNeeded()
            }
            return
        }
        guard let inputURL = FurnitureFitOneImageDebugSupport.resolvedInputURL() else {
            FurnitureFitOneImageDebugSupport.logInputNotFound()
            oneImageRunFinished = true
            DispatchQueue.main.async { [weak self] in
                self?.setProgress(1.0, text: "One-image: missing file")
                self?.finishStartupProgressIfNeeded()
            }
            return
        }
        guard let pixelBuffer = YoloEImageInference.pixelBufferFromImage(atPath: inputURL.path) else {
            logDebug("🖼️ oneImageRun: could not decode image at \(inputURL.path)")
            oneImageRunFinished = true
            DispatchQueue.main.async { [weak self] in
                self?.setProgress(1.0, text: "One-image: load failed")
                self?.finishStartupProgressIfNeeded()
            }
            return
        }
        oneImageRunAwaitingSave = true
        frameLock.lock()
        isProcessing = true
        frameLock.unlock()
        logDebug("🖼️ oneImageRun: starting pipeline for \(inputURL.path)")
        processFrameInner(pixelBuffer, arDepthSnapshot: nil)
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
        // Rotate the CoreML inference queue and clear watchdog backoff so any in-flight
        // session-N `model.prediction` closure cannot block session-(N+1) frames when it
        // eventually completes on the abandoned queue. Without this, stop→start can leave
        // the shared inference queue backed up with stale work and only the first frame
        // of the next session gets through before subsequent frames sit behind the relic.
        //
        // Serialize the write onto `detectionQueue` to match the existing timeout-path
        // rotate (line ~3342), which writes `coreMLInferenceQueue` from the same queue
        // that reads it inside `performCoreMLPredictionWithWatchdog`. Using the same
        // serialization avoids a data race on the property.
        detectionQueue.async { [weak self] in
            guard let self else { return }
            self.coreMLInferenceBackoffUntil = nil
            self.coreMLInferenceQueueGeneration &+= 1
            self.coreMLInferenceQueue = DispatchQueue(
                label: "com.furnit.coreml.inference.\(self.coreMLInferenceQueueGeneration)",
                qos: .userInitiated
            )
            if self.debugMode {
                logDebug("🔄 stop(): rotated coreMLInferenceQueue → gen=\(self.coreMLInferenceQueueGeneration), cleared backoff")
            }
        }
        lastARHeavyWorkFinishCAC = 0
        // Synchronous reset: if UI/flags were only cleared in `main.async`, the next `startIfNeeded()` could run first and keep stale state.
        hasFirstDetection = false
        segmentationCompletedOnceThisSession = false
        startupProgressActive = false
        didLogProtoLayoutDiagnostic = false
        didLogDetLayoutDiagnostic = false
        lastSegmentationMeanColorPublishAt = 0
        maskImageView.image = nil
        resetOverlayScalesForEmptyMask(clearDetectedCandidates: true, clearSelections: true)
        logFurnitureFitSize(
            "phase=session_stop cleared isProcessing + segmentation UI flags (fixes stuck frames + stale first-mask after brain off/on)"
        )
        furnitureFitCameraStartupInitiated = false
        oneImageRunAwaitingSave = false
        oneImageRunFinished = false
        pendingStillImageScanRequest = nil
        processedStillImageScanRequestID = nil
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
        if debugMode { logDebug("📥 captureOutput dispatching frame (isProcessing was false)") }
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
        currentYoloInputVerticallyFlipped = false

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
        if furnitureFitVerticalFlipModelInput {
            if YoloEImageInference.flipPixelBufferVerticallyBGRA(modelInputBuffer) {
                currentYoloInputVerticallyFlipped = true
                if debugMode { logDebug("⏱️ ONNX-STYLE: model input vertically flipped (flipud correction active)") }
            } else if debugMode {
                logDebug("⚠️ ONNX-STYLE: vertical flip of model input failed; continuing unflipped")
            }
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
        if let backoffUntil = coreMLInferenceBackoffUntil, backoffUntil > Date() {
            if debugMode {
                logDebug(
                    "⚠️ ONNX-STYLE Core ML STAGE 2 SKIPPED: cooling down after timeout for " +
                    "\(String(format: "%.2f", backoffUntil.timeIntervalSinceNow)) s"
                )
            }
            resetProcessingFlag()
            return
        }
        let processedSuccessfully = autoreleasepool { () -> Bool in
            guard let output = performCoreMLPredictionWithWatchdog(
                model: model,
                inputProvider: inputProvider,
                startedAt: t2
            ) else {
                return false
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
                return false
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
            return true
        }
        guard processedSuccessfully else {
            resetProcessingFlag()
            return
        }
    }

    private func performCoreMLPredictionWithWatchdog(
        model: MLModel,
        inputProvider: MLFeatureProvider,
        startedAt: Date
    ) -> MLFeatureProvider? {
        let semaphore = DispatchSemaphore(value: 0)
        let inferenceQueue = coreMLInferenceQueue
        var outputProvider: MLFeatureProvider?
        var predictionError: Error?

        let warningWorkItem = DispatchWorkItem { [weak self] in
            guard let self, self.debugMode else { return }
            logDebug(
                "⚠️ ONNX-STYLE Core ML inference still running after " +
                "\(String(format: "%.2f", Date().timeIntervalSince(startedAt) * 1000)) ms"
            )
            logMemory("DURING ONNX-STYLE Core ML INFERENCE")
        }
        DispatchQueue.global(qos: .utility).asyncAfter(
            deadline: .now() + coreMLInferenceWarningSeconds,
            execute: warningWorkItem
        )

        inferenceQueue.async {
            defer { semaphore.signal() }
            do {
                outputProvider = try model.prediction(from: inputProvider)
            } catch {
                predictionError = error
            }
        }

        let waitResult = semaphore.wait(timeout: .now() + coreMLInferenceTimeoutSeconds)
        warningWorkItem.cancel()

        if debugMode {
            let resultLabel = (waitResult == .success) ? "success" : "TIMED_OUT"
            logDebug("🔓 coreml semaphore returned: \(resultLabel) after \(String(format: "%.2f", Date().timeIntervalSince(startedAt) * 1000)) ms")
        }

        switch waitResult {
        case .success:
            if let predictionError {
                if debugMode {
                    logDebug("❌ ONNX-STYLE Core ML STAGE 2 FAILED: Model inference error: \(predictionError)")
                }
                return nil
            }
            coreMLInferenceBackoffUntil = nil
            return outputProvider
        case .timedOut:
            if debugMode {
                logDebug(
                    "❌ ONNX-STYLE Core ML STAGE 2 TIMED OUT after " +
                    "\(String(format: "%.2f", coreMLInferenceTimeoutSeconds * 1000)) ms; rotating inference queue"
                )
                logMemory("ONNX-STYLE Core ML INFERENCE TIMEOUT")
            }
            coreMLInferenceQueueGeneration += 1
            coreMLInferenceQueue = DispatchQueue(
                label: "com.furnit.coreml.inference.\(coreMLInferenceQueueGeneration)",
                qos: .userInitiated
            )
            coreMLInferenceBackoffUntil = Date().addingTimeInterval(1.5)
            return nil
        }
    }

    private func areaShortlistedPrimaryCandidateIndices(
        candidates: [FurnitureFitDetection],
        minConfidence: Float
    ) -> [Int] {
        Array(
            candidates.enumerated()
                .filter { _, detection in detection.confidence >= minConfidence }
                .sorted { lhs, rhs in
                    let lhsArea = lhs.element.w * lhs.element.h
                    let rhsArea = rhs.element.w * rhs.element.h
                    if abs(lhsArea - rhsArea) > 1e-6 {
                        return lhsArea > rhsArea
                    }
                    if abs(lhs.element.confidence - rhs.element.confidence) > 1e-6 {
                        return lhs.element.confidence > rhs.element.confidence
                    }
                    return lhs.offset < rhs.offset
                }
                .prefix(autoPrimaryAreaShortlistCount)
                .map(\.offset)
        )
    }

    private func weightedPrimarySelectionScore(
        detection: FurnitureFitDetection,
        areaNormalizationReference: Float
    ) -> Float {
        let normalizedArea = areaNormalizationReference > 1e-6
            ? (detection.w * detection.h) / areaNormalizationReference
            : 0
        return 0.5 * detection.confidence + 0.5 * normalizedArea
    }

    /// Primary index: among candidates with confidence ≥ ``primaryDetectionMinConfidence``, either highest confidence directly
    /// or the best 50/50 confidence+area score inside the top-area shortlist (see ``primarySelectionByHighestConfidence``).
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
        let shortlistedIndices = areaShortlistedPrimaryCandidateIndices(
            candidates: candidates,
            minConfidence: minConfidence
        )
        let shortlistAreaReference = shortlistedIndices
            .map { candidates[$0].w * candidates[$0].h }
            .max() ?? 0
        return shortlistedIndices.max { lhsIndex, rhsIndex in
            let lhsDetection = candidates[lhsIndex]
            let rhsDetection = candidates[rhsIndex]
            let lhsScore = weightedPrimarySelectionScore(
                detection: lhsDetection,
                areaNormalizationReference: shortlistAreaReference
            )
            let rhsScore = weightedPrimarySelectionScore(
                detection: rhsDetection,
                areaNormalizationReference: shortlistAreaReference
            )
            if abs(lhsScore - rhsScore) > 1e-6 {
                return lhsScore < rhsScore
            }
            if abs(lhsDetection.confidence - rhsDetection.confidence) > 1e-6 {
                return lhsDetection.confidence < rhsDetection.confidence
            }
            let lhsArea = lhsDetection.w * lhsDetection.h
            let rhsArea = rhsDetection.w * rhsDetection.h
            if abs(lhsArea - rhsArea) > 1e-6 {
                return lhsArea < rhsArea
            }
            return lhsIndex > rhsIndex
        }
    }

    private func resetAutoPrimarySelectionStability() {
        stableAutoPrimaryDetection = nil
        pendingAutoPrimaryDetection = nil
        pendingAutoPrimaryFrameCount = 0
    }

    private func matchedCandidateIndexForStableAutoPrimary(
        reference: FurnitureFitDetection,
        candidates: [FurnitureFitDetection]
    ) -> Int? {
        var bestIndex: Int?
        var bestIoU: Float = 0
        for (index, candidate) in candidates.enumerated() {
            guard candidate.classIdx == reference.classIdx else { continue }
            let iou = FurnitureFitIoU.calculate(candidate, reference)
            if iou > bestIoU {
                bestIoU = iou
                bestIndex = index
            }
        }
        guard let bestIndex, bestIoU >= autoPrimaryPersistenceIoUThreshold else { return nil }
        return bestIndex
    }

    private func isSameStableAutoPrimaryTrack(_ lhs: FurnitureFitDetection, _ rhs: FurnitureFitDetection) -> Bool {
        lhs.classIdx == rhs.classIdx && FurnitureFitIoU.calculate(lhs, rhs) >= autoPrimaryPersistenceIoUThreshold
    }

    /// Keep the current auto-primary unless a challenger is clearly better for several consecutive frames.
    private func selectStablePrimaryIndexAutoFlow(candidates: [FurnitureFitDetection], modelSide: Int) -> Int? {
        guard let preferredIndex = selectPrimaryIndexCoreFlow(candidates: candidates, modelSide: modelSide) else {
            resetAutoPrimarySelectionStability()
            return nil
        }

        let preferredCandidate = candidates[preferredIndex]
        let minimumConfidence = min(max(primaryDetectionMinConfidence, 0.05), 0.99)

        guard let stableReference = stableAutoPrimaryDetection,
              let stableIndex = matchedCandidateIndexForStableAutoPrimary(reference: stableReference, candidates: candidates) else {
            stableAutoPrimaryDetection = preferredCandidate
            pendingAutoPrimaryDetection = nil
            pendingAutoPrimaryFrameCount = 0
            return preferredIndex
        }

        let stableCandidate = candidates[stableIndex]
        guard stableCandidate.confidence >= minimumConfidence else {
            stableAutoPrimaryDetection = preferredCandidate
            pendingAutoPrimaryDetection = nil
            pendingAutoPrimaryFrameCount = 0
            return preferredIndex
        }

        if !primarySelectionByHighestConfidence {
            let shortlistedIndices = Set(
                areaShortlistedPrimaryCandidateIndices(
                    candidates: candidates,
                    minConfidence: minimumConfidence
                )
            )
            guard shortlistedIndices.contains(stableIndex) else {
                stableAutoPrimaryDetection = preferredCandidate
                pendingAutoPrimaryDetection = nil
                pendingAutoPrimaryFrameCount = 0
                return preferredIndex
            }
        }

        if stableIndex == preferredIndex {
            stableAutoPrimaryDetection = stableCandidate
            pendingAutoPrimaryDetection = nil
            pendingAutoPrimaryFrameCount = 0
            return stableIndex
        }

        let shortlistAreaReference: Float
        if primarySelectionByHighestConfidence {
            shortlistAreaReference = 0
        } else {
            shortlistAreaReference = Set(
                areaShortlistedPrimaryCandidateIndices(
                    candidates: candidates,
                    minConfidence: minimumConfidence
                )
            )
            .map { candidates[$0].w * candidates[$0].h }
            .max() ?? 0
        }
        let stableScore = primarySelectionByHighestConfidence
            ? stableCandidate.confidence
            : weightedPrimarySelectionScore(
                detection: stableCandidate,
                areaNormalizationReference: shortlistAreaReference
            )
        let preferredScore = primarySelectionByHighestConfidence
            ? preferredCandidate.confidence
            : weightedPrimarySelectionScore(
                detection: preferredCandidate,
                areaNormalizationReference: shortlistAreaReference
            )
        let switchThreshold = max(
            stableScore * autoPrimaryConfidenceSwitchGain,
            stableScore + autoPrimaryConfidenceSwitchMargin
        )

        guard preferredScore >= switchThreshold else {
            stableAutoPrimaryDetection = stableCandidate
            pendingAutoPrimaryDetection = nil
            pendingAutoPrimaryFrameCount = 0
            return stableIndex
        }

        if let pendingCandidate = pendingAutoPrimaryDetection,
           isSameStableAutoPrimaryTrack(preferredCandidate, pendingCandidate) {
            pendingAutoPrimaryFrameCount += 1
        } else {
            pendingAutoPrimaryDetection = preferredCandidate
            pendingAutoPrimaryFrameCount = 1
        }

        if pendingAutoPrimaryFrameCount >= autoPrimarySwitchRequiredFrames {
            stableAutoPrimaryDetection = preferredCandidate
            pendingAutoPrimaryDetection = nil
            pendingAutoPrimaryFrameCount = 0
            return preferredIndex
        }

        stableAutoPrimaryDetection = stableCandidate
        return stableIndex
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
        let prototypeParseStart = Date()
        guard let protoInfo = parsePrototypes(protoArray) else {
            if debugMode { logDebug("❌ ONNX-STYLE (\(stage2DebugLabel)): parse prototypes failed") }
            resetProcessingFlag()
            return
        }
        let prototypeParseMs = Date().timeIntervalSince(prototypeParseStart) * 1000
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
        // Support-surface hints (floor, carpet, tile, etc.) normally inspect the unblacklisted detections,
        // but the one-image debug path should respect blacklist.json consistently.
        let detectionParseAndFilterStart = Date()

        // Split ANE→CPU materialization out of the parse timer (see `parsePrototypes` for rationale).
        // For the 620 MB detArray this is typically the dominant cost when CoreML lazily DMAs from ANE.
        let detSyncStart = Date()
        _ = detArray.dataPointer.load(as: UInt8.self)
        lastDetDataPointerSyncMs = Date().timeIntervalSince(detSyncStart) * 1000

        if debugMode, !didLogDetLayoutDiagnostic {
            didLogDetLayoutDiagnostic = true
            let detShape = detArray.shape.map { $0.intValue }
            let detStrides = detArray.strides.map { $0.intValue }
            let dataTypeRaw = detArray.dataType.rawValue
            let dataTypeName: String
            if detArray.dataType == .float32 {
                dataTypeName = "float32"
            } else if detArray.dataType == .float16 {
                dataTypeName = "float16"
            } else if detArray.dataType == .float64 {
                dataTypeName = "float64"
            } else if detArray.dataType == .int32 {
                dataTypeName = "int32"
            } else {
                dataTypeName = "unknown"
            }
            // Replicate YoloEDetectionParser.isContiguous: standard C-contiguous strides?
            var expected = 1
            var isCContiguous = !detShape.isEmpty && detShape.count == detStrides.count
            if isCContiguous {
                for i in stride(from: detShape.count - 1, through: 0, by: -1) {
                    if detStrides[i] != expected { isCContiguous = false; break }
                    expected *= detShape[i]
                }
            }
            let parserPath: String
            switch detArray.dataType {
            case .float32:
                parserPath = "FAST:float32+directPointer"
            case .float16:
                parserPath = isCContiguous ? "FAST:float16+bulkVImage" : "SLOW:float16+rowByRowOrScalar"
            default:
                parserPath = "UNSUPPORTED"
            }
            logDebug("🔬 parseDetections layout dtype=\(dataTypeName)(\(dataTypeRaw)) shape=\(detShape) strides=\(detStrides) contiguous=\(isCContiguous) path=\(parserPath)")
        }

        let parseBlacklist: Set<Int> = (oneImageRunAwaitingSave || stillImageScanModeEnabled) ? classBlacklist.ignoredIndices : []
        let rawDetections = YoloEDetectionParser.parseDetections(
            detArray: detArray,
            confidenceThreshold: confidenceThreshold,
            classBlacklist: parseBlacklist,
            maximumDetections: 1000
        )

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

        if debugMode {
            let rawDetectionClassNames = rawDetections.map { className($0.classIdx) }.joined(separator: ", ")
            logDebug("🧾 ONNX-STYLE (\(stage2DebugLabel)) raw detections: \(rawDetectionClassNames)")
            let rawBedFamilyDetections = rawDetections.filter {
                let normalizedClassName = displayClassName($0.classIdx).lowercased()
                return normalizedClassName.contains("bed")
            }
            if !rawBedFamilyDetections.isEmpty {
                let rawBedFamilySummary = rawBedFamilyDetections.map {
                    "\(displayClassName($0.classIdx)) conf=\(String(format: "%.3f", $0.confidence)) ctr=(\(Int($0.x)),\(Int($0.y))) sz=\(Int($0.w))x\(Int($0.h)))"
                }.joined(separator: " | ")
                print("🛏️ ONNX-STYLE (\(stage2DebugLabel)) raw bed-family detections: \(rawBedFamilySummary)")
            }
        }

        var candidates: [FurnitureFitDetection]
        if rawDetections.count > 1 {
            let sorted = rawDetections.sorted { $0.confidence > $1.confidence }.filter {
                !parseBlacklist.contains($0.classIdx)
            }
            let capped = Array(sorted.prefix(300))
            candidates = FurnitureFitNMS.applySortedByConfidence(detections: capped, iouThreshold: 0.5)
        } else {
            candidates = rawDetections.filter {
                !parseBlacklist.contains($0.classIdx)
            }
        }

        let supportSurfaceHintCandidates = (oneImageRunAwaitingSave || stillImageScanModeEnabled)
            ? FurnitureFitFilter.excludingClasses(candidates, blacklist: classBlacklist.ignoredIndices)
            : candidates
        let supportSurfaceHintName = supportSurfaceHint(
            from: supportSurfaceHintCandidates,
            imageWidth: onnxSide,
            imageHeight: onnxSide
        )
        let preferRoomRaycastSizing = supportSurfaceHintName != nil

        candidates = FurnitureFitFilter.excludingClasses(candidates, blacklist: classBlacklist.ignoredIndices)
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
            logDebug("📦 ONNX-STYLE (\(stage2DebugLabel)) candidates: \(candidates.count) (parse conf≥\(confidenceThreshold), class-aware NMS IoU≤0.5, blacklist.json \(classBlacklist.ignoredIndices.count) ids)")
            for (i, d) in candidates.enumerated() {
                logDebug("   [\(i)] \(className(d.classIdx)) conf=\(String(format: "%.2f", d.confidence)) ctr=(\(Int(d.x)),\(Int(d.y))) sz=\(Int(d.w))x\(Int(d.h))")
            }
        }
        let detectionParseAndFilterMs = Date().timeIntervalSince(detectionParseAndFilterStart) * 1000

        if segmentationMode == .identifyOnly && showFullVideoWithIdentifications && debugMode && !oneImageRunAwaitingSave {
            let liveDebugMaskBaseImage = debugMode
                ? renderOriginalFrameCGImage(
                    processBuffer: processBuffer,
                    origW: bufW,
                    origH: bufH
                )
                : nil
            let primaryIdxOpt = selectStablePrimaryIndexAutoFlow(candidates: candidates, modelSide: onnxSide)
            let liveDebugFusedMaskPayload: (logits: [Float], detections: [FurnitureFitDetection])? = {
                guard debugMode,
                      let primaryIdx = primaryIdxOpt,
                      primaryIdx >= 0,
                      primaryIdx < candidates.count else { return nil }
                let primaryCandidate = candidates[primaryIdx]
                let maskSource = FurnitureFitOnnxStylePipeline.collectMaskDetections(
                    primaryIndex: primaryIdx,
                    detections: candidates,
                    protos: planes,
                    protoHeight: pH,
                    protoWidth: pW,
                    modelSide: onnxSide
                )
                let expandedPrimary = onnxStyleExpandedPrimaryForMaskBuild(primaryCandidate, onnxSide: onnxSide)
                var fusedDetections: [FurnitureFitDetection] = [expandedPrimary]
                for detection in maskSource where FurnitureFitIoU.calculate(detection, primaryCandidate) < 0.999 {
                    fusedDetections.append(detection)
                }
                let fusedMaskResult = FurnitureFitOnnxStylePipeline.buildFullFieldLogitMask(
                    planes: planes,
                    protoW: pW,
                    protoH: pH,
                    detections: fusedDetections,
                    modelSide: onnxSide
                )
                var fusedLogits = fusedMaskResult.logits
                if currentYoloInputVerticallyFlipped {
                    FurnitureFitOnnxStylePipeline.flipProtoFloatGridVertically(&fusedLogits, protoW: pW, protoH: pH)
                }
                return (fusedLogits, fusedDetections)
            }()
            let liveDebugMaskOverlayImage = debugMode
                ? renderLiveDebugMaskOverlayImage(
                    baseImage: liveDebugMaskBaseImage,
                    candidates: candidates,
                    primaryIndex: primaryIdxOpt,
                    fusedMaskLogits: liveDebugFusedMaskPayload?.logits,
                    fusedMaskDetections: liveDebugFusedMaskPayload?.detections ?? [],
                    planes: planes,
                    protoW: pW,
                    protoH: pH,
                    modelSide: onnxSide,
                    origW: bufW,
                    origH: bufH,
                    scaleX: scaleX,
                    scaleY: scaleY
                )
                : nil
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
                if self.isShowingLiveVideoIdentifications {
                    self.maskImageView.image = nil
                } else if let liveDebugMaskOverlayImage {
                    let scale = self.window?.windowScene?.screen.scale ?? self.traitCollection.displayScale
                    self.maskImageView.image = UIImage(cgImage: liveDebugMaskOverlayImage, scale: scale, orientation: .up)
                } else {
                    self.maskImageView.image = nil
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
            guard let autoIdx = selectStablePrimaryIndexAutoFlow(candidates: candidates, modelSide: onnxSide) else {
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

        let lowConfidencePrimaryContributors = lowConfidenceInsidePrimaryContributors(
            detArray: detArray,
            primary: primary,
            blacklist: classBlacklist.ignoredIndices,
            protos: planes,
            protoHeight: pH,
            protoWidth: pW,
            modelSide: onnxSide
        )
        if debugMode, !lowConfidencePrimaryContributors.isEmpty {
            logDebug(
                "🧵 ONNX-STYLE (\(stage2DebugLabel)) mask-overlap reparsed contributors: " +
                "\(lowConfidencePrimaryContributors.count) (conf≥\(furnitureFitInsidePrimaryContributorConfidenceThreshold), overlap≥0.15)"
            )
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
            var selectedFusion = selectedCandidates.map {
                onnxStyleExpandedPrimaryForMaskBuild($0, onnxSide: onnxSide)
            }
            let selectedBaseCount = selectedFusion.count
            appendUniqueMaskContributors(from: lowConfidencePrimaryContributors, to: &selectedFusion)
            let supplementalContributors = Array(selectedFusion.dropFirst(selectedBaseCount))
            let deduplicatedSupplementalContributors = limitedMaskFusionContributors(
                primary: primary,
                contributors: supplementalContributors
            )
            maskDetectionsForBuild = Array(selectedFusion.prefix(selectedBaseCount)) + deduplicatedSupplementalContributors
        } else {
            // Auto primary: Android-style fusion of primary + supporting
            // detections that intersect it (monitor/table, overlapped props,
            // etc.), with an expanded primary bbox for mask synthesis.
            var maskSource = FurnitureFitOnnxStylePipeline.collectMaskDetections(
                primaryIndex: primaryIdx,
                detections: candidates,
                protos: planes,
                protoHeight: pH,
                protoWidth: pW,
                modelSide: onnxSide
            )
            appendUniqueMaskContributors(from: lowConfidencePrimaryContributors, to: &maskSource)
            let expandedPrimary = onnxStyleExpandedPrimaryForMaskBuild(primary, onnxSide: onnxSide)
            var fusion: [FurnitureFitDetection] = [expandedPrimary]
            let contributorCandidates = maskSource.filter { FurnitureFitIoU.calculate($0, primary) < 0.999 }
            let deduplicatedContributors = limitedMaskFusionContributors(
                primary: primary,
                contributors: contributorCandidates
            )
            fusion.append(contentsOf: deduplicatedContributors)
            maskDetectionsForBuild = fusion
        }

        let maskFusionMs = 0.0
        let morphologyMs = 0.0

        let compositeDetectionsForBuild = maskDetectionsForBuild

        let primaryBufferRect = bufferRect(
            for: primary,
            imageWidth: bufW,
            imageHeight: bufH,
            scaleX: scaleX,
            scaleY: scaleY
        )
        let clippedPrimaryBufferRect = FurnitureFitGeometry.clipRectToImageBounds(
            rect: primaryBufferRect,
            imageWidth: CGFloat(bufW),
            imageHeight: CGFloat(bufH)
        )
        let primaryBx1 = Int(clippedPrimaryBufferRect.minX)
        let primaryBy1 = Int(clippedPrimaryBufferRect.minY)
        let primaryBx2 = Int(clippedPrimaryBufferRect.maxX)
        let primaryBy2 = Int(clippedPrimaryBufferRect.maxY)

        // Build the composite band from the fused detection boxes rather than the
        // thresholded small mask. This avoids cropping away weak thin parts (for
        // example the left handle of a chair) before full-res logits are sampled.
        let clipCandidates = segmentationMode == .segmentSelected ? selectedCandidates : compositeDetectionsForBuild
        let clipLeftModel = clipCandidates.map { $0.x - $0.w * 0.5 }.min() ?? (primary.x - primary.w * 0.5)
        let clipTopModel = clipCandidates.map { $0.y - $0.h * 0.5 }.min() ?? (primary.y - primary.h * 0.5)
        let clipRightModel = clipCandidates.map { $0.x + $0.w * 0.5 }.max() ?? (primary.x + primary.w * 0.5)
        let clipBottomModel = clipCandidates.map { $0.y + $0.h * 0.5 }.max() ?? (primary.y + primary.h * 0.5)
        let clipModelRect = CGRect(
            x: CGFloat(clipLeftModel),
            y: CGFloat(clipTopModel),
            width: CGFloat(max(clipRightModel - clipLeftModel, 1)),
            height: CGFloat(max(clipBottomModel - clipTopModel, 1))
        )
        let clipBufferRect = FurnitureFitGeometry.scaleBoxesFromModel(
            box: clipModelRect,
            modelShape: CGSize(width: CGFloat(onnxSide), height: CGFloat(onnxSide)),
            imageShape: CGSize(width: bufW, height: bufH),
            usesLetterbox: currentYoloUsesLetterbox,
            inputVerticallyFlipped: currentYoloInputVerticallyFlipped
        )
        let clippedCompositeBufferRect = FurnitureFitGeometry.clipRectToImageBounds(
            rect: clipBufferRect,
            imageWidth: CGFloat(bufW),
            imageHeight: CGFloat(bufH)
        )
        let bandMarginW: Float = 0  // edge band expansion disabled
        let bandMarginH: Float = 0  // edge band expansion disabled
        let cropMarginPx = FurnitureFitOnnxStylePipeline.nativeCompositeBandMarginPx
        let compBx1 = max(0, Int(floor(Float(clippedCompositeBufferRect.minX) - bandMarginW)) - cropMarginPx)
        let compBy1 = max(0, Int(floor(Float(clippedCompositeBufferRect.minY) - bandMarginH)) - cropMarginPx)
        let compBx2 = min(bufW, Int(ceil(Float(clippedCompositeBufferRect.maxX) + bandMarginW)) + cropMarginPx)
        let compBy2 = min(bufH, Int(ceil(Float(clippedCompositeBufferRect.maxY) + bandMarginH)) + cropMarginPx)

        if debugMode {
            let stage35Ms = Date().timeIntervalSince(t3) * 1000
            logDebug(
                "⏱️ ONNX-STYLE (\(stage2DebugLabel)) breakdown - " +
                "proto/decode: \(String(format: "%.2f", prototypeParseMs)) ms " +
                "(ANE-sync: \(String(format: "%.2f", lastProtoDataPointerSyncMs))), " +
                "parse/NMS/filter: \(String(format: "%.2f", detectionParseAndFilterMs)) ms " +
                "(ANE-sync: \(String(format: "%.2f", lastDetDataPointerSyncMs))), " +
                "mask fusion: \(String(format: "%.2f", maskFusionMs)) ms, " +
                "morph: \(String(format: "%.2f", morphologyMs)) ms"
            )
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
        let composedImage: CGImage? =
            currentYoloUsesLetterbox
            ? compositeLegacy1280UnionMaskCutout(
                processBuffer: processBuffer,
                planes: planes,
                protoW: pW,
                protoH: pH,
                modelSide: onnxSide,
                origW: bufW,
                origH: bufH,
                maskDetectionsForBuild: compositeDetectionsForBuild,
                x0: compBx1,
                x1: compBx2,
                y0: compBy1,
                y1: compBy2,
                primaryBx1: primaryBx1,
                primaryBy1: primaryBy1,
                primaryBx2: primaryBx2,
                primaryBy2: primaryBy2,
            )
            : nil
        let maskHasForeground = composedImage != nil

        let withDebugOverlay: CGImage? = {
            guard let base = composedImage else { return composedImage }
            if FurnitureFitOneImageDebugSupport.runEnabled, oneImageRunAwaitingSave {
                return drawCompositeContributorBboxesOnComposedImage(
                    composed: base,
                    compositeDetections: compositeDetectionsForBuild,
                    primary: primary,
                    origW: bufW,
                    origH: bufH,
                    scaleX: scaleX,
                    scaleY: scaleY
                ) ?? base
            }
            if stillImageScanModeEnabled {
                return drawOnnxStyleDebugDetectionBboxesOnComposedImage(
                    composed: base,
                    candidates: candidates,
                    primaryIndex: primaryIdx,
                    origW: bufW,
                    origH: bufH,
                    scaleX: scaleX,
                    scaleY: scaleY
                ) ?? base
            }
            guard debugMode else { return composedImage }
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
            var out: CGImage = FurnitureFitGeometry.clipCompositedImageToBounds(
                finalImage,
                imageWidth: bufW,
                imageHeight: bufH
            )
            if needsRotate {
                if let r = self.rotateCGImage90(out, clockwise: true) {
                    out = FurnitureFitGeometry.clipCompositedImageToBounds(
                        r,
                        imageWidth: bufH,
                        imageHeight: bufW
                    )
                }
            }
            if oneImageRunAwaitingSave {
                FurnitureFitOneImageDebugSupport.writeOutputPNG(
                    out,
                    filename: "alchair_furniturefit_result.png",
                    logLabel: "composite result"
                )

                if let originalFrameImage = renderOriginalFrameCGImage(
                    processBuffer: processBuffer,
                    origW: bufW,
                    origH: bufH
                ), let allDetectionsImage = drawAllDetectionBboxesOnImage(
                    baseImage: originalFrameImage,
                    candidates: candidates,
                    primaryIndex: primaryIdx,
                    origW: bufW,
                    origH: bufH,
                    scaleX: scaleX,
                    scaleY: scaleY
                ) {
                    let rotatedAllDetectionsImage: CGImage
                    if needsRotate, let rotated = rotateCGImage90(allDetectionsImage, clockwise: true) {
                        rotatedAllDetectionsImage = rotated
                    } else {
                        rotatedAllDetectionsImage = allDetectionsImage
                    }
                    FurnitureFitOneImageDebugSupport.writeOutputPNG(
                        rotatedAllDetectionsImage,
                        filename: "alchair_furniturefit_all_detections.png",
                        logLabel: "all detections"
                    )
                } else {
                    logDebug("🖼️ oneImageRun: failed to build all-detections overlay image")
                }

                oneImageRunAwaitingSave = false
                oneImageRunFinished = true
                DispatchQueue.main.async { [weak self] in
                    self?.setProgress(1.0, text: "Saved one-image result")
                    self?.finishStartupProgressIfNeeded()
                }
            }
            DispatchQueue.main.async { [weak self] in
                guard let self else { return }
                let scale = self.window?.windowScene?.screen.scale ?? self.traitCollection.displayScale
                if self.isShowingLiveVideoIdentifications {
                    self.maskImageView.image = nil
                } else {
                    self.maskImageView.image = UIImage(cgImage: out, scale: scale, orientation: .up)
                }
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

    private func appendUniqueMaskContributors(
        from additions: [FurnitureFitDetection],
        to existing: inout [FurnitureFitDetection]
    ) {
        for addition in additions {
            let alreadyPresent = existing.contains {
                $0.classIdx == addition.classIdx &&
                FurnitureFitIoU.calculate($0, addition) >= 0.95
            }
            if !alreadyPresent {
                existing.append(addition)
            }
        }
    }

    private func limitedMaskFusionContributors(
        primary: FurnitureFitDetection,
        contributors: [FurnitureFitDetection]
    ) -> [FurnitureFitDetection] {
        guard !contributors.isEmpty else { return [] }

        return contributors.enumerated()
            .map { offset, detection -> (offset: Int, detection: FurnitureFitDetection, iou: Float, centerDistance: Float, area: Float) in
                let iou = FurnitureFitIoU.calculate(primary, detection)
                let dx = primary.x - detection.x
                let dy = primary.y - detection.y
                let centerDistance = hypotf(dx, dy)
                return (offset, detection, iou, centerDistance, detection.w * detection.h)
            }
            .sorted { lhs, rhs in
                if abs(lhs.iou - rhs.iou) > 1e-6 {
                    return lhs.iou > rhs.iou
                }
                if abs(lhs.centerDistance - rhs.centerDistance) > 1e-6 {
                    return lhs.centerDistance < rhs.centerDistance
                }
                if abs(lhs.detection.confidence - rhs.detection.confidence) > 1e-6 {
                    return lhs.detection.confidence > rhs.detection.confidence
                }
                if abs(lhs.area - rhs.area) > 1e-6 {
                    return lhs.area > rhs.area
                }
                return lhs.offset < rhs.offset
            }
            .map(\.detection)
    }

    private func lowConfidenceInsidePrimaryContributors(
        detArray: MLMultiArray,
        primary: FurnitureFitDetection,
        blacklist: Set<Int>,
        protos: [Float],
        protoHeight: Int,
        protoWidth: Int,
        modelSide: Int
    ) -> [FurnitureFitDetection] {
        let lowerConfidenceThreshold = furnitureFitInsidePrimaryContributorConfidenceThreshold
        guard lowerConfidenceThreshold < confidenceThreshold else { return [] }
        guard primary.coeffs.count >= 32 else { return [] }

        let spatialSize = protoHeight * protoWidth
        let primaryBinary = FurnitureFitOnnxStylePipeline.buildBboxBinaryMask(
            detection: primary,
            protoWidth: protoWidth,
            protoHeight: protoHeight,
            modelSide: modelSide,
            spatialSize: spatialSize
        )

        let lowThresholdDetections = YoloEDetectionParser.parseDetections(
            detArray: detArray,
            confidenceThreshold: lowerConfidenceThreshold,
            classBlacklist: [],
            maximumDetections: 1000
        )
        guard !lowThresholdDetections.isEmpty else { return [] }

        if debugMode {
            let reparsedBedFamilyDetections = lowThresholdDetections.filter {
                let normalizedClassName = displayClassName($0.classIdx).lowercased()
                return normalizedClassName.contains("bed")
            }
            if !reparsedBedFamilyDetections.isEmpty {
                let reparsedBedFamilySummary = reparsedBedFamilyDetections.map {
                    "\(displayClassName($0.classIdx)) conf=\(String(format: "%.3f", $0.confidence)) ctr=(\(Int($0.x)),\(Int($0.y))) sz=\(Int($0.w))x\(Int($0.h)))"
                }.joined(separator: " | ")
                print("🛏️ ONNX-STYLE reparsed bed-family detections: \(reparsedBedFamilySummary)")
            }
        }

        let sortedDetections = lowThresholdDetections.sorted { $0.confidence > $1.confidence }
        let cappedDetections = Array(sortedDetections.prefix(300))
        let dedupedDetections = FurnitureFitNMS.applySortedByConfidence(
            detections: cappedDetections,
            iouThreshold: 0.7
        )
        let filteredDetections = dedupedDetections
            .filter { !blacklist.contains($0.classIdx) }
            .filter { $0.coeffs.count >= 32 }
        return filteredDetections
            .filter {
                let overlap = FurnitureFitOnnxStylePipeline.childOverlapsFraction(
                    childDetection: $0,
                    primaryBinary: primaryBinary,
                    protos: protos,
                    protoWidth: protoWidth,
                    protoHeight: protoHeight,
                    modelSide: modelSide
                )
                return overlap >= 0.15
            }
            .sorted { $0.confidence > $1.confidence }
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
        usesLetterbox: Bool,
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

        let geometry = nativeUpsampleGeometry(
            modelSide: modelSide,
            origW: origW,
            origH: origH,
            usesLetterbox: usesLetterbox
        )
        let protoScaleX = Float(pw) / geometry.modelInput
        let protoScaleY = Float(ph) / geometry.modelInput
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
            let modelCenterY = (Float(y) + 0.5) * geometry.imageToModelScaleY + geometry.padY
            let fy = modelCenterY * protoScaleY - 0.5
            let py0 = max(0, min(maxPy, Int(floor(fy))))
            let py1 = max(0, min(maxPy, py0 + 1))
            let ty = fy - Float(py0)

            for bx in 0..<bandW {
                let x = xStart + bx
                let modelCenterX = (Float(x) + 0.5) * geometry.imageToModelScaleX + geometry.padX
                let fx = modelCenterX * protoScaleX - 0.5
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
        upscaledPlanarMaskScratch.withUnsafeBufferPointer { maskPtr in
            guard let maskBase = maskPtr.baseAddress else { return }
            compositeCpuCameraBandAccelerated(
                outBase: outBase,
                origBase: origBase,
                maskBase: maskBase,
                origW: origW,
                camRowBytes: camRowBytes,
                bytesPerRowOut: bytesPerRowOut,
                x0: xStart,
                x1: xEnd,
                y0: yStart,
                y1: yEnd
            )
        }

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
        maskDetectionsForBuild: [FurnitureFitDetection],
        x0: Int,
        x1: Int,
        y0: Int,
        y1: Int,
        primaryBx1: Int,
        primaryBy1: Int,
        primaryBx2: Int,
        primaryBy2: Int
    ) -> CGImage? {
        guard currentYoloUsesLetterbox,
              protoW > 0,
              protoH > 0,
              modelSide > 0,
              planes.count >= 32 * protoW * protoH else { return nil }

        let xStart = max(0, min(origW, x0))
        let xEnd = max(0, min(origW, x1))
        let yStart = max(0, min(origH, y0))
        let yEnd = max(0, min(origH, y1))
        guard xStart < xEnd, yStart < yEnd else { return nil }
        let bandW = xEnd - xStart
        let bandH = yEnd - yStart

        let planeSize = protoW * protoH
        let validDetections = maskDetectionsForBuild.filter { $0.coeffs.count >= 32 }
        guard !validDetections.isEmpty else { return nil }

        var compositeBinary = [Float](repeating: 0, count: planeSize)
        var unionScratch = [Float](repeating: 0, count: planeSize)

        for detection in validDetections {
            let detectionBinary = FurnitureFitOnnxStylePipeline.buildCroppedBinaryMask(
                detection: detection,
                protos: planes,
                protoWidth: protoW,
                protoHeight: protoH,
                modelSide: modelSide,
                emitDebugLog: debugMode
            )
            vDSP_vmax(
                compositeBinary,
                1,
                detectionBinary,
                1,
                &unionScratch,
                1,
                vDSP_Length(planeSize)
            )
            swap(&compositeBinary, &unionScratch)
        }

        let protoBinaryMask = compositeBinary.map { $0 > 0 ? UInt8(255) : UInt8(0) }

        let fullMask = makeLegacy1280FullMaskFromProtoWithLetterboxFix(
            maskSmall: protoBinaryMask,
            protoW: protoW,
            protoH: protoH,
            modelInput: modelSide,
            origW: origW,
            origH: origH
        )

        var clippedFullMask = fullMask
        if bandW >= 3, bandH >= 3 {
            var bandCompact = [UInt8](repeating: 0, count: bandW * bandH)
            for by in 0..<bandH {
                let y = yStart + by
                for bx in 0..<bandW {
                    let x = xStart + bx
                    bandCompact[by * bandW + bx] = clippedFullMask[y * origW + x]
                }
            }
            let closedBandMask = morphologicalBinaryClose3x3Planar8(mask: bandCompact, width: bandW, height: bandH)
            for by in 0..<bandH {
                let y = yStart + by
                for bx in 0..<bandW {
                    let x = xStart + bx
                    clippedFullMask[y * origW + x] = closedBandMask[by * bandW + bx]
                }
            }
        }
        // Legacy 1280 union path: logits-only mask; no island hole fill (different semantics from ONNX-style composite).

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

        for y in yStart..<yEnd {
            let origRow = y * origBytesPerRow
            let outRow = y * outBytesPerRow
            let maskRow = y * origW
            for x in xStart..<xEnd {
                guard clippedFullMask[maskRow + x] != 0 else { continue }
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
        for pixelIndex in 0..<maskModel.count {
            maskModel[pixelIndex] = maskModel[pixelIndex] >= 128 ? 255 : 0
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
        for pixelIndex in 0..<fullMask.count {
            fullMask[pixelIndex] = fullMask[pixelIndex] >= 128 ? 255 : 0
        }
        return fullMask
    }

    // MARK: - process_mask_native (GPU bilinear upsample + threshold)

    /// Matches Metal `BilinearUpsampleParams` layout (8×UInt32 + 6×Float).
    private struct BilinearUpsampleParams {
        var protoW:  UInt32
        var protoH:  UInt32
        var origW:   UInt32
        var origH:   UInt32
        var xStart:  UInt32
        var yStart:  UInt32
        var bandW:   UInt32
        var bandH:   UInt32
        var modelInput: Float
        var imageToModelScaleX: Float
        var imageToModelScaleY: Float
        var padX: Float
        var padY: Float
        var logitThreshold: Float
    }

    private func nativeUpsampleGeometry(
        modelSide: Int,
        origW: Int,
        origH: Int,
        usesLetterbox: Bool
    ) -> (modelInput: Float, imageToModelScaleX: Float, imageToModelScaleY: Float, padX: Float, padY: Float) {
        let modelInput = Float(max(modelSide, 1))
        let imageWidth = Float(max(origW, 1))
        let imageHeight = Float(max(origH, 1))
        if usesLetterbox {
            let gain = min(modelInput / imageWidth, modelInput / imageHeight)
            let padX = (modelInput - imageWidth * gain) * 0.5
            let padY = (modelInput - imageHeight * gain) * 0.5
            return (modelInput, gain, gain, padX, padY)
        } else {
            return (modelInput, modelInput / imageWidth, modelInput / imageHeight, 0, 0)
        }
    }

    /// GPU-accelerated bilinear upsample of full-field logits + threshold >0,
    /// returning a UInt8 band mask.  Falls back to CPU if Metal is unavailable.
    private func gpuBilinearUpsampleAndThreshold(
        logits: [Float],
        protoW: Int,
        protoH: Int,
        modelSide: Int,
        origW: Int,
        origH: Int,
        xStart: Int,
        yStart: Int,
        bandW: Int,
        bandH: Int,
        usesLetterbox: Bool,
        logitThreshold: Float
    ) -> [UInt8]? {
        guard let device = metalDevice,
              let queue = metalCommandQueue,
              let pipeline = bilinearUpsamplePipeline else { return nil }
        let geometry = nativeUpsampleGeometry(
            modelSide: modelSide,
            origW: origW,
            origH: origH,
            usesLetterbox: usesLetterbox
        )

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

        _ = logits.withUnsafeBufferPointer { ptr in
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
            modelInput: geometry.modelInput,
            imageToModelScaleX: geometry.imageToModelScaleX,
            imageToModelScaleY: geometry.imageToModelScaleY,
            padX: geometry.padX,
            padY: geometry.padY,
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

    private func debugUpsampledBandMaskFromLogits(
        maskLogits: [Float],
        protoW: Int,
        protoH: Int,
        modelSide: Int,
        origW: Int,
        origH: Int,
        xStart: Int,
        yStart: Int,
        bandW: Int,
        bandH: Int,
        usesLetterbox: Bool
    ) -> [UInt8]? {
        guard protoW > 0,
              protoH > 0,
              bandW > 0,
              bandH > 0,
              maskLogits.count >= protoW * protoH else { return nil }

        let upsampleThreshold = FurnitureFitOnnxStylePipeline.nativeCompositeUpsampleLogitThreshold()
        if let gpuBandMask = gpuBilinearUpsampleAndThreshold(
            logits: maskLogits,
            protoW: protoW,
            protoH: protoH,
            modelSide: modelSide,
            origW: origW,
            origH: origH,
            xStart: xStart,
            yStart: yStart,
            bandW: bandW,
            bandH: bandH,
            usesLetterbox: usesLetterbox,
            logitThreshold: upsampleThreshold
        ) {
            if furnitureFitNativeMaskMorphologicalClose, bandW >= 3, bandH >= 3 {
                return morphologicalBinaryClose3x3Planar8(mask: gpuBandMask, width: bandW, height: bandH)
            }
            return gpuBandMask
        }

        let geometry = nativeUpsampleGeometry(
            modelSide: modelSide,
            origW: origW,
            origH: origH,
            usesLetterbox: usesLetterbox
        )
        let protoScaleX = Float(protoW) / geometry.modelInput
        let protoScaleY = Float(protoH) / geometry.modelInput
        let maximumProtoX = protoW - 1
        let maximumProtoY = protoH - 1
        var bandMask = [UInt8](repeating: 0, count: bandW * bandH)

        for bandY in 0..<bandH {
            let imageY = yStart + bandY
            let modelCenterY = (Float(imageY) + 0.5) * geometry.imageToModelScaleY + geometry.padY
            let protoYFloat = modelCenterY * protoScaleY - 0.5
            let protoY0 = max(0, min(maximumProtoY, Int(floor(protoYFloat))))
            let protoY1 = max(0, min(maximumProtoY, protoY0 + 1))
            let yBlend = protoYFloat - Float(protoY0)

            for bandX in 0..<bandW {
                let imageX = xStart + bandX
                let modelCenterX = (Float(imageX) + 0.5) * geometry.imageToModelScaleX + geometry.padX
                let protoXFloat = modelCenterX * protoScaleX - 0.5
                let protoX0 = max(0, min(maximumProtoX, Int(floor(protoXFloat))))
                let protoX1 = max(0, min(maximumProtoX, protoX0 + 1))
                let xBlend = protoXFloat - Float(protoX0)

                let topLeftIndex = protoY0 * protoW + protoX0
                let topRightIndex = protoY0 * protoW + protoX1
                let bottomLeftIndex = protoY1 * protoW + protoX0
                let bottomRightIndex = protoY1 * protoW + protoX1

                let topLeftLogit = maskLogits[topLeftIndex]
                let topRightLogit = maskLogits[topRightIndex]
                let bottomLeftLogit = maskLogits[bottomLeftIndex]
                let bottomRightLogit = maskLogits[bottomRightIndex]

                let blendedLogit =
                    topLeftLogit * (1 - xBlend) * (1 - yBlend) +
                    topRightLogit * xBlend * (1 - yBlend) +
                    bottomLeftLogit * (1 - xBlend) * yBlend +
                    bottomRightLogit * xBlend * yBlend

                bandMask[bandY * bandW + bandX] = blendedLogit > upsampleThreshold ? 255 : 0
            }
        }

        if furnitureFitNativeMaskMorphologicalClose, bandW >= 3, bandH >= 3 {
            bandMask = morphologicalBinaryClose3x3Planar8(mask: bandMask, width: bandW, height: bandH)
        }
        return bandMask
    }

    private func liveDebugMaskColor(
        detectionIndex: Int,
        isPrimary: Bool
    ) -> (red: UInt8, green: UInt8, blue: UInt8, alpha: UInt8) {
        if isPrimary {
            return (255, 64, 64, 170)
        }

        let palette: [(UInt8, UInt8, UInt8, UInt8)] = [
            (72, 220, 255, 120),
            (112, 255, 140, 120),
            (255, 196, 72, 120),
            (196, 136, 255, 120),
            (255, 120, 184, 120)
        ]
        let color = palette[detectionIndex % palette.count]
        return (color.0, color.1, color.2, color.3)
    }

    private func blendLiveDebugMaskPixel(
        outBase: UnsafeMutablePointer<UInt8>,
        bytesPerRow: Int,
        x: Int,
        y: Int,
        color: (red: UInt8, green: UInt8, blue: UInt8, alpha: UInt8)
    ) {
        let pixelOffset = y * bytesPerRow + x * 4
        let sourceAlpha = Float(color.alpha) / 255
        let destinationAlpha = Float(outBase[pixelOffset + 3]) / 255
        let destinationKeep = 1 - sourceAlpha

        let sourceRedPremultiplied = Float(color.red) * sourceAlpha
        let sourceGreenPremultiplied = Float(color.green) * sourceAlpha
        let sourceBluePremultiplied = Float(color.blue) * sourceAlpha

        let blendedRed = sourceRedPremultiplied + Float(outBase[pixelOffset + 0]) * destinationKeep
        let blendedGreen = sourceGreenPremultiplied + Float(outBase[pixelOffset + 1]) * destinationKeep
        let blendedBlue = sourceBluePremultiplied + Float(outBase[pixelOffset + 2]) * destinationKeep
        let blendedAlpha = sourceAlpha + destinationAlpha * destinationKeep

        outBase[pixelOffset + 0] = UInt8(max(0, min(255, Int(blendedRed.rounded()))))
        outBase[pixelOffset + 1] = UInt8(max(0, min(255, Int(blendedGreen.rounded()))))
        outBase[pixelOffset + 2] = UInt8(max(0, min(255, Int(blendedBlue.rounded()))))
        outBase[pixelOffset + 3] = UInt8(max(0, min(255, Int((blendedAlpha * 255).rounded()))))
    }

    private func renderLiveDebugMaskOverlayImage(
        baseImage: CGImage?,
        candidates: [FurnitureFitDetection],
        primaryIndex: Int?,
        fusedMaskLogits: [Float]?,
        fusedMaskDetections: [FurnitureFitDetection],
        planes: [Float],
        protoW: Int,
        protoH: Int,
        modelSide: Int,
        origW: Int,
        origH: Int,
        scaleX: Float,
        scaleY: Float
    ) -> CGImage? {
        guard !candidates.isEmpty,
              protoW > 0,
              protoH > 0,
              modelSide > 0,
              origW > 0,
              origH > 0 else { return nil }

        guard let context = CGContext(
            data: nil,
            width: origW,
            height: origH,
            bitsPerComponent: 8,
            bytesPerRow: 0,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ), let rawData = context.data else { return nil }

        let bytesPerRow = cgBitmapAllocatedBytesPerRow(context)
        let outBase = rawData.assumingMemoryBound(to: UInt8.self)
        if let baseImage {
            context.draw(baseImage, in: CGRect(x: 0, y: 0, width: origW, height: origH))
        } else {
            fillCompositeBufferTransparent(
                outBase: outBase,
                height: origH,
                bytesPerRowOut: bytesPerRow
            )
        }

        let orderedIndices = candidates.indices.filter { $0 != primaryIndex }
            + (primaryIndex.map { [$0] } ?? [])
        var paintedPixelCount = 0

        if let fusedMaskLogits,
           !fusedMaskDetections.isEmpty,
           fusedMaskLogits.count >= protoW * protoH {
            let fusedRects = fusedMaskDetections.map {
                bufferRect(
                    for: $0,
                    imageWidth: origW,
                    imageHeight: origH,
                    scaleX: scaleX,
                    scaleY: scaleY
                )
            }
            let fusedMinX = max(0, Int(floor(fusedRects.map(\.minX).min() ?? 0)))
            let fusedMinY = max(0, Int(floor(fusedRects.map(\.minY).min() ?? 0)))
            let fusedMaxX = min(origW, Int(ceil(fusedRects.map(\.maxX).max() ?? 0)))
            let fusedMaxY = min(origH, Int(ceil(fusedRects.map(\.maxY).max() ?? 0)))
            let fusedBandWidth = fusedMaxX - fusedMinX
            let fusedBandHeight = fusedMaxY - fusedMinY

            if fusedBandWidth > 0,
               fusedBandHeight > 0,
               let fusedBandMask = debugUpsampledBandMaskFromLogits(
                    maskLogits: fusedMaskLogits,
                    protoW: protoW,
                    protoH: protoH,
                    modelSide: modelSide,
                    origW: origW,
                    origH: origH,
                    xStart: fusedMinX,
                    yStart: fusedMinY,
                    bandW: fusedBandWidth,
                    bandH: fusedBandHeight,
                    usesLetterbox: currentYoloUsesLetterbox
               ) {
                let fusedFillColor: (red: UInt8, green: UInt8, blue: UInt8, alpha: UInt8) = (255, 32, 32, 215)
                for bandY in 0..<fusedBandHeight {
                    let imageY = fusedMinY + bandY
                    let rowOffset = bandY * fusedBandWidth
                    for bandX in 0..<fusedBandWidth where fusedBandMask[rowOffset + bandX] != 0 {
                        let isBoundaryPixel =
                            bandX == 0 ||
                            bandY == 0 ||
                            bandX == fusedBandWidth - 1 ||
                            bandY == fusedBandHeight - 1 ||
                            fusedBandMask[rowOffset + max(0, bandX - 1)] == 0 ||
                            fusedBandMask[rowOffset + min(fusedBandWidth - 1, bandX + 1)] == 0 ||
                            fusedBandMask[max(0, bandY - 1) * fusedBandWidth + bandX] == 0 ||
                            fusedBandMask[min(fusedBandHeight - 1, bandY + 1) * fusedBandWidth + bandX] == 0
                        let pixelColor: (red: UInt8, green: UInt8, blue: UInt8, alpha: UInt8) =
                            isBoundaryPixel ? (255, 255, 255, 255) : fusedFillColor
                        blendLiveDebugMaskPixel(
                            outBase: outBase,
                            bytesPerRow: bytesPerRow,
                            x: fusedMinX + bandX,
                            y: imageY,
                            color: pixelColor
                        )
                        paintedPixelCount += 1
                    }
                }
            }
        }

        let shouldDrawPerDetectionMasks = fusedMaskLogits == nil || fusedMaskDetections.isEmpty
        if shouldDrawPerDetectionMasks {
        for detectionIndex in orderedIndices {
            let detection = candidates[detectionIndex]
            let detectionBufferRect = bufferRect(
                for: detection,
                imageWidth: origW,
                imageHeight: origH,
                scaleX: scaleX,
                scaleY: scaleY
            )
            let xStart = max(0, Int(floor(detectionBufferRect.minX)))
            let yStart = max(0, Int(floor(detectionBufferRect.minY)))
            let xEnd = min(origW, Int(ceil(detectionBufferRect.maxX)))
            let yEnd = min(origH, Int(ceil(detectionBufferRect.maxY)))
            let bandW = xEnd - xStart
            let bandH = yEnd - yStart
            guard bandW > 0, bandH > 0 else { continue }

            let perDetectionBuilt = FurnitureFitOnnxStylePipeline.buildFullFieldLogitMask(
                planes: planes,
                protoW: protoW,
                protoH: protoH,
                detections: [detection],
                modelSide: modelSide
            )
            var perDetectionLogits = perDetectionBuilt.logits
            if currentYoloInputVerticallyFlipped {
                FurnitureFitOnnxStylePipeline.flipProtoFloatGridVertically(&perDetectionLogits, protoW: protoW, protoH: protoH)
            }
            guard let bandMask = debugUpsampledBandMaskFromLogits(
                maskLogits: perDetectionLogits,
                protoW: protoW,
                protoH: protoH,
                modelSide: modelSide,
                origW: origW,
                origH: origH,
                xStart: xStart,
                yStart: yStart,
                bandW: bandW,
                bandH: bandH,
                usesLetterbox: currentYoloUsesLetterbox
            ) else { continue }

            let maskColor = liveDebugMaskColor(
                detectionIndex: detectionIndex,
                isPrimary: detectionIndex == primaryIndex
            )
            for bandY in 0..<bandH {
                let imageY = yStart + bandY
                let bandRowOffset = bandY * bandW
                for bandX in 0..<bandW where bandMask[bandRowOffset + bandX] != 0 {
                    let isBoundaryPixel =
                        bandX == 0 ||
                        bandY == 0 ||
                        bandX == bandW - 1 ||
                        bandY == bandH - 1 ||
                        bandMask[bandRowOffset + max(0, bandX - 1)] == 0 ||
                        bandMask[bandRowOffset + min(bandW - 1, bandX + 1)] == 0 ||
                        bandMask[max(0, bandY - 1) * bandW + bandX] == 0 ||
                        bandMask[min(bandH - 1, bandY + 1) * bandW + bandX] == 0
                    let pixelColor: (red: UInt8, green: UInt8, blue: UInt8, alpha: UInt8) =
                        isBoundaryPixel
                        ? (255, 255, 255, detectionIndex == primaryIndex ? 255 : 220)
                        : maskColor
                    blendLiveDebugMaskPixel(
                        outBase: outBase,
                        bytesPerRow: bytesPerRow,
                        x: xStart + bandX,
                        y: imageY,
                        color: pixelColor
                    )
                    paintedPixelCount += 1
                }
            }
        }
        }

        if debugMode && paintedPixelCount == 0 {
            logDebug("⚠️ Live debug mask overlay produced zero painted pixels")
        }
        return context.makeImage()
    }

    /// `process_mask_native` composite: full-field matmul → GPU bilinear upsample →
    /// crop + threshold → optional morph close → CPU camera composite.
    private func compositeGpuNativeMaskCutout(
        processBuffer: CVPixelBuffer,
        maskProto: [UInt8],
        maskLogits: [Float],
        protoW: Int,
        protoH: Int,
        modelSide: Int,
        origW: Int,
        origH: Int,
        x0: Int,
        x1: Int,
        y0: Int,
        y1: Int,
        usesLetterbox: Bool,
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
            modelSide: modelSide,
            origW: origW,
            origH: origH,
            xStart: xStart,
            yStart: yStart,
            bandW: bandW,
            bandH: bandH,
            usesLetterbox: usesLetterbox,
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

    /// Draws bounding boxes for every detection whose mask logits are fused into the composite (expanded primary + overlapping contributors).
    private func drawCompositeContributorBboxesOnComposedImage(
        composed: CGImage,
        compositeDetections: [FurnitureFitDetection],
        primary: FurnitureFitDetection,
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

        let paragraphStyle = NSMutableParagraphStyle()
        paragraphStyle.lineBreakMode = .byTruncatingTail

        for d in compositeDetections {
            let r = bufferRect(for: d)
            let cgRect = CGRect(
                x: CGFloat(r.bx1),
                y: CGFloat(origH - r.by1 - r.bh),
                width: CGFloat(r.bw),
                height: CGFloat(r.bh)
            )
            let isPrimaryChip = FurnitureFitIoU.calculate(d, primary) >= 0.92
            ctx.setLineWidth(isPrimaryChip ? 4.0 : 2.5)
            ctx.setStrokeColor((isPrimaryChip ? UIColor.systemRed : UIColor.systemOrange).cgColor)
            ctx.stroke(cgRect)

            let labelText = "\(displayClassName(d.classIdx)) \(String(format: "%.2f", d.confidence))"
            let font = UIFont.systemFont(ofSize: 12, weight: isPrimaryChip ? .semibold : .regular)
            let attributes: [NSAttributedString.Key: Any] = [
                .font: font,
                .foregroundColor: UIColor.white,
                .paragraphStyle: paragraphStyle
            ]
            let attributedString = NSAttributedString(string: labelText, attributes: attributes)
            let line = CTLineCreateWithAttributedString(attributedString)
            let textBounds = CTLineGetBoundsWithOptions(line, .useOpticalBounds)
            let labelX = CGFloat(r.bx1)
            let labelY = CGFloat(origH - r.by1 + 4)
            let textBackgroundRect = CGRect(
                x: labelX - 2,
                y: labelY - textBounds.height - 2,
                width: min(CGFloat(origW) - labelX + 2, textBounds.width + 8),
                height: textBounds.height + 4
            )
            ctx.setFillColor(UIColor.black.withAlphaComponent(0.72).cgColor)
            ctx.fill(textBackgroundRect)
            ctx.saveGState()
            ctx.textMatrix = .identity
            ctx.translateBy(x: labelX, y: labelY - textBounds.height)
            ctx.setFillColor(UIColor.white.cgColor)
            CTLineDraw(line, ctx)
            ctx.restoreGState()
        }

        return ctx.makeImage()
    }

    private func renderOriginalFrameCGImage(
        processBuffer: CVPixelBuffer,
        origW: Int,
        origH: Int
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
        guard let origBase = CVPixelBufferGetBaseAddress(processBuffer)?.assumingMemoryBound(to: UInt8.self),
              let outData = ctx.data else { return nil }

        let camRowBytes = CVPixelBufferGetBytesPerRow(processBuffer)
        let outBase = outData.assumingMemoryBound(to: UInt8.self)

        for y in 0..<origH {
            let outRowPtr = outBase.advanced(by: y * bytesPerRowOut)
            let origRowPtr = origBase.advanced(by: y * camRowBytes)
            for x in 0..<origW {
                let outOff = x * 4
                let camOff = x * 4
                outRowPtr[outOff + 0] = origRowPtr[camOff + 2]
                outRowPtr[outOff + 1] = origRowPtr[camOff + 1]
                outRowPtr[outOff + 2] = origRowPtr[camOff + 0]
                outRowPtr[outOff + 3] = 255
            }
        }

        return ctx.makeImage()
    }

    private func drawAllDetectionBboxesOnImage(
        baseImage: CGImage,
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

        ctx.draw(baseImage, in: CGRect(x: 0, y: 0, width: origW, height: origH))

        func bufferRect(for detection: FurnitureFitDetection) -> CGRect {
            let scaledRect = self.bufferRect(
                for: detection,
                imageWidth: origW,
                imageHeight: origH,
                scaleX: scaleX,
                scaleY: scaleY
            )
            return CGRect(
                x: scaledRect.minX,
                y: CGFloat(origH) - scaledRect.maxY,
                width: max(1, scaledRect.width),
                height: max(1, scaledRect.height)
            )
        }

        let paragraphStyle = NSMutableParagraphStyle()
        paragraphStyle.lineBreakMode = .byTruncatingTail

        for (index, detection) in candidates.enumerated() {
            let rect = bufferRect(for: detection)
            let isPrimary = index == primaryIndex
            let strokeColor = isPrimary ? UIColor.systemRed : UIColor.systemCyan
            ctx.setStrokeColor(strokeColor.cgColor)
            ctx.setLineWidth(isPrimary ? 4.0 : 2.2)
            ctx.stroke(rect)

            let labelText = "\(displayClassName(detection.classIdx)) [\(detection.classIdx)] \(String(format: "%.2f", detection.confidence))"
            let attributes: [NSAttributedString.Key: Any] = [
                .font: UIFont.systemFont(ofSize: isPrimary ? 13 : 12, weight: isPrimary ? .semibold : .medium),
                .foregroundColor: UIColor.white,
                .paragraphStyle: paragraphStyle
            ]
            let attributedString = NSAttributedString(string: labelText, attributes: attributes)
            let line = CTLineCreateWithAttributedString(attributedString)
            let textBounds = CTLineGetBoundsWithOptions(line, .useOpticalBounds)
            let labelRect = CGRect(
                x: rect.minX,
                y: max(0, rect.minY - textBounds.height - 6),
                width: min(CGFloat(origW) - rect.minX, textBounds.width + 10),
                height: textBounds.height + 6
            )
            ctx.setFillColor(UIColor.black.withAlphaComponent(0.72).cgColor)
            ctx.fill(labelRect)
            ctx.saveGState()
            ctx.textMatrix = .identity
            ctx.translateBy(x: labelRect.minX + 5, y: labelRect.minY + 3)
            ctx.setFillColor(UIColor.white.cgColor)
            CTLineDraw(line, ctx)
            ctx.restoreGState()
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
            let isPrimary = index == primaryIndex
            ctx.setLineWidth(isPrimary ? 4.0 : 2.0)
            ctx.setStrokeColor((isPrimary ? UIColor.red : UIColor.cyan).cgColor)
            ctx.stroke(cgRect)

            let plainName = classNames[d.classIdx] ?? "unknown"
            let confidence = String(format: "%.2f", d.confidence)
            let labelText = "\(plainName) [\(d.classIdx)] (\(confidence))"
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
            ctx.setFillColor((isPrimary ? UIColor.systemRed : UIColor.black).withAlphaComponent(0.7).cgColor)
            ctx.fill(textBackgroundRect)
            ctx.saveGState()
            ctx.textMatrix = .identity
            ctx.translateBy(x: labelX, y: labelY - textBounds.height)
            ctx.setFillColor(UIColor.white.cgColor)
            CTLineDraw(line, ctx)
            ctx.restoreGState()
        }

        return ctx.makeImage()
    }

    private func squarePixelBufferAttributes() -> CFDictionary {
        let attrs: [CFString: Any] = [
            kCVPixelBufferIOSurfacePropertiesKey: [:],
            kCVPixelBufferMetalCompatibilityKey: true,
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true
        ]
        return attrs as CFDictionary
    }

    /// Stretch to fill `size`×`size` (matches Android ONNX `createScaledBitmap` preprocessing).
    private func resizeStretchToSquare(_ src: CVPixelBuffer, size: Int) -> CVPixelBuffer? {
        CVPixelBufferLockBaseAddress(src, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(src, .readOnly) }

        let srcW = CVPixelBufferGetWidth(src)
        let srcH = CVPixelBufferGetHeight(src)

        if cachedSquareSize != size || cachedSquareBuffer == nil {
            var newBuffer: CVPixelBuffer?
            guard CVPixelBufferCreate(nil, size, size, kCVPixelFormatType_32BGRA, squarePixelBufferAttributes(), &newBuffer) == kCVReturnSuccess,
                  let buf = newBuffer else { return nil }
            cachedSquareBuffer = buf
            cachedSquareSize = size
        }

        guard let dst = cachedSquareBuffer else { return nil }
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
        guard vImageScale_ARGB8888(&srcBuffer, &dstBuffer, nil, vImage_Flags(kvImageNoFlags)) == kvImageNoError else {
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

        if cachedSquareSize != size || cachedSquareBuffer == nil {
            var newBuffer: CVPixelBuffer?
            guard CVPixelBufferCreate(nil, size, size, kCVPixelFormatType_32BGRA, squarePixelBufferAttributes(), &newBuffer) == kCVReturnSuccess,
                  let buf = newBuffer else { return nil }
            cachedSquareBuffer = buf
            cachedSquareSize = size
        }

        guard let dst = cachedSquareBuffer else { return nil }
        CVPixelBufferLockBaseAddress(dst, [])
        defer { CVPixelBufferUnlockBaseAddress(dst, []) }

        guard let srcBase = CVPixelBufferGetBaseAddress(src),
              let dstBase = CVPixelBufferGetBaseAddress(dst) else { return nil }

        let scale = min(Float(size) / Float(srcW), Float(size) / Float(srcH))
        let scaledWidth = max(1, min(size, Int(round(Float(srcW) * scale))))
        let scaledHeight = max(1, min(size, Int(round(Float(srcH) * scale))))
        let padX = (size - scaledWidth) / 2
        let padY = (size - scaledHeight) / 2
        let dstRowBytes = CVPixelBufferGetBytesPerRow(dst)
        YoloUltralyticsLetterboxFill.fillOpaqueBGRA114LetterboxStrips(
            dstBase: dstBase,
            width: size,
            height: size,
            bytesPerRow: dstRowBytes,
            padX: padX,
            padY: padY,
            scaledWidth: scaledWidth,
            scaledHeight: scaledHeight
        )

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
        guard vImageScale_ARGB8888(&srcBuffer, &dstRegion, nil, vImage_Flags(kvImageNoFlags)) == kvImageNoError else {
            return nil
        }
        return dst
    }

    private func processFrameInner(_ pixelBuffer: CVPixelBuffer, arDepthSnapshot: FurnitureFitARDepthSnapshot? = nil) {
        if debugMode { logDebug("▶️ processFrameInner entered (mlModel=\(mlModel == nil ? "NIL" : "set"))") }
        guard let model = mlModel else {
            resetProcessingFlag()
            return
        }
        let frameStart = Date()
        if debugMode { logMemory("FRAME START") }
        classBlacklist.loadBlacklistOnce(debugMode: debugMode) { logDebug($0) }
        processFrameOnnxStyleCoreML(
            processBuffer: pixelBuffer,
            model: model,
            arDepthSnapshot: arDepthSnapshot,
            frameStart: frameStart
        )
    }


    // MARK: - Parse Prototypes (FIXED: Accelerate for Float16)
    // Reuses instance-level buffers and uses Accelerate for FP16 → FP32 conversion.
    private func parsePrototypes(_ proto: MLMultiArray) -> (planes: [Float], count: Int, height: Int, width: Int, shape: [Int], strides: [Int], channelAxis: Int)? {
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

        // Split the ANE→CPU materialization cost out of the parse timer. First access to
        // `.dataPointer` triggers CoreML to DMA the output buffer from ANE memory and
        // (if needed) convert Float16→Float32. This can dominate "proto/decode" time.
        let dataPointerSyncStart = Date()
        _ = proto.dataPointer.load(as: UInt8.self)
        lastProtoDataPointerSyncMs = Date().timeIntervalSince(dataPointerSyncStart) * 1000
        if protoPlanes.count != count * planeSize {
            protoPlanes = [Float](repeating: 0, count: count * planeSize)
        }
        let channelFirstContiguous = cIdx == 0 &&
            normalizedStrides.count == 3 &&
            normalizedStrides[0] == planeSize &&
            normalizedStrides[1] == w &&
            normalizedStrides[2] == 1
        let channelLastContiguous = cIdx == 2 &&
            normalizedStrides.count == 3 &&
            normalizedStrides[0] == w * count &&
            normalizedStrides[1] == count &&
            normalizedStrides[2] == 1

        if debugMode, !didLogProtoLayoutDiagnostic {
            didLogProtoLayoutDiagnostic = true
            let dataTypeRaw = proto.dataType.rawValue
            let dataTypeName: String
            if proto.dataType == .float32 {
                dataTypeName = "float32"
            } else if proto.dataType == .float16 {
                dataTypeName = "float16"
            } else if proto.dataType == .float64 {
                dataTypeName = "float64"
            } else if proto.dataType == .int32 {
                dataTypeName = "int32"
            } else {
                dataTypeName = "unknown"
            }
            let chosenPath: String
            if channelFirstContiguous {
                chosenPath = proto.dataType == .float32 ? "FAST:channelFirst+memcpy" : "FAST:channelFirst+vImageF16"
            } else if channelLastContiguous, proto.dataType == .float32 {
                chosenPath = "FAST:channelLast+blasScopy"
            } else {
                chosenPath = "SLOW:fallback(protoRawFloats+transpose)"
            }
            logDebug("🔬 parsePrototypes layout dtype=\(dataTypeName)(\(dataTypeRaw)) shape=\(shape) strides=\(normalizedStrides) cIdx=\(cIdx) channelFirstContig=\(channelFirstContiguous) channelLastContig=\(channelLastContiguous) path=\(chosenPath)")
        }

        if channelFirstContiguous {
            if proto.dataType == .float32 {
                protoPlanes.withUnsafeMutableBufferPointer { dst in
                    guard let dstBase = dst.baseAddress else { return }
                    memcpy(dstBase, proto.dataPointer, count * planeSize * MemoryLayout<Float>.size)
                }
            } else {
                let src16 = proto.dataPointer.bindMemory(to: UInt16.self, capacity: proto.count)
                protoPlanes.withUnsafeMutableBufferPointer { dst in
                    guard let dstBase = dst.baseAddress else { return }
                    var srcBuf = vImage_Buffer(
                        data: UnsafeMutableRawPointer(mutating: src16),
                        height: 1,
                        width: vImagePixelCount(count * planeSize),
                        rowBytes: count * planeSize * MemoryLayout<UInt16>.size
                    )
                    var dstBuf = vImage_Buffer(
                        data: dstBase,
                        height: 1,
                        width: vImagePixelCount(count * planeSize),
                        rowBytes: count * planeSize * MemoryLayout<Float>.size
                    )
                    vImageConvert_Planar16FtoPlanarF(&srcBuf, &dstBuf, vImage_Flags(kvImageNoFlags))
                }
            }

            return (protoPlanes, count, h, w, shape, normalizedStrides, cIdx)
        }

        if channelLastContiguous, proto.dataType == .float32 {
            let srcBase = proto.dataPointer.bindMemory(to: Float.self, capacity: total)
            protoPlanes.withUnsafeMutableBufferPointer { dst in
                guard let dstBase = dst.baseAddress else { return }
                for channel in 0..<count {
                    FurnitureFitBlas.scopy(
                        FurnitureFitBlas.Dim(planeSize),
                        srcBase.advanced(by: channel),
                        FurnitureFitBlas.Dim(count),
                        dstBase.advanced(by: channel * planeSize),
                        1
                    )
                }
            }

            return (protoPlanes, count, h, w, shape, normalizedStrides, cIdx)
        }

        if protoRawFloats.count != total {
            protoRawFloats = [Float](repeating: 0, count: total)
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
                        FurnitureFitBlas.scopy(
                            FurnitureFitBlas.Dim(planeSize),
                            srcBase.advanced(by: k),
                            FurnitureFitBlas.Dim(count),
                            dstBase.advanced(by: k * planeSize),
                            1
                        )
                    }
                }
            }
        }

        return (protoPlanes, count, h, w, shape, normalizedStrides, cIdx)
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
                return vImageMax_Planar8(
                    &s,
                    &d,
                    nil,
                    0,
                    0,
                    3,
                    3,
                    vImage_Flags(kvImageNoFlags)
                )
            }
        }
        guard errMax == kvImageNoError else {
            return morphologicalBinaryClose3x3(mask: mask, width: width, height: height)
        }
        let errMin: vImage_Error = dilated.withUnsafeBufferPointer { srcPtr in
            closed.withUnsafeMutableBufferPointer { dPtr in
                var s = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: srcPtr.baseAddress!), height: vImagePixelCount(height), width: vImagePixelCount(width), rowBytes: width)
                var d = vImage_Buffer(data: dPtr.baseAddress!, height: vImagePixelCount(height), width: vImagePixelCount(width), rowBytes: width)
                return vImageMin_Planar8(
                    &s,
                    &d,
                    nil,
                    0,
                    0,
                    3,
                    3,
                    vImage_Flags(kvImageNoFlags)
                )
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
        var scale: Float = 1.0 / 255.0
        let src = baseAddress.assumingMemoryBound(to: UInt8.self)
        let rowLength = vDSP_Length(width)

        for y in 0..<height {
            let rowStart = src.advanced(by: y * bytesPerRow)
            let rRow = rPtr.advanced(by: y * width)
            let gRow = gPtr.advanced(by: y * width)
            let bRow = bPtr.advanced(by: y * width)

            vDSP_vfltu8(rowStart.advanced(by: 2), 4, rRow, 1, rowLength)
            vDSP_vsmul(rRow, 1, &scale, rRow, 1, rowLength)

            vDSP_vfltu8(rowStart.advanced(by: 1), 4, gRow, 1, rowLength)
            vDSP_vsmul(gRow, 1, &scale, gRow, 1, rowLength)

            vDSP_vfltu8(rowStart.advanced(by: 0), 4, bRow, 1, rowLength)
            vDSP_vsmul(bRow, 1, &scale, bRow, 1, rowLength)
        }

        return array
    }

    // MARK: - Rotation Helpers

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
        let resolvedAssistedHeight = resolvedAssistedFurnitureHeightMeters(
            estimatedHeightMeters: arHeightMeters
        )

        if arAssistedSizingEnabled,
           let resolvedAssistedHeight,
           resolvedAssistedHeight.isFinite,
           resolvedAssistedHeight > 0 {
            finalARHeight = resolvedAssistedHeight
            finalHeight = max(heightMeters, resolvedAssistedHeight)
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

    private func normalizedManualFurnitureHeightOverrideMeters() -> Float? {
        guard let value = manualFurnitureHeightOverrideMeters,
              value.isFinite,
              value > 0.05 else { return nil }
        return value
    }

    private func resolvedAssistedFurnitureHeightMeters(estimatedHeightMeters: Float?) -> Float? {
        if let manualOverrideHeight = normalizedManualFurnitureHeightOverrideMeters() {
            return manualOverrideHeight
        }

        guard let estimatedHeightMeters,
              estimatedHeightMeters.isFinite,
              estimatedHeightMeters > 0.05 else { return nil }
        return arAssistedSizingEnabled
            ? max(estimatedHeightMeters, Self.minimumARFurnitureHeightMeters)
            : estimatedHeightMeters
    }

    private func normalizedARFurnitureHeightMeters() -> Float? {
        resolvedAssistedFurnitureHeightMeters(estimatedHeightMeters: lastAREstimatedHeightMeters)
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
        let tapHitPadding: CGFloat = isShowingLiveVideoIdentifications ? 22 : 10
        let candidateContains = candidateBboxesInView.contains {
            $0.insetBy(dx: -tapHitPadding, dy: -tapHitPadding).contains(pointInMask)
        }

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
