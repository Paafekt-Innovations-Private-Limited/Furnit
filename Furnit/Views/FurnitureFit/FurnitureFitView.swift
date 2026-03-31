// FurnitureFitView.swift
// Single-stage: detect → optional multi-candidate (5a–5b) → union mask → clip outside primary bbox (15d) → cutout
// With timing at every stage

import SwiftUI
import UIKit
import CoreML
import Accelerate
import AVFoundation
import ARKit
import SceneKit
import CoreText
import MetalKit

// BLAS helpers using C wrapper (BLASWrapper.m) to avoid Swift deprecation warnings
fileprivate typealias BLASInt = Int32

@inline(__always)
fileprivate func blas_scopy(_ n: BLASInt, _ x: UnsafePointer<Float>, _ incx: BLASInt, _ y: UnsafeMutablePointer<Float>, _ incy: BLASInt) {
    BlasScopy(n, x, incx, y, incy)
}

@inline(__always)
fileprivate func blas_sgemv_rowmajor(m: BLASInt, n: BLASInt, alpha: Float, A: UnsafePointer<Float>, lda: BLASInt, x: UnsafePointer<Float>, incx: BLASInt, beta: Float, y: UnsafeMutablePointer<Float>, incy: BLASInt) {
    BlasSgemv(true, false, m, n, alpha, A, lda, x, incx, beta, y, incy)
}

/// SGEMV with transpose: y = A^T * x
/// A is (m x n), A^T is (n x m), x is (m,), y is (n,)
@inline(__always)
fileprivate func blas_sgemv_rowmajor_trans(m: BLASInt, n: BLASInt, alpha: Float, A: UnsafePointer<Float>, lda: BLASInt, x: UnsafePointer<Float>, incx: BLASInt, beta: Float, y: UnsafeMutablePointer<Float>, incy: BLASInt) {
    BlasSgemv(true, true, m, n, alpha, A, lda, x, incx, beta, y, incy)
}

@inline(__always)
fileprivate func blas_sgemm_rowmajor(m: BLASInt, n: BLASInt, k: BLASInt, alpha: Float, A: UnsafePointer<Float>, lda: BLASInt, B: UnsafePointer<Float>, ldb: BLASInt, beta: Float, C: UnsafeMutablePointer<Float>, ldc: BLASInt) {
    BlasSgemm(true, false, false, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
}

/// SGEMM with A transposed: C = A^T * B
/// A is (k x m), A^T is (m x k), B is (k x n), C is (m x n)
@inline(__always)
fileprivate func blas_sgemm_rowmajor_transA(m: BLASInt, n: BLASInt, k: BLASInt, alpha: Float, A: UnsafePointer<Float>, lda: BLASInt, B: UnsafePointer<Float>, ldb: BLASInt, beta: Float, C: UnsafeMutablePointer<Float>, ldc: BLASInt) {
    BlasSgemm(true, true, false, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
}


// MARK: - Metal Mask Logic (GPU)
// Computes maskSmall (prototype resolution) on GPU: max over detections of dot(coeffs, prototypes) per pixel.
// Output is UInt8 mask (0 or 255) using the same thresholding logic as CPU: maxLogit > 0 => 255.
final class MetalMaskLogic {
    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let pipelineMaxMask: MTLComputePipelineState

    // MARK: - Cached buffers for reuse (prevents allocation per frame)
    private var cachedPlanesBuf: MTLBuffer?
    private var cachedCoeffBuf: MTLBuffer?
    private var cachedOutBuf: MTLBuffer?
    private var cachedPlanesCapacity: Int = 0
    private var cachedCoeffCapacity: Int = 0
    private var cachedOutCapacity: Int = 0

    init(device: MTLDevice) {
        self.device = device
        self.queue = device.makeCommandQueue()!
        let library = device.makeDefaultLibrary()!
        self.pipelineMaxMask = try! device.makeComputePipelineState(function: library.makeFunction(name: "sp_maxMaskFromPrototypes")!)
    }

    /// Get or create a buffer with at least the required capacity
    private func getOrCreateBuffer(cached: inout MTLBuffer?, capacity: inout Int, required: Int) -> MTLBuffer? {
        if let buf = cached, capacity >= required {
            return buf
        }
        // Allocate with some headroom to reduce reallocations
        let newCapacity = max(required, capacity * 2, 1024)
        guard let newBuf = device.makeBuffer(length: newCapacity, options: .storageModeShared) else {
            return nil
        }
        cached = newBuf
        capacity = newCapacity
        return newBuf
    }

    func buildMaskSmall(planes: [Float], coeffs: [Float], planeSize: Int, detCount: Int) -> [UInt8] {
        precondition(planes.count == 32 * planeSize, "planes size mismatch")
        precondition(coeffs.count == detCount * 32, "coeffs size mismatch")

        let planesBytes = planes.count * MemoryLayout<Float>.size
        let coeffBytes = coeffs.count * MemoryLayout<Float>.size
        let outBytes = planeSize * MemoryLayout<UInt8>.size

        // Reuse buffers instead of allocating new ones each frame
        guard let planesBuf = getOrCreateBuffer(cached: &cachedPlanesBuf, capacity: &cachedPlanesCapacity, required: planesBytes),
              let coeffBuf = getOrCreateBuffer(cached: &cachedCoeffBuf, capacity: &cachedCoeffCapacity, required: coeffBytes),
              let outBuf = getOrCreateBuffer(cached: &cachedOutBuf, capacity: &cachedOutCapacity, required: outBytes) else {
            return [UInt8](repeating: 0, count: planeSize)
        }

        // Copy data into reused buffers
        memcpy(planesBuf.contents(), planes, planesBytes)
        memcpy(coeffBuf.contents(), coeffs, coeffBytes)

        // Wrap Metal work in autoreleasepool to ensure command buffers are released
        return autoreleasepool {
            guard let cmd = queue.makeCommandBuffer(),
                  let enc = cmd.makeComputeCommandEncoder() else {
                return [UInt8](repeating: 0, count: planeSize)
            }

            enc.setComputePipelineState(pipelineMaxMask)
            enc.setBuffer(planesBuf, offset: 0, index: 0)
            enc.setBuffer(coeffBuf, offset: 0, index: 1)
            enc.setBuffer(outBuf, offset: 0, index: 2)

            var ps = UInt32(planeSize)
            var dc = UInt32(detCount)
            enc.setBytes(&ps, length: MemoryLayout<UInt32>.size, index: 3)
            enc.setBytes(&dc, length: MemoryLayout<UInt32>.size, index: 4)

            let tgW = pipelineMaxMask.threadExecutionWidth
            let threadsPerTG = MTLSize(width: tgW, height: 1, depth: 1)
            let threads = MTLSize(width: planeSize, height: 1, depth: 1)
            enc.dispatchThreads(threads, threadsPerThreadgroup: threadsPerTG)
            enc.endEncoding()

            let gpuStart = Date()
            cmd.commit()
            cmd.waitUntilCompleted()
            let gpuEnd = Date()
            let gpuMs = gpuEnd.timeIntervalSince(gpuStart) * 1000
            logDebug("⏱️ Metal GPU wait (maskSmall): \(String(format: "%.2f", gpuMs)) ms")

            let ptr = outBuf.contents().bindMemory(to: UInt8.self, capacity: planeSize)
            return Array(UnsafeBufferPointer(start: ptr, count: planeSize))
        }
    }
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
    /// Horizontal bbox extent in meters (room scale).
    let widthMeters: Float
    /// Vertical bbox extent in meters (room scale) — use this for calibration and UI labels.
    let heightMeters: Float
    /// Reserved; always `nil` when emitted from FurnitureFit. AR metric height is used only for in-view overlay scaling (scene depth is wrong on glass / reflective).
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

    // Callback for reporting estimated furniture size (room-based + optional AR height, in meters)
    var onFurnitureSizeEstimated: ((FurnitureSizeEstimate) -> Void)?

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
    private var furnitureFitCameraStartupInitiated = false
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

    private let minCombinedOverlayScale: CGFloat = 0.3
    private let maxCombinedOverlayScale: CGFloat = 3.0

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
    private var cachedSquareBuffer: CVPixelBuffer?
    private var cachedSquareSize: Int = 0
    /// Separate from letterbox buffer — ONNX/Android path stretches the full frame to the model square.
    private var cachedStretchBuffer: CVPixelBuffer?
    private var cachedStretchSize: Int = 0
    private var cachedMLArray: MLMultiArray?
    private var cachedMLArraySize: Int = 0

    /// Reused scratch for CPU fallback compositing (vDSP + BLAS); grows with frame width, never shrinks.
    private var compositeCpuScratchFloats: [Float] = []
    /// Proto mask upscaled to full frame (`origW*origH`) via vImage — one SIMD resize then a cheap composite scan.
    private var upscaledPlanarMaskScratch: [UInt8] = []
    // GPU mask builder (optional)
    private lazy var metalMaskLogic: MetalMaskLogic? = {
        guard let d = metalDevice else { return nil }
        return MetalMaskLogic(device: d)
    }()
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
    /// Set when `ARSession.pause()` was called after copying a frame so segmentation can run without ARKit queueing more `ARFrame`s (avoids delegate retention warnings).
    private var arPausedForSegmentation = false
    private let frameLock = NSLock() // Protects lastProcessTime, isProcessing, arPausedForSegmentation

    /// Serial queue for debounced AR-assisted measurement (depth → height → overlay scale). Keeps work off the main thread until the debounce fires; height can lag ~1s while the mask keeps up.
    private let measurementQueue = DispatchQueue(label: "com.furnit.furniturefit.measurement", qos: .utility)
    private let assistedMeasurementDebounceSeconds: TimeInterval = 0.85
    private struct PendingAssistedMeasurement {
        let primaryClassIdx: Int
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

    /// Thread-safe reset of isProcessing flag; resumes AR session if it was paused for segmentation.
    private func resetProcessingFlag() {
        frameLock.lock()
        isProcessing = false
        if preferImmediateNextInference {
            lastProcessTime = .distantPast
            preferImmediateNextInference = false
        }
        let resumeAR = arPausedForSegmentation
        arPausedForSegmentation = false
        frameLock.unlock()
        if resumeAR {
            DispatchQueue.main.async { [weak self] in
                guard let self, let cfg = self.arSession.configuration else { return }
                // Full AR-as-camera path, or hybrid depth companion after `pause()` between frames.
                guard self.isUsingARCameraPath || self.isARDepthCompanionSessionRunning else { return }
                self.arSession.run(cfg, options: [])
            }
        }
    }

    private func invalidatePendingAssistedMeasurement() {
        assistedMeasurementDebounceWorkItem?.cancel()
        assistedMeasurementScheduleToken &+= 1
        pendingMeasurementLock.lock()
        pendingAssistedMeasurement = nil
        pendingMeasurementLock.unlock()
    }

    private func scheduleDebouncedAssistedMeasurement(
        primaryClassIdx: Int,
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
            progressContainer.widthAnchor.constraint(equalToConstant: 280),
            progressContainer.heightAnchor.constraint(equalToConstant: 50),

            // Progress bar inside container
            progressView.leadingAnchor.constraint(equalTo: progressContainer.leadingAnchor),
            progressView.trailingAnchor.constraint(equalTo: progressContainer.trailingAnchor),
            progressView.bottomAnchor.constraint(equalTo: progressContainer.bottomAnchor),

            // Label above progress bar
            progressLabel.centerXAnchor.constraint(equalTo: progressContainer.centerXAnchor),
            progressLabel.bottomAnchor.constraint(equalTo: progressView.topAnchor, constant: -6),
            progressLabel.heightAnchor.constraint(equalToConstant: 24),
            progressLabel.widthAnchor.constraint(greaterThanOrEqualToConstant: 80)
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

    /// Updates room-based auto scale from standard furniture dimensions vs detection (model input pixels).
    /// Does not set `transform`; call `applyCurrentOverlayScaleTransform()` after AR update (or alone).
    private func updateAutoScaleFromRoom(classIdx: Int, detectedWidth: CGFloat, detectedHeight: CGFloat) {
        // In \"generic fit\" mode we no longer apply any per-class catalog-based scaling.
        // Room-based meters already come from SHARP / calibration; visual scaling is driven
        // by AR (when available) and user pinch.
        autoScaleFromRoom = 1.0
    }

    /// Combined overlay transform: room × (AR metric when valid, else 1×) × pinch (clamped).
    private func applyCurrentOverlayScaleTransform() {
        let arOn = hasARKitAssistedSizingPayload && arAssistedScaleValid
        let assistedScale: CGFloat = arOn ? autoScaleFromAR : 1.0
        let product = autoScaleFromRoom * assistedScale * userPinchScale
        let clamped = min(max(product, minCombinedOverlayScale), maxCombinedOverlayScale)
        maskImageView.transform = CGAffineTransform(scaleX: clamped, y: clamped)

        if debugMode {
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
                logDebug(
                    "📐 [OVERLAY] assist=\(assistedLabel) room=\(String(format: "%.3f", autoScaleFromRoom)) ar=\(String(format: "%.3f", autoScaleFromAR)) pinch=\(String(format: "%.3f", userPinchScale)) → combined=\(String(format: "%.3f", clamped)) wantAR=\(wantAR) arValid=\(arAssistedScaleValid) arMisses=\(arAssistedConsecutiveMisses)"
                )
            }
        }
    }

    private func updateAssistedOverlayScale(
        primaryClassIdx: Int,
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
        }

        let wantAR = hasARKitAssistedSizingPayload && !userLockedAssistedOverlayScale

        if wantAR, arDepthSnapshot != nil || arSession.currentFrame != nil {
            let nx = bboxCenterImageX / CGFloat(max(1, imageWidth))
            let ny = bboxCenterImageY / CGFloat(max(1, imageHeight))

            var distM: Float?
            var fy: Float = 1
            var distSource = "none"
            var hasDepthMap = false
            var bgraIsRotatedFromCaptured = false
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
                hasDepthMap = snap.depthMap != nil
                bgraIsRotatedFromCaptured = snap.bgraIsRotatedFromCaptured
                distM = FurnitureFitARSupport.depthMeters(snapshot: snap, normalizedBgraNX: nx, normalizedBgraNY: ny)
                fy = snap.focalLengthY
                distSource = distM != nil ? "depth_snapshot" : "depth_snapshot_miss"
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
                    distM = FurnitureFitARSupport.distanceToHorizontalPlaneMeters(
                        session: arSession,
                        frame: frame,
                        screenPoint: sp,
                        in: bounds
                    )
                    distSource = distM != nil ? "plane_raycast" : "no_metric_distance"
                } else {
                    distSource = "no_metric_distance"
                }
                let cap = frame.capturedImage
                let cw = CVPixelBufferGetWidth(cap)
                let ch = CVPixelBufferGetHeight(cap)
                let needsPortrait = (lockedOrientation == .portrait || lockedOrientation == .square)
                let bgraRotated = needsPortrait && cw > ch
                bgraIsRotatedFromCaptured = bgraRotated
                fy = FurnitureFitARSupport.focalLengthYForProcessedBGRA(
                    camera: frame.camera,
                    bgraHeight: imageHeight,
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
                lastAREstimatedHeightMeters = estH
                // Use a neutral reference height for AR scaling; do not depend on any
                // per-class hard-coded \"standard\" dimensions.
                let stdH: Float = 0.85
                if let raw = FurnitureFitARSupport.overlayScaleFromMetricHeights(
                    standardHeightMeters: stdH,
                    estimatedHeightMeters: estH
                ) {
                    let target = CGFloat(raw)
                    if snapArOverlayScaleAfterPrimaryChange {
                        // One shot after switching primary detection — avoids sluggish ramp from 1× while panning between pieces.
                        let clamped = min(max(target, 0.25), 4.0)
                        smoothedArOverlayScale = clamped
                        snapArOverlayScaleAfterPrimaryChange = false
                    } else {
                        // AR-assisted scale can jitter due to depth noise; clamp per-frame change and EMA smooth.
                        let base = smoothedArOverlayScale
                        let maxStep: CGFloat = 0.08
                        let delta = max(-maxStep, min(maxStep, target - base))
                        let clampedTarget = base + delta
                        let alpha: CGFloat = 0.16
                        smoothedArOverlayScale = base * (1.0 - alpha) + clampedTarget * alpha
                    }
                    autoScaleFromAR = smoothedArOverlayScale
                    arAssistedScaleValid = true
                    arAssistedConsecutiveMisses = 0
                    let nowLog = Date().timeIntervalSince1970
                    if nowLog - lastFurnitureFitARFrameLogAt >= furnitureFitARLogInterval {
                        lastFurnitureFitARFrameLogAt = nowLog
                        logFurnitureFitAR(
                            "platform=ios phase=frame tracking=\(trackingName) distSource=\(distSource) dist_m=\(String(format: "%.4f", d)) fy_px=\(String(format: "%.2f", fy)) bboxH_px=\(String(format: "%.2f", bboxHeightImagePx)) imageWH=(\(imageWidth)x\(imageHeight)) norm_bgra=(\(String(format: "%.4f", nx)),\(String(format: "%.4f", ny))) estH_m=\(String(format: "%.4f", estH)) stdH_m=\(String(format: "%.2f", stdH)) rawScale=\(String(format: "%.4f", raw)) smoothedScale=\(String(format: "%.4f", Double(autoScaleFromAR))) hasDepthMap=\(hasDepthMap) bgraRotated=\(bgraIsRotatedFromCaptured) snap=\(arDepthSnapshot != nil)"
                        )
                    }
                    if debugMode {
                        logDebug("📐 [AR] dist=\(String(format: "%.2f", d))m estH=\(String(format: "%.2f", estH))m stdH=\(String(format: "%.2f", stdH))m scale=\(String(format: "%.3f", raw)) smoothed=\(String(format: "%.3f", autoScaleFromAR))")
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
        userPinchScale = 1.0
        userLockedAssistedOverlayScale = false
        lastOverlayPrimaryClassIdx = -1
        smoothedTightPrimaryBbox = nil
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
        // Avoid blocking sync call if model hasn't changed
        if model === mlModel { return }
        detectionQueue.sync { self.mlModel = model }
        // Log model outputs for debugging
        if let model = model {
            let inputNames = model.modelDescription.inputDescriptionsByName.keys.joined(separator: ", ")
            let outputNames = model.modelDescription.outputDescriptionsByName.keys
            logDebug("🧠 [FurnitureFit] Model set - inputs: [\(inputNames)], outputs: [\(outputNames.joined(separator: ", "))]")
        }
        // Note: loadBlacklist() is called in startIfNeeded() to avoid repeated calls from updateUIView
    }

    func startIfNeeded() {
        // SwiftUI calls this on many `updateUIView` passes; only run setup once until `stop()`.
        if captureSession.isRunning { return }
        if furnitureFitCameraStartupInitiated { return }
        furnitureFitCameraStartupInitiated = true

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
        asyncDepthSampler.stop()
        furnitureFitCameraStartupInitiated = false
        NotificationCenter.default.removeObserver(self, name: UIDevice.orientationDidChangeNotification, object: nil)
        UIDevice.current.endGeneratingDeviceOrientationNotifications()

        DispatchQueue.main.async {
            self.isUsingARCameraPath = false
            self.isARDepthCompanionSessionRunning = false
        }
        DispatchQueue.global(qos: .userInitiated).async {
            if self.captureSession.isRunning {
                self.captureSession.stopRunning()
            }
        }
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
        // Single back-camera consumer: `AVCaptureSession` only. Do not run a parallel `ARSession` for world tracking here
        // (Fig -17281 / failed capture). `FurnitureFitAsyncDepthSampler` remains for a possible AR-only video path later.
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            startClassicCameraPathIfNeeded()
            if debugMode {
                logDebug("📷 [FurnitureFit] AVCaptureSession (classic path, no parallel ARSession)")
            }
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { granted in
                guard granted else { return }
                self.startClassicCameraPathIfNeeded()
            }
        default:
            break
        }
    }

    /// AVCapture only; pauses container `ARSession`.
    private func startClassicCameraPathIfNeeded() {
        isUsingARCameraPath = false
        isARDepthCompanionSessionRunning = false
        frameLock.lock()
        arPausedForSegmentation = false
        frameLock.unlock()
        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            self.arSession.pause()
            self.setupCamera()
            DispatchQueue.global(qos: .userInitiated).async { [weak self] in
                guard let self else { return }
                if !self.captureSession.isRunning {
                    self.captureSession.startRunning()
                }
            }
        }
        if debugMode { logDebug("📷 [FurnitureFit] AVCaptureSession (classic camera path)") }
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
        let depthSnapshot: FurnitureFitARDepthSnapshot?
        if isARDepthCompanionSessionRunning, let frame = arSession.currentFrame {
            let pbW = CVPixelBufferGetWidth(pixelBuffer)
            let pbH = CVPixelBufferGetHeight(pixelBuffer)
            depthSnapshot = FurnitureFitARSupport.makeDepthSnapshot(
                frame: frame,
                bgraWidth: pbW,
                bgraHeight: pbH,
                lockedOrientation: lockedOrientation
            )
        } else {
            depthSnapshot = nil
        }
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

    /// Android-style pipeline (stretch, NMS, primary, mask) using the same Core ML ``mlModel`` as the default letterbox path.
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

    /// Same primary index as the default letterbox Core ML path (STAGE 4: conf^1.5 × areaNorm^0.8 × center^0.5 with min conf/area gates).
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

            let score = confTerm * areaTerm * centerTerm
            if score > maxScore {
                maxScore = score
                bestIdx = i
            }
        }
        return bestIdx
    }

    /// Shared postprocess after stretch + inference: **same detection list, NMS, and primary as letterbox Core ML**; mask fusion remains Android-style bbox-limited proto + collectMaskDetections.
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

        // Detection list + NMS + primary: match default letterbox path (`confidenceThreshold`, `blacklist.json` → `clsToIgnore`, IoU 0.5, STAGE 4 scoring).
        let rawDetections = YoloEDetectionParser.parseDetections(
            detArray: detArray,
            confidenceThreshold: confidenceThreshold,
            classBlacklist: clsToIgnore
        )
        YoloEDetectionParser.releaseF16Scratch()

        if rawDetections.isEmpty {
            if debugMode { logDebug("⚠️ ONNX-STYLE (\(stage2DebugLabel)): no detections after parse (Core ML thresholds + blacklist.json)") }
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
        if isUsingARCameraPath {
            scheduleDebouncedAssistedMeasurement(
                primaryClassIdx: primary.classIdx,
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
                classIdx: primary.classIdx,
                detectedWidth: CGFloat(primary.w * scaleX),
                detectedHeight: CGFloat(primary.h * scaleY)
            )
        }

        setProgress(0.92, text: "Compositing…")
        let compStart = Date()
        // Nearest map proto → frame (Android `composeNearestProtoMaskCutoutArgb`). Bilinear upscale of a 0/255 mask
        // was smearing edges and could make most of the primary band fully opaque (A=255) while the cutout failed visually.
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

        if let finalImage = withDebugOverlay {
            let needsRotate = isLandscape && !self.isUsingARCameraPath && self.lockedOrientation != .landscape
            DispatchQueue.main.async {
                var out = finalImage
                if needsRotate {
                    if let r = self.rotateCGImage90(out, clockwise: true) { out = r }
                }
                self.maskImageView.image = UIImage(cgImage: out)
            }
        }

        if maskSmall.contains(where: { $0 > 0 }) {
            finishFirstDetectionIfNeeded()
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
                return vImageScale_Planar8(&srcBuf, &dstBuf, nil, vImage_Flags(kvImageHighQualityResampling))
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

        let rgbDim: Float = 0.92
        let inv255: Float = 1.0 / 255.0
        // Smoothstep remaps soft mask slopes → stabler alpha (less intermittent “blur” at boundary).
        let edgeLo: Float = 0.36
        let invEdgeWidth: Float = 1.0 / max(0.18, 0.62 - edgeLo)

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

        for (_, d) in candidates.enumerated() {
            let r = bufferRect(for: d)
            let cgRect = CGRect(x: r.bx1, y: origH - r.by1 - r.bh, width: r.bw, height: r.bh)
            ctx.setLineWidth(2.0)
            ctx.setStrokeColor(UIColor.cyan.cgColor)
            ctx.stroke(cgRect)

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

        let onnxStylePreferred = QualitySettings.furnitureFitOnnxStyleEnabled()
        if onnxStylePreferred, let model = mlModel {
            processFrameOnnxStyleCoreML(processBuffer: pixelBuffer, model: model, arDepthSnapshot: arDepthSnapshot, frameStart: frameStart)
            return
        }

        guard let model = mlModel else {
            resetProcessingFlag()
            return
        }

        // Camera delivers frames via dynamic videoRotationAngle
        let processBuffer = pixelBuffer

        // CRITICAL: Derive isLandscape from actual buffer dimensions, not device orientation
        // Device orientation can be unreliable (flat on table, certain angles, etc.)
        // Buffer dimensions are always correct - what the model actually sees
        let bufW = CVPixelBufferGetWidth(processBuffer)
        let bufH = CVPixelBufferGetHeight(processBuffer)
        let isLandscape = bufW > bufH

        // Keep device orientation for final display rotation (portrait UI)
        let deviceOrientation = currentDeviceOrientation

        if debugMode {
            logDebug("\n⏱️ ═══════════════════════════════════════════")
            logDebug("⏱️ FRAME START @ \(String(format: "%.3f", frameStart.timeIntervalSince1970))")
            logDebug("⏱️ Buffer: \(bufW)x\(bufH), isLandscape: \(isLandscape), device: \(deviceOrientation.rawValue)")
            logDebug("⏱️ ═══════════════════════════════════════════")
        }

        // Same resolution logic as still-image / wall measurement (`YoloEImageInference`); 26L PF → typically 640.
        let modelInputSize = YoloEImageInference.modelInputSize(for: model)
        if debugMode {
            logDebug("📐 Model input side: \(modelInputSize) (from Core ML image constraint / 26L default)")
        }

        // STAGE 1: Preprocess (resize + input prep)
        let t1 = Date()
        setProgress(0.15, text: "Preprocessing…")

        guard let sq = resizeToSquare(processBuffer, size: modelInputSize) else {
            if debugMode { logDebug("❌ STAGE 1 FAILED: Resize to square") }
            resetProcessingFlag()
            return
        }
        let resizeGain = sq.gain
        let padX = sq.padX        // Integer - exact pixel offset
        let padY = sq.padY        // Integer - exact pixel offset
        let contentW = sq.newW    // Integer - exact content width in model space
        let contentH = sq.newH    // Integer - exact content height in model space

        // Debug landscape letterbox parameters
        if debugMode && isLandscape {
            logDebug("🔶 LANDSCAPE: buf=\(bufW)x\(bufH), gain=\(resizeGain), pad=(\(padX),\(padY)), content=\(contentW)x\(contentH)")
        }

        // Check if model expects Image or MultiArray input
        let inputDesc = model.modelDescription.inputDescriptionsByName["image"]
        let expectsImage = inputDesc?.type == .image

        let inputProvider: MLFeatureProvider
        if expectsImage {
            guard let imageValue = MLFeatureValue(pixelBuffer: sq.buffer) as MLFeatureValue?,
                  let provider = try? MLDictionaryFeatureProvider(dictionary: ["image": imageValue]) else {
                if debugMode { logDebug("❌ STAGE 1 FAILED: Input prep") }
                resetProcessingFlag()
                return
            }
            inputProvider = provider
        } else {
            guard let inputArray = pixelBufferToMLMultiArray(sq.buffer),
                  let provider = try? MLDictionaryFeatureProvider(dictionary: ["image": inputArray]) else {
                if debugMode { logDebug("❌ STAGE 1 FAILED: Input prep") }
                resetProcessingFlag()
                return
            }
            inputProvider = provider
        }

        let t1End = Date()
        if debugMode {
            logDebug("⏱️ STAGE 1 - Preprocess: \(String(format: "%.2f", t1End.timeIntervalSince(t1) * 1000)) ms")
        }

        // STAGE 2: Core ML prediction (same stage name as `main` branch timing logs).
        let t2 = Date()
        setProgress(0.40, text: "Running model…")

        guard let output = try? model.prediction(from: inputProvider) else {
            if debugMode { logDebug("❌ STAGE 2 FAILED: Model inference") }
            resetProcessingFlag()
            return
        }

        let t2End = Date()
        if debugMode {
            logDebug("⏱️ STAGE 2 - Inference: \(String(format: "%.2f", t2End.timeIntervalSince(t2) * 1000)) ms")
            logMemory("AFTER INFERENCE")
        }

        // STAGE 3: Parse outputs (tensors + detections + prototypes)
        let t3 = Date()

        let detArray: MLMultiArray
        let protoArray: MLMultiArray

        if let pair = YoloEDetectionParser.extractDetectionAndProto(from: output) {
            detArray = pair.det
            protoArray = pair.proto
            if debugMode {
                logDebug("📦 Using Core ML output tensors (\(pair.detName)/\(pair.protoName))")
            }
        } else {
            if debugMode {
                logDebug("❌ STAGE 4 FAILED: Missing output tensors")
                // Log available outputs for debugging
                let availableOutputs = output.featureNames.joined(separator: ", ")
                logDebug("   Available outputs: \(availableOutputs)")
            }
            resetProcessingFlag()
            return
        }
        
        // Debug: print actual tensor shapes
        let dim2 = detArray.shape[2].intValue
        if debugMode {
            logDebug("   detArray shape: \(detArray.shape.map { $0.intValue })")
            logDebug("   protoArray shape: \(protoArray.shape.map { $0.intValue })")
        }

        // Detect tensor format:
        // - Old format: [1, 144, 8400] where 144 = features (4+numClasses+32), 8400 = anchors
        // - New format: [1, 300, 38] where 300 = detections, 38 = features per detection
        // Heuristic: if dim2 is small (< 100), it's the new format with features per detection
        let isNewFormat = dim2 < 100

        if debugMode {
            logDebug("   Format detected: \(isNewFormat ? "NEW (detections×features)" : "OLD (features×anchors)")")
        }

        if debugMode, detArray.dataType == .float16 {
            logDebug("   det dtype: float16 (Core ML mlprogram) — parser converts on read")
        }

        setProgress(0.55, text: "Parsing detections…")

        let allDets = YoloEDetectionParser.parseDetections(
            detArray: detArray,
            confidenceThreshold: confidenceThreshold,
            classBlacklist: clsToIgnore
        )
        YoloEDetectionParser.releaseF16Scratch()

        
        if debugMode {
            logDebug("   raw detections: \(allDets.count)")
            for (i, d) in allDets.enumerated() {
                logDebug("   [\(i)] \(className(d.classIdx)) (id:\(d.classIdx)) conf=\(String(format: "%.2f", d.confidence)) box=(\(Int(d.x)),\(Int(d.y))) size=\(Int(d.w))x\(Int(d.h))")
            }
        }

        if allDets.isEmpty {
            if debugMode { logDebug("⚠️ No detections found") }
            DispatchQueue.main.async {
                self.maskImageView.image = nil
                self.resetOverlayScalesForEmptyMask()
            }
            resetProcessingFlag()
            return
        }

        // STAGE 3b: Apply lightweight NMS to remove duplicate detections of the same object.
        let candidates: [UnionDet]
        if allDets.count > 1 {
            let sorted = allDets.sorted { $0.confidence > $1.confidence }
            let capped = Array(sorted.prefix(100))  // Cap for performance; typical frames have ≪100 dets.
            candidates = FurnitureFitNMS.apply(detections: capped, iouThreshold: 0.5)
        } else {
            candidates = allDets
        }
        if debugMode {
            logDebug("⏱️ STAGE 3b - candidates: \(candidates.count) (after NMS, from \(allDets.count) raw)")
        }

        setProgress(0.60, text: "Parsing prototypes…")

        guard let protoInfo = parsePrototypes(protoArray) else {
            if debugMode { logDebug("❌ STAGE 3 FAILED: Parse prototypes") }
            resetProcessingFlag()
            return
        }
        let planes = protoInfo.planes
        let pH = protoInfo.height
        let pW = protoInfo.width
        let planeSize = pH * pW

        let t3End = Date()
        if debugMode {
            logDebug("⏱️ STAGE 3 - Parse outputs: \(String(format: "%.2f", t3End.timeIntervalSince(t3) * 1000)) ms (\(allDets.count) dets, \(pW)x\(pH) protos)")
        }

        // STAGE 4: Select primary (confidence × area × center scoring) — shared with ONNX-style path.
        let t4 = Date()
        setProgress(0.65, text: "Selecting primary…")

        guard let primaryIdx = selectPrimaryIndexCoreFlow(candidates: candidates, modelSide: modelInputSize) else {
            if debugMode { logDebug("   ⚠️ No valid primary candidate") }
            DispatchQueue.main.async {
                self.maskImageView.image = nil
                self.resetOverlayScalesForEmptyMask()
            }
            resetProcessingFlag()
            return
        }

        let primary = candidates[primaryIdx]
        let t4End = Date()
        if debugMode {
            logDebug("⏱️ STAGE 4 - Select primary: \(String(format: "%.2f", t4End.timeIntervalSince(t4) * 1000)) ms")
            logDebug("   🎯 PRIMARY[\(primaryIdx)]: \(className(primary.classIdx)) conf=\(String(format: "%.2f", primary.confidence)) size=\(Int(primary.w))x\(Int(primary.h))")
        }

        // Estimate real-world furniture size: same scale on both axes to avoid aspect-ratio distortion.
        // Camera buffer aspect (bufW/bufH) often differs from room aspect (roomW/roomH); using
        // different metersPerPixelX/Y would stretch the bbox and make calibration wrong.
        let imgWidthPx = primary.w / resizeGain
        let imgHeightPx = primary.h / resizeGain

        let metersPerPixelX = roomWidthMeters / Float(bufW)
        let metersPerPixelY = roomHeightMeters / Float(bufH)
        let metersPerPixel = min(metersPerPixelX, metersPerPixelY)

        var estimatedWidth = imgWidthPx * metersPerPixel
        var estimatedHeight = imgHeightPx * metersPerPixel

        if estimatedWidth > estimatedHeight {
            swap(&estimatedWidth, &estimatedHeight)
        }

        var arHeightMeters: Float?
        if hasARKitAssistedSizingPayload,
           let arH = lastAREstimatedHeightMeters,
           arH > 0 {
            arHeightMeters = arH
        }

        if debugMode {
            logDebug("📏 [Sizing] Room: \(String(format: "%.2f", roomWidthMeters))×\(String(format: "%.2f", roomHeightMeters))m, Buffer: \(bufW)×\(bufH)px, m/px: \(String(format: "%.6f", metersPerPixel))")
            if let arH = arHeightMeters {
                logDebug("📏 [Sizing] Furniture (room-based): \(Int(imgWidthPx))×\(Int(imgHeightPx))px → \(String(format: "%.2f", estimatedWidth))×\(String(format: "%.2f", estimatedHeight))m, AR-height=\(String(format: "%.2f", arH))m")
            } else {
                logDebug("📏 [Sizing] Furniture: \(Int(imgWidthPx))×\(Int(imgHeightPx))px → \(String(format: "%.2f", estimatedWidth))×\(String(format: "%.2f", estimatedHeight))m")
            }
        }

        DispatchQueue.main.async {
            let estimate = FurnitureSizeEstimate(
                widthMeters: estimatedWidth,
                heightMeters: estimatedHeight,
                arHeightMeters: arHeightMeters
            )
            self.onFurnitureSizeEstimated?(estimate)
        }

        // STAGE 5: Filter candidates (5a prune, 5b bbox dedupe).
        // When disabled, composite uses primary detection only (same as early single-object behavior).
        var kept2: [UnionDet]
        if !useMultiCandidateStage5 {
            kept2 = [primary]
            if debugMode {
                logDebug("⏱️ STAGE 5 SKIPPED (useMultiCandidateStage5=false), kept2=primary only")
            }
        } else {

        let t5 = Date()

        // Primary bbox edges for encompasses check
        let origPX1 = primary.x - primary.w * 0.5
        let origPY1 = primary.y - primary.h * 0.5
        let origPX2 = primary.x + primary.w * 0.5
        let origPY2 = primary.y + primary.h * 0.5

        // 10% margin commented out: use primary bbox only for intersect check
        // let primaryMarginW = primary.w * 0.10
        // let primaryMarginH = primary.h * 0.10
        // let expandedPX1 = origPX1 - primaryMarginW
        // let expandedPY1 = origPY1 - primaryMarginH
        // let expandedPX2 = origPX2 + primaryMarginW
        // let expandedPY2 = origPY2 + primaryMarginH

        // Helper: check if candidate bbox fully encompasses primary bbox
        // (indicates background/room detection that wraps the primary object)
        let encompassTolerance: Float = 2.0  // pixels tolerance for floating-point noise
        func bboxEncompassesPrimary(candX1: Float, candY1: Float, candX2: Float, candY2: Float) -> Bool {
            return candX1 <= origPX1 + encompassTolerance &&
                   candY1 <= origPY1 + encompassTolerance &&
                   candX2 >= origPX2 - encompassTolerance &&
                   candY2 >= origPY2 - encompassTolerance
        }

        var prunedCandidates: [(idx: Int, det: UnionDet)] = []
        let minCandConf: Float = 0.1

        for (i, d) in candidates.enumerated() {
            if i == primaryIdx { continue }

            // Confidence filter
            if d.confidence < minCandConf { continue }

            // Candidate bbox corners
            let dx1 = d.x - d.w * 0.5
            let dy1 = d.y - d.h * 0.5
            let dx2 = d.x + d.w * 0.5
            let dy2 = d.y + d.h * 0.5

            // Skip if candidate bbox fully encompasses primary (background/room detection)
            if bboxEncompassesPrimary(candX1: dx1, candY1: dy1, candX2: dx2, candY2: dy2) {
                if debugMode {
                    logDebug("   ⏭️ [\(i)]: \(className(d.classIdx)) skipped - bbox encompasses primary")
                }
                continue
            }

            // Intersect check: primary bbox only (10% margin commented out)
            let px1 = origPX1
            let py1 = origPY1
            let px2 = origPX2
            let py2 = origPY2
            let intersects = !(dx2 < px1 || dx1 > px2 || dy2 < py1 || dy1 > py2)
            if !intersects {
                if debugMode {
                    logDebug("   ⏭️ [\(i)]: \(className(d.classIdx)) skipped - bbox doesn't intersect primary")
                }
                continue
            }

            prunedCandidates.append((i, d))
        }

        let t5a = Date()
        if debugMode {
            let pruneMs = String(format: "%.2f", t5a.timeIntervalSince(t5) * 1000)
            logDebug("⏱️ STAGE 5a - Prune: \(pruneMs) ms, \(candidates.count - 1) → \(prunedCandidates.count) candidates")
        }

        // STAGE 5b: Bbox-only dedupe to get a small set of non-duplicate candidates

        // Bbox IoU helper for cheap deduplication gate
        func bboxIoU(_ a: UnionDet, _ b: UnionDet) -> Float {
            let ax1 = a.x - a.w * 0.5, ay1 = a.y - a.h * 0.5
            let ax2 = a.x + a.w * 0.5, ay2 = a.y + a.h * 0.5
            let bx1 = b.x - b.w * 0.5, by1 = b.y - b.h * 0.5
            let bx2 = b.x + b.w * 0.5, by2 = b.y + b.h * 0.5
            let ix1 = max(ax1, bx1), iy1 = max(ay1, by1)
            let ix2 = min(ax2, bx2), iy2 = min(ay2, by2)
            let iw = max(0, ix2 - ix1), ih = max(0, iy2 - iy1)
            let inter = iw * ih
            let aArea = a.w * a.h, bArea = b.w * b.h
            let uni = aArea + bArea - inter
            return uni > 0 ? inter / uni : 0
        }

        let bboxDupeThreshold: Float = 0.7
        var bboxKept: [(idx: Int, det: UnionDet)] = []

        if prunedCandidates.isEmpty {
            if debugMode { logDebug("   No candidates to check after pruning") }
        } else {
            for candPair in prunedCandidates {
                let origIdx = candPair.idx
                let d = candPair.det

                // Reject if candidate is too large compared to primary (likely background/room)
                let tooLarge = d.w > primary.w * 1.5 && d.h > primary.h * 1.5
                if tooLarge {
                    if debugMode { logDebug("   ❌ [\(origIdx)]: \(className(d.classIdx)) TOO LARGE (bbox)") }
                    continue
                }

                var shouldSkip = false
                var replaceIndex: Int? = nil

                // Compare against primary first (never replace primary)
                let iouWithPrimary = bboxIoU(d, primary)
                if iouWithPrimary > bboxDupeThreshold {
                    if debugMode {
                        let iouPct = String(format: "%.1f", iouWithPrimary * 100)
                        logDebug("   ❌ [\(origIdx)]: \(className(d.classIdx)) DUPLICATE OF PRIMARY (bbox IoU=\(iouPct)%)")
                    }
                    shouldSkip = true
                } else {
                    // Compare with already kept non-primary candidates
                    for (idx, keptPair) in bboxKept.enumerated() {
                        let iou = bboxIoU(d, keptPair.det)
                        if iou > bboxDupeThreshold {
                            // Keep the higher-confidence box within this IoU cluster
                            if d.confidence > keptPair.det.confidence {
                                replaceIndex = idx
                            } else {
                                shouldSkip = true
                            }
                            if debugMode {
                                let iouPct = String(format: "%.1f", iou * 100)
                                logDebug("   ❌ [\(origIdx)]: \(className(d.classIdx)) DUPLICATE (bbox IoU=\(iouPct)%)")
                            }
                            break
                        }
                    }
                }

                if shouldSkip { continue }
                if let idx = replaceIndex {
                    bboxKept[idx] = candPair
                } else {
                    bboxKept.append(candPair)
                }
            }
        }

        kept2 = [primary] + bboxKept.map(\.det)

        if debugMode {
            logDebug("   📦 PRIMARY: center=(\(Int(primary.x)),\(Int(primary.y))) size=\(Int(primary.w))x\(Int(primary.h))")
            if bboxKept.isEmpty {
                logDebug("   No non-primary candidates after bbox dedupe")
            } else {
                for candPair in bboxKept {
                    let d = candPair.det
                    logDebug("   ✅ [\(candPair.idx)]: \(className(d.classIdx)) bbox=(\(Int(d.x)),\(Int(d.y))) \(Int(d.w))x\(Int(d.h))")
                }
            }
        }

        let t5End = Date()
        if debugMode {
            let filterMs = String(format: "%.2f", t5End.timeIntervalSince(t5a) * 1000)
            logDebug("⏱️ STAGE 5b - Bbox dedupe: \(filterMs) ms, kept=\(kept2.count)")
            logMemory("AFTER STAGE 5b")
        }

        } // end useMultiCandidateStage5

        if kept2.isEmpty {
            if debugMode { logDebug("⚠️ No kept detections") }
            DispatchQueue.main.async {
                self.maskImageView.image = nil
                self.resetOverlayScalesForEmptyMask()
            }
            resetProcessingFlag()
            return
        }

        // STAGE 6: Build mask (union bbox + composite)
        let t6 = Date()

        var ux1: Float = .greatestFiniteMagnitude
        var uy1: Float = .greatestFiniteMagnitude
        var ux2: Float = -.greatestFiniteMagnitude
        var uy2: Float = -.greatestFiniteMagnitude

        for d in kept2 {
            ux1 = min(ux1, d.x - d.w * 0.5)
            uy1 = min(uy1, d.y - d.h * 0.5)
            ux2 = max(ux2, d.x + d.w * 0.5)
            uy2 = max(uy2, d.y + d.h * 0.5)
        }

        let origW = CVPixelBufferGetWidth(processBuffer)
        let origH = CVPixelBufferGetHeight(processBuffer)

        // Convert Int pad values to Float for bbox calculations
        let padXf = Float(padX)
        let padYf = Float(padY)

        var bx1 = Int(round((ux1 - padXf) / resizeGain))
        var by1 = Int(round((uy1 - padYf) / resizeGain))
        var bx2 = Int(round((ux2 - padXf) / resizeGain))
        var by2 = Int(round((uy2 - padYf) / resizeGain))

        bx1 = max(0, min(origW - 1, bx1))
        by1 = max(0, min(origH - 1, by1))
        bx2 = max(0, min(origW, bx2))
        by2 = max(0, min(origH, by2))

        let t6a = Date()
        if debugMode {
            let unionMs = String(format: "%.2f", t6a.timeIntervalSince(t6) * 1000)
            logDebug("⏱️ STAGE 6a - Union bbox: \(unionMs) ms")
            logDebug("   image: [\(bx1),\(by1)]→[\(bx2),\(by2)] = \(bx2-bx1)x\(by2-by1)")
        }

        // Tight primary bbox in image coordinates (letterboxed detection → full buffer).
        var tightPBx1 = Float(max(0, Int(round((primary.x - primary.w * 0.5 - padXf) / resizeGain))))
        var tightPBy1 = Float(max(0, Int(round((primary.y - primary.h * 0.5 - padYf) / resizeGain))))
        var tightPBx2 = Float(min(origW, Int(round((primary.x + primary.w * 0.5 - padXf) / resizeGain))))
        var tightPBy2 = Float(min(origH, Int(round((primary.y + primary.h * 0.5 - padYf) / resizeGain))))

        // EMA on tight box first (stable clip + stable base for display margin).
        if let prev = smoothedTightPrimaryBbox {
            let a = bboxSmoothingAlpha
            let b = 1.0 - a
            tightPBx1 = prev.x1 * b + tightPBx1 * a
            tightPBy1 = prev.y1 * b + tightPBy1 * a
            tightPBx2 = prev.x2 * b + tightPBx2 * a
            tightPBy2 = prev.y2 * b + tightPBy2 * a
        }
        smoothedTightPrimaryBbox = (tightPBx1, tightPBy1, tightPBx2, tightPBy2)

        // Mask clip: **tight** box only (no bboxExpandMargin). With multi-candidate Stage 5, max-logit fusion
        // can otherwise pull in pixels just outside the glass / thin objects while the expanded rect still “allows” them.
        let clipMaskBx1 = max(0, Int(round(tightPBx1)))
        let clipMaskBy1 = max(0, Int(round(tightPBy1)))
        let clipMaskBx2 = min(origW, Int(round(tightPBx2)))
        let clipMaskBy2 = min(origH, Int(round(tightPBy2)))

        // UI / AR: expand from smoothed tight (margin helps tall chairs etc.; does not widen mask clip).
        let marginW = (tightPBx2 - tightPBx1) * bboxExpandMargin
        let marginH = (tightPBy2 - tightPBy1) * bboxExpandMargin
        let rawPBx1 = max(0, tightPBx1 - marginW)
        let rawPBy1 = max(0, tightPBy1 - marginH)
        let rawPBx2 = min(Float(origW), tightPBx2 + marginW)
        let rawPBy2 = min(Float(origH), tightPBy2 + marginH)

        let primaryBx1 = max(0, Int(rawPBx1))
        let primaryBy1 = max(0, Int(rawPBy1))
        let primaryBx2 = min(origW, Int(rawPBx2))
        let primaryBy2 = min(origH, Int(rawPBy2))

        if debugMode {
            logDebug("   PRIMARY bbox (UI/AR, expanded): [\(primaryBx1),\(primaryBy1)]→[\(primaryBx2),\(primaryBy2)]")
            logDebug("   MASK clip (tight primary): [\(clipMaskBx1),\(clipMaskBy1)]→[\(clipMaskBx2),\(clipMaskBy2)]")
        }

        // Store primary bbox in normalized coordinates (0-1) for gesture hit testing
        let normX1 = CGFloat(primaryBx1) / CGFloat(origW)
        let normY1 = CGFloat(primaryBy1) / CGFloat(origH)
        let normX2 = CGFloat(primaryBx2) / CGFloat(origW)
        let normY2 = CGFloat(primaryBy2) / CGFloat(origH)
        let bboxCenterImageX = CGFloat(primaryBx1 + primaryBx2) / 2
        let bboxCenterImageY = CGFloat(primaryBy1 + primaryBy2) / 2
        let bboxHeightImagePx = Float(max(1, primaryBy2 - primaryBy1))
        let useDebouncedArMeasurement = isUsingARCameraPath
        if useDebouncedArMeasurement {
            scheduleDebouncedAssistedMeasurement(
                primaryClassIdx: primary.classIdx,
                bboxCenterImageX: bboxCenterImageX,
                bboxCenterImageY: bboxCenterImageY,
                bboxHeightImagePx: bboxHeightImagePx,
                imageWidth: origW,
                imageHeight: origH,
                arDepthSnapshot: arDepthSnapshot
            )
        } else {
            DispatchQueue.main.async { [weak self] in
                self?.updateAssistedOverlayScale(
                    primaryClassIdx: primary.classIdx,
                    bboxCenterImageX: bboxCenterImageX,
                    bboxCenterImageY: bboxCenterImageY,
                    bboxHeightImagePx: bboxHeightImagePx,
                    imageWidth: origW,
                    imageHeight: origH,
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
                classIdx: primary.classIdx,
                detectedWidth: CGFloat(primary.w),
                detectedHeight: CGFloat(primary.h)
            )
            logDebug("👆 [setBbox] viewSize=\(viewW)x\(viewH) bbox=\(self.primaryBboxInView)")
        }

        // Helper: Build full-resolution mask from current kept detections
        func buildFullMask(from detections: [UnionDet]) -> (maskFull: [UInt8], positiveCount: Int, morphedProtoForFuse: [UInt8]) {
            // Compute per-pixel max logits across detections
            var maxLogits = [Float](repeating: -Float.greatestFiniteMagnitude, count: planeSize)

            let M = BLASInt(planeSize)
            let K = BLASInt(32)
            let alpha: Float = 1
            let beta: Float = 0

            // Always use SGEMM batches (better BLAS locality than repeated SGEMV for typical N≤8).
            let batchSize = 64
            var bStart = 0
            while bStart < detections.count {
                let bEnd = min(detections.count, bStart + batchSize)
                let Bn = bEnd - bStart
                let N = BLASInt(Bn)

                var B = [Float](repeating: 0, count: 32 * Bn)
                B.withUnsafeMutableBufferPointer { bPtr in
                    guard let bBase = bPtr.baseAddress else { return }
                    for j in 0..<Bn {
                        let coeffs = detections[bStart + j].coeffs
                        if coeffs.count >= 32 {
                            coeffs.withUnsafeBufferPointer { buf in
                                guard let p = buf.baseAddress else { return }
                                blas_scopy(32, p, 1, bBase.advanced(by: j), N)
                            }
                        } else {
                            for k in 0..<32 {
                                bBase[k * Bn + j] = k < coeffs.count ? coeffs[k] : 0
                            }
                        }
                    }
                }

                var C = [Float](repeating: 0, count: planeSize * Bn)

                planes.withUnsafeBufferPointer { planesPtr in
                    B.withUnsafeBufferPointer { bPtr in
                        C.withUnsafeMutableBufferPointer { cPtr in
                            let lda = M
                            let ldb = N
                            let ldc = N
                            blas_sgemm_rowmajor_transA(m: M, n: N, k: K, alpha: alpha, A: planesPtr.baseAddress!, lda: lda, B: bPtr.baseAddress!, ldb: ldb, beta: beta, C: cPtr.baseAddress!, ldc: ldc)
                        }
                    }
                }

                C.withUnsafeBufferPointer { cPtr in
                    maxLogits.withUnsafeMutableBufferPointer { mPtr in
                        guard let cBase = cPtr.baseAddress, let mBase = mPtr.baseAddress else { return }
                        if Bn == 1 {
                            vDSP_vmax(mBase, 1, cBase, 1, mBase, 1, vDSP_Length(planeSize))
                        } else {
                            for px in 0..<planeSize {
                                let row = cBase.advanced(by: px * Bn)
                                var rowMax: Float = 0
                                var rowMaxIndex: vDSP_Length = 0
                                vDSP_maxvi(row, 1, &rowMax, &rowMaxIndex, vDSP_Length(Bn))
                                if rowMax > mBase[px] { mBase[px] = rowMax }
                            }
                        }
                    }
                }

                bStart = bEnd
            }

            // Stage 14: Threshold -> build maskSmall and positive count
            var maskSmall = [UInt8](repeating: 0, count: planeSize)
            var positiveCount = 0
            maskSmall.withUnsafeMutableBufferPointer { mPtr in
                maxLogits.withUnsafeBufferPointer { lPtr in
                    guard let dst = mPtr.baseAddress, let src = lPtr.baseAddress else { return }
                    var i = 0
                    while i < planeSize {
                        if src[i] > 0 {
                            dst[i] = 255
                            positiveCount += 1
                        } else {
                            dst[i] = 0
                        }
                        i += 1
                    }
                }
            }

            let maskSmallForUpscale: [UInt8] = furnitureFitUseMorphologicalCloseMask
                ? morphologicalBinaryClose3x3Planar8(mask: maskSmall, width: pW, height: pH)
                : maskSmall
            var positiveCountFinal = 0
            for v in maskSmallForUpscale where v > 0 { positiveCountFinal += 1 }

            // Stage 15: Upscale + crop back
            let maskFull = upscaleMask(maskSmall: maskSmallForUpscale,
                                       pW: pW, pH: pH,
                                       modelInput: modelInputSize,
                                       origW: origW, origH: origH,
                                       padX: padX, padY: padY,
                                       contentW: contentW, contentH: contentH)

            return (maskFull, positiveCountFinal, maskSmallForUpscale)
        }

        // Helper: Build full-resolution mask using Metal for the heavy logits->maskSmall step
        // Logic preserved: maskSmall[i] = 255 iff maxLogit > 0.0, then reuse the same upscaleMask() path.
        func buildFullMaskMetal(from detections: [UnionDet]) -> (maskFull: [UInt8], positiveCount: Int, morphedProtoForFuse: [UInt8]) {
            guard let mm = metalMaskLogic else {
                return buildFullMask(from: detections)
            }
            let detCount = detections.count
            if detCount == 0 {
                return ([UInt8](repeating: 0, count: origW * origH), 0, [UInt8](repeating: 0, count: planeSize))
            }
            // Flatten coeffs (detCount x 32) row-major — BLAS copy per detection when possible
            var coeffFlat = [Float](repeating: 0, count: detCount * 32)
            coeffFlat.withUnsafeMutableBufferPointer { cf in
                guard let base = cf.baseAddress else { return }
                for j in 0..<detCount {
                    let c = detections[j].coeffs
                    if c.count >= 32 {
                        c.withUnsafeBufferPointer { buf in
                            guard let p = buf.baseAddress else { return }
                            blas_scopy(32, p, 1, base.advanced(by: j * 32), 1)
                        }
                    } else {
                        for k in 0..<32 {
                            base[j * 32 + k] = k < c.count ? c[k] : 0
                        }
                    }
                }
            }
            // planes is [Float] length 32*planeSize in the current scope (same as CPU path)
            var maskSmall = mm.buildMaskSmall(planes: planes, coeffs: coeffFlat, planeSize: planeSize, detCount: detCount)
            if furnitureFitUseMorphologicalCloseMask {
                maskSmall = morphologicalBinaryClose3x3Planar8(mask: maskSmall, width: pW, height: pH)
            }
            var positiveCount = 0
            for v in maskSmall where v > 0 { positiveCount += 1 }

            // Reuse upscale/crop pipeline with exact integer values from resizeToSquare
            let maskFull = upscaleMask(maskSmall: maskSmall,
                                      pW: pW, pH: pH,
                                      modelInput: modelInputSize,
                                      origW: origW, origH: origH,
                                      padX: padX, padY: padY,
                                      contentW: contentW, contentH: contentH)
            return (maskFull, positiveCount, maskSmall)
        }

        // STAGE 13–15b: Build initial mask from kept2 (union path only)
        let t13to15b = Date()
        var maskFull: [UInt8]
        let build1 = buildFullMaskMetal(from: kept2)
        maskFull = build1.maskFull
        let morphedProtoForFuse = build1.morphedProtoForFuse

        if debugMode {
            let buildPreMs = String(format: "%.2f", Date().timeIntervalSince(t13to15b) * 1000)
            logDebug("⏱️ STAGE 13–15b - Build mask: \(buildPreMs) ms, positive: \(build1.positiveCount)")
            logMemory("AFTER BUILD MASK")
        }

        // 15d: Clear outside **tight** primary bbox (same rect as fused shader). Expanded box is UI/AR only.
        let clipPx1 = clipMaskBx1
        let clipPy1 = clipMaskBy1
        let clipPx2 = clipMaskBx2
        let clipPy2 = clipMaskBy2
        if clipPy1 > 0 {
            let topBytes = clipPy1 * origW
            _ = maskFull.withUnsafeMutableBufferPointer { ptr in
                memset(ptr.baseAddress!, 0, topBytes)
            }
        }
        if clipPy2 < origH {
            let bottomStart = clipPy2 * origW
            let bottomBytes = (origH - clipPy2) * origW
            _ = maskFull.withUnsafeMutableBufferPointer { ptr in
                memset(ptr.baseAddress!.advanced(by: bottomStart), 0, bottomBytes)
            }
        }
        if clipPx1 > 0 || clipPx2 < origW {
            maskFull.withUnsafeMutableBufferPointer { ptr in
                for y in clipPy1..<clipPy2 {
                    let rowStart = y * origW
                    if clipPx1 > 0 {
                        memset(ptr.baseAddress!.advanced(by: rowStart), 0, clipPx1)
                    }
                    if clipPx2 < origW {
                        memset(ptr.baseAddress!.advanced(by: rowStart + clipPx2), 0, origW - clipPx2)
                    }
                }
            }
        }
        if debugMode {
            logDebug("⏱️ STAGE 15d - Mask cleared outside primary bbox [\(clipPx1),\(clipPy1)]→[\(clipPx2),\(clipPy2)]")
        }

        // Morph close (if enabled) is applied on prototype mask before upscale; full-res close removed for performance.
        let maskForComposite: [UInt8] = maskFull

        // Prepare flattened coeffs for fused GPU path
        let detCountFused = kept2.count
        var coeffFlatFused = [Float](repeating: 0, count: detCountFused * 32)
        for j in 0..<detCountFused {
            let c = kept2[j].coeffs
            let n = min(32, c.count)
            for k in 0..<n { coeffFlatFused[j*32 + k] = c[k] }
        }

        // STAGE 6b: Composite
        setProgress(0.92, text: "Compositing…")

        // --- 4. COMPOSITING (Fused when available) ---
        let compStart = Date()
        var composedImage: CGImage?

        if let device = metalDevice,
           let queue = metalCommandQueue,
           let textureCache = cvTextureCache {  // Use cached texture cache (created once in setupMetal)

            func makeTexture(from pixelBuffer: CVPixelBuffer, pixelFormat: MTLPixelFormat) -> MTLTexture? {
                let cache = textureCache
                var cvTexture: CVMetalTexture?
                let w = CVPixelBufferGetWidth(pixelBuffer)
                let h = CVPixelBufferGetHeight(pixelBuffer)
                let status = CVMetalTextureCacheCreateTextureFromImage(kCFAllocatorDefault, cache, pixelBuffer, nil, pixelFormat, w, h, 0, &cvTexture)
                guard status == kCVReturnSuccess, let cvTex = cvTexture, let tex = CVMetalTextureGetTexture(cvTex) else { return nil }
                return tex
            }

            // Source BGRA texture from camera buffer (use processBuffer for landscape support)
            CVPixelBufferLockBaseAddress(processBuffer, .readOnly)
            let srcTexture = makeTexture(from: processBuffer, pixelFormat: .bgra8Unorm)
            if debugMode, origW > 0, origH > 0,
               let camBase = CVPixelBufferGetBaseAddress(processBuffer)?.assumingMemoryBound(to: UInt8.self) {
                let cameraBytesPerRow = CVPixelBufferGetBytesPerRow(processBuffer)
                func rawCameraBGRAString(cx: Int, cy: Int) -> String {
                    guard cx >= 0, cx < origW, cy >= 0, cy < origH else {
                        return "(\(cx),\(cy)) out of range"
                    }
                    let rawOffset = cy * cameraBytesPerRow + cx * 4
                    let blue = camBase[rawOffset]
                    let green = camBase[rawOffset + 1]
                    let red = camBase[rawOffset + 2]
                    let alpha = camBase[rawOffset + 3]
                    return "(\(cx),\(cy)) B=\(blue) G=\(green) R=\(red) A=\(alpha)"
                }
                let frameCx = origW / 2
                let frameCy = origH / 2
                let bandX0 = max(0, primaryBx1)
                let bandX1 = min(origW, primaryBx2)
                let bandY0 = max(0, primaryBy1)
                let bandY1 = min(origH, primaryBy2)
                let bboxCx = bandX1 > bandX0 ? (bandX0 + bandX1) / 2 : frameCx
                let bboxCy = bandY1 > bandY0 ? (bandY0 + bandY1) / 2 : frameCy
                logDebug("🎨 [RAW CAMERA] frame \(rawCameraBGRAString(cx: frameCx, cy: frameCy)) | bbox \(rawCameraBGRAString(cx: bboxCx, cy: bboxCy))")
            }
            CVPixelBufferUnlockBaseAddress(processBuffer, .readOnly)

                if let src = srcTexture, let cmdBuf = queue.makeCommandBuffer() {
                // Fused + morph: upload morphed prototype mask (matches CPU vImage close before upscale).
                if furnitureFitUseMorphologicalCloseMask,
                   let fusedM = fusedMaskMorphedCompositePipeline,
                   detCountFused > 0,
                   morphedProtoForFuse.count == planeSize {
                    let protoBytes = morphedProtoForFuse.count * MemoryLayout<UInt8>.size
                    let protoBuf: MTLBuffer?
                    if let cached = cachedFusedProtoMaskBuf, cachedFusedProtoMaskCapacity >= protoBytes {
                        protoBuf = cached
                    } else {
                        let newCapacity = max(protoBytes, cachedFusedProtoMaskCapacity * 2, 1024)
                        cachedFusedProtoMaskBuf = device.makeBuffer(length: newCapacity, options: .storageModeShared)
                        cachedFusedProtoMaskCapacity = newCapacity
                        protoBuf = cachedFusedProtoMaskBuf
                    }
                    if let pb = protoBuf {
                        _ = morphedProtoForFuse.withUnsafeBytes { raw in
                            memcpy(pb.contents(), raw.baseAddress!, protoBytes)
                        }
                    }

                    let outDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .bgra8Unorm, width: origW, height: origH, mipmapped: false)
                    outDesc.usage = [.shaderWrite, .shaderRead]
                    let outTexture = device.makeTexture(descriptor: outDesc)

                    if let enc = cmdBuf.makeComputeCommandEncoder(), let out = outTexture, let protoBuf {
                        enc.setComputePipelineState(fusedM)
                        enc.setTexture(src, index: 0)
                        enc.setTexture(out, index: 1)
                        enc.setBuffer(protoBuf, offset: 0, index: 0)
                        var pW_u = UInt32(pW)
                        var pH_u = UInt32(pH)
                        var origW_u = UInt32(origW)
                        var origH_u = UInt32(origH)
                        var modelInput_u = UInt32(modelInputSize)
                        var resizeGain_f = resizeGain
                        var padX_f = Float(padX)
                        var padY_f = Float(padY)
                        enc.setBytes(&pW_u, length: MemoryLayout<UInt32>.size, index: 1)
                        enc.setBytes(&pH_u, length: MemoryLayout<UInt32>.size, index: 2)
                        enc.setBytes(&origW_u, length: MemoryLayout<UInt32>.size, index: 3)
                        enc.setBytes(&origH_u, length: MemoryLayout<UInt32>.size, index: 4)
                        enc.setBytes(&modelInput_u, length: MemoryLayout<UInt32>.size, index: 5)
                        enc.setBytes(&resizeGain_f, length: MemoryLayout<Float>.size, index: 6)
                        enc.setBytes(&padX_f, length: MemoryLayout<Float>.size, index: 7)
                        enc.setBytes(&padY_f, length: MemoryLayout<Float>.size, index: 8)
                        var bx1_u = UInt32(clipMaskBx1)
                        var by1_u = UInt32(clipMaskBy1)
                        var bx2_u = UInt32(clipMaskBx2)
                        var by2_u = UInt32(clipMaskBy2)
                        enc.setBytes(&bx1_u, length: MemoryLayout<UInt32>.size, index: 9)
                        enc.setBytes(&by1_u, length: MemoryLayout<UInt32>.size, index: 10)
                        enc.setBytes(&bx2_u, length: MemoryLayout<UInt32>.size, index: 11)
                        enc.setBytes(&by2_u, length: MemoryLayout<UInt32>.size, index: 12)

                        let w = fusedM.threadExecutionWidth
                        let h = max(1, fusedM.maxTotalThreadsPerThreadgroup / w)
                        let tg = MTLSize(width: w, height: h, depth: 1)
                        let grid = MTLSize(width: origW, height: origH, depth: 1)
                        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
                        enc.endEncoding()

                        cmdBuf.commit()
                        cmdBuf.waitUntilCompleted()

                        let bytesPerRow = origW * 4
                        var rgba = [UInt8](repeating: 0, count: origH * bytesPerRow)
                        out.getBytes(&rgba, bytesPerRow: bytesPerRow, from: MTLRegionMake2D(0, 0, origW, origH), mipmapLevel: 0)
                        if let ctx = CGContext(data: nil, width: origW, height: origH, bitsPerComponent: 8, bytesPerRow: bytesPerRow, space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) {
                            let dst = ctx.data!.assumingMemoryBound(to: UInt8.self)
                            rgba.withUnsafeBytes { srcBuf in
                                guard let srcBytes = srcBuf.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }
                                for y in 0..<origH {
                                    dst.advanced(by: y * bytesPerRow).update(from: srcBytes.advanced(by: y * bytesPerRow), count: bytesPerRow)
                                }
                            }
                            if let img = ctx.makeImage() {
                                composedImage = img
                            }
                        }
                    }
                } else if !furnitureFitUseMorphologicalCloseMask, let fused = fusedMaskCompositePipeline {
                    // Fused path: compute max logits and composite in one pass.
                    // Prepare buffers: planes (32*planeSize floats) and coeffs (detCount*32 floats)
                    let planesBytes = planes.count * MemoryLayout<Float>.size
                    let coeffBytes = coeffFlatFused.count * MemoryLayout<Float>.size

                    // Reuse cached buffers instead of allocating new ones each frame
                    let planesBuf: MTLBuffer?
                    if let cached = cachedFusedPlanesBuf, cachedFusedPlanesCapacity >= planesBytes {
                        planesBuf = cached
                    } else {
                        let newCapacity = max(planesBytes, cachedFusedPlanesCapacity * 2, 1024)
                        cachedFusedPlanesBuf = device.makeBuffer(length: newCapacity, options: .storageModeShared)
                        cachedFusedPlanesCapacity = newCapacity
                        planesBuf = cachedFusedPlanesBuf
                    }
                    let coeffBuf: MTLBuffer?
                    if let cached = cachedFusedCoeffBuf, cachedFusedCoeffCapacity >= coeffBytes {
                        coeffBuf = cached
                    } else {
                        let newCapacity = max(coeffBytes, cachedFusedCoeffCapacity * 2, 1024)
                        cachedFusedCoeffBuf = device.makeBuffer(length: newCapacity, options: .storageModeShared)
                        cachedFusedCoeffCapacity = newCapacity
                        coeffBuf = cachedFusedCoeffBuf
                    }

                    // Copy data into reused buffers
                    if let pb = planesBuf {
                        memcpy(pb.contents(), planes, planesBytes)
                    }
                    if let cb = coeffBuf {
                        memcpy(cb.contents(), coeffFlatFused, coeffBytes)
                    }

                    // Output texture
                    let outDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .bgra8Unorm, width: origW, height: origH, mipmapped: false)
                    outDesc.usage = [.shaderWrite, .shaderRead]
                    let outTexture = device.makeTexture(descriptor: outDesc)

                    if let enc = cmdBuf.makeComputeCommandEncoder(), let out = outTexture, let planesBuf, let coeffBuf {
                        enc.setComputePipelineState(fused)
                        enc.setTexture(src, index: 0)
                        enc.setTexture(out, index: 1)
                        enc.setBuffer(planesBuf, offset: 0, index: 0)
                        enc.setBuffer(coeffBuf, offset: 0, index: 1)
                        var pW_u = UInt32(pW)
                        var pH_u = UInt32(pH)
                        var det_u = UInt32(detCountFused)
                        var origW_u = UInt32(origW)
                        var origH_u = UInt32(origH)
                        var modelInput_u = UInt32(modelInputSize)
                        var resizeGain_f = resizeGain
                        var padX_f = Float(padX)  // Convert Int to Float for shader
                        var padY_f = Float(padY)  // Convert Int to Float for shader
                        enc.setBytes(&pW_u, length: MemoryLayout<UInt32>.size, index: 2)
                        enc.setBytes(&pH_u, length: MemoryLayout<UInt32>.size, index: 3)
                        enc.setBytes(&det_u, length: MemoryLayout<UInt32>.size, index: 4)
                        enc.setBytes(&origW_u, length: MemoryLayout<UInt32>.size, index: 5)
                        enc.setBytes(&origH_u, length: MemoryLayout<UInt32>.size, index: 6)
                        enc.setBytes(&modelInput_u, length: MemoryLayout<UInt32>.size, index: 7)
                        enc.setBytes(&resizeGain_f, length: MemoryLayout<Float>.size, index: 8)
                        enc.setBytes(&padX_f, length: MemoryLayout<Float>.size, index: 9)
                        enc.setBytes(&padY_f, length: MemoryLayout<Float>.size, index: 10)
                        // Outside primary bbox → transparent (GPU skips logits there).
                        var bx1_u = UInt32(clipMaskBx1)
                        var by1_u = UInt32(clipMaskBy1)
                        var bx2_u = UInt32(clipMaskBx2)
                        var by2_u = UInt32(clipMaskBy2)
                        enc.setBytes(&bx1_u, length: MemoryLayout<UInt32>.size, index: 11)
                        enc.setBytes(&by1_u, length: MemoryLayout<UInt32>.size, index: 12)
                        enc.setBytes(&bx2_u, length: MemoryLayout<UInt32>.size, index: 13)
                        enc.setBytes(&by2_u, length: MemoryLayout<UInt32>.size, index: 14)

                        let w = fused.threadExecutionWidth
                        let h = max(1, fused.maxTotalThreadsPerThreadgroup / w)
                        let tg = MTLSize(width: w, height: h, depth: 1)
                        let grid = MTLSize(width: origW, height: origH, depth: 1)
                        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
                        enc.endEncoding()

                        cmdBuf.commit()
                        cmdBuf.waitUntilCompleted()

                        // Read back as CGImage — use context-owned buffer to avoid CGBitmapContextCreateImage crash on iPhone 12 (alignment)
                        let bytesPerRow = origW * 4
                        var rgba = [UInt8](repeating: 0, count: origH * bytesPerRow)
                        out.getBytes(&rgba, bytesPerRow: bytesPerRow, from: MTLRegionMake2D(0, 0, origW, origH), mipmapLevel: 0)
                        if let ctx = CGContext(data: nil, width: origW, height: origH, bitsPerComponent: 8, bytesPerRow: bytesPerRow, space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) {
                            let dst = ctx.data!.assumingMemoryBound(to: UInt8.self)
                            rgba.withUnsafeBytes { srcBuf in
                                guard let src = srcBuf.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }
                                for y in 0..<origH {
                                    dst.advanced(by: y * bytesPerRow).update(from: src.advanced(by: y * bytesPerRow), count: bytesPerRow)
                                }
                            }
                            if let img = ctx.makeImage() {
                                composedImage = img
                            }
                        }
                    }
                } else if let pipeline = compositePipeline {
                    // Non-fused GPU path: upload mask and composite (existing path)
                    let maskDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r8Unorm, width: origW, height: origH, mipmapped: false)
                    maskDesc.usage = [.shaderRead]
                    let maskTexture = device.makeTexture(descriptor: maskDesc)
                    if let mt = maskTexture {
                        let region = MTLRegionMake2D(0, 0, origW, origH)
                        mt.replace(region: region, mipmapLevel: 0, withBytes: maskForComposite, bytesPerRow: origW)
                    }
                    let outDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .bgra8Unorm, width: origW, height: origH, mipmapped: false)
                    outDesc.usage = [.shaderWrite, .shaderRead]
                    let outTexture = device.makeTexture(descriptor: outDesc)
                    if let enc = cmdBuf.makeComputeCommandEncoder(), let out = outTexture, let maskTex = maskTexture {
                        enc.setComputePipelineState(pipeline)
                        enc.setTexture(src, index: 0)
                        enc.setTexture(maskTex, index: 1)
                        enc.setTexture(out, index: 2)
                        let w = pipeline.threadExecutionWidth
                        let h = max(1, pipeline.maxTotalThreadsPerThreadgroup / w)
                        let tg = MTLSize(width: w, height: h, depth: 1)
                        let grid = MTLSize(width: origW, height: origH, depth: 1)
                        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
                        enc.endEncoding()
                        cmdBuf.commit(); cmdBuf.waitUntilCompleted()
                        let bytesPerRow = origW * 4
                        var rgba = [UInt8](repeating: 0, count: origH * bytesPerRow)
                        out.getBytes(&rgba, bytesPerRow: bytesPerRow, from: MTLRegionMake2D(0, 0, origW, origH), mipmapLevel: 0)
                        if let ctx = CGContext(data: nil, width: origW, height: origH, bitsPerComponent: 8, bytesPerRow: bytesPerRow, space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) {
                            let dst = ctx.data!.assumingMemoryBound(to: UInt8.self)
                            rgba.withUnsafeBytes { srcBuf in
                                guard let src = srcBuf.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }
                                for y in 0..<origH {
                                    dst.advanced(by: y * bytesPerRow).update(from: src.advanced(by: y * bytesPerRow), count: bytesPerRow)
                                }
                            }
                            if let img = ctx.makeImage() {
                                composedImage = img
                            }
                        }
                    }
                }
            }
        }

        if composedImage == nil {
            // Fallback CPU compositing (use processBuffer for landscape support)
            let ctx = CGContext(data: nil, width: origW, height: origH, bitsPerComponent: 8, bytesPerRow: origW * 4, space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
            CVPixelBufferLockBaseAddress(processBuffer, .readOnly)
            let outBase = ctx.data!.assumingMemoryBound(to: UInt8.self)
            let origBase = CVPixelBufferGetBaseAddress(processBuffer)!.assumingMemoryBound(to: UInt8.self)
            let bytesPerRowOut = origW * 4
            let camRowBytes = CVPixelBufferGetBytesPerRow(processBuffer)
            let x0 = max(0, clipMaskBx1)
            let x1 = min(origW, clipMaskBx2)
            let y0 = max(0, clipMaskBy1)
            let y1 = min(origH, clipMaskBy2)
            for y in 0..<origH {
                memset(outBase.advanced(by: y * bytesPerRowOut), 0, bytesPerRowOut)
            }
            maskForComposite.withUnsafeBufferPointer { maskBuf in
                guard let maskBase = maskBuf.baseAddress else { return }
                compositeCpuCameraBandAccelerated(
                    outBase: outBase,
                    origBase: origBase,
                    maskBase: maskBase,
                    origW: origW,
                    camRowBytes: camRowBytes,
                    bytesPerRowOut: bytesPerRowOut,
                    x0: x0,
                    x1: x1,
                    y0: y0,
                    y1: y1
                )
            }
            CVPixelBufferUnlockBaseAddress(processBuffer, .readOnly)
            composedImage = ctx.makeImage()
        }

        let t_comp = Date().timeIntervalSince(compStart) * 1000
        if debugMode {
            if let img = composedImage {
                logDebug("✅ Composed image: \(img.width)x\(img.height)")
            } else {
                logDebug("⚠️ COMPOSITING RETURNED NIL IMAGE")
            }
            logDebug("🖼️ [STEP 4] Compositing: \(String(format: "%.2f", t_comp))ms")
        }

        // STAGE 7: Finalize (debug overlays drawn onto composedImage if available)
        let t7 = Date()

        // Prepare a drawing context starting from composedImage (or an empty one if nil)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bytesPerRow = origW * 4
        var overlayBuffer = [UInt8](repeating: 0, count: origH * bytesPerRow)
        let ctx: CGContext? = CGContext(data: &overlayBuffer, width: origW, height: origH, bitsPerComponent: 8, bytesPerRow: bytesPerRow, space: colorSpace, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)

        if let base = composedImage {
            // Draw the composed image as the background
            ctx?.draw(base, in: CGRect(x: 0, y: 0, width: origW, height: origH))
        }

        if let ctx = ctx, debugMode {
            // Always draw class name labels
            let font = CTFontCreateWithName("Helvetica-Bold" as CFString, 36, nil)
            for (_, d) in kept2.enumerated() {
                let dx1 = Int(round((d.x - d.w * 0.5 - padXf) / resizeGain))
                let dy1 = Int(round((d.y - d.h * 0.5 - padYf) / resizeGain))
                let dx2 = Int(round((d.x + d.w * 0.5 - padXf) / resizeGain))
                let dy2 = Int(round((d.y + d.h * 0.5 - padYf) / resizeGain))

                let clampedX1 = max(0, dx1)
                let clampedY1 = max(0, dy1)
                let clampedW = min(origW - clampedX1, dx2 - dx1)
                let clampedH = min(origH - clampedY1, dy2 - dy1)

                let detectionColor = UIColor.white
                let className = classNames[d.classIdx] ?? "unknown"
                let confidence = String(format: "%.2f", d.confidence)
                let labelText = "\(className) (\(confidence))"

                let attributes: [NSAttributedString.Key: Any] = [
                    .font: font,
                    .foregroundColor: detectionColor
                ]
                let attributedString = NSAttributedString(string: labelText, attributes: attributes)
                let line = CTLineCreateWithAttributedString(attributedString)
                let textBounds = CTLineGetBoundsWithOptions(line, .useOpticalBounds)

                let labelX = CGFloat(clampedX1)
                let labelY = CGFloat(origH - clampedY1 + 4)

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
                ctx.setFillColor(detectionColor.cgColor)
                CTLineDraw(line, ctx)
                ctx.restoreGState()

                // Draw bounding box
                ctx.setLineWidth(2.0)
                ctx.setStrokeColor(UIColor.cyan.cgColor)
                ctx.stroke(CGRect(x: clampedX1, y: origH - clampedY1 - clampedH, width: clampedW, height: clampedH))
            }

            // Draw PRIMARY bounding box in red (thick) - use same coords as cyan box
            let pDx1 = Int(round((primary.x - primary.w * 0.5 - padXf) / resizeGain))
            let pDy1 = Int(round((primary.y - primary.h * 0.5 - padYf) / resizeGain))
            let pDx2 = Int(round((primary.x + primary.w * 0.5 - padXf) / resizeGain))
            let pDy2 = Int(round((primary.y + primary.h * 0.5 - padYf) / resizeGain))
            let pClampedX1 = max(0, pDx1)
            let pClampedY1 = max(0, pDy1)
            let pClampedW = min(origW - pClampedX1, pDx2 - pDx1)
            let pClampedH = min(origH - pClampedY1, pDy2 - pDy1)
            ctx.setStrokeColor(UIColor.red.cgColor)
            ctx.setLineWidth(4.0)
            ctx.stroke(CGRect(x: pClampedX1, y: origH - pClampedY1 - pClampedH, width: pClampedW, height: pClampedH))

            // Draw union bounding box in green
            ctx.setStrokeColor(UIColor.green.cgColor)
            ctx.setLineWidth(6.0)
            ctx.stroke(CGRect(x: bx1, y: origH - by2, width: bx2 - bx1, height: by2 - by1))
        }

        if debugMode, let dbgCtx = ctx, let raw = dbgCtx.data {
            let bandX0 = max(0, clipMaskBx1)
            let bandX1 = min(origW, clipMaskBx2)
            let bandY0 = max(0, clipMaskBy1)
            let bandY1 = min(origH, clipMaskBy2)
            logPremultipliedRGBASampleIfDebug(
                outBase: raw.assumingMemoryBound(to: UInt8.self),
                bytesPerRowOut: bytesPerRow,
                width: origW,
                height: origH,
                x0: bandX0,
                x1: bandX1,
                y0: bandY0,
                y1: bandY1,
                tag: "CoreML display buffer"
            )
        }

        if let finalCtx = ctx, let img = finalCtx.makeImage() {
            composedImage = img
        }

        // Present result - handle rotation for display.
        // Capture the composed image by value BEFORE clearing it to avoid a race with the async main queue.
        if let finalImage = composedImage {
            let needsRotate = isLandscape && !self.isUsingARCameraPath && self.lockedOrientation != .landscape
            DispatchQueue.main.async {
                var cgImg = finalImage
                if needsRotate {
                    if let r = self.rotateCGImage90(cgImg, clockwise: true) { cgImg = r }
                }
                self.maskImageView.image = UIImage(cgImage: cgImg)

                // DEBUG: Inspect why maskImageView might not be visible even when we have a composed image.
                logDebug("🖼️ [DISPLAY] maskImageView:")
                logDebug("   frame=\(self.maskImageView.frame)")
                logDebug("   bounds=\(self.maskImageView.bounds)")
                logDebug("   hidden=\(self.maskImageView.isHidden)")
                logDebug("   alpha=\(self.maskImageView.alpha)")
                logDebug("   transform=\(self.maskImageView.transform)")
                logDebug("   superview=\(String(describing: self.maskImageView.superview))")
                logDebug("   window=\(String(describing: self.maskImageView.window))")
                logDebug("   image size=\(self.maskImageView.image?.size ?? .zero)")
            }
        }
        composedImage = nil
        protoRawFloats = []
        protoPlanes = []
        resetProcessingFlag()

        // Trigger first-detection UI dismissal based on mask having any positive pixels
        let hasMask = maskForComposite.contains(where: { $0 > 0 })
        if hasMask { finishFirstDetectionIfNeeded() }

        let t7End = Date()
        let frameEnd = Date()

        if debugMode {
            let finalizeMs = String(format: "%.2f", t7End.timeIntervalSince(t7) * 1000)
            let frameTotalMs = String(format: "%.2f", frameEnd.timeIntervalSince(frameStart) * 1000)
            logDebug("⏱️ STAGE 7 - Finalize: \(finalizeMs) ms")
            logMemory("FRAME END")
            logDebug("⏱️ FRAME TOTAL: \(frameTotalMs) ms")
            logDebug("⏱️ ═══════════════════════════════════════════\n")
        }
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

    // MARK: - Upscale Mask
    // Uses exact integer values from resizeToSquare to ensure pixel-perfect reverse mapping
    private func upscaleMask(maskSmall: [UInt8], pW: Int, pH: Int, modelInput: Int, origW: Int, origH: Int, padX: Int, padY: Int, contentW: Int, contentH: Int) -> [UInt8] {
        var maskModel = [UInt8](repeating: 0, count: modelInput * modelInput)
        maskModel.withUnsafeMutableBufferPointer { dstPtr in
            maskSmall.withUnsafeBufferPointer { srcPtr in
                var s = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: srcPtr.baseAddress!), height: vImagePixelCount(pH), width: vImagePixelCount(pW), rowBytes: pW)
                var d = vImage_Buffer(data: dstPtr.baseAddress!, height: vImagePixelCount(modelInput), width: vImagePixelCount(modelInput), rowBytes: modelInput)
                let flags: vImage_Flags = useBilinearUpscaling ? vImage_Flags(kvImageHighQualityResampling) : vImage_Flags(kvImageNoFlags)
                vImageScale_Planar8(&s, &d, nil, flags)
            }
        }

        // Use exact integer values from resizeToSquare (no recomputation = no drift)
        let x0 = max(0, min(modelInput - 1, padX))
        let y0 = max(0, min(modelInput - 1, padY))
        let cW = max(1, min(modelInput - x0, contentW))
        let cH = max(1, min(modelInput - y0, contentH))

        var cropped = [UInt8](repeating: 0, count: cW * cH)
        for y in 0..<cH {
            let srcRow = (y0 + y) * modelInput + x0
            let dstRow = y * cW
            for x in 0..<cW { cropped[dstRow + x] = maskModel[srcRow + x] }
        }

        var maskFull = [UInt8](repeating: 0, count: origW * origH)
        maskFull.withUnsafeMutableBufferPointer { dstPtr in
            cropped.withUnsafeBufferPointer { srcPtr in
                var s = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: srcPtr.baseAddress!), height: vImagePixelCount(cH), width: vImagePixelCount(cW), rowBytes: cW)
                var d = vImage_Buffer(data: dstPtr.baseAddress!, height: vImagePixelCount(origH), width: vImagePixelCount(origW), rowBytes: origW)
                let flags: vImage_Flags = useBilinearUpscaling ? vImage_Flags(kvImageHighQualityResampling) : vImage_Flags(kvImageNoFlags)
                vImageScale_Planar8(&s, &d, nil, flags)
            }
        }
        return maskFull
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

    // MARK: - Resize to Square
    // Returns integer pad values to ensure exact matching in reverse mapping (upscaleMask)
    private func resizeToSquare(_ src: CVPixelBuffer, size: Int) -> (buffer: CVPixelBuffer, gain: Float, padX: Int, padY: Int, newW: Int, newH: Int)? {
        CVPixelBufferLockBaseAddress(src, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(src, .readOnly) }

        let srcW = CVPixelBufferGetWidth(src)
        let srcH = CVPixelBufferGetHeight(src)

        let gain = min(Float(size) / Float(srcW), Float(size) / Float(srcH))
        let newW = Int(Float(srcW) * gain)
        let newH = Int(Float(srcH) * gain)
        // Use integer division to get exact pixel offset (same as forward placement)
        let padX = (size - newW) / 2
        let padY = (size - newH) / 2

        // Reuse cached buffer if size matches, otherwise create new one
        if cachedSquareSize != size || cachedSquareBuffer == nil {
            var newBuffer: CVPixelBuffer?
            guard CVPixelBufferCreate(nil, size, size, kCVPixelFormatType_32BGRA, nil, &newBuffer) == kCVReturnSuccess,
                  let buf = newBuffer else { return nil }
            cachedSquareBuffer = buf
            cachedSquareSize = size
        }

        guard let dst = cachedSquareBuffer else { return nil }

        CVPixelBufferLockBaseAddress(dst, [])
        defer { CVPixelBufferUnlockBaseAddress(dst, []) }

        guard let srcBase = CVPixelBufferGetBaseAddress(src),
              let dstBase = CVPixelBufferGetBaseAddress(dst) else { return nil }

        YoloUltralyticsLetterboxFill.fillOpaqueBGRA114(dstBase: dstBase, totalByteCount: size * size * 4)

        var srcBuffer = vImage_Buffer(data: srcBase, height: vImagePixelCount(srcH), width: vImagePixelCount(srcW), rowBytes: CVPixelBufferGetBytesPerRow(src))
        let dstPtr = dstBase.assumingMemoryBound(to: UInt8.self)
        let dstRowBytes = CVPixelBufferGetBytesPerRow(dst)
        let offsetPtr = dstPtr.advanced(by: padY * dstRowBytes + padX * 4)
        var dstBuffer = vImage_Buffer(data: offsetPtr, height: vImagePixelCount(newH), width: vImagePixelCount(newW), rowBytes: dstRowBytes)

        guard vImageScale_ARGB8888(&srcBuffer, &dstBuffer, nil, vImage_Flags(0)) == kvImageNoError else { return nil }

        return (buffer: dst, gain: gain, padX: padX, padY: padY, newW: newW, newH: newH)
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

    private func finishFirstDetectionIfNeeded() {
        guard !hasFirstDetection else { return }
        hasFirstDetection = true
        segmentationCompletedOnceThisSession = true
        onFirstSegmentationComplete?()
        DispatchQueue.main.async {
            UIView.animate(withDuration: 0.25) {
                self.progressContainer.alpha = 0
            } completion: { _ in
                self.progressContainer.isHidden = true
                self.progressContainer.alpha = 1
            }
        }
    }

    // MARK: - Gestures
    @objc private func handlePinch(_ gesture: UIPinchGestureRecognizer) {
        guard maskImageView.image != nil else { return }

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
        frameLock.lock()
        if isProcessing {
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
        let bgraW = CVPixelBufferGetWidth(bgra)
        let bgraH = CVPixelBufferGetHeight(bgra)
        let depthSnapshot = FurnitureFitARSupport.makeDepthSnapshot(
            frame: frame,
            bgraWidth: bgraW,
            bgraHeight: bgraH,
            lockedOrientation: lockedOrientation
        )
        // Pause AR so ARKit stops delivering frames while `detectionQueue` runs (prevents "retaining N ARFrames").
        frameLock.lock()
        arPausedForSegmentation = true
        frameLock.unlock()
        session.pause()
        detectionQueue.async { [weak self] in
            self?.processFrame(bgra, arDepthSnapshot: depthSnapshot)
        }
    }

    func session(_ session: ARSession, didFailWithError error: Error) {
        logDebug("⚠️ [FurnitureFit] ARSession failed: \(error.localizedDescription) — using AVCapture")
        DispatchQueue.main.async {
            self.startClassicCameraPathIfNeeded()
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

