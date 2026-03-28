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
            cmd.commit()
            cmd.waitUntilCompleted()

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
    var useBilinearUpscaling: Bool = false
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
    /// Prefer the bundled YOLO-E ONNX graph on iOS so segmentation decode matches Android more closely.
    private let furnitureFitUseOnnxRuntime: Bool = true
    /// When the primary object is a monitor, also include a supporting table/desk in the mask union.
    private let includeSupportingTableForMonitorScene: Bool = true
    /// Match Android's ONNX mask decode as closely as possible: primary-only, sigmoid > 0.5,
    /// direct proto-to-image mapping, and no custom morph/fused segmentation path.
    private let furnitureFitMatchAndroidSegmentation: Bool = true
    /// After bbox clip, optional 3×3 morphological close on prototype mask before upscale; disabled when
    /// [furnitureFitMatchAndroidSegmentation] is true so the mask stays aligned with Android behavior.
    private let furnitureFitUseMorphologicalCloseMask: Bool = false
    var useBilinearUpscaling: Bool = false
    var lockedOrientation: PhotoOrientation = .portrait  // Locked orientation (no rotation needed when .landscape)

    // MARK: Room Dimensions (from SHARP output, in meters)
    var roomWidthMeters: Float = 4.0   // Default fallback
    var roomHeightMeters: Float = 3.0  // Default fallback

    // Callback for reporting estimated furniture size (room-based + optional AR height, in meters)
    var onFurnitureSizeEstimated: ((FurnitureSizeEstimate) -> Void)?

    // Sizing calculator (created when room dimensions are set)
    private var sizingCalculator: FurnitureSizingCalculator?

    /// Avoid duplicate NotificationCenter registrations when `startIfNeeded()` runs more than once.
    private var didRegisterArAssistedSettingObserver = false

    // Debug mode - read from settings
    var debugMode: Bool {
        return AppStateManager.shared.qualitySettings.debugMode
    }

    // MARK: - Ignored Classes (loaded from blacklist.json)
    private var clsToIgnore: Set<Int> = []

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
    private var arBGRAReuse: CVPixelBuffer?
    private let arCIContext = CIContext(options: [.cacheIntermediates: false])
    /// View size for AR-assisted camera path (set in `layoutSubviews`).
    private var cachedARViewportSize: CGSize = .zero
    /// True while ARSession is the active camera source (not AVCaptureSession).
    private var isUsingARCameraPath = false
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
    private var latestDisplayedFurnitureHeightMeters: Float?

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

    /// Primary furniture bounding box in view coordinates (for gesture hit testing)
    private var primaryBboxInView: CGRect = .zero

    /// Match Android cutout quality: build the mask from the primary detection only.
    /// The older multi-candidate union tends to pull in nearby background boxes and makes glass/cup masks boxy.
    private var useMultiCandidateStage5 = true

    // MARK: - Metal (FIXED: stored properties instead of computed to prevent resource leak)
    private var metalDevice: MTLDevice?
    private var metalCommandQueue: MTLCommandQueue?  // FIXED: was computed property creating new queue on every access
    private var metalLibrary: MTLLibrary?            // FIXED: was computed property creating new library on every access
    private var metalCIContext: CIContext?
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
    private var cachedMLArray: MLMultiArray?
    private var cachedMLArraySize: Int = 0

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
    private var mlModel: MLModel?  // yoloe-11l 1280 model
    private let detectionQueue = DispatchQueue(label: "com.furnit.detection", qos: .userInitiated)
    private var lastProcessTime = Date.distantPast
    private var isProcessing = false
    private enum PendingInferenceSource {
        case camera
        case ar
    }
    private struct PendingInferenceFrame {
        let pixelBuffer: CVPixelBuffer
        let arDepthSnapshot: FurnitureFitARDepthSnapshot?
        let source: PendingInferenceSource
    }
    private var pendingInferenceFrame: PendingInferenceFrame?
    /// While inference is running, camera frames are dropped; when set, next completion clears `lastProcessTime` so the following frame is not delayed by `processInterval`.
    private var preferImmediateNextInference = false
    private let frameLock = NSLock() // Protects lastProcessTime and isProcessing for early-exit checks

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

    /// Thread-safe completion: either start the latest pending frame immediately or clear the processing flag.
    private func resetProcessingFlag() {
        var pendingFrameToProcess: PendingInferenceFrame?
        frameLock.lock()
        if let pendingFrame = pendingInferenceFrame {
            pendingInferenceFrame = nil
            preferImmediateNextInference = false
            lastProcessTime = Date()
            pendingFrameToProcess = pendingFrame
        } else {
            isProcessing = false
            if preferImmediateNextInference {
                lastProcessTime = .distantPast
                preferImmediateNextInference = false
            }
        }
        frameLock.unlock()
        guard let pendingFrameToProcess else { return }
        detectionQueue.async { [weak self] in
            self?.processFrame(
                pendingFrameToProcess.pixelBuffer,
                arDepthSnapshot: pendingFrameToProcess.arDepthSnapshot
            )
        }
    }

    private func storePendingInferenceFrame(
        pixelBuffer: CVPixelBuffer,
        arDepthSnapshot: FurnitureFitARDepthSnapshot?,
        source: PendingInferenceSource
    ) {
        frameLock.lock()
        pendingInferenceFrame = PendingInferenceFrame(
            pixelBuffer: pixelBuffer,
            arDepthSnapshot: arDepthSnapshot,
            source: source
        )
        preferImmediateNextInference = true
        frameLock.unlock()
    }

    private func copyPixelBuffer(_ src: CVPixelBuffer) -> CVPixelBuffer? {
        let srcWidth = CVPixelBufferGetWidth(src)
        let srcHeight = CVPixelBufferGetHeight(src)
        let pixelFormat = CVPixelBufferGetPixelFormatType(src)
        var dstBuffer: CVPixelBuffer?
        let attrs: [CFString: Any] = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true,
            kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary,
        ]
        guard CVPixelBufferCreate(
            kCFAllocatorDefault,
            srcWidth,
            srcHeight,
            pixelFormat,
            attrs as CFDictionary,
            &dstBuffer
        ) == kCVReturnSuccess,
        let dst = dstBuffer else {
            return nil
        }

        CVPixelBufferLockBaseAddress(src, .readOnly)
        CVPixelBufferLockBaseAddress(dst, [])
        defer {
            CVPixelBufferUnlockBaseAddress(src, .readOnly)
            CVPixelBufferUnlockBaseAddress(dst, [])
        }

        guard let srcBase = CVPixelBufferGetBaseAddress(src),
              let dstBase = CVPixelBufferGetBaseAddress(dst) else {
            return nil
        }

        let srcBytesPerRow = CVPixelBufferGetBytesPerRow(src)
        let dstBytesPerRow = CVPixelBufferGetBytesPerRow(dst)
        let rowBytesToCopy = min(srcBytesPerRow, dstBytesPerRow)
        for row in 0..<srcHeight {
            memcpy(
                dstBase.advanced(by: row * dstBytesPerRow),
                srcBase.advanced(by: row * srcBytesPerRow),
                rowBytesToCopy
            )
        }
        return dst
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

    private var monitorLikeClassIds: Set<Int> {
        [1063, 2675, 4105]
    }

    private var supportingTableClassIds: Set<Int> {
        [1061, 1301, 1325, 1503, 1885, 2324, 2836, 4564]
    }

    private func pickSupportingTableForMonitorScene(
        primary: UnionDet,
        primaryIdx: Int,
        candidates: [UnionDet]
    ) -> UnionDet? {
        guard includeSupportingTableForMonitorScene else { return nil }
        guard monitorLikeClassIds.contains(primary.classIdx) else { return nil }

        let primaryLeft = primary.x - primary.w * 0.5
        let primaryRight = primary.x + primary.w * 0.5
        let primaryBottom = primary.y + primary.h * 0.5
        let primaryArea = max(primary.w * primary.h, 1e-3)

        var bestCandidate: UnionDet?
        var bestScore: Float = -1

        for (idx, candidate) in candidates.enumerated() {
            if idx == primaryIdx { continue }
            if !supportingTableClassIds.contains(candidate.classIdx) { continue }

            let candidateLeft = candidate.x - candidate.w * 0.5
            let candidateRight = candidate.x + candidate.w * 0.5
            let candidateTop = candidate.y - candidate.h * 0.5
            let overlapWidth = max(0, min(primaryRight, candidateRight) - max(primaryLeft, candidateLeft))
            let horizontalOverlapRatio = overlapWidth / max(1e-3, min(primary.w, candidate.w))
            if horizontalOverlapRatio < 0.35 { continue }

            if candidate.y <= primary.y { continue }

            let verticalGap = candidateTop - primaryBottom
            if verticalGap < -primary.h * 0.20 || verticalGap > primary.h * 0.60 { continue }

            let widthRatio = candidate.w / max(primary.w, 1e-3)
            if widthRatio < 0.75 || widthRatio > 5.0 { continue }

            let areaRatio = (candidate.w * candidate.h) / primaryArea
            if areaRatio < 0.50 || areaRatio > 12.0 { continue }

            let closenessTerm = 1.0 - min(1.0, abs(verticalGap) / max(primary.h * 0.60, 1e-3))
            let score = candidate.confidence * horizontalOverlapRatio * max(0.1, closenessTerm)

            if score > bestScore {
                bestScore = score
                bestCandidate = candidate
            }
        }

        if debugMode, let bestCandidate {
            logDebug(
                "   🪑 SUPPORT TABLE: \(className(bestCandidate.classIdx)) " +
                "conf=\(String(format: "%.2f", bestCandidate.confidence)) " +
                "size=\(Int(bestCandidate.w))x\(Int(bestCandidate.h))"
            )
        }

        return bestCandidate
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

        if metalCIContext == nil {
            metalCIContext = CIContext(mtlDevice: device, options: [.cacheIntermediates: false])
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

    private func makeCGImage(from texture: MTLTexture, width: Int, height: Int) -> CGImage? {
        guard let ciContext = metalCIContext else { return nil }
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let ciImage = CIImage(
            mtlTexture: texture,
            options: [CIImageOption.colorSpace: colorSpace]
        ) else {
            return nil
        }
        let rect = CGRect(x: 0, y: 0, width: width, height: height)
        return ciContext.createCGImage(ciImage, from: rect)
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
        let settingsArOn = AppStateManager.shared.qualitySettings.enableArAssistedFurnitureSizing
        let arOn = settingsArOn
            && isUsingARCameraPath
            && arAssistedScaleValid
        let assistedScale: CGFloat = arOn ? autoScaleFromAR : 1.0
        let product = autoScaleFromRoom * assistedScale * userPinchScale
        let clamped = min(max(product, minCombinedOverlayScale), maxCombinedOverlayScale)
        let relaxedViewport = userPinchScale <= 1.15 || clamped <= 1.15
        maskImageView.contentMode = relaxedViewport ? .scaleAspectFit : .scaleAspectFill
        maskImageView.clipsToBounds = !relaxedViewport
        var clampedY = clamped
        if let displayedHeightMeters = latestDisplayedFurnitureHeightMeters,
           displayedHeightMeters > roomHeightMeters,
           bounds.height > 1,
           primaryBboxInView.height > 1 {
            let maxScreenHeight = bounds.height * 0.60
            let maxAllowedScaleY = maxScreenHeight / primaryBboxInView.height
            clampedY = min(clampedY, maxAllowedScaleY)
        }
        maskImageView.transform = CGAffineTransform(scaleX: clamped, y: clampedY)

        if debugMode {
            let wantAR = settingsArOn && isUsingARCameraPath && !userLockedAssistedOverlayScale
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
                    "📐 [OVERLAY] assist=\(assistedLabel) room=\(String(format: "%.3f", autoScaleFromRoom)) ar=\(String(format: "%.3f", autoScaleFromAR)) pinch=\(String(format: "%.3f", userPinchScale)) → combined=(x:\(String(format: "%.3f", clamped)) y:\(String(format: "%.3f", clampedY))) wantAR=\(wantAR) arValid=\(arAssistedScaleValid) arMisses=\(arAssistedConsecutiveMisses)"
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

        let wantAR = AppStateManager.shared.qualitySettings.enableArAssistedFurnitureSizing
            && isUsingARCameraPath
            && !userLockedAssistedOverlayScale

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
        latestDisplayedFurnitureHeightMeters = nil
        userPinchScale = 1.0
        userLockedAssistedOverlayScale = false
        lastOverlayPrimaryClassIdx = -1
        primaryBboxInView = .zero
        maskImageView.contentMode = .scaleAspectFill
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
        loadBlacklist()

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

        if !didRegisterArAssistedSettingObserver {
            NotificationCenter.default.addObserver(
                self,
                selector: #selector(arAssistedFurnitureSizingSettingChanged),
                name: .furnitureFitArAssistedSettingChanged,
                object: nil
            )
            didRegisterArAssistedSettingObserver = true
        }

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
        if didRegisterArAssistedSettingObserver {
            NotificationCenter.default.removeObserver(self, name: .furnitureFitArAssistedSettingChanged, object: nil)
            didRegisterArAssistedSettingObserver = false
        }
        NotificationCenter.default.removeObserver(self, name: UIDevice.orientationDidChangeNotification, object: nil)
        UIDevice.current.endGeneratingDeviceOrientationNotifications()

        DispatchQueue.main.async {
            self.arSession.pause()
            self.isUsingARCameraPath = false
        }
        DispatchQueue.global(qos: .userInitiated).async {
            if self.captureSession.isRunning {
                self.captureSession.stopRunning()
            }
        }
        hasFirstDetection = false
        segmentationCompletedOnceThisSession = false
        DispatchQueue.main.async {
            self.progressContainer.isHidden = true
            self.progressContainer.alpha = 1
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

    @objc private func arAssistedFurnitureSizingSettingChanged() {
        DispatchQueue.main.async { [weak self] in
            self?.requestCameraPermissionAndStart()
        }
    }

    private func requestCameraPermissionAndStart() {
        let wantAR = AppStateManager.shared.qualitySettings.enableArAssistedFurnitureSizing
            && FurnitureFitARSupport.isWorldTrackingSupported

        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            if wantAR {
                DispatchQueue.main.async { self.startArCameraPathIfNeeded() }
            } else {
                startClassicCameraPathIfNeeded()
            }
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { granted in
                guard granted else { return }
                if wantAR {
                    DispatchQueue.main.async { self.startArCameraPathIfNeeded() }
                } else {
                    self.startClassicCameraPathIfNeeded()
                }
            }
        default: break
        }
    }

    /// ARKit camera + world tracking (FurnitureFitARSupport). Falls back if session fails to start.
    private func startArCameraPathIfNeeded() {
        guard AppStateManager.shared.qualitySettings.enableArAssistedFurnitureSizing else {
            startClassicCameraPathIfNeeded()
            return
        }
        guard FurnitureFitARSupport.isWorldTrackingSupported else {
            startClassicCameraPathIfNeeded()
            return
        }
        if captureSession.isRunning {
            captureSession.stopRunning()
        }
        isUsingARCameraPath = true
        arSCNView.session = arSession
        arSession.delegate = self
        let config = FurnitureFitARSupport.makeWorldTrackingConfiguration()
        let sem = config.frameSemantics
        let hasScene = sem.contains(.sceneDepth)
        let hasSmoothed = sem.contains(.smoothedSceneDepth)
        logFurnitureFitAR(
            "platform=ios event=session sceneDepth=\(hasScene) smoothedSceneDepth=\(hasSmoothed) planeDetection=\(config.planeDetection)"
        )
        arSession.run(config, options: [.resetTracking, .removeExistingAnchors])
        if debugMode { logDebug("📷 [FurnitureFit] ARSession started for assisted sizing") }
    }

    private func startClassicCameraPathIfNeeded() {
        isUsingARCameraPath = false
        DispatchQueue.main.async {
            self.arSession.pause()
        }
        if !captureSession.isRunning {
            DispatchQueue.global(qos: .userInitiated).async {
                self.captureSession.startRunning()
            }
        }
        if debugMode { logDebug("📷 [FurnitureFit] AVCaptureSession (classic camera path)") }
    }

    // MARK: - Capture Delegate
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        // Early exit check BEFORE dispatching to avoid queuing frames unnecessarily
        let now = Date()
        var wasProcessing = false
        frameLock.lock()
        if isProcessing {
            wasProcessing = true
            preferImmediateNextInference = true
        }
        let shouldProcess = now.timeIntervalSince(lastProcessTime) >= processInterval && !isProcessing
        if shouldProcess {
            isProcessing = true
            lastProcessTime = now
        }
        frameLock.unlock()

        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return
        }
        guard shouldProcess else {
            if wasProcessing, let pendingBuffer = copyPixelBuffer(pixelBuffer) {
                storePendingInferenceFrame(
                    pixelBuffer: pendingBuffer,
                    arDepthSnapshot: nil,
                    source: .camera
                )
            }
            return
        }
        detectionQueue.async { [weak self] in self?.processFrame(pixelBuffer, arDepthSnapshot: nil) }
    }

    // MARK: - Main Processing Pipeline
    private func processFrame(_ pixelBuffer: CVPixelBuffer, arDepthSnapshot: FurnitureFitARDepthSnapshot? = nil) {
        autoreleasepool {
            processFrameInner(pixelBuffer, arDepthSnapshot: arDepthSnapshot)
        }
    }

    private func processFrameInner(_ pixelBuffer: CVPixelBuffer, arDepthSnapshot: FurnitureFitARDepthSnapshot? = nil) {
        let frameStart = Date()
        if debugMode { logMemory("FRAME START") }

        let shouldUseOnnxRuntime = furnitureFitUseOnnxRuntime && YOLOEOnnxRuntime.shared.isAvailable

        guard shouldUseOnnxRuntime || mlModel != nil else {
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

        let modelInputSize: Int
        if shouldUseOnnxRuntime {
            modelInputSize = YOLOEOnnxRuntime.shared.modelInputSize
            if debugMode {
                logDebug("📐 Using ONNX Runtime input size: \(modelInputSize)x\(modelInputSize)")
            }
        } else {
            guard let model = mlModel else {
                resetProcessingFlag()
                return
            }

            let imageInputDesc = model.modelDescription.inputDescriptionsByName["image"]
            if let imageConstraint = imageInputDesc?.imageConstraint {
                modelInputSize = imageConstraint.pixelsWide
                if debugMode {
                    logDebug("📐 Model expects input size: \(modelInputSize)x\(imageConstraint.pixelsHigh)")
                }
            } else {
                modelInputSize = 1280  // Fallback to default
                if debugMode {
                    logDebug("📐 Using default input size: \(modelInputSize)")
                }
            }
        }

        // STAGE 1: Preprocess (resize + input prep)
        let t1 = Date()
        setProgress(0.15, text: "Preprocessing…")

        let preprocessResult: (buffer: CVPixelBuffer, gain: Float, scaleX: Float, scaleY: Float, padX: Int, padY: Int, newW: Int, newH: Int)?
        if shouldUseOnnxRuntime {
            preprocessResult = resizeToExactSize(processBuffer, width: modelInputSize, height: modelInputSize)
        } else {
            preprocessResult = resizeToSquare(processBuffer, size: modelInputSize)
        }

        guard let sq = preprocessResult else {
            if debugMode { logDebug("❌ STAGE 1 FAILED: Resize to square") }
            resetProcessingFlag()
            return
        }
        let resizeGain = sq.gain
        let resizeScaleX = sq.scaleX
        let resizeScaleY = sq.scaleY
        let padX = sq.padX        // Integer - exact pixel offset
        let padY = sq.padY        // Integer - exact pixel offset
        let contentW = sq.newW    // Integer - exact content width in model space
        let contentH = sq.newH    // Integer - exact content height in model space
        let legacySizingGain = min(Float(modelInputSize) / Float(bufW), Float(modelInputSize) / Float(bufH))

        // Debug landscape letterbox parameters
        if debugMode && isLandscape {
            logDebug("🔶 LANDSCAPE: buf=\(bufW)x\(bufH), gain=\(resizeGain), scale=(\(resizeScaleX),\(resizeScaleY)), pad=(\(padX),\(padY)), content=\(contentW)x\(contentH)")
        }

        let t1End = Date()
        if debugMode {
            logDebug("⏱️ STAGE 1 - Preprocess: \(String(format: "%.2f", t1End.timeIntervalSince(t1) * 1000)) ms")
        }

        // STAGE 2: Model inference
        let t2 = Date()
        setProgress(0.40, text: "Running model…")

        let detArray: MLMultiArray
        let protoArray: MLMultiArray
        if shouldUseOnnxRuntime {
            guard let output = YOLOEOnnxRuntime.shared.run(pixelBuffer: sq.buffer) else {
                if debugMode { logDebug("❌ STAGE 2 FAILED: ONNX Runtime inference") }
                resetProcessingFlag()
                return
            }
            detArray = output.det
            protoArray = output.proto
            if debugMode { logDebug("📦 Using ONNX Runtime output tensors (output0/output1)") }
        } else {
            guard let model = mlModel else {
                resetProcessingFlag()
                return
            }

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

            guard let output = try? model.prediction(from: inputProvider) else {
                if debugMode { logDebug("❌ STAGE 2 FAILED: Model inference") }
                resetProcessingFlag()
                return
            }

            if let det = output.featureValue(for: "var_2374")?.multiArrayValue,
               let proto = output.featureValue(for: "var_2412")?.multiArrayValue {
                detArray = det
                protoArray = proto
                if debugMode { logDebug("📦 Using yoloe-11l output tensors (var_2374/var_2412)") }
            } else if let det = output.featureValue(for: "detections")?.multiArrayValue,
                      let proto = output.featureValue(for: "protos")?.multiArrayValue {
                detArray = det
                protoArray = proto
                if debugMode { logDebug("📦 Using generic output tensors (detections/protos)") }
            } else {
                if debugMode {
                    logDebug("❌ STAGE 4 FAILED: Missing output tensors")
                    let availableOutputs = output.featureNames.joined(separator: ", ")
                    logDebug("   Available outputs: \(availableOutputs)")
                }
                resetProcessingFlag()
                return
            }
        }

        let t2End = Date()
        if debugMode {
            logDebug("⏱️ STAGE 2 - Inference: \(String(format: "%.2f", t2End.timeIntervalSince(t2) * 1000)) ms")
            logMemory("AFTER INFERENCE")
        }

        // STAGE 3: Parse outputs (tensors + detections + prototypes)
        let t3 = Date()
        
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

        // Model must be float32
        guard detArray.dataType == .float32 else {
            logDebug("❌ STAGE 3 FAILED: Model must output float32, got \(detArray.dataType)")
            resetProcessingFlag()
            return
        }

        setProgress(0.55, text: "Parsing detections…")

        let allDets = YoloEDetectionParser.parseDetections(
            detArray: detArray,
            confidenceThreshold: confidenceThreshold,
            classBlacklist: clsToIgnore
        )

        
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

        // STAGE 3b: No NMS — use all raw detections for primary / multi-candidate stages.
        let candidates = allDets
        if debugMode {
            logDebug("⏱️ STAGE 3b - candidates: \(candidates.count) (no NMS)")
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

        // STAGE 4: Select primary
        // In two-stage mode: find 1280 detection matching 640's primary (same class, overlapping bbox)
        // In single-stage mode: score = conf^1.5 * area_norm^1.2 * center_term
        let t4 = Date()
        setProgress(0.65, text: "Selecting primary…")

        // Frame dimensions for normalization
        let frameW = Float(modelInputSize)
        let frameH = Float(modelInputSize)
        let frameArea = frameW * frameH
        let frameCx = frameW / 2
        let frameCy = frameH / 2

        var primaryIdx = -1
        var maxScore: Float = -1
        var edgeFallbackIdx = -1
        var edgeFallbackScore: Float = -1
        let minConf: Float = 0.15
        let minAreaNorm: Float = 0.02

        // Select primary using composite scoring: confidence and center carry equal weight.
        // Prefer boxes that stay clear of the frame border; only fall back to edge-touching
        // detections when there is no decent interior candidate.
        for (i, d) in candidates.enumerated() {
            // Light filters
            let areaNorm = (d.w * d.h) / frameArea
            if d.confidence < minConf || areaNorm < minAreaNorm { continue }

            // Center score: 1 at center, ~0 at corners.
            let dx = (d.x - frameCx) / frameCx
            let dy = (d.y - frameCy) / frameCy
            let centerDist = min(1.0, sqrt(dx * dx + dy * dy))
            let centerScore = 1.0 - centerDist

            let boxLeft = d.x - d.w * 0.5
            let boxTop = d.y - d.h * 0.5
            let boxRight = d.x + d.w * 0.5
            let boxBottom = d.y + d.h * 0.5
            let edgeMarginX = max(frameW * 0.04, 1.0)
            let edgeMarginY = max(frameH * 0.04, 1.0)
            let leftClearance = min(max(boxLeft / edgeMarginX, 0), 1)
            let topClearance = min(max(boxTop / edgeMarginY, 0), 1)
            let rightClearance = min(max((frameW - boxRight) / edgeMarginX, 0), 1)
            let bottomClearance = min(max((frameH - boxBottom) / edgeMarginY, 0), 1)
            let edgeClearanceScore = max(0.1, min(leftClearance, min(topClearance, min(rightClearance, bottomClearance))))
            let isInteriorCandidate = leftClearance >= 1.0
                && topClearance >= 1.0
                && rightClearance >= 1.0
                && bottomClearance >= 1.0

            // Confidence and center use the same exponent so they influence primary ranking equally.
            let confTerm = pow(d.confidence, 1.0)
            let areaTerm = pow(areaNorm, 0.8)
            let centerTerm = pow(max(0, centerScore), 1.0)
            let edgeTerm = pow(edgeClearanceScore, 1.0)

            let score = confTerm * areaTerm * centerTerm * edgeTerm

            if debugMode && score > 0.001 {
                let regionTag = isInteriorCandidate ? "interior" : "edge"
                logDebug("   [\(i)] \(className(d.classIdx)): conf=\(String(format: "%.2f", d.confidence)) area=\(String(format: "%.1f", areaNorm * 100))% center=\(String(format: "%.2f", centerScore)) edge=\(String(format: "%.2f", edgeClearanceScore)) \(regionTag) → score=\(String(format: "%.4f", score))")
            }

            if isInteriorCandidate && score > maxScore {
                maxScore = score
                primaryIdx = i
            } else if !isInteriorCandidate && score > edgeFallbackScore {
                edgeFallbackScore = score
                edgeFallbackIdx = i
            }
        }

        if primaryIdx < 0, edgeFallbackIdx >= 0 {
            primaryIdx = edgeFallbackIdx
            maxScore = edgeFallbackScore
        }

        if primaryIdx < 0 {
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
        let imgWidthPx = primary.w / legacySizingGain
        let imgHeightPx = primary.h / legacySizingGain

        let metersPerPixelX = roomWidthMeters / Float(bufW)
        let metersPerPixelY = roomHeightMeters / Float(bufH)
        let metersPerPixel = min(metersPerPixelX, metersPerPixelY)

        var estimatedWidth = imgWidthPx * metersPerPixel
        var estimatedHeight = imgHeightPx * metersPerPixel

        if estimatedWidth > estimatedHeight {
            swap(&estimatedWidth, &estimatedHeight)
        }

        var arHeightMeters: Float?
        if AppStateManager.shared.qualitySettings.enableArAssistedFurnitureSizing,
           isUsingARCameraPath,
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
            self.latestDisplayedFurnitureHeightMeters = arHeightMeters ?? estimatedHeight
            let estimate = FurnitureSizeEstimate(
                widthMeters: estimatedWidth,
                heightMeters: estimatedHeight,
                arHeightMeters: arHeightMeters
            )
            self.onFurnitureSizeEstimated?(estimate)
        }

        // STAGE 5: Filter candidates (5a prune, 5b bbox dedupe).
        // When disabled, composite uses primary detection only (same as early single-object behavior).
        let supportingTable = pickSupportingTableForMonitorScene(
            primary: primary,
            primaryIdx: primaryIdx,
            candidates: candidates
        )
        var kept2: [UnionDet]
        if !useMultiCandidateStage5 {
            kept2 = supportingTable.map { [primary, $0] } ?? [primary]
            if debugMode {
                let keptDescription = kept2.count > 1 ? "primary + support table" : "primary only"
                logDebug("⏱️ STAGE 5 SKIPPED (useMultiCandidateStage5=false), kept2=\(keptDescription)")
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
        if let supportingTable,
           !kept2.contains(where: { $0 == supportingTable }) {
            kept2.append(supportingTable)
        }

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

        func modelToImageX(_ modelX: Float) -> Float {
            (modelX - padXf) / resizeScaleX
        }

        func modelToImageY(_ modelY: Float) -> Float {
            (modelY - padYf) / resizeScaleY
        }

        func imageToModelX(_ imageX: Float) -> Float {
            imageX * resizeScaleX + padXf
        }

        func imageToModelY(_ imageY: Float) -> Float {
            imageY * resizeScaleY + padYf
        }

        var bx1 = Int(round(modelToImageX(ux1)))
        var by1 = Int(round(modelToImageY(uy1)))
        var bx2 = Int(round(modelToImageX(ux2)))
        var by2 = Int(round(modelToImageY(uy2)))

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

        // Compute PRIMARY bbox in image coordinates (for clipping mask later)
        // Always use 1280 model's primary bbox (yoloe-11l-seg-pf gave stable results)
        let primaryBx1 = max(0, Int(round(modelToImageX(primary.x - primary.w * 0.5))))
        let primaryBy1 = max(0, Int(round(modelToImageY(primary.y - primary.h * 0.5))))
        let primaryBx2 = min(origW, Int(round(modelToImageX(primary.x + primary.w * 0.5))))
        let primaryBy2 = min(origH, Int(round(modelToImageY(primary.y + primary.h * 0.5))))

        if debugMode {
            logDebug("   PRIMARY bbox: [\(primaryBx1),\(primaryBy1)]→[\(primaryBx2),\(primaryBy2)]")
        }

        // Match Android's mask construction: constrain prototype-mask generation to the primary bbox
        // in proto space instead of thresholding the whole proto plane and clipping only later.
        let protoScaleX = Float(modelInputSize) / Float(pW)
        let protoScaleY = Float(modelInputSize) / Float(pH)
        let primaryProtoBx1 = max(0, min(pW - 1, Int((primary.x - primary.w * 0.5) / protoScaleX)))
        let primaryProtoBy1 = max(0, min(pH - 1, Int((primary.y - primary.h * 0.5) / protoScaleY)))
        let primaryProtoBx2 = max(primaryProtoBx1 + 1, min(pW, Int((primary.x + primary.w * 0.5) / protoScaleX)))
        let primaryProtoBy2 = max(primaryProtoBy1 + 1, min(pH, Int((primary.y + primary.h * 0.5) / protoScaleY)))

        func clipMaskSmallToPrimaryProtoBBox(_ maskSmall: inout [UInt8]) {
            guard maskSmall.count == planeSize else { return }
            if primaryProtoBy1 > 0 {
                let topBytes = primaryProtoBy1 * pW
                _ = maskSmall.withUnsafeMutableBufferPointer { ptr in
                    memset(ptr.baseAddress!, 0, topBytes)
                }
            }
            if primaryProtoBy2 < pH {
                let bottomStart = primaryProtoBy2 * pW
                let bottomBytes = (pH - primaryProtoBy2) * pW
                _ = maskSmall.withUnsafeMutableBufferPointer { ptr in
                    memset(ptr.baseAddress!.advanced(by: bottomStart), 0, bottomBytes)
                }
            }
            if primaryProtoBx1 > 0 || primaryProtoBx2 < pW {
                maskSmall.withUnsafeMutableBufferPointer { ptr in
                    for y in primaryProtoBy1..<primaryProtoBy2 {
                        let rowStart = y * pW
                        if primaryProtoBx1 > 0 {
                            memset(ptr.baseAddress!.advanced(by: rowStart), 0, primaryProtoBx1)
                        }
                        if primaryProtoBx2 < pW {
                            memset(ptr.baseAddress!.advanced(by: rowStart + primaryProtoBx2), 0, pW - primaryProtoBx2)
                        }
                    }
                }
            }
        }

        func fillEnclosedHolesInPrimaryProtoBBox(_ maskSmall: inout [UInt8]) {
            guard maskSmall.count == planeSize else { return }
            let boxWidth = primaryProtoBx2 - primaryProtoBx1
            let boxHeight = primaryProtoBy2 - primaryProtoBy1
            guard boxWidth >= 3, boxHeight >= 3 else { return }

            var reachableBackground = [UInt8](repeating: 0, count: planeSize)
            var queue = [Int]()
            queue.reserveCapacity(boxWidth * 2 + boxHeight * 2)
            var readIndex = 0

            func enqueueIfBackground(_ x: Int, _ y: Int) {
                guard x >= primaryProtoBx1, x < primaryProtoBx2, y >= primaryProtoBy1, y < primaryProtoBy2 else { return }
                let idx = y * pW + x
                guard maskSmall[idx] == 0, reachableBackground[idx] == 0 else { return }
                reachableBackground[idx] = 1
                queue.append(idx)
            }

            for x in primaryProtoBx1..<primaryProtoBx2 {
                enqueueIfBackground(x, primaryProtoBy1)
                enqueueIfBackground(x, primaryProtoBy2 - 1)
            }
            for y in primaryProtoBy1..<primaryProtoBy2 {
                enqueueIfBackground(primaryProtoBx1, y)
                enqueueIfBackground(primaryProtoBx2 - 1, y)
            }

            while readIndex < queue.count {
                let idx = queue[readIndex]
                readIndex += 1
                let x = idx % pW
                let y = idx / pW
                enqueueIfBackground(x - 1, y)
                enqueueIfBackground(x + 1, y)
                enqueueIfBackground(x, y - 1)
                enqueueIfBackground(x, y + 1)
            }

            for y in primaryProtoBy1..<primaryProtoBy2 {
                let rowBase = y * pW
                for x in primaryProtoBx1..<primaryProtoBx2 {
                    let idx = rowBase + x
                    if maskSmall[idx] == 0, reachableBackground[idx] == 0 {
                        maskSmall[idx] = 255
                    }
                }
            }
        }

        // Store primary bbox in normalized coordinates (0-1) for gesture hit testing
        let normX1 = CGFloat(primaryBx1) / CGFloat(origW)
        let normY1 = CGFloat(primaryBy1) / CGFloat(origH)
        let normX2 = CGFloat(primaryBx2) / CGFloat(origW)
        let normY2 = CGFloat(primaryBy2) / CGFloat(origH)
        let bboxCenterImageX = CGFloat(primaryBx1 + primaryBx2) / 2
        let bboxCenterImageY = CGFloat(primaryBy1 + primaryBy2) / 2
        let bboxHeightImagePx = Float(max(1, primaryBy2 - primaryBy1))
        let useDebouncedArMeasurement = AppStateManager.shared.qualitySettings.enableArAssistedFurnitureSizing
            && isUsingARCameraPath
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
            if furnitureFitMatchAndroidSegmentation, let detection = detections.first {
                var maskProto = [Float](repeating: 0, count: planeSize)
                let bboxLeft = max(0, min(pW - 1, Int((detection.x - detection.w * 0.5) / protoScaleX)))
                let bboxTop = max(0, min(pH - 1, Int((detection.y - detection.h * 0.5) / protoScaleY)))
                let bboxRight = max(bboxLeft, min(pW - 1, Int((detection.x + detection.w * 0.5) / protoScaleX)))
                let bboxBottom = max(bboxTop, min(pH - 1, Int((detection.y + detection.h * 0.5) / protoScaleY)))

                for py in bboxTop...bboxBottom {
                    let rowBase = py * pW
                    for px in bboxLeft...bboxRight {
                        let pixelIndex = rowBase + px
                        var sum: Float = 0
                        let hwProto = pH * pW
                        let coeffCount = min(32, detection.coeffs.count)
                        for c in 0..<coeffCount {
                            let protoIndex = c * hwProto + pixelIndex
                            sum += detection.coeffs[c] * planes[protoIndex]
                        }
                        let sigmoidValue = 1.0 / (1.0 + exp(-sum))
                        if sigmoidValue > maskProto[pixelIndex] {
                            maskProto[pixelIndex] = sigmoidValue
                        }
                    }
                }

                var maskFull = [UInt8](repeating: 0, count: origW * origH)
                var positiveCount = 0
                for y in 0..<origH {
                    let modelY = imageToModelY(Float(y))
                    let protoY = max(0, min(pH - 1, Int(modelY / protoScaleY)))
                    let rowBase = y * origW
                    for x in 0..<origW {
                        let modelX = imageToModelX(Float(x))
                        let protoX = max(0, min(pW - 1, Int(modelX / protoScaleX)))
                        if maskProto[protoY * pW + protoX] > 0.5 {
                            maskFull[rowBase + x] = 255
                            positiveCount += 1
                        }
                    }
                }

                var protoMaskBinary = [UInt8](repeating: 0, count: planeSize)
                for i in 0..<planeSize where maskProto[i] > 0.5 {
                    protoMaskBinary[i] = 255
                }
                return (maskFull, positiveCount, protoMaskBinary)
            }

            // Compute per-pixel max logits across detections
            var maxLogits = [Float](repeating: -Float.greatestFiniteMagnitude, count: planeSize)

            let M = BLASInt(planeSize)
            let K = BLASInt(32)
            let alpha: Float = 1
            let beta: Float = 0

            // If list is small, SGEMV + vmax is usually faster than SGEMM + per-pixel reductions.
            let smallN = detections.count <= 8

            if smallN {
                var tmp = [Float](repeating: 0, count: planeSize)

                for d in detections {
                    // tmp = planes^T * coeffs using transposed SGEMV (no matrix copy needed)
                    planes.withUnsafeBufferPointer { planesPtr in
                        d.coeffs.withUnsafeBufferPointer { xPtr in
                            tmp.withUnsafeMutableBufferPointer { yPtr in
                                blas_sgemv_rowmajor_trans(m: K, n: M, alpha: alpha, A: planesPtr.baseAddress!, lda: M, x: xPtr.baseAddress!, incx: 1, beta: beta, y: yPtr.baseAddress!, incy: 1)
                            }
                        }
                    }

                    // maxLogits = max(maxLogits, tmp) (vectorized)
                    maxLogits.withUnsafeMutableBufferPointer { mPtr in
                        tmp.withUnsafeBufferPointer { tPtr in
                            vDSP_vmax(mPtr.baseAddress!, 1, tPtr.baseAddress!, 1,
                                      mPtr.baseAddress!, 1, vDSP_Length(planeSize))
                        }
                    }
                }
            } else {
                // Larger N: keep SGEMM batching (your current approach), but avoid tiny per-pixel vDSP_maxv calls.
                let batchSize = 64
                var bStart = 0

                while bStart < detections.count {
                    let bEnd = min(detections.count, bStart + batchSize)
                    let Bn = bEnd - bStart
                    let N = BLASInt(Bn)

                    // B is K x N in row-major layout as (k major, n minor): B[k*N + j]
                    var B = [Float](repeating: 0, count: 32 * Bn)
                    for j in 0..<Bn {
                        let coeffs = detections[bStart + j].coeffs
                        for k in 0..<32 { B[k * Bn + j] = coeffs[k] }
                    }

                    // C is M x N (row-major), each row is contiguous length N
                    var C = [Float](repeating: 0, count: planeSize * Bn)

                    // C = planes^T * B using transposed SGEMM (no matrix copy needed)
                    planes.withUnsafeBufferPointer { planesPtr in
                        B.withUnsafeBufferPointer { bPtr in
                            C.withUnsafeMutableBufferPointer { cPtr in
                                let lda = M  // leading dim of planes (32 x planeSize)
                                let ldb = N
                                let ldc = N
                                blas_sgemm_rowmajor_transA(m: M, n: N, k: K, alpha: alpha, A: planesPtr.baseAddress!, lda: lda, B: bPtr.baseAddress!, ldb: ldb, beta: beta, C: cPtr.baseAddress!, ldc: ldc)
                            }
                        }
                    }

                    // Reduce C row-wise into maxLogits (tight loop; N is small-ish, so a simple loop is fine)
                    C.withUnsafeBufferPointer { cPtr in
                        maxLogits.withUnsafeMutableBufferPointer { mPtr in
                            for px in 0..<planeSize {
                                let row = cPtr.baseAddress!.advanced(by: px * Bn)
                                var localMax = row[0]
                                if Bn > 1 {
                                    for j in 1..<Bn { if row[j] > localMax { localMax = row[j] } }
                                }
                                if localMax > mPtr[px] { mPtr[px] = localMax }
                            }
                        }
                    }

                    bStart = bEnd
                }
            }

            // Stage 14: Threshold -> build maskSmall and positive count
            var maskSmall = [UInt8](repeating: 0, count: planeSize)
            var positiveCount = 0
            for py in primaryProtoBy1..<primaryProtoBy2 {
                let rowBase = py * pW
                for px in primaryProtoBx1..<primaryProtoBx2 {
                    let index = rowBase + px
                    if maxLogits[index] > 0.0 {
                        maskSmall[index] = 255
                        positiveCount += 1
                    }
                }
            }

            clipMaskSmallToPrimaryProtoBBox(&maskSmall)
            var maskSmallForUpscale: [UInt8] = furnitureFitUseMorphologicalCloseMask
                ? morphologicalBinaryClose3x3Planar8(mask: maskSmall, width: pW, height: pH)
                : maskSmall
            clipMaskSmallToPrimaryProtoBBox(&maskSmallForUpscale)
            fillEnclosedHolesInPrimaryProtoBBox(&maskSmallForUpscale)
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
            if furnitureFitMatchAndroidSegmentation {
                return buildFullMask(from: detections)
            }
            guard let mm = metalMaskLogic else {
                return buildFullMask(from: detections)
            }
            let detCount = detections.count
            if detCount == 0 {
                return ([UInt8](repeating: 0, count: origW * origH), 0, [UInt8](repeating: 0, count: planeSize))
            }
            // Flatten coeffs (detCount x 32) row-major
            var coeffFlat = [Float](repeating: 0, count: detCount * 32)
            for j in 0..<detCount {
                let c = detections[j].coeffs
                // Safety: handle models that output !=32 coeffs (keep your original guard behavior)
                if c.count >= 32 {
                    for k in 0..<32 { coeffFlat[j*32 + k] = c[k] }
                } else {
                    for k in 0..<c.count { coeffFlat[j*32 + k] = c[k] }
                }
            }
            // planes is [Float] length 32*planeSize in the current scope (same as CPU path)
            var maskSmall = mm.buildMaskSmall(planes: planes, coeffs: coeffFlat, planeSize: planeSize, detCount: detCount)
            clipMaskSmallToPrimaryProtoBBox(&maskSmall)
            if furnitureFitUseMorphologicalCloseMask {
                maskSmall = morphologicalBinaryClose3x3Planar8(mask: maskSmall, width: pW, height: pH)
            }
            clipMaskSmallToPrimaryProtoBBox(&maskSmall)
            fillEnclosedHolesInPrimaryProtoBBox(&maskSmall)
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

        // 15d: Always clear mask outside the primary detection bbox.
        let clipPx1 = primaryBx1
        let clipPy1 = primaryBy1
        let clipPx2 = primaryBx2
        let clipPy2 = primaryBy2
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

        if !furnitureFitMatchAndroidSegmentation,
           let device = metalDevice,
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
                        var bx1_u = UInt32(primaryBx1)
                        var by1_u = UInt32(primaryBy1)
                        var bx2_u = UInt32(primaryBx2)
                        var by2_u = UInt32(primaryBy2)
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
                        composedImage = makeCGImage(from: out, width: origW, height: origH)
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
                        var bx1_u = UInt32(primaryBx1)
                        var by1_u = UInt32(primaryBy1)
                        var bx2_u = UInt32(primaryBx2)
                        var by2_u = UInt32(primaryBy2)
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
                        composedImage = makeCGImage(from: out, width: origW, height: origH)
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
                        composedImage = makeCGImage(from: out, width: origW, height: origH)
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
            for y in 0..<origH {
                let outRow = y * origW * 4
                let origRow = y * CVPixelBufferGetBytesPerRow(processBuffer)
                for x in 0..<origW {
                    let outIdx = outRow + x * 4
                    if x < primaryBx1 || x >= primaryBx2 || y < primaryBy1 || y >= primaryBy2 {
                        outBase[outIdx+3] = 0
                        continue
                    }
                    let m = maskForComposite[y * origW + x]
                    if m > 0 {
                        let origIdx = origRow + x * 4
                        let a = Float(m) / 255.0
                        // Source is BGRA, destination is premultiplied RGBA
                        outBase[outIdx+0] = UInt8(min(255, Float(origBase[origIdx+2]) * a + 0.5))
                        outBase[outIdx+1] = UInt8(min(255, Float(origBase[origIdx+1]) * a + 0.5))
                        outBase[outIdx+2] = UInt8(min(255, Float(origBase[origIdx+0]) * a + 0.5))
                        outBase[outIdx+3] = m
                    } else {
                        outBase[outIdx+3] = 0
                    }
                }
            }
            CVPixelBufferUnlockBaseAddress(processBuffer, .readOnly)
            composedImage = ctx.makeImage()
        }

        let t_comp = Date().timeIntervalSince(compStart) * 1000
        if debugMode {
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
                let dx1 = Int(round(modelToImageX(d.x - d.w * 0.5)))
                let dy1 = Int(round(modelToImageY(d.y - d.h * 0.5)))
                let dx2 = Int(round(modelToImageX(d.x + d.w * 0.5)))
                let dy2 = Int(round(modelToImageY(d.y + d.h * 0.5)))

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
            let pDx1 = Int(round(modelToImageX(primary.x - primary.w * 0.5)))
            let pDy1 = Int(round(modelToImageY(primary.y - primary.h * 0.5)))
            let pDx2 = Int(round(modelToImageX(primary.x + primary.w * 0.5)))
            let pDy2 = Int(round(modelToImageY(primary.y + primary.h * 0.5)))
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

        if let finalCtx = ctx, let img = finalCtx.makeImage() {
            composedImage = img
        }

        // Present result - handle rotation for display
        // For landscape rooms: no rotation needed since UI is also landscape.
        // For portrait rooms with a landscape buffer on the **classic camera path**, rotate 90° to portrait.
        // AR path (`isUsingARCameraPath == true`) keeps the buffer as-is (segmentation and labels stay aligned
        // even if the whole image is visually tilted, which was the last known-good behavior).
        DispatchQueue.main.async {
            if var cgImg = composedImage {
                if self.lockedOrientation == .landscape {
                    // Landscape room on landscape UI: no rotation needed
                    // The mask is already in landscape orientation matching the screen
                } else if isLandscape && !self.isUsingARCameraPath {
                    // Portrait room but landscape buffer on classic camera path: rotate back for portrait display
                    let rotatedImg = self.rotateCGImage90(cgImg, clockwise: true)
                    cgImg = rotatedImg ?? cgImg
                }
                self.maskImageView.image = UIImage(cgImage: cgImg)
                // Overlay scale (room × AR × pinch) is updated on main when primary bbox is set each frame.
            }
        }
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

    // MARK: - Parse Prototypes
    // FIXED: Reuses instance-level buffers to prevent ~26MB allocation per frame
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

        // Model must be float32
        guard proto.dataType == .float32 else { return nil }

        // FIXED: Reuse instance-level buffers instead of allocating per frame
        if protoRawFloats.count != total {
            protoRawFloats = [Float](repeating: 0, count: total)
        }
        if protoPlanes.count != count * planeSize {
            protoPlanes = [Float](repeating: 0, count: count * planeSize)
        }

        memcpy(&protoRawFloats, proto.dataPointer, total * MemoryLayout<Float>.size)

        if cIdx == 0 {
            memcpy(&protoPlanes, protoRawFloats, count * planeSize * MemoryLayout<Float>.size)
        } else if cIdx == 2 {
            for y in 0..<h {
                for x in 0..<w {
                    let baseHW = (y * w + x) * count
                    let dstBase = y * w + x
                    for k in 0..<count {
                        protoPlanes[k * planeSize + dstBase] = protoRawFloats[baseHW + k]
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
    private func resizeToSquare(_ src: CVPixelBuffer, size: Int) -> (buffer: CVPixelBuffer, gain: Float, scaleX: Float, scaleY: Float, padX: Int, padY: Int, newW: Int, newH: Int)? {
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

        memset(dstBase, 128, size * size * 4)

        var srcBuffer = vImage_Buffer(data: srcBase, height: vImagePixelCount(srcH), width: vImagePixelCount(srcW), rowBytes: CVPixelBufferGetBytesPerRow(src))
        let dstPtr = dstBase.assumingMemoryBound(to: UInt8.self)
        let dstRowBytes = CVPixelBufferGetBytesPerRow(dst)
        let offsetPtr = dstPtr.advanced(by: padY * dstRowBytes + padX * 4)
        var dstBuffer = vImage_Buffer(data: offsetPtr, height: vImagePixelCount(newH), width: vImagePixelCount(newW), rowBytes: dstRowBytes)

        guard vImageScale_ARGB8888(&srcBuffer, &dstBuffer, nil, vImage_Flags(0)) == kvImageNoError else { return nil }

        return (buffer: dst, gain: gain, scaleX: gain, scaleY: gain, padX: padX, padY: padY, newW: newW, newH: newH)
    }

    private func resizeToExactSize(_ src: CVPixelBuffer, width: Int, height: Int) -> (buffer: CVPixelBuffer, gain: Float, scaleX: Float, scaleY: Float, padX: Int, padY: Int, newW: Int, newH: Int)? {
        CVPixelBufferLockBaseAddress(src, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(src, .readOnly) }

        let srcW = CVPixelBufferGetWidth(src)
        let srcH = CVPixelBufferGetHeight(src)
        guard srcW > 0, srcH > 0 else { return nil }

        if cachedSquareBuffer == nil || cachedSquareSize != max(width, height) || CVPixelBufferGetWidth(cachedSquareBuffer!) != width || CVPixelBufferGetHeight(cachedSquareBuffer!) != height {
            var newBuffer: CVPixelBuffer?
            guard CVPixelBufferCreate(nil, width, height, kCVPixelFormatType_32BGRA, nil, &newBuffer) == kCVReturnSuccess,
                  let buffer = newBuffer else { return nil }
            cachedSquareBuffer = buffer
            cachedSquareSize = max(width, height)
        }

        guard let dst = cachedSquareBuffer else { return nil }
        CVPixelBufferLockBaseAddress(dst, [])
        defer { CVPixelBufferUnlockBaseAddress(dst, []) }

        guard let srcBase = CVPixelBufferGetBaseAddress(src),
              let dstBase = CVPixelBufferGetBaseAddress(dst) else { return nil }

        var srcBuffer = vImage_Buffer(data: srcBase, height: vImagePixelCount(srcH), width: vImagePixelCount(srcW), rowBytes: CVPixelBufferGetBytesPerRow(src))
        var dstBuffer = vImage_Buffer(data: dstBase, height: vImagePixelCount(height), width: vImagePixelCount(width), rowBytes: CVPixelBufferGetBytesPerRow(dst))
        guard vImageScale_ARGB8888(&srcBuffer, &dstBuffer, nil, vImage_Flags(0)) == kvImageNoError else { return nil }

        let scaleX = Float(width) / Float(srcW)
        let scaleY = Float(height) / Float(srcH)
        return (buffer: dst, gain: min(scaleX, scaleY), scaleX: scaleX, scaleY: scaleY, padX: 0, padY: 0, newW: width, newH: height)
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

// MARK: - ARSessionDelegate (ARKit camera for FurnitureFit when AR-assisted sizing is on)
extension FurnitureFitContainerView {
    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        guard isUsingARCameraPath else { return }
        let now = Date()
        frameLock.lock()
        if isProcessing {
            preferImmediateNextInference = true
        }
        let shouldProcess = now.timeIntervalSince(lastProcessTime) >= processInterval && !isProcessing
        if shouldProcess {
            isProcessing = true
            lastProcessTime = now
        }
        frameLock.unlock()
        guard shouldProcess else {
            return
        }
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

