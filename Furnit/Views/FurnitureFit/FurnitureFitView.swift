// FurnitureFitView.swift
// Single-stage: detect → primary selection → mask overlap filter → union mask → cutout
// With timing at every stage

import SwiftUI
import UIKit
import CoreML
import Accelerate
import AVFoundation
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

@inline(__always)
fileprivate func blas_sgemm_rowmajor(m: BLASInt, n: BLASInt, k: BLASInt, alpha: Float, A: UnsafePointer<Float>, lda: BLASInt, B: UnsafePointer<Float>, ldb: BLASInt, beta: Float, C: UnsafeMutablePointer<Float>, ldc: BLASInt) {
    BlasSgemm(true, false, false, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
}


// MARK: - Metal Mask Logic (GPU)
// Computes maskSmall (prototype resolution) on GPU: max over detections of dot(coeffs, prototypes) per pixel.
// Output is UInt8 mask (0 or 255) using the same thresholding logic as CPU: maxLogit > 0 => 255.
final class MetalMaskLogic {
    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let pipelineMaxMask: MTLComputePipelineState

    init(device: MTLDevice) {
        self.device = device
        self.queue = device.makeCommandQueue()!
        let library = device.makeDefaultLibrary()!
        self.pipelineMaxMask = try! device.makeComputePipelineState(function: library.makeFunction(name: "sp_maxMaskFromPrototypes")!)
    }

    func buildMaskSmall(planes: [Float], coeffs: [Float], planeSize: Int, detCount: Int) -> [UInt8] {
        precondition(planes.count == 32 * planeSize, "planes size mismatch")
        precondition(coeffs.count == detCount * 32, "coeffs size mismatch")

        let planesBytes = planes.count * MemoryLayout<Float>.size
        let coeffBytes = coeffs.count * MemoryLayout<Float>.size
        let outBytes = planeSize * MemoryLayout<UInt8>.size

        let planesBuf = device.makeBuffer(bytes: planes, length: planesBytes, options: .storageModeShared)!
        let coeffBuf = device.makeBuffer(bytes: coeffs, length: coeffBytes, options: .storageModeShared)!
        let outBuf = device.makeBuffer(length: outBytes, options: .storageModeShared)!

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

// MARK: - SwiftUI Wrapper
struct FurnitureFitViewSwiftUI: UIViewRepresentable {
    let mlModel: MLModel?
    var processInterval: TimeInterval = 0.1
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

// MARK: - Detection Struct
// Note: Using FurnitureFitDetection from FurnitureFitUtils.swift (single source of truth)
// typealias kept for minimal code changes
typealias UnionDet = FurnitureFitDetection

// MARK: - Main Container View
final class FurnitureFitContainerView: UIView, AVCaptureVideoDataOutputSampleBufferDelegate, UIGestureRecognizerDelegate {
    
    // MARK: Config
    var processInterval: TimeInterval = 0.1
    var confidenceThreshold: Float = 0.1
    var useBilinearUpscaling: Bool = false
    
    // Debug mode - read from settings
    var debugMode: Bool {
        return AppStateManager.shared.qualitySettings.debugMode
    }

    // MARK: - Ignored Classes (loaded from blacklist.json)
    private lazy var clsToIgnore: Set<Int> = {
        guard let url = Bundle.main.url(forResource: "blacklist", withExtension: "json"),
              let data = try? Data(contentsOf: url),
              let dict = try? JSONSerialization.jsonObject(with: data) as? [String: String] else {
            if debugMode { logDebug("⚠️ Failed to load blacklist.json") }
            return []
        }
        let blacklistSet = Set(dict.keys.compactMap { Int($0) })
        if debugMode { logDebug("✅ Loaded \(blacklistSet.count) blacklisted classes") }
        return blacklistSet
    }()

    // MARK: Camera
    private let captureSession = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let sampleQueue = DispatchQueue(label: "com.furnit.sample", qos: .userInitiated)

    // MARK: UI
    private let previewLayer = AVCaptureVideoPreviewLayer()
    private let maskImageView: UIImageView = {
        let iv = UIImageView()
        iv.contentMode = .scaleAspectFit
        iv.backgroundColor = .clear
        iv.isOpaque = false
        iv.clipsToBounds = true
        return iv
    }()
    
    // MARK: Progress UI
    private let progressView: UIProgressView = {
        let pv = UIProgressView(progressViewStyle: .default)
        pv.translatesAutoresizingMaskIntoConstraints = false
        pv.tintColor = .systemGreen
        pv.trackTintColor = UIColor(white: 1.0, alpha: 0.3)
        pv.isHidden = true
        return pv
    }()
    
    private let progressLabel: UILabel = {
        let l = UILabel()
        l.translatesAutoresizingMaskIntoConstraints = false
        l.textColor = .white
        l.font = .systemFont(ofSize: 14, weight: .medium)
        l.textAlignment = .center
        l.isHidden = true
        l.backgroundColor = UIColor.black.withAlphaComponent(0.4)
        l.layer.cornerRadius = 10
        l.clipsToBounds = true
        return l
    }()
    
    private var hasFirstDetection = false
    private var currentScale: CGFloat = 1.0
    
    // MARK: - Metal
    private var metalDevice: MTLDevice? = MTLCreateSystemDefaultDevice()
    private var metalCommandQueue: MTLCommandQueue? {
        metalDevice?.makeCommandQueue()
    }
    private var metalLibrary: MTLLibrary? {
        metalDevice?.makeDefaultLibrary()
    }
    private var compositePipeline: MTLComputePipelineState? = nil
    private var fusedMaskCompositePipeline: MTLComputePipelineState? = nil



// GPU mask builder (optional)
private lazy var metalMaskLogic: MetalMaskLogic? = {
    guard let d = metalDevice else { return nil }
    return MetalMaskLogic(device: d)
}()
    // MARK: Model & State
    private var mlModel: MLModel?
    private let detectionQueue = DispatchQueue(label: "com.furnit.detection", qos: .userInitiated)
    private var lastProcessTime = Date.distantPast
    private var isProcessing = false
    private let frameLock = NSLock() // Protects lastProcessTime and isProcessing for early-exit checks

    // MARK: Orientation
    private var currentDeviceOrientation: UIDeviceOrientation = .portrait

    /// Thread-safe reset of isProcessing flag
    private func resetProcessingFlag() {
        frameLock.lock()
        isProcessing = false
        frameLock.unlock()
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
        addSubview(maskImageView)
        maskImageView.translatesAutoresizingMaskIntoConstraints = false
        NSLayoutConstraint.activate([
            maskImageView.topAnchor.constraint(equalTo: topAnchor),
            maskImageView.bottomAnchor.constraint(equalTo: bottomAnchor),
            maskImageView.leadingAnchor.constraint(equalTo: leadingAnchor),
            maskImageView.trailingAnchor.constraint(equalTo: trailingAnchor)
        ])
        
        addSubview(progressView)
        addSubview(progressLabel)
        NSLayoutConstraint.activate([
            progressView.leadingAnchor.constraint(equalTo: leadingAnchor, constant: 40),
            progressView.trailingAnchor.constraint(equalTo: trailingAnchor, constant: -40),
            progressView.topAnchor.constraint(equalTo: safeAreaLayoutGuide.topAnchor, constant: 12),
            progressLabel.centerXAnchor.constraint(equalTo: progressView.centerXAnchor),
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
        
        setupCamera()
        setupMetal()
        if debugMode { logDebug("✅ FurnitureFitContainerView initialized") }
    }
    
    private func setupMetal() {
        guard let device = metalDevice, let library = metalLibrary else { return }
        do {
            if let fn = library.makeFunction(name: "sp_compositeMask") {
                compositePipeline = try device.makeComputePipelineState(function: fn)
            }
            if let fn2 = library.makeFunction(name: "sp_maxMaskAndComposite") {
                fusedMaskCompositePipeline = try device.makeComputePipelineState(function: fn2)
            }
        } catch {
            if debugMode { logDebug("⚠️ Metal pipeline setup failed: \(error.localizedDescription)") }
            CrashReporter.shared.report(error, context: "Metal Pipeline Setup")
        }
    }
    
    override func layoutSubviews() {
        super.layoutSubviews()
        previewLayer.frame = bounds
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
        detectionQueue.sync { self.mlModel = model }
        // Log model outputs for debugging
        if let model = model {
            let inputNames = model.modelDescription.inputDescriptionsByName.keys.joined(separator: ", ")
            let outputNames = model.modelDescription.outputDescriptionsByName.keys.joined(separator: ", ")
            logDebug("🧠 [FurnitureFit] Model set - inputs: [\(inputNames)], outputs: [\(outputNames)]")
        }
    }
    
    func startIfNeeded() {
        hasFirstDetection = false
        setProgress(0.05, text: "Starting camera…")

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

        // Set initial video rotation after camera starts
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) { [weak self] in
            guard let self = self else { return }
            self.updateVideoRotationForOrientation(self.currentDeviceOrientation)
        }
    }

    @objc private func deviceOrientationDidChange() {
        let orientation = UIDevice.current.orientation
        if orientation.isValidInterfaceOrientation {
            currentDeviceOrientation = orientation
            // Update video rotation to match device orientation for full FOV
            updateVideoRotationForOrientation(orientation)
        }
    }

    private func updateVideoRotationForOrientation(_ orientation: UIDeviceOrientation) {
        guard let conn = videoOutput.connection(with: .video) else { return }

        // Set video rotation to match device orientation for natural FOV
        switch orientation {
        case .landscapeLeft:
            conn.videoRotationAngle = 0    // Native landscape
        case .landscapeRight:
            conn.videoRotationAngle = 180  // Native landscape (flipped)
        default:
            conn.videoRotationAngle = 90   // Portrait
        }
    }

    func stop() {
        NotificationCenter.default.removeObserver(self, name: UIDevice.orientationDidChangeNotification, object: nil)
        UIDevice.current.endGeneratingDeviceOrientationNotifications()

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
        // Fixed 90° rotation - camera captures in landscape, we need portrait for model
        // The preview layer handles display orientation automatically
        if let conn = videoOutput.connection(with: .video) {
            conn.videoRotationAngle = 90
        }
        captureSession.commitConfiguration()
    }

    private func requestCameraPermissionAndStart() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            if !captureSession.isRunning {
                DispatchQueue.global(qos: .userInitiated).async {
                    self.captureSession.startRunning()
                }
            }
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { granted in
                if granted {
                    DispatchQueue.global(qos: .userInitiated).async { self.captureSession.startRunning() }
                }
            }
        default: break
        }
    }

    // MARK: - Capture Delegate
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        // Early exit check BEFORE dispatching to avoid queuing frames unnecessarily
        let now = Date()
        frameLock.lock()
        let shouldProcess = now.timeIntervalSince(lastProcessTime) >= processInterval && !isProcessing
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
        detectionQueue.async { [weak self] in self?.processFrame(pixelBuffer) }
    }

    // MARK: - Main Processing Pipeline
    private func processFrame(_ pixelBuffer: CVPixelBuffer) {
        let frameStart = Date()

        guard let model = mlModel else {
            resetProcessingFlag()
            return
        }

        // Capture orientation at start of processing
        let orientation = currentDeviceOrientation
        let isLandscape = orientation == .landscapeLeft || orientation == .landscapeRight

        if debugMode {
            logDebug("\n⏱️ ═══════════════════════════════════════════")
            logDebug("⏱️ FRAME START @ \(String(format: "%.3f", frameStart.timeIntervalSince1970))")
            logDebug("⏱️ Orientation: \(orientation.rawValue), isLandscape: \(isLandscape)")
            logDebug("⏱️ ═══════════════════════════════════════════")
        }

        // Camera now delivers correctly oriented frames via dynamic videoRotationAngle
        // No pre-rotation needed - frames match device orientation
        let processBuffer = pixelBuffer

        // Determine model's expected input size from input description
        let imageInputDesc = model.modelDescription.inputDescriptionsByName["image"]
        let modelInputSize: Int
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

        // STAGE 1: Resize to square
        let t1 = Date()
        setProgress(0.15, text: "Resizing…")

        guard let sq = resizeToSquare(processBuffer, size: modelInputSize) else {
            if debugMode { logDebug("❌ STAGE 1 FAILED: Resize to square") }
            resetProcessingFlag()
            return
        }
        let resizeGain = sq.gain
        let padX = sq.padX
        let padY = sq.padY
        
        let t1End = Date()
        if debugMode {
            logDebug("⏱️ STAGE 1 - Resize: \(String(format: "%.2f", t1End.timeIntervalSince(t1) * 1000)) ms")
        }

        // STAGE 2: Prepare model input
        let t2 = Date()
        setProgress(0.25, text: "Preprocessing…")

        // Check if model expects Image or MultiArray input
        let inputDesc = model.modelDescription.inputDescriptionsByName["image"]
        let expectsImage = inputDesc?.type == .image

        let inputProvider: MLFeatureProvider
        if expectsImage {
            // Model expects CVPixelBuffer (Image type)
            guard let imageValue = MLFeatureValue(pixelBuffer: sq.buffer) as MLFeatureValue? else {
                if debugMode { logDebug("❌ STAGE 2 FAILED: Image conversion") }
                resetProcessingFlag()
                return
            }
            guard let provider = try? MLDictionaryFeatureProvider(dictionary: ["image": imageValue]) else {
                if debugMode { logDebug("❌ STAGE 2 FAILED: Feature provider") }
                resetProcessingFlag()
                return
            }
            inputProvider = provider
        } else {
            // Model expects MLMultiArray
            guard let inputArray = pixelBufferToMLMultiArray(sq.buffer) else {
                if debugMode { logDebug("❌ STAGE 2 FAILED: MLMultiArray conversion") }
                resetProcessingFlag()
                return
            }
            guard let provider = try? MLDictionaryFeatureProvider(dictionary: ["image": inputArray]) else {
                if debugMode { logDebug("❌ STAGE 2 FAILED: Feature provider") }
                resetProcessingFlag()
                return
            }
            inputProvider = provider
        }

        let t2End = Date()
        if debugMode {
            logDebug("⏱️ STAGE 2 - Input prep (\(expectsImage ? "Image" : "MultiArray")): \(String(format: "%.2f", t2End.timeIntervalSince(t2) * 1000)) ms")
        }

        // STAGE 3: Model inference
        let t3 = Date()
        setProgress(0.40, text: "Running model…")

        guard let output = try? model.prediction(from: inputProvider) else {
            if debugMode { logDebug("❌ STAGE 3 FAILED: Model inference") }
            resetProcessingFlag()
            return
        }
        
        let t3End = Date()
        if debugMode {
            logDebug("⏱️ STAGE 3 - Inference: \(String(format: "%.2f", t3End.timeIntervalSince(t3) * 1000)) ms")
        }

        // STAGE 4: Extract tensors (handle different model output names)
        let t4 = Date()

        // Try model output names - yoloe-26l first (preferred), then yoloe-11l fallbacks
        let detArray: MLMultiArray
        let protoArray: MLMultiArray

        if let det = output.featureValue(for: "detections")?.multiArrayValue,
           let proto = output.featureValue(for: "protos")?.multiArrayValue {
            // yoloe-26l model outputs (new export with named outputs)
            detArray = det
            protoArray = proto
            if debugMode { logDebug("📦 Using yoloe-26l output tensors (detections/protos)") }
        } else if let det = output.featureValue(for: "var_2374")?.multiArrayValue,
           let proto = output.featureValue(for: "var_2412")?.multiArrayValue {
            // yoloe-11l model outputs (export with proper cv3/cv4 heads)
            detArray = det
            protoArray = proto
            if debugMode { logDebug("📦 Using yoloe-11l output tensors (var_2374/var_2412)") }
        } else if let det = output.featureValue(for: "var_2497")?.multiArrayValue,
                  let proto = output.featureValue(for: "p")?.multiArrayValue {
            // yoloe-11l model outputs (old export format)
            detArray = det
            protoArray = proto
            if debugMode { logDebug("📦 Using yoloe-11l output tensors (old format)") }
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
        let dim1 = detArray.shape[1].intValue
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

        let t4End = Date()
        if debugMode {
            logDebug("   Format detected: \(isNewFormat ? "NEW (detections×features)" : "OLD (features×anchors)")")
            logDebug("⏱️ STAGE 4 - Extract tensors: \(String(format: "%.2f", t4End.timeIntervalSince(t4) * 1000)) ms")
        }

        // STAGE 5: Copy detection tensor to float buffer
        let t5 = Date()

        let totalCount = detArray.count

        // Safety check: prevent memory allocation crash for unexpected tensor sizes
        // Standard model: ~1.2M elements for [1,144,8400]
        // Prompt-free model: ~155M elements for [1,4621,33600] - needs ~620MB
        // Allow up to 200M elements (800MB) for prompt-free models
        let maxAllowedCount = 200_000_000
        guard totalCount <= maxAllowedCount else {
            logDebug("❌ STAGE 5 FAILED: Detection tensor too large (\(totalCount) elements, max \(maxAllowedCount))")
            logDebug("   detArray shape: \(detArray.shape.map { $0.intValue })")
            logDebug("   This may indicate an incompatible model or corrupted output")
            resetProcessingFlag()
            return
        }

        let detBuf = UnsafeMutablePointer<Float>.allocate(capacity: totalCount)
        defer { detBuf.deallocate() }

        if detArray.dataType == .float32 {
            memcpy(detBuf, detArray.dataPointer, totalCount * MemoryLayout<Float>.size)
        } else if detArray.dataType == .float16 {
            let src = detArray.dataPointer.bindMemory(to: UInt16.self, capacity: totalCount)
            var srcBuf = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: src), height: 1, width: vImagePixelCount(totalCount), rowBytes: totalCount * 2)
            var dstBuf = vImage_Buffer(data: UnsafeMutableRawPointer(detBuf), height: 1, width: vImagePixelCount(totalCount), rowBytes: totalCount * 4)
            vImageConvert_Planar16FtoPlanarF(&srcBuf, &dstBuf, vImage_Flags(kvImageNoFlags))
        }

        let t5End = Date()
        if debugMode {
            logDebug("⏱️ STAGE 5 - Copy detBuf: \(String(format: "%.2f", t5End.timeIntervalSince(t5) * 1000)) ms")
        }

        // STAGE 6: Extract detections
        let t6 = Date()
        setProgress(0.55, text: "Extracting detections…")

        var allDets: [UnionDet] = []
        allDets.reserveCapacity(512)

        if isNewFormat {
            // NEW FORMAT: [1, numDetections, featuresPerDetection]
            // featuresPerDetection = 38 = 4 (bbox) + 1 (conf) + 1 (class) + 32 (mask coeffs)
            // IMPORTANT: YOLO-E outputs bbox in XYXY format (x1,y1,x2,y2), NOT XYWH!
            let numDetections = dim1
            let featuresPerDet = dim2

            // IMPORTANT: Use actual strides from MLMultiArray, not shape dimensions!
            // Shape [1, 300, 38] might have strides [14400, 48, 1] due to memory alignment
            let strides = detArray.strides.map { $0.intValue }
            let detStride = strides.count >= 2 ? strides[1] : featuresPerDet  // stride between detections
            let featStride = strides.count >= 3 ? strides[2] : 1              // stride between features

            if debugMode {
                logDebug("   NEW FORMAT: numDetections=\(numDetections), featuresPerDet=\(featuresPerDet)")
                logDebug("   Strides: \(strides), detStride=\(detStride), featStride=\(featStride)")
                // Print first detection's raw values using correct strides
                if numDetections > 0 {
                    var rawVals: [Float] = []
                    for f in 0..<min(10, featuresPerDet) {
                        let idx = 0 * detStride + f * featStride
                        rawVals.append(detBuf[idx])
                    }
                    logDebug("   First det raw[0..9] (xyxy,conf,cls,...): \(rawVals.map { String(format: "%.2f", $0) })")
                }
            }

            // Layout: [x1, y1, x2, y2, conf, class_id, coeff0..coeff31]
            // YOLO-E post-NMS export: bbox in xyxy corner format

            for detIdx in 0..<numDetections {
                // Use actual strides for memory access
                let base = detIdx * detStride

                // Read xyxy corner coordinates
                let x1 = detBuf[base + 0 * featStride]
                let y1 = detBuf[base + 1 * featStride]
                let x2 = detBuf[base + 2 * featStride]
                let y2 = detBuf[base + 3 * featStride]
                let confidence = detBuf[base + 4 * featStride]
                let classIdxFloat = detBuf[base + 5 * featStride]

                // Skip if any value is NaN or Inf
                guard x1.isFinite, y1.isFinite, x2.isFinite, y2.isFinite, confidence.isFinite, classIdxFloat.isFinite else { continue }

                // Convert xyxy to xywh (center format)
                let w = x2 - x1
                let h = y2 - y1
                let x = (x1 + x2) / 2.0  // center x
                let y = (y1 + y2) / 2.0  // center y

                // Validate ranges - confidence should be 0-1, bbox should be reasonable
                guard confidence > confidenceThreshold, confidence <= 1.0 else { continue }
                guard w > 0, h > 0, w < 2000, h < 2000 else { continue }  // Max reasonable size for 1280 input

                let classIdx = Int(classIdxFloat)

                // Bounds checking for mask coefficients
                var coeffs = [Float](repeating: 0, count: 32)
                var validCoeffs = true
                for k in 0..<32 {
                    let coeffIndex = base + (6 + k) * featStride
                    if coeffIndex < totalCount {
                        let val = detBuf[coeffIndex]
                        if val.isFinite {
                            coeffs[k] = val
                        } else {
                            if debugMode { logDebug("⚠️ Coefficient NaN/Inf for k=\(k), detIdx=\(detIdx)") }
                            validCoeffs = false
                            break
                        }
                    } else {
                        if debugMode { logDebug("⚠️ Coefficient bounds check failed for k=\(k), detIdx=\(detIdx)") }
                        validCoeffs = false
                        break
                    }
                }

                guard validCoeffs else { continue }
                guard classIdx >= 0, !clsToIgnore.contains(classIdx) else { continue }

                allDets.append(UnionDet(x: x, y: y, w: w, h: h, confidence: confidence, classIdx: classIdx, coeffs: coeffs))
            }
        } else {
            // OLD FORMAT: [1, numFeatures, numAnchors]
            let numFeatures = dim1
            let numAnchors = dim2
            let numClasses = numFeatures - 4 - 32

            if debugMode {
                logDebug("   OLD FORMAT: numFeatures=\(numFeatures), numAnchors=\(numAnchors), numClasses=\(numClasses)")
            }

            guard numFeatures >= 36, numAnchors > 0, numClasses > 0 else {
                if debugMode { logDebug("❌ STAGE 6 FAILED: Invalid tensor dims for old format") }
                resetProcessingFlag()
                return
            }

            let stride = numAnchors
            let coeffOffset = 4 + numClasses
            var tempScores = [Float](repeating: 0, count: numClasses)

            for anchor in 0..<numAnchors {
                let x = detBuf[0 * stride + anchor]
                let y = detBuf[1 * stride + anchor]
                let w = detBuf[2 * stride + anchor]
                let h = detBuf[3 * stride + anchor]

                guard x.isFinite, y.isFinite, w.isFinite, h.isFinite, w > 0, h > 0 else { continue }

                let basePtr = detBuf.advanced(by: 4 * stride + anchor)
                blas_scopy(BLASInt(numClasses), basePtr, BLASInt(stride), &tempScores, 1)

                var maxVal: Float = 0
                var maxIdx: vDSP_Length = 0
                vDSP_maxvi(tempScores, 1, &maxVal, &maxIdx, vDSP_Length(numClasses))

                let classIdx = Int(maxIdx)

                guard maxVal > confidenceThreshold, !clsToIgnore.contains(classIdx) else { continue }

                var coeffs = [Float](repeating: 0, count: 32)
                let coeffBase = detBuf.advanced(by: coeffOffset * stride + anchor)
                blas_scopy(32, coeffBase, BLASInt(stride), &coeffs, 1)

                allDets.append(UnionDet(x: x, y: y, w: w, h: h, confidence: maxVal, classIdx: classIdx, coeffs: coeffs))
            }
        }
        
        let t6End = Date()
        if debugMode {
            logDebug("⏱️ STAGE 6 - Extract detections: \(String(format: "%.2f", t6End.timeIntervalSince(t6) * 1000)) ms")
            logDebug("   raw detections: \(allDets.count)")
            // Print all detections for debugging
            for (i, d) in allDets.enumerated() {
                logDebug("   [\(i)] \(className(d.classIdx)) (id:\(d.classIdx)) conf=\(String(format: "%.2f", d.confidence)) box=(\(Int(d.x)),\(Int(d.y))) size=\(Int(d.w))x\(Int(d.h))")
            }
        }
        
        if allDets.isEmpty {
            if debugMode { logDebug("⚠️ No detections found") }
            DispatchQueue.main.async { self.maskImageView.image = nil }
            resetProcessingFlag()
            return
        }

        // STAGE 7: All detections are candidates (YOLO-E is end-to-end NMS-free)
        let candidates: [UnionDet] = allDets
        if debugMode {
            logDebug("⏱️ STAGE 7 - Candidates: \(candidates.count) dets")
        }

        // STAGE 8: Parse prototypes (moved before primary selection to enable mask-based scoring)
        let t8 = Date()
        setProgress(0.60, text: "Parsing prototypes…")

        guard let protoInfo = parsePrototypes(protoArray) else {
            if debugMode { logDebug("❌ STAGE 8 FAILED: Parse prototypes") }
            resetProcessingFlag()
            return
        }
        let planes = protoInfo.planes
        let pH = protoInfo.height
        let pW = protoInfo.width
        let planeSize = pH * pW

        let t8End = Date()
        if debugMode {
            let protoMs = String(format: "%.2f", t8End.timeIntervalSince(t8) * 1000)
            logDebug("⏱️ STAGE 8 - Parse prototypes: \(protoMs) ms, size=\(pW)x\(pH)")
        }

        // STAGE 9: Find primary using composite scoring
        // Score = conf^1.5 * area_norm^1.2 * center_term * mask_solidity_term
        let t9 = Date()
        setProgress(0.65, text: "Selecting primary…")

        // Frame dimensions for normalization
        let frameW = Float(modelInputSize)
        let frameH = Float(modelInputSize)
        let frameArea = frameW * frameH
        let frameCx = frameW / 2
        let frameCy = frameH / 2

        // Reorganize planes for BLAS: A is (planeSize x 32) row-major
        var A = [Float](repeating: 0, count: planeSize * 32)
        for ch in 0..<32 {
            for px in 0..<planeSize {
                A[px * 32 + ch] = planes[ch * planeSize + px]
            }
        }

        // Primary selection using bbox-only scoring (fast, no mask computation)
        // score = conf^1.5 * area_norm^1.2 * (0.5 + 0.5*center)
        var primaryIdx = -1
        var maxScore: Float = -1
        let minConf: Float = 0.15
        let minAreaNorm: Float = 0.02

        for (i, d) in candidates.enumerated() {
            // Light filters
            let areaNorm = (d.w * d.h) / frameArea
            if d.confidence < minConf || areaNorm < minAreaNorm { continue }

            // Center score: 1 at center, ~0 at corners
            let dx = (d.x - frameCx) / frameCx
            let dy = (d.y - frameCy) / frameCy
            let centerDist = min(1.0, sqrt(dx * dx + dy * dy))
            let centerScore = 1.0 - centerDist

            // Fast composite score (no mask computation)
            let confTerm = pow(d.confidence, 1.5)
            let areaTerm = pow(areaNorm, 1.2)
            let centerTerm = 0.5 + 0.5 * Float(centerScore)

            let score = confTerm * areaTerm * centerTerm

            if debugMode && score > 0.001 {
                logDebug("   [\(i)] \(className(d.classIdx)): conf=\(String(format: "%.2f", d.confidence)) area=\(String(format: "%.1f", areaNorm * 100))% center=\(String(format: "%.2f", centerScore)) → score=\(String(format: "%.4f", score))")
            }

            if score > maxScore {
                maxScore = score
                primaryIdx = i
            }
        }

        if primaryIdx < 0 {
            if debugMode { logDebug("   ⚠️ No valid primary candidate") }
            DispatchQueue.main.async { self.maskImageView.image = nil }
            resetProcessingFlag()
            return
        }

        let primary = candidates[primaryIdx]
        let t9End = Date()
        if debugMode {
            let primaryMs = String(format: "%.2f", t9End.timeIntervalSince(t9) * 1000)
            logDebug("⏱️ STAGE 9 - Primary selection: \(primaryMs) ms")
            logDebug("   🎯 PRIMARY[\(primaryIdx)]: \u{001B}[1m\(className(primary.classIdx))\u{001B}[0m conf=\(String(format: "%.2f", primary.confidence)) size=\(Int(primary.w))x\(Int(primary.h)) score=\(String(format: "%.4f", maxScore))")
        }

        // STAGE 10: Prune candidates BEFORE mask BLAS (Phase 3 optimization)
        // - Drop non-overlapping bboxes
        // - Drop candidates much smaller than primary (area < 5%)
        // - Drop low confidence (< 0.1)
        // - Cap to top-K (by confidence)
        let t10 = Date()

        let primaryBboxX1 = primary.x - primary.w * 0.5
        let primaryBboxY1 = primary.y - primary.h * 0.5
        let primaryBboxX2 = primary.x + primary.w * 0.5
        let primaryBboxY2 = primary.y + primary.h * 0.5
        let primaryBboxArea = primary.w * primary.h

        // Helper: check if two bboxes overlap
        func bboxesOverlap(_ d: UnionDet) -> Bool {
            let dx1 = d.x - d.w * 0.5
            let dy1 = d.y - d.h * 0.5
            let dx2 = d.x + d.w * 0.5
            let dy2 = d.y + d.h * 0.5
            return !(dx2 < primaryBboxX1 || dx1 > primaryBboxX2 || dy2 < primaryBboxY1 || dy1 > primaryBboxY2)
        }

        // Filter and collect candidates for mask BLAS
        var prunedCandidates: [(idx: Int, det: UnionDet)] = []
        let minCandConf: Float = 0.1
        let minAreaRatio: Float = 0.05  // must be at least 5% of primary area

        for (i, d) in candidates.enumerated() {
            if i == primaryIdx { continue }

            // Confidence filter
            if d.confidence < minCandConf {
                if debugMode { logDebug("   ⏭️ [\(i)] \(className(d.classIdx)): conf \(String(format: "%.2f", d.confidence)) < \(minCandConf)") }
                continue
            }

            // Size filter: drop very small candidates
            let candArea = d.w * d.h
            if candArea < primaryBboxArea * minAreaRatio {
                if debugMode { logDebug("   ⏭️ [\(i)] \(className(d.classIdx)): area \(String(format: "%.1f", candArea / primaryBboxArea * 100))% < \(Int(minAreaRatio * 100))%") }
                continue
            }

            // Bbox overlap filter: drop non-overlapping
            if !bboxesOverlap(d) {
                if debugMode { logDebug("   ⏭️ [\(i)] \(className(d.classIdx)): no bbox overlap with primary") }
                continue
            }

            prunedCandidates.append((i, d))
        }

        let t10End = Date()
        if debugMode {
            let pruneMs = String(format: "%.2f", t10End.timeIntervalSince(t10) * 1000)
            logDebug("⏱️ STAGE 10 - Prune: \(pruneMs) ms, \(candidates.count - 1) → \(prunedCandidates.count) candidates")
        }

        // STAGE 11: Batched SGEMM for all candidate logits + overlap classification
        let t11 = Date()

        // Build primary logits first (single SGEMV)
        var primaryLogits = [Float](repeating: 0, count: planeSize)
        A.withUnsafeBufferPointer { aPtr in
            primary.coeffs.withUnsafeBufferPointer { xPtr in
                primaryLogits.withUnsafeMutableBufferPointer { yPtr in
                    let m = BLASInt(planeSize)
                    let n = BLASInt(32)
                    let lda = BLASInt(32)
                    blas_sgemv_rowmajor(m: m, n: n, alpha: 1.0, A: aPtr.baseAddress!, lda: lda, x: xPtr.baseAddress!, incx: 1, beta: 0.0, y: yPtr.baseAddress!, incy: 1)
                }
            }
        }

        // Count primary mask area
        var primaryMaskArea = 0
        for i in 0..<planeSize {
            if primaryLogits[i] > 0 { primaryMaskArea += 1 }
        }

        if debugMode {
            logDebug("   📦 PRIMARY: center=(\(Int(primary.x)),\(Int(primary.y))) size=\(Int(primary.w))x\(Int(primary.h))")
            logDebug("      mask pixels: \(primaryMaskArea)")
        }

        // Overlay relation enum
        enum OverlayRelation {
            case overlay    // candidate is small and mostly inside primary (pillow on bed)
            case underlay   // candidate is big and contains primary (scene label)
            case sideBySide // barely touching
            case ambiguous  // partial overlap
        }

        func classifyRelation(candArea: Int, covPrimary: Float, covCand: Float) -> OverlayRelation {
            let ratio = Float(candArea) / Float(max(1, primaryMaskArea))
            if covCand > 0.8 && ratio < 0.5 { return .overlay }
            if covPrimary > 0.8 && ratio > 2.0 { return .underlay }
            if covPrimary < 0.2 && covCand < 0.2 { return .sideBySide }
            return .ambiguous
        }

        // Keep primary by definition
        var kept2: [UnionDet] = [primary]

        // If no candidates to check, skip SGEMM
        if prunedCandidates.isEmpty {
            if debugMode { logDebug("   No candidates to check after pruning") }
        } else {
            // Build coefficient matrix B for ALL pruned candidates: B is (32 x N) row-major
            // B[k * N + j] = coeffs[k] for candidate j
            let N = prunedCandidates.count
            var B = [Float](repeating: 0, count: 32 * N)
            for j in 0..<N {
                let coeffs = prunedCandidates[j].det.coeffs
                for k in 0..<32 { B[k * N + j] = coeffs[k] }
            }

            // SGEMM: C = A * B where A is (planeSize x 32), B is (32 x N), C is (planeSize x N)
            // C[px * N + j] = logit for pixel px, candidate j
            var C = [Float](repeating: 0, count: planeSize * N)

            A.withUnsafeBufferPointer { aPtr in
                B.withUnsafeBufferPointer { bPtr in
                    C.withUnsafeMutableBufferPointer { cPtr in
                        let M = BLASInt(planeSize)
                        let Nbl = BLASInt(N)
                        let K = BLASInt(32)
                        let lda = BLASInt(32)
                        let ldb = Nbl
                        let ldc = Nbl
                        blas_sgemm_rowmajor(m: M, n: Nbl, k: K, alpha: 1.0, A: aPtr.baseAddress!, lda: lda, B: bPtr.baseAddress!, ldb: ldb, beta: 0.0, C: cPtr.baseAddress!, ldc: ldc)
                    }
                }
            }

            // Now compute overlap metrics from C (no additional SGEMV calls!)
            for j in 0..<N {
                let (origIdx, d) = prunedCandidates[j]

                // Count candidate mask pixels and intersection with primary
                var candPixelCount = 0
                var interCount = 0

                for px in 0..<planeSize {
                    let candLogit = C[px * N + j]
                    let candPositive = candLogit > 0
                    let primPositive = primaryLogits[px] > 0

                    if candPositive {
                        candPixelCount += 1
                        if primPositive { interCount += 1 }
                    }
                }

                let covPrimary: Float = primaryMaskArea > 0 ? Float(interCount) / Float(primaryMaskArea) : 0
                let covCand: Float = candPixelCount > 0 ? Float(interCount) / Float(candPixelCount) : 0

                let relation = classifyRelation(candArea: candPixelCount, covPrimary: covPrimary, covCand: covCand)
                let ratio = Float(candPixelCount) / Float(max(1, primaryMaskArea))

                switch relation {
                case .overlay:
                    // Keep overlays - they overlap with primary
                    kept2.append(d)
                    if debugMode {
                        logDebug("   ✅ [\(origIdx)]: \(className(d.classIdx)) OVERLAY: \(String(format: "%.0f", covCand * 100))% inside primary, ratio=\(String(format: "%.1f", ratio))")
                    }

                case .underlay:
                    // Drop underlays - these are large detections that encompass the primary (e.g., scene labels)
                    if debugMode {
                        logDebug("   ❌ [\(origIdx)]: \(className(d.classIdx)) UNDERLAY: covers \(String(format: "%.0f", covPrimary * 100))% of primary, ratio=\(String(format: "%.1f", ratio))")
                    }

                case .sideBySide:
                    kept2.append(d)
                    if debugMode {
                        logDebug("   ✅ [\(origIdx)]: \(className(d.classIdx)) SIDE-BY-SIDE: covPri=\(String(format: "%.0f", covPrimary * 100))% covCand=\(String(format: "%.0f", covCand * 100))%")
                    }

                case .ambiguous:
                    // Keep ambiguous - they have partial overlap with primary
                    kept2.append(d)
                    if debugMode {
                        logDebug("   ✅ [\(origIdx)]: \(className(d.classIdx)) AMBIGUOUS: covPri=\(String(format: "%.0f", covPrimary * 100))% covCand=\(String(format: "%.0f", covCand * 100))%")
                    }
                }
            }
        }

        let t11End = Date()
        if debugMode {
            let filterMs = String(format: "%.2f", t11End.timeIntervalSince(t11) * 1000)
            logDebug("⏱️ STAGE 11 - Filter (SGEMM): \(filterMs) ms, kept=\(kept2.count)")
        }

        if kept2.isEmpty {
            if debugMode { logDebug("⚠️ No detections after filter") }
            DispatchQueue.main.async { self.maskImageView.image = nil }
            resetProcessingFlag()
            return
        }

        // STAGE 12: Compute union bbox
        let t12 = Date()

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

        var bx1 = Int(round((ux1 - padX) / resizeGain))
        var by1 = Int(round((uy1 - padY) / resizeGain))
        var bx2 = Int(round((ux2 - padX) / resizeGain))
        var by2 = Int(round((uy2 - padY) / resizeGain))

        bx1 = max(0, min(origW - 1, bx1))
        by1 = max(0, min(origH - 1, by1))
        bx2 = max(0, min(origW, bx2))
        by2 = max(0, min(origH, by2))

        let t12End = Date()
        if debugMode {
            let unionMs = String(format: "%.2f", t12End.timeIntervalSince(t12) * 1000)
            logDebug("⏱️ STAGE 12 - Union bbox: \(unionMs) ms")
            logDebug("   image: [\(bx1),\(by1)]→[\(bx2),\(by2)] = \(bx2-bx1)x\(by2-by1)")
        }

        // Compute PRIMARY bbox in image coordinates (for clipping mask later)
        let primaryBx1 = max(0, Int(round((primary.x - primary.w * 0.5 - padX) / resizeGain)))
        let primaryBy1 = max(0, Int(round((primary.y - primary.h * 0.5 - padY) / resizeGain)))
        let primaryBx2 = min(origW, Int(round((primary.x + primary.w * 0.5 - padX) / resizeGain)))
        let primaryBy2 = min(origH, Int(round((primary.y + primary.h * 0.5 - padY) / resizeGain)))
        if debugMode {
            logDebug("   PRIMARY bbox: [\(primaryBx1),\(primaryBy1)]→[\(primaryBx2),\(primaryBy2)]")
        }

        // Helper: Build full-resolution mask from current kept detections
        func buildFullMask(from detections: [UnionDet]) -> (maskFull: [UInt8], positiveCount: Int) {
            // Stage 13: Compute per-pixel max logits across detections
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
                    // tmp = A * coeffs  (A is planeSize x 32, row-major, lda = 32)
                    A.withUnsafeBufferPointer { aPtr in
                        d.coeffs.withUnsafeBufferPointer { xPtr in
                            tmp.withUnsafeMutableBufferPointer { yPtr in
                                let lda = BLASInt(32)
                                blas_sgemv_rowmajor(m: M, n: K, alpha: alpha, A: aPtr.baseAddress!, lda: lda, x: xPtr.baseAddress!, incx: 1, beta: beta, y: yPtr.baseAddress!, incy: 1)
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

                    A.withUnsafeBufferPointer { aPtr in
                        B.withUnsafeBufferPointer { bPtr in
                            C.withUnsafeMutableBufferPointer { cPtr in
                                let lda = BLASInt(32)
                                let ldb = N
                                let ldc = N
                                blas_sgemm_rowmajor(m: M, n: N, k: K, alpha: alpha, A: aPtr.baseAddress!, lda: lda, B: bPtr.baseAddress!, ldb: ldb, beta: beta, C: cPtr.baseAddress!, ldc: ldc)
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
            for i in 0..<planeSize {
                if maxLogits[i] > 0.0 {
                    maskSmall[i] = 255
                    positiveCount += 1
                }
            }

            // Stage 15: Upscale + crop back (morphology disabled)
            let maskFull = upscaleMask(maskSmall: maskSmall,
                                       pW: pW, pH: pH,
                                       modelInput: 1280,
                                       origW: origW, origH: origH,
                                       resizeGain: resizeGain,
                                       padX: padX, padY: padY)

            return (maskFull, positiveCount)
        }

        // Helper: Build full-resolution mask using Metal for the heavy logits->maskSmall step
        // Logic preserved: maskSmall[i] = 255 iff maxLogit > 0.0, then reuse the same upscaleMask() path.
        func buildFullMaskMetal(from detections: [UnionDet]) -> (maskFull: [UInt8], positiveCount: Int) {
            guard let mm = metalMaskLogic else {
                return buildFullMask(from: detections)
            }
            let detCount = detections.count
            if detCount == 0 {
                return ([UInt8](repeating: 0, count: origW * origH), 0)
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
            let maskSmall = mm.buildMaskSmall(planes: planes, coeffs: coeffFlat, planeSize: planeSize, detCount: detCount)
            var positiveCount = 0
            // Count positives (same as CPU)
            for v in maskSmall { if v > 0 { positiveCount += 1 } }

            // Reuse your existing upscale/crop pipeline exactly (same signature as your original)
            // NOTE: keep resizeGain/padX/padY mapping identical to CPU path.
            let maskFull = upscaleMask(maskSmall: maskSmall,
                                      pW: pW, pH: pH,
                                      modelInput: 1280,
                                      origW: origW, origH: origH,
                                      resizeGain: resizeGain,
                                      padX: padX, padY: padY)
            return (maskFull, positiveCount)
        }

        // STAGE 13–15b: Build initial mask from kept2 (pre-bbox filter)
        let t13to15b = Date()
        let build1 = buildFullMaskMetal(from: kept2)
        var maskFull = build1.maskFull
        if debugMode {
            let buildPreMs = String(format: "%.2f", Date().timeIntervalSince(t13to15b) * 1000)
            logDebug("⏱️ STAGE 13–15b - Build mask (pre-bbox): \(buildPreMs) ms, positive: \(build1.positiveCount)")
        }

        // STAGE 15c: Clip mask to PRIMARY bbox (zero out pixels outside primary)
        // This removes noise from detections that extend beyond the primary object
        let t15c = Date()
        var clippedCount = 0
        for y in 0..<origH {
            for x in 0..<origW {
                let idx = y * origW + x
                if maskFull[idx] > 0 {
                    // Check if pixel is outside primary bbox
                    if x < primaryBx1 || x >= primaryBx2 || y < primaryBy1 || y >= primaryBy2 {
                        maskFull[idx] = 0
                        clippedCount += 1
                    }
                }
            }
        }
        if debugMode {
            let clipMs = String(format: "%.2f", Date().timeIntervalSince(t15c) * 1000)
            logDebug("⏱️ STAGE 15c - Clip to primary bbox: \(clipMs) ms, clipped: \(clippedCount) pixels")
        }

        // Prepare flattened coeffs for fused GPU path
        let detCountFused = kept2.count
        var coeffFlatFused = [Float](repeating: 0, count: detCountFused * 32)
        for j in 0..<detCountFused {
            let c = kept2[j].coeffs
            let n = min(32, c.count)
            for k in 0..<n { coeffFlatFused[j*32 + k] = c[k] }
        }

        // STAGE 16: Composite
        setProgress(0.92, text: "Compositing…")

        // --- 4. COMPOSITING (Fused when available) ---
        let compStart = Date()
        var composedImage: CGImage?

        if let device = metalDevice,
           let queue = metalCommandQueue {

            var cvTextureCache: CVMetalTextureCache?
            CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, device, nil, &cvTextureCache)

            func makeTexture(from pixelBuffer: CVPixelBuffer, pixelFormat: MTLPixelFormat) -> MTLTexture? {
                guard let cache = cvTextureCache else { return nil }
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
                if let fused = fusedMaskCompositePipeline {
                    // Fused path: compute max logits and composite in one pass.
                    // Prepare buffers: planes (32*planeSize floats) and coeffs (detCount*32 floats)
                    let planesBytes = planes.count * MemoryLayout<Float>.size
                    let coeffBytes = coeffFlatFused.count * MemoryLayout<Float>.size
                    let planesBuf = device.makeBuffer(bytes: planes, length: planesBytes, options: .storageModeShared)
                    let coeffBuf = device.makeBuffer(bytes: coeffFlatFused, length: coeffBytes, options: .storageModeShared)

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
                        var modelInput_u = UInt32(1280)
                        var resizeGain_f = resizeGain
                        var padX_f = padX
                        var padY_f = padY
                        enc.setBytes(&pW_u, length: MemoryLayout<UInt32>.size, index: 2)
                        enc.setBytes(&pH_u, length: MemoryLayout<UInt32>.size, index: 3)
                        enc.setBytes(&det_u, length: MemoryLayout<UInt32>.size, index: 4)
                        enc.setBytes(&origW_u, length: MemoryLayout<UInt32>.size, index: 5)
                        enc.setBytes(&origH_u, length: MemoryLayout<UInt32>.size, index: 6)
                        enc.setBytes(&modelInput_u, length: MemoryLayout<UInt32>.size, index: 7)
                        enc.setBytes(&resizeGain_f, length: MemoryLayout<Float>.size, index: 8)
                        enc.setBytes(&padX_f, length: MemoryLayout<Float>.size, index: 9)
                        enc.setBytes(&padY_f, length: MemoryLayout<Float>.size, index: 10)
                        var bx1_u = UInt32(bx1)
                        var by1_u = UInt32(by1)
                        var bx2_u = UInt32(bx2)
                        var by2_u = UInt32(by2)
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

                        // Read back as CGImage
                        let bytesPerRow = origW * 4
                        var rgba = [UInt8](repeating: 0, count: origH * bytesPerRow)
                        out.getBytes(&rgba, bytesPerRow: bytesPerRow, from: MTLRegionMake2D(0, 0, origW, origH), mipmapLevel: 0)
                        if let ctx = CGContext(data: &rgba, width: origW, height: origH, bitsPerComponent: 8, bytesPerRow: bytesPerRow, space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue), let img = ctx.makeImage() {
                            composedImage = img
                        }
                    }
                } else if let pipeline = compositePipeline {
                    // Non-fused GPU path: upload mask and composite (existing path)
                    let maskDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r8Unorm, width: origW, height: origH, mipmapped: false)
                    maskDesc.usage = [.shaderRead]
                    let maskTexture = device.makeTexture(descriptor: maskDesc)
                    if let mt = maskTexture {
                        let region = MTLRegionMake2D(0, 0, origW, origH)
                        mt.replace(region: region, mipmapLevel: 0, withBytes: maskFull, bytesPerRow: origW)
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
                        if let ctx = CGContext(data: &rgba, width: origW, height: origH, bitsPerComponent: 8, bytesPerRow: bytesPerRow, space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue), let img = ctx.makeImage() {
                            composedImage = img
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
            for y in 0..<origH {
                let outRow = y * origW * 4
                let origRow = y * CVPixelBufferGetBytesPerRow(processBuffer)
                for x in 0..<origW {
                    let outIdx = outRow + x * 4
                    if x < bx1 || x >= bx2 || y < by1 || y >= by2 {
                        outBase[outIdx+3] = 0
                        continue
                    }
                    let m = maskFull[y * origW + x]
                    if m > 0 {
                        let origIdx = origRow + x * 4
                        // Source is BGRA, destination is RGBA - swap R and B
                        outBase[outIdx+0] = origBase[origIdx+2]  // R
                        outBase[outIdx+1] = origBase[origIdx+1]  // G
                        outBase[outIdx+2] = origBase[origIdx+0]  // B
                        outBase[outIdx+3] = 255                   // A
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

        // STAGE 17: Finalize (debug overlays drawn onto composedImage if available)
        let t17 = Date()

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
                let dx1 = Int(round((d.x - d.w * 0.5 - padX) / resizeGain))
                let dy1 = Int(round((d.y - d.h * 0.5 - padY) / resizeGain))
                let dx2 = Int(round((d.x + d.w * 0.5 - padX) / resizeGain))
                let dy2 = Int(round((d.y + d.h * 0.5 - padY) / resizeGain))

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

            // Draw union bounding box in green
            ctx.setStrokeColor(UIColor.green.cgColor)
            ctx.setLineWidth(6.0)
            ctx.stroke(CGRect(x: bx1, y: origH - by2, width: bx2 - bx1, height: by2 - by1))
        }

        if let finalCtx = ctx, let img = finalCtx.makeImage() {
            composedImage = img
        }

        // Present result - for landscape, rotate the composite back for the portrait UI
        DispatchQueue.main.async {
            if var cgImg = composedImage {
                if isLandscape {
                    // Rotate back for portrait display: landscapeLeft -> 90° CCW, landscapeRight -> 90° CW
                    let rotatedImg = self.rotateCGImage90(cgImg, clockwise: orientation == .landscapeLeft)
                    cgImg = rotatedImg ?? cgImg
                }
                self.maskImageView.image = UIImage(cgImage: cgImg)
            }
        }
        resetProcessingFlag()

        // Trigger first-detection UI dismissal based on mask having any positive pixels
        let hasMask = maskFull.contains(where: { $0 > 0 })
        if hasMask { finishFirstDetectionIfNeeded() }
        
        let t17End = Date()
        let frameEnd = Date()
        
        if debugMode {
            let finalizeMs = String(format: "%.2f", t17End.timeIntervalSince(t17) * 1000)
            let frameTotalMs = String(format: "%.2f", frameEnd.timeIntervalSince(frameStart) * 1000)
            logDebug("⏱️ STAGE 17 - Finalize: \(finalizeMs) ms")
            logDebug("⏱️ FRAME TOTAL: \(frameTotalMs) ms")
            logDebug("⏱️ ═══════════════════════════════════════════\n")
        }
    }

    // MARK: - Parse Prototypes
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
        
        var rawFloats = [Float](repeating: 0, count: total)
        if proto.dataType == .float16 {
            let src = proto.dataPointer.bindMemory(to: UInt16.self, capacity: total)
            var srcBuf = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: src), height: 1, width: vImagePixelCount(total), rowBytes: total * 2)
            rawFloats.withUnsafeMutableBufferPointer { dstPtr in
                var dstBuf = vImage_Buffer(data: dstPtr.baseAddress, height: 1, width: vImagePixelCount(total), rowBytes: total * 4)
                vImageConvert_Planar16FtoPlanarF(&srcBuf, &dstBuf, vImage_Flags(kvImageNoFlags))
            }
        } else if proto.dataType == .float32 {
            memcpy(&rawFloats, proto.dataPointer, total * MemoryLayout<Float>.size)
        }
        
        var planes = [Float](repeating: 0, count: count * planeSize)
        if cIdx == 0 {
            memcpy(&planes, rawFloats, count * planeSize * MemoryLayout<Float>.size)
        } else if cIdx == 2 {
            for y in 0..<h {
                for x in 0..<w {
                    let baseHW = (y * w + x) * count
                    let dstBase = y * w + x
                    for k in 0..<count {
                        planes[k * planeSize + dstBase] = rawFloats[baseHW + k]
                    }
                }
            }
        }
        return (planes, count, h, w)
    }

    // MARK: - Upscale Mask
    private func upscaleMask(maskSmall: [UInt8], pW: Int, pH: Int, modelInput: Int, origW: Int, origH: Int, resizeGain: Float, padX: Float, padY: Float) -> [UInt8] {
        var maskModel = [UInt8](repeating: 0, count: modelInput * modelInput)
        maskModel.withUnsafeMutableBufferPointer { dstPtr in
            maskSmall.withUnsafeBufferPointer { srcPtr in
                var s = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: srcPtr.baseAddress!), height: vImagePixelCount(pH), width: vImagePixelCount(pW), rowBytes: pW)
                var d = vImage_Buffer(data: dstPtr.baseAddress!, height: vImagePixelCount(modelInput), width: vImagePixelCount(modelInput), rowBytes: modelInput)
                let flags: vImage_Flags = useBilinearUpscaling ? vImage_Flags(kvImageHighQualityResampling) : vImage_Flags(kvImageNoFlags)
                vImageScale_Planar8(&s, &d, nil, flags)
            }
        }
        
        let contentW = Int(round(Float(origW) * resizeGain))
        let contentH = Int(round(Float(origH) * resizeGain))
        let x0 = max(0, min(modelInput - 1, Int(round(padX))))
        let y0 = max(0, min(modelInput - 1, Int(round(padY))))
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

    // MARK: - Resize to Square
    private func resizeToSquare(_ src: CVPixelBuffer, size: Int) -> (buffer: CVPixelBuffer, gain: Float, padX: Float, padY: Float)? {
        CVPixelBufferLockBaseAddress(src, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(src, .readOnly) }
        
        let srcW = CVPixelBufferGetWidth(src)
        let srcH = CVPixelBufferGetHeight(src)
        
        let gain = min(Float(size) / Float(srcW), Float(size) / Float(srcH))
        let newW = Int(Float(srcW) * gain)
        let newH = Int(Float(srcH) * gain)
        let padX = Float(size - newW) / 2.0
        let padY = Float(size - newH) / 2.0
        
        var dstOpt: CVPixelBuffer?
        guard CVPixelBufferCreate(nil, size, size, kCVPixelFormatType_32BGRA, nil, &dstOpt) == kCVReturnSuccess,
              let dst = dstOpt else { return nil }
        
        CVPixelBufferLockBaseAddress(dst, [])
        defer { CVPixelBufferUnlockBaseAddress(dst, []) }
        
        guard let srcBase = CVPixelBufferGetBaseAddress(src),
              let dstBase = CVPixelBufferGetBaseAddress(dst) else { return nil }
        
        memset(dstBase, 128, size * size * 4)
        
        var srcBuffer = vImage_Buffer(data: srcBase, height: vImagePixelCount(srcH), width: vImagePixelCount(srcW), rowBytes: CVPixelBufferGetBytesPerRow(src))
        let dstPtr = dstBase.assumingMemoryBound(to: UInt8.self)
        let dstRowBytes = CVPixelBufferGetBytesPerRow(dst)
        let offsetPtr = dstPtr.advanced(by: Int(padY) * dstRowBytes + Int(padX) * 4)
        var dstBuffer = vImage_Buffer(data: offsetPtr, height: vImagePixelCount(newH), width: vImagePixelCount(newW), rowBytes: dstRowBytes)
        
        guard vImageScale_ARGB8888(&srcBuffer, &dstBuffer, nil, vImage_Flags(0)) == kvImageNoError else { return nil }
        
        return (buffer: dst, gain: gain, padX: padX, padY: padY)
    }

    // MARK: - MLMultiArray
    // NOTE: Consider exporting the model with built-in image preprocessing and FP16 inputs
    // to avoid this CPU conversion entirely. MLShapedArray<Float16> can also reduce bandwidth.
    private func pixelBufferToMLMultiArray(_ pixelBuffer: CVPixelBuffer) -> MLMultiArray? {
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        guard width == 1280, height == 1280 else { return nil }
        guard let array = try? MLMultiArray(shape: [1, 3, 1280, 1280], dataType: .float32) else { return nil }
        
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

        guard let colorSpace = image.colorSpace,
              let context = CGContext(
                  data: nil,
                  width: dstWidth,
                  height: dstHeight,
                  bitsPerComponent: image.bitsPerComponent,
                  bytesPerRow: 0,
                  space: colorSpace,
                  bitmapInfo: image.bitmapInfo.rawValue
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
        guard !hasFirstDetection else { return }
        DispatchQueue.main.async {
            self.progressView.isHidden = false
            self.progressLabel.isHidden = false
            self.progressView.progress = value
            self.progressLabel.text = "  \(text)  "
        }
    }
    
    private func finishFirstDetectionIfNeeded() {
        guard !hasFirstDetection else { return }
        hasFirstDetection = true
        DispatchQueue.main.async {
            UIView.animate(withDuration: 0.25) {
                self.progressView.alpha = 0
                self.progressLabel.alpha = 0
            } completion: { _ in
                self.progressView.isHidden = true
                self.progressLabel.isHidden = true
                self.progressView.alpha = 1
                self.progressLabel.alpha = 1
            }
        }
    }

    // MARK: - Gestures
    @objc private func handlePinch(_ gesture: UIPinchGestureRecognizer) {
        guard maskImageView.image != nil else { return }
        switch gesture.state {
        case .changed:
            let newScale = currentScale * gesture.scale
            let clampedScale = min(max(newScale, 0.3), 3.0)
            maskImageView.transform = CGAffineTransform(scaleX: clampedScale, y: clampedScale)
            currentScale = clampedScale
            gesture.scale = 1.0
        case .ended, .cancelled:
            if currentScale > 0.9 && currentScale < 1.1 {
                currentScale = 1.0
                UIView.animate(withDuration: 0.2) { self.maskImageView.transform = .identity }
            }
        default: break
        }
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
    
    func gestureRecognizer(_ gestureRecognizer: UIGestureRecognizer, shouldBeRequiredToFailBy otherGestureRecognizer: UIGestureRecognizer) -> Bool {
        // Don't interfere with navigation gestures
        return false
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

