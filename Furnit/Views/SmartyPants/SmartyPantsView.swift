// SmartyPantsView.swift
// Single-stage: detect → NMS → keepOverlapping → union mask → cutout
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
struct SmartyPantsViewSwiftUI: UIViewRepresentable {
    let mlModel: MLModel?
    var processInterval: TimeInterval = 0.1
    var confidenceThreshold: Float = 0.15
    var iouThreshold: Float = 0.5
    var useBilinearUpscaling: Bool = false
    var active: Bool = false
    
    @ObservedObject private var appState = AppStateManager.shared

    func makeUIView(context: Context) -> SmartyPantsContainerView {
        let v = SmartyPantsContainerView()
        v.processInterval = processInterval
        v.confidenceThreshold = confidenceThreshold
        v.iouThreshold = iouThreshold
        v.useBilinearUpscaling = useBilinearUpscaling
        v.setModel(mlModel)
        if active { v.startIfNeeded() }
        return v
    }

    func updateUIView(_ uiView: SmartyPantsContainerView, context: Context) {
        uiView.setModel(mlModel)
        uiView.processInterval = processInterval
        uiView.confidenceThreshold = confidenceThreshold
        uiView.iouThreshold = iouThreshold
        uiView.useBilinearUpscaling = useBilinearUpscaling
        if active { uiView.startIfNeeded() } else { uiView.stop() }
    }

    static func dismantleUIView(_ uiView: SmartyPantsContainerView, coordinator: ()) {
        uiView.stop()
    }
}

// MARK: - Detection Struct
struct UnionDet {
    let x, y, w, h: Float
    let confidence: Float
    let classIdx: Int
    let coeffs: [Float]
}

// MARK: - Main Container View
final class SmartyPantsContainerView: UIView, AVCaptureVideoDataOutputSampleBufferDelegate, UIGestureRecognizerDelegate {
    
    // MARK: Config
    var processInterval: TimeInterval = 0.1
    var confidenceThreshold: Float = 0.1
    var iouThreshold: Float = 0.7
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
        if debugMode { logDebug("✅ SmartyPantsContainerView initialized") }
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
    }
    
    func startIfNeeded() {
        hasFirstDetection = false
        setProgress(0.05, text: "Starting camera…")
        requestCameraPermissionAndStart()
    }
    
    func stop() {
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

        if debugMode {
            logDebug("\n⏱️ ═══════════════════════════════════════════")
            logDebug("⏱️ FRAME START @ \(String(format: "%.3f", frameStart.timeIntervalSince1970))")
            logDebug("⏱️ ═══════════════════════════════════════════")
        }

        // STAGE 1: Resize to square
        let t1 = Date()
        setProgress(0.15, text: "Resizing…")
        
        guard let sq = resizeToSquare(pixelBuffer, size: 1280) else {
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
            guard let imageValue = try? MLFeatureValue(pixelBuffer: sq.buffer) else {
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

        // Try model output names - yoloe-11l (new export), yoloe-26l, yoloe-11l (old)
        let detArray: MLMultiArray
        let protoArray: MLMultiArray

        if let det = output.featureValue(for: "var_2374")?.multiArrayValue,
           let proto = output.featureValue(for: "var_2412")?.multiArrayValue {
            // yoloe-11l model outputs (new export with proper cv3/cv4 heads)
            detArray = det
            protoArray = proto
            if debugMode { logDebug("📦 Using yoloe-11l output tensors (var_2374/var_2412)") }
        } else if let det = output.featureValue(for: "var_2346")?.multiArrayValue,
           let proto = output.featureValue(for: "var_2429")?.multiArrayValue {
            // yoloe-26l model outputs
            detArray = det
            protoArray = proto
            if debugMode { logDebug("📦 Using yoloe-26l output tensors") }
        } else if let det = output.featureValue(for: "var_2497")?.multiArrayValue,
                  let proto = output.featureValue(for: "p")?.multiArrayValue {
            // yoloe-11l model outputs (old export)
            detArray = det
            protoArray = proto
            if debugMode { logDebug("📦 Using yoloe-11l output tensors (old)") }
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
        // - Old format (yoloe-11l): [1, 144, 8400] where 144 = features (4+numClasses+32), 8400 = anchors
        // - New format (yoloe-26l): [1, 300, 38] where 300 = detections, 38 = features per detection
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
                    logDebug("   First det raw[0..9]: \(rawVals.map { String(format: "%.2f", $0) })")
                }
            }

            // Layout: [x, y, w, h, conf, class_id, coeff0..coeff31]
            // Ultralytics NMS-free export: conf at index 4, class at index 5

            for detIdx in 0..<numDetections {
                // Use actual strides for memory access
                let base = detIdx * detStride

                let x = detBuf[base + 0 * featStride]
                let y = detBuf[base + 1 * featStride]
                let w = detBuf[base + 2 * featStride]
                let h = detBuf[base + 3 * featStride]
                let confidence = detBuf[base + 4 * featStride]
                let classIdxFloat = detBuf[base + 5 * featStride]

                // Skip if any value is NaN or Inf
                guard x.isFinite, y.isFinite, w.isFinite, h.isFinite, confidence.isFinite, classIdxFloat.isFinite else { continue }

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

        // STAGE 7: Apply NMS
        let t7 = Date()
        let boxes: [CGRect] = allDets.map { d in
            CGRect(x: CGFloat(d.x - d.w * 0.5),
                   y: CGFloat(d.y - d.h * 0.5),
                   width: CGFloat(d.w),
                   height: CGFloat(d.h))
        }
        let scores: [Float] = allDets.map { $0.confidence }
        let keptIdx = applyNMS(boxes: boxes, scores: scores, iouThreshold: iouThreshold)
        let afterNMS: [UnionDet] = keptIdx.map { allDets[$0] }
        let t7End = Date()
        if debugMode {
            let nmsMs = String(format: "%.2f", t7End.timeIntervalSince(t7) * 1000)
            logDebug("⏱️ STAGE 7 - NMS: \(nmsMs) ms, kept: \(afterNMS.count)")
        }

        // STAGE 8: Find primary (conf > 0.5, largest area)
        let t8 = Date()

        var primaryIdx = -1
        var maxArea: Float = 0
        for (i, d) in afterNMS.enumerated() {
            if d.confidence > 0.5 {
                let area = d.w * d.h
                if area > maxArea {
                    maxArea = area
                    primaryIdx = i
                }
            }
        }

        if primaryIdx < 0 {
            if debugMode { logDebug("   ⚠️ No detection with conf > 0.5") }
            DispatchQueue.main.async { self.maskImageView.image = nil }
            resetProcessingFlag()
            return
        }
        
        var primary = afterNMS[primaryIdx]
        let t8End = Date()
        if debugMode {
            let primaryMs = String(format: "%.2f", t8End.timeIntervalSince(t8) * 1000)
            logDebug("⏱️ STAGE 8 - Primary: \(primaryMs) ms")
            logDebug("   🎯 PRIMARY[\(primaryIdx)]: \u{001B}[1m\(className(primary.classIdx))\u{001B}[0m conf=\(String(format: "%.2f", primary.confidence)) size=\(Int(primary.w))x\(Int(primary.h))")
        }

        // STAGE 9: Parse prototypes
        let t9 = Date()
        setProgress(0.65, text: "Building mask…")
        
        guard let protoInfo = parsePrototypes(protoArray) else {
            if debugMode { logDebug("❌ STAGE 9 FAILED: Parse prototypes") }
            resetProcessingFlag()
            return
        }
        let planes = protoInfo.planes
        let pH = protoInfo.height
        let pW = protoInfo.width
        let planeSize = pH * pW
        
        let t9End = Date()
        if debugMode {
            let protoMs = String(format: "%.2f", t9End.timeIntervalSince(t9) * 1000)
            logDebug("⏱️ STAGE 9 - Prototypes: \(protoMs) ms")
        }

        // STAGE 10: Reorganize prototypes
        let t10 = Date()
        
        var A = [Float](repeating: 0, count: planeSize * 32)
        var zero: Float = 0
        A.withUnsafeMutableBufferPointer { dstPtr in
            planes.withUnsafeBufferPointer { srcPtr in
                for k in 0..<32 {
                    let srcStart = srcPtr.baseAddress!.advanced(by: k * planeSize)
                    let dstStart = dstPtr.baseAddress!.advanced(by: k)
                    vDSP_vsadd(srcStart, 1, &zero, dstStart, 32, vDSP_Length(planeSize))
                }
            }
        }
        
        let t10End = Date()
        if debugMode {
            let reorgMs = String(format: "%.2f", t10End.timeIntervalSince(t10) * 1000)
            logDebug("⏱️ STAGE 10 - Reorganize: \(reorgMs) ms")
        }

        // STAGE 11: Filter - use mask overlap with primary instead of bbox overlap
        let t11 = Date()

        // Build primary logits and mask in prototype space (pW x pH)
        // A is (planeSize x 32) in row-major where each row (pixel) has 32 prototype values.
        // We'll compute logits = A * coeffs (SGEMV) and threshold at 0.
        func logitsForDetection(_ coeffs: [Float]) -> [Float] {
            var result = [Float](repeating: 0, count: planeSize)
            A.withUnsafeBufferPointer { aPtr in
                coeffs.withUnsafeBufferPointer { xPtr in
                    result.withUnsafeMutableBufferPointer { yPtr in
                        let m = BLASInt(planeSize)
                        let n = BLASInt(32)
                        let lda = BLASInt(32)
                        let incx: BLASInt = 1
                        let incy: BLASInt = 1
                        blas_sgemv_rowmajor(m: m, n: n, alpha: 1.0, A: aPtr.baseAddress!, lda: lda, x: xPtr.baseAddress!, incx: incx, beta: 0.0, y: yPtr.baseAddress!, incy: incy)
                    }
                }
            }
            return result
        }

        func maskFromLogits(_ logits: [Float]) -> [UInt8] {
            var mask = [UInt8](repeating: 0, count: planeSize)
            for i in 0..<planeSize { if logits[i] > 0 { mask[i] = 255 } }
            return mask
        }

        // Primary mask in prototype space
        let primaryLogits = logitsForDetection(primary.coeffs)

        // PERF: Precompute indices of primary mask pixels (in prototype space) once.
        // This keeps Stage 11 from scanning the entire plane for every candidate.
        var primaryMaskIndices: [Int] = []
        primaryMaskIndices.reserveCapacity(planeSize / 4)
        for i in 0..<planeSize {
            if primaryLogits[i] > 0 { primaryMaskIndices.append(i) }
        }

        // Helper: compute fraction of PRIMARY mask covered by candidate mask (in prototype space)
        func intersectionCoverage(candidateCoeffs: [Float]) -> Float {
            // Fraction of PRIMARY mask pixels covered by candidate (both in prototype space).
            // PERF: Iterate only over primary mask indices (sparse), not the entire plane.
            if primaryMaskIndices.isEmpty { return 0 }
            let candLogits = logitsForDetection(candidateCoeffs)
            var interCount: Int = 0
            for idx in primaryMaskIndices {
                if candLogits[idx] > 0 { interCount += 1 }
            }
            return Float(interCount) / Float(primaryMaskIndices.count)
        }

        // Helper: compute mask density (positive pixels / mask bounding box area)
        // Returns (density, valid) - valid is false if mask is empty
        func maskDensity(coeffs: [Float]) -> (Float, Bool) {
            let logits = logitsForDetection(coeffs)
            var minX = pW, maxX = 0, minY = pH, maxY = 0
            var positiveCount = 0
            for idx in 0..<planeSize {
                if logits[idx] > 0 {
                    positiveCount += 1
                    let x = idx % pW
                    let y = idx / pW
                    if x < minX { minX = x }
                    if x > maxX { maxX = x }
                    if y < minY { minY = y }
                    if y > maxY { maxY = y }
                }
            }
            let maskW = maxX - minX + 1
            let maskH = maxY - minY + 1
            if maskW <= 0 || maskH <= 0 || positiveCount == 0 { return (0, false) }
            let boundingArea = maskW * maskH
            return (Float(positiveCount) / Float(boundingArea), true)
        }

        // Compute bbox edges for size comparison
        let pLeft = primary.x - primary.w * 0.5
        let pRight = primary.x + primary.w * 0.5
        let pTop = primary.y - primary.h * 0.5
        let pBottom = primary.y + primary.h * 0.5

        if debugMode {
            logDebug("   📦 PRIMARY: center=(\(Int(primary.x)),\(Int(primary.y))) size=\(Int(primary.w))x\(Int(primary.h))")
            logDebug("      edges: L=\(Int(pLeft)) R=\(Int(pRight)) T=\(Int(pTop)) B=\(Int(pBottom))")
        }

        var kept2: [UnionDet] = [primary]
        let threshold = AppStateManager.shared.qualitySettings.maskOverlapThreshold

        for (i, d) in afterNMS.enumerated() {
            if i == primaryIdx { continue }

            // Guard against division by zero / NaN
            let wPct = primary.w > 0 ? Int(d.w / primary.w * 100) : 0
            let hPct = primary.h > 0 ? Int(d.h / primary.h * 100) : 0

            // Mask-based overlap: fraction of PRIMARY mask pixels that are also in candidate mask
            let coverageOfCandidate = intersectionCoverage(candidateCoeffs: d.coeffs)

            if coverageOfCandidate < threshold {
                if debugMode {
                    let pct = String(format: "%.2f", coverageOfCandidate * 100)
                    let thresholdPct = String(format: "%.2f", threshold * 100)
                    logDebug("   ❌ [\(i)]: \(className(d.classIdx)) center=(\(Int(d.x)),\(Int(d.y))) size=\(Int(d.w))x\(Int(d.h)) [\(wPct)%,\(hPct)%] PRIMARY COVERED < \(thresholdPct)% (\(pct)%)")
                }
                continue
            }

            let tooLarge = d.w > primary.w * 1.5 && d.h > primary.h * 1.5
            if tooLarge {
                if debugMode {
                    logDebug("   ❌ [\(i)]: \(className(d.classIdx)) center=(\(Int(d.x)),\(Int(d.y))) size=\(Int(d.w))x\(Int(d.h)) [\(wPct)%,\(hPct)%] TOO LARGE")
                }
                continue
            }

            // Check mask density - reject sparse/fragmented masks
            let (density, densityValid) = maskDensity(coeffs: d.coeffs)
            if densityValid && density < 0.2 {
                // Less than 20% of mask bounding box is filled - likely sparse noise
                if debugMode {
                    logDebug("   ❌ [\(i)]: \(className(d.classIdx)) center=(\(Int(d.x)),\(Int(d.y))) SPARSE MASK density=\(String(format: "%.1f", density * 100))%")
                }
                continue
            }

            kept2.append(d)
            if debugMode {
                logDebug("   ✅ [\(i)]: \(className(d.classIdx)) center=(\(Int(d.x)),\(Int(d.y))) size=\(Int(d.w))x\(Int(d.h)) [\(wPct)%,\(hPct)%] (ok)")
            }
        }

        let t11End = Date()
        if debugMode {
            let filterMs = String(format: "%.2f", t11End.timeIntervalSince(t11) * 1000)
            logDebug("⏱️ STAGE 11 - Filter: \(filterMs) ms, kept=\(kept2.count)")
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

        let origW = CVPixelBufferGetWidth(pixelBuffer)
        let origH = CVPixelBufferGetHeight(pixelBuffer)

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
        let maskFull = build1.maskFull
        if debugMode {
            let buildPreMs = String(format: "%.2f", Date().timeIntervalSince(t13to15b) * 1000)
            logDebug("⏱️ STAGE 13–15b - Build mask (pre-bbox): \(buildPreMs) ms, positive: \(build1.positiveCount)")
        }

        // Prepare flattened coeffs for fused GPU path
        let detCountFused = kept2.count
        var coeffFlatFused = [Float](repeating: 0, count: detCountFused * 32)
        for j in 0..<detCountFused {
            let c = kept2[j].coeffs
            let n = min(32, c.count)
            for k in 0..<n { coeffFlatFused[j*32 + k] = c[k] }
        }

        // STAGE 15c: Filter detections by final mask coverage (bbox within mask)
        // Keep detections whose bbox area is sufficiently covered by the final maskFull.
        // Threshold is read from quality settings: bboxInMaskThreshold.
//        let t15c = Date()
//        let bboxCoverageThreshold = AppStateManager.shared.qualitySettings.bboxInMaskThreshold
//        let build1DetCount = kept2.count

        // PERF: Build an integral image (summed-area table) for maskFull once.
        // Then bbox coverage queries become O(1) instead of O(bboxArea).
//        let integralW = origW + 1
//        let integralH = origH + 1
//        var maskIntegral = [Int](repeating: 0, count: integralW * integralH)
        // maskIntegral[(y+1)*integralW + (x+1)] = sum of mask>0 in rect [0..x,0..y]
//        for y in 0..<origH {
//            var rowSum = 0
//            let srcRow = y * origW
//            let dstRow = (y + 1) * integralW
//            let prevRow = y * integralW
//            for x in 0..<origW {
//                if maskFull[srcRow + x] > 0 { rowSum += 1 }
//                maskIntegral[dstRow + (x + 1)] = maskIntegral[prevRow + (x + 1)] + rowSum
//            }
//        }

//        @inline(__always)
//        func integralSum(x1: Int, y1: Int, x2: Int, y2: Int) -> Int {
//            // sum over [x1,x2) x [y1,y2)
//            let A = maskIntegral[y1 * integralW + x1]
//            let B = maskIntegral[y1 * integralW + x2]
//            let C = maskIntegral[y2 * integralW + x1]
//            let D = maskIntegral[y2 * integralW + x2]
//            return D - B - C + A
//        }

//        func bboxFromDetectionInImageSpace(_ d: UnionDet) -> (x1: Int, y1: Int, x2: Int, y2: Int) {
//            let dx1 = Int(round((d.x - d.w * 0.5 - padX) / resizeGain))
//            let dy1 = Int(round((d.y - d.h * 0.5 - padY) / resizeGain))
//            let dx2 = Int(round((d.x + d.w * 0.5 - padX) / resizeGain))
//            let dy2 = Int(round((d.y + d.h * 0.5 - padY) / resizeGain))
//            let clampedX1 = max(0, min(origW - 1, dx1))
//            let clampedY1 = max(0, min(origH - 1, dy1))
//            let clampedX2 = max(0, min(origW, dx2))
//            let clampedY2 = max(0, min(origH, dy2))
//            return (clampedX1, clampedY1, clampedX2, clampedY2)
//        }
//
//        func coverageOfBBoxInMask(_ bbox: (x1: Int, y1: Int, x2: Int, y2: Int)) -> Float {
//            let x1 = bbox.x1, y1 = bbox.y1, x2 = bbox.x2, y2 = bbox.y2
//            let w = max(0, x2 - x1)
//            let h = max(0, y2 - y1)
//            if w == 0 || h == 0 { return 0 }
//            var covered = 0
//            let area = w * h
//            // Sample every pixel; if performance becomes an issue, stride sampling can be introduced.
//            for yy in y1..<y2 {
//                let row = yy * origW
//                for xx in x1..<x2 {
//                    if maskFull[row + xx] > 0 { covered += 1 }
//                }
//            }
//            return Float(covered) / Float(area)
//        }
//
//        var keptAfterMask: [UnionDet] = []
//        keptAfterMask.reserveCapacity(kept2.count)
//        for (i, d) in kept2.enumerated() {
//            let bbox = bboxFromDetectionInImageSpace(d)
//            let cov = coverageOfBBoxInMask(bbox)
//            if cov >= bboxCoverageThreshold {
//                keptAfterMask.append(d)
//                if debugMode {
//                    let pct = String(format: "%.2f", cov * 100)
//                    logDebug("   ✅ [\(i)] BBOX in final mask: \(pct)% >= \(Int(bboxCoverageThreshold*100))%")
//                }
//            } else if debugMode {
//                let pct = String(format: "%.2f", cov * 100)
//                logDebug("   ❌ [\(i)] BBOX in final mask: \(pct)% < \(Int(bboxCoverageThreshold*100))%")
//            }
//        }
//        kept2 = keptAfterMask

//        let t15cEnd = Date()
//        if debugMode {
//            logDebug("⏱️ STAGE 15c - BBox-in-mask filter: \(String(format: "%.2f", t15cEnd.timeIntervalSince(t15c) * 1000)) ms, kept=\(kept2.count)")
//        }

        // Rebuild final mask from survivors (collated)
        // PERF: Skip rebuild if nothing changed (same count). This avoids ~1s+ work per frame.
//        let tRebuild = Date()
//        let beforeRebuildCount = keptAfterMask.count // after Stage 15c
//        // NOTE: kept2 already equals keptAfterMask here.
//        if beforeRebuildCount != build1DetCount {
//            let build2 = buildFullMask(from: kept2)
//            maskFull = build2.maskFull
//            if debugMode {
//                logDebug("⏱️ REBUILD - Final mask from survivors: \(String(format: "%.2f", Date().timeIntervalSince(tRebuild) * 1000)) ms, positive: \(build2.positiveCount)")
//            }
//        } else if debugMode {
//            logDebug("⏱️ REBUILD - Skipped (kept count unchanged): \(String(format: "%.2f", Date().timeIntervalSince(tRebuild) * 1000)) ms")
//        }

        // STAGE 16: Composite
//        let t16 = Date()
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

            // Source BGRA texture from camera buffer
            CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
            let srcTexture = makeTexture(from: pixelBuffer, pixelFormat: .bgra8Unorm)
            CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly)

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
            // Fallback CPU compositing
            let ctx = CGContext(data: nil, width: origW, height: origH, bitsPerComponent: 8, bytesPerRow: origW * 4, space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
            CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
            let outBase = ctx.data!.assumingMemoryBound(to: UInt8.self)
            let origBase = CVPixelBufferGetBaseAddress(pixelBuffer)!.assumingMemoryBound(to: UInt8.self)
            for y in 0..<origH {
                let outRow = y * origW * 4
                let origRow = y * CVPixelBufferGetBytesPerRow(pixelBuffer)
                for x in 0..<origW {
                    let outIdx = outRow + x * 4
                    if x < bx1 || x >= bx2 || y < by1 || y >= by2 {
                        outBase[outIdx+3] = 0
                        continue
                    }
                    let m = maskFull[y * origW + x]
                    if m > 0 {
                        let origIdx = origRow + x * 4
                        outBase[outIdx+0] = origBase[origIdx+0]
                        outBase[outIdx+1] = origBase[origIdx+1]
                        outBase[outIdx+2] = origBase[origIdx+2]
                        outBase[outIdx+3] = 255
                    } else {
                        outBase[outIdx+3] = 0
                    }
                }
            }
            CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly)
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

        // Present result
        DispatchQueue.main.async {
            if let cgImg = composedImage { self.maskImageView.image = UIImage(cgImage: cgImg) }
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

    // MARK: - NMS
    func applyNMS(boxes: [CGRect], scores: [Float], iouThreshold: Float) -> [Int] {
        var indices = scores.enumerated().sorted(by: { $0.element > $1.element }).map { $0.offset }
        var keep = [Int]()
        
        while !indices.isEmpty {
            let current = indices.removeFirst()
            keep.append(current)
            
            indices.removeAll { next in
                let intersection = boxes[current].intersection(boxes[next])
                let iou = intersection.area / (boxes[current].area + boxes[next].area - intersection.area)
                return iou > CGFloat(iouThreshold)
            }
        }
        return keep
    }
    
    private func iou(_ a: UnionDet, _ b: UnionDet) -> Float {
        let ax1 = a.x - a.w * 0.5, ax2 = a.x + a.w * 0.5
        let ay1 = a.y - a.h * 0.5, ay2 = a.y + a.h * 0.5
        let bx1 = b.x - b.w * 0.5, bx2 = b.x + b.w * 0.5
        let by1 = b.y - b.h * 0.5, by2 = b.y + b.h * 0.5
        
        let ix = max(Float(0), min(ax2, bx2) - max(ax1, bx1))
        let iy = max(Float(0), min(ay2, by2) - max(ay1, by1))
        let inter = ix * iy
        let union = a.w * a.h + b.w * b.h - inter
        return union > 0 ? inter / union : 0.0
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

extension CGRect {
    var area: CGFloat { width * height }
}

