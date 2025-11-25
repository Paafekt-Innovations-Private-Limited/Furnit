import SwiftUI
import UIKit
import CoreML
import Accelerate
import AVFoundation
import Photos

// SmartyPantsView.swift
// Single-file on-device YOLOE mask decoding + optimized pipeline
// Drop into your project, then instantiate via the provided SwiftUI wrapper.

struct SmartyPantsViewSwiftUI: UIViewRepresentable {
    let mlModel: MLModel?
    var processInterval: TimeInterval = 0.07
    var scoreThreshold: Float = 0.25
    var active: Bool = false
    var debugShowTop1: Bool = false // Changed to false to show all detections by default
    var debugSaveImages: Bool = true

    func makeUIView(context: Context) -> SmartyPantsContainerView {
        let v = SmartyPantsContainerView()
        v.processInterval = processInterval
        v.scoreThreshold = scoreThreshold
        v.debugShowTopMask = debugShowTop1
        v.debugSaveImages = debugSaveImages
        v.setModel(mlModel)
        if active { v.startIfNeeded() }
        return v
    }

    func updateUIView(_ uiView: SmartyPantsContainerView, context: Context) {
        uiView.setModel(mlModel)
        uiView.processInterval = processInterval
        uiView.scoreThreshold = scoreThreshold
        uiView.debugShowTopMask = debugShowTop1
        uiView.debugSaveImages = debugSaveImages
        if active { uiView.startIfNeeded() } else { uiView.stop() }
    }

    static func dismantleUIView(_ uiView: SmartyPantsContainerView, coordinator: ()) {
        uiView.stop()
    }
}

final class SmartyPantsContainerView: UIView {
    // Public config
    var processInterval: TimeInterval = 0.07
    var scoreThreshold: Float = 0.25

    // Debug flags
    var debugShowTopMask: Bool = false
    var debugSaveImages: Bool = false

    // Camera session
    private let captureSession = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let sampleQueue = DispatchQueue(label: "com.furnit.smarty.sample", qos: .userInitiated)
    private var isSessionRunning = false
    
    // Frame counting for debug logging
    private var frameCount = 0

    // UI
    private let maskImageView: UIImageView = {
        let iv = UIImageView()
        iv.contentMode = .scaleAspectFill
        iv.backgroundColor = .clear
        iv.isOpaque = false
        iv.clipsToBounds = true
        return iv
    }()

    // ML model
    private var mlModel: MLModel?

    // Queues and throttles
    private let detectionQueue = DispatchQueue(label: "com.furnit.smarty.detection", qos: .userInitiated)
    private var lastProcessTime = Date.distantPast
    private var processing = false

    // Caches and buffers
    private var cachedScoreIdx: Int?
    private var cachedCoeffStart: Int?
    private var detectionsBuf: UnsafeMutablePointer<Float>?
    private var detectionsBufCount: Int = 0

    private var protoFloatBuf: UnsafeMutablePointer<Float>?
    private var protoFloatCount: Int = 0

    private var maskFloatBuf: UnsafeMutablePointer<Float>?
    private var maskFloatBufCount: Int = 0

    private var planar8BufA: UnsafeMutablePointer<UInt8>?
    private var planar8BufB: UnsafeMutablePointer<UInt8>?
    private var planar8BufCount: Int = 0

    // Model protos info
    private let protoK = 32
    private let protoH = 160
    private let protoW = 160

    // Init / layout
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
        addSubview(maskImageView)
        maskImageView.translatesAutoresizingMaskIntoConstraints = false
        NSLayoutConstraint.activate([
            maskImageView.leadingAnchor.constraint(equalTo: leadingAnchor),
            maskImageView.trailingAnchor.constraint(equalTo: trailingAnchor),
            maskImageView.topAnchor.constraint(equalTo: topAnchor),
            maskImageView.bottomAnchor.constraint(equalTo: bottomAnchor)
        ])
        
        print("🎭 SmartyPants: CommonInit completed")
        setupCamera()
    }
    override func layoutSubviews() {
        super.layoutSubviews()
        maskImageView.frame = bounds
    }

    deinit {
        print("🎭 SmartyPants: Deinit - cleaning up camera and buffers")
        stopCamera()
        protoFloatBuf?.deallocate()
        detectionsBuf?.deallocate()
        maskFloatBuf?.deallocate()
        planar8BufA?.deallocate()
        planar8BufB?.deallocate()
        print("🎭 SmartyPants: Cleanup completed")
    }

    // Public API
    func setModel(_ model: MLModel?) {
        detectionQueue.sync {
            self.mlModel = model
            print("🎭 SmartyPants: Model set -> \(model != nil ? "✅ LOADED" : "❌ NIL")")
        }
    }
    
    func startIfNeeded() { 
        print("🎭 SmartyPants: startIfNeeded() called")
        requestCameraPermissionAndStart()
    }
    
    func stop() { 
        print("🎭 SmartyPants: stop() called")
        stopCamera()
    }
    
    // MARK: - Camera Setup & Management
    
    private func setupCamera() {
        print("🎥 === SMARTYPANTS CAMERA SETUP STARTING ===")
        captureSession.beginConfiguration()
        print("📋 SmartyPants: Session configuration began")
        
        // Log initial state
        print("📊 SmartyPants: Initial session state:")
        print("   - isRunning: \(captureSession.isRunning)")
        print("   - canSetSessionPreset: \(captureSession.canSetSessionPreset(.hd1280x720))")
        
        captureSession.sessionPreset = .hd1280x720
        print("📐 SmartyPants: Session preset set to HD 1280x720")
        
        // Clear existing inputs/outputs
        let existingInputs = captureSession.inputs.count
        let existingOutputs = captureSession.outputs.count
        captureSession.inputs.forEach { captureSession.removeInput($0) }
        captureSession.outputs.forEach { captureSession.removeOutput($0) }
        print("🗑️ SmartyPants: Cleared \(existingInputs) inputs and \(existingOutputs) outputs")
        
        // Find camera
        print("🔍 SmartyPants: Searching for back camera...")
        let discoverySession = AVCaptureDevice.DiscoverySession(
            deviceTypes: [.builtInWideAngleCamera],
            mediaType: .video,
            position: .unspecified
        )
        
        print("📱 SmartyPants: Available camera devices:")
        for device in discoverySession.devices {
            print("   - \(device.localizedName) (position: \(device.position.rawValue))")
        }
        
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else {
            print("❌ SmartyPants: CRITICAL - No back camera available")
            captureSession.commitConfiguration()
            return
        }
        
        print("✅ SmartyPants: Found back camera: \(device.localizedName)")
        print("📊 SmartyPants: Camera device info:")
        print("   - uniqueID: \(device.uniqueID)")
        print("   - modelID: \(device.modelID)")
        print("   - isConnected: \(device.isConnected)")
        print("   - isSuspended: \(device.isSuspended)")
        
        do {
            print("🔌 SmartyPants: Creating camera input...")
            let input = try AVCaptureDeviceInput(device: device)
            
            if captureSession.canAddInput(input) {
                captureSession.addInput(input)
                print("✅ SmartyPants: Camera input added successfully")
            } else {
                print("❌ SmartyPants: CRITICAL - Cannot add camera input to session")
                captureSession.commitConfiguration()
                return
            }
            
            // Configure video output
            print("📹 SmartyPants: Configuring video output...")
            print("📊 SmartyPants: Video output settings:")
            print("   - Pixel format: kCVPixelFormatType_32BGRA")
            print("   - Sample buffer delegate queue: \(sampleQueue.label)")
            
            videoOutput.setSampleBufferDelegate(self, queue: sampleQueue)
            videoOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
            videoOutput.alwaysDiscardsLateVideoFrames = true
            
            if captureSession.canAddOutput(videoOutput) {
                captureSession.addOutput(videoOutput)
                print("✅ SmartyPants: Video output added successfully")
                
                if let connection = videoOutput.connection(with: .video) {
                    print("🔗 SmartyPants: Configuring video connection...")
                    print("   - isActive: \(connection.isActive)")
                    print("   - isEnabled: \(connection.isEnabled)")
                    print("   - isVideoMirroringSupported: \(connection.isVideoMirroringSupported)")
                    
                    connection.videoRotationAngle = 90
                    if connection.isVideoMirroringSupported {
                        connection.isVideoMirrored = false
                        print("   - Video mirroring disabled")
                    }
                    print("   - Video rotation set to 90°")
                    print("✅ SmartyPants: Video connection configured successfully")
                } else {
                    print("⚠️ SmartyPants: Warning - No video connection found")
                }
            } else {
                print("❌ SmartyPants: CRITICAL - Cannot add video output to session")
                captureSession.commitConfiguration()
                return
            }
            
            print("💾 SmartyPants: Committing session configuration...")
            captureSession.commitConfiguration()
            print("✅ SmartyPants: Camera configured successfully")
            
        } catch {
            print("❌ SmartyPants: CRITICAL - Camera setup failed:")
            print("   Error: \(error.localizedDescription)")
            if let avError = error as? AVError {
                print("   AVError code: \(avError.code.rawValue)")
                print("   AVError description: \(avError.localizedDescription)")
            }
            captureSession.commitConfiguration()
        }
        
        print("🎥 === SMARTYPANTS CAMERA SETUP COMPLETED ===")
    }
    
    private func requestCameraPermissionAndStart() {
        print("🔐 SmartyPants: Requesting camera permission...")
        AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
            DispatchQueue.main.async {
                print("🔐 SmartyPants: Camera permission result: \(granted ? "✅ GRANTED" : "❌ DENIED")")
                if granted {
                    self?.startCamera()
                } else {
                    print("❌ SmartyPants: Cannot start camera - permission denied")
                    print("💡 SmartyPants: Check Settings > Privacy & Security > Camera")
                }
            }
        }
    }
    
    private func startCamera() {
        print("🚀 === SMARTYPANTS START CAMERA REQUESTED ===")
        print("📊 SmartyPants: Pre-start session status:")
        print("   - isRunning: \(captureSession.isRunning)")
        print("   - inputs count: \(captureSession.inputs.count)")
        print("   - outputs count: \(captureSession.outputs.count)")
        
        guard !isSessionRunning else {
            print("⚠️ SmartyPants: Camera session already running")
            return
        }
        
        if !captureSession.isRunning {
            print("▶️ SmartyPants: Session not running - attempting to start...")
            DispatchQueue.global(qos: .background).async { [weak self] in
                print("🔄 SmartyPants: Starting session on background queue...")
                self?.captureSession.startRunning()
                
                DispatchQueue.main.async {
                    guard let self = self else { return }
                    self.isSessionRunning = self.captureSession.isRunning
                    print("📊 SmartyPants: Post-start session status:")
                    print("   - isRunning: \(self.captureSession.isRunning)")
                    print("   - isSessionRunning flag: \(self.isSessionRunning)")
                    
                    if self.captureSession.isRunning {
                        print("✅ SmartyPants: Camera session started successfully!")
                    } else {
                        print("❌ SmartyPants: CRITICAL - Session failed to start!")
                    }
                    
                    // Debug check after delay
                    DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
                        self.debugSessionStatus()
                    }
                }
            }
        } else {
            print("⚠️ SmartyPants: Session already running")
            isSessionRunning = true
            debugSessionStatus()
        }
        print("🚀 === SMARTYPANTS START CAMERA REQUEST COMPLETED ===")
    }
    
    private func stopCamera() {
        print("🛑 === SMARTYPANTS STOP CAMERA REQUESTED ===")
        print("📊 SmartyPants: Pre-stop session status:")
        print("   - isRunning: \(captureSession.isRunning)")
        print("   - isSessionRunning flag: \(isSessionRunning)")
        
        if captureSession.isRunning {
            print("⏹️ SmartyPants: Stopping camera session...")
            captureSession.stopRunning()
            isSessionRunning = false
            print("🛑 SmartyPants: Camera session stopped")
        } else {
            print("⚠️ SmartyPants: Session was already stopped")
            isSessionRunning = false
        }
        
        print("📊 SmartyPants: Post-stop session status:")
        print("   - isRunning: \(captureSession.isRunning)")
        print("   - isSessionRunning flag: \(isSessionRunning)")
        print("🛑 === SMARTYPANTS STOP CAMERA COMPLETED ===")
    }
    
    private func debugSessionStatus() {
        print("🔍 === SMARTYPANTS CAMERA DEBUG STATUS ===")
        print("📊 SmartyPants: Session Status:")
        print("   - isRunning: \(captureSession.isRunning)")
        print("   - isSessionRunning flag: \(isSessionRunning)")
        print("   - sessionPreset: \(captureSession.sessionPreset.rawValue)")
        print("   - inputs count: \(captureSession.inputs.count)")
        print("   - outputs count: \(captureSession.outputs.count)")
        
        print("📋 SmartyPants: Detailed Input Information:")
        for (index, input) in captureSession.inputs.enumerated() {
            print("   Input \(index): \(type(of: input))")
            if let deviceInput = input as? AVCaptureDeviceInput {
                print("      Device: \(deviceInput.device.localizedName)")
                print("      Connected: \(deviceInput.device.isConnected)")
                print("      Position: \(deviceInput.device.position.rawValue)")
            }
        }
        
        print("📤 SmartyPants: Detailed Output Information:")
        for (index, output) in captureSession.outputs.enumerated() {
            print("   Output \(index): \(type(of: output))")
            if let videoOutput = output as? AVCaptureVideoDataOutput {
                if let connection = videoOutput.connection(with: .video) {
                    print("      Connection active: \(connection.isActive)")
                    print("      Connection enabled: \(connection.isEnabled)")
                    print("      Video rotation: \(connection.videoRotationAngle)°")
                } else {
                    print("      ❌ No video connection!")
                }
            }
        }
        
        print("🔧 SmartyPants: Session Capabilities:")
        print("   - canSetSessionPreset(hd1280x720): \(captureSession.canSetSessionPreset(.hd1280x720))")
        print("   - canSetSessionPreset(high): \(captureSession.canSetSessionPreset(.high))")
        print("   - canSetSessionPreset(medium): \(captureSession.canSetSessionPreset(.medium))")
        
        print("🔍 === SMARTYPANTS DEBUG STATUS COMPLETE ===")
    }

    // MARK: - Main processing entry
    // Call this with a CVPixelBuffer from your camera capture pipeline.
    func processFrame(_ pixelBuffer: CVPixelBuffer) {
        guard let model = mlModel else { 
            print("🎭 SmartyPants: No ML model available for processing")
            return 
        }

        let now = Date()
        guard now.timeIntervalSince(lastProcessTime) >= processInterval, !processing else { return }
        lastProcessTime = now
        let frameStart = CFAbsoluteTimeGetCurrent()
        DispatchQueue.main.async { self.processing = true }

        detectionQueue.async { [weak self] in
            guard let self = self else { return }
            
            print("🎭 SmartyPants: Processing frame...")
            
            // Convert buffer -> MLMultiArray input
            guard let inputArray = self.pixelBufferToMLMultiArray(pixelBuffer, width: 640, height: 640) else {
                print("❌ SmartyPants: Failed to convert pixel buffer to MLMultiArray")
                DispatchQueue.main.async { self.processing = false }
                return
            }
            // Run model
            guard let inputProvider = try? MLDictionaryFeatureProvider(dictionary: ["image": inputArray]),
                  let output = try? model.prediction(from: inputProvider) else {
                DispatchQueue.main.async { self.processing = false }
                return
            }
            // Read outputs
            guard let prototypesArr = output.featureValue(for: "p")?.multiArrayValue,
                  let detectionsArr = output.featureValue(for: "var_2421")?.multiArrayValue else {
                DispatchQueue.main.async { self.processing = false }
                return
            }

            // shapes
            let numPredictions = detectionsArr.shape[1].intValue
            let numFeatures = detectionsArr.shape[2].intValue
            let K = self.protoK
            let HW = self.protoH * self.protoW

            print("prototypes shape: \(prototypesArr.shape), detections shape: \(detectionsArr.shape)")

            // Prepare proto buffer (Float32) and reuse
            let protoCount = prototypesArr.count
            if self.protoFloatBuf == nil || self.protoFloatCount != protoCount {
                self.protoFloatBuf?.deallocate()
                self.protoFloatBuf = UnsafeMutablePointer<Float>.allocate(capacity: protoCount)
                self.protoFloatCount = protoCount
            }
            guard let protoBuf = self.protoFloatBuf else {
                DispatchQueue.main.async { self.processing = false }
                return
            }
            // Convert float16 -> float32 (vImage)
            self.copyFloat16MultiArrayToFloatBuffer(prototypesArr, dest: protoBuf)

            // Prepare detections float buffer copy (stride-aware copy)
            let detCount = detectionsArr.count
            if self.detectionsBuf == nil || self.detectionsBufCount != detCount {
                self.detectionsBuf?.deallocate()
                self.detectionsBuf = UnsafeMutablePointer<Float>.allocate(capacity: detCount)
                self.detectionsBufCount = detCount
            }
            guard let detBuf = self.detectionsBuf else {
                DispatchQueue.main.async { self.processing = false }
                return
            }
            self.copyFloat16MultiArrayToFloatBuffer(detectionsArr, dest: detBuf) // safe copy

            // Determine (and cache) score index and coeffStart
            let coeffStartDefault = numFeatures - K
            var coeffStart = self.cachedCoeffStart ?? coeffStartDefault
            var scoreIdx = self.cachedScoreIdx ?? 4
            if self.cachedScoreIdx == nil {
                // scan likely score positions (heuristic)
                var found: Int? = nil
                for f in 4..<min(20, numFeatures - K) {
                    var valid = 0
                    let sampleStart = max(0, numPredictions - 200)
                    for p in sampleStart..<numPredictions {
                        let v = detBuf[p * numFeatures + f]
                        if v.isFinite && v >= 0 && v <= 1 { valid += 1 }
                    }
                    if valid > 50 { found = f; break }
                }
                if let f = found { scoreIdx = f }
                self.cachedScoreIdx = scoreIdx
                self.cachedCoeffStart = coeffStart
            } else {
                // ensure cached coeffStart exists
                if self.cachedCoeffStart == nil { self.cachedCoeffStart = coeffStartDefault }
                coeffStart = self.cachedCoeffStart!
            }

            print("Using cached scoreIdx: \(scoreIdx), coeffStart: \(coeffStart)")

            // Canvas
            let scale = UIScreen.main.scale
            let canvasW = Int(round(self.bounds.width * scale))
            let canvasH = Int(round(self.bounds.height * scale))
            guard canvasW > 0 && canvasH > 0 else {
                DispatchQueue.main.async { self.processing = false }
                return
            }

            // Ensure mask float buffer
            if self.maskFloatBuf == nil || self.maskFloatBufCount != HW {
                self.maskFloatBuf?.deallocate()
                self.maskFloatBuf = UnsafeMutablePointer<Float>.allocate(capacity: HW)
                self.maskFloatBufCount = HW
            }
            guard let maskFloat = self.maskFloatBuf else {
                DispatchQueue.main.async { self.processing = false }
                return
            }

            // Ensure planar8 buffers for resizing
            let dstCount = canvasW * canvasH
            if self.planar8BufCount < max(HW, dstCount) {
                self.planar8BufA?.deallocate()
                self.planar8BufB?.deallocate()
                self.planar8BufA = UnsafeMutablePointer<UInt8>.allocate(capacity: max(HW, dstCount))
                self.planar8BufB = UnsafeMutablePointer<UInt8>.allocate(capacity: max(HW, dstCount))
                self.planar8BufCount = max(HW, dstCount)
            }
            guard let planarA = self.planar8BufA, let planarB = self.planar8BufB else {
                DispatchQueue.main.async { self.processing = false }
                return
            }

            // Collect valid detections with autoscaling & basic sanity checks
            var candidates: [(pred: Int, score: Float, coeffs: [Float])] = []
            for p in 0..<numPredictions {
                let score = detBuf[p * numFeatures + scoreIdx]
                if !score.isFinite || score < self.scoreThreshold { continue }

                // Read raw coeffs
                var raw = [Float](repeating: 0, count: K)
                var maxAbsRaw: Float = 0
                for k in 0..<K {
                    let v = detBuf[p * numFeatures + coeffStart + k]
                    if !v.isFinite { maxAbsRaw = Float.infinity; break }
                    raw[k] = v
                    maxAbsRaw = max(maxAbsRaw, abs(v))
                }
                if !raw[0].isFinite || maxAbsRaw == Float.infinity { continue }

                // Auto-scale heuristics
                var scaleFactor: Float = 1.0
                if maxAbsRaw > 400 { scaleFactor = 255.0 }
                else if maxAbsRaw > 80 { scaleFactor = 64.0 }

                let coeffs = raw.map { $0 / scaleFactor }

                // Reject nearly-uniform coefficient vectors
                var cmin = Float.greatestFiniteMagnitude, cmax = -Float.greatestFiniteMagnitude
                for v in coeffs { cmin = min(cmin, v); cmax = max(cmax, v) }
                if (cmax - cmin) < 1e-4 { continue }

                candidates.append((pred: p, score: score, coeffs: coeffs))
            }

            // Sort candidates by score and limit top-N decode
            candidates.sort { $0.score > $1.score }
            let topN = min(12, candidates.count)
            let toDecode = Array(candidates.prefix(topN))
            print("Found \(candidates.count) candidates, decoding top \(topN)")

            // Prepare prototypes matrix A for BLAS: we need A as (HW x K) row-major.
            // protoBuf currently channel-major: protoBuf[c*HW + i]
            // We will compute s = A * coeffs via cblas_sgemv by providing A as row-major with stride K.
            // Build A_rowMajor buffer once per frame (HW * K)
            let Acount = HW * K
            let Abytes = Acount * MemoryLayout<Float>.size
            let Aptr = UnsafeMutablePointer<Float>.allocate(capacity: Acount)
            // Fill row-major: for row i (pixel), columns k = protoBuf[k*HW + i]
            for i in 0..<HW {
                let baseA = i * K
                for k in 0..<K {
                    Aptr[baseA + k] = protoBuf[k * HW + i]
                }
            }

            var masksAlpha: [CGImage] = []
            var colors: [UIColor] = []

            // Per-detection decode using BLAS + vDSP sigmoid
            for (idx, c) in toDecode.enumerated() {
                let coeffs = c.coeffs
                // allocate coeff vector
                var coeffVec = coeffs // [Float] length K
                // s = A * coeffVec  (A is HW x K row-major)
                // Use cblas_sgemv: y = alpha*A*x + beta*y
                let alpha: Float = 1.0
                var s = [Float](repeating: 0, count: HW)
                // cblas_sgemv expects row-major layout if we use CblasRowMajor
                cblas_sgemv(CblasRowMajor, CblasNoTrans, Int32(HW), Int32(K), alpha, Aptr, Int32(K), &coeffVec, 1, 0, &s, 1)
                // apply sigmoid in place: s = 1 / (1 + exp(-s))
                // vDSP doesn't provide exact sigmoid; use vForce's exp then compute
                var negS = s.map { -$0 }
                var expNegS = [Float](repeating: 0, count: HW)
                vvexpf(&expNegS, &negS, [Int32(HW)])
                // s = 1/(1+expNegS)
                var one: Float = 1.0
                for i in 0..<HW {
                    s[i] = 1.0 / (1.0 + expNegS[i])
                }

                // copy s into maskFloat (threshold later)
                for i in 0..<HW { maskFloat[i] = s[i] }

                // compute coverage and threshold
                var validPixels = 0
                var minV: Float = 1, maxV: Float = 0
                for i in 0..<HW {
                    let v = maskFloat[i]
                    minV = min(minV, v); maxV = max(maxV, v)
                    //kishore
                    if v > 0.5 { validPixels += 1 } else { maskFloat[i] = 0 }
                }
                let coveragePct = Float(validPixels) / Float(HW) * 100.0

                if self.debugSaveImages {
                    self.saveDebugFloatMask(maskFloat, width: self.protoW, height: self.protoH, name: "mask_proto_\(c.pred)", timestamp: "")
                }

                // Accept masks with moderate coverage
                let minCov = HW * 5 / 100
                let maxCov = HW * 90 / 100
                if validPixels < minCov || validPixels > maxCov {
                    let formattedCoverage = String(format: "%.1f", coveragePct)
                    print("Skip pred \(c.pred) cov \(formattedCoverage)%")
                    continue
                }

                // Resize to canvas with improved vImage pipeline (blur → convert → high-quality scale → post-blur)
                guard let alphaCG = self.resizeFloatMaskToAlphaImageOptimized(maskFloat: maskFloat, srcW: self.protoW, srcH: self.protoH, dstW: canvasW, dstH: canvasH, tmpU8A: planarA, tmpU8B: planarB) else {
                    print("Failed resize")
                    continue
                }

                masksAlpha.append(alphaCG)
                let assignedColor = self.colorForIndex(idx)
                colors.append(assignedColor)
                let formattedCoverage = String(format: "%.1f", coveragePct)
                print("✅ Added mask pred \(c.pred) as detection \(idx) - coverage \(formattedCoverage)%, color: \(assignedColor)")
            }

            Aptr.deallocate()

            let frameTime = CFAbsoluteTimeGetCurrent() - frameStart
            print("Frame total time: \(Int(frameTime * 1000))ms, masks: \(masksAlpha.count)")

            // Always use additive composite to show all detections in different colors
            var outImage: UIImage?
            if self.debugShowTopMask && masksAlpha.count == 1, let top = masksAlpha.first {
                print("Using single mask composite (only 1 detection)")
                outImage = self.composeSingleMask(top, color: colors.first ?? .red, canvasW: canvasW, canvasH: canvasH)
                print("Single mask composite result: \(outImage != nil)")
            } else {
                print("Using additive composite for \(masksAlpha.count) detections")
                outImage = self.compositeMasksAdditive(masksAlpha: masksAlpha, colors: colors, canvasW: canvasW, canvasH: canvasH)
                print("Additive composite result: \(outImage != nil)")
            }

            DispatchQueue.main.async {
                self.maskImageView.image = outImage
                self.processing = false
            }
        } // detectionQueue
    }

    // MARK: - Helpers

    // Convert Float16 MLMultiArray -> Float32 buffer using vImage (fast)
    private func copyFloat16MultiArrayToFloatBuffer(_ arr: MLMultiArray, dest: UnsafeMutablePointer<Float>) {
        let count = arr.count
        // If data type is float16 stored as UInt16 bits
        if arr.dataType == .float16 {
            let src = arr.dataPointer.bindMemory(to: UInt16.self, capacity: count)
            var srcBuf = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: src), height: 1, width: vImagePixelCount(count), rowBytes: count * MemoryLayout<UInt16>.size)
            var dstBuf = vImage_Buffer(data: UnsafeMutableRawPointer(dest), height: 1, width: vImagePixelCount(count), rowBytes: count * MemoryLayout<Float>.size)
            let err = vImageConvert_Planar16FtoPlanarF(&srcBuf, &dstBuf, vImage_Flags(kvImageNoFlags))
            if err != kvImageNoError {
                // fallback elementwise conversion
                for i in 0..<count {
                    dest[i] = float32FromFloat16Bits(src[i])
                }
            }
        } else {
            // float32
            let src = arr.dataPointer.bindMemory(to: Float32.self, capacity: arr.count)
            dest.initialize(from: src, count: arr.count)
        }
    }
    private func float32FromFloat16Bits(_ bits: UInt16) -> Float {
        var b = bits
        var out: Float = 0
        var sbuf = vImage_Buffer(data: &b, height: 1, width: 1, rowBytes: 2)
        var dbuf = vImage_Buffer(data: &out, height: 1, width: 1, rowBytes: 4)
        vImageConvert_Planar16FtoPlanarF(&sbuf, &dbuf, vImage_Flags(kvImageNoFlags))
        return out
    }

    // Optimized resize with pre/post blur to avoid striping
    // tmpU8A and tmpU8B are temporary planar8 buffers allocated by caller to avoid allocations
    private func resizeFloatMaskToAlphaImageOptimized(maskFloat: UnsafePointer<Float>, srcW: Int, srcH: Int, dstW: Int, dstH: Int, tmpU8A: UnsafeMutablePointer<UInt8>, tmpU8B: UnsafeMutablePointer<UInt8>) -> CGImage? {
        let srcCount = srcW * srcH

        // temp float buffer for blur
        let tmpFloat = UnsafeMutablePointer<Float>.allocate(capacity: srcCount)
        defer { tmpFloat.deallocate() }

        var srcF = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: maskFloat), height: vImagePixelCount(srcH), width: vImagePixelCount(srcW), rowBytes: srcW * MemoryLayout<Float>.size)
        var tmpF = vImage_Buffer(data: UnsafeMutableRawPointer(tmpFloat), height: vImagePixelCount(srcH), width: vImagePixelCount(srcW), rowBytes: srcW * MemoryLayout<Float>.size)

        // small 3x3 tent blur on float to remove prototype grid
        let kernel: [Float] = [1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9]
        let err = vImageConvolve_PlanarF(&srcF, &tmpF, nil, 0, 0, kernel, 3, 3, 0, vImage_Flags(kvImageEdgeExtend))
        if err != kvImageNoError {
            tmpFloat.initialize(from: maskFloat, count: srcCount)
        }

        // Convert PlanarF -> Planar8 using tmpU8A
        var tmpFForConvert = vImage_Buffer(data: UnsafeMutableRawPointer(tmpFloat), height: vImagePixelCount(srcH), width: vImagePixelCount(srcW), rowBytes: srcW * MemoryLayout<Float>.size)
        var dstU8buf = vImage_Buffer(data: tmpU8A, height: vImagePixelCount(srcH), width: vImagePixelCount(srcW), rowBytes: srcW)
        let convErr = vImageConvert_PlanarFtoPlanar8(&tmpFForConvert, &dstU8buf, 255.0, 0.0, vImage_Flags(kvImageNoFlags))
        if convErr != kvImageNoError { return nil }

        // High-quality scale to destination into tmpU8B
        var srcBuf = vImage_Buffer(data: tmpU8A, height: vImagePixelCount(srcH), width: vImagePixelCount(srcW), rowBytes: srcW)
        var dstBuf = vImage_Buffer(data: tmpU8B, height: vImagePixelCount(dstH), width: vImagePixelCount(dstW), rowBytes: dstW)
        let scaleErr = vImageScale_Planar8(&srcBuf, &dstBuf, nil, vImage_Flags(kvImageHighQualityResampling))
        if scaleErr != kvImageNoError { return nil }

        // small post box blur to smooth any remaining artifacts
        let postBufPtr = tmpU8B
        let postOutPtr = tmpU8A // reuse other buffer for output
        var postIn = vImage_Buffer(data: postBufPtr, height: vImagePixelCount(dstH), width: vImagePixelCount(dstW), rowBytes: dstW)
        var postOut = vImage_Buffer(data: postOutPtr, height: vImagePixelCount(dstH), width: vImagePixelCount(dstW), rowBytes: dstW)
        let boxErr = vImageBoxConvolve_Planar8(&postIn, &postOut, nil, 0, 0, 3, 3, UInt8(0), vImage_Flags(kvImageEdgeExtend))
        let finalPtr = (boxErr == kvImageNoError) ? postOutPtr : postBufPtr

        guard let provider = CGDataProvider(data: CFDataCreate(nil, finalPtr, dstW * dstH)) else { return nil }
        let cs = CGColorSpaceCreateDeviceGray()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue)
        guard let cg = CGImage(width: dstW, height: dstH, bitsPerComponent: 8, bitsPerPixel: 8, bytesPerRow: dstW, space: cs, bitmapInfo: bitmapInfo, provider: provider, decode: nil, shouldInterpolate: true, intent: .defaultIntent) else { return nil }
        return cg
    }

    // Composite masks with distinct colors on opaque background
    private func compositeMasksAdditive(masksAlpha: [CGImage], colors: [UIColor], canvasW: Int, canvasH: Int) -> UIImage? {
        guard masksAlpha.count == colors.count else { 
            print("compositeMasksAdditive: count mismatch - masks:\(masksAlpha.count) colors:\(colors.count)")
            return nil 
        }
        guard masksAlpha.count > 0 else {
            print("compositeMasksAdditive: no masks to composite")
            return nil
        }
        
        let scale = UIScreen.main.scale
        let size = CGSize(width: CGFloat(canvasW)/scale, height: CGFloat(canvasH)/scale)
        print("compositeMasksAdditive: canvas \(canvasW)x\(canvasH), UI size \(size), scale \(scale)")
        
        // Use opaque context with dark background
        UIGraphicsBeginImageContextWithOptions(size, true, scale)
        guard let ctx = UIGraphicsGetCurrentContext() else { 
            UIGraphicsEndImageContext()
            print("compositeMasksAdditive: failed to get graphics context")
            return nil 
        }
        
        // Fill with dark semi-transparent background to show unmasked areas
        ctx.setFillColor(UIColor.black.withAlphaComponent(0.3).cgColor)
        ctx.fill(CGRect(origin: .zero, size: size))
        
        // Composite each mask with distinct color
        for i in 0..<masksAlpha.count {
            let alphaImg = masksAlpha[i]
            let color = colors[i]
            print("compositeMasksAdditive: compositing mask \(i) - \(alphaImg.width)x\(alphaImg.height), color \(color)")
            ctx.saveGState()
            
            // Use blend mode to combine masks properly
            ctx.setBlendMode(.normal)
            ctx.clip(to: CGRect(x: 0, y: 0, width: size.width, height: size.height), mask: alphaImg)
            ctx.setFillColor(color.cgColor)
            ctx.fill(CGRect(x: 0, y: 0, width: size.width, height: size.height))
            ctx.restoreGState()
        }
        
        let out = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        print("compositeMasksAdditive: result \(out != nil), size: \(out?.size ?? .zero)")
        return out
    }

    // Compose single mask (top-1) as colored image with opaque background
    private func composeSingleMask(_ alpha: CGImage, color: UIColor, canvasW: Int, canvasH: Int) -> UIImage? {
        let scale = UIScreen.main.scale
        let size = CGSize(width: CGFloat(canvasW)/scale, height: CGFloat(canvasH)/scale)
        print("composeSingleMask: canvas \(canvasW)x\(canvasH), UI size \(size), scale \(scale)")
        print("composeSingleMask: mask \(alpha.width)x\(alpha.height), color \(color)")
        
        // Use opaque context with background
        UIGraphicsBeginImageContextWithOptions(size, true, scale)
        guard let ctx = UIGraphicsGetCurrentContext() else { 
            UIGraphicsEndImageContext()
            print("composeSingleMask: failed to get graphics context")
            return nil 
        }
        
        // Fill with dark semi-transparent background to show unmasked areas
        ctx.setFillColor(UIColor.black.withAlphaComponent(0.3).cgColor)
        ctx.fill(CGRect(origin: .zero, size: size))
        
        ctx.saveGState()
        ctx.clip(to: CGRect(x: 0, y: 0, width: size.width, height: size.height), mask: alpha)
        ctx.setFillColor(color.cgColor)
        ctx.fill(CGRect(x: 0, y: 0, width: size.width, height: size.height))
        ctx.restoreGState()
        let out = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        print("composeSingleMask: result \(out != nil), size: \(out?.size ?? .zero)")
        return out
    }

    // Save float mask as debug PNG (proto resolution)
    private func saveDebugFloatMask(_ maskFloat: UnsafePointer<Float>, width: Int, height: Int, name: String, timestamp: String = "") {
        guard debugSaveImages else { return }
        let count = width * height
        let data = UnsafeMutablePointer<UInt8>.allocate(capacity: count)
        defer { data.deallocate() }
        for i in 0..<count {
            let v = max(0.0, min(1.0, maskFloat[i]))
            data[i] = UInt8(v * 255.0)
        }
        guard let provider = CGDataProvider(data: CFDataCreate(nil, data, count)) else { return }
        let cs = CGColorSpaceCreateDeviceGray()
        guard let cg = CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 8, bytesPerRow: width, space: cs, bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue), provider: provider, decode: nil, shouldInterpolate: true, intent: .defaultIntent) else { return }
        let ui = UIImage(cgImage: cg, scale: UIScreen.main.scale, orientation: .up)
        saveDebugImage(ui, name: name, timestamp: timestamp)
    }

    // Save UIImage to Photos (debug)
    private func saveDebugImage(_ image: UIImage, name: String, timestamp: String = "") {
        guard debugSaveImages else { return }
        let ts = timestamp.isEmpty ? String(format: "%.0f", Date().timeIntervalSince1970) : timestamp
        let label = "\(name)_\(ts)"
        // overlay label
        let final = addDebugLabel(to: image, label: label)
        PHPhotoLibrary.requestAuthorization(for: .addOnly) { status in
            if status == .authorized {
                PHPhotoLibrary.shared().performChanges({
                    PHAssetChangeRequest.creationRequestForAsset(from: final)
                }) { ok, err in
                    if ok { print("Saved debug image: \(label)") } else { print("Save failed: \(err?.localizedDescription ?? "err")") }
                }
            } else { print("No photo permission") }
        }
    }
    private func addDebugLabel(to image: UIImage, label: String) -> UIImage {
        let renderer = UIGraphicsImageRenderer(size: image.size)
        return renderer.image { ctx in
            image.draw(at: .zero)
            let r = CGRect(x: 6, y: 6, width: image.size.width - 12, height: 30)
            ctx.cgContext.setFillColor(UIColor.black.withAlphaComponent(0.6).cgColor)
            ctx.cgContext.fill(r)
            let attrs: [NSAttributedString.Key: Any] = [.foregroundColor: UIColor.white, .font: UIFont.systemFont(ofSize: min(18, image.size.width/20))]
            (label as NSString).draw(in: r.insetBy(dx: 6, dy: 4), withAttributes: attrs)
        }
    }

    // MARK: - Utility conversion: CVPixelBuffer -> MLMultiArray (channels-first Float32)
    // Replace with your optimized version if available.
    func pixelBufferToMLMultiArray(_ pixelBuffer: CVPixelBuffer, width: Int, height: Int) -> MLMultiArray? {
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(pixelBuffer) else { return nil }
        let srcW = CVPixelBufferGetWidth(pixelBuffer)
        let srcH = CVPixelBufferGetHeight(pixelBuffer)
        // Create Float32 MLMultiArray [1,3,height,width]
        guard let arr = try? MLMultiArray(shape: [1, 3, NSNumber(value: height), NSNumber(value: width)], dataType: .float32) else { return nil }
        // assume kCVPixelFormatType_32BGRA
        let rowBytes = CVPixelBufferGetBytesPerRow(pixelBuffer)
        for y in 0..<min(srcH, height) {
            let row = base.advanced(by: y * rowBytes)
            for x in 0..<min(srcW, width) {
                let px = row.advanced(by: x * 4)
                let b = Float(px.load(fromByteOffset: 0, as: UInt8.self)) / 255.0
                let g = Float(px.load(fromByteOffset: 1, as: UInt8.self)) / 255.0
                let r = Float(px.load(fromByteOffset: 2, as: UInt8.self)) / 255.0
                let rIndex = 0*arr.strides[0].intValue + 0*arr.strides[1].intValue + y*arr.strides[2].intValue + x*arr.strides[3].intValue
                let gIndex = 0*arr.strides[0].intValue + 1*arr.strides[1].intValue + y*arr.strides[2].intValue + x*arr.strides[3].intValue
                let bIndex = 0*arr.strides[0].intValue + 2*arr.strides[1].intValue + y*arr.strides[2].intValue + x*arr.strides[3].intValue
                arr[rIndex] = NSNumber(value: r)
                arr[gIndex] = NSNumber(value: g)
                arr[bIndex] = NSNumber(value: b)
            }
        }
        return arr
    }

    // Color palette - Distinct RGB colors for different detections
    private func colorForIndex(_ idx: Int) -> UIColor {
        let palette: [UIColor] = [
            UIColor.red,                                           // Pure red - Detection 1
            UIColor.green,                                         // Pure green - Detection 2  
            UIColor.blue,                                          // Pure blue - Detection 3
            UIColor.yellow,                                        // Pure yellow - Detection 4
            UIColor.magenta,                                       // Pure magenta - Detection 5
            UIColor.cyan,                                          // Pure cyan - Detection 6
            UIColor.orange,                                        // Orange - Detection 7
            UIColor(red: 1.0, green: 0.0, blue: 0.5, alpha: 1.0), // Hot pink - Detection 8
            UIColor(red: 0.5, green: 1.0, blue: 0.0, alpha: 1.0), // Lime - Detection 9
            UIColor(red: 0.0, green: 0.5, blue: 1.0, alpha: 1.0), // Sky blue - Detection 10
            UIColor(red: 1.0, green: 0.5, blue: 0.0, alpha: 1.0), // Orange-red - Detection 11
            UIColor(red: 0.5, green: 0.0, blue: 1.0, alpha: 1.0)  // Purple - Detection 12
        ]
        let color = palette[idx % palette.count]
        print("Detection \(idx) assigned color: \(color)")
        return color
    }
}

// MARK: - AVCaptureVideoDataOutputSampleBufferDelegate
extension SmartyPantsContainerView: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        // Log first few frames to confirm camera is working
        frameCount += 1
        
        if frameCount <= 5 {
            print("📹 SmartyPants: Frame \(frameCount) received from camera")
            if frameCount == 1 {
                print("✅ SMARTYPANTS CAMERA IS WORKING - receiving video frames!")
            }
        } else if frameCount == 6 {
            print("📹 SmartyPants: Camera working normally - suppressing frame logs...")
        }
        
        // Extract pixel buffer
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            print("❌ SmartyPants: Failed to extract pixel buffer from sample")
            return
        }
        
        // Log pixel buffer details for first few frames
        if frameCount <= 2 {
            let width = CVPixelBufferGetWidth(pixelBuffer)
            let height = CVPixelBufferGetHeight(pixelBuffer)
            let format = CVPixelBufferGetPixelFormatType(pixelBuffer)
            print("📊 SmartyPants: Pixel buffer info: \(width)x\(height), format: \(format)")
        }
        
        // Process the frame
        processFrame(pixelBuffer)
    }
}
