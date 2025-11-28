// SmartyPantsView.swift
import SwiftUI
import UIKit
import CoreML
import Accelerate
import AVFoundation
import Photos

struct SmartyPantsViewSwiftUI: UIViewRepresentable {
    let mlModel: MLModel?
    var processInterval: TimeInterval = 0.07
    var scoreThreshold: Float = 0.25
    var active: Bool = false
    var debugShowTop1: Bool = true
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

final class SmartyPantsContainerView: UIView, AVCaptureVideoDataOutputSampleBufferDelegate {
    // Config
    var processInterval: TimeInterval = 0.07
    var scoreThreshold: Float = 0.15  // Lower threshold for more detections
    var debugShowTopMask: Bool = true
    var debugSaveImages: Bool = true

    // Camera
    private let captureSession = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let sampleQueue = DispatchQueue(label: "com.furnit.smarty.sample", qos: .userInitiated)

    // UI
    private let previewLayer = AVCaptureVideoPreviewLayer()
    private let maskImageView: UIImageView = {
        let iv = UIImageView()
        iv.contentMode = .scaleAspectFill
        iv.backgroundColor = .clear
        iv.isOpaque = false
        iv.clipsToBounds = true
        iv.alpha = 1.0  // Full opacity for furniture cutouts
        return iv
    }()

    // Model & queues
    private var mlModel: MLModel?
    private let detectionQueue = DispatchQueue(label: "com.furnit.smarty.detection", qos: .userInitiated)
    private var lastProcessTime = Date.distantPast
    private var processing = false

    // Buffers / caches
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

    // Model protos dims
    private let protoK = 32
    private let protoH = 160
    private let protoW = 160

    // Init
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
        
        // Add camera preview layer (will be hidden during segmentation)
        previewLayer.session = captureSession
        previewLayer.videoGravity = .resizeAspectFill
        previewLayer.isHidden = true  // Start hidden to avoid blue background during segmentation
        layer.addSublayer(previewLayer)
        
        addSubview(maskImageView)
        maskImageView.translatesAutoresizingMaskIntoConstraints = false
        NSLayoutConstraint.activate([
            maskImageView.leadingAnchor.constraint(equalTo: leadingAnchor),
            maskImageView.trailingAnchor.constraint(equalTo: trailingAnchor),
            maskImageView.topAnchor.constraint(equalTo: topAnchor),
            maskImageView.bottomAnchor.constraint(equalTo: bottomAnchor)
        ])
        setupCamera()
    }
    override func layoutSubviews() {
        super.layoutSubviews()
        maskImageView.frame = bounds
        previewLayer.frame = bounds
    }
    deinit {
        stopCamera()
        protoFloatBuf?.deallocate()
        detectionsBuf?.deallocate()
        maskFloatBuf?.deallocate()
        planar8BufA?.deallocate()
        planar8BufB?.deallocate()
    }

    // Public
    func setModel(_ model: MLModel?) {
        detectionQueue.sync { self.mlModel = model; print("SmartyPants: model set -> \(model != nil)") }
    }
    func startIfNeeded() { requestCameraPermissionAndStart() }
    func stop() { stopCamera() }

    // MARK: - Camera setup
    private func setupCamera() {
        captureSession.beginConfiguration()
        captureSession.sessionPreset = .hd1280x720
        captureSession.inputs.forEach { captureSession.removeInput($0) }
        captureSession.outputs.forEach { captureSession.removeOutput($0) }

        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else {
            print("No back camera")
            captureSession.commitConfiguration()
            return
        }

        do {
            let input = try AVCaptureDeviceInput(device: device)
            if captureSession.canAddInput(input) { captureSession.addInput(input) }
            videoOutput.setSampleBufferDelegate(self, queue: sampleQueue)
            videoOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
            videoOutput.alwaysDiscardsLateVideoFrames = true
            if captureSession.canAddOutput(videoOutput) { captureSession.addOutput(videoOutput) }
            if let conn = videoOutput.connection(with: .video), conn.isVideoOrientationSupported {
                conn.videoOrientation = .portrait
                if conn.isVideoMirroringSupported { conn.isVideoMirrored = false }
            }
            captureSession.commitConfiguration()
            DispatchQueue.global(qos: .userInitiated).async {
                self.captureSession.startRunning()
                DispatchQueue.main.async { print("captureSession started: \(self.captureSession.isRunning)") }
            }
        } catch {
            print("Camera setup error:", error)
            captureSession.commitConfiguration()
        }
    }

    private func stopCamera() {
        DispatchQueue.global(qos: .userInitiated).async {
            if self.captureSession.isRunning { self.captureSession.stopRunning() }
            DispatchQueue.main.async { print("captureSession stopped") }
        }
    }

    private func requestCameraPermissionAndStart() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            if !captureSession.isRunning {
                DispatchQueue.global(qos: .userInitiated).async { self.captureSession.startRunning() }
            }
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { granted in
                if granted { DispatchQueue.global(qos: .userInitiated).async { self.captureSession.startRunning() } }
            }
        default:
            print("Camera permission denied")
        }
    }

    // MARK: - Capture delegate
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        detectionQueue.async { [weak self] in self?.processFrame(pixelBuffer) }
    }

    // MARK: - Main processing
    func processFrame(_ pixelBuffer: CVPixelBuffer) {
        guard let model = mlModel else { return }
        let now = Date()
        guard now.timeIntervalSince(lastProcessTime) >= processInterval, !processing else { return }
        lastProcessTime = now
        let frameStart = CFAbsoluteTimeGetCurrent()
        DispatchQueue.main.async { self.processing = true }

        detectionQueue.async { [weak self] in
            guard let self = self else { 
                return 
            }
            // STAGE 1: Save original camera frame from pixelBuffer
            if self.debugSaveImages, let originalCameraFrame = self.pixelBufferToCGImage(pixelBuffer) {
                let originalImage = UIImage(cgImage: originalCameraFrame)
                self.saveDebugImage(originalImage, name: "stage1_original_camera", timestamp: "")
                print("📸 STAGE 1: Saved original camera frame \(originalCameraFrame.width)x\(originalCameraFrame.height)")
            }
            
            // convert to MLMultiArray (channels-first float32)
            guard let inputArray = self.pixelBufferToMLMultiArray(pixelBuffer, width: 640, height: 640) else {
                DispatchQueue.main.async { self.processing = false }; return
            }
            
            // STAGE 2: Save what we're sending to ML model (convert MLMultiArray back to image for debugging)
            if self.debugSaveImages {
                if let mlInputImage = self.mlMultiArrayToUIImage(inputArray, width: 640, height: 640) {
                    self.saveDebugImage(mlInputImage, name: "stage2_ml_input", timestamp: "")
                    print("📸 STAGE 2: Saved ML model input 640x640")
                }
            }
            guard let inputProvider = try? MLDictionaryFeatureProvider(dictionary: ["image": inputArray]),
                  let output = try? model.prediction(from: inputProvider),
                  let prototypesArr = output.featureValue(for: "p")?.multiArrayValue,
                  let detectionsArr = output.featureValue(for: "var_2421")?.multiArrayValue else {
                DispatchQueue.main.async { self.processing = false }; return
            }
            
            // STAGE 3: Save raw ML model outputs for debugging
            if self.debugSaveImages {
                print("📸 STAGE 3: ML Model outputs:")
                print("  - prototypes: \(prototypesArr.shape) = \(prototypesArr.count) values")
                print("  - detections: \(detectionsArr.shape) = \(detectionsArr.count) values")
                
                // Save a sample prototype channel as image
                self.saveRawPrototypeAsImage(prototypesArr, channelIndex: 0, name: "stage3a_prototype_ch0")
                self.saveRawPrototypeAsImage(prototypesArr, channelIndex: 15, name: "stage3b_prototype_ch15") 
                self.saveRawPrototypeAsImage(prototypesArr, channelIndex: 31, name: "stage3c_prototype_ch31")
            }

            let numPredictions = detectionsArr.shape[1].intValue
            let numFeatures = detectionsArr.shape[2].intValue
            let K = self.protoK
            let HW = self.protoH * self.protoW

            // Proto buffer
            let protoCount = prototypesArr.count
            if self.protoFloatBuf == nil || self.protoFloatCount != protoCount {
                self.protoFloatBuf?.deallocate()
                self.protoFloatBuf = UnsafeMutablePointer<Float>.allocate(capacity: protoCount)
                self.protoFloatCount = protoCount
            }
            guard let protoBuf = self.protoFloatBuf else { DispatchQueue.main.async { self.processing = false }; return }
            self.copyFloat16MultiArrayToFloatBuffer(prototypesArr, dest: protoBuf)

            // Detections buffer copy (flattened)
            let detCount = detectionsArr.count
            if self.detectionsBuf == nil || self.detectionsBufCount != detCount {
                self.detectionsBuf?.deallocate()
                self.detectionsBuf = UnsafeMutablePointer<Float>.allocate(capacity: detCount)
                self.detectionsBufCount = detCount
            }
            guard let detBuf = self.detectionsBuf else { DispatchQueue.main.async { self.processing = false }; return }
            self.copyFloat16MultiArrayToFloatBuffer(detectionsArr, dest: detBuf)

            // Determine scoreIdx & coeffStart (cache)
            let coeffStartDefault = numFeatures - K
            var coeffStart = self.cachedCoeffStart ?? coeffStartDefault
            var scoreIdx = self.cachedScoreIdx ?? 4
            
            // Debug: Print raw scores from different feature indices
            print("🔍 RAW SCORES DEBUG - First 5 predictions:")
            for p in 0..<min(5, numPredictions) {
                print("  Pred \(p):")
                for f in 0..<min(10, numFeatures) {
                    let rawVal = detBuf[p * numFeatures + f]
                    print("    feature[\(f)] = \(rawVal)")
                }
                print("    ...")
            }
            print("🎯 Using scoreIdx: \(scoreIdx)")
            if self.cachedScoreIdx == nil {
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
                if self.cachedCoeffStart == nil { self.cachedCoeffStart = coeffStartDefault }
                coeffStart = self.cachedCoeffStart!
            }

            // Prepare mask float buffer
            if self.maskFloatBuf == nil || self.maskFloatBufCount != HW {
                self.maskFloatBuf?.deallocate()
                self.maskFloatBuf = UnsafeMutablePointer<Float>.allocate(capacity: HW)
                self.maskFloatBufCount = HW
            }
            guard let maskFloat = self.maskFloatBuf else { DispatchQueue.main.async { self.processing = false }; return }

            // Planar8 buffers for resizing
            let scale = UIScreen.main.scale
            let canvasW = Int(round(self.bounds.width * UIScreen.main.scale))
//            let canvasW = Int(round(self.bounds.width))
            let canvasH = Int(round(self.bounds.height * UIScreen.main.scale))
//            let canvasH = Int(round(self.bounds.height))
            let dstCount = canvasW * canvasH
            if self.planar8BufCount < max(HW, dstCount) {
                self.planar8BufA?.deallocate()
                self.planar8BufB?.deallocate()
                self.planar8BufA = UnsafeMutablePointer<UInt8>.allocate(capacity: max(HW, dstCount))
                self.planar8BufB = UnsafeMutablePointer<UInt8>.allocate(capacity: max(HW, dstCount))
                self.planar8BufCount = max(HW, dstCount)
            }
            guard let planarA = self.planar8BufA, let planarB = self.planar8BufB else { DispatchQueue.main.async { self.processing = false }; return }

            // Candidate selection: autoscale coefficients and simple sanity
            var candidates: [(pred: Int, score: Float, coeffs: [Float])] = []
            for p in 0..<numPredictions {
                let score = detBuf[p * numFeatures + scoreIdx]
                if !score.isFinite || score < self.scoreThreshold { continue }
                var raw = [Float](repeating: 0, count: K)
                var maxAbsRaw: Float = 0
                var bad = false
                for k in 0..<K {
                    let v = detBuf[p * numFeatures + coeffStart + k]
                    if !v.isFinite { bad = true; break }
                    raw[k] = v
                    maxAbsRaw = max(maxAbsRaw, abs(v))
                }
                if bad { continue }
                // 4) REPLACE AD-HOC 64/255 WITH DATA-DRIVEN CONSTANT
                // Based on your stats: absMax=633, suggested scale=159
                let targetScale: Float = 159.0  // Data-driven constant from your statistics
                let scaleFactor = max(1.0, maxAbsRaw / targetScale)
                let coeffs = raw.map { $0 / scaleFactor }
                // reject uniform
                var cmin = Float.greatestFiniteMagnitude, cmax = -Float.greatestFiniteMagnitude
                for v in coeffs { cmin = min(cmin, v); cmax = max(cmax, v) }
                if (cmax - cmin) < 1e-4 { continue }
                candidates.append((pred: p, score: score, coeffs: coeffs))
            }

            candidates = candidates.sorted { $0.score > $1.score }
            
            // Apply NMS to remove overlapping detections and extract bboxes
            let (filteredCandidates, candidateBboxes) = applyBBoxNMSToSegmentationMasksWithBboxes(candidates: candidates, iouThreshold: 1.0)
            
            // Apply mask IoU NMS for more precise overlap detection (handles cases like chair bottom overlapping with full chair)
            // Use higher threshold (0.8) for mask IoU since we want to catch subtle overlaps that bbox NMS might miss
            let finalCandidates = applyMaskIoUNMS(candidates: filteredCandidates, maskIoUThreshold: 0.7)
            candidates = finalCandidates
            
            // Recompute bboxes for final candidates after mask IoU filtering
            let finalBboxes = computeBboxesForCandidates(finalCandidates)
            
            let topN = min(12, candidates.count)
            let toDecode = Array(candidates.prefix(topN))
            print("Found \(candidates.count) candidates after bbox NMS + mask IoU NMS, decoding top \(topN)")
            
            // Build row-major A (HW x K) for BLAS
            let Acount = HW * K
            let Aptr = UnsafeMutablePointer<Float>.allocate(capacity: Acount)
            for i in 0..<HW {
                let base = i * K
                for k in 0..<K { Aptr[base + k] = protoBuf[k * HW + i] }
            }

            // Compute masks once and cache them
            var candidateMasks: [[Float]] = []
            var validCandidates: [(Int, (pred: Int, score: Float, coeffs: [Float]))] = []
            
            for (i, cand) in toDecode.enumerated() {
                // Calculate mask for this candidate - COMPUTE ONCE
                var coeffVec = cand.coeffs
                var s = [Float](repeating: 0, count: HW)
                cblas_sgemv(CblasRowMajor, CblasNoTrans, Int32(HW), Int32(K), 1.0, Aptr, Int32(K), &coeffVec, 1, 0, &s, 1)
                
                // Store RAW mask for later use (no sigmoid yet)
                candidateMasks.append(s)
                
                // Apply sigmoid only for filtering
                var negS = s.map { -$0 }
                var expNegS = [Float](repeating: 0, count: HW)
                vvexpf(&expNegS, &negS, [Int32(HW)])
                for j in 0..<HW { 
                    s[j] = 1.0 / (1.0 + expNegS[j])
                }
                
                // REJECT SATURATED MASKS EARLY (moved from later)
                let mean = s.reduce(0, +) / Float(HW)
                
                var validPixels = 0
                for j in 0..<HW { 
                    if s[j] > 0.5 { validPixels += 1 }  // MINIMAL THRESHOLD (just noise filter)
                }
                let coverage = Float(validPixels) / Float(HW)
                
                validCandidates.append((i, cand))
                
                // Print candidate mask in 20x20 grid (sampling every 8th pixel)
                print("🎯 Candidate \(i) mask as 20x20 grid (sampling every 8th pixel):")
                let candidateGridSize = 20
                let candidateStepX = self.protoW / candidateGridSize
                let candidateStepY = self.protoH / candidateGridSize
                for row in 0..<candidateGridSize {
                    var rowStr = "  "
                    for col in 0..<candidateGridSize {
                        let x = col * candidateStepX
                        let y = row * candidateStepY
                        let pixelIdx = y * self.protoW + x
                        if pixelIdx < s.count {
                            let val = s[pixelIdx]
                            rowStr += String(format: "%.3f ", val)  // Show actual value
                        } else {
                            rowStr += "---- "                       // Out of bounds
                        }
                    }
                    print(rowStr)
                }
                print("  Score: \(String(format: "%.3f", cand.score)), Mean: \(String(format: "%.3f", mean)), Coverage: \(String(format: "%.3f", coverage))")
                print("---")
                
                let coveragePct = coverage * 100.0
                print("Candidate \(i): score=\(String(format: "%.3f", cand.score)) pred=\(cand.pred) validPixels=\(validPixels) coverage=\(String(format: "%.1f", coveragePct))%")
            }
            
            print("Valid candidates after taking out saturation... filtering: \(validCandidates.count) out of \(toDecode.count)")

            var masksAlpha: [CGImage] = []
            
            // Track the best single mask AND accumulate summed pixel values across all candidates
            var bestMask: [Float]? = nil
            var bestScore: Float = 0.0
            var bestCandidateIdx = -1
            
            // Global mask to accumulate SUMMED pixels from all candidates (instead of strongest)
            var globalMask = [Float](repeating: 0.0, count: HW)
            var candidateCount = 0

            for (originalIdx, cand) in validCandidates {
                var coeffVec = cand.coeffs
                var s = [Float](repeating: 0, count: HW)
                cblas_sgemv(CblasRowMajor, CblasNoTrans, Int32(HW), Int32(K), 1.0, Aptr, Int32(K), &coeffVec, 1, 0, &s, 1)
                
                // 🔍 DEBUG: Check raw BLAS output values
                let sampleIndices = [0, HW/4, HW/2, 3*HW/4, HW-1]
                print("🔍 Raw BLAS output for candidate \(originalIdx):")
                for sampleIdx in sampleIndices {
                    if sampleIdx < HW {
                        print("  s[\(sampleIdx)] = \(String(format: "%.3f", s[sampleIdx]))")
                    }
                }
                let sMin = s.min() ?? 0.0
                let sMax = s.max() ?? 0.0
                let sAvg = s.reduce(0, +) / Float(HW)
                print("  s range: [\(String(format: "%.3f", sMin)), \(String(format: "%.3f", sMax))], avg: \(String(format: "%.3f", sAvg))")
                
                // Use pure raw BLAS values - let auto-normalization handle everything  
                print("  Using pure raw BLAS values with SUM accumulation (anti-saturation)...")
                
                for i in 0..<HW { 
                    // SUM all raw values instead of taking max - prevents saturation!
                    globalMask[i] += s[i]  // Accumulate ALL raw values
                }
                candidateCount += 1
                
                // 🔍 DEBUG: Check final clamped values
                print("🔍 Clamped BLAS output for candidate \(originalIdx):")
                for sampleIdx in sampleIndices {
                    if sampleIdx < HW {
                        print("  clamped_s[\(sampleIdx)] = \(String(format: "%.3f", s[sampleIdx]))")
                    }
                }
                let clampedMin = s.min() ?? 0.0
                let clampedMax = s.max() ?? 0.0
                let clampedAvg = s.reduce(0, +) / Float(HW)
                print("  clamped range: [\(String(format: "%.3f", clampedMin)), \(String(format: "%.3f", clampedMax))], avg: \(String(format: "%.3f", clampedAvg))")

                // Update best candidate tracking
                if cand.score > bestScore {
                    bestMask = globalMask
                    bestScore = cand.score
                    bestCandidateIdx = originalIdx
                    print("✅ New best mask: candidate \(originalIdx), score: \(String(format: "%.3f", cand.score))")
                }
            }
            
            // Optional: Apply averaging to prevent extreme values from multiple candidates
            if candidateCount > 1 {
                print("📊 Applying SUM normalization: dividing by candidate count (\(candidateCount))")
                for i in 0..<HW {
                    globalMask[i] = globalMask[i] / Float(candidateCount)
                }
            }
            
            // No thresholding - pass ALL summed values to auto-normalization  
//            if globalMask.max() ?? 0.0 > 1.0 { // Basic sanity check - any detection at all?
                print("Processing pure summed global mask with max value: \(globalMask.max() ?? 0.0)")
                // NO sigmoid, NO manual thresholding - let min-max normalization handle everything
                
                // Count all non-zero pixels (pure sum approach)
                var nonZeroPixels = 0
                for i in 0..<HW {
                    if globalMask[i] > 0.0 { nonZeroPixels += 1 }  // Count any detection
                }
                let globalCoverage = Float(nonZeroPixels) / Float(HW) * 100.0
                print("Global summed mask coverage: \(String(format: "%.1f", globalCoverage))%, non-zero pixels: \(nonZeroPixels)")
                
                // Apply bbox masking to clean up the global mask - make everything outside bbox transparent
                var cleanMask = globalMask  // Start with raw summed values
                if finalBboxes.count > 0 {
                    let combinedBbox = getCombinedBbox(bboxes: finalBboxes)
                    
                    // Check if bbox is too small (likely an error) - skip masking if so
                    let bboxWidth = combinedBbox.x2 - combinedBbox.x1
                    let bboxHeight = combinedBbox.y2 - combinedBbox.y1
                    let bboxArea = bboxWidth * bboxHeight
                    
                    if bboxArea > 0.01 { // At least 1% of image area
                        cleanMask = applyBboxMaskToFloatArray(cleanMask, bbox: combinedBbox, width: self.protoW, height: self.protoH)
                        print("📦 Applied combined bbox masking: (\(String(format: "%.3f", combinedBbox.x1)), \(String(format: "%.3f", combinedBbox.y1))) to (\(String(format: "%.3f", combinedBbox.x2)), \(String(format: "%.3f", combinedBbox.y2)))")
                        print("📦 Bbox area: \(String(format: "%.3f", bboxArea * 100))% of image")
                    } else {
                        print("📦 WARNING: Bbox too small (area: \(String(format: "%.3f", bboxArea * 100))%), skipping bbox masking")
                    }
                } else {
                    print("📦 No final bboxes computed, skipping bbox masking")
                }
                
                // Print final summed mask stats
                print("Final SUMMED GlobalMask as \(self.protoW)x\(self.protoH) grid (showing every 8th pixel for readability):")
                
                let step = 8  // Sample every 8th pixel to make output readable
                for y in stride(from: 0, to: self.protoH, by: step) {
                    var row = ""
                    for x in stride(from: 0, to: self.protoW, by: step) {
                        let pixelIdx = y * self.protoW + x
                        let val = cleanMask[pixelIdx]
                        // Show intensity based on summed values (adjusted thresholds for sum)
                        if val > 100.0 { row += "█" }       // Very strong sum
                        else if val > 70.0 { row += "▓" }   // Strong sum
                        else if val > 40.0 { row += "▒" }   // Medium sum
                        else if val > 10.0 { row += "░" }   // Weak sum
                        else { row += "·" }                 // Background/zero
                    }
                    print(row)
                }
                
                // Print GlobalMask SUMMED VALUES in 20x20 grid format
                print("🔢 GlobalMask SUMMED VALUES - 20x20 grid:")
                let valuesGridSize = 20
                let valuesStepX = self.protoW / valuesGridSize
                let valuesStepY = self.protoH / valuesGridSize
                for row in 0..<valuesGridSize {
                    var rowStr = "  "
                    for col in 0..<valuesGridSize {
                        let x = col * valuesStepX
                        let y = row * valuesStepY
                        let pixelIdx = y * self.protoW + x
                        if pixelIdx < cleanMask.count {
                            let val = cleanMask[pixelIdx]
                            rowStr += String(format: "%.3f ", val)
                        } else {
                            rowStr += "---.- "
                        }
                    }
                    print(rowStr)
                }
                
                // Also print some statistics for final accumulated summed mask
                let minVal = cleanMask.min() ?? 0.0
                let maxVal = cleanMask.max() ?? 0.0
                let avgVal = cleanMask.reduce(0, +) / Float(cleanMask.count)
                print("Final SUMMED Mask Stats - min: \(String(format: "%.3f", minVal)), max: \(String(format: "%.3f", maxVal)), avg: \(String(format: "%.3f", avgVal)), candidates: \(candidateCount))")
                
                if self.debugSaveImages {
                    self.saveDebugFloatMask(cleanMask, width: self.protoW, height: self.protoH, name: "mask_global_summed", timestamp: "")
                    
                    // Create mask with tight cyan bbox
                    let maskWithBbox = self.addTightBboxToMask(cleanMask, width: self.protoW, height: self.protoH)
                    let meanVal = cleanMask.reduce(0, +) / Float(cleanMask.count)
                    var validPixelsCount = 0
                    for val in cleanMask { if val > 0.2 { validPixelsCount += 1 } }
                    let coverageVal = Float(validPixelsCount) / Float(cleanMask.count)
                    let scoreVal = bestScore > 0 ? bestScore : (validCandidates.first?.1.score ?? 0.0)
                    let timestamp = String(format: "sum-s%.3f-m%.3f-c%.3f", scoreVal, meanVal, coverageVal)
                    self.saveDebugFloatMask(maskWithBbox, width: self.protoW, height: self.protoH, name: "mask_global_summed_with_bbox", timestamp: timestamp)
                }
                
                // Convert to display image
                let displayMask = cleanMask
                
                // STAGE 4A: Save the raw float mask before resizing
                if self.debugSaveImages {
                    self.saveDebugFloatMask(displayMask, width: self.protoW, height: self.protoH, name: "stage4a_raw_float_mask", timestamp: "")
                    print("📸 STAGE 4A: Saved raw float mask \(self.protoW)x\(self.protoH)")
                }
                
                // Debug: Print displayMask pixel values in 20x20 grid (sampling every 8th pixel)
                print("📊 DISPLAYMASK DEBUG - 20x20 grid (sampling every 8th pixel):")
                print("  Source: \(self.protoW)x\(self.protoH) = \(displayMask.count) pixels")
                let displayGridSize = 20
                let displayStepX = self.protoW / displayGridSize
                let displayStepY = self.protoH / displayGridSize
                for row in 0..<displayGridSize {
                    var rowStr = "  "
                    for col in 0..<displayGridSize {
                        let x = col * displayStepX
                        let y = row * displayStepY
                        let idx = y * self.protoW + x
                        if idx < displayMask.count {
                            let val = displayMask[idx]
                            rowStr += String(format: "%.2f ", val)
                        } else {
                            rowStr += "---- "
                        }
                    }
                    print(rowStr)
                }
                
                guard let alphaCG = self.resizeFloatMaskToAlphaImageOptimized(maskFloat: displayMask, srcW: self.protoW, srcH: self.protoH, dstW: canvasW, dstH: canvasH, tmpU8A: planarA, tmpU8B: planarB) else {
                    print("Failed to resize best mask with bbox")
                    Aptr.deallocate()
                    DispatchQueue.main.async { self.processing = false }
                    return
                }
                
                // Debug: Print alphaCG image properties and pixel values (sampling every 8th pixel)
                print("🖼️ ALPHACG DEBUG - Image properties:")
                print("  alphaCG.width = \(alphaCG.width)")
                print("  alphaCG.height = \(alphaCG.height)")
                print("  alphaCG.bitsPerComponent = \(alphaCG.bitsPerComponent)")
                print("  alphaCG.bitsPerPixel = \(alphaCG.bitsPerPixel)")
                print("  alphaCG.bytesPerRow = \(alphaCG.bytesPerRow)")
                
                // Extract pixel data from alphaCG to examine pixel values in 20x20 grid format
                if let dataProvider = alphaCG.dataProvider,
                   let pixelData = dataProvider.data {
                    let data = CFDataGetBytePtr(pixelData)
                    let dataLength = CFDataGetLength(pixelData)
                    print("  alphaCG pixel data length = \(dataLength)")
                    print("🖼️ ALPHACG DEBUG - 20x20 grid (sampling across image):")
                    print("  Source: \(alphaCG.width)x\(alphaCG.height) = \(dataLength) pixels")
                    
                    let alphaCGGridSize = 20
                    let alphaCGStepX = alphaCG.width / alphaCGGridSize
                    let alphaCGStepY = alphaCG.height / alphaCGGridSize
                    for row in 0..<alphaCGGridSize {
                        var rowStr = "  "
                        for col in 0..<alphaCGGridSize {
                            let x = col * alphaCGStepX
                            let y = row * alphaCGStepY
                            let idx = y * alphaCG.width + x
                            if idx < dataLength {
                                let pixelValue = data?[idx] ?? 0
                                let normalized = Float(pixelValue) / 255.0
                                rowStr += String(format: "%.2f ", normalized)
                            } else {
                                rowStr += "---- "
                            }
                        }
                        print(rowStr)
                    }
                } else {
                    print("  Failed to extract pixel data from alphaCG")
                }
                
                
                masksAlpha.append(alphaCG)
                
//            }
//            else {
//                print("No meaningful pixels found in global mask")
//            }

            Aptr.deallocate()

            let frameTime = CFAbsoluteTimeGetCurrent() - frameStart
            print("Frame total time: \(Int(frameTime * 1000))ms, masks: \(masksAlpha.count)")

            // Debug: Print masksAlpha pixel values in 20x20 grid
            for (maskIndex, alphaCGImage) in masksAlpha.enumerated() {
                print("🔍 MASKSALPHA[\(maskIndex)] DEBUG - 20x20 grid pixel values:")
                print("  Image size: \(alphaCGImage.width)x\(alphaCGImage.height)")
                
                if let dataProvider = alphaCGImage.dataProvider,
                   let pixelData = dataProvider.data {
                    let data = CFDataGetBytePtr(pixelData)
                    let dataLength = CFDataGetLength(pixelData)
                    let imageWidth = alphaCGImage.width
                    let imageHeight = alphaCGImage.height
                    
                    let masksGridSize = 20
                    let masksStepX = max(1, imageWidth / masksGridSize)
                    let masksStepY = max(1, imageHeight / masksGridSize)
                    
                    for row in 0..<masksGridSize {
                        var rowStr = "  "
                        for col in 0..<masksGridSize {
                            let x = min(col * masksStepX, imageWidth - 1)
                            let y = min(row * masksStepY, imageHeight - 1)
                            let idx = y * imageWidth + x
                            
                            if idx < dataLength {
                                let pixelValue = data?[idx] ?? 0
                                let normalized = Float(pixelValue) / 255.0
                                rowStr += String(format: "%.2f ", normalized)
                            } else {
                                rowStr += "---- "
                            }
                        }
                        print(rowStr)
                    }
                    print("  Stats - min: 0, max: 255, total pixels: \(dataLength)")
                } else {
                    print("  Failed to extract pixel data from masksAlpha[\(maskIndex)]")
                }
                print("---")
            }

            // Compose PURE furniture cutouts - NO overlays!
            var outImage: UIImage?
            if masksAlpha.count > 0 {
                // STAGE 4: Save the final mask before applying it to camera frame
                if self.debugSaveImages, let firstMask = masksAlpha.first {
                    let maskUIImage = UIImage(cgImage: firstMask)
                    self.saveDebugImage(maskUIImage, name: "stage4_final_mask", timestamp: "")
                    print("📸 STAGE 4: Saved final mask before applying to camera \(firstMask.width)x\(firstMask.height)")
                }
                
                // Create PURE furniture cutout from live camera frame
                if let cameraFrameImage = self.pixelBufferToCGImage(pixelBuffer),
                   let firstMask = masksAlpha.first,
                   let thresholdMask = self.createSimpleThresholdMask(from: firstMask) {
                    
                    // STAGE 5: Save threshold mask for debugging
                    if self.debugSaveImages {
                        let thresholdUIImage = UIImage(cgImage: thresholdMask)
                        self.saveDebugImage(thresholdUIImage, name: "stage5_threshold_mask", timestamp: "")
                        print("📸 STAGE 5: Saved threshold mask \(thresholdMask.width)x\(thresholdMask.height)")
                    }
                    
                    outImage = self.createPureFurnitureCutout(cameraFrame: cameraFrameImage, mask: thresholdMask)
                    
                    // STAGE 6: Save final result
                    if self.debugSaveImages, let finalResult = outImage {
                        self.saveDebugImage(finalResult, name: "stage6_final_result", timestamp: "")
                        print("📸 STAGE 6: Saved final result \(finalResult.size)")
                    }
                    
                    print("📱 PURE CUTOUT: Created furniture-only cutout with transparent background")
                } else {
                    outImage = self.createTransparentBackground(canvasW: canvasW, canvasH: canvasH)
                    print("📱 FALLBACK: Transparent background")
                }
            } else {
                // No furniture detected - fully transparent
                outImage = self.createTransparentBackground(canvasW: canvasW, canvasH: canvasH)
                print("📱 NO FURNITURE: Transparent background")
            }

            DispatchQueue.main.async {
                // Show ONLY the furniture cutout, hide camera preview to eliminate blue overlay
                if masksAlpha.count > 0 {
                    self.previewLayer.isHidden = true  // Hide camera preview layer
                    self.maskImageView.image = outImage
                    print("UI: Showing PURE furniture cutout, camera preview HIDDEN, count: \(masksAlpha.count)")
                } else {
                    self.previewLayer.isHidden = true  // Show camera when no furniture detected
                    self.maskImageView.image = nil
                    print("UI: No furniture detected, showing camera preview")
                }
                self.processing = false
            }
        }
    }

    // MARK: - Mask IoU NMS for Precise Overlap Detection
    private func applyMaskIoUNMS(candidates: [(pred: Int, score: Float, coeffs: [Float])], maskIoUThreshold: Float) -> [(pred: Int, score: Float, coeffs: [Float])] {
        guard candidates.count > 1 else { return candidates }
        
        let HW = protoH * protoW
        let K = protoK
        let numCandidates = candidates.count
        
        print("🎯 MASK IoU NMS: Processing \(numCandidates) candidates with threshold \(maskIoUThreshold)")
        
        // Build row-major A (HW x K) for BLAS
        let Acount = HW * K
        let Aptr = UnsafeMutablePointer<Float>.allocate(capacity: Acount)
        defer { Aptr.deallocate() }
        
        guard let protoBuf = self.protoFloatBuf else { return candidates }
        for i in 0..<HW {
            let base = i * K
            for k in 0..<K { Aptr[base + k] = protoBuf[k * HW + i] }
        }
        
        // Allocate buffer for all candidate masks (numCandidates × HW)
        let masksBuffer = UnsafeMutablePointer<Float>.allocate(capacity: numCandidates * HW)
        defer { masksBuffer.deallocate() }
        
        // Copy all coefficients into a matrix for batch processing
        let coeffsMatrix = UnsafeMutablePointer<Float>.allocate(capacity: numCandidates * K)
        defer { coeffsMatrix.deallocate() }
        
        for (idx, cand) in candidates.enumerated() {
            for k in 0..<K {
                coeffsMatrix[idx * K + k] = cand.coeffs[k]
            }
        }
        
        // Single BLAS call to compute ALL masks: masksBuffer = Aptr × coeffsMatrix^T
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 
                   Int32(HW), Int32(numCandidates), Int32(K),
                   1.0, Aptr, Int32(K), coeffsMatrix, Int32(K),
                   0.0, masksBuffer, Int32(numCandidates))
        
        // Apply sigmoid to ALL masks using vectorized operations
        let totalElements = numCandidates * HW
        var negMasks = [Float](repeating: 0, count: totalElements)
        var expNegMasks = [Float](repeating: 0, count: totalElements)
        var onePlusExp = [Float](repeating: 0, count: totalElements)
        var ones = [Float](repeating: 1.0, count: totalElements)
        
        vDSP_vneg(masksBuffer, 1, &negMasks, 1, vDSP_Length(totalElements))
        vvexpf(&expNegMasks, &negMasks, [Int32(totalElements)])
        vDSP_vadd(&expNegMasks, 1, &ones, 1, &onePlusExp, 1, vDSP_Length(totalElements))
        vvdivf(masksBuffer, &ones, &onePlusExp, [Int32(totalElements)])
        
        // Mask IoU calculation with detailed logging
        var keep: [Bool] = Array(repeating: true, count: numCandidates)
        let threshold: Float = 0.5
        
        var mask1Binary = [Float](repeating: 0, count: HW)
        var mask2Binary = [Float](repeating: 0, count: HW)
        var intersectionVec = [Float](repeating: 0, count: HW)
        var unionVec = [Float](repeating: 0, count: HW)
        var mask1OnlyVec = [Float](repeating: 0, count: HW)
        var mask2OnlyVec = [Float](repeating: 0, count: HW)
        
        for i in 0..<numCandidates {
            if !keep[i] { continue }
            
            let mask1Ptr = masksBuffer + i * HW
            
            for j in (i+1)..<numCandidates {
                if !keep[j] { continue }
                
                let mask2Ptr = masksBuffer + j * HW
                
                // Vectorized thresholding
                for k in 0..<HW {
                    mask1Binary[k] = mask1Ptr[k] > threshold ? 1.0 : 0.0
                    mask2Binary[k] = mask2Ptr[k] > threshold ? 1.0 : 0.0
                }
                
                // Vectorized intersection & union
                vDSP_vmin(&mask1Binary, 1, &mask2Binary, 1, &intersectionVec, 1, vDSP_Length(HW))
                vDSP_vmax(&mask1Binary, 1, &mask2Binary, 1, &unionVec, 1, vDSP_Length(HW))
                
                // Calculate mask1-only and mask2-only areas for detailed analysis (not actually needed for IoU but good for debugging)
                // mask1Only = max(0, mask1 - mask2), mask2Only = max(0, mask2 - mask1)
                for k in 0..<HW {
                    mask1OnlyVec[k] = max(0, mask1Binary[k] - mask2Binary[k])
                    mask2OnlyVec[k] = max(0, mask2Binary[k] - mask1Binary[k])
                }
                
                var intersectionSum: Float = 0
                var unionSum: Float = 0
                var mask1Sum: Float = 0
                var mask2Sum: Float = 0
                var mask1OnlySum: Float = 0
                var mask2OnlySum: Float = 0
                
                vDSP_sve(&intersectionVec, 1, &intersectionSum, vDSP_Length(HW))
                vDSP_sve(&unionVec, 1, &unionSum, vDSP_Length(HW))
                vDSP_sve(&mask1Binary, 1, &mask1Sum, vDSP_Length(HW))
                vDSP_sve(&mask2Binary, 1, &mask2Sum, vDSP_Length(HW))
                vDSP_sve(&mask1OnlyVec, 1, &mask1OnlySum, vDSP_Length(HW))
                vDSP_sve(&mask2OnlyVec, 1, &mask2OnlySum, vDSP_Length(HW))
                
                let maskIoU = unionSum > 0 ? intersectionSum / unionSum : 0.0
                
                // Calculate overlap ratios to detect containment (chair bottom inside full chair)
                let mask1ContainedInMask2 = mask1Sum > 0 ? intersectionSum / mask1Sum : 0.0  // How much of mask1 is inside mask2
                let mask2ContainedInMask1 = mask2Sum > 0 ? intersectionSum / mask2Sum : 0.0  // How much of mask2 is inside mask1
                
                print("🔍 MASK IoU Analysis:")
                print("  Candidates \(i) vs \(j): scores \(String(format: "%.3f", candidates[i].score)) vs \(String(format: "%.3f", candidates[j].score))")
                print("  Areas: mask1=\(Int(mask1Sum)), mask2=\(Int(mask2Sum)), intersection=\(Int(intersectionSum)), union=\(Int(unionSum))")
                print("  IoU: \(String(format: "%.3f", maskIoU))")
                print("  Containment: mask1_in_mask2=\(String(format: "%.3f", mask1ContainedInMask2)), mask2_in_mask1=\(String(format: "%.3f", mask2ContainedInMask1))")
                
                // Decision logic: Remove smaller mask if high containment OR high IoU
                let shouldSuppress = maskIoU > maskIoUThreshold || 
                                   mask1ContainedInMask2 > 0.7 ||  // mask1 is 70%+ inside mask2 (chair bottom in chair)
                                   mask2ContainedInMask1 > 0.7     // mask2 is 70%+ inside mask1
                
                if shouldSuppress {
                    let reason: String
                    let suppressedCandidate: Int
                    let keepCandidate: Int
                    
                    if mask1ContainedInMask2 > 0.7 {
                        // mask1 is mostly inside mask2 - keep the larger one (mask2)
                        if mask2Sum >= mask1Sum {
                            keep[i] = false
                            reason = "mask1_contained_in_larger_mask2"
                            suppressedCandidate = i
                            keepCandidate = j
                        } else {
                            keep[j] = false
                            reason = "mask2_smaller_despite_containing_mask1"
                            suppressedCandidate = j
                            keepCandidate = i
                        }
                    } else if mask2ContainedInMask1 > 0.7 {
                        // mask2 is mostly inside mask1 - keep the larger one (mask1)
                        if mask1Sum >= mask2Sum {
                            keep[j] = false
                            reason = "mask2_contained_in_larger_mask1"
                            suppressedCandidate = j
                            keepCandidate = i
                        } else {
                            keep[i] = false
                            reason = "mask1_smaller_despite_containing_mask2"
                            suppressedCandidate = i
                            keepCandidate = j
                        }
                    } else {
                        // High IoU overlap - keep higher score
                        if candidates[i].score >= candidates[j].score {
                            keep[j] = false
                            reason = "high_IoU_lower_score"
                            suppressedCandidate = j
                            keepCandidate = i
                        } else {
                            keep[i] = false
                            reason = "high_IoU_lower_score"
                            suppressedCandidate = i
                            keepCandidate = j
                            break
                        }
                    }
                    
                    print("🚫 MASK IoU NMS: Suppressed candidate \(suppressedCandidate) (reason: \(reason))")
                    print("  Kept candidate \(keepCandidate) with area \(keepCandidate == i ? Int(mask1Sum) : Int(mask2Sum))")
                    
                    if suppressedCandidate == i { break }
                } else {
                    print("✅ MASK IoU: Keeping both candidates (no significant overlap)")
                }
                print("---")
            }
        }
        
        let finalCandidates = candidates.enumerated().compactMap { keep[$0.offset] ? $0.element : nil }
        print("🎯 MASK IoU NMS: Kept \(finalCandidates.count) out of \(numCandidates) candidates")
        
        return finalCandidates
    }

    // MARK: - Fast Bounding Box NMS for Segmentation Masks with Bbox Return
    private func applyBBoxNMSToSegmentationMasksWithBboxes(candidates: [(pred: Int, score: Float, coeffs: [Float])], iouThreshold: Float) -> ([(pred: Int, score: Float, coeffs: [Float])], [(x1: Float, y1: Float, x2: Float, y2: Float)]) {
        guard candidates.count > 1 else { 
            // For single candidate, still compute bbox
            if candidates.count == 1 {
                let bbox = computeBboxForCandidate(candidates[0])
                return (candidates, [bbox])
            }
            return (candidates, []) 
        }
        
        let HW = protoH * protoW
        let K = protoK
        let numCandidates = candidates.count
        
        // Build row-major A (HW x K) for BLAS
        let Acount = HW * K
        let Aptr = UnsafeMutablePointer<Float>.allocate(capacity: Acount)
        defer { Aptr.deallocate() }
        
        guard let protoBuf = self.protoFloatBuf else { return (candidates, []) }
        for i in 0..<HW {
            let base = i * K
            for k in 0..<K { Aptr[base + k] = protoBuf[k * HW + i] }
        }
        
        // Fast bounding box extraction for each candidate
        var bboxes: [(x1: Float, y1: Float, x2: Float, y2: Float)] = []
        
        for cand in candidates {
            var coeffVec = cand.coeffs
            var s = [Float](repeating: 0, count: HW)
            cblas_sgemv(CblasRowMajor, CblasNoTrans, Int32(HW), Int32(K), 1.0, Aptr, Int32(K), &coeffVec, 1, 0, &s, 1)
            
            // Apply sigmoid
            for j in 0..<HW { 
                s[j] = 1.0 / (1.0 + exp(-s[j]))
            }
            
            // Fast bbox extraction using threshold - higher threshold for saturated masks
            var minX = protoW, maxX = -1, minY = protoH, maxY = -1
            let threshold: Float = 0.95  // Much higher threshold for saturated masks
            
            for y in 0..<protoH {
                for x in 0..<protoW {
                    let idx = y * protoW + x
                    if s[idx] > threshold {
                        minX = min(minX, x)
                        maxX = max(maxX, x)
                        minY = min(minY, y) 
                        maxY = max(maxY, y)
                    }
                }
            }
            
            // Convert to normalized coordinates [0,1]
            if minX <= maxX && minY <= maxY {
                let x1 = Float(minX) / Float(protoW)
                let y1 = Float(minY) / Float(protoH)
                let x2 = Float(maxX) / Float(protoW)
                let y2 = Float(maxY) / Float(protoH)
                bboxes.append((x1: x1, y1: y1, x2: x2, y2: y2))
                print("📦 Computed bbox for candidate: (\(String(format: "%.3f", x1)), \(String(format: "%.3f", y1))) to (\(String(format: "%.3f", x2)), \(String(format: "%.3f", y2)))")
            } else {
                // No valid mask - use dummy bbox
                bboxes.append((x1: 0, y1: 0, x2: 0, y2: 0))
            }
        }
        
        // Fast NMS using bounding boxes
        var keep: [Bool] = Array(repeating: true, count: numCandidates)
        
        for i in 0..<numCandidates {
            if !keep[i] { continue }
            
            for j in (i+1)..<numCandidates {
                if !keep[j] { continue }
                
                let iou = calculateBBoxIoU(bbox1: bboxes[i], bbox2: bboxes[j])
                
                if iou > iouThreshold {
                    if candidates[i].score >= candidates[j].score {
                        keep[j] = false
                        print("🚫 BBOX NMS: Suppressed candidate \(j) (IoU: \(String(format: "%.3f", iou)))")
                    } else {
                        keep[i] = false
                        print("🚫 BBOX NMS: Suppressed candidate \(i) (IoU: \(String(format: "%.3f", iou)))")
                        break
                    }
                }
            }
        }
        
        let filteredCandidates = candidates.enumerated().compactMap { keep[$0.offset] ? $0.element : nil }
        let filteredBboxes = bboxes.enumerated().compactMap { keep[$0.offset] ? $0.element : nil }
        
        return (filteredCandidates, filteredBboxes)
    }
    
    // Helper function to compute bboxes for multiple candidates
    private func computeBboxesForCandidates(_ candidates: [(pred: Int, score: Float, coeffs: [Float])]) -> [(x1: Float, y1: Float, x2: Float, y2: Float)] {
        guard !candidates.isEmpty else { return [] }
        
        let HW = protoH * protoW
        let K = protoK
        
        // Build row-major A (HW x K) for BLAS
        let Acount = HW * K
        let Aptr = UnsafeMutablePointer<Float>.allocate(capacity: Acount)
        defer { Aptr.deallocate() }
        
        guard let protoBuf = self.protoFloatBuf else { return [] }
        for i in 0..<HW {
            let base = i * K
            for k in 0..<K { Aptr[base + k] = protoBuf[k * HW + i] }
        }
        
        var bboxes: [(x1: Float, y1: Float, x2: Float, y2: Float)] = []
        
        for cand in candidates {
            var coeffVec = cand.coeffs
            var s = [Float](repeating: 0, count: HW)
            cblas_sgemv(CblasRowMajor, CblasNoTrans, Int32(HW), Int32(K), 1.0, Aptr, Int32(K), &coeffVec, 1, 0, &s, 1)
            
            // Apply sigmoid
            for j in 0..<HW { 
                s[j] = 1.0 / (1.0 + exp(-s[j]))
            }
            
            // Lower bbox threshold for post-mask-IoU candidates - they may have reduced peak values
            var minX = protoW, maxX = -1, minY = protoH, maxY = -1
            
            // Find adaptive threshold based on actual mask values
            let maxVal = s.max() ?? 0.0
            let meanVal = s.reduce(0, +) / Float(HW)
            let threshold: Float = min(0.5, max(0.1, maxVal * 0.3))  // Adaptive: 30% of max, bounded [0.1, 0.5]
            
            print("📦 Bbox extraction: maxVal=\(String(format: "%.3f", maxVal)), meanVal=\(String(format: "%.3f", meanVal)), threshold=\(String(format: "%.3f", threshold))")
            
            for y in 0..<protoH {
                for x in 0..<protoW {
                    let idx = y * protoW + x
                    if s[idx] > threshold {
                        minX = min(minX, x)
                        maxX = max(maxX, x)
                        minY = min(minY, y) 
                        maxY = max(maxY, y)
                    }
                }
            }
            
            // Convert to normalized coordinates [0,1]
            if minX <= maxX && minY <= maxY {
                let x1 = Float(minX) / Float(protoW)
                let y1 = Float(minY) / Float(protoH)
                let x2 = Float(maxX) / Float(protoW)
                let y2 = Float(maxY) / Float(protoH)
                bboxes.append((x1: x1, y1: y1, x2: x2, y2: y2))
                print("📦 Recomputed bbox for final candidate with threshold \(threshold): (\(String(format: "%.3f", x1)), \(String(format: "%.3f", y1))) to (\(String(format: "%.3f", x2)), \(String(format: "%.3f", y2)))")
            } else {
                // No valid mask - use dummy bbox
                bboxes.append((x1: 0, y1: 0, x2: 0, y2: 0))
                print("📦 No valid pixels found above threshold \(threshold) - using dummy bbox")
            }
        }
        
        return bboxes
    }

    // Helper function to compute bbox for single candidate
    private func computeBboxForCandidate(_ candidate: (pred: Int, score: Float, coeffs: [Float])) -> (x1: Float, y1: Float, x2: Float, y2: Float) {
        let HW = protoH * protoW
        let K = protoK
        
        // Build row-major A (HW x K) for BLAS
        let Acount = HW * K
        let Aptr = UnsafeMutablePointer<Float>.allocate(capacity: Acount)
        defer { Aptr.deallocate() }
        
        guard let protoBuf = self.protoFloatBuf else { return (x1: 0, y1: 0, x2: 0, y2: 0) }
        for i in 0..<HW {
            let base = i * K
            for k in 0..<K { Aptr[base + k] = protoBuf[k * HW + i] }
        }
        
        var coeffVec = candidate.coeffs
        var s = [Float](repeating: 0, count: HW)
        cblas_sgemv(CblasRowMajor, CblasNoTrans, Int32(HW), Int32(K), 1.0, Aptr, Int32(K), &coeffVec, 1, 0, &s, 1)
        
        // Apply sigmoid
        for j in 0..<HW { 
            s[j] = 1.0 / (1.0 + exp(-s[j]))
        }
        
        // Fast bbox extraction using threshold - higher threshold for saturated masks
        var minX = protoW, maxX = -1, minY = protoH, maxY = -1
        let threshold: Float = 0.95  // Much higher threshold for saturated masks
        
        for y in 0..<protoH {
            for x in 0..<protoW {
                let idx = y * protoW + x
                if s[idx] > threshold {
                    minX = min(minX, x)
                    maxX = max(maxX, x)
                    minY = min(minY, y) 
                    maxY = max(maxY, y)
                }
            }
        }
        
        // Convert to normalized coordinates [0,1]
        if minX <= maxX && minY <= maxY {
            let x1 = Float(minX) / Float(protoW)
            let y1 = Float(minY) / Float(protoH)
            let x2 = Float(maxX) / Float(protoW)
            let y2 = Float(maxY) / Float(protoH)
            return (x1: x1, y1: y1, x2: x2, y2: y2)
        } else {
            return (x1: 0, y1: 0, x2: 0, y2: 0)
        }
    }

    // MARK: - Fast Bounding Box NMS for Segmentation Masks  
    private func applyBBoxNMSToSegmentationMasks(candidates: [(pred: Int, score: Float, coeffs: [Float])], iouThreshold: Float) -> [(pred: Int, score: Float, coeffs: [Float])] {
        guard candidates.count > 1 else { return candidates }
        
        let HW = protoH * protoW
        let K = protoK
        let numCandidates = candidates.count
        
        // Build row-major A (HW x K) for BLAS
        let Acount = HW * K
        let Aptr = UnsafeMutablePointer<Float>.allocate(capacity: Acount)
        defer { Aptr.deallocate() }
        
        guard let protoBuf = self.protoFloatBuf else { return candidates }
        for i in 0..<HW {
            let base = i * K
            for k in 0..<K { Aptr[base + k] = protoBuf[k * HW + i] }
        }
        
        // Fast bounding box extraction for each candidate
        var bboxes: [(x1: Float, y1: Float, x2: Float, y2: Float)] = []
        
        for cand in candidates {
            var coeffVec = cand.coeffs
            var s = [Float](repeating: 0, count: HW)
            cblas_sgemv(CblasRowMajor, CblasNoTrans, Int32(HW), Int32(K), 1.0, Aptr, Int32(K), &coeffVec, 1, 0, &s, 1)
            
            // Apply sigmoid
            for j in 0..<HW { 
                s[j] = 1.0 / (1.0 + exp(-s[j]))
            }
            
            // Fast bbox extraction using threshold - higher threshold for saturated masks
            var minX = protoW, maxX = -1, minY = protoH, maxY = -1
            let threshold: Float = 0.95  // Much higher threshold for saturated masks
            
            for y in 0..<protoH {
                for x in 0..<protoW {
                    let idx = y * protoW + x
                    if s[idx] > threshold {
                        minX = min(minX, x)
                        maxX = max(maxX, x)
                        minY = min(minY, y) 
                        maxY = max(maxY, y)
                    }
                }
            }
            
            // Convert to normalized coordinates [0,1]
            if minX <= maxX && minY <= maxY {
                let x1 = Float(minX) / Float(protoW)
                let y1 = Float(minY) / Float(protoH)
                let x2 = Float(maxX) / Float(protoW)
                let y2 = Float(maxY) / Float(protoH)
                bboxes.append((x1: x1, y1: y1, x2: x2, y2: y2))
            } else {
                // No valid mask - use dummy bbox
                bboxes.append((x1: 0, y1: 0, x2: 0, y2: 0))
            }
        }
        
        // Fast NMS using bounding boxes
        var keep: [Bool] = Array(repeating: true, count: numCandidates)
        
        for i in 0..<numCandidates {
            if !keep[i] { continue }
            
            for j in (i+1)..<numCandidates {
                if !keep[j] { continue }
                
                let iou = calculateBBoxIoU(bbox1: bboxes[i], bbox2: bboxes[j])
                
                if iou > iouThreshold {
                    if candidates[i].score >= candidates[j].score {
                        keep[j] = false
                        print("🚫 BBOX NMS: Suppressed candidate \(j) (IoU: \(String(format: "%.3f", iou)))")
                    } else {
                        keep[i] = false
                        print("🚫 BBOX NMS: Suppressed candidate \(i) (IoU: \(String(format: "%.3f", iou)))")
                        break
                    }
                }
            }
        }
        
        return candidates.enumerated().compactMap { keep[$0.offset] ? $0.element : nil }
    }
    
    // Fast bounding box IoU calculation
    private func calculateBBoxIoU(bbox1: (x1: Float, y1: Float, x2: Float, y2: Float), 
                                   bbox2: (x1: Float, y1: Float, x2: Float, y2: Float)) -> Float {
        // Intersection coordinates
        let x1 = max(bbox1.x1, bbox2.x1)
        let y1 = max(bbox1.y1, bbox2.y1)
        let x2 = min(bbox1.x2, bbox2.x2)
        let y2 = min(bbox1.y2, bbox2.y2)
        
        // Check if there's intersection
        guard x2 > x1 && y2 > y1 else { return 0.0 }
        
        // Areas
        let intersectionArea = (x2 - x1) * (y2 - y1)
        let bbox1Area = (bbox1.x2 - bbox1.x1) * (bbox1.y2 - bbox1.y1)
        let bbox2Area = (bbox2.x2 - bbox2.x1) * (bbox2.y2 - bbox2.y1)
        let unionArea = bbox1Area + bbox2Area - intersectionArea
        
        return unionArea > 0 ? intersectionArea / unionArea : 0.0
    }

    // MARK: - NMS for Segmentation Masks (Accelerate Optimized)
    private func applyNMSToSegmentationMasks(candidates: [(pred: Int, score: Float, coeffs: [Float])], iouThreshold: Float) -> [(pred: Int, score: Float, coeffs: [Float])] {
        guard candidates.count > 1 else { return candidates }
        
        let HW = protoH * protoW
        let K = protoK
        let numCandidates = candidates.count
        
        // Build row-major A (HW x K) for BLAS
        let Acount = HW * K
        let Aptr = UnsafeMutablePointer<Float>.allocate(capacity: Acount)
        defer { Aptr.deallocate() }
        
        guard let protoBuf = self.protoFloatBuf else { return candidates }
        for i in 0..<HW {
            let base = i * K
            for k in 0..<K { Aptr[base + k] = protoBuf[k * HW + i] }
        }
        
        // Allocate buffer for all candidate masks (numCandidates × HW)
        let masksBuffer = UnsafeMutablePointer<Float>.allocate(capacity: numCandidates * HW)
        defer { masksBuffer.deallocate() }
        
        // Copy all coefficients into a matrix for batch processing
        let coeffsMatrix = UnsafeMutablePointer<Float>.allocate(capacity: numCandidates * K)
        defer { coeffsMatrix.deallocate() }
        
        for (idx, cand) in candidates.enumerated() {
            for k in 0..<K {
                coeffsMatrix[idx * K + k] = cand.coeffs[k]
            }
        }
        
        // Single BLAS call to compute ALL masks: masksBuffer = Aptr × coeffsMatrix^T
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 
                   Int32(HW), Int32(numCandidates), Int32(K),
                   1.0, Aptr, Int32(K), coeffsMatrix, Int32(K),
                   0.0, masksBuffer, Int32(numCandidates))
        
        // Apply sigmoid to ALL masks using vectorized operations
        let totalElements = numCandidates * HW
        var negMasks = [Float](repeating: 0, count: totalElements)
        var expNegMasks = [Float](repeating: 0, count: totalElements)
        var onePlusExp = [Float](repeating: 0, count: totalElements)
        var ones = [Float](repeating: 1.0, count: totalElements)
        
        vDSP_vneg(masksBuffer, 1, &negMasks, 1, vDSP_Length(totalElements))
        vvexpf(&expNegMasks, &negMasks, [Int32(totalElements)])
        vDSP_vadd(&expNegMasks, 1, &ones, 1, &onePlusExp, 1, vDSP_Length(totalElements))
        vvdivf(masksBuffer, &ones, &onePlusExp, [Int32(totalElements)])
        
        // Fast vectorized IoU calculation
        var keep: [Bool] = Array(repeating: true, count: numCandidates)
        let threshold: Float = 0.5
        
        var mask1Binary = [Float](repeating: 0, count: HW)
        var mask2Binary = [Float](repeating: 0, count: HW)
        var intersectionVec = [Float](repeating: 0, count: HW)
        var unionVec = [Float](repeating: 0, count: HW)
        
        for i in 0..<numCandidates {
            if !keep[i] { continue }
            
            let mask1Ptr = masksBuffer + i * HW
            
            for j in (i+1)..<numCandidates {
                if !keep[j] { continue }
                
                let mask2Ptr = masksBuffer + j * HW
                
                // Vectorized thresholding
                for k in 0..<HW {
                    mask1Binary[k] = mask1Ptr[k] > threshold ? 1.0 : 0.0
                    mask2Binary[k] = mask2Ptr[k] > threshold ? 1.0 : 0.0
                }
                
                // Vectorized intersection & union
                vDSP_vmin(&mask1Binary, 1, &mask2Binary, 1, &intersectionVec, 1, vDSP_Length(HW))
                vDSP_vmax(&mask1Binary, 1, &mask2Binary, 1, &unionVec, 1, vDSP_Length(HW))
                
                var intersectionSum: Float = 0
                var unionSum: Float = 0
                vDSP_sve(&intersectionVec, 1, &intersectionSum, vDSP_Length(HW))
                vDSP_sve(&unionVec, 1, &unionSum, vDSP_Length(HW))
                
                let iou = unionSum > 0 ? intersectionSum / unionSum : 0.0
                
                if iou > iouThreshold {
                    if candidates[i].score >= candidates[j].score {
                        keep[j] = false
                        print("🚫 NMS: Suppressed candidate \(j) (IoU: \(String(format: "%.3f", iou)))")
                    } else {
                        keep[i] = false
                        print("🚫 NMS: Suppressed candidate \(i) (IoU: \(String(format: "%.3f", iou)))")
                        break
                    }
                }
            }
        }
        
        return candidates.enumerated().compactMap { keep[$0.offset] ? $0.element : nil }
    }

    // MARK: - Bbox Helper Functions
    
    // Combine multiple bboxes into one that encompasses all
    private func getCombinedBbox(bboxes: [(x1: Float, y1: Float, x2: Float, y2: Float)]) -> (x1: Float, y1: Float, x2: Float, y2: Float) {
        guard !bboxes.isEmpty else { return (x1: 0, y1: 0, x2: 1, y2: 1) }
        
        var minX = Float.greatestFiniteMagnitude
        var minY = Float.greatestFiniteMagnitude
        var maxX = -Float.greatestFiniteMagnitude
        var maxY = -Float.greatestFiniteMagnitude
        
        for bbox in bboxes {
            minX = min(minX, bbox.x1)
            minY = min(minY, bbox.y1)
            maxX = max(maxX, bbox.x2)
            maxY = max(maxY, bbox.y2)
        }
        
        // Add small padding
        let padding: Float = 0.05
        minX = max(0.0, minX - padding)
        minY = max(0.0, minY - padding)
        maxX = min(1.0, maxX + padding)
        maxY = min(1.0, maxY + padding)
        
        return (x1: minX, y1: minY, x2: maxX, y2: maxY)
    }
    
    // Apply bbox mask to float array - make everything outside bbox transparent (zero)
    private func applyBboxMaskToFloatArray(_ maskFloat: [Float], bbox: (x1: Float, y1: Float, x2: Float, y2: Float), width: Int, height: Int) -> [Float] {
        var result = maskFloat
        
        // Convert normalized bbox to pixel coordinates
        let x1Pixel = Int(bbox.x1 * Float(width))
        let y1Pixel = Int(bbox.y1 * Float(height))
        let x2Pixel = Int(bbox.x2 * Float(width))
        let y2Pixel = Int(bbox.y2 * Float(height))
        
        print("📦 Bbox pixels: (\(x1Pixel), \(y1Pixel)) to (\(x2Pixel), \(y2Pixel))")
        
        var insideBboxPixels = 0
        var outsideBboxPixels = 0
        
        for y in 0..<height {
            for x in 0..<width {
                let idx = y * width + x
                
                // If pixel is outside bbox, make it transparent (zero)
                if x < x1Pixel || x > x2Pixel || y < y1Pixel || y > y2Pixel {
                    result[idx] = 0.0
                    outsideBboxPixels += 1
                } else {
                    insideBboxPixels += 1
                }
            }
        }
        
        print("📦 Bbox masking: inside=\(insideBboxPixels), outside(zeroed)=\(outsideBboxPixels)")
        return result
    }

    // MARK: - Helpers (same as earlier: conversions, resize, debug save, etc.)

    private func addTightBboxToMask(_ maskFloat: [Float], width: Int, height: Int) -> [Float] {
        var result = maskFloat
        
        // Find tight bounding box of the mask
        var minX = width, maxX = -1
        var minY = height, maxY = -1
        
        for y in 0..<height {
            for x in 0..<width {
                let idx = y * width + x
                //kish
                if maskFloat[idx] > 0.2 {  // 🎯 LOWERED THRESHOLD - KEEP MORE OBJECT PIXELS
                    minX = min(minX, x)
                    maxX = max(maxX, x)
                    minY = min(minY, y)
                    maxY = max(maxY, y)
                }
            }
        }
        
        // If we found a valid bbox, draw cyan border
        if minX <= maxX && minY <= maxY {
            // Add small padding
            let padding = 2
            minX = max(0, minX - padding)
            maxX = min(width - 1, maxX + padding)
            minY = max(0, minY - padding)
            maxY = min(height - 1, maxY + padding)
            
            print("Tight bbox: (\(minX), \(minY)) to (\(maxX), \(maxY)), size: \(maxX - minX + 1)x\(maxY - minY + 1)")
            
            // Draw cyan border (value 0.8 to distinguish from mask values)
            let borderValue: Float = 0.8
            
            // Top and bottom edges
            for x in minX...maxX {
                result[minY * width + x] = borderValue
                result[maxY * width + x] = borderValue
            }
            
            // Left and right edges
            for y in minY...maxY {
                result[y * width + minX] = borderValue
                result[y * width + maxX] = borderValue
            }
        } else {
            print("No valid bbox found for mask")
        }
        
        return result
    }

    private func copyFloat16MultiArrayToFloatBuffer(_ arr: MLMultiArray, dest: UnsafeMutablePointer<Float>) {
        let count = arr.count
        if arr.dataType == .float16 {
            let src = arr.dataPointer.bindMemory(to: UInt16.self, capacity: count)
            var srcBuf = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: src), height: 1, width: vImagePixelCount(count), rowBytes: count * MemoryLayout<UInt16>.size)
            var dstBuf = vImage_Buffer(data: UnsafeMutableRawPointer(dest), height: 1, width: vImagePixelCount(count), rowBytes: count * MemoryLayout<Float>.size)
            let err = vImageConvert_Planar16FtoPlanarF(&srcBuf, &dstBuf, vImage_Flags(kvImageNoFlags))
            if err != kvImageNoError {
                for i in 0..<count { dest[i] = float32FromFloat16Bits(src[i]) }
            }
        } else {
            let src = arr.dataPointer.bindMemory(to: Float32.self, capacity: arr.count)
            dest.initialize(from: src, count: arr.count)
        }
    }
    private func float32FromFloat16Bits(_ bits: UInt16) -> Float {
        var b = bits; var out: Float = 0
        var sbuf = vImage_Buffer(data: &b, height: 1, width: 1, rowBytes: 2)
        var dbuf = vImage_Buffer(data: &out, height: 1, width: 1, rowBytes: 4)
        vImageConvert_Planar16FtoPlanarF(&sbuf, &dbuf, vImage_Flags(kvImageNoFlags))
        return out
    }

    private func resizeFloatMaskToAlphaImageOptimized(maskFloat: [Float], srcW: Int, srcH: Int, dstW: Int, dstH: Int, tmpU8A: UnsafeMutablePointer<UInt8>, tmpU8B: UnsafeMutablePointer<UInt8>) -> CGImage? {
        return maskFloat.withUnsafeBufferPointer { bufferPtr in
            return resizeFloatMaskToAlphaImageOptimizedUnsafe(maskFloat: bufferPtr.baseAddress!, srcW: srcW, srcH: srcH, dstW: dstW, dstH: dstH, tmpU8A: tmpU8A, tmpU8B: tmpU8B)
        }
    }

    private func resizeFloatMaskToAlphaImageOptimizedUnsafe(maskFloat: UnsafePointer<Float>, srcW: Int, srcH: Int, dstW: Int, dstH: Int, tmpU8A: UnsafeMutablePointer<UInt8>, tmpU8B: UnsafeMutablePointer<UInt8>) -> CGImage? {
        let srcCount = srcW * srcH
        let tmpFloat = UnsafeMutablePointer<Float>.allocate(capacity: srcCount)
        defer { tmpFloat.deallocate() }
        var srcF = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: maskFloat), height: vImagePixelCount(srcH), width: vImagePixelCount(srcW), rowBytes: srcW * MemoryLayout<Float>.size)
        var tmpF = vImage_Buffer(data: UnsafeMutableRawPointer(tmpFloat), height: vImagePixelCount(srcH), width: vImagePixelCount(srcW), rowBytes: srcW * MemoryLayout<Float>.size)
        let kernel: [Float] = [1/9,1/9,1/9,1/9,1/9,1/9,1/9,1/9,1/9]
        let err = vImageConvolve_PlanarF(&srcF, &tmpF, nil, 0, 0, kernel, 3, 3, 0, vImage_Flags(kvImageEdgeExtend))
        if err != kvImageNoError { tmpFloat.initialize(from: maskFloat, count: srcCount) }
        
        // Debug: Check float values before conversion
        print("🔍 DEBUG: Checking float values before uint8 conversion (first 20 pixels):")
        for i in 0..<min(20, srcW * srcH) {
            let val = tmpFloat[i]
            print("  float[\(i)] = \(String(format: "%.3f", val))")
        }
        
        // Find min/max values for normalization
        var minVal = Float.greatestFiniteMagnitude
        var maxVal = -Float.greatestFiniteMagnitude
        for i in 0..<srcCount {
            let val = tmpFloat[i]
            minVal = min(minVal, val)
            maxVal = max(maxVal, val)
        }
        
        print("🔍 DEBUG: Float range before normalization: [\(String(format: "%.3f", minVal)), \(String(format: "%.3f", maxVal))]")
        
        // SIMPLE: Convert all values to 0-255 range without thresholding
        print("🔍 DEBUG: Converting all values to 0-255 range without threshold")
        
        let range = maxVal - minVal
        for i in 0..<srcCount {
            let val = tmpFloat[i]
            if range > 0.001 {
                // Simple linear scaling from min-max to 0-255
                let scaledVal = (val - minVal) / range
                tmpU8A[i] = UInt8(scaledVal * 255.0)
            } else {
                tmpU8A[i] = 128  // Neutral gray if no range
            }
        }
        
        // Debug: Check uint8 values after manual conversion
        print("🔍 DEBUG: Checking uint8 values after manual conversion (first 20 pixels):")
        for i in 0..<min(20, srcW * srcH) {
            let val = tmpU8A[i]
            print("  uint8[\(i)] = \(val) (from float: \(String(format: "%.3f", tmpFloat[i])))")
        }
        var srcBuf = vImage_Buffer(data: tmpU8A, height: vImagePixelCount(srcH), width: vImagePixelCount(srcW), rowBytes: srcW)
        var dstBuf = vImage_Buffer(data: tmpU8B, height: vImagePixelCount(dstH), width: vImagePixelCount(dstW), rowBytes: dstW)
        let scaleErr = vImageScale_Planar8(&srcBuf, &dstBuf, nil, vImage_Flags(kvImageHighQualityResampling))
        if scaleErr != kvImageNoError { return nil }
        var postIn = vImage_Buffer(data: tmpU8B, height: vImagePixelCount(dstH), width: vImagePixelCount(dstW), rowBytes: dstW)
        var postOut = vImage_Buffer(data: tmpU8A, height: vImagePixelCount(dstH), width: vImagePixelCount(dstW), rowBytes: dstW)
        var background: Pixel_8 = 0
        let boxErr = vImageBoxConvolve_Planar8(&postIn, &postOut, nil, 0, 0, 3, 3, background, vImage_Flags(kvImageEdgeExtend))

        let finalPtr = (boxErr == kvImageNoError) ? tmpU8A : tmpU8B
        
        // Debug: Print some pixel values before creating CGImage
        print("🔍 DEBUG: Checking final pixel buffer values (first 20 pixels):")
        for i in 0..<min(20, dstW * dstH) {
            let val = finalPtr[i]
            let normalized = Float(val) / 255.0
            print("  pixel[\(i)] = \(val) (normalized: \(String(format: "%.3f", normalized)))")
        }
        
        guard let provider = CGDataProvider(data: CFDataCreate(nil, finalPtr, dstW * dstH)) else { return nil }
        let cs = CGColorSpaceCreateDeviceGray()
        
        // FIXED: Create proper mask image - use .none for grayscale that works as mask
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue)
        guard let cg = CGImage(width: dstW, height: dstH, bitsPerComponent: 8, bitsPerPixel: 8, bytesPerRow: dstW, space: cs, bitmapInfo: bitmapInfo, provider: provider, decode: nil, shouldInterpolate: true, intent: .defaultIntent) else { return nil }
        return cg
    }

    // Composite mask with different colors for mask area vs bbox border
    private func compositeMaskWithBbox(masksAlpha: [CGImage], canvasW: Int, canvasH: Int) -> UIImage? {
        guard masksAlpha.count > 0 else { return nil }
        let size = CGSize(width: CGFloat(canvasW), height: CGFloat(canvasH))
        UIGraphicsBeginImageContextWithOptions(size, false, UIScreen.main.scale)
        guard let ctx = UIGraphicsGetCurrentContext() else { UIGraphicsEndImageContext(); return nil }
        ctx.clear(CGRect(origin: .zero, size: size))
        
        for alphaImg in masksAlpha {
            // We need to separate the mask pixels from bbox pixels
            // The bbox pixels have value 0.8, mask pixels have values 0.0 or 1.0
            // We'll create two separate masks and composite them with different colors
            
            ctx.saveGState()
            ctx.clip(to: CGRect(x: 0, y: 0, width: size.width, height: size.height), mask: alphaImg)
            
            // For now, use a blended approach: green base with cyan highlights
            // The bbox border (value 0.8) will show as a different intensity
            ctx.setFillColor(UIColor.green.withAlphaComponent(0.6).cgColor)
            ctx.fill(CGRect(x: 0, y: 0, width: size.width, height: size.height))
            
            // Add cyan overlay for bbox areas (these will be the 0.8 value pixels)
            ctx.setBlendMode(.overlay)
            ctx.setFillColor(UIColor.cyan.withAlphaComponent(0.8).cgColor)
            ctx.fill(CGRect(x: 0, y: 0, width: size.width, height: size.height))
            
            ctx.restoreGState()
        }
        
        let out = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()

        if let image = out {
            print("Mask+Bbox image created: \(image.size)")
        } else {
            print("Failed to create mask+bbox image")
        }

        return out
    }

    // Create black background when no masks are found
    private func createBlackBackground(canvasW: Int, canvasH: Int) -> UIImage? {
        let size = CGSize(width: CGFloat(canvasW), height: CGFloat(canvasH))
        UIGraphicsBeginImageContextWithOptions(size, false, UIScreen.main.scale)
        guard let ctx = UIGraphicsGetCurrentContext() else { UIGraphicsEndImageContext(); return nil }
        
        // Fill entire background with black
        ctx.setFillColor(UIColor.black.cgColor)
        ctx.fill(CGRect(x: 0, y: 0, width: size.width, height: size.height))
        
        let out = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        
        if let image = out {
            print("Black background created: \(image.size)")
        } else {
            print("Failed to create black background")
        }
        
        return out
    }

    // Create binary black and white mask (white for detected areas, black for background)
    private func createBinaryMask(masksAlpha: [CGImage], canvasW: Int, canvasH: Int) -> UIImage? {
        guard masksAlpha.count > 0 else { return nil }
        let size = CGSize(width: CGFloat(canvasW), height: CGFloat(canvasH))
        UIGraphicsBeginImageContextWithOptions(size, false, UIScreen.main.scale)
        guard let ctx = UIGraphicsGetCurrentContext() else { UIGraphicsEndImageContext(); return nil }
        
        // Fill entire background with black
        ctx.setFillColor(UIColor.black.cgColor)
        ctx.fill(CGRect(x: 0, y: 0, width: size.width, height: size.height))
        
        for alphaImg in masksAlpha {
            ctx.saveGState()
            ctx.clip(to: CGRect(x: 0, y: 0, width: size.width, height: size.height), mask: alphaImg)
            
            // Fill detected areas with white
            ctx.setFillColor(UIColor.white.cgColor)
            ctx.fill(CGRect(x: 0, y: 0, width: size.width, height: size.height))
            
            ctx.restoreGState()
        }
        
        let out = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()

        if let image = out {
            print("Binary mask created: \(image.size)")
        } else {
            print("Failed to create binary mask")
        }

        return out
    }

    // Composite RGB debug: all masks in red
    private func compositeMasksRGB(masksAlpha: [CGImage], colors: [UIColor], canvasW: Int, canvasH: Int) -> UIImage? {
        guard masksAlpha.count > 0 else { return nil }
//        let scale = UIScreen.main.scale
//        let size = CGSize(width: CGFloat(canvasW)/scale, height: CGFloat(canvasH)/scale)
        let size = CGSize(width: CGFloat(canvasW), height: CGFloat(canvasH))
        UIGraphicsBeginImageContextWithOptions(size, false, UIScreen.main.scale)
        guard let ctx = UIGraphicsGetCurrentContext() else { UIGraphicsEndImageContext(); return nil }
        ctx.clear(CGRect(origin: .zero, size: size))
        for (index, alphaImg) in masksAlpha.enumerated() {
            ctx.saveGState()
            ctx.clip(to: CGRect(x: 0, y: 0, width: size.width, height: size.height), mask: alphaImg)
            let color = index < colors.count ? colors[index] : UIColor.red
            ctx.setFillColor(color.cgColor)
//            ctx.setFillColor(UIColor.red.cgColor)
//            ctx.setAlpha(0.9)
//            ctx.setAlpha(2.0)
            ctx.fill(CGRect(x: 0, y: 0, width: size.width, height: size.height))
            ctx.restoreGState()
        }
        let out = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()

        // Print image information
        if let image = out {
            print("Image created: \(image.size) at scale \(image.scale)")
            print("Image has \(image.cgImage?.width ?? 0) x \(image.cgImage?.height ?? 0) pixels")
        } else {
            print("Failed to create image")
        }

        return out
    }

    // Create furniture overlay: detected areas show natural furniture, background is transparent  
    private func createTransparentOverlay(masksAlpha: [CGImage], canvasW: Int, canvasH: Int) -> UIImage? {
        guard masksAlpha.count > 0 else { return nil }
        let size = CGSize(width: CGFloat(canvasW), height: CGFloat(canvasH))
        UIGraphicsBeginImageContextWithOptions(size, false, UIScreen.main.scale)
        guard let ctx = UIGraphicsGetCurrentContext() else { UIGraphicsEndImageContext(); return nil }
        
        // Start with fully transparent background (room shows through)
        ctx.clear(CGRect(x: 0, y: 0, width: size.width, height: size.height))
        
        // Simply return the alpha masks as they are - this creates natural cutouts
        // where furniture areas will be opaque (showing natural furniture from camera)
        // and background areas will be transparent (showing room)
        for (index, alphaImg) in masksAlpha.enumerated() {
            let rect = CGRect(x: 0, y: 0, width: size.width, height: size.height)
            ctx.draw(alphaImg, in: rect)
            print("Applied natural furniture alpha mask \(index)")
        }
        
        let out = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()

        if let image = out {
            print("Natural furniture alpha overlay created: \(image.size), masks applied: \(masksAlpha.count)")
        } else {
            print("Failed to create natural furniture overlay")
        }

        return out
    }

    // Helper method to create threshold mask: pixels > threshold become white (opaque), others black (transparent)
    private func createThresholdMask(from sourceImage: CGImage, threshold: Float) -> CGImage? {
        let width = sourceImage.width
        let height = sourceImage.height
        
        guard let dataProvider = sourceImage.dataProvider,
              let pixelData = dataProvider.data else { return nil }
        
        let data = CFDataGetBytePtr(pixelData)
        let dataLength = CFDataGetLength(pixelData)
        
        let outputData = UnsafeMutablePointer<UInt8>.allocate(capacity: dataLength)
        defer { outputData.deallocate() }
        
        let thresholdByte: UInt8 = UInt8(threshold * 255.0)
        
        var opaquePixels = 0
        var transparentPixels = 0
        
        for i in 0..<dataLength {
            let pixelValue = data?[i] ?? 0
            if pixelValue > thresholdByte {
                outputData[i] = 255  // Will be opaque (show overlay)
                opaquePixels += 1
            } else {
                outputData[i] = 0    // Will be transparent (show camera)
                transparentPixels += 1
            }
        }
        
        print("🎯 Threshold \(threshold) result: opaque_pixels(>\(threshold))=\(opaquePixels), transparent_pixels(≤\(threshold))=\(transparentPixels)")
        
        guard let provider = CGDataProvider(data: CFDataCreate(nil, outputData, dataLength)) else { return nil }
        let colorSpace = CGColorSpaceCreateDeviceGray()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue)
        
        return CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 8, 
                      bytesPerRow: width, space: colorSpace, bitmapInfo: bitmapInfo, 
                      provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)
    }
    
    // Inverted threshold: pixels > 0.05 become black (transparent holes), pixels ≤ 0.05 become white (show overlay)
    private func createInvertedThresholdMask(from sourceImage: CGImage) -> CGImage? {
        let width = sourceImage.width
        let height = sourceImage.height
        
        guard let dataProvider = sourceImage.dataProvider,
              let pixelData = dataProvider.data else { return nil }
        
        let data = CFDataGetBytePtr(pixelData)
        let dataLength = CFDataGetLength(pixelData)
        
        let outputData = UnsafeMutablePointer<UInt8>.allocate(capacity: dataLength)
        defer { outputData.deallocate() }
        
        let thresholdByte: UInt8 = 128  // 0.5 * 255 = 12.55 ≈ 13
        
        var opaquePixels = 0
        var transparentPixels = 0
        
        for i in 0..<dataLength {
            let pixelValue = data?[i] ?? 0
            if pixelValue > thresholdByte {
                outputData[i] = 0    // Will be transparent holes (show camera) - DETECTED OBJECTS
                transparentPixels += 1
            } else {
                outputData[i] = 255  // Will be opaque (show green overlay) - BACKGROUND
                opaquePixels += 1
            }
        }
        
        print("🔄 Inverted threshold result: detected_objects(transparent)=\(transparentPixels), background(green_overlay)=\(opaquePixels)")
        
        guard let provider = CGDataProvider(data: CFDataCreate(nil, outputData, dataLength)) else { return nil }
        let colorSpace = CGColorSpaceCreateDeviceGray()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue)
        
        return CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 8, 
                      bytesPerRow: width, space: colorSpace, bitmapInfo: bitmapInfo, 
                      provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)
    }
    
    // Flipped threshold: pixels > 0.3 become white (will show overlay), pixels ≤ 0.3 become black (transparent)
    private func createSimpleThresholdMask(from sourceImage: CGImage) -> CGImage? {
        let width = sourceImage.width
        let height = sourceImage.height
        
        guard let dataProvider = sourceImage.dataProvider,
              let pixelData = dataProvider.data else { return nil }
        
        let data = CFDataGetBytePtr(pixelData)
        let dataLength = CFDataGetLength(pixelData)
        
        let outputData = UnsafeMutablePointer<UInt8>.allocate(capacity: dataLength)
        defer { outputData.deallocate() }
        
        let thresholdByte: UInt8 = 128  // 0.5 * 255 = 12.55 ≈ 128
        
        var opaquePixels = 0
        var transparentPixels = 0
        
        for i in 0..<dataLength {
            let pixelValue = data?[i] ?? 0
            if pixelValue > thresholdByte {
                outputData[i] = 255  // Will be opaque (show red overlay) - DETECTED OBJECTS
                opaquePixels += 1
            } else {
                outputData[i] = 0    // Will be transparent (see camera) - BACKGROUND
                transparentPixels += 1
            }
        }
        
        print("🎯 Flipped threshold result: detected_objects(>0.3)=\(opaquePixels), background(≤0.3)=\(transparentPixels)")
        
        guard let provider = CGDataProvider(data: CFDataCreate(nil, outputData, dataLength)) else { return nil }
        let colorSpace = CGColorSpaceCreateDeviceGray()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue)
        
        return CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 8, 
                      bytesPerRow: width, space: colorSpace, bitmapInfo: bitmapInfo, 
                      provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)
    }

    // Create fully transparent background when no masks are found
    private func createTransparentBackground(canvasW: Int, canvasH: Int) -> UIImage? {
        let size = CGSize(width: CGFloat(canvasW), height: CGFloat(canvasH))
        UIGraphicsBeginImageContextWithOptions(size, false, UIScreen.main.scale)
        guard let ctx = UIGraphicsGetCurrentContext() else { UIGraphicsEndImageContext(); return nil }
        
        // Fill entire background with transparent (this creates a fully transparent image)
        ctx.clear(CGRect(x: 0, y: 0, width: size.width, height: size.height))
        
        let out = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        
        if let image = out {
            print("Transparent background created: \(image.size)")
        } else {
            print("Failed to create transparent background")
        }
        
        return out
    }
    
    // Debug helper: fill entire canvas with a solid red image
    private func makeSolidRedOverlay(canvasW: Int, canvasH: Int) -> UIImage? {
        let size = CGSize(width: CGFloat(canvasW), height: CGFloat(canvasH))
        UIGraphicsBeginImageContextWithOptions(size, false, UIScreen.main.scale)
        guard let ctx = UIGraphicsGetCurrentContext() else { UIGraphicsEndImageContext(); return nil }
        ctx.setFillColor(UIColor.red.withAlphaComponent(0.5).cgColor)
        ctx.fill(CGRect(origin: .zero, size: size))
        let img = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        return img
    }
    
    // Test helper: create a small green circle to verify overlay system
//    private func makeTestGreenOverlay(canvasW: Int, canvasH: Int) -> UIImage? {
//        let size = CGSize(width: CGFloat(canvasW), height: CGFloat(canvasH))
//        UIGraphicsBeginImageContextWithOptions(size, false, 1)
//        guard let ctx = UIGraphicsGetCurrentContext() else { UIGraphicsEndImageContext(); return nil }
//        
//        // Draw a small green circle in the center
//        let centerX = CGFloat(canvasW) / 2
//        let centerY = CGFloat(canvasH) / 2
//        let radius: CGFloat = 50
//        
//        ctx.setFillColor(UIColor.green.withAlphaComponent(0.8).cgColor)
//        ctx.fillEllipse(in: CGRect(x: centerX - radius, y: centerY - radius, width: radius * 2, height: radius * 2))
//        
//        let img = UIGraphicsGetImageFromCurrentImageContext()
//        UIGraphicsEndImageContext()
//        return img
//    }


    private func saveDebugFloatMask(_ maskFloat: [Float], width: Int, height: Int, name: String, timestamp: String = "") {
        maskFloat.withUnsafeBufferPointer { bufferPtr in
            saveDebugFloatMaskUnsafe(bufferPtr.baseAddress!, width: width, height: height, name: name, timestamp: timestamp)
        }
    }

    private func saveDebugFloatMaskUnsafe(_ maskFloat: UnsafePointer<Float>, width: Int, height: Int, name: String, timestamp: String = "") {
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

    private func saveDebugImage(_ image: UIImage, name: String, timestamp: String = "") {
        guard debugSaveImages else { return }
        let ts = timestamp.isEmpty ? String(format: "%.0f", Date().timeIntervalSince1970) : timestamp
        let label = "\(name)_\(ts)"
        let final = addDebugLabel(to: image, label: label)
        PHPhotoLibrary.requestAuthorization(for: .addOnly) { status in
            if status == .authorized {
                PHPhotoLibrary.shared().performChanges({ PHAssetChangeRequest.creationRequestForAsset(from: final) }) { ok, err in
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

    // Convert CVPixelBuffer to CGImage for direct frame manipulation
    private func pixelBufferToCGImage(_ pixelBuffer: CVPixelBuffer) -> CGImage? {
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        
        let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipFirst.rawValue)
        
        guard let context = CGContext(data: baseAddress, width: width, height: height, 
                                    bitsPerComponent: 8, bytesPerRow: bytesPerRow, 
                                    space: colorSpace, bitmapInfo: bitmapInfo.rawValue) else {
            return nil
        }
        
        return context.makeImage()
    }
    
    // Create PURE furniture cutout - furniture opaque, background transparent, NO overlays!
    private func createPureFurnitureCutout(cameraFrame: CGImage, mask: CGImage) -> UIImage? {
        return createManualMaskComposite(cameraFrame: cameraFrame, mask: mask)
    }
    
    // Manual pixel-by-pixel mask application - NO CLIPPING
    private func createManualMaskComposite(cameraFrame: CGImage, mask: CGImage) -> UIImage? {
        let width = cameraFrame.width
        let height = cameraFrame.height
        
        // Ensure mask is same size as camera frame
        let resizedMask: CGImage
        if mask.width != width || mask.height != height {
            print("🔧 Resizing mask from \(mask.width)x\(mask.height) to \(width)x\(height)")
            guard let resized = resizeImage(mask, to: CGSize(width: width, height: height)) else {
                print("❌ Failed to resize mask")
                return nil
            }
            resizedMask = resized
        } else {
            resizedMask = mask
        }
        
        // Extract camera frame pixels as BGRA (iOS camera format)
        guard let cameraData = extractPixelDataAsBGRA(from: cameraFrame) else {
            print("❌ Failed to extract camera pixel data")
            return nil
        }
        
        // Extract mask pixels as grayscale
        guard let maskData = extractPixelDataAsGrayscale(from: resizedMask) else {
            print("❌ Failed to extract mask pixel data")
            return nil
        }
        
        // Create output buffer (RGBA)
        let pixelCount = width * height
        let outputData = UnsafeMutablePointer<UInt8>.allocate(capacity: pixelCount * 4)
        defer { outputData.deallocate() }
        
        // Manual pixel-by-pixel composition
        print("🔧 Manual mask application: \(width)x\(height) = \(pixelCount) pixels")
        
        // Print BEFORE threshold - show raw mask values in 20x20 grid (sampling every 8th pixel)
        print("📊 BEFORE THRESHOLD - Raw mask values (0-255) in 20x20 grid (sampling every 8th pixel):")
        let step = 8  // Sample every 8th pixel to match other debug outputs
        for y in stride(from: 0, to: min(height, 160), by: step) {  // Limit to reasonable size for readability
            var row = ""
            for x in stride(from: 0, to: min(width, 160), by: step) {
                let pixelIdx = y * width + x
                if pixelIdx < pixelCount {
                    let val = maskData[pixelIdx]
                    row += String(format: "%3d ", val)
                } else {
                    row += "--- "
                }
            }
            print("  " + row)
        }
        
        var opaquePixels = 0
        var transparentPixels = 0
        
        for i in 0..<pixelCount {
            let outputIndex = i * 4
            let cameraIndex = i * 4
            let maskValue = maskData[i] // Grayscale mask value (0-255)
            
            if maskValue > 128 { // Threshold at 50% - furniture=high values, background=low values
                // Copy camera pixel where mask is active (furniture = high values)
                // Convert BGRA to RGBA
                outputData[outputIndex + 0] = cameraData[cameraIndex + 2] // R (from B)
                outputData[outputIndex + 1] = cameraData[cameraIndex + 1] // G
                outputData[outputIndex + 2] = cameraData[cameraIndex + 0] // B (from R)  
                outputData[outputIndex + 3] = 255                         // A (fully opaque)
                opaquePixels += 1
            } else {
                // Transparent where mask is inactive (background = low values)
                outputData[outputIndex + 0] = 0   // R
                outputData[outputIndex + 1] = 0   // G
                outputData[outputIndex + 2] = 0   // B  
                outputData[outputIndex + 3] = 0   // A (transparent)
                transparentPixels += 1
            }
        }
        
        // Print AFTER threshold (255 stage) - show final alpha values in 20x20 grid (sampling every 8th pixel)
        print("📊 AFTER THRESHOLD (255 stage) - Final alpha values (0/255) in 20x20 grid (sampling every 8th pixel):")
        for y in stride(from: 0, to: min(height, 160), by: step) {
            var row = ""
            for x in stride(from: 0, to: min(width, 160), by: step) {
                let pixelIdx = y * width + x
                if pixelIdx < pixelCount {
                    let outputIndex = pixelIdx * 4
                    let finalAlpha = outputData[outputIndex + 3]
                    row += String(format: "%3d ", finalAlpha)
                } else {
                    row += "--- "
                }
            }
            print("  " + row)
        }
        
        // Print threshold decisions summary (sampling every 8th pixel)
        print("📊 THRESHOLD DECISIONS - (O=opaque >128, T=transparent ≤128) sampling every 8th pixel:")
        for y in stride(from: 0, to: min(height, 160), by: step) {
            var row = ""
            for x in stride(from: 0, to: min(width, 160), by: step) {
                let pixelIdx = y * width + x
                if pixelIdx < pixelCount {
                    let maskValue = maskData[pixelIdx]
                    let decision = maskValue > 128 ? "O" : "T"  // NORMAL
                    row += "\(decision)   "
                } else {
                    row += "-   "
                }
            }
            print("  " + row)
        }
        
        print("🎯 Mask composition: opaque=\(opaquePixels), transparent=\(transparentPixels)")
        
        // Create CGImage from output buffer
        guard let provider = CGDataProvider(data: CFDataCreate(nil, outputData, pixelCount * 4)) else {
            print("❌ Failed to create data provider")
            return nil
        }
        
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
        
        guard let outputCGImage = CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32,
                                         bytesPerRow: width * 4, space: colorSpace, bitmapInfo: bitmapInfo,
                                         provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent) else {
            print("❌ Failed to create output CGImage")
            return nil
        }
        
        let result = UIImage(cgImage: outputCGImage)
        print("✅ Manual mask composite created: \(result.size)")
        
        return result
    }
    
    // Extract pixel data from CGImage as RGBA bytes
    private func extractPixelData(from image: CGImage) -> UnsafeMutablePointer<UInt8>? {
        let width = image.width
        let height = image.height
        let pixelCount = width * height
        
        let data = UnsafeMutablePointer<UInt8>.allocate(capacity: pixelCount * 4)
        
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
        
        guard let context = CGContext(data: data, width: width, height: height, bitsPerComponent: 8,
                                     bytesPerRow: width * 4, space: colorSpace, bitmapInfo: bitmapInfo.rawValue) else {
            data.deallocate()
            return nil
        }
        
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        return data
    }
    
    // Extract pixel data from CGImage as BGRA bytes (iOS camera format)
    private func extractPixelDataAsBGRA(from image: CGImage) -> UnsafeMutablePointer<UInt8>? {
        let width = image.width
        let height = image.height
        let pixelCount = width * height
        
        let data = UnsafeMutablePointer<UInt8>.allocate(capacity: pixelCount * 4)
        
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipFirst.rawValue) // BGRX format
        
        guard let context = CGContext(data: data, width: width, height: height, bitsPerComponent: 8,
                                     bytesPerRow: width * 4, space: colorSpace, bitmapInfo: bitmapInfo.rawValue) else {
            data.deallocate()
            return nil
        }
        
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        return data
    }
    
    // Extract pixel data from grayscale mask image
    private func extractPixelDataAsGrayscale(from image: CGImage) -> UnsafeMutablePointer<UInt8>? {
        let width = image.width
        let height = image.height
        let pixelCount = width * height
        
        let data = UnsafeMutablePointer<UInt8>.allocate(capacity: pixelCount)
        
        let colorSpace = CGColorSpaceCreateDeviceGray()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue)
        
        guard let context = CGContext(data: data, width: width, height: height, bitsPerComponent: 8,
                                     bytesPerRow: width, space: colorSpace, bitmapInfo: bitmapInfo.rawValue) else {
            data.deallocate()
            return nil
        }
        
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        return data
    }
    
    // Helper function to resize CGImage
    private func resizeImage(_ image: CGImage, to newSize: CGSize) -> CGImage? {
        let width = Int(newSize.width)
        let height = Int(newSize.height)
        
        let colorSpace = image.colorSpace ?? CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = image.bitmapInfo
        
        guard let context = CGContext(data: nil, width: width, height: height, 
                                     bitsPerComponent: image.bitsPerComponent, 
                                     bytesPerRow: 0, space: colorSpace, 
                                     bitmapInfo: bitmapInfo.rawValue) else {
            return nil
        }
        
        context.interpolationQuality = .high
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        return context.makeImage()
    }

    // Apply threshold mask directly to live camera frame
    private func applyMaskToLiveFeed(cameraFrame: CGImage, mask: CGImage) -> UIImage? {
        let size = CGSize(width: cameraFrame.width, height: cameraFrame.height)
        UIGraphicsBeginImageContextWithOptions(size, false, 1.0) // Use scale 1.0 to avoid issues
        guard let ctx = UIGraphicsGetCurrentContext() else { 
            UIGraphicsEndImageContext()
            return nil 
        }
        
        // Fix 180° rotation by applying transform
//        ctx.translateBy(x: size.width, y: size.height)
//        ctx.rotate(by: .pi) // Rotate 180 degrees
        
        // Start with transparent background
        ctx.clear(CGRect(origin: .zero, size: size))
        
        // Apply the mask to show ONLY furniture (no background, no overlay)
        ctx.saveGState()
        ctx.clip(to: CGRect(origin: .zero, size: size), mask: mask)
        
        // Draw ONLY the camera frame where mask is white (furniture areas)
        // This makes furniture opaque, everything else transparent
        ctx.draw(cameraFrame, in: CGRect(origin: .zero, size: size))
        
        ctx.restoreGState()
        
        let result = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        
        if let image = result {
            print("✅ Applied mask to camera frame (furniture opaque, bg transparent): \(image.size)")
        } else {
            print("❌ Failed to apply mask to camera frame")
        }
        
        return result
    }

    // Simple pixelBuffer -> MLMultiArray (channels-first Float32) (fallback)
    private func pixelBufferToMLMultiArray(_ pixelBuffer: CVPixelBuffer, width: Int, height: Int) -> MLMultiArray? {
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(pixelBuffer) else { return nil }
        let srcW = CVPixelBufferGetWidth(pixelBuffer)
        let srcH = CVPixelBufferGetHeight(pixelBuffer)
        guard let arr = try? MLMultiArray(shape: [1,3,NSNumber(value: height),NSNumber(value: width)], dataType: .float32) else { return nil }
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
    
    // Helper: Convert MLMultiArray back to UIImage for debugging what we sent to ML model
    private func mlMultiArrayToUIImage(_ array: MLMultiArray, width: Int, height: Int) -> UIImage? {
        guard array.shape.count == 4,  // [batch, channels, height, width]
              array.shape[1].intValue == 3,  // RGB channels
              array.shape[2].intValue == height,
              array.shape[3].intValue == width else {
            print("❌ MLMultiArray shape mismatch: \(array.shape)")
            return nil
        }
        
        let pixelCount = width * height
        let data = UnsafeMutablePointer<UInt8>.allocate(capacity: pixelCount * 4) // RGBA
        defer { data.deallocate() }
        
        // Extract RGB values from channels-first format and convert to RGBA
        for y in 0..<height {
            for x in 0..<width {
                let pixelIndex = (y * width + x) * 4
                
                // Get RGB values from MLMultiArray (channels-first: [batch, channel, y, x])
                let rIndex = 0*array.strides[0].intValue + 0*array.strides[1].intValue + y*array.strides[2].intValue + x*array.strides[3].intValue
                let gIndex = 0*array.strides[0].intValue + 1*array.strides[1].intValue + y*array.strides[2].intValue + x*array.strides[3].intValue  
                let bIndex = 0*array.strides[0].intValue + 2*array.strides[1].intValue + y*array.strides[2].intValue + x*array.strides[3].intValue
                
                let r = array[rIndex].floatValue
                let g = array[gIndex].floatValue
                let b = array[bIndex].floatValue
                
                // Convert to 0-255 and store as RGBA
                data[pixelIndex + 0] = UInt8(max(0, min(255, r * 255)))     // R
                data[pixelIndex + 1] = UInt8(max(0, min(255, g * 255)))     // G  
                data[pixelIndex + 2] = UInt8(max(0, min(255, b * 255)))     // B
                data[pixelIndex + 3] = 255                                  // A (fully opaque)
            }
        }
        
        guard let provider = CGDataProvider(data: CFDataCreate(nil, data, pixelCount * 4)) else { return nil }
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue)
        
        guard let cgImage = CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32, 
                                   bytesPerRow: width * 4, space: colorSpace, bitmapInfo: bitmapInfo,
                                   provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent) else {
            return nil
        }
        
        return UIImage(cgImage: cgImage)
    }
    
    // Helper: Save a single prototype channel as grayscale image for debugging
    private func saveRawPrototypeAsImage(_ prototypes: MLMultiArray, channelIndex: Int, name: String) {
        guard prototypes.shape.count >= 3,
              channelIndex < prototypes.shape[0].intValue,  // Assuming [K, H, W] format
              prototypes.shape[1].intValue == protoH,
              prototypes.shape[2].intValue == protoW else {
            print("❌ Invalid prototype shape for debugging: \(prototypes.shape)")
            return
        }
        
        let pixelCount = protoH * protoW
        let data = UnsafeMutablePointer<UInt8>.allocate(capacity: pixelCount)
        defer { data.deallocate() }
        
        // Extract one channel from prototypes and normalize
        var minVal = Float.greatestFiniteMagnitude
        var maxVal = -Float.greatestFiniteMagnitude
        
        for y in 0..<protoH {
            for x in 0..<protoW {
                let index = channelIndex * protoH * protoW + y * protoW + x
                let val = prototypes[index].floatValue
                minVal = min(minVal, val)
                maxVal = max(maxVal, val)
            }
        }
        
        let range = maxVal - minVal
        for y in 0..<protoH {
            for x in 0..<protoW {
                let index = channelIndex * protoH * protoW + y * protoW + x
                let val = prototypes[index].floatValue
                let normalized = range > 0.001 ? (val - minVal) / range : 0.0
                data[y * protoW + x] = UInt8(normalized * 255.0)
            }
        }
        
        guard let provider = CGDataProvider(data: CFDataCreate(nil, data, pixelCount)) else { return }
        let colorSpace = CGColorSpaceCreateDeviceGray()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue)
        
        guard let cgImage = CGImage(width: protoW, height: protoH, bitsPerComponent: 8, bitsPerPixel: 8,
                                   bytesPerRow: protoW, space: colorSpace, bitmapInfo: bitmapInfo,
                                   provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent) else {
            return
        }
        
        let uiImage = UIImage(cgImage: cgImage)
        saveDebugImage(uiImage, name: name, timestamp: "ch\(channelIndex)")
        print("📸 Saved prototype channel \(channelIndex): \(protoW)x\(protoH), range: [\(String(format: "%.3f", minVal)), \(String(format: "%.3f", maxVal))]")
    }

//    private func colorForIndex(_ idx: Int) -> UIColor {
//        // Make masks more visible with different colors per index
//        switch idx {
//        case 0: return UIColor(red: 0.0, green: 1.0, blue: 0.0, alpha: 0.7)  // Green
//        case 1: return UIColor(red: 1.0, green: 0.0, blue: 0.0, alpha: 0.7)  // Red  
//        case 2: return UIColor(red: 0.0, green: 0.0, blue: 1.0, alpha: 0.7)  // Blue
//        case 3: return UIColor(red: 1.0, green: 1.0, blue: 0.0, alpha: 0.7)  // Yellow
//        case 4: return UIColor(red: 1.0, green: 0.0, blue: 1.0, alpha: 0.7)  // Magenta
//        default: return UIColor(red: 0.0, green: 1.0, blue: 1.0, alpha: 0.7) // Cyan
//        }
//    }
}
