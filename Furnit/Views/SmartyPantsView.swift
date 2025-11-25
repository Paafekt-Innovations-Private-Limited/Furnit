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
    var debugShowTopMask: Bool = false
    var debugSaveImages: Bool = false

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
        
        // Add camera preview layer
        previewLayer.session = captureSession
        previewLayer.videoGravity = .resizeAspectFill
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
            // convert to MLMultiArray (channels-first float32)
            guard let inputArray = self.pixelBufferToMLMultiArray(pixelBuffer, width: 640, height: 640) else {
                DispatchQueue.main.async { self.processing = false }; return
            }
            guard let inputProvider = try? MLDictionaryFeatureProvider(dictionary: ["image": inputArray]),
                  let output = try? model.prediction(from: inputProvider),
                  let prototypesArr = output.featureValue(for: "p")?.multiArrayValue,
                  let detectionsArr = output.featureValue(for: "var_2421")?.multiArrayValue else {
                DispatchQueue.main.async { self.processing = false }; return
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
//            let canvasW = Int(round(self.bounds.width * UIScreen.main.scale))
            let canvasW = Int(round(self.bounds.width))
//            let canvasH = Int(round(self.bounds.height * UIScreen.main.scale))
            let canvasH = Int(round(self.bounds.height))
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
                var scaleFactor: Float = 1.0
                if maxAbsRaw > 400 { scaleFactor = 255.0 } else if maxAbsRaw > 80 { scaleFactor = 64.0 }
                let coeffs = raw.map { $0 / scaleFactor }
                // reject uniform
                var cmin = Float.greatestFiniteMagnitude, cmax = -Float.greatestFiniteMagnitude
                for v in coeffs { cmin = min(cmin, v); cmax = max(cmax, v) }
                if (cmax - cmin) < 1e-4 { continue }
                candidates.append((pred: p, score: score, coeffs: coeffs))
            }

            candidates.sort { $0.score > $1.score }
            let topN = min(12, candidates.count)
            let toDecode = Array(candidates.prefix(topN))
            print("Found \(candidates.count) candidates, decoding top \(topN)")
            
            // Debug: print candidate scores and coverage requirements
            for (i, cand) in toDecode.enumerated() {
                print("Candidate \(i): score=\(String(format: "%.3f", cand.score)) pred=\(cand.pred)")
            }

            // Build row-major A (HW x K) for BLAS
            let Acount = HW * K
            let Aptr = UnsafeMutablePointer<Float>.allocate(capacity: Acount)
            for i in 0..<HW {
                let base = i * K
                for k in 0..<K { Aptr[base + k] = protoBuf[k * HW + i] }
            }

            var masksAlpha: [CGImage] = []
            
            // Track the best single mask instead of accumulating all
            var bestMask: [Float]? = nil
            var bestScore: Float = 0.0
            var bestCandidateIdx = -1

            for (idx, cand) in toDecode.enumerated() {
                var coeffVec = cand.coeffs
                var s = [Float](repeating: 0, count: HW)
                cblas_sgemv(CblasRowMajor, CblasNoTrans, Int32(HW), Int32(K), 1.0, Aptr, Int32(K), &coeffVec, 1, 0, &s, 1)
                // sigmoid: s = 1/(1+exp(-s))
                var negS = s.map { -$0 }
                var expNegS = [Float](repeating: 0, count: HW)
                vvexpf(&expNegS, &negS, [Int32(HW)])
                
                // Use local mask buffer for this iteration
                var localMaskFloat = [Float](repeating: 0, count: HW)
                for i in 0..<HW { 
                    s[i] = 1.0 / (1.0 + expNegS[i])
                    localMaskFloat[i] = s[i]
                }

                // Check if this is the best mask so far (highest score with reasonable coverage)
                var validPixels = 0
                for i in 0..<HW { 
                    if localMaskFloat[i] > 0.5 { validPixels += 1 }
                }
                let coveragePct = Float(validPixels) / Float(HW) * 100.0
                
                // Only consider masks with reasonable coverage (not too sparse, not too dense)
                let minCov = HW * 2 / 100
                let maxCov = HW * 100 / 100  // Increased max coverage - was too restrictive
                
                print("Candidate \(idx): score=\(String(format: "%.3f", cand.score)), coverage=\(String(format: "%.1f", coveragePct))%, validPixels=\(validPixels), minCov=\(minCov), maxCov=\(maxCov)")
                
                if validPixels > minCov && validPixels <= maxCov {
                    if cand.score > bestScore {
                        bestMask = localMaskFloat
                        bestScore = cand.score
                        bestCandidateIdx = idx
                        print("✅ New best mask: candidate \(idx), score: \(String(format: "%.3f", cand.score)), coverage: \(String(format: "%.1f", coveragePct))%")
                    } else {
                        print("⚡ Good coverage but lower score than current best (\(String(format: "%.3f", bestScore)))")
                    }
                } else {
                    print("❌ Rejected - coverage out of range (\(String(format: "%.1f", coveragePct))%)")
                }
            }
            
            // After evaluating all candidates, process only the best mask
            if let finalMask = bestMask, bestCandidateIdx >= 0 {
                print("Processing best mask: candidate \(bestCandidateIdx), score: \(String(format: "%.3f", bestScore))")
                
                // Apply threshold to clean up the mask
                var cleanMask = finalMask
                var validPixels = 0
                for i in 0..<HW { 
                    if cleanMask[i] > 0.5 {
                        validPixels += 1
                    } else { 
                        cleanMask[i] = 0.0  
                    }
                }
                
                let coveragePct = Float(validPixels) / Float(HW) * 100.0
                print("Final mask coverage: \(String(format: "%.1f", coveragePct))%")
                
                // Print final mask in rectangular format (160x160 grid)
                print("Final CleanMask as \(self.protoW)x\(self.protoH) grid (showing every 8th pixel for readability):")
                let step = 8  // Sample every 8th pixel to make output readable
                for y in stride(from: 0, to: self.protoH, by: step) {
                    var row = ""
                    for x in stride(from: 0, to: self.protoW, by: step) {
                        let pixelIdx = y * self.protoW + x
                        let val = cleanMask[pixelIdx]
                        // Use different characters to represent value ranges
                        if val > 0.8 { row += "█" }       // Very high confidence
                        else if val > 0.6 { row += "▓" }  // High confidence  
                        else if val > 0.4 { row += "▒" }  // Medium confidence
                        else if val > 0.2 { row += "░" }  // Low confidence
                        else { row += "·" }               // Background
                    }
                    print(row)
                }
                
                // Also print some statistics for final mask
                let minVal = cleanMask.min() ?? 0.0
                let maxVal = cleanMask.max() ?? 0.0
                let avgVal = cleanMask.reduce(0, +) / Float(cleanMask.count)
                print("Final Mask Stats - min: \(String(format: "%.3f", minVal)), max: \(String(format: "%.3f", maxVal)), avg: \(String(format: "%.3f", avgVal))")
                
                if self.debugSaveImages {
                    self.saveDebugFloatMask(cleanMask, width: self.protoW, height: self.protoH, name: "mask_best_\(bestCandidateIdx)", timestamp: "")
                }
                
                // Convert to display image
                guard let alphaCG = self.resizeFloatMaskToAlphaImageOptimized(maskFloat: cleanMask, srcW: self.protoW, srcH: self.protoH, dstW: canvasW, dstH: canvasH, tmpU8A: planarA, tmpU8B: planarB) else {
                    print("Failed to resize best mask")
                    Aptr.deallocate()
                    DispatchQueue.main.async { self.processing = false }
                    return
                }
                
                masksAlpha.append(alphaCG)
                print("Added best mask with \(String(format: "%.1f", coveragePct))% coverage")
            } else {
                print("No suitable mask found")
            }

            Aptr.deallocate()

            let frameTime = CFAbsoluteTimeGetCurrent() - frameStart
            print("Frame total time: \(Int(frameTime * 1000))ms, masks: \(masksAlpha.count)")

            // Compose RGB debug overlay
            var outImage: UIImage?
            if masksAlpha.count > 0 {
                // Use a single green color for the single best mask
                let singleColor = [UIColor(red: 0.0, green: 1.0, blue: 0.0, alpha: 0.7)]
                outImage = self.compositeMasksRGB(masksAlpha: masksAlpha, colors: singleColor, canvasW: canvasW, canvasH: canvasH)
                print("📱 OVERLAY: Created image with \(masksAlpha.count) masks")
            } else {
                print("📱 OVERLAY: No masks found")
                outImage = nil
            }

            DispatchQueue.main.async {
                self.maskImageView.layer.zPosition = 9999
                // Test with solid red overlay
                self.maskImageView.image = outImage
//                if let redImg = self.makeSolidRedOverlay(canvasW: canvasW, canvasH: canvasH) {
//                    self.maskImageView.image = redImg
//                } else {
//                    self.maskImageView.image = outImage // fallback
//                }
                print("UI: set mask image -> \(outImage != nil), masks: \(masksAlpha.count)")
                self.processing = false
            }
        }
    }

    // MARK: - Helpers (same as earlier: conversions, resize, debug save, etc.)

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
        var tmpFForConvert = vImage_Buffer(data: UnsafeMutableRawPointer(tmpFloat), height: vImagePixelCount(srcH), width: vImagePixelCount(srcW), rowBytes: srcW * MemoryLayout<Float>.size)
        var dstU8buf = vImage_Buffer(data: tmpU8A, height: vImagePixelCount(srcH), width: vImagePixelCount(srcW), rowBytes: srcW)
        let convErr = vImageConvert_PlanarFtoPlanar8(&tmpFForConvert, &dstU8buf, 255.0, 0.0, vImage_Flags(kvImageNoFlags))
        if convErr != kvImageNoError { return nil }
        var srcBuf = vImage_Buffer(data: tmpU8A, height: vImagePixelCount(srcH), width: vImagePixelCount(srcW), rowBytes: srcW)
        var dstBuf = vImage_Buffer(data: tmpU8B, height: vImagePixelCount(dstH), width: vImagePixelCount(dstW), rowBytes: dstW)
        let scaleErr = vImageScale_Planar8(&srcBuf, &dstBuf, nil, vImage_Flags(kvImageHighQualityResampling))
        if scaleErr != kvImageNoError { return nil }
        var postIn = vImage_Buffer(data: tmpU8B, height: vImagePixelCount(dstH), width: vImagePixelCount(dstW), rowBytes: dstW)
        var postOut = vImage_Buffer(data: tmpU8A, height: vImagePixelCount(dstH), width: vImagePixelCount(dstW), rowBytes: dstW)
        var background: Pixel_8 = 0
        let boxErr = vImageBoxConvolve_Planar8(&postIn, &postOut, nil, 0, 0, 3, 3, background, vImage_Flags(kvImageEdgeExtend))

        let finalPtr = (boxErr == kvImageNoError) ? tmpU8A : tmpU8B
        guard let provider = CGDataProvider(data: CFDataCreate(nil, finalPtr, dstW * dstH)) else { return nil }
        let cs = CGColorSpaceCreateDeviceGray()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue)
        guard let cg = CGImage(width: dstW, height: dstH, bitsPerComponent: 8, bitsPerPixel: 8, bytesPerRow: dstW, space: cs, bitmapInfo: bitmapInfo, provider: provider, decode: nil, shouldInterpolate: true, intent: .defaultIntent) else { return nil }
        return cg
    }

    // Composite RGB debug: all masks in red
    private func compositeMasksRGB(masksAlpha: [CGImage], colors: [UIColor], canvasW: Int, canvasH: Int) -> UIImage? {
        guard masksAlpha.count > 0 else { return nil }
//        let scale = UIScreen.main.scale
//        let size = CGSize(width: CGFloat(canvasW)/scale, height: CGFloat(canvasH)/scale)
        let size = CGSize(width: CGFloat(canvasW), height: CGFloat(canvasH))
        UIGraphicsBeginImageContextWithOptions(size, false, 1)
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
    
    // Debug helper: fill entire canvas with a solid red image
    private func makeSolidRedOverlay(canvasW: Int, canvasH: Int) -> UIImage? {
        let size = CGSize(width: CGFloat(canvasW), height: CGFloat(canvasH))
        UIGraphicsBeginImageContextWithOptions(size, false, 1)
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
