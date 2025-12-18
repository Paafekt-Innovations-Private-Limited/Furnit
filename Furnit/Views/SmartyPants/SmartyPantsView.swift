// SmartyPantsView.swift
// Single-stage: detect → NMS → keepOverlapping → union mask → cutout
// With timing at every stage

import SwiftUI
import UIKit
import CoreML
import Accelerate
import AVFoundation

// MARK: - SwiftUI Wrapper
struct SmartyPantsViewSwiftUI: UIViewRepresentable {
    let mlModel: MLModel?
    var processInterval: TimeInterval = 0.1
    var confidenceThreshold: Float = 0.05
    var iouThreshold: Float = 0.5
    var useBilinearUpscaling: Bool = true
    var debugMode: Bool = true
    var active: Bool = false

    func makeUIView(context: Context) -> SmartyPantsContainerView {
        let v = SmartyPantsContainerView()
        v.processInterval = processInterval
        v.confidenceThreshold = confidenceThreshold
        v.iouThreshold = iouThreshold
        v.useBilinearUpscaling = useBilinearUpscaling
        v.debugMode = debugMode
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
        uiView.debugMode = debugMode
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
    var confidenceThreshold: Float = 0.05
    var iouThreshold: Float = 0.5
    var useBilinearUpscaling: Bool = true
    var debugMode: Bool = true
    
    // MARK: - Ignored Classes (Structure / Room / Background / Openings)
    private let clsToIgnore: Set<Int> = [
        // ROOMS
        330, 378, 881, 951, 1064, 1080, 1259, 1290, 1323, 1406, 1518, 1573, 1973,
        2115, 2116, 2142, 2152, 2234, 2390, 2410, 2475, 2476, 3122, 3331, 3377,
        3439, 3600, 3917, 3956, 3957, 4107, 4147, 4324, 4388,
        // WALLS
        571, 944, 1887, 4164, 4536,
        // FLOOR
        1692, 1758, 1881, 2037, 2320, 4162, 4535,
        // CEILING
        802,
        // SKY
        483, 2799, 2800, 3721,
        // WINDOWS / DOORS
        1380, 1880, 4501, 1888,
        // CURTAINS / BLINDS
        1234, 467,
        // TILES
        815, 4161, 4163,
        // BUILDING / STRUCTURE
        613, 615, 616, 810, 1072, 3955,
        // ABSTRACT / SCENE
        1041, 2669, 3604, 3682, 3760, 4248, 4261, 4303, 4558, 470, 3092,
    ]

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

    // MARK: Model & State
    private var mlModel: MLModel?
    private let detectionQueue = DispatchQueue(label: "com.furnit.detection", qos: .userInitiated)
    private var lastProcessTime = Date.distantPast
    private var isProcessing = false
    
    // MARK: Class Names (loaded from classes.json)
    private lazy var classNames: [Int: String] = {
        guard let url = Bundle.main.url(forResource: "classes", withExtension: "json"),
              let data = try? Data(contentsOf: url),
              let dict = try? JSONSerialization.jsonObject(with: data) as? [String: String] else {
            print("⚠️ Failed to load classes.json")
            return [:]
        }
        var result: [Int: String] = [:]
        for (key, value) in dict {
            if let id = Int(key) {
                result[id] = value
            }
        }
        print("✅ Loaded \(result.count) class names")
        return result
    }()
    
    private func className(_ id: Int) -> String {
        return classNames[id] ?? "\(id)"
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
        addGestureRecognizer(pinchGesture)
        
        let panGesture = UIPanGestureRecognizer(target: self, action: #selector(handlePan(_:)))
        panGesture.delegate = self
        maskImageView.addGestureRecognizer(panGesture)
        
        setupCamera()
        if debugMode { print("✅ SmartyPantsContainerView initialized") }
    }
    
    override func layoutSubviews() {
        super.layoutSubviews()
        previewLayer.frame = bounds
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
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        detectionQueue.async { [weak self] in self?.processFrame(pixelBuffer) }
    }

    // MARK: - Main Processing Pipeline
    private func processFrame(_ pixelBuffer: CVPixelBuffer) {
        let frameStart = Date()
        
        guard let model = mlModel else { return }
        let now = Date()
        guard now.timeIntervalSince(lastProcessTime) >= processInterval, !isProcessing else { return }
        lastProcessTime = now
        isProcessing = true

        if debugMode {
            print("\n⏱️ ═══════════════════════════════════════════")
            print("⏱️ FRAME START @ \(String(format: "%.3f", now.timeIntervalSince1970))")
            print("⏱️ ═══════════════════════════════════════════")
        }

        // STAGE 1: Resize to square
        let t1 = Date()
        setProgress(0.15, text: "Resizing…")
        
        guard let sq = resizeToSquare(pixelBuffer, size: 1280) else {
            if debugMode { print("❌ STAGE 1 FAILED: Resize to square") }
            isProcessing = false
            return
        }
        let resizeGain = sq.gain
        let padX = sq.padX
        let padY = sq.padY
        
        let t1End = Date()
        if debugMode {
            print("⏱️ STAGE 1 - Resize: \(String(format: "%.2f", t1End.timeIntervalSince(t1) * 1000)) ms")
        }

        // STAGE 2: Convert to MLMultiArray
        let t2 = Date()
        setProgress(0.25, text: "Preprocessing…")
        
        guard let inputArray = pixelBufferToMLMultiArray(sq.buffer) else {
            if debugMode { print("❌ STAGE 2 FAILED: MLMultiArray conversion") }
            isProcessing = false
            return
        }
        
        let t2End = Date()
        if debugMode {
            print("⏱️ STAGE 2 - MLMultiArray: \(String(format: "%.2f", t2End.timeIntervalSince(t2) * 1000)) ms")
        }

        // STAGE 3: Model inference
        let t3 = Date()
        setProgress(0.40, text: "Running model…")
        
        guard let inputProvider = try? MLDictionaryFeatureProvider(dictionary: ["image": inputArray]),
              let output = try? model.prediction(from: inputProvider) else {
            if debugMode { print("❌ STAGE 3 FAILED: Model inference") }
            isProcessing = false
            return
        }
        
        let t3End = Date()
        if debugMode {
            print("⏱️ STAGE 3 - Inference: \(String(format: "%.2f", t3End.timeIntervalSince(t3) * 1000)) ms")
        }

        // STAGE 4: Extract tensors
        let t4 = Date()
        
        guard let detArray = output.featureValue(for: "var_2497")?.multiArrayValue,
              let protoArray = output.featureValue(for: "p")?.multiArrayValue else {
            if debugMode { print("❌ STAGE 4 FAILED: Missing output tensors") }
            isProcessing = false
            return
        }
        
        let numFeatures = detArray.shape[1].intValue
        let numAnchors = detArray.shape[2].intValue
        let numClasses = numFeatures - 4 - 32
        
        guard numFeatures >= 36, numAnchors > 0, numClasses > 0 else {
            if debugMode { print("❌ STAGE 4 FAILED: Invalid tensor dims") }
            isProcessing = false
            return
        }
        
        let t4End = Date()
        if debugMode {
            print("⏱️ STAGE 4 - Extract tensors: \(String(format: "%.2f", t4End.timeIntervalSince(t4) * 1000)) ms")
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
            var dstBuf = vImage_Buffer(data: detBuf, height: 1, width: vImagePixelCount(totalCount), rowBytes: totalCount * 4)
            vImageConvert_Planar16FtoPlanarF(&srcBuf, &dstBuf, vImage_Flags(kvImageNoFlags))
        }
        
        let t5End = Date()
        if debugMode {
            print("⏱️ STAGE 5 - Copy detBuf: \(String(format: "%.2f", t5End.timeIntervalSince(t5) * 1000)) ms")
        }

        // STAGE 6: Extract detections
        let t6 = Date()
        setProgress(0.55, text: "Extracting detections…")
        
        let stride = numAnchors
        let coeffOffset = 4 + numClasses
        var tempScores = [Float](repeating: 0, count: numClasses)
        
        var allDets: [UnionDet] = []
        allDets.reserveCapacity(512)
        
        for anchor in 0..<numAnchors {
            let x = detBuf[0 * stride + anchor]
            let y = detBuf[1 * stride + anchor]
            let w = detBuf[2 * stride + anchor]
            let h = detBuf[3 * stride + anchor]
            
            guard x.isFinite, y.isFinite, w.isFinite, h.isFinite, w > 0, h > 0 else { continue }
            
            let basePtr = detBuf.advanced(by: 4 * stride + anchor)
            cblas_scopy(Int32(numClasses), basePtr, Int32(stride), &tempScores, 1)
            
            var maxVal: Float = 0
            var maxIdx: vDSP_Length = 0
            vDSP_maxvi(tempScores, 1, &maxVal, &maxIdx, vDSP_Length(numClasses))
            
            let classIdx = Int(maxIdx)
            
            guard maxVal > confidenceThreshold, !clsToIgnore.contains(classIdx) else { continue }
            
            var coeffs = [Float](repeating: 0, count: 32)
            let coeffBase = detBuf.advanced(by: coeffOffset * stride + anchor)
            cblas_scopy(32, coeffBase, Int32(stride), &coeffs, 1)
            
            allDets.append(UnionDet(x: x, y: y, w: w, h: h, confidence: maxVal, classIdx: classIdx, coeffs: coeffs))
        }
        
        let t6End = Date()
        if debugMode {
            print("⏱️ STAGE 6 - Extract detections: \(String(format: "%.2f", t6End.timeIntervalSince(t6) * 1000)) ms")
            print("   raw detections: \(allDets.count)")
        }
        
        if allDets.isEmpty {
            if debugMode { print("⚠️ No detections found") }
            DispatchQueue.main.async {
                self.maskImageView.image = nil
                self.isProcessing = false
            }
            return
        }

        // STAGE 7: Apply NMS
        let t7 = Date()
        let afterNMS = applyNMS(allDets)
        let t7End = Date()
        if debugMode {
            print("⏱️ STAGE 7 - NMS: \(String(format: "%.2f", t7End.timeIntervalSince(t7) * 1000)) ms, kept: \(afterNMS.count)")
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
            if debugMode { print("   ⚠️ No detection with conf > 0.5") }
            DispatchQueue.main.async {
                self.maskImageView.image = nil
                self.isProcessing = false
            }
            return
        }
        
        let primary = afterNMS[primaryIdx]
        let t8End = Date()
        if debugMode {
            print("⏱️ STAGE 8 - Primary: \(String(format: "%.2f", t8End.timeIntervalSince(t8) * 1000)) ms")
            print("   🎯 PRIMARY[\(primaryIdx)]: \(className(primary.classIdx)) conf=\(String(format: "%.2f", primary.confidence)) size=\(Int(primary.w))x\(Int(primary.h))")
        }

        // STAGE 9: Parse prototypes
        let t9 = Date()
        setProgress(0.65, text: "Building mask…")
        
        guard let protoInfo = parsePrototypes(protoArray) else {
            if debugMode { print("❌ STAGE 9 FAILED: Parse prototypes") }
            isProcessing = false
            return
        }
        let planes = protoInfo.planes
        let pH = protoInfo.height
        let pW = protoInfo.width
        let planeSize = pH * pW
        
        let t9End = Date()
        if debugMode {
            print("⏱️ STAGE 9 - Prototypes: \(String(format: "%.2f", t9End.timeIntervalSince(t9) * 1000)) ms")
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
            print("⏱️ STAGE 10 - Reorganize: \(String(format: "%.2f", t10End.timeIntervalSince(t10) * 1000)) ms")
        }

        // STAGE 11: Filter - must touch primary bbox, drop if much larger
        let t11 = Date()
        
        let pLeft = primary.x - primary.w * 0.5
        let pRight = primary.x + primary.w * 0.5
        let pTop = primary.y - primary.h * 0.5
        let pBottom = primary.y + primary.h * 0.5
        
        if debugMode {
            print("   📦 PRIMARY: center=(\(Int(primary.x)),\(Int(primary.y))) size=\(Int(primary.w))x\(Int(primary.h))")
            print("      edges: L=\(Int(pLeft)) R=\(Int(pRight)) T=\(Int(pTop)) B=\(Int(pBottom))")
        }
        
        var kept2: [UnionDet] = [primary]
        for (i, d) in afterNMS.enumerated() {
            if i == primaryIdx { continue }
            
            let dLeft = d.x - d.w * 0.5
            let dRight = d.x + d.w * 0.5
            let dTop = d.y - d.h * 0.5
            let dBottom = d.y + d.h * 0.5
            
            let wPct = Int(d.w / primary.w * 100)
            let hPct = Int(d.h / primary.h * 100)
            
            let overlaps = dRight >= pLeft && dLeft <= pRight && dBottom >= pTop && dTop <= pBottom
            
            if !overlaps {
                if debugMode {
                    print("   ❌ [\(i)]: \(className(d.classIdx)) center=(\(Int(d.x)),\(Int(d.y))) size=\(Int(d.w))x\(Int(d.h)) [\(wPct)%,\(hPct)%] NO TOUCH")
                }
                continue
            }
            
            let tooLarge = d.w > primary.w * 1.5 && d.h > primary.h * 1.5
            
            if tooLarge {
                if debugMode {
                    print("   ❌ [\(i)]: \(className(d.classIdx)) center=(\(Int(d.x)),\(Int(d.y))) size=\(Int(d.w))x\(Int(d.h)) [\(wPct)%,\(hPct)%] TOO LARGE")
                }
            } else {
                kept2.append(d)
                if debugMode {
                    print("   ✅ [\(i)]: \(className(d.classIdx)) center=(\(Int(d.x)),\(Int(d.y))) size=\(Int(d.w))x\(Int(d.h)) [\(wPct)%,\(hPct)%]")
                }
            }
        }
        
        let t11End = Date()
        if debugMode {
            print("⏱️ STAGE 11 - Filter: \(String(format: "%.2f", t11End.timeIntervalSince(t11) * 1000)) ms, kept=\(kept2.count)")
        }
        
        if kept2.isEmpty {
            if debugMode { print("⚠️ No detections after filter") }
            DispatchQueue.main.async {
                self.maskImageView.image = nil
                self.isProcessing = false
            }
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
            print("⏱️ STAGE 12 - Union bbox: \(String(format: "%.2f", t12End.timeIntervalSince(t12) * 1000)) ms")
            print("   image: [\(bx1),\(by1)]→[\(bx2),\(by2)] = \(bx2-bx1)x\(by2-by1)")
        }

        // STAGE 13: Batched GEMM
        let t13 = Date()
        setProgress(0.75, text: "Computing mask…")
        
        var maxLogits = [Float](repeating: -Float.greatestFiniteMagnitude, count: planeSize)
        
        let batchSize = 64
        let M = Int32(planeSize)
        let K = Int32(32)
        let alpha: Float = 1
        let beta: Float = 0
        
        var bStart = 0
        while bStart < kept2.count {
            let bEnd = min(kept2.count, bStart + batchSize)
            let Bn = bEnd - bStart
            let N = Int32(Bn)
            
            var B = [Float](repeating: 0, count: 32 * Bn)
            for j in 0..<Bn {
                let coeffs = kept2[bStart + j].coeffs
                for i in 0..<32 { B[i * Bn + j] = coeffs[i] }
            }
            
            var C = [Float](repeating: 0, count: planeSize * Bn)
            
            A.withUnsafeBufferPointer { aPtr in
                B.withUnsafeBufferPointer { bPtr in
                    C.withUnsafeMutableBufferPointer { cPtr in
                        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                    M, N, K, alpha,
                                    aPtr.baseAddress!, K,
                                    bPtr.baseAddress!, N,
                                    beta, cPtr.baseAddress!, N)
                    }
                }
            }
            
            C.withUnsafeBufferPointer { cPtr in
                maxLogits.withUnsafeMutableBufferPointer { maxPtr in
                    for px in 0..<planeSize {
                        var localMax: Float = 0
                        vDSP_maxv(cPtr.baseAddress!.advanced(by: px * Bn), 1, &localMax, vDSP_Length(Bn))
                        if localMax > maxPtr[px] { maxPtr[px] = localMax }
                    }
                }
            }
            bStart = bEnd
        }
        
        let t13End = Date()
        if debugMode {
            print("⏱️ STAGE 13 - GEMM: \(String(format: "%.2f", t13End.timeIntervalSince(t13) * 1000)) ms")
        }

        // STAGE 14: Threshold
        let t14 = Date()
        
        var maskSmall = [UInt8](repeating: 0, count: planeSize)
        var positiveCount = 0
        for i in 0..<planeSize {
            if maxLogits[i] > 0.0 {
                maskSmall[i] = 255
                positiveCount += 1
            }
        }
        
        let t14End = Date()
        if debugMode {
            print("⏱️ STAGE 14 - Threshold: \(String(format: "%.2f", t14End.timeIntervalSince(t14) * 1000)) ms, positive: \(positiveCount)")
        }

        // STAGE 15: Upscale
        let t15 = Date()
        setProgress(0.85, text: "Upscaling…")
        
        var maskFull = upscaleMask(maskSmall: maskSmall, pW: pW, pH: pH,
                                    modelInput: 1280, origW: origW, origH: origH,
                                    resizeGain: resizeGain, padX: padX, padY: padY)
        
        let t15End = Date()
        if debugMode {
            print("⏱️ STAGE 15 - Upscale: \(String(format: "%.2f", t15End.timeIntervalSince(t15) * 1000)) ms")
        }

        // STAGE 15b: Morph close
        let t15b = Date()
        let fullSize = origW * origH
        
        var srcBuffer = vImage_Buffer(data: &maskFull, height: vImagePixelCount(origH), width: vImagePixelCount(origW), rowBytes: origW)
        var dilated = [UInt8](repeating: 0, count: fullSize)
        var dilatedBuffer = vImage_Buffer(data: &dilated, height: vImagePixelCount(origH), width: vImagePixelCount(origW), rowBytes: origW)
        var closed = [UInt8](repeating: 0, count: fullSize)
        var closedBuffer = vImage_Buffer(data: &closed, height: vImagePixelCount(origH), width: vImagePixelCount(origW), rowBytes: origW)
        
        var kernel: [UInt8] = [1,1,1, 1,1,1, 1,1,1]
        kernel.withUnsafeBufferPointer { kernelPtr in
            vImageDilate_Planar8(&srcBuffer, &dilatedBuffer, 0, 0, kernelPtr.baseAddress!, 3, 3, vImage_Flags(kvImageNoFlags))
            vImageErode_Planar8(&dilatedBuffer, &closedBuffer, 0, 0, kernelPtr.baseAddress!, 3, 3, vImage_Flags(kvImageNoFlags))
        }
        maskFull = closed
        
        let t15bEnd = Date()
        if debugMode {
            print("⏱️ STAGE 15b - Morph: \(String(format: "%.2f", t15bEnd.timeIntervalSince(t15b) * 1000)) ms")
        }

        // STAGE 16: Composite
        let t16 = Date()
        setProgress(0.92, text: "Compositing…")
        
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(data: nil, width: origW, height: origH,
                                   bitsPerComponent: 8, bytesPerRow: origW * 4,
                                   space: colorSpace,
                                   bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue),
              let outBase = ctx.data?.assumingMemoryBound(to: UInt8.self) else {
            isProcessing = false
            return
        }
        
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        
        guard let origBase = CVPixelBufferGetBaseAddress(pixelBuffer)?.assumingMemoryBound(to: UInt8.self) else {
            isProcessing = false
            return
        }
        let origBytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        
        var totalSet = 0
        for y in 0..<origH {
            let origRow = y * origBytesPerRow
            let outRow = y * origW * 4
            let mRow = y * origW
            
            for x in 0..<origW {
                let outPx = outRow + x * 4
                
                if x < bx1 || x >= bx2 || y < by1 || y >= by2 {
                    outBase[outPx + 3] = 0
                    continue
                }
                
                let m = maskFull[mRow + x]
                if m > 0 {
                    let origPx = origRow + x * 4
                    outBase[outPx + 0] = origBase[origPx + 0]
                    outBase[outPx + 1] = origBase[origPx + 1]
                    outBase[outPx + 2] = origBase[origPx + 2]
                    outBase[outPx + 3] = 255
                    totalSet += 1
                } else {
                    outBase[outPx + 3] = 0
                }
            }
        }
        
        let t16End = Date()
        if debugMode {
            print("⏱️ STAGE 16 - Composite: \(String(format: "%.2f", t16End.timeIntervalSince(t16) * 1000)) ms, opaque: \(totalSet)")
        }

        // STAGE 17: Finalize
        let t17 = Date()
        
        ctx.setStrokeColor(UIColor.cyan.cgColor)
        ctx.setLineWidth(2.0)
        for d in kept2 {
            let dx1 = Int(round((d.x - d.w * 0.5 - padX) / resizeGain))
            let dy1 = Int(round((d.y - d.h * 0.5 - padY) / resizeGain))
            let dx2 = Int(round((d.x + d.w * 0.5 - padX) / resizeGain))
            let dy2 = Int(round((d.y + d.h * 0.5 - padY) / resizeGain))
            ctx.stroke(CGRect(x: max(0, dx1), y: max(0, dy1),
                              width: min(origW - max(0, dx1), dx2 - dx1),
                              height: min(origH - max(0, dy1), dy2 - dy1)))
        }
        
        ctx.setStrokeColor(UIColor.green.cgColor)
        ctx.setLineWidth(6.0)
        ctx.stroke(CGRect(x: bx1, y: by1, width: bx2 - bx1, height: by2 - by1))
        
        if let out = ctx.makeImage() {
            DispatchQueue.main.async {
                self.maskImageView.image = UIImage(cgImage: out)
                self.isProcessing = false
            }
        } else {
            DispatchQueue.main.async { self.isProcessing = false }
        }
        
        let t17End = Date()
        let frameEnd = Date()
        
        if debugMode {
            print("⏱️ STAGE 17 - Finalize: \(String(format: "%.2f", t17End.timeIntervalSince(t17) * 1000)) ms")
            print("⏱️ FRAME TOTAL: \(String(format: "%.2f", frameEnd.timeIntervalSince(frameStart) * 1000)) ms")
            print("⏱️ ═══════════════════════════════════════════\n")
        }
        
        if totalSet > 0 { finishFirstDetectionIfNeeded() }
    }

    // MARK: - NMS
    private func applyNMS(_ dets: [UnionDet]) -> [UnionDet] {
        guard !dets.isEmpty else { return [] }
        let sorted = dets.sorted { $0.confidence > $1.confidence }
        var kept: [UnionDet] = []
        
        for d in sorted {
            var dominated = false
            for k in kept {
                if iou(d, k) > iouThreshold {
                    dominated = true
                    break
                }
            }
            if !dominated { kept.append(d) }
        }
        return kept
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
            var dstBuf = vImage_Buffer(data: &rawFloats, height: 1, width: vImagePixelCount(total), rowBytes: total * 4)
            vImageConvert_Planar16FtoPlanarF(&srcBuf, &dstBuf, vImage_Flags(kvImageNoFlags))
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
        return true
    }
}
