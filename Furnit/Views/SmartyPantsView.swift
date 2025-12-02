// SmartyPantsView.swift
// Two-Stage Detection: Full frame -> Crop to primary bbox -> Re-detect -> UNION BOTH
// With timing logs at crucial stages

import SwiftUI
import UIKit
import CoreML
import Accelerate
import AVFoundation
import Photos

// MARK: - SwiftUI Wrapper
struct SmartyPantsViewSwiftUI: UIViewRepresentable {
    let mlModel: MLModel?
    var processInterval: TimeInterval = 0.05
    var confidenceThreshold: Float = 0.3
    var detectAllObjects: Bool = false
    var useBilinearUpscaling: Bool = true
    var maskThreshold: Float = 0.0
    var debugMode: Bool = true
    var active: Bool = false

    func makeUIView(context: Context) -> SmartyPantsContainerView {
        let v = SmartyPantsContainerView()
        v.processInterval = processInterval
        v.confidenceThreshold = confidenceThreshold
        v.detectAllObjects = detectAllObjects
        v.useBilinearUpscaling = useBilinearUpscaling
        v.maskThreshold = maskThreshold
        v.debugMode = debugMode
        v.setModel(mlModel)
        if active { v.startIfNeeded() }
        return v
    }

    func updateUIView(_ uiView: SmartyPantsContainerView, context: Context) {
        uiView.setModel(mlModel)
        uiView.processInterval = processInterval
        uiView.confidenceThreshold = confidenceThreshold
        uiView.detectAllObjects = detectAllObjects
        uiView.useBilinearUpscaling = useBilinearUpscaling
        uiView.maskThreshold = maskThreshold
        uiView.debugMode = debugMode
        if active { uiView.startIfNeeded() } else { uiView.stop() }
    }

    static func dismantleUIView(_ uiView: SmartyPantsContainerView, coordinator: ()) {
        uiView.stop()
    }
}

// MARK: - Detection Struct
struct DetectionSmarty {
    let x: Float
    let y: Float
    let width: Float
    let height: Float
    let confidence: Float
    let classIdx: Int
    let className: String
    let maskCoeffs: [Float]
}

// MARK: - Main Container View
final class SmartyPantsContainerView: UIView, AVCaptureVideoDataOutputSampleBufferDelegate, UIGestureRecognizerDelegate {
    
    // MARK: Config
    var processInterval: TimeInterval = 0.05
    var confidenceThreshold: Float = 0.3
    var debugMode: Bool = true  // Enable debug prints and image saves
    
    // Detection mode: true = detect ALL objects, false = furniture classes only
    var detectAllObjects: Bool = false
    
    // Mask upscaling: true = bilinear (smooth edges), false = nearest-neighbor (faster)
    var useBilinearUpscaling: Bool = true
    
    // Mask threshold: values above this are considered "object"
    var maskThreshold: Float = 0.0
    
    private let bboxFont: CTFont = CTFontCreateWithName("Helvetica-Bold" as CFString, 28, nil)
    private lazy var bboxAttributes: [NSAttributedString.Key: Any] = [
        .font: bboxFont,
        .foregroundColor: UIColor.white
    ]

    // MARK: Camera
    private let captureSession = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let sampleQueue = DispatchQueue(label: "com.furnit.smarty.sample", qos: .userInitiated)

    // MARK: UI
    private let previewLayer = AVCaptureVideoPreviewLayer()
    private let maskImageView: UIImageView = {
        let iv = UIImageView()
        iv.contentMode = .scaleAspectFit
        iv.backgroundColor = .clear
        iv.isOpaque = false
        iv.clipsToBounds = true
        iv.alpha = 1.0
        iv.isUserInteractionEnabled = false
        return iv
    }()
    
    // MARK: Gesture state
    private var currentScale: CGFloat = 1.0

    // MARK: Model & Queues
    private var mlModel: MLModel?
    private let detectionQueue = DispatchQueue(label: "com.furnit.smarty.detection", qos: .userInitiated)
    private var lastProcessTime = Date.distantPast
    private var isProcessing = false
    private let ciContext = CIContext(options: [.workingColorSpace: CGColorSpaceCreateDeviceRGB()])

    // MARK: Furniture & Household Classes (LVIS indices)
    private let furnitureClasses: [Int: String] = [
        // Seating
        132: "armchair", 276: "bar stool", 352: "beach chair", 364: "bean bag chair",
        402: "bench", 821: "chair", 1060: "computer chair", 1602: "feeding chair",
        1721: "folding chair", 2499: "loveseat", 2754: "music stool", 2834: "office chair",
        2939: "park bench", 3024: "church bench", 3423: "rocking chair", 3584: "seat",
        3888: "step stool", 3909: "stool", 4041: "swivel chair", 4473: "wheelchair",
        4506: "window seat",
        
        // Beds & Bedding
        375: "bed", 376: "bedcover", 377: "bed frame", 378: "bedsheet", 379: "bed sheet",
        632: "bunk bed", 714: "canopy bed", 823: "daybed", 1137: "infant bed",
        1270: "day bed", 1364: "dog bed", 2141: "hospital bed", 2599: "mattress",
        3049: "pillow", 455: "blanket", 1047: "comforter", 1425: "duvet",
        3625: "sheet", 3626: "sheets", 431: "bedspread", 2450: "linen",
        
        // Sofas & Couches
        1141: "couch", 1816: "futon", 4331: "vanity", 2936: "ottoman", 3728: "sofa",
        
        // Tables
        429: "billiard table", 1006: "cocktail table", 1061: "computer desk", 1301: "table",
        1325: "dining table", 1503: "side table", 1885: "glass table", 2247: "island",
        2319: "kitchen counter", 2322: "kitchen island",
        2324: "kitchen table", 2802: "nightstand", 2836: "office desk", 3045: "picnic table",
        3061: "table tennis table", 3145: "poker table", 3449: "round table",
        4055: "table top", 4545: "workbench", 4564: "writing desk", 1007: "coffee table",
        
        // Storage
        332: "bathroom cabinet", 517: "bookshelf", 567: "chest", 636: "bureau",
        670: "cabinet", 977: "closet", 996: "coatrack", 1396: "drawer", 1405: "dresser",
        1624: "file cabinet", 2318: "kitchen cabinet", 2614: "medicine cabinet",
        3621: "shelf", 3678: "side cabinet", 3812: "spice rack", 4004: "supermarket shelf",
        4294: "tv cabinet", 4513: "wine cabinet", 4516: "wine rack", 4433: "wardrobe",
        
        // Lighting
        382: "bedside lamp", 1302: "table lamp", 1619: "floor lamp", 2383: "lamp",
        2384: "lampshade", 732: "candle", 898: "chandelier",
        2449: "light bulb", 2451: "light fixture", 4210: "torch", 3862: "stand",
        
        // Mirrors & Decor
        334: "bathroom mirror", 2654: "mirror", 1214: "curtain", 3485: "rug",
        3046: "picture frame", 4056: "tablecloth", 4358: "vase", 3081: "plant",
        1750: "footrest", 749: "carpet", 1402: "drape", 1403: "drapery",
        
        // Electronics
        4161: "television", 4162: "tv", 1058: "computer monitor", 1059: "computer",
        3365: "remote control", 3802: "speaker",
        
        // Bathroom
        4179: "toilet seat", 4178: "toilet", 4213: "towel bar", 4212: "towel",
        386: "bathtub", 3635: "shower", 3636: "shower curtain", 387: "bath mat",
        
        // Kitchen
        3357: "refrigerator", 2914: "oven", 2637: "microwave", 3675: "sink",
        1350: "dishwasher", 3915: "stovetop", 1780: "freezer",
        
        // Misc
        213: "baby seat", 733: "car seat", 834: "changing table", 679: "cake stand",
        1143: "counter", 1144: "counter top", 1303: "desktop", 1733: "food stand",
        1801: "fruit stand", 2193: "ice shelf", 2219: "information desk",
        1099: "cot", 1183: "cradle", 3088: "playpen"
    ]

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
            maskImageView.centerXAnchor.constraint(equalTo: centerXAnchor),
            maskImageView.centerYAnchor.constraint(equalTo: centerYAnchor),
            maskImageView.widthAnchor.constraint(equalTo: widthAnchor),
            maskImageView.heightAnchor.constraint(equalTo: heightAnchor)
        ])
        
        // Pinch (zoom)
        let pinchGesture = UIPinchGestureRecognizer(target: self, action: #selector(handlePinch(_:)))
        pinchGesture.delegate = self
        self.addGestureRecognizer(pinchGesture)
        
        // Pan (drag)
        let panGesture = UIPanGestureRecognizer(target: self, action: #selector(handlePan(_:)))
        panGesture.delegate = self
        maskImageView.addGestureRecognizer(panGesture)
        
        setupCamera()
        if self.debugMode { print("✅ SmartyPantsContainerView initialized") }
    }

    
    override func layoutSubviews() {
        super.layoutSubviews()
        previewLayer.frame = bounds
    }
    
    override func point(inside point: CGPoint, with event: UIEvent?) -> Bool {
        if point.y < 100 { return false }
        return true
    }
    
    @objc private func handlePinch(_ gesture: UIPinchGestureRecognizer) {
        guard maskImageView.image != nil else { return }
        
        switch gesture.state {
        case .began:
            break
        case .changed:
            let newScale = currentScale * gesture.scale
            let clampedScale = min(max(newScale, 0.3), 3.0)
            maskImageView.transform = CGAffineTransform(scaleX: clampedScale, y: clampedScale)
            currentScale = clampedScale
            gesture.scale = 1.0
        case .ended, .cancelled:
            if currentScale > 0.9 && currentScale < 1.1 {
                currentScale = 1.0
                UIView.animate(withDuration: 0.2) {
                    self.maskImageView.transform = .identity
                }
            }
        default:
            break
        }
    }
    
    @objc private func handlePan(_ gesture: UIPanGestureRecognizer) {
        guard maskImageView.image != nil else { return }
        let translation = gesture.translation(in: self)
        switch gesture.state {
        case .began, .changed:
            maskImageView.center = CGPoint(x: maskImageView.center.x + translation.x,
                                           y: maskImageView.center.y + translation.y)
            gesture.setTranslation(.zero, in: self)
        case .ended, .cancelled:
            let halfW = maskImageView.bounds.width / 2
            let halfH = maskImageView.bounds.height / 2
            var newCenter = maskImageView.center
            newCenter.x = min(max(newCenter.x, halfW - 1000), bounds.width - halfW + 1000)
            newCenter.y = min(max(newCenter.y, halfH - 1000), bounds.height - halfH + 1000)
            UIView.animate(withDuration: 0.15) { self.maskImageView.center = newCenter }
        default:
            break
        }
    }

    func gestureRecognizer(_ gestureRecognizer: UIGestureRecognizer, shouldRecognizeSimultaneouslyWith otherGestureRecognizer: UIGestureRecognizer) -> Bool {
        return true
    }

    
    // MARK: - UIGestureRecognizerDelegate
    func gestureRecognizer(_ gestureRecognizer: UIGestureRecognizer, shouldReceive touch: UITouch) -> Bool {
        let location = touch.location(in: self)
        if location.y < 100 { return false }
        return true
    }
    
    // MARK: - Public
    func setModel(_ model: MLModel?) {
        detectionQueue.sync {
            self.mlModel = model
        }
    }
    
    func startIfNeeded() {
        requestCameraPermissionAndStart()
    }
    
    func stop() {
        stopCamera()
    }

    // MARK: - Camera Setup
    private func setupCamera() {
        captureSession.beginConfiguration()
        captureSession.sessionPreset = .hd1280x720
        captureSession.inputs.forEach { captureSession.removeInput($0) }
        captureSession.outputs.forEach { captureSession.removeOutput($0) }

        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else {
            captureSession.commitConfiguration()
            return
        }

        do {
            let input = try AVCaptureDeviceInput(device: device)
            if captureSession.canAddInput(input) {
                captureSession.addInput(input)
            }
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
            DispatchQueue.global(qos: .userInitiated).async {
                self.captureSession.startRunning()
            }
        } catch {
            captureSession.commitConfiguration()
        }
    }

    private func stopCamera() {
        DispatchQueue.global(qos: .userInitiated).async {
            if self.captureSession.isRunning {
                self.captureSession.stopRunning()
            }
        }
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
        default:
            break
        }
    }

    // MARK: - Capture Delegate
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        detectionQueue.async { [weak self] in self?.processFrame(pixelBuffer) }
    }

    // MARK: - Crop Pixel Buffer to BBox (vImage copy)
    private func cropPixelBuffer(_ pixelBuffer: CVPixelBuffer, toBBox det: DetectionSmarty, padding: Float = 0.1) -> CVPixelBuffer? {
        let cropStart = Date()
        
        let fullWf = Float(CVPixelBufferGetWidth(pixelBuffer))
        let fullHf = Float(CVPixelBufferGetHeight(pixelBuffer))
        
        let scaleX = fullWf / 640.0
        let scaleY = fullHf / 640.0
        
        let centerX = det.x * scaleX
        let centerY = det.y * scaleY
        let boxW = det.width * scaleX
        let boxH = det.height * scaleY
        
        let padW = boxW * padding
        let padH = boxH * padding
        
        var x1 = centerX - boxW / 2 - padW
        var y1 = centerY - boxH / 2 - padH
        var x2 = centerX + boxW / 2 + padW
        var y2 = centerY + boxH / 2 + padH
        
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(fullWf, x2)
        y2 = min(fullHf, y2)
        
        let cropW = Int(x2 - x1)
        let cropH = Int(y2 - y1)
        
        guard cropW > 10 && cropH > 10 else { return nil }
        
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        
        guard let srcBase = CVPixelBufferGetBaseAddress(pixelBuffer) else { return nil }
        let srcBytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        
        var out: CVPixelBuffer?
        let status = CVPixelBufferCreate(kCFAllocatorDefault, cropW, cropH, kCVPixelFormatType_32BGRA, nil, &out)
        guard status == kCVReturnSuccess, let dst = out else { return nil }
        
        CVPixelBufferLockBaseAddress(dst, [])
        defer { CVPixelBufferUnlockBaseAddress(dst, []) }
        guard let dstBase = CVPixelBufferGetBaseAddress(dst) else { return nil }
        let dstBytesPerRow = CVPixelBufferGetBytesPerRow(dst)
        
        let x1Int = Int(x1)
        let y1Int = Int(y1)
        let srcOffsetPtr = srcBase.advanced(by: y1Int * srcBytesPerRow + x1Int * 4)
        
        var srcBuf = vImage_Buffer(
            data: srcOffsetPtr,
            height: vImagePixelCount(cropH),
            width: vImagePixelCount(cropW),
            rowBytes: srcBytesPerRow
        )
        var dstBuf = vImage_Buffer(
            data: dstBase,
            height: vImagePixelCount(cropH),
            width: vImagePixelCount(cropW),
            rowBytes: dstBytesPerRow
        )
        
        let copyErr = vImageCopyBuffer(&srcBuf, &dstBuf, 4, vImage_Flags(kvImageNoFlags))
        if copyErr != kvImageNoError {
            let scaleErr = vImageScale_ARGB8888(&srcBuf, &dstBuf, nil, vImage_Flags(kvImageNoFlags))
            if scaleErr != kvImageNoError {
                let srcPtr = srcBase.assumingMemoryBound(to: UInt8.self)
                let dstPtr = dstBase.assumingMemoryBound(to: UInt8.self)
                for row in 0..<cropH {
                    let s = (y1Int + row) * srcBytesPerRow + x1Int * 4
                    let d = row * dstBytesPerRow
                    memcpy(dstPtr + d, srcPtr + s, cropW * 4)
                }
            }
        }
        
        if self.debugMode {
            let dt = Date().timeIntervalSince(cropStart) * 1000.0
            print(String(format: "⏱ cropPixelBuffer: %.2f ms (rect %dx%d)", dt, cropW, cropH))
        }
        
        return dst
    }
    
    private func processFrame(_ pixelBuffer: CVPixelBuffer) {
        let frameStart = Date()
        
        guard let model = mlModel else { return }
        let now = Date()
        guard now.timeIntervalSince(lastProcessTime) >= processInterval, !isProcessing else { return }
        lastProcessTime = now
        isProcessing = true

        if self.debugMode {
            print("\n🕒 ===== NEW FRAME @ \(now.timeIntervalSince1970) =====")
            print("🔬 ========== STAGE 1: FULL FRAME ==========")
        }

        // STAGE 1: Preprocess
        let stage1PreStart = Date()
        guard let resized = letterbox(pixelBuffer, size: 640) else {
            isProcessing = false
            return
        }
        guard let inputArray = pixelBufferToMLMultiArray(resized) else {
            isProcessing = false
            return
        }
        let stage1PreEnd = Date()
        if self.debugMode {
            print(String(format: "⏱ Stage1 preprocess (letterbox+toMultiArray): %.2f ms", stage1PreEnd.timeIntervalSince(stage1PreStart) * 1000.0))
        }

        // STAGE 1: Inference
        let stage1InfStart = Date()
        guard let inputProvider = try? MLDictionaryFeatureProvider(dictionary: ["image": inputArray]) else {
            isProcessing = false
            return
        }
        guard let output = try? model.prediction(from: inputProvider) else {
            isProcessing = false
            return
        }
        let stage1InfEnd = Date()
        if self.debugMode {
            print(String(format: "⏱ Stage1 model.prediction: %.2f ms", stage1InfEnd.timeIntervalSince(stage1InfStart) * 1000.0))
        }

        if self.debugMode {
            let names = output.featureNames.joined(separator: ", ")
            print("📤 Model outputs: \(names)")
        }

        var detectionsArray: MLMultiArray?
        if let arr = output.featureValue(for: "var_1432")?.multiArrayValue {
            detectionsArray = arr
        } else if let arr = output.featureValue(for: "var_2421")?.multiArrayValue {
            detectionsArray = arr
        } else {
            for name in output.featureNames {
                if let arr = output.featureValue(for: name)?.multiArrayValue {
                    let shape = arr.shape.map { $0.intValue }
                    if shape.count == 3 && shape[0] == 1 && shape[1] > 100 {
                        detectionsArray = arr
                        if self.debugMode { print("   → Using '\(name)' as detections: \(shape)") }
                        break
                    }
                }
            }
        }

        guard let detArray = detectionsArray else {
            isProcessing = false
            return
        }

        guard let prototypesArray = output.featureValue(for: "p")?.multiArrayValue else {
            isProcessing = false
            return
        }

        let decodeStart = Date()
        let stage1DetectionsFull = extractDetections(from: detArray)
        let decodeEnd = Date()
        if self.debugMode {
            print("📊 Stage 1: \(stage1DetectionsFull.count) detections")
            print(String(format: "⏱ Stage1 detection decode: %.2f ms", decodeEnd.timeIntervalSince(decodeStart) * 1000.0))
        }

        let sorted = stage1DetectionsFull.sorted { $0.confidence > $1.confidence }
        
        guard let primary = sorted.first else {
            DispatchQueue.main.async {
                self.maskImageView.image = nil
                self.isProcessing = false
            }
            return
        }

        if self.debugMode {
            print("🎯 Primary: \(primary.className) @ \(Int(primary.confidence * 100))%")
            print("   BBox: center(\(Int(primary.x)), \(Int(primary.y))) size(\(Int(primary.width))x\(Int(primary.height)))")
        }

        // STAGE 2
        if self.debugMode { print("\n🔬 ========== STAGE 2: CROPPED ==========") }

        var stage2Detections: [DetectionSmarty] = []
        var stage2Prototypes: MLMultiArray? = nil

        let stage2Start = Date()
        if let croppedBuffer = cropPixelBuffer(pixelBuffer, toBBox: primary, padding: 0.1),
           let resizedCrop = letterbox(croppedBuffer, size: 640),
           let cropInputArray = pixelBufferToMLMultiArray(resizedCrop),
           let cropInputProvider = try? MLDictionaryFeatureProvider(dictionary: ["image": cropInputArray]) {

            let stage2InfStart = Date()
            if let cropOutput = try? model.prediction(from: cropInputProvider) {
                let stage2InfEnd = Date()
                if self.debugMode {
                    print(String(format: "⏱ Stage2 model.prediction: %.2f ms", stage2InfEnd.timeIntervalSince(stage2InfStart) * 1000.0))
                }
                
                var cropDetArray: MLMultiArray?
                if let arr = cropOutput.featureValue(for: "var_2421")?.multiArrayValue {
                    cropDetArray = arr
                } else {
                    for name in cropOutput.featureNames {
                        if let arr = cropOutput.featureValue(for: name)?.multiArrayValue {
                            let shape = arr.shape.map { $0.intValue }
                            if shape.count == 3 && shape[0] == 1 && shape[1] > 100 {
                                cropDetArray = arr
                                break
                            }
                        }
                    }
                }

                if let detArray = cropDetArray,
                   let protoArray = cropOutput.featureValue(for: "p")?.multiArrayValue {
                    let s2DecodeStart = Date()
                    stage2Detections = extractDetections(from: detArray)
                    let s2DecodeEnd = Date()
                    stage2Prototypes = protoArray
                    if self.debugMode {
                        print("📊 Stage 2: \(stage2Detections.count) detections")
                        print(String(format: "⏱ Stage2 detection decode: %.2f ms", s2DecodeEnd.timeIntervalSince(s2DecodeStart) * 1000.0))
                    }
                }
            }
        } else {
            if self.debugMode { print("⚠️ Stage 2: Failed to crop/process") }
        }
        let stage2End = Date()
        if self.debugMode {
            print(String(format: "⏱ Stage2 total (crop+preprocess+infer+decode): %.2f ms",
                         stage2End.timeIntervalSince(stage2Start) * 1000.0))
        }
        
        let rawDetections = extractDetections(from: detArray)
        let nmsStart = Date()
        let uniqueDetections = applyNMS(rawDetections, iouThreshold: 0.7)
        let stage1Kept = keepOverlappingDetections(uniqueDetections)
        let stage2Kept = stage2Prototypes != nil
            ? keepOverlappingDetections(applyNMS(stage2Detections, iouThreshold: 0.7))
            : []
        let nmsEnd = Date()
        if self.debugMode {
            print(String(format: "⏱ NMS + keepOverlapping: %.2f ms", nmsEnd.timeIntervalSince(nmsStart) * 1000.0))
        }

        if self.debugMode {
            print("\n📊 UNION SUMMARY:")
            print("   Stage 1: keeping \(stage1Kept.count) overlapping detections")
            print("   Stage 2: keeping \(stage2Kept.count) overlapping detections")
        }

        if stage1Kept.isEmpty && stage2Kept.isEmpty {
            DispatchQueue.main.async {
                self.maskImageView.image = nil
                self.isProcessing = false
            }
            return
        }

        let cutoutStart = Date()
        generateCutoutTwoStage(
            stage1Detections: stage1Kept,
            stage1Prototypes: prototypesArray,
            stage2Detections: stage2Kept,
            stage2Prototypes: stage2Prototypes,
            primaryBBox: primary,
            originalImage: pixelBuffer
        )
        let cutoutEnd = Date()
        if self.debugMode {
            print(String(format: "⏱ generateCutoutTwoStage call: %.2f ms", cutoutEnd.timeIntervalSince(cutoutStart) * 1000.0))
            print(String(format: "🕒 Frame total (processFrame): %.2f ms", cutoutEnd.timeIntervalSince(frameStart) * 1000.0))
        }
    }

    private func applyNMS(_ detections: [DetectionSmarty], iouThreshold: Float) -> [DetectionSmarty] {
        let sorted = detections.sorted { $0.confidence > $1.confidence }
        var kept: [DetectionSmarty] = []
        kept.reserveCapacity(sorted.count)

        for det in sorted {
            var dominated = false
            for k in kept {
                if bboxIoU(det, k) > iouThreshold {
                    dominated = true
                    break
                }
            }
            if !dominated { kept.append(det) }
        }
        return kept
    }

    private func bboxIoU(_ a: DetectionSmarty, _ b: DetectionSmarty) -> Float {
        let aLeft = a.x - a.width * 0.5
        let aRight = a.x + a.width * 0.5
        let aTop = a.y - a.height * 0.5
        let aBottom = a.y + a.height * 0.5

        let bLeft = b.x - b.width * 0.5
        let bRight = b.x + b.width * 0.5
        let bTop = b.y - b.height * 0.5
        let bBottom = b.y + b.height * 0.5

        let ix1 = max(aLeft, bLeft), ix2 = min(aRight, bRight)
        let iy1 = max(aTop, bTop), iy2 = min(aBottom, bBottom)
        let iw = max(0, ix2 - ix1), ih = max(0, iy2 - iy1)
        let inter = iw * ih
        let union = a.width * a.height + b.width * b.height - inter
        return union > 0 ? inter / union : 0
    }

    
    private func letterbox(_ src: CVPixelBuffer, size: Int = 640) -> CVPixelBuffer? {
        let t0 = Date()
        
        CVPixelBufferLockBaseAddress(src, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(src, .readOnly) }

        let srcW = CVPixelBufferGetWidth(src)
        let srcH = CVPixelBufferGetHeight(src)

        var dstOpt: CVPixelBuffer?
        let status = CVPixelBufferCreate(nil, size, size, kCVPixelFormatType_32BGRA, nil, &dstOpt)
        guard status == kCVReturnSuccess, let dst = dstOpt else { return nil }

        CVPixelBufferLockBaseAddress(dst, [])
        defer { CVPixelBufferUnlockBaseAddress(dst, []) }

        guard let srcBase = CVPixelBufferGetBaseAddress(src),
              let dstBase = CVPixelBufferGetBaseAddress(dst) else { return nil }

        var srcBuffer = vImage_Buffer(data: srcBase,
                                      height: vImagePixelCount(srcH),
                                      width: vImagePixelCount(srcW),
                                      rowBytes: CVPixelBufferGetBytesPerRow(src))
        var dstBuffer = vImage_Buffer(data: dstBase,
                                      height: vImagePixelCount(size),
                                      width: vImagePixelCount(size),
                                      rowBytes: CVPixelBufferGetBytesPerRow(dst))

        let err = vImageScale_ARGB8888(&srcBuffer, &dstBuffer, nil, vImage_Flags(0))
        guard err == kvImageNoError else { return nil }

        if self.debugMode {
            let dt = Date().timeIntervalSince(t0) * 1000.0
            print(String(format: "⏱ letterbox %dx%d → %dx%d: %.2f ms",
                         srcW, srcH, size, size, dt))
        }

        return dst
    }

    private func keepOverlappingDetections(_ detections: [DetectionSmarty]) -> [DetectionSmarty] {
        guard detections.count > 0 else { return [] }
        if detections.count == 1 { return detections }

        let sorted = detections.sorted { $0.confidence > $1.confidence }
        let primary = sorted[0]
        let pLeft = primary.x - primary.width / 2
        let pRight = primary.x + primary.width / 2
        let pTop = primary.y - primary.height / 2
        let pBottom = primary.y + primary.height / 2

        var kept: [DetectionSmarty] = []
        kept.reserveCapacity(sorted.count)

        for det in sorted {
            let aLeft = det.x - det.width / 2
            let aRight = det.x + det.width / 2
            let aTop = det.y - det.height / 2
            let aBottom = det.y + det.height / 2

            if aRight < pLeft || pRight < aLeft { continue }
            if aBottom < pTop || pBottom < aTop { continue }
            kept.append(det)
        }
        return kept
    }
    
    // MARK: - Print 20x20 Binary Grid
    private func print20x20BinaryGrid(_ title: String, mask: [UInt8], width: Int, height: Int) {
        guard self.debugMode else { return }
        
        print("\n🔢 [\(title)] (20x20 binary, * = object, . = background):")
        for gy in 0..<20 {
            var rowSymbols = ""
            for gx in 0..<20 {
                let y = gy * 8 + 7
                let x = gx * 8 + 7
                if y < height && x < width {
                    let idx = y * width + x
                    rowSymbols += mask[idx] > 0 ? "*" : "."
                } else {
                    rowSymbols += " "
                }
            }
            print("   \(rowSymbols)")
        }
    }
    
    // MARK: - Save Mask to File
    private func saveMaskToFile(rawMask: [Float], width: Int, height: Int, detection: DetectionSmarty) {
        let timestamp = Int(Date().timeIntervalSince1970)
        let colorSpace = CGColorSpaceCreateDeviceGray()
        
        var minVal: Float = 0, maxVal: Float = 0
        vDSP_minv(rawMask, 1, &minVal, vDSP_Length(rawMask.count))
        vDSP_maxv(rawMask, 1, &maxVal, vDSP_Length(rawMask.count))
        let range = maxVal - minVal
        
        var grayPixels = [UInt8](repeating: 0, count: width * height)
        for i in 0..<rawMask.count {
            let normalized = range > 0 ? (rawMask[i] - minVal) / range : 0.5
            grayPixels[i] = UInt8(max(0, min(255, normalized * 255)))
        }
        
        if let provider = CGDataProvider(data: Data(grayPixels) as CFData),
           let cgImage = CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 8,
                                  bytesPerRow: width, space: colorSpace,
                                  bitmapInfo: CGBitmapInfo(rawValue: 0),
                                  provider: provider, decode: nil, shouldInterpolate: false,
                                  intent: .defaultIntent) {
            let grayImage = UIImage(cgImage: cgImage)
            PHPhotoLibrary.shared().performChanges({
                PHAssetChangeRequest.creationRequestForAsset(from: grayImage)
            }) { success, error in
                if success {
                    print("💾 Saved GRAYSCALE mask to Photos @ \(timestamp)")
                } else {
                    print("❌ Failed to save grayscale: \(error?.localizedDescription ?? "unknown")")
                }
            }
        }
        
        let scale = Float(width) / 640.0
        let mx1 = max(0, Int((detection.x - detection.width / 2) * scale))
        let my1 = max(0, Int((detection.y - detection.height / 2) * scale))
        let mx2 = min(width, Int((detection.x + detection.width / 2) * scale))
        let my2 = min(height, Int((detection.y + detection.height / 2) * scale))
        
        var binaryPixels = [UInt8](repeating: 0, count: width * height)
        for y in 0..<height {
            for x in 0..<width {
                let idx = y * width + x
                if x >= mx1 && x < mx2 && y >= my1 && y < my2 && rawMask[idx] > maskThreshold {
                    binaryPixels[idx] = 255
                }
            }
        }
        
        if let provider = CGDataProvider(data: Data(binaryPixels) as CFData),
           let cgImage = CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 8,
                                  bytesPerRow: width, space: colorSpace,
                                  bitmapInfo: CGBitmapInfo(rawValue: 0),
                                  provider: provider, decode: nil, shouldInterpolate: false,
                                  intent: .defaultIntent) {
            let binaryImage = UIImage(cgImage: cgImage)
            PHPhotoLibrary.shared().performChanges({
                PHAssetChangeRequest.creationRequestForAsset(from: binaryImage)
            }) { success, error in
                if success {
                    print("💾 Saved BINARY mask to Photos (threshold: \(self.maskThreshold)) @ \(timestamp)")
                } else {
                    print("❌ Failed to save binary: \(error?.localizedDescription ?? "unknown")")
                }
            }
        }
    }
    
    func clearOutsideUsingIntCorners(x0: Int, y0: Int, x1: Int, y1: Int, in image: CGImage) -> CGImage? {
        let t0 = Date()
        
        let width = image.width
        let height = image.height
        let imageRect = CGRect(x: 0, y: 0, width: width, height: height)

        let minX0 = min(x0, x1)
        let maxX0 = max(x0, x1)
        let minY0 = min(y0, y1)
        let maxY0 = max(y0, y1)

        var bbox = CGRect(x: CGFloat(minX0),
                          y: CGFloat(minY0),
                          width: CGFloat(maxX0 - minX0),
                          height: CGFloat(maxY0 - minY0))

        if bbox.isNull || bbox.width <= 0 || bbox.height <= 0 {
            let bytesPerPixel = 4
            let bytesPerRow = bytesPerPixel * width
            let dataSize = bytesPerRow * height
            guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) else { return nil }
            let buffer = UnsafeMutablePointer<UInt8>.allocate(capacity: dataSize)
            buffer.initialize(repeating: 0, count: dataSize)
            defer {
                buffer.deinitialize(count: dataSize)
                buffer.deallocate()
            }
            guard let ctx = CGContext(data: buffer,
                                      width: width,
                                      height: height,
                                      bitsPerComponent: 8,
                                      bytesPerRow: bytesPerRow,
                                      space: colorSpace,
                                      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
            else { return nil }
            let out = ctx.makeImage()
            if self.debugMode {
                let dt = Date().timeIntervalSince(t0) * 1000.0
                print(String(format: "⏱ clearOutsideUsingIntCorners (empty bbox): %.2f ms", dt))
            }
            return out
        }

        bbox = bbox.intersection(imageRect)

        if bbox.isNull || bbox.width <= 0 || bbox.height <= 0 {
            let bytesPerPixel = 4
            let bytesPerRow = bytesPerPixel * width
            let dataSize = bytesPerRow * height
            guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) else { return nil }
            let buffer = UnsafeMutablePointer<UInt8>.allocate(capacity: dataSize)
            buffer.initialize(repeating: 0, count: dataSize)
            defer {
                buffer.deinitialize(count: dataSize)
                buffer.deallocate()
            }
            guard let ctx = CGContext(data: buffer,
                                      width: width,
                                      height: height,
                                      bitsPerComponent: 8,
                                      bytesPerRow: bytesPerRow,
                                      space: colorSpace,
                                      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
            else { return nil }
            let out = ctx.makeImage()
            if self.debugMode {
                let dt = Date().timeIntervalSince(t0) * 1000.0
                print(String(format: "⏱ clearOutsideUsingIntCorners (clipped empty): %.2f ms", dt))
            }
            return out
        }

        let bytesPerPixel = 4
        let bytesPerRow = bytesPerPixel * width
        let dataSize = bytesPerRow * height
        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) else { return nil }

        let rawData = UnsafeMutablePointer<UInt8>.allocate(capacity: dataSize)
        rawData.initialize(repeating: 0, count: dataSize)
        defer {
            rawData.deinitialize(count: dataSize)
            rawData.deallocate()
        }

        guard let ctx = CGContext(data: rawData,
                                  width: width,
                                  height: height,
                                  bitsPerComponent: 8,
                                  bytesPerRow: bytesPerRow,
                                  space: colorSpace,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else { return nil }

        ctx.translateBy(x: 0, y: CGFloat(height))
        ctx.scaleBy(x: 1.0, y: -1.0)
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))

        let startX = 0
        let endX = width
        let startY = 0
        let endY = height

        var kx0 = Int(floor(bbox.minX))
        var ky0 = Int(floor(bbox.minY))
        var kx1 = Int(ceil(bbox.maxX))
        var ky1 = Int(ceil(bbox.maxY))

        kx0 = max(startX, min(kx0, endX))
        kx1 = max(startX, min(kx1, endX))
        ky0 = max(startY, min(ky0, endY))
        ky1 = max(startY, min(ky1, endY))

        if kx0 > kx1 { swap(&kx0, &kx1) }
        if ky0 > ky1 { swap(&ky0, &ky1) }

        for y in startY..<endY {
            let rowBase = rawData + y * bytesPerRow
            for x in startX..<endX {
                let px = rowBase + x * bytesPerPixel
                let inside = (x >= kx0 && x < kx1 && y >= ky0 && y < ky1)
                if !inside {
                    px[0] = 0
                    px[1] = 0
                    px[2] = 0
                    px[3] = 0
                }
            }
        }

        let out = ctx.makeImage()
        if self.debugMode {
            let dt = Date().timeIntervalSince(t0) * 1000.0
            print(String(format: "⏱ clearOutsideUsingIntCorners: %.2f ms", dt))
        }
        return out
    }

    
    private func generateCutoutTwoStage(
        stage1Detections: [DetectionSmarty],
        stage1Prototypes: MLMultiArray,
        stage2Detections: [DetectionSmarty],
        stage2Prototypes: MLMultiArray?,
        primaryBBox: DetectionSmarty,
        originalImage: CVPixelBuffer
    ) {
        let funcStart = Date()
        
        let shape = stage1Prototypes.shape.map { $0.intValue }
        let C = shape[1]
        let Hp = shape[2]
        let Wp = shape[3]
        let spatial = Hp * Wp

        if self.debugMode {
            print("\n🎨 Generating TWO-STAGE UNION cutout")
            print("   Stage 1: \(stage1Detections.count) detections")
            print("   Stage 2: \(stage2Detections.count) detections")
            print("📐 Prototype shape: C=\(C), H=\(Hp), W=\(Wp)")
        }

        let protoStage1Start = Date()
        var protoMatrix1 = [Float](repeating: 0, count: C * spatial)
        if stage1Prototypes.dataType == .float32 {
            let srcBase = stage1Prototypes.dataPointer.assumingMemoryBound(to: Float.self)
            memcpy(&protoMatrix1, srcBase, C * spatial * MemoryLayout<Float>.size)
        } else {
            for c in 0..<C {
                for y in 0..<Hp {
                    for x in 0..<Wp {
                        let val = stage1Prototypes[[0, c, y, x] as [NSNumber]].floatValue
                        protoMatrix1[c * spatial + (y * Wp + x)] = val
                    }
                }
            }
        }
        let protoStage1End = Date()
        if self.debugMode {
            print(String(format: "⏱ Stage1 prototype matrix build: %.2f ms",
                         protoStage1End.timeIntervalSince(protoStage1Start) * 1000.0))
        }

        var globalMask = [Float](repeating: 0, count: spatial)

        if self.debugMode { print("\n🔵 Processing Stage 1 masks (full frame)...") }

        var primaryRawMask: [Float]? = nil
        var primaryDet: DetectionSmarty? = nil
        var stage1PixelCount = 0

        let s1MaskStart = Date()
        for (detIndex, det) in stage1Detections.enumerated() {
            var rawMask = [Float](repeating: 0, count: spatial)
            let mmulStart = Date()
            vDSP_mmul(det.maskCoeffs, 1, protoMatrix1, 1, &rawMask, 1, 1, vDSP_Length(spatial), vDSP_Length(C))
            let mmulEnd = Date()
            if self.debugMode {
                print(String(format: "   ⏱ vDSP_mmul Stage1 det[%d]: %.2f ms", detIndex,
                             mmulEnd.timeIntervalSince(mmulStart) * 1000.0))
            }

            if detIndex == 0 {
                primaryRawMask = rawMask
                primaryDet = det

                var minVal: Float = 0, maxVal: Float = 0
                vDSP_minv(rawMask, 1, &minVal, vDSP_Length(spatial))
                vDSP_maxv(rawMask, 1, &maxVal, vDSP_Length(spatial))
                var mean: Float = 0
                vDSP_meanv(rawMask, 1, &mean, vDSP_Length(spatial))

                print("\n📊 PRIMARY MASK RAW VALUES (\(det.className) @ \(Int(det.confidence*100))%):")
                print("   Range: min=\(minVal), max=\(maxVal), mean=\(mean)")

                var posCount = 0, negCount = 0, zeroCount = 0
                for v in rawMask {
                    if v > 0 { posCount += 1 }
                    else if v < 0 { negCount += 1 }
                    else { zeroCount += 1 }
                }
                print("   Distribution: \(posCount) positive, \(negCount) negative, \(zeroCount) zero")

                print("   Mask coefficients (32): [\(det.maskCoeffs.map { String(format: "%.6f", $0) }.joined(separator: ", "))]")

                let scale = Float(Wp) / 640.0
                let mx1 = max(0, Int((det.x - det.width / 2) * scale))
                let my1 = max(0, Int((det.y - det.height / 2) * scale))
                let mx2 = min(Wp, Int((det.x + det.width / 2) * scale))
                let my2 = min(Hp, Int((det.y + det.height / 2) * scale))

                print("   BBox in mask coords: (\(mx1),\(my1)) → (\(mx2),\(my2))")
            }

            let scale = Float(Wp) / 640.0
            let mx1 = max(0, Int((det.x - det.width / 2) * scale))
            let my1 = max(0, Int((det.y - det.height / 2) * scale))
            let mx2 = min(Wp, Int((det.x + det.width / 2) * scale))
            let my2 = min(Hp, Int((det.y + det.height / 2) * scale))

            var addedPixels = 0
            rawMask.withUnsafeBufferPointer { rPtr in
                globalMask.withUnsafeMutableBufferPointer { gPtr in
                    if mx2 > mx1 && my2 > my1 {
                        for py in my1..<my2 {
                            let rowStart = py * Wp + mx1
                            let rowLen = mx2 - mx1
                            let base = rowStart
                            for i in 0..<rowLen {
                                let idx = base + i
                                if rPtr[idx] > maskThreshold && gPtr[idx] == 0 {
                                    gPtr[idx] = 1.0
                                    addedPixels += 1
                                }
                            }
                        }
                    }
                }
            }

            if self.debugMode && detIndex < 5 {
                print("   ✅ S1 \(det.className) @ \(Int(det.confidence*100))%: bbox(\(mx1),\(my1))→(\(mx2),\(my2)), +\(addedPixels)px")
            }
        }
        let s1MaskEnd = Date()

        for i in 0..<spatial { if globalMask[i] > 0 { stage1PixelCount += 1 } }
        if self.debugMode {
            print("   ⚙️ Mask threshold: \(maskThreshold)")
            print("   📊 After Stage 1: \(stage1PixelCount)/\(spatial) pixels (\(String(format: "%.1f", Float(stage1PixelCount)/Float(spatial)*100))%)")
            print(String(format: "⏱ Stage1 mask build+apply: %.2f ms", s1MaskEnd.timeIntervalSince(s1MaskStart) * 1000.0))
        }

        if self.debugMode, let rawMask = primaryRawMask, let det = primaryDet {
            saveMaskToFile(rawMask: rawMask, width: Wp, height: Hp, detection: det)
        }

        // Stage 2 masks
        if let proto2 = stage2Prototypes, !stage2Detections.isEmpty {
            let s2ProtoStart = Date()
            if self.debugMode { print("\n🟢 Processing Stage 2 masks (cropped → full frame)...") }

            var protoMatrix2 = [Float](repeating: 0, count: C * spatial)
            if proto2.dataType == .float32 {
                let srcBase = proto2.dataPointer.assumingMemoryBound(to: Float.self)
                memcpy(&protoMatrix2, srcBase, C * spatial * MemoryLayout<Float>.size)
            } else {
                for c in 0..<C {
                    for y in 0..<Hp {
                        for x in 0..<Wp {
                            let val = proto2[[0, c, y, x] as [NSNumber]].floatValue
                            protoMatrix2[c * spatial + (y * Wp + x)] = val
                        }
                    }
                }
            }
            let s2ProtoEnd = Date()
            if self.debugMode {
                print(String(format: "⏱ Stage2 prototype matrix build: %.2f ms",
                             s2ProtoEnd.timeIntervalSince(s2ProtoStart) * 1000.0))
            }

            let padding: Float = 0.1
            let cropX1 = max(0, primaryBBox.x - primaryBBox.width / 2 * (1 + padding))
            let cropY1 = max(0, primaryBBox.y - primaryBBox.height / 2 * (1 + padding))
            let cropX2 = min(640, primaryBBox.x + primaryBBox.width / 2 * (1 + padding))
            let cropY2 = min(640, primaryBBox.y + primaryBBox.height / 2 * (1 + padding))
            let cropW = cropX2 - cropX1
            let cropH = cropY2 - cropY1

            if self.debugMode {
                print("   Crop region (model): (\(Int(cropX1)),\(Int(cropY1)))→(\(Int(cropX2)),\(Int(cropY2))) = \(Int(cropW))x\(Int(cropH))")
            }

            let scale = Float(Wp) / 640.0

            let s2MaskStart = Date()
            for det in stage2Detections {
                var rawMask = [Float](repeating: 0, count: spatial)
                let mmulStart = Date()
                vDSP_mmul(det.maskCoeffs, 1, protoMatrix2, 1, &rawMask, 1, 1, vDSP_Length(spatial), vDSP_Length(C))
                let mmulEnd = Date()
                if self.debugMode {
                    print(String(format: "   ⏱ vDSP_mmul Stage2: %.2f ms",
                                 mmulEnd.timeIntervalSince(mmulStart) * 1000.0))
                }

                let mx1_crop = max(0, Int((det.x - det.width / 2) * scale))
                let my1_crop = max(0, Int((det.y - det.height / 2) * scale))
                let mx2_crop = min(Wp, Int((det.x + det.width / 2) * scale))
                let my2_crop = min(Hp, Int((det.y + det.height / 2) * scale))

                var addedPixels = 0

                rawMask.withUnsafeBufferPointer { rPtr in
                    globalMask.withUnsafeMutableBufferPointer { gPtr in
                        if mx2_crop > mx1_crop && my2_crop > my1_crop {
                            for py_crop in my1_crop..<my2_crop {
                                let base = py_crop * Wp
                                for px_crop in mx1_crop..<mx2_crop {
                                    let cropIdx = base + px_crop
                                    if rPtr[cropIdx] > maskThreshold {
                                        let fracX = Float(px_crop) / Float(Wp)
                                        let fracY = Float(py_crop) / Float(Hp)
                                        let fullX = cropX1 + fracX * cropW
                                        let fullY = cropY1 + fracY * cropH
                                        let mx_full = Int(fullX * scale)
                                        let my_full = Int(fullY * scale)
                                        if mx_full >= 0 && mx_full < Wp && my_full >= 0 && my_full < Hp {
                                            let fullIdx = my_full * Wp + mx_full
                                            if gPtr[fullIdx] == 0 {
                                                addedPixels += 1
                                            }
                                            gPtr[fullIdx] = 1.0
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                if self.debugMode {
                    print("   ✅ S2 \(det.className) @ \(Int(det.confidence*100))%: bbox(\(mx1_crop),\(my1_crop))→(\(mx2_crop),\(my2_crop)), +\(addedPixels)px NEW")
                }
            }
            let s2MaskEnd = Date()
            if self.debugMode {
                print(String(format: "⏱ Stage2 mask build+apply: %.2f ms",
                             s2MaskEnd.timeIntervalSince(s2MaskStart) * 1000.0))
            }
        }

        var finalPixelCount = 0
        for i in 0..<spatial { if globalMask[i] > 0 { finalPixelCount += 1 } }
        let addedByStage2 = finalPixelCount - stage1PixelCount

        if self.debugMode {
            print("\n📊 MERGED MASK: \(finalPixelCount)/\(spatial) pixels (\(String(format: "%.1f", Float(finalPixelCount)/Float(spatial)*100))%)")
            print("   Stage 1 contributed: \(stage1PixelCount) pixels")
            print("   Stage 2 added: \(addedByStage2) NEW pixels")
        }

        var binaryMask = [UInt8](repeating: 0, count: spatial)
        for i in 0..<spatial { if globalMask[i] > 0 { binaryMask[i] = 255 } }

        if self.debugMode { print20x20BinaryGrid("MERGED STAGE1+STAGE2", mask: binaryMask, width: Wp, height: Hp) }

        var minX = Wp
        var maxX = -1
        var minY = Hp
        var maxY = -1
        for y in 0..<Hp {
            let rowBase = y * Wp
            for x in 0..<Wp {
                if globalMask[rowBase + x] > 0 {
                    if x < minX { minX = x }
                    if x > maxX { maxX = x }
                    if y < minY { minY = y }
                    if y > maxY { maxY = y }
                }
            }
        }

        let maskBuildEnd = Date()
        if self.debugMode {
            print(String(format: "⏱ Mask building (Stage1+Stage2+tight bbox): %.2f ms",
                         maskBuildEnd.timeIntervalSince(funcStart) * 1000.0))
        }

        autoreleasepool {
            let renderStart = Date()
            let ciImage = CIImage(cvPixelBuffer: originalImage)
            let width = CVPixelBufferGetWidth(originalImage)
            let height = CVPixelBufferGetHeight(originalImage)

            guard let cgImage = ciContext.createCGImage(ciImage, from: ciImage.extent) else {
                if self.debugMode { print("❌ Failed to create CGImage") }
                DispatchQueue.main.async { self.isProcessing = false }
                return
            }

            guard let ctx = CGContext(
                data: nil,
                width: width,
                height: height,
                bitsPerComponent: 8,
                bytesPerRow: width * 4,
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
            ) else {
                if self.debugMode { print("❌ Failed to create CGContext") }
                DispatchQueue.main.async { self.isProcessing = false }
                return
            }

            ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))

            guard let data = ctx.data else {
                if self.debugMode { print("❌ CGContext has no data") }
                DispatchQueue.main.async { self.isProcessing = false }
                return
            }

            let pixels = data.bindMemory(to: UInt8.self, capacity: width * height * 4)

            let scaleX = Float(Wp) / Float(width)
            let scaleY = Float(Hp) / Float(height)

            if self.debugMode {
                print("🖼️ Upscaling \(Wp)×\(Hp) → \(width)×\(height)")
            }

            var opaqueCount = 0

            var xMap = [Int](repeating: 0, count: width)
            for px in 0..<width { xMap[px] = min(max(Int(Float(px) * scaleX), 0), Wp - 1) }

            let keptDetections: [DetectionSmarty] = (stage1Detections + stage2Detections)

            if keptDetections.isEmpty {
                memset(data, 0, width * height * 4)
                if self.debugMode { print("📊 Output: 0/\(width * height) opaque (0.0%)") }
            } else {
                let modelSize: Float = 640.0
                var imageRects = [(x0: Int, y0: Int, x1: Int, y1: Int)]()
                imageRects.reserveCapacity(keptDetections.count)

                for det in keptDetections {
                    let left = det.x - det.width / 2.0
                    let right = det.x + det.width / 2.0
                    let top = det.y - det.height / 2.0
                    let bottom = det.y + det.height / 2.0

                    let sx = Float(width) / modelSize
                    let sy = Float(height) / modelSize

                    var ix0 = Int(floor(left * sx))
                    var ix1 = Int(ceil(right * sx))
                    var iy0 = Int(floor(top * sy))
                    var iy1 = Int(ceil(bottom * sy))

                    ix0 = max(0, min(ix0, width))
                    ix1 = max(0, min(ix1, width))
                    iy0 = max(0, min(iy0, height))
                    iy1 = max(0, min(iy1, height))

                    if ix0 < ix1 && iy0 < iy1 {
                        imageRects.append((x0: ix0, y0: iy0, x1: ix1, y1: iy1))
                    }
                }

                var rowIntervals = Array(repeating: [(start:Int,end:Int)](), count: height)
                for r in imageRects {
                    for y in r.y0..<r.y1 {
                        rowIntervals[y].append((start: r.x0, end: r.x1))
                    }
                }

                for y in 0..<height {
                    if rowIntervals[y].isEmpty { continue }
                    var intervals = rowIntervals[y]
                    intervals.sort { $0.start < $1.start }
                    var merged: [(Int,Int)] = []
                    var cur = intervals[0]
                    for i in 1..<intervals.count {
                        let it = intervals[i]
                        if it.start <= cur.end { cur.end = max(cur.end, it.end) } else { merged.append(cur); cur = it }
                    }
                    merged.append(cur)
                    rowIntervals[y] = merged
                }

                for py in 0..<height {
                    let my = min(max(Int(Float(py) * scaleY), 0), Hp - 1)
                    let maskRowOffset = my * Wp
                    let rowBase = pixels.advanced(by: py * width * 4)

                    let intervals = rowIntervals[py]
                    if intervals.isEmpty {
                        memset(rowBase, 0, width * 4)
                        continue
                    }

                    var x = 0
                    var intervalIndex = 0

                    while x < width {
                        let nextInterval = intervalIndex < intervals.count ? intervals[intervalIndex] : (start: width, end: width)
                        if x < nextInterval.start {
                            let len = min(nextInterval.start, width) - x
                            let byteOffset = x * 4
                            memset(rowBase.advanced(by: byteOffset), 0, len * 4)
                            x += len
                            continue
                        }

                        let runEnd = min(nextInterval.end, width)
                        var pxIdx = x
                        while pxIdx < runEnd {
                            let maskIdx = maskRowOffset + xMap[pxIdx]
                            let pixelPtr = rowBase.advanced(by: pxIdx * 4)
                            if globalMask[maskIdx] > 0 {
                                pixelPtr[3] = 255
                                opaqueCount += 1
                            } else {
                                pixelPtr[0] = 0; pixelPtr[1] = 0; pixelPtr[2] = 0; pixelPtr[3] = 0
                            }
                            pxIdx += 1
                        }
                        x = runEnd
                        intervalIndex += 1
                    }
                }

                if self.debugMode {
                    print("📊 Output: \(opaqueCount)/\(width * height) opaque (\(String(format: "%.1f", Float(opaqueCount)/Float(width*height)*100))%)")
                }
            }

            if maxX >= 0 && maxY >= 0 {
                let scaleImgX = Float(width) / Float(Wp)
                let scaleImgY = Float(height) / Float(Hp)
                let x0 = CGFloat(Float(minX) * scaleImgX)
                let y0 = CGFloat(Float(minY) * scaleImgY)
                let w = CGFloat(Float(maxX - minX + 1) * scaleImgX)
                let h = CGFloat(Float(maxY - minY + 1) * scaleImgY)
                var tightRect = CGRect(x: x0, y: y0, width: w, height: h)
                tightRect = tightRect.intersection(CGRect(x: 0, y: 0, width: width, height: height))
                ctx.setStrokeColor(CGColor(red: 0, green: 1, blue: 1, alpha: 1))
                ctx.setLineWidth(3.0)
                ctx.stroke(tightRect)
            } else {
                if !stage1Detections.isEmpty {
                    let det = stage1Detections[0]
                    let modelSize: CGFloat = 640.0
                    let sX = CGFloat(width) / modelSize
                    let sY = CGFloat(height) / modelSize
                    let centerX = CGFloat(det.x) * sX
                    let centerY = CGFloat(det.y) * sY
                    let boxWidth = CGFloat(det.width) * sX
                    let boxHeight = CGFloat(det.height) * sY
                    let rect = CGRect(x: centerX - boxWidth/2, y: centerY - boxHeight/2, width: boxWidth, height: boxHeight)
                    ctx.setStrokeColor(CGColor(red: 0, green: 1, blue: 1, alpha: 1))
                    ctx.setLineWidth(3.0)
                    ctx.stroke(rect)
                }
            }

            let renderEnd = Date()
            if self.debugMode {
                print(String(format: "⏱ Rendering + upscaling + cutout: %.2f ms",
                             renderEnd.timeIntervalSince(renderStart) * 1000.0))
                print(String(format: "⏱ generateCutoutTwoStage total: %.2f ms",
                             renderEnd.timeIntervalSince(funcStart) * 1000.0))
            }

            if let outImage = ctx.makeImage() {
                DispatchQueue.main.async {
                    self.maskImageView.image = UIImage(cgImage: outImage, scale: 1.0, orientation: .up)
                    self.isProcessing = false
                    if self.debugMode { print("✅ ==================== FRAME COMPLETE ====================\n") }
                }
            } else {
                if self.debugMode { print("❌ Failed to make output image") }
                DispatchQueue.main.async { self.isProcessing = false }
            }
        }
    }



    private func drawBoundingBox(ctx: CGContext, detection: DetectionSmarty, imageWidth: Int, imageHeight: Int) {
        let originalWidth = CGFloat(imageWidth)
        let originalHeight = CGFloat(imageHeight)
        let modelSize: CGFloat = 640.0
        let scaleX = originalWidth / modelSize
        let scaleY = originalHeight / modelSize

        let centerX = CGFloat(detection.x) * scaleX
        let centerY = CGFloat(detection.y) * scaleY
        let boxWidth = CGFloat(detection.width) * scaleX
        let boxHeight = CGFloat(detection.height) * scaleY

        let x = centerX - boxWidth / 2
        let y = centerY - boxHeight / 2

        ctx.setStrokeColor(CGColor(red: 0, green: 1, blue: 1, alpha: 1))
        ctx.setLineWidth(3.0)
        let rect = CGRect(x: x, y: y, width: boxWidth, height: boxHeight)
        ctx.stroke(rect)

        let confidence = Int(detection.confidence * 100)
        let labelText = "\(detection.className) \(confidence)%"
        let attributed = NSAttributedString(string: labelText, attributes: bboxAttributes)
        let textSize = attributed.size()

        let labelPadding: CGFloat = 6
        let labelWidth = textSize.width + (labelPadding * 2)
        let labelHeight = textSize.height + (labelPadding * 2)

        var labelX: CGFloat
        var labelY: CGFloat

        if y - labelHeight - 5 >= 0 {
            labelX = max(0, min(x, originalWidth - labelWidth))
            labelY = y - labelHeight - 5
        } else if y + boxHeight + labelHeight + 5 <= originalHeight {
            labelX = max(0, min(x, originalWidth - labelWidth))
            labelY = y + boxHeight + 5
        } else {
            labelX = max(0, min(x + 5, originalWidth - labelWidth))
            labelY = max(0, y + 5)
        }

        ctx.setFillColor(CGColor(red: 0, green: 1, blue: 1, alpha: 1))
        let labelRect = CGRect(x: labelX, y: labelY, width: labelWidth, height: labelHeight)
        ctx.fill(labelRect)

        let textX = labelX + labelPadding
        let textY = labelY + labelPadding

        let line = CTLineCreateWithAttributedString(attributed)

        ctx.saveGState()
        ctx.textMatrix = .identity

        let ctm = ctx.ctm
        let isFlipped = ctm.d < 0 || ctm.ty != 0

        if isFlipped {
            ctx.translateBy(x: 0, y: CGFloat(imageHeight))
            ctx.scaleBy(x: 1.0, y: -1.0)
            let ascent = CGFloat(CTLineGetTypographicBounds(line, nil, nil, nil))
            let flippedY = CGFloat(imageHeight) - textY - ascent
            ctx.textPosition = CGPoint(x: textX, y: flippedY)
        } else {
            ctx.textPosition = CGPoint(x: textX, y: textY)
        }

        CTLineDraw(line, ctx)
        ctx.restoreGState()
    }

    // MARK: - Extract Detections (with timing)
    private func extractDetections(from detections: MLMultiArray) -> [DetectionSmarty] {
        let t0 = Date()
        var all: [DetectionSmarty] = []

        let numFeatures = detections.shape[1].intValue
        let numAnchors = detections.shape[2].intValue

        let numClasses = numFeatures - 4 - 32

        if self.debugMode {
            print("🔍 Tensor shape: [1, \(numFeatures), \(numAnchors)]")
            print("   → \(numClasses) classes, \(numAnchors) predictions")
            print("   → Mode: \(detectAllObjects ? "ALL OBJECTS" : "FURNITURE ONLY")")
            if numClasses == 4585 {
                print("   → Model: YOLOE (LVIS open-vocabulary)")
            } else if numClasses == 80 {
                print("   → Model: YOLO11-seg (COCO)")
            }
        }

        let totalCount = detections.count
        let detBuf = UnsafeMutablePointer<Float>.allocate(capacity: totalCount)
        defer { detBuf.deallocate() }

        let copyStart = Date()
        if detections.dataType == .float16 {
            let src = detections.dataPointer.bindMemory(to: UInt16.self, capacity: totalCount)
            var srcBuf = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: src),
                                       height: 1, width: vImagePixelCount(totalCount),
                                       rowBytes: totalCount * MemoryLayout<UInt16>.size)
            var dstBuf = vImage_Buffer(data: UnsafeMutableRawPointer(detBuf),
                                       height: 1, width: vImagePixelCount(totalCount),
                                       rowBytes: totalCount * MemoryLayout<Float>.size)
            vImageConvert_Planar16FtoPlanarF(&srcBuf, &dstBuf, vImage_Flags(kvImageNoFlags))
        } else if detections.dataType == .float32 {
            let src = detections.dataPointer.assumingMemoryBound(to: Float.self)
            memcpy(detBuf, src, totalCount * MemoryLayout<Float>.size)
        } else {
            for i in 0..<totalCount {
                detBuf[i] = detections[i].floatValue
            }
        }
        let copyEnd = Date()
        if self.debugMode {
            print(String(format: "⏱ extractDetections copy/convert: %.2f ms",
                         copyEnd.timeIntervalSince(copyStart) * 1000.0))
        }

        let coeffOffset = 4 + numClasses
        let stride = numAnchors

        let decodeStart = Date()
        if detectAllObjects {
            for anchor in 0..<numAnchors {
                let x = detBuf[0 * stride + anchor]
                let y = detBuf[1 * stride + anchor]
                let w = detBuf[2 * stride + anchor]
                let h = detBuf[3 * stride + anchor]

                var bestConf: Float = 0
                var bestClassIdx = -1

                var baseConfIdx = (4) * stride + anchor
                for classIdx in 0..<numClasses {
                    let conf = detBuf[baseConfIdx + classIdx * stride]
                    if conf > bestConf {
                        bestConf = conf
                        bestClassIdx = classIdx
                    }
                }

                if bestConf > confidenceThreshold && bestClassIdx >= 0 {
                    var coeffs = [Float](repeating: 0, count: 32)
                    let coeffStart = coeffOffset * stride + anchor
                    for k in 0..<32 {
                        coeffs[k] = detBuf[coeffStart + k * stride]
                    }

                    let className = furnitureClasses[bestClassIdx] ?? "object_\(bestClassIdx)"
                    all.append(DetectionSmarty(
                        x: x, y: y, width: w, height: h,
                        confidence: bestConf, classIdx: bestClassIdx, className: className,
                        maskCoeffs: coeffs
                    ))
                }
            }
        } else {
            let furnitureList = furnitureClasses.filter { $0.key < numClasses }

            for anchor in 0..<numAnchors {
                let x = detBuf[0 * stride + anchor]
                let y = detBuf[1 * stride + anchor]
                let w = detBuf[2 * stride + anchor]
                let h = detBuf[3 * stride + anchor]

                for (classIdx, className) in furnitureList {
                    let confIdx = (4 + classIdx) * stride + anchor
                    let conf = detBuf[confIdx]
                    if conf > confidenceThreshold {
                        var coeffs = [Float](repeating: 0, count: 32)
                        let coeffStart = coeffOffset * stride + anchor
                        for k in 0..<32 {
                            coeffs[k] = detBuf[coeffStart + k * stride]
                        }
                        all.append(DetectionSmarty(
                            x: x, y: y, width: w, height: h,
                            confidence: conf, classIdx: classIdx, className: className,
                            maskCoeffs: coeffs
                        ))
                    }
                }
            }
        }
        let decodeEnd = Date()

        if self.debugMode {
            print(String(format: "⏱ extractDetections decode loop: %.2f ms",
                         decodeEnd.timeIntervalSince(decodeStart) * 1000.0))

            let grouped = Dictionary(grouping: all) { $0.className }
            print("\n📊 DETECTION SUMMARY: \(all.count) total")
            for (className, dets) in grouped.sorted(by: { $0.value.count > $1.value.count }).prefix(20) {
                let confidences = dets.map { Int($0.confidence * 100) }
                print("  - \(className): \(dets.count)x, conf: \(confidences)%")
            }
            if grouped.count > 20 {
                print("  ... and \(grouped.count - 20) more classes")
            }
            let tEnd = Date()
            print(String(format: "⏱ extractDetections total: %.2f ms",
                         tEnd.timeIntervalSince(t0) * 1000.0))
        }

        return all
    }


    // MARK: - Pixel Buffer to MLMultiArray (Accelerate) — with timing
    private func pixelBufferToMLMultiArray(_ pixelBuffer: CVPixelBuffer) -> MLMultiArray? {
        let t0 = Date()
        guard let array = try? MLMultiArray(shape: [1, 3, 640, 640], dataType: .float32) else { return nil }
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else { return nil }

        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let width = 640
        let height = 640
        let pixelCount = width * height
        let src = baseAddress.assumingMemoryBound(to: UInt8.self)

        let floatSize = MemoryLayout<Float32>.size
        let planeStrideBytes = pixelCount * floatSize
        let rPtr = array.dataPointer.advanced(by: 0 * planeStrideBytes).assumingMemoryBound(to: Float32.self)
        let gPtr = array.dataPointer.advanced(by: 1 * planeStrideBytes).assumingMemoryBound(to: Float32.self)
        let bPtr = array.dataPointer.advanced(by: 2 * planeStrideBytes).assumingMemoryBound(to: Float32.self)

        var indicesR = [vDSP_Length](repeating: 0, count: width)
        var indicesG = [vDSP_Length](repeating: 0, count: width)
        var indicesB = [vDSP_Length](repeating: 0, count: width)
        for i in 0..<width {
            indicesR[i] = vDSP_Length(2 + i * 4)
            indicesG[i] = vDSP_Length(1 + i * 4)
            indicesB[i] = vDSP_Length(0 + i * 4)
        }

        var rowUInt8 = [UInt8](repeating: 0, count: width * 4)
        var rowFloat = [Float](repeating: 0, count: width * 4)

        var scaleF: Float = 1.0 / 255.0

        for y in 0..<height {
            let rowStart = src.advanced(by: y * bytesPerRow)
            memcpy(&rowUInt8, rowStart, width * 4)

            rowUInt8.withUnsafeBufferPointer { u8Ptr in
                rowFloat.withUnsafeMutableBufferPointer { fPtr in
                    vDSP_vfltu8(u8Ptr.baseAddress!, 1, fPtr.baseAddress!, 1, vDSP_Length(width * 4))
                    vDSP_vsmul(fPtr.baseAddress!, 1, &scaleF, fPtr.baseAddress!, 1, vDSP_Length(width * 4))
                }
            }

            rowFloat.withUnsafeBufferPointer { rf in
                let baseF = rf.baseAddress!
                vDSP_vgathr(baseF, indicesR, 1, rPtr.advanced(by: y * width), 1, vDSP_Length(width))
                vDSP_vgathr(baseF, indicesG, 1, gPtr.advanced(by: y * width), 1, vDSP_Length(width))
                vDSP_vgathr(baseF, indicesB, 1, bPtr.advanced(by: y * width), 1, vDSP_Length(width))
            }
        }

        if self.debugMode {
            let dt = Date().timeIntervalSince(t0) * 1000.0
            print(String(format: "⏱ pixelBufferToMLMultiArray: %.2f ms", dt))
        }

        return array
    }
    
    public func cutoutClearOutsideAccelerated(x0: Int, y0: Int, x1: Int, y1: Int, in image: CGImage) -> CGImage? {
        let t0 = Date()
        
        let width = image.width
        let height = image.height
        guard width > 0 && height > 0 else { return nil }

        var minX = min(x0, x1)
        var maxX = max(x0, x1)
        var minY = min(y0, y1)
        var maxY = max(y0, y1)

        minX = max(0, min(minX, width))
        maxX = max(0, min(maxX, width))
        minY = max(0, min(minY, height))
        maxY = max(0, min(maxY, height))

        if minX >= maxX || minY >= maxY {
            let out = makeTransparentImage(width: width, height: height)
            if self.debugMode {
                let dt = Date().timeIntervalSince(t0) * 1000.0
                print(String(format: "⏱ cutoutClearOutsideAccelerated (empty): %.2f ms", dt))
            }
            return out
        }

        let bytesPerPixel = 4
        let bytesPerRow = width * bytesPerPixel
        let bufSize = bytesPerRow * height
        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) else { return nil }

        guard let destData = malloc(bufSize) else { return nil }
        defer { free(destData) }

        guard let ctx = CGContext(data: destData,
                                  width: width,
                                  height: height,
                                  bitsPerComponent: 8,
                                  bytesPerRow: bytesPerRow,
                                  space: colorSpace,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else {
            return nil
        }

        ctx.translateBy(x: 0, y: CGFloat(height))
        ctx.scaleBy(x: 1.0, y: -1.0)
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))

        var srcBuffer = vImage_Buffer(data: destData, height: vImagePixelCount(height),
                                      width: vImagePixelCount(width), rowBytes: bytesPerRow)

        guard let zeroRow = malloc(bytesPerRow) else { return nil }
        memset(zeroRow, 0, bytesPerRow)
        defer { free(zeroRow) }

        if minY > 0 {
            let dstPtr = destData
            for r in 0..<minY {
                let rowBase = dstPtr.advanced(by: r * bytesPerRow)
                memcpy(rowBase, zeroRow, bytesPerRow)
            }
        }

        if maxY < height {
            let dstPtr = destData.advanced(by: maxY * bytesPerRow)
            for r in 0..<(height - maxY) {
                let rowBase = dstPtr.advanced(by: r * bytesPerRow)
                memcpy(rowBase, zeroRow, bytesPerRow)
            }
        }

        if minX > 0 || maxX < width {
            let leftBytes = minX * bytesPerPixel
            let rightBytes = (width - maxX) * bytesPerPixel
            for row in minY..<maxY {
                let rowBase = destData.advanced(by: row * bytesPerRow)
                if leftBytes > 0 {
                    memset(rowBase, 0, leftBytes)
                }
                if rightBytes > 0 {
                    let rightPtr = rowBase.advanced(by: maxX * bytesPerPixel)
                    memset(rightPtr, 0, rightBytes)
                }
            }
        }

        guard let outCtx = CGContext(data: destData,
                                     width: width,
                                     height: height,
                                     bitsPerComponent: 8,
                                     bytesPerRow: bytesPerRow,
                                     space: colorSpace,
                                     bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else {
            return nil
        }

        let outImage = outCtx.makeImage()
        if self.debugMode {
            let dt = Date().timeIntervalSince(t0) * 1000.0
            print(String(format: "⏱ cutoutClearOutsideAccelerated: %.2f ms", dt))
        }
        return outImage
    }

    private func makeTransparentImage(width: Int, height: Int) -> CGImage? {
        guard width > 0 && height > 0 else { return nil }
        let bytesPerPixel = 4
        let bytesPerRow = width * bytesPerPixel
        let bufSize = bytesPerRow * height
        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) else { return nil }
        guard let data = calloc(1, bufSize) else { return nil }
        defer { free(data) }

        guard let ctx = CGContext(data: data,
                                  width: width,
                                  height: height,
                                  bitsPerComponent: 8,
                                  bytesPerRow: bytesPerRow,
                                  space: colorSpace,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else { return nil }

        return ctx.makeImage()
    }

    public func cutoutClearOutsideAcceleratedUIImage(x0: Int, y0: Int, x1: Int, y1: Int, in image: UIImage) -> UIImage? {
        guard let cg = image.cgImage else { return nil }
        guard let outCG = cutoutClearOutsideAccelerated(x0: x0, y0: y0, x1: x1, y1: y1, in: cg) else { return nil }
        return UIImage(cgImage: outCG, scale: image.scale, orientation: image.imageOrientation)
    }

}
