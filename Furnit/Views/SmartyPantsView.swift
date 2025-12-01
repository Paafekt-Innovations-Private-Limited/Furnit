// SmartyPantsView.swift
// Two-Stage Detection: Full frame -> Crop to primary bbox -> Re-detect -> UNION BOTH

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
    // Default 0.0, but try -2.0 or -3.0 if masks are fragmentary
    var maskThreshold: Float = 0.0

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
    // YOLOE-pf detects 4585 classes - these are the ones we care about
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
        
        addSubview(maskImageView)
        maskImageView.translatesAutoresizingMaskIntoConstraints = false
        NSLayoutConstraint.activate([
            maskImageView.centerXAnchor.constraint(equalTo: centerXAnchor),
            maskImageView.centerYAnchor.constraint(equalTo: centerYAnchor),
            maskImageView.widthAnchor.constraint(equalTo: widthAnchor),
            maskImageView.heightAnchor.constraint(equalTo: heightAnchor)
        ])
        
        let pinchGesture = UIPinchGestureRecognizer(target: self, action: #selector(handlePinch(_:)))
        pinchGesture.delegate = self
        self.addGestureRecognizer(pinchGesture)
        
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
    
    // MARK: - UIGestureRecognizerDelegate
    func gestureRecognizer(_ gestureRecognizer: UIGestureRecognizer, shouldReceive touch: UITouch) -> Bool {
        let location = touch.location(in: self)
        if location.y < 100 { return false }
        return true
    }
    
    func gestureRecognizer(_ gestureRecognizer: UIGestureRecognizer, shouldRecognizeSimultaneouslyWith otherGestureRecognizer: UIGestureRecognizer) -> Bool {
        return gestureRecognizer is UIPinchGestureRecognizer
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

    // MARK: - Crop Pixel Buffer to BBox
    private func cropPixelBuffer(_ pixelBuffer: CVPixelBuffer, toBBox det: DetectionSmarty, padding: Float = 0.1) -> CVPixelBuffer? {
        let fullW = Float(CVPixelBufferGetWidth(pixelBuffer))
        let fullH = Float(CVPixelBufferGetHeight(pixelBuffer))
        
        // Convert model coords (640) to pixel coords
        let scaleX = fullW / 640.0
        let scaleY = fullH / 640.0
        
        let centerX = det.x * scaleX
        let centerY = det.y * scaleY
        let boxW = det.width * scaleX
        let boxH = det.height * scaleY
        
        // Add padding (outside)
        let padW = boxW * padding
        let padH = boxH * padding
        
        var x1 = centerX - boxW / 2 - padW
        var y1 = centerY - boxH / 2 - padH
        var x2 = centerX + boxW / 2 + padW
        var y2 = centerY + boxH / 2 + padH
        
        // Clamp to image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(fullW, x2)
        y2 = min(fullH, y2)
        
        let cropW = Int(x2 - x1)
        let cropH = Int(y2 - y1)
        
        guard cropW > 10 && cropH > 10 else { return nil }
        
        // Use CGContext for reliable cropping (CIContext has coordinate issues)
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        
        guard let srcBase = CVPixelBufferGetBaseAddress(pixelBuffer) else { return nil }
        let srcBytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let srcPtr = srcBase.assumingMemoryBound(to: UInt8.self)
        
        var out: CVPixelBuffer?
        CVPixelBufferCreate(kCFAllocatorDefault, cropW, cropH, kCVPixelFormatType_32BGRA, nil, &out)
        guard let dst = out else { return nil }
        
        CVPixelBufferLockBaseAddress(dst, [])
        defer { CVPixelBufferUnlockBaseAddress(dst, []) }
        
        guard let dstBase = CVPixelBufferGetBaseAddress(dst) else { return nil }
        let dstBytesPerRow = CVPixelBufferGetBytesPerRow(dst)
        let dstPtr = dstBase.assumingMemoryBound(to: UInt8.self)
        
        // Copy pixel data row by row
        let x1Int = Int(x1)
        let y1Int = Int(y1)
        
        for row in 0..<cropH {
            let srcRow = y1Int + row
            let srcOffset = srcRow * srcBytesPerRow + x1Int * 4
            let dstOffset = row * dstBytesPerRow
            memcpy(dstPtr + dstOffset, srcPtr + srcOffset, cropW * 4)
        }
        
        if self.debugMode {
            print("✂️ Cropped: (\(x1Int),\(y1Int)) → (\(Int(x2)),\(Int(y2))) = \(cropW)x\(cropH)")
        }
        
        return dst
    }

//    // MARK: - Main Processing (Two-Stage with UNION)
//    private func processFrame(_ pixelBuffer: CVPixelBuffer) {
//        guard let model = mlModel else { return }
//        let now = Date()
//        guard now.timeIntervalSince(lastProcessTime) >= processInterval, !isProcessing else { return }
//        lastProcessTime = now
//        isProcessing = true
//        
//        if self.debugMode { print("\n🔬 ========== STAGE 1: FULL FRAME ==========") }
//
//        // STAGE 1: Full frame detection
//        guard let resized = letterbox(pixelBuffer, size: 640) else {
//            isProcessing = false
//            return
//        }
//        
//        guard let inputArray = pixelBufferToMLMultiArray(resized) else {
//            isProcessing = false
//            return
//        }
//        
//        guard let inputProvider = try? MLDictionaryFeatureProvider(dictionary: ["image": inputArray]) else {
//            isProcessing = false
//            return
//        }
//        
//        guard let output = try? model.prediction(from: inputProvider) else {
//            isProcessing = false
//            return
//        }
//        
//        // Log available output names (helps debug different models)
//        if self.debugMode {
//            let names = output.featureNames.joined(separator: ", ")
//            print("📤 Model outputs: \(names)")
//        }
//        
//        var detectionsArray: MLMultiArray?
//        if let arr = output.featureValue(for: "var_1432")?.multiArrayValue {
//            detectionsArray = arr
//        } else if let arr = output.featureValue(for: "var_2421")?.multiArrayValue {
//            detectionsArray = arr
//        } else {
//            // Try to find any MLMultiArray output that looks like detections
//            for name in output.featureNames {
//                if let arr = output.featureValue(for: name)?.multiArrayValue {
//                    let shape = arr.shape.map { $0.intValue }
//                    // Detection array has shape [1, features, predictions]
//                    if shape.count == 3 && shape[0] == 1 && shape[1] > 100 {
//                        detectionsArray = arr
//                        if self.debugMode { print("   → Using '\(name)' as detections: \(shape)") }
//                        break
//                    }
//                }
//            }
//        }
//        
//        guard let detArray = detectionsArray else {
//            isProcessing = false
//            return
//        }
//        
//        guard let prototypesArray = output.featureValue(for: "p")?.multiArrayValue else {
//            isProcessing = false
//            return
//        }
//
//        let stage1Detections = extractDetections(from: detArray)
//        if self.debugMode { print("📊 Stage 1: \(stage1Detections.count) detections") }
//        
//        if stage1Detections.isEmpty {
//            DispatchQueue.main.async {
//                self.maskImageView.image = nil
//                self.isProcessing = false
//            }
//            return
//        }
//        
//        // Get primary detection
//        let sorted = stage1Detections.sorted { $0.confidence > $1.confidence }
//        let primary = sorted.first!
//        
//        if self.debugMode {
//            print("🎯 Primary: \(primary.className) @ \(Int(primary.confidence * 100))%")
//            print("   BBox: center(\(Int(primary.x)), \(Int(primary.y))) size(\(Int(primary.width))x\(Int(primary.height)))")
//        }
//        
//        // STAGE 2: Crop to primary bbox and re-detect
//        if self.debugMode { print("\n🔬 ========== STAGE 2: CROPPED ==========") }
//        
//        var stage2Detections: [DetectionSmarty] = []
//        var stage2Prototypes: MLMultiArray? = nil
//        
//        if let croppedBuffer = cropPixelBuffer(pixelBuffer, toBBox: primary, padding: 0.1),
//           let resizedCrop = letterbox(croppedBuffer, size: 640),
//           let cropInputArray = pixelBufferToMLMultiArray(resizedCrop),
//           let cropInputProvider = try? MLDictionaryFeatureProvider(dictionary: ["image": cropInputArray]),
//           let cropOutput = try? model.prediction(from: cropInputProvider) {
//            
//            // Find detections array
//            var cropDetArray: MLMultiArray?
//            if let arr = cropOutput.featureValue(for: "var_2421")?.multiArrayValue {
//                cropDetArray = arr
//            } else {
//                for name in cropOutput.featureNames {
//                    if let arr = cropOutput.featureValue(for: name)?.multiArrayValue {
//                        let shape = arr.shape.map { $0.intValue }
//                        if shape.count == 3 && shape[0] == 1 && shape[1] > 100 {
//                            cropDetArray = arr
//                            break
//                        }
//                    }
//                }
//            }
//            
//            if let detArray = cropDetArray,
//               let protoArray = cropOutput.featureValue(for: "p")?.multiArrayValue {
//                stage2Detections = extractDetections(from: detArray)
//                stage2Prototypes = protoArray
//                if self.debugMode { print("📊 Stage 2: \(stage2Detections.count) detections") }
//            }
//        } else {
//            if self.debugMode { print("⚠️ Stage 2: Failed to crop/process") }
//        }
//        
//        // UNION Stage 1 + Stage 2 masks
//        let stage1Kept = keepOverlappingDetections(stage1Detections)
//        let stage2Kept = stage2Prototypes != nil ? keepOverlappingDetections(stage2Detections) : []
//        
//        if self.debugMode {
//            print("\n📊 UNION SUMMARY:")
//            print("   Stage 1: keeping \(stage1Kept.count) overlapping detections")
//            print("   Stage 2: keeping \(stage2Kept.count) overlapping detections")
//        }
//        
//        if stage1Kept.isEmpty && stage2Kept.isEmpty {
//            DispatchQueue.main.async {
//                self.maskImageView.image = nil
//                self.isProcessing = false
//            }
//            return
//        }
//
//        // Generate combined cutout with BOTH stages
//        generateCutoutTwoStage(
//            stage1Detections: stage1Kept,
//            stage1Prototypes: prototypesArray,
//            stage2Detections: stage2Kept,
//            stage2Prototypes: stage2Prototypes,
//            primaryBBox: primary,
//            originalImage: pixelBuffer
//        )
//    }
    
    private func processFrame(_ pixelBuffer: CVPixelBuffer) {
        guard let model = mlModel else { return }
        let now = Date()
        guard now.timeIntervalSince(lastProcessTime) >= processInterval, !isProcessing else { return }
        lastProcessTime = now
        isProcessing = true

        if self.debugMode { print("\n🔬 ========== STAGE 1: FULL FRAME ==========") }

        // STAGE 1: Full frame detection
        // --- Accelerate (vImage) letterbox/resize inline to 640x640 BGRA ---
        guard let resized = letterbox(pixelBuffer, size: 640) else {
            isProcessing = false
            return
        }

        guard let inputArray = pixelBufferToMLMultiArray(resized) else {
            isProcessing = false
            return
        }

        guard let inputProvider = try? MLDictionaryFeatureProvider(dictionary: ["image": inputArray]) else {
            isProcessing = false
            return
        }

        guard let output = try? model.prediction(from: inputProvider) else {
            isProcessing = false
            return
        }

        // Log available output names (helps debug different models)
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
            // Try to find any MLMultiArray output that looks like detections
            for name in output.featureNames {
                if let arr = output.featureValue(for: name)?.multiArrayValue {
                    let shape = arr.shape.map { $0.intValue }
                    // Detection array has shape [1, features, predictions]
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

        let stage1Detections = extractDetections(from: detArray)
        if self.debugMode { print("📊 Stage 1: \(stage1Detections.count) detections") }

        if stage1Detections.isEmpty {
            DispatchQueue.main.async {
                self.maskImageView.image = nil
                self.isProcessing = false
            }
            return
        }

        // Get primary detection
        let sorted = stage1Detections.sorted { $0.confidence > $1.confidence }
        let primary = sorted.first!

        if self.debugMode {
            print("🎯 Primary: \(primary.className) @ \(Int(primary.confidence * 100))%")
            print("   BBox: center(\(Int(primary.x)), \(Int(primary.y))) size(\(Int(primary.width))x\(Int(primary.height)))")
        }

        // STAGE 2: Crop to primary bbox and re-detect
        if self.debugMode { print("\n🔬 ========== STAGE 2: CROPPED ==========") }

        var stage2Detections: [DetectionSmarty] = []
        var stage2Prototypes: MLMultiArray? = nil

        if let croppedBuffer = cropPixelBuffer(pixelBuffer, toBBox: primary, padding: 0.1),
           let resizedCrop = letterbox(croppedBuffer, size: 640),
           let cropInputArray = pixelBufferToMLMultiArray(resizedCrop),
           let cropInputProvider = try? MLDictionaryFeatureProvider(dictionary: ["image": cropInputArray]),
           let cropOutput = try? model.prediction(from: cropInputProvider) {

            // Find detections array
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
                stage2Detections = extractDetections(from: detArray)
                stage2Prototypes = protoArray
                if self.debugMode { print("📊 Stage 2: \(stage2Detections.count) detections") }
            }
        } else {
            if self.debugMode { print("⚠️ Stage 2: Failed to crop/process") }
        }

        // UNION Stage 1 + Stage 2 masks
        let stage1Kept = keepOverlappingDetections(stage1Detections)
        let stage2Kept = stage2Prototypes != nil ? keepOverlappingDetections(stage2Detections) : []

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

        // Generate combined cutout with BOTH stages
        generateCutoutTwoStage(
            stage1Detections: stage1Kept,
            stage1Prototypes: prototypesArray,
            stage2Detections: stage2Kept,
            stage2Prototypes: stage2Prototypes,
            primaryBBox: primary,
            originalImage: pixelBuffer
        )
    }

    
    private func letterbox(_ src: CVPixelBuffer, size: Int = 640) -> CVPixelBuffer? {
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

        return dst
    }


    // MARK: - Keep Overlapping Detections (NO NMS!)
    private func keepOverlappingDetections(_ detections: [DetectionSmarty]) -> [DetectionSmarty] {
        let sorted = detections.sorted { $0.confidence > $1.confidence }
        guard let primary = sorted.first else { return [] }
        
        var kept: [DetectionSmarty] = []
        
        for det in sorted {
            if bboxesOverlap(det, primary) {
                kept.append(det)
            }
        }
        
        return kept
    }
    
    // MARK: - BBox Overlap Check (any touch = true)
    private func bboxesOverlap(_ a: DetectionSmarty, _ b: DetectionSmarty) -> Bool {
        let aLeft = a.x - a.width / 2
        let aRight = a.x + a.width / 2
        let aTop = a.y - a.height / 2
        let aBottom = a.y + a.height / 2
        
        let bLeft = b.x - b.width / 2
        let bRight = b.x + b.width / 2
        let bTop = b.y - b.height / 2
        let bBottom = b.y + b.height / 2
        
        if aRight < bLeft || bRight < aLeft { return false }
        if aBottom < bTop || bBottom < aTop { return false }
        
        return true
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
        
        // Normalize raw values to 0-255 for grayscale
        var minVal: Float = 0, maxVal: Float = 0
        vDSP_minv(rawMask, 1, &minVal, vDSP_Length(rawMask.count))
        vDSP_maxv(rawMask, 1, &maxVal, vDSP_Length(rawMask.count))
        let range = maxVal - minVal
        
        var grayPixels = [UInt8](repeating: 0, count: width * height)
        for i in 0..<rawMask.count {
            let normalized = range > 0 ? (rawMask[i] - minVal) / range : 0.5
            grayPixels[i] = UInt8(max(0, min(255, normalized * 255)))
        }
        
        // Create grayscale image
        if let provider = CGDataProvider(data: Data(grayPixels) as CFData),
           let cgImage = CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 8,
                                  bytesPerRow: width, space: colorSpace,
                                  bitmapInfo: CGBitmapInfo(rawValue: 0),
                                  provider: provider, decode: nil, shouldInterpolate: false,
                                  intent: .defaultIntent) {
            let grayImage = UIImage(cgImage: cgImage)
            
            // Save to Photos
            PHPhotoLibrary.shared().performChanges({
                PHAssetChangeRequest.creationRequestForAsset(from: grayImage)
            }) { success, error in
                if success {
                    print("💾 Saved GRAYSCALE mask to Photos")
                } else {
                    print("❌ Failed to save grayscale: \(error?.localizedDescription ?? "unknown")")
                }
            }
        }
        
        // Create binary mask (threshold applied)
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
            
            // Save to Photos
            PHPhotoLibrary.shared().performChanges({
                PHAssetChangeRequest.creationRequestForAsset(from: binaryImage)
            }) { success, error in
                if success {
                    print("💾 Saved BINARY mask to Photos (threshold: \(self.maskThreshold))")
                } else {
                    print("❌ Failed to save binary: \(error?.localizedDescription ?? "unknown")")
                }
            }
        }
    }
    
//    private func generateCutoutTwoStage(
//        stage1Detections: [DetectionSmarty],
//        stage1Prototypes: MLMultiArray,
//        stage2Detections: [DetectionSmarty],
//        stage2Prototypes: MLMultiArray?,
//        primaryBBox: DetectionSmarty,
//        originalImage: CVPixelBuffer
//    ) {
//        let shape = stage1Prototypes.shape.map { $0.intValue }
//        let C = shape[1]
//        let Hp = shape[2]
//        let Wp = shape[3]
//        let spatial = Hp * Wp
//
//        if self.debugMode {
//            print("\n🎨 Generating TWO-STAGE UNION cutout")
//            print("   Stage 1: \(stage1Detections.count) detections")
//            print("   Stage 2: \(stage2Detections.count) detections")
//            print("📐 Prototype shape: C=\(C), H=\(Hp), W=\(Wp)")
//        }
//
//        // Build Stage 1 proto matrix
//        var protoMatrix1 = [Float](repeating: 0, count: C * spatial)
//        if stage1Prototypes.dataType == .float32 {
//            let srcBase = stage1Prototypes.dataPointer.assumingMemoryBound(to: Float.self)
//            memcpy(&protoMatrix1, srcBase, C * spatial * MemoryLayout<Float>.size)
//        } else {
//            for c in 0..<C {
//                for y in 0..<Hp {
//                    for x in 0..<Wp {
//                        let val = stage1Prototypes[[0, c, y, x] as [NSNumber]].floatValue
//                        protoMatrix1[c * spatial + (y * Wp + x)] = val
//                    }
//                }
//            }
//        }
//
//        // Global mask in FULL FRAME coordinates (Wp x Hp)
//        var globalMask = [Float](repeating: 0, count: spatial)
//
//        if self.debugMode { print("\n🔵 Processing Stage 1 masks (full frame)...") }
//
//        // Save primary detection's raw mask for analysis
//        var primaryRawMask: [Float]? = nil
//        var primaryDet: DetectionSmarty? = nil
//
//        // precompute scale from model input(640) to proto spatial
//        let protoScale = Float(Wp) / 640.0
//
//        for (detIndex, det) in stage1Detections.enumerated() {
//            var rawMask = [Float](repeating: 0, count: spatial)
//            vDSP_mmul(det.maskCoeffs, 1, protoMatrix1, 1, &rawMask, 1, 1, vDSP_Length(spatial), vDSP_Length(C))
//
//            // Apply sigmoid: logits -> probabilities (minimal, safe)
//            if true {
//                for i in 0..<spatial {
//                    rawMask[i] = 1.0 / (1.0 + exp(-rawMask[i]))
//                }
//            }
//
//            // Save primary detection's raw mask for analysis
//            if detIndex == 0 {
//                primaryRawMask = rawMask
//                primaryDet = det
//
//                var minVal: Float = 0, maxVal: Float = 0
//                vDSP_minv(rawMask, 1, &minVal, vDSP_Length(spatial))
//                vDSP_maxv(rawMask, 1, &maxVal, vDSP_Length(spatial))
//                var mean: Float = 0
//                vDSP_meanv(rawMask, 1, &mean, vDSP_Length(spatial))
//
//                print("\n📊 PRIMARY MASK RAW VALUES (\(det.className) @ \(Int(det.confidence*100))%):")
//                print("   Range: min=\(minVal), max=\(maxVal), mean=\(mean)")
//
//                var posCount = 0, negCount = 0, zeroCount = 0
//                for v in rawMask {
//                    if v > 0 { posCount += 1 }
//                    else if v < 0 { negCount += 1 }
//                    else { zeroCount += 1 }
//                }
//                print("   Distribution: \(posCount) positive, \(negCount) negative, \(zeroCount) zero")
//
//                print("   Mask coefficients (32): [\(det.maskCoeffs.map { String(format: "%.6f", $0) }.joined(separator: ", "))]")
//
//                // Convert detection coords if normalized (assume det.* in [0..1] if values <=1)
//                // Scale det coords to model 640-space
//                let dx = det.x <= 1.01 ? det.x * 640.0 : det.x
//                let dy = det.y <= 1.01 ? det.y * 640.0 : det.y
//                let dw = det.width <= 1.01 ? det.width * 640.0 : det.width
//                let dh = det.height <= 1.01 ? det.height * 640.0 : det.height
//
//                let mx1 = max(0, Int((dx - dw / 2.0) * protoScale))
//                let my1 = max(0, Int((dy - dh / 2.0) * protoScale))
//                let mx2 = min(Wp, Int((dx + dw / 2.0) * protoScale))
//                let my2 = min(Hp, Int((dy + dh / 2.0) * protoScale))
//
//                print("   BBox in mask coords: (\(mx1),\(my1)) → (\(mx2),\(my2))")
//
//                let centerX = (mx1 + mx2) / 2
//                let centerY = (my1 + my2) / 2
//
//                print("   Sample raw values (9-point grid):")
//                print("     ┌───────────────────────────────────────────────┐")
//                print("     │ TL(\(mx1),\(my1)): \(String(format: "%+.3f", rawMask[my1 * Wp + mx1]))  TC(\(centerX),\(my1)): \(String(format: "%+.3f", rawMask[my1 * Wp + centerX]))  TR(\(mx2-1),\(my1)): \(String(format: "%+.3f", rawMask[my1 * Wp + mx2-1]))")
//                print("     │ ML(\(mx1),\(centerY)): \(String(format: "%+.3f", rawMask[centerY * Wp + mx1]))  CC(\(centerX),\(centerY)): \(String(format: "%+.3f", rawMask[centerY * Wp + centerX]))  MR(\(mx2-1),\(centerY)): \(String(format: "%+.3f", rawMask[centerY * Wp + mx2-1]))")
//                print("     │ BL(\(mx1),\(my2-1)): \(String(format: "%+.3f", rawMask[(my2-1) * Wp + mx1]))  BC(\(centerX),\(my2-1)): \(String(format: "%+.3f", rawMask[(my2-1) * Wp + centerX]))  BR(\(mx2-1),\(my2-1)): \(String(format: "%+.3f", rawMask[(my2-1) * Wp + mx2-1]))")
//                print("     └───────────────────────────────────────────────┘")
//
//                // 10x10 sampling
//                let bboxW = max(1, mx2 - mx1)
//                let bboxH = max(1, my2 - my1)
//                print("\n   📊 RAW MASK VALUES (10x10 grid sampling bbox region):")
//                var header = "        "
//                for gridX in 0..<10 {
//                    let px = mx1 + (gridX * bboxW / 10)
//                    header += String(format: "  x%-3d ", px)
//                }
//                print(header)
//
//                for gridY in 0..<10 {
//                    let py = my1 + (gridY * bboxH / 10)
//                    var rowStr = String(format: "   y%-3d ", py)
//                    for gridX in 0..<10 {
//                        let px = mx1 + (gridX * bboxW / 10)
//                        let val = rawMask[py * Wp + px]
//                        rowStr += String(format: "%+.2f ", val)
//                    }
//                    print(rowStr)
//                }
//
//                var posInBbox = 0, negInBbox = 0
//                var aboveThreshInBbox = 0
//                for py in my1..<my2 {
//                    for px in mx1..<mx2 {
//                        let v = rawMask[py * Wp + px]
//                        if v > 0 { posInBbox += 1 }
//                        else if v < 0 { negInBbox += 1 }
//                        if v > self.maskThreshold { aboveThreshInBbox += 1 }
//                    }
//                }
//                let bboxSize = max(1, (mx2 - mx1) * (my2 - my1))
//                print("   Inside bbox: \(posInBbox)/\(bboxSize) positive, \(negInBbox) negative")
//                print("   🎚️ At threshold \(self.maskThreshold): \(aboveThreshInBbox)/\(bboxSize) pixels (\(String(format: "%.1f", Float(aboveThreshInBbox)/Float(bboxSize)*100))%)")
//            }
//
//            // Compute bbox in proto coords — handle possibles normalized coords
//            let dx = det.x <= 1.01 ? det.x * 640.0 : det.x
//            let dy = det.y <= 1.01 ? det.y * 640.0 : det.y
//            let dw = det.width <= 1.01 ? det.width * 640.0 : det.width
//            let dh = det.height <= 1.01 ? det.height * 640.0 : det.height
//
//            let mx1 = max(0, Int((dx - dw / 2.0) * protoScale))
//            let my1 = max(0, Int((dy - dh / 2.0) * protoScale))
//            let mx2 = min(Wp, Int((dx + dw / 2.0) * protoScale))
//            let my2 = min(Hp, Int((dy + dh / 2.0) * protoScale))
//
//            // Apply threshold and merge into globalMask (probabilities now)
//            var addedPixels = 0
//            for py in my1..<my2 {
//                let rowStart = py * Wp + mx1
//                let rowLen = max(0, mx2 - mx1)
//                for i in 0..<rowLen {
//                    let idx = rowStart + i
//                    if rawMask[idx] > maskThreshold && globalMask[idx] == 0 {
//                        globalMask[idx] = 1.0
//                        addedPixels += 1
//                    }
//                }
//            }
//
//            if self.debugMode && detIndex < 5 {
//                print("   ✅ S1 \(det.className) @ \(Int(det.confidence*100))%: bbox(\(mx1),\(my1))→(\(mx2),\(my2)), +\(addedPixels)px")
//            }
//        }
//
//        if stage1Detections.count > 5 {
//            print("   ... and \(stage1Detections.count - 5) more detections")
//        }
//
//        if self.debugMode {
//            print("   ⚙️ Mask threshold: \(maskThreshold)")
//        }
//
//        if self.debugMode, let rawMask = primaryRawMask, let det = primaryDet {
//            saveMaskToFile(rawMask: rawMask, width: Wp, height: Hp, detection: det)
//        }
//
//        var stage1PixelCount = 0
//        for i in 0..<spatial { if globalMask[i] > 0 { stage1PixelCount += 1 } }
//        if self.debugMode {
//            print("   📊 After Stage 1: \(stage1PixelCount)/\(spatial) pixels (\(String(format: "%.1f", Float(stage1PixelCount)/Float(spatial)*100))%)")
//        }
//
//        // ========== STAGE 2 MASKS (mapped back to full frame) ==========
//        if let proto2 = stage2Prototypes, !stage2Detections.isEmpty {
//            if self.debugMode { print("\n🟢 Processing Stage 2 masks (cropped → full frame)...") }
//
//            // Build Stage 2 proto matrix
//            var protoMatrix2 = [Float](repeating: 0, count: C * spatial)
//            if proto2.dataType == .float32 {
//                let srcBase = proto2.dataPointer.assumingMemoryBound(to: Float.self)
//                memcpy(&protoMatrix2, srcBase, C * spatial * MemoryLayout<Float>.size)
//            } else {
//                for c in 0..<C {
//                    for y in 0..<Hp {
//                        for x in 0..<Wp {
//                            let val = proto2[[0, c, y, x] as [NSNumber]].floatValue
//                            protoMatrix2[c * spatial + (y * Wp + x)] = val
//                        }
//                    }
//                }
//            }
//
//            // Calculate crop region in full frame (model coords 640x640)
//            let padding: Float = 0.1
//            let cropX1 = max(0, primaryBBox.x - primaryBBox.width / 2 * (1 + padding))
//            let cropY1 = max(0, primaryBBox.y - primaryBBox.height / 2 * (1 + padding))
//            let cropX2 = min(640, primaryBBox.x + primaryBBox.width / 2 * (1 + padding))
//            let cropY2 = min(640, primaryBBox.y + primaryBBox.height / 2 * (1 + padding))
//            let cropW = cropX2 - cropX1
//            let cropH = cropY2 - cropY1
//
//            if self.debugMode {
//                print("   Crop region (model): (\(Int(cropX1)),\(Int(cropY1)))→(\(Int(cropX2)),\(Int(cropY2))) = \(Int(cropW))x\(Int(cropH))")
//            }
//
//            for det in stage2Detections {
//                var rawMask = [Float](repeating: 0, count: spatial)
//                vDSP_mmul(det.maskCoeffs, 1, protoMatrix2, 1, &rawMask, 1, 1, vDSP_Length(spatial), vDSP_Length(C))
//
//                // sigmoid on stage2 raw mask
//                for i in 0..<spatial { rawMask[i] = 1.0 / (1.0 + exp(-rawMask[i])) }
//
//                // Stage 2 bbox in cropped 640x640 space — convert if normalized
//                let dx_c = det.x <= 1.01 ? det.x * 640.0 : det.x
//                let dy_c = det.y <= 1.01 ? det.y * 640.0 : det.y
//                let dw_c = det.width <= 1.01 ? det.width * 640.0 : det.width
//                let dh_c = det.height <= 1.01 ? det.height * 640.0 : det.height
//
//                let mx1_crop = max(0, Int((dx_c - dw_c / 2) * protoScale))
//                let my1_crop = max(0, Int((dy_c - dh_c / 2) * protoScale))
//                let mx2_crop = min(Wp, Int((dx_c + dw_c / 2) * protoScale))
//                let my2_crop = min(Hp, Int((dy_c + dh_c / 2) * protoScale))
//
//                var addedPixels = 0
//
//                // Map Stage 2 mask pixels to full frame mask
//                for py_crop in 0..<Hp {
//                    for px_crop in 0..<Wp {
//                        let cropIdx = py_crop * Wp + px_crop
//
//                        // Only process inside bbox
//                        if px_crop >= mx1_crop && px_crop < mx2_crop && py_crop >= my1_crop && py_crop < my2_crop {
//                            if rawMask[cropIdx] > maskThreshold {
//                                // Map cropped mask coords to full frame model coords
//                                let fracX = Float(px_crop) / Float(Wp)
//                                let fracY = Float(py_crop) / Float(Hp)
//
//                                // Position in full frame model coords (640x640)
//                                let fullX = cropX1 + fracX * cropW
//                                let fullY = cropY1 + fracY * cropH
//
//                                // Convert to mask coords (Wp x Hp)
//                                let mx_full = Int(fullX * protoScale)
//                                let my_full = Int(fullY * protoScale)
//
//                                if mx_full >= 0 && mx_full < Wp && my_full >= 0 && my_full < Hp {
//                                    let fullIdx = my_full * Wp + mx_full
//                                    if globalMask[fullIdx] == 0 {
//                                        addedPixels += 1
//                                    }
//                                    globalMask[fullIdx] = 1.0
//                                }
//                            }
//                        }
//                    }
//                }
//
//                if self.debugMode {
//                    print("   ✅ S2 \(det.className) @ \(Int(det.confidence*100))%: bbox(\(mx1_crop),\(my1_crop))→(\(mx2_crop),\(my2_crop)), +\(addedPixels)px NEW")
//                }
//            }
//        }
//
//        // Final count
//        var finalPixelCount = 0
//        for i in 0..<spatial { if globalMask[i] > 0 { finalPixelCount += 1 } }
//        let addedByStage2 = finalPixelCount - stage1PixelCount
//
//        if self.debugMode {
//            print("\n📊 MERGED MASK: \(finalPixelCount)/\(spatial) pixels (\(String(format: "%.1f", Float(finalPixelCount)/Float(spatial)*100))%)")
//            print("   Stage 1 contributed: \(stage1PixelCount) pixels")
//            print("   Stage 2 added: \(addedByStage2) NEW pixels")
//        }
//
//        // Convert to UInt8 binary
//        var binaryMask = [UInt8](repeating: 0, count: spatial)
//        for i in 0..<spatial { if globalMask[i] > 0 { binaryMask[i] = 255 } }
//
//        if self.debugMode {
//            print20x20BinaryGrid("MERGED STAGE1+STAGE2", mask: binaryMask, width: Wp, height: Hp)
//        }
//
//        // Render to image
//        autoreleasepool {
//            let ciImage = CIImage(cvPixelBuffer: originalImage)
//            let width = CVPixelBufferGetWidth(originalImage)
//            let height = CVPixelBufferGetHeight(originalImage)
//
//            guard let cgImage = ciContext.createCGImage(ciImage, from: ciImage.extent) else {
//                if self.debugMode { print("❌ Failed to create CGImage") }
//                DispatchQueue.main.async { self.isProcessing = false }
//                return
//            }
//
//            guard let ctx = CGContext(
//                data: nil,
//                width: width,
//                height: height,
//                bitsPerComponent: 8,
//                bytesPerRow: width * 4,
//                space: CGColorSpaceCreateDeviceRGB(),
//                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
//            ) else {
//                if self.debugMode { print("❌ Failed to create CGContext") }
//                DispatchQueue.main.async { self.isProcessing = false }
//                return
//            }
//
//            guard let data = ctx.data else {
//                if self.debugMode { print("❌ CGContext has no data") }
//                DispatchQueue.main.async { self.isProcessing = false }
//                return
//            }
//
//            ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
//            let pixels = data.bindMemory(to: UInt8.self, capacity: width * height * 4)
//
//            let scaleX = Float(Wp) / Float(width)
//            let scaleY = Float(Hp) / Float(height)
//
//            if self.debugMode {
//                print("🖼️ Upscaling \(Wp)×\(Hp) → \(width)×\(height)")
//            }
//
//            var opaqueCount = 0
//
//            // Precompute X mapping table
//            var xMap = [Int](repeating: 0, count: width)
//            for px in 0..<width {
//                xMap[px] = min(Int(Float(px) * scaleX), Wp - 1)
//            }
//
//            // Process rows
//            for py in 0..<height {
//                let my = min(Int(Float(py) * scaleY), Hp - 1)
//                let rowOffset = py * width * 4
//                let maskRowOffset = my * Wp
//
//                for px in 0..<width {
//                    let idx = rowOffset + px * 4
//                    let maskIdx = maskRowOffset + xMap[px]
//
//                    if globalMask[maskIdx] > 0 {
//                        pixels[idx + 3] = 255
//                        opaqueCount += 1
//                    } else {
//                        pixels[idx + 0] = 0
//                        pixels[idx + 1] = 0
//                        pixels[idx + 2] = 0
//                        pixels[idx + 3] = 0
//                    }
//                }
//            }
//
//            if self.debugMode {
//                print("📊 Output: \(opaqueCount)/\(width * height) opaque (\(String(format: "%.1f", Float(opaqueCount)/Float(width*height)*100))%)")
//            }
//
//            // Draw bounding box for primary
//            if !stage1Detections.isEmpty {
//                drawBoundingBox(ctx: ctx, detection: stage1Detections[0], imageWidth: width, imageHeight: height)
//            }
//
//            if let outImage = ctx.makeImage() {
//                DispatchQueue.main.async {
//                    self.maskImageView.image = UIImage(cgImage: outImage, scale: 1.0, orientation: .up)
//                    self.isProcessing = false
//                    if self.debugMode { print("✅ ==================== FRAME COMPLETE ====================\n") }
//                }
//            } else {
//                if self.debugMode { print("❌ Failed to make output image") }
//                DispatchQueue.main.async { self.isProcessing = false }
//            }
//        }
//    }
    
    private func generateCutoutTwoStage(
        stage1Detections: [DetectionSmarty],
        stage1Prototypes: MLMultiArray,
        stage2Detections: [DetectionSmarty],
        stage2Prototypes: MLMultiArray?,
        primaryBBox: DetectionSmarty,
        originalImage: CVPixelBuffer
    ) {
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

        // Build Stage 1 proto matrix
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

        // Global mask in FULL FRAME coordinates (Wp x Hp)
        var globalMask = [Float](repeating: 0, count: spatial)

        // ========== STAGE 1 MASKS ==========
        if self.debugMode { print("\n🔵 Processing Stage 1 masks (full frame)...") }

        var primaryRawMask: [Float]? = nil
        var primaryDet: DetectionSmarty? = nil
        var stage1PixelCount = 0

        for (detIndex, det) in stage1Detections.enumerated() {
            var rawMask = [Float](repeating: 0, count: spatial)
            vDSP_mmul(det.maskCoeffs, 1, protoMatrix1, 1, &rawMask, 1, 1, vDSP_Length(spatial), vDSP_Length(C))

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
                // Sampling logs omitted for brevity
            }

            // Compute bbox in proto coords
            let scale = Float(Wp) / 640.0
            let mx1 = max(0, Int((det.x - det.width / 2) * scale))
            let my1 = max(0, Int((det.y - det.height / 2) * scale))
            let mx2 = min(Wp, Int((det.x + det.width / 2) * scale))
            let my2 = min(Hp, Int((det.y + det.height / 2) * scale))

            // Optimized: pointer-based per-row loop (same logic)
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

        // Count after Stage 1
        for i in 0..<spatial { if globalMask[i] > 0 { stage1PixelCount += 1 } }
        if self.debugMode {
            print("   ⚙️ Mask threshold: \(maskThreshold)")
            print("   📊 After Stage 1: \(stage1PixelCount)/\(spatial) pixels (\(String(format: "%.1f", Float(stage1PixelCount)/Float(spatial)*100))%)")
                    
        }
        if self.debugMode, let rawMask = primaryRawMask, let det = primaryDet {
            saveMaskToFile(rawMask: rawMask, width: Wp, height: Hp, detection: det)
        }

        // ========== STAGE 2 MASKS (mapped back to full frame) ==========
        if let proto2 = stage2Prototypes, !stage2Detections.isEmpty {
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

            for det in stage2Detections {
                var rawMask = [Float](repeating: 0, count: spatial)
                vDSP_mmul(det.maskCoeffs, 1, protoMatrix2, 1, &rawMask, 1, 1, vDSP_Length(spatial), vDSP_Length(C))

                let mx1_crop = max(0, Int((det.x - det.width / 2) * scale))
                let my1_crop = max(0, Int((det.y - det.height / 2) * scale))
                let mx2_crop = min(Wp, Int((det.x + det.width / 2) * scale))
                let my2_crop = min(Hp, Int((det.y + det.height / 2) * scale))

                var addedPixels = 0

                // Optimized: iterate only bbox region with pointer access
                rawMask.withUnsafeBufferPointer { rPtr in
                    globalMask.withUnsafeMutableBufferPointer { gPtr in
                        if mx2_crop > mx1_crop && my2_crop > my1_crop {
                            for py_crop in my1_crop..<my2_crop {
                                let base = py_crop * Wp
                                for px_crop in mx1_crop..<mx2_crop {
                                    let cropIdx = base + px_crop
                                    if rPtr[cropIdx] > 0 {
                                        // Map cropped mask coords to full frame model coords
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
        }

        // Final count
        var finalPixelCount = 0
        for i in 0..<spatial { if globalMask[i] > 0 { finalPixelCount += 1 } }
        let addedByStage2 = finalPixelCount - stage1PixelCount

        if self.debugMode {
            print("\n📊 MERGED MASK: \(finalPixelCount)/\(spatial) pixels (\(String(format: "%.1f", Float(finalPixelCount)/Float(spatial)*100))%)")
            print("   Stage 1 contributed: \(stage1PixelCount) pixels")
            print("   Stage 2 added: \(addedByStage2) NEW pixels")
        }

        // Convert to UInt8 binary
        var binaryMask = [UInt8](repeating: 0, count: spatial)
        for i in 0..<spatial {
            if globalMask[i] > 0 {
                binaryMask[i] = 255
            }
        }

        if self.debugMode {
            print20x20BinaryGrid("MERGED STAGE1+STAGE2", mask: binaryMask, width: Wp, height: Hp)
        }

        // Render to image (unchanged logic; optimized writes below)
        autoreleasepool {
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

            guard let data = ctx.data else {
                if self.debugMode { print("❌ CGContext has no data") }
                DispatchQueue.main.async { self.isProcessing = false }
                return
            }

            ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
            let pixels = data.bindMemory(to: UInt8.self, capacity: width * height * 4)

            let scaleX = Float(Wp) / Float(width)
            let scaleY = Float(Hp) / Float(height)

            if self.debugMode {
                print("🖼️ Upscaling \(Wp)×\(Hp) → \(width)×\(height)")
            }

            var opaqueCount = 0

            // Precompute X mapping table
            var xMap = [Int](repeating: 0, count: width)
            for px in 0..<width {
                xMap[px] = min(Int(Float(px) * scaleX), Wp - 1)
            }

            // Optimized: pointer-advanced inner loop for pixel writes (same output)
            for py in 0..<height {
                let my = min(Int(Float(py) * scaleY), Hp - 1)
                let maskRowOffset = my * Wp
                var pixelPtr = pixels.advanced(by: py * width * 4)
                for px in 0..<width {
                    let maskIdx = maskRowOffset + xMap[px]
                    if globalMask[maskIdx] > 0 {
                        pixelPtr[3] = 255
                        opaqueCount += 1
                    } else {
                        pixelPtr[0] = 0
                        pixelPtr[1] = 0
                        pixelPtr[2] = 0
                        pixelPtr[3] = 0
                    }
                    pixelPtr = pixelPtr.advanced(by: 4)
                }
            }

            if self.debugMode {
                print("📊 Output: \(opaqueCount)/\(width * height) opaque (\(String(format: "%.1f", Float(opaqueCount)/Float(width*height)*100))%)")
                            
            }

            if !stage1Detections.isEmpty {
                drawBoundingBox(ctx: ctx, detection: stage1Detections[0], imageWidth: width, imageHeight: height)
            }

            if let outImage = ctx.makeImage() {
                DispatchQueue.main.async {
                    self.maskImageView.image = UIImage(cgImage: outImage, scale: 1.0, orientation: .up)
                    self.isProcessing = false
                    if self.debugMode {
                        print("✅ ==================== FRAME COMPLETE ====================\n")
                    }
                }
            } else {
                if self.debugMode { print("❌ Failed to make output image") }
                DispatchQueue.main.async { self.isProcessing = false }
            }
        }
    }


    
    // MARK: - Draw Bounding Box
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
        
        let font = CTFontCreateWithName("Helvetica-Bold" as CFString, 28, nil)
        let attributes: [NSAttributedString.Key: Any] = [
            .font: font,
            .foregroundColor: UIColor.white
        ]
        
        let attributedString = NSAttributedString(string: labelText, attributes: attributes)
        let framesetter = CTFramesetterCreateWithAttributedString(attributedString)
        let textSize = CTFramesetterSuggestFrameSizeWithConstraints(
            framesetter,
            CFRange(location: 0, length: 0),
            nil,
            CGSize(width: 220, height: CGFloat.greatestFiniteMagnitude),
            nil
        )
        
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
        let path = CGPath(rect: CGRect(x: textX, y: textY, width: labelWidth - 2 * labelPadding, height: labelHeight - 2 * labelPadding), transform: nil)
        let frame = CTFramesetterCreateFrame(framesetter, CFRange(location: 0, length: 0), path, nil)
        
        ctx.saveGState()
        ctx.textMatrix = .identity
        CTFrameDraw(frame, ctx)
        ctx.restoreGState()
    }

    // MARK: - Extract Detections
    private func extractDetections(from detections: MLMultiArray) -> [DetectionSmarty] {
        var all: [DetectionSmarty] = []
        
        // Get tensor dimensions
        let numFeatures = detections.shape[1].intValue
        let numAnchors = detections.shape[2].intValue
        
        // Auto-detect model type from feature count
        // YOLO11-seg: 116 features = 4 bbox + 80 classes + 32 coeffs
        // YOLOE-pf:  4621 features = 4 bbox + 4585 classes + 32 coeffs
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
        
        if detections.dataType == .float16 {
            let src = detections.dataPointer.bindMemory(to: UInt16.self, capacity: totalCount)
            var srcBuf = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: src), height: 1, width: vImagePixelCount(totalCount), rowBytes: totalCount * 2)
            var dstBuf = vImage_Buffer(data: UnsafeMutableRawPointer(detBuf), height: 1, width: vImagePixelCount(totalCount), rowBytes: totalCount * 4)
            vImageConvert_Planar16FtoPlanarF(&srcBuf, &dstBuf, vImage_Flags(kvImageNoFlags))
        } else if detections.dataType == .float32 {
            let src = detections.dataPointer.assumingMemoryBound(to: Float.self)
            memcpy(detBuf, src, totalCount * MemoryLayout<Float>.size)
        } else {
            for i in 0..<totalCount {
                detBuf[i] = detections[i].floatValue
            }
        }
        
        // Coefficient start: after bbox (4) + classes (numClasses)
        let coeffOffset = 4 + numClasses  // Feature index where mask coeffs start
        
        for anchor in 0..<numAnchors {
            let x = detBuf[0 * numAnchors + anchor]
            let y = detBuf[1 * numAnchors + anchor]
            let w = detBuf[2 * numAnchors + anchor]
            let h = detBuf[3 * numAnchors + anchor]
            
            if detectAllObjects {
                // Find the class with highest confidence for this anchor
                var bestConf: Float = 0
                var bestClassIdx = -1
                
                for classIdx in 0..<numClasses {
                    let confIdx = (4 + classIdx) * numAnchors + anchor
                    let conf = detBuf[confIdx]
                    if conf > bestConf {
                        bestConf = conf
                        bestClassIdx = classIdx
                    }
                }
                
                if bestConf > confidenceThreshold && bestClassIdx >= 0 {
                    // Use furniture name if known, otherwise use class index
                    let className = furnitureClasses[bestClassIdx] ?? "object_\(bestClassIdx)"
                    
                    var coeffs = [Float](repeating: 0, count: 32)
                    let coeffStart = coeffOffset * numAnchors + anchor
                    for i in 0..<32 {
                        coeffs[i] = detBuf[coeffStart + i * numAnchors]
                    }
                    all.append(DetectionSmarty(
                        x: x, y: y, width: w, height: h,
                        confidence: bestConf, classIdx: bestClassIdx, className: className,
                        maskCoeffs: coeffs
                    ))
                }
            } else {
                // Check only furniture classes (LVIS indices)
                for (classIdx, className) in furnitureClasses {
                    // Skip if class index is out of bounds for this model
                    guard classIdx < numClasses else { continue }
                    
                    let confIdx = (4 + classIdx) * numAnchors + anchor
                    let conf = detBuf[confIdx]
                    
                    if conf > confidenceThreshold {
                        var coeffs = [Float](repeating: 0, count: 32)
                        let coeffStart = coeffOffset * numAnchors + anchor
                        for i in 0..<32 {
                            coeffs[i] = detBuf[coeffStart + i * numAnchors]
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
        
        // Log summary (only in debug mode)
        if self.debugMode {
            let grouped = Dictionary(grouping: all) { $0.className }
            print("\n📊 DETECTION SUMMARY: \(all.count) total")
            for (className, dets) in grouped.sorted(by: { $0.value.count > $1.value.count }).prefix(20) {
                let confidences = dets.map { Int($0.confidence * 100) }
                print("  - \(className): \(dets.count)x, conf: \(confidences)%")
            }
            if grouped.count > 20 {
                print("  ... and \(grouped.count - 20) more classes")
            }
        }
        
        return all
    }

    // MARK: - Pixel Buffer to MLMultiArray (Accelerate) — fixed indices (vDSP_Length)
    private func pixelBufferToMLMultiArray(_ pixelBuffer: CVPixelBuffer) -> MLMultiArray? {
        guard let array = try? MLMultiArray(shape: [1, 3, 640, 640], dataType: .float32) else { return nil }
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else { return nil }

        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let width = 640
        let height = 640
        let pixelCount = width * height
        let src = baseAddress.assumingMemoryBound(to: UInt8.self)

        // Destination plane pointers
        let floatSize = MemoryLayout<Float32>.size
        let planeStrideBytes = pixelCount * floatSize
        let rPtr = array.dataPointer.advanced(by: 0 * planeStrideBytes).assumingMemoryBound(to: Float32.self)
        let gPtr = array.dataPointer.advanced(by: 1 * planeStrideBytes).assumingMemoryBound(to: Float32.self)
        let bPtr = array.dataPointer.advanced(by: 2 * planeStrideBytes).assumingMemoryBound(to: Float32.self)

        // Precompute index arrays (vDSP_Length) for gathers: offsets 2,1,0 with stride 4
        var indicesR = [vDSP_Length](repeating: 0, count: width)
        var indicesG = [vDSP_Length](repeating: 0, count: width)
        var indicesB = [vDSP_Length](repeating: 0, count: width)
        for i in 0..<width {
            indicesR[i] = vDSP_Length(2 + i * 4)
            indicesG[i] = vDSP_Length(1 + i * 4)
            indicesB[i] = vDSP_Length(0 + i * 4)
        }

        // Per-row temporary buffers
        var rowUInt8 = [UInt8](repeating: 0, count: width * 4)
        var rowFloat = [Float](repeating: 0, count: width * 4)

        // scale constant
        var scaleF: Float = 1.0 / 255.0

        for y in 0..<height {
            // copy row bytes into contiguous small buffer
            let rowStart = src.advanced(by: y * bytesPerRow)
            memcpy(&rowUInt8, rowStart, width * 4)

            // convert UInt8 -> Float (0..1) using vDSP (Float variants)
            rowUInt8.withUnsafeBufferPointer { u8Ptr in
                rowFloat.withUnsafeMutableBufferPointer { fPtr in
                    vDSP_vfltu8(u8Ptr.baseAddress!, 1, fPtr.baseAddress!, 1, vDSP_Length(width * 4))
                    vDSP_vsmul(fPtr.baseAddress!, 1, &scaleF, fPtr.baseAddress!, 1, vDSP_Length(width * 4))
                }
            }

            // Deinterleave rowFloat ([B,G,R,A,...]) into planar R/G/B arrays using vDSP_vgathr
            rowFloat.withUnsafeBufferPointer { rf in
                let baseF = rf.baseAddress!

                // gather R channel
                vDSP_vgathr(baseF, indicesR, 1, rPtr.advanced(by: y * width), 1, vDSP_Length(width))
                // gather G channel
                vDSP_vgathr(baseF, indicesG, 1, gPtr.advanced(by: y * width), 1, vDSP_Length(width))
                // gather B channel
                vDSP_vgathr(baseF, indicesB, 1, bPtr.advanced(by: y * width), 1, vDSP_Length(width))
            }
        }

        return array
    }



}
