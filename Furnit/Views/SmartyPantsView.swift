// SmartyPantsView.swift
// UIKit structure + Doc1 YOLO reading + Doc1 Accelerate + ALL DEBUG LOGS
import SwiftUI
import UIKit
import CoreML
import Accelerate
import AVFoundation
import Photos

private let SEGMENT_DEBUG_SAVE_IMAGES = true

// MARK: - SwiftUI Wrapper
struct SmartyPantsViewSwiftUI: UIViewRepresentable {
    let mlModel: MLModel?
    var processInterval: TimeInterval = 0.05
    var confidenceThreshold: Float = 0.3
    var active: Bool = false
    var debugSaveImages: Bool = true

    func makeUIView(context: Context) -> SmartyPantsContainerView {
        let v = SmartyPantsContainerView()
        v.processInterval = processInterval
        v.confidenceThreshold = confidenceThreshold
        v.debugSaveImages = debugSaveImages
        v.setModel(mlModel)
        if active { v.startIfNeeded() }
        return v
    }

    func updateUIView(_ uiView: SmartyPantsContainerView, context: Context) {
        uiView.setModel(mlModel)
        uiView.processInterval = processInterval
        uiView.confidenceThreshold = confidenceThreshold
        uiView.debugSaveImages = debugSaveImages
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
    var tightBBox: CGRect? = nil
}

// MARK: - Main Container View
final class SmartyPantsContainerView: UIView, AVCaptureVideoDataOutputSampleBufferDelegate {
    
    // MARK: Config
    var processInterval: TimeInterval = 0.05
    var confidenceThreshold: Float = 0.3
    var debugSaveImages: Bool = true
    var maskCutoff: Float = 0.3

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
        iv.clipsToBounds = false
        iv.alpha = 1.0
        iv.isUserInteractionEnabled = true
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

    // MARK: Furniture Classes (from Doc1)
    private let furnitureClasses: [Int: String] = [
        132: "armchair", 213: "baby seat", 276: "bar stool", 332: "bathroom cabinet",
        334: "bathroom mirror", 352: "beach chair", 364: "bean bag chair", 375: "bed",
        376: "bedcover", 377: "bed frame", 382: "bedside lamp", 402: "bench",
        429: "billiard table", 517: "bookshelf", 567: "chest", 632: "bunk bed",
        636: "bureau", 670: "cabinet", 679: "cake stand", 714: "canopy bed",
        733: "car seat", 821: "chair", 823: "daybed", 834: "changing table",
        977: "closet", 996: "coatrack", 1006: "cocktail table", 1060: "computer chair",
        1061: "computer desk", 1137: "infant bed", 1141: "couch", 1143: "counter",
        1144: "counter top", 1270: "day bed", 1301: "table", 1302: "table lamp",
        1303: "desktop", 1325: "dinning table", 1364: "dog bed", 1396: "drawer",
        1405: "dresser", 1476: "electric chair", 1503: "side table", 1602: "feeding chair",
        1624: "file cabinet", 1721: "folding chair", 1733: "food stand", 1750: "footrest",
        1801: "fruit stand", 1816: "futon", 1885: "glass table", 2141: "hospital bed",
        2193: "ice shelf", 2219: "information desk", 2247: "island", 2318: "kitchen cabinet",
        2319: "kitchen counter", 2322: "kitchen island", 2324: "kitchen table",
        2499: "loveseat", 2599: "mattress", 2614: "medicine cabinet", 2654: "mirror",
        2754: "music stool", 2802: "nightstand", 2834: "office chair", 2836: "office desk",
        2939: "park bench", 3024: "church bench", 3045: "picnic table",
        3061: "table tennis table", 3145: "poker table", 3423: "rocking chair",
        3449: "round table", 3584: "seat", 3621: "shelf", 3678: "side cabinet",
        3812: "spice rack", 3862: "stand", 3888: "step stool", 3909: "stool",
        4004: "supermarket shelf", 4041: "swivel chair", 4055: "table top",
        4056: "tablecloth", 4179: "toilet seat", 4213: "towel bar", 4294: "tv cabinet",
        4331: "vanity", 4473: "wheelchair", 4506: "window seat", 4513: "wine cabinet",
        4516: "wine rack", 4545: "workbench", 4564: "writing desk"
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
        
        // Add pinch gesture to self (parent view) so it works even when image is small
        let pinchGesture = UIPinchGestureRecognizer(target: self, action: #selector(handlePinch(_:)))
        self.addGestureRecognizer(pinchGesture)
        
        setupCamera()
        if SEGMENT_DEBUG_SAVE_IMAGES { print("✅ SmartyPantsContainerView initialized") }
    }
    
    override func layoutSubviews() {
        super.layoutSubviews()
        previewLayer.frame = bounds
    }
    
    // Pass touches in top area through to SwiftUI (for Back button)
    // Handle rest for pinch gesture
    override func point(inside point: CGPoint, with event: UIEvent?) -> Bool {
        // Top 100 points - pass through for navigation
        if point.y < 100 {
            return false
        }
        // If no image, pass through
        if maskImageView.image == nil {
            return false
        }
        return true
    }
    
    @objc private func handlePinch(_ gesture: UIPinchGestureRecognizer) {
        if SEGMENT_DEBUG_SAVE_IMAGES { print("📌 Pinch gesture: state=\(gesture.state.rawValue), scale=\(gesture.scale)") }
        switch gesture.state {
        case .changed:
            let newScale = currentScale * gesture.scale
            let clampedScale = min(max(newScale, 0.3), 3.0)
            maskImageView.transform = CGAffineTransform(scaleX: clampedScale, y: clampedScale)
            gesture.scale = 1.0
        case .ended:
            currentScale = min(max(currentScale * gesture.scale, 0.3), 3.0)
            if SEGMENT_DEBUG_SAVE_IMAGES { print("📌 Pinch ended, currentScale=\(currentScale)") }
        default:
            break
        }
    }

    // MARK: - Public
    func setModel(_ model: MLModel?) {
        detectionQueue.sync {
            self.mlModel = model
            if model != nil {
                if SEGMENT_DEBUG_SAVE_IMAGES { print("✅ Model set successfully") }
            } else {
                if SEGMENT_DEBUG_SAVE_IMAGES { print("⚠️ Model is nil") }
            }
        }
    }
    
    func startIfNeeded() {
        if SEGMENT_DEBUG_SAVE_IMAGES { print("🎬 startIfNeeded called") }
        requestCameraPermissionAndStart()
    }
    func stop() {
        if SEGMENT_DEBUG_SAVE_IMAGES { print("🛑 stop called") }
        stopCamera()
    }

    // MARK: - Camera Setup
    private func setupCamera() {
        if SEGMENT_DEBUG_SAVE_IMAGES { print("📷 Setting up camera...") }
        captureSession.beginConfiguration()
        captureSession.sessionPreset = .hd1280x720
        captureSession.inputs.forEach { captureSession.removeInput($0) }
        captureSession.outputs.forEach { captureSession.removeOutput($0) }

        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else {
            if SEGMENT_DEBUG_SAVE_IMAGES { print("❌ No back camera found") }
            captureSession.commitConfiguration()
            return
        }
        if SEGMENT_DEBUG_SAVE_IMAGES { print("✅ Found back camera: \(device.localizedName)") }

        do {
            let input = try AVCaptureDeviceInput(device: device)
            if captureSession.canAddInput(input) {
                captureSession.addInput(input)
                if SEGMENT_DEBUG_SAVE_IMAGES { print("✅ Added camera input") }
            }
            videoOutput.setSampleBufferDelegate(self, queue: sampleQueue)
            videoOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
            videoOutput.alwaysDiscardsLateVideoFrames = true
            if captureSession.canAddOutput(videoOutput) {
                captureSession.addOutput(videoOutput)
                if SEGMENT_DEBUG_SAVE_IMAGES { print("✅ Added video output") }
            }
            if let conn = videoOutput.connection(with: .video) {
                conn.videoRotationAngle = 90
                if SEGMENT_DEBUG_SAVE_IMAGES { print("✅ Set video rotation to 90°") }
            }
            captureSession.commitConfiguration()
            if SEGMENT_DEBUG_SAVE_IMAGES { print("✅ Camera configuration committed") }
            DispatchQueue.global(qos: .userInitiated).async {
                self.captureSession.startRunning()
                if SEGMENT_DEBUG_SAVE_IMAGES { print("✅ Camera session started") }
            }
        } catch {
            if SEGMENT_DEBUG_SAVE_IMAGES { print("❌ Camera setup error: \(error)") }
            captureSession.commitConfiguration()
        }
    }

    private func stopCamera() {
        DispatchQueue.global(qos: .userInitiated).async {
            if self.captureSession.isRunning {
                self.captureSession.stopRunning()
                if SEGMENT_DEBUG_SAVE_IMAGES { print("🛑 Camera session stopped") }
            }
        }
    }

    private func requestCameraPermissionAndStart() {
        if SEGMENT_DEBUG_SAVE_IMAGES { print("🔐 Checking camera permission...") }
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            if SEGMENT_DEBUG_SAVE_IMAGES { print("✅ Camera authorized") }
            if !captureSession.isRunning {
                DispatchQueue.global(qos: .userInitiated).async {
                    self.captureSession.startRunning()
                    if SEGMENT_DEBUG_SAVE_IMAGES { print("✅ Camera started after authorization check") }
                }
            }
        case .notDetermined:
            if SEGMENT_DEBUG_SAVE_IMAGES { print("⏳ Requesting camera permission...") }
            AVCaptureDevice.requestAccess(for: .video) { granted in
                if granted {
                    if SEGMENT_DEBUG_SAVE_IMAGES { print("✅ Camera permission granted") }
                    DispatchQueue.global(qos: .userInitiated).async { self.captureSession.startRunning() }
                } else {
                    if SEGMENT_DEBUG_SAVE_IMAGES { print("❌ Camera permission denied") }
                }
            }
        case .denied:
            if SEGMENT_DEBUG_SAVE_IMAGES { print("❌ Camera permission denied") }
        case .restricted:
            if SEGMENT_DEBUG_SAVE_IMAGES { print("❌ Camera permission restricted") }
        @unknown default:
            if SEGMENT_DEBUG_SAVE_IMAGES { print("❌ Unknown camera permission status") }
        }
    }

    // MARK: - Capture Delegate
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            if SEGMENT_DEBUG_SAVE_IMAGES { print("❌ No pixel buffer in sample") }
            return
        }
        detectionQueue.async { [weak self] in self?.processFrame(pixelBuffer) }
    }

    // MARK: - Main Processing
    private func processFrame(_ pixelBuffer: CVPixelBuffer) {
        guard let model = mlModel else {
            if SEGMENT_DEBUG_SAVE_IMAGES { print("⚠️ processFrame: model is nil") }
            return
        }
        let now = Date()
        guard now.timeIntervalSince(lastProcessTime) >= processInterval, !isProcessing else { return }
        lastProcessTime = now
        isProcessing = true
        
        if SEGMENT_DEBUG_SAVE_IMAGES { print("\n🔬 ========== RAW YOLO OUTPUT ==========") }

        guard let resized = resizePixelBuffer(pixelBuffer, width: 640, height: 640) else {
            if SEGMENT_DEBUG_SAVE_IMAGES { print("❌ Failed to resize pixel buffer") }
            isProcessing = false
            return
        }
        if SEGMENT_DEBUG_SAVE_IMAGES { print("✅ Resized pixel buffer to 640x640") }
        
        guard let inputArray = pixelBufferToMLMultiArray(resized) else {
            if SEGMENT_DEBUG_SAVE_IMAGES { print("❌ Failed to create MLMultiArray") }
            isProcessing = false
            return
        }
        if SEGMENT_DEBUG_SAVE_IMAGES { print("✅ Created MLMultiArray, shape: \(inputArray.shape)") }
        
        guard let inputProvider = try? MLDictionaryFeatureProvider(dictionary: ["image": inputArray]) else {
            if SEGMENT_DEBUG_SAVE_IMAGES { print("❌ Failed to create input provider") }
            isProcessing = false
            return
        }
        
        guard let output = try? model.prediction(from: inputProvider) else {
            if SEGMENT_DEBUG_SAVE_IMAGES { print("❌ Model prediction failed") }
            isProcessing = false
            return
        }
        if SEGMENT_DEBUG_SAVE_IMAGES { print("✅ Model prediction succeeded") }
        
        // Try var_1432 first, then var_2421 as fallback
        var detectionsArray: MLMultiArray?
        if let arr = output.featureValue(for: "var_1432")?.multiArrayValue {
            detectionsArray = arr
            if SEGMENT_DEBUG_SAVE_IMAGES { print("✅ Using output: var_1432") }
        } else if let arr = output.featureValue(for: "var_2421")?.multiArrayValue {
            detectionsArray = arr
            if SEGMENT_DEBUG_SAVE_IMAGES { print("✅ Using output: var_2421") }
        }
        
        guard let detArray = detectionsArray else {
            if SEGMENT_DEBUG_SAVE_IMAGES {
                print("❌ No detections array found (tried var_1432 and var_2421)")
                print("   Available outputs: \(output.featureNames)")
            }
            isProcessing = false
            return
        }
        
        guard let prototypesArray = output.featureValue(for: "p")?.multiArrayValue else {
            if SEGMENT_DEBUG_SAVE_IMAGES { print("❌ No prototypes array found") }
            isProcessing = false
            return
        }
        
        if SEGMENT_DEBUG_SAVE_IMAGES {
            print("Detections shape: \(detArray.shape)")
            print("Prototypes shape: \(prototypesArray.shape)")
        }

        // MARK: Extract Detections (FROM DOC1 - NSNumber subscripts)
        if SEGMENT_DEBUG_SAVE_IMAGES { print("\n🔍 ========== ALL DETECTIONS EXTRACTED ==========") }
        let allDetections = extractDetections(from: detArray)
        if SEGMENT_DEBUG_SAVE_IMAGES { print("📊 [DETECTION] Extracted \(allDetections.count) raw detections") }
        
        if allDetections.isEmpty {
            if SEGMENT_DEBUG_SAVE_IMAGES { print("❌ [DETECTION] No valid detections found") }
            DispatchQueue.main.async {
                self.maskImageView.image = nil
                self.isProcessing = false
            }
            return
        }

        // MARK: Hierarchical BBox NMS (FROM DOC1)
        let hierarchicalFiltered = applyHierarchicalNMS(detections: allDetections, iouThreshold: 0.9)
        if SEGMENT_DEBUG_SAVE_IMAGES { print("📊 [H-NMS] Kept \(hierarchicalFiltered.count) detections after hierarchical NMS") }
        
        // MARK: Mask IoU NMS with merging (FROM DOC1)
        let maskFiltered = applyMaskIoU(detections: hierarchicalFiltered, iouThreshold: 0.2, prototypes: prototypesArray)
        if SEGMENT_DEBUG_SAVE_IMAGES { print("📊 [MASK-FILTERED] Total kept: \(maskFiltered.count) detections") }

        if maskFiltered.isEmpty {
            if SEGMENT_DEBUG_SAVE_IMAGES { print("❌ [DETECTION] No valid detections found after mask filtering") }
            DispatchQueue.main.async {
                self.maskImageView.image = nil
                self.isProcessing = false
            }
            return
        }
        
        let best = maskFiltered.first!
        if SEGMENT_DEBUG_SAVE_IMAGES {
            print("✅ [BEST] Primary: \(best.className) @ \(Int(best.confidence * 100))%")
            print("   Position: (\(Int(best.x)), \(Int(best.y))), Size: \(Int(best.width))x\(Int(best.height))")
        }

        // MARK: Generate Pure Furniture Cutout (FROM DOC1 pattern)
        if SEGMENT_DEBUG_SAVE_IMAGES { print("\n🎨 ========== GENERATING CUTOUT ==========") }
        generatePureFurnitureCutout(detections: maskFiltered, prototypes: prototypesArray, originalImage: pixelBuffer)
    }

    // MARK: - Extract Detections (Accelerate - buffer copy + pointer arithmetic)
    private func extractDetections(from detections: MLMultiArray) -> [DetectionSmarty] {
        var all: [DetectionSmarty] = []
        
        let numAnchors = detections.shape[2].intValue
        let numFeatures = detections.shape[1].intValue
        if SEGMENT_DEBUG_SAVE_IMAGES { print("📊 Detections tensor: \(numFeatures) features x \(numAnchors) anchors") }
        
        // Copy MLMultiArray to float buffer ONCE (Accelerate)
        let totalCount = detections.count
        let detBuf = UnsafeMutablePointer<Float>.allocate(capacity: totalCount)
        defer { detBuf.deallocate() }
        
        if detections.dataType == .float16 {
            let src = detections.dataPointer.bindMemory(to: UInt16.self, capacity: totalCount)
            var srcBuf = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: src), height: 1, width: vImagePixelCount(totalCount), rowBytes: totalCount * 2)
            var dstBuf = vImage_Buffer(data: UnsafeMutableRawPointer(detBuf), height: 1, width: vImagePixelCount(totalCount), rowBytes: totalCount * 4)
            vImageConvert_Planar16FtoPlanarF(&srcBuf, &dstBuf, vImage_Flags(kvImageNoFlags))
            print("📊 Converted float16 → float32")
        } else if detections.dataType == .float32 {
            let src = detections.dataPointer.assumingMemoryBound(to: Float.self)
            memcpy(detBuf, src, totalCount * MemoryLayout<Float>.size)
            print("📊 Copied float32 buffer")
        } else {
            // Fallback for other types
            for i in 0..<totalCount {
                detBuf[i] = detections[i].floatValue
            }
            print("📊 Fallback copy for dataType: \(detections.dataType)")
        }
        
        // Shape: [1, numFeatures, numAnchors] -> index = feature * numAnchors + anchor
        for anchor in 0..<numAnchors {
            let x = detBuf[0 * numAnchors + anchor]
            let y = detBuf[1 * numAnchors + anchor]
            let w = detBuf[2 * numAnchors + anchor]
            let h = detBuf[3 * numAnchors + anchor]
            
            for (classIdx, className) in furnitureClasses {
                let confIdx = (4 + classIdx) * numAnchors + anchor
                let conf = detBuf[confIdx]
                
                if conf > confidenceThreshold {
                    var coeffs = [Float](repeating: 0, count: 32)
                    let coeffStart = (4 + 4585) * numAnchors + anchor
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
        
        // Log detection summary
        let grouped = Dictionary(grouping: all) { $0.className }
        print("\n📊 DETECTION SUMMARY:")
        print("Total detections: \(all.count)")
        print("Unique classes detected:")
        for (className, dets) in grouped.sorted(by: { $0.value.count > $1.value.count }) {
            let confidences = dets.map { Int($0.confidence * 100) }
            print("  - \(className): \(dets.count) detection(s), conf: \(confidences)%")
        }
        print("================================================\n")
        
        return all
    }

    // MARK: - Hierarchical BBox NMS (FROM DOC1)
    private func applyHierarchicalNMS(detections: [DetectionSmarty], iouThreshold: Float) -> [DetectionSmarty] {
        guard !detections.isEmpty else { return [] }
        
        var kept: [DetectionSmarty] = []
        var suppressed = Set<Int>()
        let sorted = detections.sorted { $0.confidence > $1.confidence }
        
        print("\n🔍 Hierarchical NMS Processing:")
        print("Input: \(sorted.count) detections")
        
        for (i, det) in sorted.enumerated() {
            if suppressed.contains(i) { continue }
            
            var shouldSuppress = false
            for existing in kept {
                let iou = calculateBBoxIoU(det1: det, det2: existing)
                if iou > iouThreshold {
                    shouldSuppress = true
                    if SEGMENT_DEBUG_SAVE_IMAGES {
                        print("❌ Suppressed duplicate: \(det.className) @ \(Int(det.confidence * 100))%")
                    }
                    break
                }
            }
            
            if !shouldSuppress {
                kept.append(det)
                print("✅ KEPT: \(det.className) @ \(Int(det.confidence * 100))%")
                suppressed.insert(i)
            }
        }
        
        print("Hierarchical NMS: \(sorted.count) → \(kept.count) detections")
        return kept
    }

    private func calculateBBoxIoU(det1: DetectionSmarty, det2: DetectionSmarty) -> Float {
        let x1 = max(det1.x - det1.width/2, det2.x - det2.width/2)
        let y1 = max(det1.y - det1.height/2, det2.y - det2.height/2)
        let x2 = min(det1.x + det1.width/2, det2.x + det2.width/2)
        let y2 = min(det1.y + det1.height/2, det2.y + det2.height/2)
        
        let intersection = max(0, x2 - x1) * max(0, y2 - y1)
        let area1 = det1.width * det1.height
        let area2 = det2.width * det2.height
        let union = area1 + area2 - intersection
        return union > 0 ? intersection / union : 0
    }

    // MARK: - Mask IoU NMS with Merging (FROM DOC1 - vDSP_mmul, vDSP_dotpr, vDSP_sve)
    private func applyMaskIoU(detections: [DetectionSmarty], iouThreshold: Float, prototypes: MLMultiArray) -> [DetectionSmarty] {
        guard !detections.isEmpty else { return [] }
        
        let sorted = detections.sorted { $0.confidence > $1.confidence }
        
        print("\n🔍 Mask-NMS (pure mask IoU, no class logic):")
        
        let shape = prototypes.shape.map { $0.intValue }  // [1, 32, 160, 160]
        let C = shape[1]
        let Hp = shape[2]
        let Wp = shape[3]
        let spatial = Hp * Wp
        
        print("Proto shape: C=\(C), H=\(Hp), W=\(Wp), spatial=\(spatial)")
        
        // Build proto matrix (FROM DOC1 pattern)
        var protoMatrix = [Float](repeating: 0, count: C * spatial)
        
        if prototypes.dataType == .float32 {
            print("Proto dataType: float32")
            let srcBase = prototypes.dataPointer.assumingMemoryBound(to: Float.self)
            for c in 0..<C {
                let srcChannelOffset = c * Hp * Wp
                let dstChannelOffset = c * spatial
                for y in 0..<Hp {
                    let rowOffset = y * Wp
                    for x in 0..<Wp {
                        let idx = rowOffset + x
                        let srcIndex = srcChannelOffset + idx
                        let dstIndex = dstChannelOffset + idx
                        protoMatrix[dstIndex] = srcBase[srcIndex]
                    }
                }
            }
        } else {
            print("Proto dataType: \(prototypes.dataType) (using NSNumber subscripts)")
            for c in 0..<C {
                for y in 0..<Hp {
                    for x in 0..<Wp {
                        let val = prototypes[[0, c, y, x] as [NSNumber]].floatValue
                        let dstIndex = c * spatial + (y * Wp + x)
                        protoMatrix[dstIndex] = val
                    }
                }
            }
        }
        
        // Log prototype matrix samples
        logPrototypeMatrix(prototypes, protoMatrix, C, Hp, Wp, spatial)
        
        // Generate masks using vDSP_mmul (FROM DOC1)
        var masks: [[Float]] = []
        masks.reserveCapacity(sorted.count)
        
        print("\n🔍 ========== MASK GENERATION (20x20 grid, 8th sample) ==========")
        
        for (idx, det) in sorted.enumerated() {
            var mask = [Float](repeating: 0, count: spatial)
            
            // Log mask coefficients for this detection
            print("📊 [MASK-COEFFS] Detection[\(idx)] \(det.className) coefficients: [\(det.maskCoeffs.prefix(8).map { String(format: "%.3f", $0) }.joined(separator: ", "))...]")
            
            // FROM DOC1: vDSP_mmul for coeffs × protoMatrix
            vDSP_mmul(
                det.maskCoeffs, 1,
                protoMatrix, 1,
                &mask, 1,
                1,
                vDSP_Length(spatial),
                vDSP_Length(C)
            )
            
            // Log pre-sigmoid values
            logMaskGeneration(idx, det, mask, Hp, Wp, isPreSigmoid: true)
            
            // Sigmoid
            for i in 0..<spatial {
                let v = mask[i]
                mask[i] = 1.0 / (1.0 + exp(-v))
            }
            
            // Log post-sigmoid values
            logMaskGeneration(idx, det, mask, Hp, Wp, isPreSigmoid: false)
            
            // Log mask stats
            var minVal: Float = 0, maxVal: Float = 0
            vDSP_minv(mask, 1, &minVal, vDSP_Length(spatial))
            vDSP_maxv(mask, 1, &maxVal, vDSP_Length(spatial))
            var sum: Float = 0
            vDSP_sve(mask, 1, &sum, vDSP_Length(spatial))
            let coverage = sum / Float(spatial) * 100
            print("📊 Mask[\(idx)] \(det.className): min=\(String(format: "%.3f", minVal)), max=\(String(format: "%.3f", maxVal)), coverage=\(String(format: "%.1f", coverage))%")
            
            masks.append(mask)
        }
        
        // NMS with merging (FROM DOC1)
        var kept: [DetectionSmarty] = []
        var keptMasks: [[Float]] = []
        
        for (i, det) in sorted.enumerated() {
            let candidateMask = masks[i]
            
            var mergedWithExisting = false
            var mergeTargetIndex = -1
            
            for (existingIndex, existingMask) in keptMasks.enumerated() {
                let iou = calculateMaskIoU(mask1: candidateMask, mask2: existingMask)
                if iou >= iouThreshold {
                    mergedWithExisting = true
                    mergeTargetIndex = existingIndex
                    print("🔄 MERGING (IoU \(Int(iou * 100))%) \(det.className) @ \(Int(det.confidence * 100))% with existing \(kept[mergeTargetIndex].className) @ \(Int(kept[mergeTargetIndex].confidence * 100))%")
                    break
                }
            }
            
            if mergedWithExisting && mergeTargetIndex >= 0 {
                // Merge masks (max blend)
                var mergedMask = keptMasks[mergeTargetIndex]
                for pixelIndex in 0..<candidateMask.count {
                    mergedMask[pixelIndex] = max(mergedMask[pixelIndex], candidateMask[pixelIndex])
                }
                keptMasks[mergeTargetIndex] = mergedMask
                
                // Keep higher confidence detection info
                if det.confidence > kept[mergeTargetIndex].confidence {
                    print("   → Replacing detection info with higher confidence: \(kept[mergeTargetIndex].className) @ \(Int(kept[mergeTargetIndex].confidence * 100))% → \(det.className) @ \(Int(det.confidence * 100))%")
                    kept[mergeTargetIndex] = det
                } else {
                    print("   → Keeping existing detection info: \(kept[mergeTargetIndex].className) @ \(Int(kept[mergeTargetIndex].confidence * 100))%")
                }
            } else {
                kept.append(det)
                keptMasks.append(candidateMask)
                print("✅ KEEP \(det.className) @ \(Int(det.confidence * 100))%")
            }
        }
        
        print("Mask-NMS: \(sorted.count) → \(kept.count) unique masks (by IoU)")
        return kept
    }

    // MARK: - Mask IoU Calculation (FROM DOC1 - vDSP_dotpr, vDSP_sve)
    private func calculateMaskIoU(mask1: [Float], mask2: [Float], eps: Float = 1e-7) -> Float {
        let n = min(mask1.count, mask2.count)
        guard n > 0 else { return 0 }
        
        var intersection: Float = 0
        vDSP_dotpr(mask1, 1, mask2, 1, &intersection, vDSP_Length(n))
        
        var sum1: Float = 0
        var sum2: Float = 0
        vDSP_sve(mask1, 1, &sum1, vDSP_Length(n))
        vDSP_sve(mask2, 1, &sum2, vDSP_Length(n))
        
        let union = sum1 + sum2 - intersection
        return intersection / (union + eps)
    }

    // MARK: - Generate Pure Furniture Cutout
    private func generatePureFurnitureCutout(detections: [DetectionSmarty], prototypes: MLMultiArray, originalImage: CVPixelBuffer) {
        let shape = prototypes.shape.map { $0.intValue }
        let C = shape[1]
        let Hp = shape[2]
        let Wp = shape[3]
        let spatial = Hp * Wp
        let cutoff = self.maskCutoff
        
        print("🎨 Generating cutout with \(detections.count) detections, cutoff=\(cutoff)")

        // Build proto matrix
        var protoMatrix = [Float](repeating: 0, count: C * spatial)
        
        if prototypes.dataType == .float32 {
            let srcBase = prototypes.dataPointer.assumingMemoryBound(to: Float.self)
            for c in 0..<C {
                for y in 0..<Hp {
                    for x in 0..<Wp {
                        let idx = y * Wp + x
                        protoMatrix[c * spatial + idx] = srcBase[c * spatial + idx]
                    }
                }
            }
        } else {
            for c in 0..<C {
                for y in 0..<Hp {
                    for x in 0..<Wp {
                        let val = prototypes[[0, c, y, x] as [NSNumber]].floatValue
                        protoMatrix[c * spatial + (y * Wp + x)] = val
                    }
                }
            }
        }

        // Accumulate all masks with max blend
        var globalMask = [Float](repeating: 0, count: spatial)

        print("\n🔍 ========== GLOBAL MASK ACCUMULATION (20x20 grid, 8th sample) ==========")

        for (idx, det) in detections.enumerated() {
            var mask = [Float](repeating: 0, count: spatial)
            
            // Log mask coefficients for this detection
            print("📊 [CUTOUT-COEFFS] Detection[\(idx)] \(det.className) coefficients: [\(det.maskCoeffs.prefix(8).map { String(format: "%.3f", $0) }.joined(separator: ", "))...]")
            
            vDSP_mmul(
                det.maskCoeffs, 1,
                protoMatrix, 1,
                &mask, 1,
                1,
                vDSP_Length(spatial),
                vDSP_Length(C)
            )

            // Log accumulation process
            logGlobalMaskAccumulation(idx, det, mask, &globalMask, Hp, Wp)

            // Sigmoid and max accumulation
            for i in 0..<spatial {
                let sigmoid = 1.0 / (1.0 + exp(-mask[i]))
                globalMask[i] = max(globalMask[i], sigmoid)
            }
            
            print("🎨 Accumulated mask[\(idx)] \(det.className)")
        }
        
        // Log global mask stats
        var minVal: Float = 0, maxVal: Float = 0
        vDSP_minv(globalMask, 1, &minVal, vDSP_Length(spatial))
        vDSP_maxv(globalMask, 1, &maxVal, vDSP_Length(spatial))
        var sum: Float = 0
        vDSP_sve(globalMask, 1, &sum, vDSP_Length(spatial))
        let coverage = sum / Float(spatial) * 100
        print("📊 Global mask: min=\(String(format: "%.3f", minVal)), max=\(String(format: "%.3f", maxVal)), coverage=\(String(format: "%.1f", coverage))%")
        
        // Count pixels above cutoff
        var aboveCutoff = 0
        for i in 0..<spatial {
            if globalMask[i] >= cutoff { aboveCutoff += 1 }
        }
        print("📊 Pixels above cutoff \(cutoff): \(aboveCutoff) / \(spatial) (\(String(format: "%.1f", Float(aboveCutoff) / Float(spatial) * 100))%)")

        // Create pure cutout
        autoreleasepool {
            let ciImage = CIImage(cvPixelBuffer: originalImage)
            let width = CVPixelBufferGetWidth(originalImage)
            let height = CVPixelBufferGetHeight(originalImage)
            print("📐 Original image: \(width)x\(height)")
            print("📐 Camera preset: .hd1280x720, Rotation: 90°")
            print("📐 Model input was: 640x640")

            guard let cgImage = ciContext.createCGImage(ciImage, from: ciImage.extent) else {
                print("❌ Failed to create CGImage")
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
                print("❌ Failed to create CGContext")
                DispatchQueue.main.async { self.isProcessing = false }
                return
            }
            
            guard let data = ctx.data else {
                print("❌ CGContext has no data")
                DispatchQueue.main.async { self.isProcessing = false }
                return
            }

            ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
            let pixels = data.bindMemory(to: UInt8.self, capacity: width * height * 4)

            // Apply mask (FROM DOC1 pattern)
            var opaqueCount = 0
            var transparentCount = 0
            
            for py in 0..<height {
                for px in 0..<width {
                    let idx = (py * width + px) * 4

                    let mx = Float(px) * Float(Wp) / Float(width)
                    let my = Float(py) * Float(Hp) / Float(height)

                    let x0 = Int(mx)
                    let y0 = Int(my)

                    guard x0 >= 0 && x0 < Wp && y0 >= 0 && y0 < Hp else {
                        pixels[idx + 3] = 0
                        transparentCount += 1
                        continue
                    }

                    let maskValue = globalMask[y0 * Wp + x0]
                    if maskValue >= cutoff {
                        pixels[idx + 3] = 255
                        opaqueCount += 1
                    } else {
                        pixels[idx + 3] = 0
                        transparentCount += 1
                    }
                }
            }
            
            // Log final pixel application samples
            logFinalPixelApplication(pixels, globalMask, cutoff, width, height, Wp, Hp)
            
            print("📊 Output: \(opaqueCount) opaque, \(transparentCount) transparent pixels")

            // Draw bright cyan bounding boxes AFTER transparency mask (so they stay visible) - ALWAYS
            let colors: [CGColor] = [
                CGColor(red: 0, green: 1, blue: 1, alpha: 1),      // Bright cyan
                CGColor(red: 1, green: 0, blue: 1, alpha: 1),      // Magenta
                CGColor(red: 0, green: 1, blue: 0, alpha: 1),      // Green
                CGColor(red: 1, green: 1, blue: 0, alpha: 1),      // Yellow
                CGColor(red: 1, green: 0.5, blue: 0, alpha: 1),   // Orange
                CGColor(red: 0.5, green: 0, blue: 1, alpha: 1),   // Purple
                CGColor(red: 1, green: 0, blue: 0, alpha: 1),      // Red
                CGColor(red: 0, green: 0.5, blue: 1, alpha: 1)    // Light blue
            ]
            
            for (index, detection) in detections.enumerated() {
                // Simple direct scaling - no rotation transformation
                let originalWidth = CGFloat(CVPixelBufferGetWidth(originalImage))  // 720
                let originalHeight = CGFloat(CVPixelBufferGetHeight(originalImage)) // 1280
                let modelSize: CGFloat = 640.0
                
                // Direct scaling from YOLO 640x640 to camera image dimensions
                let scaleX = originalWidth / modelSize   // 720/640 = 1.125
                let scaleY = originalHeight / modelSize  // 1280/640 = 2.0
                
                // Apply scaling directly to YOLO coordinates
                let centerX = CGFloat(detection.x) * scaleX
                let centerY = CGFloat(detection.y) * scaleY
                let boxWidth = CGFloat(detection.width) * scaleX
                let boxHeight = CGFloat(detection.height) * scaleY
                
                let x = centerX - boxWidth / 2
                let y = centerY - boxHeight / 2
                
                // Use different colors for each detection
                let color = colors[index % colors.count]
                
                // Draw bounding box
                ctx.setStrokeColor(color)
                ctx.setLineWidth(3.0)
                let rect = CGRect(x: x, y: y, width: boxWidth, height: boxHeight)
                ctx.stroke(rect)
                
                // Draw label background and text
                let confidence = Int(detection.confidence * 100)
                let labelText = "\(detection.className) \(confidence)%"
                
                // Create attributed string for label
                let font = CTFontCreateWithName("Helvetica-Bold" as CFString, 32, nil)  // Font size 32
                let attributes: [NSAttributedString.Key: Any] = [
                    .font: font,
                    .foregroundColor: UIColor.white
                ]
                let attributedString = NSAttributedString(string: labelText, attributes: attributes)
                
                // Calculate text size
                let textSize = attributedString.boundingRect(
                    with: CGSize(width: CGFloat.greatestFiniteMagnitude, height: CGFloat.greatestFiniteMagnitude),
                    options: [.usesLineFragmentOrigin, .usesFontLeading],
                    context: nil
                ).size
                
                // Position label above the bounding box
                let labelPadding: CGFloat = 4
                let labelWidth = textSize.width + (labelPadding * 2)
                let labelHeight = textSize.height + (labelPadding * 2)
                let labelX = max(0, min(x, originalWidth - labelWidth)) // Keep within bounds
                let labelY = max(0, y - labelHeight - 2) // Above the box, with small gap
                
                // Draw colored background for label
                ctx.setFillColor(color)
                let labelRect = CGRect(x: labelX, y: labelY, width: labelWidth, height: labelHeight)
                ctx.fill(labelRect)
                
                // Draw text without coordinate flipping to fix upside-down issue
                let textX = labelX + labelPadding
                let textY = labelY + labelPadding + textSize.height // Add text height to position correctly
                
                // Draw the text directly without flipping coordinates
                let line = CTLineCreateWithAttributedString(attributedString)
                ctx.textPosition = CGPoint(x: textX, y: textY)
                CTLineDraw(line, ctx)
                
                print("📦 Drew bbox for \(detection.className) @ (\(Int(x)), \(Int(y)), \(Int(boxWidth))x\(Int(boxHeight))) with color \(index % colors.count)")
                print("   🔢 Original YOLO (640x640): center(\(detection.x), \(detection.y)), size(\(detection.width), \(detection.height))")
                print("   📐 Scale factors: X=\(String(format: "%.3f", scaleX)), Y=\(String(format: "%.3f", scaleY))")
                print("   🎯 Final (scaled): center(\(Int(centerX)), \(Int(centerY))), size(\(Int(boxWidth))x\(Int(boxHeight)))")
                print("   🏷️ Label: '\(labelText)' @ (\(Int(labelX)), \(Int(labelY)))")
            }

            if let outImage = ctx.makeImage() {
                print("✅ Created output CGImage")
                DispatchQueue.main.async {
                    self.maskImageView.image = UIImage(cgImage: outImage, scale: 1.0, orientation: .up)
                    self.isProcessing = false
                    print("✅ ==================== FRAME COMPLETE ====================\n")
                }
            } else {
                print("❌ Failed to make output image")
                DispatchQueue.main.async { self.isProcessing = false }
            }
        }
    }

    // MARK: - Pixel Buffer to MLMultiArray (Accelerate - pointer arithmetic)
    private func pixelBufferToMLMultiArray(_ pixelBuffer: CVPixelBuffer) -> MLMultiArray? {
        guard let array = try? MLMultiArray(shape: [1, 3, 640, 640], dataType: .float32) else {
            print("❌ Failed to create MLMultiArray")
            return nil
        }
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            print("❌ No base address in pixel buffer")
            return nil
        }
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let srcBuffer = baseAddress.assumingMemoryBound(to: UInt8.self)
        let dstPtr = array.dataPointer.assumingMemoryBound(to: Float.self)
        
        let spatial = 640 * 640
        
        for y in 0..<640 {
            let rowOffset = y * bytesPerRow
            let yOffset = y * 640
            for x in 0..<640 {
                let srcIdx = rowOffset + x * 4
                let dstIdx = yOffset + x
                // BGRA -> RGB channels-first
                dstPtr[0 * spatial + dstIdx] = Float(srcBuffer[srcIdx + 2]) / 255.0  // R
                dstPtr[1 * spatial + dstIdx] = Float(srcBuffer[srcIdx + 1]) / 255.0  // G
                dstPtr[2 * spatial + dstIdx] = Float(srcBuffer[srcIdx + 0]) / 255.0  // B
            }
        }
        
        // Log pixel conversion samples
        logPixelToMLConversion(srcBuffer, dstPtr, spatial, bytesPerRow)
        if SEGMENT_DEBUG_SAVE_IMAGES { print("✅ [PIXEL→ML] Conversion complete\n") }
        
        return array
    }

    // MARK: - Debug Logging Methods
    private func logGridSamples20x20(_ title: String, gridSize: Int = 20, sampleOffset: Int = 7, logAction: (Int, Int, Int, Int) -> Void) {
        guard SEGMENT_DEBUG_SAVE_IMAGES else { return }
        print("\n🔍 ========== \(title.uppercased()) (20x20 grid, 8th sample) ==========")
        for gy in 0..<gridSize {
            for gx in 0..<gridSize {
                let y = gy * 8 + sampleOffset  // 8th sample in each grid cell
                let x = gx * 8 + sampleOffset
                logAction(gy, gx, x, y)
            }
        }
    }
    
    private func logPixelToMLConversion(_ srcBuffer: UnsafePointer<UInt8>, _ dstPtr: UnsafePointer<Float>, _ spatial: Int, _ bytesPerRow: Int) {
        guard SEGMENT_DEBUG_SAVE_IMAGES else { return }
        logGridSamples20x20("PIXEL BUFFER TO ML ARRAY") { gy, gx, x, y in
            guard y < 640 && x < 640 else { return }
            let srcIdx = y * bytesPerRow + x * 4
            let dstIdx = y * 640 + x
            let rVal = Float(srcBuffer[srcIdx + 2]) / 255.0
            let gVal = Float(srcBuffer[srcIdx + 1]) / 255.0
            let bVal = Float(srcBuffer[srcIdx + 0]) / 255.0
            print("📊 [PIXEL→ML] Grid[\(gy),\(gx)] Pixel(\(x),\(y)): BGRA(\(srcBuffer[srcIdx]),\(srcBuffer[srcIdx+1]),\(srcBuffer[srcIdx+2]),\(srcBuffer[srcIdx+3])) → RGB(\(String(format: "%.3f", rVal)),\(String(format: "%.3f", gVal)),\(String(format: "%.3f", bVal)))")
        }
    }
    
    private func logPrototypeMatrix(_ prototypes: MLMultiArray, _ protoMatrix: [Float], _ C: Int, _ Hp: Int, _ Wp: Int, _ spatial: Int) {
        guard SEGMENT_DEBUG_SAVE_IMAGES else { return }
        logGridSamples20x20("PROTOTYPE MATRIX") { gy, gx, x, y in
            guard y < Hp && x < Wp else { return }
            for c in 0..<min(4, C) {  // Only log first 4 channels
                let dstIndex = c * spatial + (y * Wp + x)
                let val = protoMatrix[dstIndex]
                print("📊 [PROTO-MAT] Channel[\(c)] Grid[\(gy),\(gx)] Proto(\(x),\(y)): value=\(String(format: "%.3f", val))")
            }
        }
    }
    
    private func logMaskGeneration(_ idx: Int, _ det: DetectionSmarty, _ mask: [Float], _ Hp: Int, _ Wp: Int, isPreSigmoid: Bool = true) {
        guard SEGMENT_DEBUG_SAVE_IMAGES else { return }
        let stage = isPreSigmoid ? "Pre-sigmoid" : "Post-sigmoid"
        print("📊 [MASK-GEN] Detection[\(idx)] \(det.className) - \(stage) samples:")
        logGridSamples20x20("") { gy, gx, x, y in
            guard y < Hp && x < Wp else { return }
            let maskIdx = y * Wp + x
            let val = mask[maskIdx]
            let suffix = isPreSigmoid ? "pre-sigmoid" : "post-sigmoid"
            print("    Grid[\(gy),\(gx)] Mask(\(x),\(y)): \(suffix)=\(String(format: "%.3f", val))")
        }
    }
    
    private func logGlobalMaskAccumulation(_ idx: Int, _ det: DetectionSmarty, _ mask: [Float], _ globalMask: inout [Float], _ Hp: Int, _ Wp: Int) {
        guard SEGMENT_DEBUG_SAVE_IMAGES else { return }
        print("📊 [GLOBAL-ACC] Detection[\(idx)] \(det.className) - Accumulation samples:")
        logGridSamples20x20("") { gy, gx, x, y in
            guard y < Hp && x < Wp else { return }
            let maskIdx = y * Wp + x
            let preVal = mask[maskIdx]
            let sigmoid = 1.0 / (1.0 + exp(-preVal))
            let oldGlobal = globalMask[maskIdx]
            let newGlobal = max(oldGlobal, sigmoid)
            print("    Grid[\(gy),\(gx)] Mask(\(x),\(y)): pre=\(String(format: "%.3f", preVal)), sigmoid=\(String(format: "%.3f", sigmoid)), old_global=\(String(format: "%.3f", oldGlobal)), new_global=\(String(format: "%.3f", newGlobal))")
        }
    }
    
    private func logFinalPixelApplication(_ pixels: UnsafePointer<UInt8>, _ globalMask: [Float], _ cutoff: Float, _ width: Int, _ height: Int, _ Wp: Int, _ Hp: Int) {
        guard SEGMENT_DEBUG_SAVE_IMAGES else { return }
        logGridSamples20x20("FINAL PIXEL APPLICATION") { gy, gx, gridX, gridY in
            let px = gx * (width / 20) + (width / 20) / 8
            let py = gy * (height / 20) + (height / 20) / 8
            guard px < width && py < height else { return }
            
            let idx = (py * width + px) * 4
            let mx = Float(px) * Float(Wp) / Float(width)
            let my = Float(py) * Float(Hp) / Float(height)
            let x0 = Int(mx)
            let y0 = Int(my)
            
            guard x0 >= 0 && x0 < Wp && y0 >= 0 && y0 < Hp else { return }
            
            let maskValue = globalMask[y0 * Wp + x0]
            let originalR = pixels[idx]
            let originalG = pixels[idx + 1]
            let originalB = pixels[idx + 2]
            let originalA = pixels[idx + 3]
            let newAlpha = maskValue >= cutoff ? 255 : 0
            print("📊 [FINAL-APP] Grid[\(gy),\(gx)] Pixel(\(px),\(py)): mask_coord(\(x0),\(y0)), mask_val=\(String(format: "%.3f", maskValue)), cutoff=\(cutoff), RGBA(\(originalR),\(originalG),\(originalB),\(originalA)) → alpha=\(newAlpha)")
        }
    }

    // MARK: - Resize Pixel Buffer
    private func resizePixelBuffer(_ pixelBuffer: CVPixelBuffer, width: Int, height: Int) -> CVPixelBuffer? {
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let scaleX = CGFloat(width) / CGFloat(CVPixelBufferGetWidth(pixelBuffer))
        let scaleY = CGFloat(height) / CGFloat(CVPixelBufferGetHeight(pixelBuffer))
        var out: CVPixelBuffer?
        CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA, nil, &out)
        guard let dst = out else {
            if SEGMENT_DEBUG_SAVE_IMAGES { print("❌ Failed to create output pixel buffer") }
            return nil
        }
        CIContext().render(ciImage.transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY)), to: dst)
        return dst
    }
}
