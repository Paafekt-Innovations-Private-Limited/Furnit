import SwiftUI
import AVFoundation
import CoreML
import Vision
import CoreImage
import Photos
import Accelerate

// MARK: - Camera Preview Layer
struct CameraPreviewLayer: UIViewRepresentable {
    let session: AVCaptureSession
    
    func makeUIView(context: Context) -> UIView {
        let view = UIView()
        let previewLayer = AVCaptureVideoPreviewLayer(session: session)
        previewLayer.videoGravity = .resizeAspectFill
        view.layer.addSublayer(previewLayer)
        return view
    }
    
    func updateUIView(_ uiView: UIView, context: Context) {
        if let layer = uiView.layer.sublayers?.first as? AVCaptureVideoPreviewLayer {
            layer.frame = uiView.bounds
        }
    }
}

// MARK: - Main View
struct SegmentFurniture: View {
    @Binding var capturedImage: UIImage?
    @Binding var isShowingCamera: Bool
    let roomImage: UIImage?
    
    @StateObject private var camera = FurnitureSegmentationModel()
    
    @State private var scaleMultiplier: CGFloat = 0.5
    @State private var lastScale: CGFloat = 1.0
    @State private var dragOffset: CGSize = .zero
    @State private var accumulatedOffset: CGSize = .zero
    @State private var showingSaveSuccess = false
    @State private var saveMessage = ""
    
    var body: some View {
        ZStack {
            // Camera preview for furniture detection
            CameraPreviewLayer(session: camera.session)
                .ignoresSafeArea()
            
            // Segmented furniture overlay
            if let segmented = camera.segmentedImage {
                Image(uiImage: segmented)
                    .resizable()
                    .scaledToFill()
                    .scaleEffect(scaleMultiplier)
                    .offset(x: dragOffset.width + accumulatedOffset.width,
                           y: dragOffset.height + accumulatedOffset.height)
                    .position(x: UIScreen.main.bounds.width / 2,
                             y: UIScreen.main.bounds.height / 2)
                    .gesture(
                        SimultaneousGesture(
                            // Drag gesture
                            DragGesture()
                                .onChanged { value in
                                    dragOffset = value.translation
                                }
                                .onEnded { value in
                                    accumulatedOffset.width += value.translation.width
                                    accumulatedOffset.height += value.translation.height
                                    dragOffset = .zero
                                },
                            // Pinch gesture
                            MagnificationGesture()
                                .onChanged { value in
                                    let delta = value / lastScale
                                    lastScale = value
                                    let newScale = scaleMultiplier * delta
                                    scaleMultiplier = min(max(newScale, 0.2), 2.0)
                                }
                                .onEnded { value in
                                    lastScale = 1.0
                                }
                        )
                    )
                    .ignoresSafeArea()
                    .opacity(camera.furnitureOpacity)
                    .animation(.easeOut(duration: 0.05), value: camera.furnitureOpacity)
            }
            
            // Green BBox Overlay
            if camera.currentBBox != .zero && camera.segmentedImage != nil {
                Canvas { context, size in
                    let rect = Path(camera.currentBBox)
                    context.stroke(
                        rect,
                        with: .color(.green),
                        lineWidth: 3
                    )
                }
                .ignoresSafeArea()
                .allowsHitTesting(false)
            }
            
            // Model indicator - show which model is being used
            if camera.segmentedImage != nil {
                VStack {
                    HStack {
                        Spacer()
                        
                        // Model badge
                        HStack(spacing: 4) {
                            Image(systemName: camera.usingCustomModel ? "star.fill" : "cube.fill")
                                .font(.caption2)
                            Text(camera.usingCustomModel ? "Custom Table Model" : "General Model")
                                .font(.caption2)
                        }
                        .foregroundColor(.white)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(Capsule().fill(camera.usingCustomModel ? Color.orange : Color.blue))
                        .padding(.trailing, 100)
                    }
                    .padding(.top, 70)
                    
                    Spacer()
                }
            }
            
            // Controls
            VStack {
                HStack {
                    Spacer()
                    
                    Button(action: { isShowingCamera = false }) {
                        Image(systemName: "xmark.circle.fill")
                            .font(.title)
                            .foregroundColor(.white)
                    }
                }
                .padding(.top, 60)
                .padding(.horizontal)
                
                Spacer()
                
                // Capture and Reset buttons
                if camera.segmentedImage != nil {
                    HStack(spacing: 16) {
                        // Capture - furniture + 3D room
                        Button(action: { captureFurnitureWithRoom() }) {
                            VStack {
                                Image(systemName: "camera.circle.fill")
                                Text("Capture")
                                    .font(.caption2)
                            }
                            .foregroundColor(.white)
                            .frame(width: 70, height: 70)
                            .background(Circle().fill(Color.green))
                            .shadow(radius: 5)
                        }
                        
                        // Reset
                        Button(action: {
                            camera.resetSegmentation()
                            scaleMultiplier = 0.5
                            dragOffset = .zero
                            accumulatedOffset = .zero
                        }) {
                            VStack {
                                Image(systemName: "arrow.counterclockwise")
                                Text("Reset")
                                    .font(.caption2)
                            }
                            .foregroundColor(.white)
                            .frame(width: 60, height: 60)
                            .background(Circle().fill(Color.orange))
                            .shadow(radius: 3)
                        }
                    }
                    .padding(.bottom, 50)
                    .padding(.trailing, 20)
                    .frame(maxWidth: .infinity, alignment: .trailing)
                }
            }
            
            // Gesture hint
            if camera.segmentedImage != nil {
                VStack {
                    Spacer()
                    Text("Pinch to scale • Drag to move")
                        .font(.caption)
                        .foregroundColor(.white)
                        .padding(8)
                        .background(Capsule().fill(Color.black.opacity(0.6)))
                        .padding(.bottom, 120)
                }
            }
            
            // Success message
            if showingSaveSuccess {
                VStack {
                    Spacer()
                    HStack {
                        Image(systemName: "checkmark.circle.fill")
                        Text(saveMessage)
                    }
                    .foregroundColor(.white)
                    .padding()
                    .background(Capsule().fill(Color.green))
                    Spacer().frame(height: 100)
                }
            }
            
            // Initialization progress
            if camera.isInitializing {
                ZStack {
                    Color.black.opacity(0.8)
                        .ignoresSafeArea()
                    
                    VStack(spacing: 24) {
                        ZStack {
                            Circle()
                                .fill(Color.blue.opacity(0.2))
                                .frame(width: 100, height: 100)
                            
                            Image(systemName: "camera.viewfinder")
                                .font(.system(size: 40))
                                .foregroundColor(.blue)
                        }
                        
                        VStack(spacing: 12) {
                            Text("Detecting Furniture")
                                .font(.title2)
                                .fontWeight(.semibold)
                                .foregroundColor(.white)
                            
                            Text(camera.initStage)
                                .font(.subheadline)
                                .foregroundColor(.gray)
                        }
                        
                        VStack(spacing: 8) {
                            ProgressView(value: camera.initProgress, total: 1.0)
                                .progressViewStyle(LinearProgressViewStyle(tint: .blue))
                                .frame(width: 250)
                            
                            Text("\(Int(camera.initProgress * 100))%")
                                .font(.caption)
                                .foregroundColor(.gray)
                        }
                    }
                }
            }
        }
        .onAppear {
            camera.startSession()
            print("📸 Room image: \(roomImage != nil ? "Available (\(Int(roomImage!.size.width))x\(Int(roomImage!.size.height)))" : "Not available")")
        }
        .onDisappear { camera.stopSession() }
    }
    
    private func captureFurnitureWithRoom() {
        guard let furniture = camera.segmentedImage else {
            print("❌ No furniture image")
            saveMessage = "No furniture detected!"
            showingSaveSuccess = true
            DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
                showingSaveSuccess = false
            }
            return
        }
        
        print("✅ Furniture: \(furniture.size)")
        
        guard let room = roomImage else {
            print("❌ No room image!")
            saveMessage = "No room image!"
            showingSaveSuccess = true
            DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
                showingSaveSuccess = false
            }
            return
        }
        
        print("✅ Room: \(room.size)")
        
        // Create composite - simple centered furniture
        UIGraphicsBeginImageContextWithOptions(room.size, false, room.scale)
        defer { UIGraphicsEndImageContext() }
        
        // Draw room
        room.draw(at: .zero)
        
        // Draw furniture centered
        let furnitureRect = CGRect(
            x: (room.size.width - furniture.size.width) / 2,
            y: (room.size.height - furniture.size.height) / 2,
            width: furniture.size.width,
            height: furniture.size.height
        )
        
        furniture.draw(in: furnitureRect)
        
        guard let composite = UIGraphicsGetImageFromCurrentImageContext() else {
            print("❌ Failed to create composite")
            saveMessage = "Composite failed!"
            showingSaveSuccess = true
            return
        }
        
        print("✅ Composite: \(composite.size)")
        
        // Save
        PHPhotoLibrary.requestAuthorization(for: .addOnly) { status in
            DispatchQueue.main.async {
                if status == .authorized || status == .limited {
                    PHPhotoLibrary.shared().performChanges({
                        PHAssetChangeRequest.creationRequestForAsset(from: composite)
                    }) { success, error in
                        DispatchQueue.main.async {
                            if success {
                                self.saveMessage = "Saved!"
                                self.showingSaveSuccess = true
                                DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
                                    self.showingSaveSuccess = false
                                    self.isShowingCamera = false
                                }
                            } else {
                                self.saveMessage = "Failed!"
                                self.showingSaveSuccess = true
                            }
                        }
                    }
                }
            }
        }
    }
}

// MARK: - Detection Structure
struct Detection {
    let x: Float
    let y: Float
    let width: Float
    let height: Float
    let confidence: Float
    let classIdx: Int
    let className: String
    let maskCoeffs: [Float]
    let timestamp: Date
    
    var bbox: CGRect {
        return CGRect(x: CGFloat(x), y: CGFloat(y), width: CGFloat(width), height: CGFloat(height))
    }
}

// MARK: - Cross-Attention BBox Tracker
class CrossAttentionBBoxTracker {
    private var detectionHistory: [Detection] = []
    private let historySize = 5
    
    func updateBBox(newDetection: Detection, imageSize: CGSize) -> CGRect {
        print("\n🔄 === BBox Tracking Update ===")
        print("📥 Input detection: \(newDetection.className) conf=\(String(format: "%.2f", newDetection.confidence))")
        print("📊 History size: \(detectionHistory.count)/\(historySize)")
        
        detectionHistory.append(newDetection)
        if detectionHistory.count > historySize {
            detectionHistory.removeFirst()
        }
        
        if detectionHistory.count < 2 {
            let result = convertToScreenCoordinates(detection: newDetection, imageSize: imageSize)
            print("✅ Output BBox: \(result)")
            return result
        }
        
        let history = Array(detectionHistory.dropLast())
        let smoothedBBox = crossAttentionBBox(current: newDetection, history: history, imageSize: imageSize)
        
        print("✅ Final BBox: \(smoothedBBox)")
        print("=================================\n")
        return smoothedBBox
    }
    
    func reset() {
        detectionHistory.removeAll()
        print("🔄 BBox tracker reset - history cleared")
    }
    
    private func convertToScreenCoordinates(detection: Detection, imageSize: CGSize) -> CGRect {
        let cameraWidth = Float(imageSize.width)
        let cameraHeight = Float(imageSize.height)
        let scale = cameraWidth / 640.0
        
        let camX = (detection.x - detection.width/2) * scale
        let camY = (detection.y - detection.height/2) * scale
        let camW = detection.width * scale
        let camH = detection.height * scale
        
        let screenWidth = cameraHeight
        let screenHeight = cameraWidth
        
        let rotatedX = camY
        let rotatedY = screenHeight - (camX + camW)
        let rotatedW = camH
        let rotatedH = camW
        
        return CGRect(
            x: CGFloat(rotatedX),
            y: CGFloat(rotatedY),
            width: CGFloat(rotatedW),
            height: CGFloat(rotatedH)
        )
    }
    
    private func calculateSimilarity(current: Detection, history: Detection) -> Float {
        let iou = calculateIoU(current, history)
        let classSimilarity: Float = (current.className == history.className) ? 1.0 : 0.0
        let confSimilarity = min(current.confidence, history.confidence)
        let timeDiff = current.timestamp.timeIntervalSince(history.timestamp)
        let temporalWeight = exp(-Float(timeDiff) / 0.5)
        
        return (
            0.5 * iou +
            0.2 * classSimilarity +
            0.1 * confSimilarity +
            0.2 * temporalWeight
        )
    }
    
    private func calculateIoU(_ a: Detection, _ b: Detection) -> Float {
        let x1 = max(a.x - a.width/2, b.x - b.width/2)
        let y1 = max(a.y - a.height/2, b.y - b.height/2)
        let x2 = min(a.x + a.width/2, b.x + b.width/2)
        let y2 = min(a.y + a.height/2, b.y + b.height/2)
        
        let intersection = max(0, x2 - x1) * max(0, y2 - y1)
        let union = a.width * a.height + b.width * b.height - intersection
        
        return union > 0 ? intersection / union : 0
    }
    
    private func computeAttentionWeights(current: Detection, history: [Detection]) -> [Float] {
        var scores: [Float] = []
        for historyFrame in history {
            scores.append(calculateSimilarity(current: current, history: historyFrame))
        }
        
        let maxScore = scores.max() ?? 0
        var expScores: [Float] = []
        var sumExp: Float = 0
        
        for score in scores {
            let expScore = exp(score - maxScore)
            expScores.append(expScore)
            sumExp += expScore
        }
        
        return expScores.map { $0 / max(sumExp, 0.0001) }
    }
    
    private func crossAttentionBBox(current: Detection, history: [Detection], imageSize: CGSize) -> CGRect {
        let weights = computeAttentionWeights(current: current, history: history)
        
        var weightedX: Float = 0
        var weightedY: Float = 0
        var weightedW: Float = 0
        var weightedH: Float = 0
        
        for (idx, historyFrame) in history.enumerated() {
            let weight = weights[idx]
            weightedX += weight * historyFrame.x
            weightedY += weight * historyFrame.y
            weightedW += weight * historyFrame.width
            weightedH += weight * historyFrame.height
        }
        
        let finalX = 0.8 * weightedX + 0.2 * current.x
        let finalY = 0.8 * weightedY + 0.2 * current.y
        let finalW = 0.8 * weightedW + 0.2 * current.width
        let finalH = 0.8 * weightedH + 0.2 * current.height
        
        let smoothedDetection = Detection(
            x: finalX, y: finalY, width: finalW, height: finalH,
            confidence: current.confidence,
            classIdx: current.classIdx,
            className: current.className,
            maskCoeffs: current.maskCoeffs,
            timestamp: current.timestamp
        )
        
        return convertToScreenCoordinates(detection: smoothedDetection, imageSize: imageSize)
    }
}

// MARK: - Main Model with DUAL YOLO Models
class FurnitureSegmentationModel: NSObject, ObservableObject {
    @Published var segmentedImage: UIImage?
    @Published var furnitureOpacity: Double = 0.0
    @Published var currentFPS: Double = 0.0
    @Published var lastConfidence: Float = 0.0
    @Published var lastDetectedClass: String = ""
    @Published var usingCustomModel = false // NEW: Track which model is active
    
    @Published var isInitializing = true
    @Published var initProgress: Double = 0.0
    @Published var initStage = ""
    
    @Published var currentBBox: CGRect = .zero
    private let bboxTracker = CrossAttentionBBoxTracker()
    
    let session = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let videoQueue = DispatchQueue(label: "furnitureSegQueue", qos: .userInitiated)
    
    // DUAL MODELS
    private var generalYoloModel: VNCoreMLModel?  // yolo11x-seg for all furniture
    private var customTableModel: VNCoreMLModel?   // best.pt for YOUR table
    
    private let context = CIContext(options: [.workingColorSpace: CGColorSpaceCreateDeviceRGB()])
    
    private let cocoClasses = [
        0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
        5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
        10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
        14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
        20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack",
        25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee",
        30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite", 34: "baseball bat",
        35: "baseball glove", 36: "skateboard", 37: "surfboard", 38: "tennis racket",
        39: "bottle", 40: "wine glass", 41: "cup", 42: "fork", 43: "knife",
        44: "spoon", 45: "bowl", 46: "banana", 47: "apple", 48: "sandwich",
        49: "orange", 50: "broccoli", 51: "carrot", 52: "hot dog", 53: "pizza",
        54: "donut", 55: "cake", 56: "chair", 57: "couch", 58: "potted plant",
        59: "bed", 60: "dining table", 61: "toilet", 62: "tv", 63: "laptop",
        64: "mouse", 65: "remote", 66: "keyboard", 67: "cell phone", 68: "microwave",
        69: "oven", 70: "toaster", 71: "sink", 72: "refrigerator", 73: "book",
        74: "clock", 75: "vase", 76: "scissors", 77: "teddy bear", 78: "hair drier",
        79: "toothbrush"
    ]
    
    private var lastProcessTime = Date()
    private let processInterval: TimeInterval = 0.066
    private var frameCount = 0
    private var fpsStartTime = Date()
    
    private func sigmoid(_ x: Float) -> Float {
        return 1.0 / (1.0 + exp(-x))
    }
    
    override init() {
        super.init()
        loadAllModels()
        setupCamera()
    }
    
    func resetSegmentation() {
        DispatchQueue.main.async {
            self.segmentedImage = nil
            self.furnitureOpacity = 0.0
            self.lastConfidence = 0.0
            self.lastDetectedClass = ""
            self.currentBBox = .zero
            self.usingCustomModel = false
            self.bboxTracker.reset()
        }
    }
    
    // MARK: - DUAL MODEL LOADING
    private func loadAllModels() {
        print("🔍 Loading DUAL YOLO models...")
        
        // Load general model (yolo11x-seg)
        for ext in ["mlmodelc", "mlpackage"] {
            if let modelURL = Bundle.main.url(forResource: "yolo11x-seg", withExtension: ext) {
                print("📦 Found general model: yolo11x-seg.\(ext)")
                do {
                    let model = try MLModel(contentsOf: modelURL)
                    generalYoloModel = try VNCoreMLModel(for: model)
                    print("✅ General YOLO11-seg loaded!")
                } catch {
                    print("❌ General model failed: \(error)")
                }
                break
            }
        }
        
        // Load custom table model (best.pt converted to CoreML)
        for ext in ["mlmodelc", "mlpackage"] {
            if let modelURL = Bundle.main.url(forResource: "best", withExtension: ext) {
                print("📦 Found custom table model: best.\(ext)")
                do {
                    let model = try MLModel(contentsOf: modelURL)
                    customTableModel = try VNCoreMLModel(for: model)
                    print("✅ Custom table model loaded!")
                } catch {
                    print("❌ Custom model failed: \(error)")
                }
                break
            }
        }
        
        if generalYoloModel != nil && customTableModel != nil {
            print("🎉 DUAL MODEL SYSTEM READY!")
            print("   - General: All 80 COCO classes")
            print("   - Custom: YOUR table (optimized)")
        } else if generalYoloModel != nil {
            print("⚠️ Only general model loaded")
        } else if customTableModel != nil {
            print("⚠️ Only custom table model loaded")
        } else {
            print("❌ NO MODELS LOADED!")
        }
    }
    
    private func setupCamera() {
        session.sessionPreset = .hd1280x720
        
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else {
            print("❌ No camera")
            return
        }
        
        do {
            let input = try AVCaptureDeviceInput(device: device)
            if session.canAddInput(input) {
                session.addInput(input)
            }
            
            videoOutput.setSampleBufferDelegate(self, queue: videoQueue)
            videoOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
            videoOutput.alwaysDiscardsLateVideoFrames = true
            
            if session.canAddOutput(videoOutput) {
                session.addOutput(videoOutput)
                
                if let connection = videoOutput.connection(with: .video) {
                    connection.videoRotationAngle = 90
                    connection.isVideoMirrored = false
                }
            }
            
            print("✅ Camera configured")
        } catch {
            print("❌ Camera setup failed: \(error)")
        }
    }
    
    func startSession() {
        if !session.isRunning {
            DispatchQueue.main.async {
                self.isInitializing = true
                self.initProgress = 0.0
                self.initStage = "Starting camera..."
            }
            
            DispatchQueue.global(qos: .background).async {
                self.session.startRunning()
                DispatchQueue.main.async {
                    print("✅ Camera started")
                    self.fpsStartTime = Date()
                    self.initProgress = 0.5
                    self.initStage = "Detecting furniture..."
                }
            }
        }
    }
    
    func stopSession() {
        if session.isRunning {
            session.stopRunning()
        }
    }
    
    private func updateFPS() {
        frameCount += 1
        let elapsed = Date().timeIntervalSince(fpsStartTime)
        if elapsed > 1.0 {
            DispatchQueue.main.async {
                self.currentFPS = Double(self.frameCount) / elapsed
            }
            frameCount = 0
            fpsStartTime = Date()
        }
    }
    
    // MARK: - DUAL MODEL INFERENCE STRATEGY
    private func processWithDualModels(pixelBuffer: CVPixelBuffer) {
        let now = Date()
        guard now.timeIntervalSince(lastProcessTime) >= processInterval else { return }
        
        lastProcessTime = now
        updateFPS()
        
        // Strategy: Try custom model first, fallback to general
        // Custom model is faster and more accurate for YOUR table
        
        if let customModel = customTableModel {
            // First: Try custom table model
            let customRequest = VNCoreMLRequest(model: customModel) { [weak self] request, error in
                if let error = error {
                    print("❌ Custom model error: \(error)")
                    return
                }
                
                // Check if we got good table detection
                if let detections = self?.processYOLOResults(request.results, originalImage: pixelBuffer, isCustomModel: true),
                   !detections.isEmpty {
                    // Custom model found table!
                    print("⭐ Using CUSTOM table model")
                    return
                }
                
                // Custom model didn't find table, try general model
                if let generalModel = self?.generalYoloModel {
                    print("🔄 Switching to GENERAL model")
                    let generalRequest = VNCoreMLRequest(model: generalModel) { request, error in
                        if let error = error {
                            print("❌ General model error: \(error)")
                            return
                        }
                        
                        self?.processYOLOResults(request.results, originalImage: pixelBuffer, isCustomModel: false)
                    }
                    
                    generalRequest.imageCropAndScaleOption = .scaleFill
                    
                    let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
                    do {
                        try handler.perform([generalRequest])
                    } catch {
                        print("❌ General inference failed: \(error)")
                    }
                }
            }
            
            customRequest.imageCropAndScaleOption = .scaleFill
            
            let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
            do {
                try handler.perform([customRequest])
            } catch {
                print("❌ Custom inference failed: \(error)")
            }
            
        } else if let generalModel = generalYoloModel {
            // Only general model available
            let request = VNCoreMLRequest(model: generalModel) { [weak self] request, error in
                if let error = error {
                    print("❌ YOLO error: \(error)")
                    return
                }
                
                self?.processYOLOResults(request.results, originalImage: pixelBuffer, isCustomModel: false)
            }
            
            request.imageCropAndScaleOption = .scaleFill
            
            let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
            
            do {
                try handler.perform([request])
            } catch {
                print("❌ Inference failed: \(error)")
            }
        }
    }
    
    @discardableResult
    private func processYOLOResults(_ results: [Any]?, originalImage: CVPixelBuffer, isCustomModel: Bool) -> [Detection] {
        guard let observations = results as? [VNCoreMLFeatureValueObservation] else {
            return []
        }
        
        var detectionOutput: MLMultiArray?
        var prototypeOutput: MLMultiArray?
        
        for observation in observations {
            if let multiArray = observation.featureValue.multiArrayValue {
                let shape = multiArray.shape
                
                if shape.count == 3 && shape[2].intValue == 8400 {
                    detectionOutput = multiArray
                } else if shape.count == 4 && shape[1].intValue == 32 && shape[2].intValue == 160 && shape[3].intValue == 160 {
                    prototypeOutput = multiArray
                }
            }
        }
        
        guard let detections = detectionOutput,
              let prototypes = prototypeOutput else {
            return []
        }
        
        let validDetections = extractDetections(from: detections)
        let nmsDetections = applyNMS(detections: validDetections, iouThreshold: 0.45)
        
        guard let bestDetection = nmsDetections.first else {
            DispatchQueue.main.async {
                self.segmentedImage = nil
                self.furnitureOpacity = 0.0
                self.lastConfidence = 0.0
                self.lastDetectedClass = ""
                self.currentBBox = .zero
            }
            return []
        }
        
        print("✅ Detected: \(bestDetection.className) (\(Int(bestDetection.confidence * 100))%) - \(isCustomModel ? "CUSTOM" : "GENERAL") model")
        
        DispatchQueue.main.async {
            self.lastDetectedClass = bestDetection.className
            self.usingCustomModel = isCustomModel
        }
        
        processAndApplyMask(detection: bestDetection,
                           prototypes: prototypes,
                           originalImage: originalImage)
        
        return nmsDetections
    }
    
    private func extractDetections(from detections: MLMultiArray) -> [Detection] {
        var allDetections: [Detection] = []
        let confThreshold: Float = 0.3
        let currentTime = Date()
        
        for anchor in 0..<8400 {
            let x = detections[[0, 0, anchor] as [NSNumber]].floatValue
            let y = detections[[0, 1, anchor] as [NSNumber]].floatValue
            let w = detections[[0, 2, anchor] as [NSNumber]].floatValue
            let h = detections[[0, 3, anchor] as [NSNumber]].floatValue
            
            for (classIdx, className) in cocoClasses {
                let conf = detections[[0, 4 + classIdx, anchor] as [NSNumber]].floatValue
                
                if conf > confThreshold {
                    var maskCoeffs = [Float](repeating: 0, count: 32)
                    for i in 0..<32 {
                        maskCoeffs[i] = detections[[0, 84 + i, anchor] as [NSNumber]].floatValue
                    }
                    
                    allDetections.append(Detection(
                        x: x, y: y, width: w, height: h,
                        confidence: conf, classIdx: classIdx,
                        className: className, maskCoeffs: maskCoeffs,
                        timestamp: currentTime
                    ))
                }
            }
        }
        
        return allDetections
    }
    
    private func applyNMS(detections: [Detection], iouThreshold: Float) -> [Detection] {
        guard !detections.isEmpty else { return [] }
        
        let sorted = detections.sorted { $0.confidence > $1.confidence }
        var kept: [Detection] = []
        var suppressed = Set<Int>()
        
        for (idx, detection) in sorted.enumerated() {
            if suppressed.contains(idx) { continue }
            
            kept.append(detection)
            
            for (otherIdx, other) in sorted.enumerated() where otherIdx > idx {
                if suppressed.contains(otherIdx) { continue }
                
                let iou = calculateIoU(detection, other)
                if iou > iouThreshold {
                    suppressed.insert(otherIdx)
                }
            }
        }
        
        return kept
    }
    
    private func calculateIoU(_ a: Detection, _ b: Detection) -> Float {
        let x1 = max(a.x - a.width/2, b.x - b.width/2)
        let y1 = max(a.y - a.height/2, b.y - b.height/2)
        let x2 = min(a.x + a.width/2, b.x + b.width/2)
        let y2 = min(a.y + a.height/2, b.y + b.height/2)
        
        let intersection = max(0, x2 - x1) * max(0, y2 - y1)
        let union = a.width * a.height + b.width * b.height - intersection
        
        return union > 0 ? intersection / union : 0
    }
    
    private func processAndApplyMask(detection: Detection,
                                    prototypes: MLMultiArray,
                                    originalImage: CVPixelBuffer) {
        
        DispatchQueue.main.async {
            self.lastConfidence = detection.confidence
        }
        
        let mask = generateMaskUltralytics(coefficients: detection.maskCoeffs,
                                          prototypes: prototypes)
        
        let positivePixels = mask.filter { $0 > 0.5 }.count
        print("✅ Mask pixels: \(positivePixels)")
        
        let width = CVPixelBufferGetWidth(originalImage)
        let height = CVPixelBufferGetHeight(originalImage)
        let imageSize = CGSize(width: width, height: height)
        
        let smoothedBBox = bboxTracker.updateBBox(newDetection: detection, imageSize: imageSize)
        
        DispatchQueue.main.async {
            self.currentBBox = smoothedBBox
        }
        
        applyMaskToImage(mask: mask, detection: detection, to: originalImage)
    }
    
    private func generateMaskUltralytics(coefficients: [Float], prototypes: MLMultiArray) -> [Float] {
        var mask = [Float](repeating: 0, count: 160 * 160)
        
        for y in 0..<160 {
            for x in 0..<160 {
                var sum: Float = 0
                for c in 0..<32 {
                    let protoValue = prototypes[[0, c, y, x] as [NSNumber]].floatValue
                    sum += coefficients[c] * protoValue
                }
                mask[y * 160 + x] = sigmoid(sum)
            }
        }
        
        return mask
    }
    
    private func fillMaskHoles(_ mask: [Float], threshold: Float = 0.3) -> [Float] {
        let width = 160
        let height = 160
        
        var binaryMask = [Bool](repeating: false, count: width * height)
        for i in 0..<(width * height) {
            binaryMask[i] = mask[i] > threshold
        }
        
        var isBackground = [Bool](repeating: false, count: width * height)
        var queue: [(Int, Int)] = []
        
        for x in 0..<width {
            queue.append((0, x))
            queue.append((height-1, x))
        }
        for y in 0..<height {
            queue.append((y, 0))
            queue.append((y, width-1))
        }
        
        while !queue.isEmpty {
            let (y, x) = queue.removeFirst()
            
            if x < 0 || x >= width || y < 0 || y >= height { continue }
            
            let idx = y * width + x
            if isBackground[idx] || binaryMask[idx] { continue }
            
            isBackground[idx] = true
            
            queue.append((y-1, x))
            queue.append((y+1, x))
            queue.append((y, x-1))
            queue.append((y, x+1))
        }
        
        var filledMask = mask
        for i in 0..<(width * height) {
            if !isBackground[i] {
                filledMask[i] = max(filledMask[i], 0.8)
            }
        }
        
        return filledMask
    }
    
    private func removeSmallComponents(_ mask: [Float], minSize: Int, threshold: Float = 0.3) -> [Float] {
        let width = 160
        let height = 160
        
        var binaryMask = [Bool](repeating: false, count: width * height)
        for i in 0..<(width * height) {
            binaryMask[i] = mask[i] > threshold
        }
        
        var visited = [Bool](repeating: false, count: width * height)
        var componentMask = [Bool](repeating: false, count: width * height)
        
        for startIdx in 0..<(width * height) {
            if visited[startIdx] || !binaryMask[startIdx] { continue }
            
            var queue: [Int] = [startIdx]
            var component: [Int] = []
            visited[startIdx] = true
            
            while !queue.isEmpty {
                let idx = queue.removeFirst()
                component.append(idx)
                
                let y = idx / width
                let x = idx % width
                
                let neighbors = [
                    (y-1, x), (y+1, x), (y, x-1), (y, x+1)
                ]
                
                for (ny, nx) in neighbors {
                    if ny < 0 || ny >= height || nx < 0 || nx >= width { continue }
                    
                    let nIdx = ny * width + nx
                    if visited[nIdx] || !binaryMask[nIdx] { continue }
                    
                    visited[nIdx] = true
                    queue.append(nIdx)
                }
            }
            
            if component.count >= minSize {
                for idx in component {
                    componentMask[idx] = true
                }
            }
        }
        
        var cleanedMask = [Float](repeating: 0, count: width * height)
        for i in 0..<(width * height) {
            if componentMask[i] {
                cleanedMask[i] = mask[i]
            }
        }
        
        return cleanedMask
    }
    
    private func smoothEdges(_ mask: [Float], kernelSize: Int = 3) -> [Float] {
        let width = 160
        let height = 160
        let radius = kernelSize / 2
        
        var smoothed = [Float](repeating: 0, count: width * height)
        
        let weights: [[Float]] = [
            [0.077847, 0.123317, 0.077847],
            [0.123317, 0.195346, 0.123317],
            [0.077847, 0.123317, 0.077847]
        ]
        
        for y in radius..<(height - radius) {
            for x in radius..<(width - radius) {
                var sum: Float = 0
                
                for dy in -radius...radius {
                    for dx in -radius...radius {
                        let maskValue = mask[(y + dy) * width + (x + dx)]
                        let weight = weights[dy + radius][dx + radius]
                        sum += maskValue * weight
                    }
                }
                
                smoothed[y * width + x] = sum
            }
        }
        
        for y in 0..<height {
            for x in 0..<width {
                if y < radius || y >= (height - radius) || x < radius || x >= (width - radius) {
                    smoothed[y * width + x] = mask[y * width + x]
                }
            }
        }
        
        return smoothed
    }
    
    private func applyMaskToImage(mask: [Float],
                                 detection: Detection,
                                 to pixelBuffer: CVPixelBuffer) {
        
        autoreleasepool {
            let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
            
            let width = CVPixelBufferGetWidth(pixelBuffer)
            let height = CVPixelBufferGetHeight(pixelBuffer)
            
            guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else {
                return
            }
            
            let colorSpace = CGColorSpaceCreateDeviceRGB()
            guard let ctx = CGContext(data: nil,
                                     width: width,
                                     height: height,
                                     bitsPerComponent: 8,
                                     bytesPerRow: width * 4,
                                     space: colorSpace,
                                     bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else {
                return
            }
            
            ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
            
            guard let data = ctx.data else {
                return
            }
            
            let pixels = data.bindMemory(to: UInt8.self, capacity: width * height * 4)
            
            let scale = Float(width) / 640.0
            
            let origX1 = Int((detection.x - detection.width/2) * scale)
            let origY1 = Int((detection.y - detection.height/2) * scale)
            let origX2 = Int((detection.x + detection.width/2) * scale)
            let origY2 = Int((detection.y + detection.height/2) * scale)
            
            let bboxHeight = origY2 - origY1
            let bboxWidth = origX2 - origX1
            
            let bottomExpansion: Float
            let topExpansion: Float
            let sideExpansion: Float
            
            switch detection.className {
            case "chair":
                bottomExpansion = 1.0
                topExpansion = 0.5
                sideExpansion = 0.8
                
            case "bed":
                bottomExpansion = 0.8
                topExpansion = 1.0
                sideExpansion = 0.8
                
            case "couch", "sofa":
                bottomExpansion = 0.7
                topExpansion = 0.6
                sideExpansion = 0.6
                
            case "dining table":
                bottomExpansion = 1.2
                topExpansion = 0.2
                sideExpansion = 0.5
                
            case "person":
                bottomExpansion = 0.3
                topExpansion = 0.3
                sideExpansion = 0.25
                
            default:
                bottomExpansion = 0.5
                topExpansion = 0.5
                sideExpansion = 0.4
            }
            
            let x1 = max(0, origX1 - Int(Float(bboxWidth) * sideExpansion))
            let y1 = max(0, origY1 - Int(Float(bboxHeight) * topExpansion))
            let x2 = min(width, origX2 + Int(Float(bboxWidth) * sideExpansion))
            let y2 = min(height, origY2 + Int(Float(bboxHeight) * bottomExpansion))
            
            var refinedMask = [Float](repeating: 0, count: 160 * 160)
            var dilated = [Float](repeating: 0, count: 160 * 160)
            var temp = mask
            
            let iterations = detection.className == "chair" ? 3 : 1
            let kernelRadius = detection.className == "chair" ? 2 : 1
            
            for _ in 0..<iterations {
                for y in kernelRadius..<(160-kernelRadius) {
                    for x in kernelRadius..<(160-kernelRadius) {
                        var maxVal: Float = temp[y * 160 + x]
                        for dy in -kernelRadius...kernelRadius {
                            for dx in -kernelRadius...kernelRadius {
                                maxVal = max(maxVal, temp[(y + dy) * 160 + (x + dx)])
                            }
                        }
                        dilated[y * 160 + x] = maxVal
                    }
                }
                
                for y in kernelRadius..<(160-kernelRadius) {
                    for x in kernelRadius..<(160-kernelRadius) {
                        var minVal: Float = dilated[y * 160 + x]
                        for dy in -kernelRadius...kernelRadius {
                            for dx in -kernelRadius...kernelRadius {
                                minVal = min(minVal, dilated[(y + dy) * 160 + (x + dx)])
                            }
                        }
                        refinedMask[y * 160 + x] = minVal
                    }
                }
                
                temp = refinedMask
            }
            
            var finalMask = refinedMask
            
            finalMask = fillMaskHoles(finalMask, threshold: 0.3)
            
            let minComponentSize = detection.className == "chair" ? 50 : 100
            finalMask = removeSmallComponents(finalMask, minSize: minComponentSize, threshold: 0.3)
            
            finalMask = smoothEdges(finalMask, kernelSize: 3)
            
            let threshold: Float = detection.className == "chair" ? 0.3 : 0.5
            
            for py in 0..<height {
                for px in 0..<width {
                    let idx = (py * width + px) * 4
                    
                    let maskX = Float(px) * 160.0 / Float(width)
                    let maskY = Float(py) * 160.0 / Float(height)
                    
                    let x0 = Int(maskX)
                    let y0 = Int(maskY)
                    let x1Val = min(x0 + 1, 159)
                    let y1Val = min(y0 + 1, 159)
                    
                    if x0 >= 0 && x0 < 160 && y0 >= 0 && y0 < 160 {
                        let dx = maskX - Float(x0)
                        let dy = maskY - Float(y0)
                        
                        let v00 = finalMask[y0 * 160 + x0]
                        let v10 = finalMask[y0 * 160 + x1Val]
                        let v01 = finalMask[y1Val * 160 + x0]
                        let v11 = finalMask[y1Val * 160 + x1Val]
                        
                        let v0 = v00 * (1.0 - dx) + v10 * dx
                        let v1 = v01 * (1.0 - dx) + v11 * dx
                        let maskValue = v0 * (1.0 - dy) + v1 * dy
                        
                        let inBbox = px >= x1 && px < x2 && py >= y1 && py < y2
                        
                        if maskValue > threshold && inBbox {
                            let alpha = maskValue
                            pixels[idx + 3] = UInt8(alpha * 255.0)
                            
                            pixels[idx] = UInt8(Float(pixels[idx]) * alpha)
                            pixels[idx + 1] = UInt8(Float(pixels[idx + 1]) * alpha)
                            pixels[idx + 2] = UInt8(Float(pixels[idx + 2]) * alpha)
                        } else {
                            pixels[idx + 3] = 0
                        }
                    } else {
                        pixels[idx + 3] = 0
                    }
                }
            }
            
            if let finalImage = ctx.makeImage() {
                let uiImage = UIImage(cgImage: finalImage, scale: 1.0, orientation: .up)
                
                DispatchQueue.main.async {
                    self.segmentedImage = uiImage
                    withAnimation(.easeIn(duration: 0.05)) {
                        self.furnitureOpacity = 1.0
                    }
                    
                    if self.isInitializing {
                        self.isInitializing = false
                        self.initProgress = 1.0
                        self.initStage = "Ready!"
                    }
                }
            }
        }
    }
}

extension FurnitureSegmentationModel: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        processWithDualModels(pixelBuffer: pixelBuffer)
    }
}
