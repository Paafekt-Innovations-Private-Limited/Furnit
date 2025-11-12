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
    let roomImage: UIImage? // 3D room image from previous screen
    
    @StateObject private var camera = FurnitureSegmentationModel()
    
    @State private var scaleMultiplier: CGFloat = 0.5
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
                        DragGesture()
                            .onChanged { value in dragOffset = value.translation }
                            .onEnded { value in
                                accumulatedOffset.width += value.translation.width
                                accumulatedOffset.height += value.translation.height
                                dragOffset = .zero
                            }
                    )
                    .ignoresSafeArea()
                    .opacity(camera.furnitureOpacity)
                    .animation(.easeOut(duration: 0.3), value: camera.furnitureOpacity)
            }
            
            // FPS Display
            VStack {
                HStack {
                    VStack(alignment: .leading) {
                        Text("FPS: \(camera.currentFPS, specifier: "%.1f")")
                        if camera.lastConfidence > 0 {
                            Text("Conf: \(Int(camera.lastConfidence * 100))%")
                        }
                        if !camera.lastDetectedClass.isEmpty {
                            Text("\(camera.lastDetectedClass)")
                                .font(.caption2)
                        }
                    }
                    .font(.caption)
                    .foregroundColor(.white)
                    .padding(6)
                    .background(Color.black.opacity(0.7))
                    .cornerRadius(8)
                    Spacer()
                }
                .padding()
                Spacer()
            }
            
            // Controls
            VStack {
                HStack {
                    if camera.segmentedImage != nil {
                        HStack(spacing: 6) {
                            Image(systemName: "minus.magnifyingglass")
                            Slider(value: $scaleMultiplier, in: 0.3...1.0)
                                .frame(width: 150)
                            Image(systemName: "plus.magnifyingglass")
                        }
                        .foregroundColor(.white)
                        .padding(8)
                        .background(Capsule().fill(Color.black.opacity(0.7)))
                    }
                    
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
                
                // Save and Reset buttons ONLY
                if camera.segmentedImage != nil {
                    HStack(spacing: 16) {
                        // Save - furniture + 3D room
                        Button(action: { captureFurnitureWithRoom() }) {
                            VStack {
                                Image(systemName: "square.and.arrow.down")
                                Text("Capture")
                                    .font(.caption2)
                            }
                            .foregroundColor(.white)
                            .frame(width: 60, height: 60)
                            .background(Circle().fill(Color.green))
                            .shadow(radius: 3)
                        }
                        
                        // Reset
                        Button(action: { camera.resetSegmentation() }) {
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
        
        print("📸 Creating composite with 3D room...")
        
        // CRITICAL: roomImage MUST be captured by caller BEFORE opening this sheet
        // The 3D room is hidden behind the sheet, so it CANNOT be captured from here!
        //
        // CALLER MUST DO THIS:
        // 1. Capture 3D room: let snapshot = captureRoomView()
        // 2. Pass it here: SegmentFurniture(roomImage: snapshot)
        //
        // See captureRoomView() helper function at bottom of this file
        
        guard let roomBackground = roomImage else {
            print("❌ CRITICAL: No roomImage provided by caller!")
            print("⚠️ The 3D room MUST be captured BEFORE opening this sheet")
            print("⚠️ See comments in captureFurnitureWithRoom() for instructions")
            
            saveMessage = "Error: No room provided!"
            showingSaveSuccess = true
            DispatchQueue.main.asyncAfter(deadline: .now() + 3) {
                showingSaveSuccess = false
            }
            return
        }
        
        print("✅ Using provided 3D room: \(roomBackground.size)")
        
        // Create composite WITHOUT any UI elements
        UIGraphicsBeginImageContextWithOptions(roomBackground.size, false, roomBackground.scale)
        defer { UIGraphicsEndImageContext() }
        
        // Draw 3D room background
        roomBackground.draw(at: .zero)
        
        // Calculate furniture position (matching current screen position)
        let furnitureSize = CGSize(
            width: furniture.size.width * scaleMultiplier,
            height: furniture.size.height * scaleMultiplier
        )
        
        let centerX = roomBackground.size.width / 2
        let centerY = roomBackground.size.height / 2
        
        let furnitureOrigin = CGPoint(
            x: centerX - furnitureSize.width / 2 + dragOffset.width + accumulatedOffset.width,
            y: centerY - furnitureSize.height / 2 + dragOffset.height + accumulatedOffset.height
        )
        
        // Draw furniture with transparency on top of 3D room
        furniture.draw(in: CGRect(origin: furnitureOrigin, size: furnitureSize))
        
        guard let composite = UIGraphicsGetImageFromCurrentImageContext() else {
            print("❌ Failed to create composite")
            saveMessage = "Composite failed!"
            showingSaveSuccess = true
            return
        }
        
        print("✅ Composite created: \(composite.size)")
        
        // Save to photos with proper permissions
        capturedImage = composite
        
        PHPhotoLibrary.requestAuthorization(for: .addOnly) { status in
            DispatchQueue.main.async {
                if status == .authorized || status == .limited {
                    PHPhotoLibrary.shared().performChanges({
                        PHAssetChangeRequest.creationRequestForAsset(from: composite)
                    }) { success, error in
                        DispatchQueue.main.async {
                            if success {
                                self.saveMessage = "Saved to Photos!"
                                self.showingSaveSuccess = true
                                print("✅ Composite saved: 3D room + furniture")
                                
                                DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
                                    self.showingSaveSuccess = false
                                    self.isShowingCamera = false
                                }
                            } else {
                                self.saveMessage = "Save failed!"
                                self.showingSaveSuccess = true
                                print("❌ Failed to save: \(error?.localizedDescription ?? "unknown")")
                            }
                        }
                    }
                } else {
                    self.saveMessage = "Permission denied!"
                    self.showingSaveSuccess = true
                }
            }
        }
    }

    // HELPER: This function MUST be called by the CALLER (parent view)
    // BEFORE opening SegmentFurniture sheet to capture the 3D room
    //
    // USAGE IN CALLER:
    //
    // @State private var roomSnapshot: UIImage?
    //
    // Button("Scan Furniture") {
    //     roomSnapshot = captureRoomView()  // Capture 3D room FIRST
    //     showingSegmentFurniture = true    // Then open sheet
    // }
    // .sheet(isPresented: $showingSegmentFurniture) {
    //     SegmentFurniture(
    //         capturedImage: $capturedImage,
    //         isShowingCamera: $showingSegmentFurniture,
    //         roomImage: roomSnapshot  // Pass captured room
    //     )
    // }
    //
    private func captureRoomView() -> UIImage? {
        guard let window = UIApplication.shared.connectedScenes
            .compactMap({ $0 as? UIWindowScene })
            .first?.windows
            .first(where: { $0.isKeyWindow }) else {
            return nil
        }
        
        let renderer = UIGraphicsImageRenderer(bounds: window.bounds)
        return renderer.image { context in
            window.drawHierarchy(in: window.bounds, afterScreenUpdates: false)
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
}

// MARK: - Main Model with ALL COCO Classes
class FurnitureSegmentationModel: NSObject, ObservableObject {
    @Published var segmentedImage: UIImage?
    @Published var furnitureOpacity: Double = 0.0
    @Published var currentFPS: Double = 0.0
    @Published var lastConfidence: Float = 0.0
    @Published var lastDetectedClass: String = ""
    
    let session = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let videoQueue = DispatchQueue(label: "furnitureSegQueue", qos: .userInitiated)
    
    private var yoloModel: VNCoreMLModel?
    private let context = CIContext(options: [.workingColorSpace: CGColorSpaceCreateDeviceRGB()])
    
    // ALL COCO classes (80 classes)
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
    private let processInterval: TimeInterval = 0.1
    private var frameCount = 0
    private var fpsStartTime = Date()
    
    private func sigmoid(_ x: Float) -> Float {
        return 1.0 / (1.0 + exp(-x))
    }
    
    override init() {
        super.init()
        loadYOLOModel()
        setupCamera()
    }
    
    func resetSegmentation() {
        DispatchQueue.main.async {
            self.segmentedImage = nil
            self.furnitureOpacity = 0.0
            self.lastConfidence = 0.0
            self.lastDetectedClass = ""
        }
    }
    
    private func loadYOLOModel() {
        print("🔍 Loading YOLO11-seg model...")
        
        for ext in ["mlmodelc", "mlpackage"] {
            if let modelURL = Bundle.main.url(forResource: "yolo11x-seg", withExtension: ext) {
                print("📦 Found model: yolo11x-seg.\(ext)")
                do {
                    let model = try MLModel(contentsOf: modelURL)
                    yoloModel = try VNCoreMLModel(for: model)
                    print("✅ YOLO11-seg loaded!")
                    return
                } catch {
                    print("❌ Failed: \(error)")
                }
            }
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
            DispatchQueue.global(qos: .background).async {
                self.session.startRunning()
                DispatchQueue.main.async {
                    print("✅ Camera started")
                    self.fpsStartTime = Date()
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
    
    private func processWithYOLO(pixelBuffer: CVPixelBuffer) {
        guard let model = yoloModel else { return }
        
        let now = Date()
        guard now.timeIntervalSince(lastProcessTime) >= processInterval else { return }
        
        lastProcessTime = now
        updateFPS()
        
        let request = VNCoreMLRequest(model: model) { [weak self] request, error in
            if let error = error {
                print("❌ YOLO error: \(error)")
                return
            }
            
            self?.processYOLOResults(request.results, originalImage: pixelBuffer)
        }
        
        request.imageCropAndScaleOption = .scaleFill
        
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
        
        do {
            try handler.perform([request])
        } catch {
            print("❌ Inference failed: \(error)")
        }
    }
    
    private func processYOLOResults(_ results: [Any]?, originalImage: CVPixelBuffer) {
        guard let observations = results as? [VNCoreMLFeatureValueObservation] else {
            return
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
            return
        }
        
        let validDetections = extractDetections(from: detections)
        let nmsDetections = applyNMS(detections: validDetections, iouThreshold: 0.45)
        
        guard let bestDetection = nmsDetections.first else {
            DispatchQueue.main.async {
                self.segmentedImage = nil
                self.furnitureOpacity = 0.0
                self.lastConfidence = 0.0
                self.lastDetectedClass = ""
            }
            return
        }
        
        print("✅ Detected: \(bestDetection.className) (\(Int(bestDetection.confidence * 100))%)")
        
        DispatchQueue.main.async {
            self.lastDetectedClass = bestDetection.className
        }
        
        processAndApplyMask(detection: bestDetection,
                           prototypes: prototypes,
                           originalImage: originalImage)
    }
    
    private func extractDetections(from detections: MLMultiArray) -> [Detection] {
        var allDetections: [Detection] = []
        let confThreshold: Float = 0.3
        
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
                        className: className, maskCoeffs: maskCoeffs
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
                sideExpansion = 0.5
                
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
            
            let threshold: Float = 0.5
            
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
                        
                        let v00 = mask[y0 * 160 + x0]
                        let v10 = mask[y0 * 160 + x1Val]
                        let v01 = mask[y1Val * 160 + x0]
                        let v11 = mask[y1Val * 160 + x1Val]
                        
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
                    withAnimation(.easeIn(duration: 0.3)) {
                        self.furnitureOpacity = 1.0
                    }
                }
            }
        }
    }
}

extension FurnitureSegmentationModel: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        processWithYOLO(pixelBuffer: pixelBuffer)
    }
}
