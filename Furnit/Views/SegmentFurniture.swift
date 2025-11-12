import SwiftUI
import AVFoundation
import CoreML
import Vision
import CoreImage
import Photos
import Accelerate

struct SegmentFurniture: View {
    @Binding var capturedImage: UIImage?
    @Binding var isShowingCamera: Bool
    @StateObject private var camera = FurnitureSegmentationModel()
    
    @State private var scaleMultiplier: CGFloat = 0.5
    @State private var dragOffset: CGSize = .zero
    @State private var accumulatedOffset: CGSize = .zero
    @State private var showingSaveSuccess = false
    
    var body: some View {
        ZStack {
            Color.clear.ignoresSafeArea()
            
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
            
            // Show FPS and detection info
            VStack {
                HStack {
                    VStack(alignment: .leading) {
                        Text("FPS: \(camera.currentFPS, specifier: "%.1f")")
                        if camera.lastConfidence > 0 {
                            Text("Conf: \(Int(camera.lastConfidence * 100))%")
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
            
            if showingSaveSuccess {
                VStack {
                    Spacer()
                    HStack(spacing: 12) {
                        Image(systemName: "checkmark.circle.fill")
                            .font(.title2)
                        Text("Furniture saved!")
                            .font(.headline)
                    }
                    .foregroundColor(.white)
                    .padding(.horizontal, 24)
                    .padding(.vertical, 16)
                    .background(
                        Capsule()
                            .fill(Color.green.opacity(0.95))
                            .shadow(radius: 10)
                    )
                    .padding(.bottom, 150)
                }
                .transition(.move(edge: .bottom).combined(with: .opacity))
            }
            
            VStack {
                HStack {
                    if camera.segmentedImage != nil {
                        HStack(spacing: 6) {
                            Image(systemName: "minus.magnifyingglass")
                                .foregroundColor(.white)
                                .font(.system(size: 14))
                            Slider(value: $scaleMultiplier, in: 0.3...1.0)
                                .frame(width: 150)
                                .accentColor(.white)
                            Image(systemName: "plus.magnifyingglass")
                                .foregroundColor(.white)
                                .font(.system(size: 14))
                        }
                        .padding(.horizontal, 12)
                        .padding(.vertical, 8)
                        .background(Capsule().fill(Color.black.opacity(0.7)))
                        .padding(.leading, 16)
                    }
                    
                    Spacer()
                    
                    Button(action: { isShowingCamera = false }) {
                        Image(systemName: "xmark.circle.fill")
                            .font(.title)
                            .foregroundColor(.white)
                            .background(Circle().fill(Color.black.opacity(0.5)))
                    }
                    .padding(.trailing, 16)
                }
                .padding(.top, 60)
                
                Spacer()
                
                HStack(spacing: 16) {
                    if camera.segmentedImage != nil {
                        Button(action: { saveFurniture() }) {
                            HStack(spacing: 8) {
                                Image(systemName: "square.and.arrow.down.fill")
                                    .font(.title3)
                                Text("Save")
                                    .font(.headline)
                            }
                            .foregroundColor(.white)
                            .padding(.horizontal, 24)
                            .padding(.vertical, 12)
                            .background(Capsule().fill(Color.green.opacity(0.9)))
                        }
                        
                        Button(action: {
                            camera.resetSegmentation()
                        }) {
                            HStack(spacing: 8) {
                                Image(systemName: "arrow.counterclockwise")
                                    .font(.title3)
                                Text("Retry")
                                    .font(.headline)
                            }
                            .foregroundColor(.white)
                            .padding(.horizontal, 24)
                            .padding(.vertical, 12)
                            .background(Capsule().fill(Color.orange.opacity(0.9)))
                        }
                    }
                    
                    Button(action: { isShowingCamera = false }) {
                        HStack(spacing: 8) {
                            Image(systemName: "xmark.circle.fill")
                                .font(.title3)
                            Text("Cancel")
                                .font(.headline)
                        }
                        .foregroundColor(.white)
                        .padding(.horizontal, 24)
                        .padding(.vertical, 12)
                        .background(Capsule().fill(Color.gray.opacity(0.9)))
                    }
                }
                .padding(.bottom, 50)
                .padding(.horizontal)
            }
        }
        .onAppear {
            AVCaptureDevice.requestAccess(for: .video) { granted in
                if granted {
                    camera.startSession()
                }
            }
        }
        .onDisappear {
            camera.stopSession()
        }
    }
    
    private func saveFurniture() {
        guard let image = camera.segmentedImage else { return }
        capturedImage = image
        
        PHPhotoLibrary.requestAuthorization(for: .addOnly) { status in
            DispatchQueue.main.async {
                if status == .authorized || status == .limited {
                    PHPhotoLibrary.shared().performChanges({
                        PHAssetChangeRequest.creationRequestForAsset(from: image)
                    }) { success, _ in
                        DispatchQueue.main.async {
                            if success {
                                withAnimation { showingSaveSuccess = true }
                                UINotificationFeedbackGenerator().notificationOccurred(.success)
                                DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) {
                                    withAnimation { showingSaveSuccess = false }
                                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                                        isShowingCamera = false
                                    }
                                }
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
}

// MARK: - YOLO11-Seg Complete Implementation with Debug
class FurnitureSegmentationModel: NSObject, ObservableObject {
    @Published var segmentedImage: UIImage?
    @Published var furnitureOpacity: Double = 0.0
    @Published var isProcessing = false
    @Published var detectedFurnitureTypes: [String] = []
    @Published var currentFPS: Double = 0.0
    @Published var lastConfidence: Float = 0.0
    
    private let session = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let videoQueue = DispatchQueue(label: "furnitureSegQueue", qos: .userInitiated)
    
    private var yoloModel: VNCoreMLModel?
    private let context = CIContext(options: [.workingColorSpace: CGColorSpaceCreateDeviceRGB()])
    
    private let furnitureClasses = [
        56: "chair",
        57: "couch",
        59: "bed",
        60: "dining table",
        61: "toilet"
    ]
    
    // Frame rate control
    private var lastProcessTime = Date()
    private let processInterval: TimeInterval = 0.1  // 10 FPS
    private var frameCount = 0
    private var fpsStartTime = Date()
    
    // Debug flags
    private var hasDebuggedPrototypes = false
    
    override init() {
        super.init()
        loadYOLOModel()
        setupCamera()
    }
    
    func resetSegmentation() {
        DispatchQueue.main.async {
            self.segmentedImage = nil
            self.furnitureOpacity = 0.0
            self.detectedFurnitureTypes = []
            self.lastConfidence = 0.0
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
                    print("✅ YOLO11-seg loaded successfully!")
                    return
                } catch {
                    print("⚠️ Failed to load: \(error)")
                }
            }
        }
    }
    
    private func setupCamera() {
        session.beginConfiguration()
        session.sessionPreset = .hd1280x720
        
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else {
            print("❌ No camera available")
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
            
            session.commitConfiguration()
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
        guard !isProcessing else { return }
        
        lastProcessTime = now
        updateFPS()
        
        DispatchQueue.main.async {
            self.isProcessing = true
        }
        
        let request = VNCoreMLRequest(model: model) { [weak self] request, error in
            if let error = error {
                print("❌ YOLO error: \(error)")
                DispatchQueue.main.async {
                    self?.isProcessing = false
                }
                return
            }
            
            self?.processYOLOResults(request.results, originalImage: pixelBuffer)
        }
        
        request.imageCropAndScaleOption = .scaleFill
        
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
        
        do {
            try handler.perform([request])
        } catch {
            print("❌ Failed to perform YOLO inference: \(error)")
            DispatchQueue.main.async {
                self.isProcessing = false
            }
        }
    }
    
    private func processYOLOResults(_ results: [Any]?, originalImage: CVPixelBuffer) {
        guard let observations = results as? [VNCoreMLFeatureValueObservation] else {
            DispatchQueue.main.async {
                self.isProcessing = false
            }
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
            DispatchQueue.main.async {
                self.isProcessing = false
            }
            return
        }
        
        // Debug prototypes once
        if !hasDebuggedPrototypes {
            debugPrototypes(prototypes)
            hasDebuggedPrototypes = true
        }
        
        // Extract all valid detections
        let validDetections = extractDetections(from: detections)
        
        // Apply NMS
        let nmsDetections = applyNMS(detections: validDetections, iouThreshold: 0.45)
        
        guard let bestDetection = nmsDetections.first else {
            DispatchQueue.main.async {
                self.isProcessing = false
                self.segmentedImage = nil
                self.furnitureOpacity = 0.0
                self.lastConfidence = 0.0
            }
            return
        }
        
        print("🪑 After NMS: \(bestDetection.className) (\(Int(bestDetection.confidence * 100))%)")
        
        // Process mask with all approaches
        processMaskComprehensive(detection: bestDetection,
                                prototypes: prototypes,
                                originalImage: originalImage)
    }
    
    // MARK: - Debug Prototypes
    private func debugPrototypes(_ prototypes: MLMultiArray) {
        print("\n🔍 ========== DEBUGGING PROTOTYPES ==========")
        print("📏 Shape: \(prototypes.shape)")
        print("📊 Strides: \(prototypes.strides)")
        
        // Analyze each prototype channel
        var channelStats: [(channel: Int, min: Float, max: Float, mean: Float, variance: Float)] = []
        
        for channel in 0..<32 {
            var values: [Float] = []
            
            for y in 0..<160 {
                for x in 0..<160 {
                    let val = prototypes[[0, channel, y, x] as [NSNumber]].floatValue
                    values.append(val)
                }
            }
            
            let min = values.min() ?? 0
            let max = values.max() ?? 0
            let mean = values.reduce(0, +) / Float(values.count)
            let variance = values.reduce(0) { $0 + pow($1 - mean, 2) } / Float(values.count)
            
            channelStats.append((channel, min, max, mean, variance))
            print("  Ch \(String(format: "%2d", channel)): min=\(String(format: "%7.3f", min)), max=\(String(format: "%7.3f", max)), mean=\(String(format: "%7.3f", mean)), var=\(String(format: "%7.3f", variance))")
        }
        
        // Check for patterns
        print("\n🔬 Pattern Analysis:")
        
        // Check if channels have similar ranges (might indicate proper normalization)
        let meanRange = channelStats.map { $0.max - $0.min }.reduce(0, +) / Float(channelStats.count)
        print("  Average range: \(meanRange)")
        
        // Check for dead channels
        let deadChannels = channelStats.filter { abs($0.variance) < 0.001 }
        if !deadChannels.isEmpty {
            print("  ⚠️ Dead channels (low variance): \(deadChannels.map { $0.channel })")
        }
        
        // Check spatial structure of first channel
        print("\n📐 Spatial Structure (Channel 0 center region):")
        for y in 75..<85 {
            var row = "  "
            for x in 75..<85 {
                let val = prototypes[[0, 0, y, x] as [NSNumber]].floatValue
                row += val > 0 ? "+" : "-"
            }
            print(row)
        }
        
        print("========== END DEBUG ==========\n")
    }
    
    // MARK: - Extract All Detections
    private func extractDetections(from detections: MLMultiArray) -> [Detection] {
        var allDetections: [Detection] = []
        let confThreshold: Float = 0.5
        
        for anchor in 0..<8400 {
            let x = detections[[0, 0, anchor] as [NSNumber]].floatValue
            let y = detections[[0, 1, anchor] as [NSNumber]].floatValue
            let w = detections[[0, 2, anchor] as [NSNumber]].floatValue
            let h = detections[[0, 3, anchor] as [NSNumber]].floatValue
            
            for (classIdx, className) in furnitureClasses {
                let conf = detections[[0, 4 + classIdx, anchor] as [NSNumber]].floatValue
                
                if conf > confThreshold {
                    var maskCoeffs = [Float](repeating: 0, count: 32)
                    for i in 0..<32 {
                        maskCoeffs[i] = detections[[0, 84 + i, anchor] as [NSNumber]].floatValue
                    }
                    
                    allDetections.append(Detection(
                        x: x, y: y, width: w, height: h,
                        confidence: conf,
                        classIdx: classIdx,
                        className: className,
                        maskCoeffs: maskCoeffs
                    ))
                }
            }
        }
        
        print("📊 Found \(allDetections.count) detections before NMS")
        return allDetections
    }
    
    // MARK: - Non-Maximum Suppression
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
        
        print("📊 After NMS: \(kept.count) detections")
        return kept
    }
    
    private func calculateIoU(_ a: Detection, _ b: Detection) -> Float {
        let x1 = max(a.x - a.width/2, b.x - b.width/2)
        let y1 = max(a.y - a.height/2, b.y - b.height/2)
        let x2 = min(a.x + a.width/2, b.x + b.width/2)
        let y2 = min(a.y + a.height/2, b.y + b.height/2)
        
        let intersection = max(0, x2 - x1) * max(0, y2 - y1)
        let areaA = a.width * a.height
        let areaB = b.width * b.height
        let union = areaA + areaB - intersection
        
        return union > 0 ? intersection / union : 0
    }
    
    // MARK: - Comprehensive Mask Processing (Try Multiple Approaches)
    private func processMaskComprehensive(detection: Detection,
                                         prototypes: MLMultiArray,
                                         originalImage: CVPixelBuffer) {
        
        DispatchQueue.main.async {
            self.lastConfidence = detection.confidence
        }
        
        print("\n🧪 Trying multiple mask generation approaches...")
        
        // Debug coefficients
        let coeffMin = detection.maskCoeffs.min() ?? 0
        let coeffMax = detection.maskCoeffs.max() ?? 0
        let coeffMean = detection.maskCoeffs.reduce(0, +) / Float(detection.maskCoeffs.count)
        print("📊 Coefficients: min=\(coeffMin), max=\(coeffMax), mean=\(coeffMean)")
        
        // Try different mask generation approaches
        var masks: [(name: String, mask: [Float])] = []
        
        // Approach 1: Standard matrix multiplication
        masks.append(("Standard", generateMaskStandard(coefficients: detection.maskCoeffs, prototypes: prototypes)))
        
        // Approach 2: Transposed prototypes
        masks.append(("Transposed", generateMaskTransposed(coefficients: detection.maskCoeffs, prototypes: prototypes)))
        
        // Approach 3: With sigmoid on prototypes first
        masks.append(("Proto-Sigmoid", generateMaskProtoSigmoid(coefficients: detection.maskCoeffs, prototypes: prototypes)))
        
        // Approach 4: Different channel ordering
        masks.append(("Reordered", generateMaskReordered(coefficients: detection.maskCoeffs, prototypes: prototypes)))
        
        // Find best mask (most pixels in reasonable range)
        var bestMask = masks[0]
        var bestScore = 0
        
        for (name, mask) in masks {
            let positive = mask.filter { $0 > 0 }.count
            let strong = mask.filter { $0 > 0.5 }.count
            let score = positive + strong  // Prefer masks with more positive values
            
            print("  \(name): positive=\(positive), strong=\(strong), score=\(score)")
            
            if score > bestScore && positive < 20000 {  // Avoid full white masks
                bestScore = score
                bestMask = (name, mask)
            }
        }
        
        print("✅ Using approach: \(bestMask.name)")
        
        // Apply best mask
        applyMaskFinal(mask: bestMask.mask,
                      detection: detection,
                      to: originalImage)
    }
    
    // MARK: - Different Mask Generation Approaches
    
    private func generateMaskStandard(coefficients: [Float], prototypes: MLMultiArray) -> [Float] {
        var mask = [Float](repeating: 0, count: 160 * 160)
        
        for y in 0..<160 {
            for x in 0..<160 {
                var sum: Float = 0
                for c in 0..<32 {
                    let protoValue = prototypes[[0, c, y, x] as [NSNumber]].floatValue
                    sum += coefficients[c] * protoValue
                }
                mask[y * 160 + x] = sum  // No sigmoid here
            }
        }
        
        return mask
    }
    
    private func generateMaskTransposed(coefficients: [Float], prototypes: MLMultiArray) -> [Float] {
        var mask = [Float](repeating: 0, count: 160 * 160)
        
        for y in 0..<160 {
            for x in 0..<160 {
                var sum: Float = 0
                for c in 0..<32 {
                    // Try transposed access
                    let protoValue = prototypes[[0, c, x, y] as [NSNumber]].floatValue
                    sum += coefficients[c] * protoValue
                }
                mask[y * 160 + x] = sum
            }
        }
        
        return mask
    }
    
    private func generateMaskProtoSigmoid(coefficients: [Float], prototypes: MLMultiArray) -> [Float] {
        var mask = [Float](repeating: 0, count: 160 * 160)
        
        for y in 0..<160 {
            for x in 0..<160 {
                var sum: Float = 0
                for c in 0..<32 {
                    let protoValue = prototypes[[0, c, y, x] as [NSNumber]].floatValue
                    // Apply sigmoid to prototype first
                    let sigmoidProto = 1.0 / (1.0 + exp(-protoValue))
                    sum += coefficients[c] * sigmoidProto
                }
                mask[y * 160 + x] = sum
            }
        }
        
        return mask
    }
    
    private func generateMaskReordered(coefficients: [Float], prototypes: MLMultiArray) -> [Float] {
        var mask = [Float](repeating: 0, count: 160 * 160)
        
        // Try different coefficient order (reversed)
        let reorderedCoeffs = Array(coefficients.reversed())
        
        for y in 0..<160 {
            for x in 0..<160 {
                var sum: Float = 0
                for c in 0..<32 {
                    let protoValue = prototypes[[0, c, y, x] as [NSNumber]].floatValue
                    sum += reorderedCoeffs[c] * protoValue
                }
                mask[y * 160 + x] = sum
            }
        }
        
        return mask
    }
    
    // MARK: - Apply Final Mask with Upsampling
    private func applyMaskFinal(mask: [Float],
                                detection: Detection,
                                to pixelBuffer: CVPixelBuffer) {
        
        autoreleasepool {
            let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
            
            let width = CVPixelBufferGetWidth(pixelBuffer)
            let height = CVPixelBufferGetHeight(pixelBuffer)
            
            guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else {
                DispatchQueue.main.async {
                    self.isProcessing = false
                }
                return
            }
            
            // Create context with alpha channel
            let colorSpace = CGColorSpaceCreateDeviceRGB()
            guard let ctx = CGContext(data: nil,
                                     width: width,
                                     height: height,
                                     bitsPerComponent: 8,
                                     bytesPerRow: width * 4,
                                     space: colorSpace,
                                     bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else {
                DispatchQueue.main.async {
                    self.isProcessing = false
                }
                return
            }
            
            // Draw original image
            ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
            
            // Get pixel data
            guard let data = ctx.data else {
                DispatchQueue.main.async {
                    self.isProcessing = false
                }
                return
            }
            
            let pixels = data.bindMemory(to: UInt8.self, capacity: width * height * 4)
            
            // Calculate bbox in image coordinates
            let scale = Float(width) / 640.0
            
            let bboxX = Int(detection.x * scale)
            let bboxY = Int(detection.y * scale)
            let bboxW = Int(detection.width * scale)
            let bboxH = Int(detection.height * scale)
            
            let x1 = max(0, bboxX - bboxW/2)
            let y1 = max(0, bboxY - bboxH/2)
            let x2 = min(width, bboxX + bboxW/2)
            let y2 = min(height, bboxY + bboxH/2)
            
            print("📦 Bbox: [\(x1),\(y1)] to [\(x2),\(y2)]")
            
            // Crop mask to bbox region in 160x160 space
            let maskScale = 160.0 / 640.0
            let maskBboxX = Int(detection.x * maskScale)
            let maskBboxY = Int(detection.y * maskScale)
            let maskBboxW = Int(detection.width * maskScale)
            let maskBboxH = Int(detection.height * maskScale)
            
            let maskX1 = max(0, maskBboxX - maskBboxW/2)
            let maskY1 = max(0, maskBboxY - maskBboxH/2)
            let maskX2 = min(160, maskBboxX + maskBboxW/2)
            let maskY2 = min(160, maskBboxY + maskBboxH/2)
            
            // Apply mask with bilinear interpolation
            var maskedPixels = 0
            let threshold: Float = 0.0  // Threshold at 0, not 0.5!
            
            for py in 0..<height {
                for px in 0..<width {
                    let pixelIdx = (py * width + px) * 4
                    
                    // Check if pixel is within bbox
                    if px >= x1 && px < x2 && py >= y1 && py < y2 {
                        // Map to cropped mask coordinates
                        let relX = Float(px - x1) / Float(x2 - x1)  // 0-1 in bbox
                        let relY = Float(py - y1) / Float(y2 - y1)  // 0-1 in bbox
                        
                        let maskX = Float(maskX1) + relX * Float(maskX2 - maskX1)
                        let maskY = Float(maskY1) + relY * Float(maskY2 - maskY1)
                        
                        // Bilinear interpolation
                        let x0 = Int(maskX)
                        let y0 = Int(maskY)
                        let x1 = min(x0 + 1, 159)
                        let y1 = min(y0 + 1, 159)
                        
                        if x0 >= 0 && x0 < 160 && y0 >= 0 && y0 < 160 {
                            let dx = maskX - Float(x0)
                            let dy = maskY - Float(y0)
                            
                            // Get four corner values
                            let v00 = mask[y0 * 160 + x0]
                            let v10 = mask[y0 * 160 + x1]
                            let v01 = mask[y1 * 160 + x0]
                            let v11 = mask[y1 * 160 + x1]
                            
                            // Bilinear interpolation
                            let v0 = v00 * (1 - dx) + v10 * dx
                            let v1 = v01 * (1 - dx) + v11 * dx
                            let maskValue = v0 * (1 - dy) + v1 * dy
                            
                            // Apply threshold and sigmoid for smooth edges
                            if maskValue > threshold {
                                // Use sigmoid for smooth alpha
                                let alpha = 1.0 / (1.0 + exp(-maskValue * 2))  // Scale factor for sharper edges
                                pixels[pixelIdx + 3] = UInt8(alpha * 255)
                                
                                // Pre-multiply alpha
                                if alpha < 1.0 {
                                    pixels[pixelIdx] = UInt8(Float(pixels[pixelIdx]) * alpha)
                                    pixels[pixelIdx + 1] = UInt8(Float(pixels[pixelIdx + 1]) * alpha)
                                    pixels[pixelIdx + 2] = UInt8(Float(pixels[pixelIdx + 2]) * alpha)
                                }
                                
                                maskedPixels += 1
                            } else {
                                // Make transparent
                                pixels[pixelIdx + 3] = 0
                            }
                        } else {
                            pixels[pixelIdx + 3] = 0
                        }
                    } else {
                        // Outside bbox - make transparent
                        pixels[pixelIdx + 3] = 0
                    }
                }
            }
            
            print("✅ Applied mask: \(maskedPixels) pixels kept")
            
            // Create final image
            if let finalImage = ctx.makeImage() {
                let uiImage = UIImage(cgImage: finalImage, scale: 1.0, orientation: .up)
                
                DispatchQueue.main.async {
                    self.segmentedImage = uiImage
                    self.detectedFurnitureTypes = [detection.className]
                    withAnimation(.easeIn(duration: 0.3)) {
                        self.furnitureOpacity = 1.0
                    }
                    self.isProcessing = false
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
