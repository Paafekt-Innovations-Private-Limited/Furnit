import SwiftUI
import AVFoundation
import CoreML
import Vision
import CoreImage
import Photos
import Accelerate

// MARK: - Main View (unchanged)
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
            
            // BBox Highlight Overlay - NEW!
            if camera.currentBBox != .zero && camera.segmentedImage != nil {
                Canvas { context, size in
                    let rect = Path(camera.currentBBox)
                    
                    // Outer glow
                    context.stroke(
                        rect,
                        with: .color(.green.opacity(0.3)),
                        lineWidth: 8
                    )
                    
                    // Middle glow
                    context.stroke(
                        rect,
                        with: .color(.green.opacity(0.6)),
                        lineWidth: 5
                    )
                    
                    // Sharp inner line
                    context.stroke(
                        rect,
                        with: .color(.green),
                        lineWidth: 2
                    )
                }
                .ignoresSafeArea()
                .allowsHitTesting(false)
            }
            
            // FPS Display
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
                
                HStack(spacing: 16) {
                    if camera.segmentedImage != nil {
                        Button(action: { saveFurniture() }) {
                            Label("Save", systemImage: "square.and.arrow.down.fill")
                                .foregroundColor(.white)
                                .padding(.horizontal, 20)
                                .padding(.vertical, 12)
                                .background(Capsule().fill(Color.green))
                        }
                        
                        Button(action: { camera.resetSegmentation() }) {
                            Label("Retry", systemImage: "arrow.counterclockwise")
                                .foregroundColor(.white)
                                .padding(.horizontal, 20)
                                .padding(.vertical, 12)
                                .background(Capsule().fill(Color.orange))
                        }
                    }
                }
                .padding(.bottom, 50)
            }
        }
        .onAppear { camera.startSession() }
        .onDisappear { camera.stopSession() }
    }
    
    private func saveFurniture() {
        guard let image = camera.segmentedImage else { return }
        capturedImage = image
        
        PHPhotoLibrary.requestAuthorization(for: .addOnly) { status in
            if status == .authorized || status == .limited {
                PHPhotoLibrary.shared().performChanges({
                    PHAssetChangeRequest.creationRequestForAsset(from: image)
                }) { success, _ in
                    if success {
                        DispatchQueue.main.async {
                            isShowingCamera = false
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

// MARK: - Main Model with Morphological Operations
class FurnitureSegmentationModel: NSObject, ObservableObject {
    @Published var segmentedImage: UIImage?
    @Published var furnitureOpacity: Double = 0.0
    @Published var isProcessing = false
    @Published var currentFPS: Double = 0.0
    @Published var lastConfidence: Float = 0.0
    @Published var currentBBox: CGRect = .zero  // NEW - for bbox highlight
    
    private let session = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let videoQueue = DispatchQueue(label: "furnitureSegQueue", qos: .userInitiated)
    
    private var yoloModel: VNCoreMLModel?
    private let context = CIContext(options: [.workingColorSpace: CGColorSpaceCreateDeviceRGB()])
    
    private let furnitureClasses = [
        56: "chair", 57: "couch", 59: "bed",
        60: "dining table", 61: "toilet"
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
            self.currentBBox = .zero  // Reset bbox
        }
    }
    
    private func loadYOLOModel() {
        print("🔍 Loading YOLO11-seg model...")
        
        for ext in ["mlmodelc", "mlpackage"] {
            if let modelURL = Bundle.main.url(forResource: "yolo11x-seg", withExtension: ext) {
//            if let modelURL = Bundle.main.url(forResource: "best", withExtension: ext) {
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
    
    // MARK: - Process with YOLO
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
            print("❌ Inference failed: \(error)")
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
        
        // Extract and apply NMS
        let validDetections = extractDetections(from: detections)
        let nmsDetections = applyNMS(detections: validDetections, iouThreshold: 0.45)
        
        guard let bestDetection = nmsDetections.first else {
            DispatchQueue.main.async {
                self.isProcessing = false
                self.segmentedImage = nil
                self.furnitureOpacity = 0.0
                self.lastConfidence = 0.0
                self.currentBBox = .zero
            }
            return
        }
        
        print("🪑 Detected: \(bestDetection.className) (\(Int(bestDetection.confidence * 100))%)")
        
        // Use raw YOLO bbox directly
        let bbox = CGRect(
            x: CGFloat(bestDetection.x - bestDetection.width/2),
            y: CGFloat(bestDetection.y - bestDetection.height/2),
            width: CGFloat(bestDetection.width),
            height: CGFloat(bestDetection.height)
        )
        
        DispatchQueue.main.async {
            self.currentBBox = bbox
        }
        
        processMaskWithMorphology(detection: bestDetection,
                                 prototypes: prototypes,
                                 originalImage: originalImage)
    }
    
    // MARK: - Extract Detections
    private func extractDetections(from detections: MLMultiArray) -> [Detection] {
        var allDetections: [Detection] = []
        let confThreshold: Float = 0.3

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
                        confidence: conf, classIdx: classIdx,
                        className: className, maskCoeffs: maskCoeffs
                    ))
                }
            }
        }

        return allDetections
    }
    
//    private func extractDetections(from detections: MLMultiArray) -> [Detection] {
//        var allDetections: [Detection] = []
//        let confThreshold: Float = 0.3
//        
//        // Get the number of classes from the model output
//        let numChannels = detections.shape[1].intValue  // 116 for COCO, 37 for table
//        let numClasses = numChannels - 4 - 32  // total - (x,y,w,h) - mask_coeffs
//        
//        for anchor in 0..<8400 {
//            let x = detections[[0, 0, anchor] as [NSNumber]].floatValue
//            let y = detections[[0, 1, anchor] as [NSNumber]].floatValue
//            let w = detections[[0, 2, anchor] as [NSNumber]].floatValue
//            let h = detections[[0, 3, anchor] as [NSNumber]].floatValue
//            
//            // For table model: only check class 0
//            // For COCO model: check furniture classes
//            if numClasses == 1 {
//                // Table model - only 1 class at index 0
//                let conf = detections[[0, 4, anchor] as [NSNumber]].floatValue
//                
//                if conf > confThreshold {
//                    var maskCoeffs = [Float](repeating: 0, count: 32)
//                    for i in 0..<32 {
//                        maskCoeffs[i] = detections[[0, 5 + i, anchor] as [NSNumber]].floatValue  // Changed: 5 instead of 84
//                    }
//                    
//                    allDetections.append(Detection(
//                        x: x, y: y, width: w, height: h,
//                        confidence: conf, classIdx: 0,
//                        className: "table", maskCoeffs: maskCoeffs
//                    ))
//                }
//            } else {
//                // COCO model - multiple furniture classes
//                for (classIdx, className) in furnitureClasses {
//                    let conf = detections[[0, 4 + classIdx, anchor] as [NSNumber]].floatValue
//                    
//                    if conf > confThreshold {
//                        var maskCoeffs = [Float](repeating: 0, count: 32)
//                        for i in 0..<32 {
//                            maskCoeffs[i] = detections[[0, 84 + i, anchor] as [NSNumber]].floatValue
//                        }
//                        
//                        allDetections.append(Detection(
//                            x: x, y: y, width: w, height: h,
//                            confidence: conf, classIdx: classIdx,
//                            className: className, maskCoeffs: maskCoeffs
//                        ))
//                    }
//                }
//            }
//        }
//        return allDetections
//    }
    
    // MARK: - NMS
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
    
    // MARK: - Process Mask with Morphological Operations
    private func processMaskWithMorphology(detection: Detection,
                                          prototypes: MLMultiArray,
                                          originalImage: CVPixelBuffer) {
        
        DispatchQueue.main.async {
            self.lastConfidence = detection.confidence
        }
        
        // Step 1: Generate base mask
        var mask = generateMaskUltralytics(coefficients: detection.maskCoeffs,
                                          prototypes: prototypes)
        
        // Step 2: Apply morphological operations for smoother edges
//        mask = applyMorphologicalOperations(mask: mask, width: 160, height: 160)
        
        let positivePixels = mask.filter { $0 > 0.5 }.count
        print("✅ After morphology: \(positivePixels) pixels")
        
        // Step 3: Apply to image
        applyMaskToImage(mask: mask, detection: detection, to: originalImage)
    }
    
    // MARK: - Generate Mask (Ultralytics approach)
    private func generateMaskUltralytics(coefficients: [Float], prototypes: MLMultiArray) -> [Float] {
        var mask = [Float](repeating: 0, count: 160 * 160)
        
        // Matrix multiplication
        for y in 0..<160 {
            for x in 0..<160 {
                var sum: Float = 0
                for c in 0..<32 {
                    let protoValue = prototypes[[0, c, y, x] as [NSNumber]].floatValue
                    sum += coefficients[c] * protoValue
                }
                mask[y * 160 + x] = sum
            }
        }
        
        // Apply sigmoid
        for i in 0..<mask.count {
            mask[i] = sigmoid(mask[i])
        }
        
        return mask
    }
    
    // MARK: - Morphological Operations
    private func applyMorphologicalOperations(mask: [Float], width: Int, height: Int) -> [Float] {
        var processedMask = mask
        
        // Convert to binary (0 or 1) for morphological operations
        var binaryMask = [Float](repeating: 0, count: width * height)
        for i in 0..<mask.count {
            binaryMask[i] = mask[i] > 0.5 ? 1.0 : 0.0
        }
        
        // Apply closing (dilation followed by erosion) to fill small gaps
        binaryMask = dilate(mask: binaryMask, width: width, height: height, iterations: 2)
        binaryMask = erode(mask: binaryMask, width: width, height: height, iterations: 2)
        
        // Apply opening (erosion followed by dilation) to remove small noise
        binaryMask = erode(mask: binaryMask, width: width, height: height, iterations: 1)
        binaryMask = dilate(mask: binaryMask, width: width, height: height, iterations: 1)
        
        // Blend with original mask for smoother edges
        for i in 0..<mask.count {
            processedMask[i] = binaryMask[i] * 0.3 + mask[i] * 0.7  // Weighted blend
        }
        
        return processedMask
    }
    
    // MARK: - Morphological Dilation
    private func dilate(mask: [Float], width: Int, height: Int, iterations: Int) -> [Float] {
        var result = mask
        let kernel = 3  // 3x3 kernel
        let offset = kernel / 2
        
        for _ in 0..<iterations {
            var temp = result
            
            for y in 0..<height {
                for x in 0..<width {
                    var maxVal: Float = 0
                    
                    // Check all kernel positions
                    for ky in -offset...offset {
                        for kx in -offset...offset {
                            let ny = y + ky
                            let nx = x + kx
                            
                            if ny >= 0 && ny < height && nx >= 0 && nx < width {
                                maxVal = max(maxVal, result[ny * width + nx])
                            }
                        }
                    }
                    
                    temp[y * width + x] = maxVal
                }
            }
            
            result = temp
        }
        
        return result
    }
    
    // MARK: - Morphological Erosion
    private func erode(mask: [Float], width: Int, height: Int, iterations: Int) -> [Float] {
        var result = mask
        let kernel = 3  // 3x3 kernel
        let offset = kernel / 2
        
        for _ in 0..<iterations {
            var temp = result
            
            for y in 0..<height {
                for x in 0..<width {
                    var minVal: Float = 1
                    
                    // Check all kernel positions
                    for ky in -offset...offset {
                        for kx in -offset...offset {
                            let ny = y + ky
                            let nx = x + kx
                            
                            if ny >= 0 && ny < height && nx >= 0 && nx < width {
                                minVal = min(minVal, result[ny * width + nx])
                            }
                        }
                    }
                    
                    temp[y * width + x] = minVal
                }
            }
            
            result = temp
        }
        
        return result
    }
    
    // REPLACE the entire applyMaskToImage method with this:
    private func applyMaskToImage(mask: [Float],
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
            
            ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
            
            guard let data = ctx.data else {
                DispatchQueue.main.async {
                    self.isProcessing = false
                }
                return
            }
            
            let pixels = data.bindMemory(to: UInt8.self, capacity: width * height * 4)
            
            // Calculate bbox for checking bounds only
            let scale = Float(width) / 640.0
            let x1 = max(0, Int((detection.x - detection.width/2) * scale))
            let y1 = max(0, Int((detection.y - detection.height/2) * scale))
            let x2 = min(width, Int((detection.x + detection.width/2) * scale))
            let y2 = min(height, Int((detection.y + detection.height/2) * scale))
            
            // DON'T map to cropped mask space - use FULL mask
            var maskedPixels = 0
            let threshold: Float = 0.5
            
            for py in 0..<height {
                for px in 0..<width {
                    let idx = (py * width + px) * 4
                    
                    // Map pixel to FULL 160x160 mask space (not cropped)
                    let maskX = Float(px) * 160.0 / Float(width)
                    let maskY = Float(py) * 160.0 / Float(height)
                    
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
                        let v0 = v00 * (1.0 - dx) + v10 * dx
                        let v1 = v01 * (1.0 - dx) + v11 * dx
                        let maskValue = v0 * (1.0 - dy) + v1 * dy
                        
                        // Check bbox (for validation, not cropping)
                        let inBbox = px >= x1 && px < x2 && py >= y1 && py < y2
                        
                        // Apply mask without bbox cropping
                        if maskValue > threshold {
                            // Use mask value directly for smoother alpha
                            let alpha = maskValue  // Already 0-1 from sigmoid
                            pixels[idx + 3] = UInt8(alpha * 255.0)
                            
                            // Pre-multiply alpha
                            pixels[idx] = UInt8(Float(pixels[idx]) * alpha)
                            pixels[idx + 1] = UInt8(Float(pixels[idx + 1]) * alpha)
                            pixels[idx + 2] = UInt8(Float(pixels[idx + 2]) * alpha)
                            
                            maskedPixels += 1
                        } else {
                            pixels[idx + 3] = 0
                        }
                    } else {
                        pixels[idx + 3] = 0
                    }
                }
            }
            
            print("✅ Applied: \(maskedPixels) pixels kept")
            
            if let finalImage = ctx.makeImage() {
                let uiImage = UIImage(cgImage: finalImage, scale: 1.0, orientation: .up)
                
                DispatchQueue.main.async {
                    self.segmentedImage = uiImage
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
