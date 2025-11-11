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

// MARK: - YOLO11-Seg with Proper Upsampling (Retina Masks)
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
        
        // Process mask with proper upsampling
        processMaskWithRetina(detection: bestDetection,
                             prototypes: prototypes,
                             originalImage: originalImage)
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
    
    // MARK: - Process Mask with Retina (High-Resolution)
    private func processMaskWithRetina(detection: Detection,
                                       prototypes: MLMultiArray,
                                       originalImage: CVPixelBuffer) {
        
        DispatchQueue.main.async {
            self.lastConfidence = detection.confidence
        }
        
        // 1. Generate 160x160 mask
        let lowResMask = generateMask(coefficients: detection.maskCoeffs, prototypes: prototypes)
        
        // 2. Calculate bbox in image coordinates
        let imageWidth = CVPixelBufferGetWidth(originalImage)
        let imageHeight = CVPixelBufferGetHeight(originalImage)
        
        let scale = Float(imageWidth) / 640.0
        
        let bboxX = Int(detection.x * scale)
        let bboxY = Int(detection.y * scale)
        let bboxW = Int(detection.width * scale)
        let bboxH = Int(detection.height * scale)
        
        let x1 = max(0, bboxX - bboxW/2)
        let y1 = max(0, bboxY - bboxH/2)
        let x2 = min(imageWidth, bboxX + bboxW/2)
        let y2 = min(imageHeight, bboxY + bboxH/2)
        
        let actualBboxWidth = x2 - x1
        let actualBboxHeight = y2 - y1
        
        print("📦 BBox: [\(x1), \(y1)] size: \(actualBboxWidth)x\(actualBboxHeight)")
        
        // 3. Crop mask to bbox region (from 160x160 to bbox proportion)
        let maskBbox = cropMaskToBbox(mask: lowResMask,
                                      detection: detection,
                                      maskSize: 160)
        
        // 4. Upsample cropped mask to bbox size (KEY STEP!)
        let highResMask = upsampleMask(mask: maskBbox.mask,
                                       fromSize: (maskBbox.width, maskBbox.height),
                                       toSize: (actualBboxWidth, actualBboxHeight))
        
        print("🔍 Upsampled mask from \(maskBbox.width)x\(maskBbox.height) to \(actualBboxWidth)x\(actualBboxHeight)")
        
        // 5. Apply high-res mask to image
        applyHighResMask(mask: highResMask,
                        bbox: (x1, y1, actualBboxWidth, actualBboxHeight),
                        to: originalImage,
                        className: detection.className)  // Pass className here
    }
    
    // Generate 160x160 mask
    private func generateMask(coefficients: [Float], prototypes: MLMultiArray) -> [Float] {
        var mask = [Float](repeating: 0, count: 160 * 160)
        
        for y in 0..<160 {
            for x in 0..<160 {
                var sum: Float = 0
                
                // Matrix multiplication
                for i in 0..<32 {
                    let protoValue = prototypes[[0, i, y, x] as [NSNumber]].floatValue
                    sum += coefficients[i] * protoValue
                }
                
                // Apply sigmoid AFTER the sum
                mask[y * 160 + x] = 1.0 / (1.0 + exp(-sum))
            }
        }
        
        let highConf = mask.filter { $0 > 0.5 }.count
        print("📊 Low-res mask: \(highConf)/25600 pixels (\(highConf * 100 / 25600)%)")
        
        return mask
    }
    
    // Crop mask to bbox region
    private func cropMaskToBbox(mask: [Float], detection: Detection, maskSize: Int) -> (mask: [Float], width: Int, height: Int) {
        // Convert bbox to mask coordinates (0-160)
        let scale = Float(maskSize) / 640.0
        
        let bboxX = Int(detection.x * scale)
        let bboxY = Int(detection.y * scale)
        let bboxW = Int(detection.width * scale)
        let bboxH = Int(detection.height * scale)
        
        let x1 = max(0, bboxX - bboxW/2)
        let y1 = max(0, bboxY - bboxH/2)
        let x2 = min(maskSize, bboxX + bboxW/2)
        let y2 = min(maskSize, bboxY + bboxH/2)
        
        let cropWidth = x2 - x1
        let cropHeight = y2 - y1
        
        var croppedMask = [Float](repeating: 0, count: cropWidth * cropHeight)
        
        for y in 0..<cropHeight {
            for x in 0..<cropWidth {
                let srcIdx = (y + y1) * maskSize + (x + x1)
                let dstIdx = y * cropWidth + x
                croppedMask[dstIdx] = mask[srcIdx]
            }
        }
        
        return (croppedMask, cropWidth, cropHeight)
    }
    
    // MARK: - Bilinear Upsampling (KEY FOR QUALITY!)
    private func upsampleMask(mask: [Float], fromSize: (width: Int, height: Int), toSize: (width: Int, height: Int)) -> [Float] {
        let srcWidth = fromSize.width
        let srcHeight = fromSize.height
        let dstWidth = toSize.width
        let dstHeight = toSize.height
        
        var upsampled = [Float](repeating: 0, count: dstWidth * dstHeight)
        
        let xRatio = Float(srcWidth) / Float(dstWidth)
        let yRatio = Float(srcHeight) / Float(dstHeight)
        
        for dstY in 0..<dstHeight {
            for dstX in 0..<dstWidth {
                // Find corresponding position in source
                let srcX = Float(dstX) * xRatio
                let srcY = Float(dstY) * yRatio
                
                // Get integer and fractional parts
                let x0 = Int(srcX)
                let y0 = Int(srcY)
                let x1 = min(x0 + 1, srcWidth - 1)
                let y1 = min(y0 + 1, srcHeight - 1)
                
                let dx = srcX - Float(x0)
                let dy = srcY - Float(y0)
                
                // Bilinear interpolation
                let v00 = mask[y0 * srcWidth + x0]
                let v10 = mask[y0 * srcWidth + x1]
                let v01 = mask[y1 * srcWidth + x0]
                let v11 = mask[y1 * srcWidth + x1]
                
                let v0 = v00 * (1 - dx) + v10 * dx
                let v1 = v01 * (1 - dx) + v11 * dx
                let value = v0 * (1 - dy) + v1 * dy
                
                upsampled[dstY * dstWidth + dstX] = value
            }
        }
        
        return upsampled
    }
    
    
    // MARK: - Apply High-Resolution Mask
    private func applyHighResMask(mask: [Float],
                                  bbox: (x: Int, y: Int, width: Int, height: Int),
                                  to pixelBuffer: CVPixelBuffer,
                                  className: String) {  // Add className parameter
        
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
            
            // Create context
            let colorSpace = CGColorSpaceCreateDeviceRGB()
            let bytesPerPixel = 4
            let bytesPerRow = width * bytesPerPixel
            
            guard let ctx = CGContext(data: nil,
                                     width: width,
                                     height: height,
                                     bitsPerComponent: 8,
                                     bytesPerRow: bytesPerRow,
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
            
            // Apply mask with threshold
            var maskedCount = 0
            let threshold: Float = 0.5
            
            for y in 0..<height {
                for x in 0..<width {
                    let pixelIdx = (y * width + x) * 4
                    
                    // Check if pixel is within bbox
                    if x >= bbox.x && x < (bbox.x + bbox.width) &&
                       y >= bbox.y && y < (bbox.y + bbox.height) {
                        
                        // Get mask value
                        let maskX = x - bbox.x
                        let maskY = y - bbox.y
                        let maskIdx = maskY * bbox.width + maskX
                        
                        if maskIdx >= 0 && maskIdx < mask.count {
                            let maskValue = mask[maskIdx]
                            
                            if maskValue > threshold {
                                // Keep pixel (full alpha)
                                maskedCount += 1
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
            
            print("✅ Applied high-res mask: \(maskedCount) pixels kept")
            
            // Create final image
            if let finalImage = ctx.makeImage() {
                let uiImage = UIImage(cgImage: finalImage, scale: 1.0, orientation: .up)
                
                DispatchQueue.main.async {
                    self.segmentedImage = uiImage
                    self.detectedFurnitureTypes = [className]  // Use the passed className
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
