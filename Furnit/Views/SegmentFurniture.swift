import SwiftUI
import AVFoundation
import CoreML
import Vision
import CoreImage
import Photos

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
            
            // Show FPS counter
            VStack {
                HStack {
                    Text("FPS: \(camera.currentFPS, specifier: "%.1f")")
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

// MARK: - YOLO11-Seg with Higher Frame Rate
class FurnitureSegmentationModel: NSObject, ObservableObject {
    @Published var segmentedImage: UIImage?
    @Published var furnitureOpacity: Double = 0.0
    @Published var isProcessing = false
    @Published var detectedFurnitureTypes: [String] = []
    @Published var currentFPS: Double = 0.0
    
    private let session = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let videoQueue = DispatchQueue(label: "furnitureSegQueue", qos: .userInitiated)
    
    private var yoloModel: VNCoreMLModel?
    private let context = CIContext()
    
    private let furnitureClasses = [
        56: "chair",
        57: "couch",
        59: "bed",
        60: "dining table",
        61: "toilet"
    ]
    
    // Frame rate control
    private var lastProcessTime = Date()
    private let processInterval: TimeInterval = 0.1  // 10 FPS instead of 2 FPS
    private var frameCount = 0
    private var fpsStartTime = Date()
    
    // Quality tracking
    private var bestConfidence: Float = 0
    private var confidenceHistory: [Float] = []
    private let historySize = 5
    
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
            self.bestConfidence = 0
            self.confidenceHistory.removeAll()
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
        session.sessionPreset = .hd1280x720  // HD quality
        
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
    
    private func shouldProcessFrame(confidence: Float) -> Bool {
        // Always process if we don't have a good result yet
        if segmentedImage == nil { return true }
        
        // Process if confidence is significantly better
        if confidence > bestConfidence * 1.1 { return true }
        
        // Process if confidence is stable and high
        confidenceHistory.append(confidence)
        if confidenceHistory.count > historySize {
            confidenceHistory.removeFirst()
        }
        
        if confidenceHistory.count == historySize {
            let avg = confidenceHistory.reduce(0, +) / Float(historySize)
            if avg > 0.8 && confidence > 0.75 {
                return true
            }
        }
        
        return false
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
        
        processInstanceSegmentation(detections: detections,
                                   prototypes: prototypes,
                                   originalImage: originalImage)
    }
    
    // MARK: - Dynamic Threshold Calculation
    private func calculateOtsuThreshold(_ values: [Float]) -> Float {
        let bins = 256
        var histogram = [Int](repeating: 0, count: bins)
        
        for value in values {
            let bin = min(bins - 1, Int(value * Float(bins - 1)))
            histogram[bin] += 1
        }
        
        let total = values.count
        var sum: Float = 0
        for i in 0..<bins {
            sum += Float(i) * Float(histogram[i])
        }
        
        var sumB: Float = 0
        var wB = 0
        var wF: Int
        
        var varMax: Float = 0
        var threshold: Float = 0
        
        for t in 0..<bins {
            wB += histogram[t]
            if wB == 0 { continue }
            
            wF = total - wB
            if wF == 0 { break }
            
            sumB += Float(t) * Float(histogram[t])
            
            let mB = sumB / Float(wB)
            let mF = (sum - sumB) / Float(wF)
            
            let varBetween = Float(wB) * Float(wF) * (mB - mF) * (mB - mF)
            
            if varBetween > varMax {
                varMax = varBetween
                threshold = Float(t) / Float(bins - 1)
            }
        }
        
        return threshold
    }
    
    private func calculateDynamicThreshold(mask: [Float], confidence: Float) -> Float {
        let otsuThreshold = calculateOtsuThreshold(mask)
        
        // More aggressive adjustment based on confidence
        let confAdjustment = 0.15 * (1.0 - confidence)
        
        let sorted = mask.sorted()
        let p60 = sorted[Int(Float(sorted.count) * 0.6)]
        let p70 = sorted[Int(Float(sorted.count) * 0.7)]
        
        var finalThreshold = otsuThreshold
        
        if otsuThreshold < 0.3 {
            finalThreshold = p60
        } else if otsuThreshold > 0.7 {
            finalThreshold = p70
        }
        
        finalThreshold += confAdjustment
        finalThreshold = max(0.25, min(0.75, finalThreshold))
        
        return finalThreshold
    }
    
    private func processInstanceSegmentation(detections: MLMultiArray,
                                            prototypes: MLMultiArray,
                                            originalImage: CVPixelBuffer) {
        
        let confThreshold: Float = 0.6  // Higher threshold for quality
        var bestConf: Float = 0
        var bestClass = ""
        var bestBox: (x: Float, y: Float, w: Float, h: Float)?
        var bestMaskCoeffs = [Float](repeating: 0, count: 32)
        
        // Process more anchors for better detection
        for anchor in 0..<8400 {
            let x = detections[[0, 0, anchor] as [NSNumber]].floatValue
            let y = detections[[0, 1, anchor] as [NSNumber]].floatValue
            let w = detections[[0, 2, anchor] as [NSNumber]].floatValue
            let h = detections[[0, 3, anchor] as [NSNumber]].floatValue
            
            for (classIdx, className) in furnitureClasses {
                let conf = detections[[0, 4 + classIdx, anchor] as [NSNumber]].floatValue
                
                if conf > bestConf {
                    bestConf = conf
                    bestClass = className
                    bestBox = (x, y, w, h)
                    
                    for i in 0..<32 {
                        bestMaskCoeffs[i] = detections[[0, 84 + i, anchor] as [NSNumber]].floatValue
                    }
                }
            }
        }
        
        guard bestConf > confThreshold,
              let box = bestBox else {
            DispatchQueue.main.async {
                self.isProcessing = false
                // Don't clear if we have a good result already
                if self.bestConfidence < 0.7 {
                    self.segmentedImage = nil
                    self.furnitureOpacity = 0.0
                }
            }
            return
        }
        
        // Only update if this is better than what we have
        if !shouldProcessFrame(confidence: bestConf) {
            DispatchQueue.main.async {
                self.isProcessing = false
            }
            return
        }
        
        print("🪑 Found \(bestClass) with confidence \(bestConf)")
        
        if bestConf > bestConfidence {
            bestConfidence = bestConf
        }
        
        // Generate mask with higher quality
        var mask = [Float](repeating: 0, count: 160 * 160)
        
        for y in 0..<160 {
            for x in 0..<160 {
                var value: Float = 0
                for i in 0..<32 {
                    let protoValue = prototypes[[0, i, y, x] as [NSNumber]].floatValue
                    value += bestMaskCoeffs[i] * protoValue
                }
                mask[y * 160 + x] = 1.0 / (1.0 + exp(-value))
            }
        }
        
        let threshold = calculateDynamicThreshold(mask: mask, confidence: bestConf)
        print("🎯 Dynamic threshold: \(threshold) for confidence \(bestConf)")
        
        smoothMaskWithThreshold(&mask, width: 160, height: 160, threshold: threshold)
        
        applySegmentationMask(mask: mask,
                             bbox: box,
                             to: originalImage,
                             className: bestClass)
    }
    
    // MARK: - Morphological Operations
    private func smoothMaskWithThreshold(_ mask: inout [Float], width: Int, height: Int, threshold: Float) {
        // Apply threshold
        for i in 0..<(width * height) {
            mask[i] = mask[i] > threshold ? 1.0 : 0.0
        }
        
        // Remove small islands
        removeSmallComponents(&mask, width: width, height: height, minSize: 30)
        
        // Morphological close
        dilate(&mask, width: width, height: height, iterations: 2)
        erode(&mask, width: width, height: height, iterations: 2)
        
        // Light gaussian blur
        gaussianBlur(&mask, width: width, height: height, sigma: 0.8)
        
        // Final smoothing
        for i in 0..<(width * height) {
            if mask[i] > 0.95 {
                mask[i] = 1.0
            } else if mask[i] < 0.05 {
                mask[i] = 0.0
            } else {
                mask[i] = smoothstep(0.05, 0.95, mask[i])
            }
        }
    }
    
    private func removeSmallComponents(_ mask: inout [Float], width: Int, height: Int, minSize: Int) {
        var labeled = [Int](repeating: 0, count: width * height)
        var label = 0
        
        for y in 0..<height {
            for x in 0..<width {
                let idx = y * width + x
                if mask[idx] > 0.5 && labeled[idx] == 0 {
                    label += 1
                    let size = floodFill(&labeled, mask: mask, x: x, y: y, width: width, height: height, label: label)
                    
                    if size < minSize {
                        removeComponent(&mask, labeled: labeled, width: width, height: height, label: label)
                    }
                }
            }
        }
    }
    
    private func floodFill(_ labeled: inout [Int], mask: [Float], x: Int, y: Int,
                          width: Int, height: Int, label: Int) -> Int {
        var stack = [(x, y)]
        var size = 0
        
        while !stack.isEmpty {
            let (cx, cy) = stack.removeLast()
            let idx = cy * width + cx
            
            if cx < 0 || cx >= width || cy < 0 || cy >= height { continue }
            if labeled[idx] != 0 || mask[idx] < 0.5 { continue }
            
            labeled[idx] = label
            size += 1
            
            stack.append((cx + 1, cy))
            stack.append((cx - 1, cy))
            stack.append((cx, cy + 1))
            stack.append((cx, cy - 1))
        }
        
        return size
    }
    
    private func removeComponent(_ mask: inout [Float], labeled: [Int],
                                width: Int, height: Int, label: Int) {
        for i in 0..<(width * height) {
            if labeled[i] == label {
                mask[i] = 0
            }
        }
    }
    
    private func dilate(_ mask: inout [Float], width: Int, height: Int, iterations: Int) {
        for _ in 0..<iterations {
            var temp = mask
            for y in 1..<(height-1) {
                for x in 1..<(width-1) {
                    var maxVal: Float = 0
                    for dy in -1...1 {
                        for dx in -1...1 {
                            maxVal = max(maxVal, mask[(y + dy) * width + (x + dx)])
                        }
                    }
                    temp[y * width + x] = maxVal
                }
            }
            mask = temp
        }
    }
    
    private func erode(_ mask: inout [Float], width: Int, height: Int, iterations: Int) {
        for _ in 0..<iterations {
            var temp = mask
            for y in 1..<(height-1) {
                for x in 1..<(width-1) {
                    var minVal: Float = 1.0
                    for dy in -1...1 {
                        for dx in -1...1 {
                            minVal = min(minVal, mask[(y + dy) * width + (x + dx)])
                        }
                    }
                    temp[y * width + x] = minVal
                }
            }
            mask = temp
        }
    }
    
    private func gaussianBlur(_ mask: inout [Float], width: Int, height: Int, sigma: Float) {
        let kernelSize = 5
        let halfSize = kernelSize / 2
        var kernel = [Float](repeating: 0, count: kernelSize * kernelSize)
        var sum: Float = 0
        
        for y in -halfSize...halfSize {
            for x in -halfSize...halfSize {
                let value = exp(-(Float(x*x + y*y)) / (2.0 * sigma * sigma))
                kernel[(y + halfSize) * kernelSize + (x + halfSize)] = value
                sum += value
            }
        }
        
        for i in 0..<kernel.count {
            kernel[i] /= sum
        }
        
        var temp = [Float](repeating: 0, count: width * height)
        for y in halfSize..<(height - halfSize) {
            for x in halfSize..<(width - halfSize) {
                var value: Float = 0
                for ky in -halfSize...halfSize {
                    for kx in -halfSize...halfSize {
                        value += mask[(y + ky) * width + (x + kx)] * kernel[(ky + halfSize) * kernelSize + (kx + halfSize)]
                    }
                }
                temp[y * width + x] = value
            }
        }
        mask = temp
    }
    
    private func smoothstep(_ edge0: Float, _ edge1: Float, _ x: Float) -> Float {
        let t = max(0, min(1, (x - edge0) / (edge1 - edge0)))
        return t * t * (3 - 2 * t)
    }
    
    private func applySegmentationMask(mask: [Float],
                                       bbox: (x: Float, y: Float, w: Float, h: Float),
                                       to pixelBuffer: CVPixelBuffer,
                                       className: String) {
        
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
        
        let scaleX = Float(width) / 640.0
        let scaleY = Float(height) / 640.0
        
        let x1 = Int(max(0, (bbox.x - bbox.w/2) * scaleX))
        let y1 = Int(max(0, (bbox.y - bbox.h/2) * scaleY))
        let x2 = Int(min(Float(width), (bbox.x + bbox.w/2) * scaleX))
        let y2 = Int(min(Float(height), (bbox.y + bbox.h/2) * scaleY))
        
        // Use bicubic interpolation for smoother upscaling
        for py in 0..<height {
            for px in 0..<width {
                let idx = (py * width + px) * 4
                
                if px >= x1 && px < x2 && py >= y1 && py < y2 {
                    // Bicubic interpolation
                    let fx = Float(px - x1) * 160.0 / Float(x2 - x1)
                    let fy = Float(py - y1) * 160.0 / Float(y2 - y1)
                    
                    let mx = Int(fx)
                    let my = Int(fy)
                    
                    if mx >= 0 && mx < 159 && my >= 0 && my < 159 {
                        // Bilinear interpolation for smoother edges
                        let dx = fx - Float(mx)
                        let dy = fy - Float(my)
                        
                        let v00 = mask[my * 160 + mx]
                        let v10 = mask[my * 160 + mx + 1]
                        let v01 = mask[(my + 1) * 160 + mx]
                        let v11 = mask[(my + 1) * 160 + mx + 1]
                        
                        let v0 = v00 * (1 - dx) + v10 * dx
                        let v1 = v01 * (1 - dx) + v11 * dx
                        let maskValue = v0 * (1 - dy) + v1 * dy
                        
                        let alpha = UInt8(maskValue * 255)
                        pixels[idx + 3] = alpha
                        
                        if alpha < 255 {
                            let factor = Float(alpha) / 255.0
                            pixels[idx] = UInt8(Float(pixels[idx]) * factor)
                            pixels[idx + 1] = UInt8(Float(pixels[idx + 1]) * factor)
                            pixels[idx + 2] = UInt8(Float(pixels[idx + 2]) * factor)
                        }
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
                self.detectedFurnitureTypes = [className]
                withAnimation(.easeIn(duration: 0.3)) {
                    self.furnitureOpacity = 1.0
                }
                self.isProcessing = false
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
