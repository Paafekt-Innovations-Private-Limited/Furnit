// SmartyPantsView.swift - Complete with fixed rotation and proper segmentation

import SwiftUI
import AVFoundation
import CoreImage
import CoreImage.CIFilterBuiltins
import CoreML
import Vision
import Accelerate

// MARK: - Main View

struct SmartyPantsView: View {
    @Binding var capturedImage: UIImage?
    @Binding var isShowingCamera: Bool
    var roomImage: UIImage?
    
    @StateObject private var cameraManager = SmartyPantsCameraManager()
    @State private var detectedFurniture: [SmartyPantsFurniture] = []
    @State private var selectedFurniture: SmartyPantsFurniture?
    @State private var isProcessing = false
    
    @State private var previewImage: UIImage?
    
    var body: some View {
        ZStack {
            // Camera feed
            SmartyPantsCameraPreview(session: cameraManager.session)
                .ignoresSafeArea()
            
            // Detection overlays
            GeometryReader { geometry in
                ForEach(Array(detectedFurniture.enumerated()), id: \.offset) { index, detection in
                    let isSelected = selectedFurniture?.id == detection.id
                    
                    SmartyPantsDetectionOverlay(
                        detection: detection,
                        isSelected: isSelected,
                        viewSize: geometry.size
                    )
                    .contentShape(Rectangle())
                    .onTapGesture {
                        print("🔵 TAP DETECTED on \(detection.className)")
                        handleFurnitureTap(detection)
                    }
                }
            }
            
            // Processing indicator
            if isProcessing {
                VStack {
                    Spacer()
                    HStack {
                        ProgressView()
                            .progressViewStyle(CircularProgressViewStyle(tint: .white))
                        Text("Segmenting furniture...")
                            .foregroundColor(.white)
                            .padding(.leading, 8)
                    }
                    .padding()
                    .background(Color.black.opacity(0.7))
                    .cornerRadius(10)
                    .padding(.bottom, 100)
                }
            }
            
            // Controls
            VStack {
                HStack {
                    closeButton
                    Spacer()
                    Text("SmartyPants 🧠")
                        .font(.headline)
                        .foregroundColor(.white)
                        .padding(.horizontal, 16)
                        .padding(.vertical, 8)
                        .background(Color.blue.opacity(0.8))
                        .cornerRadius(20)
                    Spacer()
                    if selectedFurniture != nil && previewImage == nil {
                        confirmButton
                    }
                }
                .padding()
                
                Spacer()
                
                // Info banner
                if !detectedFurniture.isEmpty && previewImage == nil {
                    VStack(spacing: 4) {
                        Text("\(detectedFurniture.count) furniture items")
                            .font(.caption)
                            .foregroundColor(.white)
                        if selectedFurniture != nil {
                            Text("Tap ✓ to segment")
                                .font(.caption2)
                                .foregroundColor(.green)
                        } else {
                            Text("Tap to select")
                                .font(.caption2)
                                .foregroundColor(.white.opacity(0.8))
                        }
                    }
                    .padding(.horizontal, 16)
                    .padding(.vertical, 8)
                    .background(Color.black.opacity(0.7))
                    .cornerRadius(10)
                    .padding(.bottom, 20)
                }
            }
        }
        .overlay {
            // Live segmentation preview overlay
            if let previewImage = previewImage {
                ZStack {
                    Color.black.opacity(0.95)
                        .ignoresSafeArea()
                    
                    VStack(spacing: 0) {
                        // Header
                        HStack {
                            Button(action: {
                                withAnimation {
                                    self.previewImage = nil
                                    self.selectedFurniture = nil
                                }
                            }) {
                                HStack {
                                    Image(systemName: "xmark")
                                    Text("Cancel")
                                }
                                .foregroundColor(.white)
                                .padding()
                            }
                            
                            Spacer()
                            
                            Text("Segmented Furniture")
                                .foregroundColor(.white)
                                .font(.headline)
                            
                            Spacer()
                            
                            Button(action: {
                                self.capturedImage = previewImage
                                self.isShowingCamera = false
                                print("💾 Saved furniture!")
                            }) {
                                HStack {
                                    Text("Save")
                                    Image(systemName: "checkmark")
                                }
                                .foregroundColor(.green)
                                .padding()
                            }
                        }
                        .background(Color.black.opacity(0.7))
                        
                        Spacer()
                        
                        // Preview image
                        Image(uiImage: previewImage)
                            .resizable()
                            .scaledToFit()
                            .padding()
                            .background(
                                // Checkered background to show transparency
                                ZStack {
                                    Color.white
                                    Color.gray.opacity(0.2)
                                        .mask(
                                            Canvas { context, size in
                                                let checkSize: CGFloat = 20
                                                for row in 0..<Int(size.height / checkSize) + 1 {
                                                    for col in 0..<Int(size.width / checkSize) + 1 {
                                                        if (row + col) % 2 == 0 {
                                                            let rect = CGRect(
                                                                x: CGFloat(col) * checkSize,
                                                                y: CGFloat(row) * checkSize,
                                                                width: checkSize,
                                                                height: checkSize
                                                            )
                                                            context.fill(Path(rect), with: .color(.black))
                                                        }
                                                    }
                                                }
                                            }
                                        )
                                }
                            )
                        
                        Spacer()
                    }
                }
                .transition(.move(edge: .bottom))
            }
        }
        .onAppear {
            cameraManager.startSession()
            cameraManager.onDetection = { detections in
                self.detectedFurniture = detections
            }
        }
        .onDisappear {
            print("🛑 SmartyPants view closing - stopping camera and detection")
            cameraManager.stopSession()
        }
    }
    
    // MARK: - UI Components
    
    private var closeButton: some View {
        Button(action: {
            isShowingCamera = false
        }) {
            Image(systemName: "xmark.circle.fill")
                .font(.system(size: 32))
                .foregroundColor(.white)
                .shadow(radius: 3)
        }
    }
    
    private var confirmButton: some View {
        Button(action: {
            confirmSelection()
        }) {
            Image(systemName: "checkmark.circle.fill")
                .font(.system(size: 32))
                .foregroundColor(.green)
                .shadow(radius: 3)
        }
    }
    
    // MARK: - Actions
    
    private func handleFurnitureTap(_ detection: SmartyPantsFurniture) {
        selectedFurniture = detection
        print("✅ SELECTED: \(detection.className) (\(Int(detection.confidence * 100))%)")
    }
    
    private func confirmSelection() {
        guard let selected = selectedFurniture else { return }
        
        isProcessing = true
        
        DispatchQueue.global(qos: .userInitiated).async {
            guard let pixelBuffer = self.cameraManager.currentFrame else {
                DispatchQueue.main.async { self.isProcessing = false }
                return
            }
            
            // Convert to UIImage
            let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
            let context = CIContext()
            
            // Try different orientations - change this if rotation is wrong
            // Options: .up, .down, .left, .right, .upMirrored, .downMirrored, .leftMirrored, .rightMirrored
            let rotated = ciImage.oriented(.up)
            
            guard let cgImage = context.createCGImage(rotated, from: rotated.extent) else {
                DispatchQueue.main.async { self.isProcessing = false }
                return
            }
            let fullImage = UIImage(cgImage: cgImage)
            
            print("🎨 Applying mask to segment furniture...")
            print("   Image size: \(fullImage.size)")
            print("   Mask size: \(selected.mask?.size ?? .zero)")
            print("   BBox: \(selected.bbox)")
            
            // Apply mask with bbox
            if let mask = selected.mask,
               let segmented = self.applyMaskToBBox(image: fullImage, mask: mask, bbox: selected.bbox) {
                
                DispatchQueue.main.async {
                    withAnimation {
                        self.previewImage = segmented
                    }
                    self.isProcessing = false
                    print("✅ Furniture segmented successfully!")
                }
            } else {
                DispatchQueue.main.async {
                    self.isProcessing = false
                    print("❌ Failed to segment furniture")
                }
            }
        }
    }
    
    private func applyMaskToBBox(image: UIImage, mask: UIImage, bbox: CGRect) -> UIImage? {
        guard let imageCG = image.cgImage else { return nil }
        
        let imageSize = CGSize(width: imageCG.width, height: imageCG.height)
        
        // Scale bbox from 640x640 to actual image size
        let scaleX = imageSize.width / 640.0
        let scaleY = imageSize.height / 640.0
        
        let scaledBbox = CGRect(
            x: bbox.origin.x * scaleX,
            y: bbox.origin.y * scaleY,
            width: bbox.width * scaleX,
            height: bbox.height * scaleY
        )
        
        print("📏 Scaled bbox: \(scaledBbox)")
        
        // Resize mask to match bbox size
        let maskSize = scaledBbox.size
        UIGraphicsBeginImageContextWithOptions(maskSize, false, 1.0)
        mask.draw(in: CGRect(origin: .zero, size: maskSize))
        guard let resizedMask = UIGraphicsGetImageFromCurrentImageContext(),
              let resizedMaskCG = resizedMask.cgImage else {
            UIGraphicsEndImageContext()
            return nil
        }
        UIGraphicsEndImageContext()
        
        // Create full-size mask image (transparent except for bbox region)
        UIGraphicsBeginImageContextWithOptions(imageSize, false, 1.0)
        guard let fullMaskContext = UIGraphicsGetCurrentContext() else { return nil }
        
        // Fill with black (transparent)
        fullMaskContext.setFillColor(UIColor.black.cgColor)
        fullMaskContext.fill(CGRect(origin: .zero, size: imageSize))
        
        // Draw resized mask at bbox location
        fullMaskContext.draw(resizedMaskCG, in: scaledBbox)
        
        guard let fullMaskImage = fullMaskContext.makeImage() else {
            UIGraphicsEndImageContext()
            return nil
        }
        UIGraphicsEndImageContext()
        
        // Apply mask to image
        UIGraphicsBeginImageContextWithOptions(imageSize, false, 1.0)
        defer { UIGraphicsEndImageContext() }
        
        guard let outputContext = UIGraphicsGetCurrentContext() else { return nil }
        
        // Clip to mask
        outputContext.clip(to: CGRect(origin: .zero, size: imageSize), mask: fullMaskImage)
        
        // Draw image
        outputContext.draw(imageCG, in: CGRect(origin: .zero, size: imageSize))
        
        return UIGraphicsGetImageFromCurrentImageContext()
    }
}

// MARK: - Detection Model

struct SmartyPantsFurniture: Identifiable {
    let id = UUID()
    let className: String
    let confidence: Float
    let bbox: CGRect
    let maskCoefficients: [Float]
    var mask: UIImage?
}

// MARK: - Detection Overlay

struct SmartyPantsDetectionOverlay: View {
    let detection: SmartyPantsFurniture
    let isSelected: Bool
    let viewSize: CGSize
    
    var body: some View {
        let scaledBbox = scaleBbox(detection.bbox, to: viewSize)
        
        ZStack(alignment: .topLeading) {
            // Invisible tappable background
            Rectangle()
                .fill(Color.white.opacity(0.001))
                .frame(width: scaledBbox.width, height: scaledBbox.height)
                .position(x: scaledBbox.midX, y: scaledBbox.midY)
            
            // Border
            Rectangle()
                .strokeBorder(isSelected ? Color.green : Color.blue, lineWidth: isSelected ? 3 : 2)
                .frame(width: scaledBbox.width, height: scaledBbox.height)
                .position(x: scaledBbox.midX, y: scaledBbox.midY)
            
            // Label
            Text("\(detection.className) \(Int(detection.confidence * 100))%")
                .font(.caption)
                .foregroundColor(.white)
                .padding(.horizontal, 8)
                .padding(.vertical, 4)
                .background(isSelected ? Color.green : Color.blue)
                .cornerRadius(8)
                .position(x: scaledBbox.midX, y: scaledBbox.minY - 12)
        }
    }
    
    private func scaleBbox(_ bbox: CGRect, to viewSize: CGSize) -> CGRect {
        let scaleX = viewSize.width / CGFloat(640)
        let scaleY = viewSize.height / CGFloat(640)
        
        return CGRect(
            x: bbox.origin.x * scaleX,
            y: bbox.origin.y * scaleY,
            width: bbox.width * scaleX,
            height: bbox.height * scaleY
        )
    }
}

// MARK: - Camera Manager with YOLOE Detection

class SmartyPantsCameraManager: NSObject, ObservableObject {
    let session = AVCaptureSession()
    var currentFrame: CVPixelBuffer?
    var onDetection: (([SmartyPantsFurniture]) -> Void)?
    
    private let videoOutput = AVCaptureVideoDataOutput()
    private let sessionQueue = DispatchQueue(label: "smartypants.camera")
    private let detectionQueue = DispatchQueue(label: "smartypants.detection")
    
    private var model: MLModel?
    private let confThreshold: Float = 0.5
    private let iouThreshold: Float = 0.45
    
    private var detectionTimer: Timer?
    private var isRunning = false
    
    // 97 furniture classes
    private let furnitureClasses: [Int: String] = [
        132: "armchair", 213: "baby seat", 274: "bar", 276: "bar stool",
        332: "bathroom cabinet", 334: "bathroom mirror", 352: "beach chair",
        364: "bean bag chair", 375: "bed", 376: "bedcover", 377: "bed frame",
        382: "bedside lamp", 402: "bench", 429: "billiard table",
        517: "bookshelf", 567: "chest", 632: "bunk bed", 636: "bureau",
        670: "cabinet", 679: "cake stand", 714: "canopy bed", 733: "car seat",
        821: "chair", 823: "daybed", 834: "changing table",
        977: "closet", 996: "coatrack", 1006: "cocktail table", 1060: "computer chair",
        1061: "computer desk", 1137: "infant bed", 1141: "couch", 1143: "counter",
        1144: "counter top", 1270: "day bed", 1301: "table", 1302: "table lamp",
        1303: "desktop", 1325: "dinning table",
        1364: "dog bed", 1396: "drawer", 1405: "dresser", 1476: "electric chair",
        1503: "side table", 1602: "feeding chair", 1624: "file cabinet",
        1721: "folding chair", 1733: "food stand",
        1750: "footrest", 1801: "fruit stand", 1816: "futon", 1885: "glass table",
        2141: "hospital bed", 2193: "ice shelf", 2219: "information desk",
        2247: "island", 2318: "kitchen cabinet", 2319: "kitchen counter",
        2322: "kitchen island", 2324: "kitchen table", 2499: "loveseat",
        2599: "mattress", 2614: "medicine cabinet", 2654: "mirror",
        2754: "music stool", 2802: "nightstand",
        2834: "office chair", 2836: "office desk", 2939: "park bench",
        3024: "church bench", 3045: "picnic table", 3061: "table tennis table",
        3145: "poker table", 3423: "rocking chair",
        3449: "round table", 3584: "seat", 3621: "shelf", 3678: "side cabinet",
        3812: "spice rack", 3862: "stand", 3888: "step stool", 3909: "stool",
        4004: "supermarket shelf", 4015: "sushi bar", 4041: "swivel chair",
        4055: "table top", 4056: "tablecloth",
        4179: "toilet seat", 4213: "towel bar",
        4294: "tv cabinet", 4331: "vanity", 4473: "wheelchair",
        4506: "window seat", 4513: "wine cabinet", 4516: "wine rack",
        4545: "workbench", 4564: "writing desk"
    ]
    
    override init() {
        super.init()
        setupCamera()
        loadModel()
    }
    
    deinit {
        print("🧹 SmartyPantsCameraManager deallocated")
        stopSession()
    }
    
    private func loadModel() {
        do {
            let config = MLModelConfiguration()
            config.computeUnits = .all
            
            if let modelURL = Bundle.main.url(forResource: "yoloe-11l-seg-pf", withExtension: "mlmodelc") {
                model = try MLModel(contentsOf: modelURL, configuration: config)
                print("✅ SmartyPants: Model loaded!")
                print("📥 Model inputs:")
                model?.modelDescription.inputDescriptionsByName.forEach { name, desc in
                    print("   - \(name): \(desc)")
                }
                print("📤 Model outputs:")
                model?.modelDescription.outputDescriptionsByName.forEach { name, desc in
                    print("   - \(name): \(desc)")
                }
                print("📊 Monitoring \(furnitureClasses.count) furniture classes")
            }
        } catch {
            print("❌ Failed to load: \(error)")
        }
    }
    
    private func setupCamera() {
        session.sessionPreset = .photo
        
        guard let camera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
              let input = try? AVCaptureDeviceInput(device: camera) else {
            return
        }
        
        if session.canAddInput(input) {
            session.addInput(input)
        }
        
        videoOutput.setSampleBufferDelegate(self, queue: sessionQueue)
        videoOutput.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        
        if session.canAddOutput(videoOutput) {
            session.addOutput(videoOutput)
        }
    }
    
    func startSession() {
        guard !isRunning else { return }
        isRunning = true
        
        sessionQueue.async { [weak self] in
            self?.session.startRunning()
        }
        startDetectionLoop()
        print("🎥 Camera session started")
    }
    
    func stopSession() {
        guard isRunning else { return }
        isRunning = false
        
        print("🛑 Stopping camera session and detection timer...")
        
        // Stop timer on main thread
        DispatchQueue.main.async { [weak self] in
            self?.detectionTimer?.invalidate()
            self?.detectionTimer = nil
            print("⏹️ Detection timer stopped")
        }
        
        // Stop camera session
        sessionQueue.async { [weak self] in
            self?.session.stopRunning()
            print("📷 Camera session stopped")
        }
    }
    
    private func startDetectionLoop() {
        DispatchQueue.main.async { [weak self] in
            guard let self = self, self.isRunning else { return }
            
            self.detectionTimer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { [weak self] timer in
                guard let self = self, self.isRunning else {
                    timer.invalidate()
                    return
                }
                self.detectFurniture()
            }
            print("⏱️ Detection timer started")
        }
    }
    
    private func detectFurniture() {
        guard isRunning,
              let model = model,
              let pixelBuffer = currentFrame else { return }
        
        detectionQueue.async { [weak self] in
            guard let self = self, self.isRunning else { return }
            
            guard let resized = self.resizePixelBuffer(pixelBuffer, width: 640, height: 640) else {
                return
            }
            
            guard let inputArray = self.pixelBufferToMultiArray(resized) else {
                return
            }
            
            let inputDict: [String: Any] = ["image": inputArray]
            
            guard let inputProvider = try? MLDictionaryFeatureProvider(dictionary: inputDict),
                  let output = try? model.prediction(from: inputProvider),
                  let detectionsArray = output.featureValue(for: "var_2421")?.multiArrayValue,
                  let prototypesArray = output.featureValue(for: "p")?.multiArrayValue else {
                return
            }
            
            print("✅ Prediction success!")
            print("   Detections shape: \(detectionsArray.shape)")
            print("   Prototypes shape: \(prototypesArray.shape)")
            
            let detections = self.parseDetections(detectionsArray, prototypes: prototypesArray)
            
            DispatchQueue.main.async { [weak self] in
                guard let self = self, self.isRunning else { return }
                self.onDetection?(detections)
            }
        }
    }
    
    private func parseDetections(_ detections: MLMultiArray, prototypes: MLMultiArray) -> [SmartyPantsFurniture] {
        var results: [SmartyPantsFurniture] = []
        
        let anchors = detections.shape[2].intValue
        
        print("📊 Processing \(anchors) anchors...")
        
        for anchor in 0..<anchors {
            let x = detections[[0, 0, anchor] as [NSNumber]].floatValue
            let y = detections[[0, 1, anchor] as [NSNumber]].floatValue
            let w = detections[[0, 2, anchor] as [NSNumber]].floatValue
            let h = detections[[0, 3, anchor] as [NSNumber]].floatValue
            
            for (classIdx, className) in furnitureClasses {
                let conf = detections[[0, 4 + classIdx, anchor] as [NSNumber]].floatValue
                
                if conf > confThreshold {
                    var maskCoeffs = [Float](repeating: 0, count: 32)
                    for i in 0..<32 {
                        maskCoeffs[i] = detections[[0, 4 + 4585 + i, anchor] as [NSNumber]].floatValue
                    }
                    
                    let bbox = CGRect(
                        x: CGFloat((x - w/2)),
                        y: CGFloat((y - h/2)),
                        width: CGFloat(w),
                        height: CGFloat(h)
                    )
                    
                    results.append(SmartyPantsFurniture(
                        className: className,
                        confidence: conf,
                        bbox: bbox,
                        maskCoefficients: maskCoeffs
                    ))
                }
            }
        }
        
        print("🎯 Found \(results.count) furniture detections before NMS")
        
        let nmsResults = applyNMS(detections: results)
        
        print("✅ After NMS: \(nmsResults.count) furniture items")
        
        return nmsResults.map { detection in
            let mask = generateMask(coefficients: detection.maskCoefficients, prototypes: prototypes)
            var updated = detection
            updated.mask = mask
            return updated
        }
    }
    
    private func generateMask(coefficients: [Float], prototypes: MLMultiArray) -> UIImage? {
        let protoHeight = 160
        let protoWidth = 160
        let numProtos = 32
        
        var maskData = [Float](repeating: 0, count: protoHeight * protoWidth)
        
        for h in 0..<protoHeight {
            for w in 0..<protoWidth {
                var sum: Float = 0
                for p in 0..<numProtos {
                    let protoValue = prototypes[[0, p, h, w] as [NSNumber]].floatValue
                    sum += coefficients[p] * protoValue
                }
                maskData[h * protoWidth + w] = 1.0 / (1.0 + exp(-sum))
            }
        }
        
        var pixels = maskData.map { UInt8(min(max($0, 0), 1) * 255) }
        
        guard let cgImage = CGImage(
            width: protoWidth,
            height: protoHeight,
            bitsPerComponent: 8,
            bitsPerPixel: 8,
            bytesPerRow: protoWidth,
            space: CGColorSpaceCreateDeviceGray(),
            bitmapInfo: [],
            provider: CGDataProvider(data: Data(bytes: &pixels, count: pixels.count) as CFData)!,
            decode: nil,
            shouldInterpolate: false,
            intent: .defaultIntent
        ) else {
            return nil
        }
        
        return UIImage(cgImage: cgImage)
    }
    
    private func applyNMS(detections: [SmartyPantsFurniture]) -> [SmartyPantsFurniture] {
        var results: [SmartyPantsFurniture] = []
        var sorted = detections.sorted { $0.confidence > $1.confidence }
        
        while !sorted.isEmpty {
            let best = sorted.removeFirst()
            results.append(best)
            
            sorted = sorted.filter { candidate in
                let iou = calculateIOU(best.bbox, candidate.bbox)
                return iou < iouThreshold || best.className != candidate.className
            }
        }
        
        return results
    }
    
    private func calculateIOU(_ box1: CGRect, _ box2: CGRect) -> Float {
        let intersection = box1.intersection(box2)
        if intersection.isNull { return 0 }
        
        let intersectionArea = intersection.width * intersection.height
        let unionArea = box1.width * box1.height + box2.width * box2.height - intersectionArea
        
        return Float(intersectionArea / unionArea)
    }
    
    private func pixelBufferToMultiArray(_ pixelBuffer: CVPixelBuffer) -> MLMultiArray? {
        let width = 640
        let height = 640
        
        guard let array = try? MLMultiArray(shape: [1, 3, 640, 640] as [NSNumber], dataType: .float16) else {
            return nil
        }
        
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        
        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            return nil
        }
        
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let buffer = baseAddress.assumingMemoryBound(to: UInt8.self)
        
        for y in 0..<height {
            for x in 0..<width {
                let pixelIndex = y * bytesPerRow + x * 4
                
                let b = Float(buffer[pixelIndex]) / 255.0
                let g = Float(buffer[pixelIndex + 1]) / 255.0
                let r = Float(buffer[pixelIndex + 2]) / 255.0
                
                array[[0, 0, y, x] as [NSNumber]] = NSNumber(value: r)
                array[[0, 1, y, x] as [NSNumber]] = NSNumber(value: g)
                array[[0, 2, y, x] as [NSNumber]] = NSNumber(value: b)
            }
        }
        
        return array
    }
    
    private func resizePixelBuffer(_ pixelBuffer: CVPixelBuffer, width: Int, height: Int) -> CVPixelBuffer? {
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let scaleX = CGFloat(width) / CGFloat(CVPixelBufferGetWidth(pixelBuffer))
        let scaleY = CGFloat(height) / CGFloat(CVPixelBufferGetHeight(pixelBuffer))
        let scaledImage = ciImage.transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))
        
        var newPixelBuffer: CVPixelBuffer?
        CVPixelBufferCreate(kCFAllocatorDefault, width, height,
                           kCVPixelFormatType_32BGRA, nil, &newPixelBuffer)
        
        guard let outputBuffer = newPixelBuffer else { return nil }
        
        let context = CIContext()
        context.render(scaledImage, to: outputBuffer)
        
        return outputBuffer
    }
}

extension SmartyPantsCameraManager: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard isRunning else { return }
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        currentFrame = pixelBuffer
    }
}

// MARK: - Camera Preview

struct SmartyPantsCameraPreview: UIViewRepresentable {
    let session: AVCaptureSession
    
    func makeUIView(context: Context) -> UIView {
        let view = UIView(frame: .zero)
        let previewLayer = AVCaptureVideoPreviewLayer(session: session)
        previewLayer.videoGravity = .resizeAspectFill
        view.layer.addSublayer(previewLayer)
        context.coordinator.previewLayer = previewLayer
        return view
    }
    
    func updateUIView(_ uiView: UIView, context: Context) {
        DispatchQueue.main.async {
            context.coordinator.previewLayer?.frame = uiView.bounds
        }
    }
    
    func makeCoordinator() -> Coordinator {
        Coordinator()
    }
    
    class Coordinator {
        var previewLayer: AVCaptureVideoPreviewLayer?
    }
}
