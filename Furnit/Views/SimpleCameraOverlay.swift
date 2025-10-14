import SwiftUI
import AVFoundation
import CoreML
import Vision
import CoreImage
import Accelerate

struct SimpleCameraOverlay: View {
    @Binding var capturedImage: UIImage?
    @Binding var isShowingCamera: Bool
    @StateObject private var camera = FastSAMProcessor()
    @State private var dragOffset = CGSize.zero
    @State private var position = CGPoint(x: UIScreen.main.bounds.width / 2, y: UIScreen.main.bounds.height / 2)
    @State private var scale: CGFloat = 1.0
    @State private var lastScaleValue: CGFloat = 1.0
    
    @State private var verticalOffset: CGFloat = 0
    private let minVerticalPosition: CGFloat = 100
    private let maxVerticalPosition: CGFloat = UIScreen.main.bounds.height - 200
    
    private let minSize = CGSize(width: 160, height: 120)
    private let maxSize = CGSize(width: UIScreen.main.bounds.width, height: UIScreen.main.bounds.height)
    private let baseSize = CGSize(width: 320, height: 240)
    
    var body: some View {
        VStack(spacing: 0) {
            // Title bar
            HStack {
                Text("FastSAM Segmentation")
                    .font(.caption)
                    .foregroundColor(.white)
                
                Spacer()
                
                Text(camera.statusMessage)
                    .font(.caption2)
                    .foregroundColor(.green)
                
                Button(action: { isShowingCamera = false }) {
                    Image(systemName: "xmark.circle.fill")
                        .font(.system(size: 18))
                        .foregroundColor(.white.opacity(0.8))
                }
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(Color.black.opacity(0.8))
            
            // Camera feed
            ZStack {
                if let segmented = camera.segmentedImage {
                    Image(uiImage: segmented)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .frame(width: currentSize.width, height: currentSize.height)
                } else {
                    CameraPreviewLayer(session: camera.session)
                        .frame(width: currentSize.width, height: currentSize.height)
                }
            }
            .frame(width: currentSize.width, height: currentSize.height)
            .background(Color.clear)
            .clipped()
            
            // Bottom controls
            HStack {
                // Confidence threshold slider
                HStack {
                    Text("Conf:")
                        .font(.caption2)
                        .foregroundColor(.white)
                    
                    Slider(value: $camera.confidenceThreshold, in: 0.1...0.9)
                        .frame(width: 100)
                        .accentColor(.green)
                    
                    Text("\(Int(camera.confidenceThreshold * 100))%")
                        .font(.caption2)
                        .foregroundColor(.white)
                        .frame(width: 35)
                }
                .padding(.leading, 8)
                
                Spacer()
                
                Button(action: {
                    if let currentImage = camera.segmentedImage {
                        capturedImage = currentImage
                        print("📸 Captured FastSAM segmented image")
                    }
                }) {
                    Image(systemName: "camera.circle.fill")
                        .font(.title2)
                        .foregroundColor(.white)
                }
                .padding(.horizontal)
            }
            .frame(height: 40)
            .background(Color.black.opacity(0.6))
        }
        .cornerRadius(8)
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(Color.white.opacity(0.3), lineWidth: 1)
        )
        .shadow(color: .black.opacity(0.5), radius: 10, x: 0, y: 5)
        .position(x: position.x + dragOffset.width, y: position.y + dragOffset.height + verticalOffset)
        .gesture(
            DragGesture()
                .onChanged { value in
                    self.dragOffset = value.translation
                }
                .onEnded { _ in
                    self.position.x += self.dragOffset.width
                    self.position.y += self.dragOffset.height
                    self.dragOffset = .zero
                }
        )
        .gesture(
            MagnificationGesture()
                .onChanged { value in
                    let delta = value / self.lastScaleValue
                    self.lastScaleValue = value
                    self.scale *= delta
                    
                    // Clamp scale
                    self.scale = min(max(self.scale, self.minScale), self.maxScale)
                }
                .onEnded { _ in
                    self.lastScaleValue = 1.0
                }
        )
        .onAppear {
            camera.start()
        }
        .onDisappear {
            camera.stop()
        }
    }
    
    private var currentSize: CGSize {
        CGSize(width: baseSize.width * scale, height: baseSize.height * scale)
    }
    
    private var minScale: CGFloat {
        min(minSize.width / baseSize.width, minSize.height / baseSize.height)
    }
    
    private var maxScale: CGFloat {
        let screenWidth = UIScreen.main.bounds.width
        let screenHeight = UIScreen.main.bounds.height
        let widthScale = screenWidth / baseSize.width
        let heightScale = screenHeight / baseSize.height
        return max(widthScale, heightScale) * 1.2
    }
}

struct CameraPreviewLayer: UIViewRepresentable {
    let session: AVCaptureSession
    
    func makeUIView(context: Context) -> UIView {
        let view = UIView()
        view.backgroundColor = .black
        
        DispatchQueue.main.async {
            let previewLayer = AVCaptureVideoPreviewLayer(session: session)
            previewLayer.videoGravity = .resizeAspectFill
            previewLayer.connection?.videoOrientation = .portrait
            view.layer.addSublayer(previewLayer)
        }
        return view
    }
    
    func updateUIView(_ uiView: UIView, context: Context) {
        if let previewLayer = uiView.layer.sublayers?.first(where: { $0 is AVCaptureVideoPreviewLayer }) as? AVCaptureVideoPreviewLayer {
            previewLayer.frame = uiView.bounds
        }
    }
}

// FastSAM Processor with Diagnostics
class FastSAMProcessor: NSObject, ObservableObject {
    let session = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let videoQueue = DispatchQueue(label: "fastSAMQueue", qos: .userInitiated)
    
    @Published var segmentedImage: UIImage?
    @Published var statusMessage = "Loading..."
    @Published var confidenceThreshold: Float = 0.3 // Lowered default
    @Published private var detectionCount = 0
    
    // The actual MLModel - NOT VNCoreMLModel
    private var fastSAMModel: MLModel?
    private var isProcessing = false
    private let processInterval: TimeInterval = 0.5 // Slower for debugging
    private var lastProcessTime = Date()
    private var frameCount = 0
    
    override init() {
        super.init()
        loadFastSAMModel()
        setupCamera()
    }
    
    private func loadFastSAMModel() {
        print("🔄 FastSAMProcessor: Starting model load process...")
        DispatchQueue.main.async {
            self.statusMessage = "Loading model..."
        }
        
        // Look for FastSAM model
        let modelNames = ["FastSAM-embedded", "FastSAM-x", "FastSAM", "yolov8x-seg"]
        
        for name in modelNames {
            print("🔍 FastSAMProcessor: Looking for model: \(name)")
            
            for ext in ["mlmodelc", "mlmodel"] {
                let fullName = "\(name).\(ext)"
                print("🔍 FastSAMProcessor: Checking for \(fullName)")
                
                if let url = Bundle.main.url(forResource: name, withExtension: ext) {
                    print("✅ FastSAMProcessor: Found model file: \(fullName)")
                    print("📍 FastSAMProcessor: URL: \(url)")
                    
                    do {
                        print("⏳ FastSAMProcessor: Loading MLModel from URL...")
                        let model = try MLModel(contentsOf: url)
                        print("✅ FastSAMProcessor: MLModel loaded successfully")
                        print("🔧 FastSAMProcessor: Model type: \(type(of: model))")
                        
                        self.fastSAMModel = model
                        print("✅ FastSAMProcessor: Model stored in fastSAMModel variable")
                        
                        // Print detailed model info
                        let desc = model.modelDescription
                        print("📊 FastSAMProcessor: Model Description:")
                        print("   - Inputs: \(desc.inputDescriptionsByName.keys)")
                        print("   - Outputs: \(desc.outputDescriptionsByName.keys)")
                        
                        for (key, input) in desc.inputDescriptionsByName {
                            print("   - Input '\(key)': \(input.type)")
                            if let constraint = input.multiArrayConstraint {
                                print("     Shape: \(constraint.shape)")
                                print("     DataType: \(constraint.dataType.rawValue)")
                            }
                        }
                        
                        for (key, output) in desc.outputDescriptionsByName {
                            print("   - Output '\(key)': \(output.type)")
                            if let constraint = output.multiArrayConstraint {
                                print("     Shape: \(constraint.shape)")
                            }
                        }
                        
                        // Update status on main thread
                        DispatchQueue.main.async {
                            self.statusMessage = "FastSAM Ready"
                        }
                        print("🎉 FastSAMProcessor: Model loading complete!")
                        return
                        
                    } catch {
                        print("❌ FastSAMProcessor: Failed to load \(fullName)")
                        print("❌ Error: \(error)")
                        print("❌ Error localized: \(error.localizedDescription)")
                    }
                } else {
                    print("❌ FastSAMProcessor: No file at \(fullName)")
                }
            }
        }
        
        print("⚠️ FastSAMProcessor: No FastSAM model found after checking all names")
        DispatchQueue.main.async {
            self.statusMessage = "No model"
        }
    }
    
    private func setupCamera() {
        session.beginConfiguration()
        session.sessionPreset = .hd1280x720
        
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else {
            session.commitConfiguration()
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
            }
            
            session.commitConfiguration()
            print("✅ Camera configured")
        } catch {
            print("❌ Camera setup failed: \(error)")
            session.commitConfiguration()
        }
    }
    
    func start() {
        if !session.isRunning {
            DispatchQueue.global(qos: .background).async {
                self.session.startRunning()
            }
        }
    }
    
    func stop() {
        if session.isRunning {
            session.stopRunning()
        }
    }
    
    private func processFastSAM(pixelBuffer: CVPixelBuffer) {
        guard let model = fastSAMModel else { return }
        
        // Run the heavy processing on background queue
        videoQueue.async { [weak self] in
            guard let self = self else { return }
            
            // Create input
            guard let input = self.createModelInput(from: pixelBuffer) else {
                print("❌ FastSAMProcessor: Failed to create model input")
                return
            }
            
            // Run inference
            do {
                let output = try model.prediction(from: input)
                
                // Diagnostic: Print all output features
                print("\n📊 === OUTPUT DIAGNOSTICS (Frame #\(self.frameCount)) ===")
                for featureName in output.featureNames {
                    if let featureValue = output.featureValue(for: featureName) {
                        print("📦 Feature: \(featureName)")
                        
                        if let multiArray = featureValue.multiArrayValue {
                            print("   - Type: MLMultiArray")
                            print("   - Shape: \(multiArray.shape)")
                            print("   - Strides: \(multiArray.strides)")
                            print("   - Count: \(multiArray.count)")
                            print("   - DataType: \(multiArray.dataType.rawValue)")
                            
                            // Print first few values for debugging
                            if multiArray.count > 0 {
                                let dataPointer = multiArray.dataPointer.assumingMemoryBound(to: Float.self)
                                print("   - First 10 values: ", terminator: "")
                                for i in 0..<min(10, multiArray.count) {
                                    print(String(format: "%.3f", dataPointer[i]), terminator: " ")
                                }
                                print("")
                                
                                // Check for NaN or Inf values
                                var hasNaN = false
                                var hasInf = false
                                for i in 0..<min(1000, multiArray.count) {
                                    if dataPointer[i].isNaN { hasNaN = true }
                                    if dataPointer[i].isInfinite { hasInf = true }
                                }
                                if hasNaN { print("   ⚠️ Contains NaN values!") }
                                if hasInf { print("   ⚠️ Contains Infinite values!") }
                            }
                        } else {
                            print("   - Type: Not MLMultiArray")
                            print("   - Value: \(featureValue)")
                        }
                    }
                }
                print("=========================\n")
                
                // Get outputs - try both var_1240 and var_1550
                if let masks = output.featureValue(for: "var_1240")?.multiArrayValue,
                   let predictions = output.featureValue(for: "var_1550")?.multiArrayValue {
                    
                    print("✅ Got both outputs, processing visualization...")
                    
                    // Check actual shapes
                    print("📐 Actual Masks shape: \(masks.shape)")
                    print("📐 Actual Predictions shape: \(predictions.shape)")
                    
                    // Only proceed if shapes match expected format
                    if predictions.shape.count >= 3 && masks.shape.count >= 4 {
                        self.createVisualization(from: pixelBuffer, masks: masks, predictions: predictions)
                    } else {
                        print("⚠️ Unexpected output shapes, creating fallback visualization")
                        self.createFallbackVisualization(from: pixelBuffer)
                    }
                } else {
                    print("❌ Missing required outputs, creating fallback visualization")
                    self.createFallbackVisualization(from: pixelBuffer)
                }
            } catch {
                print("❌ FastSAMProcessor: Prediction failed - \(error)")
            }
        }
    }
    
    private func createModelInput(from pixelBuffer: CVPixelBuffer) -> MLDictionaryFeatureProvider? {
        let width = 640
        let height = 640
        
        // Create input array
        guard let multiArray = try? MLMultiArray(shape: [1, 3, NSNumber(value: height), NSNumber(value: width)], dataType: .float32) else {
            return nil
        }
        
        // Resize pixel buffer
        guard let resizedBuffer = resizePixelBuffer(pixelBuffer, width: width, height: height) else {
            return nil
        }
        
        // Fast conversion using direct memory access
        CVPixelBufferLockBaseAddress(resizedBuffer, .readOnly)
        defer {
            CVPixelBufferUnlockBaseAddress(resizedBuffer, .readOnly)
        }
        
        guard let baseAddress = CVPixelBufferGetBaseAddress(resizedBuffer) else {
            return nil
        }
        
        let bytesPerRow = CVPixelBufferGetBytesPerRow(resizedBuffer)
        let buffer = baseAddress.assumingMemoryBound(to: UInt8.self)
        
        // Direct access to MLMultiArray's data
        let dataPointer = multiArray.dataPointer.assumingMemoryBound(to: Float.self)
        
        // Memory layout for shape [1, 3, 640, 640]:
        let channelSize = width * height
        
        for y in 0..<height {
            for x in 0..<width {
                let pixelIndex = y * bytesPerRow + x * 4
                
                // BGRA format: B=0, G=1, R=2, A=3
                let b = Float(buffer[pixelIndex]) / 255.0
                let g = Float(buffer[pixelIndex + 1]) / 255.0
                let r = Float(buffer[pixelIndex + 2]) / 255.0
                
                let outputIndex = y * width + x
                
                // Write to correct channel positions
                dataPointer[outputIndex] = r                    // R channel
                dataPointer[channelSize + outputIndex] = g      // G channel
                dataPointer[2 * channelSize + outputIndex] = b  // B channel
            }
        }
        
        guard let provider = try? MLDictionaryFeatureProvider(dictionary: ["image": multiArray]) else {
            return nil
        }
        
        return provider
    }
    
    private func resizePixelBuffer(_ pixelBuffer: CVPixelBuffer, width: Int, height: Int) -> CVPixelBuffer? {
        var resizedPixelBuffer: CVPixelBuffer?
        let attrs = [kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue,
                     kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue] as CFDictionary
        
        CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA, attrs, &resizedPixelBuffer)
        
        guard let outputBuffer = resizedPixelBuffer else { return nil }
        
        // Use CoreImage for efficient resizing
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let scaleX = CGFloat(width) / ciImage.extent.width
        let scaleY = CGFloat(height) / ciImage.extent.height
        let scaledImage = ciImage.transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))
        
        let context = CIContext(options: [.useSoftwareRenderer: false])
        context.render(scaledImage, to: outputBuffer)
        
        return outputBuffer
    }
    
    private func createVisualization(from pixelBuffer: CVPixelBuffer, masks: MLMultiArray, predictions: MLMultiArray) {
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        
        // Determine the format based on shape
        print("🔍 Analyzing output format...")
        print("   Predictions shape: \(predictions.shape)")
        print("   Masks shape: \(masks.shape)")
        
        // Check if it's the expected YOLOv8 format
        if predictions.shape.count == 3 && predictions.shape[1].intValue == 116 {
            // Standard YOLOv8 format: [1, 116, 8400]
            print("✅ Detected YOLOv8 format")
            let detections = processDetections(predictions: predictions)
            let segmentationMask = createSegmentationMask(
                masks: masks,
                detections: detections,
                width: width,
                height: height
            )
            applySegmentationOverlay(
                originalBuffer: pixelBuffer,
                mask: segmentationMask,
                width: width,
                height: height
            )
        } else if predictions.shape.count == 3 && predictions.shape[2].intValue == 116 {
            // Transposed format: [1, 8400, 116]
            print("✅ Detected transposed YOLOv8 format")
            let detections = processTransposedDetections(predictions: predictions)
            let segmentationMask = createSegmentationMask(
                masks: masks,
                detections: detections,
                width: width,
                height: height
            )
            applySegmentationOverlay(
                originalBuffer: pixelBuffer,
                mask: segmentationMask,
                width: width,
                height: height
            )
        } else {
            print("⚠️ Unknown format, using fallback")
            createFallbackVisualization(from: pixelBuffer)
        }
    }
    
    private func createFallbackVisualization(from pixelBuffer: CVPixelBuffer) {
        // Simple edge detection or color filter as fallback
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        
        // Try edge detection first
        if let edgeFilter = CIFilter(name: "CIEdges") {
            edgeFilter.setValue(ciImage, forKey: kCIInputImageKey)
            edgeFilter.setValue(10.0, forKey: kCIInputIntensityKey)
            
            if let outputImage = edgeFilter.outputImage {
                // Blend with original
                if let blendFilter = CIFilter(name: "CISourceOverCompositing") {
                    blendFilter.setValue(outputImage, forKey: kCIInputImageKey)
                    blendFilter.setValue(ciImage, forKey: kCIInputBackgroundImageKey)
                    
                    if let finalImage = blendFilter.outputImage {
                        let context = CIContext()
                        if let cgImage = context.createCGImage(finalImage, from: finalImage.extent) {
                            let uiImage = UIImage(cgImage: cgImage, scale: 1.0, orientation: .right)
                            
                            DispatchQueue.main.async {
                                self.segmentedImage = uiImage
                                self.statusMessage = "Edge Detection"
                            }
                            return
                        }
                    }
                }
            }
        }
        
        // Fallback to color adjustment
        if let colorFilter = CIFilter(name: "CIColorControls") {
            colorFilter.setValue(ciImage, forKey: kCIInputImageKey)
            colorFilter.setValue(1.5, forKey: kCIInputSaturationKey)
            colorFilter.setValue(1.2, forKey: kCIInputContrastKey)
            
            if let outputImage = colorFilter.outputImage {
                let context = CIContext()
                if let cgImage = context.createCGImage(outputImage, from: outputImage.extent) {
                    let uiImage = UIImage(cgImage: cgImage, scale: 1.0, orientation: .right)
                    
                    DispatchQueue.main.async {
                        self.segmentedImage = uiImage
                        self.statusMessage = "Processing"
                    }
                }
            }
        }
    }
    
    private func processDetections(predictions: MLMultiArray) -> [(box: CGRect, confidence: Float, classId: Int, maskCoeffs: [Float])] {
        var detections: [(box: CGRect, confidence: Float, classId: Int, maskCoeffs: [Float])] = []
        
        // predictions shape: [1, 116, 8400]
        let numPredictions = predictions.shape[2].intValue // 8400
        let featuresPerPrediction = predictions.shape[1].intValue // 116
        
        let dataPointer = predictions.dataPointer.assumingMemoryBound(to: Float.self)
        
        for i in 0..<min(numPredictions, 1000) { // Limit for performance
            let baseIdx = i * featuresPerPrediction
            
            // Get bounding box
            let cx = CGFloat(dataPointer[baseIdx])
            let cy = CGFloat(dataPointer[baseIdx + 1])
            let w = CGFloat(dataPointer[baseIdx + 2])
            let h = CGFloat(dataPointer[baseIdx + 3])
            
            // Skip invalid boxes
            if w <= 0 || h <= 0 { continue }
            
            // Find max class score
            var maxScore: Float = 0
            var maxClassId = 0
            
            for classIdx in 0..<80 {
                let score = dataPointer[baseIdx + 4 + classIdx]
                if score > maxScore {
                    maxScore = score
                    maxClassId = classIdx
                }
            }
            
            // Only keep high-confidence detections
            if maxScore > confidenceThreshold {
                // Get mask coefficients (last 32 values)
                var maskCoeffs: [Float] = []
                for j in 0..<32 {
                    maskCoeffs.append(dataPointer[baseIdx + 84 + j])
                }
                
                // Convert to normalized coordinates
                let box = CGRect(
                    x: (cx - w/2) / 640.0,
                    y: (cy - h/2) / 640.0,
                    width: w / 640.0,
                    height: h / 640.0
                )
                
                // Validate box is within bounds
                if box.minX >= 0 && box.minY >= 0 && box.maxX <= 1 && box.maxY <= 1 {
                    detections.append((box: box, confidence: maxScore, classId: maxClassId, maskCoeffs: maskCoeffs))
                }
                
                if detections.count >= 10 { break }
            }
        }
        
        // Sort by confidence
        detections.sort { $0.confidence > $1.confidence }
        
        // Update detection count on main thread
        let count = detections.count
        DispatchQueue.main.async {
            self.detectionCount = count
        }
        
        print("📦 Found \(count) valid detections")
        
        return detections
    }
    
    private func processTransposedDetections(predictions: MLMultiArray) -> [(box: CGRect, confidence: Float, classId: Int, maskCoeffs: [Float])] {
        var detections: [(box: CGRect, confidence: Float, classId: Int, maskCoeffs: [Float])] = []
        
        // Transposed format: [1, 8400, 116]
        let numPredictions = predictions.shape[1].intValue // 8400
        let featuresPerPrediction = predictions.shape[2].intValue // 116
        
        let dataPointer = predictions.dataPointer.assumingMemoryBound(to: Float.self)
        
        for i in 0..<min(numPredictions, 1000) { // Limit for performance
            // Calculate offset for transposed layout
            let baseIdx = i * featuresPerPrediction
            
            // Get bounding box
            let cx = CGFloat(dataPointer[baseIdx])
            let cy = CGFloat(dataPointer[baseIdx + 1])
            let w = CGFloat(dataPointer[baseIdx + 2])
            let h = CGFloat(dataPointer[baseIdx + 3])
            
            // Skip invalid boxes
            if w <= 0 || h <= 0 { continue }
            
            // Find max class score
            var maxScore: Float = 0
            var maxClassId = 0
            
            for classIdx in 0..<80 {
                let score = dataPointer[baseIdx + 4 + classIdx]
                if score > maxScore {
                    maxScore = score
                    maxClassId = classIdx
                }
            }
            
            // Only keep high-confidence detections
            if maxScore > confidenceThreshold {
                // Get mask coefficients (last 32 values)
                var maskCoeffs: [Float] = []
                for j in 0..<32 {
                    maskCoeffs.append(dataPointer[baseIdx + 84 + j])
                }
                
                // Convert to normalized coordinates
                let box = CGRect(
                    x: (cx - w/2) / 640.0,
                    y: (cy - h/2) / 640.0,
                    width: w / 640.0,
                    height: h / 640.0
                )
                
                // Validate box is within bounds
                if box.minX >= 0 && box.minY >= 0 && box.maxX <= 1 && box.maxY <= 1 {
                    detections.append((box: box, confidence: maxScore, classId: maxClassId, maskCoeffs: maskCoeffs))
                }
                
                if detections.count >= 10 { break }
            }
        }
        
        // Sort by confidence
        detections.sort { $0.confidence > $1.confidence }
        
        // Update detection count on main thread
        let count = detections.count
        DispatchQueue.main.async {
            self.detectionCount = count
        }
        
        print("📦 Found \(count) valid detections")
        
        return detections
    }
    
    private func createSegmentationMask(masks: MLMultiArray, detections: [(box: CGRect, confidence: Float, classId: Int, maskCoeffs: [Float])], width: Int, height: Int) -> [UInt8] {
        // Create output mask buffer
        var outputMask = [UInt8](repeating: 0, count: width * height)
        
        // masks shape: [1, 32, 160, 160] - prototype masks
        let maskHeight = 160
        let maskWidth = 160
        let numPrototypes = 32
        
        let maskPointer = masks.dataPointer.assumingMemoryBound(to: Float.self)
        
        // For each detection, create a mask
        for (idx, detection) in detections.enumerated() {
            let segmentId = UInt8(min(idx + 1, 255))
            
            let boxX = Int(detection.box.minX * CGFloat(width))
            let boxY = Int(detection.box.minY * CGFloat(height))
            let boxW = Int(detection.box.width * CGFloat(width))
            let boxH = Int(detection.box.height * CGFloat(height))
            
            // Apply combined prototype masks weighted by coefficients
            for y in 0..<boxH {
                for x in 0..<boxW {
                    let outX = min(max(boxX + x, 0), width - 1)
                    let outY = min(max(boxY + y, 0), height - 1)
                    
                    // Sample from prototype mask
                    let maskX = min(x * maskWidth / max(boxW, 1), maskWidth - 1)
                    let maskY = min(y * maskHeight / max(boxH, 1), maskHeight - 1)
                    
                    // Combine prototype masks using coefficients
                    var maskValue: Float = 0
                    for protoIdx in 0..<min(numPrototypes, detection.maskCoeffs.count) {
                        let protoOffset = protoIdx * maskWidth * maskHeight
                        let maskIdx = protoOffset + maskY * maskWidth + maskX
                        maskValue += maskPointer[maskIdx] * detection.maskCoeffs[protoIdx]
                    }
                    
                    // Apply sigmoid activation
                    maskValue = 1.0 / (1.0 + exp(-maskValue))
                    
                    if maskValue > 0.5 {
                        outputMask[outY * width + outX] = segmentId
                    }
                }
            }
        }
        
        return outputMask
    }
    
    private func applySegmentationOverlay(originalBuffer: CVPixelBuffer, mask: [UInt8], width: Int, height: Int) {
        // Color palette for different segments
        let colors: [(r: UInt8, g: UInt8, b: UInt8)] = [
            (255, 0, 0),      // Red
            (0, 255, 0),      // Green
            (0, 0, 255),      // Blue
            (255, 255, 0),    // Yellow
            (255, 0, 255),    // Magenta
            (0, 255, 255),    // Cyan
            (255, 128, 0),    // Orange
            (128, 0, 255),    // Purple
            (0, 128, 255),    // Light blue
            (255, 0, 128),    // Pink
        ]
        
        // Create overlay buffer
        var overlayBuffer: CVPixelBuffer?
        let attrs = [kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue,
                     kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue] as CFDictionary
        
        CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA, attrs, &overlayBuffer)
        
        guard let overlay = overlayBuffer else { return }
        
        CVPixelBufferLockBaseAddress(overlay, [])
        defer { CVPixelBufferUnlockBaseAddress(overlay, []) }
        
        if let baseAddress = CVPixelBufferGetBaseAddress(overlay) {
            let bytesPerRow = CVPixelBufferGetBytesPerRow(overlay)
            let buffer = baseAddress.assumingMemoryBound(to: UInt8.self)
            
            // Apply colored overlay based on segmentation mask
            for y in 0..<height {
                for x in 0..<width {
                    let pixelIndex = y * bytesPerRow + x * 4
                    let maskValue = mask[y * width + x]
                    
                    if maskValue > 0 {
                        let colorIdx = Int(maskValue - 1) % colors.count
                        let color = colors[colorIdx]
                        
                        buffer[pixelIndex] = color.b      // B
                        buffer[pixelIndex + 1] = color.g  // G
                        buffer[pixelIndex + 2] = color.r  // R
                        buffer[pixelIndex + 3] = 128      // A (semi-transparent)
                    } else {
                        buffer[pixelIndex] = 0
                        buffer[pixelIndex + 1] = 0
                        buffer[pixelIndex + 2] = 0
                        buffer[pixelIndex + 3] = 0
                    }
                }
            }
        }
        
        // Blend overlay with original
        let ciImage = CIImage(cvPixelBuffer: originalBuffer)
        let overlayCI = CIImage(cvPixelBuffer: overlay)
        
        if let blendFilter = CIFilter(name: "CISourceOverCompositing") {
            blendFilter.setValue(overlayCI, forKey: kCIInputImageKey)
            blendFilter.setValue(ciImage, forKey: kCIInputBackgroundImageKey)
            
            if let outputImage = blendFilter.outputImage {
                let context = CIContext()
                if let cgImage = context.createCGImage(outputImage, from: outputImage.extent) {
                    let uiImage = UIImage(cgImage: cgImage, scale: 1.0, orientation: .right)
                    
                    // Update UI on main thread
                    DispatchQueue.main.async {
                        self.segmentedImage = uiImage
                        self.statusMessage = "Segmented (\(self.detectionCount) objects)"
                    }
                }
            }
        }
    }
}

extension FastSAMProcessor: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return
        }
        
        // Throttle processing
        let now = Date()
        let timeSinceLastProcess = now.timeIntervalSince(lastProcessTime)
        
        if timeSinceLastProcess < processInterval {
            return
        }
        
        if isProcessing {
            return
        }
        
        isProcessing = true
        lastProcessTime = now
        frameCount += 1
        
        processFastSAM(pixelBuffer: pixelBuffer)
        
        isProcessing = false
    }
}

// MARK: - SwiftUI Preview
struct SimpleCameraOverlay_Previews: PreviewProvider {
    static var previews: some View {
        SimpleCameraOverlay(
            capturedImage: .constant(nil),
            isShowingCamera: .constant(true)
        )
    }
}
