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
                // Checkered background to show transparency
//                CheckeredBackground()
//                    .frame(width: currentSize.width, height: currentSize.height)
                
                if let segmented = camera.segmentedImage {
                    Image(uiImage: segmented)
                        .resizable()
                        .aspectRatio(contentMode: .fill)
                        .frame(width: currentSize.width, height: currentSize.height)
                        .clipped()
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
                // Sensitivity slider
                HStack {
                    Text("Sens:")
                        .font(.caption2)
                        .foregroundColor(.white)
                    
                    Slider(value: $camera.confidenceThreshold, in: 0.1...0.9)
                        .frame(width: 100)
                        .accentColor(.green)
                    
                    Text("\(Int((1.0 - camera.confidenceThreshold) * 100))%")
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

// Checkered background to show transparency
struct CheckeredBackground: View {
    var body: some View {
        GeometryReader { geometry in
            Path { path in
                let size: CGFloat = 10
                let rows = Int(geometry.size.height / size) + 1
                let cols = Int(geometry.size.width / size) + 1
                
                for row in 0..<rows {
                    for col in 0..<cols {
                        if (row + col) % 2 == 0 {
                            let rect = CGRect(x: CGFloat(col) * size,
                                            y: CGFloat(row) * size,
                                            width: size,
                                            height: size)
                            path.addRect(rect)
                        }
                    }
                }
            }
            .fill(Color.blue.opacity(0.2))
        }
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
            if #available(iOS 17.0, *) {
                previewLayer.connection?.videoRotationAngle = 90
            } else {
                previewLayer.connection?.videoOrientation = .portrait
            }
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

// FastSAM Processor with Background Removal
class FastSAMProcessor: NSObject, ObservableObject {
    let session = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let videoQueue = DispatchQueue(label: "fastSAMQueue", qos: .userInitiated)
    
    @Published var segmentedImage: UIImage?
    @Published var statusMessage = "Loading..."
    @Published var confidenceThreshold: Float = 0.5 // Controls sensitivity
    @Published private var detectionCount = 0
    
    // The actual MLModel - NOT VNCoreMLModel
    private var fastSAMModel: MLModel?
    private var isProcessing = false
    private let processInterval: TimeInterval = 0.1 // Process every 300ms
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
        let modelNames = ["FastSAM-x", "FastSAM-embedded", "FastSAM", "yolov8x-seg"]
        
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
                        
                        // Update status on main thread
                        DispatchQueue.main.async {
                            self.statusMessage = "FastSAM Ready"
                        }
                        print("🎉 FastSAMProcessor: Model loading complete!")
                        return
                        
                    } catch {
                        print("❌ FastSAMProcessor: Failed to load \(fullName)")
                        print("❌ Error: \(error)")
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
        print("🎥 FastSAMProcessor: Setting up camera...")
        session.beginConfiguration()
        session.sessionPreset = .hd1280x720
        print("🎥 FastSAMProcessor: Session preset set to hd1280x720")
        
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else {
            print("❌ FastSAMProcessor: Failed to get camera device")
            session.commitConfiguration()
            return
        }
        
        print("✅ FastSAMProcessor: Camera device obtained")
        
        do {
            let input = try AVCaptureDeviceInput(device: device)
            if session.canAddInput(input) {
                session.addInput(input)
                print("✅ FastSAMProcessor: Camera input added to session")
            }
            
            videoOutput.setSampleBufferDelegate(self, queue: videoQueue)
            videoOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
            videoOutput.alwaysDiscardsLateVideoFrames = true
            
            if session.canAddOutput(videoOutput) {
                session.addOutput(videoOutput)
                print("✅ FastSAMProcessor: Video output added to session")
            }
            
            session.commitConfiguration()
            print("✅ FastSAMProcessor: Camera configured successfully")
        } catch {
            print("❌ FastSAMProcessor: Camera setup failed: \(error)")
            session.commitConfiguration()
        }
    }
    
    func start() {
        print("▶️ FastSAMProcessor: Starting camera session...")
        if !session.isRunning {
            DispatchQueue.global(qos: .background).async {
                self.session.startRunning()
                print("✅ FastSAMProcessor: Camera session started")
            }
        } else {
            print("ℹ️ FastSAMProcessor: Camera session already running")
        }
    }
    
    func stop() {
        print("⏸️ FastSAMProcessor: Stopping camera session...")
        if session.isRunning {
            session.stopRunning()
            print("✅ FastSAMProcessor: Camera session stopped")
        }
    }
    
    private func processFastSAM(pixelBuffer: CVPixelBuffer) {
        print("🔄 [Frame \(frameCount)] Processing started")
        print("🔄 [Frame \(frameCount)] Confidence threshold: \(confidenceThreshold)")
        
        guard let model = fastSAMModel else {
            print("⚠️ [Frame \(frameCount)] No model, showing original")
            let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
            showOriginal(ciImage)
            return
        }
        
        // Use the actual FastSAM model for segmentation
        runFastSAMSegmentation(pixelBuffer: pixelBuffer, model: model)
    }
    
    private func runFastSAMSegmentation(pixelBuffer: CVPixelBuffer, model: MLModel) {
        print("🤖 [Frame \(frameCount)] Running FastSAM model inference...")
        
        guard let input = createModelInput(from: pixelBuffer) else {
            print("❌ [Frame \(frameCount)] Failed to create model input")
            let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
            showOriginal(ciImage)
            return
        }
        
        print("✅ [Frame \(frameCount)] Model input created")
        
        do {
            let prediction = try model.prediction(from: input)
            print("✅ [Frame \(frameCount)] Model prediction completed")
            
            // Get both outputs - NOTE: var_1240 is now "p"
            guard let detections = prediction.featureValue(for: "var_1550")?.multiArrayValue,
                  let prototypes = prediction.featureValue(for: "p")?.multiArrayValue else {
                print("❌ [Frame \(frameCount)] Failed to get model outputs")
                print("📊 Available outputs: \(model.modelDescription.outputDescriptionsByName.keys)")
                let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
                showOriginal(ciImage)
                return
            }
            
            print("📊 [Frame \(frameCount)] Detections shape: \(detections.shape)")
            print("📊 [Frame \(frameCount)] Prototypes shape: \(prototypes.shape)")
            
            // Process with proper YOLOv8-seg decoding
            processYOLOv8Segmentation(detections: detections, prototypes: prototypes, originalPixelBuffer: pixelBuffer)
            
        } catch {
            print("❌ [Frame \(frameCount)] Model prediction failed: \(error)")
            let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
            showOriginal(ciImage)
        }
    }
    
    private func processYOLOv8Segmentation(detections: MLMultiArray, prototypes: MLMultiArray, originalPixelBuffer: CVPixelBuffer) {
        print("🔍 [Frame \(frameCount)] Processing YOLOv8 segmentation...")
        
        let numDetections = detections.shape[2].intValue  // 8400
        let numValues = detections.shape[1].intValue      // 37
        let numPrototypes = prototypes.shape[1].intValue  // 32
        let protoHeight = prototypes.shape[2].intValue    // 160
        let protoWidth = prototypes.shape[3].intValue     // 160
        
        print("🔍 [Frame \(frameCount)] Num detections: \(numDetections)")
        print("🔍 [Frame \(frameCount)] Prototypes: \(numPrototypes) x \(protoHeight) x \(protoWidth)")
        
        let detPointer = detections.dataPointer.assumingMemoryBound(to: Float.self)
        
        // Find detections above threshold AND prefer larger objects
        var validDetections: [(idx: Int, conf: Float, x: Float, y: Float, w: Float, h: Float, area: Float)] = []
        
        for i in 0..<numDetections {
            let conf = detPointer[4 * numDetections + i]
            
            if conf > confidenceThreshold {
                let x = detPointer[0 * numDetections + i]
                let y = detPointer[1 * numDetections + i]
                let w = detPointer[2 * numDetections + i]
                let h = detPointer[3 * numDetections + i]
                let area = w * h
                
                // Filter: Only accept reasonably sized objects (not tiny detections)
                if area > 1000 {  // Minimum area threshold
                    validDetections.append((i, conf, x, y, w, h, area))
                }
            }
        }
        
        print("🔍 [Frame \(frameCount)] Found \(validDetections.count) valid detections (conf > \(confidenceThreshold), area > 1000)")
        
        if validDetections.isEmpty {
            print("⚠️ [Frame \(frameCount)] No valid detections found")
            let ciImage = CIImage(cvPixelBuffer: originalPixelBuffer)
            showOriginal(ciImage)
            return
        }
        
        // Sort by area (largest first) and pick the largest one with good confidence
        validDetections.sort { $0.area > $1.area }
        
        let bestDetection = validDetections[0]
        
        print("🔍 [Frame \(frameCount)] Selected detection:")
        print("   idx=\(bestDetection.idx), conf=\(bestDetection.conf)")
        print("   BBox: center=(\(bestDetection.x), \(bestDetection.y)), size=(\(bestDetection.w), \(bestDetection.h))")
        print("   Area: \(bestDetection.area)")
        
        // Get mask coefficients (last 32 values)
        var maskCoeffs = [Float](repeating: 0, count: numPrototypes)
        for i in 0..<numPrototypes {
            let coeffIdx = (numValues - numPrototypes + i) * numDetections + bestDetection.idx
            maskCoeffs[i] = detPointer[coeffIdx]
        }
        
        print("🔍 [Frame \(frameCount)] Mask coeffs: [\(maskCoeffs.prefix(5).map { String(format: "%.2f", $0) }.joined(separator: ", "))...]")
        
        // Generate segmentation mask from prototypes
        let segMask = generateSegmentationMask(prototypes: prototypes, coefficients: maskCoeffs,
                                               protoHeight: protoHeight, protoWidth: protoWidth)
        
        // Apply to image
        let ciImage = CIImage(cvPixelBuffer: originalPixelBuffer)
        applySegmentationMask(original: ciImage, mask: segMask)
    }
    
    private func generateSegmentationMask(prototypes: MLMultiArray, coefficients: [Float],
                                         protoHeight: Int, protoWidth: Int) -> CIImage {
        print("🎭 [Frame \(frameCount)] Generating segmentation mask...")
        
        let protoPointer = prototypes.dataPointer.assumingMemoryBound(to: Float.self)
        let maskSize = protoHeight * protoWidth
        
        // Weighted sum of prototypes
        var finalMask = [Float](repeating: 0, count: maskSize)
        
        for protoIdx in 0..<coefficients.count {
            let coeff = coefficients[protoIdx]
            let protoOffset = protoIdx * maskSize
            
            for i in 0..<maskSize {
                finalMask[i] += coeff * protoPointer[protoOffset + i]
            }
        }
        
        // Apply sigmoid activation
        for i in 0..<maskSize {
            finalMask[i] = 1.0 / (1.0 + exp(-finalMask[i]))
        }
        
        print("📊 [Frame \(frameCount)] AFTER SIGMOID - showing ALL values for first 50 pixels:")
        for i in 0..<min(50, maskSize) {
            print("   Pixel[\(i)] = \(String(format: "%.4f", finalMask[i]))")
        }
        
        // APPLY BINARY THRESHOLD
        let threshold: Float = 0.3
        var whitePixels = 0
        var blackPixels = 0
        var greyPixels = 0  // Should be 0!
        
        for i in 0..<maskSize {
            if finalMask[i] > threshold {
                finalMask[i] = 1.0  // WHITE
                whitePixels += 1
            } else {
                finalMask[i] = 0.0  // BLACK
                blackPixels += 1
            }
            
            // Check for any grey values (should never happen!)
            if finalMask[i] != 0.0 && finalMask[i] != 1.0 {
                greyPixels += 1
                print("⚠️ GREY PIXEL FOUND at index \(i): value = \(finalMask[i])")
            }
        }
        
        print("📊 [Frame \(frameCount)] AFTER BINARY THRESHOLD:")
        print("   White pixels (1.0): \(whitePixels)")
        print("   Black pixels (0.0): \(blackPixels)")
        print("   Grey pixels (between 0-1): \(greyPixels) ← SHOULD BE ZERO!")
        print("   Showing ALL values for first 50 pixels:")
        for i in 0..<min(50, maskSize) {
            print("   Pixel[\(i)] = \(String(format: "%.4f", finalMask[i]))")
        }
        
        // Convert to pixel data
        var pixelData = [UInt8](repeating: 0, count: maskSize)
        var zeroCount = 0
        var fullCount = 0
        var greyCount = 0
        
        for i in 0..<maskSize {
            pixelData[i] = UInt8(finalMask[i] * 255)
            
            if pixelData[i] == 0 {
                zeroCount += 1
            } else if pixelData[i] == 255 {
                fullCount += 1
            } else {
                greyCount += 1
                print("⚠️ GREY BYTE FOUND at index \(i): value = \(pixelData[i]) (should be 0 or 255!)")
            }
        }
        
        print("📊 [Frame \(frameCount)] FINAL PIXEL DATA (0-255):")
        print("   Pixels with value 0: \(zeroCount)")
        print("   Pixels with value 255: \(fullCount)")
        print("   Pixels with grey values (1-254): \(greyCount) ← SHOULD BE ZERO!")
        print("   Showing ALL values for first 50 pixels:")
        for i in 0..<min(50, maskSize) {
            print("   PixelByte[\(i)] = \(pixelData[i])")
        }
        
        // Verify all unique values
        let uniqueValues = Set(pixelData)
        print("📊 [Frame \(frameCount)] UNIQUE PIXEL VALUES IN ENTIRE MASK:")
        print("   \(uniqueValues.sorted()) ← Should only be [0, 255]")
        
        let data = Data(pixelData)
        guard let provider = CGDataProvider(data: data as CFData) else {
            return CIImage(color: .white).cropped(to: CGRect(x: 0, y: 0, width: protoWidth, height: protoHeight))
        }
        
        let colorSpace = CGColorSpaceCreateDeviceGray()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue)
        
        guard let cgImage = CGImage(
            width: protoWidth,
            height: protoHeight,
            bitsPerComponent: 8,
            bitsPerPixel: 8,
            bytesPerRow: protoWidth,
            space: colorSpace,
            bitmapInfo: bitmapInfo,
            provider: provider,
            decode: nil,
            shouldInterpolate: true,
            intent: .defaultIntent
        ) else {
            return CIImage(color: .white).cropped(to: CGRect(x: 0, y: 0, width: protoWidth, height: protoHeight))
        }
        
        print("✅ [Frame \(frameCount)] Mask image created")
        return CIImage(cgImage: cgImage)
    }
    

//    private func applySegmentationMask(original: CIImage, mask: CIImage) {
//        print("🎨 [Frame \(frameCount)] Applying segmentation mask to original image...")
//        
//        // Step 1: Scale mask
//        let scaleX = original.extent.width / mask.extent.width
//        let scaleY = original.extent.height / mask.extent.height
//        
//        let scaledMask = mask.transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))
//        print("🎨 [Frame \(frameCount)] Mask scaled to: \(scaledMask.extent.size)")
//        
//        // Step 2: MINIMAL OR NO BLUR
//        // Option A: No blur at all (sharp edges, no grey)
//        let finalMask = scaledMask.cropped(to: original.extent)
//        print("✅ [Frame \(frameCount)] Using UNBLURRED mask - NO GREY PIXELS!")
//        
//        /* Option B: Very light blur (uncomment if you want slight smoothing)
//        guard let blurFilter = CIFilter(name: "CIGaussianBlur") else {
//            showOriginal(original)
//            return
//        }
//        blurFilter.setValue(scaledMask, forKey: kCIInputImageKey)
//        blurFilter.setValue(2.0, forKey: kCIInputRadiusKey)  // REDUCED from 10.0 to 2.0
//        let finalMask = blurFilter.outputImage?.cropped(to: original.extent) ?? scaledMask
//        print("✅ [Frame \(frameCount)] Light blur applied (radius=2.0)")
//        */
//        
//        // Step 3: Apply mask
//        guard let blendFilter = CIFilter(name: "CIBlendWithMask") else {
//            showOriginal(original)
//            return
//        }
//        
//        let transparent = CIImage(color: CIColor(red: 0, green: 0, blue: 0, alpha: 0))
//            .cropped(to: original.extent)
//        
//        blendFilter.setValue(original, forKey: kCIInputImageKey)
//        blendFilter.setValue(transparent, forKey: kCIInputBackgroundImageKey)
//        blendFilter.setValue(finalMask, forKey: kCIInputMaskImageKey)
//        
//        guard let result = blendFilter.outputImage else {
//            showOriginal(original)
//            return
//        }
//        
//        print("✅ [Frame \(frameCount)] Blend completed with NO GREY from blur")
//        
//        // Step 4: Render
//        let context = CIContext(options: [
//            .workingColorSpace: CGColorSpaceCreateDeviceRGB(),
//            .outputPremultiplied: true,
//            .useSoftwareRenderer: false,
//            .highQualityDownsample: true
//        ])
//        
//        if let cgImage = context.createCGImage(result, from: result.extent) {
//            let uiImage = UIImage(cgImage: cgImage, scale: 1.0, orientation: .right)
//            
//            DispatchQueue.main.async {
//                self.segmentedImage = uiImage
//                self.statusMessage = "Chair Detected"
//            }
//        }
//    }
    
    private func applySegmentationMask(original: CIImage, mask: CIImage) {
        print("🎨 [Frame \(frameCount)] Applying segmentation mask to original image...")
        
        // Step 1: Scale mask WITHOUT INTERPOLATION
        let scaleX = original.extent.width / mask.extent.width
        let scaleY = original.extent.height / mask.extent.height
        
        // CRITICAL: Use nearest-neighbor sampling to avoid grey pixels during scaling
        let scaledMask = mask
            .transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))
            .samplingNearest()  // ← THIS PREVENTS GREY PIXELS!
        
        print("🎨 [Frame \(frameCount)] Mask scaled to: \(scaledMask.extent.size) with NEAREST NEIGHBOR (no interpolation)")
        
        let finalMask = scaledMask.cropped(to: original.extent)
        print("✅ [Frame \(frameCount)] Using UNBLURRED mask with NEAREST NEIGHBOR sampling!")
        
        // Step 2: Apply mask
        guard let blendFilter = CIFilter(name: "CIBlendWithMask") else {
            showOriginal(original)
            return
        }
        
        let transparent = CIImage(color: CIColor(red: 0, green: 0, blue: 0, alpha: 0))
            .cropped(to: original.extent)
        
        blendFilter.setValue(original, forKey: kCIInputImageKey)
        blendFilter.setValue(transparent, forKey: kCIInputBackgroundImageKey)
        blendFilter.setValue(finalMask, forKey: kCIInputMaskImageKey)
        
        guard let result = blendFilter.outputImage else {
            showOriginal(original)
            return
        }
        
        print("✅ [Frame \(frameCount)] Blend completed with nearest-neighbor scaling")
        
        // Step 3: Render without interpolation
        let context = CIContext(options: [
            .workingColorSpace: CGColorSpaceCreateDeviceRGB(),
            .outputPremultiplied: true,
            .useSoftwareRenderer: false
        ])
        
        if let cgImage = context.createCGImage(result, from: result.extent) {
            let uiImage = UIImage(cgImage: cgImage, scale: 1.0, orientation: .right)
            
            DispatchQueue.main.async {
                self.segmentedImage = uiImage
                self.statusMessage = "Chair Detected"
            }
        }
    }
    
    private func createModelInput(from pixelBuffer: CVPixelBuffer) -> MLDictionaryFeatureProvider? {
        print("🔧 [Frame \(frameCount)] Creating model input...")
        
        let width = 640
        let height = 640
        
        // Resize pixel buffer to 640x640
        guard let resizedBuffer = resizePixelBuffer(pixelBuffer, width: width, height: height) else {
            print("❌ [Frame \(frameCount)] Failed to resize pixel buffer")
            return nil
        }
        
        // Create image feature value directly from pixel buffer
        let imageValue = MLFeatureValue(pixelBuffer: resizedBuffer)
        
        guard let provider = try? MLDictionaryFeatureProvider(dictionary: ["image": imageValue]) else {
            print("❌ [Frame \(frameCount)] Failed to create feature provider")
            return nil
        }
        
        print("✅ [Frame \(frameCount)] Model input created successfully (Image type)")
        return provider
    }
    
    private func resizePixelBuffer(_ pixelBuffer: CVPixelBuffer, width: Int, height: Int) -> CVPixelBuffer? {
        var resizedPixelBuffer: CVPixelBuffer?
        let attrs = [kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue,
                     kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue] as CFDictionary
        
        CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA, attrs, &resizedPixelBuffer)
        
        guard let outputBuffer = resizedPixelBuffer else { return nil }
        
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let scaleX = CGFloat(width) / ciImage.extent.width
        let scaleY = CGFloat(height) / ciImage.extent.height
        let scaledImage = ciImage.transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))
        
        let context = CIContext(options: [.useSoftwareRenderer: false])
        context.render(scaledImage, to: outputBuffer)
        
        return outputBuffer
    }
    
    private func showOriginal(_ image: CIImage) {
        print("🖼️ [Frame \(frameCount)] Showing original as fallback")
        let context = CIContext()
        if let cgImage = context.createCGImage(image, from: image.extent) {
            let uiImage = UIImage(cgImage: cgImage, scale: 1.0, orientation: .right)
            
            DispatchQueue.main.async {
                self.segmentedImage = uiImage
                self.statusMessage = "Original"
            }
        }
    }
}

extension FastSAMProcessor: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return
        }
        
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
        
        print("\n🎬 ========== FRAME \(frameCount) ==========")
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
