import SwiftUI
import AVFoundation
import CoreML
import Vision
import CoreImage

struct SimpleCameraOverlay: View {
    @Binding var capturedImage: UIImage?
    @Binding var isShowingCamera: Bool
    @StateObject private var camera = FastSAMCameraModel()
    
    // Start fully minimized at 30%
    @State private var scaleMultiplier: CGFloat = 0.3  // Start at 30% scale (minimized)
    
    var body: some View {
        ZStack {
            // TRANSPARENT BACKGROUND - No camera preview shown!
            Color.clear
                .ignoresSafeArea()
            
            // Display segmented furniture without cropping function
            if let segmented = camera.segmentedImage {
                Image(uiImage: segmented)
                    .resizable()
                    .scaledToFill()
                    .scaleEffect(scaleMultiplier)
                    .position(x: UIScreen.main.bounds.width / 2,
                             y: UIScreen.main.bounds.height / 2)
                    .allowsHitTesting(false)
                    .ignoresSafeArea()
            }
            
            // UI controls overlay
            VStack {
                HStack {
                    // Size adjustment slider
                    HStack {
                        Image(systemName: "minus.magnifyingglass")
                            .foregroundColor(.white)
                        
                        // Slider adjusts scale from 30% to 100%
                        Slider(value: $scaleMultiplier, in: 0.3...1.0)
                            .frame(width: 120)
                            .accentColor(.white)
                        
                        Image(systemName: "plus.magnifyingglass")
                            .foregroundColor(.white)
                    }
                    .padding(8)
                    .background(Capsule().fill(Color.black.opacity(0.5)))
                    .padding(.leading)
                    
                    Spacer()
                    
                    // Close button
                    Button(action: {
                        isShowingCamera = false
                    }) {
                        Image(systemName: "xmark.circle.fill")
                            .font(.title)
                            .foregroundColor(.white)
                            .background(Circle().fill(Color.black.opacity(0.5)))
                            .shadow(radius: 3)
                    }
                    .padding()
                }
                .padding(.top, 50)
                
                Spacer()
                
                // Status indicator at bottom
                HStack {
                    // Status indicator
                    if camera.segmentedImage == nil {
                        Text("Point at furniture")
                            .font(.caption)
                            .padding(8)
                            .background(Capsule().fill(Color.black.opacity(0.5)))
                            .foregroundColor(.white)
                    } else {
                        Text("Size: \(Int(scaleMultiplier * 100))%")
                            .font(.caption)
                            .padding(8)
                            .background(Capsule().fill(Color.black.opacity(0.5)))
                            .foregroundColor(.white)
                    }
                    
                    Spacer()
                    
                    // Sensitivity adjustment slider
                    HStack {
                        Text("Sens:")
                            .font(.caption2)
                            .foregroundColor(.white)
                        
                        Slider(value: $camera.confidenceThreshold, in: 0.3...0.7)
                            .frame(width: 100)
                            .accentColor(.green)
                        
                        Text("\(Int(camera.confidenceThreshold * 100))%")
                            .font(.caption2)
                            .foregroundColor(.white)
                            .frame(width: 35)
                    }
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(Capsule().fill(Color.black.opacity(0.5)))
                }
                .padding(.bottom, 40)
                .padding(.horizontal)
            }
        }
        .onAppear {
            AVCaptureDevice.requestAccess(for: .video) { granted in
                if granted {
                    camera.startSession()
                    print("📷 Camera session started for FastSAM segmentation")
                } else {
                    print("⚠️ Camera access denied")
                }
            }
        }
        .onDisappear {
            camera.stopSession()
        }
    }
}

// FastSAMCameraModel with U2Net + Center Proximity + Size Scoring
class FastSAMCameraModel: NSObject, ObservableObject {
    @Published var segmentedImage: UIImage?
    @Published var confidenceThreshold: Float = 0.5
    
    let session = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let videoQueue = DispatchQueue(label: "videoQueue", qos: .userInitiated)
    
    private var fastSAMModel: MLModel?
    private var u2netModel: VNCoreMLModel?
    private let context = CIContext()
    
    // Throttling
    private var lastProcessTime = Date()
    private let processInterval: TimeInterval = 0.2 // Process every 200ms
    private var isProcessing = false
    
    // U2Net guidance mask
    private var u2netMask: CVPixelBuffer?
    private let u2netWeight: Float = 0.3 // Bonus weight for U2Net overlap
    private var u2netUpdateCounter = 0  // Update U2Net every N frames
    private let u2netUpdateInterval = 5 // Run U2Net every 5 frames
    
    // Frame dimensions for center calculation
    private let frameWidth: Float = 640
    private let frameHeight: Float = 640
    private let frameCenterX: Float = 320
    private let frameCenterY: Float = 320
    private let frameArea: Float = 640 * 640
    
    override init() {
        super.init()
        checkCameraAuthorization()
        loadFastSAMModel()
        loadU2NetModel()
    }
    
    private func checkCameraAuthorization() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            setupCamera()
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
                if granted {
                    DispatchQueue.main.async {
                        self?.setupCamera()
                    }
                }
            }
        default:
            print("⚠️ Camera access denied")
        }
    }
    
    private func loadFastSAMModel() {
        // Look for FastSAM-x model
        let modelNames = ["FastSAM-x", "FastSAM-embedded", "FastSAM", "yolov8x-seg"]
        
        for name in modelNames {
            print("🔍 Looking for model: \(name)")
            
            for ext in ["mlmodelc", "mlpackage"] {
                if let url = Bundle.main.url(forResource: name, withExtension: ext) {
                    print("✅ Found model file: \(name).\(ext)")
                    
                    do {
                        let model = try MLModel(contentsOf: url)
                        self.fastSAMModel = model
                        print("✅ FastSAM-x model loaded successfully")
                        return
                        
                    } catch {
                        print("❌ Failed to load \(name): \(error)")
                    }
                }
            }
        }
        
        print("⚠️ No FastSAM model found")
    }
    
    private func loadU2NetModel() {
        // Try multiple possible U2Net model names
        let modelNames = ["u2netp", "U2Net", "u2net", "U2NetP", "U2NET"]
        
        for name in modelNames {
            if let modelURL = Bundle.main.url(forResource: name, withExtension: "mlmodelc") {
                do {
                    let model = try MLModel(contentsOf: modelURL)
                    u2netModel = try VNCoreMLModel(for: model)
                    print("✅ U2-Net model loaded: \(name)")
                    return
                } catch {
                    print("Failed to load \(name): \(error)")
                }
            }
        }
        
        print("⚠️ No U2-Net model found - using FastSAM with geometric scoring only")
    }
    
    private func setupCamera() {
        session.beginConfiguration()
        session.sessionPreset = .hd1280x720
        
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else {
            print("❌ No camera found")
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
                
                // Set correct video orientation
                if let connection = videoOutput.connection(with: .video) {
                    if connection.isVideoOrientationSupported {
                        connection.videoOrientation = .portrait
                    }
                    if connection.isVideoMirroringSupported {
                        connection.isVideoMirrored = false
                    }
                }
            }
            
            session.commitConfiguration()
            print("✅ Camera configured for FastSAM")
        } catch {
            print("❌ Camera setup failed: \(error)")
        }
    }
    
    func startSession() {
        if !session.isRunning {
            DispatchQueue.global(qos: .background).async {
                self.session.startRunning()
            }
        }
    }
    
    func stopSession() {
        if session.isRunning {
            session.stopRunning()
        }
    }
    
    private func processWithFastSAM(pixelBuffer: CVPixelBuffer) {
        let now = Date()
        guard now.timeIntervalSince(lastProcessTime) >= processInterval else { return }
        guard !isProcessing else { return }
        
        isProcessing = true
        lastProcessTime = now
        
        guard let model = fastSAMModel else {
            print("⚠️ No model available")
            isProcessing = false
            return
        }
        
        // Run U2Net first if available to get foreground guidance
        if let u2net = u2netModel {
            runU2NetForGuidance(pixelBuffer: pixelBuffer)
        }
        
        // Create input for FastSAM
        guard let resizedBuffer = resizePixelBuffer(pixelBuffer, width: 640, height: 640) else {
            print("❌ Failed to resize buffer")
            isProcessing = false
            return
        }
        
        let imageValue = MLFeatureValue(pixelBuffer: resizedBuffer)
        
        guard let provider = try? MLDictionaryFeatureProvider(dictionary: ["image": imageValue]) else {
            print("❌ Failed to create feature provider")
            isProcessing = false
            return
        }
        
        do {
            let prediction = try model.prediction(from: provider)
            
            // Get FastSAM outputs
            guard let detections = prediction.featureValue(for: "var_1550")?.multiArrayValue,
                  let prototypes = prediction.featureValue(for: "p")?.multiArrayValue else {
                print("❌ Failed to get model outputs")
                isProcessing = false
                return
            }
            
            // Process segmentation with triple scoring (U2Net + Center + Size)
            processWithTripleScoring(detections: detections, prototypes: prototypes, originalPixelBuffer: pixelBuffer)
            
        } catch {
            print("❌ Prediction failed: \(error)")
        }
        
        isProcessing = false
    }
    
    private func runU2NetForGuidance(pixelBuffer: CVPixelBuffer) {
        guard let model = u2netModel else { return }
        
        // Only update U2Net every N frames
        u2netUpdateCounter += 1
        if u2netUpdateCounter < u2netUpdateInterval && u2netMask != nil {
            // Skip U2Net update, use existing mask
            return
        }
        u2netUpdateCounter = 0
        
        // Create synchronous semaphore
        let semaphore = DispatchSemaphore(value: 0)
        var newMask: CVPixelBuffer?
        
        let request = VNCoreMLRequest(model: model) { request, error in
            if let results = request.results as? [VNPixelBufferObservation],
               let maskBuffer = results.first?.pixelBuffer {
                newMask = maskBuffer
            }
            semaphore.signal()
        }
        
        request.imageCropAndScaleOption = .scaleFit
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
        
        do {
            try handler.perform([request])
            // Wait for completion (with timeout)
            _ = semaphore.wait(timeout: .now() + 0.1)
            if let mask = newMask {
                self.u2netMask = mask
            }
        } catch {
            print("❌ U2Net processing failed: \(error)")
        }
    }
    
    private func processWithTripleScoring(detections: MLMultiArray, prototypes: MLMultiArray, originalPixelBuffer: CVPixelBuffer) {
        let numDetections = detections.shape[2].intValue  // 8400
        let numValues = detections.shape[1].intValue      // 37
        let numPrototypes = prototypes.shape[1].intValue  // 32
        let protoHeight = prototypes.shape[2].intValue    // 160
        let protoWidth = prototypes.shape[3].intValue     // 160
        
        let detPointer = detections.dataPointer.assumingMemoryBound(to: Float.self)
        
        // Find best detection using triple scoring
        var bestDetection: (idx: Int, score: Float, conf: Float, x: Float, y: Float, w: Float, h: Float)? = nil
        var bestScore: Float = 0
        
        for i in 0..<numDetections {
            let conf = detPointer[4 * numDetections + i]
            
            if conf > confidenceThreshold {
                let x = detPointer[0 * numDetections + i]
                let y = detPointer[1 * numDetections + i]
                let w = detPointer[2 * numDetections + i]
                let h = detPointer[3 * numDetections + i]
                let area = w * h
                
                // === 1. CENTER PROXIMITY SCORE (0-1) ===
                let distanceFromCenter = sqrt(pow(x - frameCenterX, 2) + pow(y - frameCenterY, 2))
                let maxDistance = sqrt(pow(frameCenterX, 2) + pow(frameCenterY, 2))
                let centerProximityScore = 1.0 - (distanceFromCenter / maxDistance)
                
                // === 2. SIZE REASONABILITY SCORE (0-1) ===
                let areaRatio = area / frameArea
                var sizeScore: Float = 0
                if areaRatio >= 0.1 && areaRatio <= 0.6 {
                    // Optimal size range for furniture
                    sizeScore = 1.0
                } else if areaRatio < 0.1 {
                    // Too small - likely background object
                    sizeScore = areaRatio * 10  // Linear scale up to 0.1
                } else {
                    // Too large - likely wall/floor
                    sizeScore = max(0, 1.0 - (areaRatio - 0.6) * 2)  // Linear scale down after 0.6
                }
                
                // === 3. U2NET OVERLAP BONUS ===
                var u2netBonus: Float = 0
                if u2netMask != nil {
                    let overlapRatio = calculateU2NetOverlap(x: x, y: y, width: w, height: h)
                    u2netBonus = overlapRatio * u2netWeight
                }
                
                // === COMBINED TRIPLE SCORE ===
                // Base: conf² × center × size
                // Bonus: +U2Net overlap
                let combinedScore = (conf * conf * centerProximityScore * sizeScore) + u2netBonus
                
                // Track best scoring detection
                if combinedScore > bestScore {
                    bestScore = combinedScore
                    bestDetection = (i, combinedScore, conf, x, y, w, h)
                    
                    print("📍 Better detection found:")
                    print("   Confidence: \(String(format: "%.3f", conf))")
                    print("   Center proximity: \(String(format: "%.2f", centerProximityScore))")
                    print("   Size score: \(String(format: "%.2f", sizeScore)) (area: \(String(format: "%.2f", areaRatio)))")
                    print("   U2Net bonus: +\(String(format: "%.3f", u2netBonus))")
                    print("   TOTAL score: \(String(format: "%.3f", combinedScore))")
                }
            }
        }
        
        guard let detection = bestDetection else {
            print("⚠️ No valid detections found")
            return
        }
        
        print("✅ Selected with triple scoring:")
        print("   Final score: \(String(format: "%.3f", detection.score))")
        print("   Position: (\(String(format: "%.0f", detection.x)), \(String(format: "%.0f", detection.y)))")
        print("   Size: \(String(format: "%.0fx%.0f", detection.w, detection.h))")
        
        // Get mask coefficients
        var maskCoeffs = [Float](repeating: 0, count: numPrototypes)
        for i in 0..<numPrototypes {
            let coeffIdx = (numValues - numPrototypes + i) * numDetections + detection.idx
            maskCoeffs[i] = detPointer[coeffIdx]
        }
        
        // Generate mask
        let segMask = generateCleanMask(prototypes: prototypes, coefficients: maskCoeffs,
                                        protoHeight: protoHeight, protoWidth: protoWidth)
        
        // Apply to image
        let ciImage = CIImage(cvPixelBuffer: originalPixelBuffer)
        applyCleanSegmentation(original: ciImage, mask: segMask)
    }
    
    private func calculateU2NetOverlap(x: Float, y: Float, width: Float, height: Float) -> Float {
        guard let u2netMask = u2netMask else { return 0 }
        
        CVPixelBufferLockBaseAddress(u2netMask, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(u2netMask, .readOnly) }
        
        let maskWidth = CVPixelBufferGetWidth(u2netMask)
        let maskHeight = CVPixelBufferGetHeight(u2netMask)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(u2netMask)
        guard let baseAddress = CVPixelBufferGetBaseAddress(u2netMask) else { return 0 }
        
        // Scale detection box to mask dimensions
        let scaleX = Float(maskWidth) / frameWidth
        let scaleY = Float(maskHeight) / frameHeight
        
        let boxLeft = Int(max(0, (x - width/2) * scaleX))
        let boxRight = Int(min(Float(maskWidth-1), (x + width/2) * scaleX))
        let boxTop = Int(max(0, (y - height/2) * scaleY))
        let boxBottom = Int(min(Float(maskHeight-1), (y + height/2) * scaleY))
        
        // Count overlapping pixels
        var overlapCount = 0
        var totalCount = 0
        
        for row in boxTop...boxBottom {
            let rowPtr = baseAddress.advanced(by: row * bytesPerRow).assumingMemoryBound(to: UInt8.self)
            for col in boxLeft...boxRight {
                totalCount += 1
                if rowPtr[col] > 128 {  // U2Net mask threshold
                    overlapCount += 1
                }
            }
        }
        
        return totalCount > 0 ? Float(overlapCount) / Float(totalCount) : 0
    }
    
    private func generateCleanMask(prototypes: MLMultiArray, coefficients: [Float],
                                   protoHeight: Int, protoWidth: Int) -> CIImage {
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
        
        // Simple sigmoid and HARD binary threshold
        let threshold: Float = 0.5
        var pixelData = [UInt8](repeating: 0, count: maskSize)
        
        for i in 0..<maskSize {
            let sigmoid = 1.0 / (1.0 + exp(-finalMask[i]))
            pixelData[i] = sigmoid > threshold ? 255 : 0
        }
        
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
            shouldInterpolate: false,
            intent: .defaultIntent
        ) else {
            return CIImage(color: .white).cropped(to: CGRect(x: 0, y: 0, width: protoWidth, height: protoHeight))
        }
        
        return CIImage(cgImage: cgImage)
    }
    
    private func applyCleanSegmentation(original: CIImage, mask: CIImage) {
        // Scale mask to match original
        let scaleX = original.extent.width / mask.extent.width
        let scaleY = original.extent.height / mask.extent.height
        
        // Use nearest neighbor sampling to keep binary mask
        let scaledMask = mask
            .transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))
            .samplingNearest()
        
        // Optional: VERY light blur
        var finalMask = scaledMask
        if let blurFilter = CIFilter(name: "CIGaussianBlur") {
            blurFilter.setValue(scaledMask, forKey: kCIInputImageKey)
            blurFilter.setValue(0.5, forKey: kCIInputRadiusKey)
            if let blurred = blurFilter.outputImage {
                finalMask = blurred
            }
        }
        
        finalMask = finalMask.cropped(to: original.extent)
        
        // Apply mask using blend filter
        guard let blendFilter = CIFilter(name: "CIBlendWithMask") else {
            return
        }
        
        let transparent = CIImage(color: CIColor(red: 0, green: 0, blue: 0, alpha: 0))
            .cropped(to: original.extent)
        
        blendFilter.setValue(original, forKey: kCIInputImageKey)
        blendFilter.setValue(transparent, forKey: kCIInputBackgroundImageKey)
        blendFilter.setValue(finalMask, forKey: kCIInputMaskImageKey)
        
        guard let result = blendFilter.outputImage else {
            return
        }
        
        // Render final image
        let context = CIContext(options: [
            .workingColorSpace: CGColorSpaceCreateDeviceRGB(),
            .outputPremultiplied: true,
            .useSoftwareRenderer: false
        ])
        
        if let cgImage = context.createCGImage(result, from: result.extent) {
            let uiImage = UIImage(cgImage: cgImage, scale: 1.0, orientation: .up)
            
            DispatchQueue.main.async {
                self.segmentedImage = uiImage
            }
        }
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
}

extension FastSAMCameraModel: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        processWithFastSAM(pixelBuffer: pixelBuffer)
    }
}
