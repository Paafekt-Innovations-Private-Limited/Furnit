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

// FastSAMCameraModel with proper segmentation
class FastSAMCameraModel: NSObject, ObservableObject {
    @Published var segmentedImage: UIImage?
    let session = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let videoQueue = DispatchQueue(label: "videoQueue", qos: .userInitiated)
    
    private var fastSAMModel: MLModel?
    private let context = CIContext()
    
    // Throttling
    private var lastProcessTime = Date()
    private let processInterval: TimeInterval = 0.2 // Process every 200ms
    private var isProcessing = false
    
    // FastSAM parameters
    private let confidenceThreshold: Float = 0.5
    
    override init() {
        super.init()
        checkCameraAuthorization()
        loadFastSAMModel()
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
        let modelNames = ["FastSAM-x"
//                          , "FastSAM-embedded"
//                          , "FastSAM"
//                          , "yolov8x-seg"
        ]
        
        for name in modelNames {
            print("🔍 Looking for model: \(name)")
            
            for ext in ["mlmodelc", "mlpackage"] {
                if let url = Bundle.main.url(forResource: name, withExtension: ext) {
                    print("✅ Found model file: \(name).\(ext)")
                    
                    do {
                        let model = try MLModel(contentsOf: url)
                        self.fastSAMModel = model
                        print("✅ FastSAM-x model loaded successfully")
                        
                        // Print model info
                        let desc = model.modelDescription
                        print("📊 Model Inputs: \(desc.inputDescriptionsByName.keys)")
                        print("📊 Model Outputs: \(desc.outputDescriptionsByName.keys)")
                        return
                        
                    } catch {
                        print("❌ Failed to load \(name): \(error)")
                    }
                }
            }
        }
        
        print("⚠️ No FastSAM model found")
    }
    
    private func setupCamera() {
        session.beginConfiguration()
        session.sessionPreset = .photo  // Use photo preset for full resolution
        
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
                    // Disable video mirroring
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
            
            // Process segmentation
            processYOLOv8Segmentation(detections: detections, prototypes: prototypes, originalPixelBuffer: pixelBuffer)
            
        } catch {
            print("❌ Prediction failed: \(error)")
        }
        
        isProcessing = false
    }
    
    private func processYOLOv8Segmentation(detections: MLMultiArray, prototypes: MLMultiArray, originalPixelBuffer: CVPixelBuffer) {
        let numDetections = detections.shape[2].intValue  // 8400
        let numValues = detections.shape[1].intValue      // 37
        let numPrototypes = prototypes.shape[1].intValue  // 32
        let protoHeight = prototypes.shape[2].intValue    // 160
        let protoWidth = prototypes.shape[3].intValue     // 160
        
        let detPointer = detections.dataPointer.assumingMemoryBound(to: Float.self)
        
        // Find best detection
        var bestDetection: (idx: Int, conf: Float, x: Float, y: Float, w: Float, h: Float)? = nil
        var maxArea: Float = 0
        
        for i in 0..<numDetections {
            let conf = detPointer[4 * numDetections + i]
            
            if conf > confidenceThreshold {
                let x = detPointer[0 * numDetections + i]
                let y = detPointer[1 * numDetections + i]
                let w = detPointer[2 * numDetections + i]
                let h = detPointer[3 * numDetections + i]
                let area = w * h
                
                if area > maxArea && area > 1000 {
                    maxArea = area
                    bestDetection = (i, conf, x, y, w, h)
                }
            }
        }
        
        guard let detection = bestDetection else {
            print("⚠️ No valid detections")
            return
        }
        
        // Get mask coefficients
        var maskCoeffs = [Float](repeating: 0, count: numPrototypes)
        for i in 0..<numPrototypes {
            let coeffIdx = (numValues - numPrototypes + i) * numDetections + detection.idx
            maskCoeffs[i] = detPointer[coeffIdx]
        }
        
        // Generate mask
        let segMask = generateSegmentationMask(prototypes: prototypes, coefficients: maskCoeffs,
                                               protoHeight: protoHeight, protoWidth: protoWidth)
        
        // Apply to image
        let ciImage = CIImage(cvPixelBuffer: originalPixelBuffer)
        applySegmentationMask(original: ciImage, mask: segMask)
    }
    
    private func generateSegmentationMask(prototypes: MLMultiArray, coefficients: [Float],
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
        
        // Apply sigmoid and threshold
        let threshold: Float = 0.3
        for i in 0..<maskSize {
            let sigmoid = 1.0 / (1.0 + exp(-finalMask[i]))
            finalMask[i] = sigmoid > threshold ? 1.0 : 0.0
        }
        
        // Convert to pixel data
        var pixelData = [UInt8](repeating: 0, count: maskSize)
        for i in 0..<maskSize {
            pixelData[i] = UInt8(finalMask[i] * 255)
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
    
    private func applySegmentationMask(original: CIImage, mask: CIImage) {
        // Scale mask to match original
        let scaleX = original.extent.width / mask.extent.width
        let scaleY = original.extent.height / mask.extent.height
        
        let scaledMask = mask
            .transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))
            .samplingNearest()  // Prevent grey pixels
        
        let finalMask = scaledMask.cropped(to: original.extent)
        
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
                print("✅ FastSAM segmentation complete")
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
