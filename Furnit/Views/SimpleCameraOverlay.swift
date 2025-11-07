import SwiftUI
import AVFoundation
import CoreML
import Vision
import CoreImage
import Accelerate

struct SimpleCameraOverlay: View {
    @Binding var capturedImage: UIImage?
    @Binding var isShowingCamera: Bool
    @StateObject private var camera = MobileSAMProcessor()
    
    var body: some View {
        ZStack {
            // Fullscreen camera feed
            if let segmented = camera.segmentedImage {
                Image(uiImage: segmented)
                    .resizable()
                    .aspectRatio(contentMode: .fit)  // Changed from .fill to .fit
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .background(Color.black)
                    .ignoresSafeArea()
            } else {
                CameraPreviewLayer(session: camera.session)
                    .ignoresSafeArea()
                    .overlay(
                        // Show "Camera Loading" if black
                        Text("Camera Loading...")
                            .font(.headline)
                            .foregroundColor(.white)
                            .padding()
                            .background(Color.black.opacity(0.7))
                            .cornerRadius(8)
                            .opacity(camera.segmentedImage == nil ? 1 : 0)
                    )
            }
            
            // CENTER CROSSHAIR - Only visible when no segmented image
            if camera.segmentedImage == nil {
                VStack {
                    Rectangle()
                        .fill(Color.green)
                        .frame(width: 2, height: 30)
                    Spacer()
                        .frame(height: 10)
                    Rectangle()
                        .fill(Color.green)
                        .frame(width: 2, height: 30)
                }
                .frame(width: 2, height: 70)
                
                HStack {
                    Rectangle()
                        .fill(Color.green)
                        .frame(width: 30, height: 2)
                    Spacer()
                        .frame(width: 10)
                    Rectangle()
                        .fill(Color.green)
                        .frame(width: 30, height: 2)
                }
                .frame(width: 70, height: 2)
                
                Circle()
                    .stroke(Color.green, lineWidth: 2)
                    .frame(width: 40, height: 40)
            }
            
            // Tap indicator (yellow circle)
            if let tapPoint = camera.lastTapPoint {
                Circle()
                    .fill(Color.yellow)
                    .frame(width: 30, height: 30)
                    .position(tapPoint)
                    .animation(.easeInOut(duration: 0.3), value: tapPoint)
            }
            
            // Top bar
            VStack {
                HStack {
                    // Show "Back to Camera" button when segmented image is showing
                    if camera.segmentedImage != nil {
                        Button(action: {
                            withAnimation {
                                camera.segmentedImage = nil
                                camera.lastTapPoint = nil
                            }
                            print("🔄 Reset to live camera preview")
                        }) {
                            HStack(spacing: 6) {
                                Image(systemName: "arrow.counterclockwise")
                                    .font(.system(size: 14, weight: .semibold))
                                Text("Back to Camera")
                                    .font(.system(size: 14, weight: .medium))
                            }
                            .foregroundColor(.white)
                            .padding(.horizontal, 14)
                            .padding(.vertical, 8)
                            .background(Color.blue.opacity(0.9))
                            .cornerRadius(8)
                        }
                    } else {
                        Text("MobileSAM")
                            .font(.headline)
                            .foregroundColor(.white)
                            .padding(.horizontal, 16)
                            .padding(.vertical, 8)
                            .background(Color.black.opacity(0.7))
                            .cornerRadius(8)
                    }
                    
                    Spacer()
                    
                    Text(camera.statusMessage)
                        .font(.caption)
                        .foregroundColor(.green)
                        .padding(.horizontal, 12)
                        .padding(.vertical, 6)
                        .background(Color.black.opacity(0.7))
                        .cornerRadius(8)
                    
                    Button(action: { isShowingCamera = false }) {
                        Image(systemName: "xmark.circle.fill")
                            .font(.system(size: 28))
                            .foregroundColor(.white)
                            .padding(8)
                            .background(Color.black.opacity(0.7))
                            .clipShape(Circle())
                    }
                }
                .padding(.horizontal)
                .padding(.top, 50)
                
                Spacer()
                
                // Bottom instructions
                VStack(spacing: 10) {
                    if camera.segmentedImage == nil {
                        Text("TAP GREEN CROSSHAIR TO SEGMENT")
                            .font(.headline)
                            .foregroundColor(.white)
                            .padding(.horizontal, 16)
                            .padding(.vertical, 8)
                            .background(Color.green.opacity(0.8))
                            .cornerRadius(8)
                    } else {
                        Text("SEGMENTED! Tap 'Back to Camera' to segment more")
                            .font(.subheadline)
                            .foregroundColor(.white)
                            .padding(.horizontal, 16)
                            .padding(.vertical, 8)
                            .background(Color.orange.opacity(0.8))
                            .cornerRadius(8)
                    }
                    
                    HStack {
                        if camera.segmentedImage == nil {
                            Text("Or tap anywhere on furniture")
                                .font(.subheadline)
                                .foregroundColor(.white)
                                .padding(.horizontal, 16)
                                .padding(.vertical, 8)
                                .background(Color.black.opacity(0.7))
                                .cornerRadius(8)
                        }
                        
                        Spacer()
                        
                        Button(action: {
                            if let currentImage = camera.segmentedImage {
                                capturedImage = currentImage
                                print("📸 Captured MobileSAM segmented image")
                                // Close overlay after capture
                                isShowingCamera = false
                            }
                        }) {
                            VStack(spacing: 4) {
                                Image(systemName: camera.segmentedImage != nil ? "checkmark.circle.fill" : "camera.circle.fill")
                                    .font(.system(size: 50))
                                    .foregroundColor(camera.segmentedImage != nil ? .green : .white.opacity(0.5))
                                    .shadow(color: .black.opacity(0.5), radius: 5)
                                
                                if camera.segmentedImage != nil {
                                    Text("Capture")
                                        .font(.caption2)
                                        .foregroundColor(.white)
                                }
                            }
                        }
                        .disabled(camera.segmentedImage == nil)
                    }
                    .padding(.horizontal, 20)
                }
                .padding(.bottom, 40)
            }
        }
        .contentShape(Rectangle())
        .onTapGesture { location in
            // Only allow tapping when live camera is showing
            if camera.segmentedImage == nil {
                camera.handleTap(at: location, viewSize: UIScreen.main.bounds.size)
            }
        }
        .onAppear {
            // Longer delay to ensure previous sessions are fully stopped
            print("🎬 SimpleCameraOverlay appeared - waiting for camera...")
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                camera.start()
            }
        }
        .onDisappear {
            print("👋 SimpleCameraOverlay disappeared - stopping camera")
            camera.stop()
        }
    }
}

struct CameraPreviewLayer: UIViewRepresentable {
    let session: AVCaptureSession
    
    func makeUIView(context: Context) -> UIView {
        let view = UIView()
        view.backgroundColor = .black
        
        let previewLayer = AVCaptureVideoPreviewLayer(session: session)
        previewLayer.videoGravity = .resizeAspectFill
        
        // Fix orientation - rotate 90° clockwise to correct anticlockwise tilt
        if #available(iOS 17.0, *) {
            previewLayer.connection?.videoRotationAngle = 90
        } else {
            previewLayer.connection?.videoOrientation = .landscapeRight
        }
        
        previewLayer.frame = view.bounds
        view.layer.addSublayer(previewLayer)
        
        print("📹 Preview layer created with rotation 90° and bounds: \(view.bounds)")
        
        return view
    }
    
    func updateUIView(_ uiView: UIView, context: Context) {
        DispatchQueue.main.async {
            if let previewLayer = uiView.layer.sublayers?.first(where: { $0 is AVCaptureVideoPreviewLayer }) as? AVCaptureVideoPreviewLayer {
                previewLayer.frame = uiView.bounds
            }
        }
    }
}

// MobileSAM Processor
class MobileSAMProcessor: NSObject, ObservableObject {
    let session = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let videoQueue = DispatchQueue(label: "mobileSAMQueue", qos: .userInitiated)
    
    @Published var segmentedImage: UIImage?
    @Published var statusMessage = "Loading..."
    @Published var lastTapPoint: CGPoint?
    
    private var encoderModel: MLModel?
    private var decoderModel: MLModel?
    private var isProcessing = false
    private var currentPixelBuffer: CVPixelBuffer?
    private var currentImageEmbeddings: MLMultiArray?
    
    override init() {
        super.init()
        loadModels()
        setupCamera()
    }
    
    private func loadModels() {
        DispatchQueue.main.async {
            self.statusMessage = "Loading models..."
        }
        
        guard let encoderURL = Bundle.main.url(forResource: "MobileSAMImageEncoder", withExtension: "mlmodelc"),
              let decoderURL = Bundle.main.url(forResource: "MobileSAMMaskDecoder", withExtension: "mlmodelc") else {
            print("❌ Models not found")
            DispatchQueue.main.async {
                self.statusMessage = "Models not found"
            }
            return
        }
        
        do {
            let encoderConfig = MLModelConfiguration()
            encoderConfig.computeUnits = .all
            encoderModel = try MLModel(contentsOf: encoderURL, configuration: encoderConfig)
            
            let decoderConfig = MLModelConfiguration()
            decoderConfig.computeUnits = .all
            decoderModel = try MLModel(contentsOf: decoderURL, configuration: decoderConfig)
            
            DispatchQueue.main.async {
                self.statusMessage = "Tap to segment"
            }
            print("✅ MobileSAM models loaded successfully")
        } catch {
            print("❌ Failed to load models: \(error)")
            DispatchQueue.main.async {
                self.statusMessage = "Model error"
            }
        }
    }
    
    private func setupCamera() {
        session.beginConfiguration()
        session.sessionPreset = .hd1280x720
        
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else {
            print("❌ No camera device found")
            session.commitConfiguration()
            return
        }
        
        do {
            let input = try AVCaptureDeviceInput(device: device)
            if session.canAddInput(input) {
                session.addInput(input)
                print("✅ Camera input added")
            }
            
            videoOutput.setSampleBufferDelegate(self, queue: videoQueue)
            videoOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
            videoOutput.alwaysDiscardsLateVideoFrames = true
            
            if session.canAddOutput(videoOutput) {
                session.addOutput(videoOutput)
                print("✅ Video output added")
            }
            
            session.commitConfiguration()
            print("✅ Camera session configured")
        } catch {
            print("❌ Camera setup failed: \(error)")
            session.commitConfiguration()
        }
    }
    
    func start() {
        // Force stop any existing session first
        if session.isRunning {
            session.stopRunning()
            print("🛑 Stopped existing camera session")
        }
        
        // Wait a moment for cleanup
        DispatchQueue.global(qos: .background).asyncAfter(deadline: .now() + 0.3) {
            // Double-check the session is configured
            if self.session.inputs.isEmpty {
                print("⚠️ No camera inputs - reconfiguring")
                self.setupCamera()
            }
            
            // Start fresh
            if !self.session.isRunning {
                self.session.startRunning()
                print("📹 Camera session started successfully")
                
                // Update status on main thread
                DispatchQueue.main.async {
                    self.statusMessage = "Tap to segment"
                }
            }
        }
    }
    
    func stop() {
        if session.isRunning {
            session.stopRunning()
            print("🛑 Camera session stopped")
        }
        
        // Clear current state
        currentPixelBuffer = nil
        currentImageEmbeddings = nil
        
        // Small delay to ensure complete stop
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
            self.segmentedImage = nil
        }
    }
    
    func handleTap(at location: CGPoint, viewSize: CGSize) {
        print("👆 Tap at: \(location)")
        
        guard let pixelBuffer = currentPixelBuffer,
              let embeddings = currentImageEmbeddings,
              let decoder = decoderModel else {
            print("⚠️ Not ready for segmentation")
            DispatchQueue.main.async {
                self.statusMessage = "Not ready yet..."
            }
            return
        }
        
        DispatchQueue.main.async {
            self.lastTapPoint = location
            self.statusMessage = "Segmenting..."
        }
        
        let normalizedX = Float(location.x / viewSize.width)
        let normalizedY = Float(location.y / viewSize.height)
        
        print("📍 Normalized: (\(normalizedX), \(normalizedY))")
        
        DispatchQueue.global(qos: .userInitiated).async {
            self.runDecoder(embeddings: embeddings, point: (normalizedX, normalizedY), originalBuffer: pixelBuffer)
        }
    }
    
    private func runDecoder(embeddings: MLMultiArray, point: (x: Float, y: Float), originalBuffer: CVPixelBuffer) {
        guard let decoder = decoderModel else { return }
        
        do {
            let pointCoords = try MLMultiArray(shape: [1, 1, 2], dataType: .float32)
            pointCoords[[0, 0, 0] as [NSNumber]] = NSNumber(value: point.x)
            pointCoords[[0, 0, 1] as [NSNumber]] = NSNumber(value: point.y)
            
            let pointLabels = try MLMultiArray(shape: [1, 1], dataType: .float32)
            pointLabels[[0, 0] as [NSNumber]] = NSNumber(value: 1)
            
            print("🎭 Running decoder...")
            
            let inputDict: [String: Any] = [
                "image_embeddings": embeddings,
                "point_coords": pointCoords,
                "point_labels": pointLabels
            ]
            
            let inputProvider = try MLDictionaryFeatureProvider(dictionary: inputDict)
            let output = try decoder.prediction(from: inputProvider)
            
            guard let masks = output.featureValue(for: "masks")?.multiArrayValue else {
                print("❌ Failed to get masks from output")
                DispatchQueue.main.async {
                    self.statusMessage = "Failed to get mask"
                }
                return
            }
            
            print("✅ Got mask!")
            applyMask(masks, to: originalBuffer)
            
        } catch {
            print("❌ Decoder failed: \(error)")
            DispatchQueue.main.async {
                self.statusMessage = "Segmentation failed"
            }
        }
    }
    
    private func applyMask(_ maskArray: MLMultiArray, to pixelBuffer: CVPixelBuffer) {
        let maskHeight = maskArray.shape[2].intValue
        let maskWidth = maskArray.shape[3].intValue
        
        print("📊 Mask shape: \(maskArray.shape)")
        
        var maskData = [UInt8](repeating: 0, count: maskWidth * maskHeight)
        
        // INVERTED: Furniture (positive values) becomes BLACK (0), background becomes WHITE (255)
        for y in 0..<maskHeight {
            for x in 0..<maskWidth {
                let indices = [0, 0, y, x] as [NSNumber]
                let value = maskArray[indices].floatValue
                // INVERT: furniture = 0 (black), background = 255 (white)
                maskData[y * maskWidth + x] = value > 0.0 ? 0 : 255
            }
        }
        
        let blackPixels = maskData.filter { $0 == 0 }.count
        print("✅ Mask extracted (inverted): \(blackPixels) black pixels (furniture)")
        
        if blackPixels == 0 {
            print("⚠️ Empty mask")
            DispatchQueue.main.async {
                self.statusMessage = "No object found"
            }
            return
        }
        
        guard let provider = CGDataProvider(data: Data(maskData) as CFData),
              let colorSpace = CGColorSpace(name: CGColorSpace.linearGray) else {
            return
        }
        
        guard let cgMask = CGImage(
            width: maskWidth,
            height: maskHeight,
            bitsPerComponent: 8,
            bitsPerPixel: 8,
            bytesPerRow: maskWidth,
            space: colorSpace,
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
            provider: provider,
            decode: nil,
            shouldInterpolate: false,
            intent: .defaultIntent
        ) else {
            print("❌ Failed to create mask")
            return
        }
        
        let maskImage = CIImage(cgImage: cgMask)
        let originalImage = CIImage(cvPixelBuffer: pixelBuffer)
        
        let scaleX = originalImage.extent.width / CGFloat(maskWidth)
        let scaleY = originalImage.extent.height / CGFloat(maskHeight)
        
        let scaledMask = maskImage
            .transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))
            .samplingLinear()
        
        guard let blendFilter = CIFilter(name: "CIBlendWithMask") else { return }
        
        let transparent = CIImage(color: .clear).cropped(to: originalImage.extent)
        
        // With INVERTED mask (black=furniture, white=background):
        // CIBlendWithMask: result = inputImage * mask + backgroundImage * (1-mask)
        // Black (0) areas show backgroundImage (originalImage = furniture)
        // White (255) areas show inputImage (transparent = background)
        blendFilter.setValue(transparent, forKey: kCIInputImageKey)
        blendFilter.setValue(originalImage, forKey: kCIInputBackgroundImageKey)
        blendFilter.setValue(scaledMask, forKey: kCIInputMaskImageKey)
        
        guard let result = blendFilter.outputImage else { return }
        
        let context = CIContext()
        if let cgImage = context.createCGImage(result, from: result.extent) {
            // Rotate 90° clockwise to match camera orientation
            let uiImage = UIImage(cgImage: cgImage, scale: 1.0, orientation: .right)
            
            DispatchQueue.main.async {
                self.segmentedImage = uiImage
                self.statusMessage = "Ready to capture!"
            }
            print("✅ Segmented with inverted mask - furniture visible, background transparent!")
        }
    }
}

extension MobileSAMProcessor: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        
        currentPixelBuffer = pixelBuffer
        
        if !isProcessing {
            isProcessing = true
            
            DispatchQueue.global(qos: .userInitiated).async {
                self.runEncoder(pixelBuffer: pixelBuffer)
                self.isProcessing = false
            }
        }
    }
    
    private func runEncoder(pixelBuffer: CVPixelBuffer) {
        guard let encoder = encoderModel else { return }
        
        do {
            guard let resized = resizePixelBuffer(pixelBuffer, width: 1024, height: 1024) else {
                return
            }
            
            guard let imageArray = pixelBufferToMLMultiArray(resized) else {
                return
            }
            
            let inputDict: [String: Any] = ["image": imageArray]
            let inputProvider = try MLDictionaryFeatureProvider(dictionary: inputDict)
            
            let output = try encoder.prediction(from: inputProvider)
            
            guard let embeddings = output.featureValue(for: "image_embeddings")?.multiArrayValue else {
                return
            }
            
            currentImageEmbeddings = embeddings
            
        } catch {
            print("❌ Encoder failed: \(error)")
        }
    }
    
    private func pixelBufferToMLMultiArray(_ pixelBuffer: CVPixelBuffer) -> MLMultiArray? {
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        
        guard let array = try? MLMultiArray(shape: [1, 3, NSNumber(value: height), NSNumber(value: width)], dataType: .float32) else {
            return nil
        }
        
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        
        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            return nil
        }
        
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let buffer = baseAddress.assumingMemoryBound(to: UInt8.self)
        
        let mean: [Float] = [0.485, 0.456, 0.406]
        let std: [Float] = [0.229, 0.224, 0.225]
        
        let arrayPointer = array.dataPointer.assumingMemoryBound(to: Float.self)
        
        for y in 0..<height {
            for x in 0..<width {
                let pixelIndex = y * bytesPerRow + x * 4
                
                let b = Float(buffer[pixelIndex]) / 255.0
                let g = Float(buffer[pixelIndex + 1]) / 255.0
                let r = Float(buffer[pixelIndex + 2]) / 255.0
                
                let rIndex = 0 * height * width + y * width + x
                let gIndex = 1 * height * width + y * width + x
                let bIndex = 2 * height * width + y * width + x
                
                arrayPointer[rIndex] = (r - mean[0]) / std[0]
                arrayPointer[gIndex] = (g - mean[1]) / std[1]
                arrayPointer[bIndex] = (b - mean[2]) / std[2]
            }
        }
        
        return array
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
        
        let context = CIContext()
        context.render(scaledImage, to: outputBuffer)
        
        return outputBuffer
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
