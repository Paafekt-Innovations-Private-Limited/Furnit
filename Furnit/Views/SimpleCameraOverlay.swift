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
                CheckeredBackground()
                    .frame(width: currentSize.width, height: currentSize.height)
                
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
            .fill(Color.gray.opacity(0.2))
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
    private let processInterval: TimeInterval = 0.3 // Process every 300ms
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
        guard let model = fastSAMModel else {
            // No model, use fallback
            createFallbackVisualization(from: pixelBuffer)
            return
        }
        
        // For now, always use fallback until we fix the model output format
        createFallbackVisualization(from: pixelBuffer)
    }
    
    private func createModelInput(from pixelBuffer: CVPixelBuffer) -> MLDictionaryFeatureProvider? {
        let width = 640
        let height = 640
        
        guard let multiArray = try? MLMultiArray(shape: [1, 3, NSNumber(value: height), NSNumber(value: width)], dataType: .float32) else {
            return nil
        }
        
        guard let resizedBuffer = resizePixelBuffer(pixelBuffer, width: width, height: height) else {
            return nil
        }
        
        CVPixelBufferLockBaseAddress(resizedBuffer, .readOnly)
        defer {
            CVPixelBufferUnlockBaseAddress(resizedBuffer, .readOnly)
        }
        
        guard let baseAddress = CVPixelBufferGetBaseAddress(resizedBuffer) else {
            return nil
        }
        
        let bytesPerRow = CVPixelBufferGetBytesPerRow(resizedBuffer)
        let buffer = baseAddress.assumingMemoryBound(to: UInt8.self)
        
        let dataPointer = multiArray.dataPointer.assumingMemoryBound(to: Float.self)
        let channelSize = width * height
        
        for y in 0..<height {
            for x in 0..<width {
                let pixelIndex = y * bytesPerRow + x * 4
                
                let b = Float(buffer[pixelIndex]) / 255.0
                let g = Float(buffer[pixelIndex + 1]) / 255.0
                let r = Float(buffer[pixelIndex + 2]) / 255.0
                
                let outputIndex = y * width + x
                
                dataPointer[outputIndex] = r
                dataPointer[channelSize + outputIndex] = g
                dataPointer[2 * channelSize + outputIndex] = b
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
        
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let scaleX = CGFloat(width) / ciImage.extent.width
        let scaleY = CGFloat(height) / ciImage.extent.height
        let scaledImage = ciImage.transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))
        
        let context = CIContext(options: [.useSoftwareRenderer: false])
        context.render(scaledImage, to: outputBuffer)
        
        return outputBuffer
    }
    
    private func createFallbackVisualization(from pixelBuffer: CVPixelBuffer) {
        // Use simple background removal for now
        simpleBackgroundRemoval(from: pixelBuffer)
    }
    
    // Replace the entire simpleBackgroundRemoval and related methods with this cleaner version:

    private func simpleBackgroundRemoval(from pixelBuffer: CVPixelBuffer) {
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        
        // Sensitivity: 0.1 to 0.9 -> intensity 1.0 to 10.0
        let edgeIntensity = (1.0 - confidenceThreshold) * 9.0 + 1.0
        
        // Simple edge detection
        guard let edgeFilter = CIFilter(name: "CIEdges") else {
            showOriginal(ciImage)
            return
        }
        
        edgeFilter.setValue(ciImage, forKey: kCIInputImageKey)
        edgeFilter.setValue(edgeIntensity, forKey: kCIInputIntensityKey)
        
        guard let edges = edgeFilter.outputImage else {
            showOriginal(ciImage)
            return
        }
        
        // Convert grayscale edges to colored edges
        guard let colorMatrix = CIFilter(name: "CIColorMatrix") else {
            showOriginal(ciImage)
            return
        }
        
        colorMatrix.setValue(edges, forKey: kCIInputImageKey)
        
        // Make edges bright green (R=0, G=1, B=0)
        colorMatrix.setValue(CIVector(x: 0, y: 0, z: 0, w: 0), forKey: "inputRVector")    // No red
        colorMatrix.setValue(CIVector(x: 0, y: 1, z: 0, w: 0), forKey: "inputGVector")    // Full green from gray
        colorMatrix.setValue(CIVector(x: 0, y: 0, z: 0, w: 0), forKey: "inputBVector")    // No blue
        colorMatrix.setValue(CIVector(x: 0, y: 0, z: 0, w: 1), forKey: "inputAVector")    // Keep alpha
        
        guard let coloredEdges = colorMatrix.outputImage else {
            showOriginal(ciImage)
            return
        }
        
        // Option 1: Just show colored edges on black background (cleanest)
        if confidenceThreshold < 0.3 {
            showEdgesOnly(coloredEdges)
        }
        // Option 2: Overlay colored edges on original image
        else if confidenceThreshold < 0.7 {
            overlayEdgesOnOriginal(coloredEdges, original: ciImage)
        }
        // Option 3: Try to remove background (experimental)
        else {
            attemptBackgroundRemoval(edges: coloredEdges, original: ciImage)
        }
    }

    private func showEdgesOnly(_ edges: CIImage) {
        // Just show the colored edges on black background
        let context = CIContext()
        
        if let cgImage = context.createCGImage(edges, from: edges.extent) {
            let uiImage = UIImage(cgImage: cgImage, scale: 1.0, orientation: .right)
            
            DispatchQueue.main.async {
                self.segmentedImage = uiImage
                self.statusMessage = "Edges Only"
            }
        }
    }

    private func overlayEdgesOnOriginal(_ edges: CIImage, original: CIImage) {
        // Overlay colored edges on top of original image
        guard let overlayFilter = CIFilter(name: "CISourceOverCompositing") else {
            showEdgesOnly(edges)
            return
        }
        
        overlayFilter.setValue(edges, forKey: kCIInputImageKey)
        overlayFilter.setValue(original, forKey: kCIInputBackgroundImageKey)
        
        guard let result = overlayFilter.outputImage else {
            showEdgesOnly(edges)
            return
        }
        
        let context = CIContext()
        if let cgImage = context.createCGImage(result, from: result.extent) {
            let uiImage = UIImage(cgImage: cgImage, scale: 1.0, orientation: .right)
            
            DispatchQueue.main.async {
                self.segmentedImage = uiImage
                self.statusMessage = "Edge Overlay"
            }
        }
    }

    private func attemptBackgroundRemoval(edges: CIImage, original: CIImage) {
        // Create a mask from edges
        guard let dilateFilter = CIFilter(name: "CIMorphologyMaximum") else {
            overlayEdgesOnOriginal(edges, original: original)
            return
        }
        
        // Dilate edges to create mask regions
        dilateFilter.setValue(edges, forKey: kCIInputImageKey)
        dilateFilter.setValue(5.0, forKey: kCIInputRadiusKey)
        
        guard let dilatedEdges = dilateFilter.outputImage else {
            overlayEdgesOnOriginal(edges, original: original)
            return
        }
        
        // Blur the mask
        guard let blurFilter = CIFilter(name: "CIGaussianBlur") else {
            overlayEdgesOnOriginal(edges, original: original)
            return
        }
        
        blurFilter.setValue(dilatedEdges, forKey: kCIInputImageKey)
        blurFilter.setValue(10.0, forKey: kCIInputRadiusKey)
        
        guard let blurredMask = blurFilter.outputImage else {
            overlayEdgesOnOriginal(edges, original: original)
            return
        }
        
        // Threshold to create binary mask
        guard let thresholdFilter = CIFilter(name: "CIColorControls") else {
            overlayEdgesOnOriginal(edges, original: original)
            return
        }
        
        thresholdFilter.setValue(blurredMask, forKey: kCIInputImageKey)
        thresholdFilter.setValue(10.0, forKey: kCIInputContrastKey)
        thresholdFilter.setValue(0.0, forKey: kCIInputBrightnessKey)
        thresholdFilter.setValue(0.0, forKey: kCIInputSaturationKey)
        
        guard let mask = thresholdFilter.outputImage else {
            overlayEdgesOnOriginal(edges, original: original)
            return
        }
        
        // Use mask to cut out background
        guard let blendFilter = CIFilter(name: "CIBlendWithMask") else {
            overlayEdgesOnOriginal(edges, original: original)
            return
        }
        
        // Create transparent background
        let transparent = CIImage(color: CIColor(red: 0, green: 0, blue: 0, alpha: 0))
            .cropped(to: original.extent)
        
        blendFilter.setValue(original, forKey: kCIInputImageKey)
        blendFilter.setValue(transparent, forKey: kCIInputBackgroundImageKey)
        blendFilter.setValue(mask, forKey: kCIInputMaskImageKey)
        
        guard let maskedImage = blendFilter.outputImage else {
            overlayEdgesOnOriginal(edges, original: original)
            return
        }
        
        // Add edges on top for clarity
        guard let finalOverlay = CIFilter(name: "CISourceOverCompositing") else {
            showResult(maskedImage, status: "Masked")
            return
        }
        
        finalOverlay.setValue(edges, forKey: kCIInputImageKey)
        finalOverlay.setValue(maskedImage, forKey: kCIInputBackgroundImageKey)
        
        guard let finalImage = finalOverlay.outputImage else {
            showResult(maskedImage, status: "Masked")
            return
        }
        
        showResult(finalImage, status: "BG Removed")
    }

    private func showOriginal(_ image: CIImage) {
        let context = CIContext()
        if let cgImage = context.createCGImage(image, from: image.extent) {
            let uiImage = UIImage(cgImage: cgImage, scale: 1.0, orientation: .right)
            
            DispatchQueue.main.async {
                self.segmentedImage = uiImage
                self.statusMessage = "Original"
            }
        }
    }

    private func showResult(_ image: CIImage, status: String) {
        let context = CIContext()
        if let cgImage = context.createCGImage(image, from: image.extent) {
            let uiImage = UIImage(cgImage: cgImage, scale: 1.0, orientation: .right)
            
            DispatchQueue.main.async {
                self.segmentedImage = uiImage
                self.statusMessage = status
            }
        }
    }
    
    private func applyMorphologicalCleaning(buffer: UnsafeMutablePointer<UInt8>, width: Int, height: Int, bytesPerRow: Int) {
        // Simple erosion followed by dilation to remove noise
        let kernelSize = 3
        let halfKernel = kernelSize / 2
        
        // Create temporary buffer for morphological operations
        let tempBuffer = UnsafeMutablePointer<UInt8>.allocate(capacity: height * bytesPerRow)
        defer { tempBuffer.deallocate() }
        
        // Copy original to temp
        memcpy(tempBuffer, buffer, height * bytesPerRow)
        
        // Erosion (remove small isolated pixels)
        for y in halfKernel..<(height - halfKernel) {
            for x in halfKernel..<(width - halfKernel) {
                let centerIdx = y * bytesPerRow + x * 4
                
                // Check if all neighbors are foreground
                var allForeground = true
                for dy in -halfKernel...halfKernel {
                    for dx in -halfKernel...halfKernel {
                        let neighborIdx = (y + dy) * bytesPerRow + (x + dx) * 4
                        if tempBuffer[neighborIdx + 3] == 0 {
                            allForeground = false
                            break
                        }
                    }
                    if !allForeground { break }
                }
                
                // Update alpha channel
                if !allForeground {
                    buffer[centerIdx + 3] = 0
                }
            }
        }
        
        // Copy eroded result back to temp
        memcpy(tempBuffer, buffer, height * bytesPerRow)
        
        // Dilation (fill small holes)
        for y in halfKernel..<(height - halfKernel) {
            for x in halfKernel..<(width - halfKernel) {
                let centerIdx = y * bytesPerRow + x * 4
                
                // Check if any neighbor is foreground
                var anyForeground = false
                for dy in -halfKernel...halfKernel {
                    for dx in -halfKernel...halfKernel {
                        let neighborIdx = (y + dy) * bytesPerRow + (x + dx) * 4
                        if tempBuffer[neighborIdx + 3] > 0 {
                            anyForeground = true
                            // Also copy the color from the foreground neighbor
                            if buffer[centerIdx + 3] == 0 {
                                buffer[centerIdx] = tempBuffer[neighborIdx]
                                buffer[centerIdx + 1] = tempBuffer[neighborIdx + 1]
                                buffer[centerIdx + 2] = tempBuffer[neighborIdx + 2]
                            }
                            break
                        }
                    }
                    if anyForeground { break }
                }
                
                // Update alpha channel
                if anyForeground {
                    buffer[centerIdx + 3] = 255
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
