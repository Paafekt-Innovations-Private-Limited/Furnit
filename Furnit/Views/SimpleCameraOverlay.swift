import SwiftUI
import AVFoundation
import CoreML
import Vision
import CoreImage
import Accelerate

// LIVE AUTOMATIC SEGMENTATION
// Points camera at furniture → Automatically segments it
// No drawing required, picks closest object in center
// Filters out walls and floors

struct SimpleCameraOverlay: View {
    @Binding var capturedImage: UIImage?
    @Binding var isShowingCamera: Bool
    @StateObject private var camera = LiveMobileSAMProcessor()
    
    // Gesture state for furniture manipulation
    @State private var furnitureOffset: CGSize = .zero
    @State private var furnitureScale: CGFloat = 1.0
    @State private var furnitureRotation: Angle = .zero
    
    @State private var lastOffset: CGSize = .zero
    @State private var lastScale: CGFloat = 1.0
    @State private var lastRotation: Angle = .zero
    
    var body: some View {
        GeometryReader { geometry in
            ZStack {
                // Live camera feed with segmentation overlay
                if camera.capturedFinalImage == nil {
                    ZStack {
                        CameraPreviewLayer(session: camera.session)
                            .ignoresSafeArea()
                        
                        // Live segmentation overlay (semi-transparent)
                        if let liveSegmentation = camera.liveSegmentationOverlay {
                            Image(uiImage: liveSegmentation)
                                .resizable()
                                .aspectRatio(contentMode: .fill)
                                .ignoresSafeArea()
                                .opacity(0.6)  // Semi-transparent overlay
                        }
                        
                        // Crosshair to show center point
                        CrosshairView()
                        
                        // Instructions
                        VStack {
                            Spacer()
                            VStack(spacing: 8) {
                                Image(systemName: "viewfinder")
                                    .font(.largeTitle)
                                    .foregroundColor(.white)
                                Text("POINT AT FURNITURE")
                                    .font(.headline)
                                    .foregroundColor(.white)
                                Text("Center the object in frame")
                                    .font(.caption)
                                    .foregroundColor(.white.opacity(0.9))
                            }
                            .padding(.horizontal, 16)
                            .padding(.vertical, 12)
                            .background(Color.blue.opacity(0.8))
                            .cornerRadius(12)
                            .padding(.bottom, 120)
                        }
                    }
                } else {
                    // Show captured segmented furniture with gestures
                    if let finalImage = camera.capturedFinalImage {
                        Image(uiImage: finalImage)
                            .resizable()
                            .aspectRatio(contentMode: .fit)
                            .frame(maxWidth: .infinity, maxHeight: .infinity)
                            .ignoresSafeArea()
                            .offset(furnitureOffset)
                            .scaleEffect(furnitureScale)
                            .rotationEffect(furnitureRotation)
                            .gesture(
                                SimultaneousGesture(
                                    DragGesture()
                                        .onChanged { value in
                                            furnitureOffset = CGSize(
                                                width: lastOffset.width + value.translation.width,
                                                height: lastOffset.height + value.translation.height
                                            )
                                        }
                                        .onEnded { _ in
                                            lastOffset = furnitureOffset
                                        },
                                    
                                    SimultaneousGesture(
                                        MagnificationGesture()
                                            .onChanged { value in
                                                furnitureScale = lastScale * value
                                            }
                                            .onEnded { _ in
                                                lastScale = furnitureScale
                                            },
                                        
                                        RotationGesture()
                                            .onChanged { value in
                                                furnitureRotation = lastRotation + value
                                            }
                                            .onEnded { _ in
                                                lastRotation = furnitureRotation
                                            }
                                    )
                                )
                            )
                    }
                }
                
                // Top bar
                VStack {
                    HStack {
                        if camera.capturedFinalImage != nil {
                            // Reset transform button
                            if furnitureOffset != .zero || furnitureScale != 1.0 || furnitureRotation != .zero {
                                Button(action: {
                                    withAnimation(.spring()) {
                                        furnitureOffset = .zero
                                        furnitureScale = 1.0
                                        furnitureRotation = .zero
                                        lastOffset = .zero
                                        lastScale = 1.0
                                        lastRotation = .zero
                                    }
                                }) {
                                    HStack(spacing: 6) {
                                        Image(systemName: "arrow.counterclockwise")
                                            .font(.system(size: 14, weight: .semibold))
                                        Text("Reset Position")
                                            .font(.system(size: 14, weight: .medium))
                                    }
                                    .foregroundColor(.white)
                                    .padding(.horizontal, 14)
                                    .padding(.vertical, 8)
                                    .background(Color.orange.opacity(0.9))
                                    .cornerRadius(8)
                                }
                            } else {
                                // Start Over button
                                Button(action: {
                                    withAnimation {
                                        camera.resetCapture()
                                        resetGestures()
                                    }
                                }) {
                                    HStack(spacing: 6) {
                                        Image(systemName: "arrow.counterclockwise")
                                            .font(.system(size: 14, weight: .semibold))
                                        Text("Start Over")
                                            .font(.system(size: 14, weight: .medium))
                                    }
                                    .foregroundColor(.white)
                                    .padding(.horizontal, 14)
                                    .padding(.vertical, 8)
                                    .background(Color.blue.opacity(0.9))
                                    .cornerRadius(8)
                                }
                            }
                        } else {
                            // Live mode indicator
                            HStack(spacing: 8) {
                                Circle()
                                    .fill(Color.green)
                                    .frame(width: 10, height: 10)
                                    .opacity(camera.isSegmenting ? 1.0 : 0.3)
                                Text("LIVE")
                                    .font(.headline)
                                    .foregroundColor(.white)
                            }
                            .padding(.horizontal, 16)
                            .padding(.vertical, 8)
                            .background(Color.black.opacity(0.7))
                            .cornerRadius(8)
                        }
                        
                        Spacer()
                        
                        // Status indicator
                        if camera.capturedFinalImage == nil {
                            Text(camera.statusMessage)
                                .font(.caption)
                                .foregroundColor(camera.isSegmenting ? .green : .yellow)
                                .padding(.horizontal, 12)
                                .padding(.vertical, 6)
                                .background(Color.black.opacity(0.7))
                                .cornerRadius(8)
                        }
                        
                        // Gesture hints when captured
                        if camera.capturedFinalImage != nil {
                            VStack(alignment: .trailing, spacing: 2) {
                                HStack(spacing: 4) {
                                    Image(systemName: "hand.draw")
                                        .font(.caption2)
                                    Text("Drag • Pinch • Rotate")
                                        .font(.caption2)
                                }
                                Text("Scale: \(String(format: "%.1f", furnitureScale))x")
                                    .font(.caption2)
                            }
                            .foregroundColor(.white)
                            .padding(.horizontal, 12)
                            .padding(.vertical, 6)
                            .background(Color.black.opacity(0.7))
                            .cornerRadius(8)
                        }
                        
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
                    
                    // Bottom controls
                    VStack(spacing: 10) {
                        if camera.capturedFinalImage != nil {
                            Text("DRAG • PINCH • ROTATE FURNITURE")
                                .font(.subheadline)
                                .foregroundColor(.white)
                                .padding(.horizontal, 16)
                                .padding(.vertical, 8)
                                .background(Color.purple.opacity(0.8))
                                .cornerRadius(8)
                        } else {
                            // Detection info
                            if camera.lastDetectionInfo != "" {
                                Text(camera.lastDetectionInfo)
                                    .font(.caption)
                                    .foregroundColor(.white)
                                    .padding(.horizontal, 16)
                                    .padding(.vertical, 8)
                                    .background(Color.black.opacity(0.7))
                                    .cornerRadius(8)
                                    .multilineTextAlignment(.center)
                            }
                        }
                        
                        HStack(spacing: 12) {
                            Spacer()
                            
                            // Capture button (only show when segmentation is active)
                            if camera.capturedFinalImage == nil && camera.liveSegmentationOverlay != nil {
                                Button(action: {
                                    camera.captureCurrent()
                                }) {
                                    VStack(spacing: 4) {
                                        Image(systemName: "camera.circle.fill")
                                            .font(.system(size: 50))
                                            .foregroundColor(.green)
                                            .shadow(color: .black.opacity(0.5), radius: 5)
                                        
                                        Text("Capture")
                                            .font(.caption2)
                                            .foregroundColor(.white)
                                    }
                                }
                            }
                            
                            // Done button (when captured)
                            if camera.capturedFinalImage != nil {
                                Button(action: {
                                    if let currentImage = camera.capturedFinalImage {
                                        let transformedImage = applyTransforms(to: currentImage)
                                        capturedImage = transformedImage
                                        
                                        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
                                            isShowingCamera = false
                                        }
                                    }
                                }) {
                                    VStack(spacing: 4) {
                                        Image(systemName: "checkmark.circle.fill")
                                            .font(.system(size: 50))
                                            .foregroundColor(.green)
                                            .shadow(color: .black.opacity(0.5), radius: 5)
                                        
                                        Text("Done")
                                            .font(.caption2)
                                            .foregroundColor(.white)
                                    }
                                }
                            }
                        }
                        .padding(.horizontal, 20)
                    }
                    .padding(.bottom, 40)
                }
            }
        }
        .background(Color.clear)
        .onAppear {
            print("🎬 Live Segmentation Camera appeared")
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                camera.start()
            }
        }
        .onDisappear {
            print("👋 Live Segmentation Camera disappeared")
            camera.stop()
        }
    }
    
    private func resetGestures() {
        furnitureOffset = .zero
        furnitureScale = 1.0
        furnitureRotation = .zero
        lastOffset = .zero
        lastScale = 1.0
        lastRotation = .zero
    }
    
    private func applyTransforms(to image: UIImage) -> UIImage {
        if furnitureOffset == .zero && furnitureScale == 1.0 && furnitureRotation == .zero {
            return image
        }
        
        let renderer = UIGraphicsImageRenderer(size: image.size)
        
        let transformedImage = renderer.image { context in
            let cgContext = context.cgContext
            
            cgContext.translateBy(x: image.size.width / 2, y: image.size.height / 2)
            cgContext.rotate(by: CGFloat(furnitureRotation.radians))
            cgContext.scaleBy(x: furnitureScale, y: furnitureScale)
            cgContext.translateBy(x: furnitureOffset.width / furnitureScale, y: furnitureOffset.height / furnitureScale)
            
            if let cgImage = image.cgImage {
                cgContext.draw(cgImage, in: CGRect(
                    x: -image.size.width / 2,
                    y: -image.size.height / 2,
                    width: image.size.width,
                    height: image.size.height
                ))
            }
        }
        
        return transformedImage
    }
}

// Crosshair view to show center targeting
struct CrosshairView: View {
    var body: some View {
        ZStack {
            // Horizontal line
            Rectangle()
                .fill(Color.white.opacity(0.8))
                .frame(width: 40, height: 2)
            
            // Vertical line
            Rectangle()
                .fill(Color.white.opacity(0.8))
                .frame(width: 2, height: 40)
            
            // Center dot
            Circle()
                .fill(Color.green)
                .frame(width: 8, height: 8)
        }
    }
}

struct CameraPreviewLayer: UIViewRepresentable {
    let session: AVCaptureSession
    
    func makeUIView(context: Context) -> UIView {
        let view = UIView()
        view.backgroundColor = .clear
        
        let previewLayer = AVCaptureVideoPreviewLayer(session: session)
        previewLayer.videoGravity = .resizeAspectFill
        
        if #available(iOS 17.0, *) {
            previewLayer.connection?.videoRotationAngle = 90
        } else {
            previewLayer.connection?.videoOrientation = .landscapeRight
        }
        
        previewLayer.frame = view.bounds
        view.layer.addSublayer(previewLayer)
        
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

// LIVE AUTOMATIC SEGMENTATION PROCESSOR
class LiveMobileSAMProcessor: NSObject, ObservableObject {
    let session = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let videoQueue = DispatchQueue(label: "liveMobileSAMQueue", qos: .userInitiated)
    
    @Published var liveSegmentationOverlay: UIImage?
    @Published var capturedFinalImage: UIImage?
    @Published var statusMessage = "Loading..."
    @Published var isSegmenting = false
    @Published var lastDetectionInfo = ""
    
    private var encoderModel: MLModel?
    private var decoderModel: MLModel?
    private var isProcessing = false
    private var currentPixelBuffer: CVPixelBuffer?
    private var currentImageEmbeddings: MLMultiArray?
    
    // Frame skipping for performance
    private var frameCounter = 0
    private let processEveryNFrames = 3  // Process every 3rd frame (~10 FPS)
    
    // Last valid segmentation
    private var lastValidMask: [UInt8]?
    private var lastValidPixelBuffer: CVPixelBuffer?
    
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
                self.statusMessage = "Point at furniture"
            }
            print("✅ MobileSAM models loaded for live segmentation")
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
        } catch {
            print("❌ Camera setup failed: \(error)")
            session.commitConfiguration()
        }
    }
    
    func start() {
        if session.isRunning {
            session.stopRunning()
        }
        
        DispatchQueue.global(qos: .background).asyncAfter(deadline: .now() + 0.3) {
            if self.session.inputs.isEmpty {
                self.setupCamera()
            }
            
            if !self.session.isRunning {
                self.session.startRunning()
                print("📹 Live segmentation camera started")
            }
        }
    }
    
    func stop() {
        if session.isRunning {
            session.stopRunning()
        }
    }
    
    func captureCurrent() {
        guard let mask = lastValidMask,
              let pixelBuffer = lastValidPixelBuffer else {
            print("⚠️ No segmentation to capture")
            return
        }
        
        print("📸 Capturing current segmentation")
        createFinalSegmentedImage(mask, width: 256, height: 256, pixelBuffer: pixelBuffer, forCapture: true)
    }
    
    func resetCapture() {
        capturedFinalImage = nil
        DispatchQueue.main.async {
            self.statusMessage = "Point at furniture"
        }
    }
    
    // LIVE SEGMENTATION: Runs continuously on center point
    private func liveSegmentCenterObject(embeddings: MLMultiArray, pixelBuffer: CVPixelBuffer) {
        guard let decoder = decoderModel else { return }
        
        DispatchQueue.main.async {
            self.isSegmenting = true
        }
        
        // Use center point (0.5, 0.5) to segment whatever is in the middle
        let centerPoint = CGPoint(x: 0.5, y: 0.5)
        
        do {
            let pointCoords = try MLMultiArray(shape: [1, 1, 2], dataType: .float32)
            pointCoords[[0, 0, 0] as [NSNumber]] = NSNumber(value: Float(centerPoint.x))
            pointCoords[[0, 0, 1] as [NSNumber]] = NSNumber(value: Float(centerPoint.y))
            
            let pointLabels = try MLMultiArray(shape: [1, 1], dataType: .float32)
            pointLabels[[0, 0] as [NSNumber]] = NSNumber(value: 1)
            
            let inputDict: [String: Any] = [
                "image_embeddings": embeddings,
                "point_coords": pointCoords,
                "point_labels": pointLabels
            ]
            
            let inputProvider = try MLDictionaryFeatureProvider(dictionary: inputDict)
            let output = try decoder.prediction(from: inputProvider)
            
            guard let masks = output.featureValue(for: "masks")?.multiArrayValue else {
                DispatchQueue.main.async {
                    self.isSegmenting = false
                }
                return
            }
            
            let maskHeight = masks.shape[2].intValue
            let maskWidth = masks.shape[3].intValue
            
            var maskData = [UInt8](repeating: 0, count: maskWidth * maskHeight)
            var pixelCount = 0
            
            for y in 0..<maskHeight {
                for x in 0..<maskWidth {
                    let indices = [0, 0, y, x] as [NSNumber]
                    let value = masks[indices].floatValue
                    if value > 0.0 {
                        maskData[y * maskWidth + x] = 255
                        pixelCount += 1
                    }
                }
            }
            
            let maskPercentage = Float(pixelCount) / Float(maskWidth * maskHeight) * 100
            
            // FILTER: Reject walls (too large) and noise (too small)
            let minPercentage: Float = 2.0   // Min 2% of frame
            let maxPercentage: Float = 50.0  // Max 50% of frame (walls are bigger)
            
            if maskPercentage < minPercentage {
                print("⚠️ Detection too small (\(String(format: "%.1f", maskPercentage))%) - likely noise")
                DispatchQueue.main.async {
                    self.lastDetectionInfo = "Object too small - move closer"
                    self.liveSegmentationOverlay = nil
                    self.isSegmenting = false
                }
                return
            }
            
            if maskPercentage > maxPercentage {
                print("⚠️ Detection too large (\(String(format: "%.1f", maskPercentage))%) - likely wall/floor")
                DispatchQueue.main.async {
                    self.lastDetectionInfo = "Too large - move back or point at furniture"
                    self.liveSegmentationOverlay = nil
                    self.isSegmenting = false
                }
                return
            }
            
            // Valid detection!
            print("✅ Live detection: \(pixelCount) pixels (\(String(format: "%.1f", maskPercentage))%)")
            
            // Store for capture
            self.lastValidMask = maskData
            self.lastValidPixelBuffer = pixelBuffer
            
            // Create overlay image
            self.createFinalSegmentedImage(maskData, width: maskWidth, height: maskHeight, pixelBuffer: pixelBuffer, forCapture: false)
            
            DispatchQueue.main.async {
                self.lastDetectionInfo = "✓ Object detected (\(String(format: "%.0f", maskPercentage))% of frame)"
                self.isSegmenting = false
            }
            
        } catch {
            print("❌ Live segmentation failed: \(error)")
            DispatchQueue.main.async {
                self.isSegmenting = false
            }
        }
    }
    
    // Create segmented image with transparent background and green edges
    private func createFinalSegmentedImage(_ maskData: [UInt8], width: Int, height: Int, pixelBuffer: CVPixelBuffer, forCapture: Bool) {
        let originalImage = CIImage(cvPixelBuffer: pixelBuffer)
        let context = CIContext(options: [.useSoftwareRenderer: false])
        let rgbColorSpace = CGColorSpaceCreateDeviceRGB()
        
        var outputBuffer: CVPixelBuffer?
        let attrs = [
            kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue,
            kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue,
            kCVPixelBufferPixelFormatTypeKey: kCVPixelFormatType_32BGRA
        ] as CFDictionary
        
        CVPixelBufferCreate(
            kCFAllocatorDefault,
            Int(originalImage.extent.width),
            Int(originalImage.extent.height),
            kCVPixelFormatType_32BGRA,
            attrs,
            &outputBuffer
        )
        
        guard let outBuffer = outputBuffer else { return }
        
        CVPixelBufferLockBaseAddress(outBuffer, [])
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        
        defer {
            CVPixelBufferUnlockBaseAddress(outBuffer, [])
            CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly)
        }
        
        guard let outBaseAddress = CVPixelBufferGetBaseAddress(outBuffer),
              let inBaseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            return
        }
        
        let outBytesPerRow = CVPixelBufferGetBytesPerRow(outBuffer)
        let inBytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let imageWidth = Int(originalImage.extent.width)
        let imageHeight = Int(originalImage.extent.height)
        
        let outData = outBaseAddress.assumingMemoryBound(to: UInt8.self)
        let inData = inBaseAddress.assumingMemoryBound(to: UInt8.self)
        
        let scaleX = CGFloat(width) / CGFloat(imageWidth)
        let scaleY = CGFloat(height) / CGFloat(imageHeight)
        
        // Find edges for visualization
        var debugMaskData = maskData
        for y in 1..<(height-1) {
            for x in 1..<(width-1) {
                let idx = y * width + x
                
                if maskData[idx] == 255 {
                    var isEdge = false
                    for dy in -1...1 {
                        for dx in -1...1 {
                            if dx == 0 && dy == 0 { continue }
                            let nx = x + dx
                            let ny = y + dy
                            let nidx = ny * width + nx
                            if maskData[nidx] == 0 {
                                isEdge = true
                                break
                            }
                        }
                        if isEdge { break }
                    }
                    if isEdge {
                        debugMaskData[idx] = 128  // Mark edges
                    }
                }
            }
        }
        
        // Copy pixels with mask applied
        for y in 0..<imageHeight {
            for x in 0..<imageWidth {
                let maskX = Int(CGFloat(x) * scaleX)
                let maskY = Int(CGFloat(y) * scaleY)
                let maskX_clamped = min(max(maskX, 0), width - 1)
                let maskY_clamped = min(max(maskY, 0), height - 1)
                let maskIdx = maskY_clamped * width + maskX_clamped
                
                let outIdx = y * outBytesPerRow + x * 4
                let inIdx = y * inBytesPerRow + x * 4
                
                let maskValue = debugMaskData[maskIdx]
                
                if forCapture {
                    // For capture: solid with transparent background
                    if maskValue == 128 || maskValue == 255 {
                        // Copy original pixel
                        outData[outIdx] = inData[inIdx]
                        outData[outIdx + 1] = inData[inIdx + 1]
                        outData[outIdx + 2] = inData[inIdx + 2]
                        outData[outIdx + 3] = 255
                    } else {
                        // Transparent background
                        outData[outIdx] = 0
                        outData[outIdx + 1] = 0
                        outData[outIdx + 2] = 0
                        outData[outIdx + 3] = 0
                    }
                } else {
                    // For live overlay: green edges, transparent interior
                    if maskValue == 128 {
                        // GREEN edges for live overlay
                        outData[outIdx] = 0        // B
                        outData[outIdx + 1] = 255  // G (GREEN!)
                        outData[outIdx + 2] = 0    // R
                        outData[outIdx + 3] = 200  // Semi-transparent
                    } else if maskValue == 255 {
                        // Semi-transparent green fill
                        outData[outIdx] = 0
                        outData[outIdx + 1] = 255
                        outData[outIdx + 2] = 0
                        outData[outIdx + 3] = 50  // Very transparent
                    } else {
                        // Transparent
                        outData[outIdx] = 0
                        outData[outIdx + 1] = 0
                        outData[outIdx + 2] = 0
                        outData[outIdx + 3] = 0
                    }
                }
            }
        }
        
        // Convert to UIImage
        let ciImage = CIImage(cvPixelBuffer: outBuffer)
        if let cgImage = context.createCGImage(ciImage, from: ciImage.extent, format: CIFormat.RGBA8, colorSpace: rgbColorSpace) {
            let uiImage = UIImage(cgImage: cgImage, scale: 1.0, orientation: .right)
            
            DispatchQueue.main.async {
                if forCapture {
                    self.capturedFinalImage = uiImage
                    print("📸 Captured final segmented image")
                } else {
                    self.liveSegmentationOverlay = uiImage
                }
            }
        }
    }
}

extension LiveMobileSAMProcessor: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        
        // Skip capture if already captured
        if capturedFinalImage != nil { return }
        
        currentPixelBuffer = pixelBuffer
        
        // Frame skipping for performance
        frameCounter += 1
        if frameCounter % processEveryNFrames != 0 {
            return
        }
        
        if !isProcessing {
            isProcessing = true
            
            DispatchQueue.global(qos: .userInitiated).async {
                // 1. Run encoder
                self.runEncoder(pixelBuffer: pixelBuffer)
                
                // 2. Immediately run decoder for live segmentation
                if let embeddings = self.currentImageEmbeddings {
                    self.liveSegmentCenterObject(embeddings: embeddings, pixelBuffer: pixelBuffer)
                }
                
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

struct SimpleCameraOverlay_Previews: PreviewProvider {
    static var previews: some View {
        SimpleCameraOverlay(
            capturedImage: .constant(nil),
            isShowingCamera: .constant(true)
        )
    }
}
