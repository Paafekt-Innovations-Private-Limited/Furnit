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
    
    // Freeform drawing state
    @State private var drawingPath: [CGPoint] = []
    @State private var isDrawing: Bool = false
    
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
                // Live camera feed OR frozen frame
                if let frozenFrame = camera.frozenFrame {
                    // Show frozen frame
                    Image(uiImage: frozenFrame)
                        .resizable()
                        .aspectRatio(contentMode: .fill)
                        .ignoresSafeArea()
                } else {
                    // Live camera feed with freeform drawing capability
                    ZStack {
                        CameraPreviewLayer(session: camera.session)
                            .ignoresSafeArea()
                        
                        // Freeform drawing path overlay
                        if !drawingPath.isEmpty {
                            Path { path in
                                if let first = drawingPath.first {
                                    path.move(to: first)
                                    for point in drawingPath.dropFirst() {
                                        path.addLine(to: point)
                                    }
                                }
                            }
                            .stroke(Color.green, lineWidth: 4)
                            .shadow(color: .black.opacity(0.5), radius: 2)
                            
                            // Show filled area with transparency
                            Path { path in
                                if let first = drawingPath.first {
                                    path.move(to: first)
                                    for point in drawingPath.dropFirst() {
                                        path.addLine(to: point)
                                    }
                                    path.closeSubpath()
                                }
                            }
                            .fill(Color.green.opacity(0.15))
                        }
                        
                        // Instruction overlay on live feed
                        if !isDrawing && drawingPath.isEmpty {
                            VStack {
                                Spacer()
                                VStack(spacing: 8) {
                                    Image(systemName: "hand.draw")
                                        .font(.largeTitle)
                                        .foregroundColor(.white)
                                    Text("DRAW AROUND FURNITURE")
                                        .font(.headline)
                                        .foregroundColor(.white)
                                    Text("Trace outline with your finger")
                                        .font(.caption)
                                        .foregroundColor(.white.opacity(0.9))
                                }
                                .padding(.horizontal, 16)
                                .padding(.vertical, 12)
                                .background(Color.green.opacity(0.8))
                                .cornerRadius(12)
                                .padding(.bottom, 120)
                            }
                        } else if isDrawing {
                            VStack {
                                Spacer()
                                Text("KEEP DRAWING...")
                                    .font(.headline)
                                    .foregroundColor(.white)
                                    .padding(.horizontal, 16)
                                    .padding(.vertical, 8)
                                    .background(Color.orange.opacity(0.8))
                                    .cornerRadius(8)
                                    .padding(.bottom, 120)
                            }
                        } else if !drawingPath.isEmpty {
                            VStack {
                                Spacer()
                                HStack(spacing: 16) {
                                    // Clear button
                                    Button(action: {
                                        withAnimation {
                                            drawingPath.removeAll()
                                        }
                                    }) {
                                        HStack(spacing: 6) {
                                            Image(systemName: "xmark.circle.fill")
                                                .font(.system(size: 18))
                                            Text("Clear")
                                                .font(.system(size: 16, weight: .semibold))
                                        }
                                        .foregroundColor(.white)
                                        .padding(.horizontal, 20)
                                        .padding(.vertical, 12)
                                        .background(Color.red.opacity(0.9))
                                        .cornerRadius(10)
                                    }
                                    
                                    // Process button
                                    Button(action: {
                                        processDrawing(geometry: geometry)
                                    }) {
                                        HStack(spacing: 6) {
                                            Image(systemName: "wand.and.stars")
                                                .font(.system(size: 18))
                                            Text("Segment")
                                                .font(.system(size: 16, weight: .bold))
                                        }
                                        .foregroundColor(.white)
                                        .padding(.horizontal, 20)
                                        .padding(.vertical, 12)
                                        .background(Color.green.opacity(0.9))
                                        .cornerRadius(10)
                                    }
                                }
                                .padding(.bottom, 100)
                            }
                        }
                    }
                    .gesture(
                        DragGesture(minimumDistance: 0)
                            .onChanged { value in
                                if !isDrawing {
                                    isDrawing = true
                                    drawingPath.removeAll()
                                }
                                drawingPath.append(value.location)
                            }
                            .onEnded { _ in
                                isDrawing = false
                                print("✏️ Drawing complete: \(drawingPath.count) points")
                            }
                    )
                }
                
                // Show final segmented furniture WITH GESTURES
                if let finalImage = camera.finalSegmentedImage {
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
                
                // Top bar
                VStack {
                    HStack {
                        // Show different buttons based on state
                        if camera.finalSegmentedImage != nil {
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
                                        camera.resetAll()
                                        resetGestures()
                                        drawingPath.removeAll()
                                    }
                                    print("🔄 Reset to live camera")
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
                        } else if camera.frozenFrame != nil {
                            // Frozen state - show back to live
                            Button(action: {
                                withAnimation {
                                    camera.unfreezeFrame()
                                    drawingPath.removeAll()
                                }
                                print("🔄 Back to live camera")
                            }) {
                                HStack(spacing: 6) {
                                    Image(systemName: "play.fill")
                                        .font(.system(size: 14, weight: .semibold))
                                    Text("Back to Live")
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
                        
                        // Show gesture hints when furniture is shown
                        if camera.finalSegmentedImage != nil {
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
                    
                    // Bottom controls
                    VStack(spacing: 10) {
                        if camera.finalSegmentedImage != nil {
                            Text("DRAG • PINCH • ROTATE FURNITURE")
                                .font(.subheadline)
                                .foregroundColor(.white)
                                .padding(.horizontal, 16)
                                .padding(.vertical, 8)
                                .background(Color.purple.opacity(0.8))
                                .cornerRadius(8)
                        } else if camera.frozenFrame != nil {
                            Text("PROCESSING...")
                                .font(.headline)
                                .foregroundColor(.white)
                                .padding(.horizontal, 16)
                                .padding(.vertical, 8)
                                .background(Color.orange.opacity(0.8))
                                .cornerRadius(8)
                        }
                        
                        HStack(spacing: 12) {
                            Spacer()
                            
                            // Capture button
                            if camera.finalSegmentedImage != nil {
                                Button(action: {
                                    if let currentImage = camera.finalSegmentedImage {
                                        let transformedImage = applyTransforms(to: currentImage)
                                        
                                        print("📸 Capturing segmented furniture with transforms")
                                        capturedImage = transformedImage
                                        
                                        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
                                            isShowingCamera = false
                                            print("📸 Closed camera overlay")
                                        }
                                    }
                                }) {
                                    VStack(spacing: 4) {
                                        Image(systemName: "checkmark.circle.fill")
                                            .font(.system(size: 50))
                                            .foregroundColor(.green)
                                            .shadow(color: .black.opacity(0.5), radius: 5)
                                        
                                        Text("Capture")
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
            print("🎬 SimpleCameraOverlay appeared")
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                camera.start()
            }
        }
        .onDisappear {
            print("👋 SimpleCameraOverlay disappeared")
            camera.stop()
        }
    }
    
    private func processDrawing(geometry: GeometryProxy) {
        guard !drawingPath.isEmpty else { return }
        
        // Convert path points to normalized coordinates
        let viewWidth = geometry.size.width
        let viewHeight = geometry.size.height
        
        let normalizedPath = drawingPath.map { point in
            CGPoint(
                x: point.x / viewWidth,
                y: point.y / viewHeight
            )
        }
        
        print("✏️ Processing drawn path with \(normalizedPath.count) points")
        
        // Freeze and segment using the drawn path
        camera.freezeAndSegmentPath(normalizedPath)
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

// MobileSAM Processor
class MobileSAMProcessor: NSObject, ObservableObject {
    let session = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let videoQueue = DispatchQueue(label: "mobileSAMQueue", qos: .userInitiated)
    
    @Published var frozenFrame: UIImage?
    @Published var finalSegmentedImage: UIImage?
    @Published var statusMessage = "Loading..."
    
    private var encoderModel: MLModel?
    private var decoderModel: MLModel?
    private var isProcessing = false
    private var currentPixelBuffer: CVPixelBuffer?
    private var frozenPixelBuffer: CVPixelBuffer?
    private var currentImageEmbeddings: MLMultiArray?
    
    override init() {
        super.init()
        loadModels()
        setupCamera()
    }
    
    private func loadModels() {
        DispatchQueue.main.async {
            self.statusMessage = "Loading..."
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
                self.statusMessage = "Ready"
            }
            print("✅ MobileSAM models loaded")
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
                print("📹 Camera session started")
            }
        }
    }
    
    func stop() {
        if session.isRunning {
            session.stopRunning()
        }
    }
    
    func freezeAndSegmentPath(_ path: [CGPoint]) {
        guard let pixelBuffer = currentPixelBuffer,
              let embeddings = currentImageEmbeddings else {
            print("⚠️ No frame to freeze")
            return
        }
        
        // Create UIImage from pixel buffer
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let context = CIContext()
        if let cgImage = context.createCGImage(ciImage, from: ciImage.extent) {
            let uiImage = UIImage(cgImage: cgImage, scale: 1.0, orientation: .right)
            
            DispatchQueue.main.async {
                self.frozenFrame = uiImage
                self.statusMessage = "Segmenting..."
            }
            
            // Store frozen buffer
            frozenPixelBuffer = pixelBuffer
            
            print("❄️ Frame frozen - segmenting drawn path with \(path.count) points")
            
            // Run segmentation with path points
            segmentWithPath(embeddings: embeddings, pixelBuffer: pixelBuffer, path: path)
        }
    }
    
    private func segmentWithPath(embeddings: MLMultiArray, pixelBuffer: CVPixelBuffer, path: [CGPoint]) {
        guard let decoder = decoderModel else { return }
        
        DispatchQueue.global(qos: .userInitiated).async {
            print("✏️ Sampling points along drawn path")
            
            // Sample points evenly along the path (every Nth point)
            let sampleInterval = max(1, path.count / 20) // Sample ~20 points
            var sampledPoints: [CGPoint] = []
            
            for (index, point) in path.enumerated() where index % sampleInterval == 0 {
                sampledPoints.append(point)
            }
            
            print("🎯 Sampled \(sampledPoints.count) points from path")
            
            var allMaskData: [UInt8]?
            var maskWidth = 0
            var maskHeight = 0
            
            // Process each sampled point
            for (index, point) in sampledPoints.enumerated() {
                do {
                    // Create SINGLE point prompt
                    let pointCoords = try MLMultiArray(shape: [1, 1, 2], dataType: .float32)
                    pointCoords[[0, 0, 0] as [NSNumber]] = NSNumber(value: Float(point.x))
                    pointCoords[[0, 0, 1] as [NSNumber]] = NSNumber(value: Float(point.y))
                    
                    // Label: 1 = foreground point
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
                        continue
                    }
                    
                    maskHeight = masks.shape[2].intValue
                    maskWidth = masks.shape[3].intValue
                    
                    var maskData = [UInt8](repeating: 0, count: maskWidth * maskHeight)
                    for y in 0..<maskHeight {
                        for x in 0..<maskWidth {
                            let indices = [0, 0, y, x] as [NSNumber]
                            let value = masks[indices].floatValue
                            maskData[y * maskWidth + x] = value > 0.0 ? 255 : 0
                        }
                    }
                    
                    // Combine with previous masks (union)
                    if allMaskData == nil {
                        allMaskData = maskData
                    } else {
                        for i in 0..<maskData.count {
                            allMaskData![i] = max(allMaskData![i], maskData[i])
                        }
                    }
                    
                    print("✅ Processed point \(index + 1)/\(sampledPoints.count)")
                    
                } catch {
                    print("❌ Point segmentation failed at index \(index): \(error)")
                }
            }
            
            if let finalMask = allMaskData {
                let whitePixels = finalMask.filter { $0 == 255 }.count
                print("✅ Combined segmentation: \(whitePixels) pixels from \(sampledPoints.count) points")
                
                // Create final segmented image
                self.createFinalSegmentedImage(finalMask, width: maskWidth, height: maskHeight, pixelBuffer: pixelBuffer)
            } else {
                print("❌ No masks generated")
                DispatchQueue.main.async {
                    self.statusMessage = "No furniture detected"
                }
            }
        }
    }
    
    private func createFinalSegmentedImage(_ maskData: [UInt8], width: Int, height: Int, pixelBuffer: CVPixelBuffer) {
        guard let provider = CGDataProvider(data: Data(maskData) as CFData),
              let colorSpace = CGColorSpace(name: CGColorSpace.linearGray) else {
            return
        }
        
        guard let cgMask = CGImage(
            width: width,
            height: height,
            bitsPerComponent: 8,
            bitsPerPixel: 8,
            bytesPerRow: width,
            space: colorSpace,
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
            provider: provider,
            decode: nil,
            shouldInterpolate: true,
            intent: .defaultIntent
        ) else {
            return
        }
        
        let maskImage = CIImage(cgImage: cgMask)
        let originalImage = CIImage(cvPixelBuffer: pixelBuffer)
        
        let scaleX = originalImage.extent.width / CGFloat(width)
        let scaleY = originalImage.extent.height / CGFloat(height)
        
        let scaledMask = maskImage
            .transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))
            .samplingNearest()
        
        guard let blendFilter = CIFilter(name: "CIBlendWithMask") else { return }
        
        let transparent = CIImage(color: .clear).cropped(to: originalImage.extent)
        
        blendFilter.setValue(originalImage, forKey: kCIInputImageKey)
        blendFilter.setValue(transparent, forKey: kCIInputBackgroundImageKey)
        blendFilter.setValue(scaledMask, forKey: kCIInputMaskImageKey)
        
        guard let result = blendFilter.outputImage else { return }
        
        let context = CIContext(options: [.useSoftwareRenderer: false])
        let rgbColorSpace = CGColorSpaceCreateDeviceRGB()
        
        if let cgImage = context.createCGImage(result, from: result.extent, format: CIFormat.RGBA8, colorSpace: rgbColorSpace) {
            let uiImage = UIImage(cgImage: cgImage, scale: 1.0, orientation: .right)
            
            print("🖼️ Created final furniture with transparency")
            
            DispatchQueue.main.async {
                self.finalSegmentedImage = uiImage
                self.statusMessage = "Ready!"
            }
        }
    }
    
    func unfreezeFrame() {
        frozenFrame = nil
        frozenPixelBuffer = nil
        finalSegmentedImage = nil
        DispatchQueue.main.async {
            self.statusMessage = "Ready"
        }
    }
    
    func resetAll() {
        frozenFrame = nil
        frozenPixelBuffer = nil
        finalSegmentedImage = nil
        DispatchQueue.main.async {
            self.statusMessage = "Ready"
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

struct SimpleCameraOverlay_Previews: PreviewProvider {
    static var previews: some View {
        SimpleCameraOverlay(
            capturedImage: .constant(nil),
            isShowingCamera: .constant(true)
        )
    }
}
