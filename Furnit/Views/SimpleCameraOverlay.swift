import SwiftUI
import AVFoundation
import CoreML
import Vision
import CoreImage
import Accelerate

// COMPLETE FILE: MobileSAM with Box Prompt (Fixed)
// Uses 2 corner points to tell SAM "segment everything in this box"
// Better for multi-part furniture (bed = headboard + mattress)

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
                
                // Show final segmented furniture WITH GESTURES - on TRANSPARENT background
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
        
        // Pass view dimensions to camera for proper coordinate transformation
        camera.viewWidth = viewWidth
        camera.viewHeight = viewHeight
        
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

// MobileSAM Processor with BOX PROMPT (CORRECTED)
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
    
    // View dimensions for coordinate transformation
    var viewWidth: CGFloat = 0
    var viewHeight: CGFloat = 0
    
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
            
            print("❄️ Frame frozen - segmenting with BOX PROMPT")
            
            // Run segmentation with BOX prompt
            segmentWithBoxPrompt(embeddings: embeddings, pixelBuffer: pixelBuffer, path: path)
        }
    }
    
    // BOX PROMPT SEGMENTATION (CORRECTED)
    private func segmentWithBoxPrompt(embeddings: MLMultiArray, pixelBuffer: CVPixelBuffer, path: [CGPoint]) {
        guard let decoder = decoderModel else { return }
        
        DispatchQueue.global(qos: .userInitiated).async {
            print("📦 Using BOX PROMPT (2 corner points)")
            print("📊 View dimensions: \(self.viewWidth) x \(self.viewHeight)")
            
            // Calculate bounding box of drawn path
            let minX = path.map { $0.x }.min() ?? 0
            let maxX = path.map { $0.x }.max() ?? 1
            let minY = path.map { $0.y }.min() ?? 0
            let maxY = path.map { $0.y }.max() ?? 1
            
            let width = maxX - minX
            let height = maxY - minY
            let drawnAreaPercent = width * height * 100
            
            print("📦 Drawn region: \(String(format: "%.1f", drawnAreaPercent))% of view")
            print("   Bounds: X[\(String(format: "%.3f", minX)) to \(String(format: "%.3f", maxX))], Y[\(String(format: "%.3f", minY)) to \(String(format: "%.3f", maxY))]")
            
            let viewAspect = self.viewWidth / self.viewHeight
            
            // Transform box corners to landscape coordinates (normalized 0-1)
            let topLeft = self.transformPoint(CGPoint(x: minX, y: minY), viewAspect: viewAspect, maskAspect: 1.0)
            let bottomRight = self.transformPoint(CGPoint(x: maxX, y: maxY), viewAspect: viewAspect, maskAspect: 1.0)
            
            print("📍 Box corners (landscape, normalized):")
            print("   TL: (\(String(format: "%.3f", topLeft.x)), \(String(format: "%.3f", topLeft.y)))")
            print("   BR: (\(String(format: "%.3f", bottomRight.x)), \(String(format: "%.3f", bottomRight.y)))")
            
            var maskData: [UInt8]?
            var maskWidth = 0
            var maskHeight = 0
            
            do {
                // CRITICAL FIX: Use shape [1, 2, 2] not [1, 4]
                // [batch=1, num_points=2, coords=2]
                let pointCoords = try MLMultiArray(shape: [1, 2, 2], dataType: .float32)
                
                // Point 0: Top-left corner (normalized 0-1, NOT pixels)
                pointCoords[[0, 0, 0] as [NSNumber]] = NSNumber(value: Float(topLeft.x))
                pointCoords[[0, 0, 1] as [NSNumber]] = NSNumber(value: Float(topLeft.y))
                
                // Point 1: Bottom-right corner (normalized 0-1, NOT pixels)
                pointCoords[[0, 1, 0] as [NSNumber]] = NSNumber(value: Float(bottomRight.x))
                pointCoords[[0, 1, 1] as [NSNumber]] = NSNumber(value: Float(bottomRight.y))
                
                // Labels: both corners are foreground points (1 = foreground)
                let pointLabels = try MLMultiArray(shape: [1, 2], dataType: .float32)
                pointLabels[[0, 0] as [NSNumber]] = NSNumber(value: 1)
                pointLabels[[0, 1] as [NSNumber]] = NSNumber(value: 1)
                
                print("✅ Created point_coords shape: [1, 2, 2] (batch, points, coords)")
                print("✅ Coordinates in normalized [0, 1] range")
                
                let inputDict: [String: Any] = [
                    "image_embeddings": embeddings,
                    "point_coords": pointCoords,
                    "point_labels": pointLabels
                ]
                
                let inputProvider = try MLDictionaryFeatureProvider(dictionary: inputDict)
                let output = try decoder.prediction(from: inputProvider)
                
                guard let masks = output.featureValue(for: "masks")?.multiArrayValue else {
                    print("❌ No mask from box prompt, trying center point fallback")
                    self.segmentWithCenterPoint(embeddings: embeddings, pixelBuffer: pixelBuffer, path: path)
                    return
                }
                
                maskHeight = masks.shape[2].intValue
                maskWidth = masks.shape[3].intValue
                print("🎭 Mask dimensions: \(maskWidth) x \(maskHeight)")
                
                var tempMaskData = [UInt8](repeating: 0, count: maskWidth * maskHeight)
                var pixelCount = 0
                
                for y in 0..<maskHeight {
                    for x in 0..<maskWidth {
                        let indices = [0, 0, y, x] as [NSNumber]
                        let value = masks[indices].floatValue
                        if value > 0.0 {
                            tempMaskData[y * maskWidth + x] = 255
                            pixelCount += 1
                        }
                    }
                }
                
                maskData = tempMaskData
                let maskPercentage = Float(pixelCount) / Float(maskWidth * maskHeight) * 100
                print("✅ Box prompt result: \(pixelCount) pixels (\(String(format: "%.1f", maskPercentage))%)")
                
            } catch {
                print("❌ Box prompt failed: \(error)")
                print("   Falling back to center point")
                self.segmentWithCenterPoint(embeddings: embeddings, pixelBuffer: pixelBuffer, path: path)
                return
            }
            
            guard let combinedMask = maskData else {
                print("❌ No mask generated")
                DispatchQueue.main.async {
                    self.statusMessage = "Segmentation failed"
                    self.unfreezeFrame()
                }
                return
            }
            
            // Clip to drawn path
            let clippedMask = self.clipMaskToPath(
                maskData: combinedMask,
                maskWidth: maskWidth,
                maskHeight: maskHeight,
                path: path
            )
            
            let whitePixels = clippedMask.filter { $0 == 255 }.count
            let combinedPixels = combinedMask.filter { $0 == 255 }.count
            let retentionRatio = combinedPixels > 0 ? Float(whitePixels) / Float(combinedPixels) : 0
            print("✅ Clipped: \(whitePixels) pixels (\(Int(retentionRatio * 100))% retained)")
            
            // Refine mask
            let refinedMask = self.refineMask(clippedMask, width: maskWidth, height: maskHeight)
            let refinedPixels = refinedMask.filter { $0 == 255 }.count
            print("✨ Refined: \(refinedPixels) pixels (final)")
            
            if refinedPixels > 100 {
                self.createFinalSegmentedImage(refinedMask, width: maskWidth, height: maskHeight, pixelBuffer: pixelBuffer)
            } else {
                print("⚠️ Too few pixels, trying center point fallback")
                self.segmentWithCenterPoint(embeddings: embeddings, pixelBuffer: pixelBuffer, path: path)
            }
        }
    }
    
    // FALLBACK: Center point
    private func segmentWithCenterPoint(embeddings: MLMultiArray, pixelBuffer: CVPixelBuffer, path: [CGPoint]) {
        guard let decoder = decoderModel else { return }
        
        print("🎯 Fallback: Using CENTER POINT")
        
        let minX = path.map { $0.x }.min() ?? 0
        let maxX = path.map { $0.x }.max() ?? 1
        let minY = path.map { $0.y }.min() ?? 0
        let maxY = path.map { $0.y }.max() ?? 1
        
        let centerX = (minX + maxX) / 2
        let centerY = (minY + maxY) / 2
        
        let viewAspect = self.viewWidth / self.viewHeight
        let centerPoint = self.transformPoint(CGPoint(x: centerX, y: centerY), viewAspect: viewAspect, maskAspect: 1.0)
        
        print("   Center: (\(String(format: "%.3f", centerPoint.x)), \(String(format: "%.3f", centerPoint.y)))")
        
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
                print("❌ Center point also failed")
                DispatchQueue.main.async {
                    self.statusMessage = "Segmentation failed"
                    self.unfreezeFrame()
                }
                return
            }
            
            let maskHeight = masks.shape[2].intValue
            let maskWidth = masks.shape[3].intValue
            
            var maskData = [UInt8](repeating: 0, count: maskWidth * maskHeight)
            
            for y in 0..<maskHeight {
                for x in 0..<maskWidth {
                    let indices = [0, 0, y, x] as [NSNumber]
                    let value = masks[indices].floatValue
                    if value > 0.0 {
                        maskData[y * maskWidth + x] = 255
                    }
                }
            }
            
            // Clip and refine
            let clippedMask = self.clipMaskToPath(maskData: maskData, maskWidth: maskWidth, maskHeight: maskHeight, path: path)
            let refinedMask = self.refineMask(clippedMask, width: maskWidth, height: maskHeight)
            
            self.createFinalSegmentedImage(refinedMask, width: maskWidth, height: maskHeight, pixelBuffer: pixelBuffer)
            
        } catch {
            print("❌ Fallback failed: \(error)")
            DispatchQueue.main.async {
                self.statusMessage = "Segmentation failed"
                self.unfreezeFrame()
            }
        }
    }
    
    // MORPHOLOGICAL POST-PROCESSING
    private func refineMask(_ maskData: [UInt8], width: Int, height: Int) -> [UInt8] {
        print("🔧 Refining mask with morphological operations...")
        
        // Step 1: CLOSING (dilate then erode) - fills small holes and gaps
        let dilated1 = dilateMask(maskData, width: width, height: height, iterations: 2)
        let closed = erodeMask(dilated1, width: width, height: height, iterations: 2)
        
        // Step 2: OPENING (erode then dilate) - removes small noise
        let eroded = erodeMask(closed, width: width, height: height, iterations: 1)
        let opened = dilateMask(eroded, width: width, height: height, iterations: 1)
        
        // Step 3: Final dilation to slightly expand edges
        let finalMask = dilateMask(opened, width: width, height: height, iterations: 1)
        
        let originalWhite = maskData.filter { $0 == 255 }.count
        let refinedWhite = finalMask.filter { $0 == 255 }.count
        print("   Original: \(originalWhite) pixels → Refined: \(refinedWhite) pixels")
        
        return finalMask
    }
    
    private func dilateMask(_ maskData: [UInt8], width: Int, height: Int, iterations: Int) -> [UInt8] {
        var result = maskData
        
        for _ in 0..<iterations {
            var temp = [UInt8](repeating: 0, count: width * height)
            
            for y in 0..<height {
                for x in 0..<width {
                    let idx = y * width + x
                    
                    // If current pixel is white OR any neighbor is white, set to white
                    if result[idx] == 255 {
                        temp[idx] = 255
                    } else {
                        // Check 8-connected neighbors
                        var hasWhiteNeighbor = false
                        for dy in -1...1 {
                            for dx in -1...1 {
                                let ny = y + dy
                                let nx = x + dx
                                
                                if ny >= 0 && ny < height && nx >= 0 && nx < width {
                                    let nidx = ny * width + nx
                                    if result[nidx] == 255 {
                                        hasWhiteNeighbor = true
                                        break
                                    }
                                }
                            }
                            if hasWhiteNeighbor { break }
                        }
                        
                        if hasWhiteNeighbor {
                            temp[idx] = 255
                        }
                    }
                }
            }
            
            result = temp
        }
        
        return result
    }
    
    private func erodeMask(_ maskData: [UInt8], width: Int, height: Int, iterations: Int) -> [UInt8] {
        var result = maskData
        
        for _ in 0..<iterations {
            var temp = [UInt8](repeating: 0, count: width * height)
            
            for y in 0..<height {
                for x in 0..<width {
                    let idx = y * width + x
                    
                    if result[idx] == 0 {
                        temp[idx] = 0
                        continue
                    }
                    
                    // If current pixel is white, check if ALL neighbors are white
                    var allNeighborsWhite = true
                    for dy in -1...1 {
                        for dx in -1...1 {
                            let ny = y + dy
                            let nx = x + dx
                            
                            if ny >= 0 && ny < height && nx >= 0 && nx < width {
                                let nidx = ny * width + nx
                                if result[nidx] == 0 {
                                    allNeighborsWhite = false
                                    break
                                }
                            }
                        }
                        if !allNeighborsWhite { break }
                    }
                    
                    if allNeighborsWhite {
                        temp[idx] = 255
                    }
                }
            }
            
            result = temp
        }
        
        return result
    }
    
    // CLIP MASK to only include pixels inside drawn path
    private func clipMaskToPath(maskData: [UInt8], maskWidth: Int, maskHeight: Int, path: [CGPoint]) -> [UInt8] {
        var clippedMask = [UInt8](repeating: 0, count: maskWidth * maskHeight)
        
        let viewWidth = self.viewWidth > 0 ? self.viewWidth : UIScreen.main.bounds.width
        let viewHeight = self.viewHeight > 0 ? self.viewHeight : UIScreen.main.bounds.height
        
        let viewAspect = viewWidth / viewHeight
        let maskAspect = CGFloat(maskWidth) / CGFloat(maskHeight)
        
        print("📐 Clipping:")
        print("   View: \(Int(viewWidth))x\(Int(viewHeight)) = \(String(format: "%.3f", viewAspect)) aspect")
        print("   Mask: \(maskWidth)x\(maskHeight) = \(String(format: "%.3f", maskAspect)) aspect")
        
        // Create bezier path with ROTATION + ASPECT CORRECTION
        let bezierPath = UIBezierPath()
        if !path.isEmpty {
            let firstTransformed = transformPoint(path[0], viewAspect: viewAspect, maskAspect: maskAspect)
            bezierPath.move(to: firstTransformed)
            
            for point in path.dropFirst() {
                let transformed = transformPoint(point, viewAspect: viewAspect, maskAspect: maskAspect)
                bezierPath.addLine(to: transformed)
            }
            bezierPath.close()
        }
        
        var keptPixels = 0
        var droppedPixels = 0
        
        // Check each pixel
        for y in 0..<maskHeight {
            for x in 0..<maskWidth {
                let idx = y * maskWidth + x
                
                if maskData[idx] == 255 {
                    let maskPoint = CGPoint(
                        x: CGFloat(x) / CGFloat(maskWidth),
                        y: CGFloat(y) / CGFloat(maskHeight)
                    )
                    
                    if bezierPath.contains(maskPoint) {
                        clippedMask[idx] = 255
                        keptPixels += 1
                    } else {
                        droppedPixels += 1
                    }
                }
            }
        }
        
        print("   Kept: \(keptPixels), Dropped: \(droppedPixels)")
        
        return clippedMask
    }
    
    // Transform point: CCW rotation for camera orientation
    private func transformPoint(_ point: CGPoint, viewAspect: CGFloat, maskAspect: CGFloat) -> CGPoint {
        // Rotate 90° counter-clockwise: horizontal portrait → vertical landscape
        let rotatedPoint = CGPoint(x: point.y, y: 1.0 - point.x)
        
        // Apply aspect correction
        let rotatedViewAspect = 1.0 / viewAspect
        
        if abs(rotatedViewAspect - maskAspect) < 0.01 {
            return rotatedPoint
        }
        
        if rotatedViewAspect < maskAspect {
            let scale = rotatedViewAspect / maskAspect
            let xOffset = (1.0 - scale) / 2.0
            return CGPoint(x: xOffset + rotatedPoint.x * scale, y: rotatedPoint.y)
        } else {
            let scale = maskAspect / rotatedViewAspect
            let yOffset = (1.0 - scale) / 2.0
            return CGPoint(x: rotatedPoint.x, y: yOffset + rotatedPoint.y * scale)
        }
    }
    
    // Create furniture image with TRANSPARENT background AND RED EDGES for debugging
    private func createFinalSegmentedImage(_ maskData: [UInt8], width: Int, height: Int, pixelBuffer: CVPixelBuffer) {
        // Get original image
        let originalImage = CIImage(cvPixelBuffer: pixelBuffer)
        
        print("🖼️ Creating segmented image with RED EDGE debugging")
        
        // Create the segmented furniture image
        let context = CIContext(options: [.useSoftwareRenderer: false])
        let rgbColorSpace = CGColorSpaceCreateDeviceRGB()
        
        // Create output buffer with alpha
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
        
        // Scale factors
        let scaleX = CGFloat(width) / CGFloat(imageWidth)
        let scaleY = CGFloat(height) / CGFloat(imageHeight)
        
        // Create mask with RED EDGES for visualization
        var debugMaskData = maskData
        
        // Find edges and mark them as 128 (will render as red)
        for y in 1..<(height-1) {
            for x in 1..<(width-1) {
                let idx = y * width + x
                
                if maskData[idx] == 255 {
                    // Check if this is an edge pixel (has black neighbor)
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
                    
                    // Mark edges with value 128 (we'll make these red)
                    if isEdge {
                        debugMaskData[idx] = 128
                    }
                }
            }
        }
        
        print("🔴 Creating image with RED EDGE highlighting for debugging")
        
        // Copy pixels with mask applied
        for y in 0..<imageHeight {
            for x in 0..<imageWidth {
                // Map to mask coordinates
                let maskX = Int(CGFloat(x) * scaleX)
                let maskY = Int(CGFloat(y) * scaleY)
                let maskX_clamped = min(max(maskX, 0), width - 1)
                let maskY_clamped = min(max(maskY, 0), height - 1)
                let maskIdx = maskY_clamped * width + maskX_clamped
                
                let outIdx = y * outBytesPerRow + x * 4
                let inIdx = y * inBytesPerRow + x * 4
                
                let maskValue = debugMaskData[maskIdx]
                
                if maskValue == 128 {
                    // EDGE - draw RED for debugging
                    outData[outIdx] = 0      // B
                    outData[outIdx + 1] = 0  // G
                    outData[outIdx + 2] = 255 // R (RED!)
                    outData[outIdx + 3] = 255 // A
                } else if maskValue == 255 {
                    // INSIDE - copy original pixel
                    outData[outIdx] = inData[inIdx]
                    outData[outIdx + 1] = inData[inIdx + 1]
                    outData[outIdx + 2] = inData[inIdx + 2]
                    outData[outIdx + 3] = 255
                } else {
                    // OUTSIDE - transparent
                    outData[outIdx] = 0
                    outData[outIdx + 1] = 0
                    outData[outIdx + 2] = 0
                    outData[outIdx + 3] = 0
                }
            }
        }
        
        // Convert to UIImage
        let ciImage = CIImage(cvPixelBuffer: outBuffer)
        if let cgImage = context.createCGImage(ciImage, from: ciImage.extent, format: CIFormat.RGBA8, colorSpace: rgbColorSpace) {
            let uiImage = UIImage(cgImage: cgImage, scale: 1.0, orientation: .right)
            
            print("✅ Segmented image created successfully with red edge debugging")
            
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
