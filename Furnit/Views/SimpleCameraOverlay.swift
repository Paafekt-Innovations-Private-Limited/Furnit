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
            // ALWAYS show live camera feed (never frozen)
            CameraPreviewLayer(session: camera.session)
                .ignoresSafeArea()
            
            // Show colored outlines for each detected part
            ForEach(Array(camera.segmentedParts.enumerated()), id: \.offset) { index, part in
                if let outlineImage = part.outlineImage {
                    Image(uiImage: outlineImage)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                        .ignoresSafeArea()
                }
            }
            
            // Show final combined segmented furniture (if exists)
            if let finalImage = camera.finalSegmentedImage {
                Image(uiImage: finalImage)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .ignoresSafeArea()
            }
            
            // CENTER CROSSHAIR - Only visible when no points and no final result
            if camera.tapPoints.isEmpty && camera.finalSegmentedImage == nil {
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
            
            // Show numbered markers for each tap point (only if no final result)
            if camera.finalSegmentedImage == nil {
                ForEach(Array(camera.tapPoints.enumerated()), id: \.offset) { index, point in
                    ZStack {
                        Circle()
                            .fill(outlineColor(for: index))
                            .frame(width: 40, height: 40)
                        
                        Text("\(index + 1)")
                            .font(.system(size: 18, weight: .bold))
                            .foregroundColor(.white)
                    }
                    .position(point)
                    .animation(.easeInOut(duration: 0.3), value: camera.tapPoints.count)
                }
            }
            
            // Top bar
            VStack {
                HStack {
                    // Show "Back to Camera" button when final result is showing
                    if camera.finalSegmentedImage != nil {
                        Button(action: {
                            withAnimation {
                                camera.clearAll()
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
                    
                    // Show point count when adding points
                    if !camera.tapPoints.isEmpty && camera.finalSegmentedImage == nil {
                        HStack(spacing: 6) {
                            Image(systemName: "point.3.filled.connected.trianglepath.dotted")
                                .font(.caption)
                            Text("\(camera.tapPoints.count) part\(camera.tapPoints.count == 1 ? "" : "s")")
                                .font(.caption)
                        }
                        .foregroundColor(.yellow)
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
                
                // Bottom instructions and controls
                VStack(spacing: 10) {
                    if camera.finalSegmentedImage == nil {
                        if camera.tapPoints.isEmpty {
                            Text("TAP FURNITURE PARTS TO SELECT")
                                .font(.headline)
                                .foregroundColor(.white)
                                .padding(.horizontal, 16)
                                .padding(.vertical, 8)
                                .background(Color.green.opacity(0.8))
                                .cornerRadius(8)
                            
                            Text("Tap seat, backrest, arms - see colored outlines")
                                .font(.caption)
                                .foregroundColor(.white.opacity(0.9))
                        } else {
                            Text("TAP MORE PARTS OR COMBINE")
                                .font(.headline)
                                .foregroundColor(.white)
                                .padding(.horizontal, 16)
                                .padding(.vertical, 8)
                                .background(Color.orange.opacity(0.8))
                                .cornerRadius(8)
                        }
                    } else {
                        Text("DONE! Furniture ready for 3D room")
                            .font(.subheadline)
                            .foregroundColor(.white)
                            .padding(.horizontal, 16)
                            .padding(.vertical, 8)
                            .background(Color.green.opacity(0.8))
                            .cornerRadius(8)
                    }
                    
                    HStack(spacing: 12) {
                        // Clear Points button
                        if !camera.tapPoints.isEmpty && camera.finalSegmentedImage == nil {
                            Button(action: {
                                withAnimation {
                                    camera.clearTapPoints()
                                }
                                print("🗑️ Cleared all tap points")
                            }) {
                                HStack(spacing: 6) {
                                    Image(systemName: "trash.fill")
                                        .font(.system(size: 16))
                                    Text("Clear")
                                        .font(.system(size: 14, weight: .medium))
                                }
                                .foregroundColor(.white)
                                .padding(.horizontal, 16)
                                .padding(.vertical, 10)
                                .background(Color.red.opacity(0.8))
                                .cornerRadius(8)
                            }
                        }
                        
                        Spacer()
                        
                        // Segment All button
                        if !camera.tapPoints.isEmpty && camera.finalSegmentedImage == nil {
                            Button(action: {
                                camera.combineAllParts()
                                print("🎯 Combining all \(camera.tapPoints.count) parts")
                            }) {
                                HStack(spacing: 8) {
                                    Image(systemName: "wand.and.stars")
                                        .font(.system(size: 18))
                                    Text("Segment All")
                                        .font(.system(size: 16, weight: .bold))
                                }
                                .foregroundColor(.white)
                                .padding(.horizontal, 20)
                                .padding(.vertical, 12)
                                .background(Color.green.opacity(0.9))
                                .cornerRadius(10)
                            }
                        }
                        
                        // Capture button (only when final result ready)
                        if camera.finalSegmentedImage != nil {
                            Button(action: {
                                if let currentImage = camera.finalSegmentedImage {
                                    print("📸 Capturing segmented furniture")
                                    capturedImage = currentImage
                                    
                                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
                                        isShowingCamera = false
                                        print("📸 Closed camera overlay")
                                    }
                                } else {
                                    print("❌ No segmented image to capture!")
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
        .background(Color.clear) // Transparent background to show 3D room
        .contentShape(Rectangle())
        .onTapGesture { location in
            // Only allow tapping when no final result
            if camera.finalSegmentedImage == nil {
                camera.addTapPointAndSegment(at: location, viewSize: UIScreen.main.bounds.size)
            }
        }
        .onAppear {
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
    
    // Get outline color for each tap point
    private func outlineColor(for index: Int) -> Color {
        let colors: [Color] = [.yellow, .green, .blue, .red, .purple, .orange, .pink, .cyan]
        return colors[index % colors.count]
    }
}

struct CameraPreviewLayer: UIViewRepresentable {
    let session: AVCaptureSession
    
    func makeUIView(context: Context) -> UIView {
        let view = UIView()
        view.backgroundColor = .clear // Transparent to show 3D room
        
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

// Data model for each segmented part
struct SegmentedPart {
    let maskData: [UInt8]
    let width: Int
    let height: Int
    let outlineImage: UIImage?
}

// MobileSAM Processor
class MobileSAMProcessor: NSObject, ObservableObject {
    let session = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let videoQueue = DispatchQueue(label: "mobileSAMQueue", qos: .userInitiated)
    
    @Published var finalSegmentedImage: UIImage?
    @Published var statusMessage = "Loading..."
    @Published var tapPoints: [CGPoint] = []
    @Published var segmentedParts: [SegmentedPart] = []
    
    private var encoderModel: MLModel?
    private var decoderModel: MLModel?
    private var isProcessing = false
    private var currentPixelBuffer: CVPixelBuffer?
    private var currentImageEmbeddings: MLMultiArray?
    private var currentViewSize: CGSize = .zero
    
    // Colors for outlines
    private let outlineColors: [UIColor] = [.yellow, .green, .blue, .red, .purple, .orange, .systemPink, .cyan]
    
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
        currentPixelBuffer = nil
        currentImageEmbeddings = nil
    }
    
    func addTapPointAndSegment(at location: CGPoint, viewSize: CGSize) {
        currentViewSize = viewSize
        tapPoints.append(location)
        let index = tapPoints.count - 1
        print("📍 Added point \(tapPoints.count) at: \(location) - segmenting immediately...")
        
        DispatchQueue.main.async {
            self.statusMessage = "Detecting part \(self.tapPoints.count)..."
        }
        
        // Immediately segment this tap point
        guard let pixelBuffer = currentPixelBuffer,
              let embeddings = currentImageEmbeddings else {
            print("⚠️ Not ready for segmentation")
            return
        }
        
        DispatchQueue.global(qos: .userInitiated).async {
            self.segmentSinglePoint(location: location, viewSize: viewSize, embeddings: embeddings, pixelBuffer: pixelBuffer, index: index)
        }
    }
    
    private func segmentSinglePoint(location: CGPoint, viewSize: CGSize, embeddings: MLMultiArray, pixelBuffer: CVPixelBuffer, index: Int) {
        guard let decoder = decoderModel else { return }
        
        do {
            let normalizedX = Float(location.x / viewSize.width)
            let normalizedY = Float(location.y / viewSize.height)
            
            let pointCoords = try MLMultiArray(shape: [1, 1, 2], dataType: .float32)
            pointCoords[[0, 0, 0] as [NSNumber]] = NSNumber(value: normalizedX)
            pointCoords[[0, 0, 1] as [NSNumber]] = NSNumber(value: normalizedY)
            
            let pointLabels = try MLMultiArray(shape: [1, 1], dataType: .float32)
            pointLabels[[0, 0] as [NSNumber]] = NSNumber(value: 1)
            
            print("🎭 Running decoder for point \(index + 1)")
            
            let inputDict: [String: Any] = [
                "image_embeddings": embeddings,
                "point_coords": pointCoords,
                "point_labels": pointLabels
            ]
            
            let inputProvider = try MLDictionaryFeatureProvider(dictionary: inputDict)
            let output = try decoder.prediction(from: inputProvider)
            
            guard let masks = output.featureValue(for: "masks")?.multiArrayValue else {
                print("❌ Failed to get mask")
                return
            }
            
            let maskHeight = masks.shape[2].intValue
            let maskWidth = masks.shape[3].intValue
            
            // Extract mask data
            var maskData = [UInt8](repeating: 0, count: maskWidth * maskHeight)
            for y in 0..<maskHeight {
                for x in 0..<maskWidth {
                    let indices = [0, 0, y, x] as [NSNumber]
                    let value = masks[indices].floatValue
                    maskData[y * maskWidth + x] = value > 0.0 ? 255 : 0
                }
            }
            
            let whitePixels = maskData.filter { $0 == 255 }.count
            print("✅ Part \(index + 1): \(whitePixels) pixels detected")
            
            // Create outline image
            if let outlineImage = createOutlineImage(from: maskData, width: maskWidth, height: maskHeight, pixelBuffer: pixelBuffer, color: outlineColors[index % outlineColors.count]) {
                let part = SegmentedPart(maskData: maskData, width: maskWidth, height: maskHeight, outlineImage: outlineImage)
                
                DispatchQueue.main.async {
                    self.segmentedParts.append(part)
                    self.statusMessage = "\(self.tapPoints.count) part\(self.tapPoints.count == 1 ? "" : "s") selected"
                }
                print("✅ Outline \(index + 1) created")
            }
            
        } catch {
            print("❌ Segmentation failed: \(error)")
        }
    }
    
    private func createOutlineImage(from maskData: [UInt8], width: Int, height: Int, pixelBuffer: CVPixelBuffer, color: UIColor) -> UIImage? {
        // Create outline by detecting edges in mask
        var outlineData = [UInt8](repeating: 0, count: width * height * 4) // RGBA
        
        for y in 1..<(height-1) {
            for x in 1..<(width-1) {
                let idx = y * width + x
                let current = maskData[idx]
                
                if current == 255 {
                    // Check 8-neighbors for edge detection
                    let neighbors = [
                        maskData[idx - width - 1], maskData[idx - width], maskData[idx - width + 1],
                        maskData[idx - 1], maskData[idx + 1],
                        maskData[idx + width - 1], maskData[idx + width], maskData[idx + width + 1]
                    ]
                    
                    // If any neighbor is black, this is an edge
                    if neighbors.contains(0) {
                        let outlineIdx = idx * 4
                        var r: CGFloat = 0, g: CGFloat = 0, b: CGFloat = 0, a: CGFloat = 0
                        color.getRed(&r, green: &g, blue: &b, alpha: &a)
                        
                        outlineData[outlineIdx] = UInt8(r * 255)
                        outlineData[outlineIdx + 1] = UInt8(g * 255)
                        outlineData[outlineIdx + 2] = UInt8(b * 255)
                        outlineData[outlineIdx + 3] = 255 // Opaque outline
                    }
                }
            }
        }
        
        // Scale outline to original image size
        let originalImage = CIImage(cvPixelBuffer: pixelBuffer)
        let scaleX = originalImage.extent.width / CGFloat(width)
        let scaleY = originalImage.extent.height / CGFloat(height)
        
        // Create CGImage from outline data
        guard let provider = CGDataProvider(data: Data(outlineData) as CFData),
              let colorSpace = CGColorSpaceCreateDeviceRGB() as CGColorSpace? else {
            return nil
        }
        
        guard let cgImage = CGImage(
            width: width,
            height: height,
            bitsPerComponent: 8,
            bitsPerPixel: 32,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
            provider: provider,
            decode: nil,
            shouldInterpolate: false,
            intent: .defaultIntent
        ) else {
            return nil
        }
        
        // Scale to original size
        let ciOutline = CIImage(cgImage: cgImage)
        let scaledOutline = ciOutline.transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))
        
        let context = CIContext()
        if let scaledCGImage = context.createCGImage(scaledOutline, from: scaledOutline.extent) {
            return UIImage(cgImage: scaledCGImage, scale: 1.0, orientation: .right)
        }
        
        return nil
    }
    
    func combineAllParts() {
        guard !segmentedParts.isEmpty,
              let pixelBuffer = currentPixelBuffer else {
            print("⚠️ No parts to combine")
            return
        }
        
        DispatchQueue.main.async {
            self.statusMessage = "Combining..."
        }
        
        DispatchQueue.global(qos: .userInitiated).async {
            // Combine all masks
            let firstPart = self.segmentedParts[0]
            var combinedMask = firstPart.maskData
            
            for i in 1..<self.segmentedParts.count {
                let part = self.segmentedParts[i]
                for j in 0..<combinedMask.count {
                    combinedMask[j] = max(combinedMask[j], part.maskData[j])
                }
            }
            
            print("✅ Combined \(self.segmentedParts.count) parts")
            
            // Create final segmented image with transparency
            self.createFinalSegmentedImage(combinedMask, width: firstPart.width, height: firstPart.height, pixelBuffer: pixelBuffer)
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
            shouldInterpolate: false,
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
            .samplingLinear()
        
        guard let blendFilter = CIFilter(name: "CIBlendWithMask") else { return }
        
        let transparent = CIImage(color: .clear).cropped(to: originalImage.extent)
        
        // White mask (furniture) shows originalImage, Black mask shows transparent
        blendFilter.setValue(transparent, forKey: kCIInputBackgroundImageKey)
        blendFilter.setValue(originalImage, forKey: kCIInputImageKey)
        blendFilter.setValue(scaledMask, forKey: kCIInputMaskImageKey)
        
        guard let result = blendFilter.outputImage else { return }
        
        // Render with proper alpha channel
        let context = CIContext()
        let rgbColorSpace = CGColorSpaceCreateDeviceRGB()
        
        if let cgImage = context.createCGImage(result, from: result.extent, format: CIFormat.RGBA8, colorSpace: rgbColorSpace) {
            let uiImage = UIImage(cgImage: cgImage, scale: 1.0, orientation: .right)
            
            print("🖼️ Created final transparent furniture image")
            
            DispatchQueue.main.async {
                self.finalSegmentedImage = uiImage
                self.statusMessage = "Ready!"
                // Clear outlines since we have final result
                self.segmentedParts.removeAll()
            }
        }
    }
    
    func clearTapPoints() {
        tapPoints.removeAll()
        segmentedParts.removeAll()
        DispatchQueue.main.async {
            self.statusMessage = "Ready"
        }
    }
    
    func clearAll() {
        tapPoints.removeAll()
        segmentedParts.removeAll()
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
