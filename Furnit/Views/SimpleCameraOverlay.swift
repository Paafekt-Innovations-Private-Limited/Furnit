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
            // Live camera feed OR frozen frame
            if let frozenFrame = camera.frozenFrame {
                // Show frozen frame
                Image(uiImage: frozenFrame)
                    .resizable()
                    .aspectRatio(contentMode: .fill)
                    .ignoresSafeArea()
            } else {
                // Live camera feed
                CameraPreviewLayer(session: camera.session)
                    .ignoresSafeArea()
            }
            
            // Show all detected parts with outlines (white = unselected, green = selected)
            ForEach(Array(camera.detectedParts.enumerated()), id: \.offset) { index, part in
                if let outlineImage = part.outlineImage {
                    Image(uiImage: outlineImage)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                        .ignoresSafeArea()
                        .onTapGesture {
                            camera.togglePartSelection(index: index)
                        }
                }
            }
            
            // Show final combined furniture
            if let finalImage = camera.finalSegmentedImage {
                Image(uiImage: finalImage)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .ignoresSafeArea()
            }
            
            // CENTER CROSSHAIR - Only visible in live mode
            if camera.frozenFrame == nil && camera.detectedParts.isEmpty && camera.finalSegmentedImage == nil {
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
            
            // Top bar
            VStack {
                HStack {
                    // Show different buttons based on state
                    if camera.finalSegmentedImage != nil {
                        // Final result - show reset button
                        Button(action: {
                            withAnimation {
                                camera.resetAll()
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
                    } else if camera.frozenFrame != nil {
                        // Frozen state - show back to live
                        Button(action: {
                            withAnimation {
                                camera.unfreezeFrame()
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
                    
                    // Show selection count when parts are detected
                    if !camera.detectedParts.isEmpty && camera.finalSegmentedImage == nil {
                        let selectedCount = camera.detectedParts.filter { $0.isSelected }.count
                        HStack(spacing: 6) {
                            Image(systemName: "checkmark.circle.fill")
                                .font(.caption)
                            Text("\(selectedCount) of \(camera.detectedParts.count) selected")
                                .font(.caption)
                        }
                        .foregroundColor(.green)
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
                        if camera.frozenFrame == nil {
                            Text("FREEZE FRAME TO START SELECTION")
                                .font(.headline)
                                .foregroundColor(.white)
                                .padding(.horizontal, 16)
                                .padding(.vertical, 8)
                                .background(Color.blue.opacity(0.8))
                                .cornerRadius(8)
                            
                            Text("Tap freeze to detect all furniture parts")
                                .font(.caption)
                                .foregroundColor(.white.opacity(0.9))
                        } else if camera.detectedParts.isEmpty {
                            Text("DETECTING FURNITURE PARTS...")
                                .font(.headline)
                                .foregroundColor(.white)
                                .padding(.horizontal, 16)
                                .padding(.vertical, 8)
                                .background(Color.orange.opacity(0.8))
                                .cornerRadius(8)
                        } else {
                            Text("TAP WHITE PARTS TO SELECT (GREEN)")
                                .font(.headline)
                                .foregroundColor(.white)
                                .padding(.horizontal, 16)
                                .padding(.vertical, 8)
                                .background(Color.green.opacity(0.8))
                                .cornerRadius(8)
                            
                            Text("Tap again to deselect • Select all chair parts")
                                .font(.caption)
                                .foregroundColor(.white.opacity(0.9))
                        }
                    } else {
                        Text("READY! Furniture extracted")
                            .font(.subheadline)
                            .foregroundColor(.white)
                            .padding(.horizontal, 16)
                            .padding(.vertical, 8)
                            .background(Color.green.opacity(0.8))
                            .cornerRadius(8)
                    }
                    
                    HStack(spacing: 12) {
                        // Freeze/Unfreeze button
                        if camera.frozenFrame == nil && camera.finalSegmentedImage == nil {
                            Button(action: {
                                camera.freezeAndDetect()
                                print("❄️ Freezing frame and detecting parts...")
                            }) {
                                HStack(spacing: 8) {
                                    Image(systemName: "pause.circle.fill")
                                        .font(.system(size: 20))
                                    Text("Freeze & Detect")
                                        .font(.system(size: 16, weight: .bold))
                                }
                                .foregroundColor(.white)
                                .padding(.horizontal, 20)
                                .padding(.vertical, 12)
                                .background(Color.blue.opacity(0.9))
                                .cornerRadius(10)
                            }
                        }
                        
                        Spacer()
                        
                        // Combine Selected button
                        if !camera.detectedParts.isEmpty && camera.finalSegmentedImage == nil {
                            let selectedCount = camera.detectedParts.filter { $0.isSelected }.count
                            Button(action: {
                                camera.combineSelectedParts()
                                print("🎯 Combining \(selectedCount) selected parts")
                            }) {
                                HStack(spacing: 8) {
                                    Image(systemName: "wand.and.stars")
                                        .font(.system(size: 18))
                                    Text("Combine Selected")
                                        .font(.system(size: 16, weight: .bold))
                                }
                                .foregroundColor(.white)
                                .padding(.horizontal, 20)
                                .padding(.vertical, 12)
                                .background(selectedCount > 0 ? Color.green.opacity(0.9) : Color.gray.opacity(0.5))
                                .cornerRadius(10)
                            }
                            .disabled(selectedCount == 0)
                        }
                        
                        // Capture button
                        if camera.finalSegmentedImage != nil {
                            Button(action: {
                                if let currentImage = camera.finalSegmentedImage {
                                    print("📸 Capturing segmented furniture")
                                    capturedImage = currentImage
                                    
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

// Data model for each detected part
struct DetectedPart {
    let maskData: [UInt8]
    let width: Int
    let height: Int
    var isSelected: Bool
    var outlineImage: UIImage?
}

// MobileSAM Processor
class MobileSAMProcessor: NSObject, ObservableObject {
    let session = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let videoQueue = DispatchQueue(label: "mobileSAMQueue", qos: .userInitiated)
    
    @Published var frozenFrame: UIImage?
    @Published var detectedParts: [DetectedPart] = []
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
    
    func freezeAndDetect() {
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
                self.statusMessage = "Detecting..."
            }
            
            // Store frozen buffer
            frozenPixelBuffer = pixelBuffer
            
            print("❄️ Frame frozen - detecting all parts...")
            
            // Run detection on grid
            detectAllParts(embeddings: embeddings, pixelBuffer: pixelBuffer)
        }
    }
    
    private func detectAllParts(embeddings: MLMultiArray, pixelBuffer: CVPixelBuffer) {
        guard let decoder = decoderModel else { return }
        
        let viewWidth = UIScreen.main.bounds.width
        let viewHeight = UIScreen.main.bounds.height
        
        // Create a 5x5 grid of points
        let gridSize = 5
        var detectedMasks: [DetectedPart] = []
        
        DispatchQueue.global(qos: .userInitiated).async {
            for row in 0..<gridSize {
                for col in 0..<gridSize {
                    let normalizedX = Float(col + 1) / Float(gridSize + 1)
                    let normalizedY = Float(row + 1) / Float(gridSize + 1)
                    
                    do {
                        let pointCoords = try MLMultiArray(shape: [1, 1, 2], dataType: .float32)
                        pointCoords[[0, 0, 0] as [NSNumber]] = NSNumber(value: normalizedX)
                        pointCoords[[0, 0, 1] as [NSNumber]] = NSNumber(value: normalizedY)
                        
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
                        
                        let maskHeight = masks.shape[2].intValue
                        let maskWidth = masks.shape[3].intValue
                        
                        var maskData = [UInt8](repeating: 0, count: maskWidth * maskHeight)
                        for y in 0..<maskHeight {
                            for x in 0..<maskWidth {
                                let indices = [0, 0, y, x] as [NSNumber]
                                let value = masks[indices].floatValue
                                maskData[y * maskWidth + x] = value > 0.0 ? 255 : 0
                            }
                        }
                        
                        // SMOOTH THE MASK for cleaner edges
                        maskData = self.smoothMask(maskData, width: maskWidth, height: maskHeight)
                        
                        let whitePixels = maskData.filter { $0 == 255 }.count
                        
                        // Only keep significant segments (at least 1% of mask)
                        if whitePixels > (maskWidth * maskHeight) / 100 {
                            // Check if this mask is similar to any existing mask
                            let isDuplicate = detectedMasks.contains { existingPart in
                                self.isSimilarMask(maskData, existingPart.maskData, threshold: 0.8)
                            }
                            
                            if !isDuplicate {
                                if let outlineImage = self.createSmoothOutline(from: maskData, width: maskWidth, height: maskHeight, pixelBuffer: pixelBuffer, color: .white) {
                                    let part = DetectedPart(
                                        maskData: maskData,
                                        width: maskWidth,
                                        height: maskHeight,
                                        isSelected: false,
                                        outlineImage: outlineImage
                                    )
                                    detectedMasks.append(part)
                                    print("✅ Detected part \(detectedMasks.count): \(whitePixels) pixels")
                                }
                            }
                        }
                        
                    } catch {
                        print("❌ Detection failed at grid (\(row), \(col)): \(error)")
                    }
                }
            }
            
            DispatchQueue.main.async {
                self.detectedParts = detectedMasks
                self.statusMessage = "Found \(detectedMasks.count) parts"
                print("✅ Detection complete: \(detectedMasks.count) unique parts found")
            }
        }
    }
    
    // SMOOTH MASK: Apply morphological operations for cleaner edges
    private func smoothMask(_ maskData: [UInt8], width: Int, height: Int) -> [UInt8] {
        var smoothed = maskData
        
        // Apply closing operation (dilation then erosion) to fill small holes
        smoothed = dilate(smoothed, width: width, height: height, iterations: 2)
        smoothed = erode(smoothed, width: width, height: height, iterations: 2)
        
        return smoothed
    }
    
    private func dilate(_ data: [UInt8], width: Int, height: Int, iterations: Int) -> [UInt8] {
        var result = data
        
        for _ in 0..<iterations {
            var temp = result
            for y in 1..<(height-1) {
                for x in 1..<(width-1) {
                    let idx = y * width + x
                    if result[idx] == 0 {
                        // Check 8-neighbors
                        let neighbors = [
                            result[idx - width - 1], result[idx - width], result[idx - width + 1],
                            result[idx - 1], result[idx + 1],
                            result[idx + width - 1], result[idx + width], result[idx + width + 1]
                        ]
                        if neighbors.contains(255) {
                            temp[idx] = 255
                        }
                    }
                }
            }
            result = temp
        }
        
        return result
    }
    
    private func erode(_ data: [UInt8], width: Int, height: Int, iterations: Int) -> [UInt8] {
        var result = data
        
        for _ in 0..<iterations {
            var temp = result
            for y in 1..<(height-1) {
                for x in 1..<(width-1) {
                    let idx = y * width + x
                    if result[idx] == 255 {
                        // Check 8-neighbors
                        let neighbors = [
                            result[idx - width - 1], result[idx - width], result[idx - width + 1],
                            result[idx - 1], result[idx + 1],
                            result[idx + width - 1], result[idx + width], result[idx + width + 1]
                        ]
                        if neighbors.contains(0) {
                            temp[idx] = 0
                        }
                    }
                }
            }
            result = temp
        }
        
        return result
    }
    
    private func isSimilarMask(_ mask1: [UInt8], _ mask2: [UInt8], threshold: Float) -> Bool {
        guard mask1.count == mask2.count else { return false }
        
        var matchingPixels = 0
        for i in 0..<mask1.count {
            if (mask1[i] > 127 && mask2[i] > 127) || (mask1[i] <= 127 && mask2[i] <= 127) {
                matchingPixels += 1
            }
        }
        
        let similarity = Float(matchingPixels) / Float(mask1.count)
        return similarity > threshold
    }
    
    func togglePartSelection(index: Int) {
        guard index < detectedParts.count else { return }
        
        detectedParts[index].isSelected.toggle()
        let isSelected = detectedParts[index].isSelected
        
        print("🎯 Part \(index + 1) \(isSelected ? "selected" : "deselected")")
        
        // Update outline color with smooth rendering
        if let pixelBuffer = frozenPixelBuffer {
            let color: UIColor = isSelected ? .green : .white
            if let newOutline = createSmoothOutline(
                from: detectedParts[index].maskData,
                width: detectedParts[index].width,
                height: detectedParts[index].height,
                pixelBuffer: pixelBuffer,
                color: color
            ) {
                detectedParts[index].outlineImage = newOutline
            }
        }
        
        let selectedCount = detectedParts.filter { $0.isSelected }.count
        DispatchQueue.main.async {
            self.statusMessage = "\(selectedCount) selected"
        }
    }
    
    func combineSelectedParts() {
        let selectedParts = detectedParts.filter { $0.isSelected }
        guard !selectedParts.isEmpty,
              let pixelBuffer = frozenPixelBuffer else {
            print("⚠️ No parts selected")
            return
        }
        
        DispatchQueue.main.async {
            self.statusMessage = "Combining..."
        }
        
        DispatchQueue.global(qos: .userInitiated).async {
            // Combine all selected masks
            let firstPart = selectedParts[0]
            var combinedMask = firstPart.maskData
            
            for i in 1..<selectedParts.count {
                let part = selectedParts[i]
                for j in 0..<combinedMask.count {
                    combinedMask[j] = max(combinedMask[j], part.maskData[j])
                }
            }
            
            // Apply final smoothing to combined mask
            combinedMask = self.smoothMask(combinedMask, width: firstPart.width, height: firstPart.height)
            
            print("✅ Combined \(selectedParts.count) parts with smoothing")
            
            self.createFinalSegmentedImage(combinedMask, width: firstPart.width, height: firstPart.height, pixelBuffer: pixelBuffer)
        }
    }
    
    // CREATE SMOOTH OUTLINE with anti-aliasing and better interpolation
    private func createSmoothOutline(from maskData: [UInt8], width: Int, height: Int, pixelBuffer: CVPixelBuffer, color: UIColor) -> UIImage? {
        // Create thicker outline (3 pixels) for better visibility
        var outlineData = [UInt8](repeating: 0, count: width * height * 4)
        
        // Edge detection with wider kernel
        for y in 2..<(height-2) {
            for x in 2..<(width-2) {
                let idx = y * width + x
                let current = maskData[idx]
                
                if current == 255 {
                    // Check wider neighborhood for smoother edges
                    var isEdge = false
                    for dy in -2...2 {
                        for dx in -2...2 {
                            if maskData[(y + dy) * width + (x + dx)] == 0 {
                                isEdge = true
                                break
                            }
                        }
                        if isEdge { break }
                    }
                    
                    if isEdge {
                        let outlineIdx = idx * 4
                        var r: CGFloat = 0, g: CGFloat = 0, b: CGFloat = 0, a: CGFloat = 0
                        color.getRed(&r, green: &g, blue: &b, alpha: &a)
                        
                        outlineData[outlineIdx] = UInt8(r * 255)
                        outlineData[outlineIdx + 1] = UInt8(g * 255)
                        outlineData[outlineIdx + 2] = UInt8(b * 255)
                        outlineData[outlineIdx + 3] = 255
                    }
                }
            }
        }
        
        // Create CGImage from outline
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
            shouldInterpolate: true, // Enable interpolation for smoother result
            intent: .defaultIntent
        ) else {
            return nil
        }
        
        // Scale with bicubic interpolation for smoother edges
        let ciOutline = CIImage(cgImage: cgImage)
        
        let originalImage = CIImage(cvPixelBuffer: pixelBuffer)
        let scaleX = originalImage.extent.width / CGFloat(width)
        let scaleY = originalImage.extent.height / CGFloat(height)
        
        // Use lanczos scale transform for highest quality upscaling
        let scaledOutline = ciOutline
            .transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))
            .samplingNearest() // Use lanczos for best quality
        
        // Apply slight gaussian blur for anti-aliasing
        let blurred = scaledOutline.clampedToExtent().applyingGaussianBlur(sigma: 0.5).cropped(to: scaledOutline.extent)
        
        let context = CIContext(options: [.useSoftwareRenderer: false])
        if let scaledCGImage = context.createCGImage(blurred, from: blurred.extent) {
            return UIImage(cgImage: scaledCGImage, scale: 1.0, orientation: .right)
        }
        
        return nil
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
            shouldInterpolate: true, // Enable interpolation
            intent: .defaultIntent
        ) else {
            return
        }
        
        let maskImage = CIImage(cgImage: cgMask)
        let originalImage = CIImage(cvPixelBuffer: pixelBuffer)
        
        let scaleX = originalImage.extent.width / CGFloat(width)
        let scaleY = originalImage.extent.height / CGFloat(height)
        
        // Use bicubic for smooth scaling
        let scaledMask = maskImage
            .transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))
            .samplingNearest()
        
        guard let blendFilter = CIFilter(name: "CIBlendWithMask") else { return }
        
        let transparent = CIImage(color: .clear).cropped(to: originalImage.extent)
        
        // White mask shows furniture, black shows transparent
        blendFilter.setValue(originalImage, forKey: kCIInputImageKey)
        blendFilter.setValue(transparent, forKey: kCIInputBackgroundImageKey)
        blendFilter.setValue(scaledMask, forKey: kCIInputMaskImageKey)
        
        guard let result = blendFilter.outputImage else { return }
        
        let context = CIContext(options: [.useSoftwareRenderer: false])
        let rgbColorSpace = CGColorSpaceCreateDeviceRGB()
        
        if let cgImage = context.createCGImage(result, from: result.extent, format: CIFormat.RGBA8, colorSpace: rgbColorSpace) {
            let uiImage = UIImage(cgImage: cgImage, scale: 1.0, orientation: .right)
            
            print("🖼️ Created final furniture with smooth edges")
            
            DispatchQueue.main.async {
                self.finalSegmentedImage = uiImage
                self.statusMessage = "Ready!"
            }
        }
    }
    
    func unfreezeFrame() {
        frozenFrame = nil
        frozenPixelBuffer = nil
        detectedParts.removeAll()
        DispatchQueue.main.async {
            self.statusMessage = "Ready"
        }
    }
    
    func resetAll() {
        frozenFrame = nil
        frozenPixelBuffer = nil
        detectedParts.removeAll()
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
