import SwiftUI
import AVFoundation
import CoreML
import Vision
import CoreImage
import Photos

struct SegmentExamine: View {
    @Binding var capturedImage: UIImage?
    @Binding var isShowingCamera: Bool
    @StateObject private var camera = HybridSAMCameraModel()
    
    @State private var scaleMultiplier: CGFloat = 0.5
    @State private var dragOffset: CGSize = .zero
    @State private var accumulatedOffset: CGSize = .zero
    @State private var isInitialAppearance = true
    @State private var scannerRotation: Double = 0
    @State private var showingSaveSuccess = false
    
    var body: some View {
        ZStack {
            Color.clear.ignoresSafeArea()
            
            if !camera.isExamining && camera.segmentedImage == nil && !isInitialAppearance {
                ScanningReticleExamine(rotation: $scannerRotation)
                    .onAppear {
                        withAnimation(.linear(duration: 2.0).repeatForever(autoreverses: false)) {
                            scannerRotation = 360
                        }
                    }
            }
            
            if let segmented = camera.segmentedImage {
                Image(uiImage: segmented)
                    .resizable()
                    .scaledToFill()
                    .scaleEffect(scaleMultiplier)
                    .offset(x: dragOffset.width + accumulatedOffset.width,
                           y: dragOffset.height + accumulatedOffset.height)
                    .position(x: UIScreen.main.bounds.width / 2,
                             y: UIScreen.main.bounds.height / 2)
                    .gesture(
                        DragGesture()
                            .onChanged { value in dragOffset = value.translation }
                            .onEnded { value in
                                accumulatedOffset.width += value.translation.width
                                accumulatedOffset.height += value.translation.height
                                dragOffset = .zero
                            }
                    )
                    .ignoresSafeArea()
                    .opacity(camera.furnitureOpacity)
                    .animation(.easeOut(duration: 0.3), value: camera.furnitureOpacity)
            }
            
            if showingSaveSuccess {
                VStack {
                    Spacer()
                    HStack(spacing: 12) {
                        Image(systemName: "checkmark.circle.fill").font(.title2)
                        Text("Furniture saved!").font(.headline)
                    }
                    .foregroundColor(.white)
                    .padding(.horizontal, 24)
                    .padding(.vertical, 16)
                    .background(Capsule().fill(Color.green.opacity(0.95)).shadow(radius: 10))
                    .padding(.bottom, 150)
                }
                .transition(.move(edge: .bottom).combined(with: .opacity))
            }
            
            VStack {
                HStack {
                    HStack(spacing: 6) {
                        Image(systemName: "minus.magnifyingglass").foregroundColor(.white).font(.system(size: 14))
                        Slider(value: $scaleMultiplier, in: 0.3...1.0).frame(width: 150).accentColor(.white)
                        Image(systemName: "plus.magnifyingglass").foregroundColor(.white).font(.system(size: 14))
                    }
                    .padding(.horizontal, 12).padding(.vertical, 8)
                    .background(Capsule().fill(Color.black.opacity(0.7)))
                    .padding(.leading, 16)
                    
                    Spacer()
                    
                    Button(action: { isShowingCamera = false }) {
                        Image(systemName: "xmark.circle.fill")
                            .font(.title)
                            .foregroundColor(.white)
                            .background(Circle().fill(Color.black.opacity(0.5)))
                    }
                    .padding(.trailing, 16)
                }
                .padding(.top, 60)
                
                Spacer()
                
                VStack(spacing: 12) {
                    if !camera.isExamining {
                        if camera.segmentedImage == nil {
                            HStack(spacing: 4) {
                                ProgressView().progressViewStyle(CircularProgressViewStyle(tint: .white)).scaleEffect(0.7)
                                Text("Detecting furniture...").font(.caption)
                            }
                            .padding(8).background(Capsule().fill(Color.black.opacity(0.5))).foregroundColor(.white)
                        }
                    }
                    
                    HStack(spacing: 16) {
                        if camera.isExamining {
                            Button(action: { saveFurniture() }) {
                                HStack(spacing: 8) {
                                    Image(systemName: "square.and.arrow.down.fill").font(.title3)
                                    Text("Save").font(.headline)
                                }
                                .foregroundColor(.white).padding(.horizontal, 24).padding(.vertical, 12)
                                .background(Capsule().fill(Color.green.opacity(0.9)))
                            }
                            
                            Button(action: { withAnimation { camera.finishExamining() } }) {
                                HStack(spacing: 8) {
                                    Image(systemName: "arrow.counterclockwise").font(.title3)
                                    Text("Back").font(.headline)
                                }
                                .foregroundColor(.white).padding(.horizontal, 24).padding(.vertical, 12)
                                .background(Capsule().fill(Color.gray.opacity(0.9)))
                            }
                        } else if camera.segmentedImage != nil {
                            Button(action: {
                                withAnimation { camera.startExamining() }
                                UIImpactFeedbackGenerator(style: .medium).impactOccurred()
                            }) {
                                HStack(spacing: 8) {
                                    Image(systemName: "viewfinder.circle.fill").font(.title3)
                                    Text("Examine").font(.headline)
                                }
                                .foregroundColor(.white).padding(.horizontal, 32).padding(.vertical, 12)
                                .background(Capsule().fill(Color.green.opacity(0.9)))
                            }
                        }
                    }
                }
                .padding(.bottom, 50).padding(.horizontal)
            }
        }
        .onAppear {
            AVCaptureDevice.requestAccess(for: .video) { granted in
                if granted {
                    camera.startSession()
                    DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) {
                        withAnimation { isInitialAppearance = false }
                    }
                }
            }
        }
        .onDisappear { camera.stopSession() }
    }
    
    private func saveFurniture() {
        guard let image = camera.segmentedImage else { return }
        capturedImage = image
        
        PHPhotoLibrary.requestAuthorization(for: .addOnly) { status in
            DispatchQueue.main.async {
                if status == .authorized || status == .limited {
                    PHPhotoLibrary.shared().performChanges({
                        PHAssetChangeRequest.creationRequestForAsset(from: image)
                    }) { success, _ in
                        DispatchQueue.main.async {
                            if success {
                                withAnimation { showingSaveSuccess = true }
                                UINotificationFeedbackGenerator().notificationOccurred(.success)
                                DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) {
                                    withAnimation { showingSaveSuccess = false }
                                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                                        isShowingCamera = false
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

struct ScanningReticleExamine: View {
    @Binding var rotation: Double
    var body: some View {
        ZStack {
            Circle().stroke(Color.white.opacity(0.3), lineWidth: 2).frame(width: 120, height: 120)
            Rectangle()
                .fill(LinearGradient(gradient: Gradient(colors: [.clear, .white.opacity(0.8), .clear]), startPoint: .top, endPoint: .bottom))
                .frame(width: 2, height: 60).offset(y: -30).rotationEffect(.degrees(rotation))
            Circle().fill(Color.white.opacity(0.5)).frame(width: 8, height: 8)
        }
    }
}

// MARK: - Hybrid U2-Net + MobileSAM Camera Model
class HybridSAMCameraModel: NSObject, ObservableObject {
    @Published var segmentedImage: UIImage?
    @Published var furnitureOpacity: Double = 0.0
    @Published var isExamining: Bool = false
    
    let session = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let videoQueue = DispatchQueue(label: "hybridSAMVideoQueue", qos: .userInitiated)
    
    private var u2netModel: VNCoreMLModel?
    private var mobileSAMEncoder: MLModel?
    private var mobileSAMDecoder: MLModel?
    
    private let context = CIContext()
    private var lastProcessTime = Date()
    private let processInterval: TimeInterval = 0.2
    private var isProcessing = false
    private var u2netMask: CVPixelBuffer?
    private var lockedMask: CVPixelBuffer? = nil
    private var shouldStartExaminingOnNextFrame = false
    private var currentPixelBuffer: CVPixelBuffer?
    private var currentImageEmbeddings: MLMultiArray?
    
    override init() {
        super.init()
        print("🎯 [HybridSAMCameraModel] Initializing...")
        loadModels()
        setupCamera()
    }
    
    func startExamining() {
        print("🔍 Starting examination mode")
        shouldStartExaminingOnNextFrame = true
    }
    
    func finishExamining() {
        print("✅ Ending examination mode")
        DispatchQueue.main.async {
            self.isExamining = false
        }
        lockedMask = nil
        shouldStartExaminingOnNextFrame = false
        segmentedImage = nil
        furnitureOpacity = 0.0
    }
    
    private func loadModels() {
        // Load U2-Net
        let modelNames = ["u2netp", "U2Net", "u2net"]
        for name in modelNames {
            if let modelURL = Bundle.main.url(forResource: name, withExtension: "mlmodelc") {
                do {
                    let model = try MLModel(contentsOf: modelURL)
                    u2netModel = try VNCoreMLModel(for: model)
                    print("✅ [HybridSAM] U2-Net loaded: \(name)")
                    break
                } catch {
                    print("⚠️ Failed to load \(name): \(error)")
                }
            }
        }
        
        // Load MobileSAM Encoder
        if let encoderURL = Bundle.main.url(forResource: "MobileSAMImageEncoder", withExtension: "mlmodelc") {
            do {
                let config = MLModelConfiguration()
                config.computeUnits = .cpuAndGPU
                mobileSAMEncoder = try MLModel(contentsOf: encoderURL, configuration: config)
                print("✅ [HybridSAM] MobileSAM Encoder loaded")
            } catch {
                print("❌ Failed to load MobileSAM Encoder: \(error)")
            }
        }
        
        // Load MobileSAM Decoder
        if let decoderURL = Bundle.main.url(forResource: "MobileSAMMaskDecoder", withExtension: "mlmodelc") {
            do {
                let config = MLModelConfiguration()
                config.computeUnits = .cpuOnly  // CPU for stability
                mobileSAMDecoder = try MLModel(contentsOf: decoderURL, configuration: config)
                print("✅ [HybridSAM] MobileSAM Decoder loaded (CPU)")
            } catch {
                print("❌ Failed to load MobileSAM Decoder: \(error)")
            }
        }
    }
    
    private func setupCamera() {
        session.beginConfiguration()
        session.sessionPreset = .hd1280x720
        
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
              let input = try? AVCaptureDeviceInput(device: device) else {
            print("❌ Camera setup failed")
            return
        }
        
        if session.canAddInput(input) {
            session.addInput(input)
        }
        
        videoOutput.setSampleBufferDelegate(self, queue: videoQueue)
        videoOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        videoOutput.alwaysDiscardsLateVideoFrames = true
        
        if session.canAddOutput(videoOutput) {
            session.addOutput(videoOutput)
            videoOutput.connection(with: .video)?.videoRotationAngle = 90
        }
        
        session.commitConfiguration()
        print("✅ [HybridSAM] Camera configured")
    }
    
    func startSession() {
        if !session.isRunning {
            DispatchQueue.global(qos: .background).async {
                self.session.startRunning()
                print("✅ [HybridSAM] Camera session started")
            }
        }
    }
    
    func stopSession() {
        if session.isRunning {
            session.stopRunning()
            print("🛑 [HybridSAM] Camera session stopped")
        }
    }
    
    private func processFrame(pixelBuffer: CVPixelBuffer) {
        let now = Date()
        guard now.timeIntervalSince(lastProcessTime) >= processInterval else { return }
        guard !isProcessing else { return }
        
        isProcessing = true
        lastProcessTime = now
        
        currentPixelBuffer = pixelBuffer
        
        // Step 1: Run U2-Net to find furniture
        runU2Net(pixelBuffer: pixelBuffer)
        
        guard let u2netMask = u2netMask else {
            isProcessing = false
            return
        }
        
        var maskToUse = u2netMask
        
        if shouldStartExaminingOnNextFrame {
            print("📸 Locking current frame...")
            lockedMask = copyPixelBuffer(u2netMask)
            shouldStartExaminingOnNextFrame = false
            DispatchQueue.main.async { self.isExamining = true }
        } else if let locked = lockedMask {
            maskToUse = locked
        }
        
        // Step 2: Get center point from U2-Net mask
        guard let center = getFurnitureCenter(mask: maskToUse) else {
            print("⚠️ U2-Net didn't detect furniture")
            isProcessing = false
            return
        }
        
        print("\n🎯 === HYBRID PIPELINE ===")
        print("📍 U2-Net found furniture center at: (\(center.x), \(center.y))")
        
        // Step 3: Run MobileSAM Encoder for embeddings
        if mobileSAMEncoder != nil {
            runSAMEncoder(pixelBuffer: pixelBuffer)
        }
        
        // Step 4: Run MobileSAM Decoder with center point
        processWithMobileSAM(pixelBuffer: pixelBuffer, centerPoint: center)
        
        isProcessing = false
    }
    
    private func getFurnitureCenter(mask: CVPixelBuffer) -> CGPoint? {
        CVPixelBufferLockBaseAddress(mask, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(mask, .readOnly) }
        
        let width = CVPixelBufferGetWidth(mask)
        let height = CVPixelBufferGetHeight(mask)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(mask)
        
        guard let baseAddress = CVPixelBufferGetBaseAddress(mask) else { return nil }
        let maskPtr = baseAddress.assumingMemoryBound(to: UInt8.self)
        
        var minX = width, maxX = 0, minY = height, maxY = 0
        var hasPixels = false
        
        for y in 0..<height {
            let rowPtr = maskPtr.advanced(by: y * bytesPerRow)
            for x in 0..<width {
                if rowPtr[x] > 100 {
                    hasPixels = true
                    minX = min(minX, x)
                    maxX = max(maxX, x)
                    minY = min(minY, y)
                    maxY = max(maxY, y)
                }
            }
        }
        
        guard hasPixels else { return nil }
        
        let centerX = CGFloat(minX + maxX) / 2.0
        let centerY = CGFloat(minY + maxY) / 2.0
        
        return CGPoint(x: centerX, y: centerY)
    }
    
    private func runSAMEncoder(pixelBuffer: CVPixelBuffer) {
        guard let encoder = mobileSAMEncoder else { return }
        
        do {
            guard let resized = resizePixelBuffer(pixelBuffer, width: 1024, height: 1024) else { return }
            guard let imageArray = pixelBufferToMLMultiArray(resized) else { return }
            
            let inputDict: [String: Any] = ["image": imageArray]
            let inputProvider = try MLDictionaryFeatureProvider(dictionary: inputDict)
            let output = try encoder.prediction(from: inputProvider)
            
            guard let embeddings = output.featureValue(for: "image_embeddings")?.multiArrayValue else { return }
            
            currentImageEmbeddings = embeddings
            print("✅ SAM Encoder: embeddings shape \(embeddings.shape)")
        } catch {
            print("❌ SAM Encoder failed: \(error)")
        }
    }
    
    private func processWithMobileSAM(pixelBuffer: CVPixelBuffer, centerPoint: CGPoint) {
        guard let embeddings = currentImageEmbeddings,
              let decoder = mobileSAMDecoder else {
            print("⚠️ MobileSAM not ready, falling back to U2-Net")
            fallbackToU2Net(pixelBuffer: pixelBuffer)
            return
        }
        
        // Convert to normalized coordinates (0-1)
        let originalWidth = CVPixelBufferGetWidth(pixelBuffer)
        let originalHeight = CVPixelBufferGetHeight(pixelBuffer)
        
        let normalizedX = Float(centerPoint.x) / Float(originalWidth)
        let normalizedY = Float(centerPoint.y) / Float(originalHeight)
        
        print("📍 Point prompt (normalized): (\(String(format: "%.3f", normalizedX)), \(String(format: "%.3f", normalizedY)))")
        
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                // Create point_coords [1, 1, 2] - Float32
                let pointCoords = try MLMultiArray(shape: [1, 1, 2], dataType: .float32)
                pointCoords[[0, 0, 0] as [NSNumber]] = NSNumber(value: normalizedX)
                pointCoords[[0, 0, 1] as [NSNumber]] = NSNumber(value: normalizedY)
                
                // Create point_labels [1, 1] - Float32
                let pointLabels = try MLMultiArray(shape: [1, 1], dataType: .float32)
                pointLabels[[0, 0] as [NSNumber]] = NSNumber(value: 1)  // 1 = positive point
                
                let inputDict: [String: Any] = [
                    "image_embeddings": embeddings,
                    "point_coords": pointCoords,
                    "point_labels": pointLabels
                ]
                
                print("🔄 Running MobileSAM Decoder...")
                let inputProvider = try MLDictionaryFeatureProvider(dictionary: inputDict)
                let output = try decoder.prediction(from: inputProvider)
                
                guard let masks = output.featureValue(for: "masks")?.multiArrayValue,
                      let iouPredictions = output.featureValue(for: "iou_predictions")?.multiArrayValue else {
                    print("❌ No masks from MobileSAM")
                    self.fallbackToU2Net(pixelBuffer: pixelBuffer)
                    return
                }
                
                print("✅ MobileSAM generated masks")
                
                self.processMobileSAMMasks(
                    masks: masks,
                    iouPredictions: iouPredictions,
                    originalPixelBuffer: pixelBuffer,
                    originalSize: CGSize(width: originalWidth, height: originalHeight)
                )
                
            } catch {
                print("❌ MobileSAM error: \(error)")
                self.fallbackToU2Net(pixelBuffer: pixelBuffer)
            }
        }
    }
    
    private func processMobileSAMMasks(masks: MLMultiArray, iouPredictions: MLMultiArray, originalPixelBuffer: CVPixelBuffer, originalSize: CGSize) {
        print("\n📊 === MOBILESAM RESULTS ===")
        
        // Get IOU score
        let iouScore = iouPredictions[[0, 0] as [NSNumber]].floatValue
        print("📊 IOU Score: \(String(format: "%.3f", iouScore))")
        
        // Extract mask (256x256) and apply threshold
        let maskWidth = 256
        let maskHeight = 256
        var maskData = [UInt8](repeating: 0, count: maskWidth * maskHeight)
        
        let threshold: Float = 0.0  // MobileSAM: logits > 0 = object
        var positivePixels = 0
        
        for y in 0..<maskHeight {
            for x in 0..<maskWidth {
                let logit = masks[[0, 0, y, x] as [NSNumber]].floatValue
                if logit > threshold {
                    maskData[y * maskWidth + x] = 255
                    positivePixels += 1
                } else {
                    maskData[y * maskWidth + x] = 0
                }
            }
        }
        
        let coverage = Float(positivePixels) / Float(maskWidth * maskHeight) * 100
        print("📊 Mask coverage: \(positivePixels) pixels (\(String(format: "%.1f", coverage))%)")
        
        if positivePixels == 0 {
            print("⚠️ Empty mask - falling back to U2-Net")
            fallbackToU2Net(pixelBuffer: originalPixelBuffer)
            return
        }
        
        // Convert to CIImage and scale to original size
        let data = Data(maskData)
        guard let provider = CGDataProvider(data: data as CFData),
              let cgImage = CGImage(
                width: maskWidth,
                height: maskHeight,
                bitsPerComponent: 8,
                bitsPerPixel: 8,
                bytesPerRow: maskWidth,
                space: CGColorSpaceCreateDeviceGray(),
                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
                provider: provider,
                decode: nil,
                shouldInterpolate: false,
                intent: .defaultIntent
              ) else {
            print("❌ Failed to create mask image")
            return
        }
        
        let maskImage = CIImage(cgImage: cgImage)
        let ciImage = CIImage(cvPixelBuffer: originalPixelBuffer)
        
        // Apply mask to original image
        applyMobileSAMMask(original: ciImage, mask: maskImage, originalSize: originalSize)
    }
    
    private func applyMobileSAMMask(original: CIImage, mask: CIImage, originalSize: CGSize) {
        // Scale mask from 256x256 to original size
        let scaleX = original.extent.width / mask.extent.width
        let scaleY = original.extent.height / mask.extent.height
        let scaledMask = mask.transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))
        
        var finalMask = scaledMask
        
        // Apply blur for smooth edges
        if let blurFilter = CIFilter(name: "CIGaussianBlur") {
            blurFilter.setValue(scaledMask, forKey: kCIInputImageKey)
            blurFilter.setValue(2.0, forKey: kCIInputRadiusKey)
            if let blurred = blurFilter.outputImage {
                finalMask = blurred
            }
        }
        
        finalMask = finalMask.cropped(to: original.extent)
        
        // Blend with transparent background
        guard let blendFilter = CIFilter(name: "CIBlendWithMask") else { return }
        let transparent = CIImage(color: CIColor(red: 0, green: 0, blue: 0, alpha: 0)).cropped(to: original.extent)
        
        blendFilter.setValue(original, forKey: kCIInputImageKey)
        blendFilter.setValue(transparent, forKey: kCIInputBackgroundImageKey)
        blendFilter.setValue(finalMask, forKey: kCIInputMaskImageKey)
        
        guard let result = blendFilter.outputImage else { return }
        
        if let cgImage = context.createCGImage(result, from: result.extent) {
            let uiImage = UIImage(cgImage: cgImage, scale: 1.0, orientation: .up)
            
            DispatchQueue.main.async {
                self.segmentedImage = uiImage
                withAnimation(.easeIn(duration: 0.2)) {
                    self.furnitureOpacity = 1.0
                }
            }
            
            print("✅ MobileSAM segmentation applied")
        }
    }
    
    private func fallbackToU2Net(pixelBuffer: CVPixelBuffer) {
        print("⚠️ Using U2-Net fallback")
        guard let u2netMask = u2netMask else { return }
        
        let maskImage = CIImage(cvPixelBuffer: u2netMask)
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        
        let scaleX = ciImage.extent.width / maskImage.extent.width
        let scaleY = ciImage.extent.height / maskImage.extent.height
        let scaledMask = maskImage.transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))
        
        var finalMask = scaledMask
        
        if let blurFilter = CIFilter(name: "CIGaussianBlur") {
            blurFilter.setValue(scaledMask, forKey: kCIInputImageKey)
            blurFilter.setValue(2.0, forKey: kCIInputRadiusKey)
            if let blurred = blurFilter.outputImage {
                finalMask = blurred
            }
        }
        
        finalMask = finalMask.cropped(to: ciImage.extent)
        
        guard let blendFilter = CIFilter(name: "CIBlendWithMask") else { return }
        let transparent = CIImage(color: CIColor(red: 0, green: 0, blue: 0, alpha: 0)).cropped(to: ciImage.extent)
        
        blendFilter.setValue(ciImage, forKey: kCIInputImageKey)
        blendFilter.setValue(transparent, forKey: kCIInputBackgroundImageKey)
        blendFilter.setValue(finalMask, forKey: kCIInputMaskImageKey)
        
        guard let result = blendFilter.outputImage else { return }
        
        if let cgImage = context.createCGImage(result, from: result.extent) {
            let uiImage = UIImage(cgImage: cgImage, scale: 1.0, orientation: .up)
            
            DispatchQueue.main.async {
                self.segmentedImage = uiImage
                withAnimation(.easeIn(duration: 0.2)) {
                    self.furnitureOpacity = 1.0
                }
            }
        }
    }
    
    // CRITICAL: ImageNet normalization for MobileSAM
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
        
        // ImageNet normalization values
        let mean: [Float] = [0.485, 0.456, 0.406]
        let std: [Float] = [0.229, 0.224, 0.225]
        
        let arrayPointer = array.dataPointer.assumingMemoryBound(to: Float.self)
        
        for y in 0..<height {
            for x in 0..<width {
                let pixelIndex = y * bytesPerRow + x * 4
                
                let b = Float(buffer[pixelIndex]) / 255.0
                let g = Float(buffer[pixelIndex + 1]) / 255.0
                let r = Float(buffer[pixelIndex + 2]) / 255.0
                
                // [batch, channel, y, x] layout
                let rIndex = 0 * height * width + y * width + x
                let gIndex = 1 * height * width + y * width + x
                let bIndex = 2 * height * width + y * width + x
                
                // Apply ImageNet normalization
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
        
        context.render(scaledImage, to: outputBuffer)
        
        return outputBuffer
    }
    
    private func copyPixelBuffer(_ source: CVPixelBuffer) -> CVPixelBuffer? {
        let width = CVPixelBufferGetWidth(source)
        let height = CVPixelBufferGetHeight(source)
        let format = CVPixelBufferGetPixelFormatType(source)
        
        var copy: CVPixelBuffer?
        let attrs = [kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue!] as CFDictionary
        guard CVPixelBufferCreate(kCFAllocatorDefault, width, height, format, attrs, &copy) == kCVReturnSuccess,
              let destination = copy else { return nil }
        
        CVPixelBufferLockBaseAddress(source, .readOnly)
        CVPixelBufferLockBaseAddress(destination, [])
        defer {
            CVPixelBufferUnlockBaseAddress(source, .readOnly)
            CVPixelBufferUnlockBaseAddress(destination, [])
        }
        
        if let sourceData = CVPixelBufferGetBaseAddress(source),
           let destData = CVPixelBufferGetBaseAddress(destination) {
            let bytesPerRow = CVPixelBufferGetBytesPerRow(source)
            memcpy(destData, sourceData, bytesPerRow * height)
        }
        
        return destination
    }
    
    private func runU2Net(pixelBuffer: CVPixelBuffer) {
        guard let model = u2netModel else { return }
        
        let request = VNCoreMLRequest(model: model) { [weak self] request, error in
            if let error = error {
                print("❌ U2-Net error: \(error)")
                return
            }
            
            if let results = request.results as? [VNPixelBufferObservation],
               let maskBuffer = results.first?.pixelBuffer {
                self?.u2netMask = maskBuffer
            }
        }
        
        request.imageCropAndScaleOption = .scaleFill
        
        do {
            try VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:]).perform([request])
        } catch {
            print("❌ U2-Net handler error: \(error)")
        }
    }
}

extension HybridSAMCameraModel: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        processFrame(pixelBuffer: pixelBuffer)
    }
}
