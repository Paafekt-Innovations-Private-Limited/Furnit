import SwiftUI
import AVFoundation
import CoreML
import Vision
import CoreImage
import Photos

// SEGMENT EXAMINE - FASTSAM-X PRIMARY
// FastSAM-X runs as PRIMARY segmentation (better at complete furniture objects)
// U2-Net provides SUPPORT (stabilization, gap-filling, refinement)
// User can tap missing parts → MobileSAM decoder for additional refinement
// Result: ONE unified furniture segmentation with better handling of multi-part items

struct SegmentExamine: View {
    @Binding var capturedImage: UIImage?
    @Binding var isShowingCamera: Bool
    @StateObject private var camera = SegmentExamineModel()
    
    @State private var scaleMultiplier: CGFloat = 0.5
    @State private var dragOffset: CGSize = .zero
    @State private var accumulatedOffset: CGSize = .zero
    @State private var showingSaveSuccess = false
    
    var body: some View {
        ZStack {
            Color.clear.ignoresSafeArea()  // Transparent - shows 3D room underneath
            
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
            
            // Real Progress Bar
            if camera.isProcessingStarted {
                VStack {
                    Spacer()
                    VStack(spacing: 12) {
                        Text(camera.progressMessage)
                            .font(.headline)
                            .foregroundColor(.white)
                        
                        ZStack(alignment: .leading) {
                            // Background
                            RoundedRectangle(cornerRadius: 8)
                                .fill(Color.white.opacity(0.3))
                                .frame(width: 280, height: 8)
                            
                            // Progress
                            RoundedRectangle(cornerRadius: 8)
                                .fill(Color.green)
                                .frame(width: 280 * camera.processingProgress, height: 8)
                                .animation(.easeInOut(duration: 0.2), value: camera.processingProgress)
                        }
                        
                        Text("\(Int(camera.processingProgress * 100))%")
                            .font(.caption)
                            .foregroundColor(.white.opacity(0.8))
                    }
                    .padding(.vertical, 20)
                    .padding(.horizontal, 30)
                    .background(
                        RoundedRectangle(cornerRadius: 16)
                            .fill(Color.black.opacity(0.85))
                            .shadow(radius: 10)
                    )
                    .padding(.bottom, 200)
                }
                .transition(.move(edge: .bottom).combined(with: .opacity))
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
                    if camera.segmentedImage != nil {
                        HStack(spacing: 6) {
                            Image(systemName: "minus.magnifyingglass").foregroundColor(.white).font(.system(size: 14))
                            Slider(value: $scaleMultiplier, in: 0.3...1.0).frame(width: 150).accentColor(.white)
                            Image(systemName: "plus.magnifyingglass").foregroundColor(.white).font(.system(size: 14))
                        }
                        .padding(.horizontal, 12).padding(.vertical, 8)
                        .background(Capsule().fill(Color.black.opacity(0.7)))
                        .padding(.leading, 16)
                    }
                    
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
                
                HStack(spacing: 16) {
                    if camera.segmentedImage != nil {
                        Button(action: { saveFurniture() }) {
                            HStack(spacing: 8) {
                                Image(systemName: "square.and.arrow.down.fill").font(.title3)
                                Text("Save").font(.headline)
                            }
                            .foregroundColor(.white).padding(.horizontal, 24).padding(.vertical, 12)
                            .background(Capsule().fill(Color.green.opacity(0.9)))
                        }
                        
                        Button(action: {
                            camera.segmentedImage = nil
                            camera.furnitureOpacity = 0.0
                            camera.triggerCapture()
                        }) {
                            HStack(spacing: 8) {
                                Image(systemName: "arrow.counterclockwise").font(.title3)
                                Text("Retry").font(.headline)
                            }
                            .foregroundColor(.white).padding(.horizontal, 24).padding(.vertical, 12)
                            .background(Capsule().fill(Color.orange.opacity(0.9)))
                        }
                    }
                    
                    Button(action: { isShowingCamera = false }) {
                        HStack(spacing: 8) {
                            Image(systemName: "xmark.circle.fill").font(.title3)
                            Text("Close").font(.headline)
                        }
                        .foregroundColor(.white).padding(.horizontal, 24).padding(.vertical, 12)
                        .background(Capsule().fill(Color.gray.opacity(0.9)))
                    }
                }
                .padding(.bottom, 50).padding(.horizontal)
            }
        }
        .onAppear {
            AVCaptureDevice.requestAccess(for: .video) { granted in
                if granted {
                    camera.startSession()
                    // Wait a moment for camera to initialize, then trigger capture
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                        camera.triggerCapture()
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

// MARK: - Segment Examine Model - FastSAM-X PRIMARY
class SegmentExamineModel: NSObject, ObservableObject {
    @Published var segmentedImage: UIImage?
    @Published var furnitureOpacity: Double = 0.0
    @Published var processingProgress: Double = 0.0
    @Published var isProcessingStarted: Bool = false
    @Published var progressMessage: String = ""
    @Published var isReadyToCapture: Bool = false
    
    let session = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let videoQueue = DispatchQueue(label: "segmentExamineQueue", qos: .userInitiated)
    
    // MODEL HIERARCHY: FastSAM-X (primary) + U2-Net (support) + Canny (edge refinement)
    private var fastsamModel: VNCoreMLModel?      // Primary: complete furniture detection
    private var u2netModel: VNCoreMLModel?        // Support: saliency refinement
    
    private let context = CIContext()
    
    private var lastProcessTime = Date()
    private let processInterval: TimeInterval = 0.2
    private var isProcessing = false
    private var shouldCapture = false  // Only process when user clicks button
    private var capturePixelBuffer: CVPixelBuffer?  // Store frame for processing
    
    // Detection buffers
    private var fastsamMask: CVPixelBuffer?       // Primary detection
    private var u2netMask: CVPixelBuffer?         // Support detection
    private var cannyEdges: CVPixelBuffer?        // Edge detection
    
    // FastSAM temporal smoothing
    private var previousFastsamMasks: [CVPixelBuffer] = []
    private let temporalWindowSize = 3
    
    override init() {
        super.init()
        loadFastSAMModel()
        loadU2NetModel()
        setupCamera()
    }
    
    func triggerCapture() {
        guard !isProcessing else {
            print("⚠️ Already processing, ignoring trigger")
            return
        }
        
        print("📸 Capture triggered!")
        shouldCapture = true
    }
    
    // LOAD FASTSAM-X MODEL (PRIMARY)
    private func loadFastSAMModel() {
        print("🔍 Searching for FastSAM-X model...")
        // User confirmed the correct name is "FastSAM-x"
        let modelNames = ["FastSAM-x", "fastsam-x", "FastSAMx", "FastSAM_x", "fastsam_x", "FastSAMX", "fastsam", "FastSAM-X", "FastSAM"]
        
        for name in modelNames {
            for ext in ["mlmodelc", "mlpackage"] {
                if let modelURL = Bundle.main.url(forResource: name, withExtension: ext) {
                    print("📁 Found model file: \(name).\(ext)")
                    do {
                        let model = try MLModel(contentsOf: modelURL)
                        fastsamModel = try VNCoreMLModel(for: model)
                        print("✅ FastSAM-X loaded successfully: \(name) [PRIMARY]")
                        return
                    } catch {
                        print("❌ Failed to load \(name).\(ext): \(error.localizedDescription)")
                    }
                } else {
                    print("⚪ Not found: \(name).\(ext)")
                }
            }
        }
        print("❌ CRITICAL: No FastSAM-X model found in bundle!")
        print("⚠️ Will fall back to U2-Net only")
    }
    
    // LOAD U2-NET MODEL (SUPPORT)
    private func loadU2NetModel() {
        let modelNames = ["u2netp", "U2Net", "u2net", "U2NetP", "U2NET"]
        
        for name in modelNames {
            for ext in ["mlmodelc", "mlpackage"] {
                if let modelURL = Bundle.main.url(forResource: name, withExtension: ext) {
                    do {
                        let model = try MLModel(contentsOf: modelURL)
                        u2netModel = try VNCoreMLModel(for: model)
                        print("✅ U2-Net loaded: \(name) [SUPPORT]")
                        return
                    } catch {
                        print("⚠️ Failed to load \(name).\(ext): \(error)")
                    }
                }
            }
        }
        print("⚠️ No U2-Net model loaded - will run FastSAM-X only")
    }
    
    private func setupCamera() {
        session.beginConfiguration()
        session.sessionPreset = .hd1280x720
        
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else {
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
                
                if let connection = videoOutput.connection(with: .video) {
                    connection.videoRotationAngle = 90
                    if connection.isVideoMirroringSupported {
                        connection.isVideoMirrored = false
                    }
                }
            }
            
            session.commitConfiguration()
            print("✅ Camera configured [FastSAM-X PRIMARY]")
        } catch {
            print("❌ Camera setup failed")
        }
    }
    
    func startSession() {
        if !session.isRunning {
            DispatchQueue.global(qos: .background).async {
                self.session.startRunning()
                print("✅ Camera session started [FastSAM-X PRIMARY]")
            }
        }
    }
    
    func stopSession() {
        if session.isRunning {
            session.stopRunning()
            print("🛑 Camera session stopped")
        }
    }
    
    // MAIN PROCESSING PIPELINE
    private func processFrame(pixelBuffer: CVPixelBuffer) {
        // Always store the latest frame
        capturePixelBuffer = pixelBuffer
        
        // Only process if user triggered capture
        guard shouldCapture else {
            DispatchQueue.main.async {
                self.isReadyToCapture = true
            }
            return
        }
        
        guard !isProcessing else { return }
        
        isProcessing = true
        shouldCapture = false  // Reset trigger
        
        print("\n🎬 === STARTING SEGMENTATION PIPELINE ===")
        
        DispatchQueue.main.async {
            self.isProcessingStarted = true
            self.processingProgress = 0.0
            self.progressMessage = "Starting detection..."
        }
        
        // STEP 1: Run FastSAM-X (PRIMARY) - 40% of progress
        DispatchQueue.main.async {
            self.processingProgress = 0.1
            self.progressMessage = "Detecting furniture..."
        }
        
        if fastsamModel != nil {
            print("🎯 Running FastSAM-X...")
            runFastSAMSync(pixelBuffer: pixelBuffer)
        } else {
            print("❌ FastSAM-X model not loaded!")
        }
        
        DispatchQueue.main.async {
            self.processingProgress = 0.4
        }
        
        // STEP 2: Run U2-Net (SUPPORT) - 20% of progress
        DispatchQueue.main.async {
            self.progressMessage = "Refining details..."
        }
        
        if u2netModel != nil {
            print("🔧 Running U2-Net...")
            runU2NetSync(pixelBuffer: pixelBuffer)
        }
        
        DispatchQueue.main.async {
            self.processingProgress = 0.6
        }
        
        // STEP 3: Run Canny edge detection - 10% of progress
        DispatchQueue.main.async {
            self.progressMessage = "Detecting edges..."
        }
        
        print("🔲 Running Canny...")
        runCannyEdgeDetection(pixelBuffer: pixelBuffer)
        
        DispatchQueue.main.async {
            self.processingProgress = 0.7
        }
        
        // STEP 4: Combine masks intelligently - 15% of progress
        DispatchQueue.main.async {
            self.progressMessage = "Combining results..."
        }
        
        var finalMask: CVPixelBuffer? = nil
        
        if let fastsamMask = fastsamMask {
            print("✅ FastSAM-X provided mask")
            // Apply temporal smoothing to FastSAM to reduce flickering
            let smoothedFastsam = applyTemporalSmoothing(to: fastsamMask)
            
            if let u2netMask = u2netMask, let cannyEdges = cannyEdges {
                finalMask = combineFastSAMWithU2NetAndCanny(fastsam: smoothedFastsam, u2net: u2netMask, canny: cannyEdges)
            } else if let u2netMask = u2netMask {
                finalMask = combineFastSAMWithU2Net(fastsam: smoothedFastsam, u2net: u2netMask)
            } else if let cannyEdges = cannyEdges {
                finalMask = combineFastSAMWithCanny(fastsam: smoothedFastsam, canny: cannyEdges)
            } else {
                finalMask = smoothedFastsam
            }
        } else if let u2netMask = u2netMask {
            // Fallback to U2-Net only if FastSAM fails
            print("⚠️ Falling back to U2-Net only")
            finalMask = u2netMask
        }
        
        DispatchQueue.main.async {
            self.processingProgress = 0.85
        }
        
        // STEP 5: Apply mask to image - 15% of progress
        DispatchQueue.main.async {
            self.progressMessage = "Finalizing..."
        }
        
        if let mask = finalMask {
            let maskImage = CIImage(cvPixelBuffer: mask)
            let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
            applyMaskToImage(original: ciImage, mask: maskImage)
        }
        
        DispatchQueue.main.async {
            self.processingProgress = 1.0
            self.progressMessage = "Complete!"
            
            // Hide progress after a short delay
            DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
                self.isProcessingStarted = false
            }
        }
        
        print("🎬 === SEGMENTATION COMPLETE ===\n")
        
        isProcessing = false
    }
    
    // TEMPORAL SMOOTHING: Reduce FastSAM flickering by averaging recent frames
    private func applyTemporalSmoothing(to mask: CVPixelBuffer) -> CVPixelBuffer {
        guard let copied = copyPixelBuffer(mask) else { return mask }
        
        previousFastsamMasks.append(copied)
        if previousFastsamMasks.count > temporalWindowSize {
            previousFastsamMasks.removeFirst()
        }
        
        // If we don't have enough frames yet, just return current
        if previousFastsamMasks.count < 2 {
            return mask
        }
        
        // Average the masks
        guard let averaged = averageMasks(previousFastsamMasks) else {
            return mask
        }
        
        return averaged
    }
    
    // RUN CANNY EDGE DETECTION
    private func runCannyEdgeDetection(pixelBuffer: CVPixelBuffer) {
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        
        // Convert to grayscale
        guard let grayFilter = CIFilter(name: "CIPhotoEffectMono") else { return }
        grayFilter.setValue(ciImage, forKey: kCIInputImageKey)
        guard let grayImage = grayFilter.outputImage else { return }
        
        // Apply Gaussian blur to reduce noise
        guard let blurFilter = CIFilter(name: "CIGaussianBlur") else { return }
        blurFilter.setValue(grayImage, forKey: kCIInputImageKey)
        blurFilter.setValue(1.0, forKey: kCIInputRadiusKey)
        guard let blurredImage = blurFilter.outputImage else { return }
        
        // Apply edge detection using Sobel
        guard let edgeFilter = CIFilter(name: "CIEdges") else { return }
        edgeFilter.setValue(blurredImage, forKey: kCIInputImageKey)
        edgeFilter.setValue(2.0, forKey: kCIInputIntensityKey)
        guard let edgeImage = edgeFilter.outputImage else { return }
        
        // Convert to pixel buffer
        var edgeBuffer: CVPixelBuffer?
        let attrs = [
            kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue!,
            kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue!,
            kCVPixelBufferPixelFormatTypeKey: kCVPixelFormatType_OneComponent8
        ] as CFDictionary
        
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        
        CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_OneComponent8, attrs, &edgeBuffer)
        
        if let buffer = edgeBuffer {
            context.render(edgeImage, to: buffer)
            cannyEdges = buffer
            
            let edgeCount = countWhitePixels(in: buffer)
            print("🔲 Canny edges: \(edgeCount) pixels")
        }
    }
    
    // COMBINATION: FastSAM + Canny only
    private func combineFastSAMWithCanny(fastsam: CVPixelBuffer, canny: CVPixelBuffer) -> CVPixelBuffer? {
        guard let result = copyPixelBuffer(fastsam) else { return fastsam }
        
        CVPixelBufferLockBaseAddress(result, [])
        CVPixelBufferLockBaseAddress(canny, .readOnly)
        defer {
            CVPixelBufferUnlockBaseAddress(result, [])
            CVPixelBufferUnlockBaseAddress(canny, .readOnly)
        }
        
        let width = CVPixelBufferGetWidth(result)
        let height = CVPixelBufferGetHeight(result)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(result)
        let cannyBytesPerRow = CVPixelBufferGetBytesPerRow(canny)
        
        guard let resultPtr = CVPixelBufferGetBaseAddress(result)?.assumingMemoryBound(to: UInt8.self),
              let cannyPtr = CVPixelBufferGetBaseAddress(canny)?.assumingMemoryBound(to: UInt8.self) else {
            return fastsam
        }
        
        var addedFromCanny = 0
        
        // Add Canny edges that are near FastSAM detection
        for y in 0..<height {
            for x in 0..<width {
                let resultIdx = y * bytesPerRow + x
                let cannyIdx = y * cannyBytesPerRow + x
                
                // Skip if already detected by FastSAM
                if resultPtr[resultIdx] > 128 {
                    continue
                }
                
                // Check if this is a strong edge
                let edgeValue = cannyPtr[cannyIdx]
                if edgeValue > 200 {  // Strong edge
                    // Check if near existing detection
                    var nearExisting = false
                    let searchRadius = 30
                    
                    for dy in -searchRadius...searchRadius {
                        for dx in -searchRadius...searchRadius {
                            let ny = y + dy
                            let nx = x + dx
                            if ny >= 0 && ny < height && nx >= 0 && nx < width {
                                if resultPtr[ny * bytesPerRow + nx] > 128 {
                                    nearExisting = true
                                    break
                                }
                            }
                        }
                        if nearExisting { break }
                    }
                    
                    if nearExisting {
                        resultPtr[resultIdx] = 255
                        addedFromCanny += 1
                    }
                }
            }
        }
        
        print("🔲 Added \(addedFromCanny) pixels from Canny edges")
        
        guard let smoothed = morphologicalClosing(result, iterations: 3) else { return result }
        return smoothed
    }
    
    // COMBINATION: FastSAM + U2-Net + Canny (full pipeline)
    private func combineFastSAMWithU2NetAndCanny(fastsam: CVPixelBuffer, u2net: CVPixelBuffer, canny: CVPixelBuffer) -> CVPixelBuffer? {
        // First combine FastSAM with U2-Net
        guard let fastsamU2Net = combineFastSAMWithU2Net(fastsam: fastsam, u2net: u2net) else {
            return fastsam
        }
        
        // Then add Canny edges
        guard let result = copyPixelBuffer(fastsamU2Net) else { return fastsamU2Net }
        
        CVPixelBufferLockBaseAddress(result, [])
        CVPixelBufferLockBaseAddress(canny, .readOnly)
        defer {
            CVPixelBufferUnlockBaseAddress(result, [])
            CVPixelBufferUnlockBaseAddress(canny, .readOnly)
        }
        
        let width = CVPixelBufferGetWidth(result)
        let height = CVPixelBufferGetHeight(result)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(result)
        let cannyBytesPerRow = CVPixelBufferGetBytesPerRow(canny)
        
        guard let resultPtr = CVPixelBufferGetBaseAddress(result)?.assumingMemoryBound(to: UInt8.self),
              let cannyPtr = CVPixelBufferGetBaseAddress(canny)?.assumingMemoryBound(to: UInt8.self) else {
            return fastsamU2Net
        }
        
        var addedFromCanny = 0
        
        // Add strong Canny edges near existing detection
        for y in 0..<height {
            for x in 0..<width {
                let resultIdx = y * bytesPerRow + x
                let cannyIdx = y * cannyBytesPerRow + x
                
                if resultPtr[resultIdx] > 128 {
                    continue
                }
                
                let edgeValue = cannyPtr[cannyIdx]
                if edgeValue > 200 {  // Very strong edges only
                    var nearExisting = false
                    let searchRadius = 25
                    
                    for dy in -searchRadius...searchRadius {
                        for dx in -searchRadius...searchRadius {
                            let ny = y + dy
                            let nx = x + dx
                            if ny >= 0 && ny < height && nx >= 0 && nx < width {
                                if resultPtr[ny * bytesPerRow + nx] > 128 {
                                    nearExisting = true
                                    break
                                }
                            }
                        }
                        if nearExisting { break }
                    }
                    
                    if nearExisting {
                        resultPtr[resultIdx] = 255
                        addedFromCanny += 1
                    }
                }
            }
        }
        
        print("🔲➕ Final: Added \(addedFromCanny) pixels from Canny edges")
        
        guard let smoothed = morphologicalClosing(result, iterations: 4) else { return result }
        return smoothed
    }
    
    // COMBINATION STRATEGY: FastSAM (primary) + U2-Net (support)
    private func combineFastSAMWithU2Net(fastsam: CVPixelBuffer, u2net: CVPixelBuffer) -> CVPixelBuffer? {
        // Strategy: Use FastSAM as base, use U2-Net to:
        // 1. Fill gaps in FastSAM detection
        // 2. Refine boundaries
        // 3. Add missing salient parts
        
        guard let result = copyPixelBuffer(fastsam) else { return fastsam }
        
        CVPixelBufferLockBaseAddress(result, [])
        CVPixelBufferLockBaseAddress(u2net, .readOnly)
        defer {
            CVPixelBufferUnlockBaseAddress(result, [])
            CVPixelBufferUnlockBaseAddress(u2net, .readOnly)
        }
        
        let width = CVPixelBufferGetWidth(result)
        let height = CVPixelBufferGetHeight(result)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(result)
        let u2netBytesPerRow = CVPixelBufferGetBytesPerRow(u2net)
        
        guard let resultPtr = CVPixelBufferGetBaseAddress(result)?.assumingMemoryBound(to: UInt8.self),
              let u2netPtr = CVPixelBufferGetBaseAddress(u2net)?.assumingMemoryBound(to: UInt8.self) else {
            return fastsam
        }
        
        var fastsamPixelCount = 0
        var addedFromU2Net = 0
        
        // Count FastSAM pixels first
        for y in 0..<height {
            for x in 0..<width {
                if resultPtr[y * bytesPerRow + x] > 128 {
                    fastsamPixelCount += 1
                }
            }
        }
        
        // Only add U2-Net pixels that are:
        // 1. High confidence (>200)
        // 2. Connected to FastSAM detection (within 50px)
        for y in 0..<height {
            for x in 0..<width {
                let resultIdx = y * bytesPerRow + x
                let u2netIdx = y * u2netBytesPerRow + x
                
                // Skip if already detected by FastSAM
                if resultPtr[resultIdx] > 128 {
                    continue
                }
                
                // Check U2-Net confidence
                let u2netValue = u2netPtr[u2netIdx]
                if u2netValue > 200 {  // High confidence only
                    // Check if near existing detection
                    var nearExisting = false
                    let searchRadius = 50
                    
                    for dy in -searchRadius...searchRadius {
                        for dx in -searchRadius...searchRadius {
                            let ny = y + dy
                            let nx = x + dx
                            if ny >= 0 && ny < height && nx >= 0 && nx < width {
                                if resultPtr[ny * bytesPerRow + nx] > 128 {
                                    nearExisting = true
                                    break
                                }
                            }
                        }
                        if nearExisting { break }
                    }
                    
                    if nearExisting {
                        resultPtr[resultIdx] = 255
                        addedFromU2Net += 1
                    }
                }
            }
        }
        
        print("🔀 Combined: FastSAM=\(fastsamPixelCount)px, U2-Net added=\(addedFromU2Net)px")
        
        // Apply morphological operations for smoothing
        guard let smoothed = morphologicalClosing(result, iterations: 5) else { return result }
        
        return smoothed
    }
    
    // AVERAGE MULTIPLE MASKS (for temporal smoothing)
    private func averageMasks(_ masks: [CVPixelBuffer]) -> CVPixelBuffer? {
        guard !masks.isEmpty else { return nil }
        guard let first = masks.first else { return nil }
        
        let width = CVPixelBufferGetWidth(first)
        let height = CVPixelBufferGetHeight(first)
        
        var result: CVPixelBuffer?
        let attrs = [
            kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue!,
            kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue!,
            kCVPixelBufferPixelFormatTypeKey: kCVPixelFormatType_OneComponent8
        ] as CFDictionary
        
        CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_OneComponent8, attrs, &result)
        
        guard let averaged = result else { return nil }
        
        CVPixelBufferLockBaseAddress(averaged, [])
        defer { CVPixelBufferUnlockBaseAddress(averaged, []) }
        
        guard let resultPtr = CVPixelBufferGetBaseAddress(averaged)?.assumingMemoryBound(to: UInt8.self) else {
            return nil
        }
        
        let bytesPerRow = CVPixelBufferGetBytesPerRow(averaged)
        
        // Sum all masks
        var sums = [Int](repeating: 0, count: height * width)
        
        for mask in masks {
            CVPixelBufferLockBaseAddress(mask, .readOnly)
            defer { CVPixelBufferUnlockBaseAddress(mask, .readOnly) }
            
            guard let maskPtr = CVPixelBufferGetBaseAddress(mask)?.assumingMemoryBound(to: UInt8.self) else {
                continue
            }
            
            let maskBytesPerRow = CVPixelBufferGetBytesPerRow(mask)
            
            for y in 0..<height {
                for x in 0..<width {
                    let idx = y * width + x
                    sums[idx] += Int(maskPtr[y * maskBytesPerRow + x])
                }
            }
        }
        
        // Average and threshold
        let count = masks.count
        for y in 0..<height {
            for x in 0..<width {
                let idx = y * width + x
                let avg = sums[idx] / count
                resultPtr[y * bytesPerRow + x] = avg > 128 ? 255 : 0
            }
        }
        
        return averaged
    }
    
    // RUN FASTSAM-X (PRIMARY)
    private func runFastSAMSync(pixelBuffer: CVPixelBuffer) {
        guard let model = fastsamModel else {
            print("❌ FastSAM-X model is nil")
            return
        }
        
        print("🔍 FastSAM-X model exists, creating request...")
        
        let request = VNCoreMLRequest(model: model) { [weak self] request, error in
            if let error = error {
                print("❌ FastSAM-X error: \(error.localizedDescription)")
                return
            }
            
            print("📦 FastSAM-X request completed")
            
            if let results = request.results {
                print("📊 FastSAM-X returned \(results.count) results of types: \(results.map { type(of: $0) })")
                
                if let pixelBufferResults = results as? [VNPixelBufferObservation] {
                    print("✅ FastSAM-X returned pixel buffer observations")
                    if let maskBuffer = pixelBufferResults.first?.pixelBuffer {
                        self?.fastsamMask = maskBuffer
                        
                        let pixelCount = self?.countWhitePixels(in: maskBuffer) ?? 0
                        print("🎯 FastSAM-X detected: \(pixelCount) pixels")
                    } else {
                        print("⚠️ FastSAM-X first result has no pixelBuffer")
                    }
                } else {
                    print("⚠️ FastSAM-X results are not VNPixelBufferObservation type")
                    if let firstResult = results.first {
                        print("   First result type: \(type(of: firstResult))")
                    }
                }
            } else {
                print("⚠️ FastSAM-X returned no results")
            }
        }
        
        request.imageCropAndScaleOption = .scaleFill
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
        
        do {
            print("🚀 Performing FastSAM-X request...")
            try handler.perform([request])
            print("✅ FastSAM-X request performed")
        } catch {
            print("❌ FastSAM-X handler error: \(error.localizedDescription)")
        }
    }
    
    // RUN U2-NET (SUPPORT)
    private func runU2NetSync(pixelBuffer: CVPixelBuffer) {
        guard let model = u2netModel else { return }
        
        let request = VNCoreMLRequest(model: model) { [weak self] request, error in
            if let error = error {
                print("❌ U2-Net error: \(error)")
                return
            }
            
            if let results = request.results as? [VNPixelBufferObservation],
               let maskBuffer = results.first?.pixelBuffer {
                self?.u2netMask = maskBuffer
                
                let pixelCount = self?.countWhitePixels(in: maskBuffer) ?? 0
                print("🔧 U2-Net support: \(pixelCount) pixels")
            }
        }
        
        request.imageCropAndScaleOption = .scaleFill
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
        
        do {
            try handler.perform([request])
        } catch {
            print("❌ U2-Net handler error: \(error)")
        }
    }
    
    // MORPHOLOGICAL OPERATIONS
    private func morphologicalClosing(_ buffer: CVPixelBuffer, iterations: Int) -> CVPixelBuffer? {
        guard var current = copyPixelBuffer(buffer) else { return buffer }
        
        for _ in 0..<iterations {
            guard let dilated = dilate(current) else { break }
            current = dilated
        }
        
        for _ in 0..<(iterations / 2) {
            guard let eroded = erode(current) else { break }
            current = eroded
        }
        
        return current
    }
    
    private func dilate(_ buffer: CVPixelBuffer) -> CVPixelBuffer? {
        guard let output = copyPixelBuffer(buffer) else { return nil }
        
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        CVPixelBufferLockBaseAddress(output, [])
        defer {
            CVPixelBufferUnlockBaseAddress(buffer, .readOnly)
            CVPixelBufferUnlockBaseAddress(output, [])
        }
        
        let width = CVPixelBufferGetWidth(buffer)
        let height = CVPixelBufferGetHeight(buffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
        
        guard let inputPtr = CVPixelBufferGetBaseAddress(buffer)?.assumingMemoryBound(to: UInt8.self),
              let outputPtr = CVPixelBufferGetBaseAddress(output)?.assumingMemoryBound(to: UInt8.self) else {
            return nil
        }
        
        for y in 1..<(height - 1) {
            for x in 1..<(width - 1) {
                var maxVal: UInt8 = 0
                
                for dy in -1...1 {
                    for dx in -1...1 {
                        let idx = (y + dy) * bytesPerRow + (x + dx)
                        maxVal = max(maxVal, inputPtr[idx])
                    }
                }
                
                outputPtr[y * bytesPerRow + x] = maxVal
            }
        }
        
        return output
    }
    
    private func erode(_ buffer: CVPixelBuffer) -> CVPixelBuffer? {
        guard let output = copyPixelBuffer(buffer) else { return nil }
        
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        CVPixelBufferLockBaseAddress(output, [])
        defer {
            CVPixelBufferUnlockBaseAddress(buffer, .readOnly)
            CVPixelBufferUnlockBaseAddress(output, [])
        }
        
        let width = CVPixelBufferGetWidth(buffer)
        let height = CVPixelBufferGetHeight(buffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
        
        guard let inputPtr = CVPixelBufferGetBaseAddress(buffer)?.assumingMemoryBound(to: UInt8.self),
              let outputPtr = CVPixelBufferGetBaseAddress(output)?.assumingMemoryBound(to: UInt8.self) else {
            return nil
        }
        
        for y in 1..<(height - 1) {
            for x in 1..<(width - 1) {
                var minVal: UInt8 = 255
                
                for dy in -1...1 {
                    for dx in -1...1 {
                        let idx = (y + dy) * bytesPerRow + (x + dx)
                        minVal = min(minVal, inputPtr[idx])
                    }
                }
                
                outputPtr[y * bytesPerRow + x] = minVal
            }
        }
        
        return output
    }
    
    // HELPER FUNCTIONS
    private func countWhitePixels(in buffer: CVPixelBuffer) -> Int {
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        
        let width = CVPixelBufferGetWidth(buffer)
        let height = CVPixelBufferGetHeight(buffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
        
        guard let baseAddress = CVPixelBufferGetBaseAddress(buffer) else {
            return 0
        }
        
        let bufferPtr = baseAddress.assumingMemoryBound(to: UInt8.self)
        var count = 0
        
        for y in 0..<height {
            for x in 0..<width {
                if bufferPtr[y * bytesPerRow + x] > 128 {
                    count += 1
                }
            }
        }
        
        return count
    }
    
    private func applyMaskToImage(original: CIImage, mask: CIImage) {
        let scaleX = original.extent.width / mask.extent.width
        let scaleY = original.extent.height / mask.extent.height
        
        let scaledMask = mask
            .transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))
            .samplingNearest()
        
        var finalMask = scaledMask
        
        if let blurFilter = CIFilter(name: "CIGaussianBlur") {
            blurFilter.setValue(scaledMask, forKey: kCIInputImageKey)
            blurFilter.setValue(0.3, forKey: kCIInputRadiusKey)
            if let blurred = blurFilter.outputImage {
                finalMask = blurred
            }
        }
        
        if let colorControls = CIFilter(name: "CIColorControls") {
            colorControls.setValue(finalMask, forKey: kCIInputImageKey)
            colorControls.setValue(2.0, forKey: kCIInputContrastKey)
            if let sharpened = colorControls.outputImage {
                finalMask = sharpened
            }
        }
        
        finalMask = finalMask.cropped(to: original.extent)
        
        guard let blendFilter = CIFilter(name: "CIBlendWithMask") else { return }
        
        let transparent = CIImage(color: CIColor(red: 0, green: 0, blue: 0, alpha: 0))
            .cropped(to: original.extent)
        
        blendFilter.setValue(original, forKey: kCIInputImageKey)
        blendFilter.setValue(transparent, forKey: kCIInputBackgroundImageKey)
        blendFilter.setValue(finalMask, forKey: kCIInputMaskImageKey)
        
        guard let result = blendFilter.outputImage else { return }
        
        let renderContext = CIContext(options: [
            .workingColorSpace: CGColorSpaceCreateDeviceRGB(),
            .outputPremultiplied: true,
            .useSoftwareRenderer: false
        ])
        
        if let cgImage = renderContext.createCGImage(result, from: result.extent) {
            let uiImage = UIImage(cgImage: cgImage, scale: 1.0, orientation: .up)
            
            DispatchQueue.main.async {
                self.segmentedImage = uiImage
                withAnimation(.easeIn(duration: 0.2)) {
                    self.furnitureOpacity = 1.0
                }
            }
        }
    }
    
    private func copyPixelBuffer(_ source: CVPixelBuffer) -> CVPixelBuffer? {
        let width = CVPixelBufferGetWidth(source)
        let height = CVPixelBufferGetHeight(source)
        let format = CVPixelBufferGetPixelFormatType(source)
        
        var copy: CVPixelBuffer?
        let attrs = [
            kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue!,
            kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue!
        ] as CFDictionary
        
        let status = CVPixelBufferCreate(kCFAllocatorDefault, width, height, format, attrs, &copy)
        
        guard status == kCVReturnSuccess, let destination = copy else { return nil }
        
        CVPixelBufferLockBaseAddress(source, .readOnly)
        CVPixelBufferLockBaseAddress(destination, [])
        
        defer {
            CVPixelBufferUnlockBaseAddress(source, .readOnly)
            CVPixelBufferUnlockBaseAddress(destination, [])
        }
        
        let sourceBytesPerRow = CVPixelBufferGetBytesPerRow(source)
        let destBytesPerRow = CVPixelBufferGetBytesPerRow(destination)
        
        guard let sourceData = CVPixelBufferGetBaseAddress(source),
              let destData = CVPixelBufferGetBaseAddress(destination) else {
            return nil
        }
        
        for row in 0..<height {
            let sourceRowData = sourceData.advanced(by: row * sourceBytesPerRow)
            let destRowData = destData.advanced(by: row * destBytesPerRow)
            memcpy(destRowData, sourceRowData, min(sourceBytesPerRow, destBytesPerRow))
        }
        
        return destination
    }
}

extension SegmentExamineModel: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        processFrame(pixelBuffer: pixelBuffer)
    }
}
