import SwiftUI
import AVFoundation
import CoreML
import Vision
import CoreImage
import Photos

// SEGMENT EXAMINE - REVISED ARCHITECTURE
// FastSAM-X runs as PRIMARY segmentation (better at complete furniture objects)
// U2-Net provides SUPPORT (stabilization, gap-filling, refinement)
// User can tap missing parts → MobileSAM decoder for additional refinement
// Result: ONE unified furniture segmentation with better handling of multi-part items

struct SegmentExamine: View {
    @Binding var capturedImage: UIImage?
    @Binding var isShowingCamera: Bool
    @StateObject private var camera = SegmentExamineModel()
    
    @State private var showingSaveSuccess = false
    @State private var showRawFeed = false
    
    var body: some View {
        ZStack {
            Color.clear.ignoresSafeArea()
            
            if showRawFeed {
                Color.black.opacity(0.7).ignoresSafeArea()
                CameraPreviewLayer(session: camera.session)
                    .ignoresSafeArea()
                    .opacity(0.8)
            }
            
            if let segmented = camera.segmentedImage {
                Image(uiImage: segmented)
                    .resizable()
                    .scaledToFit()
                    .ignoresSafeArea()
                    .opacity(camera.furnitureOpacity)
                    .animation(.easeOut(duration: 0.3), value: camera.furnitureOpacity)
            }
            
            if showingSaveSuccess {
                VStack {
                    Spacer()
                    HStack(spacing: 12) {
                        Image(systemName: "checkmark.circle.fill")
                            .font(.title2)
                        Text("Furniture saved!")
                            .font(.headline)
                    }
                    .foregroundColor(.white)
                    .padding(.horizontal, 24)
                    .padding(.vertical, 16)
                    .background(
                        Capsule()
                            .fill(Color.green.opacity(0.95))
                            .shadow(radius: 10)
                    )
                    .padding(.bottom, 150)
                }
                .transition(.move(edge: .bottom).combined(with: .opacity))
            }
            
            VStack {
                Spacer()
                
                if showRawFeed {
                    VStack(spacing: 4) {
                        HStack(spacing: 6) {
                            Image(systemName: "hand.tap.fill")
                                .font(.caption)
                            Text("Tap anywhere to enhance segmentation")
                                .font(.caption)
                        }
                        .foregroundColor(.white)
                        .padding(.horizontal, 12)
                        .padding(.vertical, 8)
                        .background(
                            Capsule()
                                .fill(Color.orange.opacity(0.8))
                        )
                    }
                    .frame(maxWidth: .infinity, alignment: .top)
                    .padding(.top, 80)
                }
                
                Spacer()
                
                VStack(spacing: 16) {
                    HStack(spacing: 12) {
                        if !camera.samMasks.isEmpty {
                            Text("\(camera.samMasks.count)× enhanced")
                                .font(.caption)
                                .foregroundColor(.white)
                                .padding(.horizontal, 10)
                                .padding(.vertical, 6)
                                .background(Capsule().fill(Color.orange.opacity(0.8)))
                        }
                        
                        if !camera.samMasks.isEmpty {
                            Button(action: {
                                withAnimation {
                                    camera.clearTaps()
                                }
                                let generator = UIImpactFeedbackGenerator(style: .light)
                                generator.impactOccurred()
                            }) {
                                HStack(spacing: 6) {
                                    Image(systemName: "arrow.counterclockwise")
                                        .font(.system(size: 16))
                                    Text("Reset")
                                        .font(.system(size: 16, weight: .semibold))
                                }
                                .foregroundColor(.white)
                                .padding(.horizontal, 16)
                                .padding(.vertical, 10)
                                .background(
                                    Capsule()
                                        .fill(Color.orange.opacity(0.8))
                                )
                            }
                        }
                        
                        Button(action: {
                            withAnimation {
                                if showRawFeed {
                                    camera.clearTapIndicators()
                                }
                                showRawFeed.toggle()
                            }
                            let generator = UIImpactFeedbackGenerator(style: .medium)
                            generator.impactOccurred()
                        }) {
                            HStack(spacing: 8) {
                                Image(systemName: showRawFeed ? "checkmark.circle.fill" : "plus.circle.fill")
                                    .font(.system(size: 16))
                                Text(showRawFeed ? "Done" : "Add Parts")
                                    .font(.system(size: 16, weight: .semibold))
                            }
                            .foregroundColor(.white)
                            .padding(.horizontal, 20)
                            .padding(.vertical, 12)
                            .background(
                                Capsule()
                                    .fill(showRawFeed ? Color.green.opacity(0.9) : Color.blue.opacity(0.8))
                            )
                        }
                        
                        Button(action: {
                            isShowingCamera = false
                        }) {
                            Image(systemName: "xmark.circle.fill")
                                .font(.system(size: 32))
                                .foregroundColor(.white)
                                .background(Circle().fill(Color.black.opacity(0.5)))
                        }
                    }
                    .padding(.horizontal)
                }
                .padding(.bottom, 40)
            }
        }
        .simultaneousGesture(
            showRawFeed ?
                DragGesture(minimumDistance: 0)
                    .onEnded { value in
                        let location = value.location
                        let screenHeight = UIScreen.main.bounds.height
                        let isInBottomButtons = location.y > screenHeight - 200
                        
                        if !isInBottomButtons {
                            camera.handleTap(at: location)
                        }
                    }
                : nil
        )
        .onAppear {
            AVCaptureDevice.requestAccess(for: .video) { granted in
                if granted {
                    camera.startSession()
                }
            }
        }
        .onDisappear {
            camera.stopSession()
        }
    }
}

struct CameraPreviewLayer: UIViewRepresentable {
    let session: AVCaptureSession
    
    func makeUIView(context: Context) -> UIView {
        let view = UIView(frame: .zero)
        
        let previewLayer = AVCaptureVideoPreviewLayer(session: session)
        previewLayer.videoGravity = .resizeAspectFill
        previewLayer.connection?.videoRotationAngle = 90
        view.layer.addSublayer(previewLayer)
        
        context.coordinator.previewLayer = previewLayer
        
        return view
    }
    
    func updateUIView(_ uiView: UIView, context: Context) {
        DispatchQueue.main.async {
            context.coordinator.previewLayer?.frame = uiView.bounds
        }
    }
    
    func makeCoordinator() -> Coordinator {
        Coordinator()
    }
    
    class Coordinator {
        var previewLayer: AVCaptureVideoPreviewLayer?
    }
}

// SEGMENT EXAMINE MODEL - FASTSAM-X PRIMARY
class SegmentExamineModel: NSObject, ObservableObject {
    @Published var segmentedImage: UIImage?
    @Published var furnitureOpacity: Double = 0.0
    @Published var isExamining: Bool = false
    
    @Published var samMasks: [CVPixelBuffer] = []
    @Published var tapPoints: [CGPoint] = []
    
    let session = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let videoQueue = DispatchQueue(label: "segmentExamineQueue", qos: .userInitiated)
    
    // MODEL HIERARCHY: FastSAM-X (primary) + U2-Net (support) + MobileSAM (tap refinement)
    private var fastsamModel: VNCoreMLModel?      // Primary: complete furniture detection
    private var u2netModel: VNCoreMLModel?        // Support: saliency refinement
    private var samEncoderModel: MLModel?         // Tap refinement encoder
    private var samDecoderModel: MLModel?         // Tap refinement decoder
    
    private let context = CIContext()
    
    private var lastProcessTime = Date()
    private let processInterval: TimeInterval = 0.2
    private var isProcessing = false
    
    // Detection buffers
    private var fastsamMask: CVPixelBuffer?       // Primary detection
    private var u2netMask: CVPixelBuffer?         // Support detection
    private var lockedMask: CVPixelBuffer? = nil
    private var shouldStartExaminingOnNextFrame = false
    private var currentPixelBuffer: CVPixelBuffer?
    private var currentImageEmbeddings: MLMultiArray?
    
    // Temporal stability tracking
    private var lastFastsamMaskHash: Int = 0
    private var sceneStableFrames: Int = 0
    private let sceneChangeThreshold: Int = 5
    
    // FastSAM temporal smoothing
    private var previousFastsamMasks: [CVPixelBuffer] = []
    private let temporalWindowSize = 3
    
    override init() {
        super.init()
        checkCameraAuthorization()
        loadFastSAMModel()
        loadU2NetModel()
        loadMobileSAMModels()
    }
    
    func clearTaps() {
        print("🗑️ Clearing SAM guidance")
        DispatchQueue.main.async {
            self.tapPoints.removeAll()
            self.samMasks.removeAll()
        }
    }
    
    func clearTapIndicators() {
        print("✨ Clearing tap indicators")
        DispatchQueue.main.async {
            self.tapPoints.removeAll()
        }
    }
    
    func handleTap(at location: CGPoint) {
        guard let pixelBuffer = currentPixelBuffer else {
            print("⚠️ No frame available")
            return
        }
        
        print("👆 Tap detected at: (\(Int(location.x)), \(Int(location.y)))")
        
        let generator = UIImpactFeedbackGenerator(style: .light)
        generator.impactOccurred()
        
        segmentWithMobileSAM(at: location, pixelBuffer: pixelBuffer)
    }
    
    private func segmentWithMobileSAM(at point: CGPoint, pixelBuffer: CVPixelBuffer) {
        guard let embeddings = currentImageEmbeddings,
              let decoder = samDecoderModel else {
            print("⚠️ MobileSAM not ready")
            return
        }
        
        let normalizedX = Float(point.x / UIScreen.main.bounds.width)
        let normalizedY = Float(point.y / UIScreen.main.bounds.height)
        
        print("🎯 Running MobileSAM at normalized: (\(String(format: "%.3f", normalizedX)), \(String(format: "%.3f", normalizedY)))")
        
        DispatchQueue.global(qos: .userInitiated).async {
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
                    print("❌ No mask from MobileSAM")
                    return
                }
                
                if let samMask = self.convertMLMultiArrayToPixelBuffer(masks) {
                    let pixelCount = self.countWhitePixels(in: samMask)
                    print("✅ MobileSAM: \(pixelCount) pixels")
                    
                    DispatchQueue.main.async {
                        self.samMasks.append(samMask)
                        print("📝 Total SAM masks: \(self.samMasks.count)")
                    }
                }
                
            } catch {
                print("❌ MobileSAM error: \(error)")
            }
        }
    }
    
    func startExamining() {
        print("🔍 User started EXAMINING")
        shouldStartExaminingOnNextFrame = true
    }
    
    func finishExamining() {
        print("✅ User clicked BACK")
        DispatchQueue.main.async {
            self.isExamining = false
            self.samMasks.removeAll()
            self.tapPoints.removeAll()
        }
        lockedMask = nil
        shouldStartExaminingOnNextFrame = false
        segmentedImage = nil
        furnitureOpacity = 0.0
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
            break
        }
    }
    
    // LOAD FASTSAM-X MODEL (PRIMARY)
    private func loadFastSAMModel() {
        let modelNames = ["fastsam-x", "fastsam_x", "FastSAM-x", "FastSAMX", "fastsam"]
        
        for name in modelNames {
            for ext in ["mlmodelc", "mlpackage"] {
                if let modelURL = Bundle.main.url(forResource: name, withExtension: ext) {
                    do {
                        let model = try MLModel(contentsOf: modelURL)
                        fastsamModel = try VNCoreMLModel(for: model)
                        print("✅ FastSAM-X loaded: \(name) [PRIMARY]")
                        return
                    } catch {
                        print("⚠️ Failed to load \(name).\(ext): \(error)")
                    }
                }
            }
        }
        print("❌ No FastSAM-X model found - CRITICAL")
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
    
    private func loadMobileSAMModels() {
        guard let encoderURL = Bundle.main.url(forResource: "MobileSAMImageEncoder", withExtension: "mlmodelc"),
              let decoderURL = Bundle.main.url(forResource: "MobileSAMMaskDecoder", withExtension: "mlmodelc") else {
            print("⚠️ MobileSAM models not found")
            return
        }
        
        do {
            let encoderConfig = MLModelConfiguration()
            encoderConfig.computeUnits = .all
            samEncoderModel = try MLModel(contentsOf: encoderURL, configuration: encoderConfig)
            
            let decoderConfig = MLModelConfiguration()
            decoderConfig.computeUnits = .all
            samDecoderModel = try MLModel(contentsOf: decoderURL, configuration: decoderConfig)
            
            print("✅ MobileSAM models loaded [TAP REFINEMENT]")
        } catch {
            print("⚠️ Failed to load MobileSAM: \(error)")
        }
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
        } catch {
            print("❌ Camera setup failed")
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
    
    // MAIN PROCESSING PIPELINE
    private func processFrame(pixelBuffer: CVPixelBuffer) {
        let now = Date()
        guard now.timeIntervalSince(lastProcessTime) >= processInterval else { return }
        guard !isProcessing else { return }
        
        isProcessing = true
        lastProcessTime = now
        
        currentPixelBuffer = pixelBuffer
        
        // STEP 1: Run FastSAM-X (PRIMARY) - detects complete furniture objects
        if fastsamModel != nil {
            runFastSAMSync(pixelBuffer: pixelBuffer)
        }
        
        // STEP 2: Run U2-Net (SUPPORT) - provides saliency-based refinement
        if u2netModel != nil {
            runU2NetSync(pixelBuffer: pixelBuffer)
        }
        
        // STEP 3: Run SAM encoder for tap-based refinement
        if samEncoderModel != nil {
            runSAMEncoder(pixelBuffer: pixelBuffer)
        }
        
        // STEP 4: Combine masks intelligently
        var finalMask: CVPixelBuffer? = nil
        
        if let fastsamMask = fastsamMask {
            // Apply temporal smoothing to FastSAM to reduce flickering
            let smoothedFastsam = applyTemporalSmoothing(to: fastsamMask)
            
            if samMasks.isEmpty {
                // No tap refinements - use FastSAM + U2-Net combination
                if let u2netMask = u2netMask {
                    finalMask = combineFastSAMWithU2Net(fastsam: smoothedFastsam, u2net: u2netMask)
                } else {
                    finalMask = smoothedFastsam
                }
            } else {
                // User tapped for refinement - merge tap regions
                finalMask = mergeWithTapRefinements(base: smoothedFastsam, samRegions: samMasks)
            }
        } else if let u2netMask = u2netMask {
            // Fallback to U2-Net only if FastSAM fails
            print("⚠️ Falling back to U2-Net only")
            finalMask = u2netMask
        }
        
        // STEP 5: Lock mask when examining starts
        if shouldStartExaminingOnNextFrame {
            if let mask = finalMask, let copied = copyPixelBuffer(mask) {
                lockedMask = copied
            }
            
            shouldStartExaminingOnNextFrame = false
            
            DispatchQueue.main.async {
                self.isExamining = true
            }
        } else if let locked = lockedMask {
            finalMask = locked
        }
        
        // STEP 6: Apply mask to image
        if let mask = finalMask {
            let maskImage = CIImage(cvPixelBuffer: mask)
            let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
            applyMaskToImage(original: ciImage, mask: maskImage)
        }
        
        // STEP 7: Check for scene changes (auto-clear tap refinements)
        if let mask = finalMask {
            let currentHash = hashMask(mask)
            
            if currentHash != lastFastsamMaskHash {
                sceneStableFrames = 0
                lastFastsamMaskHash = currentHash
            } else {
                sceneStableFrames += 1
            }
            
            if sceneStableFrames < sceneChangeThreshold && !samMasks.isEmpty {
                print("🔄 Scene changed - clearing tap refinements")
                DispatchQueue.main.async {
                    self.samMasks.removeAll()
                    self.tapPoints.removeAll()
                }
            }
        }
        
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
        
        print("📊 Temporal smoothing: averaging \(previousFastsamMasks.count) frames")
        return averaged
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
        // 3. Not excessive in size
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
    
    // MERGE TAP REFINEMENTS: Add user-tapped regions to base mask
    private func mergeWithTapRefinements(base: CVPixelBuffer, samRegions: [CVPixelBuffer]) -> CVPixelBuffer? {
        guard let result = copyPixelBuffer(base) else { return base }
        
        CVPixelBufferLockBaseAddress(result, [])
        defer { CVPixelBufferUnlockBaseAddress(result, []) }
        
        let width = CVPixelBufferGetWidth(result)
        let height = CVPixelBufferGetHeight(result)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(result)
        
        guard let resultPtr = CVPixelBufferGetBaseAddress(result)?.assumingMemoryBound(to: UInt8.self) else {
            return base
        }
        
        // Simply OR all tap regions with base
        for samRegion in samRegions {
            CVPixelBufferLockBaseAddress(samRegion, .readOnly)
            defer { CVPixelBufferUnlockBaseAddress(samRegion, .readOnly) }
            
            guard let samPtr = CVPixelBufferGetBaseAddress(samRegion)?.assumingMemoryBound(to: UInt8.self) else {
                continue
            }
            
            let samBytesPerRow = CVPixelBufferGetBytesPerRow(samRegion)
            
            for y in 0..<height {
                for x in 0..<width {
                    let resultIdx = y * bytesPerRow + x
                    let samIdx = y * samBytesPerRow + x
                    
                    if samPtr[samIdx] > 128 {
                        resultPtr[resultIdx] = 255
                    }
                }
            }
        }
        
        print("➕ Merged \(samRegions.count) tap refinements")
        
        // Smooth the result
        guard let smoothed = morphologicalClosing(result, iterations: 8) else { return result }
        
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
                // Threshold: if average > 128, consider it detected
                resultPtr[y * bytesPerRow + x] = avg > 128 ? 255 : 0
            }
        }
        
        return averaged
    }
    
    // RUN FASTSAM-X (PRIMARY)
    private func runFastSAMSync(pixelBuffer: CVPixelBuffer) {
        guard let model = fastsamModel else { return }
        
        let request = VNCoreMLRequest(model: model) { [weak self] request, error in
            if let error = error {
                print("❌ FastSAM-X error: \(error)")
                return
            }
            
            if let results = request.results as? [VNPixelBufferObservation],
               let maskBuffer = results.first?.pixelBuffer {
                self?.fastsamMask = maskBuffer
                
                let pixelCount = self?.countWhitePixels(in: maskBuffer) ?? 0
                print("🎯 FastSAM-X detected: \(pixelCount) pixels")
            }
        }
        
        request.imageCropAndScaleOption = .scaleFill
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
        
        do {
            try handler.perform([request])
        } catch {
            print("❌ FastSAM-X handler error: \(error)")
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
    
    private func runSAMEncoder(pixelBuffer: CVPixelBuffer) {
        guard let encoder = samEncoderModel else { return }
        
        do {
            guard let resized = resizePixelBuffer(pixelBuffer, width: 1024, height: 1024) else { return }
            guard let imageArray = pixelBufferToMLMultiArray(resized) else { return }
            
            let inputDict: [String: Any] = ["image": imageArray]
            let inputProvider = try MLDictionaryFeatureProvider(dictionary: inputDict)
            let output = try encoder.prediction(from: inputProvider)
            
            guard let embeddings = output.featureValue(for: "image_embeddings")?.multiArrayValue else { return }
            
            currentImageEmbeddings = embeddings
        } catch {
            print("❌ SAM Encoder failed: \(error)")
        }
    }
    
    // MORPHOLOGICAL OPERATIONS (unchanged from original)
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
    
    // HELPER FUNCTIONS (unchanged)
    private func convertMLMultiArrayToPixelBuffer(_ mlArray: MLMultiArray) -> CVPixelBuffer? {
        let height = mlArray.shape[2].intValue
        let width = mlArray.shape[3].intValue
        
        var pixelBuffer: CVPixelBuffer?
        let attrs = [
            kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue!,
            kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue!,
            kCVPixelBufferPixelFormatTypeKey: kCVPixelFormatType_OneComponent8
        ] as CFDictionary
        
        let status = CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_OneComponent8, attrs, &pixelBuffer)
        
        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
            return nil
        }
        
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        
        guard let baseAddress = CVPixelBufferGetBaseAddress(buffer) else {
            return nil
        }
        
        let bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
        let bufferPtr = baseAddress.assumingMemoryBound(to: UInt8.self)
        
        for y in 0..<height {
            for x in 0..<width {
                let indices = [0, 0, y, x] as [NSNumber]
                let value = mlArray[indices].floatValue
                bufferPtr[y * bytesPerRow + x] = value > 0.0 ? 255 : 0
            }
        }
        
        return buffer
    }
    
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
    
    private func hashMask(_ mask: CVPixelBuffer) -> Int {
        CVPixelBufferLockBaseAddress(mask, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(mask, .readOnly) }
        
        let width = CVPixelBufferGetWidth(mask)
        let height = CVPixelBufferGetHeight(mask)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(mask)
        
        guard let baseAddress = CVPixelBufferGetBaseAddress(mask) else {
            return 0
        }
        
        let maskPtr = baseAddress.assumingMemoryBound(to: UInt8.self)
        
        var whitePixels = 0
        var centerMass: (x: Int, y: Int) = (0, 0)
        
        for y in stride(from: 0, to: height, by: 10) {
            for x in stride(from: 0, to: width, by: 10) {
                if maskPtr[y * bytesPerRow + x] > 128 {
                    whitePixels += 1
                    centerMass.x += x
                    centerMass.y += y
                }
            }
        }
        
        if whitePixels > 0 {
            centerMass.x /= whitePixels
            centerMass.y /= whitePixels
        }
        
        return whitePixels * 1000 + centerMass.x + centerMass.y * width
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
        
        let context = CIContext(options: [
            .workingColorSpace: CGColorSpaceCreateDeviceRGB(),
            .outputPremultiplied: true,
            .useSoftwareRenderer: false
        ])
        
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

extension SegmentExamineModel: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        processFrame(pixelBuffer: pixelBuffer)
    }
}
