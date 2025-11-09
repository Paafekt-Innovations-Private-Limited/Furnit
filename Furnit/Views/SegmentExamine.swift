import SwiftUI
import AVFoundation
import CoreML
import Vision
import CoreImage
import Photos

// SEGMENT EXAMINE
// U2-Net runs continuously for auto-segmentation
// User can tap missing parts → MobileSAM guides U2-Net to expand segmentation
// ATTENTION ENHANCEMENT: Tapped regions get brightness +30%, contrast 1.5x BEFORE U2-Net
// Result: ONE unified furniture segmentation (not separate pieces merged)

struct SegmentExamine: View {
    @Binding var capturedImage: UIImage?
    @Binding var isShowingCamera: Bool
    @StateObject private var camera = SegmentExamineModel()
    
    @State private var showingSaveSuccess = false
    @State private var showRawFeed = false  // Toggle for raw camera feed
    
    var body: some View {
        ZStack {
            // Transparent background by default (shows 3D room underneath)
            Color.clear.ignoresSafeArea()
            
            // NEW: Live camera preview layer (only when toggled on)
            if showRawFeed {
                Color.black.opacity(0.7).ignoresSafeArea()  // Dim the room background
                CameraPreviewLayer(session: camera.session)
                    .ignoresSafeArea()
                    .opacity(0.8)  // Semi-transparent so room is slightly visible
            }
            
            if let segmented = camera.segmentedImage {
                Image(uiImage: segmented)
                    .resizable()
                    .scaledToFit()
                    .ignoresSafeArea()
                    .opacity(camera.furnitureOpacity)
                    .animation(.easeOut(duration: 0.3), value: camera.furnitureOpacity)
            }
            
            // Success message
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
                
                // Mode indicator at top-center (minimal)
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
                
                // ALL BUTTONS AT BOTTOM
                VStack(spacing: 16) {
                    // Control buttons row
                    HStack(spacing: 12) {
                        // SAM guidance count badge
                        if !camera.samMasks.isEmpty {
                            Text("\(camera.samMasks.count)× enhanced")
                                .font(.caption)
                                .foregroundColor(.white)
                                .padding(.horizontal, 10)
                                .padding(.vertical, 6)
                                .background(Capsule().fill(Color.orange.opacity(0.8)))
                        }
                        
                        // Clear button - appears when SAM guidance is active
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
                        
                        // Add Parts / Done button
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
                        
                        // Close button (X)
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
            // Only process taps when in tap mode
            showRawFeed ?
                DragGesture(minimumDistance: 0)
                    .onEnded { value in
                        let location = value.location
                        let screenHeight = UIScreen.main.bounds.height
                        
                        // Exclude only the bottom button area (200px from bottom)
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

// Camera Preview Layer
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

// SEGMENT EXAMINE MODEL
class SegmentExamineModel: NSObject, ObservableObject {
    @Published var segmentedImage: UIImage?
    @Published var furnitureOpacity: Double = 0.0
    @Published var isExamining: Bool = false
    
    // SAM guidance
    @Published var samMasks: [CVPixelBuffer] = []
    @Published var tapPoints: [CGPoint] = []
    
    let session = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let videoQueue = DispatchQueue(label: "segmentExamineQueue", qos: .userInitiated)
    
    private var u2netModel: VNCoreMLModel?
    private var samEncoderModel: MLModel?
    private var samDecoderModel: MLModel?
    
    private let context = CIContext()
    
    private var lastProcessTime = Date()
    private let processInterval: TimeInterval = 0.2
    private var isProcessing = false
    
    private var u2netMask: CVPixelBuffer?
    private var lockedMask: CVPixelBuffer? = nil
    private var shouldStartExaminingOnNextFrame = false
    private var currentPixelBuffer: CVPixelBuffer?
    private var currentImageEmbeddings: MLMultiArray?
    
    // Track scene changes to auto-clear SAM guidance
    private var lastU2NetMaskHash: Int = 0
    private var sceneStableFrames: Int = 0
    private let sceneChangeThreshold: Int = 5
    
    override init() {
        super.init()
        checkCameraAuthorization()
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
    
    // ATTENTION ENHANCEMENT: Brighten and increase contrast in SAM regions
    private func enhanceImageInRegion(pixelBuffer: CVPixelBuffer, samMask: CVPixelBuffer) -> CVPixelBuffer? {
        let originalCI = CIImage(cvPixelBuffer: pixelBuffer)
        let maskCI = CIImage(cvPixelBuffer: samMask)
        
        // Scale mask to match original image size
        let scaleX = originalCI.extent.width / maskCI.extent.width
        let scaleY = originalCI.extent.height / maskCI.extent.height
        let scaledMask = maskCI.transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))
        
        // Enhance brightness +30%, contrast 1.5x
        let enhanced = originalCI
            .applyingFilter("CIColorControls", parameters: [
                "inputBrightness": 0.3,
                "inputContrast": 1.5
            ])
        
        // Blend enhanced region with original using mask
        let blended = enhanced.applyingFilter("CIBlendWithMask", parameters: [
            "inputBackgroundImage": originalCI,
            "inputMaskImage": scaledMask
        ])
        
        // Convert back to CVPixelBuffer
        var outputBuffer: CVPixelBuffer?
        let attrs = [
            kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue!,
            kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue!
        ] as CFDictionary
        
        CVPixelBufferCreate(kCFAllocatorDefault,
                           CVPixelBufferGetWidth(pixelBuffer),
                           CVPixelBufferGetHeight(pixelBuffer),
                           kCVPixelFormatType_32BGRA,
                           attrs,
                           &outputBuffer)
        
        if let output = outputBuffer {
            context.render(blended, to: output)
            return output
        }
        
        return nil
    }
    
    // Merge all SAM masks into one
    private func mergeSAMMasks() -> CVPixelBuffer? {
        guard !samMasks.isEmpty else { return nil }
        guard let first = samMasks.first else { return nil }
        
        let width = CVPixelBufferGetWidth(first)
        let height = CVPixelBufferGetHeight(first)
        
        var mergedBuffer: CVPixelBuffer?
        let attrs = [
            kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue!,
            kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue!,
            kCVPixelBufferPixelFormatTypeKey: kCVPixelFormatType_OneComponent8
        ] as CFDictionary
        
        CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_OneComponent8, attrs, &mergedBuffer)
        
        guard let merged = mergedBuffer else { return nil }
        
        CVPixelBufferLockBaseAddress(merged, [])
        defer { CVPixelBufferUnlockBaseAddress(merged, []) }
        
        let mergedData = CVPixelBufferGetBaseAddress(merged)!
        let bytesPerRow = CVPixelBufferGetBytesPerRow(merged)
        
        // Initialize to black
        memset(mergedData, 0, height * bytesPerRow)
        
        // Merge all masks (OR operation)
        for mask in samMasks {
            CVPixelBufferLockBaseAddress(mask, .readOnly)
            let maskData = CVPixelBufferGetBaseAddress(mask)!
            let maskBytesPerRow = CVPixelBufferGetBytesPerRow(mask)
            
            for y in 0..<height {
                for x in 0..<width {
                    let mergedOffset = y * bytesPerRow + x
                    let maskOffset = y * maskBytesPerRow + x
                    let maskPixel = maskData.load(fromByteOffset: maskOffset, as: UInt8.self)
                    
                    if maskPixel > 0 {
                        mergedData.storeBytes(of: UInt8(255), toByteOffset: mergedOffset, as: UInt8.self)
                    }
                }
            }
            
            CVPixelBufferUnlockBaseAddress(mask, .readOnly)
        }
        
        return merged
    }
    
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
    
    private func loadU2NetModel() {
        let modelNames = ["u2netp", "U2Net", "u2net", "U2NetP", "U2NET"]
        
        for name in modelNames {
            for ext in ["mlmodelc", "mlpackage"] {
                if let modelURL = Bundle.main.url(forResource: name, withExtension: ext) {
                    do {
                        let model = try MLModel(contentsOf: modelURL)
                        u2netModel = try VNCoreMLModel(for: model)
                        print("✅ U2-Net loaded: \(name)")
                        return
                    } catch {
                        print("⚠️ Failed to load \(name).\(ext): \(error)")
                    }
                }
            }
        }
        print("⚠️ No U2-Net model loaded")
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
            
            print("✅ MobileSAM models loaded")
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
    
    private func processFrame(pixelBuffer: CVPixelBuffer) {
        let now = Date()
        guard now.timeIntervalSince(lastProcessTime) >= processInterval else { return }
        guard !isProcessing else { return }
        
        isProcessing = true
        lastProcessTime = now
        
        currentPixelBuffer = pixelBuffer
        
        // ATTENTION ENHANCEMENT: If SAM masks exist, enhance those regions before U2-Net
        var enhancedPixelBuffer = pixelBuffer
        if !samMasks.isEmpty {
            if let mergedSAMMask = mergeSAMMasks() {
                if let enhanced = enhanceImageInRegion(pixelBuffer: pixelBuffer, samMask: mergedSAMMask) {
                    enhancedPixelBuffer = enhanced
                    print("✨ Attention enhancement applied: brightness +30%, contrast 1.5x")
                }
            }
        }
        
        // Run U2-Net on (possibly enhanced) image
        if u2netModel != nil {
            runU2NetSync(pixelBuffer: enhancedPixelBuffer)
        }
        
        // Check scene changes
        if let u2netMask = u2netMask {
            let currentHash = hashMask(u2netMask)
            
            if currentHash != lastU2NetMaskHash {
                sceneStableFrames = 0
                lastU2NetMaskHash = currentHash
            } else {
                sceneStableFrames += 1
            }
            
            if sceneStableFrames < sceneChangeThreshold && !samMasks.isEmpty {
                print("🔄 Scene changed - clearing SAM guidance")
                DispatchQueue.main.async {
                    self.samMasks.removeAll()
                    self.tapPoints.removeAll()
                }
            }
        }
        
        // Run SAM encoder
        if samEncoderModel != nil {
            runSAMEncoder(pixelBuffer: pixelBuffer)
        }
        
        // Conditional masking
        var finalMask: CVPixelBuffer? = nil
        
        if let u2netMask = u2netMask {
            if samMasks.isEmpty {
                finalMask = u2netMask
            } else {
                finalMask = conditionalMasking(baseU2Net: u2netMask, samRegions: samMasks, pixelBuffer: pixelBuffer)
            }
        }
        
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
        
        if let mask = finalMask {
            let maskImage = CIImage(cvPixelBuffer: mask)
            let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
            applyMaskToImage(original: ciImage, mask: maskImage)
        }
        
        isProcessing = false
    }
    
    private func conditionalMasking(baseU2Net: CVPixelBuffer, samRegions: [CVPixelBuffer], pixelBuffer: CVPixelBuffer) -> CVPixelBuffer? {
        guard let result = copyPixelBuffer(baseU2Net) else { return baseU2Net }
        
        CVPixelBufferLockBaseAddress(result, [])
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer {
            CVPixelBufferUnlockBaseAddress(result, [])
            CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly)
        }
        
        let width = CVPixelBufferGetWidth(result)
        let height = CVPixelBufferGetHeight(result)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(result)
        
        guard let resultPtr = CVPixelBufferGetBaseAddress(result)?.assumingMemoryBound(to: UInt8.self),
              let imagePtr = CVPixelBufferGetBaseAddress(pixelBuffer)?.assumingMemoryBound(to: UInt8.self) else {
            return baseU2Net
        }
        
        let imageBytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        
        for samRegion in samRegions {
            CVPixelBufferLockBaseAddress(samRegion, .readOnly)
            defer { CVPixelBufferUnlockBaseAddress(samRegion, .readOnly) }
            
            guard let samPtr = CVPixelBufferGetBaseAddress(samRegion)?.assumingMemoryBound(to: UInt8.self) else {
                continue
            }
            
            let samBytesPerRow = CVPixelBufferGetBytesPerRow(samRegion)
            
            var regionPixels: [(x: Int, y: Int)] = []
            var avgBrightness: Int = 0
            var minX = width, maxX = 0, minY = height, maxY = 0
            
            for y in 0..<height {
                for x in 0..<width {
                    let samIdx = y * samBytesPerRow + x
                    if samPtr[samIdx] > 128 {
                        regionPixels.append((x, y))
                        minX = min(minX, x)
                        maxX = max(maxX, x)
                        minY = min(minY, y)
                        maxY = max(maxY, y)
                        
                        let imgIdx = y * imageBytesPerRow + x * 4
                        let b = Int(imagePtr[imgIdx])
                        let g = Int(imagePtr[imgIdx + 1])
                        let r = Int(imagePtr[imgIdx + 2])
                        avgBrightness += (r + g + b) / 3
                    }
                }
            }
            
            if regionPixels.isEmpty { continue }
            avgBrightness /= regionPixels.count
            
            print("📍 SAM Region: \(regionPixels.count) pixels, brightness: \(avgBrightness)")
            
            for (x, y) in regionPixels {
                let idx = y * bytesPerRow + x
                resultPtr[idx] = 255
            }
            
            let expandMargin = 20
            for y in max(0, minY - expandMargin)..<min(height, maxY + expandMargin) {
                for x in max(0, minX - expandMargin)..<min(width, maxX + expandMargin) {
                    let idx = y * bytesPerRow + x
                    
                    if resultPtr[idx] > 128 { continue }
                    
                    let imgIdx = y * imageBytesPerRow + x * 4
                    let b = Int(imagePtr[imgIdx])
                    let g = Int(imagePtr[imgIdx + 1])
                    let r = Int(imagePtr[imgIdx + 2])
                    let brightness = (r + g + b) / 3
                    
                    if abs(brightness - avgBrightness) < 40 {
                        var hasNearbyMask = false
                        for dy in -3...3 {
                            for dx in -3...3 {
                                let ny = y + dy
                                let nx = x + dx
                                if ny >= 0 && ny < height && nx >= 0 && nx < width {
                                    if resultPtr[ny * bytesPerRow + nx] > 128 {
                                        hasNearbyMask = true
                                        break
                                    }
                                }
                            }
                            if hasNearbyMask { break }
                        }
                        
                        if hasNearbyMask {
                            resultPtr[idx] = 255
                        }
                    }
                }
            }
        }
        
        guard let smoothed = morphologicalClosing(result, iterations: 8) else { return result }
        
        print("✅ Conditional masking: \(samRegions.count) region(s) processed")
        return smoothed
    }
    
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
