import SwiftUI
import AVFoundation
import CoreML
import Vision
import CoreImage
import Photos

struct SegmentForeground: View {
    @Binding var capturedImage: UIImage?
    @Binding var isShowingCamera: Bool
    @StateObject private var camera = ForegroundCameraModel()
    
    @State private var scaleMultiplier: CGFloat = 0.5
    @State private var dragOffset: CGSize = .zero
    @State private var accumulatedOffset: CGSize = .zero
    @State private var isInitialAppearance = true
    @State private var scannerRotation: Double = 0
    @State private var lastHapticTime: Date = .distantPast
    @State private var showingSaveSuccess = false
    
    private let showSensitivitySlider = false
    private let showDebugBoxes = false
    
    var body: some View {
        ZStack {
            Color.clear
                .ignoresSafeArea()
            
            if showDebugBoxes && camera.u2netCoverageRect != .zero {
                Rectangle()
                    .stroke(Color.purple, lineWidth: 3)
                    .frame(width: camera.u2netCoverageRect.width,
                           height: camera.u2netCoverageRect.height)
                    .position(x: camera.u2netCoverageRect.midX,
                             y: camera.u2netCoverageRect.midY)
                
                Text("Foreground Area")
                    .font(.caption)
                    .foregroundColor(.purple)
                    .background(Color.black.opacity(0.7))
                    .position(x: camera.u2netCoverageRect.midX,
                             y: camera.u2netCoverageRect.minY - 20)
            }
            
            if !camera.isExamining && camera.segmentedImage == nil && !isInitialAppearance {
                ScanningReticleForeground(rotation: $scannerRotation)
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
                            .onChanged { value in
                                dragOffset = value.translation
                            }
                            .onEnded { value in
                                accumulatedOffset.width += value.translation.width
                                accumulatedOffset.height += value.translation.height
                                dragOffset = .zero
                            }
                    )
                    .ignoresSafeArea()
                    .opacity(camera.furnitureOpacity)
                    .animation(.easeOut(duration: 0.3), value: camera.furnitureOpacity)
                    .onChange(of: camera.detectionId) {
                        let now = Date()
                        if now.timeIntervalSince(lastHapticTime) > 2.0 {
                            let generator = UIImpactFeedbackGenerator(style: .medium)
                            generator.impactOccurred()
                            lastHapticTime = now
                        }
                    }
            }
            
            if showingSaveSuccess {
                VStack {
                    Spacer()
                    HStack(spacing: 12) {
                        Image(systemName: "checkmark.circle.fill")
                            .font(.title2)
                        Text("Photo saved!")
                            .font(.headline)
                    }
                    .foregroundColor(.white)
                    .padding(.horizontal, 24)
                    .padding(.vertical, 16)
                    .background(
                        Capsule()
                            .fill(Color.purple.opacity(0.95))
                            .shadow(radius: 10)
                    )
                    .padding(.bottom, 150)
                }
                .transition(.move(edge: .bottom).combined(with: .opacity))
            }
            
            VStack {
                HStack {
                    HStack(spacing: 6) {
                        Image(systemName: "minus.magnifyingglass")
                            .foregroundColor(.white)
                            .font(.system(size: 14))
                        Slider(value: $scaleMultiplier, in: 0.3...1.0)
                            .frame(width: 150)
                            .accentColor(.white)
                        Image(systemName: "plus.magnifyingglass")
                            .foregroundColor(.white)
                            .font(.system(size: 14))
                    }
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                    .background(Capsule().fill(Color.black.opacity(0.7)))
                    .padding(.leading, 16)
                    
                    Spacer()
                    
                    if accumulatedOffset != .zero || dragOffset != .zero {
                        Button(action: {
                            withAnimation(.spring()) {
                                dragOffset = .zero
                                accumulatedOffset = .zero
                            }
                        }) {
                            Image(systemName: "arrow.counterclockwise.circle.fill")
                                .font(.title2)
                                .foregroundColor(.white)
                                .background(Circle().fill(Color.black.opacity(0.5)))
                        }
                        .padding(.trailing, 8)
                        .transition(.scale)
                    }
                    
                    Button(action: {
                        isShowingCamera = false
                    }) {
                        Image(systemName: "xmark.circle.fill")
                            .font(.title)
                            .foregroundColor(.white)
                            .background(Circle().fill(Color.black.opacity(0.5)))
                            .shadow(radius: 3)
                    }
                    .padding(.trailing, 16)
                }
                .padding(.top, 60)
                
                if let hint = camera.userGuidanceHint, !camera.isExamining {
                    VStack(spacing: 4) {
                        HStack(spacing: 6) {
                            Image(systemName: hint.icon)
                                .font(.caption)
                            Text(hint.message)
                                .font(.caption)
                                .multilineTextAlignment(.center)
                        }
                        .foregroundColor(.white)
                        .padding(.horizontal, 12)
                        .padding(.vertical, 8)
                        .background(
                            Capsule()
                                .fill(hint.color.opacity(0.8))
                        )
                    }
                    .padding(.top, 8)
                    .transition(.opacity)
                }
                
                Spacer()
                
                VStack(spacing: 12) {
                    HStack {
                        if !camera.isExamining {
                            if camera.segmentedImage == nil {
                                HStack(spacing: 4) {
                                    ProgressView()
                                        .progressViewStyle(CircularProgressViewStyle(tint: .white))
                                        .scaleEffect(0.7)
                                    Text("Detecting scene...")
                                        .font(.caption)
                                }
                                .padding(8)
                                .background(Capsule().fill(Color.black.opacity(0.5)))
                                .foregroundColor(.white)
                            } else {
                                Text("Tap to capture photo")
                                    .font(.caption)
                                    .padding(8)
                                    .background(Capsule().fill(Color.black.opacity(0.5)))
                                    .foregroundColor(.white)
                            }
                        } else {
                            VStack(spacing: 4) {
                                Text("Scene ready • Size: \(Int(scaleMultiplier * 100))%")
                                    .font(.caption)
                                Text("Adjust position • Ready to save")
                                    .font(.caption2)
                                    .foregroundColor(.white.opacity(0.7))
                            }
                            .padding(8)
                            .background(Capsule().fill(Color.black.opacity(0.5)))
                            .foregroundColor(.white)
                        }
                    }
                    
                    HStack(spacing: 16) {
                        if camera.isExamining {
                            Button(action: {
                                savePhoto()
                            }) {
                                HStack(spacing: 8) {
                                    Image(systemName: "camera.fill")
                                        .font(.title3)
                                    Text("Take Photo")
                                        .font(.headline)
                                }
                                .foregroundColor(.white)
                                .padding(.horizontal, 24)
                                .padding(.vertical, 12)
                                .background(
                                    Capsule()
                                        .fill(Color.purple.opacity(0.9))
                                        .shadow(color: .purple.opacity(0.3), radius: 8)
                                )
                            }
                            
                            Button(action: {
                                withAnimation {
                                    camera.finishExamining()
                                }
                            }) {
                                HStack(spacing: 8) {
                                    Image(systemName: "arrow.counterclockwise")
                                        .font(.title3)
                                    Text("Back")
                                        .font(.headline)
                                }
                                .foregroundColor(.white)
                                .padding(.horizontal, 24)
                                .padding(.vertical, 12)
                                .background(
                                    Capsule()
                                        .fill(Color.gray.opacity(0.9))
                                        .shadow(color: .gray.opacity(0.3), radius: 8)
                                )
                            }
                        } else if camera.segmentedImage != nil {
                            Button(action: {
                                withAnimation {
                                    camera.startExamining()
                                }
                                let generator = UIImpactFeedbackGenerator(style: .medium)
                                generator.impactOccurred()
                            }) {
                                HStack(spacing: 8) {
                                    Image(systemName: "camera.aperture")
                                        .font(.title3)
                                    Text("Capture")
                                        .font(.headline)
                                }
                                .foregroundColor(.white)
                                .padding(.horizontal, 32)
                                .padding(.vertical, 12)
                                .background(
                                    Capsule()
                                        .fill(Color.purple.opacity(0.9))
                                        .shadow(color: .purple.opacity(0.3), radius: 8)
                                )
                            }
                            .scaleEffect(1.05)
                            .animation(.easeInOut(duration: 0.3).repeatForever(autoreverses: true), value: camera.segmentedImage != nil)
                        }
                    }
                }
                .padding(.bottom, 50)
                .padding(.horizontal)
            }
        }
        .onAppear {
            print("🎨 [SegmentForeground] Starting scene detection...")
            AVCaptureDevice.requestAccess(for: .video) { granted in
                if granted {
                    camera.startSession()
                    
                    let minDelay = 1.0
                    let maxDelay = 5.0
                    let checkInterval = 0.1
                    var elapsed = 0.0
                    
                    func checkForSegmentation() {
                        elapsed += checkInterval
                        
                        if camera.segmentedImage != nil && elapsed >= minDelay {
                            DispatchQueue.main.async {
                                withAnimation(.easeOut(duration: 0.4)) {
                                    isInitialAppearance = false
                                }
                            }
                        } else if elapsed >= maxDelay {
                            DispatchQueue.main.async {
                                withAnimation(.easeOut(duration: 0.4)) {
                                    isInitialAppearance = false
                                }
                            }
                        } else {
                            DispatchQueue.main.asyncAfter(deadline: .now() + checkInterval) {
                                checkForSegmentation()
                            }
                        }
                    }
                    
                    DispatchQueue.main.asyncAfter(deadline: .now() + minDelay) {
                        checkForSegmentation()
                    }
                }
            }
        }
        .onDisappear {
            print("👋 [SegmentForeground] Stopping...")
            camera.stopSession()
        }
    }
    
    private func savePhoto() {
        guard let fullImage = camera.originalCameraImage else {
            print("⚠️ No original camera image available")
            return
        }
        
        capturedImage = fullImage
        
        PHPhotoLibrary.requestAuthorization(for: .addOnly) { status in
            DispatchQueue.main.async {
                if status == .authorized || status == .limited {
                    PHPhotoLibrary.shared().performChanges({
                        PHAssetChangeRequest.creationRequestForAsset(from: fullImage)
                    }) { success, error in
                        DispatchQueue.main.async {
                            if success {
                                print("✅ Full photo saved to Photos!")
                                
                                withAnimation(.spring()) {
                                    self.showingSaveSuccess = true
                                }
                                
                                let generator = UINotificationFeedbackGenerator()
                                generator.notificationOccurred(.success)
                                
                                DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) {
                                    withAnimation {
                                        self.showingSaveSuccess = false
                                    }
                                    
                                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                                        self.isShowingCamera = false
                                    }
                                }
                            } else {
                                print("❌ Failed to save: \(error?.localizedDescription ?? "unknown error")")
                                DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                                    self.isShowingCamera = false
                                }
                            }
                        }
                    }
                } else {
                    print("⚠️ Photos access denied")
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                        self.isShowingCamera = false
                    }
                }
            }
        }
    }
}

struct ScanningReticleForeground: View {
    @Binding var rotation: Double
    
    var body: some View {
        ZStack {
            Circle()
                .stroke(Color.purple.opacity(0.3), lineWidth: 2)
                .frame(width: 120, height: 120)
            Rectangle()
                .fill(LinearGradient(gradient: Gradient(colors: [.clear, .purple.opacity(0.8), .clear]), startPoint: .top, endPoint: .bottom))
                .frame(width: 2, height: 60)
                .offset(y: -30)
                .rotationEffect(.degrees(rotation))
            Circle()
                .fill(Color.purple.opacity(0.5))
                .frame(width: 8, height: 8)
            ForEach(0..<4) { i in
                CornerBracketForeground()
                    .rotationEffect(.degrees(Double(i) * 90))
            }
        }
    }
}

struct CornerBracketForeground: View {
    var body: some View {
        Path { path in
            path.move(to: CGPoint(x: -60, y: -60))
            path.addLine(to: CGPoint(x: -40, y: -60))
            path.move(to: CGPoint(x: -60, y: -60))
            path.addLine(to: CGPoint(x: -60, y: -40))
        }
        .stroke(Color.purple.opacity(0.6), lineWidth: 2)
    }
}

class ForegroundCameraModel: NSObject, ObservableObject {
    @Published var segmentedImage: UIImage?
    @Published var originalCameraImage: UIImage?
    @Published var furnitureOpacity: Double = 0.0
    @Published var detectionId: UUID = UUID()
    @Published var isExamining: Bool = false
    @Published var userGuidanceHint: UserGuidanceHint? = nil
    @Published var u2netCoverageRect: CGRect = .zero
    
    let session = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let videoQueue = DispatchQueue(label: "foregroundVideoQueue", qos: .userInitiated)
    
    private var u2netModel: VNCoreMLModel?
    private let context = CIContext()
    
    private var lastProcessTime = Date()
    private let processInterval: TimeInterval = 0.15  // Faster updates for smooth feed
    private var isProcessing = false
    
    private var u2netMask: CVPixelBuffer?
    private var lockedMask: CVPixelBuffer? = nil
    private var lockedSegmentedImage: UIImage? = nil
    private var shouldStartExaminingOnNextFrame = false
    
    override init() {
        super.init()
        print("🎨 [ForegroundCameraModel] Initializing...")
        checkCameraAuthorization()
        loadU2NetModel()
    }
    
    func startExamining() {
        print("🔒 [ForegroundCameraModel] Capturing scene...")
        shouldStartExaminingOnNextFrame = true
    }
    
    func finishExamining() {
        print("🔓 [ForegroundCameraModel] Returning to live view...")
        DispatchQueue.main.async {
            self.isExamining = false
            self.userGuidanceHint = nil
        }
        lockedMask = nil
        lockedSegmentedImage = nil
        shouldStartExaminingOnNextFrame = false
        segmentedImage = nil
        originalCameraImage = nil
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
                        print("✅ [ForegroundCameraModel] U2-Net loaded: \(name)")
                        return
                    } catch {
                        print("⚠️ [ForegroundCameraModel] Failed to load \(name).\(ext): \(error)")
                    }
                }
            }
        }
        print("⚠️ [ForegroundCameraModel] No U2-Net model loaded")
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
            print("❌ [ForegroundCameraModel] Camera setup failed")
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
    
    private func processWithU2Net(pixelBuffer: CVPixelBuffer) {
        let now = Date()
        guard now.timeIntervalSince(lastProcessTime) >= processInterval else { return }
        guard !isProcessing else { return }
        
        isProcessing = true
        lastProcessTime = now
        
        // Capture original camera frame
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        if let cgImage = context.createCGImage(ciImage, from: ciImage.extent) {
            let originalImage = UIImage(cgImage: cgImage, scale: 1.0, orientation: .up)
            DispatchQueue.main.async {
                self.originalCameraImage = originalImage
            }
        }
        
        // Run U2-Net segmentation
        if u2netModel != nil {
            runU2NetSync(pixelBuffer: pixelBuffer)
        }
        
        guard let u2netMask = u2netMask else {
            isProcessing = false
            return
        }
        
        var maskToUse = u2netMask
        
        // Handle examination mode (freeze frame)
        if shouldStartExaminingOnNextFrame {
            print("📋 [ForegroundCameraModel] Locking current frame...")
            
            if let copiedMask = copyPixelBuffer(u2netMask) {
                lockedMask = copiedMask
                print("✅ Locked mask")
            } else {
                lockedMask = u2netMask
            }
            
            if let currentSegmented = segmentedImage {
                lockedSegmentedImage = currentSegmented
                print("✅ Locked segmented image")
            }
            
            shouldStartExaminingOnNextFrame = false
            
            DispatchQueue.main.async {
                self.isExamining = true
            }
        } else if let locked = lockedMask {
            // Use frozen mask during examination
            maskToUse = locked
        } else {
            // Live mode: use U2-Net mask directly (NO Canny processing!)
            analyzeAndProvideGuidance(mask: maskToUse)
        }
        
        // Update visualization
        updateU2NetVisualization(maskToUse)
        
        // Apply mask to create segmented image
        if !isExamining {
            let maskImage = CIImage(cvPixelBuffer: maskToUse)
            let ciImageForMask = CIImage(cvPixelBuffer: pixelBuffer)
            applyMaskToImage(original: ciImageForMask, mask: maskImage)
        } else if let frozen = lockedSegmentedImage {
            // Show frozen segmented image during examination
            DispatchQueue.main.async {
                self.segmentedImage = frozen
            }
        }
        
        isProcessing = false
    }
    
    // REMOVED: All Canny edge detection methods
    // - applyCannyEdgeSegmentation
    // - cannyEdgeDetection
    // - gaussianBlur
    // - sobelGradients
    // - nonMaximumSuppression
    // - hysteresisThreshold
    // - trackWeakEdges
    // - fillEnclosedRegion
    // - dilate
    
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
              let destData = CVPixelBufferGetBaseAddress(destination) else { return nil }
        
        for row in 0..<height {
            let sourceRowData = sourceData.advanced(by: row * sourceBytesPerRow)
            let destRowData = destData.advanced(by: row * destBytesPerRow)
            memcpy(destRowData, sourceRowData, min(sourceBytesPerRow, destBytesPerRow))
        }
        return destination
    }
    
    private func analyzeAndProvideGuidance(mask: CVPixelBuffer) {
        CVPixelBufferLockBaseAddress(mask, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(mask, .readOnly) }
        
        let width = CVPixelBufferGetWidth(mask)
        let height = CVPixelBufferGetHeight(mask)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(mask)
        guard let baseAddress = CVPixelBufferGetBaseAddress(mask) else { return }
        let maskPtr = baseAddress.assumingMemoryBound(to: UInt8.self)
        
        var minX = width, maxX = 0, minY = height, maxY = 0, pixelCount = 0
        
        for y in 0..<height {
            let rowPtr = maskPtr.advanced(by: y * bytesPerRow)
            for x in 0..<width {
                if rowPtr[x] > 128 {
                    minX = min(minX, x); maxX = max(maxX, x)
                    minY = min(minY, y); maxY = max(maxY, y)
                    pixelCount += 1
                }
            }
        }
        
        guard pixelCount > 0 else {
            DispatchQueue.main.async { self.userGuidanceHint = nil }
            return
        }
        
        let segmentArea = (maxX - minX) * (maxY - minY)
        let coverageRatio = Float(segmentArea) / Float(width * height)
        let edgeMargin = 5
        let touchesEdges = minX < edgeMargin || maxX > width - edgeMargin || minY < edgeMargin || maxY > height - edgeMargin
        
        var hint: UserGuidanceHint? = nil
        if touchesEdges && coverageRatio > 0.4 {
            hint = UserGuidanceHint(message: "Step back to see full scene", icon: "arrow.down.backward.and.arrow.up.forward", color: .orange)
        } else if coverageRatio < 0.15 {
            hint = UserGuidanceHint(message: "Move closer to subject", icon: "arrow.up.left.and.arrow.down.right", color: .blue)
        } else if coverageRatio > 0.7 {
            hint = UserGuidanceHint(message: "Move back for better framing", icon: "arrow.down.backward.and.arrow.up.forward", color: .yellow)
        }
        
        DispatchQueue.main.async {
            withAnimation { self.userGuidanceHint = hint }
        }
    }
    
    private func updateU2NetVisualization(_ mask: CVPixelBuffer) {
        // Disabled for performance
    }
    
    private func runU2NetSync(pixelBuffer: CVPixelBuffer) {
        guard let model = u2netModel else { return }
        let request = VNCoreMLRequest(model: model) { [weak self] request, error in
            if let error = error {
                print("❌ [ForegroundCameraModel] U2-Net error: \(error)")
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
            print("❌ [ForegroundCameraModel] U2-Net handler error: \(error)")
        }
    }
    
    private func applyMaskToImage(original: CIImage, mask: CIImage) {
        let scaleX = original.extent.width / mask.extent.width
        let scaleY = original.extent.height / mask.extent.height
        let scaledMask = mask.transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY)).samplingNearest()
        var finalMask = scaledMask
        
        // Light blur for softer edges
        if let blurFilter = CIFilter(name: "CIGaussianBlur") {
            blurFilter.setValue(scaledMask, forKey: kCIInputImageKey)
            blurFilter.setValue(0.5, forKey: kCIInputRadiusKey)  // Slightly more blur for smoother edges
            if let blurred = blurFilter.outputImage { finalMask = blurred }
        }
        
        // Sharpen contrast for cleaner cutout
        if let colorControls = CIFilter(name: "CIColorControls") {
            colorControls.setValue(finalMask, forKey: kCIInputImageKey)
            colorControls.setValue(1.5, forKey: kCIInputContrastKey)  // Reduced from 2.0 for softer look
            if let sharpened = colorControls.outputImage { finalMask = sharpened }
        }
        
        finalMask = finalMask.cropped(to: original.extent)
        guard let blendFilter = CIFilter(name: "CIBlendWithMask") else { return }
        
        let transparent = CIImage(color: CIColor(red: 0, green: 0, blue: 0, alpha: 0)).cropped(to: original.extent)
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
}

extension ForegroundCameraModel: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        processWithU2Net(pixelBuffer: pixelBuffer)
    }
}
