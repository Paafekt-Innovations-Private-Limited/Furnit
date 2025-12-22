import SwiftUI
import AVFoundation
import CoreML
import Vision
import CoreImage
import Photos

struct SimpleCameraOverlay: View {
    @Binding var capturedImage: UIImage?
    @Binding var isShowingCamera: Bool
    @StateObject private var camera = U2NetCameraModel()
    
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
                    .stroke(Color.green, lineWidth: 3)
                    .frame(width: camera.u2netCoverageRect.width,
                           height: camera.u2netCoverageRect.height)
                    .position(x: camera.u2netCoverageRect.midX,
                             y: camera.u2netCoverageRect.midY)
                
                Text("U2-Net Area")
                    .font(.caption)
                    .foregroundColor(.green)
                    .background(Color.black.opacity(0.7))
                    .position(x: camera.u2netCoverageRect.midX,
                             y: camera.u2netCoverageRect.minY - 20)
            }
            
            if !camera.isExamining && camera.segmentedImage == nil && !isInitialAppearance {
                ScanningReticle(rotation: $scannerRotation)
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
            
            // ✅ Success message overlay
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
                                    Text("Scanning surroundings...")
                                        .font(.caption)
                                }
                                .padding(8)
                                .background(Capsule().fill(Color.black.opacity(0.5)))
                                .foregroundColor(.white)
                            } else {
                                Text("Tap to examine furniture")
                                    .font(.caption)
                                    .padding(8)
                                    .background(Capsule().fill(Color.black.opacity(0.5)))
                                    .foregroundColor(.white)
                            }
                        } else {
                            VStack(spacing: 4) {
                                Text("Boundary set • Size: \(Int(scaleMultiplier * 100))%")
                                    .font(.caption)
                                Text("Drag to reposition • Ready to save")
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
                            // 💾 NEW: Save Furniture button
                            Button(action: {
                                saveFurniture()
                            }) {
                                HStack(spacing: 8) {
                                    Image(systemName: "square.and.arrow.down.fill")
                                        .font(.title3)
                                    Text("Save Furniture")
                                        .font(.headline)
                                }
                                .foregroundColor(.white)
                                .padding(.horizontal, 24)
                                .padding(.vertical, 12)
                                .background(
                                    Capsule()
                                        .fill(Color.green.opacity(0.9))
                                        .shadow(color: .green.opacity(0.3), radius: 8)
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
                                    Image(systemName: "viewfinder.circle.fill")
                                        .font(.title3)
                                    Text("Examine")
                                        .font(.headline)
                                }
                                .foregroundColor(.white)
                                .padding(.horizontal, 32)
                                .padding(.vertical, 12)
                                .background(
                                    Capsule()
                                        .fill(Color.green.opacity(0.9))
                                        .shadow(color: .green.opacity(0.3), radius: 8)
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
            camera.stopSession()
        }
    }
    
    // 💾 Save furniture function
    private func saveFurniture() {
        guard let image = camera.segmentedImage else { return }
        
        // Set the binding for parent view immediately
        capturedImage = image
        
        // Request Photos authorization
        PHPhotoLibrary.requestAuthorization(for: .addOnly) { status in
            DispatchQueue.main.async {
                if status == .authorized || status == .limited {
                    // Save to Photos library using modern API
                    PHPhotoLibrary.shared().performChanges({
                        PHAssetChangeRequest.creationRequestForAsset(from: image)
                    }) { success, error in
                        DispatchQueue.main.async {
                            if success {
                                logDebug("✅ Furniture saved to Photos!")
                                
                                // Show success message
                                withAnimation(.spring()) {
                                    self.showingSaveSuccess = true
                                }
                                
                                // Haptic feedback
                                let generator = UINotificationFeedbackGenerator()
                                generator.notificationOccurred(.success)
                                
                                // Hide success message and close camera after delay
                                DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) {
                                    withAnimation {
                                        self.showingSaveSuccess = false
                                    }
                                    
                                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                                        self.isShowingCamera = false
                                    }
                                }
                            } else {
                                logDebug("❌ Failed to save: \(error?.localizedDescription ?? "unknown error")")
                                // Still close camera even if save failed
                                DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                                    self.isShowingCamera = false
                                }
                            }
                        }
                    }
                } else {
                    logDebug("⚠️ Photos access denied")
                    // Still close camera even if permission denied
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                        self.isShowingCamera = false
                    }
                }
            }
        }
    }
}

struct ScanningReticle: View {
    @Binding var rotation: Double
    
    var body: some View {
        ZStack {
            Circle()
                .stroke(Color.white.opacity(0.3), lineWidth: 2)
                .frame(width: 120, height: 120)
            Rectangle()
                .fill(LinearGradient(gradient: Gradient(colors: [.clear, .white.opacity(0.8), .clear]), startPoint: .top, endPoint: .bottom))
                .frame(width: 2, height: 60)
                .offset(y: -30)
                .rotationEffect(.degrees(rotation))
            Circle()
                .fill(Color.white.opacity(0.5))
                .frame(width: 8, height: 8)
            ForEach(0..<4) { i in
                CornerBracket()
                    .rotationEffect(.degrees(Double(i) * 90))
            }
        }
    }
}

struct CornerBracket: View {
    var body: some View {
        Path { path in
            path.move(to: CGPoint(x: -60, y: -60))
            path.addLine(to: CGPoint(x: -40, y: -60))
            path.move(to: CGPoint(x: -60, y: -60))
            path.addLine(to: CGPoint(x: -60, y: -40))
        }
        .stroke(Color.white.opacity(0.6), lineWidth: 2)
    }
}

struct UserGuidanceHint {
    let message: String
    let icon: String
    let color: Color
}

class U2NetCameraModel: NSObject, ObservableObject {
    @Published var segmentedImage: UIImage?
    @Published var furnitureOpacity: Double = 0.0
    @Published var detectionId: UUID = UUID()
    @Published var isExamining: Bool = false
    @Published var userGuidanceHint: UserGuidanceHint? = nil
    @Published var u2netCoverageRect: CGRect = .zero
    
    let session = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let videoQueue = DispatchQueue(label: "videoQueue", qos: .userInitiated)
    
    private var u2netModel: VNCoreMLModel?
    private let context = CIContext()
    
    private var lastProcessTime = Date()
    private let processInterval: TimeInterval = 0.2
    private var isProcessing = false
    
    private var u2netMask: CVPixelBuffer?
    private var lockedMask: CVPixelBuffer? = nil
    private var shouldStartExaminingOnNextFrame = false
    
    override init() {
        super.init()
        checkCameraAuthorization()
        loadU2NetModel()
    }
    
    func startExamining() {
        logDebug("🔍 User started EXAMINING")
        shouldStartExaminingOnNextFrame = true
    }
    
    func finishExamining() {
        logDebug("✅ User finished EXAMINING")
        DispatchQueue.main.async {
            self.isExamining = false
            self.userGuidanceHint = nil
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
                        logDebug("✅ U2-Net loaded: \(name)")
                        return
                    } catch {
                        logDebug("⚠️ Failed to load \(name).\(ext): \(error)")
                    }
                }
            }
        }
        logDebug("⚠️ No U2-Net model loaded")
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
            logDebug("❌ Camera setup failed")
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
        
        if u2netModel != nil {
            runU2NetSync(pixelBuffer: pixelBuffer)
        }
        
        guard let u2netMask = u2netMask else {
            isProcessing = false
            return
        }
        
        var maskToUse = u2netMask
        
        if shouldStartExaminingOnNextFrame {
            logDebug("📋 Creating COPY of U2-Net mask...")
            
            if let copiedMask = copyPixelBuffer(u2netMask) {
                lockedMask = copiedMask
                logDebug("✅ Locked COPY of mask (won't be overwritten)")
            } else {
                logDebug("⚠️ Failed to copy mask, using reference")
                lockedMask = u2netMask
            }
            
            shouldStartExaminingOnNextFrame = false
            
            DispatchQueue.main.async {
                self.isExamining = true
            }
        } else if let locked = lockedMask {
            maskToUse = locked
        } else {
            analyzeAndProvideGuidance(mask: u2netMask)
        }
        
        updateU2NetVisualization(maskToUse)
        
        let maskImage = CIImage(cvPixelBuffer: maskToUse)
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        applyMaskToImage(original: ciImage, mask: maskImage)
        
        isProcessing = false
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
        
        guard status == kCVReturnSuccess, let destination = copy else {
            return nil
        }
        
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
    
    private func analyzeAndProvideGuidance(mask: CVPixelBuffer) {
        CVPixelBufferLockBaseAddress(mask, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(mask, .readOnly) }
        
        let width = CVPixelBufferGetWidth(mask)
        let height = CVPixelBufferGetHeight(mask)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(mask)
        
        guard let baseAddress = CVPixelBufferGetBaseAddress(mask) else { return }
        let maskPtr = baseAddress.assumingMemoryBound(to: UInt8.self)
        
        var minX = width, maxX = 0, minY = height, maxY = 0
        var pixelCount = 0
        
        for y in 0..<height {
            let rowPtr = maskPtr.advanced(by: y * bytesPerRow)
            for x in 0..<width {
                if rowPtr[x] > 128 {
                    minX = min(minX, x)
                    maxX = max(maxX, x)
                    minY = min(minY, y)
                    maxY = max(maxY, y)
                    pixelCount += 1
                }
            }
        }
        
        guard pixelCount > 0 else {
            DispatchQueue.main.async {
                self.userGuidanceHint = nil
            }
            return
        }
        
        let segmentWidth = maxX - minX
        let segmentHeight = maxY - minY
        let segmentArea = segmentWidth * segmentHeight
        let frameArea = width * height
        let coverageRatio = Float(segmentArea) / Float(frameArea)
        
        let edgeMargin = 5
        let touchesLeft = minX < edgeMargin
        let touchesRight = maxX > width - edgeMargin
        let touchesTop = minY < edgeMargin
        let touchesBottom = maxY > height - edgeMargin
        let touchesEdges = touchesLeft || touchesRight || touchesTop || touchesBottom
        
        var hint: UserGuidanceHint? = nil
        
        if touchesEdges && coverageRatio > 0.4 {
            hint = UserGuidanceHint(
                message: "Step back to see all edges",
                icon: "arrow.down.backward.and.arrow.up.forward",
                color: .orange
            )
        } else if coverageRatio < 0.15 {
            hint = UserGuidanceHint(
                message: "Move closer to furniture",
                icon: "arrow.up.left.and.arrow.down.right",
                color: .blue
            )
        } else if coverageRatio > 0.7 {
            hint = UserGuidanceHint(
                message: "Move back for better view",
                icon: "arrow.down.backward.and.arrow.up.forward",
                color: .yellow
            )
        }
        
        DispatchQueue.main.async {
            withAnimation {
                self.userGuidanceHint = hint
            }
        }
    }
    
    private func updateU2NetVisualization(_ mask: CVPixelBuffer) {
        CVPixelBufferLockBaseAddress(mask, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(mask, .readOnly) }
        
        let maskWidth = CVPixelBufferGetWidth(mask)
        let maskHeight = CVPixelBufferGetHeight(mask)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(mask)
        guard let baseAddress = CVPixelBufferGetBaseAddress(mask) else { return }
        let maskPtr = baseAddress.assumingMemoryBound(to: UInt8.self)
        
        var minX = maskWidth, maxX = 0
        var minY = maskHeight, maxY = 0
        var hasPixels = false
        
        for y in 0..<maskHeight {
            let rowPtr = maskPtr.advanced(by: y * bytesPerRow)
            for x in 0..<maskWidth {
                if rowPtr[x] > 128 {
                    hasPixels = true
                    minX = min(minX, x)
                    maxX = max(maxX, x)
                    minY = min(minY, y)
                    maxY = max(maxY, y)
                }
            }
        }
        
        if hasPixels {
            let screenWidth = UIScreen.main.bounds.width
            let screenHeight = UIScreen.main.bounds.height
            
            let scaleX = screenWidth / CGFloat(maskWidth)
            let scaleY = screenHeight / CGFloat(maskHeight)
            
            let rect = CGRect(
                x: CGFloat(minX) * scaleX,
                y: CGFloat(minY) * scaleY,
                width: CGFloat(maxX - minX) * scaleX,
                height: CGFloat(maxY - minY) * scaleY
            )
            
            DispatchQueue.main.async {
                self.u2netCoverageRect = rect
            }
            
            logDebug("🔍 U2-Net coverage: \(minX),\(minY) → \(maxX),\(maxY) (\(maskWidth)x\(maskHeight))")
        }
    }
    
    private func runU2NetSync(pixelBuffer: CVPixelBuffer) {
        guard let model = u2netModel else { return }
        
        let request = VNCoreMLRequest(model: model) { [weak self] request, error in
            if let error = error {
                logDebug("❌ U2-Net error: \(error)")
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
            logDebug("❌ U2-Net handler error: \(error)")
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
}

extension U2NetCameraModel: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        processWithU2Net(pixelBuffer: pixelBuffer)
    }
}
