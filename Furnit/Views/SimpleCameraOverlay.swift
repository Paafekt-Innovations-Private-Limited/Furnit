import SwiftUI
import AVFoundation
import CoreML
import Vision
import CoreImage

struct SimpleCameraOverlay: View {
    @Binding var capturedImage: UIImage?
    @Binding var isShowingCamera: Bool
    @StateObject private var camera = FastSAMCameraModel()
    
    @State private var scaleMultiplier: CGFloat = 0.5
    @State private var dragOffset: CGSize = .zero
    @State private var accumulatedOffset: CGSize = .zero
    @State private var isInitialAppearance = true
    @State private var scannerRotation: Double = 0
    @State private var lastHapticTime: Date = .distantPast
    
    private let showSensitivitySlider = false
    
    var body: some View {
        ZStack {
            Color.clear
                .ignoresSafeArea()
            
            // Scanning reticle when examining
            if !camera.isExamining && camera.segmentedImage == nil && !isInitialAppearance {
                ScanningReticle(rotation: $scannerRotation)
                    .onAppear {
                        withAnimation(.linear(duration: 2.0).repeatForever(autoreverses: false)) {
                            scannerRotation = 360
                        }
                    }
            }
            
            // Segmented furniture overlay
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
            
            VStack {
                HStack {
                    // Zoom slider
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
                
                Spacer()
                
                VStack(spacing: 12) {
                    // Status indicator
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
                                Text("Drag to reposition")
                                    .font(.caption2)
                                    .foregroundColor(.white.opacity(0.7))
                            }
                            .padding(8)
                            .background(Capsule().fill(Color.black.opacity(0.5)))
                            .foregroundColor(.white)
                        }
                    }
                    
                    // ACTION BUTTONS
                    HStack(spacing: 16) {
                        if camera.isExamining {
                            // Done Examining button
                            Button(action: {
                                withAnimation {
                                    camera.finishExamining()
                                }
                            }) {
                                HStack(spacing: 8) {
                                    Image(systemName: "checkmark.circle.fill")
                                        .font(.title3)
                                    Text("Done Examining")
                                        .font(.headline)
                                }
                                .foregroundColor(.white)
                                .padding(.horizontal, 24)
                                .padding(.vertical, 12)
                                .background(
                                    Capsule()
                                        .fill(Color.blue.opacity(0.9))
                                        .shadow(color: .blue.opacity(0.3), radius: 8)
                                )
                            }
                        } else if camera.segmentedImage != nil {
                            // Examine button
                            Button(action: {
                                withAnimation {
                                    camera.startExamining()
                                }
                                // Haptic feedback
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
                    
                    // Intelligent progress indicator
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

class FastSAMCameraModel: NSObject, ObservableObject {
    // 🎯 LOGIC:
    // Preview: U2-Net only (fast, no FastSAM)
    // Examine: Find main object (highest confidence), then find ALL touching objects
    // Exclude walls/floor by checking edges, size, aspect ratio, and U2-Net overlap
    // Max 4 objects total
    
    @Published var segmentedImage: UIImage?
    @Published var confidenceThreshold: Float = 0.50
    @Published var isAutoAdjusting = false
    @Published var furnitureOpacity: Double = 0.0
    @Published var detectionId: UUID = UUID()
    @Published var isExamining: Bool = false
    
    let session = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let videoQueue = DispatchQueue(label: "videoQueue", qos: .userInitiated)
    
    private var fastSAMModel: MLModel?
    private var u2netModel: VNCoreMLModel?
    private let context = CIContext()
    
    private var lastProcessTime = Date()
    private let processInterval: TimeInterval = 0.2
    private var isProcessing = false
    
    private var u2netMask: CVPixelBuffer?
    private var lockedFastSAMDetections: [Int]? = nil  // 🎯 Group of touching objects (max 4)
    private var lockedThreshold: Float? = nil
    private var shouldStartExaminingOnNextFrame = false
    
    private let frameWidth: Float = 640
    private let frameHeight: Float = 640
    
    override init() {
        super.init()
        checkCameraAuthorization()
        loadFastSAMModel()
        loadU2NetModel()
    }
    
    func startExamining() {
        print("🔍 User started EXAMINING")
        shouldStartExaminingOnNextFrame = true
    }
    
    func finishExamining() {
        print("✅ User finished EXAMINING - resetting for new scan")
        DispatchQueue.main.async {
            self.isExamining = false
        }
        lockedFastSAMDetections = nil
        lockedThreshold = nil
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
    
    private func loadFastSAMModel() {
        let modelNames = ["FastSAM-x", "FastSAM-embedded", "FastSAM", "yolov8x-seg"]
        
        for name in modelNames {
            for ext in ["mlmodelc", "mlpackage"] {
                if let url = Bundle.main.url(forResource: name, withExtension: ext) {
                    do {
                        let model = try MLModel(contentsOf: url)
                        self.fastSAMModel = model
                        print("✅ Model loaded: \(name)")
                        return
                    } catch {
                        continue
                    }
                }
            }
        }
        print("❌ No FastSAM model found")
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
            // 🎯 Start examining immediately when camera opens
            shouldStartExaminingOnNextFrame = true
        }
    }
    
    func stopSession() {
        if session.isRunning {
            session.stopRunning()
        }
    }
    
    private func processWithFastSAM(pixelBuffer: CVPixelBuffer) {
        let now = Date()
        guard now.timeIntervalSince(lastProcessTime) >= processInterval else { return }
        guard !isProcessing else { return }
        
        isProcessing = true
        lastProcessTime = now
        
        // Always run U2-Net for fresh detection every frame
        if u2netModel != nil {
            runU2NetAsync(pixelBuffer: pixelBuffer)
        }
        
        // Wait for U2-Net before using FastSAM
        guard let u2netMask = u2netMask else {
            isProcessing = false
            return
        }
        
        // 🎯 PREVIEW PHASE: Show U2-Net only (no FastSAM needed yet!)
        if !shouldStartExaminingOnNextFrame && lockedFastSAMDetections == nil {
            let maskImage = CIImage(cvPixelBuffer: u2netMask)
            let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
            applyMaskToImage(original: ciImage, mask: maskImage)
            isProcessing = false
            return
        }
        
        // 🎯 EXAMINE PHASE: Run FastSAM + touching logic
        guard let fastSAMModel = fastSAMModel else {
            // Fallback to U2-Net only
            let maskImage = CIImage(cvPixelBuffer: u2netMask)
            let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
            applyMaskToImage(original: ciImage, mask: maskImage)
            isProcessing = false
            return
        }
        
        guard let resizedBuffer = resizePixelBuffer(pixelBuffer, width: 640, height: 640) else {
            isProcessing = false
            return
        }
        
        let imageValue = MLFeatureValue(pixelBuffer: resizedBuffer)
        
        guard let provider = try? MLDictionaryFeatureProvider(dictionary: ["image": imageValue]) else {
            isProcessing = false
            return
        }
        
        do {
            let prediction = try fastSAMModel.prediction(from: provider)
            
            guard let detections = prediction.featureValue(for: "var_1550")?.multiArrayValue,
                  let prototypes = prediction.featureValue(for: "p")?.multiArrayValue else {
                // Use U2-Net only
                let maskImage = CIImage(cvPixelBuffer: u2netMask)
                let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
                applyMaskToImage(original: ciImage, mask: maskImage)
                isProcessing = false
                return
            }
            
            applyWithTouchingLogic(detections: detections, prototypes: prototypes,
                                  originalPixelBuffer: pixelBuffer)
            
        } catch {
            print("❌ FastSAM failed: \(error)")
        }
        
        isProcessing = false
    }
    
    private func runU2NetAsync(pixelBuffer: CVPixelBuffer) {
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
        
        request.imageCropAndScaleOption = .scaleFit
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
        
        do {
            try handler.perform([request])
        } catch {
            print("❌ U2-Net handler error: \(error)")
        }
    }
    
    // 🎯 NEW: Apply with touching logic - find main object, then ALL touching objects (exclude walls/floor)
    private func applyWithTouchingLogic(detections: MLMultiArray, prototypes: MLMultiArray,
                                       originalPixelBuffer: CVPixelBuffer) {
        
        let numDetections = detections.shape[2].intValue
        let numValues = detections.shape[1].intValue
        let numPrototypes = prototypes.shape[1].intValue
        let protoHeight = prototypes.shape[2].intValue
        let protoWidth = prototypes.shape[3].intValue
        
        let detPointer = detections.dataPointer.assumingMemoryBound(to: Float.self)
        let protoPointer = prototypes.dataPointer.assumingMemoryBound(to: Float.self)
        let maskSize = protoHeight * protoWidth
        
        guard let u2netMask = u2netMask else {
            print("❌ No U2-Net mask")
            return
        }
        
        // 🎯 Find touching detections (excluding walls/floor)
        var detectionsToUse: [Int]
        
        if let locked = lockedFastSAMDetections {
            detectionsToUse = locked
        } else if shouldStartExaminingOnNextFrame {
            // Find main object + ALL touching objects (walls/floor excluded)
            detectionsToUse = findTouchingObjects(
                detections: detections,
                prototypes: prototypes,
                protoPointer: protoPointer,
                maskSize: maskSize,
                protoWidth: protoWidth,
                protoHeight: protoHeight,
                numDetections: numDetections,
                numPrototypes: numPrototypes,
                numValues: numValues,
                u2netMask: u2netMask
            )
            
            lockedFastSAMDetections = detectionsToUse
            shouldStartExaminingOnNextFrame = false
            
            DispatchQueue.main.async {
                self.isExamining = true
            }
            
            print("✅ LOCKED \(detectionsToUse.count) detections (touching, walls/floor excluded)")
            for detIdx in detectionsToUse {
                let conf = detPointer[4 * numDetections + detIdx]
                print("   #\(detIdx): conf=\(String(format: "%.3f", conf))")
            }
        } else {
            // Show U2-Net preview
            let maskImage = CIImage(cvPixelBuffer: u2netMask)
            let ciImage = CIImage(cvPixelBuffer: originalPixelBuffer)
            applyMaskToImage(original: ciImage, mask: maskImage)
            return
        }
        
        guard !detectionsToUse.isEmpty else {
            let maskImage = CIImage(cvPixelBuffer: u2netMask)
            let ciImage = CIImage(cvPixelBuffer: originalPixelBuffer)
            applyMaskToImage(original: ciImage, mask: maskImage)
            return
        }
        
        // Generate combined mask from all touching detections
        var combinedFastSAMConfidence = [Float](repeating: 0, count: maskSize)
        
        for detIdx in detectionsToUse {
            var maskCoeffs = [Float](repeating: 0, count: numPrototypes)
            for j in 0..<numPrototypes {
                let coeffIdx = (numValues - numPrototypes + j) * numDetections + detIdx
                maskCoeffs[j] = detPointer[coeffIdx]
            }
            
            var detectionConfidence = [Float](repeating: 0, count: maskSize)
            for protoIdx in 0..<maskCoeffs.count {
                let coeff = maskCoeffs[protoIdx]
                let protoOffset = protoIdx * maskSize
                
                for i in 0..<maskSize {
                    detectionConfidence[i] += coeff * protoPointer[protoOffset + i]
                }
            }
            
            for i in 0..<maskSize {
                detectionConfidence[i] = 1.0 / (1.0 + exp(-detectionConfidence[i]))
            }
            
            for i in 0..<maskSize {
                combinedFastSAMConfidence[i] = max(combinedFastSAMConfidence[i], detectionConfidence[i])
            }
        }
        
        // Get U2-Net mask data
        CVPixelBufferLockBaseAddress(u2netMask, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(u2netMask, .readOnly) }
        
        let maskWidth = CVPixelBufferGetWidth(u2netMask)
        let maskHeight = CVPixelBufferGetHeight(u2netMask)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(u2netMask)
        
        guard let baseAddress = CVPixelBufferGetBaseAddress(u2netMask) else { return }
        let u2netPtr = baseAddress.assumingMemoryBound(to: UInt8.self)
        
        var pixelData = [UInt8](repeating: 0, count: maskSize)
        var combinedScores = [Float](repeating: 0, count: maskSize)
        var allScores: [Float] = []
        allScores.reserveCapacity(maskSize)
        
        for y in 0..<protoHeight {
            for x in 0..<protoWidth {
                let idx = y * protoWidth + x
                let sourceX = x * maskWidth / protoWidth
                let sourceY = y * maskHeight / protoHeight
                let sourceIdx = sourceY * bytesPerRow + sourceX
                
                let u2netScore = Float(u2netPtr[sourceIdx]) / 255.0
                let fastSAMScore = combinedFastSAMConfidence[idx]
                let combinedScore = max(u2netScore, fastSAMScore)
                
                combinedScores[idx] = combinedScore
                allScores.append(combinedScore)
            }
        }
        
        // Calculate threshold
        let threshold: Float
        if let locked = lockedThreshold {
            threshold = locked
        } else if isExamining {
            let calculatedThreshold = calculateOtsuThreshold(scores: allScores)
            lockedThreshold = calculatedThreshold
            threshold = calculatedThreshold
            print("✅ Threshold: \(String(format: "%.3f", threshold))")
        } else {
            threshold = calculateOtsuThreshold(scores: allScores)
        }
        
        DispatchQueue.main.async {
            self.confidenceThreshold = threshold
        }
        
        for y in 0..<protoHeight {
            for x in 0..<protoWidth {
                let idx = y * protoWidth + x
                if combinedScores[idx] > threshold {
                    pixelData[idx] = 255
                }
            }
        }
        
        // Apply gentle morphology
        pixelData = applyGentleMorphology(pixelData, width: protoWidth, height: protoHeight)
        
        // Convert to CIImage
        let data = Data(pixelData)
        guard let provider = CGDataProvider(data: data as CFData) else { return }
        
        let colorSpace = CGColorSpaceCreateDeviceGray()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue)
        
        guard let cgImage = CGImage(
            width: protoWidth,
            height: protoHeight,
            bitsPerComponent: 8,
            bitsPerPixel: 8,
            bytesPerRow: protoWidth,
            space: colorSpace,
            bitmapInfo: bitmapInfo,
            provider: provider,
            decode: nil,
            shouldInterpolate: false,
            intent: .defaultIntent
        ) else { return }
        
        let finalMask = CIImage(cgImage: cgImage)
        let ciImage = CIImage(cvPixelBuffer: originalPixelBuffer)
        applyMaskToImage(original: ciImage, mask: finalMask)
    }
    
    // 🎯 Find ALL touching objects (excluding walls/floor, max 4)
    private func findTouchingObjects(
        detections: MLMultiArray,
        prototypes: MLMultiArray,
        protoPointer: UnsafePointer<Float>,
        maskSize: Int,
        protoWidth: Int,
        protoHeight: Int,
        numDetections: Int,
        numPrototypes: Int,
        numValues: Int,
        u2netMask: CVPixelBuffer
    ) -> [Int] {
        
        let detPointer = detections.dataPointer.assumingMemoryBound(to: Float.self)
        
        print("\n🔍 Finding main object + ALL touching objects...")
        
        // STEP 1: Find main object (highest confidence with >20% U2-Net overlap)
        var bestConf: Float = 0
        var mainIdx: Int? = nil
        var mainBBox: (x: Float, y: Float, w: Float, h: Float)? = nil
        
        for i in 0..<numDetections {
            let conf = detPointer[4 * numDetections + i]
            
            if conf > 0.3 {
                let x = detPointer[0 * numDetections + i]
                let y = detPointer[1 * numDetections + i]
                let w = detPointer[2 * numDetections + i]
                let h = detPointer[3 * numDetections + i]
                
                let overlap = calculateU2NetOverlap(x: x, y: y, width: w, height: h)
                
                if conf > bestConf && overlap > 0.2 {
                    bestConf = conf
                    mainIdx = i
                    mainBBox = (x, y, w, h)
                }
            }
        }
        
        guard let mainDetection = mainIdx, let bbox = mainBBox else {
            print("⚠️ No main object found")
            return []
        }
        
        print("✅ Main object: #\(mainDetection) conf=\(String(format: "%.3f", bestConf))")
        print("   BBox: (\(Int(bbox.x)), \(Int(bbox.y)), \(Int(bbox.w))×\(Int(bbox.h)))")
        
        // STEP 2: Generate masks for ALL valid candidates (not limited to bounding box)
        struct Detection {
            let index: Int
            let mask: [UInt8]
            let bbox: (x: Float, y: Float, w: Float, h: Float)
            let pixelCount: Int
        }
        
        var candidates: [Detection] = []
        
        // Generate mask for main object
        var mainMask = [UInt8](repeating: 0, count: maskSize)
        var mainCoeffs = [Float](repeating: 0, count: numPrototypes)
        for j in 0..<numPrototypes {
            let coeffIdx = (numValues - numPrototypes + j) * numDetections + mainDetection
            mainCoeffs[j] = detPointer[coeffIdx]
        }
        
        var mainConf = [Float](repeating: 0, count: maskSize)
        for protoIdx in 0..<mainCoeffs.count {
            let coeff = mainCoeffs[protoIdx]
            let protoOffset = protoIdx * maskSize
            for pixelIdx in 0..<maskSize {
                mainConf[pixelIdx] += coeff * protoPointer[protoOffset + pixelIdx]
            }
        }
        var mainPixelCount = 0
        for pixelIdx in 0..<maskSize {
            let prob = 1.0 / (1.0 + exp(-mainConf[pixelIdx]))
            if prob > 0.5 {
                mainMask[pixelIdx] = 255
                mainPixelCount += 1
            }
        }
        
        candidates.append(Detection(index: mainDetection, mask: mainMask, bbox: bbox, pixelCount: mainPixelCount))
        
        // Check ALL other detections (not limited to bounding box)
        var checkedCount = 0
        for i in 0..<numDetections {
            if i == mainDetection { continue }
            if checkedCount >= 30 { break }  // Check up to 30 candidates
            
            let conf = detPointer[4 * numDetections + i]
            if conf < 0.3 { continue }
            
            let x = detPointer[0 * numDetections + i]
            let y = detPointer[1 * numDetections + i]
            let w = detPointer[2 * numDetections + i]
            let h = detPointer[3 * numDetections + i]
            
            // 🎯 EXCLUDE WALLS/FLOOR: Less aggressive filtering
            let areaRatio = (w * h) / Float(protoWidth * protoHeight)
            let aspectRatio = h / w
            let overlap = calculateU2NetOverlap(x: x, y: y, width: w, height: h)
            
            // 1. WALLS: Very tall, touches top AND bottom edges, extends vertically
            let touchesTop = y - h/2 < 50
            let touchesBottom = y + h/2 > Float(protoHeight) - 50
            let isVeryTall = aspectRatio > 4.0  // Much more extreme
            let isWall = touchesTop && touchesBottom && isVeryTall && areaRatio > 0.5
            
            // 2. FLOOR: Very wide, at bottom of frame, huge area
            let isVeryWide = aspectRatio < 0.25  // Much more extreme
            let isAtBottom = y > Float(protoHeight) * 0.7
            let isFloor = isVeryWide && isAtBottom && areaRatio > 0.7
            
            // 3. BACKGROUND: Almost no U2-Net overlap
            let isBackground = overlap < 0.03  // Even stricter overlap check
            
            // 4. TOO MASSIVE: Covers almost entire frame (>80%)
            let isMassive = areaRatio > 0.8
            
            // Skip only if clearly wall/floor/background
            if isWall || isFloor || isBackground || isMassive {
                continue
            }
            
            checkedCount += 1
            
            // Generate mask
            var mask = [UInt8](repeating: 0, count: maskSize)
            var coeffs = [Float](repeating: 0, count: numPrototypes)
            for j in 0..<numPrototypes {
                let coeffIdx = (numValues - numPrototypes + j) * numDetections + i
                coeffs[j] = detPointer[coeffIdx]
            }
            
            var detConf = [Float](repeating: 0, count: maskSize)
            for protoIdx in 0..<coeffs.count {
                let coeff = coeffs[protoIdx]
                let protoOffset = protoIdx * maskSize
                for pixelIdx in 0..<maskSize {
                    detConf[pixelIdx] += coeff * protoPointer[protoOffset + pixelIdx]
                }
            }
            
            var pixelCount = 0
            for pixelIdx in 0..<maskSize {
                let prob = 1.0 / (1.0 + exp(-detConf[pixelIdx]))
                if prob > 0.5 {
                    mask[pixelIdx] = 255
                    pixelCount += 1
                }
            }
            
            // Accept objects with reasonable size (not too tiny, not massive)
            if pixelCount > 500 && pixelCount < maskSize * 8 / 10 {
                candidates.append(Detection(index: i, mask: mask, bbox: (x, y, w, h), pixelCount: pixelCount))
            }
        }
        
        print("   Found \(candidates.count) valid candidates (checked \(checkedCount), filtered walls/floor)")
        
        // STEP 3: Find ALL touching objects (max 4 total)
        var touchingGroup: Set<Int> = [mainDetection]
        var toProcess: [Int] = [mainDetection]
        
        while !toProcess.isEmpty && touchingGroup.count < 4 {
            let currentIdx = toProcess.removeFirst()
            guard let currentDet = candidates.first(where: { $0.index == currentIdx }) else { continue }
            
            for otherDet in candidates {
                if touchingGroup.contains(otherDet.index) || touchingGroup.count >= 4 { continue }
                
                if masksTouch(currentDet.mask, otherDet.mask, width: protoWidth, height: protoHeight) {
                    touchingGroup.insert(otherDet.index)
                    toProcess.append(otherDet.index)
                    let conf = detPointer[4 * numDetections + otherDet.index]
                    print("   ↳ Touching: #\(otherDet.index) (conf=\(String(format: "%.3f", conf)), pixels=\(otherDet.pixelCount))")
                }
            }
        }
        
        print("✅ Found \(touchingGroup.count) touching objects (walls/floor excluded)")
        
        return Array(touchingGroup)
    }
    
    // Check if two masks touch (sparse sampling for performance)
    private func masksTouch(_ mask1: [UInt8], _ mask2: [UInt8], width: Int, height: Int) -> Bool {
        // Quick overlap check
        for i in stride(from: 0, to: mask1.count, by: 4) {
            if mask1[i] > 0 && mask2[i] > 0 {
                return true
            }
        }
        
        // Sparse edge touching check
        for y in stride(from: 0, to: height, by: 8) {
            for x in stride(from: 0, to: width, by: 8) {
                let idx = y * width + x
                
                if mask1[idx] > 0 {
                    for dy in -2...2 {
                        for dx in -2...2 {
                            let ny = y + dy
                            let nx = x + dx
                            
                            if ny >= 0 && ny < height && nx >= 0 && nx < width {
                                let neighborIdx = ny * width + nx
                                if mask2[neighborIdx] > 0 {
                                    return true
                                }
                            }
                        }
                    }
                }
            }
        }
        
        return false
    }
    
    private func applyGentleMorphology(_ input: [UInt8], width: Int, height: Int) -> [UInt8] {
        // Very small dilation only
        let smoothed = dilate(input, width: width, height: height, kernelSize: 3)
        return smoothed
    }
    
    private func dilate(_ input: [UInt8], width: Int, height: Int, kernelSize: Int) -> [UInt8] {
        var output = [UInt8](repeating: 0, count: input.count)
        let radius = kernelSize / 2
        
        for y in 0..<height {
            for x in 0..<width {
                var maxVal: UInt8 = 0
                
                for dy in -radius...radius {
                    for dx in -radius...radius {
                        let ny = y + dy
                        let nx = x + dx
                        
                        if ny >= 0 && ny < height && nx >= 0 && nx < width {
                            let idx = ny * width + nx
                            maxVal = max(maxVal, input[idx])
                        }
                    }
                }
                
                output[y * width + x] = maxVal
            }
        }
        
        return output
    }
    
    private func calculateOtsuThreshold(scores: [Float]) -> Float {
        let numBins = 100
        var histogram = [Int](repeating: 0, count: numBins)
        
        for score in scores {
            let binIndex = min(Int(score * Float(numBins)), numBins - 1)
            histogram[binIndex] += 1
        }
        
        let total = scores.count
        var sum: Float = 0
        for i in 0..<numBins {
            sum += Float(i) * Float(histogram[i])
        }
        
        var sumB: Float = 0
        var wB = 0
        var wF = 0
        var maxVariance: Float = 0
        var optimalThreshold: Float = 0.5
        
        for t in 0..<numBins {
            wB += histogram[t]
            if wB == 0 { continue }
            
            wF = total - wB
            if wF == 0 { break }
            
            sumB += Float(t * histogram[t])
            
            let mB = sumB / Float(wB)
            let mF = (sum - sumB) / Float(wF)
            
            let variance = Float(wB) * Float(wF) * (mB - mF) * (mB - mF)
            
            if variance > maxVariance {
                maxVariance = variance
                optimalThreshold = Float(t) / Float(numBins)
            }
        }
        
        let finalThreshold = max(0.10, min(0.55, optimalThreshold - 0.25))
        return finalThreshold
    }
    
    private func calculateU2NetOverlap(x: Float, y: Float, width: Float, height: Float) -> Float {
        guard let u2netMask = u2netMask else { return 0 }
        
        CVPixelBufferLockBaseAddress(u2netMask, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(u2netMask, .readOnly) }
        
        let maskWidth = CVPixelBufferGetWidth(u2netMask)
        let maskHeight = CVPixelBufferGetHeight(u2netMask)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(u2netMask)
        guard let baseAddress = CVPixelBufferGetBaseAddress(u2netMask) else { return 0 }
        
        let scaleX = Float(maskWidth) / frameWidth
        let scaleY = Float(maskHeight) / frameHeight
        
        let boxLeft = Int(max(0, (x - width/2) * scaleX))
        let boxRight = Int(min(Float(maskWidth-1), (x + width/2) * scaleX))
        let boxTop = Int(max(0, (y - height/2) * scaleY))
        let boxBottom = Int(min(Float(maskHeight-1), (y + height/2) * scaleY))
        
        var overlapCount = 0
        var totalCount = 0
        
        for row in boxTop...boxBottom {
            let rowPtr = baseAddress.advanced(by: row * bytesPerRow).assumingMemoryBound(to: UInt8.self)
            for col in boxLeft...boxRight {
                totalCount += 1
                if rowPtr[col] > 128 {
                    overlapCount += 1
                }
            }
        }
        
        return totalCount > 0 ? Float(overlapCount) / Float(totalCount) : 0
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
        
        let context = CIContext(options: [.useSoftwareRenderer: false])
        context.render(scaledImage, to: outputBuffer)
        
        return outputBuffer
    }
}

extension FastSAMCameraModel: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        processWithFastSAM(pixelBuffer: pixelBuffer)
    }
}
