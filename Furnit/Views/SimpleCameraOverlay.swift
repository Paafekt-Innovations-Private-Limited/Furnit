import SwiftUI
import AVFoundation
import CoreML
import Vision
import CoreImage

struct SimpleCameraOverlay: View {
    @Binding var capturedImage: UIImage?
    @Binding var isShowingCamera: Bool
    @StateObject private var camera = FastSAMCameraModel()
    
    // START AT 50% SCALE INSTEAD OF 30%
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
            
            if camera.segmentedImage == nil && !isInitialAppearance {
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
            
            VStack {
                HStack {
                    // SIZE SLIDER - More prominent with background
                    HStack(spacing: 6) {
                        Image(systemName: "minus.magnifyingglass")
                            .foregroundColor(.white)
                            .font(.system(size: 14))
                        Slider(value: $scaleMultiplier, in: 0.3...1.0)
                            .frame(width: 150)  // Increased from 120
                            .accentColor(.white)
                        Image(systemName: "plus.magnifyingglass")
                            .foregroundColor(.white)
                            .font(.system(size: 14))
                    }
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                    .background(Capsule().fill(Color.black.opacity(0.7)))  // Darker background
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
                .padding(.top, 60)  // Increased from 50 to avoid status bar
                
                Spacer()
                
                VStack(spacing: 12) {
                    HStack {
                        if camera.segmentedImage == nil {
                            HStack(spacing: 4) {
                                ProgressView()
                                    .progressViewStyle(CircularProgressViewStyle(tint: .white))
                                    .scaleEffect(0.7)
                                Text("Point at furniture")
                                    .font(.caption)
                            }
                            .padding(8)
                            .background(Capsule().fill(Color.black.opacity(0.5)))
                            .foregroundColor(.white)
                        } else {
                            HStack(spacing: 8) {
                                Text("Size: \(Int(scaleMultiplier * 100))%")
                                    .font(.caption)
                                    .padding(8)
                                    .background(Capsule().fill(Color.black.opacity(0.5)))
                                    .foregroundColor(.white)
                                
                                if camera.isAutoAdjusting {
                                    Text("Auto-tuning...")
                                        .font(.caption2)
                                        .padding(6)
                                        .background(Capsule().fill(Color.green.opacity(0.3)))
                                        .foregroundColor(.green)
                                }
                            }
                        }
                    }
                    
                    if showSensitivitySlider {
                        HStack {
                            Text("Sensitivity:")
                                .font(.caption)
                                .foregroundColor(.white)
                            Slider(value: $camera.confidenceThreshold, in: 0.3...0.7)
                                .frame(width: 150)
                                .accentColor(.green)
                                .disabled(camera.isAutoAdjusting)
                            Text("\(Int(camera.confidenceThreshold * 100))%")
                                .font(.caption)
                                .foregroundColor(.white)
                                .frame(width: 40)
                        }
                        .padding(.horizontal, 12)
                        .padding(.vertical, 8)
                        .background(Capsule().fill(Color.black.opacity(0.5)))
                        .opacity(camera.isAutoAdjusting ? 0.6 : 1.0)
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
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                        withAnimation(.easeOut(duration: 0.4)) {
                            isInitialAppearance = false
                        }
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
    @Published var segmentedImage: UIImage?
    @Published var confidenceThreshold: Float = 0.5
    @Published var isAutoAdjusting = false
    @Published var furnitureOpacity: Double = 0.0
    @Published var detectionId: UUID = UUID()
    
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
    private let u2netWeight: Float = 0.3
    private let u2netQueue = DispatchQueue(label: "u2netQueue", qos: .userInitiated)
    
    private let frameWidth: Float = 640
    private let frameHeight: Float = 640
    private let frameCenterX: Float = 320
    private let frameCenterY: Float = 320
    private let frameArea: Float = 640 * 640
    
    // STABILITY MECHANISMS
    private var currentDetectionBox: (x: Float, y: Float, w: Float, h: Float)?
    private var detectionLockTime: Date?
    private let lockDuration: TimeInterval = 3.0
    private var consecutiveFramesWithSameDetection = 0
    private let requiredConsecutiveFrames = 2
    
    private let minimumAcceptableScore: Float = 0.25
    
    private var detectionHistory: [(x: Float, y: Float, w: Float, h: Float, score: Float)] = []
    private let historySize = 5
    
    private var stableDetectionTime: Date?
    private let stabilityThreshold: TimeInterval = 2.0
    private var noDetectionFrames = 0
    private let autoAdjustTrigger = 15
    
    override init() {
        super.init()
        checkCameraAuthorization()
        loadFastSAMModel()
        loadU2NetModel()
        setupMemoryWarningHandler()
    }
    
    private func setupMemoryWarningHandler() {
        NotificationCenter.default.addObserver(
            forName: UIApplication.didReceiveMemoryWarningNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            self?.handleMemoryWarning()
        }
    }
    
    private func handleMemoryWarning() {
        u2netMask = nil
        detectionHistory.removeAll()
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
        print("⚠️ No U2-Net model loaded - will use FastSAM only")
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
    
    private func processWithFastSAM(pixelBuffer: CVPixelBuffer) {
        let now = Date()
        guard now.timeIntervalSince(lastProcessTime) >= processInterval else { return }
        guard !isProcessing else { return }
        
        isProcessing = true
        lastProcessTime = now
        
        guard let model = fastSAMModel else {
            isProcessing = false
            return
        }
        
        if u2netModel != nil {
            runU2NetForGuidanceAsync(pixelBuffer: pixelBuffer)
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
            let prediction = try model.prediction(from: provider)
            
            guard let detections = prediction.featureValue(for: "var_1550")?.multiArrayValue,
                  let prototypes = prediction.featureValue(for: "p")?.multiArrayValue else {
                handleNoDetection()
                isProcessing = false
                return
            }
            
            processWithStableScoring(detections: detections, prototypes: prototypes, originalPixelBuffer: pixelBuffer)
            
        } catch {
            print("❌ Prediction failed")
        }
        
        isProcessing = false
    }
    
    private func runU2NetForGuidanceAsync(pixelBuffer: CVPixelBuffer) {
        guard let model = u2netModel else { return }
        
        // RUN U2-NET ON EVERY FRAME - NO SKIPPING
        print("🔄 Processing furniture...")
        
        u2netQueue.async { [weak self] in
            guard let self = self else { return }
            
            let request = VNCoreMLRequest(model: model) { [weak self] request, error in
                if let error = error {
                    print("❌ Processing error: \(error)")
                    return
                }
                
                if let results = request.results as? [VNPixelBufferObservation],
                   let maskBuffer = results.first?.pixelBuffer {
                    self?.u2netMask = maskBuffer
                    print("✅ Furniture detected")
                } else {
                    print("⚠️ No furniture detected")
                }
            }
            
            request.imageCropAndScaleOption = .scaleFit
            let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
            
            do {
                try handler.perform([request])
            } catch {
                print("❌ Processing error: \(error)")
            }
        }
    }
    
    private func processWithStableScoring(detections: MLMultiArray, prototypes: MLMultiArray, originalPixelBuffer: CVPixelBuffer) {
        let numDetections = detections.shape[2].intValue
        let numValues = detections.shape[1].intValue
        let numPrototypes = prototypes.shape[1].intValue
        let protoHeight = prototypes.shape[2].intValue
        let protoWidth = prototypes.shape[3].intValue
        
        let detPointer = detections.dataPointer.assumingMemoryBound(to: Float.self)
        
        let now = Date()
        let isLocked = currentDetectionBox != nil &&
                       detectionLockTime != nil &&
                       now.timeIntervalSince(detectionLockTime!) < lockDuration
        
        var bestDetection: (idx: Int, score: Float, conf: Float, x: Float, y: Float, w: Float, h: Float)? = nil
        var bestScore: Float = 0
        
        print("\n📊 === DETECTION SCORING ===")
        
        for i in 0..<numDetections {
            let conf = detPointer[4 * numDetections + i]
            
            if conf > confidenceThreshold {
                let x = detPointer[0 * numDetections + i]
                let y = detPointer[1 * numDetections + i]
                let w = detPointer[2 * numDetections + i]
                let h = detPointer[3 * numDetections + i]
                let area = w * h
                let areaRatio = area / frameArea
                
                if areaRatio < 0.08 || areaRatio > 0.70 {
                    continue
                }
                
                var sizeScore: Float = 0
                
                if areaRatio >= 0.25 && areaRatio <= 0.65 {
                    let normalizedSize = (areaRatio - 0.25) / (0.65 - 0.25)
                    sizeScore = 0.80 + (normalizedSize * 0.20)
                } else if areaRatio >= 0.08 && areaRatio < 0.25 {
                    let normalizedSize = (areaRatio - 0.08) / (0.25 - 0.08)
                    sizeScore = normalizedSize * normalizedSize * normalizedSize * 0.7
                } else {
                    sizeScore = max(0, 1.0 - (areaRatio - 0.65) * 5.0)
                }
                
                let distanceFromCenter = sqrt(pow(x - frameCenterX, 2) + pow(y - frameCenterY, 2))
                let maxDistance = sqrt(pow(frameCenterX, 2) + pow(frameCenterY, 2))
                let centerProximityScore = 1.0 - (distanceFromCenter / maxDistance)
                
                var u2netBonus: Float = 0
                if u2netMask != nil {
                    let overlapRatio = calculateU2NetOverlap(x: x, y: y, width: w, height: h)
                    u2netBonus = overlapRatio * 0.2
                }
                
                var stabilityBonus: Float = 0
                if let currentBox = currentDetectionBox, isLocked {
                    let dx = abs(x - currentBox.x)
                    let dy = abs(y - currentBox.y)
                    let distance = sqrt(dx*dx + dy*dy)
                    
                    if distance < 150 {
                        stabilityBonus = 0.6
                    }
                }
                
                let baseScore = sizeScore * sizeScore * conf * centerProximityScore
                let combinedScore = baseScore + u2netBonus + stabilityBonus
                
                if combinedScore < minimumAcceptableScore {
                    continue
                }
                
                if combinedScore > bestScore {
                    bestScore = combinedScore
                    bestDetection = (i, combinedScore, conf, x, y, w, h)
                }
            }
        }
        
        guard let detection = bestDetection else {
            print("❌ No valid detection found")
            handleNoDetection()
            return
        }
        
        let areaRatio = (detection.w * detection.h) / frameArea
        print("📍 Best detection: Area=\(String(format: "%.1f%%", areaRatio*100)), Score=\(String(format: "%.2f", detection.score)), Conf=\(String(format: "%.2f", detection.conf))")
        
        let isSimilarToCurrent: Bool
        if let currentBox = currentDetectionBox {
            let dx = abs(detection.x - currentBox.x)
            let dy = abs(detection.y - currentBox.y)
            let distance = sqrt(dx*dx + dy*dy)
            isSimilarToCurrent = distance < 150
            print("📏 Distance from locked: \(String(format: "%.0f", distance))px")
        } else {
            isSimilarToCurrent = false
        }
        
        if isSimilarToCurrent {
            consecutiveFramesWithSameDetection += 1
        } else {
            consecutiveFramesWithSameDetection = 1
            currentDetectionBox = (detection.x, detection.y, detection.w, detection.h)
        }
        
        if consecutiveFramesWithSameDetection >= requiredConsecutiveFrames || isLocked {
            
            let isNewDetection = !isSimilarToCurrent && !isLocked
            
            if isNewDetection {
                detectionLockTime = now
                consecutiveFramesWithSameDetection = 0
                
                DispatchQueue.main.async {
                    self.detectionId = UUID()
                }
                
                print("🔒 LOCKED: New detection at \(String(format: "%.1f%%", areaRatio*100))")
            }
            
            noDetectionFrames = 0
            
            var maskCoeffs = [Float](repeating: 0, count: numPrototypes)
            for i in 0..<numPrototypes {
                let coeffIdx = (numValues - numPrototypes + i) * numDetections + detection.idx
                maskCoeffs[i] = detPointer[coeffIdx]
            }
            
            let segMask = generateSmartMask(prototypes: prototypes, coefficients: maskCoeffs,
                                           protoHeight: protoHeight, protoWidth: protoWidth)
            
            let ciImage = CIImage(cvPixelBuffer: originalPixelBuffer)
            applyCleanSegmentation(original: ciImage, mask: segMask)
        } else {
            print("⏳ Waiting: \(consecutiveFramesWithSameDetection)/\(requiredConsecutiveFrames) frames")
        }
    }
    
    private func handleNoDetection() {
        noDetectionFrames += 1
        
        if noDetectionFrames > 5 {
            currentDetectionBox = nil
            detectionLockTime = nil
            consecutiveFramesWithSameDetection = 0
        }
        
        DispatchQueue.main.async {
            withAnimation(.easeOut(duration: 0.5)) {
                self.furnitureOpacity = 0.0
            }
        }
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
    
    // U2NET MAIN + FASTSAM COMPLETES SAME OBJECT
    private func generateSmartMask(prototypes: MLMultiArray, coefficients: [Float],
                                   protoHeight: Int, protoWidth: Int) -> CIImage {
        let protoPointer = prototypes.dataPointer.assumingMemoryBound(to: Float.self)
        let maskSize = protoHeight * protoWidth
        
        var finalMask = [Float](repeating: 0, count: maskSize)
        
        // Generate raw FastSAM confidence values
        for protoIdx in 0..<coefficients.count {
            let coeff = coefficients[protoIdx]
            let protoOffset = protoIdx * maskSize
            
            for i in 0..<maskSize {
                finalMask[i] += coeff * protoPointer[protoOffset + i]
            }
        }
        
        // Store FastSAM confidence values
        var fastSAMConfidence = [Float](repeating: 0, count: maskSize)
        for i in 0..<maskSize {
            fastSAMConfidence[i] = 1.0 / (1.0 + exp(-finalMask[i]))
        }
        
        var pixelData = [UInt8](repeating: 0, count: maskSize)
        
        print("\n🎯 === FURNITURE DETECTION ===")
        
        if let u2netMask = u2netMask {
            CVPixelBufferLockBaseAddress(u2netMask, .readOnly)
            defer { CVPixelBufferUnlockBaseAddress(u2netMask, .readOnly) }
            
            let maskWidth = CVPixelBufferGetWidth(u2netMask)
            let maskHeight = CVPixelBufferGetHeight(u2netMask)
            let bytesPerRow = CVPixelBufferGetBytesPerRow(u2netMask)
            
            if let baseAddress = CVPixelBufferGetBaseAddress(u2netMask) {
                let u2netPtr = baseAddress.assumingMemoryBound(to: UInt8.self)
                var u2netPixels = 0
                
                // STEP 1: Get U2-Net mask - THIS IS THE MAIN FURNITURE BODY
                for y in 0..<protoHeight {
                    for x in 0..<protoWidth {
                        let sourceX = x * maskWidth / protoWidth
                        let sourceY = y * maskHeight / protoHeight
                        let sourceIdx = sourceY * bytesPerRow + sourceX
                        let targetIdx = y * protoWidth + x
                        
                        // Use threshold of 100 for main body
                        if u2netPtr[sourceIdx] > 100 {
                            pixelData[targetIdx] = 255
                            u2netPixels += 1
                        }
                    }
                }
                
                print("📊 Main body detected: \(u2netPixels) pixels")
                
                // STEP 2: Find what object FastSAM sees that overlaps with U2-Net
                // If FastSAM identifies the same furniture, add ALL its parts
                
                // First, check if FastSAM sees an object that overlaps with U2-Net
                var overlapCount = 0
                var fastSAMHighConfidenceTotal = 0
                
                for i in 0..<maskSize {
                    if fastSAMConfidence[i] > 0.65 {
                        fastSAMHighConfidenceTotal += 1
                        if pixelData[i] == 255 {
                            overlapCount += 1
                        }
                    }
                }
                
                // If there's significant overlap, FastSAM is seeing the same object
                let overlapRatio = fastSAMHighConfidenceTotal > 0 ?
                    Float(overlapCount) / Float(fastSAMHighConfidenceTotal) : 0
                
                print("📊 FastSAM overlap with furniture: \(Int(overlapRatio * 100))%")
                
                // STEP 3: If FastSAM identified the same object (>20% overlap),
                // let it add missing parts like chair stands/wheels
                var addedMissingParts = 0
                
                if overlapRatio > 0.2 {
                    // FastSAM sees the same furniture - add its missing parts
                    for y in 1..<(protoHeight-1) {
                        for x in 1..<(protoWidth-1) {
                            let idx = y * protoWidth + x
                            
                            // Skip if already marked by U2-Net
                            if pixelData[idx] == 255 { continue }
                            
                            // Add if FastSAM is confident this is part of the furniture
                            if fastSAMConfidence[idx] > 0.7 {
                                // Verify it's part of a coherent structure (not random noise)
                                var fastSAMNeighbors = 0
                                for dy in -1...1 {
                                    for dx in -1...1 {
                                        if dy == 0 && dx == 0 { continue }
                                        let nIdx = (y + dy) * protoWidth + (x + dx)
                                        if nIdx >= 0 && nIdx < maskSize {
                                            // Count both U2-Net and FastSAM neighbors
                                            if pixelData[nIdx] == 255 || fastSAMConfidence[nIdx] > 0.65 {
                                                fastSAMNeighbors += 1
                                            }
                                        }
                                    }
                                }
                                
                                // Add if it's part of a structure (not isolated)
                                if fastSAMNeighbors >= 3 {
                                    pixelData[idx] = 255
                                    addedMissingParts += 1
                                }
                            }
                        }
                    }
                    
                    print("✅ Missing parts added: +\(addedMissingParts) pixels")
                } else {
                    print("⚠️ FastSAM sees different object - skipping")
                }
                
                // STEP 4: Clean up isolated noise
                var noiseRemoved = 0
                for y in 1..<(protoHeight-1) {
                    for x in 1..<(protoWidth-1) {
                        let idx = y * protoWidth + x
                        
                        if pixelData[idx] == 255 {
                            var neighbors = 0
                            for dy in -1...1 {
                                for dx in -1...1 {
                                    if dy == 0 && dx == 0 { continue }
                                    let nIdx = (y + dy) * protoWidth + (x + dx)
                                    if pixelData[nIdx] == 255 {
                                        neighbors += 1
                                    }
                                }
                            }
                            
                            // Remove if completely isolated
                            if neighbors <= 1 {
                                pixelData[idx] = 0
                                noiseRemoved += 1
                            }
                        }
                    }
                }
                
                if noiseRemoved > 0 {
                    print("📊 Noise removed: -\(noiseRemoved) pixels")
                }
                
                print("📊 Final result: \(pixelData.filter { $0 == 255 }.count) pixels")
            }
        } else {
            // NO U2-NET: Fall back to FastSAM only
            print("⚠️ Loading furniture detector...")
            
            for i in 0..<maskSize {
                if fastSAMConfidence[i] > 0.6 {
                    pixelData[i] = 255
                }
            }
            
            // Clean up isolated pixels
            pixelData = removeIsolatedPixels(pixelData, width: protoWidth, height: protoHeight)
            
            print("📊 Temporary detection: \(pixelData.filter { $0 == 255 }.count) pixels")
        }
        
        let data = Data(pixelData)
        guard let provider = CGDataProvider(data: data as CFData) else {
            return CIImage(color: .white).cropped(to: CGRect(x: 0, y: 0, width: protoWidth, height: protoHeight))
        }
        
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
        ) else {
            return CIImage(color: .white).cropped(to: CGRect(x: 0, y: 0, width: protoWidth, height: protoHeight))
        }
        
        return CIImage(cgImage: cgImage)
    }
    
    private func removeIsolatedPixels(_ input: [UInt8], width: Int, height: Int) -> [UInt8] {
        var result = input
        
        // Remove completely isolated pixels
        for y in 1..<(height-1) {
            for x in 1..<(width-1) {
                let idx = y * width + x
                if input[idx] == 255 {
                    var neighbors = 0
                    
                    // Check 8 neighbors
                    for dy in -1...1 {
                        for dx in -1...1 {
                            if dx == 0 && dy == 0 { continue }
                            let nIdx = (y + dy) * width + (x + dx)
                            if input[nIdx] == 255 {
                                neighbors += 1
                            }
                        }
                    }
                    
                    // Remove if has fewer than 2 neighbors
                    if neighbors < 2 {
                        result[idx] = 0
                    }
                }
            }
        }
        
        return result
    }
    
    private func applyLightMorphology(_ input: [UInt8], width: Int, height: Int) -> [UInt8] {
        // This function is no longer used but kept for compatibility
        return removeIsolatedPixels(input, width: width, height: height)
    }
    
    private func applyCleanSegmentation(original: CIImage, mask: CIImage) {
        let scaleX = original.extent.width / mask.extent.width
        let scaleY = original.extent.height / mask.extent.height
        
        let scaledMask = mask
            .transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))
            .samplingNearest()
        
        var finalMask = scaledMask
        
        // Very light blur for smoother edges (reduced from 1.0 to 0.5)
        if let blurFilter = CIFilter(name: "CIGaussianBlur") {
            blurFilter.setValue(scaledMask, forKey: kCIInputImageKey)
            blurFilter.setValue(0.5, forKey: kCIInputRadiusKey)  // Reduced blur for crisper edges
            if let blurred = blurFilter.outputImage {
                finalMask = blurred
            }
        }
        
        finalMask = finalMask.cropped(to: original.extent)
        
        guard let blendFilter = CIFilter(name: "CIBlendWithMask") else {
            return
        }
        
        let transparent = CIImage(color: CIColor(red: 0, green: 0, blue: 0, alpha: 0))
            .cropped(to: original.extent)
        
        blendFilter.setValue(original, forKey: kCIInputImageKey)
        blendFilter.setValue(transparent, forKey: kCIInputBackgroundImageKey)
        blendFilter.setValue(finalMask, forKey: kCIInputMaskImageKey)
        
        guard let result = blendFilter.outputImage else {
            return
        }
        
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
