import SwiftUI
import AVFoundation
import CoreML
import Vision
import CoreImage

struct SimpleCameraOverlay: View {
    @Binding var capturedImage: UIImage?
    @Binding var isShowingCamera: Bool
    @StateObject private var camera = FastSAMCameraModel()
    
    @State private var scaleMultiplier: CGFloat = 0.3
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
                    HStack {
                        Image(systemName: "minus.magnifyingglass")
                            .foregroundColor(.white)
                        Slider(value: $scaleMultiplier, in: 0.3...1.0)
                            .frame(width: 120)
                            .accentColor(.white)
                        Image(systemName: "plus.magnifyingglass")
                            .foregroundColor(.white)
                    }
                    .padding(8)
                    .background(Capsule().fill(Color.black.opacity(0.5)))
                    .padding(.leading)
                    
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
                    .padding()
                }
                .padding(.top, 50)
                
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
    private var u2netUpdateCounter = 0
    private let u2netUpdateInterval = 5
    private let u2netQueue = DispatchQueue(label: "u2netQueue", qos: .userInitiated)
    
    private let frameWidth: Float = 640
    private let frameHeight: Float = 640
    private let frameCenterX: Float = 320
    private let frameCenterY: Float = 320
    private let frameArea: Float = 640 * 640
    
    // MUCH STRONGER STABILITY
    private var currentDetectionBox: (x: Float, y: Float, w: Float, h: Float)?
    private var detectionLockTime: Date?
    private let lockDuration: TimeInterval = 3.0  // Increased from 1.5 to 3.0 seconds!
    private var consecutiveFramesWithSameDetection = 0
    private let requiredConsecutiveFrames = 2  // Must see same detection 2 times before locking (reduced from 3)
    
    // MINIMUM SCORE THRESHOLD
    private let minimumAcceptableScore: Float = 0.25  // NEW: Reject anything below this
    
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
    }
    
    private func loadU2NetModel() {
        let modelNames = ["u2netp", "U2Net", "u2net", "U2NetP", "U2NET"]
        
        for name in modelNames {
            if let modelURL = Bundle.main.url(forResource: name, withExtension: "mlmodelc") {
                do {
                    let model = try MLModel(contentsOf: modelURL)
                    u2netModel = try VNCoreMLModel(for: model)
                    print("✅ U2-Net loaded: \(name)")
                    return
                } catch {
                    continue
                }
            }
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
                    connection.videoRotationAngle = 90 // Portrait orientation
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
        
        if let u2net = u2netModel {
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
        
        u2netUpdateCounter += 1
        if u2netUpdateCounter < u2netUpdateInterval && u2netMask != nil {
            return
        }
        u2netUpdateCounter = 0
        
        u2netQueue.async { [weak self] in
            guard let self = self else { return }
            
            let request = VNCoreMLRequest(model: model) { [weak self] request, error in
                if let results = request.results as? [VNPixelBufferObservation],
                   let maskBuffer = results.first?.pixelBuffer {
                    self?.u2netMask = maskBuffer
                }
            }
            
            request.imageCropAndScaleOption = .scaleFit
            let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
            
            do {
                try handler.perform([request])
            } catch {}
        }
    }
    
    // COMPLETELY REWRITTEN SCORING WITH STABILITY FOCUS
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
        
        for i in 0..<numDetections {
            let conf = detPointer[4 * numDetections + i]
            
            if conf > confidenceThreshold {
                let x = detPointer[0 * numDetections + i]
                let y = detPointer[1 * numDetections + i]
                let w = detPointer[2 * numDetections + i]
                let h = detPointer[3 * numDetections + i]
                let area = w * h
                let areaRatio = area / frameArea
                
                // HARD FILTERS - Skip immediately
                if areaRatio < 0.08 || areaRatio > 0.70 {
                    continue  // Too small or too large
                }
                
                // === SIZE SCORE (MOST IMPORTANT) ===
                var sizeScore: Float = 0
                
                if areaRatio >= 0.25 && areaRatio <= 0.65 {
                    // OPTIMAL: 25-65%
                    let normalizedSize = (areaRatio - 0.25) / (0.65 - 0.25)
                    sizeScore = 0.80 + (normalizedSize * 0.20)  // 0.80-1.0
                    
                } else if areaRatio >= 0.08 && areaRatio < 0.25 {
                    // PENALTY: 8-25% (small objects)
                    let normalizedSize = (areaRatio - 0.08) / (0.25 - 0.08)
                    sizeScore = normalizedSize * normalizedSize * normalizedSize * 0.7  // CUBIC penalty
                    
                } else {
                    // TOO LARGE: 65-70%
                    sizeScore = max(0, 1.0 - (areaRatio - 0.65) * 5.0)
                }
                
                // === CENTER PROXIMITY (LESS IMPORTANT) ===
                let distanceFromCenter = sqrt(pow(x - frameCenterX, 2) + pow(y - frameCenterY, 2))
                let maxDistance = sqrt(pow(frameCenterX, 2) + pow(frameCenterY, 2))
                let centerProximityScore = 1.0 - (distanceFromCenter / maxDistance)
                
                // === U2NET BONUS (SMALL) ===
                var u2netBonus: Float = 0
                if u2netMask != nil {
                    let overlapRatio = calculateU2NetOverlap(x: x, y: y, width: w, height: h)
                    u2netBonus = overlapRatio * 0.2  // Reduced from 0.3
                }
                
                // === HUGE STABILITY BONUS ===
                var stabilityBonus: Float = 0
                if let currentBox = currentDetectionBox, isLocked {
                    let dx = abs(x - currentBox.x)
                    let dy = abs(y - currentBox.y)
                    let distance = sqrt(dx*dx + dy*dy)
                    
                    if distance < 150 {  // Same detection (VERY LENIENT)
                        stabilityBonus = 0.6  // HUGE bonus - increased from 0.4
                    }
                }
                
                // === COMBINED SCORE ===
                // Size is now MUCH more important than confidence
                let baseScore = sizeScore * sizeScore * conf * centerProximityScore
                let combinedScore = baseScore + u2netBonus + stabilityBonus
                
                // REJECT if below minimum threshold
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
            handleNoDetection()
            return
        }
        
        let areaRatio = (detection.w * detection.h) / frameArea
        
        // Check if similar to current detection
        let isSimilarToCurrent: Bool
        if let currentBox = currentDetectionBox {
            let dx = abs(detection.x - currentBox.x)
            let dy = abs(detection.y - currentBox.y)
            let distance = sqrt(dx*dx + dy*dy)
            isSimilarToCurrent = distance < 150  // VERY LENIENT (was 80)
        } else {
            isSimilarToCurrent = false
        }
        
        // REQUIRE multiple consecutive frames before accepting new detection
        if isSimilarToCurrent {
            consecutiveFramesWithSameDetection += 1
        } else {
            consecutiveFramesWithSameDetection = 1
            // CRITICAL FIX: Store this detection so we can compare against it next frame!
            currentDetectionBox = (detection.x, detection.y, detection.w, detection.h)
        }
        
        // Only accept if seen consistently OR if locked
        if consecutiveFramesWithSameDetection >= requiredConsecutiveFrames || isLocked {
            
            let isNewDetection = !isSimilarToCurrent && !isLocked
            
            if isNewDetection {
                detectionLockTime = now
                consecutiveFramesWithSameDetection = 0
                
                DispatchQueue.main.async {
                    self.detectionId = UUID()
                }
                
                print("🎯 LOCKED: \(String(format: "%.1f%%", areaRatio*100)) score: \(String(format: "%.2f", detection.score))")
            }
            
            noDetectionFrames = 0
            
            // Get mask
            var maskCoeffs = [Float](repeating: 0, count: numPrototypes)
            for i in 0..<numPrototypes {
                let coeffIdx = (numValues - numPrototypes + i) * numDetections + detection.idx
                maskCoeffs[i] = detPointer[coeffIdx]
            }
            
            let segMask = generateCleanerMask(prototypes: prototypes, coefficients: maskCoeffs,
                                              protoHeight: protoHeight, protoWidth: protoWidth)
            
            let ciImage = CIImage(cvPixelBuffer: originalPixelBuffer)
            applyCleanSegmentation(original: ciImage, mask: segMask)
        } else {
            // Not enough consecutive frames yet - keep waiting
            let distanceInfo: String
            if isSimilarToCurrent {
                let dx = abs(detection.x - currentDetectionBox!.x)
                let dy = abs(detection.y - currentDetectionBox!.y)
                let distance = sqrt(dx*dx + dy*dy)
                distanceInfo = " (dist: \(String(format: "%.0f", distance))px)"
            } else {
                distanceInfo = " (first/new detection)"
            }
            print("⏳ Waiting: \(consecutiveFramesWithSameDetection)/\(requiredConsecutiveFrames) frames - Area: \(String(format: "%.1f%%", areaRatio*100))\(distanceInfo)")
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
    
    private func generateCleanerMask(prototypes: MLMultiArray, coefficients: [Float],
                                     protoHeight: Int, protoWidth: Int) -> CIImage {
        let protoPointer = prototypes.dataPointer.assumingMemoryBound(to: Float.self)
        let maskSize = protoHeight * protoWidth
        
        var finalMask = [Float](repeating: 0, count: maskSize)
        
        for protoIdx in 0..<coefficients.count {
            let coeff = coefficients[protoIdx]
            let protoOffset = protoIdx * maskSize
            
            for i in 0..<maskSize {
                finalMask[i] += coeff * protoPointer[protoOffset + i]
            }
        }
        
        let threshold: Float = 0.55
        var pixelData = [UInt8](repeating: 0, count: maskSize)
        
        for i in 0..<maskSize {
            let sigmoid = 1.0 / (1.0 + exp(-finalMask[i]))
            pixelData[i] = sigmoid > threshold ? 255 : 0
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
    
    private func applyCleanSegmentation(original: CIImage, mask: CIImage) {
        let scaleX = original.extent.width / mask.extent.width
        let scaleY = original.extent.height / mask.extent.height
        
        let scaledMask = mask
            .transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))
            .samplingNearest()
        
        var finalMask = scaledMask
        if let blurFilter = CIFilter(name: "CIGaussianBlur") {
            blurFilter.setValue(scaledMask, forKey: kCIInputImageKey)
            blurFilter.setValue(1.0, forKey: kCIInputRadiusKey)
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
