import SwiftUI
import AVFoundation
import CoreML
import Vision
import CoreImage

struct SegmentExamine: View {
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
                                
                                if camera.isAutoThresholding {
                                    Text("Auto: \(Int(camera.confidenceThreshold * 100))%")
                                        .font(.caption2)
                                        .padding(6)
                                        .background(Capsule().fill(Color.blue.opacity(0.3)))
                                        .foregroundColor(.blue)
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
                                .disabled(camera.isAutoThresholding)
                            Text("\(Int(camera.confidenceThreshold * 100))%")
                                .font(.caption)
                                .foregroundColor(.white)
                                .frame(width: 40)
                        }
                        .padding(.horizontal, 12)
                        .padding(.vertical, 8)
                        .background(Capsule().fill(Color.black.opacity(0.5)))
                        .opacity(camera.isAutoThresholding ? 0.6 : 1.0)
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

//struct ScanningReticle: View {
//    @Binding var rotation: Double
//    
//    var body: some View {
//        ZStack {
//            Circle()
//                .stroke(Color.white.opacity(0.3), lineWidth: 2)
//                .frame(width: 120, height: 120)
//            Rectangle()
//                .fill(LinearGradient(gradient: Gradient(colors: [.clear, .white.opacity(0.8), .clear]), startPoint: .top, endPoint: .bottom))
//                .frame(width: 2, height: 60)
//                .offset(y: -30)
//                .rotationEffect(.degrees(rotation))
//            Circle()
//                .fill(Color.white.opacity(0.5))
//                .frame(width: 8, height: 8)
//            ForEach(0..<4) { i in
//                CornerBracket()
//                    .rotationEffect(.degrees(Double(i) * 90))
//            }
//        }
//    }
//}
//
//struct CornerBracket: View {
//    var body: some View {
//        Path { path in
//            path.move(to: CGPoint(x: -60, y: -60))
//            path.addLine(to: CGPoint(x: -40, y: -60))
//            path.move(to: CGPoint(x: -60, y: -60))
//            path.addLine(to: CGPoint(x: -60, y: -40))
//        }
//        .stroke(Color.white.opacity(0.6), lineWidth: 2)
//    }
//}

struct FurnitureDetection {
    let idx: Int
    let score: Float
    let conf: Float
    let x: Float
    let y: Float
    let w: Float
    let h: Float
}

class FastSAMCameraModel: NSObject, ObservableObject {
    @Published var segmentedImage: UIImage?
    @Published var confidenceThreshold: Float = 0.5
    @Published var isAutoAdjusting = false
    @Published var isAutoThresholding = true
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
    
    private var trackedDetectionCenter: (x: Float, y: Float)?
    private var trackingConfidence: Float = 0.0
    private let trackingRadius: Float = 200
    
    private var consecutiveFramesWithSameDetection = 0
    private let requiredConsecutiveFrames = 2
    
    private let minimumAcceptableScore: Float = 0.25
    
    private var detectionHistory: [(x: Float, y: Float, w: Float, h: Float, score: Float)] = []
    private let historySize = 5
    
    private var stableDetectionTime: Date?
    private let stabilityThreshold: TimeInterval = 2.0
    private var noDetectionFrames = 0
    private let autoAdjustTrigger = 15
    
    private var detectionConfidenceHistory: [[Float]] = []
    private var framesSinceLastThresholdUpdate = 0
    private let thresholdUpdateInterval = 30
    private var sceneAnalysisFrames = 0
    private let requiredAnalysisFrames = 5
    
    // ✅ Pure additive accumulation variables
    private var accumulatedMask: [UInt8]?
    private var maskWidth: Int = 0
    private var maskHeight: Int = 0
    private var framesInCurrentGroup = 0
    private let maxAccumulationFrames = 30  // ✨ 30 frames (6 seconds at 5 FPS)
    
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
        detectionConfidenceHistory.removeAll()
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
        
        // Run U2-Net every frame for continuous updates
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
            
            let numDetections = detections.shape[2].intValue
            updateAdaptiveThreshold(detections: detections, numDetections: numDetections)
            
            processWithContinuousGrouping(detections: detections, prototypes: prototypes, originalPixelBuffer: pixelBuffer)
            
        } catch {
            print("❌ Prediction failed: \(error)")
        }
        
        isProcessing = false
    }
    
    private func updateAdaptiveThreshold(detections: MLMultiArray, numDetections: Int) {
        guard isAutoThresholding else { return }
        
        sceneAnalysisFrames += 1
        framesSinceLastThresholdUpdate += 1
        
        let detPointer = detections.dataPointer.assumingMemoryBound(to: Float.self)
        var confidences: [Float] = []
        
        for i in 0..<numDetections {
            let conf = detPointer[4 * numDetections + i]
            if conf > 0.2 {
                confidences.append(conf)
            }
        }
        
        detectionConfidenceHistory.append(confidences)
        if detectionConfidenceHistory.count > 5 {
            detectionConfidenceHistory.removeFirst()
        }
        
        if sceneAnalysisFrames >= requiredAnalysisFrames ||
           framesSinceLastThresholdUpdate >= thresholdUpdateInterval {
            calculateOptimalThreshold()
            framesSinceLastThresholdUpdate = 0
        }
    }
    
    private func calculateOptimalThreshold() {
        var allConfidences: [Float] = []
        for frame in detectionConfidenceHistory {
            allConfidences.append(contentsOf: frame)
        }
        
        guard allConfidences.count >= 10 else { return }
        
        allConfidences.sort()
        
        let count = allConfidences.count
        
        let lowThreshold = allConfidences[count / 3]
        let midThreshold = allConfidences[count / 2]
        let highThreshold = allConfidences[2 * count / 3]
        
        let lowScore = scoreThreshold(lowThreshold, confidences: allConfidences)
        let midScore = scoreThreshold(midThreshold, confidences: allConfidences)
        let highScore = scoreThreshold(highThreshold, confidences: allConfidences)
        
        let bestThreshold: Float
        if lowScore > midScore && lowScore > highScore {
            bestThreshold = lowThreshold
        } else if midScore > highScore {
            bestThreshold = midThreshold
        } else {
            bestThreshold = highThreshold
        }
        
        let clampedThreshold = max(0.3, min(0.7, bestThreshold))
        let smoothedThreshold = confidenceThreshold * 0.7 + clampedThreshold * 0.3
        
        if abs(smoothedThreshold - confidenceThreshold) > 0.05 {
            DispatchQueue.main.async {
                self.confidenceThreshold = smoothedThreshold
            }
            print("🎯 Auto-threshold: \(String(format: "%.2f", smoothedThreshold))")
        }
    }
    
    private func scoreThreshold(_ threshold: Float, confidences: [Float]) -> Float {
        let aboveCount = confidences.filter { $0 > threshold }.count
        let ratio = Float(aboveCount) / Float(confidences.count)
        
        if ratio < 0.1 || ratio > 0.6 {
            return 0
        }
        
        let idealRatio: Float = 0.3
        let ratioScore = 1.0 - abs(ratio - idealRatio) / idealRatio
        
        let acceptedConfidences = confidences.filter { $0 > threshold }
        let meanAccepted = acceptedConfidences.isEmpty ? 0 : acceptedConfidences.reduce(0, +) / Float(acceptedConfidences.count)
        
        let confidenceScore = meanAccepted
        
        return ratioScore * 0.6 + confidenceScore * 0.4
    }
    
    private func runU2NetForGuidanceAsync(pixelBuffer: CVPixelBuffer) {
        guard let model = u2netModel else { return }
        
        u2netQueue.async { [weak self] in
            guard let self = self else { return }
            
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
                print("❌ U2-Net error: \(error)")
            }
        }
    }
    
    private func processWithContinuousGrouping(detections: MLMultiArray, prototypes: MLMultiArray, originalPixelBuffer: CVPixelBuffer) {
        let numDetections = detections.shape[2].intValue
        let numValues = detections.shape[1].intValue
        let numPrototypes = prototypes.shape[1].intValue
        let protoHeight = prototypes.shape[2].intValue
        let protoWidth = prototypes.shape[3].intValue
        
        let detPointer = detections.dataPointer.assumingMemoryBound(to: Float.self)
        
        var allValidDetections: [FurnitureDetection] = []
        var bestDetection: FurnitureDetection? = nil
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
                
                if areaRatio < 0.05 || areaRatio > 0.85 {
                    continue
                }
                
                var sizeScore: Float = 0
                
                if areaRatio >= 0.25 && areaRatio <= 0.65 {
                    let normalizedSize = (areaRatio - 0.25) / (0.65 - 0.25)
                    sizeScore = 0.80 + (normalizedSize * 0.20)
                } else if areaRatio >= 0.05 && areaRatio < 0.25 {
                    let normalizedSize = (areaRatio - 0.05) / (0.25 - 0.05)
                    sizeScore = normalizedSize * normalizedSize * 0.7
                } else {
                    sizeScore = max(0, 1.0 - (areaRatio - 0.65) * 3.0)
                }
                
                let distanceFromCenter = sqrt(pow(x - frameCenterX, 2) + pow(y - frameCenterY, 2))
                let maxDistance = sqrt(pow(frameCenterX, 2) + pow(frameCenterY, 2))
                let centerProximityScore = 1.0 - (distanceFromCenter / maxDistance)
                
                var u2netBonus: Float = 0
                if u2netMask != nil {
                    let overlapRatio = calculateU2NetOverlap(x: x, y: y, width: w, height: h)
                    u2netBonus = overlapRatio * 0.2
                }
                
                var trackingBonus: Float = 0
                if let tracked = trackedDetectionCenter {
                    let dx = abs(x - tracked.x)
                    let dy = abs(y - tracked.y)
                    let distance = sqrt(dx*dx + dy*dy)
                    
                    if distance < trackingRadius {
                        let proximity = 1.0 - (distance / trackingRadius)
                        trackingBonus = proximity * trackingConfidence * 0.4
                    }
                }
                
                let baseScore = sizeScore * sizeScore * conf * centerProximityScore
                let combinedScore = baseScore + u2netBonus + trackingBonus
                
                if combinedScore < minimumAcceptableScore {
                    continue
                }
                
                let detection = FurnitureDetection(idx: i, score: combinedScore, conf: conf, x: x, y: y, w: w, h: h)
                allValidDetections.append(detection)
                
                if combinedScore > bestScore {
                    bestScore = combinedScore
                    bestDetection = detection
                }
            }
        }
        
        guard let primaryDetection = bestDetection else {
            print("❌ No valid detection found")
            handleNoDetection()
            return
        }
        
        let areaRatio = (primaryDetection.w * primaryDetection.h) / frameArea
        print("📍 Primary: Area=\(String(format: "%.1f%%", areaRatio*100)), Score=\(String(format: "%.2f", primaryDetection.score))")
        
        let furnitureGroup = findAllRelatedFurniture(primary: primaryDetection, allDetections: allValidDetections)
        
        if furnitureGroup.count > 1 {
            print("🔗 Grouped \(furnitureGroup.count) parts together")
        }
        
        trackedDetectionCenter = (primaryDetection.x, primaryDetection.y)
        trackingConfidence = min(1.0, trackingConfidence + 0.1)
        
        noDetectionFrames = 0
        
        let segMask = generateCombinedMask(
            detections: furnitureGroup,
            prototypes: prototypes,
            protoHeight: protoHeight,
            protoWidth: protoWidth,
            detPointer: detPointer,
            numDetections: numDetections,
            numValues: numValues,
            numPrototypes: numPrototypes
        )
        
        let ciImage = CIImage(cvPixelBuffer: originalPixelBuffer)
        applyCleanSegmentation(original: ciImage, mask: segMask)
    }
    
    private func findAllRelatedFurniture(primary: FurnitureDetection, allDetections: [FurnitureDetection]) -> [FurnitureDetection] {
        
        var furnitureGroup = [primary]
        var processedIndices = Set<Int>([primary.idx])
        
        let maxAllowedDistance = (primary.w + primary.h) * 0.6
        
        print("\n🔍 Looking for related furniture (strict mode):")
        
        for detection in allDetections {
            if processedIndices.contains(detection.idx) { continue }
            
            var isRelated = false
            
            for existingFurniture in furnitureGroup {
                let dx = abs(detection.x - existingFurniture.x)
                let dy = abs(detection.y - existingFurniture.y)
                let distance = sqrt(dx*dx + dy*dy)
                
                if distance < maxAllowedDistance {
                    isRelated = true
                    break
                }
                
                let det1Left = existingFurniture.x - existingFurniture.w / 2
                let det1Right = existingFurniture.x + existingFurniture.w / 2
                let det1Top = existingFurniture.y - existingFurniture.h / 2
                let det1Bottom = existingFurniture.y + existingFurniture.h / 2
                
                let det2Left = detection.x - detection.w / 2
                let det2Right = detection.x + detection.w / 2
                let det2Top = detection.y - detection.h / 2
                let det2Bottom = detection.y + detection.h / 2
                
                let horizontalGap = max(0, max(det2Left - det1Right, det1Left - det2Right))
                let verticalGap = max(0, max(det2Top - det1Bottom, det1Top - det2Bottom))
                
                if horizontalGap < 50 && verticalGap < 50 {
                    isRelated = true
                    break
                }
            }
            
            if isRelated {
                if u2netMask != nil {
                    let overlapRatio = calculateU2NetOverlap(x: detection.x, y: detection.y, width: detection.w, height: detection.h)
                    
                    if overlapRatio > 0.25 {
                        furnitureGroup.append(detection)
                        processedIndices.insert(detection.idx)
                        print("  ✅ Added: Area=\(String(format: "%.1f%%", (detection.w * detection.h / frameArea)*100)), Overlap=\(String(format: "%.0f%%", overlapRatio*100))")
                    } else {
                        print("  ❌ Rejected: Low U2-Net overlap (\(String(format: "%.0f%%", overlapRatio*100)))")
                    }
                } else {
                    if detection.conf > 0.65 {
                        furnitureGroup.append(detection)
                        processedIndices.insert(detection.idx)
                        print("  ✅ Added: Area=\(String(format: "%.1f%%", (detection.w * detection.h / frameArea)*100))")
                    }
                }
            }
        }
        
        return furnitureGroup
    }
    
    private func handleNoDetection() {
        noDetectionFrames += 1
        
        if noDetectionFrames > 10 {  // ✨ Was 5, now 10 frames
            trackedDetectionCenter = nil
            trackingConfidence = 0.0
            consecutiveFramesWithSameDetection = 0
            resetAccumulation()  // ✨ Reset accumulation when truly lost
        }
        
        DispatchQueue.main.async {
            withAnimation(.easeOut(duration: 0.5)) {
                self.furnitureOpacity = 0.0
            }
        }
    }
    
    // ✅ Reset accumulation function
    private func resetAccumulation() {
        accumulatedMask = nil
        framesInCurrentGroup = 0
        print("🔄 FastSAM: Manual reset of accumulation")
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
    
    // ✅ PURE ADDITIVE ACCUMULATION
    private func accumulateMaskPure(current: [UInt8], width: Int, height: Int) -> [UInt8] {
        framesInCurrentGroup += 1
        
        // ✅ Initialize or reset accumulated mask
        if accumulatedMask == nil || maskWidth != width || maskHeight != height || framesInCurrentGroup > maxAccumulationFrames {
            maskWidth = width
            maskHeight = height
            accumulatedMask = current
            
            if framesInCurrentGroup > maxAccumulationFrames {
                print("🔄 FastSAM: Reset accumulated mask after \(maxAccumulationFrames) frames")
                framesInCurrentGroup = 1
            }
            
            return current
        }
        
        guard var accumulated = accumulatedMask else {
            return current
        }
        
        // ✅ PURE ADDITIVE: Take MAX (no decay!)
        var newPixelsAdded = 0
        
        for i in 0..<(width * height) {
            let newValue = max(current[i], accumulated[i])
            
            if newValue > accumulated[i] {
                newPixelsAdded += 1
            }
            
            accumulated[i] = newValue
        }
        
        accumulatedMask = accumulated
        
        if newPixelsAdded > 0 {
            print("📊 FastSAM Frame \(framesInCurrentGroup)/\(maxAccumulationFrames) - Added \(newPixelsAdded) new pixels")
        }
        
        return accumulated
    }
    
    private func generateCombinedMask(
        detections: [FurnitureDetection],
        prototypes: MLMultiArray,
        protoHeight: Int,
        protoWidth: Int,
        detPointer: UnsafePointer<Float>,
        numDetections: Int,
        numValues: Int,
        numPrototypes: Int
    ) -> CIImage {
        
        let protoPointer = prototypes.dataPointer.assumingMemoryBound(to: Float.self)
        let maskSize = protoHeight * protoWidth
        
        var detectionMasks: [[Float]] = []
        
        for detection in detections {
            var maskCoeffs = [Float](repeating: 0, count: numPrototypes)
            for i in 0..<numPrototypes {
                let coeffIdx = (numValues - numPrototypes + i) * numDetections + detection.idx
                maskCoeffs[i] = detPointer[coeffIdx]
            }
            
            var detectionMask = [Float](repeating: 0, count: maskSize)
            
            for protoIdx in 0..<numPrototypes {
                let coeff = maskCoeffs[protoIdx]
                let protoOffset = protoIdx * maskSize
                
                for i in 0..<maskSize {
                    detectionMask[i] += coeff * protoPointer[protoOffset + i]
                }
            }
            
            for i in 0..<maskSize {
                detectionMask[i] = 1.0 / (1.0 + exp(-detectionMask[i]))
            }
            
            detectionMasks.append(detectionMask)
        }
        
        var combinedConfidence = [Float](repeating: 0, count: maskSize)
        for detectionMask in detectionMasks {
            for i in 0..<maskSize {
                combinedConfidence[i] = max(combinedConfidence[i], detectionMask[i])
            }
        }
        
        var pixelData = [UInt8](repeating: 0, count: maskSize)
        
        print("\n🎯 === MASK GENERATION (U2-Net Dominant) ===")
        
        if let u2netMask = u2netMask {
            CVPixelBufferLockBaseAddress(u2netMask, .readOnly)
            defer { CVPixelBufferUnlockBaseAddress(u2netMask, .readOnly) }
            
            let maskWidth = CVPixelBufferGetWidth(u2netMask)
            let maskHeight = CVPixelBufferGetHeight(u2netMask)
            let bytesPerRow = CVPixelBufferGetBytesPerRow(u2netMask)
            
            if let baseAddress = CVPixelBufferGetBaseAddress(u2netMask) {
                let u2netPtr = baseAddress.assumingMemoryBound(to: UInt8.self)
                var u2netPixels = 0
                
                // ✅ STEP 1: U2-Net is the PRIMARY source
                for y in 0..<protoHeight {
                    for x in 0..<protoWidth {
                        let sourceX = x * maskWidth / protoWidth
                        let sourceY = y * maskHeight / protoHeight
                        let sourceIdx = sourceY * bytesPerRow + sourceX
                        let targetIdx = y * protoWidth + x
                        
                        if u2netPtr[sourceIdx] > 100 {
                            pixelData[targetIdx] = 255
                            u2netPixels += 1
                        }
                    }
                }
                
                print("📊 U2-Net PRIMARY base: \(u2netPixels) pixels")
                
                // ✅ STEP 2: Calculate U2-Net bounding box
                var minX = protoWidth, maxX = 0, minY = protoHeight, maxY = 0
                
                for y in 0..<protoHeight {
                    for x in 0..<protoWidth {
                        let idx = y * protoWidth + x
                        if pixelData[idx] == 255 {
                            minX = min(minX, x)
                            maxX = max(maxX, x)
                            minY = min(minY, y)
                            maxY = max(maxY, y)
                        }
                    }
                }
                
                guard minX <= maxX && minY <= maxY else {
                    print("⚠️ No U2-Net mask found, skipping FastSAM enhancement")
                    return createCIImage(from: pixelData, width: protoWidth, height: protoHeight)
                }
                
                // ✅ STEP 3: REDUCED search area
                let searchMinX = max(0, minX - 20)
                let searchMaxX = min(protoWidth - 1, maxX + 20)
                let searchMinY = max(0, minY - 35)
                let searchMaxY = min(protoHeight - 1, maxY + 20)
                
                print("🔍 Search area: [\(searchMinX),\(searchMinY)] → [\(searchMaxX),\(searchMaxY)]")
                
                var addedFromFastSAM = 0
                
                // ✅ STEP 4: ONLY 3 passes with STRICTER thresholds
                for pass in 1...3 {
                    let passName = ["strict", "medium", "relaxed"][pass - 1]
                    
                    var addedThisPass = 0
                    
                    for y in searchMinY...searchMaxY {
                        for x in searchMinX...searchMaxX {
                            let idx = y * protoWidth + x
                            
                            if pixelData[idx] == 255 { continue }
                            
                            let requiredConfidence: Float = {
                                switch pass {
                                case 1: return 0.85
                                case 2: return 0.75
                                case 3: return 0.65
                                default: return 0.85
                                }
                            }()
                            
                            if combinedConfidence[idx] < requiredConfidence { continue }
                            
                            // ✅ CRITICAL: Check if this pixel is actually part of U2-Net furniture
                            let u2netSampleX = x * maskWidth / protoWidth
                            let u2netSampleY = y * maskHeight / protoHeight
                            let u2netSampleIdx = u2netSampleY * bytesPerRow + u2netSampleX
                            let u2netValue = u2netPtr[u2netSampleIdx]
                            
                            // ✅ STRICT U2-Net check: Only add if U2-Net also sees furniture here
                            if u2netValue < 30 {
                                continue
                            }
                            
                            var nearFurniture = false
                            var furnitureCount = 0
                            
                            let searchRadius = pass <= 1 ? 2 : 3
                            
                            for dy in -searchRadius...searchRadius {
                                for dx in -searchRadius...searchRadius {
                                    if y + dy >= 0 && y + dy < protoHeight &&
                                       x + dx >= 0 && x + dx < protoWidth {
                                        let nIdx = (y + dy) * protoWidth + (x + dx)
                                        if pixelData[nIdx] == 255 {
                                            nearFurniture = true
                                            furnitureCount += 1
                                        }
                                    }
                                }
                            }
                            
                            let requiredNeighbors: Int = {
                                switch pass {
                                case 1: return 3
                                case 2: return 2
                                case 3: return 2
                                default: return 2
                                }
                            }()
                            
                            if nearFurniture && furnitureCount >= requiredNeighbors {
                                pixelData[idx] = 255
                                addedFromFastSAM += 1
                                addedThisPass += 1
                            }
                        }
                    }
                    
                    if addedThisPass > 0 {
                        print("  ↳ Pass \(pass) (\(passName)): +\(addedThisPass) pixels")
                    }
                }
                
                if addedFromFastSAM > 0 {
                    print("📊 FastSAM enhancements: +\(addedFromFastSAM) pixels (U2-Net guided)")
                }
                
                print("📊 Before accumulation: \(pixelData.filter { $0 == 255 }.count) pixels")
            }
        } else {
            print("⚠️ Using FastSAM only (U2-Net not available)")
            
            var maxConfidence: Float = 0
            var seedX = 0
            var seedY = 0
            
            for y in 0..<protoHeight {
                for x in 0..<protoWidth {
                    let idx = y * protoWidth + x
                    if combinedConfidence[idx] > maxConfidence {
                        maxConfidence = combinedConfidence[idx]
                        seedX = x
                        seedY = y
                    }
                }
            }
            
            if maxConfidence > 0.90 {
                var toProcess = [(seedX, seedY)]
                var processed = Set<Int>()
                
                while !toProcess.isEmpty {
                    let (x, y) = toProcess.removeFirst()
                    let idx = y * protoWidth + x
                    
                    if processed.contains(idx) { continue }
                    processed.insert(idx)
                    
                    if combinedConfidence[idx] > 0.85 {
                        pixelData[idx] = 255
                        
                        for dy in -1...1 {
                            for dx in -1...1 {
                                if dy == 0 && dx == 0 { continue }
                                let nx = x + dx
                                let ny = y + dy
                                if nx >= 0 && nx < protoWidth && ny >= 0 && ny < protoHeight {
                                    let nIdx = ny * protoWidth + nx
                                    if !processed.contains(nIdx) {
                                        toProcess.append((nx, ny))
                                    }
                                }
                            }
                        }
                    }
                }
            }
            
            print("📊 Before accumulation: \(pixelData.filter { $0 == 255 }.count) pixels")
        }
        
        // ✅ APPLY ACCUMULATION BEFORE MORPHOLOGY
        pixelData = accumulateMaskPure(current: pixelData, width: protoWidth, height: protoHeight)
        print("📊 After accumulation: \(pixelData.filter { $0 == 255 }.count) pixels")
        
        // ✅ REDUCED morphology: 8 dilate + 5 erode
        pixelData = applyMorphologicalClosing(pixelData, width: protoWidth, height: protoHeight)
        
        print("📊 After morphology: \(pixelData.filter { $0 == 255 }.count) pixels")
        
        return createCIImage(from: pixelData, width: protoWidth, height: protoHeight)
    }
    
    private func createCIImage(from pixelData: [UInt8], width: Int, height: Int) -> CIImage {
        let data = Data(pixelData)
        guard let provider = CGDataProvider(data: data as CFData) else {
            return CIImage(color: .white).cropped(to: CGRect(x: 0, y: 0, width: width, height: height))
        }
        
        let colorSpace = CGColorSpaceCreateDeviceGray()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue)
        
        guard let cgImage = CGImage(
            width: width,
            height: height,
            bitsPerComponent: 8,
            bitsPerPixel: 8,
            bytesPerRow: width,
            space: colorSpace,
            bitmapInfo: bitmapInfo,
            provider: provider,
            decode: nil,
            shouldInterpolate: false,
            intent: .defaultIntent
        ) else {
            return CIImage(color: .white).cropped(to: CGRect(x: 0, y: 0, width: width, height: height))
        }
        
        return CIImage(cgImage: cgImage)
    }
    
    private func applyMorphologicalClosing(_ input: [UInt8], width: Int, height: Int) -> [UInt8] {
        let dilated = dilate(input, width: width, height: height, iterations: 8)
        let closed = erode(dilated, width: width, height: height, iterations: 5)
        return closed
    }
    
    private func dilate(_ input: [UInt8], width: Int, height: Int, iterations: Int) -> [UInt8] {
        var result = input
        
        for _ in 0..<iterations {
            var temp = result
            
            for y in 1..<(height - 1) {
                for x in 1..<(width - 1) {
                    let idx = y * width + x
                    
                    if result[idx] == 0 {
                        var hasWhiteNeighbor = false
                        
                        for dy in -1...1 {
                            for dx in -1...1 {
                                if dx == 0 && dy == 0 { continue }
                                let nIdx = (y + dy) * width + (x + dx)
                                if result[nIdx] == 255 {
                                    hasWhiteNeighbor = true
                                    break
                                }
                            }
                            if hasWhiteNeighbor { break }
                        }
                        
                        if hasWhiteNeighbor {
                            temp[idx] = 255
                        }
                    }
                }
            }
            
            result = temp
        }
        
        return result
    }
    
    private func erode(_ input: [UInt8], width: Int, height: Int, iterations: Int) -> [UInt8] {
        var result = input
        
        for _ in 0..<iterations {
            var temp = result
            
            for y in 1..<(height - 1) {
                for x in 1..<(width - 1) {
                    let idx = y * width + x
                    
                    if result[idx] == 255 {
                        var hasBlackNeighbor = false
                        
                        for dy in -1...1 {
                            for dx in -1...1 {
                                if dx == 0 && dy == 0 { continue }
                                let nIdx = (y + dy) * width + (x + dx)
                                if result[nIdx] == 0 {
                                    hasBlackNeighbor = true
                                    break
                                }
                            }
                            if hasBlackNeighbor { break }
                        }
                        
                        if hasBlackNeighbor {
                            temp[idx] = 0
                        }
                    }
                }
            }
            
            result = temp
        }
        
        return result
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
            blurFilter.setValue(0.5, forKey: kCIInputRadiusKey)
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
