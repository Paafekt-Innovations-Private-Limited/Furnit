// SmartyPantsView.swift
// YOLOE with BBox-Optimized Scanline Fill (captures overlapping objects!)

import SwiftUI
import AVFoundation
import CoreML
import CoreImage
import Photos
import Accelerate

// MARK: - Main View
struct SmartyPantsView: View {
    @Binding var capturedImage: UIImage?
    @Binding var isShowingCamera: Bool
    let roomImage: UIImage?
    
    @StateObject private var camera = FurnitureSegmentationModelSmarty()
    
    @State private var scaleMultiplier: CGFloat = 0.5
    @State private var lastScale: CGFloat = 1.0
    @State private var dragOffset: CGSize = .zero
    @State private var accumulatedOffset: CGSize = .zero
    @State private var showingSaveSuccess = false
    @State private var saveMessage = ""
    
    var body: some View {
        ZStack {
            Color.clear.ignoresSafeArea()
            
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
                        SimultaneousGesture(
                            DragGesture()
                                .onChanged { value in
                                    dragOffset = value.translation
                                }
                                .onEnded { value in
                                    accumulatedOffset.width += value.translation.width
                                    accumulatedOffset.height += value.translation.height
                                    dragOffset = .zero
                                },
                            MagnificationGesture()
                                .onChanged { value in
                                    let delta = value / lastScale
                                    lastScale = value
                                    let newScale = scaleMultiplier * delta
                                    scaleMultiplier = min(max(newScale, 0.3), 2.0)
                                }
                                .onEnded { value in
                                    lastScale = 1.0
                                }
                        )
                    )
                    .ignoresSafeArea()
                    .opacity(camera.furnitureOpacity)
                    .animation(.easeOut(duration: 0.3), value: camera.furnitureOpacity)
            }
            
            if camera.currentBBox != .zero && camera.segmentedImage != nil {
                Canvas { context, size in
                    let rect = Path(camera.currentBBox)
                    context.stroke(rect, with: .color(.blue.opacity(0.3)), lineWidth: 8)
                    context.stroke(rect, with: .color(.blue.opacity(0.6)), lineWidth: 5)
                    context.stroke(rect, with: .color(.blue), lineWidth: 2)
                }
                .ignoresSafeArea()
                .allowsHitTesting(false)
            }
            
            VStack {
                HStack {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("FPS: \(camera.currentFPS, specifier: "%.1f")")
                            .font(.caption)
                        
                        if camera.lastConfidence > 0 {
                            Text("Main: \(camera.allDetections.first?.className ?? "") (\(Int(camera.lastConfidence * 100))%)")
                                .font(.caption2)
                                .foregroundColor(.green)
                        }
                        
                        // Show nearby objects
                        if !camera.nearbyObjects.isEmpty {
                            Text("Nearby:")
                                .font(.caption2)
                                .foregroundColor(.orange)
                            ForEach(camera.nearbyObjects.prefix(3), id: \.self) { obj in
                                Text("• \(obj)")
                                    .font(.caption2)
                                    .foregroundColor(.yellow)
                            }
                        }
                        
                        // Show all detections
                        if camera.allDetections.count > 1 {
                            Text("All (\(camera.allDetections.count)):")
                                .font(.caption2)
                                .foregroundColor(.gray)
                            ForEach(camera.allDetections.prefix(5), id: \.className) { det in
                                Text("• \(det.className) (\(Int(det.confidence * 100))%)")
                                    .font(.caption2)
                                    .foregroundColor(.white.opacity(0.7))
                            }
                        }
                        
                        Text("YOLOE")
                            .font(.caption2)
                            .foregroundColor(.blue)
                    }
                    .foregroundColor(.white)
                    .padding(8)
                    .background(Color.black.opacity(0.8))
                    .cornerRadius(8)
                    Spacer()
                }
                .padding()
                Spacer()
            }
            
            VStack {
                HStack {
                    Spacer()
                    Button(action: { isShowingCamera = false }) {
                        Image(systemName: "xmark.circle.fill")
                            .font(.title)
                            .foregroundColor(.white)
                    }
                }
                .padding(.top, 60)
                .padding(.horizontal)
                
                Spacer()
                
                if camera.segmentedImage != nil {
                    HStack(spacing: 16) {
                        Button(action: { captureFurnitureWithRoom() }) {
                            VStack {
                                Image(systemName: "camera.circle.fill")
                                Text("Capture")
                                    .font(.caption2)
                            }
                            .foregroundColor(.white)
                            .frame(width: 70, height: 70)
                            .background(Circle().fill(Color.blue))
                            .shadow(radius: 5)
                        }
                        
                        Button(action: {
                            camera.resetSegmentation()
                            scaleMultiplier = 0.5
                            dragOffset = .zero
                            accumulatedOffset = .zero
                        }) {
                            VStack {
                                Image(systemName: "arrow.counterclockwise")
                                Text("Reset")
                                    .font(.caption2)
                            }
                            .foregroundColor(.white)
                            .frame(width: 60, height: 60)
                            .background(Circle().fill(Color.orange))
                            .shadow(radius: 3)
                        }
                    }
                    .padding(.bottom, 50)
                    .padding(.trailing, 20)
                    .frame(maxWidth: .infinity, alignment: .trailing)
                }
            }
            
            if camera.segmentedImage != nil {
                VStack {
                    Spacer()
                    Text("Pinch to scale • Drag to move")
                        .font(.caption)
                        .foregroundColor(.white)
                        .padding(8)
                        .background(Capsule().fill(Color.black.opacity(0.6)))
                        .padding(.bottom, 120)
                }
            }
            
            if showingSaveSuccess {
                VStack {
                    Spacer()
                    HStack {
                        Image(systemName: "checkmark.circle.fill")
                        Text(saveMessage)
                    }
                    .foregroundColor(.white)
                    .padding()
                    .background(Capsule().fill(Color.blue))
                    Spacer().frame(height: 100)
                }
            }
        }
        .onAppear {
            camera.startSession()
        }
        .onDisappear { camera.stopSession() }
    }
    
    private func captureFurnitureWithRoom() {
        guard let furniture = camera.segmentedImage else {
            saveMessage = "No furniture detected!"
            showingSaveSuccess = true
            DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
                showingSaveSuccess = false
            }
            return
        }
        
        guard let room = roomImage else {
            saveMessage = "No room image!"
            showingSaveSuccess = true
            DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
                showingSaveSuccess = false
            }
            return
        }
        
        UIGraphicsBeginImageContextWithOptions(room.size, false, room.scale)
        defer { UIGraphicsEndImageContext() }
        
        room.draw(at: .zero)
        
        let furnitureRect = CGRect(
            x: (room.size.width - furniture.size.width) / 2,
            y: (room.size.height - furniture.size.height) / 2,
            width: furniture.size.width,
            height: furniture.size.height
        )
        
        furniture.draw(in: furnitureRect)
        
        guard let composite = UIGraphicsGetImageFromCurrentImageContext() else {
            saveMessage = "Composite failed!"
            showingSaveSuccess = true
            return
        }
        
        PHPhotoLibrary.requestAuthorization(for: .addOnly) { status in
            DispatchQueue.main.async {
                if status == .authorized || status == .limited {
                    PHPhotoLibrary.shared().performChanges({
                        PHAssetChangeRequest.creationRequestForAsset(from: composite)
                    }) { success, error in
                        DispatchQueue.main.async {
                            if success {
                                self.saveMessage = "Saved!"
                                self.showingSaveSuccess = true
                                DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
                                    self.showingSaveSuccess = false
                                    self.isShowingCamera = false
                                }
                            } else {
                                self.saveMessage = "Failed!"
                                self.showingSaveSuccess = true
                            }
                        }
                    }
                }
            }
        }
    }
}

// MARK: - Detection Structure
struct DetectionSmarty {
    let x: Float
    let y: Float
    let width: Float
    let height: Float
    let confidence: Float
    let classIdx: Int
    let className: String
    let maskCoeffs: [Float]
}

// MARK: - Main Model with BBox-Optimized Fill
class FurnitureSegmentationModelSmarty: NSObject, ObservableObject {
    @Published var segmentedImage: UIImage?
    @Published var furnitureOpacity: Double = 0.0
    @Published var isProcessing = false
    @Published var currentFPS: Double = 0.0
    @Published var lastConfidence: Float = 0.0
    @Published var currentBBox: CGRect = .zero
    @Published var allDetections: [DetectionSmarty] = []  // NEW - all detected objects
    @Published var nearbyObjects: [String] = []  // NEW - objects close to main detection
    
    private let session = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let videoQueue = DispatchQueue(label: "yoloeVideo", qos: .userInitiated)
    private let detectionQueue = DispatchQueue(label: "yoloeDetection", qos: .userInitiated)
    
    private var mlModel: MLModel?
    private let context = CIContext(options: [.workingColorSpace: CGColorSpaceCreateDeviceRGB()])
    
    private let furnitureClasses: [Int: String] = [
        132: "armchair",
        213: "baby seat",
        225: "badminton racket",
        274: "bar",
        275: "bar code",
        276: "bar stool",
        277: "barbecue",
        278: "barbecue grill",
        279: "barbell",
        280: "barber",
        281: "barber shop",
        282: "barbie",
        283: "barge",
        284: "barista",
        285: "bark",
        286: "barley",
        287: "barn",
        288: "barn owl",
        289: "barn door",
        290: "barrel",
        291: "barricade",
        292: "barrier",
        294: "bartender",
        332: "bathroom cabinet",
        334: "bathroom mirror",
        352: "beach chair",
        364: "bean bag chair",
        375: "bed",
        376: "bedcover",
        377: "bed frame",
        378: "bedroom",
        379: "bedding",
        380: "bedpan",
        381: "bedroom window",
        382: "bedside lamp",
        402: "bench",
        429: "billiard table",
        517: "bookshelf",
        546: "underdrawers",
        552: "bracket",
        567: "chest",
        604: "bucket cabinet",
        632: "bunk bed",
        636: "bureau",
        670: "cabinet",
        671: "cabinetry",
        679: "cake stand",
        707: "candy bar",
        714: "canopy bed",
        731: "car mirror",
        733: "car seat",
        781: "cat bed",
        821: "chair",
        822: "chairlift",
        823: "daybed",
        834: "changing table",
        870: "chestnut",
        896: "chocolate bar",
        977: "closet",
        996: "coatrack",
        1006: "cocktail table",
        1060: "computer chair",
        1061: "computer desk",
        1133: "cosmetics mirror",
        1137: "infant bed",
        1141: "couch",
        1143: "counter",
        1144: "counter top",
        1167: "crack",
        1198: "crossbar",
        1204: "crowbar",
        1270: "day bed",
        1301: "table",
        1302: "table lamp",
        1303: "desktop",
        1304: "desktop computer",
        1325: "dinning table",
        1335: "dirt track",
        1364: "dog bed",
        1396: "drawer",
        1405: "dresser",
        1476: "electric chair",
        1503: "side table",
        1602: "feeding chair",
        1624: "file cabinet",
        1646: "firecracker",
        1704: "flower bed",
        1721: "folding chair",
        1733: "food stand",
        1750: "footrest",
        1801: "fruit stand",
        1816: "futon",
        1885: "glass table",
        2022: "handstand",
        2081: "high bar",
        2141: "hospital bed",
        2193: "ice shelf",
        2218: "inflatable boat",
        2219: "information desk",
        2247: "island",
        2318: "kitchen cabinet",
        2319: "kitchen counter",
        2322: "kitchen island",
        2324: "kitchen table",
        2499: "loveseat",
        2599: "mattress",
        2614: "medicine cabinet",
        2654: "mirror",
        2754: "music stool",
        2796: "newsstand",
        2802: "nightstand",
        2817: "nutcracker",
        2834: "office chair",
        2836: "office desk",
        2870: "orchestra pit",
        2939: "park bench",
        3024: "church bench",
        3045: "picnic table",
        3051: "tablet",
        3061: "table tennis table",
        3062: "table tennis",
        3145: "poker table",
        3175: "portable battery",
        3279: "race track",
        3282: "racket",
        3322: "rearview mirror",
        3403: "riverbed",
        3423: "rocking chair",
        3449: "round table",
        3502: "sand bar",
        3575: "seabed",
        3584: "seat",
        3585: "seat belt",
        3621: "shelf",
        3678: "side cabinet",
        3812: "spice rack",
        3848: "stable",
        3862: "stand",
        3863: "standing",
        3888: "step stool",
        3909: "stool",
        4004: "supermarket shelf",
        4015: "sushi bar",
        4041: "swivel chair",
        4054: "table tennis racket",
        4055: "table top",
        4056: "tablecloth",
        4057: "tablet computer",
        4058: "tableware",
        4117: "tennis racket",
        4179: "toilet seat",
        4213: "towel bar",
        4222: "track",
        4239: "train track",
        4243: "training bench",
        4294: "tv cabinet",
        4331: "vanity",
        4337: "vegetable",
        4338: "vegetable garden",
        4339: "vegetable market",
        4359: "view mirror",
        4464: "wet bar",
        4473: "wheelchair",
        4506: "window seat",
        4513: "wine cabinet",
        4516: "wine rack",
        4545: "workbench",
        4564: "writing desk"
    ]
    
    private var lastProcessTime = Date()
    private let processInterval: TimeInterval = 0.06
    private var frameCount = 0
    private var fpsStartTime = Date()
    private var lastFPSUpdate = Date()  // NEW - track last FPS update
    
    private func sigmoid(_ x: Float) -> Float {
        return 1.0 / (1.0 + exp(-x))
    }
    
    override init() {
        super.init()
        loadYOLOModel()
        setupCamera()
    }
    
    func resetSegmentation() {
        DispatchQueue.main.async {
            self.segmentedImage = nil
            self.furnitureOpacity = 0.0
            self.lastConfidence = 0.0
            self.currentBBox = .zero
        }
    }
    
    private func loadYOLOModel() {
        print("🔍 [YOLOE] Loading model...")
        
        do {
            let config = MLModelConfiguration()
            config.computeUnits = .all
            
            for ext in ["mlmodelc", "mlpackage"] {
                if let modelURL = Bundle.main.url(forResource: "yoloe-11l-seg-pf", withExtension: ext) {
                    print("📦 [YOLOE] Found: yoloe-11l-seg-pf.\(ext)")
                    mlModel = try MLModel(contentsOf: modelURL, configuration: config)
                    print("✅ [YOLOE] Model loaded!")
                    return
                }
            }
            print("❌ [YOLOE] Model file not found")
        } catch {
            print("❌ [YOLOE] Load failed: \(error)")
        }
    }
    
    private func setupCamera() {
        session.sessionPreset = .hd1280x720
        
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else {
            print("❌ No camera")
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
                    connection.isVideoMirrored = false
                }
            }
            
            print("✅ Camera configured")
        } catch {
            print("❌ Camera setup failed: \(error)")
        }
    }
    
    func startSession() {
        if !session.isRunning {
            DispatchQueue.global(qos: .background).async {
                self.session.startRunning()
                DispatchQueue.main.async {
                    self.fpsStartTime = Date()
                }
            }
        }
    }
    
    func stopSession() {
        if session.isRunning {
            session.stopRunning()
        }
    }
    
    private func updateFPS() {
        frameCount += 1
        let now = Date()
        let elapsed = now.timeIntervalSince(fpsStartTime)
        
        if elapsed >= 1.0 {
            let fps = Double(frameCount) / elapsed
            DispatchQueue.main.async {
                self.currentFPS = fps
            }
            frameCount = 0
            fpsStartTime = now
        }
    }
    
    private func processWithYOLO(pixelBuffer: CVPixelBuffer) {
        guard let model = mlModel else { return }
        
        let now = Date()
        guard now.timeIntervalSince(lastProcessTime) >= processInterval else { return }
        guard !isProcessing else { return }
        
        lastProcessTime = now
        updateFPS()
        
        DispatchQueue.main.async {
            self.isProcessing = true
        }
        
        detectionQueue.async { [weak self] in
            guard let self = self else { return }
            
            guard let resized = self.resizePixelBuffer(pixelBuffer, width: 640, height: 640),
                  let inputArray = self.pixelBufferToMLMultiArray(resized) else {
                DispatchQueue.main.async { self.isProcessing = false }
                return
            }
            
            let inputDict: [String: Any] = ["image": inputArray]
            
            guard let inputProvider = try? MLDictionaryFeatureProvider(dictionary: inputDict),
                  let output = try? model.prediction(from: inputProvider),
                  let detectionsArray = output.featureValue(for: "var_2421")?.multiArrayValue,
                  let prototypesArray = output.featureValue(for: "p")?.multiArrayValue else {
                DispatchQueue.main.async { self.isProcessing = false }
                return
            }
            
            self.processYOLOResults(detectionsArray, prototypes: prototypesArray, originalImage: pixelBuffer)
        }
    }
    
    private func processYOLOResults(_ detections: MLMultiArray, prototypes: MLMultiArray, originalImage: CVPixelBuffer) {
        let validDetections = extractDetections(from: detections)
        let nmsDetections = applyNMS(detections: validDetections, iouThreshold: 0.45)
        
        guard let bestDetection = nmsDetections.first else {
            DispatchQueue.main.async {
                self.isProcessing = false
                self.segmentedImage = nil
                self.furnitureOpacity = 0.0
                self.lastConfidence = 0.0
                self.currentBBox = .zero
                self.allDetections = []
                self.nearbyObjects = []
            }
            return
        }
        
        print("🪑 [YOLOE] \(bestDetection.className): \(Int(bestDetection.confidence * 100))%")
        
        // Store ALL detections
        DispatchQueue.main.async {
            self.allDetections = nmsDetections
        }
        
        // Find objects close to main detection (within 50 pixels)
        var nearby: [String] = []
        for detection in nmsDetections where detection.className != bestDetection.className {
            let dx = abs(detection.x - bestDetection.x)
            let dy = abs(detection.y - bestDetection.y)
            let distance = sqrt(dx*dx + dy*dy)
            
            if distance < 150 {  // Within 150 pixels in 640x640 space
                nearby.append("\(detection.className) (\(Int(detection.confidence * 100))%)")
            }
        }
        
        DispatchQueue.main.async {
            self.nearbyObjects = nearby
        }
        
        // Use ORIGINAL bbox in 640x640 space (not scaled down!)
        let bbox = CGRect(
            x: CGFloat(bestDetection.x - bestDetection.width/2),
            y: CGFloat(bestDetection.y - bestDetection.height/2),
            width: CGFloat(bestDetection.width),
            height: CGFloat(bestDetection.height)
        )
        
        DispatchQueue.main.async {
            self.currentBBox = bbox
        }
        
        processAndApplyMask(detection: bestDetection, prototypes: prototypes, originalImage: originalImage)
    }
    
    private func pixelBufferToMLMultiArray(_ pixelBuffer: CVPixelBuffer) -> MLMultiArray? {
        guard let array = try? MLMultiArray(shape: [1, 3, 640, 640] as [NSNumber], dataType: .float16) else {
            return nil
        }
        
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        
        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else { return nil }
        
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let buffer = baseAddress.assumingMemoryBound(to: UInt8.self)
        
        for y in 0..<640 {
            for x in 0..<640 {
                let pixelIndex = y * bytesPerRow + x * 4
                let r = Float(buffer[pixelIndex + 2]) / 255.0
                let g = Float(buffer[pixelIndex + 1]) / 255.0
                let b = Float(buffer[pixelIndex]) / 255.0
                
                array[[0, 0, y, x] as [NSNumber]] = NSNumber(value: r)
                array[[0, 1, y, x] as [NSNumber]] = NSNumber(value: g)
                array[[0, 2, y, x] as [NSNumber]] = NSNumber(value: b)
            }
        }
        
        return array
    }
    
    private func resizePixelBuffer(_ pixelBuffer: CVPixelBuffer, width: Int, height: Int) -> CVPixelBuffer? {
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let scaleX = CGFloat(width) / CGFloat(CVPixelBufferGetWidth(pixelBuffer))
        let scaleY = CGFloat(height) / CGFloat(CVPixelBufferGetHeight(pixelBuffer))
        let scaledImage = ciImage.transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))
        
        var newPixelBuffer: CVPixelBuffer?
        CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA, nil, &newPixelBuffer)
        
        guard let outputBuffer = newPixelBuffer else { return nil }
        
        CIContext().render(scaledImage, to: outputBuffer)
        return outputBuffer
    }
    
    private func extractDetections(from detections: MLMultiArray) -> [DetectionSmarty] {
        var allDetections: [DetectionSmarty] = []
        let confThreshold: Float = 0.3
        let anchors = detections.shape[2].intValue

        for anchor in 0..<anchors {
            let x = detections[[0, 0, anchor] as [NSNumber]].floatValue
            let y = detections[[0, 1, anchor] as [NSNumber]].floatValue
            let w = detections[[0, 2, anchor] as [NSNumber]].floatValue
            let h = detections[[0, 3, anchor] as [NSNumber]].floatValue

            for (classIdx, className) in furnitureClasses {
                let conf = detections[[0, 4 + classIdx, anchor] as [NSNumber]].floatValue

                if conf > confThreshold {
                    var maskCoeffs = [Float](repeating: 0, count: 32)
                    for i in 0..<32 {
                        maskCoeffs[i] = detections[[0, 4 + 4585 + i, anchor] as [NSNumber]].floatValue
                    }

                    allDetections.append(DetectionSmarty(
                        x: x, y: y, width: w, height: h,
                        confidence: conf, classIdx: classIdx,
                        className: className, maskCoeffs: maskCoeffs
                    ))
                }
            }
        }
        return allDetections
    }
    
    private func applyNMS(detections: [DetectionSmarty], iouThreshold: Float) -> [DetectionSmarty] {
        guard !detections.isEmpty else { return [] }
        
        let sorted = detections.sorted { $0.confidence > $1.confidence }
        var kept: [DetectionSmarty] = []
        var suppressed = Set<Int>()
        
        for (idx, detection) in sorted.enumerated() {
            if suppressed.contains(idx) { continue }
            kept.append(detection)
            
            for (otherIdx, other) in sorted.enumerated() where otherIdx > idx {
                if suppressed.contains(otherIdx) { continue }
                let iou = calculateIoU(detection, other)
                if iou > iouThreshold {
                    suppressed.insert(otherIdx)
                }
            }
        }
        return kept
    }
    
    private func calculateIoU(_ a: DetectionSmarty, _ b: DetectionSmarty) -> Float {
        let x1 = max(a.x - a.width/2, b.x - b.width/2)
        let y1 = max(a.y - a.height/2, b.y - b.height/2)
        let x2 = min(a.x + a.width/2, b.x + b.width/2)
        let y2 = min(a.y + a.height/2, b.y + b.height/2)
        
        let intersection = max(0, x2 - x1) * max(0, y2 - y1)
        let union = a.width * a.height + b.width * b.height - intersection
        return union > 0 ? intersection / union : 0
    }
    
    // MARK: - BBox + Mask Intersection ONLY
    private func processAndApplyMask(detection: DetectionSmarty, prototypes: MLMultiArray, originalImage: CVPixelBuffer) {
        DispatchQueue.main.async { self.lastConfidence = detection.confidence }
        
        let baseMask = generateMaskUltralytics(coefficients: detection.maskCoeffs, prototypes: prototypes)
        let finalMask = applyBboxFilter(mask: baseMask, detection: detection)
        applyMaskToImage(mask: finalMask, detection: detection, to: originalImage)
    }
    
    private func generateMaskUltralytics(coefficients: [Float], prototypes: MLMultiArray) -> [Float] {
        var mask = [Float](repeating: 0, count: 160 * 160)
        
        for y in 0..<160 {
            for x in 0..<160 {
                var sum: Float = 0
                for c in 0..<32 {
                    let protoValue = prototypes[[0, c, y, x] as [NSNumber]].floatValue
                    sum += coefficients[c] * protoValue
                }
                mask[y * 160 + x] = sigmoid(sum)
            }
        }
        return mask
    }
    
    // MARK: - Scanline Fill + Morphological Smoothing + Blur
    private func applyBboxFilter(mask: [Float], detection: DetectionSmarty) -> [Float] {
        let threshold: Float = 0.5
        var binary = [[UInt8]](repeating: [UInt8](repeating: 0, count: 160), count: 160)
        
        // Convert to binary
        for y in 0..<160 {
            for x in 0..<160 {
                let idx = y * 160 + x
                binary[y][x] = mask[idx] > threshold ? 1 : 0
            }
        }
        
        // Scale bbox from 640x640 to 160x160 space
        let scale: Float = 160.0 / 640.0
        
        let bboxX = Int((detection.x - detection.width/2) * scale)
        let bboxY = Int((detection.y - detection.height/2) * scale)
        let bboxW = Int(detection.width * scale)
        let bboxH = Int(detection.height * scale)
        
        let minX = max(0, bboxX)
        let minY = max(0, bboxY)
        let maxX = min(159, bboxX + bboxW)
        let maxY = min(159, bboxY + bboxH)
        
        // STEP 1: AGGRESSIVE Morphological closing (dilate 5x + erode 5x)
        // This captures monitors on tables, chair wheels, etc.
        binary = morphologicalClose(binary: binary, iterations: 5)
        
        // STEP 2: Scanline fill within bbox to close holes
        var filled = binary
        
        // Horizontal scanline fill
        for y in minY...maxY {
            var firstX = -1, lastX = -1
            for x in minX...maxX {
                if binary[y][x] == 1 {
                    if firstX == -1 { firstX = x }
                    lastX = x
                }
            }
            if firstX != -1 && lastX != -1 {
                for x in firstX...lastX {
                    filled[y][x] = 1
                }
            }
        }
        
        // Vertical scanline fill
        for x in minX...maxX {
            var firstY = -1, lastY = -1
            for y in minY...maxY {
                if binary[y][x] == 1 {
                    if firstY == -1 { firstY = y }
                    lastY = y
                }
            }
            if firstY != -1 && lastY != -1 {
                for y in firstY...lastY {
                    filled[y][x] = 1
                }
            }
        }
        
        // STEP 3: Find largest connected component
        var visited = [[Bool]](repeating: [Bool](repeating: false, count: 160), count: 160)
        var largestComponent: [(Int, Int)] = []
        
        for y in 0..<160 {
            for x in 0..<160 {
                if filled[y][x] == 1 && !visited[y][x] {
                    let component = floodFillComponent(binary: filled, startY: y, startX: x, visited: &visited)
                    if component.count > largestComponent.count {
                        largestComponent = component
                    }
                }
            }
        }
        
        // STEP 4: Create final mask from largest component
        var finalBinary = [[UInt8]](repeating: [UInt8](repeating: 0, count: 160), count: 160)
        for (y, x) in largestComponent {
            finalBinary[y][x] = 1
        }
        
        // STEP 5: Smooth edges with morphological opening (erode + dilate)
        finalBinary = morphologicalOpen(binary: finalBinary, iterations: 1)
        
        // STEP 6: Convert to float and apply Gaussian blur for anti-aliasing
        var floatMask = [Float](repeating: 0, count: 160 * 160)
        for y in 0..<160 {
            for x in 0..<160 {
                floatMask[y * 160 + x] = Float(finalBinary[y][x])
            }
        }
        
        // Apply Gaussian blur for smooth edges
        floatMask = gaussianBlur(mask: floatMask, width: 160, height: 160, sigma: 1.5)
        
        // Blend with original mask for texture preservation
        var result = [Float](repeating: 0, count: 160 * 160)
        for i in 0..<(160 * 160) {
            if floatMask[i] > 0.1 {
                result[i] = max(floatMask[i], mask[i])
            }
        }
        
        return result
    }
    
    // MARK: - Morphological Operations
    private func morphologicalClose(binary: [[UInt8]], iterations: Int) -> [[UInt8]] {
        var result = binary
        // Dilate
        for _ in 0..<iterations {
            result = dilate(binary: result)
        }
        // Erode
        for _ in 0..<iterations {
            result = erode(binary: result)
        }
        return result
    }
    
    private func morphologicalOpen(binary: [[UInt8]], iterations: Int) -> [[UInt8]] {
        var result = binary
        // Erode
        for _ in 0..<iterations {
            result = erode(binary: result)
        }
        // Dilate
        for _ in 0..<iterations {
            result = dilate(binary: result)
        }
        return result
    }
    
    private func dilate(binary: [[UInt8]]) -> [[UInt8]] {
        var result = binary
        for y in 1..<159 {
            for x in 1..<159 {
                if binary[y][x] == 0 {
                    // Check 3x3 neighborhood
                    if binary[y-1][x] == 1 || binary[y+1][x] == 1 ||
                       binary[y][x-1] == 1 || binary[y][x+1] == 1 ||
                       binary[y-1][x-1] == 1 || binary[y-1][x+1] == 1 ||
                       binary[y+1][x-1] == 1 || binary[y+1][x+1] == 1 {
                        result[y][x] = 1
                    }
                }
            }
        }
        return result
    }
    
    private func erode(binary: [[UInt8]]) -> [[UInt8]] {
        var result = binary
        for y in 1..<159 {
            for x in 1..<159 {
                if binary[y][x] == 1 {
                    // Check if any neighbor is 0
                    if binary[y-1][x] == 0 || binary[y+1][x] == 0 ||
                       binary[y][x-1] == 0 || binary[y][x+1] == 0 {
                        result[y][x] = 0
                    }
                }
            }
        }
        return result
    }
    
    // MARK: - Gaussian Blur for Anti-aliasing
    private func gaussianBlur(mask: [Float], width: Int, height: Int, sigma: Float) -> [Float] {
        // Create Gaussian kernel
        let kernelSize = 5
        let halfSize = kernelSize / 2
        var kernel = [Float](repeating: 0, count: kernelSize * kernelSize)
        var sum: Float = 0
        
        for y in 0..<kernelSize {
            for x in 0..<kernelSize {
                let dx = Float(x - halfSize)
                let dy = Float(y - halfSize)
                let value = exp(-(dx*dx + dy*dy) / (2 * sigma * sigma))
                kernel[y * kernelSize + x] = value
                sum += value
            }
        }
        
        // Normalize kernel
        for i in 0..<kernel.count {
            kernel[i] /= sum
        }
        
        // Apply convolution
        var result = [Float](repeating: 0, count: width * height)
        
        for y in halfSize..<(height - halfSize) {
            for x in halfSize..<(width - halfSize) {
                var value: Float = 0
                
                for ky in 0..<kernelSize {
                    for kx in 0..<kernelSize {
                        let py = y + ky - halfSize
                        let px = x + kx - halfSize
                        value += mask[py * width + px] * kernel[ky * kernelSize + kx]
                    }
                }
                
                result[y * width + x] = value
            }
        }
        
        return result
    }
    
    // MARK: - Flood Fill to Find Connected Component
    private func floodFillComponent(binary: [[UInt8]], startY: Int, startX: Int, visited: inout [[Bool]]) -> [(Int, Int)] {
        var component: [(Int, Int)] = []
        var stack: [(Int, Int)] = [(startY, startX)]
        
        while !stack.isEmpty {
            let (y, x) = stack.removeLast()
            
            if y < 0 || y >= 160 || x < 0 || x >= 160 { continue }
            if visited[y][x] { continue }
            if binary[y][x] == 0 { continue }
            
            visited[y][x] = true
            component.append((y, x))
            
            // 4-connected neighbors
            stack.append((y-1, x))
            stack.append((y+1, x))
            stack.append((y, x-1))
            stack.append((y, x+1))
        }
        
        return component
    }
    
    private func applyMaskToImage(mask: [Float], detection: DetectionSmarty, to pixelBuffer: CVPixelBuffer) {
        autoreleasepool {
            let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
            let width = CVPixelBufferGetWidth(pixelBuffer)
            let height = CVPixelBufferGetHeight(pixelBuffer)
            
            guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent),
                  let ctx = CGContext(data: nil, width: width, height: height,
                                     bitsPerComponent: 8, bytesPerRow: width * 4,
                                     space: CGColorSpaceCreateDeviceRGB(),
                                     bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else {
                DispatchQueue.main.async { self.isProcessing = false }
                return
            }
            
            ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
            
            guard let data = ctx.data else {
                DispatchQueue.main.async { self.isProcessing = false }
                return
            }
            
            let pixels = data.bindMemory(to: UInt8.self, capacity: width * height * 4)
            let threshold: Float = 0.5
            
            for py in 0..<height {
                for px in 0..<width {
                    let idx = (py * width + px) * 4
                    let maskX = Float(px) * 160.0 / Float(width)
                    let maskY = Float(py) * 160.0 / Float(height)
                    
                    let x0 = Int(maskX), y0 = Int(maskY)
                    let x1Val = min(x0 + 1, 159), y1Val = min(y0 + 1, 159)
                    
                    if x0 >= 0 && x0 < 160 && y0 >= 0 && y0 < 160 {
                        let dx = maskX - Float(x0), dy = maskY - Float(y0)
                        let v00 = mask[y0 * 160 + x0]
                        let v10 = mask[y0 * 160 + x1Val]
                        let v01 = mask[y1Val * 160 + x0]
                        let v11 = mask[y1Val * 160 + x1Val]
                        let maskValue = (v00 * (1.0 - dx) + v10 * dx) * (1.0 - dy) + (v01 * (1.0 - dx) + v11 * dx) * dy
                        
                        if maskValue > threshold {
                            pixels[idx + 3] = UInt8(maskValue * 255.0)
                            pixels[idx] = UInt8(Float(pixels[idx]) * maskValue)
                            pixels[idx + 1] = UInt8(Float(pixels[idx + 1]) * maskValue)
                            pixels[idx + 2] = UInt8(Float(pixels[idx + 2]) * maskValue)
                        } else {
                            pixels[idx + 3] = 0
                        }
                    } else {
                        pixels[idx + 3] = 0
                    }
                }
            }
            
            if let finalImage = ctx.makeImage() {
                let uiImage = UIImage(cgImage: finalImage, scale: 1.0, orientation: .up)
                DispatchQueue.main.async {
                    self.segmentedImage = uiImage
                    withAnimation(.easeIn(duration: 0.3)) {
                        self.furnitureOpacity = 1.0
                    }
                    self.isProcessing = false
                }
            }
        }
    }
}

extension FurnitureSegmentationModelSmarty: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        processWithYOLO(pixelBuffer: pixelBuffer)
    }
}
