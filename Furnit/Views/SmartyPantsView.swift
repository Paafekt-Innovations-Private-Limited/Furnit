import SwiftUI
import AVFoundation
import CoreML
import CoreImage
import Photos
import Accelerate

private let SEGMENT_DEBUG_SAVE_IMAGES = true

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
                                .onChanged { value in dragOffset = value.translation }
                                .onEnded { value in
                                    accumulatedOffset.width += value.translation.width
                                    accumulatedOffset.height += value.translation.height
                                    dragOffset = .zero
                                },
                            MagnificationGesture()
                                .onChanged { value in
                                    let delta = value / lastScale
                                    lastScale = value
                                    scaleMultiplier = min(max(scaleMultiplier * delta, 0.3), 2.0)
                                }
                                .onEnded { _ in lastScale = 1.0 }
                        )
                    )
                    .ignoresSafeArea()
                    .opacity(camera.furnitureOpacity)
                    .animation(.easeOut(duration: 0.3), value: camera.furnitureOpacity)
            }
            
            // (Optional) multiple bbox canvas with class labels – only when debug flag on
            if SEGMENT_DEBUG_SAVE_IMAGES && !camera.currentDetections.isEmpty && camera.segmentedImage != nil {
                Canvas { context, size in
                    guard let segmented = camera.segmentedImage else { return }
                    
                    // Calculate displayed image rect on screen
                    // Image is positioned at screen center with scale and offset applied
                    let screenCenter = CGPoint(x: UIScreen.main.bounds.width / 2, y: UIScreen.main.bounds.height / 2)
                    let imageSize = segmented.size
                    let scaledImageSize = CGSize(width: imageSize.width * scaleMultiplier,
                                                 height: imageSize.height * scaleMultiplier)
                    
                    let offsetX = dragOffset.width + accumulatedOffset.width
                    let offsetY = dragOffset.height + accumulatedOffset.height
                    
                    let imageOrigin = CGPoint(x: screenCenter.x - scaledImageSize.width / 2 + offsetX,
                                              y: screenCenter.y - scaledImageSize.height / 2 + offsetY)
                    let imageRect = CGRect(origin: imageOrigin, size: scaledImageSize)
                    
                    // Model coordinates are in 640x640 space; map each bbox accordingly
                    for (index, detection) in camera.currentDetections.enumerated() {
                        guard let tight = detection.tightBBox else { continue }
                        let modelRect = tight
                        
                        // Map modelRect from 640x640 to displayed imageRect
                        let scaleX = imageRect.width / 640.0
                        let scaleY = imageRect.height / 640.0
                        
                        let mappedRect = CGRect(
                            x: imageRect.minX + modelRect.minX * scaleX,
                            y: imageRect.minY + modelRect.minY * scaleY,
                            width: modelRect.width * scaleX,
                            height: modelRect.height * scaleY
                        )
                        
                        // Log which bbox is used for drawing
                        print("🖼️ Drawing bbox for \(detection.className) using tightBBox: \(modelRect)")
                        print("    Original model bbox: x=\(detection.x), y=\(detection.y), w=\(detection.width), h=\(detection.height)")
                        
                        let path = Path(mappedRect)
                        
                        // SINGLE THIN GREEN LINE ONLY
                        context.stroke(path, with: .color(.green), lineWidth: 2)
                        
                        // Draw class name label at top-left of bbox with background using resolved Text
                        let label = Text(index == 0 ? "\(detection.className) main" : detection.className)
                            .font(.caption2)
                            .foregroundColor(.white)
                        
                        let resolvedText = context.resolve(label)
                        let textSize = resolvedText.measure(in: size)
                        let textPosition = CGPoint(x: mappedRect.minX + 4, y: mappedRect.minY + 2)
                        let bgRect = CGRect(x: textPosition.x, y: textPosition.y, width: textSize.width, height: textSize.height)
                        
                        context.fill(Path(bgRect), with: .color(.black.opacity(0.7)))
                        context.draw(resolvedText, at: textPosition, anchor: .topLeading)
                    }
                }
                .ignoresSafeArea()
                .allowsHitTesting(false)
            }
            
            // Removed the entire VStack with visibleDetectionNames (top overlay)
            // Slider and other overlays remain unchanged
            
            /*
            VStack {
                // Slider: Less object <-> More object
                if camera.segmentedImage != nil {
                    HStack {
                        Text("Less object")
                            .font(.caption2)
                            .foregroundColor(.white.opacity(0.8))

                        Slider(
                            value: Binding(
                                get: { Double(camera.maskCutoff) },
                                set: { camera.maskCutoff = Float($0) }
                            ),
                            in: 0.1...0.8
                        )

                        Text("More object")
                            .font(.caption2)
                            .foregroundColor(.white.opacity(0.8))
                    }
                    .padding(.horizontal)
                    .padding(.top, 4)
                    .padding(.bottom, 4)
                    .background(Color.black.opacity(0.6))
                    .cornerRadius(10)
                    .padding(.horizontal)
                }

                Spacer()
            }
            */
            
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
                                Text("Capture").font(.caption2)
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
                                Text("Reset").font(.caption2)
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
                    Text("Pinch • Drag")
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
        .onAppear { camera.startSession() }
        .onDisappear { camera.stopSession() }
    }
    
    private func captureFurnitureWithRoom() {
        guard let furniture = camera.segmentedImage, let room = roomImage else { return }
        UIGraphicsBeginImageContextWithOptions(room.size, false, room.scale)
        defer { UIGraphicsEndImageContext() }
        room.draw(at: .zero)
        furniture.draw(in: CGRect(
            x: (room.size.width - furniture.size.width) / 2,
            y: (room.size.height - furniture.size.height) / 2,
            width: furniture.size.width, height: furniture.size.height
        ))
        guard let composite = UIGraphicsGetImageFromCurrentImageContext() else { return }
        PHPhotoLibrary.requestAuthorization(for: .addOnly) { status in
            DispatchQueue.main.async {
                if status == .authorized || status == .limited {
                    PHPhotoLibrary.shared().performChanges({
                        PHAssetChangeRequest.creationRequestForAsset(from: composite)
                    }) { success, _ in
                        DispatchQueue.main.async {
                            if success {
                                self.saveMessage = "Saved!"
                                self.showingSaveSuccess = true
                                DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
                                    self.showingSaveSuccess = false
                                    self.isShowingCamera = false
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

struct DetectionSmarty {
    let x: Float; let y: Float; let width: Float; let height: Float
    let confidence: Float; let classIdx: Int; let className: String
    let maskCoeffs: [Float]
    var tightBBox: CGRect? = nil
}

class FurnitureSegmentationModelSmarty: NSObject, ObservableObject {
    @Published var segmentedImage: UIImage?
    @Published var furnitureOpacity: Double = 0.0
    @Published var isProcessing = false
    @Published var currentFPS: Double = 0.0
    @Published var lastConfidence: Float = 0.0
    @Published var currentBBox: CGRect = .zero
    @Published var currentBBoxes: [CGRect] = []
    
    // Added published property to hold all mask-filtered detections
    @Published var currentDetections: [DetectionSmarty] = []

    // The slider is disabled; maskCutoff is now fixed at 0.0
    @Published var maskCutoff: Float = 0.3
    
    // Names of detections for UI (e.g. ["bed", "chair", "couch"])
    @Published var visibleDetectionNames: [String] = []
    
    private let session = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let videoQueue = DispatchQueue(label: "yoloeVideo", qos: .userInitiated)
    private let detectionQueue = DispatchQueue(label: "yoloeDetection", qos: .userInitiated)
    private var mlModel: MLModel?
    private let context = CIContext(options: [.workingColorSpace: CGColorSpaceCreateDeviceRGB()])
    
    private let furnitureClasses: [Int: String] = [
        132: "armchair", 213: "baby seat", 276: "bar stool", 332: "bathroom cabinet",
        334: "bathroom mirror", 352: "beach chair", 364: "bean bag chair", 375: "bed",
        376: "bedcover", 377: "bed frame", 382: "bedside lamp", 402: "bench",
        429: "billiard table", 517: "bookshelf", 567: "chest", 632: "bunk bed",
        636: "bureau", 670: "cabinet", 679: "cake stand", 714: "canopy bed",
        733: "car seat", 821: "chair", 823: "daybed", 834: "changing table",
        977: "closet", 996: "coatrack", 1006: "cocktail table", 1060: "computer chair",
        1061: "computer desk", 1137: "infant bed", 1141: "couch", 1143: "counter",
        1144: "counter top", 1270: "day bed", 1301: "table", 1302: "table lamp",
        1303: "desktop", 1325: "dinning table", 1364: "dog bed", 1396: "drawer",
        1405: "dresser", 1476: "electric chair", 1503: "side table", 1602: "feeding chair",
        1624: "file cabinet", 1721: "folding chair", 1733: "food stand", 1750: "footrest",
        1801: "fruit stand", 1816: "futon", 1885: "glass table", 2141: "hospital bed",
        2193: "ice shelf", 2219: "information desk", 2247: "island", 2318: "kitchen cabinet",
        2319: "kitchen counter", 2322: "kitchen island", 2324: "kitchen table",
        2499: "loveseat", 2599: "mattress", 2614: "medicine cabinet", 2654: "mirror",
        2754: "music stool", 2802: "nightstand", 2834: "office chair", 2836: "office desk",
        2939: "park bench", 3024: "church bench", 3045: "picnic table",
        3061: "table tennis table", 3145: "poker table", 3423: "rocking chair",
        3449: "round table", 3584: "seat", 3621: "shelf", 3678: "side cabinet",
        3812: "spice rack", 3862: "stand", 3888: "step stool", 3909: "stool",
        4004: "supermarket shelf", 4041: "swivel chair", 4055: "table top",
        4056: "tablecloth", 4179: "toilet seat", 4213: "towel bar", 4294: "tv cabinet",
        4331: "vanity", 4473: "wheelchair", 4506: "window seat", 4513: "wine cabinet",
        4516: "wine rack", 4545: "workbench", 4564: "writing desk"
    ]
    
    private var lastProcessTime = Date()
    private let processInterval: TimeInterval = 0.05
    private var frameCount = 0
    private var fpsStartTime = Date()
    
    private func sigmoid(_ x: Float) -> Float { 1.0 / (1.0 + exp(-x)) }
    
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
            self.currentBBoxes = []
            self.visibleDetectionNames = []
            self.currentDetections = []
        }
    }
    
    private func loadYOLOModel() {
        do {
            let config = MLModelConfiguration()
            config.computeUnits = .all
            for ext in ["mlmodelc", "mlpackage"] {
                if let url = Bundle.main.url(forResource: "yoloe-11l-seg-pf", withExtension: ext) {
                    mlModel = try MLModel(contentsOf: url, configuration: config)
                    print("✅ Model loaded")
                    return
                }
            }
        } catch { }
    }
    
    private func setupCamera() {
        session.sessionPreset = .hd1280x720
        // Try to use the 0.5x ultra-wide camera if available, else fallback to wide angle
        let device = AVCaptureDevice.default(.builtInUltraWideCamera, for: .video, position: .back) ??
                     AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back)
        guard let device = device else { return }
        do {
            let input = try AVCaptureDeviceInput(device: device)
            if session.canAddInput(input) { session.addInput(input) }
            videoOutput.setSampleBufferDelegate(self, queue: videoQueue)
            videoOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
            videoOutput.alwaysDiscardsLateVideoFrames = true
            if session.canAddOutput(videoOutput) {
                session.addOutput(videoOutput)
                videoOutput.connection(with: .video)?.videoRotationAngle = 90
            }
        } catch { }
    }
    
    func startSession() {
        if !session.isRunning {
            DispatchQueue.global(qos: .background).async {
                self.session.startRunning()
                DispatchQueue.main.async { self.fpsStartTime = Date() }
            }
        }
    }
    
    func stopSession() { if session.isRunning { session.stopRunning() } }
    
    private func updateFPS() {
        frameCount += 1
        let elapsed = Date().timeIntervalSince(fpsStartTime)
        if elapsed > 1.0 {
            DispatchQueue.main.async { self.currentFPS = Double(self.frameCount) / elapsed }
            frameCount = 0
            fpsStartTime = Date()
        }
    }
    
    private func processWithYOLO(pixelBuffer: CVPixelBuffer) {
        guard let model = mlModel else { return }
        let now = Date()
        guard now.timeIntervalSince(lastProcessTime) >= processInterval, !isProcessing else { return }
        lastProcessTime = now
        updateFPS()
        DispatchQueue.main.async { self.isProcessing = true }
        
        detectionQueue.async { [weak self] in
            guard let self = self,
                  let resized = self.resizePixelBuffer(pixelBuffer, width: 640, height: 640),
                  let inputArray = self.pixelBufferToMLMultiArray(resized),
                  let inputProvider = try? MLDictionaryFeatureProvider(dictionary: ["image": inputArray]),
                  let output = try? model.prediction(from: inputProvider),
                  let detectionsArray = output.featureValue(for: "var_2421")?.multiArrayValue,
//                  let detectionsArray = output.featureValue(for: "var_1432")?.multiArrayValue,
                  let prototypesArray = output.featureValue(for: "p")?.multiArrayValue else {
                DispatchQueue.main.async { self?.isProcessing = false }
                return
            }
            if SEGMENT_DEBUG_SAVE_IMAGES {
                print("\n🔬 ========== RAW YOLO OUTPUT ==========")
                print("Detections shape: \(detectionsArray.shape)")
                print("Prototypes shape: \(prototypesArray.shape)")
                print("\nFirst 3 anchors raw data:")
            }
            
            print("========================================\n")
            
            self.processDirectMultiMask(detectionsArray, prototypes: prototypesArray, originalImage: pixelBuffer)
        }
    }
    
    private func pixelBufferToMLMultiArray(_ pixelBuffer: CVPixelBuffer) -> MLMultiArray? {
        guard let array = try? MLMultiArray(shape: [1, 3, 640, 640], dataType: .float16) else { return nil }
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else { return nil }
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let buffer = baseAddress.assumingMemoryBound(to: UInt8.self)
        
        for y in 0..<640 {
            for x in 0..<640 {
                let idx = y * bytesPerRow + x * 4
                array[[0, 0, y, x] as [NSNumber]] = NSNumber(value: Float(buffer[idx + 2]) / 255.0)
                array[[0, 1, y, x] as [NSNumber]] = NSNumber(value: Float(buffer[idx + 1]) / 255.0)
                array[[0, 2, y, x] as [NSNumber]] = NSNumber(value: Float(buffer[idx]) / 255.0)
            }
        }
        return array
    }
    
    private func resizePixelBuffer(_ pixelBuffer: CVPixelBuffer, width: Int, height: Int) -> CVPixelBuffer? {
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let scaleX = CGFloat(width) / CGFloat(CVPixelBufferGetWidth(pixelBuffer))
        let scaleY = CGFloat(height) / CGFloat(CVPixelBufferGetHeight(pixelBuffer))
        var out: CVPixelBuffer?
        CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA, nil, &out)
        guard let dst = out else { return nil }
        CIContext().render(ciImage.transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY)), to: dst)
        return dst
    }
    
    private func extractDetections(from detections: MLMultiArray) -> [DetectionSmarty] {
        var all: [DetectionSmarty] = []
        
        print("\n🔍 ========== ALL DETECTIONS EXTRACTED ==========")
        
        for anchor in 0..<detections.shape[2].intValue {
            let x = detections[[0, 0, anchor] as [NSNumber]].floatValue
            let y = detections[[0, 1, anchor] as [NSNumber]].floatValue
            let w = detections[[0, 2, anchor] as [NSNumber]].floatValue
            let h = detections[[0, 3, anchor] as [NSNumber]].floatValue
            
            for (classIdx, className) in furnitureClasses {
                let conf = detections[[0, 4 + classIdx, anchor] as [NSNumber]].floatValue
                if conf > 0.3 {
                    var coeffs = [Float](repeating: 0, count: 32)
                    for i in 0..<32 {
                        coeffs[i] = detections[[0, 4 + 4585 + i, anchor] as [NSNumber]].floatValue
                    }
                    all.append(DetectionSmarty(
                        x: x, y: y, width: w, height: h,
                        confidence: conf, classIdx: classIdx, className: className,
                        maskCoeffs: coeffs
                    ))
                }
            }
        }
        
        let grouped = Dictionary(grouping: all) { $0.className }
        print("\n📊 DETECTION SUMMARY:")
        print("Total detections: \(all.count)")
        print("Unique classes detected:")
        for (className, detections) in grouped.sorted(by: { $0.value.count > $1.value.count }) {
            let confidences = detections.map { Int($0.confidence * 100) }
            print("  - \(className): \(detections.count) detection(s), conf: \(confidences)%")
        }
        print("================================================\n")
        
        return all
    }
    
    // Simple diverse detection selection (not used for UI names right now, but kept)
    private func getDiverseDetections(from detections: [DetectionSmarty], maxCount: Int) -> [DetectionSmarty] {
        var selected: [DetectionSmarty] = []
        var seenClasses = Set<Int>()
        
        let sorted = detections.sorted { $0.confidence > $1.confidence }
        
        for detection in sorted {
            if !seenClasses.contains(detection.classIdx) {
                selected.append(detection)
                seenClasses.insert(detection.classIdx)
                if selected.count >= maxCount { break }
            }
        }
        
        print("📊 [DIVERSE] Selected \(selected.count) diverse detections:")
        for det in selected {
            print("   - \(det.className) @ \(Int(det.confidence * 100))%")
        }
        
        return selected
    }
    
    // Main processing with multi-mask
    private func processDirectMultiMask(_ detections: MLMultiArray, prototypes: MLMultiArray, originalImage: CVPixelBuffer) {
        print("\n📱 ==================== DIRECT MULTI-MASK PROCESSING ====================")
        
        if SEGMENT_DEBUG_SAVE_IMAGES {
            saveDebugImage(pixelBuffer: originalImage, stage: "1_original")
        }

        let allDetections = extractDetections(from: detections)
        print("📊 [DETECTION] Extracted \(allDetections.count) raw detections")
        
        if SEGMENT_DEBUG_SAVE_IMAGES {
            print("\n🔍 ========== ALL DETECTIONS VALUES ==========")
            for (index, detection) in allDetections.enumerated() {
//                print("Det #\(index): \(detection.className) (\(detection.classIdx)) | Conf: \(String(format: \"%.3f\", detection.confidence)) (\(Int(detection.confidence * 100))%)")
            }
            print("============================================\n")
        }
        
        let hierarchicalDetections = applyHierarchicalNMS(detections: allDetections, iouThreshold: 1.0)
        print("📊 [H-NMS] Kept \(hierarchicalDetections.count) detections after hierarchical NMS")
        
        let maskFilteredDetections = applyMaskIoU(
            detections: hierarchicalDetections,
            iouThreshold: 0.2,
            prototypes: prototypes
        )
        
        print("📊 [MASK-FILTERED] Total kept: \(maskFilteredDetections.count) detections")

        // 🔧 BBOX: use a fixed, higher cutoff for tight bbox (independent of slider)
        let bboxCutoff: Float = 0.6
        var detectionsWithTightBBox: [DetectionSmarty] = []
        
        for detection in maskFilteredDetections {
            // Generate individual mask in mask space (160x160)
            var individualMask = [Float](repeating: 0, count: 160 * 160)
            for y in 0..<160 {
                for x in 0..<160 {
                    var sum: Float = 0
                    for c in 0..<32 {
                        sum += detection.maskCoeffs[c] * prototypes[[0, c, y, x] as [NSNumber]].floatValue
                    }
                    // ✅ Apply sigmoid BEFORE storing
                    let sigmoidValue = 1.0 / (1.0 + exp(-sum))
                    individualMask[y * 160 + x] = sigmoidValue
                }
            }
            
            // Compute tight bbox in mask space (160x160) based on bboxCutoff
            var minX = 160
            var minY = 160
            var maxX = -1
            var maxY = -1
            var pixelCount = 0
            
            for y in 0..<160 {
                for x in 0..<160 {
                    if individualMask[y * 160 + x] >= bboxCutoff {
                        if x < minX { minX = x }
                        if x > maxX { maxX = x }
                        if y < minY { minY = y }
                        if y > maxY { maxY = y }
                        pixelCount += 1
                    }
                }
            }
            
            var tightBBox: CGRect? = nil
            if maxX >= minX && maxY >= minY && pixelCount > 10 { // ✅ Require at least 10 pixels
                // Convert tight bbox from mask space (160x160) to model space (640x640)
                let scale: CGFloat = 4.0  // 640 / 160 = 4
                
                // ❌ removed extra padding – let bbox hug the mask
                let originX = max<CGFloat>(0, CGFloat(minX) * scale)
                let originY = max<CGFloat>(0, CGFloat(minY) * scale)
                let width = min<CGFloat>(640 - originX, CGFloat(maxX - minX + 1) * scale)
                let height = min<CGFloat>(640 - originY, CGFloat(maxY - minY + 1) * scale)
                
                tightBBox = CGRect(x: originX, y: originY, width: width, height: height)
            }
            
            // ✅ IMPROVED LOGGING
            print("🔎 Detection \(detection.className) @ \(Int(detection.confidence * 100))%")
            print("    Mask space: minX=\(minX), maxX=\(maxX), minY=\(minY), maxY=\(maxY), pixels=\(pixelCount)")
            
            if let tight = tightBBox {
                print("    Tight bbox (640x640): \(tight)")
                
                // Convert YOLO center bbox to corner format for comparison
                let yoloMinX = detection.x - detection.width / 2
                let yoloMinY = detection.y - detection.height / 2
                let yoloMaxX = detection.x + detection.width / 2
                let yoloMaxY = detection.y + detection.height / 2
                print("    YOLO bbox (640x640): x=[\(Int(yoloMinX))-\(Int(yoloMaxX))], y=[\(Int(yoloMinY))-\(Int(yoloMaxY))], w=\(Int(detection.width)), h=\(Int(detection.height))")
                
                // Calculate how much tighter
                let tightArea = tight.width * tight.height
                let yoloArea = CGFloat(detection.width * detection.height)
                let areaRatio = (tightArea / yoloArea) * 100
                print("    Tight is \(Int(areaRatio))% of YOLO bbox area")
            } else {
                print("    No valid tightBBox (mask too small or sparse)")
            }
            
            var detectionWithBBox = detection
            detectionWithBBox.tightBBox = tightBBox
            
            if SEGMENT_DEBUG_SAVE_IMAGES && tightBBox != nil {
                let stageName = "final_filtered_\(detection.className)_\(Int(detection.confidence * 100))pct"
                saveMaskAsImage(mask: individualMask, stage: stageName)
            }
            
            detectionsWithTightBBox.append(detectionWithBBox)
        }

        if SEGMENT_DEBUG_SAVE_IMAGES {
            print("📊 [SAVED] Generated and saved \(detectionsWithTightBBox.count) final filtered masks")
        }
        
        // Update visible detection names for UI (top 3 by confidence)
        let sortedForNames = detectionsWithTightBBox.sorted { $0.confidence > $1.confidence }
        let labelNames = sortedForNames.prefix(3).map { $0.className }
        DispatchQueue.main.async {
            self.visibleDetectionNames = labelNames
            self.currentDetections = detectionsWithTightBBox
        }
        
        guard !detectionsWithTightBBox.isEmpty else {
            print("❌ [DETECTION] No valid detections found after mask filtering")
            DispatchQueue.main.async {
                self.isProcessing = false
                self.segmentedImage = nil
                self.furnitureOpacity = 0.0
                self.lastConfidence = 0.0
                self.currentBBox = .zero
                self.currentBBoxes = []
                self.visibleDetectionNames = []
                self.currentDetections = []
            }
            return
        }
        
        let best = detectionsWithTightBBox.first!
        print("✅ [BEST] Primary: \(best.className) @ \(Int(best.confidence * 100))%")
        print("   Position: (\(Int(best.x)), \(Int(best.y))), Size: \(Int(best.width))x\(Int(best.height))")
        
        let bbox = best.tightBBox ?? CGRect(
            x: CGFloat(best.x - best.width / 2),
            y: CGFloat(best.y - best.height / 2),
            width: CGFloat(best.width),
            height: CGFloat(best.height)
        )
        
        // Calculate all bboxes for maskFilteredDetections using tightBBox only
        let allBBoxes: [CGRect] = detectionsWithTightBBox.compactMap { $0.tightBBox }
        
        DispatchQueue.main.async {
            self.currentBBox = bbox
            self.currentBBoxes = allBBoxes
            self.lastConfidence = best.confidence
        }
        
        print("\n🎨 ========== GENERATING COLORED MASKS COMPOSITE ==========")
        
        // Generate combined colored masks image
        generateColoredMasksComposite(detections: detectionsWithTightBBox, prototypes: prototypes, originalImage: originalImage)
    }
    
    private func generateColoredMasksComposite(detections: [DetectionSmarty], prototypes: MLMultiArray, originalImage: CVPixelBuffer) {
        // Palette colors for masks (RGBA), semi-transparent alpha = 0.5
        let palette: [(r: Float, g: Float, b: Float, a: Float)] = [
            (1.0, 0.0, 0.0, 0.5),   // Red
            (0.0, 1.0, 0.0, 0.5),   // Green
            (0.0, 0.0, 1.0, 0.5),   // Blue
            (1.0, 1.0, 0.0, 0.5),   // Yellow
            (1.0, 0.0, 1.0, 0.5),   // Magenta
            (0.0, 1.0, 1.0, 0.5),   // Cyan
            (1.0, 0.5, 0.0, 0.5),   // Orange
            (0.5, 0.0, 1.0, 0.5),   // Purple
            (0.0, 0.5, 0.5, 0.5),   // Teal
            (0.5, 0.5, 0.5, 0.5)    // Gray
        ]
        
        let shape = prototypes.shape.map { $0.intValue }      // [1, 32, 160, 160]
        let C = shape[1]
        let Hp = shape[2]
        let Wp = shape[3]
        let spatial = Hp * Wp
        
        var protoMatrix = [Float](repeating: 0, count: C * spatial)
        
        if prototypes.dataType == .float32 {
            let srcBase = prototypes.dataPointer.assumingMemoryBound(to: Float.self)
            for c in 0..<C {
                let srcChannelOffset = c * Hp * Wp
                let dstChannelOffset = c * spatial
                for y in 0..<Hp {
                    let rowOffset = y * Wp
                    for x in 0..<Wp {
                        let idx = rowOffset + x
                        let srcIndex = srcChannelOffset + idx
                        let dstIndex = dstChannelOffset + idx
                        protoMatrix[dstIndex] = srcBase[srcIndex]
                    }
                }
            }
        } else {
            for c in 0..<C {
                for y in 0..<Hp {
                    for x in 0..<Wp {
                        let val = prototypes[[0, c, y, x] as [NSNumber]].floatValue
                        let dstIndex = c * spatial + (y * Wp + x)
                        protoMatrix[dstIndex] = val
                    }
                }
            }
        }
        
        let maskW = 160
        let maskH = 160
        let maskSize = maskW * maskH
        
        // Generate individual masks and color them
        var masks: [[Float]] = []
        masks.reserveCapacity(detections.count)
        
        for detection in detections {
            var mask = [Float](repeating: 0, count: spatial)
            
            vDSP_mmul(
                detection.maskCoeffs, 1,
                protoMatrix, 1,
                &mask, 1,
                1,
                vDSP_Length(spatial),
                vDSP_Length(C)
            )
            
            for i in 0..<spatial {
                let v = mask[i]
                mask[i] = 1.0 / (1.0 + exp(-v))
            }
            
            masks.append(mask)
        }
        
        // Prepare to composite colored masks on original image pixels
        autoreleasepool {
            let ciImage = CIImage(cvPixelBuffer: originalImage)
            let width = CVPixelBufferGetWidth(originalImage)
            let height = CVPixelBufferGetHeight(originalImage)
            
            guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent),
                  let ctx = CGContext(
                        data: nil,
                        width: width,
                        height: height,
                        bitsPerComponent: 8,
                        bytesPerRow: width * 4,
                        space: CGColorSpaceCreateDeviceRGB(),
                        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
                  ),
                  let data = ctx.data else {
                DispatchQueue.main.async { self.isProcessing = false }
                return
            }
            
            ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
            let pixels = data.bindMemory(to: UInt8.self, capacity: width * height * 4)
            
            let cutoff = self.maskCutoff
            
            // Render only mask areas as opaque, everything else is fully transparent.
            for py in 0..<height {
                for px in 0..<width {
                    let idx = (py * width + px) * 4
                    
                    // Find the mask with highest mask value at this pixel if above cutoff
                    var maxMaskValue: Float = 0
                    var maxMaskIndex: Int? = nil
                    
                    // Convert pixel to mask coordinate space
                    let mx = Float(px) * Float(maskW) / Float(width)
                    let my = Float(py) * Float(maskH) / Float(height)
                    let x0 = Int(mx)
                    let y0 = Int(my)
                    
                    if x0 >= 0 && x0 < maskW && y0 >= 0 && y0 < maskH {
                        for (i, mask) in masks.enumerated() {
                            let maskValue = mask[y0 * maskW + x0]
                            if maskValue >= cutoff && maskValue > maxMaskValue {
                                maxMaskValue = maskValue
                                maxMaskIndex = i
                            }
                        }
                    }
                    
                    if let maxIndex = maxMaskIndex {
                        // Use only the color of the mask with highest mask value (opaque)
                        let color = palette[maxIndex % palette.count]
                        pixels[idx] = UInt8(max(0, min(255, color.r * 255.0)))
                        pixels[idx + 1] = UInt8(max(0, min(255, color.g * 255.0)))
                        pixels[idx + 2] = UInt8(max(0, min(255, color.b * 255.0)))
                        pixels[idx + 3] = 255
                    } else {
                        // No mask present, fully transparent pixel
                        pixels[idx] = 0
                        pixels[idx + 1] = 0
                        pixels[idx + 2] = 0
                        pixels[idx + 3] = 0
                    }
                }
            }
            
            if let outImage = ctx.makeImage() {
                DispatchQueue.main.async {
                    self.segmentedImage = UIImage(cgImage: outImage, scale: 1.0, orientation: .up)
                    withAnimation(.easeIn(duration: 0.3)) { self.furnitureOpacity = 1.0 }
                    self.isProcessing = false
                }
            } else {
                DispatchQueue.main.async { self.isProcessing = false }
            }
        }
    }
    
    private func applyMaskIoU(
        detections: [DetectionSmarty],
        iouThreshold: Float,
        prototypes: MLMultiArray
    ) -> [DetectionSmarty] {
        guard !detections.isEmpty else { return [] }
        
        let sorted = detections.sorted { $0.confidence > $1.confidence }
        
        print("\n🔍 Mask-NMS (pure mask IoU, no class logic):")
        
        let shape = prototypes.shape.map { $0.intValue }      // [1, 32, 160, 160]
        let C = shape[1]
        let Hp = shape[2]
        let Wp = shape[3]
        let spatial = Hp * Wp
        
        var protoMatrix = [Float](repeating: 0, count: C * spatial)
        
        if prototypes.dataType == .float32 {
            let srcBase = prototypes.dataPointer.assumingMemoryBound(to: Float.self)
            for c in 0..<C {
                let srcChannelOffset = c * Hp * Wp
                let dstChannelOffset = c * spatial
                for y in 0..<Hp {
                    let rowOffset = y * Wp
                    for x in 0..<Wp {
                        let idx = rowOffset + x
                        let srcIndex = srcChannelOffset + idx
                        let dstIndex = dstChannelOffset + idx
                        protoMatrix[dstIndex] = srcBase[srcIndex]
                    }
                }
            }
        } else {
            for c in 0..<C {
                for y in 0..<Hp {
                    for x in 0..<Wp {
                        let val = prototypes[[0, c, y, x] as [NSNumber]].floatValue
                        let dstIndex = c * spatial + (y * Wp + x)
                        protoMatrix[dstIndex] = val
                    }
                }
            }
        }
        
        var masks: [[Float]] = []
        masks.reserveCapacity(sorted.count)
        
        for det in sorted {
            var mask = [Float](repeating: 0, count: spatial)
            
            vDSP_mmul(
                det.maskCoeffs, 1,
                protoMatrix, 1,
                &mask, 1,
                1,
                vDSP_Length(spatial),
                vDSP_Length(C)
            )
            
            for i in 0..<spatial {
                let v = mask[i]
                mask[i] = 1.0 / (1.0 + exp(-v))
            }
            
            masks.append(mask)
        }
        
        var kept: [DetectionSmarty] = []
        var keptMasks: [[Float]] = []
        
        for (i, det) in sorted.enumerated() {
            let candidateMask = masks[i]
            
            var mergedWithExisting = false
            var mergeTargetIndex = -1
            
            for (existingIndex, existingMask) in keptMasks.enumerated() {
                let iou = calculateMaskIoU(mask1: candidateMask, mask2: existingMask)
                if iou >= iouThreshold {
                    mergedWithExisting = true
                    mergeTargetIndex = existingIndex
                    print("🔄 MERGING (IoU \(Int(iou * 100))%) \(det.className) @ \(Int(det.confidence * 100))% with existing \(kept[mergeTargetIndex].className) @ \(Int(kept[mergeTargetIndex].confidence * 100))%")
                    break
                }
            }
            
            if mergedWithExisting && mergeTargetIndex >= 0 {
                var mergedMask = keptMasks[mergeTargetIndex]
                for pixelIndex in 0..<candidateMask.count {
                    mergedMask[pixelIndex] = max(mergedMask[pixelIndex], candidateMask[pixelIndex])
                }
                
                keptMasks[mergeTargetIndex] = mergedMask
                
                if det.confidence > kept[mergeTargetIndex].confidence {
                    print("   → Replacing detection info with higher confidence: \(kept[mergeTargetIndex].className) @ \(Int(kept[mergeTargetIndex].confidence * 100))% → \(det.className) @ \(Int(det.confidence * 100))%")
                    kept[mergeTargetIndex] = det
                } else {
                    print("   → Keeping existing detection info: \(kept[mergeTargetIndex].className) @ \(Int(kept[mergeTargetIndex].confidence * 100))%")
                }
                
                if SEGMENT_DEBUG_SAVE_IMAGES {
                    let stageName = "merged_\(mergeTargetIndex + 1)_\(kept[mergeTargetIndex].className)_with_\(det.className)"
                    saveMaskAsImage(mask: mergedMask, stage: stageName)
                }
            } else {
                kept.append(det)
                keptMasks.append(candidateMask)
                print("✅ KEEP \(det.className) @ \(Int(det.confidence * 100))%")
                
                if SEGMENT_DEBUG_SAVE_IMAGES {
                    let stageName = "kept_\(kept.count)_\(det.className)"
                    saveMaskAsImage(mask: candidateMask, stage: stageName)
                }
            }
        }
        
        print("Mask-NMS: \(sorted.count) → \(kept.count) unique masks (by IoU)")
        return kept
    }
    
    private func calculateMaskIoU(mask1: [Float], mask2: [Float], eps: Float = 1e-7) -> Float {
        let n = min(mask1.count, mask2.count)
        guard n > 0 else { return 0 }
        
        var intersection: Float = 0
        vDSP_dotpr(mask1, 1, mask2, 1, &intersection, vDSP_Length(n))
        
        var sum1: Float = 0
        var sum2: Float = 0
        vDSP_sve(mask1, 1, &sum1, vDSP_Length(n))
        vDSP_sve(mask2, 1, &sum2, vDSP_Length(n))
        
        let union = sum1 + sum2 - intersection
        return intersection / (union + eps)
    }
    
    private func applyPostProcessingAndMask(mask: [Float], best: DetectionSmarty, to originalImage: CVPixelBuffer, stage: String) {
        print("🎨 [APPLY] Applying final enhanced mask to image")
        // This method is no longer used (replaced by generateColoredMasksComposite)
        applyMaskToImage(mask: mask, to: originalImage)
        
        print("✅ ==================== FRAME COMPLETE ====================\n")
    }
    
    private func applyMaskToImage(mask: [Float], to pixelBuffer: CVPixelBuffer) {
        autoreleasepool {
            let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
            let width = CVPixelBufferGetWidth(pixelBuffer)
            let height = CVPixelBufferGetHeight(pixelBuffer)

            let cutoff = self.maskCutoff

            guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent),
                  let ctx = CGContext(
                        data: nil,
                        width: width,
                        height: height,
                        bitsPerComponent: 8,
                        bytesPerRow: width * 4,
                        space: CGColorSpaceCreateDeviceRGB(),
                        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
                  ),
                  let data = ctx.data else {
                DispatchQueue.main.async { self.isProcessing = false }
                return
            }

            ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
            let pixels = data.bindMemory(to: UInt8.self, capacity: width * height * 4)

            for py in 0..<height {
                for px in 0..<width {
                    let idx = (py * width + px) * 4
                    let mx = Float(px) * 160.0 / Float(width)
                    let my = Float(py) * 160.0 / Float(height)
                    let x0 = Int(mx), y0 = Int(my)

                    guard x0 >= 0 && x0 < 160 && y0 >= 0 && y0 < 160 else {
                        pixels[idx + 3] = 0
                        continue
                    }

                    let maskValue = mask[y0 * 160 + x0]

                    if maskValue >= cutoff {
                        pixels[idx + 3] = 255
                    } else {
                        pixels[idx + 3] = 0
                    }
                }
            }

            if let outImage = ctx.makeImage() {
                DispatchQueue.main.async {
                    self.segmentedImage = UIImage(cgImage: outImage, scale: 1.0, orientation: .up)
                    withAnimation(.easeIn(duration: 0.3)) { self.furnitureOpacity = 1.0 }
                    self.isProcessing = false
                }
            } else {
                DispatchQueue.main.async { self.isProcessing = false }
            }
        }
    }
    
    private func binaryToFloat(_ binary: [[UInt8]]) -> [Float] {
        var result = [Float](repeating: 0, count: 160 * 160)
        for y in 0..<160 {
            for x in 0..<160 {
                result[y * 160 + x] = Float(binary[y][x])
            }
        }
        return result
    }
    
    private func saveDebugImage(pixelBuffer: CVPixelBuffer, stage: String) {
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        print("saveDebugImage: pixelBuffer size = \(width) x \(height)")
        guard width > 0 && height > 0 else {
            print("Skipping saveDebugImage due to zero size")
            return
        }
        guard CVPixelBufferGetBaseAddress(pixelBuffer) != nil else {
            print("Skipping saveDebugImage: pixelBuffer has nil baseAddress")
            return
        }
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        print("saveDebugImage: CIImage extent = \(ciImage.extent), pixelBuffer base address = \(CVPixelBufferGetBaseAddress(pixelBuffer) as Optional)")
        guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else { return }
        let uiImage = UIImage(cgImage: cgImage)
        UIImageWriteToSavedPhotosAlbum(uiImage, nil, nil, nil)
        print("💾 [SAVE] Saved: \(stage)")
    }

    private func saveDebugImageWithBBox(pixelBuffer: CVPixelBuffer, bbox: DetectionSmarty, stage: String) {
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        print("saveDebugImageWithBBox: pixelBuffer size = \(width) x \(height)")
        guard width > 0 && height > 0 else {
            print("Skipping saveDebugImageWithBBox due to zero size")
            return
        }
        guard CVPixelBufferGetBaseAddress(pixelBuffer) != nil else {
            print("Skipping saveDebugImageWithBBox: pixelBuffer has nil baseAddress")
            return
        }
        
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        print("saveDebugImageWithBBox: CIImage extent = \(ciImage.extent), pixelBuffer base address = \(CVPixelBufferGetBaseAddress(pixelBuffer) as Optional)")
        guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else { return }
        
        UIGraphicsBeginImageContextWithOptions(CGSize(width: width, height: height), false, 1.0)
        guard let ctx = UIGraphicsGetCurrentContext() else { return }
        
        ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        let scale = Float(width) / 640.0
        let bboxRect = CGRect(
            x: CGFloat((bbox.x - bbox.width/2) * scale),
            y: CGFloat((bbox.y - bbox.height/2) * scale),
            width: CGFloat(bbox.width * scale),
            height: CGFloat(bbox.height * scale)
        )
        
        if SEGMENT_DEBUG_SAVE_IMAGES {
            ctx.setStrokeColor(UIColor.green.cgColor)
            ctx.setLineWidth(3)
            ctx.stroke(bboxRect)
        }
        
        guard let finalImage = UIGraphicsGetImageFromCurrentImageContext() else { return }
        UIGraphicsEndImageContext()
        
        UIImageWriteToSavedPhotosAlbum(finalImage, nil, nil, nil)
        print("💾 [SAVE] Saved: \(stage)")
    }
    
    private func applyHierarchicalNMS(detections: [DetectionSmarty], iouThreshold: Float) -> [DetectionSmarty] {
        guard !detections.isEmpty else { return [] }
        
        var kept: [DetectionSmarty] = []
        var suppressed = Set<Int>()
        let sorted = detections.sorted { $0.confidence > $1.confidence }
        
        print("\n🔍 Hierarchical NMS Processing:")
        print("Input: \(sorted.count) detections")
        
        for (i, det) in sorted.enumerated() {
            if suppressed.contains(i) { continue }
            
            var shouldSuppress = false
            
            for existing in kept {
                let iou = calculateIoU(det1: det, det2: existing)
                
                if iou > iouThreshold {
                    shouldSuppress = true
                    if SEGMENT_DEBUG_SAVE_IMAGES {
                        print("❌ Suppressed duplicate: \(det.className) @ \(Int(det.confidence * 100))%")
                    }
                    break
                }
            }
            
            if !shouldSuppress {
                kept.append(det)
                print("✅ KEPT: \(det.className) @ \(Int(det.confidence * 100))%")
                suppressed.insert(i)
            }
        }
        
        print("Hierarchical NMS: \(sorted.count) → \(kept.count) detections")
        return kept
    }

    private func calculateIoU(det1: DetectionSmarty, det2: DetectionSmarty) -> Float {
        let x1 = max(det1.x - det1.width/2, det2.x - det2.width/2)
        let y1 = max(det1.y - det1.height/2, det2.y - det2.height/2)
        let x2 = min(det1.x + det1.width/2, det2.x + det2.width/2)
        let y2 = min(det1.y + det1.height/2, det2.y + det2.height/2)
        
        let intersection = max(0, x2 - x1) * max(0, y2 - y1)
        let area1 = det1.width * det1.height
        let area2 = det2.width * det2.height
        let union = area1 + area2 - intersection
        
        return union > 0 ? intersection / union : 0
    }

    private func saveMaskAsImage(mask: [Float], stage: String) {
        UIGraphicsBeginImageContextWithOptions(CGSize(width: 160, height: 160), false, 1.0)
        guard let ctx = UIGraphicsGetCurrentContext() else { return }
        
        for y in 0..<160 {
            for x in 0..<160 {
                let value = mask[y * 160 + x]
                let gray = CGFloat(value)
                ctx.setFillColor(UIColor(white: gray, alpha: 1.0).cgColor)
                ctx.fill(CGRect(x: x, y: y, width: 1, height: 1))
            }
        }
        
        guard let maskImage = UIGraphicsGetImageFromCurrentImageContext() else { return }
        UIGraphicsEndImageContext()
        
        UIImageWriteToSavedPhotosAlbum(maskImage, nil, nil, nil)
        print("💾 [SAVE] Saved mask: \(stage)")
    }
}

extension FurnitureSegmentationModelSmarty: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        processWithYOLO(pixelBuffer: pixelBuffer)
    }
}

