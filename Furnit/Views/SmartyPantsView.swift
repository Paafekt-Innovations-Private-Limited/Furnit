import SwiftUI
import AVFoundation
import CoreML
import CoreImage
import Photos
import Accelerate

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
            
            if camera.currentBBox != .zero && camera.segmentedImage != nil {
                Canvas { context, size in
                    let rect = Path(camera.currentBBox)
                    
                    // LIVE GREEN DEBUG LINES - Highly visible
                    context.stroke(rect, with: .color(.green.opacity(0.9)), lineWidth: 8)  // Thick green outline
                    context.stroke(rect, with: .color(.green), lineWidth: 4)  // Bright green core
                    context.stroke(rect, with: .color(.white.opacity(0.8)), lineWidth: 1)  // White inner line for contrast
                    
                    // Original blue lines (background)
                    context.stroke(rect, with: .color(.blue.opacity(0.2)), lineWidth: 6)
                    context.stroke(rect, with: .color(.blue.opacity(0.4)), lineWidth: 3)
                }
                .ignoresSafeArea()
                .allowsHitTesting(false)
            }
            
            VStack {
                HStack {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("FPS: \(camera.currentFPS, specifier: "%.1f")")
                        if camera.lastConfidence > 0 {
                            Text("\(Int(camera.lastConfidence * 100))%")
                        }
                        Text("MULTI-MASK")
                            .font(.caption2)
                            .foregroundColor(.white)
                    }
                    .font(.caption)
                    .foregroundColor(.white)
                    .padding(6)
                    .background(Color.black.opacity(0.7))
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
}

class FurnitureSegmentationModelSmarty: NSObject, ObservableObject {
    @Published var segmentedImage: UIImage?
    @Published var furnitureOpacity: Double = 0.0
    @Published var isProcessing = false
    @Published var currentFPS: Double = 0.0
    @Published var lastConfidence: Float = 0.0
    @Published var currentBBox: CGRect = .zero
    
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
    private let processInterval: TimeInterval = 0.1
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
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else { return }
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
                  let prototypesArray = output.featureValue(for: "p")?.multiArrayValue else {
                DispatchQueue.main.async { self?.isProcessing = false }
                return
            }
            
            // PRINT RAW YOLO OUTPUT INFO
            print("\n🔬 ========== RAW YOLO OUTPUT ==========")
            print("Detections shape: \(detectionsArray.shape)")
            print("Prototypes shape: \(prototypesArray.shape)")
            
            print("\nFirst 3 anchors raw data:")
            for anchor in 0..<min(3, detectionsArray.shape[2].intValue) {
                let x = detectionsArray[[0, 0, anchor] as [NSNumber]].floatValue
                let y = detectionsArray[[0, 1, anchor] as [NSNumber]].floatValue
                let w = detectionsArray[[0, 2, anchor] as [NSNumber]].floatValue
                let h = detectionsArray[[0, 3, anchor] as [NSNumber]].floatValue
                
                var maxConf: Float = 0
                var maxClass = -1
                for classIdx in 4..<4589 {
                    let conf = detectionsArray[[0, classIdx, anchor] as [NSNumber]].floatValue
                    if conf > maxConf {
                        maxConf = conf
                        maxClass = classIdx - 4
                    }
                }
                
                print("Anchor \(anchor): pos(\(Int(x)),\(Int(y))) size(\(Int(w))x\(Int(h))) maxConf:\(Int(maxConf*100))% class:\(maxClass)")
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
            
            var anchorDetections: [(String, Float)] = []
            
            for (classIdx, className) in furnitureClasses {
                let conf = detections[[0, 4 + classIdx, anchor] as [NSNumber]].floatValue
                if conf > 0.3 {
                    anchorDetections.append((className, conf))
                    
                    var coeffs = [Float](repeating: 0, count: 32)
                    for i in 0..<32 {
                        coeffs[i] = detections[[0, 4 + 4585 + i, anchor] as [NSNumber]].floatValue
                    }
                    all.append(DetectionSmarty(x: x, y: y, width: w, height: h, confidence: conf, classIdx: classIdx, className: className, maskCoeffs: coeffs))
                }
            }
            
            if !anchorDetections.isEmpty {
                print("Anchor \(anchor): pos(\(Int(x)),\(Int(y))) size(\(Int(w))x\(Int(h)))")
                for (name, conf) in anchorDetections.sorted(by: { $0.1 > $1.1 }) {
                    print("  - \(name): \(Int(conf * 100))%")
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
    
    // Simple diverse detection selection
    private func getDiverseDetections(from detections: [DetectionSmarty], maxCount: Int) -> [DetectionSmarty] {
        var selected: [DetectionSmarty] = []
        var seenClasses = Set<Int>()
        
        // Sort by confidence
        let sorted = detections.sorted { $0.confidence > $1.confidence }
        
        // First pass: Take best of each unique class
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
        
        // Save original image
//        saveDebugImage(pixelBuffer: originalImage, stage: "1_original")
        
        // Extract all detections
        let allDetections = extractDetections(from: detections)
        print("📊 [DETECTION] Extracted \(allDetections.count) raw detections")
        
        // Print all values of allDetections
        print("\n🔍 ========== ALL DETECTIONS VALUES ==========")
        for (index, detection) in allDetections.enumerated() {
            print("Det #\(index): \(detection.className) (\(detection.classIdx)) | Conf: \(String(format: "%.3f", detection.confidence)) (\(Int(detection.confidence * 100))%) | Pos: (\(String(format: "%.1f", detection.x)), \(String(format: "%.1f", detection.y))) | Size: \(String(format: "%.1f", detection.width))x\(String(format: "%.1f", detection.height)) | BBox: [\(String(format: "%.1f", detection.x - detection.width/2)), \(String(format: "%.1f", detection.y - detection.height/2)), \(String(format: "%.1f", detection.x + detection.width/2)), \(String(format: "%.1f", detection.y + detection.height/2))] | Mask: [\(detection.maskCoeffs.prefix(5).map { String(format: "%.3f", $0) }.joined(separator: ", "))...]")
        }
        print("============================================\n")
        
        // Apply HIERARCHICAL NMS
        let hierarchicalDetections = applyHierarchicalNMS(detections: allDetections, iouThreshold: 0.8)
        print("📊 [H-NMS] Kept \(hierarchicalDetections.count) detections after hierarchical NMS")
        for (index, detection) in hierarchicalDetections.enumerated() {
            print("H-NMS #\(index): \(detection.className) (\(detection.classIdx)) | Conf: \(String(format: "%.3f", detection.confidence)) (\(Int(detection.confidence * 100))%) | Pos: (\(String(format: "%.1f", detection.x)), \(String(format: "%.1f", detection.y))) | Size: \(String(format: "%.1f", detection.width))x\(String(format: "%.1f", detection.height)) | BBox: [\(String(format: "%.1f", detection.x - detection.width/2)), \(String(format: "%.1f", detection.y - detection.height/2)), \(String(format: "%.1f", detection.x + detection.width/2)), \(String(format: "%.1f", detection.y + detection.height/2))] | Mask: [\(detection.maskCoeffs.prefix(5).map { String(format: "%.3f", $0) }.joined(separator: ", "))...]")
        }
        
                
        
        // Get diverse detections (max 5 different classes)
//        let diverseDetections = getDiverseDetections(from: hierarchicalDetections, maxCount: 5)
//        print("📊 [DIVERSE] Using \(diverseDetections.count) detections")
        
        guard !hierarchicalDetections.isEmpty else {
            print("❌ [DETECTION] No valid detections found")
            DispatchQueue.main.async {
                self.isProcessing = false
                self.segmentedImage = nil
                self.furnitureOpacity = 0.0
                self.lastConfidence = 0.0
                self.currentBBox = .zero
            }
            return
        }
        
        // Save masks from hierarchicalDetections
        print("\n💾 ========== SAVING HIERARCHICAL MASKS ==========")
        for (index, detection) in hierarchicalDetections.enumerated() {
            print("Generating mask for \(detection.className) @ \(Int(detection.confidence * 100))%")
            
            // Generate individual mask for this detection
            var individualMask = [Float](repeating: 0, count: 160 * 160)
            
            for y in 0..<160 {
                for x in 0..<160 {
                    var sum: Float = 0
                    for c in 0..<32 {
                        sum += detection.maskCoeffs[c] * prototypes[[0, c, y, x] as [NSNumber]].floatValue
                    }
                    // Save raw sum values (before sigmoid) to see original form
                    individualMask[y * 160 + x] = sum
                }
            }
            
            let stageName = "hierarchical_\(index+1)_\(detection.className)_\(Int(detection.confidence * 100))pct"
            saveMaskAsImage(mask: individualMask, stage: stageName)
        }
        print("📊 [SAVED] Generated and saved \(hierarchicalDetections.count) individual masks")
        
        
        // Use the best detection for bbox
        let best = hierarchicalDetections.first!
        print("✅ [BEST] Primary: \(best.className) @ \(Int(best.confidence * 100))%")
        print("   Position: (\(Int(best.x)), \(Int(best.y))), Size: \(Int(best.width))x\(Int(best.height))")
        
        // Save image with bbox
//        saveDebugImageWithBBox(pixelBuffer: originalImage, bbox: best, stage: "2_bbox_marked")
        
        // Set UI bbox
        let bbox = CGRect(
            x: CGFloat(best.x - best.width / 2),
            y: CGFloat(best.y - best.height / 2),
            width: CGFloat(best.width),
            height: CGFloat(best.height)
        )
        
        DispatchQueue.main.async {
            self.currentBBox = bbox
            self.lastConfidence = best.confidence
        }
        
        // Generate combined mask from diverse detections - BBOX OPTIMIZED
        print("\n🎨 ========== GENERATING COMBINED MASK ==========")
        
        // Calculate union bbox of all detections
        let scale: Float = 160.0 / 640.0
        var minX = Float.infinity, minY = Float.infinity
        var maxX = -Float.infinity, maxY = -Float.infinity
        
        for detection in hierarchicalDetections {
            let x1 = (detection.x - detection.width/2) * scale
            let y1 = (detection.y - detection.height/2) * scale
            let x2 = (detection.x + detection.width/2) * scale
            let y2 = (detection.y + detection.height/2) * scale
            
            minX = min(minX, x1)
            minY = min(minY, y1)
            maxX = max(maxX, x2)
            maxY = max(maxY, y2)
        }
        
        // Clamp to mask bounds and convert to ints
        let bx1 = max(0, min(159, Int(minX)))
        let by1 = max(0, min(159, Int(minY)))
        let bx2 = max(0, min(159, Int(maxX)))
        let by2 = max(0, min(159, Int(maxY)))
        
        let bboxWidth = bx2 - bx1 + 1
        let bboxHeight = by2 - by1 + 1
        
        print("📐 [UNION BBOX] Processing area: (\(bx1),\(by1)) to (\(bx2),\(by2)) - \(bboxWidth)x\(bboxHeight) pixels")
        
        // Initialize combined mask - ONLY for bbox area (not full 160x160)
        var combinedMask = [Float](repeating: 0, count: 160 * 160)
        
        // Apply Mask IoU filtering
        let maskFilteredDetections = applyMaskIoU(detections: hierarchicalDetections, iouThreshold: 0.8, prototypes: prototypes)
        
        print("📊 [MASK-FILTERED] Final detections after Mask IoU filtering:")
        for (index, detection) in maskFilteredDetections.enumerated() {
            print("Mask-Filtered #\(index): \(detection.className) (\(detection.classIdx)) | Conf: \(String(format: "%.3f", detection.confidence)) (\(Int(detection.confidence * 100))%) | Pos: (\(String(format: "%.1f", detection.x)), \(String(format: "%.1f", detection.y))) | Size: \(String(format: "%.1f", detection.width))x\(String(format: "%.1f", detection.height))")
        }
        print("📊 [MASK-FILTERED] Total kept: \(maskFilteredDetections.count) detections")
        
        
        for (index, detection) in maskFilteredDetections.enumerated() {
            print("Processing #\(index+1): \(detection.className) @ \(Int(detection.confidence * 100))%")
            
            // Only process within bounding box area
            for y in by1...by2 {
                for x in bx1...bx2 {
                    var sum: Float = 0
                    for c in 0..<32 {
                        sum += detection.maskCoeffs[c] * prototypes[[0, c, y, x] as [NSNumber]].floatValue
                    }
                    let maskValue = sigmoid(sum)
                    
                    // Combine masks using MAX operation (keep highest confidence for each pixel)
                    let idx = y * 160 + x
                    combinedMask[idx] = max(combinedMask[idx], maskValue)
                }
            }
        }
        
//        print("📊 [BBOX OPTIMIZED] Processed \(bboxWidth * bboxHeight) pixels instead of \(160 * 160)")
        
//        let nonZeroCount = combinedMask.filter { $0 > 0.5 }.count
//        print("📊 [COMBINED] Mask has \(nonZeroCount) positive pixels before post-processing")
        
//        saveMaskAsImage(mask: combinedMask, stage: "4_combined_raw")
        
        // Apply simple post-processing
        applyPostProcessingAndMask(mask: combinedMask, best: best, to: originalImage, stage: "multi")
    }
    
    private func applyMaskIoU(
            detections: [DetectionSmarty],
            iouThreshold: Float,
            prototypes: MLMultiArray
        ) -> [DetectionSmarty] {
            guard !detections.isEmpty else { return [] }

            // Sort by confidence (high → low)
            let sorted = detections.sorted { $0.confidence > $1.confidence }

            print("\n🔍 Confidence-Aware Mask-NMS:")

            // ---- 1) Flatten prototypes into [C × (Hp*Wp)] as Float ----
            let shape = prototypes.shape.map { $0.intValue }      // [1, 32, 160, 160]
            let C = shape[1]
            let Hp = shape[2]
            let Wp = shape[3]
            let spatial = Hp * Wp                                 // 25600

            var protoMatrix = [Float](repeating: 0, count: C * spatial)

            for c in 0..<C {
                for y in 0..<Hp {
                    for x in 0..<Wp {
                        let val = prototypes[[0, c, y, x] as [NSNumber]].floatValue
                        let dstIndex = c * spatial + (y * Wp + x)
                        protoMatrix[dstIndex] = val
                    }
                }
            }

            // ---- 2) Build per-detection masks with vDSP_mmul ----
            var masks: [[Float]] = []
            masks.reserveCapacity(sorted.count)

            for det in sorted {
                var mask = [Float](repeating: 0, count: spatial)

                vDSP_mmul(
                    det.maskCoeffs, 1,           // A: 1×C
                    protoMatrix, 1,              // B: C×spatial
                    &mask, 1,                    // C: 1×spatial
                    1,
                    vDSP_Length(spatial),
                    vDSP_Length(C)
                )

                // Sigmoid → soft mask [0,1]
                for i in 0..<spatial {
                    let v = mask[i]
                    mask[i] = 1.0 / (1.0 + exp(-v))
                }

                masks.append(mask)
            }

            // ---- 3) CONFIDENCE-AWARE mask-based NMS ----
            var kept: [DetectionSmarty] = []
            var keptMasks: [[Float]] = []

            for (i, det) in sorted.enumerated() {
                
                let candidateMask = masks[i]
                let candidateConf = det.confidence
                
                var isDuplicate = false
                var suppressorInfo = ""

                for (existingIndex, existingMask) in keptMasks.enumerated() {
                    let existingDet = kept[existingIndex]
                    let existingConf = existingDet.confidence
                    let iou = calculateMaskIoU(mask1: candidateMask, mask2: existingMask)
                    
                    if iou >= iouThreshold {
                        // CONFIDENCE-AWARE LOGIC:
                        let confDiff = candidateConf - existingConf
                        
                        if confDiff >= 0.15 {
                            // Candidate is SIGNIFICANTLY higher confidence (15%+)
                            // REPLACE the existing detection
                            print("🔄 REPLACE: \(Int(candidateConf*100))% \(det.className) replaces \(Int(existingConf*100))% \(existingDet.className) (IoU: \(Int(iou*100))%)")
                            kept[existingIndex] = det
                            keptMasks[existingIndex] = candidateMask
                            isDuplicate = true // Don't add as new
                            break
                        } else if confDiff <= -0.05 {
                            // Existing is higher confidence (5%+) - suppress candidate
                            isDuplicate = true
                            suppressorInfo = "by \(Int(existingConf*100))% \(existingDet.className)"
                            break
                        } else {
                            // Similar confidence (within 5%) - keep both if different classes
                            if det.classIdx != existingDet.classIdx {
                                print("✅ KEEP BOTH: Similar conf \(Int(candidateConf*100))% \(det.className) vs \(Int(existingConf*100))% \(existingDet.className)")
                                // Don't suppress - keep both
                            } else {
                                // Same class, similar confidence - suppress lower one
                                isDuplicate = true
                                suppressorInfo = "same class, similar conf"
                                break
                            }
                        }
                    }
                }

                if !isDuplicate {
                    kept.append(det)
                    keptMasks.append(candidateMask)
                    print("✅ KEEP \(Int(candidateConf*100))% \(det.className)")

                    // Save the kept mask with confidence overlay
                    let stageName = "kept_\(kept.count)_\(det.className)_\(Int(candidateConf * 100))pct"
                    saveMaskAsImageWithConfidence(mask: candidateMask, stage: stageName, confidence: candidateConf)
                } else if !suppressorInfo.isEmpty {
                    print("❌ SUPPRESS \(Int(candidateConf*100))% \(det.className) \(suppressorInfo)")
                }
            }

            print("Confidence-Aware Mask-NMS: \(sorted.count) → \(kept.count) detections")
            return kept
        }
    
    private func calculateMaskIoU(mask1: [Float], mask2: [Float], eps: Float = 1e-7) -> Float {
            let n = min(mask1.count, mask2.count)
            guard n > 0 else { return 0 }

            // intersection = (mask1 * mask2).sum()
            var intersection: Float = 0
            vDSP_dotpr(mask1, 1, mask2, 1, &intersection, vDSP_Length(n))

            // area1, area2
            var sum1: Float = 0
            var sum2: Float = 0
            vDSP_sve(mask1, 1, &sum1, vDSP_Length(n))
            vDSP_sve(mask2, 1, &sum2, vDSP_Length(n))

            let union = sum1 + sum2 - intersection
            return intersection / (union + eps)
        }
    
    // Simple post-processing with morphology and bbox cropping
    private func applyPostProcessingAndMask(mask: [Float], best: DetectionSmarty, to originalImage: CVPixelBuffer, stage: String) {
        print("\n🔧 ========== POST-PROCESSING ==========")
//        var mask = mask
        
        // Calculate bbox in mask coordinates
        let scale: Float = 160.0 / 640.0
        let bx1 = max(0, min(159, Int((best.x - best.width/2) * scale)))
        let by1 = max(0, min(159, Int((best.y - best.height/2) * scale)))
        let bx2 = max(0, min(159, Int((best.x + best.width/2) * scale)))
        let by2 = max(0, min(159, Int((best.y + best.height/2) * scale)))
        
        print("📐 [BBOX] Mask space: (\(bx1),\(by1)) to (\(bx2),\(by2))")
        
        // Convert to binary
        var binary = [[UInt8]](repeating: [UInt8](repeating: 0, count: 160), count: 160)
        var binaryCount = 0
        for y in 0..<160 {
            for x in 0..<160 {
                binary[y][x] = mask[y * 160 + x] > 0.4 ? 1 : 0  // Slightly lower threshold
                if binary[y][x] == 1 { binaryCount += 1 }
            }
        }
        print("📊 [BINARY] Converted to binary: \(binaryCount) pixels")
        
     
        
        
//        saveMaskAsImage(mask: binaryToFloat(binary), stage: "6_\(stage)_eroded")
        
        // Convert back to float
//        var finalCount = 0
//        for y in 0..<160 {
//            for x in 0..<160 {
//                mask[y * 160 + x] = Float(binary[y][x])
//                if mask[y * 160 + x] > 0 { finalCount += 1 }
//            }
//        }
        
//        // CRITICAL: Crop mask to bbox to prevent background intrusion
//        var croppedCount = 0
//        for y in 0..<160 {
//            for x in 0..<160 {
//                if y < by1 || y > by2 || x < bx1 || x > bx2 {
//                    mask[y * 160 + x] = 0  // Zero out everything outside bbox
//                } else if mask[y * 160 + x] > 0 {
//                    croppedCount += 1
//                }
//            }
//        }
        
//        print("📊 [FINAL] Mask pixels after morphology: \(finalCount)")
//        print("📊 [FINAL] After bbox crop: \(croppedCount) pixels")
        
//        saveMaskAsImage(mask: mask, stage: "7_\(stage)_final_mask")
        
        // Enhance weak pixels: boost anything > 0 to full strength
        print("\n🔧 ========== WEAK PIXEL ENHANCEMENT ==========")
        
//        // Count pixels before enhancement
//        var zeroPixels = 0
//        var weakPixels = 0
//        var strongPixels = 0
//        var pixelDistribution = [String: Int]()
        
//        for i in 0..<mask.count {
//            let value = mask[i]
//            if value == 0.0 {
//                zeroPixels += 1
//            } else if value > 0.0 && value < 1.0 {
//                weakPixels += 1
//                let bucket = String(format: "%.1f", value)
//                pixelDistribution[bucket, default: 0] += 1
//            } else if value == 1.0 {
//                strongPixels += 1
//            }
//        }
        
//        print("📊 [BEFORE] Total pixels: \(mask.count)")
//        print("📊 [BEFORE] Zero pixels (0.0): \(zeroPixels)")
//        print("📊 [BEFORE] Weak pixels (0.0 < x < 1.0): \(weakPixels)")
//        print("📊 [BEFORE] Strong pixels (1.0): \(strongPixels)")
//        
//        if !pixelDistribution.isEmpty {
//            print("📊 [BEFORE] Weak pixel distribution:")
//            for (value, count) in pixelDistribution.sorted(by: { $0.key < $1.key }) {
//                print("           Value \(value): \(count) pixels")
//            }
//        }
        
//        print("🔧 [ENHANCE] Now boosting weak pixels to full strength...")
//        
//        // Enhance weak pixels
//        var enhancedCount = 0
//        var enhancementDetails = [String: Int]()
//        
//        for i in 0..<mask.count {
//            if mask[i] > 0.0 && mask[i] < 1.0 {
//                let oldValue = String(format: "%.3f", mask[i])
//                mask[i] = 1.0  // Boost weak pixels to full strength
//                enhancedCount += 1
//                enhancementDetails[oldValue, default: 0] += 1
//            }
//        }
        
        // Count pixels after enhancement
//        var newZeroPixels = 0
//        var newWeakPixels = 0
//        var newStrongPixels = 0
//        
//        for i in 0..<mask.count {
//            let value = mask[i]
//            if value == 0.0 {
//                newZeroPixels += 1
//            } else if value > 0.0 && value < 1.0 {
//                newWeakPixels += 1
//            } else if value == 1.0 {
//                newStrongPixels += 1
//            }
//        }
        
//        print("✅ [ENHANCED] Boosted \(enhancedCount) weak pixels to full strength")
//        print("📊 [AFTER] Zero pixels (0.0): \(newZeroPixels)")
//        print("📊 [AFTER] Weak pixels (0.0 < x < 1.0): \(newWeakPixels)")
//        print("📊 [AFTER] Strong pixels (1.0): \(newStrongPixels)")
//        
//        if !enhancementDetails.isEmpty && enhancementDetails.count <= 10 {
//            print("📊 [ENHANCED] Enhancement details:")
//            for (oldValue, count) in enhancementDetails.sorted(by: { $0.key < $1.key }) {
//                print("           \(oldValue) → 1.0: \(count) pixels")
//            }
//        } else if enhancementDetails.count > 10 {
//            print("📊 [ENHANCED] Enhanced \(enhancementDetails.count) different value ranges")
//        }
//        
//        let enhancement = newStrongPixels - strongPixels
//        print("📈 [RESULT] Strong pixel increase: +\(enhancement) pixels")
//        print("================================================\n")
//        
//        saveMaskAsImage(mask: mask, stage: "8_\(stage)_enhanced_mask")
        
        print("🎨 [APPLY] Applying final enhanced mask to image")
        applyMaskToImage(mask: mask, to: originalImage)
        
        print("✅ ==================== FRAME COMPLETE ====================\n")
    }
    
    // Apply mask to image
    private func applyMaskToImage(mask: [Float], to pixelBuffer: CVPixelBuffer) {
        autoreleasepool {
            let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
            let width = CVPixelBufferGetWidth(pixelBuffer)
            let height = CVPixelBufferGetHeight(pixelBuffer)
            
            guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent),
                  let ctx = CGContext(data: nil, width: width, height: height,
                                     bitsPerComponent: 8, bytesPerRow: width * 4,
                                     space: CGColorSpaceCreateDeviceRGB(),
                                     bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue),
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
                        pixels[idx + 3] = 0  // Transparent
                        continue
                    }
                    
                    let maskValue = mask[y0 * 160 + x0]
                    
                    if maskValue > 0.0 {
                        // Keep furniture pixels as they are (don't change RGB)
                        pixels[idx + 3] = 255  // Fully opaque
                    } else {
                        // Make background transparent
                        pixels[idx + 3] = 0    // Transparent
                    }
                }
            }
            
            // Fill holes within chair boundaries
            // Apply contour-based largest object hole filling
            fillHolesInChair(pixels: pixels, width: width, height: height)
            
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
    
    // CONTOUR-BASED LARGEST OBJECT FILLING - for chair/bed scenarios
    private func fillHolesInChair(pixels: UnsafeMutablePointer<UInt8>, width: Int, height: Int) {
        print("🪑 [CONTOUR] Starting largest object hole filling for \(width)x\(height)")
        
        // Create binary mask from alpha channel
        var binaryMask = [UInt8](repeating: 0, count: width * height)
        var initialPixelCount = 0
        for y in 0..<height {
            for x in 0..<width {
                let pixelIdx = (y * width + x) * 4
                if pixels[pixelIdx + 3] > 0 {
                    binaryMask[y * width + x] = 255
                    initialPixelCount += 1
                } else {
                    binaryMask[y * width + x] = 0
                }
            }
        }
        
        print("🪑 [CONTOUR] Initial furniture pixels: \(initialPixelCount)")
        
        // Find all connected components
        var visited = [Bool](repeating: false, count: width * height)
        var allComponents: [[Int]] = []
        
        for y in 0..<height {
            for x in 0..<width {
                let idx = y * width + x
                if binaryMask[idx] == 255 && !visited[idx] {
                    var component: [Int] = []
                    floodFill(mask: &binaryMask, visited: &visited, x: x, y: y, width: width, height: height, component: &component)
                    
                    if component.count > 50 {  // Only consider meaningful components
                        allComponents.append(component)
                    }
                }
            }
        }
        
        // Sort components by size (largest first)
        allComponents.sort { $0.count > $1.count }
        
        guard !allComponents.isEmpty else {
            print("🪑 [CONTOUR] No significant components found")
            return
        }
        
        // Take the LARGEST component (chair main body, bed main surface, etc.)
        let largestComponent = allComponents[0]
        let largestSize = largestComponent.count
        
        print("🪑 [CONTOUR] Found \(allComponents.count) components")
        print("🪑 [CONTOUR] Largest component: \(largestSize) pixels (\(Int(Float(largestSize)/Float(width*height)*100))% of image)")
        
        // Create mask with ONLY the largest component
        var cleanMask = [UInt8](repeating: 0, count: width * height)
        for idx in largestComponent {
            if idx >= 0 && idx < cleanMask.count {
                cleanMask[idx] = 255
            }
        }
        
        // Find bounding box of the largest component for efficient hole filling
        var minX = width, maxX = 0, minY = height, maxY = 0
        for idx in largestComponent {
            let x = idx % width
            let y = idx / width
            minX = min(minX, x)
            maxX = max(maxX, x)
            minY = min(minY, y)
            maxY = max(maxY, y)
        }
        
        print("🪑 [CONTOUR] Bounding box: (\(minX),\(minY)) to (\(maxX),\(maxY))")
        
        // CONTOUR HOLE FILLING - Fill interior holes in the main furniture object
        // This is perfect for chairs (fill seat holes) and beds (fill sheet gaps)
        var filledPixels = 0
        
        for y in minY...maxY {
            var inside = false
            var lastPixel: UInt8 = 0
            
            for x in minX...maxX {
                let idx = y * width + x
                let currentPixel = cleanMask[idx]
                
                // Cross from outside to inside furniture boundary
                if currentPixel == 255 && lastPixel == 0 {
                    inside = !inside
                } 
                // Cross from inside to outside furniture boundary  
                else if currentPixel == 0 && lastPixel == 255 {
                    inside = !inside
                }
                
                // Fill holes INSIDE the furniture contour
                if inside && currentPixel == 0 {
                    cleanMask[idx] = 255
                    filledPixels += 1
                }
                
                lastPixel = currentPixel
            }
        }
        
        print("🪑 [CONTOUR] Filled \(filledPixels) hole pixels inside largest object")
        print("🪑 [CONTOUR] Total pixels after filling: \(largestSize + filledPixels)")
        
        // Apply the filled contour mask back to the original image
        var appliedPixels = 0
        for y in 0..<height {
            for x in 0..<width {
                let maskIdx = y * width + x
                let pixelIdx = maskIdx * 4
                
                if cleanMask[maskIdx] == 255 {
                    if binaryMask[maskIdx] == 0 {
                        // This pixel was a hole that got filled - interpolate color from nearby furniture
                        var avgR: Int = 0, avgG: Int = 0, avgB: Int = 0, count = 0
                        
                        // Sample colors from nearby furniture pixels (3x3 neighborhood)
                        for dy in -1...1 {
                            for dx in -1...1 {
                                let ny = y + dy
                                let nx = x + dx
                                if ny >= 0 && ny < height && nx >= 0 && nx < width {
                                    let nearbyIdx = (ny * width + nx) * 4
                                    if binaryMask[ny * width + nx] == 255 {  // Original furniture pixel
                                        avgR += Int(pixels[nearbyIdx + 2])      // Red
                                        avgG += Int(pixels[nearbyIdx + 1])      // Green  
                                        avgB += Int(pixels[nearbyIdx])          // Blue
                                        count += 1
                                    }
                                }
                            }
                        }
                        
                        if count > 0 {
                            // Use interpolated color from nearby furniture
                            pixels[pixelIdx] = UInt8(avgB / count)      // Blue
                            pixels[pixelIdx + 1] = UInt8(avgG / count)  // Green
                            pixels[pixelIdx + 2] = UInt8(avgR / count)  // Red
                            pixels[pixelIdx + 3] = 255                  // Alpha
                            appliedPixels += 1
                        } else {
                            // Fallback to neutral tone if no nearby furniture
                            pixels[pixelIdx] = 100      // Blue
                            pixels[pixelIdx + 1] = 90   // Green
                            pixels[pixelIdx + 2] = 80   // Red
                            pixels[pixelIdx + 3] = 255  // Alpha
                            appliedPixels += 1
                        }
                    } else {
                        // Keep existing furniture pixel as is
                        pixels[pixelIdx + 3] = 255  // Ensure it stays opaque
                    }
                } else {
                    // Not part of largest object - make transparent
                    pixels[pixelIdx + 3] = 0
                }
            }
        }
        
        print("🪑 [CONTOUR] Applied filling to \(appliedPixels) pixels")
        print("🪑 [CONTOUR] Contour-based largest object processing complete!")
    }
    
    // Iterative flood fill to find connected components (prevents stack overflow)
    private func floodFill(mask: inout [UInt8], visited: inout [Bool], x: Int, y: Int, width: Int, height: Int, component: inout [Int]) {
        var stack: [(Int, Int)] = [(x, y)]
        
        while !stack.isEmpty {
            let (currentX, currentY) = stack.removeLast()
            let idx = currentY * width + currentX
            
            if currentX < 0 || currentX >= width || currentY < 0 || currentY >= height || 
               visited[idx] || mask[idx] == 0 {
                continue
            }
            
            visited[idx] = true
            component.append(idx)
            
            // Add 4-connected neighbors to stack
            stack.append((currentX + 1, currentY))
            stack.append((currentX - 1, currentY))
            stack.append((currentX, currentY + 1))
            stack.append((currentX, currentY - 1))
        }
    }
    
    // Helper to convert binary to float
    private func binaryToFloat(_ binary: [[UInt8]]) -> [Float] {
        var result = [Float](repeating: 0, count: 160 * 160)
        for y in 0..<160 {
            for x in 0..<160 {
                result[y * 160 + x] = Float(binary[y][x])
            }
        }
        return result
    }
    
    // Add IoU calculation helper
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

    // Add hierarchical NMS
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
                
                // Key hierarchical rule: Keep different classes even if high overlap
                if iou > iouThreshold {
                    // Only suppress if SAME class and high overlap
                    if det.classIdx == existing.classIdx {
                        shouldSuppress = true
                        print("❌ Suppressed duplicate: \(det.className) @ \(Int(det.confidence * 100))%")
                        break
                    }
                    // Different class = keep it (chair vs office chair)
                    print("✅ Keeping overlapping: \(det.className) overlaps \(existing.className)")
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

    private func saveMaskAsImageWithConfidence(mask: [Float], stage: String, confidence: Float) {
        UIGraphicsBeginImageContextWithOptions(CGSize(width: 160, height: 160), false, 2.0)
        guard let ctx = UIGraphicsGetCurrentContext() else { return }
        
        // Find min/max for normalization
        let minVal = mask.min() ?? 0
        let maxVal = mask.max() ?? 1
        let range = maxVal - minVal
        
        for y in 0..<160 {
            for x in 0..<160 {
                let rawValue = mask[y * 160 + x]
                // Normalize to 0-1 range for visualization
                let normalizedValue = range > 0 ? (rawValue - minVal) / range : 0
                let gray = CGFloat(max(0, min(1, normalizedValue)))
                ctx.setFillColor(UIColor(white: gray, alpha: 1.0).cgColor)
                ctx.fill(CGRect(x: x, y: y, width: 1, height: 1))
            }
        }
        
        // Add confidence text overlay
        let confidenceText = "\(Int(confidence * 100))%"
        let attributes: [NSAttributedString.Key: Any] = [
            .font: UIFont.boldSystemFont(ofSize: 14),
            .foregroundColor: UIColor.red
        ]
        
        let textSize = confidenceText.size(withAttributes: attributes)
        let textRect = CGRect(x: 5, y: 5, width: textSize.width, height: textSize.height)
        
        // Add white background for text
        ctx.setFillColor(UIColor.white.cgColor)
        ctx.fill(textRect.insetBy(dx: -2, dy: -1))
        
        // Draw text
        confidenceText.draw(in: textRect, withAttributes: attributes)
        
        guard let maskImage = UIGraphicsGetImageFromCurrentImageContext() else { return }
        UIGraphicsEndImageContext()
        
        UIImageWriteToSavedPhotosAlbum(maskImage, nil, nil, nil)
        print("💾 [SAVE] Saved mask with confidence: \(stage)")
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
