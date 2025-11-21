import SwiftUI
import AVFoundation
import CoreML
import CoreImage
import Photos

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
    @State private var bboxPulse: CGFloat = 1.0
    
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
                    
                    // Strong green border - multiple layers for visibility with pulse effect
                    let pulseScale = bboxPulse
                    context.stroke(rect, with: .color(.green.opacity(0.4 * pulseScale)), lineWidth: 12 * pulseScale)
                    context.stroke(rect, with: .color(.green.opacity(0.7)), lineWidth: 8)
                    context.stroke(rect, with: .color(.green), lineWidth: 4)
                    context.stroke(rect, with: .color(.white), lineWidth: 2)
                    
                    // Add corner markers for extra visibility
                    let cornerSize: CGFloat = 20 * pulseScale
                    let corners = [
                        (camera.currentBBox.minX, camera.currentBBox.minY), // Top-left
                        (camera.currentBBox.maxX, camera.currentBBox.minY), // Top-right
                        (camera.currentBBox.minX, camera.currentBBox.maxY), // Bottom-left
                        (camera.currentBBox.maxX, camera.currentBBox.maxY)  // Bottom-right
                    ]
                    
                    context.stroke(
                        Path { path in
                            for (x, y) in corners {
                                // Horizontal corner lines
                                path.move(to: CGPoint(x: x - cornerSize/2, y: y))
                                path.addLine(to: CGPoint(x: x + cornerSize/2, y: y))
                                
                                // Vertical corner lines
                                path.move(to: CGPoint(x: x, y: y - cornerSize/2))
                                path.addLine(to: CGPoint(x: x, y: y + cornerSize/2))
                            }
                        },
                        with: .color(.green.opacity(pulseScale)),
                        lineWidth: 6 * pulseScale
                    )
                }
                .ignoresSafeArea()
                .allowsHitTesting(false)
                .onAppear {
                    // Start pulsing animation
                    withAnimation(.easeInOut(duration: 0.8).repeatForever(autoreverses: true)) {
                        bboxPulse = 1.3
                    }
                }
                .onDisappear {
                    bboxPulse = 1.0
                }
            }
            
            // Mask-based bounding box (tighter fit to actual object)
            if camera.maskBBox != .zero && camera.segmentedImage != nil {
                Canvas { context, size in
                    let maskRect = Path(camera.maskBBox)
                    
                    // Blue border for mask-based bbox (different from detection bbox)
                    let pulseScale = bboxPulse * 0.8  // Slightly smaller pulse
                    context.stroke(maskRect, with: .color(.blue.opacity(0.3 * pulseScale)), lineWidth: 10 * pulseScale)
                    context.stroke(maskRect, with: .color(.blue.opacity(0.6)), lineWidth: 6)
                    context.stroke(maskRect, with: .color(.blue), lineWidth: 3)
                    context.stroke(maskRect, with: .color(.white), lineWidth: 1)
                    
                    // Add smaller corner markers for mask bbox
                    let cornerSize: CGFloat = 15 * pulseScale
                    let corners = [
                        (camera.maskBBox.minX, camera.maskBBox.minY), // Top-left
                        (camera.maskBBox.maxX, camera.maskBBox.minY), // Top-right
                        (camera.maskBBox.minX, camera.maskBBox.maxY), // Bottom-left
                        (camera.maskBBox.maxX, camera.maskBBox.maxY)  // Bottom-right
                    ]
                    
                    context.stroke(
                        Path { path in
                            for (x, y) in corners {
                                // Draw diamond-shaped corners instead of crosses
                                path.move(to: CGPoint(x: x - cornerSize/3, y: y))
                                path.addLine(to: CGPoint(x: x, y: y - cornerSize/3))
                                path.addLine(to: CGPoint(x: x + cornerSize/3, y: y))
                                path.addLine(to: CGPoint(x: x, y: y + cornerSize/3))
                                path.closeSubpath()
                            }
                        },
                        with: .color(.blue.opacity(pulseScale)),
                        lineWidth: 4 * pulseScale
                    )
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
                        if camera.maskBBox != .zero {
                            Text("MASK: \(Int(camera.maskBBox.width))×\(Int(camera.maskBBox.height))")
                                .font(.caption2)
                                .foregroundColor(.blue)
                        }
                        Text("PRODUCTION")
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
    @Published var maskBBox: CGRect = .zero  // New: mask-based bounding box
    
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
    
    // Calculate bounding box from actual mask pixels
    private func calculateMaskBoundingBox(from mask: [Float], originalImageWidth: Int, originalImageHeight: Int) -> CGRect {
        var minX = 160, maxX = 0, minY = 160, maxY = 0
        var foundPixels = false
        
        // Find bounds of mask pixels
        for y in 0..<160 {
            for x in 0..<160 {
                if mask[y * 160 + x] > 0.2 {  // Threshold for mask presence
                    minX = min(minX, x)
                    maxX = max(maxX, x)
                    minY = min(minY, y)
                    maxY = max(maxY, y)
                    foundPixels = true
                }
            }
        }
        
        guard foundPixels else { 
            print("🔍 [MASK_BBOX] No mask pixels found above threshold")
            return .zero 
        }
        
        print("🔍 [MASK_BBOX] Found mask bounds in 160x160 space: (\(minX),\(minY)) to (\(maxX),\(maxY))")
        
        // Convert from 160x160 mask space to 640x640 YOLO space
        let yoloMinX = Float(minX) * (640.0 / 160.0)
        let yoloMinY = Float(minY) * (640.0 / 160.0)
        let yoloMaxX = Float(maxX) * (640.0 / 160.0)
        let yoloMaxY = Float(maxY) * (640.0 / 160.0)
        
        print("🔍 [MASK_BBOX] Converted to YOLO 640x640 space: (\(Int(yoloMinX)),\(Int(yoloMinY))) to (\(Int(yoloMaxX)),\(Int(yoloMaxY)))")
        
        // Convert from YOLO 640x640 space to actual image space
        let imageScaleX = Float(originalImageWidth) / 640.0
        let imageScaleY = Float(originalImageHeight) / 640.0
        
        let imageMinX = yoloMinX * imageScaleX
        let imageMinY = yoloMinY * imageScaleY
        let imageMaxX = yoloMaxX * imageScaleX
        let imageMaxY = yoloMaxY * imageScaleY
        
        print("🔍 [MASK_BBOX] Converted to image space: (\(Int(imageMinX)),\(Int(imageMinY))) to (\(Int(imageMaxX)),\(Int(imageMaxY)))")
        
        // Convert to screen coordinates (same logic as YOLO detection bbox)
        let screenWidth = UIScreen.main.bounds.width
        let screenHeight = UIScreen.main.bounds.height
        
        // Account for camera rotation (90 degrees)
        let screenX = CGFloat(imageMinY) * (screenWidth / CGFloat(originalImageHeight))
        let screenY = CGFloat(imageMinX) * (screenHeight / CGFloat(originalImageWidth))
        let screenW = CGFloat(imageMaxY - imageMinY) * (screenWidth / CGFloat(originalImageHeight))
        let screenH = CGFloat(imageMaxX - imageMinX) * (screenHeight / CGFloat(originalImageWidth))
        
        let finalRect = CGRect(x: screenX, y: screenY, width: screenW, height: screenH)
        print("🔍 [MASK_BBOX] Final screen rect: \(finalRect)")
        
        return finalRect
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
            self.maskBBox = .zero  // Reset mask bbox too
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
                print("Detections shape: \(detectionsArray.shape)")  // [1, 4621, 8400]
                print("Prototypes shape: \(prototypesArray.shape)")   // [1, 32, 160, 160]
                
                // Check a few anchors for raw values
                print("\nFirst 3 anchors raw data:")
                for anchor in 0..<min(3, detectionsArray.shape[2].intValue) {
                    let x = detectionsArray[[0, 0, anchor] as [NSNumber]].floatValue
                    let y = detectionsArray[[0, 1, anchor] as [NSNumber]].floatValue
                    let w = detectionsArray[[0, 2, anchor] as [NSNumber]].floatValue
                    let h = detectionsArray[[0, 3, anchor] as [NSNumber]].floatValue
                    
                    // Check max confidence across all classes for this anchor
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
            self.processYOLOResults(detectionsArray, prototypes: prototypesArray, originalImage: pixelBuffer)
        }
    }
    
//    private func processYOLOResultss(_ detections: MLMultiArray, prototypes: MLMultiArray, originalImage: CVPixelBuffer) {
//        let nmsDetections = applyNMS(detections: extractDetections(from: detections), iouThreshold: 0.45)
//        guard let best = nmsDetections.first else {
//            DispatchQueue.main.async {
//                self.isProcessing = false
//                self.segmentedImage = nil
//                self.furnitureOpacity = 0.0
//                self.lastConfidence = 0.0
//                self.currentBBox = .zero
//            }
//            return
//        }
//
//        let bbox = CGRect(
//            x: CGFloat(best.x - best.width / 2),
//            y: CGFloat(best.y - best.height / 2),
//            width: CGFloat(best.width),
//            height: CGFloat(best.height)
//        )
//
//        DispatchQueue.main.async {
//            self.currentBBox = bbox
//            self.lastConfidence = best.confidence
//        }
//
//        // Generate and process mask
//        var mask = [Float](repeating: 0, count: 160 * 160)
//        for y in 0..<160 {
//            for x in 0..<160 {
//                var sum: Float = 0
//                for c in 0..<32 {
//                    sum += best.maskCoeffs[c] * prototypes[[0, c, y, x] as [NSNumber]].floatValue
//                }
//                mask[y * 160 + x] = sigmoid(sum)
//            }
//        }
//
//        // Post-process: morphology + scanline
//        let scale: Float = 160.0 / 640.0
//        let bx1 = max(1, min(158, Int((best.x - best.width/2) * scale)))
//        let by1 = max(1, min(158, Int((best.y - best.height/2) * scale)))
//        let bx2 = max(1, min(158, Int((best.x + best.width/2) * scale)))
//        let by2 = max(1, min(158, Int((best.y + best.height/2) * scale)))
//
//        var binary = [[UInt8]](repeating: [UInt8](repeating: 0, count: 160), count: 160)
//        for y in 0..<160 {
//            for x in 0..<160 {
//                binary[y][x] = mask[y * 160 + x] > 0.5 ? 1 : 0
//            }
//        }
//
//        // Morphological closing
//        for _ in 0..<5 {
//            var dilated = binary
//            for y in (by1+1)..<by2 {
//                for x in (bx1+1)..<bx2 {
//                    if binary[y][x] == 0 {
//                        if binary[y-1][x] == 1 || binary[y+1][x] == 1 ||
//                           binary[y][x-1] == 1 || binary[y][x+1] == 1 ||
//                           binary[y-1][x-1] == 1 || binary[y-1][x+1] == 1 ||
//                           binary[y+1][x-1] == 1 || binary[y+1][x+1] == 1 {
//                            dilated[y][x] = 1
//                        }
//                    }
//                }
//            }
//            binary = dilated
//        }
//
//        for _ in 0..<5 {
//            var eroded = binary
//            for y in (by1+1)..<by2 {
//                for x in (bx1+1)..<bx2 {
//                    if binary[y][x] == 1 {
//                        if binary[y-1][x] == 0 || binary[y+1][x] == 0 ||
//                           binary[y][x-1] == 0 || binary[y][x+1] == 0 {
//                            eroded[y][x] = 0
//                        }
//                    }
//                }
//            }
//            binary = eroded
//        }
//
//        // Scanline fill
//        for y in (by1+1)..<by2 {
//            var firstX = -1, lastX = -1
//            for x in (bx1+1)..<bx2 {
//                if binary[y][x] == 1 {
//                    if firstX == -1 { firstX = x }
//                    lastX = x
//                }
//            }
//            if firstX != -1 && lastX != -1 && lastX > firstX + 1 {
//                for x in (firstX+1)..<lastX {
//                    binary[y][x] = 1
//                }
//            }
//        }
//
//        for x in (bx1+1)..<bx2 {
//            var firstY = -1, lastY = -1
//            for y in (by1+1)..<by2 {
//                if binary[y][x] == 1 {
//                    if firstY == -1 { firstY = y }
//                    lastY = y
//                }
//            }
//            if firstY != -1 && lastY != -1 && lastY > firstY + 1 {
//                for y in (firstY+1)..<lastY {
//                    binary[y][x] = 1
//                }
//            }
//        }
//
//        // Convert to float
//        for y in 0..<160 {
//            for x in 0..<160 {
//                mask[y * 160 + x] = Float(binary[y][x])
//            }
//        }
//
//        // Crop to bbox
//        for y in 0..<160 {
//            for x in 0..<160 {
//                if y < by1 || y > by2 || x < bx1 || x > bx2 {
//                    mask[y * 160 + x] = 0
//                }
//            }
//        }
//
//        applyMaskToImage(mask: mask, to: originalImage)
//    }
    
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
            
            // Print if this anchor has detections
            if !anchorDetections.isEmpty {
                print("Anchor \(anchor): pos(\(Int(x)),\(Int(y))) size(\(Int(w))x\(Int(h)))")
                for (name, conf) in anchorDetections.sorted(by: { $0.1 > $1.1 }) {
                    print("  - \(name): \(Int(conf * 100))%")
                }
            }
        }
        
        // Summary
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
    
    private func applyNMS(detections: [DetectionSmarty], iouThreshold: Float) -> [DetectionSmarty] {
        guard !detections.isEmpty else { return [] }
        var kept: [DetectionSmarty] = []
        var suppressed = Set<Int>()
        let sorted = detections.sorted { $0.confidence > $1.confidence }
        
        print("\n🔍 NMS Processing:")
        print("Top 5 by confidence:")
        for (i, det) in sorted.prefix(5).enumerated() {
            print("  \(i+1). \(det.className): \(Int(det.confidence * 100))% at (\(Int(det.x)),\(Int(det.y)))")
        }
        
        for (i, det) in sorted.enumerated() {
            if suppressed.contains(i) { continue }
            kept.append(det)
            print("✅ KEPT #\(i): \(det.className) @ \(Int(det.confidence * 100))%")
            
            var suppressedThisRound: [String] = []
            for (j, other) in sorted.enumerated() where j > i {
                if suppressed.contains(j) { continue }
                let x1 = max(det.x - det.width/2, other.x - other.width/2)
                let y1 = max(det.y - det.height/2, other.y - other.height/2)
                let x2 = min(det.x + det.width/2, other.x + other.width/2)
                let y2 = min(det.y + det.height/2, other.y + other.height/2)
                let intersection = max(0, x2 - x1) * max(0, y2 - y1)
                let union = det.width * det.height + other.width * other.height - intersection
                if union > 0 && intersection / union > iouThreshold {
                    suppressed.insert(j)
                    suppressedThisRound.append("\(other.className)(\(Int(other.confidence * 100))%)")
                }
            }
            if !suppressedThisRound.isEmpty {
                print("   ❌ Suppressed: \(suppressedThisRound.joined(separator: ", "))")
            }
        }
        
        print("Final kept: \(kept.count) detections")
        for det in kept {
            print("  - \(det.className): \(Int(det.confidence * 100))%")
        }
        
        return kept
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
        
        // DRAW AND SAVE EDGE
        print("🎨 [EDGE] Drawing furniture edges...")
        
        // Create edge image
        UIGraphicsBeginImageContextWithOptions(CGSize(width: width, height: height), false, 1.0)
        guard let edgeCtx = UIGraphicsGetCurrentContext() else { return }
        
        // Black background
        edgeCtx.setFillColor(UIColor.black.cgColor)
        edgeCtx.fill(CGRect(x: 0, y: 0, width: width, height: height))
        
        // Draw solid yellow edges
        edgeCtx.setFillColor(UIColor.yellow.cgColor)
        
        // Find and draw edge pixels
        var edgePixelCount = 0
        for y in 1..<(height-1) {
            for x in 1..<(width-1) {
                let idx = y * width + x
                if cleanMask[idx] == 255 {
                    // Check if this pixel is on the edge (has at least one transparent neighbor)
                    let neighbors = [
                        cleanMask[(y-1) * width + x],     // top
                        cleanMask[(y+1) * width + x],     // bottom
                        cleanMask[y * width + (x-1)],     // left
                        cleanMask[y * width + (x+1)]      // right
                    ]
                    
                    if neighbors.contains(0) {
                        // This is an edge pixel - draw as solid yellow
                        edgeCtx.fill(CGRect(x: x, y: y, width: 1, height: 1))
                        edgePixelCount += 1
                    }
                }
            }
        }
        
        guard let edgeImage = UIGraphicsGetImageFromCurrentImageContext() else { return }
        UIGraphicsEndImageContext()
        
        // Save edge image
        UIImageWriteToSavedPhotosAlbum(edgeImage, nil, nil, nil)
        print("🎨 [EDGE] Detected \(edgePixelCount) edge pixels and saved image")
        
        // SOBEL EDGE DETECTION - Alternative method
        print("🔍 [SOBEL] Starting Sobel edge detection...")
        
        // Create grayscale version of cleanMask for Sobel
        var grayMask = [Float](repeating: 0, count: width * height)
        for i in 0..<(width * height) {
            grayMask[i] = Float(cleanMask[i]) / 255.0  // Convert to 0.0-1.0 range
        }
        
        // Sobel kernels
        let sobelX: [[Float]] = [
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ]
        
        let sobelY: [[Float]] = [
            [-1, -2, -1],
            [ 0,  0,  0],
            [ 1,  2,  1]
        ]
        
        // Apply Sobel edge detection
        var sobelEdges = [Float](repeating: 0, count: width * height)
        var sobelEdgeCount = 0
        
        for y in 1..<(height-1) {
            for x in 1..<(width-1) {
                var gx: Float = 0
                var gy: Float = 0
                
                // Apply Sobel kernels
                for ky in 0..<3 {
                    for kx in 0..<3 {
                        let pixelY = y - 1 + ky
                        let pixelX = x - 1 + kx
                        let pixelValue = grayMask[pixelY * width + pixelX]
                        
                        gx += pixelValue * sobelX[ky][kx]
                        gy += pixelValue * sobelY[ky][kx]
                    }
                }
                
                // Calculate gradient magnitude
                let magnitude = sqrt(gx * gx + gy * gy)
                sobelEdges[y * width + x] = magnitude
                
                // Count significant edges (threshold = 0.3)
                if magnitude > 0.3 {
                    sobelEdgeCount += 1
                }
            }
        }
        
        print("🔍 [SOBEL] Found \(sobelEdgeCount) Sobel edge pixels")
        
        // Create Sobel edge image
        UIGraphicsBeginImageContextWithOptions(CGSize(width: width, height: height), false, 1.0)
        guard let sobelCtx = UIGraphicsGetCurrentContext() else { 
            print("❌ [SOBEL] Failed to create graphics context")
            return 
        }
        
        // Black background
        sobelCtx.setFillColor(UIColor.black.cgColor)
        sobelCtx.fill(CGRect(x: 0, y: 0, width: width, height: height))
        
        // Draw Sobel edges in cyan
        for y in 0..<height {
            for x in 0..<width {
                let magnitude = sobelEdges[y * width + x]
                if magnitude > 0.3 {  // Threshold for significant edges
                    // Use magnitude as intensity (brighter = stronger edge)
                    let intensity = min(magnitude, 1.0)
                    sobelCtx.setFillColor(UIColor(red: 0, green: CGFloat(intensity), blue: CGFloat(intensity), alpha: 1).cgColor)
                    sobelCtx.fill(CGRect(x: x, y: y, width: 1, height: 1))
                }
            }
        }
        
        guard let sobelImage = UIGraphicsGetImageFromCurrentImageContext() else { 
            print("❌ [SOBEL] Failed to create Sobel edge image")
            return 
        }
        UIGraphicsEndImageContext()
        
        // Save Sobel edge image
        UIImageWriteToSavedPhotosAlbum(sobelImage, nil, nil, nil)
        print("🔍 [SOBEL] Saved Sobel edge detection image (cyan edges)")
        
        // MORPHOLOGICAL EDGE DETECTION
        print("🔶 [MORPH] Starting morphological edge detection...")
        
        // Create dilated version of cleanMask
        var dilated = [UInt8](repeating: 0, count: width * height)
        for y in 1..<(height-1) {
            for x in 1..<(width-1) {
                let idx = y * width + x
                if cleanMask[idx] == 255 {
                    // Dilate: expand by 1 pixel in all directions
                    for dy in -1...1 {
                        for dx in -1...1 {
                            let newIdx = (y + dy) * width + (x + dx)
                            if newIdx >= 0 && newIdx < (width * height) {
                                dilated[newIdx] = 255
                            }
                        }
                    }
                }
            }
        }
        
        // Morphological edge = dilated - original
        var morphEdges = [UInt8](repeating: 0, count: width * height)
        var morphEdgeCount = 0
        for i in 0..<(width * height) {
            if dilated[i] == 255 && cleanMask[i] == 0 {
                morphEdges[i] = 255
                morphEdgeCount += 1
            }
        }
        
        print("🔶 [MORPH] Found \(morphEdgeCount) morphological edge pixels")
        
        // Create morphological edge image (magenta edges)
        UIGraphicsBeginImageContextWithOptions(CGSize(width: width, height: height), false, 1.0)
        guard let morphCtx = UIGraphicsGetCurrentContext() else { 
            print("❌ [MORPH] Failed to create graphics context")
            return 
        }
        
        // Black background
        morphCtx.setFillColor(UIColor.black.cgColor)
        morphCtx.fill(CGRect(x: 0, y: 0, width: width, height: height))
        
        // Draw morphological edges in magenta
        morphCtx.setFillColor(UIColor.magenta.cgColor)
        for y in 0..<height {
            for x in 0..<width {
                if morphEdges[y * width + x] == 255 {
                    morphCtx.fill(CGRect(x: x, y: y, width: 1, height: 1))
                }
            }
        }
        
        guard let morphImage = UIGraphicsGetImageFromCurrentImageContext() else { 
            print("❌ [MORPH] Failed to create morphological edge image")
            return 
        }
        UIGraphicsEndImageContext()
        
        // Save morphological edge image
        UIImageWriteToSavedPhotosAlbum(morphImage, nil, nil, nil)
        print("🔶 [MORPH] Saved morphological edge detection image (magenta edges)")
        
        // CANNY EDGE DETECTION (simplified version)
        print("🌊 [CANNY] Starting Canny edge detection...")
        
        // Step 1: Apply Gaussian blur to reduce noise
        var blurred = [Float](repeating: 0, count: width * height)
        let gaussianKernel: [[Float]] = [
            [1, 2, 1],
            [2, 4, 2],
            [1, 2, 1]
        ]
        let kernelSum: Float = 16
        
        for y in 1..<(height-1) {
            for x in 1..<(width-1) {
                var sum: Float = 0
                for ky in 0..<3 {
                    for kx in 0..<3 {
                        let pixelY = y - 1 + ky
                        let pixelX = x - 1 + kx
                        let pixelValue = Float(cleanMask[pixelY * width + pixelX]) / 255.0
                        sum += pixelValue * gaussianKernel[ky][kx]
                    }
                }
                blurred[y * width + x] = sum / kernelSum
            }
        }
        
        // Step 2: Calculate gradients (reuse Sobel from above)
        var cannyEdges = [Float](repeating: 0, count: width * height)
        var cannyEdgeCount = 0
        
        for y in 1..<(height-1) {
            for x in 1..<(width-1) {
                var gx: Float = 0
                var gy: Float = 0
                
                // Apply Sobel kernels to blurred image
                for ky in 0..<3 {
                    for kx in 0..<3 {
                        let pixelY = y - 1 + ky
                        let pixelX = x - 1 + kx
                        let pixelValue = blurred[pixelY * width + pixelX]
                        
                        gx += pixelValue * sobelX[ky][kx]
                        gy += pixelValue * sobelY[ky][kx]
                    }
                }
                
                // Calculate gradient magnitude and direction
                let magnitude = sqrt(gx * gx + gy * gy)
                cannyEdges[y * width + x] = magnitude
                
                // Apply double threshold (simplified)
                if magnitude > 0.5 {  // High threshold
                    cannyEdgeCount += 1
                }
            }
        }
        
        // Step 3: Non-maximum suppression (simplified)
        var suppressedEdges = [Float](repeating: 0, count: width * height)
        for y in 1..<(height-1) {
            for x in 1..<(width-1) {
                let idx = y * width + x
                let magnitude = cannyEdges[idx]
                
                if magnitude > 0.5 {
                    // Check if this is a local maximum
                    let neighbors = [
                        cannyEdges[(y-1) * width + x],     // top
                        cannyEdges[(y+1) * width + x],     // bottom
                        cannyEdges[y * width + (x-1)],     // left
                        cannyEdges[y * width + (x+1)]      // right
                    ]
                    
                    if magnitude >= neighbors.max()! {
                        suppressedEdges[idx] = magnitude
                    }
                }
            }
        }
        
        print("🌊 [CANNY] Found \(cannyEdgeCount) Canny edge candidates")
        
        // Create Canny edge image (green edges)
        UIGraphicsBeginImageContextWithOptions(CGSize(width: width, height: height), false, 1.0)
        guard let cannyCtx = UIGraphicsGetCurrentContext() else { 
            print("❌ [CANNY] Failed to create graphics context")
            return 
        }
        
        // Black background
        cannyCtx.setFillColor(UIColor.black.cgColor)
        cannyCtx.fill(CGRect(x: 0, y: 0, width: width, height: height))
        
        // Draw Canny edges in green with intensity
        for y in 0..<height {
            for x in 0..<width {
                let magnitude = suppressedEdges[y * width + x]
                if magnitude > 0.3 {  // Lower threshold for display
                    let intensity = CGFloat(min(magnitude, 1.0))
                    cannyCtx.setFillColor(UIColor(red: 0, green: intensity, blue: 0, alpha: 1).cgColor)
                    cannyCtx.fill(CGRect(x: x, y: y, width: 1, height: 1))
                }
            }
        }
        
        guard let cannyImage = UIGraphicsGetImageFromCurrentImageContext() else { 
            print("❌ [CANNY] Failed to create Canny edge image")
            return 
        }
        UIGraphicsEndImageContext()
        
        // Save Canny edge image
        UIImageWriteToSavedPhotosAlbum(cannyImage, nil, nil, nil)
        print("🌊 [CANNY] Saved Canny edge detection image (green edges)")
        
        // COMBINED VISUALIZATION - All 4 methods in one image
        print("🎨 [COMBINED] Creating combined edge visualization...")
        
        UIGraphicsBeginImageContextWithOptions(CGSize(width: width, height: height), false, 1.0)
        guard let combinedCtx = UIGraphicsGetCurrentContext() else { 
            print("❌ [COMBINED] Failed to create graphics context")
            return 
        }
        
        // Black background
        combinedCtx.setFillColor(UIColor.black.cgColor)
        combinedCtx.fill(CGRect(x: 0, y: 0, width: width, height: height))
        
        // Draw all edge methods with different colors
        for y in 0..<height {
            for x in 0..<width {
                let idx = y * width + x
                var finalColor: UIColor = UIColor.black
                
                // Priority order: Canny > Morphological > Sobel > Original
                // This way, the "best" edges show on top
                
                // 1. Original edges (Yellow) - lowest priority
                if y > 0 && y < height-1 && x > 0 && x < width-1 && cleanMask[idx] == 255 {
                    let neighbors = [
                        cleanMask[(y-1) * width + x],     // top
                        cleanMask[(y+1) * width + x],     // bottom  
                        cleanMask[y * width + (x-1)],     // left
                        cleanMask[y * width + (x+1)]      // right
                    ]
                    if neighbors.contains(0) {
                        finalColor = UIColor.yellow.withAlphaComponent(0.7)
                    }
                }
                
                // 2. Sobel edges (Cyan) - medium-low priority
                let sobelMagnitude = sobelEdges[idx]
                if sobelMagnitude > 0.3 {
                    let intensity = min(sobelMagnitude, 1.0)
                    finalColor = UIColor(red: 0, green: CGFloat(intensity * 0.8), blue: CGFloat(intensity), alpha: 0.8)
                }
                
                // 3. Morphological edges (Magenta) - medium-high priority
                if morphEdges[idx] == 255 {
                    finalColor = UIColor.magenta.withAlphaComponent(0.9)
                }
                
                // 4. Canny edges (Green) - highest priority
                let cannyMagnitude = suppressedEdges[idx]
                if cannyMagnitude > 0.3 {
                    let intensity = min(cannyMagnitude, 1.0)
                    finalColor = UIColor(red: 0, green: CGFloat(intensity), blue: 0, alpha: 1.0)
                }
                
                // Draw the final color if it's not black
                if finalColor != UIColor.black {
                    combinedCtx.setFillColor(finalColor.cgColor)
                    combinedCtx.fill(CGRect(x: x, y: y, width: 1, height: 1))
                }
            }
        }
        
        guard let combinedImage = UIGraphicsGetImageFromCurrentImageContext() else { 
            print("❌ [COMBINED] Failed to create combined edge image")
            return 
        }
        UIGraphicsEndImageContext()
        
        // Save combined edge image
        UIImageWriteToSavedPhotosAlbum(combinedImage, nil, nil, nil)
        print("🎨 [COMBINED] Saved combined edge visualization!")
        print("     🟡 Yellow = Original (neighbor-based)")
        print("     🔵 Cyan = Sobel (gradient-based)")
        print("     🟣 Magenta = Morphological (dilation outline)")
        print("     🟢 Green = Canny (refined edges)")
        
        // ENHANCED MORPHOLOGICAL - Create thicker band
        print("🔶 [MORPH+] Creating enhanced morphological band...")
        
        // Create multiple dilation levels for thicker band
        var dilated2 = [UInt8](repeating: 0, count: width * height)
        var dilated3 = [UInt8](repeating: 0, count: width * height)
        
        // Second dilation (2-pixel expansion)
        for y in 2..<(height-2) {
            for x in 2..<(width-2) {
                let idx = y * width + x
                if cleanMask[idx] == 255 {
                    for dy in -2...2 {
                        for dx in -2...2 {
                            let newIdx = (y + dy) * width + (x + dx)
                            if newIdx >= 0 && newIdx < (width * height) {
                                dilated2[newIdx] = 255
                            }
                        }
                    }
                }
            }
        }
        
        // Third dilation (3-pixel expansion)
        for y in 3..<(height-3) {
            for x in 3..<(width-3) {
                let idx = y * width + x
                if cleanMask[idx] == 255 {
                    for dy in -3...3 {
                        for dx in -3...3 {
                            let newIdx = (y + dy) * width + (x + dx)
                            if newIdx >= 0 && newIdx < (width * height) {
                                dilated3[newIdx] = 255
                            }
                        }
                    }
                }
            }
        }
        
        // Create band visualization
        UIGraphicsBeginImageContextWithOptions(CGSize(width: width, height: height), false, 1.0)
        guard let bandCtx = UIGraphicsGetCurrentContext() else { return }
        
        // Black background
        bandCtx.setFillColor(UIColor.black.cgColor)
        bandCtx.fill(CGRect(x: 0, y: 0, width: width, height: height))
        
        // Draw graduated band
        for y in 0..<height {
            for x in 0..<width {
                let idx = y * width + x
                
                // Original furniture (white)
                if cleanMask[idx] == 255 {
                    bandCtx.setFillColor(UIColor.white.cgColor)
                    bandCtx.fill(CGRect(x: x, y: y, width: 1, height: 1))
                }
                // 3-pixel band (dark red)
                else if dilated3[idx] == 255 {
                    bandCtx.setFillColor(UIColor.red.withAlphaComponent(0.3).cgColor)
                    bandCtx.fill(CGRect(x: x, y: y, width: 1, height: 1))
                }
                // 2-pixel band (medium red)  
                else if dilated2[idx] == 255 {
                    bandCtx.setFillColor(UIColor.red.withAlphaComponent(0.6).cgColor)
                    bandCtx.fill(CGRect(x: x, y: y, width: 1, height: 1))
                }
                // 1-pixel band (bright red)
                else if dilated[idx] == 255 {
                    bandCtx.setFillColor(UIColor.red.cgColor)
                    bandCtx.fill(CGRect(x: x, y: y, width: 1, height: 1))
                }
            }
        }
        
        guard let bandImage = UIGraphicsGetImageFromCurrentImageContext() else { return }
        UIGraphicsEndImageContext()
        
        // Save enhanced morphological band
        UIImageWriteToSavedPhotosAlbum(bandImage, nil, nil, nil)
        print("🔶 [MORPH+] Saved enhanced morphological band visualization!")
        print("     ⚪ White = Original furniture")
        print("     🔴 Red bands = 1-3 pixel expansion zones")
        
        print("🪑 [CONTOUR] Contour-based largest object processing complete!")
    }
    
    // PRODUCTION: Real furniture colors, fully opaque, transparent background
    private func applyMaskToImage(mask: [Float], to pixelBuffer: CVPixelBuffer) {
        autoreleasepool {
            let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
            let width = CVPixelBufferGetWidth(pixelBuffer)
            let height = CVPixelBufferGetHeight(pixelBuffer)
            
            // Calculate mask bounding box before processing
            let calculatedMaskBBox = calculateMaskBoundingBox(from: mask, originalImageWidth: width, originalImageHeight: height)
            
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
                        pixels[idx + 3] = 0
                        continue
                    }
                    
                    let maskValue = mask[y0 * 160 + x0]
                    
                    if maskValue > 0.2 {
                        // Keep original colors, just set alpha to fully opaque
                        pixels[idx + 3] = 255
                    } else {
                        // Transparent background
                        pixels[idx + 3] = 0
                    }
                }
            }
            
            fillHolesInChair(pixels: pixels, width: width, height: height)
            
            if let outImage = ctx.makeImage() {
                DispatchQueue.main.async {
                    self.segmentedImage = UIImage(cgImage: outImage, scale: 1.0, orientation: .up)
                    self.maskBBox = calculatedMaskBBox  // Update mask bounding box
                    withAnimation(.easeIn(duration: 0.3)) { self.furnitureOpacity = 1.0 }
                    self.isProcessing = false
                }
            } else {
                DispatchQueue.main.async { self.isProcessing = false }
            }
        }
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
    
    private func processYOLOResults(_ detections: MLMultiArray, prototypes: MLMultiArray, originalImage: CVPixelBuffer) {
        print("\n📱 ==================== FRAME PROCESSING START ====================")
        
        let allDetections = extractDetections(from: detections)
        print("📊 [DETECTION] Extracted \(allDetections.count) raw detections")
        
        let nmsDetections = applyNMS(detections: allDetections, iouThreshold: 0.45)
        print("📊 [NMS] Filtered to \(nmsDetections.count) detections after NMS")
        
        guard let best = nmsDetections.first else {
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
        
        print("✅ [BEST] Selected: \(best.className) @ \(Int(best.confidence * 100))%")
        print("   Position: (\(Int(best.x)), \(Int(best.y))), Size: \(Int(best.width))x\(Int(best.height))")
        
        // Save original image
        saveDebugImage(pixelBuffer: originalImage, stage: "1_original")
        
        // CROPPING LOGIC STARTS
        print("\n🔍 ========== CROPPING PHASE ==========")
        let imageWidth = CVPixelBufferGetWidth(originalImage)
        let imageHeight = CVPixelBufferGetHeight(originalImage)
        print("📐 [IMAGE] Original dimensions: \(imageWidth)x\(imageHeight)")
        
        // Convert detection coords from 640x640 to actual image size
        let scaleX = Float(imageWidth) / 640.0
        let scaleY = Float(imageHeight) / 640.0
        print("📐 [SCALE] X: \(String(format: "%.2f", scaleX)), Y: \(String(format: "%.2f", scaleY))")
        
        // Calculate crop region with 30% padding
        let padding: Float = 1.3
        let cropWidth = max(100, best.width * scaleX * padding)
        let cropHeight = max(100, best.height * scaleY * padding)
        
        let centerX = best.x * scaleX
        let centerY = best.y * scaleY
        
        let cropX = Int(max(0, centerX - cropWidth/2))
        let cropY = Int(max(0, centerY - cropHeight/2))
        let cropW = Int(min(Float(imageWidth - cropX), cropWidth))
        let cropH = Int(min(Float(imageHeight - cropY), cropHeight))
        
        print("📐 [CROP] Center: (\(Int(centerX)), \(Int(centerY)))")
        print("📐 [CROP] Region: origin(\(cropX), \(cropY)), size(\(cropW)x\(cropH))")
        
        // Skip crop if too small
        if cropW >= 100 && cropH >= 100 {
            print("✅ [CROP] Size valid, attempting crop...")
            
            // Save image with bbox drawn
            saveDebugImageWithBBox(pixelBuffer: originalImage, bbox: best, stage: "2_bbox_marked")
            
            // Try cropping for better detail
            if let croppedBuffer = cropPixelBuffer(originalImage, x: cropX, y: cropY, width: cropW, height: cropH) {
                print("✅ [CROP] Successfully cropped image")
                saveDebugImage(pixelBuffer: croppedBuffer, stage: "3_cropped_region")
                
                if let croppedResized = resizePixelBuffer(croppedBuffer, width: 640, height: 640) {
                    print("✅ [CROP] Resized to 640x640")
                    saveDebugImage(pixelBuffer: croppedResized, stage: "4_cropped_resized")
                    
                    if let croppedArray = pixelBufferToMLMultiArray(croppedResized),
                       let model = mlModel,
                       let croppedInput = try? MLDictionaryFeatureProvider(dictionary: ["image": croppedArray]),
                       let croppedOutput = try? model.prediction(from: croppedInput),
                       let croppedDetections = croppedOutput.featureValue(for: "var_2421")?.multiArrayValue,
                       let croppedPrototypes = croppedOutput.featureValue(for: "p")?.multiArrayValue {
                        
                        print("✅ [CROP] Inference successful on cropped region")
                        
                        // Process multiple cropped detections
                        processCroppedMultiMask(croppedDetections, croppedPrototypes: croppedPrototypes,
                                               originalBest: best, originalImage: originalImage,
                                               originalPrototypes: prototypes)  // Pass original prototypes
                        return
                        
                    } else {
                        print("❌ [CROP] Inference failed on cropped region")
                    }
                } else {
                    print("❌ [CROP] Failed to resize cropped buffer")
                }
            } else {
                print("❌ [CROP] Failed to crop pixel buffer")
            }
        } else {
            print("⚠️ [CROP] Crop dimensions too small: \(cropW)x\(cropH), minimum 100x100")
        }
        
        // FALLBACK TO ORIGINAL
        print("⚠️ [FALLBACK] Using original detection without crop")
        processOriginalMask(best, prototypes: prototypes, originalImage: originalImage)
    }

//    private func processCroppedMultiMask(_ croppedDetections: MLMultiArray, croppedPrototypes: MLMultiArray,
//                                         originalBest: DetectionSmarty, originalImage: CVPixelBuffer,
//                                         originalPrototypes: MLMultiArray) {
//        print("\n🎨 ========== MULTI-DETECTION MASK GENERATION ==========")
//
//        let croppedAllDetections = extractDetections(from: croppedDetections)
//        print("📊 [CROP] Found \(croppedAllDetections.count) raw detections in crop")
//
//        let croppedNMS = applyNMS(detections: croppedAllDetections, iouThreshold: 0.45)
//        print("📊 [CROP] After NMS: \(croppedNMS.count) detections")
//
//        guard !croppedNMS.isEmpty else {
//            print("⚠️ [CROP] No detections found in cropped region")
//            processOriginalMask(originalBest, prototypes: originalPrototypes, originalImage: originalImage)
//            return
//        }
//
//        // Use IoU to filter truly unique detections
//        var uniqueDetections: [DetectionSmarty] = []
//
//        for detection in croppedNMS {
//            // Check if this overlaps too much with already selected detections
//            var shouldAdd = true
//
//            for existing in uniqueDetections {
//                // Calculate IoU between this detection and existing ones
//                let iou = calculateIoU(det1: detection, det2: existing)
//
//                // If high overlap AND same class, skip it
//                if iou > 0.7 && detection.classIdx == existing.classIdx {
//                    print("   ⚠️ Skipping duplicate: \(detection.className) @ \(Int(detection.confidence * 100))% (IoU=\(Int(iou*100))% with existing)")
//                    shouldAdd = false
//                    break
//                }
//
//                // If moderate overlap but different class, keep it (pillow on chair case)
//                if iou > 0.3 && iou < 0.7 && detection.classIdx != existing.classIdx {
//                    print("   ℹ️ Keeping overlapping different class: \(detection.className) on \(existing.className) (IoU=\(Int(iou*100))%)")
//                }
//            }
//
//            if shouldAdd {
//                uniqueDetections.append(detection)
//                print("   ✅ Adding: \(detection.className) @ \(Int(detection.confidence * 100))% at pos(\(Int(detection.x)),\(Int(detection.y)))")
//
//                // Limit to 4 detections
//                if uniqueDetections.count >= 4 { break }
//            }
//        }
//
//        print("📊 [MULTI] Using \(uniqueDetections.count) unique detections")
//
//        // Generate combined mask from all unique detections
//        var combinedMask = [Float](repeating: 0, count: 160 * 160)
//
//        for (index, detection) in uniqueDetections.enumerated() {
//            var detectionMask = [Float](repeating: 0, count: 160 * 160)
//
//            for y in 0..<160 {
//                for x in 0..<160 {
//                    var sum: Float = 0
//                    for c in 0..<32 {
//                        sum += detection.maskCoeffs[c] * croppedPrototypes[[0, c, y, x] as [NSNumber]].floatValue
//                    }
//                    detectionMask[y * 160 + x] = sigmoid(sum)
//                }
//            }
//
//            // Save individual masks for debugging
//            saveMaskAsImage(mask: detectionMask, stage: "5_crop_mask_\(index+1)_\(detection.className)")
//
//            // Combine masks
//            for i in 0..<(160 * 160) {
//                combinedMask[i] = max(combinedMask[i], detectionMask[i])
//            }
//        }
//
//        let nonZeroCount = combinedMask.filter { $0 > 0.5 }.count
//        print("📊 [MASK] Combined mask has \(nonZeroCount) positive pixels")
//
//        saveMaskAsImage(mask: combinedMask, stage: "5_combined_multi_mask")
//
//        let bbox = CGRect(
//            x: CGFloat(originalBest.x - originalBest.width / 2),
//            y: CGFloat(originalBest.y - originalBest.height / 2),
//            width: CGFloat(originalBest.width),
//            height: CGFloat(originalBest.height)
//        )
//
//        DispatchQueue.main.async {
//            self.currentBBox = bbox
//            self.lastConfidence = originalBest.confidence
//        }
//
//        applyPostProcessingAndMask(mask: combinedMask, best: originalBest, to: originalImage, stage: "multi")
//    }
    
    
    private func processCroppedMultiMask(_ croppedDetections: MLMultiArray, croppedPrototypes: MLMultiArray,
                                         originalBest: DetectionSmarty, originalImage: CVPixelBuffer,
                                         originalPrototypes: MLMultiArray) {
        print("\n🎨 ========== MULTI-DETECTION MASK GENERATION ==========")
        
        // First, generate the ORIGINAL mask (to avoid holes)
        var originalMask = [Float](repeating: 0, count: 160 * 160)
        for y in 0..<160 {
            for x in 0..<160 {
                var sum: Float = 0
                for c in 0..<32 {
                    sum += originalBest.maskCoeffs[c] * originalPrototypes[[0, c, y, x] as [NSNumber]].floatValue
                }
                originalMask[y * 160 + x] = sigmoid(sum)
            }
        }
        print("✅ [ORIGINAL] Generated mask for: \(originalBest.className)")
        
        let croppedAllDetections = extractDetections(from: croppedDetections)
        print("📊 [CROP] Found \(croppedAllDetections.count) raw detections")
        
//        for detection in croppedAllDetections {
//            if detection.className.contains("daybed") || detection.className.contains("day bed") {
//                print("🛏️ \(detection.className): pos(\(Int(detection.x)),\(Int(detection.y))) size(\(Int(detection.width))x\(Int(detection.height)))")
//            }
//        }
        
        // Apply HIERARCHICAL NMS instead of standard NMS
//        let hierarchicalDetections = applyHierarchicalNMS(detections: croppedAllDetections, iouThreshold: 0.45)
        let hierarchicalDetections = croppedAllDetections
        print("📊 [H-NMS] Kept \(hierarchicalDetections.count) detections after hierarchical NMS")
        
        print("📊 [HierarchyKKK] Selected \(hierarchicalDetections.count)  detections:")
            for det in hierarchicalDetections {
                print("   - \(det.className) @ \(Int(det.confidence * 100))%")
                print("   - \(det.className) @ \(Int(det.confidence * 100))% size:\(Int(det.width))x\(Int(det.height))")
            }

        
        // Take top 10 detections
//        let topDetections = Array(hierarchicalDetections.prefix(10))
        
        let topDetections = getDiverseDetections(from: hierarchicalDetections, maxCount: 10)

        
        print("📊 [MULTI] Processing top \(topDetections.count) detections:")
        for det in topDetections {
            print("   - \(det.className) @ \(Int(det.confidence * 100))% size:\(Int(det.width))x\(Int(det.height))")
        }
        
        // Generate combined mask starting with original
        var combinedMask = originalMask
        
        for (index, detection) in topDetections.enumerated() {
            var detectionMask = [Float](repeating: 0, count: 160 * 160)
            
            for y in 0..<160 {
                for x in 0..<160 {
                    var sum: Float = 0
                    for c in 0..<32 {
                        sum += detection.maskCoeffs[c] * croppedPrototypes[[0, c, y, x] as [NSNumber]].floatValue
                    }
                    detectionMask[y * 160 + x] = sigmoid(sum)
                }
            }
            
            if index < 3 {  // Save first 3 for debugging
                saveMaskAsImage(mask: detectionMask, stage: "5_crop_\(index+1)_\(detection.className)")
            }
            
            // Combine masks
            for i in 0..<(160 * 160) {
//                combinedMask[i] = max(combinedMask[i], detectionMask[i])
                combinedMask[i] = min(1.0, combinedMask[i] + detectionMask[i] * 0.5)
            }
        }
        
        let nonZeroCount = combinedMask.filter { $0 > 0.5 }.count
        print("📊 [FINAL] Combined mask has \(nonZeroCount) positive pixels")
        
        saveMaskAsImage(mask: combinedMask, stage: "5_combined_hierarchical")
        
        let bbox = CGRect(
            x: CGFloat(originalBest.x - originalBest.width / 2),
            y: CGFloat(originalBest.y - originalBest.height / 2),
            width: CGFloat(originalBest.width),
            height: CGFloat(originalBest.height)
        )
        
        DispatchQueue.main.async {
            self.currentBBox = bbox
            self.lastConfidence = originalBest.confidence
        }
        
        applyPostProcessingAndMask(mask: combinedMask, best: originalBest, to: originalImage, stage: "hierarchical")
    }
    
    private func getDiverseDetections(from detections: [DetectionSmarty], maxCount: Int) -> [DetectionSmarty] {
        var selected: [DetectionSmarty] = []
        var seenClasses = Set<Int>()
        
        // First pass: Take best of each unique class
        for detection in detections {
            if !seenClasses.contains(detection.classIdx) {
                selected.append(detection)
                seenClasses.insert(detection.classIdx)
                if selected.count >= maxCount { break }
            }
        }
        
        // Second pass: If still room, add remaining by confidence
        if selected.count < maxCount {
            for detection in detections {
                if !selected.contains(where: { $0.classIdx == detection.classIdx && $0.confidence == detection.confidence }) {
                    selected.append(detection)
                    if selected.count >= maxCount { break }
                }
            }
        }
        
        print("📊 [DIVERSE] Selected \(selected.count) diverse detections:")
        for det in selected {
            print("   - \(det.className) @ \(Int(det.confidence * 100))%")
        }
        
        return selected
    }

    // New Hierarchical NMS function
    private func applyHierarchicalNMS(detections: [DetectionSmarty], iouThreshold: Float) -> [DetectionSmarty] {
        guard !detections.isEmpty else { return [] }
        
        var kept: [DetectionSmarty] = []
        var suppressed = Set<Int>()
        let sorted = detections.sorted { $0.confidence > $1.confidence }
        
        print("\n🔍 Hierarchical NMS Processing:")
        
        for (i, det) in sorted.enumerated() {
            if suppressed.contains(i) { continue }
            
            // Check if this should be kept
            var shouldSuppress = false
            var reason = ""
            
            for existing in kept {
                let iou = calculateDetectionIoU(det1: det, det2: existing)
                let sizeRatio = (det.width * det.height) / (existing.width * existing.height)
                let isInside = isBoxInside(small: det, large: existing)
                
                // Hierarchical rules:
                if iou > iouThreshold {
                    // Rule 1: Different classes + one inside other = KEEP BOTH
                    if det.classIdx != existing.classIdx && (isInside || sizeRatio < 0.5) {
                        reason = "Different class, hierarchical (\(det.className) on \(existing.className))"
                        shouldSuppress = false
                        break
                    }
                    
                    // Rule 2: Same class + similar size = SUPPRESS
                    if det.classIdx == existing.classIdx && sizeRatio > 0.7 && sizeRatio < 1.3 {
                        reason = "Same class duplicate"
                        shouldSuppress = true
                        break
                    }
                    
                    // Rule 3: Significantly different sizes = KEEP (could be parts)
                    if sizeRatio < 0.3 || sizeRatio > 3.0 {
                        reason = "Size difference (ratio: \(String(format: "%.2f", sizeRatio)))"
                        shouldSuppress = false
                    }
                    // Rule 4: Default high IoU = SUPPRESS
                    else if iou > 0.7 {
                        reason = "High overlap"
                        shouldSuppress = true
                        break
                    }
                }
            }
            
            if !shouldSuppress {
                kept.append(det)
                if !reason.isEmpty {
                    print("✅ KEPT: \(det.className) @ \(Int(det.confidence * 100))% - \(reason)")
                } else {
                    print("✅ KEPT: \(det.className) @ \(Int(det.confidence * 100))%")
                }
            } else {
                suppressed.insert(i)
                print("❌ SUPPRESSED: \(det.className) @ \(Int(det.confidence * 100))% - \(reason)")
            }
        }
        
        print("Hierarchical NMS: \(sorted.count) → \(kept.count) detections")
        
        return kept
    }

    // Helper: Calculate IoU between detections
    private func calculateDetectionIoU(det1: DetectionSmarty, det2: DetectionSmarty) -> Float {
        let x1 = max(det1.x - det1.width/2, det2.x - det2.width/2)
        let y1 = max(det1.y - det1.height/2, det2.y - det2.height/2)
        let x2 = min(det1.x + det1.width/2, det2.x + det2.width/2)
        let y2 = min(det1.y + det1.height/2, det2.y + det2.height/2)
        
        let intersection = max(0, x2 - x1) * max(0, y2 - y1)
        let union = det1.width * det1.height + det2.width * det2.height - intersection
        
        return union > 0 ? intersection / union : 0
    }

    // Helper: Check if smaller box is inside larger box
    private func isBoxInside(small: DetectionSmarty, large: DetectionSmarty) -> Bool {
        let smallLeft = small.x - small.width/2
        let smallRight = small.x + small.width/2
        let smallTop = small.y - small.height/2
        let smallBottom = small.y + small.height/2
        
        let largeLeft = large.x - large.width/2
        let largeRight = large.x + large.width/2
        let largeTop = large.y - large.height/2
        let largeBottom = large.y + large.height/2
        
        // Check if small box is mostly inside large box (80% threshold)
        let overlapX = min(smallRight, largeRight) - max(smallLeft, largeLeft)
        let overlapY = min(smallBottom, largeBottom) - max(smallTop, largeTop)
        let overlapArea = overlapX * overlapY
        let smallArea = small.width * small.height
        
        return overlapArea > (smallArea * 0.8)
    }

    // Helper function to calculate IoU
    private func calculateIoU(det1: DetectionSmarty, det2: DetectionSmarty) -> Float {
        // Calculate intersection
        let x1 = max(det1.x - det1.width/2, det2.x - det2.width/2)
        let y1 = max(det1.y - det1.height/2, det2.y - det2.height/2)
        let x2 = min(det1.x + det1.width/2, det2.x + det2.width/2)
        let y2 = min(det1.y + det1.height/2, det2.y + det2.height/2)
        
        let intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        // Calculate union
        let area1 = det1.width * det1.height
        let area2 = det2.width * det2.height
        let union = area1 + area2 - intersection
        
        // Return IoU
        return union > 0 ? intersection / union : 0
    }


//    // Enhanced cropped mask processing with multiple detections
//    private func processCroppedMask(_ detection: DetectionSmarty, croppedPrototypes: MLMultiArray, originalBest: DetectionSmarty, originalImage: CVPixelBuffer) {
//        print("\n🎨 ========== CROPPED MASK GENERATION (MULTI-DETECTION) ==========")
//
//        // Get top detections from crop
//        let croppedAllDetections = extractDetections(from: croppedDetections)
//        let croppedNMS = applyNMS(detections: croppedAllDetections, iouThreshold: 0.45)
//
//        // Take top 4 detections
//        let topDetections = Array(croppedNMS.prefix(4))
//        print("📊 [MULTI] Processing top \(topDetections.count) detections from crop")
//
//        // Filter out duplicate of original detection
//        let filteredDetections = topDetections.filter { cropped in
//            // Check if this is essentially the same furniture as original
//            // If confidence is very similar and class is same, likely duplicate
//            let confDiff = abs(cropped.confidence - originalBest.confidence)
//            let sameClass = cropped.classIdx == originalBest.classIdx
//            let isDuplicate = sameClass && confDiff < 0.1
//
//            if isDuplicate {
//                print("   ⚠️ Skipping duplicate: \(cropped.className) @ \(Int(cropped.confidence * 100))% (original was \(Int(originalBest.confidence * 100))%)")
//            }
//            return !isDuplicate
//        }
//
//        print("📊 [MULTI] After filtering duplicates: \(filteredDetections.count) unique detections")
//
//        // Generate combined mask from all detections
//        var combinedMask = [Float](repeating: 0, count: 160 * 160)
//
//        for (index, detection) in filteredDetections.enumerated() {
//            print("   Processing #\(index + 1): \(detection.className) @ \(Int(detection.confidence * 100))%")
//
//            var detectionMask = [Float](repeating: 0, count: 160 * 160)
//            for y in 0..<160 {
//                for x in 0..<160 {
//                    var sum: Float = 0
//                    for c in 0..<32 {
//                        sum += detection.maskCoeffs[c] * croppedPrototypes[[0, c, y, x] as [NSNumber]].floatValue
//                    }
//                    detectionMask[y * 160 + x] = sigmoid(sum)
//                }
//            }
//
//            // Combine masks using maximum (union)
//            for i in 0..<(160 * 160) {
//                combinedMask[i] = max(combinedMask[i], detectionMask[i])
//            }
//        }
//
//        let nonZeroCount = combinedMask.filter { $0 > 0.5 }.count
//        print("📊 [MASK] Combined mask has \(nonZeroCount) positive pixels")
//
//        saveMaskAsImage(mask: combinedMask, stage: "5_cropped_combined_mask")
//
//        // Use original bbox for UI
//        let bbox = CGRect(
//            x: CGFloat(originalBest.x - originalBest.width / 2),
//            y: CGFloat(originalBest.y - originalBest.height / 2),
//            width: CGFloat(originalBest.width),
//            height: CGFloat(originalBest.height)
//        )
//
//        DispatchQueue.main.async {
//            self.currentBBox = bbox
//            self.lastConfidence = originalBest.confidence
//        }
//
//        applyPostProcessingAndMask(mask: combinedMask, best: originalBest, to: originalImage, stage: "cropped_multi")
//    }

    // Enhanced original mask processing with logging
    private func processOriginalMask(_ best: DetectionSmarty, prototypes: MLMultiArray, originalImage: CVPixelBuffer) {
        print("\n🎨 ========== ORIGINAL MASK GENERATION ==========")
        print("📊 [MASK] Generating from original detection")
        
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
        
        var mask = [Float](repeating: 0, count: 160 * 160)
        var nonZeroCount = 0
        
        for y in 0..<160 {
            for x in 0..<160 {
                var sum: Float = 0
                for c in 0..<32 {
                    sum += best.maskCoeffs[c] * prototypes[[0, c, y, x] as [NSNumber]].floatValue
                }
                mask[y * 160 + x] = sigmoid(sum)
                if mask[y * 160 + x] > 0.5 { nonZeroCount += 1 }
            }
        }
        
        print("📊 [MASK] Generated mask with \(nonZeroCount) positive pixels")
        saveMaskAsImage(mask: mask, stage: "5_original_mask_raw")
        
        applyPostProcessingAndMask(mask: mask, best: best, to: originalImage, stage: "original")
    }



    // Add the crop pixel buffer method
    private func cropPixelBuffer(_ pixelBuffer: CVPixelBuffer, x: Int, y: Int, width: Int, height: Int) -> CVPixelBuffer? {
        guard width > 50 && height > 50 else { return nil }
        
        let imageWidth = CVPixelBufferGetWidth(pixelBuffer)
        let imageHeight = CVPixelBufferGetHeight(pixelBuffer)
        
        let validX = max(0, min(x, imageWidth - 1))
        let validY = max(0, min(y, imageHeight - 1))
        let validWidth = min(width, imageWidth - validX)
        let validHeight = min(height, imageHeight - validY)
        
        guard validWidth >= 50 && validHeight >= 50 else { return nil }
        
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let cropRect = CGRect(x: validX, y: validY, width: validWidth, height: validHeight)
        
        guard ciImage.extent.intersects(cropRect) else { return nil }
        
        let croppedImage = ciImage.cropped(to: cropRect)
        
        var newPixelBuffer: CVPixelBuffer?
        CVPixelBufferCreate(kCFAllocatorDefault, validWidth, validHeight, kCVPixelFormatType_32BGRA, nil, &newPixelBuffer)
        
        guard let outputBuffer = newPixelBuffer else { return nil }
        context.render(croppedImage, to: outputBuffer, bounds: croppedImage.extent, colorSpace: CGColorSpaceCreateDeviceRGB())
        
        return outputBuffer
    }
    
    // Enhanced post-processing with logging
    private func applyPostProcessingAndMask(mask: [Float], best: DetectionSmarty, to originalImage: CVPixelBuffer, stage: String) {
        print("\n🔧 ========== POST-PROCESSING ==========")
        var mask = mask
        
        let scale: Float = 160.0 / 640.0
        let bx1 = max(1, min(158, Int((best.x - best.width/2) * scale)))
        let by1 = max(1, min(158, Int((best.y - best.height/2) * scale)))
        let bx2 = max(1, min(158, Int((best.x + best.width/2) * scale)))
        let by2 = max(1, min(158, Int((best.y + best.height/2) * scale)))
        
        print("📐 [BBOX] Mask space: (\(bx1),\(by1)) to (\(bx2),\(by2))")
        
        var binary = [[UInt8]](repeating: [UInt8](repeating: 0, count: 160), count: 160)
        var binaryCount = 0
        for y in 0..<160 {
            for x in 0..<160 {
                binary[y][x] = mask[y * 160 + x] > 0.2 ? 1 : 0
                if binary[y][x] == 1 { binaryCount += 1 }
            }
        }
        print("📊 [BINARY] Converted to binary: \(binaryCount) pixels")
        
        // Morphological closing
        print("🔧 [MORPH] Applying dilation (5 iterations)...")
        for i in 0..<5 {
            var dilated = binary
            var changeCount = 0
            for y in (by1+1)..<by2 {
                for x in (bx1+1)..<bx2 {
                    if binary[y][x] == 0 {
                        if binary[y-1][x] == 1 || binary[y+1][x] == 1 ||
                           binary[y][x-1] == 1 || binary[y][x+1] == 1 ||
                           binary[y-1][x-1] == 1 || binary[y-1][x+1] == 1 ||
                           binary[y+1][x-1] == 1 || binary[y+1][x+1] == 1 {
                            dilated[y][x] = 1
                            changeCount += 1
                        }
                    }
                }
            }
            binary = dilated
            print("   Iteration \(i+1): \(changeCount) pixels added")
        }
//        
//        saveMaskAsImage(mask: binaryToFloat(binary), stage: "6_\(stage)_dilated")
//        
//        print("🔧 [MORPH] Applying erosion (5 iterations)...")
//        for i in 0..<5 {
//            var eroded = binary
//            var changeCount = 0
//            for y in (by1+1)..<by2 {
//                for x in (bx1+1)..<bx2 {
//                    if binary[y][x] == 1 {
//                        if binary[y-1][x] == 0 || binary[y+1][x] == 0 ||
//                           binary[y][x-1] == 0 || binary[y][x+1] == 0 {
//                            eroded[y][x] = 0
//                            changeCount += 1
//                        }
//                    }
//                }
//            }
//            binary = eroded
//            print("   Iteration \(i+1): \(changeCount) pixels removed")
//        }
        
//        saveMaskAsImage(mask: binaryToFloat(binary), stage: "7_\(stage)_eroded")
        
//        // Convert back to float
//        var finalCount = 0
//        for y in 0..<160 {
//            for x in 0..<160 {
//                mask[y * 160 + x] = Float(binary[y][x])
//                if mask[y * 160 + x] > 0 { finalCount += 1 }
//            }
//        }
//        
//        // Crop to bbox
//        var croppedCount = 0
//        for y in 0..<160 {
//            for x in 0..<160 {
//                if y < by1 || y > by2 || x < bx1 || x > bx2 {
//                    mask[y * 160 + x] = 0
//                } else if mask[y * 160 + x] > 0 {
//                    croppedCount += 1
//                }
//            }
//        }
        
//        print("📊 [FINAL] Mask pixels after morphology: \(finalCount)")
//        print("📊 [FINAL] After bbox crop: \(croppedCount) pixels")
        
//        saveMaskAsImage(mask: mask, stage: "8_\(stage)_final_mask")
        
        print("🎨 [APPLY] Applying final mask to image")
        applyMaskToImage(mask: mask, to: originalImage)
        
        print("✅ ==================== FRAME COMPLETE ====================\n")
    }
    
    // Helper to convert binary to float for visualization
    private func binaryToFloat(_ binary: [[UInt8]]) -> [Float] {
        var result = [Float](repeating: 0, count: 160 * 160)
        for y in 0..<160 {
            for x in 0..<160 {
                result[y * 160 + x] = Float(binary[y][x])
            }
        }
        return result
    }
    
    // Debug image saving helpers
    private func saveDebugImage(pixelBuffer: CVPixelBuffer, stage: String) {
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else { return }
        let uiImage = UIImage(cgImage: cgImage)
        UIImageWriteToSavedPhotosAlbum(uiImage, nil, nil, nil)
        print("💾 [SAVE] Saved: \(stage)")
    }

    private func saveDebugImageWithBBox(pixelBuffer: CVPixelBuffer, bbox: DetectionSmarty, stage: String) {
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else { return }
        
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        
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
        
        ctx.setStrokeColor(UIColor.green.cgColor)
        ctx.setLineWidth(3)
        ctx.stroke(bboxRect)
        
        guard let finalImage = UIGraphicsGetImageFromCurrentImageContext() else { return }
        UIGraphicsEndImageContext()
        
        UIImageWriteToSavedPhotosAlbum(finalImage, nil, nil, nil)
        print("💾 [SAVE] Saved: \(stage)")
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
