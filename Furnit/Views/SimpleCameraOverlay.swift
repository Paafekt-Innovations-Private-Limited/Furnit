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
                            Text("Size: \(Int(scaleMultiplier * 100))%")
                                .font(.caption)
                                .padding(8)
                                .background(Capsule().fill(Color.black.opacity(0.5)))
                                .foregroundColor(.white)
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
    private var lockedFastSAMDetection: Int? = nil  // Lock detection for stability
    
    private let frameWidth: Float = 640
    private let frameHeight: Float = 640
    
    override init() {
        super.init()
        checkCameraAuthorization()
        loadFastSAMModel()
        loadU2NetModel()
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
            print("⏳ Waiting for U2-Net...")
            isProcessing = false
            return
        }
        
        // Process with FastSAM if available
        guard let fastSAMModel = fastSAMModel else {
            // Use U2-Net only
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
            
            applyThreeTierRefinement(detections: detections, prototypes: prototypes,
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
                print("✅ U2-Net detected furniture")
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
    
    private func applyThreeTierRefinement(detections: MLMultiArray, prototypes: MLMultiArray,
                                         originalPixelBuffer: CVPixelBuffer) {
        
        let numDetections = detections.shape[2].intValue
        let numValues = detections.shape[1].intValue
        let numPrototypes = prototypes.shape[1].intValue
        let protoHeight = prototypes.shape[2].intValue
        let protoWidth = prototypes.shape[3].intValue
        
        let detPointer = detections.dataPointer.assumingMemoryBound(to: Float.self)
        
        guard let u2netMask = u2netMask else {
            print("❌ No U2-Net mask")
            return
        }
        
        // Find or use locked FastSAM detection
        var detectionToUse: Int? = lockedFastSAMDetection
        
        if detectionToUse == nil {
            // Find detection with best U2-Net overlap (only once)
            var maxOverlap: Float = 0
            for i in 0..<numDetections {
                let conf = detPointer[4 * numDetections + i]
                if conf > 0.2 {
                    let x = detPointer[0 * numDetections + i]
                    let y = detPointer[1 * numDetections + i]
                    let w = detPointer[2 * numDetections + i]
                    let h = detPointer[3 * numDetections + i]
                    
                    let overlap = calculateU2NetOverlap(x: x, y: y, width: w, height: h)
                    if overlap > maxOverlap && overlap > 0.3 {
                        maxOverlap = overlap
                        detectionToUse = i
                    }
                }
            }
            
            // Lock this detection forever
            if let detection = detectionToUse {
                lockedFastSAMDetection = detection
                print("🔒 Locked FastSAM detection #\(detection) with \(String(format: "%.1f%%", maxOverlap * 100)) overlap")
            }
        }
        
        guard let detIdx = detectionToUse else {
            // No matching FastSAM, use U2-Net only
            let maskImage = CIImage(cvPixelBuffer: u2netMask)
            let ciImage = CIImage(cvPixelBuffer: originalPixelBuffer)
            applyMaskToImage(original: ciImage, mask: maskImage)
            return
        }
        
        // Extract coefficients for locked detection
        var maskCoeffs = [Float](repeating: 0, count: numPrototypes)
        for j in 0..<numPrototypes {
            let coeffIdx = (numValues - numPrototypes + j) * numDetections + detIdx
            maskCoeffs[j] = detPointer[coeffIdx]
        }
        
        // Generate FastSAM confidence map
        let protoPointer = prototypes.dataPointer.assumingMemoryBound(to: Float.self)
        let maskSize = protoHeight * protoWidth
        
        var fastSAMConfidence = [Float](repeating: 0, count: maskSize)
        for protoIdx in 0..<maskCoeffs.count {
            let coeff = maskCoeffs[protoIdx]
            let protoOffset = protoIdx * maskSize
            
            for i in 0..<maskSize {
                fastSAMConfidence[i] += coeff * protoPointer[protoOffset + i]
            }
        }
        
        // Convert to probabilities
        for i in 0..<maskSize {
            fastSAMConfidence[i] = 1.0 / (1.0 + exp(-fastSAMConfidence[i]))
        }
        
        // Apply three-tier strategy with CURRENT U2-Net detection
        CVPixelBufferLockBaseAddress(u2netMask, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(u2netMask, .readOnly) }
        
        let maskWidth = CVPixelBufferGetWidth(u2netMask)
        let maskHeight = CVPixelBufferGetHeight(u2netMask)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(u2netMask)
        
        guard let baseAddress = CVPixelBufferGetBaseAddress(u2netMask) else { return }
        let u2netPtr = baseAddress.assumingMemoryBound(to: UInt8.self)
        
        var pixelData = [UInt8](repeating: 0, count: maskSize)
        
        print("\n🎯 === THREE-TIER REFINEMENT ===")
        
        // First, mark all U2-Net regions
        var u2netRegions = [Bool](repeating: false, count: maskSize)
        for y in 0..<protoHeight {
            for x in 0..<protoWidth {
                let idx = y * protoWidth + x
                let sourceX = x * maskWidth / protoWidth
                let sourceY = y * maskHeight / protoHeight
                let sourceIdx = sourceY * bytesPerRow + sourceX
                
                if u2netPtr[sourceIdx] > 50 {  // Any U2-Net detection
                    u2netRegions[idx] = true
                }
            }
        }
        
        // Apply three tiers
        var tier1Pixels = 0
        var tier2Pixels = 0
        var tier3Pixels = 0
        
        for y in 0..<protoHeight {
            for x in 0..<protoWidth {
                let idx = y * protoWidth + x
                let sourceX = x * maskWidth / protoWidth
                let sourceY = y * maskHeight / protoHeight
                let sourceIdx = sourceY * bytesPerRow + sourceX
                
                let u2netValue = u2netPtr[sourceIdx]
                let fastSAMValue = fastSAMConfidence[idx]
                
                // TIER 1: Strong U2-Net (>150) - Always keep
                if u2netValue > 150 {
                    pixelData[idx] = 255
                    tier1Pixels += 1
                }
                // TIER 2: Weak U2-Net (50-150) + Strong FastSAM (>0.7) - Edge refinement
                else if u2netValue > 50 && u2netValue <= 150 && fastSAMValue > 0.7 {
                    pixelData[idx] = 255
                    tier2Pixels += 1
                }
                // TIER 3: No U2-Net but very strong FastSAM (>0.85) - Disconnected components
                else if u2netValue <= 50 && fastSAMValue > 0.85 {
                    // Check if this is a disconnected component (not just edge noise)
                    if isDisconnectedFurnitureComponent(x: x, y: y,
                                                        confidence: fastSAMConfidence,
                                                        u2netRegions: u2netRegions,
                                                        width: protoWidth, height: protoHeight) {
                        pixelData[idx] = 255
                        tier3Pixels += 1
                    }
                }
                // Everything else: Background (no action needed, already 0)
            }
        }
        
        print("📊 Tier 1 (Strong U2-Net): \(tier1Pixels) pixels")
        print("📊 Tier 2 (Edge refinement): \(tier2Pixels) pixels")
        print("📊 Tier 3 (Disconnected parts): \(tier3Pixels) pixels")
        print("📊 Total: \(tier1Pixels + tier2Pixels + tier3Pixels) pixels")
        
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
        
        // Apply fresh mask to current image
        let ciImage = CIImage(cvPixelBuffer: originalPixelBuffer)
        applyMaskToImage(original: ciImage, mask: finalMask)
    }
    
    private func isDisconnectedFurnitureComponent(x: Int, y: Int,
                                                  confidence: [Float],
                                                  u2netRegions: [Bool],
                                                  width: Int, height: Int) -> Bool {
        let idx = y * width + x
        
        // Must have high confidence cluster around this point
        var clusterSize = 0
        var touchesU2Net = false
        
        // Check 5x5 region around this pixel
        for dy in -2...2 {
            for dx in -2...2 {
                let ny = y + dy
                let nx = x + dx
                
                if ny >= 0 && ny < height && nx >= 0 && nx < width {
                    let nIdx = ny * width + nx
                    
                    // Count high confidence neighbors
                    if confidence[nIdx] > 0.8 {
                        clusterSize += 1
                    }
                    
                    // Check if directly touches U2-Net region
                    if abs(dy) <= 1 && abs(dx) <= 1 && u2netRegions[nIdx] {
                        touchesU2Net = true
                    }
                }
            }
        }
        
        // Disconnected component criteria:
        // 1. Must be a significant cluster (not noise)
        // 2. Should NOT directly touch U2-Net (that would be edge, not disconnected)
        // 3. But should be reasonably close to furniture (within ~10 pixels)
        
        if clusterSize < 15 || touchesU2Net {
            return false
        }
        
        // Check if reasonably close to U2-Net furniture
        var nearFurniture = false
        for dy in -10...10 {
            for dx in -10...10 {
                let ny = y + dy
                let nx = x + dx
                
                if ny >= 0 && ny < height && nx >= 0 && nx < width {
                    let nIdx = ny * width + nx
                    if u2netRegions[nIdx] {
                        nearFurniture = true
                        break
                    }
                }
            }
            if nearFurniture { break }
        }
        
        return nearFurniture
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
        
        // Light blur for anti-aliasing
        if let blurFilter = CIFilter(name: "CIGaussianBlur") {
            blurFilter.setValue(scaledMask, forKey: kCIInputImageKey)
            blurFilter.setValue(0.5, forKey: kCIInputRadiusKey)
            if let blurred = blurFilter.outputImage {
                finalMask = blurred
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

