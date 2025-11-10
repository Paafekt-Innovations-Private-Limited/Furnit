import SwiftUI
import AVFoundation
import CoreML
import Vision
import CoreImage
import Photos

// SEGMENT EXAMINE - FASTSAM-X PRIMARY
// FastSAM-X runs as PRIMARY segmentation (complete furniture objects)
// U2-Net provides SUPPORT (stabilization, gap-filling, refinement)

struct SegmentExamine: View {
    @Binding var capturedImage: UIImage?
    @Binding var isShowingCamera: Bool
    @StateObject private var camera = SegmentExamineModel()
    
    @State private var scaleMultiplier: CGFloat = 0.5
    @State private var dragOffset: CGSize = .zero
    @State private var accumulatedOffset: CGSize = .zero
    @State private var showingSaveSuccess = false
    
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
                        DragGesture()
                            .onChanged { value in dragOffset = value.translation }
                            .onEnded { value in
                                accumulatedOffset.width += value.translation.width
                                accumulatedOffset.height += value.translation.height
                                dragOffset = .zero
                            }
                    )
                    .ignoresSafeArea()
                    .opacity(camera.furnitureOpacity)
                    .animation(.easeOut(duration: 0.3), value: camera.furnitureOpacity)
            }
            
            if camera.isProcessingStarted {
                VStack {
                    Spacer()
                    VStack(spacing: 12) {
                        Text(camera.progressMessage)
                            .font(.headline)
                            .foregroundColor(.white)
                        ZStack(alignment: .leading) {
                            RoundedRectangle(cornerRadius: 8)
                                .fill(Color.white.opacity(0.3))
                                .frame(width: 280, height: 8)
                            RoundedRectangle(cornerRadius: 8)
                                .fill(Color.green)
                                .frame(width: 280 * camera.processingProgress, height: 8)
                                .animation(.easeInOut(duration: 0.2), value: camera.processingProgress)
                        }
                        Text("\(Int(camera.processingProgress * 100))%")
                            .font(.caption)
                            .foregroundColor(.white.opacity(0.8))
                    }
                    .padding(.vertical, 20)
                    .padding(.horizontal, 30)
                    .background(
                        RoundedRectangle(cornerRadius: 16)
                            .fill(Color.black.opacity(0.85))
                            .shadow(radius: 10)
                    )
                    .padding(.bottom, 200)
                }
                .transition(.move(edge: .bottom).combined(with: .opacity))
            }
            
            if showingSaveSuccess {
                VStack {
                    Spacer()
                    HStack(spacing: 12) {
                        Image(systemName: "checkmark.circle.fill").font(.title2)
                        Text("Furniture saved!").font(.headline)
                    }
                    .foregroundColor(.white)
                    .padding(.horizontal, 24)
                    .padding(.vertical, 16)
                    .background(Capsule().fill(Color.green.opacity(0.95)).shadow(radius: 10))
                    .padding(.bottom, 150)
                }
                .transition(.move(edge: .bottom).combined(with: .opacity))
            }
            
            VStack {
                HStack {
                    if camera.segmentedImage != nil {
                        HStack(spacing: 6) {
                            Image(systemName: "minus.magnifyingglass").foregroundColor(.white).font(.system(size: 14))
                            Slider(value: $scaleMultiplier, in: 0.3...1.0).frame(width: 150).accentColor(.white)
                            Image(systemName: "plus.magnifyingglass").foregroundColor(.white).font(.system(size: 14))
                        }
                        .padding(.horizontal, 12).padding(.vertical, 8)
                        .background(Capsule().fill(Color.black.opacity(0.7)))
                        .padding(.leading, 16)
                    }
                    
                    Spacer()
                    
                    Button(action: { isShowingCamera = false }) {
                        Image(systemName: "xmark.circle.fill")
                            .font(.title)
                            .foregroundColor(.white)
                            .background(Circle().fill(Color.black.opacity(0.5)))
                    }
                    .padding(.trailing, 16)
                }
                .padding(.top, 60)
                
                Spacer()
                
                HStack(spacing: 16) {
                    if camera.segmentedImage != nil {
                        Button(action: { saveFurniture() }) {
                            HStack(spacing: 8) {
                                Image(systemName: "square.and.arrow.down.fill").font(.title3)
                                Text("Save").font(.headline)
                            }
                            .foregroundColor(.white).padding(.horizontal, 24).padding(.vertical, 12)
                            .background(Capsule().fill(Color.green.opacity(0.9)))
                        }
                        
                        Button(action: {
                            camera.segmentedImage = nil
                            camera.furnitureOpacity = 0.0
                        }) {
                            HStack(spacing: 8) {
                                Image(systemName: "arrow.counterclockwise").font(.title3)
                                Text("Retry").font(.headline)
                            }
                            .foregroundColor(.white).padding(.horizontal, 24).padding(.vertical, 12)
                            .background(Capsule().fill(Color.orange.opacity(0.9)))
                        }
                    }
                    
                    Button(action: { isShowingCamera = false }) {
                        HStack(spacing: 8) {
                            Image(systemName: "xmark.circle.fill").font(.title3)
                            Text("Close").font(.headline)
                        }
                        .foregroundColor(.white).padding(.horizontal, 24).padding(.vertical, 12)
                        .background(Capsule().fill(Color.gray.opacity(0.9)))
                    }
                }
                .padding(.bottom, 50).padding(.horizontal)
            }
        }
        .onAppear {
            AVCaptureDevice.requestAccess(for: .video) { granted in
                if granted { camera.startSession() }
            }
        }
        .onDisappear { camera.stopSession() }
    }
    
    private func saveFurniture() {
        guard let image = camera.segmentedImage else { return }
        capturedImage = image
        
        PHPhotoLibrary.requestAuthorization(for: .addOnly) { status in
            DispatchQueue.main.async {
                if status == .authorized || status == .limited {
                    PHPhotoLibrary.shared().performChanges({
                        PHAssetChangeRequest.creationRequestForAsset(from: image)
                    }) { success, _ in
                        DispatchQueue.main.async {
                            if success {
                                withAnimation { showingSaveSuccess = true }
                                UINotificationFeedbackGenerator().notificationOccurred(.success)
                                DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) {
                                    withAnimation { showingSaveSuccess = false }
                                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                                        isShowingCamera = false
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// MARK: - Segment Examine Model - FastSAM-X PRIMARY
final class SegmentExamineModel: NSObject, ObservableObject {
    @Published var segmentedImage: UIImage?
    @Published var furnitureOpacity: Double = 0.0
    @Published var processingProgress: Double = 0.0
    @Published var isProcessingStarted: Bool = false
    @Published var progressMessage: String = ""
    @Published var isReadyToCapture: Bool = false
    
    let session = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let videoQueue = DispatchQueue(label: "segmentExamineQueue", qos: .userInitiated)
    
    // MODEL HIERARCHY: FastSAM-X (primary) + U2-Net (support)
    private var fastsamModel: VNCoreMLModel?
    private var u2netModel: VNCoreMLModel?
    
    private let context = CIContext()
    
    private var lastProcessTime = Date()
    private let processInterval: TimeInterval = 0.2
    private var isProcessing = false
    private var isFirstFrame = true
    
    // Detection buffers
    private var fastsamMask: CVPixelBuffer?
    private var u2netMask: CVPixelBuffer?
    private var cannyEdges: CVPixelBuffer?
    
    // FastSAM temporal smoothing
    private var previousFastsamMasks: [CVPixelBuffer] = []
    private let temporalWindowSize = 3
    
    override init() {
        super.init()
        loadFastSAMModel()
        loadU2NetModel()
        setupCamera()
    }
    
    // LOAD FASTSAM-X MODEL (PRIMARY)
    private func loadFastSAMModel() {
        print("🔍 Searching for FastSAM-X model...")
        let modelNames = ["FastSAM-x", "FastSAM-X", "fastsam-x", "FastSAMx", "FastSAM_x", "FastSAMX", "fastsam", "FastSAM"]
        for name in modelNames {
            for ext in ["mlmodelc", "mlpackage"] {
                if let modelURL = Bundle.main.url(forResource: name, withExtension: ext) {
                    print("📁 Found model file: \(name).\(ext)")
                    do {
                        let model = try MLModel(contentsOf: modelURL)
                        fastsamModel = try VNCoreMLModel(for: model)
                        print("✅ FastSAM-X loaded successfully: \(name) [PRIMARY]")
                        return
                    } catch {
                        print("❌ Failed to load \(name).\(ext): \(error.localizedDescription)")
                    }
                }
            }
        }
        print("❌ CRITICAL: No FastSAM-X model found in bundle! (will run support-only)")
    }
    
    // LOAD U2-NET MODEL (SUPPORT)
    private func loadU2NetModel() {
        let modelNames = ["u2netp", "U2Net", "u2net", "U2NetP", "U2NET"]
        for name in modelNames {
            for ext in ["mlmodelc", "mlpackage"] {
                if let modelURL = Bundle.main.url(forResource: name, withExtension: ext) {
                    do {
                        let model = try MLModel(contentsOf: modelURL)
                        u2netModel = try VNCoreMLModel(for: model)
                        print("✅ U2-Net loaded: \(name) [SUPPORT]")
                        return
                    } catch {
                        print("⚠️ Failed to load \(name).\(ext): \(error)")
                    }
                }
            }
        }
        print("⚠️ No U2-Net model loaded - will run FastSAM-X only")
    }
    
    private func setupCamera() {
        session.beginConfiguration()
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
                if let connection = videoOutput.connection(with: .video) {
                    connection.videoRotationAngle = 90
                    if connection.isVideoMirroringSupported { connection.isVideoMirrored = false }
                }
            }
            session.commitConfiguration()
            print("✅ Camera configured [FastSAM-X PRIMARY]")
        } catch {
            print("❌ Camera setup failed: \(error)")
        }
    }
    
    func startSession() {
        if !session.isRunning {
            DispatchQueue.global(qos: .background).async {
                self.session.startRunning()
                print("✅ Camera session started [FastSAM-X PRIMARY]")
            }
        }
    }
    
    func stopSession() {
        if session.isRunning {
            session.stopRunning()
            print("🛑 Camera session stopped")
        }
    }
    
    // MAIN PROCESSING PIPELINE - CONTINUOUS LIVE SEGMENTATION
    private func processFrame(pixelBuffer: CVPixelBuffer) {
        let now = Date()
        guard now.timeIntervalSince(lastProcessTime) >= processInterval else { return }
        guard !isProcessing else { return }
        
        isProcessing = true
        lastProcessTime = now
        
        if isFirstFrame {
            print("\n🎬 === STARTING SEGMENTATION PIPELINE ===")
            DispatchQueue.main.async {
                self.isProcessingStarted = true
                self.processingProgress = 0.0
                self.progressMessage = "Starting detection..."
            }
        }
        
        // STEP 1: FastSAM-X
        if isFirstFrame {
            DispatchQueue.main.async {
                self.processingProgress = 0.1
                self.progressMessage = "Detecting furniture..."
            }
        }
        if fastsamModel != nil { runFastSAMSync(pixelBuffer: pixelBuffer) }
        if isFirstFrame { DispatchQueue.main.async { self.processingProgress = 0.4 } }
        
        // STEP 2: U2-Net (support)
        if isFirstFrame { DispatchQueue.main.async { self.progressMessage = "Refining details..." } }
        if u2netModel != nil { runU2NetSync(pixelBuffer: pixelBuffer) }   // <— FIXED: method exists
        if isFirstFrame { DispatchQueue.main.async { self.processingProgress = 0.6 } }
        
        // STEP 3: Canny-like edges
        if isFirstFrame { DispatchQueue.main.async { self.progressMessage = "Detecting edges..." } }
        runCannyEdgeDetection(pixelBuffer: pixelBuffer)
        if isFirstFrame { DispatchQueue.main.async { self.processingProgress = 0.7 } }
        
        // STEP 4: Combine
        if isFirstFrame { DispatchQueue.main.async { self.progressMessage = "Combining results..." } }
        
        var finalMask: CVPixelBuffer? = nil
        if let fastsamMask = fastsamMask {
            let smoothedFastsam = applyTemporalSmoothing(to: fastsamMask)
            if let u2netMask = u2netMask, let cannyEdges = cannyEdges {
                finalMask = combineFastSAMWithU2NetAndCanny(fastsam: smoothedFastsam, u2net: u2netMask, canny: cannyEdges)
            } else if let u2netMask = u2netMask {
                finalMask = combineFastSAMWithU2Net(fastsam: smoothedFastsam, u2net: u2netMask)
            } else if let cannyEdges = cannyEdges {
                finalMask = combineFastSAMWithCanny(fastsam: smoothedFastsam, canny: cannyEdges)
            } else {
                finalMask = smoothedFastsam
            }
        } else if let u2netMask = u2netMask {
            finalMask = u2netMask
        }
        
        if isFirstFrame { DispatchQueue.main.async { self.processingProgress = 0.85 } }
        
        // STEP 5: Composite
        if isFirstFrame { DispatchQueue.main.async { self.progressMessage = "Finalizing..." } }
        if let mask = finalMask {
            let maskImage = CIImage(cvPixelBuffer: mask)
            let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
            applyMaskToImage(original: ciImage, mask: maskImage)
        }
        
        if isFirstFrame {
            DispatchQueue.main.async {
                self.processingProgress = 1.0
                self.progressMessage = "Complete!"
                DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
                    self.isProcessingStarted = false
                }
            }
            print("🎬 === SEGMENTATION COMPLETE ===\n")
            isFirstFrame = false
        }
        
        isProcessing = false
    }
    
    // MARK: - FastSAM glue

    private func runFastSAMSync(pixelBuffer: CVPixelBuffer) {
        guard let model = fastsamModel else {
            print("❌ FastSAM-X model is nil")
            return
        }
        
        let request = VNCoreMLRequest(model: model) { [weak self] request, error in
            guard let self = self else { return }
            if let error = error {
                print("❌ FastSAM-X error: \(error.localizedDescription)")
                return
            }
            guard let results = request.results as? [VNCoreMLFeatureValueObservation] else {
                print("⚠️ FastSAM-X results are unexpected type")
                return
            }
            var proto: MLMultiArray? = nil  // [1,32,160,160]
            var dets:  MLMultiArray? = nil  // [1,37,8400]
            for r in results {
                if r.featureName == "p" { proto = r.featureValue.multiArrayValue }
                else if r.featureName == "var_1550" { dets = r.featureValue.multiArrayValue }
            }
            guard let protoMA = proto, let detsMA = dets else {
                print("❌ FastSAM-X missing expected outputs (p / var_1550)")
                return
            }
            guard let best = self.pickBestDetection(from: detsMA, objThresh: 0.25) else {
                print("⚠️ No detection passed threshold")
                return
            }
            if let maskBuffer = self.buildMaskFromProto(protoMA, coeffs: best.coeffs, thresh: 0.5) {
                self.fastsamMask = maskBuffer
                let pixelCount = self.countWhitePixels(in: maskBuffer)
                print("🎯 FastSAM-X mask pixels: \(pixelCount)")
            } else {
                print("❌ Failed to build mask from prototypes")
            }
        }
        request.imageCropAndScaleOption = .scaleFill
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
        do { try handler.perform([request]) }
        catch { print("❌ FastSAM-X handler error: \(error.localizedDescription)") }
    }
    
    private struct FastSAMDetection {
        let obj: Float
        let coeffs: [Float]   // 32
    }
    
    private func pickBestDetection(from detsMA: MLMultiArray, objThresh: Float) -> FastSAMDetection? {
        guard detsMA.shape.count == 3,
              detsMA.shape[0].intValue == 1,
              detsMA.shape[1].intValue == 37 else {
            print("❌ Unexpected dets shape: \(detsMA.shape)")
            return nil
        }
        // let C = detsMA.shape[1].intValue  // 37  <-- removed (unused)
        let N = detsMA.shape[2].intValue       // 8400
        let ptr = detsMA.dataPointer.assumingMemoryBound(to: Float.self)
        let strideC = N                         // [1, C, N]
        
        var bestObj: Float = -1
        var bestCoeffs = [Float](repeating: 0, count: 32)
        
        for i in 0..<N {
            let rawObj = ptr[4 * strideC + i]
            let obj = sigmoid(rawObj)
            if obj < objThresh { continue }
            if obj > bestObj {
                bestObj = obj
                for k in 0..<32 {
                    bestCoeffs[k] = ptr[(5 + k) * strideC + i]
                }
            }
        }
        guard bestObj >= 0 else { return nil }
        return FastSAMDetection(obj: bestObj, coeffs: bestCoeffs)
    }
    
    private func buildMaskFromProto(_ protoMA: MLMultiArray, coeffs: [Float], thresh: Float) -> CVPixelBuffer? {
        guard protoMA.shape.count == 4,
              protoMA.shape[0].intValue == 1,
              protoMA.shape[1].intValue == 32 else {
            print("❌ Unexpected proto shape: \(protoMA.shape)")
            return nil
        }
        let K = 32
        let H = protoMA.shape[2].intValue
        let W = protoMA.shape[3].intValue
        let protoPtr = protoMA.dataPointer.assumingMemoryBound(to: Float.self)
        
        var out: CVPixelBuffer?
        let attrs: CFDictionary = [
            kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue!,
            kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue!,
            kCVPixelBufferPixelFormatTypeKey: kCVPixelFormatType_OneComponent8
        ] as CFDictionary
        
        guard CVPixelBufferCreate(kCFAllocatorDefault, W, H, kCVPixelFormatType_OneComponent8, attrs, &out) == kCVReturnSuccess,
              let outPB = out else { return nil }
        
        CVPixelBufferLockBaseAddress(outPB, [])
        defer { CVPixelBufferUnlockBaseAddress(outPB, []) }
        guard let dst = CVPixelBufferGetBaseAddress(outPB)?.assumingMemoryBound(to: UInt8.self) else { return nil }
        let dstStride = CVPixelBufferGetBytesPerRow(outPB)
        
        @inline(__always) func protoAt(_ c: Int, _ y: Int, _ x: Int) -> Float {
            return protoPtr[((c * H + y) * W) + x] // [1,K,H,W]
        }
        
        for y in 0..<H {
            let rowBase = y * dstStride
            for x in 0..<W {
                var v: Float = 0
                for c in 0..<K { v += coeffs[c] * protoAt(c, y, x) }
                let s = sigmoid(v)
                dst[rowBase + x] = (s >= thresh) ? 255 : 0
            }
        }
        return outPB
    }
    
    @inline(__always) private func sigmoid(_ x: Float) -> Float {
        1.0 / (1.0 + exp(-x))
    }
    
    // MARK: - U2-Net (SUPPORT)  ———  ADDED BACK
    private func runU2NetSync(pixelBuffer: CVPixelBuffer) {
        guard let model = u2netModel else { return }
        let request = VNCoreMLRequest(model: model) { [weak self] request, error in
            if let error = error {
                print("❌ U2-Net error: \(error)")
                return
            }
            if let results = request.results as? [VNPixelBufferObservation],
               let maskBuffer = results.first?.pixelBuffer {
                self?.u2netMask = maskBuffer
                let pixelCount = self?.countWhitePixels(in: maskBuffer) ?? 0
                print("🔧 U2-Net support: \(pixelCount) pixels")
            }
        }
        request.imageCropAndScaleOption = .scaleFill
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
        do { try handler.perform([request]) }
        catch { print("❌ U2-Net handler error: \(error)") }
    }
    
    // MARK: - Temporal smoothing / Canny / Combining / Utilities
    private func applyTemporalSmoothing(to mask: CVPixelBuffer) -> CVPixelBuffer {
        guard let copied = copyPixelBuffer(mask) else { return mask }
        previousFastsamMasks.append(copied)
        if previousFastsamMasks.count > temporalWindowSize { previousFastsamMasks.removeFirst() }
        if previousFastsamMasks.count < 2 { return mask }
        return averageMasks(previousFastsamMasks) ?? mask
    }
    
    private func runCannyEdgeDetection(pixelBuffer: CVPixelBuffer) {
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        guard let grayFilter = CIFilter(name: "CIPhotoEffectMono"),
              let blurFilter = CIFilter(name: "CIGaussianBlur"),
              let edgeFilter = CIFilter(name: "CIEdges") else { return }
        grayFilter.setValue(ciImage, forKey: kCIInputImageKey)
        guard let grayImage = grayFilter.outputImage else { return }
        blurFilter.setValue(grayImage, forKey: kCIInputImageKey)
        blurFilter.setValue(1.0, forKey: kCIInputRadiusKey)
        guard let blurred = blurFilter.outputImage else { return }
        edgeFilter.setValue(blurred, forKey: kCIInputImageKey)
        edgeFilter.setValue(2.0, forKey: kCIInputIntensityKey)
        guard let edgeImage = edgeFilter.outputImage else { return }
        
        var edgeBuffer: CVPixelBuffer?
        let attrs = [
            kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue!,
            kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue!,
            kCVPixelBufferPixelFormatTypeKey: kCVPixelFormatType_OneComponent8
        ] as CFDictionary
        
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_OneComponent8, attrs, &edgeBuffer)
        
        if let buffer = edgeBuffer {
            context.render(edgeImage, to: buffer)
            cannyEdges = buffer
            let edgeCount = countWhitePixels(in: buffer)
            print("🔲 Canny edges: \(edgeCount) pixels")
        }
    }
    
    private func combineFastSAMWithCanny(fastsam: CVPixelBuffer, canny: CVPixelBuffer) -> CVPixelBuffer? {
        guard let result = copyPixelBuffer(fastsam) else { return fastsam }
        CVPixelBufferLockBaseAddress(result, [])
        CVPixelBufferLockBaseAddress(canny, .readOnly)
        defer {
            CVPixelBufferUnlockBaseAddress(result, [])
            CVPixelBufferUnlockBaseAddress(canny, .readOnly)
        }
        let width = CVPixelBufferGetWidth(result)
        let height = CVPixelBufferGetHeight(result)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(result)
        let cannyBytesPerRow = CVPixelBufferGetBytesPerRow(canny)
        guard let resultPtr = CVPixelBufferGetBaseAddress(result)?.assumingMemoryBound(to: UInt8.self),
              let cannyPtr = CVPixelBufferGetBaseAddress(canny)?.assumingMemoryBound(to: UInt8.self) else {
            return fastsam
        }
        var addedFromCanny = 0
        for y in 0..<height {
            for x in 0..<width {
                let ridx = y * bytesPerRow + x
                let cidx = y * cannyBytesPerRow + x
                if resultPtr[ridx] > 128 { continue }
                if cannyPtr[cidx] > 200 {
                    var near = false
                    let R = 30
                    for dy in -R...R where !near {
                        for dx in -R...R {
                            let ny = y + dy, nx = x + dx
                            if ny >= 0 && ny < height && nx >= 0 && nx < width {
                                if resultPtr[ny * bytesPerRow + nx] > 128 { near = true; break }
                            }
                        }
                    }
                    if near { resultPtr[ridx] = 255; addedFromCanny += 1 }
                }
            }
        }
        print("🔲 Added \(addedFromCanny) pixels from Canny edges")
        return morphologicalClosing(result, iterations: 3) ?? result
    }
    
    private func combineFastSAMWithU2NetAndCanny(fastsam: CVPixelBuffer, u2net: CVPixelBuffer, canny: CVPixelBuffer) -> CVPixelBuffer? {
        guard let merged = combineFastSAMWithU2Net(fastsam: fastsam, u2net: u2net) else { return fastsam }
        guard let result = copyPixelBuffer(merged) else { return merged }
        CVPixelBufferLockBaseAddress(result, [])
        CVPixelBufferLockBaseAddress(canny, .readOnly)
        defer {
            CVPixelBufferUnlockBaseAddress(result, [])
            CVPixelBufferUnlockBaseAddress(canny, .readOnly)
        }
        let width = CVPixelBufferGetWidth(result)
        let height = CVPixelBufferGetHeight(result)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(result)
        let cannyBytesPerRow = CVPixelBufferGetBytesPerRow(canny)
        guard let resultPtr = CVPixelBufferGetBaseAddress(result)?.assumingMemoryBound(to: UInt8.self),
              let cannyPtr = CVPixelBufferGetBaseAddress(canny)?.assumingMemoryBound(to: UInt8.self) else {
            return merged
        }
        var addedFromCanny = 0
        for y in 0..<height {
            for x in 0..<width {
                let ridx = y * bytesPerRow + x
                if resultPtr[ridx] > 128 { continue }
                let cidx = y * cannyBytesPerRow + x
                if cannyPtr[cidx] > 200 {
                    var near = false
                    let R = 25
                    for dy in -R...R where !near {
                        for dx in -R...R {
                            let ny = y + dy, nx = x + dx
                            if ny >= 0 && ny < height && nx >= 0 && nx < width {
                                if resultPtr[ny * bytesPerRow + nx] > 128 { near = true; break }
                            }
                        }
                    }
                    if near { resultPtr[ridx] = 255; addedFromCanny += 1 }
                }
            }
        }
        print("🔲➕ Final: Added \(addedFromCanny) pixels from Canny edges")
        return morphologicalClosing(result, iterations: 4) ?? result
    }
    
    private func combineFastSAMWithU2Net(fastsam: CVPixelBuffer, u2net: CVPixelBuffer) -> CVPixelBuffer? {
        guard let result = copyPixelBuffer(fastsam) else { return fastsam }
        CVPixelBufferLockBaseAddress(result, [])
        CVPixelBufferLockBaseAddress(u2net, .readOnly)
        defer {
            CVPixelBufferUnlockBaseAddress(result, [])
            CVPixelBufferUnlockBaseAddress(u2net, .readOnly)
        }
        let width = CVPixelBufferGetWidth(result)
        let height = CVPixelBufferGetHeight(result)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(result)
        let u2netBytesPerRow = CVPixelBufferGetBytesPerRow(u2net)
        guard let resultPtr = CVPixelBufferGetBaseAddress(result)?.assumingMemoryBound(to: UInt8.self),
              let u2netPtr = CVPixelBufferGetBaseAddress(u2net)?.assumingMemoryBound(to: UInt8.self) else {
            return fastsam
        }
        var fastsamPixelCount = 0
        var addedFromU2Net = 0
        for y in 0..<height {
            for x in 0..<width {
                if resultPtr[y * bytesPerRow + x] > 128 { fastsamPixelCount += 1 }
            }
        }
        for y in 0..<height {
            for x in 0..<width {
                let ridx = y * bytesPerRow + x
                if resultPtr[ridx] > 128 { continue }
                let uidx = y * u2netBytesPerRow + x
                let val = u2netPtr[uidx]
                if val > 200 {
                    var near = false
                    let R = 50
                    for dy in -R...R where !near {
                        for dx in -R...R {
                            let ny = y + dy, nx = x + dx
                            if ny >= 0 && ny < height && nx >= 0 && nx < width {
                                if resultPtr[ny * bytesPerRow + nx] > 128 { near = true; break }
                            }
                        }
                    }
                    if near { resultPtr[ridx] = 255; addedFromU2Net += 1 }
                }
            }
        }
        print("🔀 Combined: FastSAM=\(fastsamPixelCount)px, U2-Net added=\(addedFromU2Net)px")
        return morphologicalClosing(result, iterations: 5) ?? result
    }
    
    private func morphologicalClosing(_ buffer: CVPixelBuffer, iterations: Int) -> CVPixelBuffer? {
        guard var current = copyPixelBuffer(buffer) else { return buffer }
        for _ in 0..<iterations { guard let d = dilate(current) else { break }; current = d }
        for _ in 0..<(iterations / 2) { guard let e = erode(current) else { break }; current = e }
        return current
    }
    
    private func dilate(_ buffer: CVPixelBuffer) -> CVPixelBuffer? {
        guard let output = copyPixelBuffer(buffer) else { return nil }
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        CVPixelBufferLockBaseAddress(output, [])
        defer {
            CVPixelBufferUnlockBaseAddress(buffer, .readOnly)
            CVPixelBufferUnlockBaseAddress(output, [])
        }
        let width = CVPixelBufferGetWidth(buffer)
        let height = CVPixelBufferGetHeight(buffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
        guard let inputPtr = CVPixelBufferGetBaseAddress(buffer)?.assumingMemoryBound(to: UInt8.self),
              let outputPtr = CVPixelBufferGetBaseAddress(output)?.assumingMemoryBound(to: UInt8.self) else {
            return nil
        }
        for y in 1..<(height - 1) {
            for x in 1..<(width - 1) {
                var maxVal: UInt8 = 0
                for dy in -1...1 {
                    for dx in -1...1 {
                        let idx = (y + dy) * bytesPerRow + (x + dx)
                        maxVal = max(maxVal, inputPtr[idx])
                    }
                }
                outputPtr[y * bytesPerRow + x] = maxVal
            }
        }
        return output
    }
    
    private func erode(_ buffer: CVPixelBuffer) -> CVPixelBuffer? {
        guard let output = copyPixelBuffer(buffer) else { return nil }
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        CVPixelBufferLockBaseAddress(output, [])
        defer {
            CVPixelBufferUnlockBaseAddress(buffer, .readOnly)
            CVPixelBufferUnlockBaseAddress(output, [])
        }
        let width = CVPixelBufferGetWidth(buffer)
        let height = CVPixelBufferGetHeight(buffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
        guard let inputPtr = CVPixelBufferGetBaseAddress(buffer)?.assumingMemoryBound(to: UInt8.self),
              let outputPtr = CVPixelBufferGetBaseAddress(output)?.assumingMemoryBound(to: UInt8.self) else {
            return nil
        }
        for y in 1..<(height - 1) {
            for x in 1..<(width - 1) {
                var minVal: UInt8 = 255
                for dy in -1...1 {
                    for dx in -1...1 {
                        let idx = (y + dy) * bytesPerRow + (x + dx)
                        minVal = min(minVal, inputPtr[idx])
                    }
                }
                outputPtr[y * bytesPerRow + x] = minVal
            }
        }
        return output
    }
    
    private func averageMasks(_ masks: [CVPixelBuffer]) -> CVPixelBuffer? {
        guard !masks.isEmpty, let first = masks.first else { return nil }
        let width = CVPixelBufferGetWidth(first)
        let height = CVPixelBufferGetHeight(first)
        var result: CVPixelBuffer?
        let attrs = [
            kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue!,
            kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue!,
            kCVPixelBufferPixelFormatTypeKey: kCVPixelFormatType_OneComponent8
        ] as CFDictionary
        CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_OneComponent8, attrs, &result)
        guard let averaged = result else { return nil }
        CVPixelBufferLockBaseAddress(averaged, [])
        guard let resultPtr = CVPixelBufferGetBaseAddress(averaged)?.assumingMemoryBound(to: UInt8.self) else {
            CVPixelBufferUnlockBaseAddress(averaged, []); return nil
        }
        let bytesPerRow = CVPixelBufferGetBytesPerRow(averaged)
        defer { CVPixelBufferUnlockBaseAddress(averaged, []) }
        
        var sums = [Int](repeating: 0, count: height * width)
        for mask in masks {
            CVPixelBufferLockBaseAddress(mask, .readOnly)
            if let mptr = CVPixelBufferGetBaseAddress(mask)?.assumingMemoryBound(to: UInt8.self) {
                let mstride = CVPixelBufferGetBytesPerRow(mask)
                for y in 0..<height {
                    for x in 0..<width {
                        sums[y * width + x] += Int(mptr[y * mstride + x])
                    }
                }
            }
            CVPixelBufferUnlockBaseAddress(mask, .readOnly)
        }
        let count = masks.count
        for y in 0..<height {
            for x in 0..<width {
                let avg = sums[y * width + x] / count
                resultPtr[y * bytesPerRow + x] = avg > 128 ? 255 : 0
            }
        }
        return averaged
    }
    
    private func countWhitePixels(in buffer: CVPixelBuffer) -> Int {
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        let width = CVPixelBufferGetWidth(buffer)
        let height = CVPixelBufferGetHeight(buffer)
        let stride = CVPixelBufferGetBytesPerRow(buffer)
        guard let base = CVPixelBufferGetBaseAddress(buffer) else { return 0 }
        let ptr = base.assumingMemoryBound(to: UInt8.self)
        var count = 0
        for y in 0..<height {
            for x in 0..<width where ptr[y * stride + x] > 128 { count += 1 }
        }
        return count
    }
    
    private func applyMaskToImage(original: CIImage, mask: CIImage) {
        let scaleX = original.extent.width / mask.extent.width
        let scaleY = original.extent.height / mask.extent.height
        let scaledMask = mask
            .transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))
            .samplingNearest() // Assuming you already have this CIImage extension
        var finalMask = scaledMask
        if let blurFilter = CIFilter(name: "CIGaussianBlur") {
            blurFilter.setValue(scaledMask, forKey: kCIInputImageKey)
            blurFilter.setValue(0.3, forKey: kCIInputRadiusKey)
            if let blurred = blurFilter.outputImage { finalMask = blurred }
        }
        if let colorControls = CIFilter(name: "CIColorControls") {
            colorControls.setValue(finalMask, forKey: kCIInputImageKey)
            colorControls.setValue(2.0, forKey: kCIInputContrastKey)
            if let sharpened = colorControls.outputImage { finalMask = sharpened }
        }
        finalMask = finalMask.cropped(to: original.extent)
        
        guard let blend = CIFilter(name: "CIBlendWithMask") else { return }
        let transparent = CIImage(color: CIColor(red: 0, green: 0, blue: 0, alpha: 0)).cropped(to: original.extent)
        blend.setValue(original, forKey: kCIInputImageKey)
        blend.setValue(transparent, forKey: kCIInputBackgroundImageKey)
        blend.setValue(finalMask, forKey: kCIInputMaskImageKey)
        guard let result = blend.outputImage else { return }
        
        let renderContext = CIContext(options: [
            .workingColorSpace: CGColorSpaceCreateDeviceRGB(),
            .outputPremultiplied: true,
            .useSoftwareRenderer: false
        ])
        if let cgImage = renderContext.createCGImage(result, from: result.extent) {
            let uiImage = UIImage(cgImage: cgImage, scale: 1.0, orientation: .up)
            DispatchQueue.main.async {
                self.segmentedImage = uiImage
                withAnimation(.easeIn(duration: 0.2)) { self.furnitureOpacity = 1.0 }
            }
        }
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
        guard CVPixelBufferCreate(kCFAllocatorDefault, width, height, format, attrs, &copy) == kCVReturnSuccess,
              let dest = copy else { return nil }
        CVPixelBufferLockBaseAddress(source, .readOnly)
        CVPixelBufferLockBaseAddress(dest, [])
        defer {
            CVPixelBufferUnlockBaseAddress(source, .readOnly)
            CVPixelBufferUnlockBaseAddress(dest, [])
        }
        let sStride = CVPixelBufferGetBytesPerRow(source)
        let dStride = CVPixelBufferGetBytesPerRow(dest)
        guard let sData = CVPixelBufferGetBaseAddress(source),
              let dData = CVPixelBufferGetBaseAddress(dest) else { return nil }
        for row in 0..<height {
            memcpy(dData.advanced(by: row * dStride), sData.advanced(by: row * sStride), min(sStride, dStride))
        }
        return dest
    }
}

extension SegmentExamineModel: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        processFrame(pixelBuffer: pixelBuffer)
    }
}
