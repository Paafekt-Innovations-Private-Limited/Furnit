import SwiftUI
import AVFoundation
import CoreML
import Vision
import CoreImage
import Photos

// MARK: - SegmentFurniture View
struct SegmentFurniture: View {
    @Binding var capturedImage: UIImage?
    @Binding var isShowingCamera: Bool
    @StateObject private var camera = FurnitureSegmentationModel()

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

            if camera.isProcessing {
                VStack {
                    Spacer()
                    VStack(spacing: 12) {
                        ProgressView()
                            .progressViewStyle(CircularProgressViewStyle(tint: .white))
                            .scaleEffect(1.5)

                        Text("Detecting furniture...")
                            .font(.headline)
                            .foregroundColor(.white)

                        if camera.detectedFurnitureTypes.count > 0 {
                            let found = camera.detectedFurnitureTypes.joined(separator: ", ")
                            Text(verbatim: "Found: " + found)
                                .font(.caption)
                                .foregroundColor(.white.opacity(0.9))
                        }
                    }
                    .padding(.vertical, 20)
                    .padding(.horizontal, 30)
                    .background(RoundedRectangle(cornerRadius: 16).fill(Color.black.opacity(0.85)))
                    .padding(.bottom, 200)
                }
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
                            .foregroundColor(.white)
                            .padding(.horizontal, 24)
                            .padding(.vertical, 12)
                            .background(Capsule().fill(Color.green.opacity(0.9)))
                        }

                        Button(action: { camera.resetSegmentation() }) {
                            HStack(spacing: 8) {
                                Image(systemName: "arrow.counterclockwise").font(.title3)
                                Text("Retry").font(.headline)
                            }
                            .foregroundColor(.white)
                            .padding(.horizontal, 24)
                            .padding(.vertical, 12)
                            .background(Capsule().fill(Color.orange.opacity(0.9)))
                        }
                    }

                    Button(action: { isShowingCamera = false }) {
                        HStack(spacing: 8) {
                            Image(systemName: "xmark.circle.fill").font(.title3)
                            Text("Cancel").font(.headline)
                        }
                        .foregroundColor(.white)
                        .padding(.horizontal, 24)
                        .padding(.vertical, 12)
                        .background(Capsule().fill(Color.gray.opacity(0.9)))
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
                }
            }
        }
        .onDisappear {
            camera.stopSession()
        }
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

// MARK: - YOLO11-Seg Model (Main-actor safe, AVCapture, 116/117 channel sniff)
@MainActor
class FurnitureSegmentationModel: NSObject, ObservableObject {
    @Published var segmentedImage: UIImage?
    @Published var furnitureOpacity: Double = 0.0
    @Published var isProcessing = false
    @Published var detectedFurnitureTypes: [String] = []

    private let session = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let videoQueue = DispatchQueue(label: "furnitureSegQueue", qos: .userInitiated)

    private var yoloModel: VNCoreMLModel?
    private let context = CIContext()

    // COCO ids for furniture-like classes
    private let furnitureClasses: [Int: String] = [
        56: "chair",
        57: "couch",
        59: "bed",
        60: "dining table",
        61: "toilet"
    ]

    private var lastProcessTime = Date()
    private let processInterval: TimeInterval = 0.5
    private var didLogOnce = false

    override init() {
        super.init()
        loadYOLOModel()
        setupCamera()
    }

    func resetSegmentation() {
        segmentedImage = nil
        furnitureOpacity = 0.0
    }

    private func loadYOLOModel() {
        print("🔍 Loading YOLO11-seg model...")
        for ext in ["mlmodelc", "mlpackage"] {
            if let modelURL = Bundle.main.url(forResource: "yolo11x-seg", withExtension: ext) {
                print("📦 Found model: yolo11x-seg.\(ext)")
                do {
                    let model = try MLModel(contentsOf: modelURL)
                    yoloModel = try VNCoreMLModel(for: model)
                    print("✅ YOLO11-seg loaded successfully!")
                    return
                } catch {
                    print("⚠️ Failed to load yolo11x-seg.\(ext): \(error)")
                }
            }
        }
        print("❌ No YOLO11-seg model found!")
    }

    private func setupCamera() {
        session.beginConfiguration()
        session.sessionPreset = .hd1280x720

        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else {
            print("❌ No camera available")
            session.commitConfiguration()
            return
        }

        do {
            let input = try AVCaptureDeviceInput(device: device)
            if session.canAddInput(input) { session.addInput(input) }

            videoOutput.setSampleBufferDelegate(self, queue: videoQueue)
            videoOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
            videoOutput.alwaysDiscardsLateVideoFrames = true

            if session.canAddOutput(videoOutput) { session.addOutput(videoOutput) }

            // Do NOT rotate the connection here; handle orientation downstream if needed
            session.commitConfiguration()
            print("✅ Camera configured")
        } catch {
            print("❌ Camera setup failed: \(error)")
            session.commitConfiguration()
        }
    }

    func startSession() {
        guard !session.isRunning else { return }
        DispatchQueue.global(qos: .background).async {
            self.session.startRunning()
            DispatchQueue.main.async { print("✅ Camera started") }
        }
    }

    func stopSession() {
        if session.isRunning { session.stopRunning() }
        print("🛑 Camera stopped")
    }

    // MARK: - Inference pipeline
    private func processWithYOLO(pixelBuffer: CVPixelBuffer) {
        guard let model = yoloModel else { return }

        let now = Date()
        guard now.timeIntervalSince(lastProcessTime) >= processInterval else { return }
        guard !isProcessing else { return }

        lastProcessTime = now
        isProcessing = true

        let request = VNCoreMLRequest(model: model) { [weak self] request, error in
            guard let self = self else { return }
            if let error = error {
                print("❌ YOLO error: \(error)")
                self.isProcessing = false
                return
            }
            self.processYOLOResults(request.results, originalImage: pixelBuffer)
        }
        request.imageCropAndScaleOption = .scaleFill

        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
        do {
            try handler.perform([request])
        } catch {
            print("❌ Failed to perform YOLO inference: \(error)")
            isProcessing = false
        }
    }

    private func processYOLOResults(_ results: [Any]?, originalImage: CVPixelBuffer) {
        guard let observations = results as? [VNCoreMLFeatureValueObservation] else {
            isProcessing = false
            return
        }

        var detectionOutput: MLMultiArray?
        var prototypeOutput: MLMultiArray?

        for observation in observations {
            if let multiArray = observation.featureValue.multiArrayValue {
                let shape = multiArray.shape.map { $0.intValue }
                // Typical shapes: detections [1, C, 8400]; prototypes [1, 32, Hm, Wm]
                if shape.count == 3 && shape[2] == 8400 {
                    detectionOutput = multiArray
                } else if shape.count == 4 && shape[1] == 32 {
                    prototypeOutput = multiArray
                }
            }
        }

        if let det = detectionOutput, !didLogOnce {
            didLogOnce = true
            print("Detections shape: \(det.shape.map{ $0.intValue })")
            let C = det.shape[1].intValue
            let A = det.shape[2].intValue
            if A > 0 {
                var first12: [Float] = []
                for c in 0..<min(12, C) {
                    first12.append(det[[0, NSNumber(value: c), 0]].floatValue)
                }
                print("Anchor0 first12: \(first12)")
            }
        }

        guard let detections = detectionOutput else {
            isProcessing = false
            return
        }

        detectFurniture(detections: detections, prototypes: prototypeOutput, pixelBuffer: originalImage)
    }

    @inline(__always) private func sigmoid(_ x: Float) -> Float { 1 / (1 + exp(-x)) }

    /// Works for both C=116 ([x,y,w,h,80cls,32coeff]) and C=117 ([x,y,w,h,obj,80cls,32coeff])
    private func detectFurniture(detections: MLMultiArray,
                                 prototypes: MLMultiArray?,
                                 pixelBuffer: CVPixelBuffer) {

        let C = detections.shape[1].intValue
        let A = detections.shape[2].intValue

        let hasObjness = (C >= 117)
        let clsBase = hasObjness ? 5 : 4
        let coeffBase = clsBase + 80
        let nm = max(0, C - coeffBase) // number of mask coeffs if needed later

        let confThreshold: Float = 0.25

        var detectedTypes: Set<String> = []
        var foundAny = false

        print("YOLO layout: C=\(C), hasObjness=\(hasObjness), clsBase=\(clsBase), coeffBase=\(coeffBase), nm=\(nm), anchors=\(A)")

        for anchor in 0..<A {
            let obj = hasObjness ? sigmoid(detections[[0, 4, anchor] as [NSNumber]].floatValue) : 1.0

            for classIdx in furnitureClasses.keys.sorted() {
                let raw = detections[[0, clsBase + classIdx, anchor] as [NSNumber]].floatValue
                let clsScore = sigmoid(raw)      // safe even if already squashed
                let conf = obj * clsScore
                if conf > confThreshold, let name = furnitureClasses[classIdx] {
                    detectedTypes.insert(name)
                    print("🪑 Found \(name) conf=\(conf) @anchor \(anchor)")
                    foundAny = true
                    break
                }
            }
            if foundAny { break }
        }

        guard foundAny else {
            print("⚠️ No furniture detected (C=\(C), hasObjness=\(hasObjness))")
            self.segmentedImage = nil
            self.furnitureOpacity = 0.0
            self.detectedFurnitureTypes = []
            self.isProcessing = false
            return
        }

        // Simple visual proof: red-tinted original image
        let originalCIImage = CIImage(cvPixelBuffer: pixelBuffer)
        if let colorFilter = CIFilter(name: "CIColorMonochrome") {
            colorFilter.setValue(originalCIImage, forKey: kCIInputImageKey)
            colorFilter.setValue(CIColor(red: 1, green: 0, blue: 0), forKey: "inputColor")
            colorFilter.setValue(0.3, forKey: "inputIntensity")

            if let output = colorFilter.outputImage,
               let cgImage = context.createCGImage(output, from: output.extent) {
                let uiImage = UIImage(cgImage: cgImage, scale: 1.0, orientation: .up)
                self.segmentedImage = uiImage
                self.detectedFurnitureTypes = Array(detectedTypes)
                withAnimation(.easeIn(duration: 0.3)) { self.furnitureOpacity = 1.0 }
            }
        }

        self.isProcessing = false
    }
}

// MARK: - AVCapture delegate
extension FurnitureSegmentationModel: AVCaptureVideoDataOutputSampleBufferDelegate {
    nonisolated func captureOutput(_ output: AVCaptureOutput,
                                   didOutput sampleBuffer: CMSampleBuffer,
                                   from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        Task { @MainActor in
            (self as FurnitureSegmentationModel).processWithYOLO(pixelBuffer: pixelBuffer)
        }
    }
}
