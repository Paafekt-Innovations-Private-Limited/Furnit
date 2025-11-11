import SwiftUI
import AVFoundation
import CoreML
import Vision
import CoreImage
import Photos

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
                            Text("Found: \(camera.detectedFurnitureTypes.joined(separator: ", "))")
                                .font(.caption)
                                .foregroundColor(.white.opacity(0.9))
                        }
                    }
                    .padding(.vertical, 20)
                    .padding(.horizontal, 30)
                    .background(
                        RoundedRectangle(cornerRadius: 16)
                            .fill(Color.black.opacity(0.85))
                    )
                    .padding(.bottom, 200)
                }
            }
            
            if showingSaveSuccess {
                VStack {
                    Spacer()
                    HStack(spacing: 12) {
                        Image(systemName: "checkmark.circle.fill")
                            .font(.title2)
                        Text("Furniture saved!")
                            .font(.headline)
                    }
                    .foregroundColor(.white)
                    .padding(.horizontal, 24)
                    .padding(.vertical, 16)
                    .background(
                        Capsule()
                            .fill(Color.green.opacity(0.95))
                            .shadow(radius: 10)
                    )
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
                                Image(systemName: "square.and.arrow.down.fill")
                                    .font(.title3)
                                Text("Save")
                                    .font(.headline)
                            }
                            .foregroundColor(.white)
                            .padding(.horizontal, 24)
                            .padding(.vertical, 12)
                            .background(Capsule().fill(Color.green.opacity(0.9)))
                        }
                        
                        Button(action: {
                            camera.segmentedImage = nil
                            camera.furnitureOpacity = 0.0
                        }) {
                            HStack(spacing: 8) {
                                Image(systemName: "arrow.counterclockwise")
                                    .font(.title3)
                                Text("Retry")
                                    .font(.headline)
                            }
                            .foregroundColor(.white)
                            .padding(.horizontal, 24)
                            .padding(.vertical, 12)
                            .background(Capsule().fill(Color.orange.opacity(0.9)))
                        }
                    }
                    
                    Button(action: { isShowingCamera = false }) {
                        HStack(spacing: 8) {
                            Image(systemName: "xmark.circle.fill")
                                .font(.title3)
                            Text("Cancel")
                                .font(.headline)
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

// MARK: - YOLO11-Seg Model
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
    
    private let furnitureClasses = [
        56: "chair",
        57: "couch",
        59: "bed",
        60: "dining table",
        61: "toilet"
    ]
    
    private var lastProcessTime = Date()
    private let processInterval: TimeInterval = 0.5
    
    override init() {
        super.init()
        loadYOLOModel()
        setupCamera()
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
            
            session.commitConfiguration()
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
                    print("✅ Camera started")
                }
            }
        }
    }
    
    func stopSession() {
        if session.isRunning {
            session.stopRunning()
            print("🛑 Camera stopped")
        }
    }
    
    private func processWithYOLO(pixelBuffer: CVPixelBuffer) {
        guard let model = yoloModel else { return }
        
        let now = Date()
        guard now.timeIntervalSince(lastProcessTime) >= processInterval else { return }
        guard !isProcessing else { return }
        
        lastProcessTime = now
        isProcessing = true
        
        let request = VNCoreMLRequest(model: model) { [weak self] request, error in
            if let error = error {
                print("❌ YOLO error: \(error)")
                DispatchQueue.main.async {
                    self?.isProcessing = false
                }
                return
            }
            
            self?.processYOLOResults(request.results, originalImage: pixelBuffer)
        }
        
        request.imageCropAndScaleOption = .scaleFill
        
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
        
        do {
            try handler.perform([request])
        } catch {
            print("❌ Failed to perform YOLO inference: \(error)")
            DispatchQueue.main.async {
                self.isProcessing = false
            }
        }
    }
    
    private func processYOLOResults(_ results: [Any]?, originalImage: CVPixelBuffer) {
        guard let observations = results as? [VNCoreMLFeatureValueObservation] else {
            DispatchQueue.main.async {
                self.isProcessing = false
            }
            return
        }
        
        var detectionOutput: MLMultiArray?
        var prototypeOutput: MLMultiArray?
        
        for observation in observations {
            if let multiArray = observation.featureValue.multiArrayValue {
                let shape = multiArray.shape
                
                if shape.count == 3 && shape[1].intValue == 116 && shape[2].intValue == 8400 {
                    detectionOutput = multiArray
                } else if shape.count == 4 && shape[1].intValue == 32 {
                    prototypeOutput = multiArray
                }
            }
        }
        
        guard let detections = detectionOutput,
              let prototypes = prototypeOutput else {
            DispatchQueue.main.async {
                self.isProcessing = false
            }
            return
        }
        
        processFurnitureDetections(detections: detections,
                                 prototypes: prototypes,
                                 originalImage: originalImage)
    }
    
    private func processFurnitureDetections(detections: MLMultiArray,
                                           prototypes: MLMultiArray,
                                           originalImage: CVPixelBuffer) {
        var detectedTypes: Set<String> = []
        
        let numAnchors = 8400
        let confThreshold: Float = 0.1  // Lower threshold
        
        var foundAny = false
        
        // Check for furniture
        for anchor in 0..<numAnchors {
            for classIdx in furnitureClasses.keys {
                let conf = detections[[0, 4 + classIdx, anchor] as [NSNumber]].floatValue
                
                if conf > confThreshold {
                    if let className = furnitureClasses[classIdx] {
                        detectedTypes.insert(className)
                        foundAny = true
                        print("🪑 Found \(className) with confidence \(conf)")
                        break  // Found furniture in this anchor
                    }
                }
            }
            if foundAny { break }  // Stop after finding first furniture
        }
        
        if !foundAny {
            DispatchQueue.main.async {
                print("⚠️ No furniture detected")
                self.isProcessing = false
                self.segmentedImage = nil
                self.furnitureOpacity = 0.0
            }
            return
        }
        
        // Create red tinted version for testing
        let originalCIImage = CIImage(cvPixelBuffer: originalImage)
        
        if let colorFilter = CIFilter(name: "CIColorMonochrome") {
            colorFilter.setValue(originalCIImage, forKey: kCIInputImageKey)
            colorFilter.setValue(CIColor(red: 1, green: 0, blue: 0), forKey: "inputColor")
            colorFilter.setValue(0.3, forKey: "inputIntensity")
            
            if let output = colorFilter.outputImage,
               let cgImage = context.createCGImage(output, from: output.extent) {
                let uiImage = UIImage(cgImage: cgImage, scale: 1.0, orientation: .up)
                
                DispatchQueue.main.async {
                    self.segmentedImage = uiImage
                    self.detectedFurnitureTypes = Array(detectedTypes)
                    withAnimation(.easeIn(duration: 0.3)) {
                        self.furnitureOpacity = 1.0
                    }
                    self.isProcessing = false
                }
            }
        }
    }
}

extension FurnitureSegmentationModel: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        processWithYOLO(pixelBuffer: pixelBuffer)
    }
}
