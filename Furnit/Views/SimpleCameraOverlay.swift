import SwiftUI
import AVFoundation
import CoreML
import Vision
import CoreImage

struct SimpleCameraOverlay: View {
    @Binding var capturedImage: UIImage?
    @Binding var isShowingCamera: Bool
    @StateObject private var camera = U2NetCameraModel()
    
    // Scaling control for furniture size
    @State private var furnitureScale: CGFloat = 0.7  // 70% of original size
    
    var body: some View {
        ZStack {
            // TRANSPARENT BACKGROUND - No camera preview shown!
            Color.clear
                .ignoresSafeArea()
            
            // ONLY show the segmented furniture without any frame constraints
            if let segmented = camera.segmentedImage {
                Image(uiImage: segmented)
                    .resizable()
                    .scaleEffect(furnitureScale)
                    .position(x: UIScreen.main.bounds.width / 2,
                             y: UIScreen.main.bounds.height / 2)
                    .allowsHitTesting(false)
            }
            
            // UI controls overlay
            VStack {
                HStack {
                    // Size adjustment slider
                    HStack {
                        Image(systemName: "minus.magnifyingglass")
                            .foregroundColor(.white)
                        
                        Slider(value: $furnitureScale, in: 0.3...1.2)
                            .frame(width: 120)
                            .accentColor(.white)
                        
                        Image(systemName: "plus.magnifyingglass")
                            .foregroundColor(.white)
                    }
                    .padding(8)
                    .background(Capsule().fill(Color.black.opacity(0.5)))
                    .padding(.leading)
                    
                    Spacer()
                    
                    // Close button
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
                .padding(.top, 50)  // Account for status bar
                
                Spacer()
                
                HStack {
                    // Status indicator
                    if camera.segmentedImage == nil {
                        Text("Point at furniture")
                            .font(.caption)
                            .padding(8)
                            .background(Capsule().fill(Color.black.opacity(0.5)))
                            .foregroundColor(.white)
                    } else {
                        Text("Size: \(Int(furnitureScale * 100))%")
                            .font(.caption)
                            .padding(8)
                            .background(Capsule().fill(Color.black.opacity(0.5)))
                            .foregroundColor(.white)
                    }
                    
                    Spacer()
                    
                    // Capture button
                    Button(action: {
                        if let currentImage = camera.segmentedImage {
                            capturedImage = currentImage
                            print("📸 Captured segmented furniture at scale: \(furnitureScale)")
                            isShowingCamera = false
                        }
                    }) {
                        Image(systemName: "camera.circle.fill")
                            .font(.system(size: 60))
                            .foregroundColor(camera.segmentedImage != nil ? .white : .gray)
                            .background(Circle().fill(Color.black.opacity(0.3)))
                            .shadow(radius: 3)
                    }
                    .disabled(camera.segmentedImage == nil)
                }
                .padding(.bottom, 40)
                .padding(.horizontal)
            }
        }
        .onAppear {
            AVCaptureDevice.requestAccess(for: .video) { granted in
                if granted {
                    camera.startSession()
                    print("📷 Camera session started for segmentation")
                } else {
                    print("⚠️ Camera access denied")
                }
            }
        }
        .onDisappear {
            camera.stopSession()
        }
    }
}

// Camera Preview Layer (not used in main view but kept for reference)
struct CameraPreviewLayer: UIViewRepresentable {
    let session: AVCaptureSession
    let videoGravity: AVLayerVideoGravity
    
    func makeUIView(context: Context) -> UIView {
        let view = UIView()
        view.backgroundColor = .clear
        let previewLayer = AVCaptureVideoPreviewLayer(session: session)
        previewLayer.videoGravity = videoGravity
        previewLayer.frame = view.bounds
        view.layer.addSublayer(previewLayer)
        context.coordinator.previewLayer = previewLayer
        return view
    }
    
    func updateUIView(_ uiView: UIView, context: Context) {
        context.coordinator.previewLayer?.frame = uiView.bounds
    }
    
    func makeCoordinator() -> Coordinator {
        Coordinator()
    }
    
    class Coordinator {
        var previewLayer: AVCaptureVideoPreviewLayer?
    }
}

// U2NetCameraModel with proper segmentation
class U2NetCameraModel: NSObject, ObservableObject {
    @Published var segmentedImage: UIImage?
    let session = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let videoQueue = DispatchQueue(label: "videoQueue", qos: .userInitiated)
    
    private var u2netModel: VNCoreMLModel?
    private let context = CIContext()
    
    // Throttling
    private var lastProcessTime = Date()
    private let processInterval: TimeInterval = 0.2 // Process every 200ms
    
    override init() {
        super.init()
        checkCameraAuthorization()
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
            print("⚠️ Camera access denied")
        }
    }
    
    private func loadU2NetModel() {
        // Try multiple possible model names
        let modelNames = ["u2netp", "U2Net", "u2net", "U2NetP", "U2NET", "u2net_model"]
        
        for name in modelNames {
            if let modelURL = Bundle.main.url(forResource: name, withExtension: "mlmodelc") {
                do {
                    let model = try MLModel(contentsOf: modelURL)
                    u2netModel = try VNCoreMLModel(for: model)
                    print("✅ U2-Net model loaded: \(name)")
                    return
                } catch {
                    print("Failed to load \(name): \(error)")
                }
            }
        }
        
        print("⚠️ No U2-Net model found, using fallback segmentation")
    }
    
    private func setupCamera() {
        session.beginConfiguration()
        session.sessionPreset = .high
        
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else {
            print("❌ No camera found")
            return
        }
        
        do {
            let input = try AVCaptureDeviceInput(device: device)
            if session.canAddInput(input) {
                session.addInput(input)
            }
            
            videoOutput.setSampleBufferDelegate(self, queue: videoQueue)
            videoOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
            
            if session.canAddOutput(videoOutput) {
                session.addOutput(videoOutput)
                
                // Set correct video orientation
                if let connection = videoOutput.connection(with: .video) {
                    if connection.isVideoOrientationSupported {
                        connection.videoOrientation = .portrait
                    }
                }
            }
            
            session.commitConfiguration()
            print("✅ Camera configured with portrait orientation")
        } catch {
            print("❌ Camera setup failed: \(error)")
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
    
    private func processWithU2Net(pixelBuffer: CVPixelBuffer) {
        let now = Date()
        guard now.timeIntervalSince(lastProcessTime) >= processInterval else { return }
        lastProcessTime = now
        
        if let model = u2netModel {
            let request = VNCoreMLRequest(model: model) { [weak self] request, error in
                guard let results = request.results as? [VNPixelBufferObservation],
                      let maskBuffer = results.first?.pixelBuffer else {
                    self?.fallbackSegmentation(pixelBuffer: pixelBuffer)
                    return
                }
                self?.applySegmentation(originalBuffer: pixelBuffer, maskBuffer: maskBuffer)
            }
            
            request.imageCropAndScaleOption = .scaleFit  // Maintain aspect ratio
            
            let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
            try? handler.perform([request])
        } else {
            fallbackSegmentation(pixelBuffer: pixelBuffer)
        }
    }
    
    private func fallbackSegmentation(pixelBuffer: CVPixelBuffer) {
        // Vision framework fallback for saliency detection
        let request = VNGenerateObjectnessBasedSaliencyImageRequest { [weak self] request, error in
            guard let observation = request.results?.first as? VNSaliencyImageObservation else { return }
            self?.applySegmentation(originalBuffer: pixelBuffer, maskBuffer: observation.pixelBuffer)
        }
        
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
        try? handler.perform([request])
    }
    
    private func applySegmentation(originalBuffer: CVPixelBuffer, maskBuffer: CVPixelBuffer) {
        autoreleasepool {
            CVPixelBufferLockBaseAddress(originalBuffer, .readOnly)
            CVPixelBufferLockBaseAddress(maskBuffer, .readOnly)
            defer {
                CVPixelBufferUnlockBaseAddress(originalBuffer, .readOnly)
                CVPixelBufferUnlockBaseAddress(maskBuffer, .readOnly)
            }
            
            let width = CVPixelBufferGetWidth(originalBuffer)
            let height = CVPixelBufferGetHeight(originalBuffer)
            
            // Create CIImages
            let ciImage = CIImage(cvPixelBuffer: originalBuffer)
            let maskCI = CIImage(cvPixelBuffer: maskBuffer)
            
            // Scale mask to match original dimensions
            let scaleX = CGFloat(width) / maskCI.extent.width
            let scaleY = CGFloat(height) / maskCI.extent.height
            let scaledMask = maskCI.transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))
            
            // Apply mask using blend filter for transparency
            let blendFilter = CIFilter(name: "CIBlendWithMask")
            blendFilter?.setValue(ciImage, forKey: kCIInputImageKey)
            
            // Create transparent background
            let transparent = CIImage(color: CIColor(red: 0, green: 0, blue: 0, alpha: 0))
                .cropped(to: ciImage.extent)
            blendFilter?.setValue(transparent, forKey: kCIInputBackgroundImageKey)
            blendFilter?.setValue(scaledMask, forKey: kCIInputMaskImageKey)
            
            guard let output = blendFilter?.outputImage else {
                print("❌ Blend filter failed")
                return
            }
            
            // Create CGImage preserving transparency
            let context = CIContext(options: [.useSoftwareRenderer: false])
            guard let cgImage = context.createCGImage(output, from: output.extent) else {
                print("❌ Failed to create CGImage")
                return
            }
            
            // Create final UIImage WITHOUT rotation
            let finalImage = UIImage(cgImage: cgImage, scale: 1.0, orientation: .up)
            
            DispatchQueue.main.async {
                self.segmentedImage = finalImage
                print("✅ Segmented furniture ready (size: \(finalImage.size))")
            }
        }
    }
}

extension U2NetCameraModel: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        processWithU2Net(pixelBuffer: pixelBuffer)
    }
}
