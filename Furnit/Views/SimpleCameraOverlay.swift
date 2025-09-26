import SwiftUI
import AVFoundation
import CoreML
import Vision
import CoreImage

struct SimpleCameraOverlay: View {
    @Binding var capturedImage: UIImage?
    @Binding var isShowingCamera: Bool
    @StateObject private var camera = U2NetCameraModel()
    @State private var dragOffset = CGSize.zero
    @State private var position = CGPoint(x: UIScreen.main.bounds.width / 2, y: UIScreen.main.bounds.height / 2)
    
    var body: some View {
        // Small floating window with live segmented furniture
        VStack(spacing: 0) {
            // Title bar for dragging
            HStack {
                Text("Live Furniture")
                    .font(.caption)
                    .foregroundColor(.white)
                
                Spacer()
                
                Button(action: {
                    isShowingCamera = false
                }) {
                    Image(systemName: "xmark.circle.fill")
                        .font(.system(size: 18))
                        .foregroundColor(.white.opacity(0.8))
                }
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(Color.black.opacity(0.8))
            .gesture(
                DragGesture()
                    .onChanged { value in
                        dragOffset = value.translation
                    }
                    .onEnded { value in
                        position.x += value.translation.width
                        position.y += value.translation.height
                        dragOffset = .zero
                    }
            )
            
            // Live segmented camera feed (small window)
            ZStack {
                if let segmented = camera.segmentedImage {
                    // Show segmented image with transparency already applied
                    Image(uiImage: segmented)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .frame(width: 320, height: 240)
                } else {
                    // Show camera preview while waiting for segmentation
                    CameraPreviewLayer(session: camera.session)
                        .frame(width: 320, height: 240)
                }
            }
            .frame(width: 320, height: 240)
            .background(Color.clear)
            .clipped()
        }
        .cornerRadius(8)
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(Color.white.opacity(0.3), lineWidth: 1)
        )
        .shadow(color: .black.opacity(0.5), radius: 10, x: 0, y: 5)
        .position(x: position.x + dragOffset.width, y: position.y + dragOffset.height)
        .onAppear {
            camera.checkCameraPermission()
            // Add delay to ensure camera initializes properly
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                camera.startSession()
            }
        }
        .onDisappear {
            camera.stopSession()
        }
    }
}

// Camera Preview Layer
struct CameraPreviewLayer: UIViewRepresentable {
    let session: AVCaptureSession
    
    func makeUIView(context: Context) -> UIView {
        let view = UIView(frame: CGRect(x: 0, y: 0, width: 320, height: 240))
        view.backgroundColor = .black
        
        // Create and add preview layer
        DispatchQueue.main.async {
            let previewLayer = AVCaptureVideoPreviewLayer(session: session)
            previewLayer.frame = view.bounds
            previewLayer.videoGravity = .resizeAspectFill
            previewLayer.connection?.videoOrientation = .portrait
            view.layer.addSublayer(previewLayer)
            
            // Start the session if not running
            if !session.isRunning {
                DispatchQueue.global(qos: .background).async {
                    session.startRunning()
                    print("📷 Started session from preview layer")
                }
            }
        }
        
        return view
    }
    
    func updateUIView(_ uiView: UIView, context: Context) {
        DispatchQueue.main.async {
            if let previewLayer = uiView.layer.sublayers?.first(where: { $0 is AVCaptureVideoPreviewLayer }) as? AVCaptureVideoPreviewLayer {
                previewLayer.frame = uiView.bounds
                previewLayer.connection?.videoOrientation = .portrait
            }
        }
    }
}

// U2-Net Camera Model for Furniture Segmentation
class U2NetCameraModel: NSObject, ObservableObject {
    let session = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let videoQueue = DispatchQueue(label: "videoQueue", qos: .userInitiated)
    
    @Published var segmentedImage: UIImage?
    private var u2netModel: VNCoreMLModel?
    private var lastProcessTime = Date()
    private let processInterval: TimeInterval = 0.1 // Process every 100ms for smooth preview
    
    override init() {
        super.init()
        setupCamera()
        loadU2NetModel()
        
        // Start session immediately after setup
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
            self.startSession()
        }
    }
    
    func checkCameraPermission() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            print("✅ Camera permission authorized")
            if !session.isRunning {
                setupCamera()
                startSession()
            }
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { granted in
                if granted {
                    print("✅ Camera permission granted")
                    DispatchQueue.main.async {
                        self.setupCamera()
                        self.startSession()
                    }
                }
            }
        case .denied, .restricted:
            print("❌ Camera permission denied")
        @unknown default:
            break
        }
    }
    
    private func loadU2NetModel() {
        // Try to load U2-Net model with various possible names
        let modelNames = ["u2net", "U2Net", "u2net_model", "U2NetModel", "u2net_furniture"]
        
        for name in modelNames {
            if let modelURL = Bundle.main.url(forResource: name, withExtension: "mlmodelc") {
                do {
                    let model = try MLModel(contentsOf: modelURL)
                    u2netModel = try VNCoreMLModel(for: model)
                    print("✅ U2-Net model loaded: \(name)")
                    return
                } catch {
                    print("⚠️ Failed to load model \(name): \(error)")
                }
            }
        }
        
        print("❌ U2-Net model not found - using fallback segmentation")
    }
    
    private func setupCamera() {
        guard AVCaptureDevice.authorizationStatus(for: .video) == .authorized else {
            print("⚠️ Camera not authorized yet")
            return
        }
        
        session.beginConfiguration()
        session.sessionPreset = .hd1280x720
        
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else {
            print("❌ No camera device found")
            session.commitConfiguration()
            return
        }
        
        do {
            let input = try AVCaptureDeviceInput(device: device)
            
            if session.canAddInput(input) {
                session.addInput(input)
                print("✅ Camera input added")
            }
            
            videoOutput.setSampleBufferDelegate(self, queue: videoQueue)
            videoOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
            videoOutput.alwaysDiscardsLateVideoFrames = true
            
            if session.canAddOutput(videoOutput) {
                session.addOutput(videoOutput)
                print("✅ Video output added")
            }
            
            session.commitConfiguration()
            print("✅ Camera setup completed")
        } catch {
            print("❌ Failed to create camera input: \(error)")
            session.commitConfiguration()
        }
    }
    
    func startSession() {
        if !session.isRunning {
            DispatchQueue.global(qos: .background).async { [weak self] in
                self?.session.startRunning()
                DispatchQueue.main.async {
                    print("📷 Camera session started: \(self?.session.isRunning ?? false)")
                }
            }
        }
    }
    
    func stopSession() {
        if session.isRunning {
            DispatchQueue.global(qos: .userInitiated).async {
                self.session.stopRunning()
            }
        }
    }
    
    private func processWithU2Net(pixelBuffer: CVPixelBuffer) {
        // Throttle processing for performance
        let now = Date()
        guard now.timeIntervalSince(lastProcessTime) >= processInterval else { return }
        lastProcessTime = now
        
        if let model = u2netModel {
            // Use U2-Net model
            let request = VNCoreMLRequest(model: model) { [weak self] request, error in
                if let error = error {
                    print("❌ U2-Net processing error: \(error)")
                    return
                }
                
                guard let results = request.results as? [VNPixelBufferObservation],
                      let segmentationMask = results.first?.pixelBuffer else {
                    print("⚠️ No segmentation results, using fallback")
                    self?.fallbackSegmentation(pixelBuffer: pixelBuffer)
                    return
                }
                
                self?.applySegmentation(originalBuffer: pixelBuffer, maskBuffer: segmentationMask)
            }
            
            request.imageCropAndScaleOption = .scaleFill
            
            let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
            do {
                try handler.perform([request])
            } catch {
                print("❌ Failed to perform U2-Net request: \(error)")
            }
        } else {
            // Fallback to Vision framework for object segmentation
            fallbackSegmentation(pixelBuffer: pixelBuffer)
        }
    }
    
    private func fallbackSegmentation(pixelBuffer: CVPixelBuffer) {
        // Use Vision's saliency detection as fallback for foreground extraction
        let request = VNGenerateObjectnessBasedSaliencyImageRequest { [weak self] request, error in
            guard let observation = request.results?.first as? VNSaliencyImageObservation else { return }
            
            let maskBuffer = observation.pixelBuffer
            self?.applySegmentation(originalBuffer: pixelBuffer, maskBuffer: maskBuffer)
        }
        
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
        try? handler.perform([request])
    }
    
    private func applySegmentation(originalBuffer: CVPixelBuffer, maskBuffer: CVPixelBuffer) {
        // Get dimensions
        let width = CVPixelBufferGetWidth(originalBuffer)
        let height = CVPixelBufferGetHeight(originalBuffer)
        
        // Create CIImages
        let originalCI = CIImage(cvPixelBuffer: originalBuffer)
        let maskCI = CIImage(cvPixelBuffer: maskBuffer)
        
        // Scale mask to match original dimensions
        let scaleX = CGFloat(width) / maskCI.extent.width
        let scaleY = CGFloat(height) / maskCI.extent.height
        let scaledMask = maskCI.transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))
        
        // NO INVERSION - U2-Net outputs white for foreground (furniture), black for background
        // This is what we want for the mask
        
        // Apply mask to original using CIBlendWithMask
        let blendFilter = CIFilter(name: "CIBlendWithMask")
        blendFilter?.setValue(originalCI, forKey: kCIInputImageKey)
        // Set transparent background
        let clearBackground = CIImage(color: CIColor(red: 0, green: 0, blue: 0, alpha: 0))
            .cropped(to: originalCI.extent)
        blendFilter?.setValue(clearBackground, forKey: kCIInputBackgroundImageKey)
        blendFilter?.setValue(scaledMask, forKey: kCIInputMaskImageKey)
        
        guard let maskedImage = blendFilter?.outputImage else {
            print("❌ Failed to apply mask blend")
            return
        }
        
        let context = CIContext()
        let rect = CGRect(x: 0, y: 0, width: width, height: height)
        
        // Create CGImage with alpha channel preserved
        guard let cgImage = context.createCGImage(maskedImage, from: rect, format: .RGBA8, colorSpace: CGColorSpaceCreateDeviceRGB()) else {
            print("❌ Failed to create CGImage with alpha")
            return
        }
        
        // Create UIImage with proper orientation
        let finalImage = UIImage(cgImage: cgImage, scale: 1.0, orientation: .right)
        
        DispatchQueue.main.async {
            self.segmentedImage = finalImage
            print("✅ Segmented chair with transparent background")
        }
    }
}

// Video Output Delegate
extension U2NetCameraModel: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        
        // Process every 10th frame for debugging
        if Int.random(in: 0...9) == 0 {
            print("📹 Processing frame with U2-Net")
        }
        
        processWithU2Net(pixelBuffer: pixelBuffer)
    }
}
