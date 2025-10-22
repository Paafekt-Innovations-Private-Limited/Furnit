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
    @State private var scale: CGFloat = 1.0
    @State private var lastScaleValue: CGFloat = 1.0
    
    // Vertical position constraints
    @State private var verticalOffset: CGFloat = 0
    private let minVerticalPosition: CGFloat = 100 // Minimum distance from top
    private let maxVerticalPosition: CGFloat = UIScreen.main.bounds.height - 200 // Max distance from bottom
    
    // Define min and max sizes for the camera preview
    private let minSize = CGSize(width: 280, height: 157)  // Minimum size (16:9)
    private let maxSize = CGSize(width: UIScreen.main.bounds.width, height: UIScreen.main.bounds.height)  // Full screen size
    private let baseSize = CGSize(width: 640, height: 360) // Much wider default size to show more horizontal content
    
    var body: some View {
        // Small floating window with live segmented furniture
        VStack(spacing: 0) {
            // Title bar for dragging horizontally
            HStack {
                Text("Live Furniture")
                    .font(.caption)
                    .foregroundColor(.white)
                
                Spacer()
                
                // Vertical position indicator
                Image(systemName: "arrow.up.and.down")
                    .font(.caption)
                    .foregroundColor(.white.opacity(0.6))
                
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
                        // Horizontal movement only on title bar
                        dragOffset = CGSize(
                            width: value.translation.width,
                            height: 0
                        )
                    }
                    .onEnded { value in
                        position.x += value.translation.width
                        // Constrain horizontal position
                        position.x = max(currentSize.width/2, min(UIScreen.main.bounds.width - currentSize.width/2, position.x))
                        dragOffset = .zero
                    }
            )
            
            // Live segmented camera feed (resizable window)
            ZStack {
                if let segmented = camera.segmentedImage {
                    // Show segmented image with transparency already applied
                    Image(uiImage: segmented)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .frame(width: currentSize.width, height: currentSize.height)
                } else {
                    // Show camera preview while waiting for segmentation
                    // FIXED: Using resizeAspect to show full camera view
                    CameraPreviewLayer(session: camera.session, videoGravity: .resizeAspect)
                        .frame(width: currentSize.width, height: currentSize.height)
                        .background(Color.black) // Add black background for letterboxing
                }
                
                // Vertical drag handle on the left side
                HStack {
                    // Left drag handle for vertical movement
                    Rectangle()
                        .fill(
                            LinearGradient(
                                colors: [Color.white.opacity(0.1), Color.white.opacity(0.3), Color.white.opacity(0.1)],
                                startPoint: .top,
                                endPoint: .bottom
                            )
                        )
                        .frame(width: 20)
                        .overlay(
                            VStack {
                                Image(systemName: "chevron.up")
                                    .font(.caption2)
                                    .foregroundColor(.white.opacity(0.5))
                                Spacer()
                                Image(systemName: "chevron.down")
                                    .font(.caption2)
                                    .foregroundColor(.white.opacity(0.5))
                            }
                            .padding(.vertical, 8)
                        )
                        .gesture(
                            DragGesture()
                                .onChanged { value in
                                    // Update vertical position
                                    let newY = position.y + value.translation.height
                                    // Apply constraints
                                    if newY >= minVerticalPosition && newY <= maxVerticalPosition {
                                        verticalOffset = value.translation.height
                                    }
                                }
                                .onEnded { value in
                                    let newY = position.y + value.translation.height
                                    // Apply constraints and update position
                                    position.y = max(minVerticalPosition, min(maxVerticalPosition, newY))
                                    verticalOffset = 0
                                }
                        )
                    
                    Spacer()
                }
                .frame(width: currentSize.width, height: currentSize.height)
            }
            .frame(width: currentSize.width, height: currentSize.height)
            .background(Color.clear)
            .clipped()
            .gesture(
                MagnificationGesture()
                    .onChanged { value in
                        let delta = value / lastScaleValue
                        lastScaleValue = value
                        
                        let newScale = scale * delta
                        let clampedScale = max(minScale, min(maxScale, newScale))
                        scale = clampedScale
                    }
                    .onEnded { value in
                        lastScaleValue = 1.0
                    }
            )
            
            // Bottom resize handle
            HStack {
                Spacer()
                
                // Capture button
                Button(action: {
                    if let currentImage = camera.segmentedImage {
                        capturedImage = currentImage
                        print("📸 Captured segmented furniture image")
                    }
                }) {
                    Image(systemName: "camera.circle.fill")
                        .font(.title2)
                        .foregroundColor(.white)
                }
                .padding(.horizontal)
                
                // Resize handle
                Image(systemName: "arrow.up.left.and.arrow.down.right")
                    .font(.caption)
                    .foregroundColor(.white.opacity(0.5))
                    .padding(8)
                    .gesture(
                        DragGesture()
                            .onChanged { value in
                                // Diagonal resize
                                let scaleChange = (value.translation.width + value.translation.height) / 200
                                let newScale = scale + scaleChange
                                scale = max(minScale, min(maxScale, newScale))
                            }
                    )
            }
            .frame(height: 30)
            .background(Color.black.opacity(0.6))
        }
        .cornerRadius(8)
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(Color.white.opacity(0.3), lineWidth: 1)
        )
        .shadow(color: .black.opacity(0.5), radius: 10, x: 0, y: 5)
        .position(
            x: position.x + dragOffset.width,
            y: position.y + dragOffset.height + verticalOffset
        )
        .onAppear {
            AVCaptureDevice.requestAccess(for: .video) { granted in
                if granted {
                    camera.startSession()
                } else {
                    print("⚠️ Camera access denied")
                }
            }
        }
        .onDisappear {
            camera.stopSession()
        }
    }
    
    // Computed properties for sizing
    private var currentSize: CGSize {
        CGSize(
            width: baseSize.width * scale,
            height: baseSize.height * scale
        )
    }
    
    private var minScale: CGFloat {
        max(minSize.width / baseSize.width, minSize.height / baseSize.height)
    }
    
    private var maxScale: CGFloat {
        min(maxSize.width / baseSize.width, maxSize.height / baseSize.height)
    }
}

// FIXED: Camera Preview Layer with proper aspect ratio handling
struct CameraPreviewLayer: UIViewRepresentable {
    let session: AVCaptureSession
    let videoGravity: AVLayerVideoGravity
    
    init(session: AVCaptureSession, videoGravity: AVLayerVideoGravity = .resizeAspect) {
        self.session = session
        self.videoGravity = videoGravity
    }
    
    func makeUIView(context: Context) -> UIView {
        let view = UIView()
        view.backgroundColor = .black
        
        let previewLayer = AVCaptureVideoPreviewLayer(session: session)
        // Use resizeAspectFill to match what the ML model processes
        previewLayer.videoGravity = .resizeAspectFill
        previewLayer.frame = view.bounds
        
        // Set the layer's content mode
        previewLayer.contentsGravity = .resizeAspect
        view.layer.addSublayer(previewLayer)
        
        // Store both view and layer for proper updates
        context.coordinator.previewLayer = previewLayer
        context.coordinator.containerView = view
        
        return view
    }
    
    func updateUIView(_ uiView: UIView, context: Context) {
        // Ensure the preview layer fills the view bounds properly
        CATransaction.begin()
        CATransaction.setDisableActions(true)
        context.coordinator.previewLayer?.frame = uiView.bounds
        context.coordinator.previewLayer?.videoGravity = .resizeAspect
        CATransaction.commit()
    }
    
    func makeCoordinator() -> Coordinator {
        Coordinator()
    }
    
    class Coordinator {
        var previewLayer: AVCaptureVideoPreviewLayer?
        var containerView: UIView?
    }
}

// U2NetCameraModel class (remains the same)
class U2NetCameraModel: NSObject, ObservableObject {
    @Published var segmentedImage: UIImage?
    let session = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let videoQueue = DispatchQueue(label: "videoQueue", qos: .userInitiated)
    
    private var u2netModel: VNCoreMLModel?
    private let context = CIContext()
    
    // Throttling for performance
    private var lastProcessTime = Date()
    private let processInterval: TimeInterval = 0.1 // Process every 100ms
    
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
            print("⚠️ Camera access denied or restricted")
        }
    }
    
    private func loadU2NetModel() {
        // Try multiple model names
        let modelNames = ["u2netp", "U2Net", "u2net", "U2NetP"]
        
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
        
        // Use photo preset for wider field of view
        session.sessionPreset = .photo  // Changed from hd1920x1080
        
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
                
                // FIX: Configure the video connection properly
                if let connection = videoOutput.connection(with: .video) {
                    // Don't force any specific orientation - let it use default
                    // connection.videoOrientation = .portrait  // REMOVED - this was causing rotation
                    
                    // Disable video stabilization which can crop the image
                    if connection.isVideoStabilizationSupported {
                        connection.preferredVideoStabilizationMode = .off
                    }
                    
                    // Ensure video mirroring is correct
                    if connection.isVideoMirroringSupported {
                        connection.isVideoMirrored = false
                    }
                    
                    print("✅ Video connection configured")
                }
            }
            
            session.commitConfiguration()
            print("✅ Camera setup completed with preset: \(session.sessionPreset.rawValue)")
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
    
    // ... (rest of the processing methods remain unchanged)
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
            
            // Keep scaleFill for ML model processing
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
        
        // Create UIImage with proper orientation - restore original .right orientation
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
