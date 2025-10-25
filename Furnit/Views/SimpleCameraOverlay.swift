import SwiftUI
import AVFoundation
import CoreML
import Vision
import CoreImage

struct SimpleCameraOverlay: View {
    @Binding var capturedImage: UIImage?
    @Binding var isShowingCamera: Bool
    @StateObject private var camera = U2NetCameraModel()
    
    // Start fully minimized at 30%
    @State private var scaleMultiplier: CGFloat = 0.3  // Start at 30% scale (minimized)
    
    var body: some View {
        ZStack {
            // TRANSPARENT BACKGROUND - No camera preview shown!
            Color.clear
                .ignoresSafeArea()
            
            // Display segmented furniture without cropping function
            if let segmented = camera.segmentedImage {
                Image(uiImage: segmented)
                    .resizable()
                    .scaledToFill()
                    .scaleEffect(scaleMultiplier)
                    .position(x: UIScreen.main.bounds.width / 2,
                             y: UIScreen.main.bounds.height / 2)
                    .allowsHitTesting(false)
                    .ignoresSafeArea()
            }
            
            // UI controls overlay
            VStack {
                HStack {
                    // Size adjustment slider
                    HStack {
                        Image(systemName: "minus.magnifyingglass")
                            .foregroundColor(.white)
                        
                        // Slider adjusts scale from 30% to 100%
                        Slider(value: $scaleMultiplier, in: 0.3...1.0)
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
                .padding(.top, 50)
                
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
                        Text("Size: \(Int(scaleMultiplier * 100))%")
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
                            print("📸 Captured segmented furniture")
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
        session.sessionPreset = .photo  // Use photo preset for full resolution
        
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
            videoOutput.alwaysDiscardsLateVideoFrames = true
            
            if session.canAddOutput(videoOutput) {
                session.addOutput(videoOutput)
                
                // Set correct video orientation
                if let connection = videoOutput.connection(with: .video) {
                    if connection.isVideoOrientationSupported {
                        connection.videoOrientation = .portrait
                    }
                    // Disable video mirroring
                    if connection.isVideoMirroringSupported {
                        connection.isVideoMirrored = false
                    }
                }
            }
            
            session.commitConfiguration()
            print("✅ Camera configured with photo preset")
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
            
            // Use scaleFit to preserve full camera view
            request.imageCropAndScaleOption = .scaleFit
            
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
        
        // Note: Saliency requests don't have imageCropAndScaleOption
        
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
            
            // Create CGImage WITHOUT any cropping
            let context = CIContext(options: [.useSoftwareRenderer: false])
            guard let cgImage = context.createCGImage(output, from: output.extent) else {
                print("❌ Failed to create CGImage")
                return
            }
            
            let finalImage = UIImage(cgImage: cgImage, scale: 1.0, orientation: .up)
            
            DispatchQueue.main.async {
                self.segmentedImage = finalImage
                print("✅ Segmented furniture ready (size: \(finalImage.size))")
            }
        }
    }
    
    // Helper function to find non-transparent bounds from mask
    private func getNonTransparentBounds(from maskImage: CIImage, context: CIContext) -> CGRect? {
        // Sample the mask to find bounds of non-transparent pixels
        let extent = maskImage.extent
        
        // Create a small bitmap to sample the mask efficiently
        guard let cgImage = context.createCGImage(maskImage, from: extent) else { return nil }
        
        let width = Int(extent.width)
        let height = Int(extent.height)
        
        // Sample every 10th pixel for efficiency
        let sampleRate = 10
        var minX = width
        var maxX = 0
        var minY = height
        var maxY = 0
        var foundContent = false
        
        // Create bitmap context to read pixels
        let colorSpace = CGColorSpaceCreateDeviceGray()
        let bitmapInfo = CGImageAlphaInfo.none.rawValue
        guard let bitmapContext = CGContext(data: nil,
                                            width: width,
                                            height: height,
                                            bitsPerComponent: 8,
                                            bytesPerRow: width,
                                            space: colorSpace,
                                            bitmapInfo: bitmapInfo) else {
            return nil
        }
        
        bitmapContext.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        guard let pixelData = bitmapContext.data else { return nil }
        
        let data = pixelData.bindMemory(to: UInt8.self, capacity: width * height)
        
        for y in stride(from: 0, to: height, by: sampleRate) {
            for x in stride(from: 0, to: width, by: sampleRate) {
                let pixelIndex = y * width + x
                let value = data[pixelIndex]
                
                if value > 10 { // Non-transparent threshold
                    minX = min(minX, x)
                    maxX = max(maxX, x)
                    minY = min(minY, y)
                    maxY = max(maxY, y)
                    foundContent = true
                }
            }
        }
        
        guard foundContent else { return nil }
        
        // Add padding and expand to actual boundaries (since we sampled)
        let padding = 10
        minX = max(0, minX - padding - sampleRate)
        maxX = min(width, maxX + padding + sampleRate)
        minY = max(0, minY - padding - sampleRate)
        maxY = min(height, maxY + padding + sampleRate)
        
        return CGRect(x: CGFloat(minX),
                      y: CGFloat(minY),
                      width: CGFloat(maxX - minX),
                      height: CGFloat(maxY - minY))
    }
}

extension U2NetCameraModel: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        processWithU2Net(pixelBuffer: pixelBuffer)
    }
}
