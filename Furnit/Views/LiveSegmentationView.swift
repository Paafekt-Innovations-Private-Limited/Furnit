import SwiftUI
import AVFoundation
import Vision
import CoreImage
import Metal
import MetalKit

// Live camera view with real-time U²-Net segmentation
struct LiveSegmentationView: View {
    @Environment(\.dismiss) private var dismiss
    @StateObject private var segmentationManager = LiveSegmentationManager()
    @State private var showControls = true
    
    var body: some View {
        ZStack {
            // Live segmented camera feed
            if segmentationManager.isRunning {
                LiveSegmentationPreview(manager: segmentationManager)
                    .ignoresSafeArea()
            } else {
                Color.black
                    .ignoresSafeArea()
                    .onAppear {
                        segmentationManager.startSession()
                    }
            }
            
            // UI Overlay
            if showControls {
                VStack {
                    // Top bar
                    HStack {
                        Button("Close") {
                            dismiss()
                        }
                        .foregroundColor(.white)
                        .padding()
                        .background(Color.black.opacity(0.5))
                        .cornerRadius(10)
                        
                        Spacer()
                        
                        // Segmentation mode toggle
                        Menu {
                            Button("Person Segmentation") {
                                segmentationManager.segmentationType = .person
                            }
                            Button("Object Segmentation") {
                                segmentationManager.segmentationType = .object
                            }
                            Button("Furniture Focus") {
                                segmentationManager.segmentationType = .furniture
                            }
                        } label: {
                            Label(segmentationManager.segmentationType.displayName, systemImage: "camera.filters")
                                .foregroundColor(.white)
                                .padding()
                                .background(Color.black.opacity(0.5))
                                .cornerRadius(10)
                        }
                    }
                    .padding()
                    
                    Spacer()
                    
                    // Bottom controls
                    HStack {
                        // Toggle background
                        Button(action: {
                            segmentationManager.showBackground.toggle()
                        }) {
                            Image(systemName: segmentationManager.showBackground ? "square.on.square" : "square.dashed")
                                .foregroundColor(.white)
                                .frame(width: 50, height: 50)
                                .background(Color.black.opacity(0.5))
                                .clipShape(Circle())
                        }
                        
                        Spacer()
                        
                        // Hide/Show controls
                        Button(action: {
                            withAnimation {
                                showControls.toggle()
                            }
                        }) {
                            Image(systemName: showControls ? "eye.slash" : "eye")
                                .foregroundColor(.white)
                                .frame(width: 50, height: 50)
                                .background(Color.black.opacity(0.5))
                                .clipShape(Circle())
                        }
                    }
                    .padding()
                }
            } else {
                // Minimal control to show UI again
                VStack {
                    Spacer()
                    HStack {
                        Spacer()
                        Button(action: {
                            withAnimation {
                                showControls.toggle()
                            }
                        }) {
                            Image(systemName: "eye")
                                .foregroundColor(.white)
                                .frame(width: 40, height: 40)
                                .background(Color.black.opacity(0.3))
                                .clipShape(Circle())
                        }
                        .padding()
                    }
                }
            }
        }
        .onDisappear {
            segmentationManager.stopSession()
        }
    }
}

// Live segmentation preview using Metal for performance
struct LiveSegmentationPreview: UIViewRepresentable {
    let manager: LiveSegmentationManager
    
    func makeUIView(context: Context) -> MTKView {
        let metalView = MTKView()
        metalView.device = manager.metalDevice
        metalView.delegate = manager
        metalView.framebufferOnly = false
        metalView.colorPixelFormat = .bgra8Unorm
        metalView.contentScaleFactor = UIScreen.main.scale
        metalView.enableSetNeedsDisplay = false
        metalView.isPaused = false
        metalView.preferredFramesPerSecond = 30
        
        manager.setupMetal(with: metalView)
        
        return metalView
    }
    
    func updateUIView(_ uiView: MTKView, context: Context) {}
}

// Segmentation type options
enum SegmentationType {
    case person
    case object
    case furniture
    
    var displayName: String {
        switch self {
        case .person:
            return "Person"
        case .object:
            return "Objects"
        case .furniture:
            return "Furniture"
        }
    }
}

// Live segmentation manager with real-time processing
class LiveSegmentationManager: NSObject, ObservableObject {
    // Published properties
    @Published var isRunning = false
    @Published var showBackground = false
    @Published var segmentationType: SegmentationType = .furniture {
        didSet {
            updateSegmentationRequest()
        }
    }
    
    // AVFoundation
    private let captureSession = AVCaptureSession()
    private var videoOutput = AVCaptureVideoDataOutput()
    private let videoQueue = DispatchQueue(label: "video.segmentation.queue", qos: .userInteractive)
    
    // Vision
    private var segmentationRequest: VNRequest?
    private let requestHandler = VNSequenceRequestHandler()
    
    // Metal
    var metalDevice: MTLDevice?
    private var metalCommandQueue: MTLCommandQueue?
    private var metalView: MTKView?
    private var ciContext: CIContext?
    
    // Rendering pipeline
    private var renderPipelineState: MTLRenderPipelineState?
    private var currentTexture: MTLTexture?
    private var segmentedImage: CIImage?
    
    // Performance
    private var lastFrameTime: TimeInterval = 0
    private let frameSkipThreshold: TimeInterval = 1.0 / 15.0 // Process at 15 FPS max
    
    override init() {
        super.init()
        setupMetal()
        setupCamera()
        setupSegmentation()
    }
    
    // Setup Metal for GPU acceleration
    private func setupMetal() {
        metalDevice = MTLCreateSystemDefaultDevice()
        if let device = metalDevice {
            metalCommandQueue = device.makeCommandQueue()
            ciContext = CIContext(mtlDevice: device)
            
            // Create render pipeline
            if let library = device.makeDefaultLibrary(),
               let vertexFunction = library.makeFunction(name: "vertexShader"),
               let fragmentFunction = library.makeFunction(name: "fragmentShader") {
                
                let pipelineDescriptor = MTLRenderPipelineDescriptor()
                pipelineDescriptor.vertexFunction = vertexFunction
                pipelineDescriptor.fragmentFunction = fragmentFunction
                pipelineDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
                
                renderPipelineState = try? device.makeRenderPipelineState(descriptor: pipelineDescriptor)
            }
        }
    }
    
    func setupMetal(with view: MTKView) {
        metalView = view
    }
    
    // Setup camera session
    private func setupCamera() {
        captureSession.sessionPreset = .hd1280x720 // Balance quality and performance
        
        guard let camera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
              let input = try? AVCaptureDeviceInput(device: camera) else { return }
        
        if captureSession.canAddInput(input) {
            captureSession.addInput(input)
        }
        
        videoOutput.setSampleBufferDelegate(self, queue: videoQueue)
        videoOutput.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        videoOutput.alwaysDiscardsLateVideoFrames = true
        
        if captureSession.canAddOutput(videoOutput) {
            captureSession.addOutput(videoOutput)
        }
        
        // Set video orientation
        if let connection = videoOutput.connection(with: .video) {
            connection.videoOrientation = .portrait
        }
    }
    
    // Setup Vision segmentation request
    private func setupSegmentation() {
        updateSegmentationRequest()
    }
    
    private func updateSegmentationRequest() {
        switch segmentationType {
        case .person:
            // Use built-in person segmentation
            segmentationRequest = VNGeneratePersonSegmentationRequest { [weak self] request, error in
                self?.handleSegmentation(request: request, error: error)
            }
            (segmentationRequest as? VNGeneratePersonSegmentationRequest)?.qualityLevel = .balanced
            
        case .object, .furniture:
            // For objects/furniture, we'd ideally use a custom U²-Net model
            // For now, using person segmentation with inverted mask as a placeholder
            // In production, you'd load a custom CoreML U²-Net model here
            segmentationRequest = VNGeneratePersonSegmentationRequest { [weak self] request, error in
                self?.handleSegmentation(request: request, error: error, invertMask: true)
            }
            (segmentationRequest as? VNGeneratePersonSegmentationRequest)?.qualityLevel = .balanced
        }
    }
    
    // Handle segmentation results
    private func handleSegmentation(request: VNRequest, error: Error?, invertMask: Bool = false) {
        guard error == nil,
              let results = request.results as? [VNPixelBufferObservation],
              let segmentationMask = results.first else { return }
        
        // Store the segmented result for rendering
        var maskImage = CIImage(cvPixelBuffer: segmentationMask.pixelBuffer)
        
        if invertMask {
            // Invert mask for object/furniture detection simulation
            if let invertFilter = CIFilter(name: "CIColorInvert") {
                invertFilter.setValue(maskImage, forKey: kCIInputImageKey)
                maskImage = invertFilter.outputImage ?? maskImage
            }
        }
        
        DispatchQueue.main.async { [weak self] in
            self?.segmentedImage = maskImage
        }
    }
    
    // Start camera session
    func startSession() {
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            self?.captureSession.startRunning()
            DispatchQueue.main.async {
                self?.isRunning = true
            }
        }
    }
    
    // Stop camera session
    func stopSession() {
        captureSession.stopRunning()
        isRunning = false
    }
}

// MARK: - AVCaptureVideoDataOutputSampleBufferDelegate
extension LiveSegmentationManager: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        // Frame rate limiting
        let currentTime = CACurrentMediaTime()
        if currentTime - lastFrameTime < frameSkipThreshold {
            return
        }
        lastFrameTime = currentTime
        
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        
        // Run segmentation
        if let request = segmentationRequest {
            try? requestHandler.perform([request], on: pixelBuffer, orientation: .up)
        }
        
        // Process and display
        processFrame(pixelBuffer: pixelBuffer)
    }
    
    private func processFrame(pixelBuffer: CVPixelBuffer) {
        let inputImage = CIImage(cvPixelBuffer: pixelBuffer)
        var outputImage = inputImage
        
        // Apply segmentation mask if available
        if let maskImage = segmentedImage {
            // Resize mask to match input
            let scaleX = inputImage.extent.width / maskImage.extent.width
            let scaleY = inputImage.extent.height / maskImage.extent.height
            let scaledMask = maskImage.transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))
            
            if showBackground {
                // Blend with reduced background
                if let blendFilter = CIFilter(name: "CIBlendWithMask") {
                    let dimmedBackground = inputImage.applyingFilter("CIColorControls", parameters: [
                        "inputBrightness": -0.3,
                        "inputSaturation": 0.5
                    ])
                    
                    blendFilter.setValue(inputImage, forKey: kCIInputImageKey)
                    blendFilter.setValue(dimmedBackground, forKey: kCIInputBackgroundImageKey)
                    blendFilter.setValue(scaledMask, forKey: kCIInputMaskImageKey)
                    outputImage = blendFilter.outputImage ?? inputImage
                }
            } else {
                // Remove background completely
                if let blendFilter = CIFilter(name: "CIBlendWithMask") {
                    blendFilter.setValue(inputImage, forKey: kCIInputImageKey)
                    blendFilter.setValue(CIImage(color: .clear), forKey: kCIInputBackgroundImageKey)
                    blendFilter.setValue(scaledMask, forKey: kCIInputMaskImageKey)
                    outputImage = blendFilter.outputImage ?? inputImage
                }
            }
        }
        
        // Render to Metal view
        DispatchQueue.main.async { [weak self] in
            self?.renderToMetalView(outputImage)
        }
    }
    
    private func renderToMetalView(_ image: CIImage) {
        guard let metalView = metalView,
              let drawable = metalView.currentDrawable,
              let commandBuffer = metalCommandQueue?.makeCommandBuffer() else { return }
        
        let bounds = CGRect(origin: .zero, size: metalView.drawableSize)
        let scaledImage = image.transformed(by: CGAffineTransform(scaleX: 1, y: -1)
            .translatedBy(x: 0, y: -image.extent.height))
        
        ciContext?.render(scaledImage, to: drawable.texture, commandBuffer: commandBuffer, bounds: bounds, colorSpace: CGColorSpaceCreateDeviceRGB())
        
        commandBuffer.present(drawable)
        commandBuffer.commit()
    }
}

// MARK: - MTKViewDelegate
extension LiveSegmentationManager: MTKViewDelegate {
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}
    
    func draw(in view: MTKView) {
        // Drawing is handled in processFrame
    }
}
