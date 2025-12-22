import SwiftUI
import AVFoundation
import CoreML
import Vision
import CoreImage

// U²-Net segmentation manager for real-time processing
class U2NetSegmentationManager: NSObject, ObservableObject {
    @Published var isRunning = false
    @Published var errorMessage: String?
    
    // AVFoundation
    private let captureSession = AVCaptureSession()
    private var videoOutput = AVCaptureVideoDataOutput()
    private let videoQueue = DispatchQueue(label: "u2net.processing", qos: .userInteractive)
    
    // CoreML U²-Net model
    private var u2netModel: VNCoreMLModel?
    private var segmentationRequest: VNCoreMLRequest?
    
    // Display
    var onFrameProcessed: ((UIImage) -> Void)?
    
    override init() {
        super.init()
        loadU2NetModel()
        setupCamera()
    }
    
    // Load U²-Net CoreML model
    private func loadU2NetModel() {
        // First check if U2Net.mlmodelc exists (compiled model)
        var modelURL: URL?
        
        if let url = Bundle.main.url(forResource: "U2Net", withExtension: "mlmodelc") {
            modelURL = url
            logDebug("✅ Found U2Net.mlmodelc")
        } else if let url = Bundle.main.url(forResource: "U2Net", withExtension: "mlmodel") {
            modelURL = url
            logDebug("✅ Found U2Net.mlmodel")
        } else {
            errorMessage = "U²-Net model not found. Please add U2Net.mlmodel to your project"
            logDebug("❌ U²-Net model not found in bundle")
            logDebug("📦 Please download U2Net.mlmodel and add it to your Xcode project")
            logDebug("   Download from: https://github.com/john-rocky/CoreML-Models")
            return
        }
        
        guard let finalURL = modelURL else { return }
        
        do {
            // Load CoreML model
            let mlModel = try MLModel(contentsOf: finalURL)
            logDebug("✅ U²-Net CoreML model loaded")
            
            // Create Vision wrapper
            let visionModel = try VNCoreMLModel(for: mlModel)
            self.u2netModel = visionModel
            
            // Create segmentation request
            segmentationRequest = VNCoreMLRequest(model: visionModel) { [weak self] request, error in
                self?.processU2NetResults(request: request, error: error)
            }
            segmentationRequest?.imageCropAndScaleOption = .scaleFill
            
            logDebug("✅ U²-Net ready for real-time segmentation")
            
        } catch {
            errorMessage = "Failed to load U²-Net: \(error.localizedDescription)"
            logDebug("❌ Failed to load U²-Net model: \(error)")
        }
    }
    
    // Setup camera
    private func setupCamera() {
        captureSession.sessionPreset = .hd1280x720
        
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
        
        if let connection = videoOutput.connection(with: .video) {
            connection.videoOrientation = .portrait
        }
    }
    
    // Process U²-Net segmentation results
    private func processU2NetResults(request: VNRequest, error: Error?) {
        guard error == nil else {
            logDebug("❌ U²-Net processing error: \(error!)")
            return
        }
        
        // U²-Net outputs a grayscale saliency map
        if let results = request.results as? [VNPixelBufferObservation],
           let observation = results.first {
            // Saliency map from U²-Net
            self.currentSaliencyMap = CIImage(cvPixelBuffer: observation.pixelBuffer)
        } else if let results = request.results as? [VNCoreMLFeatureValueObservation],
                  let observation = results.first,
                  let multiArray = observation.featureValue.multiArrayValue {
            // Handle MLMultiArray output format
            self.currentSaliencyMap = convertMultiArrayToCIImage(multiArray)
        }
    }
    
    private var currentSaliencyMap: CIImage?
    private var currentFrame: CIImage?
    
    // Convert MLMultiArray to CIImage for U²-Net output
    private func convertMultiArrayToCIImage(_ multiArray: MLMultiArray) -> CIImage? {
        // U²-Net typically outputs 320x320 or 512x512
        let width = multiArray.shape[2].intValue
        let height = multiArray.shape[1].intValue
        
        // Create grayscale image from saliency map
        var pixelBuffer: CVPixelBuffer?
        CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_OneComponent8, nil, &pixelBuffer)
        
        guard let buffer = pixelBuffer else { return nil }
        
        CVPixelBufferLockBaseAddress(buffer, .init(rawValue: 0))
        let baseAddress = CVPixelBufferGetBaseAddress(buffer)
        
        for y in 0..<height {
            for x in 0..<width {
                let offset = y * width + x
                let value = multiArray[offset].floatValue
                let byte = UInt8(max(0, min(255, value * 255)))
                baseAddress?.assumingMemoryBound(to: UInt8.self)[offset] = byte
            }
        }
        
        CVPixelBufferUnlockBaseAddress(buffer, .init(rawValue: 0))
        
        return CIImage(cvPixelBuffer: buffer)
    }
    
    // Apply U²-Net segmentation mask to frame
    private func applyU2NetMask(to frame: CIImage, mask: CIImage) -> UIImage? {
        let context = CIContext()
        
        // Resize mask to match frame
        let scaleX = frame.extent.width / mask.extent.width
        let scaleY = frame.extent.height / mask.extent.height
        let scaledMask = mask.transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))
        
        // Apply mask to remove background
        guard let blendFilter = CIFilter(name: "CIBlendWithMask") else { return nil }
        blendFilter.setValue(frame, forKey: kCIInputImageKey)
        blendFilter.setValue(CIImage(color: .clear), forKey: kCIInputBackgroundImageKey)
        blendFilter.setValue(scaledMask, forKey: kCIInputMaskImageKey)
        
        guard let outputImage = blendFilter.outputImage,
              let cgImage = context.createCGImage(outputImage, from: outputImage.extent) else { return nil }
        
        return UIImage(cgImage: cgImage)
    }
    
    func startSession() {
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            self?.captureSession.startRunning()
            DispatchQueue.main.async {
                self?.isRunning = true
            }
        }
    }
    
    func stopSession() {
        captureSession.stopRunning()
        isRunning = false
    }
}

// MARK: - AVCaptureVideoDataOutputSampleBufferDelegate
extension U2NetSegmentationManager: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer),
              let request = segmentationRequest else { return }
        
        currentFrame = CIImage(cvPixelBuffer: pixelBuffer)
        
        // Run U²-Net segmentation
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
        try? handler.perform([request])
        
        // Apply segmentation and display
        if let frame = currentFrame,
           let saliencyMap = currentSaliencyMap,
           let segmentedImage = applyU2NetMask(to: frame, mask: saliencyMap) {
            DispatchQueue.main.async { [weak self] in
                self?.onFrameProcessed?(segmentedImage)
            }
        }
    }
}

// U²-Net Live View
struct U2NetLiveView: View {
    @Environment(\.dismiss) private var dismiss
    @StateObject private var segmentationManager = U2NetSegmentationManager()
    @State private var currentFrame: UIImage?
    
    var body: some View {
        ZStack {
            // Display segmented output
            if let frame = currentFrame {
                Image(uiImage: frame)
                    .resizable()
                    .scaledToFill()
                    .ignoresSafeArea()
            } else {
                Color.black
                    .ignoresSafeArea()
            }
            
            // Controls overlay
            VStack {
                HStack {
                    Button("Close") {
                        dismiss()
                    }
                    .foregroundColor(.white)
                    .padding()
                    .background(Color.black.opacity(0.5))
                    .cornerRadius(10)
                    
                    Spacer()
                    
                    if let error = segmentationManager.errorMessage {
                        Text(error)
                            .foregroundColor(.red)
                            .padding()
                            .background(Color.black.opacity(0.7))
                            .cornerRadius(10)
                    }
                }
                .padding()
                
                Spacer()
            }
        }
        .onAppear {
            segmentationManager.onFrameProcessed = { image in
                currentFrame = image
            }
            segmentationManager.startSession()
        }
        .onDisappear {
            segmentationManager.stopSession()
        }
    }
}

// INTEGRATION INSTRUCTIONS:
// 1. Download U2Net.mlmodel from: https://github.com/john-rocky/CoreML-Models
// 2. Drag U2Net.mlmodel into your Xcode project
// 3. Make sure it's added to your target
// 4. Replace LiveSegmentationView() with U2NetLiveView() in ModelViewerView.swift:
//    .sheet(isPresented: $showLiveSegmentation) {
//        U2NetLiveView()
//            .preferredColorScheme(.dark)
//    }
