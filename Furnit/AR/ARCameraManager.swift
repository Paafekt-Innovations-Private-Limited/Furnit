import AVFoundation
import UIKit
import CoreVideo

protocol ARCameraManagerDelegate {
    func cameraManager(_ manager: ARCameraManager, didOutput pixelBuffer: CVPixelBuffer)
    func cameraManager(_ manager: ARCameraManager, didCaptureImage image: UIImage, pixelBuffer: CVPixelBuffer)
    func cameraManager(_ manager: ARCameraManager, didFailWithError error: Error)
}

class ARCameraManager: NSObject, ObservableObject {
    var delegate: ARCameraManagerDelegate?
    
    private let captureSession = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let photoOutput = AVCapturePhotoOutput()
    private let sessionQueue = DispatchQueue(label: "camera.session.queue")
    private let videoOutputQueue = DispatchQueue(label: "camera.video.output.queue")
    
    // Single capture mode - when true, captures single high-quality image
    private var singleCaptureMode = false
    private var capturingPhoto = false
    
    @Published var isRunning = false
    @Published var hasPermission = false
    
    override init() {
        super.init()
        checkCameraPermission()
    }
    
    deinit {
        stopCapture()
    }
    
    private func checkCameraPermission() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            hasPermission = true
        case .notDetermined:
            requestCameraPermission()
        case .denied, .restricted:
            hasPermission = false
        @unknown default:
            hasPermission = false
        }
    }
    
    private func requestCameraPermission() {
        AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
            DispatchQueue.main.async {
                self?.hasPermission = granted
            }
        }
    }
    
    func startCapture() {
        guard hasPermission else {
            delegate?.cameraManager(self, didFailWithError: CameraError.permissionDenied)
            return
        }
        
        sessionQueue.async { [weak self] in
            // Only setup if not already configured
            if self?.captureSession.inputs.isEmpty == true && self?.captureSession.outputs.isEmpty == true {
                self?.setupCaptureSession()
            }
            self?.captureSession.startRunning()
            
            DispatchQueue.main.async {
                self?.isRunning = true
            }
        }
    }
    
    func stopCapture() {
        sessionQueue.async { [weak self] in
            self?.captureSession.stopRunning()
            
            // Clean up inputs and outputs to allow restart
            self?.captureSession.beginConfiguration()
            
            // Remove all inputs
            if let inputs = self?.captureSession.inputs {
                for input in inputs {
                    self?.captureSession.removeInput(input)
                }
            }
            
            // Remove all outputs
            if let outputs = self?.captureSession.outputs {
                for output in outputs {
                    self?.captureSession.removeOutput(output)
                }
            }
            
            self?.captureSession.commitConfiguration()
            
            DispatchQueue.main.async {
                self?.isRunning = false
            }
        }
    }
    
    private func setupCaptureSession() {
        captureSession.beginConfiguration()
        
        // Configure session preset for balanced quality/performance
        captureSession.sessionPreset = .high
        
        // Setup back camera input
        do {
            guard let backCamera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else {
                throw CameraError.cameraUnavailable
            }
            
            let cameraInput = try AVCaptureDeviceInput(device: backCamera)
            
            if captureSession.canAddInput(cameraInput) {
                captureSession.addInput(cameraInput)
            } else {
                throw CameraError.cannotAddInput
            }
            
        } catch {
            delegate?.cameraManager(self, didFailWithError: error)
            captureSession.commitConfiguration()
            return
        }
        
        // Configure video output
        videoOutput.setSampleBufferDelegate(self, queue: videoOutputQueue)
        videoOutput.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        
        // Drop frames if processing is too slow to maintain real-time performance
        videoOutput.alwaysDiscardsLateVideoFrames = true
        
        if captureSession.canAddOutput(videoOutput) {
            captureSession.addOutput(videoOutput)
        } else {
            delegate?.cameraManager(self, didFailWithError: CameraError.cannotAddOutput)
            captureSession.commitConfiguration()
            return
        }
        
        // Setup photo output for high-quality single captures
        setupPhotoOutput()
        
        // Configure video orientation
        if let connection = videoOutput.connection(with: .video) {
            if connection.isVideoOrientationSupported {
                if #available(iOS 17.0, *) {
                    connection.videoRotationAngle = 0 // Portrait orientation
                } else {
                    connection.videoOrientation = .portrait
                }
            }
            
            // Enable video stabilization if available
            if connection.isVideoStabilizationSupported {
                connection.preferredVideoStabilizationMode = .auto
            }
        }
        
        captureSession.commitConfiguration()
    }
    
    private func setupPhotoOutput() {
        // Configure photo output for high-quality captures
        // Note: maxPhotoDimensions must be set after the session is running and device is connected
        if #available(iOS 16.0, *) {
            // maxPhotoDimensions will be set in captureHighQualityImage() after session is running
        } else {
            // Fallback to deprecated API for older versions
            photoOutput.isHighResolutionCaptureEnabled = true
        }
        
        if captureSession.canAddOutput(photoOutput) {
            captureSession.addOutput(photoOutput)
        } else {
            print("⚠️ Cannot add photo output to capture session")
            // Non-fatal error - video capture will still work
        }
    }
    
    // MARK: - Single Image Capture
    
    /// Enables single capture mode and captures high-quality image
    func captureHighQualityImage() {
        guard hasPermission else {
            delegate?.cameraManager(self, didFailWithError: CameraError.permissionDenied)
            return
        }
        
        guard !capturingPhoto else {
            print("📸 Already capturing photo, ignoring request")
            return
        }
        
        singleCaptureMode = true
        capturingPhoto = true
        
        // Configure maxPhotoDimensions now that session is running and device is connected
        if #available(iOS 16.0, *) {
            // Set maxPhotoDimensions on the output now that device is connected
            let maxDimensions = CMVideoDimensions(width: 4000, height: 3000)
            photoOutput.maxPhotoDimensions = maxDimensions
        }
        
        let settings: AVCapturePhotoSettings
        
        // Use HEIF format for better quality if available, fallback to JPEG
        if photoOutput.availablePhotoCodecTypes.contains(.hevc) {
            settings = AVCapturePhotoSettings(format: [AVVideoCodecKey: AVVideoCodecType.hevc])
        } else {
            settings = AVCapturePhotoSettings(format: [AVVideoCodecKey: AVVideoCodecType.jpeg])
        }
        
        if #available(iOS 16.0, *) {
            // Use newer API for iOS 16+ - set on settings as well
            let maxDimensions = CMVideoDimensions(width: 4000, height: 3000)
            settings.maxPhotoDimensions = maxDimensions
        } else {
            // Fallback to deprecated API for older versions
            settings.isHighResolutionPhotoEnabled = true
        }
        
        settings.photoQualityPrioritization = .quality
        
        sessionQueue.async { [weak self] in
            guard let self = self else { return }
            
            // Ensure photo output is available
            if self.captureSession.outputs.contains(self.photoOutput) {
                self.photoOutput.capturePhoto(with: settings, delegate: self)
                print("📸 High-quality image capture initiated")
            } else {
                print("❌ Photo output not available")
                self.capturingPhoto = false
                self.singleCaptureMode = false
            }
        }
    }
    
    /// Disables single capture mode and returns to continuous frame processing
    func disableSingleCaptureMode() {
        singleCaptureMode = false
        capturingPhoto = false
        print("🔄 Returned to continuous frame processing mode")
    }
}

// MARK: - AVCaptureVideoDataOutputSampleBufferDelegate
extension ARCameraManager: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return
        }
        
        // Only process continuous frames if not in single capture mode
        if !singleCaptureMode {
            // Pass pixel buffer to delegate for processing
            delegate?.cameraManager(self, didOutput: pixelBuffer)
        }
    }
    
    func captureOutput(_ output: AVCaptureOutput, didDrop sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        // Handle dropped frames if needed for performance monitoring
        print("🎥 Dropped video frame")
    }
}

// MARK: - AVCapturePhotoCaptureDelegate
extension ARCameraManager: AVCapturePhotoCaptureDelegate {
    func photoOutput(_ output: AVCapturePhotoOutput, didFinishProcessingPhoto photo: AVCapturePhoto, error: Error?) {
        guard error == nil else {
            print("❌ Photo capture failed: \(error!.localizedDescription)")
            capturingPhoto = false
            singleCaptureMode = false
            delegate?.cameraManager(self, didFailWithError: error!)
            return
        }
        
        guard let imageData = photo.fileDataRepresentation() else {
            print("❌ Failed to get photo data")
            capturingPhoto = false
            singleCaptureMode = false
            delegate?.cameraManager(self, didFailWithError: CameraError.cannotCreateImage)
            return
        }
        
        guard let image = UIImage(data: imageData) else {
            print("❌ Failed to create UIImage from photo data")
            capturingPhoto = false
            singleCaptureMode = false
            delegate?.cameraManager(self, didFailWithError: CameraError.cannotCreateImage)
            return
        }
        
        // Convert UIImage to CVPixelBuffer for segmentation processing
        guard let pixelBuffer = convertUIImageToPixelBuffer(image) else {
            print("❌ Failed to convert UIImage to pixel buffer")
            capturingPhoto = false
            singleCaptureMode = false
            delegate?.cameraManager(self, didFailWithError: CameraError.cannotCreatePixelBuffer)
            return
        }
        
        capturingPhoto = false
        // Keep single capture mode enabled until explicitly disabled
        
        print("✅ High-quality photo captured: \(image.size.width)x\(image.size.height)")
        
        // Notify delegate with both UIImage and pixel buffer
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            self.delegate?.cameraManager(self, didCaptureImage: image, pixelBuffer: pixelBuffer)
        }
    }
    
    private func convertUIImageToPixelBuffer(_ image: UIImage) -> CVPixelBuffer? {
        guard let cgImage = image.cgImage else { return nil }
        
        let width = Int(cgImage.width)
        let height = Int(cgImage.height)
        
        let attributes: [String: Any] = [
            kCVPixelBufferCGImageCompatibilityKey as String: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey as String: true,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:]
        ]
        
        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            width,
            height,
            kCVPixelFormatType_32BGRA,
            attributes as CFDictionary,
            &pixelBuffer
        )
        
        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
            return nil
        }
        
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        
        guard let context = CGContext(
            data: CVPixelBufferGetBaseAddress(buffer),
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue
        ) else {
            return nil
        }
        
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        return buffer
    }
}

// MARK: - Camera Errors
enum CameraError: LocalizedError {
    case permissionDenied
    case cameraUnavailable
    case cannotAddInput
    case cannotAddOutput
    case cannotCreateImage
    case cannotCreatePixelBuffer
    
    var errorDescription: String? {
        switch self {
        case .permissionDenied:
            return "Camera permission denied. Please enable camera access in Settings."
        case .cameraUnavailable:
            return "Camera is not available on this device."
        case .cannotAddInput:
            return "Cannot add camera input to capture session."
        case .cannotAddOutput:
            return "Cannot add video output to capture session."
        case .cannotCreateImage:
            return "Failed to create image from captured photo."
        case .cannotCreatePixelBuffer:
            return "Failed to create pixel buffer from image."
        }
    }
}