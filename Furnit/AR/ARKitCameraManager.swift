import ARKit
import UIKit
import SwiftUI

// ARKit-based camera manager for furniture detection
// Uses ARSession to provide continuous camera frames without continuation leaks
@MainActor
class ARKitCameraManager: NSObject, ObservableObject {
    // Published properties for UI updates
    @Published var isSessionRunning = false
    @Published var capturedImage: UIImage?
    @Published var errorMessage: String?
    @Published var currentFrameImage: UIImage?
    
    // ARKit session and configuration
    private let arSession = ARSession()
    private var arConfiguration = ARWorldTrackingConfiguration()
    
    // Latest frame data for on-demand capture
    private var latestFrame: ARFrame?
    private let frameProcessingQueue = DispatchQueue(label: "arkit.frame.processing", qos: .userInitiated)
    
    override init() {
        super.init()
        print("🎯 ARKitCameraManager initialized")
        setupARSession()
    }
    
    // MARK: - ARSession Setup
    
    // Configure ARKit session for camera frame access
    private func setupARSession() {
        // Set delegate to receive frame updates
        arSession.delegate = self
        
        // Configure for world tracking (provides camera access)
        arConfiguration = ARWorldTrackingConfiguration()
        
        // Optimize for camera frame capture rather than world tracking
        arConfiguration.worldAlignment = .gravity
        
        // Enable auto focus for sharp furniture images
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.personSegmentation) {
            // Use person segmentation if available for better object detection
            arConfiguration.frameSemantics = .personSegmentation
        }
        
        print("🎯 ARKit session configured for camera frame capture")
    }
    
    // MARK: - Session Control
    
    // Start ARKit session for camera frame access
    func startSession() {
        print("🎯 Starting ARKit camera session...")
        
        // Check ARKit availability and camera permission
        guard ARWorldTrackingConfiguration.isSupported else {
            errorMessage = "ARKit World Tracking not supported on this device"
            print("⚠️ ARKit World Tracking not supported")
            return
        }
        
        // Run the AR session on main queue (ARKit requirement)
        arSession.run(arConfiguration, options: [.resetTracking, .removeExistingAnchors])
        
        // Update session status
        isSessionRunning = true
        errorMessage = nil
        
        print("✅ ARKit camera session started successfully")
    }
    
    // Stop ARKit session when exiting AR mode
    func stopSession() {
        print("🎯 Stopping ARKit camera session...")
        
        // Pause the AR session
        arSession.pause()
        
        // Clear frame data
        latestFrame = nil
        capturedImage = nil
        currentFrameImage = nil
        
        // Update session status
        isSessionRunning = false
        
        print("✅ ARKit camera session stopped")
    }
    
    // MARK: - Frame Capture
    
    // Capture current camera frame for backend API (full resolution)
    func captureCurrentFrameForAPI() -> UIImage? {
        guard let currentFrame = latestFrame else {
            errorMessage = "No camera frame available - ensure AR session is running"
            print("⚠️ No current frame available for capture")
            return nil
        }
        
        print("📷 Capturing current ARKit camera frame for 3D generation...")
        
        // Convert ARFrame's camera image to UIImage without resizing
        guard let rawUIImage = convertPixelBufferToUIImage(currentFrame.capturedImage) else {
            errorMessage = "Failed to convert camera frame to image"
            print("⚠️ Failed to convert ARFrame to UIImage")
            return nil
        }
        
        // Fix orientation without resizing for backend API
        guard let orientationFixedImage = fixImageOrientation(rawUIImage) else {
            errorMessage = "Failed to fix image orientation"
            print("⚠️ Failed to fix image orientation")
            return nil
        }
        
        // Store captured image
        capturedImage = orientationFixedImage
        
        print("✅ ARKit frame captured successfully for 3D generation")
        print("   Image size: \(orientationFixedImage.size)")
        print("   Image orientation: \(orientationFixedImage.imageOrientation.rawValue)")
        return orientationFixedImage
    }
    
    // Capture current camera frame for furniture segmentation (legacy for DeepLabV3)
    // This is synchronous and doesn't use continuations - no more leaks!
    func captureCurrentFrame() -> UIImage? {
        guard let currentFrame = latestFrame else {
            errorMessage = "No camera frame available - ensure AR session is running"
            print("⚠️ No current frame available for capture")
            return nil
        }
        
        print("📷 Capturing current ARKit camera frame for segmentation...")
        
        // Convert ARFrame's camera image to UIImage
        guard let rawUIImage = convertPixelBufferToUIImage(currentFrame.capturedImage) else {
            errorMessage = "Failed to convert camera frame to image"
            print("⚠️ Failed to convert ARFrame to UIImage")
            return nil
        }
        
        // Preprocess image for DeepLabV3 (resize to 513x513, fix orientation)
        guard let preprocessedImage = preprocessImageForSegmentation(rawUIImage) else {
            errorMessage = "Failed to preprocess image for segmentation"
            print("⚠️ Failed to preprocess image for DeepLabV3")
            return nil
        }
        
        // Store processed image
        capturedImage = preprocessedImage
        
        print("✅ ARKit frame captured and preprocessed successfully")
        print("   Original size: \(rawUIImage.size)")
        print("   Processed size: \(preprocessedImage.size)")
        print("   Processed orientation: \(preprocessedImage.imageOrientation.rawValue)")
        return preprocessedImage
    }
    
    // Convert CVPixelBuffer from ARFrame to UIImage
    private func convertPixelBufferToUIImage(_ pixelBuffer: CVPixelBuffer) -> UIImage? {
        // Create CIImage from pixel buffer
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        
        // Create context for conversion
        let context = CIContext()
        
        // Convert to CGImage
        guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else {
            print("⚠️ Failed to create CGImage from pixel buffer")
            return nil
        }
        
        // Convert to UIImage with correct orientation
        let uiImage = UIImage(cgImage: cgImage, scale: 1.0, orientation: .right)
        
        return uiImage
    }
    
    // Fix image orientation without resizing (for backend API)
    private func fixImageOrientation(_ image: UIImage) -> UIImage? {
        // If orientation is already correct, return as-is
        if image.imageOrientation == .up {
            return image
        }
        
        // Create graphics context with original size
        UIGraphicsBeginImageContextWithOptions(image.size, false, image.scale)
        defer { UIGraphicsEndImageContext() }
        
        // Draw image in original size (this automatically fixes orientation)
        image.draw(in: CGRect(origin: .zero, size: image.size))
        
        // Get the orientation-fixed image
        guard let fixedImage = UIGraphicsGetImageFromCurrentImageContext() else {
            print("⚠️ Failed to create orientation-fixed image from graphics context")
            return nil
        }
        
        return fixedImage
    }
    
    // Preprocess image for DeepLabV3 segmentation (legacy method)
    // DeepLabV3 requires exactly 513x513 pixels with standard orientation
    private func preprocessImageForSegmentation(_ image: UIImage) -> UIImage? {
        let targetSize = CGSize(width: 513, height: 513)
        
        // Create graphics context with target size
        UIGraphicsBeginImageContextWithOptions(targetSize, false, 1.0)
        defer { UIGraphicsEndImageContext() }
        
        // Draw image in target size (this automatically handles orientation)
        image.draw(in: CGRect(origin: .zero, size: targetSize))
        
        // Get the processed image
        guard let processedImage = UIGraphicsGetImageFromCurrentImageContext() else {
            print("⚠️ Failed to create processed image from graphics context")
            return nil
        }
        
        // Ensure the image has the correct orientation (.up is standard)
        if processedImage.imageOrientation != .up {
            // Fix orientation by redrawing with .up orientation
            UIGraphicsBeginImageContextWithOptions(targetSize, false, 1.0)
            defer { UIGraphicsEndImageContext() }
            
            processedImage.draw(in: CGRect(origin: .zero, size: targetSize))
            return UIGraphicsGetImageFromCurrentImageContext()
        }
        
        return processedImage
    }
    
    // MARK: - Async Frame Capture (Alternative Method)
    
    // Async version that waits for next frame if needed
    func captureNextFrame() async -> UIImage? {
        // If we have a recent frame, use it immediately
        if let currentFrame = latestFrame {
            return convertPixelBufferToUIImage(currentFrame.capturedImage)
        }
        
        // Otherwise wait briefly for next frame
        return await withCheckedContinuation { continuation in
            // Set up one-time frame capture
            var captureCompleted = false
            
            frameProcessingQueue.asyncAfter(deadline: .now() + 0.1) {
                guard !captureCompleted else { return }
                captureCompleted = true
                
                if let frame = self.latestFrame,
                   let image = self.convertPixelBufferToUIImage(frame.capturedImage) {
                    continuation.resume(returning: image)
                } else {
                    continuation.resume(returning: nil)
                }
            }
        }
    }
}

// MARK: - ARSessionDelegate

extension ARKitCameraManager: ARSessionDelegate {
    // Receive camera frames from ARKit session
    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        // Store latest frame for on-demand capture
        latestFrame = frame
        
        // Optionally update preview image for UI (throttled to avoid performance issues)
        if currentFrameImage == nil {
            frameProcessingQueue.async {
                if let previewImage = self.convertPixelBufferToUIImage(frame.capturedImage) {
                    Task { @MainActor in
                        self.currentFrameImage = previewImage
                    }
                }
            }
        }
    }
    
    // Handle session errors
    func session(_ session: ARSession, didFailWithError error: Error) {
        errorMessage = "ARKit session failed: \(error.localizedDescription)"
        print("⚠️ ARKit session failed: \(error)")
    }
    
    // Handle session interruptions
    func sessionWasInterrupted(_ session: ARSession) {
        print("🎯 ARKit session was interrupted")
        // Session will restart automatically when interruption ends
    }
    
    // Handle session interruption end
    func sessionInterruptionEnded(_ session: ARSession) {
        print("🎯 ARKit session interruption ended")
        // Session automatically resumes
    }
}

// MARK: - Camera Permission Helper

extension ARKitCameraManager {
    // Check camera permission status for ARKit
    var cameraPermissionStatus: String {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            return "Camera access granted"
        case .denied:
            return "Camera access denied - please enable in Settings"
        case .restricted:
            return "Camera access restricted"
        case .notDetermined:
            return "Camera permission not determined"
        @unknown default:
            return "Unknown camera permission status"
        }
    }
}