import ARKit
import CoreVideo
import UIKit
import SwiftUI

// ARKit-based camera manager for furniture detection
// Uses ARSession to provide continuous camera frames without continuation leaks
class ARKitCameraManager: NSObject, ObservableObject {
    // Published properties for UI updates
    @Published var isSessionRunning = false
    @Published var capturedImage: UIImage?
    @Published var errorMessage: String?
    @Published var currentFrameImage: UIImage?
    
    // ARKit session and configuration
    private let arSession = ARSession()
    private let arSessionDelegateQueue = DispatchQueue(label: "com.furnit.arkit-camera-manager", qos: .userInitiated)
    private var arConfiguration = ARWorldTrackingConfiguration()
    
    /// Standalone copy of `ARFrame.capturedImage` — **never** store `ARFrame`; retaining it past `didUpdate` exhausts ARKit’s frame pool.
    private var latestCapturedImageBuffer: CVPixelBuffer?
    private var lastCapturedImageCopyTime: CFAbsoluteTime = 0
    /// Avoid doing a full camera buffer copy on every ARKit frame (still well under typical camera FPS).
    private let capturedImageCopyMinInterval: CFTimeInterval = 0.1
    
    override init() {
        super.init()
        logDebug("🎯 ARKitCameraManager initialized")
        setupARSession()
    }
    
    // MARK: - ARSession Setup
    
    // Configure ARKit session for camera frame access
    private func setupARSession() {
        // Set delegate to receive frame updates
        arSession.delegate = self
        arSession.delegateQueue = arSessionDelegateQueue
        
        // Configure for world tracking (provides camera access)
        arConfiguration = ARWorldTrackingConfiguration()
        
        // Optimize for camera frame capture rather than world tracking
        arConfiguration.worldAlignment = .gravity
        
        // Enable auto focus for sharp furniture images
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.personSegmentation) {
            // Use person segmentation if available for better object detection
            arConfiguration.frameSemantics = .personSegmentation
        }
        
        logDebug("🎯 ARKit session configured for camera frame capture")
        CameraOwnershipDiagnostics.log(owner: "ARKitCameraManager", event: "configured")
    }
    
    // MARK: - Session Control
    
    // Start ARKit session for camera frame access
    @MainActor
    func startSession() {
        logDebug("🎯 Starting ARKit camera session...")
        
        // Check ARKit availability and camera permission
        guard ARWorldTrackingConfiguration.isSupported else {
            errorMessage = "ARKit World Tracking not supported on this device"
            logDebug("⚠️ ARKit World Tracking not supported")
            return
        }
        
        // Run the AR session on main queue (ARKit requirement)
        CameraOwnershipDiagnostics.log(owner: "ARKitCameraManager", event: "ar_run", details: "reason=startSession")
        arSession.run(arConfiguration, options: [.resetTracking, .removeExistingAnchors])
        
        // Update session status
        isSessionRunning = true
        errorMessage = nil
        
        logDebug("✅ ARKit camera session started successfully")
    }
    
    // Stop ARKit session when exiting AR mode
    @MainActor
    func stopSession() {
        logDebug("🎯 Stopping ARKit camera session...")
        
        // Pause the AR session
        CameraOwnershipDiagnostics.log(owner: "ARKitCameraManager", event: "ar_pause", details: "reason=stopSession")
        arSession.pause()
        
        latestCapturedImageBuffer = nil
        lastCapturedImageCopyTime = 0
        capturedImage = nil
        currentFrameImage = nil
        
        // Update session status
        isSessionRunning = false
        
        logDebug("✅ ARKit camera session stopped")
    }

    deinit {
        CameraOwnershipDiagnostics.log(owner: "ARKitCameraManager", event: "deinit")
    }
    
    // MARK: - Frame Capture
    
    // Capture current camera frame for backend API (full resolution)
    @MainActor
    func captureCurrentFrameForAPI() -> UIImage? {
        guard let buffer = latestCapturedImageBuffer else {
            errorMessage = "No camera frame available - ensure AR session is running"
            logDebug("⚠️ No current frame available for capture")
            return nil
        }
        
        logDebug("📷 Capturing current ARKit camera frame for 3D generation...")
        
        guard let rawUIImage = convertPixelBufferToUIImage(buffer) else {
            errorMessage = "Failed to convert camera frame to image"
            logDebug("⚠️ Failed to convert ARFrame to UIImage")
            return nil
        }
        
        // Fix orientation without resizing for backend API
        guard let orientationFixedImage = fixImageOrientation(rawUIImage) else {
            errorMessage = "Failed to fix image orientation"
            logDebug("⚠️ Failed to fix image orientation")
            return nil
        }
        
        // Store captured image
        capturedImage = orientationFixedImage
        
        logDebug("✅ ARKit frame captured successfully for 3D generation")
        logDebug("   Image size: \(orientationFixedImage.size)")
        logDebug("   Image orientation: \(orientationFixedImage.imageOrientation.rawValue)")
        return orientationFixedImage
    }
    
    // Capture current camera frame for furniture segmentation (legacy for DeepLabV3)
    // This is synchronous and doesn't use continuations - no more leaks!
    @MainActor
    func captureCurrentFrame() -> UIImage? {
        guard let buffer = latestCapturedImageBuffer else {
            errorMessage = "No camera frame available - ensure AR session is running"
            logDebug("⚠️ No current frame available for capture")
            return nil
        }
        
        logDebug("📷 Capturing current ARKit camera frame for segmentation...")
        
        guard let rawUIImage = convertPixelBufferToUIImage(buffer) else {
            errorMessage = "Failed to convert camera frame to image"
            logDebug("⚠️ Failed to convert ARFrame to UIImage")
            return nil
        }
        
        // Preprocess image for DeepLabV3 (resize to 513x513, fix orientation)
        guard let preprocessedImage = preprocessImageForSegmentation(rawUIImage) else {
            errorMessage = "Failed to preprocess image for segmentation"
            logDebug("⚠️ Failed to preprocess image for DeepLabV3")
            return nil
        }
        
        // Store processed image
        capturedImage = preprocessedImage
        
        logDebug("✅ ARKit frame captured and preprocessed successfully")
        logDebug("   Original size: \(rawUIImage.size)")
        logDebug("   Processed size: \(preprocessedImage.size)")
        logDebug("   Processed orientation: \(preprocessedImage.imageOrientation.rawValue)")
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
            logDebug("⚠️ Failed to create CGImage from pixel buffer")
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
            logDebug("⚠️ Failed to create orientation-fixed image from graphics context")
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
            logDebug("⚠️ Failed to create processed image from graphics context")
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

    @MainActor
    func captureNextFrame() async -> UIImage? {
        if let buffer = latestCapturedImageBuffer {
            return convertPixelBufferToUIImage(buffer)
        }
        for _ in 0..<2 {
            try? await Task.sleep(nanoseconds: 50_000_000)
            if let buffer = latestCapturedImageBuffer {
                return convertPixelBufferToUIImage(buffer)
            }
        }
        return nil
    }

    /// Byte-copy into an owned buffer so `didUpdate` can return without retaining `ARFrame` (handles multi-plane YUV camera buffers).
    private static func standalonePixelBufferCopy(of src: CVPixelBuffer) -> CVPixelBuffer? {
        let w = CVPixelBufferGetWidth(src)
        let h = CVPixelBufferGetHeight(src)
        let fmt = CVPixelBufferGetPixelFormatType(src)
        var dst: CVPixelBuffer?
        let attrs: [CFString: Any] = [
            kCVPixelBufferPixelFormatTypeKey: fmt,
            kCVPixelBufferWidthKey: w,
            kCVPixelBufferHeightKey: h,
            kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary,
        ]
        guard CVPixelBufferCreate(kCFAllocatorDefault, w, h, fmt, attrs as CFDictionary, &dst) == kCVReturnSuccess,
              let out = dst else { return nil }

        CVPixelBufferLockBaseAddress(src, .readOnly)
        CVPixelBufferLockBaseAddress(out, [])
        defer {
            CVPixelBufferUnlockBaseAddress(out, [])
            CVPixelBufferUnlockBaseAddress(src, .readOnly)
        }

        let planeCount = CVPixelBufferGetPlaneCount(src)
        if planeCount == 0 {
            guard let sb = CVPixelBufferGetBaseAddress(src), let db = CVPixelBufferGetBaseAddress(out) else { return nil }
            let sbRow = CVPixelBufferGetBytesPerRow(src)
            let dbRow = CVPixelBufferGetBytesPerRow(out)
            let rowCopy = min(sbRow, dbRow)
            for y in 0..<h {
                memcpy(db.advanced(by: y * dbRow), sb.advanced(by: y * sbRow), rowCopy)
            }
        } else {
            for plane in 0..<planeCount {
                guard let sb = CVPixelBufferGetBaseAddressOfPlane(src, plane),
                      let db = CVPixelBufferGetBaseAddressOfPlane(out, plane) else { return nil }
                let ph = CVPixelBufferGetHeightOfPlane(src, plane)
                let sbRow = CVPixelBufferGetBytesPerRowOfPlane(src, plane)
                let dbRow = CVPixelBufferGetBytesPerRowOfPlane(out, plane)
                let rowCopy = min(sbRow, dbRow)
                for y in 0..<ph {
                    memcpy(db.advanced(by: y * dbRow), sb.advanced(by: y * sbRow), rowCopy)
                }
            }
        }
        return out
    }
}

// MARK: - ARSessionDelegate

extension ARKitCameraManager: ARSessionDelegate {
    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        let now = CFAbsoluteTimeGetCurrent()
        if now - lastCapturedImageCopyTime < capturedImageCopyMinInterval { return }
        lastCapturedImageCopyTime = now
        let src = frame.capturedImage
        guard let copy = Self.standalonePixelBufferCopy(of: src) else { return }
        latestCapturedImageBuffer = copy
    }
    
    // Handle session errors
    func session(_ session: ARSession, didFailWithError error: Error) {
        errorMessage = "ARKit session failed: \(error.localizedDescription)"
        logDebug("⚠️ ARKit session failed: \(error)")
    }
    
    // Handle session interruptions
    func sessionWasInterrupted(_ session: ARSession) {
        logDebug("🎯 ARKit session was interrupted")
        // Session will restart automatically when interruption ends
    }
    
    // Handle session interruption end
    func sessionInterruptionEnded(_ session: ARSession) {
        logDebug("🎯 ARKit session interruption ended")
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
