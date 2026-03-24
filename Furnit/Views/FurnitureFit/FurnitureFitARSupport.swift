// FurnitureFitARSupport.swift
// ARKit helpers: copy ARFrame to BGRA for YOLO, estimate distance, derive overlay scale vs standard furniture height.

import ARKit
import CoreGraphics
import CoreVideo
import simd
import UIKit

enum FurnitureFitARSupport {

    /// AR world tracking supported on this device.
    static var isWorldTrackingSupported: Bool {
        ARWorldTrackingConfiguration.isSupported
    }

    /// Copy `ARFrame.capturedImage` into a reusable BGRA buffer (ARKit buffers are only valid during delegate callbacks).
    ///
    /// For FurnitureFit, YOLO should see the **same orientation invariants** as the classic AVCapture path:
    /// - Landscape rooms: landscape buffers.
    /// - Portrait rooms: portrait buffers (width < height), effectively a 90° rotation vs sensor-native.
    /// This helper rotates the AR buffer to portrait only when the locked room orientation is portrait.
    static func copyCapturedImageToBGRA(
        frame: ARFrame,
        reuse: inout CVPixelBuffer?,
        ciContext: CIContext,
        lockedOrientation: PhotoOrientation
    ) -> CVPixelBuffer? {
        let src = frame.capturedImage
        let srcW = CVPixelBufferGetWidth(src)
        let srcH = CVPixelBufferGetHeight(src)
        if srcW <= 0 || srcH <= 0 { return nil }

        var ciImage = CIImage(cvPixelBuffer: src)
        let isLandscapeBuffer = srcW > srcH
        let needsPortrait = (lockedOrientation == .portrait || lockedOrientation == .square)
        if needsPortrait && isLandscapeBuffer {
            // Rotate camera buffer 90° so YOLO sees a portrait frame, matching AVCapture's videoRotationAngle = 90.
            ciImage = ciImage.oriented(.right)
        }

        let extent = ciImage.extent.integral
        let outW = Int(extent.width.rounded())
        let outH = Int(extent.height.rounded())
        if outW <= 0 || outH <= 0 { return nil }

        if reuse == nil
            || CVPixelBufferGetWidth(reuse!) != outW
            || CVPixelBufferGetHeight(reuse!) != outH
            || CVPixelBufferGetPixelFormatType(reuse!) != kCVPixelFormatType_32BGRA {
            reuse = nil
            var newBuf: CVPixelBuffer?
            let attrs: [CFString: Any] = [
                kCVPixelBufferCGImageCompatibilityKey: true,
                kCVPixelBufferCGBitmapContextCompatibilityKey: true,
            ]
            guard CVPixelBufferCreate(
                kCFAllocatorDefault,
                outW,
                outH,
                kCVPixelFormatType_32BGRA,
                attrs as CFDictionary,
                &newBuf
            ) == kCVReturnSuccess,
                let buf = newBuf else {
                return nil
            }
            reuse = buf
        }

        guard let dst = reuse else { return nil }
        ciContext.render(ciImage, to: dst, bounds: extent, colorSpace: CGColorSpaceCreateDeviceRGB())
        return dst
    }

    /// Build AR world-tracking config with scene depth when available (LiDAR).
    static func makeWorldTrackingConfiguration() -> ARWorldTrackingConfiguration {
        let config = ARWorldTrackingConfiguration()
        config.worldAlignment = .gravity
        config.planeDetection = [.horizontal, .vertical]
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) {
            config.frameSemantics.insert(.sceneDepth)
        } else if ARWorldTrackingConfiguration.supportsFrameSemantics(.smoothedSceneDepth) {
            config.frameSemantics.insert(.smoothedSceneDepth)
        }
        return config
    }

    /// Distance from camera to first horizontal plane hit along ray through screen point (meters), or nil.
    @available(iOS 14.0, *)
    static func distanceToHorizontalPlaneMeters(
        session: ARSession,
        frame: ARFrame,
        screenPoint: CGPoint,
        in viewBounds: CGRect
    ) -> Float? {
        guard viewBounds.width > 1, viewBounds.height > 1 else { return nil }
        let query = frame.raycastQuery(
            from: screenPoint,
            allowing: .estimatedPlane,
            alignment: .horizontal
        )
        let results = session.raycast(query)
        guard let first = results.first else { return nil }
        let camT = frame.camera.transform
        let camPos = SIMD3<Float>(camT.columns.3.x, camT.columns.3.y, camT.columns.3.z)
        let hitT = first.worldTransform
        let hitPos = SIMD3<Float>(hitT.columns.3.x, hitT.columns.3.y, hitT.columns.3.z)
        return simd_distance(camPos, hitPos)
    }

    /// Sample scene depth (meters) at normalized image coords (0…1, top-left). LiDAR / depth-capable devices only.
    static func sceneDepthMeters(
        frame: ARFrame,
        normalizedImagePoint: CGPoint
    ) -> Float? {
        guard let depthData = frame.sceneDepth ?? frame.smoothedSceneDepth else { return nil }
        let depthMap = depthData.depthMap
        let dmW = CVPixelBufferGetWidth(depthMap)
        let dmH = CVPixelBufferGetHeight(depthMap)
        guard dmW > 0, dmH > 0 else { return nil }

        let u = Float(min(max(normalizedImagePoint.x, 0), 1))
        let v = Float(min(max(normalizedImagePoint.y, 0), 1))
        let x = min(Int(u * Float(dmW - 1)), dmW - 1)
        let y = min(Int(v * Float(dmH - 1)), dmH - 1)

        CVPixelBufferLockBaseAddress(depthMap, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depthMap, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(depthMap) else { return nil }
        let rowBytes = CVPixelBufferGetBytesPerRow(depthMap)
        let fmt = CVPixelBufferGetPixelFormatType(depthMap)
        if fmt == kCVPixelFormatType_DepthFloat32 || fmt == kCVPixelFormatType_DisparityFloat32 {
            let stride = rowBytes / MemoryLayout<Float>.size
            let ptr = base.assumingMemoryBound(to: Float.self)
            let d = ptr[y * stride + x]
            return d.isFinite && d > 0.05 && d < 50 ? d : nil
        }
        return nil
    }

    /// Pinhole: physical height (m) from bbox height in **full image pixels** and distance in meters.
    static func estimatedPhysicalHeightMeters(
        bboxHeightPixels: Float,
        distanceMeters: Float,
        focalLengthYPixels: Float
    ) -> Float? {
        guard bboxHeightPixels > 1, distanceMeters > 0.05, focalLengthYPixels > 1 else { return nil }
        return (bboxHeightPixels / focalLengthYPixels) * distanceMeters
    }

    /// Target overlay scale vs 1.0: standard catalog height / AR-estimated height, clamped.
    static func overlayScaleFromMetricHeights(
        standardHeightMeters: Float,
        estimatedHeightMeters: Float,
        minScale: Float = 0.4,
        maxScale: Float = 2.5
    ) -> Float? {
        guard standardHeightMeters > 0.1, estimatedHeightMeters > 0.05 else { return nil }
        let raw = standardHeightMeters / estimatedHeightMeters
        return min(max(raw, minScale), maxScale)
    }
}
