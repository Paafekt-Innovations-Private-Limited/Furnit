// FurnitureFitARSupport.swift
// ARKit helpers: copy ARFrame to BGRA for YOLO, estimate distance, derive overlay scale vs standard furniture height.

import ARKit
import CoreGraphics
import CoreVideo
import simd
import UIKit

/// Depth + intrinsics captured on the **same** `ARFrame` as `copyCapturedImageToBGRA`, so ML and AR
/// sampling stay time-aligned (avoid using `arSession.currentFrame` after async inference).
struct FurnitureFitARDepthSnapshot {
    let depthMap: CVPixelBuffer?
    /// Focal length in **pixels** along BGRA **horizontal** (for width in pinhole formula).
    let focalLengthX: Float
    /// Focal length in **pixels** along BGRA **vertical** (for height in pinhole formula).
    let focalLengthY: Float
    let capturedImageWidth: Int
    let capturedImageHeight: Int
    let bgraWidth: Int
    let bgraHeight: Int
    /// True when BGRA was produced with `CIImage.oriented(.right)` from a landscape `capturedImage` (see `copyCapturedImageToBGRA`).
    let bgraIsRotatedFromCaptured: Bool
}

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

    /// Raycast against any detected/estimated plane (horizontal + vertical).
    /// Returns `(distance, alignment)` for the closest hit, or nil.
    @available(iOS 14.0, *)
    static func distanceToAnyPlaneMeters(
        session: ARSession,
        frame: ARFrame,
        screenPoint: CGPoint,
        in viewBounds: CGRect
    ) -> (distance: Float, alignment: ARRaycastQuery.TargetAlignment)? {
        guard viewBounds.width > 1, viewBounds.height > 1 else { return nil }
        let query = frame.raycastQuery(
            from: screenPoint,
            allowing: .estimatedPlane,
            alignment: .any
        )
        let results = session.raycast(query)
        guard let first = results.first else { return nil }
        let camT = frame.camera.transform
        let camPos = SIMD3<Float>(camT.columns.3.x, camT.columns.3.y, camT.columns.3.z)
        let hitT = first.worldTransform
        let hitPos = SIMD3<Float>(hitT.columns.3.x, hitT.columns.3.y, hitT.columns.3.z)
        let dist = simd_distance(camPos, hitPos)
        return (dist, first.targetAlignment)
    }

    /// Sample scene depth (meters) at normalized coords (0…1, top-left) in **depth map** space.
    static func depthMetersFromDepthMap(
        _ depthMap: CVPixelBuffer,
        normalizedDepthPoint: CGPoint
    ) -> Float? {
        let dmW = CVPixelBufferGetWidth(depthMap)
        let dmH = CVPixelBufferGetHeight(depthMap)
        guard dmW > 0, dmH > 0 else { return nil }

        let u = Float(min(max(normalizedDepthPoint.x, 0), 1))
        let v = Float(min(max(normalizedDepthPoint.y, 0), 1))
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
            // Ignore depths under 10 cm — glass / reflective surfaces often report bogus near depth.
            return d.isFinite && d > 0.1 && d < 50 ? d : nil
        }
        return nil
    }

    /// Sample scene depth (meters) at normalized image coords (0…1, top-left). LiDAR / depth-capable devices only.
    static func sceneDepthMeters(
        frame: ARFrame,
        normalizedImagePoint: CGPoint
    ) -> Float? {
        guard let depthData = frame.sceneDepth ?? frame.smoothedSceneDepth else { return nil }
        return depthMetersFromDepthMap(depthData.depthMap, normalizedDepthPoint: normalizedImagePoint)
    }

    /// Copy depth map bytes so they stay valid after the `ARFrame` delegate returns.
    private static func copyDepthMapPixelBuffer(_ src: CVPixelBuffer) -> CVPixelBuffer? {
        CVPixelBufferLockBaseAddress(src, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(src, .readOnly) }
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
        CVPixelBufferLockBaseAddress(out, [])
        defer { CVPixelBufferUnlockBaseAddress(out, []) }
        guard let sb = CVPixelBufferGetBaseAddress(src), let db = CVPixelBufferGetBaseAddress(out) else { return nil }
        let sbRow = CVPixelBufferGetBytesPerRow(src)
        let dbRow = CVPixelBufferGetBytesPerRow(out)
        let rowCopy = min(sbRow, dbRow)
        for y in 0..<h {
            memcpy(db.advanced(by: y * dbRow), sb.advanced(by: y * sbRow), rowCopy)
        }
        return out
    }

    /// Focal length in **pixels** along the **vertical** axis of the processed BGRA (matches `bboxHeightImagePx`).
    /// Uses `camera.intrinsics` + `imageResolution` — `intrinsicsForImageResolution` is not available on all SDKs.
    static func focalLengthYForProcessedBGRA(
        camera: ARCamera,
        bgraHeight: Int,
        bgraIsRotatedFromCaptured: Bool
    ) -> Float {
        let ref = camera.imageResolution
        let K = camera.intrinsics
        let fx = K.columns.0.x
        let fy = K.columns.1.y
        guard ref.width > 0, ref.height > 0 else { return fy }

        if bgraIsRotatedFromCaptured {
            // Vertical axis in BGRA corresponds to horizontal axis in captured image (see `copyCapturedImageToBGRA` + `.oriented(.right)`).
            return fx * Float(bgraHeight) / Float(ref.width)
        }
        return fy * Float(bgraHeight) / Float(ref.height)
    }

    /// Focal length in **pixels** along BGRA **horizontal** (width axis of the segmentation buffer).
    static func focalLengthXForProcessedBGRA(
        camera: ARCamera,
        bgraWidth: Int,
        bgraIsRotatedFromCaptured: Bool
    ) -> Float {
        let ref = camera.imageResolution
        let K = camera.intrinsics
        let fx = K.columns.0.x
        let fy = K.columns.1.y
        guard ref.width > 0, ref.height > 0 else { return fx }

        if bgraIsRotatedFromCaptured {
            // Horizontal axis in BGRA corresponds to vertical axis in captured image.
            return fy * Float(bgraWidth) / Float(ref.height)
        }
        return fx * Float(bgraWidth) / Float(ref.width)
    }

    /// Build metrics from an `ARFrame` aligned with the **segmentation** buffer: either the same frame as `copyCapturedImageToBGRA`,
    /// or (hybrid path) `arSession.currentFrame` sampled beside an `AVCapture` pixel buffer of size `bgraWidth`×`bgraHeight`.
    static func makeDepthSnapshot(
        frame: ARFrame,
        bgraWidth: Int,
        bgraHeight: Int,
        lockedOrientation: PhotoOrientation
    ) -> FurnitureFitARDepthSnapshot {
        let src = frame.capturedImage
        let srcW = CVPixelBufferGetWidth(src)
        let srcH = CVPixelBufferGetHeight(src)
        let isLandscapeBuffer = srcW > srcH
        let needsPortrait = (lockedOrientation == .portrait || lockedOrientation == .square)
        let bgraIsRotatedFromCaptured = needsPortrait && isLandscapeBuffer

        let focalY = focalLengthYForProcessedBGRA(
            camera: frame.camera,
            bgraHeight: bgraHeight,
            bgraIsRotatedFromCaptured: bgraIsRotatedFromCaptured
        )
        let focalX = focalLengthXForProcessedBGRA(
            camera: frame.camera,
            bgraWidth: bgraWidth,
            bgraIsRotatedFromCaptured: bgraIsRotatedFromCaptured
        )

        let depthCopy: CVPixelBuffer? = {
            guard let depthData = frame.sceneDepth ?? frame.smoothedSceneDepth else { return nil }
            return copyDepthMapPixelBuffer(depthData.depthMap)
        }()

        return FurnitureFitARDepthSnapshot(
            depthMap: depthCopy,
            focalLengthX: focalX,
            focalLengthY: focalY,
            capturedImageWidth: srcW,
            capturedImageHeight: srcH,
            bgraWidth: bgraWidth,
            bgraHeight: bgraHeight,
            bgraIsRotatedFromCaptured: bgraIsRotatedFromCaptured
        )
    }

    /// Scene depth at bbox center, using **BGRA-normalized** coords (same space as YOLO / `processBuffer`).
    static func depthMeters(
        snapshot: FurnitureFitARDepthSnapshot,
        normalizedBgraNX: CGFloat,
        normalizedBgraNY: CGFloat
    ) -> Float? {
        guard let depthMap = snapshot.depthMap else { return nil }
        let nx = CGFloat(min(max(normalizedBgraNX, 0), 1))
        let ny = CGFloat(min(max(normalizedBgraNY, 0), 1))
        let srcW = snapshot.capturedImageWidth
        let srcH = snapshot.capturedImageHeight

        let uw: CGFloat
        let uh: CGFloat
        if snapshot.bgraIsRotatedFromCaptured {
            let bw = CGFloat(snapshot.bgraWidth)
            let bh = CGFloat(snapshot.bgraHeight)
            let px_b = nx * (bw - 1)
            let py_b = ny * (bh - 1)
            // Inverse of CIImage.oriented(.right): output (x,y) → captured (x_in, y_in).
            let x_in = py_b
            let y_in = CGFloat(max(0, srcH - 1)) - px_b
            uw = x_in / CGFloat(max(srcW - 1, 1))
            uh = y_in / CGFloat(max(srcH - 1, 1))
        } else {
            uw = nx
            uh = ny
        }
        return depthMetersFromDepthMap(depthMap, normalizedDepthPoint: CGPoint(x: uw, y: uh))
    }

    /// Minimum scene depth (m) over a grid in **BGRA-normalized** coords (origin top-left), preferring the **nearest**
    /// surface in the bbox. Reduces inflated sizes when a loose YOLO box spans object + far wall and the center hits the wall.
    static func depthMetersMinInNormalizedBgraRect(
        snapshot: FurnitureFitARDepthSnapshot,
        nxMin: CGFloat,
        nyMinTop: CGFloat,
        nxMax: CGFloat,
        nyMaxTop: CGFloat,
        samplesPerAxis: Int = 5
    ) -> Float? {
        guard snapshot.depthMap != nil else { return nil }
        guard samplesPerAxis >= 1 else { return nil }
        let x0 = min(max(min(nxMin, nxMax), 0), 1)
        let x1 = min(max(max(nxMin, nxMax), 0), 1)
        let y0 = min(max(min(nyMinTop, nyMaxTop), 0), 1)
        let y1 = min(max(max(nyMinTop, nyMaxTop), 0), 1)
        var best: Float?
        for iy in 0..<samplesPerAxis {
            for ix in 0..<samplesPerAxis {
                let tx = (CGFloat(ix) + 0.5) / CGFloat(samplesPerAxis)
                let ty = (CGFloat(iy) + 0.5) / CGFloat(samplesPerAxis)
                let nx = x0 + (x1 - x0) * tx
                let ny = y0 + (y1 - y0) * ty
                if let d = depthMeters(snapshot: snapshot, normalizedBgraNX: nx, normalizedBgraNY: ny),
                   d > 0.1, d < 50, d.isFinite {
                    if best == nil || d < best! { best = d }
                }
            }
        }
        return best
    }

    /// Robust scene depth over a BGRA-normalized rect. Uses a low percentile instead of the strict minimum
    /// so a single spurious near pixel does not inflate measured object size.
    static func depthMetersPercentileInNormalizedBgraRect(
        snapshot: FurnitureFitARDepthSnapshot,
        nxMin: CGFloat,
        nyMinTop: CGFloat,
        nxMax: CGFloat,
        nyMaxTop: CGFloat,
        samplesPerAxis: Int = 7,
        percentile: Float = 0.20
    ) -> Float? {
        guard snapshot.depthMap != nil else { return nil }
        guard samplesPerAxis >= 1 else { return nil }
        let x0 = min(max(min(nxMin, nxMax), 0), 1)
        let x1 = min(max(max(nxMin, nxMax), 0), 1)
        let y0 = min(max(min(nyMinTop, nyMaxTop), 0), 1)
        let y1 = min(max(max(nyMinTop, nyMaxTop), 0), 1)

        var samples: [Float] = []
        samples.reserveCapacity(samplesPerAxis * samplesPerAxis)
        for iy in 0..<samplesPerAxis {
            for ix in 0..<samplesPerAxis {
                let tx = (CGFloat(ix) + 0.5) / CGFloat(samplesPerAxis)
                let ty = (CGFloat(iy) + 0.5) / CGFloat(samplesPerAxis)
                let nx = x0 + (x1 - x0) * tx
                let ny = y0 + (y1 - y0) * ty
                if let d = depthMeters(snapshot: snapshot, normalizedBgraNX: nx, normalizedBgraNY: ny),
                   d > 0.1, d < 50, d.isFinite {
                    samples.append(d)
                }
            }
        }

        guard !samples.isEmpty else { return nil }
        samples.sort()
        let p = min(max(percentile, 0), 1)
        let idx = min(samples.count - 1, max(0, Int(roundf(Float(samples.count - 1) * p))))
        return samples[idx]
    }

    /// Pinhole: physical height (m) from bbox height in **full image pixels** and distance in meters.
    static func estimatedPhysicalHeightMeters(
        bboxHeightPixels: Float,
        distanceMeters: Float,
        focalLengthYPixels: Float
    ) -> Float? {
        guard bboxHeightPixels > 1, distanceMeters > 0.1, focalLengthYPixels > 1 else { return nil }
        return (bboxHeightPixels / focalLengthYPixels) * distanceMeters
    }

    /// Pinhole: physical width (m) from bbox width in **full image pixels** and distance in meters.
    static func estimatedPhysicalWidthMeters(
        bboxWidthPixels: Float,
        distanceMeters: Float,
        focalLengthXPixels: Float
    ) -> Float? {
        guard bboxWidthPixels > 1, distanceMeters > 0.1, focalLengthXPixels > 1 else { return nil }
        return (bboxWidthPixels / focalLengthXPixels) * distanceMeters
    }

    /// Target overlay scale vs 1.0: AR-estimated height relative to standard catalog height, clamped.
    static func overlayScaleFromMetricHeights(
        standardHeightMeters: Float,
        estimatedHeightMeters: Float,
        minScale: Float = 0.08,
        maxScale: Float = 2.5
    ) -> Float? {
        guard standardHeightMeters > 0.1, estimatedHeightMeters > 0.1 else { return nil }
        // When the FurnitureFit overlay has already been scaled so that the standard height appears
        // correct for the current room, this additional factor adjusts it so that the final visual
        // height matches the AR-estimated height. If AR says the furniture is taller than the
        // standard, we scale up (>1); if shorter, we scale down (<1).
        let raw = estimatedHeightMeters / standardHeightMeters
        return min(max(raw, minScale), maxScale)
    }
}
