// FurnitureFitAsyncDepthSampler.swift
// ARKit session used only for periodic plane raycast depth — not a camera source (AVCapture owns video).

import ARKit
import simd
import UIKit

// MARK: - Async AR depth sampler (not a camera source)

/// Samples metric distance on a timer using **plane raycast** only — no `sceneDepth` / `smoothedSceneDepth`
/// frame semantics (avoids large depth-buffer allocations). Intended to pair with `AVCaptureSession` for Furniture Fit.
final class FurnitureFitAsyncDepthSampler {
    private let arSession = ARSession()
    private var sampleTimer: Timer?
    private let stateLock = NSLock()

    /// Weak host for mapping normalized image coords → view-space raycast points.
    weak var hostView: UIView?

    private(set) var isRunning = false

    private(set) var lastDepthMeters: Float?
    private(set) var lastFocalLengthY: Float?

    private var pendingNormX: CGFloat = 0
    private var pendingNormY: CGFloat = 0
    private var pendingBboxHeightPx: Float = 0
    private var pendingImageWidth: Int = 0
    private var pendingImageHeight: Int = 0
    private var pendingLockedOrientation: PhotoOrientation = .portrait

    static let sampleInterval: TimeInterval = 1.5

    /// Latest depth + focal length for overlay sizing (thread-safe read).
    func readLastDepthAndFocal() -> (depth: Float, fy: Float)? {
        stateLock.lock()
        let d = lastDepthMeters
        let fy = lastFocalLengthY
        stateLock.unlock()
        guard let d, let fy, d > 0.1, fy > 1 else { return nil }
        return (d, fy)
    }

    /// Called from the segmentation queue whenever the primary bbox updates.
    func updatePending(
        normalizedCenter: CGPoint,
        bboxHeightPx: Float,
        imageWidth: Int,
        imageHeight: Int,
        lockedOrientation: PhotoOrientation
    ) {
        stateLock.lock()
        pendingNormX = normalizedCenter.x
        pendingNormY = normalizedCenter.y
        pendingBboxHeightPx = bboxHeightPx
        pendingImageWidth = imageWidth
        pendingImageHeight = imageHeight
        pendingLockedOrientation = lockedOrientation
        stateLock.unlock()
    }

    func start() {
        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            guard !self.isRunning, FurnitureFitARSupport.isWorldTrackingSupported else { return }
            self.isRunning = true

            let config = ARWorldTrackingConfiguration()
            config.worldAlignment = .gravity
            config.planeDetection = [.horizontal, .vertical]

            CameraOwnershipDiagnostics.log(owner: "FurnitureFitAsyncDepthSampler.ARSession", event: "ar_run", details: "reason=start")
            self.arSession.run(config, options: [.resetTracking, .removeExistingAnchors])

            logFurnitureFitAR(
                "platform=ios event=async_depth_sampler_start interval=\(Self.sampleInterval)s planeDetection=HV no_sceneDepth_semantics"
            )

            self.sampleTimer?.invalidate()
            self.sampleTimer = Timer.scheduledTimer(withTimeInterval: Self.sampleInterval, repeats: true) { [weak self] _ in
                self?.sampleDepth()
            }
            if let t = self.sampleTimer {
                RunLoop.main.add(t, forMode: .common)
            }
            self.sampleDepth()
        }
    }

    func stop() {
        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            self.sampleTimer?.invalidate()
            self.sampleTimer = nil
            CameraOwnershipDiagnostics.log(owner: "FurnitureFitAsyncDepthSampler.ARSession", event: "ar_pause", details: "reason=stop")
            self.arSession.pause()
            self.isRunning = false
            self.stateLock.lock()
            self.lastDepthMeters = nil
            self.lastFocalLengthY = nil
            self.stateLock.unlock()
        }
    }

    deinit {
        CameraOwnershipDiagnostics.log(owner: "FurnitureFitAsyncDepthSampler", event: "deinit")
    }

    private func sampleDepth() {
        stateLock.lock()
        let nx = pendingNormX
        let ny = pendingNormY
        let imgH = pendingImageHeight
        let imgW = pendingImageWidth
        let orient = pendingLockedOrientation
        stateLock.unlock()

        guard imgH > 0, imgW > 0 else { return }
        guard let hostView, hostView.bounds.width > 1, hostView.bounds.height > 1 else { return }
        guard let frame = arSession.currentFrame else { return }
        guard frame.camera.trackingState == .normal else { return }

        let sp = CGPoint(x: nx * hostView.bounds.width, y: ny * hostView.bounds.height)

        var distM = FurnitureFitARSupport.distanceToHorizontalPlaneMeters(
            session: arSession,
            frame: frame,
            screenPoint: sp,
            in: hostView.bounds
        )

        if distM == nil, #available(iOS 14.0, *) {
            let query = frame.raycastQuery(from: sp, allowing: .estimatedPlane, alignment: .vertical)
            let results = arSession.raycast(query)
            if let first = results.first {
                let camT = frame.camera.transform
                let camPos = SIMD3<Float>(camT.columns.3.x, camT.columns.3.y, camT.columns.3.z)
                let hitT = first.worldTransform
                let hitPos = SIMD3<Float>(hitT.columns.3.x, hitT.columns.3.y, hitT.columns.3.z)
                distM = simd_distance(camPos, hitPos)
            }
        }

        guard let distance = distM, distance > 0.1, distance < 50 else { return }

        let cap = frame.capturedImage
        let cw = CVPixelBufferGetWidth(cap)
        let ch = CVPixelBufferGetHeight(cap)
        let needsPortrait = (orient == .portrait || orient == .square)
        let bgraRotated = needsPortrait && cw > ch
        let fy = FurnitureFitARSupport.focalLengthYForProcessedBGRA(
            camera: frame.camera,
            bgraHeight: imgH,
            bgraIsRotatedFromCaptured: bgraRotated
        )
        guard fy > 1 else { return }

        stateLock.lock()
        lastDepthMeters = distance
        lastFocalLengthY = fy
        stateLock.unlock()

        logFurnitureFitAR(
            "platform=ios phase=async_depth_sample dist_m=\(String(format: "%.3f", distance)) fy_px=\(String(format: "%.1f", fy)) norm=(\(String(format: "%.3f", Double(nx))),\(String(format: "%.3f", Double(ny))))"
        )
    }

    func estimatedHeightMeters(bboxHeightPx: Float) -> Float? {
        guard let pair = readLastDepthAndFocal() else { return nil }
        return FurnitureFitARSupport.estimatedPhysicalHeightMeters(
            bboxHeightPixels: bboxHeightPx,
            distanceMeters: pair.depth,
            focalLengthYPixels: pair.fy
        )
    }
}
