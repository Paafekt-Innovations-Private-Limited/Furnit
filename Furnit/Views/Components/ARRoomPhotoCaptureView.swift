import ARKit
import SwiftUI
import UIKit

/// Room-creation camera using **ARKit** so each frame includes ``ARCamera`` intrinsics (and scene depth on LiDAR).
/// Feeds ``CameraExifSidecar`` via supplemental doubles (`focalLengthPx`, image dimensions) for SHARP / wall measurement.
final class ARRoomPhotoCaptureViewController: UIViewController, ARSessionDelegate {
    var onCaptured: ((UIImage, URL?, [String: Double]) -> Void)?
    var onCancelled: (() -> Void)?

    private let arView = ARSCNView(frame: .zero)
    private var hasStartedSession = false
    private let captureProcessingQueue = DispatchQueue(
        label: "com.furnit.roomCapture.encode",
        qos: .userInitiated
    )
    private static let captureCIContext = CIContext(options: [.useSoftwareRenderer: false])

    private let captureButton = UIButton(type: .system)
    private let cancelButton = UIButton(type: .system)
    private let hintLabel = UILabel()

    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = .black

        arView.autoresizingMask = [.flexibleWidth, .flexibleHeight]
        arView.frame = view.bounds
        arView.session.delegate = self
        view.addSubview(arView)

        hintLabel.text = NSLocalizedString("camera.ar.hint", comment: "Point at the room, hold steady, then capture.")
        hintLabel.textColor = .white
        hintLabel.font = .systemFont(ofSize: 14, weight: .medium)
        hintLabel.textAlignment = .center
        hintLabel.numberOfLines = 0
        hintLabel.backgroundColor = UIColor.black.withAlphaComponent(0.45)
        hintLabel.layer.cornerRadius = 8
        hintLabel.clipsToBounds = true
        hintLabel.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(hintLabel)

        captureButton.setImage(
            UIImage(systemName: "circle.inset.filled", withConfiguration: UIImage.SymbolConfiguration(pointSize: 70)),
            for: .normal
        )
        captureButton.tintColor = .white
        captureButton.translatesAutoresizingMaskIntoConstraints = false
        captureButton.addTarget(self, action: #selector(captureTapped), for: .touchUpInside)
        view.addSubview(captureButton)

        cancelButton.setTitle(NSLocalizedString("common.cancel", comment: ""), for: .normal)
        cancelButton.setTitleColor(.white, for: .normal)
        cancelButton.titleLabel?.font = .systemFont(ofSize: 18, weight: .medium)
        cancelButton.translatesAutoresizingMaskIntoConstraints = false
        cancelButton.addTarget(self, action: #selector(cancelTapped), for: .touchUpInside)
        view.addSubview(cancelButton)

        NSLayoutConstraint.activate([
            hintLabel.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 12),
            hintLabel.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 16),
            hintLabel.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -16),

            captureButton.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            captureButton.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor, constant: -28),

            cancelButton.leadingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.leadingAnchor, constant: 20),
            cancelButton.centerYAnchor.constraint(equalTo: captureButton.centerYAnchor),
        ])
    }

    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)
        guard !hasStartedSession else { return }
        hasStartedSession = true

        let config = ARWorldTrackingConfiguration()
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) {
            config.frameSemantics.insert(.sceneDepth)
        }
        // Floor detection: measures the real camera height so room scaling doesn't
        // have to assume the 1.7 m eye-level prior.
        config.planeDetection = [.horizontal]
        arView.session.run(config, options: [.resetTracking, .removeExistingAnchors])
        CameraOwnershipDiagnostics.log(owner: "ARRoomPhotoCaptureViewController", event: "ar_session_run", details: "sceneDepth=\(ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth))")
    }

    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        arView.session.pause()
        CameraOwnershipDiagnostics.log(owner: "ARRoomPhotoCaptureViewController", event: "ar_session_pause")
    }

    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        arView.frame = view.bounds
    }

    @objc private func cancelTapped() {
        arView.session.pause()
        onCancelled?()
    }

    @objc private func captureTapped() {
        guard let frame = arView.session.currentFrame else {
            logDebug("❌ [AR] No ARFrame available yet")
            return
        }

        captureButton.isEnabled = false
        cancelButton.isEnabled = false
        let capturedImage = frame.capturedImage
        let orientation = Self.uiImageOrientationForInterface()
        let supplemental = Self.supplementalMetrics(from: frame)
        let hadSceneDepth = frame.sceneDepth != nil

        captureProcessingQueue.async { [weak self] in
            guard let image = Self.imageFromPixelBuffer(capturedImage, orientation: orientation) else {
                logDebug("❌ [AR] Failed to build UIImage from frame")
                DispatchQueue.main.async {
                    self?.captureButton.isEnabled = true
                    self?.cancelButton.isEnabled = true
                }
                return
            }

            let fileURL = Self.writeTempJPEG(image, quality: 0.88)

            DispatchQueue.main.async { [weak self] in
                guard let self else { return }
                self.arView.session.pause()
                CameraOwnershipDiagnostics.log(owner: "ARRoomPhotoCaptureViewController", event: "captured", details: "depth=\(hadSceneDepth)")
                self.onCaptured?(image, fileURL, supplemental)
            }
        }
    }

    // MARK: - Image + metrics

    private static func imageFromPixelBuffer(_ pixelBuffer: CVPixelBuffer, orientation: UIImage.Orientation) -> UIImage? {
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        guard let cgImage = captureCIContext.createCGImage(ciImage, from: ciImage.extent) else { return nil }
        let ui = UIImage(cgImage: cgImage, scale: 1.0, orientation: orientation)
        return ui.fixedOrientation()
    }

    private static func writeTempJPEG(_ image: UIImage, quality: CGFloat) -> URL? {
        guard let data = image.jpegData(compressionQuality: quality) else { return nil }
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("ar_room_capture_\(UUID().uuidString).jpg")
        do {
            try data.write(to: url, options: [.atomic])
            logDebug("📷 [AR] Wrote temp JPEG \(url.lastPathComponent) bytes=\(data.count)")
            return url
        } catch {
            logDebug("❌ [AR] Temp JPEG write failed: \(error.localizedDescription)")
            return nil
        }
    }

    private static func uiImageOrientationForInterface() -> UIImage.Orientation {
        guard let scene = UIApplication.shared.connectedScenes.first as? UIWindowScene else {
            return .right
        }
        let interfaceOrientation = scene.effectiveGeometry.interfaceOrientation
        switch interfaceOrientation {
        case .portrait: return .right
        case .portraitUpsideDown: return .left
        case .landscapeLeft: return .up
        case .landscapeRight: return .down
        default: return .right
        }
    }

    /// Values merged into ``CameraExifSidecar``; `focalLengthPx` is consumed by ``SharpCameraSidecar``.
    private static func supplementalMetrics(from frame: ARFrame) -> [String: Double] {
        let cam = frame.camera
        let intrinsics = cam.intrinsics
        let c = intrinsics.columns
        // K = [fx 0 cx; 0 fy cy; 0 0 1] in row form; column-major columns: (fx,0,0), (0,fy,0), (cx,cy,1).
        let fx = Double(c.0.x)
        let fy = Double(c.1.y)
        let cx = Double(c.2.x)
        let cy = Double(c.2.y)
        let w = Double(cam.imageResolution.width)
        let h = Double(cam.imageResolution.height)
        var out: [String: Double] = [
            "focalLengthPx": fx,
            "imageWidthPx": w,
            "imageHeightPx": h,
            "arkitFocalLengthYPx": fy,
            "arkitPrincipalXPx": cx,
            "arkitPrincipalYPx": cy,
        ]
        if frame.sceneDepth != nil {
            out["arkitSceneDepthAvailable"] = 1.0
        } else {
            out["arkitSceneDepthAvailable"] = 0.0
        }
        if let gravityDown = gravityDownInFixedImageFrame(camera: cam) {
            out["arkitGravityDownImageX"] = Double(gravityDown.x)
            out["arkitGravityDownImageY"] = Double(gravityDown.y)
            out["arkitGravityDownImageZ"] = Double(gravityDown.z)
        }
        if let cameraHeight = cameraHeightAboveFloor(frame: frame) {
            out["arkitCameraHeightM"] = Double(cameraHeight)
        }
        return out
    }

    /// Real camera height above the detected floor plane (ARKit world is gravity-aligned,
    /// +Y up). Replaces the fixed 1.7 m eye-level prior when available. The floor is taken
    /// as the lowest sizeable horizontal plane below the camera.
    private static func cameraHeightAboveFloor(frame: ARFrame) -> Float? {
        let cameraY = frame.camera.transform.columns.3.y
        var floorY: Float?
        for anchor in frame.anchors {
            guard let plane = anchor as? ARPlaneAnchor, plane.alignment == .horizontal else { continue }
            let planeY = (anchor.transform * SIMD4<Float>(plane.center.x, plane.center.y, plane.center.z, 1)).y
            let extent = plane.planeExtent
            // Ignore tiny surfaces (shelves, seats) — the floor is large and below the camera.
            guard min(extent.width, extent.height) > 0.25, planeY < cameraY - 0.4 else { continue }
            if floorY == nil || planeY < floorY! {
                floorY = planeY
            }
        }
        guard let floorY else { return nil }
        let height = cameraY - floorY
        guard height.isFinite, (0.5...2.5).contains(height) else { return nil }
        return height
    }

    /// World-down direction expressed in the pixel frame of the **fixed-orientation** captured image
    /// (pinhole convention: +X right, +Y down, +Z forward). ARKit world is gravity-aligned (+Y up),
    /// so this is exact device gravity — used for leveling instead of the GeoCalib CNN estimate.
    private static func gravityDownInFixedImageFrame(camera: ARCamera) -> SIMD3<Float>? {
        let t = camera.transform
        let rotation = simd_float3x3(
            SIMD3(t.columns.0.x, t.columns.0.y, t.columns.0.z),
            SIMD3(t.columns.1.x, t.columns.1.y, t.columns.1.z),
            SIMD3(t.columns.2.x, t.columns.2.y, t.columns.2.z)
        )
        // World down in ARKit camera space (+X right, +Y up, +Z backward, landscape-right sensor).
        let downCamera = simd_transpose(rotation) * SIMD3<Float>(0, -1, 0)
        guard downCamera.x.isFinite, downCamera.y.isFinite, downCamera.z.isFinite else { return nil }
        // ARKit camera space -> pinhole sensor frame (+X right, +Y down, +Z forward).
        let downSensor = simd_normalize(SIMD3<Float>(downCamera.x, -downCamera.y, -downCamera.z))
        // Apply the same rotation `fixedOrientation()` applies to the pixels.
        switch uiImageOrientationForInterface() {
        case .right: // sensor rotated 90° CW (portrait)
            return SIMD3(-downSensor.y, downSensor.x, downSensor.z)
        case .left: // 90° CCW (portrait upside down)
            return SIMD3(downSensor.y, -downSensor.x, downSensor.z)
        case .down: // 180°
            return SIMD3(-downSensor.x, -downSensor.y, downSensor.z)
        default: // .up, landscape-right: pixels match sensor
            return downSensor
        }
    }

    func session(_ session: ARSession, didFailWithError error: Error) {
        logDebug("⚠️ [AR] session failed: \(error.localizedDescription)")
    }
}

struct ARRoomPhotoCaptureRepresentable: UIViewControllerRepresentable {
    @Binding var capturedImage: UIImage?
    @Binding var sourceImageURL: URL?
    @Binding var captureMediaMetadata: [AnyHashable: Any]?
    @Binding var supplementalCameraDoubles: [String: Double]?
    @Environment(\.dismiss) private var dismiss

    func makeUIViewController(context: Context) -> ARRoomPhotoCaptureViewController {
        let vc = ARRoomPhotoCaptureViewController()
        vc.onCaptured = { image, url, supplemental in
            capturedImage = image
            sourceImageURL = url
            captureMediaMetadata = nil
            supplementalCameraDoubles = supplemental
            dismiss()
        }
        vc.onCancelled = {
            supplementalCameraDoubles = nil
            dismiss()
        }
        return vc
    }

    func updateUIViewController(_ uiViewController: ARRoomPhotoCaptureViewController, context: Context) {}
}

// MARK: - When to use AR vs standard camera

/// Routing for **standard** room photo capture (``CameraCaptureView`` / non–wide-angle).
/// **LiDAR is not required** for the AR path: world tracking still yields accurate intrinsics in `camera_exif.json`.
/// Scene depth is attached only when ``ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth)`` (LiDAR / depth-capable devices).
enum ARRoomPhotoCapturePolicy {
    /// Use ARKit capture only on a **physical device** that supports **world tracking**.
    /// Simulator and hardware without ARKit use ``UIImagePickerController`` (same as pre-AR behavior).
    static var useARKitForStandardRoomPhoto: Bool {
        #if targetEnvironment(simulator)
        return false
        #else
        return ARWorldTrackingConfiguration.isSupported
        #endif
    }
}
