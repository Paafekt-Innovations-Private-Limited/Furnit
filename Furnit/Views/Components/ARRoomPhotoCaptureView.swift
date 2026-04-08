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

        guard let image = Self.imageFromARFrame(frame) else {
            logDebug("❌ [AR] Failed to build UIImage from frame")
            return
        }

        let supplemental = Self.supplementalMetrics(from: frame)
        var fileURL: URL?
        if let data = image.jpegData(compressionQuality: 0.92) {
            let url = FileManager.default.temporaryDirectory.appendingPathComponent("ar_room_capture_\(UUID().uuidString).jpg")
            do {
                try data.write(to: url, options: [.atomic])
                fileURL = url
                logDebug("📷 [AR] Wrote temp JPEG \(url.lastPathComponent) bytes=\(data.count)")
            } catch {
                logDebug("❌ [AR] Temp JPEG write failed: \(error.localizedDescription)")
            }
        }

        arView.session.pause()
        CameraOwnershipDiagnostics.log(owner: "ARRoomPhotoCaptureViewController", event: "captured", details: "depth=\(frame.sceneDepth != nil)")
        onCaptured?(image, fileURL, supplemental)
    }

    // MARK: - Image + metrics

    private static func imageFromARFrame(_ frame: ARFrame) -> UIImage? {
        let pb = frame.capturedImage
        let ciImage = CIImage(cvPixelBuffer: pb)
        let context = CIContext(options: [.useSoftwareRenderer: false])
        guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else { return nil }
        let orientation = uiImageOrientationForInterface()
        let ui = UIImage(cgImage: cgImage, scale: 1.0, orientation: orientation)
        return ui.fixedOrientation()
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
        return out
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
