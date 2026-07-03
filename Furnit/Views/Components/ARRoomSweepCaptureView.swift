import ARKit
import SwiftUI
import UIKit

/// Guided LiDAR sweep capture that persists posed RGB/depth keyframes.
final class ARRoomSweepCaptureViewController: UIViewController, ARSessionDelegate {
    var onFinished: ((URL) -> Void)?
    var onCancelled: (() -> Void)?
    var onFailed: ((Error) -> Void)?
    var onFrameCaptured: ((Int) -> Void)?

    private enum CapturePolicy {
        static let minimumRotationDegrees: Float = 12
        static let minimumTranslationMeters: Float = 0.18
        static let maximumAngularVelocityDegreesPerSecond: Float = 80
        static let minimumSecondsBetweenKeyframes: TimeInterval = 0.35
    }

    private let arView = ARSCNView(frame: .zero)
    private let hintLabel = UILabel()
    private let doneButton = UIButton(type: .system)
    private let cancelButton = UIButton(type: .system)

    private var store: PosedFrameSweepStore?
    private var hasStartedSession = false
    private var lastKeyframeTransform: simd_float4x4?
    private var lastKeyframeTimestamp: TimeInterval?
    private var previousObservedTransform: simd_float4x4?
    private var previousObservedTimestamp: TimeInterval?

    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = .black

        arView.autoresizingMask = [.flexibleWidth, .flexibleHeight]
        arView.frame = view.bounds
        arView.session.delegate = self
        view.addSubview(arView)

        configureOverlay()
    }

    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)
        guard !hasStartedSession else { return }
        hasStartedSession = true
        startSweepSession()
    }

    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        arView.session.pause()
        CameraOwnershipDiagnostics.log(owner: "ARRoomSweepCaptureViewController", event: "ar_session_pause")
    }

    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        arView.frame = view.bounds
    }

    private func startSweepSession() {
        guard ARWorldTrackingConfiguration.isSupported else {
            fail(ARRoomSweepCaptureError.worldTrackingUnavailable)
            return
        }
        guard ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth)
                || ARWorldTrackingConfiguration.supportsFrameSemantics(.smoothedSceneDepth) else {
            fail(ARRoomSweepCaptureError.lidarDepthUnavailable)
            return
        }

        do {
            store = try PosedFrameSweepStore()
        } catch {
            fail(error)
            return
        }

        let config = ARWorldTrackingConfiguration()
        config.worldAlignment = .gravity
        config.planeDetection = [.horizontal, .vertical]
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) {
            config.frameSemantics.insert(.sceneDepth)
        }
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.smoothedSceneDepth) {
            config.frameSemantics.insert(.smoothedSceneDepth)
        }

        arView.session.run(config, options: [.resetTracking, .removeExistingAnchors])
        CameraOwnershipDiagnostics.log(
            owner: "ARRoomSweepCaptureViewController",
            event: "ar_session_run",
            details: "sceneDepth=\(ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth)) smoothed=\(ARWorldTrackingConfiguration.supportsFrameSemantics(.smoothedSceneDepth))"
        )
        updateHint()
    }

    private func configureOverlay() {
        hintLabel.textColor = .white
        hintLabel.font = .systemFont(ofSize: 14, weight: .medium)
        hintLabel.textAlignment = .center
        hintLabel.numberOfLines = 0
        hintLabel.backgroundColor = UIColor.black.withAlphaComponent(0.45)
        hintLabel.layer.cornerRadius = 8
        hintLabel.clipsToBounds = true
        hintLabel.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(hintLabel)

        doneButton.setTitle(NSLocalizedString("common.done", comment: "Done"), for: .normal)
        doneButton.setTitleColor(.white, for: .normal)
        doneButton.titleLabel?.font = .systemFont(ofSize: 18, weight: .semibold)
        doneButton.backgroundColor = UIColor.systemBlue.withAlphaComponent(0.9)
        doneButton.layer.cornerRadius = 10
        doneButton.translatesAutoresizingMaskIntoConstraints = false
        doneButton.addTarget(self, action: #selector(doneTapped), for: .touchUpInside)
        view.addSubview(doneButton)

        cancelButton.setTitle(NSLocalizedString("common.cancel", comment: "Cancel"), for: .normal)
        cancelButton.setTitleColor(.white, for: .normal)
        cancelButton.titleLabel?.font = .systemFont(ofSize: 18, weight: .medium)
        cancelButton.translatesAutoresizingMaskIntoConstraints = false
        cancelButton.addTarget(self, action: #selector(cancelTapped), for: .touchUpInside)
        view.addSubview(cancelButton)

        NSLayoutConstraint.activate([
            hintLabel.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 12),
            hintLabel.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 16),
            hintLabel.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -16),

            cancelButton.leadingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.leadingAnchor, constant: 20),
            cancelButton.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor, constant: -28),

            doneButton.trailingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.trailingAnchor, constant: -20),
            doneButton.centerYAnchor.constraint(equalTo: cancelButton.centerYAnchor),
            doneButton.widthAnchor.constraint(greaterThanOrEqualToConstant: 92),
            doneButton.heightAnchor.constraint(equalToConstant: 44),
        ])
    }

    private func updateHint() {
        let count = store?.frameCount ?? 0
        hintLabel.text = "Slowly pan around the room. Captured keyframes: \(count)"
    }

    @objc private func doneTapped() {
        arView.session.pause()
        guard let store else {
            fail(ARRoomSweepCaptureError.storeUnavailable)
            return
        }
        do {
            let manifestURL = try store.writeManifest()
            onFinished?(manifestURL.deletingLastPathComponent())
        } catch {
            fail(error)
        }
    }

    @objc private func cancelTapped() {
        arView.session.pause()
        onCancelled?()
    }

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        guard case .normal = frame.camera.trackingState else {
            return
        }
        guard frame.smoothedSceneDepth != nil || frame.sceneDepth != nil else {
            return
        }
        guard shouldCaptureKeyframe(frame) else {
            previousObservedTransform = frame.camera.transform
            previousObservedTimestamp = frame.timestamp
            return
        }

        do {
            guard let store else { throw ARRoomSweepCaptureError.storeUnavailable }
            _ = try store.append(frame: frame)
            lastKeyframeTransform = frame.camera.transform
            lastKeyframeTimestamp = frame.timestamp
            previousObservedTransform = frame.camera.transform
            previousObservedTimestamp = frame.timestamp
            DispatchQueue.main.async { [weak self] in
                guard let self else { return }
                self.updateHint()
                self.onFrameCaptured?(store.frameCount)
            }
        } catch {
            DispatchQueue.main.async { [weak self] in
                self?.fail(error)
            }
        }
    }

    func session(_ session: ARSession, didFailWithError error: Error) {
        fail(error)
    }

    private func shouldCaptureKeyframe(_ frame: ARFrame) -> Bool {
        let transform = frame.camera.transform
        let timestamp = frame.timestamp

        defer {
            previousObservedTransform = transform
            previousObservedTimestamp = timestamp
        }

        if let previousObservedTransform, let previousObservedTimestamp {
            let dt = max(timestamp - previousObservedTimestamp, 0.001)
            let perFrameRotation = Self.rotationDegrees(from: previousObservedTransform, to: transform)
            let angularVelocity = perFrameRotation / Float(dt)
            if angularVelocity > CapturePolicy.maximumAngularVelocityDegreesPerSecond {
                return false
            }
        }

        guard let lastKeyframeTransform, let lastKeyframeTimestamp else {
            return true
        }

        if timestamp - lastKeyframeTimestamp < CapturePolicy.minimumSecondsBetweenKeyframes {
            return false
        }

        let translation = simd_distance(Self.translation(lastKeyframeTransform), Self.translation(transform))
        let rotation = Self.rotationDegrees(from: lastKeyframeTransform, to: transform)
        return translation >= CapturePolicy.minimumTranslationMeters
            || rotation >= CapturePolicy.minimumRotationDegrees
    }

    private func fail(_ error: Error) {
        arView.session.pause()
        CameraOwnershipDiagnostics.log(
            owner: "ARRoomSweepCaptureViewController",
            event: "failed",
            details: error.localizedDescription
        )
        onFailed?(error)
    }

    private static func translation(_ transform: simd_float4x4) -> SIMD3<Float> {
        SIMD3(transform.columns.3.x, transform.columns.3.y, transform.columns.3.z)
    }

    private static func rotationDegrees(from lhs: simd_float4x4, to rhs: simd_float4x4) -> Float {
        let lhsRotation = simd_quatf(simd_float3x3(
            SIMD3(lhs.columns.0.x, lhs.columns.0.y, lhs.columns.0.z),
            SIMD3(lhs.columns.1.x, lhs.columns.1.y, lhs.columns.1.z),
            SIMD3(lhs.columns.2.x, lhs.columns.2.y, lhs.columns.2.z)
        ))
        let rhsRotation = simd_quatf(simd_float3x3(
            SIMD3(rhs.columns.0.x, rhs.columns.0.y, rhs.columns.0.z),
            SIMD3(rhs.columns.1.x, rhs.columns.1.y, rhs.columns.1.z),
            SIMD3(rhs.columns.2.x, rhs.columns.2.y, rhs.columns.2.z)
        ))
        let dot = min(Float(1), max(Float(0), abs(simd_dot(lhsRotation.vector, rhsRotation.vector))))
        return 2 * acos(dot) * 180 / .pi
    }
}

struct ARRoomSweepCaptureRepresentable: UIViewControllerRepresentable {
    var onFinished: (URL) -> Void
    var onCancelled: () -> Void
    var onFailed: (Error) -> Void
    var onFrameCaptured: (Int) -> Void = { _ in }

    func makeUIViewController(context: Context) -> ARRoomSweepCaptureViewController {
        let controller = ARRoomSweepCaptureViewController()
        controller.onFinished = onFinished
        controller.onCancelled = onCancelled
        controller.onFailed = onFailed
        controller.onFrameCaptured = onFrameCaptured
        return controller
    }

    func updateUIViewController(_ uiViewController: ARRoomSweepCaptureViewController, context: Context) {}
}

enum ARRoomSweepCaptureError: LocalizedError {
    case worldTrackingUnavailable
    case lidarDepthUnavailable
    case storeUnavailable

    var errorDescription: String? {
        switch self {
        case .worldTrackingUnavailable:
            return "ARKit world tracking is unavailable on this device."
        case .lidarDepthUnavailable:
            return "LiDAR scene depth is unavailable on this device."
        case .storeUnavailable:
            return "Sweep storage was not initialized."
        }
    }
}
