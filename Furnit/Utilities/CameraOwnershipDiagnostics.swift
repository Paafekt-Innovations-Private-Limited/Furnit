import Foundation
import AVFoundation

enum CameraOwnershipDiagnostics {
    static func log(owner: String, event: String, details: String = "") {
        let suffix = details.isEmpty ? "" : " \(details)"
        logDebug("📹 [CAMERA_OWNER] owner=\(owner) event=\(event)\(suffix)")
    }

    static func makeCaptureSessionObservers(
        session: AVCaptureSession,
        owner: String
    ) -> [NSObjectProtocol] {
        let nc = NotificationCenter.default
        return [
            nc.addObserver(
                forName: AVCaptureSession.didStartRunningNotification,
                object: session,
                queue: .main
            ) { _ in
                log(owner: owner, event: "capture_didStartRunning")
            },
            nc.addObserver(
                forName: AVCaptureSession.didStopRunningNotification,
                object: session,
                queue: .main
            ) { _ in
                log(owner: owner, event: "capture_didStopRunning")
            },
            nc.addObserver(
                forName: AVCaptureSession.wasInterruptedNotification,
                object: session,
                queue: .main
            ) { note in
                let rawReason = (note.userInfo?[AVCaptureSessionInterruptionReasonKey] as? NSNumber)?.intValue
                log(owner: owner, event: "capture_wasInterrupted", details: "reason=\(rawReason.map(String.init) ?? "nil")")
            },
            nc.addObserver(
                forName: AVCaptureSession.interruptionEndedNotification,
                object: session,
                queue: .main
            ) { _ in
                log(owner: owner, event: "capture_interruptionEnded")
            },
            nc.addObserver(
                forName: AVCaptureSession.runtimeErrorNotification,
                object: session,
                queue: .main
            ) { note in
                let nsError = note.userInfo?[AVCaptureSessionErrorKey] as? NSError
                let detail = nsError.map { "domain=\($0.domain) code=\($0.code)" } ?? "domain=nil code=nil"
                log(owner: owner, event: "capture_runtimeError", details: detail)
            },
        ]
    }

    static func removeObservers(_ tokens: [NSObjectProtocol]) {
        let nc = NotificationCenter.default
        for token in tokens {
            nc.removeObserver(token)
        }
    }
}
