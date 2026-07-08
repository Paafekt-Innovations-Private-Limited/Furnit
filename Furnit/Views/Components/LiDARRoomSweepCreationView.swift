import ARKit
import SwiftUI

private struct LiDARSplatViewerDestination: Identifiable, Hashable {
    let id = UUID()
    let result: PosedFrameSweepFusionResult

    static func == (lhs: LiDARSplatViewerDestination, rhs: LiDARSplatViewerDestination) -> Bool {
        lhs.id == rhs.id
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(id)
    }
}

struct LiDARRoomSweepCreationView: View {
    @Environment(\.dismiss) private var dismiss

    @State private var phase: Phase = .capturing
    @State private var viewerDestination: LiDARSplatViewerDestination?
    @State private var capturedFrameCount = 0

    private enum Phase: Equatable {
        case capturing
        case processing(String)
        case failed(String)
    }

    var body: some View {
        ZStack {
            switch phase {
            case .capturing:
                if Self.supportsLiDARSweep {
                    ARRoomSweepCaptureRepresentable(
                        onFinished: handleSweepFinished(sessionURL:),
                        onCancelled: { dismiss() },
                        onFailed: { error in
                            phase = .failed(error.localizedDescription)
                        },
                        onFrameCaptured: { count in
                            capturedFrameCount = count
                        }
                    )
                    .ignoresSafeArea()
                } else {
                    unavailableView
                }

            case .processing(let message):
                processingView(message: message)

            case .failed(let message):
                failureView(message: message)
            }
        }
        .navigationTitle("LiDAR Room")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .navigationBarLeading) {
                Button(L10n.Common.cancel) {
                    dismiss()
                }
            }
        }
        .navigationDestination(item: $viewerDestination) { destination in
            SplatRoomView(
                plyURL: destination.result.plyURL,
                allowSave: true,
                photoOrientation: .portrait,
                splatPlyAabbWidth: destination.result.roomWidthMeters,
                splatPlyAabbHeight: destination.result.roomHeightMeters,
                splatPlyAabbDepth: destination.result.roomDepthMeters,
                splatRoomWidth: destination.result.roomWidthMeters,
                splatRoomHeight: destination.result.roomHeightMeters,
                splatRoomDepth: destination.result.roomDepthMeters,
                roomCoordinateFrame: .arWorldMeters
            )
            .onAppear {
                logDebug("[LiDARSweep] opening fused room: \(destination.result.summary)")
            }
            .onDisappear {
                viewerDestination = nil
                capturedFrameCount = 0
                phase = .capturing
            }
        }
    }

    private static var supportsLiDARSweep: Bool {
        ARWorldTrackingConfiguration.isSupported &&
            (
                ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) ||
                ARWorldTrackingConfiguration.supportsFrameSemantics(.smoothedSceneDepth)
            )
    }

    private var unavailableView: some View {
        VStack(spacing: 18) {
            Image(systemName: "arkit")
                .font(.system(size: 48))
                .foregroundColor(.secondary)
            Text("LiDAR is required for fit-grade room capture.")
                .font(.headline)
                .multilineTextAlignment(.center)
            Button(L10n.Common.done) {
                dismiss()
            }
            .buttonStyle(.borderedProminent)
        }
        .padding(28)
    }

    private func processingView(message: String) -> some View {
        VStack(spacing: 16) {
            ProgressView()
                .scaleEffect(1.4)
            Text(message)
                .font(.headline)
            if capturedFrameCount > 0 {
                Text("Captured keyframes: \(capturedFrameCount)")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
            }
        }
        .padding(28)
    }

    private func failureView(message: String) -> some View {
        VStack(spacing: 18) {
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.system(size: 44))
                .foregroundColor(.orange)
            Text(message)
                .font(.headline)
                .multilineTextAlignment(.center)
            HStack(spacing: 12) {
                Button(L10n.Common.cancel) {
                    dismiss()
                }
                .buttonStyle(.bordered)

                Button(L10n.Common.retry) {
                    capturedFrameCount = 0
                    phase = .capturing
                }
                .buttonStyle(.borderedProminent)
            }
        }
        .padding(28)
    }

    private func handleSweepFinished(sessionURL: URL) {
        phase = .processing("Validating LiDAR alignment...")
        let frameCount = Self.loadFrameCount(sessionURL: sessionURL)
        capturedFrameCount = frameCount

        Task.detached(priority: .userInitiated) {
            do {
                let validation = try Self.validate(sessionURL: sessionURL, frameCount: frameCount)
                await MainActor.run {
                    phase = .processing("Fusing LiDAR sweep...")
                }
                let fusion = try PosedFrameSweepFusion.fuse(
                    sessionURL: sessionURL,
                    validationResult: validation
                )
                logDebug("[LiDARSweep] fusion complete: \(fusion.summary)")
                await MainActor.run {
                    phase = .processing("Opening fused room...")
                    viewerDestination = LiDARSplatViewerDestination(result: fusion)
                }
            } catch {
                logDebug("[LiDARSweep] fusion failed: \(error.localizedDescription)")
                await MainActor.run {
                    phase = .failed(error.localizedDescription)
                }
            }
        }
    }

    nonisolated private static func validate(
        sessionURL: URL,
        frameCount: Int
    ) throws -> PosedFrameSweepValidationResult? {
        guard frameCount >= 2 else {
            throw LiDARRoomSweepCreationError.notEnoughFrames
        }
        let secondIndex = min(max(1, frameCount / 2), frameCount - 1)
        return try PosedFrameSweepValidator.validate(
            sessionURL: sessionURL,
            firstFrameIndex: 0,
            secondFrameIndex: secondIndex,
            stride: 4
        )
    }

    private static func loadFrameCount(sessionURL: URL) -> Int {
        let url = sessionURL.appendingPathComponent("poses.json")
        guard let data = try? Data(contentsOf: url) else { return 0 }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return (try? decoder.decode(PosedFrameSweepManifest.self, from: data).frames.count) ?? 0
    }
}

private enum LiDARRoomSweepCreationError: LocalizedError {
    case notEnoughFrames

    var errorDescription: String? {
        switch self {
        case .notEnoughFrames:
            return "Capture at least two overlapping LiDAR keyframes before finishing."
        }
    }
}
