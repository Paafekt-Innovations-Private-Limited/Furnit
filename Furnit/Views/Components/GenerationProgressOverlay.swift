import SwiftUI

/// Overlay view showing progress during 3D room generation
/// Displays different phases: uploading, processing, downloading
struct GenerationProgressOverlay: View {
    // MARK: - Properties

    /// Current generation status
    let status: GenerationStatus

    /// Upload progress (0.0 to 1.0)
    let uploadProgress: Float

    /// Download progress (0.0 to 1.0)
    let downloadProgress: Float

    /// Status message to display
    let statusMessage: String

    /// Action when cancel button is tapped
    let onCancel: () -> Void

    // MARK: - Body

    var body: some View {
        ZStack {
            // Background overlay
            Color.black.opacity(0.6)
                .ignoresSafeArea()

            // Content card
            VStack(spacing: 24) {
                // Phase icon with animation
                phaseIconView

                // Status text
                VStack(spacing: 8) {
                    Text(phaseTitleText)
                        .font(.headline)
                        .foregroundColor(.white)

                    Text(statusMessage)
                        .font(.subheadline)
                        .foregroundColor(.white.opacity(0.8))
                        .multilineTextAlignment(.center)
                }

                // Progress indicator
                progressView

                // Cancel button
                Button(action: onCancel) {
                    Text(L10n.Common.cancel)
                        .font(.subheadline)
                        .fontWeight(.medium)
                        .foregroundColor(.white.opacity(0.8))
                        .padding(.horizontal, 24)
                        .padding(.vertical, 10)
                        .background(Color.white.opacity(0.2))
                        .cornerRadius(20)
                }
            }
            .padding(32)
            .background(
                RoundedRectangle(cornerRadius: 20)
                    .fill(Color(.systemGray6).opacity(0.95))
            )
            .padding(.horizontal, 40)
        }
    }

    // MARK: - Subviews

    /// Icon view with animation based on phase
    @ViewBuilder
    private var phaseIconView: some View {
        ZStack {
            // Background circle
            Circle()
                .fill(phaseColor.opacity(0.2))
                .frame(width: 80, height: 80)

            // Icon or spinner
            if status.showsSpinner {
                // Processing phase - rotating spinner
                ProgressView()
                    .progressViewStyle(CircularProgressViewStyle(tint: phaseColor))
                    .scaleEffect(1.5)
            } else {
                // Other phases - static or animated icon
                Image(systemName: status.iconName)
                    .font(.system(size: 36))
                    .foregroundColor(phaseColor)
                    .symbolEffect(.pulse, options: .repeating, isActive: status.showsProgress)
            }
        }
    }

    /// Progress bar for upload/download phases
    @ViewBuilder
    private var progressView: some View {
        if status.showsProgress {
            VStack(spacing: 8) {
                // Progress bar
                GeometryReader { geometry in
                    ZStack(alignment: .leading) {
                        // Background
                        RoundedRectangle(cornerRadius: 4)
                            .fill(Color.white.opacity(0.2))
                            .frame(height: 8)

                        // Progress fill
                        RoundedRectangle(cornerRadius: 4)
                            .fill(phaseColor)
                            .frame(width: geometry.size.width * CGFloat(currentProgress), height: 8)
                            .animation(.easeInOut(duration: 0.3), value: currentProgress)
                    }
                }
                .frame(height: 8)

                // Percentage text
                Text("\(Int(currentProgress * 100))%")
                    .font(.caption)
                    .fontWeight(.medium)
                    .foregroundColor(.white.opacity(0.7))
            }
            .frame(maxWidth: 200)
        } else if status.showsSpinner {
            // Processing phase - show estimated time
            Text(L10n.GenerationProgress.mayTakeFewMinutes)
                .font(.caption)
                .foregroundColor(.white.opacity(0.6))
        }
    }

    // MARK: - Computed Properties

    /// Current progress value based on phase
    private var currentProgress: Float {
        switch status {
        case .uploading:
            return uploadProgress
        case .downloading:
            return downloadProgress
        default:
            return 0
        }
    }

    /// Color for the current phase
    private var phaseColor: Color {
        switch status {
        case .uploading:
            return .blue
        case .processing:
            return .orange
        case .downloading:
            return .green
        case .completed:
            return .green
        case .failed:
            return .red
        default:
            return .gray
        }
    }

    /// Title text for current phase
    private var phaseTitleText: String {
        switch status {
        case .uploading:
            return L10n.GenerationProgress.uploadingImage
        case .processing:
            return L10n.GenerationProgress.generating3DModel
        case .downloading:
            return L10n.GenerationProgress.downloadingModel
        case .completed:
            return L10n.GenerationProgress.complete
        case .failed:
            return L10n.PhotoRoom.generationFailedTitle
        default:
            return L10n.GenerationProgress.preparing
        }
    }
}

struct SharpGenerationProgressOverlay: View {
    @ObservedObject var sharpService: SHARPService
    let onRunInBackground: () -> Void
    let onCancel: () -> Void

    private var progressValue: Double {
        sharpService.unifiedProgress
    }

    private var title: String {
        if sharpService.isDownloadingResources {
            return L10n.GenerationProgress.downloadingModel
        }
        if sharpService.isLoadingModel {
            return L10n.GenerationProgress.preparing
        }
        return L10n.GenerationProgress.generating3DModel
    }

    var body: some View {
        ZStack {
            Color.black.opacity(0.6)
                .ignoresSafeArea()

            VStack(spacing: 20) {
                ZStack {
                    Circle()
                        .fill(Color.purple.opacity(0.16))
                        .frame(width: 72, height: 72)
                    ProgressView()
                        .progressViewStyle(CircularProgressViewStyle(tint: .purple))
                        .scaleEffect(1.35)
                }

                VStack(spacing: 8) {
                    Text(title)
                        .font(.headline)
                        .foregroundColor(.primary)
                    Text(sharpService.statusMessage)
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                        .multilineTextAlignment(.center)
                }

                VStack(spacing: 8) {
                    ProgressView(value: progressValue)
                        .progressViewStyle(LinearProgressViewStyle(tint: .purple))
                    Text("\(Int(progressValue * 100))%")
                        .font(.caption)
                        .fontWeight(.medium)
                        .foregroundColor(.secondary)
                }

                HStack(spacing: 12) {
                    Button(action: onCancel) {
                        Text(L10n.Common.cancel)
                            .font(.subheadline)
                            .fontWeight(.medium)
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.bordered)

                    Button(action: onRunInBackground) {
                        Text(NSLocalizedString("sharp.runInBackground", value: "Run in Background", comment: "Continue room generation after leaving this screen"))
                            .font(.subheadline)
                            .fontWeight(.semibold)
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(.purple)
                }
            }
            .padding(28)
            .background(
                RoundedRectangle(cornerRadius: 18)
                    .fill(Color(.systemBackground).opacity(0.96))
            )
            .padding(.horizontal, 32)
        }
    }
}

struct SharpGenerationBottomBar: View {
    @ObservedObject private var sharpService = SHARPService.shared

    private var shouldShow: Bool {
        sharpService.shouldShowProgressFooter
    }

    private var progressValue: Double {
        sharpService.progressFooterValue
    }

    private var statusLine: String {
        sharpService.progressFooterMessage
    }

    private var isCompletionState: Bool {
        !sharpService.canCancelProgressFooter
    }

    var body: some View {
        if shouldShow {
            VStack(spacing: 10) {
                HStack(spacing: 12) {
                    VStack(alignment: .leading, spacing: 4) {
                        Text(statusLine)
                            .font(.system(size: 13, weight: .semibold))
                            .foregroundColor(.white)
                            .lineLimit(2)
                            .frame(maxWidth: .infinity, alignment: .leading)

                        Text("\(Int(progressValue * 100))%")
                            .font(.system(size: 12, weight: .medium))
                            .foregroundColor(.white.opacity(0.84))
                    }

                    if sharpService.canCancelProgressFooter {
                        Button {
                            sharpService.cancelGeneration()
                        } label: {
                            Image(systemName: "stop.fill")
                                .font(.system(size: 10, weight: .bold))
                                .foregroundColor(.white)
                                .frame(width: 34, height: 26)
                                .background(Color(red: 0.90, green: 0.22, blue: 0.21))
                                .cornerRadius(6)
                        }
                        .accessibilityLabel(NSLocalizedString("sharp.stopGeneration", value: "Stop generation", comment: "Stop SHARP room generation"))
                    } else {
                        Image(systemName: "checkmark.circle.fill")
                            .font(.system(size: 20, weight: .semibold))
                            .foregroundColor(.white)
                            .accessibilityHidden(true)
                    }
                }

                ProgressView(value: progressValue)
                    .progressViewStyle(LinearProgressViewStyle(tint: .white))
                    .opacity(isCompletionState ? 0.95 : 1.0)
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 12)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Color(red: 0.37, green: 0.21, blue: 0.69))
            .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
            .shadow(color: Color.black.opacity(0.28), radius: 12, x: 0, y: -4)
            .transition(.move(edge: .bottom).combined(with: .opacity))
        }
    }
}

// MARK: - Preview

#Preview("Uploading") {
    GenerationProgressOverlay(
        status: .uploading,
        uploadProgress: 0.65,
        downloadProgress: 0,
        statusMessage: "Uploading... 65%",
        onCancel: {}
    )
}

#Preview("Processing") {
    GenerationProgressOverlay(
        status: .processing,
        uploadProgress: 1.0,
        downloadProgress: 0,
        statusMessage: "Processing: pending...",
        onCancel: {}
    )
}

#Preview("Downloading") {
    GenerationProgressOverlay(
        status: .downloading,
        uploadProgress: 1.0,
        downloadProgress: 0.45,
        statusMessage: "Downloading... 45%",
        onCancel: {}
    )
}
