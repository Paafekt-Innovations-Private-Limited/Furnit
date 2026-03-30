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
