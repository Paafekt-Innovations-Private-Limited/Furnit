import SwiftUI

/// Snackbar component to display file information
/// Shows file name, type, and size with auto-dismiss capability
struct FileInfoSnackbar: View {
    // MARK: - Properties

    /// The model to display info for
    let model: USDZModel

    /// Binding to control visibility
    @Binding var isShowing: Bool

    /// Auto-dismiss timer duration (seconds)
    private let autoDismissDelay: TimeInterval = 3.0

    // MARK: - Body

    var body: some View {
        HStack(spacing: 12) {
            // File type icon
            Image(systemName: model.fileType.iconName)
                .font(.title2)
                .foregroundColor(iconColor)
                .frame(width: 40, height: 40)
                .background(iconColor.opacity(0.15))
                .cornerRadius(8)

            // File details
            VStack(alignment: .leading, spacing: 4) {
                Text(model.displayName)
                    .font(.subheadline)
                    .fontWeight(.semibold)
                    .foregroundColor(.primary)
                    .lineLimit(1)

                HStack(spacing: 8) {
                    // File type badge
                    Text(model.fileType.rawValue.uppercased())
                        .font(.caption2)
                        .fontWeight(.bold)
                        .foregroundColor(.white)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(iconColor)
                        .cornerRadius(4)

                    // File size
                    Text(model.fileSizeFormatted)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }

            Spacer()

            // Dismiss button
            Button(action: { dismissSnackbar() }) {
                Image(systemName: "xmark")
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .frame(width: 24, height: 24)
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(Color(.systemBackground))
                .shadow(color: .black.opacity(0.15), radius: 8, x: 0, y: 4)
        )
        .padding(.horizontal, 16)
        .padding(.bottom, 16)
        .transition(.move(edge: .bottom).combined(with: .opacity))
        .onAppear {
            // Auto-dismiss after delay
            DispatchQueue.main.asyncAfter(deadline: .now() + autoDismissDelay) {
                if isShowing {
                    dismissSnackbar()
                }
            }
        }
    }

    // MARK: - Helpers

    /// Color based on file type
    private var iconColor: Color {
        switch model.fileType {
        case .usdz:
            return .green
        case .ply:
            return .purple
        case .meshroom:
            return .orange
        case .glb:
            return .blue
        }
    }

    /// Dismiss with animation
    private func dismissSnackbar() {
        withAnimation(.easeInOut(duration: 0.25)) {
            isShowing = false
        }
    }
}

// MARK: - Preview

#Preview("PLY File") {
    ZStack {
        Color.gray.opacity(0.3)
            .ignoresSafeArea()

        VStack {
            Spacer()

            FileInfoSnackbar(
                model: USDZModel(
                    name: "Room_20251229_120000",
                    fileName: "Room_20251229_120000",
                    isSavedRoom: true,
                    fileType: .ply,
                    fileSize: 52_428_800 // 50 MB
                ),
                isShowing: .constant(true)
            )
        }
    }
}

#Preview("USDZ File") {
    ZStack {
        Color.gray.opacity(0.3)
            .ignoresSafeArea()

        VStack {
            Spacer()

            FileInfoSnackbar(
                model: USDZModel(
                    name: "cozy_living_room",
                    fileName: "cozy_living_room",
                    isSavedRoom: true,
                    fileType: .usdz,
                    fileSize: 15_728_640 // 15 MB
                ),
                isShowing: .constant(true)
            )
        }
    }
}
