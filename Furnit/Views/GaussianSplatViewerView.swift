import SwiftUI

/// Full-screen viewer for Gaussian splat PLY files
/// Provides immersive 3D visualization with camera controls
struct GaussianSplatViewerView: View {

    // MARK: - Properties

    /// The model containing PLY file information
    let model: USDZModel

    // MARK: - State

    /// Loading state for the PLY file
    @State private var isLoading = true

    /// Error message if loading fails
    @State private var loadError: String?

    /// Controls visibility of UI overlay
    @State private var showControls = true

    /// Zoom level for the 3D scene (1.0 = default, higher = zoomed in)
    @State private var zoomLevel: Float = 1.0

    /// Environment dismiss action
    @Environment(\.dismiss) private var dismiss

    // MARK: - Body

    var body: some View {
        ZStack {
            // Background color
            Color.black.ignoresSafeArea()

            // Main splat renderer
            if let url = model.temporaryURL {
                GaussianSplatView(
                    plyURL: url,
                    isLoading: $isLoading,
                    loadError: $loadError,
                    zoomLevel: $zoomLevel,
                    onBoundsAvailable: nil
                )
                .ignoresSafeArea()
                .onTapGesture {
                    // Toggle UI visibility on tap
                    withAnimation(.easeInOut(duration: 0.2)) {
                        showControls.toggle()
                    }
                }
            } else {
                // Error state - no URL available
                VStack(spacing: 16) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .font(.system(size: 48))
                        .foregroundColor(.orange)

                    Text("File Not Found")
                        .font(.headline)
                        .foregroundColor(.white)

                    Text("The PLY file could not be located.")
                        .font(.subheadline)
                        .foregroundColor(.gray)
                        .multilineTextAlignment(.center)

                    Button("Go Back") {
                        dismiss()
                    }
                    .buttonStyle(.borderedProminent)
                    .padding(.top, 8)
                }
                .padding()
            }

            // Loading overlay
            if isLoading {
                loadingOverlay
            }

            // Error overlay
            if let error = loadError {
                errorOverlay(message: error)
            }

            // Navigation controls overlay
            if showControls && !isLoading && loadError == nil {
                controlsOverlay
            }
        }
        .navigationBarHidden(true)
        .statusBar(hidden: !showControls)
    }

    // MARK: - Subviews

    /// Loading indicator overlay
    private var loadingOverlay: some View {
        VStack(spacing: 16) {
            ProgressView()
                .scaleEffect(1.5)
                .tint(.white)

            Text("Loading 3D Scene...")
                .font(.headline)
                .foregroundColor(.white)

            Text(model.displayName)
                .font(.subheadline)
                .foregroundColor(.gray)

            if model.fileSize != nil {
                Text(model.fileSizeFormatted)
                    .font(.caption)
                    .foregroundColor(.gray.opacity(0.8))
            }
        }
        .padding(32)
        .background(
            RoundedRectangle(cornerRadius: 16)
                .fill(.ultraThinMaterial)
        )
    }

    /// Error state overlay
    private func errorOverlay(message: String) -> some View {
        VStack(spacing: 16) {
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.system(size: 48))
                .foregroundColor(.red)

            Text("Failed to Load")
                .font(.headline)
                .foregroundColor(.white)

            Text(message)
                .font(.subheadline)
                .foregroundColor(.gray)
                .multilineTextAlignment(.center)
                .padding(.horizontal)

            Button("Go Back") {
                dismiss()
            }
            .buttonStyle(.borderedProminent)
            .tint(.red)
            .padding(.top, 8)
        }
        .padding(32)
        .background(
            RoundedRectangle(cornerRadius: 16)
                .fill(.ultraThinMaterial)
        )
    }

    /// Navigation and info controls overlay
    private var controlsOverlay: some View {
        VStack {
            // Top bar with back button and title
            HStack {
                // Back button
                Button(action: { dismiss() }) {
                    HStack(spacing: 6) {
                        Image(systemName: "chevron.left")
                            .font(.system(size: 18, weight: .semibold))

                        Text("Back")
                            .font(.body)
                    }
                    .foregroundColor(.white)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                    .background(
                        Capsule()
                            .fill(.ultraThinMaterial)
                    )
                }

                Spacer()

                // Model info badge
                VStack(alignment: .trailing, spacing: 2) {
                    Text(model.displayName)
                        .font(.subheadline)
                        .fontWeight(.semibold)
                        .foregroundColor(.white)
                        .lineLimit(1)

                    HStack(spacing: 4) {
                        Text("PLY")
                            .font(.caption2)
                            .fontWeight(.bold)
                            .foregroundColor(.purple)

                        if let _ = model.fileSize {
                            Text("•")
                                .font(.caption2)
                                .foregroundColor(.gray)

                            Text(model.fileSizeFormatted)
                                .font(.caption2)
                                .foregroundColor(.gray)
                        }
                    }
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .background(
                    RoundedRectangle(cornerRadius: 8)
                        .fill(.ultraThinMaterial)
                )
            }
            .padding(.horizontal, 16)
            .padding(.top, 8)

            Spacer()

            // Bottom hint for controls
            Text("Drag to rotate • Slide to zoom • Tap to toggle UI")
                .font(.caption)
                .foregroundColor(.white.opacity(0.7))
                .padding(.horizontal, 16)
                .padding(.vertical, 10)
                .background(
                    Capsule()
                        .fill(.ultraThinMaterial)
                )
                .padding(.bottom, 32)
        }
        .overlay(alignment: .trailing) {
            // Vertical zoom slider on right edge
            zoomSlider
        }
    }

    /// Vertical zoom slider control
    private var zoomSlider: some View {
        VStack(spacing: 8) {
            // Zoom in icon (top = higher zoom)
            Image(systemName: "plus.magnifyingglass")
                .font(.system(size: 14))
                .foregroundColor(.white.opacity(0.7))

            // Rotated slider for vertical orientation
            Slider(value: $zoomLevel, in: 0.5...3.0)
                .rotationEffect(.degrees(-90))
                .frame(width: 120)
                .tint(.white.opacity(0.8))

            // Zoom out icon (bottom = lower zoom)
            Image(systemName: "minus.magnifyingglass")
                .font(.system(size: 14))
                .foregroundColor(.white.opacity(0.7))
        }
        .padding(.vertical, 16)
        .padding(.horizontal, 8)
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(.ultraThinMaterial)
        )
        .padding(.trailing, 12)
    }
}

// MARK: - Preview

#Preview("Loading State") {
    GaussianSplatViewerView(
        model: USDZModel(
            name: "Room_20251229_120000",
            fileName: "Room_20251229_120000",
            isSavedRoom: true,
            fileType: .ply,
            fileSize: 63_000_000
        )
    )
}

#Preview("Error State") {
    let view = GaussianSplatViewerView(
        model: USDZModel(
            name: "Test_Room",
            fileName: "Test_Room",
            isSavedRoom: true,
            fileType: .ply,
            fileSize: 50_000_000
        )
    )
    return view
}
