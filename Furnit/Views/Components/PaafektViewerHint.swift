import SwiftUI

// MARK: - Shared glass surface (matches PaafektViewerToolbarCapsule)

extension View {
    func paafektGlassCapsuleSurface() -> some View {
        background {
            Capsule()
                .fill(.ultraThinMaterial)
                .background(Capsule().fill(Theme.Palette.viewerCapsuleFill))
        }
        .overlay(Capsule().stroke(Theme.Palette.hairline, lineWidth: 1))
    }
}

// MARK: - Screen-space hint anchors (2D overlay; never tied to 3D scene transforms)

extension View {
    /// Centers a hint chip below the top viewer toolbar row (screen-space).
    func paafektTopToolbarHintOverlay<Content: View>(
        isVisible: Bool = true,
        topInset: CGFloat = 52,
        @ViewBuilder content: () -> Content
    ) -> some View {
        VStack(spacing: 0) {
            if isVisible {
                content()
                    .padding(.top, topInset)
            }
            Spacer(minLength: 0)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
        .allowsHitTesting(false)
    }

    /// Centers a hint chip just above the bottom viewer action bar (screen-space).
    func paafektBottomToolbarHintOverlay<Content: View>(
        isVisible: Bool = true,
        bottomInset: CGFloat = 96,
        @ViewBuilder content: () -> Content
    ) -> some View {
        VStack(spacing: 0) {
            Spacer(minLength: 0)
            if isVisible {
                content()
                    .padding(.bottom, bottomInset)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .bottom)
        .allowsHitTesting(false)
    }
}

// MARK: - Glass hint chip (primary viewer helper pattern)

/// Glass capsule + gold monoline icon + one short caption line.
/// Screen-space only: place inside a root overlay `ZStack`, not inside scene/3D views.
struct PaafektHintChip: View {
    private let systemImage: String?
    private let assetImage: String?
    let text: String
    var maxWidth: CGFloat?

    init(systemImage: String, text: String, maxWidth: CGFloat? = nil) {
        self.systemImage = systemImage
        self.assetImage = nil
        self.text = text
        self.maxWidth = maxWidth
    }

    init(assetImage: String, text: String, maxWidth: CGFloat? = nil) {
        self.systemImage = nil
        self.assetImage = assetImage
        self.text = text
        self.maxWidth = maxWidth
    }

    var body: some View {
        HStack(alignment: .center, spacing: Theme.Space.sm) {
            hintIcon
                .frame(width: 22, height: 22)

            Text(text)
                .font(Theme.Typo.caption())
                .foregroundStyle(Theme.Palette.textPrimary)
                .lineLimit(1)
                .truncationMode(.tail)
        }
        .fixedSize(horizontal: maxWidth == nil, vertical: false)
        .frame(maxWidth: maxWidth)
        .padding(.horizontal, Theme.Space.md)
        .padding(.vertical, Theme.Space.sm)
        .paafektGlassCapsuleSurface()
        .accessibilityElement(children: .combine)
    }

    @ViewBuilder
    private var hintIcon: some View {
        if let assetImage {
            Image(assetImage)
                .renderingMode(.template)
                .resizable()
                .scaledToFit()
                .foregroundStyle(Theme.Palette.accent)
        } else if let systemImage {
            Image(systemName: systemImage)
                .font(.system(size: 15, weight: .medium))
                .foregroundStyle(Theme.Palette.accent)
        }
    }
}

// MARK: - Bottom scrim variant

/// Soft bottom gradient scrim with a hint chip floated above the toolbar region.
struct PaafektBottomScrimHint<Content: View>: View {
    @ViewBuilder let content: () -> Content

    var body: some View {
        ZStack(alignment: .bottom) {
            LinearGradient(
                colors: [.clear, Theme.Palette.background.opacity(0.88)],
                startPoint: .top,
                endPoint: .bottom
            )
            .frame(height: 120)
            .frame(maxWidth: .infinity)
            .allowsHitTesting(false)

            content()
                .padding(.horizontal, Theme.Space.lg)
                .padding(.bottom, Theme.Space.lg)
        }
        .allowsHitTesting(false)
    }
}

// MARK: - First-run coach mark (centered card over dimmed room)

enum PaafektViewerOnboarding {
    static let firstRunCoachSeenKey = "paafekt_viewer_first_run_coach_seen"
}

/// Centered card over a dimmed scrim — one-time first open acknowledgement.
struct PaafektViewerFirstRunCoachMark: View {
    let title: String
    let bodyText: String
    let confirmTitle: String
    let onConfirm: () -> Void

    var body: some View {
        ZStack {
            Theme.Palette.background.opacity(0.72)
                .ignoresSafeArea()

            VStack(spacing: Theme.Space.lg) {
                Image("PaafektMark")
                    .renderingMode(.template)
                    .resizable()
                    .scaledToFit()
                    .frame(width: 56, height: 56)
                    .foregroundStyle(Theme.Palette.accent)

                Text(title)
                    .font(Theme.Typo.title())
                    .foregroundStyle(Theme.Palette.textPrimary)
                    .multilineTextAlignment(.center)

                Text(bodyText)
                    .font(Theme.Typo.body())
                    .foregroundStyle(Theme.Palette.textSecondary)
                    .multilineTextAlignment(.center)
                    .fixedSize(horizontal: false, vertical: true)

                Button(action: onConfirm) {
                    Text(confirmTitle)
                        .font(Theme.Typo.headline())
                        .foregroundStyle(Theme.Palette.accentText)
                        .padding(.horizontal, Theme.Space.xxl)
                        .padding(.vertical, Theme.Space.md)
                        .background(Capsule().fill(Theme.Palette.accent))
                }
                .buttonStyle(.plain)
                .accessibilityLabel(confirmTitle)
            }
            .padding(Theme.Space.xl)
            .frame(maxWidth: 340)
            .paafektCardSurface()
            .padding(.horizontal, Theme.Space.xl)
        }
    }
}

/// First-run coach mark + transient navigation teaching chip (shared across room viewers).
struct PaafektViewerOnboardingLayer: View {
    @AppStorage(PaafektViewerOnboarding.firstRunCoachSeenKey) private var seenFirstRunCoach = false
    @State private var navigationTeachingVisible = false
    @State private var navigationTeachingDismissTask: Task<Void, Never>?
    let isReady: Bool

    var body: some View {
        ZStack {
            if isReady, !seenFirstRunCoach {
                PaafektViewerFirstRunCoachMark(
                    title: L10n.RoomViewer.firstRunCoachMarkTitle,
                    bodyText: L10n.RoomViewer.firstRunCoachMarkBody,
                    confirmTitle: L10n.RoomViewer.firstRunGotIt,
                    onConfirm: dismissFirstRunCoachMark
                )
                .transition(.opacity)
            }

            paafektBottomToolbarHintOverlay(
                isVisible: isReady && seenFirstRunCoach && navigationTeachingVisible,
                bottomInset: 120
            ) {
                PaafektHintChip(
                    systemImage: "hand.draw.fill",
                    text: L10n.RoomViewer.navigationTeachingHint
                )
                .transition(.opacity)
            }
        }
        .onChange(of: isReady) { _, ready in
            if ready, seenFirstRunCoach {
                restartNavigationTeachingHint()
            }
        }
        .onDisappear {
            cancelNavigationTeachingHint()
        }
    }

    private func dismissFirstRunCoachMark() {
        seenFirstRunCoach = true
        restartNavigationTeachingHint()
    }

    private func restartNavigationTeachingHint() {
        cancelNavigationTeachingHint()
        navigationTeachingVisible = true
        navigationTeachingDismissTask = Task { @MainActor in
            try? await Task.sleep(for: .seconds(3.5))
            guard !Task.isCancelled else { return }
            navigationTeachingVisible = false
        }
    }

    private func cancelNavigationTeachingHint() {
        navigationTeachingDismissTask?.cancel()
        navigationTeachingDismissTask = nil
        navigationTeachingVisible = false
    }
}
