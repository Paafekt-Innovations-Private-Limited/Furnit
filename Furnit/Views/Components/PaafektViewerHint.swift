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

    /// Centers a hint chip above the hero Fit/Capture row (screen-space).
    func paafektHeroRowHintOverlay<Content: View>(
        isVisible: Bool = true,
        bottomInset: CGFloat = 172,
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

/// Glass measurement pill for room viewer chrome (dimensions, furniture height).
struct PaafektRoomMeasurementPill: View {
    let primaryText: String
    var secondaryText: String? = nil
    var primaryColor: Color = Theme.Palette.textSecondary

    var body: some View {
        VStack(spacing: 2) {
            if let secondaryText {
                Text(secondaryText)
                    .font(Theme.Typo.caption())
                    .foregroundStyle(Theme.Palette.success)
            }
            Text(primaryText)
                .font(Theme.Typo.caption())
                .foregroundStyle(primaryColor)
        }
        .fixedSize(horizontal: true, vertical: false)
        .padding(.horizontal, Theme.Space.md)
        .padding(.vertical, Theme.Space.sm)
        .paafektGlassCapsuleSurface()
    }
}

/// Collapsed placement control in the measured lower cluster — icon only (ring = fit signal).
struct PaafektPlacementIntelligenceChip: View {
    let ringColor: Color
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            ZStack {
                Circle()
                    .fill(
                        LinearGradient(
                            colors: [Color(white: 0.22), Color(white: 0.12)],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        )
                    )
                    .frame(width: 40, height: 40)
                    .overlay(Circle().stroke(ringColor, lineWidth: 2.5))
                    .shadow(color: .black.opacity(0.35), radius: 4, x: 0, y: 2)
                Image(systemName: "square.split.2x2.fill")
                    .font(.system(size: 16, weight: .semibold))
                    .foregroundStyle(.white)
                    .symbolRenderingMode(.hierarchical)
                    .accessibilityHidden(true)
            }
        }
        .buttonStyle(.plain)
        .accessibilityLabel(L10n.RoomViewer.placementIntelligenceTitle)
        .accessibilityAddTraits(.isButton)
    }
}

/// Placement expanded card + chip row — own cluster row with spacing above the measurement pill.
struct PaafektImmersivePlacementIntelligenceRow<Expanded: View>: View {
    var isExpanded: Bool
    let ringColor: Color
    let onToggle: () -> Void
    @ViewBuilder let expandedContent: () -> Expanded

    var body: some View {
        VStack(spacing: Theme.Space.sm) {
            if isExpanded {
                expandedContent()
                    .transition(.opacity.combined(with: .move(edge: .bottom)))
            }
            PaafektPlacementIntelligenceChip(ringColor: ringColor, action: onToggle)
        }
    }
}

/// Bottom-anchored fit-mode rows (Smart Placement, furniture height) with consistent spacing tokens.
struct PaafektImmersiveFitClusterRows<Content: View>: View {
    @ViewBuilder let content: () -> Content

    var body: some View {
        VStack(spacing: Theme.Space.md) {
            content()
        }
        .frame(maxWidth: .infinity)
    }
}

// MARK: - First-run coach mark (centered card over dimmed room)

enum PaafektViewerOnboarding {
    static let firstRunCoachSeenKey = "paafekt_viewer_first_run_coach_seen"
    static let heroActionsHintSeenKey = "paafekt_viewer_hero_actions_hint_seen"
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

/// First-run coach mark + single hero-actions teaching chip (shared across room viewers).
struct PaafektViewerOnboardingLayer: View {
    let isReady: Bool
    var isChromeSummoned: Bool = true
    var heroHintBottomInset: CGFloat = 172

    @Binding private var replayTeachingHints: Bool

    @AppStorage(PaafektViewerOnboarding.firstRunCoachSeenKey) private var seenFirstRunCoach = false
    @AppStorage(PaafektViewerOnboarding.heroActionsHintSeenKey) private var seenHeroActionsHint = false
    @State private var navigationTeachingVisible = false
    @State private var heroActionsHintVisible = false
    @State private var navigationTeachingDismissTask: Task<Void, Never>?
    @State private var heroActionsHintDismissTask: Task<Void, Never>?

    init(
        isReady: Bool,
        isChromeSummoned: Bool = true,
        heroHintBottomInset: CGFloat = 172,
        replayTeachingHints: Binding<Bool> = .constant(false)
    ) {
        self.isReady = isReady
        self.isChromeSummoned = isChromeSummoned
        self.heroHintBottomInset = heroHintBottomInset
        self._replayTeachingHints = replayTeachingHints
    }

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

            paafektHeroRowHintOverlay(
                isVisible: isReady && isChromeSummoned && heroActionsHintVisible,
                bottomInset: heroHintBottomInset
            ) {
                PaafektHintChip(
                    systemImage: "hand.tap.fill",
                    text: L10n.RoomViewer.heroActionsTeachingHint,
                    maxWidth: 320
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
        .allowsHitTesting(isReady && !seenFirstRunCoach)
        .onChange(of: isReady) { _, ready in
            if ready, seenFirstRunCoach {
                restartNavigationTeachingHint()
                presentHeroActionsHintIfNeeded()
            }
        }
        .onChange(of: isChromeSummoned) { _, summoned in
            if summoned, isReady, seenFirstRunCoach {
                presentHeroActionsHintIfNeeded()
            }
        }
        .onChange(of: replayTeachingHints) { _, replay in
            guard replay else { return }
            replayTeachingHints = false
            restartNavigationTeachingHint(force: true)
            presentHeroActionsHint(force: true)
        }
        .onDisappear {
            cancelNavigationTeachingHint()
            cancelHeroActionsHint()
        }
    }

    private func dismissFirstRunCoachMark() {
        seenFirstRunCoach = true
        restartNavigationTeachingHint(force: true)
        presentHeroActionsHint(force: true)
    }

    private func presentHeroActionsHintIfNeeded() {
        guard !seenHeroActionsHint else { return }
        presentHeroActionsHint(force: false)
    }

    private func presentHeroActionsHint(force: Bool) {
        guard isReady, isChromeSummoned else { return }
        if !force, seenHeroActionsHint { return }

        cancelHeroActionsHint()
        heroActionsHintVisible = true
        heroActionsHintDismissTask = Task { @MainActor in
            try? await Task.sleep(for: .seconds(3.5))
            guard !Task.isCancelled else { return }
            heroActionsHintVisible = false
            if !force {
                seenHeroActionsHint = true
            }
        }
    }

    private func cancelHeroActionsHint() {
        heroActionsHintDismissTask?.cancel()
        heroActionsHintDismissTask = nil
        heroActionsHintVisible = false
    }

    private func restartNavigationTeachingHint(force: Bool = false) {
        cancelNavigationTeachingHint()
        guard force || seenFirstRunCoach else { return }
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
