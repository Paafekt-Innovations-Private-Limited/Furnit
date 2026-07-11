import SwiftUI

// MARK: - Immersive viewer chrome (resting ↔ summoned)

/// Default: resting (3D fills screen). Summon via gold affordance or tap; auto-hide ~3s.
@MainActor
final class PaafektViewerChromeController: ObservableObject {
    enum Phase: Equatable { case resting, summoned }

    @Published private(set) var phase: Phase = .resting

    var isSummoned: Bool { phase == .summoned }
    var isResting: Bool { phase == .resting }

    private var hideTask: Task<Void, Never>?
    static let autoHideSeconds: TimeInterval = 3

    func summon() {
        hideTask?.cancel()
        phase = .summoned
        scheduleAutoHide()
    }

    func immerse() {
        hideTask?.cancel()
        phase = .resting
    }

    func toggle() {
        if phase == .resting { summon() } else { immerse() }
    }

    func noteChromeInteraction() {
        if phase == .summoned { scheduleAutoHide() }
    }

    private func scheduleAutoHide() {
        hideTask?.cancel()
        hideTask = Task { @MainActor in
            try? await Task.sleep(for: .seconds(Self.autoHideSeconds))
            guard !Task.isCancelled else { return }
            phase = .resting
        }
    }
}

enum PaafektImmersiveChromeMotion {
    static func animation(reduceMotion: Bool) -> Animation {
        reduceMotion ? .linear(duration: 0.15) : .easeInOut(duration: 0.28)
    }

    static func transition(reduceMotion: Bool) -> AnyTransition {
        reduceMotion
            ? .opacity
            : .opacity.combined(with: .move(edge: .bottom))
    }
}

// MARK: - Resting affordances

struct PaafektImmersiveFaintBackButton: View {
    let action: () -> Void
    var restingOpacity: Double = 0.45

    var body: some View {
        Button(action: action) {
            Image(systemName: "chevron.left")
                .font(.system(size: 17, weight: .semibold))
                .foregroundStyle(Theme.Palette.textPrimary.opacity(restingOpacity))
                .frame(width: 36, height: 36)
                .background(Circle().fill(Theme.Palette.viewerCapsuleFill.opacity(0.55)))
                .overlay(Circle().stroke(Theme.Palette.hairline.opacity(0.6), lineWidth: 1))
        }
        .buttonStyle(.plain)
        .accessibilityLabel(L10n.Common.back)
    }
}

struct PaafektImmersiveGoldSummonButton: View {
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Image("PaafektIconChevronUp")
                .renderingMode(.template)
                .resizable()
                .scaledToFit()
                .frame(width: 22, height: 22)
                .foregroundStyle(Theme.Palette.accentText)
                .frame(width: 46, height: 46)
                .background(Circle().fill(Theme.Palette.accent))
                .shadow(color: .black.opacity(0.35), radius: 6, x: 0, y: 2)
        }
        .buttonStyle(.plain)
        .accessibilityLabel(L10n.RoomViewer.immersiveShowControls)
    }
}

struct PaafektImmersiveRestingMeasurementPill: View {
    let text: String

    var body: some View {
        Text(text)
            .font(Theme.Typo.caption())
            .foregroundStyle(Theme.Palette.textSecondary)
            .padding(.horizontal, Theme.Space.md)
            .padding(.vertical, Theme.Space.sm)
            .paafektGlassCapsuleSurface()
            .allowsHitTesting(false)
    }
}

/// Always-on exit while full-video segmentation is active — independent of resting vs summoned chrome.
struct PaafektSegmentationDoneButton: View {
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Text(L10n.RoomViewer.segmentationDone)
                .font(.system(size: 16, weight: .semibold))
                .foregroundStyle(Theme.Palette.accentText)
                .padding(.horizontal, Theme.Space.lg)
                .frame(height: 44)
                .background(Capsule().fill(Theme.Palette.accent))
                .shadow(color: .black.opacity(0.35), radius: 6, x: 0, y: 2)
        }
        .buttonStyle(.plain)
        .accessibilityLabel(L10n.RoomViewer.segmentationDone)
    }
}

// MARK: - Summoned toolbar (glass capsule + gold Fit/Capture)

struct PaafektImmersiveSummonedToolbar<NavContent: View, HeroContent: View>: View {
    @ObservedObject var chrome: PaafektViewerChromeController
    @ViewBuilder let navContent: () -> NavContent
    @ViewBuilder let heroContent: () -> HeroContent

    var body: some View {
        VStack(spacing: Theme.Space.sm) {
            Button {
                chrome.immerse()
            } label: {
                Text(L10n.RoomViewer.immersiveTapToHide)
                    .font(.caption2)
                    .foregroundStyle(Theme.Palette.textSecondary.opacity(0.85))
            }
            .buttonStyle(.plain)

            PaafektViewerToolbarCapsule {
                HStack(spacing: Theme.Space.sm) {
                    navContent()
                    Spacer(minLength: Theme.Space.sm)
                    heroContent()
                }
            }
        }
        .onTapGesture { chrome.noteChromeInteraction() }
    }
}

/// Compact gold action for summoned toolbar (icon + short label).
struct PaafektImmersiveCompactHeroAction: View {
    let assetName: String
    let title: String
    var isActive: Bool = false
    var isDisabled: Bool = false
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            VStack(spacing: 4) {
                Image(assetName)
                    .renderingMode(.template)
                    .resizable()
                    .scaledToFit()
                    .frame(width: 18, height: 18)
                Text(title)
                    .font(.system(size: 11, weight: .semibold))
                    .lineLimit(1)
                    .minimumScaleFactor(0.8)
            }
            .foregroundStyle(Theme.Palette.accent)
            .padding(.horizontal, Theme.Space.md)
            .padding(.vertical, Theme.Space.sm)
            .background(
                RoundedRectangle(cornerRadius: Theme.Space.md, style: .continuous)
                    .stroke(Theme.Palette.accent, lineWidth: 1.5)
                    .background(
                        RoundedRectangle(cornerRadius: Theme.Space.md, style: .continuous)
                            .fill(isActive ? Theme.Palette.accent.opacity(0.18) : Color.clear)
                    )
            )
            .opacity(isDisabled ? 0.5 : 1)
        }
        .buttonStyle(.plain)
        .disabled(isDisabled)
        .accessibilityLabel(title)
    }
}

// MARK: - Tap-to-summon on room content (does not block orbit/pinch)

/// Single-finger tap on the room layer summons chrome while resting; pinch/pan stay on the viewer below.
struct PaafektImmersiveRoomSummonTapModifier: ViewModifier {
    @ObservedObject var chrome: PaafektViewerChromeController
    var enabled: Bool
    var hideForCapture: Bool
    var onRestingTap: (() -> Void)?

    func body(content: Content) -> some View {
        if enabled, !hideForCapture {
            content.simultaneousGesture(
                TapGesture().onEnded {
                    guard chrome.isResting else { return }
                    if let onRestingTap {
                        onRestingTap()
                    } else {
                        chrome.summon()
                    }
                }
            )
        } else {
            content
        }
    }
}

extension View {
    func paafektImmersiveRoomSummonTap(
        chrome: PaafektViewerChromeController,
        enabled: Bool = true,
        hideForCapture: Bool = false,
        onRestingTap: (() -> Void)? = nil
    ) -> some View {
        modifier(
            PaafektImmersiveRoomSummonTapModifier(
                chrome: chrome,
                enabled: enabled,
                hideForCapture: hideForCapture,
                onRestingTap: onRestingTap
            )
        )
    }
}

// MARK: - Tap-to-toggle layer (legacy — prefer paafektImmersiveRoomSummonTap on room content)

struct PaafektImmersiveTapToggleLayer: View {
    @ObservedObject var chrome: PaafektViewerChromeController
    var enabled: Bool = true

    var body: some View {
        Group {
            if enabled {
                Color.clear
                    .contentShape(Rectangle())
                    .onTapGesture { chrome.toggle() }
            }
        }
        .allowsHitTesting(enabled)
    }
}

// MARK: - Full immersive chrome stack

struct PaafektImmersiveViewerChromeStack<
    SummonedToolbar: View,
    SummonedExtras: View,
    RestingAccessory: View,
    PersistentOverlay: View
>: View {
    @ObservedObject var chrome: PaafektViewerChromeController
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    let onBack: () -> Void
    var measurementText: String? = nil
    var hideForCapture: Bool = false
    @ViewBuilder let summonedToolbar: () -> SummonedToolbar
    @ViewBuilder let summonedExtras: () -> SummonedExtras
    @ViewBuilder let restingAccessory: () -> RestingAccessory
    @ViewBuilder let persistentOverlay: () -> PersistentOverlay

    var body: some View {
        ZStack {
            Color.clear
                .allowsHitTesting(false)
        }
        .overlay(alignment: .topLeading) {
            PaafektImmersiveFaintBackButton(action: onBack)
                .opacity(chrome.isResting ? 1 : 0.92)
                .padding(.horizontal, Theme.Space.lg)
                .padding(.top, Theme.Space.sm)
        }
        .overlay(alignment: .topTrailing) {
            persistentOverlay()
                .padding(.horizontal, Theme.Space.lg)
                .padding(.top, Theme.Space.sm)
        }
        .overlay(alignment: .bottom) {
            VStack(spacing: Theme.Space.sm) {
                if chrome.isSummoned {
                    summonedExtras()
                        .transition(PaafektImmersiveChromeMotion.transition(reduceMotion: reduceMotion))
                }
                if chrome.isResting {
                    restingAccessory()
                        .frame(maxWidth: .infinity)
                        .transition(.opacity)
                }
                if let measurementText, chrome.isResting {
                    PaafektImmersiveRestingMeasurementPill(text: measurementText)
                        .transition(.opacity)
                }
                HStack {
                    Spacer(minLength: 0)
                    if chrome.isResting {
                        PaafektImmersiveGoldSummonButton { chrome.summon() }
                            .transition(PaafektImmersiveChromeMotion.transition(reduceMotion: reduceMotion))
                    }
                }
                if chrome.isSummoned {
                    summonedToolbar()
                        .transition(PaafektImmersiveChromeMotion.transition(reduceMotion: reduceMotion))
                }
            }
            .padding(.horizontal, Theme.Space.lg)
            .padding(.bottom, Theme.Space.lg)
            .animation(PaafektImmersiveChromeMotion.animation(reduceMotion: reduceMotion), value: chrome.phase)
        }
        .opacity(hideForCapture ? 0 : 1)
        .allowsHitTesting(!hideForCapture)
    }
}

extension PaafektImmersiveViewerChromeStack where RestingAccessory == EmptyView, PersistentOverlay == EmptyView {
    init(
        chrome: PaafektViewerChromeController,
        onBack: @escaping () -> Void,
        measurementText: String? = nil,
        hideForCapture: Bool = false,
        @ViewBuilder summonedToolbar: @escaping () -> SummonedToolbar,
        @ViewBuilder summonedExtras: @escaping () -> SummonedExtras
    ) {
        self.chrome = chrome
        self.onBack = onBack
        self.measurementText = measurementText
        self.hideForCapture = hideForCapture
        self.summonedToolbar = summonedToolbar
        self.summonedExtras = summonedExtras
        self.restingAccessory = { EmptyView() }
        self.persistentOverlay = { EmptyView() }
    }
}
