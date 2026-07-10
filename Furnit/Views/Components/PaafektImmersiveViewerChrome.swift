import SwiftUI
import UIKit

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

// MARK: - Tap-to-summon (passthrough — does not block drag/pinch on the room)

/// UIKit tap recognizer with `cancelsTouchesInView = false` so WebGL/RealityKit gestures keep working.
struct PaafektImmersiveRoomTapToSummonOverlay: UIViewRepresentable {
    @ObservedObject var chrome: PaafektViewerChromeController
    var enabled: Bool

    func makeUIView(context: Context) -> PassthroughSummonTapView {
        let view = PassthroughSummonTapView()
        view.onTap = { chrome.summon() }
        return view
    }

    func updateUIView(_ uiView: PassthroughSummonTapView, context: Context) {
        uiView.isUserInteractionEnabled = enabled && chrome.isResting
        uiView.onTap = { chrome.summon() }
    }
}

final class PassthroughSummonTapView: UIView, UIGestureRecognizerDelegate {
    var onTap: (() -> Void)?

    override init(frame: CGRect) {
        super.init(frame: frame)
        backgroundColor = .clear
        isMultipleTouchEnabled = false
        let tap = UITapGestureRecognizer(target: self, action: #selector(handleTap))
        tap.cancelsTouchesInView = false
        tap.delegate = self
        addGestureRecognizer(tap)
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) { nil }

    @objc private func handleTap() {
        onTap?()
    }

    func gestureRecognizer(
        _ gestureRecognizer: UIGestureRecognizer,
        shouldRecognizeSimultaneouslyWith otherGestureRecognizer: UIGestureRecognizer
    ) -> Bool {
        true
    }
}

// MARK: - Tap-to-toggle layer (legacy — prefer passthrough summon overlay)

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

struct PaafektImmersiveViewerChromeStack<SummonedToolbar: View, SummonedExtras: View>: View {
    @ObservedObject var chrome: PaafektViewerChromeController
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    let onBack: () -> Void
    var measurementText: String? = nil
    var tapToSummonEnabled: Bool = true
    var hideForCapture: Bool = false
    @ViewBuilder let summonedToolbar: () -> SummonedToolbar
    @ViewBuilder let summonedExtras: () -> SummonedExtras

    var body: some View {
        ZStack {
            PaafektImmersiveRoomTapToSummonOverlay(
                chrome: chrome,
                enabled: tapToSummonEnabled && !hideForCapture
            )

            VStack {
                HStack {
                    PaafektImmersiveFaintBackButton(action: onBack)
                        .opacity(chrome.isResting ? 1 : 0.92)
                    Spacer()
                }
                .padding(.horizontal, Theme.Space.lg)
                .padding(.top, Theme.Space.sm)
                Spacer()
            }
            .allowsHitTesting(true)

            VStack {
                Spacer()
                if chrome.isSummoned {
                    summonedExtras()
                        .transition(PaafektImmersiveChromeMotion.transition(reduceMotion: reduceMotion))
                }
                if let measurementText, chrome.isResting {
                    PaafektImmersiveRestingMeasurementPill(text: measurementText)
                        .padding(.bottom, Theme.Space.sm)
                        .transition(.opacity)
                }
                HStack {
                    Spacer()
                    if chrome.isResting {
                        PaafektImmersiveGoldSummonButton { chrome.summon() }
                            .transition(PaafektImmersiveChromeMotion.transition(reduceMotion: reduceMotion))
                    }
                }
                .padding(.horizontal, Theme.Space.lg)
                .padding(.bottom, Theme.Space.lg)

                if chrome.isSummoned {
                    summonedToolbar()
                        .padding(.horizontal, Theme.Space.lg)
                        .padding(.bottom, Theme.Space.lg)
                        .transition(PaafektImmersiveChromeMotion.transition(reduceMotion: reduceMotion))
                }
            }
            .animation(PaafektImmersiveChromeMotion.animation(reduceMotion: reduceMotion), value: chrome.phase)
        }
        .opacity(hideForCapture ? 0 : 1)
        .allowsHitTesting(!hideForCapture)
    }
}
