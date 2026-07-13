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

/// Secondary summon affordance — quieter than the persistent Fit FAB (glass disk, bottom-leading).
struct PaafektImmersiveQuietSummonButton: View {
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Image("PaafektIconChevronUp")
                .renderingMode(.template)
                .resizable()
                .scaledToFit()
                .frame(width: 18, height: 18)
                .foregroundStyle(Theme.Palette.textSecondary)
                .frame(width: 38, height: 38)
                .background(Circle().fill(Theme.Palette.viewerCapsuleFill.opacity(0.72)))
                .overlay(Circle().stroke(Theme.Palette.hairline.opacity(0.6), lineWidth: 1))
        }
        .buttonStyle(.plain)
        .accessibilityLabel(L10n.RoomViewer.immersiveShowControls)
    }
}

/// Morphing persistent primary action — Fit → Segment → Done on one gold button.
enum PaafektMorphingPrimaryAction: Equatable {
    case fitEnter
    case fitExitActive
    case segment
    case done
}

enum PaafektMorphingPrimaryActionResolver {
    static func resolve(
        showingFurnitureFit: Bool,
        showFullVideoWithIdentifications: Bool,
        segmentationMode: FurnitureFitSegmentationMode,
        hasSelectedObject: Bool
    ) -> PaafektMorphingPrimaryAction {
        guard showingFurnitureFit else { return .fitEnter }
        if segmentationMode == .segmentSelected { return .done }
        if showFullVideoWithIdentifications, hasSelectedObject { return .segment }
        return .fitExitActive
    }
}

enum PaafektMorphingPrimaryActionHandler {
    static func perform(
        _ action: PaafektMorphingPrimaryAction,
        enterFit: () -> Void,
        exitFit: () -> Void,
        segment: () -> Void,
        finishSegmentation: () -> Void
    ) {
        switch action {
        case .fitEnter:
            enterFit()
        case .fitExitActive:
            exitFit()
        case .segment:
            segment()
        case .done:
            finishSegmentation()
        }
    }
}

/// Tier-0 persistent primary — morphs Fit / Segment / Done; always visible, never auto-hides.
struct PaafektMorphingPrimaryFAB: View {
    let action: PaafektMorphingPrimaryAction
    var isDisabled: Bool = false
    let onTap: () -> Void

    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    private var showsFitIcon: Bool {
        switch action {
        case .fitEnter, .fitExitActive:
            return true
        case .segment, .done:
            return false
        }
    }

    private var labelText: String {
        switch action {
        case .fitEnter, .fitExitActive:
            return L10n.RoomViewer.immersiveFitShort
        case .segment:
            return L10n.RoomViewer.segmentFurnitureAction
        case .done:
            return L10n.RoomViewer.segmentationDone
        }
    }

    private var isActiveHighlight: Bool {
        action == .fitExitActive
    }

    private var accessibilityText: String {
        switch action {
        case .fitEnter, .fitExitActive:
            return L10n.RoomViewer.heroFitFurniture
        case .segment:
            return L10n.RoomViewer.segmentFurnitureAccessibility
        case .done:
            return L10n.RoomViewer.segmentationDone
        }
    }

    var body: some View {
        Button(action: onTap) {
            HStack(spacing: Theme.Space.sm) {
                if showsFitIcon {
                    Image("PaafektIconAI")
                        .renderingMode(.template)
                        .resizable()
                        .scaledToFit()
                        .frame(width: 18, height: 18)
                }
                Text(labelText)
                    .font(.system(size: 16, weight: .semibold))
                    .lineLimit(1)
            }
            .foregroundStyle(Theme.Palette.accentText)
            .padding(.horizontal, Theme.Space.lg)
            .frame(height: 44)
            .background(
                Capsule().fill(isActiveHighlight ? Theme.Palette.accentPressed : Theme.Palette.accent)
            )
            .shadow(color: .black.opacity(0.35), radius: 6, x: 0, y: 2)
            .opacity(isDisabled ? 0.5 : 1)
        }
        .buttonStyle(.plain)
        .disabled(isDisabled)
        .accessibilityLabel(accessibilityText)
        .animation(PaafektImmersiveChromeMotion.animation(reduceMotion: reduceMotion), value: action)
    }
}

/// Tier-0 persistent Fit action — always visible, bottom-trailing, above room and camera overlays.
struct PaafektImmersiveFitFAB: View {
    var isActive: Bool = false
    var isDisabled: Bool = false
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: Theme.Space.sm) {
                Image("PaafektIconAI")
                    .renderingMode(.template)
                    .resizable()
                    .scaledToFit()
                    .frame(width: 18, height: 18)
                Text(L10n.RoomViewer.immersiveFitShort)
                    .font(.system(size: 16, weight: .semibold))
                    .lineLimit(1)
            }
            .foregroundStyle(Theme.Palette.accentText)
            .padding(.horizontal, Theme.Space.lg)
            .frame(height: 44)
            .background(
                Capsule().fill(isActive ? Theme.Palette.accentPressed : Theme.Palette.accent)
            )
            .shadow(color: .black.opacity(0.35), radius: 6, x: 0, y: 2)
            .opacity(isDisabled ? 0.5 : 1)
        }
        .buttonStyle(.plain)
        .disabled(isDisabled)
        .accessibilityLabel(L10n.RoomViewer.heroFitFurniture)
    }
}

/// Tier-0 persistent Save action — creation flow only, same gold treatment as Fit.
struct PaafektImmersiveSaveFAB: View {
    var isDisabled: Bool = false
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: Theme.Space.sm) {
                Image(systemName: "square.and.arrow.down")
                    .font(.system(size: 16, weight: .semibold))
                Text(L10n.Common.save)
                    .font(.system(size: 16, weight: .semibold))
                    .lineLimit(1)
            }
            .foregroundStyle(Theme.Palette.accentText)
            .padding(.horizontal, Theme.Space.lg)
            .frame(height: 44)
            .background(Capsule().fill(Theme.Palette.accent))
            .shadow(color: .black.opacity(0.35), radius: 6, x: 0, y: 2)
            .opacity(isDisabled ? 0.5 : 1)
        }
        .buttonStyle(.plain)
        .disabled(isDisabled)
        .accessibilityLabel(L10n.RoomViewer.saveRoom)
    }
}

struct PaafektImmersiveRestingMeasurementPill: View {
    let text: String

    var body: some View {
        Text(text)
            .font(Theme.Typo.caption())
            .foregroundStyle(Theme.Palette.textSecondary)
            .lineLimit(1)
            .fixedSize(horizontal: true, vertical: false)
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

/// Top-trailing Done for Furniture Fit — default brain flow (segmentPrimary) and full-video segment mode.
struct PaafektFurnitureFitDonePersistentOverlay: View {
    let showingFurnitureFit: Bool
    let showFullVideoWithIdentifications: Bool
    let segmentationMode: FurnitureFitSegmentationMode
    let viewerLabel: String
    let onExitFullVideoSegmentation: () -> Void
    let onExitFurnitureFit: () -> Void

    private var isActive: Bool {
        guard showingFurnitureFit else { return false }
        if showFullVideoWithIdentifications {
            return segmentationMode == .segmentSelected
        }
        return true
    }

    var body: some View {
        Group {
            if isActive {
                PaafektSegmentationDoneButton {
                    if showFullVideoWithIdentifications, segmentationMode == .segmentSelected {
                        onExitFullVideoSegmentation()
                    } else {
                        onExitFurnitureFit()
                    }
                }
            }
        }
        #if DEBUG
        .onChange(of: isActive) { _, active in
            logDebug(
                "SEGMENT_DONE: persistent visible=\(active) fit=\(showingFurnitureFit) " +
                "fullVideo=\(showFullVideoWithIdentifications) mode=\(segmentationMode) viewer=\(viewerLabel)"
            )
        }
        #endif
    }
}

/// Renders above UIKit `FurnitureFitUIView` (z ~9000). SwiftUI chrome at ~99998 can sit under the camera layer.
struct PaafektFullVideoSegmentationExitLayer: View {
    let isActive: Bool
    let viewerLabel: String
    let onDone: () -> Void

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 0) {
                Spacer(minLength: 0)
                if isActive {
                    PaafektSegmentationDoneButton(action: onDone)
                }
            }
            .padding(.horizontal, Theme.Space.lg)
            .padding(.top, Theme.Space.sm)
            Spacer(minLength: 0)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topTrailing)
        .allowsHitTesting(isActive)
        #if DEBUG
        .onChange(of: isActive) { _, active in
            logDebug("SEGMENT_DONE: overlay visible=\(active) viewer=\(viewerLabel)")
        }
        .onAppear {
            if isActive {
                logDebug("SEGMENT_DONE: overlay mounted visible=true viewer=\(viewerLabel)")
            }
        }
        #endif
    }
}

enum PaafektFullVideoSegmentationExitDiagnostics {
    static func logModeChange(
        viewer: String,
        mode: FurnitureFitSegmentationMode,
        showingFurnitureFit: Bool,
        showFullVideoWithIdentifications: Bool
    ) {
        #if DEBUG
        logDebug(
            "SEGMENT_DONE: mode=\(mode) fit=\(showingFurnitureFit) " +
            "fullVideo=\(showFullVideoWithIdentifications) viewer=\(viewer)"
        )
        #endif
    }
}

// MARK: - Summoned toolbar (glass capsule + Capture; Fit is persistent FAB)

struct PaafektImmersiveSummonedToolbar<NavContent: View, HeroContent: View>: View {
    @ObservedObject var chrome: PaafektViewerChromeController
    @ViewBuilder let navContent: () -> NavContent
    @ViewBuilder let heroContent: () -> HeroContent

    var body: some View {
        PaafektViewerToolbarCapsule {
            HStack(spacing: Theme.Space.md) {
                navContent()
                Spacer(minLength: Theme.Space.sm)
                heroContent()
            }
        }
        .fixedSize(horizontal: false, vertical: true)
        .onTapGesture { chrome.noteChromeInteraction() }
    }
}

/// Compact gold action for summoned toolbar — icon-only to avoid truncation in the glass capsule.
struct PaafektImmersiveCompactHeroAction: View {
    let assetName: String
    let title: String
    var isActive: Bool = false
    var isDisabled: Bool = false
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Image(assetName)
                .renderingMode(.template)
                .resizable()
                .scaledToFit()
                .frame(width: 20, height: 20)
                .foregroundStyle(Theme.Palette.accent)
                .frame(width: 40, height: 40)
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
    let morphingPrimaryAction: PaafektMorphingPrimaryAction
    let onMorphingPrimary: () -> Void
    var morphingPrimaryDisabled: Bool = false
    var onSave: (() -> Void)? = nil
    var saveDisabled: Bool = false
    var measurementText: String? = nil
    var hideForCapture: Bool = false
    @ViewBuilder let summonedToolbar: () -> SummonedToolbar
    @ViewBuilder let summonedExtras: () -> SummonedExtras
    @ViewBuilder let restingAccessory: () -> RestingAccessory
    @ViewBuilder let persistentOverlay: () -> PersistentOverlay

    private var persistentActionsTrailingReserve: CGFloat {
        let buttonWidth: CGFloat = 88
        let showsSave = onSave != nil
        return showsSave ? buttonWidth * 2 + Theme.Space.sm : buttonWidth
    }

    var body: some View {
        ZStack(alignment: .bottomTrailing) {
            Color.clear
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .allowsHitTesting(false)
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
                .overlay(alignment: .bottomLeading) {
                    if chrome.isResting {
                        PaafektImmersiveQuietSummonButton { chrome.summon() }
                            .padding(.horizontal, Theme.Space.lg)
                            .padding(.bottom, Theme.Space.lg)
                            .transition(PaafektImmersiveChromeMotion.transition(reduceMotion: reduceMotion))
                    }
                }
                .overlay(alignment: .bottom) {
                    VStack(spacing: Theme.Space.md) {
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
                        if chrome.isSummoned {
                            summonedToolbar()
                                .transition(PaafektImmersiveChromeMotion.transition(reduceMotion: reduceMotion))
                        }
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.horizontal, Theme.Space.lg)
                    .padding(.bottom, Theme.Space.lg)
                    .padding(.trailing, max(persistentActionsTrailingReserve, 88))
                    .animation(PaafektImmersiveChromeMotion.animation(reduceMotion: reduceMotion), value: chrome.phase)
                }
                .opacity(hideForCapture ? 0 : 1)
                .allowsHitTesting(!hideForCapture)

            HStack(spacing: Theme.Space.sm) {
                PaafektMorphingPrimaryFAB(
                    action: morphingPrimaryAction,
                    isDisabled: morphingPrimaryDisabled,
                    onTap: onMorphingPrimary
                )
                if let onSave {
                    PaafektImmersiveSaveFAB(
                        isDisabled: saveDisabled,
                        action: onSave
                    )
                }
            }
            .padding(.horizontal, Theme.Space.lg)
            .padding(.bottom, Theme.Space.lg)
        }
    }
}

extension PaafektImmersiveViewerChromeStack where RestingAccessory == EmptyView, PersistentOverlay == EmptyView {
    init(
        chrome: PaafektViewerChromeController,
        onBack: @escaping () -> Void,
        morphingPrimaryAction: PaafektMorphingPrimaryAction,
        onMorphingPrimary: @escaping () -> Void,
        morphingPrimaryDisabled: Bool = false,
        onSave: (() -> Void)? = nil,
        saveDisabled: Bool = false,
        measurementText: String? = nil,
        hideForCapture: Bool = false,
        @ViewBuilder summonedToolbar: @escaping () -> SummonedToolbar,
        @ViewBuilder summonedExtras: @escaping () -> SummonedExtras
    ) {
        self.chrome = chrome
        self.onBack = onBack
        self.morphingPrimaryAction = morphingPrimaryAction
        self.onMorphingPrimary = onMorphingPrimary
        self.morphingPrimaryDisabled = morphingPrimaryDisabled
        self.onSave = onSave
        self.saveDisabled = saveDisabled
        self.measurementText = measurementText
        self.hideForCapture = hideForCapture
        self.summonedToolbar = summonedToolbar
        self.summonedExtras = summonedExtras
        self.restingAccessory = { EmptyView() }
        self.persistentOverlay = { EmptyView() }
    }
}
