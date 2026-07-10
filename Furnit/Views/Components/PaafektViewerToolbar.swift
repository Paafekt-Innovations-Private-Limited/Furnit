import SwiftUI

// MARK: - Shared viewer toolbar chrome (glass capsule + gold active)

struct PaafektViewerToolbarCapsule<Content: View>: View {
    @ViewBuilder let content: () -> Content

    var body: some View {
        content()
            .padding(.horizontal, 14)
            .padding(.vertical, 10)
            .paafektGlassCapsuleSurface()
    }
}

struct PaafektViewerToolbarIconButton: View {
    let systemName: String
    var isActive: Bool = false
    var fontSize: CGFloat = 18
    let accessibilityLabel: String
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Image(systemName: systemName)
                .font(.system(size: fontSize, weight: .medium))
                .foregroundStyle(isActive ? Theme.Palette.accent : Theme.Palette.textPrimary)
                .frame(width: isActive ? 28 : nil, height: isActive ? 28 : nil)
                .background {
                    if isActive {
                        Circle().fill(Theme.Palette.accent.opacity(0.22))
                    }
                }
        }
        .buttonStyle(.plain)
        .accessibilityLabel(accessibilityLabel)
    }
}

/// Primary bottom-bar actions — gold capsule with monoline icon + label (mockup hierarchy).
struct PaafektViewerHeroButton: View {
    let assetName: String
    let title: String
    var isActive: Bool = false
    var isDisabled: Bool = false
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: Theme.Space.sm) {
                Image(assetName)
                    .renderingMode(.template)
                    .resizable()
                    .scaledToFit()
                    .frame(width: 18, height: 18)
                Text(title)
                    .font(Theme.Typo.headline())
                    .lineLimit(1)
                    .minimumScaleFactor(0.85)
            }
            .foregroundStyle(Theme.Palette.accentText)
            .padding(.horizontal, Theme.Space.lg)
            .padding(.vertical, Theme.Space.md)
            .frame(maxWidth: .infinity)
            .background(
                Capsule().fill(isActive ? Theme.Palette.accentPressed : Theme.Palette.accent)
            )
            .opacity(isDisabled ? 0.5 : 1)
        }
        .buttonStyle(.plain)
        .disabled(isDisabled)
        .accessibilityLabel(title)
    }
}

/// Side-by-side hero actions anchored above the home indicator.
struct PaafektViewerHeroActionsBar: View {
    var fitActive: Bool = false
    var fitDisabled: Bool = false
    var captureDisabled: Bool = false
    let onFit: () -> Void
    let onCapture: () -> Void

    var body: some View {
        HStack(spacing: Theme.Space.md) {
            PaafektViewerHeroButton(
                assetName: "PaafektIconAI",
                title: L10n.RoomViewer.heroFitFurniture,
                isActive: fitActive,
                isDisabled: fitDisabled,
                action: onFit
            )
            PaafektViewerHeroButton(
                assetName: "PaafektIconSnapshot",
                title: L10n.RoomViewer.heroCapture,
                isDisabled: captureDisabled,
                action: onCapture
            )
        }
    }
}

/// Bottom-bar AI / snapshot actions — Paafekt monoline assets, gold when active.
struct PaafektViewerBottomActionButton: View {
    let assetName: String
    var isActive: Bool = false
    var isDisabled: Bool = false
    let accessibilityLabel: String
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Image(assetName)
                .renderingMode(.template)
                .resizable()
                .scaledToFit()
                .frame(width: 22, height: 22)
                .foregroundStyle(isActive ? Theme.Palette.accent : Theme.Palette.textPrimary)
                .frame(width: 44, height: 44)
                .background(Circle().fill(Theme.Palette.viewerCapsuleFill))
                .overlay(Circle().stroke(Theme.Palette.hairline, lineWidth: 1))
                .opacity(isDisabled ? 0.5 : 1)
        }
        .buttonStyle(.plain)
        .disabled(isDisabled)
        .accessibilityLabel(accessibilityLabel)
    }
}

struct PaafektViewerBackButton: View {
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Image(systemName: "chevron.left")
                .font(.system(size: 18, weight: .semibold))
                .foregroundStyle(Theme.Palette.textPrimary)
                .frame(width: 36, height: 36)
                .background(Circle().fill(Theme.Palette.viewerCapsuleFill))
                .overlay(Circle().stroke(Theme.Palette.hairline, lineWidth: 1))
        }
        .buttonStyle(.plain)
    }
}
