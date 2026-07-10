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
