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

/// Capture-only hero action for legacy / suppress-built-in-top-chrome toolbars.
struct PaafektViewerCaptureHeroButton: View {
    var isDisabled: Bool = false
    let action: () -> Void

    var body: some View {
        PaafektViewerHeroButton(
            assetName: "PaafektIconSnapshot",
            title: L10n.RoomViewer.heroCapture,
            isDisabled: isDisabled,
            action: action
        )
    }
}

/// Shared resting-pill / ruler-hint copy for room measurement chrome.
enum PaafektRoomMeasurementDisplay {
    /// Saved AI rooms show labeled height; footprint rooms keep W×D when height is not emphasized.
    static func restingPillText(
        width: Float,
        height: Float,
        depth: Float,
        emphasizeHeight: Bool
    ) -> String? {
        if emphasizeHeight, height > 0.05, height.isFinite {
            return L10n.RoomViewer.approximateRoomHeight(height)
        }
        if width > 0.05, depth > 0.05, width.isFinite, depth.isFinite {
            return String(format: "%.1f m × %.1f m", width, depth)
        }
        if height > 0.05, height.isFinite {
            return L10n.RoomViewer.approximateRoomHeight(height)
        }
        return nil
    }

    static func rulerHintText(
        width: Float,
        height: Float,
        depth: Float,
        showFullWHD: Bool
    ) -> String? {
        guard height > 0.05, height.isFinite else { return nil }
        if showFullWHD,
           width > 0.05, depth > 0.05,
           width.isFinite, depth.isFinite {
            return L10n.RoomViewer.roomDimensionsWHDNearAccurateChip(
                width: width,
                height: height,
                depth: depth
            )
        }
        return L10n.RoomViewer.approximateRoomHeight(height)
    }
}
