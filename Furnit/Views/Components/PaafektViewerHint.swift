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

// MARK: - Glass hint chip (primary viewer helper pattern)

/// Glass capsule + gold monoline gesture icon + one short caption line.
/// Anchor above the toolbar; pair with existing auto-dismiss task logic in each viewer.
struct PaafektHintChip: View {
    let systemImage: String
    let text: String
    var maxWidth: CGFloat = 260
    var alignment: HorizontalAlignment = .leading

    var body: some View {
        HStack(alignment: .center, spacing: Theme.Space.sm) {
            Image(systemName: systemImage)
                .font(.system(size: 15, weight: .medium))
                .foregroundStyle(Theme.Palette.accent)
                .frame(width: 22, height: 22)

            Text(text)
                .font(Theme.Typo.caption())
                .foregroundStyle(Theme.Palette.textPrimary)
                .multilineTextAlignment(alignment == .leading ? .leading : .trailing)
                .fixedSize(horizontal: false, vertical: true)
        }
        .frame(maxWidth: maxWidth, alignment: frameAlignment)
        .padding(.horizontal, Theme.Space.md)
        .padding(.vertical, Theme.Space.sm)
        .paafektGlassCapsuleSurface()
        .accessibilityElement(children: .combine)
    }

    private var frameAlignment: Alignment {
        switch alignment {
        case .trailing: return .trailing
        case .center: return .center
        default: return .leading
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

// MARK: - First-run coach mark variant

/// Glass chip plus a gold "Got it" dismiss button for explicit first-run acknowledgement.
struct PaafektHintCoachMark: View {
    let systemImage: String
    let text: String
    let confirmTitle: String
    let onConfirm: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: Theme.Space.sm) {
            PaafektHintChip(systemImage: systemImage, text: text)

            Button(action: onConfirm) {
                Text(confirmTitle)
                    .font(Theme.Typo.headline())
                    .foregroundStyle(Theme.Palette.accentText)
                    .padding(.horizontal, Theme.Space.lg)
                    .padding(.vertical, Theme.Space.sm)
                    .background(Capsule().fill(Theme.Palette.accent))
            }
            .buttonStyle(.plain)
            .accessibilityLabel(confirmTitle)
        }
    }
}
