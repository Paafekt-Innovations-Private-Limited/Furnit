import SwiftUI

// MARK: - Paafekt design tokens (single source of truth)

enum Theme {

    enum Palette {
        static let background = Color(hex: 0x0E0F12)
        static let surface = Color(hex: 0x1A1C20)
        static let surfaceHi = Color(hex: 0x24272D)
        static let hairline = Color.white.opacity(0.08)
        static let textPrimary = Color(hex: 0xF4F3EF)
        static let textSecondary = Color(hex: 0x9BA0A8)
        static let accent = Color(hex: 0xC9A24B)
        static let accentPressed = Color(hex: 0xA9853A)
        static let accentText = Color(hex: 0x0E0F12)
        static let success = Color(hex: 0x3E9E6E)
        static let danger = Color(hex: 0xC85A54)
        static let glassTint = Color.black.opacity(0.45)
        static let viewerCapsuleFill = Color.black.opacity(0.55)
    }

    enum Space {
        static let xs: CGFloat = 4
        static let sm: CGFloat = 8
        static let md: CGFloat = 12
        static let lg: CGFloat = 16
        static let xl: CGFloat = 24
        static let xxl: CGFloat = 32
    }

    enum Radius {
        static let control: CGFloat = 12
        static let sheet: CGFloat = 20
    }

    enum Typo {
        static func display() -> Font { .system(size: 34, weight: .bold) }
        static func title() -> Font { .system(size: 24, weight: .semibold) }
        static func headline() -> Font { .system(size: 17, weight: .semibold) }
        static func body() -> Font { .system(size: 15, weight: .regular) }
        static func caption() -> Font { .system(size: 12, weight: .regular) }
        static func tag() -> Font { .system(size: 11, weight: .semibold, design: .monospaced) }
    }
}

// MARK: - Color hex helper

extension Color {
    init(hex: UInt32, alpha: Double = 1.0) {
        let red = Double((hex >> 16) & 0xFF) / 255.0
        let green = Double((hex >> 8) & 0xFF) / 255.0
        let blue = Double(hex & 0xFF) / 255.0
        self.init(.sRGB, red: red, green: green, blue: blue, opacity: alpha)
    }
}

// MARK: - View modifiers

extension View {
    func paafektScreenBackground() -> some View {
        background(Theme.Palette.background.ignoresSafeArea())
    }

    func paafektCardSurface() -> some View {
        background(Theme.Palette.surface)
            .clipShape(RoundedRectangle(cornerRadius: Theme.Radius.control, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: Theme.Radius.control, style: .continuous)
                    .stroke(Theme.Palette.hairline, lineWidth: 1)
            )
    }

    func paafektIconTile(size: CGFloat = 40) -> some View {
        frame(width: size, height: size)
            .background(Theme.Palette.surfaceHi)
            .clipShape(RoundedRectangle(cornerRadius: Theme.Radius.control, style: .continuous))
    }
}

// MARK: - Button styles

struct PrimaryButtonStyle: ButtonStyle {
    @Environment(\.isEnabled) private var isEnabled

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(Theme.Typo.headline())
            .foregroundStyle(isEnabled ? Theme.Palette.accentText : Theme.Palette.textSecondary)
            .padding(.horizontal, Theme.Space.lg)
            .padding(.vertical, Theme.Space.md)
            .frame(maxWidth: .infinity)
            .background(
                RoundedRectangle(cornerRadius: Theme.Radius.control, style: .continuous)
                    .fill(isEnabled
                          ? (configuration.isPressed ? Theme.Palette.accentPressed : Theme.Palette.accent)
                          : Theme.Palette.surfaceHi)
            )
            .scaleEffect(configuration.isPressed && isEnabled ? 0.98 : 1.0)
            .animation(.easeInOut(duration: 0.15), value: configuration.isPressed)
    }
}

struct SecondaryButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(Theme.Typo.headline())
            .foregroundStyle(Theme.Palette.textPrimary)
            .padding(.horizontal, Theme.Space.lg)
            .padding(.vertical, Theme.Space.md)
            .frame(maxWidth: .infinity)
            .background(
                RoundedRectangle(cornerRadius: Theme.Radius.control, style: .continuous)
                    .fill(Theme.Palette.surface)
            )
            .overlay(
                RoundedRectangle(cornerRadius: Theme.Radius.control, style: .continuous)
                    .stroke(Theme.Palette.hairline, lineWidth: 1)
            )
            .scaleEffect(configuration.isPressed ? 0.98 : 1.0)
            .animation(.easeInOut(duration: 0.15), value: configuration.isPressed)
    }
}

struct PaafektCreationCardStyle: ButtonStyle {
    enum Variant { case primary, secondary }

    let variant: Variant

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .padding(Theme.Space.lg)
            .background(
                RoundedRectangle(cornerRadius: Theme.Radius.control, style: .continuous)
                    .fill(variant == .primary ? Theme.Palette.accent.opacity(0.12) : Theme.Palette.surface)
            )
            .overlay(
                RoundedRectangle(cornerRadius: Theme.Radius.control, style: .continuous)
                    .stroke(
                        variant == .primary ? Theme.Palette.accent : Theme.Palette.hairline,
                        lineWidth: variant == .primary ? 1.5 : 1
                    )
            )
            .scaleEffect(configuration.isPressed ? 0.98 : 1.0)
    }
}
