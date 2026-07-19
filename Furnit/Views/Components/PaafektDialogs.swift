import SwiftUI

// MARK: - Building room (generation progress)

/// Dark dialog with gold monoline icon, gold progress ring, warm subtext.
struct PaafektBuildingRoomOverlay: View {
    let progress: Double
    var statusMessage: String?

    var body: some View {
        ZStack {
            Theme.Palette.background.opacity(0.88)
                .ignoresSafeArea()

            VStack(spacing: Theme.Space.lg) {
                ZStack {
                    Circle()
                        .stroke(Theme.Palette.accent.opacity(0.25), lineWidth: 6)
                        .frame(width: 72, height: 72)
                    Circle()
                        .trim(from: 0, to: CGFloat(min(max(progress, 0), 1)))
                        .stroke(
                            Theme.Palette.accent,
                            style: StrokeStyle(lineWidth: 6, lineCap: .round)
                        )
                        .frame(width: 72, height: 72)
                        .rotationEffect(.degrees(-90))
                        .animation(.easeInOut(duration: 0.3), value: progress)
                    Image(systemName: "house")
                        .font(.system(size: 26, weight: .regular))
                        .foregroundStyle(Theme.Palette.accent)
                }

                Text(L10n.PhotoRoom.buildingRoom)
                    .font(Theme.Typo.headline())
                    .foregroundStyle(Theme.Palette.textPrimary)

                if let statusMessage, !statusMessage.isEmpty {
                    Text(statusMessage)
                        .font(Theme.Typo.body())
                        .foregroundStyle(Theme.Palette.textSecondary)
                        .multilineTextAlignment(.center)
                }

                Text("\(Int(progress * 100))%")
                    .font(Theme.Typo.title())
                    .foregroundStyle(Theme.Palette.accent)

                Text(L10n.PhotoRoom.buildingRoomSubtext)
                    .font(Theme.Typo.caption())
                    .foregroundStyle(Theme.Palette.textSecondary)
                    .multilineTextAlignment(.center)
            }
            .padding(Theme.Space.xxl)
            .frame(maxWidth: 340)
            .background(
                RoundedRectangle(cornerRadius: Theme.Radius.sheet, style: .continuous)
                    .fill(Theme.Palette.surface)
            )
            .overlay(
                RoundedRectangle(cornerRadius: Theme.Radius.sheet, style: .continuous)
                    .stroke(Theme.Palette.hairline, lineWidth: 1)
            )
            .padding(.horizontal, Theme.Space.xl)
        }
    }
}

// MARK: - Saving room (in-progress — gold, not green download)

struct PaafektSavingRoomOverlay: View {
    let progress: Double
    let title: String
    var subtitle: String?

    var body: some View {
        ZStack {
            Theme.Palette.background.opacity(0.92)
                .ignoresSafeArea()

            VStack(spacing: Theme.Space.xl) {
                ZStack {
                    Circle()
                        .stroke(Theme.Palette.accent.opacity(0.25), lineWidth: 6)
                        .frame(width: 80, height: 80)
                    Circle()
                        .trim(from: 0, to: CGFloat(min(max(progress, 0), 1)))
                        .stroke(
                            Theme.Palette.accent,
                            style: StrokeStyle(lineWidth: 6, lineCap: .round)
                        )
                        .frame(width: 80, height: 80)
                        .rotationEffect(.degrees(-90))
                    Image(systemName: "house")
                        .font(.system(size: 28, weight: .regular))
                        .foregroundStyle(Theme.Palette.accent)
                }

                VStack(spacing: Theme.Space.sm) {
                    Text(title)
                        .font(Theme.Typo.headline())
                        .foregroundStyle(Theme.Palette.textPrimary)
                    if let subtitle, !subtitle.isEmpty {
                        Text(subtitle)
                            .font(Theme.Typo.body())
                            .foregroundStyle(Theme.Palette.textSecondary)
                            .multilineTextAlignment(.center)
                    }
                }

                ProgressView(value: progress)
                    .tint(Theme.Palette.accent)
                    .frame(width: 220)

                Text("\(Int(progress * 100))%")
                    .font(Theme.Typo.caption())
                    .foregroundStyle(Theme.Palette.textSecondary)
            }
            .padding(Theme.Space.xxl)
            .frame(maxWidth: 340)
            .background(
                RoundedRectangle(cornerRadius: Theme.Radius.sheet, style: .continuous)
                    .fill(Theme.Palette.surface)
            )
            .overlay(
                RoundedRectangle(cornerRadius: Theme.Radius.sheet, style: .continuous)
                    .stroke(Theme.Palette.hairline, lineWidth: 1)
            )
            .padding(.horizontal, Theme.Space.xl)
        }
    }
}

// MARK: - Save success snackbar

struct PaafektRoomSavedSnackbar: View {
    let message: String
    @Binding var isShowing: Bool

    private let autoDismissDelay: TimeInterval = 3.0

    var body: some View {
        HStack(spacing: Theme.Space.md) {
            Image(systemName: "checkmark.circle.fill")
                .font(.title3)
                .foregroundStyle(Theme.Palette.accent)

            Text(message)
                .font(Theme.Typo.body())
                .foregroundStyle(Theme.Palette.textPrimary)
                .lineLimit(2)

            Spacer(minLength: 0)
        }
        .padding(.horizontal, Theme.Space.lg)
        .padding(.vertical, Theme.Space.md)
        .background(
            RoundedRectangle(cornerRadius: Theme.Radius.control, style: .continuous)
                .fill(Theme.Palette.surface.opacity(0.96))
        )
        .overlay(
            RoundedRectangle(cornerRadius: Theme.Radius.control, style: .continuous)
                .stroke(Theme.Palette.hairline, lineWidth: 1)
        )
        .shadow(color: .black.opacity(0.2), radius: 10, x: 0, y: 4)
        .padding(.horizontal, Theme.Space.lg)
        .padding(.bottom, Theme.Space.lg)
        .transition(.move(edge: .bottom).combined(with: .opacity))
        .onAppear {
            DispatchQueue.main.asyncAfter(deadline: .now() + autoDismissDelay) {
                if isShowing {
                    withAnimation(.easeInOut(duration: 0.25)) {
                        isShowing = false
                    }
                }
            }
        }
    }
}

// MARK: - Name your room (keyboard-aware sheet)

struct PaafektNameRoomSheet: View {
    @Binding var isPresented: Bool
    @Binding var roomName: String
    var title: String = L10n.RoomViewer.nameYourRoom
    var onSave: () -> Void

    @FocusState private var isFieldFocused: Bool

    private var trimmedName: String {
        roomName.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private var isNameValid: Bool {
        DisplayNameValidation.isValid(roomName)
    }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: Theme.Space.lg) {
                    TextField(L10n.RoomViewer.roomNamePlaceholder, text: $roomName)
                        .textFieldStyle(.plain)
                        .font(Theme.Typo.body())
                        .foregroundStyle(Theme.Palette.textPrimary)
                        .padding(Theme.Space.md)
                        .background(
                            RoundedRectangle(cornerRadius: Theme.Radius.control, style: .continuous)
                                .fill(Theme.Palette.surfaceHi)
                        )
                        .overlay(
                            RoundedRectangle(cornerRadius: Theme.Radius.control, style: .continuous)
                                .stroke(Theme.Palette.hairline, lineWidth: 1)
                        )
                        .autocorrectionDisabled(true)
                        .textInputAutocapitalization(.words)
                        .focused($isFieldFocused)

                    Text(isNameValid || trimmedName.isEmpty
                         ? L10n.RoomViewer.nameHint
                         : L10n.RoomViewer.invalidRoomName)
                        .font(Theme.Typo.caption())
                        .foregroundStyle(
                            !trimmedName.isEmpty && !isNameValid
                                ? Theme.Palette.accent
                                : Theme.Palette.textSecondary
                        )

                    HStack(spacing: Theme.Space.md) {
                        Button(L10n.Common.cancel) {
                            isPresented = false
                        }
                        .buttonStyle(SecondaryButtonStyle())

                        Button(L10n.Common.save) {
                            guard isNameValid else { return }
                            onSave()
                        }
                        .buttonStyle(PrimaryButtonStyle())
                        .disabled(!isNameValid)
                    }
                }
                .padding(Theme.Space.lg)
                .frame(maxWidth: 420)
                .frame(maxWidth: .infinity)
            }
            .scrollDismissesKeyboard(.interactively)
            .paafektScreenBackground()
            .navigationTitle(title)
            .navigationBarTitleDisplayMode(.inline)
            .toolbarBackground(Theme.Palette.surface, for: .navigationBar)
            .toolbarColorScheme(.dark, for: .navigationBar)
        }
        .presentationDetents([.height(260), .medium])
        .presentationDragIndicator(.visible)
        .presentationBackground(Theme.Palette.surface)
        .onAppear {
            isFieldFocused = true
        }
    }
}

// MARK: - Delete confirmation

struct PaafektDeleteRoomDialog: View {
    @Binding var isPresented: Bool
    let onDelete: () -> Void

    var body: some View {
        ZStack {
            Color.black.opacity(0.55)
                .ignoresSafeArea()
                .onTapGesture {
                    isPresented = false
                }

            VStack(spacing: Theme.Space.lg) {
                Text(L10n.DeleteRoom.title)
                    .font(Theme.Typo.headline())
                    .foregroundStyle(Theme.Palette.textPrimary)
                    .multilineTextAlignment(.center)

                Text(L10n.DeleteRoom.message)
                    .font(Theme.Typo.body())
                    .foregroundStyle(Theme.Palette.textSecondary)
                    .multilineTextAlignment(.center)

                HStack(spacing: Theme.Space.md) {
                    Button(L10n.Common.cancel) {
                        isPresented = false
                    }
                    .buttonStyle(SecondaryButtonStyle())

                    Button(L10n.Common.delete) {
                        isPresented = false
                        onDelete()
                    }
                    .font(Theme.Typo.headline())
                    .foregroundStyle(.white)
                    .padding(.horizontal, Theme.Space.lg)
                    .padding(.vertical, Theme.Space.md)
                    .frame(maxWidth: .infinity)
                    .background(
                        RoundedRectangle(cornerRadius: Theme.Radius.control, style: .continuous)
                            .fill(Theme.Palette.danger)
                    )
                }
            }
            .padding(Theme.Space.xl)
            .frame(maxWidth: 340)
            .background(
                RoundedRectangle(cornerRadius: Theme.Radius.sheet, style: .continuous)
                    .fill(Theme.Palette.surface)
            )
            .overlay(
                RoundedRectangle(cornerRadius: Theme.Radius.sheet, style: .continuous)
                    .stroke(Theme.Palette.hairline, lineWidth: 1)
            )
            .padding(.horizontal, Theme.Space.xl)
        }
        .transition(.opacity)
    }
}

// MARK: - Error notice (non-success saves)

struct PaafektErrorNotice: View {
    @Binding var isPresented: Bool
    let message: String

    var body: some View {
        ZStack {
            Color.black.opacity(0.55)
                .ignoresSafeArea()
                .onTapGesture { isPresented = false }

            VStack(spacing: Theme.Space.lg) {
                Text(message)
                    .font(Theme.Typo.body())
                    .foregroundStyle(Theme.Palette.textPrimary)
                    .multilineTextAlignment(.center)

                Button(L10n.Common.ok) {
                    isPresented = false
                }
                .buttonStyle(PrimaryButtonStyle())
            }
            .padding(Theme.Space.xl)
            .frame(maxWidth: 340)
            .background(
                RoundedRectangle(cornerRadius: Theme.Radius.sheet, style: .continuous)
                    .fill(Theme.Palette.surface)
            )
            .overlay(
                RoundedRectangle(cornerRadius: Theme.Radius.sheet, style: .continuous)
                    .stroke(Theme.Palette.hairline, lineWidth: 1)
            )
            .padding(.horizontal, Theme.Space.xl)
        }
    }
}
