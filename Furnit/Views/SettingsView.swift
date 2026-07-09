import SwiftUI

struct SettingsView: View {
    @ObservedObject private var appState = AppStateManager.shared
    @EnvironmentObject var authManager: AuthenticationManager
    @Environment(\.dismiss) private var dismiss
    @State private var showLogoutConfirmation = false
    @State private var showDeleteAccountConfirmation = false
    @State private var accountDeletionErrorMessage: String?
    @State private var isDeletingAccount = false

    // Manual single-photo room dimensions.
    @AppStorage("singlePhotoRoom.width") private var roomWidth: Double = 4.0
    @AppStorage("singlePhotoRoom.depth") private var roomDepth: Double = 4.5
    @AppStorage("singlePhotoRoom.height") private var roomHeight: Double = 2.8

    // Room Viewer Settings
    @AppStorage("roomViewer.oscillation") private var oscillationEnabled: Bool = false
    @AppStorage("roomViewer.infiniteZoom") private var infiniteZoomEnabled: Bool = true

    var body: some View {
        NavigationView {
            Form {
                // App Info Section
                Section {
                    HStack {
                        Text(L10n.App.version)
                        Spacer()
                        Text(appState.formattedVersion)
                            .foregroundColor(.secondary)
                    }
                } header: {
                    Text(L10n.Settings.appInfo)
                }

                Section {
                    VStack(alignment: .leading, spacing: 12) {
                        HStack {
                            Image(systemName: "arrow.left.and.right")
                                .foregroundStyle(Theme.Palette.textPrimary)
                                .frame(width: 24)
                            Text(L10n.Settings.width(roomWidth))
                                .font(.headline)
                        }
                        Slider(value: $roomWidth, in: 2...8, step: 0.1)
                            .tint(Theme.Palette.accent)
                    }
                    .padding(.vertical, 4)

                    VStack(alignment: .leading, spacing: 12) {
                        HStack {
                            Image(systemName: "arrow.up.and.down")
                                .foregroundStyle(Theme.Palette.textPrimary)
                                .frame(width: 24)
                            Text(L10n.Settings.depth(roomDepth))
                                .font(.headline)
                        }
                        Slider(value: $roomDepth, in: 2...8, step: 0.1)
                            .tint(Theme.Palette.accent)
                    }
                    .padding(.vertical, 4)

                    VStack(alignment: .leading, spacing: 12) {
                        HStack {
                            Image(systemName: "arrow.up.to.line")
                                .foregroundStyle(Theme.Palette.textPrimary)
                                .frame(width: 24)
                            Text(L10n.Settings.height(roomHeight))
                                .font(.headline)
                        }
                        Slider(value: $roomHeight, in: 2.2...4, step: 0.1)
                            .tint(Theme.Palette.accent)
                    }
                    .padding(.vertical, 4)
                } header: {
                    Text(L10n.Settings.roomDimensions)
                } footer: {
                    Text(L10n.Settings.roomDimensionsFooter)
                        .font(.footnote)
                }

                // Room Viewer Settings Section
                Section {
                    Toggle(isOn: $oscillationEnabled) {
                        HStack {
                            Image(systemName: "arrow.left.arrow.right")
                                .foregroundStyle(Theme.Palette.textPrimary)
                            VStack(alignment: .leading, spacing: 2) {
                                Text(L10n.Settings.autoOrbit)
                                    .font(.headline)
                                Text(L10n.Settings.autoOrbitDescription)
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                        }
                    }
                    .tint(Theme.Palette.accent)

                    Toggle(isOn: $infiniteZoomEnabled) {
                        HStack {
                            Image(systemName: "plus.magnifyingglass")
                                .foregroundStyle(Theme.Palette.textPrimary)
                            VStack(alignment: .leading, spacing: 2) {
                                Text(L10n.Settings.infiniteZoom)
                                    .font(.headline)
                                Text(L10n.Settings.infiniteZoomDescription)
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                        }
                    }
                    .tint(Theme.Palette.accent)
                } header: {
                    Text(L10n.Settings.roomViewerSection)
                        .foregroundStyle(Theme.Palette.accent)
                }

                Section {
                    NavigationLink(destination: SettingsFurnitureFitImageScanView()) {
                        HStack {
                            Image(systemName: "photo.on.rectangle.angled")
                                .foregroundStyle(Theme.Palette.textPrimary)
                            VStack(alignment: .leading, spacing: 2) {
                                Text(L10n.Settings.imageScan)
                                    .font(.headline)
                                Text(L10n.Settings.imageScanDescription)
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                        }
                    }
                } header: {
                    Text(L10n.Settings.singleImageScanSection)
                }

                #if DEBUG
                // Developer Settings Section
                Section {
                    Toggle(isOn: $appState.qualitySettings.debugMode) {
                        HStack {
                            Image(systemName: "ladybug.fill")
                                .foregroundStyle(Theme.Palette.textPrimary)
                            VStack(alignment: .leading, spacing: 2) {
                                Text(L10n.Settings.debugMode)
                                    .font(.headline)
                                Text(L10n.Settings.debugModeDescription)
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                        }
                    }
                    .tint(Theme.Palette.accent)
                } header: {
                    Text(L10n.Settings.developer)
                } footer: {
                    Text(L10n.Settings.developerFooter)
                        .font(.footnote)
                }
                #endif

                // Legal Section
                Section {
                    Link(destination: URL(string: "https://paafekt.com/privacy")!) {
                        HStack {
                            Image(systemName: "hand.raised.fill")
                                .foregroundStyle(Theme.Palette.textPrimary)
                            Text(L10n.Settings.privacyPolicy)
                            Spacer()
                            Image(systemName: "arrow.up.right.square")
                                .foregroundColor(.secondary)
                                .font(.caption)
                        }
                    }

                    Link(destination: URL(string: "https://paafekt.com/terms")!) {
                        HStack {
                            Image(systemName: "doc.text.fill")
                                .foregroundStyle(Theme.Palette.textPrimary)
                            Text(L10n.Settings.termsOfService)
                            Spacer()
                            Image(systemName: "arrow.up.right.square")
                                .foregroundColor(.secondary)
                                .font(.caption)
                        }
                    }

                    NavigationLink(destination: CreditsView()) {
                        HStack {
                            Image(systemName: "sparkles")
                                .foregroundStyle(Theme.Palette.textPrimary)
                            Text(L10n.Settings.credits)
                        }
                    }

                    Link(destination: URL(string: "https://paafekt.com/support")!) {
                        HStack {
                            Image(systemName: "questionmark.circle.fill")
                                .foregroundStyle(Theme.Palette.textPrimary)
                            Text(L10n.Settings.support)
                            Spacer()
                            Image(systemName: "arrow.up.right.square")
                                .foregroundColor(.secondary)
                                .font(.caption)
                        }
                    }

                    NavigationLink(destination: LicensesView()) {
                        HStack {
                            Image(systemName: "doc.plaintext.fill")
                                .foregroundStyle(Theme.Palette.textPrimary)
                            Text(L10n.Settings.licenses)
                        }
                    }
                } header: {
                    Text(L10n.Settings.legal)
                }

                // Account Section
                Section {
                    if let user = authManager.currentUser {
                        HStack {
                            Text(L10n.Settings.loggedInAs)
                            Spacer()
                            Text(user.name)
                                .foregroundColor(.secondary)
                        }

                        HStack {
                            Text(L10n.Login.phoneNumber)
                            Spacer()
                            Text(user.phoneNumber)
                                .foregroundColor(.secondary)
                        }
                    }

                    Button {
                        UserDefaults.standard.set(false, forKey: "hint_seenBrainHint")
                        UserDefaults.standard.set(false, forKey: "hint_seenPinchResize")
                        UserDefaults.standard.set(false, forKey: "hint_seenArSizing")
                        UserDefaults.standard.set(false, forKey: "hint_seenPickAnother")
                    } label: {
                        HStack {
                            Image(systemName: "arrow.counterclockwise")
                                .foregroundStyle(Theme.Palette.textPrimary)
                            Text(L10n.Settings.showTipsAgain)
                        }
                    }

                    Button(action: {
                        showLogoutConfirmation = true
                    }) {
                        HStack {
                            Image(systemName: "rectangle.portrait.and.arrow.right")
                                .foregroundStyle(Theme.Palette.danger)
                            Text(L10n.Profile.logout)
                                .foregroundStyle(Theme.Palette.danger)
                        }
                    }

                    Button(role: .destructive) {
                        showDeleteAccountConfirmation = true
                    } label: {
                        HStack {
                            Image(systemName: "trash.fill")
                                .foregroundStyle(Theme.Palette.danger)
                            Text(L10n.Profile.deleteAccount)
                                .foregroundStyle(Theme.Palette.danger)
                            if isDeletingAccount {
                                Spacer()
                                ProgressView()
                            }
                        }
                    }
                    .disabled(isDeletingAccount)
                } header: {
                    Text(L10n.Settings.account)
                }
            }
            .scrollContentBackground(.hidden)
            .paafektScreenBackground()
            .navigationTitle(L10n.Settings.title)
            .alert(L10n.Profile.logoutConfirmTitle, isPresented: $showLogoutConfirmation) {
                Button(L10n.Common.cancel, role: .cancel) { }
                Button(L10n.Profile.logout, role: .destructive) {
                    authManager.logout()
                    dismiss()
                }
            } message: {
                Text(L10n.Profile.logoutConfirmMessage)
            }
            .alert(L10n.Profile.deleteAccountConfirmTitle, isPresented: $showDeleteAccountConfirmation) {
                Button(L10n.Common.cancel, role: .cancel) { }
                Button(L10n.Common.delete, role: .destructive) {
                    deleteAccount()
                }
            } message: {
                Text(L10n.Profile.deleteAccountConfirmMessage)
            }
            .alert(
                L10n.Profile.deleteAccount,
                isPresented: Binding(
                    get: { accountDeletionErrorMessage != nil },
                    set: { if !$0 { accountDeletionErrorMessage = nil } }
                )
            ) {
                Button(L10n.Common.ok, role: .cancel) {
                    accountDeletionErrorMessage = nil
                }
            } message: {
                Text(accountDeletionErrorMessage ?? "")
            }
            .navigationBarTitleDisplayMode(.large)
            .navigationBarBackButtonHidden(false)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button(L10n.Common.done) {
                        dismiss()
                    }
                }
            }
        }
    }

    private func deleteAccount() {
        guard !isDeletingAccount else { return }
        isDeletingAccount = true

        Task {
            do {
                try await authManager.deleteCurrentAccount()
                dismiss()
            } catch {
                accountDeletionErrorMessage = error.localizedDescription
            }
            isDeletingAccount = false
        }
    }
}

// Preview for development
#Preview {
    SettingsView()
        .environment(\.appState, AppStateManager.shared)
        .environmentObject(AuthenticationManager())
}
