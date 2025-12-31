import SwiftUI

struct SettingsView: View {
    @ObservedObject private var appState = AppStateManager.shared
    @EnvironmentObject var authManager: AuthenticationManager
    @Environment(\.dismiss) private var dismiss
    @State private var showLogoutConfirmation = false
    
    // Single Photo Room Dimensions
    @AppStorage("singlePhotoRoom.width") private var roomWidth: Double = 4.0
    @AppStorage("singlePhotoRoom.depth") private var roomDepth: Double = 4.5
    @AppStorage("singlePhotoRoom.height") private var roomHeight: Double = 2.8

    var body: some View {
        NavigationView {
            Form {
                // Quality Settings Section
                Section {
                    ForEach(appState.qualitySettings.availableQualities) { quality in
                        QualityOptionView(
                            quality: quality,
                            isSelected: appState.qualitySettings.isSelected(quality),
                            onSelect: {
                                appState.updateQuality(quality)
                            }
                        )
                    }
                } header: {
                    Text(L10n.Settings.quality)
                } footer: {
                    Text(L10n.Settings.qualityFooter)
                        .font(.footnote)
                }
                
                // App Info Section
                Section {
                    HStack {
                        Text(L10n.App.version)
                        Spacer()
                        Text(appState.formattedVersion)
                            .foregroundColor(.secondary)
                    }

                    HStack {
                        Text(L10n.Settings.currentQuality)
                        Spacer()
                        Text(appState.currentQuality.displayName)
                            .foregroundColor(.secondary)
                    }
                } header: {
                    Text(L10n.Settings.appInfo)
                }
                
                // Movement Speed Settings Section
                Section {
                    ForEach(MovementSpeed.allCases) { speed in
                        MovementSpeedOptionView(
                            speed: speed,
                            isSelected: appState.qualitySettings.isMovementSpeedSelected(speed),
                            onSelect: {
                                appState.updateMovementSpeed(speed)
                            }
                        )
                    }
                } header: {
                    Text(L10n.Settings.movementSpeed)
                } footer: {
                    Text(L10n.Settings.movementSpeedFooter)
                        .font(.footnote)
                }
                
                // Single Photo Room Dimensions Section
                Section {
                    VStack(alignment: .leading, spacing: 12) {
                        // Width slider
                        HStack {
                            Image(systemName: "arrow.left.and.right")
                                .foregroundColor(.blue)
                                .frame(width: 24)
                            Text(L10n.Settings.width(roomWidth))
                                .font(.headline)
                        }
                        Slider(value: $roomWidth, in: 2...8, step: 0.1)
                            .tint(.blue)
                    }
                    .padding(.vertical, 4)

                    VStack(alignment: .leading, spacing: 12) {
                        // Depth slider
                        HStack {
                            Image(systemName: "arrow.up.and.down")
                                .foregroundColor(.green)
                                .frame(width: 24)
                            Text(L10n.Settings.depth(roomDepth))
                                .font(.headline)
                        }
                        Slider(value: $roomDepth, in: 2...8, step: 0.1)
                            .tint(.green)
                    }
                    .padding(.vertical, 4)

                    VStack(alignment: .leading, spacing: 12) {
                        // Height slider
                        HStack {
                            Image(systemName: "arrow.up.to.line")
                                .foregroundColor(.orange)
                                .frame(width: 24)
                            Text(L10n.Settings.height(roomHeight))
                                .font(.headline)
                        }
                        Slider(value: $roomHeight, in: 2.2...4, step: 0.1)
                            .tint(.orange)
                    }
                    .padding(.vertical, 4)
                } header: {
                    Text(L10n.Settings.roomDimensions)
                } footer: {
                    Text(L10n.Settings.roomDimensionsFooter)
                        .font(.footnote)
                }

                // Developer Settings Section
                Section {
                    Toggle(isOn: $appState.qualitySettings.debugMode) {
                        HStack {
                            Image(systemName: "ladybug.fill")
                                .foregroundColor(.purple)
                            VStack(alignment: .leading, spacing: 2) {
                                Text(L10n.Settings.debugMode)
                                    .font(.headline)
                                Text(L10n.Settings.debugModeDescription)
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                        }
                    }
                    .tint(.purple)

                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Image(systemName: "slider.horizontal.3")
                                .foregroundColor(.orange)
                            Text(L10n.Settings.maskOverlapThreshold)
                                .font(.headline)
                        }

                        HStack {
                            Text("0.0")
                                .font(.caption)
                                .foregroundColor(.secondary)

                            Slider(
                                value: $appState.qualitySettings.maskOverlapThreshold,
                                in: 0.0...1.0,
                                step: 0.05
                            )
                            .tint(.orange)

                            Text("1.0")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }

                        HStack {
                            Text(L10n.Settings.currentValue)
                                .font(.caption)
                                .foregroundColor(.secondary)
                            Text(String(format: "%.2f", appState.qualitySettings.maskOverlapThreshold))
                                .font(.caption)
                                .foregroundColor(.orange)
                                .fontWeight(.medium)
                        }

                        Text(L10n.Settings.maskOverlapDescription)
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    .padding(.vertical, 4)
                } header: {
                    Text(L10n.Settings.developer)
                } footer: {
                    Text(L10n.Settings.developerFooter)
                        .font(.footnote)
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
                    }

                    Button(action: {
                        showLogoutConfirmation = true
                    }) {
                        HStack {
                            Image(systemName: "rectangle.portrait.and.arrow.right")
                                .foregroundColor(.red)
                            Text(L10n.Profile.logout)
                                .foregroundColor(.red)
                        }
                    }
                } header: {
                    Text(L10n.Settings.account)
                }
            }
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
}

// Custom view for each quality option
struct QualityOptionView: View {
    let quality: AssetQuality
    let isSelected: Bool
    let onSelect: () -> Void
    
    var body: some View {
        HStack {
            // Quality icon
            Image(systemName: quality.icon)
                .foregroundColor(quality.isAvailable ? .blue : .gray)
                .font(.title2)
                .frame(width: 24, height: 24)
            
            VStack(alignment: .leading, spacing: 2) {
                HStack {
                    Text(quality.displayName)
                        .font(.headline)
                        .foregroundColor(quality.isAvailable ? .primary : .secondary)
                    
                    // Premium badge for best quality
                    if quality == .best {
                        Image(systemName: "lock.fill")
                            .foregroundColor(.orange)
                            .font(.caption)
                    }
                    
                    Spacer()
                    
                    // Selection indicator
                    if isSelected && quality.isAvailable {
                        Image(systemName: "checkmark.circle.fill")
                            .foregroundColor(.blue)
                            .font(.title3)
                    }
                }
                
                Text(quality.description)
                    .font(.caption)
                    .foregroundColor(.secondary)
                
                // Special message for unavailable options
                if let message = quality.unavailableMessage {
                    Text(message)
                        .font(.caption2)
                        .foregroundColor(.orange)
                        .fontWeight(.medium)
                }
            }
        }
        .padding(.vertical, 4)
        .contentShape(Rectangle()) // Make entire area tappable
        .onTapGesture {
            if quality.isAvailable {
                onSelect()
            }
        }
        .opacity(quality.isAvailable ? 1.0 : 0.6)
    }
}

// Custom view for each movement speed option
struct MovementSpeedOptionView: View {
    let speed: MovementSpeed
    let isSelected: Bool
    let onSelect: () -> Void

    var body: some View {
        HStack {
            // Speed icon
            Image(systemName: speed.icon)
                .foregroundColor(.blue)
                .font(.title2)
                .frame(width: 24, height: 24)

            VStack(alignment: .leading, spacing: 2) {
                HStack {
                    Text(speed.displayName)
                        .font(.headline)
                        .foregroundColor(.primary)

                    Spacer()

                    // Selection indicator
                    if isSelected {
                        Image(systemName: "checkmark.circle.fill")
                            .foregroundColor(.blue)
                            .font(.title3)
                    }
                }

                Text(speed.description)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
        .padding(.vertical, 4)
        .contentShape(Rectangle()) // Make entire area tappable
        .onTapGesture {
            onSelect()
        }
    }
}

// Preview for development
#Preview {
    SettingsView()
        .environment(\.appState, AppStateManager.shared)
}
