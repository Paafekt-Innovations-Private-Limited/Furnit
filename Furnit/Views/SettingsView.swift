import SwiftUI

struct SettingsView: View {
    @ObservedObject private var appState = AppStateManager.shared
    @EnvironmentObject var authManager: AuthenticationManager
    @Environment(\.dismiss) private var dismiss
    @State private var showLogoutConfirmation = false

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
                    Text("3D Asset Quality")
                } footer: {
                    Text("Quality affects rendering detail and performance. Higher quality may impact battery life.")
                        .font(.footnote)
                }
                
                // Development Section (Debug builds only)
                #if DEBUG
                Section {
                    NavigationLink(destination: FirebaseTestView()) {
                        HStack {
                            Image(systemName: "flame")
                                .foregroundColor(.orange)
                            Text("Firebase Setup Test")
                        }
                    }
                } header: {
                    Text("Development Tools")
                }
                #endif
                
                // App Info Section
                Section {
                    HStack {
                        Text("Version")
                        Spacer()
                        Text(appState.formattedVersion)
                            .foregroundColor(.secondary)
                    }
                    
                    HStack {
                        Text("Current Quality")
                        Spacer()
                        Text(appState.currentQuality.displayName)
                            .foregroundColor(.secondary)
                    }
                } header: {
                    Text("App Information")
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
                    Text("Movement Speed")
                } footer: {
                    Text("Controls how fast the camera moves when using the joystick. Choose the speed that feels most comfortable for you.")
                        .font(.footnote)
                }

                // Account Section
                Section {
                    if let user = authManager.currentUser {
                        HStack {
                            Text("Logged in as")
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
                            Text("Logout")
                                .foregroundColor(.red)
                        }
                    }
                } header: {
                    Text("Account")
                }
            }
            .navigationTitle("Settings")
            .alert("Logout", isPresented: $showLogoutConfirmation) {
                Button("Cancel", role: .cancel) { }
                Button("Logout", role: .destructive) {
                    authManager.logout()
                    dismiss()
                }
            } message: {
                Text("Are you sure you want to logout?")
            }
            .navigationBarTitleDisplayMode(.large)
            .navigationBarBackButtonHidden(false)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") {
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