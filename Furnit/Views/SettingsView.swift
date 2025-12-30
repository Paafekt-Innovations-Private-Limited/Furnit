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
                    Text("3D Asset Quality")
                } footer: {
                    Text("Quality affects rendering detail and performance. Higher quality may impact battery life.")
                        .font(.footnote)
                }
                
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
                
                // Single Photo Room Dimensions Section
                Section {
                    VStack(alignment: .leading, spacing: 12) {
                        // Width slider
                        HStack {
                            Image(systemName: "arrow.left.and.right")
                                .foregroundColor(.blue)
                                .frame(width: 24)
                            Text("Width: \(String(format: "%.1f", roomWidth))m")
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
                            Text("Depth: \(String(format: "%.1f", roomDepth))m")
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
                            Text("Height: \(String(format: "%.1f", roomHeight))m")
                                .font(.headline)
                        }
                        Slider(value: $roomHeight, in: 2.2...4, step: 0.1)
                            .tint(.orange)
                    }
                    .padding(.vertical, 4)
                } header: {
                    Text("Single Photo Room Dimensions")
                } footer: {
                    Text("Adjust the default dimensions for rooms created from single photos. These values will be used as starting points when detecting room boundaries.")
                        .font(.footnote)
                }

                // Developer Settings Section
                Section {
                    Toggle(isOn: $appState.qualitySettings.debugMode) {
                        HStack {
                            Image(systemName: "ladybug.fill")
                                .foregroundColor(.purple)
                            VStack(alignment: .leading, spacing: 2) {
                                Text("Debug Mode")
                                    .font(.headline)
                                Text("Show detection boxes and timing information")
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
                            Text("Mask Overlap Threshold")
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
                            Text("Current value:")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            Text(String(format: "%.2f", appState.qualitySettings.maskOverlapThreshold))
                                .font(.caption)
                                .foregroundColor(.orange)
                                .fontWeight(.medium)
                        }
                        
                        Text("Controls how much mask overlap is required for detections to be merged. Lower values merge more similar objects.")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    .padding(.vertical, 4)
                } header: {
                    Text("Developer")
                } footer: {
                    Text("Enable debug mode to see detailed detection information and performance metrics. Adjust thresholds to control detection sensitivity and filtering behavior.")
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
