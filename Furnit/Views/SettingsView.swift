import SwiftUI

struct SettingsView: View {
    @ObservedObject private var appState = AppStateManager.shared
    @EnvironmentObject var authManager: AuthenticationManager
    @Environment(\.dismiss) private var dismiss
    @State private var showLogoutConfirmation = false
    @State private var showDeleteConfirmation = false
    @State private var showDeleteError = false
    @State private var showDeleteSuccess = false
    @State private var deleteErrorMessage = ""
    @State private var isDeletingAccount = false
    
    // Single Photo Room Dimensions
    @AppStorage("singlePhotoRoom.width") private var roomWidth: Double = 4.0
    @AppStorage("singlePhotoRoom.depth") private var roomDepth: Double = 4.5
    @AppStorage("singlePhotoRoom.height") private var roomHeight: Double = 2.8

    // Room Viewer Settings
    @AppStorage("roomViewer.oscillation") private var oscillationEnabled: Bool = false
    @AppStorage("roomViewer.infiniteZoom") private var infiniteZoomEnabled: Bool = true

    /// Match Android `FurnitureFitManager.KEY_SHOW_ROOM_FURNITURE_CALIBRATE_UI` — default off.
    @AppStorage("show_room_furniture_calibrate") private var showRoomFurnitureCalibrate = false

    /// Minimum confidence for choosing the **primary** furniture detection (largest box among those above this threshold).
    @AppStorage("furnitureFit.primaryDetectionMinConfidence") private var primaryDetectionMinConfidence: Double = 0.57
    @AppStorage("furnitureFit.primarySelectionByHighestConfidence") private var primarySelectionByHighestConfidence: Bool = false

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

                // Room Viewer Settings Section
                Section {
                    Toggle(isOn: $oscillationEnabled) {
                        HStack {
                            Image(systemName: "arrow.left.arrow.right")
                                .foregroundColor(.cyan)
                            VStack(alignment: .leading, spacing: 2) {
                                Text(L10n.Settings.autoOrbit)
                                    .font(.headline)
                                Text(L10n.Settings.autoOrbitDescription)
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                        }
                    }
                    .tint(.cyan)

                    Toggle(isOn: $infiniteZoomEnabled) {
                        HStack {
                            Image(systemName: "plus.magnifyingglass")
                                .foregroundColor(.orange)
                            VStack(alignment: .leading, spacing: 2) {
                                Text(L10n.Settings.infiniteZoom)
                                    .font(.headline)
                                Text(L10n.Settings.infiniteZoomDescription)
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                        }
                    }
                    .tint(.orange)
                } header: {
                    Text(L10n.Settings.roomViewerSection)
                }

                // Furniture segmentation (FurnitureFit) — YOLO-E Core ML
                Section {
                    Toggle(isOn: $appState.qualitySettings.yoloeCoreMLAllowGPU) {
                        HStack {
                            Image(systemName: "cpu")
                                .foregroundColor(.indigo)
                            VStack(alignment: .leading, spacing: 2) {
                                Text(L10n.Settings.yoloeCoreMLAllowGPU)
                                    .font(.headline)
                                Text(L10n.Settings.yoloeCoreMLAllowGPUDescription)
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                        }
                    }
                    .tint(.indigo)
                    .onChange(of: appState.qualitySettings.yoloeCoreMLAllowGPU) { _, _ in
                        Task { await YOLOEModelService.shared.reloadForComputeUnitsChange() }
                    }

                    Toggle(isOn: $appState.qualitySettings.furnitureFitARDepthCompanionEnabled) {
                        HStack {
                            Image(systemName: "arkit")
                                .foregroundColor(.cyan)
                            VStack(alignment: .leading, spacing: 2) {
                                Text(L10n.Settings.furnitureFitARCompanion)
                                    .font(.headline)
                                Text(L10n.Settings.furnitureFitARCompanionDescription)
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                                if !QualitySettings.supportsFurnitureFitARAssisted {
                                    Text(L10n.Settings.furnitureFitARCompanionUnavailable)
                                        .font(.caption2)
                                        .foregroundColor(.secondary)
                                }
                            }
                        }
                    }
                    .tint(.cyan)
                    .disabled(!QualitySettings.supportsFurnitureFitARAssisted)

                    if QualitySettings.supportsLiDARSceneDepth {
                        Toggle(isOn: $showRoomFurnitureCalibrate) {
                            HStack {
                                Image(systemName: "slider.horizontal.3")
                                    .foregroundColor(.purple)
                                VStack(alignment: .leading, spacing: 2) {
                                    Text(L10n.Settings.showRoomFurnitureCalibrate)
                                        .font(.headline)
                                    Text(L10n.Settings.showRoomFurnitureCalibrateDescription)
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                }
                            }
                        }
                        .tint(.purple)
                    }

                    VStack(alignment: .leading, spacing: 12) {
                        HStack {
                            Image(systemName: "scope")
                                .foregroundColor(.mint)
                                .frame(width: 24)
                            Text(L10n.Settings.primaryDetectionConfidence)
                                .font(.headline)
                            Spacer()
                            Text(L10n.Settings.primaryDetectionConfidencePercent(Int(primaryDetectionMinConfidence * 100)))
                                .font(.subheadline.monospacedDigit())
                                .foregroundColor(.secondary)
                        }
                        Slider(value: $primaryDetectionMinConfidence, in: 0.05...0.99, step: 0.01)
                            .tint(.mint)
                    }
                    .padding(.vertical, 4)

                    Toggle(isOn: $primarySelectionByHighestConfidence) {
                        HStack {
                            Image(systemName: "chart.bar.fill")
                                .foregroundColor(.teal)
                            VStack(alignment: .leading, spacing: 2) {
                                Text(L10n.Settings.primarySelectionByHighestConfidence)
                                    .font(.headline)
                                Text(L10n.Settings.primarySelectionByHighestConfidenceDescription)
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                        }
                    }
                    .tint(.teal)
                } header: {
                    Text(L10n.Settings.furnitureSegmentationSection)
                } footer: {
                    VStack(alignment: .leading, spacing: 8) {
                        Text(L10n.Settings.primaryDetectionConfidenceFooter)
                        Text(L10n.Settings.primarySelectionByHighestConfidenceFooter)
                    }
                    .font(.footnote)
                }

                #if DEBUG
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
                                .foregroundColor(.blue)
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
                                .foregroundColor(.blue)
                            Text(L10n.Settings.termsOfService)
                            Spacer()
                            Image(systemName: "arrow.up.right.square")
                                .foregroundColor(.secondary)
                                .font(.caption)
                        }
                    }

                    Link(destination: URL(string: "https://paafekt.com/support")!) {
                        HStack {
                            Image(systemName: "questionmark.circle.fill")
                                .foregroundColor(.blue)
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
                                .foregroundColor(.blue)
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

                    Button(role: .destructive) {
                        showDeleteConfirmation = true
                    } label: {
                        HStack {
                            if isDeletingAccount {
                                ProgressView()
                                    .tint(.red)
                            } else {
                                Image(systemName: "trash")
                                    .foregroundColor(.red)
                            }
                            Text(L10n.Profile.deleteAccount)
                                .foregroundColor(.red)
                        }
                    }
                    .disabled(isDeletingAccount || authManager.isLoading)
                } header: {
                    Text(L10n.Settings.account)
                } footer: {
                    Text("Delete Account removes your sign-in account and local account data from this device.")
                        .font(.footnote)
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
            .alert(L10n.Profile.deleteAccountConfirmTitle, isPresented: $showDeleteConfirmation) {
                Button(L10n.Common.cancel, role: .cancel) { }
                Button(L10n.Common.delete, role: .destructive) {
                    deleteAccount()
                }
            } message: {
                Text(L10n.Profile.deleteAccountConfirmMessage)
            }
            .alert(L10n.Common.error, isPresented: $showDeleteError) {
                Button(L10n.Common.ok, role: .cancel) { }
            } message: {
                Text(deleteErrorMessage)
            }
            .alert(L10n.Profile.deleteAccountSuccessTitle, isPresented: $showDeleteSuccess) {
                Button(L10n.Common.ok) {
                    dismiss()
                }
            } message: {
                Text(L10n.Profile.deleteAccountSuccessMessage)
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
                isDeletingAccount = false
                showDeleteSuccess = true
            } catch {
                isDeletingAccount = false
                deleteErrorMessage = error.localizedDescription
                showDeleteError = true
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
        .environmentObject(AuthenticationManager())
}
