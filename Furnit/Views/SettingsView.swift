import SwiftUI

struct SettingsView: View {
    @ObservedObject private var appState = AppStateManager.shared
    @Environment(\.dismiss) private var dismiss
    
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
                
                // Future Settings Placeholder
                Section {
                    Text("More settings coming soon...")
                        .foregroundColor(.secondary)
                        .font(.footnote)
                        .frame(maxWidth: .infinity, alignment: .center)
                        .padding(.vertical, 20)
                } header: {
                    Text("Additional Settings")
                }
            }
            .navigationTitle("Settings")
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

// Preview for development
#Preview {
    SettingsView()
        .environment(\.appState, AppStateManager.shared)
}