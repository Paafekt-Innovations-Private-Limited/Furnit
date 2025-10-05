import SwiftUI

struct HomeView: View {
    @StateObject private var modelManager = USDZModelManager()
    @Environment(\.appState) private var appState
    @State private var showingSettings = false
    @State private var showingRoomScanner = false
    
    var body: some View {
        NavigationStack {
            VStack {
                if modelManager.models.isEmpty {
                    ContentUnavailableView(
                        "No 3D Models Found",
                        systemImage: "cube.transparent",
                        description: Text("Add USDZ files to your Assets catalog or tap the camera button to scan a room")
                    )
                } else {
                    List(modelManager.models) { model in
                        NavigationLink(destination: ModelViewerView(model: model)) {
                            ModelRowView(model: model)
                        }
                    }
                    .listStyle(PlainListStyle())
                }
            }
            .navigationTitle("3D Room Models")
            .navigationBarTitleDisplayMode(.large)
            .toolbar {
                ToolbarItemGroup(placement: .navigationBarTrailing) {
                    // Camera button in toolbar - GUARANTEED TO SHOW
                    Button {
                        print("Camera button tapped from toolbar!")
                        showingRoomScanner = true
                    } label: {
                        Image(systemName: "camera.fill")
                            .font(.title3)
                            .foregroundColor(.blue) // Made blue to stand out
                    }
                    .accessibilityLabel("Scan Room")
                    
                    // Settings button
                    Button {
                        showingSettings = true
                    } label: {
                        Image(systemName: "gearshape")
                            .font(.title3)
                    }
                    .accessibilityLabel("Settings")
                }
            }
            .sheet(isPresented: $showingSettings) {
                SettingsView()
            }
            .sheet(isPresented: $showingRoomScanner) {
                // Temporary placeholder view for testing
                DollhouseRoomScannerView(
                        isPresented: $showingRoomScanner,
                        modelManager: modelManager  // Pass the existing modelManager
                    )
            }
        }
    }
}

#Preview {
    HomeView()
}
