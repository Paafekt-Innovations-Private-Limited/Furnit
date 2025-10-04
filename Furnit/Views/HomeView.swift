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
                RoomScannerTestView(isPresented: $showingRoomScanner)
            }
        }
    }
}

// MARK: - Test View (Replace with ModelViewer3D later)
//struct RoomScannerTestView: View {
//    @Binding var isPresented: Bool
//    
//    var body: some View {
//        NavigationView {
//            VStack(spacing: 30) {
//                Image(systemName: "camera.viewfinder")
//                    .font(.system(size: 80))
//                    .foregroundColor(.blue)
//                
//                Text("Room Scanner")
//                    .font(.largeTitle)
//                    .fontWeight(.bold)
//                
//                Text("This is a test view")
//                    .font(.title3)
//                    .foregroundColor(.secondary)
//                
//                Text("Replace this with ModelViewer3D()")
//                    .font(.caption)
//                    .foregroundColor(.secondary)
//                
//                Spacer()
//                
//                Button(action: {
//                    isPresented = false
//                }) {
//                    Text("Close")
//                        .font(.headline)
//                        .foregroundColor(.white)
//                        .frame(width: 200, height: 50)
//                        .background(Color.blue)
//                        .cornerRadius(10)
//                }
//                .padding()
//            }
//            .padding()
//            .navigationTitle("Room Scanner")
//            .navigationBarTitleDisplayMode(.inline)
//            .toolbar {
//                ToolbarItem(placement: .navigationBarLeading) {
//                    Button("Cancel") {
//                        isPresented = false
//                    }
//                }
//            }
//        }
//    }
//}

// MARK: - Alternative with Bottom Button Bar
//struct HomeViewWithBottomBar: View {
//    @StateObject private var modelManager = USDZModelManager()
//    @Environment(\.appState) private var appState
//    @State private var showingSettings = false
//    @State private var showingRoomScanner = false
//    
//    var body: some View {
//        NavigationStack {
//            VStack(spacing: 0) {
//                // List takes up available space
//                if modelManager.models.isEmpty {
//                    ContentUnavailableView(
//                        "No 3D Models Found",
//                        systemImage: "cube.transparent",
//                        description: Text("Tap the camera button below to scan a room")
//                    )
//                } else {
//                    List(modelManager.models) { model in
//                        NavigationLink(destination: ModelViewerView(model: model)) {
//                            ModelRowView(model: model)
//                        }
//                    }
//                    .listStyle(PlainListStyle())
//                }
//                
//                // Bottom button bar - ALWAYS VISIBLE
//                HStack {
//                    Spacer()
//                    
//                    Button(action: {
//                        print("Camera button tapped from bottom bar!")
//                        showingRoomScanner = true
//                    }) {
//                        VStack {
//                            Image(systemName: "camera.viewfinder")
//                                .font(.system(size: 24))
//                            Text("Scan Room")
//                                .font(.caption)
//                        }
//                        .foregroundColor(.white)
//                        .frame(width: 100, height: 60)
//                        .background(
//                            LinearGradient(
//                                gradient: Gradient(colors: [Color.purple, Color.blue]),
//                                startPoint: .topLeading,
//                                endPoint: .bottomTrailing
//                            )
//                        )
//                        .cornerRadius(10)
//                    }
//                    
//                    Spacer()
//                }
//                .padding(.vertical, 10)
//                .background(Color(UIColor.systemBackground))
//                .shadow(color: .black.opacity(0.1), radius: 5, x: 0, y: -2)
//            }
//            .navigationTitle("3D Room Models")
//            .navigationBarTitleDisplayMode(.large)
//            .toolbar {
//                ToolbarItem(placement: .navigationBarTrailing) {
//                    Button {
//                        showingSettings = true
//                    } label: {
//                        Image(systemName: "gearshape")
//                            .font(.title3)
//                    }
//                    .accessibilityLabel("Settings")
//                }
//            }
//            .sheet(isPresented: $showingSettings) {
//                SettingsView()
//            }
//            .sheet(isPresented: $showingRoomScanner) {
//                RoomScannerTestView(isPresented: $showingRoomScanner)
//            }
//        }
//    }
//}

//struct ModelRowView: View {
//    let model: USDZModel
//    
//    var body: some View {
//        HStack {
//            Image(systemName: "cube.fill")
//                .foregroundColor(.blue)
//                .font(.title2)
//                .frame(width: 40, height: 40)
//                .background(Color.blue.opacity(0.1))
//                .cornerRadius(8)
//            
//            VStack(alignment: .leading, spacing: 4) {
//                Text(model.displayName)
//                    .font(.headline)
//                    .foregroundColor(.primary)
//                
//                Text("3D Room Model")
//                    .font(.caption)
//                    .foregroundColor(.secondary)
//            }
//            
//            Spacer()
//            
//            Image(systemName: "chevron.right")
//                .foregroundColor(.secondary)
//                .font(.caption)
//        }
//        .padding(.vertical, 4)
//    }
//}

#Preview {
    HomeView()
}
