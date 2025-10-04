import SwiftUI

struct ContentView: View {
    @StateObject private var authManager = AuthenticationManager()
    
    var body: some View {
        Group {
            if authManager.isAuthenticated {
                HomeViewWithBottomBar(authManager: authManager)
            } else {
                LoginView()
            }
        }
        .animation(.easeInOut(duration: 0.3), value: authManager.isAuthenticated)
    }
}

// Alias for backward compatibility (can be removed later)
typealias HomeViewWithProfile = HomeViewWithBottomBar

// Updated HomeView with Bottom Tab Bar Navigation
struct HomeViewWithBottomBar: View {
    @ObservedObject var authManager: AuthenticationManager
    @EnvironmentObject private var appStateManager: AppStateManager
    @State private var selectedTab = 0
    
    var body: some View {
        TabView(selection: $selectedTab) {
            // Home Tab
            HomeTab()
                .tabItem {
                    Label("Home", systemImage: "house.fill")
                }
                .tag(0)
            
            // Explore/Search Tab
            ExploreTab()
                .tabItem {
                    Label("Explore", systemImage: "magnifyingglass")
                }
                .tag(1)
            
            // Favorites Tab
            FavoritesTab()
                .tabItem {
                    Label("Favorites", systemImage: "heart.fill")
                }
                .tag(2)
            
            // Profile Tab
            ProfileTab(authManager: authManager)
                .tabItem {
                    Label("Profile", systemImage: "person.circle.fill")
                }
                .tag(3)
        }
        .tint(.blue)
    }
}

// Home Tab with 3D Models List
struct HomeTab: View {
    @StateObject private var modelManager = USDZModelManager()
    @EnvironmentObject private var appStateManager: AppStateManager
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
                    // Camera button in toolbar
                    Button {
                        print("Camera button tapped from toolbar!")
                        showingRoomScanner = true
                    } label: {
                        Image(systemName: "camera.fill")
                            .font(.title3)
                            .foregroundColor(.blue)
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
                // Use the actual ModelViewer3D or SimpleCameraOverlay
                RoomScannerView(isPresented: $showingRoomScanner)
            }
        }
    }
}

// Explore Tab
struct ExploreTab: View {
    @State private var searchText = ""
    @State private var selectedCategory = "All"
    let categories = ["All", "Living Room", "Bedroom", "Kitchen", "Office", "Outdoor"]
    
    var body: some View {
        NavigationStack {
            VStack {
                // Search Bar
                HStack {
                    Image(systemName: "magnifyingglass")
                        .foregroundColor(.secondary)
                    TextField("Search models...", text: $searchText)
                        .textFieldStyle(RoundedBorderTextFieldStyle())
                }
                .padding(.horizontal)
                
                // Category Picker
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 12) {
                        ForEach(categories, id: \.self) { category in
                            CategoryChip(
                                title: category,
                                isSelected: selectedCategory == category,
                                action: { selectedCategory = category }
                            )
                        }
                    }
                    .padding(.horizontal)
                }
                .padding(.vertical, 8)
                
                // Content Area
                ScrollView {
                    LazyVGrid(columns: [
                        GridItem(.flexible()),
                        GridItem(.flexible())
                    ], spacing: 16) {
                        ForEach(0..<12) { index in
                            ModelGridItem(index: index)
                        }
                    }
                    .padding()
                }
            }
            .navigationTitle("Explore")
            .navigationBarTitleDisplayMode(.large)
        }
    }
}

// Category Chip View
struct CategoryChip: View {
    let title: String
    let isSelected: Bool
    let action: () -> Void
    
    var body: some View {
        Button(action: action) {
            Text(title)
                .font(.subheadline)
                .fontWeight(isSelected ? .semibold : .regular)
                .padding(.horizontal, 16)
                .padding(.vertical, 8)
                .background(isSelected ? Color.blue : Color.gray.opacity(0.2))
                .foregroundColor(isSelected ? .white : .primary)
                .clipShape(Capsule())
        }
    }
}

// Model Grid Item
struct ModelGridItem: View {
    let index: Int
    
    var body: some View {
        VStack {
            RoundedRectangle(cornerRadius: 12)
                .fill(Color.gray.opacity(0.2))
                .frame(height: 150)
                .overlay(
                    Image(systemName: "cube.transparent")
                        .font(.largeTitle)
                        .foregroundColor(.gray)
                )
            
            Text("Model \(index + 1)")
                .font(.caption)
                .fontWeight(.medium)
            
            Text("Category")
                .font(.caption2)
                .foregroundColor(.secondary)
        }
    }
}

// Favorites Tab
struct FavoritesTab: View {
    @State private var favoriteModels: [String] = []
    
    var body: some View {
        NavigationStack {
            VStack {
                if favoriteModels.isEmpty {
                    ContentUnavailableView(
                        "No Favorites Yet",
                        systemImage: "heart.slash",
                        description: Text("Models you mark as favorites will appear here")
                    )
                } else {
                    List(favoriteModels, id: \.self) { model in
                        HStack {
                            Image(systemName: "cube.fill")
                                .foregroundColor(.blue)
                            Text(model)
                            Spacer()
                            Image(systemName: "heart.fill")
                                .foregroundColor(.red)
                        }
                    }
                }
            }
            .navigationTitle("Favorites")
            .navigationBarTitleDisplayMode(.large)
        }
    }
}

// Profile Tab
struct ProfileTab: View {
    @ObservedObject var authManager: AuthenticationManager
    @EnvironmentObject private var appStateManager: AppStateManager
    @State private var showLogoutConfirmation = false
    
    var body: some View {
        NavigationStack {
            List {
                // User Info Section
                Section {
                    HStack {
                        Image(systemName: "person.circle.fill")
                            .font(.system(size: 60))
                            .foregroundColor(.blue)
                        
                        VStack(alignment: .leading, spacing: 4) {
                            Text(authManager.currentUser?.name ?? "User")
                                .font(.title3)
                                .fontWeight(.semibold)
                            
                            Text(authManager.currentUser?.phoneNumber ?? "")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                        
                        Spacer()
                    }
                    .padding(.vertical, 8)
                }
                
                // Account Section
                Section("Account") {
                    HStack {
                        Label("Name", systemImage: "person.fill")
                        Spacer()
                        Text(authManager.currentUser?.name ?? "")
                            .foregroundColor(.secondary)
                    }
                    
                    HStack {
                        Label("Phone", systemImage: "phone.fill")
                        Spacer()
                        Text(authManager.currentUser?.phoneNumber ?? "")
                            .foregroundColor(.secondary)
                    }
                    
                    NavigationLink(destination: EditProfileView()) {
                        Label("Edit Profile", systemImage: "pencil")
                    }
                }
                
                // Settings Section
                Section("Settings") {
                    NavigationLink(destination: NotificationSettingsView()) {
                        Label("Notifications", systemImage: "bell")
                    }
                    
                    NavigationLink(destination: PrivacySettingsView()) {
                        Label("Privacy", systemImage: "lock")
                    }
                    
                    NavigationLink(destination: GeneralSettingsView()) {
                        Label("General", systemImage: "gearshape")
                    }
                }
                
                // About Section
                Section("About") {
                    HStack {
                        Label("Version", systemImage: "info.circle")
                        Spacer()
                        Text("1.0.0")
                            .foregroundColor(.secondary)
                    }
                    
                    NavigationLink(destination: AboutView()) {
                        Label("About Furnit", systemImage: "questionmark.circle")
                    }
                    
                    NavigationLink(destination: SupportView()) {
                        Label("Help & Support", systemImage: "lifepreserver")
                    }
                }
                
                // Logout Section
                Section {
                    Button(action: {
                        showLogoutConfirmation = true
                    }) {
                        HStack {
                            Spacer()
                            Label("Logout", systemImage: "arrow.right.square")
                                .foregroundColor(.red)
                            Spacer()
                        }
                    }
                }
            }
            .navigationTitle("Profile")
            .navigationBarTitleDisplayMode(.large)
            .alert("Logout", isPresented: $showLogoutConfirmation) {
                Button("Cancel", role: .cancel) { }
                Button("Logout", role: .destructive) {
                    authManager.logout()
                }
            } message: {
                Text("Are you sure you want to logout?")
            }
        }
    }
}

// Placeholder Views for Navigation Destinations
struct EditProfileView: View {
    var body: some View {
        Text("Edit Profile")
            .navigationTitle("Edit Profile")
            .navigationBarTitleDisplayMode(.inline)
    }
}

struct NotificationSettingsView: View {
    var body: some View {
        Text("Notification Settings")
            .navigationTitle("Notifications")
            .navigationBarTitleDisplayMode(.inline)
    }
}

struct PrivacySettingsView: View {
    var body: some View {
        Text("Privacy Settings")
            .navigationTitle("Privacy")
            .navigationBarTitleDisplayMode(.inline)
    }
}

struct GeneralSettingsView: View {
    var body: some View {
        Text("General Settings")
            .navigationTitle("General")
            .navigationBarTitleDisplayMode(.inline)
    }
}

struct AboutView: View {
    var body: some View {
        Text("About Furnit")
            .navigationTitle("About")
            .navigationBarTitleDisplayMode(.inline)
    }
}

struct SupportView: View {
    var body: some View {
        Text("Help & Support")
            .navigationTitle("Support")
            .navigationBarTitleDisplayMode(.inline)
    }
}

// Model Row View
struct ModelRowView: View {
    let model: USDZModel
    
    var body: some View {
        HStack {
            Image(systemName: "cube.fill")
                .foregroundColor(.blue)
                .font(.title2)
                .frame(width: 40, height: 40)
                .background(Color.blue.opacity(0.1))
                .cornerRadius(8)
            
            VStack(alignment: .leading, spacing: 4) {
                Text(model.displayName)
                    .font(.headline)
                    .foregroundColor(.primary)
                
                Text("3D Room Model")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            
            Spacer()
            
            Image(systemName: "chevron.right")
                .foregroundColor(.secondary)
                .font(.caption)
        }
        .padding(.vertical, 4)
    }
}

// MARK: - Room Scanner View (Actual Implementation)
struct RoomScannerView: View {
    @Binding var isPresented: Bool
    @State private var capturedImage: UIImage?
    @State private var showingCamera = false
    @State private var isProcessing = false
    @State private var statusMessage = "Ready to scan room"
    @StateObject private var ar3DModelProcessor = AR3DModelProcessor()
    
    var body: some View {
        NavigationView {
            ZStack {
                // Background gradient
                LinearGradient(
                    gradient: Gradient(colors: [Color.black, Color.blue.opacity(0.3)]),
                    startPoint: .top,
                    endPoint: .bottom
                )
                .ignoresSafeArea()
                
                if showingCamera {
                    // Use your existing SimpleCameraOverlay
                    SimpleCameraOverlay(
                        capturedImage: $capturedImage,
                        isShowingCamera: $showingCamera
                    )
                    .zIndex(100)
                    .transition(.opacity)
                } else if isProcessing {
                    // Processing view
                    VStack(spacing: 20) {
                        ProgressView()
                            .progressViewStyle(CircularProgressViewStyle(tint: .white))
                            .scaleEffect(1.5)
                        
                        Text(statusMessage)
                            .font(.headline)
                            .foregroundColor(.white)
                    }
                } else {
                    // Main scanner interface
                    VStack(spacing: 30) {
                        Spacer()
                        
                        // Room/space scanning icon
                        Image(systemName: "cube.transparent")
                            .font(.system(size: 100))
                            .foregroundColor(.white)
                            .shadow(radius: 10)
                        
                        Text("Room Scanner")
                            .font(.largeTitle)
                            .fontWeight(.bold)
                            .foregroundColor(.white)
                        
                        Text("Scan rooms to create 3D models")
                            .font(.title3)
                            .foregroundColor(.white.opacity(0.8))
                        
                        VStack(spacing: 16) {
                            Text("How it works:")
                                .font(.headline)
                                .foregroundColor(.white)
                            
                            VStack(alignment: .leading, spacing: 12) {
                                HStack {
                                    Image(systemName: "1.circle.fill")
                                        .foregroundColor(.blue)
                                    Text("Point camera around room")
                                        .foregroundColor(.white.opacity(0.9))
                                }
                                
                                HStack {
                                    Image(systemName: "2.circle.fill")
                                        .foregroundColor(.blue)
                                    Text("Capture room layout")
                                        .foregroundColor(.white.opacity(0.9))
                                }
                                
                                HStack {
                                    Image(systemName: "3.circle.fill")
                                        .foregroundColor(.blue)
                                    Text("Generate 3D room model")
                                        .foregroundColor(.white.opacity(0.9))
                                }
                            }
                            .font(.subheadline)
                        }
                        .padding(.horizontal, 40)
                        
                        Spacer()
                        
                        // Start Scanning button
                        Button(action: {
                            startScanning()
                        }) {
                            HStack {
                                Image(systemName: "camera.fill")
                                Text("Start Scanning")
                                    .fontWeight(.semibold)
                            }
                            .font(.title3)
                            .foregroundColor(.white)
                            .frame(width: 250, height: 60)
                            .background(
                                LinearGradient(
                                    gradient: Gradient(colors: [Color.blue, Color.purple]),
                                    startPoint: .leading,
                                    endPoint: .trailing
                                )
                            )
                            .cornerRadius(30)
                            .shadow(radius: 10)
                        }
                        
                        Spacer()
                    }
                    .padding()
                }
            }
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Close") {
                        isPresented = false
                    }
                    .foregroundColor(.white)
                }
            }
            .preferredColorScheme(.dark)
        }
        .onChange(of: capturedImage) { _, newImage in
            if let image = newImage {
                processScannedImage(image)
            }
        }
    }
    
    private func startScanning() {
        print("🚀 Starting room scan")
        showingCamera = true
        statusMessage = "Capture room with camera"
    }
    
    private func processScannedImage(_ image: UIImage) {
        Task {
            await MainActor.run {
                showingCamera = false
                isProcessing = true
                statusMessage = "Processing room capture..."
            }
            
            // Process the room image
            print("📸 Processing room image")
            
            // Generate 3D model using your existing processor
            await MainActor.run {
                statusMessage = "Generating 3D room model..."
            }
            
            guard let generated3DModel = await ar3DModelProcessor.processImage(image) else {
                await MainActor.run {
                    statusMessage = "Failed to generate 3D room model"
                    isProcessing = false
                }
                
                // Show error and allow retry
                DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
                    capturedImage = nil
                    statusMessage = "Ready to scan room"
                }
                return
            }
            
            await MainActor.run {
                statusMessage = "3D room model created successfully!"
                isProcessing = false
            }
            
            // Here you could:
            // 1. Save the room model to the user's collection
            // 2. Open it in ModelViewerView
            // 3. Show a preview with options
            
            DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
                // Reset for next scan or close
                capturedImage = nil
                statusMessage = "Ready to scan room"
                // Optionally close the scanner
                // isPresented = false
            }
        }
    }
}

// MARK: - Room Scanner Test View (Keep for reference)
struct RoomScannerTestView: View {
    @Binding var isPresented: Bool
    
    var body: some View {
        NavigationView {
            VStack(spacing: 30) {
                Image(systemName: "camera.viewfinder")
                    .font(.system(size: 80))
                    .foregroundColor(.blue)
                
                Text("Room Scanner")
                    .font(.largeTitle)
                    .fontWeight(.bold)
                
                Text("This is a test view")
                    .font(.title3)
                    .foregroundColor(.secondary)
                
                Text("Replace this with ModelViewer3D()")
                    .font(.caption)
                    .foregroundColor(.secondary)
                
                Spacer()
                
                Button(action: {
                    isPresented = false
                }) {
                    Text("Close")
                        .font(.headline)
                        .foregroundColor(.white)
                        .frame(width: 200, height: 50)
                        .background(Color.blue)
                        .cornerRadius(10)
                }
                .padding()
            }
            .padding()
            .navigationTitle("Room Scanner")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Cancel") {
                        isPresented = false
                    }
                }
            }
        }
    }
}

#Preview {
    ContentView()
}
