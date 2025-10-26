import SwiftUI

struct ContentView: View {
    @StateObject private var authManager = AuthenticationManager()
    
    var body: some View {
        Group {
            if authManager.isAuthenticated {
                HomeViewWithBottomBar(authManager: authManager)
                    .onAppear {
                        print("✅ [ContentView] User is authenticated")
                    }
            } else {
                LoginView()
                    .onAppear {
                        print("❌ [ContentView] User is NOT authenticated")
                    }
            }
        }
        .animation(.easeInOut(duration: 0.3), value: authManager.isAuthenticated)
    }
}

// MARK: - HomeViewWithBottomBar
struct HomeViewWithBottomBar: View {
    @ObservedObject var authManager: AuthenticationManager
    @State private var selectedTab = 0
    
    var body: some View {
        TabView(selection: $selectedTab) {
            // Home Tab
            HomeTab()
                .tabItem {
                    Label("Home", systemImage: "house.fill")
                }
                .tag(0)
            
            // Explore Tab
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
        .onAppear {
            print("🏠 [HomeViewWithBottomBar] Rendering with selected tab: \(selectedTab)")
        }
        .onChange(of: selectedTab) { oldValue, newValue in
            print("🏠 [HomeViewWithBottomBar] Tab changed from \(oldValue) to \(newValue)")
        }
    }
}

// MARK: - Home Tab (UPDATED with Upload Button)
struct HomeTab: View {
    @StateObject private var modelManager = USDZModelManager()
    @State private var showingSettings = false
    @State private var showingPhotoRoomCreator = false  // ✨ NEW
    
    var body: some View {
        NavigationStack {
            VStack {
                if modelManager.models.isEmpty {
                    // Empty state with upload suggestion
                    VStack(spacing: 20) {
                        ContentUnavailableView(
                            "No 3D Models Found",
                            systemImage: "cube.transparent",
                            description: Text("Create your first room from a photo!")
                        )
                        
                        // Quick action button for empty state
                        Button(action: {
                            showingPhotoRoomCreator = true
                        }) {
                            HStack {
                                Image(systemName: "photo.badge.plus")
                                Text("Create Room from Photo")
                            }
                            .font(.headline)
                            .foregroundColor(.white)
                            .padding()
                            .background(Color.blue)
                            .cornerRadius(12)
                        }
                    }
                    .onAppear {
                        print("❌ [HomeTab] Showing 'No Models' view - modelManager.models is EMPTY")
                        print("❌ [HomeTab] Models count: \(modelManager.models.count)")
                    }
                } else {
                    List {
                        ForEach(Array(modelManager.models.enumerated()), id: \.offset) { index, model in
                            if let modelURL = model.temporaryURL {
                                NavigationLink(destination: ModelViewerView(model: model)) {
                                    HomeViewModelRow(model: model)
                                }
                                .onAppear {
                                    print("✅ [HomeTab] Created temporary URL for: \(model.fileName) -> \(modelURL)")
                                }
                            } else {
                                Text("❌ Model data unavailable: \(model.displayName)")
                                    .foregroundColor(.red)
                                    .onAppear {
                                        print("❌ [HomeTab] Could NOT create temporary URL for model: \(model.fileName)")
                                    }
                            }
                        }
                    }
                    .listStyle(PlainListStyle())
                    .onAppear {
                        print("✅ [HomeTab] Showing list with \(modelManager.models.count) models")
                    }
                }
            }
            .navigationTitle("3D Room Models")
            .navigationBarTitleDisplayMode(.large)
            .toolbar {
                // ✨ NEW: Upload Photo Button
                ToolbarItem(placement: .navigationBarLeading) {
                    Button {
                        showingPhotoRoomCreator = true
                    } label: {
                        Image(systemName: "photo.badge.plus")
                            .font(.title3)
                    }
                    .accessibilityLabel("Create room from photo")
                }
                
                // Existing Settings Button
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button {
                        showingSettings = true
                    } label: {
                        Image(systemName: "gearshape")
                            .font(.title3)
                    }
                    .accessibilityLabel("Settings")
                }
            }
            // ✨ NEW: Photo Room Creator Sheet
            .sheet(isPresented: $showingPhotoRoomCreator) {
                NavigationStack {
                    SinglePhotoRoomView()
                        .navigationBarItems(
                            trailing: Button("Done") {
                                showingPhotoRoomCreator = false
                            }
                        )
                }
            }
            // Existing Settings Sheet
            .sheet(isPresented: $showingSettings) {
                SettingsView()
            }
        }
        .onAppear {
            print("🏠 [HomeTab] onAppear - Models count: \(modelManager.models.count)")
            print("🏠 [HomeTab] Models: \(modelManager.models.map { "displayName: \($0.displayName), fileName: \($0.fileName)" })")
        }
    }
}

// MARK: - Explore Tab
struct ExploreTab: View {
    @State private var searchText = ""
    @StateObject private var modelManager = USDZModelManager()
    
    var filteredModels: [USDZModel] {
        if searchText.isEmpty {
            return modelManager.models
        }
        return modelManager.models.filter { $0.displayName.localizedCaseInsensitiveContains(searchText) }
    }
    
    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                // Search Bar
                HStack {
                    Image(systemName: "magnifyingglass")
                        .foregroundColor(.gray)
                    TextField("Search rooms...", text: $searchText)
                }
                .padding(12)
                .background(Color(.systemGray6))
                .cornerRadius(10)
                .padding()
                
                // Categories
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 12) {
                        CategoryButton(title: "All", icon: "square.grid.2x2")
                        CategoryButton(title: "Living Room", icon: "sofa.fill")
                        CategoryButton(title: "Bedroom", icon: "bed.double.fill")
                        CategoryButton(title: "Kitchen", icon: "fork.knife")
                        CategoryButton(title: "Office", icon: "desktopcomputer")
                    }
                    .padding(.horizontal)
                }
                .padding(.bottom)
                
                // Models Grid
                if filteredModels.isEmpty {
                    ContentUnavailableView(
                        "No Results",
                        systemImage: "magnifyingglass",
                        description: Text("Try a different search term")
                    )
                    .onAppear {
                        print("❌ [ExploreTab] Showing 'No Results' - filteredModels is EMPTY")
                        print("❌ [ExploreTab] Search text: '\(searchText)'")
                        print("❌ [ExploreTab] Total models: \(modelManager.models.count)")
                    }
                } else {
                    ScrollView {
                        LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 16) {
                            ForEach(Array(filteredModels.enumerated()), id: \.offset) { index, model in
                                if let modelURL = model.temporaryURL {
                                    NavigationLink(destination: ModelViewerView(model: model)) {
                                        ExploreModelCard(model: model)
                                    }
                                } else {
                                    Text("❌ Unavailable: \(model.displayName)")
                                        .foregroundColor(.red)
                                }
                            }
                        }
                        .padding()
                    }
                    .onAppear {
                        print("✅ [ExploreTab] Showing grid with \(filteredModels.count) models")
                    }
                }
            }
            .navigationTitle("Explore")
            .navigationBarTitleDisplayMode(.large)
        }
        .onAppear {
            print("🔍 [ExploreTab] onAppear - Models count: \(modelManager.models.count)")
            print("🔍 [ExploreTab] Models: \(modelManager.models.map { $0.displayName })")
        }
    }
}

// MARK: - Favorites Tab
struct FavoritesTab: View {
    @StateObject private var modelManager = USDZModelManager()
    
    var body: some View {
        NavigationStack {
            VStack {
                if modelManager.models.isEmpty {
                    ContentUnavailableView(
                        "No Favorites Yet",
                        systemImage: "heart.slash",
                        description: Text("Your favorite rooms will appear here")
                    )
                } else {
                    List {
                        ForEach(Array(modelManager.models.enumerated()), id: \.offset) { index, model in
                            if let modelURL = model.temporaryURL {
                                NavigationLink(destination: ModelViewerView(model: model)) {
                                    HomeViewModelRow(model: model)
                                }
                            }
                        }
                    }
                    .listStyle(PlainListStyle())
                }
            }
            .navigationTitle("Favorites")
            .navigationBarTitleDisplayMode(.large)
        }
    }
}

// MARK: - Profile Tab
struct ProfileTab: View {
    @ObservedObject var authManager: AuthenticationManager
    @State private var showLogoutConfirmation = false
    
    var body: some View {
        NavigationStack {
            List {
                // Profile Header
                Section {
                    HStack(spacing: 16) {
                        // Avatar
                        Circle()
                            .fill(Color.blue.gradient)
                            .frame(width: 60, height: 60)
                            .overlay(
                                Text(String(authManager.currentUser?.name.prefix(1) ?? "U"))
                                    .font(.title)
                                    .fontWeight(.semibold)
                                    .foregroundColor(.white)
                            )
                        
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
                    NavigationLink(destination: EditProfileView()) {
                        Label("Edit Profile", systemImage: "person.fill")
                    }
                    
                    NavigationLink(destination: NotificationSettingsView()) {
                        Label("Notifications", systemImage: "bell.fill")
                    }
                    
                    NavigationLink(destination: PrivacySettingsView()) {
                        Label("Privacy", systemImage: "lock.fill")
                    }
                }
                
                // App Settings
                Section("Settings") {
                    NavigationLink(destination: GeneralSettingsView()) {
                        Label("General", systemImage: "gearshape.fill")
                    }
                    
                    NavigationLink(destination: AboutView()) {
                        Label("About", systemImage: "info.circle.fill")
                    }
                    
                    NavigationLink(destination: SupportView()) {
                        Label("Help & Support", systemImage: "questionmark.circle.fill")
                    }
                }
                
                // App Info
                Section("About") {
                    HStack {
                        Label("Version", systemImage: "info.circle")
                        Spacer()
                        Text("1.0.0")
                            .foregroundColor(.secondary)
                    }
                    
                    HStack {
                        Label("Developer", systemImage: "person.2")
                        Spacer()
                        Text("Furnit Team")
                            .foregroundColor(.secondary)
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

// MARK: - Supporting Views

struct HomeViewModelRow: View {
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

struct ExploreModelCard: View {
    let model: USDZModel
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Image(systemName: "cube.fill")
                .font(.system(size: 50))
                .foregroundColor(.blue)
                .frame(maxWidth: .infinity)
                .frame(height: 120)
                .background(Color.blue.opacity(0.1))
                .cornerRadius(12)
            
            Text(model.displayName)
                .font(.subheadline)
                .fontWeight(.medium)
                .foregroundColor(.primary)
                .lineLimit(2)
        }
        .padding(8)
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(color: .black.opacity(0.1), radius: 4, x: 0, y: 2)
    }
}

struct CategoryButton: View {
    let title: String
    let icon: String
    @State private var isSelected = false
    
    var body: some View {
        Button(action: {
            isSelected.toggle()
        }) {
            HStack(spacing: 6) {
                Image(systemName: icon)
                    .font(.caption)
                Text(title)
                    .font(.subheadline)
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 8)
            .background(isSelected ? Color.blue : Color(.systemGray6))
            .foregroundColor(isSelected ? .white : .primary)
            .cornerRadius(20)
        }
    }
}

// MARK: - Placeholder Views
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

#Preview {
    ContentView()
}
