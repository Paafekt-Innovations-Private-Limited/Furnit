import SwiftUI

struct ContentView: View {
    @EnvironmentObject var authManager: AuthenticationManager

    var body: some View {
        Group {
            if authManager.isAuthenticated {
                HomeViewWithBottomBar(authManager: authManager)
                    .onAppear {
                        logDebug("✅ [ContentView] User is authenticated")
                    }
            } else {
                LoginView()
                    .onAppear {
                        logDebug("❌ [ContentView] User is NOT authenticated")
                    }
            }
        }
        .animation(.easeInOut(duration: 0.3), value: authManager.isAuthenticated)
    }
}

// MARK: - HomeViewWithBottomBar (Bottom bar removed)
struct HomeViewWithBottomBar: View {
    @ObservedObject var authManager: AuthenticationManager
    
    var body: some View {
        // Just show the Home tab directly without TabView
        HomeTab(authManager: authManager)
            .onAppear {
                if AppStateManager.shared.qualitySettings.debugMode {
                    logDebug("🏠 [HomeViewWithBottomBar] Rendering without bottom bar")
                }
            }
    }
}

// MARK: - Home Tab (WITH DELETE FUNCTIONALITY ✅)
struct HomeTab: View {
    @ObservedObject var authManager: AuthenticationManager
    @StateObject private var modelManager = USDZModelManager()
    @StateObject private var limitManager = RoomLimitManager.shared
    @State private var showingSettings = false
    @State private var showingPhotoRoomCreator = false
    @State private var showDeleteAlert = false
    @State private var roomToDelete: USDZModel?
    @State private var showingLimitAlert = false
    @State private var showingHelp = false
    
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
                            checkRoomLimitAndCreate()
                        }) {
                            HStack {
                                Image(systemName: "photo.badge.plus")
                                Text("Create Room from Photo")
                            }
                            .font(.headline)
                            .foregroundColor(.white)
                            .padding()
                            .background(Color.green)
                            .cornerRadius(12)
                        }
                    }
                    .onAppear {
                        logDebug("❌ [HomeTab] Showing 'No Models' view - modelManager.models is EMPTY")
                        logDebug("❌ [HomeTab] Models count: \(modelManager.models.count)")
                    }
                } else {
                    VStack(spacing: 0) {
                        // Room limit banner
                        HStack {
                            VStack(alignment: .leading, spacing: 4) {
                                Text("\(limitManager.remainingRooms()) of \(limitManager.roomLimit) rooms remaining")
                                    .font(.subheadline)
                                    .fontWeight(.semibold)
                                if limitManager.remainingRooms() <= 3 {
                                    Text("Delete old rooms to create new ones")
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                }
                            }
                            Spacer()
                        }
                        .padding()
                        .background(limitManager.remainingRooms() <= 3 ? Color.orange.opacity(0.1) : Color(.systemGroupedBackground))
                        
                        // ✅ Delete hint
                        HStack {
                            Text("💡 Swipe left to delete")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            Spacer()
                        }
                        .padding(.horizontal)
                        .padding(.vertical, 8)
                        .background(Color(.systemGroupedBackground))
                        
                        List {
                            ForEach(Array(modelManager.models.enumerated()), id: \.offset) { index, model in
                                modelRow(for: model, at: index)
                                    // ✅ SWIPE TO DELETE ADDED HERE
                                    .swipeActions(edge: .trailing, allowsFullSwipe: false) {
                                        Button(role: .destructive) {
                                            roomToDelete = model
                                            showDeleteAlert = true
                                        } label: {
                                            Label("Delete", systemImage: "trash")
                                        }
                                    }
                            }
                        }
                        .listStyle(PlainListStyle())
                        .onAppear {
                            if AppStateManager.shared.qualitySettings.debugMode {
                                logDebug("✅ [HomeTab] Showing list with \(modelManager.models.count) models")
                                for (idx, model) in modelManager.models.enumerated() {
                                    logDebug("   [\(idx)] \(model.displayName) - isSavedRoom: \(model.isSavedRoom)")
                                }
                            }
                        }
                    }
                }
            }
            .navigationTitle("3D Room Models")
            .navigationBarTitleDisplayMode(.large)
            .toolbar {
                // Upload Photo Button
                ToolbarItem(placement: .navigationBarLeading) {
                    Button {
                        checkRoomLimitAndCreate()
                    } label: {
                        Image(systemName: "photo.badge.plus")
                            .font(.title3)
                    }
                    .accessibilityLabel("Create room from photo")
                }
                
                // Help Button
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button {
                        showingHelp = true
                    } label: {
                        Image(systemName: "questionmark.circle")
                            .font(.title3)
                    }
                    .accessibilityLabel("Help")
                }

                // Settings Button
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
            // Photo Room Creator Sheet
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
            // Refresh models when sheet closes
            .onChange(of: showingPhotoRoomCreator) { _, isShowing in
                if !isShowing {
                    modelManager.refreshModels()
                    limitManager.updateRoomCount()
                }
            }
            // Listen for room save completion to dismiss sheet
            .onReceive(NotificationCenter.default.publisher(for: NSNotification.Name("DismissPhotoRoomSheet"))) { _ in
                showingPhotoRoomCreator = false
                limitManager.updateRoomCount()
            }
            // Settings Sheet
            .sheet(isPresented: $showingSettings) {
                SettingsView()
                    .environmentObject(authManager)
            }
            // Help Sheet
            .sheet(isPresented: $showingHelp) {
                NavigationStack {
                    SupportView()
                        .navigationBarItems(
                            trailing: Button("Done") {
                                showingHelp = false
                            }
                        )
                }
            }
            // Room Limit Alert
            .alert("Room Limit Reached", isPresented: $showingLimitAlert) {
                Button("OK", role: .cancel) { }
            } message: {
                Text("You've reached the limit of \(limitManager.roomLimit) rooms. Delete some rooms to create new ones.")
            }
            // ✅ DELETE CONFIRMATION ALERT ADDED HERE
            .alert("Delete Room?", isPresented: $showDeleteAlert) {
                Button("Cancel", role: .cancel) {
                    roomToDelete = nil
                }
                Button("Delete", role: .destructive) {
                    if let room = roomToDelete {
                        deleteRoom(room)
                    }
                }
            } message: {
                if let room = roomToDelete {
                    Text("Are you sure you want to delete '\(room.displayName)'? This action cannot be undone.")
                }
            }
        }
        .onAppear {
            if AppStateManager.shared.qualitySettings.debugMode {
                logDebug("🏠 [HomeTab] onAppear - Models count: \(modelManager.models.count)")
                logDebug("🏠 [HomeTab] Models: \(modelManager.models.map { "displayName: \($0.displayName), fileName: \($0.fileName)" })")
            }
            limitManager.updateRoomCount()
        }
    }
    
    // MARK: - Helper Functions
    
    /// Check room limit before creating a new room
    private func checkRoomLimitAndCreate() {
        limitManager.updateRoomCount()
        
        if limitManager.canCreateMoreRooms() {
            showingPhotoRoomCreator = true
        } else {
            showingLimitAlert = true
        }
    }
    
    // ✅ DELETE FUNCTION ADDED HERE
    private func deleteRoom(_ room: USDZModel) {
        logDebug("🗑️ [HomeTab] Deleting room: \(room.displayName)")
        modelManager.deleteModel(id: room.id)
        roomToDelete = nil
        limitManager.updateRoomCount()
    }
    
    // MARK: - Model Row with Logging
    private func modelRow(for model: USDZModel, at index: Int) -> some View {
        let debugMode = AppStateManager.shared.qualitySettings.debugMode
        
        if debugMode {
            let _ = logDebug("📋 [HomeTab.modelRow] ========================================")
            let _ = logDebug("📋 [HomeTab.modelRow] Creating row for model \(index)")
            let _ = logDebug("   - Display name: \(model.displayName)")
            let _ = logDebug("   - File name: \(model.fileName)")
            let _ = logDebug("   - Is saved room: \(model.isSavedRoom)")
        }
        
        return Group {
            if let modelURL = model.temporaryURL {
                if debugMode {
                    let _ = logDebug("✅ [HomeTab.modelRow] URL found for: \(model.displayName)")
                    let _ = logDebug("   - URL path: \(modelURL.path)")
                    let _ = logDebug("   - File exists: \(FileManager.default.fileExists(atPath: modelURL.path))")
                    
                    let fileInfo: Void = {
                        if FileManager.default.fileExists(atPath: modelURL.path) {
                            do {
                                let attributes = try FileManager.default.attributesOfItem(atPath: modelURL.path)
                                if let fileSize = attributes[.size] as? UInt64 {
                                    logDebug("   - File size: \(fileSize) bytes (\(fileSize / 1024 / 1024) MB)")
                                }
                            } catch {
                                logDebug("   - Error getting file info: \(error)")
                            }
                        }
                    }()
                    let _ = fileInfo
                }
                
                NavigationLink(destination: ModelViewerView(model: model)) {
                    HomeViewModelRow(model: model)
                }
                .onAppear {
                    if debugMode {
                        let _ = logDebug("👁️ [HomeTab.modelRow] Row appeared for: \(model.displayName)")
                    }
                }
            } else {
                if debugMode {
                    let _ = logDebug("❌ [HomeTab.modelRow] No URL for: \(model.displayName)")
                }
                Text("❌ Model data unavailable: \(model.displayName)")
                    .foregroundColor(.red)
            }
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
                        logDebug("❌ [ExploreTab] Showing 'No Results' - filteredModels is EMPTY")
                        logDebug("❌ [ExploreTab] Search text: '\(searchText)'")
                    }
                } else {
                    ScrollView {
                        LazyVGrid(columns: [
                            GridItem(.flexible(), spacing: 16),
                            GridItem(.flexible(), spacing: 16)
                        ], spacing: 16) {
                            ForEach(filteredModels) { model in
                                NavigationLink(destination: ModelViewerView(model: model)) {
                                    ExploreModelCard(model: model)
                                }
                                .onAppear {
                                    logDebug("✅ [ExploreTab] Displaying card for: \(model.displayName)")
                                }
                            }
                        }
                        .padding()
                    }
                    .onAppear {
                        logDebug("✅ [ExploreTab] Showing grid with \(filteredModels.count) filtered models")
                    }
                }
            }
            .navigationTitle("Explore")
            .navigationBarTitleDisplayMode(.large)
        }
        .onAppear {
            logDebug("🔍 [ExploreTab] onAppear - Total models: \(modelManager.models.count)")
        }
    }
}

// MARK: - Favorites Tab
struct FavoritesTab: View {
    @StateObject private var modelManager = USDZModelManager()
    @State private var favoriteModels: [USDZModel] = []
    
    var body: some View {
        NavigationStack {
            if favoriteModels.isEmpty {
                ContentUnavailableView(
                    "No Favorites",
                    systemImage: "heart.slash",
                    description: Text("Tap the heart icon on rooms to add them here")
                )
                .onAppear {
                    logDebug("❤️ [FavoritesTab] Showing 'No Favorites' view")
                }
            } else {
                List {
                    ForEach(favoriteModels) { model in
                        NavigationLink(destination: ModelViewerView(model: model)) {
                            HomeViewModelRow(model: model)
                        }
                    }
                }
                .listStyle(PlainListStyle())
                .onAppear {
                    if AppStateManager.shared.qualitySettings.debugMode {
                        logDebug("❤️ [FavoritesTab] Showing list with \(favoriteModels.count) favorites")
                    }
                }
            }
        }
        .navigationTitle("Favorites")
        .navigationBarTitleDisplayMode(.large)
        .onAppear {
            if AppStateManager.shared.qualitySettings.debugMode {
                logDebug("❤️ [FavoritesTab] onAppear - Favorite count: \(favoriteModels.count)")
            }
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
                    HStack(spacing: 12) {
                        // Profile Picture Placeholder
                        Circle()
                            .fill(Color.purple.gradient)
                            .frame(width: 60, height: 60)
                            .overlay(
                                Image(systemName: "person.fill")
                                    .font(.system(size: 30))
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
                .foregroundColor(.green)
                .font(.title2)
                .frame(width: 40, height: 40)
                .background(Color.green.opacity(0.1))
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
                .foregroundColor(.green)
                .frame(maxWidth: .infinity)
                .frame(height: 120)
                .background(Color.green.opacity(0.1))
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
            .background(isSelected ? Color.green : Color(.systemGray6))
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

// MARK: - FAQ Item Model
struct FAQItem: Identifiable {
    let id = UUID()
    let question: String
    let answer: String
}

// MARK: - FAQ Section Model
struct FAQSection: Identifiable {
    let id = UUID()
    let title: String
    let icon: String
    let items: [FAQItem]
}

// MARK: - Help & Support View
struct SupportView: View {
    @State private var expandedFAQs: Set<UUID> = []
    @State private var supportMessage: String = ""
    @State private var showMailComposer = false
    @State private var showCopiedAlert = false

    private let faqSections: [FAQSection] = [
        FAQSection(
            title: "Room Creation",
            icon: "camera.fill",
            items: [
                FAQItem(
                    question: "How do I create a 3D room?",
                    answer: "Tap the photo icon in the top-left corner of the home screen, then take or select a photo of your room. The app will automatically generate a 3D model from your photo."
                ),
                FAQItem(
                    question: "What kind of photos work best?",
                    answer: "For best results, take photos in good lighting with the camera held level. Try to capture the entire room including floors, walls, and ceiling edges. Avoid blurry or dark photos."
                ),
                FAQItem(
                    question: "Why is my room generation failing?",
                    answer: "Room generation may fail if the photo is too dark, blurry, or doesn't show enough room features. Try taking a new photo with better lighting and a wider angle."
                ),
                FAQItem(
                    question: "How many rooms can I create?",
                    answer: "You can create up to 10 rooms. Delete older rooms to make space for new ones. The room count is shown at the top of your home screen."
                )
            ]
        ),
        FAQSection(
            title: "Furniture Segmentation",
            icon: "square.on.square.dashed",
            items: [
                FAQItem(
                    question: "What is furniture segmentation?",
                    answer: "Furniture segmentation uses AI to identify and separate furniture items in your photos, allowing you to see how each piece would fit in your 3D room."
                ),
                FAQItem(
                    question: "How do I segment furniture from a photo?",
                    answer: "When viewing your 3D room, tap on the furniture icon to access the segmentation feature. Select a photo containing furniture, and the app will automatically detect and extract the furniture items."
                ),
                FAQItem(
                    question: "Why isn't my furniture being detected?",
                    answer: "Furniture detection works best with clear, well-lit photos where furniture is clearly visible. Make sure the furniture isn't partially hidden or blending with the background."
                ),
                FAQItem(
                    question: "Can I segment multiple furniture pieces?",
                    answer: "Yes! The AI can detect multiple furniture pieces in a single photo. Each detected item will be shown separately so you can choose which ones to use."
                )
            ]
        ),
        FAQSection(
            title: "3D Room & Fitment",
            icon: "cube.fill",
            items: [
                FAQItem(
                    question: "How do I view my 3D room?",
                    answer: "Tap on any room in your home screen to open the 3D viewer. You can rotate, zoom, and pan around the room using touch gestures."
                ),
                FAQItem(
                    question: "How do I place furniture in my room?",
                    answer: "After segmenting furniture, tap on the furniture piece to add it to your room. You can then drag to position it, pinch to resize, and rotate to adjust its orientation."
                ),
                FAQItem(
                    question: "Can I use a sample room instead of my own?",
                    answer: "Yes! The app provides sample rooms for you to experiment with. Access them from the Explore section to try out furniture placement without creating your own room first."
                ),
                FAQItem(
                    question: "How accurate is the 3D model?",
                    answer: "The 3D model provides a good approximation of your room's layout. For best accuracy, ensure your original photo captures the room dimensions clearly with visible corners and edges."
                ),
                FAQItem(
                    question: "Can I adjust room dimensions?",
                    answer: "Yes, go to Settings and look for Room Dimensions options where you can fine-tune the size and proportions of your generated room."
                )
            ]
        )
    ]

    var body: some View {
        List {
            // FAQ Sections
            ForEach(faqSections) { section in
                Section {
                    ForEach(section.items) { item in
                        FAQRowView(item: item, isExpanded: expandedFAQs.contains(item.id)) {
                            withAnimation(.easeInOut(duration: 0.2)) {
                                if expandedFAQs.contains(item.id) {
                                    expandedFAQs.remove(item.id)
                                } else {
                                    expandedFAQs.insert(item.id)
                                }
                            }
                        }
                    }
                } header: {
                    Label(section.title, systemImage: section.icon)
                        .font(.headline)
                        .foregroundColor(.primary)
                        .textCase(nil)
                }
            }

            // Contact Support Section
            Section {
                VStack(alignment: .leading, spacing: 16) {
                    Text("Can't find what you're looking for?")
                        .font(.subheadline)
                        .foregroundColor(.secondary)

                    Text("Contact our support team and we'll get back to you as soon as possible.")
                        .font(.caption)
                        .foregroundColor(.secondary)

                    // Email support button
                    Button(action: {
                        openMailComposer()
                    }) {
                        HStack {
                            Image(systemName: "envelope.fill")
                                .font(.title3)
                            VStack(alignment: .leading, spacing: 2) {
                                Text("Email Support")
                                    .font(.headline)
                                Text("support@paafekt.com")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                            Spacer()
                            Image(systemName: "arrow.up.right")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                        .padding()
                        .background(Color.green.opacity(0.1))
                        .foregroundColor(.green)
                        .cornerRadius(12)
                    }

                    // Copy email button
                    Button(action: {
                        UIPasteboard.general.string = "support@paafekt.com"
                        showCopiedAlert = true
                    }) {
                        HStack {
                            Image(systemName: "doc.on.doc")
                            Text("Copy Email Address")
                                .font(.subheadline)
                        }
                        .foregroundColor(.blue)
                    }
                }
                .padding(.vertical, 8)
            } header: {
                Label("Contact Support", systemImage: "headphones")
                    .font(.headline)
                    .foregroundColor(.primary)
                    .textCase(nil)
            }
        }
        .navigationTitle("Help & Support")
        .navigationBarTitleDisplayMode(.inline)
        .alert("Email Copied", isPresented: $showCopiedAlert) {
            Button("OK", role: .cancel) { }
        } message: {
            Text("support@paafekt.com has been copied to your clipboard.")
        }
    }

    private func openMailComposer() {
        let email = "support@paafekt.com"
        let subject = "Furnit App Support"
        let body = "Hi Furnit Support Team,\n\nI need help with:\n\n"

        let urlString = "mailto:\(email)?subject=\(subject.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? "")&body=\(body.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? "")"

        if let url = URL(string: urlString) {
            UIApplication.shared.open(url)
        }
    }
}

// MARK: - FAQ Row View
struct FAQRowView: View {
    let item: FAQItem
    let isExpanded: Bool
    let onTap: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Button(action: onTap) {
                HStack {
                    Text(item.question)
                        .font(.subheadline)
                        .fontWeight(.medium)
                        .foregroundColor(.primary)
                        .multilineTextAlignment(.leading)

                    Spacer()

                    Image(systemName: isExpanded ? "chevron.up" : "chevron.down")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(PlainButtonStyle())

            if isExpanded {
                Text(item.answer)
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .padding(.top, 8)
                    .transition(.opacity.combined(with: .move(edge: .top)))
            }
        }
        .padding(.vertical, 4)
    }
}

#Preview {
    ContentView()
}
