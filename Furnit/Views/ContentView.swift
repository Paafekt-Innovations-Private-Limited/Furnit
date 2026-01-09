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
    @State private var showingFileInfoSnackbar = false
    @State private var selectedModelForInfo: USDZModel?
    
    var body: some View {
        NavigationStack {
            VStack {
                if modelManager.models.isEmpty {
                    // Empty state with upload suggestion
                    VStack(spacing: 20) {
                        ContentUnavailableView(
                            L10n.Home.noModels,
                            systemImage: "cube.transparent",
                            description: Text(L10n.Home.noModelsDescription)
                        )

                        // Quick action button for empty state
                        Button(action: {
                            checkRoomLimitAndCreate()
                        }) {
                            HStack {
                                Image(systemName: "photo.badge.plus")
                                Text(L10n.Home.createRoom)
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
                        // Room limit banner with total memory
                        HStack {
                            VStack(alignment: .leading, spacing: 4) {
                                Text(L10n.Home.roomsRemaining(limitManager.remainingRooms(), limitManager.roomLimit))
                                    .font(.subheadline)
                                    .fontWeight(.semibold)
                                if limitManager.remainingRooms() <= 3 {
                                    Text(L10n.Home.deleteHint)
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                }
                            }
                            Spacer()
                            // Total memory of all rooms
                            VStack(alignment: .trailing, spacing: 2) {
                                Text("Total")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                                Text(totalMemoryFormatted(models: modelManager.models))
                                    .font(.subheadline)
                                    .fontWeight(.semibold)
                            }
                        }
                        .padding()
                        .background(limitManager.remainingRooms() <= 3 ? Color.orange.opacity(0.1) : Color(.systemGroupedBackground))

                        // Delete hint
                        HStack {
                            Text("💡 \(L10n.Home.swipeHint)")
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
                                            Label(L10n.Common.delete, systemImage: "trash")
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
            .navigationTitle(L10n.Home.title)
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
                    .accessibilityLabel("accessibility.createRoom".localized)
                }
                
                // Help Button
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button {
                        showingHelp = true
                    } label: {
                        Image(systemName: "questionmark.circle")
                            .font(.title3)
                    }
                    .accessibilityLabel("accessibility.help".localized)
                }

                // Settings Button
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button {
                        showingSettings = true
                    } label: {
                        Image(systemName: "gearshape")
                            .font(.title3)
                    }
                    .accessibilityLabel("accessibility.settings".localized)
                }
            }
            // Photo Room Creator Sheet
            .sheet(isPresented: $showingPhotoRoomCreator) {
                NavigationStack {
                    SinglePhotoRoomView()
                        .navigationBarItems(
                            trailing: Button(L10n.Common.back) {
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
                            trailing: Button(L10n.Common.done) {
                                showingHelp = false
                            }
                        )
                }
            }
            // Room Limit Alert
            .alert(L10n.RoomLimit.title, isPresented: $showingLimitAlert) {
                Button(L10n.Common.ok, role: .cancel) { }
            } message: {
                Text(L10n.RoomLimit.message(limitManager.roomLimit))
            }
            // Delete confirmation alert
            .alert(L10n.DeleteRoom.title, isPresented: $showDeleteAlert) {
                Button(L10n.Common.cancel, role: .cancel) {
                    roomToDelete = nil
                }
                Button(L10n.Common.delete, role: .destructive) {
                    if let room = roomToDelete {
                        deleteRoom(room)
                    }
                }
            } message: {
                if let room = roomToDelete {
                    Text(L10n.DeleteRoom.message(room.displayName))
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
        // File info snackbar overlay for PLY files
        .overlay(alignment: .bottom) {
            if showingFileInfoSnackbar, let model = selectedModelForInfo {
                FileInfoSnackbar(
                    model: model,
                    isShowing: $showingFileInfoSnackbar
                )
            }
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

    // MARK: - Total Memory Calculation
    private func totalMemoryFormatted(models: [USDZModel]) -> String {
        var totalBytes: UInt64 = 0
        for model in models {
            if let size = model.fileSize {
                totalBytes += size
            } else if let data = model.dataAsset?.data {
                totalBytes += UInt64(data.count)
            }
        }
        let formatter = ByteCountFormatter()
        formatter.countStyle = .file
        return formatter.string(fromByteCount: Int64(totalBytes))
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
            let _ = logDebug("   - File type: \(model.fileType.rawValue)")
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

                // Handle PLY files - navigate to SharpRoomView (Gaussian splat viewer)
                if model.fileType == .ply {
                    NavigationLink(destination: SharpRoomView(plyURL: modelURL, isNewRoom: false)) {
                        HomeViewModelRow(model: model)
                    }
                    .onAppear {
                        if debugMode {
                            let _ = logDebug("👁️ [HomeTab.modelRow] PLY row appeared for: \(model.displayName)")
                        }
                    }
                } else {
                    // USDZ files - navigate to viewer
                    NavigationLink(destination: ModelViewerView(model: model)) {
                        HomeViewModelRow(model: model)
                    }
                    .onAppear {
                        if debugMode {
                            let _ = logDebug("👁️ [HomeTab.modelRow] Row appeared for: \(model.displayName)")
                        }
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

    // Color based on file type
    private var iconColor: Color {
        switch model.fileType {
        case .usdz:
            return .green
        case .ply:
            return .purple
        }
    }

    var body: some View {
        HStack {
            Image(systemName: model.fileType.iconName)
                .foregroundColor(iconColor)
                .font(.title2)
                .frame(width: 40, height: 40)
                .background(iconColor.opacity(0.1))
                .cornerRadius(8)

            VStack(alignment: .leading, spacing: 4) {
                Text(model.displayName)
                    .font(.headline)
                    .foregroundColor(.primary)
                
                Text(L10n.Home.roomModel)
                    .font(.caption)
                    .foregroundColor(.secondary)

                HStack(spacing: 6) {
                    Text(model.subtitleText)
                        .font(.caption)
                        .foregroundColor(.secondary)

                    // Show file size for all rooms
                    if model.hasFileSize {
                        Text("•")
                            .font(.caption)
                            .foregroundColor(.secondary)
                        Text(model.fileSizeFormatted)
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }
            }

            Spacer()

            // Navigation chevron - both PLY and USDZ are now viewable
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

    private var faqSections: [FAQSection] {
        [
            FAQSection(
                title: "faq.roomCreation".localized,
                icon: "camera.fill",
                items: [
                    FAQItem(question: "faq.howToCreate".localized, answer: "faq.howToCreateAnswer".localized),
                    FAQItem(question: "faq.bestPhotos".localized, answer: "faq.bestPhotosAnswer".localized),
                    FAQItem(question: "faq.generationFailing".localized, answer: "faq.generationFailingAnswer".localized),
                    FAQItem(question: "faq.howManyRooms".localized, answer: "faq.howManyRoomsAnswer".localized),
                    FAQItem(question: "faq.howToSaveRoom".localized, answer: "faq.howToSaveRoomAnswer".localized)
                ]
            ),
            FAQSection(
                title: "faq.aiFeatures".localized,
                icon: "brain.head.profile",
                items: [
                    FAQItem(question: "faq.whatIsBrainIcon".localized, answer: "faq.whatIsBrainIconAnswer".localized),
                    FAQItem(question: "faq.howToScreenshot".localized, answer: "faq.howToScreenshotAnswer".localized),
                    FAQItem(question: "faq.whatIsSegmentation".localized, answer: "faq.whatIsSegmentationAnswer".localized),
                    FAQItem(question: "faq.howToSegment".localized, answer: "faq.howToSegmentAnswer".localized),
                    FAQItem(question: "faq.notDetected".localized, answer: "faq.notDetectedAnswer".localized)
                ]
            ),
            FAQSection(
                title: "faq.roomControls".localized,
                icon: "cube.fill",
                items: [
                    FAQItem(question: "faq.howToView".localized, answer: "faq.howToViewAnswer".localized),
                    FAQItem(question: "faq.howToNavigate".localized, answer: "faq.howToNavigateAnswer".localized),
                    FAQItem(question: "faq.whatIsMemoryDisplay".localized, answer: "faq.whatIsMemoryDisplayAnswer".localized),
                    FAQItem(question: "faq.sampleRoom".localized, answer: "faq.sampleRoomAnswer".localized),
                    FAQItem(question: "faq.accuracy".localized, answer: "faq.accuracyAnswer".localized),
                    FAQItem(question: "faq.adjustDimensions".localized, answer: "faq.adjustDimensionsAnswer".localized)
                ]
            )
        ]
    }

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
                    Text(L10n.Help.cantFind)
                        .font(.subheadline)
                        .foregroundColor(.secondary)

                    Text(L10n.Help.contactDescription)
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
                                Text(L10n.Help.emailSupport)
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
                            Text(L10n.Help.copyEmail)
                                .font(.subheadline)
                        }
                        .foregroundColor(.blue)
                    }
                }
                .padding(.vertical, 8)
            } header: {
                Label(L10n.Help.contactSupport, systemImage: "headphones")
                    .font(.headline)
                    .foregroundColor(.primary)
                    .textCase(nil)
            }
        }
        .navigationTitle(L10n.Help.title)
        .navigationBarTitleDisplayMode(.inline)
        .alert(L10n.Help.emailCopied, isPresented: $showCopiedAlert) {
            Button(L10n.Common.ok, role: .cancel) { }
        } message: {
            Text(L10n.Help.emailCopiedMessage)
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
