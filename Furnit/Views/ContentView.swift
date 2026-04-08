import SwiftUI

// MARK: - Lazy View Wrapper
/// Wrapper that delays view creation until it's actually rendered
/// Prevents NavigationLink from eagerly creating destination views
struct LazyView<Content: View>: View {
    let build: () -> Content

    init(_ build: @autoclosure @escaping () -> Content) {
        self.build = build
    }

    init(@ViewBuilder _ build: @escaping () -> Content) {
        self.build = build
    }

    var body: Content {
        build()
    }
}

@ViewBuilder
private func destinationView(for model: USDZModel) -> some View {
    if let modelURL = model.temporaryURL {
        if model.fileType == .ply {
            SharpRoomView(
                plyURL: modelURL,
                allowSave: false,
                photoOrientation: model.photoOrientation,
                savedRoomWidth: model.roomWidth,
                savedRoomHeight: model.roomHeight,
                savedRoomModel: model
            )
        } else if model.fileType == .meshroom {
            if let imageData = try? Data(contentsOf: modelURL),
               let image = UIImage(data: imageData) {
                MeshRoomView(
                    roomWidth: model.roomWidth ?? 4.0,
                    roomHeight: model.roomHeight ?? 3.0,
                    roomDepth: model.roomDepth ?? 4.0,
                    frontWallImage: image,
                    photoOrientation: model.photoOrientation,
                    savedRoomModel: model
                )
            } else {
                Text("Failed to load room image")
                    .foregroundColor(.red)
            }
        } else if model.fileType == .glb {
            GLBRoomView(
                glbURL: modelURL,
                photoOrientation: model.photoOrientation,
                roomWidth: model.roomWidth,
                roomHeight: model.roomHeight,
                savedRoomModel: model
            )
        } else {
            ModelViewerView(model: model)
        }
    } else {
        Text("❌ Model data unavailable: \(model.displayName)")
            .foregroundColor(.red)
    }
}

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
    @State private var renameTarget: USDZModel?
    @State private var renameDraft = ""
    @State private var createRoomHintExplanationVisible = false
    @State private var createRoomHintHideTextTask: Task<Void, Never>?

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
                                    .swipeActions(edge: .leading, allowsFullSwipe: false) {
                                        if model.isSavedRoom {
                                            Button {
                                                renameTarget = model
                                                renameDraft = model.displayName
                                            } label: {
                                                Label(L10n.Home.renameRoom, systemImage: "pencil")
                                            }
                                            .tint(.blue)
                                        }
                                    }
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
                // Create room + helper hand (same hand symbol as SharpRoomView brain hint)
                ToolbarItem(placement: .navigationBarLeading) {
                    HStack(spacing: 10) {
                        Button {
                            checkRoomLimitAndCreate()
                        } label: {
                            Image(systemName: "photo.badge.plus")
                                .font(.title3)
                        }
                        .accessibilityLabel("accessibility.createRoom".localized)

                        createRoomToolbarHint
                    }
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
                }
            }
            // Refresh models when sheet closes
            .onChange(of: showingPhotoRoomCreator) { _, isShowing in
                if !isShowing {
                    // Photo room sheet fully closed — safe to drop heavy singletons (not during in-sheet navigation).
                    Task { @MainActor in
                        SHARPService.shared.releaseResources()
                        YOLOEModelService.shared.releaseResources()
                    }
                    modelManager.refreshModels()
                    limitManager.updateRoomCount()
                    restartCreateRoomHint()
                } else {
                    cancelCreateRoomHintTasks()
                    createRoomHintExplanationVisible = false
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
            .alert(L10n.Home.renameRoom, isPresented: Binding(
                get: { renameTarget != nil },
                set: { if !$0 { renameTarget = nil } }
            )) {
                TextField(L10n.Home.roomNamePlaceholder, text: $renameDraft)
                Button(L10n.Common.cancel, role: .cancel) {
                    renameTarget = nil
                }
                Button(L10n.Common.save) {
                    if let room = renameTarget {
                        try? modelManager.updateDisplayName(for: room, newName: renameDraft)
                    }
                    renameTarget = nil
                }
            } message: {
                Text(L10n.Home.renameRoomMessage)
            }
        }
        .onAppear {
            if AppStateManager.shared.qualitySettings.debugMode {
                logDebug("🏠 [HomeTab] onAppear - Models count: \(modelManager.models.count)")
                logDebug("🏠 [HomeTab] Models: \(modelManager.models.map { "displayName: \($0.displayName), fileName: \($0.fileName)" })")
            }
            limitManager.updateRoomCount()
            restartCreateRoomHint()
        }
        .onDisappear {
            cancelCreateRoomHintTasks()
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

    private func cancelCreateRoomHintTasks() {
        createRoomHintHideTextTask?.cancel()
        createRoomHintHideTextTask = nil
    }

    private func scheduleCreateRoomHintTextAutoHide(seconds: UInt64 = 3) {
        createRoomHintHideTextTask?.cancel()
        createRoomHintHideTextTask = Task { @MainActor in
            try? await Task.sleep(for: .seconds(seconds))
            guard !Task.isCancelled else { return }
            createRoomHintExplanationVisible = false
        }
    }

    private func restartCreateRoomHint() {
        cancelCreateRoomHintTasks()
        createRoomHintExplanationVisible = true
        scheduleCreateRoomHintTextAutoHide(seconds: 3)
    }

    private func onCreateRoomHintIconTapped() {
        cancelCreateRoomHintTasks()
        createRoomHintExplanationVisible = true
        scheduleCreateRoomHintTextAutoHide(seconds: 3)
    }

    private var createRoomHintAccessibilityLabel: String {
        L10n.Home.createRoomHint + " " + L10n.RoomViewer.gestureHintToggleAccessibility
    }

    private var createRoomToolbarHint: some View {
        HStack(alignment: .center, spacing: 6) {
            if createRoomHintExplanationVisible {
                Text(L10n.Home.createRoomHint)
                    .font(.caption2.weight(.medium))
                    .foregroundStyle(.primary)
                    .multilineTextAlignment(.leading)
                    .lineLimit(2)
                    .minimumScaleFactor(0.85)
                    .frame(maxWidth: 158, alignment: .leading)
                    .fixedSize(horizontal: false, vertical: true)
                    .padding(.horizontal, 7)
                    .padding(.vertical, 4)
                    .background(
                        RoundedRectangle(cornerRadius: 6, style: .continuous)
                            .fill(Color.primary.opacity(0.12))
                    )
                    .transition(.opacity.combined(with: .move(edge: .trailing)))
            }
            Button(action: onCreateRoomHintIconTapped) {
                Image(systemName: "hand.tap.fill")
                    .symbolRenderingMode(.hierarchical)
                    .font(.title3)
                    .foregroundStyle(.primary)
            }
            .buttonStyle(.plain)
            .accessibilityLabel(createRoomHintAccessibilityLabel)
        }
        .fixedSize(horizontal: true, vertical: true)
    }

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
                // Use LazyView to ensure PLY files are only parsed when actually opened
                if model.fileType == .ply {
                    NavigationLink {
                        LazyView {
                            SharpRoomView(
                                plyURL: modelURL,
                                allowSave: false,
                                photoOrientation: model.photoOrientation,
                                savedRoomWidth: model.roomWidth,
                                savedRoomHeight: model.roomHeight,
                                savedRoomModel: model
                            )
                        }
                    } label: {
                        HomeViewModelRow(model: model)
                    }
                    .onAppear {
                        if debugMode {
                            let _ = logDebug("👁️ [HomeTab.modelRow] PLY row appeared for: \(model.displayName)")
                        }
                    }
                } else if model.fileType == .meshroom {
                    // Meshroom files - navigate to MeshRoomView (WebGL box room)
                    NavigationLink {
                        LazyView {
                            // Load image from .meshroom file
                            if let imageData = try? Data(contentsOf: modelURL),
                               let image = UIImage(data: imageData) {
                                MeshRoomView(
                                    roomWidth: model.roomWidth ?? 4.0,
                                    roomHeight: model.roomHeight ?? 3.0,
                                    roomDepth: model.roomDepth ?? 4.0,
                                    frontWallImage: image,
                                    photoOrientation: model.photoOrientation,
                                    savedRoomModel: model
                                )
                            } else {
                                // Fallback - show error
                                Text("Failed to load room image")
                                    .foregroundColor(.red)
                            }
                        }
                    } label: {
                        HomeViewModelRow(model: model)
                    }
                    .onAppear {
                        if debugMode {
                            let _ = logDebug("👁️ [HomeTab.modelRow] Meshroom row appeared for: \(model.displayName)")
                        }
                    }
                } else if model.fileType == .glb {
                    // GLB files - navigate to GLBRoomView (WebGL GLTF viewer)
                    NavigationLink {
                        LazyView {
                            GLBRoomView(
                                glbURL: modelURL,
                                photoOrientation: model.photoOrientation,
                                roomWidth: model.roomWidth,
                                roomHeight: model.roomHeight,
                                savedRoomModel: model
                            )
                        }
                    } label: {
                        HomeViewModelRow(model: model)
                    }
                    .onAppear {
                        if debugMode {
                            let _ = logDebug("👁️ [HomeTab.modelRow] GLB row appeared for: \(model.displayName)")
                        }
                    }
                } else {
                    // USDZ files - navigate to viewer
                    NavigationLink {
                        LazyView {
                            ModelViewerView(model: model)
                        }
                    } label: {
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
                                NavigationLink {
                                    LazyView {
                                        destinationView(for: model)
                                    }
                                } label: {
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
                        NavigationLink {
                            LazyView {
                                destinationView(for: model)
                            }
                        } label: {
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
                Section(L10n.Profile.sectionAccount) {
                    NavigationLink(destination: EditProfileView()) {
                        Label(L10n.Profile.editProfile, systemImage: "person.fill")
                    }
                    
                    NavigationLink(destination: NotificationSettingsView()) {
                        Label(L10n.Profile.notifications, systemImage: "bell.fill")
                    }
                    
                    NavigationLink(destination: PrivacySettingsView()) {
                        Label(L10n.Profile.privacy, systemImage: "lock.fill")
                    }
                }
                
                // App Settings
                Section(L10n.Profile.sectionSettings) {
                    NavigationLink(destination: GeneralSettingsView()) {
                        Label(L10n.Profile.general, systemImage: "gearshape.fill")
                    }
                    
                    NavigationLink(destination: AboutView()) {
                        Label(L10n.Profile.about, systemImage: "info.circle.fill")
                    }
                    
                    NavigationLink(destination: SupportView()) {
                        Label(L10n.Profile.helpSupport, systemImage: "questionmark.circle.fill")
                    }
                }
                
                // App Info
                Section(L10n.Profile.sectionAbout) {
                    HStack {
                        Label(L10n.App.version, systemImage: "info.circle")
                        Spacer()
                        Text("1.0.0")
                            .foregroundColor(.secondary)
                    }
                    
                    HStack {
                        Label(L10n.App.developer, systemImage: "person.2")
                        Spacer()
                        Text(L10n.App.developer)
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
        case .meshroom:
            return .orange
        case .glb:
            return .blue
        }
    }

    // Orientation label text
    private var orientationLabel: String {
        let title: String
        let subtitle: String
        switch model.photoOrientation {
        case .portrait, .square:
            title = NSLocalizedString("orientation.portrait", comment: "")
            subtitle = NSLocalizedString("orientation.heldVertically", comment: "")
        case .landscape:
            title = NSLocalizedString("orientation.landscape", comment: "")
            subtitle = NSLocalizedString("orientation.heldHorizontally", comment: "")
        }
        return "\(title) - \(subtitle)"
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

                if let dims = model.roomDimensionsListLine {
                    Text(dims)
                        .font(.caption.monospaced())
                        .foregroundColor(.green.opacity(0.9))
                }

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

                    // Show orientation
                    Text("•")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Text(orientationLabel)
                        .font(.caption)
                        .foregroundColor(.secondary)
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
        Text(L10n.Profile.notificationSettings)
            .navigationTitle(L10n.Profile.notifications)
            .navigationBarTitleDisplayMode(.inline)
    }
}

struct PrivacySettingsView: View {
    var body: some View {
        Text(L10n.Profile.privacySettings)
            .navigationTitle(L10n.Profile.privacy)
            .navigationBarTitleDisplayMode(.inline)
    }
}

struct GeneralSettingsView: View {
    var body: some View {
        Text(L10n.Profile.generalSettings)
            .navigationTitle(L10n.Profile.general)
            .navigationBarTitleDisplayMode(.inline)
    }
}

struct AboutView: View {
    private var appVersion: String {
        (Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String) ?? "1.0.0"
    }

    var body: some View {
        Form {
            Section {
                HStack {
                    Text(L10n.App.version)
                    Spacer()
                    Text(appVersion)
                        .foregroundColor(.secondary)
                }
                HStack {
                    Text(L10n.App.developer)
                    Spacer()
                    Text(L10n.App.developer)
                        .foregroundColor(.secondary)
                }
            } header: {
                Text(L10n.Profile.about)
            }

            Section {
                Text(L10n.Licenses.phase1Notice)
                    .font(.subheadline)
                    .foregroundColor(.secondary)
            } header: {
                Text(L10n.Licenses.title)
            }

            Section {
                VStack(alignment: .leading, spacing: 6) {
                    Text(L10n.Licenses.yoloeTitle)
                        .font(.subheadline)
                        .fontWeight(.medium)
                    Text(L10n.Licenses.yoloe)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                .padding(.vertical, 4)
            }

            Section {
                VStack(alignment: .leading, spacing: 6) {
                    Text(L10n.Licenses.sharpTitle)
                        .font(.subheadline)
                        .fontWeight(.medium)
                    Text(L10n.Licenses.sharp)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                .padding(.vertical, 4)
            }

            Section {
                VStack(alignment: .leading, spacing: 6) {
                    Text(L10n.Licenses.metalSplatterTitle)
                        .font(.subheadline)
                        .fontWeight(.medium)
                    Text(L10n.Licenses.metalSplatter)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                .padding(.vertical, 4)
            }
        }
        .navigationTitle(L10n.Profile.about)
        .navigationBarTitleDisplayMode(.inline)
    }
}

/// Licenses & Attributions (Settings → Open Source Licenses). Non-commercial Phase 1; includes YOLO11 (AGPL), Sharp ML (MIT), MetalSplatter (MIT), Firebase (Apache-2.0).
struct LicensesView: View {
    private enum LicenseURL {
        static let agpl3 = URL(string: "https://www.gnu.org/licenses/agpl-3.0.html")!
        static let mit = URL(string: "https://opensource.org/licenses/MIT")!
        static let apache2 = URL(string: "https://www.apache.org/licenses/LICENSE-2.0")!
    }

    var body: some View {
        Form {
            Section {
                Text(L10n.Licenses.phase1Notice)
                    .font(.subheadline)
                    .foregroundColor(.secondary)
            } header: {
                Text(L10n.Licenses.title)
            }

            Section {
                Text(L10n.Licenses.openSourceIntro)
                    .font(.subheadline)
                    .foregroundColor(.secondary)
            } header: {
                Text(L10n.Licenses.openSourceSection)
            }

            Section {
                VStack(alignment: .leading, spacing: 6) {
                    Text(L10n.Licenses.yoloeTitle)
                        .font(.subheadline)
                        .fontWeight(.medium)
                    Text(L10n.Licenses.yoloe)
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Link(L10n.Licenses.viewFullLicense, destination: LicenseURL.agpl3)
                        .font(.caption)
                }
                .padding(.vertical, 4)
            }

            Section {
                VStack(alignment: .leading, spacing: 6) {
                    Text(L10n.Licenses.sharpTitle)
                        .font(.subheadline)
                        .fontWeight(.medium)
                    Text(L10n.Licenses.sharp)
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Link(L10n.Licenses.viewFullLicense, destination: LicenseURL.mit)
                        .font(.caption)
                }
                .padding(.vertical, 4)
            }

            Section {
                VStack(alignment: .leading, spacing: 6) {
                    Text(L10n.Licenses.metalSplatterTitle)
                        .font(.subheadline)
                        .fontWeight(.medium)
                    Text(L10n.Licenses.metalSplatter)
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Link(L10n.Licenses.viewFullLicense, destination: LicenseURL.mit)
                        .font(.caption)
                }
                .padding(.vertical, 4)
            }

            Section {
                VStack(alignment: .leading, spacing: 6) {
                    Text(L10n.Licenses.firebaseTitle)
                        .font(.subheadline)
                        .fontWeight(.medium)
                    Text(L10n.Licenses.firebase)
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Link(L10n.Licenses.viewFullLicense, destination: LicenseURL.apache2)
                        .font(.caption)
                }
                .padding(.vertical, 4)
            }
        }
        .navigationTitle(L10n.Settings.licenses)
        .navigationBarTitleDisplayMode(.inline)
    }
}

// MARK: - FAQ Item Model
struct FAQItem: Identifiable {
    let question: String
    let answer: String

    // Use question as stable ID (UUID changes on each view render)
    var id: String { question }
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
    @State private var expandedFAQs: Set<String> = []
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
                    FAQItem(question: "faq.howToTakePhoto".localized, answer: "faq.howToTakePhotoAnswer".localized),
                    FAQItem(question: "faq.depthAwareRoomPhoto".localized, answer: "faq.depthAwareRoomPhotoAnswer".localized),
                    FAQItem(question: "faq.twoMethods".localized, answer: "faq.twoMethodsAnswer".localized),
                    FAQItem(question: "faq.whatIsAIRoom".localized, answer: "faq.whatIsAIRoomAnswer".localized),
                    FAQItem(question: "faq.whatIsManualRoom".localized, answer: "faq.whatIsManualRoomAnswer".localized),
                    FAQItem(question: "faq.whichMethodBetter".localized, answer: "faq.whichMethodBetterAnswer".localized),
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
                title: "faq.furnitureMeasurements".localized,
                icon: "ruler",
                items: [
                    FAQItem(question: "faq.arAssistedSizing".localized, answer: "faq.arAssistedSizingAnswer".localized),
                    FAQItem(question: "faq.measurementPill".localized, answer: "faq.measurementPillAnswer".localized),
                    FAQItem(question: "faq.resetOverlayScale".localized, answer: "faq.resetOverlayScaleAnswer".localized),
                    FAQItem(question: "faq.howToPlace".localized, answer: "faq.howToPlaceAnswer".localized),
                    FAQItem(question: "faq.multiplePieces".localized, answer: "faq.multiplePiecesAnswer".localized),
                    FAQItem(question: "faq.roomFitment".localized, answer: "faq.roomFitmentAnswer".localized)
                ]
            ),
            FAQSection(
                title: "faq.placementIntelligence".localized,
                icon: "paintpalette.fill",
                items: [
                    FAQItem(question: "faq.whatIsPlacementIntelligence".localized, answer: "faq.whatIsPlacementIntelligenceAnswer".localized),
                    FAQItem(question: "faq.furnitureColorAesthetic".localized, answer: "faq.furnitureColorAestheticAnswer".localized),
                    FAQItem(question: "faq.whatDoHarmonyContrastMean".localized, answer: "faq.whatDoHarmonyContrastMeanAnswer".localized),
                    FAQItem(question: "faq.whenDoesAestheticScoreLow".localized, answer: "faq.whenDoesAestheticScoreLowAnswer".localized)
                ]
            ),
            FAQSection(
                title: "faq.roomControls".localized,
                icon: "cube.fill",
                items: [
                    FAQItem(question: "faq.howToView".localized, answer: "faq.howToViewAnswer".localized),
                    FAQItem(question: "faq.howToNavigate".localized, answer: "faq.howToNavigateAnswer".localized),
                    FAQItem(question: "faq.whatDoArrowsDo".localized, answer: "faq.whatDoArrowsDoAnswer".localized),
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
        let subject = L10n.Help.emailSubject
        let body = L10n.Help.emailBody

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
            .onTapGesture {
                onTap()
            }

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
