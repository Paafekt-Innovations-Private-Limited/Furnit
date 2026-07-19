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
            SplatRoomView(
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
                roomDepth: model.roomDepth,
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
    @State private var savedRoomsListRefreshToken = UUID()
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
    @State private var renameErrorMessage = ""
    @State private var showRenameErrorAlert = false

    var body: some View {
        NavigationStack {
            VStack {
                Text(L10n.Home.librarySubtitle)
                    .font(.subheadline)
                    .foregroundStyle(Theme.Palette.textSecondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal, Theme.Space.lg)
                    .padding(.bottom, Theme.Space.sm)

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
                        }
                        .buttonStyle(PrimaryButtonStyle())
                        .padding(.horizontal, Theme.Space.lg)

                        storageLocationNotice
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
                                Text(L10n.Home.roomCount(modelManager.models.count))
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
                        .background(limitManager.remainingRooms() <= 3 ? Theme.Palette.accent.opacity(0.08) : Theme.Palette.surface)

                        storageLocationNotice

                        // Delete hint
                        HStack {
                            Text(L10n.Home.swipeHint)
                                .font(Theme.Typo.caption())
                                .foregroundStyle(Theme.Palette.textSecondary)
                            Spacer()
                        }
                        .padding(.horizontal)
                        .padding(.vertical, Theme.Space.sm)
                        .background(Theme.Palette.background)
                        
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
                                            .tint(Theme.Palette.accent)
                                        }
                                    }
                                    .swipeActions(edge: .trailing, allowsFullSwipe: false) {
                                        // Bundled samples (Scandinavian Minimal / Industrial Loft) cannot be deleted.
                                        if model.isSavedRoom {
                                            Button(role: .destructive) {
                                                roomToDelete = model
                                                showDeleteAlert = true
                                            } label: {
                                                Label(L10n.Common.delete, systemImage: "trash")
                                            }
                                        }
                                    }
                            }
                        }
                        .listStyle(PlainListStyle())
                        .scrollContentBackground(.hidden)
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
            .id(savedRoomsListRefreshToken)
            .paafektScreenBackground()
            .navigationTitle(L10n.Home.libraryTitle)
            .navigationBarTitleDisplayMode(.large)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button {
                        checkRoomLimitAndCreate()
                    } label: {
                        HStack(spacing: 8) {
                            Image(systemName: "photo.badge.plus")
                                .font(.title3)
                            Text(L10n.Home.createRoomToolbarLabel)
                                .font(.caption.weight(.semibold))
                                .lineLimit(1)
                                .minimumScaleFactor(0.85)
                        }
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
            // Room Creator Sheet
            .sheet(isPresented: $showingPhotoRoomCreator) {
                NavigationStack {
                    roomCreatorView
                }
            }
            // Refresh models when sheet closes
            .onChange(of: showingPhotoRoomCreator) { _, isShowing in
                if !isShowing {
                    refreshSavedRoomsList(forceUIReload: true)
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
            // Delete confirmation
            .overlay {
                if showDeleteAlert {
                    PaafektDeleteRoomDialog(isPresented: $showDeleteAlert) {
                        if let room = roomToDelete {
                            deleteRoom(room)
                        }
                        roomToDelete = nil
                    }
                }
            }
            .sheet(isPresented: Binding(
                get: { renameTarget != nil },
                set: { if !$0 { renameTarget = nil } }
            )) {
                PaafektNameRoomSheet(
                    isPresented: Binding(
                        get: { renameTarget != nil },
                        set: { if !$0 { renameTarget = nil } }
                    ),
                    roomName: $renameDraft,
                    title: L10n.RoomViewer.nameYourRoom,
                    onSave: {
                        if let room = renameTarget {
                            do {
                                try modelManager.updateDisplayName(for: room, newName: renameDraft)
                            } catch {
                                renameErrorMessage = error.localizedDescription
                                showRenameErrorAlert = true
                            }
                        }
                        renameTarget = nil
                    }
                )
            }
            .alert("Rename Failed", isPresented: $showRenameErrorAlert) {
                Button(L10n.Common.ok, role: .cancel) { }
            } message: {
                Text(renameErrorMessage)
            }
        }
        .onAppear {
            if AppStateManager.shared.qualitySettings.debugMode {
                logDebug("🏠 [HomeTab] onAppear - Models count: \(modelManager.models.count)")
                logDebug("🏠 [HomeTab] Models: \(modelManager.models.map { "displayName: \($0.displayName), fileName: \($0.fileName)" })")
            }
            refreshSavedRoomsList()
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

    private var storageLocationNotice: some View {
        HStack(alignment: .top, spacing: Theme.Space.md) {
            Image(systemName: "internaldrive.fill")
                .foregroundStyle(Theme.Palette.accent)
                .accessibilityHidden(true)

            VStack(alignment: .leading, spacing: Theme.Space.xs) {
                Text(L10n.Home.storageTitle)
                    .font(Theme.Typo.caption().weight(.semibold))
                    .foregroundStyle(Theme.Palette.textPrimary)
                Text(L10n.Home.storagePath)
                    .font(Theme.Typo.caption())
                    .foregroundStyle(Theme.Palette.textSecondary)
                Text(L10n.Home.storageWarning)
                    .font(Theme.Typo.caption())
                    .foregroundStyle(Theme.Palette.textSecondary)
            }

            Spacer(minLength: 0)
        }
        .padding(.horizontal, Theme.Space.lg)
        .padding(.vertical, Theme.Space.md)
        .background(Theme.Palette.surface)
    }

    private func refreshSavedRoomsList(forceUIReload: Bool = false) {
        modelManager.refreshModels()
        limitManager.updateRoomCount()
        if forceUIReload {
            savedRoomsListRefreshToken = UUID()
        }
    }

    @ViewBuilder
    private var roomCreatorView: some View {
        SinglePhotoRoomView()
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
        guard room.isSavedRoom else {
            logDebug("🗑️ [HomeTab] Skipping delete — bundled sample room: \(room.displayName)")
            roomToDelete = nil
            return
        }
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

                // Handle PLY files - navigate to SplatRoomView (Gaussian splat viewer)
                // Use LazyView to ensure PLY files are only parsed when actually opened
                if model.fileType == .ply {
                    NavigationLink {
                        LazyView {
                            SplatRoomView(
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
                    // Meshroom files - navigate to MeshRoomView
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
                                roomDepth: model.roomDepth,
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

// MARK: - Supporting Views

struct HomeViewModelRow: View {
    let model: USDZModel

    private var fileTypeTag: String {
        model.fileType.rawValue.uppercased()
    }

    /// Under the room name: Splat PLY vs manual mesh/GLB vs bundle USDZ.
    private var roomCreationKindSubtitle: String {
        switch model.fileType {
        case .ply:
            return L10n.Home.aiBased3DRoom
        case .meshroom, .glb:
            return L10n.Home.manualBased3DRoom
        case .usdz:
            return L10n.Home.roomModel
        }
    }

    var body: some View {
        HStack(spacing: Theme.Space.md) {
            Image(systemName: model.fileType.iconName)
                .foregroundStyle(Theme.Palette.textPrimary)
                .font(.title3)
                .paafektIconTile()

            VStack(alignment: .leading, spacing: Theme.Space.xs) {
                Text(model.displayName)
                    .font(Theme.Typo.headline())
                    .foregroundStyle(Theme.Palette.textPrimary)

                Text(roomCreationKindSubtitle)
                    .font(Theme.Typo.caption())
                    .foregroundStyle(Theme.Palette.textSecondary)
                    .lineLimit(1)

                HStack(spacing: Theme.Space.sm) {
                    Text(fileTypeTag)
                        .font(Theme.Typo.tag())
                        .foregroundStyle(Theme.Palette.textSecondary)

                    if model.hasFileSize {
                        Text(model.fileSizeFormatted)
                            .font(Theme.Typo.caption())
                            .foregroundStyle(Theme.Palette.textSecondary)
                    }
                }
            }

            Spacer()

            Image(systemName: "chevron.right")
                .foregroundStyle(Theme.Palette.textSecondary)
                .font(.caption)
        }
        .padding(.vertical, Theme.Space.xs)
    }
}

/// Licenses & Attributions (Settings → Open Source Licenses). Depth Anything Small (Apache-2.0), GeoCalib (code Apache-2.0 / weights CC BY 4.0), MetalSplatter (MIT), Firebase (Apache-2.0), RTMDet/MMDetection (Apache-2.0), Three.js (MIT).
struct LicensesView: View {
    private enum LicenseURL {
        static let mit = URL(string: "https://opensource.org/licenses/MIT")!
        static let apache2 = URL(string: "https://www.apache.org/licenses/LICENSE-2.0")!
        static let ccBy4 = URL(string: "https://creativecommons.org/licenses/by/4.0/legalcode")!
        static let hypersim = URL(string: "https://github.com/apple/ml-hypersim")!
        static let spzSwift = URL(string: "https://github.com/scier/spz-swift")!
        static let coco = URL(string: "https://cocodataset.org/#termsofuse")!
    }

    var body: some View {
        Form {
            Section {
                Text(L10n.Licenses.openSourceIntro)
                    .font(.subheadline)
                    .foregroundColor(.secondary)
            } header: {
                Text(L10n.Licenses.openSourceSection)
            }

            Section {
                VStack(alignment: .leading, spacing: 6) {
                    Text(L10n.Licenses.depthAnythingTitle)
                        .font(.subheadline)
                        .fontWeight(.medium)
                    Text(L10n.Licenses.depthAnything)
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Link(L10n.Licenses.viewFullLicense, destination: LicenseURL.apache2)
                        .font(.caption)
                }
                .padding(.vertical, 4)
            }

            Section {
                VStack(alignment: .leading, spacing: 6) {
                    Text(L10n.Licenses.hypersimTitle)
                        .font(.subheadline)
                        .fontWeight(.medium)
                    Text(L10n.Licenses.hypersim)
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Link(L10n.Licenses.viewFullLicense, destination: LicenseURL.hypersim)
                        .font(.caption)
                }
                .padding(.vertical, 4)
            }

            Section {
                VStack(alignment: .leading, spacing: 6) {
                    Text(L10n.Licenses.geoCalibTitle)
                        .font(.subheadline)
                        .fontWeight(.medium)
                    Text(L10n.Licenses.geoCalib)
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Link(L10n.Licenses.viewCcBy4, destination: LicenseURL.ccBy4)
                        .font(.caption)
                    Link(L10n.Licenses.viewApache2Code, destination: LicenseURL.apache2)
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
                    Text(L10n.Licenses.spzSwiftTitle)
                        .font(.subheadline)
                        .fontWeight(.medium)
                    Text(L10n.Licenses.spzSwift)
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Link(L10n.Licenses.viewFullLicense, destination: LicenseURL.spzSwift)
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

            Section {
                VStack(alignment: .leading, spacing: 6) {
                    Text(L10n.Licenses.rtmdetTitle)
                        .font(.subheadline)
                        .fontWeight(.medium)
                    Text(L10n.Licenses.rtmdet)
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Link(L10n.Licenses.viewFullLicense, destination: LicenseURL.apache2)
                        .font(.caption)
                }
                .padding(.vertical, 4)
            }

            Section {
                VStack(alignment: .leading, spacing: 6) {
                    Text(L10n.Licenses.cocoTitle)
                        .font(.subheadline)
                        .fontWeight(.medium)
                    Text(L10n.Licenses.coco)
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Link(L10n.Licenses.viewFullLicense, destination: LicenseURL.coco)
                        .font(.caption)
                }
                .padding(.vertical, 4)
            }

            Section {
                VStack(alignment: .leading, spacing: 6) {
                    Text(L10n.Licenses.threeTitle)
                        .font(.subheadline)
                        .fontWeight(.medium)
                    Text(L10n.Licenses.three)
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Link(L10n.Licenses.viewFullLicense, destination: LicenseURL.mit)
                        .font(.caption)
                }
                .padding(.vertical, 4)
            }
        }
        .navigationTitle(L10n.Settings.licenses)
        .navigationBarTitleDisplayMode(.inline)
    }
}

struct CreditsView: View {
    private struct CreditEntry: Identifiable {
        let id: String
        let title: String
        let body: String
        let url: URL
    }

    private var creditEntries: [CreditEntry] {
        [
            CreditEntry(
                id: "apple",
                title: L10n.Credits.appleTitle,
                body: L10n.Credits.appleBody,
                url: URL(string: "https://www.apple.com/")!
            ),
            CreditEntry(
                id: "google",
                title: L10n.Credits.googleTitle,
                body: L10n.Credits.googleBody,
                url: URL(string: "https://about.google/")!
            ),
            CreditEntry(
                id: "depthAnything",
                title: L10n.Credits.depthAnythingTitle,
                body: L10n.Credits.depthAnythingBody,
                url: URL(string: "https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf")!
            ),
            CreditEntry(
                id: "geoCalib",
                title: L10n.Credits.geoCalibTitle,
                body: L10n.Credits.geoCalibBody,
                url: URL(string: "https://github.com/cvg/GeoCalib")!
            ),
            CreditEntry(
                id: "rtmdet",
                title: L10n.Credits.rtmdetTitle,
                body: L10n.Credits.rtmdetBody,
                url: URL(string: "https://github.com/open-mmlab/mmdetection")!
            ),
            CreditEntry(
                id: "metalSplatter",
                title: L10n.Credits.metalSplatterTitle,
                body: L10n.Credits.metalSplatterBody,
                url: URL(string: "https://github.com/scier/MetalSplatter")!
            ),
            CreditEntry(
                id: "three",
                title: L10n.Credits.threeTitle,
                body: L10n.Credits.threeBody,
                url: URL(string: "https://threejs.org/")!
            ),
            CreditEntry(
                id: "hypersim",
                title: L10n.Credits.hypersimTitle,
                body: L10n.Credits.hypersimBody,
                url: URL(string: "https://github.com/apple/ml-hypersim")!
            ),
            CreditEntry(
                id: "coco",
                title: L10n.Credits.cocoTitle,
                body: L10n.Credits.cocoBody,
                url: URL(string: "https://cocodataset.org/")!
            ),
            CreditEntry(
                id: "openAI",
                title: L10n.Credits.openAITitle,
                body: L10n.Credits.openAIBody,
                url: URL(string: "https://openai.com/")!
            ),
            CreditEntry(
                id: "anthropic",
                title: L10n.Credits.anthropicTitle,
                body: L10n.Credits.anthropicBody,
                url: URL(string: "https://www.anthropic.com/")!
            ),
        ]
    }

    var body: some View {
        Form {
            Section {
                Text(L10n.Credits.intro)
                    .font(.subheadline)
                    .foregroundColor(.secondary)
            } header: {
                Text(L10n.Credits.title)
            }

            Section {
                Text(L10n.Credits.disclaimer)
                    .font(.subheadline)
                    .foregroundColor(.secondary)
            }

            ForEach(creditEntries) { entry in
                Section {
                    VStack(alignment: .leading, spacing: 6) {
                        Text(entry.title)
                            .font(.subheadline)
                            .fontWeight(.medium)
                        Text(entry.body)
                            .font(.caption)
                            .foregroundColor(.secondary)
                        Link(L10n.Credits.visitWebsite, destination: entry.url)
                            .font(.caption)
                    }
                    .padding(.vertical, 4)
                }
            }
        }
        .navigationTitle(L10n.Credits.title)
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
                    FAQItem(question: "faq.bestPhotos".localized, answer: "faq.bestPhotosAnswer".localized),
                    FAQItem(question: "faq.howManyRooms".localized, answer: "faq.howManyRoomsAnswer".localized),
                    FAQItem(question: "faq.howToView".localized, answer: "faq.howToViewAnswer".localized),
                    FAQItem(question: "faq.howToNavigate".localized, answer: "faq.howToNavigateAnswer".localized)
                ]
            ),
            FAQSection(
                title: "faq.aiRoomManualSetup".localized,
                icon: "cube.transparent",
                items: [
                    FAQItem(question: "faq.twoMethods".localized, answer: "faq.twoMethodsAnswer".localized),
                    FAQItem(question: "faq.whatIsAIRoom".localized, answer: "faq.whatIsAIRoomAnswer".localized),
                    FAQItem(question: "faq.whatIsManualRoom".localized, answer: "faq.whatIsManualRoomAnswer".localized),
                    FAQItem(question: "faq.whichMethodBetter".localized, answer: "faq.whichMethodBetterAnswer".localized),
                    FAQItem(question: "faq.depthAwareRoomPhoto".localized, answer: "faq.depthAwareRoomPhotoAnswer".localized)
                ]
            ),
            FAQSection(
                title: "faq.aiFeatures".localized,
                icon: "brain.head.profile",
                items: [
                    FAQItem(question: "faq.whatIsBrainIcon".localized, answer: "faq.whatIsBrainIconAnswer".localized),
                    FAQItem(question: "faq.whatIsViewfinderButton".localized, answer: "faq.whatIsViewfinderButtonAnswer".localized),
                    FAQItem(question: "faq.whatIsSegmentation".localized, answer: "faq.whatIsSegmentationAnswer".localized),
                    FAQItem(question: "faq.howToSegment".localized, answer: "faq.howToSegmentAnswer".localized),
                    FAQItem(question: "faq.arAssistedSizing".localized, answer: "faq.arAssistedSizingAnswer".localized),
                    FAQItem(question: "faq.measurementPill".localized, answer: "faq.measurementPillAnswer".localized),
                    FAQItem(question: "faq.resetOverlayScale".localized, answer: "faq.resetOverlayScaleAnswer".localized),
                    FAQItem(question: "faq.pinchOverlaySize".localized, answer: "faq.pinchOverlaySizeAnswer".localized),
                    FAQItem(question: "faq.howToPlace".localized, answer: "faq.howToPlaceAnswer".localized),
                    FAQItem(question: "faq.multiplePieces".localized, answer: "faq.multiplePiecesAnswer".localized)
                ]
            ),
            FAQSection(
                title: "faq.placementIntelligence".localized,
                icon: "square.split.2x2.fill",
                items: [
                    FAQItem(question: "faq.whatIsPlacementIntelligence".localized, answer: "faq.whatIsPlacementIntelligenceAnswer".localized),
                    FAQItem(question: "faq.freeFloorLocation".localized, answer: "faq.freeFloorLocationAnswer".localized),
                    FAQItem(question: "faq.dimensionalFit".localized, answer: "faq.dimensionalFitAnswer".localized),
                    FAQItem(question: "faq.roomFitment".localized, answer: "faq.roomFitmentAnswer".localized)
                ]
            ),
            FAQSection(
                title: "faq.aestheticGuidance".localized,
                icon: "paintpalette.fill",
                items: [
                    FAQItem(question: "faq.aestheticScore".localized, answer: "faq.aestheticScoreAnswer".localized),
                    FAQItem(question: "faq.furnitureColorAesthetic".localized, answer: "faq.furnitureColorAestheticAnswer".localized),
                    FAQItem(question: "faq.whatDoHarmonyContrastMean".localized, answer: "faq.whatDoHarmonyContrastMeanAnswer".localized),
                    FAQItem(question: "faq.aestheticUnavailable".localized, answer: "faq.aestheticUnavailableAnswer".localized),
                    FAQItem(question: "faq.whenDoesAestheticScoreLow".localized, answer: "faq.whenDoesAestheticScoreLowAnswer".localized)
                ]
            ),
            FAQSection(
                title: "faq.measurementsAccuracy".localized,
                icon: "ruler",
                items: [
                    FAQItem(question: "faq.accuracy".localized, answer: "faq.accuracyAnswer".localized),
                    FAQItem(question: "faq.adjustDimensions".localized, answer: "faq.adjustDimensionsAnswer".localized),
                    FAQItem(question: "faq.purchaseReliance".localized, answer: "faq.purchaseRelianceAnswer".localized)
                ]
            ),
            FAQSection(
                title: "faq.savedRoomsStorage".localized,
                icon: "internaldrive.fill",
                items: [
                    FAQItem(question: "faq.howToSaveRoom".localized, answer: "faq.howToSaveRoomAnswer".localized),
                    FAQItem(question: "faq.whereRoomsSaved".localized, answer: "faq.whereRoomsSavedAnswer".localized),
                    FAQItem(question: "faq.uninstallRooms".localized, answer: "faq.uninstallRoomsAnswer".localized),
                    FAQItem(question: "faq.whatIsMemoryDisplay".localized, answer: "faq.whatIsMemoryDisplayAnswer".localized)
                ]
            ),
            FAQSection(
                title: "faq.sharingExporting".localized,
                icon: "square.and.arrow.up",
                items: [
                    FAQItem(question: "faq.exportRoom".localized, answer: "faq.exportRoomAnswer".localized),
                    FAQItem(question: "faq.howToScreenshot".localized, answer: "faq.howToScreenshotAnswer".localized)
                ]
            ),
            FAQSection(
                title: "faq.privacyAccount".localized,
                icon: "lock.shield.fill",
                items: [
                    FAQItem(question: "faq.arePhotosUploaded".localized, answer: "faq.arePhotosUploadedAnswer".localized),
                    FAQItem(question: "faq.firebaseData".localized, answer: "faq.firebaseDataAnswer".localized),
                    FAQItem(question: "faq.deleteAccountEffect".localized, answer: "faq.deleteAccountEffectAnswer".localized)
                ]
            ),
            FAQSection(
                title: "faq.troubleshootingSupport".localized,
                icon: "wrench.and.screwdriver.fill",
                items: [
                    FAQItem(question: "faq.generationFailing".localized, answer: "faq.generationFailingAnswer".localized),
                    FAQItem(question: "faq.notDetected".localized, answer: "faq.notDetectedAnswer".localized),
                    FAQItem(question: "faq.whatDoArrowsDo".localized, answer: "faq.whatDoArrowsDoAnswer".localized),
                    FAQItem(question: "faq.contactSupport".localized, answer: "faq.contactSupportAnswer".localized)
                ]
            )
        ]
    }

    var body: some View {
        List {
            Section {
                VStack(alignment: .leading, spacing: 8) {
                    Label(L10n.Help.measurementAccuracyTitle, systemImage: "exclamationmark.triangle.fill")
                        .font(.headline)
                        .foregroundColor(.orange)

                    Text(L10n.Help.measurementAccuracyBody)
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                }
                .padding(.vertical, 8)
            }

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
