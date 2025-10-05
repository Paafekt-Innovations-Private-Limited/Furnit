import SwiftUI
import Foundation

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
            .onAppear {
                // Refresh model list to pick up any new dollhouse files
                modelManager.refreshModels()
            }
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
                // Use simplified multi-method scanner
                DollhouseRoomScannerView(
                        isPresented: $showingRoomScanner,
                        modelManager: modelManager
                    )
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
            Image(systemName: model.isDollhouse ? "house.fill" : "cube.fill")
                .foregroundColor(model.isDollhouse ? .green : .blue)
                .font(.title2)
                .frame(width: 40, height: 40)
                .background((model.isDollhouse ? Color.green : Color.blue).opacity(0.1))
                .cornerRadius(8)
            
            VStack(alignment: .leading, spacing: 4) {
                Text(model.displayName)
                    .font(.headline)
                    .foregroundColor(.primary)
                
                Text(model.isDollhouse ? "2.5D Dollhouse Room" : "3D Room Model")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            
            Spacer()
            
            if model.isDollhouse {
                Text("2.5D")
                    .font(.caption2)
                    .fontWeight(.semibold)
                    .foregroundColor(.green)
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(Color.green.opacity(0.1))
                    .cornerRadius(4)
            }
            
            Image(systemName: "chevron.right")
                .foregroundColor(.secondary)
                .font(.caption)
        }
        .padding(.vertical, 4)
    }
}

// MARK: - Room Capture Camera View
struct RoomCaptureCamera: View {
    @Binding var capturedImage: UIImage?
    @Binding var isShowingCamera: Bool
    var onImageCaptured: ((UIImage) -> Void)?
    var photoNumber: Int = 1
    var totalPhotos: Int = 4
    @State private var image: UIImage?
    @State private var showImagePicker = true
    @State private var sourceType: UIImagePickerController.SourceType = .camera
    
    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            
            if let image = image {
                // Preview the captured image
                VStack {
                    HStack {
                        Text("Photo \(photoNumber) of \(totalPhotos)")
                            .font(.title2)
                            .fontWeight(.bold)
                            .foregroundColor(.white)
                        
                        Spacer()
                        
                        // Progress dots
                        HStack(spacing: 8) {
                            ForEach(1...totalPhotos, id: \.self) { index in
                                Circle()
                                    .fill(index <= photoNumber ? Color.green : Color.white.opacity(0.3))
                                    .frame(width: 10, height: 10)
                            }
                        }
                    }
                    .padding(.horizontal)
                    .padding(.top, 50)
                    
                    Text("Room Preview")
                        .font(.headline)
                        .foregroundColor(.white.opacity(0.8))
                    
                    Image(uiImage: image)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .padding()
                    
                    Text(photoNumber < totalPhotos ?
                         "Move to a different corner for next photo" :
                         "All angles captured!")
                        .font(.caption)
                        .foregroundColor(.white.opacity(0.7))
                        .padding(.horizontal)
                    
                    HStack(spacing: 40) {
                        // Retake button
                        Button(action: {
                            self.image = nil
                            self.showImagePicker = true
                        }) {
                            VStack {
                                Image(systemName: "camera.rotate")
                                    .font(.title)
                                Text("Retake")
                                    .font(.caption)
                            }
                            .foregroundColor(.white)
                            .frame(width: 80, height: 80)
                            .background(Color.gray.opacity(0.3))
                            .clipShape(Circle())
                        }
                        
                        // Use/Next button
                        Button(action: {
                            if let onImageCaptured = onImageCaptured {
                                onImageCaptured(image)
                                // Reset for next photo
                                self.image = nil
                                if photoNumber < totalPhotos {
                                    self.showImagePicker = true
                                }
                            } else {
                                capturedImage = image
                                isShowingCamera = false
                            }
                        }) {
                            VStack {
                                Image(systemName: photoNumber < totalPhotos ?
                                     "arrow.right.circle.fill" : "checkmark.circle.fill")
                                    .font(.title)
                                Text(photoNumber < totalPhotos ? "Next" : "Finish")
                                    .font(.caption)
                            }
                            .foregroundColor(.white)
                            .frame(width: 80, height: 80)
                            .background(Color.green.opacity(0.7))
                            .clipShape(Circle())
                        }
                        
                        // Cancel button
                        Button(action: {
                            isShowingCamera = false
                        }) {
                            VStack {
                                Image(systemName: "xmark.circle")
                                    .font(.title)
                                Text("Cancel")
                                    .font(.caption)
                            }
                            .foregroundColor(.white)
                            .frame(width: 80, height: 80)
                            .background(Color.red.opacity(0.7))
                            .clipShape(Circle())
                        }
                    }
                    .padding(.bottom, 50)
                }
            } else {
                // Camera instructions
                VStack {
                    HStack {
                        Text("Photo \(photoNumber) of \(totalPhotos)")
                            .font(.title2)
                            .fontWeight(.bold)
                            .foregroundColor(.white)
                        
                        Spacer()
                        
                        HStack(spacing: 8) {
                            ForEach(1...totalPhotos, id: \.self) { index in
                                Circle()
                                    .fill(index < photoNumber ? Color.green : Color.white.opacity(0.3))
                                    .frame(width: 10, height: 10)
                            }
                        }
                    }
                    .padding(.horizontal)
                    .padding(.top, 50)
                    
                    Spacer()
                    
                    VStack(spacing: 20) {
                        Image(systemName: "viewfinder")
                            .font(.system(size: 150))
                            .foregroundColor(.white.opacity(0.5))
                        
                        Text(getInstructionForPhoto(photoNumber))
                            .font(.title3)
                            .foregroundColor(.white)
                            .multilineTextAlignment(.center)
                            .padding(.horizontal)
                    }
                    
                    Spacer()
                    
                    Text("Tip: Include as much of the room as possible")
                        .font(.caption)
                        .foregroundColor(.white.opacity(0.8))
                        .padding(.bottom, 100)
                }
            }
        }
        .sheet(isPresented: $showImagePicker, onDismiss: {
            if image == nil && photoNumber == 1 {
                // If no image was captured on first photo, close the camera view
                isShowingCamera = false
            }
        }) {
            ImagePicker(image: $image, sourceType: sourceType)
        }
    }
    
    private func getInstructionForPhoto(_ photoNum: Int) -> String {
        switch photoNum {
        case 1: return "Stand in the first corner\nCapture opposite wall"
        case 2: return "Move to the next corner\nCapture the adjacent wall"
        case 3: return "Move to the third corner\nCapture another view"
        case 4: return "Final corner\nCapture the last view"
        default: return "Capture the room"
        }
    }
}

// MARK: - Image Picker for Camera
struct ImagePicker: UIViewControllerRepresentable {
    @Binding var image: UIImage?
    var sourceType: UIImagePickerController.SourceType
    @Environment(\.presentationMode) var presentationMode
    
    func makeUIViewController(context: Context) -> UIImagePickerController {
        let picker = UIImagePickerController()
        picker.delegate = context.coordinator
        picker.sourceType = sourceType
        picker.allowsEditing = false
        
        // Camera settings for room scanning
        if sourceType == .camera {
            picker.cameraCaptureMode = .photo
            picker.cameraDevice = .rear
            picker.showsCameraControls = true
            
            // Add overlay view with grid guides
            let overlayView = createCameraOverlay(frame: picker.view.bounds)
            picker.cameraOverlayView = overlayView
        }
        
        return picker
    }
    
    func updateUIViewController(_ uiViewController: UIImagePickerController, context: Context) {}
    
    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }
    
    private func createCameraOverlay(frame: CGRect) -> UIView {
        let overlay = UIView(frame: frame)
        overlay.isUserInteractionEnabled = false
        
        // Add grid lines for composition help
        let gridLayer = CALayer()
        gridLayer.frame = frame
        
        // Vertical lines
        for i in 1...2 {
            let line = CALayer()
            line.frame = CGRect(x: frame.width * CGFloat(i) / 3, y: 0, width: 1, height: frame.height)
            line.backgroundColor = UIColor.white.withAlphaComponent(0.3).cgColor
            gridLayer.addSublayer(line)
        }
        
        // Horizontal lines
        for i in 1...2 {
            let line = CALayer()
            line.frame = CGRect(x: 0, y: frame.height * CGFloat(i) / 3, width: frame.width, height: 1)
            line.backgroundColor = UIColor.white.withAlphaComponent(0.3).cgColor
            gridLayer.addSublayer(line)
        }
        
        overlay.layer.addSublayer(gridLayer)
        
        // Add instruction label
        let label = UILabel(frame: CGRect(x: 20, y: 100, width: frame.width - 40, height: 40))
        label.text = "Frame the entire room"
        label.textAlignment = .center
        label.textColor = .white
        label.backgroundColor = UIColor.black.withAlphaComponent(0.5)
        label.layer.cornerRadius = 10
        label.clipsToBounds = true
        label.font = .systemFont(ofSize: 16, weight: .medium)
        overlay.addSubview(label)
        
        return overlay
    }
    
    class Coordinator: NSObject, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
        var parent: ImagePicker
        
        init(_ parent: ImagePicker) {
            self.parent = parent
        }
        
        func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
            if let uiImage = info[.originalImage] as? UIImage {
                parent.image = uiImage
            }
            parent.presentationMode.wrappedValue.dismiss()
        }
        
        func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
            parent.presentationMode.wrappedValue.dismiss()
        }
    }
}

// MARK: - Room Scanner View (Actual Implementation)
struct RoomScannerView: View {
    @Binding var isPresented: Bool
    @ObservedObject var modelManager: USDZModelManager  // Passed from HomeTab
    @State private var capturedImages: [UIImage] = []
    @State private var showingCamera = false
    @State private var isProcessing = false
    @State private var statusMessage = "Ready to scan room"
    @State private var processingComplete = false
    @State private var currentImageCount = 0
    @StateObject private var ar3DModelProcessor = AR3DModelProcessor()
    
    let requiredPhotos = 4 // Need 4 photos from different angles
    
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
                    // Multi-photo capture camera
                    RoomCaptureCamera(
                        capturedImage: .constant(nil),
                        isShowingCamera: $showingCamera,
                        onImageCaptured: { image in
                            capturedImages.append(image)
                            currentImageCount = capturedImages.count
                            
                            if capturedImages.count < requiredPhotos {
                                // Need more photos
                                statusMessage = "Photo \(capturedImages.count) of \(requiredPhotos) captured"
                                showingCamera = true // Keep camera open for next photo
                            } else {
                                // Have enough photos, start processing
                                showingCamera = false
                                processScannedImages(capturedImages)
                            }
                        },
                        photoNumber: currentImageCount + 1,
                        totalPhotos: requiredPhotos
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
                        
                        Text("\(capturedImages.count) photos captured")
                            .font(.caption)
                            .foregroundColor(.white.opacity(0.7))
                    }
                } else if processingComplete {
                    // Success view
                    VStack(spacing: 30) {
                        Image(systemName: "checkmark.circle.fill")
                            .font(.system(size: 80))
                            .foregroundColor(.green)
                        
                        Text("3D Room Model Created!")
                            .font(.title)
                            .fontWeight(.bold)
                            .foregroundColor(.white)
                        
                        Text("Your room has been scanned from \(capturedImages.count) angles and saved")
                            .font(.body)
                            .foregroundColor(.white.opacity(0.8))
                            .multilineTextAlignment(.center)
                            .padding(.horizontal)
                        
                        HStack(spacing: 20) {
                            Button(action: {
                                // Reset for another scan
                                resetScanner()
                            }) {
                                Text("Scan Another")
                                    .fontWeight(.semibold)
                                    .foregroundColor(.white)
                                    .frame(width: 150, height: 50)
                                    .background(Color.blue)
                                    .cornerRadius(25)
                            }
                            
                            Button(action: {
                                isPresented = false
                            }) {
                                Text("Done")
                                    .fontWeight(.semibold)
                                    .foregroundColor(.white)
                                    .frame(width: 150, height: 50)
                                    .background(Color.green)
                                    .cornerRadius(25)
                            }
                        }
                    }
                    .padding()
                } else {
                    // Main scanner interface (instructions)
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
                        
                        Text("Take \(requiredPhotos) photos from different angles")
                            .font(.title3)
                            .foregroundColor(.white.opacity(0.8))
                        
                        VStack(spacing: 16) {
                            Text("How to scan:")
                                .font(.headline)
                                .foregroundColor(.white)
                            
                            VStack(alignment: .leading, spacing: 12) {
                                HStack {
                                    Image(systemName: "1.circle.fill")
                                        .foregroundColor(.blue)
                                    Text("Stand in room corner")
                                        .foregroundColor(.white.opacity(0.9))
                                }
                                
                                HStack {
                                    Image(systemName: "2.circle.fill")
                                        .foregroundColor(.blue)
                                    Text("Take photo of opposite corner")
                                        .foregroundColor(.white.opacity(0.9))
                                }
                                
                                HStack {
                                    Image(systemName: "3.circle.fill")
                                        .foregroundColor(.blue)
                                    Text("Repeat from all 4 corners")
                                        .foregroundColor(.white.opacity(0.9))
                                }
                                
                                HStack {
                                    Image(systemName: "4.circle.fill")
                                        .foregroundColor(.blue)
                                    Text("AI creates complete 3D model")
                                        .foregroundColor(.white.opacity(0.9))
                                }
                            }
                            .font(.subheadline)
                        }
                        .padding(.horizontal, 40)
                        
                        // Progress indicator
                        if currentImageCount > 0 {
                            HStack {
                                ForEach(0..<requiredPhotos, id: \.self) { index in
                                    Circle()
                                        .fill(index < currentImageCount ? Color.green : Color.white.opacity(0.3))
                                        .frame(width: 12, height: 12)
                                }
                            }
                            Text("\(currentImageCount) of \(requiredPhotos) photos captured")
                                .font(.caption)
                                .foregroundColor(.white.opacity(0.7))
                        }
                        
                        Spacer()
                        
                        // Start Scanning button
                        Button(action: {
                            startScanning()
                        }) {
                            HStack {
                                Image(systemName: "camera.fill")
                                Text(currentImageCount > 0 ? "Continue Scanning" : "Start Scanning")
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
    }
    
    private func startScanning() {
        print("🚀 Starting room scan - Photo \(currentImageCount + 1) of \(requiredPhotos)")
        showingCamera = true
        statusMessage = "Capture photo \(currentImageCount + 1) of \(requiredPhotos)"
    }
    
    private func processScannedImages(_ images: [UIImage]) {
        Task {
            await MainActor.run {
                isProcessing = true
                processingComplete = false
                statusMessage = "Processing \(images.count) room photos..."
            }
            
            // Simulate processing delay
            try? await Task.sleep(nanoseconds: 3_000_000_000) // 3 seconds
            
            print("📸 Processing \(images.count) room images")
            
            await MainActor.run {
                statusMessage = "Generating 3D room model from \(images.count) views..."
            }
            
            // Simulate more processing
            try? await Task.sleep(nanoseconds: 2_000_000_000) // 2 seconds
            
            // Create a new room model
            // For now, use the first existing model's file as a placeholder
            // In a real app, you'd generate actual 3D model data from the photos
            let roomName = "Scanned Room \(Date().formatted(date: .abbreviated, time: .shortened))"
            
            // Check if there are existing models to use as template
            var fileName = "scanned_room_\(Date().timeIntervalSince1970).usdz"
            if !modelManager.models.isEmpty {
                // Use existing model file as placeholder (temporary solution)
                fileName = modelManager.models[0].fileName
            }
            
            let newRoomModel = USDZModel(
                name: roomName,
                fileName: fileName
            )
            
            // Save the new room to the model manager
            await MainActor.run {
                // Add to the shared model list (now it will appear in HomeTab)
                modelManager.models.append(newRoomModel)
                
                statusMessage = "3D room model created successfully!"
                isProcessing = false
                processingComplete = true
                print("✅ 3D room '\(roomName)' saved to collection with \(images.count) photos")
                print("📋 Total models in list: \(modelManager.models.count)")
            }
        }
    }
    
    private func resetScanner() {
        capturedImages = []
        currentImageCount = 0
        processingComplete = false
        statusMessage = "Ready to scan room"
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
