import SwiftUI
import CoreML
import Photos
import WebKit

// MARK: - Room Boundary Manager

/// Boundary manager for WebGL Gaussian splat rendering
/// Uses actual room bounds to calculate optimal camera positions
struct RoomBoundaryManager {
    let bounds: RoomBounds

    /// Room dimensions
    var width: Float { bounds.width }
    var height: Float { bounds.height }
    var depth: Float { bounds.depth }

    /// Room center
    var centerX: Float { bounds.centerX }
    var centerY: Float { bounds.centerY }
    var centerZ: Float { bounds.centerZ }

    /// Wall positions (maxZ = front wall in classic PLY, minZ = back wall)
    /// In classic PLY from SHARP, Z is negative, and the wall closest to camera
    /// is the one with the *largest* Z (least negative).
    var frontWallZ: Float { bounds.maxZ }  // closest to camera
    var backWallZ: Float { bounds.minZ }   // farthest from camera

    init(bounds: RoomBounds) {
        self.bounds = bounds
    }

    /// Default bounds when none provided
    static var defaultBounds: RoomBounds {
        RoomBounds(minX: -2, maxX: 2, minY: -1.5, maxY: 1.5, minZ: -5, maxZ: -1)
    }

    /// Calculate camera position just INSIDE the room, near the front wall,
    /// looking towards the room center.
    ///
    /// insideFactor:
    ///   - 0.0  => exactly at front wall plane
    ///   - 0.1  => 10% of room depth inside from the front wall
    ///   - 0.5  => at room center
    func getCameraAtBackWall(fovDegrees: Float = 60) -> (eye: SIMD3<Float>, target: SIMD3<Float>) {
        // Delegate to SharpRoomCameraUtils for camera position calculation
        let result = SharpRoomCameraUtils.calculateCameraPosition(
            frontWallZ: frontWallZ,
            backWallZ: backWallZ,
            centerX: centerX,
            centerY: centerY,
            centerZ: centerZ,
            insideFactor: 0.15  // 15% into the room from the front wall
        )

        logDebug("📷 [BoundaryManager] frontWallZ=\(frontWallZ), backWallZ=\(backWallZ), depth=\(depth)")
        logDebug("📷 [BoundaryManager] INSIDE room: eye=(\(result.eye.x), \(result.eye.y), \(result.eye.z)) target=(\(result.target.x), \(result.target.y), \(result.target.z))")

        return result
    }


}

// MARK: - Orbit Gesture View (like antimatter15/splat)

/// Gesture overlay that captures drag/pinch and sends to WebGL for orbit control
struct OrbitGestureView: View {
    @State private var lastDragLocation: CGPoint? = nil
    @State private var lastScale: CGFloat = 1.0

    var body: some View {
        Color.clear
            .contentShape(Rectangle())
            .gesture(
                DragGesture(minimumDistance: 1)
                    .onChanged { value in
                        let currentLocation = value.location

                        // First movement in this drag → tell JS "user started interacting"
                        if lastDragLocation == nil {
                            NotificationCenter.default.post(
                                name: NSNotification.Name("WebGLOrbitGestureState"),
                                object: nil,
                                userInfo: ["interacting": true]
                            )
                        }

                        if let lastLocation = lastDragLocation {
                            // Calculate delta from last position (not from start)
                            let deltaX = currentLocation.x - lastLocation.x
                            let deltaY = currentLocation.y - lastLocation.y

                            // Send incremental delta to JS
                            NotificationCenter.default.post(
                                name: NSNotification.Name("WebGLOrbitGesture"),
                                object: nil,
                                userInfo: ["deltaX": deltaX, "deltaY": deltaY, "incremental": true]
                            )
                        }

                        lastDragLocation = currentLocation
                    }
                    .onEnded { _ in
                        // Reset for next gesture
                        lastDragLocation = nil

                        // Tell JS "user stopped interacting"
                        NotificationCenter.default.post(
                            name: NSNotification.Name("WebGLOrbitGestureState"),
                            object: nil,
                            userInfo: ["interacting": false]
                        )
                    }
            )
            .simultaneousGesture(
                MagnificationGesture()
                    .onChanged { scale in
                        // First pinch → tell JS "user started interacting"
                        if lastScale == 1.0 {
                            NotificationCenter.default.post(
                                name: NSNotification.Name("WebGLOrbitGestureState"),
                                object: nil,
                                userInfo: ["interacting": true]
                            )
                        }

                        let incrementalScale = scale / lastScale
                        lastScale = scale

                        NotificationCenter.default.post(
                            name: NSNotification.Name("WebGLZoomGesture"),
                            object: nil,
                            userInfo: ["scale": incrementalScale, "incremental": true]
                        )
                    }
                    .onEnded { _ in
                        lastScale = 1.0

                        // Tell JS "user stopped interacting"
                        NotificationCenter.default.post(
                            name: NSNotification.Name("WebGLOrbitGestureState"),
                            object: nil,
                            userInfo: ["interacting": false]
                        )
                    }
            )
    }
}

/// Simple 3D Gaussian Splat viewer using WebGL (antimatter15/splat)
/// Renders PLY files generated by SHARP
struct SharpRoomView: View {
    let plyURL: URL
    let roomMeasurements: RoomMeasurements?  // Room measurements for display
    let allowSave: Bool  // Show save button (true for new rooms, false for viewing from home)
    let photoOrientation: PhotoOrientation  // Source photo orientation (for UI layout)
    @Environment(\.dismiss) private var dismiss

    // Parsed bounds from PLY file (used when roomMeasurements is nil)
    private let parsedBounds: RoomBounds?

    // Convenience initializer for backwards compatibility
    init(plyURL: URL, roomMeasurements: RoomMeasurements? = nil, allowSave: Bool = true, photoOrientation: PhotoOrientation = .portrait) {
        self.plyURL = plyURL
        self.roomMeasurements = roomMeasurements
        self.allowSave = allowSave
        self.photoOrientation = photoOrientation

        // Compute viewer PLY URL (prefer 3DGS for SparkJS, then classic, then base)
        let basePath = plyURL.path
        let threeDGSPath = basePath.replacingOccurrences(of: ".ply", with: "_3dgs.ply")
        let classicPath = basePath.replacingOccurrences(of: ".ply", with: "_classic.ply")

        let viewerURL: URL
        if FileManager.default.fileExists(atPath: threeDGSPath) {
            viewerURL = URL(fileURLWithPath: threeDGSPath)
        } else if FileManager.default.fileExists(atPath: classicPath) {
            viewerURL = URL(fileURLWithPath: classicPath)
        } else {
            viewerURL = plyURL
        }

        // Always parse bounds from the PLY we're actually displaying
        self.parsedBounds = Self.parsePLYBounds(from: viewerURL)
    }

    /// Parse PLY file to extract position bounds (for HomeView where roomMeasurements isn't available)
    private static func parsePLYBounds(from url: URL) -> RoomBounds? {
        guard let data = try? Data(contentsOf: url) else {
            logDebug("PLY Parser: Failed to read file")
            return nil
        }

        // Find end_header
        guard let headerEndRange = data.range(of: Data("end_header\n".utf8)) else {
            logDebug("PLY Parser: No end_header found")
            return nil
        }

        // Parse header for vertex count
        guard let headerString = String(data: data[..<headerEndRange.lowerBound], encoding: .utf8) else {
            logDebug("PLY Parser: Failed to parse header")
            return nil
        }

        var vertexCount = 0
        for line in headerString.components(separatedBy: "\n") {
            if line.hasPrefix("element vertex ") {
                let parts = line.components(separatedBy: " ")
                if parts.count >= 3, let count = Int(parts[2]) {
                    vertexCount = count
                }
            }
        }

        guard vertexCount > 0 else {
            logDebug("PLY Parser: No vertices found")
            return nil
        }

        logDebug("PLY Parser: Found \(vertexCount) vertices")

        // Binary data starts after header
        let binaryStart = headerEndRange.upperBound

        // Each vertex: x,y,z (3 floats) + other properties
        // Stride: 3 floats (xyz) + 3 floats (scale) + 4 floats (rot) + 1 float (opacity) + 3 bytes (rgb) = 47 bytes
        let bytesPerVertex = 47
        let binaryDataLength = data.count - binaryStart

        var minX: Float = .greatestFiniteMagnitude
        var maxX: Float = -.greatestFiniteMagnitude
        var minY: Float = .greatestFiniteMagnitude
        var maxY: Float = -.greatestFiniteMagnitude
        var minZ: Float = .greatestFiniteMagnitude
        var maxZ: Float = -.greatestFiniteMagnitude

        let maxVertices = min(vertexCount, binaryDataLength / bytesPerVertex)

        // Use safe byte copying to avoid alignment issues
        for i in 0..<maxVertices {
            let byteOffset = binaryStart + i * bytesPerVertex
            guard byteOffset + 12 <= data.count else { continue }

            var x: Float = 0
            var y: Float = 0
            var z: Float = 0

            _ = withUnsafeMutableBytes(of: &x) { dest in
                data.copyBytes(to: dest, from: byteOffset..<(byteOffset + 4))
            }
            _ = withUnsafeMutableBytes(of: &y) { dest in
                data.copyBytes(to: dest, from: (byteOffset + 4)..<(byteOffset + 8))
            }
            _ = withUnsafeMutableBytes(of: &z) { dest in
                data.copyBytes(to: dest, from: (byteOffset + 8)..<(byteOffset + 12))
            }

            minX = min(minX, x)
            maxX = max(maxX, x)
            minY = min(minY, y)
            maxY = max(maxY, y)
            minZ = min(minZ, z)
            maxZ = max(maxZ, z)
        }

        guard minX < maxX else {
            logDebug("PLY Parser: Invalid bounds")
            return nil
        }

        logDebug("PLY Parser: Bounds X[\(minX), \(maxX)] Y[\(minY), \(maxY)] Z[\(minZ), \(maxZ)]")

        return RoomBounds(
            minX: minX, maxX: maxX,
            minY: minY, maxY: maxY,
            minZ: minZ, maxZ: maxZ
        )
    }

    /// Get effective bounds - always use parsed bounds from the displayed PLY file
    /// (roomMeasurements.actualBounds uses different coordinate system)
    private var effectiveBounds: RoomBounds? {
        parsedBounds
    }

    @State private var isLoading = true
    @State private var error: String?
    @State private var showingFurnitureFit = false
    @State private var mlModel: MLModel? = nil

    // JS-measured front wall dimensions (from actual splat bounds)
    @State private var jsFrontWallWidth: Float?
    @State private var jsFrontWallHeight: Float?

    // Save room state
    @StateObject private var modelManager = USDZModelManager()
    @State private var isSavingRoom = false
    @State private var saveProgress: Double = 0.0
    @State private var savingTimer: Timer?
    @State private var showSaveAlert = false
    @State private var saveAlertMessage = ""
    @State private var saveWasSuccessful = false
    @State private var isDismissing = false
    @State private var showRoomNameInput = false
    @State private var roomName = ""
    @State private var showShareSheet = false
    @EnvironmentObject var authManager: AuthenticationManager

    /// Compute classic PLY URL (pre-rotated for antimatter15/splat)
    private var classicPlyURL: URL {
        let path = plyURL.path
        let classicPath = path.replacingOccurrences(of: ".ply", with: "_classic.ply")
        let classicURL = URL(fileURLWithPath: classicPath)
        if FileManager.default.fileExists(atPath: classicPath) {
            return classicURL
        }
        return plyURL
    }

    /// PLY file for WebGL viewer: prefer 3DGS (SparkJS compatible), then classic, then base
    private var viewerPlyURL: URL {
        let basePath = plyURL.path

        // Prefer 3DGS variant for Spark/SuperSplat-style viewers
        let threeDGSPath = basePath.replacingOccurrences(of: ".ply", with: "_3dgs.ply")
        if FileManager.default.fileExists(atPath: threeDGSPath) {
            return URL(fileURLWithPath: threeDGSPath)
        }

        // Fallback: classic antimatter-oriented PLY
        let classicPath = basePath.replacingOccurrences(of: ".ply", with: "_classic.ply")
        if FileManager.default.fileExists(atPath: classicPath) {
            return URL(fileURLWithPath: classicPath)
        }

        // Last resort: original .ply
        return plyURL
    }

    var body: some View {
        ZStack {
            // WebGL view using SparkJS (use classic PLY - pre-rotated for antimatter viewer)
            AntimatterSplatView(
                plyURL: classicPlyURL,
                roomBounds: roomMeasurements?.boundingBox,
                actualBounds: effectiveBounds,
                photoOrientation: photoOrientation,
                onLoaded: {
                    isLoading = false
                },
                onFrontWallDimensions: { w, h in
                    // Store JS-measured dimensions for nav title
                    jsFrontWallWidth = Float(w)
                    jsFrontWallHeight = Float(h)
                }
            )
            .ignoresSafeArea()
            .allowsHitTesting(false)  // Route all touches to SwiftUI overlays, not WKWebView

            // Gesture overlay for orbit control (captures drags and sends to WebGL)
            OrbitGestureView()
                .ignoresSafeArea()  // Cover full screen including safe areas
                .allowsHitTesting(!showingFurnitureFit && !isLoading)
                .zIndex(10)  // Above WebGL view, below other overlays

            // Loading overlay
            if isLoading {
                VStack(spacing: 16) {
                    ProgressView()
                        .scaleEffect(1.5)
                        .tint(.white)

                    Text(NSLocalizedString("photoRoom.loading", comment: ""))
                        .font(.subheadline)
                        .foregroundColor(.white)
                }
                .padding(24)
                .background(Color.black.opacity(0.7))
                .cornerRadius(12)
            }

            // Error overlay
            if let error = error {
                VStack(spacing: 16) {
                    Image(systemName: "exclamationmark.triangle")
                        .font(.largeTitle)
                        .foregroundColor(.orange)
                    Text(error)
                        .foregroundColor(.white)
                        .multilineTextAlignment(.center)
                }
                .padding(24)
                .background(Color.black.opacity(0.7))
                .cornerRadius(12)
            }

            // FurnitureFit overlay (when active) - full screen
            if showingFurnitureFit {
                FurnitureFitUIView(
                    capturedImage: .constant(nil),
                    roomImage: nil,
                    mlModel: mlModel,
                    processInterval: 0.07,
                    active: true
                )
                .ignoresSafeArea()
                .zIndex(100)
            }


            // Save progress overlay
            if isSavingRoom {
                saveRoomProgressOverlay
            }

            // Landscape layout: controls on LEFT edge (appears at bottom when phone held horizontally)
            // Rotated 90° CW so icons appear upright when phone is horizontal
            if photoOrientation == .landscape {
                // Far left edge controls for landscape - rotated 90° clockwise
                HStack(spacing: 0) {
                    VStack(spacing: 0) {
                        // Brain button (top = left when horizontal)
                        Button(action: {
                            if showingFurnitureFit {
                                showingFurnitureFit = false
                            } else {
                                // Release SHARP model to free memory for YOLO segmentation
                                SHARPService.shared.releaseResources()
                                showingFurnitureFit = true
                            }
                        }) {
                            Image(systemName: "brain.head.profile")
                                .font(.system(size: 28))
                                .foregroundColor(.white)
                                .frame(width: 60, height: 60)
                                .background(Circle().fill(showingFurnitureFit ? Color.green : Color.blue).shadow(radius: 5))
                                .rotationEffect(.degrees(90))  // Rotate icon for horizontal viewing
                        }
                        .disabled(isLoading)
                        .padding(.top, 80)

                        Spacer()
                            .allowsHitTesting(false)

                        // Orientation label (center)
                        VStack(spacing: 1) {
                            Text(NSLocalizedString("orientation.heldHorizontally", comment: ""))
                                .font(.caption2)
                            Text(NSLocalizedString("orientation.landscape", comment: ""))
                                .font(.caption2)
                                .fontWeight(.medium)
                        }
                        .foregroundColor(.white.opacity(0.8))
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(Color.black.opacity(0.4))
                        .cornerRadius(6)
                        .rotationEffect(.degrees(90))

                        Spacer()
                            .allowsHitTesting(false)

                        // Screenshot button (bottom = right when horizontal)
                        Button(action: {
                            takeScreenshot()
                        }) {
                            Image(systemName: "camera.fill")
                                .font(.system(size: 28))
                                .foregroundColor(.white)
                                .frame(width: 60, height: 60)
                                .background(Circle().fill(Color.blue).shadow(radius: 5))
                                .rotationEffect(.degrees(90))  // Rotate icon for horizontal viewing
                        }
                        .disabled(isLoading)
                        .padding(.bottom, 30)
                    }
                    .frame(width: 130)  // Fixed width column at left edge
                    .padding(.leading, 0)  // No padding - stick to left edge
                    Spacer()
                        .allowsHitTesting(false)
                }
                .zIndex(99997)
            } else {
                // Portrait layout: standard bottom controls

                // Touch-anywhere drag overlay for camera control
                SimpleJoystickOverlay(photoOrientation: photoOrientation)
                    .zIndex(99997)

                // Buttons at bottom row
                VStack {
                    Spacer()
                        .allowsHitTesting(false)
                    HStack {
                        // Brain button (bottom-left)
                        Button(action: {
                            if showingFurnitureFit {
                                showingFurnitureFit = false
                            } else {
                                // Release SHARP model to free memory for YOLO segmentation
                                SHARPService.shared.releaseResources()
                                showingFurnitureFit = true
                            }
                        }) {
                            Image(systemName: "brain.head.profile")
                                .font(.system(size: 28))
                                .foregroundColor(.white)
                                .frame(width: 60, height: 60)
                                .background(Circle().fill(showingFurnitureFit ? Color.green : Color.blue).shadow(radius: 5))
                        }
                        .disabled(isLoading)
                        .padding(.leading, 16)

                        Spacer()
                            .allowsHitTesting(false)

                        // Screenshot button (bottom-right)
                        Button(action: {
                            takeScreenshot()
                        }) {
                            Image(systemName: "camera.fill")
                                .font(.system(size: 28))
                                .foregroundColor(.white)
                                .frame(width: 60, height: 60)
                                .background(Circle().fill(Color.blue).shadow(radius: 5))
                        }
                        .disabled(isLoading)
                        .padding(.trailing, 16)
                    }
                    .padding(.bottom, 20)
                }
                .zIndex(99998)  // Higher than joystick (99997) to allow button taps
            }
        }
        .background(Color.gray)
        .navigationTitle(navigationTitleWithDimensions)
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItemGroup(placement: .navigationBarTrailing) {
                // Share button - only for authorized users
                if authManager.canShare {
                    Button(action: {
                        showShareSheet = true
                    }) {
                        Image(systemName: "square.and.arrow.up")
                    }
                    .disabled(isLoading)
                }

                // Save button (only for new rooms, not when viewing from home)
                if allowSave {
                    Button(action: {
                        showRoomNameInput = true
                    }) {
                        Image(systemName: "square.and.arrow.down")
                    }
                    .disabled(isLoading)
                }

                // Recenter button
                Button(action: {
                    NotificationCenter.default.post(name: NSNotification.Name("RecenterWebGLCamera"), object: nil)
                }) {
                    Image(systemName: "viewfinder")
                }
                .disabled(isLoading)
            }
        }
        .sheet(isPresented: $showShareSheet) {
            // Share the 3DGS format (SuperSplat compatible)
            let threeDGSPath = plyURL.path.replacingOccurrences(of: ".ply", with: "_3dgs.ply")
            let shareURL = FileManager.default.fileExists(atPath: threeDGSPath)
                ? URL(fileURLWithPath: threeDGSPath)
                : plyURL
            ShareSheet(activityItems: [shareURL])
        }
        .onAppear {
            loadMLModel()
            // Lock to portrait orientation for 3D view stability
            OrientationLockManager.shared.lockToPortrait()
            logDebug("📐 [SharpRoomView] photoOrientation = \(photoOrientation)")
        }
        .onDisappear {
            // Unlock orientation when leaving the view
            OrientationLockManager.shared.unlock()
        }
        // Room name input alert
        .alert("Save Room", isPresented: $showRoomNameInput) {
            TextField("Room name", text: $roomName)
            Button("Cancel", role: .cancel) {
                roomName = ""
            }
            Button("Save") {
                startSavingRoom()
            }
            .disabled(roomName.isEmpty)
        } message: {
            Text(NSLocalizedString("roomViewer.enterName", comment: ""))
        }
        // Save result alert
        .alert("Room Save", isPresented: $showSaveAlert) {
            Button("OK", role: .cancel) {
                if saveWasSuccessful {
                    // Show dismissing indicator
                    isDismissing = true
                    // Release resources in background, then dismiss
                    DispatchQueue.global(qos: .userInitiated).async {
                        // Release SHARP model memory
                        DispatchQueue.main.async {
                            SHARPService.shared.releaseResources()
                        }
                        // Small delay to let cleanup complete
                        Thread.sleep(forTimeInterval: 0.1)
                        DispatchQueue.main.async {
                            // Dismiss entire sheet and go back to home view
                            NotificationCenter.default.post(name: NSNotification.Name("DismissPhotoRoomSheet"), object: nil)
                        }
                    }
                }
            }
        } message: {
            Text(saveAlertMessage)
        }
        // Dismissing overlay
        .overlay {
            if isDismissing {
                ZStack {
                    Color.black.opacity(0.7)
                        .ignoresSafeArea()
                    VStack(spacing: 16) {
                        ProgressView()
                            .scaleEffect(1.5)
                            .tint(.white)
                        Text("Going back...")
                            .foregroundColor(.white)
                            .font(.headline)
                    }
                }
            }
        }
    }

    // MARK: - Screenshot

    private func takeScreenshot() {
        logDebug("📸 Taking screenshot...")

        // Capture the window
        let scenes = UIApplication.shared.connectedScenes.compactMap { $0 as? UIWindowScene }
        let windows = scenes.flatMap { $0.windows }
        guard let window = windows.first(where: { $0.isKeyWindow }) ?? windows.first else {
            logDebug("❌ No window found")
            return
        }

        let format = UIGraphicsImageRendererFormat()
        format.scale = UIScreen.main.scale
        let renderer = UIGraphicsImageRenderer(bounds: window.bounds, format: format)
        let image = renderer.image { _ in
            window.drawHierarchy(in: window.bounds, afterScreenUpdates: true)
        }

        logDebug("📸 Screenshot captured, saving to Photos...")

        // Save to photos
        UIImageWriteToSavedPhotosAlbum(image, nil, nil, nil)
        logDebug("✅ Screenshot saved to Photos")
    }

    // MARK: - Load ML Model for FurnitureFit

    private func loadMLModel() {
        guard mlModel == nil else { return }

        let candidateNames = [
            ("yoloe-11l-seg-pf", "mlmodelc"),
            ("yoloe-11l-seg-pf", "mlpackage"),
        ]

        for (name, ext) in candidateNames {
            if let url = Bundle.main.url(forResource: name, withExtension: ext) {
                do {
                    let config = MLModelConfiguration()
                    config.computeUnits = .cpuOnly
                    let model = try MLModel(contentsOf: url, configuration: config)
                    self.mlModel = model
                    logDebug("✅ [SharpRoomView] Loaded MLModel '\(name).\(ext)'")
                    break
                } catch {
                    logDebug("❌ [SharpRoomView] Failed to load \(name).\(ext): \(error)")
                }
            }
        }
    }

    // MARK: - Navigation Title with Dimensions
    private var navigationTitleWithDimensions: String {
        // Show front wall dimensions (width × height)
        let wallWidth = roomMeasurements?.frontWallWidth
            ?? jsFrontWallWidth
            ?? 4.0
        let wallHeight = roomMeasurements?.frontWallHeight
            ?? jsFrontWallHeight
            ?? 3.0

        return String(format: "%.1f × %.1f m", wallWidth, wallHeight)
    }

    // MARK: - Save Room Progress Overlay
    private var saveRoomProgressOverlay: some View {
        ZStack {
            Color.black.opacity(0.9)
                .ignoresSafeArea()

            VStack(spacing: 24) {
                ZStack {
                    Circle()
                        .fill(Color.green.opacity(0.2))
                        .frame(width: 100, height: 100)

                    Image(systemName: "square.and.arrow.down")
                        .font(.system(size: 40))
                        .foregroundColor(.green)
                }

                Text("Saving Room...")
                    .font(.title2)
                    .fontWeight(.semibold)
                    .foregroundColor(.white)

                ProgressView(value: saveProgress)
                    .progressViewStyle(LinearProgressViewStyle(tint: .green))
                    .frame(width: 200)

                Text("\(Int(saveProgress * 100))%")
                    .font(.headline)
                    .foregroundColor(.gray)

                Button("Cancel") {
                    cancelSavingRoom()
                }
                .foregroundColor(.red)
                .padding(.top, 20)
            }
        }
    }

    // MARK: - Save Room Functions
    private func startSavingRoom() {
        guard !roomName.isEmpty else { return }

        let savedName = roomName
        logDebug("💾 [SharpRoomView] Starting room save: \(savedName)")

        withAnimation(.easeIn(duration: 0.3)) {
            isSavingRoom = true
            saveProgress = 0.0
        }

        var saveStarted = false
        var saveCompleted = false
        var saveSuccess = false
        var saveError: String?

        savingTimer = Timer.scheduledTimer(withTimeInterval: 0.05, repeats: true) { timer in
            if !saveStarted || (saveStarted && saveCompleted) {
                self.saveProgress += 0.015
            }

            if self.saveProgress >= 0.6 && !saveStarted {
                saveStarted = true
                // Save the classic PLY file (pre-transformed for correct viewing)
                self.modelManager.savePLY(from: self.classicPlyURL, name: savedName, photoOrientation: self.photoOrientation) { success, error in
                    DispatchQueue.main.async {
                        saveCompleted = true
                        saveSuccess = success
                        saveError = error
                        logDebug(success ? "✅ [SharpRoomView] Room saved" : "❌ [SharpRoomView] Save failed: \(error ?? "unknown")")
                    }
                }
            }

            if self.saveProgress >= 1.0 && saveCompleted {
                timer.invalidate()
                self.savingTimer = nil

                withAnimation(.easeOut(duration: 0.3)) {
                    self.isSavingRoom = false
                }

                DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
                    if saveSuccess {
                        self.saveAlertMessage = "Room '\(savedName)' saved successfully!"
                        self.saveWasSuccessful = true
                    } else {
                        self.saveAlertMessage = "Failed to save: \(saveError ?? "Unknown error")"
                        self.saveWasSuccessful = false
                    }
                    self.showSaveAlert = true
                    self.roomName = ""
                }
            }
        }
    }

    private func cancelSavingRoom() {
        savingTimer?.invalidate()
        savingTimer = nil

        withAnimation(.easeOut(duration: 0.2)) {
            isSavingRoom = false
            saveProgress = 0.0
        }

        roomName = ""
        logDebug("❌ [SharpRoomView] Room save cancelled")
    }

}

// MARK: - Antimatter15 WebGL Splat View

/// Custom URL scheme handler to serve HTML and PLY files to WKWebView
class LocalFileSchemeHandler: NSObject, WKURLSchemeHandler {
    var plyURL: URL?
    var htmlContent: String = ""
    var loadStartTime: CFAbsoluteTime = 0

    func webView(_ webView: WKWebView, start urlSchemeTask: WKURLSchemeTask) {
        guard let requestURL = urlSchemeTask.request.url else {
            urlSchemeTask.didFailWithError(NSError(domain: "LocalFileSchemeHandler", code: -1, userInfo: nil))
            return
        }

        let path = requestURL.path
        logDebug("LocalFileSchemeHandler: Request for \(path)")

        // Serve the HTML page
        if path == "/" || path == "/index.html" || path.isEmpty {
            let data = htmlContent.data(using: .utf8) ?? Data()
            let response = HTTPURLResponse(
                url: requestURL,
                statusCode: 200,
                httpVersion: "HTTP/1.1",
                headerFields: [
                    "Content-Type": "text/html; charset=utf-8",
                    "Content-Length": "\(data.count)"
                ]
            )!
            urlSchemeTask.didReceive(response)
            urlSchemeTask.didReceive(data)
            urlSchemeTask.didFinish()
            logDebug("LocalFileSchemeHandler: Served HTML (\(data.count) bytes)")
            return
        }

        // Serve the PLY file
        if path == "/room.ply", let plyURL = plyURL {
            do {
                let readStart = CFAbsoluteTimeGetCurrent()
                let data = try Data(contentsOf: plyURL)
                let readTime = (CFAbsoluteTimeGetCurrent() - readStart) * 1000

                let response = HTTPURLResponse(
                    url: requestURL,
                    statusCode: 200,
                    httpVersion: "HTTP/1.1",
                    headerFields: [
                        "Content-Type": "application/octet-stream",
                        "Content-Length": "\(data.count)",
                        "Access-Control-Allow-Origin": "*"
                    ]
                )!
                urlSchemeTask.didReceive(response)
                urlSchemeTask.didReceive(data)
                urlSchemeTask.didFinish()

                let sizeMB = Double(data.count) / (1024 * 1024)
                logDebug("⏱️ [WebGL] PLY file read: \(String(format: "%.0f", readTime))ms (\(String(format: "%.1f", sizeMB)) MB)")
            } catch {
                logDebug("LocalFileSchemeHandler: Failed to read PLY: \(error)")
                urlSchemeTask.didFailWithError(error)
            }
            return
        }

        // Serve the local main.js (modified antimatter15/splat)
        if path == "/main.js" {
            // Try bundle resource
            if let jsURL = Bundle.main.url(forResource: "splat_main", withExtension: "js"),
               let data = try? Data(contentsOf: jsURL) {
                let response = HTTPURLResponse(
                    url: requestURL,
                    statusCode: 200,
                    httpVersion: "HTTP/1.1",
                    headerFields: [
                        "Content-Type": "application/javascript; charset=utf-8",
                        "Content-Length": "\(data.count)"
                    ]
                )!
                urlSchemeTask.didReceive(response)
                urlSchemeTask.didReceive(data)
                urlSchemeTask.didFinish()
                logDebug("LocalFileSchemeHandler: Served main.js from bundle (\(data.count) bytes)")
                return
            }

            logDebug("LocalFileSchemeHandler: main.js not found in bundle!")
        }

        // 404 for other paths
        let errorResponse = HTTPURLResponse(
            url: requestURL,
            statusCode: 404,
            httpVersion: "HTTP/1.1",
            headerFields: nil
        )!
        urlSchemeTask.didReceive(errorResponse)
        urlSchemeTask.didReceive(Data())
        urlSchemeTask.didFinish()
    }

    func webView(_ webView: WKWebView, stop urlSchemeTask: WKURLSchemeTask) {
        // Nothing to clean up
    }
}

/// WebGL-based 3D Gaussian Splat viewer using antimatter15/splat
/// Renders PLY files in a WKWebView using WebGL
struct AntimatterSplatView: UIViewRepresentable {
    let plyURL: URL
    let roomBounds: SIMD3<Float>?  // Room dimensions for camera positioning
    let actualBounds: RoomBounds?  // Actual min/max bounds for precise framing
    let photoOrientation: PhotoOrientation  // For orientation-based room rotation
    let onLoaded: () -> Void
    var onFrontWallDimensions: ((Double, Double) -> Void)? = nil

    func makeCoordinator() -> Coordinator {
        Coordinator(onLoaded: onLoaded, onFrontWallDimensions: onFrontWallDimensions)
    }

    func makeUIView(context: Context) -> WKWebView {
        let config = WKWebViewConfiguration()
        config.allowsInlineMediaPlayback = true

        // Enable JavaScript
        let prefs = WKWebpagePreferences()
        prefs.allowsContentJavaScript = true
        config.defaultWebpagePreferences = prefs

        // Register custom URL scheme handler
        let schemeHandler = LocalFileSchemeHandler()
        schemeHandler.plyURL = plyURL
        schemeHandler.htmlContent = generateSplatViewerHTML(bounds: roomBounds, actualBounds: actualBounds, orientation: self.photoOrientation)
        config.setURLSchemeHandler(schemeHandler, forURLScheme: "splat")

        // Add message handlers for communication from JS
        config.userContentController.add(context.coordinator, name: "splatLoaded")
        config.userContentController.add(context.coordinator, name: "frontWallDimensions")
        config.userContentController.add(context.coordinator, name: "cameraPose")

        let webView = WKWebView(frame: .zero, configuration: config)
        webView.navigationDelegate = context.coordinator
        webView.scrollView.isScrollEnabled = false
        webView.scrollView.bounces = false
        webView.isOpaque = false
        webView.backgroundColor = .gray
        webView.isUserInteractionEnabled = true
        webView.isMultipleTouchEnabled = true

        // Store reference
        context.coordinator.schemeHandler = schemeHandler
        context.coordinator.webView = webView

        // Load from custom scheme (HTML will auto-load room.ply)
        if let url = URL(string: "splat://local/") {
            webView.load(URLRequest(url: url))
            logDebug("AntimatterSplatView: Loading from custom scheme")
        }

        return webView
    }

    func updateUIView(_ uiView: WKWebView, context: Context) {
        // No updates needed
    }

    private func generateSplatViewerHTML(bounds: SIMD3<Float>?, actualBounds: RoomBounds?, orientation: PhotoOrientation) -> String {
        // Use RoomBoundaryManager to calculate camera position
        let roomBounds = actualBounds ?? RoomBoundaryManager.defaultBounds
        let boundaryManager = RoomBoundaryManager(bounds: roomBounds)
        let cameraSetup = boundaryManager.getCameraAtBackWall()

        let eyeX = cameraSetup.eye.x
        let eyeY = cameraSetup.eye.y
        let eyeZ = cameraSetup.eye.z

        let targetX = cameraSetup.target.x
        let targetY = cameraSetup.target.y
        let targetZ = cameraSetup.target.z

        logDebug("📐 [WebGL] Room: \(boundaryManager.width) × \(boundaryManager.height) × \(boundaryManager.depth)")
        logDebug("📐 [WebGL] Center: (\(boundaryManager.centerX), \(boundaryManager.centerY), \(boundaryManager.centerZ))")
        logDebug("📐 [WebGL] Front wall Z: \(boundaryManager.frontWallZ), Back wall Z: \(boundaryManager.backWallZ)")
        logDebug("📷 [WebGL] Camera: eye=(\(eyeX), \(eyeY), \(eyeZ)) target=(\(targetX), \(targetY), \(targetZ))")

        // SparkJS + THREE.js based Gaussian Splat viewer
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
            <title>3D Room</title>
            <style>
                * { margin: 0; padding: 0; box-sizing: border-box; }
                html, body {
                    width: 100%;
                    height: 100%;
                    overflow: hidden;
                    background: #808080;
                    touch-action: none;
                    -webkit-touch-callout: none;
                    -webkit-user-select: none;
                }
                canvas {
                    width: 100%;
                    height: 100%;
                    display: block;
                    touch-action: none;
                }
            </style>
            <script type="importmap">
            {
                "imports": {
                    "three": "https://cdnjs.cloudflare.com/ajax/libs/three.js/0.170.0/three.module.min.js",
                    "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.170.0/examples/jsm/",
                    "@sparkjsdev/spark": "https://sparkjs.dev/releases/spark/0.1.10/spark.module.js"
                }
            }
            </script>
        </head>
        <body>
            <script type="module">
                import * as THREE from 'three';
                import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
                import { SplatMesh, SparkRenderer } from '@sparkjsdev/spark';

                // Photo orientation from Swift
                const isPortrait = \(orientation == .portrait ? "true" : "false");
                console.log('[WebGL] isPortrait =', isPortrait);

                // Log JS errors
                window.addEventListener('error', function(e) {
                    console.log('[JS Error] ' + e.message + ' at ' + e.filename + ':' + e.lineno);
                });

                const jsStartTime = performance.now();

                // Scene setup
                // Use neutral gray background to blend better with room colors
                // (black makes holes/sparse areas very obvious)
                const scene = new THREE.Scene();
                scene.background = new THREE.Color(0x808080);

                // Camera - start at default, autoFrameRoom will position using Box3
                const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
                camera.position.set(0, 0, 5);
                camera.up.set(0, 1, 0);  // Normal Y-up

                // THREE.js Renderer (for SparkRenderer)
                const renderer = new THREE.WebGLRenderer({ antialias: false });  // antialias: false per SparkJS docs
                renderer.setSize(window.innerWidth, window.innerHeight);
                renderer.setPixelRatio(window.devicePixelRatio);
                document.body.appendChild(renderer.domElement);

                // SparkRenderer with settings tuned to reduce visible holes
                // - Increased maxStdDev allows larger gaussians to fill gaps
                // - preBlurAmount adds soft edges to blend sparse areas
                // - falloff controls opacity falloff (lower = more spread)
                const spark = new SparkRenderer({
                    renderer: renderer,
                    maxStdDev: 3.0,           // Larger gaussians to fill gaps
                    preBlurAmount: 0.5,       // Soft blur to blend edges
                    blurAmount: 0.3,          // Slightly more post-blur
                    falloff: 0.8,             // Gentler opacity falloff
                    focalAdjustment: 1.5      // Slightly less aggressive focal adjustment
                });
                camera.add(spark);  // Add SparkRenderer as child of camera

                // Battery optimization: only render when needed
                let needsRender = true;  // Force initial render

                // Orbit controls for touch/mouse
                const controls = new OrbitControls(camera, renderer.domElement);
                controls.enableDamping = true;
                controls.dampingFactor = 0.05;  // More glide/inertia
                controls.rotateSpeed = 0.8;     // Slightly slower, weighted feel
                controls.screenSpacePanning = false;
                controls.minDistance = 0.01;
                controls.maxDistance = 100;
                // Default target, autoFrameRoom will update using Box3
                controls.target.set(0, 0, 0);

                // No full 360° spin - we'll do back-and-forth oscillation instead
                controls.autoRotate = false;

                // Request render when controls change (handles damping animation too)
                controls.addEventListener('change', function() {
                    needsRender = true;
                });

                // --- Back-and-forth orbit state ---
                let autoOrbitEnabled = true;       // master switch for idle orbit
                let autoOrbitTime = 0;             // time accumulator
                let autoOrbitBaseAngle = 0;        // center angle around target
                let autoOrbitRadius = 5;           // distance from target to camera
                const clock = new THREE.Clock();

                // Global interaction flag (used by auto-orbit)
                window._userInteracting = false;

                // Called from Swift AND by OrbitControls
                window.setUserInteracting = function(flag) {
                    window._userInteracting = !!flag;
                };

                // Let OrbitControls also toggle interaction
                controls.addEventListener('start', () => { window._userInteracting = true; });
                controls.addEventListener('end',   () => { window._userInteracting = false; });

                // Unlimited spin around object (train-style orbit)
                controls.minAzimuthAngle = -Infinity;
                controls.maxAzimuthAngle =  Infinity;
                controls.minPolarAngle = 0.01;           // just above straight up
                controls.maxPolarAngle = Math.PI - 0.01; // just below straight down

                // Camera clamping disabled for train-style free orbit
                // Re-enable if needed for room boundary constraints
                /*
                controls.addEventListener('change', () => {
                    if (!roomBoundsForClamping) return;
                    const margin = 0.1;
                    const b = roomBoundsForClamping;
                    camera.position.x = Math.max(b.minX + margin, Math.min(b.maxX - margin, camera.position.x));
                    camera.position.z = Math.max(b.minZ + margin, Math.min(b.maxZ - margin, camera.position.z));
                    camera.position.y = Math.max(b.minY - margin, Math.min(b.maxY + margin, camera.position.y));
                });
                */

                // Save initial camera state for recenter (will be updated by autoFrameRoom)
                let initialCameraPosition = camera.position.clone();
                let initialControlsTarget = controls.target.clone();

                // Recenter function
                window.recenterCamera = function() {
                    camera.position.copy(initialCameraPosition);
                    controls.target.copy(initialControlsTarget);
                    controls.update();
                };

                // Function to update initial position (called after auto-frame)
                window.updateInitialPosition = function() {
                    initialCameraPosition = camera.position.clone();
                    initialControlsTarget = controls.target.clone();
                    console.log('Updated initial position for recenter:', initialCameraPosition);
                };

                // Load PLY splat using SparkJS
                const plyURL = 'splat://local/room.ply';
                console.log('Loading splat from:', plyURL);

                let splatMesh = null;

                try {
                    splatMesh = new SplatMesh({
                        url: plyURL,
                        maxSh: 0  // Disable spherical harmonics for cleaner look
                    });
                    scene.add(splatMesh);

                    // Classic PLY is pre-rotated, flip 180° around X + 90° around Z for correct viewing
                    splatMesh.rotation.x = Math.PI;
                    splatMesh.rotation.z = Math.PI / 2;
                    console.log('SplatMesh: rotated 180° X + 90° Z');

                    // Auto-frame using Box3 (no orientation changes)
                    function autoFrameRoom() {
                        console.log('=== autoFrameRoom() called (Box3 framing only) ===');
                        try {
                            if (!splatMesh) {
                                console.log('No splatMesh yet, retrying...');
                                setTimeout(autoFrameRoom, 200);
                                return;
                            }

                            // 1) Compute bounds on the mesh (no rotation applied)
                            const box = new THREE.Box3().setFromObject(splatMesh);
                            const size = box.getSize(new THREE.Vector3());

                            if (size.length() < 0.01) {
                                console.log('Box3 too small, waiting for splatMesh...');
                                setTimeout(autoFrameRoom, 200);
                                return;
                            }

                            const center = box.getCenter(new THREE.Vector3());
                            // After 90° Z rotation, X and Y axes are swapped
                            // Front wall width = Y axis, height = X axis
                            let roomWidth  = size.y;
                            let roomHeight = size.x;
                            let roomDepth  = size.z;

                            console.log('Box3 size:', roomWidth.toFixed(2), roomHeight.toFixed(2), roomDepth.toFixed(2));
                            console.log('Box3 center:', center.x.toFixed(2), center.y.toFixed(2), center.z.toFixed(2));

                            // 2) Raw bounds from Box3
                            let rawMinX = box.min.x;
                            let rawMaxX = box.max.x;
                            let rawMinY = box.min.y;
                            let rawMaxY = box.max.y;
                            let rawMinZ = box.min.z;
                            let rawMaxZ = box.max.z;

                            // 3) Shrink bounds to ignore foggy outer 15%
                            const fogFactor = 0.15;
                            const shrinkX = roomWidth  * fogFactor * 0.5;
                            const shrinkY = roomHeight * fogFactor * 0.5;
                            const shrinkZ = roomDepth  * fogFactor * 0.5;

                            const minX = rawMinX + shrinkX;
                            const maxX = rawMaxX - shrinkX;
                            const minY = rawMinY + shrinkY;
                            const maxY = rawMaxY - shrinkY;
                            const minZ = rawMinZ + shrinkZ;
                            const maxZ = rawMaxZ - shrinkZ;

                            const innerCenterX = (minX + maxX) / 2;
                            const innerCenterY = (minY + maxY) / 2;
                            const innerCenterZ = (minZ + maxZ) / 2;

                            // Store tightened bounds for joystick clamping
                            roomBoundsForClamping = {
                                minX, maxX,
                                minY, maxY,
                                minZ, maxZ,
                                centerX: innerCenterX,
                                centerY: innerCenterY,
                                centerZ: innerCenterZ
                            };

                            roomRadius = Math.max(
                                maxX - minX,
                                maxY - minY,
                                Math.abs(maxZ - minZ)
                            ) * 0.5;

                            // Store for oscillation camera path
                            window.roomViewParams = {
                                centerX: innerCenterX,
                                centerY: innerCenterY,
                                centerZ: innerCenterZ,
                                radius: roomRadius * 1.2
                            };

                            console.log('Inner bounds:', JSON.stringify(roomBoundsForClamping));

                            // 4) Send wall dimensions back to Swift UI
                            if (window.webkit?.messageHandlers?.frontWallDimensions) {
                                window.webkit.messageHandlers.frontWallDimensions.postMessage({
                                    width: roomWidth,
                                    height: roomHeight
                                });
                            }

                            // 5) Place camera outside, looking at inner center
                            const fov = camera.fov * (Math.PI / 180);
                            const maxDim = Math.max(roomWidth, roomHeight, roomDepth);
                            const cameraDistance = (maxDim / 2) / Math.tan(fov / 2) * 1.5;

                            const newCamPos = new THREE.Vector3(
                                innerCenterX,
                                innerCenterY,
                                innerCenterZ + cameraDistance
                            );
                            const newTarget = new THREE.Vector3(innerCenterX, innerCenterY, innerCenterZ);

                            console.log('Camera distance:', cameraDistance.toFixed(2));
                            console.log('Camera pos:', newCamPos.x.toFixed(2), newCamPos.y.toFixed(2), newCamPos.z.toFixed(2));
                            console.log('Target:', newTarget.x.toFixed(2), newTarget.y.toFixed(2), newTarget.z.toFixed(2));

                            camera.position.copy(newCamPos);
                            controls.target.copy(newTarget);
                            controls.update();

                            // 6) Update recenter base
                            initialCameraPosition.copy(camera.position);
                            initialControlsTarget.copy(controls.target);

                            // 7) Setup base orbit parameters around this view
                            autoOrbitRadius = camera.position.distanceTo(controls.target);
                            autoOrbitBaseAngle = Math.atan2(
                                camera.position.x - controls.target.x,
                                camera.position.z - controls.target.z
                            );
                            console.log('Auto-orbit base radius:', autoOrbitRadius.toFixed(2),
                                        'baseAngle:', autoOrbitBaseAngle.toFixed(2));

                            // Report camera pose to Swift for debugging
                            if (window.webkit?.messageHandlers?.cameraPose) {
                                window.webkit.messageHandlers.cameraPose.postMessage({
                                    ex: camera.position.x,
                                    ey: camera.position.y,
                                    ez: camera.position.z,
                                    tx: controls.target.x,
                                    ty: controls.target.y,
                                    tz: controls.target.z
                                });
                            }

                            console.log('=== Camera positioned using Box3 bounds ===');
                        } catch (err) {
                            console.error('autoFrameRoom error:', err);
                        }
                    }
                    console.log('Scheduling autoFrameRoom in 500ms...');
                    setTimeout(autoFrameRoom, 500);

                } catch (err) {
                    console.error('Failed to load splat:', err);
                }

                // Handle resize
                window.addEventListener('resize', () => {
                    camera.aspect = window.innerWidth / window.innerHeight;
                    camera.updateProjectionMatrix();
                    renderer.setSize(window.innerWidth, window.innerHeight);
                });

                // Room bounds for camera clamping (will be set by autoFrameRoom)
                let roomBoundsForClamping = null;
                let roomRadius = null;

                // Joystick movement handler with boundary clamping
                window.moveCamera = function(dx, dy) {
                    // Stop auto orbit when user starts walking
                    autoOrbitEnabled = false;

                    const moveSpeed = 0.03;  // Increased for better walk feel

                    // Calculate new position
                    let newX = camera.position.x + dx * moveSpeed;
                    let newZ = camera.position.z + dy * moveSpeed;

                    // Clamp to room bounds if available
                    if (roomBoundsForClamping) {
                        const margin = 0.2;  // Smaller margin to approach walls more closely
                        newX = Math.max(roomBoundsForClamping.minX + margin,
                               Math.min(roomBoundsForClamping.maxX - margin, newX));
                        newZ = Math.max(roomBoundsForClamping.minZ + margin,
                               Math.min(roomBoundsForClamping.maxZ - margin, newZ));
                    }

                    // Apply clamped movement
                    const actualDx = newX - camera.position.x;
                    const actualDz = newZ - camera.position.z;

                    camera.position.x = newX;
                    camera.position.z = newZ;
                    controls.target.x += actualDx;
                    controls.target.z += actualDz;
                    needsRender = true;  // Request render after camera move
                };

                // Orbit rotation handler (for Swift gesture overlay)
                window.orbitCamera = function(deltaX, deltaY) {
                    // Stop auto orbit when user drags
                    autoOrbitEnabled = false;

                    // Rotate around target using spherical coordinates
                    const rotateSpeed = 0.005;

                    // Get current spherical position relative to target
                    const offset = new THREE.Vector3().subVectors(camera.position, controls.target);
                    const spherical = new THREE.Spherical().setFromVector3(offset);

                    // Apply rotation
                    spherical.theta -= deltaX * rotateSpeed;  // Horizontal rotation
                    spherical.phi += deltaY * rotateSpeed;    // Vertical rotation

                    // Clamp phi to avoid flipping
                    spherical.phi = Math.max(0.1, Math.min(Math.PI - 0.1, spherical.phi));

                    // Convert back to cartesian
                    offset.setFromSpherical(spherical);
                    camera.position.copy(controls.target).add(offset);

                    controls.update();
                    needsRender = true;  // Request render after orbit
                };

                // Zoom handler (for Swift pinch gesture)
                window.zoomCamera = function(scale) {
                    // Stop auto orbit when user pinches
                    autoOrbitEnabled = false;

                    const offset = new THREE.Vector3().subVectors(camera.position, controls.target);
                    offset.multiplyScalar(1 / scale);

                    // Clamp distance
                    const dist = offset.length();
                    if (dist < 0.5) offset.setLength(0.5);
                    if (dist > 50) offset.setLength(50);

                    camera.position.copy(controls.target).add(offset);
                    controls.update();
                    needsRender = true;  // Request render after zoom
                };

                // Animation loop with battery optimization
                // - Only renders when needed (user interacting or auto-orbit active)
                // - Throttles to 30fps when idle auto-orbit is running
                // - Stops rendering completely when static
                let loadNotified = false;
                let lastRenderTime = 0;
                const IDLE_FPS = 30;     // Lower FPS for auto-orbit (saves battery)
                const IDLE_FRAME_TIME = 1000 / IDLE_FPS;

                // Request render on next frame (called when something changes)
                window.requestRender = function() {
                    needsRender = true;
                };

                function animate(currentTime) {
                    requestAnimationFrame(animate);

                    const dt = clock.getDelta();
                    let shouldRender = needsRender;
                    needsRender = false;

                    // Back-and-forth orbit when not interacting (uses base angle from autoFrameRoom)
                    if (autoOrbitEnabled && !window._userInteracting && autoOrbitRadius > 0.1) {
                        autoOrbitTime += dt;

                        const speed = 0.35;
                        const t = controls.target;

                        if (isPortrait) {
                            // Portrait: circular arc oscillation ±30°
                            const amplitude = Math.PI / 6;
                            const angle = autoOrbitBaseAngle + amplitude * Math.sin(autoOrbitTime * speed);

                            camera.position.x = t.x + autoOrbitRadius * Math.sin(angle);
                            camera.position.z = t.z + autoOrbitRadius * Math.cos(angle);
                        } else {
                            // Landscape: horizontal left-right sweep only
                            const sweepAmount = autoOrbitRadius * 0.3 * Math.sin(autoOrbitTime * speed);

                            camera.position.x = initialCameraPosition.x + sweepAmount;
                            camera.position.z = initialCameraPosition.z;
                        }

                        // Throttle auto-orbit to 30fps to save battery
                        if (currentTime - lastRenderTime >= IDLE_FRAME_TIME) {
                            shouldRender = true;
                        }
                    }

                    // Always render during user interaction (60fps)
                    if (window._userInteracting) {
                        shouldRender = true;
                    }

                    // Skip render if nothing changed (huge battery savings)
                    if (!shouldRender) {
                        return;
                    }

                    lastRenderTime = currentTime;
                    controls.update();

                    // Use SparkRenderer's update method for optimized Gaussian rendering
                    spark.update({ scene });
                    renderer.render(scene, camera);

                    // Notify Swift when loaded
                    if (!loadNotified && splatMesh) {
                        loadNotified = true;
                        const totalTime = performance.now() - jsStartTime;
                        if (window.webkit && window.webkit.messageHandlers && window.webkit.messageHandlers.splatLoaded) {
                            window.webkit.messageHandlers.splatLoaded.postMessage({ loaded: true, timeMs: totalTime });
                        }
                        console.log('Splat loaded in', totalTime, 'ms');
                    }
                }
                animate(0);
            </script>
        </body>
        </html>
        """
    }

    class Coordinator: NSObject, WKNavigationDelegate, WKScriptMessageHandler {
        var webView: WKWebView?
        var schemeHandler: LocalFileSchemeHandler?
        let onLoaded: () -> Void
        let onFrontWallDimensions: ((Double, Double) -> Void)?
        private var hasNotifiedLoaded = false
        private var loadStartTime: CFAbsoluteTime = 0

        private var joystickTimer: Timer?

        init(onLoaded: @escaping () -> Void, onFrontWallDimensions: ((Double, Double) -> Void)? = nil) {
            self.onLoaded = onLoaded
            self.onFrontWallDimensions = onFrontWallDimensions
            self.loadStartTime = CFAbsoluteTimeGetCurrent()
            logDebug("⏱️ [WebGL] Starting load...")
            super.init()

            // Listen for recenter notification
            NotificationCenter.default.addObserver(
                self,
                selector: #selector(recenterCamera),
                name: NSNotification.Name("RecenterWebGLCamera"),
                object: nil
            )

            // Listen for joystick movement
            NotificationCenter.default.addObserver(
                self,
                selector: #selector(handleJoystickMove(_:)),
                name: NSNotification.Name("WebGLJoystickMove"),
                object: nil
            )

            // Listen for orbit gesture
            NotificationCenter.default.addObserver(
                self,
                selector: #selector(handleOrbitGesture(_:)),
                name: NSNotification.Name("WebGLOrbitGesture"),
                object: nil
            )

            // Listen for zoom gesture
            NotificationCenter.default.addObserver(
                self,
                selector: #selector(handleZoomGesture(_:)),
                name: NSNotification.Name("WebGLZoomGesture"),
                object: nil
            )

            // Listen for gesture state (to pause/resume auto-orbit)
            NotificationCenter.default.addObserver(
                self,
                selector: #selector(handleOrbitGestureState(_:)),
                name: NSNotification.Name("WebGLOrbitGestureState"),
                object: nil
            )
        }

        deinit {
            joystickTimer?.invalidate()
            NotificationCenter.default.removeObserver(self)
        }

        @objc private func recenterCamera() {
            logDebug("🎯 [WebGL] Recentering camera")
            let js = "if (typeof recenterCamera === 'function') recenterCamera();"
            webView?.evaluateJavaScript(js) { _, error in
                if let error = error {
                    logDebug("❌ [WebGL] Recenter JS error: \(error)")
                }
            }
        }

        @objc private func handleJoystickMove(_ notification: Notification) {
            guard let userInfo = notification.userInfo,
                  let offset = userInfo["offset"] as? CGSize else { return }

            let dx = Float(offset.width)
            let dy = Float(-offset.height)  // Invert Y for intuitive control

            // Skip if no movement
            if abs(dx) < 0.01 && abs(dy) < 0.01 { return }

            // Call SparkJS moveCamera function
            let js = "if (typeof moveCamera === 'function') moveCamera(\(dx), \(dy));"
            webView?.evaluateJavaScript(js, completionHandler: nil)
        }

        @objc private func handleOrbitGesture(_ notification: Notification) {
            guard let userInfo = notification.userInfo,
                  let deltaX = userInfo["deltaX"] as? CGFloat,
                  let deltaY = userInfo["deltaY"] as? CGFloat else { return }

            logDebug("🎮 [WebGL] Orbit gesture: dx=\(deltaX), dy=\(deltaY)")

            // OrbitGestureView now sends incremental values directly
            let js = "if (typeof orbitCamera === 'function') orbitCamera(\(deltaX), \(deltaY));"
            webView?.evaluateJavaScript(js, completionHandler: nil)
        }

        @objc private func handleZoomGesture(_ notification: Notification) {
            guard let userInfo = notification.userInfo,
                  let scale = userInfo["scale"] as? CGFloat else { return }

            logDebug("🎚️ [WebGL] Zoom gesture: scale=\(scale)")

            // OrbitGestureView now sends incremental scale directly
            let js = "if (typeof zoomCamera === 'function') zoomCamera(\(scale));"
            webView?.evaluateJavaScript(js, completionHandler: nil)
        }

        @objc private func handleOrbitGestureState(_ notification: Notification) {
            guard let userInfo = notification.userInfo,
                  let interacting = userInfo["interacting"] as? Bool else { return }

            let js = "if (typeof setUserInteracting === 'function') setUserInteracting(\(interacting ? "true" : "false"));"
            logDebug("🎮 [WebGL] setUserInteracting(\(interacting))")
            webView?.evaluateJavaScript(js, completionHandler: nil)
        }

        func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
            let elapsed = (CFAbsoluteTimeGetCurrent() - loadStartTime) * 1000
            logDebug("⏱️ [WebGL] HTML page loaded: \(String(format: "%.0f", elapsed))ms")

            // Fallback: notify loaded after delay if JS doesn't report
            DispatchQueue.main.asyncAfter(deadline: .now() + 5.0) {
                if !self.hasNotifiedLoaded {
                    self.hasNotifiedLoaded = true
                    let total = (CFAbsoluteTimeGetCurrent() - self.loadStartTime) * 1000
                    logDebug("⏱️ [WebGL] Total load (fallback): \(String(format: "%.0f", total))ms")
                    self.onLoaded()
                }
            }
        }

        func webView(_ webView: WKWebView, didFail navigation: WKNavigation!, withError error: Error) {
            logDebug("❌ [WebGL] Navigation failed: \(error)")
        }

        func webView(_ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!, withError error: Error) {
            logDebug("❌ [WebGL] Provisional navigation failed: \(error)")
        }

        func userContentController(_ userContentController: WKUserContentController, didReceive message: WKScriptMessage) {
            if message.name == "splatLoaded" {
                if let body = message.body as? [String: Any], body["loaded"] != nil {
                    let elapsed = (CFAbsoluteTimeGetCurrent() - loadStartTime) * 1000
                    logDebug("⏱️ [WebGL] Splat rendered: \(String(format: "%.0f", elapsed))ms total")

                    if !hasNotifiedLoaded {
                        hasNotifiedLoaded = true
                        DispatchQueue.main.async {
                            self.onLoaded()
                        }
                    }
                }
            } else if message.name == "frontWallDimensions" {
                if let body = message.body as? [String: Any],
                   let w = body["width"] as? Double,
                   let h = body["height"] as? Double {
                    logDebug("📏 [WebGL] Front wall from JS: \(String(format: "%.2f", w)) × \(String(format: "%.2f", h))")
                    DispatchQueue.main.async {
                        self.onFrontWallDimensions?(w, h)
                    }
                }
            } else if message.name == "cameraPose" {
                if let body = message.body as? [String: Any],
                   let ex = body["ex"] as? Double,
                   let ey = body["ey"] as? Double,
                   let ez = body["ez"] as? Double,
                   let tx = body["tx"] as? Double,
                   let ty = body["ty"] as? Double,
                   let tz = body["tz"] as? Double {
                    logDebug("🎥 [WebGL] Camera pose JS -> Swift: eye=(\(String(format: "%.2f", ex)), \(String(format: "%.2f", ey)), \(String(format: "%.2f", ez))) target=(\(String(format: "%.2f", tx)), \(String(format: "%.2f", ty)), \(String(format: "%.2f", tz)))")
                }
            }
        }
    }
}

// MARK: - Share Sheet

/// UIActivityViewController wrapper for sharing files
struct ShareSheet: UIViewControllerRepresentable {
    let activityItems: [Any]

    func makeUIViewController(context: Context) -> UIActivityViewController {
        let controller = UIActivityViewController(
            activityItems: activityItems,
            applicationActivities: nil
        )
        return controller
    }

    func updateUIViewController(_ uiViewController: UIActivityViewController, context: Context) {}
}

// MARK: - Preview

#Preview {
    NavigationStack {
        // Preview with a sample URL (won't actually load)
        SharpRoomView(plyURL: URL(fileURLWithPath: "/sample.ply"))
    }
}

