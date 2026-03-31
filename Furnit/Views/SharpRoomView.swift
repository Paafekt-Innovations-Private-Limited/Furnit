import SwiftUI
import CoreML
import Photos
import MetalKit
import simd

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

    /// Match Android RoomBoundaryManager: depth-adaptive inset (18% for tiny rooms, up to 50% for deep).
    private static func backCenterInsetFraction(depth: Float) -> Float {
        let t = min(1.0, max(0.0, depth / 6.0))
        return 0.18 + 0.32 * t
    }

    /// Camera at back center with depth-adaptive inset (matches Android when room opened from list / room created).
    private let cameraPadding: Float = 0.3

    /// Calculate camera position using Android formula: back center, depth-adaptive inset, look at front wall.
    func getCameraAtBackCenter() -> (eye: SIMD3<Float>, target: SIMD3<Float>) {
        let result = bounds.defaultSplatCameraEyeAndTarget(cameraPadding: cameraPadding)
        let fraction = Self.backCenterInsetFraction(depth: depth)
        let insetFromBack = max(depth * fraction, cameraPadding)
        logDebug("📷 [BoundaryManager] getCameraAtBackCenter depth=\(depth) fraction=\(fraction) inset=\(insetFromBack) eye=(\(result.eye.x),\(result.eye.y),\(result.eye.z)) target=(\(result.target.x),\(result.target.y),\(result.target.z))")
        return result
    }

    /// Calculate camera position just INSIDE the room (matches Android formula for list / created room).
    func getCameraAtBackWall(fovDegrees: Float = 60) -> (eye: SIMD3<Float>, target: SIMD3<Float>) {
        return getCameraAtBackCenter()
    }
}


/// SHARP room viewer: **MetalSplatter** (Metal) for the splat layer.
struct SharpRoomView: View {
    let plyURL: URL
    let allowSave: Bool  // Show save button (true for new rooms, false for viewing from home)
    let photoOrientation: PhotoOrientation  // Source photo orientation (for UI layout)
    let savedRoomWidth: Float?  // Room width from saved metadata (for HomeView)
    let savedRoomHeight: Float?  // Room height from saved metadata (for HomeView)
    /// When opening a saved PLY from Home, used for YOLO ratio calibration + FurnitureFit targets.
    let savedRoomModel: USDZModel?
    @Environment(\.dismiss) private var dismiss

    /// PLY chosen for display: prefer `_3dgs.ply`, then `_classic.ply`, else base.
    private let viewerPlyURL: URL
    /// Classic PLY (front-rotated) when available; falls back to base.
    private let classicPlyURL: URL

    init(
        plyURL: URL,
        allowSave: Bool = true,
        photoOrientation: PhotoOrientation = .portrait,
        savedRoomWidth: Float? = nil,
        savedRoomHeight: Float? = nil,
        savedRoomModel: USDZModel? = nil
    ) {
        self.plyURL = plyURL
        self.allowSave = allowSave
        self.photoOrientation = photoOrientation
        self.savedRoomWidth = savedRoomWidth
        self.savedRoomHeight = savedRoomHeight
        self.savedRoomModel = savedRoomModel

        let basePath = plyURL.path
        let threeDGS = URL(fileURLWithPath: basePath.replacingOccurrences(of: ".ply", with: "_3dgs.ply"))
        let classic = URL(fileURLWithPath: basePath.replacingOccurrences(of: ".ply", with: "_classic.ply"))
        let fm = FileManager.default

        if fm.fileExists(atPath: threeDGS.path) {
            self.viewerPlyURL = threeDGS
        } else if fm.fileExists(atPath: classic.path) {
            self.viewerPlyURL = classic
        } else {
            self.viewerPlyURL = plyURL
        }

        if fm.fileExists(atPath: classic.path) {
            self.classicPlyURL = classic
        } else {
            self.classicPlyURL = plyURL
        }
    }

    /// Bounds from MetalSplatter (`SplatRenderer.boundingBox`) for the displayed PLY.
    @State private var metalBounds: RoomBounds?

    /// Parsed bounds from the displayed PLY file (now fed by MetalSplatter, not a second PLY parse).
    private var effectiveBounds: RoomBounds? {
        metalBounds
    }

    @State private var isLoading = true
    @State private var error: String?
    @State private var showingFurnitureFit = false
    @ObservedObject private var yoloeService = YOLOEModelService.shared

    // JS-measured front wall dimensions (from actual splat bounds)
    @State private var jsFrontWallWidth: Float?
    @State private var jsFrontWallHeight: Float?

    // Detected furniture size (from FurnitureFit, in meters - before calibration)
    @State private var detectedFurnitureWidth: Float?
    @State private var detectedFurnitureHeight: Float?
    /// Optional AR-based furniture height when ARKit world tracking is active in Furniture Fit.
    @State private var detectedFurnitureHeightAR: Float?

    // User-input real furniture dimensions for room calibration
    @State private var showFurnitureDimensionsInput = false
    @State private var inputFurnitureHeight: String = ""
    @State private var realFurnitureHeight: Float?  // Confirmed real height in meters

    // Calibrated room dimensions (computed from real furniture size)
    @State private var calibratedRoomHeight: Float?
    @State private var calibratedRoomWidth: Float?
    /// Track whether we've already auto-applied AR-based calibration this session.
    @State private var didApplyArAutoCalibration = false

    // Reject calibration when result would be unrealistically small (wrong input or wrong detected size)
    @State private var showCalibrationRejectAlert = false
    @State private var calibrationRejectMessage = ""

    // Wall-based calibration (tape-measured front wall)
    @State private var showWallCalibration = false
    @State private var inputWallWidth: String = ""
    @State private var inputWallHeight: String = ""

    /// When true, shows ⋮ Calibrate wall and the Tap to calibrate pill during brain (FurnitureFit). Default off (matches Android).
    @AppStorage("show_room_furniture_calibrate") private var showRoomFurnitureCalibrate = false

    // Save room state
    @StateObject private var modelManager = USDZModelManager()
    @State private var isSavingRoom = false
    @State private var saveProgress: Double = 0.0
    @State private var savingTimer: Timer?
    @State private var showSaveAlert = false
    @State private var saveAlertMessage = ""
    @State private var saveWasSuccessful = false
    @State private var showDiscardUnsavedAlert = false
    @State private var isDismissing = false
    @State private var showRoomNameInput = false
    @State private var roomName = ""
    @State private var showShareSheet = false
    @State private var isCapturingSnapshot = false
    /// After first Furniture Fit segmentation this viewer session, skip startup progress when toggling brain on again.
    @State private var furnitureFitInitialSegmentationDone = false
    /// Pinch zoom for MetalSplatter (`GaussianSplatView`).
    @State private var metalSplatterZoom: Float = 1.0
    @EnvironmentObject var authManager: AuthenticationManager

    var body: some View {
        sharpRoomBody
    }

    private var sharpRoomBody: some View {
        ZStack {
            metalSplatAndGestureLayer
            allOverlaysLayer
        }
        .background(Color.gray)
        .navigationBarHidden(isCapturingSnapshot)
        .navigationTitle(navigationTitleWithDimensions)
        .navigationBarTitleDisplayMode(.inline)
        .navigationBarBackButtonHidden(allowSave)
        .toolbar {
            if allowSave {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button {
                        if saveWasSuccessful {
                            dismiss()
                        } else {
                            showDiscardUnsavedAlert = true
                        }
                    } label: {
                        HStack(spacing: 4) {
                            Image(systemName: "chevron.left")
                            Text(L10n.Common.back)
                        }
                    }
                }
            }
            ToolbarItemGroup(placement: .navigationBarTrailing) {
                    // One ellipsis menu in portrait and landscape so the nav title (W×H) stays visible.
                    Menu {
                        if authManager.canShare {
                            Button(action: { showShareSheet = true }) {
                                Label(L10n.RoomViewer.share, systemImage: "square.and.arrow.up")
                            }
                        }
                        if allowSave {
                            Button(action: { showRoomNameInput = true }) {
                                Label(L10n.RoomViewer.saveRoom, systemImage: "square.and.arrow.down")
                            }
                        }
                        if showRoomFurnitureCalibrate {
                            Button(action: { showWallCalibration = true }) {
                                Label(L10n.RoomViewer.calibrateWall, systemImage: "ruler")
                            }
                        }
                        Button(action: {
                            NotificationCenter.default.post(name: NSNotification.Name("RecenterWebGLCamera"), object: nil)
                        }) {
                            Label(L10n.RoomViewer.recenterView, systemImage: "viewfinder")
                        }
                        Button(action: {
                            NotificationCenter.default.post(name: NSNotification.Name("FurnitureFitResetOverlayScale"), object: nil)
                        }) {
                            Label(L10n.RoomViewer.resetOverlayScale, systemImage: "arrow.counterclockwise")
                        }
                    } label: {
                        Image(systemName: "ellipsis.circle")
                    }
                    .disabled(isLoading)
                }
            }
        .sheet(isPresented: $showShareSheet) {
            let threeDGSPath = plyURL.path.replacingOccurrences(of: ".ply", with: "_3dgs.ply")
            let shareURL = FileManager.default.fileExists(atPath: threeDGSPath) ? URL(fileURLWithPath: threeDGSPath) : plyURL
            ShareSheet(activityItems: [shareURL])
        }
        .onAppear {
            // Do not load YOLOE here — it peaks memory with WebKit (WKWebView) and can crash in WKWebViewConfiguration.
            // Load when the user turns on Furniture Fit (brain) instead.
            if photoOrientation == .landscape { OrientationLockManager.shared.lockToLandscape() } else { OrientationLockManager.shared.lockToPortrait() }
            logDebug("📐 [SharpRoomView] photoOrientation = \(photoOrientation)")
        }
        .onChange(of: showingFurnitureFit) { _, isOn in
            if isOn {
                yoloeService.ensureModelLoaded()
            } else {
                yoloeService.releaseResources()
            }
        }
        .onChange(of: showRoomFurnitureCalibrate) { _, enabled in
            if !enabled {
                showFurnitureDimensionsInput = false
                showWallCalibration = false
            }
        }
        .onDisappear {
            OrientationLockManager.shared.unlock()
            SHARPService.shared.releaseResources()
            yoloeService.releaseResources()
        }
        .alert(L10n.RoomViewer.saveRoom, isPresented: $showRoomNameInput) {
            TextField(L10n.RoomViewer.roomName, text: $roomName)
            Button(L10n.Common.cancel, role: .cancel) { }
            Button(L10n.Common.save) { startSavingRoom() }.disabled(roomName.isEmpty)
        } message: { Text(L10n.RoomViewer.enterName) }
        .alert(L10n.RoomViewer.roomSaveTitle, isPresented: $showSaveAlert) {
            Button(L10n.Common.ok, role: .cancel) {
                if saveWasSuccessful {
                    isDismissing = true
                    DispatchQueue.global(qos: .userInitiated).async {
                        DispatchQueue.main.async { SHARPService.shared.releaseResources() }
                        Thread.sleep(forTimeInterval: 0.1)
                        DispatchQueue.main.async { NotificationCenter.default.post(name: NSNotification.Name("DismissPhotoRoomSheet"), object: nil) }
                    }
                }
            }
        } message: { Text(saveAlertMessage) }
        .alert(L10n.RoomViewer.checkMeasurement, isPresented: $showCalibrationRejectAlert) { Button(L10n.Common.ok, role: .cancel) { } } message: { Text(calibrationRejectMessage) }
        .alert(L10n.RoomPreview.unsavedTitle, isPresented: $showDiscardUnsavedAlert) {
            Button(L10n.RoomPreview.stay, role: .cancel) {}
            Button(L10n.RoomPreview.leave, role: .destructive) {
                dismiss()
            }
        } message: {
            Text(L10n.RoomPreview.unsavedMessage)
        }
        .overlay {
            if isDismissing {
                ZStack {
                    Color.black.opacity(0.7).ignoresSafeArea()
                    VStack(spacing: 16) {
                        ProgressView().scaleEffect(1.5).tint(.white)
                        Text(L10n.RoomViewer.goingBack).foregroundColor(.white).font(.headline)
                    }
                }
            }
        }
        .defersSystemGestures(on: .all)
        .disableBackSwipeIf(allowSave)
    }

    /// Native Metal splats via MetalSplatter; gestures are on `MTKView`. Menu / D-pad use the same notifications as the old WebGL viewer (`GaussianSplatView` observes them).
    private var metalSplatAndGestureLayer: some View {
        GaussianSplatView(
            plyURL: viewerPlyURL,
            isLoading: $isLoading,
            loadError: $error,
            zoomLevel: $metalSplatterZoom,
            onBoundsAvailable: { bounds in
                // Defer state mutations to the next runloop tick to avoid
                // "Modifying state during view update" warnings from SwiftUI.
                DispatchQueue.main.async {
                    metalBounds = bounds
                    seedFrontWallDimensionsFromPlyBoundsIfNeeded()
                }
            }
        )
        .ignoresSafeArea()
        .onAppear {
            if !allowSave {
                metalSplatterZoom = 1.0
            }
        }
    }

    /// WebGL reported Box3 front wall; Metal path now uses MetalSplatter's AABB (`boundingBox`) with fog trim + caps until save/YOLO updates.
    private func seedFrontWallDimensionsFromPlyBoundsIfNeeded() {
        guard jsFrontWallWidth == nil, jsFrontWallHeight == nil, let b = effectiveBounds else { return }

        // 1) Trim foggy outer edges: mirror WebGL's 15% shrink based on Box3 size.
        let fogTrim: Float = 0.15
        let trimmedWidth = max(0, b.width * (1 - fogTrim))
        let trimmedHeight = max(0, b.height * (1 - fogTrim))

        // 2) Cap to realistic room dimensions (matches WebGL SparkJS path).
        let maxRealisticWidth: Float = (photoOrientation == .portrait) ? 5.0 : 8.0
        let maxRealisticHeight: Float = (photoOrientation == .portrait) ? 3.5 : 3.2

        jsFrontWallWidth = min(trimmedWidth, maxRealisticWidth)
        jsFrontWallHeight = min(trimmedHeight, maxRealisticHeight)
    }

    /// On-screen pan pad (not in the ⋮ menu).
    private var cameraButtonsOverlay: some View {
        ZStack(alignment: .topLeading) {
            Color.clear
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .allowsHitTesting(false)
            HStack(spacing: 8) {
                Button(action: { NotificationCenter.default.post(name: NSNotification.Name("WebGLCameraMoveLeft"), object: nil) }) {
                    Image(systemName: "arrow.left")
                        .font(.system(size: 20, weight: .semibold))
                        .foregroundColor(.white)
                        .frame(width: 44, height: 44)
                        .background(Circle().fill(Color.black.opacity(0.5)))
                }
                .buttonStyle(.plain)
                VStack(spacing: 8) {
                    Button(action: { NotificationCenter.default.post(name: NSNotification.Name("WebGLCameraMoveUp"), object: nil) }) {
                        Image(systemName: "arrow.up")
                            .font(.system(size: 20, weight: .semibold))
                            .foregroundColor(.white)
                            .frame(width: 44, height: 44)
                            .background(Circle().fill(Color.black.opacity(0.5)))
                    }
                    .buttonStyle(.plain)
                    Button(action: { NotificationCenter.default.post(name: NSNotification.Name("WebGLCameraMoveDown"), object: nil) }) {
                        Image(systemName: "arrow.down")
                            .font(.system(size: 20, weight: .semibold))
                            .foregroundColor(.white)
                            .frame(width: 44, height: 44)
                            .background(Circle().fill(Color.black.opacity(0.5)))
                    }
                    .buttonStyle(.plain)
                }
                Button(action: { NotificationCenter.default.post(name: NSNotification.Name("WebGLCameraMoveRight"), object: nil) }) {
                    Image(systemName: "arrow.right")
                        .font(.system(size: 20, weight: .semibold))
                        .foregroundColor(.white)
                        .frame(width: 44, height: 44)
                        .background(Circle().fill(Color.black.opacity(0.5)))
                }
                .buttonStyle(.plain)
            }
            .padding(12)
        }
        .opacity(isCapturingSnapshot ? 0 : 1)
        .zIndex(12)
    }

    private var loadingOverlayView: some View {
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

    @ViewBuilder private var errorOverlayView: some View {
        if let err = error {
            VStack(spacing: 16) {
                Image(systemName: "exclamationmark.triangle")
                    .font(.largeTitle)
                    .foregroundColor(.orange)
                Text(err)
                    .foregroundColor(.white)
                    .multilineTextAlignment(.center)
            }
            .padding(24)
            .background(Color.black.opacity(0.7))
            .cornerRadius(12)
        }
    }

    /// Width/height for nav title, FurnitureFit, and save.
    /// - **Live session** (`allowSave == true`): WebGL Box3 (`jsFrontWall*`) until YOLO wall measure on save overwrites `jsFrontWall*` so **Furniture Fit** `room*Meters` / m-per-pixel match measured W×H.
    /// - **Opened from Home** (`allowSave == false`): **saved .meta** (`savedRoom*`) wins over WebGL (5×3.5 cap must not override measured dims).
    private var displayRoomWidth: Float {
        if allowSave {
            return calibratedRoomWidth
                ?? jsFrontWallWidth
                ?? savedRoomWidth
                ?? 4.0
        }
        return calibratedRoomWidth
            ?? savedRoomWidth
            ?? jsFrontWallWidth
            ?? 4.0
    }

    private var displayRoomHeight: Float {
        if allowSave {
            return calibratedRoomHeight
                ?? jsFrontWallHeight
                ?? savedRoomHeight
                ?? 3.0
        }
        return calibratedRoomHeight
            ?? savedRoomHeight
            ?? jsFrontWallHeight
            ?? 3.0
    }

    /// Depth (m) for save metadata: saved `.meta` when reopening from Home, else PLY bounds span Z.
    private var displayRoomDepth: Float {
        savedRoomModel?.roomDepth ?? effectiveBounds?.depth ?? 4.0
    }

    private var navigationTitleWithDimensions: String {
        String(format: "%.1f × %.1f m", displayRoomWidth, displayRoomHeight)
    }

    /// Baseline W/H before calibration — mirrors display priority minus `calibrated*`.
    private var sourceRoomWidth: Float {
        if allowSave {
            return jsFrontWallWidth ?? savedRoomWidth ?? 4.0
        }
        return savedRoomWidth ?? jsFrontWallWidth ?? 4.0
    }

    private var sourceRoomHeight: Float {
        if allowSave {
            return jsFrontWallHeight ?? savedRoomHeight ?? 3.0
        }
        return savedRoomHeight ?? jsFrontWallHeight ?? 3.0
    }

    /// Room dimensions for FurnitureFit — same chain as nav title / save.
    private var furnitureFitRoomWidth: Float { displayRoomWidth }

    private var furnitureFitRoomHeight: Float { displayRoomHeight }

    private var furnitureFitOverlayView: some View {
        FurnitureFitUIView(
            capturedImage: .constant(nil),
            roomImage: nil,
            mlModel: yoloeService.model,
            processInterval: 0.07,
            active: true,
            lockedOrientation: photoOrientation,
            roomWidthMeters: furnitureFitRoomWidth,
            roomHeightMeters: furnitureFitRoomHeight,
            onFurnitureSizeEstimated: { estimate in
                // Keep numeric room + furniture measurements stable; AR height is used only
                // for visual overlay sizing inside FurnitureFit, not for recalibrating the room.
                detectedFurnitureWidth = estimate.widthMeters
                detectedFurnitureHeight = estimate.heightMeters
                detectedFurnitureHeightAR = estimate.arHeightMeters
            },
            suppressStartupProgress: furnitureFitInitialSegmentationDone,
            onFirstSegmentationComplete: { furnitureFitInitialSegmentationDone = true }
        )
        .ignoresSafeArea()
        .zIndex(100)
    }

    private var calibrationOverlayView: some View {
        ZStack {
            Color.black.opacity(0.6)
                .ignoresSafeArea()
                .onTapGesture { showFurnitureDimensionsInput = false }
            VStack(spacing: 16) {
                Text(L10n.RoomViewer.calibrateRoomTitle).font(.headline).foregroundColor(.white)
                Text(L10n.RoomViewer.enterFurnitureHeightMeters).font(.caption).foregroundColor(.gray)
                Text(L10n.RoomViewer.furnitureFullHeightHint).font(.caption2).foregroundColor(.gray.opacity(0.9))
                if let h = detectedFurnitureHeight {
                    Text(L10n.RoomViewer.detectedMeters(h)).font(.caption2).foregroundColor(.orange)
                }
                Text(inputFurnitureHeight.isEmpty ? "0.00" : inputFurnitureHeight)
                    .font(.system(size: 32, weight: .bold, design: .monospaced))
                    .foregroundColor(.white)
                    .frame(width: 120, height: 44)
                    .background(Color.white.opacity(0.1))
                    .cornerRadius(8)
                calibrationNumberPadView
                HStack(spacing: 16) {
                    Button(L10n.Common.cancel) {
                        inputFurnitureHeight = ""
                        showFurnitureDimensionsInput = false
                    }
                    .font(.body.bold()).foregroundColor(.red)
                    .frame(width: 80, height: 40).background(Color.red.opacity(0.2)).cornerRadius(8)
                    Button(L10n.Common.apply) { applyCalibration() }
                    .font(.body.bold()).foregroundColor(.green)
                    .frame(width: 80, height: 40).background(Color.green.opacity(0.2)).cornerRadius(8)
                    .disabled(Float(inputFurnitureHeight) == nil || inputFurnitureHeight.isEmpty)
                }
            }
            .padding(20)
            .background(Color.black.opacity(0.95))
            .cornerRadius(16)
        }
        .zIndex(99999)
    }

    private var calibrationNumberPadView: some View {
        VStack(spacing: 8) {
            ForEach(0..<3, id: \.self) { row in
                HStack(spacing: 8) {
                    ForEach(1...3, id: \.self) { col in
                        let digit = row * 3 + col
                        Button(action: { appendDigit("\(digit)") }) {
                            Text("\(digit)")
                                .font(.title2.bold()).foregroundColor(.white)
                                .frame(width: 50, height: 44)
                                .background(Color.gray.opacity(0.3)).cornerRadius(8)
                        }
                    }
                }
            }
            HStack(spacing: 8) {
                Button(action: {
                    if !inputFurnitureHeight.contains(".") {
                        inputFurnitureHeight += inputFurnitureHeight.isEmpty ? "0." : "."
                    }
                }) {
                    Text(".").font(.title2.bold()).foregroundColor(.white)
                        .frame(width: 50, height: 44).background(Color.gray.opacity(0.3)).cornerRadius(8)
                }
                Button(action: { appendDigit("0") }) {
                    Text("0").font(.title2.bold()).foregroundColor(.white)
                        .frame(width: 50, height: 44).background(Color.gray.opacity(0.3)).cornerRadius(8)
                }
                Button(action: { if !inputFurnitureHeight.isEmpty { inputFurnitureHeight.removeLast() } }) {
                    Image(systemName: "delete.left").font(.title3).foregroundColor(.white)
                        .frame(width: 50, height: 44).background(Color.gray.opacity(0.3)).cornerRadius(8)
                }
            }
        }
    }

    private func applyCalibration() {
        guard let realHeight = Float(inputFurnitureHeight),
              let detectedHeight = detectedFurnitureHeight,
              detectedHeight > 0 else {
            inputFurnitureHeight = ""
            showFurnitureDimensionsInput = false
            return
        }
        let scaleFactor = realHeight / detectedHeight
        let roomH = sourceRoomHeight
        let roomW = sourceRoomWidth
        realFurnitureHeight = realHeight
        NotificationCenter.default.post(
            name: NSNotification.Name("WebGLScaleRoom"),
            object: nil,
            userInfo: ["factor": Double(scaleFactor)]
        )
        if savedRoomHeight != nil || jsFrontWallHeight != nil {
            calibratedRoomHeight = roomH * scaleFactor
        }
        if savedRoomWidth != nil || jsFrontWallWidth != nil {
            calibratedRoomWidth = roomW * scaleFactor
        }
        logDebug("📐 [Calibration] Real height: \(realHeight)m, Scale factor: \(scaleFactor)")
        inputFurnitureHeight = ""
        showFurnitureDimensionsInput = false
    }

    /// Auto-apply AR-based calibration once per session: use AR height as the \"real\" value and
    /// the room-based estimate as the detected value to compute a global scale factor.
    private func applyArAutoCalibrationIfNeeded(realHeight: Float, detectedHeight: Float) {
        guard !didApplyArAutoCalibration else { return }
        guard realHeight > 0, detectedHeight > 0 else { return }

        let roomH = sourceRoomHeight
        let roomW = sourceRoomWidth
        let heightFromSource = savedRoomHeight != nil || jsFrontWallHeight != nil
        let widthFromSource = savedRoomWidth != nil || jsFrontWallWidth != nil
        // If we run before WebGL/SHARP has reported both walls, we'd scale against 4×3 defaults and
        // the displayed room would still change when real bounds arrive — wait for real geometry.
        guard heightFromSource, widthFromSource else { return }

        didApplyArAutoCalibration = true
        let scaleFactor = realHeight / detectedHeight
        realFurnitureHeight = realHeight

        NotificationCenter.default.post(
            name: NSNotification.Name("WebGLScaleRoom"),
            object: nil,
            userInfo: ["factor": Double(scaleFactor)]
        )

        if savedRoomHeight != nil || jsFrontWallHeight != nil {
            calibratedRoomHeight = roomH * scaleFactor
        }
        if savedRoomWidth != nil || jsFrontWallWidth != nil {
            calibratedRoomWidth = roomW * scaleFactor
        }
        logDebug("📐 [AR Calibration] Auto real height: \(realHeight)m, detected=\(detectedHeight)m, scale factor: \(scaleFactor)")
    }

    private func applyWallCalibration() {
        guard let realW = Float(inputWallWidth), realW > 0,
              let realH = Float(inputWallHeight), realH > 0 else {
            return
        }
        let roomW = sourceRoomWidth
        let roomH = sourceRoomHeight
        let scaleX = Double(realW / roomW)
        let scaleY = Double(realH / roomH)
        calibratedRoomWidth = realW
        calibratedRoomHeight = realH
        NotificationCenter.default.post(
            name: NSNotification.Name("WebGLScaleRoom"),
            object: nil,
            userInfo: ["scaleX": scaleX, "scaleY": scaleY]
        )
        logDebug("📐 [Wall calibration] Real wall: \(realW)×\(realH)m, scale XY: \(scaleX), \(scaleY)")
        inputWallWidth = ""
        inputWallHeight = ""
        showWallCalibration = false
    }

    private var wallCalibrationOverlay: some View {
        ZStack {
            Color.black.opacity(0.6)
                .ignoresSafeArea()
                .onTapGesture { showWallCalibration = false }
            VStack(spacing: 16) {
                Text(L10n.RoomViewer.calibrateByWallTitle)
                    .font(.headline)
                    .foregroundColor(.white)
                Text(L10n.RoomViewer.enterWallDimensionsHint)
                    .font(.caption)
                    .foregroundColor(.gray)
                HStack(spacing: 12) {
                    TextField("Width", text: $inputWallWidth)
                        .keyboardType(.decimalPad)
                        .textFieldStyle(.roundedBorder)
                        .frame(width: 80)
                    Text("×")
                    TextField("Height", text: $inputWallHeight)
                        .keyboardType(.decimalPad)
                        .textFieldStyle(.roundedBorder)
                        .frame(width: 80)
                }
                HStack(spacing: 16) {
                    Button(L10n.Common.cancel) {
                        inputWallWidth = ""
                        inputWallHeight = ""
                        showWallCalibration = false
                    }
                    .foregroundColor(.red)
                    Button(L10n.Common.apply) { applyWallCalibration() }
                        .foregroundColor(.green)
                        .disabled((Float(inputWallWidth) ?? 0) <= 0 || (Float(inputWallHeight) ?? 0) <= 0)
                }
            }
            .padding(24)
            .background(Color.black.opacity(0.9))
            .cornerRadius(16)
        }
        .zIndex(99999)
    }

    /// Furn / Room lines; optional “Tap to calibrate” hint only when [showTapHint] is true.
    private func furnitureMeasurementPillContent(showTapHint: Bool) -> some View {
        let displayH = detectedFurnitureHeightAR ?? detectedFurnitureHeight ?? 0
        return VStack(spacing: 2) {
            if let calibH = calibratedRoomHeight {
                Text(L10n.RoomViewer.roomMetersShort(calibH)).font(.caption2).foregroundColor(.green)
            }
            Text(L10n.RoomViewer.furnitureMetersShort(realFurnitureHeight ?? displayH))
                .font(.caption.bold())
                .foregroundColor(realFurnitureHeight != nil ? .green : .white)
            if showTapHint {
                Text(L10n.RoomViewer.tapToCalibrate).font(.system(size: 9)).foregroundColor(.gray)
            }
        }
        .padding(.horizontal, 8).padding(.vertical, 4)
        .background(Color.black.opacity(0.6)).cornerRadius(6)
    }

    @ViewBuilder private var bottomBarsOverlayView: some View {
        if photoOrientation == .landscape {
            ZStack(alignment: .bottom) {
                Color.clear
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .allowsHitTesting(false)
                HStack(spacing: 20) {
                    Button(action: {
                        if showingFurnitureFit {
                            showingFurnitureFit = false
                        } else {
                            furnitureFitInitialSegmentationDone = false
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
                    HStack(spacing: 6) {
                        Image(systemName: "iphone.landscape").font(.caption)
                        Text(NSLocalizedString("orientation.heldHorizontally", comment: "")).font(.caption2)
                        Text("-").font(.caption2)
                        Text(NSLocalizedString("orientation.landscape", comment: "")).font(.caption2).fontWeight(.medium)
                    }
                    .foregroundColor(.white.opacity(0.9))
                    .padding(.horizontal, 12).padding(.vertical, 8)
                    .background(Color.black.opacity(0.5)).cornerRadius(8)
                    .allowsHitTesting(false)
                    if showingFurnitureFit {
                        if showRoomFurnitureCalibrate {
                            Button(action: { showFurnitureDimensionsInput = true }) {
                                furnitureMeasurementPillContent(showTapHint: true)
                            }
                        } else {
                            furnitureMeasurementPillContent(showTapHint: false)
                        }
                    }
                    Spacer().allowsHitTesting(false)
                    Button(action: { takeScreenshot() }) {
                        Image(systemName: "camera.fill")
                            .font(.system(size: 28)).foregroundColor(.white)
                            .frame(width: 60, height: 60)
                            .background(Circle().fill(Color.blue).shadow(radius: 5))
                    }
                    .disabled(isLoading)
                }
                .padding(.horizontal, 30).padding(.bottom, 20)
            }
            .opacity(isCapturingSnapshot ? 0 : 1)
            .zIndex(99997)
        } else {
            VStack {
                Spacer().allowsHitTesting(false)
                VStack(spacing: 1) {
                    Text(NSLocalizedString("orientation.heldVertically", comment: "")).font(.caption2)
                    Text(NSLocalizedString("orientation.portrait", comment: "")).font(.caption2).fontWeight(.medium)
                }
                .foregroundColor(.white.opacity(0.8))
                .padding(.horizontal, 8).padding(.vertical, 4)
                .background(Color.black.opacity(0.4)).cornerRadius(6)
                .padding(.bottom, 12)
                .allowsHitTesting(false)
                HStack {
                    Button(action: {
                        if showingFurnitureFit {
                            showingFurnitureFit = false
                        } else {
                            furnitureFitInitialSegmentationDone = false
                            SHARPService.shared.releaseResources()
                            showingFurnitureFit = true
                        }
                    }) {
                        Image(systemName: "brain.head.profile")
                            .font(.system(size: 28)).foregroundColor(.white)
                            .frame(width: 60, height: 60)
                            .background(Circle().fill(showingFurnitureFit ? Color.green : Color.blue).shadow(radius: 5))
                    }
                    .disabled(isLoading)
                    .padding(.leading, 16)
                    Spacer().allowsHitTesting(false)
                    VStack(spacing: 8) {
                        if showingFurnitureFit {
                            if showRoomFurnitureCalibrate {
                                Button(action: { showFurnitureDimensionsInput = true }) {
                                    furnitureMeasurementPillContent(showTapHint: true)
                                }
                            } else {
                                furnitureMeasurementPillContent(showTapHint: false)
                            }
                        }
                        Button(action: { takeScreenshot() }) {
                            Image(systemName: "camera.fill")
                                .font(.system(size: 28)).foregroundColor(.white)
                                .frame(width: 60, height: 60)
                                .background(Circle().fill(Color.blue).shadow(radius: 5))
                        }
                        .disabled(isLoading)
                    }
                    .padding(.trailing, 16)
                }
                .padding(.bottom, 20)
            }
            .opacity(isCapturingSnapshot ? 0 : 1)
            .zIndex(99998)
        }
    }

    @ViewBuilder private var allOverlaysLayer: some View {
        ZStack {
            if !isLoading { cameraButtonsOverlay }
            if isLoading { loadingOverlayView }
            errorOverlayView
            if showingFurnitureFit { furnitureFitOverlayView }
            if isSavingRoom { saveRoomProgressOverlay }
            if showFurnitureDimensionsInput, showRoomFurnitureCalibrate { calibrationOverlayView }
            if showWallCalibration, showRoomFurnitureCalibrate { wallCalibrationOverlay }
            bottomBarsOverlayView
        }
    }

    // MARK: - Number Pad Helper

    private func appendDigit(_ digit: String) {
        // Limit to reasonable length (e.g., "12.34")
        if inputFurnitureHeight.count >= 5 { return }
        // Limit decimal places to 2
        if let dotIndex = inputFurnitureHeight.firstIndex(of: ".") {
            let decimals = inputFurnitureHeight.distance(from: dotIndex, to: inputFurnitureHeight.endIndex) - 1
            if decimals >= 2 { return }
        }
        inputFurnitureHeight += digit
    }

    // MARK: - Screenshot

    private func takeScreenshot() {
        logDebug("📸 Taking screenshot...")
        isCapturingSnapshot = true
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.08) {
            let scenes = UIApplication.shared.connectedScenes.compactMap { $0 as? UIWindowScene }
            let windows = scenes.flatMap { $0.windows }
            guard let window = windows.first(where: { $0.isKeyWindow }) ?? windows.first else {
                isCapturingSnapshot = false
                logDebug("❌ No window found")
                return
            }
            // Prefer capturing just the splat renderer (Metal view) if present. Fallback to full window.
            let targetView: UIView = findFirstMTKView(in: window) ?? window
            let format = UIGraphicsImageRendererFormat()
            format.scale = targetView.traitCollection.displayScale
            let renderer = UIGraphicsImageRenderer(bounds: targetView.bounds, format: format)
            let image = renderer.image { _ in
                targetView.drawHierarchy(in: targetView.bounds, afterScreenUpdates: true)
            }
            logDebug("📸 Screenshot captured, saving to Photos...")
            UIImageWriteToSavedPhotosAlbum(image, nil, nil, nil)
            logDebug("✅ Screenshot saved to Photos")
            DispatchQueue.main.async {
                isCapturingSnapshot = false
            }
        }
    }

    /// Depth-first search for the first `MTKView` in the given view hierarchy.
    private func findFirstMTKView(in root: UIView) -> MTKView? {
        if let mtk = root as? MTKView {
            return mtk
        }
        for sub in root.subviews {
            if let found = findFirstMTKView(in: sub) {
                return found
            }
        }
        return nil
    }

    // MARK: - YOLOE model loaded via YOLOEModelService (ODR)

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

                Text(L10n.RoomViewer.savingRoomEllipsis)
                    .font(.title2)
                    .fontWeight(.semibold)
                    .foregroundColor(.white)

                ProgressView(value: saveProgress)
                    .progressViewStyle(LinearProgressViewStyle(tint: .green))
                    .frame(width: 200)

                Text("\(Int(saveProgress * 100))%")
                    .font(.headline)
                    .foregroundColor(.gray)

                Button(L10n.Common.cancel) {
                    cancelSavingRoom()
                }
                .foregroundColor(.red)
                .padding(.top, 20)
            }
        }
    }

    // MARK: - Save Room Functions

    /// Resolves `Room_<stamp>_thumbnail.png` next to the classic PLY, or legacy `thumbnail.png` (Android-style).
    private func resolvedThumbnailURL(forClassicPly classicPly: URL) -> URL {
        let folder = classicPly.deletingLastPathComponent()
        let stem = classicPly.deletingPathExtension().lastPathComponent
        let base: String
        if stem.hasSuffix("_classic") {
            base = String(stem.dropLast("_classic".count))
        } else {
            base = stem
        }
        let named = folder.appendingPathComponent("\(base)_thumbnail.png")
        let legacy = folder.appendingPathComponent("thumbnail.png")
        if FileManager.default.fileExists(atPath: named.path) { return named }
        if FileManager.default.fileExists(atPath: legacy.path) { return legacy }
        return named
    }

    private func startSavingRoom() {
        guard !roomName.isEmpty else { return }

        let savedName = roomName
        logDebug("💾 [SharpRoomView] Starting room save: \(savedName)")

        withAnimation(.easeIn(duration: 0.3)) {
            isSavingRoom = true
            saveProgress = 0.0
        }

        Task {
            await MainActor.run { saveProgress = 0.12 }
            let folder = classicPlyURL.deletingLastPathComponent()
            let thumbURL = resolvedThumbnailURL(forClassicPly: classicPlyURL)
            let thumbExists = FileManager.default.fileExists(atPath: thumbURL.path)
            let thumbImage = UIImage(contentsOfFile: thumbURL.path)
            logWallMeasurement(
                "saveRoom start name=\(savedName) folder=\(folder.path) thumb=\(thumbURL.lastPathComponent) exists=\(thumbExists) decoded=\(thumbImage != nil)",
            )
            await MainActor.run {
                YOLOEModelService.shared.ensureModelLoaded()
            }
            var yoloModel: MLModel?
            for _ in 0..<90 {
                yoloModel = await MainActor.run { YOLOEModelService.shared.model }
                if yoloModel != nil { break }
                try? await Task.sleep(nanoseconds: 100_000_000)
            }
            if yoloModel == nil {
                logWallMeasurement("saveRoom YOLO CoreML model still nil after wait — wall measure skipped")
            }
            var roomW = await MainActor.run { displayRoomWidth }
            var roomH = await MainActor.run { displayRoomHeight }
            var roomD = await MainActor.run { displayRoomDepth }
            logWallMeasurement("saveRoom before measure display W×H×D=\(roomW)×\(roomH)×\(roomD)")
            if thumbImage == nil || yoloModel == nil {
                logWallMeasurement("saveRoom wall measure skipped (need thumbnail + YOLO model)")
            } else if let m = yoloModel, let img = thumbImage {
                if let measured = await WallMeasurementEstimator.measure(roomFolder: folder, thumbnail: img, model: m) {
                    roomW = measured.widthMeters
                    roomH = measured.heightMeters
                    roomD = measured.depthMeters
                    logWallMeasurement("saveRoom applied W×H×D=\(roomW)×\(roomH)×\(roomD) mode=\(measured.calibrationMode)")
                    logDebug("📐 [SharpRoomView] YOLO wall measurement: \(roomW)×\(roomH)×\(roomD) m (\(measured.calibrationMode))")
                    await MainActor.run {
                        jsFrontWallWidth = roomW
                        jsFrontWallHeight = roomH
                    }
                } else {
                    logWallMeasurement("saveRoom measure returned nil — keeping display W×H×D=\(roomW)×\(roomH)×\(roomD)")
                }
            }
            await MainActor.run { saveProgress = 0.55 }
            await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
                modelManager.savePLY(
                    from: viewerPlyURL,
                    name: savedName,
                    photoOrientation: photoOrientation,
                    roomWidth: roomW,
                    roomHeight: roomH,
                    roomDepth: roomD
                ) { success, error in
                    logDebug(success ? "✅ [SharpRoomView] Room saved" : "❌ [SharpRoomView] Save failed: \(error ?? "unknown")")
                    Task { @MainActor in
                        saveProgress = 1.0
                        withAnimation(.easeOut(duration: 0.3)) {
                            isSavingRoom = false
                        }
                        DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
                            if success {
                                saveAlertMessage = L10n.RoomViewer.saveSuccess(savedName)
                                saveWasSuccessful = true
                            } else {
                                saveAlertMessage = L10n.RoomViewer.saveFailed(error ?? L10n.RoomViewer.saveErrorUnknown)
                                saveWasSuccessful = false
                            }
                            showSaveAlert = true
                            roomName = ""
                        }
                        continuation.resume()
                    }
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

// MARK: - Antimatter15 WebGL Splat View (removed)

/// Legacy WebGL viewer (Antimatter/SparkJS) has been removed in favor of native MetalSplatter (`GaussianSplatView`).
/// The types that used to live here (`LocalFileSchemeHandler`, `AntimatterSplatView`) have been deleted to avoid
/// duplicate rendering paths, large inline HTML, and heavy synchronous PLY reads.
///
/// If you ever need the WebGL path again, prefer rebuilding it in a dedicated file that:
/// - streams PLY data from disk
/// - loads HTML from a bundled resource
/// - uses weak script message handlers to avoid retain cycles.
/*
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

struct AntimatterSplatView: UIViewRepresentable {
    let plyURL: URL
    let roomBounds: SIMD3<Float>?  // Room dimensions for camera positioning
    let actualBounds: RoomBounds?  // Actual min/max bounds for precise framing
    let photoOrientation: PhotoOrientation  // For orientation-based room rotation
    let oscillationEnabled: Bool  // Auto-orbit setting from user preferences
    let infiniteZoom: Bool  // If true, no zoom/distance limits; can pass through walls
    let onLoaded: () -> Void
    var onFrontWallDimensions: ((Double, Double) -> Void)? = nil

    func makeCoordinator() -> Coordinator {
        Coordinator(onLoaded: onLoaded, onFrontWallDimensions: onFrontWallDimensions)
    }

    func makeUIView(context: Context) -> WKWebView {
        // Tight autorelease scope helps return transient Obj-C allocations before WebKit fully warms up.
        autoreleasepool {
            let config = WKWebViewConfiguration()
            config.allowsInlineMediaPlayback = true

            // Enable JavaScript
            let prefs = WKWebpagePreferences()
            prefs.allowsContentJavaScript = true
            config.defaultWebpagePreferences = prefs

            // Register custom URL scheme handler
            let schemeHandler = LocalFileSchemeHandler()
            schemeHandler.plyURL = plyURL
            schemeHandler.htmlContent = generateSplatViewerHTML(bounds: roomBounds, actualBounds: actualBounds, orientation: self.photoOrientation, oscillation: oscillationEnabled, infiniteZoom: infiniteZoom)
            config.setURLSchemeHandler(schemeHandler, forURLScheme: "splat")

            // Add message handlers for communication from JS
            config.userContentController.add(context.coordinator, name: "splatLoaded")
            config.userContentController.add(context.coordinator, name: "frontWallDimensions")
            config.userContentController.add(context.coordinator, name: "cameraPose")
            config.userContentController.add(context.coordinator, name: "jsLog")
            config.userContentController.add(context.coordinator, name: "box3Size")

            let webView = WKWebView(frame: .zero, configuration: config)
            webView.navigationDelegate = context.coordinator
            webView.scrollView.isScrollEnabled = false
            webView.scrollView.bounces = false
            webView.scrollView.bouncesZoom = false
            webView.scrollView.minimumZoomScale = 1.0
            webView.scrollView.maximumZoomScale = 1.0
            webView.scrollView.contentInsetAdjustmentBehavior = .never

            // Disable all scroll view gesture recognizers to let Three.js handle touches
            for gestureRecognizer in webView.scrollView.gestureRecognizers ?? [] {
                gestureRecognizer.isEnabled = false
            }

            webView.isOpaque = false
            webView.backgroundColor = .gray
            webView.isUserInteractionEnabled = true
            webView.isMultipleTouchEnabled = true

            // Store reference
            context.coordinator.schemeHandler = schemeHandler
            context.coordinator.webView = webView
            context.coordinator.lastInfiniteZoom = infiniteZoom

            // Load from custom scheme (HTML will auto-load room.ply)
            if let url = URL(string: "splat://local/") {
                webView.load(URLRequest(url: url))
                logDebug("AntimatterSplatView: Loading from custom scheme")
            }

            return webView
        }
    }

    func updateUIView(_ uiView: WKWebView, context: Context) {
        // Reload viewer when Infinite Zoom changes so the setting takes effect without leaving the room
        if context.coordinator.lastInfiniteZoom != infiniteZoom {
            context.coordinator.lastInfiniteZoom = infiniteZoom
            context.coordinator.schemeHandler?.htmlContent = generateSplatViewerHTML(bounds: roomBounds, actualBounds: actualBounds, orientation: photoOrientation, oscillation: oscillationEnabled, infiniteZoom: infiniteZoom)
            uiView.reload()
        }
    }

    private func generateSplatViewerHTML(bounds: SIMD3<Float>?, actualBounds: RoomBounds?, orientation: PhotoOrientation, oscillation: Bool, infiniteZoom: Bool) -> String {
        // Camera and framing come from Box3 in the viewer only. No Swift bounds injection.
        logDebug("📐 [WebGL] Framing from Box3 only (no Swift bounds)")

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

                // Oscillation setting from Swift
                const OSCILLATION_ENABLED = \(oscillation ? "true" : "false");
                console.log('[WebGL] oscillation =', OSCILLATION_ENABLED);

                // Infinite zoom: no distance/room bounds clamping (can pass through walls into void)
                const INFINITE_ZOOM = \(infiniteZoom ? "true" : "false");
                console.log('[WebGL] infiniteZoom =', INFINITE_ZOOM);

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

                // Camera - start facing origin, autoFrameRoom will reposition using Box3. Infinite zoom: smaller near for close-up.
                const cameraNear = INFINITE_ZOOM ? 0.001 : 0.1;
                const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, cameraNear, 1000);
                camera.position.set(0, 0, 3);  // Closer initial position
                camera.lookAt(0, 0, 0);        // Explicitly look at origin
                camera.up.set(0, 1, 0);        // Normal Y-up

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
                controls.dampingFactor = 0.08;  // Smooth deceleration
                controls.rotateSpeed = 1.5;     // Faster rotation
                controls.zoomSpeed = 2.0;       // Faster zoom
                controls.screenSpacePanning = false;
                // Portrait: allow zoom into wall (0.001); landscape 0.01 (match Android). Infinite zoom: 0 = no minimum, can pass through target.
                controls.minDistance = INFINITE_ZOOM ? 0 : (isPortrait ? 0.001 : 0.01);
                controls.maxDistance = INFINITE_ZOOM ? 1e6 : 100;
                // Default target, autoFrameRoom will update using Box3
                controls.target.set(0, 0, 0);

                // No full 360° spin - we'll do back-and-forth oscillation instead
                controls.autoRotate = false;

                // Request render when controls change (handles damping animation too)
                controls.addEventListener('change', function() {
                    needsRender = true;
                });

                // --- Back-and-forth orbit state ---
                let autoOrbitEnabled = OSCILLATION_ENABLED;  // Read from Swift setting
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

                // Toggle auto-orbit from Swift
                window.setAutoOrbit = function(enabled) {
                    autoOrbitEnabled = enabled;
                    console.log('Auto-orbit:', enabled ? 'enabled' : 'disabled');
                    needsRender = true;
                };

                // Let OrbitControls also toggle interaction
                controls.addEventListener('start', () => { window._userInteracting = true; });
                controls.addEventListener('end',   () => { window._userInteracting = false; });

                // Unlimited spin around object (train-style orbit)
                controls.minAzimuthAngle = -Infinity;
                controls.maxAzimuthAngle =  Infinity;
                // Infinite zoom: full polar range — tight min/max polar still clamps spherical phi when dollying through
                // the target (pinch stops at the wall). Bounded zoom: keep a narrow cone.
                if (INFINITE_ZOOM) {
                    controls.minPolarAngle = 0;
                    controls.maxPolarAngle = Math.PI;
                } else {
                    controls.minPolarAngle = 0.001;
                    controls.maxPolarAngle = Math.PI - 0.001;
                }

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

                // Recenter = re-frame room (always works; fixes zoom and works after segment/apply)
                window.recenterCamera = function() {
                    if (typeof autoFrameRoom === 'function') {
                        autoFrameRoom(true);
                    } else {
                        camera.position.copy(initialCameraPosition);
                        controls.target.copy(initialControlsTarget);
                        var prevDamping = controls.enableDamping;
                        var prevFactor = controls.dampingFactor;
                        controls.enableDamping = false;
                        controls.update();
                        controls.enableDamping = prevDamping;
                        controls.dampingFactor = prevFactor;
                    }
                    needsRender = true;
                    setTimeout(function() { needsRender = true; }, 0);
                    setTimeout(function() { needsRender = true; }, 50);
                };

                // Scale room in place — camera stays where it is (no re-frame)
                window.scaleRoom = function(factor) {
                    if (splatMesh) {
                        splatMesh.scale.set(factor, factor, factor);
                        console.log('Room scaled by factor:', factor);
                        needsRender = true;
                        setTimeout(function() {
                            if (typeof window.updateInitialPosition === 'function') window.updateInitialPosition();
                        }, 100);
                    }
                };
                // Non-uniform scale for wall-based calibration; camera stays where it is
                window.scaleRoomXY = function(scaleX, scaleY) {
                    if (splatMesh) {
                        var scaleZ = (scaleX + scaleY) / 2;
                        splatMesh.scale.set(scaleX, scaleY, scaleZ);
                        console.log('Room scaled by XY:', scaleX, scaleY, scaleZ);
                        needsRender = true;
                        setTimeout(function() {
                            if (typeof window.updateInitialPosition === 'function') window.updateInitialPosition();
                        }, 100);
                    }
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

                    // Classic SHARP PLY: single 180° flip around X aligns splat with camera (SparkJS).
                    // Do not add portrait-only Z rotation — it tilts portrait captures ~90° vs device upright.
                    splatMesh.rotation.x = Math.PI;
                    splatMesh.rotation.z = 0;
                    console.log('SplatMesh: rotated 180° X (portrait/landscape)');

                    let autoFrameRoomRetryCount = 0;
                    const AUTO_FRAME_MAX_RETRIES = 60;

                    function autoFrameRoom(fromRecenter) {
                        if (fromRecenter) autoFrameRoomRetryCount = 0;
                        const jsLog = function(msg) { if (window.webkit?.messageHandlers?.jsLog) window.webkit.messageHandlers.jsLog.postMessage(msg); };
                        try {
                            if (!splatMesh) {
                                autoFrameRoomRetryCount++;
                                if (autoFrameRoomRetryCount <= 3) {
                                    jsLog('No splatMesh yet, retry ' + autoFrameRoomRetryCount);
                                    setTimeout(autoFrameRoom, 300);
                                } else {
                                    jsLog('Giving up: no splatMesh after ' + autoFrameRoomRetryCount + ' retries — using default camera');
                                    camera.position.set(0, 0, 4);
                                    controls.target.set(0, 0, 0);
                                    controls.update();
                                    needsRender = true;
                                }
                                return;
                            }
                            // SplatMesh is not a standard Three.js Mesh — setFromObject() sees no geometry and returns (0,0,0).
                            // Use SparkJS getBoundingBox() which uses actual splat positions. Must wait until initialized.
                            if (!splatMesh.isInitialized) {
                                autoFrameRoomRetryCount++;
                                if (autoFrameRoomRetryCount <= 3 || autoFrameRoomRetryCount % 15 === 0) {
                                    jsLog('SplatMesh not initialized yet, retry ' + autoFrameRoomRetryCount);
                                }
                                if (autoFrameRoomRetryCount < AUTO_FRAME_MAX_RETRIES) {
                                    setTimeout(autoFrameRoom, 200);
                                } else {
                                    jsLog('Giving up: SplatMesh never initialized — using default camera');
                                    camera.position.set(0, 0, 4);
                                    controls.target.set(0, 0, 0);
                                    controls.update();
                                    needsRender = true;
                                }
                                return;
                            }
                            splatMesh.updateMatrixWorld(true);
                            let box;
                            try {
                                const localBox = splatMesh.getBoundingBox(true);
                                box = localBox.clone().applyMatrix4(splatMesh.matrixWorld);
                            } catch (e) {
                                jsLog('getBoundingBox failed: ' + e.message);
                                autoFrameRoomRetryCount++;
                                if (autoFrameRoomRetryCount < AUTO_FRAME_MAX_RETRIES) {
                                    setTimeout(autoFrameRoom, 300);
                                } else {
                                    camera.position.set(0, 0, 4);
                                    controls.target.set(0, 0, 0);
                                    controls.update();
                                    needsRender = true;
                                }
                                return;
                            }
                            const size = box.getSize(new THREE.Vector3());
                            if (size.length() < 0.01) {
                                autoFrameRoomRetryCount++;
                                if (autoFrameRoomRetryCount < AUTO_FRAME_MAX_RETRIES) {
                                    jsLog('Box3 size still zero after getBoundingBox, retry ' + autoFrameRoomRetryCount);
                                    setTimeout(autoFrameRoom, 300);
                                } else {
                                    jsLog('Giving up: Box3 stayed zero — using default camera');
                                    camera.position.set(0, 0, 4);
                                    controls.target.set(0, 0, 0);
                                    controls.update();
                                    needsRender = true;
                                }
                                return;
                            }

                            autoFrameRoomRetryCount = 0;
                            const center = box.getCenter(new THREE.Vector3());
                            // With 180° X only, use same width/height axes for both orientations (matches landscape path).
                            const roomWidth  = size.y;
                            const roomHeight = size.x;
                            let roomDepth  = size.z;

                            const box3Msg = 'Box3 raw size: ' + roomWidth.toFixed(2) + ' x ' + roomHeight.toFixed(2) + ' (isPortrait: ' + isPortrait + ')';
                            console.log(box3Msg);
                            if (window.webkit?.messageHandlers?.jsLog) window.webkit.messageHandlers.jsLog.postMessage(box3Msg);
                            if (window.webkit?.messageHandlers?.box3Size) window.webkit.messageHandlers.box3Size.postMessage({ width: roomWidth, height: roomHeight });

                            // Cap to realistic room dimensions (fog makes bounds too large)
                            const maxRealisticWidth = isPortrait ? 5.0 : 8.0;
                            const maxRealisticHeight = isPortrait ? 3.5 : 3.2;
                            if (roomWidth > maxRealisticWidth) {
                                console.log('Capping width from', roomWidth.toFixed(2), 'to', maxRealisticWidth);
                                roomWidth = maxRealisticWidth;
                            }
                            if (roomHeight > maxRealisticHeight) {
                                console.log('Capping height from', roomHeight.toFixed(2), 'to', maxRealisticHeight);
                                roomHeight = maxRealisticHeight;
                            }

                            console.log('Box3 capped size:', roomWidth.toFixed(2), roomHeight.toFixed(2), roomDepth.toFixed(2));
                            console.log('Box3 center:', center.x.toFixed(2), center.y.toFixed(2), center.z.toFixed(2));

                            // 2) Raw bounds from Box3
                            let rawMinX = box.min.x;
                            let rawMaxX = box.max.x;
                            let rawMinY = box.min.y;
                            let rawMaxY = box.max.y;
                            let rawMinZ = box.min.z;
                            let rawMaxZ = box.max.z;

                            // 3) Shrink bounds to ignore foggy outer 15% on front/sides; allow camera right up to back wall
                            const fogFactor = 0.15;
                            const shrinkX = roomWidth  * fogFactor * 0.5;
                            const shrinkY = roomHeight * fogFactor * 0.5;
                            const shrinkZ = roomDepth  * fogFactor * 0.5;
                            const backWallInset = 0.02;

                            const minX = rawMinX + shrinkX;
                            const maxX = rawMaxX - shrinkX;
                            const minY = rawMinY + shrinkY;
                            const maxY = rawMaxY - shrinkY;
                            const minZ = rawMinZ + shrinkZ;
                            const maxZ = rawMaxZ - backWallInset;

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

                            // 5) Camera in front of FRONT WALL. Distance scales with room size so after user
                            // enters measurement and we scale the room, the view zooms out to show the scaled room.
                            const frontWallZ = maxZ;
                            const distInFront = Math.max(0.012, Math.min(roomDepth * 0.28, 1.2));
                            // Look straight at wall (eye level): target at same height as camera
                            const newCamPos = new THREE.Vector3(
                                innerCenterX,
                                innerCenterY,
                                frontWallZ + distInFront
                            );
                            const newTarget = new THREE.Vector3(
                                innerCenterX,
                                innerCenterY,
                                frontWallZ
                            );

                            const msg1 = '[StartPos] Camera in front of wall — distInFront=' + distInFront.toFixed(3) + ' (roomDepth=' + roomDepth.toFixed(2) + ') minZ=' + minZ.toFixed(3) + ' maxZ=' + maxZ.toFixed(3) + ' isPortrait=' + isPortrait;
                            const msg2 = '[StartPos] pos=(' + newCamPos.x.toFixed(3) + ',' + newCamPos.y.toFixed(3) + ',' + newCamPos.z.toFixed(3) + ') tgt=(' + newTarget.x.toFixed(3) + ',' + newTarget.y.toFixed(3) + ',' + newTarget.z.toFixed(3) + ')';
                            console.log(msg1);
                            console.log(msg2);
                            if (window.webkit?.messageHandlers?.jsLog) {
                                window.webkit.messageHandlers.jsLog.postMessage(msg1);
                                window.webkit.messageHandlers.jsLog.postMessage(msg2);
                            }

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

                            // Force render after camera positioning (critical when auto-orbit is off)
                            needsRender = true;

                            // Also schedule a few more renders to ensure scene is visible
                            setTimeout(() => { needsRender = true; }, 100);
                            setTimeout(() => { needsRender = true; }, 300);

                            // Send dimensions again after delays to ensure Swift receives them
                            // (in case first message was missed during loading)
                            function sendDimensionsToSwift() {
                                if (window.webkit?.messageHandlers?.frontWallDimensions) {
                                    window.webkit.messageHandlers.frontWallDimensions.postMessage({
                                        width: roomWidth,
                                        height: roomHeight
                                    });
                                    console.log('Sent dimensions to Swift:', roomWidth.toFixed(2), 'x', roomHeight.toFixed(2));
                                }
                            }
                            setTimeout(sendDimensionsToSwift, 500);
                            setTimeout(sendDimensionsToSwift, 1500);
                            setTimeout(sendDimensionsToSwift, 3000);

                            console.log('=== Camera positioned using Box3 bounds ===');
                        } catch (err) {
                            console.error('autoFrameRoom error:', err);
                        }
                    }
                    window.autoFrameRoom = autoFrameRoom;
                    console.log('Scheduling autoFrameRoom in 500ms...');
                    setTimeout(autoFrameRoom, 500);

                    // Force immediate first render to show something while splat loads
                    needsRender = true;

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

                    // Clamp to room bounds; tiny margin at back wall so user can get right in front of it
                    if (roomBoundsForClamping) {
                        const marginSide = 0.05;
                        const marginBack = 0.02;
                        newX = Math.max(roomBoundsForClamping.minX + marginSide,
                               Math.min(roomBoundsForClamping.maxX - marginSide, newX));
                        newZ = Math.max(roomBoundsForClamping.minZ + marginSide,
                               Math.min(roomBoundsForClamping.maxZ - marginBack, newZ));
                    }

                    // Apply clamped movement
                    const actualDx = newX - camera.position.x;
                    const actualDz = newZ - camera.position.z;

                    camera.position.x = newX;
                    camera.position.z = newZ;
                    controls.target.x += actualDx;
                    controls.target.z += actualDz;
                    controls.update();  // Sync OrbitControls internal state so next frame doesn't overwrite
                    needsRender = true;  // Request render after camera move
                };

                // Move camera (and target) up/down in world Y — called from Swift "camera up" button
                window.moveCameraUp = function(dy) {
                    autoOrbitEnabled = false;
                    if (typeof dy !== 'number' || !isFinite(dy)) return;
                    camera.position.y += dy;
                    controls.target.y += dy;
                    if (!INFINITE_ZOOM && roomBoundsForClamping) {
                        const m = 0.05;
                        camera.position.y = Math.max(roomBoundsForClamping.minY + m, Math.min(roomBoundsForClamping.maxY - m, camera.position.y));
                        controls.target.y = Math.max(roomBoundsForClamping.minY + m, Math.min(roomBoundsForClamping.maxY - m, controls.target.y));
                    }
                    controls.update();  // Sync OrbitControls so next frame doesn't overwrite
                    needsRender = true;
                };

                // Two-finger pan / pan pad: shift room in screen direction (camera and target move together)
                // Swift sends screen deltas: +X = right, +Y = down. Axes swapped so drag-up = room-up when view/camera is rotated.
                window.panCamera = function(deltaX, deltaY) {
                    if (typeof deltaX !== 'number' || typeof deltaY !== 'number' || !isFinite(deltaX) || !isFinite(deltaY)) return;
                    autoOrbitEnabled = false;
                    const panSpeed = 0.06;
                    const forward = new THREE.Vector3();
                    camera.getWorldDirection(forward);
                    const right = new THREE.Vector3().crossVectors(camera.up, forward).normalize();
                    const upNorm = camera.up.clone().normalize();
                    // Map screen dx -> world up, screen dy -> world right (fixes perpendicular pan when camera/view rotated)
                    const move = right.multiplyScalar(-deltaY * panSpeed).add(upNorm.multiplyScalar(deltaX * panSpeed));
                    camera.position.add(move);
                    controls.target.add(move);
                    if (!INFINITE_ZOOM && roomBoundsForClamping) {
                        const m = 0.05;
                        camera.position.x = Math.max(roomBoundsForClamping.minX + m, Math.min(roomBoundsForClamping.maxX - m, camera.position.x));
                        camera.position.y = Math.max(roomBoundsForClamping.minY + m, Math.min(roomBoundsForClamping.maxY - m, camera.position.y));
                        camera.position.z = Math.max(roomBoundsForClamping.minZ + m, Math.min(roomBoundsForClamping.maxZ - m, camera.position.z));
                        controls.target.x = Math.max(roomBoundsForClamping.minX + m, Math.min(roomBoundsForClamping.maxX - m, controls.target.x));
                        controls.target.y = Math.max(roomBoundsForClamping.minY + m, Math.min(roomBoundsForClamping.maxY - m, controls.target.y));
                        controls.target.z = Math.max(roomBoundsForClamping.minZ + m, Math.min(roomBoundsForClamping.maxZ - m, controls.target.z));
                    }
                    controls.update();
                    needsRender = true;
                };

                // Orbit rotation handler (for Swift gesture overlay)
                window.orbitCamera = function(deltaX, deltaY) {
                    // Stop auto orbit when user drags
                    autoOrbitEnabled = false;

                    // Rotate around target using spherical coordinates
                    const rotateSpeed = 0.012;  // Increased for faster response

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

                // Zoom handler (for Swift pinch gesture) — incremental scale from Swift
                // Pinch out (scale > 1) = zoom in (closer). Pinch in (scale < 1) = zoom out (further).
                window.zoomCamera = function(scale) {
                    autoOrbitEnabled = false;
                    if (typeof scale !== 'number' || scale <= 0 || !isFinite(scale)) return;
                    const zoomSensitivity = 4.0;

                    if (INFINITE_ZOOM) {
                        // Dolly: move camera and target along view direction so we can pass through curtain
                        const forward = new THREE.Vector3().subVectors(controls.target, camera.position).normalize();
                        const step = (scale - 1) * 0.22 * zoomSensitivity;  // scale>1 -> move forward, scale<1 -> move back
                        camera.position.addScaledVector(forward, step);
                        controls.target.addScaledVector(forward, step);
                    } else {
                        const amplifiedScale = 1 + (scale - 1) * zoomSensitivity;
                        const offset = new THREE.Vector3().subVectors(camera.position, controls.target);
                        offset.multiplyScalar(1 / amplifiedScale);
                        let dist = offset.length();
                        if (dist < 0.01) offset.setLength(0.01);
                        if (dist > 50) offset.setLength(50);
                        camera.position.copy(controls.target).add(offset);
                    }

                    if (!INFINITE_ZOOM && roomBoundsForClamping) {
                        const marginSide = 0.05;
                        const marginBack = 0.02;
                        const minX = roomBoundsForClamping.minX + marginSide;
                        const maxX = roomBoundsForClamping.maxX - marginSide;
                        const minY = roomBoundsForClamping.minY + marginSide;
                        const maxY = roomBoundsForClamping.maxY - marginSide;
                        const minZ = roomBoundsForClamping.minZ + marginSide;
                        const maxZInside = roomBoundsForClamping.maxZ - marginBack;
                        camera.position.x = Math.max(minX, Math.min(maxX, camera.position.x));
                        camera.position.y = Math.max(minY, Math.min(maxY, camera.position.y));
                        // If camera is in front of the front wall (z > maxZ), don't clamp z from above — otherwise
                        // zoom-out would snap the camera through the wall and we'd see the back of the wall.
                        if (camera.position.z <= roomBoundsForClamping.maxZ) {
                            camera.position.z = Math.max(minZ, Math.min(maxZInside, camera.position.z));
                        } else {
                            camera.position.z = Math.max(minZ, camera.position.z);
                        }
                    }
                    controls.update();
                    needsRender = true;
                };

                // Animation loop with battery optimization
                // - Warm-up period: render continuously for first 5 seconds (SparkJS needs this)
                // - After warm-up: only renders when needed
                // - Throttles to 30fps when idle auto-orbit is running
                let loadNotified = false;
                let lastRenderTime = 0;
                const IDLE_FPS = 30;     // Lower FPS for auto-orbit (saves battery)
                const IDLE_FRAME_TIME = 1000 / IDLE_FPS;
                const WARMUP_DURATION = 5000;  // 5 seconds warm-up for SparkJS to fully load
                const animationStartTime = performance.now();

                // Request render on next frame (called when something changes)
                window.requestRender = function() {
                    needsRender = true;
                };

                function animate(currentTime) {
                    requestAnimationFrame(animate);

                    const dt = clock.getDelta();
                    let shouldRender = needsRender;
                    needsRender = false;

                    // Warm-up period: always render for first 5 seconds
                    // SparkJS progressively loads splat data and needs continuous rendering
                    const elapsed = performance.now() - animationStartTime;
                    const inWarmup = elapsed < WARMUP_DURATION;
                    if (inWarmup) {
                        shouldRender = true;
                    }

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
                    // But always render during warm-up period
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
        var lastInfiniteZoom: Bool?
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

            // Listen for room scale (calibration from real furniture dimensions)
            NotificationCenter.default.addObserver(
                self,
                selector: #selector(handleScaleRoom(_:)),
                name: NSNotification.Name("WebGLScaleRoom"),
                object: nil
            )

            // Listen for camera move up/down (from overlay buttons)
            NotificationCenter.default.addObserver(
                self,
                selector: #selector(handleCameraMoveUp(_:)),
                name: NSNotification.Name("WebGLCameraMoveUp"),
                object: nil
            )
            NotificationCenter.default.addObserver(
                self,
                selector: #selector(handleCameraMoveDown(_:)),
                name: NSNotification.Name("WebGLCameraMoveDown"),
                object: nil
            )
            NotificationCenter.default.addObserver(
                self,
                selector: #selector(handleCameraMoveLeft(_:)),
                name: NSNotification.Name("WebGLCameraMoveLeft"),
                object: nil
            )
            NotificationCenter.default.addObserver(
                self,
                selector: #selector(handleCameraMoveRight(_:)),
                name: NSNotification.Name("WebGLCameraMoveRight"),
                object: nil
            )
        }

        deinit {
            joystickTimer?.invalidate()
            NotificationCenter.default.removeObserver(self)
        }

        @objc private func recenterCamera() {
            runRecenter(retryOnFailure: true)
        }

        private func runRecenter(retryOnFailure: Bool) {
            DispatchQueue.main.async { [weak self] in
                guard let self = self, let wv = self.webView else { return }
                logDebug("🎯 [WebGL] Recentering camera")
                let js = "if (typeof window.recenterCamera === 'function') { window.recenterCamera(); } else if (typeof window.autoFrameRoom === 'function') { window.autoFrameRoom(true); } else if (typeof recenterCamera === 'function') { recenterCamera(); }"
                wv.evaluateJavaScript(js) { [weak self] _, error in
                    if let error = error, retryOnFailure {
                        logDebug("❌ [WebGL] Recenter JS error: \(error) — retrying once")
                        DispatchQueue.main.asyncAfter(deadline: .now() + 0.35) { [weak self] in
                            self?.runRecenter(retryOnFailure: false)
                        }
                    } else if let error = error {
                        logDebug("❌ [WebGL] Recenter JS error: \(error)")
                    }
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

            // OrbitGestureView now sends incremental values directly
            let js = "if (typeof orbitCamera === 'function') orbitCamera(\(deltaX), \(deltaY));"
            webView?.evaluateJavaScript(js, completionHandler: nil)
        }

        @objc private func handleZoomGesture(_ notification: Notification) {
            guard let userInfo = notification.userInfo,
                  let scale = userInfo["scale"] as? CGFloat else { return }

            // OrbitGestureView sends incremental scale; zoom strength is controlled in JS (amplifiedScale multiplier)
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

        @objc private func handleCameraMoveUp(_ notification: Notification) {
            let step = 0.2
            let js = "if (typeof moveCameraUp === 'function') moveCameraUp(\(step));"
            webView?.evaluateJavaScript(js, completionHandler: nil)
        }

        @objc private func handleCameraMoveDown(_ notification: Notification) {
            let step = -0.2
            let js = "if (typeof moveCameraUp === 'function') moveCameraUp(\(step));"
            webView?.evaluateJavaScript(js, completionHandler: nil)
        }

        @objc private func handleCameraMoveLeft(_ notification: Notification) {
            let step: Float = -8
            let js = "if (typeof moveCamera === 'function') moveCamera(\(step), 0);"
            webView?.evaluateJavaScript(js, completionHandler: nil)
        }

        @objc private func handleCameraMoveRight(_ notification: Notification) {
            let step: Float = 8
            let js = "if (typeof moveCamera === 'function') moveCamera(\(step), 0);"
            webView?.evaluateJavaScript(js, completionHandler: nil)
        }

        @objc private func handleScaleRoom(_ notification: Notification) {
            guard let userInfo = notification.userInfo else { return }
            let scaleX: Double? = (userInfo["scaleX"] as? NSNumber)?.doubleValue ?? userInfo["scaleX"] as? Double
            let scaleY: Double? = (userInfo["scaleY"] as? NSNumber)?.doubleValue ?? userInfo["scaleY"] as? Double
            if let sx = scaleX, let sy = scaleY {
                DispatchQueue.main.async { [weak self] in
                    guard let self = self else { return }
                    let js = "if (typeof window.scaleRoomXY === 'function') { window.scaleRoomXY(\(sx), \(sy)); }"
                    logDebug("📐 [WebGL] scaleRoomXY(\(sx), \(sy))")
                    self.webView?.evaluateJavaScript(js) { _, error in
                        if let error = error { logDebug("❌ [WebGL] scaleRoomXY JS error: \(error)") }
                    }
                }
                return
            }
            let factor: Double
            if let d = userInfo["factor"] as? Double {
                factor = d
            } else if let n = userInfo["factor"] as? NSNumber {
                factor = n.doubleValue
            } else {
                return
            }
            DispatchQueue.main.async { [weak self] in
                guard let self = self else { return }
                let js = "if (typeof window.scaleRoom === 'function') { window.scaleRoom(\(factor)); } else if (typeof scaleRoom === 'function') { scaleRoom(\(factor)); }"
                logDebug("📐 [WebGL] scaleRoom(\(factor))")
                self.webView?.evaluateJavaScript(js) { _, error in
                    if let error = error {
                        logDebug("❌ [WebGL] scaleRoom JS error: \(error)")
                    }
                }
            }
        }

        func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
            let elapsed = (CFAbsoluteTimeGetCurrent() - loadStartTime) * 1000
            logDebug("⏱️ [WebGL] HTML page loaded: \(String(format: "%.0f", elapsed))ms")

            DispatchQueue.main.asyncAfter(deadline: .now() + 5.0) {
                if !self.hasNotifiedLoaded {
                    self.hasNotifiedLoaded = true
                    let total = (CFAbsoluteTimeGetCurrent() - self.loadStartTime) * 1000
                    logDebug("⏱️ [WebGL] Total load (timeout): \(String(format: "%.0f", total))ms")
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
            } else if message.name == "box3Size" {
                if let body = message.body as? [String: Any],
                   let w = body["width"] as? Double,
                   let h = body["height"] as? Double {
                    logDebug("📐 [WebGL] Box3 width: \(String(format: "%.2f", w)) height: \(String(format: "%.2f", h))")
                }
            } else if message.name == "jsLog", let text = message.body as? String {
                logDebug("📜 [JS] \(text)")
            }
        }
    }
}
*/

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

