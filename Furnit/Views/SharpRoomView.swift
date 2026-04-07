import SwiftUI
import UIKit
import CoreML
import Photos
import MetalKit
import simd

// MARK: - Modal pause sync (AR + TextField responsiveness)

/// Bundles alert/sheet flags so one `onChange` can pause ARKit without bloating the SwiftUI type checker.
private struct SharpRoomModalPauseToken: Equatable {
    var showRoomNameInput: Bool
    var isSavingRoom: Bool
    var isRunningDepthProCalibration: Bool
    var showSaveAlert: Bool
    var showDiscardUnsavedAlert: Bool
    var showCalibrationRejectAlert: Bool
    var showWallCalibration: Bool
    var showShareSheet: Bool
    var showFurnitureDimensionsInput: Bool
    var showRoomFurnitureCalibrate: Bool
    var supportsMetricFurnitureMeasurementUI: Bool
    var isCapturingSnapshot: Bool
}

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

    /// Match ``RoomBounds`` splat framing (depth-adaptive inset; tighter back-wall standoff).
    private static func backCenterInsetFraction(depth: Float) -> Float {
        let t = min(1.0, max(0.0, depth / 6.0))
        return 0.035 + 0.065 * t
    }

    /// Camera at back center with depth-adaptive inset (matches Metal / list / preview).
    private let cameraPadding: Float = 0.05

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
    /// When opening a saved PLY from Home, used for FurnitureFit room targets and persisted metadata.
    let savedRoomModel: USDZModel?
    @Environment(\.dismiss) private var dismiss

    /// PLY for MetalSplatter: legacy `_classic.ply` when present, otherwise the canonical saved `.ply`.
    private let viewerPlyURL: URL
    /// Share/export targets. When sibling SHARP variants exist, share all of them together,
    /// plus camera EXIF sidecar metadata when present.
    private let shareableRoomURLs: [URL]
    /// Whether this room should use SHARP classic orientation/rendering even without `_classic` in the file name.
    private let viewerUsesClassicPlyBehavior: Bool

    /// SHARP **`_classic.ply`** write-time AABB (scene units); set when pushing from ``SinglePhotoRoomViewer`` after generation.
    private let sharpPlyAabbW: Float?
    private let sharpPlyAabbH: Float?
    private let sharpPlyAabbD: Float?
    /// Oriented source photo pixel size (for proportion-style compare); nil when not from fresh generation.
    private let sourcePhotoPixelWidth: Int?
    private let sourcePhotoPixelHeight: Int?

    init(
        plyURL: URL,
        allowSave: Bool = true,
        photoOrientation: PhotoOrientation = .portrait,
        savedRoomWidth: Float? = nil,
        savedRoomHeight: Float? = nil,
        savedRoomModel: USDZModel? = nil,
        sharpPlyAabbWidth: Float? = nil,
        sharpPlyAabbHeight: Float? = nil,
        sharpPlyAabbDepth: Float? = nil,
        sourcePhotoPixelWidth: Int? = nil,
        sourcePhotoPixelHeight: Int? = nil
    ) {
        self.plyURL = plyURL
        self.allowSave = allowSave
        self.photoOrientation = photoOrientation
        self.savedRoomWidth = savedRoomWidth
        self.savedRoomHeight = savedRoomHeight
        self.savedRoomModel = savedRoomModel
        self.sharpPlyAabbW = sharpPlyAabbWidth
        self.sharpPlyAabbH = sharpPlyAabbHeight
        self.sharpPlyAabbD = sharpPlyAabbDepth
        self.sourcePhotoPixelWidth = sourcePhotoPixelWidth
        self.sourcePhotoPixelHeight = sourcePhotoPixelHeight

        let basePath = plyURL.path
        let fm = FileManager.default
        let canonicalStem = Self.canonicalPlyStem(for: plyURL)
        let canonicalBaseURL = plyURL.deletingLastPathComponent().appendingPathComponent("\(canonicalStem).ply")
        let canonicalClassicURL = plyURL.deletingLastPathComponent().appendingPathComponent("\(canonicalStem)_classic.ply")
        let canonicalThreeDGSURL = plyURL.deletingLastPathComponent().appendingPathComponent("\(canonicalStem)_3dgs.ply")

        let preferredViewerCandidates: [URL]
        if basePath.hasSuffix("_classic.ply") {
            preferredViewerCandidates = [plyURL, canonicalClassicURL, canonicalBaseURL, canonicalThreeDGSURL]
        } else {
            preferredViewerCandidates = [canonicalClassicURL, canonicalBaseURL, canonicalThreeDGSURL, plyURL]
        }
        self.viewerPlyURL = preferredViewerCandidates.first(where: { fm.fileExists(atPath: $0.path) }) ?? plyURL
        self.shareableRoomURLs = Self.shareableRoomURLs(for: plyURL)
        self.viewerUsesClassicPlyBehavior = self.viewerPlyURL.path.hasSuffix("_classic.ply")
    }

    private static func canonicalPlyStem(for url: URL) -> String {
        var stem = url.deletingPathExtension().lastPathComponent
        if stem.hasSuffix("_classic") {
            stem = String(stem.dropLast("_classic".count))
        } else if stem.hasSuffix("_3dgs") {
            stem = String(stem.dropLast("_3dgs".count))
        }
        return stem
    }

    private static func shareableRoomURLs(for baseURL: URL) -> [URL] {
        let directory = baseURL.deletingLastPathComponent()
        let stem = canonicalPlyStem(for: baseURL)

        let candidates = [
            directory.appendingPathComponent("\(stem).ply"),
            directory.appendingPathComponent("\(stem)_classic.ply"),
            directory.appendingPathComponent("\(stem)_3dgs.ply"),
            directory.appendingPathComponent("\(stem)_camera_exif.json"),
            directory.appendingPathComponent("camera_exif.json"),
        ]
        let fm = FileManager.default
        var seen = Set<String>()
        return candidates.filter { url in
            guard fm.fileExists(atPath: url.path) else { return false }
            return seen.insert(url.path).inserted
        }
    }

    /// Bounds from the splat PLY (computed when loading; see `GaussianSplatView.onBoundsAvailable`).
    @State private var metalBounds: RoomBounds?

    /// Grid for ``GaussianSplatMeasurementHost/buildPointCloudForRoomGeometry`` (keep in sync with host defaults).
    private enum RoomGeometryDepthSampling {
        static let rows: Int = 48
        static let cols: Int = 48
        static let maxDistance: Float = 12
    }

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
    @State private var manualDepthProRoomDimensions: (width: Float, height: Float, depth: Float)?
    @State private var isRunningDepthProCalibration = false

    @State private var detectedFurnitureWidth: Float?
    /// Furniture height from ARKit depth when LiDAR/depth is available.
    @State private var detectedFurnitureHeightAR: Float?
    /// Bbox×room fallback height for UI / placement when AR height is unavailable.
    @State private var furnitureProportionalHeightMeters: Float?

    // User-input real furniture dimensions for room calibration
    @State private var showFurnitureDimensionsInput = false
    @State private var inputFurnitureHeight: String = ""
    @State private var realFurnitureHeight: Float?  // Confirmed real height in meters
    /// Frozen when the calibrate sheet opens so `applyCalibration` uses the same AR height the user saw.
    @State private var calibrationBaselineDetectedHeight: Float?

    // Calibrated room dimensions (computed from real furniture size)
    @State private var calibratedRoomHeight: Float?
    @State private var calibratedRoomWidth: Float?
    // Reject calibration when result would be unrealistically small (wrong input or wrong detected size)
    @State private var showCalibrationRejectAlert = false
    @State private var calibrationRejectMessage = ""

    // Wall-based calibration (tape-measured front wall)
    @State private var showWallCalibration = false
    @State private var inputWallWidth: String = ""
    @State private var inputWallHeight: String = ""

    /// When true, shows ⋮ Calibrate wall and the Tap to calibrate pill during brain (FurnitureFit). Default off (matches Android).
    @AppStorage("show_room_furniture_calibrate") private var showRoomFurnitureCalibrate = false

    /// Furniture height/wall tape UI uses scene depth; omit on non-LiDAR devices (e.g. iPhone 12) where sizing is unreliable.
    private var supportsMetricFurnitureMeasurementUI: Bool {
        QualitySettings.supportsLiDARSceneDepth
    }

    /// User-facing Infinite Zoom toggle for the room viewer (matches Settings).
    @AppStorage("roomViewer.infiniteZoom") private var infiniteZoomEnabled: Bool = true

    /// Default **on**: plane-aware / PLY×scale for saved **room metres** (ceiling / PLY span for `sceneToMeters`).
    @AppStorage("room_dimensions_bound_based") private var roomDimensionsBoundBased = true

    // Save room state
    @StateObject private var modelManager = USDZModelManager()
    @State private var isSavingRoom = false
    @State private var saveProgress: Double = 0.0
    @State private var saveProgressStatusText = L10n.RoomViewer.savingRoomEllipsis
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
    @State private var sharpRoomUIPauseApplied = false
    /// After first Furniture Fit segmentation this viewer session, skip startup progress when toggling brain on again.
    @State private var furnitureFitInitialSegmentationDone = false
    /// Pinch zoom for MetalSplatter (`GaussianSplatView`).
    @State private var metalSplatterZoom: Float = 1.0
    /// Bridges splat ``GaussianSplatView/Coordinator`` for splat depth point cloud (room intelligence).
    @StateObject private var splatMeasurementHost = GaussianSplatMeasurementHost()
    @State private var didEnableDefaultARCamera = false
    @State private var autoEnableARTask: Task<Void, Never>?
    @State private var roomModel: RoomModel?
    @State private var enhancedRoomMetadata: EnhancedRoomMetadata?
    @State private var isExtractingRoomGeometry = false
    @State private var didLoadPersistedRoomMetadata = false
    @State private var latestFitCheckResult: FitCheckResult?
    @State private var latestCornerPlacementSuggestions: [CornerPlacementSuggestion] = []
    @State private var latestEstimatedFurnitureDepthMeters: Float?
    @State private var latestAestheticScore: AestheticScore?
    /// Mean sRGB (0…1) from composited YOLOE cutout pixels; drives ``FurnitureProfile.primaryColor`` when set.
    @State private var segmentedFurnitureMeanSRGB: SIMD3<Float>?
    /// Collapsed shows only the header row; expanded shows dimensions, corners, depth note, harmony, and tips.
    @State private var isPlacementIntelligenceExpanded = false
    /// Pinch hint (top-trailing): icon always visible; text shows on load and when tapped, auto-hides after 3s.
    @State private var pinchHintExplanationVisible = false
    @State private var pinchHintHideTextTask: Task<Void, Never>?
    /// Brain hint (above brain button): text auto-hides after 3s; tap icon always stays; tap toggles text.
    @State private var brainHintExplanationVisible = false
    @State private var brainHintHideTextTask: Task<Void, Never>?
    @EnvironmentObject var authManager: AuthenticationManager

    var body: some View {
        sharpRoomBody
    }

    @ViewBuilder
    private var sharpRoomBaseLayer: some View {
        ZStack {
            metalSplatAndGestureLayer
            allOverlaysLayer
        }
        .background(Color.gray)
    }

    @ViewBuilder
    private var navigationTitleContent: some View {
        VStack(spacing: 2) {
            Text(navigationRoomMetersLine)
                .font(.headline)
                .lineLimit(2)
                .minimumScaleFactor(0.55)
                .multilineTextAlignment(.center)
            if splatMeasurementHost.arModeEnabled {
                Text("AR plane detection is measuring the room")
                    .font(.caption2)
                    .foregroundColor(.secondary)
                    .lineLimit(2)
                    .multilineTextAlignment(.center)
            }
        }
        .accessibilityElement(children: .combine)
    }

    private var sharpRoomBody: some View {
        sharpRoomBaseLayer
        .navigationBarHidden(isCapturingSnapshot)
        .navigationTitle("")
        .navigationBarTitleDisplayMode(.inline)
        // Always hide the system back: in landscape its touch target is often covered by the wide `.principal`
        // toolbar title: explicit leading `Back` stays tappable (saved list + new room).
        .navigationBarBackButtonHidden(true)
        .toolbar {
            ToolbarItem(placement: .navigationBarLeading) {
                Button {
                    if allowSave {
                        if saveWasSuccessful {
                            dismiss()
                        } else {
                            showDiscardUnsavedAlert = true
                        }
                    } else {
                        dismiss()
                    }
                } label: {
                    HStack(spacing: 4) {
                        Image(systemName: "chevron.left")
                        Text(L10n.Common.back)
                    }
                }
            }
            ToolbarItem(placement: .principal) {
                navigationTitleContent
                    .allowsHitTesting(false)
            }
            ToolbarItemGroup(placement: .navigationBarTrailing) {
                    Button {
                        splatMeasurementHost.recenterSharpRoomCamera()
                        logDebug("🎯 [SharpRoomView] Recenter (toolbar)")
                    } label: {
                        Image(systemName: "viewfinder")
                            .font(.title3)
                    }
                    .disabled(isLoading)
                    .accessibilityLabel(L10n.RoomViewer.recenterView)

                    // Ellipsis menu: share, save, calibrate, overlay reset, AR toggle, furniture actions.
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
                        if showRoomFurnitureCalibrate, supportsMetricFurnitureMeasurementUI {
                            Button(action: { showWallCalibration = true }) {
                                Label(L10n.RoomViewer.calibrateWall, systemImage: "ruler")
                            }
                        }
                        Button(action: {
                            NotificationCenter.default.post(name: NSNotification.Name("FurnitureFitResetOverlayScale"), object: nil)
                        }) {
                            Label(L10n.RoomViewer.resetOverlayScale, systemImage: "arrow.counterclockwise")
                        }
                        Button(action: {
                            let next = !splatMeasurementHost.arModeEnabled
                            logDebug("📱 [SharpRoomView] toggling in-room AR camera enabled=\(next)")
                            splatMeasurementHost.setARModeEnabled(next)
                        }) {
                            Label(
                                splatMeasurementHost.arModeEnabled
                                    ? L10n.Sharp.stillRoomCameraMode
                                    : L10n.Sharp.liveRoomCameraMode,
                                systemImage: splatMeasurementHost.arModeEnabled ? "hand.draw" : "iphone"
                            )
                        }
                        Button(action: {
                            splatMeasurementHost.rotateSelectedFurniture(by: .pi / 12)
                        }) {
                            Label("Rotate Selected Furniture", systemImage: "rotate.right")
                        }
                        Button(action: {
                            splatMeasurementHost.clearPlacedFurniture()
                        }) {
                            Label("Clear Furniture", systemImage: "trash")
                        }
                    } label: {
                        Image(systemName: "ellipsis.circle")
                    }
                    .disabled(isLoading)
                }
        }
        .sheet(isPresented: $showShareSheet) {
            ShareSheet(activityItems: shareableRoomURLs)
        }
        .onAppear {
            // Do not load YOLOE here — it peaks memory with WebKit (WKWebView) and can crash in WKWebViewConfiguration.
            // Load when the user turns on Furniture Fit (brain) instead.
            if photoOrientation == .landscape { OrientationLockManager.shared.lockToLandscape() } else { OrientationLockManager.shared.lockToPortrait() }
            logDebug("📐 [SharpRoomView] photoOrientation = \(photoOrientation)")
            loadPersistedRoomMetadataIfNeeded()
            syncModalHeavyWorkPauseForSharpRoomUI()
        }
        .onChange(of: sharpRoomModalPauseToken) { _, _ in syncModalHeavyWorkPauseForSharpRoomUI() }
        .onChange(of: showingFurnitureFit) { _, isOn in
            logFurnitureFitSize(
                "phase=toggle active=\(isOn) suppressStartupProgress=\(furnitureFitInitialSegmentationDone) " +
                "room_m=\(String(format: "%.2f", furnitureFitRoomWidth))×\(String(format: "%.2f", furnitureFitRoomHeight))×\(String(format: "%.2f", displayRoomDepth))"
            )
            if isOn {
                yoloeService.ensureModelLoaded()
                updateRoomPlacementIntelligence()
            } else {
                detectedFurnitureWidth = nil
                detectedFurnitureHeightAR = nil
                furnitureProportionalHeightMeters = nil
                latestFitCheckResult = nil
                latestCornerPlacementSuggestions = []
                latestEstimatedFurnitureDepthMeters = nil
                latestAestheticScore = nil
                segmentedFurnitureMeanSRGB = nil
                yoloeService.releaseResources()
            }
        }
        .onChange(of: segmentedFurnitureMeanSRGB) { _, _ in updateRoomPlacementIntelligence() }
        .onChange(of: detectedFurnitureWidth) { _, _ in updateRoomPlacementIntelligence() }
        .onChange(of: detectedFurnitureHeightAR) { _, _ in updateRoomPlacementIntelligence() }
        .onChange(of: furnitureProportionalHeightMeters) { _, _ in updateRoomPlacementIntelligence() }
        .onChange(of: roomModel) { _, _ in updateRoomPlacementIntelligence() }
        .onChange(of: splatMeasurementHost.arPlaneRoomMeasurement) { _, measurement in
            if let measurement {
                logARRoomMeasure(
                    "phase=arkit_planes_ready live_room=\(splatMeasurementHost.arModeEnabled ? "on" : "off") " +
                        "arkit_plane_m_W×H×D=\(String(format: "%.4f", measurement.widthMeters))×" +
                        "\(String(format: "%.4f", measurement.heightMeters))×" +
                        "\(String(format: "%.4f", measurement.depthMeters)) " +
                        "room_dims_authoritative_source=\(measurement.sourceLabel)"
                )
            } else if splatMeasurementHost.arModeEnabled {
                logARRoomMeasure("phase=arkit_planes_ready live_room=on arkit_plane_note=not_ready")
            }
            updateRoomPlacementIntelligence()
        }
        .onChange(of: showRoomFurnitureCalibrate) { _, enabled in
            if !enabled {
                showFurnitureDimensionsInput = false
                showWallCalibration = false
            }
        }
        .onChange(of: isLoading) { _, loading in
            guard !loading, !didEnableDefaultARCamera else { return }
            didEnableDefaultARCamera = true
            if allowSave {
                logDebug("📱 [SharpRoomView] Fresh SHARP room loaded — AR will not auto-start; user must enable AR after SHARP generation/resources fully settle")
            } else {
                logDebug("📱 [SharpRoomView] Saved room loaded — defaulting to still room; AR will only start on user action")
            }
        }
        .onDisappear {
            autoEnableARTask?.cancel()
            autoEnableARTask = nil
            cancelPinchHintTasks()
            cancelBrainHintTasks()
            OrientationLockManager.shared.unlock()
            splatMeasurementHost.setModalHeavyWorkPaused(false)
            splatMeasurementHost.setARModeEnabled(false)
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
        .onChange(of: isLoading) { _, loading in
            if loading {
                cancelPinchHintTasks()
                cancelBrainHintTasks()
            } else {
                restartBrainGestureHint()
            }
        }
        // Omit `.leading` so the interactive pop gesture is not deferred behind splat gestures (saved rooms).
        .defersSystemGestures(on: [.top, .bottom, .trailing])
        .disableBackSwipeIf(allowSave)
    }

    /// Native Metal splats via MetalSplatter; gestures are on `MTKView`. Menu / D-pad use the same notifications as the old WebGL viewer (`GaussianSplatView` observes them).
    private var metalSplatAndGestureLayer: some View {
        GaussianSplatView(
            plyURL: viewerPlyURL,
            isLoading: $isLoading,
            loadError: $error,
            zoomLevel: $metalSplatterZoom,
            infiniteZoom: infiniteZoomEnabled,
            arReferenceOrientation: photoOrientation,
            treatAsClassicPly: viewerUsesClassicPlyBehavior,
            onBoundsAvailable: { bounds in
                DispatchQueue.main.async {
                    metalBounds = bounds
                    seedFrontWallDimensionsFromPlyBoundsIfNeeded()
                    let plyKind = viewerUsesClassicPlyBehavior ? "classic_ply" : "base_ply"
                    logPlyBoundsDiagnostic(
                        "Metal splat AABB (\(plyKind) file=\(viewerPlyURL.lastPathComponent)) su: " +
                        "W=\(String(format: "%.3f", bounds.width)) H=\(String(format: "%.3f", bounds.height)) D=\(String(format: "%.3f", bounds.depth)) " +
                        "X[\(String(format: "%.3f", bounds.minX)),\(String(format: "%.3f", bounds.maxX))] " +
                        "Y[\(String(format: "%.3f", bounds.minY)),\(String(format: "%.3f", bounds.maxY))] " +
                        "Z[\(String(format: "%.3f", bounds.minZ)),\(String(format: "%.3f", bounds.maxZ))]"
                    )
                    logSharpRoomDimensionApproaches(metalBounds: bounds)
                }
            },
            measurementHost: splatMeasurementHost
        )
        .ignoresSafeArea()
        .onChange(of: isLoading) { _, loading in
            guard !loading else { return }
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.28) {
                // AR-only mode: disable legacy SHARP depth-grid / RoomGeometryEngine room measurement.
                logDebug("📐 [SharpRoomView] AR-only mode — skipped non-AR room geometry extraction")
            }
        }
        .onAppear {
            guard !allowSave else { return }
            Task { @MainActor in
                await Task.yield()
                metalSplatterZoom = 1.0
            }
        }
    }

    private func logSharpRoomDimensionApproaches(metalBounds _: RoomBounds) {
        logDebug("📐 [SharpRoomView] AR-only mode — skipped SHARP room dimension comparison logging")
    }

    private func seedFrontWallDimensionsFromPlyBoundsIfNeeded() {}

    // MARK: - Gesture hint chips (pinch + brain)

    private func toggleFurnitureFit() {
        if showingFurnitureFit {
            showingFurnitureFit = false
        } else {
            furnitureFitInitialSegmentationDone = false
            SHARPService.shared.releaseResources()
            showingFurnitureFit = true
        }
    }

    private func cancelPinchHintTasks() {
        pinchHintHideTextTask?.cancel()
        pinchHintHideTextTask = nil
    }

    private func schedulePinchHintTextAutoHide(seconds: UInt64 = 3) {
        pinchHintHideTextTask?.cancel()
        pinchHintHideTextTask = Task { @MainActor in
            try? await Task.sleep(for: .seconds(seconds))
            guard !Task.isCancelled else { return }
            pinchHintExplanationVisible = false
        }
    }

    private func restartPinchGestureHint() {
        cancelPinchHintTasks()
        pinchHintExplanationVisible = true
        schedulePinchHintTextAutoHide(seconds: 3)
    }

    private func onPinchHintIconTapped() {
        cancelPinchHintTasks()
        pinchHintExplanationVisible.toggle()
        if pinchHintExplanationVisible {
            schedulePinchHintTextAutoHide(seconds: 3)
        }
    }

    private func cancelBrainHintTasks() {
        brainHintHideTextTask?.cancel()
        brainHintHideTextTask = nil
    }

    private func scheduleBrainHintTextAutoHide(seconds: UInt64 = 3) {
        brainHintHideTextTask?.cancel()
        brainHintHideTextTask = Task { @MainActor in
            try? await Task.sleep(for: .seconds(seconds))
            guard !Task.isCancelled else { return }
            brainHintExplanationVisible = false
        }
    }

    private func restartBrainGestureHint() {
        cancelBrainHintTasks()
        brainHintExplanationVisible = true
        scheduleBrainHintTextAutoHide(seconds: 3)
    }

    private func onBrainHintIconTapped() {
        cancelBrainHintTasks()
        brainHintExplanationVisible.toggle()
        if brainHintExplanationVisible {
            scheduleBrainHintTextAutoHide(seconds: 3)
        }
    }

    private var pinchHintAccessibilityLabel: String {
        L10n.RoomViewer.pinchGestureHintExplanation + " " + L10n.RoomViewer.gestureHintToggleAccessibility
    }

    private var brainHintAccessibilityLabel: String {
        L10n.RoomViewer.brainGestureHintExplanation + " " + L10n.RoomViewer.gestureHintToggleAccessibility
    }

    /// Still Room ↔ Live Room quick toggle (⋮ menu also toggles). No status text — avoids clutter and overlap with the D-pad.
    private var arModeQuickTogglePill: some View {
        Button {
            let next = !splatMeasurementHost.arModeEnabled
            logDebug("📱 [SharpRoomView] quick toggle in-room AR enabled=\(next)")
            splatMeasurementHost.setARModeEnabled(next)
        } label: {
            Label(
                splatMeasurementHost.arModeEnabled ? L10n.Sharp.stillRoom : L10n.Sharp.liveRoom,
                systemImage: splatMeasurementHost.arModeEnabled ? "hand.draw.fill" : "iphone"
            )
            .font(.caption.weight(.semibold))
            .foregroundColor(.white)
            .padding(.horizontal, 10)
            .padding(.vertical, 8)
            .background(Capsule().fill((splatMeasurementHost.arModeEnabled ? Color.green : Color.blue).opacity(0.85)))
        }
        .buttonStyle(.plain)
        .accessibilityHint(L10n.Sharp.cameraModeToggleAccessibilityHint)
    }

    /// D-pad cluster only (same notifications as Metal/WebGL parity).
    private var cameraDPadCluster: some View {
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
    }

    /// Live / Still Room pill always **above** D-pad (portrait + landscape). “Live Room” is motion-tracked splat camera
    /// (no live camera passthrough). Landscape uses a trailing `Spacer` in a full-height column so controls stay top-left.
    private var cameraButtonsOverlay: some View {
        ZStack(alignment: .topLeading) {
            Color.clear
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .allowsHitTesting(false)
            VStack(alignment: .leading, spacing: 0) {
                VStack(alignment: .leading, spacing: 12) {
                    arModeQuickTogglePill
                    cameraDPadCluster
                }
                .padding(.leading, 12)
                .padding(.top, 12)
                if photoOrientation == .landscape {
                    Spacer(minLength: 0)
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        }
        .opacity(isCapturingSnapshot ? 0 : 1)
        .zIndex(18)
    }

    /// Top-trailing hint: pinch icon stays; helper text shows on load and when tapped, hides after 3s.
    private var pinchGestureHintOverlay: some View {
        ZStack(alignment: .topTrailing) {
            Color.clear
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .allowsHitTesting(false)
            VStack(alignment: .trailing, spacing: 6) {
                if pinchHintExplanationVisible {
                    Text(L10n.RoomViewer.pinchGestureHintExplanation)
                        .font(.caption2)
                        .foregroundColor(.white)
                        .multilineTextAlignment(.trailing)
                        .fixedSize(horizontal: false, vertical: true)
                        .frame(maxWidth: 220)
                        .padding(8)
                        .background(RoundedRectangle(cornerRadius: 8).fill(Color.black.opacity(0.78)))
                        .transition(.opacity)
                }
                Button(action: onPinchHintIconTapped) {
                    Image(systemName: "hand.pinch.fill")
                        .symbolRenderingMode(.hierarchical)
                        .font(.system(size: 18, weight: .medium))
                        .foregroundStyle(.white)
                        .frame(width: 40, height: 40)
                        .background(Circle().fill(Color.black.opacity(0.5)))
                }
                .buttonStyle(.plain)
                .accessibilityLabel(pinchHintAccessibilityLabel)

                if !allowSave {
                    Button {
                        Task { await runDepthProRoomCalibration() }
                    } label: {
                        HStack(spacing: 6) {
                            if isRunningDepthProCalibration {
                                ProgressView()
                                    .controlSize(.small)
                                    .tint(.white)
                            }
                            Text(isRunningDepthProCalibration ? "CALIBRATING..." : "CALIBRATE SIZE")
                                .font(.system(size: 10, weight: .semibold))
                                .foregroundColor(.white)
                        }
                        .padding(.horizontal, 10)
                        .padding(.vertical, 6)
                        .background(Capsule().fill(Color.black.opacity(0.62)))
                    }
                    .buttonStyle(.plain)
                    .disabled(isRunningDepthProCalibration || isSavingRoom)

                    if let dims = manualDepthProRoomDimensions {
                        Text(String(format: "%.1f × %.1f × %.1f m", dims.width, dims.height, dims.depth))
                            .font(.system(size: 10, weight: .medium))
                            .foregroundColor(.white.opacity(0.94))
                            .padding(.horizontal, 8)
                            .padding(.vertical, 5)
                            .background(RoundedRectangle(cornerRadius: 8).fill(Color.black.opacity(0.62)))
                    }
                }
            }
            .padding(12)
            .onAppear { restartPinchGestureHint() }
            .onDisappear { cancelPinchHintTasks() }
        }
        .opacity(isCapturingSnapshot ? 0 : 1)
        .zIndex(12)
    }

    /// Text + tap icon only; place in a ``VStack`` above the brain button so the helper sits just above the brain.
    private var brainGestureHintColumn: some View {
        VStack(alignment: .center, spacing: 6) {
            if brainHintExplanationVisible {
                Text(L10n.RoomViewer.brainGestureHintExplanation)
                    .font(.caption2)
                    .foregroundColor(.white)
                    .multilineTextAlignment(.center)
                    .fixedSize(horizontal: false, vertical: true)
                    .frame(maxWidth: 200)
                    .padding(8)
                    .background(RoundedRectangle(cornerRadius: 8).fill(Color.black.opacity(0.78)))
                    .transition(.opacity)
            }
            Button(action: onBrainHintIconTapped) {
                Image(systemName: "hand.tap.fill")
                    .symbolRenderingMode(.hierarchical)
                    .font(.system(size: 18, weight: .medium))
                    .foregroundStyle(.white)
                    .frame(width: 40, height: 40)
                    .background(Circle().fill(Color.black.opacity(0.5)))
            }
            .buttonStyle(.plain)
            .accessibilityLabel(brainHintAccessibilityLabel)
        }
        .onAppear { restartBrainGestureHint() }
        .onDisappear { cancelBrainHintTasks() }
    }

    @ViewBuilder
    private var brainButtonWithHintAbove: some View {
        VStack(alignment: .center, spacing: 6) {
            brainGestureHintColumn
            Button(action: toggleFurnitureFit) {
                Image(systemName: "brain.head.profile")
                    .font(.system(size: 28))
                    .foregroundColor(.white)
                    .frame(width: 60, height: 60)
                    .background(Circle().fill(showingFurnitureFit ? Color.green : Color.blue).shadow(radius: 5))
            }
            .disabled(isLoading)
        }
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

    private var roomModelMetersDimensions: (width: Float, height: Float, depth: Float)? {
        guard let roomModel else { return nil }
        let width = roomModel.widthMeters
        let height = roomModel.heightMeters
        let depth = roomModel.depthMeters
        // Nav / save emphasize H×D only; width can be uncertain until wall pairs land.
        guard height.isFinite, depth.isFinite,
              height > 0.05, depth > 0.05 else {
            return nil
        }
        let w = width.isFinite && width > 0.05 ? width : 0
        return (w, height, depth)
    }

    private var savedSceneDimensions: RoomRaycastDimensions? {
        guard let savedRoomModel,
              let width = savedRoomModel.roomSceneWidth,
              let height = savedRoomModel.roomSceneHeight,
              let depth = savedRoomModel.roomSceneDepth,
              width.isFinite, height.isFinite, depth.isFinite,
              width > 0.05, height > 0.05, depth > 0.05 else {
            return nil
        }
        return RoomRaycastDimensions(width: width, height: height, depth: depth)
    }

    private var roomModelSceneDimensions: RoomRaycastDimensions? {
        guard let roomModel else { return nil }
        let ext = roomModel.planeAwareSceneExtent
        guard ext.width.isFinite, ext.height.isFinite, ext.depth.isFinite,
              ext.width > 0.05, ext.height > 0.05, ext.depth > 0.05 else {
            return nil
        }
        return RoomRaycastDimensions(width: ext.width, height: ext.height, depth: ext.depth)
    }

    private var savedRoomMetersDimensions: (width: Float, height: Float, depth: Float)? {
        guard let savedRoomModel,
              let width = savedRoomModel.roomWidth,
              let height = savedRoomModel.roomHeight,
              let depth = savedRoomModel.roomDepth,
              width.isFinite, height.isFinite, depth.isFinite,
              width > 0.05, height > 0.05, depth > 0.05 else {
            return nil
        }
        return (width, height, depth)
    }

    private var arPlaneMeasuredRoomDimensions: (width: Float, height: Float, depth: Float)? {
        guard let measurement = splatMeasurementHost.arPlaneRoomMeasurement else { return nil }
        let width = measurement.widthMeters
        let height = measurement.heightMeters
        let depth = measurement.depthMeters
        guard width.isFinite, height.isFinite, depth.isFinite,
              width > 0.05, height > 0.05, depth > 0.05 else {
            return nil
        }
        return (width, height, depth)
    }

    private var activeRoomMetersDimensions: (width: Float, height: Float, depth: Float)? {
        manualDepthProRoomDimensions ?? arPlaneMeasuredRoomDimensions
    }

    /// Width/height/depth for nav title, FurnitureFit, and save.
    /// AR-only mode: dimensions come from AR plane detection, not SHARP geometry / depth-grid extraction.
    private var displayRoomWidth: Float {
        activeRoomMetersDimensions?.width ?? 0
    }

    private var displayRoomHeight: Float {
        activeRoomMetersDimensions?.height ?? 0
    }

    /// Depth in meters.
    private var displayRoomDepth: Float {
        activeRoomMetersDimensions?.depth ?? 0
    }

    /// Nav bar: **height × depth** in metres only (width deferred).
    private var navigationRoomMetersLine: String {
        let h: Float
        let d: Float
        if let r = resolvedRoomMetersDimensions {
            h = r.height
            d = r.depth
        } else {
            h = displayRoomHeight
            d = displayRoomDepth
        }
        if h.isFinite, d.isFinite, h > 0.05, d > 0.05 {
            return String(format: "%.1f m H × %.1f m D", h, d)
        }
        return L10n.RoomViewer.measuringRoom
    }

    /// Baseline W/H before calibration in meter space when available.
    private var sourceRoomWidth: Float {
        activeRoomMetersDimensions?.width ?? 0
    }

    private var sourceRoomHeight: Float {
        activeRoomMetersDimensions?.height ?? 0
    }

    private var resolvedRoomMetersDimensions: (width: Float, height: Float, depth: Float)? {
        activeRoomMetersDimensions
    }

    /// Room dimensions for FurnitureFit — same chain as nav title / save.
    private var furnitureFitRoomWidth: Float { displayRoomWidth }

    private var furnitureFitRoomHeight: Float { displayRoomHeight }

    /// Scene-unit room for Furniture Fit ratios: extracted room bounds or persisted `.meta` scene fields.
    private var furnitureFitSceneDimensions: RoomRaycastDimensions? {
        return nil
    }

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
            roomDepthMeters: displayRoomDepth,
            roomRaycastSceneDimensions: furnitureFitSceneDimensions,
            roomModel: roomModel,
            cameraFocalLengthPixels: 0,
            onFurnitureSizeEstimated: { estimate in
                detectedFurnitureWidth = estimate.widthMeters
                if let arHeight = estimate.arHeightMeters,
                   arHeight.isFinite,
                   arHeight > 0.05 {
                    detectedFurnitureHeightAR = arHeight
                    furnitureProportionalHeightMeters = nil
                } else {
                    detectedFurnitureHeightAR = nil
                    furnitureProportionalHeightMeters = estimate.heightMeters > 0.05 ? estimate.heightMeters : nil
                    logFurnitureFitSize(
                        "phase=viewer_height_fallback width_m=\(String(format: "%.3f", estimate.widthMeters)) prop_h_m=\(String(format: "%.3f", estimate.heightMeters))"
                    )
                }
            },
            suppressStartupProgress: furnitureFitInitialSegmentationDone,
            onFirstSegmentationComplete: {
                furnitureFitInitialSegmentationDone = true
                logFurnitureFitSize("phase=first_segmentation_complete viewer_session=true")
            },
            onSegmentationMaskMeanColorSRGB: { meanSRGB in
                segmentedFurnitureMeanSRGB = meanSRGB
            },
            sharpRoomSplatMeasurementHost: splatMeasurementHost
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
                if let h = calibrationBaselineDetectedHeight ?? detectedFurnitureHeightAR {
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
              let detectedHeight = calibrationBaselineDetectedHeight ?? detectedFurnitureHeightAR,
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
        let displayH = detectedFurnitureHeightAR ?? furnitureProportionalHeightMeters ?? 0
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

    @ViewBuilder
    private var roomIntelligencePlacementCard: some View {
        if showingFurnitureFit,
           roomModel != nil,
           placementIntelligenceHasFurnitureSignal,
           latestAestheticScore != nil {
            let dimensions = derivedDetectedFurnitureDimensionsForRoomIntelligence()
            let fit = latestFitCheckResult
            VStack(alignment: .leading, spacing: 4) {
                Button {
                    withAnimation(.easeInOut(duration: 0.18)) {
                        isPlacementIntelligenceExpanded.toggle()
                    }
                } label: {
                    HStack(spacing: 8) {
                        Text(L10n.RoomViewer.placementIntelligenceTitle)
                            .font(.caption.bold())
                            .foregroundColor(.white)
                        Spacer(minLength: 8)
                        if let fit {
                            Text(
                                fit.fitsInRoom
                                    ? L10n.RoomViewer.placementFitCount(max(fit.fitLocations.count, 1))
                                    : L10n.RoomViewer.placementNoFit
                            )
                            .font(.caption2.bold())
                            .foregroundColor(fit.fitsInRoom ? .green : .red)
                        } else {
                            Text(L10n.RoomViewer.placementBadgeStyleOnly)
                                .font(.caption2.bold())
                                .foregroundColor(.cyan.opacity(0.95))
                        }
                        Image(systemName: isPlacementIntelligenceExpanded ? "chevron.up" : "chevron.down")
                            .font(.caption.bold())
                            .foregroundColor(.white.opacity(0.85))
                    }
                }
                .buttonStyle(.plain)

                if isPlacementIntelligenceExpanded {
                    if dimensions == nil {
                        Text(L10n.RoomViewer.placementMetricUnavailableNote)
                            .font(.caption2)
                            .foregroundColor(.gray)
                    }
                    if let dimensions {
                        Text(
                            L10n.RoomViewer.placementDetectedSizeMeters(
                                width: Double(dimensions.widthM),
                                height: Double(dimensions.heightM),
                                depth: Double(dimensions.depthM)
                            )
                        )
                        .font(.caption2)
                        .foregroundColor(.white.opacity(0.92))
                    }
                    if let fit {
                        if fit.fitsInRoom {
                            Text(
                                fit.fitLocations.isEmpty
                                    ? "Fits room bounds. No clear free-floor region yet."
                                    : "Fits room. \(fit.fitLocations.count) candidate floor region(s)."
                            )
                            .font(.caption2)
                            .foregroundColor(.green)
                        } else {
                            Text("Current furniture footprint exceeds the detected room extents.")
                                .font(.caption2)
                                .foregroundColor(.red)
                        }
                        if let bestCorner = latestCornerPlacementSuggestions.first {
                            Text(
                                String(
                                    format: "Best corner score %.2f, rot %.0f°",
                                    bestCorner.score,
                                    bestCorner.yRotationRad * 180 / .pi
                                )
                            )
                            .font(.caption2)
                            .foregroundColor(.orange)
                        }
                        if let firstWarning = fit.warnings.first {
                            Text(firstWarning)
                                .font(.caption2)
                                .foregroundColor(.yellow)
                        } else if latestEstimatedFurnitureDepthMeters != nil {
                            Text("Depth is estimated until catalog or semantic data is available.")
                                .font(.caption2)
                                .foregroundColor(.gray)
                        }
                    }
                    if let aesthetic = latestAestheticScore {
                        Text(
                            L10n.RoomViewer.placementHarmonySummary(
                                harmonyScore: aesthetic.harmonyScore,
                                harmonyTypeName: aesthetic.harmonyType.localizedDisplayName,
                                contrastScore: aesthetic.contrastScore,
                                styleFit: aesthetic.styleCompatibilityScore
                            )
                        )
                        .font(.caption2)
                        .foregroundColor(.white.opacity(0.88))
                        ForEach(Array(aesthetic.recommendations.prefix(4).enumerated()), id: \.offset) { _, line in
                            Text("• \(line)")
                                .font(.caption2)
                                .foregroundColor(.white.opacity(0.86))
                        }
                    }
                }
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 8)
            .background(Color.black.opacity(0.72))
            .cornerRadius(8)
        }
    }

    @ViewBuilder
    private var roomIntelligencePlacementCardResetOnExit: some View {
        roomIntelligencePlacementCard
            .onChange(of: showingFurnitureFit) { _, isShowing in
                if !isShowing {
                    isPlacementIntelligenceExpanded = false
                }
            }
            .onChange(of: latestFitCheckResult?.fitsInRoom) { _, _ in
                if latestFitCheckResult == nil, latestAestheticScore == nil {
                    isPlacementIntelligenceExpanded = false
                }
            }
    }

    @ViewBuilder private var bottomBarsOverlayView: some View {
        if photoOrientation == .landscape {
            ZStack(alignment: .bottom) {
                Color.clear
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .allowsHitTesting(false)
                HStack(spacing: 20) {
                    brainButtonWithHintAbove
                    if showingFurnitureFit {
                        VStack(alignment: .trailing, spacing: 8) {
                            roomIntelligencePlacementCardResetOnExit
                            if supportsMetricFurnitureMeasurementUI {
                                if showRoomFurnitureCalibrate {
                                    Button(action: { showFurnitureDimensionsInput = true }) {
                                        furnitureMeasurementPillContent(showTapHint: true)
                                    }
                                } else {
                                    furnitureMeasurementPillContent(showTapHint: false)
                                }
                            }
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
                    brainButtonWithHintAbove
                        .padding(.leading, 16)
                    Spacer().allowsHitTesting(false)
                    VStack(spacing: 8) {
                        if showingFurnitureFit {
                            roomIntelligencePlacementCardResetOnExit
                            if supportsMetricFurnitureMeasurementUI {
                                if showRoomFurnitureCalibrate {
                                    Button(action: { showFurnitureDimensionsInput = true }) {
                                        furnitureMeasurementPillContent(showTapHint: true)
                                    }
                                } else {
                                    furnitureMeasurementPillContent(showTapHint: false)
                                }
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
            if !isLoading {
                cameraButtonsOverlay
                pinchGestureHintOverlay
            }
            if isLoading { loadingOverlayView }
            errorOverlayView
            if showingFurnitureFit { furnitureFitOverlayView }
            if isSavingRoom { saveRoomProgressOverlay }
            if showFurnitureDimensionsInput, showRoomFurnitureCalibrate, supportsMetricFurnitureMeasurementUI {
                calibrationOverlayView
                    .onAppear { calibrationBaselineDetectedHeight = detectedFurnitureHeightAR }
                    .onDisappear { calibrationBaselineDetectedHeight = nil }
            }
            if showWallCalibration, showRoomFurnitureCalibrate, supportsMetricFurnitureMeasurementUI {
                wallCalibrationOverlay
            }
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
        splatMeasurementHost.captureScreenshot { image in
            DispatchQueue.main.async {
                if let image {
                    logDebug("📸 Splat screenshot captured (Metal readback), saving to Photos...")
                    let composed = compositeSharpRoomSnapshotWithFurnitureFitIfNeeded(splatImage: image)
                    saveSharpRoomSnapshotToPhotos(composed)
                } else {
                    logDebug("📸 Metal capture unavailable; falling back to window hierarchy...")
                    captureSharpRoomSnapshotViaDrawHierarchy()
                    return
                }
                isCapturingSnapshot = false
            }
        }
    }

    private func saveSharpRoomSnapshotToPhotos(_ image: UIImage) {
        let saveBlock = {
            UIImageWriteToSavedPhotosAlbum(image, nil, nil, nil)
            logDebug("✅ Saved Sharp Room snapshot to Photos")
        }
        if #available(iOS 14, *) {
            let status = PHPhotoLibrary.authorizationStatus(for: .addOnly)
            switch status {
            case .authorized, .limited:
                saveBlock()
            case .denied, .restricted:
                logDebug("❌ Photos add-only access denied or restricted")
            case .notDetermined:
                PHPhotoLibrary.requestAuthorization(for: .addOnly) { newStatus in
                    DispatchQueue.main.async {
                        if newStatus == .authorized || newStatus == .limited {
                            saveBlock()
                        } else {
                            logDebug("❌ Photos add-only access not granted")
                        }
                    }
                }
            @unknown default:
                logDebug("❌ Unknown Photos authorization status")
            }
        } else {
            let status = PHPhotoLibrary.authorizationStatus()
            switch status {
            case .authorized, .limited:
                saveBlock()
            case .denied, .restricted:
                logDebug("❌ Photos access denied or restricted")
            case .notDetermined:
                PHPhotoLibrary.requestAuthorization { newStatus in
                    DispatchQueue.main.async {
                        if newStatus == .authorized || newStatus == .limited {
                            saveBlock()
                        } else {
                            logDebug("❌ Photos access not granted")
                        }
                    }
                }
            @unknown default:
                logDebug("❌ Unknown Photos authorization status (legacy)")
            }
        }
    }

    /// Metal capture is only the `MTKView`; Furniture Fit draws segmentation in ``FurnitureFitContainerView`` above it. Composite both into one full-window image.
    private func compositeSharpRoomSnapshotWithFurnitureFitIfNeeded(splatImage: UIImage) -> UIImage {
        guard showingFurnitureFit else { return splatImage }
        let scenes = UIApplication.shared.connectedScenes.compactMap { $0 as? UIWindowScene }
        let windows = scenes.flatMap { $0.windows }
        guard let window = windows.first(where: { $0.isKeyWindow }) ?? windows.first,
              let mtkView = findFirstMTKView(in: window) else {
            logDebug("📸 Composite: no window or MTKView — using splat-only image")
            return splatImage
        }
        let furnitureViews = collectFurnitureFitContainerViews(in: window)
        guard let furnitureView = furnitureViews.last else {
            logDebug("📸 Composite: no FurnitureFitContainerView — using splat-only image")
            return splatImage
        }

        let format = UIGraphicsImageRendererFormat()
        format.scale = window.screen.scale
        format.opaque = true
        let bounds = window.bounds
        let renderer = UIGraphicsImageRenderer(bounds: bounds, format: format)
        return renderer.image { ctx in
            UIColor(white: 0.5, alpha: 1).setFill()
            ctx.fill(bounds)

            let mtkFrame = mtkView.convert(mtkView.bounds, to: window)
            splatImage.draw(in: mtkFrame)

            let fitFrame = furnitureView.convert(furnitureView.bounds, to: window)
            ctx.cgContext.saveGState()
            ctx.cgContext.translateBy(x: fitFrame.origin.x, y: fitFrame.origin.y)
            furnitureView.drawHierarchy(in: CGRect(origin: .zero, size: furnitureView.bounds.size), afterScreenUpdates: true)
            ctx.cgContext.restoreGState()
        }
    }

    private func collectFurnitureFitContainerViews(in root: UIView) -> [FurnitureFitContainerView] {
        var out: [FurnitureFitContainerView] = []
        if let fit = root as? FurnitureFitContainerView {
            out.append(fit)
        }
        for sub in root.subviews {
            out.append(contentsOf: collectFurnitureFitContainerViews(in: sub))
        }
        return out
    }

    private func captureSharpRoomSnapshotViaDrawHierarchy() {
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.08) {
            let scenes = UIApplication.shared.connectedScenes.compactMap { $0 as? UIWindowScene }
            let windows = scenes.flatMap { $0.windows }
            guard let window = windows.first(where: { $0.isKeyWindow }) ?? windows.first else {
                isCapturingSnapshot = false
                logDebug("❌ No window found for snapshot")
                return
            }
            let targetView: UIView = findFirstMTKView(in: window) ?? window
            let format = UIGraphicsImageRendererFormat()
            format.scale = targetView.traitCollection.displayScale
            let renderer = UIGraphicsImageRenderer(bounds: targetView.bounds, format: format)
            let image = renderer.image { _ in
                targetView.drawHierarchy(in: targetView.bounds, afterScreenUpdates: true)
            }
            logDebug("📸 Hierarchy snapshot captured, saving to Photos...")
            saveSharpRoomSnapshotToPhotos(image)
            isCapturingSnapshot = false
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

                Text(saveProgressStatusText)
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

    /// Depth-buffer grid for ``RoomGeometryEngine``.
    @MainActor
    private func pointCloudForRoomGeometrySession() async -> [SIMD3<Float>] {
        splatMeasurementHost.requestRedrawForDepthMeasure()
        try? await Task.sleep(nanoseconds: 120_000_000)
        let pts = splatMeasurementHost.buildPointCloudForRoomGeometry(
            rows: RoomGeometryDepthSampling.rows,
            cols: RoomGeometryDepthSampling.cols,
            maxDistance: RoomGeometryDepthSampling.maxDistance
        )
        if pts.count >= 3 {
            logDebug(
                "📐 [PointCloud] captured \(pts.count) samples " +
                    "(\(RoomGeometryDepthSampling.rows)×\(RoomGeometryDepthSampling.cols), maxDist=\(RoomGeometryDepthSampling.maxDistance))",
            )
        } else {
            logDebug("📐 [PointCloud] capture returned \(pts.count) samples")
        }
        return pts
    }

    @MainActor
    private func currentValidatedEnhancedMetadata() -> EnhancedRoomMetadata? {
        enhancedRoomMetadata
    }

    private func measurementThumbnailForCurrentRoom() -> UIImage? {
        let roomFolder = viewerPlyURL.deletingLastPathComponent()
        var stem = viewerPlyURL.deletingPathExtension().lastPathComponent
        if stem.hasSuffix("_classic") {
            stem = String(stem.dropLast("_classic".count))
        }
        let candidates = [
            roomFolder.appendingPathComponent("\(stem)_thumbnail.jpg"),
            roomFolder.appendingPathComponent("\(stem)_thumbnail.png"),
        ]
        for url in candidates {
            if let image = UIImage(contentsOfFile: url.path) {
                return image
            }
        }
        return nil
    }

    private func measureRoomUsingMetricDepthForSave(plyBounds: RoomBounds?) async -> WallMeasurementEstimator.Result? {
        guard let thumbnail = measurementThumbnailForCurrentRoom() else {
            logDebug("📐 [SharpRoomView] Save: no thumbnail available for metric-depth wall measurement")
            return nil
        }

        _ = thumbnail
        _ = plyBounds
        logDebug("📐 [SharpRoomView] Save: automatic metric measurement disabled; use the Depth Pro calibrate button if needed")
        return nil

    }

    private func runDepthProRoomCalibration() async {
        guard !isRunningDepthProCalibration else { return }
        guard let thumbnail = measurementThumbnailForCurrentRoom() else {
            logDebug("📐 [SharpRoomView] EXPERIMENTAL CALIBRATE: missing thumbnail")
            return
        }
        let referenceDepthMeters = effectiveBounds?.depth ?? savedRoomMetersDimensions?.depth
        if let referenceDepthMeters, referenceDepthMeters > 0 {
            logDebug("📐 [SharpRoomView] EXPERIMENTAL CALIBRATE: using reference depth \(String(format: "%.3f", referenceDepthMeters))m")
        } else {
            logDebug("📐 [SharpRoomView] EXPERIMENTAL CALIBRATE: no reference depth; using unit-scale depth")
        }

        await MainActor.run {
            isRunningDepthProCalibration = true
            syncModalHeavyWorkPauseForSharpRoomUI()
        }
        defer {
            Task { @MainActor in
                isRunningDepthProCalibration = false
                syncModalHeavyWorkPauseForSharpRoomUI()
            }
        }

        var measurement = WallMeasurementEstimator.measureUsingMetricDepthAlignedPLY(
            roomURL: viewerPlyURL,
            thumbnail: thumbnail
        )

        if measurement == nil {
            await DepthProMetricDepthService.shared.generateMetricDepthIfPossible(
                roomURL: viewerPlyURL,
                thumbnail: thumbnail,
                referenceDepthMeters: referenceDepthMeters
            )
            measurement =
                WallMeasurementEstimator.measureUsingMetricDepthAlignedPLY(
                    roomURL: viewerPlyURL,
                    thumbnail: thumbnail
                ) ??
                WallMeasurementEstimator.measureUsingMetricDepthOnly(
                    roomURL: viewerPlyURL,
                    thumbnail: thumbnail,
                    photoOrientation: photoOrientation,
                    plyBounds: effectiveBounds
                )
        } else {
            logDebug("📐 [SharpRoomView] EXPERIMENTAL CALIBRATE: skipped depth model; SHARP edge/global path succeeded")
        }

        await MainActor.run {
            guard let measurement else {
                logDebug("📐 [SharpRoomView] EXPERIMENTAL CALIBRATE FAILED")
                return
            }
            manualDepthProRoomDimensions = (
                width: measurement.widthMeters,
                height: measurement.heightMeters,
                depth: measurement.depthMeters
            )
            jsFrontWallWidth = measurement.widthMeters
            jsFrontWallHeight = measurement.heightMeters
            logDebug(
                "📐 [SharpRoomView] EXPERIMENTAL CALIBRATE RESULT W×H×D=" +
                    "\(String(format: "%.3f", measurement.widthMeters))×" +
                    "\(String(format: "%.3f", measurement.heightMeters))×" +
                    "\(String(format: "%.3f", measurement.depthMeters))m " +
                    "mode=\(measurement.calibrationMode)"
            )

            if !allowSave {
                let fileName = viewerPlyURL.deletingPathExtension().lastPathComponent
                let fileExtension = viewerPlyURL.pathExtension
                do {
                    try modelManager.mergeRoomDimensionsIntoSavedRoomMetadata(
                        fileName: fileName,
                        modelFileExtension: fileExtension,
                        roomWidth: measurement.widthMeters,
                        roomHeight: measurement.heightMeters,
                        roomDepth: measurement.depthMeters
                    )
                    logDebug("✅ [SharpRoomView] EXPERIMENTAL CALIBRATE persisted saved-room dims")
                } catch {
                    logDebug("❌ [SharpRoomView] EXPERIMENTAL CALIBRATE failed to persist dims: \(error.localizedDescription)")
                }
            }
        }
    }

    private var sharpRoomModalPauseToken: SharpRoomModalPauseToken {
        SharpRoomModalPauseToken(
            showRoomNameInput: showRoomNameInput,
            isSavingRoom: isSavingRoom,
            isRunningDepthProCalibration: isRunningDepthProCalibration,
            showSaveAlert: showSaveAlert,
            showDiscardUnsavedAlert: showDiscardUnsavedAlert,
            showCalibrationRejectAlert: showCalibrationRejectAlert,
            showWallCalibration: showWallCalibration,
            showShareSheet: showShareSheet,
            showFurnitureDimensionsInput: showFurnitureDimensionsInput,
            showRoomFurnitureCalibrate: showRoomFurnitureCalibrate,
            supportsMetricFurnitureMeasurementUI: supportsMetricFurnitureMeasurementUI,
            isCapturingSnapshot: isCapturingSnapshot
        )
    }

    /// Pauses ARKit while modal UI is up so `TextField` / alerts are not competing with per-frame `ARSession` main-queue work.
    private func syncModalHeavyWorkPauseForSharpRoomUI() {
        let furnitureCalibSheet =
            showFurnitureDimensionsInput && showRoomFurnitureCalibrate && supportsMetricFurnitureMeasurementUI
        let pause =
            showRoomNameInput ||
            isSavingRoom ||
            isRunningDepthProCalibration ||
            showSaveAlert ||
            showDiscardUnsavedAlert ||
            showCalibrationRejectAlert ||
            showWallCalibration ||
            showShareSheet ||
            furnitureCalibSheet ||
            isCapturingSnapshot
        guard pause != sharpRoomUIPauseApplied else { return }
        sharpRoomUIPauseApplied = pause
        splatMeasurementHost.setModalHeavyWorkPaused(pause)
    }

    private func startSavingRoom() {
        guard !roomName.isEmpty else { return }

        let savedName = roomName
        logDebug("💾 [SharpRoomView] Starting room save: \(savedName)")

        Task {
            await MainActor.run {
                if showingFurnitureFit {
                    logDebug("💾 [SharpRoomView] Save: stopping Furniture Fit before persist")
                    showingFurnitureFit = false
                }
            }
            try? await Task.sleep(nanoseconds: 180_000_000)

            await MainActor.run {
                withAnimation(.easeIn(duration: 0.3)) {
                    isSavingRoom = true
                    saveProgress = 0.0
                    saveProgressStatusText = "Measuring room..."
                }
            }

            await MainActor.run { saveProgress = 0.15 }
            let currentPlyBounds = await MainActor.run { metalBounds }
            let metricMeasurement = await measureRoomUsingMetricDepthForSave(plyBounds: currentPlyBounds)

            let burstMeasurement: ARPlaneRoomMeasurement? = nil
            if metricMeasurement == nil {
                logDebug("📐 [SharpRoomView] Save: AR burst disabled; saving without live measurement fallback")
            }

            await MainActor.run {
                if let burstMeasurement {
                    splatMeasurementHost.updateARPlaneRoomMeasurement(burstMeasurement)
                }
                saveProgress = 0.35
                saveProgressStatusText = L10n.RoomViewer.savingRoomEllipsis
            }

            let fallbackARDimensions = await MainActor.run { activeRoomMetersDimensions }
            let roomW = metricMeasurement?.widthMeters ?? burstMeasurement?.widthMeters ?? fallbackARDimensions?.width
            let roomH = metricMeasurement?.heightMeters ?? burstMeasurement?.heightMeters ?? fallbackARDimensions?.height
            let roomD = metricMeasurement?.depthMeters ?? burstMeasurement?.depthMeters ?? fallbackARDimensions?.depth
            if let metricMeasurement, let roomW, let roomH, let roomD {
                logDebug(
                    "🟢 [SharpRoomView] Save: metric depth W×H×D=" +
                        "\(String(format: "%.3f", roomW))×\(String(format: "%.3f", roomH))×\(String(format: "%.3f", roomD))m " +
                        "mode=\(metricMeasurement.calibrationMode)"
                )
            } else if let roomW, let roomH, let roomD {
                logDebug(
                    "🟢 [SharpRoomView] Save: existing calibrated W×H×D=" +
                        "\(String(format: "%.3f", roomW))×\(String(format: "%.3f", roomH))×\(String(format: "%.3f", roomD))m"
                )
            } else {
                logDebug(
                    "🔴 [SharpRoomView] Save: burst AR returned nil dims — W×H×D=" +
                        "\(String(describing: roomW))×\(String(describing: roomH))×\(String(describing: roomD))m"
                )
            }

            await MainActor.run { saveProgress = 0.5 }
            let metadataForSave = await MainActor.run { currentValidatedEnhancedMetadata() }

            await MainActor.run {
                if let roomW, let roomH, roomW.isFinite, roomH.isFinite, roomW > 0.05, roomH > 0.05 {
                    jsFrontWallWidth = roomW
                    jsFrontWallHeight = roomH
                }
            }

            await MainActor.run { saveProgress = 0.72 }
            await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
                modelManager.savePLY(
                    from: viewerPlyURL,
                    name: savedName,
                    photoOrientation: photoOrientation,
                    roomWidth: roomW,
                    roomHeight: roomH,
                    roomDepth: roomD,
                    roomSceneWidth: nil,
                    roomSceneHeight: nil,
                    roomSceneDepth: nil
                ) { success, error in
                    logDebug(success ? "✅ [SharpRoomView] Room saved" : "❌ [SharpRoomView] Save failed: \(error ?? "unknown")")
                    Task { @MainActor in
                        if success, let metadata = metadataForSave {
                            metadata.printSaveDiagnostics()
                            do {
                                try modelManager.saveEnhancedMetadata(metadata, forSavedRoomNamed: savedName, fileType: .ply)
                                logDebug("✅ [SharpRoomView] Save: enhanced metadata persisted for saved room")
                            } catch {
                                logDebug("❌ [SharpRoomView] Failed to save enhanced metadata for saved room: \(error.localizedDescription)")
                            }
                        } else if success {
                            print(
                                "[SAVE_ENHANCED_METADATA] room=\"\(savedName)\" — no EnhancedRoomMetadata (nil). " +
                                "PLY save used display meters W×H×D=\(roomW)×\(roomH)×\(roomD)"
                            )
                        }
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
            saveProgressStatusText = L10n.RoomViewer.savingRoomEllipsis
        }

        roomName = ""
        logDebug("❌ [SharpRoomView] Room save cancelled")
    }

    private func loadPersistedRoomMetadataIfNeeded() {
        guard !didLoadPersistedRoomMetadata else { return }
        didLoadPersistedRoomMetadata = true

        if let savedRoomModel {
            if let metadata = modelManager.loadEnhancedMetadata(
                forSavedRoomNamed: savedRoomModel.fileName,
                fileType: savedRoomModel.fileType
            ) {
                enhancedRoomMetadata = metadata
                roomModel = nil
                logDebug("📐 [SharpRoomView] AR-only mode — ignored saved room geometry metadata")
                return
            }
        }

        if let metadata = modelManager.loadEnhancedMetadata(forRoomURL: viewerPlyURL) {
            enhancedRoomMetadata = metadata
            roomModel = nil
            logDebug("📐 [SharpRoomView] AR-only mode — ignored live room geometry metadata")
        }
    }

    private func scheduleRoomGeometryExtractionIfNeeded() {
        logDebug("📐 [SharpRoomView] AR-only mode — room geometry extraction disabled")
    }

    private func triggerRoomGeometryExtractionIfNeeded(force: Bool = false) {
        let _ = force
        logDebug("📐 [SharpRoomView] AR-only mode — RoomGeometryEngine / RANSAC path commented out")
    }

    private func persistEnhancedRoomMetadataIfPossible(_ metadata: EnhancedRoomMetadata) {
        do {
            try modelManager.saveEnhancedMetadata(metadata, nextTo: viewerPlyURL)
        } catch {
            logDebug("❌ [SharpRoomView] Failed to persist enhanced room metadata: \(error.localizedDescription)")
        }
    }

    private func derivedDetectedFurnitureDimensionsForRoomIntelligence() -> RoomFurnitureDimensions? {
        guard let width = detectedFurnitureWidth,
              width.isFinite,
              width > 0.05 else { return nil }

        let height = detectedFurnitureHeightAR ?? furnitureProportionalHeightMeters
        guard let height,
              height.isFinite,
              height > 0.05 else { return nil }

        let estimatedDepth = max(0.25, min(width * 0.72, 1.4))
        return RoomFurnitureDimensions(widthM: width, heightM: height, depthM: estimatedDepth)
    }

    private var authoritativeRoomModelForMetrics: RoomModel? {
        nil
    }

    /// Width, segmentation color, or full W×H×D — enough to show style hints without LiDAR height.
    private var placementIntelligenceHasFurnitureSignal: Bool {
        if let width = detectedFurnitureWidth, width.isFinite, width > 0.05 { return true }
        if segmentedFurnitureMeanSRGB != nil { return true }
        if derivedDetectedFurnitureDimensionsForRoomIntelligence() != nil { return true }
        return false
    }

    private func updateRoomPlacementIntelligence() {
        guard showingFurnitureFit else {
            latestFitCheckResult = nil
            latestCornerPlacementSuggestions = []
            latestEstimatedFurnitureDepthMeters = nil
            latestAestheticScore = nil
            return
        }
        guard let roomModel = authoritativeRoomModelForMetrics else {
            latestFitCheckResult = nil
            latestCornerPlacementSuggestions = []
            latestEstimatedFurnitureDepthMeters = nil
            latestAestheticScore = nil
            return
        }

        let hasFurnitureSignal = placementIntelligenceHasFurnitureSignal
        guard hasFurnitureSignal else {
            latestFitCheckResult = nil
            latestCornerPlacementSuggestions = []
            latestEstimatedFurnitureDepthMeters = nil
            latestAestheticScore = nil
            return
        }

        if let furniture = derivedDetectedFurnitureDimensionsForRoomIntelligence() {
            latestEstimatedFurnitureDepthMeters = furniture.depthM
            let fitEngine = FitCheckEngine(roomModel: roomModel)
            let fitResult = fitEngine.checkFit(furniture: furniture)
            let cornerPlacement = CornerPlacement(roomModel: roomModel)
            let suggestions = Array(cornerPlacement.suggestions(for: furniture).prefix(3))
            latestFitCheckResult = fitResult
            latestCornerPlacementSuggestions = suggestions

            logDebug(
                "📐 [SharpRoomView] Placement intelligence updated " +
                "furniture=\(String(format: "%.2f", furniture.widthM))×\(String(format: "%.2f", furniture.heightM))×\(String(format: "%.2f", furniture.depthM))m " +
                "fits=\(fitResult.fitsInRoom) fitLocations=\(fitResult.fitLocations.count) cornerSuggestions=\(suggestions.count)"
            )
        } else {
            latestEstimatedFurnitureDepthMeters = nil
            latestFitCheckResult = nil
            latestCornerPlacementSuggestions = []
            logDebug("📐 [SharpRoomView] Placement intelligence: metric fit skipped (no height); computing aesthetic only")
        }

        let palette = roomModel.surfacePalette
        let roomStyleTags = inferredRoomStyleTags(from: palette)
        let furnitureProfile = heuristicFurnitureProfileForAesthetic(
            roomModel: roomModel,
            segmentedMeanSRGB: segmentedFurnitureMeanSRGB
        )
        let aestheticAdvisor = AestheticAdvisor(palette: palette, roomStyleTags: roomStyleTags)
        latestAestheticScore = aestheticAdvisor.evaluate(furniture: furnitureProfile)

        logDebug(
            "📐 [SharpRoomView] Placement aesthetic harmony=\(String(format: "%.2f", latestAestheticScore?.harmonyScore ?? 0))"
        )
    }

    /// Maps splat-sampled ``SurfacePalette`` material hints to advisor style tags.
    private func inferredRoomStyleTags(from palette: SurfacePalette) -> [String] {
        var tags = Set<String>()
        let layers = [palette.floor, palette.walls, palette.ceiling]
        for layer in layers {
            guard let layer else { continue }
            switch layer.hint {
            case .wood: tags.formUnion(["rustic", "traditional"])
            case .tile: tags.insert("modern")
            case .concrete: tags.formUnion(["industrial", "modern"])
            case .carpet: tags.formUnion(["traditional", "eclectic"])
            case .plaster: tags.formUnion(["modern", "scandinavian"])
            case .brick: tags.formUnion(["traditional", "industrial"])
            case .marble: tags.formUnion(["modern", "luxury"])
            case .unknown: break
            }
        }
        if tags.isEmpty { return ["modern", "minimalist"] }
        return Array(tags).sorted().prefix(6).map { $0 }
    }

    /// Furniture color from segmented cutout mean when available; otherwise a palette-biased heuristic for coherent scores.
    private func heuristicFurnitureProfileForAesthetic(
        roomModel: RoomModel,
        segmentedMeanSRGB: SIMD3<Float>?
    ) -> FurnitureProfile {
        let palette = roomModel.surfacePalette
        let primary: SIMD3<Float>
        if let cutoutMean = segmentedMeanSRGB {
            primary = cutoutMean
        } else if let wall = palette.walls?.dominantColors.first {
            primary = SIMD3(
                min(wall.x * 0.82 + 0.06, 1),
                min(wall.y * 0.78 + 0.05, 1),
                min(wall.z * 0.74 + 0.04, 1)
            )
        } else if let floor = palette.floor?.dominantColors.first {
            primary = SIMD3(repeating: 0.38) * 0.55 + floor * 0.45
        } else if let ceiling = palette.ceiling?.dominantColors.first {
            primary = ceiling * SIMD3(0.55, 0.52, 0.48)
        } else {
            primary = SIMD3(0.44, 0.40, 0.36)
        }
        return FurnitureProfile(
            primaryColor: primary,
            accentColor: nil,
            styleTags: ["modern", "minimalist", "contemporary"]
        )
    }

    private func loadSourceCameraInfo() -> SourceCameraInfo {
        let folder = viewerPlyURL.deletingLastPathComponent()
        return CameraExifSidecar.loadSourceCameraInfo(
            roomFolder: folder,
            photoOrientation: exifOrientationHint
        )
    }

    private var exifOrientationHint: Int {
        switch photoOrientation {
        case .portrait: return 6
        case .landscape, .square: return 1
        }
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
