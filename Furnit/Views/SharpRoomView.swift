import SwiftUI
import UIKit
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

    /// PLY for MetalSplatter: `_classic.ply` when present (uchar RGB); else the room base `.ply` (no `_3dgs` in this path).
    private let viewerPlyURL: URL
    /// Classic PLY (front-rotated) when available; falls back to base.
    private let classicPlyURL: URL

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
        let classic = URL(fileURLWithPath: basePath.replacingOccurrences(of: ".ply", with: "_classic.ply"))
        let fm = FileManager.default

        if fm.fileExists(atPath: classic.path) {
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

    /// Bounds from the splat PLY (computed when loading; see `GaussianSplatView.onBoundsAvailable`).
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
    /// Frozen when the calibrate sheet opens so `applyCalibration` does not use a different per-frame pinhole height than the user saw.
    @State private var calibrationBaselineDetectedHeight: Float?

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

    /// User-facing Infinite Zoom toggle for the room viewer (matches Settings).
    @AppStorage("roomViewer.infiniteZoom") private var infiniteZoomEnabled: Bool = true

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
    /// Bridges splat ``GaussianSplatView/Coordinator`` for depth-buffer room raycast at save (and optional live refresh).
    @StateObject private var splatMeasurementHost = GaussianSplatMeasurementHost()
    /// Last successful depth-raycast room extents (scene units); refreshed after load and before save.
    @State private var raycastRoomDimensions: RoomRaycastDimensions?
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
        .navigationTitle("")
        .navigationBarTitleDisplayMode(.inline)
        .navigationBarBackButtonHidden(allowSave)
        .toolbar {
            ToolbarItem(placement: .principal) {
                VStack(spacing: 2) {
                    Text(navigationRoomMetersLine)
                        .font(.headline)
                        .lineLimit(2)
                        .minimumScaleFactor(0.55)
                        .multilineTextAlignment(.center)
                }
                .accessibilityElement(children: .combine)
            }
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
            // Share the classic/front-rotated PLY when available (matches what Metal renders).
            ShareSheet(activityItems: [classicPlyURL])
        }
        .onAppear {
            // Do not load YOLOE here — it peaks memory with WebKit (WKWebView) and can crash in WKWebViewConfiguration.
            // Load when the user turns on Furniture Fit (brain) instead.
            if photoOrientation == .landscape { OrientationLockManager.shared.lockToLandscape() } else { OrientationLockManager.shared.lockToPortrait() }
            logDebug("📐 [SharpRoomView] photoOrientation = \(photoOrientation)")
            loadPersistedRoomMetadataIfNeeded()
        }
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
                detectedFurnitureHeight = nil
                detectedFurnitureHeightAR = nil
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
        .onChange(of: detectedFurnitureHeight) { _, _ in updateRoomPlacementIntelligence() }
        .onChange(of: detectedFurnitureHeightAR) { _, _ in updateRoomPlacementIntelligence() }
        .onChange(of: roomModel) { _, _ in updateRoomPlacementIntelligence() }
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
            infiniteZoom: infiniteZoomEnabled,
            onBoundsAvailable: { bounds in
                DispatchQueue.main.async {
                    metalBounds = bounds
                    seedFrontWallDimensionsFromPlyBoundsIfNeeded()
                    let plyKind = viewerPlyURL.lastPathComponent.contains("_classic") ? "classic_ply" : "base_ply"
                    logPlyBoundsDiagnostic(
                        "Metal splat AABB (\(plyKind) file=\(viewerPlyURL.lastPathComponent)) su: " +
                        "W=\(String(format: "%.3f", bounds.width)) H=\(String(format: "%.3f", bounds.height)) D=\(String(format: "%.3f", bounds.depth)) " +
                        "X[\(String(format: "%.3f", bounds.minX)),\(String(format: "%.3f", bounds.maxX))] " +
                        "Y[\(String(format: "%.3f", bounds.minY)),\(String(format: "%.3f", bounds.maxY))] " +
                        "Z[\(String(format: "%.3f", bounds.minZ)),\(String(format: "%.3f", bounds.maxZ))]"
                    )
                    logSharpRoomDimensionApproaches(metalBounds: bounds, raycast: raycastRoomDimensions)
                }
            },
            measurementHost: splatMeasurementHost
        )
        .ignoresSafeArea()
        .onChange(of: isLoading) { _, loading in
            guard !loading else { return }
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.28) {
                if let m = splatMeasurementHost.measureRoom() {
                    raycastRoomDimensions = m
                    if let b = metalBounds {
                        logSharpRoomDimensionApproaches(metalBounds: b, raycast: m)
                    }
                }
                scheduleRoomGeometryExtractionIfNeeded()
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

    /// Fog trim + portrait/landscape caps for nav title / Furniture Fit (matches prior WebGL path).
    private func sharpRoomCompareDerivedDisplay(from b: RoomBounds) -> (trimW: Float, trimH: Float, capW: Float, capH: Float) {
        let fogTrim: Float = 0.15
        let trimmedWidth = max(0, b.width * (1 - fogTrim))
        let trimmedHeight = max(0, b.height * (1 - fogTrim))
        let maxRealisticWidth: Float = (photoOrientation == .portrait) ? 5.0 : 8.0
        let maxRealisticHeight: Float = (photoOrientation == .portrait) ? 3.5 : 3.2
        return (
            trimmedWidth,
            trimmedHeight,
            min(trimmedWidth, maxRealisticWidth),
            min(trimmedHeight, maxRealisticHeight)
        )
    }

    /// `[PLY_BOUNDS]` — depth raycast (Metal scene = loaded `viewerPlyURL`, usually **`classic_ply`**). Filter: `SHARP_ROOM_COMPARE`.
    private func logSharpRoomDimensionApproaches(metalBounds _: RoomBounds, raycast: RoomRaycastDimensions?) {
        let plyKind = viewerPlyURL.lastPathComponent.contains("_classic") ? "classic_ply" : "base_ply"
        let depthLine: String
        if let r = raycast {
            depthLine =
                "W=\(String(format: "%.3f", r.width)) H=\(String(format: "%.3f", r.height)) D=\(String(format: "%.3f", r.depth)) su"
        } else {
            depthLine = "pending"
        }
        logPlyBoundsDiagnostic(
            "SHARP_ROOM_COMPARE (5) depth_raycast (\(plyKind) render \(viewerPlyURL.lastPathComponent)) su: \(depthLine)"
        )

        // Other compare lines (sharp PLY write, meta, metal AABB, fog trim, caps, display depth, proportion) — disabled; re-enable for A/B logging.
        /*
        let dd = sharpRoomCompareDerivedDisplay(from: b)
        if let sw = sharpPlyAabbW, let sh = sharpPlyAabbH, let sd = sharpPlyAabbD {
            logPlyBoundsDiagnostic(
                "SHARP_ROOM_COMPARE (1) sharp_classic_ply_write su: W=\(String(format: "%.3f", sw)) H=\(String(format: "%.3f", sh)) D=\(String(format: "%.3f", sd))"
            )
        } else {
            logPlyBoundsDiagnostic("SHARP_ROOM_COMPARE (1) sharp_classic_ply_write su: n/a (opened from Home / preview / no generation bundle)")
        }
        if let metaW = savedRoomWidth, let metaH = savedRoomHeight {
            logPlyBoundsDiagnostic(
                "SHARP_ROOM_COMPARE (1b) saved_meta m: W=\(String(format: "%.3f", metaW)) H=\(String(format: "%.3f", metaH))"
            )
        }
        logPlyBoundsDiagnostic(
            "SHARP_ROOM_COMPARE (2) metal_splat_AABB classic_ply=from_loaded_file su: W=\(String(format: "%.3f", b.width)) H=\(String(format: "%.3f", b.height)) D=\(String(format: "%.3f", b.depth)) " +
            "X[\(String(format: "%.3f", b.minX)),\(String(format: "%.3f", b.maxX))] " +
            "Y[\(String(format: "%.3f", b.minY)),\(String(format: "%.3f", b.maxY))] " +
            "Z[\(String(format: "%.3f", b.minZ)),\(String(format: "%.3f", b.maxZ))]"
        )
        logPlyBoundsDiagnostic(
            "SHARP_ROOM_COMPARE (3) fog15pct_trim su: W=\(String(format: "%.3f", dd.trimW)) H=\(String(format: "%.3f", dd.trimH))"
        )
        let capLabel = (photoOrientation == .portrait) ? "portrait cap 5×3.5 m" : "landscape cap 8×3.2 m"
        logPlyBoundsDiagnostic(
            "SHARP_ROOM_COMPARE (4) capped_display_m: W=\(String(format: "%.3f", dd.capW)) H=\(String(format: "%.3f", dd.capH)) (\(capLabel))"
        )
        let depthM = displayRoomDepth
        logPlyBoundsDiagnostic(
            "SHARP_ROOM_COMPARE (4b) display_depth_m: \(String(format: "%.3f", depthM)) (saved .meta or splat Z span)"
        )
        if let pw = sourcePhotoPixelWidth, let ph = sourcePhotoPixelHeight, pw > 0, ph > 0 {
            let impliedH = dd.capW * Float(ph) / Float(pw)
            logPlyBoundsDiagnostic(
                "SHARP_ROOM_COMPARE (6) proportion_style: photo=\(pw)×\(ph)px " +
                "display_from_splat_cap W×H_m=\(String(format: "%.3f", dd.capW))×\(String(format: "%.3f", dd.capH)) " +
                "| if_full_photo_width_maps_to_display_W then_H_from_aspect_only_m=\(String(format: "%.3f", impliedH)) (vs capped H)"
            )
        } else {
            logPlyBoundsDiagnostic(
                "SHARP_ROOM_COMPARE (6) proportion_style: n/a (no source photo px — not from SinglePhoto SHARP flow)"
            )
        }
        */
    }

    /// The old splat-AABB fallback produced undersized saved dimensions (for example `0.697` instead of the
    /// live `2.7` depth-raycast height). Keep it commented out so save/reopen stays aligned with the preview raycast.
    private func seedFrontWallDimensionsFromPlyBoundsIfNeeded() {
        /*
        guard jsFrontWallWidth == nil, jsFrontWallHeight == nil, let b = effectiveBounds else { return }
        let dd = sharpRoomCompareDerivedDisplay(from: b)
        jsFrontWallWidth = dd.capW
        jsFrontWallHeight = dd.capH
        */
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

    private var roomModelMetersDimensions: (width: Float, height: Float, depth: Float)? {
        guard let roomModel else { return nil }
        let width = roomModel.widthMeters
        let height = roomModel.heightMeters
        let depth = roomModel.depthMeters
        guard width.isFinite, height.isFinite, depth.isFinite,
              width > 0.05, height > 0.05, depth > 0.05 else {
            return nil
        }
        return (width, height, depth)
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
        let size = roomModel.roomBounds.size
        guard size.x.isFinite, size.y.isFinite, size.z.isFinite,
              size.x > 0.05, size.y > 0.05, size.z > 0.05 else {
            return nil
        }
        return RoomRaycastDimensions(width: size.x, height: size.y, depth: size.z)
    }

    private var raycastHeightCalibratedMetersDimensions: (width: Float, height: Float, depth: Float)? {
        let scene = roomModelSceneDimensions ?? raycastRoomDimensions ?? savedSceneDimensions
        let referenceHeight = roomModelMetersDimensions?.height
            ?? savedRoomHeight
            ?? savedRoomModel?.roomHeight
            ?? 2.4
        guard let scene,
              scene.height.isFinite,
              scene.height > 0.05,
              referenceHeight.isFinite,
              referenceHeight > 1.5,
              referenceHeight < 4.5 else {
            return nil
        }

        let scale = referenceHeight / scene.height
        guard scale.isFinite, scale > 0.0001 else { return nil }
        return (
            width: scene.width * scale,
            height: referenceHeight,
            depth: scene.depth * scale
        )
    }

    private var resolvedRoomMetersDimensions: (width: Float, height: Float, depth: Float)? {
        let modelMeters = roomModelMetersDimensions
        let raycastScaled = raycastHeightCalibratedMetersDimensions

        if let modelMeters, let raycastScaled {
            let widthRatio = modelMeters.width / max(raycastScaled.width, 0.0001)
            let depthRatio = modelMeters.depth / max(raycastScaled.depth, 0.0001)
            let widthMismatch = widthRatio > 2.0 || widthRatio < 0.5
            let depthMismatch = depthRatio > 2.0 || depthRatio < 0.5

            if widthMismatch || depthMismatch {
                return (
                    width: raycastScaled.width,
                    height: modelMeters.height,
                    depth: raycastScaled.depth
                )
            }
        }

        return modelMeters
            ?? raycastScaled
    }

    /// Width/height/depth in meters for nav title, FurnitureFit, and save.
    /// The canonical live source is `roomModel`; scene-unit raycasts are diagnostic / fallback-only and must not bleed into meter fields.
    private var displayRoomWidth: Float {
        calibratedRoomWidth
            ?? resolvedRoomMetersDimensions?.width
            ?? savedRoomWidth
            ?? savedRoomModel?.roomWidth
            ?? 4.0
    }

    private var displayRoomHeight: Float {
        calibratedRoomHeight
            ?? resolvedRoomMetersDimensions?.height
            ?? savedRoomHeight
            ?? savedRoomModel?.roomHeight
            ?? 3.0
    }

    /// Depth in meters; never sourced from raw scene-unit raycasts.
    private var displayRoomDepth: Float {
        resolvedRoomMetersDimensions?.depth
            ?? savedRoomModel?.roomDepth
            ?? 4.0
    }

    /// Nav bar: show canonical room-model meters when available; otherwise fall back to scene-unit diagnostics while live extraction settles.
    private var navigationRoomMetersLine: String {
        if let meters = roomModelMetersDimensions {
            let display = resolvedRoomMetersDimensions ?? meters
            return String(
                format: "%.1f m × %.1f m × %.1f m (room model)",
                display.width, display.height, display.depth
            )
        }
        if allowSave {
            if let r = roomModelSceneDimensions ?? raycastRoomDimensions {
                let source = roomModelSceneDimensions != nil ? "room bounds" : "classic_ply depth raycast"
                return String(
                    format: "%.3f × %.3f × %.3f su (%@)",
                    r.width, r.height, r.depth, source
                )
            }
            return "Measuring room…"
        }
        if let scene = roomModelSceneDimensions {
            return String(
                format: "%.3f × %.3f × %.3f su (saved room bounds)",
                scene.width, scene.height, scene.depth
            )
        }
        if let m = savedRoomModel,
           let sw = m.roomSceneWidth, let sh = m.roomSceneHeight, let sd = m.roomSceneDepth {
            return String(
                format: "%.3f × %.3f × %.3f su (saved depth raycast)",
                sw, sh, sd
            )
        }
        return String(format: "%.1f m × %.1f m × %.1f m", displayRoomWidth, displayRoomHeight, displayRoomDepth)
    }

    /// Baseline W/H before calibration in meter space when available.
    private var sourceRoomWidth: Float {
        resolvedRoomMetersDimensions?.width
            ?? savedRoomWidth
            ?? savedRoomModel?.roomWidth
            ?? jsFrontWallWidth
            ?? raycastRoomDimensions?.width
            ?? 4.0
    }

    private var sourceRoomHeight: Float {
        resolvedRoomMetersDimensions?.height
            ?? savedRoomHeight
            ?? savedRoomModel?.roomHeight
            ?? jsFrontWallHeight
            ?? raycastRoomDimensions?.height
            ?? 3.0
    }

    /// Room dimensions for FurnitureFit — same chain as nav title / save.
    private var furnitureFitRoomWidth: Float { displayRoomWidth }

    private var furnitureFitRoomHeight: Float { displayRoomHeight }

    /// Scene-unit room for Furniture Fit ratios: prefer extracted room bounds, then persisted scene bounds, then live raycast.
    private var furnitureFitSceneDimensions: RoomRaycastDimensions? {
        if let scene = roomModelSceneDimensions { return scene }
        if !allowSave, let m = savedRoomModel,
           let w = m.roomSceneWidth, let h = m.roomSceneHeight, let d = m.roomSceneDepth {
            return RoomRaycastDimensions(width: w, height: h, depth: d)
        }
        if let r = raycastRoomDimensions { return r }
        if let m = savedRoomModel,
           let w = m.roomSceneWidth, let h = m.roomSceneHeight, let d = m.roomSceneDepth {
            return RoomRaycastDimensions(width: w, height: h, depth: d)
        }
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
                // Width may still come from the non-AR sizing path, but furniture height is
                // AR-only so pinhole/raycast cannot contaminate room calibration or placement.
                detectedFurnitureWidth = estimate.widthMeters
                if let arHeight = estimate.arHeightMeters,
                   arHeight.isFinite,
                   arHeight > 0.05 {
                    detectedFurnitureHeight = arHeight
                    detectedFurnitureHeightAR = arHeight
                } else {
                    logFurnitureFitSize(
                        "phase=viewer_height_skip reason=ar_only_height width_m=\(String(format: "%.3f", estimate.widthMeters))"
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
            }
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
                if let h = calibrationBaselineDetectedHeight ?? detectedFurnitureHeight {
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
              let detectedHeight = calibrationBaselineDetectedHeight ?? detectedFurnitureHeight,
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

    @ViewBuilder
    private var roomIntelligencePlacementCard: some View {
        if showingFurnitureFit,
           let dimensions = derivedDetectedFurnitureDimensionsForRoomIntelligence(),
           let fit = latestFitCheckResult {
            VStack(alignment: .leading, spacing: 4) {
                Button {
                    withAnimation(.easeInOut(duration: 0.18)) {
                        isPlacementIntelligenceExpanded.toggle()
                    }
                } label: {
                    HStack(spacing: 8) {
                        Text("Placement Intelligence")
                            .font(.caption.bold())
                            .foregroundColor(.white)
                        Spacer(minLength: 8)
                        Text(
                            fit.fitsInRoom
                                ? "\(max(fit.fitLocations.count, 1)) fit"
                                : "No fit"
                        )
                        .font(.caption2.bold())
                        .foregroundColor(fit.fitsInRoom ? .green : .red)
                        Image(systemName: isPlacementIntelligenceExpanded ? "chevron.up" : "chevron.down")
                            .font(.caption.bold())
                            .foregroundColor(.white.opacity(0.85))
                    }
                }
                .buttonStyle(.plain)

                if isPlacementIntelligenceExpanded {
                    Text(
                        String(
                            format: "Detected %.2f × %.2f × %.2f m",
                            dimensions.widthM,
                            dimensions.heightM,
                            dimensions.depthM
                        )
                    )
                    .font(.caption2)
                    .foregroundColor(.white.opacity(0.92))
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
                    if let aesthetic = latestAestheticScore {
                        Text(
                            String(
                                format: "Harmony %.2f (%@) · contrast %.2f · style fit %.2f",
                                aesthetic.harmonyScore,
                                aesthetic.harmonyType.rawValue,
                                aesthetic.contrastScore,
                                aesthetic.styleCompatibilityScore
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
                if latestFitCheckResult == nil {
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
                        VStack(alignment: .trailing, spacing: 8) {
                            roomIntelligencePlacementCardResetOnExit
                            if showRoomFurnitureCalibrate {
                                Button(action: { showFurnitureDimensionsInput = true }) {
                                    furnitureMeasurementPillContent(showTapHint: true)
                                }
                            } else {
                                furnitureMeasurementPillContent(showTapHint: false)
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
                            roomIntelligencePlacementCardResetOnExit
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
            if showFurnitureDimensionsInput, showRoomFurnitureCalibrate {
                calibrationOverlayView
                    .onAppear { calibrationBaselineDetectedHeight = detectedFurnitureHeight }
                    .onDisappear { calibrationBaselineDetectedHeight = nil }
            }
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

    /// Resolves the room thumbnail next to the classic PLY (jpg preferred, falls back to png / legacy).
    private func resolvedThumbnailURL(forClassicPly classicPly: URL) -> URL {
        let folder = classicPly.deletingLastPathComponent()
        let stem = classicPly.deletingPathExtension().lastPathComponent
        let base: String
        if stem.hasSuffix("_classic") {
            base = String(stem.dropLast("_classic".count))
        } else {
            base = stem
        }
        let fm = FileManager.default
        let jpg = folder.appendingPathComponent("\(base)_thumbnail.jpg")
        if fm.fileExists(atPath: jpg.path) { return jpg }
        let png = folder.appendingPathComponent("\(base)_thumbnail.png")
        if fm.fileExists(atPath: png.path) { return png }
        let legacy = folder.appendingPathComponent("thumbnail.png")
        if fm.fileExists(atPath: legacy.path) { return legacy }
        return jpg
    }

    @MainActor
    private func currentValidatedEnhancedMetadata() -> EnhancedRoomMetadata? {
        if let metadata = enhancedRoomMetadata,
           metadata.roomModel() != nil {
            return metadata
        }

        if let roomModel {
            let metadata = EnhancedRoomMetadata.from(roomModel: roomModel, preserving: enhancedRoomMetadata)
            enhancedRoomMetadata = metadata
            return metadata
        }

        return nil
    }

    @MainActor
    private func prepareEnhancedMetadataForSaveIfNeeded() async -> EnhancedRoomMetadata? {
        if let existing = currentValidatedEnhancedMetadata() {
            logDebug("📐 [SharpRoomView] Save: reusing existing room intelligence metadata")
            return existing
        }

        guard let depthQuery = splatMeasurementHost.depthQuery else {
            logDebug("📐 [SharpRoomView] Save: room intelligence unavailable — depth query missing")
            return nil
        }

        logDebug("📐 [SharpRoomView] Save: attempting room geometry extraction from live splat depth")
        splatMeasurementHost.requestRedrawForDepthMeasure()
        try? await Task.sleep(nanoseconds: 120_000_000)

        let cameraInfo = loadSourceCameraInfo()
        let usableColorReader: (any SplatColorReadable)? = {
            guard let colorReader = splatMeasurementHost.colorReader,
                  colorReader.supportsColorReadback else { return nil }
            return colorReader
        }()

        let engine = RoomGeometryEngine(
            depthQuery: depthQuery,
            colorReader: usableColorReader,
            cameraInfo: cameraInfo
        )

        do {
            let model = try engine.extractRoomModel()
            roomModel = model
            let metadata = EnhancedRoomMetadata.from(roomModel: model, preserving: enhancedRoomMetadata)
            enhancedRoomMetadata = metadata
            persistEnhancedRoomMetadataIfPossible(metadata)
            logDebug(
                "📐 [SharpRoomView] Save: extracted room intelligence W×H×D=\(String(format: "%.2f", model.widthMeters))×" +
                "\(String(format: "%.2f", model.heightMeters))×\(String(format: "%.2f", model.depthMeters))m"
            )
            return metadata
        } catch {
            logDebug("❌ [SharpRoomView] Save: room geometry extraction failed, continuing without enhanced metadata: \(error.localizedDescription)")
            return nil
        }
    }

    private func startSavingRoom() {
        guard !roomName.isEmpty else { return }

        let savedName = roomName
        logDebug("💾 [SharpRoomView] Starting room save: \(savedName)")

        Task {
            // Turn off Furniture Fit before depth measure + YOLO wall pass so camera/GPU aren’t contending with segmentation.
            await MainActor.run {
                if showingFurnitureFit {
                    logDebug("💾 [SharpRoomView] Save: stopping Furniture Fit before depth raycast")
                    showingFurnitureFit = false
                }
            }
            try? await Task.sleep(nanoseconds: 180_000_000)

            // Do not re-measure during save. The extra redraw/raycast here was producing unstable values like
            // `0.697` even when the preview had already measured the correct room height around `2.7`.
            // Use the cached preview raycast instead so save matches what the user saw before tapping Save.
            /*
            await MainActor.run {
                splatMeasurementHost.requestRedrawForDepthMeasure()
            }
            try? await Task.sleep(nanoseconds: 100_000_000)

            var sceneRaycast: RoomRaycastDimensions? = await MainActor.run {
                splatMeasurementHost.measureRoom()
            }
            if sceneRaycast == nil {
                sceneRaycast = await MainActor.run { raycastRoomDimensions }
                if sceneRaycast != nil {
                    logWallMeasurement("saveRoom depth-raycast: using cached scene units from an earlier frame")
                }
            }
            */
            let sceneRaycast: RoomRaycastDimensions? = await MainActor.run { raycastRoomDimensions }
            if sceneRaycast != nil {
                logWallMeasurement("saveRoom using cached preview scene units")
            }
            if let r = sceneRaycast {
                await MainActor.run { raycastRoomDimensions = r }
                logWallMeasurement("saveRoom scene-raycast su W×H×D=\(r.width)×\(r.height)×\(r.depth)")
                logDebug("📐 [SharpRoomView] Save: scene-raycast W×H×D=\(r.width)×\(r.height)×\(r.depth) su")
            } else {
                logWallMeasurement("saveRoom depth-raycast failed and no cached scene units")
            }

            await MainActor.run {
                withAnimation(.easeIn(duration: 0.3)) {
                    isSavingRoom = true
                    saveProgress = 0.0
                }
            }

            await MainActor.run { saveProgress = 0.12 }
            var roomW = await MainActor.run { displayRoomWidth }
            var roomH = await MainActor.run { displayRoomHeight }
            var roomD = await MainActor.run { displayRoomDepth }
            logWallMeasurement("saveRoom baseline display W×H×D=\(roomW)×\(roomH)×\(roomD)")

            /*
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
            if thumbImage == nil || yoloModel == nil {
                logWallMeasurement("saveRoom YOLO wall skipped (need thumbnail + YOLO model)")
            } else if let m = yoloModel, let img = thumbImage {
                let boundsSnapshot = await MainActor.run { effectiveBounds }
                if let measured = await WallMeasurementEstimator.measure(roomFolder: folder, thumbnail: img, model: m, photoOrientation: photoOrientation, plyBounds: boundsSnapshot) {
                    roomW = measured.widthMeters
                    roomH = measured.heightMeters
                    roomD = measured.depthMeters
                    logWallMeasurement("saveRoom YOLO wall W×H×D=\(roomW)×\(roomH)×\(roomD) mode=\(measured.calibrationMode)")
                    logDebug("📐 [SharpRoomView] YOLO wall measurement: \(roomW)×\(roomH)×\(roomD) m (\(measured.calibrationMode))")
                    await MainActor.run {
                        jsFrontWallWidth = roomW
                        jsFrontWallHeight = roomH
                    }
                } else {
                    logWallMeasurement("saveRoom YOLO measure nil — keeping display W×H×D=\(roomW)×\(roomH)×\(roomD)")
                }
            }
            */

            await MainActor.run { saveProgress = 0.32 }
            let metadataForSave = await prepareEnhancedMetadataForSaveIfNeeded()
            let modelMetersForSave = await MainActor.run { roomModelMetersDimensions }
            let modelSceneForSave = await MainActor.run { roomModelSceneDimensions }

            if let modelMetersForSave {
                roomW = modelMetersForSave.width
                roomH = modelMetersForSave.height
                roomD = modelMetersForSave.depth
                logWallMeasurement(
                    "saveRoom using room-model meters W×H×D=\(roomW)×\(roomH)×\(roomD)"
                )
                logDebug(
                    "📐 [SharpRoomView] Save: using room-model meters W×H×D=\(String(format: "%.2f", roomW))×" +
                    "\(String(format: "%.2f", roomH))×\(String(format: "%.2f", roomD))m"
                )
            } else if sceneRaycast != nil {
                logDebug(
                    "⚠️ [SharpRoomView] Save: room-model meters unavailable; keeping display-meter fallback while scene-raycast remains scene-only"
                )
            }

            await MainActor.run { saveProgress = 0.55 }
            await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
                modelManager.savePLY(
                    from: viewerPlyURL,
                    name: savedName,
                    photoOrientation: photoOrientation,
                    roomWidth: roomW,
                    roomHeight: roomH,
                    roomDepth: roomD,
                    roomSceneWidth: modelSceneForSave?.width ?? sceneRaycast?.width,
                    roomSceneHeight: modelSceneForSave?.height ?? sceneRaycast?.height,
                    roomSceneDepth: modelSceneForSave?.depth ?? sceneRaycast?.depth
                ) { success, error in
                    logDebug(success ? "✅ [SharpRoomView] Room saved" : "❌ [SharpRoomView] Save failed: \(error ?? "unknown")")
                    Task { @MainActor in
                        if success, let metadata = metadataForSave {
                            do {
                                try modelManager.saveEnhancedMetadata(metadata, forSavedRoomNamed: savedName, fileType: .ply)
                                logDebug("✅ [SharpRoomView] Save: enhanced metadata persisted for saved room")
                            } catch {
                                logDebug("❌ [SharpRoomView] Failed to save enhanced metadata for saved room: \(error.localizedDescription)")
                            }
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
                if let persistedRoomModel = metadata.roomModel() {
                    roomModel = persistedRoomModel
                    logDebug(
                        "📐 [SharpRoomView] Loaded enhanced metadata for saved room: " +
                        "sceneToMeters=\(String(format: "%.4f", metadata.sceneToMeters ?? 0))"
                    )
                    return
                }
                logDebug("📐 [SharpRoomView] Ignoring invalid saved-room enhanced metadata")
            }
        }

        if let metadata = modelManager.loadEnhancedMetadata(forRoomURL: viewerPlyURL) {
            enhancedRoomMetadata = metadata
            if let persistedRoomModel = metadata.roomModel() {
                roomModel = persistedRoomModel
                logDebug("📐 [SharpRoomView] Loaded enhanced metadata from live room sidecar")
            } else {
                logDebug("📐 [SharpRoomView] Ignoring invalid live-room enhanced metadata")
            }
        }
    }

    private func scheduleRoomGeometryExtractionIfNeeded() {
        guard savedRoomModel == nil else {
            logDebug("📐 [SharpRoomView] Skipping room geometry extraction on saved-room open")
            return
        }
        guard roomModel == nil,
              !isExtractingRoomGeometry,
              !isLoading else { return }

        DispatchQueue.main.asyncAfter(deadline: .now() + 0.15) {
            triggerRoomGeometryExtractionIfNeeded()
        }
    }

    private func triggerRoomGeometryExtractionIfNeeded(force: Bool = false) {
        guard savedRoomModel == nil else {
            logDebug("📐 [SharpRoomView] Ignoring room geometry extraction trigger for saved-room open")
            return
        }
        guard force || roomModel == nil else { return }
        guard !isExtractingRoomGeometry else { return }
        guard let depthQuery = splatMeasurementHost.depthQuery else {
            logDebug("❌ [SharpRoomView] Cannot extract room geometry — depth query unavailable")
            return
        }

        isExtractingRoomGeometry = true
        splatMeasurementHost.requestRedrawForDepthMeasure()

        Task { @MainActor in
            await Task.yield()

            let cameraInfo = loadSourceCameraInfo()
            let engine = RoomGeometryEngine(
                depthQuery: depthQuery,
                colorReader: splatMeasurementHost.colorReader,
                cameraInfo: cameraInfo
            )

            do {
                let model = try engine.extractRoomModel()
                roomModel = model
                let metadata = EnhancedRoomMetadata.from(roomModel: model, preserving: enhancedRoomMetadata)
                enhancedRoomMetadata = metadata
                persistEnhancedRoomMetadataIfPossible(metadata)
                logDebug(
                    "📐 [SharpRoomView] Room model extracted W×H×D=\(String(format: "%.2f", model.widthMeters))×" +
                    "\(String(format: "%.2f", model.heightMeters))×\(String(format: "%.2f", model.depthMeters))m"
                )
            } catch {
                logDebug("❌ [SharpRoomView] Room geometry extraction failed: \(error.localizedDescription)")
            }

            isExtractingRoomGeometry = false
        }
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

        let heightSource = detectedFurnitureHeightAR ?? detectedFurnitureHeight
        guard let height = heightSource,
              height.isFinite,
              height > 0.05 else { return nil }

        let estimatedDepth = max(0.25, min(width * 0.72, 1.4))
        return RoomFurnitureDimensions(widthM: width, heightM: height, depthM: estimatedDepth)
    }

    private func updateRoomPlacementIntelligence() {
        guard showingFurnitureFit,
              let roomModel,
              let furniture = derivedDetectedFurnitureDimensionsForRoomIntelligence() else {
            latestFitCheckResult = nil
            latestCornerPlacementSuggestions = []
            latestEstimatedFurnitureDepthMeters = nil
            latestAestheticScore = nil
            return
        }

        latestEstimatedFurnitureDepthMeters = furniture.depthM
        let fitEngine = FitCheckEngine(roomModel: roomModel)
        let fitResult = fitEngine.checkFit(furniture: furniture)
        let cornerPlacement = CornerPlacement(roomModel: roomModel)
        let suggestions = Array(cornerPlacement.suggestions(for: furniture).prefix(3))

        latestFitCheckResult = fitResult
        latestCornerPlacementSuggestions = suggestions

        let palette = roomModel.surfacePalette
        let roomStyleTags = inferredRoomStyleTags(from: palette)
        let furnitureProfile = heuristicFurnitureProfileForAesthetic(
            roomModel: roomModel,
            segmentedMeanSRGB: segmentedFurnitureMeanSRGB
        )
        let aestheticAdvisor = AestheticAdvisor(palette: palette, roomStyleTags: roomStyleTags)
        latestAestheticScore = aestheticAdvisor.evaluate(furniture: furnitureProfile)

        logDebug(
            "📐 [SharpRoomView] Placement intelligence updated " +
            "furniture=\(String(format: "%.2f", furniture.widthM))×\(String(format: "%.2f", furniture.heightM))×\(String(format: "%.2f", furniture.depthM))m " +
            "fits=\(fitResult.fitsInRoom) fitLocations=\(fitResult.fitLocations.count) cornerSuggestions=\(suggestions.count) " +
            "aesthetic_h=\(String(format: "%.2f", latestAestheticScore?.harmonyScore ?? 0))"
        )
    }

    /// Maps splat-sampled ``SurfacePalette`` material hints to advisor style tags.
    private func inferredRoomStyleTags(from palette: SurfacePalette) -> [String] {
        var tags = Set<String>()
        let layers = [palette.floor, palette.walls, palette.ceiling]
        for layer in layers {
            guard let layer else { continue }
            for hint in layer.materialHints {
                switch hint {
                case .wood: tags.formUnion(["rustic", "traditional"])
                case .tile: tags.insert("modern")
                case .concrete: tags.formUnion(["industrial", "modern"])
                case .fabric, .carpet: tags.formUnion(["traditional", "eclectic"])
                case .plaster: tags.formUnion(["modern", "scandinavian"])
                case .glass, .metal: tags.formUnion(["modern", "industrial"])
                case .unknown: break
                }
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
