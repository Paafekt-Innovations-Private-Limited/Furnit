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
    var showSaveAlert: Bool
    var showDiscardUnsavedAlert: Bool
    var showCalibrationRejectAlert: Bool
    var showWallCalibration: Bool
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
    @ObservedObject private var appState = AppStateManager.shared

    /// Exact selected PLY for MetalSplatter.
    private let viewerPlyURL: URL
    /// Whether this room should use SHARP classic orientation/rendering even without `_classic` in the file name.
    private let viewerUsesClassicPlyBehavior: Bool

    /// SHARP **`_classic.ply`** write-time AABB (scene units); set when pushing from ``SinglePhotoRoomViewer`` after generation.
    private let sharpPlyAabbW: Float?
    private let sharpPlyAabbH: Float?
    private let sharpPlyAabbD: Float?
    /// ROOM_DIMS_V7 pre-computed room dimensions in **metres** from SHARP generation (no PLY re-measurement needed).
    private let sharpRoomMetersW: Float?
    private let sharpRoomMetersH: Float?
    private let sharpRoomMetersD: Float?
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
        sharpRoomWidth: Float? = nil,
        sharpRoomHeight: Float? = nil,
        sharpRoomDepth: Float? = nil,
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
        self.sharpRoomMetersW = sharpRoomWidth
        self.sharpRoomMetersH = sharpRoomHeight
        self.sharpRoomMetersD = sharpRoomDepth
        self.sourcePhotoPixelWidth = sourcePhotoPixelWidth
        self.sourcePhotoPixelHeight = sourcePhotoPixelHeight

        self.viewerPlyURL = plyURL
        self.viewerUsesClassicPlyBehavior = self.viewerPlyURL.path.hasSuffix("_classic.ply") || (savedRoomModel?.isClassicPly ?? false)
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
    @State private var furnitureFitSegmentationMode: FurnitureFitSegmentationMode = .identifyOnly
    @State private var furnitureFitShowIdentifyLivePreview = true
    @State private var selectedFurnitureFitLabels: [String] = []
    @ObservedObject private var yoloeService = YOLOEModelService.shared

    // JS-measured front wall dimensions (from actual splat bounds)
    @State private var jsFrontWallWidth: Float?
    @State private var jsFrontWallHeight: Float?

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
    @State private var isMeasuringRoomDimensions = false
    @State private var saveProgress: Double = 0.0
    @State private var saveProgressStatusText = L10n.RoomViewer.savingRoomEllipsis
    @State private var savingTimer: Timer?
    @State private var roomMeasurementTask: Task<Void, Never>?
    @State private var backgroundRoomMeasurementTask: Task<Void, Never>?
    @State private var showSaveAlert = false
    @State private var saveAlertMessage = ""
    @State private var saveWasSuccessful = false
    @State private var showDiscardUnsavedAlert = false
    @State private var isDismissing = false
    @State private var showRoomNameInput = false
    @State private var roomName = ""
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
    @State private var persistedSplatLoadHint: SplatLoadHint?
    @State private var roomModel: RoomModel?
    @State private var enhancedRoomMetadata: EnhancedRoomMetadata?
    @State private var isExtractingRoomGeometry = false
    @State private var didLoadPersistedRoomMetadata = false
    @State private var latestFitCheckResult: FitCheckResult?
    @State private var latestCornerPlacementSuggestions: [CornerPlacementSuggestion] = []
    @State private var latestEstimatedFurnitureDepthMeters: Float?
    @State private var latestAestheticScore: AestheticScore?
    @State private var brainArAssistedSizingEnabled = false
    /// Mean sRGB (0…1) from composited YOLOE cutout pixels; drives ``FurnitureProfile.primaryColor`` when set.
    @State private var segmentedFurnitureMeanSRGB: SIMD3<Float>?
    /// Collapsed: round pill icon; expanded: detail card above the pill.
    @State private var isPlacementIntelligenceExpanded = false
    /// Pinch hint (top-trailing): icon always visible; text shows on load and when tapped, auto-hides after 3s.
    @State private var pinchHintExplanationVisible = false
    @State private var pinchHintHideTextTask: Task<Void, Never>?
    /// Brain hint (above brain button): text auto-hides after 3s; tap icon always stays; tap toggles text.
    @State private var brainHintExplanationVisible = false
    @State private var brainHintHideTextTask: Task<Void, Never>?
    /// Snapshot hint (above camera button): same behavior as ``brainHintExplanationVisible``.
    @State private var snapshotHintExplanationVisible = false
    @State private var snapshotHintHideTextTask: Task<Void, Never>?
    /// Camera sizing hint (under the left controls): explains what the camera/viewfinder button does.
    @State private var cameraSizingHintExplanationVisible = false
    @State private var cameraSizingHintHideTextTask: Task<Void, Never>?
    @State private var cameraSizingHintRequiresBrain = false
    @State private var roomDimensionsHintVisible = false
    @State private var roomDimensionsHintHideTask: Task<Void, Never>?
    @State private var showFullVideoWithIdentifications = false
    @State private var fullVideoFurnitureTapHintVisible = false
    @State private var tapHintColorIndex: Int = 0
    private let tapHintColors: [Color] = [.yellow, .cyan, .orange, .green, .pink]
    @State private var tapHintColorTimer: Timer?
    @State private var measuredRoomDimensions: MeasuredPlyRoomDimensions?
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
    private var navigationBarRoomMeasurementPrincipal: some View {
        HStack(spacing: 12) {
            Button {
                if let activeDimensions = activeRoomMetersDimensions {
                    logDebug(
                        "[ROOM_DIMS][RULER] FILE=\(viewerPlyURL.lastPathComponent) " +
                        "SOURCE=\(activeRoomMetersDimensionsSource) " +
                        "W=\(String(format: "%.4f", activeDimensions.width)) " +
                        "H=\(String(format: "%.4f", activeDimensions.height)) " +
                        "D=\(String(format: "%.4f", activeDimensions.depth))"
                    )
                } else {
                    logDebug("[ROOM_DIMS][RULER] FILE=\(viewerPlyURL.lastPathComponent) SOURCE=\(activeRoomMetersDimensionsSource) unavailable")
                }
                guard canPresentRoomDimensionsAlert else {
                    logDebug("[ROOM_DIMS][RULER] ALERT_SKIPPED file=\(viewerPlyURL.lastPathComponent) reason=other_modal_active")
                    return
                }
                if hasCalculatedRoomMeasurements {
                    logDebug("[ROOM_DIMS][RULER] FILE=\(viewerPlyURL.lastPathComponent) USING_EXISTING source=\(activeRoomMetersDimensionsSource)")
                    onRoomDimensionsIconTapped()
                } else {
                    logDebug("[ROOM_DIMS][RULER] FILE=\(viewerPlyURL.lastPathComponent) FALLBACK=START_ASYNC_MEASURE source=\(activeRoomMetersDimensionsSource)")
                    startAsyncRoomMeasurementForRuler()
                }
            } label: {
                Image(systemName: "ruler.fill")
                    .symbolRenderingMode(.hierarchical)
                    .font(.system(size: 18, weight: .medium))
                    .foregroundStyle(.primary)
            }
            .buttonStyle(.plain)
            .disabled(!canPresentRoomDimensionsAlert || isMeasuringRoomDimensions)
            .accessibilityLabel(L10n.RoomViewer.checkMeasurement)

            Button(action: onPinchHintIconTapped) {
                Image(systemName: "hand.pinch.fill")
                    .symbolRenderingMode(.hierarchical)
                    .font(.system(size: 18, weight: .medium))
                    .foregroundStyle(.primary)
            }
            .buttonStyle(.plain)
            .accessibilityLabel(pinchHintAccessibilityLabel)

            Button(action: displayAllGestureHelpers) {
                Image(systemName: "hand.tap.fill")
                    .font(.system(size: 16, weight: .medium))
                    .foregroundStyle(.primary)
            }
            .buttonStyle(.plain)
            .accessibilityLabel(L10n.RoomViewer.displayAllHelpers)

            if !selectedFurnitureFitLabels.isEmpty {
                Button {
                    NotificationCenter.default.post(name: NSNotification.Name("FurnitureFitClearSelectedObjects"), object: nil)
                } label: {
                    Text(selectedFurnitureChipTitle)
                        .font(.caption2)
                        .fontWeight(.semibold)
                        .foregroundColor(.white)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(Capsule().fill(Color.black.opacity(0.72)))
                }
                .buttonStyle(.plain)
                .transition(.opacity)
            }
        }
    }

    private var selectedFurnitureChipTitle: String {
        let labels = selectedFurnitureFitLabels
        if labels.count == 1 { return labels[0] }
        if labels.count == 2 { return "\(labels[0]), \(labels[1])" }
        return "\(labels.count) selected"
    }

    private func handleSharpRoomBackTap() {
        if allowSave {
            if saveWasSuccessful {
                dismiss()
            } else {
                showDiscardUnsavedAlert = true
            }
        } else {
            dismiss()
        }
    }

    private var navigationBarBackButton: some View {
        Button(action: handleSharpRoomBackTap) {
            Image(systemName: "chevron.left")
        }
        .accessibilityLabel(L10n.Common.back)
    }

    private var navigationBarRecenterButton: some View {
        Button {
            splatMeasurementHost.recenterSharpRoomCamera()
            logDebug("🎯 [SharpRoomView] Recenter (toolbar)")
        } label: {
            Image(systemName: "viewfinder")
                .font(.title3)
        }
        .disabled(isLoading)
        .accessibilityLabel(L10n.RoomViewer.recenterView)
    }

    private var navigationBarFullVideoIdentificationsButton: some View {
        Button {
            showFullVideoWithIdentifications.toggle()
            if showFullVideoWithIdentifications {
                if showingFurnitureFit {
                    presentFullVideoFurnitureTapHintIfNeeded()
                }
            } else {
                dismissFullVideoFurnitureTapHint()
                if furnitureFitSegmentationMode == .segmentSelected {
                    furnitureFitSegmentationMode = .identifyOnly
                    furnitureFitShowIdentifyLivePreview = true
                }
            }
        } label: {
            Image(systemName: "text.viewfinder")
                .symbolVariant(showFullVideoWithIdentifications ? .fill : .none)
                .font(.title3)
        }
        .buttonStyle(.plain)
        .disabled(isLoading)
        .accessibilityLabel(L10n.Settings.fullVideoWithIdentifications)
        .accessibilityHint(L10n.Settings.fullVideoWithIdentificationsDescription)
        .accessibilityAddTraits(showFullVideoWithIdentifications ? .isSelected : [])
    }

    private var navigationBarSaveButton: some View {
        Button(action: { showRoomNameInput = true }) {
            Image(systemName: "square.and.arrow.down")
                .font(.title3)
        }
        .disabled(isLoading)
        .accessibilityLabel(L10n.RoomViewer.saveRoom)
    }

    private var navigationBarTrailingControls: some View {
        HStack(spacing: 14) {
            navigationBarRecenterButton
            if showingFurnitureFit {
                navigationBarFullVideoIdentificationsButton
            }
            if canOfferBrainArAssist, showingFurnitureFit {
                navigationBarARButton
                    .fixedSize(horizontal: true, vertical: true)
            }
            if allowSave {
                navigationBarSaveButton
            }
        }
    }

    private var fullVideoHelperButtonsToRight: Int {
        let arButtons = (canOfferBrainArAssist && showingFurnitureFit) ? 1 : 0
        let saveButtons = allowSave ? 1 : 0
        return arButtons + saveButtons
    }

    private var fullVideoToolbarHelperOverlay: some View {
        ZStack(alignment: .topTrailing) {
            Color.clear
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .allowsHitTesting(false)
            if showingFurnitureFit {
                VStack(alignment: .trailing, spacing: 4) {
                    Capsule()
                        .fill(Color.white.opacity(0.85))
                        .frame(width: 2, height: 14)
                        .padding(.trailing, 18)
                    Text(L10n.RoomViewer.fullVideoSelectionHelper)
                        .font(.caption2)
                        .foregroundColor(.white)
                        .multilineTextAlignment(.leading)
                        .fixedSize(horizontal: false, vertical: true)
                        .frame(maxWidth: 220, alignment: .leading)
                        .padding(8)
                        .background(RoundedRectangle(cornerRadius: 8).fill(Color.black.opacity(0.78)))
                }
                .padding(.top, 6)
                .padding(.trailing, 18 + CGFloat(fullVideoHelperButtonsToRight * 34))
                .transition(.opacity)
            }
        }
        .opacity(isCapturingSnapshot ? 0 : 1)
        .allowsHitTesting(false)
        .zIndex(106)
    }

    private func toggleBrainArAssistedSizingOrShowHint() {
        guard showingFurnitureFit else { return }
        brainArAssistedSizingEnabled.toggle()
    }

    /// AR sizing control in the navigation bar (replaces the share-PLY toolbar button).
    private var navigationBarARButton: some View {
        Button(action: toggleBrainArAssistedSizingOrShowHint) {
            Image(systemName: "arrow.up.left.and.arrow.down.right")
                .symbolRenderingMode(.hierarchical)
                .font(.system(size: 15, weight: .bold))
                .foregroundStyle(.white)
                .frame(width: 32, height: 32)
                .background(
                    Circle().fill(
                        brainArAssistedSizingEnabled
                            ? Color.green.opacity(0.9)
                            : Color.black.opacity(0.45)
                    )
                )
        }
        .buttonStyle(.plain)
        .disabled(isLoading)
        .accessibilityLabel(
            brainArAssistedSizingEnabled ? L10n.RoomViewer.arSizingDisable : L10n.RoomViewer.arSizingEnable
        )
    }

    /// Optional pinch / AR sizing hint copy sits below the top toolbar row when visible.
    private var topTrailingPinchTapAndSizingHintsOverlay: some View {
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
                        .frame(maxWidth: 220, alignment: .trailing)
                        .padding(8)
                        .background(RoundedRectangle(cornerRadius: 8).fill(Color.black.opacity(0.78)))
                        .transition(.opacity)
                }
                if canOfferBrainArAssist, cameraSizingHintExplanationVisible,
                   showingFurnitureFit || cameraSizingHintRequiresBrain {
                    Text(cameraSizingHintText)
                        .font(.caption2)
                        .foregroundColor(.white)
                        .multilineTextAlignment(.center)
                        .fixedSize(horizontal: false, vertical: true)
                        .frame(maxWidth: 200)
                        .padding(8)
                        .background(RoundedRectangle(cornerRadius: 8).fill(Color.black.opacity(0.78)))
                        .transition(.opacity)
                }
            }
            .padding(.top, 52)
            .padding(.trailing, 16)
        }
        .opacity(isCapturingSnapshot ? 0 : 1)
        .zIndex(101)
        .onAppear { restartPinchGestureHint() }
        .onDisappear {
            cancelPinchHintTasks()
            cancelCameraSizingHintTasks()
        }
    }

    @ToolbarContentBuilder
    private var sharpRoomToolbarContent: some ToolbarContent {
        ToolbarItem(placement: .navigationBarLeading) {
            navigationBarBackButton
        }
        ToolbarItem(placement: .principal) {
            navigationBarRoomMeasurementPrincipal
        }
        ToolbarItem(placement: .navigationBarTrailing) {
            navigationBarTrailingControls
        }
    }

    private var sharpRoomNavigationView: some View {
        sharpRoomBaseLayer
            .navigationBarHidden(isCapturingSnapshot)
            .navigationTitle("")
            .navigationBarTitleDisplayMode(.inline)
            // Always hide the system back: in landscape its touch target is often covered by the wide `.principal`
            // toolbar title: explicit leading `Back` stays tappable (saved list + new room).
            .navigationBarBackButtonHidden(true)
            .toolbar { sharpRoomToolbarContent }
    }

    private var sharpRoomSheetAndLifecycleView: some View {
        sharpRoomNavigationView
        .onAppear {
            // Preload YOLOE when the room opens (async; not at app startup). First brain tap stays snappy.
            yoloeService.ensureModelLoaded()
            if photoOrientation == .landscape { OrientationLockManager.shared.lockToLandscape() } else { OrientationLockManager.shared.lockToPortrait() }
            logDebug("📐 [SharpRoomView] photoOrientation = \(photoOrientation)")
            loadPersistedRoomMetadataIfNeeded()
            syncModalHeavyWorkPauseForSharpRoomUI()
            warmRoomMeasurementInBackgroundIfNeeded()
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
                if canOfferBrainArAssist {
                    showCameraSizingHint(requiresBrain: false)
                }
            } else {
                dismissFullVideoFurnitureTapHint()
                cancelCameraSizingHintTasks()
                cameraSizingHintExplanationVisible = false
                brainArAssistedSizingEnabled = false
                furnitureFitSegmentationMode = .identifyOnly
                furnitureFitShowIdentifyLivePreview = true
                selectedFurnitureFitLabels = []
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
        .onChange(of: realFurnitureHeight) { _, _ in updateRoomPlacementIntelligence() }
        .onChange(of: roomModel) { _, _ in updateRoomPlacementIntelligence() }
        .onChange(of: enhancedRoomMetadata) { _, _ in updateRoomPlacementIntelligence() }
        .onChange(of: brainArAssistedSizingEnabled) { _, enabled in
            if !enabled {
                detectedFurnitureHeightAR = nil
            }
            logFurnitureFitSize("phase=sharp_room_ar_opt_in enabled=\(enabled) active=\(showingFurnitureFit)")
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
                logDebug("📱 [SharpRoomView] Fresh SHARP room loaded")
            } else {
                logDebug("📱 [SharpRoomView] Saved room loaded")
            }
        }
        .onDisappear {
            autoEnableARTask?.cancel()
            autoEnableARTask = nil
            roomMeasurementTask?.cancel()
            roomMeasurementTask = nil
            backgroundRoomMeasurementTask?.cancel()
            backgroundRoomMeasurementTask = nil
            isMeasuringRoomDimensions = false
            saveProgressStatusText = L10n.RoomViewer.savingRoomEllipsis
            cancelPinchHintTasks()
            cancelBrainHintTasks()
            cancelCameraSizingHintTasks()
            cancelRoomDimensionsHintTasks()
            dismissFullVideoFurnitureTapHint()
            OrientationLockManager.shared.unlock()
            splatMeasurementHost.setModalHeavyWorkPaused(false)
            SHARPService.shared.releaseResources()
            yoloeService.releaseResources()
        }
    }

    private var sharpRoomAlertsAndOverlayView: some View {
        sharpRoomSheetAndLifecycleView
        .alert(L10n.RoomViewer.saveRoom, isPresented: $showRoomNameInput) {
            TextField(L10n.RoomViewer.roomName, text: $roomName)
                .autocorrectionDisabled(true)
            Button(L10n.Common.cancel, role: .cancel) { }
            Button(L10n.Common.save) { startSavingRoom() }
                .disabled(roomName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
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
                cancelSnapshotHintTasks()
                cancelCameraSizingHintTasks()
                cancelRoomDimensionsHintTasks()
            } else {
                restartBrainGestureHint()
                restartSnapshotGestureHint()
            }
        }
        // Omit `.leading` so the interactive pop gesture is not deferred behind splat gestures (saved rooms).
        .defersSystemGestures(on: [.top, .trailing])
        .disableBackSwipeIf(allowSave)
    }

    private var sharpRoomBody: some View {
        sharpRoomAlertsAndOverlayView
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
            initialSharpRoomYaw: initialSharpRoomYaw,
            cachedSplatLoadHint: persistedSplatLoadHint,
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
            onSplatLoadHintAvailable: { hint in
                DispatchQueue.main.async {
                    persistedSplatLoadHint = hint
                    if metalBounds == nil {
                        metalBounds = hint.fullRoomBounds
                    }
                    do {
                        try modelManager.saveSplatLoadHint(hint, nextTo: viewerPlyURL)
                        logDebug("⏱️ [SplatLoad] metadata_persisted source=SharpRoomView file=\(viewerPlyURL.lastPathComponent) type=hint")
                    } catch {
                        logDebug("❌ [SharpRoomView] Failed to persist splat load hint: \(error.localizedDescription)")
                    }
                }
            },
            measurementHost: splatMeasurementHost
        )
        .ignoresSafeArea()
        .onChange(of: isLoading) { _, loading in
            guard !loading else { return }
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.28) {
                logDebug("📐 [SharpRoomView] Room geometry extraction remains disabled")
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
        logDebug("📐 [SharpRoomView] Skipped legacy SHARP room dimension comparison logging")
    }

    private func seedFrontWallDimensionsFromPlyBoundsIfNeeded() {}

    // MARK: - Gesture hint chips (pinch, brain, snapshot)

    private func toggleFurnitureFit() {
        if showingFurnitureFit {
            dismissFullVideoFurnitureTapHint()
            showingFurnitureFit = false
        } else {
            showFullVideoWithIdentifications = false
            furnitureFitInitialSegmentationDone = false
            furnitureFitSegmentationMode = .identifyOnly
            furnitureFitShowIdentifyLivePreview = true
            selectedFurnitureFitLabels = []
            SHARPService.shared.releaseResources()
            showingFurnitureFit = true
        }
    }

    private func dismissFullVideoFurnitureTapHint() {
        fullVideoFurnitureTapHintVisible = false
        tapHintColorTimer?.invalidate()
        tapHintColorTimer = nil
    }

    private func presentFullVideoFurnitureTapHintIfNeeded() {
        guard showFullVideoWithIdentifications else { return }
        tapHintColorIndex = 0
        fullVideoFurnitureTapHintVisible = true
        tapHintColorTimer?.invalidate()
        tapHintColorTimer = Timer.scheduledTimer(withTimeInterval: 2.0, repeats: true) { _ in
            DispatchQueue.main.async { tapHintColorIndex += 1 }
        }
    }

    private func activateSelectedFurnitureSegmentation() {
        if furnitureFitSegmentationMode == .segmentSelected {
            furnitureFitSegmentationMode = .identifyOnly
            furnitureFitShowIdentifyLivePreview = true
            return
        }
        guard canSegmentSelectedFurniture else { return }
        furnitureFitSegmentationMode = .segmentSelected
        dismissFullVideoFurnitureTapHint()
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

    private func cancelCameraSizingHintTasks() {
        cameraSizingHintHideTextTask?.cancel()
        cameraSizingHintHideTextTask = nil
    }

    private func cancelRoomDimensionsHintTasks() {
        roomDimensionsHintHideTask?.cancel()
        roomDimensionsHintHideTask = nil
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

    private func cancelSnapshotHintTasks() {
        snapshotHintHideTextTask?.cancel()
        snapshotHintHideTextTask = nil
    }

    private func scheduleSnapshotHintTextAutoHide(seconds: UInt64 = 3) {
        snapshotHintHideTextTask?.cancel()
        snapshotHintHideTextTask = Task { @MainActor in
            try? await Task.sleep(for: .seconds(seconds))
            guard !Task.isCancelled else { return }
            snapshotHintExplanationVisible = false
        }
    }

    private func restartSnapshotGestureHint() {
        cancelSnapshotHintTasks()
        snapshotHintExplanationVisible = true
        scheduleSnapshotHintTextAutoHide(seconds: 3)
    }

    private func displayAllGestureHelpers() {
        restartPinchGestureHint()
        restartBrainGestureHint()
        restartSnapshotGestureHint()
        showCameraSizingHint(requiresBrain: !showingFurnitureFit)
        roomDimensionsHintVisible = true
        scheduleRoomDimensionsHintAutoHide(seconds: 3)
    }

    private func onSnapshotHintIconTapped() {
        cancelSnapshotHintTasks()
        snapshotHintExplanationVisible.toggle()
        if snapshotHintExplanationVisible {
            scheduleSnapshotHintTextAutoHide(seconds: 3)
        }
    }

    private func scheduleCameraSizingHintAutoHide(seconds: UInt64 = 3) {
        cameraSizingHintHideTextTask?.cancel()
        cameraSizingHintHideTextTask = Task { @MainActor in
            try? await Task.sleep(for: .seconds(seconds))
            guard !Task.isCancelled else { return }
            cameraSizingHintExplanationVisible = false
        }
    }

    private func showCameraSizingHint(requiresBrain: Bool) {
        cancelCameraSizingHintTasks()
        cameraSizingHintRequiresBrain = requiresBrain
        cameraSizingHintExplanationVisible = true
        scheduleCameraSizingHintAutoHide(seconds: 3)
    }

    private func onCameraSizingHintIconTapped() {
        cancelCameraSizingHintTasks()
        cameraSizingHintRequiresBrain = false
        cameraSizingHintExplanationVisible.toggle()
        if cameraSizingHintExplanationVisible {
            scheduleCameraSizingHintAutoHide(seconds: 3)
        }
    }

    private func scheduleRoomDimensionsHintAutoHide(seconds: UInt64 = 3) {
        roomDimensionsHintHideTask?.cancel()
        roomDimensionsHintHideTask = Task { @MainActor in
            try? await Task.sleep(for: .seconds(seconds))
            guard !Task.isCancelled else { return }
            roomDimensionsHintVisible = false
        }
    }

    private func onRoomDimensionsIconTapped() {
        cancelRoomDimensionsHintTasks()
        roomDimensionsHintVisible.toggle()
        if roomDimensionsHintVisible {
            scheduleRoomDimensionsHintAutoHide(seconds: 3)
        }
    }

    private func startAsyncRoomMeasurementForRuler() {
        guard !isMeasuringRoomDimensions else { return }
        backgroundRoomMeasurementTask?.cancel()
        backgroundRoomMeasurementTask = nil
        roomMeasurementTask?.cancel()
        roomMeasurementTask = Task {
            let sourceBefore = activeRoomMetersDimensionsSource
            logDebug("[ROOM_DIMS][RULER] FILE=\(viewerPlyURL.lastPathComponent) START source_before=\(sourceBefore)")
            await MainActor.run {
                isMeasuringRoomDimensions = true
                saveProgressStatusText = L10n.RoomViewer.measuringRoom
            }
            let measured = await modelManager.measureRoomDimensionsAsync(
                forPly: viewerPlyURL,
                treatAsClassicPly: viewerUsesClassicPlyBehavior
            )
            guard !Task.isCancelled else {
                await MainActor.run {
                    isMeasuringRoomDimensions = false
                    saveProgressStatusText = L10n.RoomViewer.savingRoomEllipsis
                }
                return
            }
            if let measured,
               let savedRoomModel,
               savedRoomModel.fileType == .ply {
                try? modelManager.mergeRoomDimensionsIntoSavedRoomMetadata(
                    fileName: savedRoomModel.fileName,
                    modelFileExtension: "ply",
                    roomWidth: measured.width,
                    roomHeight: measured.height,
                    roomDepth: measured.depth
                )
            }
            await MainActor.run {
                if let measured {
                    measuredRoomDimensions = measured
                    updateRoomPlacementIntelligence()
                    logDebug(
                        "[ROOM_DIMS][RULER] FILE=\(viewerPlyURL.lastPathComponent) " +
                        "SOURCE=ASYNC_V7_PLY_MEASURE " +
                        "APPROACH=\(measured.approach.uppercased()) SHOT=\(measured.shotType) " +
                        "HAS_FOCAL=\(measured.usedFocal) TILT_DEG=\(String(format: "%.2f", measured.tiltDegrees)) " +
                        "TILT_RELIABLE=\(measured.tiltReliable) CUBOID_RATIO=\(String(format: "%.4f", measured.cuboidRatio)) " +
                        "THRESHOLD=\(String(format: "%.4f", measured.cuboidThreshold)) " +
                        "FILL_W=\(String(format: "%.4f", measured.fillWidth)) BLEND=\(String(format: "%.4f", measured.blend)) " +
                        "source_after=\(activeRoomMetersDimensionsSource) " +
                        "W=\(String(format: "%.4f", measured.width)) " +
                        "H=\(String(format: "%.4f", measured.height)) " +
                        "D=\(String(format: "%.4f", measured.depth))"
                    )
                } else {
                    logDebug("[ROOM_DIMS][RULER] FILE=\(viewerPlyURL.lastPathComponent) SOURCE=ASYNC_V7_PLY_MEASURE source_after=\(activeRoomMetersDimensionsSource) unavailable")
                }
                isMeasuringRoomDimensions = false
                saveProgressStatusText = L10n.RoomViewer.savingRoomEllipsis
                roomDimensionsHintVisible = true
                scheduleRoomDimensionsHintAutoHide(seconds: 3)
            }
        }
    }

    private func warmRoomMeasurementInBackgroundIfNeeded() {
        guard measuredRoomDimensions == nil else {
            logDebug("[ROOM_DIMS][BACKGROUND] FILE=\(viewerPlyURL.lastPathComponent) SKIP reason=measured_already_available source=\(activeRoomMetersDimensionsSource)")
            return
        }
        guard savedRoomStrictMeters == nil else {
            logDebug("[ROOM_DIMS][BACKGROUND] FILE=\(viewerPlyURL.lastPathComponent) SKIP reason=saved_meta_strict_available source=\(activeRoomMetersDimensionsSource)")
            return
        }
        guard sharpGenerationRoomMeters == nil else {
            logDebug("[ROOM_DIMS][BACKGROUND] FILE=\(viewerPlyURL.lastPathComponent) SKIP reason=sharp_v7_dims_available source=\(activeRoomMetersDimensionsSource)")
            return
        }
        guard persistedEnhancedRoomMeters == nil else {
            logDebug("[ROOM_DIMS][BACKGROUND] FILE=\(viewerPlyURL.lastPathComponent) SKIP reason=enhanced_metadata_room_model_available source=\(activeRoomMetersDimensionsSource)")
            return
        }
        backgroundRoomMeasurementTask?.cancel()
        backgroundRoomMeasurementTask = Task {
            logDebug("[ROOM_DIMS][BACKGROUND] FILE=\(viewerPlyURL.lastPathComponent) START source_before=\(activeRoomMetersDimensionsSource)")
            let measured = await modelManager.measureRoomDimensionsAsync(
                forPly: viewerPlyURL,
                treatAsClassicPly: viewerUsesClassicPlyBehavior
            )
            guard !Task.isCancelled else { return }
            await MainActor.run {
                if measuredRoomDimensions == nil {
                    measuredRoomDimensions = measured
                    updateRoomPlacementIntelligence()
                }
                if let measured {
                    logDebug(
                        "[ROOM_DIMS][BACKGROUND] FILE=\(viewerPlyURL.lastPathComponent) " +
                        "APPROACH=\(measured.approach.uppercased()) SHOT=\(measured.shotType) " +
                        "HAS_FOCAL=\(measured.usedFocal) TILT_DEG=\(String(format: "%.2f", measured.tiltDegrees)) " +
                        "TILT_RELIABLE=\(measured.tiltReliable) CUBOID_RATIO=\(String(format: "%.4f", measured.cuboidRatio)) " +
                        "THRESHOLD=\(String(format: "%.4f", measured.cuboidThreshold)) " +
                        "FILL_W=\(String(format: "%.4f", measured.fillWidth)) BLEND=\(String(format: "%.4f", measured.blend)) " +
                        "DONE source_after=\(activeRoomMetersDimensionsSource) " +
                        "W=\(String(format: "%.4f", measured.width)) " +
                        "H=\(String(format: "%.4f", measured.height)) " +
                        "D=\(String(format: "%.4f", measured.depth))"
                    )
                } else {
                    logDebug("[ROOM_DIMS][BACKGROUND] FILE=\(viewerPlyURL.lastPathComponent) DONE source_after=\(activeRoomMetersDimensionsSource) unavailable")
                }
                backgroundRoomMeasurementTask = nil
            }
        }
    }

    private var pinchHintAccessibilityLabel: String {
        L10n.RoomViewer.pinchGestureHintExplanation + " " + L10n.RoomViewer.gestureHintToggleAccessibility
    }

    private var brainHintAccessibilityLabel: String {
        L10n.RoomViewer.brainGestureHintExplanation + " " + L10n.RoomViewer.gestureHintToggleAccessibility
    }

    private var canSegmentSelectedFurniture: Bool {
        showingFurnitureFit && !selectedFurnitureFitLabels.isEmpty
    }

    private var snapshotHintAccessibilityLabel: String {
        L10n.RoomViewer.snapshotGestureHintExplanation + " " + L10n.RoomViewer.gestureHintToggleAccessibility
    }

    private var cameraSizingHintAccessibilityLabel: String {
        cameraSizingHintText + " " + L10n.RoomViewer.gestureHintToggleAccessibility
    }

    private var cameraSizingHintText: String {
        cameraSizingHintRequiresBrain
            ? L10n.RoomViewer.arFurnitureSizingRequiresBrainHint
            : L10n.RoomViewer.arFurnitureSizingHint
    }

    private var roomDimensionsHintText: String {
        if let d = activeRoomMetersDimensions {
            return L10n.RoomViewer.roomDimensionsWHDAIChip(
                width: d.width,
                height: d.height,
                depth: d.depth
            )
        }
        return "ROOM_DIMS unavailable"
    }

    private var canOfferBrainArAssist: Bool {
        QualitySettings.supportsLiDARSceneDepth &&
            appState.qualitySettings.furnitureFitARDepthCompanionRuntimeActive
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

    /// D-pad cluster (top-left). Pinch + tap hints live in ``topTrailingPinchTapAndSizingHintsOverlay``.
    private var cameraButtonsOverlay: some View {
        ZStack(alignment: .topLeading) {
            Color.clear
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .allowsHitTesting(false)
            VStack(alignment: .leading, spacing: 10) {
                cameraDPadCluster
                .padding(.leading, 12)
                .padding(.top, 12)
                if photoOrientation == .landscape {
                    Spacer(minLength: 0)
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        }
        .opacity(isCapturingSnapshot ? 0 : 1)
        .zIndex(102)
    }

    private var roomDimensionsHintOverlay: some View {
        ZStack(alignment: .top) {
            Color.clear
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .allowsHitTesting(false)
            VStack(spacing: 0) {
                if roomDimensionsHintVisible {
                    Text(roomDimensionsHintText)
                        .font(.caption2)
                        .foregroundColor(.white)
                        .multilineTextAlignment(.center)
                        .fixedSize(horizontal: false, vertical: true)
                        .frame(maxWidth: 220)
                        .padding(8)
                        .background(RoundedRectangle(cornerRadius: 8).fill(Color.black.opacity(0.78)))
                        .transition(.opacity)
                }
            }
            .padding(.top, 12)
            .onDisappear { cancelRoomDimensionsHintTasks() }
        }
        .opacity(isCapturingSnapshot ? 0 : 1)
        .zIndex(104)
    }

    private var fullVideoFurnitureTapHintOverlay: some View {
        ZStack(alignment: .top) {
            Color.clear
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .allowsHitTesting(false)
            VStack(spacing: 0) {
                if fullVideoFurnitureTapHintVisible {
                    Text(L10n.RoomViewer.fullVideoFurnitureTapHint)
                        .font(.caption2)
                        .fontWeight(.semibold)
                        .foregroundColor(tapHintColors[tapHintColorIndex % tapHintColors.count])
                        .multilineTextAlignment(.center)
                        .fixedSize(horizontal: false, vertical: true)
                        .frame(maxWidth: 260)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(RoundedRectangle(cornerRadius: 8).fill(Color.black.opacity(0.78)))
                        .transition(.opacity)
                        .animation(.easeInOut(duration: 0.6), value: tapHintColorIndex)
                }
            }
            .padding(.top, roomDimensionsHintVisible ? 56 : 12)
        }
        .allowsHitTesting(false)
        .opacity(isCapturingSnapshot ? 0 : 1)
        .zIndex(105)
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
        }
        .onAppear { restartBrainGestureHint() }
        .onDisappear { cancelBrainHintTasks() }
    }

    /// Text + tap icon above the snapshot camera; mirrors ``brainGestureHintColumn``.
    private var snapshotGestureHintColumn: some View {
        VStack(alignment: .center, spacing: 6) {
            if snapshotHintExplanationVisible {
                Text(L10n.RoomViewer.snapshotGestureHintExplanation)
                    .font(.caption2)
                    .foregroundColor(.white)
                    .multilineTextAlignment(.center)
                    .fixedSize(horizontal: false, vertical: true)
                    .frame(maxWidth: 200)
                    .padding(8)
                    .background(RoundedRectangle(cornerRadius: 8).fill(Color.black.opacity(0.78)))
                    .transition(.opacity)
            }
        }
        .onAppear { restartSnapshotGestureHint() }
        .onDisappear { cancelSnapshotHintTasks() }
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

    @ViewBuilder
    private var snapshotButtonWithHintAbove: some View {
        VStack(alignment: .center, spacing: 6) {
            snapshotGestureHintColumn
            Button(action: { takeScreenshot() }) {
                Image(systemName: "camera.fill")
                    .font(.system(size: 28))
                    .foregroundColor(.white)
                    .frame(width: 60, height: 60)
                    .background(Circle().fill(Color.blue).shadow(radius: 5))
            }
            .disabled(isLoading)
        }
    }

    @ViewBuilder
    private var segmentButton: some View {
        if showingFurnitureFit && showFullVideoWithIdentifications {
            Button(action: activateSelectedFurnitureSegmentation) {
                Text(
                    furnitureFitSegmentationMode == .segmentSelected
                        ? L10n.RoomViewer.stopSegmentationAction
                        : L10n.RoomViewer.segmentFurnitureAction
                )
                    .font(.system(size: 16, weight: .semibold))
                    .foregroundColor(.white)
                    .padding(.horizontal, 16)
                    .frame(height: 44)
                    .background(
                        Capsule().fill(
                            canSegmentSelectedFurniture
                                ? (furnitureFitSegmentationMode == .segmentSelected ? Color.green : Color.orange)
                                : Color.black.opacity(0.45)
                        )
                    )
                    .shadow(radius: 4)
            }
            .buttonStyle(.plain)
            .disabled(isLoading || (furnitureFitSegmentationMode != .segmentSelected && !canSegmentSelectedFurniture))
            .accessibilityLabel(L10n.RoomViewer.segmentFurnitureAccessibility)
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

    /// All three room dimensions from saved `.ply.meta` when present and valid.
    private var savedRoomStrictMeters: (width: Float, height: Float, depth: Float)? {
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

    /// ROOM_DIMS_V7 pre-computed metres from SHARP generation — avoids PLY re-measurement.
    private var sharpGenerationRoomMeters: (width: Float, height: Float, depth: Float)? {
        guard let w = sharpRoomMetersW, let h = sharpRoomMetersH, let d = sharpRoomMetersD,
              w.isFinite, h.isFinite, d.isFinite,
              w > 0.05, h > 0.05, d > 0.05 else {
            return nil
        }
        return (w, h, d)
    }

    /// Metric room dimensions recovered from persisted enhanced metadata / room model.
    private var persistedEnhancedRoomMeters: (width: Float, height: Float, depth: Float)? {
        guard let metadata = enhancedRoomMetadata else { return nil }
        let roomModel = metadata.roomModel()
        let width = roomModel.widthMeters
        let height = roomModel.heightMeters
        let depth = roomModel.depthMeters
        guard width.isFinite, height.isFinite, depth.isFinite,
              width > 0.05, height > 0.05, depth > 0.05 else {
            return nil
        }
        return (width, height, depth)
    }

    private var activeRoomMetersDimensionsSource: String {
        if savedRoomStrictMeters != nil { return "SAVED_META_STRICT" }
        if sharpGenerationRoomMeters != nil { return "SHARP_ROOM_DIMS_V7" }
        if persistedEnhancedRoomMeters != nil { return "ENHANCED_METADATA_ROOM_MODEL" }
        if measuredRoomDimensions != nil { return "ASYNC_V7_PLY_MEASURE" }
        if let s = savedRoomModel,
           let w = s.roomWidth, let h = s.roomHeight,
           w > 0.05, h > 0.05, w.isFinite, h.isFinite {
            if let sh = s.roomSceneHeight, sh > 1e-4,
               let sd = s.roomSceneDepth, sd > 1e-4 {
                let depthM = sd * (h / sh)
                if depthM > 0.05, depthM.isFinite {
                    return "SAVED_META_PARTIAL_PLUS_SCENE_DEPTH"
                }
            }
            if let inf = inferredMetersFromPlyScene, inf.depth > 0.05 {
                return "SAVED_META_PARTIAL_PLUS_PLY_INFERRED_DEPTH"
            }
        }
        if inferredMetersFromPlyScene != nil { return "PLY_INFERRED_REFERENCE_HEIGHT" }
        return "UNAVAILABLE"
    }

    private var hasCalculatedRoomMeasurements: Bool {
        savedRoomStrictMeters != nil ||
            sharpGenerationRoomMeters != nil ||
            persistedEnhancedRoomMeters != nil ||
            measuredRoomDimensions != nil
    }

    /// Trimmed / SHARP AABB in **scene units** (Metal bounds preferred, else init-time PLY AABB from generation).
    private var plySceneExtent: (width: Float, height: Float, depth: Float)? {
        if let b = metalBounds, b.width > 0.05, b.height > 0.05, b.depth > 0.05 {
            return (b.width, b.height, b.depth)
        }
        if let w = sharpPlyAabbW, let h = sharpPlyAabbH, let d = sharpPlyAabbD,
           w > 0.05, h > 0.05, d > 0.05 {
            return (w, h, d)
        }
        return nil
    }

    /// Maps PLY vertical span → metres using a fixed reference ceiling (same idea as bounds-based `sceneToMeters` in room geometry).
    private static let plyDisplayReferenceHeightMeters: Float = 2.44

    private var inferredMetersFromPlyScene: (width: Float, height: Float, depth: Float)? {
        guard let p = plySceneExtent, p.height > 1e-4 else { return nil }
        let scale = Self.plyDisplayReferenceHeightMeters / p.height
        let w = p.width * scale
        let h = p.height * scale
        let d = p.depth * scale
        guard w.isFinite, h.isFinite, d.isFinite, w > 0.05, h > 0.05, d > 0.05 else { return nil }
        return (w, h, d)
    }

    /// Nav, Furniture Fit, save, and overlay: saved meta → SHARP V7 → partial meta + scene depth → PLY-inferred metres.
    private var activeRoomMetersDimensions: (width: Float, height: Float, depth: Float)? {
        if let triple = savedRoomStrictMeters { return triple }
        if let triple = sharpGenerationRoomMeters { return triple }
        if let triple = persistedEnhancedRoomMeters { return triple }
        if let measured = measuredRoomDimensions { return (measured.width, measured.height, measured.depth) }
        if let s = savedRoomModel,
           let w = s.roomWidth, let h = s.roomHeight,
           w > 0.05, h > 0.05, w.isFinite, h.isFinite {
            if let sh = s.roomSceneHeight, sh > 1e-4,
               let sd = s.roomSceneDepth, sd > 1e-4 {
                let depthM = sd * (h / sh)
                if depthM > 0.05, depthM.isFinite { return (w, h, depthM) }
            }
            if let inf = inferredMetersFromPlyScene, inf.depth > 0.05 {
                return (w, h, inf.depth)
            }
        }
        return inferredMetersFromPlyScene
    }

    private var initialSharpRoomYaw: Float {
        let stem = viewerPlyURL.deletingPathExtension().lastPathComponent
        let isSavedBasePly = !allowSave &&
            savedRoomModel != nil &&
            !(savedRoomModel?.isClassicPly ?? false) &&
            viewerPlyURL.pathExtension.lowercased() == "ply" &&
            !stem.hasSuffix("_classic") &&
            !stem.hasSuffix("_3dgs")
        return isSavedBasePly ? .pi : 0
    }

    private var canPresentRoomDimensionsAlert: Bool {
        !showRoomNameInput &&
            !isSavingRoom &&
            !isMeasuringRoomDimensions &&
            !showSaveAlert &&
            !showDiscardUnsavedAlert &&
            !showCalibrationRejectAlert &&
            !showWallCalibration &&
            !showFurnitureDimensionsInput &&
            !isCapturingSnapshot
    }

    /// Shown on the save overlay while “Measuring room…” / saving (matches log-style `ROOM_DIMS`).
    private var saveOverlayRoomDimensionsLine: String {
        if let d = activeRoomMetersDimensions,
           d.width > 0.05, d.height > 0.05, d.depth > 0.05 {
            return String(
                format: "ROOM_DIMS W×H×D %.2f × %.2f × %.2f m",
                d.width, d.height, d.depth
            )
        }
        if let cw = calibratedRoomWidth, let ch = calibratedRoomHeight,
           cw > 0.05, ch > 0.05 {
            if let b = metalBounds, b.depth > 0.05 {
                return String(
                    format: "ROOM_DIMS W×H %.2f × %.2f m · D %.2f (scene)",
                    cw, ch, b.depth
                )
            }
            return String(format: "ROOM_DIMS W×H %.2f × %.2f m", cw, ch)
        }
        if let b = metalBounds, b.width > 0.05, b.height > 0.05, b.depth > 0.05 {
            return String(
                format: "ROOM_DIMS W×H×D %.2f × %.2f × %.2f (scene, PLY)",
                b.width, b.height, b.depth
            )
        }
        return "ROOM_DIMS: pending — PLY bounds not ready"
    }

    private var measureRoomProgressOverlay: some View {
        ZStack {
            Color.black.opacity(0.9)
                .ignoresSafeArea()

            VStack(spacing: 24) {
                ZStack {
                    Circle()
                        .fill(Color.green.opacity(0.2))
                        .frame(width: 100, height: 100)

                    Image(systemName: "ruler.fill")
                        .font(.system(size: 40))
                        .foregroundColor(.green)
                }

                Text(L10n.RoomViewer.measuringRoom)
                    .font(.title2)
                    .fontWeight(.semibold)
                    .foregroundColor(.white)

                Text(saveOverlayRoomDimensionsLine)
                    .font(.system(.subheadline, design: .monospaced))
                    .foregroundColor(.white.opacity(0.85))
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, 24)

                ProgressView(value: 0.55)
                    .progressViewStyle(LinearProgressViewStyle(tint: .green))
                    .frame(width: 200)
            }
        }
    }

    /// Width/height/depth for nav title, FurnitureFit, and save.
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
            sharpRoomSplatMeasurementHost: splatMeasurementHost,
            arAssistedSizingEnabled: brainArAssistedSizingEnabled && canOfferBrainArAssist,
            manualFurnitureHeightOverrideMeters: realFurnitureHeight,
            segmentationMode: furnitureFitSegmentationMode,
            onSelectedClassLabelsChanged: { labels in
                selectedFurnitureFitLabels = labels
            },
            showIdentifyLivePreview: furnitureFitShowIdentifyLivePreview,
            showFullVideoWithIdentificationsOverride: showFullVideoWithIdentifications
        )
        // Do not ignore top safe area: full-screen camera would sit under the nav bar and steal ruler taps.
        .ignoresSafeArea(edges: [.bottom, .leading, .trailing])
        .zIndex(100)
    }

    private var calibrationOverlayView: some View {
        ZStack {
            Color.black.opacity(0.6)
                .ignoresSafeArea()
                .onTapGesture { showFurnitureDimensionsInput = false }
            GeometryReader { geometry in
                let isCompactHeight = geometry.size.height < 430
                let popupWidth = min(geometry.size.width - 32, isCompactHeight ? 320 : 360)

                ScrollView(showsIndicators: false) {
                    VStack(spacing: isCompactHeight ? 10 : 16) {
                        Text(L10n.RoomViewer.calibrateRoomTitle)
                            .font(isCompactHeight ? .subheadline.bold() : .headline)
                            .foregroundColor(.white)
                        Text(L10n.RoomViewer.enterFurnitureHeightMeters)
                            .font(.caption)
                            .foregroundColor(.gray)
                            .multilineTextAlignment(.center)
                        Text(L10n.RoomViewer.furnitureFullHeightHint)
                            .font(.caption2)
                            .foregroundColor(.gray.opacity(0.9))
                            .multilineTextAlignment(.center)
                        if let h = calibrationBaselineDetectedHeight ?? detectedFurnitureHeightAR {
                            Text(L10n.RoomViewer.detectedMeters(h))
                                .font(.caption2)
                                .foregroundColor(.orange)
                        }
                        Text(inputFurnitureHeight.isEmpty ? "0.00" : inputFurnitureHeight)
                            .font(.system(size: isCompactHeight ? 28 : 32, weight: .bold, design: .monospaced))
                            .foregroundColor(.white)
                            .frame(width: isCompactHeight ? 110 : 120, height: isCompactHeight ? 40 : 44)
                            .background(Color.white.opacity(0.1))
                            .cornerRadius(8)
                        calibrationNumberPadView(compact: isCompactHeight)
                        HStack(spacing: 16) {
                            Button(L10n.Common.cancel) {
                                inputFurnitureHeight = ""
                                showFurnitureDimensionsInput = false
                            }
                            .font(.body.bold())
                            .foregroundColor(.red)
                            .frame(width: 80, height: 40)
                            .background(Color.red.opacity(0.2))
                            .cornerRadius(8)

                            Button(L10n.Common.apply) { applyCalibration() }
                                .font(.body.bold())
                                .foregroundColor(.green)
                                .frame(width: 80, height: 40)
                                .background(Color.green.opacity(0.2))
                                .cornerRadius(8)
                                .disabled(Float(inputFurnitureHeight) == nil || inputFurnitureHeight.isEmpty)
                        }
                    }
                    .frame(maxWidth: .infinity)
                    .padding(isCompactHeight ? 16 : 20)
                    .background(Color.black.opacity(0.95))
                    .cornerRadius(16)
                    .padding(.horizontal, max(16, (geometry.size.width - popupWidth) * 0.5))
                    .padding(.vertical, 16)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .center)
            }
        }
        .zIndex(99999)
    }

    private func calibrationNumberPadView(compact: Bool) -> some View {
        let buttonWidth: CGFloat = compact ? 46 : 50
        let buttonHeight: CGFloat = compact ? 40 : 44
        let buttonSpacing: CGFloat = compact ? 6 : 8

        return VStack(spacing: buttonSpacing) {
            ForEach(0..<3, id: \.self) { row in
                HStack(spacing: buttonSpacing) {
                    ForEach(1...3, id: \.self) { col in
                        let digit = row * 3 + col
                        Button(action: { appendDigit("\(digit)") }) {
                            Text("\(digit)")
                                .font(.title2.bold()).foregroundColor(.white)
                                .frame(width: buttonWidth, height: buttonHeight)
                                .background(Color.gray.opacity(0.3)).cornerRadius(8)
                        }
                    }
                }
            }
            HStack(spacing: buttonSpacing) {
                Button(action: {
                    if !inputFurnitureHeight.contains(".") {
                        inputFurnitureHeight += inputFurnitureHeight.isEmpty ? "0." : "."
                    }
                }) {
                    Text(".").font(.title2.bold()).foregroundColor(.white)
                        .frame(width: buttonWidth, height: buttonHeight).background(Color.gray.opacity(0.3)).cornerRadius(8)
                }
                Button(action: { appendDigit("0") }) {
                    Text("0").font(.title2.bold()).foregroundColor(.white)
                        .frame(width: buttonWidth, height: buttonHeight).background(Color.gray.opacity(0.3)).cornerRadius(8)
                }
                Button(action: { if !inputFurnitureHeight.isEmpty { inputFurnitureHeight.removeLast() } }) {
                    Image(systemName: "delete.left").font(.title3).foregroundColor(.white)
                        .frame(width: buttonWidth, height: buttonHeight).background(Color.gray.opacity(0.3)).cornerRadius(8)
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

        if realHeight >= max(displayRoomHeight, 0.01) {
            calibrationRejectMessage = L10n.RoomViewer.furnitureHeightMustBeLessThanRoomHeight(
                displayRoomHeight
            )
            showCalibrationRejectAlert = true
            return
        }

        let scaleFactor = realHeight / detectedHeight
        realFurnitureHeight = realHeight
        logDebug("📐 [Calibration] Real height: \(realHeight)m, overlay scale factor: \(scaleFactor)")
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
        let displayH = detectedFurnitureHeightAR ?? 0
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

    private var shouldShowArFurnitureMeasurementPill: Bool {
        showingFurnitureFit &&
            brainArAssistedSizingEnabled &&
            (detectedFurnitureHeightAR?.isFinite == true) &&
            ((detectedFurnitureHeightAR ?? 0) > 0.05)
    }

    private func placementIntelligenceRingColor(fit: FitCheckResult?) -> Color {
        guard let fit else { return .cyan }
        return fit.fitsInRoom ? .green : .red
    }

    @ViewBuilder
    private func placementIntelligenceExpandedContent(
        dimensions: RoomFurnitureDimensions?,
        fit: FitCheckResult?,
        aesthetic: AestheticScore
    ) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                Text(L10n.RoomViewer.placementIntelligenceTitle)
                    .font(.caption.bold())
                    .foregroundColor(.white)
                Spacer(minLength: 4)
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
            }
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
                    Text(L10n.RoomViewer.placementFitsRoom)
                        .font(.caption2)
                        .foregroundColor(.green)
                } else {
                    Text(L10n.RoomViewer.placementExceedsRoom)
                        .font(.caption2)
                        .foregroundColor(.red)
                }
            }
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
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
        .frame(maxWidth: 300, alignment: .leading)
        .background(Color.black.opacity(0.88))
        .cornerRadius(14)
        .shadow(color: .black.opacity(0.45), radius: 8, x: 0, y: 4)
    }

    /// Small round pill between brain and snapshot; grid icon = spatial placement; ring = fit (green/red) or style-only (cyan).
    @ViewBuilder
    private var roomIntelligencePlacementCard: some View {
        if showingFurnitureFit,
           authoritativeRoomModelForMetrics != nil,
           placementIntelligenceHasFurnitureSignal,
           let aesthetic = latestAestheticScore {
            let dimensions = derivedDetectedFurnitureDimensionsForRoomIntelligence()
            let fit = latestFitCheckResult
            VStack(spacing: 10) {
                if isPlacementIntelligenceExpanded {
                    placementIntelligenceExpandedContent(dimensions: dimensions, fit: fit, aesthetic: aesthetic)
                        .transition(.opacity.combined(with: .move(edge: .bottom)))
                }
                Button {
                    withAnimation(.easeInOut(duration: 0.18)) {
                        isPlacementIntelligenceExpanded.toggle()
                    }
                } label: {
                    ZStack {
                        Circle()
                            .fill(
                                LinearGradient(
                                    colors: [Color(white: 0.22), Color(white: 0.12)],
                                    startPoint: .topLeading,
                                    endPoint: .bottomTrailing
                                )
                            )
                            .frame(width: 46, height: 46)
                            .overlay(
                                Circle()
                                    .stroke(placementIntelligenceRingColor(fit: fit), lineWidth: 2.5)
                            )
                            .shadow(color: .black.opacity(0.35), radius: 4, x: 0, y: 2)
                        Image(systemName: "square.split.2x2.fill")
                            .font(.system(size: 18, weight: .semibold))
                            .foregroundStyle(.white)
                            .symbolRenderingMode(.hierarchical)
                            .accessibilityHidden(true)
                    }
                }
                .buttonStyle(.plain)
                .accessibilityLabel(L10n.RoomViewer.placementIntelligenceTitle)
                .accessibilityAddTraits(.isButton)
            }
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
                HStack(alignment: .bottom, spacing: 16) {
                    HStack(alignment: .bottom, spacing: 8) {
                        brainButtonWithHintAbove
                    }
                    segmentButton
                    if showingFurnitureFit {
                        roomIntelligencePlacementCardResetOnExit
                    }
                    Spacer().allowsHitTesting(false)
                    VStack(alignment: .trailing, spacing: 10) {
                        if showingFurnitureFit, shouldShowArFurnitureMeasurementPill {
                            if showRoomFurnitureCalibrate {
                                Button(action: { showFurnitureDimensionsInput = true }) {
                                    furnitureMeasurementPillContent(showTapHint: true)
                                }
                            } else {
                                furnitureMeasurementPillContent(showTapHint: false)
                            }
                        }
                        snapshotButtonWithHintAbove
                    }
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
                HStack(alignment: .bottom, spacing: 0) {
                    HStack(alignment: .bottom, spacing: 8) {
                        brainButtonWithHintAbove
                    }
                        .padding(.leading, 16)
                    segmentButton
                        .padding(.leading, 10)
                    Spacer(minLength: 4)
                    if showingFurnitureFit {
                        roomIntelligencePlacementCardResetOnExit
                    }
                    Spacer(minLength: 4)
                    VStack(spacing: 8) {
                        if showingFurnitureFit, shouldShowArFurnitureMeasurementPill {
                            if showRoomFurnitureCalibrate {
                                Button(action: { showFurnitureDimensionsInput = true }) {
                                    furnitureMeasurementPillContent(showTapHint: true)
                                }
                            } else {
                                furnitureMeasurementPillContent(showTapHint: false)
                            }
                        }
                        snapshotButtonWithHintAbove
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
                topTrailingPinchTapAndSizingHintsOverlay
                roomDimensionsHintOverlay
                fullVideoFurnitureTapHintOverlay
                fullVideoToolbarHelperOverlay
            }
            if isLoading { loadingOverlayView }
            errorOverlayView
            if showingFurnitureFit { furnitureFitOverlayView }
            if isMeasuringRoomDimensions { measureRoomProgressOverlay }
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

    private var sharpRoomModalPauseToken: SharpRoomModalPauseToken {
        SharpRoomModalPauseToken(
            showRoomNameInput: showRoomNameInput,
            isSavingRoom: isSavingRoom,
            showSaveAlert: showSaveAlert,
            showDiscardUnsavedAlert: showDiscardUnsavedAlert,
            showCalibrationRejectAlert: showCalibrationRejectAlert,
            showWallCalibration: showWallCalibration,
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
            showSaveAlert ||
            showDiscardUnsavedAlert ||
            showCalibrationRejectAlert ||
            showWallCalibration ||
            furnitureCalibSheet ||
            isCapturingSnapshot
        guard pause != sharpRoomUIPauseApplied else { return }
        sharpRoomUIPauseApplied = pause
        splatMeasurementHost.setModalHeavyWorkPaused(pause)
    }

    private func startSavingRoom() {
        let trimmedRoomName = roomName.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedRoomName.isEmpty else { return }
        guard !modelManager.hasSavedRoomNameConflict(trimmedRoomName) else {
            saveAlertMessage = L10n.RoomViewer.duplicateRoomName
            saveWasSuccessful = false
            showSaveAlert = true
            return
        }

        let savedName = trimmedRoomName
        roomName = trimmedRoomName
        logDebug("💾 [SharpRoomView] Starting room save: \(savedName)")

        // Dismiss the name-entry alert before presenting the full-screen save progress overlay.
        showRoomNameInput = false

        Task {
            await MainActor.run {
                if showingFurnitureFit {
                    logDebug("💾 [SharpRoomView] Save: stopping Furniture Fit before persist")
                    showingFurnitureFit = false
                }
                withAnimation(.easeIn(duration: 0.2)) {
                    isSavingRoom = true
                    saveProgress = 0.0
                    saveProgressStatusText = L10n.RoomViewer.savingRoomEllipsis
                }
            }
            try? await Task.sleep(nanoseconds: 220_000_000)

            await MainActor.run {
                saveProgressStatusText = L10n.RoomViewer.savingRoomEllipsis
            }

            await MainActor.run { saveProgress = 0.35; saveProgressStatusText = L10n.RoomViewer.savingRoomEllipsis }

            if savedRoomStrictMeters == nil && sharpGenerationRoomMeters == nil && measuredRoomDimensions == nil {
                backgroundRoomMeasurementTask?.cancel()
                backgroundRoomMeasurementTask = nil
                let saveMeasured = await modelManager.measureRoomDimensionsAsync(
                    forPly: viewerPlyURL,
                    treatAsClassicPly: viewerUsesClassicPlyBehavior
                )
                await MainActor.run {
                    if let saveMeasured {
                        measuredRoomDimensions = saveMeasured
                        updateRoomPlacementIntelligence()
                        logDebug(
                            "[ROOM_DIMS][SAVE] FILE=\(viewerPlyURL.lastPathComponent) " +
                            "APPROACH=\(saveMeasured.approach.uppercased()) SHOT=\(saveMeasured.shotType) " +
                            "HAS_FOCAL=\(saveMeasured.usedFocal) TILT_DEG=\(String(format: "%.2f", saveMeasured.tiltDegrees)) " +
                            "TILT_RELIABLE=\(saveMeasured.tiltReliable) CUBOID_RATIO=\(String(format: "%.4f", saveMeasured.cuboidRatio)) " +
                            "THRESHOLD=\(String(format: "%.4f", saveMeasured.cuboidThreshold)) " +
                            "FILL_W=\(String(format: "%.4f", saveMeasured.fillWidth)) BLEND=\(String(format: "%.4f", saveMeasured.blend)) " +
                            "W=\(String(format: "%.4f", saveMeasured.width)) " +
                            "H=\(String(format: "%.4f", saveMeasured.height)) " +
                            "D=\(String(format: "%.4f", saveMeasured.depth))"
                        )
                    } else {
                        logDebug("[ROOM_DIMS][SAVE] FILE=\(viewerPlyURL.lastPathComponent) FALLBACK=ASYNC_V7_UNAVAILABLE")
                    }
                }
            }

            let (fallbackDimensions, sceneExtentForMeta) = await MainActor.run {
                (activeRoomMetersDimensions, plySceneExtent)
            }
            let roomW = fallbackDimensions?.width
            let roomH = fallbackDimensions?.height
            let roomD = fallbackDimensions?.depth
            let roomDimsApproachForSave: String? = await MainActor.run {
                if sharpGenerationRoomMeters != nil { return "room_dims_v7_sharp" }
                if measuredRoomDimensions != nil { return "room_dims_v7_async" }
                return nil
            }
            if let roomW, let roomH, let roomD {
                logDebug(
                    "🟢 [SharpRoomView] Save: ROOM_DIMS W×H×D=" +
                        "\(String(format: "%.3f", roomW))×\(String(format: "%.3f", roomH))×\(String(format: "%.3f", roomD))m"
                )
            } else {
                logDebug("🔴 [SharpRoomView] Save: room dimensions unavailable")
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
                    roomDimsApproach: roomDimsApproachForSave,
                    roomSceneWidth: sceneExtentForMeta?.width,
                    roomSceneHeight: sceneExtentForMeta?.height,
                    roomSceneDepth: sceneExtentForMeta?.depth
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
                            let roomWString = roomW.map { String(format: "%.3f", $0) } ?? "nil"
                            let roomHString = roomH.map { String(format: "%.3f", $0) } ?? "nil"
                            let roomDString = roomD.map { String(format: "%.3f", $0) } ?? "nil"
                            logDebug(
                                "[SAVE_ENHANCED_METADATA] room=\"\(savedName)\" — no EnhancedRoomMetadata (nil). " +
                                "PLY save used display meters W×H×D=\(roomWString)×\(roomHString)×\(roomDString)"
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

        if let hint = modelManager.loadSplatLoadHint(forRoomURL: viewerPlyURL) {
            if hint.matches(fileURL: viewerPlyURL) {
                persistedSplatLoadHint = hint
                if metalBounds == nil {
                    metalBounds = hint.fullRoomBounds
                }
                logDebug(
                    "⏱️ [SplatLoad] metadata_hit file=\(viewerPlyURL.lastPathComponent) type=hint " +
                    "splats=\(hint.splatCount) source=sidecar"
                )
            } else {
                logDebug("⏱️ [SplatLoad] metadata_stale file=\(viewerPlyURL.lastPathComponent) type=hint reason=file_identity_mismatch")
            }
        } else {
            logDebug("⏱️ [SplatLoad] metadata_miss file=\(viewerPlyURL.lastPathComponent) type=hint")
        }

        if let savedRoomModel {
            if let metadata = modelManager.loadEnhancedMetadata(
                forSavedRoomNamed: savedRoomModel.fileName,
                fileType: savedRoomModel.fileType
            ) {
                enhancedRoomMetadata = metadata
                roomModel = nil
                if metalBounds == nil {
                    metalBounds = RoomBounds(
                        minX: metadata.aabbMin.x,
                        maxX: metadata.aabbMax.x,
                        minY: metadata.aabbMin.y,
                        maxY: metadata.aabbMax.y,
                        minZ: metadata.aabbMin.z,
                        maxZ: metadata.aabbMax.z
                    )
                }
                logDebug("📐 [SharpRoomView] Loaded saved room geometry metadata")
                return
            }
        }

        if let metadata = modelManager.loadEnhancedMetadata(forRoomURL: viewerPlyURL) {
            enhancedRoomMetadata = metadata
            roomModel = nil
            if metalBounds == nil {
                metalBounds = RoomBounds(
                    minX: metadata.aabbMin.x,
                    maxX: metadata.aabbMax.x,
                    minY: metadata.aabbMin.y,
                    maxY: metadata.aabbMax.y,
                    minZ: metadata.aabbMin.z,
                    maxZ: metadata.aabbMax.z
                )
            }
            logDebug("📐 [SharpRoomView] Loaded fresh room geometry metadata")
        }
    }

    private func scheduleRoomGeometryExtractionIfNeeded() {
        logDebug("📐 [SharpRoomView] Room geometry extraction disabled")
    }

    private func triggerRoomGeometryExtractionIfNeeded(force: Bool = false) {
        let _ = force
        logDebug("📐 [SharpRoomView] RoomGeometryEngine / RANSAC path disabled")
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

        let height = realFurnitureHeight ?? detectedFurnitureHeightAR ?? furnitureProportionalHeightMeters
        guard let height,
              height.isFinite,
              height > 0.05 else { return nil }

        let estimatedDepth = max(0.25, min(width * 0.72, 1.4))
        return RoomFurnitureDimensions(widthM: width, heightM: height, depthM: estimatedDepth)
    }

    /// Room geometry for placement intelligence: live RANSAC model, persisted enhanced metadata, or a stub box from
    /// ``activeRoomMetersDimensions`` when full extraction is disabled (matches nav / Furniture Fit room dims).
    private var authoritativeRoomModelForMetrics: RoomModel? {
        if let roomModel { return roomModel }
        if let metadata = enhancedRoomMetadata {
            return metadata.roomModel()
        }
        guard let dims = activeRoomMetersDimensions,
              dims.width > 0.05, dims.height > 0.05, dims.depth > 0.05 else {
            return nil
        }
        return PlacementIntelligenceRoomStub.axisAlignedBoxMeters(
            width: dims.width,
            height: dims.height,
            depth: dims.depth
        )
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

            // High-frequency placement logging disabled for performance.
        } else {
            latestEstimatedFurnitureDepthMeters = nil
            latestFitCheckResult = nil
            latestCornerPlacementSuggestions = []
            // Silence metric-fit skipped logs to avoid console spam.
        }

        let palette = roomModel.surfacePalette
        let roomStyleTags = inferredRoomStyleTags(from: palette)
        let furnitureProfile = heuristicFurnitureProfileForAesthetic(
            roomModel: roomModel,
            segmentedMeanSRGB: segmentedFurnitureMeanSRGB
        )
        let aestheticAdvisor = AestheticAdvisor(palette: palette, roomStyleTags: roomStyleTags)
        latestAestheticScore = aestheticAdvisor.evaluate(furniture: furnitureProfile)

        // Silence aesthetic harmony logs (can be re-enabled behind a verbose flag).
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

// MARK: - Placement intelligence fallback room

/// Minimal axis-aligned room used when ``RoomGeometryEngine`` output is unavailable but nav / Furniture Fit
/// already have W×H×D metres (same source as ``activeRoomMetersDimensions``).
private enum PlacementIntelligenceRoomStub {
    static func axisAlignedBoxMeters(width: Float, height: Float, depth: Float) -> RoomModel {
        let w = max(width, 0.2)
        let h = max(height, 0.2)
        let d = max(depth, 0.2)
        let wHalf = w * 0.5
        let dHalf = d * 0.5
        let aabb = AABB3(
            min: SIMD3<Float>(-wHalf, 0, -dHalf),
            max: SIMD3<Float>(wHalf, h, dHalf)
        )
        let floor = DetectedPlane(type: .floor, normal: SIMD3<Float>(0, 1, 0), pointOnPlane: .zero)
        let ceiling = DetectedPlane(type: .ceiling, normal: SIMD3<Float>(0, -1, 0), pointOnPlane: SIMD3<Float>(0, h, 0))
        let walls: [DetectedPlane] = [
            DetectedPlane(type: .wall, normal: SIMD3<Float>(1, 0, 0), pointOnPlane: SIMD3<Float>(-wHalf, 0, 0)),
            DetectedPlane(type: .wall, normal: SIMD3<Float>(-1, 0, 0), pointOnPlane: SIMD3<Float>(wHalf, 0, 0)),
            DetectedPlane(type: .wall, normal: SIMD3<Float>(0, 0, 1), pointOnPlane: SIMD3<Float>(0, 0, -dHalf)),
            DetectedPlane(type: .wall, normal: SIMD3<Float>(0, 0, -1), pointOnPlane: SIMD3<Float>(0, 0, dHalf))
        ]
        let uvMin = SIMD2<Float>(-wHalf, -dHalf)
        let uvMax = SIMD2<Float>(wHalf, dHalf)
        let freeFloor = FreeFloorRegion(
            polygon: [
                uvMin,
                SIMD2<Float>(wHalf, -dHalf),
                uvMax,
                SIMD2<Float>(-wHalf, dHalf)
            ],
            areaSqM: w * d,
            uvBounds: FloorUVBounds(min: uvMin, max: uvMax)
        )
        return RoomModel(
            aabb: aabb,
            floor: floor,
            ceiling: ceiling,
            walls: walls,
            corners: [],
            freeFloorRegions: [freeFloor],
            surfacePalette: .empty,
            cameraInfo: nil,
            sceneToMeters: 1.0
        )
    }
}

// MARK: - Preview

#Preview {
    NavigationStack {
        // Preview with a sample URL (won't actually load)
        SharpRoomView(plyURL: URL(fileURLWithPath: "/sample.ply"))
    }
}
