import SwiftUI
import UIKit
import CoreML
import Photos
import MetalKit
import simd

// MARK: - Modal pause sync (AR + TextField responsiveness)

/// Bundles alert/sheet flags so one `onChange` can pause ARKit without bloating the SwiftUI type checker.
private struct SplatRoomModalPauseToken: Equatable {
    var showRoomNameInput: Bool
    var isSavingRoom: Bool
    var showSaveErrorNotice: Bool
    var showDiscardUnsavedAlert: Bool
    var showCalibrationRejectAlert: Bool
    var showWallCalibration: Bool
    var showFurnitureDimensionsInput: Bool
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
    /// In classic PLY from Splat, Z is negative, and the wall closest to camera
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

/// Splat room viewer: **MetalSplatter** (Metal) for the splat layer.
struct SplatRoomView: View {
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
    /// Whether this room should use Splat classic orientation/rendering even without `_classic` in the file name.
    private let viewerUsesClassicPlyBehavior: Bool
    /// Coordinate-frame contract for save/load/render behavior.
    private let viewerRoomCoordinateFrame: RoomCoordinateFrame

    /// Splat **`_classic.ply`** write-time AABB (scene units); set when pushing from ``SinglePhotoRoomViewer`` after generation.
    private let splatPlyAabbW: Float?
    private let splatPlyAabbH: Float?
    private let splatPlyAabbD: Float?
    /// ROOM_DIMS_V7 pre-computed room dimensions in **metres** from Splat generation (no PLY re-measurement needed).
    private let splatRoomMetersW: Float?
    private let splatRoomMetersH: Float?
    private let splatRoomMetersD: Float?
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
        splatPlyAabbWidth: Float? = nil,
        splatPlyAabbHeight: Float? = nil,
        splatPlyAabbDepth: Float? = nil,
        splatRoomWidth: Float? = nil,
        splatRoomHeight: Float? = nil,
        splatRoomDepth: Float? = nil,
        roomCoordinateFrame: RoomCoordinateFrame = .classicSplatPly,
        sourcePhotoPixelWidth: Int? = nil,
        sourcePhotoPixelHeight: Int? = nil
    ) {
        self.plyURL = plyURL
        self.allowSave = allowSave
        self.photoOrientation = photoOrientation
        self.savedRoomWidth = savedRoomWidth
        self.savedRoomHeight = savedRoomHeight
        self.savedRoomModel = savedRoomModel
        self.splatPlyAabbW = splatPlyAabbWidth
        self.splatPlyAabbH = splatPlyAabbHeight
        self.splatPlyAabbD = splatPlyAabbDepth
        self.splatRoomMetersW = splatRoomWidth
        self.splatRoomMetersH = splatRoomHeight
        self.splatRoomMetersD = splatRoomDepth
        self.sourcePhotoPixelWidth = sourcePhotoPixelWidth
        self.sourcePhotoPixelHeight = sourcePhotoPixelHeight

        self.viewerPlyURL = plyURL
        self.viewerUsesClassicPlyBehavior = self.viewerPlyURL.path.hasSuffix("_classic.ply") || (savedRoomModel?.isClassicPly ?? false)
        self.viewerRoomCoordinateFrame = savedRoomModel?.roomCoordinateFrame ?? roomCoordinateFrame
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
    @ObservedObject private var rtmdetService = RTMDetModelService.shared

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
    @State private var saveAlertMessage = ""
    @State private var saveWasSuccessful = false
    @State private var showSaveSuccessSnackbar = false
    @State private var saveSuccessSnackbarMessage = ""
    @State private var showSaveErrorNotice = false
    @State private var showDiscardUnsavedAlert = false
    @State private var isDismissing = false
    @State private var showRoomNameInput = false
    @State private var roomName = ""
    @State private var isCapturingSnapshot = false
    @State private var splatRoomUIPauseApplied = false
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
    /// Mean sRGB (0…1) from composited RTMDet cutout pixels; drives ``FurnitureProfile.primaryColor`` when set.
    @State private var segmentedFurnitureMeanSRGB: SIMD3<Float>?
    /// Collapsed: round pill icon; expanded: detail card above the pill.
    @State private var isPlacementIntelligenceExpanded = false
    // MARK: - Onboarding hint coordinator (one hint at a time, priority-ordered)
    enum OnboardingHint: Equatable {
        case pickAnother   // G — "Not this one? Tap to pick another."
        case arSizing      // E — "Tap the brain icon, then size-match…"
        case pinchResize   // A′ — "Pinch resizes the overlay."
    }
    @AppStorage("hint_seenPinchResize") private var seenPinchResize = false
    @AppStorage("hint_seenArSizing") private var seenArSizing = false
    @AppStorage("hint_seenPickAnother") private var seenPickAnother = false
    @State private var replayTeachingHints = false
    @State private var selectionJustBecamePrimary = false
    @State private var onboardingHintDismissTask: Task<Void, Never>?
    @State private var forceShowHints = false

    private var activeOnboardingHint: OnboardingHint? {
        let mode = currentRoomViewerMode
        let ignoreSeenFlags = forceShowHints
        // Priority 1 – G
        if (selectionJustBecamePrimary || ignoreSeenFlags) && (ignoreSeenFlags || !seenPickAnother) && (mode == .furnitureFit || mode == .fullVideo) {
            return .pickAnother
        }
        // Priority 2 – E (merged D)
        if mode == .furnitureFit && canOfferBrainArAssist && (ignoreSeenFlags || !seenArSizing) {
            return .arSizing
        }
        // Priority 3 – A′
        if mode == .furnitureFit && (ignoreSeenFlags || !seenPinchResize) {
            return .pinchResize
        }
        return nil
    }

    private func isHintEligible(_ hint: OnboardingHint) -> Bool {
        let mode = currentRoomViewerMode
        switch hint {
        case .pickAnother:
            return (mode == .furnitureFit || mode == .fullVideo)
        case .arSizing:
            return mode == .furnitureFit && canOfferBrainArAssist
        case .pinchResize:
            return mode == .furnitureFit
        }
    }

    private func isHintVisible(_ hint: OnboardingHint) -> Bool {
        if forceShowHints {
            return isHintEligible(hint)
        }
        return activeOnboardingHint == hint
    }

    private enum RoomViewerMode { case browsing, furnitureFit, fullVideo }
    private var currentRoomViewerMode: RoomViewerMode {
        if showingFurnitureFit && showFullVideoWithIdentifications { return .fullVideo }
        if showingFurnitureFit { return .furnitureFit }
        return .browsing
    }

    private func scheduleOnboardingHintAutoDismiss() {
        onboardingHintDismissTask?.cancel()
        guard activeOnboardingHint != nil else {
            onboardingHintDismissTask = nil
            return
        }
        let dismissDelay: Double = forceShowHints ? 5 : 3
        onboardingHintDismissTask = Task { @MainActor in
            try? await Task.sleep(for: .seconds(dismissDelay))
            guard !Task.isCancelled else { return }
            if forceShowHints {
                forceShowHints = false
            } else {
                if let hint = activeOnboardingHint {
                    markOnboardingHintSeen(hint)
                }
            }
        }
    }

    private func markOnboardingHintSeen(_ hint: OnboardingHint) {
        switch hint {
        case .pickAnother:
            seenPickAnother = true
            selectionJustBecamePrimary = false
        case .arSizing:    seenArSizing = true
        case .pinchResize: seenPinchResize = true
        }
        forceShowHints = false
        onboardingHintDismissTask?.cancel()
        onboardingHintDismissTask = nil
        scheduleOnboardingHintAutoDismiss()
    }

    private func showHintsOnDemand() {
        forceShowHints = true
        replayTeachingHints = true
        onboardingHintDismissTask?.cancel()
        scheduleOnboardingHintAutoDismiss()
    }

    @State private var showFullVideoWithIdentifications = false
    @State private var measuredRoomDimensions: MeasuredPlyRoomDimensions?
    @StateObject private var immersiveChrome = PaafektViewerChromeController()
    var body: some View {
        splatRoomBody
    }

    @ViewBuilder
    private var splatRoomBaseLayer: some View {
        ZStack {
            metalSplatAndGestureLayer
            allOverlaysLayer
        }
        .background(Color.gray)
    }

    private func onSplatRoomDimensionsRulerTapped() {
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
        } else {
            logDebug("[ROOM_DIMS][RULER] FILE=\(viewerPlyURL.lastPathComponent) FALLBACK=START_ASYNC_MEASURE source=\(activeRoomMetersDimensionsSource)")
            startAsyncRoomMeasurementForRuler()
        }
    }

    @ViewBuilder
    private var navigationBarRoomMeasurementPrincipal: some View {
        HStack(spacing: 12) {
            Button(action: onSplatRoomDimensionsRulerTapped) {
                if let d = activeRoomMetersDimensions {
                    Text(L10n.RoomViewer.approximateRoomHeight(d.height))
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundColor(.white)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(Capsule().fill(Color.black.opacity(0.55)))
                } else {
                    Image(systemName: "ruler.fill")
                        .symbolRenderingMode(.hierarchical)
                        .font(.system(size: 18, weight: .medium))
                        .foregroundStyle(.primary)
                }
            }
            .buttonStyle(.plain)
            .disabled(!canPresentRoomDimensionsAlert || isMeasuringRoomDimensions)
            .accessibilityLabel(L10n.RoomViewer.checkMeasurement)

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
                        .background(Capsule().fill(Theme.Palette.viewerCapsuleFill))
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

    private func handleSplatRoomBackTap() {
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
        Button(action: handleSplatRoomBackTap) {
            Image(systemName: "chevron.left")
        }
        .accessibilityLabel(L10n.Common.back)
    }

    private var navigationBarRecenterButton: some View {
        Button {
            splatMeasurementHost.recenterSplatRoomCamera()
            logDebug("🎯 [SplatRoomView] Recenter (toolbar)")
        } label: {
            Image(systemName: "viewfinder")
                .font(.title3)
        }
        .disabled(isLoading)
        .accessibilityLabel(L10n.RoomViewer.recenterView)
    }

    private func toggleFullVideoIdentifications() {
        showFullVideoWithIdentifications.toggle()
        if showFullVideoWithIdentifications {
            furnitureFitSegmentationMode = .identifyOnly
            furnitureFitShowIdentifyLivePreview = true
        } else {
            // Back to the brain default: auto-segment the highest-confidence primary.
            furnitureFitSegmentationMode = .segmentPrimary
            furnitureFitShowIdentifyLivePreview = true
            selectionJustBecamePrimary = true
            scheduleOnboardingHintAutoDismiss()
        }
    }

    private var fullVideoIdentificationsFloatingButton: some View {
        Button(action: toggleFullVideoIdentifications) {
            Image(systemName: "text.viewfinder")
                .symbolVariant(showFullVideoWithIdentifications ? .fill : .none)
                .font(.system(size: 16, weight: .semibold))
                .foregroundStyle(showingFurnitureFit ? Color.cyan : .white)
                .frame(width: 36, height: 36)
                .background(Circle().fill(Color.black.opacity(0.62)))
                .overlay(
                    Circle().stroke(
                        showFullVideoWithIdentifications ? Color.cyan.opacity(0.9) : Color.white.opacity(0.18),
                        lineWidth: 1
                    )
                )
        }
        .buttonStyle(.plain)
        .disabled(isLoading)
        .accessibilityLabel(L10n.Settings.fullVideoWithIdentifications)
        .accessibilityHint(L10n.Settings.fullVideoWithIdentificationsDescription)
        .accessibilityAddTraits(showFullVideoWithIdentifications ? .isSelected : [])
    }

    /// Kept for API compatibility; the button is now rendered inside
    /// ``topTrailingActionButtonsOverlay`` so it stacks cleanly with the AR
    /// sizing button and hint.
    private var fullVideoModeFloatingButtonOverlay: some View {
        EmptyView()
    }

    private var navigationBarSaveButton: some View {
        Button(action: {
            roomName = ""
            showRoomNameInput = true
        }) {
            Image(systemName: "square.and.arrow.down")
                .font(.title3)
        }
        .disabled(isLoading)
        .accessibilityLabel(L10n.RoomViewer.saveRoom)
    }

    private var navigationBarTrailingControls: some View {
        HStack(spacing: 14) {
            Button {
                showHintsOnDemand()
            } label: {
                Image(systemName: "questionmark.circle")
                    .font(.system(size: 18, weight: .medium))
                    .foregroundStyle(.white.opacity(0.85))
            }
            .buttonStyle(.plain)
            .disabled(isLoading)
            .accessibilityLabel("Show hints")

            navigationBarRecenterButton
            if allowSave {
                navigationBarSaveButton
            }
        }
    }

    private var fullVideoToolbarHelperOverlay: some View {
        ZStack(alignment: .topTrailing) {
            Color.clear
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .allowsHitTesting(false)
            if isHintVisible(.pickAnother) {
                PaafektHintChip(
                    systemImage: "text.viewfinder",
                    text: L10n.RoomViewer.fullVideoSelectionHelper
                )
                .padding(.top, 6)
                .padding(.trailing, 20)
                .offset(y: 50)
                .transition(.opacity)
            }
        }
        .opacity(isCapturingSnapshot ? 0 : 1)
        .allowsHitTesting(false)
        .zIndex(106)
    }

    private func toggleBrainArAssistedSizingOrShowHint() {
        guard showingFurnitureFit else { return }
        if isHintVisible(.arSizing) {
            markOnboardingHintSeen(.arSizing)
        }
        brainArAssistedSizingEnabled.toggle()
    }

    private var arSizingButton: some View {
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

    /// AR sizing button + hint + full-video button stacked vertically in the
    /// top-trailing corner. Lives in the SwiftUI content layer (not the system
    /// toolbar) so the hint can anchor inline — same pattern as
    /// ``brainButtonWithHintAbove``.
    private var topTrailingActionButtonsOverlay: some View {
        ZStack(alignment: .topTrailing) {
            Color.clear
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .allowsHitTesting(false)
            VStack(alignment: .trailing, spacing: 8) {
                if canOfferBrainArAssist, showingFurnitureFit {
                    arSizingButton
                    if isHintVisible(.arSizing) {
                        PaafektHintChip(
                            systemImage: "arrow.up.left.and.arrow.down.right",
                            text: L10n.RoomViewer.arFurnitureSizingRequiresBrainHint                        )
                        .transition(.opacity)
                    }
                }
                if showingFurnitureFit {
                    fullVideoIdentificationsFloatingButton
                        .transition(.opacity)
                }
            }
            .padding(.top, 4)
            .padding(.trailing, 16)
        }
        .opacity(isCapturingSnapshot ? 0 : 1)
        .zIndex(108)
    }

    /// Pinch-resize hint (A′) — shown only when coordinator says so.
    private var topTrailingPinchHintOverlay: some View {
        paafektTopToolbarHintOverlay {
            VStack(spacing: 6) {
                if isHintVisible(.pinchResize) {
                    PaafektHintChip(
                        systemImage: "hand.pinch.fill",
                        text: L10n.RoomViewer.pinchGestureHintExplanation
                    )
                    .transition(.opacity)
                }
            }
        }
        .opacity(isCapturingSnapshot ? 0 : 1)
        .zIndex(101)
    }

    @ToolbarContentBuilder
    private var splatRoomToolbarContent: some ToolbarContent {
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

    private var splatRoomNavigationView: some View {
        splatRoomBaseLayer
            .navigationBarHidden(true)
            .navigationTitle("")
            .navigationBarTitleDisplayMode(.inline)
            .navigationBarBackButtonHidden(true)
            .toolbar(.hidden, for: .navigationBar)
    }

    private func splatRoomPerformOnAppear() {
        // Preload RTMDet when the room opens (async; not at app startup). First brain tap stays snappy.
        rtmdetService.ensureModelLoaded()
        if photoOrientation == .landscape { OrientationLockManager.shared.lockToLandscape() } else { OrientationLockManager.shared.lockToPortrait() }
        logDebug("📐 [SplatRoomView] photoOrientation = \(photoOrientation)")
        loadPersistedRoomMetadataIfNeeded()
        syncModalHeavyWorkPauseForSplatRoomUI()
        warmRoomMeasurementInBackgroundIfNeeded()
    }

    private func splatRoomPerformOnDisappear() {
        autoEnableARTask?.cancel()
        autoEnableARTask = nil
        roomMeasurementTask?.cancel()
        roomMeasurementTask = nil
        backgroundRoomMeasurementTask?.cancel()
        backgroundRoomMeasurementTask = nil
        isMeasuringRoomDimensions = false
        saveProgressStatusText = L10n.RoomViewer.savingRoomEllipsis
        onboardingHintDismissTask?.cancel()
        OrientationLockManager.shared.unlock()
        splatMeasurementHost.setModalHeavyWorkPaused(false)
        rtmdetService.releaseResources()
    }

    private func splatRoomHandleShowingFurnitureFitChange(isOn: Bool) {
        logFurnitureFitSize(
            "phase=toggle active=\(isOn) suppressStartupProgress=\(furnitureFitInitialSegmentationDone) " +
            "room_m=\(String(format: "%.2f", furnitureFitRoomWidth))×\(String(format: "%.2f", furnitureFitRoomHeight))×\(String(format: "%.2f", displayRoomDepth))"
        )
        if isOn {
            rtmdetService.ensureModelLoaded()
            updateRoomPlacementIntelligence()
            scheduleOnboardingHintAutoDismiss()
        } else {
            onboardingHintDismissTask?.cancel()
            brainArAssistedSizingEnabled = false
            showFullVideoWithIdentifications = false
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
        }
    }

    private func splatRoomHandleBrainArAssistedSizingChange(enabled: Bool) {
        if !enabled {
            detectedFurnitureHeightAR = nil
        }
        logFurnitureFitSize("phase=splat_room_ar_opt_in enabled=\(enabled) active=\(showingFurnitureFit)")
    }

    private func splatRoomHandleIsLoadingForDefaultARCamera(loading: Bool) {
        guard !loading, !didEnableDefaultARCamera else { return }
        didEnableDefaultARCamera = true
        if allowSave {
            logDebug("📱 [SplatRoomView] Fresh Splat room loaded")
        } else {
            logDebug("📱 [SplatRoomView] Saved room loaded")
        }
    }

    private func splatRoomHandleIsLoadingForHints(loading: Bool) {
        if loading {
            onboardingHintDismissTask?.cancel()
        } else {
            scheduleOnboardingHintAutoDismiss()
        }
    }

    // MARK: - Navigation chrome + lifecycle (split for Swift compiler type-check)

    private var splatRoomLifecycleStageAppear: some View {
        splatRoomNavigationView
            .onAppear {
                splatRoomPerformOnAppear()
            }
            .onChange(of: splatRoomModalPauseToken) { _, _ in syncModalHeavyWorkPauseForSplatRoomUI() }
    }

    private var splatRoomLifecycleStageFurnitureFit: some View {
        splatRoomLifecycleStageAppear
            .onChange(of: showingFurnitureFit) { _, isOn in
                splatRoomHandleShowingFurnitureFitChange(isOn: isOn)
            }
    }

    private var splatRoomLifecycleStageLabels: some View {
        splatRoomLifecycleStageFurnitureFit
            .onChange(of: furnitureFitSegmentationMode) { _, mode in
                PaafektFullVideoSegmentationExitDiagnostics.logModeChange(
                    viewer: "SplatRoomView",
                    mode: mode,
                    showingFurnitureFit: showingFurnitureFit,
                    showFullVideoWithIdentifications: showFullVideoWithIdentifications
                )
            }
            .onChange(of: selectedFurnitureFitLabels) { oldLabels, newLabels in
                restoreFullVideoIdentifyAfterSegmentPinsLost(oldLabels: oldLabels, newLabels: newLabels)
            }
    }

    private var splatRoomLifecycleStagePlacementSignals: some View {
        splatRoomLifecycleStageLabels
            .onChange(of: segmentedFurnitureMeanSRGB) { _, _ in updateRoomPlacementIntelligence() }
            .onChange(of: detectedFurnitureWidth) { _, _ in updateRoomPlacementIntelligence() }
            .onChange(of: detectedFurnitureHeightAR) { _, _ in updateRoomPlacementIntelligence() }
            .onChange(of: furnitureProportionalHeightMeters) { _, _ in updateRoomPlacementIntelligence() }
            .onChange(of: realFurnitureHeight) { _, _ in updateRoomPlacementIntelligence() }
            .onChange(of: roomModel) { _, _ in updateRoomPlacementIntelligence() }
            .onChange(of: enhancedRoomMetadata) { _, _ in updateRoomPlacementIntelligence() }
    }

    private var splatRoomLifecycleStageBrainAr: some View {
        splatRoomLifecycleStagePlacementSignals
            .onChange(of: brainArAssistedSizingEnabled) { _, enabled in
                splatRoomHandleBrainArAssistedSizingChange(enabled: enabled)
            }
    }

    private var splatRoomLifecycleStageLoadingLog: some View {
        splatRoomLifecycleStageBrainAr
            .onChange(of: isLoading) { _, loading in
                splatRoomHandleIsLoadingForDefaultARCamera(loading: loading)
            }
    }

    private var splatRoomSheetAndLifecycleView: some View {
        splatRoomLifecycleStageLoadingLog
            .onDisappear {
                splatRoomPerformOnDisappear()
            }
    }

    private var splatRoomAfterSaveRoomAlert: some View {
        splatRoomSheetAndLifecycleView
            .sheet(isPresented: $showRoomNameInput) {
                PaafektNameRoomSheet(
                    isPresented: $showRoomNameInput,
                    roomName: $roomName,
                    onSave: { startSavingRoom() }
                )
            }
    }

    private var splatRoomAfterSaveResultAlert: some View {
        splatRoomAfterSaveRoomAlert
            .overlay {
                if showSaveErrorNotice {
                    PaafektErrorNotice(isPresented: $showSaveErrorNotice, message: saveAlertMessage)
                }
            }
            .overlay(alignment: .bottom) {
                if showSaveSuccessSnackbar {
                    PaafektRoomSavedSnackbar(
                        message: saveSuccessSnackbarMessage,
                        isShowing: $showSaveSuccessSnackbar
                    )
                }
            }
    }

    private var splatRoomAfterCalibrationRejectAlert: some View {
        splatRoomAfterSaveResultAlert
            .alert(L10n.RoomViewer.checkMeasurement, isPresented: $showCalibrationRejectAlert) { Button(L10n.Common.ok, role: .cancel) { } } message: { Text(calibrationRejectMessage) }
    }

    private var splatRoomAfterDiscardAlert: some View {
        splatRoomAfterCalibrationRejectAlert
            .alert(L10n.RoomPreview.unsavedTitle, isPresented: $showDiscardUnsavedAlert) {
                Button(L10n.RoomPreview.stay, role: .cancel) {}
                Button(L10n.RoomPreview.leave, role: .destructive) {
                    dismiss()
                }
            } message: {
                Text(L10n.RoomPreview.unsavedMessage)
            }
    }

    private var splatRoomAlertsAndOverlayView: some View {
        splatRoomAfterDiscardAlert
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
                splatRoomHandleIsLoadingForHints(loading: loading)
            }
            // Omit `.leading` so the interactive pop gesture is not deferred behind splat gestures (saved rooms).
            .defersSystemGestures(on: [.top, .trailing])
            .disableBackSwipeIf(allowSave)
    }

    private var splatRoomBody: some View {
        splatRoomAlertsAndOverlayView
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
            initialSplatRoomYaw: initialSplatRoomYaw,
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
                    logSplatRoomDimensionApproaches(metalBounds: bounds)
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
                        logDebug("⏱️ [SplatLoad] metadata_persisted source=SplatRoomView file=\(viewerPlyURL.lastPathComponent) type=hint")
                    } catch {
                        logDebug("❌ [SplatRoomView] Failed to persist splat load hint: \(error.localizedDescription)")
                    }
                }
            },
            measurementHost: splatMeasurementHost
        )
        .ignoresSafeArea()
        .onChange(of: isLoading) { _, loading in
            guard !loading else { return }
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.28) {
                logDebug("📐 [SplatRoomView] Room geometry extraction remains disabled")
            }
        }
        .onAppear {
            guard !allowSave else { return }
            Task { @MainActor in
                await Task.yield()
                metalSplatterZoom = 1.0
            }
        }
        .paafektImmersiveRoomSummonTap(
            chrome: immersiveChrome,
            enabled: !(showingFurnitureFit && showFullVideoWithIdentifications),
            hideForCapture: isCapturingSnapshot,
            onRestingTap: {
                if showingFurnitureFit,
                   showFullVideoWithIdentifications,
                   furnitureFitSegmentationMode == .segmentSelected {
                    activateSelectedFurnitureSegmentation()
                } else {
                    immersiveChrome.summon()
                }
            }
        )
    }

    private func logSplatRoomDimensionApproaches(metalBounds _: RoomBounds) {
        logDebug("📐 [SplatRoomView] Skipped legacy Splat room dimension comparison logging")
    }

    private func seedFrontWallDimensionsFromPlyBoundsIfNeeded() {}

    // MARK: - Furniture Fit toggle

    private func toggleFurnitureFit() {
        if showingFurnitureFit {
            showingFurnitureFit = false
        } else {
            showFullVideoWithIdentifications = false
            furnitureFitInitialSegmentationDone = false
            furnitureFitSegmentationMode = .segmentPrimary
            furnitureFitShowIdentifyLivePreview = true
            selectedFurnitureFitLabels = []
            showingFurnitureFit = true
            selectionJustBecamePrimary = true
            scheduleOnboardingHintAutoDismiss()
        }
    }

    private func restoreFullVideoIdentifyAfterSegmentPinsLost(oldLabels: [String], newLabels: [String]) {
        guard showingFurnitureFit else { return }
        guard furnitureFitSegmentationMode == .segmentSelected else { return }
        guard newLabels.isEmpty, !oldLabels.isEmpty else { return }
        showFullVideoWithIdentifications = true
        furnitureFitSegmentationMode = .identifyOnly
        furnitureFitShowIdentifyLivePreview = true
    }

    private func activateSelectedFurnitureSegmentation() {
        if furnitureFitSegmentationMode == .segmentSelected {
            furnitureFitSegmentationMode = .identifyOnly
            furnitureFitShowIdentifyLivePreview = true
            return
        }
        guard canSegmentSelectedFurniture else { return }
        furnitureFitSegmentationMode = .segmentSelected
    }

    // Old hint infrastructure removed — all hints flow through activeOnboardingHint.

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
            }
        }
    }

    private func warmRoomMeasurementInBackgroundIfNeeded() {
        guard !allowSave else {
            logDebug("[ROOM_DIMS][BACKGROUND] FILE=\(viewerPlyURL.lastPathComponent) SKIP reason=fresh_room_save_deferred")
            return
        }
        guard measuredRoomDimensions == nil else {
            logDebug("[ROOM_DIMS][BACKGROUND] FILE=\(viewerPlyURL.lastPathComponent) SKIP reason=measured_already_available source=\(activeRoomMetersDimensionsSource)")
            return
        }
        guard savedRoomStrictMeters == nil else {
            logDebug("[ROOM_DIMS][BACKGROUND] FILE=\(viewerPlyURL.lastPathComponent) SKIP reason=saved_meta_strict_available source=\(activeRoomMetersDimensionsSource)")
            return
        }
        guard generationRoomMeters == nil else {
            logDebug("[ROOM_DIMS][BACKGROUND] FILE=\(viewerPlyURL.lastPathComponent) SKIP reason=splat_v7_dims_available source=\(activeRoomMetersDimensionsSource)")
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


    private var canSegmentSelectedFurniture: Bool {
        showingFurnitureFit && !selectedFurnitureFitLabels.isEmpty
    }

    private var roomDimensionsHintText: String {
        if let d = activeRoomMetersDimensions {
            return PaafektRoomMeasurementDisplay.rulerHintText(
                width: d.width,
                height: d.height,
                depth: d.depth,
                showFullWHD: true
            ) ?? "ROOM_DIMS unavailable"
        }
        return "ROOM_DIMS unavailable"
    }

    private var canOfferBrainArAssist: Bool {
        QualitySettings.supportsLiDARSceneDepth &&
            appState.qualitySettings.furnitureFitARDepthCompanionRuntimeActive
    }

    /// F — static mode pill shown whenever mode == fullVideo. No timer, no color cycling.
    private var fullVideoModePillOverlay: some View {
        ZStack(alignment: .top) {
            Color.clear.frame(maxWidth: .infinity, maxHeight: .infinity).allowsHitTesting(false)
            if currentRoomViewerMode == .fullVideo {
                PaafektHintChip(
                    systemImage: "hand.tap.fill",
                    text: L10n.RoomViewer.fullVideoFurnitureTapHint,
                    maxWidth: 280
                )
                .padding(.top, activeRoomMetersDimensions != nil ? 36 : 12)
            }
        }
        .allowsHitTesting(false)
        .opacity(isCapturingSnapshot ? 0 : 1)
        .zIndex(99_999)
    }

    private var splatRestingMeasurementPillText: String? {
        guard let d = activeRoomMetersDimensions else { return nil }
        return PaafektRoomMeasurementDisplay.restingPillText(
            width: d.width,
            height: d.height,
            depth: d.depth,
            emphasizeHeight: false
        )
    }

    private var splatMorphingPrimaryAction: PaafektMorphingPrimaryAction {
        PaafektMorphingPrimaryActionResolver.resolve(
            showingFurnitureFit: showingFurnitureFit,
            showFullVideoWithIdentifications: showFullVideoWithIdentifications,
            segmentationMode: furnitureFitSegmentationMode,
            hasSelectedObject: !selectedFurnitureFitLabels.isEmpty
        )
    }

    private var splatMorphingPrimaryDisabled: Bool {
        isLoading || (splatMorphingPrimaryAction == .segment && !canSegmentSelectedFurniture)
    }

    private func handleSplatMorphingPrimaryTap() {
        immersiveChrome.noteChromeInteraction()
        let action = splatMorphingPrimaryAction
        guard action != .segment || canSegmentSelectedFurniture else { return }
        PaafektMorphingPrimaryActionHandler.perform(
            action,
            enterFit: toggleFurnitureFit,
            exitFit: toggleFurnitureFit,
            segment: activateSelectedFurnitureSegmentation,
            finishSegmentation: activateSelectedFurnitureSegmentation
        )
    }

    private var splatImmersiveChromeOverlay: some View {
        PaafektImmersiveViewerChromeStack(
            chrome: immersiveChrome,
            onBack: handleSplatRoomBackTap,
            morphingPrimaryAction: splatMorphingPrimaryAction,
            onMorphingPrimary: handleSplatMorphingPrimaryTap,
            morphingPrimaryDisabled: splatMorphingPrimaryDisabled,
            onSave: allowSave ? {
                immersiveChrome.noteChromeInteraction()
                roomName = ""
                showRoomNameInput = true
            } : nil,
            saveDisabled: isLoading || isSavingRoom,
            measurementText: splatRestingMeasurementPillText,
            hideForCapture: isCapturingSnapshot
        ) {
            PaafektImmersiveSummonedToolbar(chrome: immersiveChrome) {
                HStack(spacing: Theme.Space.sm) {
                    PaafektViewerToolbarIconButton(
                        systemName: "viewfinder",
                        accessibilityLabel: L10n.RoomViewer.recenterView
                    ) {
                        immersiveChrome.noteChromeInteraction()
                        splatMeasurementHost.recenterSplatRoomCamera()
                    }
                    PaafektViewerToolbarIconButton(
                        systemName: "ruler",
                        accessibilityLabel: L10n.RoomViewer.checkMeasurement
                    ) {
                        immersiveChrome.noteChromeInteraction()
                        onSplatRoomDimensionsRulerTapped()
                    }
                    PaafektViewerToolbarIconButton(
                        systemName: "hand.pinch",
                        accessibilityLabel: L10n.RoomViewer.pinchGestureHintExplanation
                    ) {
                        immersiveChrome.noteChromeInteraction()
                        showHintsOnDemand()
                    }
                    PaafektViewerToolbarIconButton(
                        systemName: "square.stack.3d.up",
                        accessibilityLabel: L10n.RoomViewer.displayAllHelpers
                    ) {
                        immersiveChrome.noteChromeInteraction()
                        showHintsOnDemand()
                    }
                    if !selectedFurnitureFitLabels.isEmpty {
                        Button {
                            immersiveChrome.noteChromeInteraction()
                            NotificationCenter.default.post(
                                name: NSNotification.Name("FurnitureFitClearSelectedObjects"),
                                object: nil
                            )
                        } label: {
                            Text(selectedFurnitureChipTitle)
                                .font(.caption2)
                                .fontWeight(.semibold)
                                .foregroundColor(.white)
                                .padding(.horizontal, 8)
                                .padding(.vertical, 4)
                                .background(Capsule().fill(Theme.Palette.viewerCapsuleFill))
                        }
                        .buttonStyle(.plain)
                    }
                    if showingFurnitureFit {
                        PaafektViewerToolbarIconButton(
                            systemName: "text.viewfinder",
                            isActive: showFullVideoWithIdentifications,
                            accessibilityLabel: L10n.Settings.fullVideoWithIdentifications
                        ) {
                            immersiveChrome.noteChromeInteraction()
                            toggleFullVideoIdentifications()
                        }
                        if canOfferBrainArAssist {
                            PaafektViewerToolbarIconButton(
                                systemName: "arrow.up.left.and.arrow.down.right",
                                isActive: brainArAssistedSizingEnabled,
                                accessibilityLabel: brainArAssistedSizingEnabled
                                    ? L10n.RoomViewer.arSizingDisable
                                    : L10n.RoomViewer.arSizingEnable
                            ) {
                                immersiveChrome.noteChromeInteraction()
                                brainArAssistedSizingEnabled.toggle()
                            }
                        }
                    }
                }
            } heroContent: {
                PaafektImmersiveCompactHeroAction(
                    assetName: "PaafektIconSnapshot",
                    title: L10n.RoomViewer.immersiveCaptureShort,
                    isDisabled: isLoading
                ) {
                    immersiveChrome.noteChromeInteraction()
                    takeScreenshot()
                }
            }
        } summonedExtras: {
            PaafektImmersiveFitClusterRows {
                if showingFurnitureFit {
                    roomIntelligencePlacementCardResetOnExit
                }
                if showingFurnitureFit, shouldShowArFurnitureMeasurementPill {
                    furnitureMeasurementPillContent(showTapHint: false)
                }
            }
        } restingAccessory: {
            EmptyView()
        } persistentOverlay: {
            EmptyView()
        }
        .zIndex(99998)
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

    /// ROOM_DIMS_V7 pre-computed metres from Splat generation — avoids PLY re-measurement.
    private var generationRoomMeters: (width: Float, height: Float, depth: Float)? {
        guard let w = splatRoomMetersW, let h = splatRoomMetersH, let d = splatRoomMetersD,
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
        if generationRoomMeters != nil {
            switch viewerRoomCoordinateFrame {
            case .arWorldMeters:
                return "LIDAR_SWEEP_FUSION"
            case .planarRoomMeters:
                return "PLANAR_ROOM"
            case .depthAnythingImageDepthMeters:
                return "DEPTH_ANYTHING_METRIC"
            case .classicSplatPly, .canonicalSplatPly:
                return "GENERATED_ROOM_DIMS"
            }
        }
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
        if viewerRoomCoordinateFrame.usesNativeMeterSceneUnits, plySceneExtent != nil {
            return viewerRoomCoordinateFrame == .planarRoomMeters ? "PLY_PLANAR_ROOM_METERS" : "PLY_AR_WORLD_METERS"
        }
        if inferredMetersFromPlyScene != nil { return "PLY_INFERRED_REFERENCE_HEIGHT" }
        return "UNAVAILABLE"
    }

    private var hasCalculatedRoomMeasurements: Bool {
        savedRoomStrictMeters != nil ||
            generationRoomMeters != nil ||
            persistedEnhancedRoomMeters != nil ||
            measuredRoomDimensions != nil
    }

    /// Trimmed / Splat AABB in **scene units** (Metal bounds preferred, else init-time PLY AABB from generation).
    private var plySceneExtent: (width: Float, height: Float, depth: Float)? {
        if let b = metalBounds, b.width > 0.05, b.height > 0.05, b.depth > 0.05 {
            return (b.width, b.height, b.depth)
        }
        if let w = splatPlyAabbW, let h = splatPlyAabbH, let d = splatPlyAabbD,
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

    /// Nav, Furniture Fit, save, and overlay: saved meta → Splat V7 → partial meta + scene depth → PLY-inferred metres.
    private var activeRoomMetersDimensions: (width: Float, height: Float, depth: Float)? {
        if let triple = savedRoomStrictMeters { return triple }
        if let triple = generationRoomMeters { return triple }
        if let triple = persistedEnhancedRoomMeters { return triple }
        if let measured = measuredRoomDimensions { return (measured.width, measured.height, measured.depth) }
        if viewerRoomCoordinateFrame.usesNativeMeterSceneUnits, let p = plySceneExtent {
            return p
        }
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

    private var initialSplatRoomYaw: Float {
        if viewerRoomCoordinateFrame.usesNativeMeterSceneUnits {
            return 0
        }
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
            !showSaveErrorNotice &&
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
            mlModel: rtmdetService.model,
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
            splatRoomMeasurementHost: splatMeasurementHost,
            arAssistedSizingEnabled: brainArAssistedSizingEnabled && canOfferBrainArAssist,
            manualFurnitureHeightOverrideMeters: realFurnitureHeight,
            segmentationMode: furnitureFitSegmentationMode,
            onSelectedClassLabelsChanged: { labels in
                selectedFurnitureFitLabels = labels
            },
            onSegmentationModeChangeRequested: { mode in
                // RTMDET-TAP-SEGMENT-OK (verified working 2026-06-10): canonical wiring. INVARIANT: every
                // screen hosting FurnitureFitUIView MUST pass this closure or tap-to-segment silently
                // reverts (updateUIView re-applies the stale @State). See ModelViewer/GLB/MeshRoomView.
                logDebug("BRAIN FLOW: FurnitureFit requested segmentationMode=\(mode)")
                furnitureFitSegmentationMode = mode
            },
            showIdentifyLivePreview: furnitureFitShowIdentifyLivePreview,
            showFullVideoWithIdentificationsOverride: showFullVideoWithIdentifications
        )
        .ignoresSafeArea()
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
        return VStack(spacing: Theme.Space.sm) {
            PaafektRoomMeasurementPill(
                primaryText: L10n.RoomViewer.furnitureHeightEstimate(realFurnitureHeight ?? displayH),
                secondaryText: calibratedRoomHeight.map { L10n.RoomViewer.roomMetersShort($0) },
                primaryColor: realFurnitureHeight != nil ? Theme.Palette.success : Theme.Palette.textPrimary
            )
            if showTapHint {
                Text(L10n.RoomViewer.tapToCalibrate)
                    .font(Theme.Typo.caption())
                    .foregroundStyle(Theme.Palette.textSecondary)
            }
        }
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
            PaafektImmersivePlacementIntelligenceRow(
                isExpanded: isPlacementIntelligenceExpanded,
                ringColor: placementIntelligenceRingColor(fit: fit),
                onToggle: {
                    withAnimation(.easeInOut(duration: 0.18)) {
                        isPlacementIntelligenceExpanded.toggle()
                    }
                }
            ) {
                placementIntelligenceExpandedContent(dimensions: dimensions, fit: fit, aesthetic: aesthetic)
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

    @ViewBuilder private var allOverlaysLayer: some View {
        ZStack {
            // Full-video task helper stays visible even after chrome auto-hides.
            if !isLoading {
                fullVideoModePillOverlay
            }
            if !isLoading, immersiveChrome.isSummoned {
                topTrailingPinchHintOverlay
                topTrailingActionButtonsOverlay
                fullVideoToolbarHelperOverlay
            }
            if isLoading { loadingOverlayView }
            errorOverlayView
            if showingFurnitureFit { furnitureFitOverlayView }
            if isMeasuringRoomDimensions { measureRoomProgressOverlay }
            if isSavingRoom { saveRoomProgressOverlay }
            if showFurnitureDimensionsInput, supportsMetricFurnitureMeasurementUI {
                calibrationOverlayView
                    .onAppear { calibrationBaselineDetectedHeight = detectedFurnitureHeightAR }
                    .onDisappear { calibrationBaselineDetectedHeight = nil }
            }
            if showWallCalibration, supportsMetricFurnitureMeasurementUI {
                wallCalibrationOverlay
            }
            splatImmersiveChromeOverlay
            PaafektViewerOnboardingLayer(
                isReady: !isLoading,
                isChromeSummoned: immersiveChrome.isSummoned,
                heroHintBottomInset: showingFurnitureFit ? 220 : 172,
                replayTeachingHints: $replayTeachingHints
            )
                .zIndex(100_000)
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
                    let composed = compositeSplatRoomSnapshotWithFurnitureFitIfNeeded(splatImage: image)
                    saveSplatRoomSnapshotToPhotos(composed)
                } else {
                    logDebug("📸 Metal capture unavailable; falling back to window hierarchy...")
                    captureSplatRoomSnapshotViaDrawHierarchy()
                    return
                }
                isCapturingSnapshot = false
            }
        }
    }

    private func saveSplatRoomSnapshotToPhotos(_ image: UIImage) {
        let saveBlock = {
            UIImageWriteToSavedPhotosAlbum(image, nil, nil, nil)
            logDebug("✅ Saved Splat Room snapshot to Photos")
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
    private func compositeSplatRoomSnapshotWithFurnitureFitIfNeeded(splatImage: UIImage) -> UIImage {
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

    private func captureSplatRoomSnapshotViaDrawHierarchy() {
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
            saveSplatRoomSnapshotToPhotos(image)
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

    // MARK: - RTMDet model loaded via RTMDetModelService

    // MARK: - Save Room Progress Overlay
    private var saveRoomProgressOverlay: some View {
        PaafektSavingRoomOverlay(
            progress: saveProgress,
            title: saveProgressStatusText,
            subtitle: saveOverlayRoomDimensionsLine
        )
        .overlay(alignment: .bottom) {
            Button(L10n.Common.cancel) {
                cancelSavingRoom()
            }
            .foregroundStyle(Theme.Palette.danger)
            .padding(.bottom, Theme.Space.xxl)
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

    private var splatRoomModalPauseToken: SplatRoomModalPauseToken {
        SplatRoomModalPauseToken(
            showRoomNameInput: showRoomNameInput,
            isSavingRoom: isSavingRoom,
            showSaveErrorNotice: showSaveErrorNotice,
            showDiscardUnsavedAlert: showDiscardUnsavedAlert,
            showCalibrationRejectAlert: showCalibrationRejectAlert,
            showWallCalibration: showWallCalibration,
            showFurnitureDimensionsInput: showFurnitureDimensionsInput,
            supportsMetricFurnitureMeasurementUI: supportsMetricFurnitureMeasurementUI,
            isCapturingSnapshot: isCapturingSnapshot
        )
    }

    /// Pauses ARKit while modal UI is up so `TextField` / alerts are not competing with per-frame `ARSession` main-queue work.
    private func syncModalHeavyWorkPauseForSplatRoomUI() {
        let furnitureCalibSheet =
            showFurnitureDimensionsInput && supportsMetricFurnitureMeasurementUI
        let pause =
            showRoomNameInput ||
            isSavingRoom ||
            showSaveErrorNotice ||
            showDiscardUnsavedAlert ||
            showCalibrationRejectAlert ||
            showWallCalibration ||
            furnitureCalibSheet ||
            isCapturingSnapshot
        guard pause != splatRoomUIPauseApplied else { return }
        splatRoomUIPauseApplied = pause
        splatMeasurementHost.setModalHeavyWorkPaused(pause)
    }

    private func startSavingRoom() {
        let trimmedRoomName = roomName.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedRoomName.isEmpty else { return }
        guard !modelManager.hasSavedRoomNameConflict(trimmedRoomName) else {
            saveAlertMessage = L10n.RoomViewer.duplicateRoomName
            saveWasSuccessful = false
            showSaveErrorNotice = true
            return
        }

        let savedName = trimmedRoomName
        roomName = trimmedRoomName
        logDebug("💾 [SplatRoomView] Starting room save: \(savedName)")

        // Dismiss the name-entry alert before presenting the full-screen save progress overlay.
        showRoomNameInput = false

        Task {
            await MainActor.run {
                if showingFurnitureFit {
                    logDebug("💾 [SplatRoomView] Save: stopping Furniture Fit before persist")
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

            let shouldMeasureRoomDimensionsAfterSave = await MainActor.run {
                savedRoomStrictMeters == nil && generationRoomMeters == nil && measuredRoomDimensions == nil
            }
            let savedPlyFileName = modelManager.savedPLYFileName(forRoomName: savedName)
            let savedPlyURL = modelManager.savedPLYURL(forRoomName: savedName)

            let (fallbackDimensions, sceneExtentForMeta) = await MainActor.run {
                (activeRoomMetersDimensions, plySceneExtent)
            }
            let roomW = fallbackDimensions?.width
            let roomH = fallbackDimensions?.height
            let roomD = fallbackDimensions?.depth
            let roomDimsApproachForSave: String? = await MainActor.run {
                if generationRoomMeters != nil { return "room_dims_v7_splat" }
                if measuredRoomDimensions != nil { return "room_dims_v7_async" }
                return nil
            }
            if let roomW, let roomH, let roomD {
                logDebug(
                    "🟢 [SplatRoomView] Save: ROOM_DIMS W×H×D=" +
                        "\(String(format: "%.3f", roomW))×\(String(format: "%.3f", roomH))×\(String(format: "%.3f", roomD))m"
                )
            } else {
                logDebug("🔴 [SplatRoomView] Save: room dimensions unavailable")
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
                    roomSceneDepth: sceneExtentForMeta?.depth,
                    measureMissingRoomDimensions: false,
                    isClassicPly: viewerUsesClassicPlyBehavior,
                    roomCoordinateFrame: viewerRoomCoordinateFrame
                ) { success, error in
                    logDebug(success ? "✅ [SplatRoomView] Room saved" : "❌ [SplatRoomView] Save failed: \(error ?? "unknown")")
                    Task { @MainActor in
                        if success, let metadata = metadataForSave {
                            metadata.printSaveDiagnostics()
                            do {
                                try modelManager.saveEnhancedMetadata(metadata, forSavedRoomNamed: savedName, fileType: .ply)
                                logDebug("✅ [SplatRoomView] Save: enhanced metadata persisted for saved room")
                            } catch {
                                logDebug("❌ [SplatRoomView] Failed to save enhanced metadata for saved room: \(error.localizedDescription)")
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
                        if success {
                            startPostSaveRoomDimensionMeasurementIfNeeded(
                                shouldMeasure: shouldMeasureRoomDimensionsAfterSave,
                                savedFileName: savedPlyFileName,
                                savedRoomURL: savedPlyURL,
                                treatAsClassicPly: viewerUsesClassicPlyBehavior
                            )
                        }
                        saveProgress = 1.0
                        withAnimation(.easeOut(duration: 0.3)) {
                            isSavingRoom = false
                        }
                        DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
                            if success {
                                saveSuccessSnackbarMessage = L10n.RoomViewer.saveSuccess(savedName)
                                saveWasSuccessful = true
                                withAnimation { showSaveSuccessSnackbar = true }
                                isDismissing = true
                                DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) {
                                    NotificationCenter.default.post(name: NSNotification.Name("DismissPhotoRoomSheet"), object: nil)
                                }
                            } else {
                                saveAlertMessage = L10n.RoomViewer.saveFailed(error ?? L10n.RoomViewer.saveErrorUnknown)
                                saveWasSuccessful = false
                                showSaveErrorNotice = true
                            }
                            roomName = ""
                        }
                        continuation.resume()
                    }
                }
            }
        }
    }

    private func startPostSaveRoomDimensionMeasurementIfNeeded(
        shouldMeasure: Bool,
        savedFileName: String,
        savedRoomURL: URL,
        treatAsClassicPly: Bool
    ) {
        guard shouldMeasure else {
            logDebug("[ROOM_DIMS][SAVE_ASYNC] FILE=\(savedRoomURL.lastPathComponent) SKIP reason=dimensions_already_available")
            return
        }

        let manager = modelManager
        Task(priority: .utility) {
            logDebug("[ROOM_DIMS][SAVE_ASYNC] FILE=\(savedRoomURL.lastPathComponent) START")
            let measured = await manager.measureRoomDimensionsAsync(
                forPly: savedRoomURL,
                treatAsClassicPly: treatAsClassicPly
            )
            if let measured {
                do {
                    try manager.mergeMeasuredRoomDimensionsIntoSavedRoomMetadata(
                        fileName: savedFileName,
                        modelFileExtension: "ply",
                        measured: measured
                    )
                    logDebug(
                        "[ROOM_DIMS][SAVE_ASYNC] FILE=\(savedRoomURL.lastPathComponent) " +
                        "APPROACH=\(measured.approach.uppercased()) SHOT=\(measured.shotType) " +
                        "HAS_FOCAL=\(measured.usedFocal) TILT_DEG=\(String(format: "%.2f", measured.tiltDegrees)) " +
                        "TILT_RELIABLE=\(measured.tiltReliable) CUBOID_RATIO=\(String(format: "%.4f", measured.cuboidRatio)) " +
                        "THRESHOLD=\(String(format: "%.4f", measured.cuboidThreshold)) " +
                        "FILL_W=\(String(format: "%.4f", measured.fillWidth)) BLEND=\(String(format: "%.4f", measured.blend)) " +
                        "W=\(String(format: "%.4f", measured.width)) " +
                        "H=\(String(format: "%.4f", measured.height)) " +
                        "D=\(String(format: "%.4f", measured.depth))"
                    )
                    await MainActor.run {
                        manager.refreshModels()
                    }
                } catch {
                    logDebug("[ROOM_DIMS][SAVE_ASYNC] FILE=\(savedRoomURL.lastPathComponent) MERGE_FAILED error=\(error.localizedDescription)")
                }
            } else {
                logDebug("[ROOM_DIMS][SAVE_ASYNC] FILE=\(savedRoomURL.lastPathComponent) DONE unavailable")
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
        logDebug("❌ [SplatRoomView] Room save cancelled")
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
                logDebug("📐 [SplatRoomView] Loaded saved room geometry metadata")
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
            logDebug("📐 [SplatRoomView] Loaded fresh room geometry metadata")
        }
    }

    private func scheduleRoomGeometryExtractionIfNeeded() {
        logDebug("📐 [SplatRoomView] Room geometry extraction disabled")
    }

    private func triggerRoomGeometryExtractionIfNeeded(force: Bool = false) {
        let _ = force
        logDebug("📐 [SplatRoomView] RoomGeometryEngine / RANSAC path disabled")
    }

    private func persistEnhancedRoomMetadataIfPossible(_ metadata: EnhancedRoomMetadata) {
        do {
            try modelManager.saveEnhancedMetadata(metadata, nextTo: viewerPlyURL)
        } catch {
            logDebug("❌ [SplatRoomView] Failed to persist enhanced room metadata: \(error.localizedDescription)")
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
        SplatRoomView(plyURL: URL(fileURLWithPath: "/sample.ply"))
    }
}
