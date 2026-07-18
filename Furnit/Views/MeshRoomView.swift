import SwiftUI
import WebKit
import UIKit
import CoreML

/// WebGL-based mesh room viewer - renders box room geometry using Three.js
/// Exports to GLTF/GLB format for universal 3D viewing
struct MeshRoomView: View {
    let roomWidth: Float
    let roomHeight: Float
    let roomDepth: Float
    let frontWallImage: UIImage
    let photoOrientation: PhotoOrientation

    // Boundary coordinates (normalized 0-1) for texturing walls
    var leftX: CGFloat = 0.12
    var rightX: CGFloat = 0.88
    var ceilingY: CGFloat = 0.15
    var floorY: CGFloat = 0.85

    /// Set when opening a saved mesh room from Home (the detector ratio calibration).
    var savedRoomModel: USDZModel? = nil

    @Environment(\.dismiss) private var dismiss
    @EnvironmentObject var authManager: AuthenticationManager

    @State private var isLoading = true
    @State private var error: String? = nil

    // Room name for saving
    @State private var showRoomNameInput = false
    @State private var roomName = ""
    @State private var isSavingRoom = false
    @State private var saveAlertMessage = ""
    @State private var saveWasSuccessful = false
    @State private var showSaveSuccessSnackbar = false
    @State private var saveSuccessSnackbarMessage = ""
    @State private var showSaveErrorNotice = false
    @State private var showDiscardUnsavedAlert = false

    // Brain mode (furniture detection)
    @State private var showingFurnitureFit = false
    @State private var furnitureFitSegmentationMode: FurnitureFitSegmentationMode = .identifyOnly
    @State private var furnitureFitShowIdentifyLivePreview = true
    @State private var selectedFurnitureFitLabels: [String] = []
    @State private var furnitureFitInitialSegmentationDone = false
    @State private var brainArAssistedSizingEnabled = false
    @ObservedObject private var rtmdetService = RTMDetModelService.shared
    @ObservedObject private var appState = AppStateManager.shared
    @State private var detectedFurnitureHeightAR: Float?
    @State private var detectedFurnitureWidth: Float?
    @State private var furnitureProportionalHeightMeters: Float?
    @State private var latestFitCheckResult: FitCheckResult?
    @State private var latestAestheticScore: AestheticScore?
    @State private var segmentedFurnitureMeanSRGB: SIMD3<Float>?
    @State private var isPlacementIntelligenceExpanded = false
    @State private var showFurnitureDimensionsInput = false
    @State private var inputFurnitureHeight: String = ""
    @State private var realFurnitureHeight: Float?
    @State private var showCalibrationRejectAlert = false
    @State private var calibrationRejectMessage = ""
    @State private var roomCalibrationScaleFactor: Float = 1.0
    @State private var calibrationBaselineDetectedHeight: Float?

    // WebView reference for GLTF export
    @State private var webView: WKWebView?

    // Model manager for saving rooms (also used for the detector ratio metadata merge when viewing saved room)
    @StateObject private var modelManager = USDZModelManager()

    @State private var replayTeachingHints = false
    @State private var arSizingHintExplanationVisible = false
    @State private var arSizingHintHideTextTask: Task<Void, Never>?
    @State private var arSizingHintRequiresBrain = false
    /// Ruler tap: show W×H×D chip below top safe area (matches Splat room dimensions hint).
    @State private var roomDimensionsHintVisible = false
    @State private var roomDimensionsHintHideTask: Task<Void, Never>?
    @State private var showFullVideoWithIdentifications = false
    @State private var fullVideoFurnitureTapHintVisible = false
    @State private var fullVideoSelectionHelperVisible = false
    @State private var fullVideoSelectionHelperHideTask: Task<Void, Never>?
    /// Pinch-zoom hint (top-left with D-pad) — same as ``SplatRoomView``.
    @State private var pinchHintExplanationVisible = false
    @State private var pinchHintHideTextTask: Task<Void, Never>?
    @StateObject private var immersiveChrome = PaafektViewerChromeController()

    private var canOfferBrainArAssist: Bool {
        QualitySettings.supportsLiDARSceneDepth &&
            appState.qualitySettings.furnitureFitARDepthCompanionRuntimeActive
    }

    private var canSegmentSelectedFurniture: Bool {
        showingFurnitureFit && !selectedFurnitureFitLabels.isEmpty
    }

    private var supportsMetricFurnitureMeasurementUI: Bool {
        QualitySettings.supportsLiDARSceneDepth
    }

    private var calibratedRoomWidth: Float {
        roomWidth * roomCalibrationScaleFactor
    }

    private var calibratedRoomHeight: Float {
        roomHeight * roomCalibrationScaleFactor
    }

    private var calibratedRoomDepth: Float {
        roomDepth * roomCalibrationScaleFactor
    }

    private var shouldShowArFurnitureMeasurementPill: Bool {
        showingFurnitureFit &&
            brainArAssistedSizingEnabled &&
            supportsMetricFurnitureMeasurementUI &&
            (detectedFurnitureHeightAR?.isFinite == true) &&
            ((detectedFurnitureHeightAR ?? 0) > 0.05)
    }

    private var authoritativeRoomModelForMetrics: RoomModel? {
        guard calibratedRoomWidth > 0.05,
              calibratedRoomHeight > 0.05,
              calibratedRoomDepth > 0.05 else {
            return nil
        }
        return MeshPlacementIntelligenceRoomStub.axisAlignedBoxMeters(
            width: calibratedRoomWidth,
            height: calibratedRoomHeight,
            depth: calibratedRoomDepth
        )
    }

    private var placementIntelligenceHasFurnitureSignal: Bool {
        if let width = detectedFurnitureWidth, width.isFinite, width > 0.05 { return true }
        if segmentedFurnitureMeanSRGB != nil { return true }
        if derivedDetectedFurnitureDimensionsForRoomIntelligence() != nil { return true }
        return false
    }

    // MARK: - Main layout (split for Swift compiler type-check)

    @ViewBuilder
    private var meshRoomLoadingOverlay: some View {
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
    }

    @ViewBuilder
    private var meshRoomSavingOverlay: some View {
        if isSavingRoom {
            VStack(spacing: 16) {
                ProgressView()
                    .scaleEffect(1.5)
                    .tint(.white)

                Text(L10n.RoomViewer.exporting3DModelEllipsis)
                    .font(.subheadline)
                    .foregroundColor(.white)
            }
            .padding(24)
            .background(Color.black.opacity(0.7))
            .cornerRadius(12)
        }
    }

    @ViewBuilder
    private var meshRoomErrorOverlay: some View {
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
    }

    @ViewBuilder
    private var meshRoomCameraChromeWhenReady: some View {
        if !isLoading, immersiveChrome.isSummoned {
            topTrailingPinchTapAndSizingHintsOverlay
            navigationTeachingHintBottomOverlay
        }
    }

    @ViewBuilder
    private var meshFurnitureFitCameraOverlay: some View {
        if showingFurnitureFit {
            FurnitureFitUIView(
                capturedImage: .constant(nil),
                roomImage: nil,
                mlModel: rtmdetService.model,
                processInterval: 0.07,
                active: true,
                lockedOrientation: photoOrientation,
                roomWidthMeters: calibratedRoomWidth,
                roomHeightMeters: calibratedRoomHeight,
                roomDepthMeters: calibratedRoomDepth,
                onFurnitureSizeEstimated: { estimate in
                    detectedFurnitureWidth = estimate.widthMeters > 0.05 ? estimate.widthMeters : nil
                    if let arHeight = estimate.arHeightMeters,
                       arHeight.isFinite,
                       arHeight > 0.05 {
                        detectedFurnitureHeightAR = arHeight
                        furnitureProportionalHeightMeters = nil
                    } else {
                        detectedFurnitureHeightAR = nil
                        furnitureProportionalHeightMeters = estimate.heightMeters > 0.05 ? estimate.heightMeters : nil
                    }
                },
                suppressStartupProgress: furnitureFitInitialSegmentationDone,
                onFirstSegmentationComplete: { furnitureFitInitialSegmentationDone = true },
                onSegmentationMaskMeanColorSRGB: { meanSRGB in
                    segmentedFurnitureMeanSRGB = meanSRGB
                },
                arAssistedSizingEnabled: brainArAssistedSizingEnabled && canOfferBrainArAssist,
                manualFurnitureHeightOverrideMeters: realFurnitureHeight,
                segmentationMode: furnitureFitSegmentationMode,
                onSelectedClassLabelsChanged: { labels in
                    selectedFurnitureFitLabels = labels
                },
                onSegmentationModeChangeRequested: { mode in
                    // RTMDET-TAP-SEGMENT-OK (verified working 2026-06-10): tapping furniture renders the
                    // segmented cutout. INVARIANT: every screen hosting FurnitureFitUIView MUST pass this
                    // closure. Without it, a tap sets the view's segmentationMode internally but the @State
                    // stays .identifyOnly, so the next updateUIView clobbers it back and the cutout never
                    // renders. If tap-to-segment breaks again, check this wiring first. Mirror SplatRoomView.
                    logDebug("BRAIN FLOW: FurnitureFit requested segmentationMode=\(mode)")
                    furnitureFitSegmentationMode = mode
                },
                showIdentifyLivePreview: furnitureFitShowIdentifyLivePreview,
                showFullVideoWithIdentificationsOverride: showFullVideoWithIdentifications
            )
            .ignoresSafeArea()
            .zIndex(100)
        }
    }

    @ViewBuilder
    private var meshRoomCalibrationGateOverlay: some View {
        if showFurnitureDimensionsInput, supportsMetricFurnitureMeasurementUI {
            calibrationOverlayView
                .onAppear { calibrationBaselineDetectedHeight = detectedFurnitureHeightAR }
                .onDisappear { calibrationBaselineDetectedHeight = nil }
        }
    }


    private var meshRoomMainZStack: some View {
        ZStack {
            // WebGL mesh viewer - OrbitControls in Three.js handles touch directly
            MeshWebGLView(
                roomWidth: roomWidth,
                roomHeight: roomHeight,
                roomDepth: roomDepth,
                frontWallImage: frontWallImage,
                photoOrientation: photoOrientation,
                leftX: leftX,
                rightX: rightX,
                ceilingY: ceilingY,
                floorY: floorY,
                webViewRef: $webView,
                onLoaded: {
                    isLoading = false
                },
                onGLBExported: { glbData in
                    saveGLBRoom(glbData: glbData)
                }
            )
            .ignoresSafeArea()
            .allowsHitTesting(!isLoading)
            .paafektImmersiveRoomSummonTap(
                chrome: immersiveChrome,
                enabled: !(showingFurnitureFit && showFullVideoWithIdentifications),
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

            meshRoomLoadingOverlay
            meshRoomSavingOverlay
            meshRoomErrorOverlay
            meshRoomCameraChromeWhenReady
            meshFurnitureFitCameraOverlay

            roomDimensionsHintOverlay
            // Full-video task helper stays visible even after chrome auto-hides.
            fullVideoFurnitureTapHintOverlay
            if immersiveChrome.isSummoned {
                fullVideoModeFloatingButtonOverlay
                fullVideoToolbarHelperOverlay
            }
            meshRoomCalibrationGateOverlay
            meshImmersiveChromeOverlay
            PaafektViewerOnboardingLayer(
                isReady: !isLoading,
                isChromeSummoned: immersiveChrome.isSummoned,
                heroHintBottomInset: showingFurnitureFit ? 220 : 172,
                replayTeachingHints: $replayTeachingHints
            )
                .zIndex(100_000)
        }
    }

    private var meshRoomWithNavigationChrome: some View {
        meshRoomMainZStack
            .background(Color.gray)
            .navigationTitle("")
            .navigationBarTitleDisplayMode(.inline)
            .navigationBarBackButtonHidden(true)
            .toolbar(.hidden, for: .navigationBar)
    }

    private func meshRoomPerformOnAppear() {
        // Preload RTMDet when the room opens so the first brain tap can start segmentation without waiting.
        rtmdetService.ensureModelLoaded()
        if photoOrientation == .landscape {
            OrientationLockManager.shared.lockToLandscape()
        } else {
            OrientationLockManager.shared.lockToPortrait()
        }
    }

    private func meshRoomPerformOnDisappear() {
        cancelPinchHintTasks()
        cancelARSizingHintTasks()
        cancelRoomDimensionsHintTasks()
        dismissFullVideoFurnitureTapHint()
        brainArAssistedSizingEnabled = false
        rtmdetService.releaseResources()
        OrientationLockManager.shared.unlock()
    }

    private func meshRoomHandleIsLoadingChange(loading: Bool) {
        if loading {
            cancelPinchHintTasks()
            cancelARSizingHintTasks()
            cancelRoomDimensionsHintTasks()
        }
    }

    private func meshRoomHandleShowingFurnitureFitChange(isOn: Bool) {
        if isOn {
            rtmdetService.ensureModelLoaded()
            if canOfferBrainArAssist {
                showARSizingHint(requiresBrain: false)
            }
            updateRoomPlacementIntelligence()
        } else {
            dismissFullVideoFurnitureTapHint()
            cancelARSizingHintTasks()
            arSizingHintExplanationVisible = false
            brainArAssistedSizingEnabled = false
            showFullVideoWithIdentifications = false
            furnitureFitSegmentationMode = .identifyOnly
            furnitureFitShowIdentifyLivePreview = true
            selectedFurnitureFitLabels = []
            detectedFurnitureWidth = nil
            detectedFurnitureHeightAR = nil
            furnitureProportionalHeightMeters = nil
            latestFitCheckResult = nil
            latestAestheticScore = nil
            segmentedFurnitureMeanSRGB = nil
            isPlacementIntelligenceExpanded = false
            showFurnitureDimensionsInput = false
        }
    }

    private var meshRoomAfterAppearAndLoading: some View {
        meshRoomWithNavigationChrome
            .onAppear {
                meshRoomPerformOnAppear()
            }
            .onChange(of: isLoading) { _, loading in
                meshRoomHandleIsLoadingChange(loading: loading)
            }
            .onChange(of: showingFurnitureFit) { _, isOn in
                meshRoomHandleShowingFurnitureFitChange(isOn: isOn)
            }
            .onChange(of: furnitureFitSegmentationMode) { _, mode in
                PaafektFullVideoSegmentationExitDiagnostics.logModeChange(
                    viewer: "MeshRoomView",
                    mode: mode,
                    showingFurnitureFit: showingFurnitureFit,
                    showFullVideoWithIdentifications: showFullVideoWithIdentifications
                )
            }
    }

    private var meshRoomAfterSelectionObserver: some View {
        meshRoomAfterAppearAndLoading
            .onChange(of: selectedFurnitureFitLabels) { oldLabels, newLabels in
                restoreFullVideoIdentifyAfterSegmentPinsLost(oldLabels: oldLabels, newLabels: newLabels)
            }
    }

    private var meshRoomAfterPlacementObservers: some View {
        meshRoomAfterSelectionObserver
            .onChange(of: segmentedFurnitureMeanSRGB) { _, _ in updateRoomPlacementIntelligence() }
            .onChange(of: detectedFurnitureWidth) { _, _ in updateRoomPlacementIntelligence() }
            .onChange(of: detectedFurnitureHeightAR) { _, _ in updateRoomPlacementIntelligence() }
            .onChange(of: furnitureProportionalHeightMeters) { _, _ in updateRoomPlacementIntelligence() }
    }

    private var meshRoomAfterCalibrationObservers: some View {
        meshRoomAfterPlacementObservers
            .onChange(of: realFurnitureHeight) { _, _ in updateRoomPlacementIntelligence() }
            .onChange(of: roomCalibrationScaleFactor) { _, _ in updateRoomPlacementIntelligence() }
            .onDisappear {
                meshRoomPerformOnDisappear()
            }
    }

    private var meshRoomWithSystemAlerts: some View {
        meshRoomAfterCalibrationObservers
            .sheet(isPresented: $showRoomNameInput) {
                PaafektNameRoomSheet(
                    isPresented: $showRoomNameInput,
                    roomName: $roomName,
                    onSave: { requestGLBExport() }
                )
            }
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
            .alert(L10n.RoomPreview.unsavedTitle, isPresented: $showDiscardUnsavedAlert) {
                Button(L10n.RoomPreview.stay, role: .cancel) {}
                Button(L10n.RoomPreview.leave, role: .destructive) {
                    dismiss()
                }
            } message: {
                Text(L10n.RoomPreview.unsavedMessage)
            }
            .alert(L10n.RoomViewer.checkMeasurement, isPresented: $showCalibrationRejectAlert) {
                Button(L10n.Common.ok, role: .cancel) {}
            } message: {
                Text(calibrationRejectMessage)
            }
    }

    var body: some View {
        meshRoomWithSystemAlerts
            .defersSystemGestures(on: [.top, .trailing])
            .disableBackSwipe()
    }

    // MARK: - Ruler + hint tasks (matches SplatRoomView)

    private var canPresentMeshRoomDimensionsAlert: Bool {
        !showRoomNameInput &&
            !isSavingRoom &&
            !showSaveSuccessSnackbar &&
            !showSaveErrorNotice &&
            !showDiscardUnsavedAlert
    }

    private var meshRoomDimensionsHintText: String {
        L10n.RoomViewer.roomDimensionsWHDManualChip(
            width: calibratedRoomWidth,
            height: calibratedRoomHeight,
            depth: calibratedRoomDepth
        )
    }

    private func furnitureMeasurementPillContent(showTapHint: Bool) -> some View {
        let displayHeight = detectedFurnitureHeightAR ?? 0
        return VStack(spacing: Theme.Space.sm) {
            PaafektRoomMeasurementPill(
                primaryText: L10n.RoomViewer.furnitureHeightEstimate(realFurnitureHeight ?? displayHeight),
                secondaryText: L10n.RoomViewer.roomMetersShort(calibratedRoomHeight),
                primaryColor: realFurnitureHeight != nil ? Theme.Palette.success : Theme.Palette.textPrimary
            )
            if showTapHint {
                Text(L10n.RoomViewer.tapToCalibrate)
                    .font(Theme.Typo.caption())
                    .foregroundStyle(Theme.Palette.textSecondary)
            }
        }
    }

    private var calibrationOverlayView: some View {
        ZStack {
            Color.black.opacity(0.6)
                .ignoresSafeArea()
                .onTapGesture { showFurnitureDimensionsInput = false }
            VStack(spacing: 16) {
                Text(L10n.RoomViewer.calibrateRoomTitle)
                    .font(.headline)
                    .foregroundColor(.white)
                Text(L10n.RoomViewer.enterFurnitureHeightMeters)
                    .font(.caption)
                    .foregroundColor(.gray)
                Text(L10n.RoomViewer.furnitureFullHeightHint)
                    .font(.caption2)
                    .foregroundColor(.gray.opacity(0.9))
                if let height = calibrationBaselineDetectedHeight ?? detectedFurnitureHeightAR {
                    Text(L10n.RoomViewer.detectedMeters(height))
                        .font(.caption2)
                        .foregroundColor(.orange)
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
                    .font(.body.bold())
                    .foregroundColor(.red)
                    .frame(width: 80, height: 40)
                    .background(Color.red.opacity(0.2))
                    .cornerRadius(8)

                    Button(L10n.Common.apply) {
                        applyCalibration()
                    }
                    .font(.body.bold())
                    .foregroundColor(.green)
                    .frame(width: 80, height: 40)
                    .background(Color.green.opacity(0.2))
                    .cornerRadius(8)
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
                    ForEach(1...3, id: \.self) { column in
                        let digit = row * 3 + column
                        Button(action: { appendDigit("\(digit)") }) {
                            Text("\(digit)")
                                .font(.title2.bold())
                                .foregroundColor(.white)
                                .frame(width: 50, height: 44)
                                .background(Color.gray.opacity(0.3))
                                .cornerRadius(8)
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
                    Text(".")
                        .font(.title2.bold())
                        .foregroundColor(.white)
                        .frame(width: 50, height: 44)
                        .background(Color.gray.opacity(0.3))
                        .cornerRadius(8)
                }
                Button(action: { appendDigit("0") }) {
                    Text("0")
                        .font(.title2.bold())
                        .foregroundColor(.white)
                        .frame(width: 50, height: 44)
                        .background(Color.gray.opacity(0.3))
                        .cornerRadius(8)
                }
                Button(action: {
                    if !inputFurnitureHeight.isEmpty {
                        inputFurnitureHeight.removeLast()
                    }
                }) {
                    Image(systemName: "delete.left")
                        .font(.title3.bold())
                        .foregroundColor(.white)
                        .frame(width: 50, height: 44)
                        .background(Color.gray.opacity(0.3))
                        .cornerRadius(8)
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

        if realHeight >= currentRoomHeightForFurnitureCalibration {
            calibrationRejectMessage = L10n.RoomViewer.furnitureHeightMustBeLessThanRoomHeight(
                currentRoomHeightForFurnitureCalibration
            )
            showCalibrationRejectAlert = true
            return
        }

        let scaleFactor = realHeight / detectedHeight
        realFurnitureHeight = realHeight
        logDebug("📐 [Mesh calibration] Real height: \(realHeight)m, overlay scale factor: \(scaleFactor)")
        inputFurnitureHeight = ""
        showFurnitureDimensionsInput = false
    }

    private var currentRoomHeightForFurnitureCalibration: Float {
        max(roomHeight, 0.01)
    }

    private func appendDigit(_ digit: String) {
        if inputFurnitureHeight.count >= 5 { return }
        if let dotIndex = inputFurnitureHeight.firstIndex(of: ".") {
            let decimals = inputFurnitureHeight.distance(from: dotIndex, to: inputFurnitureHeight.endIndex) - 1
            if decimals >= 2 { return }
        }
        inputFurnitureHeight += digit
    }

    private var navigationBarRoomMeasurementPrincipal: some View {
        HStack(spacing: 12) {
            Button {
                guard canPresentMeshRoomDimensionsAlert else { return }
                onMeshRoomDimensionsRulerTapped()
            } label: {
                Image(systemName: "ruler.fill")
                    .symbolRenderingMode(.hierarchical)
                    .font(.system(size: 18, weight: .medium))
                    .foregroundStyle(.primary)
            }
            .buttonStyle(.plain)
            .disabled(!canPresentMeshRoomDimensionsAlert || isLoading)
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

    private func toggleBrainArAssistedSizingOrShowHint() {
        guard showingFurnitureFit else { return }
        brainArAssistedSizingEnabled.toggle()
    }

    private func toggleFullVideoIdentifications() {
        showFullVideoWithIdentifications.toggle()
        if showFullVideoWithIdentifications {
            cancelFullVideoSelectionHelper()
            // Enter the tap-to-segment flow: show live identifications and wait for a tap.
            furnitureFitSegmentationMode = .identifyOnly
            furnitureFitShowIdentifyLivePreview = true
            presentFullVideoFurnitureTapHintIfNeeded()
        } else {
            dismissFullVideoFurnitureTapHint()
            // Back to the brain default: auto-segment the highest-confidence primary.
            furnitureFitSegmentationMode = .segmentPrimary
            furnitureFitShowIdentifyLivePreview = true
            presentFullVideoSelectionHelperIfNeeded()
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

    private var fullVideoModeFloatingButtonOverlay: some View {
        ZStack(alignment: .topTrailing) {
            Color.clear
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .allowsHitTesting(false)
            if showingFurnitureFit {
                fullVideoIdentificationsFloatingButton
                    .padding(.top, 54)
                    .padding(.trailing, canOfferBrainArAssist ? 58 : 16)
                    .transition(.opacity)
            }
        }
        .zIndex(107)
    }

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

    private var navigationBarTrailingControls: some View {
        HStack(spacing: 14) {
            Button(action: {
                NotificationCenter.default.post(name: NSNotification.Name("RecenterMeshCamera"), object: nil)
            }) {
                Image(systemName: "viewfinder")
            }
            .disabled(isLoading)

            if canOfferBrainArAssist, showingFurnitureFit {
                navigationBarARButton
                    .fixedSize(horizontal: true, vertical: true)
            }

            Button(action: {
                roomName = ""
                showRoomNameInput = true
            }) {
                Image(systemName: "square.and.arrow.down")
            }
            .disabled(isLoading || isSavingRoom)
            .accessibilityLabel(L10n.RoomViewer.saveRoom)
        }
    }

    private var fullVideoToolbarHelperOverlay: some View {
        ZStack(alignment: .topTrailing) {
            Color.clear
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .allowsHitTesting(false)
            if showingFurnitureFit &&
                !showFullVideoWithIdentifications &&
                furnitureFitSegmentationMode == .segmentPrimary &&
                fullVideoSelectionHelperVisible {
                PaafektHintChip(
                    systemImage: "text.viewfinder",
                    text: L10n.RoomViewer.fullVideoSelectionHelper
                )
                .padding(.top, 6)
                .padding(.trailing, canOfferBrainArAssist ? 62 : 20)
                .offset(y: 50)
                .transition(.opacity)
            }
        }
        .allowsHitTesting(false)
        .zIndex(106)
    }

    /// Optional AR sizing hint copy sits below the top toolbar row when visible.
    private var topTrailingPinchTapAndSizingHintsOverlay: some View {
        paafektTopToolbarHintOverlay(
            isVisible: canOfferBrainArAssist && arSizingHintExplanationVisible
                && (showingFurnitureFit || arSizingHintRequiresBrain)
        ) {
            PaafektHintChip(
                systemImage: "arrow.up.left.and.arrow.down.right",
                text: arSizingHintText            )
            .transition(.opacity)
        }
        .zIndex(101)
        .onDisappear {
            cancelPinchHintTasks()
            cancelARSizingHintTasks()
        }
    }

    private var navigationTeachingHintBottomOverlay: some View {
        paafektBottomToolbarHintOverlay(isVisible: pinchHintExplanationVisible) {
            PaafektHintChip(
                systemImage: "hand.draw.fill",
                text: L10n.RoomViewer.navigationTeachingHint
            )
            .transition(.opacity)
        }
        .zIndex(101)
    }

    private var roomDimensionsHintOverlay: some View {
        ZStack(alignment: .top) {
            Color.clear
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .allowsHitTesting(false)
            VStack(spacing: 0) {
                if roomDimensionsHintVisible {
                    PaafektHintChip(
                        systemImage: "ruler.fill",
                        text: meshRoomDimensionsHintText,
                        maxWidth: 240
                    )
                    .transition(.opacity)
                }
            }
            .padding(.top, 12)
            .onDisappear { cancelRoomDimensionsHintTasks() }
        }
        .zIndex(104)
    }

    private var fullVideoFurnitureTapHintOverlay: some View {
        ZStack(alignment: .top) {
            Color.clear
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .allowsHitTesting(false)
            VStack(spacing: 0) {
                if fullVideoFurnitureTapHintVisible {
                    PaafektHintChip(
                        systemImage: "hand.tap.fill",
                        text: L10n.RoomViewer.fullVideoFurnitureTapHint,
                        maxWidth: 280
                    )
                    .transition(.opacity)
                }
            }
            .padding(.top, roomDimensionsHintVisible ? 56 : 12)
        }
        .allowsHitTesting(false)
        .zIndex(99_999)
    }

    private func dismissFullVideoFurnitureTapHint() {
        fullVideoFurnitureTapHintVisible = false
    }

    private func cancelFullVideoSelectionHelper() {
        fullVideoSelectionHelperHideTask?.cancel()
        fullVideoSelectionHelperHideTask = nil
        fullVideoSelectionHelperVisible = false
    }

    private func presentFullVideoSelectionHelperIfNeeded() {
        guard showingFurnitureFit,
              !showFullVideoWithIdentifications,
              furnitureFitSegmentationMode == .segmentPrimary else {
            cancelFullVideoSelectionHelper()
            return
        }
        fullVideoSelectionHelperHideTask?.cancel()
        fullVideoSelectionHelperVisible = true
        fullVideoSelectionHelperHideTask = Task { @MainActor in
            try? await Task.sleep(for: .seconds(3))
            guard !Task.isCancelled else { return }
            fullVideoSelectionHelperVisible = false
        }
    }

    private func presentFullVideoFurnitureTapHintIfNeeded() {
        guard showFullVideoWithIdentifications else { return }
        fullVideoFurnitureTapHintVisible = true
    }

    private func cancelARSizingHintTasks() {
        arSizingHintHideTextTask?.cancel()
        arSizingHintHideTextTask = nil
    }

    private func scheduleARSizingHintTextAutoHide(seconds: UInt64 = 3) {
        arSizingHintHideTextTask?.cancel()
        arSizingHintHideTextTask = Task { @MainActor in
            try? await Task.sleep(for: .seconds(seconds))
            guard !Task.isCancelled else { return }
            arSizingHintExplanationVisible = false
        }
    }

    private func showARSizingHint(requiresBrain: Bool) {
        cancelARSizingHintTasks()
        arSizingHintRequiresBrain = requiresBrain
        arSizingHintExplanationVisible = true
        scheduleARSizingHintTextAutoHide(seconds: 3)
    }

    private func cancelRoomDimensionsHintTasks() {
        roomDimensionsHintHideTask?.cancel()
        roomDimensionsHintHideTask = nil
    }

    private func scheduleRoomDimensionsHintAutoHide(seconds: UInt64 = 3) {
        roomDimensionsHintHideTask?.cancel()
        roomDimensionsHintHideTask = Task { @MainActor in
            try? await Task.sleep(for: .seconds(seconds))
            guard !Task.isCancelled else { return }
            roomDimensionsHintVisible = false
        }
    }

    private func onMeshRoomDimensionsRulerTapped() {
        cancelRoomDimensionsHintTasks()
        roomDimensionsHintVisible.toggle()
        if roomDimensionsHintVisible {
            scheduleRoomDimensionsHintAutoHide(seconds: 3)
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

    private var pinchHintAccessibilityLabel: String {
        L10n.RoomViewer.pinchGestureHintExplanation + " " + L10n.RoomViewer.gestureHintToggleAccessibility
    }

    private var arSizingHintText: String {
        arSizingHintRequiresBrain
            ? L10n.RoomViewer.arFurnitureSizingRequiresBrainHint
            : L10n.RoomViewer.arFurnitureSizingHint
    }

    private var meshRestingMeasurementPillText: String? {
        let width = calibratedRoomWidth
        let depth = calibratedRoomDepth
        if width > 0.05, depth > 0.05, width.isFinite, depth.isFinite {
            return String(format: "%.1f m × %.1f m", width, depth)
        }
        if calibratedRoomHeight > 0.05, calibratedRoomHeight.isFinite {
            return L10n.RoomViewer.approximateRoomHeight(calibratedRoomHeight)
        }
        return nil
    }

    private var meshMorphingPrimaryAction: PaafektMorphingPrimaryAction {
        PaafektMorphingPrimaryActionResolver.resolve(
            showingFurnitureFit: showingFurnitureFit,
            showFullVideoWithIdentifications: showFullVideoWithIdentifications,
            segmentationMode: furnitureFitSegmentationMode,
            hasSelectedObject: !selectedFurnitureFitLabels.isEmpty
        )
    }

    private var meshMorphingPrimaryDisabled: Bool {
        isLoading || isSavingRoom ||
            (meshMorphingPrimaryAction == .segment && !canSegmentSelectedFurniture)
    }

    private func handleMeshMorphingPrimaryTap() {
        immersiveChrome.noteChromeInteraction()
        let action = meshMorphingPrimaryAction
        guard action != .segment || canSegmentSelectedFurniture else { return }
        PaafektMorphingPrimaryActionHandler.perform(
            action,
            enterFit: toggleMeshFurnitureFit,
            exitFit: toggleMeshFurnitureFit,
            segment: activateSelectedFurnitureSegmentation,
            finishSegmentation: activateSelectedFurnitureSegmentation
        )
    }

    private var meshImmersiveChromeOverlay: some View {
        PaafektImmersiveViewerChromeStack(
            chrome: immersiveChrome,
            onBack: {
                if savedRoomModel == nil {
                    if saveWasSuccessful {
                        dismiss()
                    } else {
                        showDiscardUnsavedAlert = true
                    }
                } else {
                    dismiss()
                }
            },
            morphingPrimaryAction: meshMorphingPrimaryAction,
            onMorphingPrimary: handleMeshMorphingPrimaryTap,
            morphingPrimaryDisabled: meshMorphingPrimaryDisabled,
            onSave: savedRoomModel == nil ? {
                immersiveChrome.noteChromeInteraction()
                roomName = ""
                showRoomNameInput = true
            } : nil,
            saveDisabled: isLoading || isSavingRoom,
            measurementText: meshRestingMeasurementPillText,
            hideForCapture: false
        ) {
            PaafektImmersiveSummonedToolbar(chrome: immersiveChrome) {
                HStack(spacing: Theme.Space.sm) {
                    PaafektViewerToolbarIconButton(
                        systemName: "viewfinder",
                        accessibilityLabel: L10n.RoomViewer.recenterView
                    ) {
                        immersiveChrome.noteChromeInteraction()
                        NotificationCenter.default.post(name: NSNotification.Name("RecenterMeshCamera"), object: nil)
                    }
                    PaafektViewerToolbarIconButton(
                        systemName: "ruler",
                        accessibilityLabel: L10n.RoomViewer.checkMeasurement
                    ) {
                        immersiveChrome.noteChromeInteraction()
                        onMeshRoomDimensionsRulerTapped()
                    }
                    PaafektViewerToolbarIconButton(
                        systemName: "hand.pinch",
                        accessibilityLabel: pinchHintAccessibilityLabel
                    ) {
                        immersiveChrome.noteChromeInteraction()
                        onPinchHintIconTapped()
                    }
                    PaafektViewerToolbarIconButton(
                        systemName: "square.stack.3d.up",
                        accessibilityLabel: L10n.RoomViewer.displayAllHelpers
                    ) {
                        immersiveChrome.noteChromeInteraction()
                        displayAllGestureHelpers()
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
                                toggleBrainArAssistedSizingOrShowHint()
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

    private func toggleMeshFurnitureFit() {
        if showingFurnitureFit {
            dismissFullVideoFurnitureTapHint()
            cancelFullVideoSelectionHelper()
            showingFurnitureFit = false
        } else {
            showFullVideoWithIdentifications = false
            furnitureFitInitialSegmentationDone = false
            furnitureFitSegmentationMode = .segmentPrimary
            furnitureFitShowIdentifyLivePreview = true
            selectedFurnitureFitLabels = []
            showingFurnitureFit = true
            presentFullVideoSelectionHelperIfNeeded()
        }
    }

    private func displayAllGestureHelpers() {
        replayTeachingHints = true
        restartPinchGestureHint()
        showARSizingHint(requiresBrain: !showingFurnitureFit)
        roomDimensionsHintVisible = true
        scheduleRoomDimensionsHintAutoHide(seconds: 3)
    }

    /// After pinned segment targets leave the scene, labels become empty; reopen full-video tap-to-pick flow.
    private func restoreFullVideoIdentifyAfterSegmentPinsLost(oldLabels: [String], newLabels: [String]) {
        guard showingFurnitureFit else { return }
        guard furnitureFitSegmentationMode == .segmentSelected else { return }
        guard newLabels.isEmpty, !oldLabels.isEmpty else { return }
        dismissFullVideoFurnitureTapHint()
        showFullVideoWithIdentifications = true
        furnitureFitSegmentationMode = .identifyOnly
        furnitureFitShowIdentifyLivePreview = true
        presentFullVideoFurnitureTapHintIfNeeded()
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
                Text(fit.fitsInRoom ? L10n.RoomViewer.placementFitsRoom : L10n.RoomViewer.placementExceedsRoom)
                    .font(.caption2)
                    .foregroundColor(fit.fitsInRoom ? .green : .red)
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

    @ViewBuilder
    private var roomIntelligencePlacementCard: some View {
        if showingFurnitureFit, authoritativeRoomModelForMetrics != nil {
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
                if let aesthetic = latestAestheticScore {
                    placementIntelligenceExpandedContent(dimensions: dimensions, fit: fit, aesthetic: aesthetic)
                }
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

    // MARK: - Request GLB Export from JavaScript
    private func requestGLBExport() {
        isSavingRoom = true
        webView?.evaluateJavaScript("exportGLB()") { _, error in
            if let error = error {
                logDebug("❌ [MeshRoomView] Error requesting GLB export: \(error)")
                DispatchQueue.main.async {
                    isSavingRoom = false
                    saveAlertMessage = L10n.RoomViewer.meshExportFailed
                    showSaveErrorNotice = true
                }
            }
            // GLB data will come back via the message handler
        }
    }

    // MARK: - Save GLB Room
    private func saveGLBRoom(glbData: Data) {
        modelManager.saveGLBRoom(
            glbData: glbData,
            name: roomName,
            photoOrientation: photoOrientation,
            roomWidth: roomWidth,
            roomHeight: roomHeight,
            roomDepth: roomDepth
        ) { success, errorMessage in
            isSavingRoom = false
            saveWasSuccessful = success

            if success {
                saveSuccessSnackbarMessage = L10n.RoomViewer.saveSuccess(roomName)
                withAnimation { showSaveSuccessSnackbar = true }
                DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) {
                    NotificationCenter.default.post(name: NSNotification.Name("DismissPhotoRoomSheet"), object: nil)
                }
            } else {
                saveAlertMessage = errorMessage ?? L10n.RoomViewer.meshSaveFailedGeneric
                showSaveErrorNotice = true
            }
            roomName = ""
        }
    }

    // MARK: - Take Screenshot
    private func takeScreenshot() {
        logDebug("📸 [MeshRoomView] Taking screenshot...")

        // Capture the window
        let scenes = UIApplication.shared.connectedScenes.compactMap { $0 as? UIWindowScene }
        let windows = scenes.flatMap { $0.windows }
        guard let window = windows.first(where: { $0.isKeyWindow }) ?? windows.first else {
            logDebug("❌ [MeshRoomView] No window found")
            return
        }

        let format = UIGraphicsImageRendererFormat()
        format.scale = window.traitCollection.displayScale
        let renderer = UIGraphicsImageRenderer(bounds: window.bounds, format: format)
        let image = renderer.image { _ in
            window.drawHierarchy(in: window.bounds, afterScreenUpdates: true)
        }

        logDebug("📸 [MeshRoomView] Screenshot captured, saving to Photos...")

        // Save to photos
        UIImageWriteToSavedPhotosAlbum(image, nil, nil, nil)
        logDebug("✅ [MeshRoomView] Screenshot saved to Photos")
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

    private func updateRoomPlacementIntelligence() {
        guard showingFurnitureFit else {
            latestFitCheckResult = nil
            latestAestheticScore = nil
            return
        }
        guard let roomModel = authoritativeRoomModelForMetrics else {
            latestFitCheckResult = nil
            latestAestheticScore = nil
            return
        }
        guard placementIntelligenceHasFurnitureSignal else {
            latestFitCheckResult = nil
            latestAestheticScore = nil
            return
        }

        if let furniture = derivedDetectedFurnitureDimensionsForRoomIntelligence() {
            let fitEngine = FitCheckEngine(roomModel: roomModel)
            latestFitCheckResult = fitEngine.checkFit(furniture: furniture)
        } else {
            latestFitCheckResult = nil
        }

        let palette = roomModel.surfacePalette
        let roomStyleTags = inferredRoomStyleTags(from: palette)
        let furnitureProfile = heuristicFurnitureProfileForAesthetic(
            roomModel: roomModel,
            segmentedMeanSRGB: segmentedFurnitureMeanSRGB
        )
        let aestheticAdvisor = AestheticAdvisor(palette: palette, roomStyleTags: roomStyleTags)
        latestAestheticScore = aestheticAdvisor.evaluate(furniture: furnitureProfile)
    }

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

    // MARK: - RTMDet model loaded via RTMDetModelService
}

private enum MeshPlacementIntelligenceRoomStub {
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

// MARK: - WebGL Mesh View (UIViewRepresentable)
struct MeshWebGLView: UIViewRepresentable {
    let roomWidth: Float
    let roomHeight: Float
    let roomDepth: Float
    let frontWallImage: UIImage
    let photoOrientation: PhotoOrientation

    // Boundary coordinates for wall texturing (normalized 0-1)
    let leftX: CGFloat
    let rightX: CGFloat
    let ceilingY: CGFloat
    let floorY: CGFloat

    @Binding var webViewRef: WKWebView?
    let onLoaded: () -> Void
    let onGLBExported: (Data) -> Void

    func makeUIView(context: Context) -> WKWebView {
        let config = WKWebViewConfiguration()
        config.allowsInlineMediaPlayback = true
        config.setURLSchemeHandler(
            BundledWebViewAssetSchemeHandler(),
            forURLScheme: BundledWebViewAsset.scheme
        )

        // Add message handler for JS -> Swift communication
        config.userContentController.add(context.coordinator, name: "meshViewer")

        let webView = WKWebView(frame: .zero, configuration: config)
        context.coordinator.webView = webView
        webView.navigationDelegate = context.coordinator

        // Completely disable WebView scrolling - let Three.js handle all touch
        webView.scrollView.isScrollEnabled = false
        webView.scrollView.bounces = false
        webView.scrollView.bouncesZoom = false
        webView.scrollView.minimumZoomScale = 1.0
        webView.scrollView.maximumZoomScale = 1.0
        webView.scrollView.contentInsetAdjustmentBehavior = .never

        // Disable all scroll view gesture recognizers
        for gestureRecognizer in webView.scrollView.gestureRecognizers ?? [] {
            gestureRecognizer.isEnabled = false
        }

        webView.isOpaque = false
        webView.backgroundColor = .gray
        webView.isUserInteractionEnabled = true
        webView.isMultipleTouchEnabled = true

        // Add edge pan gesture to block system back swipe
        let edgePan = UIScreenEdgePanGestureRecognizer(target: context.coordinator, action: #selector(Coordinator.handleEdgePan(_:)))
        edgePan.edges = .left
        edgePan.cancelsTouchesInView = false
        edgePan.delaysTouchesBegan = false
        edgePan.delegate = context.coordinator
        webView.addGestureRecognizer(edgePan)

        // Find and disable the navigation controller's back gesture
        DispatchQueue.main.async {
            if let navController = webView.findNavigationController() {
                navController.interactivePopGestureRecognizer?.isEnabled = false
            }
        }

        // Store reference
        DispatchQueue.main.async {
            webViewRef = webView
        }

        let html = generateMeshViewerHTML()
        let baseURL = URL(string: BundledWebViewAsset.assetURLString(for: ""))!
        if BundledWebViewAsset.bundledBaseURL() != nil {
            logDebug("📄 [MeshViewer] loadHTMLString (bundled scheme URLs) htmlBytes=\(html.utf8.count)")
            webView.loadHTMLString(html, baseURL: baseURL)
        } else {
            logDebug("❌ [MeshViewer] Missing bundled WebView vendor assets")
        }

        return webView
    }

    func updateUIView(_ uiView: WKWebView, context: Context) {}

    func makeCoordinator() -> Coordinator {
        Coordinator(onLoaded: onLoaded, onGLBExported: onGLBExported)
    }

    class Coordinator: NSObject, WKScriptMessageHandler, UIGestureRecognizerDelegate, WKNavigationDelegate {
        let onLoaded: () -> Void
        let onGLBExported: (Data) -> Void
        weak var webView: WKWebView?

        init(onLoaded: @escaping () -> Void, onGLBExported: @escaping (Data) -> Void) {
            self.onLoaded = onLoaded
            self.onGLBExported = onGLBExported
            super.init()

            // Listen for recenter notification
            NotificationCenter.default.addObserver(
                self,
                selector: #selector(recenterCamera),
                name: NSNotification.Name("RecenterMeshCamera"),
                object: nil
            )
            NotificationCenter.default.addObserver(
                self,
                selector: #selector(nudgeMeshCameraLeft),
                name: NSNotification.Name("WebGLCameraMoveLeft"),
                object: nil
            )
            NotificationCenter.default.addObserver(
                self,
                selector: #selector(nudgeMeshCameraRight),
                name: NSNotification.Name("WebGLCameraMoveRight"),
                object: nil
            )
            NotificationCenter.default.addObserver(
                self,
                selector: #selector(nudgeMeshCameraUp),
                name: NSNotification.Name("WebGLCameraMoveUp"),
                object: nil
            )
            NotificationCenter.default.addObserver(
                self,
                selector: #selector(nudgeMeshCameraDown),
                name: NSNotification.Name("WebGLCameraMoveDown"),
                object: nil
            )
        }

        deinit {
            NotificationCenter.default.removeObserver(self)
        }

        func webView(_ webView: WKWebView, didStartProvisionalNavigation navigation: WKNavigation!) {
            logDebug("🌐 [MeshViewer] WK didStartProvisionalNavigation url=\(webView.url?.absoluteString ?? "nil")")
        }

        func webView(_ webView: WKWebView, didCommit navigation: WKNavigation!) {
            logDebug("🌐 [MeshViewer] WK didCommit url=\(webView.url?.absoluteString ?? "nil")")
        }

        func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
            logDebug("🌐 [MeshViewer] WK didFinish url=\(webView.url?.absoluteString ?? "nil")")
        }

        func webView(_ webView: WKWebView, didFail navigation: WKNavigation!, withError error: Error) {
            logDebug("❌ [MeshViewer] WK didFail error=\(error.localizedDescription)")
        }

        func webView(_ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!, withError error: Error) {
            logDebug("❌ [MeshViewer] WK didFailProvisionalNavigation error=\(error.localizedDescription)")
        }

        @objc private func recenterCamera() {
            webView?.evaluateJavaScript("if (typeof recenterCamera === 'function') recenterCamera();", completionHandler: nil)
        }

        @objc private func nudgeMeshCameraLeft() {
            webView?.evaluateJavaScript("if (typeof moveCamera === 'function') moveCamera(-8, 0);", completionHandler: nil)
        }

        @objc private func nudgeMeshCameraRight() {
            webView?.evaluateJavaScript("if (typeof moveCamera === 'function') moveCamera(8, 0);", completionHandler: nil)
        }

        @objc private func nudgeMeshCameraUp() {
            webView?.evaluateJavaScript("if (typeof moveCameraUp === 'function') moveCameraUp(0.2);", completionHandler: nil)
        }

        @objc private func nudgeMeshCameraDown() {
            webView?.evaluateJavaScript("if (typeof moveCameraUp === 'function') moveCameraUp(-0.2);", completionHandler: nil)
        }

        @objc func handleEdgePan(_ gesture: UIScreenEdgePanGestureRecognizer) {
            // Do nothing - this gesture just blocks the system back swipe
        }

        // Always recognize our edge gesture, blocking others
        func gestureRecognizer(_ gestureRecognizer: UIGestureRecognizer, shouldBeRequiredToFailBy otherGestureRecognizer: UIGestureRecognizer) -> Bool {
            return true
        }

        func userContentController(_ userContentController: WKUserContentController, didReceive message: WKScriptMessage) {
            guard let body = message.body as? [String: Any],
                  let event = body["event"] as? String else { return }

            switch event {
            case "loaded":
                logDebug("✅ [MeshViewer] JS reported loaded")
                DispatchQueue.main.async {
                    self.onLoaded()
                }
            case "jsLog":
                let parts = body
                    .filter { $0.key != "event" }
                    .map { "\($0.key)=\($0.value)" }
                    .sorted()
                    .joined(separator: " ")
                logDebug("🧩 [MeshViewer] JS \(parts)")
            case "jsError":
                let message = body["message"] as? String ?? "unknown"
                let parts = body
                    .filter { $0.key != "event" && $0.key != "message" }
                    .map { "\($0.key)=\($0.value)" }
                    .sorted()
                    .joined(separator: " ")
                logDebug("❌ [MeshViewer] JS error message=\(message) \(parts)")
            case "glbExported":
                // Receive base64-encoded GLB data
                if let base64Data = body["data"] as? String,
                   let glbData = Data(base64Encoded: base64Data) {
                    logDebug("✅ [MeshViewer] Received GLB data: \(glbData.count) bytes")
                    DispatchQueue.main.async {
                        self.onGLBExported(glbData)
                    }
                } else {
                    logDebug("❌ [MeshViewer] Failed to decode GLB data")
                }
            default:
                break
            }
        }
    }

    // MARK: - Generate Three.js HTML with GLTF Export
    private func generateMeshViewerHTML() -> String {
        // Convert image to base64
        let imageData = frontWallImage.jpegData(compressionQuality: 0.85) ?? Data()
        let base64Image = imageData.base64EncodedString()

        let isPortrait = photoOrientation == .portrait
        let threeModuleURL = BundledWebViewAsset.assetURLString(for: "three/build/three.module.js")

        return """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
            <style>
                * { margin: 0; padding: 0; }
                html, body {
                    width: 100%;
                    height: 100%;
                    overflow: hidden;
                    background: #808080;
                    touch-action: none;
                }
                canvas {
                    display: block;
                    width: 100%;
                    height: 100%;
                    touch-action: none;
                }
            </style>
        </head>
        <body>
            <script type="importmap">
            {
                "imports": {
                    "three": "\(threeModuleURL)",
                    "three/addons/": "\(BundledWebViewAsset.assetURLString(for: "three/examples/jsm/"))"
                }
            }
            </script>
            <script type="module">
                import * as THREE from 'three';
                import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

                window.addEventListener('error', function (event) {
                    meshPost('jsError', {
                        message: 'window.error: ' + (event && event.message ? event.message : 'unknown'),
                        filename: event && event.filename ? event.filename : '',
                        lineno: event && event.lineno ? event.lineno : 0,
                        colno: event && event.colno ? event.colno : 0
                    });
                });
                window.addEventListener('unhandledrejection', function (event) {
                    let reason = event && event.reason;
                    let message = (reason && reason.message) ? reason.message : String(reason);
                    meshPost('jsError', { message: 'unhandledrejection: ' + message });
                });

                function meshPost(event, extra) {
                    try {
                        if (window.webkit && window.webkit.messageHandlers && window.webkit.messageHandlers.meshViewer) {
                            var o = { event: event };
                            if (extra) { for (var k in extra) { o[k] = extra[k]; } }
                            window.webkit.messageHandlers.meshViewer.postMessage(o);
                        }
                    } catch (e) {}
                }
                function safeViewportSize() {
                    var w = Math.max(1, window.innerWidth || (document.documentElement && document.documentElement.clientWidth) || 1);
                    var h = Math.max(1, window.innerHeight || (document.documentElement && document.documentElement.clientHeight) || 1);
                    return { w: w, h: h };
                }
                var vp = safeViewportSize();
                meshPost('jsLog', { step: 'module_after_import', w: vp.w, h: vp.h, dpr: window.devicePixelRatio || 1 });

                // Room dimensions from Swift (in meters)
                const roomWidth = \(roomWidth);
                const roomHeight = \(roomHeight);
                const roomDepth = \(roomDepth);
                const isPortrait = \(isPortrait ? "true" : "false");

                // Boundary coordinates from Swift (normalized 0-1)
                // These define where the front wall edges are in the source image
                const leftX = \(leftX);      // Left wall boundary (0.12 = 12% from left)
                const rightX = \(rightX);    // Right wall boundary (0.88 = 88% from left)
                const ceilingY = \(ceilingY); // Ceiling boundary (0.15 = 15% from top)
                const floorY = \(floorY);    // Floor boundary (0.85 = 85% from top)

                console.log('[MeshViewer] Room dimensions:', roomWidth, 'x', roomHeight, 'x', roomDepth, 'm');
                console.log('[MeshViewer] Boundaries: L=' + leftX + ', R=' + rightX + ', T=' + ceilingY + ', B=' + floorY);

                // Scene setup
                const scene = new THREE.Scene();
                scene.background = new THREE.Color(0x404040);

                // Create a group to hold the room for export
                const roomGroup = new THREE.Group();
                roomGroup.name = 'Room';
                scene.add(roomGroup);

                // Camera — avoid 0×0 innerWidth/innerHeight on first WKWebView layout (NaN aspect → blank WebGL).
                const camera = new THREE.PerspectiveCamera(70, vp.w / vp.h, 0.1, 100);

                // Renderer
                const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
                renderer.setSize(vp.w, vp.h);
                renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
                renderer.outputColorSpace = THREE.SRGBColorSpace;
                document.body.appendChild(renderer.domElement);

                renderer.domElement.addEventListener('webglcontextlost', function (ev) {
                    meshPost('jsError', { message: 'webglcontextlost' });
                    ev.preventDefault();
                }, false);

                function applyViewportSize() {
                    vp = safeViewportSize();
                    camera.aspect = vp.w / vp.h;
                    camera.updateProjectionMatrix();
                    renderer.setSize(vp.w, vp.h);
                    meshPost('jsLog', { step: 'viewport_apply', w: vp.w, h: vp.h });
                }
                applyViewportSize();
                window.addEventListener('resize', applyViewportSize);
                requestAnimationFrame(function () {
                    applyViewportSize();
                    requestAnimationFrame(applyViewportSize);
                });
                try {
                    if (typeof ResizeObserver !== 'undefined') {
                        var ro = new ResizeObserver(function () { applyViewportSize(); });
                        ro.observe(document.documentElement);
                    }
                } catch (e0) {}

                // Orbit controls - single-finger rotate matches Splat room feel (no inertia, ~0.005 rad/px).
                const controls = new OrbitControls(camera, renderer.domElement);
                controls.enableDamping = false;
                controls.rotateSpeed = 0.7;
                controls.zoomSpeed = 2.5;       // Fast zoom
                controls.panSpeed = 1.5;        // Fast pan
                controls.enableZoom = true;
                controls.enablePan = true;
                controls.minDistance = 0.3;
                controls.maxDistance = Math.max(roomWidth, roomDepth) * 1.5;
                controls.maxPolarAngle = Math.PI * 0.95;
                controls.minPolarAngle = Math.PI * 0.05;

                // Touch-specific settings
                controls.touches = {
                    ONE: THREE.TOUCH.ROTATE,
                    TWO: THREE.TOUCH.DOLLY_PAN
                };

                // Set initial camera position - back center of room looking at front wall.
                // Target placed on the front wall so single-finger rotate orbits around it (matches Splat room).
                const targetY = roomHeight * 0.5;
                controls.target.set(0, targetY, -roomDepth * 0.5);
                camera.position.set(0, targetY, roomDepth * 0.35);
                controls.update();

                // Save initial camera state for recenter
                let initialCameraPos = camera.position.clone();
                let initialTarget = controls.target.clone();

                // Recenter function
                window.recenterCamera = function() {
                    camera.position.copy(initialCameraPos);
                    controls.target.copy(initialTarget);
                    controls.update();
                    console.log('[MeshViewer] Camera recentered');
                };

                window.scaleRoom = function(factor) {
                    if (typeof factor !== 'number' || !isFinite(factor) || factor <= 0) return;
                    roomGroup.scale.set(factor, factor, factor);
                    roomBoundsForClamping.minX = -roomWidth * 0.5 * factor;
                    roomBoundsForClamping.maxX = roomWidth * 0.5 * factor;
                    roomBoundsForClamping.maxY = roomHeight * factor;
                    roomBoundsForClamping.minZ = -roomDepth * 0.5 * factor;
                    roomBoundsForClamping.maxZ = roomDepth * 0.5 * factor;
                    controls.maxDistance = Math.max(roomWidth, roomDepth) * 1.5 * factor;
                    const targetYScaled = roomHeight * 0.5 * factor;
                    const cameraZScaled = roomDepth * 0.35 * factor;
                    initialTarget.set(0, targetYScaled, -roomDepth * 0.5 * factor);
                    initialCameraPos.set(0, targetYScaled, cameraZScaled);
                    camera.position.copy(initialCameraPos);
                    controls.target.copy(initialTarget);
                    controls.update();
                    console.log('[MeshViewer] Room scaled by factor:', factor);
                };

                // D-pad: same behavior as Splat WebGL — walk on XZ, vertical on Y (not orbit).
                const roomBoundsForClamping = {
                    minX: -roomWidth * 0.5,
                    maxX: roomWidth * 0.5,
                    minY: 0,
                    maxY: roomHeight,
                    minZ: -roomDepth * 0.5,
                    maxZ: roomDepth * 0.5
                };

                window.moveCamera = function(dx, dy) {
                    const moveSpeed = 0.03;
                    let newX = camera.position.x + dx * moveSpeed;
                    let newZ = camera.position.z + dy * moveSpeed;
                    const marginSide = 0.05;
                    const marginBack = 0.02;
                    newX = Math.max(roomBoundsForClamping.minX + marginSide,
                           Math.min(roomBoundsForClamping.maxX - marginSide, newX));
                    newZ = Math.max(roomBoundsForClamping.minZ + marginSide,
                           Math.min(roomBoundsForClamping.maxZ - marginBack, newZ));
                    const actualDx = newX - camera.position.x;
                    const actualDz = newZ - camera.position.z;
                    camera.position.x = newX;
                    camera.position.z = newZ;
                    controls.target.x += actualDx;
                    controls.target.z += actualDz;
                    controls.update();
                };

                window.moveCameraUp = function(dy) {
                    if (typeof dy !== 'number' || !isFinite(dy)) return;
                    camera.position.y += dy;
                    controls.target.y += dy;
                    const m = 0.05;
                    camera.position.y = Math.max(roomBoundsForClamping.minY + m, Math.min(roomBoundsForClamping.maxY - m, camera.position.y));
                    controls.target.y = Math.max(roomBoundsForClamping.minY + m, Math.min(roomBoundsForClamping.maxY - m, controls.target.y));
                    controls.update();
                };

                // Lighting - brighter for textured walls
                const ambientLight = new THREE.AmbientLight(0xffffff, 1.0);
                roomGroup.add(ambientLight);

                const frontLight = new THREE.DirectionalLight(0xffffff, 0.3);
                frontLight.position.set(0, roomHeight, roomDepth * 0.3);
                roomGroup.add(frontLight);

                // Load source image
                const img = new Image();
                img.onload = function() {
                    console.log('[MeshViewer] Image loaded:', img.width, 'x', img.height);
                    meshPost('jsLog', { step: 'photo_image_onload', iw: img.width, ih: img.height });
                    buildRoomWithTextures(img);
                };
                img.onerror = function() {
                    console.error('[MeshViewer] Failed to load image');
                    meshPost('jsError', { message: 'data:image/jpeg base64 failed to decode in Image()' });
                    buildRoomGray(); // Build room with gray walls as fallback
                };
                img.src = 'data:image/jpeg;base64,\(base64Image)';

                // Create a texture from a portion of the source image
                function createTextureFromRegion(sourceImg, x, y, w, h) {
                    const canvas = document.createElement('canvas');
                    canvas.width = Math.max(1, Math.round(w));
                    canvas.height = Math.max(1, Math.round(h));
                    const ctx = canvas.getContext('2d');
                    ctx.drawImage(sourceImg, x, y, w, h, 0, 0, canvas.width, canvas.height);

                    const texture = new THREE.CanvasTexture(canvas);
                    texture.colorSpace = THREE.SRGBColorSpace;
                    texture.needsUpdate = true;
                    return texture;
                }

                function buildRoomWithTextures(sourceImg) {
                    try {
                    const imgW = sourceImg.width;
                    const imgH = sourceImg.height;

                    // Calculate pixel coordinates for each region
                    const pxLeftX = leftX * imgW;
                    const pxRightX = rightX * imgW;
                    const pxCeilingY = ceilingY * imgH;
                    const pxFloorY = floorY * imgH;

                    console.log('[MeshViewer] Pixel boundaries:', pxLeftX, pxRightX, pxCeilingY, pxFloorY);

                    // Front wall: center region (leftX to rightX, ceilingY to floorY)
                    const frontTexture = createTextureFromRegion(
                        sourceImg,
                        pxLeftX, pxCeilingY,
                        pxRightX - pxLeftX, pxFloorY - pxCeilingY
                    );

                    // Left wall: left region (0 to leftX, ceilingY to floorY)
                    const leftTexture = createTextureFromRegion(
                        sourceImg,
                        0, pxCeilingY,
                        pxLeftX, pxFloorY - pxCeilingY
                    );

                    // Right wall: right region (rightX to 1, ceilingY to floorY)
                    const rightTexture = createTextureFromRegion(
                        sourceImg,
                        pxRightX, pxCeilingY,
                        imgW - pxRightX, pxFloorY - pxCeilingY
                    );

                    // Ceiling: top region (leftX to rightX, 0 to ceilingY)
                    const ceilingTexture = createTextureFromRegion(
                        sourceImg,
                        pxLeftX, 0,
                        pxRightX - pxLeftX, pxCeilingY
                    );

                    // Floor: bottom region (leftX to rightX, floorY to 1)
                    const floorTexture = createTextureFromRegion(
                        sourceImg,
                        pxLeftX, pxFloorY,
                        pxRightX - pxLeftX, imgH - pxFloorY
                    );

                    // Create materials with textures
                    const frontMaterial = new THREE.MeshBasicMaterial({
                        map: frontTexture,
                        side: THREE.DoubleSide
                    });

                    const leftMaterial = new THREE.MeshBasicMaterial({
                        map: leftTexture,
                        side: THREE.DoubleSide
                    });

                    const rightMaterial = new THREE.MeshBasicMaterial({
                        map: rightTexture,
                        side: THREE.DoubleSide
                    });

                    const ceilingMaterial = new THREE.MeshBasicMaterial({
                        map: ceilingTexture,
                        side: THREE.DoubleSide
                    });

                    const floorMaterial = new THREE.MeshBasicMaterial({
                        map: floorTexture,
                        side: THREE.DoubleSide
                    });

                    // Front wall - at z = -roomDepth/2
                    const frontGeometry = new THREE.PlaneGeometry(roomWidth, roomHeight);
                    const frontWall = new THREE.Mesh(frontGeometry, frontMaterial);
                    frontWall.position.set(0, roomHeight / 2, -roomDepth / 2);
                    frontWall.name = 'FrontWall';
                    roomGroup.add(frontWall);

                    // Floor - at y = 0
                    const floorGeometry = new THREE.PlaneGeometry(roomWidth, roomDepth);
                    const floor = new THREE.Mesh(floorGeometry, floorMaterial);
                    floor.rotation.x = -Math.PI / 2;
                    floor.position.set(0, 0, 0);
                    floor.name = 'Floor';
                    roomGroup.add(floor);

                    // Ceiling - at y = roomHeight
                    const ceilingGeometry = new THREE.PlaneGeometry(roomWidth, roomDepth);
                    const ceiling = new THREE.Mesh(ceilingGeometry, ceilingMaterial);
                    ceiling.rotation.x = Math.PI / 2;
                    ceiling.position.set(0, roomHeight, 0);
                    ceiling.name = 'Ceiling';
                    roomGroup.add(ceiling);

                    // Left wall - at x = -roomWidth/2
                    const leftGeometry = new THREE.PlaneGeometry(roomDepth, roomHeight);
                    const leftWall = new THREE.Mesh(leftGeometry, leftMaterial);
                    leftWall.rotation.y = Math.PI / 2;
                    leftWall.position.set(-roomWidth / 2, roomHeight / 2, 0);
                    leftWall.name = 'LeftWall';
                    roomGroup.add(leftWall);

                    // Right wall - at x = roomWidth/2
                    const rightGeometry = new THREE.PlaneGeometry(roomDepth, roomHeight);
                    const rightWall = new THREE.Mesh(rightGeometry, rightMaterial);
                    rightWall.rotation.y = -Math.PI / 2;
                    rightWall.position.set(roomWidth / 2, roomHeight / 2, 0);
                    rightWall.name = 'RightWall';
                    roomGroup.add(rightWall);

                    // No back wall - open for camera to enter

                    console.log('[MeshViewer] Room built with textures');
                    meshPost('jsLog', { step: 'room_built_textures' });
                    window.webkit.messageHandlers.meshViewer.postMessage({ event: 'loaded' });
                    } catch (e) {
                        meshPost('jsError', { message: 'buildRoomWithTextures: ' + ((e && e.message) ? e.message : String(e)) });
                        buildRoomGray();
                    }
                }

                // Fallback: build room with gray walls
                function buildRoomGray() {
                    const wallMaterial = new THREE.MeshStandardMaterial({
                        color: 0x666666,
                        side: THREE.DoubleSide,
                        roughness: 0.8
                    });

                    const floorMaterial = new THREE.MeshStandardMaterial({
                        color: 0x444444,
                        side: THREE.DoubleSide,
                        roughness: 0.9
                    });

                    const ceilingMaterial = new THREE.MeshStandardMaterial({
                        color: 0x999999,
                        side: THREE.DoubleSide,
                        roughness: 0.7
                    });

                    // Front wall
                    const frontGeometry = new THREE.PlaneGeometry(roomWidth, roomHeight);
                    const frontWall = new THREE.Mesh(frontGeometry, wallMaterial);
                    frontWall.position.set(0, roomHeight / 2, -roomDepth / 2);
                    frontWall.name = 'FrontWall';
                    roomGroup.add(frontWall);

                    // Floor
                    const floorGeometry = new THREE.PlaneGeometry(roomWidth, roomDepth);
                    const floor = new THREE.Mesh(floorGeometry, floorMaterial);
                    floor.rotation.x = -Math.PI / 2;
                    floor.position.set(0, 0, 0);
                    floor.name = 'Floor';
                    roomGroup.add(floor);

                    // Ceiling
                    const ceilingGeometry = new THREE.PlaneGeometry(roomWidth, roomDepth);
                    const ceiling = new THREE.Mesh(ceilingGeometry, ceilingMaterial);
                    ceiling.rotation.x = Math.PI / 2;
                    ceiling.position.set(0, roomHeight, 0);
                    ceiling.name = 'Ceiling';
                    roomGroup.add(ceiling);

                    // Left wall
                    const leftGeometry = new THREE.PlaneGeometry(roomDepth, roomHeight);
                    const leftWall = new THREE.Mesh(leftGeometry, wallMaterial);
                    leftWall.rotation.y = Math.PI / 2;
                    leftWall.position.set(-roomWidth / 2, roomHeight / 2, 0);
                    leftWall.name = 'LeftWall';
                    roomGroup.add(leftWall);

                    // Right wall
                    const rightGeometry = new THREE.PlaneGeometry(roomDepth, roomHeight);
                    const rightWall = new THREE.Mesh(rightGeometry, wallMaterial);
                    rightWall.rotation.y = -Math.PI / 2;
                    rightWall.position.set(roomWidth / 2, roomHeight / 2, 0);
                    rightWall.name = 'RightWall';
                    roomGroup.add(rightWall);

                    console.log('[MeshViewer] Room built (gray fallback)');
                    window.webkit.messageHandlers.meshViewer.postMessage({ event: 'loaded' });
                }

                // GLTF Export function - called from Swift
                window.exportGLB = async function() {
                    console.log('[MeshViewer] Starting GLB export...');

                    let GLTFExporter;
                    try {
                        ({ GLTFExporter } = await import('three/addons/exporters/GLTFExporter.js'));
                    } catch (error) {
                        console.error('[MeshViewer] Failed to load GLTFExporter:', error);
                        window.webkit.messageHandlers.meshViewer.postMessage({
                            event: 'exportError',
                            message: (error && error.message) ? error.message : 'Failed to load export module'
                        });
                        return;
                    }

                    const exporter = new GLTFExporter();

                    exporter.parse(
                        roomGroup,
                        function(glb) {
                            console.log('[MeshViewer] GLB export complete, size:', glb.byteLength);

                            // Convert ArrayBuffer to base64
                            const bytes = new Uint8Array(glb);
                            let binary = '';
                            const chunkSize = 8192;
                            for (let i = 0; i < bytes.byteLength; i += chunkSize) {
                                const chunk = bytes.subarray(i, Math.min(i + chunkSize, bytes.byteLength));
                                binary += String.fromCharCode.apply(null, chunk);
                            }
                            const base64 = btoa(binary);

                            // Send to Swift
                            window.webkit.messageHandlers.meshViewer.postMessage({
                                event: 'glbExported',
                                data: base64
                            });
                        },
                        function(error) {
                            console.error('[MeshViewer] GLB export error:', error);
                            window.webkit.messageHandlers.meshViewer.postMessage({
                                event: 'exportError',
                                message: error.message || 'Export failed'
                            });
                        },
                        { binary: true }
                    );
                };

                // Animation loop
                function animate() {
                    requestAnimationFrame(animate);
                    controls.update();
                    renderer.render(scene, camera);
                }
                animate();

                // Resize: handled by applyViewportSize() above (WKWebView often reports 0×0 until layout).

                console.log('[MeshViewer] Initialized');
                meshPost('jsLog', { step: 'init_complete', w: vp.w, h: vp.h });

            </script>
        </body>
        </html>
        """
    }
}
