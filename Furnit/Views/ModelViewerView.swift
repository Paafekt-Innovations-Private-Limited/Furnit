import SwiftUI
import RealityKit
import Combine
import Photos
import CoreML
import AVFoundation
import UIKit
import simd

struct ModelViewerView: View {
    @ObservedObject private var rtmdetService = RTMDetModelService.shared
    @Environment(\.horizontalSizeClass) var horizontalSizeClass
    @Environment(\.verticalSizeClass) var verticalSizeClass
    @EnvironmentObject var authManager: AuthenticationManager
    let model: USDZModel
    @Environment(\.dismiss) private var dismiss
    @Environment(\.presentationMode) var presentationMode
    @AppStorage("singlePhotoRoom.width") private var defaultRoomWidth: Double = 4.0
    @AppStorage("singlePhotoRoom.depth") private var defaultRoomDepth: Double = 4.5
    @AppStorage("singlePhotoRoom.height") private var defaultRoomHeight: Double = 2.8

    // Camera movement state
    @StateObject private var cameraMovementManager = RealityKitCameraMovementManager()

    // Required for RealityKitView
    @StateObject private var arObjectPlacementManager = RealityKitObjectPlacementManager()
    @State private var isARActive = false

    // Camera/Segmentation state
    @State private var showingCameraPreview = false
    @State private var showingSegmentExamine = false
    @State private var showingSegmentForeground = false
    @State private var showingSegmentFurniture = false
    @State private var showingFurnitureFit = false  // FurnitureFit: RTMDet segmentation
    @State private var furnitureFitSegmentationMode: FurnitureFitSegmentationMode = .identifyOnly
    @State private var furnitureFitShowIdentifyLivePreview = true
    @State private var selectedFurnitureFitLabels: [String] = []
    @State private var furnitureFitInitialSegmentationDone = false
    @State private var detectedFurnitureWidth: Float?
    @State private var furnitureFitEstimatedHeightM: Float?
    @State private var detectedFurnitureHeightAR: Float?
    @State private var capturedImage: UIImage? = nil
    @State private var roomSnapshot: UIImage? = nil
    @State private var latestFitCheckResult: FitCheckResult?
    @State private var latestAestheticScore: AestheticScore?
    @State private var segmentedFurnitureMeanSRGB: SIMD3<Float>?
    @State private var isPlacementIntelligenceExpanded = false
    
    // ARView snapshot trigger (proper way to capture 3D content)
    @State private var shouldCaptureARViewSnapshot = false
    
    // Furniture hint
    @State private var showFurnitureHint = true
    @State private var showFullVideoWithIdentifications = false
    @State private var brainArAssistedSizingEnabled = false
    @State private var fullVideoFurnitureTapHintVisible = false
    @State private var pinchHintExplanationVisible = false
    @State private var pinchHintHideTextTask: Task<Void, Never>?
    @State private var replayTeachingHints = false
    @State private var roomDimensionsHintVisible = false
    @State private var roomDimensionsHintHideTask: Task<Void, Never>?
    @State private var arSizingHintExplanationVisible = false
    @State private var arSizingHintHideTextTask: Task<Void, Never>?
    @State private var arSizingHintRequiresBrain = false
    @State private var fullVideoSelectionHelperVisible = false
    @State private var fullVideoSelectionHelperHideTask: Task<Void, Never>?

    @State private var isCapturingSnapshot = false
    @StateObject private var immersiveChrome = PaafektViewerChromeController()
    @Environment(\.modelViewerSuppressBuiltInTopChrome) private var suppressBuiltInTopChrome
    @Environment(\.modelViewerExternalCameraReset) private var externalCameraReset
    @State private var shouldResetCamera = false

    init(model: USDZModel) {
        self.model = model
    }

    private var cameraResetBinding: Binding<Bool> {
        externalCameraReset ?? $shouldResetCamera
    }

    private var canSegmentSelectedFurniture: Bool {
        showingFurnitureFit && !selectedFurnitureFitLabels.isEmpty
    }

    private var canOfferBrainArAssist: Bool {
        QualitySettings.supportsLiDARSceneDepth &&
        AppStateManager.shared.qualitySettings.furnitureFitARDepthCompanionRuntimeActive
    }

    private var effectiveRoomDimensions: (width: Float, height: Float, depth: Float) {
        // Depth Anything: only persisted inference dims — never Settings defaults or mesh/bounds.
        if model.roomCoordinateFrame == .depthAnythingImageDepthMeters {
            guard let width = model.roomWidth,
                  let height = model.roomHeight,
                  let depth = model.roomDepth,
                  width.isFinite, height.isFinite, depth.isFinite,
                  width > 0.05, height > 0.05, depth > 0.05 else {
                return (0, 0, 0)
            }
            return (width, height, depth)
        }
        if let width = model.roomWidth,
           let height = model.roomHeight,
           let depth = model.roomDepth,
           width.isFinite, height.isFinite, depth.isFinite,
           width > 0.05, height > 0.05, depth > 0.05 {
            return (width, height, depth)
        }
        // Depth Anything / LiDAR / Swift Splat rooms must not fall back to Settings defaults.
        if model.roomCoordinateFrame.usesNativeMeterSceneUnits {
            return (0, 0, 0)
        }
        return (
            model.roomWidth ?? Float(defaultRoomWidth),
            model.roomHeight ?? Float(defaultRoomHeight),
            model.roomDepth ?? Float(defaultRoomDepth)
        )
    }

    private var authoritativeRoomModelForMetrics: RoomModel? {
        let dims = effectiveRoomDimensions
        guard dims.width.isFinite, dims.height.isFinite, dims.depth.isFinite,
              dims.width > 0.05, dims.height > 0.05, dims.depth > 0.05 else {
            return nil
        }
        return ModelViewerPlacementIntelligenceRoomStub.axisAlignedBoxMeters(
            width: dims.width,
            height: dims.height,
            depth: dims.depth
        )
    }

    @ViewBuilder
    private var furnitureFitCameraOverlay: some View {
        if showingFurnitureFit {
            FurnitureFitUIView(
                capturedImage: $capturedImage,
                roomImage: nil,
                mlModel: rtmdetService.model,
                processInterval: 0.07,
                active: true,
                lockedOrientation: model.photoOrientation,
                roomWidthMeters: effectiveRoomDimensions.width,
                roomHeightMeters: effectiveRoomDimensions.height,
                roomDepthMeters: effectiveRoomDimensions.depth,
                onFurnitureSizeEstimated: { estimate in
                    detectedFurnitureWidth = estimate.widthMeters
                    furnitureFitEstimatedHeightM = estimate.heightMeters
                    detectedFurnitureHeightAR = estimate.arHeightMeters
                },
                suppressStartupProgress: furnitureFitInitialSegmentationDone,
                onFirstSegmentationComplete: { furnitureFitInitialSegmentationDone = true },
                onSegmentationMaskMeanColorSRGB: { meanSRGB in
                    segmentedFurnitureMeanSRGB = meanSRGB
                },
                arAssistedSizingEnabled: brainArAssistedSizingEnabled && canOfferBrainArAssist,
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
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .ignoresSafeArea()
            .zIndex(9000)
        }
    }

    @ViewBuilder
    private var furnitureHeightEstimateOverlay: some View {
        if showingFurnitureFit, let fh = furnitureFitEstimatedHeightM {
            VStack {
                Spacer()
                PaafektRoomMeasurementPill(
                    primaryText: L10n.RoomViewer.furnitureHeightEstimate(fh)
                )
                .padding(.bottom, 96)
            }
            .zIndex(9001)
            .allowsHitTesting(false)
        }
    }

    private var modelViewerRealityAndFurnitureUnderlay: some View {
        ZStack {
            RealityKitView(
                model: model,
                cameraMovementManager: cameraMovementManager,
                arObjectPlacementManager: arObjectPlacementManager,
                isARActive: isARActive,
                shouldCaptureSnapshot: $shouldCaptureARViewSnapshot,
                capturedSnapshot: $roomSnapshot,
                shouldResetCamera: cameraResetBinding  // ✅ Camera reset trigger
            )
            .allowsHitTesting(!(showingCameraPreview || showingSegmentExamine || showingSegmentForeground || showingSegmentFurniture || showingFurnitureFit))
            .ignoresSafeArea(.all)
            furnitureFitCameraOverlay
            furnitureHeightEstimateOverlay
            Color.clear
        }
        .paafektImmersiveRoomSummonTap(
            chrome: immersiveChrome,
            enabled: !(showingFurnitureFit && showFullVideoWithIdentifications),
            hideForCapture: isCapturingSnapshot,
            onRestingTap: {
                if showingFurnitureFit,
                   showFullVideoWithIdentifications,
                   furnitureFitSegmentationMode == .segmentSelected {
                    furnitureFitSegmentationMode = .identifyOnly
                    furnitureFitShowIdentifyLivePreview = true
                } else {
                    immersiveChrome.summon()
                }
            }
        )
    }

    private var fullVideoFurnitureTapBubbleOverlay: some View {
        Group {
            if fullVideoFurnitureTapHintVisible {
                VStack {
                    PaafektHintChip(
                        systemImage: "hand.tap.fill",
                        text: L10n.RoomViewer.fullVideoFurnitureTapHint,
                        maxWidth: 280
                    )
                    .padding(.top, 12)
                    Spacer()
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .allowsHitTesting(false)
                .opacity(isCapturingSnapshot ? 0 : 1)
                .zIndex(99_999)
            }
        }
    }

    @ViewBuilder
    private var modelViewerRoomDimensionsHintLayer: some View {
        if !suppressBuiltInTopChrome {
            roomDimensionsHintOverlay
        }
    }

    @ViewBuilder
    private var modelViewerTopChromeLayer: some View {
        if !suppressBuiltInTopChrome {
            modelViewerTopChrome
        }
    }

    /// Legacy top chrome when embedded with `modelViewerSuppressBuiltInTopChrome`.
    private var modelViewerTopChrome: some View {
        VStack {
            HStack {
                backButton
                    .allowsHitTesting(true)
                Spacer()
                topToolbarContent
            }
            .padding()
            Spacer()
                .allowsHitTesting(false)
        }
        .opacity(isCapturingSnapshot ? 0 : 1)
        .zIndex(99999) // HIGHEST POSSIBLE Z-INDEX
        .allowsHitTesting(true)
    }

    private func exitFullVideoSegmentation() {
        furnitureFitSegmentationMode = .identifyOnly
        furnitureFitShowIdentifyLivePreview = true
    }

    private func activateModelViewerSelectedFurnitureSegmentation() {
        if furnitureFitSegmentationMode == .segmentSelected {
            exitFullVideoSegmentation()
            return
        }
        guard canSegmentSelectedFurniture else { return }
        furnitureFitSegmentationMode = .segmentSelected
        dismissFullVideoFurnitureTapHint()
    }

    private var modelViewerMorphingPrimaryAction: PaafektMorphingPrimaryAction {
        PaafektMorphingPrimaryActionResolver.resolve(
            showingFurnitureFit: showingFurnitureFit,
            showFullVideoWithIdentifications: showFullVideoWithIdentifications,
            segmentationMode: furnitureFitSegmentationMode,
            hasSelectedObject: !selectedFurnitureFitLabels.isEmpty
        )
    }

    private var modelViewerMorphingPrimaryDisabled: Bool {
        isCapturingSnapshot ||
            (modelViewerMorphingPrimaryAction == .segment && !canSegmentSelectedFurniture)
    }

    private func handleModelViewerMorphingPrimaryTap() {
        immersiveChrome.noteChromeInteraction()
        let action = modelViewerMorphingPrimaryAction
        guard action != .segment || canSegmentSelectedFurniture else { return }
        PaafektMorphingPrimaryActionHandler.perform(
            action,
            enterFit: toggleFurnitureFit,
            exitFit: toggleFurnitureFit,
            segment: activateModelViewerSelectedFurnitureSegmentation,
            finishSegmentation: activateModelViewerSelectedFurnitureSegmentation
        )
    }

    private var modelViewerBottomHeroChrome: some View {
        VStack {
            Spacer()
            ZStack(alignment: .bottomTrailing) {
                ZStack(alignment: .bottom) {
                    VStack(spacing: 10) {
                        PaafektViewerCaptureHeroButton(
                            isDisabled: isCapturingSnapshot,
                            action: saveFurnitureFitSnapshot
                        )
                    }
                    roomIntelligencePlacementCardResetOnExit
                        .padding(.bottom, 56)
                }
                .padding(.horizontal, Theme.Space.lg)
                .padding(.bottom, 20)
                .padding(.trailing, 88)
                .opacity(isCapturingSnapshot ? 0 : 1)
                .allowsHitTesting(!isCapturingSnapshot)

                PaafektMorphingPrimaryFAB(
                    action: modelViewerMorphingPrimaryAction,
                    isDisabled: modelViewerMorphingPrimaryDisabled,
                    onTap: handleModelViewerMorphingPrimaryTap
                )
                .padding(.horizontal, Theme.Space.lg)
                .padding(.bottom, 20)
            }
        }
        .zIndex(99998)
        .allowsHitTesting(true)
    }

    private var modelViewerRestingMeasurementPillText: String? {
        let dims = effectiveRoomDimensions
        return PaafektRoomMeasurementDisplay.restingPillText(
            width: dims.width,
            height: dims.height,
            depth: dims.depth,
            emphasizeHeight: model.roomCoordinateFrame == .depthAnythingImageDepthMeters
        )
    }

    private var modelViewerImmersiveChromeOverlay: some View {
        PaafektImmersiveViewerChromeStack(
            chrome: immersiveChrome,
            onBack: {
                if #available(iOS 15.0, *) {
                    dismiss()
                } else {
                    presentationMode.wrappedValue.dismiss()
                }
            },
            morphingPrimaryAction: modelViewerMorphingPrimaryAction,
            onMorphingPrimary: handleModelViewerMorphingPrimaryTap,
            morphingPrimaryDisabled: modelViewerMorphingPrimaryDisabled,
            measurementText: modelViewerRestingMeasurementPillText,
            hideForCapture: isCapturingSnapshot
        ) {
            PaafektImmersiveSummonedToolbar(chrome: immersiveChrome) {
                HStack(spacing: Theme.Space.sm) {
                    PaafektViewerToolbarIconButton(
                        systemName: "viewfinder",
                        accessibilityLabel: L10n.RoomViewer.recenterView
                    ) {
                        immersiveChrome.noteChromeInteraction()
                        cameraResetBinding.wrappedValue = true
                    }
                    PaafektViewerToolbarIconButton(
                        systemName: "ruler",
                        accessibilityLabel: L10n.RoomViewer.checkMeasurement
                    ) {
                        immersiveChrome.noteChromeInteraction()
                        onRoomDimensionsRulerTapped()
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
                    isDisabled: isCapturingSnapshot
                ) {
                    immersiveChrome.noteChromeInteraction()
                    saveFurnitureFitSnapshot()
                }
            }
        } summonedExtras: {
            PaafektImmersiveFitClusterRows {
                if showingFurnitureFit {
                    roomIntelligencePlacementCardResetOnExit
                }
            }
        } restingAccessory: {
            EmptyView()
        } persistentOverlay: {
            EmptyView()
        }
        .zIndex(99998)
    }

    private var modelViewerInteractiveStack: some View {
        ZStack {
            modelViewerRealityAndFurnitureUnderlay
            modelViewerRoomDimensionsHintLayer
            // Full-video task helper stays visible even after chrome auto-hides.
            fullVideoFurnitureTapBubbleOverlay
            if suppressBuiltInTopChrome || immersiveChrome.isSummoned {
                fullVideoToolbarHelperOverlay
                topTrailingPinchAndSizingHintsOverlay
                navigationTeachingHintBottomOverlay
            }
            if suppressBuiltInTopChrome {
                fullVideoModeFloatingButtonOverlay
                modelViewerTopChromeLayer
                modelViewerBottomHeroChrome
            } else {
                modelViewerImmersiveChromeOverlay
            }
            PaafektViewerOnboardingLayer(
                isReady: true,
                isChromeSummoned: suppressBuiltInTopChrome || immersiveChrome.isSummoned,
                heroHintBottomInset: showingFurnitureFit ? 220 : 172,
                replayTeachingHints: $replayTeachingHints
            )
                .zIndex(100_000)
        }
    }

    private var modelViewerInGeometry: some View {
        GeometryReader { (_: GeometryProxy) in
            modelViewerInteractiveStack
        }
    }

    private var modelViewerNavigationChrome: some View {
        Group {
            if suppressBuiltInTopChrome {
                modelViewerInGeometry
                    .statusBarHidden(true)
                    .preferredColorScheme(.dark)
            } else {
                modelViewerInGeometry
                    .navigationBarHidden(true)
                    .statusBarHidden(true)
                    .preferredColorScheme(.dark)
                    .toolbar(.hidden, for: .navigationBar)
            }
        }
    }

    private var modelViewerWithSessionObservers: some View {
        modelViewerNavigationChrome
            .onChange(of: showingCameraPreview) { _, _ in manageARSessionForOverlays() }
            .onChange(of: showingSegmentExamine) { _, _ in manageARSessionForOverlays() }
            .onChange(of: showingSegmentForeground) { _, _ in manageARSessionForOverlays() }
            .onChange(of: showingSegmentFurniture) { _, _ in manageARSessionForOverlays() }
    }

    private var modelViewerWithFurnitureObservers: some View {
        modelViewerWithSessionObservers
            .onChange(of: showingFurnitureFit) { _, isOn in
                modelViewerHandleShowingFurnitureFitChanged(isOn: isOn)
            }
            .onChange(of: furnitureFitSegmentationMode) { _, mode in
                PaafektFullVideoSegmentationExitDiagnostics.logModeChange(
                    viewer: "ModelViewerView",
                    mode: mode,
                    showingFurnitureFit: showingFurnitureFit,
                    showFullVideoWithIdentifications: showFullVideoWithIdentifications
                )
            }
            .onChange(of: selectedFurnitureFitLabels) { oldLabels, newLabels in
                restoreFullVideoIdentifyAfterSegmentPinsLost(oldLabels: oldLabels, newLabels: newLabels)
            }
    }

    private func modelViewerHandleShowingFurnitureFitChanged(isOn: Bool) {
        manageARSessionForOverlays()
        if isOn {
            rtmdetService.ensureModelLoaded()
            updateRoomPlacementIntelligence()
            presentFullVideoSelectionHelperIfNeeded()
        } else {
            dismissFullVideoFurnitureTapHint()
            cancelFullVideoSelectionHelper()
            showFullVideoWithIdentifications = false
            furnitureFitSegmentationMode = .identifyOnly
            furnitureFitShowIdentifyLivePreview = true
            selectedFurnitureFitLabels = []
            roomSnapshot = nil
            capturedImage = nil
            detectedFurnitureWidth = nil
            furnitureFitEstimatedHeightM = nil
            detectedFurnitureHeightAR = nil
            latestFitCheckResult = nil
            latestAestheticScore = nil
            segmentedFurnitureMeanSRGB = nil
            isPlacementIntelligenceExpanded = false
            brainArAssistedSizingEnabled = false
            cancelARSizingHintTasks()
            arSizingHintExplanationVisible = false
        }
    }

    private var modelViewerRoot: some View {
        modelViewerWithFurnitureObservers
            .onChange(of: segmentedFurnitureMeanSRGB) { _, _ in updateRoomPlacementIntelligence() }
            .onChange(of: detectedFurnitureWidth) { _, _ in updateRoomPlacementIntelligence() }
            .onChange(of: detectedFurnitureHeightAR) { _, _ in updateRoomPlacementIntelligence() }
            .onChange(of: furnitureFitEstimatedHeightM) { _, _ in updateRoomPlacementIntelligence() }
    }

    var body: some View {
        modelViewerRoot
            .onAppear {
                modelViewerPerformOnAppear()
            }
            .onDisappear {
                modelViewerPerformOnDisappear()
            }
    }

    private func modelViewerPerformOnAppear() {
        isARActive = true
        rtmdetService.ensureModelLoaded()
        // Deferred reset so RealityKit has a frame to settle before reframing.
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
            cameraResetBinding.wrappedValue = true
        }
        if model.photoOrientation == .landscape {
            OrientationLockManager.shared.lockToLandscape()
        } else {
            OrientationLockManager.shared.lockToPortrait()
        }
    }

    private func modelViewerPerformOnDisappear() {
        dismissFullVideoFurnitureTapHint()
        cancelPinchHintTasks()
        cancelRoomDimensionsHintTasks()
        cancelARSizingHintTasks()
        cancelFullVideoSelectionHelper()
        // Returning from a room viewer must restore the portrait-only room library.
        OrientationLockManager.shared.lockToPortrait()
    }

    private func dismissFullVideoFurnitureTapHint() {
        fullVideoFurnitureTapHintVisible = false
    }

    private func presentFullVideoFurnitureTapHintIfNeeded() {
        guard showFullVideoWithIdentifications else { return }
        fullVideoFurnitureTapHintVisible = true
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

    // MARK: - Overlays & Controls

    private var backButton: some View {
        Button(action: {
            // Try both dismiss methods for better compatibility
            if #available(iOS 15.0, *) {
                dismiss()
            } else {
                presentationMode.wrappedValue.dismiss()
            }
        }) {
            HStack(spacing: 6) {
                Image(systemName: "chevron.left")
                    .font(.system(size: 16, weight: .medium))
                Text("Back")
                    .font(.system(size: 16, weight: .medium))
            }
            .foregroundColor(.white)
            .padding(.horizontal, 16)
            .padding(.vertical, 8)
            .background(Color.black.opacity(0.7))
            .cornerRadius(20)
        }
    }

    private var topToolbarContent: some View {
        PaafektViewerToolbarCapsule {
            HStack(spacing: 12) {
                PaafektViewerToolbarIconButton(
                    systemName: "ruler.fill",
                    accessibilityLabel: L10n.RoomViewer.checkMeasurement,
                    action: onRoomDimensionsRulerTapped
                )

                PaafektViewerToolbarIconButton(
                    systemName: "hand.pinch.fill",
                    accessibilityLabel: pinchHintAccessibilityLabel,
                    action: onPinchHintIconTapped
                )

                PaafektViewerToolbarIconButton(
                    systemName: "hand.tap.fill",
                    fontSize: 16,
                    accessibilityLabel: L10n.RoomViewer.displayAllHelpers,
                    action: displayAllGestureHelpers
                )

                PaafektViewerToolbarIconButton(
                    systemName: "viewfinder",
                    accessibilityLabel: L10n.RoomViewer.recenterView,
                    action: { cameraResetBinding.wrappedValue = true }
                )

                if showingFurnitureFit && canOfferBrainArAssist {
                    PaafektViewerToolbarIconButton(
                        systemName: "arrow.up.left.and.arrow.down.right",
                        isActive: brainArAssistedSizingEnabled,
                        fontSize: 15,
                        accessibilityLabel: brainArAssistedSizingEnabled ? L10n.RoomViewer.arSizingDisable : L10n.RoomViewer.arSizingEnable,
                        action: toggleBrainArAssistedSizingOrShowHint
                    )
                }
            }
        }
    }

    private var fullVideoIdentificationsFloatingButton: some View {
        Button(action: toggleFullVideoIdentifications) {
            Image(systemName: "text.viewfinder")
                .font(.system(size: 16, weight: .semibold))
                .symbolVariant(showFullVideoWithIdentifications ? .fill : .none)
                .foregroundStyle(Color.cyan)
                .frame(width: 36, height: 36)
                .background(Circle().fill(Color.black.opacity(0.68)))
                .overlay(
                    Circle().stroke(
                        showFullVideoWithIdentifications ? Color.cyan.opacity(0.9) : Color.white.opacity(0.18),
                        lineWidth: 1
                    )
                )
        }
        .buttonStyle(.plain)
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
                    .padding(.top, 88)
                    .padding(.trailing, 8)
            }
        }
        .opacity(isCapturingSnapshot ? 0 : 1)
        .zIndex(99997)
    }

    private var roomDimensionsHintText: String {
        let dims = effectiveRoomDimensions
        if let text = PaafektRoomMeasurementDisplay.rulerHintText(
            width: dims.width,
            height: dims.height,
            depth: dims.depth,
            showFullWHD: model.roomCoordinateFrame == .depthAnythingImageDepthMeters
        ) {
            return text
        }
        if model.roomCoordinateFrame.usesNativeMeterSceneUnits {
            return "ROOM_DIMS unavailable"
        }
        return "3D Room"
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
                        text: roomDimensionsHintText,
                        maxWidth: 280
                    )
                    .transition(.opacity)
                }
            }
            .padding(.top, 12)
        }
        .allowsHitTesting(false)
        .zIndex(104)
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
                    text: L10n.RoomViewer.fullVideoSelectionHelper                )
                .padding(.top, 6)
                .padding(.trailing, 54)
                .offset(y: 108)
            }
        }
        .allowsHitTesting(false)
        .zIndex(106)
    }

    private var topTrailingPinchAndSizingHintsOverlay: some View {
        paafektTopToolbarHintOverlay(isVisible: arSizingHintExplanationVisible && (showingFurnitureFit || arSizingHintRequiresBrain)) {
            PaafektHintChip(
                systemImage: "arrow.up.left.and.arrow.down.right",
                text: arSizingHintText,
                maxWidth: 220
            )
            .transition(.opacity)
        }
        .zIndex(101)
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

    private var pinchHintAccessibilityLabel: String {
        L10n.RoomViewer.pinchGestureHintExplanation + " " + L10n.RoomViewer.gestureHintToggleAccessibility
    }

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

    private func toggleFurnitureFit() {
        logDebug("BRAIN FLOW: tap received")
        showFurnitureHint = false
        if showingFurnitureFit {
            dismissFullVideoFurnitureTapHint()
            cancelFullVideoSelectionHelper()
            showingFurnitureFit = false
        } else {
            logDebug("BRAIN FLOW: loading RTMDet and opening FurnitureFit")
            rtmdetService.ensureModelLoaded()
            showFullVideoWithIdentifications = false
            cancelFullVideoSelectionHelper()
            brainArAssistedSizingEnabled = false
            // Brain default: auto-segment the highest-confidence detection (no tap needed). The
            // tap-to-select flow lives behind the full-video (text.viewfinder) toolbar button.
            furnitureFitSegmentationMode = .segmentPrimary
            furnitureFitShowIdentifyLivePreview = true
            selectedFurnitureFitLabels = []
            showingCameraPreview = false
            showingSegmentExamine = false
            showingSegmentForeground = false
            showingSegmentFurniture = false
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.2) {
                logDebug("BRAIN FLOW: showing FurnitureFit overlay")
                self.furnitureFitInitialSegmentationDone = false
                self.showingFurnitureFit = true
            }
        }
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

    private func toggleBrainArAssistedSizingOrShowHint() {
        guard showingFurnitureFit else {
            showARSizingHint(requiresBrain: true)
            return
        }
        brainArAssistedSizingEnabled.toggle()
        showARSizingHint(requiresBrain: false)
    }

    private func displayAllGestureHelpers() {
        onRoomDimensionsRulerTapped()
        replayTeachingHints = true
        restartPinchGestureHint()
        if canOfferBrainArAssist {
            showARSizingHint(requiresBrain: !showingFurnitureFit)
        }
    }

    private func onRoomDimensionsRulerTapped() {
        cancelRoomDimensionsHintTasks()
        roomDimensionsHintVisible.toggle()
        if roomDimensionsHintVisible {
            scheduleRoomDimensionsHintAutoHide(seconds: 3)
        }
    }

    private var arSizingHintText: String {
        arSizingHintRequiresBrain
            ? L10n.RoomViewer.arFurnitureSizingRequiresBrainHint
            : L10n.RoomViewer.arFurnitureSizingHint
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

    private func derivedDetectedFurnitureDimensionsForRoomIntelligence() -> RoomFurnitureDimensions? {
        guard let width = detectedFurnitureWidth, width.isFinite, width > 0.05 else { return nil }
        let height = furnitureFitEstimatedHeightM ?? detectedFurnitureHeightAR
        guard let height, height.isFinite, height > 0.05 else { return nil }
        let estimatedDepth = max(0.25, min(width * 0.72, 1.4))
        return RoomFurnitureDimensions(widthM: width, heightM: height, depthM: estimatedDepth)
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

    private func updateRoomPlacementIntelligence() {
        guard showingFurnitureFit, let roomModel = authoritativeRoomModelForMetrics else {
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

    @ViewBuilder
    private var roomIntelligencePlacementCardResetOnExit: some View {
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
            .onChange(of: showingFurnitureFit) { _, isShowing in
                if !isShowing { isPlacementIntelligenceExpanded = false }
            }
            .onChange(of: latestFitCheckResult?.fitsInRoom) { _, _ in
                if latestFitCheckResult == nil, latestAestheticScore == nil {
                    isPlacementIntelligenceExpanded = false
                }
            }
        }
    }

    private func manageARSessionForOverlays() {
        let shouldRunAR = !(showingCameraPreview ||
                            showingSegmentExamine ||
                            showingSegmentForeground ||
                            showingSegmentFurniture ||
                            showingFurnitureFit)
        if isARActive != shouldRunAR {
            isARActive = shouldRunAR
        }
    }
    
    private func saveUIImageToPhotos(_ image: UIImage) {
        let saveBlock = {
            UIImageWriteToSavedPhotosAlbum(image, nil, nil, nil)
            logDebug("✅ Saved image to Photos via UIImageWriteToSavedPhotosAlbum")
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
            // Fallback for iOS < 14
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
    
    private func saveFurnitureFitSnapshot() {
        isCapturingSnapshot = true
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.06) {
            guard let uiImage = captureAppWindowImage() else {
                isCapturingSnapshot = false
                logDebug("❌ Failed to capture app window image")
                return
            }
            self.saveUIImageToPhotos(uiImage)
            DispatchQueue.main.async {
                self.isCapturingSnapshot = false
            }
        }
    }
    
    private func captureAppWindowImage() -> UIImage? {
        // Prefer the windows from the active window scenes.  Avoid using
        // `UIApplication.shared.windows`, which was deprecated in iOS 15.  We
        // instead collect windows from all connected scenes and search for
        // the key window.  If none exists, we use the first available window.
        let windows: [UIWindow] = {
            // Gather windows from all connected scenes
            let scenes = UIApplication.shared.connectedScenes.compactMap { $0 as? UIWindowScene }
            let sceneWindows = scenes.flatMap { $0.windows }
            // Fallback: if no scenes produce windows (unlikely on modern iOS),
            // return an empty array.
            return sceneWindows
        }()
        guard let window = windows.first(where: { $0.isKeyWindow }) ?? windows.first else {
            return nil
        }
        let format = UIGraphicsImageRendererFormat()
        format.scale = window.traitCollection.displayScale
        let renderer = UIGraphicsImageRenderer(bounds: window.bounds, format: format)
        let image = renderer.image { _ in
            // Using drawHierarchy provides a rendered snapshot with effects
            window.drawHierarchy(in: window.bounds, afterScreenUpdates: true)
        }
        return image
    }
    
    // MARK: - RTMDet model loaded via RTMDetModelService

}

struct FurnitureFitUIView: UIViewRepresentable {
    @Binding var capturedImage: UIImage?

    @AppStorage("furnitureFit.showFullVideoWithIdentifications") private var showFullVideoWithIdentifications: Bool = false

    var roomImage: UIImage?
    var mlModel: MLModel?
    var processInterval: Double = 0.07
    /// Minimum detector confidence (0…1) for parsing RTMDet candidates.
    /// Matches the updated iOS Core ML path (was 0.25).
    var scoreThreshold: Float = 0.10
    var active: Bool = true
    var lockedOrientation: PhotoOrientation = .portrait  // Room's photo orientation

    // Room dimensions from Splat (in meters) for furniture sizing
    var roomWidthMeters: Float = 4.0
    var roomHeightMeters: Float = 3.0
    var roomDepthMeters: Float = 4.0
    /// Splat raycast / saved `.meta` scene units for ratio fitment logs.
    var roomRaycastSceneDimensions: RoomRaycastDimensions? = nil
    var roomModel: RoomModel? = nil
    var cameraFocalLengthPixels: Float = 0

    // Callback for reporting estimated furniture size (room-based + optional AR height, in meters)
    var onFurnitureSizeEstimated: ((FurnitureSizeEstimate) -> Void)?
    /// Splat Room: skip “Starting camera…” progress after the first segmentation this session.
    var suppressStartupProgress: Bool = false
    var onFirstSegmentationComplete: (() -> Void)?
    /// Mean straight sRGB of the composited furniture cutout (throttled); optional for placement / aesthetic UI.
    var onSegmentationMaskMeanColorSRGB: ((SIMD3<Float>) -> Void)? = nil
    /// Splat Room only: splat depth for furniture sizing.
    var splatRoomMeasurementHost: GaussianSplatMeasurementHost? = nil
    /// Per-view opt-in for AR-assisted sizing. Splat Room keeps this off until the user taps the AR chip.
    var arAssistedSizingEnabled: Bool = true
    /// Optional manual override from the room calibrate sheet. When present, AR overlay sizing uses this
    /// height instead of the raw per-frame estimate so the mask scales to the user-confirmed furniture size.
    var manualFurnitureHeightOverrideMeters: Float? = nil
    /// Brain now starts in identify-only mode; Segment enables selected-class masking.
    var segmentationMode: FurnitureFitSegmentationMode = .identifyOnly
    var onSelectedClassLabelsChanged: (([String]) -> Void)? = nil
    var onSegmentationModeChangeRequested: ((FurnitureFitSegmentationMode) -> Void)? = nil
    var showIdentifyLivePreview: Bool = true
    var showFullVideoWithIdentificationsOverride: Bool? = nil

    func makeUIView(context: Context) -> FurnitureFitContainerView {
        let view = FurnitureFitContainerView()
        view.setModel(mlModel)
        view.lockedOrientation = lockedOrientation
        view.roomWidthMeters = roomWidthMeters
        view.roomHeightMeters = roomHeightMeters
        view.roomDepthMeters = roomDepthMeters
        view.roomRaycastSceneDimensions = roomRaycastSceneDimensions
        view.roomModel = roomModel
        view.cameraFocalLengthPixels = cameraFocalLengthPixels
        view.splatRoomMeasurementHost = splatRoomMeasurementHost
        view.confidenceThreshold = scoreThreshold
        view.showFullVideoWithIdentifications = showFullVideoWithIdentificationsOverride ?? showFullVideoWithIdentifications
        view.onFurnitureSizeEstimated = onFurnitureSizeEstimated
        view.suppressStartupProgress = suppressStartupProgress
        view.onFirstSegmentationComplete = onFirstSegmentationComplete
        view.onSegmentationMaskMeanColorSRGB = onSegmentationMaskMeanColorSRGB
        view.arAssistedSizingEnabled = arAssistedSizingEnabled
        view.manualFurnitureHeightOverrideMeters = manualFurnitureHeightOverrideMeters
        view.segmentationMode = segmentationMode
        view.onSelectedClassLabelsChanged = onSelectedClassLabelsChanged
        view.onSegmentationModeChangeRequested = onSegmentationModeChangeRequested
        view.showIdentifyLivePreview = showIdentifyLivePreview
        return view
    }

    func updateUIView(_ uiView: FurnitureFitContainerView, context: Context) {
        let needsCameraPathRestart = uiView.arAssistedSizingEnabled != arAssistedSizingEnabled

        func applyConfiguration() {
            uiView.setModel(mlModel)
            uiView.processInterval = processInterval
            uiView.lockedOrientation = lockedOrientation
            uiView.roomWidthMeters = roomWidthMeters
            uiView.roomHeightMeters = roomHeightMeters
            uiView.roomDepthMeters = roomDepthMeters
            uiView.roomRaycastSceneDimensions = roomRaycastSceneDimensions
            uiView.roomModel = roomModel
            uiView.splatRoomMeasurementHost = splatRoomMeasurementHost
            uiView.cameraFocalLengthPixels = cameraFocalLengthPixels
            uiView.confidenceThreshold = scoreThreshold
            uiView.showFullVideoWithIdentifications = showFullVideoWithIdentificationsOverride ?? showFullVideoWithIdentifications
            uiView.onFurnitureSizeEstimated = onFurnitureSizeEstimated
            uiView.suppressStartupProgress = suppressStartupProgress
            uiView.onFirstSegmentationComplete = onFirstSegmentationComplete
            uiView.onSegmentationMaskMeanColorSRGB = onSegmentationMaskMeanColorSRGB
            uiView.arAssistedSizingEnabled = arAssistedSizingEnabled
            uiView.manualFurnitureHeightOverrideMeters = manualFurnitureHeightOverrideMeters
            uiView.segmentationMode = segmentationMode
            uiView.onSelectedClassLabelsChanged = onSelectedClassLabelsChanged
            uiView.onSegmentationModeChangeRequested = onSegmentationModeChangeRequested
            uiView.showIdentifyLivePreview = showIdentifyLivePreview
        }

        applyConfiguration()
        if active {
            guard mlModel != nil else {
                uiView.showModelUnavailable()
                return
            }
            if needsCameraPathRestart {
                uiView.reconfigureAssistedSizingModeIfNeeded()
            }
            uiView.startIfNeeded()
        } else {
            uiView.stop()
        }
    }

    static func dismantleUIView(_ uiView: FurnitureFitContainerView, coordinator: ()) {
        uiView.setModel(nil)
        uiView.splatRoomMeasurementHost = nil
        uiView.stop()
    }
}

private enum ModelViewerPlacementIntelligenceRoomStub {
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

private struct ModelViewerSuppressBuiltInTopChromeKey: EnvironmentKey {
    static let defaultValue = false
}

private struct ModelViewerExternalCameraResetKey: EnvironmentKey {
    static let defaultValue: Binding<Bool>? = nil
}

extension EnvironmentValues {
    var modelViewerSuppressBuiltInTopChrome: Bool {
        get { self[ModelViewerSuppressBuiltInTopChromeKey.self] }
        set { self[ModelViewerSuppressBuiltInTopChromeKey.self] = newValue }
    }

    var modelViewerExternalCameraReset: Binding<Bool>? {
        get { self[ModelViewerExternalCameraResetKey.self] }
        set { self[ModelViewerExternalCameraResetKey.self] = newValue }
    }
}
