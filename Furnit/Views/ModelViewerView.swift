import SwiftUI
import RealityKit
import Combine
import Photos
import CoreML
import AVFoundation
import UIKit
import simd

struct ModelViewerView: View {
    @ObservedObject private var yoloeService = YOLOEModelService.shared
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
    @State private var shouldResetCamera = false  // ✅ Trigger camera reset on appear

    // Required for RealityKitView
    @StateObject private var arObjectPlacementManager = RealityKitObjectPlacementManager()
    @State private var isARActive = false

    // Camera/Segmentation state
    @State private var showingCameraPreview = false
    @State private var showingSegmentExamine = false
    @State private var showingSegmentForeground = false
    @State private var showingSegmentFurniture = false
    @State private var showingFurnitureFit = false  // FurnitureFit: YOLOE segmentation
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
    @State private var tapHintColorIndex: Int = 0
    private let tapHintColors: [Color] = [.yellow, .cyan, .orange, .green, .pink]
    @State private var tapHintColorTimer: Timer?
    @State private var pinchHintExplanationVisible = false
    @State private var pinchHintHideTextTask: Task<Void, Never>?
    @State private var brainHintExplanationVisible = false
    @State private var brainHintHideTextTask: Task<Void, Never>?
    @State private var snapshotHintExplanationVisible = false
    @State private var snapshotHintHideTextTask: Task<Void, Never>?
    @State private var roomDimensionsHintVisible = false
    @State private var roomDimensionsHintHideTask: Task<Void, Never>?
    @State private var arSizingHintExplanationVisible = false
    @State private var arSizingHintHideTextTask: Task<Void, Never>?
    @State private var arSizingHintRequiresBrain = false

    @State private var isCapturingSnapshot = false

    init(model: USDZModel) {
        self.model = model
    }

    private var canSegmentSelectedFurniture: Bool {
        showingFurnitureFit && !selectedFurnitureFitLabels.isEmpty
    }

    private var canOfferBrainArAssist: Bool {
        QualitySettings.supportsLiDARSceneDepth &&
        AppStateManager.shared.qualitySettings.furnitureFitARDepthCompanionRuntimeActive
    }

    private var effectiveRoomDimensions: (width: Float, height: Float, depth: Float) {
        (
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

    var body: some View {
        GeometryReader { geometry in
            ZStack {
                // Main content ZStack
                ZStack {
                    RealityKitView(
                        model: model,
                        cameraMovementManager: cameraMovementManager,
                        arObjectPlacementManager: arObjectPlacementManager,
                        isARActive: isARActive,
                        shouldCaptureSnapshot: $shouldCaptureARViewSnapshot,
                        capturedSnapshot: $roomSnapshot,
                        shouldResetCamera: $shouldResetCamera  // ✅ Camera reset trigger
                    )
                    .allowsHitTesting(!(showingCameraPreview || showingSegmentExamine || showingSegmentForeground || showingSegmentFurniture || showingFurnitureFit))
                    .ignoresSafeArea(.all)
                    // FurnitureFit overlay (YOLOE Core ML)
                    if showingFurnitureFit {
                        FurnitureFitUIView(
                            capturedImage: $capturedImage,
                            roomImage: nil,
                            mlModel: yoloeService.model,
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
                            showIdentifyLivePreview: furnitureFitShowIdentifyLivePreview,
                            showFullVideoWithIdentificationsOverride: showFullVideoWithIdentifications
                        )
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                        .zIndex(9000)
                    }

                    if showingFurnitureFit, let fh = furnitureFitEstimatedHeightM {
                        VStack {
                            Spacer()
                            Text(String(format: "Furniture height ~ %.2f m", fh))
                                .font(.caption.bold())
                                .foregroundColor(.white)
                                .padding(.horizontal, 10)
                                .padding(.vertical, 6)
                                .background(Color.black.opacity(0.55))
                                .cornerRadius(8)
                                .padding(.bottom, 96)
                        }
                        .zIndex(9001)
                        .allowsHitTesting(false)
                    }

                    Color.clear
                }

                roomDimensionsHintOverlay
                fullVideoToolbarHelperOverlay
                topTrailingPinchAndSizingHintsOverlay

                if fullVideoFurnitureTapHintVisible {
                    VStack {
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
                            .animation(.easeInOut(duration: 0.6), value: tapHintColorIndex)
                            .padding(.top, 12)
                        Spacer()
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .allowsHitTesting(false)
                    .opacity(isCapturingSnapshot ? 0 : 1)
                    .zIndex(9002)
                }
                
                // Top controls
                VStack {
                    HStack {
                        backButton
                            .allowsHitTesting(true)
                        Spacer()
                        topToolbarContent
                    }
                    .padding()
                    Spacer()
                }
                .opacity(isCapturingSnapshot ? 0 : 1)
                .zIndex(99999) // HIGHEST POSSIBLE Z-INDEX
                .allowsHitTesting(true)

                // TOPMOST BRAIN ICON - ALWAYS ON TOP
                VStack {
                    Spacer()
                    HStack {
                        VStack(spacing: 16) {
                            HStack(alignment: .bottom, spacing: 8) {
                                brainButtonWithHintAbove
                            }

                            if showingFurnitureFit && showFullVideoWithIdentifications {
                                Button(action: {
                                    if furnitureFitSegmentationMode == .segmentSelected {
                                        furnitureFitSegmentationMode = .identifyOnly
                                        furnitureFitShowIdentifyLivePreview = true
                                    } else {
                                        guard canSegmentSelectedFurniture else { return }
                                        furnitureFitSegmentationMode = .segmentSelected
                                        dismissFullVideoFurnitureTapHint()
                                    }
                                }) {
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
                                .disabled(furnitureFitSegmentationMode != .segmentSelected && !canSegmentSelectedFurniture)
                                .accessibilityLabel(L10n.RoomViewer.segmentFurnitureAccessibility)
                            }
                        }
                        .padding(.leading, 16)
                        .padding(.bottom, 20)
                        Spacer()
                        roomIntelligencePlacementCardResetOnExit
                            .padding(.bottom, 26)
                        Spacer()
                    }
                }
                .opacity(isCapturingSnapshot ? 0 : 1)
                .zIndex(99998) // SECOND HIGHEST Z-INDEX
                .allowsHitTesting(true)
                
                // Camera controls handled by RealityKitGestureHandlers in RealityKitView

                // Screenshot button - ALWAYS VISIBLE (bottom-right, matching SharpRoomView)
                VStack {
                    Spacer()
                    HStack {
                        Spacer()
                        snapshotButtonWithHintAbove
                        .padding(.trailing, 20)
                        .padding(.bottom, 20)
                    }
                }
                .opacity(isCapturingSnapshot ? 0 : 1)
                .zIndex(99996)
            }
        }
        .navigationBarHidden(true)
        .statusBarHidden(true)
        .preferredColorScheme(.dark)
        .onChange(of: showingCameraPreview) { _, _ in manageARSessionForOverlays() }
        .onChange(of: showingSegmentExamine) { _, _ in manageARSessionForOverlays() }
        .onChange(of: showingSegmentForeground) { _, _ in manageARSessionForOverlays() }
        .onChange(of: showingSegmentFurniture) { _, _ in manageARSessionForOverlays() }
        .onChange(of: showingFurnitureFit) { _, isOn in
            manageARSessionForOverlays()
            if isOn {
                yoloeService.ensureModelLoaded()
                restartBrainGestureHint()
                restartSnapshotGestureHint()
                restartPinchGestureHint()
                updateRoomPlacementIntelligence()
            } else {
                dismissFullVideoFurnitureTapHint()
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
        .onChange(of: segmentedFurnitureMeanSRGB) { _, _ in updateRoomPlacementIntelligence() }
        .onChange(of: detectedFurnitureWidth) { _, _ in updateRoomPlacementIntelligence() }
        .onChange(of: detectedFurnitureHeightAR) { _, _ in updateRoomPlacementIntelligence() }
        .onChange(of: furnitureFitEstimatedHeightM) { _, _ in updateRoomPlacementIntelligence() }
        .onAppear {
            isARActive = true
            yoloeService.ensureModelLoaded()
            // ✅ Reset camera to optimal position every time view appears
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                shouldResetCamera = true
            }
            // Lock orientation based on photo orientation
            if model.photoOrientation == .landscape {
                OrientationLockManager.shared.lockToLandscape()
            } else {
                OrientationLockManager.shared.lockToPortrait()
            }
            restartBrainGestureHint()
            restartSnapshotGestureHint()
            restartPinchGestureHint()
        }
        .onDisappear {
            dismissFullVideoFurnitureTapHint()
            cancelPinchHintTasks()
            cancelBrainHintTasks()
            cancelSnapshotHintTasks()
            cancelRoomDimensionsHintTasks()
            cancelARSizingHintTasks()
            OrientationLockManager.shared.unlock()
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
        HStack(spacing: 12) {
            Button(action: onRoomDimensionsRulerTapped) {
                Image(systemName: "ruler.fill")
                    .font(.system(size: 18, weight: .medium))
                    .foregroundStyle(.white)
            }
            .buttonStyle(.plain)
            .accessibilityLabel(L10n.RoomViewer.checkMeasurement)

            Button(action: onPinchHintIconTapped) {
                Image(systemName: "hand.pinch.fill")
                    .font(.system(size: 18, weight: .medium))
                    .foregroundStyle(.white)
            }
            .buttonStyle(.plain)
            .accessibilityLabel(pinchHintAccessibilityLabel)

            Button(action: displayAllGestureHelpers) {
                Image(systemName: "hand.tap.fill")
                    .font(.system(size: 16, weight: .medium))
                    .foregroundStyle(.white)
            }
            .buttonStyle(.plain)
            .accessibilityLabel(L10n.RoomViewer.displayAllHelpers)

            Button(action: {
                shouldResetCamera = true
            }) {
                Image(systemName: "viewfinder")
                    .font(.system(size: 18, weight: .medium))
                    .foregroundStyle(.white)
            }
            .buttonStyle(.plain)
            .accessibilityLabel(L10n.RoomViewer.recenterView)

            if showingFurnitureFit {
                Button(action: toggleFullVideoIdentifications) {
                    Image(systemName: "text.viewfinder")
                        .font(.system(size: 18, weight: .medium))
                        .symbolVariant(showFullVideoWithIdentifications ? .fill : .none)
                        .foregroundStyle(Color.cyan)
                }
                .buttonStyle(.plain)
                .accessibilityLabel(L10n.Settings.fullVideoWithIdentifications)
                .accessibilityHint(L10n.Settings.fullVideoWithIdentificationsDescription)
                .accessibilityAddTraits(showFullVideoWithIdentifications ? .isSelected : [])
            }

            if showingFurnitureFit && canOfferBrainArAssist {
                Button(action: toggleBrainArAssistedSizingOrShowHint) {
                    Image(systemName: "arrow.up.left.and.arrow.down.right")
                        .font(.system(size: 15, weight: .bold))
                        .foregroundStyle(.white)
                        .frame(width: 28, height: 28)
                        .background(
                            Circle().fill(
                                brainArAssistedSizingEnabled
                                    ? Color.green.opacity(0.9)
                                    : Color.white.opacity(0.18)
                            )
                        )
                }
                .buttonStyle(.plain)
                .accessibilityLabel(
                    brainArAssistedSizingEnabled ? L10n.RoomViewer.arSizingDisable : L10n.RoomViewer.arSizingEnable
                )
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
        .background(Color.black.opacity(0.72))
        .clipShape(Capsule())
    }

    private var roomDimensionsHintText: String {
        let dims = effectiveRoomDimensions
        if dims.width > 0.05, dims.height > 0.05, dims.depth > 0.05,
           dims.width.isFinite, dims.height.isFinite, dims.depth.isFinite {
            if model.roomWidth != nil || model.roomHeight != nil || model.roomDepth != nil {
                return L10n.RoomViewer.roomDimensionsWHDAIChip(width: dims.width, height: dims.height, depth: dims.depth)
            }
            return L10n.RoomViewer.roomDimensionsWHDManualChip(width: dims.width, height: dims.height, depth: dims.depth)
        }
        if dims.width > 0.05, dims.height > 0.05,
           dims.width.isFinite, dims.height.isFinite {
            if model.roomWidth != nil || model.roomHeight != nil {
                return L10n.RoomViewer.roomDimensionsWHDAIChip(width: dims.width, height: dims.height, depth: max(dims.depth, 0.05))
            }
            return L10n.RoomViewer.roomDimensionsWHManualChip(width: dims.width, height: dims.height)
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
        }
        .allowsHitTesting(false)
        .zIndex(104)
    }

    private var fullVideoToolbarHelperOverlay: some View {
        ZStack(alignment: .topTrailing) {
            Color.clear
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .allowsHitTesting(false)
            if showingFurnitureFit && showFullVideoWithIdentifications {
                VStack(alignment: .trailing, spacing: 4) {
                    Image(systemName: "arrow.up")
                        .font(.caption.weight(.bold))
                        .foregroundColor(.cyan)
                        .padding(.trailing, 18)
                    Text(L10n.RoomViewer.fullVideoSelectionHelper)
                        .font(.caption2)
                        .foregroundColor(.cyan)
                        .multilineTextAlignment(.leading)
                        .fixedSize(horizontal: false, vertical: true)
                        .frame(maxWidth: 220, alignment: .leading)
                        .padding(8)
                        .background(
                            RoundedRectangle(cornerRadius: 8)
                                .fill(Color.black.opacity(0.78))
                                .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color.cyan.opacity(0.75), lineWidth: 1))
                        )
                }
                .padding(.top, 6)
                .padding(.trailing, 52)
                .offset(y: -24)
            }
        }
        .allowsHitTesting(false)
        .zIndex(106)
    }

    private var topTrailingPinchAndSizingHintsOverlay: some View {
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
                }
                if arSizingHintExplanationVisible, showingFurnitureFit || arSizingHintRequiresBrain {
                    Text(arSizingHintText)
                        .font(.caption2)
                        .foregroundColor(.white)
                        .multilineTextAlignment(.center)
                        .fixedSize(horizontal: false, vertical: true)
                        .frame(maxWidth: 200)
                        .padding(8)
                        .background(RoundedRectangle(cornerRadius: 8).fill(Color.black.opacity(0.78)))
                }
            }
            .padding(.top, 52)
            .padding(.trailing, 16)
        }
        .allowsHitTesting(false)
        .zIndex(101)
    }

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
    }

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
    }

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
            .contentShape(Circle())
            .frame(width: 76, height: 76)
        }
    }

    private var snapshotButtonWithHintAbove: some View {
        VStack(alignment: .center, spacing: 6) {
            snapshotGestureHintColumn
            Button(action: saveFurnitureFitSnapshot) {
                Image(systemName: "camera.fill")
                    .font(.system(size: 28))
                    .foregroundColor(.white)
                    .frame(width: 60, height: 60)
                    .background(Circle().fill(Color.blue).shadow(radius: 5))
            }
            .disabled(isCapturingSnapshot)
        }
    }

    private var pinchHintAccessibilityLabel: String {
        L10n.RoomViewer.pinchGestureHintExplanation + " " + L10n.RoomViewer.gestureHintToggleAccessibility
    }

    private func toggleFurnitureFit() {
        logDebug("BRAIN FLOW: tap received")
        showFurnitureHint = false
        if showingFurnitureFit {
            dismissFullVideoFurnitureTapHint()
            showingFurnitureFit = false
        } else {
            logDebug("BRAIN FLOW: loading YOLOE and opening FurnitureFit")
            yoloeService.ensureModelLoaded()
            showFullVideoWithIdentifications = false
            brainArAssistedSizingEnabled = false
            furnitureFitSegmentationMode = .identifyOnly
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
        restartPinchGestureHint()
        restartBrainGestureHint()
        restartSnapshotGestureHint()
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
            VStack(spacing: 10) {
                if isPlacementIntelligenceExpanded, let aesthetic = latestAestheticScore {
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
                        Image(systemName: "square.split.2x2.fill")
                            .font(.system(size: 18, weight: .semibold))
                            .foregroundStyle(.white)
                            .symbolRenderingMode(.hierarchical)
                    }
                }
                .buttonStyle(.plain)
                .accessibilityLabel(L10n.RoomViewer.placementIntelligenceTitle)
                .accessibilityAddTraits(.isButton)
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
    
    // MARK: - YOLOE model loaded via YOLOEModelService (ODR)

}

struct FurnitureFitUIView: UIViewRepresentable {
    @Binding var capturedImage: UIImage?

    /// Synced with Settings → Furniture segmentation → primary detection confidence.
    @AppStorage("furnitureFit.primaryDetectionMinConfidence") private var primaryDetectionMinConfidenceStorage: Double = 0.57
    @AppStorage("furnitureFit.primarySelectionByHighestConfidence") private var primarySelectionByHighestConfidence: Bool = false
    @AppStorage("furnitureFit.showFullVideoWithIdentifications") private var showFullVideoWithIdentifications: Bool = false

    var roomImage: UIImage?
    var mlModel: MLModel?  // yoloe-11l-seg-pf via YOLOEModelService
    var processInterval: Double = 0.07
    /// Minimum detector confidence (0…1) for parsing YOLOE candidates.
    /// Matches the updated iOS Core ML path (was 0.25).
    var scoreThreshold: Float = 0.10
    var active: Bool = true
    var lockedOrientation: PhotoOrientation = .portrait  // Room's photo orientation

    // Room dimensions from SHARP (in meters) for furniture sizing
    var roomWidthMeters: Float = 4.0
    var roomHeightMeters: Float = 3.0
    var roomDepthMeters: Float = 4.0
    /// Splat raycast / saved `.meta` scene units for ratio fitment logs.
    var roomRaycastSceneDimensions: RoomRaycastDimensions? = nil
    var roomModel: RoomModel? = nil
    var cameraFocalLengthPixels: Float = 0

    // Callback for reporting estimated furniture size (room-based + optional AR height, in meters)
    var onFurnitureSizeEstimated: ((FurnitureSizeEstimate) -> Void)?
    /// Sharp Room: skip “Starting camera…” progress after the first segmentation this session.
    var suppressStartupProgress: Bool = false
    var onFirstSegmentationComplete: (() -> Void)?
    /// Mean straight sRGB of the composited furniture cutout (throttled); optional for placement / aesthetic UI.
    var onSegmentationMaskMeanColorSRGB: ((SIMD3<Float>) -> Void)? = nil
    /// Sharp Room only: splat depth for furniture sizing.
    var sharpRoomSplatMeasurementHost: GaussianSplatMeasurementHost? = nil
    /// Per-view opt-in for AR-assisted sizing. Sharp Room keeps this off until the user taps the AR chip.
    var arAssistedSizingEnabled: Bool = true
    /// Optional manual override from the room calibrate sheet. When present, AR overlay sizing uses this
    /// height instead of the raw per-frame estimate so the mask scales to the user-confirmed furniture size.
    var manualFurnitureHeightOverrideMeters: Float? = nil
    /// Brain now starts in identify-only mode; Segment enables selected-class masking.
    var segmentationMode: FurnitureFitSegmentationMode = .identifyOnly
    var onSelectedClassLabelsChanged: (([String]) -> Void)? = nil
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
        view.sharpRoomSplatMeasurementHost = sharpRoomSplatMeasurementHost
        view.confidenceThreshold = scoreThreshold
        view.primaryDetectionMinConfidence = Self.clampPrimaryDetectionConfidence(primaryDetectionMinConfidenceStorage)
        view.primarySelectionByHighestConfidence = primarySelectionByHighestConfidence
        view.showFullVideoWithIdentifications = showFullVideoWithIdentificationsOverride ?? showFullVideoWithIdentifications
        view.onFurnitureSizeEstimated = onFurnitureSizeEstimated
        view.suppressStartupProgress = suppressStartupProgress
        view.onFirstSegmentationComplete = onFirstSegmentationComplete
        view.onSegmentationMaskMeanColorSRGB = onSegmentationMaskMeanColorSRGB
        view.arAssistedSizingEnabled = arAssistedSizingEnabled
        view.manualFurnitureHeightOverrideMeters = manualFurnitureHeightOverrideMeters
        view.segmentationMode = segmentationMode
        view.onSelectedClassLabelsChanged = onSelectedClassLabelsChanged
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
            uiView.sharpRoomSplatMeasurementHost = sharpRoomSplatMeasurementHost
            uiView.cameraFocalLengthPixels = cameraFocalLengthPixels
            uiView.confidenceThreshold = scoreThreshold
            uiView.primaryDetectionMinConfidence = Self.clampPrimaryDetectionConfidence(primaryDetectionMinConfidenceStorage)
            uiView.primarySelectionByHighestConfidence = primarySelectionByHighestConfidence
            uiView.showFullVideoWithIdentifications = showFullVideoWithIdentificationsOverride ?? showFullVideoWithIdentifications
            uiView.onFurnitureSizeEstimated = onFurnitureSizeEstimated
            uiView.suppressStartupProgress = suppressStartupProgress
            uiView.onFirstSegmentationComplete = onFirstSegmentationComplete
            uiView.onSegmentationMaskMeanColorSRGB = onSegmentationMaskMeanColorSRGB
            uiView.arAssistedSizingEnabled = arAssistedSizingEnabled
            uiView.manualFurnitureHeightOverrideMeters = manualFurnitureHeightOverrideMeters
            uiView.segmentationMode = segmentationMode
            uiView.onSelectedClassLabelsChanged = onSelectedClassLabelsChanged
            uiView.showIdentifyLivePreview = showIdentifyLivePreview
        }

        applyConfiguration()
        if active {
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
        uiView.sharpRoomSplatMeasurementHost = nil
        uiView.stop()
    }

    private static func clampPrimaryDetectionConfidence(_ raw: Double) -> Float {
        Float(min(max(raw, 0.05), 0.99))
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
