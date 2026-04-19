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

    /// Set when opening a saved mesh room from Home (YOLO ratio calibration).
    var savedRoomModel: USDZModel? = nil

    @Environment(\.dismiss) private var dismiss
    @EnvironmentObject var authManager: AuthenticationManager

    @State private var isLoading = true
    @State private var error: String? = nil

    // Room name for saving
    @State private var showRoomNameInput = false
    @State private var roomName = ""
    @State private var isSavingRoom = false
    @State private var showSaveAlert = false
    @State private var saveAlertMessage = ""
    @State private var saveWasSuccessful = false
    @State private var showDiscardUnsavedAlert = false

    // Brain mode (furniture detection)
    @State private var showingFurnitureFit = false
    @State private var furnitureFitSegmentationMode: FurnitureFitSegmentationMode = .identifyOnly
    @State private var furnitureFitShowIdentifyLivePreview = true
    @State private var selectedFurnitureFitLabels: [String] = []
    @State private var furnitureFitInitialSegmentationDone = false
    @State private var brainArAssistedSizingEnabled = false
    @ObservedObject private var yoloeService = YOLOEModelService.shared
    @ObservedObject private var appState = AppStateManager.shared
    @AppStorage("show_room_furniture_calibrate") private var showRoomFurnitureCalibrate = false
    @State private var detectedFurnitureHeightAR: Float?
    @State private var showFurnitureDimensionsInput = false
    @State private var inputFurnitureHeight: String = ""
    @State private var realFurnitureHeight: Float?
    @State private var roomCalibrationScaleFactor: Float = 1.0
    @State private var calibrationBaselineDetectedHeight: Float?

    // WebView reference for GLTF export
    @State private var webView: WKWebView?

    // Model manager for saving rooms (also used for YOLO ratio metadata merge when viewing saved room)
    @StateObject private var modelManager = USDZModelManager()

    /// Brain hint (above brain): 3s auto-hide; tap hand toggles — same as ``SharpRoomView``.
    @State private var brainHintExplanationVisible = false
    @State private var brainHintHideTextTask: Task<Void, Never>?
    /// Snapshot hint (above camera): same behavior.
    @State private var snapshotHintExplanationVisible = false
    @State private var snapshotHintHideTextTask: Task<Void, Never>?
    /// AR sizing hint (under the sizing button): same behavior as pinch/brain helpers.
    @State private var arSizingHintExplanationVisible = false
    @State private var arSizingHintHideTextTask: Task<Void, Never>?
    @State private var arSizingHintRequiresBrain = false
    /// Ruler tap: show W×H×D chip below top safe area (matches Sharp room dimensions hint).
    @State private var roomDimensionsHintVisible = false
    @State private var roomDimensionsHintHideTask: Task<Void, Never>?
    @State private var showFullVideoWithIdentifications = false
    @State private var fullVideoFurnitureTapHintVisible = false
    @State private var tapHintColorIndex: Int = 0
    private let tapHintColors: [Color] = [.yellow, .cyan, .orange, .green, .pink]
    @State private var tapHintColorTimer: Timer?
    /// Pinch-zoom hint (top-left with D-pad) — same as ``SharpRoomView``.
    @State private var pinchHintExplanationVisible = false
    @State private var pinchHintHideTextTask: Task<Void, Never>?

    private var canOfferBrainArAssist: Bool {
        QualitySettings.supportsFurnitureFitARAssisted &&
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

    var body: some View {
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

            // Saving overlay
            if isSavingRoom {
                VStack(spacing: 16) {
                    ProgressView()
                        .scaleEffect(1.5)
                        .tint(.white)

                    Text("Exporting 3D model...")
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

            if !isLoading {
                cameraButtonsOverlay
                topTrailingPinchTapAndSizingHintsOverlay
            }

            // FurnitureFit overlay (when active) - full screen camera for furniture detection
            if showingFurnitureFit {
                FurnitureFitUIView(
                    capturedImage: .constant(nil),
                    roomImage: nil,
                    mlModel: yoloeService.model,
                    processInterval: 0.07,
                    active: true,
                    lockedOrientation: photoOrientation,
                    roomWidthMeters: calibratedRoomWidth,
                    roomHeightMeters: calibratedRoomHeight,
                    roomDepthMeters: calibratedRoomDepth,
                    onFurnitureSizeEstimated: { estimate in
                        if let arHeight = estimate.arHeightMeters,
                           arHeight.isFinite,
                           arHeight > 0.05 {
                            detectedFurnitureHeightAR = arHeight
                        } else {
                            detectedFurnitureHeightAR = nil
                        }
                    },
                    suppressStartupProgress: furnitureFitInitialSegmentationDone,
                    onFirstSegmentationComplete: { furnitureFitInitialSegmentationDone = true },
                    arAssistedSizingEnabled: brainArAssistedSizingEnabled && canOfferBrainArAssist,
                    segmentationMode: furnitureFitSegmentationMode,
                    onSelectedClassLabelsChanged: { labels in
                        selectedFurnitureFitLabels = labels
                    },
                    showIdentifyLivePreview: furnitureFitShowIdentifyLivePreview,
                    showFullVideoWithIdentificationsOverride: showFullVideoWithIdentifications
                )
                .ignoresSafeArea(edges: [.bottom, .leading, .trailing])
                .zIndex(100)
            }

            // Room dimensions chip (ruler) — same placement idea as ``SharpRoomView``.
            roomDimensionsHintOverlay
            fullVideoFurnitureTapHintOverlay
            if showFurnitureDimensionsInput, showRoomFurnitureCalibrate, supportsMetricFurnitureMeasurementUI {
                calibrationOverlayView
                    .onAppear { calibrationBaselineDetectedHeight = detectedFurnitureHeightAR }
                    .onDisappear { calibrationBaselineDetectedHeight = nil }
            }

            // Controls based on orientation
            if photoOrientation == .landscape {
                landscapeControls
            } else {
                portraitControls
            }
        }
        .background(Color.gray)
        .navigationTitle("")
        .navigationBarTitleDisplayMode(.inline)
        .navigationBarBackButtonHidden(savedRoomModel == nil)
        .toolbar {
            ToolbarItem(placement: .navigationBarLeading) {
                if savedRoomModel == nil {
                    Button {
                        if saveWasSuccessful {
                            dismiss()
                        } else {
                            showDiscardUnsavedAlert = true
                        }
                    } label: {
                        Image(systemName: "chevron.left")
                    }
                    .accessibilityLabel(L10n.Common.back)
                }
            }
            ToolbarItem(placement: .principal) {
                navigationBarRoomMeasurementPrincipal
            }
            ToolbarItem(placement: .navigationBarTrailing) {
                Button(action: {
                    NotificationCenter.default.post(name: NSNotification.Name("RecenterMeshCamera"), object: nil)
                }) {
                    Image(systemName: "viewfinder")
                }
                .disabled(isLoading)
            }
            ToolbarItem(placement: .navigationBarTrailing) {
                navigationBarFullVideoIdentificationsButton
            }
            if canOfferBrainArAssist, showingFurnitureFit {
                ToolbarItem(placement: .navigationBarTrailing) {
                    navigationBarARButton
                        .fixedSize(horizontal: true, vertical: true)
                }
            }
            ToolbarItem(placement: .navigationBarTrailing) {
                Button(action: {
                    showRoomNameInput = true
                }) {
                    Image(systemName: "square.and.arrow.down")
                }
                .disabled(isLoading || isSavingRoom)
                .accessibilityLabel(L10n.RoomViewer.saveRoom)
            }
        }
        .onAppear {
            // Do not load YOLOE eagerly here — keep manual room memory low until the user enables brain mode.
            if photoOrientation == .landscape {
                OrientationLockManager.shared.lockToLandscape()
            } else {
                OrientationLockManager.shared.lockToPortrait()
            }
        }
        .onChange(of: isLoading) { _, loading in
            if loading {
                cancelPinchHintTasks()
                cancelBrainHintTasks()
                cancelSnapshotHintTasks()
                cancelARSizingHintTasks()
                cancelRoomDimensionsHintTasks()
            } else {
                restartPinchGestureHint()
                restartBrainGestureHint()
                restartSnapshotGestureHint()
            }
        }
        .onChange(of: showingFurnitureFit) { _, isOn in
            if isOn {
                yoloeService.ensureModelLoaded()
                if canOfferBrainArAssist {
                    showARSizingHint(requiresBrain: false)
                }
            } else {
                dismissFullVideoFurnitureTapHint()
                cancelARSizingHintTasks()
                arSizingHintExplanationVisible = false
                brainArAssistedSizingEnabled = false
                furnitureFitSegmentationMode = .identifyOnly
                furnitureFitShowIdentifyLivePreview = true
                selectedFurnitureFitLabels = []
                detectedFurnitureHeightAR = nil
                showFurnitureDimensionsInput = false
                yoloeService.releaseResources()
            }
        }
        .onDisappear {
            cancelPinchHintTasks()
            cancelBrainHintTasks()
            cancelSnapshotHintTasks()
            cancelARSizingHintTasks()
            cancelRoomDimensionsHintTasks()
            dismissFullVideoFurnitureTapHint()
            brainArAssistedSizingEnabled = false
            yoloeService.releaseResources()
            OrientationLockManager.shared.unlock()
        }
        .alert(L10n.RoomViewer.saveRoom, isPresented: $showRoomNameInput) {
            TextField("", text: $roomName, prompt: Text(L10n.RoomViewer.roomName))
                .autocorrectionDisabled(true)
            Button(L10n.Common.cancel, role: .cancel) {
                roomName = ""
            }
            Button(L10n.Common.save) {
                requestGLBExport()
            }
            .disabled(roomName.isEmpty)
        }
        .alert(L10n.RoomViewer.roomSavedAlertTitle, isPresented: $showSaveAlert) {
            Button(L10n.Common.ok, role: .cancel) {}
        } message: {
            Text(saveAlertMessage)
        }
        .alert(L10n.RoomPreview.unsavedTitle, isPresented: $showDiscardUnsavedAlert) {
            Button(L10n.RoomPreview.stay, role: .cancel) {}
            Button(L10n.RoomPreview.leave, role: .destructive) {
                dismiss()
            }
        } message: {
            Text(L10n.RoomPreview.unsavedMessage)
        }
        .defersSystemGestures(on: .all)
        .disableBackSwipe()
    }

    // MARK: - Ruler + hint tasks (matches SharpRoomView)

    private var canPresentMeshRoomDimensionsAlert: Bool {
        !showRoomNameInput &&
            !isSavingRoom &&
            !showSaveAlert &&
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
        return VStack(spacing: 2) {
            Text(L10n.RoomViewer.roomMetersShort(calibratedRoomHeight))
                .font(.caption2)
                .foregroundColor(roomCalibrationScaleFactor == 1.0 ? .white : .green)
            Text(L10n.RoomViewer.furnitureMetersShort(realFurnitureHeight ?? displayHeight))
                .font(.caption.bold())
                .foregroundColor(realFurnitureHeight != nil ? .green : .white)
            if showTapHint {
                Text(L10n.RoomViewer.tapToCalibrate)
                    .font(.system(size: 9))
                    .foregroundColor(.gray)
            }
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(Color.black.opacity(0.6))
        .cornerRadius(6)
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

        let scaleFactor = realHeight / detectedHeight
        roomCalibrationScaleFactor = scaleFactor
        realFurnitureHeight = realHeight
        NotificationCenter.default.post(
            name: NSNotification.Name("MeshRoomScaleRoom"),
            object: nil,
            userInfo: ["factor": Double(scaleFactor)]
        )
        logDebug("📐 [Mesh calibration] Real height: \(realHeight)m, scale factor: \(scaleFactor)")
        inputFurnitureHeight = ""
        showFurnitureDimensionsInput = false
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
        HStack(spacing: 8) {
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

    private func toggleBrainArAssistedSizingOrShowHint() {
        guard showingFurnitureFit else { return }
        brainArAssistedSizingEnabled.toggle()
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
        }
        .buttonStyle(.plain)
        .disabled(isLoading)
        .accessibilityLabel(L10n.Settings.fullVideoWithIdentifications)
        .accessibilityHint(L10n.Settings.fullVideoWithIdentificationsDescription)
        .accessibilityAddTraits(showFullVideoWithIdentifications ? .isSelected : [])
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

    /// Pinch + tap gesture helpers in the top-trailing corner (same pattern as ``SharpRoomView``).
    private var topTrailingPinchTapAndSizingHintsOverlay: some View {
        ZStack(alignment: .topTrailing) {
            Color.clear
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .allowsHitTesting(false)
            VStack(alignment: .trailing, spacing: 6) {
                HStack(spacing: 8) {
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

                    Button(action: displayAllGestureHelpers) {
                        Image(systemName: "hand.tap.fill")
                            .font(.system(size: 16, weight: .medium))
                            .foregroundStyle(.white)
                            .frame(width: 40, height: 40)
                            .background(Circle().fill(Color.black.opacity(0.5)))
                    }
                    .buttonStyle(.plain)
                    .accessibilityLabel(L10n.RoomViewer.displayAllHelpers)
                }
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
                if canOfferBrainArAssist, arSizingHintExplanationVisible,
                   showingFurnitureFit || arSizingHintRequiresBrain {
                    Text(arSizingHintText)
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
        .zIndex(101)
        .onAppear { restartPinchGestureHint() }
        .onDisappear {
            cancelPinchHintTasks()
            cancelARSizingHintTasks()
        }
    }

    private var roomDimensionsHintOverlay: some View {
        ZStack(alignment: .top) {
            Color.clear
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .allowsHitTesting(false)
            VStack(spacing: 0) {
                if roomDimensionsHintVisible {
                    Text(meshRoomDimensionsHintText)
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
        .zIndex(105)
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

    private func onSnapshotHintIconTapped() {
        cancelSnapshotHintTasks()
        snapshotHintExplanationVisible.toggle()
        if snapshotHintExplanationVisible {
            scheduleSnapshotHintTextAutoHide(seconds: 3)
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

    private var brainHintAccessibilityLabel: String {
        L10n.RoomViewer.brainGestureHintExplanation + " " + L10n.RoomViewer.gestureHintToggleAccessibility
    }

    private var snapshotHintAccessibilityLabel: String {
        L10n.RoomViewer.snapshotGestureHintExplanation + " " + L10n.RoomViewer.gestureHintToggleAccessibility
    }

    private var arSizingHintText: String {
        arSizingHintRequiresBrain
            ? L10n.RoomViewer.arFurnitureSizingRequiresBrainHint
            : L10n.RoomViewer.arFurnitureSizingHint
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
        .onAppear { restartBrainGestureHint() }
        .onDisappear { cancelBrainHintTasks() }
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
        .onAppear { restartSnapshotGestureHint() }
        .onDisappear { cancelSnapshotHintTasks() }
    }

    private func displayAllGestureHelpers() {
        restartPinchGestureHint()
        restartBrainGestureHint()
        restartSnapshotGestureHint()
        showARSizingHint(requiresBrain: !showingFurnitureFit)
        roomDimensionsHintVisible = true
        scheduleRoomDimensionsHintAutoHide(seconds: 3)
    }

    private var brainButtonWithHintAbove: some View {
        VStack(alignment: .center, spacing: 6) {
            brainGestureHintColumn
            Button(action: {
                if showingFurnitureFit {
                    dismissFullVideoFurnitureTapHint()
                    showingFurnitureFit = false
                } else {
                    showFullVideoWithIdentifications = false
                    furnitureFitInitialSegmentationDone = false
                    furnitureFitSegmentationMode = .identifyOnly
                    furnitureFitShowIdentifyLivePreview = true
                    selectedFurnitureFitLabels = []
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

    /// D-pad cluster only (same notifications as ``SharpRoomView`` / GLB viewer).
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
        .zIndex(102)
    }

    // MARK: - Portrait Controls
    private var portraitControls: some View {
        VStack {
            Spacer()

            // Orientation label
            HStack(spacing: 6) {
                Image(systemName: "iphone")
                    .font(.caption)
                Text(NSLocalizedString("orientation.heldVertically", comment: ""))
                    .font(.caption2)
                Text("-")
                    .font(.caption2)
                Text(NSLocalizedString("orientation.portrait", comment: ""))
                    .font(.caption2)
                    .fontWeight(.medium)
            }
            .foregroundColor(.white.opacity(0.9))
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background(Color.black.opacity(0.5))
            .cornerRadius(8)
            .padding(.bottom, 12)

            // Bottom controls: brain + segment + snapshot (full-video toggle is in the nav bar next to recenter)
            HStack {
                VStack(spacing: 10) {
                    brainButtonWithHintAbove
                }
                .padding(.leading, 16)
                segmentButton
                    .padding(.leading, 10)

                Spacer()

                VStack(spacing: 10) {
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
        .zIndex(99998)
    }

    // MARK: - Landscape Controls
    private var landscapeControls: some View {
        VStack {
            Spacer()

            // Horizontal bottom bar
            HStack(spacing: 20) {
                VStack(spacing: 10) {
                    brainButtonWithHintAbove
                }
                segmentButton

                // Orientation label
                HStack(spacing: 6) {
                    Image(systemName: "iphone.landscape")
                        .font(.caption)
                    Text(NSLocalizedString("orientation.heldHorizontally", comment: ""))
                        .font(.caption2)
                    Text("-")
                        .font(.caption2)
                    Text(NSLocalizedString("orientation.landscape", comment: ""))
                        .font(.caption2)
                        .fontWeight(.medium)
                }
                .foregroundColor(.white.opacity(0.9))
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .background(Color.black.opacity(0.5))
                .cornerRadius(8)

                Spacer()

                VStack(spacing: 10) {
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
            .padding(.horizontal, 30)
            .padding(.bottom, 20)
        }
        .zIndex(99997)
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
                    showSaveAlert = true
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
                saveAlertMessage = L10n.RoomViewer.saveSuccess(roomName)
                // Post notification to dismiss the sheet and refresh home view
                NotificationCenter.default.post(name: NSNotification.Name("DismissPhotoRoomSheet"), object: nil)
            } else {
                saveAlertMessage = errorMessage ?? L10n.RoomViewer.meshSaveFailedGeneric
            }
            showSaveAlert = true
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

    // MARK: - YOLOE model loaded via YOLOEModelService (ODR)
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

        // Add message handler for JS -> Swift communication
        config.userContentController.add(context.coordinator, name: "meshViewer")

        let webView = WKWebView(frame: .zero, configuration: config)
        context.coordinator.webView = webView

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

        // Historical manual-room path: Three.js from jsDelivr + loadHTMLString (requires network).
        // This matches the pre–App Store review bundling flow (~Apr 2026) that users relied on.
        let html = generateMeshViewerHTML()
        logDebug("📄 [MeshViewer] loadHTMLString (CDN three.js) htmlBytes=\(html.utf8.count)")
        webView.loadHTMLString(html, baseURL: nil)

        return webView
    }

    func updateUIView(_ uiView: WKWebView, context: Context) {}

    func makeCoordinator() -> Coordinator {
        Coordinator(onLoaded: onLoaded, onGLBExported: onGLBExported)
    }

    class Coordinator: NSObject, WKScriptMessageHandler, UIGestureRecognizerDelegate {
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
            NotificationCenter.default.addObserver(
                self,
                selector: #selector(scaleMeshRoom(_:)),
                name: NSNotification.Name("MeshRoomScaleRoom"),
                object: nil
            )
        }

        deinit {
            NotificationCenter.default.removeObserver(self)
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

        @objc private func scaleMeshRoom(_ notification: Notification) {
            guard let factor = notification.userInfo?["factor"] as? Double, factor.isFinite, factor > 0 else { return }
            let js = "if (typeof window.scaleRoom === 'function') { window.scaleRoom(\(factor)); }"
            webView?.evaluateJavaScript(js, completionHandler: nil)
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
                DispatchQueue.main.async {
                    self.onLoaded()
                }
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
                    "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
                    "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"
                }
            }
            </script>
            <script type="module">
                import * as THREE from 'three';
                import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
                import { GLTFExporter } from 'three/addons/exporters/GLTFExporter.js';

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

                // Orbit controls - smooth touch interactions
                const controls = new OrbitControls(camera, renderer.domElement);
                controls.enableDamping = true;
                controls.dampingFactor = 0.05;  // Quick response
                controls.rotateSpeed = 3.0;     // Fast rotation for touch
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

                // Set initial camera position - back center of room looking at front wall
                const targetY = roomHeight * 0.5;
                controls.target.set(0, targetY, 0);
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
                    initialTarget.set(0, targetYScaled, 0);
                    initialCameraPos.set(0, targetYScaled, cameraZScaled);
                    camera.position.copy(initialCameraPos);
                    controls.target.copy(initialTarget);
                    controls.update();
                    console.log('[MeshViewer] Room scaled by factor:', factor);
                };

                // D-pad: same behavior as Sharp WebGL — walk on XZ, vertical on Y (not orbit).
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
                window.exportGLB = function() {
                    console.log('[MeshViewer] Starting GLB export...');

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
