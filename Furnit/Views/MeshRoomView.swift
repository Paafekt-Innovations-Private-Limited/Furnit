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
    @State private var furnitureFitInitialSegmentationDone = false
    @State private var brainArAssistedSizingEnabled = false
    @ObservedObject private var yoloeService = YOLOEModelService.shared

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
    /// Ruler tap: show W×H×D chip below top safe area (matches Sharp room dimensions hint).
    @State private var roomDimensionsHintVisible = false
    @State private var roomDimensionsHintHideTask: Task<Void, Never>?
    /// Pinch-zoom hint (top-left with D-pad) — same as ``SharpRoomView``.
    @State private var pinchHintExplanationVisible = false
    @State private var pinchHintHideTextTask: Task<Void, Never>?

    private var canOfferBrainArAssist: Bool {
        QualitySettings.supportsFurnitureFitARAssisted &&
            AppStateManager.shared.qualitySettings.furnitureFitARDepthCompanionRuntimeActive
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
                    roomWidthMeters: roomWidth,
                    roomHeightMeters: roomHeight,
                    onFurnitureSizeEstimated: { _ in },
                    suppressStartupProgress: furnitureFitInitialSegmentationDone,
                    onFirstSegmentationComplete: { furnitureFitInitialSegmentationDone = true },
                    arAssistedSizingEnabled: brainArAssistedSizingEnabled && canOfferBrainArAssist
                )
                .ignoresSafeArea()
                .zIndex(100)
            }

            // Room dimensions chip (ruler) — same placement idea as ``SharpRoomView``.
            roomDimensionsHintOverlay

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
            if savedRoomModel == nil {
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
            ToolbarItem(placement: .principal) {
                navigationBarRoomMeasurementPrincipal
            }
            ToolbarItemGroup(placement: .navigationBarTrailing) {
                // Save Room button
                Button(action: {
                    showRoomNameInput = true
                }) {
                    Image(systemName: "square.and.arrow.down")
                }
                .disabled(isLoading || isSavingRoom)

                // Recenter button
                Button(action: {
                    NotificationCenter.default.post(name: NSNotification.Name("RecenterMeshCamera"), object: nil)
                }) {
                    Image(systemName: "viewfinder")
                }
                .disabled(isLoading)
            }
        }
        .onAppear {
            yoloeService.ensureModelLoaded()
            // Lock orientation based on photo orientation
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
            } else {
                brainArAssistedSizingEnabled = false
                yoloeService.releaseResources()
            }
        }
        .onDisappear {
            cancelPinchHintTasks()
            cancelBrainHintTasks()
            cancelSnapshotHintTasks()
            cancelARSizingHintTasks()
            cancelRoomDimensionsHintTasks()
            OrientationLockManager.shared.unlock()
        }
        .alert("Save Room", isPresented: $showRoomNameInput) {
            TextField("Room name", text: $roomName)
            Button("Cancel", role: .cancel) {
                roomName = ""
            }
            Button("Save") {
                requestGLBExport()
            }
            .disabled(roomName.isEmpty)
        }
        .alert("Room Saved", isPresented: $showSaveAlert) {
            Button("OK", role: .cancel) {}
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
            width: roomWidth,
            height: roomHeight,
            depth: roomDepth
        )
    }

    private var navigationBarRoomMeasurementPrincipal: some View {
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
        .zIndex(13)
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

    private func restartARSizingHint() {
        cancelARSizingHintTasks()
        arSizingHintExplanationVisible = true
        scheduleARSizingHintTextAutoHide(seconds: 3)
    }

    private func onARSizingHintIconTapped() {
        cancelARSizingHintTasks()
        arSizingHintExplanationVisible.toggle()
        if arSizingHintExplanationVisible {
            scheduleARSizingHintTextAutoHide(seconds: 3)
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

    private var arSizingHintAccessibilityLabel: String {
        arSizingHintText + " " + L10n.RoomViewer.gestureHintToggleAccessibility
    }

    private var arSizingHintText: String {
        showingFurnitureFit
            ? L10n.RoomViewer.arFurnitureSizingHint
            : L10n.RoomViewer.arFurnitureSizingRequiresBrainHint
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
            Button(action: onSnapshotHintIconTapped) {
                Image(systemName: "hand.tap.fill")
                    .symbolRenderingMode(.hierarchical)
                    .font(.system(size: 18, weight: .medium))
                    .foregroundStyle(.white)
                    .frame(width: 40, height: 40)
                    .background(Circle().fill(Color.black.opacity(0.5)))
            }
            .buttonStyle(.plain)
            .accessibilityLabel(snapshotHintAccessibilityLabel)
        }
        .onAppear { restartSnapshotGestureHint() }
        .onDisappear { cancelSnapshotHintTasks() }
    }

    private var arSizingGestureHintColumn: some View {
        VStack(alignment: .center, spacing: 6) {
            if arSizingHintExplanationVisible {
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
            Button(action: onARSizingHintIconTapped) {
                Image(systemName: "hand.tap.fill")
                    .symbolRenderingMode(.hierarchical)
                    .font(.system(size: 18, weight: .medium))
                    .foregroundStyle(.white)
                    .frame(width: 40, height: 40)
                    .background(Circle().fill(Color.black.opacity(0.5)))
            }
            .buttonStyle(.plain)
            .accessibilityLabel(arSizingHintAccessibilityLabel)
        }
        .onAppear { restartARSizingHint() }
        .onDisappear { cancelARSizingHintTasks() }
    }

    private var brainButtonWithHintAbove: some View {
        VStack(alignment: .center, spacing: 6) {
            brainGestureHintColumn
            Button(action: {
                if showingFurnitureFit {
                    showingFurnitureFit = false
                } else {
                    furnitureFitInitialSegmentationDone = false
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

    private var pinchGestureHintOverlay: some View {
        VStack(alignment: .leading, spacing: 6) {
            if pinchHintExplanationVisible {
                Text(L10n.RoomViewer.pinchGestureHintExplanation)
                    .font(.caption2)
                    .foregroundColor(.white)
                    .multilineTextAlignment(.leading)
                    .fixedSize(horizontal: false, vertical: true)
                    .frame(maxWidth: 220, alignment: .leading)
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
        }
        .onAppear { restartPinchGestureHint() }
        .onDisappear { cancelPinchHintTasks() }
        .zIndex(12)
    }

    private var cameraButtonsOverlay: some View {
        ZStack(alignment: .topLeading) {
            Color.clear
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .allowsHitTesting(false)
            VStack(alignment: .leading, spacing: 10) {
                cameraDPadCluster
                pinchGestureHintOverlay
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.leading, 12)
                    .padding(.top, 12)
                if photoOrientation == .landscape {
                    Spacer(minLength: 0)
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        }
        .zIndex(18)
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

            // Bottom controls: brain + snapshot with tap helpers (SharpRoomView parity)
            HStack {
                VStack(spacing: 10) {
                    brainButtonWithHintAbove

                    if canOfferBrainArAssist {
                        VStack(spacing: 6) {
                            Button(action: {
                                if showingFurnitureFit {
                                    brainArAssistedSizingEnabled.toggle()
                                } else {
                                    arSizingHintExplanationVisible = true
                                    scheduleARSizingHintTextAutoHide(seconds: 3)
                                }
                            }) {
                                Image(systemName: "arrow.up.left.and.arrow.down.right")
                                    .font(.system(size: 20, weight: .bold))
                                    .foregroundColor(.white)
                                    .frame(width: 40, height: 40)
                                    .background(
                                        Circle().fill(
                                            brainArAssistedSizingEnabled
                                                ? Color.green.opacity(0.9)
                                                : Color.black.opacity(0.45)
                                        )
                                    )
                            }
                            .disabled(isLoading || !showingFurnitureFit)
                            .accessibilityLabel(
                                brainArAssistedSizingEnabled ? L10n.RoomViewer.arSizingDisable : L10n.RoomViewer.arSizingEnable
                            )
                            arSizingGestureHintColumn
                        }
                    }
                }
                .padding(.leading, 16)

                Spacer()

                snapshotButtonWithHintAbove
                    .padding(.trailing, 16)
            }
            .padding(.bottom, 20)
        }
    }

    // MARK: - Landscape Controls
    private var landscapeControls: some View {
        VStack {
            Spacer()

            // Horizontal bottom bar
            HStack(spacing: 20) {
                VStack(spacing: 10) {
                    brainButtonWithHintAbove

                    if canOfferBrainArAssist {
                        VStack(spacing: 6) {
                            Button(action: {
                                if showingFurnitureFit {
                                    brainArAssistedSizingEnabled.toggle()
                                } else {
                                    arSizingHintExplanationVisible = true
                                    scheduleARSizingHintTextAutoHide(seconds: 3)
                                }
                            }) {
                                Image(systemName: "arrow.up.left.and.arrow.down.right")
                                    .font(.system(size: 20, weight: .bold))
                                    .foregroundColor(.white)
                                    .frame(width: 40, height: 40)
                                    .background(
                                        Circle().fill(
                                            brainArAssistedSizingEnabled
                                                ? Color.green.opacity(0.9)
                                                : Color.black.opacity(0.45)
                                        )
                                    )
                            }
                            .disabled(isLoading || !showingFurnitureFit)
                            .accessibilityLabel(
                                brainArAssistedSizingEnabled ? L10n.RoomViewer.arSizingDisable : L10n.RoomViewer.arSizingEnable
                            )
                            arSizingGestureHintColumn
                        }
                    }
                }

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

                snapshotButtonWithHintAbove
            }
            .padding(.horizontal, 30)
            .padding(.bottom, 20)
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
                    saveAlertMessage = "Failed to export 3D model"
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
                saveAlertMessage = "Room '\(roomName)' saved successfully!"
                // Post notification to dismiss the sheet and refresh home view
                NotificationCenter.default.post(name: NSNotification.Name("DismissPhotoRoomSheet"), object: nil)
            } else {
                saveAlertMessage = errorMessage ?? "Failed to save room"
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

        // Load the HTML with Three.js
        let html = generateMeshViewerHTML()
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
        }

        deinit {
            NotificationCenter.default.removeObserver(self)
        }

        @objc private func recenterCamera() {
            webView?.evaluateJavaScript("if (typeof recenterCamera === 'function') recenterCamera();", completionHandler: nil)
        }

        @objc private func nudgeMeshCameraLeft() {
            webView?.evaluateJavaScript("if (typeof meshCameraNudge === 'function') meshCameraNudge('left');", completionHandler: nil)
        }

        @objc private func nudgeMeshCameraRight() {
            webView?.evaluateJavaScript("if (typeof meshCameraNudge === 'function') meshCameraNudge('right');", completionHandler: nil)
        }

        @objc private func nudgeMeshCameraUp() {
            webView?.evaluateJavaScript("if (typeof meshCameraNudge === 'function') meshCameraNudge('up');", completionHandler: nil)
        }

        @objc private func nudgeMeshCameraDown() {
            webView?.evaluateJavaScript("if (typeof meshCameraNudge === 'function') meshCameraNudge('down');", completionHandler: nil)
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

                // Camera - start inside the room looking at front wall
                const camera = new THREE.PerspectiveCamera(70, window.innerWidth / window.innerHeight, 0.1, 100);

                // Renderer
                const renderer = new THREE.WebGLRenderer({ antialias: true });
                renderer.setSize(window.innerWidth, window.innerHeight);
                renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
                renderer.outputColorSpace = THREE.SRGBColorSpace;
                document.body.appendChild(renderer.domElement);

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
                const initialCameraPos = camera.position.clone();
                const initialTarget = controls.target.clone();

                // Recenter function
                window.recenterCamera = function() {
                    camera.position.copy(initialCameraPos);
                    controls.target.copy(initialTarget);
                    controls.update();
                    console.log('[MeshViewer] Camera recentered');
                };

                window.meshCameraNudge = function(direction) {
                    const stepH = 0.08;
                    const stepV = 0.06;
                    const offset = new THREE.Vector3();
                    offset.copy(camera.position).sub(controls.target);
                    const spherical = new THREE.Spherical();
                    spherical.setFromVector3(offset);
                    if (direction === 'left') spherical.theta += stepH;
                    else if (direction === 'right') spherical.theta -= stepH;
                    else if (direction === 'up') spherical.phi -= stepV;
                    else if (direction === 'down') spherical.phi += stepV;
                    spherical.phi = Math.max(controls.minPolarAngle, Math.min(controls.maxPolarAngle, spherical.phi));
                    offset.setFromSpherical(spherical);
                    camera.position.copy(controls.target).add(offset);
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
                    buildRoomWithTextures(img);
                };
                img.onerror = function() {
                    console.error('[MeshViewer] Failed to load image');
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

                    // Notify Swift that we're loaded
                    window.webkit.messageHandlers.meshViewer.postMessage({ event: 'loaded' });
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

                // Handle resize
                window.addEventListener('resize', () => {
                    camera.aspect = window.innerWidth / window.innerHeight;
                    camera.updateProjectionMatrix();
                    renderer.setSize(window.innerWidth, window.innerHeight);
                });

                console.log('[MeshViewer] Initialized');

            </script>
        </body>
        </html>
        """
    }
}
