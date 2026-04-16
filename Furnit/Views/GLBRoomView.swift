import SwiftUI
import WebKit
import UIKit
import CoreML

// MARK: - Navigation Back Swipe Disabler
/// Wraps content and disables the navigation back swipe gesture
struct NavigationBackSwipeDisabler<Content: View>: UIViewControllerRepresentable {
    let content: Content

    init(@ViewBuilder content: () -> Content) {
        self.content = content()
    }

    func makeUIViewController(context: Context) -> UIHostingController<Content> {
        let hostingController = BackSwipeDisabledHostingController(rootView: content)
        return hostingController
    }

    func updateUIViewController(_ uiViewController: UIHostingController<Content>, context: Context) {
        uiViewController.rootView = content
    }
}

class BackSwipeDisabledHostingController<Content: View>: UIHostingController<Content> {
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        navigationController?.interactivePopGestureRecognizer?.isEnabled = false
    }

    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)
        navigationController?.interactivePopGestureRecognizer?.isEnabled = false
    }

    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        // Re-enable for other views
        navigationController?.interactivePopGestureRecognizer?.isEnabled = true
    }
}

// MARK: - View Modifier to Disable Back Swipe
struct DisableBackSwipeModifier: ViewModifier {
    func body(content: Content) -> some View {
        content.background(BackSwipeDisablerView())
    }
}

struct BackSwipeDisablerView: UIViewRepresentable {
    func makeUIView(context: Context) -> UIView {
        let view = BackSwipeBlockerUIView()
        return view
    }

    func updateUIView(_ uiView: UIView, context: Context) {}
}

class BackSwipeBlockerUIView: UIView {
    private var observer: NSObjectProtocol?

    override func didMoveToWindow() {
        super.didMoveToWindow()
        disableBackSwipe()

        // Also observe when the scene becomes active in case navigation controller wasn't ready
        observer = NotificationCenter.default.addObserver(
            forName: UIScene.didActivateNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            self?.disableBackSwipe()
        }
    }

    override func willMove(toWindow newWindow: UIWindow?) {
        super.willMove(toWindow: newWindow)
        if newWindow == nil {
            // Re-enable when leaving
            enableBackSwipe()
            if let observer = observer {
                NotificationCenter.default.removeObserver(observer)
            }
        }
    }

    private func disableBackSwipe() {
        DispatchQueue.main.async { [weak self] in
            self?.findAndDisableBackSwipe()
        }
    }

    private func enableBackSwipe() {
        var responder: UIResponder? = self
        while responder != nil {
            if let nav = responder as? UINavigationController {
                nav.interactivePopGestureRecognizer?.isEnabled = true
                return
            }
            if let vc = responder as? UIViewController, let nav = vc.navigationController {
                nav.interactivePopGestureRecognizer?.isEnabled = true
                return
            }
            responder = responder?.next
        }
    }

    private func findAndDisableBackSwipe() {
        // Method 1: Walk responder chain
        var responder: UIResponder? = self
        while responder != nil {
            if let nav = responder as? UINavigationController {
                nav.interactivePopGestureRecognizer?.isEnabled = false
                return
            }
            if let vc = responder as? UIViewController, let nav = vc.navigationController {
                nav.interactivePopGestureRecognizer?.isEnabled = false
                return
            }
            responder = responder?.next
        }

        // Method 2: Search through all window scenes
        let scenes = UIApplication.shared.connectedScenes.compactMap { $0 as? UIWindowScene }
        for scene in scenes {
            for window in scene.windows {
                if let rootVC = window.rootViewController {
                    findAndDisableInViewController(rootVC)
                }
            }
        }
    }

    private func findAndDisableInViewController(_ vc: UIViewController) {
        if let nav = vc as? UINavigationController {
            nav.interactivePopGestureRecognizer?.isEnabled = false
        }
        if let nav = vc.navigationController {
            nav.interactivePopGestureRecognizer?.isEnabled = false
        }
        for child in vc.children {
            findAndDisableInViewController(child)
        }
        if let presented = vc.presentedViewController {
            findAndDisableInViewController(presented)
        }
    }
}

extension View {
    func disableBackSwipe() -> some View {
        self.modifier(DisableBackSwipeModifier())
    }

    @ViewBuilder
    func disableBackSwipeIf(_ condition: Bool) -> some View {
        if condition {
            self.disableBackSwipe()
        } else {
            self
        }
    }
}

// MARK: - UIView Extension to Find Navigation Controller
extension UIView {
    func findNavigationController() -> UINavigationController? {
        var responder: UIResponder? = self
        while responder != nil {
            if let nav = responder as? UINavigationController {
                return nav
            }
            if let vc = responder as? UIViewController, let nav = vc.navigationController {
                return nav
            }
            responder = responder?.next
        }
        return nil
    }
}

/// WebGL-based GLB/GLTF room viewer - loads and renders GLB 3D models using Three.js
struct GLBRoomView: View {
    let glbURL: URL
    let photoOrientation: PhotoOrientation
    let roomWidth: Float?
    let roomHeight: Float?
    let roomDepth: Float?
    /// When opening a saved GLB from Home (YOLO ratio calibration + FurnitureFit).
    var savedRoomModel: USDZModel? = nil

    @Environment(\.dismiss) private var dismiss
    @EnvironmentObject var authManager: AuthenticationManager

    @State private var isLoading = true
    @State private var error: String? = nil
    @State private var showingFurnitureFit = false
    @State private var furnitureFitSegmentationMode: FurnitureFitSegmentationMode = .identifyOnly
    @State private var furnitureFitShowIdentifyLivePreview = true
    @State private var selectedFurnitureFitLabels: [String] = []
    @State private var furnitureFitInitialSegmentationDone = false
    @State private var brainArAssistedSizingEnabled = false
    @ObservedObject private var yoloeService = YOLOEModelService.shared
    @ObservedObject private var appState = AppStateManager.shared
    @StateObject private var savedRoomsModelManager = USDZModelManager()
    @State private var showDiscardUnsavedAlert = false

    /// Brain / snapshot tap helpers — same as ``MeshRoomView`` (parity for GLB opened from Home).
    @State private var brainHintExplanationVisible = false
    @State private var brainHintHideTextTask: Task<Void, Never>?
    @State private var snapshotHintExplanationVisible = false
    @State private var snapshotHintHideTextTask: Task<Void, Never>?
    @State private var arSizingHintExplanationVisible = false
    @State private var arSizingHintHideTextTask: Task<Void, Never>?
    @State private var arSizingHintRequiresBrain = false
    @State private var roomDimensionsHintVisible = false
    @State private var roomDimensionsHintHideTask: Task<Void, Never>?
    @AppStorage("furnitureFit.showFullVideoWithIdentifications") private var showFullVideoWithIdentifications: Bool = false
    @State private var fullVideoFurnitureTapHintVisible = false
    @State private var tapHintColorIndex: Int = 0
    private let tapHintColors: [Color] = [.yellow, .cyan, .orange, .green, .pink]
    @State private var tapHintColorTimer: Timer?
    /// Pinch-zoom hint (top-left with D-pad) — same as ``SharpRoomView`` / ``MeshRoomView``.
    @State private var pinchHintExplanationVisible = false
    @State private var pinchHintHideTextTask: Task<Void, Never>?

    private var canOfferBrainArAssist: Bool {
        QualitySettings.supportsFurnitureFitARAssisted &&
            appState.qualitySettings.furnitureFitARDepthCompanionRuntimeActive
    }

    private var canSegmentSelectedFurniture: Bool {
        showingFurnitureFit && !selectedFurnitureFitLabels.isEmpty
    }

    var body: some View {
        ZStack {
            // WebGL GLB viewer - OrbitControls in Three.js handles touch directly
            GLBWebGLView(
                glbURL: glbURL,
                photoOrientation: photoOrientation,
                onLoaded: {
                    isLoading = false
                },
                onError: { errorMessage in
                    error = errorMessage
                    isLoading = false
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
                    roomWidthMeters: roomWidth ?? 4.0,
                    roomHeightMeters: roomHeight ?? 3.0,
                    onFurnitureSizeEstimated: { _ in },
                    suppressStartupProgress: furnitureFitInitialSegmentationDone,
                    onFirstSegmentationComplete: { furnitureFitInitialSegmentationDone = true },
                    arAssistedSizingEnabled: brainArAssistedSizingEnabled && canOfferBrainArAssist,
                    segmentationMode: furnitureFitSegmentationMode,
                    onSelectedClassLabelsChanged: { labels in
                        selectedFurnitureFitLabels = labels
                    },
                    showIdentifyLivePreview: furnitureFitShowIdentifyLivePreview
                )
                .ignoresSafeArea(edges: [.bottom, .leading, .trailing])
                .zIndex(100)
            }

            roomDimensionsHintOverlay
            fullVideoFurnitureTapHintOverlay

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
                        showDiscardUnsavedAlert = true
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
                    NotificationCenter.default.post(name: NSNotification.Name("RecenterGLBCamera"), object: nil)
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
        }
        .onAppear {
            // Preload YOLOE when the room opens (async; not at app startup).
            yoloeService.ensureModelLoaded()
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
        .alert(L10n.RoomPreview.unsavedTitle, isPresented: $showDiscardUnsavedAlert) {
            Button(L10n.RoomPreview.stay, role: .cancel) {}
            Button(L10n.RoomPreview.leave, role: .destructive) {
                dismiss()
            }
        } message: {
            Text(L10n.RoomPreview.unsavedMessage)
        }
        .defersSystemGestures(on: .all)
        .disableBackSwipeIf(savedRoomModel == nil)
    }

    // MARK: - Ruler + tap helpers (MeshRoomView parity for saved GLB from Home)

    private var canPresentGLBRoomDimensionsAlert: Bool {
        !showDiscardUnsavedAlert
    }

    private var glbRoomDimensionsHintText: String {
        if let w = roomWidth, let h = roomHeight, let d = roomDepth,
           w > 0.05, h > 0.05, d > 0.05, w.isFinite, h.isFinite, d.isFinite {
            return L10n.RoomViewer.roomDimensionsWHDManualChip(width: w, height: h, depth: d)
        }
        if let w = roomWidth, let h = roomHeight, w > 0.05, h > 0.05, w.isFinite, h.isFinite {
            return L10n.RoomViewer.roomDimensionsWHManualChip(width: w, height: h)
        }
        return "3D Room"
    }

    private var navigationBarRoomMeasurementPrincipal: some View {
        HStack(spacing: 8) {
                Button {
                    guard canPresentGLBRoomDimensionsAlert else { return }
                    onGLBRoomDimensionsRulerTapped()
                } label: {
                    Image(systemName: "ruler.fill")
                        .symbolRenderingMode(.hierarchical)
                        .font(.system(size: 18, weight: .medium))
                        .foregroundStyle(.primary)
                }
                .buttonStyle(.plain)
                .disabled(!canPresentGLBRoomDimensionsAlert || isLoading)
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

    /// Pinch + tap gesture helpers in the top-trailing corner (same pattern as ``SharpRoomView`` / ``MeshRoomView``).
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
                    Text(glbRoomDimensionsHintText)
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
        .zIndex(14)
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

    private func cancelARSizingHintTasks() {
        arSizingHintHideTextTask?.cancel()
        arSizingHintHideTextTask = nil
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

    private func onGLBRoomDimensionsRulerTapped() {
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

    private var brainButtonWithHintAbove: some View {
        VStack(alignment: .center, spacing: 6) {
            brainGestureHintColumn
            Button(action: {
                if showingFurnitureFit {
                    dismissFullVideoFurnitureTapHint()
                    showingFurnitureFit = false
                } else {
                    furnitureFitInitialSegmentationDone = false
                    furnitureFitSegmentationMode = .identifyOnly
                    furnitureFitShowIdentifyLivePreview = true
                    selectedFurnitureFitLabels = []
                    showingFurnitureFit = true
                    presentFullVideoFurnitureTapHintIfNeeded()
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

    private func displayAllGestureHelpers() {
        restartPinchGestureHint()
        restartBrainGestureHint()
        restartSnapshotGestureHint()
        showARSizingHint(requiresBrain: !showingFurnitureFit)
        roomDimensionsHintVisible = true
        scheduleRoomDimensionsHintAutoHide(seconds: 3)
    }

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
        .zIndex(18)
    }

    // MARK: - YOLOE model loaded via YOLOEModelService (ODR)

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

            HStack {
                brainButtonWithHintAbove
                    .padding(.leading, 16)
                segmentButton
                    .padding(.leading, 10)

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

            HStack(spacing: 20) {
                brainButtonWithHintAbove
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

                snapshotButtonWithHintAbove
            }
            .padding(.horizontal, 30)
            .padding(.bottom, 20)
        }
    }

    // MARK: - Take Screenshot
    private func takeScreenshot() {
        logDebug("📸 [GLBRoomView] Taking screenshot...")

        // Capture the window
        let scenes = UIApplication.shared.connectedScenes.compactMap { $0 as? UIWindowScene }
        let windows = scenes.flatMap { $0.windows }
        guard let window = windows.first(where: { $0.isKeyWindow }) ?? windows.first else {
            logDebug("❌ [GLBRoomView] No window found")
            return
        }

        let format = UIGraphicsImageRendererFormat()
        format.scale = window.traitCollection.displayScale
        let renderer = UIGraphicsImageRenderer(bounds: window.bounds, format: format)
        let image = renderer.image { _ in
            window.drawHierarchy(in: window.bounds, afterScreenUpdates: true)
        }

        logDebug("📸 [GLBRoomView] Screenshot captured, saving to Photos...")

        // Save to photos
        UIImageWriteToSavedPhotosAlbum(image, nil, nil, nil)
        logDebug("✅ [GLBRoomView] Screenshot saved to Photos")
    }
}

// MARK: - WebGL GLB View (UIViewRepresentable)
struct GLBWebGLView: UIViewRepresentable {
    let glbURL: URL
    let photoOrientation: PhotoOrientation
    let onLoaded: () -> Void
    let onError: (String) -> Void

    func makeUIView(context: Context) -> WKWebView {
        let config = WKWebViewConfiguration()
        config.allowsInlineMediaPlayback = true

        // Add message handler for JS -> Swift communication
        config.userContentController.add(context.coordinator, name: "glbViewer")

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

        if let glbData = try? Data(contentsOf: glbURL) {
            let html = generateGLBViewerHTML(glbData: glbData)
            logDebug("📄 [GLBViewer] loadHTMLString (CDN three.js) htmlBytes=\(html.utf8.count)")
            webView.loadHTMLString(html, baseURL: nil)
        } else {
            logDebug("❌ [GLBRoomView] Failed to load GLB file: \(glbURL.path)")
            DispatchQueue.main.async {
                onError("Failed to load 3D model file")
            }
        }

        return webView
    }

    func updateUIView(_ uiView: WKWebView, context: Context) {}

    func makeCoordinator() -> Coordinator {
        Coordinator(onLoaded: onLoaded, onError: onError)
    }

    class Coordinator: NSObject, WKScriptMessageHandler, UIGestureRecognizerDelegate {
        let onLoaded: () -> Void
        let onError: (String) -> Void
        weak var webView: WKWebView?

        init(onLoaded: @escaping () -> Void, onError: @escaping (String) -> Void) {
            self.onLoaded = onLoaded
            self.onError = onError
            super.init()

            // Listen for recenter notification
            NotificationCenter.default.addObserver(
                self,
                selector: #selector(recenterCamera),
                name: NSNotification.Name("RecenterGLBCamera"),
                object: nil
            )
            NotificationCenter.default.addObserver(
                self,
                selector: #selector(nudgeGLBCameraLeft),
                name: NSNotification.Name("WebGLCameraMoveLeft"),
                object: nil
            )
            NotificationCenter.default.addObserver(
                self,
                selector: #selector(nudgeGLBCameraRight),
                name: NSNotification.Name("WebGLCameraMoveRight"),
                object: nil
            )
            NotificationCenter.default.addObserver(
                self,
                selector: #selector(nudgeGLBCameraUp),
                name: NSNotification.Name("WebGLCameraMoveUp"),
                object: nil
            )
            NotificationCenter.default.addObserver(
                self,
                selector: #selector(nudgeGLBCameraDown),
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

        @objc private func nudgeGLBCameraLeft() {
            webView?.evaluateJavaScript("if (typeof moveCamera === 'function') moveCamera(-8, 0);", completionHandler: nil)
        }

        @objc private func nudgeGLBCameraRight() {
            webView?.evaluateJavaScript("if (typeof moveCamera === 'function') moveCamera(8, 0);", completionHandler: nil)
        }

        @objc private func nudgeGLBCameraUp() {
            webView?.evaluateJavaScript("if (typeof moveCameraUp === 'function') moveCameraUp(0.2);", completionHandler: nil)
        }

        @objc private func nudgeGLBCameraDown() {
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
                DispatchQueue.main.async {
                    self.onLoaded()
                }
            case "error":
                let errorMessage = body["message"] as? String ?? "Unknown error"
                DispatchQueue.main.async {
                    self.onError(errorMessage)
                }
            default:
                break
            }
        }
    }

    // MARK: - Generate Three.js HTML for GLB Loading
    private func generateGLBViewerHTML(glbData: Data) -> String {
        let base64GLB = glbData.base64EncodedString()
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
                import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

                const isPortrait = \(isPortrait ? "true" : "false");

                console.log('[GLBViewer] Starting...');

                let roomBoundsForClamping = null;

                // Scene setup
                const scene = new THREE.Scene();
                scene.background = new THREE.Color(0x808080);

                // Camera
                const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
                camera.position.set(0, 2, 5);

                // Renderer
                const renderer = new THREE.WebGLRenderer({ antialias: true });
                renderer.setSize(window.innerWidth, window.innerHeight);
                renderer.setPixelRatio(window.devicePixelRatio);
                document.body.appendChild(renderer.domElement);

                // Orbit controls
                const controls = new OrbitControls(camera, renderer.domElement);
                controls.enableDamping = true;
                controls.dampingFactor = 0.05;  // Quick response
                controls.rotateSpeed = 3.0;     // Fast rotation for touch
                controls.zoomSpeed = 2.5;       // Fast zoom
                controls.enableZoom = true;
                controls.enablePan = false;
                controls.minDistance = 0.5;
                controls.maxDistance = 20;

                // D-pad / Sharp parity: walk on XZ, vertical Y (same as embedded Sharp WebGL).
                window.moveCamera = function(dx, dy) {
                    const moveSpeed = 0.03;
                    let newX = camera.position.x + dx * moveSpeed;
                    let newZ = camera.position.z + dy * moveSpeed;
                    if (roomBoundsForClamping) {
                        const marginSide = 0.05;
                        const marginBack = 0.02;
                        newX = Math.max(roomBoundsForClamping.minX + marginSide,
                               Math.min(roomBoundsForClamping.maxX - marginSide, newX));
                        newZ = Math.max(roomBoundsForClamping.minZ + marginSide,
                               Math.min(roomBoundsForClamping.maxZ - marginBack, newZ));
                    }
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
                    if (roomBoundsForClamping) {
                        const m = 0.05;
                        camera.position.y = Math.max(roomBoundsForClamping.minY + m, Math.min(roomBoundsForClamping.maxY - m, camera.position.y));
                        controls.target.y = Math.max(roomBoundsForClamping.minY + m, Math.min(roomBoundsForClamping.maxY - m, controls.target.y));
                    }
                    controls.update();
                };

                // Listen for orbit commands from Swift (touch drag)
                window.orbitCamera = function(deltaX, deltaY) {
                    const spherical = new THREE.Spherical();
                    const offset = new THREE.Vector3();
                    offset.copy(camera.position).sub(controls.target);
                    spherical.setFromVector3(offset);

                    spherical.theta -= deltaX * 0.012;  // Increased for faster response
                    spherical.phi -= deltaY * 0.012;
                    spherical.phi = Math.max(0.1, Math.min(Math.PI - 0.1, spherical.phi));

                    offset.setFromSpherical(spherical);
                    camera.position.copy(controls.target).add(offset);
                    camera.lookAt(controls.target);
                };

                // Lighting
                const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
                scene.add(ambientLight);

                const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
                directionalLight.position.set(5, 10, 5);
                scene.add(directionalLight);

                // Load GLB from base64
                const base64GLB = '\(base64GLB)';

                try {
                    // Decode base64 to ArrayBuffer
                    const binaryString = atob(base64GLB);
                    const bytes = new Uint8Array(binaryString.length);
                    for (let i = 0; i < binaryString.length; i++) {
                        bytes[i] = binaryString.charCodeAt(i);
                    }
                    const arrayBuffer = bytes.buffer;

                    console.log('[GLBViewer] GLB data size:', arrayBuffer.byteLength);

                    // Load with GLTFLoader
                    const loader = new GLTFLoader();
                    loader.parse(arrayBuffer, '', function(gltf) {
                        console.log('[GLBViewer] GLB loaded successfully');

                        const model = gltf.scene;
                        scene.add(model);

                        // Get model bounds
                        const box = new THREE.Box3().setFromObject(model);
                        const center = box.getCenter(new THREE.Vector3());
                        const size = box.getSize(new THREE.Vector3());

                        console.log('[GLBViewer] Model bounds - center:', center, 'size:', size);

                        // Center the model horizontally but keep floor at y=0
                        model.position.x = -center.x;
                        model.position.z = -center.z;
                        // Adjust Y so floor is at 0
                        model.position.y = -box.min.y;

                        const boxWorld = new THREE.Box3().setFromObject(model);
                        roomBoundsForClamping = {
                            minX: boxWorld.min.x + 0.05,
                            maxX: boxWorld.max.x - 0.05,
                            minY: boxWorld.min.y + 0.05,
                            maxY: boxWorld.max.y - 0.05,
                            minZ: boxWorld.min.z + 0.05,
                            maxZ: boxWorld.max.z - 0.02
                        };

                        // Position camera inside the room looking at front wall
                        // After centering: room spans -size/2 to +size/2 in X and Z
                        // Front wall is at Z = -roomDepth/2, back (open) is at Z = +roomDepth/2
                        const roomWidth = size.x;
                        const roomHeight = size.y;
                        const roomDepth = size.z;

                        // Camera at back part of room, eye level, looking at front wall
                        const eyeHeight = roomHeight * 0.5;
                        // Position camera 30% into the room from back (+Z side)
                        const cameraZ = roomDepth * 0.2;
                        // Look at center-front of room (towards -Z)
                        const targetZ = -roomDepth * 0.2;

                        camera.position.set(0, eyeHeight, cameraZ);
                        controls.target.set(0, eyeHeight, targetZ);
                        controls.update();

                        // Save initial camera state for recenter
                        const initialCameraPos = camera.position.clone();
                        const initialTarget = controls.target.clone();

                        // Recenter function
                        window.recenterCamera = function() {
                            camera.position.copy(initialCameraPos);
                            controls.target.copy(initialTarget);
                            controls.update();
                            console.log('[GLBViewer] Camera recentered');
                        };

                        console.log('[GLBViewer] Room size:', roomWidth.toFixed(2), 'x', roomHeight.toFixed(2), 'x', roomDepth.toFixed(2));
                        console.log('[GLBViewer] Camera at (0,', eyeHeight.toFixed(2), ',', cameraZ.toFixed(2), '), looking at (0,', eyeHeight.toFixed(2), ',', targetZ.toFixed(2), ')');

                        // Notify Swift that we're loaded
                        window.webkit.messageHandlers.glbViewer.postMessage({ event: 'loaded' });

                    }, function(error) {
                        console.error('[GLBViewer] GLB parse error:', error);
                        window.webkit.messageHandlers.glbViewer.postMessage({
                            event: 'error',
                            message: 'Failed to parse 3D model'
                        });
                    });

                } catch (error) {
                    console.error('[GLBViewer] Error:', error);
                    window.webkit.messageHandlers.glbViewer.postMessage({
                        event: 'error',
                        message: error.message || 'Failed to load 3D model'
                    });
                }

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

            </script>
        </body>
        </html>
        """
    }
}
