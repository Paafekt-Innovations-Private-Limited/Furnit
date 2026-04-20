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
    @State private var furnitureFitEstimatedHeightM: Float?
    @State private var capturedImage: UIImage? = nil
    @State private var roomSnapshot: UIImage? = nil
    
    // ARView snapshot trigger (proper way to capture 3D content)
    @State private var shouldCaptureARViewSnapshot = false
    
    // Furniture hint
    @State private var showFurnitureHint = true
    @State private var showFullVideoWithIdentifications = false
    @State private var fullVideoFurnitureTapHintVisible = false
    @State private var tapHintColorIndex: Int = 0
    private let tapHintColors: [Color] = [.yellow, .cyan, .orange, .green, .pink]
    @State private var tapHintColorTimer: Timer?

    @State private var isCapturingSnapshot = false

    init(model: USDZModel) {
        self.model = model
    }

    private var canSegmentSelectedFurniture: Bool {
        showingFurnitureFit && !selectedFurnitureFitLabels.isEmpty
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
                            roomWidthMeters: model.roomWidth ?? 4.0,
                            roomHeightMeters: model.roomHeight ?? 3.0,
                            onFurnitureSizeEstimated: { estimate in
                                furnitureFitEstimatedHeightM = estimate.heightMeters
                            },
                            suppressStartupProgress: furnitureFitInitialSegmentationDone,
                            onFirstSegmentationComplete: { furnitureFitInitialSegmentationDone = true },
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
                
                // TOPMOST BACK BUTTON - ALWAYS ON TOP
                VStack {
                    HStack {
                        backButton
                            .allowsHitTesting(true)
                        Spacer()
                    }
                    .padding()
                    Spacer()
                }
                .opacity(isCapturingSnapshot ? 0 : 1)
                .zIndex(99999) // HIGHEST POSSIBLE Z-INDEX
                .allowsHitTesting(true)

                // Memory info HUD (top-left, below back button)
                if model.hasFileSize && !showingFurnitureFit {
                    VStack {
                        HStack {
                            Text(model.fileSizeFormatted)
                                .font(.caption.monospacedDigit())
                                .foregroundColor(.white)
                                .padding(8)
                                .background(Color.black.opacity(0.5))
                                .cornerRadius(8)
                            Spacer()
                        }
                        .padding(.leading, 16)
                        .padding(.top, 60)
                        Spacer()
                    }
                    .opacity(isCapturingSnapshot ? 0 : 1)
                    .zIndex(99998)
                }
                
                // TOPMOST BRAIN ICON - ALWAYS ON TOP
                VStack {
                    Spacer()
                    HStack {
                        VStack(spacing: 16) {
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
                                    .font(.system(size: 22, weight: .medium))
                                    .symbolVariant(showFullVideoWithIdentifications ? .fill : .none)
                                    .foregroundStyle(.white)
                                    .frame(width: 52, height: 52)
                                    .background(
                                        Circle().fill(
                                            showFullVideoWithIdentifications
                                                ? Color.cyan.opacity(0.88)
                                                : Color.black.opacity(0.45)
                                        )
                                    )
                                    .shadow(radius: 4)
                            }
                            .buttonStyle(.plain)
                            .accessibilityLabel(L10n.Settings.fullVideoWithIdentifications)
                            .accessibilityHint(L10n.Settings.fullVideoWithIdentificationsDescription)
                            .accessibilityAddTraits(showFullVideoWithIdentifications ? .isSelected : [])

                            HStack(alignment: .bottom, spacing: 8) {
                                // Brain button
                                Button(action: {
                                    logDebug("BRAIN FLOW: tap received")
                                    // Dismiss hint on first touch
                                    showFurnitureHint = false
                                    
                                    if showingFurnitureFit {
                                        dismissFullVideoFurnitureTapHint()
                                        showingFurnitureFit = false
                                    } else {
                                        logDebug("BRAIN FLOW: loading YOLOE and opening FurnitureFit")
                                        yoloeService.ensureModelLoaded()
                                        showFullVideoWithIdentifications = false
                                        furnitureFitSegmentationMode = .identifyOnly
                                        furnitureFitShowIdentifyLivePreview = true
                                        selectedFurnitureFitLabels = []

                                        // Hide other overlays
                                        showingCameraPreview = false
                                        showingSegmentExamine = false
                                        showingSegmentForeground = false
                                        showingSegmentFurniture = false
                                        
                                        // Open after the current UI update cycle; no AR snapshot needed here.
                                        DispatchQueue.main.asyncAfter(deadline: .now() + 0.2) {
                                            logDebug("BRAIN FLOW: showing FurnitureFit overlay")
                                            self.furnitureFitInitialSegmentationDone = false
                                            self.showingFurnitureFit = true
                                        }
                                    }
                                }) {
                                    Image(systemName: "brain.head.profile")
                                        .font(.system(size: 28))
                                        .foregroundColor(.white)
                                        .frame(width: 60, height: 60)
                                        .background(Circle().fill(showingFurnitureFit ? Color.green : Color.blue).shadow(radius: 5))
                                }
                                .contentShape(Circle())
                                .frame(width: 76, height: 76)

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
                        // Snapshot button - always visible (green share button removed; retain blue camera only)
                        Button(action: {
                            saveFurnitureFitSnapshot()
                        }) {
                            Image(systemName: "camera.fill")
                                .font(.system(size: 28))
                                .foregroundColor(.white)
                                .frame(width: 60, height: 60)
                                .background(Circle().fill(Color.blue).shadow(radius: 5))
                        }
                        .disabled(isCapturingSnapshot)
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
            } else {
                dismissFullVideoFurnitureTapHint()
                furnitureFitSegmentationMode = .identifyOnly
                furnitureFitShowIdentifyLivePreview = true
                selectedFurnitureFitLabels = []
                roomSnapshot = nil
                capturedImage = nil
                furnitureFitEstimatedHeightM = nil
                yoloeService.releaseResources()
            }
        }
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
        }
        .onDisappear {
            dismissFullVideoFurnitureTapHint()
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
