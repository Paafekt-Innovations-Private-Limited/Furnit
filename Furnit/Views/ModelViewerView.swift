import SwiftUI
import RealityKit
import Combine
import Photos
import CoreML
import AVFoundation
import UIKit

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
    @State private var furnitureFitInitialSegmentationDone = false
    @State private var capturedImage: UIImage? = nil
    @State private var roomSnapshot: UIImage? = nil
    
    // ARView snapshot trigger (proper way to capture 3D content)
    @State private var shouldCaptureARViewSnapshot = false
    
    // Furniture hint
    @State private var showFurnitureHint = true

    @State private var isCapturingSnapshot = false

    init(model: USDZModel) {
        self.model = model
    }

    private var anyCameraOverlayActive: Bool {
        showingCameraPreview || showingSegmentExamine || showingSegmentForeground || showingSegmentFurniture || showingFurnitureFit
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
                    // FurnitureFit overlay (yoloe-11l 1280)
                    if showingFurnitureFit {
                        FurnitureFitUIView(
                            capturedImage: $capturedImage,
                            roomImage: roomSnapshot,
                            mlModel: yoloeService.model,
                            processInterval: 0.07,
                            active: true,
                            lockedOrientation: model.photoOrientation,
                            roomWidthMeters: model.roomWidth ?? 4.0,
                            roomHeightMeters: model.roomHeight ?? 3.0,
                            onFurnitureSizeEstimated: nil,
                            suppressStartupProgress: furnitureFitInitialSegmentationDone,
                            onFirstSegmentationComplete: { furnitureFitInitialSegmentationDone = true }
                        )
                        .zIndex(9000)
                    }

                    if isLandscape(geometry: geometry) {
                        landscapeControls
                    } else {
                        portraitControls
                    }
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
                            // Brain button
                            Button(action: {
                                logDebug("BRAIN FLOW: tap received")
                                // Dismiss hint on first touch
                                showFurnitureHint = false
                                
                                if showingFurnitureFit {
                                    showingFurnitureFit = false
                                } else {
                                    logDebug("BRAIN FLOW: requesting snapshot")
                                    // Trigger ARView snapshot
                                    shouldCaptureARViewSnapshot = true
                                    
                                    // Hide other overlays
                                    showingCameraPreview = false
                                    showingSegmentExamine = false
                                    showingSegmentForeground = false
                                    showingSegmentFurniture = false
                                    
                                    // Wait briefly for snapshot to complete
                                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.2) {
                                        logDebug("BRAIN FLOW: showing FurnitureFit overlay")
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
                            let screen = screenSizeFromContext()
                            saveFurnitureFitSnapshot(screen)
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
        .onChange(of: showingFurnitureFit) { _, _ in manageARSessionForOverlays() }
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
            OrientationLockManager.shared.unlock()
        }
    }

    // MARK: - Overlays & Controls

    private func isLandscape(geometry: GeometryProxy) -> Bool {
        geometry.size.width > geometry.size.height
    }

    // PORTRAIT controls
    private var portraitControls: some View {
        VStack {
            Spacer()
            
            HStack(alignment: .bottom, spacing: 0) {
                VStack(spacing: 16) {
                    // cameraButton
                    // segmentExamineButton
                    // segmentForegroundButton
                    // segmentFurnitureButton
                    // smartyPantsButton - NOW AT TOPMOST LEVEL
                }
                .padding(.leading, 16)
                .padding(.bottom, 20)
                
                Spacer()
                
                // VirtualJoystick - NOW AT TOPMOST LEVEL
                
                Spacer()
            }
            .padding(.bottom, 20)
        }
    }

    // LANDSCAPE controls
    private var landscapeControls: some View {
        HStack {
            Spacer()
            
            VStack(spacing: 16) {
                // cameraButton
                // segmentExamineButton
                // segmentForegroundButton
                // segmentFurnitureButton
                // smartyPantsButton - NOW AT TOPMOST LEVEL
                Spacer()
                // VirtualJoystick - NOW AT TOPMOST LEVEL
                Spacer()
            }
            .padding()
        }
    }

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

    private var cameraButton: some View {
        Button(action: {
            if showingCameraPreview {
                showingCameraPreview = false
            } else {
                showingSegmentExamine = false
                showingSegmentForeground = false
                showingSegmentFurniture = false
                showingFurnitureFit = false
                showingCameraPreview = true
            }
        }) {
            Image(systemName: "camera.viewfinder")
                .font(.system(size: 28)).foregroundColor(.white)
                .frame(width: 60, height: 60)
                .background(Circle().fill(showingCameraPreview ? Color.green : Color.blue).shadow(radius: 5))
        }
    }

    private var segmentExamineButton: some View {
        Button(action: {
            showingCameraPreview = false
            showingSegmentForeground = false
            showingSegmentFurniture = false
            showingFurnitureFit = false
            showingSegmentExamine.toggle()
        }) {
            Image(systemName: "crop.rotate")
                .font(.system(size: 28)).foregroundColor(.white)
                .frame(width: 60, height: 60)
                .background(Circle().fill(showingSegmentExamine ? Color.green : Color.blue).shadow(radius: 5))
        }
    }

    private var segmentForegroundButton: some View {
        Button(action: {
            showingCameraPreview = false
            showingSegmentExamine = false
            showingSegmentFurniture = false
            showingFurnitureFit = false
            showingSegmentForeground.toggle()
        }) {
            Image(systemName: "viewfinder")
                .font(.system(size: 28)).foregroundColor(.white)
                .frame(width: 60, height: 60)
                .background(Circle().fill(showingSegmentForeground ? Color.purple : Color.blue).shadow(radius: 5))
        }
    }

    private var segmentFurnitureButton: some View {
        ZStack {
            Button(action: {
                // Dismiss hint on first touch
                showFurnitureHint = false
                
                if showingSegmentFurniture {
                    showingSegmentFurniture = false
                } else {
                    // Trigger ARView snapshot (will be handled by RealityKitView)
                    shouldCaptureARViewSnapshot = true
                    
                    // Hide other overlays
                    showingCameraPreview = false
                    showingSegmentExamine = false
                    showingSegmentForeground = false
                    showingFurnitureFit = false
                    
                    // Wait briefly for snapshot to complete, then open scanner
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.2) {
                        self.showingSegmentFurniture = true
                    }
                }
            }) {
                Image(systemName: "chair.fill")
                    .font(.system(size: 28)).foregroundColor(.white)
                    .frame(width: 60, height: 60)
                    .background(Circle().fill(showingSegmentFurniture ? Color.green : Color.blue).shadow(radius: 5))
            }
            
            // Hint badge on button
            if showFurnitureHint {
                VStack {
                    HStack {
                        Spacer()
                        Image(systemName: "hand.tap.fill")
                            .font(.caption)
                            .foregroundColor(.white)
                            .padding(4)
                            .background(Circle().fill(Color.orange))
                    }
                    Spacer()
                }
                .frame(width: 60, height: 60)
            }
        }
    }

    // FurnitureFit Button (YOLOE - 168 furniture classes!)
    private var smartyPantsButton: some View {
        ZStack {
            Button(action: {
                // Dismiss hint on first touch
                showFurnitureHint = false
                
                if showingFurnitureFit {
                    showingFurnitureFit = false
                } else {
                    // Trigger ARView snapshot
                    shouldCaptureARViewSnapshot = true
                    
                    // Hide other overlays
                    showingCameraPreview = false
                    showingSegmentExamine = false
                    showingSegmentForeground = false
                    showingSegmentFurniture = false
                    
                    // Wait briefly for snapshot to complete
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.2) {
                        self.showingFurnitureFit = true
                    }
                }
            }) {
                Image(systemName: "brain.head.profile")
                    .font(.system(size: 28)).foregroundColor(.white)
                    .frame(width: 60, height: 60)
                    .background(Circle().fill(showingFurnitureFit ? Color.green : Color.blue).shadow(radius: 5))
            }
            
            // Hint badge on button
            if showFurnitureHint {
                VStack {
                    HStack {
                        Spacer()
                        Image(systemName: "hand.tap.fill")
                            .font(.caption)
                            .foregroundColor(.white)
                            .padding(4)
                            .background(Circle().fill(Color.orange))
                    }
                    Spacer()
                }
                .frame(width: 60, height: 60)
            }
        }
    }

    private func manageARSessionForOverlays() {
        let shouldRunAR = !anyCameraOverlayActive
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
    
    /// Screen size from window scene context (avoids UIScreen.main deprecated in iOS 26).
    private func screenSizeFromContext() -> CGSize {
        let scene = UIApplication.shared.connectedScenes.first as? UIWindowScene
        let window = scene?.windows.first(where: { $0.isKeyWindow }) ?? scene?.windows.first
        return window?.windowScene?.screen.bounds.size ?? window?.bounds.size ?? CGSize(width: 393, height: 852)
    }

    private func saveFurnitureFitSnapshot(_ size: CGSize) {
        // Hide UI chrome briefly so it doesn't appear in the snapshot
        isCapturingSnapshot = true
        // Allow the UI to update before capturing
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

    private func shareFurnitureFitSnapshot(_ size: CGSize) {
        // Hide UI chrome briefly so it doesn't appear in the snapshot
        isCapturingSnapshot = true
        // Allow the UI to update before capturing
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.06) {
            guard let uiImage = self.captureAppWindowImage() else {
                self.isCapturingSnapshot = false
                logDebug("❌ Failed to capture app window image for sharing")
                return
            }
            DispatchQueue.main.async {
                self.isCapturingSnapshot = false
                self.presentShareSheet(with: uiImage)
            }
        }
    }

    private func presentShareSheet(with image: UIImage) {
        let activityVC = UIActivityViewController(
            activityItems: [image],
            applicationActivities: nil
        )

        // Find the presenting view controller
        if let windowScene = UIApplication.shared.connectedScenes.first as? UIWindowScene,
           let rootVC = windowScene.windows.first?.rootViewController {
            var presentingVC = rootVC
            while let presented = presentingVC.presentedViewController {
                presentingVC = presented
            }

            // For iPad, set popover source
            if let popover = activityVC.popoverPresentationController {
                popover.sourceView = presentingVC.view
                popover.sourceRect = CGRect(x: presentingVC.view.bounds.midX, y: presentingVC.view.bounds.maxY - 100, width: 0, height: 0)
            }

            presentingVC.present(activityVC, animated: true)
            logDebug("📤 Share sheet presented")
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

    var roomImage: UIImage?
    var mlModel: MLModel?  // yoloe-11l 1280 model
    var processInterval: Double = 0.07
    var scoreThreshold: Float = 0.25
    var active: Bool = true
    var lockedOrientation: PhotoOrientation = .portrait  // Room's photo orientation

    // Room dimensions from SHARP (in meters) for furniture sizing
    var roomWidthMeters: Float = 4.0
    var roomHeightMeters: Float = 3.0

    // Callback for reporting estimated furniture size (room-based + optional AR height, in meters)
    var onFurnitureSizeEstimated: ((FurnitureSizeEstimate) -> Void)?
    /// Sharp Room: skip “Starting camera…” progress after the first segmentation this session.
    var suppressStartupProgress: Bool = false
    var onFirstSegmentationComplete: (() -> Void)?

    func makeUIView(context: Context) -> FurnitureFitContainerView {
        let view = FurnitureFitContainerView()
        view.setModel(mlModel)
        view.lockedOrientation = lockedOrientation
        view.roomWidthMeters = roomWidthMeters
        view.roomHeightMeters = roomHeightMeters
        view.confidenceThreshold = scoreThreshold
        view.onFurnitureSizeEstimated = onFurnitureSizeEstimated
        view.suppressStartupProgress = suppressStartupProgress
        view.onFirstSegmentationComplete = onFirstSegmentationComplete
        return view
    }

    func updateUIView(_ uiView: FurnitureFitContainerView, context: Context) {
        uiView.setModel(mlModel)
        uiView.processInterval = processInterval
        uiView.lockedOrientation = lockedOrientation
        uiView.roomWidthMeters = roomWidthMeters
        uiView.roomHeightMeters = roomHeightMeters
        uiView.confidenceThreshold = scoreThreshold
        uiView.onFurnitureSizeEstimated = onFurnitureSizeEstimated
        uiView.suppressStartupProgress = suppressStartupProgress
        uiView.onFirstSegmentationComplete = onFirstSegmentationComplete
        if active { uiView.startIfNeeded() } else { uiView.stop() }
    }
}

