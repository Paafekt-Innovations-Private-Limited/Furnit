import SwiftUI
import RealityKit
import Combine

struct ModelViewerView: View {
    @Environment(\.horizontalSizeClass) var horizontalSizeClass
    @Environment(\.verticalSizeClass) var verticalSizeClass
    let model: USDZModel
    @Environment(\.dismiss) private var dismiss

    // Camera movement state
    @StateObject private var cameraMovementManager = RealityKitCameraMovementManager()
    @State private var joystickOffset: CGSize = .zero

    // Required for RealityKitView
    @StateObject private var arObjectPlacementManager = RealityKitObjectPlacementManager()
    @State private var isARActive = false

    // Screenshot state
    @State private var showScreenshotAlert = false
    @State private var screenshotMessage = "Screenshot saved to Photos"

    // Camera/Segmentation state
    @State private var showingCameraPreview = false
    @State private var showingSegmentExamine = false
    @State private var showingSegmentForeground = false
    @State private var showingSegmentFurniture = false
    @State private var capturedImage: UIImage? = nil

    // Progress bar state
    @State private var isInitializingCamera = false
    @State private var cameraInitProgress: Double = 0.0
    @State private var initializationTimer: Timer?

    init(model: USDZModel) {
        self.model = model
    }

    private var anyCameraOverlayActive: Bool {
        showingCameraPreview || showingSegmentExamine || showingSegmentForeground || showingSegmentFurniture
    }

    var body: some View {
        GeometryReader { geometry in
            ZStack {
                RealityKitView(
                    model: model,
                    cameraMovementManager: cameraMovementManager,
                    arObjectPlacementManager: arObjectPlacementManager,
                    isARActive: isARActive
                )
                .allowsHitTesting(!(showingCameraPreview || showingSegmentExamine || showingSegmentForeground || showingSegmentFurniture))
                .ignoresSafeArea(.all)

                if isInitializingCamera { cameraInitializationOverlay }

                if showingCameraPreview {
                    SimpleCameraOverlay(
                        capturedImage: $capturedImage,
                        isShowingCamera: $showingCameraPreview
                    )
                    .zIndex(9000)
                }

                if showingSegmentExamine {
                    SegmentExamine(
                        capturedImage: $capturedImage,
                        isShowingCamera: $showingSegmentExamine
                    )
                    .zIndex(9999)
                }

                if showingSegmentForeground {
                    SegmentForeground(
                        capturedImage: $capturedImage,
                        isShowingCamera: $showingSegmentForeground
                    )
                    .zIndex(9000)
                }

                if showingSegmentFurniture {
                    SegmentFurniture(
                        capturedImage: $capturedImage,
                        isShowingCamera: $showingSegmentFurniture
                    )
                    .zIndex(9000)
                }

                if isLandscape(geometry: geometry) {
                    landscapeControls
                } else {
                    portraitControls
                }
            }
        }
        .navigationBarHidden(true)
        .statusBarHidden(true)
        .preferredColorScheme(.dark)
        .alert("Screenshot", isPresented: $showScreenshotAlert) {
            Button("OK", role: .cancel) { }
        } message: { Text(screenshotMessage) }
        .onChange(of: showingCameraPreview) { _, _ in manageARSessionForOverlays() }
        .onChange(of: showingSegmentExamine) { _, _ in manageARSessionForOverlays() }
        .onChange(of: showingSegmentForeground) { _, _ in manageARSessionForOverlays() }
        .onChange(of: showingSegmentFurniture) { _, _ in manageARSessionForOverlays() }
        .onAppear { isARActive = true }
    }

    // MARK: - Overlays & Controls

    private var cameraInitializationOverlay: some View {
        ZStack {
            Color.black.opacity(0.9).ignoresSafeArea()
            VStack(spacing: 24) {
                ZStack {
                    Circle().fill(Color.blue.opacity(0.2)).frame(width: 100, height: 100)
                    Image(systemName: "camera.fill").font(.system(size: 40)).foregroundColor(.blue)
                }
                VStack(spacing: 12) {
                    Text("Initializing Camera").font(.title2).fontWeight(.semibold).foregroundColor(.white)
                    Text(progressMessage).font(.subheadline).foregroundColor(.gray)
                        .multilineTextAlignment(.center).padding(.horizontal, 40)
                }
                VStack(spacing: 8) {
                    ProgressView(value: cameraInitProgress, total: 1.0)
                        .progressViewStyle(LinearProgressViewStyle(tint: .blue))
                        .frame(width: 250)
                    Text("\(Int(cameraInitProgress * 100))%").font(.caption).foregroundColor(.gray)
                }
                Button(action: { cancelCameraInitialization() }) {
                    Text("Cancel").font(.body).foregroundColor(.white)
                        .padding(.horizontal, 32).padding(.vertical, 12)
                        .background(Color.red.opacity(0.8)).cornerRadius(25)
                }
                .padding(.top, 8)
            }
        }
        .transition(.opacity)
    }

    private var progressMessage: String {
        if cameraInitProgress < 0.3 { return "Checking camera permissions..." }
        else if cameraInitProgress < 0.6 { return "Loading U²-Net model..." }
        else if cameraInitProgress < 0.9 { return "Preparing segmentation..." }
        else { return "Almost ready..." }
    }

    private func isLandscape(geometry: GeometryProxy) -> Bool {
        geometry.size.width > geometry.size.height
    }

    // PORTRAIT controls - JOYSTICK MOVED DOWN
    private var portraitControls: some View {
        VStack {
            HStack { backButton; Spacer(); screenshotButton }.padding()
            Spacer()
            HStack(alignment: .top) {
                VStack(spacing: 16) {
                    cameraButton
                    segmentExamineButton
                    segmentForegroundButton
                    segmentFurnitureButton
                }
                .padding(.leading, 8)

                Spacer()

                VirtualJoystick(joystickOffset: $joystickOffset)
                    .onChange(of: joystickOffset) { _, newOffset in
                        cameraMovementManager.updateJoystickInput(newOffset)
                    }
                    .padding(.top, 120)  // Moved down from 60
                    .padding(.bottom, 20) // Added bottom padding

                Spacer()
            }
            .padding(.bottom, 40)
            .padding(.horizontal)
        }
    }

    // LANDSCAPE controls - JOYSTICK ADJUSTED
    private var landscapeControls: some View {
        HStack {
            VStack {
                HStack(spacing: 12) { backButton; screenshotButton }
                Spacer()
            }
            .padding()
            Spacer()
            VStack(spacing: 16) {
                cameraButton
                segmentExamineButton
                segmentForegroundButton
                segmentFurnitureButton
                Spacer()
                VirtualJoystick(joystickOffset: $joystickOffset)
                    .onChange(of: joystickOffset) { _, newOffset in
                        cameraMovementManager.updateJoystickInput(newOffset)
                    }
                    .padding(.top, 40)
                Spacer()
            }
            .padding()
        }
    }

    private var backButton: some View {
        Button("Back") { dismiss() }
            .foregroundColor(.white)
            .padding(.horizontal, 16).padding(.vertical, 8)
            .background(Color.black.opacity(0.7)).cornerRadius(20)
    }

    private var screenshotButton: some View {
        Button(action: { takeScreenshot() }) {
            Image(systemName: "camera.circle.fill")
                .font(.system(size: 32)).foregroundColor(.white)
                .background(Circle().fill(Color.black.opacity(0.5))).shadow(radius: 5)
        }
    }

    // BUTTON 1: Simple camera overlay
    private var cameraButton: some View {
        Button(action: {
            if showingCameraPreview {
                showingCameraPreview = false
            } else {
                showingSegmentExamine = false
                showingSegmentForeground = false
                showingSegmentFurniture = false
                startCameraInitialization()
            }
        }) {
            Image(systemName: "camera.viewfinder")
                .font(.system(size: 28)).foregroundColor(.white)
                .frame(width: 60, height: 60)
                .background(Circle().fill(showingCameraPreview ? Color.green : Color.blue).shadow(radius: 5))
        }
    }

    // BUTTON 2: SegmentExamine (FastSAM)
    private var segmentExamineButton: some View {
        Button(action: {
            showingCameraPreview = false
            showingSegmentForeground = false
            showingSegmentFurniture = false
            showingSegmentExamine.toggle()
        }) {
            Image(systemName: "crop.rotate")
                .font(.system(size: 28)).foregroundColor(.white)
                .frame(width: 60, height: 60)
                .background(Circle().fill(showingSegmentExamine ? Color.green : Color.blue).shadow(radius: 5))
        }
    }

    // BUTTON 3: SegmentForeground
    private var segmentForegroundButton: some View {
        Button(action: {
            showingCameraPreview = false
            showingSegmentExamine = false
            showingSegmentFurniture = false
            showingSegmentForeground.toggle()
        }) {
            Image(systemName: "viewfinder")
                .font(.system(size: 28)).foregroundColor(.white)
                .frame(width: 60, height: 60)
                .background(Circle().fill(showingSegmentForeground ? Color.purple : Color.blue).shadow(radius: 5))
        }
    }

    // BUTTON 4: SegmentFurniture (YOLO11-seg)
    private var segmentFurnitureButton: some View {
        Button(action: {
            showingCameraPreview = false
            showingSegmentExamine = false
            showingSegmentForeground = false
            showingSegmentFurniture.toggle()
        }) {
            Image(systemName: "chair.fill")
                .font(.system(size: 28)).foregroundColor(.white)
                .frame(width: 60, height: 60)
                .background(Circle().fill(showingSegmentFurniture ? Color.green : Color.blue).shadow(radius: 5))
        }
    }

    // MARK: - Init/cancel flow

    private func startCameraInitialization() {
        isARActive = false
        isInitializingCamera = true
        cameraInitProgress = 0.0

        initializationTimer = Timer.scheduledTimer(withTimeInterval: 0.05, repeats: true) { timer in
            self.cameraInitProgress += 0.02
            if self.cameraInitProgress >= 1.0 {
                timer.invalidate()
                self.initializationTimer = nil
                withAnimation(.easeOut(duration: 0.3)) { self.isInitializingCamera = false }
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
                    self.showingCameraPreview = true
                    manageARSessionForOverlays()
                }
            }
        }
    }

    private func cancelCameraInitialization() {
        initializationTimer?.invalidate()
        initializationTimer = nil
        withAnimation(.easeOut(duration: 0.2)) {
            isInitializingCamera = false
            cameraInitProgress = 0.0
        }
        manageARSessionForOverlays()
    }

    private func takeScreenshot() {
        guard let windowScene = UIApplication.shared.connectedScenes.first as? UIWindowScene,
              let window = windowScene.windows.first else {
            screenshotMessage = "Failed to capture screenshot"
            showScreenshotAlert = true
            return
        }
        let renderer = UIGraphicsImageRenderer(bounds: window.bounds)
        let screenshot = renderer.image { ctx in window.drawHierarchy(in: window.bounds, afterScreenUpdates: true) }
        let saver = ScreenshotSaver()
        saver.onComplete = { success, error in
            DispatchQueue.main.async {
                self.screenshotMessage = success ? "Screenshot saved to Photos" :
                "Failed: \(error?.localizedDescription ?? "Please enable Photos access in Settings")"
                self.showScreenshotAlert = true
            }
        }
        UIImageWriteToSavedPhotosAlbum(screenshot, saver, #selector(ScreenshotSaver.image(_:didFinishSavingWithError:contextInfo:)), nil)
    }

    // MARK: - AR <-> Overlay coordination

    private func manageARSessionForOverlays() {
        let shouldRunAR = !anyCameraOverlayActive && !isInitializingCamera
        if isARActive != shouldRunAR {
            isARActive = shouldRunAR
        }
    }
}

// Screenshot helper
class ScreenshotSaver: NSObject {
    var onComplete: ((Bool, Error?) -> Void)?
    @objc func image(_ image: UIImage, didFinishSavingWithError error: Error?, contextInfo: UnsafeRawPointer) {
        onComplete?(error == nil, error)
    }
}
