import SwiftUI
import RealityKit
import Combine
import Photos

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

    // Camera/Segmentation state
    @State private var showingCameraPreview = false
    @State private var showingSegmentExamine = false
    @State private var showingSegmentForeground = false
    @State private var showingSegmentFurniture = false
    @State private var capturedImage: UIImage? = nil
    @State private var roomSnapshot: UIImage? = nil

    // Progress bar state
    @State private var isInitializingCamera = false
    @State private var cameraInitProgress: Double = 0.0
    @State private var initializationTimer: Timer?
    
    // Furniture segmentation progress
    @State private var isInitializingFurniture = false
    @State private var furnitureInitProgress: Double = 0.0
    @State private var furnitureInitTimer: Timer?

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
                if isInitializingFurniture { furnitureInitializationOverlay }

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
                        isShowingCamera: $showingSegmentFurniture,
                        roomImage: roomSnapshot
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
    
    private var furnitureInitializationOverlay: some View {
        ZStack {
            Color.black.opacity(0.9).ignoresSafeArea()
            VStack(spacing: 24) {
                ZStack {
                    Circle().fill(Color.green.opacity(0.2)).frame(width: 100, height: 100)
                    Image(systemName: "chair.fill").font(.system(size: 40)).foregroundColor(.green)
                }
                VStack(spacing: 12) {
                    Text("Initializing Furniture Segmentation").font(.title2).fontWeight(.semibold).foregroundColor(.white)
                    Text(furnitureProgressMessage).font(.subheadline).foregroundColor(.gray)
                        .multilineTextAlignment(.center).padding(.horizontal, 40)
                }
                VStack(spacing: 8) {
                    ProgressView(value: furnitureInitProgress, total: 1.0)
                        .progressViewStyle(LinearProgressViewStyle(tint: .green))
                        .frame(width: 250)
                    Text("\(Int(furnitureInitProgress * 100))%").font(.caption).foregroundColor(.gray)
                }
                Button(action: { cancelFurnitureInitialization() }) {
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
        else if cameraInitProgress < 0.6 { return "Loading YOLO11-seg model..." }
        else if cameraInitProgress < 0.9 { return "Preparing segmentation..." }
        else { return "Almost ready..." }
    }
    
    private var furnitureProgressMessage: String {
        if furnitureInitProgress < 0.3 { return "Capturing room view..." }
        else if furnitureInitProgress < 0.6 { return "Loading YOLO11 segmentation model..." }
        else if furnitureInitProgress < 0.9 { return "Initializing camera..." }
        else { return "Starting furniture detection..." }
    }

    private func isLandscape(geometry: GeometryProxy) -> Bool {
        geometry.size.width > geometry.size.height
    }

    // PORTRAIT controls - NO SCREENSHOT BUTTON
    private var portraitControls: some View {
        VStack {
            HStack {
                backButton
                Spacer()
                // Removed screenshot button
            }.padding()
            
            Spacer()
            
            HStack(alignment: .bottom, spacing: 0) {
                VStack(spacing: 16) {
                    cameraButton
                    segmentExamineButton
                    segmentForegroundButton
                    segmentFurnitureButton
                }
                .padding(.leading, 16)
                .padding(.bottom, 20)
                
                Spacer()
                
                VirtualJoystick(joystickOffset: $joystickOffset)
                    .onChange(of: joystickOffset) { _, newOffset in
                        cameraMovementManager.updateJoystickInput(newOffset)
                    }
                    .padding(.bottom, 20)
                    .padding(.trailing, 100)
                
                Spacer()
            }
            .padding(.bottom, 20)
        }
    }

    // LANDSCAPE controls - NO SCREENSHOT BUTTON
    private var landscapeControls: some View {
        HStack {
            VStack {
                HStack(spacing: 12) {
                    backButton
                    // Removed screenshot button
                }
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
                    .padding(.top, 20)
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

    private var segmentFurnitureButton: some View {
        Button(action: {
            if showingSegmentFurniture {
                showingSegmentFurniture = false
            } else {
                startFurnitureInitialization()
            }
        }) {
            Image(systemName: "chair.fill")
                .font(.system(size: 28)).foregroundColor(.white)
                .frame(width: 60, height: 60)
                .background(Circle().fill(showingSegmentFurniture ? Color.green : Color.blue).shadow(radius: 5))
        }
    }
    
    // MARK: - Furniture Initialization with Progress
    
    private func startFurnitureInitialization() {
        // Hide other overlays
        showingCameraPreview = false
        showingSegmentExamine = false
        showingSegmentForeground = false
        
        // Start progress
        isARActive = false
        isInitializingFurniture = true
        furnitureInitProgress = 0.0
        
        // Simulate progress
        furnitureInitTimer = Timer.scheduledTimer(withTimeInterval: 0.03, repeats: true) { timer in
            self.furnitureInitProgress += 0.015
            
            // Capture room at 30%
            if self.furnitureInitProgress >= 0.3 && self.furnitureInitProgress < 0.35 && self.roomSnapshot == nil {
                self.captureRoomSnapshot()
            }
            
            if self.furnitureInitProgress >= 1.0 {
                timer.invalidate()
                self.furnitureInitTimer = nil
                withAnimation(.easeOut(duration: 0.3)) {
                    self.isInitializingFurniture = false
                }
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
                    self.showingSegmentFurniture = true
                    self.manageARSessionForOverlays()
                }
            }
        }
    }
    
    private func cancelFurnitureInitialization() {
        furnitureInitTimer?.invalidate()
        furnitureInitTimer = nil
        withAnimation(.easeOut(duration: 0.2)) {
            isInitializingFurniture = false
            furnitureInitProgress = 0.0
        }
        manageARSessionForOverlays()
    }
    
    private func captureRoomSnapshot() {
        if let windowScene = UIApplication.shared.connectedScenes.first as? UIWindowScene,
           let window = windowScene.windows.first {
            let renderer = UIGraphicsImageRenderer(bounds: window.bounds)
            self.roomSnapshot = renderer.image { context in
                window.layer.render(in: context.cgContext)
            }
            print("📸 Room captured during initialization")
        }
    }

    // MARK: - Camera Init

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
                    self.manageARSessionForOverlays()
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

    private func manageARSessionForOverlays() {
        let shouldRunAR = !anyCameraOverlayActive && !isInitializingCamera && !isInitializingFurniture
        if isARActive != shouldRunAR {
            isARActive = shouldRunAR
        }
    }
}
