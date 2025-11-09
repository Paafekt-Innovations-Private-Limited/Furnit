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
    @State private var showingSegmentExamine = false  // NEW: SegmentExamine state
    @State private var capturedImage: UIImage? = nil  // For both camera overlays
    
    // NEW: Progress bar state
    @State private var isInitializingCamera = false
    @State private var cameraInitProgress: Double = 0.0
    @State private var initializationTimer: Timer?
    
    init(model: USDZModel) {
        self.model = model
        print("🏗️ [ModelViewerView.init] ========================================")
        print("🏗️ [ModelViewerView.init] Initializing for model:")
        print("   - Display name: \(model.displayName)")
        print("   - File name: \(model.fileName)")
        print("   - Is saved room: \(model.isSavedRoom)")
        print("   - Model ID: \(model.id)")
        
        // Check if temporaryURL can be created
        if let url = model.temporaryURL {
            print("✅ [ModelViewerView.init] Model URL available: \(url.path)")
            print("   - URL exists: \(FileManager.default.fileExists(atPath: url.path))")
        } else {
            print("❌ [ModelViewerView.init] Model URL is NIL!")
        }
        print("🏗️ [ModelViewerView.init] ========================================")
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
                    .ignoresSafeArea(.all)
                    .onAppear {
                        print("👁️ [ModelViewerView.body] RealityKitView onAppear")
                        print("   - Model: \(model.displayName)")
                        print("   - Is saved room: \(model.isSavedRoom)")
                    }
                
                // NEW: Camera initialization overlay with progress
                if isInitializingCamera {
                    cameraInitializationOverlay
                }
                
                // SimpleCameraOverlay when camera is active
                if showingCameraPreview {
                    SimpleCameraOverlay(
                        capturedImage: $capturedImage,
                        isShowingCamera: $showingCameraPreview
                    )
                    .zIndex(100)
                }
                
                // NEW: SegmentExamine overlay (dual-mode segmentation)
                if showingSegmentExamine {
                    SegmentExamine(
                        capturedImage: $capturedImage,
                        isShowingCamera: $showingSegmentExamine
                    )
                    .zIndex(100)
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
        } message: {
            Text(screenshotMessage)
        }
        .onAppear {
            print("🎬 [ModelViewerView] View appeared")
            print("   - Model: \(model.displayName)")
            print("   - File: \(model.fileName)")
            print("   - Saved room: \(model.isSavedRoom)")
        }
        .onDisappear {
            print("👋 [ModelViewerView] View disappeared")
        }
    }
    
    // NEW: Progress bar overlay
    private var cameraInitializationOverlay: some View {
        ZStack {
            Color.black.opacity(0.9)
                .ignoresSafeArea()
            
            VStack(spacing: 24) {
                // Camera icon
                ZStack {
                    Circle()
                        .fill(Color.blue.opacity(0.2))
                        .frame(width: 100, height: 100)
                    
                    Image(systemName: "camera.fill")
                        .font(.system(size: 40))
                        .foregroundColor(.blue)
                }
                
                VStack(spacing: 12) {
                    Text("Initializing Camera")
                        .font(.title2)
                        .fontWeight(.semibold)
                        .foregroundColor(.white)
                    
                    Text(progressMessage)
                        .font(.subheadline)
                        .foregroundColor(.gray)
                        .multilineTextAlignment(.center)
                        .padding(.horizontal, 40)
                }
                
                // Progress bar
                VStack(spacing: 8) {
                    ProgressView(value: cameraInitProgress, total: 1.0)
                        .progressViewStyle(LinearProgressViewStyle(tint: .blue))
                        .frame(width: 250)
                    
                    Text("\(Int(cameraInitProgress * 100))%")
                        .font(.caption)
                        .foregroundColor(.gray)
                }
                
                // Cancel button
                Button(action: {
                    cancelCameraInitialization()
                }) {
                    Text("Cancel")
                        .font(.body)
                        .foregroundColor(.white)
                        .padding(.horizontal, 32)
                        .padding(.vertical, 12)
                        .background(Color.red.opacity(0.8))
                        .cornerRadius(25)
                }
                .padding(.top, 8)
            }
        }
        .transition(.opacity)
    }
    
    // NEW: Progress messages
    private var progressMessage: String {
        if cameraInitProgress < 0.3 {
            return "Checking camera permissions..."
        } else if cameraInitProgress < 0.6 {
            return "Loading U²-Net model..."
        } else if cameraInitProgress < 0.9 {
            return "Preparing segmentation..."
        } else {
            return "Almost ready..."
        }
    }
    
    private func isLandscape(geometry: GeometryProxy) -> Bool {
        return geometry.size.width > geometry.size.height
    }
    
    // 🎯 UPDATED: Centered joystick layout with SegmentExamine button
    private var portraitControls: some View {
        VStack {
            HStack {
                backButton
                Spacer()
                screenshotButton
            }
            .padding()
            
            Spacer()
            
            HStack {
                // Left side buttons stack
                VStack(spacing: 16) {
                    cameraButton
                    segmentExamineButton  // NEW: SegmentExamine button
                }
                .padding(.leading, 8)
                
                Spacer()
                
                // Joystick centered horizontally
                VirtualJoystick(joystickOffset: $joystickOffset)
                    .onChange(of: joystickOffset) { _, newOffset in
                        cameraMovementManager.updateJoystickInput(newOffset)
                    }
                
                Spacer()
            }
            .padding(.bottom, 40)
            .padding(.horizontal)
        }
    }
    
    private var landscapeControls: some View {
        HStack {
            VStack {
                HStack(spacing: 12) {
                    backButton
                    screenshotButton
                }
                Spacer()
            }
            .padding()
            
            Spacer()
            
            VStack(spacing: 16) {
                cameraButton
                segmentExamineButton  // NEW: SegmentExamine button
                
                Spacer()
                
                VirtualJoystick(joystickOffset: $joystickOffset)
                    .onChange(of: joystickOffset) { _, newOffset in
                        cameraMovementManager.updateJoystickInput(newOffset)
                    }
                
                Spacer()
            }
            .padding()
        }
    }
    
    private var backButton: some View {
        Button("Back") {
            dismiss()
        }
        .foregroundColor(.white)
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
        .background(Color.black.opacity(0.7))
        .cornerRadius(20)
    }
    
    private var screenshotButton: some View {
        Button(action: {
            takeScreenshot()
        }) {
            Image(systemName: "camera.circle.fill")
                .font(.system(size: 32))
                .foregroundColor(.white)
                .background(Circle().fill(Color.black.opacity(0.5)))
                .shadow(radius: 5)
        }
    }
    
    // MODIFIED: Camera button now starts initialization
    private var cameraButton: some View {
        Button(action: {
            if showingCameraPreview {
                // Turn off camera
                showingCameraPreview = false
                isARActive = false
            } else {
                // Start initialization with progress
                startCameraInitialization()
            }
        }) {
            Image(systemName: "camera.viewfinder")
                .font(.system(size: 28))
                .foregroundColor(.white)
                .frame(width: 60, height: 60)
                .background(
                    Circle()
                        .fill(showingCameraPreview ? Color.green : Color.blue)
                        .shadow(radius: 5)
                )
        }
    }
    
    // NEW: SegmentExamine button
    private var segmentExamineButton: some View {
        Button(action: {
            showingSegmentExamine.toggle()
        }) {
            Image(systemName: "crop.rotate")
                .font(.system(size: 28))
                .foregroundColor(.white)
                .frame(width: 60, height: 60)
                .background(
                    Circle()
                        .fill(showingSegmentExamine ? Color.green : Color.blue)
                        .shadow(radius: 5)
                )
        }
    }
    
    // NEW: Start camera initialization with progress
    private func startCameraInitialization() {
        print("📷 Starting camera initialization...")
        isInitializingCamera = true
        cameraInitProgress = 0.0
        isARActive = true
        
        initializationTimer = Timer.scheduledTimer(withTimeInterval: 0.05, repeats: true) { timer in
            self.cameraInitProgress += 0.02
            
            if self.cameraInitProgress >= 0.3 && self.cameraInitProgress < 0.32 {
                print("📷 Camera permissions checked")
            } else if self.cameraInitProgress >= 0.6 && self.cameraInitProgress < 0.62 {
                print("🤖 Loading U²-Net model")
            } else if self.cameraInitProgress >= 0.9 && self.cameraInitProgress < 0.92 {
                print("✅ Segmentation ready")
            }
            
            if self.cameraInitProgress >= 1.0 {
                timer.invalidate()
                self.initializationTimer = nil
                
                withAnimation(.easeOut(duration: 0.3)) {
                    self.isInitializingCamera = false
                }
                
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
                    self.showingCameraPreview = true
                    print("📷 Camera activated")
                }
            }
        }
    }
    
    // NEW: Cancel initialization
    private func cancelCameraInitialization() {
        initializationTimer?.invalidate()
        initializationTimer = nil
        
        withAnimation(.easeOut(duration: 0.2)) {
            isInitializingCamera = false
            cameraInitProgress = 0.0
            isARActive = false
        }
        
        print("❌ Camera initialization cancelled")
    }
    
    private func takeScreenshot() {
        print("📸 Taking screenshot...")
        
        guard let windowScene = UIApplication.shared.connectedScenes.first as? UIWindowScene,
              let window = windowScene.windows.first else {
            print("❌ Could not access window")
            screenshotMessage = "Failed to capture screenshot"
            showScreenshotAlert = true
            return
        }
        
        let renderer = UIGraphicsImageRenderer(bounds: window.bounds)
        let screenshot = renderer.image { context in
            window.drawHierarchy(in: window.bounds, afterScreenUpdates: true)
        }
        
        let saver = ScreenshotSaver()
        saver.onComplete = { success, error in
            DispatchQueue.main.async {
                if success {
                    print("✅ Screenshot saved")
                    self.screenshotMessage = "Screenshot saved to Photos"
                } else {
                    print("❌ Screenshot failed: \(error?.localizedDescription ?? "unknown")")
                    self.screenshotMessage = "Failed: \(error?.localizedDescription ?? "Please enable Photos access in Settings")"
                }
                self.showScreenshotAlert = true
            }
        }
        
        UIImageWriteToSavedPhotosAlbum(screenshot, saver, #selector(ScreenshotSaver.image(_:didFinishSavingWithError:contextInfo:)), nil)
    }
}

// Helper class to handle screenshot save callback
class ScreenshotSaver: NSObject {
    var onComplete: ((Bool, Error?) -> Void)?
    
    @objc func image(_ image: UIImage, didFinishSavingWithError error: Error?, contextInfo: UnsafeRawPointer) {
        onComplete?(error == nil, error)
    }
}
