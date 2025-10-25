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
                
                // SimpleCameraOverlay when camera is active
                if showingCameraPreview {
                    SimpleCameraOverlay(
                        capturedImage: .constant(nil),
                        isShowingCamera: $showingCameraPreview
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
    }
    
    private func isLandscape(geometry: GeometryProxy) -> Bool {
        return geometry.size.width > geometry.size.height
    }
    
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
                VStack(spacing: 16) {
                    cameraButton
                }
                
                VStack(spacing: 16) {
                    modelInfoPanel
                    
                    VirtualJoystick(joystickOffset: $joystickOffset)
                        .onChange(of: joystickOffset) { _, newOffset in
                            cameraMovementManager.updateJoystickInput(newOffset)
                        }
                }
                
                Spacer()
            }
            .padding()
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
                
                Spacer()
                
                modelInfoPanel
                
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
    
    private var cameraButton: some View {
        Button(action: {
            showingCameraPreview.toggle()
            isARActive = showingCameraPreview
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
    
    private var modelInfoPanel: some View {
        VStack(spacing: 8) {
            Text(model.displayName)
                .font(.title3)
                .fontWeight(.semibold)
                .foregroundColor(.white)
            
            Text("Use gestures to rotate, zoom, and explore the room")
                .font(.caption)
                .foregroundColor(.white.opacity(0.8))
                .multilineTextAlignment(.center)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
        .background(Color.black.opacity(0.7))
        .cornerRadius(16)
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
        
        // Create a screenshot saver to handle the callback
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
