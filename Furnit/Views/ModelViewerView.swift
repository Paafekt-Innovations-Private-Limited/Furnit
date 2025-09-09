import SwiftUI
import SceneKit
import AVFoundation

// MARK: - AR Workflow State Management
enum ARWorkflowState: Equatable {
    case inactive
    case pointing           // User should point camera at furniture
    case capturing          // Capturing high-quality image
    case segmenting         // Processing image segmentation
    case processing         // Creating 3D object and placing in scene
    case complete           // Object placed successfully
    case error(String)      // Error occurred
    
    var displayMessage: String {
        switch self {
        case .inactive:
            return "AR mode inactive"
        case .pointing:
            return "Point camera at furniture objects like chair or table"
        case .capturing:
            return "Capturing high-quality image..."
        case .segmenting:
            return "Identifying and segmenting furniture..."
        case .processing:
            return "Processing and placing object in room..."
        case .complete:
            return "Object placed successfully! You can now move it around."
        case .error(let message):
            return "Error: \(message)"
        }
    }
    
    var isProcessing: Bool {
        switch self {
        case .capturing, .segmenting, .processing:
            return true
        default:
            return false
        }
    }
}

struct ModelViewerView: View {
    @Environment(\.horizontalSizeClass) var horizontalSizeClass
    @Environment(\.verticalSizeClass) var verticalSizeClass
    let model: USDZModel
    @Environment(\.dismiss) private var dismiss
    
    // Camera movement state
    @StateObject private var cameraMovementManager = CameraMovementManager()
    @State private var joystickOffset: CGSize = .zero
    
    // AR state
    @StateObject private var arCameraManager = ARCameraManager()
    @StateObject private var segmentationProcessor = ObjectSegmentationProcessor()
    @StateObject private var objectPlacementManager = ARObjectPlacementManager()
    @State private var isARActive = false
    @State private var arWorkflowState: ARWorkflowState = .inactive
    @State private var showingCameraPermissionAlert = false
    @State private var countdownTimer: Timer?
    @State private var countdownSeconds = 3
    @State private var showCountdown = false
    
    var body: some View {
        GeometryReader { geometry in
            ZStack {
                SceneKitView(model: model, cameraMovementManager: cameraMovementManager, arObjectPlacementManager: objectPlacementManager)
                    .ignoresSafeArea(.all)
                
                if isLandscape(geometry: geometry) {
                    landscapeControls
                } else {
                    portraitControls
                }
                
                // AR Status Message and Countdown
                if isARActive {
                    VStack {
                        if showCountdown {
                            countdownOverlay
                        } else {
                            arStatusMessage
                        }
                        Spacer()
                    }
                    .transition(.opacity.combined(with: .move(edge: .top)))
                }
            }
        }
        .navigationBarHidden(true)
        .statusBarHidden(true)
        .preferredColorScheme(.dark)
        .onAppear {
            setupARComponents()
        }
        .onDisappear {
            stopAR()
            stopCountdownTimer()
        }
        .alert("Camera Permission Required", isPresented: $showingCameraPermissionAlert) {
            Button("Settings") {
                openAppSettings()
            }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("This app needs camera access to identify furniture objects for AR placement. Please enable camera access in Settings.")
        }
    }
    
    private func isLandscape(geometry: GeometryProxy) -> Bool {
        return geometry.size.width > geometry.size.height
    }
    
    private var portraitControls: some View {
        VStack {
            // Top controls - back button
            HStack {
                backButton
                Spacer()
            }
            .padding()
            
            Spacer()
            
            // Bottom controls - AR button, info and joystick
            HStack {
                // AR Button (bottom left)
                VStack {
                    Spacer()
                    ARButton(isARActive: $isARActive, isProcessing: arWorkflowState.isProcessing) {
                        if arWorkflowState == .pointing {
                            startCapture()
                        } else {
                            toggleAR()
                        }
                    }
                    .padding(.bottom, 20)
                }
                
                VStack(spacing: 16) {
                    modelInfoPanel
                    
                    // Joystick for camera movement (bottom center in portrait)
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
            // Left side controls - back button and AR button
            VStack {
                backButton
                Spacer()
                
                // AR Button (bottom left in landscape)
                ARButton(isARActive: $isARActive, isProcessing: arWorkflowState.isProcessing) {
                    if arWorkflowState == .pointing {
                        startCapture()
                    } else {
                        toggleAR()
                    }
                }
                .padding(.bottom, 20)
            }
            .padding()
            
            Spacer()
            
            // Right side controls - info and joystick
            VStack(spacing: 16) {
                Spacer()
                
                modelInfoPanel
                
                // Joystick for camera movement (right side in landscape)
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
    
    private var arStatusMessage: some View {
        HStack {
            // Status icon based on workflow state
            statusIcon
                .font(.title2)
                .foregroundColor(.white)
            
            VStack(alignment: .leading, spacing: 4) {
                Text(arWorkflowState.displayMessage)
                    .font(.body)
                    .foregroundColor(.white)
                
                // Progress indicator for processing states
                if arWorkflowState.isProcessing {
                    ProgressView()
                        .progressViewStyle(CircularProgressViewStyle(tint: .white))
                        .scaleEffect(0.8)
                }
            }
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 12)
        .background(
            RoundedRectangle(cornerRadius: 25)
                .fill(backgroundColorForState.opacity(0.8))
                .shadow(radius: 4)
        )
        .padding(.top, 60) // Account for status bar and safe area
        .animation(.easeInOut(duration: 0.3), value: arWorkflowState)
    }
    
    private var countdownOverlay: some View {
        VStack {
            Text("Get ready!")
                .font(.headline)
                .foregroundColor(.white)
            
            Text("\(countdownSeconds)")
                .font(.system(size: 72, weight: .bold, design: .rounded))
                .foregroundColor(.white)
                .scaleEffect(1.2)
                .animation(.spring(), value: countdownSeconds)
                
            Text("Point camera at furniture")
                .font(.body)
                .foregroundColor(.white.opacity(0.8))
        }
        .padding(.horizontal, 30)
        .padding(.vertical, 20)
        .background(
            RoundedRectangle(cornerRadius: 25)
                .fill(Color.orange.opacity(0.9))
                .shadow(radius: 8)
        )
        .padding(.top, 60)
    }
    
    private var statusIcon: some View {
        switch arWorkflowState {
        case .inactive:
            return Image(systemName: "camera.viewfinder")
        case .pointing:
            return Image(systemName: "viewfinder")
        case .capturing:
            return Image(systemName: "camera.circle")
        case .segmenting:
            return Image(systemName: "scissors")
        case .processing:
            return Image(systemName: "cube.box")
        case .complete:
            return Image(systemName: "checkmark.circle.fill")
        case .error:
            return Image(systemName: "exclamationmark.triangle.fill")
        }
    }
    
    private var backgroundColorForState: Color {
        switch arWorkflowState {
        case .inactive:
            return .gray
        case .pointing:
            return .blue
        case .capturing:
            return .orange
        case .segmenting, .processing:
            return .purple
        case .complete:
            return .green
        case .error:
            return .red
        }
    }
    
    // MARK: - AR Functionality
    
    private func setupARComponents() {
        // Setup delegates
        arCameraManager.delegate = self
        segmentationProcessor.delegate = self
        objectPlacementManager.delegate = self
        
        print("🔧 AR components setup complete")
    }
    
    private func toggleAR() {
        if isARActive {
            stopAR()
        } else {
            startAR()
        }
    }
    
    private func startAR() {
        guard arCameraManager.hasPermission else {
            showingCameraPermissionAlert = true
            return
        }
        
        print("🚀 Starting AR mode...")
        
        isARActive = true
        arWorkflowState = .pointing
        
        // Clean up any existing objects before starting new session
        objectPlacementManager.clearAllObjects()
        
        // Start camera capture in continuous mode for preview
        arCameraManager.startCapture()
        objectPlacementManager.startPlacement()
        
        print("📱 AR mode active - point camera at furniture objects")
    }
    
    private func startCapture() {
        guard arWorkflowState == .pointing else {
            print("⚠️ Cannot start capture - not in pointing state")
            return
        }
        
        // Start countdown
        arWorkflowState = .pointing
        showCountdown = true
        countdownSeconds = 3
        startCountdownTimer()
    }
    
    private func startCountdownTimer() {
        countdownTimer?.invalidate()
        countdownTimer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { _ in
            DispatchQueue.main.async {
                self.countdownSeconds -= 1
                
                if self.countdownSeconds <= 0 {
                    self.stopCountdownTimer()
                    self.showCountdown = false
                    self.captureImage()
                }
            }
        }
    }
    
    private func stopCountdownTimer() {
        countdownTimer?.invalidate()
        countdownTimer = nil
    }
    
    private func captureImage() {
        print("📸 Capturing high-quality image...")
        arWorkflowState = .capturing
        arCameraManager.captureHighQualityImage()
    }
    
    private func stopAR() {
        print("🛑 Stopping AR mode...")
        
        isARActive = false
        arWorkflowState = .inactive
        showCountdown = false
        stopCountdownTimer()
        
        arCameraManager.stopCapture()
        arCameraManager.disableSingleCaptureMode()
        objectPlacementManager.stopPlacement()
        
        print("✅ AR mode stopped")
    }
    
    private func openAppSettings() {
        guard let settingsUrl = URL(string: UIApplication.openSettingsURLString) else {
            return
        }
        
        if UIApplication.shared.canOpenURL(settingsUrl) {
            UIApplication.shared.open(settingsUrl)
        }
    }
}

// MARK: - ARCameraManagerDelegate
extension ModelViewerView: ARCameraManagerDelegate {
    func cameraManager(_ manager: ARCameraManager, didOutput pixelBuffer: CVPixelBuffer) {
        // Only process continuous frames when in pointing state (for preview)
        guard arWorkflowState == .pointing && !showCountdown else { return }
        
        // Process frame for segmentation preview (optional - could be disabled to save processing)
        // segmentationProcessor.processFrame(pixelBuffer)
    }
    
    func cameraManager(_ manager: ARCameraManager, didCaptureImage image: UIImage, pixelBuffer: CVPixelBuffer) {
        print("✅ High-quality image captured, starting segmentation...")
        arWorkflowState = .segmenting
        
        // Process the captured high-quality image for segmentation
        segmentationProcessor.processFrame(pixelBuffer)
    }
    
    func cameraManager(_ manager: ARCameraManager, didFailWithError error: Error) {
        print("❌ Camera error: \(error.localizedDescription)")
        
        DispatchQueue.main.async {
            self.arWorkflowState = .error(error.localizedDescription)
            
            // Auto-stop AR after error
            DispatchQueue.main.asyncAfter(deadline: .now() + 3.0) {
                self.stopAR()
            }
        }
    }
}

// MARK: - ObjectSegmentationProcessorDelegate
extension ModelViewerView: ObjectSegmentationProcessorDelegate {
    func processor(_ processor: ObjectSegmentationProcessor, didSegment object: SegmentedObject) {
        print("✅ Segmentation complete, placing object in scene...")
        arWorkflowState = .processing
        
        // Place segmented object in 3D scene
        objectPlacementManager.placeSegmentedObject(object)
    }
    
    func processor(_ processor: ObjectSegmentationProcessor, didFailWithError error: Error) {
        print("❌ Segmentation error: \(error.localizedDescription)")
        
        DispatchQueue.main.async {
            self.arWorkflowState = .error("Failed to segment furniture: \(error.localizedDescription)")
            
            // Auto-return to pointing state after error
            DispatchQueue.main.asyncAfter(deadline: .now() + 3.0) {
                self.arWorkflowState = .pointing
            }
        }
    }
}

// MARK: - ARObjectPlacementManagerDelegate
extension ModelViewerView: ARObjectPlacementManagerDelegate {
    func placementManager(_ manager: ARObjectPlacementManager, didPlaceObject node: SCNNode) {
        print("✅ Placed AR object: \(node.name ?? "Unknown")")
        
        DispatchQueue.main.async {
            self.arWorkflowState = .complete
            
            // Disable single capture mode and return to normal mode
            self.arCameraManager.disableSingleCaptureMode()
            
            // Auto-return to inactive state and stop AR after success
            DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
                self.stopAR()
            }
        }
    }
    
    func placementManager(_ manager: ARObjectPlacementManager, didUpdateObject node: SCNNode) {
        print("🔄 Updated AR object: \(node.name ?? "Unknown")")
        
        DispatchQueue.main.async {
            self.arWorkflowState = .complete
            
            // Disable single capture mode and return to normal mode
            self.arCameraManager.disableSingleCaptureMode()
            
            // Auto-return to inactive state and stop AR after success
            DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
                self.stopAR()
            }
        }
    }
    
    func placementManager(_ manager: ARObjectPlacementManager, didRemoveObject node: SCNNode) {
        print("🗑️ Removed AR object: \(node.name ?? "Unknown")")
    }
}