import SwiftUI
import SceneKit
import Combine

struct ModelViewerView: View {
    @Environment(\.horizontalSizeClass) var horizontalSizeClass
    @Environment(\.verticalSizeClass) var verticalSizeClass
    let model: USDZModel
    @Environment(\.dismiss) private var dismiss
    
    // Camera movement state
    @StateObject private var cameraMovementManager = CameraMovementManager()
    @State private var joystickOffset: CGSize = .zero
    
    // AR functionality state
    @StateObject private var scnViewCaptureManager = SCNViewCaptureManager()
    @StateObject private var objectSegmentationProcessor = ObjectSegmentationProcessor()
    @StateObject private var arObjectPlacementManager = ARObjectPlacementManager()
    @State private var isARActive = false
    @State private var arStatusMessage = "Point at furniture objects"
    @State private var isProcessingAR = false
    
    var body: some View {
        GeometryReader { geometry in
            ZStack {
                SceneKitView(
                    model: model, 
                    cameraMovementManager: cameraMovementManager,
                    scnViewCaptureManager: scnViewCaptureManager,
                    arObjectPlacementManager: arObjectPlacementManager,
                    isARActive: isARActive
                )
                    .ignoresSafeArea(.all)
                
                if isLandscape(geometry: geometry) {
                    landscapeControls
                } else {
                    portraitControls
                }
                
                // AR status overlay
                if isARActive {
                    arStatusOverlay
                }
            }
        }
        .navigationBarHidden(true)
        .statusBarHidden(true)
        .preferredColorScheme(.dark)
        .onAppear {
            setupARManagers()
        }
        .onDisappear {
            cleanupAR()
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
            
            // Bottom controls - info, joystick, and AR button
            HStack {
                VStack(spacing: 16) {
                    // AR Button at bottom-left
                    ARButton(isARActive: $isARActive) {
                        toggleARMode()
                    }
                }
                
                VStack(spacing: 16) {
                    if !isARActive {
                        modelInfoPanel
                    }
                    
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
            // Left side controls - back button
            VStack {
                backButton
                Spacer()
            }
            .padding()
            
            Spacer()
            
            // Right side controls - info, joystick, and AR button
            VStack(spacing: 16) {
                // AR Button at bottom-left (moved to right side in landscape)
                ARButton(isARActive: $isARActive) {
                    toggleARMode()
                }
                
                Spacer()
                
                if !isARActive {
                    modelInfoPanel
                }
                
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
    
    // MARK: - AR Status Overlay
    private var arStatusOverlay: some View {
        VStack {
            Spacer()
            
            HStack {
                Spacer()
                
                VStack(spacing: 12) {
                    Text(arStatusMessage)
                        .font(.headline)
                        .foregroundColor(.white)
                        .multilineTextAlignment(.center)
                    
                    if isProcessingAR {
                        ProgressView()
                            .progressViewStyle(CircularProgressViewStyle(tint: .white))
                            .scaleEffect(0.8)
                    }
                    
                    if arObjectPlacementManager.isReadyToPlace {
                        Text("Tap to place furniture")
                            .font(.caption)
                            .foregroundColor(.white)
                            .opacity(0.8)
                    }
                }
                .padding(.horizontal, 20)
                .padding(.vertical, 16)
                .background(
                    RoundedRectangle(cornerRadius: 12)
                        .fill(arObjectPlacementManager.isReadyToPlace ? Color.blue.opacity(0.8) : Color.black.opacity(0.7))
                )
                .animation(.easeInOut(duration: 0.3), value: arObjectPlacementManager.isReadyToPlace)
                
                Spacer()
            }
            
            Spacer().frame(height: 120) // Space above bottom controls
        }
    }
    
    // MARK: - AR Functionality
    
    // Setup AR managers with scene references
    private func setupARManagers() {
        // Setup will be completed when SceneKitView is ready
    }
    
    // Toggle AR mode on/off
    private func toggleARMode() {
        if isARActive {
            // Stop AR mode
            stopARMode()
        } else {
            // Start AR mode
            startARMode()
        }
    }
    
    // Start AR mode and snapshot capture
    private func startARMode() {
        isARActive = true
        arStatusMessage = "Preparing AR mode..."
        
        // Start capture session (immediate for SCNView)
        scnViewCaptureManager.startSession()
        
        // Clear any existing AR objects
        arObjectPlacementManager.resetForNewSession()
        
        // Start AR processing pipeline
        Task {
            await performARCapture()
        }
    }
    
    // Stop AR mode and clean up
    private func stopARMode() {
        isARActive = false
        isProcessingAR = false
        arStatusMessage = "Point at furniture objects"
        
        // Stop capture session
        scnViewCaptureManager.stopSession()
        
        // Clear AR objects
        arObjectPlacementManager.clearAllObjects()
    }
    
    // Perform AR capture and processing
    private func performARCapture() async {
        arStatusMessage = "Positioning camera for furniture capture..."
        
        // Wait a moment for AR camera positioning to complete
        try? await Task.sleep(nanoseconds: 500_000_000) // 0.5 seconds
        
        arStatusMessage = "Capturing snapshot..."
        isProcessingAR = true
        
        // Capture snapshot from SCNView
        guard let capturedImage = await scnViewCaptureManager.capturePhoto() else {
            await MainActor.run {
                arStatusMessage = "Failed to capture snapshot"
                isProcessingAR = false
            }
            return
        }
        
        await MainActor.run {
            arStatusMessage = "Processing image..."
        }
        
        // Process image for furniture segmentation
        guard let segmentedImage = await objectSegmentationProcessor.processImage(capturedImage) else {
            await MainActor.run {
                arStatusMessage = "No furniture detected"
                isProcessingAR = false
            }
            return
        }
        
        await MainActor.run {
            // Prepare for object placement
            arObjectPlacementManager.prepareForPlacement(with: segmentedImage)
            arStatusMessage = "Ready to place furniture"
            isProcessingAR = false
        }
    }
    
    // Clean up AR resources
    private func cleanupAR() {
        scnViewCaptureManager.stopSession()
        arObjectPlacementManager.clearAllObjects()
    }
}