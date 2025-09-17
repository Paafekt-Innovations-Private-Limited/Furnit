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
    
    // AR functionality state
    @StateObject private var arKitCameraManager = ARKitCameraManager()
    @StateObject private var ar3DModelProcessor = AR3DModelProcessor()
    @StateObject private var arProcessingStateManager = ARProcessingStateManager()
    @StateObject private var arObjectPlacementManager = RealityKitObjectPlacementManager()
    @State private var isARActive = false
    @State private var arStatusMessage = "Point at furniture objects"
    @State private var isProcessingAR = false
    
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
            // Configure 3D model processor with quality settings
            ar3DModelProcessor.setQualitySettings(AppStateManager.shared.qualitySettings)
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
                    VStack(spacing: 4) {
                        Text(arProcessingStateManager.currentState.displayMessage)
                            .font(.headline)
                            .foregroundColor(.white)
                            .multilineTextAlignment(.center)
                        
                        if let secondaryMessage = arProcessingStateManager.currentState.secondaryMessage {
                            Text(secondaryMessage)
                                .font(.caption)
                                .foregroundColor(.white.opacity(0.8))
                                .multilineTextAlignment(.center)
                        }
                    }
                    
                    if arProcessingStateManager.currentState.showsProgress {
                        VStack(spacing: 8) {
                            ProgressView(value: arProcessingStateManager.currentState.progressValue)
                                .progressViewStyle(LinearProgressViewStyle(tint: .white))
                                .frame(width: 120, height: 4)
                            
                            if arProcessingStateManager.currentState.isProcessing {
                                HStack(spacing: 4) {
                                    ProgressView()
                                        .progressViewStyle(CircularProgressViewStyle(tint: .white))
                                        .scaleEffect(0.6)
                                    
                                    let elapsed = arProcessingStateManager.formattedElapsedTime
                                    if !elapsed.isEmpty {
                                        Text(elapsed)
                                            .font(.caption2)
                                            .foregroundColor(.white.opacity(0.7))
                                    }
                                }
                            }
                        }
                    }
                    
                    if arProcessingStateManager.currentState.isReadyToPlace {
                        HStack(spacing: 8) {
                            Image(systemName: "hand.tap")
                                .foregroundColor(.white)
                                .font(.caption)
                            
                            Text("Tap anywhere to place")
                                .font(.caption)
                                .foregroundColor(.white)
                                .opacity(0.8)
                        }
                        .animation(.easeInOut(duration: 1.0).repeatForever(autoreverses: true), value: arProcessingStateManager.currentState)
                    }
                }
                .padding(.horizontal, 20)
                .padding(.vertical, 16)
                .background(
                    RoundedRectangle(cornerRadius: 12)
                        .fill(arProcessingStateManager.currentState.statusColor.opacity(0.8))
                )
                .animation(.easeInOut(duration: 0.3), value: arProcessingStateManager.currentState)
                
                Spacer()
            }
            
            Spacer().frame(height: 120) // Space above bottom controls
        }
    }
    
    // MARK: - AR Functionality
    
    // Setup AR managers with scene references
    private func setupARManagers() {
        // Setup will be completed when RealityKitView is ready
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
    
    // Start AR mode and ARKit camera session
    private func startARMode() {
        isARActive = true
        arStatusMessage = "Starting ARKit camera..."
        
        // Initialize AR processing state
        arProcessingStateManager.reset()
        arProcessingStateManager.startARSession()
        
        // Start ARKit session for camera frame access
        arKitCameraManager.startSession()
        
        // Clear any existing AR objects
        arObjectPlacementManager.resetForNewSession()
        
        // Wait for ARKit to initialize, then begin AR processing
        Task {
            // Give ARKit time to initialize and start receiving frames
            try? await Task.sleep(nanoseconds: 1_500_000_000) // 1.5 seconds
            
            await MainActor.run {
                arStatusMessage = "Ready - tap to capture furniture"
                arProcessingStateManager.updateState(.pointing)
            }
            
            // Start AR processing pipeline
            await performARCapture()
        }
    }
    
    // Stop AR mode and clean up
    private func stopARMode() {
        isARActive = false
        isProcessingAR = false
        arStatusMessage = "Point at furniture objects"
        
        // Reset AR processing state
        arProcessingStateManager.reset()
        
        // Stop ARKit session
        arKitCameraManager.stopSession()
        
        // Clear AR objects
        arObjectPlacementManager.clearAllObjects()
    }
    
    // Perform AR capture and processing using backend API
    private func performARCapture() async {
        await MainActor.run {
            arStatusMessage = "Ready - tap to capture furniture"
            arProcessingStateManager.updateState(.pointing)
        }
        
        // Wait for user to tap (this will be triggered by AR button or gesture)
        // For now, automatically capture after a short delay for testing
        try? await Task.sleep(nanoseconds: 2_000_000_000) // 2 seconds
        
        await MainActor.run {
            arStatusMessage = "Capturing ARKit frame..."
            arProcessingStateManager.beginCapture()
            isProcessingAR = true
        }
        
        // Capture current frame from ARKit session (full resolution for API)
        guard let capturedImage = await MainActor.run(body: { arKitCameraManager.captureCurrentFrameForAPI() }) else {
            await MainActor.run {
                let errorMsg = arKitCameraManager.errorMessage ?? "Failed to capture frame"
                arStatusMessage = errorMsg
                arProcessingStateManager.setError(errorMsg)
                isProcessingAR = false
            }
            return
        }
        
        await MainActor.run {
            arStatusMessage = "Uploading image for 3D generation..."
            arProcessingStateManager.beginUpload()
        }
        
        // Process image using backend API for 3D model generation
        guard let generated3DModel = await ar3DModelProcessor.processImage(capturedImage) else {
            await MainActor.run {
                let errorMsg = ar3DModelProcessor.errorMessage ?? "3D generation failed"
                arStatusMessage = errorMsg
                arProcessingStateManager.setError(errorMsg)
                isProcessingAR = false
            }
            return
        }
        
        await MainActor.run {
            // Prepare for object placement in 3D scene with generated model
            arObjectPlacementManager.prepareForPlacement(with3DModel: generated3DModel)
            arStatusMessage = "Tap to place 3D model in room"
            arProcessingStateManager.readyForPlacement()
            isProcessingAR = false
        }
    }
    
    // Clean up AR resources
    private func cleanupAR() {
        arKitCameraManager.stopSession()
        arObjectPlacementManager.clearAllObjects()
        arProcessingStateManager.reset()
        ar3DModelProcessor.reset()
    }
}