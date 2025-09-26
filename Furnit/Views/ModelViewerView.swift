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
    @StateObject private var qrCodeDetectionService = QRCodeDetectionService()
    @StateObject private var assetDownloadService = AssetDownloadService()
    @State private var isARActive = false
    @State private var arStatusMessage = "Point at furniture objects"
    @State private var isProcessingAR = false
    @State private var shouldRestartScanning = false
    @State private var isInContinuousMode = false // Track if we're in continuous scanning mode

    // Object manipulation state
    @State private var showDeleteConfirmation = false
    @State private var showObjectMovementJoystick = false
    @State private var objectMovementJoystickOffset: CGSize = .zero

    private var gestureHandlers: RealityKitGestureHandlers? {
        // Access gesture handlers through the coordinator
        return nil // TODO: We'll need to access this through RealityKitView
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
                
                if isLandscape(geometry: geometry) {
                    landscapeControls
                } else {
                    portraitControls
                }
                
                // AR status overlay
                if isARActive {
                    arStatusOverlay
                }

                // Object manipulation guidance overlay
                objectManipulationGuidanceOverlay
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
        .onChange(of: shouldRestartScanning) { _, newValue in
            if newValue {
                restartARScanning()
                shouldRestartScanning = false // Reset the trigger
            }
        }
        .alert("Delete Object", isPresented: $showDeleteConfirmation) {
            Button("Delete", role: .destructive) {
                deleteSelectedObject()
            }
            Button("Cancel", role: .cancel) { }
        } message: {
            Text("Are you sure you want to delete this object? This action cannot be undone.")
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
            
            // Bottom controls - info, joystick, and AR button (hidden during object manipulation)
            HStack {
                VStack(spacing: 16) {
                    // AR Button at bottom-left (hidden during object manipulation)
                    if !arObjectPlacementManager.isManipulatingObject {
                        ARButton(isARActive: Binding(
                            get: {
                                // Show red (active) only when processing, blue when ready to start
                                isProcessingAR
                            },
                            set: { _ in toggleARMode() }
                        )) {
                            toggleARMode()
                        }
                    }
                }

                VStack(spacing: 16) {
                    if !isARActive {
                        modelInfoPanel
                    }

                    // Joystick for camera movement (hidden during object manipulation)
                    if !arObjectPlacementManager.isManipulatingObject {
                        VirtualJoystick(joystickOffset: $joystickOffset)
                            .onChange(of: joystickOffset) { _, newOffset in
                                cameraMovementManager.updateJoystickInput(newOffset)
                            }
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
            
            // Right side controls - info, joystick, and AR button (hidden during object manipulation)
            VStack(spacing: 16) {
                // AR Button at bottom-left (moved to right side in landscape, hidden during object manipulation)
                if !arObjectPlacementManager.isManipulatingObject {
                    ARButton(isARActive: Binding(
                        get: {
                            // Show red (active) only when processing, blue when ready to start
                            isProcessingAR
                        },
                        set: { _ in toggleARMode() }
                    )) {
                        toggleARMode()
                    }
                }

                Spacer()

                if !isARActive {
                    modelInfoPanel
                }

                // Joystick for camera movement (right side in landscape, hidden during object manipulation)
                if !arObjectPlacementManager.isManipulatingObject {
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

    // MARK: - Object Manipulation Guidance Overlay
    private var objectManipulationGuidanceOverlay: some View {
        VStack {
            // Show guidance when objects are present but not being manipulated
            if !arObjectPlacementManager.placedObjects.isEmpty && !arObjectPlacementManager.isManipulatingObject && !isARActive {
                HStack {
                    Spacer()

                    VStack(spacing: 8) {
                        HStack(spacing: 8) {
                            Image(systemName: "hand.tap.fill")
                                .foregroundColor(.white)
                                .font(.caption)

                            Text("Long press on object to manipulate")
                                .font(.caption)
                                .foregroundColor(.white)
                        }
                    }
                    .padding(.horizontal, 16)
                    .padding(.vertical, 10)
                    .background(
                        RoundedRectangle(cornerRadius: 8)
                            .fill(Color.blue.opacity(0.8))
                    )
                    .animation(.easeInOut(duration: 2.0).repeatForever(autoreverses: true), value: arObjectPlacementManager.placedObjects.count)

                    Spacer()
                }
                .transition(.opacity)
            }

            // Show manipulation instructions and control buttons when manipulating object
            if arObjectPlacementManager.isManipulatingObject {
                ZStack {
                    // Main instruction overlay (top center)
                    VStack {
                        HStack {
                            Spacer()

                            VStack(spacing: 12) {
                                HStack(spacing: 8) {
                                    Image(systemName: "hand.point.up.left.fill")
                                        .foregroundColor(.white)
                                        .font(.headline)

                                    Text("Object Selected")
                                        .font(.headline)
                                        .foregroundColor(.white)
                                        .fontWeight(.semibold)
                                }

                                VStack(spacing: 6) {
                                    HStack(spacing: 8) {
                                        Image(systemName: "arrow.left.and.right")
                                            .foregroundColor(.white.opacity(0.8))
                                            .font(.caption)

                                        Text("Swipe horizontally to rotate")
                                            .font(.caption)
                                            .foregroundColor(.white.opacity(0.8))
                                    }
                                }
                            }
                            .padding(.horizontal, 20)
                            .padding(.vertical, 16)
                            .background(
                                RoundedRectangle(cornerRadius: 12)
                                    .fill(Color.green.opacity(0.9))
                            )

                            Spacer()
                        }

                        Spacer()
                    }

                    // Control buttons (bottom right)
                    VStack {
                        Spacer()

                        HStack {
                            Spacer()

                            VStack(spacing: 12) {
                                // Show Controls Button (toggles object movement joystick)
                                Button(action: {
                                    withAnimation(.easeInOut(duration: 0.3)) {
                                        showObjectMovementJoystick.toggle()
                                    }
                                    print("🎮 Object movement joystick: \(showObjectMovementJoystick ? "SHOWN" : "HIDDEN")")
                                }) {
                                    Image(systemName: showObjectMovementJoystick ? "xmark" : "gamecontroller.fill")
                                        .font(.system(size: 20))
                                        .foregroundColor(.white)
                                        .frame(width: 50, height: 50)
                                        .background(showObjectMovementJoystick ? Color.blue.opacity(0.8) : Color.black.opacity(0.7))
                                        .clipShape(Circle())
                                        .overlay(
                                            Circle()
                                                .stroke(Color.white.opacity(0.3), lineWidth: 1)
                                        )
                                        .animation(.easeInOut(duration: 0.2), value: showObjectMovementJoystick)
                                }

                                // Cancel Button
                                Button(action: {
                                    cancelManipulation()
                                }) {
                                    Image(systemName: "xmark")
                                        .font(.system(size: 20, weight: .bold))
                                        .foregroundColor(.white)
                                        .frame(width: 50, height: 50)
                                        .background(Color.black.opacity(0.7))
                                        .clipShape(Circle())
                                        .overlay(
                                            Circle()
                                                .stroke(Color.white.opacity(0.3), lineWidth: 1)
                                        )
                                }

                                // Delete Button
                                Button(action: {
                                    showDeleteConfirmation = true
                                }) {
                                    Image(systemName: "trash.fill")
                                        .font(.system(size: 20))
                                        .foregroundColor(.white)
                                        .frame(width: 50, height: 50)
                                        .background(Color.red.opacity(0.8))
                                        .clipShape(Circle())
                                        .overlay(
                                            Circle()
                                                .stroke(Color.white.opacity(0.3), lineWidth: 1)
                                        )
                                }
                            }
                            .padding(.trailing, 20)
                            .padding(.bottom, 100) // Space above the bottom UI elements
                        }
                    }
                    // Object movement joystick (bottom center when controls are active)
                    if showObjectMovementJoystick {
                        VStack {
                            Spacer()

                            HStack {
                                Spacer()

                                VStack(spacing: 8) {
                                    // Joystick title/instruction
                                    Text("Move Object")
                                        .font(.caption)
                                        .foregroundColor(.white)
                                        .padding(.horizontal, 12)
                                        .padding(.vertical, 4)
                                        .background(
                                            RoundedRectangle(cornerRadius: 8)
                                                .fill(Color.black.opacity(0.6))
                                        )

                                    // The actual joystick for object movement
                                    VirtualJoystick(joystickOffset: $objectMovementJoystickOffset)
                                        .onChange(of: objectMovementJoystickOffset) { _, newOffset in
                                            // Connect joystick input to object movement
                                            arObjectPlacementManager.handleObjectMovement(translation: newOffset)
                                        }
                                }

                                Spacer()
                            }
                            .padding(.bottom, 120) // Space above bottom UI elements
                        }
                        .transition(.move(edge: .bottom).combined(with: .opacity))
                        .animation(.easeInOut(duration: 0.3), value: showObjectMovementJoystick)
                    }
                }
                .transition(.scale.combined(with: .opacity))
                .animation(.easeInOut(duration: 0.3), value: arObjectPlacementManager.isManipulatingObject)
            }

            Spacer()
        }
    }

    // MARK: - Object Manipulation Actions

    // Cancel object manipulation mode
    private func cancelManipulation() {
        // Hide movement joystick if it's showing
        showObjectMovementJoystick = false

        // For now, directly call the object placement manager
        // In the future, we could access gesture handlers through RealityKitView
        arObjectPlacementManager.endObjectManipulation()
        print("❌ Object manipulation cancelled by user")
    }

    // Delete the currently selected object
    private func deleteSelectedObject() {
        guard let selectedObject = arObjectPlacementManager.selectedObject else {
            print("⚠️ No object selected for deletion")
            return
        }

        print("🗑️ Starting object deletion process...")

        // Hide movement joystick if it's showing
        showObjectMovementJoystick = false

        // End manipulation mode FIRST to clean up UI state
        arObjectPlacementManager.endObjectManipulation()

        // Small delay to ensure UI state is updated before removing object
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
            // Remove the object from the placement manager
            self.arObjectPlacementManager.removeObject(selectedObject.id)

            print("🗑️ Selected object deleted successfully")
        }
    }

    // MARK: - AR Functionality
    
    // Setup AR managers with scene references
    private func setupARManagers() {
        // Setup callback for object placement to clean up UI state only
        arObjectPlacementManager.onObjectPlaced = {
            DispatchQueue.main.async {
                // Reset AR processing state to end current session
                self.arProcessingStateManager.reset()

                // Reset UI state flags to end AR mode
                self.isProcessingAR = false
                self.isInContinuousMode = false
                self.isARActive = false

                print("✅ Object placed - AR session completed, user can manually start next scan")
            }
        }
        print("✅ Set up AR managers with continuous scanning mode")
    }
    
    // Simple toggle: Blue button starts everything, Red button stops everything
    private func toggleARMode() {
        if isProcessingAR {
            // Red button pressed - stop/cancel everything
            stopARMode()
            print("📱 Red button pressed - stopping AR processing")
        } else {
            // Blue button pressed - start AR mode (which auto-starts processing)
            startARMode()
            print("📱 Blue button pressed - starting AR mode with automatic processing")
        }
    }
    
    // Start AR mode and automatically begin capture process
    private func startARMode() {
        isARActive = true
        isInContinuousMode = false // Reset continuous mode when starting fresh
        isProcessingAR = true // Immediately show red button (stop available)
        arStatusMessage = "Starting ARKit camera..."

        // Initialize AR processing state
        arProcessingStateManager.reset()
        arProcessingStateManager.startARSession()

        // Keep existing placed objects visible - don't clear them
        // Only reset placement state for new capture, not existing objects
        arObjectPlacementManager.isReadyToPlace = false

        // Lightweight restart - directly start capture process without camera session delays
        Task {
            await MainActor.run {
                arStatusMessage = "Starting capture for next object..."
                arProcessingStateManager.updateState(.pointing)
                print("🎯 Lightweight AR restart - starting capture process immediately")
            }

            // Directly trigger capture process (camera will start if needed)
            await performARCapture()
        }
    }
    
    // Stop AR mode and clean up
    private func stopARMode() {
        isARActive = false
        isProcessingAR = false
        isInContinuousMode = false // Reset continuous mode when stopping
        arStatusMessage = "Point at furniture objects"

        // Reset AR processing state
        arProcessingStateManager.reset()

        // Stop ARKit session with explicit cleanup
        arKitCameraManager.stopSession()

        // Clear AR objects and reset placement state
        arObjectPlacementManager.clearAllObjects()
        arObjectPlacementManager.isReadyToPlace = false

        // Reset any ongoing AR services
        qrCodeDetectionService.reset()
        ar3DModelProcessor.reset()

        print("🛑 AR mode stopped completely with full cleanup")
    }

    // Restart AR scanning mode after object placement
    private func restartARScanning() {
        // Reset processing state to scanning mode
        isProcessingAR = false
        arStatusMessage = "Point at furniture objects"

        // Restart AR session since it was stopped after capture
        isARActive = true
        isInContinuousMode = true

        // Reset AR processing state to pointing (ready for next scan)
        arProcessingStateManager.startARSession()

        // Restart ARKit camera session for next capture
        arKitCameraManager.startSession()

        // Reset object placement manager for next object (keep placed objects)
        arObjectPlacementManager.isReadyToPlace = false

        print("🔄 Restarted AR scanning mode with camera session - ready for next object")
        print("📱 Button shows camera icon for continuous scanning")
    }
    
    // Perform AR capture and processing using backend API
    private func performARCapture() async {
        // Start camera session if not already running (for subsequent captures)
        await MainActor.run {
            if !arKitCameraManager.isSessionRunning {
                arStatusMessage = "Starting camera for capture..."
                arKitCameraManager.startSession()
                print("📷 Started camera session for capture")
            } else {
                arStatusMessage = "Ready - tap to capture furniture"
            }
            arProcessingStateManager.updateState(.pointing)
        }

        // Give camera a moment to initialize if just started
        if !arKitCameraManager.isSessionRunning {
            try? await Task.sleep(nanoseconds: 1_000_000_000) // 1 second
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

                // Stop physical camera session on capture error to prevent resource waste
                arKitCameraManager.stopSession()
                print("📷 Physical camera session stopped due to capture error")
            }
            return
        }

        // Stop physical camera session immediately after successful image capture
        // This prevents unnecessary battery drain during processing
        await MainActor.run {
            arKitCameraManager.stopSession()
            print("📷 Physical camera session stopped immediately after image capture")
        }
        
        // First, check for QR codes in the captured image
        await MainActor.run {
            arStatusMessage = "Scanning for QR codes..."
            arProcessingStateManager.beginQRDetection()
        }

        let qrResult = await qrCodeDetectionService.detectQRCode(in: capturedImage)

        if qrResult.hasQRCode, let qrURL = qrResult.extractedURL {
            // QR code found with valid URL - download asset directly
            print("🔍 QR code detected with URL: \(qrURL.absoluteString)")

            await MainActor.run {
                arStatusMessage = "QR code found! Downloading 3D model..."
                arProcessingStateManager.beginAssetDownload()
            }

            // Validate it's a 3D asset URL using enhanced validation
            if await qrCodeDetectionService.isValid3DAssetURL(qrURL) {
                // Monitor download progress
                let progressCancellable = assetDownloadService.$downloadProgress
                    .sink { progress in
                        Task { @MainActor in
                            arProcessingStateManager.updateAssetDownloadProgress(progress)
                        }
                    }

                // Download the 3D asset
                let downloadResult = await assetDownloadService.download3DAsset(from: qrURL)

                // Cancel progress monitoring
                progressCancellable.cancel()

                if downloadResult.success, let downloadedEntity = downloadResult.entity {
                    // Successfully downloaded 3D asset
                    await MainActor.run {
                        arObjectPlacementManager.prepareForPlacement(with3DModel: downloadedEntity)
                        arStatusMessage = "3D model ready! Tap to place"
                        arProcessingStateManager.readyForPlacement()
                        isProcessingAR = false

                        // Camera session was already stopped after capture
                        print("✅ QR asset download completed, ready for placement")
                    }

                    // Clean up downloaded file if needed
                    if let fileURL = downloadResult.fileURL {
                        assetDownloadService.cleanupDownloadedFile(at: fileURL)
                    }

                    print("✅ QR code workflow completed successfully")
                    return

                } else {
                    // Download failed, fall back to backend processing
                    let errorMsg = downloadResult.errorMessage ?? "Failed to download 3D model"
                    print("⚠️ QR asset download failed: \(errorMsg). Falling back to backend processing.")

                    await MainActor.run {
                        arStatusMessage = "Download failed, generating 3D model..."
                    }
                }
            } else {
                // Not a 3D asset URL, fall back to backend processing
                print("⚠️ QR URL is not a 3D asset. Falling back to backend processing.")

                await MainActor.run {
                    arStatusMessage = "QR code found but not a 3D asset, generating model..."
                }
            }
        } else {
            // No QR code found, proceed with backend processing
            print("📱 No QR code detected, proceeding with backend 3D generation")

            await MainActor.run {
                arStatusMessage = "No QR code found, generating 3D model..."
            }
        }

        // Backend processing workflow (original flow)
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

                // Camera session was already stopped after capture
                print("❌ 3D model generation failed")
            }
            return
        }
        
        await MainActor.run {
            // Prepare for object placement in 3D scene with generated model
            arObjectPlacementManager.prepareForPlacement(with3DModel: generated3DModel)
            arStatusMessage = "Tap to place 3D model in room"
            arProcessingStateManager.readyForPlacement()
            isProcessingAR = false

            // Camera session was already stopped after capture
            print("✅ 3D model generation completed, ready for placement")
        }
    }
    
    // Clean up AR resources
    private func cleanupAR() {
        // Ensure camera session is completely stopped
        arKitCameraManager.stopSession()

        // Reset all AR-related state
        isARActive = false
        isProcessingAR = false
        isInContinuousMode = false

        // Clear AR objects and reset managers
        arObjectPlacementManager.clearAllObjects()
        arObjectPlacementManager.isReadyToPlace = false
        arProcessingStateManager.reset()
        ar3DModelProcessor.reset()
        qrCodeDetectionService.reset()

        print("🧹 Complete AR cleanup performed on view dismissal")
    }
}