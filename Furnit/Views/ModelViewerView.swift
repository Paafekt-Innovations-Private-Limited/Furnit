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
    
    // Camera overlay state
    @State private var showCameraOverlay = false
    @State private var capturedOverlayImage: UIImage?
    
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
    @State private var isInContinuousMode = false

    // Object manipulation state
    @State private var showDeleteConfirmation = false
    @State private var showObjectMovementJoystick = false
    @State private var objectMovementJoystickOffset: CGSize = .zero
    
    var body: some View {
        GeometryReader { geometry in
            ZStack {
                // Main 3D room view
                RealityKitView(
                    model: model,
                    cameraMovementManager: cameraMovementManager,
                    arObjectPlacementManager: arObjectPlacementManager,
                    isARActive: isARActive
                )
                .ignoresSafeArea(.all)
                
                // Regular controls
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
                
                // Simple camera overlay - Image Plate style floating window
                if showCameraOverlay {
                    SimpleCameraOverlay(
                        capturedImage: $capturedOverlayImage,
                        isShowingCamera: $showCameraOverlay
                    )
                    .transition(.scale.combined(with: .opacity))
                    .zIndex(100)
                }
            }
        }
        .navigationBarHidden(true)
        .statusBarHidden(true)
        .preferredColorScheme(.dark)
        .onAppear {
            setupARManagers()
            ar3DModelProcessor.setQualitySettings(AppStateManager.shared.qualitySettings)
        }
        .onDisappear {
            cleanupAR()
        }
        .onChange(of: capturedOverlayImage) { _, newImage in
            if let image = newImage {
                // Handle the captured image if needed
                print("📸 Image captured from SimpleCameraOverlay")
                // You can process the image here if needed
            }
        }
        .onChange(of: shouldRestartScanning) { _, newValue in
            if newValue {
                restartARScanning()
                shouldRestartScanning = false
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
            
            // Bottom controls
            HStack(alignment: .bottom) {
                // Left side - Camera overlay button
                VStack(spacing: 16) {
                    if !arObjectPlacementManager.isManipulatingObject {
                        CameraOverlayButton {
                            showCameraOverlay.toggle()
                            // Reset captured image when opening camera
                            if showCameraOverlay {
                                capturedOverlayImage = nil
                            }
                        }
                    }
                    
                    Spacer()
                }
                
                Spacer()
                
                // Center/Right side - Info and joystick
                VStack(spacing: 16) {
                    if !isARActive && !arObjectPlacementManager.isManipulatingObject {
                        modelInfoPanel
                    }

                    // Joystick for camera movement
                    if !arObjectPlacementManager.isManipulatingObject {
                        VirtualJoystick(joystickOffset: $joystickOffset)
                            .onChange(of: joystickOffset) { _, newOffset in
                                cameraMovementManager.updateJoystickInput(newOffset)
                            }
                    }
                }
            }
            .padding()
        }
    }
    
    private var landscapeControls: some View {
        HStack {
            // Left side controls
            VStack {
                backButton
                Spacer()
            }
            .padding()
            
            Spacer()
            
            // Right side controls
            VStack(spacing: 16) {
                // Camera overlay button at top
                if !arObjectPlacementManager.isManipulatingObject {
                    CameraOverlayButton {
                        showCameraOverlay.toggle()
                        // Reset captured image when opening camera
                        if showCameraOverlay {
                            capturedOverlayImage = nil
                        }
                    }
                }

                Spacer()

                if !isARActive && !arObjectPlacementManager.isManipulatingObject {
                    modelInfoPanel
                }

                // Joystick for camera movement at bottom
                if !arObjectPlacementManager.isManipulatingObject {
                    VirtualJoystick(joystickOffset: $joystickOffset)
                        .onChange(of: joystickOffset) { _, newOffset in
                            cameraMovementManager.updateJoystickInput(newOffset)
                        }
                }
            }
            .padding()
        }
    }
    
    private var backButton: some View {
        Button(action: {
            dismiss()
        }) {
            HStack(spacing: 4) {
                Image(systemName: "chevron.left")
                    .font(.system(size: 14, weight: .semibold))
                Text("Back")
                    .font(.system(size: 16))
            }
            .foregroundColor(.white)
            .padding(.horizontal, 16)
            .padding(.vertical, 8)
            .background(Color.black.opacity(0.7))
            .cornerRadius(20)
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
            
            Spacer().frame(height: 120)
        }
    }

    // MARK: - Object Manipulation Guidance Overlay
    private var objectManipulationGuidanceOverlay: some View {
        VStack {
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

            if arObjectPlacementManager.isManipulatingObject {
                ZStack {
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

                    VStack {
                        Spacer()

                        HStack {
                            Spacer()

                            VStack(spacing: 12) {
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
                            .padding(.bottom, 100)
                        }
                    }
                    
                    if showObjectMovementJoystick {
                        VStack {
                            Spacer()

                            HStack {
                                Spacer()

                                VStack(spacing: 8) {
                                    Text("Move Object")
                                        .font(.caption)
                                        .foregroundColor(.white)
                                        .padding(.horizontal, 12)
                                        .padding(.vertical, 4)
                                        .background(
                                            RoundedRectangle(cornerRadius: 8)
                                                .fill(Color.black.opacity(0.6))
                                        )

                                    VirtualJoystick(joystickOffset: $objectMovementJoystickOffset)
                                        .onChange(of: objectMovementJoystickOffset) { _, newOffset in
                                            arObjectPlacementManager.handleObjectMovement(translation: newOffset)
                                        }
                                }

                                Spacer()
                            }
                            .padding(.bottom, 120)
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
    private func cancelManipulation() {
        showObjectMovementJoystick = false
        arObjectPlacementManager.endObjectManipulation()
        print("❌ Object manipulation cancelled by user")
    }

    private func deleteSelectedObject() {
        guard let selectedObject = arObjectPlacementManager.selectedObject else {
            print("⚠️ No object selected for deletion")
            return
        }

        print("🗑️ Starting object deletion process...")
        showObjectMovementJoystick = false
        arObjectPlacementManager.endObjectManipulation()

        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
            self.arObjectPlacementManager.removeObject(selectedObject.id)
            print("🗑️ Selected object deleted successfully")
        }
    }

    // MARK: - AR Functionality
    private func setupARManagers() {
        arObjectPlacementManager.onObjectPlaced = {
            DispatchQueue.main.async {
                self.arProcessingStateManager.reset()
                self.isProcessingAR = false
                self.isInContinuousMode = false
                self.isARActive = false
                print("✅ Object placed - AR session completed")
            }
        }
        print("✅ Set up AR managers with continuous scanning mode")
    }
    
    private func toggleARMode() {
        if isProcessingAR {
            stopARMode()
            print("📱 Red button pressed - stopping AR processing")
        } else {
            startARMode()
            print("📱 Blue button pressed - starting AR mode with automatic processing")
        }
    }
    
    private func startARMode() {
        isARActive = true
        isInContinuousMode = false
        isProcessingAR = true
        arStatusMessage = "Starting ARKit camera..."
        arProcessingStateManager.reset()
        arProcessingStateManager.startARSession()
        arObjectPlacementManager.isReadyToPlace = false

        Task {
            await MainActor.run {
                arStatusMessage = "Starting capture for next object..."
                arProcessingStateManager.updateState(.pointing)
                print("🎯 Lightweight AR restart - starting capture process immediately")
            }
            await performARCapture()
        }
    }
    
    private func stopARMode() {
        isARActive = false
        isProcessingAR = false
        isInContinuousMode = false
        arStatusMessage = "Point at furniture objects"
        arProcessingStateManager.reset()
        arKitCameraManager.stopSession()
        arObjectPlacementManager.clearAllObjects()
        arObjectPlacementManager.isReadyToPlace = false
        qrCodeDetectionService.reset()
        ar3DModelProcessor.reset()
        print("🛑 AR mode stopped completely with full cleanup")
    }

    private func restartARScanning() {
        isProcessingAR = false
        arStatusMessage = "Point at furniture objects"
        isARActive = true
        isInContinuousMode = true
        arProcessingStateManager.startARSession()
        arKitCameraManager.startSession()
        arObjectPlacementManager.isReadyToPlace = false
        print("🔄 Restarted AR scanning mode with camera session - ready for next object")
    }
    
    private func performARCapture() async {
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

        if !arKitCameraManager.isSessionRunning {
            try? await Task.sleep(nanoseconds: 1_000_000_000)
        }
        
        try? await Task.sleep(nanoseconds: 2_000_000_000)
        
        await MainActor.run {
            arStatusMessage = "Capturing ARKit frame..."
            arProcessingStateManager.beginCapture()
            isProcessingAR = true
        }
        
        guard let capturedImage = await MainActor.run(body: { arKitCameraManager.captureCurrentFrameForAPI() }) else {
            await MainActor.run {
                let errorMsg = arKitCameraManager.errorMessage ?? "Failed to capture frame"
                arStatusMessage = errorMsg
                arProcessingStateManager.setError(errorMsg)
                isProcessingAR = false
                arKitCameraManager.stopSession()
                print("📷 Physical camera session stopped due to capture error")
            }
            return
        }

        await MainActor.run {
            arKitCameraManager.stopSession()
            print("📷 Physical camera session stopped immediately after image capture")
        }
        
        await MainActor.run {
            arStatusMessage = "Scanning for QR codes..."
            arProcessingStateManager.beginQRDetection()
        }

        let qrResult = await qrCodeDetectionService.detectQRCode(in: capturedImage)

        if qrResult.hasQRCode, let qrURL = qrResult.extractedURL {
            print("🔍 QR code detected with URL: \(qrURL.absoluteString)")

            await MainActor.run {
                arStatusMessage = "QR code found! Downloading 3D model..."
                arProcessingStateManager.beginAssetDownload()
            }

            if await qrCodeDetectionService.isValid3DAssetURL(qrURL) {
                let progressCancellable = assetDownloadService.$downloadProgress
                    .sink { progress in
                        Task { @MainActor in
                            arProcessingStateManager.updateAssetDownloadProgress(progress)
                        }
                    }

                let downloadResult = await assetDownloadService.download3DAsset(from: qrURL)
                progressCancellable.cancel()

                if downloadResult.success, let downloadedEntity = downloadResult.entity {
                    await MainActor.run {
                        arObjectPlacementManager.prepareForPlacement(with3DModel: downloadedEntity)
                        arStatusMessage = "3D model ready! Tap to place"
                        arProcessingStateManager.readyForPlacement()
                        isProcessingAR = false
                        print("✅ QR asset download completed, ready for placement")
                    }

                    if let fileURL = downloadResult.fileURL {
                        assetDownloadService.cleanupDownloadedFile(at: fileURL)
                    }

                    print("✅ QR code workflow completed successfully")
                    return

                } else {
                    let errorMsg = downloadResult.errorMessage ?? "Failed to download 3D model"
                    print("⚠️ QR asset download failed: \(errorMsg). Falling back to backend processing.")

                    await MainActor.run {
                        arStatusMessage = "Download failed, generating 3D model..."
                    }
                }
            } else {
                print("⚠️ QR URL is not a 3D asset. Falling back to backend processing.")

                await MainActor.run {
                    arStatusMessage = "QR code found but not a 3D asset, generating model..."
                }
            }
        } else {
            print("📱 No QR code detected, proceeding with backend 3D generation")

            await MainActor.run {
                arStatusMessage = "No QR code found, generating 3D model..."
            }
        }

        await MainActor.run {
            arStatusMessage = "Uploading image for 3D generation..."
            arProcessingStateManager.beginUpload()
        }

        guard let generated3DModel = await ar3DModelProcessor.processImage(capturedImage) else {
            await MainActor.run {
                let errorMsg = ar3DModelProcessor.errorMessage ?? "3D generation failed"
                arStatusMessage = errorMsg
                arProcessingStateManager.setError(errorMsg)
                isProcessingAR = false
                print("❌ 3D model generation failed")
            }
            return
        }
        
        await MainActor.run {
            arObjectPlacementManager.prepareForPlacement(with3DModel: generated3DModel)
            arStatusMessage = "Tap to place 3D model in room"
            arProcessingStateManager.readyForPlacement()
            isProcessingAR = false
            print("✅ 3D model generation completed, ready for placement")
        }
    }
    
    private func cleanupAR() {
        arKitCameraManager.stopSession()
        isARActive = false
        isProcessingAR = false
        isInContinuousMode = false
        arObjectPlacementManager.clearAllObjects()
        arObjectPlacementManager.isReadyToPlace = false
        arProcessingStateManager.reset()
        ar3DModelProcessor.reset()
        qrCodeDetectionService.reset()
        print("🧹 Complete AR cleanup performed on view dismissal")
    }
}

// Camera overlay button
struct CameraOverlayButton: View {
    let action: () -> Void
    
    var body: some View {
        Button(action: action) {
            Image(systemName: "camera.viewfinder")
                .font(.title2)
                .foregroundColor(.white)
                .frame(width: 60, height: 60)
                .background(
                    LinearGradient(
                        colors: [Color.purple, Color.blue],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    )
                )
                .clipShape(Circle())
                .shadow(color: .black.opacity(0.3), radius: 4, x: 0, y: 2)
                .overlay(
                    Circle()
                        .stroke(Color.white.opacity(0.3), lineWidth: 1)
                )
        }
    }
}
