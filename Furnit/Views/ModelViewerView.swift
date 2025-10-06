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
    
    // Camera preview state (using your existing SimpleCameraOverlay)
    @State private var showingCameraPreview = false
    @State private var capturedImageFromPreview: UIImage?
    @State private var isInitializingCamera = false
    @State private var cameraInitProgress: Double = 0.0
    @State private var initializationTimer: Timer?
    
    // AR functionality state (SimpleCameraOverlay handles camera)
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

    private var gestureHandlers: RealityKitGestureHandlers? {
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
                
                // Camera initialization overlay
                if isInitializingCamera {
                    cameraInitializationOverlay
                }
                
                // Use your existing SimpleCameraOverlay when AR is active
                if showingCameraPreview {
                    SimpleCameraOverlay(
                        capturedImage: $capturedImageFromPreview,
                        isShowingCamera: $showingCameraPreview
                    )
                    .zIndex(100) // Ensure it's on top
                    .transition(.opacity)
                }
                
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
            ar3DModelProcessor.setQualitySettings(AppStateManager.shared.qualitySettings)
        }
        .onDisappear {
            cleanupAR()
        }
        .onChange(of: shouldRestartScanning) { _, newValue in
            if newValue {
                restartARScanning()
                shouldRestartScanning = false
            }
        }
        .onChange(of: capturedImageFromPreview) { _, newImage in
            if let image = newImage {
                // Process the captured/segmented image from SimpleCameraOverlay
                Task {
                    await performARProcessingWithSegmentedImage(image)
                }
                // Reset for next capture
                capturedImageFromPreview = nil
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
            
            // Bottom controls - info, joystick, and AR button
            HStack {
                VStack(spacing: 16) {
                    // AR Button at bottom-left
                    if !arObjectPlacementManager.isManipulatingObject {
                        ARButton(isARActive: Binding(
                            get: { showingCameraPreview },  // Show active state when camera preview is showing
                            set: { _ in toggleARMode() }
                        )) {
                            toggleARMode()
                        }
                    }
                }

                VStack(spacing: 16) {
                    if !isARActive && !arObjectPlacementManager.isManipulatingObject {
                        modelInfoPanel
                    }

                    // Joystick for camera movement (NO AMPLIFICATION)
                    if !arObjectPlacementManager.isManipulatingObject {
                        VirtualJoystick(joystickOffset: $joystickOffset)
                            .onChange(of: joystickOffset) { _, newOffset in
                                // Direct input without amplification
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
            
            // Right side controls
            VStack(spacing: 16) {
                // AR Button at top
                if !arObjectPlacementManager.isManipulatingObject {
                    ARButton(isARActive: Binding(
                        get: { showingCameraPreview },  // Show active state when camera preview is showing
                        set: { _ in toggleARMode() }
                    )) {
                        toggleARMode()
                    }
                }

                Spacer()

                if !isARActive && !arObjectPlacementManager.isManipulatingObject {
                    modelInfoPanel
                }

                // Joystick with visual feedback (NO AMPLIFICATION)
                if !arObjectPlacementManager.isManipulatingObject {
                    VStack(spacing: 4) {
                        // Show movement indicator
                        if joystickOffset != .zero {
                            Text("Moving")
                                .font(.caption2)
                                .foregroundColor(.white.opacity(0.6))
                                .animation(.easeInOut(duration: 0.2), value: joystickOffset)
                        }
                        
                        VirtualJoystick(joystickOffset: $joystickOffset)
                            .onChange(of: joystickOffset) { _, newOffset in
                                // Direct input without amplification - fixed in camera manager
                                cameraMovementManager.updateJoystickInput(newOffset)
                            }
                            .frame(width: 120, height: 120)
                            .overlay(
                                Circle()
                                    .stroke(joystickOffset != .zero ? Color.blue : Color.white.opacity(0.3), lineWidth: 2)
                                    .animation(.easeInOut(duration: 0.2), value: joystickOffset)
                            )
                    }
                }

                Spacer()
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
    
    // MARK: - Camera Initialization Overlay
    private var cameraInitializationOverlay: some View {
        ZStack {
            // Semi-transparent background
            Color.black.opacity(0.7)
                .ignoresSafeArea()
                .onTapGesture {
                    // Allow canceling by tapping outside
                    cancelCameraInitialization()
                }
            
            VStack(spacing: 24) {
                // Camera icon with pulse animation
                Image(systemName: "camera.fill")
                    .font(.system(size: 50))
                    .foregroundColor(.white)
                    .scaleEffect(isInitializingCamera ? 1.1 : 1.0)
                    .animation(
                        .easeInOut(duration: 0.8).repeatForever(autoreverses: true),
                        value: isInitializingCamera
                    )
                
                VStack(spacing: 12) {
                    Text("Initializing Camera")
                        .font(.headline)
                        .foregroundColor(.white)
                    
                    Text("Starting U²-Net segmentation...")
                        .font(.caption)
                        .foregroundColor(.white.opacity(0.8))
                }
                
                // Progress bar
                VStack(spacing: 8) {
                    ProgressView(value: cameraInitProgress)
                        .progressViewStyle(LinearProgressViewStyle(tint: .white))
                        .frame(width: 200)
                        .scaleEffect(x: 1, y: 2)
                    
                    Text("\(Int(cameraInitProgress * 100))%")
                        .font(.caption2)
                        .foregroundColor(.white.opacity(0.7))
                }
                
                // Loading indicators
                HStack(spacing: 8) {
                    ForEach(0..<3, id: \.self) { index in
                        Circle()
                            .fill(Color.white)
                            .frame(width: 8, height: 8)
                            .opacity(isInitializingCamera ? 1.0 : 0.3)
                            .animation(
                                .easeInOut(duration: 0.6)
                                    .repeatForever(autoreverses: true)
                                    .delay(Double(index) * 0.2),
                                value: isInitializingCamera
                            )
                    }
                }
                
                // Cancel button
                Button(action: {
                    cancelCameraInitialization()
                }) {
                    Text("Cancel")
                        .font(.caption)
                        .foregroundColor(.white.opacity(0.8))
                        .padding(.horizontal, 20)
                        .padding(.vertical, 8)
                        .background(
                            Capsule()
                                .stroke(Color.white.opacity(0.3), lineWidth: 1)
                        )
                }
                .padding(.top, 8)
            }
            .padding(32)
            .background(
                RoundedRectangle(cornerRadius: 20)
                    .fill(Color.black.opacity(0.9))
                    .overlay(
                        RoundedRectangle(cornerRadius: 20)
                            .stroke(Color.white.opacity(0.2), lineWidth: 1)
                    )
            )
            .onTapGesture {
                // Prevent closing when tapping on the dialog
            }
        }
        .transition(.opacity)
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
        if showingCameraPreview {
            // Stop camera preview and processing
            showingCameraPreview = false
            stopARMode()
            print("📱 Camera preview stopped")
        } else {
            // Start camera initialization with progress
            startCameraInitialization()
        }
    }
    
    private func startCameraInitialization() {
        isInitializingCamera = true
        cameraInitProgress = 0.0
        isARActive = true
        
        // Simulate camera initialization with progress updates
        initializationTimer = Timer.scheduledTimer(withTimeInterval: 0.05, repeats: true) { timer in
            self.cameraInitProgress += 0.02
            
            // Add progress milestones
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
                
                // Complete initialization
                withAnimation(.easeOut(duration: 0.3)) {
                    self.isInitializingCamera = false
                }
                
                // Show camera overlay after a brief delay
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
                    self.showingCameraPreview = true
                    self.arStatusMessage = "Live furniture segmentation"
                    print("📷 SimpleCameraOverlay activated with U²-Net segmentation")
                }
            }
        }
    }
    
    private func cancelCameraInitialization() {
        // Stop the timer
        initializationTimer?.invalidate()
        initializationTimer = nil
        
        // Reset states
        withAnimation(.easeOut(duration: 0.2)) {
            isInitializingCamera = false
            cameraInitProgress = 0.0
            isARActive = false
        }
        
        print("❌ Camera initialization cancelled")
    }
    
    private func startARMode() {
        // This can be called directly if needed
        startCameraInitialization()
    }
    
    private func stopARMode() {
        // Hide camera preview
        showingCameraPreview = false
        isInitializingCamera = false
        cameraInitProgress = 0.0
        isARActive = false
        isProcessingAR = false
        isInContinuousMode = false
        arStatusMessage = "Point at furniture objects"
        arProcessingStateManager.reset()
        arObjectPlacementManager.clearAllObjects()
        arObjectPlacementManager.isReadyToPlace = false
        qrCodeDetectionService.reset()
        ar3DModelProcessor.reset()
        print("🛑 AR mode and camera preview stopped")
    }

    private func restartARScanning() {
        // Simply show the camera preview again
        showingCameraPreview = true
        isProcessingAR = false
        arStatusMessage = "Point at furniture objects"
        isARActive = true
        arObjectPlacementManager.isReadyToPlace = false
        print("🔄 SimpleCameraOverlay restarted for next capture")
    }
    
    // Process the segmented image from SimpleCameraOverlay
    private func performARProcessingWithSegmentedImage(_ segmentedImage: UIImage) async {
        isProcessingAR = true
        showingCameraPreview = false // Hide camera overlay during processing
        
        await MainActor.run {
            arStatusMessage = "Processing segmented furniture..."
            arProcessingStateManager.beginCapture()
        }
        
        // The image is already segmented by U2-Net in SimpleCameraOverlay
        print("📸 Received pre-segmented furniture image from SimpleCameraOverlay")
        
        // Check for QR codes first
        await MainActor.run {
            arStatusMessage = "Scanning for QR codes..."
            arProcessingStateManager.beginQRDetection()
        }

        let qrResult = await qrCodeDetectionService.detectQRCode(in: segmentedImage)

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

        // Backend processing with the segmented image
        await MainActor.run {
            arStatusMessage = "Uploading segmented furniture for 3D generation..."
            arProcessingStateManager.beginUpload()
        }

        // Process segmented image using backend API
        guard let generated3DModel = await ar3DModelProcessor.processImage(segmentedImage) else {
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
            print("✅ 3D model generation completed from segmented image")
        }
    }
    
    private func cleanupAR() {
        // Hide camera overlay and reset initialization
        showingCameraPreview = false
        isInitializingCamera = false
        cameraInitProgress = 0.0
        
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

