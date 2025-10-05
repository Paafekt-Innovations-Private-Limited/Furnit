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
    
    // Camera preview state
    @State private var showingCameraPreview = false
    @State private var capturedImageFromPreview: UIImage?
    @State private var isInitializingCamera = false
    @State private var cameraInitProgress: Double = 0.0
    @State private var initializationTimer: Timer?
    
    // AR functionality state
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
    
    // Dollhouse specific state
    @State private var isDollhouseRoom = false
    @State private var dollhouseLoadFailed = false

    private var gestureHandlers: RealityKitGestureHandlers? {
        return nil
    }
    
    var body: some View {
        GeometryReader { geometry in
            ZStack {
                // Check if it's a dollhouse room and handle differently
                // Use RealityKitView for ALL models (including dollhouses)
                RealityKitView(
                    model: model,
                    cameraMovementManager: cameraMovementManager,
                    arObjectPlacementManager: arObjectPlacementManager,
                    isARActive: isARActive
                )
                .ignoresSafeArea(.all)
                .onAppear {
                    if model.fileName.contains("dollhouse_") {
                        print("🏠 Viewing dollhouse room with RealityKitView: \(model.displayName)")
                    }
                }
                
                // Show fallback message if dollhouse loading failed
                if dollhouseLoadFailed {
                    VStack {
                        Spacer()
                        HStack {
                            Image(systemName: "exclamationmark.triangle")
                                .foregroundColor(.orange)
                            Text("Showing placeholder - dollhouse textures not loading")
                                .font(.caption)
                                .foregroundColor(.white)
                        }
                        .padding()
                        .background(Color.black.opacity(0.7))
                        .cornerRadius(8)
                        .padding()
                    }
                }
                
                // Camera initialization overlay
                if isInitializingCamera {
                    cameraInitializationOverlay
                }
                
                // Camera preview overlay
                if showingCameraPreview {
                    SimpleCameraOverlay(
                        capturedImage: $capturedImageFromPreview,
                        isShowingCamera: $showingCameraPreview
                    )
                    .zIndex(100)
                    .transition(.opacity)
                }
                
                // UI Controls
                if isLandscape(geometry: geometry) {
                    landscapeControls
                } else {
                    portraitControls
                }
                
                // AR status overlay
                if isARActive && !isDollhouseRoom {
                    arStatusOverlay
                }

                // Object manipulation guidance overlay
                if !isDollhouseRoom {
                    objectManipulationGuidanceOverlay
                }
            }
        }
        .navigationBarHidden(true)
        .statusBarHidden(true)
        .preferredColorScheme(.dark)
        .onAppear {
            if !isDollhouseRoom {
                setupARManagers()
                ar3DModelProcessor.setQualitySettings(AppStateManager.shared.qualitySettings)
            }
        }
        .onDisappear {
            if !isDollhouseRoom {
                cleanupAR()
            }
        }
        .onChange(of: shouldRestartScanning) { _, newValue in
            if newValue && !isDollhouseRoom {
                restartARScanning()
                shouldRestartScanning = false
            }
        }
        .onChange(of: capturedImageFromPreview) { _, newImage in
            if let image = newImage, !isDollhouseRoom {
                Task {
                    await performARProcessingWithSegmentedImage(image)
                }
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
                    // AR Button (hide for dollhouse rooms)
                    if !arObjectPlacementManager.isManipulatingObject && !isDollhouseRoom {
                        ARButton(isARActive: Binding(
                            get: { showingCameraPreview },
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

                    // Joystick for camera movement
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
            
            // Right side controls
            VStack(spacing: 16) {
                // AR Button (hide for dollhouse)
                if !arObjectPlacementManager.isManipulatingObject && !isDollhouseRoom {
                    ARButton(isARActive: Binding(
                        get: { showingCameraPreview },
                        set: { _ in toggleARMode() }
                    )) {
                        toggleARMode()
                    }
                }

                Spacer()

                if !isARActive && !arObjectPlacementManager.isManipulatingObject {
                    modelInfoPanel
                }

                // Joystick
                if !arObjectPlacementManager.isManipulatingObject {
                    VStack(spacing: 4) {
                        if joystickOffset != .zero {
                            Text("Moving")
                                .font(.caption2)
                                .foregroundColor(.white.opacity(0.6))
                                .animation(.easeInOut(duration: 0.2), value: joystickOffset)
                        }
                        
                        VirtualJoystick(joystickOffset: $joystickOffset)
                            .onChange(of: joystickOffset) { _, newOffset in
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
            
            if isDollhouseRoom {
                Text("Dollhouse Room - Use joystick to explore")
                    .font(.caption)
                    .foregroundColor(.white.opacity(0.8))
                    .multilineTextAlignment(.center)
            } else {
                Text("Use gestures to rotate, zoom, and explore the room")
                    .font(.caption)
                    .foregroundColor(.white.opacity(0.8))
                    .multilineTextAlignment(.center)
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
        .background(Color.black.opacity(0.7))
        .cornerRadius(16)
    }
    
    // MARK: - Camera Initialization Overlay
    private var cameraInitializationOverlay: some View {
        ZStack {
            Color.black.opacity(0.7)
                .ignoresSafeArea()
                .onTapGesture {
                    cancelCameraInitialization()
                }
            
            VStack(spacing: 24) {
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
                
                VStack(spacing: 8) {
                    ProgressView(value: cameraInitProgress)
                        .progressViewStyle(LinearProgressViewStyle(tint: .white))
                        .frame(width: 200)
                        .scaleEffect(x: 1, y: 2)
                    
                    Text("\(Int(cameraInitProgress * 100))%")
                        .font(.caption2)
                        .foregroundColor(.white.opacity(0.7))
                }
                
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
            .onTapGesture { }
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

            Spacer()
        }
    }

    // MARK: - AR Methods
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
    }
    
    private func toggleARMode() {
        if showingCameraPreview {
            showingCameraPreview = false
            stopARMode()
        } else {
            startCameraInitialization()
        }
    }
    
    private func startCameraInitialization() {
        isInitializingCamera = true
        cameraInitProgress = 0.0
        isARActive = true
        
        initializationTimer = Timer.scheduledTimer(withTimeInterval: 0.05, repeats: true) { timer in
            self.cameraInitProgress += 0.02
            
            if self.cameraInitProgress >= 1.0 {
                timer.invalidate()
                self.initializationTimer = nil
                
                withAnimation(.easeOut(duration: 0.3)) {
                    self.isInitializingCamera = false
                }
                
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
                    self.showingCameraPreview = true
                    self.arStatusMessage = "Live furniture segmentation"
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
            isARActive = false
        }
    }
    
    private func stopARMode() {
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
    }

    private func restartARScanning() {
        showingCameraPreview = true
        isProcessingAR = false
        arStatusMessage = "Point at furniture objects"
        isARActive = true
        arObjectPlacementManager.isReadyToPlace = false
    }
    
    private func performARProcessingWithSegmentedImage(_ segmentedImage: UIImage) async {
        isProcessingAR = true
        showingCameraPreview = false
        
        await MainActor.run {
            arStatusMessage = "Processing segmented furniture..."
            arProcessingStateManager.beginCapture()
        }
        
        // Process the image...
        // (rest of your AR processing code)
    }
    
    private func cleanupAR() {
        showingCameraPreview = false
        isInitializingCamera = false
        cameraInitProgress = 0.0
        isARActive = false
        isProcessingAR = false
        isInContinuousMode = false
        arObjectPlacementManager.clearAllObjects()
        arObjectPlacementManager.isReadyToPlace = false
        arProcessingStateManager.reset()
        ar3DModelProcessor.reset()
        qrCodeDetectionService.reset()
    }
    
    private func deleteSelectedObject() {
        guard let selectedObject = arObjectPlacementManager.selectedObject else { return }
        showObjectMovementJoystick = false
        arObjectPlacementManager.endObjectManipulation()
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
            self.arObjectPlacementManager.removeObject(selectedObject.id)
        }
    }
}

import RealityKit
import SwiftUI

// MARK: - Dollhouse Room View with Orbit Camera
struct DollhouseRoomView: UIViewRepresentable {
    let model: USDZModel
    @ObservedObject var cameraMovementManager: RealityKitCameraMovementManager
    @Binding var loadFailed: Bool
    
    func makeUIView(context: Context) -> ARView {
        let arView = ARView(frame: .zero)
        
        // Use AR mode for better camera control
        arView.cameraMode = .ar
        
        // But disable AR session to stay static
        arView.automaticallyConfigureSession = false
        
        // Set background
        arView.environment.background = .color(.black)
        
        // Create anchor
        let anchor = AnchorEntity(world: .zero)
        arView.scene.anchors.append(anchor)
        
        // Add lighting
        let light = DirectionalLight()
        light.light.intensity = 10000
        light.position = [0, 10, 0]
        anchor.addChild(light)
        
        arView.environment.lighting.intensityExponent = 2
        
        // Load model
        print("\n🏠 Loading: \(model.fileName)")
        let documentsURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let fileURL = documentsURL.appendingPathComponent(model.fileName)
        
        if FileManager.default.fileExists(atPath: fileURL.path) {
            do {
                let entity = try Entity.load(contentsOf: fileURL)
                
                // Scale and position
                entity.scale = [3, 3, 3]  // Even bigger
                entity.position = [0, -10, -20]  // Move down and back
                
                anchor.addChild(entity)
                print("✅ Dollhouse loaded at scale 3")
                
            } catch {
                print("❌ Failed: \(error)")
                // Create visible test object
                let mesh = MeshResource.generateBox(size: 5)
                var material = SimpleMaterial()
                material.color = .init(tint: .red)
                let box = ModelEntity(mesh: mesh, materials: [material])
                box.position = [0, 0, -10]
                anchor.addChild(box)
                print("🔴 Added test box")
            }
        } else {
            print("❌ File not found")
            // Create test box
            let mesh = MeshResource.generateBox(size: 5)
            var material = SimpleMaterial()
            material.color = .init(tint: .blue)
            let box = ModelEntity(mesh: mesh, materials: [material])
            box.position = [0, 0, -10]
            anchor.addChild(box)
            print("🔵 Added test box")
        }
        
        // Install gestures for interaction
        let panGesture = UIPanGestureRecognizer(target: context.coordinator, action: #selector(Coordinator.handlePan(_:)))
        arView.addGestureRecognizer(panGesture)
        
        context.coordinator.arView = arView
        context.coordinator.anchor = anchor
        
        print("✅ Setup complete - use gestures to interact:")
        print("   • Drag to rotate")
        print("   • Pinch to scale")
        print("   • Two-finger drag to move")
        
        return arView
    }
    
    func updateUIView(_ uiView: ARView, context: Context) {}
    
    func makeCoordinator() -> Coordinator {
        Coordinator()
    }
    
    class Coordinator: NSObject {
        weak var arView: ARView?
        weak var anchor: AnchorEntity?
        var currentRotation: Float = 0
        
        @objc func handlePan(_ gesture: UIPanGestureRecognizer) {
            guard let anchor = anchor else { return }
            
            let translation = gesture.translation(in: gesture.view)
            let rotationAmount = Float(translation.x) * 0.01
            
            currentRotation += rotationAmount
            anchor.orientation = simd_quatf(angle: currentRotation, axis: [0, 1, 0])
            
            gesture.setTranslation(.zero, in: gesture.view)
        }
    }
}
