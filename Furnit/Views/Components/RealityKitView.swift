import SwiftUI
import RealityKit
import ARKit

struct RealityKitView: UIViewRepresentable {
    let model: USDZModel
    let cameraMovementManager: CameraMovementManager
    let arObjectPlacementManager: ARObjectPlacementManager
    let isARActive: Bool
    
    // Access quality settings from environment
    @Environment(\.appState) private var appState
    
    func makeCoordinator() -> Coordinator {
        Coordinator()
    }
    
    func makeUIView(context: Context) -> ARView {
        let arView = ARView(frame: .zero)
        
        // Configure ARView for room viewing
        arView.automaticallyConfigureSession = false
        arView.renderOptions.insert(.disablePersonOcclusion)
        arView.renderOptions.insert(.disableMotionBlur)
        
        // Apply quality settings
        let quality = appState.currentQuality
        configureRenderingQuality(arView: arView, quality: quality)
        
        print("🎨 Applying quality setting: \(quality.displayName)")
        
        // Set up coordinator
        context.coordinator.setupGestures(for: arView)
        loadModel(into: arView, coordinator: context.coordinator)
        
        // Set up camera movement manager with the ARView
        cameraMovementManager.setARView(arView)
        
        return arView
    }
    
    func updateUIView(_ uiView: ARView, context: Context) {
        // Update rendering quality if settings changed
        let currentQuality = appState.currentQuality
        configureRenderingQuality(arView: uiView, quality: currentQuality)
    }
    
    class Coordinator {
        var gestureHandlers: RealityKitGestureHandlers?
        var scene: RealityKit.Scene?
        weak var arObjectPlacementManager: ARObjectPlacementManager?
        
        func setupGestures(for arView: ARView) {
            gestureHandlers = RealityKitGestureHandlers(arView: arView)
            
            // Add tap gesture for AR object placement
            let tapGesture = UITapGestureRecognizer(target: self, action: #selector(handleTap(_:)))
            arView.addGestureRecognizer(tapGesture)
        }
        
        @objc func handleTap(_ gesture: UITapGestureRecognizer) {
            // Handle AR object placement if AR is active
            Task { @MainActor in
                if let arManager = arObjectPlacementManager,
                   arManager.isReadyToPlace {
                    let location = gesture.location(in: gesture.view)
                    let _ = arManager.handleTapToPlace(at: location)
                }
            }
        }
    }
    
    private func loadModel(into arView: ARView, coordinator: Coordinator) {
        guard let dataAsset = model.dataAsset else {
            print("Failed to load data asset for model: \(model.name)")
            return
        }
        
        do {
            let tempURL = createTemporaryFile(from: dataAsset.data, fileName: "\(model.fileName).usdz")
            
            // Load USDZ model using RealityKit's Entity loading
            Task { @MainActor in
                do {
                    let modelEntity = try await Entity.load(contentsOf: tempURL)
                    
                    // Calculate model bounds for camera positioning
                    let bounds = modelEntity.components[ModelComponent.self]?.mesh.bounds
                    setupCamera(for: arView, with: bounds)
                    setupLighting(for: arView)
                    
                    // Set up gesture handlers and world anchor first
                    coordinator.gestureHandlers?.setupWorldAnchor()
                    
                    // Add model to world anchor for gesture control
                    coordinator.gestureHandlers?.addToWorld(modelEntity)
                    coordinator.scene = arView.scene
                    
                    // Set up boundary manager first
                    let boundaryManager = RealityKitBoundaryManager(arView: arView)
                    boundaryManager.calculateRoomBounds(from: modelEntity)
                    coordinator.gestureHandlers?.setBoundaryManager(boundaryManager)
                    
                    // Calculate initial safe camera position using boundary manager
                    let targetPosition = SIMD3<Float>(0, 1.5, 0) // Default target
                    let safeCameraPosition = boundaryManager.getSafeCameraPosition(near: targetPosition)
                    
                    print("📷 Safe camera position calculated: \(safeCameraPosition)")
                    print("   Room center: \(boundaryManager.getRoomCenter())")
                    print("   Room dimensions: \(boundaryManager.getRoomDimensions())")
                    
                    // Position world anchor to place camera at safe position
                    // Since camera is at origin, we move world content to opposite position
                    if let gestureHandlers = coordinator.gestureHandlers,
                       let worldAnchor = gestureHandlers.worldAnchor {
                        var worldTransform = worldAnchor.transform
                        worldTransform.translation = -safeCameraPosition
                        worldAnchor.transform = worldTransform
                        print("🌍 World anchor positioned for safe camera placement")
                        
                        // Validate camera is now within bounds by calculating effective position
                        // Camera is at origin, but we need to check relative to the moved world content
                        let cameraPositionInWorldSpace = arView.cameraTransform.translation
                        let worldAnchorPosition = worldAnchor.transform.translation
                        let effectiveCameraPosition = cameraPositionInWorldSpace - worldAnchorPosition
                        
                        let isWithinBounds = boundaryManager.isPositionWithinBounds(effectiveCameraPosition)
                        print("✅ Camera within bounds check: \(isWithinBounds)")
                        print("   Camera in world space: \(cameraPositionInWorldSpace)")
                        print("   World anchor position: \(worldAnchorPosition)")
                        print("   Effective camera position: \(effectiveCameraPosition)")
                        
                        if !isWithinBounds {
                            print("⚠️ Camera still outside bounds after positioning")
                            if let bounds = boundaryManager.getCurrentBounds() {
                                print("   Bounds: min(\(bounds.min)), max(\(bounds.max))")
                            }
                        }
                    }
                    
                    // Set up AR manager with world anchor reference
                    coordinator.arObjectPlacementManager = self.arObjectPlacementManager
                    self.arObjectPlacementManager.setSceneReferences(arView: arView, scene: arView.scene)
                    
                    // Pass world anchor reference to object placement manager
                    if let gestureHandlers = coordinator.gestureHandlers,
                       let worldAnchor = gestureHandlers.worldAnchor {
                        self.arObjectPlacementManager.setWorldAnchor(worldAnchor)
                    }
                    
                    // Update camera movement manager to use same world anchor as gesture handlers
                    self.cameraMovementManager.setARView(arView)
                    
                    // Share the world anchor between gesture handlers and camera movement manager
                    if let gestureHandlers = coordinator.gestureHandlers,
                       let worldAnchor = gestureHandlers.worldAnchor {
                        self.cameraMovementManager.setWorldAnchor(worldAnchor)
                    }
                    
                    // Set up camera movement callback
                    self.cameraMovementManager.onCameraMove = {
                        // Camera movement callback - ready for future enhancements
                    }
                    
                    // Calculate boundaries after scene is set
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
                        self.cameraMovementManager.updateBoundaries()
                    }
                    
                    print("✅ RealityKit model loaded successfully")
                    
                } catch {
                    print("Error loading USDZ model with RealityKit: \(error.localizedDescription)")
                }
                
                // Clean up temporary file
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
                    try? FileManager.default.removeItem(at: tempURL)
                }
            }
            
        } catch {
            print("Error creating temporary file: \(error.localizedDescription)")
        }
    }
    
    private func createTemporaryFile(from data: Data, fileName: String) -> URL {
        let tempDirectory = FileManager.default.temporaryDirectory
        let tempURL = tempDirectory.appendingPathComponent(fileName)
        
        try? data.write(to: tempURL)
        return tempURL
    }
    
    private func setupCamera(for arView: ARView, with bounds: BoundingBox?) {
        // Calculate room dimensions from bounds
        let roomSize: SIMD3<Float>
        let roomCenter: SIMD3<Float>
        
        if let bounds = bounds {
            roomSize = bounds.max - bounds.min
            roomCenter = (bounds.min + bounds.max) / 2
        } else {
            // Default room size
            roomSize = SIMD3<Float>(5, 3, 5)
            roomCenter = SIMD3<Float>(0, 1.5, 0)
        }
        
        // Position camera INSIDE the room, slightly above floor level
        let cameraHeight = bounds?.min.y ?? 0.0 + (roomSize.y * 0.4) // 40% up from floor
        let viewingDistance = min(roomSize.x, roomSize.z) * 0.3 // 30% of smaller horizontal dimension
        
        // Position camera inside room, looking toward the center
        let cameraPosition = SIMD3<Float>(
            roomCenter.x - viewingDistance,
            cameraHeight,
            roomCenter.z + viewingDistance * 0.5
        )
        
        // Create camera transform
        var cameraTransform = Transform.identity
        cameraTransform.translation = cameraPosition
        
        // Look at room center
        let lookDirection = normalize(roomCenter - cameraPosition)
        cameraTransform.rotation = simd_quatf(from: SIMD3<Float>(0, 0, -1), to: lookDirection)
        
        // Note: Camera positioning is now handled by world anchor transforms
        // ARView.cameraTransform is read-only in RealityKit
        
        print("📷 Camera configured at position: \(cameraPosition)")
    }
    
    private func setupLighting(for arView: ARView) {
        let quality = appState.currentQuality
        let lightingMultiplier = quality.lightingIntensity
        
        // Create ambient light
        let ambientLightComponent = DirectionalLightComponent(
            color: .white,
            intensity: Float(300 * lightingMultiplier),
            isRealWorldProxy: false
        )
        
        let ambientLightEntity = Entity()
        ambientLightEntity.components.set(ambientLightComponent)
        
        // Create key light
        let keyLightComponent = DirectionalLightComponent(
            color: .white,
            intensity: Float(800 * lightingMultiplier),
            isRealWorldProxy: false
        )
        
        let keyLightEntity = Entity()
        keyLightEntity.components.set(keyLightComponent)
        keyLightEntity.orientation = simd_quatf(angle: .pi / 4, axis: SIMD3<Float>(1, 0, 0))
        keyLightEntity.position = SIMD3<Float>(5, 10, 5)
        
        // Add lights to scene
        let lightingAnchor = AnchorEntity(.world(transform: matrix_identity_float4x4))
        lightingAnchor.addChild(ambientLightEntity)
        lightingAnchor.addChild(keyLightEntity)
        arView.scene.addAnchor(lightingAnchor)
        
        print("💡 Applied lighting intensity: \(lightingMultiplier)x for \(quality.displayName) quality")
    }
    
    // Configure rendering quality based on user settings
    private func configureRenderingQuality(arView: ARView, quality: AssetQuality) {
        switch quality {
        case .standard:
            arView.renderOptions.remove(.disableAREnvironmentLighting)
            if #available(iOS 15.0, *) {
                arView.environment.sceneUnderstanding.options = []
            }
        case .high:
            arView.renderOptions.insert(.disableAREnvironmentLighting)
            if #available(iOS 15.0, *) {
                arView.environment.sceneUnderstanding.options = .collision
            }
        case .best:
            arView.renderOptions.insert(.disableAREnvironmentLighting)
            if #available(iOS 15.0, *) {
                arView.environment.sceneUnderstanding.options = [.collision, .physics]
            }
        }
        
        print("🔄 Updated rendering quality to: \(quality.displayName)")
    }
}

// MARK: - Extensions for SIMD math operations are defined in RealityKitObjectPlacementManager.swift