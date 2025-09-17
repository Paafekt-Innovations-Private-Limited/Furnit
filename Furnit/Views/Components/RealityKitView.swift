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
        // Use .nonAR mode for custom camera control that allows rotation without moving position
        let arView = ARView(frame: .zero, cameraMode: .nonAR, automaticallyConfigureSession: false)
        
        // Configure ARView for room viewing in non-AR mode
        arView.renderOptions.insert(.disablePersonOcclusion)
        arView.renderOptions.insert(.disableMotionBlur)
        
        // Apply quality settings
        let quality = appState.currentQuality
        configureRenderingQuality(arView: arView, quality: quality)
        
        print("🎨 Applying quality setting: \(quality.displayName)")
        
        // Set up coordinator and custom camera for non-AR mode
        context.coordinator.setupGestures(for: arView)
        context.coordinator.setupCustomCamera(for: arView)
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
        
        // Custom camera control for non-AR mode
        var cameraEntity: PerspectiveCamera?
        var cameraAnchor: AnchorEntity?
        
        func setupGestures(for arView: ARView) {
            gestureHandlers = RealityKitGestureHandlers(arView: arView)
            
            // Add tap gesture for AR object placement
            let tapGesture = UITapGestureRecognizer(target: self, action: #selector(handleTap(_:)))
            arView.addGestureRecognizer(tapGesture)
        }
        
        // Set up custom camera for non-AR mode with controllable rotation
        func setupCustomCamera(for arView: ARView) {
            // Create perspective camera entity with reasonable field of view
            cameraEntity = PerspectiveCamera()
            cameraEntity?.camera.fieldOfViewInDegrees = 75.0
            
            // Create camera anchor at a default position inside the room
            cameraAnchor = AnchorEntity(world: SIMD3<Float>(0, 1.5, 3))
            
            // Add camera entity to anchor
            if let camera = cameraEntity, let anchor = cameraAnchor {
                anchor.addChild(camera)
                arView.scene.addAnchor(anchor)
                
                // Pass camera references to gesture handlers for direct camera control
                gestureHandlers?.setCameraReferences(camera: camera, cameraAnchor: anchor)
                
                print("📷 Custom camera set up for non-AR mode at position: \(anchor.transform.translation)")
            }
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
                    
                    // In non-AR mode, simply add model to scene directly (no world anchor needed)
                    let modelAnchor = AnchorEntity(world: SIMD3<Float>(0, 0, 0))
                    modelAnchor.addChild(modelEntity)
                    arView.scene.addAnchor(modelAnchor)
                    coordinator.scene = arView.scene
                    
                    // Set up boundary manager for camera constraints
                    let boundaryManager = RealityKitBoundaryManager(arView: arView)
                    boundaryManager.calculateRoomBounds(from: modelEntity)
                    coordinator.gestureHandlers?.setBoundaryManager(boundaryManager)
                    
                    // Position custom camera inside the room bounds
                    if let cameraAnchor = coordinator.cameraAnchor {
                        let safeCameraPosition = boundaryManager.getSafeCameraPosition(near: SIMD3<Float>(0, 1.5, 0))
                        cameraAnchor.transform.translation = safeCameraPosition
                        
                        // Make camera look at the room center
                        let roomCenter = boundaryManager.getRoomCenter()
                        cameraAnchor.look(at: roomCenter, from: safeCameraPosition, relativeTo: nil)
                        
                        print("📷 Custom camera positioned at: \(safeCameraPosition)")
                        print("📷 Camera looking at room center: \(roomCenter)")
                    }
                    
                    // Set up lighting
                    setupLighting(for: arView)
                    
                    // Set up AR object placement manager with scene references
                    coordinator.arObjectPlacementManager = self.arObjectPlacementManager
                    self.arObjectPlacementManager.setSceneReferences(arView: arView, scene: arView.scene)
                    
                    // Set up camera movement manager with custom camera references
                    self.cameraMovementManager.setARView(arView)
                    
                    // Share camera references with camera movement manager for joystick control
                    if let cameraAnchor = coordinator.cameraAnchor {
                        self.cameraMovementManager.setCameraAnchor(cameraAnchor)
                    }
                    
                    // Set up camera movement callback
                    self.cameraMovementManager.onCameraMove = {
                        // Camera movement callback - ready for future enhancements
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