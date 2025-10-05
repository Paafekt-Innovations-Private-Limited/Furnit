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
        context.coordinator.setupGestures(for: arView, placementManager: arObjectPlacementManager)
        context.coordinator.setupCustomCamera(for: arView)
        loadModel(into: arView, coordinator: context.coordinator)
        
        // Set up camera movement manager with the ARView
        cameraMovementManager.setupARView(arView)
        
        return arView
    }
    
    func updateUIView(_ uiView: ARView, context: Context) {
        // Update rendering quality if settings changed
        let currentQuality = appState.currentQuality
        configureRenderingQuality(arView: uiView, quality: currentQuality)

        // Update movement speed if settings changed (only when actually different)
        let currentMovementSpeed = appState.currentMovementSpeed
        switch currentMovementSpeed {
        case .slow:
            cameraMovementManager.setSpeed(.slow)
        case .normal:
            cameraMovementManager.setSpeed(.normal)
        case .fast:
            cameraMovementManager.setSpeed(.fast)
        }
    }
    
    class Coordinator {
        var gestureHandlers: RealityKitGestureHandlers?
        var scene: RealityKit.Scene?
        weak var arObjectPlacementManager: ARObjectPlacementManager?

        // Custom camera control for non-AR mode
        var cameraEntity: PerspectiveCamera?
        var cameraAnchor: AnchorEntity?

        // World anchor for object placement (the model anchor)
        var worldAnchor: AnchorEntity?
        
        func setupGestures(for arView: ARView, placementManager: ARObjectPlacementManager) {
            gestureHandlers = RealityKitGestureHandlers(arView: arView)

            // Connect object placement manager to gesture handlers for manipulation support
            gestureHandlers?.setObjectPlacementManager(placementManager)

            // Add tap gesture for AR object placement
            let tapGesture = UITapGestureRecognizer(target: self, action: #selector(handleTap(_:)))
            arView.addGestureRecognizer(tapGesture)
        }
        
        // Set up custom camera for non-AR mode with controllable rotation
        func setupCustomCamera(for arView: ARView) {
            print("🎥 [RealityKitView.Coordinator] setupCustomCamera called")
            
            // Create perspective camera entity with reasonable field of view
            cameraEntity = PerspectiveCamera()
            cameraEntity?.camera.fieldOfViewInDegrees = 75.0
            
            // Create camera anchor at lower height inside the room
            cameraAnchor = AnchorEntity(world: SIMD3<Float>(0, 1.2, 3))
            print("🎥 [RealityKitView.Coordinator] Created camera anchor at: \(cameraAnchor?.transform.translation ?? SIMD3<Float>(0,0,0))")
            
            // Add camera entity to anchor with no offset (prevents orbital rotation)
            if let camera = cameraEntity, let anchor = cameraAnchor {
                camera.position = SIMD3<Float>(0, 0, 0) // No offset from anchor center
                anchor.addChild(camera)
                arView.scene.addAnchor(anchor)
                print("🎥 [RealityKitView.Coordinator] Camera entity added to anchor and anchor added to scene")
                
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
    
    // In RealityKitView.swift, replace your existing loadModel function with this:

    private func loadModel(into arView: ARView, coordinator: Coordinator) {
        // Check if this is a dollhouse model (stored in Documents)
        if model.fileName.contains("dollhouse_") {
            print("🏠 Loading dollhouse model: \(model.fileName)")
            
            // Get Documents directory
            let documentsURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
            let fileURL = documentsURL.appendingPathComponent(model.fileName)
            
            print("📁 Dollhouse path: \(fileURL.path)")
            print("📁 File exists: \(FileManager.default.fileExists(atPath: fileURL.path))")
            
            if FileManager.default.fileExists(atPath: fileURL.path) {
                Task { @MainActor in
                    do {
                        let modelEntity = try await Entity.load(contentsOf: fileURL)
                        
//                            Kishore
//                            // Fix rotation ONLY for dollhouse models
//                            modelEntity.orientation = simd_quatf(angle: -.pi/2, axis: [1, 0, 0])
//                            print("🔄 Applied dollhouse rotation fix")
                        // Ensure model has proper materials for visibility
                        ensureModelHasMaterials(modelEntity)
                        
                        // Scale up dollhouse models (they're small from SceneKit export)
                        modelEntity.scale = SIMD3<Float>(repeating: 2.0)
                        
                        // Calculate model bounds for camera positioning
                        let bounds = modelEntity.components[ModelComponent.self]?.mesh.bounds
                        if let bounds = bounds {
                            print("📦 Dollhouse bounds: min(\(bounds.min)), max(\(bounds.max))")
                        }
                        
                        // Create anchor and add model
                        let modelAnchor = AnchorEntity(world: SIMD3<Float>(0, 0, 0))
                        modelAnchor.addChild(modelEntity)
                        arView.scene.addAnchor(modelAnchor)
                        coordinator.scene = arView.scene
                        coordinator.worldAnchor = modelAnchor
                        
                        // Set up boundary manager for camera constraints
                        let boundaryManager = RealityKitBoundaryManager(arView: arView)
                        boundaryManager.calculateRoomBounds(from: modelEntity)
                        coordinator.gestureHandlers?.setBoundaryManager(boundaryManager)
                        
                        // Share boundary manager with camera movement manager
                        self.cameraMovementManager.setupARView(arView)
                        self.cameraMovementManager.setBoundaryManager(boundaryManager)
                        
                        // Set initial movement speed from settings - use faster speed for dollhouse
                        switch appState.currentMovementSpeed {
                        case .slow:
                            self.cameraMovementManager.setSpeed(.normal)  // Bump up slow to normal for dollhouse
                            print("🏠 Dollhouse speed: Using normal speed instead of slow")
                        case .normal:
                            self.cameraMovementManager.setSpeed(.fast)   // Bump up normal to fast for dollhouse
                            print("🏠 Dollhouse speed: Using fast speed instead of normal")
                        case .fast:
                            self.cameraMovementManager.setSpeed(.fast)   // Keep fast as fast
                            print("🏠 Dollhouse speed: Using fast speed")
                        }
                        
                        // Position camera inside the dollhouse room
                        if let cameraAnchor = coordinator.cameraAnchor {
                            // Dollhouse rooms need camera positioned inside, at eye level
                            let safeCameraPosition = SIMD3<Float>(0, 1.5, 0) // Start at center, eye level
                            cameraAnchor.transform.translation = safeCameraPosition
                            cameraAnchor.transform.rotation = simd_quatf(angle: 0, axis: SIMD3<Float>(0, 1, 0))
                            
                            print("📷 Camera positioned inside dollhouse at: \(safeCameraPosition)")
                        }
                        
                        // Set up lighting
                        setupLighting(for: arView)
                        
                        // Set up AR object placement manager (even though not used for dollhouses)
                        coordinator.arObjectPlacementManager = self.arObjectPlacementManager
                        self.arObjectPlacementManager.setSceneReferences(arView: arView, scene: arView.scene)
                        
                        if let worldAnchor = coordinator.worldAnchor {
                            self.arObjectPlacementManager.setWorldAnchor(worldAnchor)
                        }
                        
                        // Connect camera references for joystick control - AFTER all setup is complete
                        if let cameraAnchor = coordinator.cameraAnchor {
                            print("🎮 [RealityKitView] Setting camera anchor for DOLLHOUSE model")
                            self.cameraMovementManager.setCameraAnchor(cameraAnchor)
                            
                            // DON'T override the dollhouse speed - it was already set correctly above
                            print("🔧 [DOLLHOUSE] Keeping dollhouse speed as set above (no override)")
                            
                        } else {
                            print("❌ [RealityKitView] No camera anchor found for DOLLHOUSE model!")
                        }
                        
                        // Set up camera movement callback
                        self.cameraMovementManager.onCameraMove = {
                            // Camera movement callback - ready for future enhancements
                        }
                        
                        print("✅ Dollhouse model loaded successfully")
                        
                    } catch {
                        print("❌ Error loading dollhouse model: \(error)")
                    }
                }
            } else {
                print("❌ Dollhouse file not found at: \(fileURL.path)")
            }
            
        } else {
            // Regular model loading from bundle
            guard let dataAsset = model.dataAsset else {
                print("Failed to load data asset for model: \(model.name)")
                return
            }
            
            do {
                let tempURL = createTemporaryFile(from: dataAsset.data, fileName: "\(model.fileName).usdz")
                
                Task { @MainActor in
                    do {
                        let modelEntity = try await Entity.load(contentsOf: tempURL)
                        
                        // Rest of your existing regular model loading code...
                        ensureModelHasMaterials(modelEntity)
                        
                        // Calculate model bounds for camera positioning
                        let bounds = modelEntity.components[ModelComponent.self]?.mesh.bounds
                        if let bounds = bounds {
                            print("📦 Model bounds after loading: min(\(bounds.min)), max(\(bounds.max))")
                        } else {
                            print("📦 Model bounds after loading: no bounds")
                        }
                        
                        let modelAnchor = AnchorEntity(world: SIMD3<Float>(0, 0, 0))
                        modelAnchor.addChild(modelEntity)
                        arView.scene.addAnchor(modelAnchor)
                        coordinator.scene = arView.scene
                        coordinator.worldAnchor = modelAnchor
                        
                        // Set up boundary manager for camera constraints
                        let boundaryManager = RealityKitBoundaryManager(arView: arView)
                        boundaryManager.calculateRoomBounds(from: modelEntity)
                        coordinator.gestureHandlers?.setBoundaryManager(boundaryManager)
                        
                        self.cameraMovementManager.setupARView(arView)
                        self.cameraMovementManager.setBoundaryManager(boundaryManager)
                        
                        // Set initial movement speed from settings
                        switch appState.currentMovementSpeed {
                        case .slow:
                            self.cameraMovementManager.setSpeed(.slow)
                        case .normal:
                            self.cameraMovementManager.setSpeed(.normal)
                        case .fast:
                            self.cameraMovementManager.setSpeed(.fast)
                        }
                        
                        // Position custom camera inside the room bounds
                        if let cameraAnchor = coordinator.cameraAnchor {
                            let safeCameraPosition = boundaryManager.getSafeCameraPosition(near: SIMD3<Float>(0, 1.2, 0))
                            cameraAnchor.transform.translation = safeCameraPosition
                            cameraAnchor.transform.rotation = simd_quatf(angle: 0, axis: SIMD3<Float>(0, 1, 0))
                            
                            print("📷 Custom camera positioned at: \(safeCameraPosition)")
                        }
                        
                        // Set up lighting
                        setupLighting(for: arView)
                        
                        // Set up AR object placement manager
                        coordinator.arObjectPlacementManager = self.arObjectPlacementManager
                        self.arObjectPlacementManager.setSceneReferences(arView: arView, scene: arView.scene)
                        
                        if let worldAnchor = coordinator.worldAnchor {
                            self.arObjectPlacementManager.setWorldAnchor(worldAnchor)
                            print("🌍 Connected world anchor to object placement manager")
                        }
                        
                        // Connect camera references for joystick control
                        if let cameraAnchor = coordinator.cameraAnchor {
                            print("🎮 [RealityKitView] Setting camera anchor for REGULAR model")
                            self.cameraMovementManager.setCameraAnchor(cameraAnchor)
                        } else {
                            print("❌ [RealityKitView] No camera anchor found for REGULAR model!")
                        }
                        
                        self.cameraMovementManager.onCameraMove = {
                            // Camera movement callback
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
            intensity: Float(300 * 3.0),
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
        
        // Create additional light specifically for placed 3D objects
        let objectLightComponent = DirectionalLightComponent(
            color: .white,
            intensity: Float(1500 * lightingMultiplier), // Brighter for 3D objects
            isRealWorldProxy: false
        )

        let objectLightEntity = Entity()
        objectLightEntity.components.set(objectLightComponent)
        objectLightEntity.orientation = simd_quatf(angle: .pi / 6, axis: SIMD3<Float>(1, 0, 0)) // 30 degrees down
        objectLightEntity.position = SIMD3<Float>(0, 8, 2) // Above and slightly forward

        // Add lights to scene
        let lightingAnchor = AnchorEntity(.world(transform: matrix_identity_float4x4))
        lightingAnchor.addChild(ambientLightEntity)
        lightingAnchor.addChild(keyLightEntity)
        lightingAnchor.addChild(objectLightEntity)
        arView.scene.addAnchor(lightingAnchor)

        print("💡 Applied lighting intensity: \(lightingMultiplier)x for \(quality.displayName) quality")
        print("💡 Added dedicated lighting for placed 3D objects")
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

    // Ensure loaded model has proper materials for visibility
    private func ensureModelHasMaterials(_ entity: Entity) {
        // Check if entity itself has a model component and materials
        if var modelComponent = entity.components[ModelComponent.self] {
            if modelComponent.materials.isEmpty {
                // Add default material if none exists
                let defaultMaterial = SimpleMaterial(color: .white, roughness: 0.5, isMetallic: false)
                modelComponent.materials = [defaultMaterial]
                entity.components.set(modelComponent)
                print("🎨 Added default white material to model entity")
            } else {
                print("🎨 Model has \(modelComponent.materials.count) existing materials")
                // Ensure materials are not transparent
                for (index, material) in modelComponent.materials.enumerated() {
                    if let simpleMaterial = material as? SimpleMaterial {
                        print("🎨 Material \(index): color=\(simpleMaterial.color), roughness=\(simpleMaterial.roughness)")
                    }
                }
            }
        }

        // Recursively check child entities
        for child in entity.children {
            ensureModelHasMaterials(child)
        }
    }
}

// MARK: - Extensions for SIMD math operations are defined in RealityKitObjectPlacementManager.swift
