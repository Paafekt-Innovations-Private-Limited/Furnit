import SwiftUI
import RealityKit
import ARKit

struct RealityKitView: UIViewRepresentable {
    let model: USDZModel
    let cameraMovementManager: CameraMovementManager
    let arObjectPlacementManager: ARObjectPlacementManager
    let isARActive: Bool

    // ✅ NEW: Snapshot capability - for capturing clean 3D room
    @Binding var shouldCaptureSnapshot: Bool
    @Binding var capturedSnapshot: UIImage?

    // ✅ NEW: Camera reset trigger - resets camera to optimal position
    @Binding var shouldResetCamera: Bool
    
    // Access quality settings from environment
    @Environment(\.appState) private var appState
    
    func makeCoordinator() -> Coordinator {
        Coordinator()
    }
    
    func makeUIView(context: Context) -> ARView {
        print("🎨 [RealityKitView.makeUIView] ========================================")
        print("🎨 [RealityKitView.makeUIView] Creating ARView for model: \(model.displayName)")
        print("   - File name: \(model.fileName)")
        print("   - Is saved room: \(model.isSavedRoom)")
        
        // Use .nonAR mode for custom camera control that allows rotation without moving position
        let arView = ARView(frame: .zero, cameraMode: .nonAR, automaticallyConfigureSession: false)
        
        // ✅ NEW: Store ARView reference in Coordinator for snapshot
        context.coordinator.arView = arView
        
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

        // Note: Camera will be registered with GlobalCameraController AFTER model loads
        // (see loadModel - camera is added to scene after model for proper precedence)

        loadModel(into: arView, coordinator: context.coordinator)

        // Set up camera movement manager with the ARView (for other features)
        cameraMovementManager.setupARView(arView)
        if let cameraAnchor = context.coordinator.cameraAnchor {
            cameraMovementManager.setCameraAnchor(cameraAnchor)
        }

        return arView
    }
    
    func updateUIView(_ uiView: ARView, context: Context) {
        // ✅ ALWAYS register camera with global controller (both anchor and camera entity)
        if let cameraAnchor = context.coordinator.cameraAnchor {
            GlobalCameraController.shared.registerRealityKitCamera(cameraAnchor, camera: context.coordinator.cameraEntity, arView: uiView)
            cameraMovementManager.setCameraAnchor(cameraAnchor)
        }

        // ✅ Check if model changed - reset camera position if so
        if context.coordinator.currentModelID != model.id {
            print("🔄 [RealityKitView.updateUIView] MODEL CHANGED! Resetting camera position...")
            print("   Old model: \(context.coordinator.currentModelID?.uuidString ?? "nil")")
            print("   New model: \(model.id.uuidString) (\(model.displayName))")

            // Reset camera to optimal position using stored boundary manager
            if let cameraAnchor = context.coordinator.cameraAnchor,
               let boundaryManager = context.coordinator.boundaryManager,
               boundaryManager.bounds != nil {
                let (cameraPosition, lookAtPosition) = boundaryManager.getOptimalCameraPosition()
                cameraAnchor.transform.translation = cameraPosition

                let lookDirection = normalize(lookAtPosition - cameraPosition)
                let lookRotation = simd_quatf(from: SIMD3<Float>(0, 0, -1), to: lookDirection)
                cameraAnchor.transform.rotation = lookRotation

                print("📷 [RealityKitView.updateUIView] Camera RESET to: \(cameraPosition)")
            }

            context.coordinator.currentModelID = model.id
        }

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
        
        // ✅ Handle camera reset requests (triggered on view appear)
        if shouldResetCamera {
            print("🔄 [RealityKitView.updateUIView] CAMERA RESET TRIGGERED")
            if let cameraAnchor = context.coordinator.cameraAnchor,
               let boundaryManager = context.coordinator.boundaryManager,
               boundaryManager.bounds != nil {
                let (cameraPosition, lookAtPosition) = boundaryManager.getOptimalCameraPosition()
                cameraAnchor.transform.translation = cameraPosition

                let lookDirection = normalize(lookAtPosition - cameraPosition)
                let lookRotation = simd_quatf(from: SIMD3<Float>(0, 0, -1), to: lookDirection)
                cameraAnchor.transform.rotation = lookRotation

                print("📷 [RealityKitView] Camera RESET to optimal position: \(cameraPosition)")
            } else {
                print("⚠️ [RealityKitView] Cannot reset camera - missing cameraAnchor or boundaryManager")
            }

            // Clear the flag
            DispatchQueue.main.async {
                self.shouldResetCamera = false
            }
        }

        // ✅ FIXED: Handle snapshot requests
        if shouldCaptureSnapshot {
            print("📸 [RealityKitView] Snapshot requested, capturing ARView...")
            
            // Capture the bindings to mutate them in the async closure
            let capturedSnapshotBinding = $capturedSnapshot
            let shouldCaptureSnapshotBinding = $shouldCaptureSnapshot
            
            // Use ARView's built-in snapshot method to capture ONLY 3D content
            uiView.snapshot(saveToHDR: false) { image in
                DispatchQueue.main.async {
                    capturedSnapshotBinding.wrappedValue = image
                    shouldCaptureSnapshotBinding.wrappedValue = false
                    
                    if let image = image {
                        print("✅ [RealityKitView] Snapshot captured: \(Int(image.size.width))x\(Int(image.size.height))")
                    } else {
                        print("❌ [RealityKitView] Snapshot failed - no image returned")
                    }
                }
            }
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

        // ✅ NEW: Store ARView reference for snapshot
        weak var arView: ARView?

        // ✅ Track current model to detect room changes
        var currentModelID: UUID?
        var boundaryManager: RealityKitBoundaryManager?
        
        func setupGestures(for arView: ARView, placementManager: ARObjectPlacementManager) {
            gestureHandlers = RealityKitGestureHandlers(arView: arView)
            self.arObjectPlacementManager = placementManager

            // Connect object placement manager to gesture handlers for manipulation support
            gestureHandlers?.setObjectPlacementManager(placementManager)

            // Add tap gesture for AR object placement
            let tapGesture = UITapGestureRecognizer(target: self, action: #selector(handleTap(_:)))
            arView.addGestureRecognizer(tapGesture)
        }
        
        // Set up custom camera for non-AR mode with controllable rotation
        // NOTE: This only CREATES the camera, does NOT add to scene yet (that happens after model loads)
        func setupCustomCamera(for arView: ARView) {
            // Create perspective camera entity with reasonable field of view
            cameraEntity = PerspectiveCamera()
            cameraEntity?.camera.fieldOfViewInDegrees = 75.0

            // Create camera anchor at lower height inside the room
            cameraAnchor = AnchorEntity(world: SIMD3<Float>(0, 1.2, 3))
            cameraAnchor?.name = "CustomCameraAnchor" // Give it a name for identification

            // Add camera entity to anchor with no offset (prevents orbital rotation)
            if let camera = cameraEntity, let anchor = cameraAnchor {
                camera.position = SIMD3<Float>(0, 0, 0) // No offset from anchor center
                camera.name = "CustomPerspectiveCamera" // Name the camera
                anchor.addChild(camera)
                // ❌ DON'T add to scene here - will add AFTER model loads
                // arView.scene.addAnchor(anchor)

                // Pass camera references to gesture handlers for direct camera control
                gestureHandlers?.setCameraReferences(camera: camera, cameraAnchor: anchor)

                print("📷 Custom camera CREATED (will add to scene after model loads)")
            }
        }

        // Add camera to scene - called AFTER model loads to ensure camera takes precedence
        func addCameraToScene(arView: ARView) {
            guard let cameraAnchor = cameraAnchor else {
                print("❌ [addCameraToScene] No camera anchor to add!")
                return
            }

            // First, find and log any existing cameras in the scene
            var existingCameraCount = 0
            for anchor in arView.scene.anchors {
                func findCameras(in entity: Entity) {
                    if entity is PerspectiveCamera {
                        existingCameraCount += 1
                        print("⚠️ Found existing PerspectiveCamera: \(entity.name.isEmpty ? "unnamed" : entity.name)")
                    }
                    for child in entity.children {
                        findCameras(in: child)
                    }
                }
                findCameras(in: anchor)
            }
            print("📷 [addCameraToScene] Found \(existingCameraCount) existing cameras in scene")

            // Add our camera anchor to the scene LAST
            arView.scene.addAnchor(cameraAnchor)
            print("📷 [addCameraToScene] Camera anchor added to scene as LAST anchor")
            print("   Total anchors in scene: \(arView.scene.anchors.count)")
            
            // ✅ Try a more aggressive approach - remove ALL cameras then add ours
            if let cameraEntity = cameraEntity {
                print("🧹 Clearing all existing cameras from scene before adding ours")
                
                // Collect all existing camera entities
                var existingCameras: [PerspectiveCamera] = []
                for anchor in arView.scene.anchors {
                    func collectCameras(in entity: Entity) {
                        if let perspectiveCamera = entity as? PerspectiveCamera,
                           perspectiveCamera !== cameraEntity {
                            existingCameras.append(perspectiveCamera)
                        }
                        for child in entity.children {
                            collectCameras(in: child)
                        }
                    }
                    collectCameras(in: anchor)
                }
                
                // Remove all existing cameras
                for camera in existingCameras {
                    camera.parent?.removeChild(camera)
                    print("🗑️ Removed existing camera: \(camera.name)")
                }
                
                print("✅ [addCameraToScene] Scene cleared. Our camera should now be the only one.")
                print("   Camera Entity: \(cameraEntity)")
                print("   Camera Name: \(cameraEntity.name)")
                print("   Camera FOV: \(cameraEntity.camera.fieldOfViewInDegrees)")
                print("   Camera Position: \(cameraAnchor.transform.translation)")
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
    
    // ✅ FIXED: Handle both bundle rooms and saved rooms
    private func loadModel(into arView: ARView, coordinator: Coordinator) {
        print("🎨 [RealityKitView.loadModel] ========================================")
        print("🎨 [RealityKitView.loadModel] Starting to load model: \(model.displayName)")
        print("   - Is saved room: \(model.isSavedRoom)")
        
        // Get the model URL (works for both bundle and saved rooms)
        guard let modelURL = model.temporaryURL else {
            print("❌ [RealityKitView.loadModel] CRITICAL: No URL for model!")
            print("   - Model name: \(model.name)")
            print("   - File name: \(model.fileName)")
            print("   - Is saved room: \(model.isSavedRoom)")
            return
        }
        
        print("🎨 [RealityKitView.loadModel] Got model URL: \(modelURL.path)")
        print("   - Last path component: \(modelURL.lastPathComponent)")
        
        // Verify file exists
        let fileExists = FileManager.default.fileExists(atPath: modelURL.path)
        print("   - File exists: \(fileExists)")
        
        if fileExists {
            do {
                let attributes = try FileManager.default.attributesOfItem(atPath: modelURL.path)
                if let fileSize = attributes[.size] as? UInt64 {
                    print("   - File size: \(fileSize) bytes (\(Double(fileSize) / 1024.0 / 1024.0) MB)")
                }
                let isReadable = FileManager.default.isReadableFile(atPath: modelURL.path)
                print("   - Is readable: \(isReadable)")
            } catch {
                print("   - Error getting file attributes: \(error)")
            }
        } else {
            print("❌ [RealityKitView.loadModel] CRITICAL: File does not exist at path!")
            return
        }
        
        // Load USDZ model using RealityKit's Entity loading
        print("🎨 [RealityKitView.loadModel] Starting async entity load...")
        
        Task { @MainActor in
            do {
                print("🎨 [RealityKitView.loadModel] Calling Entity.load(contentsOf:)...")
                let modelEntity = try await Entity.load(contentsOf: modelURL)
                
                print("✅ [RealityKitView.loadModel] Entity loaded successfully!")
                print("   - Entity name: '\(modelEntity.name)'")
                print("   - Entity position: \(modelEntity.position)")
                print("   - Entity scale: \(modelEntity.scale)")
                print("   - Has children: \(modelEntity.children.count)")
                
                if !modelEntity.children.isEmpty {
                    print("   - Child entities:")
                    for (index, child) in modelEntity.children.enumerated().prefix(5) {
                        print("     [\(index)] \(child.name.isEmpty ? "unnamed" : child.name) - position: \(child.position)")
                    }
                    if modelEntity.children.count > 5 {
                        print("     ... and \(modelEntity.children.count - 5) more children")
                    }
                }

                // Ensure model has proper materials for visibility
                ensureModelHasMaterials(modelEntity)

                // Calculate model bounds for camera positioning
                let bounds = modelEntity.components[ModelComponent.self]?.mesh.bounds
                if let bounds = bounds {
                    print("📦 Model bounds after loading: min(\(bounds.min)), max(\(bounds.max))")
                } else {
                    print("📦 Model bounds after loading: no bounds")
                }
                
                // Check for lights in the scene
                print("🎨 [RealityKitView.loadModel] Checking for lights in scene...")
                var lightCount = 0
                
                func countLights(in entity: Entity) {
                    if entity.components[PointLightComponent.self] != nil {
                        lightCount += 1
                        print("     💡 Found PointLight: \(entity.name.isEmpty ? "unnamed" : entity.name)")
                    }
                    if entity.components[DirectionalLightComponent.self] != nil {
                        lightCount += 1
                        print("     💡 Found DirectionalLight: \(entity.name.isEmpty ? "unnamed" : entity.name)")
                    }
                    if entity.components[SpotLightComponent.self] != nil {
                        lightCount += 1
                        print("     💡 Found SpotLight: \(entity.name.isEmpty ? "unnamed" : entity.name)")
                    }
                    
                    for child in entity.children {
                        countLights(in: child)
                    }
                }
                
                countLights(in: modelEntity)
                print("   - Total lights found in model: \(lightCount)")
                
                if lightCount == 0 {
                    print("⚠️ [RealityKitView.loadModel] WARNING: NO LIGHTS IN SCENE!")
                    print("   - This explains the black screen!")
                    print("   - Adding emergency lighting...")
                    
                    // Add emergency lighting directly to the model entity
                    let pointLight = PointLight()
                    pointLight.light.intensity = 2000
                    pointLight.light.attenuationRadius = 100
                    pointLight.position = [0, 2, 0]
                    modelEntity.addChild(pointLight)
                    print("   - ✅ Added emergency point light at [0, 2, 0]")
                    
                    let ambientLight = PointLight()
                    ambientLight.light.intensity = 1000
                    ambientLight.light.attenuationRadius = 100
                    ambientLight.position = [0, 3, 2]
                    modelEntity.addChild(ambientLight)
                    print("   - ✅ Added emergency ambient light at [0, 3, 2]")
                    
                    let fillLight = PointLight()
                    fillLight.light.intensity = 1500
                    fillLight.light.attenuationRadius = 100
                    fillLight.position = [2, 1, 2]
                    modelEntity.addChild(fillLight)
                    print("   - ✅ Added emergency fill light at [2, 1, 2]")
                }
                
                // Clean up any previous model anchor to avoid state pollution
                print("🎨 [RealityKitView.loadModel] Adding model to scene...")
                if let oldAnchor = coordinator.worldAnchor {
                    arView.scene.removeAnchor(oldAnchor)
                    coordinator.worldAnchor = nil
                    print("🧹 [RealityKitView] Removed previous model anchor from scene")
                }

                // Also remove camera anchor if it exists (will re-add after model)
                if let cameraAnchor = coordinator.cameraAnchor {
                    arView.scene.removeAnchor(cameraAnchor)
                    print("🧹 [RealityKitView] Removed camera anchor (will re-add after model)")
                }

                let modelAnchor = AnchorEntity(world: SIMD3<Float>(0, 0, 0))
                modelAnchor.addChild(modelEntity)
                arView.scene.addAnchor(modelAnchor)
                coordinator.scene = arView.scene
                
                // Store and broadcast the new world/model anchor
                coordinator.worldAnchor = modelAnchor
                // Ensure object placement manager uses the fresh scene and anchor
                arObjectPlacementManager.setSceneReferences(arView: arView, scene: arView.scene)
                arObjectPlacementManager.setWorldAnchor(modelAnchor)
                print("📌 [RealityKitView] World anchor set on placement manager")

                // Set up boundary manager for camera constraints
                let boundaryManager = RealityKitBoundaryManager(arView: arView)
                // Option B: Ensure fresh bounds per model load (avoid inheriting previous room bounds)
                boundaryManager.reset()
                print("🧹 [RealityKitView] Boundary manager reset before calculating new room bounds")
                boundaryManager.calculateRoomBounds(from: modelEntity)
                coordinator.gestureHandlers?.setBoundaryManager(boundaryManager)

                // ✅ Store boundary manager and model ID in coordinator for camera reset on revisit
                coordinator.boundaryManager = boundaryManager
                coordinator.currentModelID = self.model.id
                print("📝 [RealityKitView] Stored model ID: \(self.model.id) for tracking")

                // Share boundary manager with camera movement manager
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
                print("🔍 [RealityKitView] === CAMERA POSITIONING DEBUG ===")
                print("   cameraAnchor exists: \(coordinator.cameraAnchor != nil)")
                print("   boundaryManager.bounds exists: \(boundaryManager.bounds != nil)")

                if let cameraAnchor = coordinator.cameraAnchor, let bounds = boundaryManager.bounds {
                    print("✅ [RealityKitView] BOUNDS AVAILABLE - using BACK-LEFT CORNER positioning")
                    print("   Room bounds min: \(bounds.min)")
                    print("   Room bounds max: \(bounds.max)")

                    // ✅ Use BACK-LEFT CORNER camera positioning
                    let (cameraPosition, lookAtPosition) = boundaryManager.getOptimalCameraPosition()

                    print("📍 [RealityKitView] Camera position from getOptimalCameraPosition():")
                    print("   Position: \(cameraPosition)")
                    print("   LookAt: \(lookAtPosition)")

                    // Set camera position
                    let oldPosition = cameraAnchor.transform.translation
                    cameraAnchor.transform.translation = cameraPosition
                    print("📷 [RealityKitView] Camera translation SET:")
                    print("   OLD position: \(oldPosition)")
                    print("   NEW position: \(cameraAnchor.transform.translation)")

                    // Make camera look at the calculated target point
                    let lookDirection = normalize(lookAtPosition - cameraPosition)
                    let lookRotation = simd_quatf(from: SIMD3<Float>(0, 0, -1), to: lookDirection)
                    cameraAnchor.transform.rotation = lookRotation

                    print("📷 [RealityKitView] Camera BACK-LEFT CORNER positioned:")
                    print("   📍 Final Position: \(cameraAnchor.transform.translation)")
                    print("   👁️ Looking at: \(lookAtPosition)")
                    print("   🧭 Direction: \(lookDirection)")

                    // ✅ Add camera to scene AFTER model and AFTER positioning (ensures camera takes precedence)
                    coordinator.addCameraToScene(arView: arView)

                    // Register with GlobalCameraController
                    GlobalCameraController.shared.registerRealityKitCamera(cameraAnchor, camera: coordinator.cameraEntity)
                    print("✅ [RealityKitView] Camera registered with GlobalCameraController")
                } else if let cameraAnchor = coordinator.cameraAnchor {
                    // Fallback if no bounds - use default position
                    print("⚠️ [RealityKitView] NO BOUNDS - using DEFAULT position")
                    let defaultPosition = SIMD3<Float>(0, 1.5, 3)
                    cameraAnchor.transform.translation = defaultPosition

                    // Look toward origin
                    let lookDirection = normalize(SIMD3<Float>(0, 1.4, 0) - defaultPosition)
                    let lookRotation = simd_quatf(from: SIMD3<Float>(0, 0, -1), to: lookDirection)
                    cameraAnchor.transform.rotation = lookRotation

                    print("📷 Custom camera positioned at default: \(defaultPosition) (no bounds available)")

                    // ✅ Add camera to scene AFTER model and AFTER positioning
                    coordinator.addCameraToScene(arView: arView)

                    // Register with GlobalCameraController
                    GlobalCameraController.shared.registerRealityKitCamera(cameraAnchor, camera: coordinator.cameraEntity)
                    print("✅ [RealityKitView] Camera registered with GlobalCameraController")
                } else {
                    print("❌ [RealityKitView] NO CAMERA ANCHOR - cannot position camera!")
                }
                print("🔍 [RealityKitView] === END CAMERA POSITIONING DEBUG ===")
                
                // Set up camera movement manager with custom camera references
                self.cameraMovementManager.setupARView(arView)
                
                // Share camera references with camera movement manager for joystick control
                if let cameraAnchor = coordinator.cameraAnchor {
                    self.cameraMovementManager.setCameraAnchor(cameraAnchor)
                }
                
                // Set up camera movement callback
                self.cameraMovementManager.onCameraMove = {
                    // Camera movement callback - ready for future enhancements
                }
                
                print("✅ [RealityKitView.loadModel] Complete setup finished successfully")
                print("🎨 [RealityKitView.loadModel] ========================================")
                
            } catch {
                print("❌ [RealityKitView.loadModel] FAILED TO LOAD ENTITY!")
                print("   - Error: \(error)")
                print("   - Error description: \(error.localizedDescription)")
                print("   - Model URL: \(modelURL.path)")
                print("🎨 [RealityKitView.loadModel] ========================================")
            }
        }
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
            }
        }

        // Recursively check child entities
        for child in entity.children {
            ensureModelHasMaterials(child)
        }
    }
}

// MARK: - Extensions for SIMD math operations are defined in RealityKitObjectPlacementManager.swift

