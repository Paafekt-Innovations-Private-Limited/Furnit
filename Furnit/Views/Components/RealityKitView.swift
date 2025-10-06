import SwiftUI
import RealityKit
import ARKit
import Combine

struct RealityKitView: UIViewRepresentable {
    let model: USDZModel
    @ObservedObject var cameraMovementManager: RealityKitCameraMovementManager
    @ObservedObject var arObjectPlacementManager: RealityKitObjectPlacementManager
    let isARActive: Bool
    
    func makeUIView(context: Context) -> ARView {
        let arView = ARView(frame: .zero)
        
        // Configure AR session
        if isARActive {
            let config = ARWorldTrackingConfiguration()
            config.planeDetection = [.horizontal]
            arView.session.run(config)
        } else {
            // Non-AR mode
            arView.cameraMode = .nonAR
        }
        
        // Set the ARView reference immediately for proper timing
        cameraMovementManager.setARView(arView)
        print("🎮 [RealityKitView] ARView set in camera movement manager (sync)")
        
        // Set up the scene
        setupScene(arView: arView, context: context)
        
        // Setup gesture handlers with the ARView
        context.coordinator.setupGestureHandlers(for: arView)
        
        return arView
    }
    
    func updateUIView(_ uiView: ARView, context: Context) {
        // CRITICAL: Ensure ARView reference is always current
        if cameraMovementManager.arView !== uiView {
            print("🔄 [RealityKitView] Updating ARView reference in movement manager")
            cameraMovementManager.setARView(uiView)
            
            // Also update coordinator's reference
            if context.coordinator.arView !== uiView {
                context.coordinator.arView = uiView
                context.coordinator.setupGestureHandlers(for: uiView)
            }
        }
        
        // Update AR session if needed
        if isARActive {
            if uiView.session.configuration == nil {
                let config = ARWorldTrackingConfiguration()
                config.planeDetection = [.horizontal]
                uiView.session.run(config)
            }
        } else {
            if uiView.session.configuration != nil {
                uiView.session.pause()
            }
        }
    }
    
    func makeCoordinator() -> Coordinator {
        Coordinator(
            model: model,
            cameraMovementManager: cameraMovementManager,
            arObjectPlacementManager: arObjectPlacementManager
        )
    }
    
    private func setupScene(arView: ARView, context: Context) {
        // Check if it's a dollhouse room
        let isDollhouse = model.fileName.contains("dollhouse_")
        
        print("🔍 [setupScene] isDollhouse: \(isDollhouse), isARActive: \(isARActive)")
        
        // Create main anchor
        let anchor = AnchorEntity(world: .zero)
        arView.scene.anchors.append(anchor)
        
        // Set up lighting
        setupLighting(arView: arView, isDollhouse: isDollhouse)
        
        // Load the model AND set up boundaries FIRST
        if isDollhouse {
            loadDollhouseModel(arView: arView, anchor: anchor, context: context)
        } else {
            loadRegularModel(arView: arView, anchor: anchor, context: context)
        }
        
        // Set up camera for non-AR mode
        // Set up camera immediately for better timing
        print("🔍 [setupScene] About to check camera setup - isARActive: \(isARActive)")
        if !isARActive || isDollhouse {
            print("✅ [setupScene] Calling setupCamera")
            setupCamera(arView: arView, anchor: anchor, isDollhouse: isDollhouse, context: context)
        } else {
            print("❌ [setupScene] Skipping setupCamera because isARActive = true")
        }
    }
    
    private func setupLighting(arView: ARView, isDollhouse: Bool) {
        // Configure lighting based on model type
        if isDollhouse {
            // Brighter lighting for dollhouse interiors
            arView.environment.lighting.intensityExponent = 2.0
            
            // Add directional light
            let light = DirectionalLight()
            light.light.intensity = 10000
            light.light.isRealWorldProxy = false
            light.position = [0, 10, 0]
            light.look(at: [0, 0, 0], from: light.position, relativeTo: nil)
            
            // Add to scene
            let lightAnchor = AnchorEntity(world: light.position)
            lightAnchor.addChild(light)
            arView.scene.anchors.append(lightAnchor)
            
            print("💡 Added dedicated lighting for dollhouse")
        } else {
            // Standard lighting for regular models
            arView.environment.lighting.intensityExponent = 1.0
        }
    }
    
    private func loadDollhouseModel(arView: ARView, anchor: AnchorEntity, context: Context) {
        print("🏠 Loading dollhouse model: \(model.fileName)")
        
        // Get documents directory URL
        let documentsURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let fileURL = documentsURL.appendingPathComponent(model.fileName)
        
        if FileManager.default.fileExists(atPath: fileURL.path) {
            do {
                // Load the USDZ file
                let entity = try Entity.load(contentsOf: fileURL)
                
                // Scale and position for dollhouse
                entity.scale = [1, 1, 1]
                entity.position = [0, 0, 0]
                
                // Add to anchor
                anchor.addChild(entity)
                
                // Set up boundary manager for dollhouse
                setupBoundaryManager(arView: arView, modelEntity: entity)
                
                print("✅ Dollhouse model loaded successfully")
                
            } catch {
                print("❌ Failed to load dollhouse: \(error)")
                // Add placeholder
                addPlaceholderModel(to: anchor)
            }
        } else {
            print("❌ Dollhouse file not found: \(fileURL.path)")
            // Add placeholder
            addPlaceholderModel(to: anchor)
        }
    }
    
    private func loadRegularModel(arView: ARView, anchor: AnchorEntity, context: Context) {
        print("📦 Loading regular model: \(model.fileName)")
        
        // Try to load from bundle
        guard let url = Bundle.main.url(forResource: model.fileName, withExtension: nil) else {
            print("❌ Model not found in bundle: \(model.fileName)")
            addPlaceholderModel(to: anchor)
            return
        }
        
        do {
            let entity = try Entity.load(contentsOf: url)
            entity.scale = [1, 1, 1]
            entity.position = [0, 0, 0]
            anchor.addChild(entity)
            
            // Set up boundary manager
            setupBoundaryManager(arView: arView, modelEntity: entity)
            
            print("✅ Regular model loaded successfully")
            
        } catch {
            print("❌ Failed to load model: \(error)")
            addPlaceholderModel(to: anchor)
        }
    }
    
    private func setupBoundaryManager(arView: ARView, modelEntity: Entity) {
        // Create and configure boundary manager
        let boundaryManager = RealityKitBoundaryManager(arView: arView)
        
        // Calculate bounds from the model entity
        boundaryManager.calculateRoomBounds(from: modelEntity)
        
        // Set boundary manager in camera movement manager
        cameraMovementManager.setBoundaryManager(boundaryManager)
        
        // Log the bounds for debugging
        if let bounds = boundaryManager.getCurrentBounds() {
            print("📏 Boundary manager configured with bounds: min=\(bounds.min), max=\(bounds.max)")
        }
    }
    
    private func setupCamera(arView: ARView, anchor: AnchorEntity, isDollhouse: Bool, context: Context) {
        print("🔥 [setupCamera] CALLED - isDollhouse: \(isDollhouse)")
            
        // Create camera entity
        let cameraEntity = PerspectiveCamera()
        cameraEntity.camera.fieldOfViewInDegrees = 60
        
        // Create camera anchor
        let cameraAnchor = AnchorEntity(world: .zero)
        print("🔥 [setupCamera] Camera anchor created: \(cameraAnchor)")
            
        
        
        // Set initial position based on model type
        if isDollhouse {
            // For dollhouse, try to get center from boundary manager
            if let boundaryManager = cameraMovementManager.getBoundaryManager(),
               let bounds = boundaryManager.getCurrentBounds() {
                // Start camera at center of the room but closer to entrance
                let center = (bounds.min + bounds.max) / 2.0
                cameraAnchor.position = [center.x, 1.5, center.z]
                print("📷 Camera positioned at dollhouse center: \(cameraAnchor.position)")
            } else {
                // Fallback position if bounds not available
                cameraAnchor.position = [0, 1.5, 0]
                print("📷 Camera positioned at fallback position for dollhouse")
            }
        } else {
            // Position camera for regular model viewing
            cameraAnchor.position = [0, 1.5, 5]
            print("📷 Camera positioned for regular model at: \(cameraAnchor.position)")
        }
        
        // Add camera to anchor
        cameraAnchor.addChild(cameraEntity)
        arView.scene.anchors.append(cameraAnchor)
        
        // Set camera in ARView
        arView.cameraMode = .nonAR
        
        // Store camera anchor in coordinator
        context.coordinator.cameraAnchor = cameraAnchor
        print("🔥 [setupCamera] Coordinator camera anchor set")
            
        print("🔥 [setupCamera] About to set camera anchor in movement manager")
        print("🔥 [setupCamera] Movement manager exists: \(cameraMovementManager)")
            
        // Set camera anchor in movement manager immediately
        cameraMovementManager.setCameraAnchor(cameraAnchor)
        print("🔥 [setupCamera] setCameraAnchor called")
            
        print("🎮 [RealityKitView] Camera anchor set in movement manager (sync)")
        
        // Force a readiness check after both are set
//        if cameraMovementManager.arView != nil {
//            print("✅ Both ARView and camera anchor should now be ready")
//            print("✅ Final ready state: \(cameraMovementManager.isReady)")
//            
//            // Force ready state if both components are set but readiness check failed
//            if !cameraMovementManager.isReady {
//                print("🔧 Manually setting ready state to true")
//                cameraMovementManager.forceReady()
//            }
//        }
        
        // Set appropriate movement speed for dollhouse
//        if isDollhouse {
//            cameraMovementManager.setSpeed(.fast)
//            print("🏃 Set fast movement speed for dollhouse")
//        }
        
        // For dollhouse rooms, use the SAME system as first 2 rooms (the working approach)
//        if isDollhouse {
//            // Optimize camera movement settings for dollhouse exploration  
//            cameraMovementManager.optimizeForDollhouse()
//            
//            // Use the EXACT same working system as first 2 rooms
//            cameraMovementManager.setCameraMovementEnabled(true)  
//            print("🏠 Using same proven camera system as first 2 rooms, optimized for dollhouse")
//            
//            // Force ready state for dollhouse
//            cameraMovementManager.forceReady()
//            print("✅ Dollhouse ready with working approach - no timer needed!")
//        } else {
            // For regular models, use the normal camera movement system
            cameraMovementManager.setSpeed(.normal)
            cameraMovementManager.setCameraMovementEnabled(true)
            print("📦 Using standard camera movement for regular model")
//        }
        
        // Store world anchor reference for object placement
        arObjectPlacementManager.setWorldAnchor(anchor)
    }
    
    private func addPlaceholderModel(to anchor: AnchorEntity) {
        // Create a simple box as placeholder
        let mesh = MeshResource.generateBox(size: 2)
        var material = SimpleMaterial()
        material.color = .init(tint: .gray)
        let entity = ModelEntity(mesh: mesh, materials: [material])
        anchor.addChild(entity)
        print("📦 Added placeholder box model")
    }
    
    // Enhanced joystick movement system for dollhouses using camera manager
    private func setupSimpleJoystickMovement(cameraAnchor: AnchorEntity, context: Context) {
        print("🕹️ Setting up enhanced joystick movement system for dollhouse...")
        
        // Store direct references in coordinator to avoid weak reference issues
        context.coordinator.cameraAnchor = cameraAnchor
        
        // Verify references are set correctly
        print("🔍 Direct references check:")
        print("   - ARView: \(context.coordinator.arView != nil)")
        print("   - CameraAnchor: \(context.coordinator.cameraAnchor != nil)")
        
        // Set up a timer that uses the camera manager's sophisticated movement handling
        let movementTimer = Timer.scheduledTimer(withTimeInterval: 1.0/60.0, repeats: true) { _ in
            // Get current joystick state from the camera manager
            let joystickOffset = self.cameraMovementManager.currentJoystickOffset
            
            // Only move if joystick is active
            let magnitude = sqrt(joystickOffset.width * joystickOffset.width + joystickOffset.height * joystickOffset.height)
            guard magnitude > 5.0 else { return } // Use same dead zone
            
            // Debug: Log that our timer is running with joystick input
            if magnitude > 20.0 { // Only for significant input
                print("🎮 [Timer] Processing joystick: magnitude=\(String(format: "%.1f", magnitude))")
            }
            
            // Use direct references to avoid the "No ARView" issue
            guard let arView = context.coordinator.arView,
                  let cameraAnchor = context.coordinator.cameraAnchor else {
                if magnitude > 20.0 { // Only log for significant input
                    print("⚠️ [Timer] Missing direct references: arView=\(context.coordinator.arView != nil), cameraAnchor=\(context.coordinator.cameraAnchor != nil)")
                }
                return
            }
            
            // Add debug before calling movement method
            if magnitude > 20.0 {
                print("🎯 [Timer] About to call camera movement method")
            }
            
            // Use the camera movement manager's sophisticated movement handling
            // This should produce the same detailed logs as the first 2 rooms
            self.cameraMovementManager.updateCameraPositionWithDirectReferences(
                arView: arView,
                cameraAnchor: cameraAnchor
            )
            
            // Log successful processing
            if magnitude > 20.0 {
                print("✅ [Timer] Camera movement method completed")
            }
        }
        
        // Store timer in coordinator so it gets cleaned up
        context.coordinator.cameraUpdateTimer = movementTimer
        print("🕹️ Enhanced joystick movement timer started with 60fps updates!")
    }
    
    // MARK: - Coordinator
    class Coordinator: NSObject {
        let model: USDZModel
        let cameraMovementManager: RealityKitCameraMovementManager
        let arObjectPlacementManager: RealityKitObjectPlacementManager
        var gestureHandlers: RealityKitGestureHandlers?
        var cameraAnchor: AnchorEntity?
        var arView: ARView?
        var cameraUpdateTimer: Timer?
        
        init(model: USDZModel,
             cameraMovementManager: RealityKitCameraMovementManager,
             arObjectPlacementManager: RealityKitObjectPlacementManager) {
            self.model = model
            self.cameraMovementManager = cameraMovementManager
            self.arObjectPlacementManager = arObjectPlacementManager
            super.init()
        }
        
        deinit {
            cameraUpdateTimer?.invalidate()
        }
        
        func setupGestureHandlers(for arView: ARView) {
            self.arView = arView
            gestureHandlers = RealityKitGestureHandlers(arView: arView)
            gestureHandlers?.setObjectPlacementManager(arObjectPlacementManager)
            
            // Set camera references if available
            if let cameraAnchor = cameraAnchor {
                if let camera = cameraAnchor.children.first as? PerspectiveCamera {
                    gestureHandlers?.setCameraReferences(camera: camera, cameraAnchor: cameraAnchor)
                }
            }
        }
    }
}
