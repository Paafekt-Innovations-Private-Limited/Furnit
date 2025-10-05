import SwiftUI
import RealityKit
import ARKit
import Combine

struct RealityKitView: UIViewRepresentable {
    let model: USDZModel
    @ObservedObject var cameraMovementManager: RealityKitCameraMovementManager
    @ObservedObject var arObjectPlacementManager: RealityKitObjectPlacementManager
    let isARActive: Bool
    
    // Add a loading state
    @State private var isLoading = true
    
    func makeUIView(context: Context) -> ARView {
        let arView = ARView(frame: .zero)
        
        print("🚀 [RealityKitView] Starting ARView setup...")
        
        // Configure AR session
        if isARActive {
            let config = ARWorldTrackingConfiguration()
            config.planeDetection = [.horizontal]
            arView.session.run(config)
            print("📱 AR session configured and started")
        } else {
            // Non-AR mode
            arView.cameraMode = .nonAR
            print("🎮 Non-AR mode configured")
        }
        
        // Set the ARView reference immediately (synchronously first for safety)
        cameraMovementManager.setARView(arView)
        print("🎮 [RealityKitView] ARView set in camera movement manager (sync)")
        
        // Set up the scene asynchronously to prevent blocking
        Task {
            await setupSceneAsync(arView: arView, context: context)
        }
        
        // Setup gesture handlers with the ARView
        context.coordinator.setupGestureHandlers(for: arView)
        
        return arView
    }
    
    func updateUIView(_ uiView: ARView, context: Context) {
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
    
    private func setupSceneAsync(arView: ARView, context: Context) async {
        print("🏗️ [RealityKitView] Starting async scene setup...")
        
        // Check if it's a dollhouse room
        let isDollhouse = model.fileName.contains("dollhouse_")
        
        // Create main anchor on main thread
        await MainActor.run {
            let anchor = AnchorEntity(world: .zero)
            arView.scene.anchors.append(anchor)
            context.coordinator.worldAnchor = anchor
            print("⚓ World anchor created and added to scene")
        }
        
        // Set up lighting on main thread
        await MainActor.run {
            setupLighting(arView: arView, isDollhouse: isDollhouse)
        }
        
        // Load model asynchronously (this is the heavy operation)
        let modelEntity = await loadModelAsync(isDollhouse: isDollhouse)
        
        // Add model to scene on main thread
        await MainActor.run {
            guard let anchor = context.coordinator.worldAnchor else {
                print("❌ World anchor not found!")
                self.isLoading = false
                return
            }
            
            if let entity = modelEntity {
                anchor.addChild(entity)
                print("✅ Model added to scene successfully")
                
                // Set up boundary manager
                setupBoundaryManager(arView: arView, modelEntity: entity)
                
                // Set up camera for non-AR mode after model is loaded
                if !self.isARActive {
                    self.setupCamera(arView: arView, anchor: anchor, isDollhouse: isDollhouse, context: context)
                }
            } else {
                print("❌ Failed to load model, adding placeholder")
                self.addPlaceholderModel(to: anchor)
                
                // Still set up camera with placeholder
                if !self.isARActive {
                    self.setupCamera(arView: arView, anchor: anchor, isDollhouse: isDollhouse, context: context)
                }
            }
            
            // Mark loading as complete
            self.isLoading = false
        }
        
        print("🎉 [RealityKitView] Async scene setup completed")
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
    
    private func loadModelAsync(isDollhouse: Bool) async -> Entity? {
        print("📦 [RealityKitView] Starting async model loading...")
        
        if isDollhouse {
            return await loadDollhouseModelAsync()
        } else {
            return await loadRegularModelAsync()
        }
    }
    
    private func loadDollhouseModelAsync() async -> Entity? {
        print("🏠 Loading dollhouse model: \(model.fileName)")
        
        // Get documents directory URL
        let documentsURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let fileURL = documentsURL.appendingPathComponent(model.fileName)
        
        // Check if file exists
        guard FileManager.default.fileExists(atPath: fileURL.path) else {
            print("❌ Dollhouse file not found: \(fileURL.path)")
            return nil
        }
        
        do {
            // Load the USDZ file asynchronously
            print("🔄 Loading dollhouse from: \(fileURL.path)")
            let entity = try await Entity.load(contentsOf: fileURL)
            
            // Configure entity on background thread
            entity.scale = [1, 1, 1]
            entity.position = [0, 0, 0]
            
            print("✅ Dollhouse model loaded successfully")
            return entity
            
        } catch {
            print("❌ Failed to load dollhouse: \(error.localizedDescription)")
            return nil
        }
    }
    
    private func loadRegularModelAsync() async -> Entity? {
        print("📦 Loading regular model: \(model.fileName)")
        
        // Try to load from bundle
        guard let url = Bundle.main.url(forResource: model.fileName, withExtension: nil) else {
            print("❌ Model not found in bundle: \(model.fileName)")
            return nil
        }
        
        do {
            print("🔄 Loading regular model from bundle: \(url.path)")
            let entity = try await Entity.load(contentsOf: url)
            entity.scale = [1, 1, 1]
            entity.position = [0, 0, 0]
            
            print("✅ Regular model loaded successfully")
            return entity
            
        } catch {
            print("❌ Failed to load regular model: \(error.localizedDescription)")
            return nil
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
        print("📷 [RealityKitView] Setting up camera...")
        
        // Create camera entity
        let cameraEntity = PerspectiveCamera()
        cameraEntity.camera.fieldOfViewInDegrees = 60
        
        // Create camera anchor
        let cameraAnchor = AnchorEntity(world: .zero)
        
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
        
        // Set camera anchor in movement manager (now synchronously since we're already on main thread)
        cameraMovementManager.setCameraAnchor(cameraAnchor)
        print("🎮 [RealityKitView] Camera anchor set in movement manager")
        
        // Verify readiness
        if cameraMovementManager.arView != nil {
            print("✅ Both ARView and camera anchor are now ready")
        }
        
        // Set appropriate movement speed for dollhouse
        if isDollhouse {
            cameraMovementManager.setSpeed(.fast)
            print("🏃 Set fast movement speed for dollhouse")
        }
        
        // Store world anchor reference for object placement
        arObjectPlacementManager.setWorldAnchor(anchor)
        
        print("📷 [RealityKitView] Camera setup completed")
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
    
    // MARK: - Coordinator
    class Coordinator: NSObject {
        let model: USDZModel
        let cameraMovementManager: RealityKitCameraMovementManager
        let arObjectPlacementManager: RealityKitObjectPlacementManager
        var gestureHandlers: RealityKitGestureHandlers?
        var cameraAnchor: AnchorEntity?
        var worldAnchor: AnchorEntity?
        
        init(model: USDZModel,
             cameraMovementManager: RealityKitCameraMovementManager,
             arObjectPlacementManager: RealityKitObjectPlacementManager) {
            self.model = model
            self.cameraMovementManager = cameraMovementManager
            self.arObjectPlacementManager = arObjectPlacementManager
            super.init()
        }
        
        func setupGestureHandlers(for arView: ARView) {
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
