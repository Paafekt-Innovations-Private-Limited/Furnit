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

        // Load the model and center camera once added
        loadModel(into: arView, coordinator: context.coordinator)

        // Set up camera movement manager with the ARView
        cameraMovementManager.setARView(arView)

        return arView
    }

    func updateUIView(_ uiView: ARView, context: Context) {
        // Update rendering quality if settings changed
        let currentQuality = appState.currentQuality
        configureRenderingQuality(arView: uiView, quality: currentQuality)

        // Update movement speed if settings changed
        switch appState.currentMovementSpeed {
        case .slow:   cameraMovementManager.setSpeed(.slow)
        case .normal: cameraMovementManager.setSpeed(.normal)
        case .fast:   cameraMovementManager.setSpeed(.fast)
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
            // Create perspective camera entity with reasonable field of view
            let camera = PerspectiveCamera()
            camera.camera.fieldOfViewInDegrees = 75.0
            self.cameraEntity = camera

            // Anchor will be positioned later once model bounds are known
            let anchor = AnchorEntity(world: SIMD3<Float>(0, 1.2, 3))
            self.cameraAnchor = anchor

            // Add camera entity to anchor with no offset (prevents orbital rotation)
            camera.position = SIMD3<Float>(0, 0, 0) // No offset from anchor center
            anchor.addChild(camera)
            arView.scene.addAnchor(anchor)

            // Pass camera references to gesture handlers for direct camera control
            gestureHandlers?.setCameraReferences(camera: camera, cameraAnchor: anchor)

            print("📷 Custom camera set up for non-AR mode (will be centered when model loads)")
        }

        @objc func handleTap(_ gesture: UITapGestureRecognizer) {
            // Handle AR object placement if AR is active
            Task { @MainActor in
                if let arManager = arObjectPlacementManager,
                   arManager.isReadyToPlace {
                    let location = gesture.location(in: gesture.view)
                    _ = arManager.handleTapToPlace(at: location)
                }
            }
        }
    }

    // MARK: - Model loading & scene setup

    private func loadModel(into arView: ARView, coordinator: Coordinator) {
        let isDollhouse = model.fileName.contains("dollhouse_")

        Task { @MainActor in
            do {
                let modelURL: URL
                let shouldCleanup: Bool

                if isDollhouse {
                    // Dollhouse: Load from documents directory
                    let documentsURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
                    modelURL = documentsURL.appendingPathComponent(model.fileName)
                    shouldCleanup = false // Never delete user's saved dollhouse
                    guard FileManager.default.fileExists(atPath: modelURL.path) else {
                        print("❌ Dollhouse file not found: \(modelURL.path)")
                        return
                    }
                    print("🏠 Loading dollhouse from documents: \(model.fileName)")
                } else {
                    // Regular model: Load from bundle via data asset → temp file
                    guard let dataAsset = model.dataAsset else {
                        print("❌ Failed to load data asset for model: \(model.name)")
                        return
                    }
                    modelURL = createTemporaryFile(from: dataAsset.data, fileName: "\(model.fileName).usdz")
                    shouldCleanup = true
                    print("📦 Loading model from bundle: \(model.fileName)")
                }

                // Async initializer (works in async contexts)
                let modelEntity = try await Entity(contentsOf: modelURL)

                // Ensure model has proper materials for visibility
                ensureModelHasMaterials(modelEntity)

                // In non-AR mode, simply add model to scene directly
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
                self.cameraMovementManager.setARView(arView)
                self.cameraMovementManager.setBoundaryManager(boundaryManager)

                // Set initial movement speed from settings
                switch appState.currentMovementSpeed {
                case .slow:   self.cameraMovementManager.setSpeed(.slow)
                case .normal: self.cameraMovementManager.setSpeed(.normal)
                case .fast:   self.cameraMovementManager.setSpeed(.fast)
                }

                // ---- CENTER CAMERA IN THE ROOM ----
                if let camAnchor = coordinator.cameraAnchor, let cam = coordinator.cameraEntity {
                    centerCameraInRoom(arView: arView,
                                       cameraAnchor: camAnchor,
                                       camera: cam,
                                       modelEntity: modelEntity)
                    print("📷 Camera centered inside room on load")
                }

                // Lighting
                setupLighting(for: arView)

                // Object placement wiring
                coordinator.arObjectPlacementManager = self.arObjectPlacementManager
                self.arObjectPlacementManager.setSceneReferences(arView: arView, scene: arView.scene)
                if let worldAnchor = coordinator.worldAnchor {
                    self.arObjectPlacementManager.setWorldAnchor(worldAnchor)
                    print("🌍 Connected world anchor to object placement manager")
                }

                // Camera movement manager wiring
                self.cameraMovementManager.setARView(arView)
                if let cameraAnchor = coordinator.cameraAnchor {
                    self.cameraMovementManager.setCameraAnchor(cameraAnchor)
                }
                self.cameraMovementManager.onCameraMove = { /* ready for future enhancements */ }

                print("✅ RealityKit model loaded successfully")

                // Clean up temporary file only if it was created for bundle models
                if shouldCleanup {
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
                        try? FileManager.default.removeItem(at: modelURL)
                    }
                }

            } catch {
                print("Error loading USDZ model with RealityKit: \(error.localizedDescription)")
            }
        }
    }

    // MARK: - Camera centering

    /// Position camera at the room center with a head-height offset, looking at center.
    private func centerCameraInRoom(arView: ARView,
                                    cameraAnchor: AnchorEntity,
                                    camera: PerspectiveCamera,
                                    modelEntity: Entity) {
        // Use world-space visual bounds for accuracy with nested transforms/inside-out rooms
        let vb = modelEntity.visualBounds(relativeTo: nil)
        let center = vb.center
        let extents = vb.extents

        // Human eye height w.r.t. room height; clamp to sane indoor values
        let roomHeight = max(extents.y, 2.0)
        let eyeHeight: Float = min(1.6, max(1.2, roomHeight * 0.4))

        // Place at geometric center in XZ; tiny +Z epsilon to avoid near-plane clipping
        let epsilon: Float = 0.001
        let eye    = SIMD3<Float>(center.x, center.y + eyeHeight, center.z + epsilon)
        let target = SIMD3<Float>(center.x, center.y + eyeHeight, center.z)

        // Broadly-supported API across RealityKit versions
        cameraAnchor.look(at: target, from: eye, relativeTo: nil)

        // Near/far tuned for indoor scenes
        camera.camera.near = 0.001
        camera.camera.far  = 100.0

        print("📐 Bounds center: \(center), extents: \(extents)")
        print("👁️ Eye @ \(eye), target \(target)")
    }

    // MARK: - Helpers

    private func createTemporaryFile(from data: Data, fileName: String) -> URL {
        let tempDirectory = FileManager.default.temporaryDirectory
        let tempURL = tempDirectory.appendingPathComponent(fileName)
        try? data.write(to: tempURL)
        return tempURL
    }

    private func setupLighting(for arView: ARView) {
        let quality = appState.currentQuality
        let lightingMultiplier = quality.lightingIntensity

        // Ambient-ish directional (soft)
        let ambient = DirectionalLightComponent(color: .white,
                                                intensity: Float(300 * 3.0),
                                                isRealWorldProxy: false)
        let ambientEntity = Entity()
        ambientEntity.components.set(ambient)

        // Key light
        let key = DirectionalLightComponent(color: .white,
                                            intensity: Float(800 * lightingMultiplier),
                                            isRealWorldProxy: false)
        let keyEntity = Entity()
        keyEntity.components.set(key)
        keyEntity.orientation = simd_quatf(angle: .pi / 4, axis: SIMD3<Float>(1, 0, 0))
        keyEntity.position = SIMD3<Float>(5, 10, 5)

        // Dedicated light for placed 3D objects
        let obj = DirectionalLightComponent(color: .white,
                                            intensity: Float(1500 * lightingMultiplier),
                                            isRealWorldProxy: false)
        let objEntity = Entity()
        objEntity.components.set(obj)
        objEntity.orientation = simd_quatf(angle: .pi / 6, axis: SIMD3<Float>(1, 0, 0))
        objEntity.position = SIMD3<Float>(0, 8, 2)

        let lightingAnchor = AnchorEntity(.world(transform: matrix_identity_float4x4))
        lightingAnchor.addChild(ambientEntity)
        lightingAnchor.addChild(keyEntity)
        lightingAnchor.addChild(objEntity)
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
        if var modelComponent = entity.components[ModelComponent.self] {
            if modelComponent.materials.isEmpty {
                let defaultMaterial = SimpleMaterial(color: .white, roughness: 0.5, isMetallic: false)
                modelComponent.materials = [defaultMaterial]
                entity.components.set(modelComponent)
                print("🎨 Added default white material to model entity")
            } else {
                print("🎨 Model has \(modelComponent.materials.count) existing materials")
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
