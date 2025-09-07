import SwiftUI
import SceneKit

struct SceneKitView: UIViewRepresentable {
    let model: USDZModel
    let cameraMovementManager: CameraMovementManager
    
    func makeCoordinator() -> Coordinator {
        Coordinator()
    }
    
    func makeUIView(context: Context) -> SCNView {
        let scnView = SCNView()
        
        scnView.allowsCameraControl = false
        scnView.autoenablesDefaultLighting = false
        scnView.antialiasingMode = .multisampling4X
        scnView.backgroundColor = UIColor.black
        scnView.rendersContinuously = false
        
        context.coordinator.setupGestures(for: scnView)
        loadScene(into: scnView, coordinator: context.coordinator)
        
        // Set up camera movement manager with the scene view
        cameraMovementManager.setSceneView(scnView)
        
        return scnView
    }
    
    func updateUIView(_ uiView: SCNView, context: Context) {
        if uiView.scene == nil {
            loadScene(into: uiView, coordinator: context.coordinator)
        }
    }
    
    class Coordinator {
        var gestureHandlers: GestureHandlers?
        var scene: SCNScene?
        
        func setupGestures(for scnView: SCNView) {
            gestureHandlers = GestureHandlers(scnView: scnView)
        }
    }
    
    private func loadScene(into scnView: SCNView, coordinator: Coordinator) {
        guard let dataAsset = model.dataAsset else {
            print("Failed to load data asset for model: \(model.name)")
            return
        }
        
        do {
            let tempURL = createTemporaryFile(from: dataAsset.data, fileName: "\(model.fileName).usdz")
            
            let loadingOptions: [SCNSceneSource.LoadingOption: Any] = [
                .animationImportPolicy: SCNSceneSource.AnimationImportPolicy.playRepeatedly,
                .checkConsistency: true,
                .strictConformance: false
            ]
            
            let scene = try SCNScene(url: tempURL, options: loadingOptions)
            
            let boundingBox = calculateSceneBounds(scene)
            setupCamera(for: scene, with: boundingBox)
            setupLighting(for: scene)
            
            let boundaryManager = BoundaryManager(scnView: scnView)
            boundaryManager.calculateRoomBounds(from: scene)
            coordinator.gestureHandlers?.setBoundaryManager(boundaryManager)
            
            DispatchQueue.main.async {
                scnView.scene = scene
                coordinator.scene = scene
                
                // Update camera movement manager with scene and boundaries
                self.cameraMovementManager.setSceneView(scnView)
                
                // Make sure boundaries are calculated after scene is set
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
                    self.cameraMovementManager.updateBoundaries()
                }
            }
            
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
                try? FileManager.default.removeItem(at: tempURL)
            }
            
        } catch {
            print("Error loading USDZ file: \(error.localizedDescription)")
        }
    }
    
    private func createTemporaryFile(from data: Data, fileName: String) -> URL {
        let tempDirectory = FileManager.default.temporaryDirectory
        let tempURL = tempDirectory.appendingPathComponent(fileName)
        
        try? data.write(to: tempURL)
        return tempURL
    }
    
    private func calculateSceneBounds(_ scene: SCNScene) -> (min: SCNVector3, max: SCNVector3) {
        var minVec = SCNVector3(Float.greatestFiniteMagnitude, Float.greatestFiniteMagnitude, Float.greatestFiniteMagnitude)
        var maxVec = SCNVector3(-Float.greatestFiniteMagnitude, -Float.greatestFiniteMagnitude, -Float.greatestFiniteMagnitude)
        
        scene.rootNode.enumerateChildNodes { node, _ in
            let (localMin, localMax) = node.boundingBox
            let worldTransform = node.worldTransform
            
            let corners = [
                worldTransform * SCNVector3(localMin.x, localMin.y, localMin.z),
                worldTransform * SCNVector3(localMax.x, localMin.y, localMin.z),
                worldTransform * SCNVector3(localMin.x, localMax.y, localMin.z),
                worldTransform * SCNVector3(localMax.x, localMax.y, localMin.z),
                worldTransform * SCNVector3(localMin.x, localMin.y, localMax.z),
                worldTransform * SCNVector3(localMax.x, localMin.y, localMax.z),
                worldTransform * SCNVector3(localMin.x, localMax.y, localMax.z),
                worldTransform * SCNVector3(localMax.x, localMax.y, localMax.z)
            ]
            
            for corner in corners {
                minVec.x = min(minVec.x, corner.x)
                minVec.y = min(minVec.y, corner.y)
                minVec.z = min(minVec.z, corner.z)
                maxVec.x = max(maxVec.x, corner.x)
                maxVec.y = max(maxVec.y, corner.y)
                maxVec.z = max(maxVec.z, corner.z)
            }
        }
        
        return (min: minVec, max: maxVec)
    }
    
    private func setupCamera(for scene: SCNScene, with boundingBox: (min: SCNVector3, max: SCNVector3)) {
        // Create camera node with proper configuration
        let cameraNode = SCNNode()
        cameraNode.camera = SCNCamera()
        
        // Calculate room center point
        let roomCenter = SCNVector3(
            (boundingBox.min.x + boundingBox.max.x) / 2,
            (boundingBox.min.y + boundingBox.max.y) / 2,
            (boundingBox.min.z + boundingBox.max.z) / 2
        )
        
        // Calculate room dimensions
        let roomSize = SCNVector3(
            boundingBox.max.x - boundingBox.min.x,
            boundingBox.max.y - boundingBox.min.y,
            boundingBox.max.z - boundingBox.min.z
        )
        
        // Position camera INSIDE the room, slightly above floor level
        // Use a smaller distance to stay within room boundaries
        let cameraHeight = boundingBox.min.y + (roomSize.y * 0.4) // 40% up from floor
        let viewingDistance = min(roomSize.x, roomSize.z) * 0.3 // 30% of smaller horizontal dimension
        
        // Position camera inside room, looking toward the center
        let cameraPosition = SCNVector3(
            roomCenter.x - viewingDistance,
            cameraHeight,
            roomCenter.z + viewingDistance * 0.5
        )
        
        cameraNode.position = cameraPosition
        
        // Look at a point slightly above the room center for better view
        let lookAtPoint = SCNVector3(
            roomCenter.x,
            roomCenter.y,
            roomCenter.z
        )
        cameraNode.look(at: lookAtPoint)
        
        // Configure camera properties for room viewing
        cameraNode.camera?.fieldOfView = 75 // Wider field of view for room interiors
        cameraNode.camera?.zNear = 0.1
        cameraNode.camera?.zFar = Double(max(roomSize.x, max(roomSize.y, roomSize.z)) * 3)
        
        scene.rootNode.addChildNode(cameraNode)
    }
    
    private func setupLighting(for scene: SCNScene) {
        scene.lightingEnvironment.contents = UIColor.systemBackground
        scene.lightingEnvironment.intensity = 0.8
        
        let ambientLightNode = SCNNode()
        ambientLightNode.light = SCNLight()
        ambientLightNode.light!.type = .ambient
        ambientLightNode.light!.color = UIColor(white: 0.6, alpha: 1.0)
        ambientLightNode.light!.intensity = 300
        scene.rootNode.addChildNode(ambientLightNode)
        
        let keyLightNode = SCNNode()
        keyLightNode.light = SCNLight()
        keyLightNode.light!.type = .directional
        keyLightNode.light!.color = UIColor.white
        keyLightNode.light!.intensity = 800
        keyLightNode.position = SCNVector3(x: 5, y: 10, z: 5)
        keyLightNode.look(at: SCNVector3(0, 0, 0))
        scene.rootNode.addChildNode(keyLightNode)
        
        let fillLightNode = SCNNode()
        fillLightNode.light = SCNLight()
        fillLightNode.light!.type = .directional
        fillLightNode.light!.color = UIColor(white: 0.9, alpha: 1.0)
        fillLightNode.light!.intensity = 400
        fillLightNode.position = SCNVector3(x: -3, y: 5, z: 8)
        fillLightNode.look(at: SCNVector3(0, 0, 0))
        scene.rootNode.addChildNode(fillLightNode)
    }
}

extension SCNMatrix4 {
    static func * (matrix: SCNMatrix4, vector: SCNVector3) -> SCNVector3 {
        let x = matrix.m11 * vector.x + matrix.m21 * vector.y + matrix.m31 * vector.z + matrix.m41
        let y = matrix.m12 * vector.x + matrix.m22 * vector.y + matrix.m32 * vector.z + matrix.m42
        let z = matrix.m13 * vector.x + matrix.m23 * vector.y + matrix.m33 * vector.z + matrix.m43
        return SCNVector3(x, y, z)
    }
}
