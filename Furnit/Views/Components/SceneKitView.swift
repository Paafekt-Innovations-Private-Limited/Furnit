import SwiftUI
import SceneKit

struct SceneKitView: UIViewRepresentable {
    let model: USDZModel
    let cameraMovementManager: CameraMovementManager
    let arObjectPlacementManager: ARObjectPlacementManager?
    
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
        
        context.coordinator.setupGestures(for: scnView, arObjectPlacementManager: arObjectPlacementManager)
        loadScene(into: scnView, coordinator: context.coordinator)
        
        // Set up camera movement manager with the scene view
        cameraMovementManager.setSceneView(scnView)
        
        // Set up AR object placement manager with the scene view
        arObjectPlacementManager?.setSceneView(scnView)
        
        return scnView
    }
    
    func updateUIView(_ uiView: SCNView, context: Context) {
        if uiView.scene == nil {
            loadScene(into: uiView, coordinator: context.coordinator)
        }
    }
    
    class Coordinator {
        var gestureHandlers: GestureHandlers?
        var arObjectGestureHandler: ARObjectGestureHandler?
        var scene: SCNScene?
        
        func setupGestures(for scnView: SCNView, arObjectPlacementManager: ARObjectPlacementManager?) {
            gestureHandlers = GestureHandlers(scnView: scnView)
            
            // Setup AR object manipulation gestures
            if let arManager = arObjectPlacementManager {
                arObjectGestureHandler = ARObjectGestureHandler(scnView: scnView, arManager: arManager)
                arObjectGestureHandler?.setupGestures()
            }
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

// MARK: - AR Object Gesture Handler
class ARObjectGestureHandler {
    private weak var scnView: SCNView?
    private weak var arManager: ARObjectPlacementManager?
    
    private var selectedObject: SCNNode?
    private var initialTouchPoint: CGPoint = .zero
    private var initialObjectPosition: SCNVector3 = SCNVector3Zero
    private var isDragging = false
    
    init(scnView: SCNView, arManager: ARObjectPlacementManager) {
        self.scnView = scnView
        self.arManager = arManager
    }
    
    func setupGestures() {
        guard let scnView = scnView else { return }
        
        // Pan gesture for moving objects
        let panGesture = UIPanGestureRecognizer(target: self, action: #selector(handlePanGesture(_:)))
        scnView.addGestureRecognizer(panGesture)
        
        // Tap gesture for selecting objects
        let tapGesture = UITapGestureRecognizer(target: self, action: #selector(handleTapGesture(_:)))
        scnView.addGestureRecognizer(tapGesture)
        
        // Pinch gesture for scaling objects
        let pinchGesture = UIPinchGestureRecognizer(target: self, action: #selector(handlePinchGesture(_:)))
        scnView.addGestureRecognizer(pinchGesture)
        
        // Rotation gesture for rotating objects
        let rotationGesture = UIRotationGestureRecognizer(target: self, action: #selector(handleRotationGesture(_:)))
        scnView.addGestureRecognizer(rotationGesture)
        
        print("✨ AR object manipulation gestures setup complete")
    }
    
    @objc private func handleTapGesture(_ gesture: UITapGestureRecognizer) {
        guard let scnView = scnView else { return }
        
        let touchLocation = gesture.location(in: scnView)
        let hitResults = scnView.hitTest(touchLocation, options: [:])
        
        // Clear previous selection
        clearSelection()
        
        // Find AR objects (persistent or active)
        for hitResult in hitResults {
            let node = hitResult.node
            
            // Check if this is an AR object by checking if it's managed by ARObjectPlacementManager
            if isARObject(node) {
                selectObject(node)
                break
            }
        }
    }
    
    @objc private func handlePanGesture(_ gesture: UIPanGestureRecognizer) {
        guard let scnView = scnView,
              let selectedObject = selectedObject else { return }
        
        let touchLocation = gesture.location(in: scnView)
        
        switch gesture.state {
        case .began:
            initialTouchPoint = touchLocation
            initialObjectPosition = selectedObject.position
            isDragging = true
            
        case .changed:
            let translation = CGPoint(
                x: touchLocation.x - initialTouchPoint.x,
                y: touchLocation.y - initialTouchPoint.y
            )
            
            // Convert screen translation to 3D world coordinates
            let worldTranslation = convertScreenToWorldTranslation(translation, for: selectedObject)
            
            selectedObject.position = SCNVector3(
                initialObjectPosition.x + worldTranslation.x,
                initialObjectPosition.y + worldTranslation.y,
                initialObjectPosition.z + worldTranslation.z
            )
            
        case .ended, .cancelled:
            isDragging = false
            print("📋 Moved AR object to position: \(selectedObject.position)")
            
        default:
            break
        }
    }
    
    @objc private func handlePinchGesture(_ gesture: UIPinchGestureRecognizer) {
        guard let selectedObject = selectedObject else { return }
        
        switch gesture.state {
        case .changed:
            let scale = Float(gesture.scale)
            let currentScale = selectedObject.scale
            
            // Apply scaling with limits
            let newScale = SCNVector3(
                max(0.5, min(3.0, currentScale.x * scale)), // Limit scale between 0.5x and 3.0x
                max(0.5, min(3.0, currentScale.y * scale)),
                max(0.5, min(3.0, currentScale.z * scale))
            )
            
            selectedObject.scale = newScale
            gesture.scale = 1.0 // Reset gesture scale
            
        case .ended:
            print("🔍 Scaled AR object to: \(selectedObject.scale)")
            
        default:
            break
        }
    }
    
    @objc private func handleRotationGesture(_ gesture: UIRotationGestureRecognizer) {
        guard let selectedObject = selectedObject else { return }
        
        switch gesture.state {
        case .changed:
            // Apply rotation around Y axis (vertical rotation)
            let rotationY = Float(gesture.rotation)
            selectedObject.eulerAngles.y += rotationY
            gesture.rotation = 0.0 // Reset gesture rotation
            
        case .ended:
            print("🌀 Rotated AR object to: \(selectedObject.eulerAngles)")
            
        default:
            break
        }
    }
    
    private func isARObject(_ node: SCNNode) -> Bool {
        // Check if node name contains AR object identifiers
        guard let nodeName = node.name else { return false }
        return nodeName.contains("ar_object_") || nodeName.contains("_interactive")
    }
    
    private func selectObject(_ node: SCNNode) {
        selectedObject = node
        
        // Add visual selection indicator
        addSelectionIndicator(to: node)
        
        // Add haptic feedback
        let impact = UIImpactFeedbackGenerator(style: .medium)
        impact.impactOccurred()
        
        print("✨ Selected AR object: \(node.name ?? "Unknown")")
    }
    
    private func clearSelection() {
        if let selected = selectedObject {
            removeSelectionIndicator(from: selected)
        }
        selectedObject = nil
    }
    
    private func addSelectionIndicator(to node: SCNNode) {
        // Remove existing selection indicator
        removeSelectionIndicator(from: node)
        
        // Create selection outline
        guard let geometry = node.geometry else { return }
        
        let selectionNode = SCNNode(geometry: geometry)
        selectionNode.name = "selection_indicator"
        
        // Create wireframe material for selection
        let selectionMaterial = SCNMaterial()
        selectionMaterial.diffuse.contents = UIColor.cyan.withAlphaComponent(0.3)
        selectionMaterial.emission.contents = UIColor.cyan.withAlphaComponent(0.5)
        selectionMaterial.fillMode = .lines
        selectionMaterial.isDoubleSided = true
        
        selectionNode.geometry?.materials = [selectionMaterial]
        selectionNode.scale = SCNVector3(1.05, 1.05, 1.05) // Slightly larger for visibility
        
        node.addChildNode(selectionNode)
        
        // Add pulsing animation
        let pulseAction = SCNAction.sequence([
            SCNAction.scale(to: 1.1, duration: 0.5),
            SCNAction.scale(to: 1.05, duration: 0.5)
        ])
        let repeatAction = SCNAction.repeatForever(pulseAction)
        selectionNode.runAction(repeatAction, forKey: "selection_pulse")
    }
    
    private func removeSelectionIndicator(from node: SCNNode) {
        let selectionNodes = node.childNodes.filter { $0.name == "selection_indicator" }
        for selectionNode in selectionNodes {
            selectionNode.removeAllActions()
            selectionNode.removeFromParentNode()
        }
    }
    
    private func convertScreenToWorldTranslation(_ screenTranslation: CGPoint, for node: SCNNode) -> SCNVector3 {
        guard let scnView = scnView,
              let cameraNode = scnView.scene?.rootNode.childNodes.first(where: { $0.camera != nil }) else {
            return SCNVector3Zero
        }
        
        // Get camera transform
        let cameraTransform = cameraNode.transform
        
        // Calculate right and up vectors from camera
        let rightVector = SCNVector3(cameraTransform.m11, cameraTransform.m12, cameraTransform.m13)
        let upVector = SCNVector3(cameraTransform.m21, cameraTransform.m22, cameraTransform.m23)
        
        // Scale translation based on distance from camera
        let distance = distanceBetweenNodes(cameraNode, node)
        let scaleFactor = distance * 0.001 // Adjust scaling as needed
        
        // Convert screen translation to world coordinates
        let worldTranslation = SCNVector3(
            rightVector.x * Float(screenTranslation.x) * scaleFactor + upVector.x * Float(-screenTranslation.y) * scaleFactor,
            rightVector.y * Float(screenTranslation.x) * scaleFactor + upVector.y * Float(-screenTranslation.y) * scaleFactor,
            rightVector.z * Float(screenTranslation.x) * scaleFactor + upVector.z * Float(-screenTranslation.y) * scaleFactor
        )
        
        return worldTranslation
    }
    
    private func distanceBetweenNodes(_ node1: SCNNode, _ node2: SCNNode) -> Float {
        let position1 = node1.position
        let position2 = node2.position
        
        return sqrt(
            pow(position2.x - position1.x, 2) +
            pow(position2.y - position1.y, 2) +
            pow(position2.z - position1.z, 2)
        )
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
