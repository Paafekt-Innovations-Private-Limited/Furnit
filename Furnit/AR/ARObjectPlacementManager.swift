import SceneKit
import Metal
import MetalKit
import CoreVideo
import UIKit

protocol ARObjectPlacementManagerDelegate {
    func placementManager(_ manager: ARObjectPlacementManager, didPlaceObject node: SCNNode)
    func placementManager(_ manager: ARObjectPlacementManager, didUpdateObject node: SCNNode)
    func placementManager(_ manager: ARObjectPlacementManager, didRemoveObject node: SCNNode)
}

class ARObjectPlacementManager: ObservableObject {
    var delegate: ARObjectPlacementManagerDelegate?
    
    private weak var sceneView: SCNView?
    private var metalRenderingPipeline: MetalRenderingPipeline?
    
    // Currently placed AR objects
    private var placedObjects: [String: SCNNode] = [:]
    // Persistent objects that remain after AR stops
    private var persistentObjects: [String: SCNNode] = [:]
    
    // Configuration
    private let defaultObjectSize: Float = 1.0 // 1 meter
    private let placementHeight: Float = 0.1 // 10cm above ground
    
    @Published var isActive = false
    @Published var objectsArePersistent = false // Track if objects should persist after AR stops
    
    init() {
        do {
            metalRenderingPipeline = try MetalRenderingPipeline()
            print("✅ Metal rendering pipeline initialized successfully")
        } catch {
            print("❌ Failed to initialize Metal rendering pipeline: \(error)")
            print("🔄 Will use fallback rendering without Metal acceleration")
        }
    }
    
    func setSceneView(_ sceneView: SCNView) {
        self.sceneView = sceneView
    }
    
    func startPlacement() {
        isActive = true
        
        // Clear all existing objects when starting new AR session
        clearAllObjects()
        
        // Ensure debug cube is visible when AR starts
        if let sceneView = sceneView, let scene = sceneView.scene {
            ensureDebugCubeExists(in: scene)
        }
        
        print("🎯 AR object placement started")
    }
    
    func stopPlacement() {
        isActive = false
        
        // Move placed objects to persistent storage instead of removing them
        for (key, node) in placedObjects {
            persistentObjects[key] = node
            // Make objects slightly more transparent to indicate they're persistent
            node.opacity = 0.8
            // Enable interaction for persistent objects
            enableObjectInteraction(node)
        }
        
        placedObjects.removeAll()
        objectsArePersistent = true
        
        print("🛑 AR object placement stopped - \(persistentObjects.count) objects persisted")
    }
    
    func removeAllPlacedObjects() {
        // Remove only actively placed objects, not persistent ones
        for (_, node) in placedObjects {
            node.removeFromParentNode()
            delegate?.placementManager(self, didRemoveObject: node)
        }
        placedObjects.removeAll()
        print("🗑️ Removed all actively placed AR objects")
    }
    
    func removeAllPersistentObjects() {
        // Remove all persistent objects from scene
        for (_, node) in persistentObjects {
            node.removeFromParentNode()
            delegate?.placementManager(self, didRemoveObject: node)
        }
        persistentObjects.removeAll()
        objectsArePersistent = false
        print("🗑️ Removed all persistent AR objects")
    }
    
    func clearAllObjects() {
        // Remove both active and persistent objects
        removeAllPlacedObjects()
        removeAllPersistentObjects()
    }
    
    func placeSegmentedObject(_ segmentedObject: SegmentedObject) {
        print("🎯 placeSegmentedObject called - isActive: \(isActive), sceneView: \(sceneView != nil), scene: \(sceneView?.scene != nil), metalPipeline: \(metalRenderingPipeline != nil)")
        
        guard isActive else {
            print("⚠️ Cannot place object - placement not active")
            return
        }
        
        guard let sceneView = sceneView else {
            print("⚠️ Cannot place object - scene view not available")
            return
        }
        
        guard let scene = sceneView.scene else {
            print("⚠️ Cannot place object - scene not available")
            return
        }
        
        // Use furniture type as unique identifier instead of UUID
        let furnitureKey = "ar_object_\(segmentedObject.objectClass.displayName)"
        
        // Check if object of this furniture type already exists in active or persistent objects
        if let existingNode = placedObjects[furnitureKey] {
            print("🔄 Updating existing active \(segmentedObject.objectClass.displayName) instead of creating new one")
            updateExistingObject(existingNode, with: segmentedObject)
            return
        }
        
        if let existingNode = persistentObjects[furnitureKey] {
            print("🔄 Updating existing persistent \(segmentedObject.objectClass.displayName) and moving to active")
            // Move from persistent to active objects
            persistentObjects.removeValue(forKey: furnitureKey)
            placedObjects[furnitureKey] = existingNode
            // Restore full opacity since it's now active
            existingNode.opacity = 1.0
            updateExistingObject(existingNode, with: segmentedObject)
            return
        }
        
        print("🆕 Creating new \(segmentedObject.objectClass.displayName) object")
        
        // Create material - try furniture texture first, then fallback
        let material: SCNMaterial
        if let furnitureTexture = segmentedObject.createFurnitureTexture() {
            print("✅ Using extracted furniture texture")
            material = createMaterialFromFurnitureImage(furnitureTexture)
        } else if let metalPipeline = metalRenderingPipeline,
                  let processedTexture = metalPipeline.processSegmentedObject(segmentedObject),
                  let materialFromTexture = createMaterialFromTexture(processedTexture) {
            print("✅ Using Metal-processed texture")
            material = materialFromTexture
        } else {
            print("🔄 Using fallback material creation")
            material = createFallbackMaterial(for: segmentedObject)
        }
        
        // Create 3D plane geometry for the segmented object
        let plane = createPlaneGeometry(for: segmentedObject, material: material)
        
        // Calculate placement position in 3D space near room origin (only for new objects)
        let placementPosition = calculateRoomPlacementPosition(for: segmentedObject, in: scene)
        
        // Create scene node
        let objectNode = SCNNode(geometry: plane)
        objectNode.position = placementPosition
        objectNode.name = furnitureKey
        
        // Configure node properties
        configureObjectNode(objectNode, for: segmentedObject.objectClass)
        
        // Add to scene
        scene.rootNode.addChildNode(objectNode)
        
        // Store reference using furniture type key
        placedObjects[furnitureKey] = objectNode
        
        // Debug scene hierarchy and node properties
        print("🏗️ Scene root node children count: \(scene.rootNode.childNodes.count)")
        print("🎯 Object node properties:")
        print("   - Name: \(objectNode.name ?? "nil")")
        print("   - Position: \(objectNode.position)")
        print("   - Scale: \(objectNode.scale)")
        print("   - Opacity: \(objectNode.opacity)")
        print("   - Hidden: \(objectNode.isHidden)")
        print("   - Geometry: \(String(describing: objectNode.geometry))")
        print("   - Materials count: \(objectNode.geometry?.materials.count ?? 0)")
        
        if let plane = objectNode.geometry as? SCNPlane {
            print("   - Plane width: \(plane.width)")
            print("   - Plane height: \(plane.height)")
            print("   - Material diffuse: \(String(describing: plane.materials.first?.diffuse.contents))")
        }
        
        // Check camera position for reference
        if let cameraNode = scene.rootNode.childNodes.first(where: { $0.camera != nil }) {
            let distance = sqrt(
                pow(objectNode.position.x - cameraNode.position.x, 2) +
                pow(objectNode.position.y - cameraNode.position.y, 2) +
                pow(objectNode.position.z - cameraNode.position.z, 2)
            )
            print("📏 Distance from camera: \(distance)")
            
            // Check if object is in front of camera
            let toObject = SCNVector3(
                objectNode.position.x - cameraNode.position.x,
                objectNode.position.y - cameraNode.position.y,
                objectNode.position.z - cameraNode.position.z
            )
            let cameraForward = SCNVector3(-cameraNode.transform.m31, -cameraNode.transform.m32, -cameraNode.transform.m33)
            let dotProduct = toObject.x * cameraForward.x + toObject.y * cameraForward.y + toObject.z * cameraForward.z
            print("📐 Object in front of camera: \(dotProduct > 0)")
        }
        
        // Make object more visible by increasing opacity and scale
        objectNode.opacity = 1.0 // Full opacity
        objectNode.scale = SCNVector3(2.0, 2.0, 2.0) // Double size for testing
        
        print("🔧 Enhanced object visibility: opacity=1.0, scale=2x")
        
        // Notify delegate
        delegate?.placementManager(self, didPlaceObject: objectNode)
        
        print("✅ Placed AR object: \(segmentedObject.objectClass.displayName) at position \(placementPosition)")
        
        // Add a test cube at a known position for debugging
        addDebugTestCube(to: scene)
    }
    
    private func addDebugTestCube(to scene: SCNScene) {
        ensureDebugCubeExists(in: scene)
    }
    
    private func ensureDebugCubeExists(in scene: SCNScene) {
        // Check if debug cube already exists
        let existingCube = scene.rootNode.childNodes.first { $0.name == "debug_test_cube" }
        if existingCube != nil {
            print("🟥 Debug test cube already exists")
            return
        }
        
        // Create a bright red cube for debugging visibility
        let cubeGeometry = SCNBox(width: 0.5, height: 0.5, length: 0.5, chamferRadius: 0.02)
        let cubeMaterial = SCNMaterial()
        cubeMaterial.diffuse.contents = UIColor.red
        cubeMaterial.lightingModel = .constant // Ensure it's always visible
        cubeGeometry.materials = [cubeMaterial]
        
        let cubeNode = SCNNode(geometry: cubeGeometry)
        cubeNode.position = SCNVector3(0, 1, -2) // Fixed position: center, 1m up, 2m forward
        cubeNode.name = "debug_test_cube"
        
        scene.rootNode.addChildNode(cubeNode)
        print("🟥 Added debug test cube at position (0, 1, -2)")
    }
    
    private func updateExistingObject(_ existingNode: SCNNode, with segmentedObject: SegmentedObject) {
        // Update the material/texture of the existing object without changing position
        let material: SCNMaterial
        if let furnitureTexture = segmentedObject.createFurnitureTexture() {
            print("✅ Updating existing object with new furniture texture")
            material = createMaterialFromFurnitureImage(furnitureTexture)
        } else if let metalPipeline = metalRenderingPipeline,
                  let processedTexture = metalPipeline.processSegmentedObject(segmentedObject),
                  let materialFromTexture = createMaterialFromTexture(processedTexture) {
            print("✅ Updating existing object with Metal-processed texture")
            material = materialFromTexture
        } else {
            print("🔄 Updating existing object with fallback material")
            material = createFallbackMaterial(for: segmentedObject)
        }
        
        // Update the geometry's material
        if let plane = existingNode.geometry as? SCNPlane {
            plane.materials = [material]
            print("🔄 Updated \(segmentedObject.objectClass.displayName) material at position: \(existingNode.position)")
        }
        
        // Optionally update plane dimensions if the segmented object size changed significantly
        if let plane = existingNode.geometry as? SCNPlane {
            let aspectRatio = segmentedObject.boundingBox.width / segmentedObject.boundingBox.height
            let planeHeight = CGFloat(defaultObjectSize) * 0.8
            let planeWidth = planeHeight * aspectRatio
            
            // Only update if the size change is significant (more than 20% difference)
            let sizeChangeThreshold: CGFloat = 0.2
            let widthDiff = abs(plane.width - planeWidth) / plane.width
            let heightDiff = abs(plane.height - planeHeight) / plane.height
            
            if widthDiff > sizeChangeThreshold || heightDiff > sizeChangeThreshold {
                plane.width = planeWidth
                plane.height = planeHeight
                print("📏 Updated plane dimensions: \(planeWidth) x \(planeHeight)")
            }
        }
        
        // Notify delegate about the update
        delegate?.placementManager(self, didUpdateObject: existingNode)
    }
    
    private func createMaterialFromTexture(_ texture: MTLTexture) -> SCNMaterial? {
        // Create UIImage from Metal texture
        guard let image = createUIImage(from: texture) else {
            return nil
        }
        
        let material = SCNMaterial()
        material.diffuse.contents = image
        material.isDoubleSided = true
        
        // Configure material for transparency
        material.blendMode = .alpha
        material.transparency = 0.9
        material.transparencyMode = .aOne
        
        // Disable lighting for billboard effect
        material.lightingModel = .constant
        
        return material
    }
    
    private func createMaterialFromFurnitureImage(_ furnitureImage: UIImage) -> SCNMaterial {
        let material = SCNMaterial()
        material.diffuse.contents = furnitureImage
        material.isDoubleSided = true
        
        // Configure material for transparency
        material.blendMode = .alpha
        material.transparency = 1.0
        material.transparencyMode = .aOne
        
        // Use constant lighting to preserve the furniture appearance
        material.lightingModel = .constant
        
        print("🖼️ Created material from furniture image: \(furnitureImage.size.width)x\(furnitureImage.size.height)")
        return material
    }
    
    private func createFallbackMaterial(for segmentedObject: SegmentedObject) -> SCNMaterial {
        let material = SCNMaterial()
        
        // Create a more visible colored plane as fallback based on furniture type
        let color: UIColor
        switch segmentedObject.objectClass {
        case .chair:
            color = UIColor.systemBlue.withAlphaComponent(0.8)
        case .sofa:
            color = UIColor.systemGreen.withAlphaComponent(0.8)
        case .diningTable:
            color = UIColor.systemOrange.withAlphaComponent(0.8)
        default:
            color = UIColor.systemPurple.withAlphaComponent(0.8)
        }
        
        material.diffuse.contents = color
        material.isDoubleSided = true
        
        // Configure material for better visibility
        material.blendMode = .alpha
        material.transparency = 0.8
        material.transparencyMode = .aOne
        
        // Add some lighting for depth perception
        material.lightingModel = .lambert
        material.specular.contents = UIColor.white.withAlphaComponent(0.3)
        
        print("📝 Created fallback \(segmentedObject.objectClass.displayName) material with color: \(color)")
        return material
    }
    
    private func createUIImage(from texture: MTLTexture) -> UIImage? {
        // This is a simplified conversion - in production you'd use more efficient methods
        let width = texture.width
        let height = texture.height
        let bytesPerPixel = 4
        let bytesPerRow = width * bytesPerPixel
        let totalBytes = height * bytesPerRow
        
        var pixelData = [UInt8](repeating: 0, count: totalBytes)
        
        texture.getBytes(&pixelData, bytesPerRow: bytesPerRow, from: MTLRegion(origin: MTLOrigin(x: 0, y: 0, z: 0), size: MTLSize(width: width, height: height, depth: 1)), mipmapLevel: 0)
        
        guard let dataProvider = CGDataProvider(data: Data(pixelData) as CFData) else {
            return nil
        }
        
        guard let cgImage = CGImage(
            width: width,
            height: height,
            bitsPerComponent: 8,
            bitsPerPixel: 32,
            bytesPerRow: bytesPerRow,
            space: CGColorSpace(name: CGColorSpace.sRGB)!,
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
            provider: dataProvider,
            decode: nil,
            shouldInterpolate: false,
            intent: .defaultIntent
        ) else {
            return nil
        }
        
        return UIImage(cgImage: cgImage)
    }
    
    private func createPlaneGeometry(for segmentedObject: SegmentedObject, material: SCNMaterial) -> SCNPlane {
        // Calculate plane dimensions based on segmented object bounding box
        let aspectRatio = segmentedObject.boundingBox.width / segmentedObject.boundingBox.height
        let planeHeight = CGFloat(defaultObjectSize) * 0.8 // Make it smaller for better visibility
        let planeWidth = planeHeight * aspectRatio
        
        print("📐 Plane dimensions: \(planeWidth) x \(planeHeight), aspect ratio: \(aspectRatio)")
        
        let plane = SCNPlane(width: planeWidth, height: planeHeight)
        plane.materials = [material]
        
        // Make sure the plane has proper corner radius for better visibility
        plane.cornerRadius = 0.05
        
        return plane
    }
    
    private func calculateRoomPlacementPosition(for segmentedObject: SegmentedObject, in scene: SCNScene) -> SCNVector3 {
        // Place objects visible to the camera by positioning them in the camera's view frustum
        
        guard let cameraNode = scene.rootNode.childNodes.first(where: { $0.camera != nil }) else {
            // Fallback to origin-centered placement
            return SCNVector3(0, 1, -2)
        }
        
        let cameraPosition = cameraNode.position
        
        // Calculate forward direction from camera
        let cameraTransform = cameraNode.transform
        let forwardVector = SCNVector3(-cameraTransform.m31, -cameraTransform.m32, -cameraTransform.m33)
        let normalizedForward = normalizeVector(forwardVector)
        
        // Place object in front of camera at a reasonable distance and height
        let placementDistance: Float = 3.0 // 3 meters in front
        let heightOffset: Float = -2.0 // 2 meters below camera (but still visible)
        
        let visiblePlacement = SCNVector3(
            cameraPosition.x + normalizedForward.x * placementDistance,
            max(0.5, cameraPosition.y + heightOffset), // At least 0.5m above floor, but below camera
            cameraPosition.z + normalizedForward.z * placementDistance
        )
        
        print("🏠 Placing furniture in camera view: \(visiblePlacement)")
        print("📍 Camera position: \(cameraPosition)")
        print("📍 Camera forward: \(normalizedForward)")
        print("📍 Visible placement: \(visiblePlacement)")
        
        let distance = sqrt(
            pow(visiblePlacement.x - cameraPosition.x, 2) +
            pow(visiblePlacement.y - cameraPosition.y, 2) +
            pow(visiblePlacement.z - cameraPosition.z, 2)
        )
        print("📏 Distance from camera: \(distance)")
        
        return visiblePlacement
    }
    
    private func normalizeVector(_ vector: SCNVector3) -> SCNVector3 {
        let length = sqrt(vector.x * vector.x + vector.y * vector.y + vector.z * vector.z)
        if length == 0 { return vector }
        
        return SCNVector3(
            vector.x / length,
            vector.y / length,
            vector.z / length
        )
    }
    
    private func configureObjectNode(_ node: SCNNode, for objectClass: FurnitureClass) {
        // Configure node based on object type
        switch objectClass {
        case .chair:
            node.scale = SCNVector3(0.8, 0.8, 0.8)
        case .sofa:
            node.scale = SCNVector3(1.2, 1.0, 1.2)
        case .diningTable:
            node.scale = SCNVector3(1.0, 0.6, 1.0) // Tables are typically wider but shorter
        default:
            node.scale = SCNVector3(1.0, 1.0, 1.0)
        }
        
        // Make plane face the camera (billboard behavior)
        let constraint = SCNBillboardConstraint()
        constraint.freeAxes = [.Y] // Only rotate around Y axis
        node.constraints = [constraint]
        
        // Add subtle animation
        let floatAction = SCNAction.sequence([
            SCNAction.moveBy(x: 0, y: 0.05, z: 0, duration: 2.0),
            SCNAction.moveBy(x: 0, y: -0.05, z: 0, duration: 2.0)
        ])
        let repeatAction = SCNAction.repeatForever(floatAction)
        node.runAction(repeatAction, forKey: "floating")
    }
    
    private func enableObjectInteraction(_ node: SCNNode) {
        // Enable tap gesture recognition for the node
        node.name = (node.name ?? "") + "_interactive"
        
        // Add subtle glow effect to indicate interactivity
        let glowMaterial = SCNMaterial()
        glowMaterial.emission.contents = UIColor.white.withAlphaComponent(0.1)
        
        if let geometry = node.geometry {
            var materials = geometry.materials
            for i in 0..<materials.count {
                materials[i].emission.contents = UIColor.white.withAlphaComponent(0.05)
            }
            geometry.materials = materials
        }
        
        print("✨ Enabled interaction for persistent object: \(node.name ?? "Unknown")")
    }
    
    func removeObject(withName name: String) {
        guard let node = placedObjects[name] else { return }
        
        node.removeFromParentNode()
        placedObjects.removeValue(forKey: name)
        
        delegate?.placementManager(self, didRemoveObject: node)
        print("🗑️ Removed AR object: \(name)")
    }
    
    func getPlacedObjectsCount() -> Int {
        return placedObjects.count
    }
    
    func getPersistentObjectsCount() -> Int {
        return persistentObjects.count
    }
    
    func getAllPlacedObjects() -> [SCNNode] {
        return Array(placedObjects.values)
    }
    
    func getAllPersistentObjects() -> [SCNNode] {
        return Array(persistentObjects.values)
    }
    
    func getAllObjects() -> [SCNNode] {
        return Array(placedObjects.values) + Array(persistentObjects.values)
    }
}