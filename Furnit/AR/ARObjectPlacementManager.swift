import SceneKit
import UIKit

@MainActor
class ARObjectPlacementManager: ObservableObject {
    // Published properties for UI updates
    @Published var placedObjects: [ARPlacedObject] = []
    @Published var isReadyToPlace = false
    
    // Scene references
    weak var sceneView: SCNView?
    weak var scene: SCNScene?
    
    // Object placement properties
    private var segmentedImage: UIImage?
    private var generatedModel: SCNNode? // 3D model from backend API
    private let placementHeight: Float = 0.05 // Small lift above floor for visual clarity
    
    // Placement distance constraints for better user experience
    private let maxPlacementDistance: Float = 50.0 // Maximum distance from camera for placement
    private let preferredPlacementDistance: Float = 3.0 // Reduced from 8.0 to 3.0 for closer placement
    
    init() {}
    
    // Set scene references for object placement
    func setSceneReferences(sceneView: SCNView, scene: SCNScene) {
        self.sceneView = sceneView
        self.scene = scene
    }
    
    // Prepare for object placement with segmented image
    func prepareForPlacement(with segmentedImage: UIImage) {
        self.segmentedImage = segmentedImage
        self.generatedModel = nil
        isReadyToPlace = true
    }
    
    // Prepare for object placement with 3D model from backend API
    func prepareForPlacement(with3DModel model: SCNNode) {
        self.generatedModel = model.clone()
        self.segmentedImage = nil
        isReadyToPlace = true
        print("✅ Ready to place 3D model with \(model.childNodes.count) child nodes")
    }
    
    // Handle tap gesture to place object in 3D scene
    func handleTapToPlace(at screenPoint: CGPoint) -> Bool {
        guard let sceneView = sceneView,
              let scene = scene,
              isReadyToPlace,
              (segmentedImage != nil || generatedModel != nil) else {
            print("⚠️ Not ready to place object")
            return false
        }
        
        // Determine optimal placement position with distance constraints
        let placementPosition = determinePlacementPosition(for: screenPoint, in: sceneView)
        
        // Check if we're placing a 3D model or segmented image
        if let model = generatedModel {
            // Place 3D model from backend API
            return place3DModel(model, at: placementPosition, in: scene)
        } else if let segmentedImage = segmentedImage {
            // Create and place segmented image as 2D plane
            // Use default upward normal for segmented images since we have a determined position
            let defaultNormal = SCNVector3(0, 1, 0)
            if let arObject = createARObject(from: segmentedImage, at: placementPosition, normal: defaultNormal) {
                // Add to scene
                scene.rootNode.addChildNode(arObject.node)
                
                // Track placed object
                let placedObject = ARPlacedObject(
                    id: UUID(),
                    node: arObject.node,
                    originalImage: segmentedImage,
                    position: placementPosition,
                    is3DModel: false
                )
                
                placedObjects.append(placedObject)
                
                print("✅ AR object placed successfully at position: \(placementPosition)")
                return true
            }
        }
        
        return false
    }
    
    // Determine optimal placement position with distance constraints and fallback strategy
    private func determinePlacementPosition(for screenPoint: CGPoint, in sceneView: SCNView) -> SCNVector3 {
        // Get camera node for distance calculations
        guard let cameraNode = sceneView.pointOfView else {
            print("⚠️ Camera not available, using fallback placement")
            return SCNVector3(0, 0, -preferredPlacementDistance)
        }
        
        let cameraPosition = cameraNode.worldPosition
        print("📍 Camera position: \(cameraPosition)")
        
        // Perform hit test to find potential placement surfaces
        let hitTestOptions: [SCNHitTestOption: Any] = [
            .boundingBoxOnly: false,
            .firstFoundOnly: false // Get all hits to find the closest suitable one
        ]
        
        let hitResults = sceneView.hitTest(screenPoint, options: hitTestOptions)
        
        // Look for a hit within acceptable distance
        for hitResult in hitResults {
            let hitPosition = hitResult.worldCoordinates
            let distance = calculateDistance(from: cameraPosition, to: hitPosition)
            
            print("🎯 Hit test result: position(\(hitPosition)), distance(\(distance))")
            
            // Check if this hit is within acceptable distance
            if distance <= maxPlacementDistance {
                let hitNormal = hitResult.worldNormal
                
                // Adjust position slightly above the surface
                let adjustedPosition = SCNVector3(
                    hitPosition.x + hitNormal.x * placementHeight,
                    hitPosition.y + hitNormal.y * placementHeight,
                    hitPosition.z + hitNormal.z * placementHeight
                )
                
                print("✅ Found suitable placement surface at distance \(distance)")
                return adjustedPosition
            }
        }
        
        // No suitable hit found within distance limit, use fallback placement strategy
        print("⚠️ No suitable surface within \(maxPlacementDistance) units, using fallback placement")
        return calculateFallbackPlacement(for: screenPoint, camera: cameraNode)
    }
    
    // Calculate fallback placement position in front of camera with proper floor detection
    private func calculateFallbackPlacement(for screenPoint: CGPoint, camera: SCNNode) -> SCNVector3 {
        guard let sceneView = sceneView else {
            return SCNVector3(0, 0, -preferredPlacementDistance)
        }
        
        // Get camera's transform
        let cameraTransform = camera.worldTransform
        let cameraPosition = SCNVector3(
            cameraTransform.m41,
            cameraTransform.m42,
            cameraTransform.m43
        )
        
        // Convert screen point to normalized device coordinates
        let viewportSize = sceneView.bounds.size
        let normalizedX = (screenPoint.x / viewportSize.width) * 2.0 - 1.0
        let normalizedY = ((viewportSize.height - screenPoint.y) / viewportSize.height) * 2.0 - 1.0
        
        // Calculate direction from camera through the screen point
        let fieldOfView = camera.camera?.fieldOfView ?? 75.0
        let aspect = Double(viewportSize.width / viewportSize.height)
        
        // Convert to radians and calculate direction
        let fovRadians = fieldOfView * Double.pi / 180.0
        let directionX = Float(normalizedX * tan(fovRadians / 2.0) * aspect)
        let directionY = Float(normalizedY * tan(fovRadians / 2.0))
        let directionZ = Float(-1.0) // Forward in camera space
        
        // Transform direction to world space
        let localDirection = SCNVector3(directionX, directionY, directionZ).normalized
        let worldDirection = camera.convertVector(localDirection, to: nil)
        
        // Determine if user is tapping in lower half of screen (floor) or upper half (wall/air)
        // iOS coordinate system: origin at top-left, Y increases downward
        let isTappingFloor = screenPoint.y > viewportSize.height * 0.6 // Lower 40% of screen
        
        var placementPosition: SCNVector3
        
        if isTappingFloor {
            // Place on floor - find intersection with floor plane
            placementPosition = calculateFloorPlacement(cameraPosition: cameraPosition, worldDirection: worldDirection)
        } else {
            // Place at eye level in front of camera
            placementPosition = SCNVector3(
                cameraPosition.x + worldDirection.x * preferredPlacementDistance,
                cameraPosition.y + worldDirection.y * preferredPlacementDistance,
                cameraPosition.z + worldDirection.z * preferredPlacementDistance
            )
        }
        
        print("🎯 Fallback placement: camera(\(cameraPosition)), direction(\(worldDirection))")
        print("   Tapping floor: \(isTappingFloor), final position: \(placementPosition)")
        
        return placementPosition
    }
    
    // Calculate placement on floor plane
    private func calculateFloorPlacement(cameraPosition: SCNVector3, worldDirection: SCNVector3) -> SCNVector3 {
        // Detect floor height from scene or use reasonable default
        let floorHeight = detectFloorHeight() ?? 0.0
        
        // Calculate intersection of camera ray with floor plane (Y = floorHeight)
        // Ray equation: P = cameraPosition + t * worldDirection
        // Plane equation: Y = floorHeight
        // Solve for t: floorHeight = cameraPosition.y + t * worldDirection.y
        
        if abs(worldDirection.y) > 0.001 { // Avoid division by zero
            let t = (floorHeight - cameraPosition.y) / worldDirection.y
            
            // Only use positive t (forward direction) and reasonable distance
            if t > 0 && t <= 6.0 { // Limit to 6 units for floor placement
                return SCNVector3(
                    cameraPosition.x + worldDirection.x * t,
                    floorHeight + placementHeight, // Slightly above floor
                    cameraPosition.z + worldDirection.z * t
                )
            }
        }
        
        // Fallback: place on floor at preferred distance horizontally
        let horizontalDirection = SCNVector3(worldDirection.x, 0, worldDirection.z).normalized
        let floorPlacementDistance = min(preferredPlacementDistance, 4.0) // Cap at 4 units for floor placement
        return SCNVector3(
            cameraPosition.x + horizontalDirection.x * floorPlacementDistance,
            floorHeight + placementHeight,
            cameraPosition.z + horizontalDirection.z * floorPlacementDistance
        )
    }
    
    // Detect floor height from scene geometry or use default
    private func detectFloorHeight() -> Float? {
        guard let scene = scene else { return nil }
        
        // Look for geometry with low Y coordinates to find floor
        var minY: Float = Float.greatestFiniteMagnitude
        var hasGeometry = false
        
        scene.rootNode.enumerateChildNodes { node, _ in
            guard node.geometry != nil else { return }
            
            let (minBound, _) = node.boundingBox
            let worldMin = node.convertPosition(minBound, to: nil)
            
            if worldMin.y < minY {
                minY = worldMin.y
                hasGeometry = true
            }
        }
        
        // Return detected floor height or reasonable default
        return hasGeometry ? minY : 0.0
    }
    
    // Calculate distance between two 3D points
    private func calculateDistance(from point1: SCNVector3, to point2: SCNVector3) -> Float {
        let dx = point2.x - point1.x
        let dy = point2.y - point1.y
        let dz = point2.z - point1.z
        return sqrt(dx * dx + dy * dy + dz * dz)
    }
    
    // Place 3D model from backend API in the scene with persistence enhancements
    private func place3DModel(_ model: SCNNode, at position: SCNVector3, in scene: SCNScene) -> Bool {
        // Clone the model to avoid modifying the original
        let placedModelNode = model.clone()
        
        // Position the model at the specified location
        placedModelNode.position = position
        
        // Enhance object persistence and visibility
        enhanceObjectPersistence(placedModelNode)
        
        // Add debug visualization to help identify placement
        addDebugVisualization(at: position, in: scene, for: placedModelNode)
        
        // Add to scene with strong reference
        scene.rootNode.addChildNode(placedModelNode)
        
        // Track placed object
        let placedObject = ARPlacedObject(
            id: UUID(),
            node: placedModelNode,
            originalImage: nil,
            position: position,
            is3DModel: true
        )
        
        placedObjects.append(placedObject)
        
        // Reset placement state
        isReadyToPlace = false
        generatedModel = nil
        
        print("✅ 3D model placed successfully at position: \(position)")
        print("   Model has \(placedModelNode.childNodes.count) child nodes")
        
        // Log detailed placement information for debugging
        logPlacementDebugInfo(placedModelNode, at: position)
        
        return true
    }
    
    // Enhance object properties for better persistence and visibility
    private func enhanceObjectPersistence(_ node: SCNNode) {
        // Give node a unique name for tracking
        node.name = "AR_Placed_Object_\(UUID().uuidString)"
        
        // Set rendering order to ensure visibility
        node.renderingOrder = 100
        
        // Ensure all child nodes are properly configured
        node.enumerateChildNodes { childNode, _ in
            // Prevent culling by camera movement
            childNode.categoryBitMask = 1
            
            // Ensure materials render properly at distance
            if let geometry = childNode.geometry {
                for material in geometry.materials {
                    // Improve visibility at distance
                    material.isDoubleSided = true
                    
                    // Enhance lighting for better visibility
                    if material.lightingModel == .constant {
                        material.lightingModel = .blinn
                    }
                    
                    // Ensure proper depth testing
                    material.writesToDepthBuffer = true
                    material.readsFromDepthBuffer = true
                }
            }
        }
        
        print("🔧 Enhanced object persistence for node: \(node.name ?? "unnamed")")
    }
    
    // Monitor placed objects to ensure they remain visible when camera moves
    func validateObjectVisibility() {
        guard let sceneView = sceneView,
              let cameraNode = sceneView.pointOfView else { return }
        
        let cameraPosition = cameraNode.worldPosition
        var invisibleCount = 0
        
        for placedObject in placedObjects {
            let distance = calculateDistance(from: cameraPosition, to: placedObject.position)
            
            // Check if object is beyond reasonable viewing distance
            if distance > 200.0 {
                print("⚠️ Object at \(placedObject.position) is \(distance) units from camera - may be invisible")
                invisibleCount += 1
            }
            
            // Ensure object is still in scene
            if placedObject.node.parent == nil {
                print("⚠️ Object node has been removed from scene - re-adding")
                scene?.rootNode.addChildNode(placedObject.node)
            }
        }
        
        if invisibleCount > 0 {
            print("📊 Visibility check: \(invisibleCount)/\(placedObjects.count) objects may be invisible")
        }
    }
    
    // Create 3D AR object from segmented image
    private func createARObject(from image: UIImage, at position: SCNVector3, normal: SCNVector3) -> (node: SCNNode, size: CGSize)? {
        
        // Estimate object dimensions based on image analysis
        let objectSize = estimateObjectDimensions(from: image)
        
        // Create geometry based on object type (for now, use plane with image texture)
        let plane = SCNPlane(width: objectSize.width, height: objectSize.height)
        
        // Create material with the segmented image
        let material = SCNMaterial()
        material.diffuse.contents = image
        material.transparency = 1.0
        material.isDoubleSided = true
        
        // Enable transparency for segmented objects
        material.blendMode = .alpha
        material.writesToDepthBuffer = false
        
        plane.materials = [material]
        
        // Create node
        let objectNode = SCNNode(geometry: plane)
        objectNode.position = position
        
        // Orient object to face the camera while staying perpendicular to surface
        if let camera = sceneView?.pointOfView {
            let cameraPosition = camera.position
            let lookDirection = SCNVector3(
                cameraPosition.x - position.x,
                0, // Keep level with horizon
                cameraPosition.z - position.z
            )
            
            // Normalize look direction
            let length = sqrt(lookDirection.x * lookDirection.x + lookDirection.z * lookDirection.z)
            if length > 0 {
                let normalizedDirection = SCNVector3(
                    lookDirection.x / length,
                    0,
                    lookDirection.z / length
                )
                
                // Calculate rotation to face camera
                let angle = atan2(normalizedDirection.x, normalizedDirection.z)
                objectNode.rotation = SCNVector4(0, 1, 0, angle)
            }
        }
        
        // Add subtle shadow plane beneath object
        addShadowPlane(to: objectNode, size: objectSize)
        
        // Add slight hover animation for visual appeal
        addHoverAnimation(to: objectNode)
        
        return (node: objectNode, size: objectSize)
    }
    
    // Estimate object dimensions from segmented image
    private func estimateObjectDimensions(from image: UIImage) -> CGSize {
        // Analyze the segmented image to estimate real-world dimensions
        let imageSize = image.size
        let aspectRatio = imageSize.width / imageSize.height
        
        // Default furniture dimensions (can be improved with ML-based size estimation)
        let baseWidth: CGFloat = 1.0 // 1 meter base width
        
        // Adjust dimensions based on aspect ratio and furniture type
        let width = baseWidth
        let height = baseWidth / aspectRatio
        
        // Ensure reasonable minimum and maximum sizes
        let minSize: CGFloat = 0.3
        let maxSize: CGFloat = 2.0
        
        return CGSize(
            width: max(minSize, min(maxSize, width)),
            height: max(minSize, min(maxSize, height))
        )
    }
    
    // Add shadow plane beneath AR object for better ground integration
    private func addShadowPlane(to parentNode: SCNNode, size: CGSize) {
        let shadowPlane = SCNPlane(width: size.width * 1.2, height: size.height * 1.2)
        
        let shadowMaterial = SCNMaterial()
        shadowMaterial.diffuse.contents = UIColor.black.withAlphaComponent(0.3)
        shadowMaterial.writesToDepthBuffer = false
        shadowMaterial.blendMode = .multiply
        
        shadowPlane.materials = [shadowMaterial]
        
        let shadowNode = SCNNode(geometry: shadowPlane)
        shadowNode.position = SCNVector3(0, -Float(size.height) * 0.5 - 0.01, 0)
        shadowNode.rotation = SCNVector4(1, 0, 0, -Float.pi / 2) // Lay flat on ground
        
        parentNode.addChildNode(shadowNode)
    }
    
    // Add subtle hover animation to AR objects
    private func addHoverAnimation(to node: SCNNode) {
        let hoverAnimation = CABasicAnimation(keyPath: "position.y")
        hoverAnimation.fromValue = node.position.y
        hoverAnimation.toValue = node.position.y + 0.02 // 2cm hover
        hoverAnimation.duration = 2.0
        hoverAnimation.autoreverses = true
        hoverAnimation.repeatCount = .infinity
        hoverAnimation.timingFunction = CAMediaTimingFunction(name: .easeInEaseOut)
        
        node.addAnimation(hoverAnimation, forKey: "hover")
    }
    
    // Remove specific AR object
    func removeObject(_ objectId: UUID) {
        if let index = placedObjects.firstIndex(where: { $0.id == objectId }) {
            let object = placedObjects[index]
            object.node.removeFromParentNode()
            placedObjects.remove(at: index)
            
            print("🗑️ Removed AR object with ID: \(objectId)")
        }
    }
    
    // Clear all placed AR objects and debug markers
    func clearAllObjects() {
        for placedObject in placedObjects {
            placedObject.node.removeFromParentNode()
        }
        placedObjects.removeAll()
        
        // Also remove debug markers
        clearDebugMarkers()
        
        isReadyToPlace = false
        segmentedImage = nil
        
        print("🧹 Cleared all AR objects and debug markers")
    }
    
    // Clear debug visualization markers
    private func clearDebugMarkers() {
        guard let scene = scene else { return }
        
        // Remove debug markers by name
        let debugMarkerNames = ["DEBUG_PLACEMENT_MARKER", "DEBUG_CAMERA_MARKER", "DEBUG_BOUNDING_BOX"]
        
        scene.rootNode.enumerateChildNodes { node, _ in
            if let nodeName = node.name,
               debugMarkerNames.contains(nodeName) {
                node.removeFromParentNode()
            }
        }
        
        print("🧹 Cleared debug markers")
    }
    
    // Reset for new AR session
    func resetForNewSession() {
        clearAllObjects()
        print("🔄 Reset AR placement manager for new session")
    }
    
    // MARK: - Debug Visualization Methods
    
    // Add debug visualization to help identify placement location and model structure
    private func addDebugVisualization(at position: SCNVector3, in scene: SCNScene, for modelNode: SCNNode) {
        // Calculate distance from camera for color coding
        var distanceFromCamera: Float = 0
        var cameraPosition = SCNVector3Zero
        
        if let sceneView = sceneView,
           let cameraNode = sceneView.pointOfView {
            cameraPosition = cameraNode.worldPosition
            distanceFromCamera = calculateDistance(from: cameraPosition, to: position)
        }
        
        // Create a bright colored sphere at the placement position with distance-based color
        let debugSphere = SCNSphere(radius: 0.1) // 10cm radius for better visibility
        let debugMaterial = SCNMaterial()
        
        // Color-code by distance: Green (close), Yellow (medium), Red (far)
        if distanceFromCamera <= 15.0 {
            debugMaterial.diffuse.contents = UIColor.green
            debugMaterial.emission.contents = UIColor.green.withAlphaComponent(0.5)
        } else if distanceFromCamera <= 30.0 {
            debugMaterial.diffuse.contents = UIColor.yellow
            debugMaterial.emission.contents = UIColor.yellow.withAlphaComponent(0.5)
        } else {
            debugMaterial.diffuse.contents = UIColor.red
            debugMaterial.emission.contents = UIColor.red.withAlphaComponent(0.5)
        }
        
        debugSphere.materials = [debugMaterial]
        
        let debugSphereNode = SCNNode(geometry: debugSphere)
        debugSphereNode.position = position
        debugSphereNode.name = "DEBUG_PLACEMENT_MARKER"
        
        // Add the debug sphere to the scene
        scene.rootNode.addChildNode(debugSphereNode)
        
        // Add a smaller camera reference marker for comparison
        addCameraReferenceMarker(at: cameraPosition, in: scene)
        
        // Create bounding box visualization for the model
        addBoundingBoxVisualization(for: modelNode)
        
        print("🔍 Added debug visualization at position: \(position)")
        print("   Distance from camera: \(distanceFromCamera) units")
        print("   Camera position: \(cameraPosition)")
    }
    
    // Add a blue sphere at camera position for reference
    private func addCameraReferenceMarker(at position: SCNVector3, in scene: SCNScene) {
        let cameraSphere = SCNSphere(radius: 0.05) // 5cm radius
        let cameraMaterial = SCNMaterial()
        cameraMaterial.diffuse.contents = UIColor.blue
        cameraMaterial.emission.contents = UIColor.blue.withAlphaComponent(0.3)
        cameraSphere.materials = [cameraMaterial]
        
        let cameraSphereNode = SCNNode(geometry: cameraSphere)
        cameraSphereNode.position = position
        cameraSphereNode.name = "DEBUG_CAMERA_MARKER"
        
        // Add the camera reference sphere to the scene
        scene.rootNode.addChildNode(cameraSphereNode)
        
        print("📍 Added camera reference marker at: \(position)")
    }
    
    // Add bounding box visualization to the model
    private func addBoundingBoxVisualization(for node: SCNNode) {
        let (minBound, maxBound) = node.boundingBox
        
        // Calculate bounding box dimensions
        let width = maxBound.x - minBound.x
        let height = maxBound.y - minBound.y
        let depth = maxBound.z - minBound.z
        
        // Create wireframe box geometry
        let box = SCNBox(width: CGFloat(width), height: CGFloat(height), length: CGFloat(depth), chamferRadius: 0)
        let wireframeMaterial = SCNMaterial()
        wireframeMaterial.diffuse.contents = UIColor.cyan.withAlphaComponent(0.3)
        wireframeMaterial.fillMode = .lines
        wireframeMaterial.isDoubleSided = true
        box.materials = [wireframeMaterial]
        
        let boundingBoxNode = SCNNode(geometry: box)
        boundingBoxNode.name = "DEBUG_BOUNDING_BOX"
        
        // Position the bounding box at the center of the model's bounds
        let center = SCNVector3(
            (minBound.x + maxBound.x) / 2,
            (minBound.y + maxBound.y) / 2,
            (minBound.z + maxBound.z) / 2
        )
        boundingBoxNode.position = center
        
        // Add as child of the model node
        node.addChildNode(boundingBoxNode)
        
        print("📦 Added bounding box visualization: size(\(width), \(height), \(depth)), center(\(center))")
    }
    
    // Log detailed placement information for debugging
    private func logPlacementDebugInfo(_ node: SCNNode, at position: SCNVector3) {
        print("🔍 DEBUG: Placement Analysis")
        print("   Placement Position: \(position)")
        print("   Node World Position: \(node.worldPosition)")
        print("   Node Local Position: \(node.position)")
        print("   Node Scale: \(node.scale)")
        print("   Node Name: \(node.name ?? "unnamed")")
        
        // Log bounding box information
        let (minBound, maxBound) = node.boundingBox
        print("   Node Bounding Box: min(\(minBound)), max(\(maxBound))")
        
        // Log child node hierarchy
        print("   Child Nodes: \(node.childNodes.count)")
        for (index, childNode) in node.childNodes.enumerated() {
            print("     Child \(index): \(childNode.name ?? "unnamed"), pos(\(childNode.position)), scale(\(childNode.scale))")
            
            // Check for geometry
            if let geometry = childNode.geometry {
                print("       Geometry: \(type(of: geometry)), materials: \(geometry.materials.count)")
            }
        }
        
        // Log camera distance for perspective
        if let sceneView = sceneView,
           let cameraNode = sceneView.pointOfView {
            let cameraPosition = cameraNode.worldPosition
            let distance = sqrt(
                pow(position.x - cameraPosition.x, 2) +
                pow(position.y - cameraPosition.y, 2) +
                pow(position.z - cameraPosition.z, 2)
            )
            print("   Distance from Camera: \(distance)")
        }
    }
}

// MARK: - ARPlacedObject Model
struct ARPlacedObject: Identifiable {
    let id: UUID
    let node: SCNNode
    let originalImage: UIImage?
    let position: SCNVector3
    let is3DModel: Bool // Whether this is a 3D model or segmented image
    let createdAt = Date()
    
    // Calculate estimated dimensions
    var estimatedSize: CGSize {
        if is3DModel {
            // For 3D models, calculate from bounding box
            let (minBound, maxBound) = node.boundingBox
            return CGSize(
                width: CGFloat(maxBound.x - minBound.x),
                height: CGFloat(maxBound.y - minBound.y)
            )
        } else if let geometry = node.geometry as? SCNPlane {
            // For segmented images, use plane dimensions
            return CGSize(width: geometry.width, height: geometry.height)
        }
        return CGSize(width: 1.0, height: 1.0)
    }
    
    // Get object type description
    var objectType: String {
        return is3DModel ? "3D Model" : "Segmented Image"
    }
}

// MARK: - Extensions for 3D Math
extension SCNVector3 {
    // Vector addition
    static func + (left: SCNVector3, right: SCNVector3) -> SCNVector3 {
        return SCNVector3(left.x + right.x, left.y + right.y, left.z + right.z)
    }
    
    // Vector subtraction
    static func - (left: SCNVector3, right: SCNVector3) -> SCNVector3 {
        return SCNVector3(left.x - right.x, left.y - right.y, left.z - right.z)
    }
    
    // Vector scaling
    static func * (vector: SCNVector3, scalar: Float) -> SCNVector3 {
        return SCNVector3(vector.x * scalar, vector.y * scalar, vector.z * scalar)
    }
    
    // Vector length
    var length: Float {
        return sqrt(x * x + y * y + z * z)
    }
    
    // Vector normalization
    var normalized: SCNVector3 {
        let len = length
        return len > 0 ? SCNVector3(x / len, y / len, z / len) : SCNVector3(0, 0, 0)
    }
}