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
    private let placementHeight: Float = 0.05 // Small lift above floor for visual clarity
    
    init() {}
    
    // Set scene references for object placement
    func setSceneReferences(sceneView: SCNView, scene: SCNScene) {
        self.sceneView = sceneView
        self.scene = scene
    }
    
    // Prepare for object placement with segmented image
    func prepareForPlacement(with segmentedImage: UIImage) {
        self.segmentedImage = segmentedImage
        isReadyToPlace = true
    }
    
    // Handle tap gesture to place object in 3D scene
    func handleTapToPlace(at screenPoint: CGPoint) -> Bool {
        guard let sceneView = sceneView,
              let scene = scene,
              let segmentedImage = segmentedImage,
              isReadyToPlace else {
            print("⚠️ Not ready to place object")
            return false
        }
        
        // Perform hit test to find placement location
        let hitTestOptions: [SCNHitTestOption: Any] = [
            .boundingBoxOnly: false,
            .firstFoundOnly: true
        ]
        
        let hitResults = sceneView.hitTest(screenPoint, options: hitTestOptions)
        
        guard let firstHit = hitResults.first else {
            print("⚠️ No valid placement surface found")
            return false
        }
        
        // Calculate placement position
        let hitPosition = firstHit.worldCoordinates
        let hitNormal = firstHit.worldNormal
        
        // Adjust position slightly above the surface
        let placementPosition = SCNVector3(
            hitPosition.x + hitNormal.x * placementHeight,
            hitPosition.y + hitNormal.y * placementHeight,
            hitPosition.z + hitNormal.z * placementHeight
        )
        
        // Create and place the AR object
        if let arObject = createARObject(from: segmentedImage, at: placementPosition, normal: hitNormal) {
            // Add to scene
            scene.rootNode.addChildNode(arObject.node)
            
            // Track placed object
            let placedObject = ARPlacedObject(
                id: UUID(),
                node: arObject.node,
                originalImage: segmentedImage,
                position: placementPosition
            )
            
            placedObjects.append(placedObject)
            
            print("✅ AR object placed successfully at position: \(placementPosition)")
            return true
        }
        
        return false
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
    
    // Clear all placed AR objects
    func clearAllObjects() {
        for placedObject in placedObjects {
            placedObject.node.removeFromParentNode()
        }
        placedObjects.removeAll()
        isReadyToPlace = false
        segmentedImage = nil
        
        print("🧹 Cleared all AR objects")
    }
    
    // Reset for new AR session
    func resetForNewSession() {
        clearAllObjects()
        print("🔄 Reset AR placement manager for new session")
    }
}

// MARK: - ARPlacedObject Model
struct ARPlacedObject: Identifiable {
    let id: UUID
    let node: SCNNode
    let originalImage: UIImage
    let position: SCNVector3
    let createdAt = Date()
    
    // Calculate estimated dimensions
    var estimatedSize: CGSize {
        if let geometry = node.geometry as? SCNPlane {
            return CGSize(width: geometry.width, height: geometry.height)
        }
        return CGSize(width: 1.0, height: 1.0)
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