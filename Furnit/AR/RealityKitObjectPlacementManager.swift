import RealityKit
import ARKit
import UIKit
import simd

@MainActor
class RealityKitObjectPlacementManager: ObservableObject {
    // Published properties for UI updates
    @Published var placedObjects: [RealityKitPlacedObject] = []
    @Published var isReadyToPlace = false
    
    // Scene references
    weak var arView: ARView?
    weak var scene: RealityKit.Scene?
    
    // World anchor reference for navigation compatibility
    weak var worldAnchor: AnchorEntity?
    
    // Object placement properties  
    private var generatedModel: Entity? // 3D model from backend API
    private let placementHeight: Float = 0.05 // Small lift above floor for visual clarity
    
    // Placement distance constraints for better user experience
    private let maxPlacementDistance: Float = 8.0 // Maximum distance from camera for placement (reduced from 50.0)
    private let preferredPlacementDistance: Float = 2.5 // Preferred distance when using fallback placement (reduced from 3.0)
    private let minPlacementDistance: Float = 1.0 // Minimum distance to avoid placing too close to camera
    
    init() {}
    
    // Set scene references for object placement
    func setSceneReferences(arView: ARView, scene: RealityKit.Scene) {
        self.arView = arView
        self.scene = scene
    }
    
    // Set world anchor reference for navigation compatibility
    func setWorldAnchor(_ anchor: AnchorEntity) {
        self.worldAnchor = anchor
        print("🌍 World anchor reference set for object placement")
    }
    
    // Prepare for object placement with 3D model from backend API
    func prepareForPlacement(with3DModel model: Entity) {
        self.generatedModel = model.clone(recursive: true)
        isReadyToPlace = true
        print("✅ Ready to place 3D model with \(model.children.count) child entities")
    }
    
    // Handle tap gesture to place object in 3D scene
    func handleTapToPlace(at screenPoint: CGPoint) -> Bool {
        guard let arView = arView,
              let scene = scene,
              isReadyToPlace,
              generatedModel != nil else {
            print("⚠️ Not ready to place object")
            return false
        }
        
        // Determine optimal placement position with distance constraints
        let placementPosition = determinePlacementPosition(for: screenPoint, in: arView)
        
        // Place 3D model from backend API
        guard let modelToPlace = generatedModel else {
            print("⚠️ No generated model available for placement")
            return false
        }
        return place3DModel(modelToPlace, at: placementPosition, in: scene)
    }
    
    // Determine optimal placement position using RealityKit raycasting
    private func determinePlacementPosition(for screenPoint: CGPoint, in arView: ARView) -> SIMD3<Float> {
        let cameraTransform = arView.cameraTransform
        let cameraPosition = cameraTransform.translation
        print("📍 Camera position: \(cameraPosition)")
        
        // Perform raycast to find potential placement surfaces
        let raycastResults = arView.raycast(from: screenPoint, allowing: .existingPlaneGeometry, alignment: .any)
        
        // Look for a hit within acceptable distance
        for result in raycastResults {
            let hitPosition = SIMD3<Float>(result.worldTransform.columns.3.x, 
                                         result.worldTransform.columns.3.y, 
                                         result.worldTransform.columns.3.z)
            let distance = simd_length(hitPosition - cameraPosition)
            
            print("🎯 Raycast result: position(\(hitPosition)), distance(\(distance))")
            
            // Check if this hit is within acceptable distance
            if distance <= maxPlacementDistance {
                // Adjust position slightly above the surface
                let adjustedPosition = hitPosition + SIMD3<Float>(0, placementHeight, 0)
                
                print("✅ Found suitable placement surface at distance \(distance)")
                return adjustedPosition
            }
        }
        
        // No suitable hit found within distance limit, use fallback placement strategy
        print("⚠️ No suitable surface within \(maxPlacementDistance) units, using fallback placement")
        return calculateFallbackPlacement(for: screenPoint, in: arView)
    }
    
    // Calculate fallback placement position in front of camera
    private func calculateFallbackPlacement(for screenPoint: CGPoint, in arView: ARView) -> SIMD3<Float> {
        let cameraTransform = arView.cameraTransform
        let cameraPosition = cameraTransform.translation
        
        // Convert screen point to world direction
        let worldDirection = screenPointToWorldDirection(screenPoint, in: arView)
        
        // Determine if user is tapping in lower half of screen (floor) or upper half (wall/air)
        let viewportSize = arView.bounds.size
        let isTappingFloor = screenPoint.y > viewportSize.height * 0.6 // Lower 40% of screen
        
        var placementPosition: SIMD3<Float>
        
        if isTappingFloor {
            // Place on floor - find intersection with floor plane
            placementPosition = calculateFloorPlacement(cameraPosition: cameraPosition, worldDirection: worldDirection)
        } else {
            // Place at eye level in front of camera at better viewing distance
            placementPosition = cameraPosition + worldDirection * preferredPlacementDistance
        }

        // Ensure minimum distance from camera for visibility
        let distanceFromCamera = simd_length(placementPosition - cameraPosition)
        if distanceFromCamera < minPlacementDistance {
            let direction = simd_normalize(placementPosition - cameraPosition)
            placementPosition = cameraPosition + direction * minPlacementDistance
            print("📐 Adjusted placement to minimum distance: \(minPlacementDistance)m")
        }

        print("🎯 Fallback placement: camera(\(cameraPosition)), direction(\(worldDirection))")
        print("   Tapping floor: \(isTappingFloor), final position: \(placementPosition)")

        // Validate placement position is reasonable and within view
        let validatedPosition = validatePlacementPosition(placementPosition, cameraPosition: cameraPosition)

        return validatedPosition
    }
    
    // Convert screen point to world direction for raycasting
    private func screenPointToWorldDirection(_ screenPoint: CGPoint, in arView: ARView) -> SIMD3<Float> {
        // Convert screen point to normalized device coordinates
        let viewportSize = arView.bounds.size
        let normalizedX = Float((screenPoint.x / viewportSize.width) * 2.0 - 1.0)
        let normalizedY = Float(((viewportSize.height - screenPoint.y) / viewportSize.height) * 2.0 - 1.0)
        
        // Get camera's field of view and aspect ratio
        let fieldOfView: Float = 75.0 // Default FOV
        let aspect = Float(viewportSize.width / viewportSize.height)
        
        // Convert to camera space direction
        let fovRadians = fieldOfView * Float.pi / 180.0
        let directionX = normalizedX * tan(fovRadians / 2.0) * aspect
        let directionY = normalizedY * tan(fovRadians / 2.0)
        let directionZ: Float = -1.0 // Forward in camera space
        
        // Transform to world space using camera transform
        let cameraTransform = arView.cameraTransform
        let localDirection = normalize(SIMD3<Float>(directionX, directionY, directionZ))
        
        // Transform direction from camera space to world space
        let worldDirection = cameraTransform.rotation.act(localDirection)
        
        return worldDirection
    }
    
    // Calculate placement on floor plane
    private func calculateFloorPlacement(cameraPosition: SIMD3<Float>, worldDirection: SIMD3<Float>) -> SIMD3<Float> {
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
                return SIMD3<Float>(
                    cameraPosition.x + worldDirection.x * t,
                    floorHeight + placementHeight, // Slightly above floor
                    cameraPosition.z + worldDirection.z * t
                )
            }
        }
        
        // Fallback: place on floor at preferred distance horizontally
        let horizontalDirection = normalize(SIMD3<Float>(worldDirection.x, 0, worldDirection.z))
        let floorPlacementDistance = min(preferredPlacementDistance, 4.0) // Cap at 4 units for floor placement
        return SIMD3<Float>(
            cameraPosition.x + horizontalDirection.x * floorPlacementDistance,
            floorHeight + placementHeight,
            cameraPosition.z + horizontalDirection.z * floorPlacementDistance
        )
    }
    
    // Detect floor height from scene geometry or use default
    private func detectFloorHeight() -> Float? {
        guard let scene = scene else { return nil }
        
        // Look for anchors with low Y coordinates to find floor
        var minY: Float = Float.greatestFiniteMagnitude
        var hasGeometry = false
        
        for anchor in scene.anchors {
            if let modelEntity = anchor.children.first(where: { $0.components.has(ModelComponent.self) }) {
                let bounds = modelEntity.components[ModelComponent.self]?.mesh.bounds
                if let bounds = bounds {
                    let worldMin = anchor.transform.matrix * SIMD4<Float>(bounds.min, 1)
                    
                    if worldMin.y < minY {
                        minY = worldMin.y
                        hasGeometry = true
                    }
                }
            }
        }
        
        // Return detected floor height or reasonable default
        return hasGeometry ? minY : 0.0
    }
    
    // Place 3D model from backend API in the scene
    private func place3DModel(_ model: Entity, at position: SIMD3<Float>, in scene: RealityKit.Scene) -> Bool {
        // Clone the model to avoid modifying the original
        let placedModelEntity = model.clone(recursive: true)

        // Ensure model has proper scale and materials for visibility
        adjustModelForVisibility(placedModelEntity)

        // Set position on the model entity directly
        placedModelEntity.position = position
        
        // Create anchor entity for tracking (use world anchor or create independent)
        let anchorEntity: AnchorEntity
        if let worldAnchor = worldAnchor {
            worldAnchor.addChild(placedModelEntity)
            anchorEntity = worldAnchor
            print("📦 Added model to world anchor for navigation compatibility")
        } else {
            // Fallback: create independent anchor
            anchorEntity = AnchorEntity(.world(transform: Transform(translation: position).matrix))
            anchorEntity.addChild(placedModelEntity)
            scene.addAnchor(anchorEntity)
            print("⚠️ No world anchor available, created independent anchor")
        }
        
        // Track placed object
        let placedObject = RealityKitPlacedObject(
            id: UUID(),
            entity: placedModelEntity,
            anchorEntity: anchorEntity,
            originalImage: nil,
            position: position,
            is3DModel: true
        )
        
        placedObjects.append(placedObject)
        
        // Reset placement state
        isReadyToPlace = false
        generatedModel = nil
        
        // Add debug visualization to help locate placed objects
        addDebugVisualization(for: placedModelEntity, at: position)

        print("✅ 3D model placed successfully at position: \(position)")
        print("   Model has \(placedModelEntity.children.count) child entities")
        print("   Model scale: \(placedModelEntity.scale)")
        print("   Model bounds: \(getEntityBounds(placedModelEntity))")
        print("   Distance from camera: \(simd_length(position - (arView?.cameraTransform.translation ?? SIMD3<Float>(0, 0, 0))))m")

        // Log detailed entity hierarchy for debugging
        logEntityHierarchy(placedModelEntity, level: 0)

        return true
    }
    
    // Create AR entity from segmented image
    // Segmentation-based methods removed - using backend 3D model generation instead
    
    // Add shadow plane beneath AR object for better ground integration
    // Temporarily disabled due to SimpleMaterial color API complexity
    private func addShadowPlane(to parentEntity: Entity, size: CGSize) {
        // TODO: Implement shadow plane with correct RealityKit material API
        print("🔧 Shadow plane temporarily disabled - material API needs fixing")
    }

    // Add debug visualization to help locate placed objects
    private func addDebugVisualization(for entity: Entity, at position: SIMD3<Float>) {
        guard let scene = scene else { return }

        // Create debug sphere at placement position
        let debugSphere = ModelEntity(
            mesh: .generateSphere(radius: 0.05),
            materials: [SimpleMaterial(color: .red, roughness: 0.5, isMetallic: false)]
        )
        debugSphere.name = "debug_sphere"
        debugSphere.position = position + SIMD3<Float>(0, 0.1, 0) // Slightly above placed object

        // Create debug anchor for the sphere
        let debugAnchor = AnchorEntity(.world(transform: Transform(translation: debugSphere.position).matrix))
        debugAnchor.addChild(debugSphere)
        scene.addAnchor(debugAnchor)

        print("🔴 Debug sphere added at: \(debugSphere.position)")

        // Remove debug sphere after 10 seconds
        DispatchQueue.main.asyncAfter(deadline: .now() + 10.0) {
            scene.removeAnchor(debugAnchor)
            print("🔴 Debug sphere removed")
        }
    }

    // Get entity bounds for debugging
    private func getEntityBounds(_ entity: Entity) -> String {
        if let modelComponent = entity.components[ModelComponent.self] {
            let bounds = modelComponent.mesh.bounds
            let size = bounds.max - bounds.min
            return "size(\(size)), min(\(bounds.min)), max(\(bounds.max))"
        }

        // Check children for bounds
        for child in entity.children {
            if let modelComponent = child.components[ModelComponent.self] {
                let bounds = modelComponent.mesh.bounds
                let size = bounds.max - bounds.min
                return "child size(\(size)), min(\(bounds.min)), max(\(bounds.max))"
            }
        }

        return "no bounds found"
    }

    // Log detailed entity hierarchy for debugging visibility issues
    private func logEntityHierarchy(_ entity: Entity, level: Int) {
        let indent = String(repeating: "  ", count: level)
        let hasModel = entity.components.has(ModelComponent.self)
        let materialCount = entity.components[ModelComponent.self]?.materials.count ?? 0

        print("\(indent)🔍 Entity: \(entity.name.isEmpty ? "unnamed" : entity.name)")
        print("\(indent)   - Has ModelComponent: \(hasModel)")
        print("\(indent)   - Material count: \(materialCount)")
        print("\(indent)   - Position: \(entity.position)")
        print("\(indent)   - Scale: \(entity.scale)")
        print("\(indent)   - Children: \(entity.children.count)")

        if hasModel, let modelComponent = entity.components[ModelComponent.self] {
            let bounds = modelComponent.mesh.bounds
            print("\(indent)   - Bounds: min(\(bounds.min)), max(\(bounds.max))")

            for (index, material) in modelComponent.materials.enumerated() {
                if let simpleMaterial = material as? SimpleMaterial {
                    print("\(indent)   - Material \(index): color=\(simpleMaterial.color), roughness=\(simpleMaterial.roughness)")
                } else {
                    print("\(indent)   - Material \(index): type=\(type(of: material))")
                }
            }
        }

        // Recursively log children
        for child in entity.children {
            logEntityHierarchy(child, level: level + 1)
        }
    }

    // Validate placement position is reasonable and within camera view
    private func validatePlacementPosition(_ position: SIMD3<Float>, cameraPosition: SIMD3<Float>) -> SIMD3<Float> {
        var validatedPosition = position

        // Ensure object is not too close or too far from camera
        let distance = simd_length(position - cameraPosition)
        let minDistance: Float = 0.5 // 50cm minimum
        let maxDistance: Float = 8.0 // 8m maximum

        if distance < minDistance {
            // Too close - move further away
            let direction = simd_normalize(position - cameraPosition)
            validatedPosition = cameraPosition + direction * minDistance
            print("📐 Object too close (\(distance)m), moved to \(minDistance)m")
        } else if distance > maxDistance {
            // Too far - bring closer
            let direction = simd_normalize(position - cameraPosition)
            validatedPosition = cameraPosition + direction * maxDistance
            print("📐 Object too far (\(distance)m), moved to \(maxDistance)m")
        }

        // Ensure object is at reasonable height (not underground or floating too high)
        let minHeight: Float = 0.0  // Floor level
        let maxHeight: Float = 3.0  // 3m ceiling

        if validatedPosition.y < minHeight {
            validatedPosition.y = minHeight
            print("📐 Object below floor, moved to floor level")
        } else if validatedPosition.y > maxHeight {
            validatedPosition.y = maxHeight
            print("📐 Object above ceiling, moved to ceiling level")
        }

        if validatedPosition != position {
            print("✅ Position validated: \(position) → \(validatedPosition)")
        }

        return validatedPosition
    }

    // Adjust model scale and materials for better visibility
    private func adjustModelForVisibility(_ entity: Entity) {
        // Fix scale hierarchy issues first
        fixEntityScaleHierarchy(entity)

        // Check if entity has model component and get bounds
        if let modelComponent = entity.components[ModelComponent.self] {
            let bounds = modelComponent.mesh.bounds
            let size = bounds.max - bounds.min
            let maxDimension = max(size.x, max(size.y, size.z))

            print("📏 Original model size: \(size), max dimension: \(maxDimension)")

            // If model is too small (less than 10cm), scale it up
            if maxDimension < 0.1 {
                let scaleFactor: Float = 0.5 / maxDimension // Scale to 50cm
                entity.scale = SIMD3<Float>(scaleFactor, scaleFactor, scaleFactor)
                print("🔧 Scaled up tiny model by factor: \(scaleFactor)")
            }
            // If model is extremely large (more than 5m), scale it down
            else if maxDimension > 5.0 {
                let scaleFactor: Float = 2.0 / maxDimension // Scale to 2m
                entity.scale = SIMD3<Float>(scaleFactor, scaleFactor, scaleFactor)
                print("🔧 Scaled down large model by factor: \(scaleFactor)")
            }

            // Ensure model has materials
            if modelComponent.materials.isEmpty {
                var updatedComponent = modelComponent
                updatedComponent.materials = [SimpleMaterial(color: .white, roughness: 0.5, isMetallic: false)]
                entity.components.set(updatedComponent)
                print("🎨 Added default material to placed model")
            }
        }

        // Recursively adjust child entities
        for child in entity.children {
            adjustModelForVisibility(child)
        }
    }

    // Fix entity scale hierarchy issues that make models invisible
    private func fixEntityScaleHierarchy(_ entity: Entity) {
        let currentScale = entity.scale
        let minVisibleScale: Float = 0.1 // Any scale smaller than this is problematic

        // Check if any component of the scale is too small
        if currentScale.x < minVisibleScale || currentScale.y < minVisibleScale || currentScale.z < minVisibleScale {
            print("⚠️ Found problematic scale on entity '\(entity.name.isEmpty ? "unnamed" : entity.name)': \(currentScale)")

            // Reset to reasonable scale - preserve proportions but make visible
            let scaleFactor: Float = 1.0 / max(currentScale.x, max(currentScale.y, currentScale.z))
            let correctedScale = currentScale * scaleFactor
            entity.scale = correctedScale

            print("🔧 Corrected entity scale from \(currentScale) to \(correctedScale)")
        }

        // Recursively fix child entities
        for child in entity.children {
            fixEntityScaleHierarchy(child)
        }
    }

    // Remove specific AR object
    func removeObject(_ objectId: UUID) {
        if let index = placedObjects.firstIndex(where: { $0.id == objectId }) {
            let object = placedObjects[index]
            scene?.removeAnchor(object.anchorEntity)
            placedObjects.remove(at: index)
            
            print("🗑️ Removed AR object with ID: \(objectId)")
        }
    }
    
    // Clear all placed AR objects
    func clearAllObjects() {
        for placedObject in placedObjects {
            scene?.removeAnchor(placedObject.anchorEntity)
        }
        placedObjects.removeAll()
        
        isReadyToPlace = false
        generatedModel = nil
        
        print("🧹 Cleared all AR objects")
    }
    
    // Reset for new AR session
    func resetForNewSession() {
        clearAllObjects()
        print("🔄 Reset RealityKit placement manager for new session")
    }
    
    // Monitor placed objects to ensure they remain visible when camera moves
    func validateObjectVisibility() {
        guard let arView = arView else { return }
        
        let cameraPosition = arView.cameraTransform.translation
        var invisibleCount = 0
        
        for placedObject in placedObjects {
            let distance = simd_length(placedObject.position - cameraPosition)
            
            // Check if object is beyond reasonable viewing distance
            if distance > 200.0 {
                print("⚠️ Object at \(placedObject.position) is \(distance) units from camera - may be invisible")
                invisibleCount += 1
            }
            
            // Ensure object is still in scene
            if scene!.anchors.first(where: { $0 === placedObject.anchorEntity }) == nil {
                print("⚠️ Object anchor has been removed from scene - re-adding")
                scene?.addAnchor(placedObject.anchorEntity)
            }
        }
        
        if invisibleCount > 0 {
            print("📊 Visibility check: \(invisibleCount)/\(placedObjects.count) objects may be invisible")
        }
    }
}

// MARK: - RealityKit Placed Object Model
struct RealityKitPlacedObject: Identifiable {
    let id: UUID
    let entity: Entity
    let anchorEntity: AnchorEntity
    let originalImage: UIImage?
    let position: SIMD3<Float>
    let is3DModel: Bool
    let createdAt = Date()
    
    // Compatibility properties for existing code
    var node: Entity { return entity }
    
    // Calculate estimated dimensions
    var estimatedSize: CGSize {
        if is3DModel {
            // For 3D models, calculate from bounding box
            if let modelComponent = entity.components[ModelComponent.self] {
                let bounds = modelComponent.mesh.bounds
                let size = bounds.max - bounds.min
                return CGSize(width: CGFloat(size.x), height: CGFloat(size.y))
            }
        }
        return CGSize(width: 1.0, height: 1.0)
    }
    
    // Get object type description
    var objectType: String {
        return is3DModel ? "3D Model" : "Segmented Image"
    }
}

// MARK: - Extensions for SIMD math operations
extension SIMD3 where Scalar == Float {
    static func + (left: SIMD3<Float>, right: SIMD3<Float>) -> SIMD3<Float> {
        return SIMD3<Float>(left.x + right.x, left.y + right.y, left.z + right.z)
    }
    
    static func - (left: SIMD3<Float>, right: SIMD3<Float>) -> SIMD3<Float> {
        return SIMD3<Float>(left.x - right.x, left.y - right.y, left.z - right.z)
    }
    
    static func * (vector: SIMD3<Float>, scalar: Float) -> SIMD3<Float> {
        return SIMD3<Float>(vector.x * scalar, vector.y * scalar, vector.z * scalar)
    }
    
    var length: Float {
        return sqrt(x * x + y * y + z * z)
    }
    
    var normalized: SIMD3<Float> {
        let len = length
        return len > 0 ? SIMD3<Float>(x / len, y / len, z / len) : SIMD3<Float>(0, 0, 0)
    }
}

// MARK: - Compatibility Aliases
// These aliases ensure compatibility with existing code that expects the old SceneKit types
typealias ARObjectPlacementManager = RealityKitObjectPlacementManager
typealias ARPlacedObject = RealityKitPlacedObject
typealias CameraMovementManager = RealityKitCameraMovementManager