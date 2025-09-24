import RealityKit
import ARKit
import UIKit
import simd

@MainActor
class RealityKitObjectPlacementManager: ObservableObject {
    // Published properties for UI updates
    @Published var placedObjects: [RealityKitPlacedObject] = []
    @Published var isReadyToPlace = false

    // Object manipulation properties
    @Published var isManipulatingObject = false // Track if we're in manipulation mode
    @Published var selectedObject: RealityKitPlacedObject? // Currently selected object for manipulation

    // Scene references
    weak var arView: ARView?
    weak var scene: RealityKit.Scene?

    // World anchor reference for navigation compatibility
    weak var worldAnchor: AnchorEntity?

    // Object placement properties
    private var generatedModel: Entity? // 3D model from backend API
    private let placementHeight: Float = 0.05 // Small lift above floor for visual clarity

    // Callback for successful object placement (to trigger return to scanning mode)
    var onObjectPlaced: (() -> Void)?

    // Callbacks for manipulation mode changes
    var onManipulationStart: (() -> Void)?
    var onManipulationEnd: (() -> Void)?
    
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
    
    // Calculate fallback placement position (furniture always goes on floor)
    private func calculateFallbackPlacement(for screenPoint: CGPoint, in arView: ARView) -> SIMD3<Float> {
        let cameraTransform = arView.cameraTransform
        let cameraPosition = cameraTransform.translation

        // Convert screen point to world direction
        let worldDirection = screenPointToWorldDirection(screenPoint, in: arView)

        print("🎯 Fallback placement: camera(\(cameraPosition)), direction(\(worldDirection))")
        print("   Furniture placement: ALWAYS on floor (ignoring tap location)")

        // For furniture, ALWAYS place on floor regardless of where user taps
        // This ensures consistent, predictable placement for furniture objects
        let placementPosition = calculateFloorPlacement(cameraPosition: cameraPosition, worldDirection: worldDirection)

        // Ensure minimum distance from camera for visibility (but keep on floor)
        let distanceFromCamera = simd_length(placementPosition - cameraPosition)
        var finalPosition = placementPosition

        if distanceFromCamera < minPlacementDistance {
            let horizontalDirection = normalize(SIMD3<Float>(worldDirection.x, 0, worldDirection.z))
            finalPosition = SIMD3<Float>(
                cameraPosition.x + horizontalDirection.x * minPlacementDistance,
                placementPosition.y, // Keep same floor height
                cameraPosition.z + horizontalDirection.z * minPlacementDistance
            )
            print("📐 Adjusted furniture to minimum distance: \(minPlacementDistance)m, keeping floor level")
        }

        print("   🪑 Final furniture position: \(finalPosition) (Y=floor)")

        // Validate placement position is reasonable and within view
        let validatedPosition = validatePlacementPosition(finalPosition, cameraPosition: cameraPosition)

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
    
    // Calculate placement on floor plane (furniture always goes on floor)
    private func calculateFloorPlacement(cameraPosition: SIMD3<Float>, worldDirection: SIMD3<Float>) -> SIMD3<Float> {
        // Get reliable floor height (always returns a value, defaults to 0.0)
        let floorHeight = detectFloorHeight()
        let furnitureGroundClearance: Float = 0.01 // Small clearance above floor

        print("🏠 Calculating floor placement with floor height: \(floorHeight)")

        // Try to intersect camera ray with floor plane for natural placement
        if abs(worldDirection.y) > 0.001 { // Avoid division by zero
            let t = (floorHeight - cameraPosition.y) / worldDirection.y

            // Use positive t (forward direction) with reasonable distance limits
            if t > 0.5 && t <= 8.0 { // Min 0.5m, max 8m for floor placement
                let intersectionPoint = SIMD3<Float>(
                    cameraPosition.x + worldDirection.x * t,
                    floorHeight + furnitureGroundClearance,
                    cameraPosition.z + worldDirection.z * t
                )

                print("🎯 Floor intersection found at distance: \(t)m, position: \(intersectionPoint)")
                return intersectionPoint
            } else {
                print("⚠️ Floor intersection too close (\(t)m) or too far, using horizontal placement")
            }
        } else {
            print("⚠️ Camera pointing horizontally, cannot intersect floor plane")
        }

        // Fallback: Place horizontally at comfortable distance on floor
        let horizontalDirection = normalize(SIMD3<Float>(worldDirection.x, 0, worldDirection.z))
        let safeDistance: Float = 2.5 // Comfortable viewing distance for furniture
        let floorPosition = SIMD3<Float>(
            cameraPosition.x + horizontalDirection.x * safeDistance,
            floorHeight + furnitureGroundClearance,
            cameraPosition.z + horizontalDirection.z * safeDistance
        )

        print("🎯 Horizontal floor placement at: \(floorPosition)")
        return floorPosition
    }
    
    // Detect floor height from scene geometry or use default (always returns a value)
    private func detectFloorHeight() -> Float {
        guard let scene = scene else {
            print("🏠 No scene available, using default floor height: 0.0")
            return 0.0
        }

        // First try: Look for ARKit plane anchors (floor detection)
        if let arView = arView {
            let planeAnchors = arView.scene.anchors.compactMap { $0 as? AnchorEntity }
            for anchor in planeAnchors {
                // Check if this is a horizontal plane (floor)
                let transform = anchor.transform
                let yPosition = transform.translation.y

                // Floor planes should be roughly horizontal and below camera
                if yPosition < 1.0 && yPosition > -1.0 { // Within reasonable floor range
                    print("🏠 Found ARKit plane at height: \(yPosition)")
                    return yPosition
                }
            }
        }

        // Second try: Look for existing objects to determine floor level
        var lowestObjectY: Float = Float.greatestFiniteMagnitude
        var hasObjects = false

        for placedObject in placedObjects {
            let objectY = placedObject.entity.position.y
            if objectY < lowestObjectY {
                lowestObjectY = objectY
                hasObjects = true
            }
        }

        if hasObjects {
            print("🏠 Detected floor from existing objects at height: \(lowestObjectY)")
            return lowestObjectY
        }

        // Fallback: Use world origin as floor level (most reliable)
        print("🏠 Using default floor height: 0.0 (world origin)")
        return 0.0
    }
    
    // Place 3D model from backend API in the scene
    private func place3DModel(_ model: Entity, at position: SIMD3<Float>, in scene: RealityKit.Scene) -> Bool {
        // Clone the model to avoid modifying the original
        let placedModelEntity = model.clone(recursive: true)

        // Detect and correct hierarchy position offsets internally within the model
        detectAndCorrectHierarchyOffsets(for: placedModelEntity)

        // Reset all scales to unity and get original mesh dimensions
        resetAllEntityScales(in: placedModelEntity)

        guard let originalMeshSize = calculateEntityBounds(placedModelEntity) else {
            print("⚠️ Could not calculate entity bounds, using default scale")
            return false
        }

        // Apply single root-level scale for target sizing
        let targetSize: Float = 1.8 // Target 1.8 meters for largest dimension
        applyTargetScale(to: placedModelEntity, targetSize: targetSize, originalBounds: originalMeshSize)

        // Fix materials to preserve colors and ensure visibility
        fixModelMaterials(for: placedModelEntity)

        // Use the original calculated position (respecting user tap when available)
        let basePosition = position
        print("🎯 Base calculated position: \(basePosition)")

        // Apply bounds-aware floor grounding to ensure furniture sits ON the floor
        let floorHeight = detectFloorHeight()
        let furnitureGroundClearance: Float = 0.01 // Small clearance above floor
        let groundedPosition = calculateGroundedPosition(
            basePosition: basePosition,
            entity: placedModelEntity,
            floorHeight: floorHeight,
            clearance: furnitureGroundClearance
        )

        print("🎯 Final grounded position: \(groundedPosition)")

        // Set position on the model entity directly
        placedModelEntity.position = groundedPosition
        
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
            originalImage: nil
        )
        
        placedObjects.append(placedObject)
        
        // Reset placement state
        isReadyToPlace = false
        generatedModel = nil
        
        // Clean placement completed without debug clutter

        print("✅ 3D model placed successfully at position: \(groundedPosition)")
        print("   Model has \(placedModelEntity.children.count) child entities")
        print("   Model scale: \(placedModelEntity.scale)")
        print("   Model bounds: \(getEntityBounds(placedModelEntity))")
        print("   Distance from camera: \(simd_length(groundedPosition - (arView?.cameraTransform.translation ?? SIMD3<Float>(0, 0, 0))))m")

        // Log detailed entity hierarchy for debugging
        logEntityHierarchy(placedModelEntity, level: 0)

        // Trigger callback to return to scanning mode after successful placement
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
            self.onObjectPlaced?()
            print("🔄 Triggered callback to return to AR scanning mode")
        }

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



    // Debug visualizations removed for clean AR experience
    private func addModelBoundsVisualization(for entity: Entity, at position: SIMD3<Float>) {
        // Debug visualization removed for clean AR experience
    }

    private func addMicroScaleDebugMarkers(for entity: Entity, at position: SIMD3<Float>) {
        // Debug visualization removed for clean AR experience
    }

    // Get a guaranteed visible position directly in front of camera for testing
    private func getGuaranteedVisiblePosition() -> SIMD3<Float> {
        // Get current camera position from ARView
        guard let arView = arView else {
            print("⚠️ No ARView available, using default position")
            return SIMD3<Float>(0, 1, -2) // Default position in front of origin
        }

        // Get camera transform
        let cameraTransform = arView.cameraTransform
        let cameraPosition = cameraTransform.translation

        // Calculate forward direction from camera (negative Z in camera space)
        let rotationMatrix = cameraTransform.matrix
        let forwardDirection = -normalize(SIMD3<Float>(
            rotationMatrix.columns.2.x,
            rotationMatrix.columns.2.y,
            rotationMatrix.columns.2.z
        ))

        // Place object 2 meters in front of camera at eye level
        let testPosition = cameraPosition + (forwardDirection * 2.0)

        print("🎯 Guaranteed position: \(testPosition)")
        return testPosition
    }

    // Get bounds description for logging
    private func getEntityBounds(_ entity: Entity) -> String {
        func findBounds(_ entity: Entity) -> BoundingBox? {
            if let modelComponent = entity.components[ModelComponent.self] {
                return modelComponent.mesh.bounds
            }
            for child in entity.children {
                if let bounds = findBounds(child) {
                    return bounds
                }
            }
            return nil
        }

        if let bounds = findBounds(entity) {
            return "min(\(bounds.min)), max(\(bounds.max))"
        }
        return "no bounds found"
    }

    // Log detailed entity hierarchy for debugging visibility issues
    private func logEntityHierarchy(_ entity: Entity, level: Int) {
        let indent = String(repeating: "  ", count: level)
        let hasModel = entity.components.has(ModelComponent.self)
        let materialCount = entity.components[ModelComponent.self]?.materials.count ?? 0

        let entityAddress = Unmanaged.passUnretained(entity).toOpaque()

        print("\(indent)🔍 Entity: \(entity.name.isEmpty ? "unnamed" : entity.name) [\(entityAddress)]")
        print("\(indent)   - Has ModelComponent: \(hasModel)")
        print("\(indent)   - Material count: \(materialCount)")
        print("\(indent)   - Position: \(entity.position)")
        print("\(indent)   - Scale: \(entity.scale)")
        print("\(indent)   - Children: \(entity.children.count)")

        if hasModel, let modelComponent = entity.components[ModelComponent.self] {
            let bounds = modelComponent.mesh.bounds
            print("\(indent)   - Bounds: min(\(bounds.min)), max(\(bounds.max))")

            // Log material information
            for (index, material) in modelComponent.materials.enumerated() {
                print("\(indent)   - Material \(index): \(type(of: material))")
            }
        }

        // Recursively log children
        for child in entity.children {
            logEntityHierarchy(child, level: level + 1)
        }
    }

    // Scale calculation and material validation methods continue below...
    // (Note: Debug visualization methods have been removed for clean AR experience)

    // Detect and correct hierarchy position offsets that can cause objects to be placed incorrectly
    private func detectAndCorrectHierarchyOffsets(for entity: Entity) {
        print("🔍 Detecting and fixing hierarchy position offsets...")

        // Define threshold for problematic position offsets (anything > 5 meters)
        let offsetThreshold: Float = 5.0
        var correctedCount = 0

        // Recursively search for and fix entities with large position offsets
        func findAndFixOffsets(_ currentEntity: Entity, accumulated: SIMD3<Float> = SIMD3<Float>(0, 0, 0)) {
            let currentPosition = currentEntity.position
            let newAccumulated = accumulated + currentPosition

            // Check if this entity has a significant offset
            let offsetMagnitude = simd_length(currentPosition)
            if offsetMagnitude > offsetThreshold {
                print("🚨 Found large position offset in entity '\(currentEntity.name)':")
                print("   Original position: \(currentPosition)")
                print("   Offset magnitude: \(offsetMagnitude)m")

                // Reset to zero position - the parent hierarchy should handle positioning
                currentEntity.position = SIMD3<Float>(0, 0, 0)
                correctedCount += 1

                print("   ✅ Reset position to origin")
            }

            // Continue recursively with children
            for child in currentEntity.children {
                findAndFixOffsets(child, accumulated: newAccumulated)
            }
        }

        // Start recursive search
        findAndFixOffsets(entity)

        if correctedCount > 0 {
            print("✅ No significant hierarchy offsets detected - no correction needed")
        } else {
            print("🔧 Corrected \(correctedCount) entities with large position offsets")
        }
    }

    // Validate and ensure placement position is reasonable for AR viewing
    private func validatePlacementPosition(_ position: SIMD3<Float>, cameraPosition: SIMD3<Float>) -> SIMD3<Float> {
        var validatedPosition = position

        // Ensure object is within reasonable distance from camera
        let distance = simd_length(position - cameraPosition)
        let maxDistance: Float = 8.0   // 8 meters max for furniture
        let minDistance: Float = 1.0   // 1 meter min for furniture viewing

        if distance > maxDistance {
            let horizontalDirection = normalize(SIMD3<Float>(position.x - cameraPosition.x, 0, position.z - cameraPosition.z))
            validatedPosition = SIMD3<Float>(
                cameraPosition.x + horizontalDirection.x * maxDistance,
                position.y, // Keep floor height
                cameraPosition.z + horizontalDirection.z * maxDistance
            )
            print("📐 Furniture too far (\(distance)m), moved to \(maxDistance)m (keeping floor level)")
        } else if distance < minDistance {
            let horizontalDirection = normalize(SIMD3<Float>(position.x - cameraPosition.x, 0, position.z - cameraPosition.z))
            validatedPosition = SIMD3<Float>(
                cameraPosition.x + horizontalDirection.x * minDistance,
                position.y, // Keep floor height
                cameraPosition.z + horizontalDirection.z * minDistance
            )
            print("📐 Furniture too close (\(distance)m), moved to \(minDistance)m (keeping floor level)")
        }

        // Note: Floor grounding is now handled at entity placement time with bounds awareness
        // This validation only handles distance constraints, not floor positioning

        // Sanity check: ensure Y coordinate is reasonable (basic validation)
        if validatedPosition.y < -2.0 || validatedPosition.y > 5.0 {
            validatedPosition.y = 0.5 // Default to reasonable height above world origin
            print("📐 Position Y coordinate invalid (\(position.y)), reset to safe default (0.5)")
        }

        print("✅ Validated furniture position: \(validatedPosition) (bounds-aware grounding will be applied)")
        return validatedPosition
    }

    // Reset all entity scales in hierarchy to unity (1,1,1)
    private func resetAllEntityScales(in entity: Entity) {
        print("🔍 Resetting ALL entity scales to (1,1,1) throughout hierarchy...")

        // Recursively reset all scales to unity
        func resetEntityScale(_ entity: Entity) {
            let originalScale = entity.scale
            entity.scale = SIMD3<Float>(1, 1, 1)

            print("🔧 Reset entity '\(entity.name.isEmpty ? "" : entity.name)' scale:")
            print("   Before: \(originalScale)")
            print("   After: \(entity.scale)")

            // Recursively process children
            for child in entity.children {
                resetEntityScale(child)
            }
        }

        // Start the recursive reset
        resetEntityScale(entity)
        print("✅ Scale reset complete - all entities now at unity scale")
    }

    // Calculate final entity bounds after scale reset
    private func calculateEntityBounds(_ entity: Entity) -> SIMD3<Float>? {
        // Find the first ModelComponent in the hierarchy to get bounds
        func findModelBounds(_ entity: Entity) -> BoundingBox? {
            if let modelComponent = entity.components[ModelComponent.self] {
                return modelComponent.mesh.bounds
            }
            for child in entity.children {
                if let bounds = findModelBounds(child) {
                    return bounds
                }
            }
            return nil
        }

        if let bounds = findModelBounds(entity) {
            let dimensions = bounds.extents
            print("📏 Found ModelComponent bounds at unity scale: \(dimensions)")
            return dimensions
        }

        print("⚠️ No ModelComponent bounds found")
        return nil
    }

    // Calculate floor-grounded position accounting for object bottom bounds
    // This ensures the bottom of the furniture touches the floor instead of the center/anchor point
    private func calculateGroundedPosition(basePosition: SIMD3<Float>, entity: Entity, floorHeight: Float, clearance: Float) -> SIMD3<Float> {
        // Get object bounds to calculate bottom offset
        if let unscaledBounds = calculateEntityBounds(entity) {
            // Get the entity's actual scale factor to calculate scaled bounds
            let entityScale = entity.scale

            // Calculate scaled bounds (account for the actual applied scale)
            let scaledBounds = SIMD3<Float>(
                unscaledBounds.x * entityScale.x,
                unscaledBounds.y * entityScale.y,
                unscaledBounds.z * entityScale.z
            )

            // Calculate the bottom extent of the scaled object (assuming anchor is at center)
            let bottomOffset = scaledBounds.y / 2.0 // Half height from center to bottom

            // Adjust Y position so the bottom of the object sits on the floor
            // Formula: objectCenter.y = floorHeight + clearance + bottomOffset
            let groundedY = floorHeight + clearance + bottomOffset

            print("📏 Grounding object:")
            print("   Unscaled bounds.y=\(unscaledBounds.y), entity scale=\(entityScale.y)")
            print("   Scaled bounds.y=\(scaledBounds.y), bottomOffset=\(bottomOffset)")
            print("   Original Y: \(basePosition.y) -> Grounded Y: \(groundedY)")

            return SIMD3<Float>(basePosition.x, groundedY, basePosition.z)
        } else {
            // No bounds available, use original position
            print("⚠️ No bounds available for grounding, using original position")
            return basePosition
        }
    }

    // Apply smart scaling based on calculated bounds
    private func applyTargetScale(to entity: Entity, targetSize: Float, originalBounds: SIMD3<Float>) {
        // Calculate the maximum dimension to scale uniformly
        let maxDimension = max(originalBounds.x, max(originalBounds.y, originalBounds.z))

        // Calculate scale factor to achieve target size
        let scaleFactor = targetSize / maxDimension
        let finalScale = SIMD3<Float>(scaleFactor, scaleFactor, scaleFactor)

        // Apply the calculated scale to the root entity
        entity.scale = finalScale

        print("🎯 Applied single root-level scale:")
        print("   Original mesh size: \(originalBounds)")
        print("   Max dimension: \(maxDimension)m")
        print("   Target size: \(targetSize)m")
        print("   Root scale factor: \(scaleFactor)")
        print("   Final root entity scale: \(finalScale)")
    }

    // Fix material validation and enhancement
    private func fixModelMaterials(for entity: Entity) {
        print("🔧 Fixing materials throughout entity hierarchy...")

        // Recursively process entity hierarchy for material fixes
        func processEntityMaterials(_ entity: Entity, level: Int = 0) {
            let entityDisplayName = entity.name.isEmpty ? "unnamed" : entity.name

            // Check if entity has ModelComponent with materials
            if let modelComponent = entity.components[ModelComponent.self] {
                print("🎯 Processing materials for entity '\(entityDisplayName)' at address: \(Unmanaged.passUnretained(entity).toOpaque())")
                print("   📋 Original material count: \(modelComponent.materials.count)")

                var updatedMaterials: [Material] = []
                var materialChanged = false

                // Process each material
                for (index, material) in modelComponent.materials.enumerated() {
                    print("   📋 Original material \(index): \(type(of: material))")

                    if let fixedMaterial = validateAndFixMaterial(material, index: index, entityName: entityDisplayName) {
                        updatedMaterials.append(fixedMaterial)
                        materialChanged = true
                        print("   🔄 Replaced material \(index) with \(type(of: fixedMaterial))")
                    } else {
                        updatedMaterials.append(material)
                        print("   ✅ Kept original material \(index)")
                    }
                }

                // Apply updated materials if any changes were made
                if materialChanged {
                    var updatedModelComponent = modelComponent
                    updatedModelComponent.materials = updatedMaterials
                    entity.components[ModelComponent.self] = updatedModelComponent
                    print("   📝 Updated materials array with \(updatedMaterials.count) materials")
                    print("   ✅ Applied updated ModelComponent with fixed materials")
                }
            }

            // Recursively process children
            for child in entity.children {
                processEntityMaterials(child, level: level + 1)
            }
        }

        // Start processing from root entity
        processEntityMaterials(entity)
        print("✅ Material fixes complete")
    }

    // Validate and fix individual materials (preserve good materials, only fix problematic ones)
    private func validateAndFixMaterial(_ material: Material, index: Int, entityName: String) -> Material? {
        if let pbrMaterial = material as? PhysicallyBasedMaterial {
            print("🔍 Inspecting PhysicallyBasedMaterial \(index) on entity '\(entityName)'")

            // First check if the material has a good texture - if so, preserve it completely
            if let baseTexture = pbrMaterial.baseColor.texture {
                print("   🖼️ Found texture - preserving PBR material as-is (no conversion needed)")
                print("   ✅ Material has texture, skipping all modifications")
                return nil // Keep original PBR material unchanged
            }

            // No texture, check if the color is problematic
            let baseColorTint = pbrMaterial.baseColor.tint
            print("   🎨 No texture found, checking color: \(baseColorTint)")

            // Check if color needs enhancement (more conservative check)
            if shouldEnhanceColor(baseColorTint) {
                print("   ⚠️ Color appears problematic, applying enhancement...")

                // Only now convert to UnlitMaterial with enhanced color
                let enhancedColor = enhanceColorIfNeeded(baseColorTint)
                let preservedMaterial = UnlitMaterial(color: enhancedColor)

                print("   🔧 Converted PBR → UnlitMaterial with enhanced color: \(enhancedColor)")
                print("   🌟 Material should now be visible and realistic")
                return preservedMaterial
            } else {
                print("   ✅ Color looks good, preserving original PBR material")
                return nil // Keep original PBR material unchanged
            }

        } else if let simpleMaterial = material as? SimpleMaterial {
            print("🔍 Inspecting SimpleMaterial \(index) on entity '\(entityName)'")

            // SimpleMaterials from RealityKit are typically well-formed
            // Only apply fixes if there are obvious issues
            print("   ✅ SimpleMaterial appears healthy")
            return nil // No fix needed for SimpleMaterials

        } else {
            // Other material types (UnlitMaterial, VideoMaterial, etc.)
            print("🔍 Found other material type \(type(of: material)) \(index) on entity '\(entityName)'")
            print("   ⚠️ Unknown material type - using fallback SimpleMaterial for safety")

            // Create safe fallback for unknown material types using dark grey UnlitMaterial
            let darkGreyColor = UIColor(red: 0.3, green: 0.3, blue: 0.3, alpha: 1.0)
            let fallbackMaterial = UnlitMaterial(color: darkGreyColor)

            print("   🔧 Converted unknown → UnlitMaterial")
            print("   🌟 Material should now be guaranteed visible")

            return fallbackMaterial
        }
    }

    // Fix problematic entity scales in hierarchy
    private func fixEntityScaleHierarchy(_ entity: Entity) {
        // Check if entity has problematic scale values
        let currentScale = entity.scale
        let hasProblematicScale = currentScale.x < 0.001 || currentScale.y < 0.001 || currentScale.z < 0.001 ||
                                 currentScale.x > 1000 || currentScale.y > 1000 || currentScale.z > 1000

        if hasProblematicScale {
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

        print("🗑️ Cleared all \(placedObjects.count) AR objects")
    }

    // Reset for new AR session
    func resetForNewSession() {
        clearAllObjects()
        isReadyToPlace = false
        generatedModel = nil

        print("🔄 Reset placement manager for new AR session")
    }

    // Diagnose object visibility
    func diagnoseObjectVisibility() {
        guard let arView = arView else {
            print("📊 No ARView available for diagnosis")
            return
        }

        let cameraPosition = arView.cameraTransform.translation
        var invisibleCount = 0

        print("📊 Diagnosing visibility of \(placedObjects.count) placed objects...")

        for placedObject in placedObjects {
            let distance = simd_length(placedObject.entity.position - cameraPosition)

            // Check if object is beyond reasonable viewing distance
            if distance > 200.0 {
                print("⚠️ Object at \(placedObject.entity.position) is \(distance) units from camera - may be invisible")
                invisibleCount += 1
            }
        }

        if invisibleCount > 0 {
            print("🚨 Found \(invisibleCount) potentially invisible objects")
        } else {
            print("✅ All objects appear to be within reasonable viewing distance")
        }
    }

    // Check if color needs enhancement (conservative approach)
    private func shouldEnhanceColor(_ color: UIColor) -> Bool {
        // Extract RGB components to check if color is truly problematic
        var red: CGFloat = 0, green: CGFloat = 0, blue: CGFloat = 0, alpha: CGFloat = 0
        color.getRed(&red, green: &green, blue: &blue, alpha: &alpha)

        // Only enhance colors that are truly problematic:
        // 1. Nearly invisible (very dark) - all RGB < 0.15
        let isNearlyInvisible = red < 0.15 && green < 0.15 && blue < 0.15

        // 2. Pure grey from failed processing - very low saturation AND low brightness
        let maxComponent = max(red, max(green, blue))
        let minComponent = min(red, min(green, blue))
        let isProcessingGrey = (maxComponent - minComponent) < 0.05 && maxComponent < 0.3

        // 3. Completely white/transparent (indicates missing material data)
        let isWhiteOrTransparent = (red > 0.95 && green > 0.95 && blue > 0.95) || alpha < 0.1

        let needsEnhancement = isNearlyInvisible || isProcessingGrey || isWhiteOrTransparent

        if needsEnhancement {
            print("   🔍 Color needs enhancement - RGB(\(red), \(green), \(blue)), Alpha(\(alpha))")
            print("      Nearly invisible: \(isNearlyInvisible), Processing grey: \(isProcessingGrey), White/transparent: \(isWhiteOrTransparent)")
        } else {
            print("   ✅ Color looks good - RGB(\(red), \(green), \(blue)), keeping original")
        }

        return needsEnhancement
    }

    // Smart color enhancement for dark/grey materials from Stable Fast 3D
    private func enhanceColorIfNeeded(_ originalColor: UIColor) -> UIColor {
        // Extract RGB components to check if color is too dark or grey
        var red: CGFloat = 0, green: CGFloat = 0, blue: CGFloat = 0, alpha: CGFloat = 0
        originalColor.getRed(&red, green: &green, blue: &blue, alpha: &alpha)

        // Check if color is too dark (all RGB values < 0.3) or too grey
        let isDark = red < 0.3 && green < 0.3 && blue < 0.3
        let maxComponent = max(red, max(green, blue))
        let minComponent = min(red, min(green, blue))
        let isGrey = (maxComponent - minComponent) < 0.1 && maxComponent < 0.5

        if isDark || isGrey {
            // Define realistic dark grey furniture color palette
            let furnitureColors = [
                UIColor(red: 0.25, green: 0.25, blue: 0.25, alpha: 1.0), // Dark charcoal
                UIColor(red: 0.35, green: 0.35, blue: 0.35, alpha: 1.0), // Medium charcoal
                UIColor(red: 0.3, green: 0.32, blue: 0.33, alpha: 1.0),  // Dark slate grey
                UIColor(red: 0.28, green: 0.3, blue: 0.32, alpha: 1.0),  // Steel grey
                UIColor(red: 0.32, green: 0.31, blue: 0.3, alpha: 1.0)   // Anthracite grey
            ]

            // Use object count to vary colors across different placed objects
            let colorIndex = placedObjects.count % furnitureColors.count
            let enhancedColor = furnitureColors[colorIndex]

            print("   🎨 Enhanced problematic color RGB(\(red), \(green), \(blue)) → dark grey \(enhancedColor)")
            return enhancedColor
        }

        // Color is already good, return original
        return originalColor
    }

    // MARK: - Object Manipulation Methods

    // Handle long press gesture to select objects for manipulation
    func handleLongPress(at screenPoint: CGPoint) -> Bool {
        guard let arView = arView else {
            print("⚠️ No ARView available for hit testing")
            return false
        }

        // Perform hit test to find which object was touched
        if let hitObject = hitTestPlacedObjects(at: screenPoint, in: arView) {
            startObjectManipulation(object: hitObject)
            return true
        }

        print("📍 Long press hit test found no placed objects at point: \(screenPoint)")
        return false
    }

    // Hit test to find which placed object was touched
    private func hitTestPlacedObjects(at screenPoint: CGPoint, in arView: ARView) -> RealityKitPlacedObject? {
        // Convert screen point to world ray for hit testing
        let raycastResults = arView.raycast(from: screenPoint, allowing: .existingPlaneGeometry, alignment: .any)

        // Check each placed object to see if the ray intersects with it
        for placedObject in placedObjects {
            if isEntityHit(entity: placedObject.entity, screenPoint: screenPoint, arView: arView) {
                print("🎯 Hit test found placed object at: \(placedObject.entity.position)")
                return placedObject
            }
        }

        return nil
    }

    // Check if a specific entity is hit by the screen point
    private func isEntityHit(entity: Entity, screenPoint: CGPoint, arView: ARView) -> Bool {
        // Get entity's world position and bounds
        guard let entityBounds = calculateEntityBounds(entity) else {
            print("⚠️ No bounds available for entity hit testing")
            return false
        }

        let entityPosition = entity.position
        let entityScale = entity.scale

        // Calculate scaled bounds for hit testing
        let scaledBounds = SIMD3<Float>(
            entityBounds.x * entityScale.x,
            entityBounds.y * entityScale.y,
            entityBounds.z * entityScale.z
        )

        // Simple distance-based hit test (more accurate than ray intersection for complex models)
        let cameraTransform = arView.cameraTransform
        let cameraPosition = cameraTransform.translation

        // Calculate approximate hit sphere radius based on object size
        let hitRadius = max(scaledBounds.x, max(scaledBounds.y, scaledBounds.z)) / 2.0

        // Project screen point to world direction
        let worldDirection = screenPointToWorldDirection(screenPoint, in: arView)

        // Calculate closest point on camera ray to entity center
        let toEntity = entityPosition - cameraPosition
        let projectionLength = dot(toEntity, worldDirection)

        // Ensure projection is in front of camera
        if projectionLength > 0 {
            let closestPoint = cameraPosition + worldDirection * projectionLength
            let distanceToEntity = simd_length(closestPoint - entityPosition)

            print("🎯 Hit test: entity at \(entityPosition), hit distance: \(distanceToEntity), radius: \(hitRadius)")

            // Check if hit point is within entity bounds
            return distanceToEntity <= hitRadius
        }

        return false
    }

    // Start object manipulation mode
    private func startObjectManipulation(object: RealityKitPlacedObject) {
        selectedObject = object
        isManipulatingObject = true
        onManipulationStart?()

        print("🎯 Started manipulating object at position: \(object.entity.position)")
        print("✋ Object manipulation mode active - camera controls disabled")
    }

    // End object manipulation mode
    func endObjectManipulation() {
        selectedObject = nil
        isManipulatingObject = false
        onManipulationEnd?()

        print("✅ Ended object manipulation mode - camera controls re-enabled")
    }

    // Handle pan gesture for object rotation during manipulation mode
    func handleObjectRotation(translation: CGPoint) {
        guard let selectedObject = selectedObject else {
            print("⚠️ No object selected for rotation")
            return
        }

        // Convert horizontal translation to rotation angle
        let rotationSensitivity: Float = 0.01
        let rotationAngle = Float(translation.x) * rotationSensitivity

        // Apply rotation around Y-axis (vertical axis) to the selected object
        let currentRotation = selectedObject.entity.transform.rotation
        let yAxisRotation = simd_quatf(angle: rotationAngle, axis: SIMD3<Float>(0, 1, 0))
        let newRotation = yAxisRotation * currentRotation

        // Update the object's rotation
        selectedObject.entity.transform.rotation = newRotation

        print("🔄 Rotating object by \(rotationAngle) radians around Y-axis")
    }

    // The next methods continue with the proper implementation...
    // (Note: All duplicate methods have been removed to resolve compilation errors)
}

// MARK: - RealityKit Placed Object Model
struct RealityKitPlacedObject: Identifiable {
    let id: UUID
    let entity: Entity
    let anchorEntity: AnchorEntity
    let originalImage: UIImage?

    var position: SIMD3<Float> {
        return entity.position
    }
}

// MARK: - Compatibility Aliases
// These aliases ensure compatibility with existing code that expects the old SceneKit types
typealias ARObjectPlacementManager = RealityKitObjectPlacementManager
typealias ARPlacedObject = RealityKitPlacedObject
typealias CameraMovementManager = RealityKitCameraMovementManager
