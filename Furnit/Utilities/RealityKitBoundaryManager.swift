import RealityKit
import simd

// RealityKit-based boundary manager to replace SceneKit boundary manager
class RealityKitBoundaryManager {
    weak var arView: ARView?
    
    // Room boundary properties
    private var roomBounds: (min: SIMD3<Float>, max: SIMD3<Float>)?
    private let boundaryPadding: Float = 0.5 // Padding from walls in meters
    
    // ✅ NEW: Public accessor for bounds (used by camera positioning)
    var bounds: (min: SIMD3<Float>, max: SIMD3<Float>)? {
        return roomBounds
    }
    
    init(arView: ARView) {
        self.arView = arView
    }
    
    // Calculate room boundaries from the loaded model entity
    func calculateRoomBounds(from modelEntity: Entity) {
        // Find the overall bounding box of all mesh entities
        var minBounds = SIMD3<Float>(repeating: Float.greatestFiniteMagnitude)
        var maxBounds = SIMD3<Float>(repeating: -Float.greatestFiniteMagnitude)
        var hasGeometry = false
        
        // Recursively find all entities with model components
        findBounds(in: modelEntity, minBounds: &minBounds, maxBounds: &maxBounds, hasGeometry: &hasGeometry)
        
        if hasGeometry {
            let newBounds = (min: minBounds, max: maxBounds)
            let newRoomSize = maxBounds - minBounds
            
            // Option A: Always accept new bounds for each model load to avoid stale state
            if let existingBounds = roomBounds {
                let existingSize = existingBounds.max - existingBounds.min
                let sizeRatio = (newRoomSize.x * newRoomSize.y * newRoomSize.z) / (existingSize.x * existingSize.y * existingSize.z)
                print("ℹ️ [BoundaryManager] Previous bounds exist. Accepting new bounds regardless of size ratio (Option A). Size ratio: \(sizeRatio)")
            }
            
            roomBounds = newBounds
            print("🏠 Room bounds calculated: min(\(minBounds)), max(\(maxBounds))")
            print("   Room dimensions: \(newRoomSize.x) x \(newRoomSize.y) x \(newRoomSize.z)")
            
            // Log if this is an update vs initial calculation
            if roomBounds != nil {
                print("   ✅ Bounds validated and accepted")
            }
        } else {
            print("⚠️ No geometry found for boundary calculation")
            // Set default room bounds only if no bounds exist yet
            if roomBounds == nil {
                roomBounds = (
                    min: SIMD3<Float>(-5, 0, -5),
                    max: SIMD3<Float>(5, 3, 5)
                )
            }
        }
    }
    
    // Recursively find bounds from entity hierarchy
    private func findBounds(in entity: Entity, minBounds: inout SIMD3<Float>, maxBounds: inout SIMD3<Float>, hasGeometry: inout Bool) {
        // Check if entity has a model component with bounds
        if let modelComponent = entity.components[ModelComponent.self] {
            let bounds = modelComponent.mesh.bounds
            
            // Get transform relative to world coordinates for consistent bounds
            // We use nil to get world transform which should be consistent
            let worldTransform = entity.transformMatrix(relativeTo: nil)
            
            // Calculate all corners of the bounding box
            let corners = [
                SIMD3<Float>(bounds.min.x, bounds.min.y, bounds.min.z),
                SIMD3<Float>(bounds.max.x, bounds.min.y, bounds.min.z),
                SIMD3<Float>(bounds.min.x, bounds.max.y, bounds.min.z),
                SIMD3<Float>(bounds.max.x, bounds.max.y, bounds.min.z),
                SIMD3<Float>(bounds.min.x, bounds.min.y, bounds.max.z),
                SIMD3<Float>(bounds.max.x, bounds.min.y, bounds.max.z),
                SIMD3<Float>(bounds.min.x, bounds.max.y, bounds.max.z),
                SIMD3<Float>(bounds.max.x, bounds.max.y, bounds.max.z)
            ]
            
            // Transform corners and update bounds
            for corner in corners {
                let transformedCorner = transformPoint(corner, by: worldTransform)
                
                minBounds.x = min(minBounds.x, transformedCorner.x)
                minBounds.y = min(minBounds.y, transformedCorner.y)
                minBounds.z = min(minBounds.z, transformedCorner.z)
                
                maxBounds.x = max(maxBounds.x, transformedCorner.x)
                maxBounds.y = max(maxBounds.y, transformedCorner.y)
                maxBounds.z = max(maxBounds.z, transformedCorner.z)
            }
            
            hasGeometry = true
            
            // Debug log for troubleshooting
            print("🔍 Entity bounds: \(entity.name)")
            print("   Local bounds: min(\(bounds.min)), max(\(bounds.max))")
            print("   Transformed: min(\(SIMD3<Float>(minBounds.x, minBounds.y, minBounds.z))), max(\(SIMD3<Float>(maxBounds.x, maxBounds.y, maxBounds.z)))")
        }
        
        // Recursively check children
        for child in entity.children {
            findBounds(in: child, minBounds: &minBounds, maxBounds: &maxBounds, hasGeometry: &hasGeometry)
        }
    }
    
    // Transform a point by a 4x4 matrix
    private func transformPoint(_ point: SIMD3<Float>, by matrix: float4x4) -> SIMD3<Float> {
        let point4 = SIMD4<Float>(point.x, point.y, point.z, 1.0)
        let transformed = matrix * point4
        return SIMD3<Float>(transformed.x, transformed.y, transformed.z)
    }
    
    // Constrain camera position to stay within room boundaries
    func constrainCameraPosition(_ position: SIMD3<Float>) -> SIMD3<Float> {
        guard let bounds = roomBounds else {
            return position // No constraints if bounds not calculated
        }
        
        var constrainedPosition = position
        
        // Apply padding to create boundaries inside the room
        let minConstraint = bounds.min + SIMD3<Float>(boundaryPadding, 0, boundaryPadding)
        let maxConstraint = bounds.max - SIMD3<Float>(boundaryPadding, 0, boundaryPadding)
        
        // Constrain X position (left-right movement)
        constrainedPosition.x = max(minConstraint.x, min(maxConstraint.x, position.x))
        
        // Constrain Z position (forward-backward movement)
        constrainedPosition.z = max(minConstraint.z, min(maxConstraint.z, position.z))
        
        // Allow Y movement within reasonable limits but don't constrain to room height
        // This allows camera to be positioned above furniture
        let minY = bounds.min.y + 0.5  // At least 0.5m above floor
        let maxY = bounds.max.y + 2.0  // Allow some height above room ceiling
        constrainedPosition.y = max(minY, min(maxY, position.y))
        
        // Debug logging when position is constrained
        if constrainedPosition.x != position.x || constrainedPosition.z != position.z {
            print("🚧 Camera position constrained: \(position) -> \(constrainedPosition)")
        }
        
        return constrainedPosition
    }
    
    // Get room center point for camera targeting
    func getRoomCenter() -> SIMD3<Float> {
        guard let bounds = roomBounds else {
            return SIMD3<Float>(0, 1, 0) // Default center
        }
        
        return (bounds.min + bounds.max) / 2
    }
    
    // Get room dimensions for camera positioning
    func getRoomDimensions() -> SIMD3<Float> {
        guard let bounds = roomBounds else {
            return SIMD3<Float>(5, 3, 5) // Default room size
        }
        
        return bounds.max - bounds.min
    }
    
    // Get floor height for object placement
    func getFloorHeight() -> Float {
        return roomBounds?.min.y ?? 0.0
    }
    
    // Check if a point is within room boundaries
    func isPositionWithinBounds(_ position: SIMD3<Float>) -> Bool {
        guard let bounds = roomBounds else { return true }
        
        return position.x >= bounds.min.x && position.x <= bounds.max.x &&
               position.y >= bounds.min.y && position.y <= bounds.max.y &&
               position.z >= bounds.min.z && position.z <= bounds.max.z
    }
    
    // Get current room bounds for debugging
    func getCurrentBounds() -> (min: SIMD3<Float>, max: SIMD3<Float>)? {
        return roomBounds
    }
    
    // Get safe camera position within bounds
    func getSafeCameraPosition(near targetPosition: SIMD3<Float>) -> SIMD3<Float> {
        guard let bounds = roomBounds else { return targetPosition }
        
        let roomCenter = getRoomCenter()
        let roomSize = getRoomDimensions()
        
        // Position camera inside room at lower height and reasonable distance from center
        let eyeLevelHeight: Float = 1.2 // Lower viewing height in meters
        let cameraHeight = bounds.min.y + eyeLevelHeight // Height from floor
        let viewingDistance = min(roomSize.x, roomSize.z) * 0.3 // 30% of smaller horizontal dimension
        
        let safePosition = SIMD3<Float>(
            roomCenter.x - viewingDistance,
            cameraHeight,
            roomCenter.z + viewingDistance * 0.5
        )
        
        return constrainCameraPosition(safePosition)
    }
    
    // ✅ Get optimal camera position for viewing the room
    // Returns a tuple with camera position and look-at target
    // STRATEGY: Position camera at BACK-LEFT corner for every room
    func getOptimalCameraPosition() -> (position: SIMD3<Float>, lookAt: SIMD3<Float>) {
        print("🎯🎯🎯 [BoundaryManager] === BACK-LEFT CORNER CAMERA CALCULATION ===")

        guard let bounds = roomBounds else {
            print("   ⚠️ NO BOUNDS - using default position")
            let defaultPosition = SIMD3<Float>(0, 1.5, 3)
            let defaultLookAt = SIMD3<Float>(0, 1.4, 0)
            return (position: defaultPosition, lookAt: defaultLookAt)
        }

        let roomSize = getRoomDimensions()
        let roomCenter = getRoomCenter()

        print("   📦 Room bounds:")
        print("      MIN: X=\(bounds.min.x), Y=\(bounds.min.y), Z=\(bounds.min.z)")
        print("      MAX: X=\(bounds.max.x), Y=\(bounds.max.y), Z=\(bounds.max.z)")
        print("   📏 Room size: \(roomSize.x)m x \(roomSize.y)m x \(roomSize.z)m")
        print("   🎯 Room center: X=\(roomCenter.x), Y=\(roomCenter.y), Z=\(roomCenter.z)")
        print("   🧱 Boundary padding: \(boundaryPadding)m")

        // Camera positioning strategy: INSIDE the room near back wall corner, looking toward front wall
        // Position camera INSIDE room at back wall corner, looking at front wall
        // This gives the feeling of standing in the back corner of the home

        let cameraHeight = roomCenter.y + 0.4  // Raise camera higher - eye level above center
        
        // Position camera near the back wall corner (back-left corner)
        let wallPadding: Float = 0.3  // 30cm from walls for realistic positioning
        let camX = bounds.min.x + wallPadding  // Near left wall
        let camZ = bounds.max.z - wallPadding  // Near back wall
        
        print("   📐 BACK-CORNER positioning (inside room at back wall):")
        print("   📐 Camera X: \(bounds.min.x) + \(wallPadding) = \(camX) (NEAR LEFT WALL)")
        print("   📐 Camera Y: \(roomCenter.y) (CENTER HEIGHT)")
        print("   📐 Camera Z: \(bounds.max.z) - \(wallPadding) = \(camZ) (NEAR BACK WALL)")

        let cameraPosition = SIMD3<Float>(camX, cameraHeight, camZ)

        // Look-at point: Front wall, but slightly toward the center for better view
        let lookX = roomCenter.x  // Center X for balanced view
        let lookY = roomCenter.y  // Center height
        let lookZ = bounds.min.z  // FRONT wall (MIN Z) where photo is
        let lookAtPosition = SIMD3<Float>(lookX, lookY, lookZ)

        print("   📐 Looking at FRONT/PHOTO wall:")
        print("   📐 LookAt X: \(roomCenter.x) (CENTER)")
        print("   📐 LookAt Y: \(roomCenter.y) (CENTER HEIGHT)")
        print("   📐 LookAt Z: \(bounds.min.z) = \(lookZ) (FRONT/PHOTO wall)")

        print("   📷 FINAL CAMERA POSITION:")
        print("      X=\(cameraPosition.x), Y=\(cameraPosition.y), Z=\(cameraPosition.z)")
        print("   👁️ LOOK-AT POSITION:")
        print("      X=\(lookAtPosition.x), Y=\(lookAtPosition.y), Z=\(lookAtPosition.z)")
        print("   ✅ Strategy: BACK-LEFT CORNER (against walls) → looking toward front center")
        print("🎯🎯🎯 [BoundaryManager] === END CALCULATION ===")

        return (position: cameraPosition, lookAt: lookAtPosition)
    }
    
    // Reset boundary calculations
    func reset() {
        roomBounds = nil
        print("🔄 Boundary manager reset")
    }
}

// MARK: - Extensions for SIMD operations are defined in RealityKitObjectPlacementManager.swift

