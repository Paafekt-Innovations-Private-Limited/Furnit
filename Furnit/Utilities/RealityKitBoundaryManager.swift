import RealityKit
import simd

// RealityKit-based boundary manager to replace SceneKit boundary manager
class RealityKitBoundaryManager {
    weak var arView: ARView?
    
    // Room boundary properties
    private var roomBounds: (min: SIMD3<Float>, max: SIMD3<Float>)?
    private let boundaryPadding: Float = 0.5 // Padding from walls in meters
    
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
            
            // Validate bounds to prevent dramatic shrinking
            if let existingBounds = roomBounds {
                let existingSize = existingBounds.max - existingBounds.min
                let sizeRatio = (newRoomSize.x * newRoomSize.y * newRoomSize.z) / (existingSize.x * existingSize.y * existingSize.z)
                
                // If new bounds are dramatically smaller (less than 10% of original), reject them
                if sizeRatio < 0.1 {
                    print("🚫 Rejecting dramatically smaller bounds:")
                    print("   Existing: \(existingSize.x) x \(existingSize.y) x \(existingSize.z)")
                    print("   New: \(newRoomSize.x) x \(newRoomSize.y) x \(newRoomSize.z)")
                    print("   Size ratio: \(sizeRatio) - keeping existing bounds")
                    return
                }
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
        
        // Position camera inside room, at reasonable height and distance from center
        let cameraHeight = bounds.min.y + (roomSize.y * 0.4) // 40% up from floor
        let viewingDistance = min(roomSize.x, roomSize.z) * 0.3 // 30% of smaller horizontal dimension
        
        let safePosition = SIMD3<Float>(
            roomCenter.x - viewingDistance,
            cameraHeight,
            roomCenter.z + viewingDistance * 0.5
        )
        
        return constrainCameraPosition(safePosition)
    }
    
    // Reset boundary calculations
    func reset() {
        roomBounds = nil
        print("🔄 Boundary manager reset")
    }
}

// MARK: - Extensions for SIMD operations are defined in RealityKitObjectPlacementManager.swift