import SceneKit

class BoundaryManager {
    weak var scnView: SCNView?
    private var roomBounds: RoomBounds?
    
    struct RoomBounds {
        let minX: Float
        let maxX: Float
        let minY: Float
        let maxY: Float
        let minZ: Float
        let maxZ: Float
        
        func contains(position: SCNVector3) -> Bool {
            return position.x >= minX && position.x <= maxX &&
                   position.y >= minY && position.y <= maxY &&
                   position.z >= minZ && position.z <= maxZ
        }
    }
    
    init(scnView: SCNView) {
        self.scnView = scnView
    }
    
    func calculateRoomBounds(from scene: SCNScene) {
        var minX: Float = Float.greatestFiniteMagnitude
        var maxX: Float = -Float.greatestFiniteMagnitude
        var minY: Float = Float.greatestFiniteMagnitude
        var maxY: Float = -Float.greatestFiniteMagnitude
        var minZ: Float = Float.greatestFiniteMagnitude
        var maxZ: Float = -Float.greatestFiniteMagnitude
        
        scene.rootNode.enumerateChildNodes { node, _ in
            if let geometry = node.geometry {
                let (localMin, localMax) = geometry.boundingBox
                let worldTransform = node.worldTransform
                
                let corners = [
                    SCNVector3(localMin.x, localMin.y, localMin.z),
                    SCNVector3(localMax.x, localMin.y, localMin.z),
                    SCNVector3(localMin.x, localMax.y, localMin.z),
                    SCNVector3(localMax.x, localMax.y, localMin.z),
                    SCNVector3(localMin.x, localMin.y, localMax.z),
                    SCNVector3(localMax.x, localMin.y, localMax.z),
                    SCNVector3(localMin.x, localMax.y, localMax.z),
                    SCNVector3(localMax.x, localMax.y, localMax.z)
                ]
                
                for corner in corners {
                    let worldCorner = worldTransform * corner
                    minX = min(minX, worldCorner.x)
                    maxX = max(maxX, worldCorner.x)
                    minY = min(minY, worldCorner.y)
                    maxY = max(maxY, worldCorner.y)
                    minZ = min(minZ, worldCorner.z)
                    maxZ = max(maxZ, worldCorner.z)
                }
            }
        }
        
        let padding: Float = 0.5
        roomBounds = RoomBounds(
            minX: minX + padding,
            maxX: maxX - padding,
            minY: minY + 0.2,
            maxY: maxY - 0.2,
            minZ: minZ + padding,
            maxZ: maxZ - padding
        )
        
        // Debug logging to verify boundary calculation
        print("🏠 Room bounds calculated: X(\(minX + padding) to \(maxX - padding)), Z(\(minZ + padding) to \(maxZ - padding))")
    }
    
    func constrainCameraPosition(_ position: SCNVector3) -> SCNVector3 {
        guard let bounds = roomBounds else { return position }
        
        let constrainedX = max(bounds.minX, min(bounds.maxX, position.x))
        let constrainedY = max(bounds.minY, min(bounds.maxY, position.y))
        let constrainedZ = max(bounds.minZ, min(bounds.maxZ, position.z))
        
        return SCNVector3(constrainedX, constrainedY, constrainedZ)
    }
    
    func isPositionValid(_ position: SCNVector3) -> Bool {
        guard let bounds = roomBounds else { return true }
        return bounds.contains(position: position)
    }
}

