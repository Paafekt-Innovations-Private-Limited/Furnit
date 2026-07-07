import RealityKit
import simd

// RealityKit-based boundary manager to replace SceneKit boundary manager
class RealityKitBoundaryManager {
    weak var arView: ARView?
    
    // Room boundary properties
    private var roomBounds: (min: SIMD3<Float>, max: SIMD3<Float>)?
    /// Padding from walls when constraining camera (allow navigating close to walls; was 0.5)
    private let boundaryPadding: Float = 0.15
    
    /// Match Android RoomBoundaryManager.CAMERA_PADDING
    private let cameraPadding: Float = 0.05
    
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
                logDebug("ℹ️ [BoundaryManager] Previous bounds exist. Accepting new bounds regardless of size ratio (Option A). Size ratio: \(sizeRatio)")
            }
            
            roomBounds = newBounds
            logDebug("🏠 Room bounds calculated: min(\(minBounds)), max(\(maxBounds))")
            logDebug("   Room dimensions: \(newRoomSize.x) x \(newRoomSize.y) x \(newRoomSize.z)")
            
            // Log if this is an update vs initial calculation
            if roomBounds != nil {
                logDebug("   ✅ Bounds validated and accepted")
            }
        } else {
            logDebug("⚠️ No geometry found for boundary calculation")
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
            logDebug("🔍 Entity bounds: \(entity.name)")
            logDebug("   Local bounds: min(\(bounds.min)), max(\(bounds.max))")
            logDebug("   Transformed: min(\(SIMD3<Float>(minBounds.x, minBounds.y, minBounds.z))), max(\(SIMD3<Float>(maxBounds.x, maxBounds.y, maxBounds.z)))")
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
        let depthZ = bounds.max.z - bounds.min.z
        let planeZ = (bounds.min.z + bounds.max.z) * 0.5

        // Zero-thickness flat photo plane (Depth Anything --flat-mesh): keep camera on the −Z
        // photographer side; the default min/max Z clamp breaks when min.z ≈ max.z.
        if depthZ < boundaryPadding * 2 {
            let minConstraint = bounds.min + SIMD3<Float>(boundaryPadding, 0, boundaryPadding)
            let maxConstraint = bounds.max - SIMD3<Float>(boundaryPadding, 0, boundaryPadding)
            constrainedPosition.x = max(minConstraint.x, min(maxConstraint.x, position.x))
            constrainedPosition.y = max(bounds.min.y + 0.5, min(bounds.max.y + 2.0, position.y))
            let maxCameraZ = planeZ - 0.05
            if position.z > maxCameraZ {
                constrainedPosition.z = maxCameraZ
            } else {
                constrainedPosition.z = position.z
            }
            return constrainedPosition
        }

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
            logDebug("🚧 Camera position constrained: \(position) -> \(constrainedPosition)")
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
    
    /// Depth-adaptive inset fraction (matches Android RoomBoundaryManager.backCenterInsetFraction).
    /// Shallow rooms: smaller fraction (camera stays near back). Deep rooms: larger fraction (camera further in).
    private func backCenterInsetFraction(depth: Float) -> Float {
        let t = min(1.0, max(0.0, depth / 6.0))
        return 0.035 + 0.065 * t  // matches RoomBounds (stay near back wall)
    }
    
    /// Camera at back CENTER with depth-adaptive inset (matches Android when room opened from list / room created).
    /// One formula works for shallow and deep rooms.
    func getCameraAtBackCenter() -> (position: SIMD3<Float>, lookAt: SIMD3<Float>) {
        let debugMode = AppStateManager.shared.qualitySettings.debugMode
        
        guard let bounds = roomBounds else {
            let defaultPosition = SIMD3<Float>(0, 1.5, 3)
            let defaultLookAt = SIMD3<Float>(0, 1.4, 0)
            return (position: defaultPosition, lookAt: defaultLookAt)
        }
        
        let roomCenter = getRoomCenter()
        let depth = bounds.max.z - bounds.min.z  // backWallZ - frontWallZ
        let fraction = backCenterInsetFraction(depth: depth)
        let insetFromBack = max(depth * fraction, cameraPadding)
        
        let camX = roomCenter.x
        let camY = roomCenter.y + 0.4
        let camZ = bounds.max.z - insetFromBack  // back wall, pushed into room
        
        let targetX = roomCenter.x
        let targetY = roomCenter.y
        let targetZ = bounds.min.z  // front wall (where photo is)
        
        if debugMode {
            logDebug("🎯 [BoundaryManager] getCameraAtBackCenter depth=\(depth) fraction=\(fraction) inset=\(insetFromBack) pos=(\(camX),\(camY),\(camZ)) lookAt=(\(targetX),\(targetY),\(targetZ))")
        }
        
        return (
            position: SIMD3<Float>(camX, camY, camZ),
            lookAt: SIMD3<Float>(targetX, targetY, targetZ)
        )
    }

    private func depthAnythingViewportAspect(photoOrientation: PhotoOrientation) -> Float {
        if let arView,
           arView.bounds.width > 1,
           arView.bounds.height > 1 {
            return Float(arView.bounds.width / arView.bounds.height)
        }
        return photoOrientation == .landscape ? (19.5 / 9.0) : (9.0 / 19.5)
    }

    private func depthAnythingImagePlaneStandoff(
        width: Float,
        height: Float,
        span: Float,
        photoOrientation: PhotoOrientation
    ) -> Float {
        let halfFovRadians = Float.pi / 6.0
        let viewportAspect = max(depthAnythingViewportAspect(photoOrientation: photoOrientation), 1.0)

        if photoOrientation == .landscape {
            // Match RealityKit horizontal FOV in landscape: fit the full photo plane in view.
            let fitWidth = width / (2 * tan(halfFovRadians))
            let verticalHalfFov = atan(tan(halfFovRadians) / viewportAspect)
            let fitHeight = height / (2 * tan(verticalHalfFov))
            return max(max(fitWidth, fitHeight) * 0.86, 0.85)
        }

        let fitHeight = height / (2 * tan(halfFovRadians))
        let horizontalHalfFov = atan(tan(halfFovRadians) * viewportAspect)
        let fitWidth = width / (2 * tan(horizontalHalfFov))
        return max(max(span / (2 * tan(halfFovRadians)), fitHeight, fitWidth, span * 0.5), 1.0)
    }

    /// Photographer viewpoint for Depth Anything `--flat-mesh` USDZ: plane at z≈0, camera on −Z.
    func getCameraForDepthAnythingImagePlane(photoOrientation: PhotoOrientation = .portrait) -> (position: SIMD3<Float>, lookAt: SIMD3<Float>) {
        let debugMode = AppStateManager.shared.qualitySettings.debugMode

        guard let bounds = roomBounds else {
            return (SIMD3(0, 0, -2.5), SIMD3(0, 0, 0))
        }

        let center = getRoomCenter()
        let width = max(bounds.max.x - bounds.min.x, 0.1)
        let height = max(bounds.max.y - bounds.min.y, 0.1)
        let span = max(width, height)
        let planeZ = (bounds.min.z + bounds.max.z) * 0.5
        let standoff = depthAnythingImagePlaneStandoff(
            width: width,
            height: height,
            span: span,
            photoOrientation: photoOrientation
        )

        let position = SIMD3<Float>(center.x, center.y, planeZ - standoff)
        let lookAt = SIMD3<Float>(center.x, center.y, planeZ)

        if debugMode {
            let viewportAspect = depthAnythingViewportAspect(photoOrientation: photoOrientation)
            logDebug(
                "🎯 [BoundaryManager] getCameraForDepthAnythingImagePlane "
                    + "orientation=\(photoOrientation.rawValue) width=\(width)m height=\(height)m "
                    + "viewportAspect=\(String(format: "%.2f", viewportAspect)) "
                    + "span=\(span)m standoff=\(standoff)m planeZ=\(planeZ) "
                    + "pos=\(position) lookAt=\(lookAt)"
            )
        }

        return (position, lookAt)
    }

    /// Camera in front of image-depth meshes where near geometry is negative Z and the far wall is max Z.
    func getCameraAtFrontCenter() -> (position: SIMD3<Float>, lookAt: SIMD3<Float>) {
        let debugMode = AppStateManager.shared.qualitySettings.debugMode

        guard let bounds = roomBounds else {
            let defaultPosition = SIMD3<Float>(0, 1.5, -3)
            let defaultLookAt = SIMD3<Float>(0, 1.4, 0)
            return (position: defaultPosition, lookAt: defaultLookAt)
        }

        let roomCenter = getRoomCenter()
        let depth = max(bounds.max.z - bounds.min.z, 0.1)
        let frontOffset = max(depth * 0.08, 0.35)

        let camX = roomCenter.x
        let camY = roomCenter.y + 0.4
        let camZ = bounds.min.z - frontOffset

        let targetX = roomCenter.x
        let targetY = roomCenter.y
        let targetZ = bounds.max.z

        if debugMode {
            logDebug("🎯 [BoundaryManager] getCameraAtFrontCenter depth=\(depth) offset=\(frontOffset) pos=(\(camX),\(camY),\(camZ)) lookAt=(\(targetX),\(targetY),\(targetZ))")
        }

        return (
            position: SIMD3<Float>(camX, camY, camZ),
            lookAt: SIMD3<Float>(targetX, targetY, targetZ)
        )
    }
    
    // ✅ Get optimal camera position for viewing the room (delegates to Android-matching back-center formula)
    // Used when room is opened from list or when room is created.
    func getOptimalCameraPosition(
        roomCoordinateFrame: RoomCoordinateFrame = .sharpCanonicalPly,
        photoOrientation: PhotoOrientation = .portrait
    ) -> (position: SIMD3<Float>, lookAt: SIMD3<Float>) {
        if roomCoordinateFrame == .depthAnythingImageDepthMeters {
            return getCameraForDepthAnythingImagePlane(photoOrientation: photoOrientation)
        }
        if roomCoordinateFrame.usesFrontFacingRealityKitCamera {
            return getCameraAtFrontCenter()
        }
        return getCameraAtBackCenter()
    }
    
    // Reset boundary calculations
    func reset() {
        roomBounds = nil
        logDebug("🔄 Boundary manager reset")
    }
}

// MARK: - Extensions for SIMD operations are defined in RealityKitObjectPlacementManager.swift
