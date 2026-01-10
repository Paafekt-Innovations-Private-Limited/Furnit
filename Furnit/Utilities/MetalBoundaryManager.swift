import simd

/// Boundary manager for Metal/MetalSplatter rendering
/// Mirrors RealityKitBoundaryManager functionality for Gaussian splat rooms
class MetalBoundaryManager {

    // Room boundary properties
    private var roomBounds: (min: SIMD3<Float>, max: SIMD3<Float>)?
    private let boundaryPadding: Float = 0.3  // 30cm from walls (same as RealityKit)
    private let minDepth: Float = 5.0  // Minimum room depth for proper camera placement

    // Public accessor for bounds
    var bounds: (min: SIMD3<Float>, max: SIMD3<Float>)? {
        return roomBounds
    }

    // MARK: - Initialization

    init() {}

    // MARK: - Bounds Calculation

    /// Calculate and store room bounds from MetalSplatter bounding box
    /// Ensures minimum depth of 5 for proper camera placement
    func calculateRoomBounds(from boundingBox: (min: SIMD3<Float>, max: SIMD3<Float>)) {
        var adjustedMax = boundingBox.max
        let depth = boundingBox.max.z - boundingBox.min.z

        // Ensure minimum depth
        if depth < minDepth {
            adjustedMax.z = boundingBox.min.z + minDepth
            logDebug("📐 [MetalBoundaryManager] Depth adjusted from \(depth) to \(minDepth)")
        }

        roomBounds = (boundingBox.min, adjustedMax)

        let size = adjustedMax - boundingBox.min
        logDebug("🏠 [MetalBoundaryManager] Room bounds set:")
        logDebug("   Min: \(boundingBox.min)")
        logDebug("   Max: \(adjustedMax)")
        logDebug("   Size: \(size.x) × \(size.y) × \(size.z)")
    }

    // MARK: - Camera Positioning

    /// Get optimal camera position for viewing the room
    /// Strategy: Position camera at BACK-LEFT corner, looking at FRONT wall center
    /// (Same as RealityKitBoundaryManager.getOptimalCameraPosition)
    func getOptimalCameraPosition() -> (position: SIMD3<Float>, lookAt: SIMD3<Float>) {
        guard let bounds = roomBounds else {
            logDebug("⚠️ [MetalBoundaryManager] No bounds - using default position")
            return (
                position: SIMD3<Float>(0, 1.5, 3),
                lookAt: SIMD3<Float>(0, 1.4, 0)
            )
        }

        let roomCenter = getRoomCenter()

        // Camera position: back-left corner inside the room
        let cameraPosition = SIMD3<Float>(
            bounds.min.x + boundaryPadding,    // Near left wall
            roomCenter.y + 0.4,                 // Eye level (above center)
            bounds.max.z - boundaryPadding      // Near back wall
        )

        // Look at front wall center
        let lookAtPosition = SIMD3<Float>(
            roomCenter.x,       // Center X
            roomCenter.y,       // Center Y
            bounds.min.z        // Front wall
        )

        logDebug("📷 [MetalBoundaryManager] Camera position: \(cameraPosition)")
        logDebug("🎯 [MetalBoundaryManager] Looking at: \(lookAtPosition)")

        return (position: cameraPosition, lookAt: lookAtPosition)
    }

    // MARK: - Room Properties

    /// Get room center point
    func getRoomCenter() -> SIMD3<Float> {
        guard let bounds = roomBounds else {
            return SIMD3<Float>(0, 1, 0)
        }
        return (bounds.min + bounds.max) / 2
    }

    /// Get room dimensions
    func getRoomDimensions() -> SIMD3<Float> {
        guard let bounds = roomBounds else {
            return SIMD3<Float>(5, 3, 5)
        }
        return bounds.max - bounds.min
    }

    /// Get floor height
    func getFloorHeight() -> Float {
        return roomBounds?.min.y ?? 0.0
    }

    // MARK: - Boundary Constraints

    /// Constrain camera position to stay within room boundaries
    func constrainCameraPosition(_ position: SIMD3<Float>) -> SIMD3<Float> {
        guard let bounds = roomBounds else {
            return position
        }

        var constrained = position

        // Apply padding to create boundaries inside the room
        let minConstraint = bounds.min + SIMD3<Float>(boundaryPadding, 0, boundaryPadding)
        let maxConstraint = bounds.max - SIMD3<Float>(boundaryPadding, 0, boundaryPadding)

        // Constrain X (left-right)
        constrained.x = max(minConstraint.x, min(maxConstraint.x, position.x))

        // Constrain Z (forward-backward)
        constrained.z = max(minConstraint.z, min(maxConstraint.z, position.z))

        // Constrain Y (vertical) - allow some room above/below
        let minY = bounds.min.y + 0.5
        let maxY = bounds.max.y + 2.0
        constrained.y = max(minY, min(maxY, position.y))

        return constrained
    }

    /// Check if position is within room boundaries
    func isPositionWithinBounds(_ position: SIMD3<Float>) -> Bool {
        guard let bounds = roomBounds else { return true }

        return position.x >= bounds.min.x && position.x <= bounds.max.x &&
               position.y >= bounds.min.y && position.y <= bounds.max.y &&
               position.z >= bounds.min.z && position.z <= bounds.max.z
    }

    // MARK: - Reset

    func reset() {
        roomBounds = nil
        logDebug("🔄 [MetalBoundaryManager] Reset")
    }
}
