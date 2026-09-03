import RealityKit
import SceneKit
import simd
import UIKit

/// One-finger look contract shared by the immediate preview and the reconstructed photo room.
enum DepthAnythingPhotoCameraInteraction {
    static let dragRotationRadiansPerPoint: Float = 0.005
    static let maximumYawRadians: Float = .pi / 6
    static let maximumPitchRadians: Float = .pi / 5
    static let dPadHorizontalStep: Float = 0.24
    static let dPadVerticalStep: Float = 0.20
    static let positionTranslationMetersPerPoint: Float = 0.005
    static let pinchDollyMetersPerScaleDelta: Float = 2.5

    static func rotationDelta(for translation: CGPoint) -> (yaw: Float, pitch: Float) {
        (
            yaw: -Float(translation.x) * dragRotationRadiansPerPoint,
            pitch: -Float(translation.y) * dragRotationRadiansPerPoint
        )
    }

    static func clampedPreviewRotation(yaw: Float, pitch: Float) -> (yaw: Float, pitch: Float) {
        (
            yaw: max(-maximumYawRadians, min(maximumYawRadians, yaw)),
            pitch: max(-maximumPitchRadians, min(maximumPitchRadians, pitch))
        )
    }

    /// The saved-room D-pad is expressed as world-space movement notifications. Projective photo
    /// rooms reinterpret those inputs as bounded look steps; the immediate preview does the same.
    static func rotationNudge(forWorldDelta delta: SIMD3<Float>) -> (yaw: Float, pitch: Float) {
        (
            yaw: -delta.x * 0.4,
            pitch: delta.y * 0.48
        )
    }

    /// Before depth exists, use the same envelope formula as the saved projective room with the
    /// preview's placeholder representative depth. First-save reconstruction replaces these
    /// provisional limits with limits derived from the inferred depth distribution.
    static func movementEnvelope(
        representativeDepth: Float
    ) -> (forward: Float, lateral: Float, backward: Float, vertical: Float) {
        let safeDepth = max(representativeDepth, 0.2)
        let forward = min(max(safeDepth * 0.35, 0.75), 1.40)
        return (
            forward: forward,
            lateral: min(max(safeDepth * 0.12, 0.24), 0.48),
            backward: min(max(safeDepth * 0.08, 0.18), 0.32),
            vertical: min(max(forward * 0.18, 0.10), 0.24)
        )
    }

    static func constrainedPreviewPosition(
        _ position: SIMD3<Float>,
        representativeDepth: Float
    ) -> SIMD3<Float> {
        let envelope = movementEnvelope(representativeDepth: representativeDepth)
        return SIMD3<Float>(
            min(max(position.x, -envelope.lateral), envelope.lateral),
            min(max(position.y, -envelope.vertical), envelope.vertical),
            min(max(position.z, -envelope.forward), envelope.backward)
        )
    }
}

/// Shared cover camera math for Depth Anything flat-photo rooms (RealityKit list viewer + SceneKit preview).
enum DepthAnythingFlatPhotoCameraFraming {
    static func viewportAspect(
        photoOrientation: PhotoOrientation,
        viewportSize: CGSize? = nil
    ) -> Float {
        let lockedLandscapeAspect = Float(19.5 / 9.0)
        let lockedPortraitAspect = Float(9.0 / 19.5)
        if let viewportSize,
           viewportSize.width > 1,
           viewportSize.height > 1 {
            let live = Float(viewportSize.width / viewportSize.height)
            switch photoOrientation {
            case .landscape where live < 1.05:
                return lockedLandscapeAspect
            case .portrait where live > 0.95:
                return lockedPortraitAspect
            default:
                return live
            }
        }
        return photoOrientation == .landscape ? lockedLandscapeAspect : lockedPortraitAspect
    }

    static func imagePlaneStandoff(
        planeWidthMeters: Float,
        planeHeightMeters: Float,
        photoOrientation: PhotoOrientation,
        viewportAspect: Float,
        zoom: CGFloat = 1
    ) -> Float {
        let width = max(planeWidthMeters, 0.05)
        let height = max(planeHeightMeters, 0.05)
        let halfFovRadians = Float.pi / 6.0
        let aspect = max(viewportAspect, 0.01)

        let fitWidth: Float
        let fitHeight: Float
        if photoOrientation == .landscape {
            fitWidth = width / (2 * tan(halfFovRadians))
            let verticalHalfFov = atan(tan(halfFovRadians) / aspect)
            fitHeight = height / (2 * tan(verticalHalfFov))
        } else {
            fitHeight = height / (2 * tan(halfFovRadians))
            let horizontalHalfFov = atan(tan(halfFovRadians) * aspect)
            fitWidth = width / (2 * tan(horizontalHalfFov))
        }

        // Cover framing for both photo orientations — fills the viewport (portrait contain left vertical letterboxing).
        let fitDistance = min(fitWidth, fitHeight) * 0.98
        let clampedZoom = Float(min(max(zoom, 0.55), 4.0))
        // Cover can legitimately place the camera well under 0.85 m (e.g. landscape photo on a portrait phone).
        // A high floor forced letterboxing by keeping the camera too far from the plane.
        return max(fitDistance, 0.2) / clampedZoom
    }

    static func verticalFieldOfViewDegrees(
        photoOrientation: PhotoOrientation,
        viewportAspect: Float
    ) -> CGFloat {
        let halfFovRadians = CGFloat.pi / 6.0
        if photoOrientation == .landscape {
            let verticalHalfFov = atan(tan(halfFovRadians) / CGFloat(max(viewportAspect, 0.01)))
            return verticalHalfFov * 2 * 180 / .pi
        }
        return 60
    }

    /// SceneKit `SCNPlane` front face is +Z; preview camera sits on +Z. Standoff matches list viewer (`getCameraForDepthAnythingImagePlane`).
    static func sceneKitPreviewCameraPose(
        planeWidthMeters: Float,
        planeHeightMeters: Float,
        photoOrientation: PhotoOrientation,
        viewportSize: CGSize,
        cameraOffset: CGSize,
        cameraZoom: CGFloat,
        planeZ: Float = 0
    ) -> (position: SCNVector3, lookAt: SCNVector3, verticalFieldOfViewDegrees: CGFloat) {
        let viewportAspect = viewportAspect(photoOrientation: photoOrientation, viewportSize: viewportSize)
        let standoff = imagePlaneStandoff(
            planeWidthMeters: planeWidthMeters,
            planeHeightMeters: planeHeightMeters,
            photoOrientation: photoOrientation,
            viewportAspect: viewportAspect,
            zoom: cameraZoom
        )
        let verticalFOV = verticalFieldOfViewDegrees(
            photoOrientation: photoOrientation,
            viewportAspect: viewportAspect
        )
        let panUnit = CGFloat(max(planeWidthMeters, planeHeightMeters, 0.05)) * 0.09
        let centerX = Float(cameraOffset.width * panUnit)
        let centerY = Float(cameraOffset.height * panUnit)
        let lookAt = SCNVector3(centerX, centerY, planeZ)
        let position = SCNVector3(centerX, centerY, planeZ + standoff)
        return (position, lookAt, verticalFOV)
    }

    /// Match `RealityKitView.configureDepthAnythingCameraFieldOfView`: preserve the authored
    /// capture projection, then crop only the axis required to aspect-fill the live viewport.
    static func projectiveDisplayVerticalFieldOfViewDegrees(
        projection: DepthAnythingProjectionCamera,
        photoOrientation: PhotoOrientation,
        viewportSize: CGSize
    ) -> CGFloat {
        projectiveDisplayVerticalFieldOfViewDegrees(
            authoredVerticalFieldOfViewDegrees: projection.verticalFieldOfViewDegrees,
            imageWidth: projection.imageWidth,
            imageHeight: projection.imageHeight,
            photoOrientation: photoOrientation,
            viewportSize: viewportSize
        )
    }

    static func projectiveDisplayVerticalFieldOfViewDegrees(
        authoredVerticalFieldOfViewDegrees: Float,
        imageWidth: Int,
        imageHeight: Int,
        photoOrientation: PhotoOrientation,
        viewportSize: CGSize
    ) -> CGFloat {
        let sourceVerticalHalfFOV = CGFloat(authoredVerticalFieldOfViewDegrees) * .pi / 360
        let imageAspect = CGFloat(imageWidth) / CGFloat(max(imageHeight, 1))
        let sourceHorizontalHalfFOV = atan(tan(sourceVerticalHalfFOV) * imageAspect)
        let liveViewportAspect = CGFloat(
            viewportAspect(photoOrientation: photoOrientation, viewportSize: viewportSize)
        )
        let horizontalCoverVerticalHalfFOV = atan(
            tan(sourceHorizontalHalfFOV) / max(liveViewportAspect, 0.01)
        )
        return 2 * min(sourceVerticalHalfFOV, horizontalCoverVerticalHalfFOV) * 180 / .pi
    }

    /// SceneKit equivalent of the saved room's capture camera: eye at the authored optical origin,
    /// forward along -Z, with the preview proxy sized from the same pixel focal lengths.
    static func sceneKitProjectivePreviewCameraPose(
        projection: DepthAnythingProjectionCamera,
        photoOrientation: PhotoOrientation,
        viewportSize: CGSize,
        cameraYaw: Float,
        cameraPitch: Float,
        cameraPosition: SIMD3<Float>
    ) -> (position: SIMD3<Float>, rotation: simd_quatf, verticalFieldOfViewDegrees: CGFloat) {
        let yawRotation = simd_quatf(angle: cameraYaw, axis: SIMD3<Float>(0, 1, 0))
        let pitchRotation = simd_quatf(angle: cameraPitch, axis: SIMD3<Float>(1, 0, 0))
        let verticalFOV = projectiveDisplayVerticalFieldOfViewDegrees(
            projection: projection,
            photoOrientation: photoOrientation,
            viewportSize: viewportSize
        )
        return (cameraPosition, yawRotation * pitchRotation, verticalFOV)
    }
}

// RealityKit-based boundary manager to replace SceneKit boundary manager
class RealityKitBoundaryManager {
    weak var arView: ARView?
    
    // Room boundary properties
    private var roomBounds: (min: SIMD3<Float>, max: SIMD3<Float>)?
    private(set) var usesCapturedPhotoFrustum = false
    private var completedPhotoCameraBounds: (min: SIMD3<Float>, max: SIMD3<Float>)?
    /// Padding from walls when constraining camera (allow navigating close to walls; was 0.5)
    private let boundaryPadding: Float = 0.15
    
    /// Match Android RoomBoundaryManager.CAMERA_PADDING
    private let cameraPadding: Float = 0.05
    
    // ✅ NEW: Public accessor for bounds (used by camera positioning)
    var bounds: (min: SIMD3<Float>, max: SIMD3<Float>)? {
        return roomBounds
    }

    /// True only for geometry with a real three-dimensional interior. Depth Anything's immediate
    /// photo preview is a near-zero-depth plane and must retain exterior inspection controls.
    var hasNavigableInterior: Bool {
        if completedPhotoCameraBounds != nil { return true }
        guard let bounds = roomBounds else { return false }
        let size = bounds.max - bounds.min
        return !usesCapturedPhotoFrustum &&
            size.x > boundaryPadding * 2 &&
            size.y > boundaryPadding * 2 &&
            size.z > boundaryPadding * 2
    }

    func setUsesCapturedPhotoFrustum(_ enabled: Bool) {
        usesCapturedPhotoFrustum = enabled
    }

    /// The capture eye is the rear edge of the completed-photo volume. As on Android/glTF, negative
    /// Z points forward into the authored room, so the useful allowance is intentionally one-way.
    func setCompletedPhotoCameraEnvelope(
        forwardTranslation: Float?,
        lateralTranslation: Float?,
        backwardTranslation: Float?
    ) {
        guard let forwardTranslation,
              forwardTranslation.isFinite,
              forwardTranslation > 0 else {
            completedPhotoCameraBounds = nil
            return
        }
        // Match Android's version-5 capture-eye envelope exactly.
        let forward = min(max(forwardTranslation, 0.75), 1.40)
        let lateral = min(max(lateralTranslation ?? forward * 0.34, 0.24), 0.48)
        let backward = min(max(backwardTranslation ?? forward * 0.25, 0.18), 0.32)
        let vertical = min(max(forward * 0.18, 0.10), 0.24)
        completedPhotoCameraBounds = (
            min: SIMD3<Float>(-lateral, -vertical, -forward),
            max: SIMD3<Float>(lateral, vertical, backward)
        )
    }

    func isInsideNavigableInterior(_ position: SIMD3<Float>) -> Bool {
        hasNavigableInterior && isPositionWithinBounds(position)
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
        if let cameraBounds = completedPhotoCameraBounds {
            return SIMD3<Float>(
                min(max(position.x, cameraBounds.min.x), cameraBounds.max.x),
                min(max(position.y, cameraBounds.min.y), cameraBounds.max.y),
                min(max(position.z, cameraBounds.min.z), cameraBounds.max.z)
            )
        }
        guard let bounds = roomBounds else {
            return position // No constraints if bounds not calculated
        }

        var constrainedPosition = position
        let depthZ = bounds.max.z - bounds.min.z
        let planeZ = (bounds.min.z + bounds.max.z) * 0.5

        // Zero-thickness flat photo plane (Depth Anything --flat-mesh): keep camera on the −Z
        // photographer side; the default min/max Z clamp breaks when min.z ≈ max.z.
        if usesCapturedPhotoFrustum || depthZ < boundaryPadding * 2 {
            let minConstraint = bounds.min + SIMD3<Float>(boundaryPadding, 0, boundaryPadding)
            let maxConstraint = bounds.max - SIMD3<Float>(boundaryPadding, 0, boundaryPadding)
            constrainedPosition.x = max(minConstraint.x, min(maxConstraint.x, position.x))
            constrainedPosition.y = max(bounds.min.y + 0.5, min(bounds.max.y + 2.0, position.y))
            let maxCameraZ = (usesCapturedPhotoFrustum ? bounds.min.z : planeZ) - 0.05
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
        if let cameraBounds = completedPhotoCameraBounds {
            return position.x >= cameraBounds.min.x && position.x <= cameraBounds.max.x &&
                position.y >= cameraBounds.min.y && position.y <= cameraBounds.max.y &&
                position.z >= cameraBounds.min.z && position.z <= cameraBounds.max.z
        }
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
    
    /// Camera at back center, kept just inside the authored wall so an opaque back wall cannot
    /// obstruct the view. The viewport-aware field-of-view fit determines the visible room span.
    func getCameraAtBackCenter() -> (position: SIMD3<Float>, lookAt: SIMD3<Float>) {
        let debugMode = AppStateManager.shared.qualitySettings.debugMode
        
        guard let bounds = roomBounds else {
            let defaultPosition = SIMD3<Float>(0, 1.5, 3)
            let defaultLookAt = SIMD3<Float>(0, 1.4, 0)
            return (position: defaultPosition, lookAt: defaultLookAt)
        }
        
        let roomCenter = getRoomCenter()
        let depth = bounds.max.z - bounds.min.z  // backWallZ - frontWallZ
        let insetFromBack = min(cameraPadding, max(depth * 0.1, 0.001))
        
        let camX = roomCenter.x
        let camY = roomCenter.y + 0.4
        let camZ = bounds.max.z - insetFromBack  // back wall, pushed into room
        
        let targetX = roomCenter.x
        let targetY = roomCenter.y
        let targetZ = bounds.min.z  // front wall (where photo is)
        
        if debugMode {
            logDebug("🎯 [BoundaryManager] getCameraAtBackCenter depth=\(depth) inset=\(insetFromBack) pos=(\(camX),\(camY),\(camZ)) lookAt=(\(targetX),\(targetY),\(targetZ))")
        }
        
        return (
            position: SIMD3<Float>(camX, camY, camZ),
            lookAt: SIMD3<Float>(targetX, targetY, targetZ)
        )
    }

    /// Vertical lens angle needed to contain the complete front wall for the current viewport.
    /// The position and lens are derived from scene bounds; no bundled-room coordinates are used.
    func verticalFieldOfViewToFrameFrontWall(
        cameraPosition: SIMD3<Float>,
        lookAtPosition: SIMD3<Float>,
        photoOrientation: PhotoOrientation,
        viewportSize: CGSize?,
        minimumDegrees: Float = 60
    ) -> Float? {
        guard let bounds = roomBounds else { return nil }

        let direction = lookAtPosition - cameraPosition
        guard simd_length_squared(direction) > 1e-8 else { return nil }
        let forward = simd_normalize(direction)
        let worldUp = SIMD3<Float>(0, 1, 0)
        var right = simd_cross(forward, worldUp)
        if simd_length_squared(right) < 1e-8 {
            right = SIMD3<Float>(1, 0, 0)
        } else {
            right = simd_normalize(right)
        }
        let viewUp = simd_normalize(simd_cross(right, forward))
        let viewportAspect = max(
            DepthAnythingFlatPhotoCameraFraming.viewportAspect(
                photoOrientation: photoOrientation,
                viewportSize: viewportSize
            ),
            0.01
        )

        var maximumHorizontalSlope: Float = 0
        var maximumVerticalSlope: Float = 0
        for x in [bounds.min.x, bounds.max.x] {
            for y in [bounds.min.y, bounds.max.y] {
                let cameraToCorner = SIMD3<Float>(x, y, bounds.min.z) - cameraPosition
                let forwardDistance = simd_dot(cameraToCorner, forward)
                guard forwardDistance > 0.01 else { continue }
                maximumHorizontalSlope = max(
                    maximumHorizontalSlope,
                    abs(simd_dot(cameraToCorner, right)) / forwardDistance
                )
                maximumVerticalSlope = max(
                    maximumVerticalSlope,
                    abs(simd_dot(cameraToCorner, viewUp)) / forwardDistance
                )
            }
        }

        let framingMargin: Float = 1.06
        let requiredVerticalSlope = max(
            maximumVerticalSlope,
            maximumHorizontalSlope / viewportAspect
        ) * framingMargin
        let requiredDegrees = 2 * atan(requiredVerticalSlope) * 180 / .pi
        return min(max(requiredDegrees, minimumDegrees), 125)
    }

    private func depthAnythingViewportAspect(photoOrientation: PhotoOrientation) -> Float {
        DepthAnythingFlatPhotoCameraFraming.viewportAspect(
            photoOrientation: photoOrientation,
            viewportSize: arView?.bounds.size
        )
    }

    private func depthAnythingImagePlaneStandoff(
        width: Float,
        height: Float,
        span: Float,
        photoOrientation: PhotoOrientation
    ) -> Float {
        _ = span
        return DepthAnythingFlatPhotoCameraFraming.imagePlaneStandoff(
            planeWidthMeters: width,
            planeHeightMeters: height,
            photoOrientation: photoOrientation,
            viewportAspect: depthAnythingViewportAspect(photoOrientation: photoOrientation)
        )
    }

    /// Photographer viewpoint for Depth Anything `--flat-mesh` USDZ: plane at z≈0, camera on −Z.
    func getCameraForDepthAnythingImagePlane(
        photoOrientation: PhotoOrientation = .portrait,
        inferencePlaneWidthMeters: Float? = nil,
        inferencePlaneHeightMeters: Float? = nil
    ) -> (position: SIMD3<Float>, lookAt: SIMD3<Float>) {
        let debugMode = AppStateManager.shared.qualitySettings.debugMode

        guard let bounds = roomBounds else {
            return (SIMD3(0, 0, -2.5), SIMD3(0, 0, 0))
        }

        let center = getRoomCenter()
        let boundsWidth = max(bounds.max.x - bounds.min.x, 0.1)
        let boundsHeight = max(bounds.max.y - bounds.min.y, 0.1)
        // Frame to the rendered mesh span — stored inference W×H can overshoot mesh bounds and letterbox.
        let width = boundsWidth
        let height = boundsHeight
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
        roomCoordinateFrame: RoomCoordinateFrame = .canonicalSplatPly,
        photoOrientation: PhotoOrientation = .portrait,
        inferencePlaneWidthMeters: Float? = nil,
        inferencePlaneHeightMeters: Float? = nil
    ) -> (position: SIMD3<Float>, lookAt: SIMD3<Float>) {
        if roomCoordinateFrame == .depthAnythingImageDepthMeters && !hasNavigableInterior {
            return getCameraForDepthAnythingImagePlane(
                photoOrientation: photoOrientation,
                inferencePlaneWidthMeters: inferencePlaneWidthMeters,
                inferencePlaneHeightMeters: inferencePlaneHeightMeters
            )
        }
        if roomCoordinateFrame.usesFrontFacingRealityKitCamera && !hasNavigableInterior {
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
