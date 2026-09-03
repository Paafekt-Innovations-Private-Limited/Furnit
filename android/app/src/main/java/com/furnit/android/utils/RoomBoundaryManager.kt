package com.furnit.android.utils

import com.furnit.android.utils.LogUtil
import io.github.sceneview.math.Position

/**
 * RoomBoundaryManager - Manages room boundaries and camera positioning
 * (Matches iOS RoomBoundaryManager architecture)
 *
 * Responsibilities:
 * - Calculate room bounds from dimensions or loaded model
 * - Position camera optimally to view the entire room
 * - Constrain camera movement within room boundaries
 * - Provide boundary query methods
 */
class RoomBoundaryManager {

    companion object {
        private const val TAG = "RoomBoundaryManager"

        // Default room dimensions (doubled for better visibility)
        const val DEFAULT_WIDTH = 8.0f
        const val DEFAULT_DEPTH = 9.0f
        const val DEFAULT_HEIGHT = 5.6f

        // Camera positioning constants
        const val CAMERA_PADDING = 0.3f      // Distance from walls
        const val EYE_LEVEL_HEIGHT = 1.6f    // Standing eye level from floor (meters)
        const val EYE_LEVEL_OFFSET = 0.2f    // Above room center (legacy)
        const val BOUNDARY_PADDING = 0.5f    // For constraining movement

        // Field of view for camera distance calculation
        // Portrait mode has narrower horizontal FOV (~45 degrees typical for phone cameras)
        const val PORTRAIT_HORIZONTAL_FOV = 45f
        const val LANDSCAPE_HORIZONTAL_FOV = 60f
    }

    /**
     * Room bounds in world coordinates
     */
    data class RoomBounds(
        val minX: Float,
        val maxX: Float,
        val minY: Float,
        val maxY: Float,
        val minZ: Float,
        val maxZ: Float
    ) {
        val width: Float get() = maxX - minX
        val height: Float get() = maxY - minY
        val depth: Float get() = maxZ - minZ

        val centerX: Float get() = (minX + maxX) / 2f
        val centerY: Float get() = (minY + maxY) / 2f
        val centerZ: Float get() = (minZ + maxZ) / 2f

        val center: Position get() = Position(centerX, centerY, centerZ)

        // Front wall is at minZ (negative Z direction)
        val frontWallZ: Float get() = minZ
        // Back wall is at maxZ (positive Z direction)
        val backWallZ: Float get() = maxZ

        val floorY: Float get() = minY
        val ceilingY: Float get() = maxY
    }

    /**
     * Camera position and look-at target
     */
    data class CameraSetup(
        val position: Position,
        val lookAt: Position
    )

    private var roomBounds: RoomBounds? = null

    /**
     * Initialize bounds for a model whose bounding box was centered at the world origin.
     * X/Z are symmetric; Y spans [-height/2, +height/2] (not floor-at-zero).
     */
    fun initializeFromCenteredExtents(
        width: Float = DEFAULT_WIDTH,
        height: Float = DEFAULT_HEIGHT,
        depth: Float = DEFAULT_DEPTH,
    ) {
        val halfWidth = width / 2f
        val halfHeight = height / 2f
        val halfDepth = depth / 2f

        roomBounds = RoomBounds(
            minX = -halfWidth,
            maxX = halfWidth,
            minY = -halfHeight,
            maxY = halfHeight,
            minZ = -halfDepth,
            maxZ = halfDepth,
        )

        LogUtil.d(TAG, "Initialized centered room bounds:")
        LogUtil.d(TAG, "  Size: ${width}x${height}x${depth}")
        LogUtil.d(TAG, "  X: ${-halfWidth} to ${halfWidth}")
        LogUtil.d(TAG, "  Y: ${-halfHeight} to ${halfHeight}")
        LogUtil.d(TAG, "  Z: ${-halfDepth} to ${halfDepth}")
    }

    /**
     * Initialize with room dimensions (used for GlbGenerator rooms)
     * Room is centered at origin: X from -width/2 to +width/2
     *                             Y from 0 to height
     *                             Z from -depth/2 to +depth/2
     */
    fun initializeFromDimensions(
        width: Float = DEFAULT_WIDTH,
        depth: Float = DEFAULT_DEPTH,
        height: Float = DEFAULT_HEIGHT
    ) {
        val halfWidth = width / 2f
        val halfDepth = depth / 2f

        roomBounds = RoomBounds(
            minX = -halfWidth,
            maxX = halfWidth,
            minY = 0f,
            maxY = height,
            minZ = -halfDepth,  // Front wall (where photo is)
            maxZ = halfDepth    // Back wall (camera side)
        )

        LogUtil.d(TAG, "Initialized room bounds from dimensions:")
        LogUtil.d(TAG, "  Size: ${width}x${height}x${depth}")
        LogUtil.d(TAG, "  X: ${-halfWidth} to ${halfWidth}")
        LogUtil.d(TAG, "  Y: 0 to $height")
        LogUtil.d(TAG, "  Z: ${-halfDepth} to ${halfDepth}")
    }

    /**
     * Get optimal camera position to view the room
     * Strategy: Position camera at the back wall (imaginary back wall),
     * near the left corner, looking toward the front wall where the photo is.
     * This matches the iOS RealityKitBoundaryManager positioning.
     *
     * @param isPortrait true if device is in portrait orientation
     * @param horizontalFovDegrees horizontal field of view in degrees (unused, kept for API compat)
     */
    @Suppress("UNUSED_PARAMETER")
    fun getOptimalCameraPosition(isPortrait: Boolean = true, horizontalFovDegrees: Float = 60f): CameraSetup {
        val bounds = roomBounds ?: run {
            // Use defaults if not initialized
            initializeFromDimensions()
            roomBounds!!
        }

        // Camera positioning strategy: Far back, see entire room as small box

        // Position camera WAY back to see the whole room
        val camX = bounds.centerX                  // Center X
        val camY = bounds.centerY + 2.0f           // Slightly elevated
        val camZ = bounds.backWallZ + 10.0f        // Far behind the room

        // Look at the center of the room
        val targetX = bounds.centerX
        val targetY = bounds.centerY
        val targetZ = bounds.centerZ

        val cameraSetup = CameraSetup(
            position = Position(camX, camY, camZ),
            lookAt = Position(targetX, targetY, targetZ)
        )

        LogUtil.d(TAG, "Camera position (back-wall corner) for ${if (isPortrait) "PORTRAIT" else "LANDSCAPE"}:")
        LogUtil.d(TAG, "  Room: ${bounds.width}x${bounds.height}x${bounds.depth}")
        LogUtil.d(TAG, "  Camera at back-left corner: ($camX, $camY, $camZ)")
        LogUtil.d(TAG, "  Looking at front wall: ($targetX, $targetY, $targetZ)")

        return cameraSetup
    }

    /**
     * Get camera position at back-left corner
     * This gives a perspective view of the room from the imaginary back wall
     */
    fun getCameraAtBackLeftCorner(): CameraSetup {
        val bounds = roomBounds ?: run {
            initializeFromDimensions()
            roomBounds!!
        }

        // Back-left corner with padding, eye level above room center
        val camX = bounds.minX + CAMERA_PADDING
        val camY = bounds.centerY + 0.4f  // Slightly above room center
        val camZ = bounds.backWallZ - CAMERA_PADDING

        // Look at front wall center (where the photo is)
        val targetX = bounds.centerX
        val targetY = bounds.centerY
        val targetZ = bounds.frontWallZ

        return CameraSetup(
            position = Position(camX, camY, camZ),
            lookAt = Position(targetX, targetY, targetZ)
        )
    }

    /**
     * Get a centered, straight-on room view.
     * Camera sits at the room center and looks directly at the center of the front wall.
     */
    fun getCameraCenteredView(): CameraSetup {
        val bounds = roomBounds ?: run {
            initializeFromDimensions()
            roomBounds!!
        }

        val camX = bounds.centerX
        val camY = bounds.centerY
        val camZ = bounds.centerZ

        val targetX = bounds.centerX
        val targetY = bounds.centerY
        val targetZ = bounds.frontWallZ

        LogUtil.d(TAG, "  Room position/bounds: min=(${bounds.minX}, ${bounds.minY}, ${bounds.minZ}) max=(${bounds.maxX}, ${bounds.maxY}, ${bounds.maxZ}) center=(${bounds.centerX}, ${bounds.centerY}, ${bounds.centerZ})")
        LogUtil.d(TAG, "  Camera (centered front-wall view): pos=($camX, $camY, $camZ) lookAt=($targetX, $targetY, $targetZ)")
        return CameraSetup(
            position = Position(camX, camY, camZ),
            lookAt = Position(targetX, targetY, targetZ)
        )
    }

    /**
     * Which horizontal axis carries the room's view depth (longest X/Z span; Y is always up).
     * Some baked rooms may export depth along X instead of Z.
     */
    private enum class ViewDepthAxis { X, Z }

    private data class ViewDepthFrame(
        val axis: ViewDepthAxis,
        val span: Float,
        val front: Float,
        val back: Float,
        val lateralCenter: Float,
        val centerY: Float,
    )

    private fun resolveViewDepth(bounds: RoomBounds): ViewDepthFrame {
        // Some exports can have real room depth along X while Z is a thin slab (~0.8 m).
        // Keep Z as the established default; only remap when its span is implausibly small.
        val depthTooThin = bounds.depth < minOf(bounds.width * 0.5f, 1.5f)
        val useXAsDepth = bounds.width > bounds.depth && depthTooThin
        return if (useXAsDepth) {
            ViewDepthFrame(
                axis = ViewDepthAxis.X,
                span = bounds.width,
                front = bounds.minX,
                back = bounds.maxX,
                lateralCenter = bounds.centerZ,
                centerY = bounds.centerY,
            )
        } else {
            ViewDepthFrame(
                axis = ViewDepthAxis.Z,
                span = bounds.depth,
                front = bounds.minZ,
                back = bounds.maxZ,
                lateralCenter = bounds.centerX,
                centerY = bounds.centerY,
            )
        }
    }

    private fun cameraPositionForDepthFrame(
        frame: ViewDepthFrame,
        depthCoordinate: Float,
        eyeYOffset: Float,
    ): Position {
        return when (frame.axis) {
            ViewDepthAxis.X -> Position(depthCoordinate, frame.centerY + eyeYOffset, frame.lateralCenter)
            ViewDepthAxis.Z -> Position(frame.lateralCenter, frame.centerY + eyeYOffset, depthCoordinate)
        }
    }

    private fun lookAtForDepthFrame(frame: ViewDepthFrame): Position {
        return when (frame.axis) {
            ViewDepthAxis.X -> Position(frame.front, frame.centerY, frame.lateralCenter)
            ViewDepthAxis.Z -> Position(frame.lateralCenter, frame.centerY, frame.front)
        }
    }

    /**
     * Depth-adaptive inset from back wall (matches iOS RealityKitBoundaryManager.backCenterInsetFraction).
     * Shallow rooms: smaller fraction (camera stays near back). Deep rooms: slightly further in.
     */
    private fun backCenterInsetFraction(depth: Float): Float {
        val t = (depth / 6f).coerceIn(0f, 1f)
        return 0.035f + 0.065f * t
    }

    /**
     * Bounds-driven camera: pick longest horizontal axis as view depth, then place the camera
     * outside the back face looking at the front for bundled-room views.
     */
    fun getCameraOutsideBackView(standoffFraction: Float = 0.3f): CameraSetup {
        val bounds = roomBounds ?: run {
            initializeFromCenteredExtents()
            roomBounds!!
        }
        val frame = resolveViewDepth(bounds)
        val standoff = frame.span * standoffFraction
        val position = cameraPositionForDepthFrame(frame, frame.back + standoff, eyeYOffset = 0f)
        val lookAt = lookAtForDepthFrame(frame)

        LogUtil.d(
            TAG,
            "[OutsideBack] axis=${frame.axis} span=${frame.span} standoff=$standoff " +
                "bbox=${bounds.width}x${bounds.height}x${bounds.depth} -> pos=(${position.x}, ${position.y}, ${position.z}) " +
                "lookAt=(${lookAt.x}, ${lookAt.y}, ${lookAt.z})",
        )
        return CameraSetup(position = position, lookAt = lookAt)
    }

    /**
     * Bounds-driven interior camera: same depth-axis selection as [getCameraOutsideBackView],
     * inset from the back wall toward the front (user-generated / file-system rooms).
     */
    fun getCameraAtBackCenter(): CameraSetup {
        val bounds = roomBounds ?: run {
            initializeFromCenteredExtents()
            roomBounds!!
        }
        val frame = resolveViewDepth(bounds)
        val fraction = backCenterInsetFraction(frame.span)
        val insetFromBack = (frame.span * fraction).coerceAtLeast(CAMERA_PADDING)
        val position = cameraPositionForDepthFrame(frame, frame.back - insetFromBack, eyeYOffset = 0.4f)
        val lookAt = lookAtForDepthFrame(frame)

        LogUtil.d(
            TAG,
            "[BackCenter] axis=${frame.axis} span=${frame.span}m fraction=$fraction inset=${insetFromBack}m " +
                "-> pos=(${position.x}, ${position.y}, ${position.z}) lookAt=(${lookAt.x}, ${lookAt.y}, ${lookAt.z})",
        )
        return CameraSetup(position = position, lookAt = lookAt)
    }

    /**
     * Constrain camera position within room boundaries
     * Allows movement inside the room but prevents going through walls
     */
    fun constrainCameraPosition(position: Position): Position {
        val bounds = roomBounds ?: return position

        val constrainedX = position.x.coerceIn(
            bounds.minX + BOUNDARY_PADDING,
            bounds.maxX - BOUNDARY_PADDING
        )

        // Allow camera slightly above floor to above ceiling
        val constrainedY = position.y.coerceIn(
            bounds.minY + 0.5f,
            bounds.maxY + 2.0f
        )

        val constrainedZ = position.z.coerceIn(
            bounds.minZ + BOUNDARY_PADDING,
            bounds.maxZ + 2.0f  // Allow camera to be outside back wall
        )

        return Position(constrainedX, constrainedY, constrainedZ)
    }

    /**
     * Check if a position is within room bounds
     */
    fun isPositionWithinBounds(position: Position): Boolean {
        val bounds = roomBounds ?: return true

        return position.x >= bounds.minX && position.x <= bounds.maxX &&
               position.y >= bounds.minY && position.y <= bounds.maxY &&
               position.z >= bounds.minZ && position.z <= bounds.maxZ
    }

    /**
     * Get room center position
     */
    fun getRoomCenter(): Position {
        val bounds = roomBounds ?: run {
            initializeFromDimensions()
            roomBounds!!
        }
        return bounds.center
    }

    /**
     * Get room dimensions
     */
    fun getRoomDimensions(): Position {
        val bounds = roomBounds ?: run {
            initializeFromDimensions()
            roomBounds!!
        }
        return Position(bounds.width, bounds.height, bounds.depth)
    }

    /**
     * Get floor height (Y position)
     */
    fun getFloorHeight(): Float {
        return roomBounds?.floorY ?: 0f
    }

    /**
     * Get current room bounds
     */
    fun getBounds(): RoomBounds? = roomBounds

    /**
     * Reset/clear bounds
     */
    fun reset() {
        roomBounds = null
    }
}
