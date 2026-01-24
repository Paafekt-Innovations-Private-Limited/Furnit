package com.furnit.android.utils

import android.util.Log
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

        // Default room dimensions (matches GlbGenerator defaults)
        const val DEFAULT_WIDTH = 4.0f
        const val DEFAULT_DEPTH = 4.5f
        const val DEFAULT_HEIGHT = 2.8f

        // Camera positioning constants
        const val CAMERA_PADDING = 0.3f      // Distance from walls
        const val EYE_LEVEL_HEIGHT = 1.6f    // Standing eye level from floor (meters)
        const val EYE_LEVEL_OFFSET = 0.2f    // Above room center (legacy)
        const val BOUNDARY_PADDING = 0.5f    // For constraining movement
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

        Log.d(TAG, "Initialized room bounds from dimensions:")
        Log.d(TAG, "  Size: ${width}x${height}x${depth}")
        Log.d(TAG, "  X: ${-halfWidth} to ${halfWidth}")
        Log.d(TAG, "  Y: 0 to $height")
        Log.d(TAG, "  Z: ${-halfDepth} to ${halfDepth}")
    }

    /**
     * Get optimal camera position to view the room
     * Strategy: Position at the middle of the back wall, looking at front wall (photo)
     *
     * Camera is placed:
     * - X: center of room
     * - Y: eye level (1.6m from floor, typical standing height)
     * - Z: at the back wall (inside the room)
     */
    fun getOptimalCameraPosition(fovDegrees: Float = 60f): CameraSetup {
        val bounds = roomBounds ?: run {
            // Use defaults if not initialized
            initializeFromDimensions()
            roomBounds!!
        }

        // Camera position: Center of back wall at eye level
        val camX = bounds.centerX                      // Center X
        val camY = EYE_LEVEL_HEIGHT                    // Eye level (1.6m from floor)
        val camZ = bounds.backWallZ - CAMERA_PADDING   // At back wall, slightly inside

        // Look at the center of the front wall (where the photo is)
        val targetX = bounds.centerX
        val targetY = bounds.centerY
        val targetZ = bounds.frontWallZ

        val cameraSetup = CameraSetup(
            position = Position(camX, camY, camZ),
            lookAt = Position(targetX, targetY, targetZ)
        )

        Log.d(TAG, "Camera at back wall center:")
        Log.d(TAG, "  Room bounds: X[${bounds.minX}, ${bounds.maxX}], Y[${bounds.minY}, ${bounds.maxY}], Z[${bounds.minZ}, ${bounds.maxZ}]")
        Log.d(TAG, "  Camera position: (${camX}, ${camY}, ${camZ})")
        Log.d(TAG, "  LookAt: (${targetX}, ${targetY}, ${targetZ})")

        return cameraSetup
    }

    /**
     * Get camera position at back-left corner
     * This gives a perspective view of the room
     */
    fun getCameraAtBackLeftCorner(): CameraSetup {
        val bounds = roomBounds ?: run {
            initializeFromDimensions()
            roomBounds!!
        }

        // Back-left corner with padding, at eye level
        val camX = bounds.minX + CAMERA_PADDING
        val camY = EYE_LEVEL_HEIGHT
        val camZ = bounds.backWallZ - CAMERA_PADDING

        // Look at front-center
        val targetX = bounds.centerX
        val targetY = bounds.centerY
        val targetZ = bounds.frontWallZ

        return CameraSetup(
            position = Position(camX, camY, camZ),
            lookAt = Position(targetX, targetY, targetZ)
        )
    }

    /**
     * Get camera position centered at back wall, looking at front wall
     * Best for viewing the photo directly
     */
    fun getCameraCenteredView(): CameraSetup {
        val bounds = roomBounds ?: run {
            initializeFromDimensions()
            roomBounds!!
        }

        // Centered at back wall, eye level
        val camX = bounds.centerX
        val camY = EYE_LEVEL_HEIGHT
        val camZ = bounds.backWallZ - CAMERA_PADDING  // At back wall, inside room

        // Look at front wall center
        val targetX = bounds.centerX
        val targetY = bounds.centerY
        val targetZ = bounds.frontWallZ

        return CameraSetup(
            position = Position(camX, camY, camZ),
            lookAt = Position(targetX, targetY, targetZ)
        )
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
