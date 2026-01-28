package com.furnit.android

import androidx.test.ext.junit.runners.AndroidJUnit4
import com.furnit.android.utils.RoomBoundaryManager
import io.github.sceneview.math.Position
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Tests for RoomBoundaryManager - Camera positioning and room bounds calculations
 * Verifies:
 * - Doubled room dimensions (8x9x5.6m)
 * - Camera positioned far back to see room as small box
 * - Correct boundary calculations
 * - Camera constraint behavior
 */
@RunWith(AndroidJUnit4::class)
class RoomBoundaryManagerTest {

    private lateinit var manager: RoomBoundaryManager

    @Before
    fun setup() {
        manager = RoomBoundaryManager()
    }

    @Test
    fun testDefaultDimensions() {
        // Verify doubled room dimensions
        assertEquals("Default width should be 8.0", 8.0f, RoomBoundaryManager.DEFAULT_WIDTH)
        assertEquals("Default depth should be 9.0", 9.0f, RoomBoundaryManager.DEFAULT_DEPTH)
        assertEquals("Default height should be 5.6", 5.6f, RoomBoundaryManager.DEFAULT_HEIGHT)
    }

    @Test
    fun testInitializeFromDimensions() {
        manager.initializeFromDimensions(8.0f, 9.0f, 5.6f)

        val bounds = manager.getBounds()
        assertNotNull("Bounds should be initialized", bounds)

        // Room centered at origin: X from -4 to +4, Y from 0 to 5.6, Z from -4.5 to +4.5
        assertEquals("minX should be -4.0", -4.0f, bounds!!.minX)
        assertEquals("maxX should be 4.0", 4.0f, bounds.maxX)
        assertEquals("minY should be 0.0", 0.0f, bounds.minY)
        assertEquals("maxY should be 5.6", 5.6f, bounds.maxY)
        assertEquals("minZ should be -4.5", -4.5f, bounds.minZ)
        assertEquals("maxZ should be 4.5", 4.5f, bounds.maxZ)
    }

    @Test
    fun testBoundsProperties() {
        manager.initializeFromDimensions(8.0f, 9.0f, 5.6f)
        val bounds = manager.getBounds()!!

        assertEquals("Width should be 8.0", 8.0f, bounds.width)
        assertEquals("Height should be 5.6", 5.6f, bounds.height)
        assertEquals("Depth should be 9.0", 9.0f, bounds.depth)

        assertEquals("centerX should be 0.0", 0.0f, bounds.centerX)
        assertEquals("centerY should be 2.8", 2.8f, bounds.centerY)
        assertEquals("centerZ should be 0.0", 0.0f, bounds.centerZ)

        assertEquals("frontWallZ should be -4.5", -4.5f, bounds.frontWallZ)
        assertEquals("backWallZ should be 4.5", 4.5f, bounds.backWallZ)
        assertEquals("floorY should be 0.0", 0.0f, bounds.floorY)
        assertEquals("ceilingY should be 5.6", 5.6f, bounds.ceilingY)
    }

    @Test
    fun testOptimalCameraPositionFarBack() {
        manager.initializeFromDimensions(8.0f, 9.0f, 5.6f)

        val cameraSetup = manager.getOptimalCameraPosition(isPortrait = true)

        // Camera should be far behind the room (Z = backWallZ + 10 = 4.5 + 10 = 14.5)
        assertEquals("Camera X should be centered", 0.0f, cameraSetup.position.x)
        assertTrue("Camera Y should be elevated (centerY + 2)", cameraSetup.position.y > 4.0f)
        assertEquals("Camera Z should be far behind room", 14.5f, cameraSetup.position.z)

        // Should look at room center
        assertEquals("Target X should be room center", 0.0f, cameraSetup.lookAt.x)
        assertEquals("Target Y should be room center", 2.8f, cameraSetup.lookAt.y)
        assertEquals("Target Z should be room center", 0.0f, cameraSetup.lookAt.z)
    }

    @Test
    fun testCameraAtBackLeftCorner() {
        manager.initializeFromDimensions(8.0f, 9.0f, 5.6f)

        val cameraSetup = manager.getCameraAtBackLeftCorner()

        // Should be at back-left corner with padding
        val expectedX = -4.0f + RoomBoundaryManager.CAMERA_PADDING
        val expectedZ = 4.5f - RoomBoundaryManager.CAMERA_PADDING

        assertEquals("Camera X at back-left", expectedX, cameraSetup.position.x, 0.01f)
        assertTrue("Camera Y above room center", cameraSetup.position.y > 2.8f)
        assertEquals("Camera Z at back wall", expectedZ, cameraSetup.position.z, 0.01f)
    }

    @Test
    fun testCameraCenteredView() {
        manager.initializeFromDimensions(8.0f, 9.0f, 5.6f)

        val cameraSetup = manager.getCameraCenteredView()

        // Should be centered at back wall
        assertEquals("Camera X should be centered", 0.0f, cameraSetup.position.x)
        assertTrue("Camera Y above room center", cameraSetup.position.y > 2.8f)

        // Should look at front wall
        assertEquals("Target Z should be front wall", -4.5f, cameraSetup.lookAt.z)
    }

    @Test
    fun testConstrainCameraPosition() {
        manager.initializeFromDimensions(8.0f, 9.0f, 5.6f)

        // Test position outside bounds (too far left)
        val outsideLeft = Position(-10.0f, 1.6f, 0.0f)
        val constrainedLeft = manager.constrainCameraPosition(outsideLeft)
        assertTrue("X should be constrained to min bound", constrainedLeft.x >= -4.0f + RoomBoundaryManager.BOUNDARY_PADDING)

        // Test position outside bounds (too far right)
        val outsideRight = Position(10.0f, 1.6f, 0.0f)
        val constrainedRight = manager.constrainCameraPosition(outsideRight)
        assertTrue("X should be constrained to max bound", constrainedRight.x <= 4.0f - RoomBoundaryManager.BOUNDARY_PADDING)

        // Test position within bounds
        val insideBounds = Position(0.0f, 1.6f, 0.0f)
        val constrainedInside = manager.constrainCameraPosition(insideBounds)
        assertEquals("Position inside bounds should not change X", 0.0f, constrainedInside.x, 0.01f)
        assertEquals("Position inside bounds should not change Y", 1.6f, constrainedInside.y, 0.01f)
        assertEquals("Position inside bounds should not change Z", 0.0f, constrainedInside.z, 0.01f)
    }

    @Test
    fun testIsPositionWithinBounds() {
        manager.initializeFromDimensions(8.0f, 9.0f, 5.6f)

        // Position at room center should be within bounds
        assertTrue("Center should be within bounds",
            manager.isPositionWithinBounds(Position(0.0f, 2.8f, 0.0f)))

        // Position outside X bounds
        assertFalse("Outside X should not be within bounds",
            manager.isPositionWithinBounds(Position(-5.0f, 2.8f, 0.0f)))

        // Position outside Y bounds
        assertFalse("Outside Y should not be within bounds",
            manager.isPositionWithinBounds(Position(0.0f, 7.0f, 0.0f)))

        // Position outside Z bounds
        assertFalse("Outside Z should not be within bounds",
            manager.isPositionWithinBounds(Position(0.0f, 2.8f, 6.0f)))
    }

    @Test
    fun testGetRoomCenter() {
        manager.initializeFromDimensions(8.0f, 9.0f, 5.6f)

        val center = manager.getRoomCenter()
        assertEquals("Center X", 0.0f, center.x)
        assertEquals("Center Y", 2.8f, center.y)
        assertEquals("Center Z", 0.0f, center.z)
    }

    @Test
    fun testGetRoomDimensions() {
        manager.initializeFromDimensions(8.0f, 9.0f, 5.6f)

        val dimensions = manager.getRoomDimensions()
        assertEquals("Width", 8.0f, dimensions.x)
        assertEquals("Height", 5.6f, dimensions.y)
        assertEquals("Depth", 9.0f, dimensions.z)
    }

    @Test
    fun testGetFloorHeight() {
        manager.initializeFromDimensions(8.0f, 9.0f, 5.6f)
        assertEquals("Floor Y should be 0", 0.0f, manager.getFloorHeight())
    }

    @Test
    fun testReset() {
        manager.initializeFromDimensions(8.0f, 9.0f, 5.6f)
        assertNotNull("Bounds should exist before reset", manager.getBounds())

        manager.reset()
        assertNull("Bounds should be null after reset", manager.getBounds())
    }

    @Test
    fun testAutoInitializeOnQuery() {
        // Don't call initializeFromDimensions - should auto-initialize on first query
        val center = manager.getRoomCenter()

        // Should use default dimensions
        assertEquals("Auto-init center X", 0.0f, center.x)
        assertEquals("Auto-init center Y", 2.8f, center.y)  // 5.6 / 2
        assertEquals("Auto-init center Z", 0.0f, center.z)
    }

    @Test
    fun testCustomDimensions() {
        // Test with custom dimensions (not defaults)
        manager.initializeFromDimensions(6.0f, 7.0f, 4.0f)
        val bounds = manager.getBounds()!!

        assertEquals("Custom width", 6.0f, bounds.width)
        assertEquals("Custom height", 4.0f, bounds.height)
        assertEquals("Custom depth", 7.0f, bounds.depth)

        assertEquals("Custom minX", -3.0f, bounds.minX)
        assertEquals("Custom maxX", 3.0f, bounds.maxX)
    }
}
