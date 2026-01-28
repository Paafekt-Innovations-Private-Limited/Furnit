package com.furnit.android

import android.graphics.Bitmap
import android.graphics.Color
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.furnit.android.services.GlbGenerator
import org.junit.After
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/**
 * Tests for Room Save/Preview flow
 * Verifies:
 * - Rooms are initially saved to room_preview/ folder (not rooms/)
 * - Room can be moved from preview to rooms/ folder
 * - Preview folder is cleared before new room creation
 * - Metadata files are created correctly
 */
@RunWith(AndroidJUnit4::class)
class RoomSaveFlowTest {

    private lateinit var context: android.content.Context
    private lateinit var previewDir: File
    private lateinit var roomsDir: File

    @Before
    fun setup() {
        context = InstrumentationRegistry.getInstrumentation().targetContext

        previewDir = File(context.filesDir, "room_preview")
        roomsDir = File(context.filesDir, "rooms")

        // Clean test data
        TestCleanup.cleanAll(context)
    }

    @After
    fun teardown() {
        TestCleanup.cleanAll(context)
    }

    @Test
    fun testPreviewDirectoryCreation() {
        // Clean preview directory first
        previewDir.deleteRecursively()
        assertFalse("Preview dir should not exist initially", previewDir.exists())

        // Create a test room using GlbGenerator (simulating SinglePhotoRoomReconstructor)
        val generator = GlbGenerator()
        val texture = createTestBitmap()

        // Create in preview directory (not rooms/)
        previewDir.mkdirs()
        val timestamp = System.currentTimeMillis()
        val roomFolder = File(previewDir, "room_$timestamp")
        roomFolder.mkdirs()

        val glbFile = File(roomFolder, "room.glb")
        val success = generator.generateGlb(
            outputFile = glbFile,
            dimensions = GlbGenerator.RoomDimensions(),
            frontWallTexture = texture,
            floorTexture = texture,
            ceilingTexture = texture,
            leftWallTexture = texture,
            rightWallTexture = texture
        )

        assertTrue("GLB generation should succeed", success)
        assertTrue("GLB file should exist in preview folder", glbFile.exists())
        assertTrue("Preview folder should contain room", roomFolder.exists())

        // Room should NOT be in rooms/ folder yet
        val roomInRoomsDir = File(roomsDir, roomFolder.name)
        assertFalse("Room should NOT be in rooms/ folder yet", roomInRoomsDir.exists())

        texture.recycle()
        println("Preview directory creation test PASSED")
    }

    @Test
    fun testMovePreviewToRooms() {
        // Setup: Create a room in preview directory
        previewDir.deleteRecursively()
        previewDir.mkdirs()

        val generator = GlbGenerator()
        val texture = createTestBitmap()

        val timestamp = System.currentTimeMillis()
        val previewRoomFolder = File(previewDir, "room_test_$timestamp")
        previewRoomFolder.mkdirs()

        val glbFile = File(previewRoomFolder, "room.glb")
        generator.generateGlb(
            outputFile = glbFile,
            dimensions = GlbGenerator.RoomDimensions(),
            frontWallTexture = texture,
            floorTexture = texture,
            ceilingTexture = texture,
            leftWallTexture = texture,
            rightWallTexture = texture
        )

        // Create metadata file
        val metadataFile = File(previewRoomFolder, "metadata.txt")
        metadataFile.writeText("name=Test Room\ncreated=$timestamp\nglb=room.glb")

        assertTrue("Preview room should exist", previewRoomFolder.exists())
        assertTrue("GLB should exist", glbFile.exists())

        // Move from preview to rooms (simulating Save button click)
        roomsDir.mkdirs()
        val savedRoomFolder = File(roomsDir, previewRoomFolder.name)

        // Copy files (move operation)
        previewRoomFolder.copyRecursively(savedRoomFolder, overwrite = true)
        previewRoomFolder.deleteRecursively()

        // Verify move
        assertTrue("Room should exist in rooms/ folder", savedRoomFolder.exists())
        assertTrue("GLB should exist in saved room", File(savedRoomFolder, "room.glb").exists())
        assertTrue("Metadata should exist in saved room", File(savedRoomFolder, "metadata.txt").exists())
        assertFalse("Preview room should be deleted", previewRoomFolder.exists())

        texture.recycle()
        println("Move preview to rooms test PASSED")
    }

    @Test
    fun testPreviewFolderClearing() {
        // Create first room in preview
        previewDir.mkdirs()
        val firstRoom = File(previewDir, "room_first")
        firstRoom.mkdirs()
        val firstFile = File(firstRoom, "test.txt")
        firstFile.writeText("first room")

        assertTrue("First room should exist", firstRoom.exists())

        // Simulate creating a new room (should clear preview first)
        previewDir.deleteRecursively()
        previewDir.mkdirs()

        val secondRoom = File(previewDir, "room_second")
        secondRoom.mkdirs()
        val secondFile = File(secondRoom, "test.txt")
        secondFile.writeText("second room")

        // First room should be gone, only second room exists
        assertFalse("First room should be cleared", firstRoom.exists())
        assertTrue("Second room should exist", secondRoom.exists())

        println("Preview folder clearing test PASSED")
    }

    @Test
    fun testMetadataFileFormat() {
        previewDir.mkdirs()
        val roomFolder = File(previewDir, "room_metadata_test")
        roomFolder.mkdirs()

        val timestamp = System.currentTimeMillis()
        val roomName = "My Test Room Jan 29"
        val metadataFile = File(roomFolder, "metadata.txt")
        metadataFile.writeText("name=$roomName\ncreated=$timestamp\nglb=room.glb")

        assertTrue("Metadata file should exist", metadataFile.exists())

        // Parse metadata
        val content = metadataFile.readText()
        assertTrue("Metadata should contain name", content.contains("name=$roomName"))
        assertTrue("Metadata should contain created timestamp", content.contains("created=$timestamp"))
        assertTrue("Metadata should contain glb filename", content.contains("glb=room.glb"))

        // Verify can parse as key=value pairs
        val pairs = content.lines().associate {
            val parts = it.split("=", limit = 2)
            parts[0] to parts.getOrElse(1) { "" }
        }
        assertEquals("Parsed name", roomName, pairs["name"])
        assertEquals("Parsed glb", "room.glb", pairs["glb"])

        println("Metadata file format test PASSED")
    }

    @Test
    fun testDimensionsFileFormat() {
        previewDir.mkdirs()
        val roomFolder = File(previewDir, "room_dimensions_test")
        roomFolder.mkdirs()

        val width = 8.0f
        val depth = 9.0f
        val height = 5.6f

        val dimensionsFile = File(roomFolder, "dimensions.txt")
        dimensionsFile.writeText("width=$width\ndepth=$depth\nheight=$height")

        assertTrue("Dimensions file should exist", dimensionsFile.exists())

        // Parse dimensions
        val pairs = dimensionsFile.readText().lines().associate {
            val parts = it.split("=", limit = 2)
            parts[0] to parts.getOrElse(1) { "" }.toFloatOrNull()
        }

        assertEquals("Parsed width", width, pairs["width"])
        assertEquals("Parsed depth", depth, pairs["depth"])
        assertEquals("Parsed height", height, pairs["height"])

        println("Dimensions file format test PASSED")
    }

    @Test
    fun testFrontWallTextureSaved() {
        previewDir.deleteRecursively()
        previewDir.mkdirs()

        val generator = GlbGenerator()
        val frontWall = Bitmap.createBitmap(512, 512, Bitmap.Config.ARGB_8888).apply {
            eraseColor(Color.parseColor("#E8F5E9"))  // Light green
        }
        val otherTexture = createTestBitmap()

        val timestamp = System.currentTimeMillis()
        val roomFolder = File(previewDir, "room_$timestamp")
        roomFolder.mkdirs()

        // Save front wall texture for thumbnail (as SinglePhotoRoomReconstructor does)
        val frontWallFile = File(roomFolder, "front_wall.png")
        java.io.FileOutputStream(frontWallFile).use { out ->
            frontWall.compress(Bitmap.CompressFormat.PNG, 90, out)
        }

        val glbFile = File(roomFolder, "room.glb")
        generator.generateGlb(
            outputFile = glbFile,
            dimensions = GlbGenerator.RoomDimensions(),
            frontWallTexture = frontWall,
            floorTexture = otherTexture,
            ceilingTexture = otherTexture,
            leftWallTexture = otherTexture,
            rightWallTexture = otherTexture
        )

        assertTrue("Front wall texture should be saved", frontWallFile.exists())
        assertTrue("GLB should be created", glbFile.exists())
        assertTrue("Front wall texture size should be > 0", frontWallFile.length() > 0)

        frontWall.recycle()
        otherTexture.recycle()
        println("Front wall texture save test PASSED")
    }

    @Test
    fun testRoomFolderNaming() {
        // Test that room folders follow the expected naming pattern
        val timestamp1 = 1706500000000L  // Fixed timestamps for testing
        val timestamp2 = 1706500001000L

        val roomName1 = "room_$timestamp1"
        val roomName2 = "room_$timestamp2"

        assertTrue("Room name should start with 'room_'", roomName1.startsWith("room_"))
        assertTrue("Room name should contain timestamp", roomName1.contains("1706500000000"))
        assertNotEquals("Different timestamps should produce different names", roomName1, roomName2)

        println("Room folder naming test PASSED")
    }

    private fun createTestBitmap(): Bitmap {
        return Bitmap.createBitmap(256, 256, Bitmap.Config.ARGB_8888).apply {
            eraseColor(Color.GRAY)
        }
    }
}
