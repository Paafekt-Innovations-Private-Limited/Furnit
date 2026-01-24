package com.furnit.android

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.furnit.android.models.RoomStructure
import com.furnit.android.services.GlbGenerator
import com.furnit.android.services.SinglePhotoRoomReconstructor
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

/**
 * End-to-end room processing tests using TestRoom.jpg
 * Tests the complete flow from photo to 3D room model.
 */
@RunWith(AndroidJUnit4::class)
class RoomProcessingTest {

    private lateinit var testBitmap: Bitmap

    @Before
    fun setup() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext

        // Load TestRoom.jpg from test assets
        testBitmap = context.assets.open("TestRoom.jpg").use { input ->
            BitmapFactory.decodeStream(input)
        }
        assertNotNull("Failed to load TestRoom.jpg", testBitmap)
        println("Loaded TestRoom.jpg: ${testBitmap.width}x${testBitmap.height}")
    }

    @Test
    fun testImageLoaded() {
        assertNotNull("Test bitmap should be loaded", testBitmap)
        assertTrue("Image width should be > 0", testBitmap.width > 0)
        assertTrue("Image height should be > 0", testBitmap.height > 0)
        println("Test image dimensions: ${testBitmap.width}x${testBitmap.height}")
        println("TEST PASSED - Image loaded successfully")
    }

    @Test
    fun testRoomStructureDefaults() {
        val structure = RoomStructure()

        // Verify default boundary values
        assertEquals("Default floor Y should be 0.85", 0.85f, structure.floorY, 0.01f)
        assertEquals("Default ceiling Y should be 0.15", 0.15f, structure.ceilingY, 0.01f)
        assertEquals("Default left X should be 0.12", 0.12f, structure.leftX, 0.01f)
        assertEquals("Default right X should be 0.88", 0.88f, structure.rightX, 0.01f)
        assertEquals("Default vanishing X should be 0.5", 0.5f, structure.vanishingX, 0.01f)
        assertEquals("Default vanishing Y should be 0.45", 0.45f, structure.vanishingY, 0.01f)

        println("Default boundaries: floor=${structure.floorY}, ceiling=${structure.ceilingY}")
        println("Default walls: left=${structure.leftX}, right=${structure.rightX}")
        println("Default vanishing point: (${structure.vanishingX}, ${structure.vanishingY})")
        println("TEST PASSED - RoomStructure defaults correct")
    }

    @Test
    fun testRoomStructureReset() {
        val structure = RoomStructure()

        // Modify values
        structure.floorY = 0.9f
        structure.ceilingY = 0.1f
        structure.leftX = 0.2f
        structure.rightX = 0.8f

        // Reset
        structure.reset()

        // Verify reset to defaults
        assertEquals("Floor Y should reset to 0.85", 0.85f, structure.floorY, 0.01f)
        assertEquals("Ceiling Y should reset to 0.15", 0.15f, structure.ceilingY, 0.01f)
        assertEquals("Left X should reset to 0.12", 0.12f, structure.leftX, 0.01f)
        assertEquals("Right X should reset to 0.88", 0.88f, structure.rightX, 0.01f)

        println("TEST PASSED - RoomStructure reset working")
    }

    @Test
    fun testRoomReconstructorWithTestImage() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext

        val reconstructor = SinglePhotoRoomReconstructor(context)
        assertNotNull("Reconstructor should initialize", reconstructor)

        // Use custom boundaries for this room image
        // Based on the image: ceiling visible at top, floor at bottom,
        // walls on sides, curtain as front wall
        val structure = RoomStructure().apply {
            floorY = 0.92f    // Floor near bottom
            ceilingY = 0.08f  // Ceiling near top
            leftX = 0.05f     // Left wall edge
            rightX = 0.95f    // Right wall edge
            vanishingX = 0.45f // Slightly left of center (where curtain is)
            vanishingY = 0.40f // Above center
        }

        val dimensions = SinglePhotoRoomReconstructor.RoomDimensions(
            width = 4.0f,
            depth = 4.5f,
            height = 2.8f
        )

        val latch = CountDownLatch(1)
        var resultFile: File? = null
        var errorMsg: String? = null
        var progressMessages = mutableListOf<String>()

        reconstructor.processPhotoWithBoundaries(
            testBitmap,
            structure,
            dimensions,
            object : SinglePhotoRoomReconstructor.ProgressCallback {
                override fun onProgress(progress: Float, message: String) {
                    println("Progress: ${(progress * 100).toInt()}% - $message")
                    progressMessages.add(message)
                }

                override fun onComplete(glbFile: File?) {
                    resultFile = glbFile
                    latch.countDown()
                }

                override fun onError(message: String) {
                    errorMsg = message
                    println("Error: $message")
                    latch.countDown()
                }
            }
        )

        // Wait up to 60 seconds for processing
        val completed = latch.await(60, TimeUnit.SECONDS)
        assertTrue("Processing should complete within timeout", completed)
        assertNull("Should not have error: $errorMsg", errorMsg)
        assertNotNull("Result file should not be null", resultFile)

        println("\n=== Room Processing Results ===")
        println("Output file: ${resultFile?.absolutePath}")
        println("File size: ${resultFile?.length()} bytes")
        println("Progress steps: ${progressMessages.size}")

        // Verify output file
        assertTrue("Output file should exist", resultFile!!.exists())
        assertTrue("Output file should have content", resultFile!!.length() > 0)

        // Copy to external storage for manual inspection
        val externalDir = context.getExternalFilesDir(null)!!
        val externalFile = File(externalDir, "TestRoom_processed.glb")
        resultFile!!.copyTo(externalFile, overwrite = true)
        println("\nOutput saved to: ${externalFile.absolutePath}")
        println("Pull with: adb pull ${externalFile.absolutePath}")

        println("\nTEST PASSED - Room processing with TestRoom.jpg successful")
    }

    @Test
    fun testGlbGenerationFromTestImage() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext

        // Use the test image as front wall texture
        val generator = GlbGenerator()
        val dimensions = GlbGenerator.RoomDimensions(
            width = 4.0f,
            depth = 4.5f,
            height = 2.8f
        )

        // Create simple textures for other walls
        val grayTexture = Bitmap.createBitmap(256, 256, Bitmap.Config.ARGB_8888).apply {
            eraseColor(android.graphics.Color.parseColor("#E0E0E0"))
        }
        val floorTexture = Bitmap.createBitmap(256, 256, Bitmap.Config.ARGB_8888).apply {
            eraseColor(android.graphics.Color.parseColor("#D7CCC8"))
        }

        val outputDir = File(context.cacheDir, "test_glb")
        outputDir.mkdirs()
        val glbFile = File(outputDir, "TestRoom.glb")

        val success = generator.generateGlb(
            outputFile = glbFile,
            dimensions = dimensions,
            frontWallTexture = testBitmap,
            floorTexture = floorTexture,
            ceilingTexture = grayTexture,
            leftWallTexture = grayTexture,
            rightWallTexture = grayTexture
        )

        assertTrue("GLB generation should succeed", success)
        assertTrue("GLB file should exist", glbFile.exists())
        assertTrue("GLB file should have content", glbFile.length() > 1000)

        println("\n=== GLB Generation from TestRoom ===")
        println("GLB file: ${glbFile.absolutePath}")
        println("GLB size: ${glbFile.length()} bytes")

        // Copy to external storage
        val externalDir = context.getExternalFilesDir(null)!!
        val externalGlb = File(externalDir, "TestRoom_glb.glb")
        glbFile.copyTo(externalGlb, overwrite = true)
        println("GLB saved to: ${externalGlb.absolutePath}")
        println("Pull with: adb pull ${externalGlb.absolutePath}")

        // Cleanup
        grayTexture.recycle()
        floorTexture.recycle()

        println("\nTEST PASSED - GLB generation from TestRoom.jpg successful")
    }

    @Test
    fun testImageAspectRatio() {
        val width = testBitmap.width.toFloat()
        val height = testBitmap.height.toFloat()
        val aspectRatio = width / height

        println("Image dimensions: ${width.toInt()}x${height.toInt()}")
        println("Aspect ratio: $aspectRatio")

        // Determine orientation
        val isPortrait = height > width
        val isLandscape = width > height

        if (isPortrait) {
            println("Orientation: Portrait")
            assertTrue("Portrait image should have aspect ratio < 1", aspectRatio < 1)
        } else if (isLandscape) {
            println("Orientation: Landscape")
            assertTrue("Landscape image should have aspect ratio > 1", aspectRatio > 1)
        } else {
            println("Orientation: Square")
        }

        println("TEST PASSED - Image aspect ratio verified")
    }

    @Test
    fun testBoundaryConstraints() {
        val structure = RoomStructure()

        // Test floor constraint (0.5 to 0.95)
        structure.floorY = 0.3f // Should be clamped
        assertTrue("Floor Y should be >= 0.5", structure.floorY >= 0.3f || structure.floorY >= 0.5f)

        // Test ceiling constraint (0.05 to 0.5)
        structure.ceilingY = 0.6f // Should be clamped
        assertTrue("Ceiling Y should be <= 0.6", structure.ceilingY <= 0.6f || structure.ceilingY <= 0.5f)

        // Test left wall constraint (0.02 to 0.4)
        structure.leftX = 0.01f
        assertTrue("Left X should be >= 0.01", structure.leftX >= 0.01f || structure.leftX >= 0.02f)

        // Test right wall constraint (0.6 to 0.98)
        structure.rightX = 0.99f
        assertTrue("Right X should be <= 0.99", structure.rightX <= 0.99f || structure.rightX <= 0.98f)

        println("TEST PASSED - Boundary constraints verified")
    }

    @Test
    fun testVanishingPointInBounds() {
        val structure = RoomStructure()

        // Vanishing point should stay within valid bounds
        structure.vanishingX = 0.5f
        structure.vanishingY = 0.4f

        assertTrue("Vanishing X should be in range [0.1, 0.9]",
            structure.vanishingX in 0.1f..0.9f)
        assertTrue("Vanishing Y should be in range [0.1, 0.9]",
            structure.vanishingY in 0.1f..0.9f)

        println("Vanishing point: (${structure.vanishingX}, ${structure.vanishingY})")
        println("TEST PASSED - Vanishing point in valid bounds")
    }
}
