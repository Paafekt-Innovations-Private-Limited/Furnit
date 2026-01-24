package com.furnit.android

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.furnit.android.services.GlbGenerator
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Test GLB generation for room models.
 * Verifies that GlbGenerator creates valid GLB files with proper structure.
 */
@RunWith(AndroidJUnit4::class)
class GlbGeneratorTest {

    companion object {
        private const val GLB_MAGIC = 0x46546C67  // "glTF"
        private const val GLB_VERSION = 2
        private const val JSON_CHUNK_TYPE = 0x4E4F534A  // "JSON"
        private const val BIN_CHUNK_TYPE = 0x004E4942   // "BIN\0"
    }

    @Test
    fun testGlbGeneration() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext

        // Create test textures with distinct colors
        val floorTexture = createColoredBitmap(Color.parseColor("#D7CCC8"), 256, 256, "Floor")
        val ceilingTexture = createColoredBitmap(Color.WHITE, 256, 256, "Ceiling")
        val frontWallTexture = createColoredBitmap(Color.parseColor("#E8F5E9"), 256, 256, "Front")
        val leftWallTexture = createColoredBitmap(Color.parseColor("#E3F2FD"), 256, 256, "Left")
        val rightWallTexture = createColoredBitmap(Color.parseColor("#FFF3E0"), 256, 256, "Right")

        // Create output file
        val outputDir = File(context.cacheDir, "glb_test")
        outputDir.mkdirs()
        val glbFile = File(outputDir, "test_room.glb")

        // Generate GLB
        val generator = GlbGenerator()
        val dimensions = GlbGenerator.RoomDimensions(
            width = 4.0f,
            depth = 4.5f,
            height = 2.8f
        )

        val success = generator.generateGlb(
            outputFile = glbFile,
            dimensions = dimensions,
            frontWallTexture = frontWallTexture,
            floorTexture = floorTexture,
            ceilingTexture = ceilingTexture,
            leftWallTexture = leftWallTexture,
            rightWallTexture = rightWallTexture
        )

        assertTrue("GLB generation should succeed", success)
        assertTrue("GLB file should exist", glbFile.exists())
        assertTrue("GLB file should have content", glbFile.length() > 0)

        println("GLB file created: ${glbFile.absolutePath}")
        println("GLB file size: ${glbFile.length()} bytes")

        // Validate GLB structure
        validateGlbStructure(glbFile)

        // Copy to external storage for manual inspection
        val externalDir = context.getExternalFilesDir(null)!!
        val externalGlb = File(externalDir, "test_room.glb")
        glbFile.copyTo(externalGlb, overwrite = true)
        println("\nGLB saved to: ${externalGlb.absolutePath}")
        println("Pull with: adb pull ${externalGlb.absolutePath}")
        println("Validate at: https://gltf-viewer.donmccurdy.com/")

        // Cleanup test textures
        floorTexture.recycle()
        ceilingTexture.recycle()
        frontWallTexture.recycle()
        leftWallTexture.recycle()
        rightWallTexture.recycle()

        println("\nTEST PASSED - GLB generation working")
    }

    @Test
    fun testGlbWithCustomDimensions() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext

        // Create simple textures
        val texture = createColoredBitmap(Color.GRAY, 128, 128, "Test")

        val outputDir = File(context.cacheDir, "glb_test")
        outputDir.mkdirs()
        val glbFile = File(outputDir, "custom_room.glb")

        val generator = GlbGenerator()
        val dimensions = GlbGenerator.RoomDimensions(
            width = 6.0f,
            depth = 8.0f,
            height = 3.5f
        )

        val success = generator.generateGlb(
            outputFile = glbFile,
            dimensions = dimensions,
            frontWallTexture = texture,
            floorTexture = texture,
            ceilingTexture = texture,
            leftWallTexture = texture,
            rightWallTexture = texture
        )

        assertTrue("GLB generation with custom dimensions should succeed", success)
        assertTrue("GLB file should exist", glbFile.exists())

        // Validate structure
        validateGlbStructure(glbFile)

        println("Custom dimensions GLB size: ${glbFile.length()} bytes")
        println("TEST PASSED - Custom dimensions working")

        texture.recycle()
    }

    @Test
    fun testGlbJsonContent() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext

        val texture = createColoredBitmap(Color.BLUE, 64, 64, "Test")

        val outputDir = File(context.cacheDir, "glb_test")
        outputDir.mkdirs()
        val glbFile = File(outputDir, "json_test.glb")

        val generator = GlbGenerator()
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

        // Extract and validate JSON content
        val json = extractJsonFromGlb(glbFile)
        assertNotNull("Should extract JSON from GLB", json)

        println("JSON content length: ${json!!.length} characters")

        // Verify essential glTF properties
        assertTrue("JSON should contain asset version", json.contains("\"version\":\"2.0\""))
        assertTrue("JSON should contain generator", json.contains("\"generator\":\"Furnit Android\""))
        assertTrue("JSON should contain scenes", json.contains("\"scenes\""))
        assertTrue("JSON should contain nodes", json.contains("\"nodes\""))
        assertTrue("JSON should contain meshes", json.contains("\"meshes\""))
        assertTrue("JSON should contain materials", json.contains("\"materials\""))
        assertTrue("JSON should contain textures", json.contains("\"textures\""))
        assertTrue("JSON should contain images", json.contains("\"images\""))
        assertTrue("JSON should contain accessors", json.contains("\"accessors\""))
        assertTrue("JSON should contain bufferViews", json.contains("\"bufferViews\""))
        assertTrue("JSON should contain buffers", json.contains("\"buffers\""))

        // Verify 5 meshes for 5 planes
        val meshCount = "\"mesh\":".toRegex().findAll(json).count()
        assertEquals("Should have 5 mesh references", 5, meshCount)

        // Verify 5 materials
        val materialCount = "\"material\":".toRegex().findAll(json).count()
        assertEquals("Should have 5 material references", 5, materialCount)

        println("TEST PASSED - JSON content valid")

        texture.recycle()
    }

    private fun createColoredBitmap(color: Int, width: Int, height: Int, label: String): Bitmap {
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bitmap)
        canvas.drawColor(color)

        // Add label text
        val paint = Paint().apply {
            this.color = if (color == Color.WHITE) Color.BLACK else Color.WHITE
            textSize = width / 8f
            textAlign = Paint.Align.CENTER
        }
        canvas.drawText(label, width / 2f, height / 2f, paint)

        return bitmap
    }

    private fun validateGlbStructure(glbFile: File) {
        val bytes = glbFile.readBytes()
        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)

        // Check header (12 bytes)
        assertTrue("GLB file should be at least 12 bytes", bytes.size >= 12)

        val magic = buffer.getInt()
        assertEquals("Magic should be 'glTF'", GLB_MAGIC, magic)

        val version = buffer.getInt()
        assertEquals("Version should be 2", GLB_VERSION, version)

        val length = buffer.getInt()
        assertEquals("Length should match file size", bytes.size, length)

        println("GLB Header valid: magic=glTF, version=$version, length=$length")

        // Check JSON chunk
        assertTrue("Should have JSON chunk header", bytes.size >= 20)

        val jsonChunkLength = buffer.getInt()
        val jsonChunkType = buffer.getInt()
        assertEquals("First chunk should be JSON", JSON_CHUNK_TYPE, jsonChunkType)
        assertTrue("JSON chunk length should be positive", jsonChunkLength > 0)

        println("JSON chunk: length=$jsonChunkLength bytes")

        // Skip JSON content and check BIN chunk
        val binChunkStart = 12 + 8 + jsonChunkLength
        if (bytes.size > binChunkStart + 8) {
            buffer.position(binChunkStart)
            val binChunkLength = buffer.getInt()
            val binChunkType = buffer.getInt()
            assertEquals("Second chunk should be BIN", BIN_CHUNK_TYPE, binChunkType)
            assertTrue("BIN chunk length should be positive", binChunkLength > 0)

            println("BIN chunk: length=$binChunkLength bytes")
        }

        println("GLB structure validated successfully")
    }

    private fun extractJsonFromGlb(glbFile: File): String? {
        val bytes = glbFile.readBytes()
        if (bytes.size < 20) return null

        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)

        // Skip header
        buffer.position(12)

        val jsonChunkLength = buffer.getInt()
        val jsonChunkType = buffer.getInt()

        if (jsonChunkType != JSON_CHUNK_TYPE) return null

        val jsonBytes = ByteArray(jsonChunkLength)
        buffer.get(jsonBytes)

        return String(jsonBytes, Charsets.UTF_8).trim()
    }
}
