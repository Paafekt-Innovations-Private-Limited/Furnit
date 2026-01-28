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
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Tests for GlbGenerator with doubleSided materials
 * Verifies:
 * - DoubleSided property is set in materials
 * - Doubled room dimensions (8x9x5.6m)
 * - 5 textured planes (floor, ceiling, front, left, right walls)
 * - Valid GLB structure with correct materials
 */
@RunWith(AndroidJUnit4::class)
class GlbDoubleSidedTest {

    companion object {
        private const val GLB_MAGIC = 0x46546C67  // "glTF"
        private const val JSON_CHUNK_TYPE = 0x4E4F534A  // "JSON"
    }

    private lateinit var context: android.content.Context
    private lateinit var testDir: File

    @Before
    fun setup() {
        context = InstrumentationRegistry.getInstrumentation().targetContext
        testDir = File(context.cacheDir, "glb_doublesided_test")
        testDir.mkdirs()
    }

    @After
    fun teardown() {
        testDir.deleteRecursively()
    }

    @Test
    fun testDoubleSidedMaterialsInJson() {
        val generator = GlbGenerator()
        val texture = createTestBitmap()
        val glbFile = File(testDir, "doublesided_test.glb")

        val success = generator.generateGlb(
            outputFile = glbFile,
            dimensions = GlbGenerator.RoomDimensions(8.0f, 9.0f, 5.6f),
            frontWallTexture = texture,
            floorTexture = texture,
            ceilingTexture = texture,
            leftWallTexture = texture,
            rightWallTexture = texture
        )

        assertTrue("GLB generation should succeed", success)

        // Extract and check JSON
        val json = extractJsonFromGlb(glbFile)
        assertNotNull("Should extract JSON", json)

        // Count doubleSided occurrences - should be 5 (one per material)
        val doubleSidedCount = "\"doubleSided\":true".toRegex().findAll(json!!).count()
        assertEquals("Should have 5 doubleSided:true materials (floor, ceiling, 3 walls)", 5, doubleSidedCount)

        println("DoubleSided materials test PASSED - found $doubleSidedCount materials with doubleSided:true")
        texture.recycle()
    }

    @Test
    fun testDoubledRoomDimensions() {
        // Verify default dimensions are doubled (8x9x5.6m vs original 4x4.5x2.8m)
        val dimensions = GlbGenerator.RoomDimensions()
        assertEquals("Default width should be 8.0", 8.0f, dimensions.width)
        assertEquals("Default depth should be 9.0", 9.0f, dimensions.depth)
        assertEquals("Default height should be 5.6", 5.6f, dimensions.height)

        println("Doubled room dimensions test PASSED")
    }

    @Test
    fun testGlbContains5Meshes() {
        val generator = GlbGenerator()
        val texture = createTestBitmap()
        val glbFile = File(testDir, "mesh_count_test.glb")

        generator.generateGlb(
            outputFile = glbFile,
            dimensions = GlbGenerator.RoomDimensions(),
            frontWallTexture = texture,
            floorTexture = texture,
            ceilingTexture = texture,
            leftWallTexture = texture,
            rightWallTexture = texture
        )

        val json = extractJsonFromGlb(glbFile)!!

        // Count mesh references
        val meshCount = "\"mesh\":".toRegex().findAll(json).count()
        assertEquals("Should have 5 meshes (floor, ceiling, front, left, right)", 5, meshCount)

        // Count material references
        val materialCount = "\"material\":".toRegex().findAll(json).count()
        assertEquals("Should have 5 materials", 5, materialCount)

        println("5 meshes/materials test PASSED")
        texture.recycle()
    }

    @Test
    fun testMaterialNames() {
        val generator = GlbGenerator()
        val texture = createTestBitmap()
        val glbFile = File(testDir, "material_names_test.glb")

        generator.generateGlb(
            outputFile = glbFile,
            dimensions = GlbGenerator.RoomDimensions(),
            frontWallTexture = texture,
            floorTexture = texture,
            ceilingTexture = texture,
            leftWallTexture = texture,
            rightWallTexture = texture
        )

        val json = extractJsonFromGlb(glbFile)!!

        // Check material names are present
        assertTrue("Should have floor material", json.contains("floor_material"))
        assertTrue("Should have ceiling material", json.contains("ceiling_material"))
        assertTrue("Should have front_wall material", json.contains("front_wall_material"))
        assertTrue("Should have left_wall material", json.contains("left_wall_material"))
        assertTrue("Should have right_wall material", json.contains("right_wall_material"))

        println("Material names test PASSED")
        texture.recycle()
    }

    @Test
    fun testDoubleSidedPropertyPosition() {
        val generator = GlbGenerator()
        val texture = createTestBitmap()
        val glbFile = File(testDir, "doublesided_position_test.glb")

        generator.generateGlb(
            outputFile = glbFile,
            dimensions = GlbGenerator.RoomDimensions(),
            frontWallTexture = texture,
            floorTexture = texture,
            ceilingTexture = texture,
            leftWallTexture = texture,
            rightWallTexture = texture
        )

        val json = extractJsonFromGlb(glbFile)!!

        // doubleSided should appear after pbrMetallicRoughness and before name
        // Pattern: ...},"doubleSided":true,"name":"xxx_material"
        val materialPattern = "\"pbrMetallicRoughness\".*?\"doubleSided\":true.*?\"name\":\"\\w+_material\"".toRegex()
        val matches = materialPattern.findAll(json).count()

        assertEquals("All 5 materials should have doubleSided in correct position", 5, matches)
        println("DoubleSided property position test PASSED")
        texture.recycle()
    }

    @Test
    fun testGlbGeneratorString() {
        val generator = GlbGenerator()
        val texture = createTestBitmap()
        val glbFile = File(testDir, "generator_string_test.glb")

        generator.generateGlb(
            outputFile = glbFile,
            dimensions = GlbGenerator.RoomDimensions(),
            frontWallTexture = texture,
            floorTexture = texture,
            ceilingTexture = texture,
            leftWallTexture = texture,
            rightWallTexture = texture
        )

        val json = extractJsonFromGlb(glbFile)!!
        assertTrue("Should have Furnit Android generator string", json.contains("\"generator\":\"Furnit Android\""))

        println("Generator string test PASSED")
        texture.recycle()
    }

    @Test
    fun testGlbVersion() {
        val generator = GlbGenerator()
        val texture = createTestBitmap()
        val glbFile = File(testDir, "version_test.glb")

        generator.generateGlb(
            outputFile = glbFile,
            dimensions = GlbGenerator.RoomDimensions(),
            frontWallTexture = texture,
            floorTexture = texture,
            ceilingTexture = texture,
            leftWallTexture = texture,
            rightWallTexture = texture
        )

        // Check GLB header version
        val bytes = glbFile.readBytes()
        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)

        val magic = buffer.getInt()
        assertEquals("Magic should be glTF", GLB_MAGIC, magic)

        val version = buffer.getInt()
        assertEquals("Version should be 2", 2, version)

        // Also check JSON version
        val json = extractJsonFromGlb(glbFile)!!
        assertTrue("JSON should specify version 2.0", json.contains("\"version\":\"2.0\""))

        println("GLB version test PASSED")
        texture.recycle()
    }

    private fun createTestBitmap(): Bitmap {
        return Bitmap.createBitmap(128, 128, Bitmap.Config.ARGB_8888).apply {
            eraseColor(Color.GRAY)
        }
    }

    private fun extractJsonFromGlb(glbFile: File): String? {
        val bytes = glbFile.readBytes()
        if (bytes.size < 20) return null

        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)

        // Skip header (12 bytes)
        buffer.position(12)

        val jsonChunkLength = buffer.getInt()
        val jsonChunkType = buffer.getInt()

        if (jsonChunkType != JSON_CHUNK_TYPE) return null

        val jsonBytes = ByteArray(jsonChunkLength)
        buffer.get(jsonBytes)

        return String(jsonBytes, Charsets.UTF_8).trim()
    }
}
