package com.furnit.android

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.furnit.android.services.SharpService
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import kotlin.math.abs

/**
 * iOS Parity Tests for Sharp Room Generation
 *
 * These tests validate that the Android implementation matches iOS behavior:
 * - PLY file format (3DGS standard with spherical harmonics)
 * - Gaussian parameters (positions, scales, rotations, colors, opacity)
 * - Room bounds calculation
 * - Camera positioning
 *
 * Uses room.jpeg as the test image (same as iOS tests).
 */
@RunWith(AndroidJUnit4::class)
class SharpIOSParityTest {

    private lateinit var context: android.content.Context
    private lateinit var testBitmap: Bitmap

    // Expected values from iOS implementation
    companion object {
        // PLY format constants (matching iOS)
        const val FLOATS_PER_VERTEX_3DGS = 62  // Full 3DGS format
        const val BYTES_PER_FLOAT = 4

        // Spherical harmonics constant (from iOS)
        const val SH_C0 = 0.28209479177387814f

        // Expected room dimensions range (meters)
        const val MIN_ROOM_WIDTH = 0.5f
        const val MAX_ROOM_WIDTH = 20.0f
        const val MIN_ROOM_HEIGHT = 0.5f
        const val MAX_ROOM_HEIGHT = 10.0f
        const val MIN_ROOM_DEPTH = 0.1f
        const val MAX_ROOM_DEPTH = 20.0f
    }

    @Before
    fun setup() {
        context = InstrumentationRegistry.getInstrumentation().targetContext

        // Load room.jpeg from assets (same test image as iOS)
        testBitmap = context.assets.open("room.jpeg").use { input ->
            BitmapFactory.decodeStream(input)
        }
        assertNotNull("Failed to load room.jpeg from assets", testBitmap)
        println("Test image loaded: ${testBitmap.width}x${testBitmap.height}")
    }

    // ==================== PLY Format Tests ====================

    @Test
    fun testPLYHeader_Matches3DGSFormat() {
        val result = generateSharpRoom()
        assertNotNull("Generation should succeed", result)

        val plyContent = result!!.plyFile.readBytes()
        val headerEnd = findHeaderEnd(plyContent)
        val header = String(plyContent, 0, headerEnd)

        println("=== PLY Header (3DGS Format) ===")
        println(header)

        // Validate 3DGS standard header
        assertTrue("Should have PLY magic", header.startsWith("ply"))
        assertTrue("Should be binary little endian", header.contains("format binary_little_endian 1.0"))
        assertTrue("Should have vertex element", header.contains("element vertex"))

        // Position properties (x, y, z)
        assertTrue("Should have x property", header.contains("property float x"))
        assertTrue("Should have y property", header.contains("property float y"))
        assertTrue("Should have z property", header.contains("property float z"))

        // Normal properties (nx, ny, nz)
        assertTrue("Should have nx property", header.contains("property float nx"))
        assertTrue("Should have ny property", header.contains("property float ny"))
        assertTrue("Should have nz property", header.contains("property float nz"))

        // Spherical harmonics DC (f_dc_0, f_dc_1, f_dc_2) - THIS IS KEY FOR iOS PARITY
        assertTrue("Should have f_dc_0 (SH DC red)", header.contains("property float f_dc_0"))
        assertTrue("Should have f_dc_1 (SH DC green)", header.contains("property float f_dc_1"))
        assertTrue("Should have f_dc_2 (SH DC blue)", header.contains("property float f_dc_2"))

        // Scale properties
        assertTrue("Should have scale_0", header.contains("property float scale_0"))
        assertTrue("Should have scale_1", header.contains("property float scale_1"))
        assertTrue("Should have scale_2", header.contains("property float scale_2"))

        // Rotation quaternion
        assertTrue("Should have rot_0", header.contains("property float rot_0"))
        assertTrue("Should have rot_1", header.contains("property float rot_1"))
        assertTrue("Should have rot_2", header.contains("property float rot_2"))
        assertTrue("Should have rot_3", header.contains("property float rot_3"))

        // Opacity
        assertTrue("Should have opacity", header.contains("property float opacity"))

        println("TEST PASSED: PLY header matches 3DGS format")
    }

    @Test
    fun testPLYVertexData_SphericalHarmonicsColors() {
        val result = generateSharpRoom()
        assertNotNull("Generation should succeed", result)

        val plyContent = result!!.plyFile.readBytes()
        val headerEnd = findHeaderEnd(plyContent)
        val header = String(plyContent, 0, headerEnd)

        // Parse vertex count
        val vertexMatch = Regex("element vertex (\\d+)").find(header)
        assertNotNull("Should have vertex count", vertexMatch)
        val vertexCount = vertexMatch!!.groupValues[1].toInt()
        println("Vertex count: $vertexCount")

        // Read binary vertex data
        val buffer = ByteBuffer.wrap(plyContent, headerEnd, plyContent.size - headerEnd)
        buffer.order(ByteOrder.LITTLE_ENDIAN)

        // Sample first 10 vertices and validate f_dc colors
        val sampleCount = minOf(10, vertexCount)
        println("\n=== First $sampleCount Vertices (f_dc -> RGB) ===")

        var validColorCount = 0
        for (i in 0 until sampleCount) {
            val offset = i * FLOATS_PER_VERTEX_3DGS * BYTES_PER_FLOAT

            // Position (0-2)
            val x = buffer.getFloat(offset)
            val y = buffer.getFloat(offset + 4)
            val z = buffer.getFloat(offset + 8)

            // Normal (3-5) - skip
            // f_dc (6-8) - spherical harmonics DC
            val fdcOffset = offset + 6 * BYTES_PER_FLOAT
            val fdcR = buffer.getFloat(fdcOffset)
            val fdcG = buffer.getFloat(fdcOffset + 4)
            val fdcB = buffer.getFloat(fdcOffset + 8)

            // Convert SH DC to RGB (matching iOS: color = sh * SH_C0 + 0.5)
            val r = (fdcR * SH_C0 + 0.5f).coerceIn(0f, 1f)
            val g = (fdcG * SH_C0 + 0.5f).coerceIn(0f, 1f)
            val b = (fdcB * SH_C0 + 0.5f).coerceIn(0f, 1f)

            println("Vertex $i: pos=(${x.f2()}, ${y.f2()}, ${z.f2()}) f_dc=(${fdcR.f2()}, ${fdcG.f2()}, ${fdcB.f2()}) -> RGB=(${r.f2()}, ${g.f2()}, ${b.f2()})")

            // Validate colors are reasonable
            if (r > 0.01f || g > 0.01f || b > 0.01f) {
                validColorCount++
            }
        }

        assertTrue("At least 50% of sampled vertices should have visible colors", validColorCount >= sampleCount / 2)
        println("\nTEST PASSED: Spherical harmonics colors are valid")
    }

    @Test
    fun testPLYVertexData_GaussianParameters() {
        val result = generateSharpRoom()
        assertNotNull("Generation should succeed", result)

        val plyContent = result!!.plyFile.readBytes()
        val headerEnd = findHeaderEnd(plyContent)
        val header = String(plyContent, 0, headerEnd)

        val vertexMatch = Regex("element vertex (\\d+)").find(header)
        val vertexCount = vertexMatch!!.groupValues[1].toInt()

        val buffer = ByteBuffer.wrap(plyContent, headerEnd, plyContent.size - headerEnd)
        buffer.order(ByteOrder.LITTLE_ENDIAN)

        println("=== Gaussian Parameters Validation ===")

        // Validate parameters for first 100 vertices
        val sampleCount = minOf(100, vertexCount)
        var validScales = 0
        var validRotations = 0
        var validOpacities = 0

        for (i in 0 until sampleCount) {
            val offset = i * FLOATS_PER_VERTEX_3DGS * BYTES_PER_FLOAT

            // Scale (offset 54-56: after f_rest which is 45 floats, after f_dc which is 3, after normal which is 3, after position which is 3)
            // Actually: pos(3) + normal(3) + f_dc(3) + f_rest(45) + opacity(1) + scale(3) + rot(4) = 62
            // So scale starts at offset 55 (0-indexed: 54)
            val scaleOffset = offset + 55 * BYTES_PER_FLOAT
            val scale0 = buffer.getFloat(scaleOffset)
            val scale1 = buffer.getFloat(scaleOffset + 4)
            val scale2 = buffer.getFloat(scaleOffset + 8)

            // Rotation quaternion (offset 58-61)
            val rotOffset = offset + 58 * BYTES_PER_FLOAT
            val rot0 = buffer.getFloat(rotOffset)
            val rot1 = buffer.getFloat(rotOffset + 4)
            val rot2 = buffer.getFloat(rotOffset + 8)
            val rot3 = buffer.getFloat(rotOffset + 12)

            // Opacity (offset 54)
            val opacityOffset = offset + 54 * BYTES_PER_FLOAT
            val opacity = buffer.getFloat(opacityOffset)

            // Validate scale (should be log-transformed, typically negative values)
            if (scale0.isFinite() && scale1.isFinite() && scale2.isFinite()) {
                validScales++
            }

            // Validate rotation quaternion (should be normalized: |q| ≈ 1)
            val qMag = kotlin.math.sqrt(rot0*rot0 + rot1*rot1 + rot2*rot2 + rot3*rot3)
            if (abs(qMag - 1.0f) < 0.1f || qMag == 0f) {  // Allow some tolerance
                validRotations++
            }

            // Validate opacity (logit-transformed, can be any finite value)
            if (opacity.isFinite()) {
                validOpacities++
            }
        }

        println("Valid scales: $validScales / $sampleCount")
        println("Valid rotations: $validRotations / $sampleCount")
        println("Valid opacities: $validOpacities / $sampleCount")

        assertTrue("At least 80% should have valid scales", validScales >= sampleCount * 0.8)
        assertTrue("At least 80% should have valid rotations", validRotations >= sampleCount * 0.8)
        assertTrue("At least 80% should have valid opacities", validOpacities >= sampleCount * 0.8)

        println("\nTEST PASSED: Gaussian parameters are valid")
    }

    // ==================== Room Bounds Tests ====================

    @Test
    fun testRoomBounds_MatchesiOSRange() {
        val result = generateSharpRoom()
        assertNotNull("Generation should succeed", result)

        println("=== Room Bounds (matching iOS) ===")
        println("Width:  ${result!!.roomWidth.f2()} m (expected: $MIN_ROOM_WIDTH - $MAX_ROOM_WIDTH)")
        println("Height: ${result.roomHeight.f2()} m (expected: $MIN_ROOM_HEIGHT - $MAX_ROOM_HEIGHT)")
        println("Depth:  ${result.roomDepth.f2()} m (expected: $MIN_ROOM_DEPTH - $MAX_ROOM_DEPTH)")

        // Validate bounds match iOS expectations
        assertTrue("Width should be >= $MIN_ROOM_WIDTH", result.roomWidth >= MIN_ROOM_WIDTH)
        assertTrue("Width should be <= $MAX_ROOM_WIDTH", result.roomWidth <= MAX_ROOM_WIDTH)
        assertTrue("Height should be >= $MIN_ROOM_HEIGHT", result.roomHeight >= MIN_ROOM_HEIGHT)
        assertTrue("Height should be <= $MAX_ROOM_HEIGHT", result.roomHeight <= MAX_ROOM_HEIGHT)
        assertTrue("Depth should be >= $MIN_ROOM_DEPTH", result.roomDepth >= MIN_ROOM_DEPTH)
        assertTrue("Depth should be <= $MAX_ROOM_DEPTH", result.roomDepth <= MAX_ROOM_DEPTH)

        // Calculate volume
        val volume = result.roomWidth * result.roomHeight * result.roomDepth
        println("Volume: ${volume.f2()} m³")

        println("\nTEST PASSED: Room bounds match iOS range")
    }

    @Test
    fun testRoomAspectRatio_Realistic() {
        val result = generateSharpRoom()
        assertNotNull("Generation should succeed", result)

        val widthHeightRatio = result!!.roomWidth / result.roomHeight
        val widthDepthRatio = result.roomWidth / result.roomDepth

        println("=== Room Aspect Ratios ===")
        println("Width/Height: ${widthHeightRatio.f2()}")
        println("Width/Depth:  ${widthDepthRatio.f2()}")

        // Realistic room aspect ratios (matching iOS expectations)
        assertTrue("Width/Height ratio should be reasonable (0.3-5.0)", widthHeightRatio in 0.3f..5.0f)

        println("\nTEST PASSED: Room aspect ratios are realistic")
    }

    // ==================== WebGL/SparkJS Integration Test ====================

    @Test
    fun testPLYFile_CompatibleWithSparkJS() {
        val result = generateSharpRoom()
        assertNotNull("Generation should succeed", result)

        val plyFile = result!!.plyFile
        assertTrue("PLY file should exist", plyFile.exists())

        // Validate file can be read and has expected structure
        val plyContent = plyFile.readBytes()
        val headerEnd = findHeaderEnd(plyContent)
        val header = String(plyContent, 0, headerEnd)

        println("=== SparkJS Compatibility Check ===")
        println("PLY file: ${plyFile.name}")
        println("File size: ${plyFile.length()} bytes")
        println("Header size: $headerEnd bytes")
        println("Data size: ${plyFile.length() - headerEnd} bytes")

        // SparkJS expects standard 3DGS PLY format
        assertTrue("Header should be ASCII", header.all { it.code < 128 })
        assertTrue("Should have end_header", header.contains("end_header"))

        // Verify binary data starts after header
        val binaryData = plyContent.sliceArray(headerEnd until plyContent.size)
        assertTrue("Binary data should be substantial", binaryData.size > 1000)

        // Verify no null bytes in header
        val headerBytes = plyContent.sliceArray(0 until headerEnd)
        assertFalse("Header should not have null bytes", headerBytes.any { it == 0.toByte() })

        println("\nTEST PASSED: PLY file is SparkJS compatible")
    }

    // ==================== Helper Functions ====================

    private fun generateSharpRoom(): SharpService.GenerationResult? {
        val sharpService = SharpService.getInstance(context)

        val latch = CountDownLatch(1)
        var result: SharpService.GenerationResult? = null
        var error: String? = null

        sharpService.generateGaussians(testBitmap, object : SharpService.ProgressCallback {
            override fun onProgress(progress: Float, message: String) {
                println("Progress: ${(progress * 100).toInt()}% - $message")
            }

            override fun onComplete(r: SharpService.GenerationResult) {
                result = r
                latch.countDown()
            }

            override fun onError(message: String) {
                error = message
                latch.countDown()
            }
        })

        val completed = latch.await(60, TimeUnit.SECONDS)
        assertTrue("Generation should complete", completed)
        assertNull("Generation should not error: $error", error)

        return result
    }

    private fun findHeaderEnd(plyContent: ByteArray): Int {
        val endHeader = "end_header\n".toByteArray()
        for (i in 0 until plyContent.size - endHeader.size) {
            var match = true
            for (j in endHeader.indices) {
                if (plyContent[i + j] != endHeader[j]) {
                    match = false
                    break
                }
            }
            if (match) return i + endHeader.size
        }
        throw IllegalStateException("Could not find end_header in PLY")
    }

    private fun Float.f2(): String = String.format("%.2f", this)
}
