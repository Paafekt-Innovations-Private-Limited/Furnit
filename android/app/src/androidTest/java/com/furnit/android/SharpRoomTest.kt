package com.furnit.android

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.furnit.android.services.SharpService
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

/**
 * Test Sharp Room PLY generation from room.jpeg
 * Tests SharpService which generates 3D Gaussian splats from a single photo.
 * Validates PLY file format, room dimensions, and output quality.
 */
@RunWith(AndroidJUnit4::class)
class SharpRoomTest {

    @Test
    fun testSharpRoomGenerationFromRoomImage() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext

        // Load room.jpeg from assets
        val bitmap = context.assets.open("room.jpeg").use { input ->
            BitmapFactory.decodeStream(input)
        }
        assertNotNull("Failed to load room.jpeg", bitmap)
        println("Loaded room.jpeg: ${bitmap.width}x${bitmap.height}")

        // Initialize SharpService
        val sharpService = SharpService.getInstance(context)

        // Generate Gaussian splats
        val latch = CountDownLatch(1)
        var generationResult: SharpService.GenerationResult? = null
        var errorMessage: String? = null

        sharpService.generateGaussians(bitmap, object : SharpService.ProgressCallback {
            override fun onProgress(progress: Float, message: String) {
                println("Sharp generation: ${(progress * 100).toInt()}% - $message")
            }

            override fun onComplete(result: SharpService.GenerationResult) {
                generationResult = result
                latch.countDown()
            }

            override fun onError(message: String) {
                errorMessage = message
                latch.countDown()
            }
        })

        // Wait for generation (15 minutes for FP32 split model)
        val completed = latch.await(900, TimeUnit.SECONDS)
        assertTrue("Sharp generation timed out", completed)
        assertNull("Sharp generation failed: $errorMessage", errorMessage)
        assertNotNull("Generation result should not be null", generationResult)

        val result = generationResult!!
        println("\n=== Sharp Room Generation Results ===")
        println("PLY file: ${result.plyFile.absolutePath}")
        println("PLY file size: ${result.plyFile.length()} bytes")
        println("Classic PLY: ${result.classicPlyFile.absolutePath}")
        println("Room dimensions: ${result.roomWidth}m x ${result.roomHeight}m x ${result.roomDepth}m")

        // Verify PLY file was created
        assertTrue("PLY file should exist", result.plyFile.exists())
        assertTrue("PLY file should have content", result.plyFile.length() > 1000)
        assertTrue("Classic PLY file should exist", result.classicPlyFile.exists())

        // Verify PLY header format (3DGS standard with spherical harmonics)
        val plyHeader = result.plyFile.readText().take(1500)
        assertTrue("PLY should have valid header", plyHeader.contains("ply"))
        assertTrue("PLY should have binary format", plyHeader.contains("format binary_little_endian"))
        assertTrue("PLY should have vertex element", plyHeader.contains("element vertex"))
        assertTrue("PLY should have position properties", plyHeader.contains("property float x"))
        assertTrue("PLY should have f_dc (SH colors)", plyHeader.contains("property float f_dc_0"))
        assertTrue("PLY should have scale properties", plyHeader.contains("property float scale_0"))
        assertTrue("PLY should have rotation properties", plyHeader.contains("property float rot_0"))
        assertTrue("PLY should have opacity", plyHeader.contains("property float opacity"))
        println("\nPLY header:\n${plyHeader.take(800)}...")

        // Verify room dimensions are reasonable
        assertTrue("Room width should be > 0", result.roomWidth > 0)
        assertTrue("Room height should be > 0", result.roomHeight > 0)
        assertTrue("Room depth should be > 0", result.roomDepth > 0)
        assertTrue("Room width should be reasonable (< 20m)", result.roomWidth < 20)
        assertTrue("Room height should be reasonable (< 10m)", result.roomHeight < 10)
        assertTrue("Room depth should be reasonable (< 20m)", result.roomDepth < 20)

        println("\nTEST PASSED - Sharp room PLY generation working")
    }

    @Test
    fun testSharpPLYVertexCount() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext

        // Load room.jpeg
        val bitmap = context.assets.open("room.jpeg").use { input ->
            BitmapFactory.decodeStream(input)
        }
        assertNotNull("Failed to load room.jpeg", bitmap)

        val sharpService = SharpService.getInstance(context)

        val latch = CountDownLatch(1)
        var result: SharpService.GenerationResult? = null

        sharpService.generateGaussians(bitmap, object : SharpService.ProgressCallback {
            override fun onProgress(progress: Float, message: String) {}
            override fun onComplete(r: SharpService.GenerationResult) {
                result = r
                latch.countDown()
            }
            override fun onError(message: String) {
                latch.countDown()
            }
        })

        latch.await(180, TimeUnit.SECONDS)
        assertNotNull("Result should not be null", result)

        // Parse vertex count from PLY header
        val plyHeader = result!!.plyFile.readText().take(500)
        val vertexMatch = Regex("element vertex (\\d+)").find(plyHeader)
        assertNotNull("Should find vertex count in PLY", vertexMatch)

        val vertexCount = vertexMatch!!.groupValues[1].toInt()
        println("PLY vertex count: $vertexCount")

        // Should have substantial number of vertices for a room
        assertTrue("Should have at least 1000 vertices", vertexCount >= 1000)
        assertTrue("Should not exceed 100000 vertices", vertexCount <= 100000)

        // Verify file size matches expected vertex data
        // 3DGS format: 62 floats per vertex = 248 bytes per vertex
        val plyContent = result!!.plyFile.readBytes()
        val headerEndStr = "end_header\n"
        var headerEnd = 0
        for (i in 0 until plyContent.size - headerEndStr.length) {
            if (String(plyContent, i, headerEndStr.length) == headerEndStr) {
                headerEnd = i + headerEndStr.length
                break
            }
        }
        val expectedDataSize = vertexCount * 62 * 4  // 62 floats * 4 bytes
        val actualFileSize = result!!.plyFile.length()
        val actualDataSize = actualFileSize - headerEnd

        println("Header size: $headerEnd bytes")
        println("Expected data size: $expectedDataSize bytes (62 floats/vertex)")
        println("Actual data size: $actualDataSize bytes")

        // Allow some tolerance for header size estimation
        assertTrue("Data size should be close to expected",
            actualDataSize >= expectedDataSize * 0.9 && actualDataSize <= expectedDataSize * 1.1)

        println("\nTEST PASSED - PLY vertex count validation")
    }

    @Test
    fun testSharpRoomThumbnailGeneration() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext

        // Load room.jpeg
        val bitmap = context.assets.open("room.jpeg").use { input ->
            BitmapFactory.decodeStream(input)
        }
        assertNotNull("Failed to load room.jpeg", bitmap)

        val sharpService = SharpService.getInstance(context)

        val latch = CountDownLatch(1)
        var result: SharpService.GenerationResult? = null

        sharpService.generateGaussians(bitmap, object : SharpService.ProgressCallback {
            override fun onProgress(progress: Float, message: String) {}
            override fun onComplete(r: SharpService.GenerationResult) {
                result = r
                latch.countDown()
            }
            override fun onError(message: String) {
                latch.countDown()
            }
        })

        latch.await(180, TimeUnit.SECONDS)
        assertNotNull("Result should not be null", result)

        // Check thumbnail was saved
        val roomFolder = result!!.plyFile.parentFile
        assertNotNull("Room folder should exist", roomFolder)

        val thumbnailFile = File(roomFolder, "thumbnail.png")
        assertTrue("Thumbnail file should exist", thumbnailFile.exists())
        assertTrue("Thumbnail should have content", thumbnailFile.length() > 1000)

        // Verify thumbnail is valid image
        val thumbnailBitmap = BitmapFactory.decodeFile(thumbnailFile.absolutePath)
        assertNotNull("Thumbnail should be valid image", thumbnailBitmap)
        println("Thumbnail: ${thumbnailBitmap.width}x${thumbnailBitmap.height}")

        // Check metadata was saved
        val metadataFile = File(roomFolder, "metadata.txt")
        assertTrue("Metadata file should exist", metadataFile.exists())
        val metadata = metadataFile.readText()
        assertTrue("Metadata should have name", metadata.contains("name="))
        assertTrue("Metadata should have type=sharp", metadata.contains("type=sharp"))
        println("Metadata:\n$metadata")

        println("\nTEST PASSED - Sharp room thumbnail and metadata generation")
    }

    @Test
    fun testSharpRoomBoundingBox() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext

        // Load room.jpeg
        val bitmap = context.assets.open("room.jpeg").use { input ->
            BitmapFactory.decodeStream(input)
        }
        assertNotNull("Failed to load room.jpeg", bitmap)
        println("Input image: ${bitmap.width}x${bitmap.height}")

        val sharpService = SharpService.getInstance(context)

        val latch = CountDownLatch(1)
        var result: SharpService.GenerationResult? = null

        sharpService.generateGaussians(bitmap, object : SharpService.ProgressCallback {
            override fun onProgress(progress: Float, message: String) {
                println("Progress: ${(progress * 100).toInt()}% - $message")
            }
            override fun onComplete(r: SharpService.GenerationResult) {
                result = r
                latch.countDown()
            }
            override fun onError(message: String) {
                println("Error: $message")
                latch.countDown()
            }
        })

        latch.await(180, TimeUnit.SECONDS)
        assertNotNull("Result should not be null", result)

        // Validate bounding box (room dimensions)
        val width = result!!.roomWidth
        val height = result!!.roomHeight
        val depth = result!!.roomDepth

        println("\n=== Room Bounding Box ===")
        println("Width (X): ${String.format("%.2f", width)} meters")
        println("Height (Y): ${String.format("%.2f", height)} meters")
        println("Depth (Z): ${String.format("%.2f", depth)} meters")
        println("Volume: ${String.format("%.2f", width * height * depth)} cubic meters")

        // Room should have reasonable proportions
        val aspectRatio = width / height
        println("Width/Height ratio: ${String.format("%.2f", aspectRatio)}")
        assertTrue("Room aspect ratio should be reasonable", aspectRatio > 0.5 && aspectRatio < 3.0)

        // Save result info for inspection
        val externalDir = context.getExternalFilesDir(null)!!
        val infoFile = File(externalDir, "sharp_room_info.txt")
        infoFile.writeText("""
            Sharp Room Generation Results
            ==============================
            Input: room.jpeg (${bitmap.width}x${bitmap.height})
            PLY: ${result!!.plyFile.absolutePath}
            PLY Size: ${result!!.plyFile.length()} bytes

            Bounding Box:
            - Width: ${String.format("%.2f", width)} m
            - Height: ${String.format("%.2f", height)} m
            - Depth: ${String.format("%.2f", depth)} m
            - Volume: ${String.format("%.2f", width * height * depth)} m³
        """.trimIndent())
        println("\nSaved info to: ${infoFile.absolutePath}")
        println("adb pull ${infoFile.absolutePath}")

        println("\nTEST PASSED - Sharp room bounding box validation")
    }

    @Test
    fun testSharpRoomSaveAndLabel() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext

        // Load room.jpeg
        val bitmap = context.assets.open("room.jpeg").use { input ->
            BitmapFactory.decodeStream(input)
        }
        assertNotNull("Failed to load room.jpeg", bitmap)

        val sharpService = SharpService.getInstance(context)

        val latch = CountDownLatch(1)
        var result: SharpService.GenerationResult? = null

        sharpService.generateGaussians(bitmap, object : SharpService.ProgressCallback {
            override fun onProgress(progress: Float, message: String) {}
            override fun onComplete(r: SharpService.GenerationResult) {
                result = r
                latch.countDown()
            }
            override fun onError(message: String) {
                latch.countDown()
            }
        })

        latch.await(180, TimeUnit.SECONDS)
        assertNotNull("Result should not be null", result)

        // Create annotated image with room info label
        val annotated = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(annotated)

        // Draw label background
        val bgPaint = Paint().apply {
            color = Color.argb(200, 0, 0, 0)
            style = Paint.Style.FILL
        }
        canvas.drawRect(0f, 0f, annotated.width.toFloat(), 100f, bgPaint)

        // Draw room info label
        val textPaint = Paint().apply {
            color = Color.WHITE
            textSize = 40f
            isAntiAlias = true
        }
        val label = "Sharp Room: ${String.format("%.1f", result!!.roomWidth)}m x ${String.format("%.1f", result!!.roomHeight)}m x ${String.format("%.1f", result!!.roomDepth)}m"
        canvas.drawText(label, 20f, 60f, textPaint)

        // Draw bounding box indicator
        val boxPaint = Paint().apply {
            color = Color.GREEN
            style = Paint.Style.STROKE
            strokeWidth = 8f
        }
        val margin = 50f
        canvas.drawRect(margin, 120f, annotated.width - margin, annotated.height - margin, boxPaint)

        // Save annotated image
        val externalDir = context.getExternalFilesDir(null)!!
        val outputFile = File(externalDir, "room_sharp_annotated.png")
        FileOutputStream(outputFile).use { out ->
            annotated.compress(Bitmap.CompressFormat.PNG, 100, out)
        }

        println("\n=== Sharp Room Annotated Image ===")
        println("Label: $label")
        println("Saved to: ${outputFile.absolutePath}")
        println("adb pull ${outputFile.absolutePath}")

        println("\nTEST PASSED - Sharp room label and save")
    }
}
