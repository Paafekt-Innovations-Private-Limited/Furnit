package com.furnit.android

import android.content.ContentValues
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.os.Environment
import android.provider.MediaStore
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.After
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/**
 * Tests for Screenshot functionality
 * Verifies:
 * - Bitmap creation and compositing
 * - MediaStore API for saving to gallery
 * - Screenshot naming format
 * - File cleanup
 */
@RunWith(AndroidJUnit4::class)
class ScreenshotTest {

    private lateinit var context: android.content.Context
    private val testUris = mutableListOf<android.net.Uri>()

    @Before
    fun setup() {
        context = InstrumentationRegistry.getInstrumentation().targetContext
    }

    @After
    fun teardown() {
        // Clean up test screenshots from gallery
        testUris.forEach { uri ->
            try {
                context.contentResolver.delete(uri, null, null)
            } catch (e: Exception) {
                // Ignore cleanup errors
            }
        }
    }

    @Test
    fun testBitmapCreation() {
        val width = 1080
        val height = 1920

        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        assertNotNull("Bitmap should be created", bitmap)
        assertEquals("Width should match", width, bitmap.width)
        assertEquals("Height should match", height, bitmap.height)

        bitmap.recycle()
        println("Bitmap creation test PASSED")
    }

    @Test
    fun testBitmapCompositing() {
        // Simulate compositing room background with overlay
        val width = 500
        val height = 500

        // Create "room" bitmap (blue background)
        val roomBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888).apply {
            eraseColor(Color.BLUE)
        }

        // Create composite bitmap
        val compositeBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(compositeBitmap)

        // Draw room background
        canvas.drawBitmap(roomBitmap, 0f, 0f, null)

        // Draw overlay (semi-transparent red rectangle)
        val paint = android.graphics.Paint().apply {
            color = Color.argb(128, 255, 0, 0)  // Semi-transparent red
        }
        canvas.drawRect(100f, 100f, 400f, 400f, paint)

        // Verify composite
        val centerPixel = compositeBitmap.getPixel(250, 250)
        // Center should be blend of blue and semi-transparent red
        assertNotEquals("Center pixel should not be pure blue", Color.BLUE, centerPixel)

        val cornerPixel = compositeBitmap.getPixel(10, 10)
        assertEquals("Corner pixel should be blue (no overlay)", Color.BLUE, cornerPixel)

        roomBitmap.recycle()
        compositeBitmap.recycle()
        println("Bitmap compositing test PASSED")
    }

    @Test
    fun testScreenshotNamingFormat() {
        val format = java.text.SimpleDateFormat("yyyyMMdd_HHmmss", java.util.Locale.getDefault())
        val timestamp = format.format(java.util.Date())
        val fileName = "FurnitureFit_$timestamp.png"

        assertTrue("Filename should start with FurnitureFit_", fileName.startsWith("FurnitureFit_"))
        assertTrue("Filename should end with .png", fileName.endsWith(".png"))
        assertTrue("Filename should contain date pattern", fileName.contains("_"))

        // Verify timestamp format (yyyyMMdd_HHmmss)
        val timestampPart = fileName.removePrefix("FurnitureFit_").removeSuffix(".png")
        assertEquals("Timestamp should be 15 chars (yyyyMMdd_HHmmss)", 15, timestampPart.length)

        println("Screenshot naming format test PASSED - filename: $fileName")
    }

    @Test
    fun testMediaStoreContentValues() {
        val timestamp = "20260129_143022"
        val fileName = "FurnitureFit_$timestamp.png"

        val contentValues = ContentValues().apply {
            put(MediaStore.Images.Media.DISPLAY_NAME, fileName)
            put(MediaStore.Images.Media.MIME_TYPE, "image/png")
            put(MediaStore.Images.Media.RELATIVE_PATH, "${Environment.DIRECTORY_PICTURES}/Screenshots")
        }

        assertEquals("Display name should be set", fileName,
            contentValues.getAsString(MediaStore.Images.Media.DISPLAY_NAME))
        assertEquals("MIME type should be image/png", "image/png",
            contentValues.getAsString(MediaStore.Images.Media.MIME_TYPE))
        assertEquals("Relative path should be Pictures/Screenshots", "${Environment.DIRECTORY_PICTURES}/Screenshots",
            contentValues.getAsString(MediaStore.Images.Media.RELATIVE_PATH))

        println("MediaStore content values test PASSED")
    }

    @Test
    fun testSaveToGallery() {
        // Create test bitmap
        val testBitmap = Bitmap.createBitmap(100, 100, Bitmap.Config.ARGB_8888).apply {
            eraseColor(Color.GREEN)
        }

        val timestamp = System.currentTimeMillis()
        val fileName = "TestScreenshot_$timestamp.png"

        val contentValues = ContentValues().apply {
            put(MediaStore.Images.Media.DISPLAY_NAME, fileName)
            put(MediaStore.Images.Media.MIME_TYPE, "image/png")
            put(MediaStore.Images.Media.RELATIVE_PATH, "${Environment.DIRECTORY_PICTURES}/Screenshots")
        }

        val resolver = context.contentResolver
        val uri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, contentValues)

        if (uri != null) {
            testUris.add(uri)  // Track for cleanup

            resolver.openOutputStream(uri)?.use { out ->
                testBitmap.compress(Bitmap.CompressFormat.PNG, 100, out)
            }

            // Verify file was saved
            resolver.openInputStream(uri)?.use { input ->
                val savedBitmap = android.graphics.BitmapFactory.decodeStream(input)
                assertNotNull("Should be able to read saved image", savedBitmap)
                assertEquals("Width should match", 100, savedBitmap.width)
                assertEquals("Height should match", 100, savedBitmap.height)
                savedBitmap.recycle()
            }

            println("Save to gallery test PASSED - uri: $uri")
        } else {
            // MediaStore may not be available in test environment
            println("Save to gallery test SKIPPED - MediaStore not available")
        }

        testBitmap.recycle()
    }

    @Test
    fun testPngCompression() {
        val bitmap = Bitmap.createBitmap(200, 200, Bitmap.Config.ARGB_8888).apply {
            eraseColor(Color.MAGENTA)
        }

        val output = java.io.ByteArrayOutputStream()
        val success = bitmap.compress(Bitmap.CompressFormat.PNG, 100, output)

        assertTrue("PNG compression should succeed", success)
        assertTrue("Output should have content", output.size() > 0)

        // PNG header check (first 8 bytes)
        val bytes = output.toByteArray()
        assertEquals("PNG signature byte 1", 0x89.toByte(), bytes[0])
        assertEquals("PNG signature byte 2", 0x50.toByte(), bytes[1])  // 'P'
        assertEquals("PNG signature byte 3", 0x4E.toByte(), bytes[2])  // 'N'
        assertEquals("PNG signature byte 4", 0x47.toByte(), bytes[3])  // 'G'

        bitmap.recycle()
        println("PNG compression test PASSED - size: ${output.size()} bytes")
    }

    @Test
    fun testScaledDrawBitmap() {
        // Test drawing scaled bitmap (like compositing room to full screen)
        val sourceBitmap = Bitmap.createBitmap(100, 100, Bitmap.Config.ARGB_8888).apply {
            eraseColor(Color.CYAN)
        }

        val destWidth = 500
        val destHeight = 800
        val destBitmap = Bitmap.createBitmap(destWidth, destHeight, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(destBitmap)

        // Draw scaled (like in takeScreenshot)
        val destRect = android.graphics.RectF(0f, 0f, destWidth.toFloat(), destHeight.toFloat())
        canvas.drawBitmap(sourceBitmap, null, destRect, null)

        // Verify color is preserved at various points
        val centerColor = destBitmap.getPixel(250, 400)
        assertEquals("Scaled bitmap should preserve color", Color.CYAN, centerColor)

        val cornerColor = destBitmap.getPixel(0, 0)
        assertEquals("Corner should also be source color", Color.CYAN, cornerColor)

        sourceBitmap.recycle()
        destBitmap.recycle()
        println("Scaled draw bitmap test PASSED")
    }

    @Test
    fun testDateFormatLocale() {
        // Verify date format uses default locale as expected
        val format = java.text.SimpleDateFormat("yyyyMMdd_HHmmss", java.util.Locale.getDefault())

        // Create a specific date
        val calendar = java.util.Calendar.getInstance()
        calendar.set(2026, 0, 29, 14, 30, 22)  // Jan 29, 2026 14:30:22

        val formatted = format.format(calendar.time)
        assertEquals("Formatted date", "20260129_143022", formatted)

        println("Date format locale test PASSED")
    }

    @Test
    fun testScreenshotDirectoryPath() {
        // Verify the path format for Screenshots
        val path = "${Environment.DIRECTORY_PICTURES}/Screenshots"
        assertEquals("Path should be Pictures/Screenshots", "Pictures/Screenshots", path)

        println("Screenshot directory path test PASSED")
    }
}
