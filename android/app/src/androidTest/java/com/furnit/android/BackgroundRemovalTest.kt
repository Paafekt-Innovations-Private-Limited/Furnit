package com.furnit.android

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.furnit.android.services.SmartyPantsManager
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

/**
 * Test background removal segmentation on bus.jpg
 * Verifies that detected objects are kept and background is transparent.
 */
@RunWith(AndroidJUnit4::class)
class BackgroundRemovalTest {

    @Test
    fun testBackgroundRemovalOnBusImage() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext

        // Load bus.jpg from assets
        val bitmap = context.assets.open("bus.jpg").use { input ->
            BitmapFactory.decodeStream(input)
        }
        assertNotNull("Failed to load bus.jpg", bitmap)
        println("Loaded bus.jpg: ${bitmap.width}x${bitmap.height}")

        // Initialize SmartyPantsManager with ONNX
        val manager = SmartyPantsManager(context)
        manager.initializeOnnx("yoloe-11l-seg-pf.onnx")

        // Run segmentation with detections
        val latch = CountDownLatch(1)
        var resultMask: Bitmap? = null
        var detections: List<DetectionResult> = emptyList()

        manager.segmentWithDetectionsAsync(bitmap) { result ->
            resultMask = result?.mask
            detections = result?.detections ?: emptyList()
            latch.countDown()
        }

        // Wait for result
        val completed = latch.await(60, TimeUnit.SECONDS)
        assertTrue("Segmentation timed out", completed)

        // Check detections
        println("Detections found: ${detections.size}")
        for ((i, det) in detections.withIndex()) {
            println("  [$i] ${det.label}: ${String.format("%.1f%%", det.confidence * 100)}")
        }
        assertTrue("Should detect at least 1 object", detections.isNotEmpty())

        // Check that bus or person was detected
        val labels = detections.map { it.label }
        assertTrue("Should detect bus or person", labels.contains("bus") || labels.contains("person"))

        // Check mask was generated
        assertNotNull("Mask should not be null", resultMask)
        val mask = resultMask!!
        println("Mask generated: ${mask.width}x${mask.height}")

        // Analyze mask pixels - should have transparent and non-transparent regions
        val pixels = IntArray(mask.width * mask.height)
        mask.getPixels(pixels, 0, mask.width, 0, 0, mask.width, mask.height)

        var transparentPixels = 0
        var opaquePixels = 0
        for (pixel in pixels) {
            val alpha = Color.alpha(pixel)
            if (alpha == 0) transparentPixels++ else opaquePixels++
        }

        println("Pixel analysis:")
        println("  Transparent (background): $transparentPixels")
        println("  Opaque (objects): $opaquePixels")
        println("  Object coverage: ${String.format("%.1f%%", opaquePixels * 100.0 / pixels.size)}")

        // Verify both transparent and opaque regions exist
        assertTrue("Should have transparent background pixels", transparentPixels > 1000)
        assertTrue("Should have opaque object pixels", opaquePixels > 1000)

        // Save cutout image for visual inspection
        val externalDir = context.getExternalFilesDir(null)!!
        val cutoutFile = File(externalDir, "test_cutout.png")
        FileOutputStream(cutoutFile).use { out ->
            mask.compress(Bitmap.CompressFormat.PNG, 100, out)
        }
        println("Saved cutout to: ${cutoutFile.absolutePath}")

        // Create composite image: cutout on dark background
        val composite = Bitmap.createBitmap(mask.width, mask.height, Bitmap.Config.ARGB_8888)
        val compositePixels = IntArray(mask.width * mask.height)
        for (i in pixels.indices) {
            val alpha = Color.alpha(pixels[i])
            if (alpha > 0) {
                compositePixels[i] = pixels[i]
            } else {
                compositePixels[i] = Color.argb(255, 40, 40, 40)  // Dark gray background
            }
        }
        composite.setPixels(compositePixels, 0, mask.width, 0, 0, mask.width, mask.height)

        val compositeFile = File(externalDir, "test_cutout_composite.png")
        FileOutputStream(compositeFile).use { out ->
            composite.compress(Bitmap.CompressFormat.PNG, 100, out)
        }
        println("Saved composite to: ${compositeFile.absolutePath}")

        // Cleanup
        manager.close()

        println("\nTEST PASSED - Background removal working")
        println("Pull files with: adb pull ${cutoutFile.absolutePath}")
        println("Pull files with: adb pull ${compositeFile.absolutePath}")
    }
}
