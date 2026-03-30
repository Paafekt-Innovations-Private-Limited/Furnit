package com.furnit.android

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.furnit.android.services.FurnitureFitManager
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

/**
 * Test that saves the final mask from FurnitureFitManager for visual inspection.
 */
@RunWith(AndroidJUnit4::class)
class FinalMaskTest {

    @Test
    fun testFinalMaskOnBusImage() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext

        // Load bus.jpg from assets
        val bitmap = context.assets.open("bus.jpg").use { input ->
            BitmapFactory.decodeStream(input)
        }
        assertNotNull("Failed to load bus.jpg", bitmap)
        println("Loaded bus.jpg: ${bitmap.width}x${bitmap.height}")

        // Initialize FurnitureFitManager with ONNX
        val manager = FurnitureFitManager(context)
        manager.initializeOnnx("yoloe-26l-seg-pf_seg_o2m.onnx")

        // Run segmentation
        val latch = CountDownLatch(1)
        var resultMask: Bitmap? = null

        manager.segmentImageAsync(bitmap) { mask ->
            resultMask = mask
            latch.countDown()
        }

        // Wait for result
        val completed = latch.await(30, TimeUnit.SECONDS)
        assertTrue("Segmentation timed out", completed)

        // Check mask was generated
        assertNotNull("Mask should not be null", resultMask)
        val mask = resultMask!!
        println("Mask generated: ${mask.width}x${mask.height}")

        // Analyze mask pixels
        val pixels = IntArray(mask.width * mask.height)
        mask.getPixels(pixels, 0, mask.width, 0, 0, mask.width, mask.height)

        var greenPixels = 0
        var transparentPixels = 0
        for (pixel in pixels) {
            val alpha = (pixel shr 24) and 0xFF
            if (alpha > 0) greenPixels++ else transparentPixels++
        }

        println("Mask analysis:")
        println("  Total pixels: ${pixels.size}")
        println("  Green (masked) pixels: $greenPixels")
        println("  Transparent pixels: $transparentPixels")
        println("  Coverage: ${String.format("%.1f", greenPixels * 100.0 / pixels.size)}%")

        // Save mask to file for visual inspection
        val maskFile = File(context.cacheDir, "final_mask_test.png")
        FileOutputStream(maskFile).use { out ->
            mask.compress(Bitmap.CompressFormat.PNG, 100, out)
        }
        println("Saved mask to: ${maskFile.absolutePath}")

        // Create overlay image (mask on top of original)
        val overlay = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val scaledMask = Bitmap.createScaledBitmap(mask, overlay.width, overlay.height, true)
        val overlayPixels = IntArray(overlay.width * overlay.height)
        val maskPixelsScaled = IntArray(overlay.width * overlay.height)
        overlay.getPixels(overlayPixels, 0, overlay.width, 0, 0, overlay.width, overlay.height)
        scaledMask.getPixels(maskPixelsScaled, 0, overlay.width, 0, 0, overlay.width, overlay.height)

        for (i in overlayPixels.indices) {
            val maskAlpha = (maskPixelsScaled[i] shr 24) and 0xFF
            if (maskAlpha > 0) {
                // Blend green onto original
                val origR = (overlayPixels[i] shr 16) and 0xFF
                val origG = (overlayPixels[i] shr 8) and 0xFF
                val origB = overlayPixels[i] and 0xFF
                val blendR = (origR * 0.5 + 0 * 0.5).toInt()
                val blendG = (origG * 0.5 + 255 * 0.5).toInt()
                val blendB = (origB * 0.5 + 0 * 0.5).toInt()
                overlayPixels[i] = (0xFF shl 24) or (blendR shl 16) or (blendG shl 8) or blendB
            }
        }
        overlay.setPixels(overlayPixels, 0, overlay.width, 0, 0, overlay.width, overlay.height)

        val overlayFile = File(context.cacheDir, "final_mask_overlay.png")
        FileOutputStream(overlayFile).use { out ->
            overlay.compress(Bitmap.CompressFormat.PNG, 100, out)
        }
        println("Saved overlay to: ${overlayFile.absolutePath}")

        // Verify mask has reasonable coverage (not all transparent, not all green)
        assertTrue("Mask should have some green pixels", greenPixels > 100)
        assertTrue("Mask should have some transparent pixels", transparentPixels > 100)

        // Cleanup
        manager.close()

        println("\nTEST PASSED - Final mask saved for inspection")
        println("Pull files with: adb pull ${maskFile.absolutePath}")
        println("Pull files with: adb pull ${overlayFile.absolutePath}")
    }
}
