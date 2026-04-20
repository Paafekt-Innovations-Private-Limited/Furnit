package com.furnit.android

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.furnit.android.services.FurnitureFitManager
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

/**
 * End-to-end instrumentation coverage for the iOS-parity YOLOE 11L PF ONNX pipeline on Android.
 */
@RunWith(AndroidJUnit4::class)
class Yoloe11lSegPfOnnxTest {

    @Test
    fun testYoloe11lSegPfEndToEndOnBusImage() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val bitmap = context.assets.open("bus.jpg").use(BitmapFactory::decodeStream)
        assertNotNull("Failed to load bus.jpg", bitmap)

        val manager = FurnitureFitManager(context)
        manager.initializeOnnx()

        val latch = CountDownLatch(1)
        var resultMask: Bitmap? = null
        var detections: List<DetectionResult> = emptyList()
        val wallClockStart = System.nanoTime()

        manager.segmentWithDetectionsAsync(bitmap) { result ->
            resultMask = result?.mask
            detections = result?.detections ?: emptyList()
            latch.countDown()
        }

        val completed = latch.await(60, TimeUnit.SECONDS)
        assertTrue("YOLOE 11L segmentation timed out", completed)
        println("YOLOE 11L end-to-end wall time: ${(System.nanoTime() - wallClockStart) / 1_000_000L}ms")

        assertTrue("Expected at least one detection from YOLOE 11L", detections.isNotEmpty())
        println("YOLOE 11L detections: ${detections.size}")
        detections.take(5).forEachIndexed { index, detection ->
            println("  [$index] ${detection.label}: ${String.format("%.2f", detection.confidence)}")
        }

        val mask = resultMask
        assertNotNull("Expected non-null cutout mask from YOLOE 11L", mask)

        val pixels = IntArray(mask!!.width * mask.height)
        mask.getPixels(pixels, 0, mask.width, 0, 0, mask.width, mask.height)
        val opaquePixels = pixels.count { Color.alpha(it) > 0 }
        val transparentPixels = pixels.size - opaquePixels
        println("YOLOE 11L mask coverage: opaque=$opaquePixels transparent=$transparentPixels total=${pixels.size}")
        assertTrue("Expected some opaque cutout pixels", opaquePixels > 100)
        assertTrue("Expected some transparent background pixels", transparentPixels > 100)

        val outputFile = File(context.cacheDir, "yoloe11l_seg_pf_cutout.png")
        FileOutputStream(outputFile).use { stream ->
            mask.compress(Bitmap.CompressFormat.PNG, 100, stream)
        }
        println("Saved YOLOE 11L cutout to: ${outputFile.absolutePath}")

        manager.close()
    }
}
