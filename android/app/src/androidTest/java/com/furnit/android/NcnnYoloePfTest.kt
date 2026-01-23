package com.furnit.android

import android.graphics.BitmapFactory
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.furnit.android.services.NcnnYoloe
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Test YOLOE model with NCNN backend.
 * Runs inference on bus.jpg and verifies it detects "bus" and "person".
 */
@RunWith(AndroidJUnit4::class)
class NcnnYoloePfTest {

    @Test
    fun testYoloePfDetectsBusImage() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext

        // Check if NCNN library is available
        if (!NcnnYoloe.isAvailable()) {
            val error = NcnnYoloe.getLoadError() ?: "unknown"
            println("NCNN not available: $error")
            println("Skipping test - NCNN library not loaded")
            return
        }

        // Initialize NCNN with YOLOE-PF model
        val ncnn = NcnnYoloe()
        val initSuccess = ncnn.init(
            context = context,
            paramAsset = "yoloe-pf.param",
            binAsset = "yoloe-pf.bin",
            useGpu = false,  // CPU for test stability
            numThreads = 4
        )

        assertTrue("NCNN initialization should succeed", initSuccess)

        // Load bus.jpg from assets
        val bitmap = context.assets.open("bus.jpg").use { input ->
            BitmapFactory.decodeStream(input)
        }
        assertNotNull("Failed to load bus.jpg", bitmap)
        println("Loaded bus.jpg: ${bitmap.width}x${bitmap.height}")

        // Run detection
        val startTime = System.currentTimeMillis()
        val detections = ncnn.detect(
            bitmap = bitmap,
            confThreshold = 0.25f,
            iouThreshold = 0.45f
        )
        val inferenceTime = System.currentTimeMillis() - startTime

        println("Inference time: ${inferenceTime}ms")
        println("Found ${detections.size} detections")

        // Print all detections
        println("\n=== DETECTIONS ===")
        for ((idx, det) in detections.withIndex()) {
            println("[$idx] ${det.label} (class ${det.classId}): " +
                    "conf=${String.format("%.3f", det.confidence)}, " +
                    "bbox=(${det.x.toInt()},${det.y.toInt()},${det.width.toInt()},${det.height.toInt()})")
        }
        println("==================\n")

        // Verify we found detections
        assertTrue("Should find at least 1 detection", detections.isNotEmpty())

        // Check for expected objects in bus.jpg
        val labels = detections.map { it.label.lowercase() }
        val hasBus = labels.any { it == "bus" }
        val hasPerson = labels.any { it == "person" }

        println("Found bus: $hasBus")
        println("Found person: $hasPerson")

        // bus.jpg should have a bus and people
        assertTrue("Should detect 'bus' in bus.jpg", hasBus)
        assertTrue("Should detect 'person' in bus.jpg", hasPerson)

        // Test mask generation
        val result = ncnn.detectWithMask(
            bitmap = bitmap,
            confThreshold = 0.25f,
            iouThreshold = 0.45f,
            maskThreshold = 0.5f
        )

        val mask = result.mask
        println("Mask generated: ${mask != null}")
        if (mask != null) {
            println("Mask size: ${mask.width}x${mask.height}")
        }

        // Cleanup
        ncnn.release()

        println("\nTEST PASSED - YOLOE-PF correctly identifies objects!")
    }
}
