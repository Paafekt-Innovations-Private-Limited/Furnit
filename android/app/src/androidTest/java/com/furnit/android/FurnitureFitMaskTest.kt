package com.furnit.android

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.File
import java.io.FileOutputStream
import kotlin.math.exp

@RunWith(AndroidJUnit4::class)
class FurnitureFitMaskTest {

    @Test
    fun testMaskGenerationWithBusImage() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext

        // Copy ONNX model from assets to cache
        val modelFile = File(context.cacheDir, "yolo11l-seg.onnx")
        context.assets.open("yoloe-11l-seg-pf.onnx").use { input ->
            FileOutputStream(modelFile).use { output ->
                input.copyTo(output)
            }
        }

        // Copy bus.jpg test image from assets
        val busFile = File(context.cacheDir, "bus.jpg")
        context.assets.open("bus.jpg").use { input ->
            FileOutputStream(busFile).use { output ->
                input.copyTo(output)
            }
        }

        // Load image
        val bitmap = BitmapFactory.decodeFile(busFile.absolutePath)
        assertNotNull("Failed to load bus.jpg", bitmap)
        println("Loaded bus.jpg: ${bitmap.width}x${bitmap.height}")

        // Resize to 640x640
        val inputW = 640
        val inputH = 640
        val resized = Bitmap.createScaledBitmap(bitmap, inputW, inputH, true)
            .copy(Bitmap.Config.ARGB_8888, false)

        // Prepare input tensor (NCHW format)
        val floatCount = 1 * 3 * inputH * inputW
        val inputFloats = FloatArray(floatCount)
        val intValues = IntArray(inputW * inputH)
        resized.getPixels(intValues, 0, inputW, 0, 0, inputW, inputH)

        val hw = inputH * inputW
        for (y in 0 until inputH) {
            val rowOff = y * inputW
            for (x in 0 until inputW) {
                val v = intValues[rowOff + x]
                val r = ((v shr 16) and 0xFF) / 255.0f
                val g = ((v shr 8) and 0xFF) / 255.0f
                val b = (v and 0xFF) / 255.0f
                val pixelIdx = rowOff + x
                inputFloats[0 * hw + pixelIdx] = r
                inputFloats[1 * hw + pixelIdx] = g
                inputFloats[2 * hw + pixelIdx] = b
            }
        }

        println("Input prepared: ${inputFloats.size} floats, range [${inputFloats.minOrNull()}, ${inputFloats.maxOrNull()}]")

        // Create ONNX session
        val ortEnv = OrtEnvironment.getEnvironment()
        val ortSession = ortEnv.createSession(modelFile.absolutePath, OrtSession.SessionOptions())

        val inputName = ortSession.inputInfo.keys.first()
        val shapeLong = longArrayOf(1, 3, inputH.toLong(), inputW.toLong())
        val tensor = OnnxTensor.createTensor(ortEnv, java.nio.FloatBuffer.wrap(inputFloats), shapeLong)

        // Run inference
        val results = ortSession.run(mapOf(inputName to tensor))

        // Get outputs
        val detResult = results.get(0)
        val protoResult = results.get(1)

        val detValue = detResult?.value
        val protoValue = protoResult?.value

        // Parse detection tensor [1, 116, 8400]
        @Suppress("UNCHECKED_CAST")
        val det3d = detValue as Array<Array<FloatArray>>
        println("Det shape: [${det3d.size}][${det3d[0].size}][${det3d[0][0].size}]")

        val numFeatures = det3d[0].size  // 116
        val numAnchors = det3d[0][0].size  // 8400
        val numClasses = 80
        val numMaskCoeffs = 32
        val classStartIdx = 4
        val maskCoeffStartIdx = 4 + numClasses  // 84

        // Parse proto tensor [1, 32, 160, 160]
        val proto = extractFloatArray(protoValue)
        println("Proto size: ${proto.size}")

        // Get proto dimensions
        val numProtos = 32
        val protoH = 160
        val protoW = 160
        val protoScale = inputW.toFloat() / protoW  // 4.0

        println("Proto scale: $protoScale")

        // Find detections
        val confThreshold = 0.25f

        // COCO class names
        val cocoClasses = arrayOf(
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
            "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
            "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
            "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
            "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
            "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
            "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
        )

        data class Detection(
            val x: Float, val y: Float, val w: Float, val h: Float,
            val conf: Float, val classId: Int, val coeffs: FloatArray
        )

        val detections = mutableListOf<Detection>()

        for (anchor in 0 until numAnchors) {
            var maxScore = Float.MIN_VALUE
            var bestClass = -1
            for (c in 0 until numClasses) {
                val score = det3d[0][classStartIdx + c][anchor]
                if (score > maxScore) {
                    maxScore = score
                    bestClass = c
                }
            }

            if (maxScore > confThreshold) {
                val x = det3d[0][0][anchor]
                val y = det3d[0][1][anchor]
                val bw = det3d[0][2][anchor]
                val bh = det3d[0][3][anchor]

                val coeffs = FloatArray(numMaskCoeffs)
                for (c in 0 until numMaskCoeffs) {
                    coeffs[c] = det3d[0][maskCoeffStartIdx + c][anchor]
                }

                detections.add(Detection(x, y, bw, bh, maxScore, bestClass, coeffs))
            }
        }

        println("Found ${detections.size} detections above $confThreshold")
        assertTrue("Should find at least 1 detection on bus image", detections.isNotEmpty())

        // Sort and take top 5
        val topDets = detections.sortedByDescending { it.conf }.take(5)

        println("\n=== TOP DETECTIONS ===")
        for ((i, d) in topDets.withIndex()) {
            val label = cocoClasses.getOrElse(d.classId) { "unknown" }
            println("  [$i] $label: conf=${String.format("%.3f", d.conf)} bbox=(${d.x.toInt()},${d.y.toInt()},${d.w.toInt()},${d.h.toInt()})")
        }
        println("======================\n")

        // Create annotated image with bboxes and labels
        val annotated = resized.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = android.graphics.Canvas(annotated)
        val boxPaint = android.graphics.Paint().apply {
            color = android.graphics.Color.GREEN
            style = android.graphics.Paint.Style.STROKE
            strokeWidth = 3f
        }
        val textPaint = android.graphics.Paint().apply {
            color = android.graphics.Color.GREEN
            textSize = 24f
            style = android.graphics.Paint.Style.FILL
        }
        val bgPaint = android.graphics.Paint().apply {
            color = android.graphics.Color.BLACK
            style = android.graphics.Paint.Style.FILL
        }

        for (d in topDets) {
            val left = d.x - d.w / 2
            val top = d.y - d.h / 2
            val right = d.x + d.w / 2
            val bottom = d.y + d.h / 2

            // Draw bbox
            canvas.drawRect(left, top, right, bottom, boxPaint)

            // Draw label background
            val label = "${cocoClasses.getOrElse(d.classId) { "?" }} ${String.format("%.2f", d.conf)}"
            val textWidth = textPaint.measureText(label)
            canvas.drawRect(left, top - 28, left + textWidth + 8, top, bgPaint)

            // Draw label text
            canvas.drawText(label, left + 4, top - 6, textPaint)
        }

        // Save annotated image to external storage (accessible via adb)
        val externalDir = context.getExternalFilesDir(null)!!
        val annotatedFile = java.io.File(externalDir, "test_annotated.png")
        java.io.FileOutputStream(annotatedFile).use { out ->
            annotated.compress(Bitmap.CompressFormat.PNG, 100, out)
        }
        println("Saved annotated image to: ${annotatedFile.absolutePath}")

        // Generate mask
        val maskProto = FloatArray(protoH * protoW)

        for (detection in topDets) {
            val bboxLeft = ((detection.x - detection.w / 2) / protoScale).toInt().coerceIn(0, protoW - 1)
            val bboxTop = ((detection.y - detection.h / 2) / protoScale).toInt().coerceIn(0, protoH - 1)
            val bboxRight = ((detection.x + detection.w / 2) / protoScale).toInt().coerceIn(0, protoW - 1)
            val bboxBottom = ((detection.y + detection.h / 2) / protoScale).toInt().coerceIn(0, protoH - 1)

            println("  BBox in proto coords: ($bboxLeft, $bboxTop) - ($bboxRight, $bboxBottom)")

            for (py in bboxTop..bboxBottom) {
                for (px in bboxLeft..bboxRight) {
                    var sum = 0f
                    for (c in 0 until numProtos) {
                        // Proto is [1, 32, 160, 160], flattened as [c * H * W + y * W + x]
                        val protoIdx = c * protoH * protoW + py * protoW + px
                        sum += detection.coeffs[c] * proto[protoIdx]
                    }
                    val sigmoidVal = 1f / (1f + exp(-sum))
                    if (sigmoidVal > maskProto[py * protoW + px]) {
                        maskProto[py * protoW + px] = sigmoidVal
                    }
                }
            }
        }

        // Check mask values
        val maskMin = maskProto.minOrNull() ?: 0f
        val maskMax = maskProto.maxOrNull() ?: 0f
        val maskAboveThreshold = maskProto.count { it > 0.5f }

        println("Mask range: [$maskMin, $maskMax]")
        println("Mask pixels > 0.5: $maskAboveThreshold")

        assertTrue("Mask max should be > 0.5 for valid detections", maskMax > 0.5f)
        assertTrue("Should have some mask pixels > 0.5", maskAboveThreshold > 100)

        // Create mask bitmap
        val maskBitmap = Bitmap.createBitmap(protoW, protoH, Bitmap.Config.ARGB_8888)
        val pixels = IntArray(protoW * protoH)
        for (i in pixels.indices) {
            val v = maskProto[i]
            val alpha = if (v > 0.5f) 0xCC else 0x00
            pixels[i] = (alpha shl 24) or 0x00FF00
        }
        maskBitmap.setPixels(pixels, 0, protoW, 0, 0, protoW, protoH)

        // Save mask to file for visual inspection
        val maskFile = File(externalDir, "test_mask_output.png")
        FileOutputStream(maskFile).use { out ->
            maskBitmap.compress(Bitmap.CompressFormat.PNG, 100, out)
        }
        println("Saved mask to: ${maskFile.absolutePath}")

        // Cleanup
        tensor.close()
        results.close()
        ortSession.close()

        println("TEST PASSED")
    }

    private fun extractFloatArray(value: Any?): FloatArray {
        return when (value) {
            is FloatArray -> value
            is Array<*> -> flattenArrayToFloat(value)
            is java.nio.FloatBuffer -> {
                val arr = FloatArray(value.remaining())
                value.get(arr)
                arr
            }
            else -> FloatArray(0)
        }
    }

    private fun flattenArrayToFloat(arr: Array<*>): FloatArray {
        val list = ArrayList<Float>()
        fun rec(a: Any?) {
            when (a) {
                is Float -> list.add(a)
                is FloatArray -> for (v in a) list.add(v)
                is Array<*> -> for (e in a) rec(e)
            }
        }
        rec(arr)
        return list.toFloatArray()
    }
}
