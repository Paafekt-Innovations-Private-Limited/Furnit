package com.furnit.android

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.json.JSONObject
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor
import java.io.File

/**
 * Python-Kotlin parity test for ExecuTorch SHARP Part1.
 *
 * Python (mobile-like): test_sharp_split_mobile.py --image PXL_room.jpg -o python_part1_baseline.json
 * Kotlin: This test loads same image, runs Part1.pte one-patch forward, compares with baseline.
 *
 * Requires: sharp_split_part1.pte in app models dir (push via push_sharp_executorch_models.sh)
 */
@RunWith(AndroidJUnit4::class)
class ExecutorchSharpParityTest {

    companion object {
        private const val TAG = "ExecParityTest"
        private const val PATCH_SIZE = 384
        private const val IMAGE_SIZE = 1536
        private const val FEATURE_DIM = 1024
    }

    private lateinit var context: android.content.Context
    private lateinit var testContext: android.content.Context

    @Before
    fun setup() {
        context = InstrumentationRegistry.getInstrumentation().targetContext
        testContext = InstrumentationRegistry.getInstrumentation().context
    }

    private fun preprocessPatch(bitmap: Bitmap): FloatArray {
        val width = bitmap.width
        val height = bitmap.height
        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)
        val floatArray = FloatArray(3 * width * height)
        val channelSize = width * height
        for (i in pixels.indices) {
            floatArray[i] = ((pixels[i] shr 16) and 0xFF) / 255f
            floatArray[channelSize + i] = ((pixels[i] shr 8) and 0xFF) / 255f
            floatArray[2 * channelSize + i] = (pixels[i] and 0xFF) / 255f
        }
        return floatArray
    }

    private fun checksumFloats(arr: FloatArray, n: Int = 32): List<Float> =
        (0 until minOf(n, arr.size)).map { arr[it] }

    @Test
    fun testPart1OnePatchParityWithPython() {
        val cpuInternal = File(context.filesDir, "models_cpu")
        val ptePath = File(cpuInternal, "sharp_split_part1.pte").takeIf { it.exists() }
            ?: context.getExternalFilesDir("models_cpu")?.let { File(it, "sharp_split_part1.pte").takeIf { f -> f.exists() } }
            ?: run {
                Log.w(TAG, "SKIP: sharp_split_part1.pte not found. Push via: ./push_sharp_executorch_models.sh executorch_models")
                return
            }

        val bitmap = testContext.assets.open("PXL_room.jpg").use { BitmapFactory.decodeStream(it) }
        assertNotNull("PXL_room.jpg", bitmap)

        val scaled = Bitmap.createScaledBitmap(bitmap!!, IMAGE_SIZE, IMAGE_SIZE, true)
        val patchBitmap = Bitmap.createBitmap(scaled, 0, 0, PATCH_SIZE, PATCH_SIZE)
        scaled.recycle()
        bitmap.recycle()

        val patchData = preprocessPatch(patchBitmap)
        patchBitmap.recycle()

        val inputTensor = Tensor.fromBlob(
            patchData,
            longArrayOf(1, 3, PATCH_SIZE.toLong(), PATCH_SIZE.toLong())
        )

        val loadStart = System.currentTimeMillis()
        val module = Module.load(ptePath.absolutePath)
        val loadMs = System.currentTimeMillis() - loadStart

        val forwardStart = System.currentTimeMillis()
        val outputs = module.forward(EValue.from(inputTensor))
        val forwardMs = System.currentTimeMillis() - forwardStart

        module.destroy()

        assertTrue("Part1 returns 2 outputs (tokens, block5)", outputs.size >= 2)
        val tokens = outputs[0].toTensor().getDataAsFloatArray()
        val block5 = outputs[1].toTensor().getDataAsFloatArray()

        Log.d(TAG, "Android Part1: load=${loadMs}ms forward=${forwardMs}ms tokens=${tokens.size} block5=${block5.size}")

        val tokensChecksum = checksumFloats(tokens)
        val block5Checksum = checksumFloats(block5)

        val baselineJson = try {
            testContext.assets.open("python_part1_baseline.json").use { it.bufferedReader().readText() }
        } catch (e: Exception) {
            Log.w(TAG, "No python_part1_baseline.json - run: python test_sharp_split_mobile.py -o app/src/androidTest/assets/python_part1_baseline.json")
            return
        }

        val baseline = JSONObject(baselineJson)
        val pyTokensChecksum = baseline.getJSONArray("tokens_checksum")
        val tolerance = 5e-3f  // Allow minor float differences across runtimes

        for (i in 0 until minOf(8, pyTokensChecksum.length(), tokensChecksum.size)) {
            val pyVal = pyTokensChecksum.getDouble(i).toFloat()
            val ktVal = tokensChecksum[i]
            val diff = kotlin.math.abs(pyVal - ktVal)
            assertTrue("tokens[$i] Py=$pyVal Kt=$ktVal diff=$diff", diff < tolerance)
        }

        Log.d(TAG, "PASS: Kotlin output matches Python baseline (tokens checksum)")
    }
}
