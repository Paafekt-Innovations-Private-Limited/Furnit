package com.furnit.android.services

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import kotlin.math.abs
import kotlin.math.max
import kotlin.random.Random
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Device-side checks for the NEON mask head. The scalar reference in [RTMDetMaskHead] is covered on
 * the JVM by RTMDetMaskHeadTest; the native kernel needs a real arm64 device, so it lives here.
 *
 * Run with:  ./gradlew :app:connectedDebugAndroidTest --tests '*RTMDetMaskHeadNativeTest*'
 */
@RunWith(AndroidJUnit4::class)
class RTMDetMaskHeadNativeTest {

    private companion object {
        const val TAG = "RTMDetMaskHeadNative"
        const val MASK_SIDE = 160
        const val INPUT_SIZE = 640
        const val CHANNELS = 8
        const val PIXELS = MASK_SIDE * MASK_SIDE

        const val PRIOR_X = 271.5f
        const val PRIOR_Y = 118.25f
        const val LEVEL_STRIDE = 8f
    }

    private fun coeffs(seed: Int) = Random(seed).let { r -> FloatArray(169) { r.nextFloat() * 2f - 1f } }

    private fun featuresNhwc(seed: Int) =
        Random(seed).let { r -> FloatArray(CHANNELS * PIXELS) { r.nextFloat() * 4f - 2f } }

    private fun scalarPlane(coeffs: FloatArray, features: FloatArray): FloatArray {
        val maskStride = INPUT_SIZE.toFloat() / MASK_SIDE.toFloat()
        val denominator = max(1f, LEVEL_STRIDE * 8f)
        val out = FloatArray(PIXELS)
        RTMDetMaskHead.buildPlaneScalar(
            coeffs = coeffs,
            maskFeat = features,
            maskFeatIsNhwc = true,
            maskSide = MASK_SIDE,
            relXBase = (PRIOR_X - 0.5f * maskStride) / denominator,
            relXStep = -maskStride / denominator,
            relYBase = (PRIOR_Y - 0.5f * maskStride) / denominator,
            relYStep = -maskStride / denominator,
            out = out,
        )
        return out
    }

    private fun nativePlane(coeffs: FloatArray, features: FloatArray): FloatArray =
        requireNotNull(
            RTMDetMaskHead.buildPlane(
                coeffs = coeffs,
                maskFeat = features,
                maskFeatIsNhwc = true,
                maskSide = MASK_SIDE,
                inputSize = INPUT_SIZE,
                priorX = PRIOR_X,
                priorY = PRIOR_Y,
                levelStride = LEVEL_STRIDE,
            ),
        )

    @Test
    fun nativeLibraryIsPackagedAndLoads() {
        assertTrue(
            "libfurnit_rtmdet.so did not load; the mask head would silently fall back to scalar",
            RTMDetMaskHead.isNativeAvailable,
        )
    }

    @Test
    fun nativeMatchesTheScalarReference() {
        val coefficients = coeffs(seed = 101)
        val features = featuresNhwc(seed = 202)

        val native = nativePlane(coefficients, features)
        val scalar = scalarPlane(coefficients, features)

        assertEquals(PIXELS, native.size)
        var worst = 0f
        var thresholdDisagreements = 0
        for (i in 0 until PIXELS) {
            worst = max(worst, abs(native[i] - scalar[i]))
            if ((native[i] >= 0.5f) != (scalar[i] >= 0.5f)) thresholdDisagreements++
        }
        Log.i(TAG, "native vs scalar: worst=$worst thresholdDisagreements=$thresholdDisagreements")

        // The kernel uses fused multiply-add, which skips the intermediate rounding the scalar
        // path performs, so the two agree to float noise rather than bit-for-bit.
        assertTrue("worst deviation was $worst", worst < 2e-4f)
        assertEquals("mask threshold decisions must not change", 0, thresholdDisagreements)
    }

    @Test
    fun nativeIsSubstantiallyFasterThanScalar() {
        val coefficients = coeffs(seed = 303)
        val features = featuresNhwc(seed = 404)

        repeat(3) { nativePlane(coefficients, features); scalarPlane(coefficients, features) }

        val iterations = 20
        val nativeStart = System.nanoTime()
        repeat(iterations) { nativePlane(coefficients, features) }
        val nativeMillis = (System.nanoTime() - nativeStart) / 1_000_000.0 / iterations

        val scalarStart = System.nanoTime()
        repeat(iterations) { scalarPlane(coefficients, features) }
        val scalarMillis = (System.nanoTime() - scalarStart) / 1_000_000.0 / iterations

        Log.i(
            TAG,
            "mask plane: native=${"%.2f".format(nativeMillis)}ms " +
                "scalar=${"%.2f".format(scalarMillis)}ms " +
                "speedup=${"%.1f".format(scalarMillis / nativeMillis)}x",
        )
        assertTrue(
            "native ${nativeMillis}ms was not faster than scalar ${scalarMillis}ms",
            nativeMillis < scalarMillis,
        )
    }

    @Test
    fun repeatedInvocationsAreStable() {
        val coefficients = coeffs(seed = 505)
        val features = featuresNhwc(seed = 606)
        val first = nativePlane(coefficients, features)
        repeat(5) {
            val again = nativePlane(coefficients, features)
            for (i in 0 until PIXELS) {
                assertEquals("pixel $i on repeat", first[i], again[i], 0f)
            }
        }
    }
}
