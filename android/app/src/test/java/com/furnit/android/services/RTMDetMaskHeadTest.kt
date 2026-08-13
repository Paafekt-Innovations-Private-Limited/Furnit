package com.furnit.android.services

import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.max
import kotlin.random.Random
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Guards the two changes the NEON mask head depends on:
 *
 *  - mask features are now read NHWC (`pixel * 8 + channel`) instead of NCHW, so a pixel's eight
 *    values share a cache line rather than being gathered across a 102 KB channel stride;
 *  - `relX`/`relY` are expressed as affine terms in `x`/`y` so the row component hoists out of the
 *    inner loop.
 *
 * Neither may change the plane the pipeline sees. The native kernel itself needs an arm64 device,
 * so it is covered by RTMDetMaskHeadNativeTest under androidTest; everything here runs on the JVM
 * against the scalar reference that also serves as the runtime fallback.
 */
class RTMDetMaskHeadTest {

    private val maskSide = 24
    private val inputSize = 640
    private val channels = 8
    private val pixels = maskSide * maskSide

    private val priorX = 271.5f
    private val priorY = 118.25f
    private val levelStride = 8f

    private fun randomCoeffs(seed: Int): FloatArray {
        val random = Random(seed)
        return FloatArray(169) { random.nextFloat() * 2f - 1f }
    }

    private fun randomFeaturesNchw(seed: Int): FloatArray {
        val random = Random(seed)
        return FloatArray(channels * pixels) { random.nextFloat() * 4f - 2f }
    }

    private fun nchwToNhwc(nchw: FloatArray): FloatArray {
        val nhwc = FloatArray(nchw.size)
        for (c in 0 until channels) {
            for (p in 0 until pixels) {
                nhwc[p * channels + c] = nchw[c * pixels + p]
            }
        }
        return nhwc
    }

    private fun affineTerms(): FloatArray {
        val maskStride = inputSize.toFloat() / maskSide.toFloat()
        val denominator = max(1f, levelStride * 8f)
        return floatArrayOf(
            (priorX - 0.5f * maskStride) / denominator,
            -maskStride / denominator,
            (priorY - 0.5f * maskStride) / denominator,
            -maskStride / denominator,
        )
    }

    private fun runScalar(features: FloatArray, isNhwc: Boolean, coeffs: FloatArray): FloatArray {
        val (xBase, xStep, yBase, yStep) = affineTerms().let {
            listOf(it[0], it[1], it[2], it[3])
        }
        val out = FloatArray(pixels)
        RTMDetMaskHead.buildPlaneScalar(
            coeffs = coeffs,
            maskFeat = features,
            maskFeatIsNhwc = isNhwc,
            maskSide = maskSide,
            relXBase = xBase,
            relXStep = xStep,
            relYBase = yBase,
            relYStep = yStep,
            out = out,
        )
        return out
    }

    /** The implementation exactly as it stood before this change: NCHW gather, gridX arithmetic. */
    private fun legacyPlane(featuresNchw: FloatArray, coeffs: FloatArray): FloatArray {
        val w1 = 0
        val w2 = w1 + 80
        val w3 = w2 + 64
        val b1 = w3 + 8
        val b2 = b1 + 8
        val b3 = b2 + 8
        val maskStride = inputSize.toFloat() / maskSide.toFloat()
        val out = FloatArray(pixels)
        val input = FloatArray(10)
        val hidden1 = FloatArray(8)
        val hidden2 = FloatArray(8)

        for (y in 0 until maskSide) {
            for (x in 0 until maskSide) {
                val pos = y * maskSide + x
                val gridX = (x + 0.5f) * maskStride
                val gridY = (y + 0.5f) * maskStride
                input[0] = (priorX - gridX) / max(1f, levelStride * 8f)
                input[1] = (priorY - gridY) / max(1f, levelStride * 8f)
                for (c in 0 until 8) input[2 + c] = featuresNchw[c * pixels + pos]

                for (o in 0 until 8) {
                    var sum = coeffs[b1 + o]
                    for (i in 0 until 10) sum += coeffs[w1 + o * 10 + i] * input[i]
                    hidden1[o] = max(0f, sum)
                }
                for (o in 0 until 8) {
                    var sum = coeffs[b2 + o]
                    for (i in 0 until 8) sum += coeffs[w2 + o * 8 + i] * hidden1[i]
                    hidden2[o] = max(0f, sum)
                }
                var logit = coeffs[b3]
                for (i in 0 until 8) logit += coeffs[w3 + i] * hidden2[i]
                out[pos] = if (logit >= 0f) {
                    (1.0 / (1.0 + exp(-logit.toDouble()))).toFloat()
                } else {
                    val z = exp(logit.toDouble())
                    (z / (1.0 + z)).toFloat()
                }
            }
        }
        return out
    }

    @Test
    fun nhwcAndNchwFeatureLayoutsProduceIdenticalPlanes() {
        val coeffs = randomCoeffs(seed = 11)
        val nchw = randomFeaturesNchw(seed = 22)
        val nhwc = nchwToNhwc(nchw)

        val fromNchw = runScalar(nchw, isNhwc = false, coeffs = coeffs)
        val fromNhwc = runScalar(nhwc, isNhwc = true, coeffs = coeffs)

        assertEquals(pixels, fromNhwc.size)
        for (i in 0 until pixels) {
            assertEquals("pixel $i", fromNchw[i], fromNhwc[i], 0f)
        }
    }

    @Test
    fun affineCoordinatesMatchTheOriginalGridArithmetic() {
        val coeffs = randomCoeffs(seed = 33)
        val nchw = randomFeaturesNchw(seed = 44)

        val legacy = legacyPlane(nchw, coeffs)
        val current = runScalar(nchwToNhwc(nchw), isNhwc = true, coeffs = coeffs)

        var worst = 0f
        for (i in 0 until pixels) {
            worst = max(worst, abs(legacy[i] - current[i]))
        }
        // Both are sigmoid outputs in [0,1]; the only difference is float association order in the
        // coordinate term, so they agree far tighter than any downstream mask threshold.
        assertTrue("worst deviation from the legacy formulation was $worst", worst < 1e-5f)
    }

    @Test
    fun planeValuesAreProbabilities() {
        val plane = runScalar(
            nchwToNhwc(randomFeaturesNchw(seed = 55)),
            isNhwc = true,
            coeffs = randomCoeffs(seed = 66),
        )
        plane.forEachIndexed { index, value ->
            assertTrue("pixel $index was $value", value.isFinite() && value in 0f..1f)
        }
    }

    @Test
    fun wrongCoefficientCountIsRejected() {
        val result = RTMDetMaskHead.buildPlane(
            coeffs = FloatArray(168),
            maskFeat = FloatArray(channels * pixels),
            maskFeatIsNhwc = true,
            maskSide = maskSide,
            inputSize = inputSize,
            priorX = priorX,
            priorY = priorY,
            levelStride = levelStride,
        )
        assertEquals(null, result)
    }

    @Test
    fun undersizedFeatureBufferIsRejected() {
        val result = RTMDetMaskHead.buildPlane(
            coeffs = randomCoeffs(seed = 77),
            maskFeat = FloatArray(channels * pixels - 1),
            maskFeatIsNhwc = true,
            maskSide = maskSide,
            inputSize = inputSize,
            priorX = priorX,
            priorY = priorY,
            levelStride = levelStride,
        )
        assertEquals(null, result)
    }

    @Test
    fun buildPlaneFallsBackToScalarWhenNativeIsUnavailable() {
        // The arm64 library cannot load on the JVM, so this exercises the fallback that keeps
        // behaviour identical when the native kernel is missing on a device too.
        val coeffs = randomCoeffs(seed = 88)
        val nhwc = nchwToNhwc(randomFeaturesNchw(seed = 99))

        val viaPublicEntryPoint = RTMDetMaskHead.buildPlane(
            coeffs = coeffs,
            maskFeat = nhwc,
            maskFeatIsNhwc = true,
            maskSide = maskSide,
            inputSize = inputSize,
            priorX = priorX,
            priorY = priorY,
            levelStride = levelStride,
        )

        requireNotNull(viaPublicEntryPoint)
        val reference = runScalar(nhwc, isNhwc = true, coeffs = coeffs)
        for (i in 0 until pixels) {
            assertEquals("pixel $i", reference[i], viaPublicEntryPoint[i], 0f)
        }
    }
}
