package com.furnit.android.services

import com.furnit.android.utils.LogUtil
import kotlin.math.exp
import kotlin.math.max

/**
 * RTMDet dynamic-conv mask head.
 *
 * Each detection carries 169 floats encoding a per-instance MLP (10 -> 8 -> 8 -> 1) that must be
 * evaluated at every mask pixel: ~152 MACs x 25,600 pixels = ~3.9M MACs per detection. The scalar
 * Kotlin implementation measured ~49 ms per plane on a Pixel 9a, about a hundred times the
 * arithmetic cost, because it computed one output at a time and gathered mask features across a
 * 102 KB channel stride.
 *
 * [nativeBuildMaskPlaneNhwc] is a NEON kernel that computes all eight outputs per input in two
 * float32x4 lanes and reads NHWC features, so a pixel's eight values share a cache line. When the
 * native library is unavailable the scalar path here produces the same plane, so behaviour never
 * depends on the library loading.
 */
internal object RTMDetMaskHead {

    private const val TAG = "RTMDetMaskHead"
    private const val COEFF_COUNT = 169
    private const val CHANNELS = 8

    private const val W1 = 0
    private const val W2 = W1 + 80
    private const val W3 = W2 + 64
    private const val B1 = W3 + 8
    private const val B2 = B1 + 8
    private const val B3 = B2 + 8

    val isNativeAvailable: Boolean = try {
        System.loadLibrary("furnit_rtmdet")
        LogUtil.i(TAG, "NEON mask head loaded")
        true
    } catch (error: Throwable) {
        LogUtil.w(TAG, "NEON mask head unavailable, using scalar mask head: ${error.message}")
        false
    }

    @JvmStatic
    private external fun nativeBuildMaskPlaneNhwc(
        coeffs: FloatArray,
        maskFeatNhwc: FloatArray,
        maskSide: Int,
        relXBase: Float,
        relXStep: Float,
        relYBase: Float,
        relYStep: Float,
        out: FloatArray,
    ): Boolean

    /**
     * Evaluates the mask head into a `maskSide * maskSide` plane of sigmoid values.
     *
     * [maskFeatIsNhwc] selects the feature layout: NHWC (`pixel * 8 + channel`) is what LiteRT
     * emits natively and what the NEON kernel needs; NCHW (`channel * pixels + pixel`) is only
     * produced by the retired ONNX path and always takes the scalar route.
     *
     * Returns null when the inputs do not satisfy the contract.
     */
    fun buildPlane(
        coeffs: FloatArray,
        maskFeat: FloatArray,
        maskFeatIsNhwc: Boolean,
        maskSide: Int,
        inputSize: Int,
        priorX: Float,
        priorY: Float,
        levelStride: Float,
    ): FloatArray? {
        if (coeffs.size != COEFF_COUNT || maskSide <= 0) return null
        val pixels = maskSide * maskSide
        if (maskFeat.size < pixels * CHANNELS) return null

        // relX depends only on x and relY only on y, so express both as affine terms and let the
        // kernels hoist the row component out of the inner loop:
        //   relX = (priorX - (x + 0.5) * maskStride) / denominator
        val maskStride = inputSize.toFloat() / maskSide.toFloat()
        val denominator = max(1f, levelStride * 8f)
        val relXBase = (priorX - 0.5f * maskStride) / denominator
        val relXStep = -maskStride / denominator
        val relYBase = (priorY - 0.5f * maskStride) / denominator
        val relYStep = -maskStride / denominator

        val out = FloatArray(pixels)
        if (isNativeAvailable && maskFeatIsNhwc) {
            val ok = try {
                nativeBuildMaskPlaneNhwc(
                    coeffs, maskFeat, maskSide,
                    relXBase, relXStep, relYBase, relYStep,
                    out,
                )
            } catch (error: Throwable) {
                LogUtil.e(TAG, "NEON mask head failed, falling back to scalar: ${error.message}", error)
                false
            }
            if (ok) return out
        }

        buildPlaneScalar(
            coeffs = coeffs,
            maskFeat = maskFeat,
            maskFeatIsNhwc = maskFeatIsNhwc,
            maskSide = maskSide,
            relXBase = relXBase,
            relXStep = relXStep,
            relYBase = relYBase,
            relYStep = relYStep,
            out = out,
        )
        return out
    }

    /** Reference implementation. Also the fallback whenever the native library is not loaded. */
    fun buildPlaneScalar(
        coeffs: FloatArray,
        maskFeat: FloatArray,
        maskFeatIsNhwc: Boolean,
        maskSide: Int,
        relXBase: Float,
        relXStep: Float,
        relYBase: Float,
        relYStep: Float,
        out: FloatArray,
    ) {
        val pixels = maskSide * maskSide
        val input = FloatArray(10)
        val hidden1 = FloatArray(CHANNELS)
        val hidden2 = FloatArray(CHANNELS)

        for (y in 0 until maskSide) {
            val relY = relYBase + relYStep * y.toFloat()
            for (x in 0 until maskSide) {
                val pos = y * maskSide + x
                input[0] = relXBase + relXStep * x.toFloat()
                input[1] = relY
                if (maskFeatIsNhwc) {
                    val base = pos * CHANNELS
                    for (c in 0 until CHANNELS) input[2 + c] = maskFeat[base + c]
                } else {
                    for (c in 0 until CHANNELS) input[2 + c] = maskFeat[c * pixels + pos]
                }

                for (o in 0 until CHANNELS) {
                    var sum = coeffs[B1 + o]
                    for (i in 0 until 10) {
                        sum += coeffs[W1 + o * 10 + i] * input[i]
                    }
                    hidden1[o] = max(0f, sum)
                }

                for (o in 0 until CHANNELS) {
                    var sum = coeffs[B2 + o]
                    for (i in 0 until CHANNELS) {
                        sum += coeffs[W2 + o * CHANNELS + i] * hidden1[i]
                    }
                    hidden2[o] = max(0f, sum)
                }

                var logit = coeffs[B3]
                for (i in 0 until CHANNELS) {
                    logit += coeffs[W3 + i] * hidden2[i]
                }
                out[pos] = sigmoid(logit)
            }
        }
    }

    private fun sigmoid(value: Float): Float {
        return if (value >= 0f) {
            val z = exp(-value.toDouble())
            (1.0 / (1.0 + z)).toFloat()
        } else {
            val z = exp(value.toDouble())
            (z / (1.0 + z)).toFloat()
        }
    }
}
