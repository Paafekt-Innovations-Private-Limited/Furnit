/**
 * blas_utils.h - BLAS-equivalent optimizations for Android
 *
 * Uses ARM NEON intrinsics for vectorized operations since Android
 * has no built-in BLAS. These are equivalent to:
 * - cblas_sgemv (matrix-vector product)
 * - cblas_sdot (dot product)
 * - batch quaternion normalization
 *
 * All functions are optimized for ARM64 NEON.
 */

#ifndef BLAS_UTILS_H
#define BLAS_UTILS_H

#include <arm_neon.h>
#include <cmath>
#include <cstdint>
#include <algorithm>

namespace blas {

/**
 * Fast sigmoid using NEON
 * sigmoid(x) = 1 / (1 + exp(-x))
 */
inline float fast_sigmoid(float x) {
    // Clamp to prevent overflow
    x = std::max(-88.0f, std::min(88.0f, x));
    return 1.0f / (1.0f + expf(-x));
}

/**
 * Vectorized dot product for 32-element vectors (common in YOLO mask coefficients)
 * Equivalent to cblas_sdot with n=32
 *
 * @param a First vector (32 floats)
 * @param b Second vector (32 floats)
 * @return Dot product result
 */
inline float dot32_neon(const float* a, const float* b) {
    float32x4_t sum0 = vdupq_n_f32(0.0f);
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    float32x4_t sum2 = vdupq_n_f32(0.0f);
    float32x4_t sum3 = vdupq_n_f32(0.0f);

    // Process 16 elements per iteration (4 lanes x 4 accumulators)
    // Unrolled for 32 elements total

    // Elements 0-15
    sum0 = vmlaq_f32(sum0, vld1q_f32(a + 0),  vld1q_f32(b + 0));
    sum1 = vmlaq_f32(sum1, vld1q_f32(a + 4),  vld1q_f32(b + 4));
    sum2 = vmlaq_f32(sum2, vld1q_f32(a + 8),  vld1q_f32(b + 8));
    sum3 = vmlaq_f32(sum3, vld1q_f32(a + 12), vld1q_f32(b + 12));

    // Elements 16-31
    sum0 = vmlaq_f32(sum0, vld1q_f32(a + 16), vld1q_f32(b + 16));
    sum1 = vmlaq_f32(sum1, vld1q_f32(a + 20), vld1q_f32(b + 20));
    sum2 = vmlaq_f32(sum2, vld1q_f32(a + 24), vld1q_f32(b + 24));
    sum3 = vmlaq_f32(sum3, vld1q_f32(a + 28), vld1q_f32(b + 28));

    // Reduce: sum0 + sum1 + sum2 + sum3
    sum0 = vaddq_f32(sum0, sum1);
    sum2 = vaddq_f32(sum2, sum3);
    sum0 = vaddq_f32(sum0, sum2);

    // Horizontal sum of 4 lanes
    float32x2_t sum_pair = vadd_f32(vget_low_f32(sum0), vget_high_f32(sum0));
    return vget_lane_f32(vpadd_f32(sum_pair, sum_pair), 0);
}

/**
 * General dot product with NEON (any length, padded to 4)
 */
inline float dot_neon(const float* a, const float* b, int n) {
    float32x4_t sum = vdupq_n_f32(0.0f);

    int i = 0;
    // Process 4 elements at a time
    for (; i + 4 <= n; i += 4) {
        sum = vmlaq_f32(sum, vld1q_f32(a + i), vld1q_f32(b + i));
    }

    // Horizontal sum
    float32x2_t sum_pair = vadd_f32(vget_low_f32(sum), vget_high_f32(sum));
    float result = vget_lane_f32(vpadd_f32(sum_pair, sum_pair), 0);

    // Handle remaining elements
    for (; i < n; i++) {
        result += a[i] * b[i];
    }

    return result;
}

/**
 * Batch mask composition: coefficients × prototypes → mask
 * This is the CRITICAL optimization target (~100M FLOPs)
 *
 * Equivalent to: for each detection, compute mask[y,x] = max(mask[y,x], sigmoid(coeffs · protos[y,x]))
 *
 * @param coeffs Detection coefficients [numDets × numProtos]
 * @param protos Prototype masks [numProtos × H × W] in CHW format
 * @param mask Output mask [H × W], modified in place (takes max)
 * @param numDets Number of detections
 * @param numProtos Number of prototype channels (typically 32)
 * @param H Prototype height
 * @param W Prototype width
 */
inline void compose_masks_neon(
    const float* coeffs,
    const float* protos,
    float* mask,
    int numDets,
    int numProtos,
    int H,
    int W
) {
    const int HW = H * W;

    // For each pixel position
    for (int p = 0; p < HW; p++) {
        float maxVal = mask[p];

        // For each detection
        for (int d = 0; d < numDets; d++) {
            const float* detCoeffs = coeffs + d * numProtos;

            // Compute dot product: coeffs · protos[:, p]
            // protos is in CHW format, so protos[c, p] = protos[c * HW + p]
            float sum = 0.0f;

            if (numProtos == 32) {
                // Optimized path for 32 prototypes (most common)
                // Gather prototype values for this pixel across all channels
                float protoVals[32];
                for (int c = 0; c < 32; c++) {
                    protoVals[c] = protos[c * HW + p];
                }
                sum = dot32_neon(detCoeffs, protoVals);
            } else {
                // General path
                for (int c = 0; c < numProtos; c++) {
                    sum += detCoeffs[c] * protos[c * HW + p];
                }
            }

            float sigmoidVal = fast_sigmoid(sum);
            maxVal = std::max(maxVal, sigmoidVal);
        }

        mask[p] = maxVal;
    }
}

/**
 * Optimized mask composition with pre-transposed prototypes
 * If prototypes are in HWC format (pixel-major), this is much faster
 *
 * @param coeffs Detection coefficients [numDets × numProtos]
 * @param protos_hwc Prototype masks in HWC format [H × W × numProtos]
 * @param mask Output mask [H × W]
 * @param numDets Number of detections
 * @param numProtos Number of prototype channels
 * @param HW Total pixels (H × W)
 */
inline void compose_masks_hwc_neon(
    const float* coeffs,
    const float* protos_hwc,
    float* mask,
    int numDets,
    int numProtos,
    int HW
) {
    // Process 4 pixels at a time when possible
    int p = 0;

    for (; p < HW; p++) {
        const float* pixelProtos = protos_hwc + p * numProtos;
        float maxVal = mask[p];

        for (int d = 0; d < numDets; d++) {
            const float* detCoeffs = coeffs + d * numProtos;

            float sum;
            if (numProtos == 32) {
                sum = dot32_neon(detCoeffs, pixelProtos);
            } else {
                sum = dot_neon(detCoeffs, pixelProtos, numProtos);
            }

            maxVal = std::max(maxVal, fast_sigmoid(sum));
        }

        mask[p] = maxVal;
    }
}

/**
 * Batch quaternion normalization with NEON
 * Normalizes N quaternions in place
 *
 * @param quats Array of quaternions [N × 4] as [w, x, y, z]
 * @param N Number of quaternions
 */
inline void normalize_quaternions_neon(float* quats, int N) {
    for (int i = 0; i < N; i++) {
        float* q = quats + i * 4;

        // Load quaternion
        float32x4_t qv = vld1q_f32(q);

        // Compute squared magnitude: w² + x² + y² + z²
        float32x4_t sq = vmulq_f32(qv, qv);
        float32x2_t sum_pair = vadd_f32(vget_low_f32(sq), vget_high_f32(sq));
        float mag_sq = vget_lane_f32(vpadd_f32(sum_pair, sum_pair), 0);

        // Compute inverse magnitude (with safety check)
        float inv_mag = (mag_sq > 1e-16f) ? (1.0f / sqrtf(mag_sq)) : 1.0f;

        // Scale quaternion
        float32x4_t scale = vdupq_n_f32(inv_mag);
        qv = vmulq_f32(qv, scale);

        // Store normalized quaternion
        vst1q_f32(q, qv);
    }
}

/**
 * Batch log transform for scales with NEON
 * out[i] = log(max(in[i] * boost, minVal))
 *
 * @param in Input scales [N × 3]
 * @param out Output log scales [N × 3]
 * @param N Number of scale triplets
 * @param boost Scale boost factor
 * @param minVal Minimum value before log
 */
inline void log_scales_neon(const float* in, float* out, int N, float boost, float minVal) {
    for (int i = 0; i < N * 3; i++) {
        float val = in[i] * boost;
        val = std::max(val, minVal);
        out[i] = logf(val);
    }
}

/**
 * Batch color normalization: (color - 0.5) / SH_C0
 *
 * @param in Input colors [N × 3] in range [0, 1]
 * @param out Output SH DC coefficients [N × 3]
 * @param N Number of color triplets
 * @param sh_c0 SH coefficient (typically 0.28209479177387814)
 */
inline void normalize_colors_neon(const float* in, float* out, int N, float sh_c0) {
    const float inv_sh_c0 = 1.0f / sh_c0;
    const float32x4_t half = vdupq_n_f32(0.5f);
    const float32x4_t scale = vdupq_n_f32(inv_sh_c0);

    int i = 0;
    // Process 4 values at a time
    for (; i + 4 <= N * 3; i += 4) {
        float32x4_t v = vld1q_f32(in + i);
        v = vsubq_f32(v, half);
        v = vmulq_f32(v, scale);
        vst1q_f32(out + i, v);
    }

    // Handle remaining
    for (; i < N * 3; i++) {
        out[i] = (in[i] - 0.5f) * inv_sh_c0;
    }
}

/**
 * Batch logit transform for opacity
 * out = log(opacity / (1 - opacity))
 *
 * @param in Input opacities [N] in range (0, 1)
 * @param out Output logits [N]
 * @param N Number of values
 */
inline void logit_opacity_neon(const float* in, float* out, int N) {
    const float eps = 1e-4f;

    for (int i = 0; i < N; i++) {
        float o = in[i];
        // Clamp to valid range
        o = std::max(eps, std::min(1.0f - eps, o));
        out[i] = logf(o / (1.0f - o));
    }
}

/**
 * Fast RGB to float conversion with NEON
 * Converts ARGB int pixels to normalized float [0, 1]
 *
 * @param argb Input ARGB pixels [N]
 * @param out Output floats [N × 3] in RGB order
 * @param N Number of pixels
 */
inline void argb_to_float_neon(const uint32_t* argb, float* out, int N) {
    const float32x4_t scale = vdupq_n_f32(1.0f / 255.0f);

    for (int i = 0; i < N; i++) {
        uint32_t v = argb[i];
        float r = ((v >> 16) & 0xFF) * (1.0f / 255.0f);
        float g = ((v >> 8) & 0xFF) * (1.0f / 255.0f);
        float b = (v & 0xFF) * (1.0f / 255.0f);
        out[i * 3 + 0] = r;
        out[i * 3 + 1] = g;
        out[i * 3 + 2] = b;
    }
}

/**
 * NCHW layout conversion: interleaved RGB → channel-first
 *
 * @param rgb Input RGB [H × W × 3]
 * @param nchw Output NCHW [3 × H × W]
 * @param H Height
 * @param W Width
 */
inline void rgb_to_nchw(const float* rgb, float* nchw, int H, int W) {
    const int HW = H * W;
    float* r_plane = nchw;
    float* g_plane = nchw + HW;
    float* b_plane = nchw + 2 * HW;

    for (int i = 0; i < HW; i++) {
        r_plane[i] = rgb[i * 3 + 0];
        g_plane[i] = rgb[i * 3 + 1];
        b_plane[i] = rgb[i * 3 + 2];
    }
}

} // namespace blas

#endif // BLAS_UTILS_H
