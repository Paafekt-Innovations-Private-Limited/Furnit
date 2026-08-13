// RTMDet dynamic-conv mask head, NEON accelerated.
//
// Each detection carries 169 floats that encode a tiny per-instance MLP, evaluated at every
// one of the 160x160 mask pixels:
//
//   input (10) : [relX, relY, feat0..feat7]
//   layer 1    : W1[8x10], b1[8] -> ReLU
//   layer 2    : W2[8x8],  b2[8] -> ReLU
//   layer 3    : W3[1x8],  b3    -> sigmoid
//
//   coeff packing: W1 = [0,80)  W2 = [80,144)  W3 = [144,152)
//                  b1 = [152,160)  b2 = [160,168)  b3 = [168]
//
// That is ~152 MACs x 25,600 pixels = ~3.9M MACs per detection. The Kotlin scalar version
// measured ~49 ms per plane on a Pixel 9a, roughly a hundred times the arithmetic cost,
// because it evaluated one output at a time and gathered mask features across a channel
// stride of 102 KB.
//
// This kernel fixes both halves. Mask features arrive NHWC, so the eight values for a pixel
// sit in one cache line, and the eight MLP outputs are computed together in two float32x4
// lanes: the weights are pre-transposed to column-major so each input contributes a single
// vector FMA across all eight outputs.
//
// arm64-v8a only, so NEON is unconditional and needs no runtime dispatch.

#include <jni.h>

#include <arm_neon.h>
#include <cmath>
#include <cstdint>

namespace {

constexpr int kMlpInputs = 10;
constexpr int kMlpHidden = 8;
constexpr int kCoeffCount = 169;

constexpr int kW1 = 0;
constexpr int kW2 = kW1 + 80;
constexpr int kW3 = kW2 + 64;
constexpr int kB1 = kW3 + 8;
constexpr int kB2 = kB1 + 8;
constexpr int kB3 = kB2 + 8;

// Matches the Kotlin reference: branch on sign so neither exp overflows.
inline float Sigmoid(float value) {
    if (value >= 0.0f) {
        return 1.0f / (1.0f + std::exp(-value));
    }
    const float z = std::exp(value);
    return z / (1.0f + z);
}

struct Float8 {
    float32x4_t lo;
    float32x4_t hi;
};

inline Float8 Load8(const float* source) {
    return Float8{vld1q_f32(source), vld1q_f32(source + 4)};
}

inline Float8 FusedMultiplyAdd(Float8 accumulator, const Float8& weights, float scalar) {
    accumulator.lo = vfmaq_n_f32(accumulator.lo, weights.lo, scalar);
    accumulator.hi = vfmaq_n_f32(accumulator.hi, weights.hi, scalar);
    return accumulator;
}

inline Float8 Relu(Float8 value) {
    const float32x4_t zero = vdupq_n_f32(0.0f);
    value.lo = vmaxq_f32(value.lo, zero);
    value.hi = vmaxq_f32(value.hi, zero);
    return value;
}

inline void Store8(const Float8& value, float* destination) {
    vst1q_f32(destination, value.lo);
    vst1q_f32(destination + 4, value.hi);
}

}  // namespace

extern "C" JNIEXPORT jboolean JNICALL
Java_com_furnit_android_services_RTMDetMaskHead_nativeBuildMaskPlaneNhwc(
        JNIEnv* env,
        jclass,
        jfloatArray coeffsArray,
        jfloatArray maskFeatNhwcArray,
        jint maskSide,
        jfloat relXBase,
        jfloat relXStep,
        jfloat relYBase,
        jfloat relYStep,
        jfloatArray outArray) {
    if (coeffsArray == nullptr || maskFeatNhwcArray == nullptr || outArray == nullptr) {
        return JNI_FALSE;
    }
    if (maskSide <= 0) {
        return JNI_FALSE;
    }

    const jsize coeffCount = env->GetArrayLength(coeffsArray);
    const jsize featCount = env->GetArrayLength(maskFeatNhwcArray);
    const jsize outCount = env->GetArrayLength(outArray);
    const int64_t pixels = static_cast<int64_t>(maskSide) * static_cast<int64_t>(maskSide);
    if (coeffCount != kCoeffCount ||
        featCount < pixels * kMlpHidden ||
        outCount < pixels) {
        return JNI_FALSE;
    }

    auto* coeffs = static_cast<float*>(env->GetPrimitiveArrayCritical(coeffsArray, nullptr));
    if (coeffs == nullptr) {
        return JNI_FALSE;
    }
    auto* feat = static_cast<float*>(env->GetPrimitiveArrayCritical(maskFeatNhwcArray, nullptr));
    if (feat == nullptr) {
        env->ReleasePrimitiveArrayCritical(coeffsArray, coeffs, JNI_ABORT);
        return JNI_FALSE;
    }
    auto* out = static_cast<float*>(env->GetPrimitiveArrayCritical(outArray, nullptr));
    if (out == nullptr) {
        env->ReleasePrimitiveArrayCritical(maskFeatNhwcArray, feat, JNI_ABORT);
        env->ReleasePrimitiveArrayCritical(coeffsArray, coeffs, JNI_ABORT);
        return JNI_FALSE;
    }

    // Transpose both weight matrices to column-major once per detection, so an input value
    // becomes one broadcast FMA across all eight outputs instead of eight separate dot products.
    float w1Columns[kMlpInputs][kMlpHidden];
    for (int input = 0; input < kMlpInputs; ++input) {
        for (int output = 0; output < kMlpHidden; ++output) {
            w1Columns[input][output] = coeffs[kW1 + output * kMlpInputs + input];
        }
    }
    float w2Columns[kMlpHidden][kMlpHidden];
    for (int input = 0; input < kMlpHidden; ++input) {
        for (int output = 0; output < kMlpHidden; ++output) {
            w2Columns[input][output] = coeffs[kW2 + output * kMlpHidden + input];
        }
    }

    Float8 w1Vectors[kMlpInputs];
    for (int input = 0; input < kMlpInputs; ++input) {
        w1Vectors[input] = Load8(w1Columns[input]);
    }
    Float8 w2Vectors[kMlpHidden];
    for (int input = 0; input < kMlpHidden; ++input) {
        w2Vectors[input] = Load8(w2Columns[input]);
    }

    const Float8 bias1 = Load8(coeffs + kB1);
    const Float8 bias2 = Load8(coeffs + kB2);
    const float32x4_t w3Low = vld1q_f32(coeffs + kW3);
    const float32x4_t w3High = vld1q_f32(coeffs + kW3 + 4);
    const float bias3 = coeffs[kB3];

    float hidden1[kMlpHidden];
    const float* featCursor = feat;
    int64_t pixel = 0;

    for (int y = 0; y < maskSide; ++y) {
        // relX depends only on x and relY only on y, so the caller passes the affine terms and
        // the row value is hoisted out of the inner loop.
        const float relY = relYBase + relYStep * static_cast<float>(y);
        const Float8 rowBias = FusedMultiplyAdd(bias1, w1Vectors[1], relY);

        for (int x = 0; x < maskSide; ++x, ++pixel, featCursor += kMlpHidden) {
            const float relX = relXBase + relXStep * static_cast<float>(x);

            Float8 accumulator = FusedMultiplyAdd(rowBias, w1Vectors[0], relX);
            accumulator = FusedMultiplyAdd(accumulator, w1Vectors[2], featCursor[0]);
            accumulator = FusedMultiplyAdd(accumulator, w1Vectors[3], featCursor[1]);
            accumulator = FusedMultiplyAdd(accumulator, w1Vectors[4], featCursor[2]);
            accumulator = FusedMultiplyAdd(accumulator, w1Vectors[5], featCursor[3]);
            accumulator = FusedMultiplyAdd(accumulator, w1Vectors[6], featCursor[4]);
            accumulator = FusedMultiplyAdd(accumulator, w1Vectors[7], featCursor[5]);
            accumulator = FusedMultiplyAdd(accumulator, w1Vectors[8], featCursor[6]);
            accumulator = FusedMultiplyAdd(accumulator, w1Vectors[9], featCursor[7]);
            Store8(Relu(accumulator), hidden1);

            Float8 second = bias2;
            second = FusedMultiplyAdd(second, w2Vectors[0], hidden1[0]);
            second = FusedMultiplyAdd(second, w2Vectors[1], hidden1[1]);
            second = FusedMultiplyAdd(second, w2Vectors[2], hidden1[2]);
            second = FusedMultiplyAdd(second, w2Vectors[3], hidden1[3]);
            second = FusedMultiplyAdd(second, w2Vectors[4], hidden1[4]);
            second = FusedMultiplyAdd(second, w2Vectors[5], hidden1[5]);
            second = FusedMultiplyAdd(second, w2Vectors[6], hidden1[6]);
            second = FusedMultiplyAdd(second, w2Vectors[7], hidden1[7]);
            second = Relu(second);

            const float32x4_t weighted = vfmaq_f32(vmulq_f32(second.lo, w3Low), second.hi, w3High);
            out[pixel] = Sigmoid(bias3 + vaddvq_f32(weighted));
        }
    }

    env->ReleasePrimitiveArrayCritical(outArray, out, 0);
    env->ReleasePrimitiveArrayCritical(maskFeatNhwcArray, feat, JNI_ABORT);
    env->ReleasePrimitiveArrayCritical(coeffsArray, coeffs, JNI_ABORT);
    return JNI_TRUE;
}
