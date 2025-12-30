#include <metal_stdlib>
using namespace metal;

// planes: [32 * planeSize] float, where planes[k*planeSize + i] is prototype k at pixel i
// coeffs: [detCount * 32] float, row-major: coeffs[det*32 + k]
// outMask: [planeSize] uint8, 0 or 255
kernel void maxMaskFromPrototypes(
    device const float* planes        [[ buffer(0) ]],
    device const float* coeffs        [[ buffer(1) ]],
    device uchar*       outMask       [[ buffer(2) ]],
    constant uint&      planeSize     [[ buffer(3) ]],
    constant uint&      detCount      [[ buffer(4) ]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= planeSize) return;

    float maxLogit = -INFINITY;

    // For each detection: dot32(coeffs, prototypes[:,gid])
    for (uint d = 0; d < detCount; ++d) {
        float acc = 0.0f;
        uint cBase = d * 32u;
        // Unrolled 32 MACs
        #pragma unroll
        for (uint k = 0; k < 32u; ++k) {
            acc += coeffs[cBase + k] * planes[k * planeSize + gid];
        }
        maxLogit = max(maxLogit, acc);
    }

    // Same thresholding as CPU path: maxLogit > 0 => 255
    outMask[gid] = (maxLogit > 0.0f) ? (uchar)255 : (uchar)0;
}
