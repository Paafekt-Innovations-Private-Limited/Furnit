#include <metal_stdlib>
using namespace metal;

// Composite mask over camera texture, output to output texture
// Note: Source is BGRA (s.r=B, s.g=G, s.b=R), output texture is bgra8Unorm (memory order B,G,R,A)
kernel void sp_compositeMask(texture2d<float, access::read>  src   [[texture(0)]],
                          texture2d<float, access::read>  mask  [[texture(1)]],
                          texture2d<float, access::write> out   [[texture(2)]],
                          uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) return;
    float  m = mask.read(gid).r; // R8Unorm -> normalized float in [0,1]
    if (m <= 0.0f) {
        out.write(float4(0.0, 0.0, 0.0, 0.0), gid);
        return;
    }
    float4 s = src.read(gid);  // BGRA: s.r=Blue, s.g=Green, s.b=Red
    // Premultiplied alpha so soft mask values (from bilinear upscale) blend smoothly.
    out.write(float4(s.b * m, s.g * m, s.r * m, m), gid);
}

// Build maskSmall in prototype space: max over detections of dot(A[pixel], coeffs[j]) then threshold > 0
kernel void sp_maxMaskFromPrototypes(const device float *planes   [[buffer(0)]],
                                  const device float *coeffs   [[buffer(1)]],
                                  device uchar *outMask        [[buffer(2)]],
                                  constant uint &planeSize     [[buffer(3)]],
                                  constant uint &detCount      [[buffer(4)]],
                                  uint gid [[thread_position_in_grid]]) {
    if (gid >= planeSize) return;
    float maxLogit = -3.4e38f;
    // 32 prototype channels per pixel
    for (uint j = 0; j < detCount; ++j) {
        float dotv = 0.0f;
        for (uint k = 0; k < 32; ++k) {
            float a = planes[k * planeSize + gid];
            float c = coeffs[j * 32 + k];
            dotv += a * c;
        }
        if (dotv > maxLogit) maxLogit = dotv;
    }
    outMask[gid] = (maxLogit > 0.0f) ? (uchar)255 : (uchar)0;
}

// Bilinear sample of one prototype plane at fractional (px, py) in [0, pW-1]×[0, pH-1].
static inline float sp_samplePlaneBilinear(constant float *planes, uint pW, uint pH, uint planeSize, float px, float py, uint k) {
    px = clamp(px, 0.0f, float(pW - 1));
    py = clamp(py, 0.0f, float(pH - 1));
    float fx = floor(px);
    float fy = floor(py);
    uint x0 = uint(fx);
    uint y0 = uint(fy);
    uint x1 = min(x0 + 1, pW - 1);
    uint y1 = min(y0 + 1, pH - 1);
    float tx = px - fx;
    float ty = py - fy;
    uint b = k * planeSize;
    float v00 = planes[b + y0 * pW + x0];
    float v10 = planes[b + y0 * pW + x1];
    float v01 = planes[b + y1 * pW + x0];
    float v11 = planes[b + y1 * pW + x1];
    return mix(mix(v00, v10, tx), mix(v01, v11, tx), ty);
}

// Fused kernel: compute max(A·coeffs) in prototype space and composite into output
// Buffers:
//  b0: planes (float[32 * pW * pH]) laid out as 32 planes contiguous
//  b1: coeffs (float[detCount * 32]) row-major per detection
// Bytes:
//  i2: pW (uint32)
//  i3: pH (uint32)
//  i4: detCount (uint32)
//  i5: origW (uint32)
//  i6: origH (uint32)
//  i7: modelInput (uint32)  // expected 1280
//  i8: resizeGain (float)
//  i9: padX (float)
//  i10: padY (float)
//  i11: bx1 (uint32)
//  i12: by1 (uint32)
//  i13: bx2 (uint32)
//  i14: by2 (uint32)
// Textures:
//  t0: source BGRA8 camera frame (origW x origH)
//  t1: output BGRA8
kernel void sp_maxMaskAndComposite(texture2d<float, access::read>  src   [[texture(0)]],
                                    texture2d<float, access::write> outTex [[texture(1)]],
                                    constant float *planes [[buffer(0)]],
                                    constant float *coeffs [[buffer(1)]],
                                    constant uint &pW [[buffer(2)]],
                                    constant uint &pH [[buffer(3)]],
                                    constant uint &detCount [[buffer(4)]],
                                    constant uint &origW [[buffer(5)]],
                                    constant uint &origH [[buffer(6)]],
                                    constant uint &modelInput [[buffer(7)]],
                                    constant float &resizeGain [[buffer(8)]],
                                    constant float &padX [[buffer(9)]],
                                    constant float &padY [[buffer(10)]],
                                    constant uint &bx1 [[buffer(11)]],
                                    constant uint &by1 [[buffer(12)]],
                                    constant uint &bx2 [[buffer(13)]],
                                    constant uint &by2 [[buffer(14)]],
                                    uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= origW || gid.y >= origH) return;

    // Outside union bbox? Make transparent and early out
    if (gid.x < bx1 || gid.x >= bx2 || gid.y < by1 || gid.y >= by2) {
        outTex.write(float4(0.0, 0.0, 0.0, 0.0), gid);
        return;
    }

    // Map output pixel to model space (1280x1280) then to prototype space (pW x pH)
    float x_model = (float(gid.x) * resizeGain) + padX;
    float y_model = (float(gid.y) * resizeGain) + padY;

    // Normalize to [0,1] in model space, then scale to prototype grid
    float u = clamp(x_model / float(modelInput), 0.0f, 1.0f);
    float v = clamp(y_model / float(modelInput), 0.0f, 1.0f);
    float px = u * float(pW - 1);
    float py = v * float(pH - 1);

    uint planeSize = pW * pH;

    // Bilinear sampling in prototype space (avoids blocky / jagged mask edges from nearest-neighbor).
    float maxLogit = -3.4e38f;
    for (uint j = 0; j < detCount; ++j) {
        float dotv = 0.0f;
        for (uint k = 0; k < 32; ++k) {
            float a = sp_samplePlaneBilinear(planes, pW, pH, planeSize, px, py, k);
            float c = coeffs[j * 32 + k];
            dotv += a * c;
        }
        if (dotv > maxLogit) maxLogit = dotv;
    }

    // Soft alpha across the decision boundary (premultiplied BGRA for smooth edges).
    const float edgeW = 0.22f;
    float a = smoothstep(-edgeW, edgeW, maxLogit);
    float4 s = src.read(uint2(gid.x, gid.y));
    outTex.write(float4(s.b * a, s.g * a, s.r * a, a), gid);
}

