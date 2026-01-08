#include <metal_stdlib>
using namespace metal;

// Composite mask over camera texture, output to output texture
// Note: Source is BGRA (s.r=B, s.g=G, s.b=R), output needs RGBA for CGContext
kernel void sp_compositeMask(texture2d<float, access::read>  src   [[texture(0)]],
                          texture2d<float, access::read>  mask  [[texture(1)]],
                          texture2d<float, access::write> out   [[texture(2)]],
                          uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) return;
    float4 s = src.read(gid);  // BGRA: s.r=Blue, s.g=Green, s.b=Red
    float  m = mask.read(gid).r; // R8Unorm -> normalized float in [0,1]
    // Swap B and R for correct RGB output
    float4 outCol = float4(s.b, s.g, s.r, m);  // R, G, B, A
    out.write(outCol, gid);
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

    // Nearest neighbor sampling in prototype space for performance
    uint ix = (uint)round(px);
    uint iy = (uint)round(py);
    uint planeSize = pW * pH;
    uint protIdx = iy * pW + ix;

    // Compute max over detections of dot(prototypes[:, protIdx], coeffs[j][:])
    float maxLogit = -3.4e38f;
    for (uint j = 0; j < detCount; ++j) {
        float dotv = 0.0f;
        for (uint k = 0; k < 32; ++k) {
            float a = planes[k * planeSize + protIdx];
            float c = coeffs[j * 32 + k];
            dotv += a * c;
        }
        if (dotv > maxLogit) maxLogit = dotv;
    }

    // Threshold at 0 and composite
    // Note: Source is BGRA (s.r=B, s.g=G, s.b=R), output needs RGBA
    float4 out = float4(0.0, 0.0, 0.0, 0.0);
    if (maxLogit > 0.0f) {
        float4 s = src.read(uint2(gid.x, gid.y));
        out = float4(s.b, s.g, s.r, 1.0);  // Swap B and R for correct RGB output
    }
    outTex.write(out, gid);
}

