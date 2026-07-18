#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// process_mask_native: bilinear upsample (align_corners=False) + threshold >0
// ---------------------------------------------------------------------------

struct BilinearUpsampleParams {
    uint protoW;
    uint protoH;
    uint origW;
    uint origH;
    uint xStart;
    uint yStart;
    uint bandW;
    uint bandH;
    float modelInput;
    float imageToModelScaleX;
    float imageToModelScaleY;
    float padX;
    float padY;
    float logitThreshold;
};

/// Bilinear-upsample logits from proto space to image space (band region only)
/// and threshold at `logitThreshold`.  Output is a UInt8 band mask (255 = opaque, 0 = clear).
/// Matches PyTorch F.interpolate(mode='bilinear', align_corners=False).
kernel void sp_bilinearUpsampleThreshold(
    device const float* logits         [[buffer(0)]],
    device uchar*       mask           [[buffer(1)]],
    constant BilinearUpsampleParams& p [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= p.bandW || gid.y >= p.bandH) return;

    uint imgX = p.xStart + gid.x;
    uint imgY = p.yStart + gid.y;

    float protoScaleX = float(p.protoW) / p.modelInput;
    float protoScaleY = float(p.protoH) / p.modelInput;
    float modelCenterX = (float(imgX) + 0.5f) * p.imageToModelScaleX + p.padX;
    float modelCenterY = (float(imgY) + 0.5f) * p.imageToModelScaleY + p.padY;
    float fx = modelCenterX * protoScaleX - 0.5f;
    float fy = modelCenterY * protoScaleY - 0.5f;

    int maxPx = int(p.protoW) - 1;
    int maxPy = int(p.protoH) - 1;

    int px0 = clamp(int(floor(fx)), 0, maxPx);
    int px1 = clamp(px0 + 1,        0, maxPx);
    int py0 = clamp(int(floor(fy)), 0, maxPy);
    int py1 = clamp(py0 + 1,        0, maxPy);

    float tx = fx - float(px0);
    float ty = fy - float(py0);

    float v00 = logits[py0 * int(p.protoW) + px0];
    float v10 = logits[py0 * int(p.protoW) + px1];
    float v01 = logits[py1 * int(p.protoW) + px0];
    float v11 = logits[py1 * int(p.protoW) + px1];

    float logit =
        v00 * (1.0f - tx) * (1.0f - ty) +
        v10 * tx          * (1.0f - ty) +
        v01 * (1.0f - tx) * ty          +
        v11 * tx          * ty;

    mask[gid.y * p.bandW + gid.x] = (logit > p.logitThreshold) ? (uchar)255 : (uchar)0;
}
