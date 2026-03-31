#include <metal_stdlib>
using namespace metal;

// Fullscreen composite: splats render to an intermediate RT with clear alpha 0 (premultiplied RGB = 0);
// this pass blends over mid-gray so empty pixels and edges are not black.

struct FullscreenV {
    float4 position [[position]];
    float2 uv;
};

vertex FullscreenV compositeOverGrayVertex(uint vid [[vertex_id]]) {
    FullscreenV out;
    out.uv = float2((vid << 1) & 2, vid & 2);
    out.position = float4(out.uv * 2.0 - 1.0, 0.0, 1.0);
    out.uv.y = 1.0 - out.uv.y;
    return out;
}

// Matches Swift `compositeParams`: [exposure, shadowLift] via `setFragmentBytes(..., index: 0)`.
struct CompositeParams {
    float exposure;
    float shadowLift;
};

// exposure + shadowLift: MetalSplatter 1.x outputs linear RGB from the splat shader (`sRGBToLinear` in SplatProcessing.metal).
// shadowLift nudges dark regions only (luma mask); use exposure ≈ 1 when PLY f_dc was corrected for Metal (`correctPLYColors`).
fragment float4 compositeOverGrayFragment(FullscreenV in [[stage_in]],
                                          texture2d<float> splatTex [[texture(0)]],
                                          constant CompositeParams &params [[buffer(0)]]) {
    constexpr sampler s(filter::linear);
    float4 c = splatTex.sample(s, in.uv);
    c.rgb *= params.exposure;
    float luma = dot(c.rgb, float3(0.299, 0.587, 0.114));
    float mask = 1.0 - smoothstep(0.0, 0.25, luma);
    c.rgb += params.shadowLift * mask;
    float3 gray = float3(0.5, 0.5, 0.5);
    float3 result = c.rgb + gray * (1.0 - c.a);
    return float4(result, 1.0);
}
