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

// Swift: `setFragmentBytes` with [exposure, shadowLift] at index 0.
// Passthrough composite: keep splat colors as-is and only fill empty pixels with a neutral gray background.
fragment float4 compositeOverGrayFragment(FullscreenV in [[stage_in]],
                                          texture2d<float> splatTex [[texture(0)]],
                                          constant float *params [[buffer(0)]]) {
    constexpr sampler s(filter::linear, mip_filter::none, address::clamp_to_edge);
    float exposure = params[0];
    float shadowLift = params[1];
    float4 c = splatTex.sample(s, in.uv);
    // Exposure and shadowLift are set to neutral values from Swift (1.0 and 0.0 respectively).
    c.rgb *= exposure;
    // Subtle gray background so “void” pixels are present without harsh black borders.
    float3 gray = float3(0.24, 0.24, 0.24);
    float3 result = c.rgb + gray * (1.0 - c.a);
    return float4(saturate(result), 1.0);
}
