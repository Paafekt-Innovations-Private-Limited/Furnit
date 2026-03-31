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
// Linear sampling avoids blocky resample when copying scratch → drawable; light S-curve for contrast.
fragment float4 compositeOverGrayFragment(FullscreenV in [[stage_in]],
                                          texture2d<float> splatTex [[texture(0)]],
                                          constant float *params [[buffer(0)]]) {
    constexpr sampler s(filter::linear, mip_filter::none, address::clamp_to_edge);
    float exposure = params[0];
    float shadowLift = params[1];
    float4 c = splatTex.sample(s, in.uv);
    c.rgb *= exposure;
    // Soft highlight blend toward 1-exp(-x) only when rgb is high; midtones mostly unchanged.
    float3 highlightWeight = smoothstep(0.7, 1.0, c.rgb);
    c.rgb = mix(c.rgb, 1.0 - exp(-c.rgb), highlightWeight);
    // Gentle S-curve (blend toward full smoothstep to reduce mushy / over-soft mids)
    float3 curved = c.rgb * c.rgb * (3.0 - 2.0 * c.rgb);
    // Lower mix keeps mids from going uniformly milky when many splats have soft alpha.
    c.rgb = mix(c.rgb, curved, 0.38);
    float luma = dot(c.rgb, float3(0.299, 0.587, 0.114));
    float mask = 1.0 - smoothstep(0.0, 0.2, luma);
    c.rgb += shadowLift * mask;
    // Darker than 0.5 so (1−α) gray fill does not read as heavy fog over the splat field.
    float3 gray = float3(0.32, 0.32, 0.32);
    float3 result = c.rgb + gray * (1.0 - c.a);
    return float4(saturate(result), 1.0);
}
