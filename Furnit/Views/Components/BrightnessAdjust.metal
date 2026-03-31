#include <metal_stdlib>
using namespace metal;

// Post-process after splat rendering: `brightnessAdjust` multiplies RGB (exposure from buffer 0).
// The drawable texture cannot be used as a compute read_write target on iOS; Swift renders splats
// to an intermediate texture, runs this kernel, then uses the fullscreen pass below to the drawable.

// Full-screen triangle: present an off-screen color texture to the drawable.
// ``CAMetalDrawable.texture`` is not a valid blit destination on iOS → use a render pass instead.

struct FullscreenBlitVertexOut {
    float4 position [[position]];
    float2 uv;
};

vertex FullscreenBlitVertexOut fullscreenBlitVertex(uint vid [[vertex_id]]) {
    FullscreenBlitVertexOut out;
    float2 pos;
    if (vid == 0) {
        pos = float2(-1.0, -1.0);
    } else if (vid == 1) {
        pos = float2(3.0, -1.0);
    } else {
        pos = float2(-1.0, 3.0);
    }
    out.position = float4(pos, 0.0, 1.0);
    out.uv = float2((pos.x + 1.0) * 0.5, (1.0 - pos.y) * 0.5);
    return out;
}

fragment float4 fullscreenBlitFragment(
    FullscreenBlitVertexOut in [[stage_in]],
    texture2d<float> source [[texture(0)]])
{
    constexpr sampler texSampler(coord::normalized, filter::nearest, address::clamp_to_edge);
    return source.sample(texSampler, in.uv);
}

kernel void brightnessAdjust(
    texture2d<float, access::read_write> tex [[texture(0)]],
    constant float &exposure [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= tex.get_width() || gid.y >= tex.get_height()) {
        return;
    }
    float4 c = tex.read(gid);
    c.rgb = min(c.rgb * exposure, 1.0);
    tex.write(c, gid);
}
