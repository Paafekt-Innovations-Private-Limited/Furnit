#include <metal_stdlib>
using namespace metal;

// MARK: - Vertex Input (float4 to match Swift SIMD4<Float>)
struct SplatVertex {
    float4 posRadius [[attribute(0)]];  // xyz = position, w = radius
    float4 colAlpha  [[attribute(1)]];  // rgb = color, a = opacity
};

// MARK: - Camera Uniforms (must match Swift CameraUniforms)
struct CameraUniforms {
    float4x4 viewMatrix;
    float4x4 projectionMatrix;
    uint debugScreenSpace;  // 0 = normal camera, 1 = screen-space bypass
    uint3 padding;          // Pad to 16-byte alignment
};

// MARK: - Vertex Output
struct VSOut {
    float4 position [[position]];
    float4 color;
    float  radius;
    float  pointSize [[point_size]];
};

// MARK: - Main Vertex Shader
vertex VSOut splat_vertex(SplatVertex v [[stage_in]],
                          constant CameraUniforms &camera [[buffer(1)]]) {
    VSOut out;
    out.color = v.colAlpha;
    out.radius = v.posRadius.w;

    // DEBUG: Screen-space bypass - normalize XY to fill screen
    if (camera.debugScreenSpace == 1) {
        // Map SHARP coordinates directly to NDC [-1, +1]
        // SHARP XY range is roughly [-1, +1], Z is depth
        // We'll map X,Y directly and ignore Z for 2D view
        float x = v.posRadius.x;
        float y = v.posRadius.y;

        // Scale to fill more of the screen (SHARP coords are small)
        // Multiply by 2-3x to make the cloud bigger
        float scale = 2.5;
        float ndcX = x * scale;
        float ndcY = y * scale;

        out.position = float4(ndcX, ndcY, 0.0, 1.0);
        out.pointSize = clamp(v.posRadius.w * 100.0, 4.0, 64.0);

        return out;
    }

    // Normal camera path
    float4 worldPos = float4(v.posRadius.xyz, 1.0);

    // Transform through view and projection
    float4 viewPos = camera.viewMatrix * worldPos;
    float4 clipPos = camera.projectionMatrix * viewPos;

    out.position = clipPos;

    // Point size based on distance (perspective scaling)
    float dist = max(-viewPos.z, 0.1);  // Distance from camera
    float baseSize = v.posRadius.w * 500.0;  // Scale factor for visibility
    out.pointSize = clamp(baseSize / dist, 1.0, 64.0);

    return out;
}

// MARK: - Debug Vertex Shader (bypass camera, direct 2D mapping)
// Use this to verify data packing is correct before debugging camera issues
vertex VSOut splat_vertex_debug(SplatVertex v [[stage_in]],
                                constant CameraUniforms &camera [[buffer(1)]]) {
    VSOut out;

    // Direct 2D mapping: bypass camera matrices entirely
    // X: room is -2 to +2, map to NDC -0.5 to +0.5
    // Y: room is 0 to 2.8, map to NDC -1 to +1
    float ndcX = v.posRadius.x * 0.25;
    float ndcY = v.posRadius.y / 2.8 * 2.0 - 1.0;

    out.position = float4(ndcX, ndcY, 0.0, 1.0);
    out.color = v.colAlpha;
    out.radius = v.posRadius.w;
    out.pointSize = clamp(v.posRadius.w * 50.0, 2.0, 32.0);

    return out;
}

// MARK: - Fragment Shader (simple colored points)
fragment half4 splat_fragment(VSOut in [[stage_in]]) {
    return half4(in.color);
}

// MARK: - Gaussian Fragment Shader (soft circles)
fragment half4 splat_fragment_gaussian(VSOut in [[stage_in]],
                                       float2 pointCoord [[point_coord]]) {
    // pointCoord is [0,1] within the point sprite
    float2 centered = pointCoord * 2.0 - 1.0;  // [-1, 1]
    float r2 = dot(centered, centered);

    // Discard outside circle
    if (r2 > 1.0) {
        discard_fragment();
    }

    // Gaussian falloff: exp(-0.5 * r^2)
    float alpha = in.color.a * exp(-r2 * 2.0);

    // Premultiplied alpha
    half3 rgb = half3(in.color.rgb) * half(alpha);
    return half4(rgb, half(alpha));
}

// MARK: - Line Vertex/Fragment Shaders (for room wireframe)

struct LineVertex {
    float3 position [[attribute(0)]];
    float3 color    [[attribute(1)]];
};

struct LineVSOut {
    float4 position [[position]];
    float3 color;
};

vertex LineVSOut line_vertex(LineVertex v [[stage_in]],
                              constant CameraUniforms &camera [[buffer(1)]]) {
    LineVSOut out;
    float4 worldPos = float4(v.position, 1.0);
    float4 viewPos = camera.viewMatrix * worldPos;
    out.position = camera.projectionMatrix * viewPos;
    out.color = v.color;
    return out;
}

fragment half4 line_fragment(LineVSOut in [[stage_in]]) {
    return half4(half3(in.color), 1.0);
}
