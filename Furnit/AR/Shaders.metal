#include <metal_stdlib>
using namespace metal;

// Vertex structure for rendering
struct Vertex {
    float3 position [[attribute(0)]];
    float2 texCoord [[attribute(1)]];
};

struct VertexOut {
    float4 position [[position]];
    float2 texCoord;
};

// Vertex shader for texture mapping
vertex VertexOut vertex_main(Vertex in [[stage_in]]) {
    VertexOut out;
    out.position = float4(in.position, 1.0);
    out.texCoord = in.texCoord;
    return out;
}

// Fragment shader to apply segmentation mask
fragment float4 segmentation_fragment(VertexOut in [[stage_in]],
                                    texture2d<float> colorTexture [[texture(0)]],
                                    texture2d<float> maskTexture [[texture(1)]]) {
    constexpr sampler textureSampler(mag_filter::linear, min_filter::linear);
    
    // Sample the original color and mask
    float4 color = colorTexture.sample(textureSampler, in.texCoord);
    float mask = maskTexture.sample(textureSampler, in.texCoord).r;
    
    // Apply mask - only show pixels where mask is white (1.0)
    color.a = mask;
    
    return color;
}

// Compute shader for efficient background removal
kernel void remove_background(texture2d<float, access::read> inputTexture [[texture(0)]],
                             texture2d<float, access::read> maskTexture [[texture(1)]],
                             texture2d<float, access::write> outputTexture [[texture(2)]],
                             uint2 gid [[thread_position_in_grid]]) {
    
    if (gid.x >= outputTexture.get_width() || gid.y >= outputTexture.get_height()) {
        return;
    }
    
    // Read input color and mask value
    float4 inputColor = inputTexture.read(gid);
    float maskValue = maskTexture.read(gid).r;
    
    // Apply segmentation: keep pixel if mask is white, make transparent otherwise
    float4 outputColor = inputColor;
    outputColor.a = maskValue;
    
    // Write the result
    outputTexture.write(outputColor, gid);
}

// Compute shader for mask refinement (edge smoothing)
kernel void refine_mask(texture2d<float, access::read> inputMask [[texture(0)]],
                       texture2d<float, access::write> outputMask [[texture(1)]],
                       uint2 gid [[thread_position_in_grid]]) {
    
    if (gid.x >= outputMask.get_width() || gid.y >= outputMask.get_height()) {
        return;
    }
    
    // Simple 3x3 gaussian blur for smoother edges
    float sum = 0.0;
    float weights[9] = {1, 2, 1, 2, 4, 2, 1, 2, 1};
    float totalWeight = 16.0;
    
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int2 coord = int2(gid) + int2(dx, dy);
            if (coord.x >= 0 && coord.x < int(inputMask.get_width()) &&
                coord.y >= 0 && coord.y < int(inputMask.get_height())) {
                
                float maskValue = inputMask.read(uint2(coord)).r;
                int index = (dy + 1) * 3 + (dx + 1);
                sum += maskValue * weights[index];
            }
        }
    }
    
    float refinedMask = sum / totalWeight;
    outputMask.write(float4(refinedMask, refinedMask, refinedMask, 1.0), gid);
}

// Fragment shader for compositing segmented object in 3D scene
fragment float4 composite_fragment(VertexOut in [[stage_in]],
                                  texture2d<float> segmentedTexture [[texture(0)]],
                                  constant float &opacity [[buffer(0)]]) {
    constexpr sampler textureSampler(mag_filter::linear, min_filter::linear);
    
    float4 color = segmentedTexture.sample(textureSampler, in.texCoord);
    
    // Apply overall opacity control
    color.a *= opacity;
    
    return color;
}