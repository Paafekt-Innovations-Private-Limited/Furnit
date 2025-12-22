#include <metal_stdlib>
using namespace metal;

// Kernel function to refine segmentation mask using morphological operations
kernel void refineSegmentationMask(texture2d<float, access::read> inputTexture [[texture(0)]],
                                  texture2d<float, access::write> outputTexture [[texture(1)]],
                                  uint2 gid [[thread_position_in_grid]]) {
    
    // Check bounds
    if (gid.x >= inputTexture.get_width() || gid.y >= inputTexture.get_height()) {
        return;
    }
    
    // Morphological closing operation (dilation followed by erosion)
    // This helps fill small gaps and smooth the segmentation boundaries
    
    const int kernelSize = 3;
    const int halfKernel = kernelSize / 2;
    
    // Dilation pass - find maximum value in neighborhood
    float maxValue = 0.0;
    for (int dy = -halfKernel; dy <= halfKernel; dy++) {
        for (int dx = -halfKernel; dx <= halfKernel; dx++) {
            int2 sampleCoord = int2(gid) + int2(dx, dy);
            
            // Check bounds
            if (sampleCoord.x >= 0 && sampleCoord.x < int(inputTexture.get_width()) &&
                sampleCoord.y >= 0 && sampleCoord.y < int(inputTexture.get_height())) {
                
                float4 sampleColor = inputTexture.read(uint2(sampleCoord));
                maxValue = max(maxValue, sampleColor.r);
            }
        }
    }
    
    // Apply Gaussian blur for smooth edges
    float blurredValue = 0.0;
    float totalWeight = 0.0;
    
    for (int dy = -halfKernel; dy <= halfKernel; dy++) {
        for (int dx = -halfKernel; dx <= halfKernel; dx++) {
            int2 sampleCoord = int2(gid) + int2(dx, dy);
            
            // Check bounds
            if (sampleCoord.x >= 0 && sampleCoord.x < int(inputTexture.get_width()) &&
                sampleCoord.y >= 0 && sampleCoord.y < int(inputTexture.get_height())) {
                
                // Gaussian weight (approximation)
                float distance = sqrt(float(dx * dx + dy * dy));
                float weight = exp(-distance * distance / 2.0);
                
                float4 sampleColor = inputTexture.read(uint2(sampleCoord));
                blurredValue += sampleColor.r * weight;
                totalWeight += weight;
            }
        }
    }
    
    if (totalWeight > 0.0) {
        blurredValue /= totalWeight;
    }
    
    // Combine dilation and blur for final result
    float finalValue = mix(blurredValue, maxValue, 0.3);
    
    // Apply threshold to create clean binary mask
    finalValue = finalValue > 0.5 ? 1.0 : 0.0;
    
    // Write result
    outputTexture.write(float4(finalValue, finalValue, finalValue, 1.0), gid);
}

// Kernel function to apply alpha blending with segmentation mask
kernel void applyAlphaBlending(texture2d<float, access::read> originalTexture [[texture(0)]],
                              texture2d<float, access::read> maskTexture [[texture(1)]],
                              texture2d<float, access::write> outputTexture [[texture(2)]],
                              uint2 gid [[thread_position_in_grid]]) {
    
    // Check bounds
    if (gid.x >= originalTexture.get_width() || gid.y >= originalTexture.get_height()) {
        return;
    }
    
    // Read original color and mask value
    float4 originalColor = originalTexture.read(gid);
    float4 maskColor = maskTexture.read(gid);
    
    // Use mask as alpha channel
    float alpha = maskColor.r;
    
    // Apply anti-aliasing to mask edges
    // Create smooth falloff at the edges
    const int falloffRadius = 2;
    float falloffAlpha = alpha;
    
    if (alpha > 0.0 && alpha < 1.0) {
        // We're at an edge, apply smooth falloff
        float edgeFactor = 0.0;
        float totalSamples = 0.0;
        
        for (int dy = -falloffRadius; dy <= falloffRadius; dy++) {
            for (int dx = -falloffRadius; dx <= falloffRadius; dx++) {
                int2 sampleCoord = int2(gid) + int2(dx, dy);
                
                if (sampleCoord.x >= 0 && sampleCoord.x < int(maskTexture.get_width()) &&
                    sampleCoord.y >= 0 && sampleCoord.y < int(maskTexture.get_height())) {
                    
                    float4 sampleMask = maskTexture.read(uint2(sampleCoord));
                    float distance = sqrt(float(dx * dx + dy * dy));
                    float weight = exp(-distance / float(falloffRadius));
                    
                    edgeFactor += sampleMask.r * weight;
                    totalSamples += weight;
                }
            }
        }
        
        if (totalSamples > 0.0) {
            falloffAlpha = edgeFactor / totalSamples;
        }
    }
    
    // Create output color with alpha channel
    float4 outputColor = float4(originalColor.rgb, falloffAlpha);
    
    // Apply additional edge enhancement for cleaner segmentation
    if (falloffAlpha > 0.1) {
        // Enhance contrast slightly for better object definition
        outputColor.rgb = pow(outputColor.rgb, 0.9);
        
        // Boost saturation slightly for more vibrant furniture colors
        float luminance = dot(outputColor.rgb, float3(0.299, 0.587, 0.114));
        outputColor.rgb = mix(float3(luminance), outputColor.rgb, 1.1);
    }
    
    // Write final result
    outputTexture.write(outputColor, gid);
}

// Additional utility kernel for bilateral filtering (edge-preserving smoothing)
kernel void bilateralFilter(texture2d<float, access::read> inputTexture [[texture(0)]],
                           texture2d<float, access::write> outputTexture [[texture(1)]],
                           uint2 gid [[thread_position_in_grid]]) {
    
    // Check bounds
    if (gid.x >= inputTexture.get_width() || gid.y >= inputTexture.get_height()) {
        return;
    }
    
    const int radius = 3;
    const float sigmaSpace = 2.0;
    const float sigmaColor = 0.1;
    
    float4 centerColor = inputTexture.read(gid);
    float4 filteredColor = float4(0.0);
    float totalWeight = 0.0;
    
    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            int2 sampleCoord = int2(gid) + int2(dx, dy);
            
            // Check bounds
            if (sampleCoord.x >= 0 && sampleCoord.x < int(inputTexture.get_width()) &&
                sampleCoord.y >= 0 && sampleCoord.y < int(inputTexture.get_height())) {
                
                float4 sampleColor = inputTexture.read(uint2(sampleCoord));
                
                // Spatial weight (based on distance)
                float spatialDistance = sqrt(float(dx * dx + dy * dy));
                float spatialWeight = exp(-spatialDistance * spatialDistance / (2.0 * sigmaSpace * sigmaSpace));
                
                // Color weight (based on color similarity)
                float colorDistance = length(sampleColor.rgb - centerColor.rgb);
                float colorWeight = exp(-colorDistance * colorDistance / (2.0 * sigmaColor * sigmaColor));
                
                float weight = spatialWeight * colorWeight;
                filteredColor += sampleColor * weight;
                totalWeight += weight;
            }
        }
    }
    
    if (totalWeight > 0.0) {
        filteredColor /= totalWeight;
    } else {
        filteredColor = centerColor;
    }
    
    outputTexture.write(filteredColor, gid);
}