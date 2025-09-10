import Metal
import MetalKit
import UIKit
import CoreGraphics

class MetalRenderingPipeline {
    // Metal device and command queue
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let library: MTLLibrary
    
    // Render pipeline states
    private var segmentationRefinementPipeline: MTLComputePipelineState?
    private var alphaBlendingPipeline: MTLComputePipelineState?
    
    // Texture cache for efficient texture creation
    private var textureCache: CVMetalTextureCache?
    
    init?() {
        // Get the default Metal device
        guard let device = MTLCreateSystemDefaultDevice() else {
            print("❌ Failed to create Metal device")
            return nil
        }
        
        self.device = device
        
        // Create command queue
        guard let commandQueue = device.makeCommandQueue() else {
            print("❌ Failed to create Metal command queue")
            return nil
        }
        self.commandQueue = commandQueue
        
        // Load shader library
        guard let library = device.makeDefaultLibrary() else {
            print("❌ Failed to create Metal library")
            return nil
        }
        self.library = library
        
        // Create texture cache
        let result = CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, device, nil, &textureCache)
        guard result == kCVReturnSuccess, textureCache != nil else {
            print("❌ Failed to create Metal texture cache")
            return nil
        }
        
        setupComputePipelines()
    }
    
    // Set up compute pipeline states for image processing
    private func setupComputePipelines() {
        do {
            // Create segmentation refinement pipeline
            guard let segmentationFunction = library.makeFunction(name: "refineSegmentationMask") else {
                print("❌ Failed to find refineSegmentationMask function")
                return
            }
            
            segmentationRefinementPipeline = try device.makeComputePipelineState(function: segmentationFunction)
            
            // Create alpha blending pipeline
            guard let blendingFunction = library.makeFunction(name: "applyAlphaBlending") else {
                print("❌ Failed to find applyAlphaBlending function")
                return
            }
            
            alphaBlendingPipeline = try device.makeComputePipelineState(function: blendingFunction)
            
            print("✅ Metal compute pipelines created successfully")
            
        } catch {
            print("❌ Failed to create compute pipelines: \(error)")
        }
    }
    
    // Refine segmentation mask using Metal shaders for better quality
    func refineSegmentationMask(_ inputTexture: MTLTexture) -> MTLTexture? {
        guard let pipeline = segmentationRefinementPipeline else {
            print("❌ Segmentation refinement pipeline not available")
            return nil
        }
        
        // Create output texture
        let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: inputTexture.pixelFormat,
            width: inputTexture.width,
            height: inputTexture.height,
            mipmapped: false
        )
        textureDescriptor.usage = [.shaderRead, .shaderWrite]
        
        guard let outputTexture = device.makeTexture(descriptor: textureDescriptor) else {
            print("❌ Failed to create output texture")
            return nil
        }
        
        // Create command buffer and encoder
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            print("❌ Failed to create command buffer or encoder")
            return nil
        }
        
        // Set up compute pass
        encoder.setComputePipelineState(pipeline)
        encoder.setTexture(inputTexture, index: 0)
        encoder.setTexture(outputTexture, index: 1)
        
        // Calculate thread groups and threads per group
        let threadsPerGroup = MTLSize(width: 16, height: 16, depth: 1)
        let threadGroups = MTLSize(
            width: (inputTexture.width + threadsPerGroup.width - 1) / threadsPerGroup.width,
            height: (inputTexture.height + threadsPerGroup.height - 1) / threadsPerGroup.height,
            depth: 1
        )
        
        encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()
        
        // Commit and wait for completion
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        return outputTexture
    }
    
    // Apply alpha blending with segmentation mask
    func applyAlphaBlending(originalTexture: MTLTexture, maskTexture: MTLTexture) -> MTLTexture? {
        guard let pipeline = alphaBlendingPipeline else {
            print("❌ Alpha blending pipeline not available")
            return nil
        }
        
        // Create output texture with RGBA format
        let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba8Unorm,
            width: originalTexture.width,
            height: originalTexture.height,
            mipmapped: false
        )
        textureDescriptor.usage = [.shaderRead, .shaderWrite]
        
        guard let outputTexture = device.makeTexture(descriptor: textureDescriptor) else {
            print("❌ Failed to create output texture")
            return nil
        }
        
        // Create command buffer and encoder
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            print("❌ Failed to create command buffer or encoder")
            return nil
        }
        
        // Set up compute pass
        encoder.setComputePipelineState(pipeline)
        encoder.setTexture(originalTexture, index: 0)
        encoder.setTexture(maskTexture, index: 1)
        encoder.setTexture(outputTexture, index: 2)
        
        // Calculate thread groups and threads per group
        let threadsPerGroup = MTLSize(width: 16, height: 16, depth: 1)
        let threadGroups = MTLSize(
            width: (originalTexture.width + threadsPerGroup.width - 1) / threadsPerGroup.width,
            height: (originalTexture.height + threadsPerGroup.height - 1) / threadsPerGroup.height,
            depth: 1
        )
        
        encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()
        
        // Commit and wait for completion
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        return outputTexture
    }
    
    // Convert UIImage to Metal texture
    func createTexture(from image: UIImage) -> MTLTexture? {
        guard let cgImage = image.cgImage else {
            print("❌ Failed to get CGImage from UIImage")
            return nil
        }
        
        let width = cgImage.width
        let height = cgImage.height
        
        // Create texture descriptor
        let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba8Unorm,
            width: width,
            height: height,
            mipmapped: false
        )
        textureDescriptor.usage = [.shaderRead, .shaderWrite]
        
        guard let texture = device.makeTexture(descriptor: textureDescriptor) else {
            print("❌ Failed to create Metal texture")
            return nil
        }
        
        // Create bitmap context
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bytesPerRow = width * 4
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
        
        guard let context = CGContext(data: nil,
                                     width: width,
                                     height: height,
                                     bitsPerComponent: 8,
                                     bytesPerRow: bytesPerRow,
                                     space: colorSpace,
                                     bitmapInfo: bitmapInfo.rawValue) else {
            print("❌ Failed to create bitmap context")
            return nil
        }
        
        // Draw image into context
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        guard let pixelData = context.data else {
            print("❌ Failed to get pixel data from context")
            return nil
        }
        
        // Copy pixel data to texture
        let region = MTLRegionMake2D(0, 0, width, height)
        texture.replace(region: region, mipmapLevel: 0, withBytes: pixelData, bytesPerRow: bytesPerRow)
        
        return texture
    }
    
    // Convert Metal texture to UIImage
    func createImage(from texture: MTLTexture) -> UIImage? {
        let width = texture.width
        let height = texture.height
        let bytesPerRow = width * 4
        let imageSize = height * bytesPerRow
        
        // Allocate buffer for pixel data
        let pixelBytes = UnsafeMutableRawPointer.allocate(byteCount: imageSize, alignment: 1)
        defer { pixelBytes.deallocate() }
        
        // Copy texture data to buffer
        let region = MTLRegionMake2D(0, 0, width, height)
        texture.getBytes(pixelBytes, bytesPerRow: bytesPerRow, from: region, mipmapLevel: 0)
        
        // Create bitmap context
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
        
        guard let context = CGContext(data: pixelBytes,
                                     width: width,
                                     height: height,
                                     bitsPerComponent: 8,
                                     bytesPerRow: bytesPerRow,
                                     space: colorSpace,
                                     bitmapInfo: bitmapInfo.rawValue),
              let cgImage = context.makeImage() else {
            print("❌ Failed to create CGImage from texture")
            return nil
        }
        
        return UIImage(cgImage: cgImage)
    }
    
    // Process segmented image with Metal for better performance and quality
    func processSegmentedImage(_ originalImage: UIImage, maskImage: UIImage) -> UIImage? {
        // Convert images to Metal textures
        guard let originalTexture = createTexture(from: originalImage),
              let maskTexture = createTexture(from: maskImage) else {
            print("❌ Failed to create input textures")
            return nil
        }
        
        // Refine the segmentation mask
        guard let refinedMask = refineSegmentationMask(maskTexture) else {
            print("❌ Failed to refine segmentation mask")
            return nil
        }
        
        // Apply alpha blending
        guard let blendedTexture = applyAlphaBlending(originalTexture: originalTexture, 
                                                     maskTexture: refinedMask) else {
            print("❌ Failed to apply alpha blending")
            return nil
        }
        
        // Convert back to UIImage
        return createImage(from: blendedTexture)
    }
}