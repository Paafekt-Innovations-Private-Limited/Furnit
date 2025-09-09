import Metal
import MetalKit
import CoreVideo
import CoreImage
import UIKit

class MetalRenderingPipeline {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let ciContext: CIContext
    
    // Compute pipelines
    private var backgroundRemovalPipeline: MTLComputePipelineState?
    private var maskRefinementPipeline: MTLComputePipelineState?
    
    // Render pipelines
    private var compositePipeline: MTLRenderPipelineState?
    
    // Texture cache for efficient CVPixelBuffer to MTLTexture conversion
    private var textureCache: CVMetalTextureCache?
    
    init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MetalError.deviceNotAvailable
        }
        
        self.device = device
        
        guard let commandQueue = device.makeCommandQueue() else {
            throw MetalError.commandQueueCreationFailed
        }
        
        self.commandQueue = commandQueue
        
        // Create CIContext with Metal device for efficient image processing
        self.ciContext = CIContext(mtlDevice: device, options: [.workingColorSpace: CGColorSpace(name: CGColorSpace.sRGB)!])
        
        // Create texture cache
        let result = CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, device, nil, &textureCache)
        guard result == kCVReturnSuccess else {
            throw MetalError.textureCacheCreationFailed
        }
        
        try setupPipelines()
    }
    
    private func setupPipelines() throws {
        guard let library = device.makeDefaultLibrary() else {
            throw MetalError.libraryNotFound
        }
        
        // Setup compute pipelines
        try setupComputePipelines(library: library)
        
        // Setup render pipelines
        try setupRenderPipelines(library: library)
    }
    
    private func setupComputePipelines(library: MTLLibrary) throws {
        // Background removal compute pipeline
        guard let backgroundRemovalFunction = library.makeFunction(name: "remove_background") else {
            throw MetalError.functionNotFound
        }
        
        backgroundRemovalPipeline = try device.makeComputePipelineState(function: backgroundRemovalFunction)
        
        // Mask refinement compute pipeline
        guard let maskRefinementFunction = library.makeFunction(name: "refine_mask") else {
            throw MetalError.functionNotFound
        }
        
        maskRefinementPipeline = try device.makeComputePipelineState(function: maskRefinementFunction)
    }
    
    private func setupRenderPipelines(library: MTLLibrary) throws {
        // Composite render pipeline
        let renderPipelineDescriptor = MTLRenderPipelineDescriptor()
        
        guard let vertexFunction = library.makeFunction(name: "vertex_main"),
              let fragmentFunction = library.makeFunction(name: "composite_fragment") else {
            throw MetalError.functionNotFound
        }
        
        renderPipelineDescriptor.vertexFunction = vertexFunction
        renderPipelineDescriptor.fragmentFunction = fragmentFunction
        
        // Set up vertex descriptor for stage_in attributes
        let vertexDescriptor = MTLVertexDescriptor()
        
        // Position attribute (index 0)
        vertexDescriptor.attributes[0].format = .float3
        vertexDescriptor.attributes[0].offset = 0
        vertexDescriptor.attributes[0].bufferIndex = 0
        
        // Texture coordinate attribute (index 1) 
        vertexDescriptor.attributes[1].format = .float2
        vertexDescriptor.attributes[1].offset = MemoryLayout<Float>.size * 3
        vertexDescriptor.attributes[1].bufferIndex = 0
        
        // Layout for buffer 0
        vertexDescriptor.layouts[0].stride = MemoryLayout<Float>.size * 5 // 3 position + 2 texCoord
        vertexDescriptor.layouts[0].stepRate = 1
        vertexDescriptor.layouts[0].stepFunction = .perVertex
        
        renderPipelineDescriptor.vertexDescriptor = vertexDescriptor
        
        // Configure for transparent rendering
        renderPipelineDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
        renderPipelineDescriptor.colorAttachments[0].isBlendingEnabled = true
        renderPipelineDescriptor.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
        renderPipelineDescriptor.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
        renderPipelineDescriptor.colorAttachments[0].sourceAlphaBlendFactor = .one
        renderPipelineDescriptor.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha
        
        compositePipeline = try device.makeRenderPipelineState(descriptor: renderPipelineDescriptor)
    }
    
    // MARK: - Public Processing Methods
    
    func processSegmentedObject(_ segmentedObject: SegmentedObject) -> MTLTexture? {
        // Convert CIImage to MTLTexture
        guard let inputTexture = createTexture(from: segmentedObject.originalImage) else {
            print("❌ Failed to create input texture")
            return nil
        }
        
        // Convert segmentation mask to MTLTexture
        guard let maskTexture = createTexture(from: segmentedObject.segmentationMask) else {
            print("❌ Failed to create mask texture")
            return nil
        }
        
        // Process the segmentation
        return removeBackground(inputTexture: inputTexture, maskTexture: maskTexture)
    }
    
    private func removeBackground(inputTexture: MTLTexture, maskTexture: MTLTexture) -> MTLTexture? {
        guard let pipeline = backgroundRemovalPipeline else { return nil }
        
        // Create output texture
        let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: inputTexture.pixelFormat,
            width: inputTexture.width,
            height: inputTexture.height,
            mipmapped: false
        )
        textureDescriptor.usage = [.shaderWrite, .shaderRead]
        
        guard let outputTexture = device.makeTexture(descriptor: textureDescriptor) else {
            return nil
        }
        
        // Create command buffer and encoder
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            return nil
        }
        
        // Setup compute command
        computeEncoder.setComputePipelineState(pipeline)
        computeEncoder.setTexture(inputTexture, index: 0)
        computeEncoder.setTexture(maskTexture, index: 1)
        computeEncoder.setTexture(outputTexture, index: 2)
        
        // Calculate thread groups
        let threadsPerThreadgroup = MTLSize(width: 8, height: 8, depth: 1)
        let threadgroupsPerGrid = MTLSize(
            width: (inputTexture.width + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width,
            height: (inputTexture.height + threadsPerThreadgroup.height - 1) / threadsPerThreadgroup.height,
            depth: 1
        )
        
        computeEncoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        computeEncoder.endEncoding()
        
        // Commit and wait
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        return outputTexture
    }
    
    // MARK: - Texture Creation Helpers
    
    private func createTexture(from ciImage: CIImage) -> MTLTexture? {
        let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm,
            width: Int(ciImage.extent.width),
            height: Int(ciImage.extent.height),
            mipmapped: false
        )
        textureDescriptor.usage = [.shaderRead, .shaderWrite, .renderTarget]
        
        guard let texture = device.makeTexture(descriptor: textureDescriptor) else {
            print("❌ Failed to create texture from CIImage")
            return nil
        }
        
        // Create command buffer for rendering
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            print("❌ Failed to create command buffer for CIImage rendering")
            return nil
        }
        
        // Render CIImage to Metal texture
        ciContext.render(ciImage, to: texture, commandBuffer: commandBuffer, bounds: ciImage.extent, colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!)
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        return texture
    }
    
    private func createTexture(from pixelBuffer: CVPixelBuffer) -> MTLTexture? {
        guard let textureCache = textureCache else { 
            print("❌ Metal texture cache not available")
            return nil 
        }
        
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        let pixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer)
        
        print("📊 CVPixelBuffer info: \(width)x\(height), format: \(pixelFormat)")
        
        // Convert common pixel format codes to readable names for debugging
        let formatName: String
        switch pixelFormat {
        case kCVPixelFormatType_OneComponent8: formatName = "OneComponent8"
        case kCVPixelFormatType_32BGRA: formatName = "32BGRA"  
        case kCVPixelFormatType_32ARGB: formatName = "32ARGB"
        case 1278226488: formatName = "Unknown1278226488"
        default: formatName = "Unknown(\(pixelFormat))"
        }
        print("📊 Pixel format name: \(formatName)")
        
        // Map CVPixelBuffer format to Metal format
        let metalPixelFormat: MTLPixelFormat
        switch pixelFormat {
        case kCVPixelFormatType_OneComponent8:
            metalPixelFormat = .r8Unorm
        case kCVPixelFormatType_32BGRA:
            metalPixelFormat = .bgra8Unorm
        case kCVPixelFormatType_32ARGB:
            metalPixelFormat = .rgba8Unorm
        default:
            print("⚠️ Unsupported pixel format \(pixelFormat), need to convert CVPixelBuffer")
            // Instead of failing, let's convert the CVPixelBuffer to a supported format
            return createTextureFromUnsupportedPixelBuffer(pixelBuffer)
        }
        
        var texture: CVMetalTexture?
        let result = CVMetalTextureCacheCreateTextureFromImage(
            kCFAllocatorDefault,
            textureCache,
            pixelBuffer,
            nil,
            metalPixelFormat,
            width,
            height,
            0,
            &texture
        )
        
        if result != kCVReturnSuccess {
            print("❌ CVMetalTextureCacheCreateTextureFromImage failed with result: \(result)")
            return nil
        }
        
        guard let cvTexture = texture else {
            print("❌ CVMetalTexture creation returned nil")
            return nil
        }
        
        let metalTexture = CVMetalTextureGetTexture(cvTexture)
        if metalTexture == nil {
            print("❌ Failed to get MTLTexture from CVMetalTexture")
        } else {
            print("✅ Successfully created Metal texture: \(width)x\(height)")
        }
        
        return metalTexture
    }
    
    private func createTextureFromUnsupportedPixelBuffer(_ pixelBuffer: CVPixelBuffer) -> MTLTexture? {
        print("🔄 Converting unsupported CVPixelBuffer format to Metal texture...")
        
        // Lock the pixel buffer for reading
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        
        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            print("❌ Failed to get CVPixelBuffer base address")
            return nil
        }
        
        // Create a new r8Unorm texture
        let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r8Unorm,
            width: width,
            height: height,
            mipmapped: false
        )
        textureDescriptor.usage = [.shaderRead, .shaderWrite]
        
        guard let texture = device.makeTexture(descriptor: textureDescriptor) else {
            print("❌ Failed to create Metal texture for CVPixelBuffer conversion")
            return nil
        }
        
        // Copy pixel data directly, converting to grayscale
        let buffer = baseAddress.assumingMemoryBound(to: UInt8.self)
        var grayscaleData = [UInt8](repeating: 0, count: width * height)
        
        // Simple conversion - take first component as grayscale (since it's a mask)
        for y in 0..<height {
            for x in 0..<width {
                let pixelIndex = y * bytesPerRow + x
                let outputIndex = y * width + x
                
                if pixelIndex < bytesPerRow * height && outputIndex < grayscaleData.count {
                    grayscaleData[outputIndex] = buffer[pixelIndex]
                }
            }
        }
        
        // Upload to Metal texture
        let region = MTLRegion(origin: MTLOrigin(x: 0, y: 0, z: 0),
                              size: MTLSize(width: width, height: height, depth: 1))
        
        texture.replace(region: region, mipmapLevel: 0, withBytes: grayscaleData, bytesPerRow: width)
        
        print("✅ Successfully converted unsupported CVPixelBuffer to Metal texture: \(texture.width)x\(texture.height)")
        return texture
    }
    
    func createTextureFromUIImage(_ image: UIImage) -> MTLTexture? {
        guard let cgImage = image.cgImage else { return nil }
        
        let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm,
            width: cgImage.width,
            height: cgImage.height,
            mipmapped: false
        )
        textureDescriptor.usage = [.shaderRead]
        
        guard let texture = device.makeTexture(descriptor: textureDescriptor) else {
            return nil
        }
        
        // Create CIImage and render to texture
        let ciImage = CIImage(cgImage: cgImage)
        ciContext.render(ciImage, to: texture, commandBuffer: nil, bounds: ciImage.extent, colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!)
        
        return texture
    }
}

// MARK: - Metal Errors
enum MetalError: LocalizedError {
    case deviceNotAvailable
    case commandQueueCreationFailed
    case textureCacheCreationFailed
    case libraryNotFound
    case functionNotFound
    
    var errorDescription: String? {
        switch self {
        case .deviceNotAvailable:
            return "Metal device not available"
        case .commandQueueCreationFailed:
            return "Failed to create Metal command queue"
        case .textureCacheCreationFailed:
            return "Failed to create texture cache"
        case .libraryNotFound:
            return "Metal shader library not found"
        case .functionNotFound:
            return "Metal shader function not found"
        }
    }
}