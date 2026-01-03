import Foundation
import UIKit
import CoreML
import Metal
import MetalKit
import MetalPerformanceShaders
import Accelerate

/// On-device 3D Gaussian generation using Apple's SHARP model
/// Replaces remote Jarvis-based Room3DGenerationService with local CoreML inference
@MainActor
class SHARPService: ObservableObject {

    // MARK: - Published Properties

    /// Current generation status
    @Published var status: GenerationStatus = .idle

    /// Human-readable status message
    @Published var statusMessage: String = "Ready"

    /// Processing progress (0.0 to 1.0)
    @Published var progress: Float = 0.0

    // MARK: - Model Configuration

    /// Input image size expected by SHARP model
    private static let inputSize: Int = 1536

    /// Gaussian parameter count per splat: pos(3) + scale(3) + rot(4) + opacity(1) + sh(3) = 14
    private static let paramsPerGaussian: Int = 14

    // MARK: - CoreML Model

    /// The compiled SHARP CoreML model
    private var model: MLModel?

    /// Metal device for GPU acceleration
    private let metalDevice: MTLDevice?

    /// Command queue for Metal operations
    private let commandQueue: MTLCommandQueue?

    /// Metal Performance Shaders image scaler
    private let imageScaler: MPSImageBilinearScale?

    // MARK: - File Management

    /// Directory for saving generated PLY files
    private var modelsDirectory: URL {
        let documentsDirectory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        return documentsDirectory.appendingPathComponent("SavedRooms", isDirectory: true)
    }

    // MARK: - Initialization

    init() {
        // Initialize Metal
        self.metalDevice = MTLCreateSystemDefaultDevice()
        self.commandQueue = metalDevice?.makeCommandQueue()

        if let device = metalDevice {
            self.imageScaler = MPSImageBilinearScale(device: device)
            logDebug("SHARP: Metal initialized with device: \(device.name)")
        } else {
            self.imageScaler = nil
            logDebug("SHARP: Metal not available, using CPU fallback")
        }

        // Load CoreML model
        Task {
            await loadModel()
        }
    }

    // MARK: - Model Loading

    /// Load the SHARP CoreML model
    private func loadModel() async {
        logDebug("SHARP: Loading CoreML model...")

        // Check physical memory - INT8 model (~663MB) needs ~2GB available
        let physicalMemory = ProcessInfo.processInfo.physicalMemory
        let memoryLimit: UInt64 = 2 * 1024 * 1024 * 1024  // 2GB minimum

        logDebug("SHARP: Device RAM: \(physicalMemory / 1024 / 1024)MB")

        if physicalMemory < memoryLimit {
            logDebug("SHARP: Device has insufficient RAM, need 2GB+")
            logDebug("SHARP: Model will not be loaded - use fallback or remote API")
            return
        }

        do {
            // Configure for CPU-only to avoid GPU memory contention with Metal preprocessing
            let config = MLModelConfiguration()
            config.computeUnits = .cpuOnly  // CPU-only to avoid GPU/ANE memory issues

            // Use the Xcode auto-generated SHARP model class with async loading
            logDebug("SHARP: Loading via auto-generated model class (async)...")
            let sharpModel = try await SHARP.load(configuration: config)
            model = sharpModel.model
            logDebug("SHARP: Model loaded successfully")

        } catch {
            logDebug("SHARP: Failed to load model: \(error)")
        }
    }

    // MARK: - Main Generation API

    /// Generate 3D Gaussian splat from an image
    /// - Parameter image: The source image
    /// - Returns: URL to the saved PLY file
    func generateGaussians(from image: UIImage) async throws -> URL {
        logDebug("SHARP: Starting Gaussian generation")

        // Load model on-demand if not already loaded
        if model == nil {
            logDebug("SHARP: Model not loaded yet, loading now...")
            await loadModel()
        }

        guard model != nil else {
            throw GenerationError.serverError("SHARP model failed to load. Check device memory.")
        }

        // Reset state
        status = .processing
        statusMessage = "Preprocessing image..."
        progress = 0.0

        // Step 1: Preprocess image
        progress = 0.1
        let inputArray = try await preprocessImage(image)
        logDebug("SHARP: Image preprocessed to \(Self.inputSize)x\(Self.inputSize)")

        // Step 2: Run inference
        statusMessage = "Generating 3D Gaussians..."
        progress = 0.2
        let gaussianParams = try await runInference(inputArray)
        logDebug("SHARP: Generated \(gaussianParams.count / Self.paramsPerGaussian) Gaussians")

        // Step 3: Write PLY file
        statusMessage = "Saving model..."
        progress = 0.8
        let plyURL = try await writePLY(gaussianParams)
        logDebug("SHARP: Saved PLY to \(plyURL.path)")

        // Complete
        progress = 1.0
        status = .completed(fileURL: plyURL)
        statusMessage = "Complete!"

        return plyURL
    }

    /// Cancel current generation (if any)
    func cancelGeneration() {
        status = .failed("Cancelled")
        statusMessage = "Cancelled"
    }

    // MARK: - Image Preprocessing

    /// Preprocess image for SHARP model input
    /// - Parameter image: Source UIImage
    /// - Returns: MLMultiArray with shape [1, 3, 1536, 1536]
    private func preprocessImage(_ image: UIImage) async throws -> MLMultiArray {
        guard let cgImage = image.cgImage else {
            throw GenerationError.invalidImage
        }

        // Use Metal for fast image scaling if available
        if let device = metalDevice, let commandQueue = commandQueue, let scaler = imageScaler {
            return try await preprocessWithMetal(cgImage, device: device, commandQueue: commandQueue, scaler: scaler)
        } else {
            return try preprocessWithCPU(cgImage)
        }
    }

    /// Metal-accelerated image preprocessing
    private func preprocessWithMetal(
        _ cgImage: CGImage,
        device: MTLDevice,
        commandQueue: MTLCommandQueue,
        scaler: MPSImageBilinearScale
    ) async throws -> MLMultiArray {
        let size = Self.inputSize

        // Create source texture from CGImage
        let textureLoader = MTKTextureLoader(device: device)
        let sourceTexture = try await textureLoader.newTexture(cgImage: cgImage, options: [
            .textureUsage: MTLTextureUsage.shaderRead.rawValue,
            .textureStorageMode: MTLStorageMode.shared.rawValue
        ])

        // Create destination texture
        let destDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba8Unorm,
            width: size,
            height: size,
            mipmapped: false
        )
        destDescriptor.usage = [.shaderRead, .shaderWrite]
        destDescriptor.storageMode = .shared

        guard let destTexture = device.makeTexture(descriptor: destDescriptor) else {
            throw GenerationError.serverError("Failed to create Metal texture")
        }

        // Scale image using MPS
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw GenerationError.serverError("Failed to create command buffer")
        }

        scaler.encode(commandBuffer: commandBuffer, sourceTexture: sourceTexture, destinationTexture: destTexture)
        commandBuffer.commit()
        await commandBuffer.completed()

        // Read back pixel data
        var pixelData = [UInt8](repeating: 0, count: size * size * 4)
        destTexture.getBytes(
            &pixelData,
            bytesPerRow: size * 4,
            from: MTLRegion(origin: MTLOrigin(x: 0, y: 0, z: 0), size: MTLSize(width: size, height: size, depth: 1)),
            mipmapLevel: 0
        )

        // Convert to MLMultiArray [1, 3, H, W] with float values in [0, 1]
        return try createMLMultiArray(from: pixelData, size: size)
    }

    /// CPU fallback for image preprocessing using Accelerate
    private func preprocessWithCPU(_ cgImage: CGImage) throws -> MLMultiArray {
        let size = Self.inputSize

        // Create bitmap context for resizing
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        var pixelData = [UInt8](repeating: 0, count: size * size * 4)

        guard let context = CGContext(
            data: &pixelData,
            width: size,
            height: size,
            bitsPerComponent: 8,
            bytesPerRow: size * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            throw GenerationError.invalidImage
        }

        // Draw resized image
        context.interpolationQuality = .high
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: size, height: size))

        return try createMLMultiArray(from: pixelData, size: size)
    }

    /// Convert pixel data to MLMultiArray using vDSP for fast memory operations
    private func createMLMultiArray(from pixelData: [UInt8], size: Int) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: [1, 3, NSNumber(value: size), NSNumber(value: size)], dataType: .float32)
        let pixelCount = size * size
        let ptr = array.dataPointer.bindMemory(to: Float.self, capacity: 3 * pixelCount)

        // Process one channel at a time to minimize memory usage
        // Write directly to MLMultiArray without intermediate arrays
        pixelData.withUnsafeBufferPointer { srcBuffer in
            let src = srcBuffer.baseAddress!

            // Red channel (stride 4 from offset 0)
            for i in 0..<pixelCount {
                ptr[i] = Float(src[i * 4]) / 255.0
            }

            // Green channel (stride 4 from offset 1)
            let gPtr = ptr.advanced(by: pixelCount)
            for i in 0..<pixelCount {
                gPtr[i] = Float(src[i * 4 + 1]) / 255.0
            }

            // Blue channel (stride 4 from offset 2)
            let bPtr = ptr.advanced(by: 2 * pixelCount)
            for i in 0..<pixelCount {
                bPtr[i] = Float(src[i * 4 + 2]) / 255.0
            }
        }

        return array
    }

    // MARK: - Model Inference

    /// Run SHARP model inference
    private func runInference(_ input: MLMultiArray) async throws -> [Float] {
        guard let model = model else {
            throw GenerationError.serverError("Model not loaded")
        }

        logDebug("SHARP: Running inference...")

        // Create feature provider
        let inputFeatures = try MLDictionaryFeatureProvider(dictionary: ["image": input])

        // Run prediction
        let output = try await model.prediction(from: inputFeatures)

        logDebug("SHARP: Inference complete, extracting output...")

        // Extract Gaussian parameters
        guard let gaussians = output.featureValue(for: "gaussians")?.multiArrayValue else {
            throw GenerationError.serverError("Invalid model output")
        }

        // Model outputs Float16 - convert to Float32 for PLY writing
        let count = gaussians.count
        var result = [Float](repeating: 0, count: count)

        // Check data type and convert accordingly
        if gaussians.dataType == .float16 {
            let srcPtr = gaussians.dataPointer.bindMemory(to: Float16.self, capacity: count)
            for i in 0..<count {
                result[i] = Float(srcPtr[i])
            }
        } else {
            // Float32 fallback
            let srcPtr = gaussians.dataPointer.bindMemory(to: Float.self, capacity: count)
            result.withUnsafeMutableBufferPointer { destPtr in
                vDSP_mmov(srcPtr, destPtr.baseAddress!, vDSP_Length(count), 1, 1, 1)
            }
        }

        logDebug("SHARP: Extracted \(count) values")
        return result
    }

    // MARK: - Splat Filtering

    /// Maximum number of splats for mobile rendering (higher = better quality, more memory)
    private static let maxSplats: Int = 500_000

    /// Minimum opacity threshold (0-1) for keeping a splat
    private static let minOpacity: Float = 0.01

    /// Filter Gaussians by opacity and limit count for mobile rendering
    private func filterGaussians(_ params: [Float]) -> [Float] {
        let inputCount = params.count / Self.paramsPerGaussian
        logDebug("SHARP: Filtering \(inputCount) Gaussians...")

        // Debug: sample first few gaussians to see value ranges
        if inputCount > 0 {
            let idx = 0
            let o = idx * Self.paramsPerGaussian
            logDebug("SHARP DEBUG: Sample Gaussian 0:")
            logDebug("  pos: (\(params[o+0]), \(params[o+1]), \(params[o+2]))")
            logDebug("  scale: (\(params[o+3]), \(params[o+4]), \(params[o+5]))")
            logDebug("  rot: (\(params[o+6]), \(params[o+7]), \(params[o+8]), \(params[o+9]))")
            logDebug("  opacity: \(params[o+10])")
            logDebug("  color: (\(params[o+11]), \(params[o+12]), \(params[o+13]))")
        }
        if inputCount > 1000 {
            let idx = 1000
            let o = idx * Self.paramsPerGaussian
            logDebug("SHARP DEBUG: Sample Gaussian 1000:")
            logDebug("  pos: (\(params[o+0]), \(params[o+1]), \(params[o+2]))")
            logDebug("  scale: (\(params[o+3]), \(params[o+4]), \(params[o+5]))")
            logDebug("  opacity: \(params[o+10])")
            logDebug("  color: (\(params[o+11]), \(params[o+12]), \(params[o+13]))")
        }

        // First pass: collect indices of splats above opacity threshold with their opacities
        var validSplats: [(index: Int, opacity: Float)] = []
        validSplats.reserveCapacity(inputCount / 4)  // Estimate ~25% will pass

        for i in 0..<inputCount {
            let offset = i * Self.paramsPerGaussian
            let opacity = params[offset + 10]  // opacity is at index 10

            if opacity >= Self.minOpacity {
                validSplats.append((index: i, opacity: opacity))
            }
        }

        logDebug("SHARP: \(validSplats.count) splats above opacity threshold \(Self.minOpacity)")

        // If still too many, sort by opacity and keep top N
        var selectedIndices: [Int]
        if validSplats.count > Self.maxSplats {
            // Sort by opacity descending and take top maxSplats
            validSplats.sort { $0.opacity > $1.opacity }
            selectedIndices = validSplats.prefix(Self.maxSplats).map { $0.index }
            logDebug("SHARP: Limited to \(Self.maxSplats) highest-opacity splats")
        } else {
            selectedIndices = validSplats.map { $0.index }
        }

        // Build filtered output
        var filtered = [Float]()
        filtered.reserveCapacity(selectedIndices.count * Self.paramsPerGaussian)

        for idx in selectedIndices {
            let offset = idx * Self.paramsPerGaussian
            for j in 0..<Self.paramsPerGaussian {
                filtered.append(params[offset + j])
            }
        }

        logDebug("SHARP: Output \(filtered.count / Self.paramsPerGaussian) filtered Gaussians")
        return filtered
    }

    // MARK: - PLY Writing

    /// Write Gaussian parameters to PLY file
    private func writePLY(_ params: [Float]) async throws -> URL {
        // Filter Gaussians for mobile rendering
        let filteredParams = filterGaussians(params)
        let gaussianCount = filteredParams.count / Self.paramsPerGaussian

        // Ensure output directory exists
        try FileManager.default.createDirectory(at: modelsDirectory, withIntermediateDirectories: true)

        // Generate filename
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "yyyyMMdd_HHmmss"
        let timestamp = dateFormatter.string(from: Date())
        let fileName = "Room_\(timestamp).ply"
        let fileURL = modelsDirectory.appendingPathComponent(fileName)

        // Build PLY content
        let plyContent = """
        ply
        format binary_little_endian 1.0
        element vertex \(gaussianCount)
        property float x
        property float y
        property float z
        property float scale_0
        property float scale_1
        property float scale_2
        property float rot_0
        property float rot_1
        property float rot_2
        property float rot_3
        property float opacity
        property float f_dc_0
        property float f_dc_1
        property float f_dc_2
        end_header\n
        """

        // Write header
        var data = Data(plyContent.utf8)

        // Write binary vertex data
        // Each Gaussian: pos(3) + scale(3) + rot(4) + opacity(1) + sh(3) = 14 floats
        // Using raw SHARP output values directly
        for i in 0..<gaussianCount {
            let offset = i * Self.paramsPerGaussian

            // Position - swap Y and Z for coordinate system conversion (SHARP uses different axes)
            var x = filteredParams[offset + 0]
            var y = filteredParams[offset + 2]  // Z becomes Y
            var z = -filteredParams[offset + 1] // -Y becomes Z
            data.append(Data(bytes: &x, count: 4))
            data.append(Data(bytes: &y, count: 4))
            data.append(Data(bytes: &z, count: 4))

            // Scale - convert to log for renderer, clamp minimum to prevent needle-thin splats
            let minScale: Float = 0.001
            let rawS0 = filteredParams[offset + 3]
            let rawS1 = filteredParams[offset + 4]
            let rawS2 = filteredParams[offset + 5]
            var s0 = log(max(rawS0, minScale))
            var s1 = log(max(rawS1, minScale))
            var s2 = log(max(rawS2, minScale))
            data.append(Data(bytes: &s0, count: 4))
            data.append(Data(bytes: &s1, count: 4))
            data.append(Data(bytes: &s2, count: 4))

            // Rotation quaternion - MUST normalize (SHARP outputs unnormalized)
            let rawR0 = filteredParams[offset + 6]
            let rawR1 = filteredParams[offset + 7]
            let rawR2 = filteredParams[offset + 8]
            let rawR3 = filteredParams[offset + 9]
            let mag = sqrt(rawR0*rawR0 + rawR1*rawR1 + rawR2*rawR2 + rawR3*rawR3)
            let invMag = mag > 1e-8 ? 1.0 / mag : 1.0
            var r0 = rawR0 * invMag
            var r1 = rawR1 * invMag
            var r2 = rawR2 * invMag
            var r3 = rawR3 * invMag
            data.append(Data(bytes: &r0, count: 4))
            data.append(Data(bytes: &r1, count: 4))
            data.append(Data(bytes: &r2, count: 4))
            data.append(Data(bytes: &r3, count: 4))

            // Opacity - convert to logit (inverse sigmoid) for renderer
            let rawOpacity = filteredParams[offset + 10]
            let clampedOpacity = min(max(rawOpacity, 1e-4), 1.0 - 1e-4)
            var opacity = log(clampedOpacity / (1.0 - clampedOpacity))
            data.append(Data(bytes: &opacity, count: 4))

            // Color - use raw RGB values (MetalSplatter may handle conversion)
            var sh0 = filteredParams[offset + 11]
            var sh1 = filteredParams[offset + 12]
            var sh2 = filteredParams[offset + 13]
            data.append(Data(bytes: &sh0, count: 4))
            data.append(Data(bytes: &sh1, count: 4))
            data.append(Data(bytes: &sh2, count: 4))
        }

        // Write file
        try data.write(to: fileURL)

        // Verify
        let attributes = try FileManager.default.attributesOfItem(atPath: fileURL.path)
        let fileSize = attributes[.size] as? UInt64 ?? 0
        logDebug("SHARP: PLY file saved (\(fileSize / 1024) KB)")

        return fileURL
    }
}
