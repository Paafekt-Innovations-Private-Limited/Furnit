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
            // Configuration matching mlsharpondevice2 style
            let config = MLModelConfiguration()
            config.computeUnits = .cpuOnly

            // Use FP32 model
            logDebug("SHARP: Loading via auto-generated model class (async) - FP32...")
            let sharpModel = try await SHARP_fp32_1536.load(configuration: config)
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
            status = .failed("SHARP model failed to load. Check device memory.")
            statusMessage = "Model load failed"
            throw GenerationError.serverError("SHARP model failed to load. Check device memory.")
        }

        // Reset state
        status = .processing
        statusMessage = "Preprocessing image..."
        progress = 0.0

        do {
            // Step 1: Preprocess image
            progress = 0.1
            let inputBuffer = try await preprocessImage(image)
            logDebug("SHARP: Image preprocessed to \(Self.inputSize)x\(Self.inputSize)")

            // Step 2: Run inference
            statusMessage = "Generating 3D Gaussians..."
            progress = 0.2
            let gaussianParams = try await runInference(inputBuffer)
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
        } catch {
            // Update status on failure so UI can show error
            status = .failed(error.localizedDescription)
            statusMessage = "Generation failed"
            logDebug("SHARP: Generation failed: \(error)")
            throw error
        }
    }

    /// Cancel current generation (if any)
    func cancelGeneration() {
        status = .failed("Cancelled")
        statusMessage = "Cancelled"
    }

    // MARK: - Image Preprocessing

    /// Preprocess image for SHARP model input
    /// - Parameter image: Source UIImage
    /// - Returns: CVPixelBuffer sized to 1536x1536 (model expects Image type)
    private func preprocessImage(_ image: UIImage) async throws -> CVPixelBuffer {
        guard let cgImage = image.cgImage else {
            throw GenerationError.invalidImage
        }

        let size = Self.inputSize

        // Create CVPixelBuffer for CoreML Image input
        var pixelBuffer: CVPixelBuffer?
        let attrs = [
            kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue!,
            kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue!,
            kCVPixelBufferMetalCompatibilityKey: kCFBooleanTrue!
        ] as CFDictionary

        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            size,
            size,
            kCVPixelFormatType_32BGRA,
            attrs,
            &pixelBuffer
        )

        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
            throw GenerationError.serverError("Failed to create pixel buffer")
        }

        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }

        guard let context = CGContext(
            data: CVPixelBufferGetBaseAddress(buffer),
            width: size,
            height: size,
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
        ) else {
            throw GenerationError.serverError("Failed to create graphics context")
        }

        // Draw resized image into pixel buffer
        context.interpolationQuality = .high
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: size, height: size))

        return buffer
    }

    // MARK: - Model Inference

    /// Run SHARP model inference
    /// SHARP_fp16 outputs 5 separate arrays that we combine into interleaved format
    private func runInference(_ input: CVPixelBuffer) async throws -> [Float] {
        guard let model = model else {
            throw GenerationError.serverError("Model not loaded")
        }

        logDebug("SHARP: Running inference...")

        // Validate input pixel buffer
        let width = CVPixelBufferGetWidth(input)
        let height = CVPixelBufferGetHeight(input)
        let pixelFormat = CVPixelBufferGetPixelFormatType(input)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(input)

        logDebug("SHARP INPUT VALIDATION:")
        logDebug("  Dimensions: \(width)x\(height) (expected 1536x1536)")
        logDebug("  Pixel format: \(pixelFormat) (BGRA=1111970369)")
        logDebug("  Bytes per row: \(bytesPerRow)")

        // Sample pixel values to check for NaN/Inf/range issues
        CVPixelBufferLockBaseAddress(input, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(input, .readOnly) }

        if let baseAddress = CVPixelBufferGetBaseAddress(input) {
            let ptr = baseAddress.assumingMemoryBound(to: UInt8.self)
            let totalBytes = height * bytesPerRow

            // Sample corners and center
            var minVal: UInt8 = 255
            var maxVal: UInt8 = 0
            let samplePoints = [0, totalBytes/4, totalBytes/2, 3*totalBytes/4, totalBytes-4]
            for offset in samplePoints {
                if offset >= 0 && offset < totalBytes {
                    let val = ptr[offset]
                    minVal = min(minVal, val)
                    maxVal = max(maxVal, val)
                }
            }
            logDebug("  Pixel value range: \(minVal)-\(maxVal) (expected 0-255)")

            // Log first pixel (BGRA order)
            if totalBytes >= 4 {
                logDebug("  First pixel BGRA: (\(ptr[0]), \(ptr[1]), \(ptr[2]), \(ptr[3]))")
            }
        }

        // Create feature provider with CVPixelBuffer as Image type
        let imageFeature = MLFeatureValue(pixelBuffer: input)
        let inputFeatures = try MLDictionaryFeatureProvider(dictionary: ["image": imageFeature])

        // Run prediction (mlprogram requires async in iOS 16+)
        let output = try await model.prediction(from: inputFeatures)

        logDebug("SHARP: Inference complete, extracting outputs...")

        // SHARP_f32 outputs separate arrays:
        // - var_5420: positions (1 × N × 3)
        // - var_5424: scales (1 × N × 3)
        // - var_5412: rotations (1 × N × 4)
        // - var_5415: colors (1 × N × 3)
        // - var_5416: opacity (1 × N)
        guard let positions = output.featureValue(for: "var_5420")?.multiArrayValue,
              let scales = output.featureValue(for: "var_5424")?.multiArrayValue,
              let rotations = output.featureValue(for: "var_5412")?.multiArrayValue,
              let colors = output.featureValue(for: "var_5415")?.multiArrayValue,
              let opacities = output.featureValue(for: "var_5416")?.multiArrayValue else {
            // Log available features for debugging
            let availableFeatures = output.featureNames.joined(separator: ", ")
            logDebug("SHARP: Available features: \(availableFeatures)")
            throw GenerationError.serverError("Invalid model output - missing features")
        }

        // Debug: log actual shapes and data types to verify layout
        logDebug("SHARP: Output shapes and types:")
        logDebug("  positions: \(positions.shape) strides: \(positions.strides) dtype: \(positions.dataType.rawValue)")
        logDebug("  scales: \(scales.shape) strides: \(scales.strides) dtype: \(scales.dataType.rawValue)")
        logDebug("  rotations: \(rotations.shape) strides: \(rotations.strides) dtype: \(rotations.dataType.rawValue)")
        logDebug("  colors: \(colors.shape) strides: \(colors.strides) dtype: \(colors.dataType.rawValue)")
        logDebug("  opacities: \(opacities.shape) strides: \(opacities.strides) dtype: \(opacities.dataType.rawValue)")
        // MLMultiArrayDataType: 65552 = float16, 65568 = float32

        // Get gaussian count from positions array (shape: 1 × N × 3)
        let gaussianCount = positions.shape[1].intValue
        logDebug("SHARP: Found \(gaussianCount) gaussians")

        // Combine into interleaved format: pos(3) + scale(3) + rot(4) + opacity(1) + color(3) = 14
        var result = [Float](repeating: 0, count: gaussianCount * Self.paramsPerGaussian)

        // Check if Float16 or Float32
        let isFloat16 = positions.dataType == .float16
        logDebug("SHARP: Data type is \(isFloat16 ? "Float16" : "Float32")")

        // Debug: sample first gaussian values using subscript (safe, respects strides)
        logDebug("SHARP: Sample gaussian 0 (via subscript):")
        logDebug("  pos: (\(positions[[0,0,0] as [NSNumber]]), \(positions[[0,0,1] as [NSNumber]]), \(positions[[0,0,2] as [NSNumber]]))")
        logDebug("  scale: (\(scales[[0,0,0] as [NSNumber]]), \(scales[[0,0,1] as [NSNumber]]), \(scales[[0,0,2] as [NSNumber]]))")
        logDebug("  rot: (\(rotations[[0,0,0] as [NSNumber]]), \(rotations[[0,0,1] as [NSNumber]]), \(rotations[[0,0,2] as [NSNumber]]), \(rotations[[0,0,3] as [NSNumber]]))")
        logDebug("  color: (\(colors[[0,0,0] as [NSNumber]]), \(colors[[0,0,1] as [NSNumber]]), \(colors[[0,0,2] as [NSNumber]]))")
        logDebug("  opacity: \(opacities[[0,0] as [NSNumber]])")

        // Get strides for proper indexing (strides are in element counts, not bytes)
        let posStrides = positions.strides.map { $0.intValue }
        let scaleStrides = scales.strides.map { $0.intValue }
        let rotStrides = rotations.strides.map { $0.intValue }
        let colorStrides = colors.strides.map { $0.intValue }
        let opacityStrides = opacities.strides.map { $0.intValue }

        logDebug("SHARP: Using strides - pos:\(posStrides) scale:\(scaleStrides) rot:\(rotStrides) color:\(colorStrides) opacity:\(opacityStrides)")

        // Interleave data using stride-aware indexing
        if isFloat16 {
            let posPtr = positions.dataPointer.bindMemory(to: Float16.self, capacity: positions.count)
            let scalePtr = scales.dataPointer.bindMemory(to: Float16.self, capacity: scales.count)
            let rotPtr = rotations.dataPointer.bindMemory(to: Float16.self, capacity: rotations.count)
            let colorPtr = colors.dataPointer.bindMemory(to: Float16.self, capacity: colors.count)
            let opacityPtr = opacities.dataPointer.bindMemory(to: Float16.self, capacity: opacities.count)

            for i in 0..<gaussianCount {
                let offset = i * Self.paramsPerGaussian
                let posBase = i * posStrides[1]
                let scaleBase = i * scaleStrides[1]
                let rotBase = i * rotStrides[1]
                let colorBase = i * colorStrides[1]

                result[offset + 0] = Float(posPtr[posBase + 0 * posStrides[2]])
                result[offset + 1] = Float(posPtr[posBase + 1 * posStrides[2]])
                result[offset + 2] = Float(posPtr[posBase + 2 * posStrides[2]])
                result[offset + 3] = Float(scalePtr[scaleBase + 0 * scaleStrides[2]])
                result[offset + 4] = Float(scalePtr[scaleBase + 1 * scaleStrides[2]])
                result[offset + 5] = Float(scalePtr[scaleBase + 2 * scaleStrides[2]])
                result[offset + 6] = Float(rotPtr[rotBase + 0 * rotStrides[2]])
                result[offset + 7] = Float(rotPtr[rotBase + 1 * rotStrides[2]])
                result[offset + 8] = Float(rotPtr[rotBase + 2 * rotStrides[2]])
                result[offset + 9] = Float(rotPtr[rotBase + 3 * rotStrides[2]])
                result[offset + 10] = Float(opacityPtr[i * opacityStrides[1]])
                result[offset + 11] = Float(colorPtr[colorBase + 0 * colorStrides[2]])
                result[offset + 12] = Float(colorPtr[colorBase + 1 * colorStrides[2]])
                result[offset + 13] = Float(colorPtr[colorBase + 2 * colorStrides[2]])
            }
        } else {
            // Float32
            let posPtr = positions.dataPointer.bindMemory(to: Float.self, capacity: positions.count)
            let scalePtr = scales.dataPointer.bindMemory(to: Float.self, capacity: scales.count)
            let rotPtr = rotations.dataPointer.bindMemory(to: Float.self, capacity: rotations.count)
            let colorPtr = colors.dataPointer.bindMemory(to: Float.self, capacity: colors.count)
            let opacityPtr = opacities.dataPointer.bindMemory(to: Float.self, capacity: opacities.count)

            for i in 0..<gaussianCount {
                let offset = i * Self.paramsPerGaussian
                let posBase = i * posStrides[1]
                let scaleBase = i * scaleStrides[1]
                let rotBase = i * rotStrides[1]
                let colorBase = i * colorStrides[1]

                result[offset + 0] = posPtr[posBase + 0 * posStrides[2]]
                result[offset + 1] = posPtr[posBase + 1 * posStrides[2]]
                result[offset + 2] = posPtr[posBase + 2 * posStrides[2]]
                result[offset + 3] = scalePtr[scaleBase + 0 * scaleStrides[2]]
                result[offset + 4] = scalePtr[scaleBase + 1 * scaleStrides[2]]
                result[offset + 5] = scalePtr[scaleBase + 2 * scaleStrides[2]]
                result[offset + 6] = rotPtr[rotBase + 0 * rotStrides[2]]
                result[offset + 7] = rotPtr[rotBase + 1 * rotStrides[2]]
                result[offset + 8] = rotPtr[rotBase + 2 * rotStrides[2]]
                result[offset + 9] = rotPtr[rotBase + 3 * rotStrides[2]]
                result[offset + 10] = opacityPtr[i * opacityStrides[1]]
                result[offset + 11] = colorPtr[colorBase + 0 * colorStrides[2]]
                result[offset + 12] = colorPtr[colorBase + 1 * colorStrides[2]]
                result[offset + 13] = colorPtr[colorBase + 2 * colorStrides[2]]
            }
        }

        logDebug("SHARP: Extracted \(result.count) values (\(gaussianCount) gaussians × 14 params)")
        return result
    }

    // MARK: - Splat Filtering

    /// Maximum number of splats for mobile rendering (higher = better quality, more memory)
    private static let maxSplats: Int = 500_000

    /// Minimum opacity threshold (0-1) for keeping a splat (higher = less black cloud)
    private static let minOpacity: Float = 0.05

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

    // MARK: - Depth Doubling

    /// Double the depth (Z range) of splats to give more 3D separation
    private func doubleDepth(_ params: inout [Float]) {
        let count = params.count / Self.paramsPerGaussian
        guard count > 0 else { return }

        // Find Z range (position Z is at offset 2)
        var zMin: Float = .greatestFiniteMagnitude
        var zMax: Float = -.greatestFiniteMagnitude
        var zSum: Float = 0

        for i in 0..<count {
            let z = params[i * Self.paramsPerGaussian + 2]
            zMin = min(zMin, z)
            zMax = max(zMax, z)
            zSum += z
        }

        let zCenter = zSum / Float(count)
        let depthScale: Float = 6.0  // More depth separation

        logDebug("SHARP: Depth doubling - zMin: \(zMin), zMax: \(zMax), zCenter: \(zCenter)")

        // Scale Z positions away from center
        for i in 0..<count {
            let offset = i * Self.paramsPerGaussian + 2
            let z = params[offset]
            params[offset] = zCenter + (z - zCenter) * depthScale
        }

        // Log new range
        var newZMin: Float = .greatestFiniteMagnitude
        var newZMax: Float = -.greatestFiniteMagnitude
        for i in 0..<count {
            let z = params[i * Self.paramsPerGaussian + 2]
            newZMin = min(newZMin, z)
            newZMax = max(newZMax, z)
        }
        logDebug("SHARP: After doubling - zMin: \(newZMin), zMax: \(newZMax)")
    }

    // MARK: - PLY Writing

    /// Write Gaussian parameters to PLY file
    private func writePLY(_ params: [Float]) async throws -> URL {
        // Filter Gaussians for mobile rendering
        var filteredParams = filterGaussians(params)

        // Double the depth for more 3D separation
        doubleDepth(&filteredParams)

        let gaussianCount = filteredParams.count / Self.paramsPerGaussian

        // Ensure output directory exists
        try FileManager.default.createDirectory(at: modelsDirectory, withIntermediateDirectories: true)

        // Generate filename
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "yyyyMMdd_HHmmss"
        let timestamp = dateFormatter.string(from: Date())
        let fileName = "Room_\(timestamp).ply"
        let fileURL = modelsDirectory.appendingPathComponent(fileName)

        // Build PLY content - use uchar red/green/blue for proper color display
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
        property uchar red
        property uchar green
        property uchar blue
        end_header\n
        """

        // Write header
        var data = Data(plyContent.utf8)

        // Track bounding box for measurements
        var minX: Float = .greatestFiniteMagnitude
        var maxX: Float = -.greatestFiniteMagnitude
        var minY: Float = .greatestFiniteMagnitude
        var maxY: Float = -.greatestFiniteMagnitude
        var minZ: Float = .greatestFiniteMagnitude
        var maxZ: Float = -.greatestFiniteMagnitude

        // Write binary vertex data
        // Each Gaussian: pos(3) + scale(3) + rot(4) + opacity(1) + sh(3) = 14 floats
        // Using raw SHARP output values directly
        for i in 0..<gaussianCount {
            let offset = i * Self.paramsPerGaussian

            // Position - face front, tilt 90° left, correct mirror
            let origX = filteredParams[offset + 0]
            let origY = filteredParams[offset + 1]
            let origZ = filteredParams[offset + 2]
            var x = -origY        // -Y becomes X (flip to fix mirror)
            var y = -origX        // -X becomes Y
            var z = -origZ        // Z negated

            // Track bounding box
            minX = min(minX, x); maxX = max(maxX, x)
            minY = min(minY, y); maxY = max(maxY, y)
            minZ = min(minZ, z); maxZ = max(maxZ, z)

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

            // Color - convert linear [0,1] to sRGB [0,255] with gamma
            let rawR = filteredParams[offset + 11]
            let rawG = filteredParams[offset + 12]
            let rawB = filteredParams[offset + 13]

            // Apply gamma correction (linear to sRGB) and slight brightness boost
            let gamma: Float = 1.0 / 2.2
            let brightness: Float = 1.1
            let r = pow(min(max(rawR * brightness, 0), 1), gamma)
            let g = pow(min(max(rawG * brightness, 0), 1), gamma)
            let b = pow(min(max(rawB * brightness, 0), 1), gamma)

            // Convert to 0-255 uchar
            var red = UInt8(min(max(Int(r * 255), 0), 255))
            var green = UInt8(min(max(Int(g * 255), 0), 255))
            var blue = UInt8(min(max(Int(b * 255), 0), 255))
            data.append(Data(bytes: &red, count: 1))
            data.append(Data(bytes: &green, count: 1))
            data.append(Data(bytes: &blue, count: 1))
        }

        // Write file
        try data.write(to: fileURL)

        // Verify
        let attributes = try FileManager.default.attributesOfItem(atPath: fileURL.path)
        let fileSize = attributes[.size] as? UInt64 ?? 0
        logDebug("SHARP: PLY file saved (\(fileSize / 1024) KB)")

        // Log room measurements (SHARP units are roughly meters)
        let width = maxX - minX
        let height = maxY - minY
        let depth = maxZ - minZ
        logDebug("SHARP: Room measurements:")
        logDebug("  Width (X):  \(String(format: "%.2f", width)) units (\(minX) to \(maxX))")
        logDebug("  Height (Y): \(String(format: "%.2f", height)) units (\(minY) to \(maxY))")
        logDebug("  Depth (Z):  \(String(format: "%.2f", depth)) units (\(minZ) to \(maxZ))")
        logDebug("  Splats: \(gaussianCount)")

        return fileURL
    }
}
