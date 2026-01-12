import Foundation
import UIKit
import CoreML
import Metal
import MetalKit
import MetalPerformanceShaders
import Accelerate
import simd

/// On-device 3D Gaussian generation using Apple's SHARP model
/// Replaces remote Jarvis-based Room3DGenerationService with local CoreML inference
@MainActor
class LandscapeSHARPService: ObservableObject {

    // MARK: - Published Properties

    /// Current generation status
    @Published var status: GenerationStatus = .idle

    /// Human-readable status message
    @Published var statusMessage: String = "Ready"

    /// Processing progress (0.0 to 1.0)
    @Published var progress: Float = 0.0

    /// Room measurements from plane detection (available after generation)
    @Published var roomMeasurements: RoomMeasurements?

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

    /// Directory for generated PLY files (temp until user explicitly saves)
    private var modelsDirectory: URL {
        let tempDirectory = FileManager.default.temporaryDirectory
        return tempDirectory.appendingPathComponent("SHARPModels", isDirectory: true)
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

    /// Whether model is currently loading
    @Published var isLoadingModel: Bool = false

    /// Load the SHARP CoreML model
    private func loadModel() async {
        logDebug("SHARP: Loading CoreML model...")

        isLoadingModel = true
        statusMessage = "Getting ready..."
        progress = 0.1

        // Check physical memory - INT8 model (~663MB) needs ~2GB available
        let physicalMemory = ProcessInfo.processInfo.physicalMemory
        let memoryLimit: UInt64 = 2 * 1024 * 1024 * 1024  // 2GB minimum

        logDebug("SHARP: Device RAM: \(physicalMemory / 1024 / 1024)MB")

        if physicalMemory < memoryLimit {
            logDebug("SHARP: Device has insufficient RAM, need 2GB+")
            logDebug("SHARP: Model will not be loaded - use fallback or remote API")
            isLoadingModel = false
            statusMessage = "Not enough space on device"
            return
        }

        progress = 0.3
        statusMessage = "Setting things up..."

        do {
            // Configuration matching mlsharpondevice2 style
            let config = MLModelConfiguration()
            config.computeUnits = .cpuOnly

            // Use FP32 model
            logDebug("SHARP: Loading via auto-generated model class (async) - FP32...")
            let sharpModel = try await SHARP_fp32_1536.load(configuration: config)
            model = sharpModel.model
            logDebug("SHARP: Model loaded successfully")

            progress = 1.0
            statusMessage = "Ready!"

        } catch {
            logDebug("SHARP: Failed to load model: \(error)")
            statusMessage = "Couldn't get ready"
        }

        isLoadingModel = false
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
            statusMessage = "Something went wrong"
            throw GenerationError.serverError("SHARP model failed to load. Check device memory.")
        }

        // Reset state
        status = .processing
        statusMessage = "Preparing your photo..."
        progress = 0.0

        do {
            // Step 1: Preprocess image
            progress = 0.1
            let inputBuffer = try await preprocessImage(image)
            logDebug("SHARP: Image preprocessed to \(Self.inputSize)x\(Self.inputSize)")

            // Step 2: Run inference
            statusMessage = "Creating your 3D room..."
            progress = 0.2
            let gaussianParams = try await runInference(inputBuffer)
            logDebug("SHARP: Generated \(gaussianParams.count / Self.paramsPerGaussian) Gaussians")

            // Step 3: Write PLY files (original + classic for antimatter15)
            statusMessage = "Almost done..."
            progress = 0.8
            let plyURLs = try await writePLY(gaussianParams)
            logDebug("SHARP: Saved PLY to \(plyURLs.original.path)")
            logDebug("SHARP: Saved Classic PLY to \(plyURLs.classic.path)")
            logDebug("SHARP: Saved 3DGS PLY to \(plyURLs.threeDGS.path)")
            let plyURL = plyURLs.original

            // Complete
            progress = 1.0
            status = .completed(fileURL: plyURL)
            statusMessage = "Done!"

            return plyURL
        } catch {
            // Update status on failure so UI can show error
            status = .failed(error.localizedDescription)
            statusMessage = "Couldn't create room"
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

    /// Hard cutoff for almost-transparent splats
    private static let minAlpha: Float = 0.10

    /// For "fluffy fog" filter: if alpha < this AND scale > fogScaleThreshold, drop it
    private static let fogAlphaThreshold: Float = 0.25
    private static let fogScaleThreshold: Float = 0.030

    /// Treat very dark splats as black fog and drop them
    private static let blackBrightnessThreshold: Float = 0.05  // ~13/255 - very conservative

    /// Filter Gaussians - opacity, fog and black-fog removal
    /// Params layout per gaussian:
    /// [0] pos.x, [1] pos.y, [2] pos.z,
    /// [3] scale.x, [4] scale.y, [5] scale.z,
    /// [6] rot.x, [7] rot.y, [8] rot.z, [9] rot.w,
    /// [10] opacity (alpha 0-1),
    /// [11] colorR (0-1), [12] colorG (0-1), [13] colorB (0-1)
    private func filterGaussians(_ params: [Float]) -> [Float] {
        let gaussianCount = params.count / Self.paramsPerGaussian
        guard gaussianCount > 0 else { return [] }

        // --- Brightness stats (for debug) ---
        var minBrightness: Float = 1.0
        var maxBrightness: Float = 0.0
        let statsCount = min(gaussianCount, 50_000)  // don't scan all 1.1M for stats

        for i in 0..<statsCount {
            let offset = i * Self.paramsPerGaussian
            let r = params[offset + 11]
            let g = params[offset + 12]
            let b = params[offset + 13]
            let brightness = max(0.0, min(1.0, (r + g + b) / 3.0))
            minBrightness = min(minBrightness, brightness)
            maxBrightness = max(maxBrightness, brightness)
        }

        logDebug(String(
            format: "SHARP BRIGHTNESS: min=%.3f max=%.3f",
            minBrightness, maxBrightness
        ))

        // --- Main filtering pass ---
        var filtered: [Float] = []
        filtered.reserveCapacity(params.count)

        var opacityFiltered = 0
        var fogFiltered = 0
        var blackFogFiltered = 0

        for i in 0..<gaussianCount {
            let offset = i * Self.paramsPerGaussian

            let sx = params[offset + 3]
            let sy = params[offset + 4]

            let alpha = params[offset + 10]      // already 0-1 at this point
            let r = params[offset + 11]
            let g = params[offset + 12]
            let b = params[offset + 13]

            // 1) Basic opacity cull
            if alpha < Self.minAlpha {
                opacityFiltered += 1
                continue
            }

            // 2) Fog cull: big splats with low alpha (tend to be smeary blobs)
            let maxScaleXY = max(sx, sy)
            if alpha < Self.fogAlphaThreshold && maxScaleXY > Self.fogScaleThreshold {
                fogFiltered += 1
                continue
            }

            // 3) Black-fog cull: very dark splats (cause the black patch)
            let brightness = (r + g + b) / 3.0
            if brightness < Self.blackBrightnessThreshold {
                blackFogFiltered += 1
                continue
            }

            // If we got here, keep this gaussian (copy all 14 floats)
            for j in 0..<Self.paramsPerGaussian {
                filtered.append(params[offset + j])
            }
        }

        let kept = filtered.count / Self.paramsPerGaussian
        logDebug("SHARP: Filtering \(gaussianCount) Gaussians...")
        logDebug("SHARP: Filtered \(opacityFiltered) by opacity, " +
                 "\(fogFiltered) by fog, \(blackFogFiltered) by black fog, kept \(kept)")

        return filtered
    }

    // MARK: - PLY Writing

    /// Write Gaussian parameters to PLY file
    /// Returns tuple: (originalURL, classicURL, threeDGSURL) for different viewer compatibility
    private func writePLY(_ params: [Float]) async throws -> (original: URL, classic: URL, threeDGS: URL) {
        // Filter Gaussians for mobile rendering
        let filteredParams = filterGaussians(params)

        let gaussianCount = filteredParams.count / Self.paramsPerGaussian

        // Ensure output directory exists
        try FileManager.default.createDirectory(at: modelsDirectory, withIntermediateDirectories: true)

        // Generate filenames
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "yyyyMMdd_HHmmss"
        let timestamp = dateFormatter.string(from: Date())
        let fileName = "Room_\(timestamp).ply"
        let classicFileName = "Room_\(timestamp)_classic.ply"
        let threeDGSFileName = "Room_\(timestamp)_3dgs.ply"
        let fileURL = modelsDirectory.appendingPathComponent(fileName)
        let classicFileURL = modelsDirectory.appendingPathComponent(classicFileName)
        let threeDGSFileURL = modelsDirectory.appendingPathComponent(threeDGSFileName)

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

        // 3DGS format header (SuperSplat compatible)
        // Must use f_dc_* for colors (SuperSplat requirement)
        let threeDGSHeader = """
        ply
        format binary_little_endian 1.0
        element vertex \(gaussianCount)
        property float x
        property float y
        property float z
        property float nx
        property float ny
        property float nz
        property float f_dc_0
        property float f_dc_1
        property float f_dc_2
        property float opacity
        property float scale_0
        property float scale_1
        property float scale_2
        property float rot_0
        property float rot_1
        property float rot_2
        property float rot_3
        end_header\n
        """

        // SH coefficient constant: C0 = 0.5 * sqrt(1/pi)
        let SH_C0: Float = 0.28209479177387814

        // Write headers for all files
        var data = Data(plyContent.utf8)
        var classicData = Data(plyContent.utf8)
        var threeDGSData = Data(threeDGSHeader.utf8)

        // Track bounding box and positions for measurements
        var minX: Float = .greatestFiniteMagnitude
        var maxX: Float = -.greatestFiniteMagnitude
        var minY: Float = .greatestFiniteMagnitude
        var maxY: Float = -.greatestFiniteMagnitude
        var minZ: Float = .greatestFiniteMagnitude
        var maxZ: Float = -.greatestFiniteMagnitude
        var positions: [(Float, Float, Float)] = []
        positions.reserveCapacity(gaussianCount)

        // Debug: Log final PLY values for first few gaussians
        var debugCount = 0
        let debugMax = 3

        // Write binary vertex data
        // Each Gaussian: pos(3) + scale(3) + rot(4) + opacity(1) + sh(3) = 14 floats
        for i in 0..<gaussianCount {
            let offset = i * Self.paramsPerGaussian

            // Position - Python transform (x, -y, -z) - DO NOT modify further, breaks quality
            let origX = filteredParams[offset + 0]
            let origY = filteredParams[offset + 1]
            let origZ = filteredParams[offset + 2]
            var x = origX
            var y = -origY
            var z = -origZ

            // Track bounding box and collect positions
            minX = min(minX, x); maxX = max(maxX, x)
            minY = min(minY, y); maxY = max(maxY, y)
            minZ = min(minZ, z); maxZ = max(maxZ, z)
            positions.append((x, y, z))

            // Original PLY
            data.append(Data(bytes: &x, count: 4))
            data.append(Data(bytes: &y, count: 4))
            data.append(Data(bytes: &z, count: 4))

            // Classic PLY: front (180° Y) + upright (180° Z): (x, y, z) → (x, -y, -z)
            var classicX = x
            var classicY = -y
            var classicZ = -z
            classicData.append(Data(bytes: &classicX, count: 4))
            classicData.append(Data(bytes: &classicY, count: 4))
            classicData.append(Data(bytes: &classicZ, count: 4))

            // 3DGS format: position
            threeDGSData.append(Data(bytes: &x, count: 4))
            threeDGSData.append(Data(bytes: &y, count: 4))
            threeDGSData.append(Data(bytes: &z, count: 4))

            // 3DGS format: normals (zeros)
            var nx: Float = 0.0
            var ny: Float = 0.0
            var nz: Float = 0.0
            threeDGSData.append(Data(bytes: &nx, count: 4))
            threeDGSData.append(Data(bytes: &ny, count: 4))
            threeDGSData.append(Data(bytes: &nz, count: 4))

            // Scale - convert to log for renderer (match Python's scale_boost=1.3)
            let minScale: Float = 0.001
            let scaleBoost: Float = 1.3  // Match Python test_sharp_coreml.py
            let rawS0 = filteredParams[offset + 3] * scaleBoost
            let rawS1 = filteredParams[offset + 4] * scaleBoost
            let rawS2 = filteredParams[offset + 5] * scaleBoost
            var s0 = log(max(rawS0, minScale))
            var s1 = log(max(rawS1, minScale))
            var s2 = log(max(rawS2, minScale))
            data.append(Data(bytes: &s0, count: 4))
            data.append(Data(bytes: &s1, count: 4))
            data.append(Data(bytes: &s2, count: 4))
            classicData.append(Data(bytes: &s0, count: 4))
            classicData.append(Data(bytes: &s1, count: 4))
            classicData.append(Data(bytes: &s2, count: 4))

            // Rotation quaternion - normalize only (don't transform, preserves quality)
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
            classicData.append(Data(bytes: &r0, count: 4))
            classicData.append(Data(bytes: &r1, count: 4))
            classicData.append(Data(bytes: &r2, count: 4))
            classicData.append(Data(bytes: &r3, count: 4))

            // Opacity - convert to logit (inverse sigmoid) for renderer
            let rawOpacity = filteredParams[offset + 10]
            let clampedOpacity = min(max(rawOpacity, 1e-4), 1.0 - 1e-4)
            var opacity = log(clampedOpacity / (1.0 - clampedOpacity))
            data.append(Data(bytes: &opacity, count: 4))
            classicData.append(Data(bytes: &opacity, count: 4))

            // Color - get raw values from SHARP
            let rawR = filteredParams[offset + 11]
            let rawG = filteredParams[offset + 12]
            let rawB = filteredParams[offset + 13]

            // 3DGS format: f_dc colors as SH coefficients
            // Formula: f_dc = (color - 0.5) / SH_C0
            // SparkJS shows correct colors with RGB order, so use RGB (not BGR)
            var f_dc_0 = (rawR - 0.5) / SH_C0  // R
            var f_dc_1 = (rawG - 0.5) / SH_C0  // G
            var f_dc_2 = (rawB - 0.5) / SH_C0  // B
            threeDGSData.append(Data(bytes: &f_dc_0, count: 4))
            threeDGSData.append(Data(bytes: &f_dc_1, count: 4))
            threeDGSData.append(Data(bytes: &f_dc_2, count: 4))

            // 3DGS format: opacity, scale, rotation
            threeDGSData.append(Data(bytes: &opacity, count: 4))
            threeDGSData.append(Data(bytes: &s0, count: 4))
            threeDGSData.append(Data(bytes: &s1, count: 4))
            threeDGSData.append(Data(bytes: &s2, count: 4))
            threeDGSData.append(Data(bytes: &r0, count: 4))
            threeDGSData.append(Data(bytes: &r1, count: 4))
            threeDGSData.append(Data(bytes: &r2, count: 4))
            threeDGSData.append(Data(bytes: &r3, count: 4))

            // Apply gamma correction (linear to sRGB) and slight brightness boost for SparkJS format
            let gamma: Float = 1.0 / 2.2
            let brightness: Float = 1.1
            let finalR = pow(min(max(rawR * brightness, 0), 1), gamma)
            let finalG = pow(min(max(rawG * brightness, 0), 1), gamma)
            let finalB = pow(min(max(rawB * brightness, 0), 1), gamma)

            // Convert to 0-255 uchar for original/classic PLY
            var red = UInt8(min(max(Int(finalR * 255), 0), 255))
            var green = UInt8(min(max(Int(finalG * 255), 0), 255))
            var blue = UInt8(min(max(Int(finalB * 255), 0), 255))
            data.append(Data(bytes: &red, count: 1))
            data.append(Data(bytes: &green, count: 1))
            data.append(Data(bytes: &blue, count: 1))
            classicData.append(Data(bytes: &red, count: 1))
            classicData.append(Data(bytes: &green, count: 1))
            classicData.append(Data(bytes: &blue, count: 1))

            // Debug logging for first few gaussians
            if debugCount < debugMax {
                let quatLen = sqrt(r0*r0 + r1*r1 + r2*r2 + r3*r3)
                logDebug("SHARP PLY GAUSSIAN \(i) (final values):")
                logDebug("  pos: (\(x), \(y), \(z))")
                logDebug("  scale (log): (\(s0), \(s1), \(s2))")
                logDebug("  rot: (\(r0), \(r1), \(r2), \(r3)) len=\(quatLen)")
                logDebug("  opacity (logit): \(opacity)")
                logDebug("  f_dc (SH): (\(f_dc_0), \(f_dc_1), \(f_dc_2))")
                logDebug("  color (uchar): (\(red), \(green), \(blue))")
                debugCount += 1
            }
        }

        // Write all files
        try data.write(to: fileURL)
        try classicData.write(to: classicFileURL)
        try threeDGSData.write(to: threeDGSFileURL)

        // Verify
        let attributes = try FileManager.default.attributesOfItem(atPath: fileURL.path)
        let fileSize = attributes[.size] as? UInt64 ?? 0
        let threeDGSAttributes = try FileManager.default.attributesOfItem(atPath: threeDGSFileURL.path)
        let threeDGSSize = threeDGSAttributes[.size] as? UInt64 ?? 0
        logDebug("SHARP: PLY file saved (\(fileSize / 1024) KB)")
        logDebug("SHARP: Classic PLY saved (point inversion for antimatter15/splat)")
        logDebug("SHARP: 3DGS PLY saved (\(threeDGSSize / 1024) KB) - SuperSplat compatible")

        // Log room measurements (SHARP units are roughly meters)
        let width = maxX - minX
        let height = maxY - minY
        let depth = maxZ - minZ
        logDebug("SHARP: Room bounding box:")
        logDebug("  Width (X):  \(String(format: "%.2f", width)) units (\(minX) to \(maxX))")
        logDebug("  Height (Y): \(String(format: "%.2f", height)) units (\(minY) to \(maxY))")
        logDebug("  Depth (Z):  \(String(format: "%.2f", depth)) units (\(minZ) to \(maxZ))")
        logDebug("  Splats: \(gaussianCount)")

        // Detect front wall and compute measurements
        if let measurements = RoomMeasurement.measureRoom(positions: positions) {
            await MainActor.run {
                self.roomMeasurements = measurements
            }
            logDebug("SHARP: Front wall detected:")
            logDebug("  Width:  \(String(format: "%.2f", measurements.frontWallWidth)) units")
            logDebug("  Height: \(String(format: "%.2f", measurements.frontWallHeight)) units")
            logDebug("  Confidence: \(String(format: "%.0f", measurements.confidence * 100))%")
        } else {
            logDebug("SHARP: Could not detect front wall")
        }

        return (original: fileURL, classic: classicFileURL, threeDGS: threeDGSFileURL)
    }
}
