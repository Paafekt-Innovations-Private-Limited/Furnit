import Foundation
import UIKit
import CoreML

/// App sandbox temp directory for unsaved Sharp previews (`Room_*_classic.ply`, thumbnails). Not `Documents/SavedRooms`.
private func sharpModelsTemporaryDirectoryURL() -> URL {
    FileManager.default.temporaryDirectory.appendingPathComponent("SHARPModels", isDirectory: true)
}

/// PLY URL plus axis-aligned bounds in **scene units** for **`_classic.ply`** (same vertex frame as in-app Metal viewer).
struct SHARPGenerationResult: Sendable {
    let plyURL: URL
    let plyAabbWidth: Float
    let plyAabbHeight: Float
    let plyAabbDepth: Float
}

/// On-device 3D Gaussian generation using Apple's SHARP model
/// Replaces remote Jarvis-based Room3DGenerationService with local CoreML inference
@MainActor
class SHARPService: ObservableObject {

    // MARK: - Singleton
    /// Shared instance - reuses loaded model across views to prevent memory pressure
    static let shared = SHARPService()

    // MARK: - On-Demand Resources

    /// Tag for the SHARP model in On-Demand Resources
    private static let sharpModelTag = "SHARPModel"

    /// Resource request for ODR
    private var resourceRequest: NSBundleResourceRequest?
    /// Background model load started from UI flows like old room-generation screens.
    private var modelLoadTask: Task<Void, Never>?

    /// Whether the ODR resources are currently being downloaded
    @Published var isDownloadingResources: Bool = false

    /// Download progress (0.0 to 1.0) for ODR
    @Published var downloadProgress: Double = 0.0

    /// Whether ODR resources are available locally
    @Published var resourcesAvailable: Bool = false

    // MARK: - Published Properties

    /// Current generation status
    @Published var status: GenerationStatus = .idle

    /// Human-readable status message
    @Published var statusMessage: String = L10n.Sharp.ready

    /// Processing progress (0.0 to 1.0)
    @Published var progress: Float = 0.0

    // MARK: - Model Configuration

    /// Input image size expected by SHARP model
    private static let inputSize: Int = 1536

    /// Gaussian parameter count per splat: pos(3) + scale(3) + rot(4) + opacity(1) + sh(3) = 14
    private static let paramsPerGaussian: Int = 14

    /// Longer edge of picked photos is often 4k+; decoding/drawing that while loading SHARP peaks RAM on 4GB devices (e.g. iPhone 12).
    /// Aggressive caps on ≤6GB phones shrink the `UIImage` before the 1536² stretch (model input size unchanged).
    private static func maxSourcePixelDimensionBeforeSharp() -> CGFloat {
        let b = ProcessInfo.processInfo.physicalMemory
        let conservativeCap: CGFloat
        if b < 4 * 1024 * 1024 * 1024 {
            conservativeCap = 1024
        } else if b < 6 * 1024 * 1024 * 1024 {
            conservativeCap = 1280
        } else {
            conservativeCap = CGFloat(inputSize)
        }
        // The SHARP model always consumes 1536². Keeping a larger source image only increases decode pressure.
        return min(conservativeCap, CGFloat(inputSize))
    }

    /// Creates an oriented working image for SHARP in a single render pass.
    /// This avoids allocating a full-resolution `.fixedOrientation()` intermediate before downsizing.
    static func prepareImageForSharp(_ image: UIImage) -> UIImage {
        let orientedSize = orientedPixelSize(for: image)
        let w = orientedSize.width
        let h = orientedSize.height
        guard w > 1, h > 1 else { return image }

        let maxEdge = max(w, h)
        let limit = maxSourcePixelDimensionBeforeSharp()
        let needsOrientationFix = image.imageOrientation != .up
        let needsResize = maxEdge > limit + 0.5
        guard needsOrientationFix || needsResize else { return image }

        let scaleDown = min(1, limit / maxEdge)
        let newW = max(1, Int((w * scaleDown).rounded(.down)))
        let newH = max(1, Int((h * scaleDown).rounded(.down)))
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        format.opaque = false
        let renderer = UIGraphicsImageRenderer(size: CGSize(width: newW, height: newH), format: format)
        let rendered = renderer.image { _ in
            image.draw(in: CGRect(x: 0, y: 0, width: newW, height: newH))
        }
        logDebug(
            "SHARP: Prepared photo for memory/orientation \(Int(w))×\(Int(h)) orient=\(image.imageOrientation.rawValue) " +
            "→ \(newW)×\(newH) (max edge ≤\(Int(limit))px)"
        )
        return rendered
    }

    /// Pixel size after applying UIImage orientation semantics.
    private static func orientedPixelSize(for image: UIImage) -> CGSize {
        let baseWidth: CGFloat
        let baseHeight: CGFloat
        if let cgImage = image.cgImage {
            baseWidth = CGFloat(cgImage.width)
            baseHeight = CGFloat(cgImage.height)
        } else {
            baseWidth = image.size.width * image.scale
            baseHeight = image.size.height * image.scale
        }

        switch image.imageOrientation {
        case .left, .leftMirrored, .right, .rightMirrored:
            return CGSize(width: baseHeight, height: baseWidth)
        default:
            return CGSize(width: baseWidth, height: baseHeight)
        }
    }

    /// Match Android's post-inference correction when a non-square photo was stretched into SHARP's square input.
    static func sharpAspectCorrectionFactors(for imageSize: CGSize) -> (x: Float, y: Float) {
        let width = max(Float(imageSize.width), 1)
        let height = max(Float(imageSize.height), 1)
        guard abs(width - height) > 0.5 else { return (1, 1) }

        let rawCorrX = Float(inputSize) / width
        let rawCorrY = Float(inputSize) / height
        let geomMean = sqrt(rawCorrX * rawCorrY)
        guard geomMean.isFinite, geomMean > 0 else { return (1, 1) }
        return (rawCorrX / geomMean, rawCorrY / geomMean)
    }

    /// Wall-measurement thumbnail: avoid full-res `pngData()` (second memory spike after inference).
    private static func imageForWallMeasurementThumbnail(_ image: UIImage, maxPixel: CGFloat = 1024) -> UIImage {
        let w = image.size.width * image.scale
        let h = image.size.height * image.scale
        guard w > 1, h > 1 else { return image }
        let maxEdge = max(w, h)
        guard maxEdge > maxPixel else { return image }
        let scaleDown = maxPixel / maxEdge
        let newW = max(1, Int((w * scaleDown).rounded(.down)))
        let newH = max(1, Int((h * scaleDown).rounded(.down)))
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        format.opaque = false
        let renderer = UIGraphicsImageRenderer(size: CGSize(width: newW, height: newH), format: format)
        return renderer.image { _ in
            image.draw(in: CGRect(x: 0, y: 0, width: newW, height: newH))
        }
    }

    // MARK: - CoreML Model

    /// The compiled SHARP CoreML model
    private var model: MLModel?

    /// Progress observation for ODR download
    private var progressObservation: NSKeyValueObservation?

    // MARK: - File Management

    /// Directory for generated PLY files (temp until user explicitly saves)
    private var modelsDirectory: URL { sharpModelsTemporaryDirectoryURL() }

    /// Deletes all files under the temp `SHARPModels` folder. Call on cold launch so restarting from Xcode
    /// after an abandoned Sharp Room preview does not leave `Room_*_classic.ply` (and thumbnails) on disk.
    nonisolated static func purgeTemporarySharpModelsDirectoryAtLaunch() {
        let directoryURL = sharpModelsTemporaryDirectoryURL()
        let fileManager = FileManager.default
        guard fileManager.fileExists(atPath: directoryURL.path) else { return }
        do {
            let childURLs = try fileManager.contentsOfDirectory(
                at: directoryURL,
                includingPropertiesForKeys: nil,
                options: [.skipsHiddenFiles]
            )
            for url in childURLs {
                try fileManager.removeItem(at: url)
            }
            logDebug("SHARP: purged \(childURLs.count) preview file(s) from temporary SHARPModels at launch")
        } catch {
            logDebug("SHARP: purge SHARPModels at launch failed: \(error.localizedDescription)")
        }
    }

    // MARK: - Initialization

    init() {
        logDebug("SHARP: Service initialized (CPU-only inference, Metal deferred)")
    }

    // MARK: - On-Demand Resources

    /// Check if running from Xcode (development) vs App Store
    private var isRunningFromXcode: Bool {
        #if DEBUG
        // Debug builds are always from Xcode - ODR only works in App Store/TestFlight
        return true
        #else
        return false
        #endif
    }

    /// Check if SHARP model resources are available locally
    func checkResourceAvailability() async {
        // When running from Xcode, model is bundled - skip ODR
        if isRunningFromXcode {
            logDebug("SHARP: Running from Xcode - model bundled locally, skipping ODR")
            resourcesAvailable = true
            return
        }

        let tags: Set<String> = [Self.sharpModelTag]

        // Check if resources are already downloaded
        let available = Bundle.main.preservationPriority(forTag: Self.sharpModelTag) > 0
        logDebug("SHARP: ODR availability check - preservationPriority > 0: \(available)")

        // Also check by attempting conditional begin access
        let request = NSBundleResourceRequest(tags: tags)
        request.loadingPriority = NSBundleResourceRequestLoadingPriorityUrgent

        let conditionallyAvailable = await request.conditionallyBeginAccessingResources()
        resourcesAvailable = conditionallyAvailable
        logDebug("SHARP: ODR conditionallyBeginAccessingResources: \(conditionallyAvailable)")

        if conditionallyAvailable {
            // Keep the request alive to prevent purging
            self.resourceRequest = request
        }
    }

    /// Download SHARP model resources if not available
    /// - Returns: true if resources are available after this call
    func downloadResourcesIfNeeded() async throws -> Bool {
        // When running from Xcode, model is bundled - no download needed
        if isRunningFromXcode {
            logDebug("SHARP: Running from Xcode - model bundled, no download needed")
            resourcesAvailable = true
            return true
        }

        // Already available
        if resourcesAvailable {
            logDebug("SHARP: Resources already available")
            return true
        }

        // Already downloading
        if isDownloadingResources {
            logDebug("SHARP: Download already in progress")
            // Wait for existing download to complete
            while isDownloadingResources {
                try await Task.sleep(nanoseconds: 100_000_000) // 100ms
            }
            return resourcesAvailable
        }

        logDebug("SHARP: Starting ODR download...")
        isDownloadingResources = true
        downloadProgress = 0.0
        statusMessage = L10n.Sharp.downloadingEngine

        let tags: Set<String> = [Self.sharpModelTag]
        let request = NSBundleResourceRequest(tags: tags)
        request.loadingPriority = NSBundleResourceRequestLoadingPriorityUrgent

        // Observe download progress
        progressObservation = request.progress.observe(\.fractionCompleted, options: [.new]) { [weak self] progress, _ in
            Task { @MainActor in
                self?.downloadProgress = progress.fractionCompleted
                let percent = Int(progress.fractionCompleted * 100)
                self?.statusMessage = L10n.Sharp.downloadingEnginePercent(percent)
            }
        }

        do {
            try await request.beginAccessingResources()
            logDebug("SHARP: ODR download complete")

            resourcesAvailable = true
            isDownloadingResources = false
            downloadProgress = 1.0
            statusMessage = L10n.Sharp.downloadComplete

            // Keep the request alive
            self.resourceRequest = request
            progressObservation = nil

            return true
        } catch {
            logDebug("SHARP: ODR download failed: \(error)")

            isDownloadingResources = false
            downloadProgress = 0.0
            statusMessage = L10n.Sharp.downloadFailed
            progressObservation = nil

            throw error
        }
    }

    /// Release ODR resources (call when no longer needed to free disk space)
    func releaseResources() {
        modelLoadTask?.cancel()
        modelLoadTask = nil
        resourceRequest?.endAccessingResources()
        resourceRequest = nil
        resourcesAvailable = false
        model = nil
        isLoadingModel = false
        logDebug("SHARP: Released ODR resources")
    }

    /// Drop the in-memory CoreML model after PLY is written so `WKWebView` / WebKit can allocate.
    /// Without this, peak RAM (SHARP + YOLOE + WebKit) can fail heap allocation at `WKWebViewConfiguration()`.
    func releaseInferenceMemoryAfterGeneration() {
        modelLoadTask?.cancel()
        modelLoadTask = nil
        model = nil
        isLoadingModel = false
        logDebug("SHARP: Inference model released after generation (free RAM for splat viewer)")
    }

    // MARK: - Model Loading

    /// Whether model is currently loading
    @Published var isLoadingModel: Bool = false

    /// Ensure model is loaded — call from view's onAppear to re-trigger loading
    /// after releaseResources() has cleared the model.
    func ensureModelLoaded() {
        guard model == nil && !isLoadingModel else { return }
        modelLoadTask?.cancel()
        modelLoadTask = Task { [weak self] in
            guard let self else { return }
            await loadModel()
        }
    }

    /// Load the SHARP CoreML model
    private func loadModel() async {
        // Prevent double-loading if another Task already started
        guard !isLoadingModel && model == nil else { return }
        defer { modelLoadTask = nil }
        logDebug("SHARP: Loading CoreML model...")

        isLoadingModel = true
        statusMessage = L10n.Sharp.gettingReady
        progress = 0.1

        // Check physical memory. FP32 SHARP at 1536² is heavy on low-memory devices; below 2 GB we bail out.
        let physicalMemory = ProcessInfo.processInfo.physicalMemory
        let twoGB: UInt64 = 2 * 1024 * 1024 * 1024

        logDebug("SHARP: Device RAM: \(physicalMemory / 1024 / 1024)MB")

        if physicalMemory < twoGB {
            logDebug("SHARP: Device has <2GB RAM — skipping on-device SHARP")
            isLoadingModel = false
            statusMessage = L10n.Sharp.notEnoughSpace
            return
        }

        // Best-effort ODR download — model may already be compiled into the bundle
        if !resourcesAvailable && !isRunningFromXcode {
            do {
                let downloaded = try await downloadResourcesIfNeeded()
                if Task.isCancelled {
                    isLoadingModel = false
                    return
                }
                if downloaded {
                    logDebug("SHARP: ODR download succeeded")
                } else {
                    logDebug("SHARP: ODR download returned false — will try bundle load anyway")
                }
            } catch {
                logDebug("SHARP: ODR download failed (\(error)) — model may be in app bundle, proceeding...")
            }
        }

        progress = 0.3
        statusMessage = L10n.Sharp.settingThingsUp

        do {
            // CPU-only: avoids GPU IOSurface / Neural Engine allocation failures on constrained devices.
            let config = MLModelConfiguration()
            config.computeUnits = .cpuOnly

            logDebug("SHARP: Loading via auto-generated model class (async) - FP32, computeUnits=cpuOnly")
            let sharpModel = try await SHARP_fp32_1536.load(configuration: config)
            if Task.isCancelled {
                isLoadingModel = false
                return
            }
            model = sharpModel.model
            logDebug("SHARP: Model loaded successfully with computeUnits=cpuOnly")

            progress = 1.0
            statusMessage = L10n.Sharp.ready
        } catch {
            logDebug("SHARP: Failed to load model: \(error)")
            statusMessage = L10n.Sharp.couldNotGetReady
        }

        isLoadingModel = false
    }

    // MARK: - Main Generation API

    /// Generate 3D Gaussian splat from an image
    /// - Parameters:
    ///   - image: The source image (possibly downscaled for SHARP memory).
    ///   - sourceImageURL: Original photo file URL when known (library pick) — used to write `camera_exif.json` (focal + subject distance).
    ///   - photoLibraryAssetLocalId: `PHAsset.localIdentifier` when the image came from the library (EXIF when `imageURL` is nil).
    ///   - captureMediaMetadata: `UIImagePickerController.InfoKey.mediaMetadata` from in-app camera when available.
    /// - Returns: URL of the `_classic.ply` written by `writePLY` (Metal viewer frame).
    func generateGaussians(
        from image: UIImage,
        sourceImageURL: URL? = nil,
        captureMediaMetadata: [AnyHashable: Any]? = nil,
        photoLibraryAssetLocalId: String? = nil,
    ) async throws -> SHARPGenerationResult {
        logDebug("SHARP: Starting Gaussian generation")
        logMemorySnapshot("SHARPService.generateGaussians", details: "phase=start")

        // Load model on-demand if not already loaded
        if model == nil {
            logDebug("SHARP: Model not loaded yet, loading now...")
            await loadModel()
            logMemorySnapshot("SHARPService.generateGaussians", details: "phase=after_model_load")
        }

        guard model != nil else {
            status = .failed("SHARP model failed to load. Check device memory.")
            statusMessage = L10n.Sharp.somethingWentWrong
            throw GenerationError.serverError("SHARP model failed to load. Check device memory.")
        }

        // Reset state
        status = .processing
        statusMessage = L10n.Sharp.preparingPhoto
        progress = 0.0

        do {
            // Compute the small working image in a tight scope so the full-res originals
            // are released before Core ML prediction (saves ~100 MB on 4 GB devices).
            let preparedImage: (UIImage, CGSize) = autoreleasepool {
                let orientedSize = Self.orientedPixelSize(for: image)
                let prepared = Self.prepareImageForSharp(image)
                return (prepared, orientedSize)
            }
            var workingImage: UIImage? = preparedImage.0
            let orientedSize = preparedImage.1

            progress = 0.1
            statusMessage = L10n.Sharp.creatingRoom
            progress = 0.2
            logMemorySnapshot(
                "SHARPService.generateGaussians",
                details: "phase=before_inference working_px=\(Int((workingImage?.size.width ?? 0) * (workingImage?.scale ?? 1)))x\(Int((workingImage?.size.height ?? 0) * (workingImage?.scale ?? 1)))"
            )
            guard let workingImageForInference = workingImage else {
                throw GenerationError.invalidImage
            }
            var gaussianParams: [Float]? = try await preprocessAndRunInference(workingImage: workingImageForInference)
            let gaussianCount = (gaussianParams?.count ?? 0) / Self.paramsPerGaussian
            logDebug("SHARP: Generated \(gaussianCount) Gaussians")
            logMemorySnapshot(
                "SHARPService.generateGaussians",
                details: "phase=after_inference gaussian_count=\(gaussianCount)"
            )

            statusMessage = L10n.Sharp.almostDone
            progress = 0.8
            let plyURLs = try await writePLY(
                gaussianParams ?? [],
                sourceImageSize: orientedSize,
                applyAspectCorrection: true
            )
            gaussianParams?.removeAll(keepingCapacity: false)
            gaussianParams = nil
            logMemorySnapshot(
                "SHARPService.generateGaussians",
                details: "phase=after_write_ply ply=\(plyURLs.classic.lastPathComponent)"
            )
            logMemorySnapshot("SHARPService.generateGaussians", details: "phase=after_drop_gaussian_params")
            logSharpMilestone(
                "PLY written on Swift/Core ML path (not C++): \(plyURLs.classic.lastPathComponent) — Metal viewer loads this file",
            )
            logDebug("SHARP: Saved classic PLY to \(plyURLs.classic.path)")
            guard let workingImageForThumbnail = workingImage else {
                throw GenerationError.invalidImage
            }
            let thumbSource = autoreleasepool {
                Self.imageForWallMeasurementThumbnail(workingImageForThumbnail, maxPixel: 1024)
            }
            saveThumbnailForWallMeasurement(image: thumbSource, classicPlyURL: plyURLs.classic)
            workingImage = nil
            logMemorySnapshot("SHARPService.generateGaussians", details: "phase=after_drop_working_image")
            let roomFolder = plyURLs.classic.deletingLastPathComponent()
            await CameraExifSidecar.writeMerged(
                roomFolder: roomFolder,
                imageURL: sourceImageURL,
                mediaMetadata: captureMediaMetadata,
                photoLibraryAssetLocalId: photoLibraryAssetLocalId,
            )
            let plyURL = plyURLs.classic

            // Complete
            progress = 1.0
            status = .completed(fileURL: plyURL)
            statusMessage = L10n.Sharp.done

            releaseInferenceMemoryAfterGeneration()
            logMemorySnapshot("SHARPService.generateGaussians", details: "phase=after_release_inference_memory")

            return SHARPGenerationResult(
                plyURL: plyURL,
                plyAabbWidth: plyURLs.aabbWidth,
                plyAabbHeight: plyURLs.aabbHeight,
                plyAabbDepth: plyURLs.aabbDepth
            )
        } catch {
            // Update status on failure so UI can show error
            status = .failed(error.localizedDescription)
            statusMessage = L10n.Sharp.couldNotCreateRoom
            logSharpMilestone("generation failed: \(error.localizedDescription)")
            logDebug("SHARP: Generation failed: \(error)")
            releaseInferenceMemoryAfterGeneration()
            logMemorySnapshot("SHARPService.generateGaussians", details: "phase=failed_after_release")
            throw error
        }
    }

    /// Cancel current generation (if any)
    func cancelGeneration() {
        status = .failed(L10n.Sharp.cancelled)
        statusMessage = L10n.Sharp.cancelled
        releaseInferenceMemoryAfterGeneration()
    }

    // MARK: - Image Preprocessing

    /// Logs a few spatially separated BGRA samples so we can see whether color is already gone
    /// before SHARP inference or only later in the export/render path.
    private func logInputColorDiagnostics(_ buffer: CVPixelBuffer) {
        let width = CVPixelBufferGetWidth(buffer)
        let height = CVPixelBufferGetHeight(buffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
        guard width > 0, height > 0 else { return }

        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        guard let baseAddress = CVPixelBufferGetBaseAddress(buffer) else { return }

        let pointer = baseAddress.assumingMemoryBound(to: UInt8.self)
        let samplePoints: [(x: Int, y: Int)] = [
            (max(0, width / 16), max(0, height / 16)),
            (width / 2, height / 2),
            (max(0, width / 4), max(0, (height * 3) / 4)),
            (max(0, (width * 3) / 4), max(0, height / 3)),
            (max(0, width / 2), max(0, height / 4)),
        ]

        var minimumR: UInt8 = 255
        var minimumG: UInt8 = 255
        var minimumB: UInt8 = 255
        var maximumR: UInt8 = 0
        var maximumG: UInt8 = 0
        var maximumB: UInt8 = 0
        var grayscaleSampleCount = 0

        for (x, y) in samplePoints {
            let clampedX = min(max(x, 0), width - 1)
            let clampedY = min(max(y, 0), height - 1)
            let offset = clampedY * bytesPerRow + clampedX * 4
            let blue = pointer[offset + 0]
            let green = pointer[offset + 1]
            let red = pointer[offset + 2]
            let alpha = pointer[offset + 3]
            let channelSpread = max(abs(Int(red) - Int(green)), abs(Int(green) - Int(blue)))
            let looksGray = channelSpread < 5
            if looksGray { grayscaleSampleCount += 1 }

            minimumR = min(minimumR, red); minimumG = min(minimumG, green); minimumB = min(minimumB, blue)
            maximumR = max(maximumR, red); maximumG = max(maximumG, green); maximumB = max(maximumB, blue)

            logDebug(
                "  🎨 Pixel(\(clampedX),\(clampedY)) BGRA=(\(blue), \(green), \(red), \(alpha)) " +
                "rgb_spread=\(channelSpread) \(looksGray ? "GRAY?" : "COLOR")"
            )
        }

        logDebug(
            "  🎨 Channel ranges R[\(minimumR)-\(maximumR)] G[\(minimumG)-\(maximumG)] B[\(minimumB)-\(maximumB)] " +
            "gray_like_samples=\(grayscaleSampleCount)/\(samplePoints.count)"
        )
    }

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
        // CPU-only SHARP: omit Metal compatibility to avoid extra IOSurface / GPU allocator pressure.
        let attrs = [
            kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue!,
            kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue!
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

        // Draw resized image into pixel buffer (pool releases transient decode pressure from Core Graphics)
        autoreleasepool {
            context.interpolationQuality = .high
            context.draw(cgImage, in: CGRect(x: 0, y: 0, width: size, height: size))
        }

        return buffer
    }

    // MARK: - Model Inference

    /// Preprocess then infer in one scope so the 1536² `CVPixelBuffer` can be released before PLY allocation.
    private func preprocessAndRunInference(workingImage: UIImage) async throws -> [Float] {
        let inputBuffer = try await preprocessImage(workingImage)
        logDebug("SHARP: Image preprocessed to \(Self.inputSize)x\(Self.inputSize)")
        logInputColorDiagnostics(inputBuffer)
        return try await runInference(inputBuffer)
    }

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

        // Unlock before passing to CoreML - CoreML needs to manage its own buffer access
        CVPixelBufferUnlockBaseAddress(input, .readOnly)

        URLCache.shared.removeAllCachedResponses()

        // Create feature provider with CVPixelBuffer as Image type
        let imageFeature = MLFeatureValue(pixelBuffer: input)
        let inputFeatures = try MLDictionaryFeatureProvider(dictionary: ["image": imageFeature])

        logSharpMilestone(
            "Core ML prediction starting (cpuOnly — can take several minutes; heartbeat every 15s until this step completes)",
        )
        let t0 = CFAbsoluteTimeGetCurrent()
        // Core ML gives no callbacks mid-flight; detached heartbeat proves the app is alive in Console.
        let heartbeat = Task.detached(priority: .utility) {
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 15_000_000_000)
                guard !Task.isCancelled else { break }
                let elapsed = CFAbsoluteTimeGetCurrent() - t0
                logSharpMilestone(String(format: "Core ML prediction still running… %.0fs elapsed", elapsed))
            }
        }
        defer { heartbeat.cancel() }
        // Run prediction (mlprogram requires async in iOS 16+)
        let output = try await model.prediction(from: inputFeatures)
        let elapsed = CFAbsoluteTimeGetCurrent() - t0
        logSharpMilestone(String(format: "Core ML prediction finished in %.1fs — extracting outputs", elapsed))

        logDebug("SHARP: Inference complete, extracting outputs...")

        // Drain Core ML Obj-C temporaries while copying into Swift `[Float]` (reduces peak RAM on 4GB devices).
        return try autoreleasepool { () throws -> [Float] in
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
    }

    // MARK: - Splat Filtering

    /// Hard cutoff for almost-transparent splats (lower = keep more semi-transparent splats)
    /// Reduced from 0.30 to 0.08 to preserve wall/ceiling areas that were showing as black patches
    private static let minAlpha: Float = 0.08

    /// For "fluffy fog" filter: if alpha < this AND scale > fogScaleThreshold, drop it
    /// Reduced from 0.50 to 0.20 to keep more of the scene (only remove truly foggy splats)
    private static let fogAlphaThreshold: Float = 0.20
    private static let fogScaleThreshold: Float = 0.025  // Slightly higher threshold

    /// Edge margins (percentage of bbox to trim from edges)
    /// Reduced from 12% to 8% to keep more edge content
    private static let edgeMarginX: Float = 0.08  // 8% margin on X
    private static let edgeMarginY: Float = 0.08  // 8% margin on Y
    private static let edgeMarginZ: Float = 0.08  // 8% margin on Z (depth)

    /// Filter Gaussians by opacity and limit count for mobile rendering
    ///
    /// NOTE: Filtering is DISABLED as of Jan 2026. The opacity/fog/edge filters were too
    /// aggressive and removed valid wall/ceiling/corner gaussians, causing black patches
    /// in the rendered room. The SHARP model output is now used directly without filtering.
    private func filterGaussians(_ params: [Float]) -> [Float] {
        let inputCount = params.count / Self.paramsPerGaussian
        logDebug("SHARP: Processing \(inputCount) Gaussians (no filtering)")

        // Return all gaussians without filtering - filtering was causing black patches
        // by removing valid wall/ceiling splats that had lower opacity
        return params
    }

    // MARK: - PLY Writing

    /// Write Gaussian parameters to a single Metal-facing `_classic.ply` (uchar RGB, Y/Z flip for in-app viewer).
    private func writePLY(
        _ params: [Float],
        sourceImageSize: CGSize,
        applyAspectCorrection: Bool,
    ) async throws -> (classic: URL, aabbWidth: Float, aabbHeight: Float, aabbDepth: Float) {
        // Filter Gaussians for mobile rendering
        let filteredParams = filterGaussians(params)

        let gaussianCount = filteredParams.count / Self.paramsPerGaussian
        let aspectCorrection = applyAspectCorrection
            ? Self.sharpAspectCorrectionFactors(for: sourceImageSize)
            : (x: Float(1), y: Float(1))
        logDebug(
            "SHARP: PLY aspect correction image=\(Int(sourceImageSize.width))x\(Int(sourceImageSize.height)) " +
            "corr=(\(String(format: "%.4f", aspectCorrection.x)), \(String(format: "%.4f", aspectCorrection.y)))"
        )

        // Ensure output directory exists
        try FileManager.default.createDirectory(at: modelsDirectory, withIntermediateDirectories: true)

        // Generate filenames
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "yyyyMMdd_HHmmss"
        let timestamp = dateFormatter.string(from: Date())
        let classicFileName = "Room_\(timestamp)_classic.ply"
        let classicFileURL = modelsDirectory.appendingPathComponent(classicFileName)

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

        // SH coefficient constant: C0 = 0.5 * sqrt(1/pi) (debug logging)
        let SH_C0: Float = 0.28209479177387814

        // Per-vertex byte size: 11 floats (44 B) + 3 UInt8 (3 B) = 47 B
        let standardVertexBytes = 47

        // Pre-compute per-gaussian values for classic PLY.
        // Stored in a flat struct-of-arrays to avoid a second pass over `filteredParams`.
        struct GaussianRow {
            var x, y, z: Float             // transformed position
            var s0, s1, s2: Float          // log-scale
            var r0, r1, r2, r3: Float      // normalised quaternion
            var opacity: Float             // logit
            var rawR, rawG, rawB: Float    // linear colour [0,1]
        }

        var rows = [GaussianRow]()
        rows.reserveCapacity(gaussianCount)

        let minScale: Float = 0.001
        let scaleBoost: Float = 1.3

        for i in 0..<gaussianCount {
            let o = i * Self.paramsPerGaussian
            let x = filteredParams[o + 0] * aspectCorrection.x
            let y = -(filteredParams[o + 1] * aspectCorrection.y)
            let z = -filteredParams[o + 2]

            let rr0 = filteredParams[o + 6]
            let rr1 = filteredParams[o + 7]
            let rr2 = filteredParams[o + 8]
            let rr3 = filteredParams[o + 9]
            let mag = sqrt(rr0*rr0 + rr1*rr1 + rr2*rr2 + rr3*rr3)
            let inv = mag > 1e-8 ? 1.0 / mag : 1.0

            let rawOp = filteredParams[o + 10]
            let clOp = min(max(rawOp, 1e-4), 1.0 - 1e-4)

            rows.append(GaussianRow(
                x: x, y: y, z: z,
                s0: log(max(filteredParams[o + 3] * scaleBoost, minScale)),
                s1: log(max(filteredParams[o + 4] * scaleBoost, minScale)),
                s2: log(max(filteredParams[o + 5] * scaleBoost, minScale)),
                r0: rr0 * inv, r1: rr1 * inv, r2: rr2 * inv, r3: rr3 * inv,
                opacity: log(clOp / (1.0 - clOp)),
                rawR: filteredParams[o + 11],
                rawG: filteredParams[o + 12],
                rawB: filteredParams[o + 13]
            ))
        }

        // AABB in **`_classic.ply` file space** (x, -y, -z) — matches MetalSplatter load path for SHARP classic.
        var clMinX: Float = .greatestFiniteMagnitude
        var clMaxX: Float = -.greatestFiniteMagnitude
        var clMinY: Float = .greatestFiniteMagnitude
        var clMaxY: Float = -.greatestFiniteMagnitude
        var clMinZ: Float = .greatestFiniteMagnitude
        var clMaxZ: Float = -.greatestFiniteMagnitude
        for g in rows {
            let cx = g.x
            let cy = -g.y
            let cz = -g.z
            clMinX = min(clMinX, cx); clMaxX = max(clMaxX, cx)
            clMinY = min(clMinY, cy); clMaxY = max(clMaxY, cy)
            clMinZ = min(clMinZ, cz); clMaxZ = max(clMaxZ, cz)
        }
        let width = clMaxX - clMinX
        let height = clMaxY - clMinY
        let depth = clMaxZ - clMinZ

        // Debug first few gaussians
        for i in 0..<min(3, rows.count) {
            let g = rows[i]
            let quatLen = sqrt(g.r0*g.r0 + g.r1*g.r1 + g.r2*g.r2 + g.r3*g.r3)
            let f0 = (g.rawR - 0.5) / SH_C0
            let f1 = (g.rawG - 0.5) / SH_C0
            let f2 = (g.rawB - 0.5) / SH_C0
            let gamma: Float = 1.0 / 2.2; let bright: Float = 1.1
            let red = UInt8(min(max(Int(pow(min(max(g.rawR * bright, 0), 1), gamma) * 255), 0), 255))
            let green = UInt8(min(max(Int(pow(min(max(g.rawG * bright, 0), 1), gamma) * 255), 0), 255))
            let blue = UInt8(min(max(Int(pow(min(max(g.rawB * bright, 0), 1), gamma) * 255), 0), 255))
            logDebug("SHARP PLY GAUSSIAN \(i) (final values):")
            logDebug("  pos: (\(g.x), \(g.y), \(g.z))")
            logDebug("  scale (log): (\(g.s0), \(g.s1), \(g.s2))")
            logDebug("  rot: (\(g.r0), \(g.r1), \(g.r2), \(g.r3)) len=\(quatLen)")
            logDebug("  opacity (logit): \(g.opacity)")
            logDebug("  f_dc (SH): (\(f0), \(f1), \(f2))")
            logDebug("  color (uchar): (\(red), \(green), \(blue))")
        }

        // Helper: write uchar-RGB PLY with optional Y/Z flip (classic uses flip for Metal).
        func writeStandardPLY(toURL url: URL, flipYZ: Bool) throws {
            let headerData = Data(plyContent.utf8)
            var buf = Data(capacity: headerData.count + gaussianCount * standardVertexBytes)
            buf.append(headerData)
            let gamma: Float = 1.0 / 2.2
            let brightness: Float = 1.1
            for g in rows {
                var px = g.x; var py = flipYZ ? -g.y : g.y; var pz = flipYZ ? -g.z : g.z
                buf.append(Data(bytes: &px, count: 4))
                buf.append(Data(bytes: &py, count: 4))
                buf.append(Data(bytes: &pz, count: 4))
                var s0 = g.s0; var s1 = g.s1; var s2 = g.s2
                buf.append(Data(bytes: &s0, count: 4))
                buf.append(Data(bytes: &s1, count: 4))
                buf.append(Data(bytes: &s2, count: 4))
                var r0 = g.r0; var r1 = g.r1; var r2 = g.r2; var r3 = g.r3
                buf.append(Data(bytes: &r0, count: 4))
                buf.append(Data(bytes: &r1, count: 4))
                buf.append(Data(bytes: &r2, count: 4))
                buf.append(Data(bytes: &r3, count: 4))
                var op = g.opacity
                buf.append(Data(bytes: &op, count: 4))
                let fR = pow(min(max(g.rawR * brightness, 0), 1), gamma)
                let fG = pow(min(max(g.rawG * brightness, 0), 1), gamma)
                let fB = pow(min(max(g.rawB * brightness, 0), 1), gamma)
                var red = UInt8(min(max(Int(fR * 255), 0), 255))
                var green = UInt8(min(max(Int(fG * 255), 0), 255))
                var blue = UInt8(min(max(Int(fB * 255), 0), 255))
                buf.append(Data(bytes: &red, count: 1))
                buf.append(Data(bytes: &green, count: 1))
                buf.append(Data(bytes: &blue, count: 1))
            }
            try buf.write(to: url)
        }

        // Classic PLY only: (x, -y, -z) via flipYZ — matches MetalSplatter / GaussianSplatView.
        try writeStandardPLY(toURL: classicFileURL, flipYZ: true)

        let attributes = try FileManager.default.attributesOfItem(atPath: classicFileURL.path)
        let classicSize = attributes[.size] as? UInt64 ?? 0
        logDebug("SHARP: Classic PLY saved (\(classicSize / 1024) KB) — only export (Metal uchar RGB)")

        logPlyBoundsDiagnostic(
            "SHARP classic_ply room bounds (vertex frame written to *_classic.ply; in-app Metal uses this file): " +
            "W=\(String(format: "%.3f", width)) H=\(String(format: "%.3f", height)) D=\(String(format: "%.3f", depth)) " +
            "X[\(String(format: "%.3f", clMinX)),\(String(format: "%.3f", clMaxX))] " +
            "Y[\(String(format: "%.3f", clMinY)),\(String(format: "%.3f", clMaxY))] " +
            "Z[\(String(format: "%.3f", clMinZ)),\(String(format: "%.3f", clMaxZ))] splats=\(gaussianCount)"
        )

        let firstRowXYZ: String = {
            guard let r = rows.first else { return "n/a" }
            return String(format: "%.6f,%.6f,%.6f", r.x, r.y, r.z)
        }()
        logDebug(
            "[ROOM_CREATE_COMPARE] phase=sharp_ply_export " +
            "inputSquare=\(Self.inputSize) sourceImage_px=\(Int(sourceImageSize.width))x\(Int(sourceImageSize.height)) " +
            "applyAspectCorrection=\(applyAspectCorrection) " +
            "aspect_xy=(\(String(format: "%.6f", aspectCorrection.x)),\(String(format: "%.6f", aspectCorrection.y))) " +
            "classic_aabb_vertexFrame_su_whd=(\(String(format: "%.6f", width)),\(String(format: "%.6f", height)),\(String(format: "%.6f", depth))) " +
            "classic_aabb_min_xyz=(\(String(format: "%.6f", clMinX)),\(String(format: "%.6f", clMinY)),\(String(format: "%.6f", clMinZ))) " +
            "classic_aabb_max_xyz=(\(String(format: "%.6f", clMaxX)),\(String(format: "%.6f", clMaxY)),\(String(format: "%.6f", clMaxZ))) " +
            "first_gaussian_row_xyz_pre_writePLYFlip=(\(firstRowXYZ)) splats=\(gaussianCount) file=\(classicFileName)"
        )

        rows.removeAll(keepingCapacity: false)
        logMemorySnapshot("SHARPService.writePLY", details: "phase=after_drop_rows ply=\(classicFileName) splats=\(gaussianCount)")

        return (classic: classicFileURL, aabbWidth: width, aabbHeight: height, aabbDepth: depth)
    }

    /// Writes `Room_<stamp>_thumbnail.jpg` next to the classic PLY so [WallMeasurementEstimator] can load it on save.
    /// JPEG uses ~1/3 the encoding memory of PNG for photo-type images.
    private func saveThumbnailForWallMeasurement(image: UIImage, classicPlyURL: URL) {
        let stem = classicPlyURL.deletingPathExtension().lastPathComponent
        let base: String
        if stem.hasSuffix("_classic") {
            base = String(stem.dropLast("_classic".count))
        } else {
            base = stem
        }
        let out = classicPlyURL.deletingLastPathComponent().appendingPathComponent("\(base)_thumbnail.jpg")
        guard let data = image.jpegData(compressionQuality: 0.85) else {
            logWallMeasurement("persist_thumbnail fail encode path=\(out.path)")
            return
        }
        do {
            try data.write(to: out)
            logWallMeasurement("persist_thumbnail ok path=\(out.path) bytes=\(data.count)")
        } catch {
            logWallMeasurement("persist_thumbnail fail write \(error.localizedDescription) path=\(out.path)")
        }
    }
}
