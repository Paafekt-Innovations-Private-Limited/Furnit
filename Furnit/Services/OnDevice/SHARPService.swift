import Foundation
import UIKit
import CoreML
import simd

/// App sandbox temp directory for unsaved Sharp previews (`Room_*.ply`, sidecars, thumbnails). Not `Documents/SavedRooms`.
private func sharpModelsTemporaryDirectoryURL() -> URL {
    FileManager.default.temporaryDirectory.appendingPathComponent("SHARPModels", isDirectory: true)
}

/// PLY URL plus axis-aligned bounds in **scene units** for the freshly-created preview PLY.
struct SHARPGenerationResult: Sendable {
    let plyURL: URL
    let plyAabbWidth: Float
    let plyAabbHeight: Float
    let plyAabbDepth: Float
    let roomWidth: Float?
    let roomHeight: Float?
    let roomDepth: Float?
}

/// On-device 3D Gaussian generation using Apple's SHARP model.
/// Runs local Core ML inference and writes temporary preview files without uploading room photos.
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

    /// 3DGS export uses a slightly larger linear scale so external viewers match expected splat size.
    private static let threeDGSMetalViewerLinearScaleFactor: Float = 1.65

    /// Gaussian parameter count per splat: pos(3) + scale(3) + rot(4) + opacity(1) + sh(3) = 14
    private static let paramsPerGaussian: Int = 14

    /// Returns the original image for SHARP. We no longer materialize a second full-resolution
    /// upright copy here because that extra 4K/5K allocation is enough to tip 4 GB devices
    /// into jetsam during the long Core ML prediction. Orientation is applied directly while
    /// drawing into the 1536×1536 model input buffer inside `preprocessImage`.
    static func prepareImageForSharp(_ image: UIImage) -> UIImage {
        let orientedSize = orientedPixelSize(for: image)
        let w = orientedSize.width
        let h = orientedSize.height
        guard w > 1, h > 1 else { return image }

        logDebug(
            "[GREEN][SHARP_PREPROCESS] SOURCE=\(Int(w))X\(Int(h)) ORIENT=\(image.imageOrientation.rawValue) " +
            "PREPARED=\(Int(w))X\(Int(h)) MODE=DEFER_ORIENTATION_TO_MODEL_INPUT"
        )
        return image
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

    /// Check if SHARP model resources are available locally
    func checkResourceAvailability() async {
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
    @Published var isBackgroundGenerationActive: Bool = false

    var hasActiveSharpWork: Bool {
        if isDownloadingResources || isLoadingModel { return true }
        if case .processing = status { return true }
        return false
    }

    var unifiedProgress: Double {
        if isDownloadingResources { return min(max(downloadProgress, 0.0), 1.0) }
        return min(max(Double(progress), 0.0), 1.0)
    }

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

        // Check physical memory. Keep only a very low-end guard here so devices that previously
        // worked (for example 4 GB-class phones) are not blocked from testing. The real fix for
        // memory pressure is reducing peak allocations, not an aggressive RAM gate.
        let physicalMemory = ProcessInfo.processInfo.physicalMemory
        let minimumSupportedRam: UInt64 = 2 * 1024 * 1024 * 1024

        logDebug("SHARP: Device RAM: \(physicalMemory / 1024 / 1024)MB")

        if physicalMemory < minimumSupportedRam {
            logDebug("SHARP: Device has <2GB RAM — skipping on-device SHARP to avoid memory termination during inference")
            isLoadingModel = false
            statusMessage = L10n.Sharp.notEnoughSpace
            return
        }

        // Best-effort ODR download. Debug/Xcode installs still need this now that the model
        // lives in tagged asset packs instead of the app bundle.
        if !resourcesAvailable {
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
                logDebug("SHARP: ODR download failed (\(error)) — trying app-bundle fallback")
            }
        }

        progress = 0.3
        statusMessage = L10n.Sharp.settingThingsUp

        do {
            // CPU-only: avoids GPU IOSurface / Neural Engine allocation failures on constrained devices.
            let config = MLModelConfiguration()
            config.computeUnits = .cpuOnly

            logDebug("SHARP: Loading bundled/ODR model dynamically - FP32, computeUnits=cpuOnly")
            let candidateModels = [
                ("SHARP_fp32_1536", "mlmodelc"),
                ("SHARP_fp32_1536", "mlpackage"),
            ]

            var loadedModel: MLModel?
            for (name, ext) in candidateModels {
                guard let url = Bundle.main.url(forResource: name, withExtension: ext) else { continue }
                let loadURL: URL
                if ext == "mlpackage" {
                    loadURL = try await MLModel.compileModel(at: url)
                } else {
                    loadURL = url
                }
                loadedModel = try MLModel(contentsOf: loadURL, configuration: config)
                logDebug("SHARP: Loaded '\(name).\(ext)' successfully")
                break
            }

            if Task.isCancelled {
                isLoadingModel = false
                return
            }

            guard let loadedModel else {
                throw GenerationError.serverError("SHARP model resource is missing")
            }

            model = loadedModel
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
    ///   - supplementalCameraDoubles: e.g. ARKit intrinsics from ``ARRoomPhotoCaptureViewController`` (merged into `camera_exif.json`).
    /// - Returns: URL of the `_classic.ply` written by `writePLY` for the fresh preview.
    func generateGaussians(
        from image: UIImage,
        sourceImageURL: URL? = nil,
        captureMediaMetadata: [AnyHashable: Any]? = nil,
        photoLibraryAssetLocalId: String? = nil,
        supplementalCameraDoubles: [String: Double]? = nil,
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
            status = .failed("SHARP model failed to load. Check model download, storage, or device memory.")
            statusMessage = L10n.Sharp.somethingWentWrong
            throw GenerationError.serverError("SHARP model failed to load. Check model download, storage, or device memory.")
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
            let mergedExif = await CameraExifSidecar.collectMerged(
                imageURL: sourceImageURL,
                mediaMetadata: captureMediaMetadata,
                photoLibraryAssetLocalId: photoLibraryAssetLocalId,
                supplementalDoubles: supplementalCameraDoubles
            )
            let sharpCamera = SharpCameraSidecar.infoIfPossible(
                sourceImageSize: orientedSize,
                inputSquarePx: Self.inputSize,
                exif: mergedExif
            )

            let plyURLs = try await writePLY(
                gaussianParams ?? [],
                sourceImageSize: orientedSize,
                applyAspectCorrection: false,
                sharpCamera: sharpCamera
            )
            let plyURL = plyURLs.classic
            gaussianParams?.removeAll(keepingCapacity: false)
            gaussianParams = nil
            logMemorySnapshot(
                "SHARPService.generateGaussians",
                details: "phase=after_write_ply ply=\(plyURL.lastPathComponent)"
            )
            logMemorySnapshot("SHARPService.generateGaussians", details: "phase=after_drop_gaussian_params")
            logSharpMilestone(
                "PLY written on Swift/Core ML path (not C++): \(plyURL.lastPathComponent) — fresh preview loads classic PLY",
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
            logWallMeasurement(
                "camera_exif_sidecar request folder=\(roomFolder.lastPathComponent) " +
                    "source_url=\(sourceImageURL?.lastPathComponent ?? "nil") " +
                    "has_media_metadata=\(captureMediaMetadata != nil) " +
                    "phasset_id=\(photoLibraryAssetLocalId ?? "nil")"
            )
            if !mergedExif.isEmpty {
                CameraExifSidecar.mergeDerivedValues(roomFolder: roomFolder, additions: mergedExif)
            } else {
                await CameraExifSidecar.writeMerged(
                    roomFolder: roomFolder,
                    imageURL: sourceImageURL,
                    mediaMetadata: captureMediaMetadata,
                    photoLibraryAssetLocalId: photoLibraryAssetLocalId,
                    supplementalDoubles: supplementalCameraDoubles,
                )
            }
            let exif = mergedExif.isEmpty ? CameraExifSidecar.load(roomFolder: roomFolder) : mergedExif
            SharpCameraSidecar.writeIfPossible(
                roomURL: plyURLs.classic,
                sourceImageSize: orientedSize,
                inputSquarePx: Self.inputSize,
                exif: exif
            )

            // Complete
            progress = 1.0
            status = .completed(fileURL: plyURL)
            statusMessage = L10n.Sharp.done

            releaseInferenceMemoryAfterGeneration()
            logMemorySnapshot("SHARPService.generateGaussians", details: "phase=after_release_inference_memory")
            logDepthPro("AUTO RUN DEFERRED AFTER SHARP GENERATION TO AVOID POST-GENERATION MEMORY SPIKE")

            return SHARPGenerationResult(
                plyURL: plyURL,
                plyAabbWidth: plyURLs.aabbWidth,
                plyAabbHeight: plyURLs.aabbHeight,
                plyAabbDepth: plyURLs.aabbDepth,
                roomWidth: plyURLs.roomWidth,
                roomHeight: plyURLs.roomHeight,
                roomDepth: plyURLs.roomDepth
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
        isBackgroundGenerationActive = false
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

        // Normalize EXIF orientation into a 1536×1536 intermediate only. This preserves the
        // lower-peak-memory path (no second full-resolution upright image) while keeping SHARP's
        // input upright like the old working path.
        let modelRect = CGRect(x: 0, y: 0, width: size, height: size)
        guard let orientedModelCGImage = autoreleasepool(invoking: { () -> CGImage? in
            let format = UIGraphicsImageRendererFormat()
            format.scale = 1
            format.opaque = false
            let renderer = UIGraphicsImageRenderer(size: CGSize(width: size, height: size), format: format)
            let rendered = renderer.image { _ in
                image.draw(in: modelRect)
            }
            return rendered.cgImage
        }) else {
            throw GenerationError.serverError("Failed to normalize image orientation for SHARP input")
        }

        autoreleasepool {
            context.interpolationQuality = .high
            context.draw(orientedModelCGImage, in: modelRect)
        }

        let orientedSize = Self.orientedPixelSize(for: image)
        logDebug(
            "[GREEN][SHARP_PREPROCESS] SOURCE_DRAW=\(Int(orientedSize.width))X\(Int(orientedSize.height)) " +
            "MODEL_INPUT=\(size)X\(size) RESIZE=DIRECT_SQUISH PAD=NONE ORIENT=RENDERED_AT_MODEL_INPUT"
        )

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

    /// Write Gaussian parameters to a single `_classic.ply` variant used by the in-app Metal renderer.
    private func writePLY(
        _ params: [Float],
        sourceImageSize: CGSize,
        applyAspectCorrection: Bool,
        sharpCamera: SharpCameraSidecar.Info?,
    ) async throws -> (
        classic: URL,
        aabbWidth: Float,
        aabbHeight: Float,
        aabbDepth: Float,
        roomWidth: Float?,
        roomHeight: Float?,
        roomDepth: Float?
    ) {
        // Filter Gaussians for mobile rendering
        let filteredParams = filterGaussians(params)

        let gaussianCount = filteredParams.count / Self.paramsPerGaussian
        let aspectCorrection = applyAspectCorrection
            ? Self.sharpAspectCorrectionFactors(for: sourceImageSize)
            : (x: Float(1), y: Float(1))
        let unprojectionFocalPx: Float = {
            guard let sharpCamera else { return 0 }
            // Unprojection is done in source-image coordinates, so it must use the
            // EXIF-derived focal scaled to the same source image size.
            return sharpCamera.sourceFocalPx
        }()
        let resizedFxPx: Float = {
            guard let sharpCamera, unprojectionFocalPx > 0.01 else { return 0 }
            return unprojectionFocalPx * (Float(sharpCamera.inputSquarePx) / Float(sharpCamera.sourceImageWidthPx))
        }()
        let resizedFyPx: Float = {
            guard let sharpCamera, unprojectionFocalPx > 0.01 else { return 0 }
            return unprojectionFocalPx * (Float(sharpCamera.inputSquarePx) / Float(sharpCamera.sourceImageHeightPx))
        }()
        let unprojectionScaleX: Float = {
            guard let sharpCamera, resizedFxPx > 0.01 else { return 1 }
            return Float(sharpCamera.inputSquarePx) / (2 * resizedFxPx)
        }()
        let unprojectionScaleY: Float = {
            guard let sharpCamera, resizedFyPx > 0.01 else { return 1 }
            return Float(sharpCamera.inputSquarePx) / (2 * resizedFyPx)
        }()
        logDebug(
            "[GREEN][SHARP_UNPROJECT] IMAGE=\(Int(sourceImageSize.width))X\(Int(sourceImageSize.height)) " +
            "corr=(\(String(format: "%.4f", aspectCorrection.x)), \(String(format: "%.4f", aspectCorrection.y))) " +
            "FOCAL_PX=\(String(format: "%.2f", unprojectionFocalPx)) " +
            "FX_FY_RESIZED=(\(String(format: "%.2f", resizedFxPx)),\(String(format: "%.2f", resizedFyPx))) " +
            "UNPROJECT_XY=(\(String(format: "%.4f", unprojectionScaleX)),\(String(format: "%.4f", unprojectionScaleY)))"
        )

        // Ensure output directory exists
        try FileManager.default.createDirectory(at: modelsDirectory, withIntermediateDirectories: true)

        // Generate filenames
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "yyyyMMdd_HHmmss"
        let timestamp = dateFormatter.string(from: Date())
        let classicFileName = "Room_\(timestamp)_classic.ply"
        let classicFileURL = modelsDirectory.appendingPathComponent(classicFileName)

        let plyHeaderPrefix = """
        ply
        format binary_little_endian 1.0

        """

        let classicPlyContent = """
        \(plyHeaderPrefix)element vertex \(gaussianCount)
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

        // Per-vertex byte size: 11 floats (44 B) + 3 UInt8 (3 B) = 47 B
        let standardVertexBytes = 47

        func matrixGet(_ m: simd_float3x3, _ row: Int, _ col: Int) -> Float {
            m[col][row]
        }

        func matrixSet(_ m: inout simd_float3x3, _ row: Int, _ col: Int, _ value: Float) {
            m[col][row] = value
        }

        func rotationMatrixFromQuaternion(wxyz q: SIMD4<Float>) -> simd_float3x3 {
            let norm = simd_length(q)
            let nq = norm > 1e-8 ? q / norm : SIMD4<Float>(1, 0, 0, 0)
            let w = nq.x
            let x = nq.y
            let y = nq.z
            let z = nq.w

            let r0 = SIMD3<Float>(1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y))
            let r1 = SIMD3<Float>(2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x))
            let r2 = SIMD3<Float>(2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y))
            return simd_float3x3(
                columns: (
                    SIMD3<Float>(r0.x, r1.x, r2.x),
                    SIMD3<Float>(r0.y, r1.y, r2.y),
                    SIMD3<Float>(r0.z, r1.z, r2.z)
                )
            )
        }

        func quaternionFromRotationMatrix(_ m: simd_float3x3) -> SIMD4<Float> {
            let trace = matrixGet(m, 0, 0) + matrixGet(m, 1, 1) + matrixGet(m, 2, 2)
            let q: SIMD4<Float>
            if trace > 0 {
                let s = sqrt(max(trace + 1, 1e-8)) * 2
                q = SIMD4<Float>(
                    0.25 * s,
                    (matrixGet(m, 2, 1) - matrixGet(m, 1, 2)) / s,
                    (matrixGet(m, 0, 2) - matrixGet(m, 2, 0)) / s,
                    (matrixGet(m, 1, 0) - matrixGet(m, 0, 1)) / s
                )
            } else if matrixGet(m, 0, 0) > matrixGet(m, 1, 1), matrixGet(m, 0, 0) > matrixGet(m, 2, 2) {
                let s = sqrt(max(1 + matrixGet(m, 0, 0) - matrixGet(m, 1, 1) - matrixGet(m, 2, 2), 1e-8)) * 2
                q = SIMD4<Float>(
                    (matrixGet(m, 2, 1) - matrixGet(m, 1, 2)) / s,
                    0.25 * s,
                    (matrixGet(m, 0, 1) + matrixGet(m, 1, 0)) / s,
                    (matrixGet(m, 0, 2) + matrixGet(m, 2, 0)) / s
                )
            } else if matrixGet(m, 1, 1) > matrixGet(m, 2, 2) {
                let s = sqrt(max(1 + matrixGet(m, 1, 1) - matrixGet(m, 0, 0) - matrixGet(m, 2, 2), 1e-8)) * 2
                q = SIMD4<Float>(
                    (matrixGet(m, 0, 2) - matrixGet(m, 2, 0)) / s,
                    (matrixGet(m, 0, 1) + matrixGet(m, 1, 0)) / s,
                    0.25 * s,
                    (matrixGet(m, 1, 2) + matrixGet(m, 2, 1)) / s
                )
            } else {
                let s = sqrt(max(1 + matrixGet(m, 2, 2) - matrixGet(m, 0, 0) - matrixGet(m, 1, 1), 1e-8)) * 2
                q = SIMD4<Float>(
                    (matrixGet(m, 1, 0) - matrixGet(m, 0, 1)) / s,
                    (matrixGet(m, 0, 2) + matrixGet(m, 2, 0)) / s,
                    (matrixGet(m, 1, 2) + matrixGet(m, 2, 1)) / s,
                    0.25 * s
                )
            }
            let norm = simd_length(q)
            return norm > 1e-8 ? q / norm : SIMD4<Float>(1, 0, 0, 0)
        }

        func jacobiEigenDecompositionSymmetric(_ input: simd_float3x3) -> (values: SIMD3<Float>, vectors: simd_float3x3) {
            var a = input
            var v = matrix_identity_float3x3

            func rotate(_ p: Int, _ q: Int) {
                let app = matrixGet(a, p, p)
                let aqq = matrixGet(a, q, q)
                let apq = matrixGet(a, p, q)
                if abs(apq) < 1e-7 { return }
                let phi = 0.5 * atan2(2 * apq, aqq - app)
                let c = cos(phi)
                let s = sin(phi)

                for r in 0..<3 where r != p && r != q {
                    let arp = matrixGet(a, r, p)
                    let arq = matrixGet(a, r, q)
                    matrixSet(&a, r, p, c * arp - s * arq)
                    matrixSet(&a, p, r, matrixGet(a, r, p))
                    matrixSet(&a, r, q, s * arp + c * arq)
                    matrixSet(&a, q, r, matrixGet(a, r, q))
                }

                let newApp = c * c * app - 2 * s * c * apq + s * s * aqq
                let newAqq = s * s * app + 2 * s * c * apq + c * c * aqq
                matrixSet(&a, p, p, newApp)
                matrixSet(&a, q, q, newAqq)
                matrixSet(&a, p, q, 0)
                matrixSet(&a, q, p, 0)

                for r in 0..<3 {
                    let vrp = matrixGet(v, r, p)
                    let vrq = matrixGet(v, r, q)
                    matrixSet(&v, r, p, c * vrp - s * vrq)
                    matrixSet(&v, r, q, s * vrp + c * vrq)
                }
            }

            for _ in 0..<12 {
                let candidates: [(Float, Int, Int)] = [
                    (abs(matrixGet(a, 0, 1)), 0, 1),
                    (abs(matrixGet(a, 0, 2)), 0, 2),
                    (abs(matrixGet(a, 1, 2)), 1, 2),
                ]
                guard let pivot = candidates.max(by: { $0.0 < $1.0 }), pivot.0 > 1e-7 else { break }
                rotate(pivot.1, pivot.2)
            }

            var values = [matrixGet(a, 0, 0), matrixGet(a, 1, 1), matrixGet(a, 2, 2)]
            var vectors = [
                SIMD3<Float>(matrixGet(v, 0, 0), matrixGet(v, 1, 0), matrixGet(v, 2, 0)),
                SIMD3<Float>(matrixGet(v, 0, 1), matrixGet(v, 1, 1), matrixGet(v, 2, 1)),
                SIMD3<Float>(matrixGet(v, 0, 2), matrixGet(v, 1, 2), matrixGet(v, 2, 2)),
            ]
            let order = (0..<3).sorted { values[$0] > values[$1] }
            values = order.map { values[$0] }
            vectors = order.map { vectors[$0] }

            var rotation = simd_float3x3(columns: (vectors[0], vectors[1], vectors[2]))
            if simd_determinant(rotation) < 0 {
                rotation.columns.2 *= -1
            }
            return (
                SIMD3<Float>(max(values[0], 1e-8), max(values[1], 1e-8), max(values[2], 1e-8)),
                rotation
            )
        }

        func transformScalesAndRotation(
            scales linearScales: SIMD3<Float>,
            quaternion wxyz: SIMD4<Float>
        ) -> (scales: SIMD3<Float>, quaternion: SIMD4<Float>) {
            let transform = simd_float3x3(diagonal: SIMD3<Float>(unprojectionScaleX, unprojectionScaleY, 1))
            if abs(unprojectionScaleX - 1) < 1e-6, abs(unprojectionScaleY - 1) < 1e-6 {
                return (linearScales, wxyz)
            }
            let rotation = rotationMatrixFromQuaternion(wxyz: wxyz)
            let diagonal = simd_float3x3(diagonal: linearScales * linearScales)
            let covariance = rotation * diagonal * rotation.transpose
            let transformed = transform * covariance * transform.transpose
            let decomposition = jacobiEigenDecompositionSymmetric(transformed)
            let singularValues = SIMD3<Float>(
                sqrt(decomposition.values.x),
                sqrt(decomposition.values.y),
                sqrt(decomposition.values.z)
            )
            let quaternion = quaternionFromRotationMatrix(decomposition.vectors)
            return (singularValues, quaternion)
        }

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
            let x = filteredParams[o + 0] * aspectCorrection.x * unprojectionScaleX
            let y = -(filteredParams[o + 1] * aspectCorrection.y * unprojectionScaleY)
            let z = -filteredParams[o + 2]

            let rr0 = filteredParams[o + 6]
            let rr1 = filteredParams[o + 7]
            let rr2 = filteredParams[o + 8]
            let rr3 = filteredParams[o + 9]
            let mag = sqrt(rr0*rr0 + rr1*rr1 + rr2*rr2 + rr3*rr3)
            let inv = mag > 1e-8 ? 1.0 / mag : 1.0

            let rawOp = filteredParams[o + 10]
            let clOp = min(max(rawOp, 1e-4), 1.0 - 1e-4)

            let transformed = transformScalesAndRotation(
                scales: SIMD3<Float>(
                    max(filteredParams[o + 3] * scaleBoost, minScale),
                    max(filteredParams[o + 4] * scaleBoost, minScale),
                    max(filteredParams[o + 5] * scaleBoost, minScale)
                ),
                quaternion: SIMD4<Float>(rr0 * inv, rr1 * inv, rr2 * inv, rr3 * inv)
            )

            rows.append(GaussianRow(
                x: x, y: y, z: z,
                s0: log(max(transformed.scales.x, minScale)),
                s1: log(max(transformed.scales.y, minScale)),
                s2: log(max(transformed.scales.z, minScale)),
                r0: transformed.quaternion.x,
                r1: transformed.quaternion.y,
                r2: transformed.quaternion.z,
                r3: transformed.quaternion.w,
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

        struct BackWallDimensions {
            let zMode: Float
            let zMedian: Float
            let zMean: Float
            let floorDiagonal: Float
            let trimmedXSpan: Float
            let trimmedYSpan: Float
            let trimmedZSpan: Float
            let rawWidth: Float
            let rawHeight: Float
            let legacyWidth: Float
            let legacyHeight: Float
            let legacyDepth: Float
            let width: Float
            let height: Float
            let depth: Float
            let count: Int
            let band: Float
        }

        // Measurement approach guide:
        // - ROOM_DIMS: legacy back-wall percentile baseline kept for comparison logs only.
        // - ROOM_DIMS_APP: deferred app-bound path; V7 is now consumed asynchronously in SharpRoomView.
        // - ROOM_DIMS_V3: floor-plane RANSAC + PCA diagnostics used by V7 tilt / PCA inputs.
        // - ROOM_DIMS_V4: scene-extension correction reference, not wired to UI.
        // - ROOM_DIMS_V5: adaptive scene-extension reference, not wired to UI.
        // - ROOM_DIMS_V6: auto corner/straight reference, not wired to UI.
        // - ROOM_DIMS_V7: active production candidate used by the async preview/save measurement path.
        //
        // Non-active approaches are intentionally left isolated below so they can be deleted
        // later without touching the app-bound measurement path.

        // Experimental comparison-only path. Keep isolated so it is easy to remove if it
        // does not outperform the current ROOM_DIMS_APP approach.
        struct FloorAlignedPCADimensions {
            let floorNormal: SIMD3<Float>
            let tiltDegrees: Float
            let tiltRawDegrees: Float
            let tiltReliable: Bool
            let sampleCount: Int
            let ransacIterations: Int
            let ransacInliers: Int
            let epsilon: Float
            let lambda1: Float
            let lambda2: Float
            let span1: Float
            let span2: Float
            let width: Float
            let height: Float
            let depth: Float
        }

        // Experimental comparison-only path based on SHARP's own scene-extension math.
        // Kept separate so it can be removed without touching ROOM_DIMS or ROOM_DIMS_APP.
        struct SceneExtensionDimensions {
            let usedFocal: Bool
            let approach: String
            let focalPx: Float
            let imageWidth: Float
            let imageHeight: Float
            let fovDiagonal: Float
            let minDepth: Float
            let maxLateral: Float
            let sceneExtension: Float
            let rawWidth: Float
            let rawHeight: Float
            let width: Float
            let height: Float
            let depth: Float
        }

        // Experimental comparison-only adaptive variant of V4.
        // Kept fully separate so removal is trivial if the comparison is not useful.
        struct AdaptiveSceneExtensionDimensions {
            let usedFocal: Bool
            let approach: String
            let focalPx: Float
            let imageWidth: Float
            let imageHeight: Float
            let fovDiagonal: Float
            let minDepth: Float
            let backWallZ: Float
            let maxLateral: Float
            let sceneExtension: Float
            let fovWidthAtBackWall: Float
            let fovHeightAtBackWall: Float
            let fillRatioWidth: Float
            let fillRatioHeight: Float
            let blendWidth: Float
            let blendHeight: Float
            let rawWidth: Float
            let rawHeight: Float
            let correctedWidth: Float
            let correctedHeight: Float
            let width: Float
            let height: Float
            let depth: Float
        }

        // Experimental comparison-only auto-router between corner-shot and straight-on formulas.
        // Kept separate so it can be removed without affecting the active ROOM_DIMS_APP path.
        struct AutoShotDimensions {
            let usedFocal: Bool
            let shotType: String
            let orientationLabel: String
            let isCornerShot: Bool
            let cuboidRatio: Float
            let cuboidThreshold: Float
            let tiltDegrees: Float
            let trimmedXSpan: Float
            let trimmedYSpan: Float
            let trimmedZSpan: Float
            let floorDiagonal: Float
            let fillWidth: Float
            let blend: Float
            let rawWidth: Float
            let rawHeight: Float
            let width: Float
            let height: Float
            let depth: Float
        }

        // Experimental comparison-only rhombus decomposition for corner shots.
        // Kept fully separate so it can be deleted without touching active measurements.
        struct RhombusShotDimensions {
            let usedFocal: Bool
            let shotType: String
            let orientationLabel: String
            let isCornerShot: Bool
            let cuboidRatio: Float
            let cuboidThreshold: Float
            let tiltDegrees: Float
            let tiltReliable: Bool
            let trimmedXSpan: Float
            let trimmedYSpan: Float
            let trimmedZSpan: Float
            let floorDiagonal: Float
            let shortDiagonal: Float
            let pcaSpan1: Float
            let pcaSpan2: Float
            let rawWidth: Float
            let rawHeight: Float
            let width: Float
            let height: Float
            let depth: Float
        }

        func estimateBackWallDimensions() -> BackWallDimensions? {
            guard !rows.isEmpty else { return nil }

            let xs = rows.map(\.x).filter(\.isFinite).sorted()
            let ys = rows.map(\.y).filter(\.isFinite).sorted()
            let depths = rows.map { abs($0.z) }.filter { $0.isFinite && $0 > 0.01 }.sorted()
            guard xs.count >= 64, ys.count >= 64, depths.count >= 64 else { return nil }

            func percentileIndex(_ p: Float, count: Int) -> Int {
                guard count > 1 else { return 0 }
                let raw = Int((Float(count - 1) * p).rounded(.down))
                return min(max(raw, 0), count - 1)
            }

            let xP3 = xs[percentileIndex(0.03, count: xs.count)]
            let xP97 = xs[percentileIndex(0.97, count: xs.count)]
            let yP3 = ys[percentileIndex(0.03, count: ys.count)]
            let yP97 = ys[percentileIndex(0.97, count: ys.count)]
            let zP3 = depths[percentileIndex(0.03, count: depths.count)]
            let zP97 = depths[percentileIndex(0.97, count: depths.count)]

            let trimmedXSpan = xP97 - xP3
            let trimmedYSpan = yP97 - yP3
            let trimmedZSpan = zP97 - zP3
            guard trimmedXSpan.isFinite, trimmedYSpan.isFinite, trimmedZSpan.isFinite,
                  trimmedXSpan > 0.01, trimmedZSpan > 0.01 else { return nil }

            let floorDiagonal = sqrt(max(0, trimmedXSpan * trimmedXSpan + trimmedZSpan * trimmedZSpan))
            let trimmedDepths = depths.filter { $0 >= zP3 && $0 <= zP97 }
            guard trimmedDepths.count >= 64 else { return nil }

            let binCount = 200
            let binWidth = max(trimmedZSpan / Float(binCount), 1e-4)
            var histogram = [Int](repeating: 0, count: binCount)
            for z in trimmedDepths {
                var index = Int(((z - zP3) / binWidth).rounded(.down))
                index = min(max(index, 0), binCount - 1)
                histogram[index] += 1
            }
            guard let peakIndex = histogram.enumerated().max(by: { $0.element < $1.element })?.offset else {
                return nil
            }

            let zMode = zP3 + (Float(peakIndex) + 0.5) * binWidth
            let band = max(0.10 * zMode, 1e-4)
            let backWallRows = rows.filter { abs(abs($0.z) - zMode) < band }
            guard backWallRows.count >= 64 else { return nil }

            let backWallXs = backWallRows.map(\.x).filter(\.isFinite).sorted()
            let backWallYs = backWallRows.map(\.y).filter(\.isFinite).sorted()
            let backWallDepths = backWallRows.map { abs($0.z) }.filter { $0.isFinite && $0 > 0.01 }.sorted()
            guard backWallXs.count >= 64, backWallYs.count >= 64, backWallDepths.count >= 64 else { return nil }

            let idx5 = percentileIndex(0.05, count: backWallXs.count)
            let idx95 = percentileIndex(0.95, count: backWallXs.count)
            let yIdx5 = percentileIndex(0.05, count: backWallYs.count)
            let yIdx95 = percentileIndex(0.95, count: backWallYs.count)
            let rawWidth = backWallXs[idx95] - backWallXs[idx5]
            let rawHeight = backWallYs[yIdx95] - backWallYs[yIdx5]
            let zMedian = backWallDepths[backWallDepths.count / 2]
            let zMean = backWallDepths.reduce(0, +) / Float(backWallDepths.count)
            let legacyWidth = rawWidth / 1.2
            let legacyHeight = rawHeight / 1.2
            let legacyDepth = zMedian * 1.08
            let finalDepth = floorDiagonal
            let finalWidth: Float
            if rawHeight > rawWidth {
                finalWidth = sqrt(max(0, rawHeight * rawHeight - rawWidth * rawWidth))
            } else {
                finalWidth = rawWidth
            }
            let finalHeight: Float
            if trimmedYSpan > trimmedXSpan {
                finalHeight = sqrt(max(0, trimmedYSpan * trimmedYSpan - trimmedXSpan * trimmedXSpan))
            } else {
                finalHeight = trimmedYSpan
            }

            return BackWallDimensions(
                zMode: zMode,
                zMedian: zMedian,
                zMean: zMean,
                floorDiagonal: floorDiagonal,
                trimmedXSpan: trimmedXSpan,
                trimmedYSpan: trimmedYSpan,
                trimmedZSpan: trimmedZSpan,
                rawWidth: rawWidth,
                rawHeight: rawHeight,
                legacyWidth: legacyWidth,
                legacyHeight: legacyHeight,
                legacyDepth: legacyDepth,
                width: finalWidth,
                height: finalHeight,
                depth: finalDepth,
                count: backWallRows.count,
                band: band
            )
        }

        func estimateRoomDimensionsV3() -> FloorAlignedPCADimensions? {
            guard rows.count >= 128 else { return nil }

            let positions = rows.map { SIMD3<Float>($0.x, $0.y, abs($0.z)) }
            let sampleCount = min(50_000, positions.count)
            let ransacIterations = 500
            let epsilon: Float = 0.05

            struct SeedableRNG {
                var state: UInt64

                init(seed: UInt64) {
                    self.state = seed == 0 ? 1 : seed
                }

                mutating func next() -> UInt64 {
                    state ^= state << 13
                    state ^= state >> 7
                    state ^= state << 17
                    return state
                }
            }

            func percentileIndex(_ p: Float, count: Int) -> Int {
                guard count > 1 else { return 0 }
                let raw = Int((Float(count - 1) * p).rounded(.down))
                return min(max(raw, 0), count - 1)
            }

            let xs = positions.map(\.x).filter(\.isFinite).sorted()
            let depths = positions.map(\.z).filter { $0.isFinite && $0 > 0.01 }.sorted()
            guard xs.count >= 64, depths.count >= 64 else { return nil }
            let zP3 = depths[percentileIndex(0.03, count: depths.count)]
            let zP97 = depths[percentileIndex(0.97, count: depths.count)]
            let trimmedZSpan = zP97 - zP3
            guard trimmedZSpan > 0.01 else { return nil }
            let trimmedDepths = depths.filter { $0 >= zP3 && $0 <= zP97 }
            guard trimmedDepths.count >= 64 else { return nil }
            let binCount = 200
            let binWidth = max(trimmedZSpan / Float(binCount), 1e-4)
            var histogram = [Int](repeating: 0, count: binCount)
            for depth in trimmedDepths {
                var bucket = Int(((depth - zP3) / binWidth).rounded(.down))
                bucket = min(max(bucket, 0), binCount - 1)
                histogram[bucket] += 1
            }
            guard let peakIndex = histogram.enumerated().max(by: { $0.element < $1.element })?.offset else {
                return nil
            }
            let backWallZ = zP3 + (Float(peakIndex) + 0.5) * binWidth
            let band = max(0.10 * backWallZ, 1e-4)
            let backWallPoints = positions.filter { abs($0.z - backWallZ) < band }
            let backWallXs = backWallPoints.map(\.x).filter(\.isFinite).sorted()
            guard backWallXs.count >= 64 else { return nil }
            let x5 = backWallXs[percentileIndex(0.05, count: backWallXs.count)]
            let x95 = backWallXs[percentileIndex(0.95, count: backWallXs.count)]
            let rawWidthSeed = x95 - x5
            let ransacSeed = UInt64(bitPattern: Int64((backWallZ * 1_000_000 + rawWidthSeed * 1_000).rounded()))
            var rng = SeedableRNG(seed: ransacSeed)

            var sampled: [SIMD3<Float>] = []
            sampled.reserveCapacity(sampleCount)
            for _ in 0..<sampleCount {
                let index = Int(rng.next() % UInt64(positions.count))
                sampled.append(positions[index])
            }
            guard sampled.count >= 3 else { return nil }

            var bestNormal = SIMD3<Float>(0, 1, 0)
            var bestD: Float = 0
            var bestInliers = 0

            for _ in 0..<ransacIterations {
                let i0 = Int(rng.next() % UInt64(sampled.count))
                let i1 = Int(rng.next() % UInt64(sampled.count))
                let i2 = Int(rng.next() % UInt64(sampled.count))
                let p0 = sampled[i0]
                let p1 = sampled[i1]
                let p2 = sampled[i2]
                let v1 = p1 - p0
                let v2 = p2 - p0
                let crossVector = simd_cross(v1, v2)
                let crossLength = simd_length(crossVector)
                guard crossLength > 1e-6 else { continue }

                var normal = crossVector / crossLength
                var d = -simd_dot(normal, p0)
                guard abs(normal.y) > 0.8 else { continue }
                if normal.y < 0 {
                    normal *= -1
                    d *= -1
                }

                var inliers = 0
                for point in sampled {
                    if abs(simd_dot(normal, point) + d) < epsilon {
                        inliers += 1
                    }
                }

                if inliers > bestInliers {
                    bestInliers = inliers
                    bestNormal = normal
                    bestD = d
                }
            }

            guard bestInliers > 0 else { return nil }

            let target = SIMD3<Float>(0, 1, 0)
            let clampedCosTheta = min(max(simd_dot(bestNormal, target), -1), 1)
            let rotationAxisRaw = simd_cross(bestNormal, target)
            let sinTheta = simd_length(rotationAxisRaw)
            let rotationMatrix: simd_float3x3
            if sinTheta < 1e-6 {
                rotationMatrix = matrix_identity_float3x3
            } else {
                let axis = rotationAxisRaw / sinTheta
                let ax = axis.x
                let ay = axis.y
                let az = axis.z
                let c = clampedCosTheta
                let s = sinTheta
                let t = 1 - c
                rotationMatrix = simd_float3x3(
                    SIMD3<Float>(t * ax * ax + c, t * ax * ay - s * az, t * ax * az + s * ay),
                    SIMD3<Float>(t * ax * ay + s * az, t * ay * ay + c, t * ay * az - s * ax),
                    SIMD3<Float>(t * ax * az - s * ay, t * ay * az + s * ax, t * az * az + c)
                )
            }

            var rotatedY: [Float] = []
            var floorPoints: [SIMD2<Float>] = []
            rotatedY.reserveCapacity(positions.count)
            floorPoints.reserveCapacity(positions.count)

            for position in positions {
                let rotated = rotationMatrix * position
                rotatedY.append(rotated.y)
                floorPoints.append(SIMD2<Float>(rotated.x, rotated.z))
            }

            guard rotatedY.count >= 64, floorPoints.count >= 64 else { return nil }
            rotatedY.sort()
            let yP3 = rotatedY[percentileIndex(0.03, count: rotatedY.count)]
            let yP97 = rotatedY[percentileIndex(0.97, count: rotatedY.count)]
            let roomHeight = yP97 - yP3

            let n = Float(floorPoints.count)
            let meanX = floorPoints.reduce(0) { $0 + $1.x } / n
            let meanZ = floorPoints.reduce(0) { $0 + $1.y } / n

            var cxx: Float = 0
            var cxz: Float = 0
            var czz: Float = 0
            for point in floorPoints {
                let dx = point.x - meanX
                let dz = point.y - meanZ
                cxx += dx * dx
                cxz += dx * dz
                czz += dz * dz
            }
            cxx /= n
            cxz /= n
            czz /= n

            let trace = cxx + czz
            let determinant = cxx * czz - cxz * cxz
            let discriminant = sqrt(max(0, trace * trace / 4 - determinant))
            let lambda1 = trace / 2 + discriminant
            let lambda2 = trace / 2 - discriminant

            let e1: SIMD2<Float>
            if abs(cxz) > 1e-6 {
                e1 = simd_normalize(SIMD2<Float>(lambda1 - czz, cxz))
            } else {
                e1 = SIMD2<Float>(1, 0)
            }
            let e2 = SIMD2<Float>(-e1.y, e1.x)

            var uValues: [Float] = []
            var vValues: [Float] = []
            uValues.reserveCapacity(floorPoints.count)
            vValues.reserveCapacity(floorPoints.count)

            for point in floorPoints {
                let centered = SIMD2<Float>(point.x - meanX, point.y - meanZ)
                uValues.append(simd_dot(centered, e1))
                vValues.append(simd_dot(centered, e2))
            }

            uValues.sort()
            vValues.sort()
            let uP3 = uValues[percentileIndex(0.03, count: uValues.count)]
            let uP97 = uValues[percentileIndex(0.97, count: uValues.count)]
            let vP3 = vValues[percentileIndex(0.03, count: vValues.count)]
            let vP97 = vValues[percentileIndex(0.97, count: vValues.count)]
            let span1 = uP97 - uP3
            let span2 = vP97 - vP3

            let roomWidth = min(span1, span2)
            let roomDepth = max(span1, span2)
            let originalTiltDegrees = acos(abs(bestNormal.y)) * 180 / .pi
            let tiltReliable = originalTiltDegrees < 25 && bestInliers > 3000
            let tiltDegrees: Float = tiltReliable ? originalTiltDegrees : 8.0
            _ = bestD // retained for quick future debugging/removal without touching the RANSAC core

            return FloorAlignedPCADimensions(
                floorNormal: bestNormal,
                tiltDegrees: tiltDegrees,
                tiltRawDegrees: originalTiltDegrees,
                tiltReliable: tiltReliable,
                sampleCount: sampleCount,
                ransacIterations: ransacIterations,
                ransacInliers: bestInliers,
                epsilon: epsilon,
                lambda1: lambda1,
                lambda2: lambda2,
                span1: span1,
                span2: span2,
                width: roomWidth,
                height: roomHeight,
                depth: roomDepth
            )
        }

        func estimateRoomDimensionsV4() -> SceneExtensionDimensions? {
            guard !rows.isEmpty else { return nil }

            let xs = rows.map(\.x).filter(\.isFinite).sorted()
            let ys = rows.map(\.y).filter(\.isFinite).sorted()
            let depths = rows.map { abs($0.z) }.filter { $0.isFinite && $0 > 0.01 }.sorted()
            guard xs.count >= 64, ys.count >= 64, depths.count >= 64 else { return nil }

            func percentileIndex(_ p: Float, count: Int) -> Int {
                guard count > 1 else { return 0 }
                let raw = Int((Float(count - 1) * p).rounded(.down))
                return min(max(raw, 0), count - 1)
            }

            let xP3 = xs[percentileIndex(0.03, count: xs.count)]
            let xP97 = xs[percentileIndex(0.97, count: xs.count)]
            let zP3 = depths[percentileIndex(0.03, count: depths.count)]
            let zP97 = depths[percentileIndex(0.97, count: depths.count)]

            let trimmedXSpan = xP97 - xP3
            let trimmedZSpan = zP97 - zP3
            guard trimmedXSpan.isFinite, trimmedZSpan.isFinite,
                  trimmedXSpan > 0.01, trimmedZSpan > 0.01 else { return nil }

            let floorDiagonal = sqrt(max(0, trimmedXSpan * trimmedXSpan + trimmedZSpan * trimmedZSpan))
            let trimmedDepths = depths.filter { $0 >= zP3 && $0 <= zP97 }
            guard trimmedDepths.count >= 64 else { return nil }

            let binCount = 200
            let binWidth = max(trimmedZSpan / Float(binCount), 1e-4)
            var histogram = [Int](repeating: 0, count: binCount)
            for depth in trimmedDepths {
                var bucket = Int(((depth - zP3) / binWidth).rounded(.down))
                bucket = min(max(bucket, 0), binCount - 1)
                histogram[bucket] += 1
            }
            guard let peakIndex = histogram.enumerated().max(by: { $0.element < $1.element })?.offset else {
                return nil
            }

            let zMode = zP3 + (Float(peakIndex) + 0.5) * binWidth
            let band = max(0.10 * zMode, 1e-4)
            let backWallRows = rows.filter { abs(abs($0.z) - zMode) < band }
            guard backWallRows.count >= 64 else { return nil }

            let backWallXs = backWallRows.map(\.x).filter(\.isFinite).sorted()
            let backWallYs = backWallRows.map(\.y).filter(\.isFinite).sorted()
            guard backWallXs.count >= 64, backWallYs.count >= 64 else { return nil }

            let idx5 = percentileIndex(0.05, count: backWallXs.count)
            let idx95 = percentileIndex(0.95, count: backWallXs.count)
            let yIdx5 = percentileIndex(0.05, count: backWallYs.count)
            let yIdx95 = percentileIndex(0.95, count: backWallYs.count)
            let rawWidth = backWallXs[idx95] - backWallXs[idx5]
            let rawHeight = backWallYs[yIdx95] - backWallYs[yIdx5]
            let hasFocal = unprojectionFocalPx > 0.01
            if !hasFocal {
                return SceneExtensionDimensions(
                    usedFocal: false,
                    approach: "FALLBACK_NO_FOCAL",
                    focalPx: 0,
                    imageWidth: Float(sourceImageSize.width),
                    imageHeight: Float(sourceImageSize.height),
                    fovDiagonal: 0,
                    minDepth: zP3,
                    maxLateral: 0,
                    sceneExtension: 0,
                    rawWidth: rawWidth,
                    rawHeight: rawHeight,
                    width: rawWidth / 1.2,
                    height: rawHeight / 1.2,
                    depth: floorDiagonal
                )
            }

            let imageWidth = Float(sourceImageSize.width)
            let imageHeight = Float(sourceImageSize.height)
            let fovDiagonal = sqrt(
                max(0, (imageWidth / unprojectionFocalPx) * (imageWidth / unprojectionFocalPx) +
                       (imageHeight / unprojectionFocalPx) * (imageHeight / unprojectionFocalPx))
            )
            let minDepth = zP3
            let maxDisparity: Float = 0.08
            let maxLateral = maxDisparity * fovDiagonal * minDepth
            let sceneExtension = 2 * maxLateral
            let correctedWidth = max(0, rawWidth - sceneExtension)
            let correctedHeight = max(0, rawHeight - sceneExtension)

            return SceneExtensionDimensions(
                usedFocal: true,
                approach: "SHARP_SCENE_EXTENSION",
                focalPx: unprojectionFocalPx,
                imageWidth: imageWidth,
                imageHeight: imageHeight,
                fovDiagonal: fovDiagonal,
                minDepth: minDepth,
                maxLateral: maxLateral,
                sceneExtension: sceneExtension,
                rawWidth: rawWidth,
                rawHeight: rawHeight,
                width: correctedWidth,
                height: correctedHeight,
                depth: floorDiagonal
            )
        }

        func estimateRoomDimensionsV5() -> AdaptiveSceneExtensionDimensions? {
            guard !rows.isEmpty else { return nil }

            let xs = rows.map(\.x).filter(\.isFinite).sorted()
            let ys = rows.map(\.y).filter(\.isFinite).sorted()
            let depths = rows.map { abs($0.z) }.filter { $0.isFinite && $0 > 0.01 }.sorted()
            guard xs.count >= 64, ys.count >= 64, depths.count >= 64 else { return nil }

            func percentileIndex(_ p: Float, count: Int) -> Int {
                guard count > 1 else { return 0 }
                let raw = Int((Float(count - 1) * p).rounded(.down))
                return min(max(raw, 0), count - 1)
            }

            func blendFactor(_ fillRatio: Float) -> Float {
                let normalized = (fillRatio - 0.55) / (0.75 - 0.55)
                return min(max(normalized, 0), 1)
            }

            let xP3 = xs[percentileIndex(0.03, count: xs.count)]
            let xP97 = xs[percentileIndex(0.97, count: xs.count)]
            let zP3 = depths[percentileIndex(0.03, count: depths.count)]
            let zP97 = depths[percentileIndex(0.97, count: depths.count)]

            let trimmedXSpan = xP97 - xP3
            let trimmedZSpan = zP97 - zP3
            guard trimmedXSpan.isFinite, trimmedZSpan.isFinite,
                  trimmedXSpan > 0.01, trimmedZSpan > 0.01 else { return nil }

            let floorDiagonal = sqrt(max(0, trimmedXSpan * trimmedXSpan + trimmedZSpan * trimmedZSpan))
            let trimmedDepths = depths.filter { $0 >= zP3 && $0 <= zP97 }
            guard trimmedDepths.count >= 64 else { return nil }

            let binCount = 200
            let binWidth = max(trimmedZSpan / Float(binCount), 1e-4)
            var histogram = [Int](repeating: 0, count: binCount)
            for depth in trimmedDepths {
                var bucket = Int(((depth - zP3) / binWidth).rounded(.down))
                bucket = min(max(bucket, 0), binCount - 1)
                histogram[bucket] += 1
            }
            guard let peakIndex = histogram.enumerated().max(by: { $0.element < $1.element })?.offset else {
                return nil
            }

            let backWallZ = zP3 + (Float(peakIndex) + 0.5) * binWidth
            let band = max(0.10 * backWallZ, 1e-4)
            let backWallRows = rows.filter { abs(abs($0.z) - backWallZ) < band }
            guard backWallRows.count >= 64 else { return nil }

            let backWallXs = backWallRows.map(\.x).filter(\.isFinite).sorted()
            let backWallYs = backWallRows.map(\.y).filter(\.isFinite).sorted()
            guard backWallXs.count >= 64, backWallYs.count >= 64 else { return nil }

            let idx5 = percentileIndex(0.05, count: backWallXs.count)
            let idx95 = percentileIndex(0.95, count: backWallXs.count)
            let yIdx5 = percentileIndex(0.05, count: backWallYs.count)
            let yIdx95 = percentileIndex(0.95, count: backWallYs.count)
            let rawWidth = backWallXs[idx95] - backWallXs[idx5]
            let rawHeight = backWallYs[yIdx95] - backWallYs[yIdx5]
            let hasFocal = unprojectionFocalPx > 0.01
            if !hasFocal {
                return AdaptiveSceneExtensionDimensions(
                    usedFocal: false,
                    approach: "FALLBACK_NO_FOCAL",
                    focalPx: 0,
                    imageWidth: Float(sourceImageSize.width),
                    imageHeight: Float(sourceImageSize.height),
                    fovDiagonal: 0,
                    minDepth: zP3,
                    backWallZ: zP3,
                    maxLateral: 0,
                    sceneExtension: 0,
                    fovWidthAtBackWall: 0,
                    fovHeightAtBackWall: 0,
                    fillRatioWidth: 0,
                    fillRatioHeight: 0,
                    blendWidth: 0,
                    blendHeight: 0,
                    rawWidth: rawWidth,
                    rawHeight: rawHeight,
                    correctedWidth: rawWidth / 1.2,
                    correctedHeight: rawHeight / 1.2,
                    width: rawWidth / 1.2,
                    height: rawHeight / 1.2,
                    depth: floorDiagonal
                )
            }

            let imageWidth = Float(sourceImageSize.width)
            let imageHeight = Float(sourceImageSize.height)
            let fovDiagonal = sqrt(
                max(0, (imageWidth / unprojectionFocalPx) * (imageWidth / unprojectionFocalPx) +
                       (imageHeight / unprojectionFocalPx) * (imageHeight / unprojectionFocalPx))
            )
            let minDepth = zP3
            let maxDisparity: Float = 0.08
            let maxLateral = maxDisparity * fovDiagonal * minDepth
            let sceneExtension = 2 * maxLateral

            let fovWidthAtBackWall = imageWidth * backWallZ / unprojectionFocalPx
            let fovHeightAtBackWall = imageHeight * backWallZ / unprojectionFocalPx
            guard fovWidthAtBackWall > 1e-6, fovHeightAtBackWall > 1e-6 else { return nil }

            let fillRatioWidth = rawWidth / fovWidthAtBackWall
            let fillRatioHeight = rawHeight / fovHeightAtBackWall
            let blendWidth = blendFactor(fillRatioWidth)
            let blendHeight = blendFactor(fillRatioHeight)

            let correctedWidth = rawWidth - sceneExtension
            let correctedHeight = rawHeight - sceneExtension * 0.5
            let blendedWidth = correctedWidth + blendWidth * (rawWidth - correctedWidth)
            let blendedHeight = correctedHeight + blendHeight * (rawHeight - correctedHeight)

            return AdaptiveSceneExtensionDimensions(
                usedFocal: true,
                approach: "ADAPTIVE_SCENE_EXTENSION",
                focalPx: unprojectionFocalPx,
                imageWidth: imageWidth,
                imageHeight: imageHeight,
                fovDiagonal: fovDiagonal,
                minDepth: minDepth,
                backWallZ: backWallZ,
                maxLateral: maxLateral,
                sceneExtension: sceneExtension,
                fovWidthAtBackWall: fovWidthAtBackWall,
                fovHeightAtBackWall: fovHeightAtBackWall,
                fillRatioWidth: fillRatioWidth,
                fillRatioHeight: fillRatioHeight,
                blendWidth: blendWidth,
                blendHeight: blendHeight,
                rawWidth: rawWidth,
                rawHeight: rawHeight,
                correctedWidth: correctedWidth,
                correctedHeight: correctedHeight,
                width: max(blendedWidth, 0.5),
                height: max(blendedHeight, 0.5),
                depth: floorDiagonal
            )
        }

        func estimateRoomDimensionsV6() -> AutoShotDimensions? {
            guard !rows.isEmpty else { return nil }
            let tiltDegrees = estimateRoomDimensionsV3()?.tiltDegrees ?? 0

            let xs = rows.map(\.x).filter(\.isFinite).sorted()
            let ys = rows.map(\.y).filter(\.isFinite).sorted()
            let depths = rows.map { abs($0.z) }.filter { $0.isFinite && $0 > 0.01 }.sorted()
            guard xs.count >= 64, ys.count >= 64, depths.count >= 64 else { return nil }

            func percentileIndex(_ p: Float, count: Int) -> Int {
                guard count > 1 else { return 0 }
                let raw = Int((Float(count - 1) * p).rounded(.down))
                return min(max(raw, 0), count - 1)
            }

            let xP3 = xs[percentileIndex(0.03, count: xs.count)]
            let xP97 = xs[percentileIndex(0.97, count: xs.count)]
            let yP3 = ys[percentileIndex(0.03, count: ys.count)]
            let yP97 = ys[percentileIndex(0.97, count: ys.count)]
            let zP3 = depths[percentileIndex(0.03, count: depths.count)]
            let zP97 = depths[percentileIndex(0.97, count: depths.count)]

            let trimmedXSpan = xP97 - xP3
            let trimmedYSpan = yP97 - yP3
            let trimmedZSpan = zP97 - zP3
            guard trimmedXSpan.isFinite, trimmedYSpan.isFinite, trimmedZSpan.isFinite,
                  trimmedXSpan > 0.01, trimmedYSpan > 0.01, trimmedZSpan > 0.01 else { return nil }

            let floorDiagonal = sqrt(max(0, trimmedXSpan * trimmedXSpan + trimmedZSpan * trimmedZSpan))
            let maxSpan = max(trimmedXSpan, max(trimmedYSpan, trimmedZSpan))
            let minSpan = min(trimmedXSpan, min(trimmedYSpan, trimmedZSpan))
            guard minSpan > 1e-6 else { return nil }
            let cuboidRatio = maxSpan / minSpan
            let imageWidth = Float(sourceImageSize.width)
            let imageHeight = Float(sourceImageSize.height)
            let isLandscape = imageWidth > imageHeight
            let imageAspect = imageHeight > 1e-6 ? (imageWidth / imageHeight) : 1
            let cuboidThreshold: Float = isLandscape ? (1.45 * imageAspect) : 1.45
            let isCornerShot = cuboidRatio < cuboidThreshold
            let orientationLabel = isLandscape ? "LANDSCAPE" : "PORTRAIT"

            let trimmedDepths = depths.filter { $0 >= zP3 && $0 <= zP97 }
            guard trimmedDepths.count >= 64 else { return nil }

            let binCount = 200
            let binWidth = max(trimmedZSpan / Float(binCount), 1e-4)
            var histogram = [Int](repeating: 0, count: binCount)
            for depth in trimmedDepths {
                var bucket = Int(((depth - zP3) / binWidth).rounded(.down))
                bucket = min(max(bucket, 0), binCount - 1)
                histogram[bucket] += 1
            }
            guard let peakIndex = histogram.enumerated().max(by: { $0.element < $1.element })?.offset else {
                return nil
            }

            let backWallZ = zP3 + (Float(peakIndex) + 0.5) * binWidth
            let band = max(0.10 * backWallZ, 1e-4)
            let backWallRows = rows.filter { abs(abs($0.z) - backWallZ) < band }
            guard backWallRows.count >= 64 else { return nil }

            let backWallXs = backWallRows.map(\.x).filter(\.isFinite).sorted()
            let backWallYs = backWallRows.map(\.y).filter(\.isFinite).sorted()
            guard backWallXs.count >= 64, backWallYs.count >= 64 else { return nil }

            let idx5 = percentileIndex(0.05, count: backWallXs.count)
            let idx95 = percentileIndex(0.95, count: backWallXs.count)
            let yIdx5 = percentileIndex(0.05, count: backWallYs.count)
            let yIdx95 = percentileIndex(0.95, count: backWallYs.count)
            let rawWidth = backWallXs[idx95] - backWallXs[idx5]
            let rawHeight = backWallYs[yIdx95] - backWallYs[yIdx5]

            let roomWidth: Float
            let roomHeight: Float
            let roomDepth: Float
            let shotType: String
            let fillWidth: Float
            let blend: Float
            let hasFocal = unprojectionFocalPx > 0.01

            if !hasFocal {
                shotType = "FALLBACK_NO_FOCAL"
                roomWidth = rawWidth / 1.2
                roomHeight = rawHeight / 1.2
                roomDepth = floorDiagonal
                fillWidth = 0
                blend = 0
            } else if isCornerShot {
                shotType = "CORNER"
                roomWidth = rawWidth / 1.2
                if tiltDegrees < 12 {
                    roomHeight = rawHeight / 1.2
                } else {
                    roomHeight = rawHeight
                }
                let diagSquared = floorDiagonal * floorDiagonal
                let widthSquared = roomWidth * roomWidth
                if diagSquared > widthSquared {
                    roomDepth = sqrt(max(0, diagSquared - widthSquared))
                } else {
                    roomDepth = floorDiagonal
                }
                fillWidth = 0
                blend = 0
            } else {
                shotType = "STRAIGHT"
                let fovDiagonal = sqrt(
                    max(0, (imageWidth / unprojectionFocalPx) * (imageWidth / unprojectionFocalPx) +
                           (imageHeight / unprojectionFocalPx) * (imageHeight / unprojectionFocalPx))
                )
                let maxDisparity: Float = 0.08
                let minDepth = zP3
                let maxLateral = maxDisparity * fovDiagonal * minDepth
                let sceneExtension = 2 * maxLateral
                let fovWidthAtBackWall = imageWidth * backWallZ / unprojectionFocalPx
                let fillRatioW = fovWidthAtBackWall > 1e-6 ? (rawWidth / fovWidthAtBackWall) : 0
                let widthBlend = min(max((fillRatioW - 0.55) / (0.75 - 0.55), 0), 1)
                let correctedW = rawWidth - sceneExtension
                let correctedH = rawHeight - sceneExtension
                roomWidth = max(correctedW + widthBlend * (rawWidth - correctedW), 0.5)
                if tiltDegrees < 12 {
                    roomHeight = max(correctedH + widthBlend * (rawHeight - correctedH), 0.5)
                } else {
                    roomHeight = rawHeight
                }
                roomDepth = floorDiagonal
                fillWidth = fillRatioW
                blend = widthBlend
            }

            return AutoShotDimensions(
                usedFocal: hasFocal,
                shotType: shotType,
                orientationLabel: orientationLabel,
                isCornerShot: isCornerShot,
                cuboidRatio: cuboidRatio,
                cuboidThreshold: cuboidThreshold,
                tiltDegrees: tiltDegrees,
                trimmedXSpan: trimmedXSpan,
                trimmedYSpan: trimmedYSpan,
                trimmedZSpan: trimmedZSpan,
                floorDiagonal: floorDiagonal,
                fillWidth: fillWidth,
                blend: blend,
                rawWidth: rawWidth,
                rawHeight: rawHeight,
                width: roomWidth,
                height: roomHeight,
                depth: roomDepth
            )
        }

        func estimateRoomDimensionsV7() -> RhombusShotDimensions? {
            guard !rows.isEmpty else { return nil }
            let v3 = estimateRoomDimensionsV3()

            let xs = rows.map(\.x).filter(\.isFinite).sorted()
            let ys = rows.map(\.y).filter(\.isFinite).sorted()
            let depths = rows.map { abs($0.z) }.filter { $0.isFinite && $0 > 0.01 }.sorted()
            guard xs.count >= 64, ys.count >= 64, depths.count >= 64 else { return nil }

            func percentileIndex(_ p: Float, count: Int) -> Int {
                guard count > 1 else { return 0 }
                let raw = Int((Float(count - 1) * p).rounded(.down))
                return min(max(raw, 0), count - 1)
            }

            let xP3 = xs[percentileIndex(0.03, count: xs.count)]
            let xP97 = xs[percentileIndex(0.97, count: xs.count)]
            let yP3 = ys[percentileIndex(0.03, count: ys.count)]
            let yP97 = ys[percentileIndex(0.97, count: ys.count)]
            let zP3 = depths[percentileIndex(0.03, count: depths.count)]
            let zP97 = depths[percentileIndex(0.97, count: depths.count)]

            let trimmedXSpan = xP97 - xP3
            let trimmedYSpan = yP97 - yP3
            let trimmedZSpan = zP97 - zP3
            guard trimmedXSpan.isFinite, trimmedYSpan.isFinite, trimmedZSpan.isFinite,
                  trimmedXSpan > 0.01, trimmedYSpan > 0.01, trimmedZSpan > 0.01 else { return nil }

            let floorDiagonal = sqrt(max(0, trimmedXSpan * trimmedXSpan + trimmedZSpan * trimmedZSpan))
            let imageWidth = Float(sourceImageSize.width)
            let imageHeight = Float(sourceImageSize.height)
            let isLandscape = imageWidth > imageHeight
            let imageAspect = min(imageWidth, imageHeight) > 1e-6 ? (max(imageWidth, imageHeight) / min(imageWidth, imageHeight)) : 1
            let cuboidThreshold: Float = isLandscape ? (1.50 * imageAspect) : 1.45
            let maxSpan = max(trimmedXSpan, max(trimmedYSpan, trimmedZSpan))
            let minSpan = min(trimmedXSpan, min(trimmedYSpan, trimmedZSpan))
            guard minSpan > 1e-6 else { return nil }
            let cuboidRatio = maxSpan / minSpan
            let isCornerShot = cuboidRatio < cuboidThreshold
            let orientationLabel = isLandscape ? "LANDSCAPE" : "PORTRAIT"

            let trimmedDepths = depths.filter { $0 >= zP3 && $0 <= zP97 }
            guard trimmedDepths.count >= 64 else { return nil }

            let binCount = 200
            let binWidth = max(trimmedZSpan / Float(binCount), 1e-4)
            var histogram = [Int](repeating: 0, count: binCount)
            for depth in trimmedDepths {
                var bucket = Int(((depth - zP3) / binWidth).rounded(.down))
                bucket = min(max(bucket, 0), binCount - 1)
                histogram[bucket] += 1
            }
            guard let peakIndex = histogram.enumerated().max(by: { $0.element < $1.element })?.offset else {
                return nil
            }

            let backWallZ = zP3 + (Float(peakIndex) + 0.5) * binWidth
            let band = max(0.10 * backWallZ, 1e-4)
            let backWallRows = rows.filter { abs(abs($0.z) - backWallZ) < band }
            guard backWallRows.count >= 64 else { return nil }

            let backWallXs = backWallRows.map(\.x).filter(\.isFinite).sorted()
            let backWallYs = backWallRows.map(\.y).filter(\.isFinite).sorted()
            guard backWallXs.count >= 64, backWallYs.count >= 64 else { return nil }

            let idx5 = percentileIndex(0.05, count: backWallXs.count)
            let idx95 = percentileIndex(0.95, count: backWallXs.count)
            let yIdx5 = percentileIndex(0.05, count: backWallYs.count)
            let yIdx95 = percentileIndex(0.95, count: backWallYs.count)
            let rawWidth = backWallXs[idx95] - backWallXs[idx5]
            let rawHeight = backWallYs[yIdx95] - backWallYs[yIdx5]

            let pcaSpan1 = v3?.span1 ?? 0
            let pcaSpan2 = v3?.span2 ?? 0
            let shortDiagonal = min(pcaSpan1, pcaSpan2)
            let tiltDegrees = v3?.tiltDegrees ?? 8.0
            let tiltReliable = v3?.tiltReliable ?? false

            let roomWidth: Float
            let roomHeight: Float
            let roomDepth: Float
            let shotType: String
            let hasFocal = unprojectionFocalPx > 0.01

            if !hasFocal {
                shotType = "FALLBACK_NO_FOCAL"
                roomWidth = rawWidth / 1.2
                roomHeight = rawHeight / 1.2
                roomDepth = floorDiagonal
            } else if isCornerShot {
                roomWidth = rawWidth / 1.2
                if tiltDegrees < 12 {
                    roomHeight = rawHeight / 1.2
                } else {
                    roomHeight = rawHeight
                }
                let product = floorDiagonal * shortDiagonal / 2
                let sumOfSquares = floorDiagonal * floorDiagonal
                let diffSquared = sumOfSquares - 2 * product
                if diffSquared >= 0 {
                    let sum = sqrt(max(0, sumOfSquares + 2 * product))
                    let diff = sqrt(max(0, diffSquared))
                    let side1 = (sum + diff) / 2
                    let side2 = (sum - diff) / 2
                    let rhombusRatio = max(side1, side2) > 1e-6 ? abs(side1 - side2) / max(side1, side2) : .greatestFiniteMagnitude
                    if rhombusRatio < 0.5 && side1 > 0.5 && side2 > 0.5 {
                        shotType = "CORNER_V6_FALLBACK"
                    } else {
                        shotType = "CORNER_V6_FALLBACK"
                    }
                } else {
                    shotType = "CORNER_V6_FALLBACK"
                }
                let widthSquared = roomWidth * roomWidth
                let diagSquared = floorDiagonal * floorDiagonal
                if diagSquared > widthSquared {
                    roomDepth = sqrt(max(0, diagSquared - widthSquared))
                } else {
                    roomDepth = floorDiagonal
                }
            } else {
                shotType = "STRAIGHT"
                let fovDiagonal = sqrt(
                    max(0, (imageWidth / unprojectionFocalPx) * (imageWidth / unprojectionFocalPx) +
                           (imageHeight / unprojectionFocalPx) * (imageHeight / unprojectionFocalPx))
                )
                let maxLateral = Float(0.08) * fovDiagonal * zP3
                let sceneExtension = 2 * maxLateral
                let fovWidthAtBackWall = imageWidth * backWallZ / unprojectionFocalPx
                let fillRatioW = fovWidthAtBackWall > 1e-6 ? (rawWidth / fovWidthAtBackWall) : 0
                let blend = min(max((fillRatioW - 0.55) / 0.20, 0), 1)
                let correctedW = rawWidth - sceneExtension
                let correctedH = rawHeight - sceneExtension
                roomWidth = max(correctedW + blend * (rawWidth - correctedW), 0.5)
                if tiltDegrees < 12 {
                    roomHeight = max(correctedH + blend * (rawHeight - correctedH), 0.5)
                } else {
                    roomHeight = rawHeight
                }
                roomDepth = floorDiagonal
            }

            return RhombusShotDimensions(
                usedFocal: hasFocal,
                shotType: shotType,
                orientationLabel: orientationLabel,
                isCornerShot: isCornerShot,
                cuboidRatio: cuboidRatio,
                cuboidThreshold: cuboidThreshold,
                tiltDegrees: tiltDegrees,
                tiltReliable: tiltReliable,
                trimmedXSpan: trimmedXSpan,
                trimmedYSpan: trimmedYSpan,
                trimmedZSpan: trimmedZSpan,
                floorDiagonal: floorDiagonal,
                shortDiagonal: isCornerShot ? shortDiagonal : 0,
                pcaSpan1: pcaSpan1,
                pcaSpan2: pcaSpan2,
                rawWidth: rawWidth,
                rawHeight: rawHeight,
                width: roomWidth,
                height: roomHeight,
                depth: roomDepth
            )
        }

        // Debug first few gaussians
        for i in 0..<min(3, rows.count) {
            let g = rows[i]
            let quatLen = sqrt(g.r0*g.r0 + g.r1*g.r1 + g.r2*g.r2 + g.r3*g.r3)
            let gamma: Float = 1.0 / 2.2; let bright: Float = 1.1
            let red = UInt8(min(max(Int(pow(min(max(g.rawR * bright, 0), 1), gamma) * 255), 0), 255))
            let green = UInt8(min(max(Int(pow(min(max(g.rawG * bright, 0), 1), gamma) * 255), 0), 255))
            let blue = UInt8(min(max(Int(pow(min(max(g.rawB * bright, 0), 1), gamma) * 255), 0), 255))
            logDebug("SHARP PLY GAUSSIAN \(i) (final values):")
            logDebug("  pos: (\(g.x), \(g.y), \(g.z))")
            logDebug("  scale (log): (\(g.s0), \(g.s1), \(g.s2))")
            logDebug("  rot: (\(g.r0), \(g.r1), \(g.r2), \(g.r3)) len=\(quatLen)")
            logDebug("  opacity (logit): \(g.opacity)")
            logDebug("  color (uchar): (\(red), \(green), \(blue))")
        }

        // Helper: write uchar-RGB PLY with Y/Z flip for the in-app Metal renderer.
        func writeClassicPLY(toURL url: URL) throws {
            let headerData = Data(classicPlyContent.utf8)
            var buf = Data(capacity: headerData.count + gaussianCount * standardVertexBytes)
            buf.append(headerData)
            let gamma: Float = 1.0 / 2.2
            let brightness: Float = 1.1
            for g in rows {
                var px = g.x; var py = -g.y; var pz = -g.z
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

        try writeClassicPLY(toURL: classicFileURL)

        let classicAttributes = try FileManager.default.attributesOfItem(atPath: classicFileURL.path)
        let classicSize = classicAttributes[.size] as? UInt64 ?? 0
        logDebug("SHARP: Classic PLY saved (\(classicSize / 1024) KB) — Metal uchar RGB")

        logPlyBoundsDiagnostic(
            "SHARP classic_ply room bounds (vertex frame written to *_classic.ply; in-app Metal uses this file): " +
            "W=\(String(format: "%.3f", width)) H=\(String(format: "%.3f", height)) D=\(String(format: "%.3f", depth)) " +
            "X[\(String(format: "%.3f", clMinX)),\(String(format: "%.3f", clMaxX))] " +
            "Y[\(String(format: "%.3f", clMinY)),\(String(format: "%.3f", clMaxY))] " +
            "Z[\(String(format: "%.3f", clMinZ)),\(String(format: "%.3f", clMaxZ))] splats=\(gaussianCount)"
        )

        if let identity = SplatLoadHint.fileIdentity(for: classicFileURL) {
            let trimSampleStride = max(1, rows.count / 60_000)
            var sampledPositions: [SIMD3<Float>] = []
            sampledPositions.reserveCapacity(min(rows.count, 60_000))
            var classicPositionSum = SIMD3<Float>.zero
            for (index, row) in rows.enumerated() {
                let classicPosition = SIMD3<Float>(row.x, -row.y, -row.z)
                classicPositionSum += classicPosition
                if index % trimSampleStride == 0 {
                    sampledPositions.append(classicPosition)
                }
            }
            if sampledPositions.isEmpty, let firstRow = rows.first {
                sampledPositions.append(SIMD3<Float>(firstRow.x, -firstRow.y, -firstRow.z))
            }

            let xs = sampledPositions.map(\.x).sorted()
            let ys = sampledPositions.map(\.y).sorted()
            let zs = sampledPositions.map(\.z).sorted()
            let hi = min(sampledPositions.count - 1, max(0, Int(Float(sampledPositions.count) * 0.97)))
            let lo = min(sampledPositions.count - 1, max(0, Int(Float(sampledPositions.count) * 0.03)))
            let framingMin = SIMD3<Float>(xs[lo], ys[lo], zs[lo])
            let framingMax = SIMD3<Float>(xs[hi], ys[hi], zs[hi])
            let centroid = classicPositionSum / Float(max(rows.count, 1))
            let hint = SplatLoadHint(
                fileByteCount: identity.byteCount,
                fileModificationTimeIntervalSince1970: identity.modificationTime,
                splatCount: gaussianCount,
                fullBoundsMin: SIMD3<Float>(clMinX, clMinY, clMinZ),
                fullBoundsMax: SIMD3<Float>(clMaxX, clMaxY, clMaxZ),
                framingBoundsMin: framingMin,
                framingBoundsMax: framingMax,
                centroid: centroid
            )
            let hintURL = SplatLoadHint.sidecarURL(forRoomURL: classicFileURL)
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            let hintData = try encoder.encode(hint)
            try hintData.write(to: hintURL, options: [.atomic])
            logDebug(
                "⏱️ [SplatLoad] hint_saved source=SHARPService file=\(classicFileURL.lastPathComponent) " +
                "sampled=\(sampledPositions.count)/\(rows.count) sidecar=\(hintURL.lastPathComponent)"
            )
        } else {
            logDebug("⏱️ [SplatLoad] hint_save_skipped source=SHARPService file=\(classicFileURL.lastPathComponent) reason=file_identity_unavailable")
        }

        logDebug("[ROOM_DIMS_APP] DEFERRED source=async_preview_measurement file=\(classicFileName) splats=\(gaussianCount)")

        rows.removeAll(keepingCapacity: false)
        logMemorySnapshot("SHARPService.writePLY", details: "phase=after_drop_rows ply=\(classicFileName) splats=\(gaussianCount)")

        return (
            classic: classicFileURL,
            aabbWidth: width,
            aabbHeight: height,
            aabbDepth: depth,
            roomWidth: nil,
            roomHeight: nil,
            roomDepth: nil
        )
    }

    /// Writes `Room_<stamp>_thumbnail.jpg` next to the classic PLY for downstream room metadata consumers.
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
