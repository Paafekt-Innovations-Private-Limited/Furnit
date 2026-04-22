import Foundation
import CoreML

/// Centralized YOLOE model manager with On-Demand Resources (ODR) support.
///
/// The YOLOE model (~60 MB compiled) is used for furniture detection across all room
/// views (SharpRoomView, MeshRoomView, GLBRoomView, ModelViewerView). Delivering it
/// via ODR keeps it out of the initial app download and lets the OS purge it when
/// disk space is tight.
///
/// Usage:
///   let service = YOLOEModelService.shared
///   await service.ensureModelLoaded()       // triggers ODR download if needed
///   if let model = service.model { … }
@MainActor
class YOLOEModelService: ObservableObject {

    // MARK: - Singleton

    static let shared = YOLOEModelService()

    // MARK: - On-Demand Resources

    /// ODR tags for each YOLOE `.mlpackage` in the target (see `project.pbxproj` → Compile Sources → Asset Tags).
    /// Split tags so 11L vs 26L caches invalidate independently; bump a tag when replacing that package.
    private static let yoloeOdrTag11l = "yoloe_11l_seg_pf"
    private static let yoloeOdrTag26l = "yoloe_26l_seg_pf"
    private static let yoloeOdrTag26lO2m = "yoloe_26l_seg_pf_o2m"

    private static var allYoloeOdrTags: Set<String> {
        [yoloeOdrTag11l, yoloeOdrTag26l, yoloeOdrTag26lO2m]
    }

    /// Keeps the ODR request alive so the OS doesn't purge the downloaded resource
    private var resourceRequest: NSBundleResourceRequest?

    /// KVO observation for download progress
    private var progressObservation: NSKeyValueObservation?

    // MARK: - Published State

    /// The loaded CoreML model — nil until download + load completes
    @Published var model: MLModel?

    /// True while the model is being downloaded or loaded
    @Published var isLoadingModel: Bool = false

    /// True while an ODR download is in progress
    @Published var isDownloadingResources: Bool = false

    /// Download progress (0.0 – 1.0)
    @Published var downloadProgress: Double = 0.0

    /// Human-readable status for UI consumption
    @Published var statusMessage: String = ""

    /// Whether the ODR resource is available locally
    @Published var resourcesAvailable: Bool = false

    // MARK: - Init

    private init() {
        // Don't eagerly load — the model is only needed when a room view appears
    }

    // MARK: - Public API

    /// Ensure the model is loaded; safe to call repeatedly (no-op when already loaded).
    /// Call from each room view's `.onAppear`.
    func ensureModelLoaded() {
        guard model == nil && !isLoadingModel else { return }
        Task {
            await loadModel()
        }
    }

    /// Wait until `model` is non-nil or `maxWaitSeconds` elapses (for one-shot room calibration after ODR).
    func waitForModelReady(maxWaitSeconds: TimeInterval = 45) async {
        if model != nil { return }
        ensureModelLoaded()
        let deadline = Date().addingTimeInterval(maxWaitSeconds)
        while model == nil, Date() < deadline {
            try? await Task.sleep(nanoseconds: 100_000_000)
        }
    }

    /// Release model and ODR resources — call when the app goes to background
    /// or memory pressure occurs.
    func releaseResources() {
        progressObservation?.invalidate()
        progressObservation = nil
        resourceRequest?.endAccessingResources()
        resourceRequest = nil
        resourcesAvailable = false
        model = nil
        isLoadingModel = false
        isDownloadingResources = false
        downloadProgress = 0.0
        statusMessage = ""
        YoloEDetectionParser.trimScratchBuffers()
        logDebug("YOLOE: Released model + ODR resources (memory)")
    }

    // MARK: - ODR

    /// Check whether the YOLOE resource tag is already on-device
    func checkResourceAvailability() async {
        let request = NSBundleResourceRequest(tags: Self.allYoloeOdrTags)
        request.loadingPriority = NSBundleResourceRequestLoadingPriorityUrgent

        let conditionallyAvailable = await request.conditionallyBeginAccessingResources()
        resourcesAvailable = conditionallyAvailable
        logDebug("YOLOE: ODR conditionallyBeginAccessingResources: \(conditionallyAvailable)")

        if conditionallyAvailable {
            self.resourceRequest = request
        }
    }

    /// Download the YOLOE model via ODR if not yet available.
    /// - Returns: `true` when the resource is available after this call.
    func downloadResourcesIfNeeded() async throws -> Bool {
        if resourcesAvailable { return true }

        if isDownloadingResources {
            // Wait for an existing download to finish
            while isDownloadingResources {
                try await Task.sleep(nanoseconds: 100_000_000)
            }
            return resourcesAvailable
        }

        logDebug("YOLOE: Starting ODR download…")
        isDownloadingResources = true
        downloadProgress = 0.0
        statusMessage = "Downloading detection model…"

        let request = NSBundleResourceRequest(tags: Self.allYoloeOdrTags)
        request.loadingPriority = NSBundleResourceRequestLoadingPriorityUrgent

        progressObservation = request.progress.observe(\.fractionCompleted, options: [.new]) { [weak self] progress, _ in
            Task { @MainActor in
                self?.downloadProgress = progress.fractionCompleted
                let percent = Int(progress.fractionCompleted * 100)
                self?.statusMessage = "Downloading detection model… \(percent)%"
            }
        }

        do {
            try await request.beginAccessingResources()
            logDebug("YOLOE: ODR download complete")

            resourcesAvailable = true
            isDownloadingResources = false
            downloadProgress = 1.0
            statusMessage = "Download complete"
            self.resourceRequest = request
            progressObservation = nil
            return true
        } catch {
            logDebug("YOLOE: ODR download failed: \(error)")
            isDownloadingResources = false
            downloadProgress = 0.0
            statusMessage = "Download failed"
            progressObservation = nil
            throw error
        }
    }

    // MARK: - Model Loading

    /// Internal load routine — handles ODR download then CoreML compilation.
    private func loadModel() async {
        guard !isLoadingModel && model == nil else { return }
        logDebug("YOLOE: Loading CoreML model…")

        isLoadingModel = true
        statusMessage = "Preparing detection model…"

        // Best-effort ODR download. Debug/Xcode installs still need this now that the model
        // lives in tagged asset packs instead of the app bundle.
        if !resourcesAvailable {
            do {
                let downloaded = try await downloadResourcesIfNeeded()
                if downloaded {
                    logDebug("YOLOE: ODR download succeeded")
                } else {
                    logDebug("YOLOE: ODR download returned false — will try bundle load anyway")
                }
            } catch {
                let ns = error as NSError
                if ns.domain == "NSCocoaErrorDomain", ns.code == 4994 {
                    logDebug("YOLOE: ODR not configured for this bundle, using app bundle")
                } else {
                    logDebug("YOLOE: ODR download failed (\(error)) — using app bundle")
                }
            }
        }

        statusMessage = "Loading detection model…"

        // Prefer the single shipped YOLOE package: `yoloe-11l-seg-pf.mlpackage`.
        let candidateNames = [
            ("yoloe-11l-seg-pf", "mlmodelc"),
            ("yoloe-11l-seg-pf", "mlpackage"),
        ]

        let config = MLModelConfiguration()
        // Keep YOLO-E on CPU only: the shipped Core ML export has previously SIGABRTed during
        // prediction on accelerated paths, and that crash is outside Swift error handling.
        config.computeUnits = .cpuOnly
        logDebug("YOLOE: Core ML computeUnits=cpuOnly (stable)")

        for (name, ext) in candidateNames {
            if let url = Bundle.main.url(forResource: name, withExtension: ext) {
                do {
                    let loadedModel = try MLModel(contentsOf: url, configuration: config)
                    self.model = loadedModel
                    logDebug("YOLOE: Loaded '\(name).\(ext)' successfully")
                    statusMessage = "Ready"
                    isLoadingModel = false
                    return
                } catch {
                    logDebug("YOLOE: Failed to load \(name).\(ext): \(error)")
                }
            }
        }

        logDebug("YOLOE: No model variant could be loaded")
        statusMessage = "Detection model unavailable"
        isLoadingModel = false
    }
}
