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

    /// ODR tag for the YOLOE Core ML package. Must change when shipping a new model so TestFlight/App Store
    /// clients do not reuse a cached asset from an older build (same tag = stale model).
    private static let yoloeModelTag = "yoloe_model_v26"

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

    // MARK: - Development vs Release

    private var isRunningFromXcode: Bool {
        #if DEBUG
        return true
        #else
        return false
        #endif
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

    /// Reload after toggling Core ML GPU in Settings (`computeUnits` are fixed at load time).
    func reloadForComputeUnitsChange() async {
        model = nil
        isLoadingModel = false
        await loadModel()
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
        logDebug("YOLOE: Released model + ODR resources (memory)")
    }

    // MARK: - ODR

    /// Check whether the YOLOE resource tag is already on-device
    func checkResourceAvailability() async {
        if isRunningFromXcode {
            logDebug("YOLOE: Running from Xcode — model bundled locally, skipping ODR")
            resourcesAvailable = true
            return
        }

        let tags: Set<String> = [Self.yoloeModelTag]
        let request = NSBundleResourceRequest(tags: tags)
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
        if isRunningFromXcode {
            logDebug("YOLOE: Running from Xcode — model bundled, no download needed")
            resourcesAvailable = true
            return true
        }

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

        let tags: Set<String> = [Self.yoloeModelTag]
        let request = NSBundleResourceRequest(tags: tags)
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

        // Best-effort ODR download (same pattern as SHARPService)
        if !resourcesAvailable && !isRunningFromXcode {
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

        // One-to-many 26L seg export (`scripts/export_yoloe26_onemany_user.py` → `_seg_o2m`), then legacy names.
        let candidateNames = [
            ("yoloe-26l-seg-pf_seg_o2m", "mlmodelc"),
            ("yoloe-26l-seg-pf_seg_o2m", "mlpackage"),
            ("yoloe-26l-seg-pf", "mlmodelc"),
            ("yoloe-26l-seg-pf", "mlpackage"),
        ]

        let config = MLModelConfiguration()
        // Default CPU-only: this Core ML export can SIGABRT inside `prediction` when using GPU
        // or ANE (Swift `try?` does not catch that). Optional GPU is in Settings.
        let allowGPU = UserDefaults.standard.bool(forKey: QualitySettings.yoloeCoreMLAllowGPUKey)
        config.computeUnits = allowGPU ? .cpuAndGPU : .cpuOnly
        logDebug("YOLOE: Core ML computeUnits=\(allowGPU ? "cpuAndGPU (experimental)" : "cpuOnly (stable)")")

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
