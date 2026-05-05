import Foundation
import CoreML

/// Centralized YOLOE model manager with On-Demand Resources (ODR) support.
///
/// The YOLOE model (~60 MB compiled) is used for furniture detection across all room
/// views (SharpRoomView, MeshRoomView, GLBRoomView, ModelViewerView).
/// It is embedded in the **main app bundle** by default so loading does not depend on
/// ODR’s streaming-unzip service (which can fail with error 4099 on Simulator).
/// ODR remains as a fallback path if a build tags the model pack as on-demand only.
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

    /// ODR tag for the shipped YOLOE `.mlpackage` in the target (see `project.pbxproj` → Resources → Asset Tags).
    private static let yoloeOdrTag11l = "yoloe_11l_seg_pf"

    private static var allYoloeOdrTags: Set<String> {
        [yoloeOdrTag11l]
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

    /// Unload Core ML from RAM but keep the ODR request alive (no `endAccessingResources`).
    /// Use on memory warnings so the next `ensureModelLoaded()` does not pay for a full ODR remount.
    func releaseLoadedModelOnlyPreservingODR() {
        model = nil
        isLoadingModel = false
        YoloEDetectionParser.trimScratchBuffers()
        logDebug("YOLOE: Unloaded Core ML model (ODR mount preserved)")
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
        defer { isDownloadingResources = false }

        downloadProgress = 0.0
        statusMessage = L10n.YOLOE.downloadingModel

        let tags = Self.allYoloeOdrTags

        do {
            // Fresh `NSBundleResourceRequest` per attempt — cannot call `beginAccessingResources` twice on one instance.
            let maxAttempts = 5
            for attempt in 1...maxAttempts {
                progressObservation?.invalidate()
                progressObservation = nil

                let request = NSBundleResourceRequest(tags: tags)
                request.loadingPriority = NSBundleResourceRequestLoadingPriorityUrgent

                progressObservation = request.progress.observe(\.fractionCompleted, options: [.new]) { [weak self] progress, _ in
                    Task { @MainActor in
                        self?.downloadProgress = progress.fractionCompleted
                        let percent = Int(progress.fractionCompleted * 100)
                        self?.statusMessage = L10n.YOLOE.downloadingModelPercent(percent)
                    }
                }

                do {
                    if await request.conditionallyBeginAccessingResources() {
                        logDebug("YOLOE: ODR resources available (conditionallyBeginAccessingResources)")
                        resourcesAvailable = true
                        downloadProgress = 1.0
                        statusMessage = L10n.YOLOE.downloadComplete
                        self.resourceRequest = request
                        progressObservation?.invalidate()
                        progressObservation = nil
                        return true
                    }

                    try await FurnitODRBeginAccessing.beginAccessingResources(request)
                    logDebug("YOLOE: ODR download complete")
                    resourcesAvailable = true
                    downloadProgress = 1.0
                    statusMessage = L10n.YOLOE.downloadComplete
                    self.resourceRequest = request
                    progressObservation?.invalidate()
                    progressObservation = nil
                    return true
                } catch {
                    progressObservation?.invalidate()
                    progressObservation = nil

                    let ns = error as NSError
                    let retryableODRFailure = (ns.domain == NSCocoaErrorDomain && ns.code == 4099)
                        || (ns.domain == NSCocoaErrorDomain && ns.code == -1)
                        || error.localizedDescription.localizedCaseInsensitiveContains("streaming unzip")
                        || error.localizedDescription.localizedCaseInsensitiveContains("connection invalidated")
                        || error.localizedDescription.localizedCaseInsensitiveContains("more than once")
                        || error.localizedDescription.localizedCaseInsensitiveContains("wrong time")
                    if retryableODRFailure, attempt < maxAttempts {
                        logDebug(
                            "YOLOE: ODR attempt \(attempt)/\(maxAttempts) failed — \(error) domain=\(ns.domain) code=\(ns.code) — retrying",
                        )
                        try await Task.sleep(nanoseconds: 500_000_000)
                        continue
                    }
                    throw error
                }
            }
            // Fallback for the compiler: the loop should always `return true` or `throw`.
            return resourcesAvailable
        } catch {
            logDebug("YOLOE: ODR download failed: \(error)")
            downloadProgress = 0.0
            statusMessage = L10n.YOLOE.downloadFailed
            progressObservation?.invalidate()
            progressObservation = nil
            throw error
        }
    }

    // MARK: - Model Loading

    private static let yoloeBundledCandidates: [(name: String, ext: String)] = [
        ("yoloe-11l-seg-pf", "mlmodelc"),
        ("yoloe-11l-seg-pf", "mlpackage"),
    ]

    /// Loads YOLOE from the main bundle if present (`mlmodelc` first, then `mlpackage`).
    /// - Returns: `true` if ``model`` was assigned.
    private func loadBundledYoloEIfAvailable(configuration config: MLModelConfiguration) -> Bool {
        for (name, ext) in Self.yoloeBundledCandidates {
            guard let url = Bundle.main.url(forResource: name, withExtension: ext) else { continue }
            do {
                let loadedModel = try MLModel(contentsOf: url, configuration: config)
                model = loadedModel
                logDebug("YOLOE: Loaded '\(name).\(ext)' successfully")
                return true
            } catch {
                logDebug("YOLOE: Failed to load \(name).\(ext): \(error)")
            }
        }
        return false
    }

    /// Internal load routine — prefers the app bundle (reliable); ODR only when the model is hosted as tagged resources.
    private func loadModel() async {
        guard !isLoadingModel && model == nil else { return }
        logDebug("YOLOE: Loading CoreML model…")

        isLoadingModel = true
        statusMessage = L10n.YOLOE.preparingModel

        let config = MLModelConfiguration()
        config.computeUnits = .cpuOnly
        logDebug("YOLOE: Core ML computeUnits=cpuOnly (stable)")

        // 1) Main bundle (shipping default after removing ODR-only tags — avoids flaky streaming unzip).
        if loadBundledYoloEIfAvailable(configuration: config) {
            statusMessage = L10n.YOLOE.ready
            isLoadingModel = false
            return
        }

        // 2) On-demand resources when the model is still delivered only via asset tags.
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
                if ns.domain == NSCocoaErrorDomain, ns.code == 4994 {
                    logDebug("YOLOE: ODR not configured for this bundle, using app bundle")
                } else {
                    logDebug("YOLOE: ODR download failed (\(error)) — will try bundle load anyway")
                }
            }
        }

        statusMessage = L10n.YOLOE.loadingModel

        if loadBundledYoloEIfAvailable(configuration: config) {
            statusMessage = L10n.YOLOE.ready
            isLoadingModel = false
            return
        }

        logDebug("YOLOE: No model variant could be loaded")
        statusMessage = L10n.YOLOE.unavailable
        isLoadingModel = false
    }
}
