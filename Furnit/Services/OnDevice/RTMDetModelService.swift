import Foundation

/// Owns the single RTMDet-Ins-m LiteRT interpreter used by still-image, live-camera,
/// and room-measurement segmentation. The interpreter requires the Metal delegate;
/// there is deliberately no CPU or Core ML fallback.
@MainActor
final class RTMDetModelService: ObservableObject {

    static let shared = RTMDetModelService()

    private static let rtmdetModelTag = "RTMDetModel"

    @Published var model: RTMDetLiteRuntime?
    @Published var isLoadingModel = false
    @Published var isDownloadingResources = false
    @Published var downloadProgress: Double = 0.0
    @Published var resourcesAvailable = false
    @Published var statusMessage = ""
    @Published var loadErrorMessage: String?

    private var resourceRequest: NSBundleResourceRequest?
    private var progressObservation: NSKeyValueObservation?

    private init() {}

    private static let bundledSubdirectories: [String?] = [
        nil,
        "Models/RTMDet",
        "Furnit/Models/RTMDet",
    ]

    func ensureModelLoaded() {
        guard model == nil && !isLoadingModel else { return }
        Task {
            await loadModel()
        }
    }

    /// Returns the shared runtime, waiting for an in-flight ODR/model load when necessary.
    func modelForInference() async -> RTMDetLiteRuntime? {
        if let model { return model }

        if !isLoadingModel {
            await loadModel()
        } else {
            while isLoadingModel {
                try? await Task.sleep(nanoseconds: 100_000_000)
            }
        }
        return model
    }

    func releaseResources() {
        progressObservation?.invalidate()
        progressObservation = nil
        // Destroy the interpreter before relinquishing its on-demand model file.
        // LiteRT may keep the FlatBuffer memory-mapped for the interpreter lifetime.
        model = nil
        resourceRequest?.endAccessingResources()
        resourceRequest = nil
        isLoadingModel = false
        isDownloadingResources = false
        downloadProgress = 0.0
        resourcesAvailable = false
        statusMessage = ""
        loadErrorMessage = nil
        logDebug("RTMDet-Ins-m released LiteRT Metal runtime + ODR resources")
    }

    private func loadModel() async {
        guard !isLoadingModel && model == nil else { return }

        isLoadingModel = true
        loadErrorMessage = nil
        statusMessage = "Preparing RTMDet-Ins-m model..."
        defer {
            isLoadingModel = false
        }

        await ensureODRReadyIfNeeded()

        var failures: [String] = []
        for subdirectory in Self.bundledSubdirectories {
            guard let url = Bundle.main.url(
                forResource: RTMDetLiteRuntime.modelName,
                withExtension: RTMDetLiteRuntime.modelExtension,
                subdirectory: subdirectory
            ) else {
                continue
            }

            do {
                let loadedModel = try await Task.detached(priority: .userInitiated) {
                    try RTMDetLiteRuntime(modelURL: url)
                }.value
                model = loadedModel
                let location = subdirectory ?? "<bundle-root>"
                statusMessage = "RTMDet-Ins-m ready (LiteRT Metal \(loadedModel.runtimeVersion), \(location))"
                logDebug(
                    "RTMDet-Ins-m loaded with verified full LiteRT Metal delegation "
                        + "\(loadedModel.delegationSummary) location=\(location)"
                )
                loadErrorMessage = nil
                return
            } catch {
                let location = subdirectory ?? "<bundle-root>"
                failures.append("\(url.lastPathComponent) @ \(location): \(error.localizedDescription)")
            }
        }

        statusMessage = "RTMDet-Ins-m model not available"
        loadErrorMessage = failures.isEmpty
            ? "The on-demand rtmdet-ins-m-raw-fp16.tflite model could not be found."
            : failures.joined(separator: "\n")
    }

    private func ensureODRReadyIfNeeded() async {
        if hasBundledModelURL() {
            logDebug("RTMDet-Ins-m embedded in app bundle — skipping ODR")
            return
        }

        if resourcesAvailable { return }

        let tags: Set<String> = [Self.rtmdetModelTag]
        let conditionalRequest = NSBundleResourceRequest(tags: tags)
        conditionalRequest.loadingPriority = NSBundleResourceRequestLoadingPriorityUrgent
        if await conditionalRequest.conditionallyBeginAccessingResources() {
            resourceRequest = conditionalRequest
            resourcesAvailable = true
            downloadProgress = 1.0
            logDebug("RTMDet-Ins-m ODR resources available (conditionallyBeginAccessingResources)")
            return
        }

        guard !isDownloadingResources else {
            while isDownloadingResources {
                try? await Task.sleep(nanoseconds: 100_000_000)
            }
            return
        }

        isDownloadingResources = true
        downloadProgress = 0.0
        statusMessage = "Downloading RTMDet-Ins-m model..."
        defer { isDownloadingResources = false }

        let request = NSBundleResourceRequest(tags: tags)
        request.loadingPriority = NSBundleResourceRequestLoadingPriorityUrgent
        progressObservation = request.progress.observe(\.fractionCompleted, options: [.new]) { [weak self] progress, _ in
            Task { @MainActor in
                self?.downloadProgress = progress.fractionCompleted
                let percent = Int(progress.fractionCompleted * 100)
                self?.statusMessage = "Downloading RTMDet-Ins-m model... \(percent)%"
            }
        }

        do {
            try await FurnitODRBeginAccessing.beginAccessingResources(request)
            resourceRequest = request
            resourcesAvailable = true
            downloadProgress = 1.0
            statusMessage = "RTMDet-Ins-m download complete"
            logDebug("RTMDet-Ins-m ODR download complete")
        } catch {
            progressObservation?.invalidate()
            progressObservation = nil
            downloadProgress = 0.0
            logDebug("RTMDet-Ins-m ODR download failed: \(error)")
        }
    }

    private func hasBundledModelURL() -> Bool {
        Self.bundledSubdirectories.contains { subdirectory in
            Bundle.main.url(
                forResource: RTMDetLiteRuntime.modelName,
                withExtension: RTMDetLiteRuntime.modelExtension,
                subdirectory: subdirectory
            ) != nil
        }
    }
}
