import Foundation
import CoreML

/// Developer-facing loader for RTMDet-Ins-m Core ML experiments.
///
/// This intentionally does not replace the existing RTMDet runtime yet.
/// It only supports the still-image validation path until output parsing
/// and mask quality are proven on device.
@MainActor
final class RTMDetModelService: ObservableObject {

    static let shared = RTMDetModelService()

    private static let rtmdetModelTag = "RTMDetModel"

    @Published var model: MLModel?
    @Published var isLoadingModel = false
    @Published var isDownloadingResources = false
    @Published var downloadProgress: Double = 0.0
    @Published var resourcesAvailable = false
    @Published var statusMessage = ""
    @Published var loadErrorMessage: String?

    private var resourceRequest: NSBundleResourceRequest?
    private var progressObservation: NSKeyValueObservation?

    private init() {}

    private static let bundledCandidates: [(name: String, ext: String)] = [
        ("rtmdet-ins-m", "mlmodelc"),
        ("rtmdet-ins-m", "mlpackage"),
        ("rtmdet_ins_m", "mlmodelc"),
        ("rtmdet_ins_m", "mlpackage"),
        ("rtmdet-ins-m-coreml", "mlmodelc"),
        ("rtmdet-ins-m-coreml", "mlpackage"),
    ]

    private static let computeUnitFallbacks: [MLComputeUnits] = [
        .cpuAndNeuralEngine,
        .cpuAndGPU,
        .all,
        .cpuOnly,
    ]

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

    func releaseResources() {
        progressObservation?.invalidate()
        progressObservation = nil
        resourceRequest?.endAccessingResources()
        resourceRequest = nil
        model = nil
        isLoadingModel = false
        isDownloadingResources = false
        downloadProgress = 0.0
        resourcesAvailable = false
        statusMessage = ""
        loadErrorMessage = nil
        logDebug("RTMDet-Ins-m released model + ODR resources")
    }

    private func loadModel() async {
        guard !isLoadingModel && model == nil else { return }

        isLoadingModel = true
        loadErrorMessage = nil
        statusMessage = "Preparing RTMDet-Ins-m model..."
        defer {
            isLoadingModel = false
        }

        var failures: [String] = []

        await ensureODRReadyIfNeeded()

        for computeUnits in Self.computeUnitFallbacks {
            let config = MLModelConfiguration()
            config.computeUnits = computeUnits

            for subdirectory in Self.bundledSubdirectories {
                for (name, ext) in Self.bundledCandidates {
                    guard let url = Bundle.main.url(forResource: name, withExtension: ext, subdirectory: subdirectory) else {
                        continue
                    }
                    do {
                        let loadURL: URL
                        if ext == "mlpackage" {
                            loadURL = try await MLModel.compileModel(at: url)
                        } else {
                            loadURL = url
                        }
                        let loadedModel = try await Task.detached(priority: .userInitiated) {
                            try MLModel(contentsOf: loadURL, configuration: config)
                        }.value
                        model = loadedModel
                        let location = subdirectory ?? "<bundle-root>"
                        statusMessage = "RTMDet-Ins-m ready (\(computeUnits.debugName), \(location))"
                        logDebug("RTMDet-Ins-m loaded with computeUnits=\(computeUnits.debugName) location=\(location)")
                        loadErrorMessage = nil
                        return
                    } catch {
                        let location = subdirectory ?? "<bundle-root>"
                        failures.append("\(name).\(ext) @ \(location) / \(computeUnits.debugName): \(error.localizedDescription)")
                    }
                }
            }
        }

        statusMessage = "RTMDet-Ins-m model not available"
        loadErrorMessage = failures.isEmpty
            ? "Add a bundled Core ML model named rtmdet-ins-m.mlpackage or rtmdet-ins-m.mlmodelc to the iOS target, preferably under Furnit/Models/RTMDet/."
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
        for subdirectory in Self.bundledSubdirectories {
            for (name, ext) in Self.bundledCandidates {
                if Bundle.main.url(forResource: name, withExtension: ext, subdirectory: subdirectory) != nil {
                    return true
                }
            }
        }
        return false
    }
}

private extension MLComputeUnits {
    var debugName: String {
        switch self {
        case .all:
            return "all"
        case .cpuAndNeuralEngine:
            return "cpuAndNeuralEngine"
        case .cpuAndGPU:
            return "cpuAndGPU"
        case .cpuOnly:
            return "cpuOnly"
        @unknown default:
            return "unknown"
        }
    }
}
