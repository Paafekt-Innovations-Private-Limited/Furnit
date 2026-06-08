import Foundation
import CoreML

/// Developer-facing loader for RTMDet-Ins-m Core ML experiments.
///
/// This intentionally does not replace the existing YOLOE runtime yet.
/// It only supports the still-image validation path until output parsing
/// and mask quality are proven on device.
@MainActor
final class RTMDetModelService: ObservableObject {

    static let shared = RTMDetModelService()

    @Published var model: MLModel?
    @Published var isLoadingModel = false
    @Published var statusMessage = ""
    @Published var loadErrorMessage: String?

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
        model = nil
        isLoadingModel = false
        statusMessage = ""
        loadErrorMessage = nil
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

        for computeUnits in Self.computeUnitFallbacks {
            let config = MLModelConfiguration()
            config.computeUnits = computeUnits

            for subdirectory in Self.bundledSubdirectories {
                for (name, ext) in Self.bundledCandidates {
                    guard let url = Bundle.main.url(forResource: name, withExtension: ext, subdirectory: subdirectory) else {
                        continue
                    }
                    do {
                        model = try MLModel(contentsOf: url, configuration: config)
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
