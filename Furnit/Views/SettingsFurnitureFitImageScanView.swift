import SwiftUI
import PhotosUI
@preconcurrency import CoreML

@MainActor
struct SettingsFurnitureFitImageScanView: View {
    private enum ScanBackend: String, CaseIterable, Identifiable {
        case rtmdetInsM = "RTMDet-Ins-m"
        case yoloe = "YOLOE"

        var id: String { rawValue }
    }

    @ObservedObject private var yoloeService = YOLOEModelService.shared
    @ObservedObject private var rtmdetService = RTMDetModelService.shared
    @State private var selectedBackend: ScanBackend = .rtmdetInsM
    @State private var selectedPhotoItem: PhotosPickerItem?
    @State private var selectedImage: UIImage?
    @State private var scanRequestID = UUID()
    @State private var isLoadingSelectedPhoto = false
    @State private var loadErrorMessage: String?

    var body: some View {
        let currentSelectedImage = selectedImage
        let currentScanRequestID = scanRequestID
        let loadedModel = currentModel
        let isModelLoading = currentIsModelLoading
        let currentStatusText = statusText
        let shouldShowLoadingOverlay = isLoadingSelectedPhoto || loadedModel == nil || isModelLoading

        GeometryReader { proxy in
            VStack(alignment: .leading, spacing: 16) {
                Picker("Backend", selection: $selectedBackend) {
                    ForEach(ScanBackend.allCases) { backend in
                        Text(backend.rawValue).tag(backend)
                    }
                }
                .pickerStyle(.segmented)

                PhotosPicker(selection: $selectedPhotoItem, matching: .images) {
                    ZStack {
                        RoundedRectangle(cornerRadius: 18, style: .continuous)
                            .fill(Color.secondary.opacity(0.14))

                        if let currentSelectedImage {
                            Image(uiImage: currentSelectedImage)
                                .resizable()
                                .scaledToFit()
                                .frame(maxWidth: .infinity, maxHeight: .infinity)
                                .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))

                            if loadedModel != nil {
                                overlayView(
                                    image: currentSelectedImage,
                                    scanRequestID: currentScanRequestID,
                                    mlModel: loadedModel
                                )
                                .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
                            }
                        } else {
                            VStack(spacing: 12) {
                                Image(systemName: "photo.badge.plus")
                                    .font(.system(size: 34))
                                    .foregroundStyle(.secondary)
                                Text(L10n.Settings.imageScanTapToChoose)
                                    .font(.headline)
                                Text(L10n.Settings.imageScanTapToChooseSubtitle)
                                    .font(.subheadline)
                                    .foregroundStyle(.secondary)
                                    .multilineTextAlignment(.center)
                            }
                            .padding(24)
                        }

                        if shouldShowLoadingOverlay {
                            VStack(spacing: 10) {
                                ProgressView()
                                Text(currentStatusText)
                                    .font(.footnote)
                                    .foregroundStyle(.secondary)
                            }
                            .padding(.horizontal, 16)
                            .padding(.vertical, 12)
                            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
                        }
                    }
                    .frame(maxWidth: .infinity)
                    .frame(minHeight: 360, idealHeight: proxy.size.height * 0.7)
                }
                .buttonStyle(.plain)

                Text(L10n.Settings.imageScanFootnote)
                    .font(.footnote)
                    .foregroundStyle(.secondary)

                if let loadErrorMessage {
                    Text(loadErrorMessage)
                        .font(.footnote)
                        .foregroundStyle(.red)
                }
            }
            .padding()
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
        }
        .navigationTitle(L10n.Settings.imageScan)
        .navigationBarTitleDisplayMode(.inline)
        .task {
            ensureSelectedModelLoaded()
        }
        .onChange(of: selectedBackend) { _, _ in
            ensureSelectedModelLoaded()
        }
        .onChange(of: selectedPhotoItem) { _, newItem in
            Task {
                await loadSelectedPhoto(from: newItem)
            }
        }
    }

    @ViewBuilder
    private func overlayView(image: UIImage, scanRequestID: UUID, mlModel: MLModel?) -> some View {
        switch selectedBackend {
        case .yoloe:
            SettingsFurnitureFitStillImageScannerRepresentable(
                selectedImage: image,
                scanRequestID: scanRequestID,
                mlModel: mlModel
            )
            .allowsHitTesting(false)
        case .rtmdetInsM:
            RTMDetStillImageOverlay(
                image: image,
                scanRequestID: scanRequestID,
                mlModel: mlModel
            )
            .allowsHitTesting(false)
        }
    }

    private var currentModel: MLModel? {
        switch selectedBackend {
        case .yoloe:
            return yoloeService.model
        case .rtmdetInsM:
            return rtmdetService.model
        }
    }

    private var currentIsModelLoading: Bool {
        switch selectedBackend {
        case .yoloe:
            return yoloeService.isLoadingModel
        case .rtmdetInsM:
            return rtmdetService.isLoadingModel
        }
    }

    private var statusText: String {
        if isLoadingSelectedPhoto {
            return L10n.Settings.imageScanLoadingPhoto
        }
        let serviceMessage: String?
        switch selectedBackend {
        case .yoloe:
            serviceMessage = yoloeService.statusMessage.nilIfEmpty
        case .rtmdetInsM:
            serviceMessage = rtmdetService.statusMessage.nilIfEmpty
        }
        if let message = serviceMessage {
            return message
        }
        return L10n.Settings.imageScanPreparingModel
    }

    private func ensureSelectedModelLoaded() {
        switch selectedBackend {
        case .yoloe:
            yoloeService.ensureModelLoaded()
        case .rtmdetInsM:
            rtmdetService.ensureModelLoaded()
        }
    }

    private func loadSelectedPhoto(from item: PhotosPickerItem?) async {
        guard let item else { return }
        await MainActor.run {
            isLoadingSelectedPhoto = true
            loadErrorMessage = nil
        }

        do {
            guard let imageData = try await item.loadTransferable(type: Data.self),
                  let uiImage = UIImage(data: imageData) else {
                await MainActor.run {
                    isLoadingSelectedPhoto = false
                    loadErrorMessage = L10n.Settings.imageScanLoadFailed
                }
                return
            }

            await MainActor.run {
                selectedImage = uiImage
                scanRequestID = UUID()
                isLoadingSelectedPhoto = false
            }
        } catch {
            await MainActor.run {
                isLoadingSelectedPhoto = false
                loadErrorMessage = error.localizedDescription
            }
        }
    }
}

private struct RTMDetStillImageOverlay: View {
    let image: UIImage
    let scanRequestID: UUID
    let mlModel: MLModel?

    @State private var result: RTMDetInferenceResult?
    @State private var errorMessage: String?
    @State private var isRunning = false

    var body: some View {
        GeometryReader { proxy in
            ZStack(alignment: .topLeading) {
                if let overlayMaskImage = result?.overlayMaskImage {
                    Image(uiImage: overlayMaskImage)
                        .resizable()
                        .scaledToFit()
                        .frame(width: proxy.size.width, height: proxy.size.height)
                }

                if let detections = result?.detections {
                    let layout = imageLayout(in: proxy.size)
                    ForEach(Array(detections.enumerated()), id: \.offset) { index, detection in
                        let rect = mappedRect(for: detection, layout: layout)
                        Rectangle()
                            .path(in: rect)
                            .stroke(index == 0 ? Color.green : Color.orange, lineWidth: index == 0 ? 3 : 2)

                        Text(labelText(for: detection))
                            .font(.caption2.monospacedDigit())
                            .padding(.horizontal, 6)
                            .padding(.vertical, 3)
                            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 6, style: .continuous))
                            .position(x: rect.minX + 52, y: max(layout.origin.y + 10, rect.minY - 10))
                    }
                }

                if isRunning {
                    ProgressView()
                        .padding(10)
                        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 10, style: .continuous))
                        .padding(12)
                } else if let errorMessage {
                    Text(errorMessage)
                        .font(.caption)
                        .foregroundStyle(.red)
                        .padding(10)
                        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 10, style: .continuous))
                        .padding(12)
                } else if let result {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Detections: \(result.detections.count)")
                            .font(.caption.bold())
                        ForEach(result.outputSummary.prefix(3), id: \.self) { line in
                            Text(line)
                                .font(.caption2.monospaced())
                        }
                    }
                    .padding(10)
                    .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 10, style: .continuous))
                    .padding(12)
                }
            }
            .frame(width: proxy.size.width, height: proxy.size.height)
        }
        .task(id: scanRequestID) {
            await runInference()
        }
    }

    private func runInference() async {
        guard let mlModel else {
            await MainActor.run {
                result = nil
                errorMessage = "RTMDet model not loaded"
                isRunning = false
            }
            return
        }

        await MainActor.run {
            isRunning = true
            errorMessage = nil
        }

        do {
            let inferenceResult = try RTMDetImageInference.runInstanceSegmentation(image: image, model: mlModel)
            await MainActor.run {
                result = inferenceResult
                errorMessage = nil
                isRunning = false
            }
        } catch {
            await MainActor.run {
                result = nil
                errorMessage = error.localizedDescription
                isRunning = false
            }
        }
    }

    private func imageLayout(in containerSize: CGSize) -> (origin: CGPoint, size: CGSize) {
        guard image.size.width > 0, image.size.height > 0 else {
            return (.zero, containerSize)
        }
        let scale = min(containerSize.width / image.size.width, containerSize.height / image.size.height)
        let width = image.size.width * scale
        let height = image.size.height * scale
        let origin = CGPoint(
            x: (containerSize.width - width) * 0.5,
            y: (containerSize.height - height) * 0.5
        )
        return (origin, CGSize(width: width, height: height))
    }

    private func mappedRect(for detection: FurnitureFitDetection, layout: (origin: CGPoint, size: CGSize)) -> CGRect {
        let x = layout.origin.x + CGFloat(detection.x - detection.w * 0.5) * layout.size.width / image.size.width
        let y = layout.origin.y + CGFloat(detection.y - detection.h * 0.5) * layout.size.height / image.size.height
        let width = CGFloat(detection.w) * layout.size.width / image.size.width
        let height = CGFloat(detection.h) * layout.size.height / image.size.height
        return CGRect(x: x, y: y, width: width, height: height)
    }

    private func labelText(for detection: FurnitureFitDetection) -> String {
        "#\(detection.classIdx) \(Int(detection.confidence * 100))%"
    }
}

private struct SettingsFurnitureFitStillImageScannerRepresentable: UIViewRepresentable {
    let selectedImage: UIImage
    let scanRequestID: UUID
    let mlModel: MLModel?

    @AppStorage("furnitureFit.primaryDetectionMinConfidence") private var primaryDetectionMinConfidenceStorage: Double = 0.57
    @AppStorage("furnitureFit.primarySelectionByHighestConfidence") private var primarySelectionByHighestConfidence: Bool = false

    func makeUIView(context: Context) -> FurnitureFitContainerView {
        let view = FurnitureFitContainerView()
        view.backgroundColor = .clear
        view.stillImageScanModeEnabled = true
        applyConfiguration(to: view)
        return view
    }

    func updateUIView(_ uiView: FurnitureFitContainerView, context: Context) {
        applyConfiguration(to: uiView)
        uiView.submitStillImageForScanning(selectedImage, requestID: scanRequestID)
        uiView.startIfNeeded()
    }

    static func dismantleUIView(_ uiView: FurnitureFitContainerView, coordinator: ()) {
        uiView.stop()
    }

    private func applyConfiguration(to view: FurnitureFitContainerView) {
        view.setModel(mlModel)
        view.processInterval = 0.07
        view.confidenceThreshold = 0.10
        view.primaryDetectionMinConfidence = Float(min(max(primaryDetectionMinConfidenceStorage, 0.05), 0.99))
        view.primarySelectionByHighestConfidence = primarySelectionByHighestConfidence
        view.useBilinearUpscaling = true
        view.stillImageScanModeEnabled = true
        view.showFullVideoWithIdentifications = false
        view.showIdentifyLivePreview = false
        view.segmentationMode = .identifyOnly
        view.suppressStartupProgress = false
        view.arAssistedSizingEnabled = false
    }
}

private extension String {
    var nilIfEmpty: String? {
        isEmpty ? nil : self
    }
}
