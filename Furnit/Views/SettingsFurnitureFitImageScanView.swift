import SwiftUI
import PhotosUI
@preconcurrency import CoreML

@MainActor
struct SettingsFurnitureFitImageScanView: View {
    @ObservedObject private var yoloeService = YOLOEModelService.shared
    @State private var selectedPhotoItem: PhotosPickerItem?
    @State private var selectedImage: UIImage?
    @State private var scanRequestID = UUID()
    @State private var isLoadingSelectedPhoto = false
    @State private var loadErrorMessage: String?

    var body: some View {
        let currentSelectedImage = selectedImage
        let currentScanRequestID = scanRequestID
        let loadedModel = yoloeService.model
        let isModelLoading = yoloeService.isLoadingModel
        let currentStatusText = statusText
        let shouldShowLoadingOverlay = isLoadingSelectedPhoto || loadedModel == nil || isModelLoading

        GeometryReader { proxy in
            VStack(alignment: .leading, spacing: 16) {
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
                                SettingsFurnitureFitStillImageScannerRepresentable(
                                    selectedImage: currentSelectedImage,
                                    scanRequestID: currentScanRequestID,
                                    mlModel: loadedModel
                                )
                                .allowsHitTesting(false)
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
            yoloeService.ensureModelLoaded()
        }
        .onChange(of: selectedPhotoItem) { _, newItem in
            Task {
                await loadSelectedPhoto(from: newItem)
            }
        }
    }

    private var statusText: String {
        if isLoadingSelectedPhoto {
            return L10n.Settings.imageScanLoadingPhoto
        }
        if let message = yoloeService.statusMessage.nilIfEmpty {
            return message
        }
        return L10n.Settings.imageScanPreparingModel
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
