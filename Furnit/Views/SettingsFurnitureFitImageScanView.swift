import SwiftUI
import PhotosUI
import CoreML

struct SettingsFurnitureFitImageScanView: View {
    @ObservedObject private var yoloeService = YOLOEModelService.shared
    @State private var selectedPhotoItem: PhotosPickerItem?
    @State private var selectedImage: UIImage?
    @State private var scanRequestID = UUID()
    @State private var isLoadingSelectedPhoto = false
    @State private var loadErrorMessage: String?

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                PhotosPicker(selection: $selectedPhotoItem, matching: .images) {
                    ZStack {
                        RoundedRectangle(cornerRadius: 18, style: .continuous)
                            .fill(Color.secondary.opacity(0.14))

                        if let selectedImage {
                            Image(uiImage: selectedImage)
                                .resizable()
                                .scaledToFit()
                                .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))

                            if yoloeService.model != nil {
                                SettingsFurnitureFitStillImageScannerRepresentable(
                                    selectedImage: selectedImage,
                                    scanRequestID: scanRequestID,
                                    mlModel: yoloeService.model
                                )
                                .allowsHitTesting(false)
                                .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
                            }
                        } else {
                            VStack(spacing: 12) {
                                Image(systemName: "photo.badge.plus")
                                    .font(.system(size: 34))
                                    .foregroundStyle(.secondary)
                                Text("Tap to choose a photo")
                                    .font(.headline)
                                Text("The selected image is scanned with the Furniture Fit pipeline.")
                                    .font(.subheadline)
                                    .foregroundStyle(.secondary)
                                    .multilineTextAlignment(.center)
                            }
                            .padding(24)
                        }

                        if isLoadingSelectedPhoto || yoloeService.model == nil || yoloeService.isLoadingModel {
                            VStack(spacing: 10) {
                                ProgressView()
                                Text(statusText)
                                    .font(.footnote)
                                    .foregroundStyle(.secondary)
                            }
                            .padding(.horizontal, 16)
                            .padding(.vertical, 12)
                            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
                        }
                    }
                    .frame(maxWidth: .infinity)
                    .frame(height: 320)
                }
                .buttonStyle(.plain)

                Text("Tap the preview to pick a photo from your library. Bounding boxes on the preview include the class name, class ID, and confidence.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)

                if let loadErrorMessage {
                    Text(loadErrorMessage)
                        .font(.footnote)
                        .foregroundStyle(.red)
                }
            }
            .padding()
        }
        .navigationTitle("Image Scan")
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
            return "Loading photo…"
        }
        if let message = yoloeService.statusMessage.nilIfEmpty {
            return message
        }
        return "Preparing detection model…"
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
                    loadErrorMessage = "Could not load the selected photo."
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
