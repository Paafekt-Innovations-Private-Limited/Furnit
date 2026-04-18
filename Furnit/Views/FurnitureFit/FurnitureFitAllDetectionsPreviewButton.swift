import SwiftUI
import UIKit

enum FurnitureFitDebugPreviewStore {
    static let allDetectionsFilename = "alchair_furniturefit_all_detections.png"

    static func allDetectionsImageURL() -> URL? {
        let fileManager = FileManager.default
        let documentsDirectory = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first
        let candidateURLs = [
            documentsDirectory?.appendingPathComponent(allDetectionsFilename),
            documentsDirectory?.appendingPathComponent("test_images/\(allDetectionsFilename)"),
        ].compactMap { $0 }

        for candidateURL in candidateURLs where fileManager.fileExists(atPath: candidateURL.path) {
            return candidateURL
        }
        return nil
    }

    static func loadAllDetectionsImage() -> UIImage? {
        guard let imageURL = allDetectionsImageURL() else { return nil }
        return UIImage(contentsOfFile: imageURL.path)
    }
}

struct FurnitureFitAllDetectionsPreviewButton: View {
    @State private var previewImage: UIImage?
    @State private var showingPreview = false
    @State private var showingMissingAlert = false

    var body: some View {
        Button(action: presentPreviewIfAvailable) {
            Image(systemName: "photo.badge.magnifyingglass")
                .font(.system(size: 16, weight: .semibold))
                .foregroundColor(.white)
                .frame(width: 34, height: 34)
                .background(Circle().fill(Color.black.opacity(0.72)).shadow(radius: 3))
        }
        .buttonStyle(.plain)
        .accessibilityLabel("Show all detections image")
        .sheet(isPresented: $showingPreview) {
            if let previewImage {
                FurnitureFitAllDetectionsPreviewSheet(image: previewImage)
            }
        }
        .alert("Detections image not found", isPresented: $showingMissingAlert) {
            Button("OK", role: .cancel) {}
        } message: {
            Text("Run the one-image brain debug first so `alchair_furniturefit_all_detections.png` is saved.")
        }
    }

    private func presentPreviewIfAvailable() {
        guard let loadedImage = FurnitureFitDebugPreviewStore.loadAllDetectionsImage() else {
            showingMissingAlert = true
            return
        }
        previewImage = loadedImage
        showingPreview = true
    }
}

private struct FurnitureFitAllDetectionsPreviewSheet: View {
    let image: UIImage
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            ZStack {
                Color.black.ignoresSafeArea()
                ScrollView([.horizontal, .vertical]) {
                    Image(uiImage: image)
                        .resizable()
                        .scaledToFit()
                        .frame(maxWidth: .infinity)
                        .padding()
                }
            }
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") { dismiss() }
                }
            }
            .navigationTitle("All Detections")
            .navigationBarTitleDisplayMode(.inline)
        }
    }
}
