import SwiftUI
import PhotosUI
@preconcurrency import CoreML

@MainActor
struct SettingsFurnitureFitImageScanView: View {
    @ObservedObject private var rtmdetService = RTMDetModelService.shared
    @State private var selectedPhotoItem: PhotosPickerItem?
    @State private var selectedImage: UIImage?
    @State private var scanRequestID = UUID()
    @State private var isLoadingSelectedPhoto = false
    @State private var loadErrorMessage: String?

    var body: some View {
        let currentSelectedImage = selectedImage
        let currentScanRequestID = scanRequestID
        let loadedModel = rtmdetService.model
        let isModelLoading = rtmdetService.isLoadingModel
        let currentStatusText = statusText
        let shouldShowLoadingOverlay = isLoadingSelectedPhoto || loadedModel == nil || isModelLoading

        GeometryReader { proxy in
            // Read the scalar here so the @Sendable PhotosPicker label closure captures a Sendable
            // CGFloat instead of the non-Sendable GeometryProxy.
            let idealImageHeight = proxy.size.height * 0.7
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
                                RTMDetStillImageOverlay(
                                    image: currentSelectedImage,
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
                    .frame(minHeight: 360, idealHeight: idealImageHeight)
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
            rtmdetService.ensureModelLoaded()
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
        if let message = rtmdetService.statusMessage.nilIfEmpty {
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

private struct RTMDetStillImageOverlay: View {
    let image: UIImage
    let scanRequestID: UUID
    let mlModel: MLModel?

    private static let confidenceThreshold: Float = 0.30
    private static let detectionLimit = 30
    private static let areaSeedCount = 3
    private static let groupOverlapTau: CGFloat = 0.15
    private static let classBlacklist = FurnitureFitClassBlacklist()

    private struct StillImageScanResult {
        let mergedMaskImage: UIImage?
        let mergedMaskBoundingBox: CGRect?
        let detections: [FurnitureFitDetection]
        let sourcePixelSize: CGSize
    }

    @State private var result: StillImageScanResult?
    @State private var errorMessage: String?
    @State private var isRunning = false

    var body: some View {
        GeometryReader { proxy in
            ZStack(alignment: .topLeading) {
                if let mergedMaskImage = result?.mergedMaskImage {
                    Image(uiImage: mergedMaskImage)
                        .resizable()
                        .scaledToFit()
                        .frame(width: proxy.size.width, height: proxy.size.height)
                }

                if let result {
                    let layout = imageLayout(in: proxy.size, sourceSize: result.sourcePixelSize)
                    ForEach(Array(result.detections.enumerated()), id: \.offset) { _, detection in
                        let rect = mappedRect(
                            for: detection,
                            layout: layout,
                            sourceSize: result.sourcePixelSize
                        )
                        Rectangle()
                            .path(in: rect)
                            .stroke(Color.orange, lineWidth: 2)

                        Text(labelText(for: detection))
                            .font(.caption2.monospacedDigit())
                            .padding(.horizontal, 6)
                            .padding(.vertical, 3)
                            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 6, style: .continuous))
                            .position(x: rect.minX + 52, y: max(layout.origin.y + 10, rect.minY - 10))
                    }

                    if let mergedMaskBoundingBox = result.mergedMaskBoundingBox {
                        let rect = mappedRect(
                            for: mergedMaskBoundingBox,
                            layout: layout,
                            sourceSize: result.sourcePixelSize
                        )
                        Rectangle()
                            .path(in: rect)
                            .stroke(Color.green, lineWidth: 3)
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
                } else if result != nil {
                    EmptyView()
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
            let debugMode = AppStateManager.shared.qualitySettings.debugMode
            Self.classBlacklist.loadBlacklistOnce(debugMode: debugMode) { message in
                if debugMode { print(message) }
            }
            let inferenceResult = try RTMDetImageInference.runInstanceSegmentation(
                image: image,
                model: mlModel,
                confidenceThreshold: Self.confidenceThreshold,
                classBlacklist: Self.classBlacklist.ignoredIndices,
                allowedClassIndices: nil,
                maxMaskCount: Self.detectionLimit,
                maxDetectionCount: Self.detectionLimit,
                buildInstanceMasks: true,
                restrictInstanceMasksToFrameCenter: false,
                debug: debugMode
            )
            let scanResult = buildStillImageScanResult(from: inferenceResult)
            await MainActor.run {
                result = scanResult
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

    private func buildStillImageScanResult(from inferenceResult: RTMDetInferenceResult) -> StillImageScanResult {
        let sourcePixelSize = sourcePixelSize(from: inferenceResult) ?? fallbackSourcePixelSize
        guard let groupedIndices = largestOverlapClusterIndices(for: inferenceResult.detections) else {
            return StillImageScanResult(
                mergedMaskImage: nil,
                mergedMaskBoundingBox: nil,
                detections: inferenceResult.detections,
                sourcePixelSize: sourcePixelSize
            )
        }

        logSelectedMaskAlignmentDiagnostics(
            outputSummary: inferenceResult.outputSummary,
            selectedIndices: groupedIndices
        )
        let groupedMasks = groupedIndices.compactMap { index -> UIImage? in
            index < inferenceResult.instanceMaskImages.count ? inferenceResult.instanceMaskImages[index] : nil
        }
        let mergedMaskImage = combinedInstanceMaskImage(groupedMasks)
        let mergedMaskBoundingBox = mergedMaskImage.flatMap { alphaBoundingBox(in: $0) }
        return StillImageScanResult(
            mergedMaskImage: mergedMaskImage,
            mergedMaskBoundingBox: mergedMaskBoundingBox,
            detections: inferenceResult.detections,
            sourcePixelSize: sourcePixelSize
        )
    }

    private func logSelectedMaskAlignmentDiagnostics(
        outputSummary: [String],
        selectedIndices: [Int]
    ) {
        guard AppStateManager.shared.qualitySettings.debugMode else { return }
        let selectedIndexSet = Set(selectedIndices)
        for line in outputSummary where line.hasPrefix("rawMaskPlaneAlign[") {
            guard let closeBracket = line.firstIndex(of: "]") else { continue }
            let rawIndex = line[line.index(line.startIndex, offsetBy: "rawMaskPlaneAlign[".count)..<closeBracket]
            guard let index = Int(rawIndex), selectedIndexSet.contains(index) else { continue }
            print("🧪 [ImageScan MASK_ALIGN selected] \(line)")
        }
    }

    private var fallbackSourcePixelSize: CGSize {
        if let cgImage = image.cgImage {
            return CGSize(width: cgImage.width, height: cgImage.height)
        }
        return CGSize(width: max(1, image.size.width), height: max(1, image.size.height))
    }

    private func sourcePixelSize(from inferenceResult: RTMDetInferenceResult) -> CGSize? {
        let images = [inferenceResult.overlayMaskImage] + inferenceResult.instanceMaskImages
        for image in images {
            if let cgImage = image?.cgImage {
                return CGSize(width: cgImage.width, height: cgImage.height)
            }
        }
        return nil
    }

    private func combinedInstanceMaskImage(_ masks: [UIImage]) -> UIImage? {
        guard let firstMask = masks.first else { return nil }
        guard masks.count > 1 else { return firstMask }
        let format = UIGraphicsImageRendererFormat.default()
        format.opaque = false
        format.scale = firstMask.scale
        let renderer = UIGraphicsImageRenderer(size: firstMask.size, format: format)
        return renderer.image { _ in
            let bounds = CGRect(origin: .zero, size: firstMask.size)
            for mask in masks {
                mask.draw(in: bounds)
            }
        }
    }

    private func alphaBoundingBox(in image: UIImage) -> CGRect? {
        guard let cgImage = image.cgImage else { return nil }
        let width = cgImage.width
        let height = cgImage.height
        guard width > 0, height > 0 else { return nil }

        var rgba = [UInt8](repeating: 0, count: width * height * 4)
        let drewImage = rgba.withUnsafeMutableBytes { buffer in
            guard let baseAddress = buffer.baseAddress,
                  let context = CGContext(
                    data: baseAddress,
                    width: width,
                    height: height,
                    bitsPerComponent: 8,
                    bytesPerRow: width * 4,
                    space: CGColorSpaceCreateDeviceRGB(),
                    bitmapInfo: CGBitmapInfo.byteOrder32Big
                        .union(CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue))
                        .rawValue
                  ) else {
                return false
            }
            context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
            return true
        }
        guard drewImage else { return nil }

        var minX = width
        var minY = height
        var maxX = -1
        var maxY = -1

        for y in 0..<height {
            let rowOffset = y * width * 4
            for x in 0..<width {
                let alpha = rgba[rowOffset + x * 4 + 3]
                guard alpha > 0 else { continue }
                minX = min(minX, x)
                minY = min(minY, y)
                maxX = max(maxX, x)
                maxY = max(maxY, y)
            }
        }

        guard maxX >= minX, maxY >= minY else { return nil }
        return CGRect(
            x: CGFloat(minX),
            y: CGFloat(minY),
            width: CGFloat(maxX - minX + 1),
            height: CGFloat(maxY - minY + 1)
        )
    }

    private func imageLayout(
        in containerSize: CGSize,
        sourceSize: CGSize
    ) -> (origin: CGPoint, size: CGSize) {
        guard sourceSize.width > 0, sourceSize.height > 0 else {
            return (.zero, containerSize)
        }
        let scale = min(containerSize.width / sourceSize.width, containerSize.height / sourceSize.height)
        let width = sourceSize.width * scale
        let height = sourceSize.height * scale
        let origin = CGPoint(
            x: (containerSize.width - width) * 0.5,
            y: (containerSize.height - height) * 0.5
        )
        return (origin, CGSize(width: width, height: height))
    }

    private func mappedRect(
        for imageRect: CGRect,
        layout: (origin: CGPoint, size: CGSize),
        sourceSize: CGSize
    ) -> CGRect {
        let x = layout.origin.x + imageRect.minX * layout.size.width / sourceSize.width
        let y = layout.origin.y + imageRect.minY * layout.size.height / sourceSize.height
        let width = imageRect.width * layout.size.width / sourceSize.width
        let height = imageRect.height * layout.size.height / sourceSize.height
        return CGRect(x: x, y: y, width: width, height: height)
    }

    private func mappedRect(
        for detection: FurnitureFitDetection,
        layout: (origin: CGPoint, size: CGSize),
        sourceSize: CGSize
    ) -> CGRect {
        mappedRect(
            for: detection.boundingBox,
            layout: layout,
            sourceSize: sourceSize
        )
    }

    private func labelText(for detection: FurnitureFitDetection) -> String {
        "#\(detection.classIdx) \(Int(detection.confidence * 100))%"
    }

    private func largestOverlapClusterIndices(for detections: [FurnitureFitDetection]) -> [Int]? {
        guard !detections.isEmpty else { return nil }

        let seedIndices = detections.indices
            .sorted {
                let lhsArea = detections[$0].boundingBox.width * detections[$0].boundingBox.height
                let rhsArea = detections[$1].boundingBox.width * detections[$1].boundingBox.height
                return lhsArea > rhsArea
            }
            .prefix(Self.areaSeedCount)
        var bestCluster: [Int] = []
        var bestClusterArea: CGFloat = -1

        for seedIndex in seedIndices {
            var cluster = [seedIndex]
            var inCluster: Set<Int> = [seedIndex]
            var frontier = [seedIndex]

            while let currentIndex = frontier.popLast() {
                let currentBox = detections[currentIndex].boundingBox
                for candidateIndex in detections.indices where !inCluster.contains(candidateIndex) {
                    let candidateBox = detections[candidateIndex].boundingBox
                    guard overlapCoefficient(currentBox, candidateBox) >= Self.groupOverlapTau else {
                        continue
                    }
                    inCluster.insert(candidateIndex)
                    cluster.append(candidateIndex)
                    frontier.append(candidateIndex)
                }
            }

            let clusterArea = unionRectArea(for: cluster, detections: detections)
            if clusterArea > bestClusterArea {
                bestClusterArea = clusterArea
                bestCluster = cluster
            }
        }

        return bestCluster.isEmpty ? nil : bestCluster
    }

    private func unionRectArea(for indices: [Int], detections: [FurnitureFitDetection]) -> CGFloat {
        let unionRect = indices
            .map { detections[$0].boundingBox }
            .reduce(CGRect.null) { $0.union($1) }
        guard !unionRect.isNull else { return 0 }
        return unionRect.width * unionRect.height
    }

    private func overlapCoefficient(_ a: CGRect, _ b: CGRect) -> CGFloat {
        let ix = max(a.minX, b.minX)
        let iy = max(a.minY, b.minY)
        let iw = min(a.maxX, b.maxX) - ix
        let ih = min(a.maxY, b.maxY) - iy
        guard iw > 0, ih > 0 else { return 0 }

        let intersection = iw * ih
        let minArea = min(a.width * a.height, b.width * b.height)
        return minArea > 0 ? intersection / minArea : 0
    }
}

private extension String {
    var nilIfEmpty: String? {
        isEmpty ? nil : self
    }
}
