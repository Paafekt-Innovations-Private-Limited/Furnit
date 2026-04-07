import Foundation
import ImageIO
import Photos

/// Writes `camera_exif.json` next to SHARP outputs for [WallMeasurementEstimator]: focal lengths + **SubjectDistance** (m).
///
/// Library picks often have **no** `imageURL`; we use **PHAsset** image data (preserves EXIF) when available.
/// In-app camera uses `UIImagePickerController` **mediaMetadata** (`{Exif}`).
enum CameraExifSidecar {
    private static let fileName = "camera_exif.json"
    private static let stemSuffix = "_camera_exif.json"

    /// Merge order: file URL → Photo Library asset data → camera `mediaMetadata` (later keys fill gaps).
    static func writeMerged(
        roomFolder: URL,
        imageURL: URL?,
        mediaMetadata: [AnyHashable: Any]?,
        photoLibraryAssetLocalId: String?,
    ) async {
        let merged = await collectMerged(
            imageURL: imageURL,
            mediaMetadata: mediaMetadata,
            photoLibraryAssetLocalId: photoLibraryAssetLocalId
        )
        if merged["focalLengthMm"] == nil, merged["focalLengthPx"] == nil {
            logWallMeasurement(
                "camera_exif_sidecar merged missing focal fields folder=\(roomFolder.lastPathComponent) " +
                    "note=original photo EXIF may be stripped (e.g. screenshot or chat-app export)"
            )
        }
        guard !merged.isEmpty else {
            logWallMeasurement(
                "camera_exif_sidecar skip (no EXIF) folder=\(roomFolder.lastPathComponent) " +
                    "hasURL=\(imageURL != nil) hasPHAsset=\(!(photoLibraryAssetLocalId ?? "").isEmpty) hasMeta=\(mediaMetadata != nil)",
            )
            return
        }
        persist(roomFolder: roomFolder, additions: merged)
    }

    static func collectMerged(
        imageURL: URL?,
        mediaMetadata: [AnyHashable: Any]?,
        photoLibraryAssetLocalId: String?,
    ) async -> [String: Double] {
        logWallMeasurement(
            "camera_exif_sidecar collect begin " +
                "hasURL=\(imageURL != nil) hasPHAsset=\(!(photoLibraryAssetLocalId ?? "").isEmpty) hasMeta=\(mediaMetadata != nil)"
        )
        var merged: [String: Double] = [:]
        func absorb(_ other: [String: Double]) {
            for (key, value) in other where merged[key] == nil {
                merged[key] = value
            }
        }
        if let u = imageURL {
            let extracted = extractFromImageSource(url: u)
            logWallMeasurement("camera_exif_sidecar write source=url file=\(u.lastPathComponent) keys=\(extracted.keys.sorted())")
            absorb(extracted)
        }
        if let id = photoLibraryAssetLocalId, !id.isEmpty {
            let fromAsset = await extractFromPhotoLibraryAsset(localIdentifier: id)
            logWallMeasurement("camera_exif_sidecar write source=phasset id=\(id) keys=\(fromAsset.keys.sorted())")
            absorb(fromAsset)
            if fromAsset.isEmpty {
                logWallMeasurement("camera_exif_sidecar phAsset id=\(id) extracted no fields (iCloud/network?)")
            }
        }
        let cameraMetadata = extractFromCameraMetadata(mediaMetadata)
        if !cameraMetadata.isEmpty {
            logWallMeasurement("camera_exif_sidecar write source=camera_metadata keys=\(cameraMetadata.keys.sorted())")
        }
        absorb(cameraMetadata)
        logWallMeasurement("camera_exif_sidecar merged keys=\(merged.keys.sorted())")
        return merged
    }

    // MARK: - Photo Library (full file bytes + EXIF)

    private static func extractFromPhotoLibraryAsset(localIdentifier: String) async -> [String: Double] {
        await withCheckedContinuation { continuation in
            let fetch = PHAsset.fetchAssets(withLocalIdentifiers: [localIdentifier], options: nil)
            if fetch.count > 1 {
                logWallMeasurement("camera_exif_sidecar phAsset fetch returned \(fetch.count) matches id=\(localIdentifier)")
            }
            guard let asset = fetch.firstObject else {
                logWallMeasurement("camera_exif_sidecar phAsset fetch empty id=\(localIdentifier)")
                continuation.resume(returning: [:])
                return
            }
            let opts = PHImageRequestOptions()
            opts.isNetworkAccessAllowed = true
            opts.deliveryMode = .highQualityFormat
            opts.version = .current
            opts.isSynchronous = false
            var resumed = false
            PHImageManager.default().requestImageDataAndOrientation(for: asset, options: opts) { data, _, _, info in
                guard !resumed else {
                    logWallMeasurement("camera_exif_sidecar phAsset duplicate callback ignored id=\(localIdentifier)")
                    return
                }
                resumed = true
                if data == nil {
                    let cancelled = (info?[PHImageCancelledKey] as? Bool) == true
                    logWallMeasurement("camera_exif_sidecar phAsset data nil id=\(localIdentifier) cancelled=\(cancelled)")
                }
                guard let data else {
                    continuation.resume(returning: [:])
                    return
                }
                Task {
                    var extracted = extractFromImageData(data)
                    logWallMeasurement(
                        "camera_exif_sidecar phAsset extracted id=\(localIdentifier) bytes=\(data.count) keys=\(extracted.keys.sorted())"
                    )

                    if extracted["focalLengthMm"] == nil, extracted["focalLength35mmEquivMm"] == nil {
                        let fromContentInput = await extractFromPhotoLibraryContentEditingInput(asset: asset)
                        if !fromContentInput.isEmpty {
                            logWallMeasurement(
                                "camera_exif_sidecar phAsset content_edit_input extracted id=\(localIdentifier) keys=\(fromContentInput.keys.sorted())"
                            )
                            for (key, value) in fromContentInput where extracted[key] == nil {
                                extracted[key] = value
                            }
                        }
                    }

                    if extracted["focalLengthMm"] == nil, extracted["focalLength35mmEquivMm"] == nil {
                        let fromResource = await extractFromPhotoLibraryAssetResource(asset: asset)
                        if !fromResource.isEmpty {
                            logWallMeasurement(
                                "camera_exif_sidecar phAsset resource extracted id=\(localIdentifier) keys=\(fromResource.keys.sorted())"
                            )
                            for (key, value) in fromResource where extracted[key] == nil {
                                extracted[key] = value
                            }
                        }
                    }

                    continuation.resume(returning: extracted)
                }
            }
        }
    }

    private static func extractFromPhotoLibraryContentEditingInput(asset: PHAsset) async -> [String: Double] {
        await withCheckedContinuation { continuation in
            let options = PHContentEditingInputRequestOptions()
            options.isNetworkAccessAllowed = true
            asset.requestContentEditingInput(with: options) { input, _ in
                guard let url = input?.fullSizeImageURL else {
                    continuation.resume(returning: [:])
                    return
                }
                let extracted = extractFromImageSource(url: url)
                logWallMeasurement("camera_exif_sidecar content_edit_input url=\(url.lastPathComponent) keys=\(extracted.keys.sorted())")
                continuation.resume(returning: extracted)
            }
        }
    }

    private static func extractFromPhotoLibraryAssetResource(asset: PHAsset) async -> [String: Double] {
        let resources = PHAssetResource.assetResources(for: asset)
        guard let resource = resources.first(where: {
            $0.type == .photo || $0.type == .fullSizePhoto || $0.type == .alternatePhoto
        }) ?? resources.first else {
            logWallMeasurement("camera_exif_sidecar asset_resource none")
            return [:]
        }

        return await withCheckedContinuation { continuation in
            let options = PHAssetResourceRequestOptions()
            options.isNetworkAccessAllowed = true
            var data = Data()
            PHAssetResourceManager.default().requestData(for: resource, options: options) { chunk in
                data.append(chunk)
            } completionHandler: { error in
                if let error {
                    logWallMeasurement("camera_exif_sidecar asset_resource read failed name=\(resource.originalFilename) error=\(error.localizedDescription)")
                    continuation.resume(returning: [:])
                    return
                }
                let extracted = extractFromImageData(data)
                logWallMeasurement(
                    "camera_exif_sidecar asset_resource read ok name=\(resource.originalFilename) bytes=\(data.count) keys=\(extracted.keys.sorted())"
                )
                continuation.resume(returning: extracted)
            }
        }
    }

    private static func extractFromImageData(_ data: Data) -> [String: Double] {
        guard let src = CGImageSourceCreateWithData(data as CFData, nil),
              let props = CGImageSourceCopyPropertiesAtIndex(src, 0, nil) as? [String: Any] else {
            logWallMeasurement("camera_exif_sidecar image_data read failed bytes=\(data.count)")
            return [:]
        }
        let result = doublesFromImageProperties(props)
        logWallMeasurement("camera_exif_sidecar image_data extracted keys=\(result.keys.sorted())")
        return result
    }

    // MARK: - ImageIO (file URL)

    private static func extractFromImageSource(url: URL) -> [String: Double] {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
              let props = CGImageSourceCopyPropertiesAtIndex(src, 0, nil) as? [String: Any] else {
            logWallMeasurement("camera_exif_sidecar image_url read failed path=\(url.path)")
            return [:]
        }
        let result = doublesFromImageProperties(props)
        logWallMeasurement("camera_exif_sidecar image_url extracted path=\(url.lastPathComponent) keys=\(result.keys.sorted())")
        return result
    }

    private static func doublesFromImageProperties(_ props: [String: Any]) -> [String: Double] {
        var result: [String: Double] = [:]
        if let exif = exifDictionary(from: props) {
            result.merge(doublesFromExifDictionary(exif)) { _, new in new }
        }
        if let pixelWidth = doubleFromExifValue(props[kCGImagePropertyPixelWidth as String]), pixelWidth > 1 {
            result["imageWidthPx"] = pixelWidth
        }
        if let pixelHeight = doubleFromExifValue(props[kCGImagePropertyPixelHeight as String]), pixelHeight > 1 {
            result["imageHeightPx"] = pixelHeight
        }
        if result["focalLengthMm"] == nil,
           let tiff = tiffDictionary(from: props),
           let v = doubleFromExifValue(tiff["FocalLength"]), v > 0.01 {
            result["focalLengthMm"] = v
        }
        logWallMeasurement("camera_exif_sidecar image_properties extracted keys=\(result.keys.sorted())")
        return result
    }

    private static func exifDictionary(from imageProperties: [String: Any]) -> [String: Any]? {
        if let d = imageProperties[kCGImagePropertyExifDictionary as String] as? [String: Any] { return d }
        if let d = imageProperties["{Exif}"] as? [String: Any] { return d }
        return nil
    }

    private static func tiffDictionary(from imageProperties: [String: Any]) -> [String: Any]? {
        if let d = imageProperties[kCGImagePropertyTIFFDictionary as String] as? [String: Any] { return d }
        if let d = imageProperties["{TIFF}"] as? [String: Any] { return d }
        return nil
    }

    // MARK: - UIImagePicker mediaMetadata

    private static func extractFromCameraMetadata(_ meta: [AnyHashable: Any]?) -> [String: Double] {
        guard let meta else { return [:] }
        let exifKeyCandidates: [AnyHashable] = [
            "{Exif}",
            kCGImagePropertyExifDictionary as String,
        ]
        for key in exifKeyCandidates {
            if let exif = meta[key] as? [String: Any] {
                let d = doublesFromExifDictionary(exif)
                if !d.isEmpty {
                    logWallMeasurement("camera_exif_sidecar media_metadata extracted via key=\(key) keys=\(d.keys.sorted())")
                    return d
                }
            }
        }
        for (_, value) in meta {
            guard let dict = value as? [String: Any] else { continue }
            let d = doublesFromExifDictionary(dict)
            if !d.isEmpty {
                logWallMeasurement("camera_exif_sidecar media_metadata extracted via nested dict keys=\(d.keys.sorted())")
                return d
            }
        }
        logWallMeasurement("camera_exif_sidecar media_metadata no usable EXIF keys")
        return [:]
    }

    private static func doublesFromExifDictionary(_ exif: [String: Any]) -> [String: Double] {
        var result: [String: Double] = [:]
        if let v = doubleFromExifValue(exif[kCGImagePropertyExifFocalLength as String]), v > 0.01 {
            result["focalLengthMm"] = v
        }
        if result["focalLengthMm"] == nil, let v = doubleFromExifValue(exif["FocalLength"]), v > 0.01 {
            result["focalLengthMm"] = v
        }
        if let v = doubleFromExifValue(exif[kCGImagePropertyExifFocalLenIn35mmFilm as String]), v > 0.5 {
            result["focalLength35mmEquivMm"] = v
        }
        if result["focalLength35mmEquivMm"] == nil, let v = doubleFromExifValue(exif["FocalLenIn35mmFilm"]), v > 0.5 {
            result["focalLength35mmEquivMm"] = v
        }
        // Subject distance in meters (not SubjectDistRange — that is macro/normal/distant enum).
        if let v = doubleFromExifValue(exif[kCGImagePropertyExifSubjectDistance as String]), v > 0 {
            result["subjectDistanceMeters"] = v
        }
        if result["subjectDistanceMeters"] == nil, let v = doubleFromExifValue(exif["SubjectDistance"]), v > 0 {
            result["subjectDistanceMeters"] = v
        }
        if let v = doubleFromExifValue(exif[kCGImagePropertyExifPixelXDimension as String]), v > 1 {
            result["exifPixelXDimension"] = v
        }
        if let v = doubleFromExifValue(exif[kCGImagePropertyExifPixelYDimension as String]), v > 1 {
            result["exifPixelYDimension"] = v
        }
        if let lensSpec = exif[kCGImagePropertyExifLensSpecification as String] as? [NSNumber],
           let minFocal = lensSpec.first?.doubleValue,
           minFocal > 0.01 {
            result["lensMinFocalLengthMm"] = minFocal
        }
        return result
    }

    /// EXIF values may be `NSNumber`, rational dictionaries, or rational strings.
    private static func doubleFromExifValue(_ value: Any?) -> Double? {
        guard let value else { return nil }
        if let n = value as? NSNumber { return n.doubleValue }
        if let s = value as? String {
            let t = s.trimmingCharacters(in: .whitespacesAndNewlines)
            if let d = Double(t) { return d }
            let parts = t.split(separator: "/")
            if parts.count == 2,
               let a = Double(parts[0]),
               let b = Double(parts[1]),
               b != 0 {
                return a / b
            }
            return nil
        }
        if let d = value as? [String: Any],
           let num = d["Numerator"] as? NSNumber,
           let den = d["Denominator"] as? NSNumber,
           den.doubleValue != 0 {
            return num.doubleValue / den.doubleValue
        }
        return nil
    }

    // MARK: - Persist

    private static func persist(roomFolder: URL, additions: [String: Double]) {
        let url = sidecarURL(forRoomFolder: roomFolder)
        logWallMeasurement("camera_exif_sidecar persist generic path=\(url.path) keys=\(additions.keys.sorted())")
        persist(to: url, additions: additions)
    }

    private static func loadExistingDoubles(at url: URL) -> [String: Double] {
        guard let data = try? Data(contentsOf: url),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return [:]
        }
        var out: [String: Double] = [:]
        for (k, v) in obj {
            if let d = v as? Double {
                out[k] = d
            } else if let n = v as? NSNumber {
                out[k] = n.doubleValue
            }
        }
        return out
    }

    static func sidecarURL(forRoomFolder roomFolder: URL) -> URL {
        roomFolder.appendingPathComponent(fileName)
    }

    static func sidecarURL(forRoomURL roomURL: URL) -> URL {
        let roomFolder = roomURL.deletingLastPathComponent()
        let stem = canonicalStem(forRoomURL: roomURL)
        return roomFolder.appendingPathComponent("\(stem)\(stemSuffix)")
    }

    static func load(roomFolder: URL) -> [String: Double] {
        let url = sidecarURL(forRoomFolder: roomFolder)
        let values = loadExistingDoubles(at: url)
        logWallMeasurement("camera_exif_sidecar load generic path=\(url.path) keys=\(values.keys.sorted())")
        return values
    }

    static func load(roomURL: URL) -> [String: Double] {
        let specificURL = sidecarURL(forRoomURL: roomURL)
        let specific = loadExistingDoubles(at: specificURL)
        if !specific.isEmpty {
            logWallMeasurement("camera_exif_sidecar load room-specific path=\(specificURL.path) keys=\(specific.keys.sorted())")
            return specific
        }
        logWallMeasurement("camera_exif_sidecar room-specific missing path=\(specificURL.path); falling back to generic")
        return load(roomFolder: roomURL.deletingLastPathComponent())
    }

    static func mergeDerivedValues(roomFolder: URL, additions: [String: Double]) {
        guard !additions.isEmpty else { return }
        persist(roomFolder: roomFolder, additions: additions)
    }

    static func mergeDerivedValues(roomURL: URL, additions: [String: Double]) {
        guard !additions.isEmpty else { return }
        logWallMeasurement("camera_exif_sidecar merge room-specific path=\(sidecarURL(forRoomURL: roomURL).path) keys=\(additions.keys.sorted())")
        persist(to: sidecarURL(forRoomURL: roomURL), additions: additions)
    }

    static func loadSourceCameraInfo(roomFolder: URL, photoOrientation: Int? = nil) -> SourceCameraInfo {
        let values = load(roomFolder: roomFolder)
        let focalLengthMM = values["focalLengthMm"].map(Float.init)
        let focalLength35 = values["focalLength35mmEquivMm"].map(Float.init)
        let subjectDistance = values["subjectDistanceMeters"].map(Float.init)
        let imageWidthPx = values["imageWidthPx"].map { Int($0.rounded()) }
        let imageHeightPx = values["imageHeightPx"].map { Int($0.rounded()) }

        let sensorWidthMM: Float?
        if let focalLengthMM, let focalLength35, focalLengthMM > 0, focalLength35 > 0 {
            sensorWidthMM = 36 * focalLengthMM / focalLength35
        } else {
            sensorWidthMM = nil
        }

        return SourceCameraInfo(
            focalLengthMM: focalLengthMM,
            sensorWidthMM: sensorWidthMM,
            focalLength35mmEquivalentMM: focalLength35,
            subjectDistanceMeters: subjectDistance,
            imageWidthPx: imageWidthPx,
            imageHeightPx: imageHeightPx,
            photoOrientation: photoOrientation
        )
    }

    static func copySidecarIfPresent(fromRoomURL sourceRoomURL: URL, toSavedRoomURL destinationRoomURL: URL) {
        let fileManager = FileManager.default
        let candidates = [
            sidecarURL(forRoomURL: sourceRoomURL),
            sidecarURL(forRoomFolder: sourceRoomURL.deletingLastPathComponent()),
        ]
        guard let sourceURL = candidates.first(where: { fileManager.fileExists(atPath: $0.path) }) else { return }
        let destinationURL = sidecarURL(forRoomURL: destinationRoomURL)
        do {
            if fileManager.fileExists(atPath: destinationURL.path) {
                try fileManager.removeItem(at: destinationURL)
            }
            try fileManager.copyItem(at: sourceURL, to: destinationURL)
            logWallMeasurement("camera_exif_sidecar copied src=\(sourceURL.lastPathComponent) dst=\(destinationURL.lastPathComponent)")
        } catch {
            logWallMeasurement("camera_exif_sidecar copy failed \(error.localizedDescription) dst=\(destinationURL.path)")
        }
    }

    private static func persist(to url: URL, additions: [String: Double]) {
        var out = loadExistingDoubles(at: url)
        for (k, v) in additions { out[k] = v }
        guard !out.isEmpty else { return }
        let jsonObject: [String: Any] = out.mapValues { $0 as NSNumber }
        guard JSONSerialization.isValidJSONObject(jsonObject),
              let data = try? JSONSerialization.data(withJSONObject: jsonObject, options: [.sortedKeys]) else {
            logWallMeasurement("camera_exif_sidecar fail serialize path=\(url.path)")
            return
        }
        do {
            try data.write(to: url, options: [.atomic])
            logWallMeasurement("camera_exif_sidecar ok path=\(url.path) keys=\(out.keys.sorted())")
        } catch {
            logWallMeasurement("camera_exif_sidecar fail write \(error.localizedDescription) path=\(url.path)")
        }
    }

    private static func canonicalStem(forRoomURL roomURL: URL) -> String {
        var stem = roomURL.deletingPathExtension().lastPathComponent
        if stem.hasSuffix("_classic") {
            stem = String(stem.dropLast("_classic".count))
        } else if stem.hasSuffix("_3dgs") {
            stem = String(stem.dropLast("_3dgs".count))
        }
        return stem
    }
}
