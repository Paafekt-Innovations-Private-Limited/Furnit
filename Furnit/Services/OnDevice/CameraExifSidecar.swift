import Foundation
import ImageIO
import Photos

/// Writes `camera_exif.json` next to SHARP outputs for [WallMeasurementEstimator]: focal lengths + **SubjectDistance** (m).
///
/// Library picks often have **no** `imageURL`; we use **PHAsset** image data (preserves EXIF) when available.
/// In-app camera uses `UIImagePickerController` **mediaMetadata** (`{Exif}`).
enum CameraExifSidecar {
    private static let fileName = "camera_exif.json"

    /// Merge order: file URL → Photo Library asset data → camera `mediaMetadata` (later keys fill gaps).
    static func writeMerged(
        roomFolder: URL,
        imageURL: URL?,
        mediaMetadata: [AnyHashable: Any]?,
        photoLibraryAssetLocalId: String?,
    ) async {
        var merged: [String: Double] = [:]
        func absorb(_ other: [String: Double]) {
            for (key, value) in other where merged[key] == nil {
                merged[key] = value
            }
        }
        if let u = imageURL {
            absorb(extractFromImageSource(url: u))
        }
        if let id = photoLibraryAssetLocalId, !id.isEmpty {
            let fromAsset = await extractFromPhotoLibraryAsset(localIdentifier: id)
            absorb(fromAsset)
            if fromAsset.isEmpty {
                logWallMeasurement("camera_exif_sidecar phAsset id=\(id) extracted no fields (iCloud/network?)")
            }
        }
        absorb(extractFromCameraMetadata(mediaMetadata))
        guard !merged.isEmpty else {
            logWallMeasurement(
                "camera_exif_sidecar skip (no EXIF) folder=\(roomFolder.lastPathComponent) " +
                    "hasURL=\(imageURL != nil) hasPHAsset=\(!(photoLibraryAssetLocalId ?? "").isEmpty) hasMeta=\(mediaMetadata != nil)",
            )
            return
        }
        persist(roomFolder: roomFolder, additions: merged)
    }

    // MARK: - Photo Library (full file bytes + EXIF)

    private static func extractFromPhotoLibraryAsset(localIdentifier: String) async -> [String: Double] {
        await withCheckedContinuation { continuation in
            let fetch = PHAsset.fetchAssets(withLocalIdentifiers: [localIdentifier], options: nil)
            guard let asset = fetch.firstObject else {
                logWallMeasurement("camera_exif_sidecar phAsset fetch empty id=\(localIdentifier)")
                continuation.resume(returning: [:])
                return
            }
            let opts = PHImageRequestOptions()
            opts.isNetworkAccessAllowed = true
            opts.deliveryMode = .highQualityFormat
            opts.version = .current
            PHImageManager.default().requestImageDataAndOrientation(for: asset, options: opts) { data, _, _, info in
                if data == nil {
                    let cancelled = (info?[PHImageCancelledKey] as? Bool) == true
                    logWallMeasurement("camera_exif_sidecar phAsset data nil id=\(localIdentifier) cancelled=\(cancelled)")
                }
                guard let data else {
                    continuation.resume(returning: [:])
                    return
                }
                continuation.resume(returning: extractFromImageData(data))
            }
        }
    }

    private static func extractFromImageData(_ data: Data) -> [String: Double] {
        guard let src = CGImageSourceCreateWithData(data as CFData, nil),
              let props = CGImageSourceCopyPropertiesAtIndex(src, 0, nil) as? [String: Any] else {
            return [:]
        }
        return doublesFromImageProperties(props)
    }

    // MARK: - ImageIO (file URL)

    private static func extractFromImageSource(url: URL) -> [String: Double] {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
              let props = CGImageSourceCopyPropertiesAtIndex(src, 0, nil) as? [String: Any] else {
            return [:]
        }
        return doublesFromImageProperties(props)
    }

    private static func doublesFromImageProperties(_ props: [String: Any]) -> [String: Double] {
        var result: [String: Double] = [:]
        if let exif = exifDictionary(from: props) {
            result.merge(doublesFromExifDictionary(exif)) { _, new in new }
        }
        if result["focalLengthMm"] == nil,
           let tiff = tiffDictionary(from: props),
           let v = doubleFromExifValue(tiff["FocalLength"]), v > 0.01 {
            result["focalLengthMm"] = v
        }
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
                if !d.isEmpty { return d }
            }
        }
        for (_, value) in meta {
            guard let dict = value as? [String: Any] else { continue }
            let d = doublesFromExifDictionary(dict)
            if !d.isEmpty { return d }
        }
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
        let url = roomFolder.appendingPathComponent(fileName)
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
}
