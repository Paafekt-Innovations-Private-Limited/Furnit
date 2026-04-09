import Foundation

/// Persists SHARP generation camera metadata next to the exported classic PLY so later
/// metric-depth alignment can reproject splats with the same camera model used at generation time.
enum SharpCameraSidecar {
    private static let stemSuffix = "_sharp_camera.json"

    struct Info {
        let sourceImageWidthPx: Int
        let sourceImageHeightPx: Int
        let inputSquarePx: Int
        let sourceFocalPx: Float
        let sourceCxPx: Float
        let sourceCyPx: Float
        let internalFxPx: Float
        let internalFyPx: Float
        let internalCxPx: Float
        let internalCyPx: Float
    }

    static func infoIfPossible(
        sourceImageSize: CGSize,
        inputSquarePx: Int,
        exif: [String: Double]
    ) -> Info? {
        let sourceWidth = Int(sourceImageSize.width.rounded())
        let sourceHeight = Int(sourceImageSize.height.rounded())
        guard sourceWidth > 0, sourceHeight > 0 else {
            return nil
        }
        guard let sourceFocalPx = exactFocalLengthPx(
            exif: exif,
            sourceWidth: sourceWidth,
            sourceHeight: sourceHeight
        ), sourceFocalPx > 0.01 else {
            return nil
        }

        let sourceCxPx = Float(sourceWidth) * 0.5
        let sourceCyPx = Float(sourceHeight) * 0.5
        let scaleX = Float(inputSquarePx) / Float(sourceWidth)
        let scaleY = Float(inputSquarePx) / Float(sourceHeight)

        return Info(
            sourceImageWidthPx: sourceWidth,
            sourceImageHeightPx: sourceHeight,
            inputSquarePx: inputSquarePx,
            sourceFocalPx: sourceFocalPx,
            sourceCxPx: sourceCxPx,
            sourceCyPx: sourceCyPx,
            internalFxPx: sourceFocalPx * scaleX,
            internalFyPx: sourceFocalPx * scaleY,
            internalCxPx: Float(inputSquarePx) * 0.5,
            internalCyPx: Float(inputSquarePx) * 0.5
        )
    }

    static func sidecarURL(forRoomURL roomURL: URL) -> URL {
        let roomFolder = roomURL.deletingLastPathComponent()
        let stem = canonicalStem(forRoomURL: roomURL)
        return roomFolder.appendingPathComponent("\(stem)\(stemSuffix)")
    }

    static func writeIfPossible(
        roomURL: URL,
        sourceImageSize: CGSize,
        inputSquarePx: Int,
        exif: [String: Double]
    ) {
        guard let info = infoIfPossible(
            sourceImageSize: sourceImageSize,
            inputSquarePx: inputSquarePx,
            exif: exif
        ) else {
            let sourceWidth = Int(sourceImageSize.width.rounded())
            let sourceHeight = Int(sourceImageSize.height.rounded())
            guard sourceWidth > 0, sourceHeight > 0 else {
                logWallMeasurement("[RED][SHARP_CAMERA] skip invalid source size room=\(roomURL.lastPathComponent)")
                return
            }
            logWallMeasurement("[RED][SHARP_CAMERA] skip missing focal room=\(roomURL.lastPathComponent)")
            return
        }

        let payload: [String: Double] = [
            "sharpSourceImageWidthPx": Double(info.sourceImageWidthPx),
            "sharpSourceImageHeightPx": Double(info.sourceImageHeightPx),
            "sharpInputSquarePx": Double(info.inputSquarePx),
            "sharpSourceFocalPx": Double(info.sourceFocalPx),
            "sharpSourceCxPx": Double(info.sourceCxPx),
            "sharpSourceCyPx": Double(info.sourceCyPx),
            "sharpInternalFxPx": Double(info.internalFxPx),
            "sharpInternalFyPx": Double(info.internalFyPx),
            "sharpInternalCxPx": Double(info.internalCxPx),
            "sharpInternalCyPx": Double(info.internalCyPx),
        ]

        let url = sidecarURL(forRoomURL: roomURL)
        let jsonObject: [String: Any] = payload.mapValues { $0 as NSNumber }
        guard JSONSerialization.isValidJSONObject(jsonObject),
              let data = try? JSONSerialization.data(withJSONObject: jsonObject, options: [.sortedKeys]) else {
            logWallMeasurement("sharp_camera_sidecar fail serialize path=\(url.path)")
            return
        }
        do {
            try data.write(to: url, options: [.atomic])
            logWallMeasurement(
                "[GREEN][SHARP_CAMERA] ok path=\(url.path) source=\(info.sourceImageWidthPx)x\(info.sourceImageHeightPx) " +
                    "FOCAL_PX=\(String(format: "%.2f", info.sourceFocalPx)) " +
                    "INTERNAL_FX_FY=(\(String(format: "%.2f", info.internalFxPx)),\(String(format: "%.2f", info.internalFyPx)))"
            )
        } catch {
            logWallMeasurement("[RED][SHARP_CAMERA] fail write \(error.localizedDescription) path=\(url.path)")
        }
    }

    static func load(roomURL: URL) -> Info? {
        let url = sidecarURL(forRoomURL: roomURL)
        guard let data = try? Data(contentsOf: url),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            logWallMeasurement("[RED][SHARP_CAMERA] missing path=\(url.path)")
            return nil
        }

        guard let sourceWidth = numeric(obj["sharpSourceImageWidthPx"]).map({ Int($0.rounded()) }), sourceWidth > 0,
              let sourceHeight = numeric(obj["sharpSourceImageHeightPx"]).map({ Int($0.rounded()) }), sourceHeight > 0,
              let inputSquare = numeric(obj["sharpInputSquarePx"]).map({ Int($0.rounded()) }), inputSquare > 0,
              let sourceFocalPx = numeric(obj["sharpSourceFocalPx"]).map(Float.init), sourceFocalPx > 0.01,
              let sourceCxPx = numeric(obj["sharpSourceCxPx"]).map(Float.init),
              let sourceCyPx = numeric(obj["sharpSourceCyPx"]).map(Float.init),
              let internalFxPx = numeric(obj["sharpInternalFxPx"]).map(Float.init),
              let internalFyPx = numeric(obj["sharpInternalFyPx"]).map(Float.init),
              let internalCxPx = numeric(obj["sharpInternalCxPx"]).map(Float.init),
              let internalCyPx = numeric(obj["sharpInternalCyPx"]).map(Float.init) else {
            logWallMeasurement("[RED][SHARP_CAMERA] invalid path=\(url.path)")
            return nil
        }

        logWallMeasurement(
            "[GREEN][SHARP_CAMERA] load path=\(url.path) source=\(sourceWidth)x\(sourceHeight) " +
                "FOCAL_PX=\(String(format: "%.2f", sourceFocalPx))"
        )

        return Info(
            sourceImageWidthPx: sourceWidth,
            sourceImageHeightPx: sourceHeight,
            inputSquarePx: inputSquare,
            sourceFocalPx: sourceFocalPx,
            sourceCxPx: sourceCxPx,
            sourceCyPx: sourceCyPx,
            internalFxPx: internalFxPx,
            internalFyPx: internalFyPx,
            internalCxPx: internalCxPx,
            internalCyPx: internalCyPx
        )
    }

    static func copySidecarIfPresent(fromRoomURL sourceRoomURL: URL, toSavedRoomURL destinationRoomURL: URL) {
        let fileManager = FileManager.default
        let sourceURL = sidecarURL(forRoomURL: sourceRoomURL)
        guard fileManager.fileExists(atPath: sourceURL.path) else { return }
        let destinationURL = sidecarURL(forRoomURL: destinationRoomURL)
        do {
            if fileManager.fileExists(atPath: destinationURL.path) {
                try fileManager.removeItem(at: destinationURL)
            }
            try fileManager.copyItem(at: sourceURL, to: destinationURL)
            logWallMeasurement("[GREEN][SHARP_CAMERA] copied src=\(sourceURL.lastPathComponent) dst=\(destinationURL.lastPathComponent)")
        } catch {
            logWallMeasurement("[RED][SHARP_CAMERA] copy failed \(error.localizedDescription) dst=\(destinationURL.path)")
        }
    }

    private static func numeric(_ any: Any?) -> Double? {
        if let d = any as? Double { return d }
        if let n = any as? NSNumber { return n.doubleValue }
        if let i = any as? Int { return Double(i) }
        if let s = any as? String { return Double(s) }
        return nil
    }

    private static func exactFocalLengthPx(exif: [String: Double], sourceWidth: Int, sourceHeight: Int) -> Float? {
        let focal35MM: Float
        if let focal35 = exif["focalLength35mmEquivMm"].map(Float.init), focal35 > 0.01 {
            focal35MM = focal35
        } else if let focalMM = exif["focalLengthMm"].map(Float.init), focalMM > 0.01 {
            if focalMM < 10.0 {
                // Match Python SHARP's fallback when only a small raw focal value is available.
                focal35MM = focalMM * 8.4
            } else {
                focal35MM = focalMM
            }
        } else if let focalPx = exif["focalLengthPx"].map(Float.init), focalPx > 0.01 {
            return focalPx
        } else {
            return nil
        }
        let diagonal = hypot(Float(sourceWidth), Float(sourceHeight))
        let diagonal35mm: Float = hypot(36.0, 24.0)
        guard diagonal > 1, diagonal35mm > 0.01 else { return nil }
        return focal35MM * diagonal / diagonal35mm
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
