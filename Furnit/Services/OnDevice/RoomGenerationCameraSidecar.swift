import Foundation

/// Persists Splat generation camera metadata next to the exported classic PLY so later
/// metric-depth alignment can reproject splats with the same camera model used at generation time.
enum RoomGenerationCameraSidecar {
    private static let stemSuffix = "_generation_camera.json"

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
                logWallMeasurement("[RED][ROOM_GENERATION_CAMERA] skip invalid source size room=\(roomURL.lastPathComponent)")
                return
            }
            logWallMeasurement("[RED][ROOM_GENERATION_CAMERA] skip missing focal room=\(roomURL.lastPathComponent)")
            return
        }

        let payload: [String: Double] = [
            "roomGenerationSourceImageWidthPx": Double(info.sourceImageWidthPx),
            "roomGenerationSourceImageHeightPx": Double(info.sourceImageHeightPx),
            "roomGenerationInputSquarePx": Double(info.inputSquarePx),
            "roomGenerationSourceFocalPx": Double(info.sourceFocalPx),
            "roomGenerationSourceCxPx": Double(info.sourceCxPx),
            "roomGenerationSourceCyPx": Double(info.sourceCyPx),
            "roomGenerationInternalFxPx": Double(info.internalFxPx),
            "roomGenerationInternalFyPx": Double(info.internalFyPx),
            "roomGenerationInternalCxPx": Double(info.internalCxPx),
            "roomGenerationInternalCyPx": Double(info.internalCyPx),
        ]

        let url = sidecarURL(forRoomURL: roomURL)
        let jsonObject: [String: Any] = payload.mapValues { $0 as NSNumber }
        guard JSONSerialization.isValidJSONObject(jsonObject),
              let data = try? JSONSerialization.data(withJSONObject: jsonObject, options: [.sortedKeys]) else {
            logWallMeasurement("generation_camera_sidecar fail serialize path=\(url.path)")
            return
        }
        do {
            try data.write(to: url, options: [.atomic])
            logWallMeasurement(
                "[GREEN][ROOM_GENERATION_CAMERA] ok path=\(url.path) source=\(info.sourceImageWidthPx)x\(info.sourceImageHeightPx) " +
                    "FOCAL_PX=\(String(format: "%.2f", info.sourceFocalPx)) " +
                    "INTERNAL_FX_FY=(\(String(format: "%.2f", info.internalFxPx)),\(String(format: "%.2f", info.internalFyPx)))"
            )
        } catch {
            logWallMeasurement("[RED][ROOM_GENERATION_CAMERA] fail write \(error.localizedDescription) path=\(url.path)")
        }
    }

    static func load(roomURL: URL) -> Info? {
        let url = sidecarURL(forRoomURL: roomURL)
        guard let data = try? Data(contentsOf: url),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            logWallMeasurement("[RED][ROOM_GENERATION_CAMERA] missing path=\(url.path)")
            return nil
        }

        guard let sourceWidth = numeric(obj["roomGenerationSourceImageWidthPx"]).map({ Int($0.rounded()) }), sourceWidth > 0,
              let sourceHeight = numeric(obj["roomGenerationSourceImageHeightPx"]).map({ Int($0.rounded()) }), sourceHeight > 0,
              let inputSquare = numeric(obj["roomGenerationInputSquarePx"]).map({ Int($0.rounded()) }), inputSquare > 0,
              let sourceFocalPx = numeric(obj["roomGenerationSourceFocalPx"]).map(Float.init), sourceFocalPx > 0.01,
              let sourceCxPx = numeric(obj["roomGenerationSourceCxPx"]).map(Float.init),
              let sourceCyPx = numeric(obj["roomGenerationSourceCyPx"]).map(Float.init),
              let internalFxPx = numeric(obj["roomGenerationInternalFxPx"]).map(Float.init),
              let internalFyPx = numeric(obj["roomGenerationInternalFyPx"]).map(Float.init),
              let internalCxPx = numeric(obj["roomGenerationInternalCxPx"]).map(Float.init),
              let internalCyPx = numeric(obj["roomGenerationInternalCyPx"]).map(Float.init) else {
            logWallMeasurement("[RED][ROOM_GENERATION_CAMERA] invalid path=\(url.path)")
            return nil
        }

        logWallMeasurement(
            "[GREEN][ROOM_GENERATION_CAMERA] load path=\(url.path) source=\(sourceWidth)x\(sourceHeight) " +
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
            logWallMeasurement("[GREEN][ROOM_GENERATION_CAMERA] copied src=\(sourceURL.lastPathComponent) dst=\(destinationURL.lastPathComponent)")
        } catch {
            logWallMeasurement("[RED][ROOM_GENERATION_CAMERA] copy failed \(error.localizedDescription) dst=\(destinationURL.path)")
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
                // Match Python Splat's fallback when only a small raw focal value is available.
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
