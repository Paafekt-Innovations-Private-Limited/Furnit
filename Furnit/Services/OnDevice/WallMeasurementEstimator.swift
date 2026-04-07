import CoreGraphics
import CoreML
import Foundation
import UIKit
import Vision

/// YOLO wall segmentation + SHARP monodepth (on disk) + camera EXIF → front wall width/height in meters
/// plus **camera-to-wall depth** (`Result.depthMeters`) for room metadata.
/// UserDefaults keys match Android `WallMeasurementEstimator` prefs.
///
/// **Room-scene boxes** (e.g. “hospital room”) often span floor + wall + ceiling. Tier 3 crops the bbox
/// (trim top ~10%, bottom ~25%) before using it as a wall strip. **Tier 4** uses a conservative full-image crop when no label matches.
enum WallMeasurementEstimator {

    struct Result {
        let widthMeters: Float
        let heightMeters: Float
        /// Distance from camera to the front wall (monodepth median at wall rect, or assumed Z).
        let depthMeters: Float
        let calibrationMode: String
    }

    // MARK: - Constants

    private static let lvisWall = 571
    private static let standardDoorM: Float = 2.03
    /// YOLO-E anchor floor — low enough for wall/room classes (not Furniture Fit’s 0.25).
    private static let classScoreFloor: Float = 0.05

    private static let keyEnabled = "wall_measurement_yolo_on_save"
    private static let keyCalibration = "wall_measurement_calibration"
    private static let keyCeiling = "wall_measurement_assumed_ceiling_m"
    private static let keySensorMm = "wall_measurement_sensor_width_mm"
    private static let keyAssumedZ = "wall_measurement_assumed_depth_m"
    private static let keySharpGlobalScale = "wall_measurement_sharp_global_scale"
    private static let metricDepthCandidateFileNames = [
        "depthpro_metric_depth.bin",
        "metric_depth.bin",
        "sharp_metric_depth.bin",
    ]

    private static let calAuto = "auto"
    private static let calDoor = "door"
    private static let calCeiling = "ceiling"

    private struct CameraIntrinsics {
        let fx: Float
        let fy: Float
        let cx: Float
        let cy: Float
        let focalLengthMM: Float?
        let sensorWidthMM: Float?
        let imageWidth: Int
        let imageHeight: Int
        let horizontalFOVDegrees: Float
        let verticalFOVDegrees: Float
        let source: String
    }

    private static func logMetricDepthMeasurement(_ message: String) {
        logWallMeasurement(message.uppercased())
    }

    private static var sharpGlobalMetricScale: Float {
        floatPref(keySharpGlobalScale, default: 1.08)
    }

    // MARK: - Vision horizon (PLY Z-span → angle-aware depth heuristic)

    private static func cgImagePropertyOrientation(_ o: UIImage.Orientation) -> CGImagePropertyOrientation {
        switch o {
        case .up: return .up
        case .upMirrored: return .upMirrored
        case .down: return .down
        case .downMirrored: return .downMirrored
        case .left: return .left
        case .leftMirrored: return .leftMirrored
        case .right: return .right
        case .rightMirrored: return .rightMirrored
        @unknown default: return .up
        }
    }

    /// `VNHorizonObservation.angle` (rad): horizon vs image x-axis; used as a cheap pitch proxy for gallery photos.
    private static func horizonPitchRadians(thumbnail: UIImage) -> Float? {
        guard let cgImage = thumbnail.cgImage else {
            logWallMeasurement("horizon skip: no CGImage")
            return nil
        }
        let orientation = cgImagePropertyOrientation(thumbnail.imageOrientation)
        let request = VNDetectHorizonRequest()
        let handler = VNImageRequestHandler(cgImage: cgImage, orientation: orientation, options: [:])
        do {
            try handler.perform([request])
        } catch {
            logWallMeasurement("horizon Vision perform failed: \(error.localizedDescription)")
            return nil
        }
        guard let obs = request.results?.first else {
            logWallMeasurement("horizon no observation")
            return nil
        }
        return Float(obs.angle)
    }

    /// Angle-aware depth from splat **Z span** (~ground distance) vs tilted optical axis: `depth ≈ zSpan * cos(pitch)`.
    private static func zSpanWithHorizonDecomposition(rawZSpan: Float, pitchRad: Float?) -> (
        depthAlong: Float,
        heightAlong: Float,
        applied: Bool
    ) {
        guard rawZSpan.isFinite, rawZSpan > 0,
              let pitch = pitchRad, pitch.isFinite
        else {
            return (rawZSpan, 0, false)
        }
        let p = Double(pitch)
        let c = cos(p)
        let s = sin(p)
        let raw = Double(rawZSpan)
        let along = Float(max(0.05, raw * c))
        let vertical = Float(raw * s)
        return (along, vertical, true)
    }

    // MARK: - Label matchers

    private static let doorNegative = [
        "doormat", "doorbell", "doorstop", "car door",
    ]

    private static func isDoorLabel(_ raw: String) -> Bool {
        let l = raw.lowercased()
        guard l.contains("door") else { return false }
        return !doorNegative.contains(where: { l.contains($0) })
    }

    private static let wallNegative = [
        "wall lamp", "wallpaper", "wall clock", "wall socket", "wall outlet",
        "wall plug", "wall sconce", "wall light", "wall strip", "wall sticker", "wall decal",
    ]

    private static func isWallLabel(_ raw: String) -> Bool {
        let l = raw.lowercased()
        if wallNegative.contains(where: { l.contains($0) }) { return false }
        return l.range(of: #"\bwall\b"#, options: .regularExpression) != nil
    }

    /// Scene labels: “living room”, “hospital room”, “bedroom”, etc. Bbox is usually the whole visible scene, not the wall surface alone.
    private static let roomObjectNegative = [
        "hospital bed", "building block", "building material",
        "office chair", "office desk", "office supply",
        "kitchen knife", "kitchen cabinet", "kitchen counter", "kitchen floor",
        "kitchen hood", "kitchen island", "kitchen sink", "kitchen table",
        "kitchen utensil", "kitchen window", "kitchenware",
        "bathroom accessory", "bathroom door", "bathroom mirror", "bathroom sink",
        "bathroom cabinet", "bathroom window",
        "brick building", "glass building", "church tower", "empire state building",
    ]

    private static func isRoomSceneLabel(_ raw: String) -> Bool {
        let l = raw.lowercased()
        if roomObjectNegative.contains(where: { l.contains($0) }) { return false }
        return l.range(of: #"\broom\b"#, options: .regularExpression) != nil
    }

    // MARK: - Prefs

    private static func floatPref(_ key: String, default def: Float) -> Float {
        guard let o = UserDefaults.standard.object(forKey: key) else { return def }
        switch o {
        case let d as Double: return Float(d)
        case let f as Float: return f
        case let n as NSNumber: return n.floatValue
        case let i as Int: return Float(i)
        default: return def
        }
    }

    private static var isEnabled: Bool {
        if UserDefaults.standard.object(forKey: keyEnabled) == nil { return true }
        return UserDefaults.standard.bool(forKey: keyEnabled)
    }

    // MARK: - Public entry point

    static func measure(
        roomFolder: URL,
        thumbnail: UIImage,
        model: MLModel?,
        photoOrientation: PhotoOrientation = .portrait,
        plyBounds: RoomBounds? = nil
    ) async -> Result? {
        let ceilingM = floatPref(keyCeiling, default: 2.5)
        let sensorMm = floatPref(keySensorMm, default: 6.4)
        let assumedZ = floatPref(keyAssumedZ, default: 2.5)
        let calPref = UserDefaults.standard.string(forKey: keyCalibration) ?? calAuto

        logWallMeasurement(
            "measure begin folder=\(roomFolder.lastPathComponent) enabled=\(isEnabled) cal=\(calPref) ceiling_m=\(ceilingM) assumed_z_m=\(assumedZ) sensor_mm=\(sensorMm) plyBounds=\(plyBounds != nil)",
        )

        guard isEnabled else {
            logWallMeasurement("measure skip: \(keyEnabled) is false")
            return nil
        }
        guard let model else {
            logWallMeasurement("measure abort: no ML model")
            return nil
        }

        // --- YOLO detections (needed for depth + fallback dimensions) ---

        let detsRaw: [FurnitureFitDetection]
        let mapping: YoloEImageInference.OnnxStyleMapping
        do {
            (detsRaw, mapping) = try YoloEImageInference.runDetections(
                image: thumbnail,
                model: model,
                classBlacklist: [],
                confidenceThreshold: classScoreFloor,
            )
        } catch {
            logWallMeasurement("measure abort: yolo \(error.localizedDescription)")
            return nil
        }

        let iw = mapping.sourceWidth
        let ih = mapping.sourceHeight
        logWallMeasurement("reference_image size=\(iw)x\(ih) yolo_side=\(mapping.modelSide) class_score_floor=\(classScoreFloor)")
        guard iw > 8, ih > 8 else {
            logWallMeasurement("measure abort: image too small")
            return nil
        }

        let mapped = detsRaw.map { YoloEImageInference.mapDetectionToSourceImage(det: $0, mapping: mapping) }
        let sample = mapped.prefix(5).map { "c\($0.classIdx):\($0.confidence)" }.joined(separator: ",")
        logWallMeasurement("yolo_detections count=\(mapped.count) sample=[\(sample)]")

        let names = loadClassNames()
        let (wallRect, wallSource) = findWallRect(
            detections: mapped,
            names: names,
            imageWidth: iw,
            imageHeight: ih,
        )
        logWallMeasurement(
            "wall_rect source=\(wallSource) rect=[\(Int(wallRect.minX)),\(Int(wallRect.minY)),\(Int(wallRect.maxX)),\(Int(wallRect.maxY))] px",
        )

        let exif = loadExifJson(roomFolder: roomFolder)
        let (focalPx, focalHow) = focalLengthPx(imageWidth: iw, exif: exif, sensorMm: sensorMm)
        logWallMeasurement("focal_px=\(focalPx) focal_reason=\(focalHow) exif_json=\(exif != nil)")

        let metricDepth = loadMetricDepthBuffer(roomFolder: roomFolder)
        if let metricDepth,
           let intrinsics = resolveCameraIntrinsics(
               imageWidth: iw,
               imageHeight: ih,
               exif: exif,
               sensorMmFallback: sensorMm
           ),
           let metricResult = measureFromMetricDepth(
               metricDepth.buffer,
               sourceName: metricDepth.sourceName,
               wallRect: wallRect,
               intrinsics: intrinsics
           ) {
            return metricResult
        } else if metricDepth != nil {
            logMetricDepthMeasurement("METRIC DEPTH PRESENT BUT CAMERA INTRINSICS COULD NOT BE RESOLVED; FALLING BACK")
        }

        // --- Depth: monodepth → EXIF subject distance → assumed Z ---

        let monoURL = roomFolder.appendingPathComponent("sharp_monodepth.bin")
        let mono = MonodepthBuffer.load(url: monoURL)
        if mono != nil {
            logWallMeasurement("monodepth_file=true path=\(monoURL.path)")
        } else {
            logWallMeasurement("monodepth_file=false path=\(monoURL.path)")
        }

        let horizonPitch = horizonPitchRadians(thumbnail: thumbnail)
        if let hp = horizonPitch {
            let deg = hp * 180 / Float.pi
            logWallMeasurement(
                "horizon observe pitch_rad=\(String(format: "%.5f", hp)) pitch_deg=\(String(format: "%.2f", deg)) " +
                    "(VNDetectHorizonRequest; SHARP Z-span → cos/sin split)"
            )
        } else {
            logWallMeasurement("horizon observe unavailable — no VN horizon angle (ply_z left un-tilt-corrected)")
        }

        /// Depth for pinhole W/H: same basis as saved room depth when PLY bounds exist (Z span of loaded splat AABB).
        let pinholeDepth: Float
        let pinholeDepthSource: String
        if let ply = plyBounds, ply.depth > 0.1 {
            let (zAdj, zHeightAlong, didHorizon) = zSpanWithHorizonDecomposition(
                rawZSpan: ply.depth,
                pitchRad: horizonPitch,
            )
            if didHorizon {
                logWallMeasurement(
                    "ply_z_horizon raw_z_span_su=\(String(format: "%.6f", ply.depth)) " +
                        "depth_along≈z*cos=\(String(format: "%.6f", zAdj)) " +
                        "height_along≈z*sin=\(String(format: "%.6f", zHeightAlong))"
                )
            }
            pinholeDepth = zAdj
            pinholeDepthSource = didHorizon ? "ply_z_span_horizon" : "ply_z_span"
        } else if let mono, let md = mono.medianAt(rect: wallRect, imageWidth: iw, imageHeight: ih), md > 0 {
            pinholeDepth = md
            pinholeDepthSource = "monodepth"
        } else if let sd = exif.flatMap({ numericExifValue($0["subjectDistanceMeters"]) }), sd > 0.1, sd < 30 {
            pinholeDepth = Float(sd)
            pinholeDepthSource = "exif_subject_distance"
        } else {
            pinholeDepth = max(0.5, min(20, assumedZ))
            pinholeDepthSource = "assumed_z"
        }
        logWallMeasurement(
            "pinhole_depth value=\(String(format: "%.6f", pinholeDepth)) source=\(pinholeDepthSource) " +
                "(W/H use same Z as room depth when plyBounds present)"
        )

        // --- Width / Height: YOLO pinhole (targets the wall surface) ---
        // PLY AABB measures the whole 3D scene, not the front wall — it over-estimates
        // width (side-wall splats) and under-estimates height (fog trim cuts floor/ceiling).
        // YOLO specifically detects the wall rect and projects to meters via pinhole camera.

        let maxRealisticWidth: Float = (photoOrientation == .portrait) ? 5.0 : 8.0
        let maxRealisticHeight: Float = (photoOrientation == .portrait) ? 3.5 : 3.2

        guard focalPx.isFinite, focalPx > 1e-3 else {
            logWallMeasurement("measure abort: invalid focal_px")
            return nil
        }

        let wallPxW = Float(max(1, wallRect.width))
        let wallPxH = Float(max(1, wallRect.height))
        let rawW = (wallPxW / focalPx) * pinholeDepth
        let rawH = (wallPxH / focalPx) * pinholeDepth

        let (scale, calMode) = calibrationScale(
            rawH: rawH,
            detections: mapped,
            names: names,
            mono: mono,
            wallDepth: pinholeDepth,
            focalPx: focalPx,
            imageWidth: iw,
            imageHeight: ih,
            calPref: calPref,
            ceilingM: ceilingM,
        )

        var wm = (rawW * scale).clamped(to: 1.5...maxRealisticWidth)
        var hm = (rawH * scale).clamped(to: 1.5...maxRealisticHeight)

        // PLY bounds as sanity cap: the front wall can't be wider/taller than the
        // whole scene AABB (un-trimmed). Catches outlier YOLO+pinhole over-estimates.
        if let bounds = plyBounds {
            if wm > bounds.width { wm = bounds.width.clamped(to: 1.5...maxRealisticWidth) }
            if hm > bounds.height { hm = bounds.height.clamped(to: 1.5...maxRealisticHeight) }
            logWallMeasurement(
                "ply_sanity_cap scene_w=\(String(format: "%.3f", bounds.width)) scene_h=\(String(format: "%.3f", bounds.height)) " +
                    "capped_w=\(String(format: "%.3f", wm)) capped_h=\(String(format: "%.3f", hm))",
            )
        }

        // Saved room depth: PLY Z span when available (same units as pinhole when ply_z_span was used).
        let roomDepth: Float
        let depthMode: String
        if let bounds = plyBounds, bounds.depth > 0.1 {
            let (roomZ, roomVert, didHorizonRoom) = zSpanWithHorizonDecomposition(
                rawZSpan: bounds.depth,
                pitchRad: horizonPitch,
            )
            roomDepth = roomZ
            if didHorizonRoom,
               pinholeDepthSource != "ply_z_span",
               pinholeDepthSource != "ply_z_span_horizon" {
                logWallMeasurement(
                    "ply_z_horizon room_depth_only raw=\(String(format: "%.6f", bounds.depth)) " +
                        "cos_depth=\(String(format: "%.6f", roomZ)) sin_h=\(String(format: "%.6f", roomVert)) " +
                        "pinhole_source=\(pinholeDepthSource)"
                )
            }
            if didHorizonRoom {
                depthMode =
                    (pinholeDepthSource == "ply_z_span" || pinholeDepthSource == "ply_z_span_horizon")
                    ? "ply_z_horizon"
                    : "\(pinholeDepthSource)+ply_z_horizon"
            } else {
                depthMode = pinholeDepthSource == "ply_z_span" ? "ply_z" : "\(pinholeDepthSource)+ply_z"
            }
        } else {
            roomDepth = pinholeDepth
            depthMode = pinholeDepthSource
        }

        logWallMeasurement(
            "measure_final mode=\(calMode) width_m=\(String(format: "%.3f", wm)) height_m=\(String(format: "%.3f", hm)) " +
                "depth=\(String(format: "%.3f", roomDepth))(\(depthMode)) wall_source=\(wallSource) " +
                "focal_px=\(String(format: "%.1f", focalPx))(\(focalHow)) " +
                "raw_geom_m w=\(String(format: "%.3f", rawW)) h=\(String(format: "%.3f", rawH)) scale=\(String(format: "%.4f", scale)) " +
                "pinhole_z=\(String(format: "%.3f", pinholeDepth))(\(pinholeDepthSource))",
        )
        return Result(widthMeters: wm, heightMeters: hm, depthMeters: roomDepth, calibrationMode: calMode)
    }

    static func measureUsingMetricDepthOnly(
        roomURL: URL,
        thumbnail: UIImage,
        photoOrientation: PhotoOrientation = .portrait,
        plyBounds: RoomBounds? = nil
    ) -> Result? {
        let sensorMm = floatPref(keySensorMm, default: 6.4)
        let roomFolder = roomURL.deletingLastPathComponent()
        let exif = loadExifJson(roomURL: roomURL)
        let imageSize = sourceImagePixelSize(for: thumbnail)

        logMetricDepthMeasurement(
            "METRIC ONLY BEGIN ROOM=\(roomURL.lastPathComponent) IMAGE=\(imageSize.width)X\(imageSize.height) ORIENTATION=\(photoOrientation.rawValue)"
        )

        guard let metricDepth = loadMetricDepthBuffer(roomFolder: roomFolder, roomURL: roomURL) else {
            return nil
        }
        guard let intrinsics = resolveCameraIntrinsics(
            imageWidth: imageSize.width,
            imageHeight: imageSize.height,
            exif: exif,
            sensorMmFallback: sensorMm
        ) else {
            logMetricDepthMeasurement("METRIC ONLY FAILED TO RESOLVE CAMERA INTRINSICS")
            return nil
        }

        let wallRect = metricDepthOnlyRect(
            imageWidth: imageSize.width,
            imageHeight: imageSize.height,
            photoOrientation: photoOrientation
        )
        logMetricDepthMeasurement(
            "METRIC ONLY RECT [\(Int(wallRect.minX)),\(Int(wallRect.minY)),\(Int(wallRect.maxX)),\(Int(wallRect.maxY))]"
        )

        return measureFromMetricDepth(
            metricDepth.buffer,
            sourceName: metricDepth.sourceName,
            wallRect: wallRect,
            intrinsics: intrinsics,
            plyBounds: plyBounds
        )
    }

    static func measureUsingMetricDepthAlignedPLY(
        roomURL: URL,
        thumbnail: UIImage
    ) -> Result? {
        let sensorMm = floatPref(keySensorMm, default: 6.4)
        let roomFolder = roomURL.deletingLastPathComponent()
        let exif = loadExifJson(roomURL: roomURL)
        let sharpCamera = SharpCameraSidecar.load(roomURL: roomURL)
        let fallbackImageSize = sourceImagePixelSize(for: thumbnail)
        let imageSize: (width: Int, height: Int) = sharpCamera.map {
            ($0.sourceImageWidthPx, $0.sourceImageHeightPx)
        } ?? fallbackImageSize

        logMetricDepthMeasurement(
            "METRIC PLY ALIGN BEGIN ROOM=\(roomURL.lastPathComponent) IMAGE=\(imageSize.width)X\(imageSize.height)"
        )

        guard let ply = BinaryClassicPly.load(url: roomURL) else {
            logMetricDepthMeasurement("METRIC PLY ALIGN FAILED TO LOAD PLY")
            return nil
        }

        if let sharpCamera,
           let edgeLayout = measureUsingImageEdgeLayout(
               thumbnail: thumbnail,
               ply: ply,
               sharpCamera: sharpCamera
           ) {
            logMetricDepthMeasurement(
                "METRIC PLY ALIGN RESULT WIDTH_M=\(String(format: "%.4f", edgeLayout.result.widthMeters)) " +
                    "HEIGHT_M=\(String(format: "%.4f", edgeLayout.result.heightMeters)) DEPTH_M=\(String(format: "%.4f", edgeLayout.result.depthMeters)) " +
                    "SOURCE=\(edgeLayout.source)"
            )
            return edgeLayout.result
        }
        logMetricDepthMeasurement("METRIC PLY ALIGN EDGE LAYOUT FALLBACK=GLOBAL_SCALE_RANSAC")

        if let globalRansac = measureUsingSharpGlobalScaleRansac(ply: ply) {
            logMetricDepthMeasurement(
                "METRIC PLY ALIGN RESULT WIDTH_M=\(String(format: "%.4f", globalRansac.result.widthMeters)) " +
                    "HEIGHT_M=\(String(format: "%.4f", globalRansac.result.heightMeters)) DEPTH_M=\(String(format: "%.4f", globalRansac.result.depthMeters)) " +
                    "SOURCE=\(globalRansac.source)"
            )
            return globalRansac.result
        }
        logMetricDepthMeasurement("METRIC PLY ALIGN GLOBAL SCALE FALLBACK=DEPTH_WARP")

        guard let metricDepth = loadMetricDepthBuffer(roomFolder: roomFolder, roomURL: roomURL) else {
            return nil
        }
        guard let intrinsics = resolveCameraIntrinsics(
            imageWidth: imageSize.width,
            imageHeight: imageSize.height,
            exif: exif,
            sensorMmFallback: sensorMm,
            sharpCamera: sharpCamera
        ) else {
            logMetricDepthMeasurement("METRIC PLY ALIGN FAILED TO RESOLVE CAMERA INTRINSICS")
            return nil
        }

        let cx = intrinsics.cx
        let cy = intrinsics.cy
        let fx = intrinsics.fx
        let fy = intrinsics.fy
        let imageWidth = Float(intrinsics.imageWidth)
        let imageHeight = Float(intrinsics.imageHeight)

        var correctedMinX: Float = .greatestFiniteMagnitude
        var correctedMaxX: Float = -.greatestFiniteMagnitude
        var correctedMinY: Float = .greatestFiniteMagnitude
        var correctedMaxY: Float = -.greatestFiniteMagnitude
        var correctedMinZ: Float = .greatestFiniteMagnitude
        var correctedMaxZ: Float = -.greatestFiniteMagnitude
        var validCount = 0
        var skippedOutOfImage = 0
        var skippedNoDepth = 0
        var skippedInvalidZ = 0
        var minScale: Float = .greatestFiniteMagnitude
        var maxScale: Float = -.greatestFiniteMagnitude
        var scaleSum: Double = 0
        var correctedXs: [Float] = []
        var correctedYs: [Float] = []
        var correctedZs: [Float] = []
        var correctedPoints: [SIMD3<Float>] = []
        correctedXs.reserveCapacity(min(ply.vertexCount, 900_000))
        correctedYs.reserveCapacity(min(ply.vertexCount, 900_000))
        correctedZs.reserveCapacity(min(ply.vertexCount, 900_000))
        correctedPoints.reserveCapacity(min(ply.vertexCount, 900_000))

        for index in 0..<ply.vertexCount {
            let point = ply.position(at: index)
            let z = point.z
            guard point.x.isFinite, point.y.isFinite, z.isFinite, z > 0.05 else {
                skippedInvalidZ += 1
                continue
            }

            let pixelX = fx * point.x / z + cx
            let pixelY = fy * point.y / z + cy
            guard pixelX.isFinite, pixelY.isFinite,
                  pixelX >= 0, pixelX <= imageWidth - 1,
                  pixelY >= 0, pixelY <= imageHeight - 1 else {
                skippedOutOfImage += 1
                continue
            }

            guard let metricZ = metricDepth.buffer.sampleBilinear(
                pixelX: pixelX,
                pixelY: pixelY,
                imageWidth: intrinsics.imageWidth,
                imageHeight: intrinsics.imageHeight
            ),
            metricZ.isFinite, metricZ > 0.05 else {
                skippedNoDepth += 1
                continue
            }

            let scale = metricZ / z
            guard scale.isFinite, scale > 0.01, scale < 100 else {
                skippedNoDepth += 1
                continue
            }

            let correctedX = point.x * scale
            let correctedY = point.y * scale
            let correctedZ = metricZ

            correctedMinX = min(correctedMinX, correctedX)
            correctedMaxX = max(correctedMaxX, correctedX)
            correctedMinY = min(correctedMinY, correctedY)
            correctedMaxY = max(correctedMaxY, correctedY)
            correctedMinZ = min(correctedMinZ, correctedZ)
            correctedMaxZ = max(correctedMaxZ, correctedZ)
            correctedXs.append(correctedX)
            correctedYs.append(correctedY)
            correctedZs.append(correctedZ)
            correctedPoints.append(SIMD3<Float>(correctedX, correctedY, correctedZ))
            minScale = min(minScale, scale)
            maxScale = max(maxScale, scale)
            scaleSum += Double(scale)
            validCount += 1
        }

        guard validCount > 1024,
              correctedMinX.isFinite, correctedMaxX.isFinite,
              correctedMinY.isFinite, correctedMaxY.isFinite,
              correctedMinZ.isFinite, correctedMaxZ.isFinite else {
            logMetricDepthMeasurement(
                "METRIC PLY ALIGN FAILED VALID=\(validCount) SKIP_OOB=\(skippedOutOfImage) " +
                    "SKIP_NODEPTH=\(skippedNoDepth) SKIP_INVALID_Z=\(skippedInvalidZ)"
            )
            return nil
        }

        let rawWidthMeters = correctedMaxX - correctedMinX
        let rawHeightMeters = correctedMaxY - correctedMinY
        let rawDepthMeters = correctedMaxZ - correctedMinZ
        let meanScale = Float(scaleSum / Double(max(validCount, 1)))

        logMetricDepthMeasurement(
            "METRIC PLY ALIGN STATS SOURCE=\(metricDepth.sourceName) VERTICES=\(ply.vertexCount) VALID=\(validCount) " +
                "SKIP_OOB=\(skippedOutOfImage) SKIP_NODEPTH=\(skippedNoDepth) SKIP_INVALID_Z=\(skippedInvalidZ) " +
                "SCALE_MIN=\(String(format: "%.4f", minScale)) SCALE_MEAN=\(String(format: "%.4f", meanScale)) SCALE_MAX=\(String(format: "%.4f", maxScale))"
        )
        logMetricDepthMeasurement(
            "METRIC PLY ALIGN BOUNDS MIN=(\(String(format: "%.4f", correctedMinX)),\(String(format: "%.4f", correctedMinY)),\(String(format: "%.4f", correctedMinZ))) " +
                "MAX=(\(String(format: "%.4f", correctedMaxX)),\(String(format: "%.4f", correctedMaxY)),\(String(format: "%.4f", correctedMaxZ)))"
        )

        if let ransacResult = measureUsingPlaneRansac(points: correctedPoints, source: "RANSAC_PLANES_DEPTH_WARP") {
            logMetricDepthMeasurement(
                "METRIC PLY ALIGN RESULT WIDTH_M=\(String(format: "%.4f", ransacResult.result.widthMeters)) " +
                    "HEIGHT_M=\(String(format: "%.4f", ransacResult.result.heightMeters)) DEPTH_M=\(String(format: "%.4f", ransacResult.result.depthMeters)) " +
                    "RAW_WHD=(\(String(format: "%.4f", rawWidthMeters)),\(String(format: "%.4f", rawHeightMeters)),\(String(format: "%.4f", rawDepthMeters))) " +
                    "FX=\(String(format: "%.2f", intrinsics.fx)) FY=\(String(format: "%.2f", intrinsics.fy)) SOURCE=\(ransacResult.source)"
            )
            return ransacResult.result
        }
        logMetricDepthMeasurement("METRIC PLY ALIGN RANSAC FALLBACK=BACK_WALL_PERCENTILES")

        let trimLowerPercentile: Float = 0.03
        let trimUpperPercentile: Float = 0.97

        guard let trimmedX = percentileBounds(
            values: correctedXs,
            lowerPercentile: trimLowerPercentile,
            upperPercentile: trimUpperPercentile
        ),
        let trimmedY = percentileBounds(
            values: correctedYs,
            lowerPercentile: trimLowerPercentile,
            upperPercentile: trimUpperPercentile
        ),
        let trimmedZ = percentileBounds(
            values: correctedZs,
            lowerPercentile: trimLowerPercentile,
            upperPercentile: trimUpperPercentile
        ) else {
            logMetricDepthMeasurement("METRIC PLY ALIGN FAILED TO COMPUTE TRIMMED BOUNDS")
            return nil
        }

        let depthLowerPercentile: Float = 0.05
        let backWallLowerPercentile: Float = 0.80
        let depthUpperPercentile: Float = 0.95

        guard let zDepthBand = percentileBounds(
            values: correctedZs,
            lowerPercentile: depthLowerPercentile,
            upperPercentile: depthUpperPercentile
        ),
        let zBackWallBand = percentileBounds(
            values: correctedZs,
            lowerPercentile: backWallLowerPercentile,
            upperPercentile: depthUpperPercentile
        ) else {
            logMetricDepthMeasurement("METRIC PLY ALIGN FAILED TO COMPUTE Z BANDS")
            return nil
        }

        var backWallXs: [Float] = []
        var backWallYs: [Float] = []
        backWallXs.reserveCapacity(correctedXs.count / 5)
        backWallYs.reserveCapacity(correctedYs.count / 5)

        for index in correctedZs.indices {
            let z = correctedZs[index]
            if z >= zBackWallBand.min, z <= zBackWallBand.max {
                backWallXs.append(correctedXs[index])
                backWallYs.append(correctedYs[index])
            }
        }

        let wallWidthPercentileLow: Float = 0.05
        let wallWidthPercentileHigh: Float = 0.95

        let widthHeightSource: String
        let widthMeters: Float
        let heightMeters: Float
        if backWallXs.count > 1024,
           let backWallX = percentileBounds(
               values: backWallXs,
               lowerPercentile: wallWidthPercentileLow,
               upperPercentile: wallWidthPercentileHigh
           ),
           let backWallY = percentileBounds(
               values: backWallYs,
               lowerPercentile: wallWidthPercentileLow,
               upperPercentile: wallWidthPercentileHigh
           ) {
            widthMeters = backWallX.max - backWallX.min
            heightMeters = backWallY.max - backWallY.min
            widthHeightSource = "BACK_WALL_P\(Int(backWallLowerPercentile * 100))_P\(Int(depthUpperPercentile * 100))"
            logMetricDepthMeasurement(
                "METRIC PLY ALIGN BACK_WALL COUNT=\(backWallXs.count) " +
                    "Z_BAND=(\(String(format: "%.4f", zBackWallBand.min)),\(String(format: "%.4f", zBackWallBand.max))) " +
                    "X_BOUNDS_P\(Int(wallWidthPercentileLow * 100))_P\(Int(wallWidthPercentileHigh * 100))=(\(String(format: "%.4f", backWallX.min)),\(String(format: "%.4f", backWallX.max))) " +
                    "Y_BOUNDS_P\(Int(wallWidthPercentileLow * 100))_P\(Int(wallWidthPercentileHigh * 100))=(\(String(format: "%.4f", backWallY.min)),\(String(format: "%.4f", backWallY.max)))"
            )
        } else {
            widthMeters = trimmedX.max - trimmedX.min
            heightMeters = trimmedY.max - trimmedY.min
            widthHeightSource = "ALL_POINTS_TRIMMED_FALLBACK"
            logMetricDepthMeasurement(
                "METRIC PLY ALIGN BACK_WALL FALLBACK COUNT=\(backWallXs.count) " +
                    "USING_TRIMMED_ALL_POINTS"
            )
        }

        let depthMeters = zDepthBand.max - zDepthBand.min

        guard widthMeters.isFinite, heightMeters.isFinite, depthMeters.isFinite,
              widthMeters > 0.05, heightMeters > 0.05, depthMeters > 0.05 else {
            logMetricDepthMeasurement(
                "METRIC PLY ALIGN INVALID WIDTH=\(String(format: "%.4f", widthMeters)) " +
                    "HEIGHT=\(String(format: "%.4f", heightMeters)) DEPTH=\(String(format: "%.4f", depthMeters))"
            )
            return nil
        }

        logMetricDepthMeasurement(
            "METRIC PLY ALIGN TRIMMED_BOUNDS P\(Int(trimLowerPercentile * 100))–P\(Int(trimUpperPercentile * 100)) " +
                "MIN=(\(String(format: "%.4f", trimmedX.min)),\(String(format: "%.4f", trimmedY.min)),\(String(format: "%.4f", trimmedZ.min))) " +
                "MAX=(\(String(format: "%.4f", trimmedX.max)),\(String(format: "%.4f", trimmedY.max)),\(String(format: "%.4f", trimmedZ.max)))"
        )
        logMetricDepthMeasurement(
            "METRIC PLY ALIGN DEPTH_BAND P\(Int(depthLowerPercentile * 100))–P\(Int(depthUpperPercentile * 100)) " +
                "MIN_Z=\(String(format: "%.4f", zDepthBand.min)) MAX_Z=\(String(format: "%.4f", zDepthBand.max))"
        )
        logMetricDepthMeasurement(
            "METRIC PLY ALIGN RESULT WIDTH_M=\(String(format: "%.4f", widthMeters)) " +
                "HEIGHT_M=\(String(format: "%.4f", heightMeters)) DEPTH_M=\(String(format: "%.4f", depthMeters)) " +
                "RAW_WHD=(\(String(format: "%.4f", rawWidthMeters)),\(String(format: "%.4f", rawHeightMeters)),\(String(format: "%.4f", rawDepthMeters))) " +
                "FX=\(String(format: "%.2f", intrinsics.fx)) FY=\(String(format: "%.2f", intrinsics.fy)) " +
                "SOURCE=\(widthHeightSource)"
        )

        return Result(
            widthMeters: widthMeters,
            heightMeters: heightMeters,
            depthMeters: depthMeters,
            calibrationMode: "metric_depth_ply"
        )
    }

    private static func measureUsingImageEdgeLayout(
        thumbnail: UIImage,
        ply: BinaryClassicPly,
        sharpCamera: SharpCameraSidecar.Info
    ) -> (result: Result, source: String)? {
        guard let rawCentroid = rawCentroid(of: ply) else {
            logMetricDepthMeasurement("METRIC PLY ALIGN EDGE LAYOUT FAILED=NO_CENTROID")
            return nil
        }

        if let voxelVoid = measureUsingSharpVoxelVoid(ply: ply, rawCentroid: rawCentroid) {
            logMetricDepthMeasurement(
                "METRIC PLY ALIGN RESULT WIDTH_M=\(String(format: "%.4f", voxelVoid.result.widthMeters)) " +
                    "HEIGHT_M=\(String(format: "%.4f", voxelVoid.result.heightMeters)) DEPTH_M=\(String(format: "%.4f", voxelVoid.result.depthMeters)) " +
                    "SOURCE=\(voxelVoid.source)"
            )
            return voxelVoid
        }
        logMetricDepthMeasurement("METRIC PLY ALIGN VOXEL VOID FALLBACK=2D_OCCUPANCY")

        if let occupancy = detectRoomBoundsFromOccupancy(ply: ply, sharpCamera: sharpCamera) {
            let rectSource = CGRect(
                x: CGFloat(occupancy.leftX),
                y: CGFloat(occupancy.topY),
                width: CGFloat(max(1, occupancy.rightX - occupancy.leftX)),
                height: CGFloat(max(1, occupancy.bottomY - occupancy.topY))
            )
            let depthMeters = rawCentroid.z * sharpGlobalMetricScale
            let fx = sharpCamera.sourceFocalPx
            let fy = sharpCamera.sourceFocalPx
            let cx = sharpCamera.sourceCxPx
            let cy = sharpCamera.sourceCyPx

            let leftM = (Float(rectSource.minX) - cx) * depthMeters / fx
            let rightM = (Float(rectSource.maxX) - cx) * depthMeters / fx
            let topM = (Float(rectSource.minY) - cy) * depthMeters / fy
            let bottomM = (Float(rectSource.maxY) - cy) * depthMeters / fy

            let widthMeters = abs(rightM - leftM)
            let heightMeters = abs(bottomM - topM)

            guard widthMeters.isFinite, heightMeters.isFinite, depthMeters.isFinite,
                  widthMeters > 0.10, heightMeters > 0.10, depthMeters > 0.10 else {
                logMetricDepthMeasurement(
                    "METRIC PLY ALIGN OCCUPANCY INVALID WIDTH=\(String(format: "%.4f", widthMeters)) " +
                        "HEIGHT=\(String(format: "%.4f", heightMeters)) DEPTH=\(String(format: "%.4f", depthMeters))"
                )
                return nil
            }

            logMetricDepthMeasurement(
                "METRIC PLY ALIGN OCCUPANCY SCALE=\(String(format: "%.4f", sharpGlobalMetricScale)) " +
                    "RAW_CENTROID=(\(String(format: "%.4f", rawCentroid.x)),\(String(format: "%.4f", rawCentroid.y)),\(String(format: "%.4f", rawCentroid.z))) " +
                    "DEPTH_M=\(String(format: "%.4f", depthMeters))"
            )
            logMetricDepthMeasurement(
                "METRIC PLY ALIGN OCCUPANCY RECT_SOURCE=[\(occupancy.leftX),\(occupancy.topY),\(occupancy.rightX),\(occupancy.bottomY)] " +
                    "GRID=\(occupancy.gridW)X\(occupancy.gridH) CELL_THRESHOLD=\(occupancy.cellThreshold) " +
                    "LINE_THRESHOLD_X=\(occupancy.minOccupiedCols) LINE_THRESHOLD_Y=\(occupancy.minOccupiedRows) " +
                    "MAX_CELL=\(occupancy.maxCellCount)"
            )
            logMetricDepthMeasurement(
                "METRIC PLY ALIGN OCCUPANCY DENSITY TOP=\(occupancy.topOccupied) BOTTOM=\(occupancy.bottomOccupied) " +
                    "LEFT=\(occupancy.leftOccupied) RIGHT=\(occupancy.rightOccupied) PROJECTED=\(occupancy.projectedCount)"
            )

            return (
                Result(
                    widthMeters: widthMeters,
                    heightMeters: heightMeters,
                    depthMeters: depthMeters,
                    calibrationMode: "sharp_occupancy_layout"
                ),
                "OCCUPANCY_LAYOUT_SHARP_GLOBAL_SCALE_1_08"
            )
        }

        guard let grayscale = grayscaleBuffer(for: thumbnail) else {
            logMetricDepthMeasurement("METRIC PLY ALIGN EDGE LAYOUT FAILED=NO_GRAYSCALE")
            return nil
        }

        let thumbWidth = grayscale.width
        let thumbHeight = grayscale.height
        let verticalScores = verticalEdgeScores(
            grayscale: grayscale.values,
            width: thumbWidth,
            height: thumbHeight,
            yStart: Int(Float(thumbHeight) * 0.18),
            yEnd: Int(Float(thumbHeight) * 0.92)
        )
        let smoothVertical = smoothedScores(verticalScores, radius: 6)

        let leftRange = Int(Float(thumbWidth) * 0.06)...Int(Float(thumbWidth) * 0.45)
        let rightRange = Int(Float(thumbWidth) * 0.55)...Int(Float(thumbWidth) * 0.94)
        guard let leftX = strongestPeak(in: smoothVertical, range: leftRange),
              let rightX = strongestPeak(in: smoothVertical, range: rightRange),
              rightX - leftX > Int(Float(thumbWidth) * 0.18) else {
            logMetricDepthMeasurement("METRIC PLY ALIGN EDGE LAYOUT FAILED=VERTICAL_EDGES")
            return nil
        }

        let innerMargin = max(4, Int(Float(rightX - leftX) * 0.10))
        let xStart = max(1, leftX + innerMargin)
        let xEnd = min(thumbWidth - 2, rightX - innerMargin)
        guard xEnd > xStart else {
            logMetricDepthMeasurement("METRIC PLY ALIGN EDGE LAYOUT FAILED=HORIZONTAL_WINDOW")
            return nil
        }

        let horizontalScores = horizontalEdgeScores(
            grayscale: grayscale.values,
            width: thumbWidth,
            height: thumbHeight,
            xStart: xStart,
            xEnd: xEnd
        )
        let smoothHorizontal = smoothedScores(horizontalScores, radius: 6)

        let topRange = Int(Float(thumbHeight) * 0.05)...Int(Float(thumbHeight) * 0.45)
        let bottomRange = Int(Float(thumbHeight) * 0.55)...Int(Float(thumbHeight) * 0.95)
        guard let topY = strongestPeak(in: smoothHorizontal, range: topRange),
              let bottomY = strongestPeak(in: smoothHorizontal, range: bottomRange),
              bottomY - topY > Int(Float(thumbHeight) * 0.15) else {
            logMetricDepthMeasurement("METRIC PLY ALIGN EDGE LAYOUT FAILED=HORIZONTAL_EDGES")
            return nil
        }

        let sourceScaleX = CGFloat(sharpCamera.sourceImageWidthPx) / CGFloat(thumbWidth)
        let sourceScaleY = CGFloat(sharpCamera.sourceImageHeightPx) / CGFloat(thumbHeight)
        let topSourceY = CGFloat(topY) * sourceScaleY
        let fallbackBottomSourceY = CGFloat(bottomY) * sourceScaleY
        let leftSourceX = CGFloat(leftX) * sourceScaleX
        let rightSourceX = CGFloat(rightX) * sourceScaleX
        let sourceXInset = max(CGFloat(8), (rightSourceX - leftSourceX) * 0.15)
        let xBandMin = leftSourceX + sourceXInset
        let xBandMax = rightSourceX - sourceXInset
        let floorJunctionSourceY = detectFloorWallJunction(
            ply: ply,
            sharpCamera: sharpCamera,
            minX: Float(xBandMin),
            maxX: Float(xBandMax),
            minY: Float(topSourceY),
            fallbackY: Float(fallbackBottomSourceY)
        )
        let bottomSourceY = CGFloat(floorJunctionSourceY ?? Float(fallbackBottomSourceY))
        let rectThumb = CGRect(
            x: CGFloat(leftX),
            y: CGFloat(topY),
            width: CGFloat(rightX - leftX),
            height: max(1, bottomSourceY / sourceScaleY - CGFloat(topY))
        )
        let rectSource = CGRect(
            x: rectThumb.minX * sourceScaleX,
            y: rectThumb.minY * sourceScaleY,
            width: rectThumb.width * sourceScaleX,
            height: max(1, bottomSourceY - topSourceY)
        )

        let depthMeters = rawCentroid.z * sharpGlobalMetricScale
        let fx = sharpCamera.sourceFocalPx
        let fy = sharpCamera.sourceFocalPx
        let cx = sharpCamera.sourceCxPx
        let cy = sharpCamera.sourceCyPx

        let leftM = (Float(rectSource.minX) - cx) * depthMeters / fx
        let rightM = (Float(rectSource.maxX) - cx) * depthMeters / fx
        let topM = (Float(rectSource.minY) - cy) * depthMeters / fy
        let bottomM = (Float(rectSource.maxY) - cy) * depthMeters / fy

        let widthMeters = abs(rightM - leftM)
        let heightMeters = abs(bottomM - topM)

        guard widthMeters.isFinite, heightMeters.isFinite, depthMeters.isFinite,
              widthMeters > 0.10, heightMeters > 0.10, depthMeters > 0.10 else {
            logMetricDepthMeasurement(
                "METRIC PLY ALIGN EDGE LAYOUT INVALID WIDTH=\(String(format: "%.4f", widthMeters)) " +
                    "HEIGHT=\(String(format: "%.4f", heightMeters)) DEPTH=\(String(format: "%.4f", depthMeters))"
            )
            return nil
        }

        logMetricDepthMeasurement(
            "METRIC PLY ALIGN EDGE LAYOUT SCALE=\(String(format: "%.4f", sharpGlobalMetricScale)) " +
                "RAW_CENTROID=(\(String(format: "%.4f", rawCentroid.x)),\(String(format: "%.4f", rawCentroid.y)),\(String(format: "%.4f", rawCentroid.z))) " +
                "DEPTH_M=\(String(format: "%.4f", depthMeters))"
        )
        logMetricDepthMeasurement(
            "METRIC PLY ALIGN EDGE LAYOUT RECT_THUMB=[\(leftX),\(topY),\(rightX),\(bottomY)] " +
                "RECT_SOURCE=[\(Int(rectSource.minX.rounded())),\(Int(rectSource.minY.rounded())),\(Int(rectSource.maxX.rounded())),\(Int(rectSource.maxY.rounded()))] " +
                "BOTTOM_SOURCE_USED=\(Int(bottomSourceY.rounded()))"
        )
        logMetricDepthMeasurement(
            "METRIC PLY ALIGN EDGE LAYOUT SCORES LEFT=\(String(format: "%.2f", smoothVertical[leftX])) " +
                "RIGHT=\(String(format: "%.2f", smoothVertical[rightX])) TOP=\(String(format: "%.2f", smoothHorizontal[topY])) " +
                "BOTTOM=\(String(format: "%.2f", smoothHorizontal[bottomY]))"
        )

        return (
            Result(
                widthMeters: widthMeters,
                heightMeters: heightMeters,
                depthMeters: depthMeters,
                calibrationMode: "sharp_edge_layout"
            ),
            "EDGE_LAYOUT_SHARP_GLOBAL_SCALE_1_08"
        )
    }

    private static func measureUsingSharpVoxelVoid(
        ply: BinaryClassicPly,
        rawCentroid: SIMD3<Float>
    ) -> (result: Result, source: String)? {
        let scale = sharpGlobalMetricScale
        let baseVoxelSize: Float = 0.05
        let maxVoxelCount = 4_500_000
        let paddingVoxels = 2
        let depthMeters = rawCentroid.z * scale

        var scaledPoints: [SIMD3<Float>] = []
        scaledPoints.reserveCapacity(min(ply.vertexCount, 900_000))
        var minPoint = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
        var maxPoint = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)

        for index in 0..<ply.vertexCount {
            let point = ply.position(at: index) * scale
            guard point.x.isFinite, point.y.isFinite, point.z.isFinite else { continue }
            scaledPoints.append(point)
            minPoint = simd_min(minPoint, point)
            maxPoint = simd_max(maxPoint, point)
        }

        guard scaledPoints.count >= 4096 else {
            logMetricDepthMeasurement("METRIC PLY ALIGN VOXEL VOID FAILED=INSUFFICIENT_POINTS COUNT=\(scaledPoints.count)")
            return nil
        }

        var voxelSize = baseVoxelSize
        var gridX = 0
        var gridY = 0
        var gridZ = 0
        while true {
            gridX = Int(ceil((maxPoint.x - minPoint.x) / voxelSize)) + 1 + paddingVoxels * 2
            gridY = Int(ceil((maxPoint.y - minPoint.y) / voxelSize)) + 1 + paddingVoxels * 2
            gridZ = Int(ceil((maxPoint.z - minPoint.z) / voxelSize)) + 1 + paddingVoxels * 2
            let total = gridX * gridY * gridZ
            if total <= maxVoxelCount || voxelSize >= 0.15 {
                break
            }
            voxelSize *= 1.2
        }

        guard gridX > 0, gridY > 0, gridZ > 0 else {
            logMetricDepthMeasurement("METRIC PLY ALIGN VOXEL VOID FAILED=INVALID_GRID")
            return nil
        }

        let origin = minPoint - SIMD3<Float>(repeating: Float(paddingVoxels) * voxelSize)
        let totalVoxels = gridX * gridY * gridZ
        var occupied = [UInt8](repeating: 0, count: totalVoxels)

        @inline(__always)
        func linearIndex(_ x: Int, _ y: Int, _ z: Int) -> Int {
            (z * gridY + y) * gridX + x
        }

        @inline(__always)
        func voxelCoord(for point: SIMD3<Float>) -> SIMD3<Int> {
            SIMD3<Int>(
                Int(((point.x - origin.x) / voxelSize).rounded(.down)).clamped(to: 0...(gridX - 1)),
                Int(((point.y - origin.y) / voxelSize).rounded(.down)).clamped(to: 0...(gridY - 1)),
                Int(((point.z - origin.z) / voxelSize).rounded(.down)).clamped(to: 0...(gridZ - 1))
            )
        }

        for point in scaledPoints {
            let voxel = voxelCoord(for: point)
            occupied[linearIndex(voxel.x, voxel.y, voxel.z)] = 1
        }

        let centroid = rawCentroid * scale
        let centroidVoxel = voxelCoord(for: centroid)
        guard let start = nearestEmptyVoxel(
            around: centroidVoxel,
            occupied: occupied,
            gridX: gridX,
            gridY: gridY,
            gridZ: gridZ
        ) else {
            logMetricDepthMeasurement("METRIC PLY ALIGN VOXEL VOID FAILED=NO_EMPTY_START")
            return nil
        }

        var visited = [UInt8](repeating: 0, count: totalVoxels)
        var queue: [SIMD3<Int>] = [start]
        queue.reserveCapacity(min(totalVoxels / 8, 500_000))
        visited[linearIndex(start.x, start.y, start.z)] = 1
        var head = 0
        var minEmpty = start
        var maxEmpty = start
        var visitedCount = 0

        let neighborOffsets = [
            SIMD3<Int>(1, 0, 0), SIMD3<Int>(-1, 0, 0),
            SIMD3<Int>(0, 1, 0), SIMD3<Int>(0, -1, 0),
            SIMD3<Int>(0, 0, 1), SIMD3<Int>(0, 0, -1),
        ]

        while head < queue.count {
            let current = queue[head]
            head += 1
            visitedCount += 1
            minEmpty = SIMD3<Int>(min(minEmpty.x, current.x), min(minEmpty.y, current.y), min(minEmpty.z, current.z))
            maxEmpty = SIMD3<Int>(max(maxEmpty.x, current.x), max(maxEmpty.y, current.y), max(maxEmpty.z, current.z))

            for offset in neighborOffsets {
                let next = current &+ offset
                guard next.x >= 0, next.x < gridX,
                      next.y >= 0, next.y < gridY,
                      next.z >= 0, next.z < gridZ else { continue }
                let idx = linearIndex(next.x, next.y, next.z)
                if visited[idx] != 0 || occupied[idx] != 0 { continue }
                visited[idx] = 1
                queue.append(next)
            }
        }

        guard visitedCount > 128 else {
            logMetricDepthMeasurement("METRIC PLY ALIGN VOXEL VOID FAILED=EMPTY_REGION_TOO_SMALL COUNT=\(visitedCount)")
            return nil
        }

        let widthMeters = Float(maxEmpty.x - minEmpty.x + 1) * voxelSize
        let heightMeters = Float(maxEmpty.y - minEmpty.y + 1) * voxelSize
        let interiorDepthMeters = Float(maxEmpty.z - minEmpty.z + 1) * voxelSize

        guard widthMeters.isFinite, heightMeters.isFinite, depthMeters.isFinite,
              widthMeters > 0.10, heightMeters > 0.10, depthMeters > 0.10 else {
            logMetricDepthMeasurement(
                "METRIC PLY ALIGN VOXEL VOID INVALID WIDTH=\(String(format: "%.4f", widthMeters)) " +
                    "HEIGHT=\(String(format: "%.4f", heightMeters)) DEPTH=\(String(format: "%.4f", depthMeters))"
            )
            return nil
        }

        let occupiedCount = occupied.reduce(into: 0) { partial, value in
            partial += Int(value)
        }
        logMetricDepthMeasurement(
            "METRIC PLY ALIGN VOXEL VOID SCALE=\(String(format: "%.4f", scale)) VOXEL_M=\(String(format: "%.4f", voxelSize)) " +
                "RAW_CENTROID=(\(String(format: "%.4f", rawCentroid.x)),\(String(format: "%.4f", rawCentroid.y)),\(String(format: "%.4f", rawCentroid.z))) " +
                "DEPTH_M=\(String(format: "%.4f", depthMeters))"
        )
        logMetricDepthMeasurement(
            "METRIC PLY ALIGN VOXEL VOID GRID=\(gridX)X\(gridY)X\(gridZ) OCCUPIED=\(occupiedCount) " +
                "EMPTY_REGION=\(visitedCount) START=(\(start.x),\(start.y),\(start.z))"
        )
        logMetricDepthMeasurement(
            "METRIC PLY ALIGN VOXEL VOID GAP_X=(\(minEmpty.x),\(maxEmpty.x)) GAP_Y=(\(minEmpty.y),\(maxEmpty.y)) GAP_Z=(\(minEmpty.z),\(maxEmpty.z)) " +
                "WIDTH_M=\(String(format: "%.4f", widthMeters)) HEIGHT_M=\(String(format: "%.4f", heightMeters)) " +
                "INTERIOR_DEPTH_M=\(String(format: "%.4f", interiorDepthMeters))"
        )

        return (
            Result(
                widthMeters: widthMeters,
                heightMeters: heightMeters,
                depthMeters: depthMeters,
                calibrationMode: "sharp_voxel_void"
            ),
            "VOXEL_VOID_SHARP_GLOBAL_SCALE_1_08"
        )
    }

    private static func nearestEmptyVoxel(
        around start: SIMD3<Int>,
        occupied: [UInt8],
        gridX: Int,
        gridY: Int,
        gridZ: Int
    ) -> SIMD3<Int>? {
        @inline(__always)
        func linearIndex(_ x: Int, _ y: Int, _ z: Int) -> Int {
            (z * gridY + y) * gridX + x
        }

        if occupied[linearIndex(start.x, start.y, start.z)] == 0 {
            return start
        }

        for radius in 1...8 {
            for dx in -radius...radius {
                for dy in -radius...radius {
                    for dz in -radius...radius {
                        let x = start.x + dx
                        let y = start.y + dy
                        let z = start.z + dz
                        guard x >= 0, x < gridX, y >= 0, y < gridY, z >= 0, z < gridZ else { continue }
                        if occupied[linearIndex(x, y, z)] == 0 {
                            return SIMD3<Int>(x, y, z)
                        }
                    }
                }
            }
        }
        return nil
    }

    private static func detectRoomBoundsFromOccupancy(
        ply: BinaryClassicPly,
        sharpCamera: SharpCameraSidecar.Info
    ) -> (
        topY: Int,
        bottomY: Int,
        leftX: Int,
        rightX: Int,
        gridW: Int,
        gridH: Int,
        cellThreshold: Int,
        minOccupiedCols: Int,
        minOccupiedRows: Int,
        maxCellCount: Int,
        topOccupied: Int,
        bottomOccupied: Int,
        leftOccupied: Int,
        rightOccupied: Int,
        projectedCount: Int
    )? {
        let gridW = 96
        let gridH = 96
        let fx = sharpCamera.sourceFocalPx
        let fy = sharpCamera.sourceFocalPx
        let cx = sharpCamera.sourceCxPx
        let cy = sharpCamera.sourceCyPx
        let imgW = Float(sharpCamera.sourceImageWidthPx)
        let imgH = Float(sharpCamera.sourceImageHeightPx)
        guard imgW > 1, imgH > 1 else { return nil }

        var grid = [Int](repeating: 0, count: gridW * gridH)
        var projectedCount = 0
        var maxCellCount = 0

        for index in 0..<ply.vertexCount {
            let point = ply.position(at: index)
            guard point.z.isFinite, point.z > 0.05, point.x.isFinite, point.y.isFinite else { continue }

            let px = fx * point.x / point.z + cx
            let py = fy * point.y / point.z + cy
            guard px >= 0, px <= imgW - 1, py >= 0, py <= imgH - 1 else { continue }

            let gx = Int((px / imgW) * Float(gridW)).clamped(to: 0...(gridW - 1))
            let gy = Int((py / imgH) * Float(gridH)).clamped(to: 0...(gridH - 1))
            let idx = gy * gridW + gx
            grid[idx] += 1
            maxCellCount = max(maxCellCount, grid[idx])
            projectedCount += 1
        }

        guard projectedCount > 4096, maxCellCount > 0 else {
            logMetricDepthMeasurement("METRIC PLY ALIGN OCCUPANCY FAILED=INSUFFICIENT_PROJECTIONS COUNT=\(projectedCount)")
            return nil
        }

        let cellThreshold = max(8, Int(Float(maxCellCount) * 0.12))
        let minOccupiedCols = max(6, gridW / 5)
        let minOccupiedRows = max(6, gridH / 5)

        func rowOccupiedCount(_ row: Int) -> Int {
            var occupied = 0
            let base = row * gridW
            for col in 0..<gridW where grid[base + col] >= cellThreshold {
                occupied += 1
            }
            return occupied
        }

        func colOccupiedCount(_ col: Int) -> Int {
            var occupied = 0
            for row in 0..<gridH where grid[row * gridW + col] >= cellThreshold {
                occupied += 1
            }
            return occupied
        }

        var topRow = 0
        var topOccupied = 0
        for row in 0..<gridH {
            let occupied = rowOccupiedCount(row)
            if occupied >= minOccupiedCols {
                topRow = row
                topOccupied = occupied
                break
            }
        }

        var bottomRow = gridH - 1
        var bottomOccupied = 0
        for row in stride(from: gridH - 1, through: 0, by: -1) {
            let occupied = rowOccupiedCount(row)
            if occupied >= minOccupiedCols {
                bottomRow = row
                bottomOccupied = occupied
                break
            }
        }

        var leftCol = 0
        var leftOccupied = 0
        for col in 0..<gridW {
            let occupied = colOccupiedCount(col)
            if occupied >= minOccupiedRows {
                leftCol = col
                leftOccupied = occupied
                break
            }
        }

        var rightCol = gridW - 1
        var rightOccupied = 0
        for col in stride(from: gridW - 1, through: 0, by: -1) {
            let occupied = colOccupiedCount(col)
            if occupied >= minOccupiedRows {
                rightCol = col
                rightOccupied = occupied
                break
            }
        }

        guard rightCol > leftCol, bottomRow > topRow else {
            logMetricDepthMeasurement(
                "METRIC PLY ALIGN OCCUPANCY FAILED=INVALID_BOUNDS TOP=\(topRow) BOTTOM=\(bottomRow) LEFT=\(leftCol) RIGHT=\(rightCol)"
            )
            return nil
        }

        let topY = Int((Float(topRow) / Float(gridH)) * imgH)
        let bottomY = Int((Float(bottomRow + 1) / Float(gridH)) * imgH)
        let leftX = Int((Float(leftCol) / Float(gridW)) * imgW)
        let rightX = Int((Float(rightCol + 1) / Float(gridW)) * imgW)

        return (
            topY: topY,
            bottomY: bottomY,
            leftX: leftX,
            rightX: rightX,
            gridW: gridW,
            gridH: gridH,
            cellThreshold: cellThreshold,
            minOccupiedCols: minOccupiedCols,
            minOccupiedRows: minOccupiedRows,
            maxCellCount: maxCellCount,
            topOccupied: topOccupied,
            bottomOccupied: bottomOccupied,
            leftOccupied: leftOccupied,
            rightOccupied: rightOccupied,
            projectedCount: projectedCount
        )
    }

    private static func detectFloorWallJunction(
        ply: BinaryClassicPly,
        sharpCamera: SharpCameraSidecar.Info,
        minX: Float,
        maxX: Float,
        minY: Float,
        fallbackY: Float
    ) -> Float? {
        let fx = sharpCamera.sourceFocalPx
        let fy = sharpCamera.sourceFocalPx
        let cx = sharpCamera.sourceCxPx
        let cy = sharpCamera.sourceCyPx
        let imageHeight = Float(sharpCamera.sourceImageHeightPx)
        let bucketCount = 96
        var buckets = [[Float]](repeating: [], count: bucketCount)
        var projectedCount = 0

        for index in 0..<ply.vertexCount {
            let point = ply.position(at: index)
            guard point.x.isFinite, point.y.isFinite, point.z.isFinite, point.z > 0.05 else { continue }
            let pixelX = fx * point.x / point.z + cx
            let pixelY = fy * point.y / point.z + cy
            guard pixelX.isFinite, pixelY.isFinite,
                  pixelX >= minX, pixelX <= maxX,
                  pixelY >= minY, pixelY <= imageHeight - 1 else { continue }
            let bucketFloat = (pixelY / max(imageHeight, 1)) * Float(bucketCount)
            let bucket = Int(bucketFloat).clamped(to: 0...(bucketCount - 1))
            buckets[bucket].append(point.z)
            projectedCount += 1
        }

        guard projectedCount > 2048 else {
            logMetricDepthMeasurement(
                "METRIC PLY ALIGN FLOOR_JUNCTION FAILED=INSUFFICIENT_POINTS COUNT=\(projectedCount) " +
                    "XBAND=(\(String(format: "%.1f", minX)),\(String(format: "%.1f", maxX)))"
            )
            return nil
        }

        var medians = [Float](repeating: 0, count: bucketCount)
        var hasValue = [Bool](repeating: false, count: bucketCount)
        var variances = [Float](repeating: 0, count: bucketCount)
        for index in 0..<bucketCount where !buckets[index].isEmpty {
            let sorted = buckets[index].sorted()
            medians[index] = sorted[sorted.count / 2]
            hasValue[index] = true
            if buckets[index].count >= 2 {
                let mean = buckets[index].reduce(0, +) / Float(buckets[index].count)
                let variance = buckets[index].reduce(0) { partial, value in
                    let delta = value - mean
                    return partial + delta * delta
                } / Float(buckets[index].count)
                variances[index] = variance
            }
        }
        fillMissingMedians(values: &medians, hasValue: hasValue)
        fillMissingMedians(values: &variances, hasValue: hasValue)
        let smoothMedians = smoothedScores(medians, radius: 2)
        let smoothVariances = smoothedScores(variances, radius: 2)

        let startBucket = max(Int((minY / max(imageHeight, 1)) * Float(bucketCount)), bucketCount / 3)
        let endBucket = max(startBucket + 1, Int(Float(bucketCount) * 0.95))
        var bestVarianceBucket = Int((fallbackY / max(imageHeight, 1)) * Float(bucketCount)).clamped(to: 0...(bucketCount - 2))
        var bestVarianceJump: Float = 0
        var bestMedianBucket = bestVarianceBucket
        var bestMedianDrop: Float = 0

        if endBucket > startBucket {
            for index in startBucket..<min(endBucket, bucketCount - 1) {
                let varianceJump = smoothVariances[index + 1] - smoothVariances[index]
                if varianceJump.isFinite, varianceJump > bestVarianceJump {
                    bestVarianceJump = varianceJump
                    bestVarianceBucket = index + 1
                }

                let medianDrop = smoothMedians[index] - smoothMedians[index + 1]
                if medianDrop.isFinite, medianDrop > bestMedianDrop {
                    bestMedianDrop = medianDrop
                    bestMedianBucket = index
                }
            }
        }

        let selectedBucket: Int
        let selectionSource: String
        if bestVarianceJump > 0.002 {
            selectedBucket = bestVarianceBucket
            selectionSource = "VARIANCE_JUMP"
        } else if bestMedianDrop > 0.01 {
            selectedBucket = bestMedianBucket
            selectionSource = "MEDIAN_DROP"
        } else {
            logMetricDepthMeasurement("METRIC PLY ALIGN FLOOR_JUNCTION FALLBACK=EDGE_BOTTOM")
            return nil
        }

        let junctionY = (Float(selectedBucket) + 0.5) / Float(bucketCount) * imageHeight
        let zHere = smoothMedians[selectedBucket]
        let zNext = smoothMedians[min(selectedBucket + 1, bucketCount - 1)]
        let varianceHere = smoothVariances[selectedBucket]
        let variancePrev = smoothVariances[max(0, selectedBucket - 1)]
        logMetricDepthMeasurement(
            "METRIC PLY ALIGN FLOOR_JUNCTION XBAND=(\(String(format: "%.1f", minX)),\(String(format: "%.1f", maxX))) " +
                "MIN_Y=\(String(format: "%.1f", minY)) FALLBACK_Y=\(String(format: "%.1f", fallbackY)) POINTS=\(projectedCount) " +
                "SELECT=\(selectionSource) BUCKET=\(selectedBucket) JUNCTION_Y=\(String(format: "%.1f", junctionY)) " +
                "VAR_PREV=\(String(format: "%.4f", variancePrev)) VAR_CUR=\(String(format: "%.4f", varianceHere)) " +
                "VAR_JUMP=\(String(format: "%.4f", bestVarianceJump)) Z0=\(String(format: "%.4f", zHere)) " +
                "Z1=\(String(format: "%.4f", zNext)) MEDIAN_DROP=\(String(format: "%.4f", bestMedianDrop))"
        )
        return junctionY
    }

    private static func measureUsingSharpGlobalScaleRansac(
        ply: BinaryClassicPly
    ) -> (result: Result, source: String)? {
        guard ply.vertexCount >= 4096 else {
            logMetricDepthMeasurement("METRIC PLY ALIGN GLOBAL SCALE SKIP=INSUFFICIENT_POINTS COUNT=\(ply.vertexCount)")
            return nil
        }

        let stride = max(1, ply.vertexCount / 20_000)
        var sampled: [SIMD3<Float>] = []
        sampled.reserveCapacity(min(ply.vertexCount, 20_000))
        var centroidSum = SIMD3<Float>.zero
        var centroidCount = 0

        for index in 0..<ply.vertexCount {
            let point = ply.position(at: index)
            guard point.x.isFinite, point.y.isFinite, point.z.isFinite else { continue }
            centroidSum += point
            centroidCount += 1
            if index % stride == 0 {
                sampled.append(point * sharpGlobalMetricScale)
            }
        }

        guard centroidCount >= 4096 else {
            logMetricDepthMeasurement("METRIC PLY ALIGN GLOBAL SCALE SKIP=INVALID_CENTROID COUNT=\(centroidCount)")
            return nil
        }

        let centroid = centroidSum / Float(centroidCount)
        let scaledCentroid = centroid * sharpGlobalMetricScale
        logMetricDepthMeasurement(
            "METRIC PLY ALIGN GLOBAL SCALE SCALE=\(String(format: "%.4f", sharpGlobalMetricScale)) " +
                "RAW_CENTROID=(\(String(format: "%.4f", centroid.x)),\(String(format: "%.4f", centroid.y)),\(String(format: "%.4f", centroid.z))) " +
                "SCALED_CENTROID=(\(String(format: "%.4f", scaledCentroid.x)),\(String(format: "%.4f", scaledCentroid.y)),\(String(format: "%.4f", scaledCentroid.z))) " +
                "USED=\(sampled.count)"
        )

        guard let ransac = measureUsingPlaneRansac(
            points: sampled,
            source: "RANSAC_PLANES_SHARP_GLOBAL_SCALE_1_08"
        ) else {
            logMetricDepthMeasurement("METRIC PLY ALIGN GLOBAL SCALE RANSAC FAILED")
            return nil
        }

        return (
            Result(
                widthMeters: ransac.result.widthMeters,
                heightMeters: ransac.result.heightMeters,
                depthMeters: ransac.result.depthMeters,
                calibrationMode: "sharp_global_scale_ransac"
            ),
            ransac.source
        )
    }

    private static func measureUsingPlaneRansac(
        points: [SIMD3<Float>],
        source: String
    ) -> (result: Result, source: String)? {
        guard points.count >= 4096 else {
            logMetricDepthMeasurement("METRIC PLY ALIGN RANSAC SKIP=INSUFFICIENT_POINTS COUNT=\(points.count)")
            return nil
        }

        let sampled = uniformSubsample(points: points, targetCount: 20_000)
        guard sampled.count >= 2048 else {
            logMetricDepthMeasurement("METRIC PLY ALIGN RANSAC SKIP=INSUFFICIENT_SUBSAMPLE COUNT=\(sampled.count)")
            return nil
        }

        var config = RoomGeometryConfig()
        config.useBoundsBasedRoomSize = false
        config.ransacIterations = 160
        config.ransacInlierThreshold = 0.05
        config.ransacMinInlierFraction = 0.05

        let engine = RoomGeometryEngine(
            depthQuery: NullMetricDepthQuery(points: sampled),
            colorReader: NullMetricColorReader(),
            cameraInfo: SourceCameraInfo(),
            config: config
        )

        srand48(42)

        let sceneBounds = engine.computeAABB(points: sampled)
        let sceneFloor: DetectedPlane
        do {
            sceneFloor = try engine.extractFloorPlane(points: sampled, bounds: sceneBounds)
        } catch {
            logMetricDepthMeasurement("METRIC PLY ALIGN RANSAC FAILED FLOOR ERROR=\(String(describing: error).uppercased())")
            return nil
        }

        let transform = RoomSpaceTransform(floor: sceneFloor)
        let roomPoints = transform.transformPoints(sampled)
        let roomBounds = engine.computeAABB(points: roomPoints)
        let roomFloor = DetectedPlane(type: .floor, normal: SIMD3<Float>(0, 1, 0), pointOnPlane: .zero)
        let roomCeiling = engine.extractCeilingPlane(points: roomPoints, bounds: roomBounds, floor: roomFloor) ??
            engine.syntheticCeiling(floor: roomFloor, bounds: roomBounds)

        let roomWalls: [DetectedPlane]
        do {
            roomWalls = try engine.extractWallPlanes(points: roomPoints, floor: roomFloor, ceiling: roomCeiling)
        } catch {
            logMetricDepthMeasurement("METRIC PLY ALIGN RANSAC FAILED WALLS ERROR=\(String(describing: error).uppercased())")
            return nil
        }

        let roomModel = RoomModel(
            aabb: roomBounds,
            floor: roomFloor,
            ceiling: roomCeiling,
            walls: roomWalls,
            corners: [],
            freeFloorRegions: [],
            surfacePalette: .empty,
            cameraInfo: nil,
            sceneToMeters: 1.0
        )

        let extent = roomModel.planeAwareSceneExtent
        let footprint = roomModel.interiorFootprintSceneUnits
        let height = roomModel.heightMeters
        let wallCount = roomWalls.count
        let opposingPairs = opposingWallPairCount(walls: roomWalls)

        logMetricDepthMeasurement(
            "METRIC PLY ALIGN RANSAC SAMPLE TOTAL=\(points.count) USED=\(sampled.count) " +
                "ROOM_BOUNDS=(\(String(format: "%.4f", roomBounds.size.x)),\(String(format: "%.4f", roomBounds.size.y)),\(String(format: "%.4f", roomBounds.size.z)))"
        )
        logMetricDepthMeasurement(
            "METRIC PLY ALIGN RANSAC FLOOR NORMAL=(\(String(format: "%.4f", sceneFloor.normal.x)),\(String(format: "%.4f", sceneFloor.normal.y)),\(String(format: "%.4f", sceneFloor.normal.z))) " +
                "POINT=(\(String(format: "%.4f", sceneFloor.pointOnPlane.x)),\(String(format: "%.4f", sceneFloor.pointOnPlane.y)),\(String(format: "%.4f", sceneFloor.pointOnPlane.z)))"
        )
        logMetricDepthMeasurement(
            "METRIC PLY ALIGN RANSAC CEILING POINT=(\(String(format: "%.4f", roomCeiling.pointOnPlane.x)),\(String(format: "%.4f", roomCeiling.pointOnPlane.y)),\(String(format: "%.4f", roomCeiling.pointOnPlane.z))) " +
                "HEIGHT_M=\(String(format: "%.4f", height))"
        )
        logMetricDepthMeasurement(
            "METRIC PLY ALIGN RANSAC WALLS COUNT=\(wallCount) OPPOSING_PAIRS=\(opposingPairs) " +
                "FOOTPRINT_M=(\(String(format: "%.4f", footprint.width)),\(String(format: "%.4f", footprint.depth)))"
        )

        for (index, wall) in roomWalls.enumerated().prefix(6) {
            logMetricDepthMeasurement(
                "METRIC PLY ALIGN RANSAC WALL[\(index)] NORMAL=(\(String(format: "%.4f", wall.normal.x)),\(String(format: "%.4f", wall.normal.y)),\(String(format: "%.4f", wall.normal.z))) " +
                    "POINT=(\(String(format: "%.4f", wall.pointOnPlane.x)),\(String(format: "%.4f", wall.pointOnPlane.y)),\(String(format: "%.4f", wall.pointOnPlane.z)))"
            )
        }

        guard extent.width.isFinite, extent.height.isFinite, extent.depth.isFinite,
              extent.width > 0.05, extent.height > 0.05, extent.depth > 0.05,
              wallCount >= 2 else {
            logMetricDepthMeasurement(
                "METRIC PLY ALIGN RANSAC INVALID WIDTH=\(String(format: "%.4f", extent.width)) " +
                    "HEIGHT=\(String(format: "%.4f", extent.height)) DEPTH=\(String(format: "%.4f", extent.depth)) " +
                    "WALLS=\(wallCount)"
            )
            return nil
        }

        return (
            Result(
                widthMeters: extent.width,
                heightMeters: extent.height,
                depthMeters: extent.depth,
                calibrationMode: "metric_depth_ply_ransac"
            ),
            source
        )
    }

    private static func rawCentroid(of ply: BinaryClassicPly) -> SIMD3<Float>? {
        var sum = SIMD3<Float>.zero
        var count = 0
        for index in 0..<ply.vertexCount {
            let point = ply.position(at: index)
            guard point.x.isFinite, point.y.isFinite, point.z.isFinite else { continue }
            sum += point
            count += 1
        }
        guard count > 0 else { return nil }
        return sum / Float(count)
    }

    private static func grayscaleBuffer(
        for image: UIImage,
        maxLongEdge: Int = 384
    ) -> (values: [UInt8], width: Int, height: Int)? {
        let pixelSize = sourceImagePixelSize(for: image)
        guard pixelSize.width > 0, pixelSize.height > 0 else { return nil }

        let scale = min(1.0, Float(maxLongEdge) / Float(max(pixelSize.width, pixelSize.height)))
        let width = max(16, Int((Float(pixelSize.width) * scale).rounded()))
        let height = max(16, Int((Float(pixelSize.height) * scale).rounded()))

        let format = UIGraphicsImageRendererFormat.default()
        format.opaque = true
        format.scale = 1
        let renderer = UIGraphicsImageRenderer(size: CGSize(width: width, height: height), format: format)
        let normalized = renderer.image { _ in
            image.draw(in: CGRect(x: 0, y: 0, width: width, height: height))
        }
        guard let cgImage = normalized.cgImage else { return nil }

        var rgba = [UInt8](repeating: 0, count: width * height * 4)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(
            data: &rgba,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
        ) else {
            return nil
        }
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))

        var gray = [UInt8](repeating: 0, count: width * height)
        for index in 0..<(width * height) {
            let base = index * 4
            let r = Int(rgba[base])
            let g = Int(rgba[base + 1])
            let b = Int(rgba[base + 2])
            gray[index] = UInt8((77 * r + 150 * g + 29 * b) >> 8)
        }
        return (gray, width, height)
    }

    private static func verticalEdgeScores(
        grayscale: [UInt8],
        width: Int,
        height: Int,
        yStart: Int,
        yEnd: Int
    ) -> [Float] {
        var scores = [Float](repeating: 0, count: width)
        guard width > 2, height > 2 else { return scores }
        let ys = max(1, yStart)
        let ye = min(height - 2, yEnd)
        guard ye > ys else { return scores }
        for y in ys...ye {
            let row = y * width
            for x in 1..<(width - 1) {
                let left = Int(grayscale[row + x - 1])
                let right = Int(grayscale[row + x + 1])
                scores[x] += Float(abs(right - left))
            }
        }
        return scores
    }

    private static func horizontalEdgeScores(
        grayscale: [UInt8],
        width: Int,
        height: Int,
        xStart: Int,
        xEnd: Int
    ) -> [Float] {
        var scores = [Float](repeating: 0, count: height)
        guard width > 2, height > 2 else { return scores }
        let xs = max(1, xStart)
        let xe = min(width - 2, xEnd)
        guard xe > xs else { return scores }
        for y in 1..<(height - 1) {
            let rowAbove = (y - 1) * width
            let rowBelow = (y + 1) * width
            for x in xs...xe {
                let above = Int(grayscale[rowAbove + x])
                let below = Int(grayscale[rowBelow + x])
                scores[y] += Float(abs(below - above))
            }
        }
        return scores
    }

    private static func smoothedScores(_ values: [Float], radius: Int) -> [Float] {
        guard !values.isEmpty, radius > 0 else { return values }
        var smoothed = [Float](repeating: 0, count: values.count)
        for index in values.indices {
            let lo = max(0, index - radius)
            let hi = min(values.count - 1, index + radius)
            let count = hi - lo + 1
            guard count > 0 else { continue }
            var sum: Float = 0
            for i in lo...hi {
                sum += values[i]
            }
            smoothed[index] = sum / Float(count)
        }
        return smoothed
    }

    private static func fillMissingMedians(values: inout [Float], hasValue: [Bool]) {
        guard values.count == hasValue.count, !values.isEmpty else { return }

        var lastKnown: Float?
        for index in values.indices {
            if hasValue[index] {
                lastKnown = values[index]
            } else if let lastKnown {
                values[index] = lastKnown
            }
        }

        var nextKnown: Float?
        for index in values.indices.reversed() {
            if hasValue[index] {
                nextKnown = values[index]
            } else if let nextKnown {
                values[index] = nextKnown
            }
        }
    }

    private static func strongestPeak(
        in scores: [Float],
        range: ClosedRange<Int>
    ) -> Int? {
        guard !scores.isEmpty else { return nil }
        let lower = max(0, range.lowerBound)
        let upper = min(scores.count - 1, range.upperBound)
        guard upper >= lower else { return nil }
        var bestIndex = lower
        var bestScore = -Float.greatestFiniteMagnitude
        for index in lower...upper where scores[index].isFinite && scores[index] > bestScore {
            bestScore = scores[index]
            bestIndex = index
        }
        guard bestScore > 0 else { return nil }
        return bestIndex
    }

    // MARK: - Wall rect (tiered)

    /// Wall rect + source tag. **Tier 4** always provides a conservative full-image crop so measure does not abort here.
    private static func findWallRect(
        detections: [FurnitureFitDetection],
        names: [Int: String],
        imageWidth: Int,
        imageHeight: Int,
    ) -> (CGRect, String) {
        let iw = CGFloat(imageWidth)
        let ih = CGFloat(imageHeight)
        let imgArea = Float(imageWidth * imageHeight)

        func logPick(_ source: String, _ d: FurnitureFitDetection) {
            let lbl = names[d.classIdx] ?? "?"
            let areaPx = d.w * d.h
            let frac = areaPx / max(imgArea, 1)
            logWallMeasurement(
                "yolo_wall_pick source=\(source) id=\(d.classIdx) label=\"\(lbl)\" area_px=\(Int(areaPx)) frac_image=\(String(format: "%.3f", frac)) conf=\(d.confidence)",
            )
        }

        // Tier 1: LVIS wall class
        if let best = detections.filter({ $0.classIdx == lvisWall }).max(by: { $0.w * $0.h < $1.w * $1.h }) {
            logPick("class_571_wall", best)
            return (detToRect(best, iw: iw, ih: ih), "class_571_wall")
        }
        logWallMeasurement("wall_pick_skip tier=1 class_571 reason=no_detections")

        // Tier 2: `\bwall\b` label (not wall lamp / wallpaper, …)
        if let best = detections.filter({ isWallLabel(names[$0.classIdx] ?? "") }).max(by: { $0.w * $0.h < $1.w * $1.h }) {
            logPick("label_wall", best)
            return (detToRect(best, iw: iw, ih: ih), "label_wall")
        }
        logWallMeasurement("wall_pick_skip tier=2 wall_word reason=no_label_matches")

        // Tier 3: room scene — crop top 10% + bottom 25% to approximate wall band (not floor + ceiling).
        if let best = detections.filter({ isRoomSceneLabel(names[$0.classIdx] ?? "") }).max(by: { $0.w * $0.h < $1.w * $1.h }) {
            logPick("label_room_scene_raw", best)
            let raw = detToRect(best, iw: iw, ih: ih)
            let cropTop = raw.height * 0.10
            let cropBottom = raw.height * 0.25
            let newH = max(CGFloat(1), raw.height - cropTop - cropBottom)
            let adjusted = CGRect(x: raw.minX, y: raw.minY + cropTop, width: raw.width, height: newH)
            let label = names[best.classIdx] ?? "?"
            logWallMeasurement(
                "room_scene_crop label=\"\(label)\" raw_h=\(Int(raw.height)) → adjusted_h=\(Int(adjusted.height)) " +
                    "(trim top 10% ceiling band + bottom 25% floor band)",
            )
            return (adjusted, "label_room_scene")
        }
        logWallMeasurement("wall_pick_skip tier=3 room_scene reason=no_label_matches")

        // Tier 4: conservative full-frame crop (no YOLO wall/room)
        let margin: CGFloat = 0.05
        let fullWall = CGRect(
            x: iw * margin,
            y: ih * 0.10,
            width: iw * (1 - 2 * margin),
            height: ih * 0.65,
        )
        logWallMeasurement("wall_pick tier=4 full_image_crop (no class_571 / wall / room label)")
        return (fullWall, "full_image")
    }

    // MARK: - Calibration scale

    private static func calibrationScale(
        rawH: Float,
        detections: [FurnitureFitDetection],
        names: [Int: String],
        mono: MonodepthBuffer?,
        wallDepth: Float,
        focalPx: Float,
        imageWidth: Int,
        imageHeight: Int,
        calPref: String,
        ceilingM: Float,
    ) -> (Float, String) {
        let tryDoor = calPref == calAuto || calPref == calDoor
        if tryDoor, let doorRect = doorBounds(detections: detections, names: names, imageWidth: imageWidth, imageHeight: imageHeight) {
            let dDepth: Float
            if let mono, let md = mono.medianAt(rect: doorRect, imageWidth: imageWidth, imageHeight: imageHeight), md > 0 {
                dDepth = md
            } else {
                dDepth = wallDepth
            }
            let doorH = (Float(doorRect.height) / focalPx) * dDepth
            if doorH > 0.1 {
                let s = standardDoorM / doorH
                logWallMeasurement("door_cal doorH_m=\(String(format: "%.3f", doorH)) scale=\(String(format: "%.4f", s))")
                return (s, "door")
            }
        }

        let c = ceilingM.clamped(to: 2.0...4.5)
        if rawH > 1e-6 {
            let s = c / rawH
            if calPref == calDoor {
                logWallMeasurement("ceiling_fallback door calibration unavailable — scale from ceiling pref")
                return (s, "ceiling_fallback")
            }
            return (s, "ceiling")
        }

        logWallMeasurement("calibration_scale fallback 1.0 (rawH near zero)")
        return (1.0, "none")
    }

    // MARK: - Door detection

    private static func doorBounds(
        detections: [FurnitureFitDetection],
        names: [Int: String],
        imageWidth: Int,
        imageHeight: Int,
    ) -> CGRect? {
        let doors = detections.filter { isDoorLabel(names[$0.classIdx] ?? "") }
        guard let best = doors.max(by: { ($0.confidence * $0.w * $0.h) < ($1.confidence * $1.w * $1.h) }) else { return nil }
        return detToRect(best, iw: CGFloat(imageWidth), ih: CGFloat(imageHeight))
    }

    // MARK: - Focal length

    private static func focalLengthPx(imageWidth: Int, exif: [String: Double]?, sensorMm: Float) -> (Float, String) {
        if let fpx = exif?["focalLengthPx"], fpx > 1 {
            return (Float(fpx), "sidecar focalLengthPx")
        }
        let s = sensorMm.clamped(to: Float(3)...Float(12))
        if let f = exif?["focalLengthMm"], f > 0.1 {
            return (Float(f / Double(s)) * Float(imageWidth), "exif focalLengthMm/\(String(format: "%.2f", s))mm")
        }
        if let f35 = exif?["focalLength35mmEquivMm"], f35 > 1 {
            return (Float(f35 / 36.0) * Float(imageWidth), "exif focalLength35mmEquivMm/36")
        }
        return (Float(4.5 / Double(s)) * Float(imageWidth), "fallback 4.5mm/\(String(format: "%.2f", s))mm sensor")
    }

    private static func loadMetricDepthBuffer(roomFolder: URL, roomURL: URL? = nil) -> (buffer: MonodepthBuffer, sourceName: String)? {
        for fileName in metricDepthCandidateFileNames(roomURL: roomURL) {
            let url = roomFolder.appendingPathComponent(fileName)
            if let buffer = MonodepthBuffer.load(url: url) {
                logMetricDepthMeasurement(
                    "METRIC DEPTH FILE FOUND PATH=\(url.path) SIZE=\(buffer.w)X\(buffer.h) CHANNELS=\(buffer.c)"
                )
                return (buffer, fileName)
            }
        }
        logMetricDepthMeasurement(
            "METRIC DEPTH FILE NOT FOUND CANDIDATES=\(metricDepthCandidateFileNames(roomURL: roomURL).joined(separator: ","))"
        )
        return nil
    }

    private static func metricDepthCandidateFileNames(roomURL: URL?) -> [String] {
        guard let roomURL else { return metricDepthCandidateFileNames }
        let stem = canonicalStem(forRoomURL: roomURL)
        return ["\(stem)_depthpro_metric_depth.bin"] + metricDepthCandidateFileNames
    }

    private static func resolveCameraIntrinsics(
        imageWidth: Int,
        imageHeight: Int,
        exif: [String: Double]?,
        sensorMmFallback: Float,
        sharpCamera: SharpCameraSidecar.Info? = nil
    ) -> CameraIntrinsics? {
        guard imageWidth > 0, imageHeight > 0 else { return nil }
        if let sharpCamera,
           sharpCamera.sourceImageWidthPx == imageWidth,
           sharpCamera.sourceImageHeightPx == imageHeight,
           sharpCamera.sourceFocalPx > 0.01 {
            let fx = sharpCamera.sourceFocalPx
            let fy = sharpCamera.sourceFocalPx
            let cx = sharpCamera.sourceCxPx
            let cy = sharpCamera.sourceCyPx
            let hFov = 2 * atan(Float(imageWidth) / max(2 * fx, 1e-6)) * 180 / .pi
            let vFov = 2 * atan(Float(imageHeight) / max(2 * fy, 1e-6)) * 180 / .pi

            logMetricDepthMeasurement(
                "METRIC INTRINSICS IMAGE=\(imageWidth)X\(imageHeight) FOCAL_PX=\(String(format: "%.2f", fx)) " +
                    "SOURCE=SHARP_GENERATION_CAMERA HFOV_DEG=\(String(format: "%.2f", hFov)) VFOV_DEG=\(String(format: "%.2f", vFov)) " +
                    "CX=\(String(format: "%.2f", cx)) CY=\(String(format: "%.2f", cy))"
            )

            return CameraIntrinsics(
                fx: fx,
                fy: fy,
                cx: cx,
                cy: cy,
                focalLengthMM: exif?["focalLengthMm"].map(Float.init),
                sensorWidthMM: exif?["sensorWidthMm"].map(Float.init),
                imageWidth: imageWidth,
                imageHeight: imageHeight,
                horizontalFOVDegrees: hFov,
                verticalFOVDegrees: vFov,
                source: "SHARP_GENERATION_CAMERA"
            )
        }
        if let focalLengthPx = exif?["focalLengthPx"].map(Float.init), focalLengthPx > 0.01 {
            let fx = focalLengthPx
            let fy = focalLengthPx
            let cx = Float(imageWidth - 1) * 0.5
            let cy = Float(imageHeight - 1) * 0.5
            let hFov = 2 * atan(Float(imageWidth) / max(2 * fx, 1e-6)) * 180 / .pi
            let vFov = 2 * atan(Float(imageHeight) / max(2 * fy, 1e-6)) * 180 / .pi

            logMetricDepthMeasurement(
                "METRIC INTRINSICS IMAGE=\(imageWidth)X\(imageHeight) FOCAL_PX=\(String(format: "%.2f", focalLengthPx)) " +
                    "SOURCE=SIDECAR_FOCAL_PX HFOV_DEG=\(String(format: "%.2f", hFov)) VFOV_DEG=\(String(format: "%.2f", vFov))"
            )

            return CameraIntrinsics(
                fx: fx,
                fy: fy,
                cx: cx,
                cy: cy,
                focalLengthMM: exif?["focalLengthMm"].map(Float.init),
                sensorWidthMM: exif?["sensorWidthMm"].map(Float.init),
                imageWidth: imageWidth,
                imageHeight: imageHeight,
                horizontalFOVDegrees: hFov,
                verticalFOVDegrees: vFov,
                source: "SIDECAR_FOCAL_PX"
            )
        }

        let focalLengthMM: Float
        let focalSource: String
        if let exactFocalLengthMM = exif?["focalLengthMm"].map(Float.init), exactFocalLengthMM > 0.01 {
            focalLengthMM = exactFocalLengthMM
            focalSource = "EXIF_FOCAL_MM"
        } else {
            focalLengthMM = 4.5
            focalSource = "FALLBACK_4_5MM"
            logMetricDepthMeasurement(
                "METRIC INTRINSICS MISSING EXIF FOCAL LENGTH AND FOCAL PX; USING FALLBACK FOCAL \(String(format: "%.3f", focalLengthMM))MM"
            )
        }

        let sensorWidthMM: Float
        let sensorSource: String
        if let derivedSensorWidth = exif?["sensorWidthMm"].map(Float.init), derivedSensorWidth > 0.01 {
            sensorWidthMM = derivedSensorWidth
            sensorSource = "EXIF_SENSOR_WIDTH"
        } else if let focal35 = exif?["focalLength35mmEquivMm"].map(Float.init), focal35 > 0.01, focalLengthMM > 0.01 {
            sensorWidthMM = 36.0 * focalLengthMM / focal35
            sensorSource = "DERIVED_FROM_35MM_EQUIV"
        } else {
            sensorWidthMM = max(sensorMmFallback, 0.01)
            sensorSource = "SETTINGS_SENSOR_WIDTH_FALLBACK"
            logMetricDepthMeasurement(
                "METRIC INTRINSICS SENSOR WIDTH UNKNOWN; USING FALLBACK SENSOR WIDTH \(String(format: "%.3f", sensorWidthMM))MM"
            )
        }

        guard sensorWidthMM > 0.01 else { return nil }
        let fx = (focalLengthMM / sensorWidthMM) * Float(imageWidth)
        let sensorHeightMM = sensorWidthMM * Float(imageHeight) / Float(imageWidth)
        let fy = (focalLengthMM / sensorHeightMM) * Float(imageHeight)
        let cx = Float(imageWidth - 1) * 0.5
        let cy = Float(imageHeight - 1) * 0.5
        let hFov = 2 * atan(Float(imageWidth) / max(2 * fx, 1e-6)) * 180 / .pi
        let vFov = 2 * atan(Float(imageHeight) / max(2 * fy, 1e-6)) * 180 / .pi

        logMetricDepthMeasurement(
            "METRIC INTRINSICS IMAGE=\(imageWidth)X\(imageHeight) FOCAL_MM=\(String(format: "%.4f", focalLengthMM)) " +
                "FOCAL_SOURCE=\(focalSource) SENSOR_WIDTH_MM=\(String(format: "%.4f", sensorWidthMM)) SENSOR_SOURCE=\(sensorSource) " +
                "FX=\(String(format: "%.2f", fx)) FY=\(String(format: "%.2f", fy)) " +
                "HFOV_DEG=\(String(format: "%.2f", hFov)) VFOV_DEG=\(String(format: "%.2f", vFov))"
        )

        return CameraIntrinsics(
            fx: fx,
            fy: fy,
            cx: cx,
            cy: cy,
            focalLengthMM: focalLengthMM,
            sensorWidthMM: sensorWidthMM,
            imageWidth: imageWidth,
            imageHeight: imageHeight,
            horizontalFOVDegrees: hFov,
            verticalFOVDegrees: vFov,
            source: sensorSource
        )
    }

    private static func measureFromMetricDepth(
        _ metricDepth: MonodepthBuffer,
        sourceName: String,
        wallRect: CGRect,
        intrinsics: CameraIntrinsics,
        plyBounds: RoomBounds? = nil
    ) -> Result? {
        let leftXPixel = Int(wallRect.minX.rounded()).clamped(to: 0...intrinsics.imageWidth - 1)
        let rightXPixel = Int((wallRect.maxX - 1).rounded()).clamped(to: 0...intrinsics.imageWidth - 1)
        let topYPixel = Int(wallRect.minY.rounded()).clamped(to: 0...intrinsics.imageHeight - 1)
        let bottomYPixel = Int((wallRect.maxY - 1).rounded()).clamped(to: 0...intrinsics.imageHeight - 1)
        let centerXPixel = Int(((wallRect.minX + wallRect.maxX) * 0.5).rounded()).clamped(to: 0...intrinsics.imageWidth - 1)
        let centerYPixel = Int(((wallRect.minY + wallRect.maxY) * 0.5).rounded()).clamped(to: 0...intrinsics.imageHeight - 1)

        guard let centerDepth = metricDepth.medianAt(
            pixelX: centerXPixel,
            pixelY: centerYPixel,
            imageWidth: intrinsics.imageWidth,
            imageHeight: intrinsics.imageHeight
        ),
        let leftDepth = metricDepth.medianAt(
            pixelX: leftXPixel,
            pixelY: centerYPixel,
            imageWidth: intrinsics.imageWidth,
            imageHeight: intrinsics.imageHeight
        ),
        let rightDepth = metricDepth.medianAt(
            pixelX: rightXPixel,
            pixelY: centerYPixel,
            imageWidth: intrinsics.imageWidth,
            imageHeight: intrinsics.imageHeight
        ),
        let topDepth = metricDepth.medianAt(
            pixelX: centerXPixel,
            pixelY: topYPixel,
            imageWidth: intrinsics.imageWidth,
            imageHeight: intrinsics.imageHeight
        ),
        let bottomDepth = metricDepth.medianAt(
            pixelX: centerXPixel,
            pixelY: bottomYPixel,
            imageWidth: intrinsics.imageWidth,
            imageHeight: intrinsics.imageHeight
        ) else {
            logMetricDepthMeasurement(
                "METRIC DEPTH SAMPLING FAILED RECT=[\(Int(wallRect.minX)),\(Int(wallRect.minY)),\(Int(wallRect.maxX)),\(Int(wallRect.maxY))]"
            )
            return nil
        }

        let leftXCenterPlane = (Float(leftXPixel) - intrinsics.cx) * centerDepth / intrinsics.fx
        let rightXCenterPlane = (Float(rightXPixel) - intrinsics.cx) * centerDepth / intrinsics.fx
        let topYCenterPlane = (Float(topYPixel) - intrinsics.cy) * centerDepth / intrinsics.fy
        let bottomYCenterPlane = (Float(bottomYPixel) - intrinsics.cy) * centerDepth / intrinsics.fy

        let centerPlaneWidthMeters = abs(rightXCenterPlane - leftXCenterPlane)
        let centerPlaneHeightMeters = abs(bottomYCenterPlane - topYCenterPlane)
        let depthMeters = centerDepth

        let samples = [leftDepth, rightDepth, topDepth, bottomDepth].filter { $0.isFinite && $0 > 0.01 }
        let maxDepthRatio = samples.map { $0 / max(centerDepth, 1e-6) }.max() ?? 1
        let minDepthRatio = samples.map { $0 / max(centerDepth, 1e-6) }.min() ?? 1
        logMetricDepthMeasurement(
            "METRIC UNIFORM WALL DEPTH SOURCE=\(sourceName) " +
                "CENTER_Z_M=\(String(format: "%.4f", centerDepth)) " +
                "MAX_RATIO=\(String(format: "%.3f", maxDepthRatio)) MIN_RATIO=\(String(format: "%.3f", minDepthRatio)) " +
                "WIDTH_CENTER_PLANE=\(String(format: "%.4f", centerPlaneWidthMeters)) " +
                "HEIGHT_CENTER_PLANE=\(String(format: "%.4f", centerPlaneHeightMeters))"
        )

        var widthMeters = centerPlaneWidthMeters
        var heightMeters = centerPlaneHeightMeters

        if let plyBounds {
            let cappedWidth = min(widthMeters, max(0.05, plyBounds.width))
            let cappedHeight = min(heightMeters, max(0.05, plyBounds.height))
            if cappedWidth != widthMeters || cappedHeight != heightMeters {
                logMetricDepthMeasurement(
                    "METRIC PLY SANITY CAP SCENE_W=\(String(format: "%.4f", plyBounds.width)) " +
                        "SCENE_H=\(String(format: "%.4f", plyBounds.height)) " +
                        "WIDTH=\(String(format: "%.4f", widthMeters))→\(String(format: "%.4f", cappedWidth)) " +
                        "HEIGHT=\(String(format: "%.4f", heightMeters))→\(String(format: "%.4f", cappedHeight))"
                )
            }
            widthMeters = cappedWidth
            heightMeters = cappedHeight
        }

        guard widthMeters.isFinite, heightMeters.isFinite, depthMeters.isFinite,
              widthMeters > 0.05, heightMeters > 0.05, depthMeters > 0.05 else {
            logMetricDepthMeasurement(
                "METRIC GEOMETRY INVALID WIDTH=\(String(format: "%.4f", widthMeters)) " +
                    "HEIGHT=\(String(format: "%.4f", heightMeters)) DEPTH=\(String(format: "%.4f", depthMeters))"
            )
            return nil
        }

        logMetricDepthMeasurement(
            "METRIC DEPTH SAMPLES SOURCE=\(sourceName) CENTER_Z_M=\(String(format: "%.4f", centerDepth)) " +
                "LEFT_Z_M=\(String(format: "%.4f", leftDepth)) RIGHT_Z_M=\(String(format: "%.4f", rightDepth)) " +
                "TOP_Z_M=\(String(format: "%.4f", topDepth)) BOTTOM_Z_M=\(String(format: "%.4f", bottomDepth))"
        )
        logMetricDepthMeasurement(
            "METRIC PROJECTION PIXELS LEFT=\(leftXPixel) RIGHT=\(rightXPixel) TOP=\(topYPixel) BOTTOM=\(bottomYPixel) " +
                "CENTER=(\(centerXPixel),\(centerYPixel))"
        )
        logMetricDepthMeasurement(
            "METRIC ROOM RESULT WIDTH_M=\(String(format: "%.4f", widthMeters)) " +
                "HEIGHT_M=\(String(format: "%.4f", heightMeters)) DEPTH_M=\(String(format: "%.4f", depthMeters)) " +
                "HFOV_DEG=\(String(format: "%.2f", intrinsics.horizontalFOVDegrees)) VFOV_DEG=\(String(format: "%.2f", intrinsics.verticalFOVDegrees)) " +
                "FX=\(String(format: "%.2f", intrinsics.fx)) FY=\(String(format: "%.2f", intrinsics.fy))"
        )

        return Result(
            widthMeters: widthMeters,
            heightMeters: heightMeters,
            depthMeters: depthMeters,
            calibrationMode: "metric_depth"
        )
    }

    // MARK: - Helpers

    private static func detToRect(_ d: FurnitureFitDetection, iw: CGFloat, ih: CGFloat) -> CGRect {
        let l = CGFloat(d.x - d.w * 0.5).clamped(to: 0...iw - 1)
        let t = CGFloat(d.y - d.h * 0.5).clamped(to: 0...ih - 1)
        let r = CGFloat(d.x + d.w * 0.5).clamped(to: 0...iw - 1)
        let b = CGFloat(d.y + d.h * 0.5).clamped(to: 0...ih - 1)
        return CGRect(x: l, y: t, width: max(1, r - l), height: max(1, b - t))
    }

    private static func sourceImagePixelSize(for image: UIImage) -> (width: Int, height: Int) {
        if let cgImage = image.cgImage {
            return (cgImage.width, cgImage.height)
        }
        let width = max(1, Int((image.size.width * image.scale).rounded()))
        let height = max(1, Int((image.size.height * image.scale).rounded()))
        return (width, height)
    }

    private static func metricDepthOnlyRect(
        imageWidth: Int,
        imageHeight: Int,
        photoOrientation: PhotoOrientation
    ) -> CGRect {
        let iw = CGFloat(max(imageWidth, 1))
        let ih = CGFloat(max(imageHeight, 1))
        let insetXFrac: CGFloat = photoOrientation == .portrait ? 0.08 : 0.05
        let insetTopFrac: CGFloat = photoOrientation == .portrait ? 0.12 : 0.10
        let insetBottomFrac: CGFloat = photoOrientation == .portrait ? 0.16 : 0.12
        let x = iw * insetXFrac
        let y = ih * insetTopFrac
        let width = iw * (1 - insetXFrac * 2)
        let height = ih * max(0.2, 1 - insetTopFrac - insetBottomFrac)
        return CGRect(x: x, y: y, width: max(1, width), height: max(1, height))
    }

    private static func loadExifJson(roomFolder: URL) -> [String: Double]? {
        loadExifJson(roomURL: nil, roomFolder: roomFolder)
    }

    private static func loadExifJson(roomURL: URL) -> [String: Double]? {
        loadExifJson(roomURL: roomURL, roomFolder: roomURL.deletingLastPathComponent())
    }

    private static func loadExifJson(roomURL: URL?, roomFolder: URL) -> [String: Double]? {
        let urls: [URL]
        if let roomURL {
            let stem = canonicalStem(forRoomURL: roomURL)
            urls = [
                roomFolder.appendingPathComponent("\(stem)_camera_exif.json"),
                roomFolder.appendingPathComponent("camera_exif.json"),
            ]
        } else {
            urls = [roomFolder.appendingPathComponent("camera_exif.json")]
        }
        guard let u = urls.first(where: { FileManager.default.fileExists(atPath: $0.path) }),
              let data = try? Data(contentsOf: u),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return nil }
        var out: [String: Double] = [:]
        if let v = numericExifValue(obj["focalLengthMm"]) { out["focalLengthMm"] = v }
        if let v = numericExifValue(obj["focalLength35mmEquivMm"]) { out["focalLength35mmEquivMm"] = v }
        if let v = numericExifValue(obj["focalLengthPx"]) { out["focalLengthPx"] = v }
        if let v = numericExifValue(obj["subjectDistanceMeters"]) { out["subjectDistanceMeters"] = v }
        if let v = numericExifValue(obj["imageWidthPx"]) { out["imageWidthPx"] = v }
        if let v = numericExifValue(obj["imageHeightPx"]) { out["imageHeightPx"] = v }
        if let focal = out["focalLengthMm"],
           let focal35 = out["focalLength35mmEquivMm"],
           focal > 0.01,
           focal35 > 0.01 {
            out["sensorWidthMm"] = 36.0 * focal / focal35
        }
        return out.isEmpty ? nil : out
    }

    private static func canonicalStem(forRoomURL roomURL: URL) -> String {
        var stem = roomURL.deletingPathExtension().lastPathComponent
        if stem.hasSuffix("_classic") {
            stem = String(stem.dropLast("_classic".count))
        }
        return stem
    }

    private static func numericExifValue(_ any: Any?) -> Double? {
        guard let any else { return nil }
        if let d = any as? Double { return d }
        if let n = any as? NSNumber { return n.doubleValue }
        if let i = any as? Int { return Double(i) }
        return nil
    }

    private static var cachedClassNames: [Int: String]?
    private static func loadClassNames() -> [Int: String] {
        if let c = cachedClassNames { return c }
        guard let url = Bundle.main.url(forResource: "classes", withExtension: "json"),
              let data = try? Data(contentsOf: url),
              let dict = try? JSONSerialization.jsonObject(with: data) as? [String: String] else {
            cachedClassNames = [:]
            return [:]
        }
        var m: [Int: String] = [:]
        for (k, v) in dict {
            if let id = Int(k) { m[id] = v }
        }
        cachedClassNames = m
        return m
    }

    // MARK: - Monodepth buffer

    private struct MonodepthBuffer {
        let w: Int
        let h: Int
        let c: Int
        let data: [Float]

        static func load(url: URL) -> MonodepthBuffer? {
            guard let raw = try? Data(contentsOf: url), raw.count >= 12 else { return nil }
            let w = readLE32(raw, 0)
            let h = readLE32(raw, 4)
            let c = readLE32(raw, 8)
            guard w > 0, h > 0, c > 0 else { return nil }
            let n = w * h * c
            guard raw.count >= 12 + n * 4 else { return nil }
            var floats = [Float](repeating: 0, count: n)
            floats.withUnsafeMutableBytes { dst in
                raw.withUnsafeBytes { srcBuf in
                    let src = srcBuf.baseAddress!.advanced(by: 12)
                    memcpy(dst.baseAddress, src, n * 4)
                }
            }
            return MonodepthBuffer(w: w, h: h, c: c, data: floats)
        }

        private static func readLE32(_ d: Data, _ o: Int) -> Int {
            d.subdata(in: o..<(o + 4)).withUnsafeBytes {
                Int(Int32(littleEndian: $0.load(as: Int32.self)))
            }
        }

        func medianAt(rect: CGRect, imageWidth: Int, imageHeight: Int) -> Float? {
            let cx = Float((rect.minX + rect.maxX) * 0.5).clamped(to: 0...Float(imageWidth - 1))
            let cy = Float((rect.minY + rect.maxY) * 0.5).clamped(to: 0...Float(imageHeight - 1))
            let mx = Int(cx / Float(imageWidth) * Float(w)).clamped(to: 0...w - 1)
            let my = Int(cy / Float(imageHeight) * Float(h)).clamped(to: 0...h - 1)
            return medianAt(mappedX: mx, mappedY: my)
        }

        func medianAt(
            pixelX: Int,
            pixelY: Int,
            imageWidth: Int,
            imageHeight: Int
        ) -> Float? {
            guard imageWidth > 0, imageHeight > 0 else { return nil }
            let mappedX = Int(Float(pixelX) / Float(imageWidth) * Float(w)).clamped(to: 0...w - 1)
            let mappedY = Int(Float(pixelY) / Float(imageHeight) * Float(h)).clamped(to: 0...h - 1)
            return medianAt(mappedX: mappedX, mappedY: mappedY)
        }

        private func medianAt(mappedX mx: Int, mappedY my: Int) -> Float? {
            var samples: [Float] = []
            samples.reserveCapacity(49)
            for dy in -3...3 {
                for dx in -3...3 {
                    let x = (mx + dx).clamped(to: 0...w - 1)
                    let y = (my + dy).clamped(to: 0...h - 1)
                    let idx = (y * w + x) * c
                    if idx < data.count {
                        let v = data[idx]
                        if v.isFinite, v > 0 { samples.append(v) }
                    }
                }
            }
            guard !samples.isEmpty else { return nil }
            samples.sort()
            return samples[samples.count / 2]
        }

        func sampleBilinear(
            pixelX: Float,
            pixelY: Float,
            imageWidth: Int,
            imageHeight: Int
        ) -> Float? {
            guard imageWidth > 0, imageHeight > 0, c > 0 else { return nil }
            let mappedX = (pixelX / Float(imageWidth) * Float(w - 1)).clamped(to: 0...Float(w - 1))
            let mappedY = (pixelY / Float(imageHeight) * Float(h - 1)).clamped(to: 0...Float(h - 1))
            let x0 = Int(floor(mappedX)).clamped(to: 0...w - 1)
            let y0 = Int(floor(mappedY)).clamped(to: 0...h - 1)
            let x1 = min(x0 + 1, w - 1)
            let y1 = min(y0 + 1, h - 1)
            let tx = mappedX - Float(x0)
            let ty = mappedY - Float(y0)

            guard let v00 = sampleAt(x: x0, y: y0),
                  let v10 = sampleAt(x: x1, y: y0),
                  let v01 = sampleAt(x: x0, y: y1),
                  let v11 = sampleAt(x: x1, y: y1) else {
                return nil
            }

            let top = v00 * (1 - tx) + v10 * tx
            let bottom = v01 * (1 - tx) + v11 * tx
            let value = top * (1 - ty) + bottom * ty
            return value.isFinite && value > 0 ? value : nil
        }

        private func sampleAt(x: Int, y: Int) -> Float? {
            let idx = (y * w + x) * c
            guard idx < data.count else { return nil }
            let value = data[idx]
            return value.isFinite && value > 0 ? value : nil
        }
    }

    private static func percentileBounds(
        values: [Float],
        lowerPercentile: Float,
        upperPercentile: Float
    ) -> (min: Float, max: Float)? {
        guard !values.isEmpty else { return nil }
        var sorted = values
        sorted.sort()
        let lastIndex = sorted.count - 1
        let lowerIndex = Int((Float(lastIndex) * lowerPercentile).rounded(.down)).clamped(to: 0...lastIndex)
        let upperIndex = Int((Float(lastIndex) * upperPercentile).rounded(.down)).clamped(to: 0...lastIndex)
        guard lowerIndex <= upperIndex else { return nil }
        return (sorted[lowerIndex], sorted[upperIndex])
    }

    private static func uniformSubsample(
        points: [SIMD3<Float>],
        targetCount: Int
    ) -> [SIMD3<Float>] {
        guard targetCount > 0, points.count > targetCount else { return points }
        let stride = max(1, points.count / targetCount)
        var sampled: [SIMD3<Float>] = []
        sampled.reserveCapacity(targetCount)
        var index = 0
        while index < points.count, sampled.count < targetCount {
            sampled.append(points[index])
            index += stride
        }
        return sampled
    }

    private static func opposingWallPairCount(walls: [DetectedPlane]) -> Int {
        var count = 0
        for i in 0..<walls.count {
            for j in (i + 1)..<walls.count where simd_dot(walls[i].normal, walls[j].normal) < -0.45 {
                count += 1
            }
        }
        return count
    }

    private final class NullMetricDepthQuery: SplatDepthQueryable {
        let points: [SIMD3<Float>]

        init(points: [SIMD3<Float>]) {
            self.points = points
        }

        func depthAt(screenPoint _: CGPoint) -> Float? { nil }
        func unproject(screenPoint _: CGPoint, depth _: Float) -> SIMD3<Float>? { nil }
        var viewportSize: CGSize { .zero }
        var trimmedSplatPositions: [SIMD3<Float>]? { points }
    }

    private final class NullMetricColorReader: SplatColorReadable {
        var supportsColorReadback: Bool { false }
        func colorAt(screenPoint _: CGPoint) -> SIMD3<Float>? { nil }
    }

    private struct BinaryClassicPly {
        let data: Data
        let vertexCount: Int
        let dataOffset: Int
        let vertexStride: Int

        static func load(url: URL) -> BinaryClassicPly? {
            guard let data = try? Data(contentsOf: url),
                  let headerRange = data.range(of: Data("end_header\n".utf8)) ??
                    data.range(of: Data("end_header\r\n".utf8)) else {
                return nil
            }
            let headerData = Data(data[..<headerRange.upperBound])
            guard let header = String(data: headerData, encoding: .utf8) else {
                return nil
            }
            guard header.contains("format binary_little_endian 1.0") else {
                return nil
            }
            let vertexCount = header
                .split(separator: "\n")
                .compactMap { line -> Int? in
                    let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
                    guard trimmed.hasPrefix("element vertex ") else { return nil }
                    return Int(trimmed.replacingOccurrences(of: "element vertex ", with: ""))
                }
                .first ?? 0
            let vertexStride = 47
            guard vertexCount > 0, data.count >= headerRange.upperBound + vertexCount * vertexStride else {
                return nil
            }
            return BinaryClassicPly(
                data: data,
                vertexCount: vertexCount,
                dataOffset: headerRange.upperBound,
                vertexStride: vertexStride
            )
        }

        func position(at index: Int) -> SIMD3<Float> {
            let base = dataOffset + index * vertexStride
            return SIMD3<Float>(
                readFloatLE(at: base),
                readFloatLE(at: base + 4),
                readFloatLE(at: base + 8)
            )
        }

        private func readFloatLE(at offset: Int) -> Float {
            var value: UInt32 = 0
            data.withUnsafeBytes { rawBuffer in
                guard let baseAddress = rawBuffer.baseAddress else { return }
                memcpy(&value, baseAddress.advanced(by: offset), MemoryLayout<UInt32>.size)
            }
            return Float(bitPattern: UInt32(littleEndian: value))
        }
    }
}

private extension Float {
    func clamped(to r: ClosedRange<Float>) -> Float {
        Swift.min(r.upperBound, Swift.max(r.lowerBound, self))
    }
}

private extension Int {
    func clamped(to r: ClosedRange<Int>) -> Int {
        Swift.min(r.upperBound, Swift.max(r.lowerBound, self))
    }
}

private extension CGFloat {
    func clamped(to r: ClosedRange<CGFloat>) -> CGFloat {
        Swift.min(r.upperBound, Swift.max(r.lowerBound, self))
    }
}
