import CoreGraphics
import CoreML
import Foundation
import UIKit

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

    private static let calAuto = "auto"
    private static let calDoor = "door"
    private static let calCeiling = "ceiling"

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

        // --- Depth: monodepth → EXIF subject distance → assumed Z ---

        let monoURL = roomFolder.appendingPathComponent("sharp_monodepth.bin")
        let mono = MonodepthBuffer.load(url: monoURL)
        if mono != nil {
            logWallMeasurement("monodepth_file=true path=\(monoURL.path)")
        } else {
            logWallMeasurement("monodepth_file=false path=\(monoURL.path)")
        }

        let wallDepth: Float
        let depthSource: String
        if let mono, let md = mono.medianAt(rect: wallRect, imageWidth: iw, imageHeight: ih), md > 0 {
            wallDepth = md
            depthSource = "monodepth"
        } else if let sd = exif.flatMap({ numericExifValue($0["subjectDistanceMeters"]) }), sd > 0.1, sd < 30 {
            wallDepth = Float(sd)
            depthSource = "exif_subject_distance"
        } else {
            wallDepth = max(0.5, min(20, assumedZ))
            depthSource = "assumed_z"
        }
        logWallMeasurement("wall_depth value=\(wallDepth) source=\(depthSource)")

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
        let rawW = (wallPxW / focalPx) * wallDepth
        let rawH = (wallPxH / focalPx) * wallDepth

        let (scale, calMode) = calibrationScale(
            rawH: rawH,
            detections: mapped,
            names: names,
            mono: mono,
            wallDepth: wallDepth,
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

        // PLY depth (Z span) is a direct 3D measurement — better than camera-to-wall distance for room depth.
        let roomDepth: Float
        let depthMode: String
        if let bounds = plyBounds, bounds.depth > 0.1 {
            roomDepth = bounds.depth
            depthMode = "\(depthSource)+ply_z"
        } else {
            roomDepth = wallDepth
            depthMode = depthSource
        }

        logWallMeasurement(
            "measure_final mode=\(calMode) width_m=\(String(format: "%.3f", wm)) height_m=\(String(format: "%.3f", hm)) " +
                "depth=\(String(format: "%.3f", roomDepth))(\(depthMode)) wall_source=\(wallSource) " +
                "focal_px=\(String(format: "%.1f", focalPx))(\(focalHow)) " +
                "raw_geom_m w=\(String(format: "%.3f", rawW)) h=\(String(format: "%.3f", rawH)) scale=\(String(format: "%.4f", scale))",
        )
        return Result(widthMeters: wm, heightMeters: hm, depthMeters: roomDepth, calibrationMode: calMode)
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
        let s = sensorMm.clamped(to: Float(3)...Float(12))
        if let f = exif?["focalLengthMm"], f > 0.1 {
            return (Float(f / Double(s)) * Float(imageWidth), "exif focalLengthMm/\(String(format: "%.2f", s))mm")
        }
        if let f35 = exif?["focalLength35mmEquivMm"], f35 > 1 {
            return (Float(f35 / 36.0) * Float(imageWidth), "exif focalLength35mmEquivMm/36")
        }
        return (Float(4.5 / Double(s)) * Float(imageWidth), "fallback 4.5mm/\(String(format: "%.2f", s))mm sensor")
    }

    // MARK: - Helpers

    private static func detToRect(_ d: FurnitureFitDetection, iw: CGFloat, ih: CGFloat) -> CGRect {
        let l = CGFloat(d.x - d.w * 0.5).clamped(to: 0...iw - 1)
        let t = CGFloat(d.y - d.h * 0.5).clamped(to: 0...ih - 1)
        let r = CGFloat(d.x + d.w * 0.5).clamped(to: 0...iw - 1)
        let b = CGFloat(d.y + d.h * 0.5).clamped(to: 0...ih - 1)
        return CGRect(x: l, y: t, width: max(1, r - l), height: max(1, b - t))
    }

    private static func loadExifJson(roomFolder: URL) -> [String: Double]? {
        let u = roomFolder.appendingPathComponent("camera_exif.json")
        guard let data = try? Data(contentsOf: u),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return nil }
        var out: [String: Double] = [:]
        if let v = numericExifValue(obj["focalLengthMm"]) { out["focalLengthMm"] = v }
        if let v = numericExifValue(obj["focalLength35mmEquivMm"]) { out["focalLength35mmEquivMm"] = v }
        if let v = numericExifValue(obj["subjectDistanceMeters"]) { out["subjectDistanceMeters"] = v }
        return out.isEmpty ? nil : out
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
