import CoreGraphics
import CoreML
import Foundation
import UIKit

/// YOLO wall segmentation + SHARP monodepth (on disk) + camera EXIF → front wall width/height in meters.
/// UserDefaults keys match Android `WallMeasurementEstimator` prefs.
enum WallMeasurementEstimator {

    struct Result {
        let widthMeters: Float
        let heightMeters: Float
        let calibrationMode: String
    }

    private static let lvisWall = 571
    private static let standardDoorM: Float = 2.03

    /// YOLOE `classes.json` is not byte-identical to LVIS IDs; e.g. **537 is "bowl"** in our bundle, not door.
    /// Door calibration must use **label text**, not a hard-coded LVIS door id, or a bowl can drive scale and inflate width/height.
    private static let doorLabelNegativeSubstrings: [String] = [
        "doormat", "doorbell", "doorstop", "car door",
    ]

    private static func isDoorCalibrationLabel(_ raw: String) -> Bool {
        let lower = raw.lowercased()
        guard lower.contains("door") else { return false }
        if doorLabelNegativeSubstrings.contains(where: { lower.contains($0) }) { return false }
        return true
    }

    /// YOLO-E outputs ~33k anchor rows. Floor **0** keeps almost all anchors; tier 1 then picks the **largest-area class 571**,
    /// which can be a **0.01-confidence** box and **beat** a better tier-2 “room” box — wrong wall height. **0.05** drops
    /// anchor noise only (not Furniture Fit’s 0.25); semantic wall/room/heuristic tiers unchanged.
    private static let yoloWallMeasureClassScoreFloor: Float = 0.05

    private static let keyEnabled = "wall_measurement_yolo_on_save"
    private static let keyCalibration = "wall_measurement_calibration"
    private static let keyCeiling = "wall_measurement_assumed_ceiling_m"
    private static let keySensorMm = "wall_measurement_sensor_width_mm"
    private static let keyAssumedZ = "wall_measurement_assumed_depth_m"

    private static let calAuto = "auto"
    private static let calDoor = "door"
    private static let calCeiling = "ceiling"

    /// SwiftUI `@AppStorage` and other writers may store numbers as Double, Float, Int, or NSNumber.
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

    /// Default **on** when the key has never been set (`bool(forKey:)` is false for missing keys).
    private static var isWallMeasurementEnabled: Bool {
        if UserDefaults.standard.object(forKey: keyEnabled) == nil {
            return true
        }
        return UserDefaults.standard.bool(forKey: keyEnabled)
    }

    /// Runs YOLO + depth; call from a background task. Pass `thumbnail` from `thumbnail.png` in `roomFolder`.
    static func measure(roomFolder: URL, thumbnail: UIImage, model: MLModel?) async -> Result? {
        let ceilingPref = floatPref(keyCeiling, default: 2.5)
        let sensorMm = floatPref(keySensorMm, default: 6.4)
        let assumedZ = floatPref(keyAssumedZ, default: 2.5)
        let calStr = UserDefaults.standard.string(forKey: keyCalibration) ?? calAuto
        logWallMeasurement(
            "measure begin folder=\(roomFolder.path) pref_enabled=\(isWallMeasurementEnabled) calibration=\(calStr) " +
                "ceiling_m=\(ceilingPref) assumed_z_m=\(assumedZ) sensor_mm=\(sensorMm)",
        )

        guard isWallMeasurementEnabled else {
            logWallMeasurement("measure skip: \(keyEnabled) is false")
            return nil
        }
        guard let model else {
            logWallMeasurement("measure abort: no ML model")
            return nil
        }

        let detsRaw: [FurnitureFitDetection]
        let letterboxMap: YoloEImageInference.LetterboxMapping
        do {
            // Same YOLO-E + parser as Furniture Fit; **never** pass `blacklist.json` classes here (wall/door would be removed).
            (detsRaw, letterboxMap) = try YoloEImageInference.runDetections(
                image: thumbnail,
                model: model,
                classBlacklist: [],
                confidenceThreshold: yoloWallMeasureClassScoreFloor,
            )
        } catch {
            logWallMeasurement("measure abort: yolo \(error.localizedDescription)")
            return nil
        }
        let iw = letterboxMap.sourceWidth
        let ih = letterboxMap.sourceHeight
        logWallMeasurement(
            "reference_image size=\(iw)x\(ih) yolo_side=\(letterboxMap.modelSide) class_score_floor=\(yoloWallMeasureClassScoreFloor) (FurnitureFit pipeline, classBlacklist=empty)",
        )
        guard iw > 8, ih > 8 else {
            logWallMeasurement("measure abort: image too small")
            return nil
        }

        let dets = detsRaw.map { YoloEImageInference.mapDetectionToSourceImage(det: $0, mapping: letterboxMap) }
        let sample = dets.prefix(5).map { "c\($0.classIdx):\($0.confidence)" }.joined(separator: ",")
        logWallMeasurement("yolo_detections count=\(dets.count) sample=[\(sample)]")

        guard let (wallRect, wallSource) = wallBoundsWithSource(detections: dets, imageWidth: iw, imageHeight: ih) else {
            logWallMeasurement("measure abort: no wall-like detection")
            return nil
        }
        logWallMeasurement(
            "wall_rect source=\(wallSource) rect=[\(Int(wallRect.minX)),\(Int(wallRect.minY)),\(Int(wallRect.maxX)),\(Int(wallRect.maxY))] px",
        )

        let exif = loadExifJson(roomFolder: roomFolder)
        let (focalPx, focalReason) = focalLengthPixelsWithReason(imageWidth: iw, exif: exif)
        logWallMeasurement("focal_px=\(focalPx) focal_reason=\(focalReason) exif_json=\(exif != nil)")

        let monoURL = roomFolder.appendingPathComponent("sharp_monodepth.bin")
        let mono = MonodepthBuffer.load(url: monoURL)
        if let mono {
            logWallMeasurement("monodepth_file=true path=\(monoURL.path) buffer=\(mono.w)x\(mono.h)x\(mono.c)")
        } else {
            logWallMeasurement("monodepth_file=\(FileManager.default.fileExists(atPath: monoURL.path)) path=\(monoURL.path) buffer=nil")
        }

        let wallPixelW = max(1, Float(wallRect.width))
        let wallPixelH = max(1, Float(wallRect.height))

        let wallDepthSharp: Float
        if let mono {
            wallDepthSharp = mono.medianAt(rect: wallRect, imageWidth: iw, imageHeight: ih) ?? .nan
        } else {
            wallDepthSharp = .nan
        }
        logWallMeasurement("wall_depth_sharp median=\(wallDepthSharp) (invalid → assumed Z path)")

        if !focalPx.isFinite || focalPx <= 1e-3 {
            logWallMeasurement("measure abort: invalid focal_px")
            return nil
        }

        if !wallDepthSharp.isFinite || wallDepthSharp <= 0 {
            let z = assumedZ
            let zc = max(0.5, min(20, z))
            let wm = (wallPixelW / focalPx) * zc
            var hm = (wallPixelH / focalPx) * zc
            // Without monodepth: short bbox height (thin strip) makes hm meaningless — use ceiling pref.
            // Tall bboxes (high fracH) can yield hm slightly above ceiling pref; that is still plausible — do not force ceiling.
            let ceiling = max(2.0, min(4.5, ceilingPref))
            let fracH = Float(wallRect.height) / Float(ih)
            let thinStrip = fracH < 0.28
            let hmTooShort = hm < 1.8
            var heightRule: String
            if thinStrip || hmTooShort {
                logWallMeasurement(
                    "assumed_depth_z: bbox height unreliable (thin strip or short hm: hm=\(hm)m fracH=\(String(format: "%.3f", fracH))) — using ceiling \(ceiling)m for height",
                )
                hm = ceiling
                heightRule = "hm=ceiling_pref_m (thin_strip fracH<0.28 OR hm<1.8m before clamp)"
            } else {
                hm = min(max(hm, 1.8), 4.5)
                if hm > ceiling * 1.08 {
                    logWallMeasurement(
                        "assumed_depth_z: geometry hm=\(hm)m (fracH=\(String(format: "%.3f", fracH))) above ceiling pref \(ceiling)m — kept within 1.8…4.5m",
                    )
                }
                heightRule = "hm=geometry_from_pixels_clamped_1.8_4.5m (tall_bbox fracH≥0.28)"
            }
            logWallMeasurement("result assumed_depth_z wm=\(wm)m hm=\(hm)m Z=\(zc) wallSource=\(wallSource)")
            logWallMeasurement(
                "measure_final mode=assumed_depth_z width_m=\(wm) height_m=\(hm) " +
                    "wall_bbox_px_w=\(Int(wallPixelW)) wall_bbox_px_h=\(Int(wallPixelH)) image_px=\(iw)x\(ih) fracH=\(String(format: "%.3f", fracH)) " +
                    "focal_px=\(focalPx) (\(focalReason)) Z_assumed_m=\(zc) (UserDefaults \(keyAssumedZ)) " +
                    "formula wm=(wall_bbox_px_w/focal_px)*Z hm derived per heightRule " +
                    "height_rule=\(heightRule) wall_detection_source=\(wallSource) monodepth=absent",
            )
            return Result(widthMeters: wm, heightMeters: hm, calibrationMode: "assumed_depth_z")
        }

        let wSharp = (wallPixelW / focalPx) * wallDepthSharp
        let hSharp = (wallPixelH / focalPx) * wallDepthSharp
        logWallMeasurement("sharp_geom wSharp=\(wSharp) hSharp=\(hSharp) (pre-scale)")

        let calMode = UserDefaults.standard.string(forKey: keyCalibration) ?? calAuto

        var scale: Float?
        var modeStr = "ceiling"

        let wantDoor = calMode == calDoor || calMode == calAuto
        if wantDoor, let mono {
            if let doorRect = doorBounds(detections: dets, imageWidth: iw, imageHeight: ih) {
                let dDepth = mono.medianAt(rect: doorRect, imageWidth: iw, imageHeight: ih) ?? wallDepthSharp
                let doorHSharp = (Float(doorRect.height) / focalPx) * dDepth
                logWallMeasurement(
                    "door_calibration door_rect=[\(Int(doorRect.minX)),\(Int(doorRect.minY)),\(Int(doorRect.maxX)),\(Int(doorRect.maxY))] dDepth=\(dDepth) doorHSharp=\(doorHSharp)",
                )
                if doorHSharp > 1e-4 {
                    scale = standardDoorM / doorHSharp
                    modeStr = "door"
                }
            }
        }

        if scale == nil, calMode != calDoor {
            let ceiling = floatPref(keyCeiling, default: 2.5)
            let c = max(2.0, min(4.5, ceiling))
            if hSharp > 1e-6 {
                scale = c / hSharp
                modeStr = "ceiling"
            }
        }

        if scale == nil, calMode == calDoor {
            let ceiling = floatPref(keyCeiling, default: 2.5)
            let c = max(2.0, min(4.5, ceiling))
            if hSharp > 1e-6 {
                scale = c / hSharp
                modeStr = "ceiling_fallback"
            }
        }

        guard let s = scale, s.isFinite, s > 0 else {
            logWallMeasurement("measure abort: scale nil or invalid (mode would be \(calMode))")
            return nil
        }

        let widthM = wSharp * s
        let heightM = hSharp * s
        logWallMeasurement("measure ok mode=\(modeStr) scale=\(s) W×H=\(widthM)×\(heightM)m wallSource=\(wallSource)")
        let scaleWhy: String
        switch modeStr {
        case "door":
            scaleWhy = "scale=\(standardDoorM)m_std_door/doorHSharp (door bbox + depth → metric scale)"
        case "ceiling":
            scaleWhy = "scale=ceiling_pref_m/hSharp (room height from SHARP depth + ceiling pref)"
        case "ceiling_fallback":
            scaleWhy = "scale=ceiling_pref_m/hSharp (door calibration unavailable in door-only mode)"
        default:
            scaleWhy = "scale=\(s)"
        }
        logWallMeasurement(
            "measure_final mode=\(modeStr) width_m=\(widthM) height_m=\(heightM) " +
                "wall_bbox_px_w=\(Int(wallPixelW)) wall_bbox_px_h=\(Int(wallPixelH)) image_px=\(iw)x\(ih) " +
                "focal_px=\(focalPx) (\(focalReason)) wall_depth_median_m=\(wallDepthSharp) " +
                "pre_scale wSharp_m=\(wSharp) hSharp_m=\(hSharp) scale=\(s) (\(scaleWhy)) " +
                "formula width_m=wSharp*scale height_m=hSharp*scale wall_detection_source=\(wallSource) monodepth=used",
        )
        return Result(widthMeters: widthM, heightMeters: heightM, calibrationMode: modeStr)
    }

    private struct MonodepthBuffer {
        let w: Int
        let h: Int
        let c: Int
        let data: [Float]

        static func load(url: URL) -> MonodepthBuffer? {
            guard let data = try? Data(contentsOf: url), data.count >= 12 else { return nil }
            let w = readLE32(data, 0)
            let h = readLE32(data, 4)
            let c = readLE32(data, 8)
            if w <= 0 || h <= 0 || c <= 0 { return nil }
            let expected = w * h * c
            let need = 12 + expected * 4
            guard data.count >= need else { return nil }
            var floats = [Float](repeating: 0, count: expected)
            floats.withUnsafeMutableBytes { dst in
                data.withUnsafeBytes { raw in
                    let src = raw.baseAddress!.advanced(by: 12)
                    memcpy(dst.baseAddress, src, expected * 4)
                }
            }
            return MonodepthBuffer(w: w, h: h, c: c, data: floats)
        }

        private static func readLE32(_ data: Data, _ offset: Int) -> Int {
            data.subdata(in: offset..<(offset + 4)).withUnsafeBytes {
                Int(Int32(littleEndian: $0.load(as: Int32.self)))
            }
        }

        func medianAt(rect: CGRect, imageWidth: Int, imageHeight: Int) -> Float? {
            let cx = Float((rect.minX + rect.maxX) * 0.5).clamped(to: 0...Float(imageWidth - 1))
            let cy = Float((rect.minY + rect.maxY) * 0.5).clamped(to: 0...Float(imageHeight - 1))
            let mx = Int(cx / Float(imageWidth) * Float(w)).clamped(to: 0...w - 1)
            let my = Int(cy / Float(imageHeight) * Float(h)).clamped(to: 0...h - 1)
            var samples: [Float] = []
            samples.reserveCapacity(64)
            for dy in -3...3 {
                for dx in -3...3 {
                    let x = (mx + dx).clamped(to: 0...w - 1)
                    let y = (my + dy).clamped(to: 0...h - 1)
                    let idx = (y * w + x) * c
                    if idx < data.count {
                        let v = data[idx]
                        if v.isFinite && v > 0 { samples.append(v) }
                    }
                }
            }
            guard !samples.isEmpty else { return nil }
            samples.sort()
            return samples[samples.count / 2]
        }
    }

    private static func loadExifJson(roomFolder: URL) -> [String: Double]? {
        let u = roomFolder.appendingPathComponent("camera_exif.json")
        guard let data = try? Data(contentsOf: u),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return nil }
        var out: [String: Double] = [:]
        if let v = obj["focalLengthMm"] as? Double { out["focalLengthMm"] = v }
        if let v = obj["focalLength35mmEquivMm"] as? Double { out["focalLength35mmEquivMm"] = v }
        return out.isEmpty ? nil : out
    }

    /// Substrings that contain "wall" but are not building walls (do **not** use furniture blacklist.json here).
    private static let wallLabelNegativeSubstrings: [String] = [
        "wall lamp", "wallpaper", "wall clock", "wall socket", "wall outlet", "wall plug",
        "wall sconce", "wall light", "wall strip", "wall sticker", "wall decal",
    ]

    /// Labels that look like venue/interior in `classes.json` but are **objects** (not a scene wall).
    private static let venueLabelNegativeSubstrings: [String] = [
        "hospital bed", "building block", "building material", "office chair", "office desk", "office supply",
        "kitchen knife", "kitchen cabinet", "kitchen counter", "kitchen floor", "kitchen hood", "kitchen island",
        "kitchen sink", "kitchen table", "kitchen utensil", "kitchen window", "kitchenware",
        "bathroom accessory", "bathroom door", "bathroom mirror", "bathroom sink", "bathroom cabinet", "bathroom window",
        "brick building", "glass building", "church tower", "empire state building",
    ]

    /// Interior / venue / building semantics from LVIS-style `classes.json` (e.g. hotel, hospital, lobby, facade).
    private static let interiorVenueLabelPattern =
        #"(?i)\b(hotel|hospital|hallway|facade|lobby|hall|ballroom|classroom|restaurant|office|building|kitchen|bedroom|bathroom|boutique|skyscraper|apartment|penthouse|studio|warehouse|playroom|bookstore|showroom|building\s+facade|office\s+building|home\s+interior|boutique\s+hotel|hotel\s+lobby|hotel\s+room|hospital\s+room|interior\s+design|living\s+room|dining\s+room|family\s+room|guest\s+room|meeting\s+room|conference\s+hall|entrance\s+hall|elevator\s+lobby|banquet\s+hall|concert\s+hall|lecture\s+hall|kindergarden\s+classroom|office\s+window|office\s+cubicle|computer\s+room|dance\s+room|dressing\s+room|laundry\s+room|clean\s+room|auto\s+showroom|bus\s+interior|car\s+interior|home\s+office|wine\s+cellar|city\s+hall|department\s+store|coffee\s+shop|fastfood\s+restaurant|fabric\s+store|general\s+store|convenience\s+store|clothing\s+store|childs\s+room|factory\s+workshop|lecture\s+room|waiting\s+room|locker\s+room|storage\s+room|engine\s+room|greenhouse)\b"#

    /// True if class name suggests a real wall/room or scene-scale venue from `classes.json`, excluding object-level false positives.
    private static func isWallLikeLabel(_ rawName: String) -> Bool {
        let lower = rawName.lowercased()
        if wallLabelNegativeSubstrings.contains(where: { lower.contains($0) }) {
            return false
        }
        // Whole-word "wall" or "room" (e.g. "living room"), not "wallpaper" / "bedroom" alone unless it has \broom\b
        if let _ = lower.range(of: #"\bwall\b"#, options: .regularExpression) {
            return true
        }
        if let _ = lower.range(of: #"\broom\b"#, options: .regularExpression) {
            return true
        }
        if venueLabelNegativeSubstrings.contains(where: { lower.contains($0) }) {
            return false
        }
        if let _ = lower.range(of: interiorVenueLabelPattern, options: .regularExpression) {
            return true
        }
        return false
    }

    private static func focalLengthPixelsWithReason(imageWidth: Int, exif: [String: Double]?) -> (Float, String) {
        let sensorMm = floatPref(keySensorMm, default: 6.4)
        let s = max(3, min(12, sensorMm))
        if let f = exif?["focalLengthMm"], f > 0.1 {
            let px = Float((f / Double(s)) * Double(imageWidth))
            return (px, "exif focalLengthMm/\(String(format: "%.2f", s))mm sensor * imageWidth")
        }
        if let f35 = exif?["focalLength35mmEquivMm"], f35 > 1 {
            let px = Float((f35 / 36.0) * Double(imageWidth))
            return (px, "exif focalLength35mmEquivMm/36 * imageWidth")
        }
        let px = Float((4.5 / Double(s)) * Double(imageWidth))
        return (px, "fallback 4.5mm/\(String(format: "%.2f", s))mm sensor, no exif (\(keySensorMm) pref)")
    }

    private static func focalLengthPixels(imageWidth: Int, exif: [String: Double]?) -> Float {
        focalLengthPixelsWithReason(imageWidth: imageWidth, exif: exif).0
    }

    private static func wallBoundsWithSource(detections: [FurnitureFitDetection], imageWidth: Int, imageHeight: Int) -> (CGRect, String)? {
        let iw = CGFloat(imageWidth)
        let ih = CGFloat(imageHeight)
        let names = loadClassNames()
        let imgArea = Float(imageWidth * imageHeight)

        logWallMeasurement(
            "wall_pick_priority: 1) LVIS class \(lvisWall) (wall) — **preferred** 2) semantic wall/room/venue labels " +
                "3) heuristic_wide_strip 4) largest_wide_bbox — first tier with ≥1 candidate wins; within tier: largest bbox area; " +
                "yolo_anchor_class_score_floor=\(yoloWallMeasureClassScoreFloor) (not 0.25 Furniture Fit; drops raw anchor noise only)",
        )

        func logPick(_ source: String, _ d: FurnitureFitDetection) {
            let lbl = names[d.classIdx] ?? "?"
            let areaPx = d.w * d.h
            let frac = areaPx / max(imgArea, 1)
            logWallMeasurement(
                "yolo_wall_pick source=\(source) id=\(d.classIdx) label=\"\(lbl)\" area_px=\(Int(areaPx)) frac_image=\(String(format: "%.3f", frac)) conf=\(d.confidence) (furniture blacklist.json NOT used)",
            )
        }

        func logPickReason(_ source: String, _ rule: String) {
            logWallMeasurement("yolo_wall_pick_reason source=\(source) rule=\(rule)")
        }

        // 1) LVIS wall class — largest bbox among that class.
        let byWallClass = detections.filter { $0.classIdx == lvisWall }
        if byWallClass.isEmpty {
            logWallMeasurement("wall_pick_tier_skip tier=1 LVIS_class_\(lvisWall) reason=no_detections_try_next_tier")
        }
        if let best = byWallClass.max(by: { $0.w * $0.h < $1.w * $1.h }) {
            logPick("class_571", best)
            logPickReason("class_571", "tier1_LVIS_wall_class_\(lvisWall)_largest_bbox_area (wall detection preferred)")
            return (detToRect(best, iw: iw, ih: ih), "class_571")
        }

        // 2) Semantic labels from classes.json: wall/room + venue/interior (hotel, hospital, lobby, facade, …).
        let byLabel = detections.filter { isWallLikeLabel(names[$0.classIdx] ?? "") }
        if byLabel.isEmpty {
            logWallMeasurement("wall_pick_tier_skip tier=2 semantic_wall_room_venue reason=no_label_matches_try_next_tier")
        }
        if let best = byLabel.max(by: { $0.w * $0.h < $1.w * $1.h }) {
            logPick("label_wall_room_venue", best)
            logPickReason("label_wall_room_venue", "tier2_classes_json_semantic_largest_bbox_area")
            return (detToRect(best, iw: iw, ih: ih), "label_wall_room_venue")
        }

        let boxes: [YoloCalibrationBox] = detections.map {
            YoloCalibrationBox(
                label: names[$0.classIdx] ?? "obj",
                centerX: CGFloat($0.x),
                centerY: CGFloat($0.y),
                width: CGFloat($0.w),
                height: CGFloat($0.h),
                confidence: $0.confidence
            )
        }
        let frac = YoloRatioCalibration.wallHeightFractionOrFullFrame(imageSize: CGSize(width: iw, height: ih), boxes: boxes)
        if frac >= 0.99 {
            logWallMeasurement("wall_pick_abort reason=wallHeightFractionOrFullFrame=\(String(format: "%.3f", frac))>=0.99 (no distinct wall strip)")
            return nil
        }

        // 3) Wide-strip geometry heuristic — largest among boxes that look like a wall panel.
        let heurDets = detections.filter { d in
            let wf = CGFloat(d.w) / iw
            let hf = CGFloat(d.h) / ih
            let aspect = CGFloat(d.w) / max(CGFloat(d.h), 1)
            return wf >= 0.55 && hf > 0.04 && hf < 0.55 && aspect >= 1.8
        }
        if heurDets.isEmpty {
            logWallMeasurement("wall_pick_tier_skip tier=3 heuristic_wide_strip reason=no_boxes_match_wide_strip_geometry")
        }
        if let best = heurDets.max(by: { $0.w * $0.h < $1.w * $1.h }) {
            logPick("heuristic_wide_strip", best)
            logPickReason("heuristic_wide_strip", "tier3_wide_strip_w≥0.55_image h∈(0.04,0.55)_image aspect≥1.8_largest_area")
            return (detToRect(best, iw: iw, ih: ih), "heuristic_wide_strip")
        }

        // 4) Fallback: largest wide bbox (avoid tall narrow / tiny boxes). No confidence gate — geometry only.
        let wideFallback = detections.filter { d in
            let ar = d.w / max(d.h, 1e-4)
            let fracA = (d.w * d.h) / max(imgArea, 1)
            return ar >= 1.25 && fracA >= 0.02
        }
        if wideFallback.isEmpty {
            logWallMeasurement("wall_pick_tier_skip tier=4 largest_wide_bbox reason=no_box_aspect≥1.25_area≥2pct_image")
        }
        if let best = wideFallback.max(by: { $0.w * $0.h < $1.w * $1.h }) {
            logPick("largest_wide_bbox", best)
            logPickReason("largest_wide_bbox", "tier4_fallback_aspect≥1.25_frac_area≥0.02_largest_area")
            return (detToRect(best, iw: iw, ih: ih), "largest_wide_bbox")
        }

        logWallMeasurement("measure abort: wall_pick all tiers empty")
        return nil
    }

    private static func doorBounds(detections: [FurnitureFitDetection], imageWidth: Int, imageHeight: Int) -> CGRect? {
        let names = loadClassNames()
        let doors = detections.filter { isDoorCalibrationLabel(names[$0.classIdx] ?? "") }
        guard let best = doors.max(by: { ($0.confidence * $0.w * $0.h) < ($1.confidence * $1.w * $1.h) }) else { return nil }
        return detToRect(best, iw: CGFloat(imageWidth), ih: CGFloat(imageHeight))
    }

    private static func detToRect(_ d: FurnitureFitDetection, iw: CGFloat, ih: CGFloat) -> CGRect {
        let l = CGFloat(d.x - d.w * 0.5).clamped(to: 0...iw - 1)
        let t = CGFloat(d.y - d.h * 0.5).clamped(to: 0...ih - 1)
        let r = CGFloat(d.x + d.w * 0.5).clamped(to: 0...iw - 1)
        let b = CGFloat(d.y + d.h * 0.5).clamped(to: 0...ih - 1)
        return CGRect(x: l, y: t, width: max(1, r - l), height: max(1, b - t))
    }

    private static var cachedClassNames: [Int: String]?
    private static func loadClassNames() -> [Int: String] {
        if let cachedClassNames { return cachedClassNames }
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
