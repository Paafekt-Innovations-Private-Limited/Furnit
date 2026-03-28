package com.furnit.android.services

import android.content.Context
import android.content.SharedPreferences
import android.graphics.Rect
import com.furnit.android.utils.LogUtil
import com.furnit.android.utils.YoloRatioCalibration
import org.json.JSONObject
import java.io.File
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.max

/**
 * YOLO segmentation + SHARP monodepth (persisted) + camera EXIF → front-wall width/height in meters.
 * Uses [YoloEImageInference] (same NCNN thresholds / model side as [FurnitureFitManager]; no furniture blacklist).
 * LVIS class IDs are used when the NCNN export includes them; otherwise wall-like boxes fall back to
 * [YoloRatioCalibration] heuristics (COCO 80).
 */
object WallMeasurementEstimator {

    /** Logcat filter: `adb logcat | grep WALL_MEAS` */
    private const val TAG = "WALL_MEAS"

    const val PREF_ENABLED = "wall_measurement_yolo_on_save"
    const val PREF_CALIBRATION = "wall_measurement_calibration"
    const val PREF_ASSUMED_CEILING_M = "wall_measurement_assumed_ceiling_m"
    const val PREF_SENSOR_WIDTH_MM = "wall_measurement_sensor_width_mm"
    const val PREF_ASSUMED_DEPTH_M = "wall_measurement_assumed_depth_m"
    const val PREF_SCALE_DEPTH = "wall_measurement_scale_depth"

    /** Calibration: "auto" | "door" | "ceiling" */
    const val CAL_AUTO = "auto"
    const val CAL_DOOR = "door"
    const val CAL_CEILING = "ceiling"

    private const val LVIS_WALL = 571
    private const val LVIS_DOOR = 537
    private const val STANDARD_DOOR_M = 2.03f

    /** ~33k NCNN anchors; 0 keeps all and tier 1 can pick a 0.01-conf “wall” over a better tier-2 label. 0.05 drops anchor noise only (not Furniture Fit 0.25). */
    private const val YOLO_WALL_MEASURE_CLASS_SCORE_FLOOR = 0.05f

    @Volatile
    private var cachedClassNames: Map<Int, String>? = null

    data class Result(
        val widthMeters: Float,
        val heightMeters: Float,
        val calibrationMode: String,
    )

    fun measure(context: Context, roomFolder: File, prefs: SharedPreferences): Result? {
        LogUtil.i(
            TAG,
            "measure begin folder=${roomFolder.absolutePath} pref_enabled=${prefs.getBoolean(PREF_ENABLED, true)} " +
                "calibration=${prefs.getString(PREF_CALIBRATION, CAL_AUTO)} scale_depth=${prefs.getBoolean(PREF_SCALE_DEPTH, false)} " +
                "ceiling_m=${prefs.getFloat(PREF_ASSUMED_CEILING_M, 2.5f)} assumed_z_m=${prefs.getFloat(PREF_ASSUMED_DEPTH_M, 2.5f)} sensor_mm=${prefs.getFloat(PREF_SENSOR_WIDTH_MM, 6.4f)}",
        )
        if (!prefs.getBoolean(PREF_ENABLED, true)) {
            LogUtil.i(TAG, "measure skip: pref $PREF_ENABLED is false")
            return null
        }
        val bmpFile = YoloRatioCalibration.pickReferenceImageFile(roomFolder) ?: run {
            LogUtil.w(TAG, "measure abort: no reference image (thumbnail/front_wall/room.jpeg)")
            return null
        }
        val bitmap = YoloRatioCalibration.decodeBitmapReference(bmpFile) ?: run {
            LogUtil.w(TAG, "measure abort: decode failed ${bmpFile.absolutePath}")
            return null
        }
        val iw = bitmap.width
        val ih = bitmap.height
        LogUtil.i(TAG, "reference_image=${bmpFile.name} size=${iw}x${ih} class_score_floor=$YOLO_WALL_MEASURE_CLASS_SCORE_FLOOR")
        if (iw <= 8 || ih <= 8) {
            LogUtil.w(TAG, "measure abort: image too small")
            return null
        }

        if (!NcnnYoloe.isAvailable()) {
            LogUtil.w(TAG, "measure abort: NcnnYoloe native library not loaded")
            return null
        }
        val dets = YoloEImageInference.runDetectionsUnfiltered(context.applicationContext, bitmap, YOLO_WALL_MEASURE_CLASS_SCORE_FLOOR)
        val classNames = loadClassNames(context.applicationContext)
        LogUtil.i(
            TAG,
            "yolo_detections count=${dets.size} sample=${dets.take(5).joinToString { "c${it.classId}:${classNameFor(it, classNames)}:${it.confidence}" }}",
        )

        val wallBounds = wallBoundsFromDetections(dets, iw, ih, classNames)
        if (wallBounds == null) {
            LogUtil.w(TAG, "measure abort: no wall-like detection")
            return null
        }
        val (wallRect, wallSource) = wallBounds
        LogUtil.i(TAG, "wall_rect source=$wallSource rect=[${wallRect.left},${wallRect.top},${wallRect.right},${wallRect.bottom}] px")

        val exif = loadCameraExif(roomFolder)
        val (focalPx, focalReason) = focalLengthPixelsWithReason(iw, exif, prefs)
        LogUtil.i(TAG, "focal_px=$focalPx focal_reason=$focalReason exif_json=${exif != null} sensor_mm=${prefs.getFloat(PREF_SENSOR_WIDTH_MM, 6.4f)}")
        if (!focalPx.isFinite() || focalPx <= 1e-3f) {
            LogUtil.w(TAG, "measure abort: invalid focal_px")
            return null
        }

        val monoFile = File(roomFolder, "sharp_monodepth.bin")
        val mono = loadMonodepthBin(monoFile)
        LogUtil.i(TAG, "monodepth_file=${monoFile.exists()} path=${monoFile.absolutePath} buffer=${mono != null} ${mono?.let { "${it.w}x${it.h}x${it.c}" } ?: ""}")
        val wallPixelW = max(1f, wallRect.width().toFloat())
        val wallPixelH = max(1f, wallRect.height().toFloat())

        val wallDepthSharp: Float = if (mono != null) {
            medianMonodepthAt(mono, wallRect, iw, ih) ?: Float.NaN
        } else {
            Float.NaN
        }
        LogUtil.i(TAG, "wall_depth_sharp median=$wallDepthSharp (invalid → assumed Z path)")

        if (!wallDepthSharp.isFinite() || wallDepthSharp <= 0f) {
            val z = prefs.getFloat(PREF_ASSUMED_DEPTH_M, 2.5f).coerceIn(0.5f, 20f)
            val wm = (wallPixelW / focalPx) * z
            var hm = (wallPixelH / focalPx) * z
            val ceiling = prefs.getFloat(PREF_ASSUMED_CEILING_M, 2.5f).coerceIn(2.0f, 4.5f)
            val fracH = wallPixelH / ih.toFloat()
            val thinStrip = fracH < 0.28f
            val hmTooShort = hm < 1.8f
            val heightRule: String = if (thinStrip || hmTooShort) {
                LogUtil.i(
                    TAG,
                    "assumed_depth_z: bbox height unreliable (thin strip or short hm: hm=${hm}m fracH=$fracH) — using ceiling_m=$ceiling for height",
                )
                hm = ceiling
                "hm=ceiling_pref_m (thin_strip fracH<0.28 OR hm<1.8m)"
            } else {
                hm = hm.coerceIn(1.8f, 4.5f)
                if (hm > ceiling * 1.08f) {
                    LogUtil.i(
                        TAG,
                        "assumed_depth_z: geometry hm=${hm}m (fracH=$fracH) above ceiling pref $ceiling — kept within 1.8…4.5m",
                    )
                }
                "hm=geometry_from_pixels_clamped_1.8_4.5m (tall_bbox fracH>=0.28)"
            }
            LogUtil.i(TAG, "result assumed_depth_z wm=${wm}m hm=${hm}m Z=$z wSharpN/A")
            LogUtil.i(
                TAG,
                "measure_final mode=assumed_depth_z width_m=$wm height_m=$hm " +
                    "wall_bbox_px_w=$wallPixelW wall_bbox_px_h=$wallPixelH image_px=${iw}x$ih fracH=${"%.3f".format(fracH)} " +
                    "focal_px=$focalPx ($focalReason) Z_assumed_m=$z (pref $PREF_ASSUMED_DEPTH_M) " +
                    "formula wm=(wall_bbox_px_w/focal_px)*Z hm per height_rule height_rule=$heightRule " +
                "wall_detection_source=$wallSource monodepth=absent",
            )
            return Result(wm, hm, "assumed_depth_z")
        }

        val wSharp = (wallPixelW / focalPx) * wallDepthSharp
        val hSharp = (wallPixelH / focalPx) * wallDepthSharp
        LogUtil.i(TAG, "sharp_geom wSharp=$wSharp hSharp=$hSharp (pre-scale)")

        val calibrationMode = prefs.getString(PREF_CALIBRATION, CAL_AUTO) ?: CAL_AUTO

        var scale: Float? = null
        var modeStr = "ceiling"

        val wantDoor = calibrationMode == CAL_DOOR || calibrationMode == CAL_AUTO
        if (wantDoor && mono != null) {
            val doorRect = findDoorDetection(dets, iw, ih, classNames)
            if (doorRect != null) {
                val dDepth = medianMonodepthAt(mono, doorRect, iw, ih) ?: wallDepthSharp
                val doorHSharp = (doorRect.height().toFloat() / focalPx) * dDepth
                LogUtil.i(TAG, "door_calibration door_rect=[${doorRect.left},${doorRect.top},${doorRect.right},${doorRect.bottom}] dDepth=$dDepth doorHSharp=$doorHSharp")
                if (doorHSharp > 1e-4f) {
                    scale = STANDARD_DOOR_M / doorHSharp
                    modeStr = "door"
                    LogUtil.i(TAG, "door_calibration scale=$scale (std_door_m=$STANDARD_DOOR_M)")
                }
            } else {
                LogUtil.i(TAG, "door_calibration skipped: no door detection")
            }
        }

        if (scale == null && calibrationMode != CAL_DOOR) {
            val ceiling = prefs.getFloat(PREF_ASSUMED_CEILING_M, 2.5f).coerceIn(2.0f, 4.5f)
            if (hSharp > 1e-6f) {
                scale = ceiling / hSharp
                modeStr = "ceiling"
                LogUtil.i(TAG, "ceiling_calibration ceiling_m=$ceiling scale=$scale")
            }
        }

        if (scale == null && calibrationMode == CAL_DOOR) {
            val ceiling = prefs.getFloat(PREF_ASSUMED_CEILING_M, 2.5f).coerceIn(2.0f, 4.5f)
            if (hSharp > 1e-6f) {
                scale = ceiling / hSharp
                modeStr = "ceiling_fallback"
                LogUtil.i(TAG, "ceiling_fallback door-only mode failed ceiling_m=$ceiling scale=$scale")
            }
        }

        val s = scale
        if (s == null || !s.isFinite() || s <= 0f) {
            LogUtil.w(TAG, "measure abort: could not calibrate scale (scale=$scale hSharp=$hSharp)")
            return null
        }

        val widthM = wSharp * s
        val heightM = hSharp * s
        LogUtil.i(TAG, "measure ok mode=$modeStr scale=$s wm=${widthM}m hm=${heightM}m wallSource=$wallSource")
        val scaleWhy = when (modeStr) {
            "door" -> "scale=${STANDARD_DOOR_M}m_std_door/doorHSharp (door bbox + depth → metric scale)"
            "ceiling" -> "scale=ceiling_pref_m/hSharp (room height from SHARP depth + ceiling pref)"
            "ceiling_fallback" -> "scale=ceiling_pref_m/hSharp (door calibration unavailable in door-only mode)"
            else -> "scale=$s"
        }
        LogUtil.i(
            TAG,
            "measure_final mode=$modeStr width_m=$widthM height_m=$heightM " +
                "wall_bbox_px_w=$wallPixelW wall_bbox_px_h=$wallPixelH image_px=${iw}x$ih " +
                "focal_px=$focalPx ($focalReason) wall_depth_median_m=$wallDepthSharp " +
                "pre_scale wSharp_m=$wSharp hSharp_m=$hSharp scale=$s ($scaleWhy) " +
                "formula width_m=wSharp*scale height_m=hSharp*scale wall_detection_source=$wallSource monodepth=used",
        )
        return Result(widthM, heightM, modeStr)
    }

    private data class MonoBuffer(val w: Int, val h: Int, val c: Int, val data: FloatArray)

    private fun loadMonodepthBin(f: File): MonoBuffer? {
        if (!f.isFile || f.length() < 12) return null
        return try {
            FileInputStream(f).use { ins ->
                val hdr = ByteArray(12)
                if (ins.read(hdr) != 12) return null
                val bb = ByteBuffer.wrap(hdr).order(ByteOrder.LITTLE_ENDIAN)
                val w = bb.int
                val h = bb.int
                val c = bb.int
                if (w <= 0 || h <= 0 || c <= 0) return null
                val n = w * h * c
                val fb = ByteArray(n * 4)
                if (ins.read(fb) != n * 4) return null
                val data = FloatArray(n)
                val fbb = ByteBuffer.wrap(fb).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
                var off = 0
                while (off < n && fbb.hasRemaining()) {
                    data[off++] = fbb.get()
                }
                LogUtil.i(TAG, "loadMonodepthBin ok ${f.name} ${w}x${h}x$c floats=${data.size}")
                MonoBuffer(w, h, c, data)
            }
        } catch (e: Exception) {
            LogUtil.w(TAG, "loadMonodepthBin fail ${f.absolutePath}: ${e.message}")
            null
        }
    }

    private fun medianMonodepthAt(mono: MonoBuffer, rect: Rect, imageW: Int, imageH: Int): Float? {
        val cx = ((rect.left + rect.right) / 2f).coerceIn(0f, imageW - 1f)
        val cy = ((rect.top + rect.bottom) / 2f).coerceIn(0f, imageH - 1f)
        val mx = (cx / imageW * mono.w).toInt().coerceIn(0, mono.w - 1)
        val my = (cy / imageH * mono.h).toInt().coerceIn(0, mono.h - 1)
        val samples = ArrayList<Float>(64)
        for (dy in -3..3) {
            for (dx in -3..3) {
                val x = (mx + dx).coerceIn(0, mono.w - 1)
                val y = (my + dy).coerceIn(0, mono.h - 1)
                val idx = (y * mono.w + x) * mono.c
                if (idx < mono.data.size) {
                    val v = mono.data[idx]
                    if (v.isFinite() && v > 0f) samples.add(v)
                }
            }
        }
        if (samples.isEmpty()) {
            LogUtil.w(TAG, "medianMonodepthAt: no valid samples at mx=$mx my=$my")
            return null
        }
        samples.sort()
        val med = samples[samples.size / 2]
        LogUtil.d(TAG, "medianMonodepthAt samples=${samples.size} median=$med")
        return med
    }

    /** Substrings that contain "wall" but are not building walls (furniture blacklist.json is not applied here). */
    private val wallLabelNegativeSubstrings = listOf(
        "wall lamp", "wallpaper", "wall clock", "wall socket", "wall outlet", "wall plug",
        "wall sconce", "wall light", "wall strip", "wall sticker", "wall decal",
    )

    private val regexWordWall = Regex("\\bwall\\b", RegexOption.IGNORE_CASE)
    private val regexWordRoom = Regex("\\broom\\b", RegexOption.IGNORE_CASE)

    /** Object-level labels in classes.json — block before venue regex (same list as iOS). */
    private val venueLabelNegativeSubstrings = listOf(
        "hospital bed", "building block", "building material", "office chair", "office desk", "office supply",
        "kitchen knife", "kitchen cabinet", "kitchen counter", "kitchen floor", "kitchen hood", "kitchen island",
        "kitchen sink", "kitchen table", "kitchen utensil", "kitchen window", "kitchenware",
        "bathroom accessory", "bathroom door", "bathroom mirror", "bathroom sink", "bathroom cabinet", "bathroom window",
        "brick building", "glass building", "church tower", "empire state building",
    )

    /** Interior / venue tokens from LVIS-style classes.json (hotel, hospital, lobby, facade, …). */
    private val interiorVenueRegex = Regex(
        """(?i)\b(hotel|hospital|hallway|facade|lobby|hall|ballroom|classroom|restaurant|office|building|kitchen|bedroom|bathroom|boutique|skyscraper|apartment|penthouse|studio|warehouse|playroom|bookstore|showroom|building\s+facade|office\s+building|home\s+interior|boutique\s+hotel|hotel\s+lobby|hotel\s+room|hospital\s+room|interior\s+design|living\s+room|dining\s+room|family\s+room|guest\s+room|meeting\s+room|conference\s+hall|entrance\s+hall|elevator\s+lobby|banquet\s+hall|concert\s+hall|lecture\s+hall|kindergarden\s+classroom|office\s+window|office\s+cubicle|computer\s+room|dance\s+room|dressing\s+room|laundry\s+room|clean\s+room|auto\s+showroom|bus\s+interior|car\s+interior|home\s+office|wine\s+cellar|city\s+hall|department\s+store|coffee\s+shop|fastfood\s+restaurant|fabric\s+store|general\s+store|convenience\s+store|clothing\s+store|childs\s+room|factory\s+workshop|lecture\s+room|waiting\s+room|locker\s+room|storage\s+room|engine\s+room|greenhouse)\b""",
    )

    private fun isWallLikeLabel(rawName: String): Boolean {
        val lower = rawName.lowercase()
        if (wallLabelNegativeSubstrings.any { lower.contains(it) }) return false
        if (regexWordWall.containsMatchIn(lower) || regexWordRoom.containsMatchIn(lower)) return true
        if (venueLabelNegativeSubstrings.any { lower.contains(it) }) return false
        return interiorVenueRegex.containsMatchIn(lower)
    }

    private fun logWallPick(source: String, d: NcnnYoloe.Detection, iw: Int, ih: Int, classNames: Map<Int, String>) {
        val imgArea = max(1, iw * ih).toFloat()
        val areaPx = d.width * d.height
        val frac = areaPx / imgArea
        LogUtil.i(
            TAG,
            "yolo_wall_pick source=$source id=${d.classId} label=\"${classNameFor(d, classNames)}\" area_px=${areaPx.toInt()} " +
                "frac_image=${"%.3f".format(frac)} conf=${d.confidence} (furniture blacklist.json NOT used)",
        )
    }

    private fun logWallPickReason(source: String, rule: String) {
        LogUtil.i(TAG, "yolo_wall_pick_reason source=$source rule=$rule")
    }

    /** Returns wall rect and source tag for debugging. */
    private fun wallBoundsFromDetections(
        dets: List<NcnnYoloe.Detection>,
        iw: Int,
        ih: Int,
        classNames: Map<Int, String>,
    ): Pair<Rect, String>? {
        val imgArea = max(1, iw * ih).toFloat()

        LogUtil.i(
            TAG,
                "wall_pick_priority: 1) LVIS class $LVIS_WALL (wall) — preferred 2) semantic wall/room/venue " +
                "3) heuristic_wide_strip 4) largest_wide_bbox — first tier with ≥1 candidate wins; within tier: largest bbox area; " +
                "yolo_anchor_class_score_floor=$YOLO_WALL_MEASURE_CLASS_SCORE_FLOOR (not 0.25 Furniture Fit; drops raw anchor noise only)",
        )

        // 1) LVIS wall class — largest bbox among that class.
        val byClass = dets.filter { it.classId == LVIS_WALL }
        if (byClass.isEmpty()) {
            LogUtil.i(TAG, "wall_pick_tier_skip tier=1 LVIS_class_$LVIS_WALL reason=no_detections_try_next_tier")
        }
        if (byClass.isNotEmpty()) {
            val best = byClass.maxByOrNull { it.width * it.height } ?: return null
            logWallPick("class_$LVIS_WALL", best, iw, ih, classNames)
            logWallPickReason(
                "class_$LVIS_WALL",
                "tier1_LVIS_wall_class_${LVIS_WALL}_largest_bbox_area (wall detection preferred)",
            )
            return detToRect(best, iw, ih) to "class_$LVIS_WALL"
        }

        // 2) classes.json semantics: wall/room + venue/interior (hotel, hospital, lobby, facade, …).
        val byLabel = dets.filter { isWallLikeLabel(classNameFor(it, classNames)) }
        if (byLabel.isEmpty()) {
            LogUtil.i(TAG, "wall_pick_tier_skip tier=2 semantic_wall_room_venue reason=no_label_matches_try_next_tier")
        }
        if (byLabel.isNotEmpty()) {
            val best = byLabel.maxByOrNull { it.width * it.height } ?: return null
            logWallPick("label_wall_room_venue", best, iw, ih, classNames)
            logWallPickReason("label_wall_room_venue", "tier2_classes_json_semantic_largest_bbox_area")
            return detToRect(best, iw, ih) to "label_wall_room_venue"
        }

        val boxes = dets.map { det ->
            YoloRatioCalibration.CalibrationBox(
                label = classNameFor(det, classNames),
                centerX = det.x,
                centerY = det.y,
                width = det.width,
                height = det.height,
                confidence = det.confidence,
            )
        }
        val fracWall = YoloRatioCalibration.wallHeightFractionOrFullFrame(iw, ih, boxes)
        if (fracWall >= 0.99f) {
            LogUtil.w(
                TAG,
                "wall_pick_abort reason=wallHeightFractionOrFullFrame=${"%.3f".format(fracWall)}>=0.99 (no distinct wall strip)",
            )
            return null
        }

        // 3) Wide-strip geometry — largest among detections that look like a wall panel.
        val heurDets = dets.filter { d ->
            val wf = d.width / iw.toFloat()
            val hf = d.height / ih.toFloat()
            val aspect = d.width / d.height.coerceAtLeast(1e-4f)
            wf >= 0.55f && hf in 0.04f..0.55f && aspect >= 1.8f
        }
        if (heurDets.isEmpty()) {
            LogUtil.i(TAG, "wall_pick_tier_skip tier=3 heuristic_wide_strip reason=no_boxes_match_wide_strip_geometry")
        }
        if (heurDets.isNotEmpty()) {
            val best = heurDets.maxByOrNull { it.width * it.height } ?: return null
            logWallPick("heuristic_wide_strip", best, iw, ih, classNames)
            logWallPickReason(
                "heuristic_wide_strip",
                "tier3_wide_strip_w>=0.55_image h in (0.04,0.55)_image aspect>=1.8 largest_area",
            )
            return detToRect(best, iw, ih) to "heuristic_wide_strip"
        }

        // 4) Fallback: largest plausible wide bbox (not raw global max area — avoids tiny/tall boxes).
        val wideFallback = dets.filter { d ->
            val ar = d.width / d.height.coerceAtLeast(1e-4f)
            val fracA = (d.width * d.height) / imgArea
            ar >= 1.25f && fracA >= 0.02f
        }
        if (wideFallback.isEmpty()) {
            LogUtil.i(TAG, "wall_pick_tier_skip tier=4 largest_wide_bbox reason=no_box_aspect>=1.25_area>=2pct_image")
        }
        if (wideFallback.isNotEmpty()) {
            val best = wideFallback.maxByOrNull { it.width * it.height } ?: return null
            logWallPick("largest_wide_bbox", best, iw, ih, classNames)
            logWallPickReason(
                "largest_wide_bbox",
                "tier4_fallback_aspect>=1.25_frac_area>=0.02_largest_area",
            )
            return detToRect(best, iw, ih) to "largest_wide_bbox"
        }

        LogUtil.w(TAG, "measure abort: wall_pick all tiers empty")
        return null
    }

    private fun detToRect(d: NcnnYoloe.Detection, iw: Int, ih: Int): Rect {
        val maxX = (iw - 1).coerceAtLeast(0)
        val maxY = (ih - 1).coerceAtLeast(0)
        val l = d.left.toInt().coerceIn(0, maxX)
        val t = d.top.toInt().coerceIn(0, maxY)
        val r = d.right.toInt().coerceIn(l + 1, iw.coerceAtLeast(l + 1))
        val b = d.bottom.toInt().coerceIn(t + 1, ih.coerceAtLeast(t + 1))
        return Rect(l, t, r, b)
    }

    private fun findDoorDetection(
        dets: List<NcnnYoloe.Detection>,
        iw: Int,
        ih: Int,
        classNames: Map<Int, String>,
    ): Rect? {
        val door = dets.filter {
            it.classId == LVIS_DOOR || classNameFor(it, classNames).contains("door", ignoreCase = true)
        }
            .maxByOrNull { it.confidence * it.width * it.height }
        return door?.let { detToRect(it, iw, ih) }
    }

    private fun classNameFor(detection: NcnnYoloe.Detection, classNames: Map<Int, String>): String {
        return classNames[detection.classId]?.takeIf { it.isNotBlank() }
            ?: detection.label.takeIf { it.isNotBlank() && it != "object" }
            ?: "object"
    }

    private fun loadClassNames(context: Context): Map<Int, String> {
        cachedClassNames?.let { return it }
        val loaded = try {
            context.assets.open("classes.json").bufferedReader().use { reader ->
                val json = JSONObject(reader.readText())
                buildMap {
                    json.keys().forEach { key ->
                        val id = key.toIntOrNull() ?: return@forEach
                        val label = json.optString(key).trim()
                        if (label.isNotEmpty()) {
                            put(id, label)
                        }
                    }
                }
            }
        } catch (e: Exception) {
            LogUtil.w(TAG, "loadClassNames failed: ${e.message}")
            emptyMap()
        }
        cachedClassNames = loaded
        return loaded
    }

    private fun loadCameraExif(folder: File): JSONObject? {
        val f = File(folder, "camera_exif.json")
        if (!f.isFile) return null
        return try {
            JSONObject(f.readText())
        } catch (_: Exception) {
            null
        }
    }

    private fun focalLengthPixelsWithReason(imageWidthPx: Int, exif: JSONObject?, prefs: SharedPreferences): Pair<Float, String> {
        val sensorMm = prefs.getFloat(PREF_SENSOR_WIDTH_MM, 6.4f).coerceIn(3f, 12f)
        if (exif != null) {
            val focalMm = exif.optDouble("focalLengthMm", Double.NaN).toFloat()
            if (focalMm.isFinite() && focalMm > 0.1f) {
                val px = (focalMm / sensorMm) * imageWidthPx
                return px to "exif focalLengthMm/${"%.2f".format(sensorMm)}mm sensor * imageWidth"
            }
            val fl35 = exif.optDouble("focalLength35mmEquivMm", Double.NaN).toFloat()
            if (fl35.isFinite() && fl35 > 1f) {
                val px = (fl35 / 36f) * imageWidthPx
                return px to "exif focalLength35mmEquivMm/36 * imageWidth"
            }
        }
        val px = (4.5f / sensorMm) * imageWidthPx
        return px to "fallback 4.5mm/${"%.2f".format(sensorMm)}mm sensor, no exif ($PREF_SENSOR_WIDTH_MM pref)"
    }

}
