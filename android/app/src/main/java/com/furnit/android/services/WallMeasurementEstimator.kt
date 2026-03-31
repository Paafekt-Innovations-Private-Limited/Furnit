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
import kotlin.math.roundToInt

/**
 * YOLO segmentation + SHARP monodepth (persisted) + camera EXIF → front-wall width/height in meters
 * plus **camera-to-wall depth** ([Result.depthMeters]) for room metadata (matches iOS `WallMeasurementEstimator`).
 *
 * Uses [YoloEImageInference] (same NCNN thresholds / model side as [FurnitureFitManager]; no furniture blacklist).
 * Tier 1: LVIS wall 571. Tier 2: `\bwall\b` (excluding wall lamp / wallpaper, …). Tier 3: `\broom\b` scene labels with a
 * **vertical crop** (trim ~10% top / ~25% bottom) so floor+ceiling scene boxes do not inflate height. Tier 4: conservative
 * full-image crop. Calibration: door (auto/door) then ceiling / `ceiling_fallback`; depth for geometry is monodepth median
 * at the wall rect or assumed Z from [PREF_ASSUMED_DEPTH_M].
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
    private const val STANDARD_DOOR_M = 2.03f

    // YOLOE classes.json is not identical to LVIS: e.g. 537 is "bowl" in assets, not door. Never match door by a fixed id.
    private val doorLabelNegatives = listOf("doormat", "doorbell", "doorstop", "car door")

    private fun isDoorCalibrationLabel(raw: String): Boolean {
        val lower = raw.lowercase()
        if (!lower.contains("door")) return false
        return doorLabelNegatives.none { lower.contains(it) }
    }

    /** ~33k NCNN anchors; 0 keeps all and tier 1 can pick a 0.01-conf “wall” over a better tier-2 label. 0.05 drops anchor noise only (not Furniture Fit 0.25). */
    private const val YOLO_WALL_MEASURE_CLASS_SCORE_FLOOR = 0.05f

    @Volatile
    private var cachedClassNames: Map<Int, String>? = null

    data class Result(
        val widthMeters: Float,
        val heightMeters: Float,
        /** Camera-to-front-wall distance (m): monodepth median at wall rect, or assumed Z. */
        val depthMeters: Float,
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

        val (wallRect, wallSource) = wallBoundsFromDetections(dets, iw, ih, classNames)
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

        val subjectDistanceM = exif?.let { jo ->
            val d = jo.optDouble("subjectDistanceMeters", Double.NaN)
            if (d.isFinite() && d > 0.1 && d < 30.0) d.toFloat() else null
        }
        val wallDepthMono = mono?.let { medianMonodepthAt(it, wallRect, iw, ih) }
        val wallDepth: Float
        val depthSource: String
        when {
            wallDepthMono != null && wallDepthMono.isFinite() && wallDepthMono > 0f -> {
                wallDepth = wallDepthMono
                depthSource = "monodepth"
            }
            subjectDistanceM != null -> {
                wallDepth = subjectDistanceM
                depthSource = "exif_subject_distance"
            }
            else -> {
                wallDepth = prefs.getFloat(PREF_ASSUMED_DEPTH_M, 2.5f).coerceIn(0.5f, 20f)
                depthSource = "assumed_z"
            }
        }
        LogUtil.i(TAG, "wall_depth value=$wallDepth source=$depthSource")

        val rawW = (wallPixelW / focalPx) * wallDepth
        val rawH = (wallPixelH / focalPx) * wallDepth
        LogUtil.i(TAG, "raw_geom_m rawW=$rawW rawH=$rawH (pre-scale)")

        val calibrationMode = prefs.getString(PREF_CALIBRATION, CAL_AUTO) ?: CAL_AUTO
        val (scale, modeStr) = calibrationScale(
            rawH = rawH,
            dets = dets,
            classNames = classNames,
            mono = mono,
            wallDepth = wallDepth,
            focalPx = focalPx,
            iw = iw,
            ih = ih,
            calibrationMode = calibrationMode,
            prefs = prefs,
        )

        if (!scale.isFinite() || scale <= 0f) {
            LogUtil.w(TAG, "measure abort: invalid scale=$scale")
            return null
        }

        var widthM = rawW * scale
        var heightM = rawH * scale
        widthM = widthM.coerceIn(1.5f, 12f)
        heightM = heightM.coerceIn(1.5f, 5f)
        LogUtil.i(TAG, "measure ok mode=$modeStr scale=$scale wm=${widthM}m hm=${heightM}m wallSource=$wallSource")
        val scaleWhy = when (modeStr) {
            "door" -> "scale=${STANDARD_DOOR_M}m_std_door/doorH_raw (door bbox + depth → metric scale)"
            "ceiling" -> "scale=ceiling_pref_m/rawH"
            "ceiling_fallback" -> "scale=ceiling_pref_m/rawH (door calibration unavailable in door-only mode)"
            "none" -> "scale=1.0 (rawH near zero)"
            else -> "scale=$scale"
        }
        LogUtil.i(
            TAG,
            "measure_final mode=$modeStr width_m=$widthM height_m=$heightM depth_meters=$wallDepth ($depthSource) " +
                "wall_bbox_px_w=$wallPixelW wall_bbox_px_h=$wallPixelH image_px=${iw}x$ih " +
                "focal_px=$focalPx ($focalReason) " +
                "pre_scale rawW_m=$rawW rawH_m=$rawH scale=$scale ($scaleWhy) " +
                "formula width_m=rawW*scale height_m=rawH*scale wall_source=$wallSource",
        )
        return Result(widthM, heightM, wallDepth, modeStr)
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

    /** Door (auto/door) then ceiling; matches iOS `calibrationScale`. */
    private fun calibrationScale(
        rawH: Float,
        dets: List<NcnnYoloe.Detection>,
        classNames: Map<Int, String>,
        mono: MonoBuffer?,
        wallDepth: Float,
        focalPx: Float,
        iw: Int,
        ih: Int,
        calibrationMode: String,
        prefs: SharedPreferences,
    ): Pair<Float, String> {
        val tryDoor = calibrationMode == CAL_AUTO || calibrationMode == CAL_DOOR
        if (tryDoor) {
            val doorRect = findDoorDetection(dets, iw, ih, classNames)
            if (doorRect != null) {
                val dDepth = if (mono != null) {
                    medianMonodepthAt(mono, doorRect, iw, ih) ?: wallDepth
                } else {
                    wallDepth
                }
                val doorH = (doorRect.height().toFloat() / focalPx) * dDepth
                if (doorH > 0.1f) {
                    val s = STANDARD_DOOR_M / doorH
                    LogUtil.i(TAG, "door_cal doorH_m=$doorH scale=$s")
                    return s to "door"
                }
            }
        }
        val ceiling = prefs.getFloat(PREF_ASSUMED_CEILING_M, 2.5f).coerceIn(2.0f, 4.5f)
        if (rawH > 1e-6f) {
            val s = ceiling / rawH
            if (calibrationMode == CAL_DOOR) {
                LogUtil.i(TAG, "ceiling_fallback door calibration unavailable — scale from ceiling pref")
                return s to "ceiling_fallback"
            }
            LogUtil.i(TAG, "ceiling_calibration ceiling_m=$ceiling scale=$s")
            return s to "ceiling"
        }
        LogUtil.w(TAG, "calibration_scale fallback 1.0 (rawH near zero)")
        return 1.0f to "none"
    }

    /** Substrings that contain "wall" but are not building walls (furniture blacklist.json is not applied here). */
    private val wallLabelNegativeSubstrings = listOf(
        "wall lamp", "wallpaper", "wall clock", "wall socket", "wall outlet", "wall plug",
        "wall sconce", "wall light", "wall strip", "wall sticker", "wall decal",
    )

    private val regexWordWall = Regex("\\bwall\\b", RegexOption.IGNORE_CASE)
    private val regexRoomWord = Regex("\\broom\\b", RegexOption.IGNORE_CASE)

    /** Object-level labels — block spurious “room” matches (same list as iOS). */
    private val venueLabelNegativeSubstrings = listOf(
        "hospital bed", "building block", "building material", "office chair", "office desk", "office supply",
        "kitchen knife", "kitchen cabinet", "kitchen counter", "kitchen floor", "kitchen hood", "kitchen island",
        "kitchen sink", "kitchen table", "kitchen utensil", "kitchen window", "kitchenware",
        "bathroom accessory", "bathroom door", "bathroom mirror", "bathroom sink", "bathroom cabinet", "bathroom window",
        "brick building", "glass building", "church tower", "empire state building",
    )

    private fun isRoomSceneLabel(rawName: String): Boolean {
        val lower = rawName.lowercase()
        if (venueLabelNegativeSubstrings.any { lower.contains(it) }) return false
        return regexRoomWord.containsMatchIn(lower)
    }

    private fun isWallWordLabel(rawName: String): Boolean {
        val lower = rawName.lowercase()
        if (wallLabelNegativeSubstrings.any { lower.contains(it) }) return false
        return regexWordWall.containsMatchIn(lower)
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

    /** Wall rect and source tag; tier 4 always returns a conservative full-image crop (parity with iOS). */
    private fun wallBoundsFromDetections(
        dets: List<NcnnYoloe.Detection>,
        iw: Int,
        ih: Int,
        classNames: Map<Int, String>,
    ): Pair<Rect, String> {
        LogUtil.i(
            TAG,
            "wall_pick: 1) class_$LVIS_WALL 2) label \\bwall\\b 3) label \\broom\\b (cropped) 4) full_image_crop — largest area per tier; " +
                "yolo_anchor_class_score_floor=$YOLO_WALL_MEASURE_CLASS_SCORE_FLOOR",
        )

        val byClass = dets.filter { it.classId == LVIS_WALL }
        if (byClass.isEmpty()) {
            LogUtil.i(TAG, "wall_pick_skip tier=1 class_$LVIS_WALL reason=no_detections")
        }
        if (byClass.isNotEmpty()) {
            val best = byClass.maxByOrNull { it.width * it.height }!!
            logWallPick("class_571_wall", best, iw, ih, classNames)
            logWallPickReason("class_571_wall", "tier1_LVIS_wall_largest_area")
            return detToRect(best, iw, ih) to "class_571_wall"
        }

        val byWallWord = dets.filter { isWallWordLabel(classNameFor(it, classNames)) }
        if (byWallWord.isEmpty()) {
            LogUtil.i(TAG, "wall_pick_skip tier=2 wall_word reason=no_label_matches")
        }
        if (byWallWord.isNotEmpty()) {
            val best = byWallWord.maxByOrNull { it.width * it.height }!!
            logWallPick("label_wall", best, iw, ih, classNames)
            logWallPickReason("label_wall", "tier2_word_wall_largest_area")
            return detToRect(best, iw, ih) to "label_wall"
        }

        val byRoom = dets.filter { isRoomSceneLabel(classNameFor(it, classNames)) }
        if (byRoom.isEmpty()) {
            LogUtil.i(TAG, "wall_pick_skip tier=3 room_scene reason=no_label_matches")
        }
        if (byRoom.isNotEmpty()) {
            val best = byRoom.maxByOrNull { it.width * it.height }!!
            logWallPick("label_room_scene_raw", best, iw, ih, classNames)
            logWallPickReason("label_room_scene_raw", "tier3_word_room_largest_area_before_crop")
            val raw = detToRect(best, iw, ih)
            val rh = raw.height()
            val cropTop = (rh * 0.10).roundToInt().coerceAtLeast(0)
            val cropBottom = (rh * 0.25).roundToInt().coerceAtLeast(0)
            val newH = max(1, rh - cropTop - cropBottom)
            val adjusted = Rect(raw.left, raw.top + cropTop, raw.right, raw.top + cropTop + newH)
            val label = classNameFor(best, classNames)
            LogUtil.i(
                TAG,
                "room_scene_crop label=\"$label\" raw_h=$rh → adjusted_h=${adjusted.height()} " +
                    "(trim top 10% ceiling band + bottom 25% floor band)",
            )
            return adjusted to "label_room_scene"
        }

        val margin = 0.05
        val l = (iw * margin).roundToInt().coerceIn(0, iw - 1)
        val t = (ih * 0.10).roundToInt().coerceIn(0, ih - 1)
        val wPx = (iw * (1 - 2 * margin)).roundToInt().coerceAtLeast(1)
        val hPx = (ih * 0.65).roundToInt().coerceAtLeast(1)
        val fullWall = Rect(l, t, (l + wPx).coerceAtMost(iw), (t + hPx).coerceAtMost(ih))
        LogUtil.i(TAG, "wall_pick tier=4 full_image_crop (no class_571_wall / wall / room label)")
        return fullWall to "full_image"
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
            isDoorCalibrationLabel(classNameFor(it, classNames))
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
