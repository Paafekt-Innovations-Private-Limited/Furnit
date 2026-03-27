package com.furnit.android.utils

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Rect
import com.furnit.android.DetectionResult
import com.furnit.android.services.NcnnYoloe
import java.io.File
import kotlin.math.max

/**
 * Ratio-only calibration from YOLOe boxes (no metric units): wall / furniture height as
 * fraction of image height, plus helpers to crop live frames toward a target fraction.
 *
 * Kept in lockstep with iOS [Furnit/Services/OnDevice/YoloRatioCalibration.swift].
 */
object YoloRatioCalibration {

    /** Core furniture categories used for per-class target height fractions (matches iOS). */
    val FURNITURE_CANONICAL_LABELS: Set<String> = setOf(
        "chair", "couch", "bed", "dining table", "table", "wardrobe", "desk", "bench", "ottoman",
    )

    @Deprecated("Use FURNITURE_CANONICAL_LABELS", ReplaceWith("FURNITURE_CANONICAL_LABELS"))
    val FURNITURE_LABELS: Set<String> = FURNITURE_CANONICAL_LABELS

    /**
     * Map raw COCO / UI labels to canonical keys stored in metadata (matches iOS `canonicalLabel`).
     */
    fun canonicalLabel(raw: String): String {
        val trimmed = raw.trim().lowercase()
        if (trimmed.isEmpty()) return "unknown"
        if (trimmed.contains("sofa") || trimmed == "couch") return "couch"
        if (trimmed.contains("chair")) return "chair"
        if (trimmed.contains("bed")) return "bed"
        if (trimmed.contains("dining") && trimmed.contains("table")) return "dining table"
        if (trimmed.contains("wardrobe") || trimmed.contains("closet")) return "wardrobe"
        if (trimmed.contains("desk")) return "desk"
        if (trimmed.contains("ottoman")) return "ottoman"
        if (trimmed.contains("bench")) return "bench"
        if (trimmed.contains("table")) return "table"
        return trimmed
    }

    @Deprecated("Use canonicalLabel", ReplaceWith("canonicalLabel(raw)"))
    fun canonicalFurnitureLabel(raw: String): String = canonicalLabel(raw)

    /**
     * Wall-like boxes: wide vs height, moderate vertical extent, often near horizontal edges.
     * Matches iOS `wallHeightFractionOrFullFrame(imageSize:boxes:)` (not the tier-3 wall-strip heuristic).
     */
    fun wallHeightFractionOrFullFrame(
        imageWidth: Int,
        imageHeight: Int,
        boxes: List<CalibrationBox>,
    ): Float {
        val iw = max(imageWidth, 1).toFloat()
        val ih = max(imageHeight, 1).toFloat()

        var best: CalibrationBox? = null
        var bestArea = 0f

        for (box in boxes) {
            val labelCanon = canonicalLabel(box.label)
            if (labelCanon in FURNITURE_CANONICAL_LABELS) continue

            val w = box.width
            val h = box.height
            if (w <= 1f || h <= 1f) continue

            val widthFrac = w / iw
            val heightFrac = h / ih
            val aspect = w / max(h, 1f)

            val nearEdge =
                (box.centerX - w * 0.5f) / iw < 0.08f ||
                    (box.centerX + w * 0.5f) / iw > 0.92f
            val wideEnough = widthFrac > 0.35f
            val moderateHeight = heightFrac > 0.08f && heightFrac < 0.85f
            val flatEnough = aspect > 2.0f

            if (wideEnough && moderateHeight && flatEnough && (nearEdge || widthFrac > 0.55f)) {
                val area = w * h
                if (area > bestArea) {
                    bestArea = area
                    best = box
                }
            }
        }

        best?.let { wallBox ->
            val frac = (wallBox.height / ih).coerceIn(0.02f, 1f)
            return frac.coerceAtMost(1f)
        }
        return 1f
    }

    /**
     * Per canonical furniture label, 75th percentile of bbox height / image height (matches iOS).
     */
    fun furnitureHeightFractionsByLabel(
        imageHeight: Int,
        boxes: List<CalibrationBox>,
    ): Map<String, Float> {
        if (imageHeight <= 0) return emptyMap()
        val imageHeightSafe = max(imageHeight, 1).toFloat()
        val byLabel = mutableMapOf<String, MutableList<Float>>()
        for (box in boxes) {
            val canon = canonicalLabel(box.label)
            if (canon !in FURNITURE_CANONICAL_LABELS) continue
            val frac = (box.height / imageHeightSafe).coerceIn(0.01f, 0.95f)
            byLabel.getOrPut(canon) { mutableListOf() }.add(frac)
        }
        val out = mutableMapOf<String, Float>()
        for ((label, values) in byLabel) {
            if (values.isEmpty()) continue
            val sorted = values.sorted()
            // Same index as Swift: min(count-1, Int(Double(count)*0.75 - 0.001))
            val idx = kotlin.math.min(
                sorted.size - 1,
                (sorted.size * 0.75 - 0.001).toInt().coerceAtLeast(0),
            )
            out[label] = sorted[idx]
        }
        return out
    }

    /**
     * Pick a reference image for offline ratio capture (matches list thumbnails when possible).
     */
    fun pickReferenceImageFile(roomFolder: File): File? {
        if (!roomFolder.isDirectory) return null
        val candidates = listOf(
            File(roomFolder, "thumbnail.png"),
            File(roomFolder, "front_wall.png"),
            File(roomFolder, "room.jpeg"),
            File(roomFolder, "room.jpg"),
            File(roomFolder, "photo.jpg"),
        )
        return candidates.firstOrNull { it.isFile && it.length() > 0L }
    }

    fun decodeBitmapReference(file: File, maxSide: Int = 1280): Bitmap? {
        val options = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeFile(file.absolutePath, options)
        var sample = 1
        val maxDim = max(options.outWidth, options.outHeight)
        while (maxDim / sample > maxSide) sample *= 2
        val decodeOptions = BitmapFactory.Options().apply { inSampleSize = sample.coerceAtLeast(1) }
        return BitmapFactory.decodeFile(file.absolutePath, decodeOptions)?.let { bmp ->
            if (bmp.config != Bitmap.Config.ARGB_8888) bmp.copy(Bitmap.Config.ARGB_8888, false) else bmp
        }
    }

    /**
     * Target height fraction for a detection label; falls back to [defaultFrac].
     */
    fun targetHeightFracForLabel(
        label: String,
        storedByClass: Map<String, Float>,
        defaultFrac: Float,
    ): Float {
        val canon = canonicalLabel(label)
        storedByClass[canon]?.let { return it.coerceIn(0.06f, 0.85f) }
        if (canon == "couch") storedByClass["sofa"]?.let { return it.coerceIn(0.06f, 0.85f) }
        return defaultFrac.coerceIn(0.06f, 0.85f)
    }

    /**
     * Crop rect so that [bboxHeightPx] / cropHeight ≈ [targetHeightFrac] (clamped), centered on (cx, cy).
     */
    fun cropRectForTargetFurnitureFrac(
        frameWidth: Int,
        frameHeight: Int,
        centerXPx: Float,
        centerYPx: Float,
        bboxWidthPx: Float,
        bboxHeightPx: Float,
        targetHeightFrac: Float,
    ): Rect {
        val rTarget = targetHeightFrac.coerceIn(0.08f, 0.55f)
        val minCropH = (bboxHeightPx * 1.06f).toInt().coerceAtLeast(1)
        // Desired crop height so bbox occupies ~rTarget of vertical FOV in the crop
        var cropH = (bboxHeightPx / rTarget).toInt().coerceIn(minCropH, frameHeight)
        var cropW = max(bboxWidthPx * 2.1f, cropH * 0.88f).toInt()
            .coerceIn((bboxWidthPx * 1.25f).toInt().coerceAtLeast(1), frameWidth)

        var left = (centerXPx - cropW / 2f).toInt()
        var top = (centerYPx - cropH / 2f).toInt()
        left = left.coerceIn(0, max(0, frameWidth - cropW))
        top = top.coerceIn(0, max(0, frameHeight - cropH))
        cropW = cropW.coerceAtMost(frameWidth - left)
        cropH = cropH.coerceAtMost(frameHeight - top)
        return Rect(left, top, left + cropW, top + cropH)
    }

    /**
     * Map pixel-space center bbox to a square [modelSize] space (matches overlay / ONNX convention).
     */
    fun detectionResultInModelSquare(
        centerX: Float,
        centerY: Float,
        widthPx: Float,
        heightPx: Float,
        frameWidth: Int,
        frameHeight: Int,
        modelSize: Int,
        confidence: Float,
        label: String,
    ): DetectionResult {
        val basis = max(frameWidth, frameHeight).toFloat().coerceAtLeast(1f)
        val s = modelSize / basis
        return DetectionResult(
            x = centerX * s,
            y = centerY * s,
            w = widthPx * s,
            h = heightPx * s,
            confidence = confidence,
            label = label,
        )
    }

    /**
     * Paste a smaller RGBA bitmap into a transparent full-frame buffer at [left],[top].
     */
    fun pasteMaskOntoFullFrame(
        fullWidth: Int,
        fullHeight: Int,
        croppedMask: Bitmap,
        left: Int,
        top: Int,
    ): Bitmap {
        val out = Bitmap.createBitmap(fullWidth, fullHeight, Bitmap.Config.ARGB_8888)
        val canvas = android.graphics.Canvas(out)
        canvas.drawBitmap(croppedMask, left.toFloat(), top.toFloat(), null)
        return out
    }

    data class CalibrationBox(
        val label: String,
        val centerX: Float,
        val centerY: Float,
        val width: Float,
        val height: Float,
        val confidence: Float,
    )

    fun fromNcnnDetections(detections: List<NcnnYoloe.Detection>): List<CalibrationBox> =
        detections.map {
            CalibrationBox(
                label = it.label,
                centerX = it.x,
                centerY = it.y,
                width = it.width,
                height = it.height,
                confidence = it.confidence,
            )
        }
}
