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
 */
object YoloRatioCalibration {

    /** COCO-style labels we treat as furniture for per-room reference ratios. */
    val FURNITURE_LABELS: Set<String> = setOf(
        "chair", "couch", "bed", "dining table", "toilet", "tv", "refrigerator",
        "sink", "oven", "microwave", "bench", "potted plant"
    )

    private val LABEL_ALIASES: Map<String, String> = mapOf(
        "sofa" to "couch",
        "wardrobe" to "refrigerator",
    )

    fun canonicalFurnitureLabel(raw: String): String {
        val key = raw.trim().lowercase()
        return LABEL_ALIASES[key] ?: key
    }

    /**
     * Wall-like box: wide strip (no dedicated COCO "wall" class).
     */
    fun wallHeightFractionOrFullFrame(
        imageWidth: Int,
        imageHeight: Int,
        boxes: List<CalibrationBox>,
    ): Float {
        if (imageHeight <= 0) return 1f
        val candidates = boxes.filter { b ->
            val wFrac = b.width / imageWidth.toFloat()
            val hFrac = b.height / imageHeight.toFloat()
            val aspect = b.width / b.height.coerceAtLeast(1e-4f)
            wFrac >= 0.55f && hFrac in 0.04f..0.55f && aspect >= 1.8f
        }
        if (candidates.isEmpty()) return 1f
        val best = candidates.maxByOrNull { it.width * it.height } ?: return 1f
        return (best.height / imageHeight.toFloat()).coerceIn(0.02f, 1f)
    }

    /**
     * Per canonical label, 75th percentile of bbox height / image height.
     */
    fun furnitureHeightFractionsByLabel(
        imageHeight: Int,
        boxes: List<CalibrationBox>,
    ): Map<String, Float> {
        if (imageHeight <= 0) return emptyMap()
        val byLabel = mutableMapOf<String, MutableList<Float>>()
        for (b in boxes) {
            val canon = canonicalFurnitureLabel(b.label)
            if (canon !in FURNITURE_LABELS) continue
            val frac = (b.height / imageHeight.toFloat()).coerceIn(0.01f, 0.95f)
            byLabel.getOrPut(canon) { mutableListOf() }.add(frac)
        }
        val out = mutableMapOf<String, Float>()
        for ((label, heights) in byLabel) {
            if (heights.isEmpty()) continue
            val sorted = heights.sorted()
            val idx = ((sorted.size - 1) * 0.75).toInt().coerceIn(0, sorted.lastIndex)
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
        val canon = canonicalFurnitureLabel(label)
        storedByClass[canon]?.let { return it.coerceIn(0.06f, 0.85f) }
        // Common synonyms
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
