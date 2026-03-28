package com.furnit.android.ar

import android.graphics.ImageFormat
import android.graphics.Matrix
import android.hardware.HardwareBuffer
import android.media.Image
import com.google.ar.core.CameraIntrinsics
import com.google.ar.core.Coordinates2d
import com.google.ar.core.Frame
import com.google.ar.core.Plane
import com.google.ar.core.TrackingState
import com.furnit.android.utils.LogUtil
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Metric helpers for AR-assisted FurnitureFit overlay scale (Android / ARCore), mirroring iOS [FurnitureFitARSupport].
 */
object FurnitureFitArMetrics {

    data class DistanceEstimate(
        val meters: Float,
        val source: String
    )

    data class DistanceEstimateDebug(
        val estimate: DistanceEstimate?,
        val diagnostic: String
    )

    private fun supportsDepthImageFormat(format: Int): Boolean {
        return format == ImageFormat.DEPTH16 || format == HardwareBuffer.D_16
    }

    private fun depthMillimetersFromSample(sample: Int, format: Int): Int {
        return if (format == HardwareBuffer.D_16) sample and 0xFFFF else sample and 0x1FFF
    }

    private fun depthConfidenceFromSample(sample: Int, format: Int): Float {
        return if (format == HardwareBuffer.D_16) {
            if ((sample and 0xFFFF) > 0) 1f else 0f
        } else {
            ((sample ushr 13) and 0x7) / 7f
        }
    }

    /** First horizontal plane hit from a screen pixel (view coordinates, matches [Frame.hitTest]). */
    fun horizontalPlaneHitDistanceMeters(frame: Frame, xPx: Float, yPx: Float): Float? {
        if (frame.camera.trackingState != TrackingState.TRACKING) return null
        for (hit in frame.hitTest(xPx, yPx)) {
            val t = hit.trackable
            if (t is Plane && t.trackingState == TrackingState.TRACKING && t.isPoseInPolygon(hit.hitPose)) {
                if (t.type == Plane.Type.HORIZONTAL_UPWARD_FACING || t.type == Plane.Type.HORIZONTAL_DOWNWARD_FACING) {
                    return hit.distance
                }
            }
        }
        return null
    }

    /** Any tracked plane hit (horizontal or vertical) from a screen pixel. */
    private fun anyPlaneHitDistanceMeters(frame: Frame, xPx: Float, yPx: Float): Pair<Float, String>? {
        if (frame.camera.trackingState != TrackingState.TRACKING) return null
        for (hit in frame.hitTest(xPx, yPx)) {
            val t = hit.trackable
            if (t is Plane && t.trackingState == TrackingState.TRACKING) {
                val label = when (t.type) {
                    Plane.Type.HORIZONTAL_UPWARD_FACING -> "plane_horiz_up"
                    Plane.Type.HORIZONTAL_DOWNWARD_FACING -> "plane_horiz_down"
                    Plane.Type.VERTICAL -> "plane_vertical"
                    else -> "plane_other"
                }
                return hit.distance to label
            }
        }
        return null
    }

    /**
     * Depth-first metric distance for a furniture bbox center in raw camera-image pixels.
     * Uses a small median sample around the center to resist holes/noise in the depth image.
     * Falls back to horizontal plane hit distance when depth is not available.
     */
    fun metricDistanceEstimate(
        frame: Frame,
        imageX: Float,
        imageY: Float,
        bboxHeightPixels: Float,
        rawImageWidth: Int,
        rawImageHeight: Int,
    ): DistanceEstimate? = metricDistanceEstimateDebug(
        frame = frame,
        imageX = imageX,
        imageY = imageY,
        bboxHeightPixels = bboxHeightPixels,
        rawImageWidth = rawImageWidth,
        rawImageHeight = rawImageHeight,
    ).estimate

    fun metricDistanceEstimateDebug(
        frame: Frame,
        imageX: Float,
        imageY: Float,
        bboxHeightPixels: Float,
        rawImageWidth: Int,
        rawImageHeight: Int,
    ): DistanceEstimateDebug {
        val depthDebug = depthDistanceMetersDebug(
            frame = frame,
            imageX = imageX,
            imageY = imageY,
            bboxHeightPixels = bboxHeightPixels,
            rawImageWidth = rawImageWidth,
            rawImageHeight = rawImageHeight,
        )
        depthDebug.meters?.let {
            return DistanceEstimateDebug(
                estimate = DistanceEstimate(it, "depth16"),
                diagnostic = depthDebug.diagnostic,
            )
        }

        val viewOut = FloatArray(2)
        if (!imagePixelsToViewPixels(frame, imageX, imageY, viewOut)) {
            return DistanceEstimateDebug(null, "${depthDebug.diagnostic}+view_transform_failed")
        }
        if (frame.camera.trackingState != TrackingState.TRACKING) {
            return DistanceEstimateDebug(null, "${depthDebug.diagnostic}+tracking_not_tracking")
        }

        val horizPlane = horizontalPlaneHitDistanceMeters(frame, viewOut[0], viewOut[1])
        if (horizPlane != null) {
            return DistanceEstimateDebug(
                estimate = DistanceEstimate(horizPlane, "plane"),
                diagnostic = "${depthDebug.diagnostic}+plane_ok",
            )
        }

        val anyPlane = anyPlaneHitDistanceMeters(frame, viewOut[0], viewOut[1])
        if (anyPlane != null) {
            return DistanceEstimateDebug(
                estimate = DistanceEstimate(anyPlane.first, anyPlane.second),
                diagnostic = "${depthDebug.diagnostic}+${anyPlane.second}",
            )
        }

        return DistanceEstimateDebug(null, "${depthDebug.diagnostic}+plane_miss")
    }

    fun captureSparseMetricAnchors(
        frame: Frame,
        orientedImageWidth: Int,
        orientedImageHeight: Int,
        rawImageWidth: Int,
        rawImageHeight: Int,
        orientedToRawInverse: Matrix?,
        gridSize: Int = 7,
        marginFraction: Float = 0.05f,
        minimumConfidence: Float = 0.15f,
    ): List<MetricAnchor> {
        val depthImage = try {
            frame.acquireDepthImage16Bits()
        } catch (_: Exception) {
            null
        } ?: return emptyList()

        return try {
            val strict = sampleSparseMetricAnchorsFromDepthImage(
                depthImage = depthImage,
                orientedImageWidth = orientedImageWidth,
                orientedImageHeight = orientedImageHeight,
                rawImageWidth = rawImageWidth,
                rawImageHeight = rawImageHeight,
                orientedToRawInverse = orientedToRawInverse,
                gridSize = gridSize,
                marginFraction = marginFraction,
                minimumConfidence = minimumConfidence,
                rejectDepthDiscontinuity = true,
                minDepthMillimeters = 300,
                maxDepthMillimeters = 8000,
            )
            if (strict.size >= 5) {
                strict
            } else {
                sampleSparseMetricAnchorsFromDepthImage(
                    depthImage = depthImage,
                    orientedImageWidth = orientedImageWidth,
                    orientedImageHeight = orientedImageHeight,
                    rawImageWidth = rawImageWidth,
                    rawImageHeight = rawImageHeight,
                    orientedToRawInverse = orientedToRawInverse,
                    gridSize = max(gridSize, 9),
                    marginFraction = marginFraction * 0.5f,
                    minimumConfidence = 0f,
                    rejectDepthDiscontinuity = false,
                    minDepthMillimeters = 200,
                    maxDepthMillimeters = 10000,
                )
            }
        } finally {
            depthImage.close()
        }
    }

    private data class DepthDistanceDebug(
        val meters: Float?,
        val diagnostic: String,
    )

    private fun depthDistanceMeters(
        frame: Frame,
        imageX: Float,
        imageY: Float,
        bboxHeightPixels: Float,
        rawImageWidth: Int,
        rawImageHeight: Int,
    ): Float? = depthDistanceMetersDebug(
        frame = frame,
        imageX = imageX,
        imageY = imageY,
        bboxHeightPixels = bboxHeightPixels,
        rawImageWidth = rawImageWidth,
        rawImageHeight = rawImageHeight,
    ).meters

    private fun depthDistanceMetersDebug(
        frame: Frame,
        imageX: Float,
        imageY: Float,
        bboxHeightPixels: Float,
        rawImageWidth: Int,
        rawImageHeight: Int,
    ): DepthDistanceDebug {
        val depthImage = try {
            frame.acquireDepthImage16Bits()
        } catch (_: Exception) {
            null
        } ?: return DepthDistanceDebug(null, "depth_image_unavailable")

        return try {
            sampleDepthMetersFromImageDebug(
                depthImage = depthImage,
                imageX = imageX,
                imageY = imageY,
                bboxHeightPixels = bboxHeightPixels,
                rawImageWidth = rawImageWidth,
                rawImageHeight = rawImageHeight,
            )
        } finally {
            depthImage.close()
        }
    }

    private fun sampleDepthMetersFromImage(
        depthImage: Image,
        imageX: Float,
        imageY: Float,
        bboxHeightPixels: Float,
        rawImageWidth: Int,
        rawImageHeight: Int,
    ): Float? = sampleDepthMetersFromImageDebug(
        depthImage = depthImage,
        imageX = imageX,
        imageY = imageY,
        bboxHeightPixels = bboxHeightPixels,
        rawImageWidth = rawImageWidth,
        rawImageHeight = rawImageHeight,
    ).meters

    /**
     * Sample depth across a grid inside the bbox region, not just 5 points near the center.
     * ARCore DEPTH16 is sparse (typically 160×120 or 240×180); a small object like a cup may
     * have zero valid depth pixels at its center. Sampling a 5×5 grid across the inner 70% of
     * the bbox dramatically increases the chance of hitting a valid depth pixel.
     *
     * Two passes:
     *   1. Inner bbox (70% inset) with 5×5 grid, biased toward the lower-central region
     *      (furniture sits on surfaces → bottom half is more likely to have valid depth).
     *   2. If pass 1 fails: wider bbox (full + 20% margin) with 7×7 grid, relaxed thresholds.
     */
    private fun sampleDepthMetersFromImageDebug(
        depthImage: Image,
        imageX: Float,
        imageY: Float,
        bboxHeightPixels: Float,
        rawImageWidth: Int,
        rawImageHeight: Int,
    ): DepthDistanceDebug {
        if (!supportsDepthImageFormat(depthImage.format)) {
            return DepthDistanceDebug(null, "depth_format_${depthImage.format}")
        }
        val depthFormat = depthImage.format
        val depthW = depthImage.width
        val depthH = depthImage.height
        if (depthW <= 0 || depthH <= 0 || rawImageWidth <= 0 || rawImageHeight <= 0) {
            return DepthDistanceDebug(null, "depth_invalid_dims")
        }

        val imagePlane = depthImage.planes.firstOrNull() ?: return DepthDistanceDebug(null, "depth_plane_missing")
        val buf = imagePlane.buffer.duplicate()
        val rowStride = imagePlane.rowStride
        val pixelStride = imagePlane.pixelStride
        if (pixelStride < 2 || rowStride <= 0) {
            return DepthDistanceDebug(null, "depth_stride_invalid")
        }

        val bboxW = bboxHeightPixels * 0.85f
        val bboxHalfW = bboxW * 0.5f
        val bboxHalfH = bboxHeightPixels * 0.5f

        val pass1 = sampleBboxGridDepth(
            buf, rowStride, pixelStride, depthFormat, depthW, depthH,
            rawImageWidth, rawImageHeight,
            imageX - bboxHalfW * 0.7f, imageY - bboxHalfH * 0.5f,
            imageX + bboxHalfW * 0.7f, imageY + bboxHalfH * 0.7f,
            gridCols = 5, gridRows = 5,
            minMm = 100, maxMm = 50000,
        )
        if (pass1.validDepths.isNotEmpty()) {
            val formatLabel = if (depthFormat == HardwareBuffer.D_16) "d16" else "depth16"
            logDepthSampleDiag("pass1", pass1, imageX, imageY, bboxHeightPixels)
            val median = medianWithIqrFilter(pass1.validDepths)
            if (median != null) {
                return DepthDistanceDebug(median, "depth_ok_${formatLabel}_grid")
            }
        }

        val pass2 = sampleBboxGridDepth(
            buf, rowStride, pixelStride, depthFormat, depthW, depthH,
            rawImageWidth, rawImageHeight,
            imageX - bboxHalfW * 1.2f, imageY - bboxHalfH * 1.2f,
            imageX + bboxHalfW * 1.2f, imageY + bboxHalfH * 1.2f,
            gridCols = 7, gridRows = 7,
            minMm = 80, maxMm = 60000,
        )
        if (pass2.validDepths.isNotEmpty()) {
            val formatLabel = if (depthFormat == HardwareBuffer.D_16) "d16" else "depth16"
            logDepthSampleDiag("pass2_wide", pass2, imageX, imageY, bboxHeightPixels)
            val median = medianWithIqrFilter(pass2.validDepths)
            if (median != null) {
                return DepthDistanceDebug(median, "depth_ok_${formatLabel}_wide")
            }
        }

        logDepthSampleDiag("fail", pass1, imageX, imageY, bboxHeightPixels)
        return DepthDistanceDebug(null, "depth_no_valid_samples")
    }

    private data class GridSampleResult(
        val totalSamples: Int,
        val zeroOrInvalid: Int,
        val belowMin: Int,
        val aboveMax: Int,
        val lowConfidence: Int,
        val validDepths: ArrayList<Float>,
        val regionX1: Float, val regionY1: Float,
        val regionX2: Float, val regionY2: Float,
    )

    private fun sampleBboxGridDepth(
        buf: java.nio.ByteBuffer,
        rowStride: Int,
        pixelStride: Int,
        depthFormat: Int,
        depthW: Int,
        depthH: Int,
        rawImageWidth: Int,
        rawImageHeight: Int,
        imgX1: Float, imgY1: Float,
        imgX2: Float, imgY2: Float,
        gridCols: Int, gridRows: Int,
        minMm: Int, maxMm: Int,
    ): GridSampleResult {
        val clampedX1 = imgX1.coerceIn(0f, rawImageWidth.toFloat())
        val clampedY1 = imgY1.coerceIn(0f, rawImageHeight.toFloat())
        val clampedX2 = imgX2.coerceIn(0f, rawImageWidth.toFloat())
        val clampedY2 = imgY2.coerceIn(0f, rawImageHeight.toFloat())
        val spanX = (clampedX2 - clampedX1).coerceAtLeast(1f)
        val spanY = (clampedY2 - clampedY1).coerceAtLeast(1f)

        val totalSamples = gridCols * gridRows
        var zeroOrInvalid = 0
        var belowMin = 0
        var aboveMax = 0
        var lowConfidence = 0
        val validDepths = ArrayList<Float>(totalSamples)

        for (row in 0 until gridRows) {
            val imgY = clampedY1 + spanY * (row.toFloat() / max(gridRows - 1, 1).toFloat())
            for (col in 0 until gridCols) {
                val imgX = clampedX1 + spanX * (col.toFloat() / max(gridCols - 1, 1).toFloat())
                val nx = imgX / rawImageWidth.toFloat()
                val ny = imgY / rawImageHeight.toFloat()
                val depthPx = min(max((nx * (depthW - 1)).roundToInt(), 0), depthW - 1)
                val depthPy = min(max((ny * (depthH - 1)).roundToInt(), 0), depthH - 1)
                val sample = readDepth16Sample(buf, rowStride, pixelStride, depthPx, depthPy)
                val mm = depthMillimetersFromSample(sample, depthFormat)
                val confidence = depthConfidenceFromSample(sample, depthFormat)
                when {
                    mm <= 0 -> zeroOrInvalid++
                    mm < minMm -> belowMin++
                    mm > maxMm -> aboveMax++
                    confidence <= 0f -> lowConfidence++
                    else -> validDepths += mm / 1000f
                }
            }
        }
        return GridSampleResult(
            totalSamples, zeroOrInvalid, belowMin, aboveMax, lowConfidence,
            validDepths, clampedX1, clampedY1, clampedX2, clampedY2,
        )
    }

    /**
     * Median of the middle 50% (IQR) to reject outliers from mixed foreground/background depth.
     * Returns null if fewer than 2 samples survive.
     */
    private fun medianWithIqrFilter(depths: ArrayList<Float>): Float? {
        if (depths.size < 2) return if (depths.size == 1) depths[0] else null
        depths.sort()
        val q1Idx = depths.size / 4
        val q3Idx = (depths.size * 3) / 4
        val iqrSlice = depths.subList(q1Idx, q3Idx + 1)
        if (iqrSlice.isEmpty()) return depths[depths.size / 2]
        return iqrSlice[iqrSlice.size / 2]
    }

    private var lastDepthDiagLogMs = 0L

    private fun logDepthSampleDiag(
        passLabel: String,
        result: GridSampleResult,
        imageX: Float,
        imageY: Float,
        bboxH: Float,
    ) {
        val now = android.os.SystemClock.elapsedRealtime()
        if (now - lastDepthDiagLogMs < 800L) return
        lastDepthDiagLogMs = now
        LogUtil.furnitureFitAr(buildString {
            append("platform=android phase=depth_sample pass=$passLabel ")
            append("total=${result.totalSamples} valid=${result.validDepths.size} ")
            append("zero=${result.zeroOrInvalid} belowMin=${result.belowMin} aboveMax=${result.aboveMax} lowConf=${result.lowConfidence} ")
            append("bboxCenter=(${String.format("%.1f", imageX)},${String.format("%.1f", imageY)}) bboxH=${String.format("%.1f", bboxH)} ")
            append("region=(${String.format("%.0f", result.regionX1)},${String.format("%.0f", result.regionY1)})->(${String.format("%.0f", result.regionX2)},${String.format("%.0f", result.regionY2)})")
            if (result.validDepths.isNotEmpty()) {
                result.validDepths.sort()
                append(" medianM=${String.format("%.3f", result.validDepths[result.validDepths.size / 2])}")
                append(" rangeM=${String.format("%.3f", result.validDepths.first())}..${String.format("%.3f", result.validDepths.last())}")
            }
        })
    }

    private fun sampleSparseMetricAnchorsFromDepthImage(
        depthImage: Image,
        orientedImageWidth: Int,
        orientedImageHeight: Int,
        rawImageWidth: Int,
        rawImageHeight: Int,
        orientedToRawInverse: Matrix?,
        gridSize: Int,
        marginFraction: Float,
        minimumConfidence: Float,
        rejectDepthDiscontinuity: Boolean,
        minDepthMillimeters: Int,
        maxDepthMillimeters: Int,
    ): List<MetricAnchor> {
        if (!supportsDepthImageFormat(depthImage.format)) return emptyList()
        val depthFormat = depthImage.format
        if (orientedImageWidth <= 0 || orientedImageHeight <= 0 || rawImageWidth <= 0 || rawImageHeight <= 0) return emptyList()
        val plane = depthImage.planes.firstOrNull() ?: return emptyList()
        val buffer = plane.buffer.duplicate()
        val rowStride = plane.rowStride
        val pixelStride = plane.pixelStride
        if (pixelStride < 2 || rowStride <= 0) return emptyList()

        val depthWidth = depthImage.width
        val depthHeight = depthImage.height
        val anchors = ArrayList<MetricAnchor>(gridSize * gridSize)
        val marginX = orientedImageWidth * marginFraction
        val marginY = orientedImageHeight * marginFraction
        val usableWidth = (orientedImageWidth - 2f * marginX).coerceAtLeast(1f)
        val usableHeight = (orientedImageHeight - 2f * marginY).coerceAtLeast(1f)

        for (gridY in 0 until gridSize) {
            val orientedY = if (gridSize == 1) orientedImageHeight * 0.5f else marginY + usableHeight * (gridY.toFloat() / (gridSize - 1).toFloat())
            for (gridX in 0 until gridSize) {
                val orientedX = if (gridSize == 1) orientedImageWidth * 0.5f else marginX + usableWidth * (gridX.toFloat() / (gridSize - 1).toFloat())
                val rawPoint = floatArrayOf(orientedX, orientedY)
                orientedToRawInverse?.mapPoints(rawPoint)
                val rawX = rawPoint[0]
                val rawY = rawPoint[1]
                if (!rawX.isFinite() || !rawY.isFinite()) continue

                val depthPx = min(max(((rawX / rawImageWidth.toFloat()) * (depthWidth - 1)).roundToInt(), 0), depthWidth - 1)
                val depthPy = min(max(((rawY / rawImageHeight.toFloat()) * (depthHeight - 1)).roundToInt(), 0), depthHeight - 1)
                val sample = readDepth16Sample(buffer, rowStride, pixelStride, depthPx, depthPy)
                val millimeters = depthMillimetersFromSample(sample, depthFormat)
                val confidence = depthConfidenceFromSample(sample, depthFormat)
                if (millimeters !in minDepthMillimeters..maxDepthMillimeters || confidence < minimumConfidence) continue
                if (rejectDepthDiscontinuity &&
                    hasDepthDiscontinuity(buffer, rowStride, pixelStride, depthPx, depthPy, depthWidth, depthHeight, millimeters)
                ) continue

                anchors += MetricAnchor(
                    pixelX = orientedX.roundToInt(),
                    pixelY = orientedY.roundToInt(),
                    depthMeters = millimeters / 1000f,
                    confidence = confidence,
                    originalImageWidth = orientedImageWidth,
                    originalImageHeight = orientedImageHeight,
                )
            }
        }
        return anchors
    }

    private fun hasDepthDiscontinuity(
        buffer: java.nio.ByteBuffer,
        rowStride: Int,
        pixelStride: Int,
        depthPx: Int,
        depthPy: Int,
        depthWidth: Int,
        depthHeight: Int,
        centerMillimeters: Int,
    ): Boolean {
        val center = centerMillimeters.toFloat().coerceAtLeast(1f)
        val neighbors = arrayOf(
            depthPx - 1 to depthPy,
            depthPx + 1 to depthPy,
            depthPx to depthPy - 1,
            depthPx to depthPy + 1,
        )
        for ((neighborX, neighborY) in neighbors) {
            if (neighborX !in 0 until depthWidth || neighborY !in 0 until depthHeight) continue
            val sample = readDepth16Sample(buffer, rowStride, pixelStride, neighborX, neighborY)
            val millimeters = sample and 0x1FFF
            val confidence = (sample ushr 13) and 0x7
            if (millimeters <= 0 || confidence <= 0) continue
            if (abs(millimeters - centerMillimeters) / center > 0.3f) {
                return true
            }
        }
        return false
    }

    private fun readDepth16Sample(
        buf: java.nio.ByteBuffer,
        rowStride: Int,
        pixelStride: Int,
        x: Int,
        y: Int,
    ): Int {
        val offset = y * rowStride + x * pixelStride
        val lo = buf.get(offset).toInt() and 0xFF
        val hi = buf.get(offset + 1).toInt() and 0xFF
        return lo or (hi shl 8)
    }

    fun estimatedPhysicalHeightMeters(
        bboxHeightPixels: Float,
        distanceMeters: Float,
        focalLengthYPixels: Float,
    ): Float? {
        if (!bboxHeightPixels.isFinite() || !distanceMeters.isFinite() || !focalLengthYPixels.isFinite()) return null
        if (focalLengthYPixels == 0f) return null
        if (bboxHeightPixels <= 1f || distanceMeters <= 0.1f || focalLengthYPixels <= 1f) return null
        return (bboxHeightPixels / focalLengthYPixels) * distanceMeters
    }

    /**
     * **Diagnostic / legacy only:** `roomHeight × (bbox_h / image_h)` is **not** physically
     * stable when the camera moves — the bbox shrinks in pixels as distance increases. Do not use
     * for AR overlay scale or primary furniture height; prefer [estimatedPhysicalHeightMeters].
     */
    fun approximateHeightFromRoomAndBboxFraction(
        roomHeightMeters: Float,
        bboxHeightPixels: Float,
        imageHeightPixels: Int,
    ): Float? {
        if (roomHeightMeters <= 0.1f || imageHeightPixels <= 0) return null
        if (!bboxHeightPixels.isFinite()) return null
        val frac = (bboxHeightPixels / imageHeightPixels.toFloat()).coerceIn(0.06f, 0.92f)
        return roomHeightMeters * frac
    }

    fun overlayScaleFromMetricHeights(
        standardHeightMeters: Float,
        estimatedHeightMeters: Float,
        minScale: Float = 0.4f,
        maxScale: Float = 2.5f,
    ): Float? {
        if (standardHeightMeters <= 0.1f || estimatedHeightMeters <= 0.1f) return null
        // When the overlay is already sized so that the reference height appears correct for
        // the room, this factor adjusts it so the final visual height matches the AR-estimated
        // height. If AR says the object is taller than the reference, we scale up (>1); if
        // shorter, we scale down (<1).
        val raw = estimatedHeightMeters / standardHeightMeters
        return raw.coerceIn(minScale, maxScale)
    }

    /**
     * [fy] in pixels for the camera image; scales if [imageHeight] differs from [CameraIntrinsics] image height.
     */
    fun focalLengthYPixelsForImage(intrinsics: CameraIntrinsics, imageWidth: Int, imageHeight: Int): Float {
        val dim = intrinsics.imageDimensions
        val intrW = max(dim[0], 1)
        val intrH = max(dim[1], 1)
        val fl = intrinsics.focalLength
        val fyBase = if (fl.size >= 2) fl[1] else fl[0]
        val scaleX = imageWidth.toFloat() / intrW.toFloat()
        val scaleY = imageHeight.toFloat() / intrH.toFloat()
        val scale = kotlin.math.sqrt(scaleX * scaleY)
        return fyBase * scale
    }

    /**
     * Maps a point from CPU image pixels to view pixels for [Frame.hitTest].
     * @return false if ARCore could not transform (caller should skip hit test).
     */
    fun imagePixelsToViewPixels(frame: Frame, imageX: Float, imageY: Float, outView: FloatArray): Boolean {
        return try {
            val input = floatArrayOf(imageX, imageY)
            frame.transformCoordinates2d(Coordinates2d.IMAGE_PIXELS, input, Coordinates2d.VIEW, outView)
            true
        } catch (_: Exception) {
            false
        }
    }
}
