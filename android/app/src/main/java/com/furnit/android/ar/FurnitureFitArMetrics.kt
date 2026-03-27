package com.furnit.android.ar

import android.graphics.ImageFormat
import android.graphics.Matrix
import android.media.Image
import com.google.ar.core.CameraIntrinsics
import com.google.ar.core.Coordinates2d
import com.google.ar.core.Frame
import com.google.ar.core.Plane
import com.google.ar.core.TrackingState
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
    ): DistanceEstimate? {
        depthDistanceMeters(frame, imageX, imageY, bboxHeightPixels, rawImageWidth, rawImageHeight)?.let {
            return DistanceEstimate(it, "depth16")
        }

        val viewOut = FloatArray(2)
        if (!imagePixelsToViewPixels(frame, imageX, imageY, viewOut)) return null
        val plane = horizontalPlaneHitDistanceMeters(frame, viewOut[0], viewOut[1]) ?: return null
        return DistanceEstimate(plane, "plane")
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

    private fun depthDistanceMeters(
        frame: Frame,
        imageX: Float,
        imageY: Float,
        bboxHeightPixels: Float,
        rawImageWidth: Int,
        rawImageHeight: Int,
    ): Float? {
        val depthImage = try {
            frame.acquireDepthImage16Bits()
        } catch (_: Exception) {
            null
        } ?: return null

        return try {
            sampleDepthMetersFromImage(
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
    ): Float? {
        if (depthImage.format != ImageFormat.DEPTH16) return null
        val depthW = depthImage.width
        val depthH = depthImage.height
        if (depthW <= 0 || depthH <= 0 || rawImageWidth <= 0 || rawImageHeight <= 0) return null

        val plane = depthImage.planes.firstOrNull() ?: return null
        val buf = plane.buffer.duplicate()
        val rowStride = plane.rowStride
        val pixelStride = plane.pixelStride
        if (pixelStride < 2 || rowStride <= 0) return null

        val sampleOffset = (bboxHeightPixels * 0.12f).coerceIn(2f, 24f)
        val samples = arrayOf(
            floatArrayOf(imageX, imageY),
            floatArrayOf(imageX, imageY - sampleOffset),
            floatArrayOf(imageX, imageY + sampleOffset),
            floatArrayOf(imageX - sampleOffset * 0.35f, imageY),
            floatArrayOf(imageX + sampleOffset * 0.35f, imageY),
        )

        val validDepths = ArrayList<Float>(samples.size)
        for (pt in samples) {
            val nx = pt[0] / rawImageWidth.toFloat()
            val ny = pt[1] / rawImageHeight.toFloat()
            val depthPx = min(max((nx * (depthW - 1)).toInt(), 0), depthW - 1)
            val depthPy = min(max((ny * (depthH - 1)).toInt(), 0), depthH - 1)
            val sample = readDepth16Sample(buf, rowStride, pixelStride, depthPx, depthPy)
            val mm = sample and 0x1FFF
            val confidence = (sample ushr 13) and 0x7
            if (mm in 50..50000 && confidence > 0) {
                validDepths += mm / 1000f
            }
        }
        if (validDepths.isEmpty()) return null
        validDepths.sort()
        return validDepths[validDepths.size / 2]
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
        if (depthImage.format != ImageFormat.DEPTH16) return emptyList()
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
                val millimeters = sample and 0x1FFF
                val confidence = ((sample ushr 13) and 0x7) / 7f
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
        if (standardHeightMeters <= 0.1f || estimatedHeightMeters <= 0.05f) return null
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
