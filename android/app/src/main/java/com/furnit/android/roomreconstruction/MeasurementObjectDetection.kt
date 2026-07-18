package com.furnit.android.roomreconstruction

import android.content.Context
import android.graphics.Bitmap
import com.furnit.android.DetectionResult
import com.furnit.android.services.FurnitureFitManager
import com.furnit.android.services.SegmentationResult
import com.furnit.android.utils.LogUtil
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sqrt

data class MeasurementObjectRect(
    val classIdx: Int,
    val confidence: Float,
    val leftX: Int,
    val rightX: Int,
    val topY: Int,
    val bottomY: Int,
)

data class ObjectBBoxMeasurement(
    val classIdx: Int,
    val width: Float,
    val height: Float,
    val depth: Float,
    val bboxLeftX: Int,
    val bboxRightX: Int,
    val bboxTopY: Int,
    val bboxBottomY: Int,
)

object MeasurementObjectDetection {
    private const val TAG = "MeasurementObjectDetect"

    fun detectMeasurementObject(context: Context, image: Bitmap): MeasurementObjectRect? {
        val manager = FurnitureFitManager(context)
        val latch = CountDownLatch(1)
        var segmentation: SegmentationResult? = null
        manager.detectWithDetectionsAsync(image) { result ->
            segmentation = result
            latch.countDown()
        }
        if (!latch.await(45, TimeUnit.SECONDS)) {
            LogUtil.w(TAG, "RTMDet measurement detection timed out")
            return null
        }
        val detections = segmentation?.detections ?: return null
        val imageWidth = image.width
        val imageHeight = image.height
        val detection = selectMeasurementObjectBBox(detections, imageWidth, imageHeight) ?: return null
        val rect = clampedBBox(detection, imageWidth, imageHeight)
        if (rect.leftX >= rect.rightX || rect.topY >= rect.bottomY) return null
        return MeasurementObjectRect(
            classIdx = detection.classId,
            confidence = detection.confidence,
            leftX = rect.leftX,
            rightX = rect.rightX,
            topY = rect.topY,
            bottomY = rect.bottomY,
        )
    }

    fun measureObjectBBox(
        objectRect: MeasurementObjectRect?,
        depth: FloatArray,
        imageWidth: Int,
        imageHeight: Int,
        fx: Float,
        fy: Float,
    ): ObjectBBoxMeasurement? {
        if (objectRect == null) return null
        val depthSample = depthPercentile(
            depth, imageWidth, imageHeight,
            objectRect.leftX, objectRect.rightX, objectRect.topY, objectRect.bottomY, 0.20f,
        ) ?: return null
        val bboxWidthPixels = (objectRect.rightX - objectRect.leftX).toFloat()
        val bboxHeightPixels = (objectRect.bottomY - objectRect.topY).toFloat()
        val widthMeters = bboxWidthPixels * depthSample / fx
        val heightMeters = bboxHeightPixels * depthSample / fy
        if (!widthMeters.isFinite() || !heightMeters.isFinite() || !depthSample.isFinite() ||
            widthMeters <= 0.03f || heightMeters <= 0.03f || depthSample <= 0.1f
        ) {
            return null
        }
        return ObjectBBoxMeasurement(
            classIdx = objectRect.classIdx,
            width = widthMeters,
            height = heightMeters,
            depth = depthSample,
            bboxLeftX = objectRect.leftX,
            bboxRightX = objectRect.rightX,
            bboxTopY = objectRect.topY,
            bboxBottomY = objectRect.bottomY,
        )
    }

    private fun selectMeasurementObjectBBox(
        detections: List<DetectionResult>,
        imageWidth: Int,
        imageHeight: Int,
    ): DetectionResult? {
        val imageArea = max(1, imageWidth * imageHeight).toFloat()
        val centerX = imageWidth * 0.5f
        val centerY = imageHeight * 0.5f
        val maxCenterDistance = max(1f, sqrt(centerX * centerX + centerY * centerY))
        val filtered = detections.filter { detection ->
            val area = detection.w * detection.h
            val areaFraction = area / imageArea
            detection.confidence >= RoomMeasurementConstants.OBJECT_BBOX_CONFIDENCE_THRESHOLD &&
                detection.w >= 8f &&
                detection.h >= 8f &&
                areaFraction >= 0.002f &&
                areaFraction <= 0.85f
        }
        val anchorPool = filtered.filter { RoomMeasurementConstants.objectAnchorHeightMeters.containsKey(it.classId) }
        val pool = if (anchorPool.isEmpty()) filtered else anchorPool
        return pool.maxByOrNull { detection ->
            objectBBoxScore(detection, imageArea, centerX, centerY, maxCenterDistance)
        }
    }

    private fun objectBBoxScore(
        detection: DetectionResult,
        imageArea: Float,
        centerX: Float,
        centerY: Float,
        maxCenterDistance: Float,
    ): Float {
        val areaFraction = ((detection.w * detection.h) / imageArea).coerceIn(0f, 1f)
        val dx = detection.x - centerX
        val dy = detection.y - centerY
        val centerDistance = sqrt(dx * dx + dy * dy) / maxCenterDistance
        var score = detection.confidence * 0.55f + sqrt(areaFraction) * 0.45f - centerDistance * 0.10f
        if (RoomMeasurementConstants.objectAnchorHeightMeters.containsKey(detection.classId)) {
            score += 0.20f
        }
        return score
    }

    private data class BBox(val leftX: Int, val rightX: Int, val topY: Int, val bottomY: Int)

    private fun clampedBBox(detection: DetectionResult, imageWidth: Int, imageHeight: Int): BBox {
        val maxX = max(0, imageWidth - 1)
        val maxY = max(0, imageHeight - 1)
        val leftX = min(max(floor(detection.x - detection.w * 0.5f).toInt(), 0), maxX)
        val rightX = min(max(ceil(detection.x + detection.w * 0.5f).toInt(), 0), maxX)
        val topY = min(max(floor(detection.y - detection.h * 0.5f).toInt(), 0), maxY)
        val bottomY = min(max(ceil(detection.y + detection.h * 0.5f).toInt(), 0), maxY)
        return BBox(leftX, rightX, topY, bottomY)
    }

    private fun depthPercentile(
        depth: FloatArray,
        imageWidth: Int,
        imageHeight: Int,
        leftX: Int,
        rightX: Int,
        topY: Int,
        bottomY: Int,
        fraction: Float,
    ): Float? {
        val x0 = leftX.coerceIn(0, imageWidth - 1)
        val x1 = rightX.coerceIn(0, imageWidth - 1)
        val y0 = topY.coerceIn(0, imageHeight - 1)
        val y1 = bottomY.coerceIn(0, imageHeight - 1)
        if (x0 >= x1 || y0 >= y1) return null
        val maxSpan = max(x1 - x0, y1 - y0)
        val step = max(1, maxSpan / 160)
        val samples = ArrayList<Float>()
        var y = y0
        while (y <= y1) {
            var x = x0
            while (x <= x1) {
                val value = depth[y * imageWidth + x]
                if (value.isFinite() && value > 0.1f && value < 50f) samples += value
                x += step
            }
            y += step
        }
        if (samples.isEmpty()) return null
        samples.sort()
        val index = ((samples.size - 1) * fraction).roundToInt().coerceIn(0, samples.lastIndex)
        return samples[index]
    }
}
