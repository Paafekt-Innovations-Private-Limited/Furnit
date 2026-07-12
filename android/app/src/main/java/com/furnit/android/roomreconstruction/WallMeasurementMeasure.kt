package com.furnit.android.roomreconstruction

import kotlin.math.abs
import kotlin.math.roundToInt

data class WallMeasurement(val width: Float, val height: Float, val depth: Float)

object WallMeasurementMeasure {
    fun measureWall(
        depth: FloatArray,
        imageWidth: Int,
        imageHeight: Int,
        fx: Float,
        fy: Float,
        wallMargin: Float,
    ): WallMeasurement {
        val margin = wallMargin.coerceIn(0f, 0.45f)
        val rectX = margin * imageWidth
        val rectY = margin * imageHeight
        val rectWidth = (1f - 2f * margin) * imageWidth
        val rectHeight = (1f - 2f * margin) * imageHeight

        val centerX = (imageWidth - 1) * 0.5f
        val centerY = (imageHeight - 1) * 0.5f
        val leftX = rectX.roundToInt().coerceIn(0, imageWidth - 1)
        val rightX = (rectX + rectWidth - 1f).roundToInt().coerceIn(0, imageWidth - 1)
        val topY = rectY.roundToInt().coerceIn(0, imageHeight - 1)
        val bottomY = (rectY + rectHeight - 1f).roundToInt().coerceIn(0, imageHeight - 1)
        val sampleCenterX = (rectX + rectWidth * 0.5f).roundToInt().coerceIn(0, imageWidth - 1)
        val sampleCenterY = (rectY + rectHeight * 0.5f).roundToInt().coerceIn(0, imageHeight - 1)

        val centerDepth = medianAt(depth, imageWidth, imageHeight, sampleCenterX, sampleCenterY)
            ?: centerDepthFallback(depth)
        val leftPlane = (leftX - centerX) * centerDepth / fx
        val rightPlane = (rightX - centerX) * centerDepth / fx
        val topPlane = (topY - centerY) * centerDepth / fy
        val bottomPlane = (bottomY - centerY) * centerDepth / fy
        return WallMeasurement(
            width = abs(rightPlane - leftPlane),
            height = abs(bottomPlane - topPlane),
            depth = centerDepth,
        )
    }

    fun sanitizeRoomMeasurement(spread: WallMeasurement, wallFallback: WallMeasurement): WallMeasurement {
        if (isPlausible(spread)) return spread
        if (isPlausible(wallFallback)) return wallFallback
        return spread
    }

    private fun isPlausible(measurement: WallMeasurement): Boolean {
        return measurement.width in RoomMeasurementConstants.PLAUSIBLE_SPAN_MIN..RoomMeasurementConstants.PLAUSIBLE_SPAN_MAX &&
            measurement.height in RoomMeasurementConstants.PLAUSIBLE_SPAN_MIN..RoomMeasurementConstants.PLAUSIBLE_SPAN_MAX &&
            measurement.depth in RoomMeasurementConstants.PLAUSIBLE_SPAN_MIN..RoomMeasurementConstants.PLAUSIBLE_SPAN_MAX
    }

    private fun medianAt(
        depth: FloatArray,
        imageWidth: Int,
        imageHeight: Int,
        x: Int,
        y: Int,
        radius: Int = 5,
    ): Float? {
        val values = ArrayList<Float>()
        val yStart = maxOf(0, y - radius)
        val yEnd = minOf(imageHeight - 1, y + radius)
        val xStart = maxOf(0, x - radius)
        val xEnd = minOf(imageWidth - 1, x + radius)
        for (row in yStart..yEnd) {
            for (col in xStart..xEnd) {
                val value = depth[row * imageWidth + col]
                if (value.isFinite() && value > 0f) values += value
            }
        }
        if (values.isEmpty()) return null
        return RoomMath.median(values)
    }

    private fun centerDepthFallback(depth: FloatArray): Float {
        val valid = depth.filter { it.isFinite() && it > 0f }
        if (valid.isEmpty()) return 3.0f
        return RoomMath.median(valid.sorted())
    }
}
