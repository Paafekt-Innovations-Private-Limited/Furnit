package com.furnit.android.roomreconstruction

import kotlin.math.roundToInt

object DepthSpreadMeasure {
    data class SpreadDims(val width: Float, val height: Float, val depth: Float)

    fun measureDepthSpread(
        pointGrid: LeveledDepthPointGrid,
        imageWidth: Int,
        imageHeight: Int,
        wallMargin: Float,
        scale: Float,
        cameraHeightPriorMeters: Float,
        sampleStep: Int = 8,
    ): SpreadDims? {
        if (imageWidth <= 1 || imageHeight <= 1) return null
        val margin = wallMargin.coerceIn(0f, 0.45f)
        val leftX = (margin * imageWidth).roundToInt()
        val rightX = ((1f - margin) * imageWidth).roundToInt() - 1
        val topY = (margin * imageHeight).roundToInt()
        val bottomY = ((1f - margin) * imageHeight).roundToInt() - 1
        if (leftX >= rightX || topY >= bottomY) return null

        val leveledX = ArrayList<Float>(512)
        val leveledY = ArrayList<Float>(512)
        val leveledZ = ArrayList<Float>(512)
        val ceilingClearances = ArrayList<Float>(128)
        val ceilingDepths = ArrayList<Float>(128)
        val ceilingRowCutoff = topY + RoomMeasurementConstants.CEILING_BAND_ROW_FRACTION * (bottomY - topY)

        var y = topY
        while (y <= bottomY) {
            var x = leftX
            while (x <= rightX) {
                val point = pointGrid.point(x, y, scale)
                if (point != null && point.z > 0f) {
                    leveledX += point.x
                    leveledY += point.y
                    leveledZ += point.z
                    if (y < ceilingRowCutoff && point.y < -RoomMeasurementConstants.MINIMUM_CEILING_CLEARANCE_METERS) {
                        ceilingClearances += -point.y
                        ceilingDepths += point.z
                    }
                }
                x += sampleStep
            }
            y += sampleStep
        }

        if (leveledZ.size < 32) return null
        val sortedZ = leveledZ.sorted()
        val roomDepth = RoomMath.percentile(sortedZ.toFloatArray(), 0.80) ?: return null
        if (roomDepth <= 0f) return null

        val wallX = ArrayList<Float>()
        val wallY = ArrayList<Float>()
        val farCutoff = 0.6f * roomDepth
        for (index in leveledZ.indices) {
            if (leveledZ[index] > farCutoff) {
                wallX += leveledX[index]
                wallY += leveledY[index]
            }
        }
        if (wallX.size < 64) {
            wallX.clear()
            wallY.clear()
            wallX.addAll(leveledX)
            wallY.addAll(leveledY)
        }
        val sortedWallX = wallX.sorted()
        val sortedWallY = wallY.sorted()
        val xLow = RoomMath.percentile(sortedWallX.toFloatArray(), 0.04) ?: return null
        val xHigh = RoomMath.percentile(sortedWallX.toFloatArray(), 0.96) ?: return null
        val yLow = RoomMath.percentile(sortedWallY.toFloatArray(), 0.03) ?: return null
        val yHigh = RoomMath.percentile(sortedWallY.toFloatArray(), 0.97) ?: return null
        var width = xHigh - xLow
        var height = yHigh - yLow

        if (ceilingClearances.size >= 32) {
            var clearance = RoomMath.median(ceilingClearances)
            val fit = robustLineFit(ceilingDepths, ceilingClearances)
            if (fit != null && fit.first > 0.02f && fit.second > RoomMeasurementConstants.MINIMUM_CEILING_CLEARANCE_METERS) {
                clearance = fit.second
            }
            val ceilingHeight = cameraHeightPriorMeters + clearance
            if (ceilingHeight in RoomMeasurementConstants.CEILING_ANCHORED_HEIGHT_MIN..RoomMeasurementConstants.CEILING_ANCHORED_HEIGHT_MAX) {
                height = ceilingHeight
            }
        }

        if (!width.isFinite() || !height.isFinite() || !roomDepth.isFinite() ||
            width <= 0f || height <= 0f
        ) {
            return null
        }
        return SpreadDims(width, height, roomDepth)
    }

    private fun robustLineFit(x: List<Float>, y: List<Float>): Pair<Float, Float>? {
        if (x.size != y.size || x.size < 16) return null
        val order = x.indices.sortedBy { x[it] }
        val half = x.size / 2
        val slopes = ArrayList<Float>(half)
        for (index in 0 until half) {
            val a = order[index]
            val b = order[index + half]
            val dx = x[b] - x[a]
            if (dx > 1e-3f) slopes += (y[b] - y[a]) / dx
        }
        if (slopes.size < 8) return null
        val slope = RoomMath.median(slopes)
        val residuals = FloatArray(x.size) { index -> y[index] - slope * x[index] }
        return slope to RoomMath.median(residuals)
    }
}
