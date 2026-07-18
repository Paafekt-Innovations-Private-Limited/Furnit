package com.furnit.android.roomreconstruction

import kotlin.math.roundToInt

data class Point3(val x: Float, val y: Float, val z: Float)

data class RoomExtentResult(
    val width: Float,
    val depth: Float,
    val height: Float,
    val confidence: Float,
    val approximate: Boolean,
    val debug: String,
)

data class DepthMaskResult(
    val valid: BooleanArray,
    val debug: String,
)

data class ObjectDetectionBox(
    val cls: Int,
    val box: IntArray,
    val conf: Float,
) {
    val x0: Int get() = box[0]
    val y0: Int get() = box[1]
    val x1: Int get() = box[2]
    val y1: Int get() = box[3]
}

object RoomExtentMeasure {
    fun buildInvalidDepthMask(
        depth: FloatArray,
        width: Int,
        height: Int,
        detections: List<ObjectDetectionBox>,
        focalPx: Float,
        cx: Float,
        cy: Float,
    ): DepthMaskResult {
        val pixelCount = (width * height).coerceAtLeast(0)
        val valid = BooleanArray(pixelCount) { true }
        if (width <= 1 || height <= 1 || depth.size != pixelCount) {
            return DepthMaskResult(valid, "invalid input valid=0/$pixelCount")
        }

        fun index(x: Int, y: Int) = y * width + x

        val central = ArrayList<Float>()
        for (y in height / 4 until 3 * height / 4) {
            for (x in width / 4 until 3 * width / 4) {
                val sample = depth[index(x, y)]
                if (sample.isFinite() && sample > 0f) central += sample
            }
        }
        central.sort()
        val wallRef = if (central.isEmpty()) 0f else central[central.size / 2]
        val beyondCut = if (wallRef > 0f) wallRef * 1.5f else Float.POSITIVE_INFINITY

        var beyond = 0
        for (i in depth.indices) {
            if (depth[i].isFinite() && depth[i] > beyondCut) {
                valid[i] = false
                beyond++
            }
        }

        var window = 0
        val sorted = depth.filter { it.isFinite() && it > 0f }.sorted()
        if (sorted.isNotEmpty()) {
            val p98Index = ((sorted.size - 1) * 0.98).toInt().coerceIn(0, sorted.lastIndex)
            val p98 = sorted[p98Index]
            for (i in depth.indices) {
                if (depth[i].isFinite() && depth[i] >= p98) {
                    if (valid[i]) window++
                    valid[i] = false
                }
            }
        }

        var mirror = 0
        for (detection in detections) {
            val x0 = detection.x0.coerceIn(0, width - 1)
            val y0 = detection.y0.coerceIn(0, height - 1)
            val x1 = detection.x1.coerceIn(0, width - 1)
            val y1 = detection.y1.coerceIn(0, height - 1)
            if (x0 >= x1 || y0 >= y1) continue

            val objectDepths = ArrayList<Float>()
            for (y in y0..y1) {
                for (x in x0..x1) {
                    val sample = depth[index(x, y)]
                    if (sample.isFinite() && sample > 0f) objectDepths += sample
                }
            }
            objectDepths.sort()
            if (objectDepths.isEmpty()) continue
            val p20 = objectDepths[minOf(objectDepths.lastIndex, objectDepths.size / 5)]
            if (wallRef > 0f && p20 > wallRef * 1.3f) {
                val padX = maxOf(1, (x1 - x0) / 2)
                val padY = maxOf(1, (y1 - y0) / 2)
                val mx0 = maxOf(0, x0 - padX)
                val mx1 = minOf(width - 1, x1 + padX)
                val my0 = maxOf(0, y0 - padY)
                val my1 = minOf(height - 1, y1 + padY)
                for (y in my0..my1) {
                    for (x in mx0..mx1) {
                        val i = index(x, y)
                        if (valid[i]) mirror++
                        valid[i] = false
                    }
                }
            }
        }

        val validCount = valid.count { it }
        return DepthMaskResult(
            valid,
            "wallRef=%.3f beyondCut=%.3f beyond=%d window=%d mirror≈%dpx valid=%d/%d focal=%.1f c=(%.1f,%.1f)".format(
                wallRef, beyondCut, beyond, window, mirror, validCount, pixelCount, focalPx, cx, cy,
            ),
        )
    }

    fun roomExtentPoints(
        pointGrid: LeveledDepthPointGrid,
        width: Int,
        height: Int,
        valid: BooleanArray,
        stride: Int = 2,
    ): List<Point3> {
        if (width <= 1 || height <= 1 || valid.size != width * height) return emptyList()
        val sampleStep = maxOf(1, stride)
        val points = ArrayList<Point3>()
        var y = 0
        while (y < height) {
            var x = 0
            while (x < width) {
                val index = y * width + x
                val point = pointGrid.point(x, y)
                if (valid[index] && point != null && point.z > 0f) {
                    points += Point3(point.x, -point.y, point.z)
                }
                x += sampleStep
            }
            y += sampleStep
        }
        return points
    }

    fun roomExtentFromWalls(
        points: List<Point3>,
        scale: Float,
        cameraHeight: Float = 1.65f,
    ): RoomExtentResult {
        if (points.size <= 500) {
            return RoomExtentResult(0f, 0f, 0f, 0.2f, true, "too few points ${points.size}")
        }

        val metricPoints = points.map { Point3(it.x * scale, it.y * scale, it.z * scale) }
        val ys = metricPoints.map { it.y }.sorted()
        val floorY = percentile(ys, 0.05)
        val ceilingY = percentile(ys, 0.95)
        var ceilingClearance = (ceilingY - floorY - cameraHeight).coerceIn(0.4f, 1.8f)
        val height = cameraHeight + ceilingClearance

        val lowBand = floorY + 0.4f
        val highBand = ceilingY - 0.4f
        val slab = metricPoints.filter { it.y > lowBand && it.y < highBand }
        val sourcePoints = if (slab.size > 300) slab else metricPoints

        val xSpan = robustSpan(sourcePoints.map { it.x })
        val zSpan = robustSpan(sourcePoints.map { it.z })
        var width = xSpan.second - xSpan.first
        var depth = zSpan.second - zSpan.first

        var approximate = false
        var confidence = 0.7f
        fun flag(condition: Boolean) {
            if (condition) {
                approximate = true
                confidence = 0.3f
            }
        }
        flag(height !in 2.0f..3.6f)
        flag(width !in 1.2f..12f)
        flag(depth !in 1.2f..15f)

        width = width.coerceIn(1.0f, 12f)
        depth = depth.coerceIn(1.0f, 15f)

        return RoomExtentResult(
            width = width,
            depth = depth,
            height = height,
            confidence = confidence,
            approximate = approximate,
            debug = "floorY=%.3f ceilY=%.3f clr=%.3f slab=%d pts=%d W=%.3f D=%.3f H=%.3f".format(
                floorY, ceilingY, ceilingClearance, slab.size, points.size, width, depth, height,
            ),
        )
    }

    private fun robustSpan(values: List<Float>): Pair<Float, Float> {
        val sorted = values.sorted()
        if (sorted.isEmpty()) return 0f to 0f
        return percentile(sorted, 0.025) to percentile(sorted, 0.975)
    }

    private fun percentile(sorted: List<Float>, fraction: Double): Float {
        if (sorted.isEmpty()) return 0f
        val index = ((sorted.size - 1) * fraction).roundToInt().coerceIn(0, sorted.lastIndex)
        return sorted[index]
    }
}
