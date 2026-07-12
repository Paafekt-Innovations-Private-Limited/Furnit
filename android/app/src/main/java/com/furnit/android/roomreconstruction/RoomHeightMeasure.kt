package com.furnit.android.roomreconstruction

import kotlin.math.abs
import kotlin.math.atan
import kotlin.math.tan

data class RoomHeightResult(
    val height: Float,
    val confidence: Float,
    val approximate: Boolean,
    val vFloor: Float?,
    val vCeil: Float?,
    val vHorizon: Float?,
    val normalSign: Float,
    val debug: String,
    val selfCheckDebug: String,
)

data class RoomWidthResult(
    val width: Float,
    val confidence: Float,
    val approximate: Boolean,
    val debug: String,
)

object RoomHeightMeasure {
    fun roomHeightSingleView(
        pointGrid: LeveledDepthPointGrid,
        fy: Float,
        cy: Float,
        pitch: Float,
        cameraHeight: Float = 1.60f,
    ): RoomHeightResult {
        val variants = listOf(
            1f to 1f,
            -1f to 1f,
            1f to -1f,
            -1f to -1f,
        )
        var best: RoomHeightResult? = null
        for ((pitchSign, normalSign) in variants) {
            val result = estimateOnce(
                pointGrid = pointGrid,
                fy = fy,
                cy = cy,
                pitch = pitch * pitchSign,
                cameraHeight = cameraHeight,
                normalSign = normalSign,
                variantLabel = "pitchSign=${pitchSign.toInt()} normalSign=${normalSign.toInt()}",
            )
            if (result.selfCheckDebug.contains("ok=true") && result.confidence >= 0.5f) {
                return result
            }
            if (best == null || result.confidence > best!!.confidence) {
                best = result
            }
        }
        return best ?: RoomHeightResult(
            height = 0f,
            confidence = 0.2f,
            approximate = true,
            vFloor = null,
            vCeil = null,
            vHorizon = null,
            normalSign = 1f,
            debug = "height_unavailable",
            selfCheckDebug = "ok=false no_variants",
        )
    }

    fun roomWidthSingleView(
        pointGrid: LeveledDepthPointGrid,
        vFloor: Float?,
        vHorizon: Float?,
        vCeil: Float?,
        normalSign: Float,
        cameraHeight: Float = 1.60f,
    ): RoomWidthResult {
        if (vFloor == null || vHorizon == null || vCeil == null ||
            vFloor <= vHorizon || vFloor <= vCeil
        ) {
            return RoomWidthResult(0f, 0.2f, true, "junctions_missing")
        }

        val backWall = backWallPixelWidth(pointGrid, vFloor, vCeil, normalSign)
        val denominator = vFloor - vHorizon
        if (backWall.span <= 0f || denominator <= 1f) {
            return RoomWidthResult(
                0f, 0.2f, true,
                "backwall_missing count=${backWall.count} denom=${denominator.toInt()}",
            )
        }

        val clampedCameraHeight = cameraHeight.coerceIn(1.55f, 1.75f)
        var roomWidth = backWall.span * clampedCameraHeight / denominator
        var confidence = if (backWall.count > 2_000) 0.8f else 0.5f
        if (roomWidth !in 1.0f..12.0f) confidence = 0.3f
        roomWidth = roomWidth.coerceIn(1.0f, 12.0f)
        return RoomWidthResult(
            width = roomWidth,
            confidence = confidence,
            approximate = confidence < 0.7f,
            debug = "W=%.3f span=%dpx xL=%d xR=%d count=%d denom=%d".format(
                roomWidth, backWall.span.toInt(), backWall.xLeft, backWall.xRight,
                backWall.count, denominator.toInt(),
            ),
        )
    }

    private fun estimateOnce(
        pointGrid: LeveledDepthPointGrid,
        fy: Float,
        cy: Float,
        pitch: Float,
        cameraHeight: Float,
        normalSign: Float,
        variantLabel: String,
    ): RoomHeightResult {
        val width = pointGrid.width
        val clampedCameraHeight = cameraHeight.coerceIn(1.55f, 1.75f)
        val horizonRow = cy + fy * tan(pitch)

        val floorRows = ArrayList<Float>(180)
        val ceilingRows = ArrayList<Float>(180)
        val xStart = maxOf(1, width * 2 / 5)
        val xEnd = minOf(width - 2, width * 3 / 5)
        val xStep = maxOf(1, (xEnd - xStart) / 180)
        var x = xStart
        while (x <= xEnd) {
            floorWallRow(pointGrid, x, normalSign)?.let { floorRows += it }
            ceilingWallRow(pointGrid, x, normalSign)?.let { ceilingRows += it }
            x += xStep
        }

        if (floorRows.size < 10 || ceilingRows.size < 10) {
            return RoomHeightResult(
                0f, 0.2f, true, null, null, horizonRow, normalSign,
                "junctions_missing f=${floorRows.size} c=${ceilingRows.size} $variantLabel",
                "vCeil=nan vHorizon=$horizonRow vFloor=nan ok=false $variantLabel",
            )
        }

        val floorRow = RoomMath.median(floorRows)
        val ceilingRow = RoomMath.median(ceilingRows)
        val selfCheckOK = floorRow > ceilingRow && ceilingRow < horizonRow && horizonRow < floorRow
        val selfCheck = "vCeil=%.1f vHorizon=%.1f vFloor=%.1f ok=%s %s".format(
            ceilingRow, horizonRow, floorRow, if (selfCheckOK) "true" else "false", variantLabel,
        )

        val alpha = atan((floorRow - horizonRow) / fy)
        val beta = atan((horizonRow - ceilingRow) / fy)
        if (alpha <= 0.02f || beta <= 0f) {
            return RoomHeightResult(
                0f, 0.2f, true, floorRow, ceilingRow, horizonRow, normalSign,
                "bad_angles a=%.3f b=%.3f vH=%.1f vF=%.1f vC=%.1f %s".format(
                    alpha, beta, horizonRow, floorRow, ceilingRow, variantLabel,
                ),
                selfCheck,
            )
        }

        var roomHeight = clampedCameraHeight * (tan(alpha) + tan(beta)) / tan(alpha)
        val floorStd = RoomMath.stdev(floorRows)
        val ceilingStd = RoomMath.stdev(ceilingRows)
        val rowsAgree = floorStd < 40f && ceilingStd < 40f
        var confidence = if (selfCheckOK && rowsAgree) 0.85f else if (selfCheckOK) 0.5f else 0.3f
        if (roomHeight !in 2.0f..3.6f) confidence = minOf(confidence, 0.3f)
        roomHeight = roomHeight.coerceIn(2.0f, 3.6f)

        return RoomHeightResult(
            height = roomHeight,
            confidence = confidence,
            approximate = confidence < 0.7f,
            vFloor = floorRow,
            vCeil = ceilingRow,
            vHorizon = horizonRow,
            normalSign = normalSign,
            debug = "H=%.3f hc=%.2f a=%.1fdeg b=%.1fdeg vH=%d vF=%d vC=%d fN=%d cN=%d fSd=%d cSd=%d %s".format(
                roomHeight, clampedCameraHeight,
                Math.toDegrees(alpha.toDouble()).toFloat(),
                Math.toDegrees(beta.toDouble()).toFloat(),
                horizonRow.toInt(), floorRow.toInt(), ceilingRow.toInt(),
                floorRows.size, ceilingRows.size, floorStd.toInt(), ceilingStd.toInt(),
                variantLabel,
            ),
            selfCheckDebug = selfCheck,
        )
    }

    private enum class Surface { FLOOR, CEILING, WALL }

    private fun classify(normal: Vec3): Surface {
        return when {
            normal.y > 0.7f -> Surface.FLOOR
            normal.y < -0.7f -> Surface.CEILING
            else -> Surface.WALL
        }
    }

    private fun worldNormal(pointGrid: LeveledDepthPointGrid, x: Int, y: Int, normalSign: Float): Vec3? {
        val width = pointGrid.width
        val height = pointGrid.height
        if (x <= 0 || x >= width - 1 || y <= 0 || y >= height - 1) return null
        val left = pointGrid.point(x - 1, y) ?: return null
        val right = pointGrid.point(x + 1, y) ?: return null
        val up = pointGrid.point(x, y - 1) ?: return null
        val down = pointGrid.point(x, y + 1) ?: return null
        val cross = Vec3.cross(right - left, down - up)
        val length = cross.length()
        if (length <= 1e-6f) return null
        val leveled = cross * (normalSign / length)
        return Vec3(leveled.x, -leveled.y, leveled.z).normalized()
    }

    private fun floorWallRow(pointGrid: LeveledDepthPointGrid, col: Int, normalSign: Float): Float? {
        var lastFloor = -1
        var y = pointGrid.height - 2
        while (y >= 1) {
            val normal = worldNormal(pointGrid, col, y, normalSign)
            if (normal != null) {
                when (classify(normal)) {
                    Surface.FLOOR -> lastFloor = y
                    Surface.WALL -> if (lastFloor >= 0) return lastFloor.toFloat()
                    else -> Unit
                }
            }
            y--
        }
        return if (lastFloor >= 0) lastFloor.toFloat() else null
    }

    private fun ceilingWallRow(pointGrid: LeveledDepthPointGrid, col: Int, normalSign: Float): Float? {
        var lastCeiling = -1
        var y = 1
        while (y < pointGrid.height - 1) {
            val normal = worldNormal(pointGrid, col, y, normalSign)
            if (normal != null) {
                when (classify(normal)) {
                    Surface.CEILING -> lastCeiling = y
                    Surface.WALL -> if (lastCeiling >= 0) return lastCeiling.toFloat()
                    else -> Unit
                }
            }
            y++
        }
        return if (lastCeiling >= 0) lastCeiling.toFloat() else null
    }

    private data class BackWallSpan(val span: Float, val count: Int, val xLeft: Int, val xRight: Int)

    private fun backWallPixelWidth(
        pointGrid: LeveledDepthPointGrid,
        vFloor: Float,
        vCeil: Float,
        normalSign: Float,
    ): BackWallSpan {
        val viewDirection = pointGrid.viewDirectionHorizontal ?: return BackWallSpan(0f, 0, 0, 0)
        fun isBackWall(normal: Vec3): Boolean {
            val length = kotlin.math.sqrt(normal.x * normal.x + normal.z * normal.z)
            if (length < 0.3f) return false
            val dot = (normal.x / length) * viewDirection.first + (normal.z / length) * viewDirection.second
            return abs(dot) > 0.7f
        }

        val yTop = (vCeil + (vFloor - vCeil) * 0.35f).toInt()
        val yBottom = (vCeil + (vFloor - vCeil) * 0.65f).toInt()
        val minY = maxOf(1, minOf(pointGrid.height - 2, yTop))
        val maxY = maxOf(1, minOf(pointGrid.height - 2, yBottom))
        if (minY > maxY) return BackWallSpan(0f, 0, 0, 0)

        val xs = ArrayList<Int>()
        for (y in minY..maxY) {
            for (x in 1 until pointGrid.width - 1) {
                val normal = worldNormal(pointGrid, x, y, normalSign) ?: continue
                if (isBackWall(normal)) xs += x
            }
        }
        if (xs.size <= 200) return BackWallSpan(0f, xs.size, 0, 0)
        xs.sort()
        val leftIndex = ((xs.size - 1) * 0.025).toInt().coerceIn(0, xs.lastIndex)
        val rightIndex = ((xs.size - 1) * 0.975).toInt().coerceIn(0, xs.lastIndex)
        val xLeft = xs[leftIndex]
        val xRight = xs[rightIndex]
        return BackWallSpan((xRight - xLeft).toFloat(), xs.size, xLeft, xRight)
    }
}
