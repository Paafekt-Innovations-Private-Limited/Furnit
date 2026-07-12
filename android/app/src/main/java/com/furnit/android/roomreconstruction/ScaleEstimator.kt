package com.furnit.android.roomreconstruction

typealias RoomDepthMap = Array<FloatArray>

data class ScaleObservation(
    val source: String,
    val tier: AnchorTier,
    val depthScale: Double,
    val detConf: Double,
    val impliedRoomHeight: Double,
    val debug: String,
)

enum class AnchorTier(val value: Int) {
    ARCHITECTURAL(1),
    CODE_FIXTURE(2),
    CAMERA_HEIGHT(3),
    FURNITURE(4),
}

data class ScaleObjectBox(
    val classIdx: Int,
    val confidence: Float,
    val leftX: Float,
    val topY: Float,
    val rightX: Float,
    val bottomY: Float,
)

data class SceneContext(
    val rawDepth: RoomDepthMap,
    val focalPx: Double,
    val levelingRotation: Mat3,
    val furnitureBoxes: List<FloatArray>,
    val objectBoxes: List<ScaleObjectBox>,
    val imageWidth: Int,
    val imageHeight: Int,
    val rawCameraHeightMeters: Double?,
    val fallbackDepthScale: Double,
    val impliedRoomHeightForScale: (Double) -> Double?,
)

data class ScaleEstimatorResult(
    val depthScale: Double,
    val confidence: Double,
    val source: String,
    val observations: List<ScaleObservation>,
    val clampedOut: List<ScaleObservation>,
)

interface ScaleAnchor {
    fun candidates(ctx: SceneContext): List<ScaleObservation>
}

class TileAnchor : ScaleAnchor {
    override fun candidates(ctx: SceneContext): List<ScaleObservation> = emptyList()
}

class ObjectAnchor : ScaleAnchor {
    override fun candidates(ctx: SceneContext): List<ScaleObservation> {
        return ctx.objectBoxes.mapNotNull { objectBox ->
            val prior = priorFor(objectBox, ctx) ?: return@mapNotNull null
            val rawDepth = depthPercentile(ctx.rawDepth, objectBox, 0.20) ?: return@mapNotNull null
            val pixelSize = when (prior.dimension) {
                Dimension.WIDTH -> objectBox.rightX - objectBox.leftX
                Dimension.HEIGHT -> objectBox.bottomY - objectBox.topY
            }
            if (pixelSize <= 8f || ctx.focalPx <= 1.0) return@mapNotNull null
            val rawSize = pixelSize * rawDepth / ctx.focalPx
            if (!rawSize.isFinite() || rawSize <= 0.05) return@mapNotNull null
            val scale = prior.meters / rawSize
            val impliedHeight = ctx.impliedRoomHeightForScale(scale) ?: return@mapNotNull null
            if (!scale.isFinite() || scale <= 0.0) return@mapNotNull null
            ScaleObservation(
                source = prior.source,
                tier = prior.tier,
                depthScale = scale,
                detConf = objectBox.confidence.toDouble(),
                impliedRoomHeight = impliedHeight,
                debug = "cls=${objectBox.classIdx} conf=%.2f px=%.0f rawSize=%.3fm prior=%.2fm tier=${prior.tier.value}".format(
                    objectBox.confidence, pixelSize, rawSize, prior.meters,
                ),
            )
        }
    }

    private enum class Dimension { WIDTH, HEIGHT }

    private data class Prior(val source: String, val dimension: Dimension, val meters: Double, val tier: AnchorTier)

    private fun priorFor(objectBox: ScaleObjectBox, ctx: SceneContext): Prior? {
        return when (objectBox.classIdx) {
            56, 57, 58, 59 -> null
            61 -> {
                val imageHeight = maxOf(ctx.imageHeight, 1).toFloat()
                val bottomMargin = (imageHeight - objectBox.bottomY) / imageHeight
                val heightFraction = (objectBox.bottomY - objectBox.topY) / imageHeight
                if (bottomMargin <= 0.05f && heightFraction < 0.20f) {
                    Prior("toilet_seat", Dimension.HEIGHT, 0.40, AnchorTier.CODE_FIXTURE)
                } else {
                    Prior("toilet_full", Dimension.HEIGHT, 0.78, AnchorTier.CODE_FIXTURE)
                }
            }
            else -> null
        }
    }

    private fun depthPercentile(depthMap: RoomDepthMap, box: ScaleObjectBox, fraction: Double): Double? {
        if (depthMap.isEmpty()) return null
        val width = depthMap[0].size
        val height = depthMap.size
        val left = box.leftX.toInt().coerceIn(0, width - 1)
        val right = box.rightX.toInt().coerceIn(0, width - 1)
        val top = box.topY.toInt().coerceIn(0, height - 1)
        val bottom = box.bottomY.toInt().coerceIn(0, height - 1)
        if (left >= right || top >= bottom) return null
        val maxSpan = maxOf(right - left, bottom - top)
        val step = maxOf(1, maxSpan / 120)
        val samples = ArrayList<Float>()
        var row = top
        while (row <= bottom) {
            var column = left
            while (column <= right) {
                val depth = depthMap[row][column]
                if (depth.isFinite() && depth > 0.1f && depth < 50f) samples += depth
                column += step
            }
            row += step
        }
        if (samples.isEmpty()) return null
        samples.sort()
        val index = ((fraction * (samples.size - 1)).toInt()).coerceIn(0, samples.lastIndex)
        return samples[index].toDouble()
    }
}

class ScaleEstimator(
    private val anchors: List<ScaleAnchor> = listOf(TileAnchor(), ObjectAnchor()),
) {
    fun estimate(ctx: SceneContext): ScaleEstimatorResult {
        val observations = anchors.flatMap { it.candidates(ctx) }
        val resolved = resolveScale(observations, ctx.rawCameraHeightMeters, ctx.impliedRoomHeightForScale)
        return ScaleEstimatorResult(
            depthScale = resolved.scale,
            confidence = resolved.confidence,
            source = resolved.source,
            observations = resolved.pool,
            clampedOut = resolved.clampedOut,
        )
    }

    private data class ResolvedScale(
        val scale: Double,
        val confidence: Double,
        val source: String,
        val pool: List<ScaleObservation>,
        val clampedOut: List<ScaleObservation>,
    )

    private fun resolveScale(
        observations: List<ScaleObservation>,
        cameraHeightRawMeters: Double?,
        impliedRoomHeightForScale: (Double) -> Double?,
    ): ResolvedScale {
        val valid = observations.filter { observation ->
            observation.depthScale.isFinite() &&
                observation.depthScale > 0.0 &&
                observation.impliedRoomHeight in 1.9..3.6
        }
        val clampedOut = observations.filter { candidate ->
            valid.none { kept ->
                kept.source == candidate.source &&
                    kept.depthScale == candidate.depthScale &&
                    kept.impliedRoomHeight == candidate.impliedRoomHeight
            }
        }

        val tier1 = valid.filter { it.tier == AnchorTier.ARCHITECTURAL && it.detConf > 0.5 }
        if (tier1.isNotEmpty()) {
            return ResolvedScale(medianScale(tier1), 0.9, "tier1_architectural", tier1, clampedOut)
        }

        val tier2 = valid.filter { it.tier == AnchorTier.CODE_FIXTURE && it.detConf > 0.5 }
        if (tier2.isNotEmpty()) {
            return ResolvedScale(medianScale(tier2), 0.75, "tier2_fixture", tier2, clampedOut)
        }

        if (cameraHeightRawMeters == null || !cameraHeightRawMeters.isFinite() || cameraHeightRawMeters <= 0.0) {
            return ResolvedScale(1.0, 0.05, "tier3_camera_height_unavailable", emptyList(), clampedOut)
        }
        val cameraHeightMeters = 1.65.coerceIn(1.55, 1.75)
        val scale = cameraHeightMeters / cameraHeightRawMeters
        val impliedHeight = impliedRoomHeightForScale(scale) ?: Double.NaN
        val cameraObservation = ScaleObservation(
            source = "camera_height",
            tier = AnchorTier.CAMERA_HEIGHT,
            depthScale = scale,
            detConf = 1.0,
            impliedRoomHeight = impliedHeight,
            debug = "clampedH=%.2fm rawH=%.3fm".format(cameraHeightMeters, cameraHeightRawMeters),
        )
        return ResolvedScale(scale, 0.5, "tier3_camera_height", listOf(cameraObservation), clampedOut)
    }

    private fun medianScale(observations: List<ScaleObservation>): Double {
        val scales = observations.map { it.depthScale }.sorted()
        if (scales.isEmpty()) return 1.0
        val middle = scales.size / 2
        return if (scales.size % 2 == 0) {
            (scales[middle - 1] + scales[middle]) * 0.5
        } else {
            scales[middle]
        }
    }
}

object VanishingPointGravity {
    data class Result(val levelingRotation: Mat3, val confidence: Double, val debug: String)

    fun refine(levelingRotation: Mat3, focalPx: Double): Result {
        return Result(levelingRotation, 0.0, "vp_refiner_unimplemented_using_input_gravity")
    }
}
