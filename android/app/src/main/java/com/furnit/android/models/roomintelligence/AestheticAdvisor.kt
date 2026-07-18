package com.furnit.android.models.roomintelligence

import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.sqrt

/**
 * Deterministic, localization-free aesthetic heuristics. Inputs and outputs use straight sRGB.
 */
object AestheticAdvisor {
    private val compatibleStyles = mapOf(
        "modern" to setOf("modern", "minimalist", "contemporary", "industrial"),
        "rustic" to setOf("rustic", "farmhouse", "traditional", "eclectic"),
        "scandinavian" to setOf("scandinavian", "minimalist", "modern", "nordic"),
        "industrial" to setOf("industrial", "modern", "eclectic"),
        "traditional" to setOf("traditional", "rustic", "classic"),
    )

    @JvmStatic
    fun evaluate(
        palette: SurfacePalette,
        roomStyleTags: Collection<String>,
        furniture: FurnitureAestheticProfile,
    ): AestheticEvaluation {
        val weightedRoomColors = weightedColors(palette)
        if (weightedRoomColors.isEmpty()) {
            return AestheticEvaluation.Unavailable
        }

        val harmony = harmony(furniture.primaryColor, weightedRoomColors)
        val contrast = contrast(furniture.primaryColor, weightedRoomColors.map { it.color })
        val style = styleCompatibility(roomStyleTags, furniture.styleTags)
        val recommendations = recommendations(harmony, contrast, style)
        return AestheticEvaluation.Available(
            harmonyScore = harmony.score.coerceIn(0f, 1f),
            harmonyType = harmony.type,
            contrastScore = contrast.coerceIn(0f, 1f),
            styleCompatibilityScore = style.coerceIn(0f, 1f),
            recommendations = recommendations,
        )
    }

    private data class LchColor(val lightness: Float, val chroma: Float, val hue: Float)
    private data class WeightedColor(val color: StraightSrgbColor, val weight: Float)
    private data class Harmony(val score: Float, val type: HarmonyType)

    private fun harmony(
        furnitureColor: StraightSrgbColor,
        roomColors: List<WeightedColor>,
    ): Harmony {
        val furnitureLch = toLch(furnitureColor)
        if (furnitureLch.chroma < 12f) {
            val bestScore = roomColors.maxOf { room ->
                val lightnessAlignment =
                    1f - min(abs(toLch(room.color).lightness - furnitureLch.lightness) / 40f, 1f)
                ((0.62f + lightnessAlignment * 0.24f) * room.weight).coerceIn(0f, 1f)
            }
            return Harmony(bestScore, HarmonyType.NEUTRAL)
        }

        var best = Harmony(0.18f, HarmonyType.CLASH)
        for (room in roomColors) {
            val roomLch = toLch(room.color)
            val hueDelta = hueDifference(furnitureLch.hue, roomLch.hue)
            val candidates = listOf(
                HarmonyType.ANALOGOUS to bellCurve(hueDelta, 18f, 18f),
                HarmonyType.COMPLEMENTARY to bellCurve(hueDelta, 180f, 22f),
                HarmonyType.TRIADIC to bellCurve(hueDelta, 120f, 16f),
                HarmonyType.SPLIT_COMPLEMENTARY to bellCurve(hueDelta, 150f, 18f),
                HarmonyType.NEUTRAL to
                    ((1f - max(furnitureLch.chroma, roomLch.chroma) / 70f).coerceIn(0f, 1f) * 0.92f),
            )
            val bestCandidate = candidates.maxByOrNull { it.second } ?: continue
            val chromaBalance = 1f - min(abs(furnitureLch.chroma - roomLch.chroma) / 55f, 1f)
            val lightnessBalance =
                1f - min(abs(furnitureLch.lightness - roomLch.lightness) / 45f, 1f)
            val score = (
                (bestCandidate.second * 0.62f + chromaBalance * 0.20f +
                    lightnessBalance * 0.18f) * room.weight
                ).coerceIn(0f, 1f)
            if (score > best.score) {
                best = Harmony(score, bestCandidate.first)
            }
        }
        return if (best.score < 0.34f) best.copy(type = HarmonyType.CLASH) else best
    }

    private fun contrast(
        furnitureColor: StraightSrgbColor,
        roomColors: List<StraightSrgbColor>,
    ): Float {
        val furnitureLightness = toLch(furnitureColor).lightness
        return roomColors.maxOf { roomColor ->
            val difference = abs(furnitureLightness - toLch(roomColor).lightness)
            val penalty = if (difference < 8f) 0.25f else 0f
            (bellCurve(difference, 26f, 18f) - penalty).coerceIn(0f, 1f)
        }
    }

    private fun styleCompatibility(
        roomStyleTags: Collection<String>,
        furnitureStyleTags: Collection<String>,
    ): Float {
        val roomTags = roomStyleTags.map(String::trim).map(String::lowercase).filter(String::isNotEmpty)
        val furnitureTags =
            furnitureStyleTags.map(String::trim).map(String::lowercase).filter(String::isNotEmpty)
        if (roomTags.isEmpty() || furnitureTags.isEmpty()) return 0.5f

        var matches = 0
        for (roomTag in roomTags) {
            val compatible = compatibleStyles[roomTag].orEmpty()
            matches += furnitureTags.count { it == roomTag || it in compatible }
        }
        return (matches.toFloat() / max(roomTags.size, furnitureTags.size)).coerceIn(0f, 1f)
    }

    private fun recommendations(
        harmony: Harmony,
        contrast: Float,
        style: Float,
    ): List<AestheticRecommendationCode> {
        val result = mutableListOf<AestheticRecommendationCode>()
        when (harmony.type) {
            HarmonyType.CLASH -> result += AestheticRecommendationCode.USE_NEUTRAL_BRIDGE
            HarmonyType.ANALOGOUS -> result += AestheticRecommendationCode.ANALOGOUS_HARMONY
            HarmonyType.COMPLEMENTARY ->
                result += AestheticRecommendationCode.COMPLEMENTARY_FOCAL_POINT
            else -> Unit
        }
        when {
            contrast < 0.15f -> result += AestheticRecommendationCode.INCREASE_CONTRAST
            contrast > 0.80f -> result += AestheticRecommendationCode.SOFTEN_CONTRAST
        }
        if (style < 0.30f) result += AestheticRecommendationCode.STYLE_MISMATCH
        if (result.isEmpty()) result += AestheticRecommendationCode.BROADLY_COMPATIBLE
        return result.toList()
    }

    private fun weightedColors(palette: SurfacePalette): List<WeightedColor> = buildList {
        palette.floor?.dominantColors?.forEach { add(WeightedColor(it.clamped(), 1f / 1.12f)) }
        palette.walls?.dominantColors?.forEach { add(WeightedColor(it.clamped(), 1f)) }
        palette.ceiling?.dominantColors?.forEach { add(WeightedColor(it.clamped(), 0.72f / 1.12f)) }
    }

    private fun bellCurve(value: Float, target: Float, tolerance: Float): Float {
        val normalized = (value - target) / tolerance
        return exp((-0.5f * normalized * normalized).toDouble()).toFloat()
    }

    private fun hueDifference(left: Float, right: Float): Float {
        val difference = abs(left - right)
        return if (difference > 180f) 360f - difference else difference
    }

    private fun toLch(color: StraightSrgbColor): LchColor {
        fun linearize(component: Float): Float {
            val clamped = component.coerceIn(0f, 1f)
            return if (clamped <= 0.04045f) {
                clamped / 12.92f
            } else {
                ((clamped + 0.055f) / 1.055f).pow(2.4f)
            }
        }

        val red = linearize(color.red)
        val green = linearize(color.green)
        val blue = linearize(color.blue)
        val x = red * 0.4124f + green * 0.3576f + blue * 0.1805f
        val y = red * 0.2126f + green * 0.7152f + blue * 0.0722f
        val z = red * 0.0193f + green * 0.1192f + blue * 0.9505f

        fun labFunction(value: Float): Float =
            if (value > 0.008856f) value.pow(1f / 3f) else 7.787f * value + 16f / 116f

        val fx = labFunction(x / 0.95047f)
        val fy = labFunction(y)
        val fz = labFunction(z / 1.08883f)
        val lightness = 116f * fy - 16f
        val a = 500f * (fx - fy)
        val b = 200f * (fy - fz)
        val chroma = sqrt(a * a + b * b)
        var hue = Math.toDegrees(atan2(b, a).toDouble()).toFloat()
        if (hue < 0f) hue += 360f
        return LchColor(lightness, chroma, hue)
    }
}

object RoomStyleInference {
    @JvmStatic
    fun infer(palette: SurfacePalette): List<String> {
        val styles = sortedSetOf<String>()
        listOfNotNull(palette.floor, palette.walls, palette.ceiling).forEach { surface ->
            when (surface.materialHint) {
                MaterialHint.WOOD -> styles += listOf("rustic", "traditional")
                MaterialHint.TILE -> styles += "modern"
                MaterialHint.CONCRETE -> styles += listOf("industrial", "modern")
                MaterialHint.CARPET -> styles += listOf("traditional", "eclectic")
                MaterialHint.PLASTER -> styles += listOf("modern", "scandinavian")
                MaterialHint.BRICK -> styles += listOf("traditional", "industrial")
                MaterialHint.MARBLE -> styles += listOf("modern", "luxury")
                MaterialHint.UNKNOWN -> Unit
            }
        }
        return styles.take(6)
    }
}
