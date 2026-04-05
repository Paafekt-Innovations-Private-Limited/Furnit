package com.furnit.android.models.roomintelligence

import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.sqrt

data class FurnitureProfile(
    val primaryColor: Vec3f,
    val accentColor: Vec3f?,
    val styleTags: List<String>,
)

enum class HarmonyType {
    ANALOGOUS,
    COMPLEMENTARY,
    TRIADIC,
    SPLIT_COMPLEMENTARY,
    NEUTRAL,
    CLASH,
}

data class AestheticScore(
    val harmonyScore: Float,
    val harmonyType: HarmonyType,
    val contrastScore: Float,
    val styleCompatibilityScore: Float,
    val recommendations: List<String>,
)

private data class LchColor(
    val l: Float,
    val c: Float,
    val h: Float,
)

private data class WeightedRoomColor(
    val color: Vec3f,
    val weight: Float,
)

class AestheticAdvisor(
    private val palette: SurfacePalette,
    private val roomStyleTags: List<String> = emptyList(),
) {
    fun evaluate(furniture: FurnitureProfile): AestheticScore {
        val harmony = harmonyScore(furniture.primaryColor)
        val contrast = contrastScore(furniture.primaryColor)
        val style = styleScore(furniture.styleTags)
        val recommendations = buildRecommendations(harmony, contrast, style)
        return AestheticScore(
            harmonyScore = harmony.first,
            harmonyType = harmony.second,
            contrastScore = contrast,
            styleCompatibilityScore = style,
            recommendations = recommendations,
        )
    }

    private fun harmonyScore(furnitureColor: Vec3f): Pair<Float, HarmonyType> {
        val roomColors = weightedRoomColors()
        if (roomColors.isEmpty()) return 0.58f to HarmonyType.NEUTRAL

        val furnitureLch = srgbToLch(furnitureColor)
        if (furnitureLch.c < 12f) {
            val bestNeutral = roomColors.maxOfOrNull { room ->
                val roomLch = srgbToLch(room.color)
                val lightnessAlignment = 1f - min(abs(roomLch.l - furnitureLch.l) / 40f, 1f)
                clamp01((0.62f + lightnessAlignment * 0.24f) * room.weight)
            } ?: 0.72f
            return bestNeutral to HarmonyType.NEUTRAL
        }

        var bestScore = 0.18f
        var bestType = HarmonyType.CLASH
        roomColors.forEach { room ->
            val roomLch = srgbToLch(room.color)
            val hueDelta = hueDifference(furnitureLch.h, roomLch.h)

            val analogous = bellCurveScore(hueDelta, 18f, 18f)
            val complementary = bellCurveScore(hueDelta, 180f, 22f)
            val triadic = bellCurveScore(hueDelta, 120f, 16f)
            val splitComplementary = bellCurveScore(hueDelta, 150f, 18f)
            val neutral = clamp01(1f - max(furnitureLch.c, roomLch.c) / 70f) * 0.92f

            val bestRaw = listOf(
                HarmonyType.ANALOGOUS to analogous,
                HarmonyType.COMPLEMENTARY to complementary,
                HarmonyType.TRIADIC to triadic,
                HarmonyType.SPLIT_COMPLEMENTARY to splitComplementary,
                HarmonyType.NEUTRAL to neutral,
            ).maxByOrNull { it.second } ?: (HarmonyType.CLASH to 0.18f)

            val chromaBalance = 1f - min(abs(furnitureLch.c - roomLch.c) / 55f, 1f)
            val lightnessBalance = 1f - min(abs(furnitureLch.l - roomLch.l) / 45f, 1f)
            val weightedScore = clamp01(
                (bestRaw.second * 0.62f + chromaBalance * 0.20f + lightnessBalance * 0.18f) * room.weight,
            )
            if (weightedScore > bestScore) {
                bestScore = weightedScore
                bestType = bestRaw.first
            }
        }

        return if (bestScore < 0.34f) bestScore to HarmonyType.CLASH else bestScore to bestType
    }

    private fun contrastScore(furnitureColor: Vec3f): Float {
        val furnitureLch = srgbToLch(furnitureColor)
        val references = listOfNotNull(
            palette.floor?.dominantColors?.firstOrNull(),
            palette.walls?.dominantColors?.firstOrNull(),
            palette.ceiling?.dominantColors?.firstOrNull(),
        )
        if (references.isEmpty()) return 0.5f

        return references.maxOfOrNull { color ->
            val roomLch = srgbToLch(color)
            val delta = abs(furnitureLch.l - roomLch.l)
            val perceptualSeparation = bellCurveScore(delta, 26f, 18f)
            val safetyPenalty = if (delta < 8f) 0.25f else 0f
            clamp01(perceptualSeparation - safetyPenalty)
        } ?: 0.5f
    }

    private fun styleScore(furnitureTags: List<String>): Float {
        val normalizedRoom = roomStyleTags.map { it.lowercase() }
        val normalizedFurniture = furnitureTags.map { it.lowercase() }
        if (normalizedRoom.isEmpty() || normalizedFurniture.isEmpty()) return 0.5f
        var matches = 0
        normalizedRoom.forEach { roomTag ->
            val compatible = COMPATIBLE_STYLES[roomTag].orEmpty()
            matches += normalizedFurniture.count { it == roomTag || compatible.contains(it) }
        }
        return min(matches.toFloat() / max(normalizedFurniture.size, normalizedRoom.size).toFloat(), 1f)
    }

    private fun buildRecommendations(
        harmony: Pair<Float, HarmonyType>,
        contrast: Float,
        style: Float,
    ): List<String> {
        val result = mutableListOf<String>()
        when (harmony.second) {
            HarmonyType.CLASH -> result += "A neutral finish would sit more comfortably against this room palette."
            HarmonyType.ANALOGOUS -> result += "The furniture color sits close to the room palette, so the look will feel calm."
            HarmonyType.COMPLEMENTARY -> result += "The furniture color can work as a focal contrast piece here."
            else -> Unit
        }
        when {
            contrast < 0.15f -> result += "Contrast is low; the piece may blend into the room."
            contrast > 0.80f -> result += "Contrast is strong; this piece will stand out sharply."
        }
        if (style < 0.30f && roomStyleTags.isNotEmpty()) {
            result += "Style fit looks weak against the room's dominant material cues."
        }
        if (result.isEmpty()) {
            result += "Color and style look broadly compatible with the room."
        }
        return result
    }

    private fun weightedRoomColors(): List<WeightedRoomColor> {
        val colors = mutableListOf<WeightedRoomColor>()
        palette.floor?.dominantColors?.forEach { colors += WeightedRoomColor(it, 1.00f) }
        palette.walls?.dominantColors?.forEach { colors += WeightedRoomColor(it, 1.12f) }
        palette.ceiling?.dominantColors?.forEach { colors += WeightedRoomColor(it, 0.72f) }
        if (colors.isEmpty()) return emptyList()
        val maxWeight = colors.maxOf { it.weight }
        return colors.map { WeightedRoomColor(it.color, clamp01(it.weight / maxWeight)) }
    }

    companion object {
        private val COMPATIBLE_STYLES = mapOf(
            "modern" to setOf("modern", "minimalist", "contemporary", "industrial"),
            "rustic" to setOf("rustic", "farmhouse", "traditional", "eclectic"),
            "scandinavian" to setOf("scandinavian", "minimalist", "modern", "nordic"),
            "industrial" to setOf("industrial", "modern", "eclectic"),
            "traditional" to setOf("traditional", "rustic", "classic"),
        )
    }
}

private fun srgbToLch(rgb: Vec3f): LchColor {
    fun linearize(component: Float): Float {
        val clamped = component.coerceIn(0f, 1f)
        return if (clamped <= 0.04045f) clamped / 12.92f else ((clamped + 0.055f) / 1.055f).pow(2.4f)
    }

    val r = linearize(rgb.x)
    val g = linearize(rgb.y)
    val b = linearize(rgb.z)
    val x = r * 0.4124f + g * 0.3576f + b * 0.1805f
    val y = r * 0.2126f + g * 0.7152f + b * 0.0722f
    val z = r * 0.0193f + g * 0.1192f + b * 0.9505f

    fun f(value: Float): Float {
        return if (value > 0.008856f) value.pow(1f / 3f) else (7.787f * value + 16f / 116f)
    }

    val fx = f(x / 0.95047f)
    val fy = f(y / 1.00000f)
    val fz = f(z / 1.08883f)
    val l = 116f * fy - 16f
    val a = 500f * (fx - fy)
    val bLab = 200f * (fy - fz)
    val c = sqrt(a * a + bLab * bLab)
    var h = atan2(bLab, a) * (180f / Math.PI.toFloat())
    if (h < 0f) h += 360f
    return LchColor(l = l, c = c, h = h)
}

private fun hueDifference(lhs: Float, rhs: Float): Float {
    val diff = abs(lhs - rhs)
    return if (diff > 180f) 360f - diff else diff
}

private fun clamp01(value: Float): Float = value.coerceIn(0f, 1f)

private fun bellCurveScore(value: Float, target: Float, tolerance: Float): Float {
    if (tolerance <= 0f) return if (value == target) 1f else 0f
    val normalized = (value - target) / tolerance
    return exp(-0.5f * normalized * normalized)
}
