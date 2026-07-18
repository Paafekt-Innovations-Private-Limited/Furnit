package com.furnit.android.models.roomintelligence

import java.util.Collections
import kotlin.math.max
import kotlin.math.min

private fun <T> immutableCopy(values: Collection<T>): List<T> =
    Collections.unmodifiableList(values.toList())

enum class RoomIntelligenceStatus {
    MEASURING,
    STYLE_ONLY,
    FITS_BY_DIMENSIONS,
    DOES_NOT_FIT,
}

data class RoomDimensions(
    val widthMeters: Float,
    val heightMeters: Float,
    val depthMeters: Float,
) {
    val isUsableForFit: Boolean
        get() = widthMeters.isFinite() && widthMeters > 0f &&
            heightMeters.isFinite() && heightMeters > 0f &&
            depthMeters.isFinite() && depthMeters > 0f
}

enum class DimensionSource {
    ARCORE_METRIC,
    DERIVED_FROM_WIDTH_RATIO,
}

data class FurnitureDimensions(
    val widthMeters: Float,
    val heightMeters: Float,
    val widthSource: DimensionSource = DimensionSource.ARCORE_METRIC,
    val heightSource: DimensionSource = DimensionSource.ARCORE_METRIC,
) {
    val depthMeters: Float = deriveDepthMeters(widthMeters)
    val depthSource: DimensionSource = DimensionSource.DERIVED_FROM_WIDTH_RATIO

    val isUsableForFit: Boolean
        get() = widthMeters.isFinite() && widthMeters > 0f &&
            heightMeters.isFinite() && heightMeters > 0f &&
            depthMeters.isFinite() && depthMeters > 0f

    companion object {
        const val MIN_DERIVED_DEPTH_METERS = 0.25f
        const val MAX_DERIVED_DEPTH_METERS = 1.4f
        const val DEPTH_TO_WIDTH_RATIO = 0.72f

        /**
         * Returns null until both measured dimensions are finite and positive.
         */
        @JvmStatic
        fun fromMeasuredWidthAndHeight(
            widthMeters: Float,
            heightMeters: Float,
        ): FurnitureDimensions? {
            if (!widthMeters.isFinite() || widthMeters <= 0f ||
                !heightMeters.isFinite() || heightMeters <= 0f
            ) {
                return null
            }
            return FurnitureDimensions(
                widthMeters = widthMeters,
                heightMeters = heightMeters,
            )
        }

        @JvmStatic
        fun deriveDepthMeters(widthMeters: Float): Float {
            if (!widthMeters.isFinite() || widthMeters <= 0f) return Float.NaN
            return min(
                max(widthMeters * DEPTH_TO_WIDTH_RATIO, MIN_DERIVED_DEPTH_METERS),
                MAX_DERIVED_DEPTH_METERS,
            )
        }
    }
}

data class StraightSrgbColor(
    val red: Float,
    val green: Float,
    val blue: Float,
) {
    init {
        require(red.isFinite() && green.isFinite() && blue.isFinite()) {
            "Color components must be finite"
        }
    }

    fun clamped(): StraightSrgbColor = StraightSrgbColor(
        red.coerceIn(0f, 1f),
        green.coerceIn(0f, 1f),
        blue.coerceIn(0f, 1f),
    )
}

enum class MaterialHint {
    WOOD,
    TILE,
    CARPET,
    CONCRETE,
    BRICK,
    PLASTER,
    MARBLE,
    UNKNOWN,
}

class SurfaceColors(
    dominantColors: Collection<StraightSrgbColor>,
    val materialHint: MaterialHint = MaterialHint.UNKNOWN,
) {
    val dominantColors: List<StraightSrgbColor> = immutableCopy(dominantColors)
}

data class SurfacePalette(
    val floor: SurfaceColors? = null,
    val walls: SurfaceColors? = null,
    val ceiling: SurfaceColors? = null,
) {
    val isEmpty: Boolean
        get() = listOfNotNull(floor, walls, ceiling).all { it.dominantColors.isEmpty() }

    companion object {
        @JvmField
        val EMPTY = SurfacePalette()
    }
}

class FurnitureAestheticProfile(
    val primaryColor: StraightSrgbColor,
    styleTags: Collection<String> = emptyList(),
    val accentColor: StraightSrgbColor? = null,
) {
    val styleTags: List<String> = immutableCopy(styleTags)
}

enum class HarmonyType {
    ANALOGOUS,
    COMPLEMENTARY,
    TRIADIC,
    SPLIT_COMPLEMENTARY,
    NEUTRAL,
    CLASH,
}

enum class AestheticRecommendationCode {
    AESTHETIC_UNAVAILABLE,
    USE_NEUTRAL_BRIDGE,
    ANALOGOUS_HARMONY,
    COMPLEMENTARY_FOCAL_POINT,
    INCREASE_CONTRAST,
    SOFTEN_CONTRAST,
    STYLE_MISMATCH,
    BROADLY_COMPATIBLE,
}

sealed class AestheticEvaluation {
    class Available(
        val harmonyScore: Float,
        val harmonyType: HarmonyType,
        val contrastScore: Float,
        val styleCompatibilityScore: Float,
        recommendations: Collection<AestheticRecommendationCode>,
    ) : AestheticEvaluation() {
        val recommendations: List<AestheticRecommendationCode> = immutableCopy(recommendations)

        init {
            require(harmonyScore in 0f..1f)
            require(contrastScore in 0f..1f)
            require(styleCompatibilityScore in 0f..1f)
        }
    }

    object Unavailable : AestheticEvaluation() {
        val recommendationCode: AestheticRecommendationCode =
            AestheticRecommendationCode.AESTHETIC_UNAVAILABLE
    }
}

data class DimensionFitResult(
    val fits: Boolean,
    val rotatedFootprint: Boolean,
)

class RoomIntelligenceInput(
    val roomDimensions: RoomDimensions?,
    val furnitureDimensions: FurnitureDimensions?,
    val roomPalette: SurfacePalette = SurfacePalette.EMPTY,
    roomStyleTags: Collection<String> = emptyList(),
    val furnitureAestheticProfile: FurnitureAestheticProfile? = null,
) {
    val roomStyleTags: List<String> = immutableCopy(roomStyleTags)
}

data class RoomIntelligenceResult(
    val status: RoomIntelligenceStatus,
    val dimensionFit: DimensionFitResult?,
    val aesthetic: AestheticEvaluation,
    val furnitureDimensions: FurnitureDimensions?,
    val fitBasis: FitBasis = FitBasis.ROOM_ENVELOPE_ONLY,
    val placementLocationEvaluated: Boolean = false,
)

enum class FitBasis {
    ROOM_ENVELOPE_ONLY,
}
