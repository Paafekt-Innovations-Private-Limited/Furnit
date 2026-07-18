package com.furnit.android.models.roomintelligence

/**
 * Pure evaluator that can be called from both GLBRoomActivity and FurnitureFitFragment.
 *
 * It makes only whole-room dimension claims. It deliberately does not infer free floor area or
 * placement locations.
 */
object RoomIntelligenceEngine {
    @JvmStatic
    fun evaluate(input: RoomIntelligenceInput): RoomIntelligenceResult {
        val aesthetic = input.furnitureAestheticProfile?.let { furnitureProfile ->
            val styleTags = if (input.roomStyleTags.isEmpty()) {
                RoomStyleInference.infer(input.roomPalette)
            } else {
                input.roomStyleTags
            }
            AestheticAdvisor.evaluate(input.roomPalette, styleTags, furnitureProfile)
        } ?: AestheticEvaluation.Unavailable

        val room = input.roomDimensions
        if (room == null || !room.isUsableForFit) {
            return RoomIntelligenceResult(
                status = RoomIntelligenceStatus.MEASURING,
                dimensionFit = null,
                aesthetic = aesthetic,
                furnitureDimensions = input.furnitureDimensions,
            )
        }

        val furniture = input.furnitureDimensions
        if (furniture == null || !furniture.isUsableForFit) {
            return RoomIntelligenceResult(
                status = if (aesthetic is AestheticEvaluation.Available) {
                    RoomIntelligenceStatus.STYLE_ONLY
                } else {
                    RoomIntelligenceStatus.MEASURING
                },
                dimensionFit = null,
                aesthetic = aesthetic,
                furnitureDimensions = null,
            )
        }

        val unrotatedFits =
            furniture.widthMeters <= room.widthMeters &&
                furniture.depthMeters <= room.depthMeters
        val rotatedFits =
            furniture.depthMeters <= room.widthMeters &&
                furniture.widthMeters <= room.depthMeters
        val heightFits = furniture.heightMeters <= room.heightMeters
        val fits = heightFits && (unrotatedFits || rotatedFits)
        return RoomIntelligenceResult(
            status = if (fits) {
                RoomIntelligenceStatus.FITS_BY_DIMENSIONS
            } else {
                RoomIntelligenceStatus.DOES_NOT_FIT
            },
            dimensionFit = DimensionFitResult(
                fits = fits,
                rotatedFootprint = fits && !unrotatedFits && rotatedFits,
            ),
            aesthetic = aesthetic,
            furnitureDimensions = furniture,
        )
    }

    @JvmStatic
    fun evaluateMeasuredFurniture(
        roomDimensions: RoomDimensions?,
        furnitureWidthMeters: Float?,
        furnitureHeightMeters: Float?,
        roomPalette: SurfacePalette = SurfacePalette.EMPTY,
        roomStyleTags: Collection<String> = emptyList(),
        furnitureAestheticProfile: FurnitureAestheticProfile? = null,
    ): RoomIntelligenceResult {
        val furnitureDimensions =
            if (furnitureWidthMeters != null && furnitureHeightMeters != null) {
                FurnitureDimensions.fromMeasuredWidthAndHeight(
                    furnitureWidthMeters,
                    furnitureHeightMeters,
                )
            } else {
                null
            }
        return evaluate(
            RoomIntelligenceInput(
                roomDimensions = roomDimensions,
                furnitureDimensions = furnitureDimensions,
                roomPalette = roomPalette,
                roomStyleTags = roomStyleTags,
                furnitureAestheticProfile = furnitureAestheticProfile,
            ),
        )
    }
}
