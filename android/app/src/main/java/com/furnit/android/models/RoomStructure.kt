package com.furnit.android.models

import android.graphics.PointF
import android.graphics.RectF
import java.io.Serializable

/**
 * RoomStructure - Holds room boundary data for manual room creation
 * (Matches Swift's RoomStructure)
 *
 * Boundary values are percentages (0.0 - 1.0) of image dimensions
 */
data class RoomStructure(
    // Boundary values from manual adjustment (as percentages)
    var floorY: Float = 0.85f,
    var ceilingY: Float = 0.15f,
    var leftX: Float = 0.12f,
    var rightX: Float = 0.88f,
    var vanishingX: Float = 0.5f,
    var vanishingY: Float = 0.45f,

    // Computed regions
    var floorRegion: RectF? = null,
    var ceilingRegion: RectF? = null,
    var vanishingPoint: PointF? = null,

    // Wall lines (for advanced processing)
    var wallLines: List<Line> = emptyList()
) : Serializable {

    data class Line(
        val start: PointF,
        val end: PointF,
        val confidence: Float
    ) : Serializable

    companion object {
        // Default boundary values
        const val DEFAULT_FLOOR_Y = 0.85f
        const val DEFAULT_CEILING_Y = 0.15f
        const val DEFAULT_LEFT_X = 0.12f
        const val DEFAULT_RIGHT_X = 0.88f
        const val DEFAULT_VANISHING_X = 0.5f
        const val DEFAULT_VANISHING_Y = 0.45f
    }

    fun reset() {
        floorY = DEFAULT_FLOOR_Y
        ceilingY = DEFAULT_CEILING_Y
        leftX = DEFAULT_LEFT_X
        rightX = DEFAULT_RIGHT_X
        vanishingX = DEFAULT_VANISHING_X
        vanishingY = DEFAULT_VANISHING_Y
    }
}
