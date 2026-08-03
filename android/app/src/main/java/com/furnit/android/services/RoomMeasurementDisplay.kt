package com.furnit.android.services

import kotlin.math.max

/**
 * Shared resting-pill / ruler-hint copy for room measurement chrome (Swift PaafektRoomMeasurementDisplay parity).
 */
object RoomMeasurementDisplay {
    private const val MIN_VALID_METERS = 0.05f

    fun restingPillText(
        width: Float,
        height: Float,
        depth: Float,
        emphasizeHeight: Boolean,
        approximateHeightFormatter: (Float) -> String,
        widthDepthFormatter: (Float, Float) -> String,
    ): String? {
        if (emphasizeHeight && height > MIN_VALID_METERS && height.isFinite()) {
            return approximateHeightFormatter(height)
        }
        if (width > MIN_VALID_METERS && depth > MIN_VALID_METERS && width.isFinite() && depth.isFinite()) {
            return widthDepthFormatter(width, depth)
        }
        if (height > MIN_VALID_METERS && height.isFinite()) {
            return approximateHeightFormatter(height)
        }
        return null
    }

    fun meshRoomWidthMeters(measuredWidth: Float, imageWidth: Int, imageHeight: Int): Float {
        val aspectSafeWidth = max(measuredWidth, 2.0f)
        if (imageWidth <= 0) return aspectSafeWidth
        return max(aspectSafeWidth, 2.0f * imageWidth.toFloat() / max(imageHeight, 1).toFloat())
    }
}
