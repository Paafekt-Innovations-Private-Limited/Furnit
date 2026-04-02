package com.furnit.android.utils

object SharpRoomDimensionSanitizer {
    const val DEFAULT_DISPLAY_WIDTH_M: Float = 4f
    const val DEFAULT_DISPLAY_HEIGHT_M: Float = 3f
    const val DEFAULT_DISPLAY_DEPTH_M: Float = 4.5f

    fun sanitizeMeters(width: Float, height: Float, depth: Float): Triple<Float, Float, Float> {
        val safeW = if (width.isFinite() && width > 0f) width else DEFAULT_DISPLAY_WIDTH_M
        val safeH = if (height.isFinite() && height > 0f) height else DEFAULT_DISPLAY_HEIGHT_M
        val safeD = if (depth.isFinite() && depth > 0f) depth else DEFAULT_DISPLAY_DEPTH_M
        return Triple(safeW, safeH, safeD)
    }
}
