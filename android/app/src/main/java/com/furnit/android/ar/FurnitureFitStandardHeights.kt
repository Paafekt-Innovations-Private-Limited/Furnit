package com.furnit.android.ar

import com.furnit.android.utils.YoloRatioCalibration

/**
 * Reference height in meters for AR-assisted overlay scale.
 *
 * NOTE: We intentionally do **not** vary this by class anymore — relying on
 * per-label \"standard\" sizes was too restrictive for generic fitment. The
 * AR path can still estimate absolute height in meters; this is only a
 * neutral reference when a height is required.
 */
object FurnitureFitStandardHeights {

    /** Default when label is unknown (matches iOS fallback 0.85). */
    private const val DEFAULT_HEIGHT_M = 0.85f

    fun heightMetersForLabel(@Suppress("UNUSED_PARAMETER") label: String): Float {
        // Single neutral value for all labels.
        return DEFAULT_HEIGHT_M
    }
}
