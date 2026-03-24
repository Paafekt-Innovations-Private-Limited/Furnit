package com.furnit.android.ar

import com.furnit.android.utils.YoloRatioCalibration

/**
 * Typical furniture heights (meters) for AR-assisted overlay scale, aligned with iOS [FurnitureDimensions] ideas.
 * Android path uses COCO-style labels from YOLOE.
 */
object FurnitureFitStandardHeights {

    /** Default when label is unknown (matches iOS fallback 0.85). */
    private const val DEFAULT_HEIGHT_M = 0.85f

    fun heightMetersForLabel(label: String): Float {
        val key = YoloRatioCalibration.canonicalFurnitureLabel(label)
        return when (key) {
            "chair", "bench" -> 0.9f
            "couch" -> 0.85f
            "bed" -> 0.6f
            "dining table" -> 0.75f
            else -> DEFAULT_HEIGHT_M
        }
    }
}
