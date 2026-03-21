package com.furnit.android.services

import android.graphics.Bitmap

/**
 * Shared SHARP image preprocessing (1536×1536 path used by ExecuTorch INT8 and tests).
 *
 * Used for consistent SHARP input sizing across Kotlin tests and the INT8 pipeline.
 */
object SharpImagePreprocessor {

    const val INPUT_SIZE = 1536

    /**
     * Resize image to 1536×1536 for SHARP inference.
     * Uses bilinear filtering (filter=true).
     */
    fun resizeForSharp(bitmap: Bitmap): Bitmap {
        return Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)
    }
}
