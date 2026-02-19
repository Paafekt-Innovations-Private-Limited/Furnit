package com.furnit.android.services

import android.graphics.Bitmap

/**
 * Shared SHARP image preprocessing. Matches SplitOnnxSharp / OnnxSharp exactly.
 *
 * Both ONNX and Native Pt backends use this to ensure identical input handling.
 */
object SharpImagePreprocessor {

    const val INPUT_SIZE = 1536

    /**
     * Resize image to 1536×1536 for SHARP inference. Matches ONNX backend.
     * Uses bilinear filtering (filter=true).
     */
    fun resizeForSharp(bitmap: Bitmap): Bitmap {
        return Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)
    }
}
