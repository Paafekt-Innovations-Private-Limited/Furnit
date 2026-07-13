package com.furnit.android.utils

import android.graphics.Bitmap
import android.graphics.ImageFormat
import androidx.camera.core.ImageProxy

/**
 * Cheap pre-inference gate so Fit does not burn the detector on covered-lens / fully black frames.
 * Thresholds match iOS `FurnitureFitFrameUsability`.
 */
object FurnitureFitFrameUsability {
    /** Mean Rec.601 / Y luma (0…255). Covered lens frames are typically well below this. */
    const val MAX_MEAN_LUMINANCE = 12
    const val DEFAULT_SAMPLE_STEP = 16

    fun isFullyDark(imageProxy: ImageProxy, sampleStep: Int = DEFAULT_SAMPLE_STEP): Boolean {
        if (sampleStep <= 0 || imageProxy.width <= 0 || imageProxy.height <= 0) return false
        return when (imageProxy.format) {
            ImageFormat.YUV_420_888 -> isFullyDarkYPlane(imageProxy, sampleStep)
            else -> false
        }
    }

    fun isFullyDark(bitmap: Bitmap, sampleStep: Int = DEFAULT_SAMPLE_STEP): Boolean {
        val width = bitmap.width
        val height = bitmap.height
        if (sampleStep <= 0 || width <= 0 || height <= 0) return false
        if (bitmap.isRecycled) return false

        var sum = 0.0
        var count = 0
        var y = 0
        while (y < height) {
            var x = 0
            while (x < width) {
                val pixel = bitmap.getPixel(x, y)
                val r = (pixel shr 16) and 0xFF
                val g = (pixel shr 8) and 0xFF
                val b = pixel and 0xFF
                sum += 0.299 * r + 0.587 * g + 0.114 * b
                count++
                x += sampleStep
            }
            y += sampleStep
        }
        if (count <= 0) return false
        return (sum / count) <= MAX_MEAN_LUMINANCE
    }

    private fun isFullyDarkYPlane(imageProxy: ImageProxy, sampleStep: Int): Boolean {
        val yPlane = imageProxy.planes.getOrNull(0) ?: return false
        val buffer = yPlane.buffer.duplicate()
        val rowStride = yPlane.rowStride
        val width = imageProxy.width
        val height = imageProxy.height
        var sum = 0L
        var count = 0
        var y = 0
        while (y < height) {
            val rowStart = y * rowStride
            var x = 0
            while (x < width) {
                sum += buffer.get(rowStart + x).toInt() and 0xFF
                count++
                x += sampleStep
            }
            y += sampleStep
        }
        if (count <= 0) return false
        return (sum.toDouble() / count) <= MAX_MEAN_LUMINANCE
    }
}
