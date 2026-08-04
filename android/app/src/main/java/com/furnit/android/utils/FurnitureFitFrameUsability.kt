package com.furnit.android.utils

import android.graphics.Bitmap
import android.graphics.ImageFormat
import android.graphics.PixelFormat
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
            PixelFormat.RGBA_8888 -> isFullyDarkRgbaPlane(imageProxy, sampleStep)
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
        val pixelStride = yPlane.pixelStride
        val width = imageProxy.width
        val height = imageProxy.height
        if (rowStride <= 0 || pixelStride <= 0) return false
        var sum = 0L
        var count = 0
        var y = 0
        while (y < height) {
            val rowStart = y * rowStride
            var x = 0
            while (x < width) {
                val offset = rowStart + x * pixelStride
                if (offset >= buffer.limit()) return false
                sum += buffer.get(offset).toInt() and 0xFF
                count++
                x += sampleStep
            }
            y += sampleStep
        }
        if (count <= 0) return false
        return (sum.toDouble() / count) <= MAX_MEAN_LUMINANCE
    }

    private fun isFullyDarkRgbaPlane(imageProxy: ImageProxy, sampleStep: Int): Boolean {
        val plane = imageProxy.planes.firstOrNull() ?: return false
        val buffer = plane.buffer.duplicate()
        val rowStride = plane.rowStride
        val pixelStride = plane.pixelStride
        val width = imageProxy.width
        val height = imageProxy.height
        if (rowStride <= 0 || pixelStride < 4) return false

        var sum = 0.0
        var count = 0
        var y = 0
        while (y < height) {
            val rowStart = y * rowStride
            var x = 0
            while (x < width) {
                val offset = rowStart + x * pixelStride
                if (offset + 2 >= buffer.limit()) return false
                val red = buffer.get(offset).toInt() and 0xff
                val green = buffer.get(offset + 1).toInt() and 0xff
                val blue = buffer.get(offset + 2).toInt() and 0xff
                sum += 0.299 * red + 0.587 * green + 0.114 * blue
                count++
                x += sampleStep
            }
            y += sampleStep
        }
        if (count <= 0) return false
        return (sum / count) <= MAX_MEAN_LUMINANCE
    }
}
