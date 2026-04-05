package com.furnit.android.utils

import android.graphics.Bitmap
import com.furnit.android.models.roomintelligence.Vec3f

object FurnitureSegmentationMeanColor {
    fun meanStraightSrgb(bitmap: Bitmap, alphaThreshold: Int = 16): Vec3f? {
        val width = bitmap.width
        val height = bitmap.height
        if (width <= 0 || height <= 0) return null

        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)

        var sumR = 0.0
        var sumG = 0.0
        var sumB = 0.0
        var count = 0
        for (pixel in pixels) {
            val alpha = (pixel ushr 24) and 0xFF
            if (alpha < alphaThreshold) continue
            val alphaF = alpha / 255.0
            if (alphaF <= 1.0 / 255.0) continue
            val rPremul = (pixel shr 16) and 0xFF
            val gPremul = (pixel shr 8) and 0xFF
            val bPremul = pixel and 0xFF
            sumR += minOf(1.0, rPremul / alphaF / 255.0)
            sumG += minOf(1.0, gPremul / alphaF / 255.0)
            sumB += minOf(1.0, bPremul / alphaF / 255.0)
            count++
        }

        if (count <= 0) return null
        return Vec3f(
            x = (sumR / count).toFloat(),
            y = (sumG / count).toFloat(),
            z = (sumB / count).toFloat(),
        )
    }
}
