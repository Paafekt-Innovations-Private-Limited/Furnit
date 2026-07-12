package com.furnit.android.roomreconstruction

import kotlin.math.atan
import kotlin.math.tan

data class ResolvedFocal(
    val fx: Float,
    val fy: Float,
    val source: String,
    val horizontalFOVDegrees: Float,
    val clamped: Boolean,
)

object FocalResolver {
    private const val MINIMUM_HORIZONTAL_FOV_DEGREES = 55f
    private const val MAXIMUM_HORIZONTAL_FOV_DEGREES = 88f
    private const val DEFAULT_HORIZONTAL_FOV_DEGREES = 70f

    fun resolve(
        vpFocalPx: Float?,
        geoCalib: GeoCalibCalibrationResult?,
        imageWidth: Int,
        imageHeight: Int,
    ): ResolvedFocal {
        val width = maxOf(imageWidth, 1).toFloat()

        if (vpFocalPx != null && vpFocalPx.isFinite() && vpFocalPx > 1f) {
            val clamped = clampFocal(vpFocalPx, width)
            return ResolvedFocal(clamped.focalPx, clamped.focalPx, "vp", clamped.fovDegrees, clamped.clamped)
        }

        if (geoCalib != null && geoCalib.sourceWidth > 0) {
            val uniformScale = width / maxOf(geoCalib.sourceWidth, 1).toFloat()
            val focal = geoCalib.focalLengthXPixels * uniformScale
            if (focal.isFinite() && focal > 1f) {
                val clamped = clampFocal(focal, width)
                val source = if (clamped.clamped) "geocalib_clamped" else "geocalib"
                return ResolvedFocal(clamped.focalPx, clamped.focalPx, source, clamped.fovDegrees, clamped.clamped)
            }
        }

        val defaultFocal = focalPixelsForHorizontalFov(DEFAULT_HORIZONTAL_FOV_DEGREES, width)
        val clamped = clampFocal(defaultFocal, width)
        return ResolvedFocal(clamped.focalPx, clamped.focalPx, "default_70deg", clamped.fovDegrees, clamped.clamped)
    }

    private data class ClampedFocal(val focalPx: Float, val fovDegrees: Float, val clamped: Boolean)

    private fun clampFocal(focalPx: Float, imageWidth: Float): ClampedFocal {
        val minFocal = focalPixelsForHorizontalFov(MAXIMUM_HORIZONTAL_FOV_DEGREES, imageWidth)
        val maxFocal = focalPixelsForHorizontalFov(MINIMUM_HORIZONTAL_FOV_DEGREES, imageWidth)
        val clampedFocal = focalPx.coerceIn(minFocal, maxFocal)
        return ClampedFocal(
            focalPx = clampedFocal,
            fovDegrees = horizontalFovDegrees(clampedFocal, imageWidth),
            clamped = kotlin.math.abs(clampedFocal - focalPx) > 0.5f,
        )
    }

    private fun focalPixelsForHorizontalFov(fovDegrees: Float, imageWidth: Float): Float {
        return (imageWidth * 0.5f) / tan(Math.toRadians(fovDegrees.toDouble() / 2.0).toFloat())
    }

    private fun horizontalFovDegrees(focalPx: Float, imageWidth: Float): Float {
        return (2f * atan((imageWidth * 0.5f) / maxOf(focalPx, 1f)) * 180f / Math.PI.toFloat())
    }
}
