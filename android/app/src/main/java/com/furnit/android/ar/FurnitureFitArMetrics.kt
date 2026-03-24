package com.furnit.android.ar

import com.google.ar.core.CameraIntrinsics
import com.google.ar.core.Coordinates2d
import com.google.ar.core.Frame
import com.google.ar.core.Plane
import com.google.ar.core.TrackingState
import kotlin.math.max

/**
 * Metric helpers for AR-assisted FurnitureFit overlay scale (Android / ARCore), mirroring iOS [FurnitureFitARSupport].
 */
object FurnitureFitArMetrics {

    /** First horizontal plane hit from a screen pixel (view coordinates, matches [Frame.hitTest]). */
    fun horizontalPlaneHitDistanceMeters(frame: Frame, xPx: Float, yPx: Float): Float? {
        if (frame.camera.trackingState != TrackingState.TRACKING) return null
        for (hit in frame.hitTest(xPx, yPx)) {
            val t = hit.trackable
            if (t is Plane && t.trackingState == TrackingState.TRACKING && t.isPoseInPolygon(hit.hitPose)) {
                if (t.type == Plane.Type.HORIZONTAL_UPWARD_FACING || t.type == Plane.Type.HORIZONTAL_DOWNWARD_FACING) {
                    return hit.distance
                }
            }
        }
        return null
    }

    fun estimatedPhysicalHeightMeters(
        bboxHeightPixels: Float,
        distanceMeters: Float,
        focalLengthYPixels: Float,
    ): Float? {
        if (bboxHeightPixels <= 1f || distanceMeters <= 0.05f || focalLengthYPixels <= 1f) return null
        return (bboxHeightPixels / focalLengthYPixels) * distanceMeters
    }

    fun overlayScaleFromMetricHeights(
        standardHeightMeters: Float,
        estimatedHeightMeters: Float,
        minScale: Float = 0.4f,
        maxScale: Float = 2.5f,
    ): Float? {
        if (standardHeightMeters <= 0.1f || estimatedHeightMeters <= 0.05f) return null
        val raw = standardHeightMeters / estimatedHeightMeters
        return raw.coerceIn(minScale, maxScale)
    }

    /**
     * [fy] in pixels for the camera image; scales if [imageHeight] differs from [CameraIntrinsics] image height.
     */
    fun focalLengthYPixelsForImage(intrinsics: CameraIntrinsics, imageWidth: Int, imageHeight: Int): Float {
        val dim = intrinsics.imageDimensions
        val intrW = max(dim[0], 1)
        val intrH = max(dim[1], 1)
        val fl = intrinsics.focalLength
        val fyBase = if (fl.size >= 2) fl[1] else fl[0]
        val scaleX = imageWidth.toFloat() / intrW.toFloat()
        val scaleY = imageHeight.toFloat() / intrH.toFloat()
        val scale = kotlin.math.sqrt(scaleX * scaleY)
        return fyBase * scale
    }

    /**
     * Maps a point from CPU image pixels to view pixels for [Frame.hitTest].
     * @return false if ARCore could not transform (caller should skip hit test).
     */
    fun imagePixelsToViewPixels(frame: Frame, imageX: Float, imageY: Float, outView: FloatArray): Boolean {
        return try {
            val input = floatArrayOf(imageX, imageY)
            frame.transformCoordinates2d(Coordinates2d.IMAGE_PIXELS, input, Coordinates2d.VIEW, outView)
            true
        } catch (_: Exception) {
            false
        }
    }
}
