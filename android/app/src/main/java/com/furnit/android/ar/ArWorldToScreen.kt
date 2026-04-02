package com.furnit.android.ar

import android.opengl.Matrix
import com.google.ar.core.Frame

/**
 * Projects a world-space point to GL surface pixel coordinates (top-left origin).
 */
object ArWorldToScreen {

    /**
     * @return true if the point projects in front of the camera and inside a reasonable NDC range.
     */
    fun project(
        frame: Frame,
        worldX: Float,
        worldY: Float,
        worldZ: Float,
        viewWidthPx: Int,
        viewHeightPx: Int,
        outScreenXY: FloatArray,
    ): Boolean {
        val camera = frame.camera
        val viewM = FloatArray(16)
        val projM = FloatArray(16)
        camera.getViewMatrix(viewM, 0)
        camera.getProjectionMatrix(projM, 0, 0.1f, 100f)

        val world = floatArrayOf(worldX, worldY, worldZ, 1f)
        val eye = FloatArray(4)
        val clip = FloatArray(4)
        Matrix.multiplyMV(eye, 0, viewM, 0, world, 0)
        Matrix.multiplyMV(clip, 0, projM, 0, eye, 0)
        val w = clip[3]
        if (kotlin.math.abs(w) < 1e-6f) return false
        // In camera space, +Z is toward user in ARCore; points in front have negative eyeZ typically.
        // Use clip space w > 0 as rough "in front" check after projection.
        if (w <= 0f) return false
        val ndcX = clip[0] / w
        val ndcY = clip[1] / w
        if (ndcX < -1.1f || ndcX > 1.1f || ndcY < -1.1f || ndcY > 1.1f) return false
        outScreenXY[0] = (ndcX + 1f) * 0.5f * viewWidthPx
        outScreenXY[1] = (1f - ndcY) * 0.5f * viewHeightPx
        return true
    }
}
