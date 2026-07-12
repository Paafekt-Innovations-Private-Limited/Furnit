package com.furnit.android.roomreconstruction

import kotlin.math.sqrt

class LeveledDepthPointGrid(
    depth: FloatArray,
    val width: Int,
    val height: Int,
    fx: Float,
    fy: Float,
    cx: Float,
    cy: Float,
    rotation: Mat3,
) {
    val viewDirectionHorizontal: Pair<Float, Float>?
    private val points: Array<Vec3>
    private val valid: BooleanArray

    init {
        val pixelCount = (width * height).coerceAtLeast(0)
        points = Array(pixelCount) { Vec3(0f, 0f, 0f) }
        valid = BooleanArray(pixelCount)

        if (width > 1 && height > 1 && depth.size == pixelCount && fx > 1f && fy > 1f) {
            for (y in 0 until height) {
                val cameraYUnit = (y - cy) / fy
                for (x in 0 until width) {
                    val index = y * width + x
                    val d = depth[index]
                    if (!d.isFinite() || d <= 0f) continue
                    val cameraPoint = Vec3(
                        (x - cx) * d / fx,
                        cameraYUnit * d,
                        d,
                    )
                    points[index] = rotation * cameraPoint
                    valid[index] = true
                }
            }
        }

        val view = rotation * Vec3(0f, 0f, 1f)
        val horizontalLength = sqrt(view.x * view.x + view.z * view.z)
        viewDirectionHorizontal = if (horizontalLength > 1e-4f) {
            view.x / horizontalLength to view.z / horizontalLength
        } else {
            null
        }
    }

    fun point(x: Int, y: Int, scale: Float = 1f): Vec3? {
        if (x < 0 || x >= width || y < 0 || y >= height) return null
        val index = y * width + x
        if (index >= valid.size || !valid[index]) return null
        val point = points[index]
        return if (kotlin.math.abs(scale - 1f) > 1e-6f) point * scale else point
    }
}
