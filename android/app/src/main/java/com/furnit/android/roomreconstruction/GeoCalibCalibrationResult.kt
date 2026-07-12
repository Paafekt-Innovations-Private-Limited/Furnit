package com.furnit.android.roomreconstruction

import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

data class GeoCalibCalibrationResult(
    val focalLengthXPixels: Float,
    val focalLengthYPixels: Float,
    val rollRadians: Float,
    val pitchRadians: Float,
    val finalCost: Float,
    val iterations: Int,
    val sourceWidth: Int,
    val sourceHeight: Int,
) {
    fun levelingRotationMatrix(): Mat3 = levelingRotationMatrix(gravityVectorCamera())

    fun gravityVectorCamera(): Vec3 = gravityVector(rollRadians, pitchRadians)

    companion object {
        fun gravityVector(rollRadians: Float, pitchRadians: Float): Vec3 {
            val sr = sin(rollRadians)
            val cr = cos(rollRadians)
            val sp = sin(pitchRadians)
            val cp = cos(pitchRadians)
            return Vec3(-sr * cp, -cr * cp, sp).normalized()
        }

        fun levelingRotationMatrix(gravityDown: Vec3): Mat3 {
            return Mat3.rotationFromTo(gravityDown.normalized(), Vec3(0f, -1f, 0f))
        }
    }
}
