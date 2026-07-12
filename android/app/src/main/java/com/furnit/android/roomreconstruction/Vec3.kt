package com.furnit.android.roomreconstruction

import kotlin.math.sqrt

data class Vec3(val x: Float, val y: Float, val z: Float) {
    fun length(): Float = sqrt(x * x + y * y + z * z)

    fun normalized(): Vec3 {
        val length = length()
        if (length <= 1e-6f) return this
        return Vec3(x / length, y / length, z / length)
    }

    operator fun minus(other: Vec3): Vec3 = Vec3(x - other.x, y - other.y, z - other.z)

    operator fun plus(other: Vec3): Vec3 = Vec3(x + other.x, y + other.y, z + other.z)

    operator fun times(scale: Float): Vec3 = Vec3(x * scale, y * scale, z * scale)

    companion object {
        fun cross(a: Vec3, b: Vec3): Vec3 {
            return Vec3(
                a.y * b.z - a.z * b.y,
                a.z * b.x - a.x * b.z,
                a.x * b.y - a.y * b.x,
            )
        }

        fun dot(a: Vec3, b: Vec3): Float = a.x * b.x + a.y * b.y + a.z * b.z
    }
}

data class Mat3(val m: FloatArray) {
    init {
        require(m.size == 9)
    }

    operator fun times(v: Vec3): Vec3 {
        return Vec3(
            m[0] * v.x + m[1] * v.y + m[2] * v.z,
            m[3] * v.x + m[4] * v.y + m[5] * v.z,
            m[6] * v.x + m[7] * v.y + m[8] * v.z,
        )
    }

    companion object {
        fun identity(): Mat3 = Mat3(
            floatArrayOf(
                1f, 0f, 0f,
                0f, 1f, 0f,
                0f, 0f, 1f,
            ),
        )

        fun rotationFromTo(source: Vec3, target: Vec3): Mat3 {
            val src = source.normalized()
            val tgt = target.normalized()
            val cosine = Vec3.dot(src, tgt).coerceIn(-1f, 1f)
            if (cosine > 0.9999f) return identity()
            if (cosine < -0.9999f) {
                return Mat3(
                    floatArrayOf(
                        1f, 0f, 0f,
                        0f, -1f, 0f,
                        0f, 0f, -1f,
                    ),
                )
            }
            val axis = Vec3.cross(src, tgt).normalized()
            val angle = kotlin.math.acos(cosine.toDouble()).toFloat()
            val c = kotlin.math.cos(angle.toDouble()).toFloat()
            val s = kotlin.math.sin(angle.toDouble()).toFloat()
            val t = 1f - c
            val x = axis.x
            val y = axis.y
            val z = axis.z
            return Mat3(
                floatArrayOf(
                    t * x * x + c, t * x * y - s * z, t * x * z + s * y,
                    t * x * y + s * z, t * y * y + c, t * y * z - s * x,
                    t * x * z - s * y, t * y * z + s * x, t * z * z + c,
                ),
            )
        }
    }
}
