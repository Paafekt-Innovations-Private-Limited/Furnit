package com.furnit.android.roomreconstruction

import kotlin.math.sqrt
import kotlin.math.roundToInt

object RoomMath {
    fun median(values: List<Float>): Float {
        if (values.isEmpty()) return 0f
        val sorted = values.sorted()
        val middle = sorted.size / 2
        return if (sorted.size % 2 == 0) {
            (sorted[middle - 1] + sorted[middle]) * 0.5f
        } else {
            sorted[middle]
        }
    }

    fun median(values: FloatArray): Float {
        if (values.isEmpty()) return 0f
        val sorted = values.sorted()
        val middle = sorted.size / 2
        return if (sorted.size % 2 == 0) {
            (sorted[middle - 1] + sorted[middle]) * 0.5f
        } else {
            sorted[middle]
        }
    }

    fun upperMedian(values: List<Float>): Float {
        if (values.isEmpty()) return 0f
        val sorted = values.sorted()
        return sorted[sorted.size / 2]
    }

    fun stdev(values: List<Float>): Float {
        if (values.size <= 1) return 999f
        val mean = values.sum() / values.size
        val variance = values.sumOf { ((it - mean) * (it - mean)).toDouble() }.toFloat() / values.size
        return sqrt(variance)
    }

    fun percentile(sorted: List<Float>, fraction: Double): Float? {
        if (sorted.isEmpty()) return null
        val index = ((sorted.size - 1) * fraction).roundToInt().coerceIn(0, sorted.lastIndex)
        return sorted[index]
    }

    fun percentile(sorted: FloatArray, fraction: Double): Float? {
        if (sorted.isEmpty()) return null
        val copy = sorted.sorted()
        val index = ((copy.size - 1) * fraction).roundToInt().coerceIn(0, copy.lastIndex)
        return copy[index]
    }

    fun scaleDepthFlat(depth: FloatArray, scale: Float): FloatArray {
        if (kotlin.math.abs(scale - 1f) <= 1e-4f) return depth
        return FloatArray(depth.size) { index ->
            val value = depth[index]
            if (value.isFinite() && value > 0f) value * scale else value
        }
    }

    fun depthToRows(depth: FloatArray, width: Int, height: Int): Array<FloatArray> {
        require(depth.size == width * height)
        return Array(height) { row ->
            depth.copyOfRange(row * width, row * width + width)
        }
    }
}
