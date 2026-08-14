package com.furnit.android.roomreconstruction

import android.graphics.Bitmap
import com.furnit.android.utils.LogUtil
import java.util.ArrayDeque
import kotlin.math.max
import kotlin.math.min

data class LayeredDepthRoomCompletionResult(
    val foregroundMask: ByteArray,
    val completedBackground: Bitmap,
    val completedBackgroundDepth: FloatArray,
    val nearestReliableDepth: Float,
    val representativeRoomDepth: Float,
)

/** Deterministic two-layer completion used only for saved single-photo rooms. */
object LayeredDepthRoomCompletion {
    private const val TAG = "LayeredDepthCompletion"

    fun complete(
        photo: Bitmap,
        depth: FloatArray,
        width: Int,
        height: Int,
        semanticForegroundMask: ByteArray?,
    ): LayeredDepthRoomCompletionResult? {
        require(photo.width == width && photo.height == height && depth.size == width * height)
        val pixelCount = width * height
        val semanticCandidates = ByteArray(pixelCount)
        if (semanticForegroundMask?.size == pixelCount) {
            for (index in semanticCandidates.indices) {
                if (semanticForegroundMask[index].toInt() != 0) semanticCandidates[index] = 1
            }
        }
        val depthCandidates = ByteArray(pixelCount)
        addDepthDiscontinuityForeground(depthCandidates, depth, width, height)
        val foreground = ByteArray(pixelCount)
        filterComponents(semanticCandidates, width, height)?.let { filtered ->
            for (index in foreground.indices) if (filtered[index].toInt() != 0) foreground[index] = 1
        }
        filterComponents(depthCandidates, width, height)?.let { filtered ->
            for (index in foreground.indices) if (filtered[index].toInt() != 0) foreground[index] = 1
        }
        val foregroundFraction = foreground.count { it.toInt() != 0 }.toFloat() / max(pixelCount, 1)
        if (foregroundFraction !in 0.002f..0.42f) return null
        val expanded = dilate(foreground, width, height, max(3, min(width, height) / 180))
        val completedDepth = completeInverseDepth(depth, expanded, width, height) ?: return null
        val completedBitmap = completeColor(photo, expanded, width, height) ?: return null
        val depthSampleStride = max(1, depth.size / 100_000)
        val reliableDepths = depth.indices.asSequence()
            .filter { it % depthSampleStride == 0 }
            .map { depth[it] }
            .filter { it.isFinite() && it > 0.05f }
            .sorted()
            .toList()
        val nearestReliableDepth = reliableDepths.getOrNull((reliableDepths.size * 0.05f).toInt()) ?: 1f
        val representativeRoomDepth = reliableDepths.getOrNull((reliableDepths.size * 0.50f).toInt())
            ?: nearestReliableDepth
        LogUtil.i(
            TAG,
            "layered foreground=${foreground.count { it.toInt() != 0 }}/$pixelCount " +
                "nearestReliableDepth=$nearestReliableDepth representativeRoomDepth=$representativeRoomDepth",
        )
        return LayeredDepthRoomCompletionResult(
            foregroundMask = foreground,
            completedBackground = completedBitmap,
            completedBackgroundDepth = completedDepth,
            nearestReliableDepth = nearestReliableDepth,
            representativeRoomDepth = representativeRoomDepth,
        )
    }

    private fun addDepthDiscontinuityForeground(
        mask: ByteArray,
        depth: FloatArray,
        width: Int,
        height: Int,
    ) {
        val radius = max(6, min(width, height) / 24)
        val horizontalMax = FloatArray(depth.size)
        for (row in 0 until height) {
            val deque = ArrayDeque<Int>()
            var nextColumn = 0
            for (column in 0 until width) {
                val rightEdge = min(width - 1, column + radius)
                while (nextColumn <= rightEdge) {
                    while (deque.isNotEmpty() && depth[row * width + deque.last()] <= depth[row * width + nextColumn]) {
                        deque.removeLast()
                    }
                    deque.addLast(nextColumn)
                    nextColumn++
                }
                while (deque.isNotEmpty() && deque.first() < column - radius) deque.removeFirst()
                horizontalMax[row * width + column] = depth[row * width + deque.first()]
            }
        }
        for (column in 0 until width) {
            val deque = ArrayDeque<Int>()
            var nextRow = 0
            for (row in 0 until height) {
                val bottomEdge = min(height - 1, row + radius)
                while (nextRow <= bottomEdge) {
                    while (deque.isNotEmpty() && horizontalMax[deque.last() * width + column] <= horizontalMax[nextRow * width + column]) {
                        deque.removeLast()
                    }
                    deque.addLast(nextRow)
                    nextRow++
                }
                while (deque.isNotEmpty() && deque.first() < row - radius) deque.removeFirst()
                val index = row * width + column
                val sample = depth[index]
                val localBackground = horizontalMax[deque.first() * width + column]
                if (sample.isFinite() && localBackground.isFinite() && sample > 0.05f &&
                    localBackground - sample > max(0.08f, localBackground * 0.04f)
                ) {
                    mask[index] = 1
                }
            }
        }
    }

    private fun filterComponents(mask: ByteArray, width: Int, height: Int): ByteArray? {
        val pixelCount = width * height
        val visited = BooleanArray(pixelCount)
        val output = ByteArray(pixelCount)
        val queue = IntArray(pixelCount)
        val minimumArea = max(24, pixelCount / 1600)
        val maximumArea = (pixelCount * 0.42f).toInt()
        for (seed in mask.indices) {
            if (mask[seed].toInt() == 0 || visited[seed]) continue
            var head = 0
            var tail = 0
            queue[tail++] = seed
            visited[seed] = true
            var borderMask = 0
            while (head < tail) {
                val index = queue[head++]
                val row = index / width
                val column = index - row * width
                if (column == 0) borderMask = borderMask or 1
                if (column == width - 1) borderMask = borderMask or 2
                if (row == 0) borderMask = borderMask or 4
                if (row == height - 1) borderMask = borderMask or 8
                val neighbors = intArrayOf(
                    if (column > 0) index - 1 else -1,
                    if (column + 1 < width) index + 1 else -1,
                    if (row > 0) index - width else -1,
                    if (row + 1 < height) index + width else -1,
                )
                for (neighbor in neighbors) {
                    if (neighbor >= 0 && mask[neighbor].toInt() != 0 && !visited[neighbor]) {
                        visited[neighbor] = true
                        queue[tail++] = neighbor
                    }
                }
            }
            if (tail !in minimumArea..maximumArea || Integer.bitCount(borderMask) >= 2) continue
            for (offset in 0 until tail) output[queue[offset]] = 1
        }
        val fraction = output.count { it.toInt() != 0 }.toFloat() / max(pixelCount, 1)
        return if (fraction in 0.002f..0.42f) output else null
    }

    private fun dilate(mask: ByteArray, width: Int, height: Int, passes: Int): ByteArray {
        var current = mask.copyOf()
        repeat(passes) {
            val expanded = current.copyOf()
            for (row in 0 until height) for (column in 0 until width) {
                val index = row * width + column
                if (current[index].toInt() == 0) continue
                for (y in max(0, row - 1)..min(height - 1, row + 1)) {
                    for (x in max(0, column - 1)..min(width - 1, column + 1)) expanded[y * width + x] = 1
                }
            }
            current = expanded
        }
        return current
    }

    private fun nearestKnown(mask: ByteArray, width: Int, height: Int): IntArray? {
        val nearest = IntArray(mask.size) { -1 }
        val queue = IntArray(mask.size)
        var head = 0
        var tail = 0
        for (index in mask.indices) if (mask[index].toInt() == 0) {
            nearest[index] = index
            queue[tail++] = index
        }
        if (tail == 0) return null
        while (head < tail) {
            val index = queue[head++]
            val row = index / width
            val column = index - row * width
            val neighbors = intArrayOf(
                if (column > 0) index - 1 else -1,
                if (column + 1 < width) index + 1 else -1,
                if (row > 0) index - width else -1,
                if (row + 1 < height) index + width else -1,
            )
            for (neighbor in neighbors) if (neighbor >= 0 && nearest[neighbor] < 0) {
                nearest[neighbor] = nearest[index]
                queue[tail++] = neighbor
            }
        }
        return nearest
    }

    private fun completeInverseDepth(
        depth: FloatArray,
        mask: ByteArray,
        width: Int,
        height: Int,
    ): FloatArray? {
        val nearest = nearestKnown(mask, width, height) ?: return null
        var inverse = FloatArray(depth.size) { index ->
            val source = if (mask[index].toInt() == 0) index else nearest[index]
            val value = depth[source]
            if (value.isFinite() && value > 0.05f) 1f / value else 0f
        }
        var relaxed = inverse.copyOf()
        repeat(24) {
            for (row in 1 until height - 1) for (column in 1 until width - 1) {
                val index = row * width + column
                if (mask[index].toInt() != 0) {
                    relaxed[index] = (inverse[index - 1] + inverse[index + 1] + inverse[index - width] + inverse[index + width]) * 0.25f
                }
            }
            val swap = inverse
            inverse = relaxed
            relaxed = swap
        }
        return depth.copyOf().also { completed ->
            for (index in completed.indices) if (mask[index].toInt() != 0 && inverse[index] > 1e-5f) {
                completed[index] = 1f / inverse[index]
            }
        }
    }

    private fun completeColor(photo: Bitmap, mask: ByteArray, width: Int, height: Int): Bitmap? {
        val nearest = nearestKnown(mask, width, height) ?: return null
        var pixels = IntArray(mask.size)
        photo.getPixels(pixels, 0, width, 0, 0, width, height)
        for (index in pixels.indices) pixels[index] = pixels[index] or 0xff000000.toInt()
        for (index in pixels.indices) if (mask[index].toInt() != 0) pixels[index] = pixels[nearest[index]]
        var relaxed = pixels.copyOf()
        repeat(8) {
            for (row in 1 until height - 1) for (column in 1 until width - 1) {
                val index = row * width + column
                if (mask[index].toInt() == 0) continue
                val neighbors = intArrayOf(pixels[index - 1], pixels[index + 1], pixels[index - width], pixels[index + width])
                var red = 0; var green = 0; var blue = 0
                for (color in neighbors) {
                    red += (color shr 16) and 0xff
                    green += (color shr 8) and 0xff
                    blue += color and 0xff
                }
                relaxed[index] = 0xff000000.toInt() or ((red / 4) shl 16) or ((green / 4) shl 8) or (blue / 4)
            }
            val swap = pixels
            pixels = relaxed
            relaxed = swap
        }
        return Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888).also {
            it.setPixels(pixels, 0, width, 0, 0, width, height)
        }
    }
}
