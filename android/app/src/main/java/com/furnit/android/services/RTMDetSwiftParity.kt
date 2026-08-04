package com.furnit.android.services

import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sqrt

/** Pure RTMDet post-processing rules shared with the Swift implementation. */
internal object RTMDetSwiftParity {
    const val MODEL_SIDE = 640
    const val SOURCE_MASK_SIDE = 80
    const val MASK_SIDE = 160
    const val CONFIDENCE_THRESHOLD = 0.30f
    const val NMS_IOU_THRESHOLD = 0.50f
    const val MAX_DETECTION_COUNT = 200
    const val MASK_AFFINITY_THRESHOLD = 0.12f
    const val MASK_AFFINITY_BIT_THRESHOLD = 0.50f
    const val MASK_RENDER_THRESHOLD = 0.30f
    const val MASK_RENDER_ANTIALIAS_HALF_WIDTH = 0.05f

    data class Box(
        val x: Float,
        val y: Float,
        val width: Float,
        val height: Float,
        val confidence: Float,
        val classId: Int,
    ) {
        val area: Float
            get() = max(0f, width) * max(0f, height)
    }

    data class ClusterSelection(
        val representativeIndex: Int,
        val memberIndices: List<Int>,
    )

    /** Inclusive pixel bounds, matching Swift's closed-range compositor loops. */
    data class PixelBounds(
        val minX: Int,
        val minY: Int,
        val maxX: Int,
        val maxY: Int,
    )

    fun rawMaskRenderAlpha(probability: Float): Int {
        if (!probability.isFinite()) return 0
        val lower = MASK_RENDER_THRESHOLD - MASK_RENDER_ANTIALIAS_HALF_WIDTH
        val upper = MASK_RENDER_THRESHOLD + MASK_RENDER_ANTIALIAS_HALF_WIDTH
        return when {
            probability <= lower -> 0
            probability >= upper -> 255
            else -> (((probability - lower) / (upper - lower)) * 255f).roundToInt()
        }
    }

    /**
     * Android's raw ONNX export exposes the native 8x80x80 feature map. Swift's Core ML wrapper
     * inserts `interpolate(scale_factor: 2, mode: bilinear, align_corners: false)`, so reproduce
     * that node before running the shared dynamic mask head.
     */
    fun upsampleMaskFeaturesAlignCornersFalse(
        source: FloatArray,
        channels: Int = 8,
        sourceSide: Int = SOURCE_MASK_SIDE,
        targetSide: Int = MASK_SIDE,
    ): FloatArray {
        require(channels > 0 && sourceSide > 0 && targetSide > 0)
        require(source.size >= channels * sourceSide * sourceSide)
        if (sourceSide == targetSide) {
            return source.copyOf(channels * sourceSide * sourceSide)
        }

        val sourcePixels = sourceSide * sourceSide
        val targetPixels = targetSide * targetSide
        val output = FloatArray(channels * targetPixels)
        val lower = IntArray(targetSide)
        val upper = IntArray(targetSide)
        val weight = FloatArray(targetSide)
        val scale = sourceSide.toFloat() / targetSide.toFloat()
        for (target in 0 until targetSide) {
            val coordinate = ((target + 0.5f) * scale - 0.5f).coerceIn(0f, (sourceSide - 1).toFloat())
            val low = floor(coordinate.toDouble()).toInt()
            lower[target] = low
            upper[target] = min(sourceSide - 1, low + 1)
            weight[target] = coordinate - low
        }

        for (channel in 0 until channels) {
            val sourceChannel = channel * sourcePixels
            val targetChannel = channel * targetPixels
            for (y in 0 until targetSide) {
                val topRow = sourceChannel + lower[y] * sourceSide
                val bottomRow = sourceChannel + upper[y] * sourceSide
                val wy = weight[y]
                val outputRow = targetChannel + y * targetSide
                for (x in 0 until targetSide) {
                    val x0 = lower[x]
                    val x1 = upper[x]
                    val wx = weight[x]
                    val top = source[topRow + x0] + (source[topRow + x1] - source[topRow + x0]) * wx
                    val bottom = source[bottomRow + x0] + (source[bottomRow + x1] - source[bottomRow + x0]) * wx
                    output[outputRow + x] = top + (bottom - top) * wy
                }
            }
        }
        return output
    }

    fun classAwareNms(
        candidates: List<Box>,
        iouThreshold: Float = NMS_IOU_THRESHOLD,
        limit: Int = MAX_DETECTION_COUNT,
    ): List<Int> {
        if (candidates.isEmpty()) return emptyList()
        val sortedIndices = candidates.indices.sortedWith { leftIndex, rightIndex ->
            val left = candidates[leftIndex]
            val right = candidates[rightIndex]
            when {
                abs(left.confidence - right.confidence) > 1e-6f ->
                    right.confidence.compareTo(left.confidence)
                left.area != right.area -> left.area.compareTo(right.area)
                else -> leftIndex.compareTo(rightIndex)
            }
        }
        val kept = ArrayList<Int>(min(limit.coerceAtLeast(1), candidates.size))
        for (candidateIndex in sortedIndices) {
            val candidate = candidates[candidateIndex]
            val suppressed = kept.any { keptIndex ->
                val existing = candidates[keptIndex]
                existing.classId == candidate.classId && iou(existing, candidate) > iouThreshold
            }
            if (!suppressed) {
                kept += candidateIndex
                if (kept.size >= limit.coerceAtLeast(1)) break
            }
        }
        return kept
    }

    fun selectDefaultCluster(
        candidates: List<Box>,
        clusters: List<List<Int>>,
        frameWidth: Float,
        frameHeight: Float,
        preferCenter: Boolean = true,
        confidenceFloor: Float = CONFIDENCE_THRESHOLD,
    ): ClusterSelection? {
        if (candidates.isEmpty()) return null
        val usableClusters = if (clusters.isEmpty()) candidates.indices.map { listOf(it) } else clusters
        var best: RankedCluster? = null
        for (members in usableClusters) {
            val validMembers = members.filter { it in candidates.indices }.distinct().sorted()
            if (validMembers.isEmpty()) continue
            val representative = validMembers.reduce { current, challenger ->
                if (isBetterRepresentative(challenger, current, candidates)) challenger else current
            }
            var minX = Float.POSITIVE_INFINITY
            var minY = Float.POSITIVE_INFINITY
            var maxX = Float.NEGATIVE_INFINITY
            var maxY = Float.NEGATIVE_INFINITY
            for (index in validMembers) {
                val box = candidates[index]
                minX = min(minX, box.x - box.width * 0.5f)
                minY = min(minY, box.y - box.height * 0.5f)
                maxX = max(maxX, box.x + box.width * 0.5f)
                maxY = max(maxY, box.y + box.height * 0.5f)
            }
            val width = maxX - minX
            val height = maxY - minY
            if (!width.isFinite() || !height.isFinite() || width <= 0f || height <= 0f) continue
            val centerX = max(1f, frameWidth) * 0.5f
            val centerY = max(1f, frameHeight) * 0.5f
            val dx = ((minX + maxX) * 0.5f - centerX) / max(centerX, 1f)
            val dy = ((minY + maxY) * 0.5f - centerY) / max(centerY, 1f)
            val ranked = RankedCluster(
                representative = representative,
                members = validMembers,
                confidence = candidates[representative].confidence,
                centerDistance = sqrt(dx * dx + dy * dy),
                area = width * height,
            )
            if (best == null || isBetterCluster(ranked, best, preferCenter, confidenceFloor)) {
                best = ranked
            }
        }
        return best?.let { ClusterSelection(it.representative, it.members) }
    }

    fun paddedSourceBounds(
        box: Box,
        modelWidth: Float,
        modelHeight: Float,
        sourceWidth: Int,
        sourceHeight: Int,
    ): PixelBounds? {
        if (sourceWidth <= 0 || sourceHeight <= 0 || modelWidth <= 0f || modelHeight <= 0f) return null
        val scaleX = sourceWidth.toFloat() / modelWidth
        val scaleY = sourceHeight.toFloat() / modelHeight
        val x1 = (box.x - box.width * 0.5f) * scaleX
        val y1 = (box.y - box.height * 0.5f) * scaleY
        val x2 = (box.x + box.width * 0.5f) * scaleX
        val y2 = (box.y + box.height * 0.5f) * scaleY
        val boxWidth = max(0f, x2 - x1)
        val boxHeight = max(0f, y2 - y1)
        val padX = ceil(boxWidth * 0.20f).toInt()
        val padTop = ceil(boxHeight * 1.00f).toInt()
        val padBottom = ceil(boxHeight * 0.25f).toInt()
        val minX = (floor(x1.toDouble()).toInt() - padX).coerceIn(0, sourceWidth - 1)
        val minY = (floor(y1.toDouble()).toInt() - padTop).coerceIn(0, sourceHeight - 1)
        val maxX = (ceil(x2.toDouble()).toInt() + padX).coerceIn(0, sourceWidth - 1)
        val maxY = (ceil(y2.toDouble()).toInt() + padBottom).coerceIn(0, sourceHeight - 1)
        return PixelBounds(minX, minY, maxX, maxY).takeIf { it.maxX >= it.minX && it.maxY >= it.minY }
    }

    private data class RankedCluster(
        val representative: Int,
        val members: List<Int>,
        val confidence: Float,
        val centerDistance: Float,
        val area: Float,
    )

    private fun isBetterRepresentative(challenger: Int, current: Int, candidates: List<Box>): Boolean {
        val left = candidates[challenger]
        val right = candidates[current]
        if (abs(left.confidence - right.confidence) > 1e-6f) return left.confidence > right.confidence
        if (abs(left.area - right.area) > 1e-6f) return left.area > right.area
        return challenger < current
    }

    private fun isBetterCluster(
        challenger: RankedCluster,
        current: RankedCluster,
        preferCenter: Boolean,
        confidenceFloor: Float,
    ): Boolean {
        if (preferCenter) {
            val challengerEligible = challenger.confidence >= confidenceFloor
            val currentEligible = current.confidence >= confidenceFloor
            if (challengerEligible != currentEligible) return challengerEligible
            if (challengerEligible && abs(challenger.centerDistance - current.centerDistance) > 0.08f) {
                return challenger.centerDistance < current.centerDistance
            }
        }
        if (abs(challenger.confidence - current.confidence) > 1e-6f) {
            return challenger.confidence > current.confidence
        }
        if (abs(challenger.area - current.area) > 1f) return challenger.area > current.area
        return challenger.representative < current.representative
    }

    private fun iou(left: Box, right: Box): Float {
        val intersectionWidth = max(
            0f,
            min(left.x + left.width * 0.5f, right.x + right.width * 0.5f) -
                max(left.x - left.width * 0.5f, right.x - right.width * 0.5f),
        )
        val intersectionHeight = max(
            0f,
            min(left.y + left.height * 0.5f, right.y + right.height * 0.5f) -
                max(left.y - left.height * 0.5f, right.y - right.height * 0.5f),
        )
        val intersection = intersectionWidth * intersectionHeight
        val union = left.area + right.area - intersection
        return if (union > 0f) intersection / union else 0f
    }
}
