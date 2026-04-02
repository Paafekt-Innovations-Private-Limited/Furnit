package com.furnit.android.services

import com.furnit.android.ar.MetricAnchor
import com.furnit.android.utils.LogUtil
import java.util.Arrays
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sqrt

object MetricScaleEstimator {

    data class SharpMonodepthInfo(
        val width: Int,
        val height: Int,
        val channels: Int,
    )

    data class EstimationResult(
        val scale: Float,
        val isValid: Boolean,
        val fallbackReason: String?,
        val survivingAnchors: Int,
        val coefficientOfVariation: Float,
        val arcoreMedianDepthMeters: Float,
        val sharpMedianDepthUnits: Float,
        val rawMedianRatio: Float,
        val monodepthWidth: Int,
        val monodepthHeight: Int,
        val monodepthChannels: Int,
    )

    fun estimateFromGaussianDepthProxy(
        anchors: List<MetricAnchor>,
        gaussianDepthUnits: Float,
        minScale: Float = 0.2f,
        maxScale: Float = 8.0f,
    ): EstimationResult {
        if (!gaussianDepthUnits.isFinite() || gaussianDepthUnits <= 0f) {
            return EstimationResult(
                scale = 1f,
                isValid = false,
                fallbackReason = "invalid_gaussian_depth_proxy",
                survivingAnchors = 0,
                coefficientOfVariation = Float.POSITIVE_INFINITY,
                arcoreMedianDepthMeters = Float.NaN,
                sharpMedianDepthUnits = gaussianDepthUnits,
                rawMedianRatio = Float.NaN,
                monodepthWidth = 0,
                monodepthHeight = 0,
                monodepthChannels = 0,
            )
        }

        val anchorDepths = FloatArray(anchors.size)
        val anchorWeights = FloatArray(anchors.size)
        var validCount = 0
        for (anchor in anchors) {
            if (!anchor.depthMeters.isFinite() || anchor.depthMeters !in 0.2f..10f) continue
            if (!anchor.confidence.isFinite() || anchor.confidence <= 0f) continue
            anchorDepths[validCount] = anchor.depthMeters
            anchorWeights[validCount] = anchor.confidence.coerceAtLeast(0.05f)
            validCount++
        }
        if (validCount < 3) {
            return EstimationResult(
                scale = 1f,
                isValid = false,
                fallbackReason = "insufficient_arcore_anchors",
                survivingAnchors = validCount,
                coefficientOfVariation = Float.POSITIVE_INFINITY,
                arcoreMedianDepthMeters = Float.NaN,
                sharpMedianDepthUnits = gaussianDepthUnits,
                rawMedianRatio = Float.NaN,
                monodepthWidth = 0,
                monodepthHeight = 0,
                monodepthChannels = 0,
            )
        }

        val ratios = FloatArray(validCount)
        for (i in 0 until validCount) {
            ratios[i] = anchorDepths[i] / gaussianDepthUnits
        }
        val rawMedianRatio = median(ratios, validCount)

        val filteredRatios = FloatArray(validCount)
        val filteredWeights = FloatArray(validCount)
        val filteredDepths = FloatArray(validCount)
        var filteredCount = 0
        var sumRatio = 0f
        var sumSqRatio = 0f
        for (i in 0 until validCount) {
            val ratio = ratios[i]
            if (!ratio.isFinite() || ratio <= 0f) continue
            if (ratio < rawMedianRatio * 0.5f || ratio > rawMedianRatio * 2.0f) continue
            filteredRatios[filteredCount] = ratio
            filteredWeights[filteredCount] = anchorWeights[i]
            filteredDepths[filteredCount] = anchorDepths[i]
            filteredCount++
            sumRatio += ratio
            sumSqRatio += ratio * ratio
        }
        if (filteredCount < 3) {
            return EstimationResult(
                scale = 1f,
                isValid = false,
                fallbackReason = "gaussian_proxy_outlier_rejection_left_too_few",
                survivingAnchors = filteredCount,
                coefficientOfVariation = Float.POSITIVE_INFINITY,
                arcoreMedianDepthMeters = median(anchorDepths, validCount),
                sharpMedianDepthUnits = gaussianDepthUnits,
                rawMedianRatio = rawMedianRatio,
                monodepthWidth = 0,
                monodepthHeight = 0,
                monodepthChannels = 0,
            )
        }

        val weightedScale = weightedMedian(filteredRatios, filteredWeights, filteredCount)
        val meanRatio = sumRatio / filteredCount.toFloat()
        val variance = max(0f, sumSqRatio / filteredCount.toFloat() - meanRatio * meanRatio)
        val cv = if (meanRatio > 1e-6f) sqrt(variance) / meanRatio else Float.POSITIVE_INFINITY
        LogUtil.i(
            "SHARP_METRIC_SCALE",
            "[proxy] anchors=${anchors.size} filtered=$filteredCount gaussianDepth=$gaussianDepthUnits rawMedianRatio=$rawMedianRatio weightedScale=$weightedScale cv=$cv",
        )
        val valid = weightedScale.isFinite() && weightedScale in minScale..maxScale && cv.isFinite() && cv <= 0.5f
        return EstimationResult(
            scale = if (valid) weightedScale else 1f,
            isValid = valid,
            fallbackReason = if (valid) null else if (!cv.isFinite() || cv > 0.5f) "gaussian_proxy_ratio_variance_too_high" else "gaussian_proxy_scale_out_of_range",
            survivingAnchors = filteredCount,
            coefficientOfVariation = cv,
            arcoreMedianDepthMeters = median(filteredDepths, filteredCount),
            sharpMedianDepthUnits = gaussianDepthUnits,
            rawMedianRatio = rawMedianRatio,
            monodepthWidth = 0,
            monodepthHeight = 0,
            monodepthChannels = 0,
        )
    }

    fun estimateFromMatchedMonodepth(
        anchors: List<MetricAnchor>,
        monodepth: SharpMonodepthInfo?,
        sampleMonodepthChannel: (IntArray, IntArray, Int) -> FloatArray?,
        minScale: Float = 0.2f,
        maxScale: Float = 8.0f,
    ): EstimationResult {
        if (monodepth == null ||
            monodepth.width <= 0 ||
            monodepth.height <= 0 ||
            monodepth.channels <= 0
        ) {
            return EstimationResult(
                scale = 1f,
                isValid = false,
                fallbackReason = "missing_monodepth_buffer",
                survivingAnchors = 0,
                coefficientOfVariation = Float.POSITIVE_INFINITY,
                arcoreMedianDepthMeters = Float.NaN,
                sharpMedianDepthUnits = Float.NaN,
                rawMedianRatio = Float.NaN,
                monodepthWidth = monodepth?.width ?: 0,
                monodepthHeight = monodepth?.height ?: 0,
                monodepthChannels = monodepth?.channels ?: 0,
            )
        }

        val validAnchors = ArrayList<MetricAnchor>(anchors.size)
        for (anchor in anchors) {
            if (!anchor.depthMeters.isFinite() || anchor.depthMeters !in 0.2f..10f) continue
            if (!anchor.confidence.isFinite() || anchor.confidence <= 0f) continue
            if (anchor.originalImageWidth <= 0 || anchor.originalImageHeight <= 0) continue
            validAnchors += anchor
        }
        if (validAnchors.size < 5) {
            return EstimationResult(
                scale = 1f,
                isValid = false,
                fallbackReason = "insufficient_arcore_anchors",
                survivingAnchors = validAnchors.size,
                coefficientOfVariation = Float.POSITIVE_INFINITY,
                arcoreMedianDepthMeters = Float.NaN,
                sharpMedianDepthUnits = Float.NaN,
                rawMedianRatio = Float.NaN,
                monodepthWidth = monodepth.width,
                monodepthHeight = monodepth.height,
                monodepthChannels = monodepth.channels,
            )
        }

        logMonodepthDiagnostics(monodepth, sampleMonodepthChannel)

        val mappedXs = IntArray(validAnchors.size)
        val mappedYs = IntArray(validAnchors.size)
        for (i in validAnchors.indices) {
            val anchor = validAnchors[i]
            mappedXs[i] = mapPixel(anchor.pixelX, anchor.originalImageWidth, monodepth.width)
            mappedYs[i] = mapPixel(anchor.pixelY, anchor.originalImageHeight, monodepth.height)
        }

        val sampledDepths = sampleNearestSurfaceDepths(monodepth, mappedXs, mappedYs, sampleMonodepthChannel)
            ?: return EstimationResult(
                scale = 1f,
                isValid = false,
                fallbackReason = "monodepth_sampling_failed",
                survivingAnchors = 0,
                coefficientOfVariation = Float.POSITIVE_INFINITY,
                arcoreMedianDepthMeters = Float.NaN,
                sharpMedianDepthUnits = Float.NaN,
                rawMedianRatio = Float.NaN,
                monodepthWidth = monodepth.width,
                monodepthHeight = monodepth.height,
                monodepthChannels = monodepth.channels,
            )

        val pairingIndices = IntArray(validAnchors.size)
        val pairingRatios = FloatArray(validAnchors.size)
        val pairingSharpDepths = FloatArray(validAnchors.size)
        var pairingCount = 0
        for (i in validAnchors.indices) {
            val sharpDepth = sampledDepths[i]
            if (!sharpDepth.isFinite() || sharpDepth <= 0f) continue
            val ratio = validAnchors[i].depthMeters / sharpDepth
            if (!ratio.isFinite() || ratio <= 0f) continue
            pairingIndices[pairingCount] = i
            pairingRatios[pairingCount] = ratio
            pairingSharpDepths[pairingCount] = sharpDepth
            pairingCount++
        }

        val logCount = min(10, pairingCount)
        for (index in 0 until logCount) {
            val pairingIndex = pairingIndices[index]
            val anchor = validAnchors[pairingIndex]
            LogUtil.i(
                "SHARP_METRIC_SCALE",
                "[pairing] #$index anchor=(${anchor.pixelX},${anchor.pixelY}) " +
                    "arcore=${anchor.depthMeters}m conf=${anchor.confidence} " +
                    "sharp=(${mappedXs[pairingIndex]},${mappedYs[pairingIndex]}) depth=${pairingSharpDepths[index]} " +
                    "ratio=${pairingRatios[index]}",
            )
        }

        if (pairingCount < 5) {
            return EstimationResult(
                scale = 1f,
                isValid = false,
                fallbackReason = "insufficient_monodepth_pairings",
                survivingAnchors = pairingCount,
                coefficientOfVariation = Float.POSITIVE_INFINITY,
                arcoreMedianDepthMeters = Float.NaN,
                sharpMedianDepthUnits = Float.NaN,
                rawMedianRatio = Float.NaN,
                monodepthWidth = monodepth.width,
                monodepthHeight = monodepth.height,
                monodepthChannels = monodepth.channels,
            )
        }

        val rawMedianRatio = median(pairingRatios, pairingCount)
        val filteredRatios = FloatArray(pairingCount)
        val filteredWeights = FloatArray(pairingCount)
        val filteredArDepths = FloatArray(pairingCount)
        val filteredSharpDepths = FloatArray(pairingCount)
        var filteredCount = 0
        var sumRatio = 0f
        var sumSqRatio = 0f
        for (i in 0 until pairingCount) {
            val ratio = pairingRatios[i]
            if (ratio < rawMedianRatio * 0.5f || ratio > rawMedianRatio * 2.0f) continue
            val anchor = validAnchors[pairingIndices[i]]
            filteredRatios[filteredCount] = ratio
            filteredWeights[filteredCount] = anchor.confidence.coerceAtLeast(0.05f)
            filteredArDepths[filteredCount] = anchor.depthMeters
            filteredSharpDepths[filteredCount] = pairingSharpDepths[i]
            filteredCount++
            sumRatio += ratio
            sumSqRatio += ratio * ratio
        }
        if (filteredCount < 5) {
            return EstimationResult(
                scale = 1f,
                isValid = false,
                fallbackReason = "ratio_outlier_rejection_left_too_few",
                survivingAnchors = filteredCount,
                coefficientOfVariation = Float.POSITIVE_INFINITY,
                arcoreMedianDepthMeters = medianPairingAnchorDepths(validAnchors, pairingIndices, pairingCount),
                sharpMedianDepthUnits = median(pairingSharpDepths, pairingCount),
                rawMedianRatio = rawMedianRatio,
                monodepthWidth = monodepth.width,
                monodepthHeight = monodepth.height,
                monodepthChannels = monodepth.channels,
            )
        }

        val weightedScale = weightedMedian(filteredRatios, filteredWeights, filteredCount)
        val meanRatio = sumRatio / filteredCount.toFloat()
        val variance = max(0f, sumSqRatio / filteredCount.toFloat() - meanRatio * meanRatio)
        val cv = if (meanRatio > 1e-6f) sqrt(variance) / meanRatio else Float.POSITIVE_INFINITY

        LogUtil.i(
            "SHARP_METRIC_SCALE",
            "[estimate] anchors=${anchors.size} validAnchors=${validAnchors.size} pairings=$pairingCount " +
                "filtered=$filteredCount rawMedianRatio=$rawMedianRatio weightedScale=$weightedScale cv=$cv",
        )

        if (!weightedScale.isFinite() || weightedScale !in minScale..maxScale) {
            return EstimationResult(
                scale = 1f,
                isValid = false,
                fallbackReason = "scale_out_of_range",
                survivingAnchors = filteredCount,
                coefficientOfVariation = cv,
                arcoreMedianDepthMeters = median(filteredArDepths, filteredCount),
                sharpMedianDepthUnits = median(filteredSharpDepths, filteredCount),
                rawMedianRatio = rawMedianRatio,
                monodepthWidth = monodepth.width,
                monodepthHeight = monodepth.height,
                monodepthChannels = monodepth.channels,
            )
        }
        if (!cv.isFinite() || cv > 0.5f) {
            return EstimationResult(
                scale = 1f,
                isValid = false,
                fallbackReason = "ratio_variance_too_high",
                survivingAnchors = filteredCount,
                coefficientOfVariation = cv,
                arcoreMedianDepthMeters = median(filteredArDepths, filteredCount),
                sharpMedianDepthUnits = median(filteredSharpDepths, filteredCount),
                rawMedianRatio = rawMedianRatio,
                monodepthWidth = monodepth.width,
                monodepthHeight = monodepth.height,
                monodepthChannels = monodepth.channels,
            )
        }

        return EstimationResult(
            scale = weightedScale,
            isValid = true,
            fallbackReason = null,
            survivingAnchors = filteredCount,
            coefficientOfVariation = cv,
            arcoreMedianDepthMeters = median(filteredArDepths, filteredCount),
            sharpMedianDepthUnits = median(filteredSharpDepths, filteredCount),
            rawMedianRatio = rawMedianRatio,
            monodepthWidth = monodepth.width,
            monodepthHeight = monodepth.height,
            monodepthChannels = monodepth.channels,
        )
    }

    private fun logMonodepthDiagnostics(
        monodepth: SharpMonodepthInfo,
        sampleMonodepthChannel: (IntArray, IntArray, Int) -> FloatArray?,
    ) {
        val sampleGrid = 9
        val sampleCount = sampleGrid * sampleGrid
        val xs = IntArray(sampleCount)
        val ys = IntArray(sampleCount)
        var index = 0
        for (gy in 0 until sampleGrid) {
            val y = ((gy.toFloat() / (sampleGrid - 1).coerceAtLeast(1)) * (monodepth.height - 1)).roundToInt()
            for (gx in 0 until sampleGrid) {
                xs[index] = ((gx.toFloat() / (sampleGrid - 1).coerceAtLeast(1)) * (monodepth.width - 1)).roundToInt()
                ys[index] = y
                index++
            }
        }
        val depths = sampleNearestSurfaceDepths(monodepth, xs, ys, sampleMonodepthChannel)
        var minDepth = Float.POSITIVE_INFINITY
        var maxDepth = Float.NEGATIVE_INFINITY
        var sumDepth = 0.0
        var count = 0
        if (depths != null) {
            for (value in depths) {
                if (!value.isFinite() || value <= 0f) continue
                minDepth = min(minDepth, value)
                maxDepth = max(maxDepth, value)
                sumDepth += value.toDouble()
                count++
            }
        }
        val centerX = monodepth.width / 2
        val centerY = monodepth.height / 2
        val cornerX = (monodepth.width * 0.1f).roundToInt().coerceIn(0, monodepth.width - 1)
        val cornerY = (monodepth.height * 0.1f).roundToInt().coerceIn(0, monodepth.height - 1)
        val probeXs = intArrayOf(centerX, cornerX)
        val probeYs = intArrayOf(centerY, cornerY)
        val probeDepths = sampleNearestSurfaceDepths(monodepth, probeXs, probeYs, sampleMonodepthChannel)
        LogUtil.i(
            "SHARP_METRIC_SCALE",
            "[monodepth] size=${monodepth.width}x${monodepth.height}x${monodepth.channels} " +
                "min=$minDepth max=$maxDepth mean=${if (count > 0) (sumDepth / count).toFloat() else Float.NaN} " +
                "center($centerX,$centerY)=${probeDepths?.getOrNull(0)} corner($cornerX,$cornerY)=${probeDepths?.getOrNull(1)}",
        )
    }

    private fun sampleNearestSurfaceDepths(
        monodepth: SharpMonodepthInfo,
        xs: IntArray,
        ys: IntArray,
        sampleMonodepthChannel: (IntArray, IntArray, Int) -> FloatArray?,
    ): FloatArray? {
        if (xs.size != ys.size) return null
        val best = FloatArray(xs.size) { Float.NaN }
        for (channel in 0 until monodepth.channels) {
            val values = sampleMonodepthChannel(xs, ys, channel) ?: return null
            if (values.size != xs.size) return null
            for (i in values.indices) {
                val value = values[i]
                if (!value.isFinite() || value <= 0f) continue
                if (!best[i].isFinite() || value < best[i]) {
                    best[i] = value
                }
            }
        }
        return best
    }

    private fun medianPairingAnchorDepths(
        validAnchors: List<MetricAnchor>,
        pairingIndices: IntArray,
        pairingCount: Int,
    ): Float {
        val values = FloatArray(pairingCount)
        for (i in 0 until pairingCount) {
            values[i] = validAnchors[pairingIndices[i]].depthMeters
        }
        return median(values, pairingCount)
    }

    private fun mapPixel(pixel: Int, sourceSize: Int, targetSize: Int): Int {
        val denom = max(sourceSize - 1, 1)
        val normalized = pixel.toFloat() / denom.toFloat()
        return (normalized * (targetSize - 1)).roundToInt().coerceIn(0, targetSize - 1)
    }

    private fun median(values: FloatArray, count: Int): Float {
        if (count <= 0) return Float.NaN
        val sorted = values.copyOf(count)
        Arrays.sort(sorted)
        return sorted[count / 2]
    }

    private fun weightedMedian(values: FloatArray, weights: FloatArray, count: Int): Float {
        if (count <= 0) return Float.NaN
        val order = IntArray(count) { it }
        for (i in 1 until count) {
            val currentIndex = order[i]
            val currentValue = values[currentIndex]
            var j = i - 1
            while (j >= 0 && values[order[j]] > currentValue) {
                order[j + 1] = order[j]
                j--
            }
            order[j + 1] = currentIndex
        }
        var totalWeight = 0f
        for (i in 0 until count) {
            totalWeight += weights[i]
        }
        var cumulativeWeight = 0f
        for (i in 0 until count) {
            val index = order[i]
            cumulativeWeight += weights[index]
            if (cumulativeWeight >= totalWeight * 0.5f) {
                return values[index]
            }
        }
        return values[order[count - 1]]
    }
}
