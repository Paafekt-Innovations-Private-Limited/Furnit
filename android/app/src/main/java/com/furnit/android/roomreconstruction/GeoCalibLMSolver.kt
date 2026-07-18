package com.furnit.android.roomreconstruction

import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.sin
import kotlin.math.sqrt
import kotlin.math.tan

object GeoCalibLMSolver {
    data class Fields(
        val width: Int,
        val height: Int,
        val upX: FloatArray,
        val upY: FloatArray,
        val latitudeRadians: FloatArray,
        val upConfidence: FloatArray,
        val latitudeConfidence: FloatArray,
    )

    data class Result(
        val focalPixels: Float,
        val rollRadians: Float,
        val pitchRadians: Float,
        val finalCost: Float,
        val iterations: Int,
    )

    private data class Sample(
        val x: Float,
        val y: Float,
        val upX: Float,
        val upY: Float,
        val latitudeSin: Float,
        val upWeight: Float,
        val latitudeWeight: Float,
    )

    private data class State(var roll: Float, var pitch: Float, var logFocal: Float)

    fun solve(
        fields: Fields,
        contentMinX: Int = 0,
        contentMinY: Int = 0,
        contentMaxX: Int = Int.MAX_VALUE,
        contentMaxY: Int = Int.MAX_VALUE,
    ): Result? {
        val samples = makeSamples(fields, 8, contentMinX, contentMinY, contentMaxX, contentMaxY)
        if (samples.size < 64) return null

        val initialFocal = maxOf(12f, 0.7f * maxOf(fields.width, fields.height))
        var state = State(0f, 0f, ln(initialFocal))
        var lambda = 0.1f
        var previous = evaluate(state, samples, fields.width, fields.height)
        var completedIterations = 0

        for (iteration in 0 until 20) {
            val system = linearizedSystem(state, previous.residuals, samples, fields.width, fields.height)
            val delta = solveDamped(system, lambda) ?: break
            val candidate = State(
                roll = (state.roll + delta.first).coerceIn(-1.3f, 1.3f),
                pitch = (state.pitch + delta.second).coerceIn(-1.3f, 1.3f),
                logFocal = (state.logFocal + delta.third).coerceIn(
                    ln(minFocal(fields.height)),
                    ln(maxFocal(fields.height)),
                ),
            )
            val next = evaluate(candidate, samples, fields.width, fields.height)
            completedIterations = iteration + 1
            if (next.cost < previous.cost) {
                val improvement = previous.cost - next.cost
                state = candidate
                previous = next
                lambda = maxOf(1e-6f, lambda * 0.1f)
                if (improvement < maxOf(1e-8f, previous.cost * 1e-5f)) break
            } else {
                lambda = minOf(100f, lambda * 10f)
            }
        }

        val focal = exp(state.logFocal)
        if (!focal.isFinite() || focal <= 1f) return null
        return Result(focal, state.roll, state.pitch, previous.cost, completedIterations)
    }

    private fun makeSamples(
        fields: Fields,
        stride: Int,
        contentMinX: Int,
        contentMinY: Int,
        contentMaxX: Int,
        contentMaxY: Int,
    ): List<Sample> {
        val samples = ArrayList<Sample>()
        var y = maxOf(stride / 2, contentMinY)
        while (y < minOf(fields.height, contentMaxY + 1)) {
            var x = maxOf(stride / 2, contentMinX)
            while (x < minOf(fields.width, contentMaxX + 1)) {
                val i = y * fields.width + x
                val ux = fields.upX[i]
                val uy = fields.upY[i]
                val lat = fields.latitudeRadians[i]
                if (ux.isFinite() && uy.isFinite() && lat.isFinite()) {
                    val upNorm = maxOf(1e-6f, sqrt(ux * ux + uy * uy))
                    val upConfidence = fields.upConfidence[i].coerceIn(0f, 1f)
                    val latConfidence = fields.latitudeConfidence[i].coerceIn(0f, 1f)
                    if (upConfidence > 1e-4f || latConfidence > 1e-4f) {
                        samples += Sample(
                            x = x.toFloat(),
                            y = y.toFloat(),
                            upX = ux / upNorm,
                            upY = uy / upNorm,
                            latitudeSin = sin(lat),
                            upWeight = maxOf(0.05f, upConfidence),
                            latitudeWeight = maxOf(0.05f, latConfidence),
                        )
                    }
                }
                x += stride
            }
            y += stride
        }
        return samples
    }

    private data class EvalResult(val cost: Float, val residuals: List<FloatArray>)

    private fun evaluate(state: State, samples: List<Sample>, width: Int, height: Int): EvalResult {
        var totalCost = 0f
        val residuals = ArrayList<FloatArray>(samples.size)
        for (sample in samples) {
            val prediction = predictedFields(state, sample, width, height)
            val upRX = sample.upX - prediction.upX
            val upRY = sample.upY - prediction.upY
            val latR = sample.latitudeSin - prediction.latitudeSin
            val upSquared = upRX * upRX + upRY * upRY
            val latSquared = latR * latR
            totalCost += sample.upWeight * huberCost(upSquared, 1e-2f)
            totalCost += sample.latitudeWeight * huberCost(latSquared, 1e-2f)
            residuals += floatArrayOf(upRX, upRY, latR)
        }
        return EvalResult(totalCost / maxOf(1, samples.size), residuals)
    }

    private fun linearizedSystem(
        state: State,
        residuals: List<FloatArray>,
        samples: List<Sample>,
        width: Int,
        height: Int,
    ): Pair<FloatArray, Array<FloatArray>> {
        val gradient = FloatArray(3)
        val hessian = Array(3) { FloatArray(3) }
        val eps = floatArrayOf(1e-3f, 1e-3f, 1e-3f)
        samples.forEachIndexed { index, sample ->
            val base = predictedFields(state, sample, width, height)
            val residual = residuals[index]
            val upSquared = residual[0] * residual[0] + residual[1] * residual[1]
            val latSquared = residual[2] * residual[2]
            val upWeight = sample.upWeight * huberWeight(upSquared, 1e-2f)
            val latWeight = sample.latitudeWeight * huberWeight(latSquared, 1e-2f)
            val jacobian = Array(3) { FloatArray(3) }
            for (parameter in 0 until 3) {
                val stepped = state.copy()
                when (parameter) {
                    0 -> stepped.roll += eps[parameter]
                    1 -> stepped.pitch += eps[parameter]
                    else -> stepped.logFocal += eps[parameter]
                }
                val next = predictedFields(stepped, sample, width, height)
                jacobian[0][parameter] = (next.upX - base.upX) / eps[parameter]
                jacobian[1][parameter] = (next.upY - base.upY) / eps[parameter]
                jacobian[2][parameter] = (next.latitudeSin - base.latitudeSin) / eps[parameter]
            }
            val weights = floatArrayOf(upWeight, upWeight, latWeight)
            for (row in 0 until 3) {
                for (p in 0 until 3) {
                    gradient[p] += weights[row] * jacobian[row][p] * residual[row]
                    for (q in 0 until 3) {
                        hessian[p][q] += weights[row] * jacobian[row][p] * jacobian[row][q]
                    }
                }
            }
        }
        return gradient to hessian
    }

    private data class PredictedFields(val upX: Float, val upY: Float, val latitudeSin: Float)

    private fun predictedFields(state: State, sample: Sample, width: Int, height: Int): PredictedFields {
        val focal = exp(state.logFocal)
        val cx = width * 0.5f
        val cy = height * 0.5f
        val u = (sample.x - cx) / focal
        val v = (sample.y - cy) / focal
        val sr = sin(state.roll)
        val cr = cos(state.roll)
        val sp = sin(state.pitch)
        val cp = cos(state.pitch)
        val gx = -sr * cp
        val gy = -cr * cp
        val gz = sp
        var upX = gx - gz * u
        var upY = gy - gz * v
        val upNorm = maxOf(1e-6f, sqrt(upX * upX + upY * upY))
        upX /= upNorm
        upY /= upNorm
        val rayNorm = maxOf(1e-6f, sqrt(u * u + v * v + 1f))
        val latitudeSin = ((u * gx + v * gy + gz) / rayNorm).coerceIn(-1f, 1f)
        return PredictedFields(upX, upY, latitudeSin)
    }

    private fun huberCost(squared: Float, scale: Float): Float {
        val normalized = squared / (scale * scale)
        return if (normalized <= 1f) squared else (2f * sqrt(maxOf(normalized, 0f)) - 1f) * scale * scale
    }

    private fun huberWeight(squared: Float, scale: Float): Float {
        val normalized = squared / (scale * scale)
        return if (normalized <= 1f) 1f else 1f / maxOf(sqrt(normalized), Float.MIN_VALUE)
    }

    private fun solveDamped(
        system: Pair<FloatArray, Array<FloatArray>>,
        lambda: Float,
    ): Triple<Float, Float, Float>? {
        val a = system.second.map { it.copyOf() }.toTypedArray()
        val g = system.first
        for (i in 0 until 3) {
            a[i][i] += maxOf(1e-6f, abs(a[i][i]) * lambda)
        }
        return solve3x3(a, g)
    }

    private fun solve3x3(matrix: Array<FloatArray>, rhs: FloatArray): Triple<Float, Float, Float>? {
        val a = matrix.map { it.copyOf() }.toTypedArray()
        val b = rhs.copyOf()
        for (pivot in 0 until 3) {
            var best = pivot
            var bestValue = abs(a[pivot][pivot])
            for (row in pivot + 1 until 3) {
                val value = abs(a[row][pivot])
                if (value > bestValue) {
                    best = row
                    bestValue = value
                }
            }
            if (bestValue <= 1e-9f) return null
            if (best != pivot) {
                val tempRow = a[best]
                a[best] = a[pivot]
                a[pivot] = tempRow
                val tempB = b[best]
                b[best] = b[pivot]
                b[pivot] = tempB
            }
            val pivotValue = a[pivot][pivot]
            for (column in pivot until 3) a[pivot][column] /= pivotValue
            b[pivot] /= pivotValue
            for (row in 0 until 3) {
                if (row == pivot) continue
                val factor = a[row][pivot]
                if (factor == 0f) continue
                for (column in pivot until 3) a[row][column] -= factor * a[pivot][column]
                b[row] -= factor * b[pivot]
            }
        }
        return Triple(b[0], b[1], b[2])
    }

    private fun minFocal(height: Int): Float = fovToFocal(150f, height.toFloat())
    private fun maxFocal(height: Int): Float = fovToFocal(5f, height.toFloat())
    private fun fovToFocal(degrees: Float, size: Float): Float {
        val radians = Math.toRadians(degrees.toDouble() / 2.0).toFloat()
        return size / (2f * tan(radians))
    }
}
