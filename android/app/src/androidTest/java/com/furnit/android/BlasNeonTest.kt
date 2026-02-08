package com.furnit.android

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.furnit.android.services.NcnnSharp
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import kotlin.math.abs
import kotlin.math.ln
import kotlin.math.sqrt

/**
 * Test BLAS-equivalent NEON optimizations in native code.
 *
 * Tests:
 * - Quaternion normalization (nativeNormalizeQuaternions)
 * - Full PLY transformation (nativeTransformForPly)
 * - Correctness against Kotlin reference implementations
 * - Performance comparison
 *
 * Run with: ./gradlew connectedAndroidTest -Pandroid.testInstrumentationRunnerArguments.class=com.furnit.android.BlasNeonTest
 */
@RunWith(AndroidJUnit4::class)
class BlasNeonTest {

    private lateinit var ncnnSharp: NcnnSharp
    private var neonAvailable = false

    companion object {
        private const val EPSILON = 1e-5f
        private const val SH_C0 = 0.28209479177387814f
    }

    @Before
    fun setup() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        neonAvailable = NcnnSharp.isNeonTransformAvailable()
        if (neonAvailable) {
            ncnnSharp = NcnnSharp(context)
        }
        println("NEON transform available: $neonAvailable")
    }

    // ============================================================================
    // Kotlin Reference Implementations (for comparison)
    // ============================================================================

    /**
     * Reference quaternion normalization in Kotlin.
     */
    private fun normalizeQuaternionKotlin(q: FloatArray, offset: Int) {
        val w = q[offset]
        val x = q[offset + 1]
        val y = q[offset + 2]
        val z = q[offset + 3]
        val mag = sqrt(w * w + x * x + y * y + z * z)
        val invMag = if (mag > 1e-8f) 1f / mag else 1f
        q[offset] = w * invMag
        q[offset + 1] = x * invMag
        q[offset + 2] = y * invMag
        q[offset + 3] = z * invMag
    }

    /**
     * Reference opacity logit in Kotlin.
     */
    private fun opacityLogitKotlin(opacity: Float): Float {
        val o = opacity.coerceIn(1e-4f, 1f - 1e-4f)
        return ln(o / (1f - o))
    }

    /**
     * Reference color to SH DC in Kotlin.
     */
    private fun colorToShKotlin(color: Float): Float {
        val c = color.coerceIn(0f, 1f)
        return (c - 0.5f) / SH_C0
    }

    /**
     * Reference log scale in Kotlin.
     */
    private fun logScaleKotlin(scale: Float, boost: Float, minScale: Float): Float {
        return ln(maxOf(scale * boost, minScale))
    }

    // ============================================================================
    // Quaternion Normalization Tests
    // ============================================================================

    @Test
    fun testQuaternionNormalization_SingleQuaternion() {
        if (!neonAvailable) {
            println("Skipping test - NEON not available")
            return
        }

        // Test quaternion [2, 0, 0, 0] should normalize to [1, 0, 0, 0]
        val quaternions = floatArrayOf(2f, 0f, 0f, 0f)
        val expected = floatArrayOf(1f, 0f, 0f, 0f)

        ncnnSharp.nativeNormalizeQuaternions(quaternions, 1)

        println("Input: [2, 0, 0, 0]")
        println("Output: [${quaternions[0]}, ${quaternions[1]}, ${quaternions[2]}, ${quaternions[3]}]")
        println("Expected: [1, 0, 0, 0]")

        for (i in 0..3) {
            assertEquals("Component $i mismatch", expected[i], quaternions[i], EPSILON)
        }
        println("PASSED: Single quaternion normalization")
    }

    @Test
    fun testQuaternionNormalization_UnitQuaternion() {
        if (!neonAvailable) {
            println("Skipping test - NEON not available")
            return
        }

        // Already normalized quaternion should stay the same
        val quaternions = floatArrayOf(0.5f, 0.5f, 0.5f, 0.5f)
        val original = quaternions.copyOf()

        ncnnSharp.nativeNormalizeQuaternions(quaternions, 1)

        println("Input (unit quaternion): [${original.joinToString()}]")
        println("Output: [${quaternions.joinToString()}]")

        // Check magnitude is 1
        val mag = sqrt(quaternions[0] * quaternions[0] + quaternions[1] * quaternions[1] +
                       quaternions[2] * quaternions[2] + quaternions[3] * quaternions[3])
        assertEquals("Magnitude should be 1", 1f, mag, EPSILON)
        println("PASSED: Unit quaternion preserved")
    }

    @Test
    fun testQuaternionNormalization_MultipleQuaternions() {
        if (!neonAvailable) {
            println("Skipping test - NEON not available")
            return
        }

        val numQuaternions = 1000
        val quaternions = FloatArray(numQuaternions * 4)
        val reference = FloatArray(numQuaternions * 4)

        // Generate random unnormalized quaternions
        val random = java.util.Random(42)
        for (i in 0 until numQuaternions) {
            val offset = i * 4
            quaternions[offset] = random.nextFloat() * 10f - 5f
            quaternions[offset + 1] = random.nextFloat() * 10f - 5f
            quaternions[offset + 2] = random.nextFloat() * 10f - 5f
            quaternions[offset + 3] = random.nextFloat() * 10f - 5f
            // Copy for reference
            reference[offset] = quaternions[offset]
            reference[offset + 1] = quaternions[offset + 1]
            reference[offset + 2] = quaternions[offset + 2]
            reference[offset + 3] = quaternions[offset + 3]
        }

        // Normalize with NEON
        val neonStart = System.nanoTime()
        ncnnSharp.nativeNormalizeQuaternions(quaternions, numQuaternions)
        val neonTime = System.nanoTime() - neonStart

        // Normalize reference with Kotlin
        val kotlinStart = System.nanoTime()
        for (i in 0 until numQuaternions) {
            normalizeQuaternionKotlin(reference, i * 4)
        }
        val kotlinTime = System.nanoTime() - kotlinStart

        println("NEON time: ${neonTime / 1000}µs for $numQuaternions quaternions")
        println("Kotlin time: ${kotlinTime / 1000}µs for $numQuaternions quaternions")
        println("Speedup: ${kotlinTime.toFloat() / neonTime}x")

        // Verify results match
        var maxError = 0f
        for (i in 0 until numQuaternions * 4) {
            val error = abs(quaternions[i] - reference[i])
            if (error > maxError) maxError = error
            assertEquals("Mismatch at index $i", reference[i], quaternions[i], EPSILON)
        }
        println("Max error: $maxError")
        println("PASSED: Multiple quaternion normalization")
    }

    @Test
    fun testQuaternionNormalization_ZeroQuaternion() {
        if (!neonAvailable) {
            println("Skipping test - NEON not available")
            return
        }

        // Zero quaternion - should handle gracefully (return identity or unchanged)
        val quaternions = floatArrayOf(0f, 0f, 0f, 0f)

        ncnnSharp.nativeNormalizeQuaternions(quaternions, 1)

        println("Zero quaternion output: [${quaternions.joinToString()}]")

        // Just verify it didn't crash and produced finite values
        for (i in 0..3) {
            assertTrue("Component $i should be finite", quaternions[i].isFinite())
        }
        println("PASSED: Zero quaternion handled gracefully")
    }

    // ============================================================================
    // Full PLY Transform Tests
    // ============================================================================

    @Test
    fun testPlyTransform_SmallBatch() {
        if (!neonAvailable) {
            println("Skipping test - NEON not available")
            return
        }

        val numGaussians = 10
        val scaleBoost = 1.3f
        val minScale = 0.001f

        // Create test data
        val positions = FloatArray(numGaussians * 3)
        val scales = FloatArray(numGaussians * 3)
        val rotations = FloatArray(numGaussians * 4)
        val colors = FloatArray(numGaussians * 3)
        val opacities = FloatArray(numGaussians)

        val random = java.util.Random(123)
        for (i in 0 until numGaussians) {
            // Positions: random [-5, 5]
            positions[i * 3] = random.nextFloat() * 10f - 5f
            positions[i * 3 + 1] = random.nextFloat() * 10f - 5f
            positions[i * 3 + 2] = random.nextFloat() * 10f - 5f

            // Scales: random [0.01, 1]
            scales[i * 3] = random.nextFloat() * 0.99f + 0.01f
            scales[i * 3 + 1] = random.nextFloat() * 0.99f + 0.01f
            scales[i * 3 + 2] = random.nextFloat() * 0.99f + 0.01f

            // Rotations: random unnormalized quaternions
            rotations[i * 4] = random.nextFloat() * 2f - 1f
            rotations[i * 4 + 1] = random.nextFloat() * 2f - 1f
            rotations[i * 4 + 2] = random.nextFloat() * 2f - 1f
            rotations[i * 4 + 3] = random.nextFloat() * 2f - 1f

            // Colors: random [0, 1]
            colors[i * 3] = random.nextFloat()
            colors[i * 3 + 1] = random.nextFloat()
            colors[i * 3 + 2] = random.nextFloat()

            // Opacities: random [0.1, 0.9]
            opacities[i] = random.nextFloat() * 0.8f + 0.1f
        }

        // Run NEON transform
        val result = ncnnSharp.nativeTransformForPly(
            positions, scales, rotations, colors, opacities,
            numGaussians, scaleBoost, minScale
        )

        assertNotNull("Result should not be null", result)
        assertEquals("Result size", numGaussians * 14, result!!.size)

        println("PLY Transform result for $numGaussians Gaussians:")

        // Verify each Gaussian
        for (i in 0 until numGaussians) {
            val offset = i * 14

            // Position: x unchanged, y and z negated
            assertEquals("Position X", positions[i * 3], result[offset], EPSILON)
            assertEquals("Position Y (negated)", -positions[i * 3 + 1], result[offset + 1], EPSILON)
            assertEquals("Position Z (negated)", -positions[i * 3 + 2], result[offset + 2], EPSILON)

            // Scale: log transformed
            val expectedScaleX = logScaleKotlin(scales[i * 3], scaleBoost, minScale)
            val expectedScaleY = logScaleKotlin(scales[i * 3 + 1], scaleBoost, minScale)
            val expectedScaleZ = logScaleKotlin(scales[i * 3 + 2], scaleBoost, minScale)
            assertEquals("Scale X", expectedScaleX, result[offset + 3], EPSILON)
            assertEquals("Scale Y", expectedScaleY, result[offset + 4], EPSILON)
            assertEquals("Scale Z", expectedScaleZ, result[offset + 5], EPSILON)

            // Rotation: normalized (check magnitude = 1)
            val rotW = result[offset + 6]
            val rotX = result[offset + 7]
            val rotY = result[offset + 8]
            val rotZ = result[offset + 9]
            val rotMag = sqrt(rotW * rotW + rotX * rotX + rotY * rotY + rotZ * rotZ)
            assertEquals("Rotation magnitude", 1f, rotMag, EPSILON)

            // Opacity: logit transformed
            val expectedOpacity = opacityLogitKotlin(opacities[i])
            assertEquals("Opacity", expectedOpacity, result[offset + 10], EPSILON)

            // Colors: SH DC transformed
            val expectedR = colorToShKotlin(colors[i * 3])
            val expectedG = colorToShKotlin(colors[i * 3 + 1])
            val expectedB = colorToShKotlin(colors[i * 3 + 2])
            assertEquals("Color R", expectedR, result[offset + 11], EPSILON)
            assertEquals("Color G", expectedG, result[offset + 12], EPSILON)
            assertEquals("Color B", expectedB, result[offset + 13], EPSILON)

            if (i < 3) {
                println("Gaussian $i:")
                println("  Position: [${result[offset]}, ${result[offset + 1]}, ${result[offset + 2]}]")
                println("  Scale: [${result[offset + 3]}, ${result[offset + 4]}, ${result[offset + 5]}]")
                println("  Rotation: [${result[offset + 6]}, ${result[offset + 7]}, ${result[offset + 8]}, ${result[offset + 9]}]")
                println("  Opacity: ${result[offset + 10]}")
                println("  Color: [${result[offset + 11]}, ${result[offset + 12]}, ${result[offset + 13]}]")
            }
        }
        println("PASSED: Small batch PLY transform")
    }

    @Test
    fun testPlyTransform_LargeBatch() {
        if (!neonAvailable) {
            println("Skipping test - NEON not available")
            return
        }

        // Test with realistic Gaussian count (100K, similar to actual model output)
        val numGaussians = 100_000
        val scaleBoost = 1.3f
        val minScale = 0.001f

        println("Testing PLY transform with $numGaussians Gaussians...")

        // Create test data
        val positions = FloatArray(numGaussians * 3)
        val scales = FloatArray(numGaussians * 3)
        val rotations = FloatArray(numGaussians * 4)
        val colors = FloatArray(numGaussians * 3)
        val opacities = FloatArray(numGaussians)

        val random = java.util.Random(456)
        for (i in 0 until numGaussians) {
            positions[i * 3] = random.nextFloat() * 10f - 5f
            positions[i * 3 + 1] = random.nextFloat() * 10f - 5f
            positions[i * 3 + 2] = random.nextFloat() * 10f - 5f

            scales[i * 3] = random.nextFloat() * 0.5f + 0.01f
            scales[i * 3 + 1] = random.nextFloat() * 0.5f + 0.01f
            scales[i * 3 + 2] = random.nextFloat() * 0.5f + 0.01f

            rotations[i * 4] = random.nextFloat() * 2f - 1f
            rotations[i * 4 + 1] = random.nextFloat() * 2f - 1f
            rotations[i * 4 + 2] = random.nextFloat() * 2f - 1f
            rotations[i * 4 + 3] = random.nextFloat() * 2f - 1f

            colors[i * 3] = random.nextFloat()
            colors[i * 3 + 1] = random.nextFloat()
            colors[i * 3 + 2] = random.nextFloat()

            opacities[i] = random.nextFloat() * 0.8f + 0.1f
        }

        // Time NEON transform
        val neonStart = System.currentTimeMillis()
        val result = ncnnSharp.nativeTransformForPly(
            positions, scales, rotations, colors, opacities,
            numGaussians, scaleBoost, minScale
        )
        val neonTime = System.currentTimeMillis() - neonStart

        assertNotNull("Result should not be null", result)
        assertEquals("Result size", numGaussians * 14, result!!.size)

        println("NEON PLY transform: ${neonTime}ms for $numGaussians Gaussians")
        println("Throughput: ${numGaussians / (neonTime / 1000.0)} Gaussians/sec")

        // Time Kotlin reference implementation
        val kotlinStart = System.currentTimeMillis()
        val kotlinResult = FloatArray(numGaussians * 14)
        val refQuats = rotations.copyOf()

        for (i in 0 until numGaussians) {
            val inOffset = i * 3
            val rotOffset = i * 4
            val outOffset = i * 14

            // Position
            kotlinResult[outOffset] = positions[inOffset]
            kotlinResult[outOffset + 1] = -positions[inOffset + 1]
            kotlinResult[outOffset + 2] = -positions[inOffset + 2]

            // Scale
            kotlinResult[outOffset + 3] = logScaleKotlin(scales[inOffset], scaleBoost, minScale)
            kotlinResult[outOffset + 4] = logScaleKotlin(scales[inOffset + 1], scaleBoost, minScale)
            kotlinResult[outOffset + 5] = logScaleKotlin(scales[inOffset + 2], scaleBoost, minScale)

            // Rotation (normalize in place)
            normalizeQuaternionKotlin(refQuats, rotOffset)
            kotlinResult[outOffset + 6] = refQuats[rotOffset]
            kotlinResult[outOffset + 7] = refQuats[rotOffset + 1]
            kotlinResult[outOffset + 8] = refQuats[rotOffset + 2]
            kotlinResult[outOffset + 9] = refQuats[rotOffset + 3]

            // Opacity
            kotlinResult[outOffset + 10] = opacityLogitKotlin(opacities[i])

            // Colors
            kotlinResult[outOffset + 11] = colorToShKotlin(colors[inOffset])
            kotlinResult[outOffset + 12] = colorToShKotlin(colors[inOffset + 1])
            kotlinResult[outOffset + 13] = colorToShKotlin(colors[inOffset + 2])
        }
        val kotlinTime = System.currentTimeMillis() - kotlinStart

        println("Kotlin PLY transform: ${kotlinTime}ms for $numGaussians Gaussians")
        println("Speedup: ${kotlinTime.toFloat() / neonTime}x")

        // Verify results match (sample check to avoid slow full comparison)
        val sampleIndices = listOf(0, 1000, 50000, 99999)
        var maxError = 0f
        for (idx in sampleIndices) {
            for (j in 0 until 14) {
                val i = idx * 14 + j
                val error = abs(result[i] - kotlinResult[i])
                if (error > maxError) maxError = error
                assertEquals("Mismatch at Gaussian $idx, component $j", kotlinResult[i], result[i], EPSILON)
            }
        }
        println("Max error (sampled): $maxError")
        println("PASSED: Large batch PLY transform")
    }

    // ============================================================================
    // Edge Case Tests
    // ============================================================================

    @Test
    fun testPlyTransform_EdgeCases() {
        if (!neonAvailable) {
            println("Skipping test - NEON not available")
            return
        }

        val numGaussians = 5
        val scaleBoost = 1.3f
        val minScale = 0.001f

        // Edge case values
        val positions = floatArrayOf(
            0f, 0f, 0f,           // Origin
            1e10f, 1e10f, 1e10f,  // Large positive
            -1e10f, -1e10f, -1e10f, // Large negative
            1e-10f, 1e-10f, 1e-10f, // Small positive
            Float.MAX_VALUE / 2, 0f, 0f  // Near max
        )

        val scales = floatArrayOf(
            0.001f, 0.001f, 0.001f,  // Minimum scale
            1f, 1f, 1f,              // Unit scale
            0.0001f, 0.0001f, 0.0001f, // Below min (should clamp)
            10f, 10f, 10f,           // Large scale
            0.5f, 0.5f, 0.5f         // Normal scale
        )

        val rotations = floatArrayOf(
            1f, 0f, 0f, 0f,          // Identity quaternion
            0f, 1f, 0f, 0f,          // 180° around X
            0.5f, 0.5f, 0.5f, 0.5f,  // Normalized
            10f, 10f, 10f, 10f,      // Unnormalized (should normalize)
            0.001f, 0.001f, 0.001f, 0.001f  // Near-zero (edge case)
        )

        val colors = floatArrayOf(
            0f, 0f, 0f,        // Black
            1f, 1f, 1f,        // White
            0.5f, 0.5f, 0.5f,  // Gray
            -0.1f, 1.1f, 0.5f, // Out of range (should clamp)
            0.25f, 0.75f, 0.5f // Normal
        )

        val opacities = floatArrayOf(
            0.0001f,  // Near zero
            0.9999f,  // Near one
            0.5f,     // Middle
            0.0f,     // Zero (should clamp)
            1.0f      // One (should clamp)
        )

        val result = ncnnSharp.nativeTransformForPly(
            positions, scales, rotations, colors, opacities,
            numGaussians, scaleBoost, minScale
        )

        assertNotNull("Result should not be null", result)

        println("Edge case results:")
        for (i in 0 until numGaussians) {
            val offset = i * 14
            println("Gaussian $i:")
            println("  Position: [${result!![offset]}, ${result[offset + 1]}, ${result[offset + 2]}]")
            println("  Scale: [${result[offset + 3]}, ${result[offset + 4]}, ${result[offset + 5]}]")
            println("  Rotation mag: ${sqrt(result[offset + 6] * result[offset + 6] + result[offset + 7] * result[offset + 7] + result[offset + 8] * result[offset + 8] + result[offset + 9] * result[offset + 9])}")
            println("  Opacity: ${result[offset + 10]}")
            println("  Color: [${result[offset + 11]}, ${result[offset + 12]}, ${result[offset + 13]}]")

            // Verify all values are finite
            for (j in 0 until 14) {
                assertTrue("Value at [$i, $j] should be finite", result[offset + j].isFinite())
            }

            // Verify rotation is normalized
            val rotMag = sqrt(result[offset + 6] * result[offset + 6] +
                             result[offset + 7] * result[offset + 7] +
                             result[offset + 8] * result[offset + 8] +
                             result[offset + 9] * result[offset + 9])
            assertEquals("Rotation $i should be normalized", 1f, rotMag, 0.01f)
        }
        println("PASSED: Edge cases handled correctly")
    }

    // ============================================================================
    // Performance Benchmark
    // ============================================================================

    @Test
    fun testPerformanceBenchmark() {
        if (!neonAvailable) {
            println("Skipping benchmark - NEON not available")
            return
        }

        println("\n=== BLAS/NEON Performance Benchmark ===\n")

        // Test different batch sizes
        val batchSizes = listOf(1000, 10_000, 100_000, 500_000, 1_000_000)

        for (numGaussians in batchSizes) {
            // Skip very large tests on low-memory devices
            try {
                val positions = FloatArray(numGaussians * 3) { it * 0.001f }
                val scales = FloatArray(numGaussians * 3) { 0.1f }
                val rotations = FloatArray(numGaussians * 4) { if (it % 4 == 0) 1f else 0f }
                val colors = FloatArray(numGaussians * 3) { 0.5f }
                val opacities = FloatArray(numGaussians) { 0.5f }

                // Warmup
                ncnnSharp.nativeTransformForPly(
                    positions, scales, rotations, colors, opacities,
                    numGaussians, 1.3f, 0.001f
                )

                // Benchmark
                val iterations = 3
                var totalTime = 0L
                for (iter in 0 until iterations) {
                    val start = System.currentTimeMillis()
                    ncnnSharp.nativeTransformForPly(
                        positions, scales, rotations, colors, opacities,
                        numGaussians, 1.3f, 0.001f
                    )
                    totalTime += System.currentTimeMillis() - start
                }
                val avgTime = totalTime / iterations

                val throughput = numGaussians / (avgTime / 1000.0)
                println("$numGaussians Gaussians: ${avgTime}ms avg (${String.format("%.2f", throughput / 1_000_000)}M Gaussians/sec)")

            } catch (e: OutOfMemoryError) {
                println("$numGaussians Gaussians: Skipped (OOM)")
            }
        }

        println("\n=== Benchmark Complete ===")
    }
}
