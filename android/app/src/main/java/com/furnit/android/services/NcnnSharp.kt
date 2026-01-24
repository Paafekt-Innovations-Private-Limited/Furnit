package com.furnit.android.services

import android.content.Context
import android.content.res.AssetManager
import android.graphics.Bitmap
import android.graphics.Color
import android.util.Log
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * NcnnSharp provides 3D Gaussian Splat generation using NCNN.
 *
 * This class generates 3D Gaussian splat representations from single images,
 * creating PLY files compatible with Gaussian splatting renderers.
 *
 * The SHARP model generates ~1.1M Gaussians from a 1536x1536 input image:
 * - positions (1 × N × 3)
 * - scales (1 × N × 3)
 * - rotations (1 × N × 4)
 * - colors (1 × N × 3)
 * - opacity (1 × N)
 *
 * Model files required in assets folder:
 * - sharp.ncnn.param (NCNN graph definition)
 * - sharp.ncnn.bin (NCNN weights)
 */
class NcnnSharp(private val context: Context) {

    companion object {
        private const val TAG = "NcnnSharp"

        // Input size expected by SHARP model
        const val INPUT_SIZE = 1536

        // Gaussian parameters per splat: pos(3) + scale(3) + rot(4) + opacity(1) + color(3) = 14
        const val PARAMS_PER_GAUSSIAN = 14

        // Grid size for Gaussian generation
        private const val GRID_SIZE = 128  // Higher resolution grid

        // Room dimensions (in meters)
        private const val ROOM_WIDTH = 4.0f
        private const val ROOM_HEIGHT = 3.0f
        private const val ROOM_DEPTH = 5.0f

        // Splat scale
        private const val SPLAT_SCALE = 0.015f

        // Try to load native library
        private var libraryLoaded = false
        private var libraryLoadError: String? = null

        init {
            try {
                System.loadLibrary("sharp_ncnn")
                libraryLoaded = true
                Log.i(TAG, "SHARP NCNN library loaded")
            } catch (e: UnsatisfiedLinkError) {
                Log.w(TAG, "SHARP NCNN library not available: ${e.message}")
                libraryLoaded = false
                libraryLoadError = e.message
            }
        }

        fun isNativeAvailable(): Boolean = libraryLoaded
    }

    private var nativeHandle: Long = 0
    private var isInitialized = false

    /**
     * Initialize NCNN model from assets.
     * Returns true if native model is available, false if using fallback.
     */
    fun init(
        paramAsset: String = "sharp.ncnn.param",
        binAsset: String = "sharp.ncnn.bin",
        useGpu: Boolean = true
    ): Boolean {
        if (libraryLoaded) {
            return try {
                nativeHandle = nativeInit(
                    context.assets,
                    paramAsset,
                    binAsset,
                    useGpu,
                    4  // threads
                )
                isInitialized = nativeHandle != 0L
                Log.i(TAG, "NCNN SHARP initialized: $isInitialized")
                isInitialized
            } catch (e: Exception) {
                Log.e(TAG, "NCNN init failed: ${e.message}")
                false
            }
        }
        Log.i(TAG, "Using fallback Gaussian generation (native library not available)")
        return false
    }

    /**
     * Generate Gaussians from image.
     * Uses native NCNN if available, otherwise falls back to depth-aware generation.
     */
    fun generateGaussians(bitmap: Bitmap): GaussianResult {
        return if (isInitialized && nativeHandle != 0L) {
            generateGaussiansNative(bitmap)
        } else {
            generateGaussiansFallback(bitmap)
        }
    }

    /**
     * Native NCNN inference (when model is available)
     */
    private fun generateGaussiansNative(bitmap: Bitmap): GaussianResult {
        val scaledBitmap = if (bitmap.width != INPUT_SIZE || bitmap.height != INPUT_SIZE) {
            Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)
        } else {
            bitmap
        }

        val rawParams = nativeInfer(nativeHandle, scaledBitmap)

        if (rawParams == null || rawParams.isEmpty()) {
            Log.w(TAG, "Native inference returned no results, using fallback")
            return generateGaussiansFallback(bitmap)
        }

        val gaussianCount = rawParams.size / PARAMS_PER_GAUSSIAN
        Log.d(TAG, "Native inference generated $gaussianCount Gaussians")

        return GaussianResult(
            params = rawParams,
            gaussianCount = gaussianCount,
            roomWidth = ROOM_WIDTH,
            roomHeight = ROOM_HEIGHT,
            roomDepth = ROOM_DEPTH
        )
    }

    /**
     * Fallback generation using depth-aware algorithm.
     * Creates a proper 3D room with walls, floor, ceiling from image.
     */
    private fun generateGaussiansFallback(bitmap: Bitmap): GaussianResult {
        Log.d(TAG, "Generating Gaussians with depth-aware fallback")

        val startTime = System.currentTimeMillis()

        // Scale image for processing
        val processSize = 256
        val scaledBitmap = Bitmap.createScaledBitmap(bitmap, processSize, processSize, true)

        val gaussians = mutableListOf<FloatArray>()

        // Generate front wall (main image)
        generateFrontWall(scaledBitmap, gaussians)

        // Generate floor
        generateFloor(scaledBitmap, gaussians)

        // Generate ceiling
        generateCeiling(scaledBitmap, gaussians)

        // Generate side walls
        generateSideWalls(scaledBitmap, gaussians)

        // Generate back wall (fade to black)
        generateBackWall(gaussians)

        // Combine into single array
        val params = FloatArray(gaussians.size * PARAMS_PER_GAUSSIAN)
        var minX = Float.MAX_VALUE
        var maxX = Float.MIN_VALUE
        var minY = Float.MAX_VALUE
        var maxY = Float.MIN_VALUE
        var minZ = Float.MAX_VALUE
        var maxZ = Float.MIN_VALUE

        for ((i, g) in gaussians.withIndex()) {
            val offset = i * PARAMS_PER_GAUSSIAN
            for (j in 0 until PARAMS_PER_GAUSSIAN) {
                params[offset + j] = g[j]
            }
            // Track bounds
            minX = min(minX, g[0])
            maxX = max(maxX, g[0])
            minY = min(minY, g[1])
            maxY = max(maxY, g[1])
            minZ = min(minZ, g[2])
            maxZ = max(maxZ, g[2])
        }

        val width = maxX - minX
        val height = maxY - minY
        val depth = maxZ - minZ

        val elapsed = System.currentTimeMillis() - startTime
        Log.d(TAG, "Generated ${gaussians.size} Gaussians in ${elapsed}ms")
        Log.d(TAG, "Room bounds: ${width}x${height}x${depth}")

        return GaussianResult(
            params = params,
            gaussianCount = gaussians.size,
            roomWidth = width,
            roomHeight = height,
            roomDepth = depth
        )
    }

    private fun generateFrontWall(image: Bitmap, gaussians: MutableList<FloatArray>) {
        val width = image.width
        val height = image.height

        val stepX = width / GRID_SIZE
        val stepY = height / GRID_SIZE

        for (gridY in 0 until GRID_SIZE) {
            for (gridX in 0 until GRID_SIZE) {
                val pixelX = (gridX * stepX + stepX / 2).coerceIn(0, width - 1)
                val pixelY = (gridY * stepY + stepY / 2).coerceIn(0, height - 1)

                val pixel = image.getPixel(pixelX, pixelY)
                val r = Color.red(pixel) / 255f
                val g = Color.green(pixel) / 255f
                val b = Color.blue(pixel) / 255f

                // Skip very dark pixels
                if ((r + g + b) / 3f < 0.05f) continue

                // Map to 3D space
                val x = (gridX.toFloat() / GRID_SIZE - 0.5f) * ROOM_WIDTH
                val y = (0.5f - gridY.toFloat() / GRID_SIZE) * ROOM_HEIGHT
                val z = -ROOM_DEPTH * 0.4f  // Front wall position

                // Estimate depth from brightness (brighter = closer)
                val brightness = (r + g + b) / 3f
                val depthOffset = (1f - brightness) * 0.3f

                gaussians.add(createGaussian(
                    x, y, z + depthOffset,
                    SPLAT_SCALE * 1.5f, SPLAT_SCALE * 1.5f, SPLAT_SCALE * 0.3f,
                    r, g, b, 0.95f
                ))
            }
        }
    }

    private fun generateFloor(image: Bitmap, gaussians: MutableList<FloatArray>) {
        val floorY = -ROOM_HEIGHT / 2f
        val gridSize = GRID_SIZE * 2 / 3

        // Sample floor color from bottom portion of image
        val floorStartY = (image.height * 0.75f).toInt()

        for (gridZ in 0 until gridSize) {
            for (gridX in 0 until gridSize) {
                val sampleX = (gridX * image.width / gridSize).coerceIn(0, image.width - 1)
                val sampleY = (floorStartY + gridZ * (image.height - floorStartY) / gridSize)
                    .coerceIn(0, image.height - 1)

                val pixel = image.getPixel(sampleX, sampleY)
                val r = Color.red(pixel) / 255f
                val g = Color.green(pixel) / 255f
                val b = Color.blue(pixel) / 255f

                val x = (gridX.toFloat() / gridSize - 0.5f) * ROOM_WIDTH
                val z = (gridZ.toFloat() / gridSize - 0.5f) * ROOM_DEPTH

                gaussians.add(createGaussian(
                    x, floorY, z,
                    SPLAT_SCALE * 2.5f, SPLAT_SCALE * 0.2f, SPLAT_SCALE * 2.5f,
                    r, g, b, 0.9f
                ))
            }
        }
    }

    private fun generateCeiling(image: Bitmap, gaussians: MutableList<FloatArray>) {
        val ceilingY = ROOM_HEIGHT / 2f
        val gridSize = GRID_SIZE / 2

        // Sample ceiling from top portion
        val ceilingEndY = (image.height * 0.2f).toInt()

        for (gridZ in 0 until gridSize) {
            for (gridX in 0 until gridSize) {
                val sampleX = (gridX * image.width / gridSize).coerceIn(0, image.width - 1)
                val sampleY = (gridZ * ceilingEndY / gridSize).coerceIn(0, image.height - 1)

                val pixel = image.getPixel(sampleX, sampleY)
                val r = Color.red(pixel) / 255f
                val g = Color.green(pixel) / 255f
                val b = Color.blue(pixel) / 255f

                val x = (gridX.toFloat() / gridSize - 0.5f) * ROOM_WIDTH
                val z = (gridZ.toFloat() / gridSize - 0.5f) * ROOM_DEPTH

                gaussians.add(createGaussian(
                    x, ceilingY, z,
                    SPLAT_SCALE * 2f, SPLAT_SCALE * 0.3f, SPLAT_SCALE * 2f,
                    r, g, b, 0.8f
                ))
            }
        }
    }

    private fun generateSideWalls(image: Bitmap, gaussians: MutableList<FloatArray>) {
        val gridSize = GRID_SIZE / 2

        // Left wall (sample from left edge)
        val leftX = -ROOM_WIDTH / 2f
        for (gridZ in 0 until gridSize) {
            for (gridY in 0 until gridSize) {
                val sampleX = (image.width * 0.1f).toInt().coerceIn(0, image.width - 1)
                val sampleY = (gridY * image.height / gridSize).coerceIn(0, image.height - 1)

                val pixel = image.getPixel(sampleX, sampleY)
                val r = Color.red(pixel) / 255f
                val g = Color.green(pixel) / 255f
                val b = Color.blue(pixel) / 255f

                val y = (0.5f - gridY.toFloat() / gridSize) * ROOM_HEIGHT
                val z = (gridZ.toFloat() / gridSize - 0.5f) * ROOM_DEPTH

                gaussians.add(createGaussian(
                    leftX, y, z,
                    SPLAT_SCALE * 0.3f, SPLAT_SCALE * 2f, SPLAT_SCALE * 2f,
                    r, g, b, 0.7f,
                    rotY = 90f
                ))
            }
        }

        // Right wall (sample from right edge)
        val rightX = ROOM_WIDTH / 2f
        for (gridZ in 0 until gridSize) {
            for (gridY in 0 until gridSize) {
                val sampleX = (image.width * 0.9f).toInt().coerceIn(0, image.width - 1)
                val sampleY = (gridY * image.height / gridSize).coerceIn(0, image.height - 1)

                val pixel = image.getPixel(sampleX, sampleY)
                val r = Color.red(pixel) / 255f
                val g = Color.green(pixel) / 255f
                val b = Color.blue(pixel) / 255f

                val y = (0.5f - gridY.toFloat() / gridSize) * ROOM_HEIGHT
                val z = (gridZ.toFloat() / gridSize - 0.5f) * ROOM_DEPTH

                gaussians.add(createGaussian(
                    rightX, y, z,
                    SPLAT_SCALE * 0.3f, SPLAT_SCALE * 2f, SPLAT_SCALE * 2f,
                    r, g, b, 0.7f,
                    rotY = -90f
                ))
            }
        }
    }

    private fun generateBackWall(gaussians: MutableList<FloatArray>) {
        val backZ = ROOM_DEPTH / 2f
        val gridSize = GRID_SIZE / 3

        // Dark background for back wall
        for (gridY in 0 until gridSize) {
            for (gridX in 0 until gridSize) {
                val x = (gridX.toFloat() / gridSize - 0.5f) * ROOM_WIDTH
                val y = (0.5f - gridY.toFloat() / gridSize) * ROOM_HEIGHT

                // Gradient from edges to center (darker at edges)
                val distFromCenter = sqrt(
                    (gridX.toFloat() / gridSize - 0.5f).pow(2) +
                            (gridY.toFloat() / gridSize - 0.5f).pow(2)
                )
                val darkness = 0.1f + (1f - distFromCenter) * 0.2f

                gaussians.add(createGaussian(
                    x, y, backZ,
                    SPLAT_SCALE * 3f, SPLAT_SCALE * 3f, SPLAT_SCALE * 0.3f,
                    darkness, darkness, darkness, 0.5f
                ))
            }
        }
    }

    private fun createGaussian(
        x: Float, y: Float, z: Float,
        scaleX: Float, scaleY: Float, scaleZ: Float,
        r: Float, g: Float, b: Float,
        opacity: Float,
        rotY: Float = 0f
    ): FloatArray {
        // Convert rotation angle to quaternion
        val rotRad = Math.toRadians(rotY.toDouble()).toFloat()
        val rotW = cos(rotRad / 2f)
        val rotYq = sin(rotRad / 2f)

        return floatArrayOf(
            x, y, z,                    // position (0-2)
            scaleX, scaleY, scaleZ,     // scale (3-5)
            rotW, 0f, rotYq, 0f,        // rotation quaternion (6-9)
            opacity,                     // opacity (10)
            r, g, b                      // color (11-13)
        )
    }

    fun release() {
        if (nativeHandle != 0L) {
            nativeRelease(nativeHandle)
            nativeHandle = 0
        }
        isInitialized = false
    }

    // Native methods (to be implemented when NCNN model is available)
    private external fun nativeInit(
        assetManager: AssetManager,
        paramAsset: String,
        binAsset: String,
        useGpu: Boolean,
        numThreads: Int
    ): Long

    private external fun nativeInfer(handle: Long, bitmap: Bitmap): FloatArray?

    private external fun nativeRelease(handle: Long)

    /**
     * Result of Gaussian generation
     */
    data class GaussianResult(
        val params: FloatArray,      // Interleaved params: pos(3) + scale(3) + rot(4) + opacity(1) + color(3)
        val gaussianCount: Int,
        val roomWidth: Float,
        val roomHeight: Float,
        val roomDepth: Float
    ) {
        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (other !is GaussianResult) return false
            return params.contentEquals(other.params) && gaussianCount == other.gaussianCount
        }

        override fun hashCode(): Int {
            return params.contentHashCode() + gaussianCount
        }
    }
}
