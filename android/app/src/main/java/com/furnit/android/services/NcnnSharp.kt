package com.furnit.android.services

import android.content.Context
import android.content.res.AssetManager
import android.graphics.Bitmap
import android.util.Log
import kotlin.math.max
import kotlin.math.min

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
 * Model files must be pushed to device external storage:
 *   adb push sharp.ncnn.param /sdcard/Android/data/com.furnit.android/files/models/
 *   adb push sharp.ncnn.bin /sdcard/Android/data/com.furnit.android/files/models/
 */
class NcnnSharp(private val context: Context) {

    companion object {
        private const val TAG = "NcnnSharp"

        // Model filenames
        private const val PARAM_FILENAME = "sharp.ncnn.param"
        private const val BIN_FILENAME = "sharp.ncnn.bin"

        // Input size expected by SHARP model
        const val INPUT_SIZE = 1536

        // Gaussian parameters per splat: pos(3) + scale(3) + rot(4) + opacity(1) + color(3) = 14
        const val PARAMS_PER_GAUSSIAN = 14

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
    private val modelsDir by lazy { java.io.File(context.filesDir, "models") }

    /**
     * Check if model files are available.
     */
    fun isModelReady(): Boolean {
        val paramFile = java.io.File(modelsDir, PARAM_FILENAME)
        val binFile = java.io.File(modelsDir, BIN_FILENAME)
        return paramFile.exists() && binFile.exists()
    }

    /**
     * Prepare model files by copying from external storage if needed.
     * Returns true if model is ready.
     */
    fun ensureModelReady(): Boolean {
        if (isModelReady()) {
            Log.d(TAG, "Model already available in $modelsDir")
            return true
        }

        modelsDir.mkdirs()

        // Check external storage
        val externalModelsDir = context.getExternalFilesDir("models")
        val externalParam = java.io.File(externalModelsDir, PARAM_FILENAME)
        val externalBin = java.io.File(externalModelsDir, BIN_FILENAME)

        if (externalParam.exists() && externalBin.exists()) {
            Log.d(TAG, "Copying NCNN model from external storage")
            try {
                externalParam.copyTo(java.io.File(modelsDir, PARAM_FILENAME), overwrite = true)
                externalBin.copyTo(java.io.File(modelsDir, BIN_FILENAME), overwrite = true)
                Log.d(TAG, "Model copied successfully")
                return true
            } catch (e: Exception) {
                Log.e(TAG, "Failed to copy model: ${e.message}")
            }
        }

        Log.w(TAG, """
            Model not found. Push model files manually:
            adb push sharp.ncnn.param /sdcard/Android/data/com.furnit.android/files/models/
            adb push sharp.ncnn.bin /sdcard/Android/data/com.furnit.android/files/models/
        """.trimIndent())

        return false
    }

    /**
     * Initialize NCNN model.
     * Model files must be in filesDir/models/.
     * Throws exception if library not available or model fails to load.
     */
    fun init(
        useGpu: Boolean = false,
        numThreads: Int = 4
    ): Boolean {
        if (!libraryLoaded) {
            throw UnsatisfiedLinkError("SHARP NCNN library not available: $libraryLoadError")
        }

        if (!isModelReady()) {
            throw IllegalStateException("Model not ready. Call ensureModelReady() first.")
        }

        val paramPath = java.io.File(modelsDir, PARAM_FILENAME).absolutePath
        val binPath = java.io.File(modelsDir, BIN_FILENAME).absolutePath

        try {
            Log.i(TAG, "Loading SHARP NCNN model from: $modelsDir")
            nativeHandle = nativeInitFromPath(
                paramPath,
                binPath,
                useGpu,
                numThreads
            )
            isInitialized = nativeHandle != 0L
            if (!isInitialized) {
                throw RuntimeException("Failed to load SHARP NCNN model")
            }
            Log.i(TAG, "NCNN SHARP initialized successfully")
            return true
        } catch (e: Exception) {
            Log.e(TAG, "NCNN init failed: ${e.message}", e)
            throw e
        }
    }

    /**
     * Generate Gaussians from image using NCNN.
     * Throws exception if model is not initialized.
     */
    fun generateGaussians(bitmap: Bitmap): GaussianResult {
        if (!isInitialized || nativeHandle == 0L) {
            throw IllegalStateException("SHARP NCNN model not initialized. Call init() first.")
        }
        return generateGaussiansNative(bitmap)
    }

    /**
     * Native NCNN inference
     */
    private fun generateGaussiansNative(bitmap: Bitmap): GaussianResult {
        val scaledBitmap = if (bitmap.width != INPUT_SIZE || bitmap.height != INPUT_SIZE) {
            Log.d(TAG, "Scaling bitmap from ${bitmap.width}x${bitmap.height} to ${INPUT_SIZE}x${INPUT_SIZE}")
            Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)
        } else {
            bitmap
        }

        Log.d(TAG, "Running NCNN inference...")
        val rawParams = nativeInfer(nativeHandle, scaledBitmap)

        if (rawParams == null || rawParams.isEmpty()) {
            throw RuntimeException("SHARP NCNN inference failed - no output")
        }

        val gaussianCount = rawParams.size / PARAMS_PER_GAUSSIAN
        Log.d(TAG, "NCNN inference generated $gaussianCount Gaussians")

        // Calculate bounding box from positions
        var minX = Float.MAX_VALUE
        var maxX = Float.MIN_VALUE
        var minY = Float.MAX_VALUE
        var maxY = Float.MIN_VALUE
        var minZ = Float.MAX_VALUE
        var maxZ = Float.MIN_VALUE

        for (i in 0 until gaussianCount) {
            val offset = i * PARAMS_PER_GAUSSIAN
            val x = rawParams[offset + 0]
            val y = rawParams[offset + 1]
            val z = rawParams[offset + 2]
            minX = min(minX, x)
            maxX = max(maxX, x)
            minY = min(minY, y)
            maxY = max(maxY, y)
            minZ = min(minZ, z)
            maxZ = max(maxZ, z)
        }

        val roomWidth = maxX - minX
        val roomHeight = maxY - minY
        val roomDepth = maxZ - minZ

        Log.d(TAG, "Room bounds: ${roomWidth}x${roomHeight}x${roomDepth} meters")

        return GaussianResult(
            params = rawParams,
            gaussianCount = gaussianCount,
            roomWidth = roomWidth,
            roomHeight = roomHeight,
            roomDepth = roomDepth
        )
    }

    fun release() {
        if (nativeHandle != 0L) {
            nativeRelease(nativeHandle)
            nativeHandle = 0
        }
        isInitialized = false
    }

    // Native methods
    private external fun nativeInit(
        assetManager: AssetManager,
        paramAsset: String,
        binAsset: String,
        useGpu: Boolean,
        numThreads: Int
    ): Long

    private external fun nativeInitFromPath(
        paramPath: String,
        binPath: String,
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
