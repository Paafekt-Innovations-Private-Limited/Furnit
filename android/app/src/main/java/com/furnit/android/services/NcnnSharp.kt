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

        // Model filenames (full model)
        private const val PARAM_FILENAME = "sharp.ncnn.param"
        private const val BIN_FILENAME = "sharp.ncnn.bin"

        // Component model filenames
        private val COMPONENT_FILES = listOf(
            "sharp_single_patch_embed.ncnn.param",
            "sharp_single_patch_embed.ncnn.bin",
            "sharp_single_patch_encoder.ncnn.param",
            "sharp_single_patch_encoder.ncnn.bin",
            "sharp_image_encoder.ncnn.param",
            "sharp_image_encoder.ncnn.bin",
            "gaussian_head.ncnn.param",
            "gaussian_head.ncnn.bin",
            "patch_cls_token.bin",
            "patch_pos_embed.bin"
        )

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
    private var componentHandle: Long = 0
    private var isInitialized = false
    private var useComponentMode = false
    private val modelsDir by lazy { java.io.File(context.filesDir, "models") }

    /**
     * Check if full model files are available.
     */
    fun isModelReady(): Boolean {
        val paramFile = java.io.File(modelsDir, PARAM_FILENAME)
        val binFile = java.io.File(modelsDir, BIN_FILENAME)
        return paramFile.exists() && binFile.exists()
    }

    /**
     * Check if component model files are available (in internal OR external storage).
     */
    fun isComponentModelReady(): Boolean {
        // Check internal storage first
        val internalReady = COMPONENT_FILES.all { filename ->
            java.io.File(modelsDir, filename).exists()
        }
        if (internalReady) return true

        // Check external storage (files will be copied on init)
        val externalModelsDir = context.getExternalFilesDir("models") ?: return false
        return COMPONENT_FILES.all { filename ->
            java.io.File(externalModelsDir, filename).exists()
        }
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
     *
     * @param useComponentMode If true, use component-based inference (slower but works).
     *                         If false, use full model (may crash due to channel mismatch).
     */
    fun init(
        useGpu: Boolean = false,
        numThreads: Int = 4,
        useComponentMode: Boolean = false
    ): Boolean {
        if (!libraryLoaded) {
            throw UnsatisfiedLinkError("SHARP NCNN library not available: $libraryLoadError")
        }

        this.useComponentMode = useComponentMode

        if (useComponentMode) {
            return initComponents()
        } else {
            return initFullModel(useGpu, numThreads)
        }
    }

    /**
     * Initialize component-based model (slower but works correctly).
     */
    private fun initComponents(): Boolean {
        // Always ensure component files are in internal storage
        val internalReady = COMPONENT_FILES.all { filename ->
            java.io.File(modelsDir, filename).exists()
        }

        if (!internalReady) {
            Log.d(TAG, "Component models not in internal storage. Copying from external...")
            val externalModelsDir = context.getExternalFilesDir("models")
            if (externalModelsDir != null) {
                modelsDir.mkdirs()
                var allCopied = true
                for (filename in COMPONENT_FILES) {
                    val external = java.io.File(externalModelsDir, filename)
                    val internal = java.io.File(modelsDir, filename)
                    if (external.exists() && !internal.exists()) {
                        try {
                            Log.d(TAG, "Copying $filename (${external.length() / 1024 / 1024} MB)...")
                            external.copyTo(internal)
                            Log.d(TAG, "Copied $filename")
                        } catch (e: Exception) {
                            Log.e(TAG, "Failed to copy $filename: ${e.message}")
                            allCopied = false
                        }
                    } else if (!external.exists()) {
                        Log.w(TAG, "Missing external file: $filename")
                        allCopied = false
                    }
                }
                if (!allCopied) {
                    throw IllegalStateException(
                        "Component models not available. Push files to: ${externalModelsDir.absolutePath}"
                    )
                }
            } else {
                throw IllegalStateException("External storage not available")
            }
        } else {
            Log.d(TAG, "Component models already in internal storage")
        }

        try {
            Log.i(TAG, "Loading SHARP NCNN components from: $modelsDir")
            componentHandle = nativeInitComponents(modelsDir.absolutePath)
            isInitialized = componentHandle != 0L
            if (!isInitialized) {
                throw RuntimeException("Failed to load SHARP NCNN components")
            }
            Log.i(TAG, "NCNN SHARP components initialized successfully")
            return true
        } catch (e: Exception) {
            Log.e(TAG, "Component init failed: ${e.message}", e)
            throw e
        }
    }

    /**
     * Initialize full model (fast but may crash due to channel mismatch).
     */
    private fun initFullModel(useGpu: Boolean, numThreads: Int): Boolean {
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
        val handleReady = if (useComponentMode) componentHandle != 0L else nativeHandle != 0L
        if (!isInitialized || !handleReady) {
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

        Log.d(TAG, "Running NCNN inference (component mode: $useComponentMode)...")
        val rawParams = if (useComponentMode) {
            nativeInferComponents(componentHandle, scaledBitmap)
        } else {
            nativeInfer(nativeHandle, scaledBitmap)
        }

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
        if (componentHandle != 0L) {
            nativeReleaseComponents(componentHandle)
            componentHandle = 0
        }
        isInitialized = false
        useComponentMode = false
    }

    // Native methods - Full model
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

    // Native methods - Component mode
    private external fun nativeInitComponents(modelDir: String): Long

    private external fun nativeInferComponents(handle: Long, bitmap: Bitmap): FloatArray?

    private external fun nativeReleaseComponents(handle: Long)

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
