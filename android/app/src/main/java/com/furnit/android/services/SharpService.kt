package com.furnit.android.services

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.sqrt

/**
 * SharpService - On-device 3D Gaussian Splat generation using NCNN.
 *
 * Singleton pattern matching iOS SHARPService.shared:
 * - Model loaded once and reused across views
 * - Lazy initialization on first use
 * - No fallbacks - NCNN only
 */
class SharpService private constructor(private val context: Context) {

    companion object {
        private const val TAG = "SharpService"
        private const val SH_C0 = 0.28209479177387814f
        private const val BYTES_PER_VERTEX = 62 * 4  // 62 floats per vertex

        @Volatile
        private var instance: SharpService? = null

        /**
         * Shared singleton instance (matches iOS SHARPService.shared)
         */
        fun getInstance(context: Context): SharpService {
            return instance ?: synchronized(this) {
                instance ?: SharpService(context.applicationContext).also {
                    instance = it
                    Log.d(TAG, "SharpService singleton created")
                }
            }
        }
    }

    // Backend instances (lazy init)
    private val ncnnSharp by lazy { NcnnSharp(context) }
    private val splitOnnxSharp by lazy { SplitOnnxSharp.getInstance(context) }
    private val onnxSharp by lazy { OnnxSharp.getInstance(context) }
    private val executorchSharp by lazy { ExecutorchSharp.getInstance(context) }
    private val litertSharp by lazy { LiteRTSharp.getInstance(context) }
    private var isInitialized = false
    private var currentBackendId: String? = null  // Track which backend was initialized
    private var useOnnx = false
    private var useSplitOnnx = false
    private var useExecutorch = false
    private var useLiteRT = false

    data class GenerationResult(
        val plyFile: File,
        val classicPlyFile: File,
        val roomWidth: Float,
        val roomHeight: Float,
        val roomDepth: Float
    )

    interface ProgressCallback {
        fun onProgress(progress: Float, message: String)
        fun onComplete(result: GenerationResult)
        fun onError(message: String)
    }

    /**
     * Check if SHARP model is available (ExecuTorch, NCNN component, NCNN full, Split ONNX, or regular ONNX)
     */
    fun isModelReady(): Boolean {
        if (splitOnnxSharp.isModelReady() || onnxSharp.isModelReady()) return true
        if (BackendConfig.ENABLE_NCNN && (ncnnSharp.isComponentModelReady() || ncnnSharp.isModelReady())) return true
        if (BackendConfig.ENABLE_EXECUTORCH && executorchSharp.isModelReady()) return true
        if (BackendConfig.ENABLE_LITERT && litertSharp.isModelReady()) return true
        return false
    }

    /**
     * Initialize model - respects user preference for NCNN vs ONNX
     * When NCNN is selected in settings, only use NCNN (no fallback)
     */
    suspend fun initialize(): Boolean {
        val prefs = context.getSharedPreferences("furnit_prefs", android.content.Context.MODE_PRIVATE)

        // Read new 3-way pref with backward compat migration
        val requestedBackend: String
        val existingBackend = prefs.getString("inference_backend", null)
        if (existingBackend != null) {
            requestedBackend = existingBackend
        } else {
            // Migrate old boolean pref
            val useNcnn = prefs.getBoolean("use_ncnn_backend", false)
            requestedBackend = if (useNcnn) "ncnn" else "onnx"
            prefs.edit()
                .putString("inference_backend", requestedBackend)
                .remove("use_ncnn_backend")
                .apply()
        }

        val inferenceBackend = BackendConfig.normalize(requestedBackend)
        if (inferenceBackend != requestedBackend) {
            Log.w(TAG, "Backend '$requestedBackend' disabled; falling back to '$inferenceBackend'")
            prefs.edit().putString("inference_backend", inferenceBackend).apply()
        }

        // Re-initialize if the user switched backends in Settings
        if (isInitialized && currentBackendId == inferenceBackend) return true
        if (isInitialized && currentBackendId != inferenceBackend) {
            Log.d(TAG, "Backend changed from '$currentBackendId' to '$inferenceBackend' — re-initializing")
            release()
        }

        // Reset runtime flags before selecting a backend.
        useOnnx = false
        useSplitOnnx = false
        useExecutorch = false
        useLiteRT = false

        if (inferenceBackend == "ncnn") {
            // User explicitly wants NCNN - no fallback to ONNX
            Log.d(TAG, "NCNN backend selected in settings")
            // Try component mode first (works correctly), then full model
            if (ncnnSharp.isComponentModelReady()) {
                try {
                    ncnnSharp.init(useGpu = false, numThreads = 4, useComponentMode = true)
                    isInitialized = true
                    Log.d(TAG, "NCNN component mode initialized successfully")
                    return true
                } catch (e: Exception) {
                    Log.e(TAG, "NCNN component init failed: ${e.message}")
                }
            }
            if (ncnnSharp.ensureModelReady()) {
                try {
                    ncnnSharp.init(useGpu = false, numThreads = 4, useComponentMode = false)
                    isInitialized = true
                    Log.d(TAG, "NCNN full model initialized successfully")
                    return true
                } catch (e: Exception) {
                    Log.e(TAG, "NCNN init failed: ${e.message}")
                    return false
                }
            } else {
                Log.e(TAG, "NCNN model files not found. Push model files to device.")
                return false
            }
        }

        if (inferenceBackend == "litert") {
            // User explicitly wants LiteRT (TFLite FP16). No fallback to ONNX.
            Log.d(TAG, "LiteRT backend selected in settings")
            if (litertSharp.isModelReady()) {
                if (litertSharp.initialize()) {
                    isInitialized = true
                    useLiteRT = true
                    currentBackendId = "litert"
                    Log.d(TAG, "LiteRT SHARP initialized successfully")
                    return true
                } else {
                    Log.e(TAG, "LiteRT SHARP init failed")
                    return false
                }
            } else {
                Log.e(TAG, "LiteRT model files not found. Push .tflite files to device.")
                return false
            }
        }

        if (inferenceBackend == "executorch") {
            // User explicitly wants ExecuTorch
            Log.d(TAG, "ExecuTorch backend selected in settings")
            if (executorchSharp.isModelReady()) {
                if (executorchSharp.initialize()) {
                    isInitialized = true
                    useExecutorch = true
                    Log.d(TAG, "ExecuTorch SHARP initialized successfully")
                    return true
                } else {
                    Log.e(TAG, "ExecuTorch SHARP init failed")
                    return false
                }
            } else {
                Log.e(TAG, "ExecuTorch SHARP model not found. Push sharp.pte to device.")
                return false
            }
        }

        // ONNX selected (or fallback from LiteRT) - use ONNX for room generation
        Log.d(TAG, "Backend: $inferenceBackend - using ONNX for room generation")

        // Try Split ONNX first (memory-efficient 4-part model)
        if (splitOnnxSharp.isModelReady()) {
            Log.d(TAG, "Split ONNX model ready - using memory-efficient 4-part inference")
            isInitialized = true
            useSplitOnnx = true
            currentBackendId = inferenceBackend
            return true
        } else {
            Log.w(TAG, "Split ONNX not ready, missing: ${splitOnnxSharp.getMissingFiles()}")
        }

        // Fall back to regular ONNX with mmap (may cause OOM on some devices)
        if (onnxSharp.ensureModelDownloaded()) {
            try {
                if (onnxSharp.initialize()) {
                    isInitialized = true
                    useOnnx = true
                    currentBackendId = inferenceBackend
                    Log.d(TAG, "ONNX with mmap initialized successfully")
                    return true
                }
            } catch (e: Exception) {
                Log.e(TAG, "ONNX init failed: ${e.message}")
            }
        }

        Log.e(TAG, "No model backend available")
        return false
    }

    /**
     * Generate 3D Gaussian splats from an image
     */
    fun generateGaussians(image: Bitmap, callback: ProgressCallback) {
        Log.d(TAG, "Starting generation: ${image.width}x${image.height}")

        Thread {
            try {
                callback.onProgress(0.1f, "Preparing...")

                // Initialize (or re-initialize if backend preference changed)
                callback.onProgress(0.15f, "Loading SHARP model...")
                val initialized = kotlinx.coroutines.runBlocking { initialize() }
                if (!initialized) {
                    callback.onError("SHARP model not available. Push model files to device.")
                    return@Thread
                }

                if (useLiteRT) {
                    // Use LiteRT backend (TFLite FP16 + GPU)
                    callback.onProgress(0.2f, "Running SHARP (LiteRT)...")
                    val result = kotlinx.coroutines.runBlocking {
                        litertSharp.inferStreaming(image) { progress, message ->
                            // LiteRTSharp reports progress in 0..1 for its internal pipeline.
                            // Map to 0.20..0.99 to keep overall progress monotonic.
                            val mapped = (0.2f + 0.79f * progress).coerceIn(0.2f, 0.99f)
                            callback.onProgress(mapped, message)
                        }
                    }

                    if (result == null) {
                        callback.onError("SHARP LiteRT inference failed")
                        return@Thread
                    }

                    Log.d(TAG, "Generated ${result.gaussianCount} Gaussians (LiteRT)")
                    Log.d(TAG, "Room: ${result.roomWidth}m x ${result.roomHeight}m x ${result.roomDepth}m")

                    saveMetadata(result.plyFile.parentFile!!, image, "sharp_litert")

                    callback.onProgress(1.0f, "Done!")
                    callback.onComplete(GenerationResult(
                        plyFile = result.plyFile,
                        classicPlyFile = result.classicPlyFile,
                        roomWidth = result.roomWidth,
                        roomHeight = result.roomHeight,
                        roomDepth = result.roomDepth
                    ))
                } else if (useExecutorch) {
                    // Use ExecuTorch backend
                    callback.onProgress(0.2f, "Running SHARP (ExecuTorch)...")
                    val result = kotlinx.coroutines.runBlocking {
                        executorchSharp.inferStreaming(image) { progress, message ->
                            callback.onProgress(progress, message)
                        }
                    }

                    if (result == null) {
                        callback.onError("SHARP ExecuTorch inference failed")
                        return@Thread
                    }

                    Log.d(TAG, "Generated ${result.gaussianCount} Gaussians (ExecuTorch)")
                    Log.d(TAG, "Room: ${result.roomWidth}m x ${result.roomHeight}m x ${result.roomDepth}m")

                    saveMetadata(result.plyFile.parentFile!!, image, "sharp_executorch")

                    callback.onProgress(1.0f, "Done!")
                    callback.onComplete(GenerationResult(
                        plyFile = result.plyFile,
                        classicPlyFile = result.classicPlyFile,
                        roomWidth = result.roomWidth,
                        roomHeight = result.roomHeight,
                        roomDepth = result.roomDepth
                    ))
                } else if (useSplitOnnx) {
                    // Use Split ONNX backend (memory-efficient 4-part model)
                    callback.onProgress(0.2f, "Running SHARP (Split ONNX - memory efficient)...")
                    val result = kotlinx.coroutines.runBlocking {
                        splitOnnxSharp.inferStreaming(image) { progress, message ->
                            callback.onProgress(progress, message)
                        }
                    }

                    if (result == null) {
                        callback.onError("SHARP Split ONNX inference failed")
                        return@Thread
                    }

                    Log.d(TAG, "Generated ${result.gaussianCount} Gaussians (Split ONNX)")
                    Log.d(TAG, "Room: ${result.roomWidth}m x ${result.roomHeight}m x ${result.roomDepth}m")

                    // Save thumbnail and metadata
                    saveMetadata(result.plyFile.parentFile!!, image, "sharp_split_onnx")

                    callback.onProgress(1.0f, "Done!")
                    callback.onComplete(GenerationResult(
                        plyFile = result.plyFile,
                        classicPlyFile = result.classicPlyFile,
                        roomWidth = result.roomWidth,
                        roomHeight = result.roomHeight,
                        roomDepth = result.roomDepth
                    ))
                } else if (useOnnx) {
                    // Use ONNX backend (streaming)
                    callback.onProgress(0.2f, "Running SHARP (ONNX)...")
                    val result = kotlinx.coroutines.runBlocking {
                        onnxSharp.inferStreaming(image) { progress, message ->
                            callback.onProgress(progress, message)
                        }
                    }

                    if (result == null) {
                        callback.onError("SHARP ONNX inference failed")
                        return@Thread
                    }

                    Log.d(TAG, "Generated ${result.gaussianCount} Gaussians (ONNX)")
                    Log.d(TAG, "Room: ${result.roomWidth}m x ${result.roomHeight}m x ${result.roomDepth}m")

                    // Save thumbnail and metadata
                    saveMetadata(result.plyFile.parentFile!!, image, "sharp_onnx")

                    callback.onProgress(1.0f, "Done!")
                    callback.onComplete(GenerationResult(
                        plyFile = result.plyFile,
                        classicPlyFile = result.classicPlyFile,
                        roomWidth = result.roomWidth,
                        roomHeight = result.roomHeight,
                        roomDepth = result.roomDepth
                    ))
                } else {
                    // Use NCNN backend
                    callback.onProgress(0.2f, "Running SHARP (NCNN)...")

                    val gaussianResult = ncnnSharp.generateGaussians(image)

                    callback.onProgress(0.6f, "Writing PLY file...")

                    Log.d(TAG, "Generated ${gaussianResult.gaussianCount} Gaussians (NCNN)")
                    Log.d(TAG, "Room: ${gaussianResult.roomWidth}m x ${gaussianResult.roomHeight}m x ${gaussianResult.roomDepth}m")

                    // Create output directory
                    val roomsDir = File(context.filesDir, "sharp_rooms")
                    roomsDir.mkdirs()

                    val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
                    val roomFolder = File(roomsDir, "room_$timestamp")
                    roomFolder.mkdirs()

                    val plyFile = File(roomFolder, "room.ply")
                    val classicPlyFile = File(roomFolder, "room_classic.ply")

                    // Write PLY file from Gaussian parameters
                    writePlyFile(plyFile, gaussianResult)
                    plyFile.copyTo(classicPlyFile, overwrite = true)

                    // Save thumbnail and metadata
                    saveMetadata(roomFolder, image, "sharp_ncnn")

                    callback.onProgress(1.0f, "Done!")
                    callback.onComplete(GenerationResult(
                        plyFile = plyFile,
                        classicPlyFile = classicPlyFile,
                        roomWidth = gaussianResult.roomWidth,
                        roomHeight = gaussianResult.roomHeight,
                        roomDepth = gaussianResult.roomDepth
                    ))
                }

            } catch (e: Exception) {
                Log.e(TAG, "Generation failed", e)
                callback.onError("Failed: ${e.message}")
            }
        }.start()
    }

    private fun saveMetadata(roomFolder: File, image: Bitmap, modelType: String) {
        // Save thumbnail
        val thumbnailFile = File(roomFolder, "thumbnail.png")
        FileOutputStream(thumbnailFile).use { out ->
            image.compress(Bitmap.CompressFormat.PNG, 90, out)
        }

        // Save metadata
        val metadataFile = File(roomFolder, "metadata.txt")
        val dateFormat = SimpleDateFormat("MMM d", Locale.getDefault())
        val roomName = "AI Room ${dateFormat.format(Date())}"
        metadataFile.writeText("name=$roomName\ncreated=${System.currentTimeMillis()}\ntype=$modelType")
    }

    /**
     * Write Gaussian parameters to PLY file in standard 3DGS format.
     */
    private fun writePlyFile(file: File, result: NcnnSharp.GaussianResult) {
        val gaussianCount = result.gaussianCount
        val params = result.params

        // Build PLY header
        val header = buildString {
            append("ply\n")
            append("format binary_little_endian 1.0\n")
            append("element vertex $gaussianCount\n")
            append("property float x\n")
            append("property float y\n")
            append("property float z\n")
            append("property float nx\n")
            append("property float ny\n")
            append("property float nz\n")
            for (i in 0 until 3) append("property float f_dc_$i\n")
            for (i in 0 until 45) append("property float f_rest_$i\n")
            append("property float opacity\n")
            append("property float scale_0\n")
            append("property float scale_1\n")
            append("property float scale_2\n")
            append("property float rot_0\n")
            append("property float rot_1\n")
            append("property float rot_2\n")
            append("property float rot_3\n")
            append("end_header\n")
        }

        FileOutputStream(file).use { fos ->
            fos.write(header.toByteArray(Charsets.UTF_8))

            val vertexBuffer = ByteBuffer.allocate(BYTES_PER_VERTEX)
            vertexBuffer.order(ByteOrder.LITTLE_ENDIAN)

            for (i in 0 until gaussianCount) {
                vertexBuffer.clear()
                val offset = i * NcnnSharp.PARAMS_PER_GAUSSIAN

                // Position (flip Y and Z for coordinate system)
                val x = params[offset + 0]
                val y = -params[offset + 1]
                val z = -params[offset + 2]
                vertexBuffer.putFloat(x)
                vertexBuffer.putFloat(y)
                vertexBuffer.putFloat(z)

                // Normals (unused)
                vertexBuffer.putFloat(0f)
                vertexBuffer.putFloat(0f)
                vertexBuffer.putFloat(0f)

                // Colors -> SH DC coefficients
                val r = params[offset + 11].coerceIn(0f, 1f)
                val g = params[offset + 12].coerceIn(0f, 1f)
                val b = params[offset + 13].coerceIn(0f, 1f)
                vertexBuffer.putFloat((r - 0.5f) / SH_C0)
                vertexBuffer.putFloat((g - 0.5f) / SH_C0)
                vertexBuffer.putFloat((b - 0.5f) / SH_C0)

                // Higher order SH (45 zeros)
                for (j in 0 until 45) vertexBuffer.putFloat(0f)

                // Opacity -> logit transform
                val rawOpacity = params[offset + 10].coerceIn(1e-4f, 1f - 1e-4f)
                vertexBuffer.putFloat(ln(rawOpacity / (1f - rawOpacity)))

                // Scale -> log transform
                val scaleBoost = 1.3f
                val minScale = 0.001f
                vertexBuffer.putFloat(ln(max(params[offset + 3] * scaleBoost, minScale)))
                vertexBuffer.putFloat(ln(max(params[offset + 4] * scaleBoost, minScale)))
                vertexBuffer.putFloat(ln(max(params[offset + 5] * scaleBoost, minScale)))

                // Rotation quaternion (normalize)
                val rw = params[offset + 6]
                val rx = params[offset + 7]
                val ry = params[offset + 8]
                val rz = params[offset + 9]
                val mag = sqrt(rw * rw + rx * rx + ry * ry + rz * rz)
                val invMag = if (mag > 1e-8f) 1f / mag else 1f
                vertexBuffer.putFloat(rw * invMag)
                vertexBuffer.putFloat(rx * invMag)
                vertexBuffer.putFloat(ry * invMag)
                vertexBuffer.putFloat(rz * invMag)

                fos.write(vertexBuffer.array())
            }
        }

        Log.d(TAG, "Wrote PLY file: ${file.absolutePath} (${file.length()} bytes)")
    }

    /**
     * Release resources
     */
    fun release() {
        when {
            useLiteRT -> litertSharp.release()
            useExecutorch -> executorchSharp.release()
            useOnnx -> onnxSharp.release()
            else -> ncnnSharp.release()
        }
        isInitialized = false
        currentBackendId = null
    }
}
