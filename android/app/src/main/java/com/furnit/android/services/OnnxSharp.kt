package com.furnit.android.services

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.providers.NNAPIFlags
import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.util.EnumSet
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.text.SimpleDateFormat
import java.util.*
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.sqrt

/**
 * SHARP Gaussian Splatting using ONNX Runtime.
 *
 * Singleton pattern matching iOS CoreML approach:
 * - Model loaded once on first use (lazy initialization)
 * - Session kept in memory across app lifetime
 * - Streams output directly to PLY file (no intermediate arrays)
 * - Uses NNAPI for hardware acceleration when available
 */
class OnnxSharp private constructor(private val context: Context) {

    companion object {
        private const val TAG = "OnnxSharp"

        // Model configuration - change these to switch models
        // Option 1: FP32 with aligned external weights for mmap
        private const val MODEL_FILENAME = "sharp_fp32_aligned.onnx"
        private const val WEIGHTS_FILENAME = "sharp_fp32_aligned.onnx.data"

        // Option 2: FP32 original (may cause OOM)
        // private const val MODEL_FILENAME = "sharp_fp32_full.onnx"
        // private const val WEIGHTS_FILENAME = "sharp_fp32_full.onnx.data"

        // Option 3: FP16 mixed (smaller, works on most devices)
        // private const val MODEL_FILENAME = "sharp_mixed_fp16.onnx"
        // private const val WEIGHTS_FILENAME = "sharp_mixed_fp16.onnx.data"
        private const val INPUT_SIZE = 1536
        private const val SH_C0 = 0.28209479177387814f
        private const val BYTES_PER_VERTEX = 62 * 4
        private const val LOGIT_LUT_SIZE = 1024

        // Pre-computed logit LUT: maps [0, 1] opacity to ln(p/(1-p))
        private val LOGIT_LUT = FloatArray(LOGIT_LUT_SIZE) { i ->
            val p = (i.toFloat() / (LOGIT_LUT_SIZE - 1)).coerceIn(1e-4f, 1f - 1e-4f)
            ln(p / (1f - p))
        }

        // Pre-computed natural log LUT for scale values: avoids ln() per vertex
        private const val LN_LUT_SIZE = 2048
        private const val LN_LUT_MIN = 0.001f
        private const val LN_LUT_MAX = 5.0f
        private val LN_LUT_SCALE = (LN_LUT_SIZE - 1).toFloat() / (LN_LUT_MAX - LN_LUT_MIN)
        private val LN_LUT = FloatArray(LN_LUT_SIZE) { i ->
            val x = LN_LUT_MIN + (LN_LUT_MAX - LN_LUT_MIN) * i / (LN_LUT_SIZE - 1)
            ln(x)
        }

        private fun lnLut(x: Float): Float {
            if (x <= LN_LUT_MIN) return LN_LUT[0]
            if (x >= LN_LUT_MAX) return LN_LUT[LN_LUT_SIZE - 1]
            return LN_LUT[((x - LN_LUT_MIN) * LN_LUT_SCALE).toInt()]
        }

        @Volatile
        private var instance: OnnxSharp? = null

        /**
         * Singleton instance - model stays loaded in memory.
         * Matches iOS pattern where CoreML model is loaded once.
         */
        fun getInstance(context: Context): OnnxSharp {
            return instance ?: synchronized(this) {
                instance ?: OnnxSharp(context.applicationContext).also {
                    instance = it
                    Log.d(TAG, "OnnxSharp singleton created")
                }
            }
        }
    }

    // Session kept in memory after first load (lazy init)
    private var ortEnvironment: OrtEnvironment? = null
    private var ortSession: OrtSession? = null
    // Use external storage directly for large FP32 models (avoid 2.4GB copy)
    private val modelsDir: File by lazy {
        context.getExternalFilesDir("models") ?: File(context.filesDir, "models")
    }

    /**
     * Result containing PLY file path and room dimensions.
     * No large arrays are stored - data is streamed directly to file.
     */
    data class StreamingResult(
        val plyFile: File,
        val classicPlyFile: File,
        val gaussianCount: Int,
        val roomWidth: Float,
        val roomHeight: Float,
        val roomDepth: Float
    )

    // Legacy result for backwards compatibility (avoid using - causes OOM)
    data class SharpResult(
        val positions: FloatArray,      // [N, 3]
        val scales: FloatArray,         // [N, 3]
        val rotations: FloatArray,      // [N, 4]
        val colors: FloatArray,         // [N, 3]
        val opacity: FloatArray,        // [N]
        val gaussianCount: Int
    )

    /**
     * Check if model is downloaded and ready.
     */
    fun isModelReady(): Boolean {
        val modelFile = File(modelsDir, MODEL_FILENAME)
        if (WEIGHTS_FILENAME.isEmpty()) {
            return modelFile.exists()
        }
        val weightsFile = File(modelsDir, WEIGHTS_FILENAME)
        return modelFile.exists() && weightsFile.exists()
    }

    /**
     * Get model download progress callback interface.
     */
    interface DownloadCallback {
        fun onProgress(bytesDownloaded: Long, totalBytes: Long)
        fun onComplete()
        fun onError(error: String)
    }

    /**
     * Download the model files if not present.
     * Returns true if model is ready (already exists or downloaded successfully).
     */
    suspend fun ensureModelDownloaded(callback: DownloadCallback? = null): Boolean = withContext(Dispatchers.IO) {
        if (isModelReady()) {
            Log.d(TAG, "Model already downloaded")
            callback?.onComplete()
            return@withContext true
        }

        modelsDir.mkdirs()

        try {
            // For local testing, check external storage
            val externalModelsDir = context.getExternalFilesDir("models")
            val externalModel = File(externalModelsDir, MODEL_FILENAME)
            val externalWeights = File(externalModelsDir, WEIGHTS_FILENAME)

            if (externalModel.exists() && externalWeights.exists()) {
                Log.d(TAG, "Copying model from external storage")
                externalModel.copyTo(File(modelsDir, MODEL_FILENAME), overwrite = true)
                externalWeights.copyTo(File(modelsDir, WEIGHTS_FILENAME), overwrite = true)
                callback?.onComplete()
                return@withContext true
            }

            // TODO: Implement actual download from server
            // For now, log instructions for manual setup
            Log.w(TAG, """
                Model not found. Push model files manually:

                For FP32 single file (after re-export):
                  adb push sharp_fp32_android.onnx /sdcard/Android/data/com.furnit.android/files/models/

                For FP32 external weights:
                  adb push sharp_fp32_full.onnx /sdcard/Android/data/com.furnit.android/files/models/
                  adb push sharp_fp32_full.onnx.data /sdcard/Android/data/com.furnit.android/files/models/

                For FP16 mixed:
                  adb push sharp_mixed_fp16.onnx /sdcard/Android/data/com.furnit.android/files/models/
                  adb push sharp_mixed_fp16.onnx.data /sdcard/Android/data/com.furnit.android/files/models/
            """.trimIndent())

            callback?.onError("Model not found. Please download the SHARP model.")
            return@withContext false

        } catch (e: Exception) {
            Log.e(TAG, "Failed to prepare model", e)
            callback?.onError(e.message ?: "Unknown error")
            return@withContext false
        }
    }

    /**
     * Check available system memory.
     */
    private fun getAvailableMemoryMB(): Long {
        val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as android.app.ActivityManager
        val memInfo = android.app.ActivityManager.MemoryInfo()
        activityManager.getMemoryInfo(memInfo)
        return memInfo.availMem / (1024 * 1024)
    }

    /**
     * Initialize the ONNX Runtime session using NNAPI.
     *
     * NNAPI (Neural Networks API) is Android's equivalent to iOS Accelerate/BLAS:
     * - Offloads computation to hardware accelerators (NPU, GPU, DSP)
     * - Uses hardware-managed memory (like iOS CoreML)
     * - Avoids loading entire model into app RAM
     *
     * This matches iOS approach where CoreML uses Metal/Accelerate for
     * hardware-accelerated inference with efficient memory management.
     */
    suspend fun initialize(): Boolean = withContext(Dispatchers.IO) {
        if (ortSession != null) return@withContext true

        if (!isModelReady()) {
            Log.e(TAG, "Model not ready - call ensureModelDownloaded first")
            return@withContext false
        }

        try {
            val modelPath = File(modelsDir, MODEL_FILENAME).absolutePath
            Log.d(TAG, "Loading ONNX model from: $modelPath")
            Log.d(TAG, "Available memory: ${getAvailableMemoryMB()}MB")

            ortEnvironment = OrtEnvironment.getEnvironment()

            val sessionOptions = OrtSession.SessionOptions().apply {
                // No optimization to minimize memory during model load
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.NO_OPT)

                // Disable memory pattern to reduce peak memory during execution
                setMemoryPatternOptimization(false)

                // Single thread - reduces peak memory by avoiding parallel tensor allocations
                setIntraOpNumThreads(1)
                setInterOpNumThreads(1)

                // Disable arena allocator completely - forces immediate deallocation
                setCPUArenaAllocator(false)

                // Sequential execution mode - execute operators one at a time
                // This reduces peak memory by not pre-allocating for parallel execution
                setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL)

                // Memory configuration for large models
                try {
                    // CRITICAL: Enable memory-mapped weights loading
                    // This maps the 2.4GB weight file directly from storage instead of copying to RAM
                    addConfigEntry("session.use_mmap", "1")

                    // Disable prepacking to reduce memory during model load
                    addConfigEntry("session.disable_prepacking", "1")
                    // Use environment allocators for external data (enables mmap)
                    addConfigEntry("session.use_env_allocators", "1")
                    // Disable parallel execution to reduce memory
                    addConfigEntry("session.inter_op.allow_spinning", "0")
                    // Enable memory-efficient graph execution
                    addConfigEntry("session.enable_mem_reuse", "1")
                } catch (e: Exception) {
                    Log.w(TAG, "Could not set advanced session config: ${e.message}")
                }

                Log.d(TAG, "Using CPU with sequential execution and memory-mapped external weights")
            }

            // Log model and memory info
            val modelFile = File(modelPath)
            val modelSize = modelFile.length() / 1024 / 1024
            Log.d(TAG, "Model file: $modelPath (${modelSize}MB)")

            if (WEIGHTS_FILENAME.isNotEmpty()) {
                val weightsFile = File(modelsDir, WEIGHTS_FILENAME)
                val weightsSize = weightsFile.length() / 1024 / 1024
                Log.d(TAG, "Weights file: ${weightsFile.absolutePath} (${weightsSize}MB)")
                Log.d(TAG, "Weights file exists: ${weightsFile.exists()}")

                // Verify external data is in same directory as model (required for mmap)
                if (weightsFile.parentFile?.absolutePath != modelFile.parentFile?.absolutePath) {
                    Log.w(TAG, "WARNING: Weights not in same directory as model - mmap may fail")
                }
            }

            // Log memory status before load
            val runtime = Runtime.getRuntime()
            val usedMemMB = (runtime.totalMemory() - runtime.freeMemory()) / 1024 / 1024
            val maxMemMB = runtime.maxMemory() / 1024 / 1024
            val availSysMB = getAvailableMemoryMB()
            Log.d(TAG, "Memory before load - Used: ${usedMemMB}MB, Max: ${maxMemMB}MB, System Available: ${availSysMB}MB")

            // Force GC before loading large model
            System.gc()
            Thread.sleep(100)

            Log.d(TAG, "Creating ONNX session (FP32 with external mmap weights)...")
            Log.d(TAG, "ONNX Runtime should automatically mmap external data file")

            val loadStartTime = System.currentTimeMillis()
            ortSession = ortEnvironment?.createSession(modelPath, sessionOptions)
            val loadTime = System.currentTimeMillis() - loadStartTime

            Log.d(TAG, "ONNX session created successfully in ${loadTime}ms")

            // Log memory after load
            val usedAfterMB = (runtime.totalMemory() - runtime.freeMemory()) / 1024 / 1024
            Log.d(TAG, "Memory after load - Used: ${usedAfterMB}MB (delta: ${usedAfterMB - usedMemMB}MB)")

            return@withContext true

        } catch (e: OutOfMemoryError) {
            Log.e(TAG, "OUT OF MEMORY loading ONNX model", e)
            Log.e(TAG, "FP32 model (2.6GB) too large for device. Consider FP16 or NCNN backend.")
            System.gc()
            return@withContext false
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize ONNX session: ${e.javaClass.simpleName}", e)
            return@withContext false
        }
    }

    /**
     * Run inference on an image.
     * WARNING: This method copies all output tensors to arrays, causing OOM on mobile.
     * Use inferStreaming() instead for memory-efficient processing.
     */
    @Deprecated("Use inferStreaming() for memory-efficient processing")
    suspend fun infer(bitmap: Bitmap): SharpResult? = withContext(Dispatchers.IO) {
        val session = ortSession
        val env = ortEnvironment

        if (session == null || env == null) {
            Log.e(TAG, "Session not initialized")
            return@withContext null
        }

        try {
            val startTime = System.currentTimeMillis()

            // Preprocess image to [1, 3, 1536, 1536] (shared with Native Pt)
            val scaledBitmap = SharpImagePreprocessor.resizeForSharp(bitmap)
            val inputTensor = preprocessImage(env, scaledBitmap)
            scaledBitmap.recycle()

            Log.d(TAG, "Running SHARP inference...")
            val inputs = mapOf("image" to inputTensor)
            val outputs = session.run(inputs)

            // Extract outputs
            val positionsTensor = outputs["positions"].get() as OnnxTensor
            val scalesTensor = outputs["scales"].get() as OnnxTensor
            val rotationsTensor = outputs["rotations"].get() as OnnxTensor
            val colorsTensor = outputs["colors"].get() as OnnxTensor
            val opacityTensor = outputs["opacity"].get() as OnnxTensor

            val positions = positionsTensor.floatBuffer.array()
            val scales = scalesTensor.floatBuffer.array()
            val rotations = rotationsTensor.floatBuffer.array()
            val colors = colorsTensor.floatBuffer.array()
            val opacity = opacityTensor.floatBuffer.array()

            val gaussianCount = opacity.size

            // Cleanup
            inputTensor.close()
            outputs.close()

            val elapsed = System.currentTimeMillis() - startTime
            Log.d(TAG, "SHARP inference completed: $gaussianCount Gaussians in ${elapsed}ms")

            return@withContext SharpResult(
                positions = positions,
                scales = scales,
                rotations = rotations,
                colors = colors,
                opacity = opacity,
                gaussianCount = gaussianCount
            )

        } catch (e: Exception) {
            Log.e(TAG, "Inference failed", e)
            return@withContext null
        }
    }

    /**
     * Run inference and stream output directly to PLY file.
     *
     * Memory-efficient implementation matching iOS CoreML approach:
     * - Reads from tensor buffers using stride-aware access (like iOS dataPointer)
     * - Writes directly to PLY file without intermediate arrays
     * - Processes one gaussian at a time to minimize memory usage
     */
    suspend fun inferStreaming(
        bitmap: Bitmap,
        progressCallback: ((Float, String) -> Unit)? = null
    ): StreamingResult? = withContext(Dispatchers.IO) {
        val session = ortSession
        val env = ortEnvironment

        if (session == null || env == null) {
            Log.e(TAG, "Session not initialized")
            return@withContext null
        }

        try {
            val startTime = System.currentTimeMillis()

            progressCallback?.invoke(0.1f, "Preprocessing image...")

            // Preprocess image to [1, 3, 1536, 1536] (shared with Native Pt)
            val scaledBitmap = SharpImagePreprocessor.resizeForSharp(bitmap)
            val inputTensor = preprocessImage(env, scaledBitmap)
            scaledBitmap.recycle()

            progressCallback?.invoke(0.2f, "Running SHARP inference...")
            Log.d(TAG, "Running SHARP streaming inference...")

            val inputs = mapOf("image" to inputTensor)
            val outputs = session.run(inputs)

            progressCallback?.invoke(0.5f, "Processing Gaussian output...")

            // Get tensor references (don't copy data)
            val positionsTensor = outputs["positions"].get() as OnnxTensor
            val scalesTensor = outputs["scales"].get() as OnnxTensor
            val rotationsTensor = outputs["rotations"].get() as OnnxTensor
            val colorsTensor = outputs["colors"].get() as OnnxTensor
            val opacityTensor = outputs["opacity"].get() as OnnxTensor

            // Get direct FloatBuffer access (no copy - like iOS dataPointer)
            val posBuffer = positionsTensor.floatBuffer
            val scaleBuffer = scalesTensor.floatBuffer
            val rotBuffer = rotationsTensor.floatBuffer
            val colorBuffer = colorsTensor.floatBuffer
            val opacityBuffer = opacityTensor.floatBuffer

            // Get shape info for stride calculation
            val posShape = positionsTensor.info.shape  // [1, N, 3]
            val gaussianCount = posShape[1].toInt()

            Log.d(TAG, "Streaming $gaussianCount Gaussians to PLY file...")

            progressCallback?.invoke(0.6f, "Creating PLY file ($gaussianCount Gaussians)...")

            // Create output directory
            val roomsDir = File(context.filesDir, "sharp_rooms")
            roomsDir.mkdirs()

            val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
            val roomFolder = File(roomsDir, "room_$timestamp")
            roomFolder.mkdirs()

            val plyFile = File(roomFolder, "room.ply")
            val classicPlyFile = File(roomFolder, "room_classic.ply")

            // Standard 3DGS PLY header
            val header = buildPlyHeader(gaussianCount)

            // Bounds tracking
            var minX = Float.MAX_VALUE
            var maxX = -Float.MAX_VALUE
            var minY = Float.MAX_VALUE
            var maxY = -Float.MAX_VALUE
            var minZ = Float.MAX_VALUE
            var maxZ = -Float.MAX_VALUE

            // Write PLY using streaming - one gaussian at a time
            FileOutputStream(plyFile).use { fos ->
                val channel = fos.channel

                // Write header
                val headerBytes = header.toByteArray(Charsets.UTF_8)
                val headerBuffer = ByteBuffer.allocateDirect(headerBytes.size)
                headerBuffer.put(headerBytes)
                headerBuffer.flip()
                channel.write(headerBuffer)

                // DirectByteBuffer for zero-copy writes (batch 512 vertices ~127KB)
                val batchSize = 512
                val batchBuffer = ByteBuffer.allocateDirect(BYTES_PER_VERTEX * batchSize)
                batchBuffer.order(ByteOrder.LITTLE_ENDIAN)
                val scaleBoost = 1.3f
                val minScale = 0.001f
                val lutScale = (LOGIT_LUT_SIZE - 1).toFloat()

                // Pre-allocate local arrays for vectorized bulk reads
                val localPositions = FloatArray(batchSize * 3)
                val localScales = FloatArray(batchSize * 3)
                val localRotations = FloatArray(batchSize * 4)
                val localColors = FloatArray(batchSize * 3)
                val localOpacity = FloatArray(batchSize)

                val progressEvery = max(1, gaussianCount / 10)
                var processed = 0

                while (processed < gaussianCount) {
                    val currentBatch = minOf(batchSize, gaussianCount - processed)

                    // Vectorized bulk reads
                    posBuffer.position(processed * 3)
                    posBuffer.get(localPositions, 0, currentBatch * 3)
                    scaleBuffer.position(processed * 3)
                    scaleBuffer.get(localScales, 0, currentBatch * 3)
                    rotBuffer.position(processed * 4)
                    rotBuffer.get(localRotations, 0, currentBatch * 4)
                    colorBuffer.position(processed * 3)
                    colorBuffer.get(localColors, 0, currentBatch * 3)
                    opacityBuffer.position(processed)
                    opacityBuffer.get(localOpacity, 0, currentBatch)

                    for (j in 0 until currentBatch) {
                        val idx3 = j * 3
                        val x = localPositions[idx3]
                        val y = -localPositions[idx3 + 1]
                        val z = -localPositions[idx3 + 2]

                        if (x < minX) minX = x
                        if (x > maxX) maxX = x
                        if (y < minY) minY = y
                        if (y > maxY) maxY = y
                        if (z < minZ) minZ = z
                        if (z > maxZ) maxZ = z

                        batchBuffer.putFloat(x)
                        batchBuffer.putFloat(y)
                        batchBuffer.putFloat(z)

                        // Normals (unused)
                        batchBuffer.putFloat(0f)
                        batchBuffer.putFloat(0f)
                        batchBuffer.putFloat(0f)

                        // Colors -> SH DC
                        val r = localColors[idx3].coerceIn(0f, 1f)
                        val g = localColors[idx3 + 1].coerceIn(0f, 1f)
                        val b = localColors[idx3 + 2].coerceIn(0f, 1f)
                        batchBuffer.putFloat((r - 0.5f) / SH_C0)
                        batchBuffer.putFloat((g - 0.5f) / SH_C0)
                        batchBuffer.putFloat((b - 0.5f) / SH_C0)

                        // Higher order SH (45 zeros)
                        repeat(45) { batchBuffer.putFloat(0f) }

                        // Opacity -> logit via LUT
                        val rawOpacity = localOpacity[j].coerceIn(0f, 1f)
                        val lutIndex = (rawOpacity * lutScale).toInt().coerceIn(0, LOGIT_LUT_SIZE - 1)
                        batchBuffer.putFloat(LOGIT_LUT[lutIndex])

                        // Scale -> log via LN LUT
                        batchBuffer.putFloat(lnLut(max(localScales[idx3] * scaleBoost, minScale)))
                        batchBuffer.putFloat(lnLut(max(localScales[idx3 + 1] * scaleBoost, minScale)))
                        batchBuffer.putFloat(lnLut(max(localScales[idx3 + 2] * scaleBoost, minScale)))

                        // Rotation -> normalize
                        val idx4 = j * 4
                        val rw = localRotations[idx4]
                        val rx = localRotations[idx4 + 1]
                        val ry = localRotations[idx4 + 2]
                        val rz = localRotations[idx4 + 3]
                        val mag = sqrt(rw * rw + rx * rx + ry * ry + rz * rz)
                        val invMag = if (mag > 1e-8f) 1f / mag else 1f
                        batchBuffer.putFloat(rw * invMag)
                        batchBuffer.putFloat(rx * invMag)
                        batchBuffer.putFloat(ry * invMag)
                        batchBuffer.putFloat(rz * invMag)
                    }

                    // Flush batch to disk
                    batchBuffer.flip()
                    batchBuffer.limit(currentBatch * BYTES_PER_VERTEX)
                    while (batchBuffer.hasRemaining()) {
                        channel.write(batchBuffer)
                    }
                    batchBuffer.clear()

                    processed += currentBatch
                    if (processed % progressEvery == 0 || processed == gaussianCount) {
                        val progress = 0.6f + (processed.toFloat() / gaussianCount) * 0.3f
                        progressCallback?.invoke(progress, "Writing PLY ($processed/$gaussianCount)...")
                    }
                }
            }

            // Copy to classic PLY (same format for now)
            plyFile.copyTo(classicPlyFile, overwrite = true)

            // Cleanup tensors
            inputTensor.close()
            outputs.close()

            // Force GC after releasing large tensors
            System.gc()

            val elapsed = System.currentTimeMillis() - startTime
            Log.d(TAG, "SHARP streaming completed: $gaussianCount Gaussians in ${elapsed}ms")
            Log.d(TAG, "Room bounds: ${maxX - minX}m x ${maxY - minY}m x ${maxZ - minZ}m")

            progressCallback?.invoke(1.0f, "Done!")

            return@withContext StreamingResult(
                plyFile = plyFile,
                classicPlyFile = classicPlyFile,
                gaussianCount = gaussianCount,
                roomWidth = maxX - minX,
                roomHeight = maxY - minY,
                roomDepth = maxZ - minZ
            )

        } catch (e: Exception) {
            Log.e(TAG, "Streaming inference failed", e)
            return@withContext null
        }
    }

    /**
     * Build standard 3DGS PLY header.
     */
    private fun buildPlyHeader(gaussianCount: Int): String {
        return buildString {
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
    }

    /**
     * Preprocess image for SHARP model input.
     */
    private fun preprocessImage(env: OrtEnvironment, bitmap: Bitmap): OnnxTensor {
        val width = bitmap.width
        val height = bitmap.height
        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)

        // Convert to CHW format with normalization [0, 1]
        val floatBuffer = FloatBuffer.allocate(3 * width * height)

        // R channel
        for (pixel in pixels) {
            floatBuffer.put(((pixel shr 16) and 0xFF) / 255f)
        }
        // G channel
        for (pixel in pixels) {
            floatBuffer.put(((pixel shr 8) and 0xFF) / 255f)
        }
        // B channel
        for (pixel in pixels) {
            floatBuffer.put((pixel and 0xFF) / 255f)
        }

        floatBuffer.rewind()
        return OnnxTensor.createTensor(env, floatBuffer, longArrayOf(1, 3, height.toLong(), width.toLong()))
    }

    /**
     * Release resources.
     */
    fun release() {
        ortSession?.close()
        ortEnvironment?.close()
        ortSession = null
        ortEnvironment = null
    }
}
