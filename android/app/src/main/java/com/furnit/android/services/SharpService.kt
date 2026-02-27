package com.furnit.android.services

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.atomic.AtomicBoolean
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
        private const val PLY_BATCH_SIZE = 512
        private const val LOGIT_LUT_SIZE = 1024
        private const val LN_LUT_SIZE = 2048
        private const val LN_LUT_MIN = 0.001f
        private const val LN_LUT_MAX = 5.0f

        // Pre-computed logit LUT (matches ONNX): maps [0,1] opacity to ln(p/(1-p))
        private val LOGIT_LUT = FloatArray(LOGIT_LUT_SIZE) { i ->
            val p = (i.toFloat() / (LOGIT_LUT_SIZE - 1)).coerceIn(1e-4f, 1f - 1e-4f)
            ln(p / (1f - p))
        }
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
        private val ZERO_SH_BLOCK = ByteArray(45 * 4)

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
    // private val pythonSharp by lazy { PythonSharp.getInstance(context) }  // Needs Chaquopy
    private val torchMobileSharp by lazy { TorchMobileSharp.getInstance(context) }
    private val nativePtSharp by lazy { NativePtSharp.getInstance(context) }
    private val onnxInt8Sharp by lazy { OnnxInt8Sharp.getInstance(context) }
    private val splitOnnxFp16Sharp by lazy { SplitOnnxFp16Sharp.getInstance(context) }
    private val executorchFp16Sharp by lazy { ExecutorchFp16Sharp.getInstance(context) }
    private val executorchInt8Sharp by lazy { ExecutorchInt8Sharp.getInstance(context) }
    private val zeroSHBuffer: ByteBuffer by lazy {
        ByteBuffer.allocateDirect(45 * 4).apply { order(ByteOrder.LITTLE_ENDIAN) }
    }
    private val plyBatch: ByteBuffer by lazy {
        ByteBuffer.allocateDirect(BYTES_PER_VERTEX * PLY_BATCH_SIZE).apply { order(ByteOrder.LITTLE_ENDIAN) }
    }
    private var isInitialized = false
    private var currentBackendId: String? = null
    private var useOnnx = false
    private var useSplitOnnx = false
    private var useOnnxInt8 = false
    private var useOnnxFp16 = false
    private var useExecutorch = false
    private var useExecutorchFp16 = false
    private var useExecutorchInt8 = false
    private var usePython = false
    private var useLiteRT = false
    private var useNativePt = false
    /** When initialize() returns false, holds a user-facing message (e.g. which files to push). */
    private var lastInitFailureMessage: String? = null

    data class GenerationResult(
        val plyFile: File,
        val classicPlyFile: File,
        val roomWidth: Float,
        val roomHeight: Float,
        val roomDepth: Float,
        val roomCenterX: Float? = null,
        val roomCenterY: Float? = null,
        val roomCenterZ: Float? = null
    )

    interface ProgressCallback {
        fun onProgress(progress: Float, message: String)
        fun onComplete(result: GenerationResult)
        fun onError(message: String)
    }

    /** Handle to cancel a background generation and release resources when user chooses non-AI path. */
    interface GenerationHandle {
        fun cancel()
    }

    /**
     * Preload ExecuTorch Part1 encoder when user opens the SHARP screen.
     * Runs warmup forward to hide "stuck at 5%" stall at Generate time.
     */
    suspend fun preloadSharpModels() = withContext(Dispatchers.IO) {
        val prefs = context.getSharedPreferences("furnit_prefs", Context.MODE_PRIVATE)
        val backend = prefs.getString("inference_backend", "executorch_int8") ?: "executorch_int8"
        val effective = BackendConfig.normalize(backend)
        when {
            effective == "executorch" && BackendConfig.ENABLE_EXECUTORCH && executorchSharp.isModelReady() -> {
                Log.d(TAG, "Preloading ExecuTorch Part1...")
                executorchSharp.preloadAndWarmup()
                Log.d(TAG, "ExecuTorch preload done")
            }
            effective == "executorch_fp16" && BackendConfig.ENABLE_EXECUTORCH_FP16 && executorchFp16Sharp.isModelReady() -> {
                Log.d(TAG, "Preloading ExecuTorch FP16 Part1...")
                executorchFp16Sharp.preloadAndWarmup()
                Log.d(TAG, "ExecuTorch FP16 preload done")
            }
            effective == "executorch_int8" && BackendConfig.ENABLE_EXECUTORCH_INT8 -> {
                Log.d(TAG, "ExecuTorch INT8 backend selected – no explicit preload step")
            }
            effective == "onnx" && splitOnnxSharp.isModelReady() -> {
                Log.d(TAG, "Preloading Split ONNX sessions (all 4 parts)...")
                splitOnnxSharp.preloadSessions()
                Log.d(TAG, "Split ONNX preload done")
            }
            effective == "onnx_fp16" && BackendConfig.ENABLE_ONNX_FP16 && splitOnnxFp16Sharp.isModelReady() -> {
                Log.d(TAG, "Preloading ONNX FP16 Part 1...")
                splitOnnxFp16Sharp.preloadSessions()
                Log.d(TAG, "ONNX FP16 preload done")
            }
            effective == "onnx_int8" && BackendConfig.ENABLE_ONNX_INT8 -> {
                Log.d(TAG, "Preloading ONNX INT8 single model session...")
                onnxInt8Sharp.initialize()
                Log.d(TAG, "ONNX INT8 preload done")
            }
            else -> { /* ncnn, litert, native_pt, etc.: no preload */ }
        }
        // Native .pt: skip preload — full 2.5GB model OOMs; split mode loads Part1..4 on-demand if not preloaded.
    }

    /**
     * Check if SHARP model is available (ExecuTorch, NCNN component, NCNN full, Split ONNX, or regular ONNX)
     */
    fun isModelReady(): Boolean {
        if (splitOnnxSharp.isModelReady() || onnxSharp.isModelReady()) return true
        if (BackendConfig.ENABLE_ONNX_INT8 && onnxInt8Sharp.isModelReady()) return true
        if (BackendConfig.ENABLE_ONNX_FP16 && splitOnnxFp16Sharp.isModelReady()) return true
        if (BackendConfig.ENABLE_NCNN && (ncnnSharp.isComponentModelReady() || ncnnSharp.isModelReady())) return true
        if (BackendConfig.ENABLE_EXECUTORCH && executorchSharp.isModelReady()) return true
        if (BackendConfig.ENABLE_EXECUTORCH_FP16 && executorchFp16Sharp.isModelReady()) return true
        if (BackendConfig.ENABLE_EXECUTORCH_INT8) return true
        if (BackendConfig.ENABLE_LITERT && litertSharp.isModelReady()) return true
        if (BackendConfig.ENABLE_NATIVE_PT && nativePtSharp.isModelReady()) return true
        return false
    }

    /**
     * Initialize model - respects user preference for NCNN vs ONNX
     * When NCNN is selected in settings, only use NCNN (no fallback)
     */
    suspend fun initialize(): Boolean {
        lastInitFailureMessage = null
        val prefs = context.getSharedPreferences("furnit_prefs", android.content.Context.MODE_PRIVATE)

        // Read new 3-way pref with backward compat migration
        val requestedBackend: String
        val existingBackend = prefs.getString("inference_backend", null)
        if (existingBackend != null) {
            requestedBackend = existingBackend
        } else {
            // Migrate old boolean pref
            val useNcnn = prefs.getBoolean("use_ncnn_backend", false)
            requestedBackend = if (useNcnn) "ncnn" else "executorch_int8"
            prefs.edit()
                .putString("inference_backend", requestedBackend)
                .remove("use_ncnn_backend")
                .apply()
        }

        var effectiveBackend = BackendConfig.normalize(requestedBackend)
        if (effectiveBackend != requestedBackend) {
            Log.w(TAG, "Backend '$requestedBackend' disabled; falling back to '$effectiveBackend'")
            prefs.edit().putString("inference_backend", effectiveBackend).apply()
        }

        // Re-initialize if the user switched backends in Settings
        if (isInitialized && currentBackendId == effectiveBackend) return true
        if (isInitialized && currentBackendId != effectiveBackend) {
            Log.d(TAG, "Backend changed from '$currentBackendId' to '$effectiveBackend' — re-initializing")
            release()
        }

        // Reset runtime flags before selecting a backend.
        useOnnx = false
        useSplitOnnx = false
        useOnnxInt8 = false
        useOnnxFp16 = false
        useExecutorch = false
        useExecutorchFp16 = false
        useExecutorchInt8 = false
        useLiteRT = false
        usePython = false
        useNativePt = false

        if (effectiveBackend == "native_pt") {
            Log.d(TAG, "Native .pt backend selected (no fallback)")
            if (nativePtSharp.isModelReady()) {
                if (nativePtSharp.initialize()) {
                    isInitialized = true
                    useNativePt = true
                    currentBackendId = "native_pt"
                    Log.d(TAG, "Native .pt SHARP initialized successfully")
                    return true
                }
            }
            Log.e(TAG, "Native .pt engine not available. Push sharp_scripted.ptl to device. No fallback.")
            return false
        }

        if (effectiveBackend == "python") {
            Log.w(TAG, "Python backend not available (needs Chaquopy + ARM PyTorch). Falling back to ONNX.")
            effectiveBackend = "onnx"
        }

        if (effectiveBackend == "torch_mobile") {
            Log.d(TAG, "PyTorch Mobile backend selected -- pre-loading model")
            if (torchMobileSharp.isModelReady()) {
                // Initialize pre-loads the model into memory NOW
                // so it's ready when user taps Generate
                if (torchMobileSharp.initialize()) {
                    isInitialized = true
                    currentBackendId = "torch_mobile"
                    Log.d(TAG, "PyTorch Mobile initialized + pre-loaded successfully")
                    return true
                }
            }
            Log.w(TAG, "PyTorch Mobile model not found. Falling back to ONNX.")
            effectiveBackend = "onnx"
        }

        if (effectiveBackend == "ncnn") {
            Log.d(TAG, "NCNN backend selected in settings")
            // Component mode only - full model hangs at conv_106
            if (ncnnSharp.isComponentModelReady()) {
                try {
                    ncnnSharp.init(useGpu = true, numThreads = 4, useComponentMode = true)
                    isInitialized = true
                    Log.d(TAG, "NCNN component mode initialized successfully")
                    return true
                } catch (e: Exception) {
                    Log.e(TAG, "NCNN component init failed: ${e.message}")
                }
            }
            // No component files: fall back to ONNX instead of full model (avoids hang)
            Log.w(TAG, "NCNN component files not found. Falling back to ONNX.")
            effectiveBackend = "onnx"
        }

        if (effectiveBackend == "litert") {
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

        if (effectiveBackend == "executorch") {
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

        if (effectiveBackend == "executorch_fp16") {
            Log.d(TAG, "ExecuTorch FP16 backend selected in settings")
            if (executorchFp16Sharp.isModelReady()) {
                if (executorchFp16Sharp.initialize()) {
                    isInitialized = true
                    useExecutorchFp16 = true
                    currentBackendId = "executorch_fp16"
                    Log.d(TAG, "ExecuTorch FP16 SHARP initialized successfully")
                    return true
                } else {
                    Log.e(TAG, "ExecuTorch FP16 SHARP init failed")
                    return false
                }
            } else {
                val missing = executorchFp16Sharp.getMissingFiles()
                Log.e(TAG, "ExecuTorch FP16 models not found, missing: $missing")
                lastInitFailureMessage = "FP16 .pte models not found. Push sharp_split_part*_fp16.pte to ${executorchFp16Sharp.getModelsDirPath()}"
                return false
            }
        }

        // ExecuTorch INT8 backend
        if (effectiveBackend == "executorch_int8") {
            Log.d(TAG, "ExecuTorch INT8 backend selected in settings")
            if (executorchInt8Sharp.initialize()) {
                isInitialized = true
                useExecutorchInt8 = true
                currentBackendId = "executorch_int8"
                Log.d(TAG, "ExecuTorch INT8 SHARP initialized successfully")
                return true
            } else {
                Log.e(TAG, "ExecuTorch INT8 SHARP init failed")
                return false
            }
        }

        // ONNX INT8 backend (single model)
        if (effectiveBackend == "onnx_int8") {
            if (onnxInt8Sharp.isModelReady()) {
                if (onnxInt8Sharp.initialize()) {
                    Log.d(TAG, "ONNX INT8 single model initialized successfully")
                    isInitialized = true
                    useOnnxInt8 = true
                    currentBackendId = "onnx_int8"
                    return true
                } else {
                    Log.e(TAG, "ONNX INT8 session init failed")
                    lastInitFailureMessage = "INT8 model init failed (OOM?). Check logcat for details."
                    return false
                }
            } else {
                Log.e(TAG, "ONNX INT8 not ready. Push sharp_single_int8.onnx + .data to ${onnxInt8Sharp.getModelsDirPath()}")
                lastInitFailureMessage = "INT8 model not found. Push sharp_single_int8.onnx and sharp_single_int8.onnx.data to ${onnxInt8Sharp.getModelsDirPath()}"
                return false
            }
        }

        // ONNX FP16 split backend
        if (effectiveBackend == "onnx_fp16") {
            if (splitOnnxFp16Sharp.isModelReady()) {
                Log.d(TAG, "ONNX FP16 split model ready — using 4-part FP16 inference")
                isInitialized = true
                useOnnxFp16 = true
                currentBackendId = "onnx_fp16"
                return true
            } else {
                val missing = splitOnnxFp16Sharp.getMissingFiles()
                Log.e(TAG, "ONNX FP16 not ready, missing: $missing")
                lastInitFailureMessage = "FP16 models not found. Push sharp_part*_fp16.onnx to ${splitOnnxFp16Sharp.getModelsDirPath()}"
                return false
            }
        }

        // ONNX FP32 selected (or fallback from NCNN)
        Log.d(TAG, "Backend: $effectiveBackend - using ONNX for room generation")
        Log.d(TAG, "ONNX init: splitOnnxSharp.isModelReady=${splitOnnxSharp.isModelReady()} onnxSharp.isModelReady=${onnxSharp.isModelReady()}")

        // Try Split ONNX first (memory-efficient 4-part model)
        if (splitOnnxSharp.isModelReady()) {
            Log.d(TAG, "Split ONNX model ready - using memory-efficient 4-part inference")
            isInitialized = true
            useSplitOnnx = true
            currentBackendId = effectiveBackend
            return true
        } else {
            val missing = splitOnnxSharp.getMissingFiles()
            val path = splitOnnxSharp.getModelsDirPath()
            Log.w(TAG, "Split ONNX not ready, missing: $missing")
            Log.w(TAG, "Push split ONNX to device: $path")
            Log.w(TAG, "Example: adb push sharp_part1.onnx $path/ && adb push sharp_part1.onnx.data $path/  (repeat for part2, part3, part4)")
        }

        // Fall back to regular ONNX with mmap (may cause OOM on some devices)
        if (onnxSharp.ensureModelDownloaded()) {
            try {
                if (onnxSharp.initialize()) {
                    isInitialized = true
                    useOnnx = true
                    currentBackendId = effectiveBackend
                    Log.d(TAG, "ONNX with mmap initialized successfully")
                    return true
                }
            } catch (e: Exception) {
                Log.e(TAG, "ONNX init failed: ${e.message}")
            }
        }

        Log.e(TAG, "No model backend available")
        val path = splitOnnxSharp.getModelsDirPath()
        val missing = splitOnnxSharp.getMissingFiles()
        lastInitFailureMessage = "SHARP model not found. Push split ONNX files to device:\n" +
            "  $path\n" +
            "  Files: ${missing.joinToString(", ")}\n" +
            "  Example: adb push sharp_part1.onnx $path/"
        return false
    }

    private val generationCancelled = AtomicBoolean(false)

    /**
     * Start generation in background. Returns a handle to cancel. When user chooses Manual/Back,
     * call handle.cancel() then release() to free memory.
     */
    fun startGenerationInBackground(image: Bitmap, callback: ProgressCallback): GenerationHandle {
        generationCancelled.set(false)
        val handle = object : GenerationHandle {
            override fun cancel() {
                generationCancelled.set(true)
                Log.d(TAG, "Generation cancelled by user")
            }
        }
        Thread {
            generateGaussiansInternal(image, callback) { generationCancelled.get() }
        }.start()
        return handle
    }

    /**
     * Generate 3D Gaussian splats from an image (blocks until complete).
     * For background start with cancellation, use startGenerationInBackground.
     */
    fun generateGaussians(image: Bitmap, callback: ProgressCallback) {
        generationCancelled.set(false)
        Thread {
            generateGaussiansInternal(image, callback) { false }
        }.start()
    }

    private fun generateGaussiansInternal(image: Bitmap, callback: ProgressCallback, isCancelled: () -> Boolean) {
        Log.d(TAG, "Starting generation: ${image.width}x${image.height}")

        try {
                callback.onProgress(0.1f, "Preparing...")

                // Initialize (or re-initialize if backend preference changed)
                callback.onProgress(0.15f, "Loading SHARP model...")
                val initialized = kotlinx.coroutines.runBlocking { initialize() }
                if (!initialized) {
                    callback.onError(lastInitFailureMessage ?: "SHARP model not available. Push model files to device.")
                    return
                }
                if (isCancelled()) return

                if (currentBackendId == "torch_mobile") {
                    callback.onProgress(0.2f, "Running SHARP (PyTorch Mobile)...")
                    val result = kotlinx.coroutines.runBlocking {
                        torchMobileSharp.inferStreaming(image) { progress, message ->
                            val mapped = (0.2f + 0.79f * progress).coerceIn(0.2f, 0.99f)
                            callback.onProgress(mapped, message)
                        }
                    }
                    if (result == null) {
                        callback.onError("PyTorch Mobile inference failed")
                        return
                    }
                    Log.d(TAG, "Generated ${result.gaussianCount} Gaussians (PyTorch Mobile)")
                    saveMetadata(result.plyFile.parentFile!!, image, "sharp_torch_mobile")
                    callback.onProgress(1.0f, "Done!")
                    callback.onComplete(GenerationResult(
                        plyFile = result.plyFile,
                        classicPlyFile = result.classicPlyFile,
                        roomWidth = result.roomWidth,
                        roomHeight = result.roomHeight,
                        roomDepth = result.roomDepth
                    ))
                } else if (useLiteRT) {
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
                        return
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
                } else if (useNativePt) {
                    callback.onProgress(0.2f, "Running SHARP (Native .pt - LibTorch)...")
                    val result = kotlinx.coroutines.runBlocking {
                        nativePtSharp.inferStreaming(image, { progress, message ->
                            val mapped = (0.2f + 0.79f * progress).coerceIn(0.2f, 0.99f)
                            callback.onProgress(mapped, message)
                        }, isCancelled)
                    }
                    if (result == null) {
                        if (isCancelled()) return
                        callback.onError("Native .pt inference failed")
                        return
                    }
                    Log.d(TAG, "Generated ${result.gaussianCount} Gaussians (Native .pt)")
                    saveMetadata(result.plyFile.parentFile!!, image, "sharp_native_pt")
                    callback.onProgress(1.0f, "Done!")
                    callback.onComplete(GenerationResult(
                        plyFile = result.plyFile,
                        classicPlyFile = result.classicPlyFile,
                        roomWidth = result.roomWidth,
                        roomHeight = result.roomHeight,
                        roomDepth = result.roomDepth
                    ))
                } else if (useExecutorch) {
                    callback.onProgress(0.2f, "Running SHARP (ExecuTorch)...")
                    val result = kotlinx.coroutines.runBlocking {
                        executorchSharp.inferStreaming(image) { progress, message ->
                            callback.onProgress(progress, message)
                        }
                    }

                    if (result == null) {
                        callback.onError("SHARP ExecuTorch inference failed")
                        return
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
                } else if (useExecutorchFp16) {
                    Log.d(TAG, "generateGaussians: invoking ExecuTorch FP16 inferStreaming")
                    callback.onProgress(0.2f, "Running SHARP (ExecuTorch FP16)...")
                    val result = kotlinx.coroutines.runBlocking {
                        executorchFp16Sharp.inferStreaming(image) { progress, message ->
                            callback.onProgress(progress, message)
                        }
                    }

                    if (result == null) {
                        callback.onError("SHARP ExecuTorch FP16 inference failed")
                        return
                    }

                    Log.d(TAG, "Generated ${result.gaussianCount} Gaussians (ExecuTorch FP16)")
                    Log.d(TAG, "Room: ${result.roomWidth}m x ${result.roomHeight}m x ${result.roomDepth}m")

                    saveMetadata(result.plyFile.parentFile!!, image, "sharp_executorch_fp16")

                    callback.onProgress(1.0f, "Done!")
                    callback.onComplete(GenerationResult(
                        plyFile = result.plyFile,
                        classicPlyFile = result.classicPlyFile,
                        roomWidth = result.roomWidth,
                        roomHeight = result.roomHeight,
                        roomDepth = result.roomDepth
                    ))
                } else if (useExecutorchInt8) {
                    Log.d(TAG, "generateGaussians: invoking ExecuTorch INT8 inferStreaming")
                    callback.onProgress(0.2f, "Running SHARP (ExecuTorch INT8)...")
                    val result = kotlinx.coroutines.runBlocking {
                        executorchInt8Sharp.inferStreaming(image) { progress, message ->
                            val mapped = (0.2f + 0.79f * progress).coerceIn(0.2f, 0.99f)
                            callback.onProgress(mapped, message)
                        }
                    }

                    if (result == null) {
                        callback.onError("SHARP ExecuTorch INT8 inference failed")
                        return
                    }

                    Log.d(TAG, "Generated ${result.gaussianCount} Gaussians (ExecuTorch INT8)")
                    Log.d(TAG, "Room: ${result.roomWidth}m x ${result.roomHeight}m x ${result.roomDepth}m")
                    val isPortraitFeed = image.height > image.width
                    Log.d(TAG, "VIEWER_FEED isPortrait=$isPortraitFeed roomWidth=${result.roomWidth} roomHeight=${result.roomHeight} roomDepth=${result.roomDepth} path=${result.plyFile.parentFile?.absolutePath}")

                    saveMetadata(result.plyFile.parentFile!!, image, "sharp_executorch_int8")

                    callback.onProgress(1.0f, "Done!")
                    callback.onComplete(GenerationResult(
                        plyFile = result.plyFile,
                        classicPlyFile = result.classicPlyFile,
                        roomWidth = result.roomWidth,
                        roomHeight = result.roomHeight,
                        roomDepth = result.roomDepth,
                        roomCenterX = result.roomCenterX,
                        roomCenterY = result.roomCenterY,
                        roomCenterZ = result.roomCenterZ
                    ))
                } else if (useOnnxInt8) {
                    Log.d(TAG, "generateGaussians: invoking ONNX INT8 inferStreaming")
                    callback.onProgress(0.2f, "Running SHARP (ONNX INT8)...")
                    val result = kotlinx.coroutines.runBlocking {
                        onnxInt8Sharp.inferStreaming(image, { progress, message ->
                            callback.onProgress(progress, message)
                        }, isCancelled)
                    }

                    if (result == null) {
                        if (isCancelled()) return
                        Log.e(TAG, "generateGaussians: ONNX INT8 inferStreaming returned null")
                        callback.onError("SHARP ONNX INT8 inference failed")
                        return
                    }

                    Log.d(TAG, "generateGaussians: ONNX INT8 SUCCESS ${result.gaussianCount} Gaussians room=${result.roomWidth}x${result.roomHeight}x${result.roomDepth}")
                    Log.d(TAG, "Room: ${result.roomWidth}m x ${result.roomHeight}m x ${result.roomDepth}m")

                    saveMetadata(result.plyFile.parentFile!!, image, "sharp_onnx_int8")

                    callback.onProgress(1.0f, "Done!")
                    callback.onComplete(GenerationResult(
                        plyFile = result.plyFile,
                        classicPlyFile = result.classicPlyFile,
                        roomWidth = result.roomWidth,
                        roomHeight = result.roomHeight,
                        roomDepth = result.roomDepth
                    ))
                } else if (useOnnxFp16) {
                    Log.d(TAG, "generateGaussians: invoking ONNX FP16 inferStreaming")
                    callback.onProgress(0.2f, "Running SHARP (ONNX FP16)...")
                    val result = kotlinx.coroutines.runBlocking {
                        splitOnnxFp16Sharp.inferStreaming(image, { progress, message ->
                            callback.onProgress(progress, message)
                        }, isCancelled)
                    }

                    if (result == null) {
                        if (isCancelled()) return
                        Log.e(TAG, "generateGaussians: ONNX FP16 inferStreaming returned null")
                        callback.onError("SHARP ONNX FP16 inference failed")
                        return
                    }

                    Log.d(TAG, "generateGaussians: ONNX FP16 SUCCESS ${result.gaussianCount} Gaussians")
                    saveMetadata(result.plyFile.parentFile!!, image, "sharp_onnx_fp16")
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
                    Log.d(TAG, "generateGaussians: invoking Split ONNX inferStreaming")
                    callback.onProgress(0.2f, "Running SHARP (Split ONNX - memory efficient)...")
                    val result = kotlinx.coroutines.runBlocking {
                        splitOnnxSharp.inferStreaming(image, { progress, message ->
                            callback.onProgress(progress, message)
                        }, isCancelled)
                    }

                    if (result == null) {
                        if (isCancelled()) return
                        Log.e(TAG, "generateGaussians: Split ONNX inferStreaming returned null")
                        callback.onError("SHARP Split ONNX inference failed")
                        return
                    }

                    Log.d(TAG, "generateGaussians: Split ONNX SUCCESS ${result.gaussianCount} Gaussians room=${result.roomWidth}x${result.roomHeight}x${result.roomDepth}")
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
                        return
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
            if (isCancelled()) {
                Log.d(TAG, "Generation stopped (cancelled)")
                return
            }
            Log.e(TAG, "Generation failed", e)
            callback.onError("Failed: ${e.message}")
        }
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
        Log.d(TAG, "Room saved: name='$roomName' type=$modelType path=${roomFolder.absolutePath}")
    }

    /**
     * Write Gaussian parameters to PLY file in standard 3DGS format.
     * Uses ONNX-style optimizations: LOGIT_LUT, LN_LUT, zeroSHBlock, batch writes.
     */
    private fun writePlyFile(file: File, result: NcnnSharp.GaussianResult) {
        val gaussianCount = result.gaussianCount
        val params = result.params

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
            val channel = fos.channel

            val batchBuffer = plyBatch; batchBuffer.clear()
            val scaleBoost = 1.3f
            val minScale = 0.001f
            val lutScale = (LOGIT_LUT_SIZE - 1).toFloat()

            var processed = 0
            while (processed < gaussianCount) {
                val currentBatch = minOf(PLY_BATCH_SIZE, gaussianCount - processed)
                for (j in 0 until currentBatch) {
                    val i = processed + j
                    val offset = i * NcnnSharp.PARAMS_PER_GAUSSIAN

                    val x = params[offset + 0]
                    val y = -params[offset + 1]
                    val z = -params[offset + 2]
                    batchBuffer.putFloat(x)
                    batchBuffer.putFloat(y)
                    batchBuffer.putFloat(z)

                    batchBuffer.putFloat(0f)
                    batchBuffer.putFloat(0f)
                    batchBuffer.putFloat(0f)

                    // Colors (NCNN native now outputs RGB after BGR swap)
                    val r = params[offset + 11].coerceIn(0f, 1f)
                    val g = params[offset + 12].coerceIn(0f, 1f)
                    val b = params[offset + 13].coerceIn(0f, 1f)
                    batchBuffer.putFloat((r - 0.5f) / SH_C0)
                    batchBuffer.putFloat((g - 0.5f) / SH_C0)
                    batchBuffer.putFloat((b - 0.5f) / SH_C0)

                    zeroSHBuffer.clear(); batchBuffer.put(zeroSHBuffer)

                    val rawOpacity = params[offset + 10].coerceIn(0f, 1f)
                    val lutIndex = (rawOpacity * lutScale).toInt().coerceIn(0, LOGIT_LUT_SIZE - 1)
                    batchBuffer.putFloat(LOGIT_LUT[lutIndex])

                    batchBuffer.putFloat(lnLut(max(params[offset + 3] * scaleBoost, minScale)))
                    batchBuffer.putFloat(lnLut(max(params[offset + 4] * scaleBoost, minScale)))
                    batchBuffer.putFloat(lnLut(max(params[offset + 5] * scaleBoost, minScale)))

                    val rw = params[offset + 6]
                    val rx = params[offset + 7]
                    val ry = params[offset + 8]
                    val rz = params[offset + 9]
                    val mag = sqrt(rw * rw + rx * rx + ry * ry + rz * rz)
                    val invMag = if (mag > 1e-8f) 1f / mag else 1f
                    batchBuffer.putFloat(rw * invMag)
                    batchBuffer.putFloat(rx * invMag)
                    batchBuffer.putFloat(ry * invMag)
                    batchBuffer.putFloat(rz * invMag)
                }
                batchBuffer.flip()
                batchBuffer.limit(currentBatch * BYTES_PER_VERTEX)
                while (batchBuffer.hasRemaining()) {
                    channel.write(batchBuffer)
                }
                batchBuffer.clear()
                processed += currentBatch
            }
        }

        Log.d(TAG, "Wrote PLY file: ${file.absolutePath} (${file.length()} bytes)")
    }

    /**
     * Release resources
     */
    fun release() {
        when {
            useNativePt -> nativePtSharp.release()
            useLiteRT -> litertSharp.release()
            useExecutorch -> executorchSharp.release()
            useExecutorchFp16 -> executorchFp16Sharp.release()
            useExecutorchInt8 -> { /* ExecuTorch INT8 drop-in holds no long-lived native session to release */ }
            useOnnx -> onnxSharp.release()
            useSplitOnnx -> { /* Do not close preloaded sessions while inference may be running */ }
            useOnnxFp16 -> { /* FP16 sessions created/closed per-part during inference */ }
            useOnnxInt8 -> onnxInt8Sharp.release()
            else -> ncnnSharp.release()
        }
        isInitialized = false
        currentBackendId = null
    }
}
