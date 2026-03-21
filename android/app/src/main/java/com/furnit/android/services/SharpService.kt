package com.furnit.android.services

import android.content.Context
import android.graphics.Bitmap
import com.furnit.android.utils.LogUtil
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.atomic.AtomicBoolean

/**
 * SharpService — on-device 3D Gaussian Splat room generation via **ExecuTorch INT8** only
 * ([ExecutorchInt8Sharp]).
 */
class SharpService private constructor(private val context: Context) {

    companion object {
        private const val TAG = "SharpService"

        @Volatile
        private var instance: SharpService? = null

        fun getInstance(context: Context): SharpService {
            return instance ?: synchronized(this) {
                instance ?: SharpService(context.applicationContext).also {
                    instance = it
                    LogUtil.d(TAG, "SharpService singleton created")
                }
            }
        }
    }

    private val executorchInt8Sharp by lazy { ExecutorchInt8Sharp.getInstance(context) }
    private var isInitialized = false
    private var currentBackendId: String? = null
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
     * Preload / sync ExecuTorch models when user opens the SHARP screen.
     */
    suspend fun preloadSharpModels() = withContext(Dispatchers.IO) {
        if (!BackendConfig.ENABLE_EXECUTORCH_INT8) return@withContext
        val prefs = context.getSharedPreferences("furnit_prefs", Context.MODE_PRIVATE)
        val backend = prefs.getString("inference_backend", "executorch_int8") ?: "executorch_int8"
        val effective = BackendConfig.normalize(backend)
        if (effective == "executorch_int8") {
            LogUtil.d(TAG, "ExecuTorch INT8 – sync models from external if present")
            executorchInt8Sharp.syncModelsFromExternal()
        }
    }

    fun isModelReady(): Boolean = BackendConfig.ENABLE_EXECUTORCH_INT8

    suspend fun initialize(): Boolean {
        lastInitFailureMessage = null
        val prefs = context.getSharedPreferences("furnit_prefs", Context.MODE_PRIVATE)

        val requestedBackend: String
        val existingBackend = prefs.getString("inference_backend", null)
        if (existingBackend != null) {
            requestedBackend = existingBackend
        } else {
            val useNcnn = prefs.getBoolean("use_ncnn_backend", false)
            requestedBackend = if (useNcnn) "ncnn" else "executorch_int8"
            prefs.edit()
                .putString("inference_backend", requestedBackend)
                .remove("use_ncnn_backend")
                .apply()
        }

        var effectiveBackend = BackendConfig.normalize(requestedBackend)
        if (effectiveBackend != requestedBackend) {
            LogUtil.w(TAG, "Backend '$requestedBackend' removed/disabled; using '$effectiveBackend'")
            prefs.edit().putString("inference_backend", effectiveBackend).apply()
        }

        if (isInitialized && currentBackendId == effectiveBackend) return true
        if (isInitialized && currentBackendId != effectiveBackend) {
            LogUtil.d(TAG, "Backend changed from '$currentBackendId' to '$effectiveBackend' — re-initializing")
            release()
        }

        if (!BackendConfig.ENABLE_EXECUTORCH_INT8) {
            lastInitFailureMessage = "ExecuTorch INT8 is disabled in this build."
            LogUtil.e(TAG, lastInitFailureMessage!!)
            return false
        }

        LogUtil.d(TAG, "ExecuTorch INT8 SHARP (CPU vs Vulkan per Settings → ExecutorchInt8Sharp)")
        if (executorchInt8Sharp.initialize()) {
            isInitialized = true
            currentBackendId = effectiveBackend
            LogUtil.d(TAG, "ExecuTorch INT8 SHARP initialized successfully")
            return true
        }
        LogUtil.e(TAG, "ExecuTorch INT8 SHARP init failed")
        lastInitFailureMessage =
            "ExecuTorch INT8 init failed. Push split .pte models to device storage (see android push scripts / README)."
        return false
    }

    private val generationCancelled = AtomicBoolean(false)

    fun startGenerationInBackground(image: Bitmap, callback: ProgressCallback): GenerationHandle {
        generationCancelled.set(false)
        val handle = object : GenerationHandle {
            override fun cancel() {
                generationCancelled.set(true)
                LogUtil.d(TAG, "Generation cancelled by user")
            }
        }
        Thread {
            generateGaussiansInternal(image, callback) { generationCancelled.get() }
        }.start()
        return handle
    }

    fun generateGaussians(image: Bitmap, callback: ProgressCallback) {
        generationCancelled.set(false)
        Thread {
            generateGaussiansInternal(image, callback) { false }
        }.start()
    }

    private fun generateGaussiansInternal(image: Bitmap, callback: ProgressCallback, isCancelled: () -> Boolean) {
        LogUtil.d(TAG, "Starting generation: ${image.width}x${image.height}")

        try {
            callback.onProgress(0.1f, "Preparing...")
            callback.onProgress(0.15f, "Loading SHARP model...")
            val initialized = kotlinx.coroutines.runBlocking { initialize() }
            if (!initialized) {
                callback.onError(lastInitFailureMessage ?: "SHARP model not available. Push model files to device.")
                return
            }
            if (isCancelled()) return

            LogUtil.d(TAG, "generateGaussians: invoking ExecuTorch INT8 inferStreaming")
            callback.onProgress(0.2f, "Running SHARP (ExecuTorch INT8)...")
            val result = kotlinx.coroutines.runBlocking {
                executorchInt8Sharp.inferStreaming(
                    bitmap = image,
                    progressCallback = { progress: Float, message: String ->
                        val mapped = (0.2f + 0.79f * progress).coerceIn(0.2f, 0.99f)
                        callback.onProgress(mapped, message)
                    }
                )
            }

            if (result == null) {
                callback.onError("SHARP ExecuTorch INT8 inference failed")
                return
            }

            LogUtil.d(TAG, "Generated ${result.gaussianCount} Gaussians (ExecuTorch INT8)")
            LogUtil.d(TAG, "Room: ${result.roomWidth}m x ${result.roomHeight}m x ${result.roomDepth}m")
            val isPortraitFeed = image.height > image.width
            LogUtil.d(
                TAG,
                "VIEWER_FEED isPortrait=$isPortraitFeed roomWidth=${result.roomWidth} roomHeight=${result.roomHeight} roomDepth=${result.roomDepth} path=${result.plyFile.parentFile?.absolutePath}"
            )

            saveMetadata(
                result.plyFile.parentFile!!,
                image,
                "sharp_executorch_int8",
                result.roomWidth,
                result.roomHeight,
                result.roomDepth,
                result.roomCenterX,
                result.roomCenterY,
                result.roomCenterZ
            )

            callback.onProgress(1.0f, "Done!")
            callback.onComplete(
                GenerationResult(
                    plyFile = result.plyFile,
                    classicPlyFile = result.classicPlyFile,
                    roomWidth = result.roomWidth,
                    roomHeight = result.roomHeight,
                    roomDepth = result.roomDepth,
                    roomCenterX = result.roomCenterX,
                    roomCenterY = result.roomCenterY,
                    roomCenterZ = result.roomCenterZ
                )
            )
        } catch (e: Exception) {
            if (isCancelled()) {
                LogUtil.d(TAG, "Generation stopped (cancelled)")
                return
            }
            LogUtil.e(TAG, "Generation failed", e)
            callback.onError("Failed: ${e.message}")
        }
    }

    private fun saveMetadata(
        roomFolder: File,
        image: Bitmap,
        modelType: String,
        roomWidth: Float? = null,
        roomHeight: Float? = null,
        roomDepth: Float? = null,
        roomCenterX: Float? = null,
        roomCenterY: Float? = null,
        roomCenterZ: Float? = null
    ) {
        val thumbnailFile = File(roomFolder, "thumbnail.png")
        FileOutputStream(thumbnailFile).use { out ->
            image.compress(Bitmap.CompressFormat.PNG, 90, out)
        }

        val metadataFile = File(roomFolder, "metadata.txt")
        val dateFormat = SimpleDateFormat("MMM d", Locale.getDefault())
        val roomName = "AI Room ${dateFormat.format(Date())}"
        val photoOrientation = if (image.height > image.width) "portrait" else "landscape"
        val sb = StringBuilder()
        sb.append("name=$roomName\n")
        sb.append("created=${System.currentTimeMillis()}\n")
        sb.append("type=$modelType\n")
        sb.append("photoOrientation=$photoOrientation\n")
        roomWidth?.let { sb.append("roomWidth=$it\n") }
        roomHeight?.let { sb.append("roomHeight=$it\n") }
        roomDepth?.let { sb.append("roomDepth=$it\n") }
        roomCenterX?.let { sb.append("roomCenterX=$it\n") }
        roomCenterY?.let { sb.append("roomCenterY=$it\n") }
        roomCenterZ?.let { sb.append("roomCenterZ=$it\n") }
        metadataFile.writeText(sb.toString())
        LogUtil.d(TAG, "Room saved: name='$roomName' type=$modelType path=${roomFolder.absolutePath} dims=${roomWidth}x${roomHeight}x${roomDepth} orientation=$photoOrientation")
    }

    /**
     * Release native caches only (e.g. ExecuTorch INT8 Part1+Part2 cache). Call on trim memory to reduce pressure.
     */
    fun releaseNativeCaches() {
        try {
            executorchInt8Sharp.releaseNativeCaches()
        } catch (_: Throwable) { }
    }

    fun release() {
        if (isInitialized) {
            try {
                executorchInt8Sharp.releaseNativeCaches()
            } catch (_: Throwable) { }
        }
        isInitialized = false
        currentBackendId = null
    }
}
