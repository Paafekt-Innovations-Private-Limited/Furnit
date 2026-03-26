package com.furnit.android.services

import android.content.Context
import android.graphics.Bitmap
import android.net.Uri
import com.furnit.android.ar.MetricAnchor
import com.furnit.android.models.PhotoOrientation
import com.furnit.android.utils.LogUtil
import com.furnit.android.utils.RoomDisplayName
import com.furnit.android.utils.RoomFolderMetadata
import com.furnit.android.utils.SharpRoomDimensionSanitizer
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.util.Date
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
            LogUtil.d(TAG, "ExecuTorch INT8 – hydrate APK assets + sync external models if present")
            executorchInt8Sharp.hydrateBundledAndExternalModels()
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
            executorchInt8Sharp.consumeInitializeFailureDetail()
                ?: "ExecuTorch INT8 init failed. Push split .pte models to device storage (see android push scripts / README)."
        return false
    }

    private val generationCancelled = AtomicBoolean(false)

    /** Cancel the current background [startGenerationInBackground] job (same flag as [GenerationHandle.cancel]). */
    fun cancelGeneration() {
        generationCancelled.set(true)
        LogUtil.d(TAG, "Generation cancel requested")
    }

    /**
     * @param viewerPhotoOrientation "portrait" / "landscape" from SinglePhoto / EXIF fallback.
     * @param viewerPhotoWideAngle 0.5× ultra-wide; with [orientationLockedByUser] false, automatic landscape → portrait in metadata.
     * @param orientationLockedByUser True after user tapped the orientation row (do not apply ultra-wide portrait coercion for explicit landscape).
     * @param sourcePhotoUri Original image URI for [PhotoOrientation.detect] when orientation is null.
     * @param sourcePhotoPath Original file path for [PhotoOrientation.detectFromFile] when URI is unavailable.
     */
    fun startGenerationInBackground(
        image: Bitmap,
        callback: ProgressCallback,
        viewerPhotoOrientation: String? = null,
        viewerPhotoWideAngle: Boolean = false,
        orientationLockedByUser: Boolean = false,
        sourcePhotoUri: Uri? = null,
        sourcePhotoPath: String? = null,
        metricAnchors: List<MetricAnchor>? = null,
    ): GenerationHandle {
        generationCancelled.set(false)
        val handle = object : GenerationHandle {
            override fun cancel() {
                generationCancelled.set(true)
                LogUtil.d(TAG, "Generation cancelled by user")
            }
        }
        Thread {
            generateGaussiansInternal(
                image,
                callback,
                { generationCancelled.get() },
                viewerPhotoOrientation,
                viewerPhotoWideAngle,
                orientationLockedByUser,
                sourcePhotoUri,
                sourcePhotoPath,
                metricAnchors,
            )
        }.start()
        return handle
    }

    fun generateGaussians(
        image: Bitmap,
        callback: ProgressCallback,
        viewerPhotoOrientation: String? = null,
        viewerPhotoWideAngle: Boolean = false,
        orientationLockedByUser: Boolean = false,
        sourcePhotoUri: Uri? = null,
        sourcePhotoPath: String? = null,
        metricAnchors: List<MetricAnchor>? = null,
    ) {
        generationCancelled.set(false)
        Thread {
            generateGaussiansInternal(
                image,
                callback,
                { false },
                viewerPhotoOrientation,
                viewerPhotoWideAngle,
                orientationLockedByUser,
                sourcePhotoUri,
                sourcePhotoPath,
                metricAnchors,
            )
        }.start()
    }

    private fun generateGaussiansInternal(
        image: Bitmap,
        callback: ProgressCallback,
        isCancelled: () -> Boolean,
        viewerPhotoOrientation: String? = null,
        viewerPhotoWideAngle: Boolean = false,
        orientationLockedByUser: Boolean = false,
        sourcePhotoUri: Uri? = null,
        sourcePhotoPath: String? = null,
        metricAnchors: List<MetricAnchor>? = null,
    ) {
        LogUtil.d(TAG, "Starting generation: ${image.width}x${image.height}")
        LogUtil.d(
            TAG,
            "[SHARP_ORIENTATION] ExecuTorch generation start " +
                "bitmap=${image.width}x${image.height} " +
                "viewerPhotoOrientation=${viewerPhotoOrientation ?: "null"} " +
                "photoWideAngle=$viewerPhotoWideAngle " +
                "orientationLockedByUser=$orientationLockedByUser " +
                "sourceUri=${if (sourcePhotoUri != null) "yes" else "no"} " +
                "sourcePath=${if (sourcePhotoPath.isNullOrBlank()) "no" else "yes"} " +
                "metricAnchors=${metricAnchors?.size ?: 0}",
        )

        try {
            callback.onProgress(0.1f, "Preparing...")
            callback.onProgress(0.15f, "Loading SHARP model...")
            val initialized = kotlinx.coroutines.runBlocking { initialize() }
            if (!initialized) {
                callback.onError(lastInitFailureMessage ?: "SHARP model not available. Push model files to device.")
                return
            }
            if (isCancelled()) {
                callback.onError("SHARP_CANCELLED")
                return
            }

            LogUtil.d(TAG, "generateGaussians: invoking ExecuTorch INT8 inferStreaming")
            callback.onProgress(0.2f, "Running SHARP (ExecuTorch INT8)...")
            val result = kotlinx.coroutines.runBlocking {
                executorchInt8Sharp.inferStreaming(
                    bitmap = image,
                    metricAnchors = metricAnchors,
                    progressCallback = { progress: Float, message: String ->
                        val mapped = (0.2f + 0.79f * progress).coerceIn(0.2f, 0.99f)
                        callback.onProgress(mapped, message)
                    }
                )
            }

            if (isCancelled()) {
                callback.onError("SHARP_CANCELLED")
                return
            }
            if (result == null) {
                val detail = executorchInt8Sharp.consumeInferStreamingFailureDetail()
                callback.onError(detail ?: "SHARP ExecuTorch INT8 inference failed")
                return
            }

            LogUtil.d(TAG, "Generated ${result.gaussianCount} Gaussians (ExecuTorch INT8)")
            LogUtil.d(TAG, "Room: ${result.roomWidth}m x ${result.roomHeight}m x ${result.roomDepth}m")
            val (sanW, sanH, sanD) = SharpRoomDimensionSanitizer.sanitizeMeters(
                result.roomWidth,
                result.roomHeight,
                result.roomDepth,
            )
            LogUtil.i(
                "SHARP_ROOM_MEAS",
                "[room_created] infer_done gaussians=${result.gaussianCount} " +
                    "bbox_raw W×H×D=${result.roomWidth}×${result.roomHeight}×${result.roomDepth} " +
                    "bbox_san W×H×D=$sanW×$sanH×$sanD " +
                    "center=(${result.roomCenterX},${result.roomCenterY},${result.roomCenterZ}) " +
                    "folder=${result.plyFile.parentFile?.absolutePath} classicPly=${result.classicPlyFile.name}",
            )
            val feedOrientation = PhotoOrientation.fromBitmapDimensions(image)
            LogUtil.d(
                TAG,
                "[SHARP_ORIENTATION] post-infer bitmap_layout=${feedOrientation.value} " +
                    "(SHARP input pixels ${image.width}x${image.height}) " +
                    "room_raw=${result.roomWidth}x${result.roomHeight}x${result.roomDepth} " +
                    "room_san=$sanW×$sanH×$sanD " +
                    "path=${result.plyFile.parentFile?.absolutePath}",
            )

            saveMetadata(
                roomFolder = result.plyFile.parentFile!!,
                image = image,
                modelType = "sharp_executorch_int8",
                roomWidth = sanW,
                roomHeight = sanH,
                roomDepth = sanD,
                roomCenterX = result.roomCenterX,
                roomCenterY = result.roomCenterY,
                roomCenterZ = result.roomCenterZ,
                viewerPhotoOrientation = viewerPhotoOrientation,
                viewerPhotoWideAngle = viewerPhotoWideAngle,
                orientationLockedByUser = orientationLockedByUser,
                sourcePhotoUri = sourcePhotoUri,
                sourcePhotoPath = sourcePhotoPath,
            )

            if (isCancelled()) {
                callback.onError("SHARP_CANCELLED")
                return
            }

            callback.onProgress(1.0f, "Done!")
            callback.onComplete(
                GenerationResult(
                    plyFile = result.plyFile,
                    classicPlyFile = result.classicPlyFile,
                    roomWidth = sanW,
                    roomHeight = sanH,
                    roomDepth = sanD,
                    roomCenterX = result.roomCenterX,
                    roomCenterY = result.roomCenterY,
                    roomCenterZ = result.roomCenterZ
                )
            )
        } catch (e: Exception) {
            if (isCancelled()) {
                LogUtil.d(TAG, "Generation stopped (cancelled)")
                callback.onError("SHARP_CANCELLED")
                return
            }
            LogUtil.e(TAG, "Generation failed", e)
            callback.onError("Failed: ${e.message}")
        }
    }

    /**
     * Writes room `metadata.txt` + [RoomFolderMetadata].
     *
     * **Orientation in metadata (Swift parity):**
     * 1) Non-null [viewerPhotoOrientation] (`portrait` / `landscape`) — e.g. SinglePhotoRoom user toggle; wins.
     * 2) Else [PhotoOrientation.detect] / [PhotoOrientation.detectFromFile] on [sourcePhotoUri] or [sourcePhotoPath]
     *    (encoded dimensions + EXIF rotation + portrait-first bias when rotation is 0).
     * 3) Else [PhotoOrientation.fromBitmapDimensions] (synthetic bitmaps / tests with no file source).
     *
     * **0.5× ultra-wide:** When [viewerPhotoWideAngle] is true and orientation was **not** locked by the user,
     * [PhotoOrientation.coercePortraitForUltraWide] maps automatic landscape → portrait (sensor buffer quirk).
     */
    private fun saveMetadata(
        roomFolder: File,
        image: Bitmap,
        modelType: String,
        roomWidth: Float? = null,
        roomHeight: Float? = null,
        roomDepth: Float? = null,
        roomCenterX: Float? = null,
        roomCenterY: Float? = null,
        roomCenterZ: Float? = null,
        viewerPhotoOrientation: String? = null,
        viewerPhotoWideAngle: Boolean = false,
        orientationLockedByUser: Boolean = false,
        sourcePhotoUri: Uri? = null,
        sourcePhotoPath: String? = null,
    ) {
        val thumbnailFile = File(roomFolder, "thumbnail.png")
        FileOutputStream(thumbnailFile).use { out ->
            image.compress(Bitmap.CompressFormat.PNG, 90, out)
        }

        val metadataFile = File(roomFolder, "metadata.txt")
        val createdAtMillis = System.currentTimeMillis()
        val roomName = RoomDisplayName.aiRoomWithTimestamp(Date(createdAtMillis))
        val normalizedViewer = viewerPhotoOrientation?.trim()?.lowercase()
        val bitmapLayoutOrientation = PhotoOrientation.fromBitmapDimensions(image)
        fun orientationEnumToMetadataString(o: PhotoOrientation): String {
            val adjusted = if (orientationLockedByUser) {
                o
            } else {
                PhotoOrientation.coercePortraitForUltraWide(o, viewerPhotoWideAngle)
            }
            return when (adjusted) {
                PhotoOrientation.LANDSCAPE -> "landscape"
                PhotoOrientation.PORTRAIT -> "portrait"
                PhotoOrientation.SQUARE -> "portrait"
            }
        }
        val photoOrientation: String
        val orientationDecisionSource: String
        val exifDetectOrientation: PhotoOrientation?
        when (normalizedViewer) {
            "landscape", "portrait" -> {
                exifDetectOrientation = null
                val coercedUltraWide = !orientationLockedByUser &&
                    viewerPhotoWideAngle &&
                    normalizedViewer == "landscape"
                photoOrientation = if (coercedUltraWide) "portrait" else normalizedViewer!!
                orientationDecisionSource = if (coercedUltraWide) {
                    "explicit_viewer_ultrawide_to_portrait"
                } else {
                    "explicit_viewer"
                }
            }
            else -> {
                exifDetectOrientation = when {
                    sourcePhotoUri != null -> runCatching {
                        PhotoOrientation.detect(context, sourcePhotoUri)
                    }.getOrNull()
                    !sourcePhotoPath.isNullOrBlank() -> runCatching {
                        PhotoOrientation.detectFromFile(sourcePhotoPath.trim())
                    }.getOrNull()
                    else -> null
                }
                val inferredEnum = exifDetectOrientation ?: bitmapLayoutOrientation
                photoOrientation = orientationEnumToMetadataString(inferredEnum)
                orientationDecisionSource = when {
                    exifDetectOrientation != null && sourcePhotoUri != null -> "exif_content_uri"
                    exifDetectOrientation != null -> "exif_file_path"
                    else -> "bitmap_pixels_only"
                }
            }
        }
        LogUtil.d(
            TAG,
            "[SHARP_ORIENTATION] room metadata written " +
                "final=$photoOrientation " +
                "decision=$orientationDecisionSource " +
                "bitmap_layout=${bitmapLayoutOrientation.value} " +
                "exif_detect=${exifDetectOrientation?.value ?: "n/a"} " +
                "viewer_raw=${viewerPhotoOrientation ?: "null"} " +
                "wide=$viewerPhotoWideAngle locked=$orientationLockedByUser " +
                "image=${image.width}x${image.height}",
        )
        val sb = StringBuilder()
        sb.append("name=$roomName\n")
        sb.append("created=$createdAtMillis\n")
        sb.append("type=$modelType\n")
        sb.append("photoOrientation=$photoOrientation\n")
        sb.append("photoWideAngle=$viewerPhotoWideAngle\n")
        sb.append("previewOnly=true\n")
        roomWidth?.let { sb.append("roomWidth=$it\n") }
        roomHeight?.let { sb.append("roomHeight=$it\n") }
        roomDepth?.let { sb.append("roomDepth=$it\n") }
        roomCenterX?.let { sb.append("roomCenterX=$it\n") }
        roomCenterY?.let { sb.append("roomCenterY=$it\n") }
        roomCenterZ?.let { sb.append("roomCenterZ=$it\n") }
        metadataFile.writeText(sb.toString())
        RoomFolderMetadata.writeToFolder(
            roomFolder,
            RoomFolderMetadata.Snapshot(
                name = roomName,
                createdAt = createdAtMillis,
                type = modelType,
                photoOrientation = photoOrientation,
                photoWideAngle = viewerPhotoWideAngle,
                roomWidth = roomWidth,
                roomHeight = roomHeight,
                roomDepth = roomDepth,
                roomCenterX = roomCenterX,
                roomCenterY = roomCenterY,
                roomCenterZ = roomCenterZ,
                previewOnly = true,
            )
        )
        LogUtil.d(
            TAG,
            "Room saved: name='$roomName' type=$modelType path=${roomFolder.absolutePath} " +
                "dims=${roomWidth}x${roomHeight}x${roomDepth} photoOrientation=$photoOrientation wide=$viewerPhotoWideAngle",
        )
        LogUtil.i(
            "SHARP_ROOM_MEAS",
            "[metadata_written] roomWidth=$roomWidth roomHeight=$roomHeight roomDepth=$roomDepth " +
                "center=($roomCenterX,$roomCenterY,$roomCenterZ) photoOrientation=$photoOrientation " +
                "path=${roomFolder.absolutePath}",
        )
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
