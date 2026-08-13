package com.furnit.android.services

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Bitmap.Config
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.os.Handler
import android.os.Looper
import com.furnit.android.BuildConfig
import com.furnit.android.utils.LogUtil
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtException
import ai.onnxruntime.OrtProvider
import ai.onnxruntime.OrtSession
import ai.onnxruntime.OrtSession.SessionOptions
import ai.onnxruntime.providers.NNAPIFlags
import com.furnit.android.DetectionResult
import com.furnit.android.ar.ArSupportChecker
import java.io.File
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.atomic.AtomicReference
import java.util.EnumSet
import org.json.JSONObject
import kotlin.math.ceil
import kotlin.math.exp
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sqrt

// Result containing mask and detections
data class SegmentationResult(
    val mask: Bitmap?,
    val detections: List<DetectionResult>,
    val inputSize: Int = 640,
    val primaryDetection: DetectionResult? = null,
    val detectionClusters: List<List<Int>> = emptyList(),
    val sourceWidth: Int = inputSize,
    val sourceHeight: Int = inputSize,
)

/**
 * FurnitureFitManager handles furniture detection and selected-object cutout generation.
 *
 * Inference backend: FP16 LiteRT, using the GPU when supported and XNNPACK CPU otherwise. The
 * renderer outputs real camera pixels only inside the selected mask and leaves every other pixel
 * transparent so the room remains visible behind it.
 */
class FurnitureFitManager(private val context: Context) {
    private val mainHandler = Handler(Looper.getMainLooper())
    private enum class CallbackDelivery { MAIN, INFERENCE }
    /**
     * Session generation for queued/running work. Swift rotates its Core ML inference queue when
     * Furniture Fit stops so an abandoned prediction cannot hold up the next session. ONNX Runtime
     * exposes an explicit per-run termination primitive, so Android invalidates queued work and
     * terminates the active run to provide the same stop -> immediate restart guarantee without
     * running two calls against the shared scratch buffers concurrently.
     */
    private val inferenceGeneration = AtomicLong(0L)
    private val activeRunOptions = AtomicReference<OrtSession.RunOptions?>(null)
    private val bboxExpandMargin = 0.08f
    private val includeSupportingTableForMonitorScene = true
    private val enableMorphCloseForMask = false
    private val monitorLikeClassIds = setOf(62, 63) // COCO: tv, laptop
    private val supportingTableClassIds = setOf(60) // COCO: dining table
    private val classNames: Map<Int, String> by lazy(LazyThreadSafetyMode.NONE) { loadClassNames() }
    private val ignoredClassIds: Set<Int> by lazy(LazyThreadSafetyMode.NONE) { loadIgnoredClassIds() }

    companion object {
        private const val TAG = "FurnitureFitManager"
        private const val RTMDET_TFLITE_MODEL_ASSET = RoomGenerationAssets.RTMDET_INS_M_RAW_FP16_TFLITE
        private const val DEFAULT_CONFIDENCE_THRESHOLD = 0.10f
        private const val RTMDET_CONFIDENCE_THRESHOLD = RTMDetSwiftParity.CONFIDENCE_THRESHOLD
        private const val LOW_CONFIDENCE_OVERSIZED_THRESHOLD = 0.65f
        private const val OVERSIZED_BOX_AREA_FRACTION = 0.45f
        private const val RTMDET_MASK_KEEP_THRESHOLD = 0.80f
        private const val DEFAULT_NMS_IOU_THRESHOLD = RTMDetSwiftParity.NMS_IOU_THRESHOLD
        private const val DEFAULT_MAX_DETECTIONS = RTMDetSwiftParity.MAX_DETECTION_COUNT
        private const val RAW_MASK_AFFINITY_THRESHOLD = RTMDetSwiftParity.MASK_AFFINITY_THRESHOLD
        private const val RAW_MASK_AFFINITY_BIT_THRESHOLD = RTMDetSwiftParity.MASK_AFFINITY_BIT_THRESHOLD
        // Legacy postprocessed-prototype models retain their established 0.5 contour. Raw RTMDet
        // rendering uses RTMDetSwiftParity.rawMaskRenderAlpha (Swift's 0.30 contour) directly.
        private const val RAW_MASK_RENDER_THRESHOLD = 0.50f
        private const val RAW_MASK_RENDER_ANTIALIAS_HALF_WIDTH = 0.05f

        private const val RTMDET_INPUT_SIZE = RTMDetSwiftParity.MODEL_SIDE
        private const val RTMDET_SOURCE_MASK_SIDE = RTMDetSwiftParity.SOURCE_MASK_SIDE
        private const val RTMDET_MASK_SIDE = RTMDetSwiftParity.MASK_SIDE
        private const val RTMDET_EMBEDDED_PREPROCESS_METADATA = "furnit.rtmdet.preprocess"
        private const val RTMDET_EMBEDDED_PREPROCESS_VALUE = "bgr_mean_std"
        private val RTMDET_DETECTION_OUTPUTS = linkedSetOf(
            "cls_80", "bbox_80",
            "cls_40", "bbox_40",
            "cls_20", "bbox_20",
        )

        private val sharedBackendLock = Any()
        @Volatile private var sharedBackend: OnnxBackend? = null
        @Volatile private var sharedLiteRtBackend: RTMDetLiteRtBackend? = null
        private val sharedResourceGeneration = java.util.concurrent.atomic.AtomicLong(0L)

        /** Reusable memory belongs to one serial inference lane and is never shared across lanes. */
        private class InferenceWorkspace {
            var inputBuffer: FloatBuffer? = null
            var inputFloatCount: Int = 0
            var inputTensor: OnnxTensor? = null
            var inputShape: LongArray? = null
            var preparedBitmap: Bitmap? = null
            var preparedPixels = IntArray(0)
            var framePixels = IntArray(0)
            var outputPixels = IntArray(0)
            val preprocessPaint = Paint(Paint.FILTER_BITMAP_FLAG)

            fun close() {
                inputTensor?.close()
                inputTensor = null
                inputShape = null
                inputBuffer = null
                inputFloatCount = 0
                preparedBitmap?.takeIf { !it.isRecycled }?.recycle()
                preparedBitmap = null
                preparedPixels = IntArray(0)
                framePixels = IntArray(0)
                outputPixels = IntArray(0)
            }
        }

        private data class InferenceLane(
            val generation: Long,
            val executor: ExecutorService,
            val workspace: InferenceWorkspace,
        )

        private val sharedInferenceLaneLock = Any()
        private val sharedInferenceLifecycleLock = Any()
        private val sharedInferenceWorkspace = ThreadLocal<InferenceWorkspace>()
        private val sharedReleaseExecutor = Executors.newSingleThreadExecutor()
        private var sharedInferenceLaneGeneration = 0L
        @Volatile private var sharedInferenceLane = createInferenceLane(sharedInferenceLaneGeneration)
        private var sharedAcceptedLaneTaskCount = 0
        private var pendingSharedReleaseGeneration: Long? = null
        private val lastCutoutAlphaStatsLogMs = AtomicLong(0L)

        private data class CreatedOrtSession(
            val session: OrtSession,
            val options: SessionOptions,
            val executionProvider: String,
        )

        private fun createInferenceLane(generation: Long): InferenceLane = InferenceLane(
            generation = generation,
            executor = Executors.newSingleThreadExecutor { runnable ->
                Thread(runnable, "FurnitureFitInference-$generation")
            },
            workspace = InferenceWorkspace(),
        )

        private fun noteSharedLaneTaskAccepted() {
            synchronized(sharedInferenceLifecycleLock) {
                sharedAcceptedLaneTaskCount += 1
            }
        }

        private fun noteSharedLaneTaskFinished() {
            val releaseGeneration = synchronized(sharedInferenceLifecycleLock) {
                check(sharedAcceptedLaneTaskCount > 0) { "Unbalanced Furniture Fit inference task count" }
                sharedAcceptedLaneTaskCount -= 1
                if (sharedAcceptedLaneTaskCount == 0) {
                    pendingSharedReleaseGeneration.also { pendingSharedReleaseGeneration = null }
                } else {
                    null
                }
            }
            if (releaseGeneration != null) scheduleSharedBackendRelease(releaseGeneration)
        }

        private fun executeOnSharedInferenceLane(task: () -> Unit): Boolean {
            synchronized(sharedInferenceLaneLock) {
                val lane = sharedInferenceLane
                noteSharedLaneTaskAccepted()
                return try {
                    lane.executor.execute {
                        sharedInferenceWorkspace.set(lane.workspace)
                        try {
                            task()
                        } finally {
                            sharedInferenceWorkspace.remove()
                            noteSharedLaneTaskFinished()
                        }
                    }
                    true
                } catch (_: java.util.concurrent.RejectedExecutionException) {
                    noteSharedLaneTaskFinished()
                    false
                }
            }
        }

        /**
         * Literal Swift-style queue rotation. Existing work stays on its retired serial queue;
         * subsequent frames use a new queue with independent scratch memory.
         */
        private fun rotateSharedInferenceLane(): Long {
            synchronized(sharedInferenceLaneLock) {
                val retiredLane = sharedInferenceLane
                sharedInferenceLaneGeneration += 1L
                val nextLane = createInferenceLane(sharedInferenceLaneGeneration)
                sharedInferenceLane = nextLane

                noteSharedLaneTaskAccepted()
                try {
                    retiredLane.executor.execute {
                        try {
                            retiredLane.workspace.close()
                        } finally {
                            noteSharedLaneTaskFinished()
                        }
                    }
                } catch (_: java.util.concurrent.RejectedExecutionException) {
                    retiredLane.workspace.close()
                    noteSharedLaneTaskFinished()
                }
                retiredLane.executor.shutdown()
                return nextLane.generation
            }
        }

        private fun currentInferenceWorkspace(): InferenceWorkspace =
            checkNotNull(sharedInferenceWorkspace.get()) {
                "Furniture Fit inference scratch accessed outside its owning serial lane"
            }

        /**
         * Metric overlay sizing uses ARCore depth/planes when the device supports ARCore; otherwise non-metric fallback.
         */
        fun isArAssistedFurnitureSizingEnabled(context: android.content.Context): Boolean =
            ArSupportChecker.isArCoreSupported(context)

        /** True when a room viewer's [initializeAuto] has already created the shared interpreter. */
        fun isSharedBackendReady(): Boolean = sharedLiteRtBackend != null

        /**
         * Mirrors Swift `RTMDetModelService.releaseResources()` when a room viewer disappears.
         * Release is serialized after any accepted inference so an in-flight ONNX call is never
         * closed underneath itself. A new viewer initialization cancels a queued stale release.
         */
        fun releaseSharedResourcesAsync() {
            val releaseGeneration = sharedResourceGeneration.incrementAndGet()
            // Retire the current workspace even when it is idle. The shared backend itself is
            // released only after every accepted task on all retired lanes has completed.
            rotateSharedInferenceLane()
            val releaseNow = synchronized(sharedInferenceLifecycleLock) {
                pendingSharedReleaseGeneration = releaseGeneration
                if (sharedAcceptedLaneTaskCount == 0) {
                    pendingSharedReleaseGeneration = null
                    true
                } else {
                    false
                }
            }
            if (releaseNow) scheduleSharedBackendRelease(releaseGeneration)
        }

        private fun scheduleSharedBackendRelease(releaseGeneration: Long) {
            sharedReleaseExecutor.execute {
                if (sharedResourceGeneration.get() != releaseGeneration) return@execute
                synchronized(sharedInferenceLifecycleLock) {
                    if (sharedAcceptedLaneTaskCount != 0) {
                        pendingSharedReleaseGeneration = releaseGeneration
                        return@execute
                    }
                }
                synchronized(sharedBackendLock) {
                    if (sharedResourceGeneration.get() != releaseGeneration) return@execute
                    synchronized(sharedInferenceLifecycleLock) {
                        if (sharedAcceptedLaneTaskCount != 0) {
                            pendingSharedReleaseGeneration = releaseGeneration
                            return@execute
                        }
                    }
                    val backend = sharedBackend
                    sharedBackend = null
                    val liteRtBackend = sharedLiteRtBackend
                    sharedLiteRtBackend = null
                    backend?.session?.close()
                    backend?.options?.close()
                    liteRtBackend?.close()
                }
                LogUtil.i(TAG, "Released shared RTMDet resources after room viewer disappeared")
            }
        }

        private fun sharedLiteRtBackend(context: Context): RTMDetLiteRtBackend? {
            // A newly appearing viewer supersedes any release queued by the previous viewer.
            sharedResourceGeneration.incrementAndGet()
            synchronized(sharedInferenceLifecycleLock) {
                pendingSharedReleaseGeneration = null
            }
            sharedLiteRtBackend?.let { return it }
            synchronized(sharedBackendLock) {
                sharedLiteRtBackend?.let { return it }
                val backend = RTMDetLiteRtBackend.create(
                    context = context.applicationContext,
                    assetName = RTMDET_TFLITE_MODEL_ASSET,
                ) ?: return null
                sharedLiteRtBackend = backend
                return backend
            }
        }

        private fun sharedOnnxBackend(context: Context, onnxAssetName: String): OnnxBackend? {
            // A newly appearing viewer supersedes any release queued by the previous viewer.
            sharedResourceGeneration.incrementAndGet()
            synchronized(sharedInferenceLifecycleLock) {
                pendingSharedReleaseGeneration = null
            }
            sharedBackend?.takeIf { it.assetName == onnxAssetName }?.let { return it }
            synchronized(sharedBackendLock) {
                sharedBackend?.takeIf { it.assetName == onnxAssetName }?.let { return it }

                val initStartNanos = System.nanoTime()
                return try {
                    val appContext = context.applicationContext
                    val file = copyAssetToFile(appContext, onnxAssetName)
                    val env = OrtEnvironment.getEnvironment()
                    val createdSession = createRtmdetSession(env, file)
                    val opts = createdSession.options
                    val session = createdSession.session
                    val modelMetadata = session.metadata.customMetadata
                    val backend = OnnxBackend(
                        env = env,
                        session = session,
                        options = opts,
                        assetName = onnxAssetName,
                        inputName = session.inputInfo.entries.firstOrNull()?.key ?: "input",
                        inputWidth = session.inputInfo.entries.firstOrNull()
                            ?.value
                            ?.info
                            ?.let { it as? ai.onnxruntime.TensorInfo }
                            ?.shape
                            ?.getOrNull(3)
                            ?.toInt()
                            ?.takeIf { it > 0 }
                            ?: RTMDET_INPUT_SIZE,
                        inputHeight = session.inputInfo.entries.firstOrNull()
                            ?.value
                            ?.info
                            ?.let { it as? ai.onnxruntime.TensorInfo }
                            ?.shape
                            ?.getOrNull(2)
                            ?.toInt()
                            ?.takeIf { it > 0 }
                            ?: RTMDET_INPUT_SIZE,
                        embedsRtmdetPreprocess = modelMetadata[RTMDET_EMBEDDED_PREPROCESS_METADATA] ==
                            RTMDET_EMBEDDED_PREPROCESS_VALUE,
                        executionProvider = createdSession.executionProvider,
                        isRtmdetRaw = session.outputInfo.containsKey("cls_80") &&
                                session.outputInfo.containsKey("bbox_80") &&
                                session.outputInfo.containsKey("kernel_80") &&
                                session.outputInfo.containsKey("mask_feat"),
                    )
                    sharedBackend = backend
                    LogUtil.i(
                        TAG,
                        "Loaded shared ONNX model '$onnxAssetName' provider=${createdSession.executionProvider} " +
                            "embeddedPreprocess=${backend.embedsRtmdetPreprocess} " +
                            "in ${(System.nanoTime() - initStartNanos) / 1_000_000L}ms",
                    )
                    for ((name, info) in session.inputInfo) {
                        LogUtil.i(TAG, "ONNX input: $name -> ${info.info}")
                    }
                    for ((name, info) in session.outputInfo) {
                        LogUtil.i(TAG, "ONNX output: $name -> ${info.info}")
                    }
                    backend
                } catch (e: Exception) {
                    LogUtil.e(TAG, "Failed to load shared ONNX model '$onnxAssetName': ${e.message}", e)
                    null
                }
            }
        }

        /**
         * Swift requests CPU + Neural Engine before falling back to CPU/GPU. NNAPI with relaxed
         * FP16 is the corresponding Android hardware path. Session creation performs NNAPI graph
         * compilation, so a device with an incompatible accelerator fails here and immediately
         * falls back to the already-proven XNNPACK path; plain ORT CPU remains the final fallback.
         */
        private fun createRtmdetSession(env: OrtEnvironment, file: File): CreatedOrtSession {
            val availableProviders = OrtEnvironment.getAvailableProviders()
            LogUtil.i(TAG, "Available ONNX execution providers: $availableProviders")

            if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.Q &&
                availableProviders.contains(OrtProvider.NNAPI)
            ) {
                val options = SessionOptions()
                try {
                    options.setOptimizationLevel(SessionOptions.OptLevel.ALL_OPT)
                    options.setExecutionMode(SessionOptions.ExecutionMode.SEQUENTIAL)
                    options.addNnapi(EnumSet.of(NNAPIFlags.USE_FP16, NNAPIFlags.CPU_DISABLED))
                    LogUtil.i(TAG, "Compiling RTMDet for NNAPI (FP16, accelerator-only)")
                    return CreatedOrtSession(
                        session = env.createSession(file.absolutePath, options),
                        options = options,
                        executionProvider = "NNAPI_FP16",
                    )
                } catch (e: Exception) {
                    options.close()
                    LogUtil.w(TAG, "NNAPI RTMDet session unavailable; trying XNNPACK: ${e.message}")
                }
            }

            if (availableProviders.contains(OrtProvider.XNNPACK)) {
                val options = SessionOptions()
                try {
                    val workerCount = Runtime.getRuntime().availableProcessors().coerceIn(2, 4)
                    options.setOptimizationLevel(SessionOptions.OptLevel.ALL_OPT)
                    options.setExecutionMode(SessionOptions.ExecutionMode.SEQUENTIAL)
                    options.setIntraOpNumThreads(1)
                    options.setInterOpNumThreads(1)
                    options.addConfigEntry("session.intra_op.allow_spinning", "0")
                    options.addXnnpack(mapOf("intra_op_num_threads" to workerCount.toString()))
                    LogUtil.i(TAG, "Compiling RTMDet for XNNPACK workers=$workerCount")
                    return CreatedOrtSession(
                        session = env.createSession(file.absolutePath, options),
                        options = options,
                        executionProvider = "XNNPACK",
                    )
                } catch (e: Exception) {
                    options.close()
                    LogUtil.w(TAG, "XNNPACK RTMDet session unavailable; using ORT CPU: ${e.message}")
                }
            }

            val options = SessionOptions()
            LogUtil.i(TAG, "Compiling RTMDet for default ORT CPU")
            return try {
                CreatedOrtSession(
                    session = env.createSession(file.absolutePath, options),
                    options = options,
                    executionProvider = "CPU",
                )
            } catch (e: Exception) {
                options.close()
                throw e
            }
        }

        @Throws(IOException::class)
        private fun copyAssetToFile(context: Context, assetName: String): File {
            val outFile = File(context.cacheDir, assetName)
            val installToken = context.packageManager
                .getPackageInfo(context.packageName, 0)
                .lastUpdateTime
                .toString()
            val tokenFile = File(context.cacheDir, "$assetName.install-token")
            if (
                outFile.exists() && outFile.length() > 0L &&
                runCatching { tokenFile.takeIf(File::isFile)?.readText() }.getOrNull() == installToken
            ) {
                return outFile
            }
            context.assets.open(assetName).use { input ->
                java.io.FileOutputStream(outFile).use { output ->
                    input.copyTo(output)
                }
            }
            tokenFile.writeText(installToken)
            return outFile
        }
    }

    // ONNX Runtime objects. @Volatile: written on the loader/main thread, read + closed on the
    // single-thread inferenceExecutor.
    @Volatile private var ortBackend: OnnxBackend? = null
    @Volatile private var ortEnv: OrtEnvironment? = null
    @Volatile private var ortSession: OrtSession? = null
    @Volatile private var liteRtBackend: RTMDetLiteRtBackend? = null

    /**
     * Resolves the backend to run a frame on, reviving it when the shared instance has been closed.
     *
     * [liteRtBackend] caches the process-wide singleton owned by the companion, but
     * `scheduleSharedBackendRelease` closes that singleton and clears only the companion's own
     * reference — every manager instance is left holding a dead object. The next frame then failed
     * `check(!executor.isShutdown)` inside `RTMDetLiteRtBackend.run` and threw
     * "RTMDet LiteRT backend is closed", which repeated for every subsequent frame and disabled
     * segmentation until the process was restarted. Re-acquiring through `sharedLiteRtBackend`
     * reuses the live singleton when one exists and rebuilds it otherwise.
     */
    private fun activeLiteRtBackend(): RTMDetLiteRtBackend? {
        liteRtBackend?.takeIf { !it.isClosed }?.let { return it }
        val revived = try {
            sharedLiteRtBackend(context)
        } catch (error: Exception) {
            LogUtil.e(TAG, "Could not revive closed RTMDet LiteRT backend: ${error.message}", error)
            null
        }
        if (revived != null) {
            LogUtil.i(TAG, "Revived shared RTMDet LiteRT backend after it was closed")
            liteRtBackend = revived
        }
        return revived
    }
    private var loadedOnnxAssetName: String? = null

    private data class OnnxBackend(
        val env: OrtEnvironment,
        val session: OrtSession,
        @Suppress("unused") val options: SessionOptions,
        val assetName: String,
        val inputName: String,
        val inputWidth: Int,
        val inputHeight: Int,
        val embedsRtmdetPreprocess: Boolean,
        val executionProvider: String,
        val isRtmdetRaw: Boolean,
    )

    /** Initializes the packaged FP16 LiteRT RTMDet backend. */
    fun initializeAuto(): Boolean {
        LogUtil.i(TAG, "Initializing furniture segmentation backend...")

        val backend = try {
            sharedLiteRtBackend(context)
        } catch (e: Exception) {
            LogUtil.w(TAG, "LiteRT initialization failed: ${e.message}")
            null
        }
        if (backend == null) {
            LogUtil.e(TAG, "LiteRT initialization failed - segmentation disabled")
            return false
        }

        liteRtBackend = backend
        ortBackend = null
        ortEnv = null
        ortSession = null
        loadedOnnxAssetName = null
        LogUtil.i(TAG, "Using ${backend.executionProvider} RTMDet backend")
        return true
    }

    private fun inputFloatBuffer(floatCount: Int): FloatBuffer {
        val workspace = currentInferenceWorkspace()
        val current = workspace.inputBuffer
        val buffer = if (current == null || workspace.inputFloatCount != floatCount) {
            workspace.inputTensor?.close()
            workspace.inputTensor = null
            workspace.inputShape = null
            ByteBuffer
                .allocateDirect(floatCount * java.lang.Float.BYTES)
                .order(ByteOrder.nativeOrder())
                .asFloatBuffer()
                .also {
                    workspace.inputBuffer = it
                    workspace.inputFloatCount = floatCount
                    LogUtil.i(TAG, "Allocated direct ONNX input buffer: floats=$floatCount bytes=${floatCount * java.lang.Float.BYTES}")
                }
        } else {
            current
        }
        buffer.clear()
        return buffer
    }

    private fun inputTensor(env: OrtEnvironment, buffer: FloatBuffer, shape: LongArray): OnnxTensor {
        val workspace = currentInferenceWorkspace()
        val currentTensor = workspace.inputTensor
        val currentShape = workspace.inputShape
        if (currentTensor != null && currentShape != null && currentShape.contentEquals(shape)) {
            buffer.rewind()
            return currentTensor
        }

        currentTensor?.close()
        buffer.rewind()
        val tensor = OnnxTensor.createTensor(env, buffer, shape)
        workspace.inputTensor = tensor
        workspace.inputShape = shape.copyOf()
        LogUtil.i(TAG, "Created reusable ONNX input tensor: shape=${shape.contentToString()}")
        return tensor
    }

    fun segmentImageAsync(frame: Bitmap?, callback: (Bitmap?) -> Unit) {
        // Wrapper that discards detection info
        segmentWithDetectionsAsync(frame) { result ->
            callback(result?.mask)
        }
    }

    fun segmentWithDetectionsAsync(
        frame: Bitmap?,
        callback: (SegmentationResult?) -> Unit,
    ) {
        analyzeFrameAsync(frame, includeMask = true, selectedClassIds = emptySet(), pinnedDetections = null, callback = callback)
    }

    /** Live-frame variant: release the frame-admission gate before posting overlay work to main. */
    fun segmentWithDetectionsOnInferenceThreadAsync(
        frame: Bitmap?,
        callback: (SegmentationResult?) -> Unit,
    ) {
        analyzeFrameAsync(
            frame,
            includeMask = true,
            selectedClassIds = emptySet(),
            pinnedDetections = null,
            callbackDelivery = CallbackDelivery.INFERENCE,
            callback = callback,
        )
    }

    fun detectWithDetectionsAsync(
        frame: Bitmap?,
        requireClusters: Boolean = false,
        callback: (SegmentationResult?) -> Unit,
    ) {
        analyzeFrameAsync(
            frame,
            includeMask = false,
            selectedClassIds = emptySet(),
            pinnedDetections = null,
            requireClusters = requireClusters,
            callback = callback,
        )
    }

    fun detectWithDetectionsOnInferenceThreadAsync(
        frame: Bitmap?,
        requireClusters: Boolean = false,
        callback: (SegmentationResult?) -> Unit,
    ) {
        analyzeFrameAsync(
            frame,
            includeMask = false,
            selectedClassIds = emptySet(),
            pinnedDetections = null,
            requireClusters = requireClusters,
            callbackDelivery = CallbackDelivery.INFERENCE,
            callback = callback,
        )
    }

    fun segmentSelectedClassesAsync(
        frame: Bitmap?,
        selectedClassIds: Set<Int>,
        callback: (SegmentationResult?) -> Unit,
    ) {
        analyzeFrameAsync(frame, includeMask = true, selectedClassIds = selectedClassIds, pinnedDetections = null, callback = callback)
    }

    /**
     * Segment only the object instances matching [pinnedDetections] (same class + IoU to live box), not every box of that class.
     */
    fun segmentSelectedInstancesAsync(
        frame: Bitmap?,
        pinnedDetections: List<DetectionResult>,
        callback: (SegmentationResult?) -> Unit,
    ) {
        analyzeFrameAsync(
            frame,
            includeMask = true,
            selectedClassIds = emptySet(),
            pinnedDetections = pinnedDetections,
            callback = callback,
        )
    }

    fun segmentSelectedInstancesOnInferenceThreadAsync(
        frame: Bitmap?,
        pinnedDetections: List<DetectionResult>,
        callback: (SegmentationResult?) -> Unit,
    ) {
        analyzeFrameAsync(
            frame,
            includeMask = true,
            selectedClassIds = emptySet(),
            pinnedDetections = pinnedDetections,
            callbackDelivery = CallbackDelivery.INFERENCE,
            callback = callback,
        )
    }

    private fun analyzeFrameAsync(
        frame: Bitmap?,
        includeMask: Boolean,
        selectedClassIds: Set<Int>,
        pinnedDetections: List<DetectionResult>? = null,
        requireClusters: Boolean = false,
        callbackDelivery: CallbackDelivery = CallbackDelivery.MAIN,
        callback: (SegmentationResult?) -> Unit,
    ) {
        if (frame == null) {
            if (callbackDelivery == CallbackDelivery.MAIN) {
                mainHandler.postDelayed({ deliverCallback(callback, CallbackDelivery.MAIN, null) }, 200)
            } else {
                deliverCallback(callback, callbackDelivery, null)
            }
            return
        }

        val acceptedGeneration = inferenceGeneration.get()
        val submitted = executeOnSharedInferenceLane inferenceTask@{
            if (inferenceGeneration.get() != acceptedGeneration) {
                deliverCallback(callback, callbackDelivery, null)
                return@inferenceTask
            }

            try {
                if (inferenceGeneration.get() != acceptedGeneration) {
                    deliverCallback(callback, callbackDelivery, null)
                    return@inferenceTask
                }
                if (ortSession != null) {
                    val runOptions = try {
                        OrtSession.RunOptions()
                    } catch (e: Exception) {
                        LogUtil.e(TAG, "Could not create ONNX run options", e)
                        deliverCallback(callback, callbackDelivery, null)
                        return@inferenceTask
                    }
                    activeRunOptions.set(runOptions)
                    try {
                        runOnnxInferenceWithDetections(
                            frame,
                            includeMask,
                            selectedClassIds,
                            pinnedDetections,
                            requireClusters,
                            runOptions,
                            acceptedGeneration,
                            callbackDelivery,
                            callback,
                        )
                    } finally {
                        activeRunOptions.compareAndSet(runOptions, null)
                        try {
                            runOptions.close()
                        } catch (e: Exception) {
                            LogUtil.w(TAG, "Could not close ONNX run options: ${e.message}")
                        }
                    }
                    return@inferenceTask
                }

                if (liteRtBackend != null) {
                    runLiteRtInferenceWithDetections(
                        frame = frame,
                        includeMask = includeMask,
                        selectedClassIds = selectedClassIds,
                        pinnedDetections = pinnedDetections,
                        requireClusters = requireClusters,
                        acceptedGeneration = acceptedGeneration,
                        callbackDelivery = callbackDelivery,
                        callback = callback,
                    )
                    return@inferenceTask
                }

                deliverCallback(callback, callbackDelivery, null)
            } catch (e: Exception) {
                LogUtil.e("FurnitureFitManager", "inference error", e)
                deliverCallback(callback, callbackDelivery, null)
            }
        }
        if (!submitted) {
            deliverCallback(callback, callbackDelivery, null)
        }
    }

    private fun deliverCallback(
        callback: (SegmentationResult?) -> Unit,
        delivery: CallbackDelivery,
        result: SegmentationResult?,
    ) {
        val invoke = {
            try {
                callback(result)
            } catch (exception: Exception) {
                LogUtil.e(TAG, "Furniture Fit result callback failed", exception)
            }
        }
        if (delivery == CallbackDelivery.MAIN && Looper.myLooper() != Looper.getMainLooper()) {
            mainHandler.post(invoke)
        } else {
            invoke()
        }
    }

    /**
     * Upscales the 80x80 RTMDet mask to frame resolution and writes a cutout bitmap.
     * Non-object pixels must be fully transparent; otherwise the live camera frame leaks over the
     * rendered room as a visible rectangular/background patch.
     */
    private fun composeProtoMaskCutoutArgb(
        framePixels: IntArray,
        outPixels: IntArray,
        maskProto: FloatArray,
        frameW: Int,
        frameH: Int,
        protoW: Int,
        protoH: Int,
        x0: Int,
        x1: Int,
        y0: Int,
        y1: Int,
    ) {
        outPixels.fill(0x00000000)
        if (x0 >= x1 || y0 >= y1 || frameW <= 0 || frameH <= 0) return
        val xStart = x0.coerceIn(0, frameW)
        val xEnd = x1.coerceIn(0, frameW)
        val yStart = y0.coerceIn(0, frameH)
        val yEnd = y1.coerceIn(0, frameH)
        if (xStart >= xEnd || yStart >= yEnd) return

        val scaleX = protoW.toFloat() / frameW.toFloat()
        val scaleY = protoH.toFloat() / frameH.toFloat()
        val maxProtoX = protoW - 1
        val maxProtoY = protoH - 1
        val lowerThreshold = RAW_MASK_RENDER_THRESHOLD - RAW_MASK_RENDER_ANTIALIAS_HALF_WIDTH
        val upperThreshold = RAW_MASK_RENDER_THRESHOLD + RAW_MASK_RENDER_ANTIALIAS_HALF_WIDTH
        val thresholdSpan = upperThreshold - lowerThreshold

        for (y in yStart until yEnd) {
            val sampleY = (y + 0.5f) * scaleY - 0.5f
            val baseY = floor(sampleY.toDouble()).toInt()
            val weightY = (sampleY - baseY).coerceIn(0f, 1f)
            val topRow = baseY.coerceIn(0, maxProtoY) * protoW
            val bottomRow = (baseY + 1).coerceIn(0, maxProtoY) * protoW
            val frameRow = y * frameW
            for (x in xStart until xEnd) {
                val sampleX = (x + 0.5f) * scaleX - 0.5f
                val baseX = floor(sampleX.toDouble()).toInt()
                val weightX = (sampleX - baseX).coerceIn(0f, 1f)
                val leftColumn = baseX.coerceIn(0, maxProtoX)
                val rightColumn = (baseX + 1).coerceIn(0, maxProtoX)
                val top = maskProto[topRow + leftColumn] + (maskProto[topRow + rightColumn] - maskProto[topRow + leftColumn]) * weightX
                val bottom = maskProto[bottomRow + leftColumn] + (maskProto[bottomRow + rightColumn] - maskProto[bottomRow + leftColumn]) * weightX
                val maskValue = top + (bottom - top) * weightY
                if (!maskValue.isFinite()) continue
                val alpha = when {
                    maskValue <= lowerThreshold -> 0
                    maskValue >= upperThreshold -> 255
                    else -> (((maskValue - lowerThreshold) / thresholdSpan) * 255f)
                        .roundToInt()
                }
                if (alpha == 0) continue
                outPixels[frameRow + x] = (alpha shl 24) or (framePixels[frameRow + x] and 0x00FFFFFF)
            }
        }
    }

    /** Swift's raw RTMDet compositor: bilinear sampling, per-instance crop, max-alpha union. */
    private fun composeRtmdetRawMaskCutoutArgb(
        framePixels: IntArray,
        outPixels: IntArray,
        maskPlanes: List<FloatArray?>,
        detections: List<Detection>,
        selectedIndices: List<Int>,
        frameW: Int,
        frameH: Int,
        modelW: Int,
        modelH: Int,
    ): Int {
        outPixels.fill(0x00000000)
        if (frameW <= 0 || frameH <= 0 || selectedIndices.isEmpty()) return 0

        val maskSide = RTMDET_MASK_SIDE
        val maxMaskIndex = maskSide - 1
        val xLower = IntArray(frameW)
        val xUpper = IntArray(frameW)
        val xWeight = FloatArray(frameW)
        val yLower = IntArray(frameH)
        val yUpper = IntArray(frameH)
        val yWeight = FloatArray(frameH)

        fun fillAxis(count: Int, lower: IntArray, upper: IntArray, weight: FloatArray) {
            val scale = maskSide.toFloat() / count.toFloat()
            for (index in 0 until count) {
                val coordinate = ((index + 0.5f) * scale - 0.5f).coerceIn(0f, maxMaskIndex.toFloat())
                val low = floor(coordinate.toDouble()).toInt()
                lower[index] = low
                upper[index] = min(maxMaskIndex, low + 1)
                weight[index] = coordinate - low
            }
        }
        fillAxis(frameW, xLower, xUpper, xWeight)
        fillAxis(frameH, yLower, yUpper, yWeight)

        var paintedPixels = 0
        for (rawIndex in selectedIndices) {
            val plane = maskPlanes.getOrNull(rawIndex) ?: continue
            if (plane.size != maskSide * maskSide) continue
            val detection = detections.getOrNull(rawIndex) ?: continue
            val bounds = RTMDetSwiftParity.paddedSourceBounds(
                box = detection.toSwiftParityBox(),
                modelWidth = modelW.toFloat(),
                modelHeight = modelH.toFloat(),
                sourceWidth = frameW,
                sourceHeight = frameH,
            ) ?: continue

            for (y in bounds.minY..bounds.maxY) {
                val upperRow = yLower[y] * maskSide
                val lowerRow = yUpper[y] * maskSide
                val wy = yWeight[y]
                val frameRow = y * frameW
                for (x in bounds.minX..bounds.maxX) {
                    val x0 = xLower[x]
                    val x1 = xUpper[x]
                    val wx = xWeight[x]
                    val top = plane[upperRow + x0] + (plane[upperRow + x1] - plane[upperRow + x0]) * wx
                    val bottom = plane[lowerRow + x0] + (plane[lowerRow + x1] - plane[lowerRow + x0]) * wx
                    val alpha = RTMDetSwiftParity.rawMaskRenderAlpha(top + (bottom - top) * wy)
                    if (alpha <= 0) continue
                    val pixelIndex = frameRow + x
                    val existingAlpha = (outPixels[pixelIndex] ushr 24) and 0xFF
                    if (alpha <= existingAlpha) continue
                    if (existingAlpha == 0) paintedPixels++
                    outPixels[pixelIndex] = (alpha shl 24) or (framePixels[pixelIndex] and 0x00FFFFFF)
                }
            }
        }
        return paintedPixels
    }

    private fun logCutoutAlphaStats(stage: String, pixels: IntArray, width: Int, height: Int) {
        if (!BuildConfig.DEBUG) return
        val now = android.os.SystemClock.elapsedRealtime()
        val previous = lastCutoutAlphaStatsLogMs.get()
        if (now - previous < 1_000L || !lastCutoutAlphaStatsLogMs.compareAndSet(previous, now)) return
        if (pixels.isEmpty() || width <= 0 || height <= 0) return
        var opaqueCount = 0
        var translucentCount = 0
        for (pixel in pixels) {
            val alpha = (pixel ushr 24) and 0xFF
            when {
                alpha == 0 -> Unit
                alpha == 255 -> opaqueCount++
                else -> translucentCount++
            }
        }
        fun alphaAt(x: Int, y: Int): Int {
            val safeX = x.coerceIn(0, width - 1)
            val safeY = y.coerceIn(0, height - 1)
            return (pixels[safeY * width + safeX] ushr 24) and 0xFF
        }
        LogUtil.i(
            TAG,
            "Cutout alpha[$stage]: size=${width}x$height opaque=$opaqueCount translucent=$translucentCount " +
                "transparent=${pixels.size - opaqueCount - translucentCount} " +
                "corners=${alphaAt(0, 0)},${alphaAt(width - 1, 0)},${alphaAt(0, height - 1)},${alphaAt(width - 1, height - 1)}",
        )
    }

    private fun runOnnxInference(frame: Bitmap, callback: (Bitmap?) -> Unit) {
        try {
            val session = ortSession ?: run {
                LogUtil.e("FurnitureFitManager", "ortSession is null")
                mainHandler.post { callback(null) }
                return
            }
            val env = ortEnv ?: run {
                LogUtil.e("FurnitureFitManager", "ortEnv is null")
                mainHandler.post { callback(null) }
                return
            }

            // Use first input info to determine shape
            val firstInput = session.inputInfo.entries.firstOrNull()
            if (firstInput == null) {
                LogUtil.w("FurnitureFitManager", "ONNX session has no inputs")
                mainHandler.post { callback(null) }
                return
            }

            val inputName = firstInput.key
            val tensorInfo = firstInput.value.info

            // Use model-declared input H/W instead of hardcoding 640.
            var inputH = 640
            var inputW = 640
            if (tensorInfo is ai.onnxruntime.TensorInfo) {
                val sh = tensorInfo.shape
                // Expected [1,3,H,W] (NCHW). H/W may be -1 if dynamic.
                if (sh.size == 4) {
                    val hCandidate = sh[2].toInt()
                    val wCandidate = sh[3].toInt()
                    if (hCandidate > 0 && wCandidate > 0) {
                        inputH = hCandidate
                        inputW = wCandidate
                    }
                }
            }

            // Resize to model input size
            LogUtil.d(
                "FurnitureFitManager",
                "Resizing frame ${frame.width}x${frame.height} to ${inputW}x${inputH}"
            )
            val resized = Bitmap.createScaledBitmap(frame, inputW, inputH, true).copy(Config.ARGB_8888, false)

            // Prepare float array in NCHW format
            val floatCount = 1 * 3 * inputH * inputW
            val inputFloats = FloatArray(floatCount)
            val intValues = IntArray(resized.width * resized.height)
            resized.getPixels(intValues, 0, resized.width, 0, 0, resized.width, resized.height)

            // Fill NCHW layout: channel-first ordering
            val hw = inputH * inputW
            for (y in 0 until inputH) {
                val rowOff = y * inputW
                for (x in 0 until inputW) {
                    val v = intValues[rowOff + x]
                    val r = ((v shr 16) and 0xFF) / 255.0f
                    val g = ((v shr 8) and 0xFF) / 255.0f
                    val b = (v and 0xFF) / 255.0f
                    val pixelIdx = rowOff + x
                    inputFloats[0 * hw + pixelIdx] = r
                    inputFloats[1 * hw + pixelIdx] = g
                    inputFloats[2 * hw + pixelIdx] = b
                }
            }

            val shapeLong = longArrayOf(1, 3, inputH.toLong(), inputW.toLong())
            LogUtil.d("FurnitureFitManager", "Creating input tensor with shape ${shapeLong.toList()}")
            LogUtil.d(
                "FurnitureFitManager",
                "Input sample - R[0]=${inputFloats[0]}, G[0]=${inputFloats[hw]}, B[0]=${inputFloats[2 * hw]}"
            )
            LogUtil.d("FurnitureFitManager", "Input range - min=${inputFloats.minOrNull()}, max=${inputFloats.maxOrNull()}")

            val tensor = OnnxTensor.createTensor(
                env,
                java.nio.FloatBuffer.wrap(inputFloats),
                shapeLong
            )

            var maskResult: Bitmap? = null

            LogUtil.d("FurnitureFitManager", "Running ONNX inference...")
            session.run(mapOf(inputName to tensor)).use { results ->
                LogUtil.d("FurnitureFitManager", "Inference complete, processing outputs...")

                val outInfos = ortSession!!.outputInfo.entries.toList()

                // Find detection output (3D) and prototype output (4D)
                var detIndex = -1
                var protoIndex = -1
                var detShape: LongArray? = null
                var protoShape: LongArray? = null

                for (i in outInfos.indices) {
                    val info = outInfos[i].value.info
                    if (info is ai.onnxruntime.TensorInfo) {
                        val sh = info.shape
                        LogUtil.d("FurnitureFitManager", "Output $i shape: ${sh.toList()}")
                        if (sh.size == 3 && detIndex == -1) {
                            detIndex = i
                            detShape = sh
                        }
                        if (sh.size == 4 && protoIndex == -1) {
                            protoIndex = i
                            protoShape = sh
                        }
                    }
                }

                if (detIndex == -1 || protoIndex == -1 || detShape == null || protoShape == null) {
                    LogUtil.w("FurnitureFitManager", "Could not find detection/prototype outputs")
                    mainHandler.post { callback(null) }
                    tensor.close()
                    return
                }

                LogUtil.d("FurnitureFitManager", "Detection output[$detIndex] shape: ${detShape.toList()}")
                LogUtil.d("FurnitureFitManager", "Proto output[$protoIndex] shape: ${protoShape.toList()}")

                val detResult = results.get(detIndex)
                val protoResult = results.get(protoIndex)

                val detValue = detResult?.value
                val protoValue = protoResult?.value

                val numFeatures = detShape[1].toInt()
                val numAnchors = detShape[2].toInt()

                val numMaskCoeffs = 32

                // Input size must match the model export (some exports are 768 rather than 640).
                val numClasses = numFeatures - 4 - numMaskCoeffs
                val classStartIdx = 4
                val maskCoeffStartIdx = 4 + numClasses

                val numProtos = protoShape[1].toInt()
                val protoH = protoShape[2].toInt()
                val protoW = protoShape[3].toInt()

                LogUtil.d(
                    "FurnitureFitManager",
                    "Features=$numFeatures Anchors=$numAnchors Classes=$numClasses MaskCoeffs=$numMaskCoeffs Protos=$numProtos ProtoSize=${protoW}x${protoH}"
                )

                if (numFeatures < (4 + numMaskCoeffs + 1) || numAnchors <= 0 || numProtos <= 0) {
                    LogUtil.e("FurnitureFitManager", "Invalid tensor dimensions")
                    mainHandler.post { callback(null) }
                    tensor.close()
                    return
                }

                LogUtil.d("FurnitureFitManager", "DetValue type: ${detValue?.javaClass}")

                // Extract detection tensor (3D preferred)
                var det3d: Array<Array<FloatArray>>? = null
                var detFlat: FloatArray? = null

                when (detValue) {
                    is Array<*> -> {
                        try {
                            @Suppress("UNCHECKED_CAST")
                            det3d = detValue as Array<Array<FloatArray>>
                            LogUtil.d(
                                "FurnitureFitManager",
                                "Det as 3D array: [${det3d.size}][${det3d[0].size}][${det3d[0][0].size}]"
                            )
                        } catch (e: Exception) {
                            LogUtil.w(
                                "FurnitureFitManager",
                                "Failed to cast as 3D array, trying flatten: ${e.message}"
                            )
                            detFlat = extractFloatArray(detValue)
                        }
                    }
                    is FloatArray -> {
                        detFlat = detValue
                        LogUtil.d("FurnitureFitManager", "Det as flat FloatArray: ${detFlat.size}")
                    }
                    is java.nio.FloatBuffer -> {
                        detFlat = FloatArray(detValue.remaining())
                        detValue.get(detFlat)
                        LogUtil.d("FurnitureFitManager", "Det as FloatBuffer: ${detFlat.size}")
                    }
                    else -> {
                        LogUtil.w("FurnitureFitManager", "Unknown det type: ${detValue?.javaClass}")
                        detFlat = extractFloatArray(detValue)
                    }
                }

                if (det3d == null && (detFlat == null || detFlat.isEmpty())) {
                    LogUtil.e("FurnitureFitManager", "Could not extract detection tensor")
                    mainHandler.post { callback(null) }
                    tensor.close()
                    return
                }

                // Extract prototype tensor
                LogUtil.d("FurnitureFitManager", "Extracting proto array...")
                val proto = extractFloatArray(protoValue)
                LogUtil.d("FurnitureFitManager", "Proto extracted: ${proto.size} floats")

                if (proto.isEmpty()) {
                    LogUtil.w("FurnitureFitManager", "Empty proto output")
                    mainHandler.post { callback(null) }
                    tensor.close()
                    return
                }

                LogUtil.d(
                    "FurnitureFitManager",
                    "Proto[0]=${proto[0]}, Proto[1]=${proto[1]}, Proto[160]=${proto.getOrNull(160)}, Proto[25600]=${proto.getOrNull(25600)}"
                )

                val stride = numAnchors

                val confThreshold = 0.25f
                val iouThreshold = 0.5f
                val maxDetections = 100

                val detections = mutableListOf<Detection>()

                val det3dArr = det3d
                val getDetValue: (Int, Int) -> Float = if (det3dArr != null) {
                    { feature, anchor -> det3dArr[0][feature][anchor] }
                } else {
                    { feature, anchor -> detFlat!![feature * stride + anchor] }
                }

                // Debug: find global max class score
                var globalMaxScore = Float.MIN_VALUE
                var globalMaxAnchor = -1
                var globalMaxClass = -1
                for (anchor in 0 until numAnchors step 10) {
                    for (c in 0 until numClasses) {
                        val score = getDetValue(classStartIdx + c, anchor)
                        if (score > globalMaxScore) {
                            globalMaxScore = score
                            globalMaxAnchor = anchor
                            globalMaxClass = c
                        }
                    }
                }
                LogUtil.d(
                    "FurnitureFitManager",
                    "Global max class score: $globalMaxScore at anchor $globalMaxAnchor, class $globalMaxClass"
                )

                val dbgAnchor = 100
                val dbgX = getDetValue(0, dbgAnchor)
                val dbgY = getDetValue(1, dbgAnchor)
                val dbgW = getDetValue(2, dbgAnchor)
                val dbgH = getDetValue(3, dbgAnchor)
                val dbgC0 = getDetValue(4, dbgAnchor)
                val dbgC1 = getDetValue(5, dbgAnchor)
                LogUtil.d(
                    "FurnitureFitManager",
                    "Anchor[$dbgAnchor]: bbox=($dbgX,$dbgY,$dbgW,$dbgH), class0=$dbgC0, class1=$dbgC1"
                )

                LogUtil.d(
                    "FurnitureFitManager",
                    "Scanning $numAnchors anchors with conf threshold $confThreshold..."
                )
                val startTime = System.currentTimeMillis()

                for (anchor in 0 until numAnchors) {
                    var maxClassScore = Float.MIN_VALUE
                    var bestClassId = -1
                    for (c in 0 until numClasses) {
                        val score = getDetValue(classStartIdx + c, anchor)
                        if (score > maxClassScore) {
                            maxClassScore = score
                            bestClassId = c
                        }
                    }

                    if (maxClassScore > confThreshold) {
                        val x = getDetValue(0, anchor)
                        val y = getDetValue(1, anchor)
                        val bw = getDetValue(2, anchor)
                        val bh = getDetValue(3, anchor)

                        if (x.isFinite() && y.isFinite() && bw.isFinite() && bh.isFinite() && bw > 0 && bh > 0) {
                            val coeffs = FloatArray(numProtos)
                            for (c in 0 until numProtos) {
                                coeffs[c] = getDetValue(maskCoeffStartIdx + c, anchor)
                            }

                            detections.add(
                                Detection(
                                    anchorIdx = anchor,
                                    x = x, y = y, w = bw, h = bh,
                                    confidence = maxClassScore,
                                    classId = bestClassId,
                                    coeffs = coeffs
                                )
                            )
                        }
                    }
                }

                val scanTime = System.currentTimeMillis() - startTime
                LogUtil.d(
                    "FurnitureFitManager",
                    "Found ${detections.size} detections above conf $confThreshold in ${scanTime}ms"
                )

                if (detections.isEmpty()) {
                    mainHandler.post { callback(null) }
                    tensor.close()
                    return
                }

                val topDets = detections.sortedByDescending { it.confidence }.take(5)
                LogUtil.d("FurnitureFitManager", "=== TOP DETECTIONS ===")
                for ((idx, det) in topDets.withIndex()) {
                    val label = labelForClassId(det.classId)
                    LogUtil.d("FurnitureFitManager", "  [$idx] $label: conf=${String.format("%.3f", det.confidence)}")
                }
                LogUtil.d("FurnitureFitManager", "======================")

                val sortedDets = detections.sortedByDescending { it.confidence }.take(maxDetections)

                val keepDets = mutableListOf<Detection>()
                val suppressed = BooleanArray(sortedDets.size)

                for (i in sortedDets.indices) {
                    if (suppressed[i]) continue
                    keepDets.add(sortedDets[i])

                    for (j in i + 1 until sortedDets.size) {
                        if (suppressed[j]) continue
                        val iou = calculateIoU(sortedDets[i], sortedDets[j])
                        if (iou > iouThreshold) suppressed[j] = true
                    }
                }

                LogUtil.d("FurnitureFitManager", "After NMS: ${keepDets.size} detections kept")

                if (keepDets.isNotEmpty()) {
                    val firstDet = keepDets[0]
                    LogUtil.d(
                        "FurnitureFitManager",
                        "First det coeffs[0..3]: ${firstDet.coeffs[0]}, ${firstDet.coeffs[1]}, ${firstDet.coeffs[2]}, ${firstDet.coeffs[3]}"
                    )
                }

                // --- FIX #2: protoScale must use actual model inputW/inputH (768 vs 640) ---
                val protoScaleX = inputW.toFloat() / protoW.toFloat()
                val protoScaleY = inputH.toFloat() / protoH.toFloat()

                val maskProto = FloatArray(protoH * protoW)

                // Generate combined mask from all detections
                // NOTE: Your bbox values might already be in input pixel coords OR model head coords.
                // The biggest real-world cause of "green zigzag" here was resizing to 640 when the model is 768.
                for (detection in keepDets) {
                    // Convert bbox from input coords to proto coords
                    val bboxLeft = ((detection.x - detection.w / 2f) / protoScaleX).toInt().coerceIn(0, protoW - 1)
                    val bboxTop = ((detection.y - detection.h / 2f) / protoScaleY).toInt().coerceIn(0, protoH - 1)
                    val bboxRight = ((detection.x + detection.w / 2f) / protoScaleX).toInt().coerceIn(0, protoW - 1)
                    val bboxBottom = ((detection.y + detection.h / 2f) / protoScaleY).toInt().coerceIn(0, protoH - 1)

                    // Only compute mask within bbox region
                    for (py in bboxTop..bboxBottom) {
                        val rowBase = py * protoW
                        for (px in bboxLeft..bboxRight) {
                            var sum = 0f
                            val p = rowBase + px
                            val hwProto = protoH * protoW
                            var c = 0
                            while (c < numProtos) {
                                val protoIdx = c * hwProto + p
                                sum += detection.coeffs[c] * proto[protoIdx]
                                c++
                            }
                            val sigmoidVal = 1f / (1f + exp(-sum))
                            if (sigmoidVal > maskProto[p]) {
                                maskProto[p] = sigmoidVal
                            }
                        }
                    }
                }

                // Debug mask values
                val maskMin = maskProto.minOrNull() ?: 0f
                val maskMax = maskProto.maxOrNull() ?: 0f
                val maskAbove05 = maskProto.count { it > 0.5f }
                LogUtil.d("FurnitureFitManager", "Mask stats: min=$maskMin, max=$maskMax, pixels>0.5=$maskAbove05")

                // Create mask bitmap from computed maskProto values
                val maskBmp = Bitmap.createBitmap(protoW, protoH, Config.ARGB_8888)
                val pixels = IntArray(protoW * protoH)
                for (i in pixels.indices) {
                    val v = maskProto[i]
                    // Semi-transparent green where mask > 0.5
                    val alpha = if (v > 0.5f) 0xCC else 0x00
                    pixels[i] = (alpha shl 24) or 0x00FF00
                }
                maskBmp.setPixels(pixels, 0, protoW, 0, 0, protoW, protoH)

                // Scale mask to original frame size
                val outMask = Bitmap.createScaledBitmap(maskBmp, frame.width, frame.height, true)
                maskResult = outMask
                LogUtil.d("FurnitureFitManager", "Mask generated: ${outMask.width}x${outMask.height}")
            }

            tensor.close()
            val finalMask = maskResult
            mainHandler.post { callback(finalMask) }
        } catch (e: OrtException) {
            LogUtil.e("FurnitureFitManager", "ONNX inference failed", e)
            mainHandler.post { callback(null) }
        } catch (e: Exception) {
            LogUtil.e("FurnitureFitManager", "ONNX inference exception", e)
            mainHandler.post { callback(null) }
        }
    }

    private fun runLiteRtInferenceWithDetections(
        frame: Bitmap,
        includeMask: Boolean,
        selectedClassIds: Set<Int>,
        pinnedDetections: List<DetectionResult>?,
        requireClusters: Boolean,
        acceptedGeneration: Long,
        callbackDelivery: CallbackDelivery,
        callback: (SegmentationResult?) -> Unit,
    ) {
        val result = try {
            runLiteRtSegmentationOnce(
                frame = frame,
                includeMask = includeMask,
                selectedClassIds = selectedClassIds,
                pinnedDetections = pinnedDetections,
                requireClusters = requireClusters,
                acceptedGeneration = acceptedGeneration,
            )
        } catch (e: Exception) {
            if (inferenceGeneration.get() != acceptedGeneration) {
                LogUtil.d(TAG, "Discarded abandoned LiteRT inference generation=$acceptedGeneration")
            } else {
                LogUtil.e(TAG, "LiteRT inference with detections failed", e)
            }
            null
        }
        deliverCallback(callback, callbackDelivery, result)
    }

    /** One persistent LiteRT GPU forward followed by the same Swift-parity Kotlin postprocess. */
    private fun runLiteRtSegmentationOnce(
        frame: Bitmap,
        includeMask: Boolean,
        selectedClassIds: Set<Int>,
        pinnedDetections: List<DetectionResult>?,
        requireClusters: Boolean,
        acceptedGeneration: Long,
    ): SegmentationResult? {
        val backend = activeLiteRtBackend() ?: return null
        val totalStartNanos = System.nanoTime()
        val inputW = backend.inputWidth
        val inputH = backend.inputHeight
        val usesLetterboxPreprocess = inputH >= 1280 || inputW >= 1280
        LogUtil.i(
            TAG,
            "Furniture segmentation LiteRT start: ${frame.width}x${frame.height} -> " +
                "${if (usesLetterboxPreprocess) "letterbox" else "stretch"} ${inputW}x$inputH " +
                "provider=${backend.executionProvider} embeddedPreprocess=true",
        )

        val preprocessStartNanos = System.nanoTime()
        val preparedBitmap = preprocessFrameForModel(frame, inputW, inputH, usesLetterboxPreprocess)
        val intValues = reusablePreparedPixels(inputW * inputH)
        preparedBitmap.getPixels(intValues, 0, inputW, 0, 0, inputW, inputH)
        val resizeAndReadMillis = elapsedMillis(preprocessStartNanos)

        val requireRawMasks = includeMask || requireClusters
        val run = backend.run(
            argbPixels = intValues,
            requireKernels = requireRawMasks,
            requireMaskFeat = requireRawMasks,
        )
        if (inferenceGeneration.get() != acceptedGeneration) {
            LogUtil.d(TAG, "Discarded abandoned LiteRT inference generation=$acceptedGeneration")
            return null
        }

        val preprocessMillis = resizeAndReadMillis + run.inputPackingMillis
        val inferenceAndOutputMillis = run.inferenceMillis + run.outputCopyMillis
        LogUtil.i(
            TAG,
            "Furniture segmentation preprocess: ${preprocessMillis}ms " +
                "(resize+read=${resizeAndReadMillis} inputBgrPack=${run.inputPackingMillis})",
        )
        LogUtil.i(
            TAG,
            "Furniture segmentation inference: ${inferenceAndOutputMillis}ms " +
                "(LiteRT invoke=${run.inferenceMillis} native=${run.nativeInferenceMillis ?: -1} " +
                "outputNhwcToNchw=${run.outputCopyMillis})",
        )

        val raw = RtmdetRawOutputs(
            levels = listOf(
                RtmdetLevelOutput(run.cls80, run.bbox80, run.kernel80, side = 80, stride = 8f),
                RtmdetLevelOutput(run.cls40, run.bbox40, run.kernel40, side = 40, stride = 16f),
                RtmdetLevelOutput(run.cls20, run.bbox20, run.kernel20, side = 20, stride = 32f),
            ),
            maskFeat = run.maskFeat,
        )
        return handleRtmdetRawResults(
            frame = frame,
            inputW = inputW,
            inputH = inputH,
            includeMask = includeMask,
            selectedClassIds = selectedClassIds,
            pinnedDetections = pinnedDetections,
            requireClusters = requireClusters,
            results = null,
            rawOverride = raw,
            totalStartNanos = totalStartNanos,
            preprocessMillis = preprocessMillis,
            inferenceMillis = inferenceAndOutputMillis,
        )
    }

    private fun runOnnxInferenceWithDetections(
        frame: Bitmap,
        includeMask: Boolean,
        selectedClassIds: Set<Int>,
        pinnedDetections: List<DetectionResult>?,
        requireClusters: Boolean = false,
        runOptions: OrtSession.RunOptions,
        acceptedGeneration: Long,
        callbackDelivery: CallbackDelivery,
        callback: (SegmentationResult?) -> Unit,
    ) {
        val result = try {
            runOnnxSegmentationOnce(
                frame,
                includeMask,
                selectedClassIds,
                pinnedDetections,
                requireClusters,
                runOptions,
                acceptedGeneration,
            )
        } catch (e: Exception) {
            LogUtil.e("FurnitureFitManager", "ONNX inference with detections failed", e)
            null
        }
        deliverCallback(callback, callbackDelivery, result)
    }

    /** Single ONNX forward + mask; no main-thread hop. */
    private fun runOnnxSegmentationOnce(
        frame: Bitmap,
        includeMask: Boolean,
        selectedClassIds: Set<Int>,
        pinnedDetections: List<DetectionResult>? = null,
        requireClusters: Boolean = false,
        runOptions: OrtSession.RunOptions,
        acceptedGeneration: Long,
    ): SegmentationResult? {
        return try {
            val totalStartNanos = System.nanoTime()
            val backend = ortBackend
            val session = backend?.session ?: ortSession ?: return null
            val env = backend?.env ?: ortEnv ?: return null
            val inputName = backend?.inputName ?: session.inputInfo.entries.firstOrNull()?.key ?: return null
            val inputW = backend?.inputWidth ?: RTMDET_INPUT_SIZE
            val inputH = backend?.inputHeight ?: RTMDET_INPUT_SIZE
            val usesLetterboxPreprocess = inputH >= 1280 || inputW >= 1280
            val isRtmdetRaw = backend?.isRtmdetRaw ?: isRtmdetRawSession(session)
            val embedsRtmdetPreprocess = backend?.embedsRtmdetPreprocess == true
            LogUtil.i(
                TAG,
                "Furniture segmentation ONNX start: ${frame.width}x${frame.height} -> " +
                    "${if (usesLetterboxPreprocess) "letterbox" else "stretch"} ${inputW}x${inputH} " +
                    "provider=${backend?.executionProvider ?: "unknown"} " +
                    "embeddedPreprocess=$embedsRtmdetPreprocess",
            )

            val preprocessStartNanos = System.nanoTime()
            val preparedBitmap = preprocessFrameForModel(frame, inputW, inputH, usesLetterboxPreprocess)
            val intValues = reusablePreparedPixels(inputW * inputH)
            preparedBitmap.getPixels(intValues, 0, inputW, 0, 0, inputW, inputH)
            val inputBuffer = inputFloatBuffer(3 * inputH * inputW)
            val hw = inputH * inputW
            for (y in 0 until inputH) {
                val rowOff = y * inputW
                for (x in 0 until inputW) {
                    val v = intValues[rowOff + x]
                    val pixelIdx = rowOff + x
                    val r = ((v shr 16) and 0xFF).toFloat()
                    val g = ((v shr 8) and 0xFF).toFloat()
                    val b = (v and 0xFF).toFloat()
                    if (isRtmdetRaw) {
                        if (embedsRtmdetPreprocess) {
                            // Match Swift's Core ML ImageType(BGR): raw 0...255 BGR enters the
                            // graph, where mean/std normalization is part of the model contract.
                            inputBuffer.put(0 * hw + pixelIdx, b)
                            inputBuffer.put(1 * hw + pixelIdx, g)
                            inputBuffer.put(2 * hw + pixelIdx, r)
                        } else {
                            // Backward compatibility for the original Android raw-head asset.
                            inputBuffer.put(0 * hw + pixelIdx, (b - 103.53f) / 57.375f)
                            inputBuffer.put(1 * hw + pixelIdx, (g - 116.28f) / 57.12f)
                            inputBuffer.put(2 * hw + pixelIdx, (r - 123.675f) / 58.395f)
                        }
                    } else {
                        inputBuffer.put(0 * hw + pixelIdx, r / 255.0f)
                        inputBuffer.put(1 * hw + pixelIdx, g / 255.0f)
                        inputBuffer.put(2 * hw + pixelIdx, b / 255.0f)
                    }
                }
            }
            val preprocessMillis = elapsedMillis(preprocessStartNanos)
            LogUtil.i(TAG, "Furniture segmentation preprocess: ${preprocessMillis}ms")

            val shapeLong = longArrayOf(1, 3, inputH.toLong(), inputW.toLong())
            val tensor = inputTensor(env, inputBuffer, shapeLong)

            val inferenceStartNanos = System.nanoTime()
            val requestedOutputs =
                if (isRtmdetRaw && !includeMask && !requireClusters && selectedClassIds.isEmpty() && pinnedDetections.isNullOrEmpty()) {
                    RTMDET_DETECTION_OUTPUTS
                } else {
                    null
                }
            val inputs = mapOf(inputName to tensor)
            val runResults =
                if (requestedOutputs == null) {
                    session.run(inputs, runOptions)
                } else {
                    session.run(inputs, requestedOutputs, runOptions)
                }
            runResults.use { results ->
                val inferenceMillis = elapsedMillis(inferenceStartNanos)
                LogUtil.i(TAG, "Furniture segmentation inference: ${inferenceMillis}ms")

                if (isRtmdetRaw) {
                    return handleRtmdetRawResults(
                        frame = frame,
                        inputW = inputW,
                        inputH = inputH,
                        includeMask = includeMask,
                        selectedClassIds = selectedClassIds,
                        pinnedDetections = pinnedDetections,
                        requireClusters = requireClusters,
                        results = results,
                        totalStartNanos = totalStartNanos,
                        preprocessMillis = preprocessMillis,
                        inferenceMillis = inferenceMillis,
                    )
                }

                val outputSelectStartNanos = System.nanoTime()
                val outputs = selectDetectionProtoOutputs(session, results) ?: run {
                    LogUtil.e(TAG, "Could not find detection/prototype outputs in segmentation session")
                    return null
                }
                LogUtil.i(
                    TAG,
                    "Furniture segmentation outputs: det=${outputs.detectionName}${outputs.detectionShape.contentToString()} proto=${outputs.protoName}${outputs.protoShape.contentToString()} in ${elapsedMillis(outputSelectStartNanos)}ms"
                )

                val detValue = outputs.detectionValue
                val protoValue = outputs.protoValue
                val numMaskCoeffs = outputs.protoShape[1].toInt()

                val parseStartNanos = System.nanoTime()
                val proto = extractFloatArray(protoValue)
                val protoH = outputs.protoShape[2].toInt()
                val protoW = outputs.protoShape[3].toInt()
                val detections = parseDetectionsForCurrentModel(
                    outputs = outputs,
                    detValue = detValue,
                    confidenceThreshold = DEFAULT_CONFIDENCE_THRESHOLD,
                )

                if (detections.isEmpty()) {
                    LogUtil.i(TAG, "Furniture segmentation detections: 0 candidates in ${elapsedMillis(parseStartNanos)}ms")
                    return SegmentationResult(null, emptyList(), inputW, null)
                }

                val sortedDets = detections.sortedByDescending { it.confidence }.take(300)
                val keepDets = mutableListOf<Detection>()
                val suppressed = BooleanArray(sortedDets.size)
                for (i in sortedDets.indices) {
                    if (suppressed[i]) continue
                    keepDets.add(sortedDets[i])
                    for (j in i + 1 until sortedDets.size) {
                        if (suppressed[j]) continue
                        if (sortedDets[i].classId != sortedDets[j].classId) continue
                        val iou = calculateIoU(sortedDets[i], sortedDets[j])
                        if (iou > DEFAULT_NMS_IOU_THRESHOLD) suppressed[j] = true
                    }
                }
                LogUtil.i(
                    TAG,
                    "Furniture segmentation detections: raw=${detections.size} kept=${keepDets.size} parse+nms=${elapsedMillis(parseStartNanos)}ms"
                )

                val pinList = pinnedDetections.orEmpty()
                val restrictToSelection =
                    selectedClassIds.isNotEmpty() || pinList.isNotEmpty()
                val primaryCandidates = when {
                    pinList.isNotEmpty() -> {
                        val iouPinThreshold = 0.45f
                        keepDets.filter { det ->
                            pinList.any { pin ->
                                det.classId == pin.classId && calculateIoU(det, pin) >= iouPinThreshold
                            }
                        }
                    }
                    selectedClassIds.isEmpty() -> keepDets
                    else -> keepDets.filter { it.classId in selectedClassIds }
                }
                val primaryDet = if (!restrictToSelection) {
                    pickPrimaryOnnxDetection(
                        detections = primaryCandidates,
                        frameWidth = inputW.toFloat(),
                        frameHeight = inputH.toFloat(),
                        minimumConfidence = DEFAULT_CONFIDENCE_THRESHOLD,
                    )
                } else {
                    pickPrimaryOnnxDetection(
                        detections = primaryCandidates,
                        frameWidth = inputW.toFloat(),
                        frameHeight = inputH.toFloat(),
                        minimumConfidence = DEFAULT_CONFIDENCE_THRESHOLD,
                    )
                }
                val maskSourceDetections = if (restrictToSelection) primaryCandidates else emptyList()
                val maskDetectionsForBuild = if (restrictToSelection) {
                    maskSourceDetections.map { detection ->
                        expandedPrimaryForMaskBuild(
                            primaryDetection = detection,
                            frameWidth = inputW.toFloat(),
                            frameHeight = inputH.toFloat(),
                        )
                    }
                } else if (primaryDet != null) {
                    listOf(expandedPrimaryForMaskBuild(
                        primaryDetection = primaryDet,
                        frameWidth = inputW.toFloat(),
                        frameHeight = inputH.toFloat(),
                    ))
                } else {
                    emptyList()
                }

                val orderedDisplayDetections = if (primaryDet != null) {
                    buildList {
                        add(primaryDet)
                        for (detection in keepDets) {
                            if (detection.anchorIdx != primaryDet.anchorIdx) {
                                add(detection)
                            }
                        }
                    }
                } else {
                    keepDets
                }
                val detectionResults = orderedDisplayDetections
                    .take(DEFAULT_MAX_DETECTIONS)
                    .map { detection ->
                        DetectionResult(
                            x = detection.x,
                            y = detection.y,
                            w = detection.w,
                            h = detection.h,
                            confidence = detection.confidence,
                            label = labelForClassId(detection.classId),
                            classId = detection.classId,
                        )
                    }
                if (primaryDet != null) {
                    val topLabels = keepDets
                        .take(3)
                        .joinToString(", ") { "${labelForClassId(it.classId)}:${String.format("%.2f", it.confidence)}" }
                    LogUtil.d(
                        TAG,
                        "Primary=${labelForClassId(primaryDet.classId)} conf=${String.format("%.2f", primaryDet.confidence)} " +
                            "maskBuildDets=${maskDetectionsForBuild.size} keepDets=${keepDets.size} top=[$topLabels]",
                    )
                }

                if (!includeMask) {
                    LogUtil.i(TAG, "Furniture segmentation total (detections only): ${elapsedMillis(totalStartNanos)}ms")
                    return SegmentationResult(
                        mask = null,
                        detections = detectionResults,
                        inputSize = inputW,
                        primaryDetection = detectionResults.firstOrNull(),
                    )
                }

                var maskResult: Bitmap? = null
                if (primaryDet != null && proto.isNotEmpty()) {
                    val maskBuildStartNanos = System.nanoTime()
                    val protoScaleX = inputW.toFloat() / protoW.toFloat()
                    val protoScaleY = inputH.toFloat() / protoH.toFloat()
                    val maskProto = FloatArray(protoH * protoW)

                    for (detection in maskDetectionsForBuild) {
                        val bboxLeft = ((detection.x - detection.w / 2f) / protoScaleX).toInt().coerceIn(0, protoW - 1)
                        val bboxTop = ((detection.y - detection.h / 2f) / protoScaleY).toInt().coerceIn(0, protoH - 1)
                        val bboxRight = ((detection.x + detection.w / 2f) / protoScaleX).toInt().coerceIn(0, protoW - 1)
                        val bboxBottom = ((detection.y + detection.h / 2f) / protoScaleY).toInt().coerceIn(0, protoH - 1)

                        for (py in bboxTop..bboxBottom) {
                            val rowBase = py * protoW
                            for (px in bboxLeft..bboxRight) {
                                var sum = 0f
                                val p = rowBase + px
                                val hwProto = protoH * protoW
                                var coeffIndex = 0
                                while (coeffIndex < numMaskCoeffs) {
                                    val protoIdx = coeffIndex * hwProto + p
                                    sum += detection.coeffs[coeffIndex] * proto[protoIdx]
                                    coeffIndex++
                                }
                                val sigmoidVal = 1f / (1f + exp(-sum))
                                if (sigmoidVal > maskProto[p]) {
                                    maskProto[p] = sigmoidVal
                                }
                            }
                        }
                    }

                    if (enableMorphCloseForMask) {
                        applyMorphClose3x3ToFloatMask(
                            mask = maskProto,
                            width = protoW,
                            height = protoH,
                            threshold = 0.5f,
                        )
                    }
                    val clipCandidates = if (!restrictToSelection) maskDetectionsForBuild else primaryCandidates
                    val clipLeftModel = clipCandidates.minOfOrNull { it.x - it.w / 2f } ?: (primaryDet.x - primaryDet.w / 2f)
                    val clipTopModel = clipCandidates.minOfOrNull { it.y - it.h / 2f } ?: (primaryDet.y - primaryDet.h / 2f)
                    val clipRightModel = clipCandidates.maxOfOrNull { it.x + it.w / 2f } ?: (primaryDet.x + primaryDet.w / 2f)
                    val clipBottomModel = clipCandidates.maxOfOrNull { it.y + it.h / 2f } ?: (primaryDet.y + primaryDet.h / 2f)
                    val protoClipLeft = floor((clipLeftModel / protoScaleX).toDouble()).toInt().coerceIn(0, protoW)
                    val protoClipTop = floor((clipTopModel / protoScaleY).toDouble()).toInt().coerceIn(0, protoH)
                    val protoClipRight = ceil((clipRightModel / protoScaleX).toDouble()).toInt().coerceIn(0, protoW)
                    val protoClipBottom = ceil((clipBottomModel / protoScaleY).toDouble()).toInt().coerceIn(0, protoH)
                    clipProtoMaskOutsideRect(
                        mask = maskProto,
                        protoW = protoW,
                        protoH = protoH,
                        clipX0 = protoClipLeft,
                        clipY0 = protoClipTop,
                        clipX1 = protoClipRight,
                        clipY1 = protoClipBottom,
                    )

                    val frameW = frame.width
                    val frameH = frame.height
                    val sxf = frameW.toFloat() / inputW.toFloat()
                    val syf = frameH.toFloat() / inputH.toFloat()
                    val tightFx0 = clipLeftModel * sxf
                    val tightFx1 = clipRightModel * sxf
                    val tightFy0 = clipTopModel * syf
                    val tightFy1 = clipBottomModel * syf
                    val bandMarginW = max(1f, tightFx1 - tightFx0) * bboxExpandMargin
                    val bandMarginH = max(1f, tightFy1 - tightFy0) * bboxExpandMargin
                    val bandX0 = floor((tightFx0 - bandMarginW).toDouble()).toInt().coerceIn(0, frameW)
                    val bandX1 = ceil((tightFx1 + bandMarginW).toDouble()).toInt().coerceIn(0, frameW)
                    val bandY0 = floor((tightFy0 - bandMarginH).toDouble()).toInt().coerceIn(0, frameH)
                    val bandY1 = ceil((tightFy1 + bandMarginH).toDouble()).toInt().coerceIn(0, frameH)

                    val framePixels = reusableFramePixels(frameW * frameH)
                    frame.getPixels(framePixels, 0, frameW, 0, 0, frameW, frameH)

                    val outPixels = reusableOutputPixels(frameW * frameH)
                    composeProtoMaskCutoutArgb(
                        framePixels = framePixels,
                        outPixels = outPixels,
                        maskProto = maskProto,
                        frameW = frameW,
                        frameH = frameH,
                        protoW = protoW,
                        protoH = protoH,
                        x0 = bandX0,
                        x1 = bandX1,
                        y0 = bandY0,
                        y1 = bandY1,
                    )

                    val maskBmp = Bitmap.createBitmap(frameW, frameH, Config.ARGB_8888)
                    maskBmp.setHasAlpha(true)
                    maskBmp.setPixels(outPixels, 0, frameW, 0, 0, frameW, frameH)
                    maskResult = maskBmp
                    logCutoutAlphaStats("generic", outPixels, frameW, frameH)
                    LogUtil.i(TAG, "Furniture cutout mask build: ${elapsedMillis(maskBuildStartNanos)}ms")
                }

                LogUtil.i(TAG, "Furniture segmentation total: ${elapsedMillis(totalStartNanos)}ms")
                SegmentationResult(maskResult, detectionResults, inputW, detectionResults.firstOrNull())
            }
        } catch (e: Exception) {
            if (inferenceGeneration.get() != acceptedGeneration) {
                LogUtil.d(TAG, "Discarded abandoned ONNX inference generation=$acceptedGeneration")
            } else {
                LogUtil.e(TAG, "ONNX segmentation once failed", e)
            }
            null
        }
    }

    private data class DetectionProtoOutputs(
        val detectionName: String,
        val protoName: String,
        val detectionValue: Any?,
        val protoValue: Any?,
        val detectionShape: LongArray,
        val protoShape: LongArray,
    )

    private data class RtmdetLevelOutput(
        val cls: FloatArray,
        val bbox: FloatArray,
        val kernel: FloatArray,
        val side: Int,
        val stride: Float,
    )

    private data class RtmdetRawOutputs(
        val levels: List<RtmdetLevelOutput>,
        val maskFeat: FloatArray,
    )

    private fun elapsedMillis(startNanos: Long): Long =
        (System.nanoTime() - startNanos) / 1_000_000L

    private fun isRtmdetRawSession(session: OrtSession): Boolean {
        return session.outputInfo.containsKey("cls_80") &&
                session.outputInfo.containsKey("bbox_80") &&
                session.outputInfo.containsKey("kernel_80") &&
                session.outputInfo.containsKey("mask_feat")
    }

    private fun preprocessFrameForModel(
        frame: Bitmap,
        inputW: Int,
        inputH: Int,
        usesLetterbox: Boolean,
    ): Bitmap {
        val workspace = currentInferenceWorkspace()
        val current = workspace.preparedBitmap
        val output = if (
            current == null || current.isRecycled || current.width != inputW || current.height != inputH
        ) {
            current?.takeIf { !it.isRecycled }?.recycle()
            Bitmap.createBitmap(inputW, inputH, Config.ARGB_8888).also {
                workspace.preparedBitmap = it
            }
        } else {
            current
        }
        val canvas = Canvas(output)
        val destination = if (!usesLetterbox) {
            RectF(0f, 0f, inputW.toFloat(), inputH.toFloat())
        } else {
            val scale = min(inputW.toFloat() / frame.width.toFloat(), inputH.toFloat() / frame.height.toFloat())
            val scaledW = max(1f, frame.width * scale)
            val scaledH = max(1f, frame.height * scale)
            val left = (inputW - scaledW) * 0.5f
            val top = (inputH - scaledH) * 0.5f
            RectF(left, top, left + scaledW, top + scaledH)
        }

        canvas.drawColor(if (usesLetterbox) Color.rgb(114, 114, 114) else Color.TRANSPARENT)
        canvas.drawBitmap(frame, null, destination, workspace.preprocessPaint)
        return output
    }

    private fun reusablePreparedPixels(size: Int): IntArray {
        val workspace = currentInferenceWorkspace()
        if (workspace.preparedPixels.size != size) workspace.preparedPixels = IntArray(size)
        return workspace.preparedPixels
    }

    private fun reusableFramePixels(size: Int): IntArray {
        val workspace = currentInferenceWorkspace()
        if (workspace.framePixels.size != size) workspace.framePixels = IntArray(size)
        return workspace.framePixels
    }

    private fun reusableOutputPixels(size: Int): IntArray {
        val workspace = currentInferenceWorkspace()
        if (workspace.outputPixels.size != size) workspace.outputPixels = IntArray(size)
        return workspace.outputPixels
    }

    private fun selectDetectionProtoOutputs(
        session: OrtSession,
        results: OrtSession.Result,
    ): DetectionProtoOutputs? {
        val outputEntries = session.outputInfo.entries.toList()
        val knownPairs = listOf(
            "detections" to "protos",
            "output0" to "output1",
            "output" to "proto",
        )
        for ((detName, protoName) in knownPairs) {
            val detIndex = outputEntries.indexOfFirst { it.key == detName }
            val protoIndex = outputEntries.indexOfFirst { it.key == protoName }
            if (detIndex == -1 || protoIndex == -1) continue
            val detectionValue = results.get(detIndex)?.value
            val protoValue = results.get(protoIndex)?.value
            val detectionInfo = session.outputInfo[detName]?.info as? ai.onnxruntime.TensorInfo
            val protoInfo = session.outputInfo[protoName]?.info as? ai.onnxruntime.TensorInfo
            if (detectionValue != null && protoValue != null && detectionInfo != null && protoInfo != null) {
                return DetectionProtoOutputs(
                    detectionName = detName,
                    protoName = protoName,
                    detectionValue = detectionValue,
                    protoValue = protoValue,
                    detectionShape = detectionInfo.shape,
                    protoShape = protoInfo.shape,
                )
            }
        }

        var detectionCandidate: Pair<String, ai.onnxruntime.TensorInfo>? = null
        var protoCandidate: Pair<String, ai.onnxruntime.TensorInfo>? = null
        for ((name, nodeInfo) in session.outputInfo) {
            val tensorInfo = nodeInfo.info as? ai.onnxruntime.TensorInfo ?: continue
            val shape = tensorInfo.shape
            if (shape.size == 4 && shape.getOrNull(1) == 32L && protoCandidate == null) {
                protoCandidate = name to tensorInfo
            } else if (shape.size == 3 && detectionCandidate == null) {
                detectionCandidate = name to tensorInfo
            }
        }
        val det = detectionCandidate ?: return null
        val proto = protoCandidate ?: return null
        val detIndex = outputEntries.indexOfFirst { it.key == det.first }
        val protoIndex = outputEntries.indexOfFirst { it.key == proto.first }
        if (detIndex == -1 || protoIndex == -1) return null
        return DetectionProtoOutputs(
            detectionName = det.first,
            protoName = proto.first,
            detectionValue = results.get(detIndex)?.value,
            protoValue = results.get(protoIndex)?.value,
            detectionShape = det.second.shape,
            protoShape = proto.second.shape,
        )
    }

    private fun handleRtmdetRawResults(
        frame: Bitmap,
        inputW: Int,
        inputH: Int,
        includeMask: Boolean,
        selectedClassIds: Set<Int>,
        pinnedDetections: List<DetectionResult>?,
        requireClusters: Boolean,
        results: OrtSession.Result?,
        rawOverride: RtmdetRawOutputs? = null,
        totalStartNanos: Long,
        preprocessMillis: Long = 0,
        inferenceMillis: Long = 0,
    ): SegmentationResult? {
        val parseStartNanos = System.nanoTime()
        val raw = rawOverride ?: results?.let {
            extractRtmdetRawOutputs(
                results = it,
                requireKernels = includeMask || requireClusters,
                requireMaskFeat = includeMask || requireClusters,
            )
        } ?: run {
            LogUtil.e(TAG, "RTMDet raw outputs missing")
            return null
        }
        val detections = decodeRtmdetRawCandidates(raw, inputW, inputH)
        if (detections.isEmpty()) {
            LogUtil.i(TAG, "RTMDet raw detections: 0 candidates in ${elapsedMillis(parseStartNanos)}ms")
            return SegmentationResult(null, emptyList(), inputW, null)
        }

        val nmsIndices = RTMDetSwiftParity.classAwareNms(
            candidates = detections.map { it.toSwiftParityBox() },
            iouThreshold = DEFAULT_NMS_IOU_THRESHOLD,
            limit = DEFAULT_MAX_DETECTIONS,
        )
        // Swift's live layer ranks the NMS result again: confidence first and, for an exact score
        // tie, larger area first. Keep graph/mask indices aligned to that user-visible ordering.
        val keepDets = nmsIndices
            .map(detections::get)
            .sortedWith { left, right ->
                if (left.confidence == right.confidence) {
                    (right.w * right.h).compareTo(left.w * left.h)
                } else {
                    right.confidence.compareTo(left.confidence)
                }
            }
        val parseNmsMillis = elapsedMillis(parseStartNanos)
        LogUtil.i(
            TAG,
            "RTMDet raw detections: raw=${detections.size} kept=${keepDets.size} parse+nms=${parseNmsMillis}ms",
        )

        val pinList = pinnedDetections.orEmpty()
        if (!includeMask && !requireClusters && selectedClassIds.isEmpty() && pinList.isEmpty()) {
            val fastSelection = RTMDetSwiftParity.selectDefaultCluster(
                candidates = keepDets.map { it.toSwiftParityBox() },
                clusters = keepDets.indices.map { listOf(it) },
                frameWidth = inputW.toFloat(),
                frameHeight = inputH.toFloat(),
            )
            val detectionResults = keepDets.map { detection ->
                    DetectionResult(
                        x = detection.x,
                        y = detection.y,
                        w = detection.w,
                        h = detection.h,
                        confidence = detection.confidence,
                        label = labelForClassId(detection.classId),
                        classId = detection.classId,
                    )
                }
            val primaryResult = fastSelection?.representativeIndex?.let(detectionResults::getOrNull)
            LogUtil.i(TAG, "RTMDet raw total (detections only, fast): ${elapsedMillis(totalStartNanos)}ms")
            LogUtil.i(
                TAG,
                "stageMillis: preprocess=$preprocessMillis inference=$inferenceMillis " +
                    "parse+nms=$parseNmsMillis maskPlanes=0 maskBuild=0 total=${elapsedMillis(totalStartNanos)} " +
                    "(planes=0 includeMask=false fast=true)",
            )
            return SegmentationResult(
                mask = null,
                detections = detectionResults,
                inputSize = inputW,
                primaryDetection = primaryResult,
                detectionClusters = emptyList(),
                sourceWidth = frame.width,
                sourceHeight = frame.height,
            )
        }

        val rawMaskPlaneStartNanos = System.nanoTime()
        val rawMaskPlanes = keepDets.map { buildRtmdetRawMaskPlane(it, raw.maskFeat) }
        val affinityGraph = makeMaskAffinityGraph(rawMaskPlanes)
        val rawMaskPlaneMillis = elapsedMillis(rawMaskPlaneStartNanos)
        LogUtil.i(
            TAG,
            "RTMDet raw mask affinity: planes=${rawMaskPlanes.count { it != null }} clusters=${affinityGraph.clusterCount()} in ${rawMaskPlaneMillis}ms",
        )

        val selectedSeedIndices = if (pinList.isNotEmpty()) {
            selectedSeedIndicesForPins(keepDets, pinList, affinityGraph)
        } else {
            emptyList()
        }
        val selectedMaskRawIndices = if (selectedSeedIndices.isNotEmpty()) {
            affinityGraph.transitiveGroup(selectedSeedIndices).ifEmpty { selectedSeedIndices }.sorted()
        } else {
            emptyList()
        }
        val detectionResults = keepDets.map { detection ->
                DetectionResult(
                    x = detection.x,
                    y = detection.y,
                    w = detection.w,
                    h = detection.h,
                    confidence = detection.confidence,
                    label = labelForClassId(detection.classId),
                    classId = detection.classId,
                )
            }
        val detectionClusters = buildDisplayClustersFromAffinityGraph(
            orderedDisplayRawIndices = keepDets.indices.toList(),
            affinityGraph = affinityGraph,
        )
        val restrictToSelection = selectedClassIds.isNotEmpty() || pinList.isNotEmpty()
        val defaultSelection = if (!restrictToSelection) {
            RTMDetSwiftParity.selectDefaultCluster(
                candidates = keepDets.map { it.toSwiftParityBox() },
                clusters = detectionClusters,
                frameWidth = inputW.toFloat(),
                frameHeight = inputH.toFloat(),
                preferCenter = true,
                confidenceFloor = RTMDET_CONFIDENCE_THRESHOLD,
            )
        } else {
            null
        }
        val selectedClassRawIndices = if (selectedClassIds.isNotEmpty() && pinList.isEmpty()) {
            keepDets.indices.filter { keepDets[it].classId in selectedClassIds }
        } else {
            emptyList()
        }
        val primaryRawIndex = when {
            pinList.isNotEmpty() -> selectedSeedIndices.firstOrNull() ?: -1
            selectedClassRawIndices.isNotEmpty() -> selectedClassRawIndices.first()
            else -> defaultSelection?.representativeIndex ?: -1
        }
        val maskRawIndices = when {
            selectedMaskRawIndices.isNotEmpty() -> selectedMaskRawIndices
            pinList.isNotEmpty() -> emptyList()
            selectedClassRawIndices.isNotEmpty() -> selectedClassRawIndices
            !restrictToSelection -> defaultSelection?.memberIndices.orEmpty()
            else -> emptyList()
        }
        val primaryResult = detectionResults.getOrNull(primaryRawIndex)

        if (!includeMask) {
            LogUtil.i(TAG, "RTMDet raw total (detections only): ${elapsedMillis(totalStartNanos)}ms")
            LogUtil.i(
                TAG,
                "stageMillis: preprocess=$preprocessMillis inference=$inferenceMillis " +
                    "parse+nms=$parseNmsMillis maskPlanes=$rawMaskPlaneMillis maskBuild=0 total=${elapsedMillis(totalStartNanos)} " +
                    "(planes=${rawMaskPlanes.count { it != null }} includeMask=false)",
            )
            return SegmentationResult(
                mask = null,
                detections = detectionResults,
                inputSize = inputW,
                primaryDetection = primaryResult,
                detectionClusters = detectionClusters,
                sourceWidth = frame.width,
                sourceHeight = frame.height,
            )
        }

        var maskBuildMillis = 0L
        var maskResult: Bitmap? = null
        if (maskRawIndices.isNotEmpty()) {
            val maskBuildStartNanos = System.nanoTime()
            val frameW = frame.width
            val frameH = frame.height
            val framePixels = reusableFramePixels(frameW * frameH)
            frame.getPixels(framePixels, 0, frameW, 0, 0, frameW, frameH)
            val outPixels = reusableOutputPixels(frameW * frameH)
            val paintedPixels = composeRtmdetRawMaskCutoutArgb(
                framePixels = framePixels,
                outPixels = outPixels,
                maskPlanes = rawMaskPlanes,
                detections = keepDets,
                selectedIndices = maskRawIndices,
                frameW = frameW,
                frameH = frameH,
                modelW = inputW,
                modelH = inputH,
            )
            if (paintedPixels > 0) {
                maskResult = Bitmap.createBitmap(frameW, frameH, Config.ARGB_8888).also { maskBmp ->
                    maskBmp.setHasAlpha(true)
                    maskBmp.setPixels(outPixels, 0, frameW, 0, 0, frameW, frameH)
                }
                logCutoutAlphaStats("rtmdet", outPixels, frameW, frameH)
            }
            maskBuildMillis = elapsedMillis(maskBuildStartNanos)
            LogUtil.i(TAG, "RTMDet cutout mask build: ${maskBuildMillis}ms")
        }

        LogUtil.i(TAG, "RTMDet raw total: ${elapsedMillis(totalStartNanos)}ms")
        // One consolidated breakdown line (mirrors the iOS `stageMillis:`), so the bottleneck is
        // readable at a glance: preprocess → inference (ONNX backbone) → parse+nms → maskBuild (cutout).
        LogUtil.i(
            TAG,
            "stageMillis: preprocess=$preprocessMillis inference=$inferenceMillis " +
                "parse+nms=$parseNmsMillis maskPlanes=$rawMaskPlaneMillis maskBuild=$maskBuildMillis total=${elapsedMillis(totalStartNanos)} " +
                "(planes=${maskRawIndices.size} includeMask=true)",
        )
        return SegmentationResult(
            mask = maskResult,
            detections = detectionResults,
            inputSize = inputW,
            primaryDetection = primaryResult,
            detectionClusters = detectionClusters,
            sourceWidth = frame.width,
            sourceHeight = frame.height,
        )
    }

    private fun extractRtmdetRawOutputs(
        results: OrtSession.Result,
        requireKernels: Boolean,
        requireMaskFeat: Boolean,
    ): RtmdetRawOutputs? {
        fun floats(name: String): FloatArray {
            val value = results.get(name).orElse(null) ?: return FloatArray(0)
            val tensor = value as? OnnxTensor
            if (tensor != null) {
                val buffer = tensor.floatBuffer ?: return FloatArray(0)
                return FloatArray(buffer.remaining()).also(buffer::get)
            }
            return extractFloatArray(value.value)
        }
        val cls80 = floats("cls_80")
        val cls40 = floats("cls_40")
        val cls20 = floats("cls_20")
        val bbox80 = floats("bbox_80")
        val bbox40 = floats("bbox_40")
        val bbox20 = floats("bbox_20")
        val kernel80 = floats("kernel_80")
        val kernel40 = floats("kernel_40")
        val kernel20 = floats("kernel_20")
        val sourceMaskFeat = floats("mask_feat")
        if (listOf(cls80, cls40, cls20, bbox80, bbox40, bbox20).any { it.isEmpty() }) {
            return null
        }
        if (requireKernels && listOf(kernel80, kernel40, kernel20).any { it.isEmpty() }) {
            return null
        }
        if (requireMaskFeat && sourceMaskFeat.isEmpty()) {
            return null
        }
        val maskFeat = if (sourceMaskFeat.isEmpty()) {
            sourceMaskFeat
        } else {
            val sourceArea = sourceMaskFeat.size / 8
            val sourceSide = sqrt(sourceArea.toDouble()).roundToInt()
            if (sourceSide * sourceSide * 8 != sourceMaskFeat.size) return null
            when (sourceSide) {
                RTMDET_MASK_SIDE -> sourceMaskFeat
                RTMDET_SOURCE_MASK_SIDE -> RTMDetSwiftParity.upsampleMaskFeaturesAlignCornersFalse(
                    source = sourceMaskFeat,
                    channels = 8,
                    sourceSide = sourceSide,
                    targetSide = RTMDET_MASK_SIDE,
                )
                else -> return null
            }
        }
        return RtmdetRawOutputs(
            levels = listOf(
                RtmdetLevelOutput(cls80, bbox80, kernel80, side = 80, stride = 8f),
                RtmdetLevelOutput(cls40, bbox40, kernel40, side = 40, stride = 16f),
                RtmdetLevelOutput(cls20, bbox20, kernel20, side = 20, stride = 32f),
            ),
            maskFeat = maskFeat,
        )
    }

    private fun decodeRtmdetRawCandidates(
        raw: RtmdetRawOutputs,
        inputW: Int,
        inputH: Int,
    ): List<Detection> {
        val detections = mutableListOf<Detection>()
        var anchorBase = 0
        for (level in raw.levels) {
            val side = level.side
            val hw = side * side
            for (y in 0 until side) {
                for (x in 0 until side) {
                    val pos = y * side + x
                    var bestClass = -1
                    var bestLogit = -Float.MAX_VALUE
                    // Swift lets every non-blacklisted COCO class compete and applies sigmoid once
                    // after the raw-logit argmax (sigmoid is monotonic).
                    for (classId in 0 until (level.cls.size / hw)) {
                        if (classId in ignoredClassIds) continue
                        val logit = level.cls[classId * hw + pos]
                        if (logit > bestLogit) {
                            bestLogit = logit
                            bestClass = classId
                        }
                    }
                    val bestScore = sigmoid(bestLogit)
                    if (bestClass < 0 || bestScore < RTMDET_CONFIDENCE_THRESHOLD) continue

                    val centerX = (x + 0.5f) * level.stride
                    val centerY = (y + 0.5f) * level.stride
                    val left = level.bbox[0 * hw + pos]
                    val top = level.bbox[1 * hw + pos]
                    val right = level.bbox[2 * hw + pos]
                    val bottom = level.bbox[3 * hw + pos]
                    if (!left.isFinite() || !top.isFinite() || !right.isFinite() || !bottom.isFinite()) continue

                    val x1 = (centerX - left).coerceIn(0f, inputW.toFloat())
                    val y1 = (centerY - top).coerceIn(0f, inputH.toFloat())
                    val x2 = (centerX + right).coerceIn(0f, inputW.toFloat())
                    val y2 = (centerY + bottom).coerceIn(0f, inputH.toFloat())
                    val width = x2 - x1
                    val height = y2 - y1
                    if (width <= 1f || height <= 1f) continue

                    val kernel = if (level.kernel.isNotEmpty()) {
                        FloatArray(169).also { coeffs ->
                            for (i in 0 until 169) {
                                coeffs[i] = level.kernel[i * hw + pos]
                            }
                        }
                    } else {
                        FloatArray(0)
                    }
                    detections.add(
                        Detection(
                            anchorIdx = anchorBase + pos,
                            x = (x1 + x2) * 0.5f,
                            y = (y1 + y2) * 0.5f,
                            w = width,
                            h = height,
                            confidence = bestScore,
                            classId = bestClass,
                            coeffs = kernel,
                            priorX = centerX,
                            priorY = centerY,
                            levelStride = level.stride,
                        )
                    )
                }
            }
            anchorBase += hw
        }
        return detections.sortedByDescending { it.confidence }
    }

    private fun buildRtmdetRawMaskPlane(
        detection: Detection,
        maskFeat: FloatArray,
    ): FloatArray? {
        if (detection.coeffs.size != 169 || maskFeat.size < 8 * RTMDET_MASK_SIDE * RTMDET_MASK_SIDE) return null
        val maskSide = RTMDET_MASK_SIDE
        val hw = maskSide * maskSide
        val maskStride = RTMDET_INPUT_SIZE.toFloat() / maskSide.toFloat()
        val out = FloatArray(hw)
        val w1 = 0
        val w2 = w1 + 80
        val w3 = w2 + 64
        val b1 = w3 + 8
        val b2 = b1 + 8
        val b3 = b2 + 8
        val input = FloatArray(10)
        val hidden1 = FloatArray(8)
        val hidden2 = FloatArray(8)

        for (y in 0 until maskSide) {
            for (x in 0 until maskSide) {
                val pos = y * maskSide + x
                val gridX = (x + 0.5f) * maskStride
                val gridY = (y + 0.5f) * maskStride
                input[0] = (detection.priorX - gridX) / max(1f, detection.levelStride * 8f)
                input[1] = (detection.priorY - gridY) / max(1f, detection.levelStride * 8f)
                for (c in 0 until 8) {
                    input[2 + c] = maskFeat[c * hw + pos]
                }

                for (o in 0 until 8) {
                    var sum = detection.coeffs[b1 + o]
                    for (i in 0 until 10) {
                        sum += detection.coeffs[w1 + o * 10 + i] * input[i]
                    }
                    hidden1[o] = max(0f, sum)
                }

                for (o in 0 until 8) {
                    var sum = detection.coeffs[b2 + o]
                    for (i in 0 until 8) {
                        sum += detection.coeffs[w2 + o * 8 + i] * hidden1[i]
                    }
                    hidden2[o] = max(0f, sum)
                }

                var logit = detection.coeffs[b3]
                for (i in 0 until 8) {
                    logit += detection.coeffs[w3 + i] * hidden2[i]
                }
                out[pos] = sigmoid(logit)
            }
        }
        return out
    }

    private fun refineRtmdetMaskPlaneForDetection(
        plane: FloatArray,
        detection: Detection,
        inputW: Int,
        inputH: Int,
        protoW: Int,
        protoH: Int,
    ): FloatArray {
        if (plane.size != protoW * protoH || protoW <= 0 || protoH <= 0) return plane
        val protoScaleX = inputW.toFloat() / protoW.toFloat()
        val protoScaleY = inputH.toFloat() / protoH.toFloat()
        val x0 = floor(((detection.x - detection.w / 2f) / protoScaleX).toDouble()).toInt().coerceIn(0, protoW - 1)
        val y0 = floor(((detection.y - detection.h / 2f) / protoScaleY).toDouble()).toInt().coerceIn(0, protoH - 1)
        val x1 = ceil(((detection.x + detection.w / 2f) / protoScaleX).toDouble()).toInt().coerceIn(0, protoW - 1)
        val y1 = ceil(((detection.y + detection.h / 2f) / protoScaleY).toDouble()).toInt().coerceIn(0, protoH - 1)
        if (x1 < x0 || y1 < y0) return FloatArray(plane.size)

        val lowThreshold = 0.50f
        val highThreshold = RTMDET_MASK_KEEP_THRESHOLD
        val visited = BooleanArray(plane.size)
        val queue = IntArray(plane.size)
        val componentPixels = IntArray(plane.size)
        val bboxArea = max(1, (x1 - x0 + 1) * (y1 - y0 + 1))
        val centerX = ((detection.x / protoScaleX).toInt()).coerceIn(x0, x1)
        val centerY = ((detection.y / protoScaleY).toInt()).coerceIn(y0, y1)

        var bestCount = 0
        var bestScore = -Float.MAX_VALUE
        val keep = BooleanArray(plane.size)

        for (startY in y0..y1) {
            for (startX in x0..x1) {
                val start = startY * protoW + startX
                if (visited[start] || plane[start] <= lowThreshold) continue

                var head = 0
                var tail = 0
                var count = 0
                var highCount = 0
                var valueSum = 0f
                var minCenterDistanceSq = Int.MAX_VALUE

                visited[start] = true
                queue[tail++] = start
                while (head < tail) {
                    val idx = queue[head++]
                    componentPixels[count++] = idx
                    val px = idx % protoW
                    val py = idx / protoW
                    val value = plane[idx]
                    valueSum += value
                    if (value >= highThreshold) highCount++
                    val dx = px - centerX
                    val dy = py - centerY
                    val distSq = dx * dx + dy * dy
                    if (distSq < minCenterDistanceSq) minCenterDistanceSq = distSq

                    fun push(nx: Int, ny: Int) {
                        if (nx < x0 || nx > x1 || ny < y0 || ny > y1) return
                        val next = ny * protoW + nx
                        if (visited[next] || plane[next] <= lowThreshold) return
                        visited[next] = true
                        queue[tail++] = next
                    }
                    push(px - 1, py)
                    push(px + 1, py)
                    push(px, py - 1)
                    push(px, py + 1)
                }

                if (count <= 0 || highCount <= 0) continue
                val areaFraction = count.toFloat() / bboxArea.toFloat()
                val averageValue = valueSum / count.toFloat()
                val distancePenalty = minCenterDistanceSq.toFloat() / bboxArea.toFloat()
                val hugePenalty = if (areaFraction > 0.75f && detection.confidence < 0.75f) areaFraction * 2f else 0f
                val score = highCount.toFloat() * 4f + count.toFloat() * averageValue - distancePenalty * 50f - hugePenalty * count.toFloat()
                if (score > bestScore) {
                    bestScore = score
                    bestCount = count
                    keep.fill(false)
                    for (i in 0 until count) {
                        keep[componentPixels[i]] = true
                    }
                }
            }
        }

        val refined = FloatArray(plane.size)
        var keptPixels = 0
        for (i in plane.indices) {
            if (keep[i] && plane[i] >= RTMDET_MASK_KEEP_THRESHOLD) {
                refined[i] = plane[i]
                keptPixels++
            }
        }
        val rawPixels = plane.count { it > lowThreshold }
        LogUtil.i(
            TAG,
            "RTMDet mask refine: ${labelForClassId(detection.classId)} conf=${String.format("%.2f", detection.confidence)} " +
                "raw=$rawPixels kept=$keptPixels bboxProto=${x1 - x0 + 1}x${y1 - y0 + 1} best=$bestCount",
        )
        return refined
    }

    private data class MaskBitset(
        val words: LongArray,
        val onCount: Int,
    )

    private class MaskAffinityGraph(private val neighbors: List<List<Int>>) {
        val nodeCount: Int
            get() = neighbors.size

        fun transitiveGroup(seedIndices: List<Int>): List<Int> {
            if (seedIndices.isEmpty() || neighbors.isEmpty()) return emptyList()
            val inGroup = linkedSetOf<Int>()
            val frontier = ArrayDeque<Int>()
            for (seed in seedIndices) {
                if (seed in neighbors.indices && inGroup.add(seed)) {
                    frontier.add(seed)
                }
            }
            while (frontier.isNotEmpty()) {
                val current = frontier.removeLast()
                for (neighbor in neighbors[current]) {
                    if (neighbor in neighbors.indices && inGroup.add(neighbor)) {
                        frontier.add(neighbor)
                    }
                }
            }
            return inGroup.sorted()
        }

        fun clusterCount(): Int {
            if (neighbors.isEmpty()) return 0
            val visited = BooleanArray(neighbors.size)
            var clusters = 0
            for (index in neighbors.indices) {
                if (visited[index]) continue
                clusters++
                for (member in transitiveGroup(listOf(index))) {
                    visited[member] = true
                }
            }
            return clusters
        }
    }

    private fun makeMaskAffinityGraph(planes: List<FloatArray?>): MaskAffinityGraph {
        val bitsets = planes.map { maskBitsetForPlane(it) }
        val neighbors = Array(planes.size) { mutableListOf<Int>() }
        if (bitsets.size <= 1) return MaskAffinityGraph(neighbors.map { it.toList() })

        for (leftIndex in 0 until bitsets.lastIndex) {
            val left = bitsets[leftIndex]
            if (left.onCount <= 0) continue
            for (rightIndex in (leftIndex + 1) until bitsets.size) {
                val right = bitsets[rightIndex]
                if (right.onCount <= 0) continue
                val intersection = bitsetIntersectionCount(left.words, right.words)
                val affinity = intersection.toFloat() / max(1, min(left.onCount, right.onCount)).toFloat()
                if (affinity >= RAW_MASK_AFFINITY_THRESHOLD) {
                    neighbors[leftIndex].add(rightIndex)
                    neighbors[rightIndex].add(leftIndex)
                }
            }
        }
        return MaskAffinityGraph(neighbors.map { it.toList() })
    }

    private fun maskBitsetForPlane(plane: FloatArray?): MaskBitset {
        if (plane == null || plane.isEmpty()) return MaskBitset(LongArray(0), 0)
        val words = LongArray((plane.size + 63) / 64)
        var onCount = 0
        for (index in plane.indices) {
            val value = plane[index]
            if (value.isFinite() && value > RAW_MASK_AFFINITY_BIT_THRESHOLD) {
                words[index shr 6] = words[index shr 6] or (1L shl (index and 63))
                onCount++
            }
        }
        return MaskBitset(words, onCount)
    }

    private fun bitsetIntersectionCount(left: LongArray, right: LongArray): Int {
        val count = min(left.size, right.size)
        var total = 0
        for (index in 0 until count) {
            total += java.lang.Long.bitCount(left[index] and right[index])
        }
        return total
    }

    private fun buildDisplayClustersFromAffinityGraph(
        orderedDisplayRawIndices: List<Int>,
        affinityGraph: MaskAffinityGraph,
    ): List<List<Int>> {
        if (orderedDisplayRawIndices.isEmpty()) return emptyList()
        if (affinityGraph.nodeCount <= 0) return orderedDisplayRawIndices.indices.map { listOf(it) }

        val rawToDisplay = mutableMapOf<Int, Int>()
        for ((displayIndex, rawIndex) in orderedDisplayRawIndices.withIndex()) {
            if (rawIndex >= 0) rawToDisplay[rawIndex] = displayIndex
        }

        val visited = BooleanArray(orderedDisplayRawIndices.size)
        val clusters = mutableListOf<List<Int>>()
        for (displayIndex in orderedDisplayRawIndices.indices) {
            if (visited[displayIndex]) continue
            val rawIndex = orderedDisplayRawIndices[displayIndex]
            val rawGroup = if (rawIndex >= 0) {
                affinityGraph.transitiveGroup(listOf(rawIndex)).ifEmpty { listOf(rawIndex) }
            } else {
                emptyList()
            }
            val displayGroup = rawGroup.mapNotNull { rawToDisplay[it] }.distinct().sorted()
                .ifEmpty { listOf(displayIndex) }
            for (member in displayGroup) {
                if (member in visited.indices) visited[member] = true
            }
            clusters += displayGroup
        }
        return clusters
    }

    private fun selectedSeedIndicesForPins(
        detections: List<Detection>,
        pins: List<DetectionResult>,
        affinityGraph: MaskAffinityGraph,
    ): List<Int> {
        if (detections.isEmpty() || pins.isEmpty()) return emptyList()
        val seedIndices = mutableListOf<Int>()
        val coveredClusterMembers = mutableSetOf<Int>()
        val usedDetectionIndices = mutableSetOf<Int>()
        val pinMatchIouThreshold = 0.45f

        for (pin in pins) {
            var bestIndex = -1
            var bestIou = 0f
            for ((index, detection) in detections.withIndex()) {
                if (detection.classId != pin.classId) continue
                val iou = calculateIoU(detection, pin)
                if (iou > bestIou) {
                    bestIou = iou
                    bestIndex = index
                }
            }
            if (bestIndex < 0 || bestIou < pinMatchIouThreshold || bestIndex in usedDetectionIndices) continue

            usedDetectionIndices += bestIndex
            val clusterMembers = affinityGraph.transitiveGroup(listOf(bestIndex)).ifEmpty { listOf(bestIndex) }
            if (clusterMembers.any { it in coveredClusterMembers }) {
                coveredClusterMembers += clusterMembers
                continue
            }
            seedIndices += bestIndex
            coveredClusterMembers += clusterMembers
        }
        return seedIndices
    }

    private fun primaryDetectionScore(
        detection: Detection,
        frameWidth: Float,
        frameHeight: Float,
        maxCandidateArea: Float,
        minimumConfidence: Float,
    ): Float {
        val width = detection.w
        val height = detection.h
        val confidence = detection.confidence
        if (!detection.x.isFinite() || !detection.y.isFinite() || !width.isFinite() || !height.isFinite() || !confidence.isFinite()) {
            return -1f
        }
        if (frameWidth <= 1f || frameHeight <= 1f || width <= 0f || height <= 0f) {
            return -1f
        }

        val area = width * height
        val frameArea = frameWidth * frameHeight
        val frameAreaFraction = (area / frameArea.coerceAtLeast(1f)).coerceIn(0f, 1f)
        if (frameAreaFraction > OVERSIZED_BOX_AREA_FRACTION && confidence < LOW_CONFIDENCE_OVERSIZED_THRESHOLD) {
            return -1f
        }
        val areaNormalized = (area / maxCandidateArea.coerceAtLeast(1f)).coerceIn(0f, 1f)
        val minimumAreaNormalized = 0.005f
        if (confidence < minimumConfidence || areaNormalized < minimumAreaNormalized) {
            return -1f
        }

        val centerX = frameWidth * 0.5f
        val centerY = frameHeight * 0.5f
        val dx = (detection.x - centerX) / centerX.coerceAtLeast(1f)
        val dy = (detection.y - centerY) / centerY.coerceAtLeast(1f)
        val centerDistance = kotlin.math.sqrt(dx * dx + dy * dy).coerceIn(0f, 1.4142136f)
        val centerScore = (1f - centerDistance / 1.4142136f).coerceIn(0f, 1f)

        return 0.70f * centerScore + 0.20f * confidence + 0.10f * areaNormalized
    }

    private fun pickPrimaryOnnxDetection(
        detections: List<Detection>,
        frameWidth: Float,
        frameHeight: Float,
        minimumConfidence: Float,
    ): Detection? {
        if (detections.isEmpty()) return null

        val viable = detections
            .asSequence()
            .filter { detection ->
                detection.confidence.isFinite() &&
                    detection.confidence >= minimumConfidence &&
                    detection.w.isFinite() &&
                    detection.h.isFinite() &&
                    detection.w > 0f &&
                    detection.h > 0f
            }
            .toList()
        if (viable.isEmpty()) return detections.maxByOrNull { it.confidence }

        val maxCandidateArea = viable.maxOf { it.w * it.h }.coerceAtLeast(1f)
        val selected = viable.maxWithOrNull { lhs, rhs ->
            val lhsScore = primaryDetectionScore(lhs, frameWidth, frameHeight, maxCandidateArea, minimumConfidence)
            val rhsScore = primaryDetectionScore(rhs, frameWidth, frameHeight, maxCandidateArea, minimumConfidence)
            if (lhsScore == rhsScore) {
                val confidenceDelta = lhs.confidence.compareTo(rhs.confidence)
                if (confidenceDelta != 0) {
                    confidenceDelta
                } else {
                    (lhs.w * lhs.h).compareTo(rhs.w * rhs.h)
                }
            } else {
                lhsScore.compareTo(rhsScore)
            }
        } ?: detections.maxByOrNull { it.confidence }
        if (selected != null) {
            val summary = viable.take(4).joinToString(" | ") { detection ->
                val areaFraction = (detection.w * detection.h / (frameWidth * frameHeight).coerceAtLeast(1f)).coerceIn(0f, 1f)
                val centerDistance = kotlin.math.sqrt(
                    ((detection.x - frameWidth * 0.5f) / (frameWidth * 0.5f).coerceAtLeast(1f)).let { it * it } +
                        ((detection.y - frameHeight * 0.5f) / (frameHeight * 0.5f).coerceAtLeast(1f)).let { it * it },
                )
                "${labelForClassId(detection.classId)}:${String.format("%.2f", detection.confidence)} " +
                    "box=${String.format("%.2f", areaFraction)} center=${String.format("%.2f", centerDistance)}"
            }
            LogUtil.i(
                TAG,
                "Primary selection: selected=${labelForClassId(selected.classId)}:${String.format("%.2f", selected.confidence)} " +
                    "bbox=${String.format("%.2f", (selected.w * selected.h / (frameWidth * frameHeight).coerceAtLeast(1f)).coerceIn(0f, 1f))} " +
                    "pool=[$summary]",
            )
        }
        return selected
    }

    private fun pickSupportingTableForMonitorScene(
        primaryDetection: Detection,
        detections: List<Detection>,
    ): Detection? {
        if (!includeSupportingTableForMonitorScene) return null
        if (!monitorLikeClassIds.contains(primaryDetection.classId)) return null

        val primaryLeft = primaryDetection.x - primaryDetection.w / 2f
        val primaryRight = primaryDetection.x + primaryDetection.w / 2f
        val primaryBottom = primaryDetection.y + primaryDetection.h / 2f
        val primaryArea = max(1e-3f, primaryDetection.w * primaryDetection.h)

        var bestDetection: Detection? = null
        var bestScore = -1f

        for (detection in detections) {
            if (detection === primaryDetection) continue
            if (!supportingTableClassIds.contains(detection.classId)) continue

            val candidateLeft = detection.x - detection.w / 2f
            val candidateRight = detection.x + detection.w / 2f
            val candidateTop = detection.y - detection.h / 2f
            val overlapWidth = max(0f, min(primaryRight, candidateRight) - max(primaryLeft, candidateLeft))
            val horizontalOverlapRatio = overlapWidth / max(1e-3f, min(primaryDetection.w, detection.w))
            if (horizontalOverlapRatio < 0.35f) continue

            if (detection.y <= primaryDetection.y) continue

            val verticalGap = candidateTop - primaryBottom
            if (verticalGap < -primaryDetection.h * 0.20f || verticalGap > primaryDetection.h * 0.60f) continue

            val widthRatio = detection.w / max(1e-3f, primaryDetection.w)
            if (widthRatio < 0.75f || widthRatio > 5.0f) continue

            val areaRatio = (detection.w * detection.h) / primaryArea
            if (areaRatio < 0.50f || areaRatio > 12.0f) continue

            val closenessTerm = 1f - min(1f, kotlin.math.abs(verticalGap) / max(primaryDetection.h * 0.60f, 1e-3f))
            val score = detection.confidence * horizontalOverlapRatio * max(0.1f, closenessTerm)

            if (score > bestScore) {
                bestScore = score
                bestDetection = detection
            }
        }

        if (bestDetection != null) {
            LogUtil.d(
                "FurnitureFitManager",
                "Support table picked for monitor scene: class=${bestDetection.classId} conf=${bestDetection.confidence}",
            )
        }

        return bestDetection
    }

    private fun calculateIoUForMaskSelection(first: Detection, second: Detection): Float {
        val firstX1 = first.x - first.w / 2f
        val firstY1 = first.y - first.h / 2f
        val firstX2 = first.x + first.w / 2f
        val firstY2 = first.y + first.h / 2f

        val secondX1 = second.x - second.w / 2f
        val secondY1 = second.y - second.h / 2f
        val secondX2 = second.x + second.w / 2f
        val secondY2 = second.y + second.h / 2f

        val interX1 = max(firstX1, secondX1)
        val interY1 = max(firstY1, secondY1)
        val interX2 = min(firstX2, secondX2)
        val interY2 = min(firstY2, secondY2)
        val interW = max(0f, interX2 - interX1)
        val interH = max(0f, interY2 - interY1)
        val interArea = interW * interH
        val unionArea = first.w * first.h + second.w * second.h - interArea
        return if (unionArea > 0f) interArea / unionArea else 0f
    }

    private fun collectMaskDetections(
        primaryDetection: Detection,
        detections: List<Detection>,
    ): List<Detection> {
        val supportingTableDetection = pickSupportingTableForMonitorScene(primaryDetection, detections)
        val primaryLeft = primaryDetection.x - primaryDetection.w / 2f
        val primaryTop = primaryDetection.y - primaryDetection.h / 2f
        val primaryRight = primaryDetection.x + primaryDetection.w / 2f
        val primaryBottom = primaryDetection.y + primaryDetection.h / 2f
        val encompassTolerance = 2f
        // Fusion-only confidence floor (redundant with parse threshold); kept for reference.
        // val minimumCandidateConfidence = 0.1f
        val bboxDuplicateThreshold = 0.7f

        val bboxKept = mutableListOf<Detection>()
        for (detection in detections) {
            if (detection == primaryDetection) continue

            val candidateLeft = detection.x - detection.w / 2f
            val candidateTop = detection.y - detection.h / 2f
            val candidateRight = detection.x + detection.w / 2f
            val candidateBottom = detection.y + detection.h / 2f

            val encompassesPrimary =
                candidateLeft <= primaryLeft + encompassTolerance &&
                    candidateTop <= primaryTop + encompassTolerance &&
                    candidateRight >= primaryRight - encompassTolerance &&
                    candidateBottom >= primaryBottom - encompassTolerance
            if (encompassesPrimary) continue

            val intersectsPrimary =
                !(candidateRight < primaryLeft || candidateLeft > primaryRight || candidateBottom < primaryTop || candidateTop > primaryBottom)
            if (!intersectsPrimary) continue

            val tooLarge =
                detection.w > primaryDetection.w * 1.5f &&
                    detection.h > primaryDetection.h * 1.5f
            if (tooLarge) continue

            if (calculateIoUForMaskSelection(detection, primaryDetection) > bboxDuplicateThreshold) continue

            var shouldSkip = false
            var replaceIndex = -1
            for ((index, keptDetection) in bboxKept.withIndex()) {
                val iou = calculateIoUForMaskSelection(detection, keptDetection)
                if (iou > bboxDuplicateThreshold) {
                    if (detection.confidence > keptDetection.confidence) {
                        replaceIndex = index
                    } else {
                        shouldSkip = true
                    }
                    break
                }
            }
            if (shouldSkip) continue
            if (replaceIndex >= 0) {
                bboxKept[replaceIndex] = detection
            } else {
                bboxKept += detection
            }
        }

        val maskDetections = mutableListOf(primaryDetection)
        maskDetections += bboxKept
        if (supportingTableDetection != null && !maskDetections.contains(supportingTableDetection)) {
            maskDetections += supportingTableDetection
        }
        return maskDetections
    }

    private fun expandedPrimaryForMaskBuild(
        primaryDetection: Detection,
        frameWidth: Float,
        frameHeight: Float,
    ): Detection {
        val maxHalfW = min(primaryDetection.x, frameWidth - primaryDetection.x)
        val maxHalfH = min(primaryDetection.y, frameHeight - primaryDetection.y)
        val capW = 2f * max(maxHalfW, 1f)
        val capH = 2f * max(maxHalfH, 1f)
        val expandedW = min(primaryDetection.w * (1f + 2f * bboxExpandMargin), capW)
        val expandedH = min(primaryDetection.h * (1f + 2f * bboxExpandMargin), capH)
        return primaryDetection.copy(
            w = expandedW,
            h = expandedH,
        )
    }

    private fun clipProtoMaskOutsideRect(
        mask: FloatArray,
        protoW: Int,
        protoH: Int,
        clipX0: Int,
        clipY0: Int,
        clipX1: Int,
        clipY1: Int,
    ) {
        if (protoW <= 0 || protoH <= 0 || mask.size != protoW * protoH) return
        val x0 = clipX0.coerceIn(0, protoW)
        val y0 = clipY0.coerceIn(0, protoH)
        val x1 = clipX1.coerceIn(0, protoW)
        val y1 = clipY1.coerceIn(0, protoH)
        if (x0 >= x1 || y0 >= y1) {
            mask.fill(0f)
            return
        }

        for (y in 0 until protoH) {
            val rowBase = y * protoW
            if (y < y0 || y >= y1) {
                for (x in 0 until protoW) {
                    mask[rowBase + x] = 0f
                }
                continue
            }
            for (x in 0 until x0) {
                mask[rowBase + x] = 0f
            }
            for (x in x1 until protoW) {
                mask[rowBase + x] = 0f
            }
        }
    }

    private fun extractFloatArray(value: Any?): FloatArray {
        return when (value) {
            is FloatArray -> value
            is Array<*> -> flattenArrayToFloat(value)
            is java.nio.FloatBuffer -> {
                val arr = FloatArray(value.remaining())
                value.get(arr)
                arr
            }
            else -> {
                LogUtil.w("FurnitureFitManager", "Unknown output type: ${value?.javaClass}")
                FloatArray(0)
            }
        }
    }

    private fun calculateIoU(det1: Detection, det2: Detection): Float {
        val x1Min = det1.x - det1.w / 2
        val y1Min = det1.y - det1.h / 2
        val x1Max = det1.x + det1.w / 2
        val y1Max = det1.y + det1.h / 2

        val x2Min = det2.x - det2.w / 2
        val y2Min = det2.y - det2.h / 2
        val x2Max = det2.x + det2.w / 2
        val y2Max = det2.y + det2.h / 2

        val interXMin = maxOf(x1Min, x2Min)
        val interYMin = maxOf(y1Min, y2Min)
        val interXMax = minOf(x1Max, x2Max)
        val interYMax = minOf(y1Max, y2Max)

        val interWidth = maxOf(0f, interXMax - interXMin)
        val interHeight = maxOf(0f, interYMax - interYMin)
        val interArea = interWidth * interHeight

        val area1 = det1.w * det1.h
        val area2 = det2.w * det2.h
        val unionArea = area1 + area2 - interArea

        return if (unionArea > 0) interArea / unionArea else 0f
    }

    private fun calculateIoU(det: Detection, pin: DetectionResult): Float {
        val x1Min = det.x - det.w / 2
        val y1Min = det.y - det.h / 2
        val x1Max = det.x + det.w / 2
        val y1Max = det.y + det.h / 2

        val x2Min = pin.x - pin.w / 2
        val y2Min = pin.y - pin.h / 2
        val x2Max = pin.x + pin.w / 2
        val y2Max = pin.y + pin.h / 2

        val interXMin = maxOf(x1Min, x2Min)
        val interYMin = maxOf(y1Min, y2Min)
        val interXMax = minOf(x1Max, x2Max)
        val interYMax = minOf(y1Max, y2Max)

        val interWidth = maxOf(0f, interXMax - interXMin)
        val interHeight = maxOf(0f, interYMax - interYMin)
        val interArea = interWidth * interHeight

        val area1 = det.w * det.h
        val area2 = pin.w * pin.h
        val unionArea = area1 + area2 - interArea

        return if (unionArea > 0) interArea / unionArea else 0f
    }

    private fun applyMorphClose3x3ToFloatMask(
        mask: FloatArray,
        width: Int,
        height: Int,
        threshold: Float,
    ) {
        if (width <= 0 || height <= 0 || mask.size != width * height) return

        val binaryMask = BooleanArray(mask.size) { idx -> mask[idx] > threshold }
        val dilatedMask = dilate3x3(binaryMask, width, height)
        val closedMask = erode3x3(dilatedMask, width, height)

        for (index in mask.indices) {
            mask[index] = if (closedMask[index]) 1f else 0f
        }
    }

    private fun applyMorphClose3x3ToBitmapMask(frame: Bitmap, mask: Bitmap): Bitmap {
        val width = mask.width
        val height = mask.height
        if (width <= 0 || height <= 0) return mask

        val framePixels = IntArray(width * height)
        val maskPixels = IntArray(width * height)
        frame.getPixels(framePixels, 0, width, 0, 0, width, height)
        mask.getPixels(maskPixels, 0, width, 0, 0, width, height)

        val binaryMask = BooleanArray(maskPixels.size) { idx ->
            ((maskPixels[idx] ushr 24) and 0xFF) > 0
        }
        val dilatedMask = dilate3x3(binaryMask, width, height)
        val closedMask = erode3x3(dilatedMask, width, height)

        val outputPixels = IntArray(width * height)
        for (index in outputPixels.indices) {
            outputPixels[index] = if (closedMask[index]) {
                framePixels[index]
            } else {
                0x00000000
            }
        }

        return Bitmap.createBitmap(outputPixels, width, height, Config.ARGB_8888)
    }

    private fun dilate3x3(mask: BooleanArray, width: Int, height: Int): BooleanArray {
        val outputMask = BooleanArray(mask.size)
        for (y in 0 until height) {
            for (x in 0 until width) {
                var isForeground = false
                val yStart = maxOf(0, y - 1)
                val yEnd = minOf(height - 1, y + 1)
                val xStart = maxOf(0, x - 1)
                val xEnd = minOf(width - 1, x + 1)
                for (kernelY in yStart..yEnd) {
                    val rowOffset = kernelY * width
                    for (kernelX in xStart..xEnd) {
                        if (mask[rowOffset + kernelX]) {
                            isForeground = true
                            break
                        }
                    }
                    if (isForeground) break
                }
                outputMask[y * width + x] = isForeground
            }
        }
        return outputMask
    }

    private fun erode3x3(mask: BooleanArray, width: Int, height: Int): BooleanArray {
        val outputMask = BooleanArray(mask.size)
        for (y in 0 until height) {
            for (x in 0 until width) {
                var isForeground = true
                val yStart = maxOf(0, y - 1)
                val yEnd = minOf(height - 1, y + 1)
                val xStart = maxOf(0, x - 1)
                val xEnd = minOf(width - 1, x + 1)
                for (kernelY in yStart..yEnd) {
                    val rowOffset = kernelY * width
                    for (kernelX in xStart..xEnd) {
                        if (!mask[rowOffset + kernelX]) {
                            isForeground = false
                            break
                        }
                    }
                    if (!isForeground) break
                }
                outputMask[y * width + x] = isForeground
            }
        }
        return outputMask
    }

    // Inner class for detection data
    private data class Detection(
        val anchorIdx: Int,
        val x: Float, val y: Float, val w: Float, val h: Float,
        val confidence: Float,
        val classId: Int,
        val coeffs: FloatArray,
        val priorX: Float = x,
        val priorY: Float = y,
        val levelStride: Float = 0f,
    )

    private fun Detection.toSwiftParityBox(): RTMDetSwiftParity.Box = RTMDetSwiftParity.Box(
        x = x,
        y = y,
        width = w,
        height = h,
        confidence = confidence,
        classId = classId,
    )

    private fun parseDetectionsForCurrentModel(
        outputs: DetectionProtoOutputs,
        detValue: Any?,
        confidenceThreshold: Float,
    ): List<Detection> {
        val dim1 = outputs.detectionShape[1].toInt()
        val dim2 = outputs.detectionShape[2].toInt()
        val numMaskCoeffs = outputs.protoShape[1].toInt()
        val detections = mutableListOf<Detection>()

        val detFlat = extractFloatArray(detValue)
        if (detFlat.isEmpty()) {
            LogUtil.w(TAG, "Detection tensor was empty after extraction")
            return emptyList()
        }

        val isEndToEndFormat = dim2 < 100
        if (isEndToEndFormat) {
            val numDetections = dim1
            val featuresPerDetection = dim2
            LogUtil.i(
                TAG,
                "Segmentation parser layout: end-to-end [1,$numDetections,$featuresPerDetection] with $numMaskCoeffs mask coeffs"
            )

            for (detIndex in 0 until numDetections) {
                val base = detIndex * featuresPerDetection
                if (base + 6 + numMaskCoeffs > detFlat.size) break

                val x1 = detFlat[base + 0]
                val y1 = detFlat[base + 1]
                val x2 = detFlat[base + 2]
                val y2 = detFlat[base + 3]
                val confidence = detFlat[base + 4]
                val classIdxFloat = detFlat[base + 5]
                if (!x1.isFinite() || !y1.isFinite() || !x2.isFinite() || !y2.isFinite() ||
                    !confidence.isFinite() || !classIdxFloat.isFinite()
                ) {
                    continue
                }

                val width = x2 - x1
                val height = y2 - y1
                if (confidence < confidenceThreshold || confidence > 1f || width <= 0f || height <= 0f) {
                    continue
                }

                val classId = classIdxFloat.toInt()
                if (classId < 0 || classId in ignoredClassIds) continue

                val coeffs = FloatArray(numMaskCoeffs)
                var validCoefficients = true
                for (coeffIndex in 0 until numMaskCoeffs) {
                    val value = detFlat[base + 6 + coeffIndex]
                    if (!value.isFinite()) {
                        validCoefficients = false
                        break
                    }
                    coeffs[coeffIndex] = value
                }
                if (!validCoefficients) continue

                detections.add(
                    Detection(
                        anchorIdx = detIndex,
                        x = (x1 + x2) * 0.5f,
                        y = (y1 + y2) * 0.5f,
                        w = width,
                        h = height,
                        confidence = confidence,
                        classId = classId,
                        coeffs = coeffs,
                    )
                )
            }
            return detections
        }

        val det3d = detValue as? Array<Array<FloatArray>>
        val numFeatures = dim1
        val numAnchors = dim2
        val numClasses = numFeatures - 4 - numMaskCoeffs
        if (numFeatures < 4 + numMaskCoeffs + 1 || numAnchors <= 0 || numClasses <= 0) {
            LogUtil.e(
                TAG,
                "Invalid one-to-many detection layout: features=$numFeatures anchors=$numAnchors maskCoeffs=$numMaskCoeffs classes=$numClasses"
            )
            return emptyList()
        }

        LogUtil.i(
            TAG,
            "Segmentation parser layout: one-to-many [1,$numFeatures,$numAnchors] with $numClasses classes and $numMaskCoeffs mask coeffs"
        )
        val getDetValue: (Int, Int) -> Float = if (det3d != null) {
            { feature, anchor -> det3d[0][feature][anchor] }
        } else {
            { feature, anchor -> detFlat[feature * numAnchors + anchor] }
        }
        val classStartIdx = 4
        val maskCoeffStartIdx = 4 + numClasses

        for (anchor in 0 until numAnchors) {
            var maxScore = Float.MIN_VALUE
            var bestClass = -1
            for (classIndex in 0 until numClasses) {
                val score = getDetValue(classStartIdx + classIndex, anchor)
                if (score > maxScore) {
                    maxScore = score
                    bestClass = classIndex
                }
            }
            if (maxScore < confidenceThreshold || bestClass in ignoredClassIds) continue

            val x = getDetValue(0, anchor)
            val y = getDetValue(1, anchor)
            val width = getDetValue(2, anchor)
            val height = getDetValue(3, anchor)
            if (!x.isFinite() || !y.isFinite() || !width.isFinite() || !height.isFinite() || width <= 0f || height <= 0f) {
                continue
            }

            val coeffs = FloatArray(numMaskCoeffs)
            var validCoefficients = true
            for (coeffIndex in 0 until numMaskCoeffs) {
                val value = getDetValue(maskCoeffStartIdx + coeffIndex, anchor)
                if (!value.isFinite()) {
                    validCoefficients = false
                    break
                }
                coeffs[coeffIndex] = value
            }
            if (!validCoefficients) continue

            detections.add(
                Detection(
                    anchorIdx = anchor,
                    x = x,
                    y = y,
                    w = width,
                    h = height,
                    confidence = maxScore,
                    classId = bestClass,
                    coeffs = coeffs,
                )
            )
        }
        return detections
    }

    fun close() {
        // RTMDet runtimes are shared for the process. Screen-level managers are lightweight
        // handles and must not close a singleton interpreter/session while another screen may use it.
        abandonAcceptedInference(rotateSharedLane = false)
        liteRtBackend = null
        ortBackend = null
        ortSession = null
        ortEnv = null
        loadedOnnxAssetName = null
    }

    /**
     * Abandons all work accepted by the current Furniture Fit session.
     *
     * Call this at the same lifecycle boundary where Swift rotates
     * `coreMLInferenceQueue`. A queued frame is rejected by its generation check; an ONNX call
     * already inside `OrtSession.run` is terminated through its private [OrtSession.RunOptions].
     */
    fun rotateInferenceQueueForNewSession() {
        abandonAcceptedInference(rotateSharedLane = true)
    }

    private fun abandonAcceptedInference(rotateSharedLane: Boolean) {
        val nextGeneration = inferenceGeneration.incrementAndGet()
        val activeOptions = activeRunOptions.get()
        if (activeOptions != null) {
            try {
                activeOptions.setTerminate(true)
            } catch (e: Exception) {
                LogUtil.w(TAG, "Could not terminate abandoned ONNX inference: ${e.message}")
            }
        }
        val laneGeneration = if (rotateSharedLane) rotateSharedInferenceLane() else null
        LogUtil.d(
            TAG,
            "Abandoned Furniture Fit inference generation=$nextGeneration" +
                (laneGeneration?.let { "; rotated lane=$it" } ?: ""),
        )
    }

    private fun flattenArrayToFloat(arr: Array<*>): FloatArray {
        val list = ArrayList<Float>()
        fun rec(a: Any?) {
            when (a) {
                is Float -> list.add(a)
                is Double -> list.add(a.toFloat())
                is Int -> list.add(a.toFloat())
                is FloatArray -> for (v in a) list.add(v)
                is DoubleArray -> for (v in a) list.add(v.toFloat())
                is Array<*> -> for (e in a) rec(e)
                else -> {}
            }
        }
        rec(arr)
        return list.toFloatArray()
    }

    private fun loadClassNames(): Map<Int, String> {
        return try {
            context.assets.open("classes.json").bufferedReader().use { reader ->
                val json = JSONObject(reader.readText())
                buildMap {
                    json.keys().forEach { key ->
                        val id = key.toIntOrNull() ?: return@forEach
                        val label = json.optString(key).trim()
                        if (label.isNotEmpty()) put(id, label)
                    }
                }
            }
        } catch (e: Exception) {
            LogUtil.w("FurnitureFitManager", "loadClassNames failed: ${e.message}")
            emptyMap()
        }
    }

    private fun loadIgnoredClassIds(): Set<Int> {
        return try {
            context.assets.open("blacklist.json").bufferedReader().use { reader ->
                val json = JSONObject(reader.readText())
                buildSet {
                    json.keys().forEach { key ->
                        key.toIntOrNull()?.let { add(it) }
                    }
                }
            }
        } catch (e: Exception) {
            LogUtil.w("FurnitureFitManager", "loadIgnoredClassIds failed: ${e.message}")
            emptySet()
        }
    }

    private fun labelForClassId(classId: Int): String {
        // Current segmentation export uses 80 COCO classes (4 box + 80 class + 32 mask coeffs).
        // classes.json is a larger Furnit taxonomy, so using it for class IDs 0..79 maps chairs to
        // unrelated labels like "almond".
        if (classId in COCO_CLASS_NAMES.indices) {
            return COCO_CLASS_NAMES[classId]
        }
        return classNames[classId]?.takeIf { it.isNotBlank() } ?: "class_$classId"
    }

    private fun sigmoid(value: Float): Float {
        return if (value >= 0f) {
            val z = exp(-value)
            (1.0 / (1.0 + z)).toFloat()
        } else {
            val z = exp(value)
            (z / (1.0 + z)).toFloat()
        }
    }
}

private val COCO_CLASS_NAMES = arrayOf(
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
)
