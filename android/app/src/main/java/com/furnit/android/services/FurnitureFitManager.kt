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
import com.furnit.android.DetectionResult
import com.furnit.android.ar.ArSupportChecker
import com.furnit.android.utils.LogUtil
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicLong
import org.json.JSONObject
import kotlin.math.exp
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min

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
     * Furniture Fit stops so an abandoned prediction cannot hold up the next session. Android
     * invalidates queued work and rotates the serial inference lane to provide the same stop ->
     * immediate restart guarantee without sharing scratch buffers across calls.
     */
    private val inferenceGeneration = AtomicLong(0L)
    private val classNames: Map<Int, String> by lazy(LazyThreadSafetyMode.NONE) { loadClassNames() }
    private val ignoredClassIds: Set<Int> by lazy(LazyThreadSafetyMode.NONE) { loadIgnoredClassIds() }

    companion object {
        private const val TAG = "FurnitureFitManager"
        private const val RTMDET_TFLITE_MODEL_ASSET = RoomGenerationAssets.RTMDET_INS_M_RAW_FP16_TFLITE
        private const val RTMDET_CONFIDENCE_THRESHOLD = RTMDetSwiftParity.CONFIDENCE_THRESHOLD
        private const val DEFAULT_NMS_IOU_THRESHOLD = RTMDetSwiftParity.NMS_IOU_THRESHOLD
        private const val DEFAULT_MAX_DETECTIONS = RTMDetSwiftParity.MAX_DETECTION_COUNT
        private const val RAW_MASK_AFFINITY_THRESHOLD = RTMDetSwiftParity.MASK_AFFINITY_THRESHOLD
        private const val RAW_MASK_AFFINITY_BIT_THRESHOLD = RTMDetSwiftParity.MASK_AFFINITY_BIT_THRESHOLD
        private const val RTMDET_INPUT_SIZE = RTMDetSwiftParity.MODEL_SIDE
        private const val RTMDET_MASK_SIDE = RTMDetSwiftParity.MASK_SIDE

        private val sharedBackendLock = Any()
        @Volatile private var sharedLiteRtBackend: RTMDetLiteRtBackend? = null
        private val sharedResourceGeneration = java.util.concurrent.atomic.AtomicLong(0L)

        /** Reusable memory belongs to one serial inference lane and is never shared across lanes. */
        private class InferenceWorkspace {
            var preparedBitmap: Bitmap? = null
            var preparedPixels = IntArray(0)
            var framePixels = IntArray(0)
            var outputPixels = IntArray(0)
            val preprocessPaint = Paint(Paint.FILTER_BITMAP_FLAG)

            fun close() {
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
         * Release is serialized after any accepted inference so the LiteRT interpreter is never
         * closed underneath an active call. A new viewer initialization cancels a stale release.
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
                    val liteRtBackend = sharedLiteRtBackend
                    sharedLiteRtBackend = null
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

    }

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
        LogUtil.i(TAG, "Using ${backend.executionProvider} RTMDet backend")
        return true
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
            maskFeatIsNhwc = true,
        )
        return handleRtmdetRawResults(
            frame = frame,
            inputW = inputW,
            inputH = inputH,
            includeMask = includeMask,
            selectedClassIds = selectedClassIds,
            pinnedDetections = pinnedDetections,
            requireClusters = requireClusters,
            raw = raw,
            totalStartNanos = totalStartNanos,
            preprocessMillis = preprocessMillis,
            inferenceMillis = inferenceAndOutputMillis,
        )
    }

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
        /** True when [maskFeat] uses LiteRT's native `pixel * 8 + channel` layout. */
        val maskFeatIsNhwc: Boolean,
    )

    private fun elapsedMillis(startNanos: Long): Long =
        (System.nanoTime() - startNanos) / 1_000_000L

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

    private fun handleRtmdetRawResults(
        frame: Bitmap,
        inputW: Int,
        inputH: Int,
        includeMask: Boolean,
        selectedClassIds: Set<Int>,
        pinnedDetections: List<DetectionResult>?,
        requireClusters: Boolean,
        raw: RtmdetRawOutputs,
        totalStartNanos: Long,
        preprocessMillis: Long = 0,
        inferenceMillis: Long = 0,
    ): SegmentationResult? {
        val parseStartNanos = System.nanoTime()
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
        val rawMaskPlanes = keepDets.map { buildRtmdetRawMaskPlane(it, raw.maskFeat, raw.maskFeatIsNhwc) }
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
        // readable at a glance: preprocess → LiteRT inference → parse+nms → maskBuild (cutout).
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
        maskFeatIsNhwc: Boolean,
    ): FloatArray? = RTMDetMaskHead.buildPlane(
        coeffs = detection.coeffs,
        maskFeat = maskFeat,
        maskFeatIsNhwc = maskFeatIsNhwc,
        maskSide = RTMDET_MASK_SIDE,
        inputSize = RTMDET_INPUT_SIZE,
        priorX = detection.priorX,
        priorY = detection.priorY,
        levelStride = detection.levelStride,
    )

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

    fun close() {
        // The RTMDet interpreter is shared for the process. Screen-level managers are lightweight
        // handles and must not close it while another screen may use it.
        abandonAcceptedInference(rotateSharedLane = false)
        liteRtBackend = null
    }

    /**
     * Abandons all work accepted by the current Furniture Fit session.
     *
     * Call this at the same lifecycle boundary where Swift rotates `coreMLInferenceQueue`. A queued
     * frame is rejected by its generation check and subsequent work uses a fresh serial lane.
     */
    fun rotateInferenceQueueForNewSession() {
        abandonAcceptedInference(rotateSharedLane = true)
    }

    private fun abandonAcceptedInference(rotateSharedLane: Boolean) {
        val nextGeneration = inferenceGeneration.incrementAndGet()
        val laneGeneration = if (rotateSharedLane) rotateSharedInferenceLane() else null
        LogUtil.d(
            TAG,
            "Abandoned Furniture Fit inference generation=$nextGeneration" +
                (laneGeneration?.let { "; rotated lane=$it" } ?: ""),
        )
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
