package com.furnit.android.ar

import android.app.Activity
import android.graphics.Bitmap
import android.opengl.GLES20
import android.opengl.GLSurfaceView
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import com.furnit.android.utils.LogUtil
import com.google.ar.core.ArCoreApk
import com.google.ar.core.Config
import com.google.ar.core.Frame
import com.google.ar.core.Session
import com.google.ar.core.exceptions.CameraNotAvailableException
import com.google.ar.core.exceptions.UnavailableException
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executor
import java.util.concurrent.TimeUnit
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10

private const val TAG = "FurnitureFitAr"

data class ArPhotoCaptureResult(
    val bitmap: Bitmap,
    val metricAnchors: List<MetricAnchor>,
)

private data class PendingPhotoCaptureRequest(
    val callback: (ArPhotoCaptureResult?) -> Unit,
    val startedAtMs: Long,
    var attempts: Int = 0,
    var bestBitmap: Bitmap? = null,
    var bestAnchors: List<MetricAnchor> = emptyList(),
)

/**
 * ARCore path for FurnitureFit: [GLSurfaceView] + [Session], CPU bitmaps for segmentation, and smoothed
 * overlay scale from depth/plane distance + pinhole dimensions ([FurnitureFitArMetrics]).
 *
 * Physical width and height use **pinhole × metric distance only**. We do **not** use
 * `roomHeight × (bbox_h / image_h)` for sizing — that shrinks as the camera moves away because
 * the bbox gets smaller in pixels even though the real object does not.
 */
class FurnitureFitArCameraController(
    private val activity: Activity,
    private val inferenceExecutor: Executor,
) : GLSurfaceView.Renderer {

    val glSurfaceView: GLSurfaceView = GLSurfaceView(activity).apply {
        preserveEGLContextOnPause = true
        setEGLContextClientVersion(2)
        setEGLConfigChooser(8, 8, 8, 8, 16, 0)
        setRenderer(this@FurnitureFitArCameraController)
        renderMode = GLSurfaceView.RENDERMODE_CONTINUOUSLY
    }

    private val displayRotationHelper = DisplayRotationHelper(activity)
    private val backgroundRenderer = ArBackgroundRenderer()

    @Volatile
    private var session: Session? = null
    private var installRequested = false

    private val sessionGlLock = Any()

    /** Invoked on [inferenceExecutor] with a decoded camera bitmap. */
    var onBitmapFrame: ((Bitmap) -> Unit)? = null

    /** If false, the GL thread skips posting new frames (e.g. while segmentation is running). */
    var shouldPostBitmapFrame: () -> Boolean = { true }

    @Volatile
    private var pendingPhotoCaptureRequest: PendingPhotoCaptureRequest? = null
    private val pendingInferenceBitmapLock = Any()
    private var pendingInferenceBitmap: Bitmap? = null

    private val bboxLock = Any()
    private var bboxCenterImageX = 0f
    private var bboxCenterImageY = 0f
    private var bboxWidthImagePx = 0f
    private var bboxHeightImagePx = 0f
    private var bboxLabel = ""
    private var bboxSelectionKey = ""
    private var bboxHintValid = false

    private val scaleLock = Any()
    private var smoothedArOverlayScale = 1f
    private var arOverlayScaleValid = false
    private var lastEstimatedWidthMeters = Float.NaN
    private var lastEstimatedHeightMeters = Float.NaN
    private var lastMetricDistanceMeters = Float.NaN
    private var lastMetricDistanceSource: String? = null
    private var lastMetricDistanceDiagnostic: String? = null

    private var lastInferencePostMs = 0L
    /** Min time between frames handed to segmentation (iOS ~0.07s; slightly lower here for responsiveness). */
    var minFrameIntervalMs: Long = 55L

    @Volatile
    private var preferImmediateNextBitmap = false

    private val measurementHandler = Handler(Looper.getMainLooper())
    /** Debounced apply of AR overlay scale (mirrors iOS `assistedMeasurementDebounceSeconds` ~0.85s). */
    private val assistedMeasurementDebounceMs = 550L
    @Volatile
    private var latestRawScale: Float? = null
    @Volatile
    private var pendingEstWForApply = Float.NaN
    @Volatile
    private var pendingEstHForApply = Float.NaN
    private var snapAfterPrimaryChange = false

    /**
     * Tier 1 (provisional): latest smoothed pinhole dimensions from the GL thread, updated every
     * frame with a valid pair. Segmentation / UI can read these without waiting for debounce.
     */
    @Volatile
    private var provisionalWidthMeters: Float? = null

    @Volatile
    private var provisionalHeightMeters: Float? = null

    @Volatile
    private var provisionalRawScale: Float? = null

    @Volatile
    private var provisionalDistanceMeters: Float? = null

    @Volatile
    private var provisionalHeightTimestampMs: Long = 0L

    /** Monotonic clock when [applyPendingArOverlayScaleFromLatest] last committed dimensions. */
    @Volatile
    private var lastCommittedHeightAtMs: Long = 0L

    private val debounceApplyRunnable = Runnable {
        applyPendingArOverlayScaleFromLatest()
        onAssistedMeasurementUpdated?.invoke()
    }

    /** Throttle [LogUtil.furnitureFitAr] frame snapshots (avoid logcat flood). */
    private var lastFurnitureFitArFrameLogMs = 0L
    /** Throttle render-loop heartbeat so we can tell whether GL/AR is alive even before metrics. */
    private var lastFurnitureFitArHeartbeatLogMs = 0L
    /** Throttle logs while GL runs but segmentation has not set [bboxHintValid] yet. */
    private var lastFurnitureFitArWaitBboxLogMs = 0L
    /** Throttle logs when bbox exists but pinhole+fallback cannot produce [estH]. */
    private var lastFurnitureFitArFrameSkipLogMs = 0L
    /** Throttle [setBboxHint] lines (inference can call every frame). */
    private var lastFurnitureFitArBboxHintLogMs = 0L

    /**
     * Called on the main thread after debounced AR metric sizing is applied. Used so the generated
     * room calibration pill refreshes — the pill is not driven by the GL thread every frame.
     */
    var onAssistedMeasurementUpdated: (() -> Unit)? = null

    /**
     * Room photo lock from FurnitureFit / generated room (`"portrait"` or `"landscape"`), matching CameraX
     * target rotation so segmentation bitmap aspect matches the locked activity.
     */
    var lockedPhotoOrientation: String = "portrait"

    /**
     * Calibrated front-wall height (m) from the generated room — logged for diagnostics only (not used for
     * furniture height; room×bbox fraction is not distance-invariant).
     */
    @Volatile
    var roomHeightMetersForFallback: Float = 0f

    /**
     * EMAs of pinhole dimensions from ARCore depth/plane — hold stable when a frame has bad depth.
     * Reset when the primary detection label changes; not reset on [clearBboxHint] (hold last estimate).
     */
    private var smoothedPinholeHeightMeters = Float.NaN
    private var smoothedPinholeWidthMeters = Float.NaN

    fun setBboxHint(
        centerImageX: Float,
        centerImageY: Float,
        widthImagePx: Float,
        heightImagePx: Float,
        label: String,
        selectionKey: String = label,
    ) {
        val selectionChanged = synchronized(bboxLock) {
            val previousSelectionKey = bboxSelectionKey
            bboxCenterImageX = centerImageX
            bboxCenterImageY = centerImageY
            bboxWidthImagePx = widthImagePx
            bboxHeightImagePx = heightImagePx
            bboxLabel = label
            bboxSelectionKey = selectionKey
            bboxHintValid = centerImageX.isFinite() && centerImageY.isFinite() &&
                widthImagePx.isFinite() && widthImagePx > 1f &&
                heightImagePx.isFinite() && heightImagePx > 1f
            previousSelectionKey != selectionKey
        }
        val nowMs = SystemClock.elapsedRealtime()
        if (bboxHintValid && nowMs - lastFurnitureFitArBboxHintLogMs >= 400L) {
            lastFurnitureFitArBboxHintLogMs = nowMs
            LogUtil.furnitureFitAr(
                "platform=android phase=bbox_hint center=(${String.format("%.1f", centerImageX)},${String.format("%.1f", centerImageY)}) " +
                    "bboxWH_px=(${String.format("%.1f", widthImagePx)}x${String.format("%.1f", heightImagePx)}) " +
                    "label=${label.take(48)}",
            )
        }
        if (selectionChanged) {
            smoothedPinholeWidthMeters = Float.NaN
            smoothedPinholeHeightMeters = Float.NaN
            snapAfterPrimaryChange = true
            latestRawScale = null
            pendingEstWForApply = Float.NaN
            pendingEstHForApply = Float.NaN
            measurementHandler.removeCallbacks(debounceApplyRunnable)
            provisionalWidthMeters = null
            provisionalHeightMeters = null
            provisionalRawScale = null
            provisionalDistanceMeters = null
            provisionalHeightTimestampMs = 0L
            synchronized(scaleLock) {
                arOverlayScaleValid = false
                smoothedArOverlayScale = 1f
                lastEstimatedWidthMeters = Float.NaN
                lastEstimatedHeightMeters = Float.NaN
                lastMetricDistanceMeters = Float.NaN
                lastMetricDistanceSource = null
                lastMetricDistanceDiagnostic = null
            }
        }
    }

    /**
     * Segmentation had no bbox this frame — stop sampling depth at the old center only.
     * Keep the last estimated dimensions and overlay scale (matches iOS AR_HOLD): a single missed
     * detection or bad frame must not snap the pill to 0.00m or reset zoom to 1×.
     */
    fun clearBboxHint() {
        synchronized(bboxLock) {
            bboxHintValid = false
        }
        pendingEstWForApply = Float.NaN
        pendingEstHForApply = Float.NaN
        measurementHandler.removeCallbacks(debounceApplyRunnable)
        // Keep provisional tier until stale (matches iOS AR_HOLD); GL stops refreshing timestamp.
    }

    fun isArOverlayScaleValid(): Boolean = synchronized(scaleLock) { arOverlayScaleValid }

    fun getSmoothedArOverlayScale(): Float = synchronized(scaleLock) { smoothedArOverlayScale }

    /**
     * Last AR-estimated furniture height in meters from the pinhole model, when available.
     * Returns null when AR overlay scale is not currently valid.
     */
    fun getLastEstimatedHeightMeters(): Float? = synchronized(scaleLock) {
        lastEstimatedHeightMeters.takeIf { it.isFinite() }
    }

    fun getLastEstimatedWidthMeters(): Float? = synchronized(scaleLock) {
        lastEstimatedWidthMeters.takeIf { it.isFinite() }
    }

    /**
     * Tier 1: latest smoothed AR height (GL thread), for immediate use before debounce commits.
     * Null if never updated, out of range, or older than [provisionalStalenessMs].
     */
    fun getProvisionalHeightMeters(
        provisionalStalenessMs: Long = 2000L,
    ): Float? {
        val h = provisionalHeightMeters ?: return null
        if (!h.isFinite() || h < 0.055f || h > 4.8f) return null
        val age = SystemClock.elapsedRealtime() - provisionalHeightTimestampMs
        if (provisionalHeightTimestampMs <= 0L || age > provisionalStalenessMs) return null
        return h
    }

    fun getProvisionalWidthMeters(
        provisionalStalenessMs: Long = 2000L,
    ): Float? {
        val widthMeters = provisionalWidthMeters ?: return null
        if (!widthMeters.isFinite() || widthMeters < 0.055f || widthMeters > 6.5f) return null
        val age = SystemClock.elapsedRealtime() - provisionalHeightTimestampMs
        if (provisionalHeightTimestampMs <= 0L || age > provisionalStalenessMs) return null
        return widthMeters
    }

    /** Tier 1 distance (meters) paired with [getProvisionalHeightMeters], same staleness rule. */
    fun getProvisionalDistanceMeters(
        provisionalStalenessMs: Long = 2000L,
    ): Float? {
        val d = provisionalDistanceMeters ?: return null
        if (!d.isFinite() || d <= 0f) return null
        val age = SystemClock.elapsedRealtime() - provisionalHeightTimestampMs
        if (provisionalHeightTimestampMs <= 0L || age > provisionalStalenessMs) return null
        return d
    }

    fun getProvisionalHeightAgeMs(): Long {
        if (provisionalHeightTimestampMs <= 0L) return -1L
        return (SystemClock.elapsedRealtime() - provisionalHeightTimestampMs).coerceAtLeast(0L)
    }

    fun getCommittedHeightAgeMs(): Long {
        if (lastCommittedHeightAtMs <= 0L) return -1L
        return (SystemClock.elapsedRealtime() - lastCommittedHeightAtMs).coerceAtLeast(0L)
    }

    /**
     * Skip debounce and commit current provisional dimensions/scale immediately (main thread).
     * No-op if either dimension is missing or invalid. Optional fast-path after stable tracking.
     */
    fun forceCommitProvisional() {
        measurementHandler.post {
            val widthMeters = provisionalWidthMeters
            val h = provisionalHeightMeters
            val raw = provisionalRawScale
            if (widthMeters == null || !widthMeters.isFinite() || widthMeters < 0.055f || widthMeters > 6.5f) {
                return@post
            }
            if (h == null || !h.isFinite() || h < 0.055f || h > 4.8f) return@post
            if (raw == null || !raw.isFinite() || raw <= 0f) return@post
            measurementHandler.removeCallbacks(debounceApplyRunnable)
            pendingEstWForApply = widthMeters
            pendingEstHForApply = h
            latestRawScale = raw
            applyPendingArOverlayScaleFromLatest()
            onAssistedMeasurementUpdated?.invoke()
        }
    }

    fun getLastMetricDistanceMeters(): Float? = synchronized(scaleLock) {
        lastMetricDistanceMeters.takeIf { it.isFinite() }
    }

    fun getLastMetricDistanceSource(): String? = synchronized(scaleLock) {
        lastMetricDistanceSource
    }

    fun getLastMetricDistanceDiagnostic(): String? = synchronized(scaleLock) {
        lastMetricDistanceDiagnostic
    }

    /**
     * Call when one segmentation pass finishes so the next camera bitmap is not also delayed by [minFrameIntervalMs]
     * (mirrors iOS `preferImmediateNextInference`).
     */
    fun onInferenceFinished() {
        if (preferImmediateNextBitmap) {
            lastInferencePostMs = 0L
            preferImmediateNextBitmap = false
        }
        val latestPendingBitmap = synchronized(pendingInferenceBitmapLock) {
            val pendingBitmap = pendingInferenceBitmap
            pendingInferenceBitmap = null
            pendingBitmap
        }
        if (latestPendingBitmap != null && shouldPostBitmapFrame()) {
            lastInferencePostMs = SystemClock.elapsedRealtime()
            val consumer = onBitmapFrame
            if (consumer != null) {
                inferenceExecutor.execute { consumer(latestPendingBitmap) }
            } else if (!latestPendingBitmap.isRecycled) {
                latestPendingBitmap.recycle()
            }
        }
    }

    fun requestPhotoCapture(callback: (ArPhotoCaptureResult?) -> Unit) {
        pendingPhotoCaptureRequest = PendingPhotoCaptureRequest(
            callback = callback,
            startedAtMs = SystemClock.elapsedRealtime(),
        )
    }

    fun onHostResume() {
        displayRotationHelper.onResume()
        if (!tryCreateSession()) {
            return
        }
        try {
            session?.resume()
        } catch (e: CameraNotAvailableException) {
            LogUtil.w(TAG, "ARCore resume: camera not available: ${e.message}")
        }
        glSurfaceView.onResume()
    }

    fun onHostPause() {
        displayRotationHelper.onPause()
        val latch = CountDownLatch(1)
        glSurfaceView.queueEvent {
            try {
                session?.pause()
            } catch (_: Exception) {
            } finally {
                latch.countDown()
            }
        }
        glSurfaceView.onPause()
        try {
            latch.await(2, TimeUnit.SECONDS)
        } catch (_: InterruptedException) {
            Thread.currentThread().interrupt()
        }
    }

    fun destroy() {
        measurementHandler.removeCallbacks(debounceApplyRunnable)
        onAssistedMeasurementUpdated = null
        smoothedPinholeWidthMeters = Float.NaN
        smoothedPinholeHeightMeters = Float.NaN
        val latch = CountDownLatch(1)
        glSurfaceView.queueEvent {
            synchronized(sessionGlLock) {
                try {
                    session?.pause()
                } catch (_: Exception) {
                }
                try {
                    session?.close()
                } catch (_: Exception) {
                }
                session = null
            }
            latch.countDown()
        }
        displayRotationHelper.onPause()
        try {
            glSurfaceView.onPause()
            latch.await(3, TimeUnit.SECONDS)
        } catch (_: InterruptedException) {
            Thread.currentThread().interrupt()
        }
        synchronized(pendingInferenceBitmapLock) {
            pendingInferenceBitmap?.takeIf { !it.isRecycled }?.recycle()
            pendingInferenceBitmap = null
        }
        pendingPhotoCaptureRequest?.bestBitmap?.takeIf { !it.isRecycled }?.recycle()
        pendingPhotoCaptureRequest = null
        synchronized(sessionGlLock) {
            session = null
        }
    }

    private fun applyPendingArOverlayScaleFromLatest() {
        val estimatedWidthMeters = pendingEstWForApply
        val estH = pendingEstHForApply
        if (!estimatedWidthMeters.isFinite() || !estH.isFinite()) return
        val applied = synchronized(scaleLock) {
            lastEstimatedWidthMeters = estimatedWidthMeters
            lastEstimatedHeightMeters = estH
            val raw = latestRawScale
            if (raw == null || !raw.isFinite() || raw <= 0f) {
                arOverlayScaleValid = false
                return@synchronized null
            }
            if (snapAfterPrimaryChange) {
                smoothedArOverlayScale = raw.coerceIn(0.25f, 4f)
                snapAfterPrimaryChange = false
            } else {
                val base = smoothedArOverlayScale
                val maxStep = 0.08f
                val target = raw.coerceIn(0.25f, 4f)
                val delta = (target - base).coerceIn(-maxStep, maxStep)
                val clampedTarget = base + delta
                val alpha = 0.16f
                smoothedArOverlayScale = base * (1f - alpha) + clampedTarget * alpha
            }
            arOverlayScaleValid = true
            smoothedArOverlayScale to arOverlayScaleValid
        }
        if (applied != null) {
            val (smoothedOut, validOut) = applied
            lastCommittedHeightAtMs = SystemClock.elapsedRealtime()
            LogUtil.furnitureFitAr(
                "platform=android phase=debounce_apply estW_m=${String.format("%.4f", estimatedWidthMeters)} " +
                    "estH_m=${String.format("%.4f", estH)} " +
                    "smoothedScale=${String.format("%.4f", smoothedOut)} valid=$validOut",
            )
        }
    }

    private fun tryCreateSession(): Boolean {
        synchronized(sessionGlLock) {
            if (session != null) return true
        }
        try {
            when (ArCoreApk.getInstance().requestInstall(activity, !installRequested)) {
                ArCoreApk.InstallStatus.INSTALL_REQUESTED -> {
                    installRequested = true
                    return false
                }
                ArCoreApk.InstallStatus.INSTALLED -> Unit
            }
        } catch (e: Exception) {
            LogUtil.w(TAG, "ARCore install check failed: ${e.message}")
            return false
        }
        return try {
            val newSession = Session(activity)
            val config = Config(newSession).apply {
                planeFindingMode = Config.PlaneFindingMode.HORIZONTAL_AND_VERTICAL
                updateMode = Config.UpdateMode.LATEST_CAMERA_IMAGE
                if (newSession.isDepthModeSupported(Config.DepthMode.AUTOMATIC)) {
                    depthMode = Config.DepthMode.AUTOMATIC
                }
            }
            newSession.configure(config)
            synchronized(sessionGlLock) {
                if (session != null) {
                    try {
                        newSession.close()
                    } catch (_: Exception) {
                    }
                    return true
                }
                session = newSession
            }
            val depthLabel = if (config.depthMode == Config.DepthMode.AUTOMATIC) "AUTOMATIC" else "off"
            LogUtil.furnitureFitAr(
                "platform=android event=session depthMode=$depthLabel planes=${config.planeFindingMode}",
            )
            true
        } catch (e: UnavailableException) {
            LogUtil.w(TAG, "ARCore session unavailable: ${e.message}")
            false
        }
    }

    override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
        GLES20.glClearColor(0.1f, 0.1f, 0.1f, 1f)
        backgroundRenderer.createOnGlThread()
    }

    override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {
        displayRotationHelper.onSurfaceChanged(width, height)
        GLES20.glViewport(0, 0, width, height)
    }

    override fun onDrawFrame(gl: GL10?) {
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT or GLES20.GL_DEPTH_BUFFER_BIT)
        val sess = synchronized(sessionGlLock) { session } ?: return
        displayRotationHelper.updateSessionIfNeeded(sess)
        try {
            sess.setCameraTextureName(backgroundRenderer.getTextureId())
        } catch (_: Exception) {
            return
        }
        val frame = try {
            sess.update()
        } catch (_: CameraNotAvailableException) {
            return
        } catch (_: Throwable) {
            return
        }
        if (frame.timestamp == 0L) {
            return
        }
        val nowMs = SystemClock.elapsedRealtime()
        if (nowMs - lastFurnitureFitArHeartbeatLogMs >= 2000L) {
            lastFurnitureFitArHeartbeatLogMs = nowMs
            LogUtil.furnitureFitAr(
                "platform=android phase=gl_heartbeat ts=${frame.timestamp} " +
                    "shouldPost=${shouldPostBitmapFrame()} pendingCapture=${pendingPhotoCaptureRequest != null}",
            )
        }
        backgroundRenderer.draw(frame)

        val image = try {
            frame.acquireCameraImage()
        } catch (_: Exception) {
            null
        } ?: return

        try {
            val rawW = image.width
            val rawH = image.height
            // AR overlay scale uses bbox + intrinsics only — inverse matrix does not require decoding YUV.
            updateArScaleForCurrentFrame(frame, rawW, rawH)

            val captureRequest = pendingPhotoCaptureRequest
            if (captureRequest != null) {
                val rawBitmap = image.yuv420888ToBitmap()
                if (rawBitmap == null) {
                    pendingPhotoCaptureRequest = null
                    Handler(Looper.getMainLooper()).post { captureRequest.callback(null) }
                    return
                }
                val (orientedBitmap, _) = rawBitmap.rotateToMatchLockedRoomPhoto(lockedPhotoOrientation)
                if (orientedBitmap !== rawBitmap) {
                    rawBitmap.recycle()
                }
                val metricAnchors = FurnitureFitArMetrics.captureSparseMetricAnchors(
                    frame = frame,
                    orientedImageWidth = orientedBitmap.width,
                    orientedImageHeight = orientedBitmap.height,
                    rawImageWidth = rawW,
                    rawImageHeight = rawH,
                    lockedPhotoOrientation = lockedPhotoOrientation,
                )
                captureRequest.attempts += 1
                val elapsedMs = SystemClock.elapsedRealtime() - captureRequest.startedAtMs
                val tracking = frame.camera.trackingState
                if (metricAnchors.size >= captureRequest.bestAnchors.size) {
                    captureRequest.bestBitmap?.takeIf { it !== orientedBitmap && !it.isRecycled }?.recycle()
                    captureRequest.bestBitmap = orientedBitmap
                    captureRequest.bestAnchors = metricAnchors
                } else if (orientedBitmap !== captureRequest.bestBitmap && !orientedBitmap.isRecycled) {
                    orientedBitmap.recycle()
                }
                val done = metricAnchors.size >= 5 || captureRequest.attempts >= 20 || elapsedMs >= 1800L
                LogUtil.d(
                    TAG,
                    "AR photo capture attempt=${captureRequest.attempts} tracking=$tracking anchors=${metricAnchors.size} " +
                        "best=${captureRequest.bestAnchors.size} elapsedMs=$elapsedMs done=$done"
                )
                if (done) {
                    pendingPhotoCaptureRequest = null
                    val finalBitmap = captureRequest.bestBitmap
                    val finalAnchors = captureRequest.bestAnchors
                    Handler(Looper.getMainLooper()).post {
                        captureRequest.callback(finalBitmap?.let { ArPhotoCaptureResult(it, finalAnchors) })
                    }
                }
                return
            }

            if (!shouldPostBitmapFrame()) {
                preferImmediateNextBitmap = true
                val rawBitmap = image.yuv420888ToBitmap() ?: return
                val (orientedBitmap, _) = rawBitmap.rotateToMatchLockedRoomPhoto(lockedPhotoOrientation)
                if (orientedBitmap !== rawBitmap) {
                    rawBitmap.recycle()
                }
                synchronized(pendingInferenceBitmapLock) {
                    pendingInferenceBitmap?.takeIf { it !== orientedBitmap && !it.isRecycled }?.recycle()
                    pendingInferenceBitmap = orientedBitmap
                }
                return
            }

            val now = SystemClock.elapsedRealtime()
            val allowPost = shouldPostBitmapFrame() && (now - lastInferencePostMs >= minFrameIntervalMs)
            if (!allowPost) {
                return
            }

            // Expensive: YUV → bitmap only when we actually feed segmentation (throttled by [minFrameIntervalMs]).
            val rawBmp = image.yuv420888ToBitmap() ?: return
            val (orientedBmp, _) = rawBmp.rotateToMatchLockedRoomPhoto(lockedPhotoOrientation)
            if (orientedBmp !== rawBmp) {
                rawBmp.recycle()
            }

            lastInferencePostMs = now
            val consumer = onBitmapFrame
            if (consumer != null) {
                inferenceExecutor.execute { consumer(orientedBmp) }
            } else {
                orientedBmp.recycle()
            }
        } finally {
            image.close()
        }
    }

    private fun orientedExtentInRawCamera(
        centerImageX: Float,
        centerImageY: Float,
        extentImagePixels: Float,
        alongOrientedXAxis: Boolean,
        rawImageWidth: Int,
        rawImageHeight: Int,
    ): FurnitureFitArMetrics.RawCameraExtent? {
        val halfExtentPixels = extentImagePixels * 0.5f
        val startOrientedX = centerImageX - if (alongOrientedXAxis) halfExtentPixels else 0f
        val startOrientedY = centerImageY - if (alongOrientedXAxis) 0f else halfExtentPixels
        val endOrientedX = centerImageX + if (alongOrientedXAxis) halfExtentPixels else 0f
        val endOrientedY = centerImageY + if (alongOrientedXAxis) 0f else halfExtentPixels
        val startRaw = mapOrientedImagePixelToRawCameraPixel(
            startOrientedX,
            startOrientedY,
            rawImageWidth,
            rawImageHeight,
            lockedPhotoOrientation,
        )
        val endRaw = mapOrientedImagePixelToRawCameraPixel(
            endOrientedX,
            endOrientedY,
            rawImageWidth,
            rawImageHeight,
            lockedPhotoOrientation,
        )
        return FurnitureFitArMetrics.rawCameraExtentFromMappedEndpoints(
            startRawX = startRaw.first,
            startRawY = startRaw.second,
            endRawX = endRaw.first,
            endRawY = endRaw.second,
        )
    }

    private fun updateArScaleForCurrentFrame(
        frame: Frame,
        rawImageWidth: Int,
        rawImageHeight: Int,
    ) {
        val tracking = frame.camera.trackingState
        val nowMs = SystemClock.elapsedRealtime()

        val hint: Boolean
        val cx: Float
        val cy: Float
        val bboxWidthOrientedPixels: Float
        val bboxHeightOrientedPixels: Float
        val labelForLog: String
        synchronized(bboxLock) {
            hint = bboxHintValid
            cx = bboxCenterImageX
            cy = bboxCenterImageY
            bboxWidthOrientedPixels = bboxWidthImagePx
            bboxHeightOrientedPixels = bboxHeightImagePx
            labelForLog = bboxLabel
        }
        if (!hint) {
            if (nowMs - lastFurnitureFitArWaitBboxLogMs >= 2000L) {
                lastFurnitureFitArWaitBboxLogMs = nowMs
                LogUtil.furnitureFitAr(
                    "platform=android phase=wait_bbox tracking=${tracking.name} " +
                        "roomFallback_m=${String.format("%.3f", roomHeightMetersForFallback)} " +
                        "rawWH=(${rawImageWidth}x${rawImageHeight}) " +
                        "note=no_bbox_hint_until_segmentation_sets_primary",
                )
            }
            return
        }

        val (rawCx, rawCy) = mapOrientedImagePixelToRawCameraPixel(
            cx,
            cy,
            rawImageWidth,
            rawImageHeight,
            lockedPhotoOrientation,
        )
        val rawWidthExtent = orientedExtentInRawCamera(
            centerImageX = cx,
            centerImageY = cy,
            extentImagePixels = bboxWidthOrientedPixels,
            alongOrientedXAxis = true,
            rawImageWidth = rawImageWidth,
            rawImageHeight = rawImageHeight,
        ) ?: return
        val rawHeightExtent = orientedExtentInRawCamera(
            centerImageX = cx,
            centerImageY = cy,
            extentImagePixels = bboxHeightOrientedPixels,
            alongOrientedXAxis = false,
            rawImageWidth = rawImageWidth,
            rawImageHeight = rawImageHeight,
        ) ?: return
        val wRaw = rawWidthExtent.pixels
        val hRaw = rawHeightExtent.pixels
        // Loose segmentation boxes can span most of the frame; huge spans blow up pinhole (meters) when depth glitches.
        val widthAxisImagePixels =
            if (rawWidthExtent.focalAxis == FurnitureFitArMetrics.RawCameraAxis.X) rawImageWidth else rawImageHeight
        val heightAxisImagePixels =
            if (rawHeightExtent.focalAxis == FurnitureFitArMetrics.RawCameraAxis.X) rawImageWidth else rawImageHeight
        val maxWidthSpanPx = widthAxisImagePixels.toFloat() * 0.68f
        val maxHeightSpanPx = heightAxisImagePixels.toFloat() * 0.68f
        val hRawForDepthGrid = hRaw.coerceIn(28f, maxOf(96f, maxHeightSpanPx))
        val wRawForPinhole = wRaw.coerceIn(20f, maxWidthSpanPx * 1.08f)
        val hRawForPinhole = hRaw.coerceIn(20f, maxHeightSpanPx * 1.08f)

        val intrinsics = frame.camera.imageIntrinsics
        val widthFocalLengthPixels = FurnitureFitArMetrics.focalLengthPixelsForImage(
            intrinsics,
            rawImageWidth,
            rawImageHeight,
            rawWidthExtent.focalAxis,
        )
        val heightFocalLengthPixels = FurnitureFitArMetrics.focalLengthPixelsForImage(
            intrinsics,
            rawImageWidth,
            rawImageHeight,
            rawHeightExtent.focalAxis,
        )
        val viewW = glSurfaceView.width.coerceAtLeast(0)
        val viewH = glSurfaceView.height.coerceAtLeast(0)
        val distDebug = FurnitureFitArMetrics.metricDistanceEstimateDebug(
            frame = frame,
            imageX = rawCx,
            imageY = rawCy,
            bboxHeightPixels = hRawForDepthGrid,
            rawImageWidth = rawImageWidth,
            rawImageHeight = rawImageHeight,
            viewWidthPx = viewW,
            viewHeightPx = viewH,
        )
        val distEstimate = distDebug.estimate
        synchronized(scaleLock) {
            lastMetricDistanceMeters = distEstimate?.meters ?: Float.NaN
            lastMetricDistanceSource = distEstimate?.source
            lastMetricDistanceDiagnostic = distDebug.diagnostic
        }
        // Reject far-plane / multipath spikes (e.g. 15–20 m indoors) that drive absurd pinhole heights.
        val distForPinhole = distEstimate?.takeIf { d ->
            d.meters.isFinite() && d.meters in 0.12f..9.5f
        }
        val pinholeRawWidthMeters = distForPinhole?.let { distanceEstimate ->
            FurnitureFitArMetrics.estimatedPhysicalWidthMeters(
                wRawForPinhole,
                distanceEstimate.meters,
                widthFocalLengthPixels,
            )
        }
        val pinholeRawHeightMeters = distForPinhole?.let { distanceEstimate ->
            FurnitureFitArMetrics.estimatedPhysicalHeightMeters(
                hRawForPinhole,
                distanceEstimate.meters,
                heightFocalLengthPixels,
            )
        }
        val instantWidthMeters = pinholeRawWidthMeters?.takeIf { it.isFinite() && it in 0.055f..6.5f }
        val instantHeightMeters = pinholeRawHeightMeters?.takeIf { it.isFinite() && it in 0.055f..4.8f }
        if (instantWidthMeters != null && instantHeightMeters != null) {
            if (!smoothedPinholeWidthMeters.isFinite()) {
                smoothedPinholeWidthMeters = instantWidthMeters
            } else {
                val maximumWidthStepMeters = kotlin.math.max(smoothedPinholeWidthMeters * 0.22f, 0.06f)
                val clampedWidthMeters = instantWidthMeters.coerceIn(
                    smoothedPinholeWidthMeters - maximumWidthStepMeters,
                    smoothedPinholeWidthMeters + maximumWidthStepMeters,
                )
                val smoothingAlpha = 0.22f
                smoothedPinholeWidthMeters =
                    smoothingAlpha * clampedWidthMeters + (1f - smoothingAlpha) * smoothedPinholeWidthMeters
            }
            if (!smoothedPinholeHeightMeters.isFinite()) {
                smoothedPinholeHeightMeters = instantHeightMeters
            } else {
                val maxStep = kotlin.math.max(smoothedPinholeHeightMeters * 0.22f, 0.06f)
                val clamped = instantHeightMeters.coerceIn(
                    smoothedPinholeHeightMeters - maxStep,
                    smoothedPinholeHeightMeters + maxStep,
                )
                val alpha = 0.22f
                smoothedPinholeHeightMeters = alpha * clamped + (1f - alpha) * smoothedPinholeHeightMeters
            }
        }

        val fallbackHDiag = FurnitureFitArMetrics.approximateHeightFromRoomAndBboxFraction(
            roomHeightMetersForFallback,
            hRawForPinhole,
            rawImageHeight,
        )
        val estimatedWidthMeters = smoothedPinholeWidthMeters.takeIf { it.isFinite() && it >= 0.055f }
        val estH = smoothedPinholeHeightMeters.takeIf { it.isFinite() && it >= 0.055f }
        if (estimatedWidthMeters == null || estH == null) {
            if (nowMs - lastFurnitureFitArFrameSkipLogMs >= 500L) {
                lastFurnitureFitArFrameSkipLogMs = nowMs
                val distSrc = distEstimate?.source ?: "none"
                val distM = distEstimate?.meters
                LogUtil.furnitureFitAr(
                    buildString {
                        append("platform=android phase=frame_skip reason=no_pinhole_yet ")
                        append("tracking=${tracking.name} ")
                        append("distSource=$distSrc ")
                        if (distM != null) append("dist_m=${String.format("%.4f", distM)} ") else append("dist_m=null ")
                        append("distDiag=${distDebug.diagnostic} ")
                        append("focalWH_px=(${String.format("%.2f", widthFocalLengthPixels)}x")
                        append("${String.format("%.2f", heightFocalLengthPixels)}) ")
                        append("label=${labelForLog.take(48)} ")
                        append("pinholeRawWH_m=(")
                        append(if (pinholeRawWidthMeters != null && pinholeRawWidthMeters.isFinite()) String.format("%.4f", pinholeRawWidthMeters) else "null")
                        append("x")
                        append(if (pinholeRawHeightMeters != null && pinholeRawHeightMeters.isFinite()) String.format("%.4f", pinholeRawHeightMeters) else "null")
                        append(") ")
                        append(" room_frac_diag_m=")
                        append(if (fallbackHDiag != null && fallbackHDiag.isFinite()) String.format("%.4f", fallbackHDiag) else "null")
                        append(" note=room_frac_not_used_for_scale")
                        append(" room_m=${String.format("%.3f", roomHeightMetersForFallback)}")
                    },
                )
            }
            return
        }
        val stdH = 0.85f
        val raw = FurnitureFitArMetrics.overlayScaleFromMetricHeights(stdH, estH, 0.25f, 4f)
        if (nowMs - lastFurnitureFitArFrameLogMs >= 500L) {
            lastFurnitureFitArFrameLogMs = nowMs
            val distSrc = distEstimate?.source ?: "smooth_hold"
            val distM = distEstimate?.meters
            val nx = rawCx / rawImageWidth.coerceAtLeast(1)
            val ny = rawCy / rawImageHeight.coerceAtLeast(1)
            LogUtil.furnitureFitAr(
                buildString {
                    append("platform=android phase=frame ")
                    append("tracking=${tracking.name} ")
                    append("distSource=$distSrc ")
                    if (distM != null) append("dist_m=${String.format("%.4f", distM)} ") else append("dist_m=null ")
                    append("distDiag=${distDebug.diagnostic} ")
                    append("focalWH_px=(${String.format("%.2f", widthFocalLengthPixels)}x")
                    append("${String.format("%.2f", heightFocalLengthPixels)}) ")
                    append("bboxWH_px=(${String.format("%.2f", wRawForPinhole)}x")
                    append("${String.format("%.2f", hRawForPinhole)}) ")
                    append("bboxAxes=(${rawWidthExtent.focalAxis},${rawHeightExtent.focalAxis}) ")
                    append("label=${labelForLog.take(48)} ")
                    append("rawWH=(${rawImageWidth}x${rawImageHeight}) ")
                    append("oriented_xy=(${String.format("%.1f", cx)},${String.format("%.1f", cy)}) ")
                    append("raw_xy=(${String.format("%.1f", rawCx)},${String.format("%.1f", rawCy)}) ")
                    append("norm_raw=(${String.format("%.4f", nx)},${String.format("%.4f", ny)}) ")
                    append("pinholeInstantWH_m=(")
                    append(if (instantWidthMeters != null) String.format("%.4f", instantWidthMeters) else "null")
                    append("x")
                    append(if (instantHeightMeters != null) String.format("%.4f", instantHeightMeters) else "null")
                    append(") ")
                    append("smoothedEstWH_m=(${String.format("%.4f", estimatedWidthMeters)}x")
                    append("${String.format("%.4f", estH)}) ")
                    append("stdH_m=$stdH ")
                    append("rawScale=${raw?.let { String.format("%.4f", it) } ?: "null"} ")
                    append("room_m_diag=${String.format("%.3f", roomHeightMetersForFallback)}")
                },
            )
        }
        val distForProvisional = distForPinhole?.meters ?: distEstimate?.meters?.takeIf { it.isFinite() && it > 0f }
        provisionalWidthMeters = estimatedWidthMeters
        provisionalHeightMeters = estH
        provisionalRawScale = raw
        provisionalDistanceMeters = distForProvisional
        provisionalHeightTimestampMs = SystemClock.elapsedRealtime()

        pendingEstWForApply = estimatedWidthMeters
        pendingEstHForApply = estH
        latestRawScale = raw
        measurementHandler.removeCallbacks(debounceApplyRunnable)
        measurementHandler.postDelayed(debounceApplyRunnable, assistedMeasurementDebounceMs)
    }
}
