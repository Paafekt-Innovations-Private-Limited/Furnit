package com.furnit.android.ar

import android.app.Activity
import android.graphics.Bitmap
import android.graphics.Matrix
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
import java.util.concurrent.Executor
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
 * ARCore path for FurnitureFit: [GLSurfaceView] + [Session], CPU bitmaps for YOLO, and smoothed
 * overlay scale from horizontal plane hit distance + pinhole height ([FurnitureFitArMetrics]).
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

    /** Invoked on [inferenceExecutor] with a decoded camera bitmap. */
    var onBitmapFrame: ((Bitmap) -> Unit)? = null

    /** If false, the GL thread skips posting new frames (e.g. while YOLO is running). */
    var shouldPostBitmapFrame: () -> Boolean = { true }

    @Volatile
    private var pendingPhotoCaptureRequest: PendingPhotoCaptureRequest? = null

    private val bboxLock = Any()
    private var bboxCenterImageX = 0f
    private var bboxCenterImageY = 0f
    private var bboxHeightImagePx = 0f
    private var bboxLabel = ""
    private var bboxHintValid = false

    private val scaleLock = Any()
    private var smoothedArOverlayScale = 1f
    private var arOverlayScaleValid = false
    private var lastEstimatedHeightMeters = Float.NaN

    private var lastInferencePostMs = 0L
    /** Min time between frames handed to YOLO (iOS ~0.07s; slightly lower here for responsiveness). */
    var minFrameIntervalMs: Long = 55L

    @Volatile
    private var preferImmediateNextBitmap = false

    private val measurementHandler = Handler(Looper.getMainLooper())
    /** Debounced apply of AR overlay scale (mirrors iOS `assistedMeasurementDebounceSeconds` ~0.85s). */
    private val assistedMeasurementDebounceMs = 550L
    @Volatile
    private var latestRawScale: Float? = null
    @Volatile
    private var pendingEstHForApply = Float.NaN
    private var snapAfterPrimaryChange = false

    private val debounceApplyRunnable = Runnable {
        applyPendingArOverlayScaleFromLatest()
        onAssistedMeasurementUpdated?.invoke()
    }

    /**
     * Called on the main thread after debounced AR metric sizing is applied (including
     * [lastEstimatedHeightMeters]). Used so the Sharp room calibration pill refreshes — the pill
     * is not driven by the GL thread every frame.
     */
    var onAssistedMeasurementUpdated: (() -> Unit)? = null

    /**
     * Room photo lock from FurnitureFit / Sharp room (`"portrait"` or `"landscape"`), matching CameraX
     * target rotation so segmentation bitmap aspect matches the locked activity.
     */
    var lockedPhotoOrientation: String = "portrait"

    /** Cache for [orientedToRawInverseForDimensions] — avoids per-frame bitmap work for AR metrics. */
    private var cachedInverseRawW = -1
    private var cachedInverseRawH = -1
    private var cachedInverseOrientation: String = ""
    private var cachedOrientedToRawInverse: Matrix? = null

    private fun orientedInverseForCurrentCamera(rawW: Int, rawH: Int): Matrix? {
        if (rawW == cachedInverseRawW && rawH == cachedInverseRawH &&
            lockedPhotoOrientation == cachedInverseOrientation
        ) {
            return cachedOrientedToRawInverse
        }
        cachedInverseRawW = rawW
        cachedInverseRawH = rawH
        cachedInverseOrientation = lockedPhotoOrientation
        cachedOrientedToRawInverse = orientedToRawInverseForDimensions(rawW, rawH, lockedPhotoOrientation)
        return cachedOrientedToRawInverse
    }

    fun setBboxHint(centerImageX: Float, centerImageY: Float, heightImagePx: Float, label: String) {
        val labelChanged = synchronized(bboxLock) {
            val prev = bboxLabel
            bboxCenterImageX = centerImageX
            bboxCenterImageY = centerImageY
            bboxHeightImagePx = heightImagePx
            bboxLabel = label
            bboxHintValid = centerImageX.isFinite() && centerImageY.isFinite() && heightImagePx.isFinite()
            prev != label
        }
        if (labelChanged) {
            snapAfterPrimaryChange = true
            latestRawScale = null
            pendingEstHForApply = Float.NaN
            measurementHandler.removeCallbacks(debounceApplyRunnable)
            synchronized(scaleLock) {
                arOverlayScaleValid = false
                smoothedArOverlayScale = 1f
                lastEstimatedHeightMeters = Float.NaN
            }
        }
    }

    fun clearBboxHint() {
        synchronized(bboxLock) {
            bboxHintValid = false
        }
        latestRawScale = null
        pendingEstHForApply = Float.NaN
        measurementHandler.removeCallbacks(debounceApplyRunnable)
        synchronized(scaleLock) {
            arOverlayScaleValid = false
            smoothedArOverlayScale = 1f
            lastEstimatedHeightMeters = Float.NaN
        }
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

    /**
     * Call when one segmentation pass finishes so the next camera bitmap is not also delayed by [minFrameIntervalMs]
     * (mirrors iOS `preferImmediateNextInference`).
     */
    fun onInferenceFinished() {
        if (preferImmediateNextBitmap) {
            lastInferencePostMs = 0L
            preferImmediateNextBitmap = false
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
        glSurfaceView.onPause()
        session?.pause()
        displayRotationHelper.onPause()
    }

    fun destroy() {
        measurementHandler.removeCallbacks(debounceApplyRunnable)
        onAssistedMeasurementUpdated = null
        onHostPause()
        pendingPhotoCaptureRequest?.bestBitmap?.takeIf { !it.isRecycled }?.recycle()
        pendingPhotoCaptureRequest = null
        session?.close()
        session = null
    }

    private fun applyPendingArOverlayScaleFromLatest() {
        val estH = pendingEstHForApply
        if (!estH.isFinite()) return
        synchronized(scaleLock) {
            lastEstimatedHeightMeters = estH
            val raw = latestRawScale
            if (raw == null || !raw.isFinite() || raw <= 0f) {
                arOverlayScaleValid = false
                return
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
        }
    }

    private fun tryCreateSession(): Boolean {
        if (session != null) return true
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
            session = newSession
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
        val sess = session ?: return
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
            val orientedInverse = orientedInverseForCurrentCamera(rawW, rawH)
            updateArScaleForCurrentFrame(frame, rawW, rawH, orientedInverse)

            val captureRequest = pendingPhotoCaptureRequest
            if (captureRequest != null) {
                val rawBitmap = image.yuv420888ToBitmap()
                if (rawBitmap == null) {
                    pendingPhotoCaptureRequest = null
                    Handler(Looper.getMainLooper()).post { captureRequest.callback(null) }
                    return
                }
                val (orientedBitmap, orientedToRawInverse) = rawBitmap.rotateToMatchLockedRoomPhoto(lockedPhotoOrientation)
                if (orientedBitmap !== rawBitmap) {
                    rawBitmap.recycle()
                }
                val metricAnchors = FurnitureFitArMetrics.captureSparseMetricAnchors(
                    frame = frame,
                    orientedImageWidth = orientedBitmap.width,
                    orientedImageHeight = orientedBitmap.height,
                    rawImageWidth = rawW,
                    rawImageHeight = rawH,
                    orientedToRawInverse = orientedToRawInverse,
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
            }

            val now = SystemClock.elapsedRealtime()
            val allowPost = shouldPostBitmapFrame() && (now - lastInferencePostMs >= minFrameIntervalMs)
            if (!allowPost) {
                return
            }

            // Expensive: YUV → bitmap only when we actually feed YOLO (throttled by [minFrameIntervalMs]).
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

    private fun updateArScaleForCurrentFrame(
        frame: Frame,
        rawImageWidth: Int,
        rawImageHeight: Int,
        orientedToRawInverse: Matrix?,
    ) {
        val hint: Boolean
        val cx: Float
        val cy: Float
        val hPx: Float
        synchronized(bboxLock) {
            hint = bboxHintValid
            cx = bboxCenterImageX
            cy = bboxCenterImageY
            hPx = bboxHeightImagePx
        }
        if (!hint) {
            return
        }

        val rawCx: Float
        val rawCy: Float
        val hRaw: Float
        if (orientedToRawInverse != null) {
            val c = floatArrayOf(cx, cy)
            orientedToRawInverse.mapPoints(c)
            rawCx = c[0]
            rawCy = c[1]
            hRaw = orientedBboxHeightToRawPixels(cx, cy, hPx, orientedToRawInverse)
        } else {
            rawCx = cx
            rawCy = cy
            hRaw = hPx
        }

        val intrinsics = frame.camera.imageIntrinsics
        val fy = FurnitureFitArMetrics.focalLengthYPixelsForImage(intrinsics, rawImageWidth, rawImageHeight)
        val distEstimate = FurnitureFitArMetrics.metricDistanceEstimate(
            frame = frame,
            imageX = rawCx,
            imageY = rawCy,
            bboxHeightPixels = hRaw,
            rawImageWidth = rawImageWidth,
            rawImageHeight = rawImageHeight,
        ) ?: return
        val estH = FurnitureFitArMetrics.estimatedPhysicalHeightMeters(hRaw, distEstimate.meters, fy) ?: return
        val stdH = 0.85f
        val raw = FurnitureFitArMetrics.overlayScaleFromMetricHeights(stdH, estH, 0.25f, 4f)
        LogUtil.d(
            TAG,
            "AR metric sizing source=${distEstimate.source} dist=${distEstimate.meters}m estH=$estH fy=$fy bboxH=$hRaw rawCenter=($rawCx,$rawCy) overlayRaw=$raw"
        )
        pendingEstHForApply = estH
        latestRawScale = raw
        measurementHandler.removeCallbacks(debounceApplyRunnable)
        measurementHandler.postDelayed(debounceApplyRunnable, assistedMeasurementDebounceMs)
    }
}
