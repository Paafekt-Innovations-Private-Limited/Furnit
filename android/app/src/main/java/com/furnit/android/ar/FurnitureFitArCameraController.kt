package com.furnit.android.ar

import android.app.Activity
import android.graphics.Bitmap
import android.graphics.Matrix
import android.opengl.GLES20
import android.opengl.GLSurfaceView
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

    private val bboxLock = Any()
    private var bboxCenterImageX = 0f
    private var bboxCenterImageY = 0f
    private var bboxHeightImagePx = 0f
    private var bboxLabel = ""
    private var bboxHintValid = false

    private val scaleLock = Any()
    private var smoothedArOverlayScale = 1f
    private var arOverlayScaleValid = false

    private var lastInferencePostMs = 0L
    var minFrameIntervalMs: Long = 85L

    /**
     * Room photo lock from FurnitureFit / Sharp room (`"portrait"` or `"landscape"`), matching CameraX
     * target rotation so segmentation bitmap aspect matches the locked activity.
     */
    var lockedPhotoOrientation: String = "portrait"

    fun setBboxHint(centerImageX: Float, centerImageY: Float, heightImagePx: Float, label: String) {
        synchronized(bboxLock) {
            bboxCenterImageX = centerImageX
            bboxCenterImageY = centerImageY
            bboxHeightImagePx = heightImagePx
            bboxLabel = label
            bboxHintValid = heightImagePx > 2f
        }
    }

    fun clearBboxHint() {
        synchronized(bboxLock) {
            bboxHintValid = false
        }
        synchronized(scaleLock) {
            arOverlayScaleValid = false
            smoothedArOverlayScale = 1f
        }
    }

    fun isArOverlayScaleValid(): Boolean = synchronized(scaleLock) { arOverlayScaleValid }

    fun getSmoothedArOverlayScale(): Float = synchronized(scaleLock) { smoothedArOverlayScale }

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
        onHostPause()
        session?.close()
        session = null
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
            val rawBmp = image.yuv420888ToBitmap() ?: return
            val (orientedBmp, orientedToRawInverse) = rawBmp.rotateToMatchLockedRoomPhoto(lockedPhotoOrientation)
            if (orientedBmp !== rawBmp) {
                rawBmp.recycle()
            }

            updateArScaleForCurrentFrame(frame, rawW, rawH, orientedToRawInverse)

            val now = SystemClock.elapsedRealtime()
            val allowPost = shouldPostBitmapFrame() && (now - lastInferencePostMs >= minFrameIntervalMs)
            if (allowPost) {
                lastInferencePostMs = now
                val consumer = onBitmapFrame
                if (consumer != null) {
                    inferenceExecutor.execute { consumer(orientedBmp) }
                } else {
                    orientedBmp.recycle()
                }
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
        val label: String
        synchronized(bboxLock) {
            hint = bboxHintValid
            cx = bboxCenterImageX
            cy = bboxCenterImageY
            hPx = bboxHeightImagePx
            label = bboxLabel
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

        val viewOut = FloatArray(2)
        if (!FurnitureFitArMetrics.imagePixelsToViewPixels(frame, rawCx, rawCy, viewOut)) {
            return
        }
        val dist = FurnitureFitArMetrics.horizontalPlaneHitDistanceMeters(frame, viewOut[0], viewOut[1])
            ?: return

        val intrinsics = frame.camera.imageIntrinsics
        val fy = FurnitureFitArMetrics.focalLengthYPixelsForImage(intrinsics, rawImageWidth, rawImageHeight)
        val stdH = FurnitureFitStandardHeights.heightMetersForLabel(label)
        val estH = FurnitureFitArMetrics.estimatedPhysicalHeightMeters(hRaw, dist, fy) ?: return
        val rawScale = FurnitureFitArMetrics.overlayScaleFromMetricHeights(stdH, estH) ?: return
        synchronized(scaleLock) {
            val alpha = 0.28f
            smoothedArOverlayScale = smoothedArOverlayScale * (1f - alpha) + rawScale * alpha
            arOverlayScaleValid = true
        }
    }
}
