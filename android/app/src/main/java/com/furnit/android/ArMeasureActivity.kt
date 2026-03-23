package com.furnit.android

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Color
import android.opengl.GLES20
import android.opengl.GLSurfaceView
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.Gravity
import android.view.MotionEvent
import android.view.ViewGroup
import android.widget.Button
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import com.furnit.android.ar.ArBackgroundRenderer
import com.furnit.android.ar.ArWorldToScreen
import com.furnit.android.ar.ArMeasureOverlayView
import com.furnit.android.ar.DisplayRotationHelper
import com.google.ar.core.Anchor
import com.google.ar.core.Config
import com.google.ar.core.Frame
import com.google.ar.core.Plane
import com.google.ar.core.Session
import com.google.ar.core.TrackingState
import com.google.ar.core.exceptions.CameraNotAvailableException
import com.google.ar.core.exceptions.UnavailableException
import com.google.ar.core.ArCoreApk
import java.util.concurrent.ConcurrentLinkedQueue
import java.util.concurrent.atomic.AtomicReference
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10
import kotlin.math.sqrt

/**
 * Experimental ARCore distance measure: tap two points on a tracked plane; metric length in meters.
 * Optional extras describe SHARP bbox for calibration UX in the caller.
 */
class ArMeasureActivity : AppCompatActivity(), GLSurfaceView.Renderer {

    private lateinit var glSurfaceView: GLSurfaceView
    private lateinit var overlayView: ArMeasureOverlayView
    private lateinit var statusText: TextView
    private lateinit var distanceText: TextView

    /** Must be created after [super.onCreate]; Activity base context is null during field init. */
    private lateinit var displayRotationHelper: DisplayRotationHelper
    private val backgroundRenderer = ArBackgroundRenderer()
    private val mainHandler = Handler(Looper.getMainLooper())

    @Volatile
    private var session: Session? = null
    private var installRequested = false
    private var viewportWidth = 1
    private var viewportHeight = 1

    private val anchors = mutableListOf<Anchor>()
    private val pendingHits = ConcurrentLinkedQueue<Pair<Float, Float>>()
    private val lastUiStatus = AtomicReference("")
    private val lastUiDistText = AtomicReference("")

    private val cameraPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission(),
    ) { granted ->
        if (!granted) {
            Toast.makeText(this, R.string.ar_measure_camera_required, Toast.LENGTH_LONG).show()
            finish()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        displayRotationHelper = DisplayRotationHelper(this)
        WindowCompat.setDecorFitsSystemWindows(window, false)
        window.statusBarColor = Color.TRANSPARENT

        glSurfaceView = GLSurfaceView(this).apply {
            preserveEGLContextOnPause = true
            setEGLContextClientVersion(2)
            setEGLConfigChooser(8, 8, 8, 8, 16, 0)
            setRenderer(this@ArMeasureActivity)
            renderMode = GLSurfaceView.RENDERMODE_CONTINUOUSLY
            setOnTouchListener { _, e ->
                if (e.action == MotionEvent.ACTION_DOWN) {
                    pendingHits.offer(e.x to e.y)
                    true
                } else {
                    false
                }
            }
        }

        overlayView = ArMeasureOverlayView(this).apply {
            setBackgroundColor(Color.TRANSPARENT)
        }

        statusText = TextView(this).apply {
            setTextColor(Color.WHITE)
            textSize = 14f
            setShadowLayer(4f, 0f, 0f, Color.BLACK)
            text = getString(R.string.ar_measure_status_init)
        }
        distanceText = TextView(this).apply {
            setTextColor(Color.WHITE)
            textSize = 20f
            setTypeface(null, android.graphics.Typeface.BOLD)
            setShadowLayer(4f, 0f, 0f, Color.BLACK)
            text = ""
        }

        val topPanel = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(32, 48, 32, 16)
            setBackgroundColor(Color.parseColor("#66000000"))
            addView(statusText)
            addView(distanceText)
        }

        val resetBtn = Button(this).apply {
            text = getString(R.string.ar_measure_reset)
            setOnClickListener { resetMeasurement() }
        }
        val doneBtn = Button(this).apply {
            text = getString(R.string.ar_measure_done)
            setOnClickListener { finishWithResult() }
        }
        val bottomRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER
            setPadding(24, 16, 24, 40)
            setBackgroundColor(Color.parseColor("#66000000"))
            addView(resetBtn, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
            addView(doneBtn, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
        }

        val root = FrameLayout(this).apply {
            addView(
                glSurfaceView,
                FrameLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.MATCH_PARENT,
                ),
            )
            addView(
                overlayView,
                FrameLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.MATCH_PARENT,
                ),
            )
            addView(
                topPanel,
                FrameLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                ).apply { gravity = Gravity.TOP },
            )
            addView(
                bottomRow,
                FrameLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                ).apply { gravity = Gravity.BOTTOM },
            )
        }

        setContentView(root)
        ViewCompat.setOnApplyWindowInsetsListener(root) { v, insets ->
            val bars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            topPanel.setPadding(32, bars.top + 16, 32, 16)
            bottomRow.setPadding(24, 16, 24, bars.bottom + 16)
            insets
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    override fun onResume() {
        super.onResume()
        displayRotationHelper.onResume()
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            return
        }
        if (!tryCreateSession()) {
            return
        }
        try {
            session?.resume()
        } catch (e: CameraNotAvailableException) {
            Toast.makeText(this, R.string.ar_measure_camera_unavailable, Toast.LENGTH_LONG).show()
            finish()
            return
        }
        glSurfaceView.onResume()
    }

    override fun onPause() {
        super.onPause()
        glSurfaceView.onPause()
        session?.pause()
        displayRotationHelper.onPause()
    }

    override fun onDestroy() {
        session?.close()
        session = null
        super.onDestroy()
    }

    private fun tryCreateSession(): Boolean {
        if (session != null) return true
        try {
            when (ArCoreApk.getInstance().requestInstall(this, !installRequested)) {
                ArCoreApk.InstallStatus.INSTALL_REQUESTED -> {
                    installRequested = true
                    return false
                }
                ArCoreApk.InstallStatus.INSTALLED -> { }
            }
        } catch (e: Exception) {
            Toast.makeText(this, getString(R.string.ar_measure_arcore_failed, e.message ?: ""), Toast.LENGTH_LONG).show()
            finish()
            return false
        }

        return try {
            val newSession = Session(this)
            val config = Config(newSession).apply {
                planeFindingMode = Config.PlaneFindingMode.HORIZONTAL_AND_VERTICAL
                updateMode = Config.UpdateMode.LATEST_CAMERA_IMAGE
            }
            newSession.configure(config)
            session = newSession
            true
        } catch (e: UnavailableException) {
            Toast.makeText(this, getString(R.string.ar_measure_session_unavailable, e.message ?: ""), Toast.LENGTH_LONG).show()
            finish()
            false
        }
    }

    override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
        GLES20.glClearColor(0.1f, 0.1f, 0.1f, 1f)
        backgroundRenderer.createOnGlThread()
    }

    override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {
        viewportWidth = width.coerceAtLeast(1)
        viewportHeight = height.coerceAtLeast(1)
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
        } catch (e: CameraNotAvailableException) {
            return
        } catch (_: Throwable) {
            return
        }

        if (frame.timestamp == 0L) {
            return
        }

        backgroundRenderer.draw(frame)

        processHits(frame)
        updateUiFromFrame(frame, sess)
        updateOverlay(frame)
    }

    private fun processHits(frame: Frame) {
        var tap: Pair<Float, Float>? = null
        while (true) {
            val p = pendingHits.poll() ?: break
            tap = p
        }
        if (tap == null) return
        val x = tap.first
        val y = tap.second

        if (frame.camera.trackingState != TrackingState.TRACKING) {
            mainHandler.post {
                Toast.makeText(this, R.string.ar_measure_tracking_lost, Toast.LENGTH_SHORT).show()
            }
            return
        }

        var placed: Anchor? = null
        val hits = frame.hitTest(x, y)
        for (hit in hits) {
            val t = hit.trackable
            if (t is Plane && t.trackingState == TrackingState.TRACKING && t.isPoseInPolygon(hit.hitPose)) {
                placed = hit.createAnchor()
                break
            }
        }

        if (placed == null) {
            mainHandler.post {
                Toast.makeText(this, R.string.ar_measure_no_plane_hit, Toast.LENGTH_SHORT).show()
            }
            return
        }

        synchronized(anchors) {
            while (anchors.size >= 2) {
                anchors.removeAt(0).detach()
            }
            anchors.add(placed)
        }
    }

    private fun updateUiFromFrame(frame: Frame, sess: Session) {
        val cam = frame.camera.trackingState
        val planes = sess.getAllTrackables(Plane::class.java).count {
            it.trackingState == TrackingState.TRACKING
        }
        val dist = synchronized(anchors) {
            if (anchors.size < 2) Float.NaN
            else distanceMeters(anchors[0].pose, anchors[1].pose)
        }
        val statusStr = when (cam) {
            TrackingState.TRACKING -> getString(R.string.ar_measure_status_tracking, planes)
            TrackingState.PAUSED -> getString(R.string.ar_measure_status_paused)
            else -> getString(R.string.ar_measure_status_not_tracking)
        }
        val distStr = if (!dist.isNaN()) {
            getString(R.string.ar_measure_distance_m, dist)
        } else {
            getString(R.string.ar_measure_tap_two_points)
        }
        if (statusStr == lastUiStatus.get() && distStr == lastUiDistText.get()) return
        lastUiStatus.set(statusStr)
        lastUiDistText.set(distStr)
        mainHandler.post {
            statusText.text = statusStr
            distanceText.text = distStr
        }
    }

    private fun updateOverlay(frame: Frame) {
        val pts = synchronized(anchors) {
            if (anchors.isEmpty()) emptyList()
            else {
                val out = mutableListOf<Pair<Float, Float>>()
                val scratch = FloatArray(2)
                for (a in anchors) {
                    val p = a.pose
                    val tr = FloatArray(3)
                    p.getTranslation(tr, 0)
                    if (ArWorldToScreen.project(
                            frame,
                            tr[0],
                            tr[1],
                            tr[2],
                            viewportWidth,
                            viewportHeight,
                            scratch,
                        )
                    ) {
                        out.add(scratch[0] to scratch[1])
                    }
                }
                out
            }
        }
        overlayView.setProjectedPoints(pts)
    }

    private fun distanceMeters(poseA: com.google.ar.core.Pose, poseB: com.google.ar.core.Pose): Float {
        val a = FloatArray(3)
        val b = FloatArray(3)
        poseA.getTranslation(a, 0)
        poseB.getTranslation(b, 0)
        val dx = a[0] - b[0]
        val dy = a[1] - b[1]
        val dz = a[2] - b[2]
        return sqrt(dx * dx + dy * dy + dz * dz)
    }

    private fun resetMeasurement() {
        synchronized(anchors) {
            anchors.forEach { it.detach() }
            anchors.clear()
        }
        lastUiDistText.set("")
        mainHandler.post {
            distanceText.text = getString(R.string.ar_measure_tap_two_points)
        }
    }

    private fun finishWithResult() {
        val dist = synchronized(anchors) {
            if (anchors.size < 2) Float.NaN
            else distanceMeters(anchors[0].pose, anchors[1].pose)
        }
        if (dist.isNaN()) {
            Toast.makeText(this, R.string.ar_measure_need_two_points, Toast.LENGTH_SHORT).show()
            return
        }
        val data = Intent().apply {
            putExtra(RESULT_EXTRA_DISTANCE_M, dist)
            putExtra(RESULT_EXTRA_ANCHOR_COUNT, anchors.size)
        }
        setResult(RESULT_OK, data)
        finish()
    }

    companion object {
        const val EXTRA_SHARP_ROOM_WIDTH_M = "extra_sharp_room_width_m"
        const val EXTRA_SHARP_ROOM_HEIGHT_M = "extra_sharp_room_height_m"
        const val EXTRA_SHARP_ROOM_DEPTH_M = "extra_sharp_room_depth_m"

        const val RESULT_EXTRA_DISTANCE_M = "ar_result_distance_m"
        const val RESULT_EXTRA_ANCHOR_COUNT = "ar_result_anchor_count"
    }
}
