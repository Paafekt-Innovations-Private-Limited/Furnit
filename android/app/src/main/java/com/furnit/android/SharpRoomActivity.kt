package com.furnit.android

import android.Manifest
import android.annotation.SuppressLint
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.os.Bundle
import android.os.Environment
import com.furnit.android.utils.DebugLogger
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.view.ViewTreeObserver
import android.content.pm.ActivityInfo
import android.view.WindowManager
import android.webkit.*
import android.widget.*
import androidx.activity.result.contract.ActivityResultContracts
import com.furnit.android.ar.ArSupportChecker
import com.furnit.android.ar.FurnitureFitArCameraController
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import androidx.core.view.ViewCompat
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.WindowInsetsControllerCompat
import androidx.lifecycle.lifecycleScope
import androidx.webkit.WebViewAssetLoader
import com.furnit.android.utils.CrashReporter
import com.furnit.android.utils.RoomFolderMetadata
import com.furnit.android.utils.RoomYoloRatioCapture
import com.furnit.android.services.FurnitureFitManager
import com.furnit.android.services.RatioSegmentationParams
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.ByteArrayInputStream
import java.io.File
import java.io.FileOutputStream
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

/**
 * SharpRoomActivity - WebGL-based 3D Gaussian Splat viewer
 * (Matches Swift's SharpRoomView)
 *
 * Uses THREE.js and SparkJS to render PLY files in a WebView
 */
class SharpRoomActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "SharpRoomActivity"
        const val EXTRA_PLY_PATH = "ply_path"
        const val EXTRA_ROOM_FOLDER = "room_folder"
        const val EXTRA_ROOM_WIDTH = "room_width"
        const val EXTRA_ROOM_HEIGHT = "room_height"
        const val EXTRA_ROOM_DEPTH = "room_depth"
        const val EXTRA_ROOM_CENTER_X = "room_center_x"
        const val EXTRA_ROOM_CENTER_Y = "room_center_y"
        const val EXTRA_ROOM_CENTER_Z = "room_center_z"
        const val EXTRA_ALLOW_SAVE = "allow_save"
        /** True if the photo was taken with the wide-angle (0.5x) lens; used to adjust initial camera position. */
        const val EXTRA_PHOTO_WIDE_ANGLE = "photo_wide_angle"
    }

    private lateinit var webView: WebView
    private lateinit var loadingOverlay: FrameLayout
    private lateinit var brainProgressOverlay: FrameLayout
    private lateinit var brainDetectionOverlay: FrameLayout
    private lateinit var brainDetectionOverlayView: FurnitureFitOverlayView
    private lateinit var titleView: TextView
    private var plyPath: String? = null
    private var roomFolder: String? = null
    private var allowSave: Boolean = true

    // Room dimensions (from intent or JS-measured)
    private var roomWidth: Float = 4.0f
    private var roomHeight: Float = 3.0f
    private var roomDepth: Float = 4.5f
    private var roomCenterX: Float = 0f
    private var roomCenterY: Float = 0f
    private var roomCenterZ: Float = 0f
    /** Isotropic scale for displayed dims vs raw SHARP bbox (ARCore calibration). */
    private var arDisplayScale: Float = 1f
    private var photoOrientation: String = "portrait"
    /** True when the photo was taken with wide-angle (0.5x) lens; viewer camera position is adjusted for wider FOV. */
    private var photoWideAngle: Boolean = false
    private var hasSavedDimensions: Boolean = false  // True if dimensions were passed from saved room

    // Brain (SmartyPants) overlay: show progress in same Activity so room stays visible
    private var brainOverlayVisible = false
    private var furnitureFitManager: FurnitureFitManager? = null
    private var cameraProvider: ProcessCameraProvider? = null
    /** Brain flow: ARCore camera when AR-assisted sizing is on and supported. */
    private var brainArController: FurnitureFitArCameraController? = null
    /** [setContentView] root — used to insert/remove AR [GLSurfaceView] for brain mode. */
    private lateinit var sharpRoomContentRoot: FrameLayout
    private val cameraExecutor: ExecutorService = Executors.newSingleThreadExecutor()
    /** True while one frame is in inference; drop new frames so overlay shows current view when camera moves. */
    private val isBrainInferenceRunning = AtomicBoolean(false)
    /** Per-room YOLO height fractions for brain (SmartyPants) ROI; null if not on disk. */
    private var brainRatioParams: RatioSegmentationParams? = null
    /** Status bar inset top (set from window insets) so arrow overlay can sit below top bar in portrait and landscape. */
    private var statusBarInsetTop = 0
    private val cameraPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        DebugLogger.d(TAG, "Brain: camera permission result isGranted=$isGranted")
        if (isGranted) {
            showBrainProgressOverlay()
            startBrainDetection()
        } else {
            DebugLogger.d(TAG, "Brain: camera permission denied")
            Toast.makeText(this, getString(R.string.camera_permission_required), Toast.LENGTH_LONG).show()
        }
    }

    @SuppressLint("SetJavaScriptEnabled", "ClickableViewAccessibility")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Enable true edge-to-edge display (matching iOS ignoresSafeArea)
        WindowCompat.setDecorFitsSystemWindows(window, false)
        window.statusBarColor = Color.TRANSPARENT
        window.navigationBarColor = Color.TRANSPARENT

        WindowInsetsControllerCompat(window, window.decorView).let { controller ->
            controller.isAppearanceLightStatusBars = false
            controller.isAppearanceLightNavigationBars = false
        }

        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.P) {
            window.attributes.layoutInDisplayCutoutMode =
                WindowManager.LayoutParams.LAYOUT_IN_DISPLAY_CUTOUT_MODE_SHORT_EDGES
        }

        plyPath = intent.getStringExtra(EXTRA_PLY_PATH)
        roomFolder = intent.getStringExtra(EXTRA_ROOM_FOLDER)
        allowSave = intent.getBooleanExtra(EXTRA_ALLOW_SAVE, true)

        // Load saved dimensions and orientation from intent (if available)
        var savedWidth = intent.getFloatExtra(EXTRA_ROOM_WIDTH, 0f)
        var savedHeight = intent.getFloatExtra(EXTRA_ROOM_HEIGHT, 0f)
        var savedDepth = intent.getFloatExtra(EXTRA_ROOM_DEPTH, 4.5f)
        roomDepth = savedDepth
        roomCenterX = intent.getFloatExtra(EXTRA_ROOM_CENTER_X, 0f)
        roomCenterY = intent.getFloatExtra(EXTRA_ROOM_CENTER_Y, 0f)
        roomCenterZ = intent.getFloatExtra(EXTRA_ROOM_CENTER_Z, 0f)
        var rawOrientation = intent.getStringExtra("photo_orientation")?.trim()?.lowercase()
        photoWideAngle = intent.getBooleanExtra(EXTRA_PHOTO_WIDE_ANGLE, false)

        // Single disk source: room_meta.json (or legacy metadata.txt via RoomFolderMetadata). Same parser as home list.
        roomFolder?.let { folderPath ->
            val disk = RoomFolderMetadata.readFromFolder(File(folderPath))
            if (disk != null) {
                disk.roomWidth?.takeIf { it > 0f }?.let { savedWidth = it }
                disk.roomHeight?.takeIf { it > 0f }?.let { savedHeight = it }
                disk.roomDepth?.takeIf { it > 0f }?.let { roomDepth = it }
                disk.roomCenterX?.let { roomCenterX = it }
                disk.roomCenterY?.let { roomCenterY = it }
                disk.roomCenterZ?.let { roomCenterZ = it }
                rawOrientation = disk.normalizedOrientation()
                photoWideAngle = disk.photoWideAngle
                disk.arDisplayScale?.takeIf { it > 0f }?.let { arDisplayScale = it }
                DebugLogger.d(
                    TAG,
                    "RoomFolderMetadata: ${savedWidth}x${savedHeight}x${roomDepth} orientation=${disk.normalizedOrientation()} wide=$photoWideAngle arDisplayScale=$arDisplayScale"
                )
                if (disk.yoloFurnitureHeightFracByClass.isNotEmpty()) {
                    brainRatioParams = RatioSegmentationParams(
                        furnitureHeightFracByClass = disk.yoloFurnitureHeightFracByClass,
                        defaultTargetHeightFrac = 0.26f,
                    )
                    DebugLogger.d(TAG, "Brain ratio targets loaded: ${disk.yoloFurnitureHeightFracByClass.keys}")
                }
            }
        }

        roomFolder?.let { folderPath ->
            lifecycleScope.launch(Dispatchers.IO) {
                runCatching {
                    RoomYoloRatioCapture.captureIfMissing(applicationContext, File(folderPath))
                }.onFailure { e ->
                    DebugLogger.eDebugMode(TAG, "Room YOLO ratio capture failed", e)
                }
            }
        }

        photoOrientation = if (rawOrientation == "landscape") "landscape" else "portrait"

        // Lock orientation based on room's photo orientation (no auto-rotate)
        requestedOrientation = if (photoOrientation == "landscape") {
            ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE
        } else {
            ActivityInfo.SCREEN_ORIENTATION_PORTRAIT
        }

        // Use saved dimensions if provided, otherwise use defaults
        if (savedWidth > 0f && savedHeight > 0f) {
            roomWidth = savedWidth
            roomHeight = savedHeight
            hasSavedDimensions = true
        } else {
            roomWidth = 4.0f
            roomHeight = 3.0f
            hasSavedDimensions = false
        }

        DebugLogger.d(TAG, "Opening SharpRoomActivity with PLY: $plyPath, dims: ${roomWidth}x${roomHeight}x${roomDepth}, hasSaved: $hasSavedDimensions, photoOrientation: $photoOrientation, photoWideAngle: $photoWideAngle")
        DebugLogger.d(TAG, "SharpRoom intent roomWidth=$roomWidth roomHeight=$roomHeight roomDepth=$roomDepth isPortrait=${photoOrientation != "landscape"} wideAngle=$photoWideAngle")
        val isPortraitReceived = photoOrientation != "landscape"
        DebugLogger.d(TAG, "VIEWER_RECEIVED isPortrait=$isPortraitReceived roomWidth=$roomWidth roomHeight=$roomHeight roomDepth=$roomDepth path=$roomFolder")

        if (plyPath == null) {
            Toast.makeText(this, getString(R.string.sharp_room_no_ply), Toast.LENGTH_SHORT).show()
            finish()
            return
        }

        val rootLayout = FrameLayout(this)
        rootLayout.setBackgroundColor(Color.parseColor("#808080"))

        // Copy PLY file to internal files dir for WebViewAssetLoader
        val plyFile = File(plyPath!!)
        val internalPlyDir = File(filesDir, "webview_assets")
        internalPlyDir.mkdirs()
        val internalPlyFile = File(internalPlyDir, "room.ply")
        if (plyFile.exists()) {
            plyFile.copyTo(internalPlyFile, overwrite = true)
            DebugLogger.d(TAG, "Copied PLY to internal storage: ${internalPlyFile.absolutePath}")
        }

        // WebViewAssetLoader serves files from internal storage via https:// URL
        // This allows SparkJS fetch() to work properly
        val assetLoader = WebViewAssetLoader.Builder()
            .addPathHandler("/assets/", WebViewAssetLoader.AssetsPathHandler(this))
            .addPathHandler("/files/", WebViewAssetLoader.InternalStoragePathHandler(this, internalPlyDir))
            .build()

        // WebView for 3D rendering
        webView = WebView(this).apply {
            settings.javaScriptEnabled = true
            settings.domStorageEnabled = true
            settings.allowFileAccess = true
            settings.allowContentAccess = true
            settings.mixedContentMode = WebSettings.MIXED_CONTENT_ALWAYS_ALLOW
            settings.mediaPlaybackRequiresUserGesture = false
            setBackgroundColor(Color.TRANSPARENT)

            webViewClient = object : WebViewClient() {
                override fun onPageFinished(view: WebView?, url: String?) {
                    super.onPageFinished(view, url)
                    DebugLogger.d(TAG, "WebView page loaded")
                    // Hide loading after a delay for splat rendering
                    postDelayed({
                        loadingOverlay.visibility = View.GONE
                    }, 2000)
                }

                override fun onReceivedError(view: WebView?, request: WebResourceRequest?, error: WebResourceError?) {
                    // Don't log favicon as error (we intercept it; this is a fallback if something else fails)
                    if (request?.url?.toString()?.contains("favicon") == true) return
                    DebugLogger.eDebugMode(TAG, "WebView error: ${error?.description}")
                }

                // Use WebViewAssetLoader to serve files
                override fun shouldInterceptRequest(view: WebView?, request: WebResourceRequest?): WebResourceResponse? {
                    val url = request?.url ?: return null
                    val urlString = url.toString()
                    // Suppress favicon request so we don't get net::ERR_NAME_NOT_RESOLVED (matches iOS: no favicon error)
                    if (urlString.endsWith("/favicon.ico") || urlString.contains("favicon.ico")) {
                        return WebResourceResponse("image/png", null, ByteArrayInputStream(ByteArray(0)))
                    }
                    DebugLogger.d(TAG, "shouldInterceptRequest: $url")
                    return assetLoader.shouldInterceptRequest(url)
                }
            }

            // WebGL console → logcat only when Settings → Debug Mode is ON (DebugLogger)
            webChromeClient = object : WebChromeClient() {
                override fun onConsoleMessage(message: ConsoleMessage?): Boolean {
                    message?.let { m ->
                        DebugLogger.d(TAG, "JSConsole: ${m.message()} -- ${m.sourceId()}:${m.lineNumber()}")
                    }
                    return true
                }
            }

            // Add JavaScript interface for communication
            addJavascriptInterface(WebAppInterface(), "Android")
        }
        rootLayout.addView(webView, FrameLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.MATCH_PARENT
        ))

        // No gesture overlay - let WebView's OrbitControls handle all gestures
        // (rotation, zoom, pan) directly like iOS

        // Top bar
        val topBar = createTopBar()
        rootLayout.addView(topBar, FrameLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT
        ).apply { gravity = Gravity.TOP })

        // Bottom controls
        val bottomControls = createBottomControls()
        rootLayout.addView(bottomControls, FrameLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT
        ).apply { gravity = Gravity.BOTTOM })

        // Camera arrow overlay (up/down/left/right) — same as iOS
        val cameraArrowOverlay = createCameraArrowOverlay()
        rootLayout.addView(cameraArrowOverlay, FrameLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.MATCH_PARENT
        ).apply { gravity = Gravity.TOP or Gravity.START })

        // Loading overlay
        loadingOverlay = createLoadingOverlay()
        rootLayout.addView(loadingOverlay)

        // Brain progress overlay (on top; room stays visible underneath)
        brainProgressOverlay = createBrainProgressOverlay()
        brainProgressOverlay.visibility = View.GONE
        brainProgressOverlay.elevation = 20f
        rootLayout.addView(brainProgressOverlay)

        // Brain detection overlay: live segmentation on top of room (updates as you point at objects; Done to dismiss)
        brainDetectionOverlay = FrameLayout(this).apply {
            visibility = View.GONE
            elevation = 21f
            layoutParams = FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT
            )
        }
        brainDetectionOverlayView = FurnitureFitOverlayView(this).apply {
            layoutParams = FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT
            )
            onTouchOutsideFurniture = { ev -> webView.dispatchTouchEvent(ev) }
        }
        brainDetectionOverlay.addView(brainDetectionOverlayView)
        val doneBtn = TextView(this).apply {
            text = getString(R.string.common_done)
            setTextColor(Color.WHITE)
            setPadding(dpToPx(24), dpToPx(12), dpToPx(24), dpToPx(12))
            textSize = 16f
            setBackgroundColor(Color.parseColor("#80000000"))
            val lp = FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            )
            lp.gravity = Gravity.BOTTOM or Gravity.CENTER_HORIZONTAL
            lp.bottomMargin = dpToPx(80)
            layoutParams = lp
            setOnClickListener {
                hideBrainDetectionOverlay()
            }
        }
        brainDetectionOverlay.addView(doneBtn)
        rootLayout.addView(brainDetectionOverlay)

        setContentView(rootLayout)
        sharpRoomContentRoot = rootLayout

        // Apply status bar insets; position top bar below status bar and arrow overlay below top bar (portrait + landscape)
        ViewCompat.setOnApplyWindowInsetsListener(rootLayout) { _, insets ->
            val statusBar = insets.getInsets(WindowInsetsCompat.Type.statusBars())
            statusBarInsetTop = statusBar.top
            topBar.setPadding(
                topBar.paddingLeft,
                statusBarInsetTop,
                topBar.paddingRight,
                topBar.paddingBottom
            )
            updateCameraArrowOverlayTop(topBar, cameraArrowOverlay)
            cameraArrowOverlay.post { updateCameraArrowOverlayTop(topBar, cameraArrowOverlay) }
            topBar.viewTreeObserver.addOnGlobalLayoutListener(object : ViewTreeObserver.OnGlobalLayoutListener {
                override fun onGlobalLayout() {
                    updateCameraArrowOverlayTop(topBar, cameraArrowOverlay)
                }
            })
            insets
        }
        ViewCompat.requestApplyInsets(rootLayout)

        // Load the WebGL viewer
        loadWebGLViewer()
    }

    private fun dpToPx(dp: Int): Int {
        return (dp * resources.displayMetrics.density).toInt()
    }

    private fun effRoomWidth(): Float = roomWidth * arDisplayScale
    private fun effRoomHeight(): Float = roomHeight * arDisplayScale
    private fun effRoomDepth(): Float = roomDepth * arDisplayScale
    private fun effRoomCenterX(): Float = roomCenterX * arDisplayScale
    private fun effRoomCenterY(): Float = roomCenterY * arDisplayScale
    private fun effRoomCenterZ(): Float = roomCenterZ * arDisplayScale

    /** Position arrow overlay so it sits just below the top bar (works in portrait and landscape). */
    private fun updateCameraArrowOverlayTop(topBar: View, arrowOverlay: View) {
        val top = statusBarInsetTop + topBar.height
        arrowOverlay.setPadding(0, top, 0, 0)
    }

    private fun createTopBar(): FrameLayout {
        return FrameLayout(this).apply {
            setPadding(dpToPx(16), dpToPx(48), dpToPx(16), dpToPx(12))

            // Rounded dark background container
            val barContainer = LinearLayout(this@SharpRoomActivity).apply {
                orientation = LinearLayout.HORIZONTAL
                gravity = Gravity.CENTER_VERTICAL
                val bg = GradientDrawable().apply {
                    shape = GradientDrawable.RECTANGLE
                    cornerRadius = dpToPx(25).toFloat()
                    setColor(Color.parseColor("#1C1C1E"))
                }
                background = bg
                setPadding(dpToPx(8), dpToPx(8), dpToPx(8), dpToPx(8))
            }

            // Back button (circle with arrow)
            val backBtn = TextView(this@SharpRoomActivity).apply {
                text = "〈"
                textSize = 20f
                setTextColor(Color.WHITE)
                gravity = Gravity.CENTER
                val bg = GradientDrawable().apply {
                    shape = GradientDrawable.OVAL
                    setColor(Color.parseColor("#3A3A3C"))
                }
                background = bg
                val size = dpToPx(40)
                layoutParams = LinearLayout.LayoutParams(size, size)
                setOnClickListener { finish() }
            }
            barContainer.addView(backBtn)

            // Title with dimensions
            titleView = TextView(this@SharpRoomActivity).apply {
                text = String.format("%.1f × %.1f m", effRoomWidth(), effRoomHeight())
                textSize = 17f
                setTypeface(null, Typeface.BOLD)
                setTextColor(Color.WHITE)
                gravity = Gravity.CENTER
                layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
            }
            barContainer.addView(titleView)

            // Recenter button (circle with viewfinder icon)
            val recenterBtn = TextView(this@SharpRoomActivity).apply {
                text = "⌖"  // Viewfinder-like symbol
                textSize = 20f
                setTextColor(Color.WHITE)
                gravity = Gravity.CENTER
                val bg = GradientDrawable().apply {
                    shape = GradientDrawable.OVAL
                    setColor(Color.parseColor("#3A3A3C"))
                }
                background = bg
                val size = dpToPx(40)
                val params = LinearLayout.LayoutParams(size, size)
                params.setMargins(dpToPx(8), 0, 0, 0)
                layoutParams = params
                setOnClickListener { recenterCamera() }
            }
            barContainer.addView(recenterBtn)

            // Help button (circle with ?)
            val helpBtn = TextView(this@SharpRoomActivity).apply {
                text = "?"
                textSize = 18f
                setTextColor(Color.WHITE)
                gravity = Gravity.CENTER
                val bg = GradientDrawable().apply {
                    shape = GradientDrawable.OVAL
                    setColor(Color.parseColor("#3A3A3C"))
                }
                background = bg
                val size = dpToPx(40)
                val params = LinearLayout.LayoutParams(size, size)
                params.setMargins(dpToPx(8), 0, 0, 0)
                layoutParams = params
                setOnClickListener { showHelpDialog() }
            }
            barContainer.addView(helpBtn)

            // Save button (circle with upload icon) - only if allowed
            if (allowSave) {
                val saveBtn = TextView(this@SharpRoomActivity).apply {
                    text = "↓" // Download/save arrow (matching iOS square.and.arrow.down)
                    textSize = 20f
                    setTextColor(Color.WHITE)
                    gravity = Gravity.CENTER
                    val bg = GradientDrawable().apply {
                        shape = GradientDrawable.OVAL
                        setColor(Color.parseColor("#3A3A3C"))
                    }
                    background = bg
                    val size = dpToPx(40)
                    val params = LinearLayout.LayoutParams(size, size)
                    params.setMargins(dpToPx(8), 0, 0, 0)
                    layoutParams = params
                    setOnClickListener { showSaveDialog() }
                }
                barContainer.addView(saveBtn)
            }

            // Share button (circle with share icon)
            val shareBtn = TextView(this@SharpRoomActivity).apply {
                text = "↑" // Share arrow (matching iOS square.and.arrow.up)
                textSize = 20f
                setTextColor(Color.WHITE)
                gravity = Gravity.CENTER
                val bg = GradientDrawable().apply {
                    shape = GradientDrawable.OVAL
                    setColor(Color.parseColor("#3A3A3C"))
                }
                background = bg
                val size = dpToPx(40)
                val params = LinearLayout.LayoutParams(size, size)
                params.setMargins(dpToPx(8), 0, 0, 0)
                layoutParams = params
                setOnClickListener { sharePlyFile() }
            }
            barContainer.addView(shareBtn)

            addView(barContainer, FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            ))
        }
    }

    private fun showHelpDialog() {
        AlertDialog.Builder(this)
            .setTitle("3D Room Controls")
            .setMessage("• Drag to rotate view\n• Pinch to zoom\n• Two-finger drag to pan\n• Tap recenter button to reset view")
            .setPositiveButton("OK", null)
            .show()
    }

    private fun createBottomControls(): FrameLayout {
        return FrameLayout(this).apply {
            setPadding(dpToPx(20), 0, dpToPx(20), dpToPx(40))

            // Left: Brain/AI button + orientation helper text (like Swift)
            val leftBottomRow = LinearLayout(this@SharpRoomActivity).apply {
                orientation = LinearLayout.HORIZONTAL
                gravity = Gravity.CENTER_VERTICAL
                layoutParams = FrameLayout.LayoutParams(
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT
                ).apply {
                    gravity = Gravity.START or Gravity.BOTTOM
                    bottomMargin = dpToPx(20)
                }
            }
            val brainBtn = TextView(this@SharpRoomActivity).apply {
                text = "\uD83E\uDDE0" // Brain emoji
                textSize = 24f
                gravity = Gravity.CENTER
                val bg = GradientDrawable().apply {
                    shape = GradientDrawable.OVAL
                    setColor(Color.parseColor("#007AFF"))
                }
                background = bg
                val size = dpToPx(56)
                layoutParams = LinearLayout.LayoutParams(size, size)
                setOnClickListener {
                    val roomId = roomFolder?.let { File(it).name }
                    DebugLogger.d(TAG, "Brain click: ROOM_ID=$roomId ROOM_FOLDER=$roomFolder")
                    if (ContextCompat.checkSelfPermission(this@SharpRoomActivity, Manifest.permission.CAMERA)
                        != PackageManager.PERMISSION_GRANTED) {
                        DebugLogger.d(TAG, "Brain: requesting CAMERA permission")
                        cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
                    } else {
                        DebugLogger.d(TAG, "Brain: permission OK, showing progress and starting detection")
                        showBrainProgressOverlay()
                        startBrainDetection()
                    }
                }
            }
            leftBottomRow.addView(brainBtn)
            val orientationLabel = TextView(this@SharpRoomActivity).apply {
                text = if (photoOrientation == "landscape") getString(R.string.orientation_held_horizontally) else getString(R.string.orientation_held_vertically)
                setTextColor(Color.WHITE)
                setPadding(dpToPx(12), 0, 0, 0)
                textSize = 14f
                alpha = 0.9f
            }
            leftBottomRow.addView(orientationLabel)
            addView(leftBottomRow)

            // No joystick - use OrbitControls touch gestures for navigation (matching iOS)

            // Right: Camera/Screenshot button
            val cameraBtn = TextView(this@SharpRoomActivity).apply {
                text = "\uD83D\uDCF7" // Camera emoji
                textSize = 24f
                gravity = Gravity.CENTER
                val bg = GradientDrawable().apply {
                    shape = GradientDrawable.OVAL
                    setColor(Color.parseColor("#007AFF"))
                }
                background = bg
                val size = dpToPx(56)
                layoutParams = FrameLayout.LayoutParams(size, size).apply {
                    gravity = Gravity.END or Gravity.BOTTOM
                    bottomMargin = dpToPx(20)
                }
                setOnClickListener {
                    takeScreenshot()
                }
            }
            addView(cameraBtn)
        }
    }

    /** Camera move arrows (up/down/left/right) — matches iOS SharpRoomView. */
    private fun createCameraArrowOverlay(): FrameLayout {
        val paddingPx = dpToPx(12)
        val buttonSizePx = dpToPx(44)
        val arrowColor = Color.WHITE
        val circleBg = GradientDrawable().apply {
            shape = GradientDrawable.OVAL
            setColor(Color.argb(128, 0, 0, 0))
        }

        fun makeArrowButton(arrowChar: String, onClick: () -> Unit): TextView {
            return TextView(this).apply {
                text = arrowChar
                setTextColor(arrowColor)
                textSize = 20f
                setTypeface(null, Typeface.BOLD)
                gravity = Gravity.CENTER
                background = circleBg
                setPadding(0, 0, 0, 0)
                layoutParams = FrameLayout.LayoutParams(buttonSizePx, buttonSizePx)
                setOnClickListener { onClick() }
            }
        }

        val container = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            setPadding(paddingPx, paddingPx, paddingPx, paddingPx)
            gravity = Gravity.CENTER_VERTICAL
        }

        container.addView(makeArrowButton("\u2190") { runMoveCamera(-8.0, 0.0) }) // Left
        val upDownColumn = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dpToPx(8), 0, dpToPx(8), 0)
        }
        upDownColumn.addView(makeArrowButton("\u2191") { runMoveCameraUp(0.2) })  // Up
        val downBtn = makeArrowButton("\u2193") { runMoveCameraUp(-0.2) }  // Down
        downBtn.layoutParams = LinearLayout.LayoutParams(buttonSizePx, buttonSizePx).apply { topMargin = dpToPx(8) }
        upDownColumn.addView(downBtn)
        container.addView(upDownColumn)
        container.addView(makeArrowButton("\u2192") { runMoveCamera(8.0, 0.0) }) // Right

        return FrameLayout(this).apply {
            isClickable = false
            addView(container, FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            ).apply { gravity = Gravity.TOP or Gravity.START })
        }
    }

    private fun runMoveCamera(dx: Double, dy: Double) {
        val js = "if (typeof moveCamera === 'function') moveCamera($dx, $dy);"
        webView.evaluateJavascript(js, null)
    }

    private fun runMoveCameraUp(dy: Double) {
        val js = "if (typeof moveCameraUp === 'function') moveCameraUp($dy);"
        webView.evaluateJavascript(js, null)
    }

    private fun takeScreenshot() {
        try {
            // Capture WebView content
            val bitmap = Bitmap.createBitmap(webView.width, webView.height, Bitmap.Config.ARGB_8888)
            val canvas = Canvas(bitmap)
            webView.draw(canvas)

            // Save to Pictures folder
            val timeStamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())
            val fileName = "Room_$timeStamp.png"
            val picturesDir = getExternalFilesDir(Environment.DIRECTORY_PICTURES)
            val file = File(picturesDir, fileName)

            FileOutputStream(file).use { out ->
                bitmap.compress(Bitmap.CompressFormat.PNG, 100, out)
            }

            Toast.makeText(this, getString(R.string.sharp_room_screenshot_saved, fileName), Toast.LENGTH_SHORT).show()
            DebugLogger.d(TAG, "Screenshot saved: ${file.absolutePath}")

            // Share the screenshot
            val uri: android.net.Uri = FileProvider.getUriForFile(this, "${packageName}.fileprovider", file)
            val shareIntent = Intent(Intent.ACTION_SEND).apply {
                type = "image/png"
                putExtra(Intent.EXTRA_STREAM, uri)
                addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            }
            startActivity(Intent.createChooser(shareIntent, "Share Screenshot"))

        } catch (e: Exception) {
            DebugLogger.eDebugMode(TAG, "Failed to take screenshot", e)
            Toast.makeText(this, getString(R.string.sharp_room_screenshot_failed), Toast.LENGTH_SHORT).show()
        }
    }

    private fun createLoadingOverlay(): FrameLayout {
        return FrameLayout(this).apply {
            setBackgroundColor(Color.parseColor("#CC000000"))
            layoutParams = FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT
            )

            val content = LinearLayout(this@SharpRoomActivity).apply {
                orientation = LinearLayout.VERTICAL
                gravity = Gravity.CENTER
                setPadding(48, 48, 48, 48)
                setBackgroundColor(Color.parseColor("#F5F5F5"))

                val progress = ProgressBar(this@SharpRoomActivity).apply {
                    isIndeterminate = true
                }
                addView(progress)

                val text = TextView(this@SharpRoomActivity).apply {
                    text = getString(R.string.sharp_room_loading)
                    textSize = 16f
                    setTextColor(Color.parseColor("#333333"))
                    gravity = Gravity.CENTER
                    setPadding(0, 24, 0, 0)
                }
                addView(text)
            }

            addView(content, FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            ).apply { gravity = Gravity.CENTER })
        }
    }

    /** Progress overlay when brain is tapped: room stays visible underneath (semi-transparent). */
    private fun createBrainProgressOverlay(): FrameLayout {
        return FrameLayout(this).apply {
            setBackgroundColor(Color.parseColor("#80000000"))
            layoutParams = FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT
            )
            val content = LinearLayout(this@SharpRoomActivity).apply {
                orientation = LinearLayout.VERTICAL
                gravity = Gravity.CENTER_HORIZONTAL
                setBackgroundColor(Color.parseColor("#99000000"))
                setPadding(dpToPx(32), dpToPx(16), dpToPx(32), dpToPx(16))
                val lp = FrameLayout.LayoutParams(
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT
                )
                lp.gravity = Gravity.TOP or Gravity.CENTER_HORIZONTAL
                lp.topMargin = dpToPx(120)
                layoutParams = lp

                val label = TextView(this@SharpRoomActivity).apply {
                    text = getString(R.string.smartypants_detecting_furniture)
                    setTextColor(Color.WHITE)
                    textSize = 14f
                    setPadding(dpToPx(8), dpToPx(4), dpToPx(8), dpToPx(8))
                }
                addView(label)
                val progress = ProgressBar(this@SharpRoomActivity, null, android.R.attr.progressBarStyleHorizontal).apply {
                    layoutParams = LinearLayout.LayoutParams(dpToPx(250), ViewGroup.LayoutParams.WRAP_CONTENT)
                    max = 100
                    progress = 15
                    progressDrawable.colorFilter = android.graphics.PorterDuffColorFilter(0xFF4CAF50.toInt(), android.graphics.PorterDuff.Mode.SRC_IN)
                }
                addView(progress)
            }
            addView(content)
        }
    }

    private fun showBrainProgressOverlay() {
        brainOverlayVisible = true
        brainProgressOverlay.visibility = View.VISIBLE
    }

    private fun hideBrainProgressOverlay() {
        brainProgressOverlay.visibility = View.GONE
    }

    private fun showBrainDetectionOverlay(mask: Bitmap?, detections: List<DetectionResult>, inputSize: Int) {
        brainDetectionOverlayView.setMaskAndDetections(mask, detections, inputSize)
        brainDetectionOverlay.visibility = View.VISIBLE
    }

    private fun hideBrainDetectionOverlay() {
        DebugLogger.d(TAG, "Brain: hideBrainDetectionOverlay() - user Done or Back, stopping camera")
        brainOverlayVisible = false
        brainDetectionOverlay.visibility = View.GONE
        stopBrainDetection()
    }

    private fun startBrainDetection() {
        DebugLogger.d(TAG, "Brain: startBrainDetection() - initializing SmartyPants on IO thread")
        roomFolder?.let { path ->
            RoomFolderMetadata.readFromFolder(File(path))?.let { disk ->
                if (disk.yoloFurnitureHeightFracByClass.isNotEmpty()) {
                    brainRatioParams = RatioSegmentationParams(
                        furnitureHeightFracByClass = disk.yoloFurnitureHeightFracByClass,
                        defaultTargetHeightFrac = 0.26f,
                    )
                }
            }
        }
        lifecycleScope.launch {
            val manager = withContext(Dispatchers.IO) {
                val m = FurnitureFitManager(this@SharpRoomActivity)
                if (m.initializeAuto()) m else null
            }
            if (manager == null) {
                DebugLogger.eDebugMode(TAG, "Brain: SmartyPants failed to initialize")
                runOnUiThread {
                    hideBrainProgressOverlay()
                    Toast.makeText(this@SharpRoomActivity, getString(R.string.sharp_room_smartypants_failed), Toast.LENGTH_SHORT).show()
                }
                return@launch
            }
            DebugLogger.d(TAG, "Brain: SmartyPants OK, binding camera on UI thread")
            furnitureFitManager = manager
            runOnUiThread { bindBrainCamera(manager) }
        }
    }

    private fun shouldUseArBrainCamera(): Boolean {
        return FurnitureFitManager.isArAssistedFurnitureSizingEnabled(this) &&
            ArSupportChecker.isArCoreSupported(this)
    }

    @SuppressLint("UnsafeOptInUsageError")
    private fun bindBrainCamera(manager: FurnitureFitManager) {
        if (shouldUseArBrainCamera()) {
            bindBrainArCoreCamera(manager)
            return
        }
        DebugLogger.d(TAG, "Brain: bindBrainCamera() - getting ProcessCameraProvider")
        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener({
            val provider = providerFuture.get()
            cameraProvider = provider
            provider.unbindAll()
            DebugLogger.d(TAG, "Brain: building ImageAnalysis and binding to BACK_CAMERA")
            val analysis = ImageAnalysis.Builder()
                .setTargetResolution(android.util.Size(768, 768))
                // Match display so ImageProxy.rotationDegrees + toBitmapSafe() align mask with portrait/landscape UI
                .setTargetRotation(displayRotationForCameraX())
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
            var frameCount = 0
            val hasFirstResult = BooleanArray(1) { false }
            analysis.setAnalyzer(cameraExecutor) { imageProxy ->
                try {
                    val bitmap = imageProxy.toBitmapSafe() ?: return@setAnalyzer
                    // Only process one frame at a time; drop others so we show current view when camera moves (no "chair forever")
                    if (isBrainInferenceRunning.get()) {
                        return@setAnalyzer
                    }
                    isBrainInferenceRunning.set(true)
                    frameCount++
                    if (frameCount == 1 || frameCount % 30 == 0) {
                        DebugLogger.d(TAG, "Brain: analysis frame $frameCount (camera active)")
                    }
                    manager.segmentWithDetectionsAsync(bitmap, brainRatioParams) { result ->
                        runOnUiThread {
                            isBrainInferenceRunning.set(false)
                            if (!hasFirstResult[0]) {
                                hasFirstResult[0] = true
                                DebugLogger.d(TAG, "Brain: first result - hiding progress, showing detection overlay")
                                hideBrainProgressOverlay()
                                brainDetectionOverlay.visibility = View.VISIBLE
                            }
                            val mask = result?.mask
                            val dets = result?.detections ?: emptyList()
                            val size = result?.inputSize ?: 640
                            brainDetectionOverlayView.setMaskAndDetections(
                                mask,
                                dets,
                                size,
                                result?.autoRatioOverlayScale ?: 1f,
                            )
                        }
                    }
                } finally {
                    imageProxy.close()
                }
            }
            try {
                provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, analysis)
                DebugLogger.d(TAG, "Brain: camera bound successfully - live segmentation running")
            } catch (e: Exception) {
                DebugLogger.eDebugMode(TAG, "Brain camera bind failed", e)
                runOnUiThread {
                    hideBrainProgressOverlay()
                    Toast.makeText(this@SharpRoomActivity, getString(R.string.sharp_room_camera_error, e.message ?: ""), Toast.LENGTH_SHORT).show()
                    CrashReporter.report(this@SharpRoomActivity, e, "Sharp room brain / camera bind")
                }
            }
        }, ContextCompat.getMainExecutor(this))
    }

    @SuppressLint("UnsafeOptInUsageError")
    private fun bindBrainArCoreCamera(manager: FurnitureFitManager) {
        DebugLogger.d(TAG, "Brain: bindBrainArCoreCamera() - ARCore path")
        cameraProvider?.unbindAll()
        cameraProvider = null
        brainArController?.let { existing ->
            try {
                sharpRoomContentRoot.removeView(existing.glSurfaceView)
            } catch (_: Exception) { }
            existing.destroy()
        }
        val controller = FurnitureFitArCameraController(this, cameraExecutor)
        brainArController = controller
        controller.lockedPhotoOrientation = photoOrientation
        val lp = FrameLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.MATCH_PARENT,
        )
        sharpRoomContentRoot.addView(controller.glSurfaceView, 1, lp)
        controller.glSurfaceView.visibility = View.INVISIBLE

        val hasFirstResult = BooleanArray(1) { false }
        controller.shouldPostBitmapFrame = { !isBrainInferenceRunning.get() }
        controller.onBitmapFrame = arBitmap@{ bitmap ->
            if (isBrainInferenceRunning.get()) {
                return@arBitmap
            }
            isBrainInferenceRunning.set(true)
            manager.segmentWithDetectionsAsync(bitmap, brainRatioParams) { result ->
                runOnUiThread {
                    isBrainInferenceRunning.set(false)
                    if (!hasFirstResult[0]) {
                        hasFirstResult[0] = true
                        DebugLogger.d(TAG, "Brain: first result (ARCore) - hiding progress, showing detection overlay")
                        hideBrainProgressOverlay()
                        brainDetectionOverlay.visibility = View.VISIBLE
                    }
                    val mask = result?.mask
                    val dets = result?.detections ?: emptyList()
                    val size = result?.inputSize ?: 640
                    val ratioScale = result?.autoRatioOverlayScale ?: 1f
                    brainDetectionOverlayView.setMaskAndDetections(
                        mask,
                        dets,
                        size,
                        brainEffectiveOverlayScale(ratioScale),
                    )
                    if (mask != null && dets.isNotEmpty()) {
                        val det = dets.first()
                        val inp = size.coerceAtLeast(1).toFloat()
                        val scaleX = bitmap.width / inp
                        val scaleY = bitmap.height / inp
                        brainArController?.setBboxHint(
                            det.x * scaleX,
                            det.y * scaleY,
                            det.h * scaleY,
                            det.label,
                        )
                    } else {
                        brainArController?.clearBboxHint()
                    }
                }
            }
        }
        controller.onHostResume()
    }

    private fun brainEffectiveOverlayScale(ratioScale: Float): Float {
        val ar = brainArController
        return if (ar != null && ar.isArOverlayScaleValid()) {
            ar.getSmoothedArOverlayScale().coerceIn(0.25f, 4f)
        } else {
            ratioScale
        }
    }

    private fun stopBrainDetection() {
        DebugLogger.d(TAG, "Brain: stopBrainDetection() - unbinding camera / AR")
        brainArController?.let { c ->
            try {
                sharpRoomContentRoot.removeView(c.glSurfaceView)
            } catch (_: Exception) { }
            c.destroy()
        }
        brainArController = null
        try {
            cameraProvider?.unbindAll()
        } catch (_: Exception) { }
        cameraProvider = null
    }

    private fun loadWebGLViewer() {
        val plyFile = File(plyPath!!)
        if (!plyFile.exists()) {
            DebugLogger.eDebugMode(TAG, "PLY file not found: $plyPath")
            Toast.makeText(this, getString(R.string.sharp_room_ply_not_found), Toast.LENGTH_SHORT).show()
            finish()
            return
        }

        DebugLogger.d(TAG, "Loading PLY file: ${plyFile.name} (${plyFile.length()} bytes)")

        // Load HTML using WebViewAssetLoader base URL
        // SparkJS will fetch PLY from https://appassets.androidplatform.net/files/room.ply
        val html = generateWebGLHTML()
        webView.loadDataWithBaseURL(
            "https://appassets.androidplatform.net/",
            html,
            "text/html",
            "UTF-8",
            null
        )
    }

    private fun generateWebGLHTML(): String {
        // Check auto-orbit + debug logging (same keys as Settings / DebugLogger)
        val prefs = getSharedPreferences("furnit_prefs", MODE_PRIVATE)
        val autoOrbitEnabled = prefs.getBoolean("auto_orbit_enabled", false)
        val sharpJsBool = if (prefs.getBoolean("debug_mode", false)) "true" else "false"
        // Use isPortrait like iOS for consistency
        val isPortrait = photoOrientation != "landscape"
        DebugLogger.d(TAG, "[SharpRoom] Building WebView HTML: photoOrientation=$photoOrientation isPortrait=$isPortrait photoWideAngle=$photoWideAngle (this activity = PLY/splat room)")
        // Which end of the room slab we place the camera on Z for **landscape** (portrait overrides in JS — see below).
        // true = min-Z rail (camera at minZ - dist, target minZ). Needed for landscape: mesh uses Rx=0/Rz=0 (WebView
        // upside-down fix) so "front" maps to this side; matches current good landscape behavior.
        // Portrait uses the same Rx+Rz as iOS; frameFromWorldBox uses Swift SharpRoomView rule (front at maxZ).
        val webglEntranceMinZ = true
        val fallbackW = effRoomWidth().toDouble()
        val fallbackH = effRoomHeight().toDouble()
        val fallbackD = effRoomDepth().toDouble()
        val fallbackCx = effRoomCenterX().toDouble()
        val fallbackCy = effRoomCenterY().toDouble()
        val fallbackCz = effRoomCenterZ().toDouble()
        val usedWideLens = photoWideAngle

        // SparkJS implementation matching iOS exactly
        return """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>3D Room</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        html, body {
            width: 100%;
            height: 100%;
            overflow: hidden;
            background: #808080;
            touch-action: none;
            -webkit-touch-callout: none;
            -webkit-user-select: none;
        }
        canvas {
            width: 100%;
            height: 100%;
            display: block;
            touch-action: none;
        }
    </style>
    <script type="importmap">
    {
        "imports": {
            "three": "https://cdnjs.cloudflare.com/ajax/libs/three.js/0.170.0/three.module.min.js",
            "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.170.0/examples/jsm/",
            "@sparkjsdev/spark": "https://sparkjs.dev/releases/spark/0.1.10/spark.module.js"
        }
    }
    </script>
</head>
<body>
    <script type="module">
        import * as THREE from 'three';
        import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
        import { SplatMesh, SparkRenderer } from '@sparkjsdev/spark';

        const SHARP_ROOM_DEBUG = $sharpJsBool;
        const _sharpConsoleLog = console.log.bind(console);
        console.log = function() { if (SHARP_ROOM_DEBUG) _sharpConsoleLog.apply(console, arguments); };
        function sharpAndroidLog(msg) {
            if (SHARP_ROOM_DEBUG && window.Android && window.Android.log) window.Android.log(msg);
        }

        console.log('[WebGL] SparkJS Gaussian Splat viewer initializing...');
        // Orientation and fallback dimensions from Kotlin (module scope so autoFrameRoom can use them)
        const isPortrait = $isPortrait;
        const entranceUseMinZ = $webglEntranceMinZ;
        const usedWideLens = $usedWideLens;
        const fallbackRoomWidth = $fallbackW;
        const fallbackRoomHeight = $fallbackH;
        const fallbackRoomDepth = $fallbackD;
        const fallbackRoomCenterX = $fallbackCx;
        const fallbackRoomCenterY = $fallbackCy;
        const fallbackRoomCenterZ = $fallbackCz;
        console.log('[SharpRoom] orientation: ' + (isPortrait ? 'portrait' : 'landscape') + ' (isPortrait=' + isPortrait + '), wideAngle(0.5x): ' + usedWideLens + ', fallbackDims: ' + fallbackRoomWidth.toFixed(2) + 'x' + fallbackRoomHeight.toFixed(2) + 'x' + fallbackRoomDepth.toFixed(2) + ', center: ' + fallbackRoomCenterX.toFixed(2) + ',' + fallbackRoomCenterY.toFixed(2) + ',' + fallbackRoomCenterZ.toFixed(2));

        // Scene setup (matching iOS exactly)
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x808080);

        // Camera — landscape infinite pinch: smaller near (see iOS INFINITE_ZOOM) so close dollying still draws; portrait keeps 0.1.
        const cameraNear = isPortrait ? 0.1 : 0.001;
        const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, cameraNear, 1000);
        camera.position.set(0, 0, 5);
        camera.up.set(0, 1, 0);

        // THREE.js Renderer (antialias: false per SparkJS docs)
        const renderer = new THREE.WebGLRenderer({ antialias: false });
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.setPixelRatio(window.devicePixelRatio);
        document.body.appendChild(renderer.domElement);

        // SparkRenderer with settings matching iOS exactly
        const spark = new SparkRenderer({
            renderer: renderer,
            maxStdDev: 3.0,
            preBlurAmount: 0.5,
            blurAmount: 0.3,
            falloff: 0.8,
            focalAdjustment: 1.5
        });
        camera.add(spark);

        // Orbit controls (low sensitivity: higher damping = less oscillation, low rotateSpeed = slow, comfortable drag)
        const controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.25;   // Settle quickly so orbit does not oscillate
        controls.rotateSpeed = 0.25;     // Slow rotation for touch so room does not move too fast
        controls.screenSpacePanning = false;
        // Portrait: small minDistance + capped max (room stays framed). Landscape: infinite dolly — minDistance 0 lets
        // pinch-zoom pass through the target / walls along the view axis (matches iOS INFINITE_ZOOM idea).
        const roomMaxDim = Math.max(fallbackRoomWidth, fallbackRoomHeight, fallbackRoomDepth);
        if (isPortrait) {
            controls.minDistance = 0.001;
            controls.maxDistance = Math.max(6, Math.min(25, roomMaxDim * 2.5));
        } else {
            controls.minDistance = 0;
            controls.maxDistance = 1e6;
            controls.zoomSpeed = 2.0;
            // Built-in OrbitControls dolly only moves camera toward a fixed target → stalls ~0.02m and cannot pass
            // through walls. Same as iOS INFINITE_ZOOM zoomCamera: move camera + target together along view ray.
            controls.enableZoom = false;
        }
        controls.target.set(0, 0, 0);
        controls.minAzimuthAngle = -Infinity;
        controls.maxAzimuthAngle = Infinity;
        // Portrait: narrow polar cone (comfortable tilt). Landscape: full 0..π — tight min/max polar was still clamping
        // spherical phi when dollying through the target (pinch “infinite” felt stuck at the wall).
        if (isPortrait) {
            controls.minPolarAngle = 0.001;
            controls.maxPolarAngle = Math.PI - 0.001;
        } else {
            controls.minPolarAngle = 0;
            controls.maxPolarAngle = Math.PI;
        }

        let initialCameraPosition = camera.position.clone();
        let initialControlsTarget = controls.target.clone();
        let needsRender = true;

        // Auto-orbit settings (matches iOS)
        const OSCILLATION_ENABLED = $autoOrbitEnabled;
        let autoOrbitEnabled = OSCILLATION_ENABLED;
        let autoOrbitTime = 0;
        let autoOrbitBaseAngle = 0;
        let autoOrbitRadius = 0;

        // Warm-up rendering (matches iOS - 5 seconds)
        const WARMUP_DURATION = 5000;
        const animationStartTime = performance.now();

        // Room dimensions (will be set by autoFrameRoom)
        let measuredRoomWidth = 4.0;
        let measuredRoomHeight = 3.0;
        let cameraFramedAt = 0;
        // Current room dims for benchmark log (set when framing; used when user stops moving)
        let currentRoomW = fallbackRoomWidth, currentRoomH = fallbackRoomHeight, currentRoomD = fallbackRoomDepth;
        /** Tightened bounds after autoFrame (matches iOS joystick / zoom clamping). */
        let roomBoundsForClamping = null;

        let benchmarkLogTimeout = null;
        controls.addEventListener('change', function() {
            needsRender = true;
            if (benchmarkLogTimeout) clearTimeout(benchmarkLogTimeout);
            benchmarkLogTimeout = setTimeout(function() {
                benchmarkLogTimeout = null;
                const p = camera.position;
                const t = controls.target;
                const dist = p.distanceTo(t);
                console.log('[BENCHMARK_CAMERA] portrait=' + (isPortrait ? 1 : 0) + ' posX=' + p.x.toFixed(4) + ' posY=' + p.y.toFixed(4) + ' posZ=' + p.z.toFixed(4) + ' tgtX=' + t.x.toFixed(4) + ' tgtY=' + t.y.toFixed(4) + ' tgtZ=' + t.z.toFixed(4) + ' distance=' + dist.toFixed(4) + ' roomW=' + currentRoomW.toFixed(4) + ' roomH=' + currentRoomH.toFixed(4) + ' roomD=' + currentRoomD.toFixed(4));
            }, 500);
        });

        if (!isPortrait) {
            const LANDSCAPE_DOLLY_SENS = 4.0;
            const LANDSCAPE_DOLLY_STEP = 0.22;
            const PINCH_THRESHOLD = 0.012;
            function landscapeDollyAlongView(scale) {
                if (typeof scale !== 'number' || !isFinite(scale) || scale <= 0) return;
                let forward = new THREE.Vector3().subVectors(controls.target, camera.position);
                if (forward.lengthSq() < 1e-14) {
                    camera.getWorldDirection(forward);
                    forward.negate();
                } else {
                    forward.normalize();
                }
                const step = (scale - 1) * LANDSCAPE_DOLLY_STEP * LANDSCAPE_DOLLY_SENS;
                camera.position.addScaledVector(forward, step);
                controls.target.addScaledVector(forward, step);
                controls.update();
                needsRender = true;
                autoOrbitEnabled = false;
                const dist = camera.position.distanceTo(controls.target);
                const line = '[LANDSCAPE_DOLLY] scale=' + scale.toFixed(4) + ' step=' + step.toFixed(5) + ' dist=' + dist.toFixed(5) +
                    ' cam=' + camera.position.x.toFixed(3) + ',' + camera.position.y.toFixed(3) + ',' + camera.position.z.toFixed(3) +
                    ' tgt=' + controls.target.x.toFixed(3) + ',' + controls.target.y.toFixed(3) + ',' + controls.target.z.toFixed(3);
                _sharpConsoleLog(line);
                sharpAndroidLog(line);
            }
            let lastPinchDist = 0;
            const canvas = renderer.domElement;
            canvas.addEventListener('touchstart', function(ev) {
                if (ev.touches.length === 2) {
                    const dx = ev.touches[0].clientX - ev.touches[1].clientX;
                    const dy = ev.touches[0].clientY - ev.touches[1].clientY;
                    lastPinchDist = Math.hypot(dx, dy);
                }
            }, { passive: true });
            canvas.addEventListener('touchmove', function(ev) {
                if (ev.touches.length !== 2 || lastPinchDist <= 0) return;
                const dx = ev.touches[0].clientX - ev.touches[1].clientX;
                const dy = ev.touches[0].clientY - ev.touches[1].clientY;
                const dist = Math.hypot(dx, dy);
                const scale = dist / lastPinchDist;
                lastPinchDist = dist;
                if (Math.abs(scale - 1) < PINCH_THRESHOLD) return;
                ev.preventDefault();
                ev.stopPropagation();
                landscapeDollyAlongView(scale);
            }, { passive: false, capture: true });
            canvas.addEventListener('touchend', function(ev) {
                if (ev.touches.length < 2) lastPinchDist = 0;
            }, { passive: true });
            canvas.addEventListener('wheel', function(ev) {
                ev.preventDefault();
                const scale = ev.deltaY > 0 ? 1.07 : 0.93;
                landscapeDollyAlongView(scale);
            }, { passive: false });
        }

        let splatMesh = null;

        // Auto-frame when mesh has valid bounds (called from onLoad when PLY ready, or by polling)
        let frameAttempts = 0;
        const maxFrameAttempts = 150;  // 150 * 200ms = 30s for large PLY (e.g. 292MB)
        /** Stops duplicate work when multiple setTimeout(autoFrameRoom) chains run (onLoad + initial poll). */
        let framingComplete = false;

        // Load PLY using SparkJS SplatMesh (matching iOS exactly)
        // URL served by WebViewAssetLoader; onLoad runs when PLY is loaded and decoded
        const plyURL = 'https://appassets.androidplatform.net/files/room.ply';
        console.log('[WebGL] Loading splat from:', plyURL);

        try {
            splatMesh = new SplatMesh({
                url: plyURL,
                maxSh: 0,
                onLoad: function(mesh) {
                    console.log('[WebGL] SplatMesh onLoad — scheduling autoFrameRoom');
                    setTimeout(autoFrameRoom, 600);
                }
            });
            // Portrait: identity. Landscape: 180° about local Y only (fixes upside-down from Rx+Ry combo).
            scene.add(splatMesh);
            splatMesh.scale.set(1, 1, 1);
            if (isPortrait) {
                splatMesh.rotation.set(0, 0, 0);
            } else {
                splatMesh.rotation.set(0, Math.PI, 0);
            }
            console.log('[WebGL] SplatMesh: portrait identity; landscape Ry=π only');
            setTimeout(autoFrameRoom, 500);
        } catch (err) {
            console.error('[WebGL] Failed to create SplatMesh:', err);
        }

        function giveUpDefaultCamera() {
            framingComplete = true;
            camera.position.set(0, 0, 4);
            controls.target.set(0, 0, 0);
            controls.update();
            needsRender = true;
            if (window.Android) window.Android.onLoaded();
        }

        function tryMetadataFallbackBox() {
            const metaMax = Math.max(fallbackRoomWidth, fallbackRoomHeight, fallbackRoomDepth);
            if (metaMax < 0.05) return false;
            if (splatMesh) {
                splatMesh.position.set(0, 0, 0);
                splatMesh.updateMatrixWorld(true);
            }
            const cx = fallbackRoomCenterX, cy = fallbackRoomCenterY, cz = fallbackRoomCenterZ;
            const hw = fallbackRoomWidth * 0.5, hh = fallbackRoomHeight * 0.5, hd = fallbackRoomDepth * 0.5;
            const b = new THREE.Box3(
                new THREE.Vector3(cx - hw, cy - hh, cz - hd),
                new THREE.Vector3(cx + hw, cy + hh, cz + hd)
            );
            frameFromWorldBox(b, 'metadata_fallback');
            return true;
        }

        /**
         * ROOM: splatMesh position stays (0,0,0) + load-time rotation only — we do NOT slide the room for framing.
         * CAMERA: portrait → SharpRoomView.swift (outside maxZ, target maxZ). landscape → min-Z rail (WebView/identity mesh).
         * Zoom / step back along view: change distInFront (metres). Smaller = closer to wall; larger = farther in front.
         */
        function frameFromWorldBox(box, frameSource) {
            if (framingComplete) {
                sharpAndroidLog('[SharpRoom] frameFromWorldBox SKIPPED (framingComplete already true) source=' + frameSource);
                return;
            }
            framingComplete = true; // block duplicate onLoad + poll (recenter clears this first)

            const size = box.getSize(new THREE.Vector3());
            let roomWidth, roomHeight, roomDepth;
            if (isPortrait) {
                roomWidth = size.x;
                roomHeight = size.y;
            } else {
                roomWidth = size.y;
                roomHeight = size.x;
            }
            roomDepth = size.z;

            const maxRealisticWidth = isPortrait ? 5.0 : 8.0;
            const maxRealisticHeight = isPortrait ? 3.5 : 3.2;
            if (roomWidth > maxRealisticWidth) roomWidth = maxRealisticWidth;
            if (roomHeight > maxRealisticHeight) roomHeight = maxRealisticHeight;

            const rawMinX = box.min.x, rawMaxX = box.max.x;
            const rawMinY = box.min.y, rawMaxY = box.max.y;
            const rawMinZ = box.min.z, rawMaxZ = box.max.z;

            const fogFactor = 0.15;
            const shrinkX = roomWidth * fogFactor * 0.5;
            const shrinkY = roomHeight * fogFactor * 0.5;
            // Spark AABB can be paper-thin on Z after rotation (~9mm). Fixed 20mm back inset made minZ > maxZ.
            const zSpanRaw = Math.max(1e-6, rawMaxZ - rawMinZ);
            const shrinkZ = Math.min(roomDepth * fogFactor * 0.5, zSpanRaw * 0.2);
            const backWallInset = Math.min(0.02, zSpanRaw * 0.12);

            const minX = rawMinX + shrinkX;
            const maxX = rawMaxX - shrinkX;
            const minY = rawMinY + shrinkY;
            const maxY = rawMaxY - shrinkY;
            let minZ = rawMinZ + shrinkZ;
            let maxZ = rawMaxZ - backWallInset;
            if (minZ >= maxZ) {
                const midZ = (rawMinZ + rawMaxZ) * 0.5;
                const halfZ = zSpanRaw * 0.45;
                minZ = midZ - halfZ;
                maxZ = midZ + halfZ;
            }

            const innerCenterX = (minX + maxX) / 2;
            const innerCenterY = (minY + maxY) / 2;
            const innerCenterZ = (minZ + maxZ) / 2;

            roomBoundsForClamping = {
                minX, maxX, minY, maxY, minZ, maxZ,
                centerX: innerCenterX, centerY: innerCenterY, centerZ: innerCenterZ
            };

            currentRoomW = roomWidth;
            currentRoomH = roomHeight;
            currentRoomD = roomDepth;

            // Initial camera: thin Z slab (rotated splat AABB) needs distance from floor span, not 9mm depth.
            const thinZSlab = zSpanRaw < 0.08;
            let entranceZ, cameraZ;
            let wallSide;
            let distInFront;
            if (thinZSlab) {
                if (isPortrait) {
                    // Match SharpRoomView.swift when Z span is tiny: still put camera outside maxZ (front wall).
                    wallSide = 'thinZ_portrait_maxZ_swift';
                    const roomSpan = Math.max(roomWidth, roomHeight, fallbackRoomWidth, fallbackRoomHeight);
                    distInFront = Math.max(0.75, Math.min(2.0, 0.56 * roomSpan));
                    const frontWallZ = maxZ;
                    entranceZ = frontWallZ;
                    cameraZ = frontWallZ + distInFront;
                } else {
                    // Landscape: keep center-rail (look +Z); works with identity mesh rotation + minZ convention.
                    wallSide = 'thinZ_centerRail';
                    const roomSpan = Math.max(roomWidth, roomHeight, fallbackRoomWidth, fallbackRoomHeight);
                    distInFront = Math.max(0.75, Math.min(2.0, 0.56 * roomSpan));
                    const targetZ = innerCenterZ + Math.min(0.12, Math.max(zSpanRaw, 0.02));
                    entranceZ = targetZ;
                    cameraZ = targetZ - distInFront;
                }
            } else {
                const FRONT_DIST_K = 0.28;
                const FRONT_DIST_CAP = 1.2;
                const depthForStandoff = Math.max(roomDepth, fallbackRoomDepth, zSpanRaw, 0.15);
                const depthProduct = depthForStandoff * FRONT_DIST_K;
                distInFront = Math.max(0.012, Math.min(depthProduct, FRONT_DIST_CAP));
                if (isPortrait) {
                    // SharpRoomView.swift: frontWallZ = maxZ, camera at maxZ+dist, target maxZ (look into room -Z).
                    wallSide = 'maxZ_front_swift_portrait';
                    entranceZ = maxZ;
                    cameraZ = maxZ + distInFront;
                } else if (entranceUseMinZ) {
                    wallSide = 'minZ';
                    entranceZ = minZ;
                    cameraZ = minZ - distInFront;
                } else {
                    wallSide = 'maxZ';
                    entranceZ = maxZ;
                    cameraZ = maxZ + distInFront;
                }
            }
            const distDbg = '[SharpRoom_DIST] src=' + frameSource + ' thinZ=' + (thinZSlab ? 1 : 0) + ' sizeXYZ=' + size.x.toFixed(3) + ',' + size.y.toFixed(3) + ',' + size.z.toFixed(3) + ' zSpanRaw=' + zSpanRaw.toFixed(4) + ' distInFront=' + distInFront.toFixed(4) + ' minZ=' + minZ.toFixed(4) + ' maxZ=' + maxZ.toFixed(4) + ' innerZ=' + innerCenterZ.toFixed(4) + ' wallSide=' + wallSide + ' targetZ=' + entranceZ.toFixed(4) + ' cameraZ=' + cameraZ.toFixed(4);
            console.log(distDbg);
            sharpAndroidLog(distDbg);

            const newCamPos = new THREE.Vector3(innerCenterX, innerCenterY, cameraZ);
            const newTarget = new THREE.Vector3(innerCenterX, innerCenterY, entranceZ);

            camera.position.copy(newCamPos);
            controls.target.copy(newTarget);
            controls.update();

            initialCameraPosition.copy(camera.position);
            initialControlsTarget.copy(controls.target);

            autoOrbitRadius = camera.position.distanceTo(controls.target);
            autoOrbitBaseAngle = Math.atan2(
                camera.position.x - controls.target.x,
                camera.position.z - controls.target.z
            );

            measuredRoomWidth = roomWidth;
            measuredRoomHeight = roomHeight;

            const camPos = camera.position;
            const tgt = controls.target;
            const dist = camPos.distanceTo(tgt);
            console.log('[SharpRoom] CAMERA_FRAME wallSide=' + wallSide + ' source=' + frameSource + ' isPortrait=' + (isPortrait ? 1 : 0) +
                ' roomW=' + roomWidth.toFixed(2) + ' roomH=' + roomHeight.toFixed(2) + ' roomD=' + roomDepth.toFixed(2) +
                ' distance=' + dist.toFixed(2) + ' camPos=' + camPos.x.toFixed(2) + ',' + camPos.y.toFixed(2) + ',' + camPos.z.toFixed(2) +
                ' target=' + tgt.x.toFixed(2) + ',' + tgt.y.toFixed(2) + ',' + tgt.z.toFixed(2) +
                ' innerZ=' + innerCenterZ.toFixed(3) + ' minZ=' + minZ.toFixed(3) + ' maxZ=' + maxZ.toFixed(3));

            console.log('[WebGL] Camera framed wallSide=' + wallSide + ' dist=' + distInFront.toFixed(3));
            cameraFramedAt = performance.now();
            needsRender = true;

            function sendDimensionsToAndroid() {
                if (window.Android && window.Android.onDimensionsMeasured) {
                    window.Android.onDimensionsMeasured(measuredRoomWidth, measuredRoomHeight);
                    console.log('[WebGL] Sent dimensions to Android:', measuredRoomWidth.toFixed(2), 'x', measuredRoomHeight.toFixed(2));
                }
            }
            sendDimensionsToAndroid();
            setTimeout(sendDimensionsToAndroid, 500);
            setTimeout(sendDimensionsToAndroid, 1500);
            setTimeout(sendDimensionsToAndroid, 3000);

            if (window.Android) window.Android.onLoaded();
        }

        function autoFrameRoom(fromRecenter) {
            if (fromRecenter) {
                framingComplete = false;
                frameAttempts = 0;
            }
            if (framingComplete) return;
            frameAttempts++;
            console.log('[WebGL] autoFrameRoom() attempt:', frameAttempts, 'fromRecenter=', !!fromRecenter);

            if (!splatMesh) {
                if (frameAttempts < maxFrameAttempts) {
                    setTimeout(autoFrameRoom, 200);
                } else {
                    console.error('[WebGL] Gave up waiting for splatMesh');
                    tryMetadataFallbackBox() || giveUpDefaultCamera();
                }
                return;
            }

            // Measure PLY at identity position; frameFromWorldBox applies mesh translation for viewer-centric framing.
            splatMesh.position.set(0, 0, 0);
            splatMesh.updateMatrixWorld(true);

            const metaMaxDim = Math.max(fallbackRoomWidth, fallbackRoomHeight, fallbackRoomDepth);

            if (typeof splatMesh.isInitialized === 'boolean' && !splatMesh.isInitialized) {
                if (frameAttempts >= 10 && metaMaxDim > 0.05 && tryMetadataFallbackBox()) return;
                if (frameAttempts < maxFrameAttempts) {
                    console.log('[WebGL] SplatMesh not initialized yet, retry...');
                    setTimeout(autoFrameRoom, 200);
                } else {
                    tryMetadataFallbackBox() || giveUpDefaultCamera();
                }
                return;
            }

            splatMesh.updateMatrixWorld(true);
            let box;
            try {
                const localBox = splatMesh.getBoundingBox(true);
                box = localBox.clone().applyMatrix4(splatMesh.matrixWorld);
            } catch (e) {
                console.warn('[WebGL] getBoundingBox failed:', e);
                if (frameAttempts >= 8 && metaMaxDim > 0.05 && tryMetadataFallbackBox()) return;
                if (frameAttempts < maxFrameAttempts) {
                    setTimeout(autoFrameRoom, 300);
                } else {
                    tryMetadataFallbackBox() || giveUpDefaultCamera();
                }
                return;
            }

            const size = box.getSize(new THREE.Vector3());
            if (size.length() < 0.01) {
                if (frameAttempts >= 8 && metaMaxDim > 0.05 && tryMetadataFallbackBox()) return;
                if (frameAttempts < maxFrameAttempts) {
                    console.log('[WebGL] Box3 size near zero after getBoundingBox, retry...');
                    setTimeout(autoFrameRoom, 200);
                } else {
                    tryMetadataFallbackBox() || giveUpDefaultCamera();
                }
                return;
            }

            frameFromWorldBox(box, 'spark_getBoundingBox');
        }

        window.autoFrameRoom = autoFrameRoom;

        // Camera controls (called from Android)
        window.orbitCamera = function(deltaX, deltaY) {
            autoOrbitEnabled = false;
            const rotateSpeed = 0.002;   // Slow rotation so touch does not move room too fast
            const offset = new THREE.Vector3().subVectors(camera.position, controls.target);
            const spherical = new THREE.Spherical().setFromVector3(offset);
            spherical.theta -= deltaX * rotateSpeed;
            spherical.phi += deltaY * rotateSpeed;
            spherical.phi = Math.max(0.1, Math.min(Math.PI - 0.1, spherical.phi));
            offset.setFromSpherical(spherical);
            camera.position.copy(controls.target).add(offset);
            controls.update();
            needsRender = true;
        };

        // Joystick: same mapping as SharpRoomView.swift (dx → world X, dy → world Z) + room clamping
        window.moveCamera = function(dx, dy) {
            autoOrbitEnabled = false;
            const moveSpeed = 0.03;
            let newX = camera.position.x + dx * moveSpeed;
            let newZ = camera.position.z + dy * moveSpeed;
            if (roomBoundsForClamping) {
                const marginSide = 0.05;
                const marginBack = 0.02;
                newX = Math.max(roomBoundsForClamping.minX + marginSide,
                    Math.min(roomBoundsForClamping.maxX - marginSide, newX));
                // Portrait starts outside maxZ (Swift front-wall view); do not clamp Z down to maxZ (SharpRoomView.swift).
                if (isPortrait && camera.position.z > roomBoundsForClamping.maxZ) {
                    newZ = Math.max(roomBoundsForClamping.minZ + marginSide, newZ);
                } else {
                    newZ = Math.max(roomBoundsForClamping.minZ + marginSide,
                        Math.min(roomBoundsForClamping.maxZ - marginBack, newZ));
                }
            }
            const actualDx = newX - camera.position.x;
            const actualDz = newZ - camera.position.z;
            camera.position.x = newX;
            camera.position.z = newZ;
            controls.target.x += actualDx;
            controls.target.z += actualDz;
            controls.update();
            needsRender = true;
        };

        window.moveCameraUp = function(dy) {
            autoOrbitEnabled = false;
            if (typeof dy !== 'number' || !isFinite(dy)) return;
            camera.position.y += dy;
            controls.target.y += dy;
            if (roomBoundsForClamping) {
                const m = 0.05;
                camera.position.y = Math.max(roomBoundsForClamping.minY + m,
                    Math.min(roomBoundsForClamping.maxY - m, camera.position.y));
                controls.target.y = Math.max(roomBoundsForClamping.minY + m,
                    Math.min(roomBoundsForClamping.maxY - m, controls.target.y));
            }
            controls.update();
            needsRender = true;
        };

        window.recenterCamera = function() {
            if (typeof autoFrameRoom === 'function') {
                autoFrameRoom(true);
            } else {
                camera.position.copy(initialCameraPosition);
                controls.target.copy(initialControlsTarget);
                controls.update();
            }
            needsRender = true;
            setTimeout(function() { needsRender = true; }, 0);
            setTimeout(function() { needsRender = true; }, 50);
        };

        window.addEventListener('resize', () => {
            // Camera aspect matches window (activity is locked to portrait/landscape). Uniform scaling only—no independent X/Y scale.
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
            needsRender = true;
        });

        // Animation loop with warm-up and auto-orbit (matching iOS exactly)
        const clock = new THREE.Clock();
        let lastRenderTime = 0;
        const IDLE_FPS = 30;
        const IDLE_FRAME_TIME = 1000 / IDLE_FPS;

        function animate(currentTime) {
            requestAnimationFrame(animate);

            const dt = clock.getDelta();
            let shouldRender = needsRender;
            needsRender = false;

            // Warm-up period: always render for first 5 seconds
            const elapsed = performance.now() - animationStartTime;
            const inWarmup = elapsed < WARMUP_DURATION;
            if (inWarmup) {
                shouldRender = true;
            }

            // Auto-orbit when enabled and not interacting (skip for 500ms after framing so initial position is visible)
            const orbitCooldown = 500;
            if (autoOrbitEnabled && autoOrbitRadius > 0.1 && (performance.now() - cameraFramedAt) > orbitCooldown) {
                autoOrbitTime += dt;
                const speed = 0.35;
                const t = controls.target;

                if (isPortrait) {
                    const amplitude = Math.PI / 6;
                    const angle = autoOrbitBaseAngle + amplitude * Math.sin(autoOrbitTime * speed);
                    camera.position.x = t.x + autoOrbitRadius * Math.sin(angle);
                    camera.position.z = t.z + autoOrbitRadius * Math.cos(angle);
                } else {
                    const sweepAmount = autoOrbitRadius * 0.3 * Math.sin(autoOrbitTime * speed);
                    camera.position.x = initialCameraPosition.x + sweepAmount;
                    camera.position.z = initialCameraPosition.z;
                }

                if (currentTime - lastRenderTime >= IDLE_FRAME_TIME) {
                    shouldRender = true;
                }
            }

            if (!shouldRender && !inWarmup) {
                return;
            }

            lastRenderTime = currentTime;
            controls.update();

            // Use SparkRenderer's update method for optimized Gaussian rendering
            spark.update({ scene });
            renderer.render(scene, camera);
        }
        animate(0);

        console.log('[WebGL] SparkJS viewer ready');
    </script>
</body>
</html>
        """.trimIndent()
    }

    private fun showSaveDialog() {
        val input = EditText(this).apply {
            hint = "Enter room name"
            setPadding(48, 32, 48, 32)
        }

        AlertDialog.Builder(this)
            .setTitle("Save Room")
            .setMessage("Enter a name for your room")
            .setView(input)
            .setPositiveButton("Save") { _, _ ->
                val name = input.text.toString().ifEmpty { "AI Room" }
                saveRoom(name)
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun saveRoom(name: String) {
        val folder = roomFolder
        if (folder == null) {
            Toast.makeText(this, getString(R.string.sharp_room_cannot_save), Toast.LENGTH_SHORT).show()
            return
        }

        try {
            // Update metadata with user's name and dimensions
            val metadataFile = File(folder, "metadata.txt")
            val metadata = StringBuilder()
            metadata.append("name=$name\n")
            metadata.append("created=${System.currentTimeMillis()}\n")
            metadata.append("type=sharp\n")
            metadata.append("roomWidth=$roomWidth\n")
            metadata.append("roomHeight=$roomHeight\n")
            metadata.append("roomDepth=$roomDepth\n")
            metadata.append("roomCenterX=$roomCenterX\n")
            metadata.append("roomCenterY=$roomCenterY\n")
            metadata.append("roomCenterZ=$roomCenterZ\n")
            metadata.append("photoOrientation=${if (photoOrientation == "landscape") "landscape" else "portrait"}\n")
            metadata.append("photoWideAngle=$photoWideAngle\n")
            metadata.append("arDisplayScale=$arDisplayScale\n")
            metadataFile.writeText(metadata.toString())
            val folderFile = File(folder)
            val snapshotToWrite = RoomFolderMetadata.snapshotPreservingYoloFields(
                folderFile,
                RoomFolderMetadata.Snapshot(
                    name = name,
                    createdAt = System.currentTimeMillis(),
                    type = "sharp",
                    photoOrientation = if (photoOrientation == "landscape") "landscape" else "portrait",
                    photoWideAngle = photoWideAngle,
                    roomWidth = roomWidth,
                    roomHeight = roomHeight,
                    roomDepth = roomDepth,
                    roomCenterX = roomCenterX,
                    roomCenterY = roomCenterY,
                    roomCenterZ = roomCenterZ,
                    arDisplayScale = arDisplayScale,
                ),
            )
            RoomFolderMetadata.writeToFolder(folderFile, snapshotToWrite)

            Toast.makeText(this, getString(R.string.sharp_room_saved, name), Toast.LENGTH_SHORT).show()
            DebugLogger.d(TAG, "Room saved: $name at $folder with dims: ${roomWidth}x${roomHeight}x${roomDepth}")

            // Go to room list screen (same as GLBRoomActivity / ModelDetailActivity after save)
            val intent = Intent(this, ContentActivity::class.java)
            intent.flags = Intent.FLAG_ACTIVITY_CLEAR_TOP or Intent.FLAG_ACTIVITY_NEW_TASK
            startActivity(intent)
            finish()

        } catch (e: Exception) {
            DebugLogger.eDebugMode(TAG, "Failed to save room", e)
            Toast.makeText(this, getString(R.string.sharp_room_save_failed, e.message ?: ""), Toast.LENGTH_SHORT).show()
            CrashReporter.report(this, e, "Sharp room save")
        }
    }

    private fun recenterCamera() {
        webView.evaluateJavascript(
            "if(typeof recenterCamera==='function')recenterCamera();",
            null
        )
    }

    private fun sharePlyFile() {
        val plyFile = File(plyPath ?: return)
        if (!plyFile.exists()) {
            Toast.makeText(this, getString(R.string.sharp_room_ply_not_found), Toast.LENGTH_SHORT).show()
            return
        }

        try {
            val uri: android.net.Uri = FileProvider.getUriForFile(
                this,
                "${packageName}.fileprovider",
                plyFile
            )

            val shareIntent = Intent(Intent.ACTION_SEND).apply {
                type = "application/octet-stream"
                putExtra(Intent.EXTRA_STREAM, uri)
                putExtra(Intent.EXTRA_SUBJECT, "3D Room PLY File")
                putExtra(Intent.EXTRA_TEXT, "Check out this 3D room scan!")
                addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            }

            startActivity(Intent.createChooser(shareIntent, "Share PLY File"))
            DebugLogger.d(TAG, "Sharing PLY file: ${plyFile.name}")

        } catch (e: Exception) {
            DebugLogger.eDebugMode(TAG, "Failed to share PLY file", e)
            Toast.makeText(this, getString(R.string.sharp_room_share_failed, e.message ?: ""), Toast.LENGTH_SHORT).show()
            CrashReporter.report(this, e, "Sharp room share PLY")
        }
    }

    // JavaScript interface for communication from WebView
    inner class WebAppInterface {
        @JavascriptInterface
        fun onLoaded() {
            runOnUiThread {
                loadingOverlay.visibility = View.GONE
                DebugLogger.d(TAG, "WebGL viewer reported loaded")
            }
        }

        @JavascriptInterface
        fun onDimensionsMeasured(width: Float, height: Float) {
            runOnUiThread {
                // Only use JS-measured dimensions if no saved dimensions were provided
                if (!hasSavedDimensions) {
                    roomWidth = width
                    roomHeight = height
                    // Update title
                    titleView.text = String.format("%.1f × %.1f m", roomWidth, roomHeight)
                    DebugLogger.d(TAG, "WebGL dimensions measured (using): ${roomWidth}x${roomHeight}")
                } else {
                    DebugLogger.d(TAG, "WebGL dimensions measured (ignored, using saved): ${width}x${height}")
                }
            }
        }

        @JavascriptInterface
        fun log(message: String) {
            DebugLogger.d(TAG, "WebGL: $message")
        }
    }

    override fun onResume() {
        super.onResume()
        brainArController?.onHostResume()
    }

    override fun onPause() {
        brainArController?.onHostPause()
        super.onPause()
    }

    @Deprecated("Deprecated in Java")
    override fun onBackPressed() {
        if (brainDetectionOverlay.visibility == View.VISIBLE) {
            hideBrainDetectionOverlay()
            return
        }
        if (brainProgressOverlay.visibility == View.VISIBLE || brainDetectionOverlay.visibility == View.VISIBLE) {
            stopBrainDetection()
            hideBrainProgressOverlay()
            if (brainDetectionOverlay.visibility == View.VISIBLE) hideBrainDetectionOverlay()
            else brainOverlayVisible = false
            return
        }
        if (webView.canGoBack()) {
            webView.goBack()
        } else {
            super.onBackPressed()
        }
    }

    override fun onDestroy() {
        stopBrainDetection()
        if (!cameraExecutor.isShutdown) {
            cameraExecutor.shutdown()
        }
        furnitureFitManager?.close()
        webView.destroy()
        super.onDestroy()
    }
}

