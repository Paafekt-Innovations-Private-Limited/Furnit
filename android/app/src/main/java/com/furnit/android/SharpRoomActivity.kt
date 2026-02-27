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
import android.util.Base64
import android.util.Log
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.content.pm.ActivityInfo
import android.view.WindowManager
import android.webkit.*
import android.widget.*
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsControllerCompat
import androidx.lifecycle.lifecycleScope
import androidx.webkit.WebViewAssetLoader
import com.furnit.android.models.Model
import com.furnit.android.models.ModelManager
import com.furnit.android.services.FurnitureFitManager
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
    private var photoOrientation: String = "portrait"
    /** True when the photo was taken with wide-angle (0.5x) lens; viewer camera position is adjusted for wider FOV. */
    private var photoWideAngle: Boolean = false
    private var hasSavedDimensions: Boolean = false  // True if dimensions were passed from saved room

    // Calibration state
    private var showCalibrationOverlay = false
    private var detectedFurnitureHeight: Float? = null

    // Brain (SmartyPants) overlay: show progress in same Activity so room stays visible
    private var brainOverlayVisible = false
    private var furnitureFitManager: FurnitureFitManager? = null
    private var cameraProvider: ProcessCameraProvider? = null
    private val cameraExecutor: ExecutorService = Executors.newSingleThreadExecutor()
    /** True while one frame is in inference; drop new frames so overlay shows current view when camera moves. */
    private val isBrainInferenceRunning = AtomicBoolean(false)
    private val cameraPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        Log.d(TAG, "Brain: camera permission result isGranted=$isGranted")
        if (isGranted) {
            showBrainProgressOverlay()
            startBrainDetection()
        } else {
            Log.d(TAG, "Brain: camera permission denied")
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

        // Fallback: if list didn't pass dimensions/orientation, read from room folder metadata.txt (e.g. old list or first open)
        if ((savedWidth <= 0f || savedHeight <= 0f) && roomFolder != null) {
            val metaFile = File(roomFolder, "metadata.txt")
            if (metaFile.exists()) {
                try {
                    metaFile.readLines().forEach { line ->
                        when {
                            line.startsWith("roomWidth=") -> savedWidth = line.substringAfter("roomWidth=").toFloatOrNull() ?: savedWidth
                            line.startsWith("roomHeight=") -> savedHeight = line.substringAfter("roomHeight=").toFloatOrNull() ?: savedHeight
                            line.startsWith("roomDepth=") -> {
                                roomDepth = line.substringAfter("roomDepth=").toFloatOrNull() ?: roomDepth
                            }
                            line.startsWith("roomCenterX=") -> roomCenterX = line.substringAfter("roomCenterX=").toFloatOrNull() ?: roomCenterX
                            line.startsWith("roomCenterY=") -> roomCenterY = line.substringAfter("roomCenterY=").toFloatOrNull() ?: roomCenterY
                            line.startsWith("roomCenterZ=") -> roomCenterZ = line.substringAfter("roomCenterZ=").toFloatOrNull() ?: roomCenterZ
                            line.startsWith("photoOrientation=") -> rawOrientation = line.substringAfter("photoOrientation=").trim().lowercase()
                        }
                    }
                    Log.d(TAG, "Loaded from metadata.txt: ${savedWidth}x${savedHeight}x${roomDepth} orientation=$rawOrientation")
                } catch (e: Exception) {
                    Log.w(TAG, "Failed to read metadata.txt", e)
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

        Log.d(TAG, "Opening SharpRoomActivity with PLY: $plyPath, dims: ${roomWidth}x${roomHeight}x${roomDepth}, hasSaved: $hasSavedDimensions, photoOrientation: $photoOrientation, photoWideAngle: $photoWideAngle")
        Log.d(TAG, "SharpRoom intent roomWidth=$roomWidth roomHeight=$roomHeight roomDepth=$roomDepth isPortrait=${photoOrientation != "landscape"} wideAngle=$photoWideAngle")
        val isPortraitReceived = photoOrientation != "landscape"
        Log.d(TAG, "VIEWER_RECEIVED isPortrait=$isPortraitReceived roomWidth=$roomWidth roomHeight=$roomHeight roomDepth=$roomDepth path=$roomFolder")

        if (plyPath == null) {
            Toast.makeText(this, "No PLY file provided", Toast.LENGTH_SHORT).show()
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
            Log.d(TAG, "Copied PLY to internal storage: ${internalPlyFile.absolutePath}")
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

            webChromeClient = object : WebChromeClient() {
                override fun onConsoleMessage(message: ConsoleMessage?): Boolean {
                    Log.d(TAG, "WebGL: ${message?.message()}")
                    return true
                }
            }

            webViewClient = object : WebViewClient() {
                override fun onPageFinished(view: WebView?, url: String?) {
                    super.onPageFinished(view, url)
                    Log.d(TAG, "WebView page loaded")
                    // Hide loading after a delay for splat rendering
                    postDelayed({
                        loadingOverlay.visibility = View.GONE
                    }, 2000)
                }

                override fun onReceivedError(view: WebView?, request: WebResourceRequest?, error: WebResourceError?) {
                    // Don't log favicon as error (we intercept it; this is a fallback if something else fails)
                    if (request?.url?.toString()?.contains("favicon") == true) return
                    Log.e(TAG, "WebView error: ${error?.description}")
                }

                // Use WebViewAssetLoader to serve files
                override fun shouldInterceptRequest(view: WebView?, request: WebResourceRequest?): WebResourceResponse? {
                    val url = request?.url ?: return null
                    val urlString = url.toString()
                    // Suppress favicon request so we don't get net::ERR_NAME_NOT_RESOLVED (matches iOS: no favicon error)
                    if (urlString.endsWith("/favicon.ico") || urlString.contains("favicon.ico")) {
                        return WebResourceResponse("image/png", null, ByteArrayInputStream(ByteArray(0)))
                    }
                    Log.d(TAG, "shouldInterceptRequest: $url")
                    return assetLoader.shouldInterceptRequest(url)
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
            text = "Done"
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

        // Load the WebGL viewer
        loadWebGLViewer()
    }

    private fun dpToPx(dp: Int): Int {
        return (dp * resources.displayMetrics.density).toInt()
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
                text = String.format("%.1f × %.1f m", roomWidth, roomHeight)
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
                    Log.d(TAG, "Brain click: ROOM_ID=$roomId ROOM_FOLDER=$roomFolder")
                    if (ContextCompat.checkSelfPermission(this@SharpRoomActivity, Manifest.permission.CAMERA)
                        != PackageManager.PERMISSION_GRANTED) {
                        Log.d(TAG, "Brain: requesting CAMERA permission")
                        cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
                    } else {
                        Log.d(TAG, "Brain: permission OK, showing progress and starting detection")
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

            Toast.makeText(this, "Screenshot saved: $fileName", Toast.LENGTH_SHORT).show()
            Log.d(TAG, "Screenshot saved: ${file.absolutePath}")

            // Share the screenshot
            val uri: android.net.Uri = FileProvider.getUriForFile(this, "${packageName}.fileprovider", file)
            val shareIntent = Intent(Intent.ACTION_SEND).apply {
                type = "image/png"
                putExtra(Intent.EXTRA_STREAM, uri)
                addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            }
            startActivity(Intent.createChooser(shareIntent, "Share Screenshot"))

        } catch (e: Exception) {
            Log.e(TAG, "Failed to take screenshot", e)
            Toast.makeText(this, "Failed to capture screenshot", Toast.LENGTH_SHORT).show()
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
                    text = "Loading 3D Room..."
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
        Log.d(TAG, "Brain: hideBrainDetectionOverlay() - user Done or Back, stopping camera")
        brainOverlayVisible = false
        brainDetectionOverlay.visibility = View.GONE
        stopBrainDetection()
    }

    private fun startBrainDetection() {
        Log.d(TAG, "Brain: startBrainDetection() - initializing SmartyPants on IO thread")
        lifecycleScope.launch {
            val manager = withContext(Dispatchers.IO) {
                val m = FurnitureFitManager(this@SharpRoomActivity)
                if (m.initializeAuto()) m else null
            }
            if (manager == null) {
                Log.e(TAG, "Brain: SmartyPants failed to initialize")
                runOnUiThread {
                    hideBrainProgressOverlay()
                    Toast.makeText(this@SharpRoomActivity, "SmartyPants failed to initialize", Toast.LENGTH_SHORT).show()
                }
                return@launch
            }
            Log.d(TAG, "Brain: SmartyPants OK, binding camera on UI thread")
            furnitureFitManager = manager
            runOnUiThread { bindBrainCamera(manager) }
        }
    }

    @SuppressLint("UnsafeOptInUsageError")
    private fun bindBrainCamera(manager: FurnitureFitManager) {
        Log.d(TAG, "Brain: bindBrainCamera() - getting ProcessCameraProvider")
        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener({
            val provider = providerFuture.get()
            cameraProvider = provider
            provider.unbindAll()
            Log.d(TAG, "Brain: building ImageAnalysis and binding to BACK_CAMERA")
            val analysis = ImageAnalysis.Builder()
                .setTargetResolution(android.util.Size(768, 768))
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
                        Log.d(TAG, "Brain: analysis frame $frameCount (camera active)")
                    }
                    manager.segmentWithDetectionsAsync(bitmap) { result ->
                        runOnUiThread {
                            isBrainInferenceRunning.set(false)
                            if (!hasFirstResult[0]) {
                                hasFirstResult[0] = true
                                Log.d(TAG, "Brain: first result - hiding progress, showing detection overlay")
                                hideBrainProgressOverlay()
                                brainDetectionOverlay.visibility = View.VISIBLE
                            }
                            val mask = result?.mask
                            val dets = result?.detections ?: emptyList()
                            val size = result?.inputSize ?: 640
                            brainDetectionOverlayView.setMaskAndDetections(mask, dets, size)
                        }
                    }
                } finally {
                    imageProxy.close()
                }
            }
            try {
                provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, analysis)
                Log.d(TAG, "Brain: camera bound successfully - live segmentation running")
            } catch (e: Exception) {
                Log.e(TAG, "Brain camera bind failed", e)
                runOnUiThread {
                    hideBrainProgressOverlay()
                    Toast.makeText(this@SharpRoomActivity, "Camera error: ${e.message}", Toast.LENGTH_SHORT).show()
                }
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun stopBrainDetection() {
        Log.d(TAG, "Brain: stopBrainDetection() - unbinding camera")
        try {
            cameraProvider?.unbindAll()
        } catch (_: Exception) { }
        cameraProvider = null
    }

    private fun loadWebGLViewer() {
        val plyFile = File(plyPath!!)
        if (!plyFile.exists()) {
            Log.e(TAG, "PLY file not found: $plyPath")
            Toast.makeText(this, "PLY file not found", Toast.LENGTH_SHORT).show()
            finish()
            return
        }

        Log.d(TAG, "Loading PLY file: ${plyFile.name} (${plyFile.length()} bytes)")

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
        // Check auto-orbit setting from SharedPreferences
        val prefs = getSharedPreferences("furnit_prefs", MODE_PRIVATE)
        val autoOrbitEnabled = prefs.getBoolean("auto_orbit_enabled", false)
        // Use isPortrait like iOS for consistency
        val isPortrait = photoOrientation != "landscape"
        Log.d(TAG, "[SharpRoom] Building WebView HTML: photoOrientation=$photoOrientation isPortrait=$isPortrait photoWideAngle=$photoWideAngle (this activity = PLY/splat room)")
        // Pass known room dimensions and bbox center so camera can be framed when Box3 is not yet valid (SparkJS mesh bounds update async)
        val fallbackW = roomWidth.toDouble()
        val fallbackH = roomHeight.toDouble()
        val fallbackD = roomDepth.toDouble()
        val fallbackCx = roomCenterX.toDouble()
        val fallbackCy = roomCenterY.toDouble()
        val fallbackCz = roomCenterZ.toDouble()
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

        console.log('[WebGL] SparkJS Gaussian Splat viewer initializing...');
        // Orientation and fallback dimensions from Kotlin (module scope so autoFrameRoom can use them)
        const isPortrait = $isPortrait;
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

        // Camera
        const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
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
        controls.minDistance = 0.01;
        // Limit zoom-out so the room stays a reasonable size (max ~2.5× largest room dimension, cap 6–25m)
        const roomMaxDim = Math.max(fallbackRoomWidth, fallbackRoomHeight, fallbackRoomDepth);
        controls.maxDistance = Math.max(6, Math.min(25, roomMaxDim * 2.5));
        controls.target.set(0, 0, 0);
        controls.minAzimuthAngle = -Infinity;
        controls.maxAzimuthAngle = Infinity;
        controls.minPolarAngle = 0.01;
        controls.maxPolarAngle = Math.PI - 0.01;

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

        let splatMesh = null;

        // Auto-frame when mesh has valid bounds (called from onLoad when PLY ready, or by polling)
        let frameAttempts = 0;
        const maxFrameAttempts = 150;  // 150 * 200ms = 30s for large PLY (e.g. 292MB)

        // Load PLY using SparkJS SplatMesh (matching iOS exactly)
        // URL served by WebViewAssetLoader; onLoad runs when PLY is loaded and decoded
        const plyURL = 'https://appassets.androidplatform.net/files/room.ply';
        console.log('[WebGL] Loading splat from:', plyURL);

        try {
            splatMesh = new SplatMesh({
                url: plyURL,
                maxSh: 0,  // Disable spherical harmonics for cleaner look
                onLoad: function(mesh) {
                    console.log('[WebGL] SplatMesh onLoad - PLY loaded, framing camera (delay 600ms for bounds)');
                    setTimeout(autoFrameRoom, 600);
                }
            });
            if (isPortrait) {
                splatMesh.rotation.y = Math.PI / 2;
                console.log('[WebGL] SplatMesh: portrait - applied 90° Y rotation so room aspect matches photo');
            } else {
                console.log('[WebGL] SplatMesh: landscape - no rotation');
            }
            scene.add(splatMesh);

            // Fallback: start polling in case onLoad is slow or not supported
            setTimeout(autoFrameRoom, 500);
        } catch (err) {
            console.error('[WebGL] Failed to create SplatMesh:', err);
        }

        function autoFrameRoom() {
            frameAttempts++;
            console.log('[WebGL] autoFrameRoom() called, attempt:', frameAttempts);

            if (!splatMesh) {
                if (frameAttempts < maxFrameAttempts) {
                    setTimeout(autoFrameRoom, 200);
                } else {
                    console.error('[WebGL] Gave up waiting for splatMesh');
                }
                return;
            }

            // Coordinate sync: ensure world matrix is updated (rotation applied) before Box3
            splatMesh.updateMatrixWorld(true);
            let box = new THREE.Box3().setFromObject(splatMesh);
            let size = box.getSize(new THREE.Vector3());

            if (size.length() < 0.01) {
                // Use Kotlin-provided dimensions when Box3 is not yet valid (e.g. SparkJS). Center mesh so benchmark formula is in room space.
                if (fallbackRoomWidth > 0.1 && fallbackRoomHeight > 0.1 && fallbackRoomDepth > 0.1) {
                    try {
                        // Portrait only: center mesh at origin so benchmark formula is in room space (mesh has 90° Y rotation).
                        // Landscape: do not move mesh — it was already correct before; centering broke the view (grey).
                        if (isPortrait) {
                            const cx = fallbackRoomCenterX, cy = fallbackRoomCenterY, cz = fallbackRoomCenterZ;
                            splatMesh.position.set(cz, -cy, -cx);
                            splatMesh.updateMatrixWorld(true);
                        }
                        // Portrait benchmark Feb 28: pos 0.114,-0.58,0 tgt -0.742,-0.58,0 dist=0.856. Landscape: posY=0.002*H posZ=-0.13*D tgtZ=-0.444*D.
                        const W = fallbackRoomWidth, H = fallbackRoomHeight, D = fallbackRoomDepth;
                        if (isPortrait) {
                            const P_CAM_D = 0.076, P_CAM_Y = -0.133, P_TGT_D = -0.494;
                            camera.position.set(P_CAM_D * D, P_CAM_Y * H, 0);
                            controls.target.set(P_TGT_D * D, P_CAM_Y * H, 0);
                        } else {
                            const L_CAM_X = 0, L_CAM_Y = 0.00207, L_CAM_Z = -0.130, L_TGT_Z = -0.444;
                            camera.position.set(L_CAM_X * W, L_CAM_Y * H, L_CAM_Z * D);
                            controls.target.set(L_CAM_X * W, 0, L_TGT_Z * D);
                        }
                        controls.update();
                        currentRoomW = fallbackRoomWidth;
                        currentRoomH = fallbackRoomHeight;
                        currentRoomD = fallbackRoomDepth;
                        initialCameraPosition.copy(camera.position);
                        initialControlsTarget.copy(controls.target);
                        const camPos = camera.position;
                        console.log('[SharpRoom] CAMERA_FRAME fallback=1 benchmark isPortrait=' + (isPortrait ? 1 : 0) + ' fallbackW=' + fallbackRoomWidth.toFixed(2) + ' fallbackH=' + fallbackRoomHeight.toFixed(2) + ' fallbackD=' + fallbackRoomDepth.toFixed(2) + ' camPos=' + camPos.x.toFixed(3) + ',' + camPos.y.toFixed(3) + ',' + camPos.z.toFixed(3));
                        if (window.Android) window.Android.onLoaded();
                    } catch (err) {
                        console.error('[SharpRoom] fallback camera error:', err);
                        camera.position.set(0, 0, 3);
                        controls.target.set(0, 0, 0);
                        controls.update();
                        if (window.Android) window.Android.onLoaded();
                    }
                    // Retry once mesh bounds are ready so we can use Box3 and center mesh (attempts 2–4)
                    if (frameAttempts <= 4) {
                        setTimeout(autoFrameRoom, 1200);
                    }
                    return;
                }
                if (frameAttempts < maxFrameAttempts) {
                    console.log('[WebGL] Box3 too small, waiting for splatMesh to load...');
                    setTimeout(autoFrameRoom, 200);
                } else {
                    console.error('[WebGL] Gave up - mesh has no geometry');
                    camera.position.set(0, 0, 5);
                    controls.target.set(0, 0, 0);
                    controls.update();
                    if (window.Android) window.Android.onLoaded();
                }
                return;
            }

            // Ultralytics: center mesh at origin so camera framing is consistent (model may output offset origin)
            const boxCenterBefore = box.getCenter(new THREE.Vector3());
            splatMesh.position.sub(boxCenterBefore);
            splatMesh.updateMatrixWorld(true);
            box = new THREE.Box3().setFromObject(splatMesh);
            size = box.getSize(new THREE.Vector3());
            const center = box.getCenter(new THREE.Vector3());

            // Box3 axis mapping: portrait has 90° Y rotation so depth is along X; landscape depth along Z.
            // Use same semantic: roomWidth, roomHeight, roomDepth; depthAxis = axis from back wall to front wall.
            let roomWidth, roomHeight, roomDepth;
            let depthAxisMin, depthAxisMax;  // same formula for both orientations (like portrait)
            if (isPortrait) {
                roomWidth = size.z;
                roomHeight = size.y;
                roomDepth = size.x;
                depthAxisMin = box.min.x;
                depthAxisMax = box.max.x;
            } else {
                roomWidth = size.x;
                roomHeight = size.y;
                roomDepth = size.z;
                depthAxisMin = box.min.z;
                depthAxisMax = box.max.z;
            }

            // Cap to realistic room dimensions (fog makes bounds too large) - matching iOS
            const maxRealisticWidth = isPortrait ? 5.0 : 8.0;
            const maxRealisticHeight = isPortrait ? 3.5 : 3.2;
            if (roomWidth > maxRealisticWidth) roomWidth = maxRealisticWidth;
            if (roomHeight > maxRealisticHeight) roomHeight = maxRealisticHeight;

            // Portrait benchmark Feb 28: pos 0.114,-0.58,0 tgt -0.742,-0.58,0 dist=0.856. Landscape from benchmarks.
            if (isPortrait) {
                const P_CAM_D = 0.076, P_CAM_Y = -0.133, P_TGT_D = -0.494;
                camera.position.set(center.x + P_CAM_D * roomDepth, center.y + P_CAM_Y * roomHeight, center.z);
                controls.target.set(center.x + P_TGT_D * roomDepth, center.y + P_CAM_Y * roomHeight, center.z);
            } else {
                const L_CAM_X = 0, L_CAM_Y = 0.00207, L_CAM_Z = -0.130, L_TGT_Z = -0.444;
                camera.position.set(center.x + L_CAM_X * roomWidth, center.y + L_CAM_Y * roomHeight, center.z + L_CAM_Z * roomDepth);
                controls.target.set(center.x + L_CAM_X * roomWidth, center.y, center.z + L_TGT_Z * roomDepth);
            }

            currentRoomW = roomWidth;
            currentRoomH = roomHeight;
            currentRoomD = roomDepth;

            // Single-line structured log for logcat (WebView console -> Log.d SharpRoomActivity)
            const camPos = camera.position;
            const tgt = controls.target;
            const dist = camera.position.distanceTo(controls.target);
            console.log('[SharpRoom] CAMERA_FRAME benchmark isPortrait=' + (isPortrait ? 1 : 0) + ' roomW=' + roomWidth.toFixed(2) + ' roomH=' + roomHeight.toFixed(2) + ' roomD=' + roomDepth.toFixed(2) + ' distance=' + dist.toFixed(2) + ' camPos=' + camPos.x.toFixed(2) + ',' + camPos.y.toFixed(2) + ',' + camPos.z.toFixed(2) + ' target=' + tgt.x.toFixed(2) + ',' + tgt.y.toFixed(2) + ',' + tgt.z.toFixed(2));
            // Essential: sync OrbitControls after manual position/target change (prevents override)
            controls.update();

            initialCameraPosition.copy(camera.position);
            initialControlsTarget.copy(controls.target);

            // Setup auto-orbit parameters (matches iOS)
            const cameraDistance = camera.position.distanceTo(controls.target);
            autoOrbitRadius = cameraDistance;
            autoOrbitBaseAngle = Math.atan2(
                camera.position.x - controls.target.x,
                camera.position.z - controls.target.z
            );

            // Store measured dimensions
            measuredRoomWidth = roomWidth;
            measuredRoomHeight = roomHeight;

            console.log('[WebGL] Camera positioned at distance:', cameraDistance.toFixed(2));
            cameraFramedAt = performance.now();
            needsRender = true;

            // Send dimensions to Android (multiple times to ensure delivery)
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

            if (window.Android) {
                window.Android.onLoaded();
            }
        }

        // Camera controls (called from Android)
        window.orbitCamera = function(deltaX, deltaY) {
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

        window.moveCamera = function(dx, dy) {
            const moveSpeed = 0.05;
            camera.position.x += dx * moveSpeed;
            camera.position.z -= dy * moveSpeed;
            controls.target.x += dx * moveSpeed;
            controls.target.z -= dy * moveSpeed;
            needsRender = true;
        };

        window.recenterCamera = function() {
            camera.position.copy(initialCameraPosition);
            controls.target.copy(initialControlsTarget);
            controls.update();
            needsRender = true;
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

                // Circular arc oscillation ±30°
                const amplitude = Math.PI / 6;
                const angle = autoOrbitBaseAngle + amplitude * Math.sin(autoOrbitTime * speed);

                camera.position.x = t.x + autoOrbitRadius * Math.sin(angle);
                camera.position.z = t.z + autoOrbitRadius * Math.cos(angle);

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
            Toast.makeText(this, "Cannot save: room folder not found", Toast.LENGTH_SHORT).show()
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
            metadataFile.writeText(metadata.toString())

            Toast.makeText(this, "Room '$name' saved!", Toast.LENGTH_SHORT).show()
            Log.d(TAG, "Room saved: $name at $folder with dims: ${roomWidth}x${roomHeight}x${roomDepth}")

            // Go to room list screen (same as GLBRoomActivity / ModelDetailActivity after save)
            val intent = Intent(this, ContentActivity::class.java)
            intent.flags = Intent.FLAG_ACTIVITY_CLEAR_TOP or Intent.FLAG_ACTIVITY_NEW_TASK
            startActivity(intent)
            finish()

        } catch (e: Exception) {
            Log.e(TAG, "Failed to save room", e)
            Toast.makeText(this, "Failed to save: ${e.message}", Toast.LENGTH_SHORT).show()
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
            Toast.makeText(this, "PLY file not found", Toast.LENGTH_SHORT).show()
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
            Log.d(TAG, "Sharing PLY file: ${plyFile.name}")

        } catch (e: Exception) {
            Log.e(TAG, "Failed to share PLY file", e)
            Toast.makeText(this, "Failed to share: ${e.message}", Toast.LENGTH_SHORT).show()
        }
    }

    // JavaScript interface for communication from WebView
    inner class WebAppInterface {
        @JavascriptInterface
        fun onLoaded() {
            runOnUiThread {
                loadingOverlay.visibility = View.GONE
                Log.d(TAG, "WebGL viewer reported loaded")
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
                    Log.d(TAG, "WebGL dimensions measured (using): ${roomWidth}x${roomHeight}")
                } else {
                    Log.d(TAG, "WebGL dimensions measured (ignored, using saved): ${width}x${height}")
                }
            }
        }

        @JavascriptInterface
        fun log(message: String) {
            Log.d(TAG, "WebGL: $message")
        }
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
