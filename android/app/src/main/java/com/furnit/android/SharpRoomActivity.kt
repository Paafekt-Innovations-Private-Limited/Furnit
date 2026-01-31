package com.furnit.android

import android.annotation.SuppressLint
import android.content.Intent
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
import android.view.MotionEvent
import android.view.View
import android.view.ViewGroup
import android.content.pm.ActivityInfo
import android.view.WindowManager
import android.webkit.*
import android.widget.*
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.FileProvider
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsControllerCompat
import androidx.webkit.WebViewAssetLoader
import com.furnit.android.models.Model
import com.furnit.android.models.ModelManager
import java.io.File
import java.io.FileOutputStream
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlin.math.abs

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
        const val EXTRA_ALLOW_SAVE = "allow_save"
    }

    private lateinit var webView: WebView
    private lateinit var loadingOverlay: FrameLayout
    private lateinit var joystickView: View
    private lateinit var titleView: TextView
    private var plyPath: String? = null
    private var roomFolder: String? = null
    private var allowSave: Boolean = true

    // Room dimensions (from intent or JS-measured)
    private var roomWidth: Float = 4.0f
    private var roomHeight: Float = 3.0f
    private var roomDepth: Float = 4.5f
    private var photoOrientation: String = "portrait"
    private var hasSavedDimensions: Boolean = false  // True if dimensions were passed from saved room

    // Calibration state
    private var showCalibrationOverlay = false
    private var detectedFurnitureHeight: Float? = null

    // Joystick state
    private var joystickCenterX = 0f
    private var joystickCenterY = 0f
    private var joystickKnobX = 0f
    private var joystickKnobY = 0f
    private val joystickMaxOffset = 70f

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

        // Load saved dimensions from intent (if available)
        val savedWidth = intent.getFloatExtra(EXTRA_ROOM_WIDTH, 0f)
        val savedHeight = intent.getFloatExtra(EXTRA_ROOM_HEIGHT, 0f)
        roomDepth = intent.getFloatExtra(EXTRA_ROOM_DEPTH, 4.5f)
        photoOrientation = intent.getStringExtra("photo_orientation") ?: "portrait"

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

        Log.d(TAG, "Opening SharpRoomActivity with PLY: $plyPath, dims: ${roomWidth}x${roomHeight}x${roomDepth}, hasSaved: $hasSavedDimensions")

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
                    Log.e(TAG, "WebView error: ${error?.description}")
                }

                // Use WebViewAssetLoader to serve files
                override fun shouldInterceptRequest(view: WebView?, request: WebResourceRequest?): WebResourceResponse? {
                    val url = request?.url ?: return null
                    Log.d(TAG, "shouldInterceptRequest: $url")

                    // Let WebViewAssetLoader handle appassets.androidplatform.net URLs
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
            .setMessage("• Drag on screen to rotate view\n\n• Use joystick to walk around\n\n• Tap save icon to save your room")
            .setPositiveButton("OK", null)
            .show()
    }

    private fun createBottomControls(): FrameLayout {
        return FrameLayout(this).apply {
            setPadding(dpToPx(20), 0, dpToPx(20), dpToPx(40))

            // Left: Brain/AI button (FurnitureFit)
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
                layoutParams = FrameLayout.LayoutParams(size, size).apply {
                    gravity = Gravity.START or Gravity.BOTTOM
                    bottomMargin = dpToPx(20)
                }
                setOnClickListener {
                    // Launch FurnitureFitActivity for AI furniture detection
                    val intent = Intent(this@SharpRoomActivity, FurnitureFitActivity::class.java)
                    intent.putExtra("ROOM_ID", roomFolder?.let { File(it).name })
                    intent.putExtra("ROOM_NAME", "Sharp Room")
                    intent.putExtra("PHOTO_ORIENTATION", photoOrientation)
                    startActivity(intent)
                }
            }
            addView(brainBtn)

            // Center: Joystick container
            val joystickContainer = FrameLayout(this@SharpRoomActivity).apply {
                val size = dpToPx(120)
                layoutParams = FrameLayout.LayoutParams(size, size).apply {
                    gravity = Gravity.CENTER_HORIZONTAL or Gravity.BOTTOM
                    bottomMargin = dpToPx(20)
                }

                // Outer ring
                val outerRing = View(this@SharpRoomActivity).apply {
                    val bg = GradientDrawable().apply {
                        shape = GradientDrawable.OVAL
                        setColor(Color.parseColor("#40FFFFFF"))
                        setStroke(dpToPx(2), Color.parseColor("#60FFFFFF"))
                    }
                    background = bg
                }
                addView(outerRing, FrameLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.MATCH_PARENT
                ))

                // Joystick knob
                val knobSize = dpToPx(50)
                val knob = View(this@SharpRoomActivity).apply {
                    val bg = GradientDrawable().apply {
                        shape = GradientDrawable.OVAL
                        setColor(Color.parseColor("#CCFFFFFF"))
                    }
                    background = bg
                }
                val knobParams = FrameLayout.LayoutParams(knobSize, knobSize).apply {
                    gravity = Gravity.CENTER
                }
                addView(knob, knobParams)

                joystickView = knob

                // Touch handling
                setOnTouchListener { _, event ->
                    handleJoystick(event, this, knob, size.toFloat(), knobSize.toFloat())
                    true
                }

                post {
                    joystickCenterX = width / 2f
                    joystickCenterY = height / 2f
                }
            }
            addView(joystickContainer)

            // Center label (above joystick): Orientation info
            val orientationLabel = LinearLayout(this@SharpRoomActivity).apply {
                orientation = LinearLayout.VERTICAL
                gravity = Gravity.CENTER
                val bg = GradientDrawable().apply {
                    shape = GradientDrawable.RECTANGLE
                    cornerRadius = dpToPx(8).toFloat()
                    setColor(Color.parseColor("#80000000"))
                }
                background = bg
                setPadding(dpToPx(12), dpToPx(4), dpToPx(12), dpToPx(4))
                layoutParams = FrameLayout.LayoutParams(
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT
                ).apply {
                    gravity = Gravity.CENTER_HORIZONTAL or Gravity.BOTTOM
                    bottomMargin = dpToPx(160)
                }

                val isLandscape = photoOrientation == "landscape"
                val line1 = TextView(this@SharpRoomActivity).apply {
                    text = if (isLandscape) "held horizontally" else "held vertically"
                    textSize = 12f
                    setTextColor(Color.WHITE)
                    gravity = Gravity.CENTER
                }
                addView(line1)

                val line2 = TextView(this@SharpRoomActivity).apply {
                    text = if (isLandscape) "Landscape" else "Portrait"
                    textSize = 14f
                    setTypeface(null, Typeface.BOLD)
                    setTextColor(Color.WHITE)
                    gravity = Gravity.CENTER
                }
                addView(line2)
            }
            addView(orientationLabel)

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
            val uri = FileProvider.getUriForFile(this, "${packageName}.fileprovider", file)
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

    private fun handleJoystick(
        event: MotionEvent,
        container: View,
        knob: View,
        containerSize: Float,
        knobSize: Float
    ): Boolean {
        val centerX = containerSize / 2f
        val centerY = containerSize / 2f

        when (event.actionMasked) {
            MotionEvent.ACTION_DOWN, MotionEvent.ACTION_MOVE -> {
                var dx = event.x - centerX
                var dy = event.y - centerY

                // Clamp to max offset
                val distance = kotlin.math.sqrt(dx * dx + dy * dy)
                if (distance > joystickMaxOffset) {
                    val scale = joystickMaxOffset / distance
                    dx *= scale
                    dy *= scale
                }

                // Move knob
                knob.translationX = dx
                knob.translationY = dy

                // Send movement to WebGL (normalize to -1 to 1)
                val normalizedX = dx / joystickMaxOffset
                val normalizedY = -dy / joystickMaxOffset  // Invert Y

                webView.evaluateJavascript(
                    "if(typeof moveCamera==='function')moveCamera($normalizedX, $normalizedY);",
                    null
                )
            }
            MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                // Return knob to center
                knob.animate()
                    .translationX(0f)
                    .translationY(0f)
                    .setDuration(150)
                    .start()

                // Stop movement
                webView.evaluateJavascript(
                    "if(typeof moveCamera==='function')moveCamera(0, 0);",
                    null
                )
            }
        }
        return true
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

        // Orbit controls
        const controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        controls.rotateSpeed = 3.0;  // Fast rotation for touch
        controls.screenSpacePanning = false;
        controls.minDistance = 0.01;
        controls.maxDistance = 100;
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

        controls.addEventListener('change', function() {
            needsRender = true;
        });

        let splatMesh = null;

        // Load PLY using SparkJS SplatMesh (matching iOS exactly)
        // URL served by WebViewAssetLoader
        const plyURL = 'https://appassets.androidplatform.net/files/room.ply';
        console.log('[WebGL] Loading splat from:', plyURL);

        try {
            splatMesh = new SplatMesh({
                url: plyURL,
                maxSh: 0  // Disable spherical harmonics for cleaner look
            });
            scene.add(splatMesh);

            // Classic PLY rotation matching iOS: 180° X + 90° Z
            splatMesh.rotation.x = Math.PI;
            splatMesh.rotation.z = Math.PI / 2;
            console.log('[WebGL] SplatMesh: rotated 180° X + 90° Z');

            // Start auto-framing after delay for async load
            setTimeout(autoFrameRoom, 500);

        } catch (err) {
            console.error('[WebGL] Failed to create SplatMesh:', err);
        }

        // Auto-frame when loaded (matching iOS exactly)
        let frameAttempts = 0;
        const maxFrameAttempts = 50;

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

            const box = new THREE.Box3().setFromObject(splatMesh);
            const size = box.getSize(new THREE.Vector3());

            if (size.length() < 0.01) {
                if (frameAttempts < maxFrameAttempts) {
                    console.log('[WebGL] Box3 too small, waiting for splatMesh to load...');
                    setTimeout(autoFrameRoom, 200);
                } else {
                    console.error('[WebGL] Gave up - mesh has no geometry');
                    camera.position.set(0, 0, 5);
                    controls.target.set(0, 0, 0);
                    controls.update();
                    if (window.Android) {
                        window.Android.onLoaded();
                    }
                }
                return;
            }

            const center = box.getCenter(new THREE.Vector3());
            // After 90° Z rotation, X and Y axes are swapped
            let roomWidth = size.y;
            let roomHeight = size.x;
            let roomDepth = size.z;

            console.log('[WebGL] Box3 size:', roomWidth.toFixed(2), roomHeight.toFixed(2), roomDepth.toFixed(2));
            console.log('[WebGL] Box3 center:', center.x.toFixed(2), center.y.toFixed(2), center.z.toFixed(2));

            // Shrink bounds to ignore foggy outer 15% (matching iOS)
            const fogFactor = 0.15;
            const shrinkX = roomWidth * fogFactor * 0.5;
            const shrinkY = roomHeight * fogFactor * 0.5;
            const shrinkZ = roomDepth * fogFactor * 0.5;

            const innerCenterX = center.x;
            const innerCenterY = center.y;
            const innerCenterZ = center.z;

            const roomRadius = Math.max(roomWidth, roomHeight, roomDepth) * 0.5;

            // Position camera at back wall center looking at room center
            const cameraDistance = roomRadius * 1.5;
            camera.position.set(innerCenterX, innerCenterY, innerCenterZ + cameraDistance);
            controls.target.set(innerCenterX, innerCenterY, innerCenterZ);
            controls.update();

            initialCameraPosition.copy(camera.position);
            initialControlsTarget.copy(controls.target);

            // Setup auto-orbit parameters (matches iOS)
            autoOrbitRadius = camera.position.distanceTo(controls.target);
            autoOrbitBaseAngle = Math.atan2(
                camera.position.x - controls.target.x,
                camera.position.z - controls.target.z
            );

            // Store measured dimensions
            measuredRoomWidth = roomWidth;
            measuredRoomHeight = roomHeight;

            console.log('[WebGL] Camera positioned at distance:', cameraDistance);
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

        // Note: autoFrameRoom is called from loadSplat() after mesh loads

        // Camera controls (called from Android)
        window.orbitCamera = function(deltaX, deltaY) {
            const rotateSpeed = 0.015;  // Fast rotation for touch
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

            // Auto-orbit when enabled and not interacting
            if (autoOrbitEnabled && autoOrbitRadius > 0.1) {
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
            metadata.append("photoOrientation=$photoOrientation\n")
            metadataFile.writeText(metadata.toString())

            Toast.makeText(this, "Room '$name' saved!", Toast.LENGTH_SHORT).show()
            Log.d(TAG, "Room saved: $name at $folder with dims: ${roomWidth}x${roomHeight}x${roomDepth}")

            // Finish and return to home
            setResult(RESULT_OK)
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
            val uri = FileProvider.getUriForFile(
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

    override fun onBackPressed() {
        if (webView.canGoBack()) {
            webView.goBack()
        } else {
            super.onBackPressed()
        }
    }

    override fun onDestroy() {
        webView.destroy()
        super.onDestroy()
    }
}
