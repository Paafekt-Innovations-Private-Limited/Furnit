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
import android.webkit.*
import android.widget.*
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.FileProvider
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
    private var plyPath: String? = null
    private var roomFolder: String? = null
    private var allowSave: Boolean = true

    // Gesture tracking
    private var lastTouchX = 0f
    private var lastTouchY = 0f
    private var isDragging = false

    // Joystick state
    private var joystickCenterX = 0f
    private var joystickCenterY = 0f
    private var joystickKnobX = 0f
    private var joystickKnobY = 0f
    private val joystickMaxOffset = 70f

    @SuppressLint("SetJavaScriptEnabled", "ClickableViewAccessibility")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        plyPath = intent.getStringExtra(EXTRA_PLY_PATH)
        roomFolder = intent.getStringExtra(EXTRA_ROOM_FOLDER)
        allowSave = intent.getBooleanExtra(EXTRA_ALLOW_SAVE, true)

        Log.d(TAG, "Opening SharpRoomActivity with PLY: $plyPath")

        if (plyPath == null) {
            Toast.makeText(this, "No PLY file provided", Toast.LENGTH_SHORT).show()
            finish()
            return
        }

        val rootLayout = FrameLayout(this)
        rootLayout.setBackgroundColor(Color.parseColor("#808080"))

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
            }

            // Add JavaScript interface for communication
            addJavascriptInterface(WebAppInterface(), "Android")
        }
        rootLayout.addView(webView, FrameLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.MATCH_PARENT
        ))

        // Gesture overlay for orbit controls
        val gestureOverlay = View(this).apply {
            setBackgroundColor(Color.TRANSPARENT)
            setOnTouchListener { _, event ->
                handleGesture(event)
                true
            }
        }
        rootLayout.addView(gestureOverlay, FrameLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.MATCH_PARENT
        ))

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

            // Title
            val title = TextView(this@SharpRoomActivity).apply {
                text = "3D Room View"
                textSize = 17f
                setTypeface(null, Typeface.BOLD)
                setTextColor(Color.WHITE)
                gravity = Gravity.CENTER
                layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
            }
            barContainer.addView(title)

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

            // Save/Share button (circle with upload icon) - only if allowed
            if (allowSave) {
                val saveBtn = TextView(this@SharpRoomActivity).apply {
                    text = "\u21E7" // Upload arrow
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
                    Toast.makeText(this@SharpRoomActivity, "AI Furniture Detection coming soon", Toast.LENGTH_SHORT).show()
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

                val line1 = TextView(this@SharpRoomActivity).apply {
                    text = "held vertically"
                    textSize = 12f
                    setTextColor(Color.WHITE)
                    gravity = Gravity.CENTER
                }
                addView(line1)

                val line2 = TextView(this@SharpRoomActivity).apply {
                    text = "Portrait"
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

    private fun handleGesture(event: MotionEvent): Boolean {
        when (event.actionMasked) {
            MotionEvent.ACTION_DOWN -> {
                lastTouchX = event.x
                lastTouchY = event.y
                isDragging = true
            }
            MotionEvent.ACTION_MOVE -> {
                if (isDragging) {
                    val deltaX = event.x - lastTouchX
                    val deltaY = event.y - lastTouchY
                    lastTouchX = event.x
                    lastTouchY = event.y

                    // Send orbit command to WebGL
                    webView.evaluateJavascript(
                        "if(typeof orbitCamera==='function')orbitCamera($deltaX, $deltaY);",
                        null
                    )
                }
            }
            MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                isDragging = false
            }
        }
        return true
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

        // Read PLY file and encode as base64 for inline loading
        val plyData = plyFile.readBytes()
        val plyBase64 = Base64.encodeToString(plyData, Base64.NO_WRAP)

        Log.d(TAG, "Loading PLY file: ${plyFile.name} (${plyData.size} bytes)")

        // Generate and load HTML
        val html = generateWebGLHTML(plyBase64)
        webView.loadDataWithBaseURL(
            "https://local/",
            html,
            "text/html",
            "UTF-8",
            null
        )
    }

    private fun generateWebGLHTML(plyBase64: String): String {
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
        }
        canvas {
            width: 100%;
            height: 100%;
            display: block;
        }
    </style>
    <script type="importmap">
    {
        "imports": {
            "three": "https://cdnjs.cloudflare.com/ajax/libs/three.js/0.170.0/three.module.min.js",
            "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.170.0/examples/jsm/",
            "@aspect/splat": "https://cdn.jsdelivr.net/npm/@aspect/splat@0.1.0/dist/splat.module.js"
        }
    }
    </script>
</head>
<body>
    <script type="module">
        import * as THREE from 'three';
        import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

        console.log('WebGL viewer initializing...');

        // Scene setup
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x808080);

        // Camera
        const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
        camera.position.set(0, 0, 5);
        camera.up.set(0, 1, 0);

        // Renderer
        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.setPixelRatio(window.devicePixelRatio);
        document.body.appendChild(renderer.domElement);

        // Orbit controls
        const controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        controls.rotateSpeed = 0.8;
        controls.target.set(0, 0, 0);

        // Initial camera position
        let initialCameraPosition = camera.position.clone();
        let initialControlsTarget = controls.target.clone();

        // Load PLY data from base64
        const plyBase64 = '$plyBase64';
        const binaryString = atob(plyBase64);
        const bytes = new Uint8Array(binaryString.length);
        for (let i = 0; i < binaryString.length; i++) {
            bytes[i] = binaryString.charCodeAt(i);
        }

        console.log('PLY data loaded:', bytes.length, 'bytes');

        // Parse PLY file
        function parsePLY(data) {
            const text = new TextDecoder().decode(data);
            const headerEnd = text.indexOf('end_header\n');
            if (headerEnd === -1) {
                console.error('Invalid PLY: no end_header');
                return null;
            }

            const header = text.substring(0, headerEnd);
            const vertexMatch = header.match(/element vertex (\d+)/);
            if (!vertexMatch) {
                console.error('Invalid PLY: no vertex count');
                return null;
            }

            const vertexCount = parseInt(vertexMatch[1]);
            console.log('Parsing', vertexCount, 'vertices');

            // Binary data starts after header
            const headerBytes = new TextEncoder().encode(text.substring(0, headerEnd + 11));
            const binaryStart = headerBytes.length;

            // Each vertex: 11 floats (44 bytes) + 3 bytes RGB = 47 bytes
            const bytesPerVertex = 47;
            const positions = [];
            const colors = [];

            const dataView = new DataView(data.buffer);

            for (let i = 0; i < vertexCount; i++) {
                const offset = binaryStart + i * bytesPerVertex;
                if (offset + bytesPerVertex > data.length) break;

                // Position
                const x = dataView.getFloat32(offset, true);
                const y = dataView.getFloat32(offset + 4, true);
                const z = dataView.getFloat32(offset + 8, true);

                positions.push(x, y, z);

                // Skip scale (12 bytes), rotation (16 bytes), opacity (4 bytes)
                // RGB at offset + 44
                const r = data[offset + 44] / 255;
                const g = data[offset + 45] / 255;
                const b = data[offset + 46] / 255;

                colors.push(r, g, b);
            }

            console.log('Parsed', positions.length / 3, 'vertices');
            return { positions, colors };
        }

        // Create point cloud from PLY with larger splats for better visualization
        const plyData = parsePLY(bytes);
        if (plyData) {
            const geometry = new THREE.BufferGeometry();
            geometry.setAttribute('position', new THREE.Float32BufferAttribute(plyData.positions, 3));
            geometry.setAttribute('color', new THREE.Float32BufferAttribute(plyData.colors, 3));

            // Use sprite-based rendering for softer splats
            const canvas = document.createElement('canvas');
            canvas.width = 64;
            canvas.height = 64;
            const ctx = canvas.getContext('2d');
            const gradient = ctx.createRadialGradient(32, 32, 0, 32, 32, 32);
            gradient.addColorStop(0, 'rgba(255,255,255,1)');
            gradient.addColorStop(0.3, 'rgba(255,255,255,0.8)');
            gradient.addColorStop(0.7, 'rgba(255,255,255,0.3)');
            gradient.addColorStop(1, 'rgba(255,255,255,0)');
            ctx.fillStyle = gradient;
            ctx.fillRect(0, 0, 64, 64);

            const splatTexture = new THREE.CanvasTexture(canvas);

            const material = new THREE.PointsMaterial({
                size: 0.12,
                vertexColors: true,
                sizeAttenuation: true,
                map: splatTexture,
                transparent: true,
                alphaTest: 0.01,
                depthWrite: false,
                blending: THREE.AdditiveBlending
            });

            const points = new THREE.Points(geometry, material);
            scene.add(points);

            // Calculate bounds and center camera
            geometry.computeBoundingBox();
            const box = geometry.boundingBox;
            const center = new THREE.Vector3();
            box.getCenter(center);

            const size = new THREE.Vector3();
            box.getSize(size);
            const maxDim = Math.max(size.x, size.y, size.z);
            const fov = camera.fov * (Math.PI / 180);
            const cameraDistance = (maxDim / 2) / Math.tan(fov / 2) * 1.5;

            camera.position.set(center.x, center.y, center.z + cameraDistance);
            controls.target.copy(center);
            controls.update();

            // Save initial position
            initialCameraPosition.copy(camera.position);
            initialControlsTarget.copy(controls.target);

            console.log('Room centered at:', center);
            console.log('Camera at distance:', cameraDistance);

            // Notify Android that loading is complete
            if (window.Android) {
                window.Android.onLoaded();
            }
        }

        // Camera control functions (called from Android)
        window.orbitCamera = function(deltaX, deltaY) {
            const rotateSpeed = 0.005;
            const offset = new THREE.Vector3().subVectors(camera.position, controls.target);
            const spherical = new THREE.Spherical().setFromVector3(offset);

            spherical.theta -= deltaX * rotateSpeed;
            spherical.phi += deltaY * rotateSpeed;
            spherical.phi = Math.max(0.1, Math.min(Math.PI - 0.1, spherical.phi));

            offset.setFromSpherical(spherical);
            camera.position.copy(controls.target).add(offset);
            controls.update();
        };

        window.moveCamera = function(dx, dy) {
            const moveSpeed = 0.05;
            camera.position.x += dx * moveSpeed;
            camera.position.z -= dy * moveSpeed;
            controls.target.x += dx * moveSpeed;
            controls.target.z -= dy * moveSpeed;
        };

        window.recenterCamera = function() {
            camera.position.copy(initialCameraPosition);
            controls.target.copy(initialControlsTarget);
            controls.update();
        };

        // Resize handler
        window.addEventListener('resize', () => {
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        });

        // Animation loop
        function animate() {
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }
        animate();

        console.log('WebGL viewer ready');
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
            // Update metadata with user's name
            val metadataFile = File(folder, "metadata.txt")
            metadataFile.writeText("name=$name\ncreated=${System.currentTimeMillis()}\ntype=sharp")

            Toast.makeText(this, "Room '$name' saved!", Toast.LENGTH_SHORT).show()
            Log.d(TAG, "Room saved: $name at $folder")

            // Finish and return to home
            setResult(RESULT_OK)
            finish()

        } catch (e: Exception) {
            Log.e(TAG, "Failed to save room", e)
            Toast.makeText(this, "Failed to save: ${e.message}", Toast.LENGTH_SHORT).show()
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
