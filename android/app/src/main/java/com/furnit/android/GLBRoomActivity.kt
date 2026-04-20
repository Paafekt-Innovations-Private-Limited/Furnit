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
import com.furnit.android.utils.CrashReporter
import com.furnit.android.utils.LogUtil
import com.furnit.android.utils.RoomDisplayName
import com.furnit.android.utils.RoomFolderMetadata
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
import com.furnit.android.models.ModelManager
import java.io.File
import java.io.FileOutputStream
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * GLBRoomActivity - WebGL-based GLB/GLTF 3D room viewer
 * (Matches Swift's GLBRoomView exactly)
 *
 * Uses THREE.js and GLTFLoader to render GLB files in a WebView
 */
class GLBRoomActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "GLBRoomActivity"
        const val EXTRA_GLB_PATH = "glb_path"
        const val EXTRA_ROOM_NAME = "room_name"
        const val EXTRA_ROOM_ID = "room_id"
        const val EXTRA_ROOM_WIDTH = "room_width"
        const val EXTRA_ROOM_HEIGHT = "room_height"
        const val EXTRA_IS_PREVIEW = "is_preview"
        const val EXTRA_PHOTO_ORIENTATION = "photo_orientation"
    }

    private lateinit var webView: WebView
    private lateinit var loadingOverlay: FrameLayout
    private lateinit var titleView: TextView
    private var glbPath: String? = null
    private var roomName: String = "3D Room"
    private var roomId: String? = null
    private var isPreviewMode: Boolean = false
    private var photoOrientation: String = "portrait"

    // Room dimensions
    private var roomWidth: Float = 4.0f
    private var roomHeight: Float = 3.0f

    @SuppressLint("SetJavaScriptEnabled")
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

        glbPath = intent.getStringExtra(EXTRA_GLB_PATH)
        roomName = intent.getStringExtra(EXTRA_ROOM_NAME) ?: "3D Room"
        roomId = intent.getStringExtra(EXTRA_ROOM_ID)
        isPreviewMode = intent.getBooleanExtra(EXTRA_IS_PREVIEW, false)
        roomWidth = intent.getFloatExtra(EXTRA_ROOM_WIDTH, 4.0f)
        roomHeight = intent.getFloatExtra(EXTRA_ROOM_HEIGHT, 3.0f)
        photoOrientation = intent.getStringExtra(EXTRA_PHOTO_ORIENTATION) ?: "portrait"

        // Lock orientation based on room's photo orientation (no auto-rotate)
        requestedOrientation = if (photoOrientation == "landscape") {
            ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE
        } else {
            ActivityInfo.SCREEN_ORIENTATION_PORTRAIT
        }

        LogUtil.d(TAG, "Opening GLBRoomActivity - path: $glbPath, roomId: $roomId, preview: $isPreviewMode, orientation: $photoOrientation")

        if (glbPath == null) {
            Toast.makeText(this, getString(R.string.glb_room_no_file), Toast.LENGTH_SHORT).show()
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
            setBackgroundColor(Color.TRANSPARENT)

            webChromeClient = object : WebChromeClient() {
                override fun onConsoleMessage(message: ConsoleMessage?): Boolean {
                    LogUtil.d(TAG, "WebGL: ${message?.message()}")
                    return true
                }
            }

            webViewClient = object : WebViewClient() {
                override fun onPageFinished(view: WebView?, url: String?) {
                    super.onPageFinished(view, url)
                    LogUtil.d(TAG, "WebView page loaded")
                }

                override fun onReceivedError(view: WebView?, request: WebResourceRequest?, error: WebResourceError?) {
                    LogUtil.e(TAG, "WebView error: ${error?.description}")
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

            val barContainer = LinearLayout(this@GLBRoomActivity).apply {
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

            // Back button
            val backBtn = TextView(this@GLBRoomActivity).apply {
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
                setOnClickListener { handleBackNavigation() }
            }
            barContainer.addView(backBtn)

            // Title with dimensions
            titleView = TextView(this@GLBRoomActivity).apply {
                text = if (roomWidth > 0 && roomHeight > 0) {
                    String.format("%.1f × %.1f m", roomWidth, roomHeight)
                } else {
                    roomName
                }
                textSize = 17f
                setTypeface(null, Typeface.BOLD)
                setTextColor(Color.WHITE)
                gravity = Gravity.CENTER
                layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
            }
            barContainer.addView(titleView)

            // Recenter button
            val recenterBtn = TextView(this@GLBRoomActivity).apply {
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

            // Save button (only in preview mode)
            if (isPreviewMode) {
                val saveBtn = TextView(this@GLBRoomActivity).apply {
                    text = "↓"  // Download/save arrow (matching iOS square.and.arrow.down)
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

            val arBtn = TextView(this@GLBRoomActivity).apply {
                text = getString(R.string.model_viewer_ar)
                textSize = 13f
                setTypeface(null, Typeface.BOLD)
                setTextColor(Color.WHITE)
                gravity = Gravity.CENTER
                val bg = GradientDrawable().apply {
                    shape = GradientDrawable.RECTANGLE
                    cornerRadius = dpToPx(16).toFloat()
                    setColor(Color.parseColor("#3A3A3C"))
                }
                background = bg
                setPadding(dpToPx(14), dpToPx(8), dpToPx(14), dpToPx(8))
                setOnClickListener { openFurnitureFit(enableArAssistedSizing = true) }
            }
            addView(
                arBtn,
                FrameLayout.LayoutParams(
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                ).apply {
                    gravity = Gravity.END or Gravity.TOP
                    topMargin = dpToPx(68)
                },
            )
        }
    }

    private fun createBottomControls(): FrameLayout {
        return FrameLayout(this).apply {
            setPadding(dpToPx(20), 0, dpToPx(20), dpToPx(40))

            // Left: Brain/AI button
            val brainBtn = TextView(this@GLBRoomActivity).apply {
                text = "\uD83E\uDDE0"  // Brain emoji
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
                setOnClickListener { openFurnitureFit(enableArAssistedSizing = false) }
            }
            addView(brainBtn)

            // Center: Orientation label
            val orientationLabel = LinearLayout(this@GLBRoomActivity).apply {
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
                    bottomMargin = dpToPx(20)
                }

                val isLandscape = photoOrientation == "landscape"
                val line1 = TextView(this@GLBRoomActivity).apply {
                    text = if (isLandscape) "held horizontally" else "held vertically"
                    textSize = 12f
                    setTextColor(Color.WHITE)
                    gravity = Gravity.CENTER
                }
                addView(line1)

                val line2 = TextView(this@GLBRoomActivity).apply {
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
            val cameraBtn = TextView(this@GLBRoomActivity).apply {
                text = "\uD83D\uDCF7"  // Camera emoji
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
                setOnClickListener { takeScreenshot() }
            }
            addView(cameraBtn)
        }
    }

    private fun openFurnitureFit(enableArAssistedSizing: Boolean) {
        val roomFolder = intent.getStringExtra("ROOM_FOLDER") ?: glbPath?.let { path -> File(path).parent }
        LogUtil.d(
            TAG,
            "FurnitureFit launch: ROOM_ID=$roomId ROOM_FOLDER=$roomFolder arAssist=$enableArAssistedSizing",
        )
        val intent = Intent(this@GLBRoomActivity, FurnitureFitActivity::class.java)
        intent.putExtra("ROOM_ID", roomId)
        intent.putExtra("ROOM_NAME", roomName)
        roomFolder?.let { intent.putExtra("ROOM_FOLDER", it) }
        intent.putExtra("ROOM_WIDTH", roomWidth)
        intent.putExtra("ROOM_HEIGHT", roomHeight)
        intent.putExtra("PHOTO_ORIENTATION", photoOrientation)
        intent.putExtra(FurnitureFitActivity.EXTRA_ENABLE_AR_ASSISTED_SIZING, enableArAssistedSizing)
        startActivity(intent)
    }

    private fun createLoadingOverlay(): FrameLayout {
        return FrameLayout(this).apply {
            setBackgroundColor(Color.parseColor("#CC000000"))
            layoutParams = FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT
            )

            val content = LinearLayout(this@GLBRoomActivity).apply {
                orientation = LinearLayout.VERTICAL
                gravity = Gravity.CENTER
                setPadding(48, 48, 48, 48)
                setBackgroundColor(Color.parseColor("#F5F5F5"))

                val progress = ProgressBar(this@GLBRoomActivity).apply {
                    isIndeterminate = true
                }
                addView(progress)

                val text = TextView(this@GLBRoomActivity).apply {
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

    private fun recenterCamera() {
        webView.evaluateJavascript(
            "if(typeof recenterCamera==='function')recenterCamera();",
            null
        )
    }

    private fun loadWebGLViewer() {
        val glbFile = File(glbPath!!)
        if (!glbFile.exists()) {
            LogUtil.e(TAG, "GLB file not found: $glbPath")
            Toast.makeText(this, getString(R.string.glb_room_not_found), Toast.LENGTH_SHORT).show()
            finish()
            return
        }

        LogUtil.d(TAG, "Loading GLB file: ${glbFile.name} (${glbFile.length()} bytes)")

        // Read GLB and convert to base64
        val glbData = glbFile.readBytes()
        val base64GLB = Base64.encodeToString(glbData, Base64.NO_WRAP)

        val html = generateWebGLHTML(base64GLB)
        webView.loadDataWithBaseURL(null, html, "text/html", "UTF-8", null)
    }

    private fun generateWebGLHTML(base64GLB: String): String {
        val isPortrait = photoOrientation == "portrait"

        // Three.js GLB viewer matching iOS GLBRoomView exactly
        return """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <style>
        * { margin: 0; padding: 0; }
        html, body {
            width: 100%;
            height: 100%;
            overflow: hidden;
            background: #808080;
            touch-action: none;
        }
        canvas {
            display: block;
            width: 100%;
            height: 100%;
            touch-action: none;
        }
    </style>
</head>
<body>
    <script type="importmap">
    {
        "imports": {
            "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
            "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"
        }
    }
    </script>
    <script type="module">
        import * as THREE from 'three';
        import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
        import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

        const isPortrait = $isPortrait;

        console.log('[GLBViewer] Starting...');

        // Scene setup
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x808080);

        // Camera
        const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
        camera.position.set(0, 2, 5);

        // Renderer
        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.setPixelRatio(window.devicePixelRatio);
        document.body.appendChild(renderer.domElement);

        // Orbit controls - matching iOS settings exactly
        const controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        controls.rotateSpeed = 3.0;     // Fast rotation for touch
        controls.zoomSpeed = 2.5;       // Fast zoom
        controls.enableZoom = true;
        controls.enablePan = false;
        controls.minDistance = 0.5;
        controls.maxDistance = 20;

        let initialCameraPosition = null;
        let initialControlsTarget = null;

        // Camera orbit function (called from Android)
        window.orbitCamera = function(deltaX, deltaY) {
            const rotateSpeed = 0.012;  // Matching iOS
            const spherical = new THREE.Spherical();
            const offset = new THREE.Vector3();
            offset.copy(camera.position).sub(controls.target);
            spherical.setFromVector3(offset);

            spherical.theta -= deltaX * rotateSpeed;
            spherical.phi -= deltaY * rotateSpeed;
            spherical.phi = Math.max(0.1, Math.min(Math.PI - 0.1, spherical.phi));

            offset.setFromSpherical(spherical);
            camera.position.copy(controls.target).add(offset);
            camera.lookAt(controls.target);
        };

        // Recenter function
        window.recenterCamera = function() {
            if (initialCameraPosition && initialControlsTarget) {
                camera.position.copy(initialCameraPosition);
                controls.target.copy(initialControlsTarget);
                controls.update();
                console.log('[GLBViewer] Camera recentered');
            }
        };

        // Lighting
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        scene.add(ambientLight);

        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(5, 10, 5);
        scene.add(directionalLight);

        // Load GLB from base64
        const base64GLB = '$base64GLB';

        try {
            // Decode base64 to ArrayBuffer
            const binaryString = atob(base64GLB);
            const bytes = new Uint8Array(binaryString.length);
            for (let i = 0; i < binaryString.length; i++) {
                bytes[i] = binaryString.charCodeAt(i);
            }
            const arrayBuffer = bytes.buffer;

            console.log('[GLBViewer] GLB data size:', arrayBuffer.byteLength);

            // Load with GLTFLoader
            const loader = new GLTFLoader();
            loader.parse(arrayBuffer, '', function(gltf) {
                console.log('[GLBViewer] GLB loaded successfully');

                const model = gltf.scene;
                scene.add(model);

                // Get model bounds
                const box = new THREE.Box3().setFromObject(model);
                const center = box.getCenter(new THREE.Vector3());
                const size = box.getSize(new THREE.Vector3());

                console.log('[GLBViewer] Model bounds - center:', center, 'size:', size);

                // Center the model horizontally but keep floor at y=0
                model.position.x = -center.x;
                model.position.z = -center.z;
                model.position.y = -box.min.y;

                // Centered Android room view: camera in the middle of the room, looking straight at the front wall
                const roomWidth = size.x;
                const roomHeight = size.y;
                const roomDepth = size.z;
                const halfDepth = roomDepth * 0.5;
                const camX = 0;
                const camY = roomHeight * 0.5;
                const camZ = 0;
                const targetX = 0;
                const targetY = camY;
                const targetZ = -halfDepth;

                camera.position.set(camX, camY, camZ);
                controls.target.set(targetX, targetY, targetZ);
                controls.update();

                // Save initial camera state for recenter
                initialCameraPosition = camera.position.clone();
                initialControlsTarget = controls.target.clone();

                console.log('[GLBViewer] Room size:', roomWidth.toFixed(2), 'x', roomHeight.toFixed(2), 'x', roomDepth.toFixed(2));
                console.log('[GLBViewer] Camera (center/front-wall): (', camX.toFixed(2), ',', camY.toFixed(2), ',', camZ.toFixed(2), ') lookAt (0,', targetY.toFixed(2), ',', targetZ.toFixed(2), ')');

                // Notify Android that we're loaded
                if (window.Android) {
                    window.Android.onLoaded();
                }

            }, function(error) {
                console.error('[GLBViewer] GLB parse error:', error);
                if (window.Android) {
                    window.Android.onError('Failed to parse 3D model');
                }
            });

        } catch (error) {
            console.error('[GLBViewer] Error:', error);
            if (window.Android) {
                window.Android.onError(error.message || 'Failed to load 3D model');
            }
        }

        // Animation loop
        function animate() {
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }
        animate();

        // Handle resize
        window.addEventListener('resize', () => {
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        });

        console.log('[GLBViewer] Viewer ready');
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

        val dialog = AlertDialog.Builder(this)
            .setTitle("Save Room")
            .setMessage("Enter a name for your room")
            .setView(input)
            .setPositiveButton("Save", null)
            .setNegativeButton("Cancel", null)
            .create()
        dialog.setOnShowListener {
            dialog.getButton(AlertDialog.BUTTON_POSITIVE).setOnClickListener {
                val typedName = input.text.toString().trim()
                if (typedName.isNotEmpty() && !ModelManager.isRoomNameAvailable(this, typedName)) {
                    Toast.makeText(this, getString(R.string.home_room_name_duplicate), Toast.LENGTH_SHORT).show()
                    return@setOnClickListener
                }
                val name = if (typedName.isEmpty()) {
                    ModelManager.findAvailableRoomName(this, RoomDisplayName.myRoomWithTimestamp())
                } else {
                    typedName
                }
                saveRoom(name)
                dialog.dismiss()
            }
        }
        dialog.show()
    }

    private fun saveRoom(name: String) {
        val path = glbPath ?: return
        if (!ModelManager.isRoomNameAvailable(this, name)) {
            Toast.makeText(this, getString(R.string.home_room_name_duplicate), Toast.LENGTH_SHORT).show()
            return
        }

        try {
            val glbFile = File(path)
            val previewRoomFolder = glbFile.parentFile

            if (previewRoomFolder != null) {
                val roomsDir = File(filesDir, "rooms")
                roomsDir.mkdirs()

                val savedRoomFolder = File(roomsDir, previewRoomFolder.name)
                previewRoomFolder.copyRecursively(savedRoomFolder, overwrite = true)

                val metadataFile = File(savedRoomFolder, "metadata.txt")
                val metadata = StringBuilder()
                val createdAtMillis = System.currentTimeMillis()
                metadata.append("name=$name\n")
                metadata.append("created=$createdAtMillis\n")
                metadata.append("type=manual\n")
                metadata.append("roomWidth=$roomWidth\n")
                metadata.append("roomHeight=$roomHeight\n")
                metadata.append("photoOrientation=$photoOrientation\n")
                metadataFile.writeText(metadata.toString())
                val glbSnapshot = RoomFolderMetadata.snapshotPreservingYoloFields(
                    savedRoomFolder,
                    RoomFolderMetadata.Snapshot(
                        name = name,
                        createdAt = createdAtMillis,
                        type = "manual",
                        photoOrientation = if (photoOrientation == "landscape") "landscape" else "portrait",
                        photoWideAngle = false,
                        roomWidth = roomWidth,
                        roomHeight = roomHeight,
                    ),
                )
                RoomFolderMetadata.writeToFolder(savedRoomFolder, glbSnapshot)

                previewRoomFolder.parentFile?.deleteRecursively()

                Toast.makeText(this, getString(R.string.glb_room_saved, name), Toast.LENGTH_SHORT).show()
                LogUtil.d(TAG, "Room saved: $name at ${savedRoomFolder.absolutePath}")

                val intent = Intent(this, ContentActivity::class.java)
                intent.flags = Intent.FLAG_ACTIVITY_CLEAR_TOP or Intent.FLAG_ACTIVITY_NEW_TASK
                startActivity(intent)
                finish()
            }
        } catch (e: Exception) {
            LogUtil.e(TAG, "Failed to save room", e)
            Toast.makeText(this, getString(R.string.glb_room_error, e.message ?: ""), Toast.LENGTH_SHORT).show()
            CrashReporter.report(this, e, "GLB room — save room")
        }
    }

    private fun takeScreenshot() {
        try {
            val bitmap = Bitmap.createBitmap(webView.width, webView.height, Bitmap.Config.ARGB_8888)
            val canvas = Canvas(bitmap)
            webView.draw(canvas)

            val timeStamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())
            val fileName = "Room_$timeStamp.png"
            val picturesDir = getExternalFilesDir(Environment.DIRECTORY_PICTURES)
            val file = File(picturesDir, fileName)

            FileOutputStream(file).use { out ->
                bitmap.compress(Bitmap.CompressFormat.PNG, 100, out)
            }

            Toast.makeText(this, getString(R.string.glb_room_screenshot_saved, fileName), Toast.LENGTH_SHORT).show()
            LogUtil.d(TAG, "Screenshot saved: ${file.absolutePath}")

            val uri = FileProvider.getUriForFile(this, "${packageName}.fileprovider", file)
            val shareIntent = Intent(Intent.ACTION_SEND).apply {
                type = "image/png"
                putExtra(Intent.EXTRA_STREAM, uri)
                addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            }
            startActivity(Intent.createChooser(shareIntent, "Share Screenshot"))

        } catch (e: Exception) {
            LogUtil.e(TAG, "Failed to take screenshot", e)
            Toast.makeText(this, getString(R.string.glb_room_screenshot_failed), Toast.LENGTH_SHORT).show()
            CrashReporter.report(this, e, "GLB room — screenshot / share")
        }
    }

    // JavaScript interface for communication from WebView
    inner class WebAppInterface {
        @JavascriptInterface
        fun onLoaded() {
            runOnUiThread {
                loadingOverlay.visibility = View.GONE
                LogUtil.d(TAG, "WebGL viewer reported loaded")
            }
        }

        @JavascriptInterface
        fun onError(message: String) {
            runOnUiThread {
                loadingOverlay.visibility = View.GONE
                Toast.makeText(this@GLBRoomActivity, message, Toast.LENGTH_LONG).show()
                LogUtil.e(TAG, "WebGL error: $message")
                CrashReporter.report(
                    this@GLBRoomActivity,
                    RuntimeException(message),
                    "GLB room — WebGL viewer",
                )
            }
        }

        @JavascriptInterface
        fun log(message: String) {
            LogUtil.d(TAG, "WebGL: $message")
        }
    }

    private fun handleBackNavigation() {
        if (isPreviewMode) {
            if (webView.canGoBack()) {
                webView.goBack()
            } else {
                showUnsavedPreviewLeaveDialog()
            }
        } else {
            if (webView.canGoBack()) {
                webView.goBack()
            } else {
                finish()
            }
        }
    }

    private fun showUnsavedPreviewLeaveDialog() {
        AlertDialog.Builder(this)
            .setTitle(R.string.room_preview_leave_title)
            .setMessage(R.string.room_preview_leave_message)
            .setNegativeButton(R.string.room_preview_leave_stay, null)
            .setPositiveButton(R.string.room_preview_leave_confirm) { _, _ -> finish() }
            .show()
    }

    override fun onBackPressed() {
        handleBackNavigation()
    }

    override fun onDestroy() {
        webView.destroy()
        super.onDestroy()
    }
}
