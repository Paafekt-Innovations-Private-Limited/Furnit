package com.furnit.android

import android.annotation.SuppressLint
import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.content.res.ColorStateList
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.os.Bundle
import android.os.Environment
import com.furnit.android.utils.CrashReporter
import com.furnit.android.theme.PaafektColors
import com.furnit.android.theme.PaafektDialogs
import com.furnit.android.theme.PaafektDrawables
import com.furnit.android.theme.PaafektSnackbar
import com.furnit.android.theme.PaafektFirstRunCoachMarkController
import com.furnit.android.theme.PaafektHintController
import com.furnit.android.theme.PaafektImmersiveChromeController
import com.furnit.android.theme.PaafektHintViews
import com.furnit.android.theme.PaafektSpace
import com.furnit.android.theme.PaafektViewerToolbar
import com.furnit.android.utils.LogUtil
import com.furnit.android.utils.RoomDisplayName
import com.furnit.android.utils.RoomFolderMetadata
import android.view.Gravity
import android.view.GestureDetector
import android.view.MotionEvent
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
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.FileProvider
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.WindowInsetsControllerCompat
import androidx.webkit.WebViewAssetLoader
import androidx.lifecycle.lifecycleScope
import com.furnit.android.ar.rotateToMatchLockedRoomPhoto
import com.furnit.android.models.ModelManager
import com.furnit.android.services.FurnitureFitManager
import com.furnit.android.services.SegmentationResult
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger

/**
 * GLBRoomActivity - WebGL-based GLB/GLTF 3D room viewer
 * (Matches Swift's GLBRoomView exactly)
 *
 * Uses THREE.js and GLTFLoader to render GLB files in a WebView
 */
class GLBRoomActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "GLBRoomActivity"
        private const val GLB_MAGIC = 0x46546C67 // "glTF"
        private const val GLB_VERSION = 2
        /** Bundled Three.js (iOS WebViewVendor parity) — no CDN/network required in WebView. */
        private const val VIEWER_THREE_BASE =
            "https://appassets.androidplatform.net/assets/vendor/three"
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
    private lateinit var rootLayout: FrameLayout
    private lateinit var topBar: FrameLayout
    private lateinit var cameraDpadOverlay: FrameLayout
    private lateinit var bottomControls: FrameLayout
    private lateinit var immersiveRestingChrome: FrameLayout
    private lateinit var windowInsetsController: WindowInsetsControllerCompat
    private var bottomControlsInnerColumn: LinearLayout? = null
    private var immersiveBackButton: View? = null
    private var immersiveSummonButton: View? = null
    private val immersiveChrome = PaafektImmersiveChromeController()
    private lateinit var immersiveTapDetector: GestureDetector
    private lateinit var brainDetectionOverlay: FrameLayout
    private lateinit var brainDetectionOverlayView: FurnitureFitOverlayView
    private lateinit var brainCameraPreview: PreviewView
    private lateinit var brainProgressOverlay: FrameLayout
    private lateinit var brainProgressLabel: TextView
    private lateinit var brainProgressBar: ProgressBar
    private lateinit var hintController: PaafektHintController
    private lateinit var firstRunCoachController: PaafektFirstRunCoachMarkController
    private lateinit var cameraExecutor: ExecutorService
    private var cameraProvider: ProcessCameraProvider? = null
    private var boundPreview: Preview? = null
    private var furnitureFitManager: FurnitureFitManager? = null
    private val isBrainInferenceRunning = AtomicBoolean(false)
    private val brainSessionGeneration = AtomicInteger(0)
    private var brainAcceptingUpdates = false
    private var brainButton: LinearLayout? = null
    private var trailingArSizingButton: ImageButton? = null
    private var inlineBrainArAssistedSizingEnabled = false
    private var brainSegmentButton: TextView? = null
    private var brainFullVideoButton: ImageButton? = null
    @Volatile private var inlineBrainMode: InlineBrainMode = InlineBrainMode.DEFAULT_SEGMENT
    @Volatile private var inlineBrainFullVideoEnabled = false
    @Volatile private var inlineBrainSelectedPins: List<DetectionResult> = emptyList()
    private var glbPath: String? = null
    private var glbViewerCacheDir: File? = null
    private var roomName: String = "3D Room"
    private var roomId: String? = null
    private var isPreviewMode: Boolean = false
    private var photoOrientation: String = "portrait"

    // Room dimensions
    private var roomWidth: Float = 4.0f
    private var roomHeight: Float = 3.0f
    private var roomDepth: Float = 4.5f

    private enum class InlineBrainMode {
        DEFAULT_SEGMENT,
        IDENTIFY,
        SEGMENT_SELECTED,
    }

    private val cameraPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission(),
    ) { granted ->
        if (granted) {
            startInlineBrainSegmentation()
        } else {
            Toast.makeText(this, getString(R.string.camera_permission_required), Toast.LENGTH_SHORT).show()
        }
    }

    @SuppressLint("SetJavaScriptEnabled")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        cameraExecutor = Executors.newSingleThreadExecutor()

        // Edge-to-edge is completed in installImmersiveEdgeToEdge() after views are built.
        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.P) {
            window.attributes.layoutInDisplayCutoutMode =
                WindowManager.LayoutParams.LAYOUT_IN_DISPLAY_CUTOUT_MODE_SHORT_EDGES
        }

        glbPath = intent.getStringExtra(EXTRA_GLB_PATH)
        roomName = intent.getStringExtra(EXTRA_ROOM_NAME) ?: "3D Room"
        roomId = intent.getStringExtra(EXTRA_ROOM_ID)
        isPreviewMode = intent.getBooleanExtra(EXTRA_IS_PREVIEW, false)
        roomWidth = intent.getFloatExtra(EXTRA_ROOM_WIDTH, RoomDefaults.widthMeters(this))
        roomHeight = intent.getFloatExtra(EXTRA_ROOM_HEIGHT, RoomDefaults.heightMeters(this))
        roomDepth = intent.getFloatExtra("ROOM_DEPTH", RoomDefaults.depthMeters(this))
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

        rootLayout = FrameLayout(this).apply {
            fitsSystemWindows = false
            setBackgroundColor(Color.parseColor("#808080"))
        }

        // WebView for 3D rendering — full-bleed behind system bars; insets apply to chrome only.
        webView = WebView(this).apply {
            fitsSystemWindows = false
            settings.javaScriptEnabled = true
            settings.domStorageEnabled = true
            settings.allowFileAccess = true
            settings.allowContentAccess = true
            settings.mixedContentMode = WebSettings.MIXED_CONTENT_ALWAYS_ALLOW
            setBackgroundColor(Color.TRANSPARENT)

            webChromeClient = object : WebChromeClient() {
                override fun onConsoleMessage(message: ConsoleMessage?): Boolean {
                    val text = message?.message().orEmpty()
                    val line = "WebGL [${message?.messageLevel()}] ${message?.sourceId()}:${message?.lineNumber()} $text"
                    when (message?.messageLevel()) {
                        ConsoleMessage.MessageLevel.ERROR -> LogUtil.e(TAG, line)
                        ConsoleMessage.MessageLevel.WARNING -> LogUtil.w(TAG, line)
                        else -> LogUtil.d(TAG, line)
                    }
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

            immersiveTapDetector = GestureDetector(
                this@GLBRoomActivity,
                object : GestureDetector.SimpleOnGestureListener() {
                    override fun onSingleTapUp(e: MotionEvent): Boolean {
                        if (immersiveChrome.phase == PaafektImmersiveChromeController.Phase.RESTING
                            && !(inlineBrainFullVideoEnabled && inlineBrainMode == InlineBrainMode.IDENTIFY)
                        ) {
                            immersiveChrome.summon()
                        }
                        return false
                    }
                },
            )
            setOnTouchListener { _, event ->
                immersiveTapDetector.onTouchEvent(event)
                false
            }
        }
        rootLayout.addView(webView, FrameLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.MATCH_PARENT
        ))

        brainCameraPreview = PreviewView(this).apply {
            implementationMode = PreviewView.ImplementationMode.COMPATIBLE
            scaleType = PreviewView.ScaleType.FILL_CENTER
            visibility = View.GONE
            setBackgroundColor(Color.BLACK)
            elevation = 10f
        }
        rootLayout.addView(
            brainCameraPreview,
            FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT,
            ),
        )

        // No gesture overlay - let WebView's OrbitControls handle all gestures
        // (rotation, zoom, pan) directly like iOS

        // Top bar
        topBar = createTopBar()
        rootLayout.addView(topBar, FrameLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT
        ).apply { gravity = Gravity.TOP })

        hintController = PaafektHintController(rootLayout)
        firstRunCoachController = PaafektFirstRunCoachMarkController(rootLayout)

        // Gesture navigation only — no on-screen d-pad (immersive-first).
        cameraDpadOverlay = FrameLayout(this)
        cameraDpadOverlay.visibility = View.GONE

        // Bottom controls (summoned chrome)
        bottomControls = createBottomControls()
        rootLayout.addView(bottomControls, FrameLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT
        ).apply { gravity = Gravity.BOTTOM })

        immersiveRestingChrome = createImmersiveRestingChrome()
        rootLayout.addView(
            immersiveRestingChrome,
            FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT,
            ),
        )

        immersiveChrome.onPhaseChanged = { refreshImmersiveChromeVisibility() }
        refreshImmersiveChromeVisibility(animate = false)

        // Loading overlay
        loadingOverlay = createLoadingOverlay()
        rootLayout.addView(loadingOverlay)

        brainProgressOverlay = createBrainProgressOverlay().apply {
            visibility = View.GONE
            elevation = 20f
        }
        rootLayout.addView(brainProgressOverlay)

        brainDetectionOverlay = FrameLayout(this).apply {
            visibility = View.GONE
            elevation = 21f
            setBackgroundColor(Color.TRANSPARENT)
        }
        brainDetectionOverlayView = FurnitureFitOverlayView(this).apply {
            setBackgroundColor(Color.TRANSPARENT)
            layoutParams = FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT,
            )
            onTouchOutsideFurniture = { event -> webView.dispatchTouchEvent(event) }
            onDetectionTapped = { detection -> toggleInlineBrainSelectedCluster(detection) }
        }
        brainDetectionOverlay.addView(brainDetectionOverlayView)
        brainSegmentButton = createInlineBrainSegmentButton().apply {
            visibility = View.GONE
        }
        brainDetectionOverlay.addView(
            brainSegmentButton,
            FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT,
                ViewGroup.LayoutParams.WRAP_CONTENT,
            ).apply {
                gravity = Gravity.CENTER_HORIZONTAL or Gravity.BOTTOM
                bottomMargin = dpToPx(120)
            },
        )
        rootLayout.addView(
            brainDetectionOverlay,
            FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT,
            ),
        )

        brainFullVideoButton = createBrainFullVideoButton()
        rootLayout.addView(
            brainFullVideoButton,
            FrameLayout.LayoutParams(dpToPx(36), dpToPx(36)).apply {
                gravity = Gravity.END or Gravity.TOP
                topMargin = dpToPx(116)
                marginEnd = dpToPx(16)
            },
        )

        setContentView(rootLayout)
        installImmersiveEdgeToEdge()
        ensureNavigationChromeOnTop()

        rootLayout.post {
            firstRunCoachController.showIfNeeded(this) {
                hintController.showBottomCentered(
                    this,
                    R.drawable.ic_gesture_tap,
                    R.string.room_viewer_hero_actions_teaching_hint,
                    bottomMarginDp = 188,
                )
            }
        }

        // Load the WebGL viewer
        loadWebGLViewer()
    }

    /** WebView fills the window; system-bar insets pad chrome overlays only. */
    private fun installImmersiveEdgeToEdge() {
        WindowCompat.setDecorFitsSystemWindows(window, false)
        window.statusBarColor = Color.TRANSPARENT
        window.navigationBarColor = Color.TRANSPARENT

        windowInsetsController = WindowInsetsControllerCompat(window, window.decorView).apply {
            isAppearanceLightStatusBars = false
            isAppearanceLightNavigationBars = false
            systemBarsBehavior = WindowInsetsControllerCompat.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
        }
        enterImmersiveMode()

        ViewCompat.setOnApplyWindowInsetsListener(rootLayout) { _, insets ->
            applyChromeWindowInsets(insets.getInsets(WindowInsetsCompat.Type.systemBars()))
            notifyWebViewViewportChanged()
            WindowInsetsCompat.CONSUMED
        }
        ViewCompat.requestApplyInsets(rootLayout)
    }

    private fun enterImmersiveMode() {
        if (!::windowInsetsController.isInitialized) return
        windowInsetsController.hide(WindowInsetsCompat.Type.systemBars())
    }

    private fun applyChromeWindowInsets(bars: androidx.core.graphics.Insets) {
        val side = PaafektSpace.lg(this)
        if (::topBar.isInitialized) {
            topBar.setPadding(side, bars.top + PaafektSpace.sm(this), side, 0)
        }
        if (::bottomControls.isInitialized) {
            bottomControls.setPadding(side, 0, side, bars.bottom + PaafektSpace.sm(this))
        }
        bottomControlsInnerColumn?.layoutParams?.let { lp ->
            if (lp is FrameLayout.LayoutParams) {
                lp.bottomMargin = bars.bottom + PaafektSpace.viewerBottomInset(this)
                bottomControlsInnerColumn?.layoutParams = lp
            }
        }
        immersiveBackButton?.layoutParams?.let { lp ->
            if (lp is FrameLayout.LayoutParams) {
                lp.topMargin = bars.top + PaafektSpace.sm(this)
                immersiveBackButton?.layoutParams = lp
            }
        }
        immersiveSummonButton?.layoutParams?.let { lp ->
            if (lp is FrameLayout.LayoutParams) {
                lp.bottomMargin = bars.bottom + PaafektSpace.viewerBottomInset(this) + dpToPx(8)
                immersiveSummonButton?.layoutParams = lp
            }
        }
    }

    private fun notifyWebViewViewportChanged() {
        if (!::webView.isInitialized) return
        webView.post {
            if (webView.width <= 0 || webView.height <= 0) return@post
            webView.evaluateJavascript(
                "if(typeof resizeViewer==='function'){resizeViewer();}",
                null,
            )
        }
    }

    private fun createImmersiveRestingChrome(): FrameLayout {
        return FrameLayout(this).apply {
            val back = PaafektViewerToolbar.createFloatingBackButton(this@GLBRoomActivity) {
                handleBackNavigation()
            }.apply {
                alpha = 0.55f
                contentDescription = getString(R.string.photo_room_back)
            }
            immersiveBackButton = back
            addView(
                back,
                FrameLayout.LayoutParams(
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                ).apply {
                    gravity = Gravity.START or Gravity.TOP
                    topMargin = PaafektSpace.sm(this@GLBRoomActivity)
                    marginStart = PaafektSpace.lg(this@GLBRoomActivity)
                },
            )

            val measurementPill = TextView(this@GLBRoomActivity).apply {
                text = restingMeasurementPillText()
                textSize = 12f
                setTextColor(PaafektColors.textSecondary)
                setPadding(dpToPx(12), dpToPx(8), dpToPx(12), dpToPx(8))
                background = PaafektDrawables.hintChip()
            }
            addView(
                measurementPill,
                FrameLayout.LayoutParams(
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                ).apply {
                    gravity = Gravity.BOTTOM or Gravity.CENTER_HORIZONTAL
                    bottomMargin = dpToPx(72)
                },
            )

            val summonGold = ImageButton(this@GLBRoomActivity).apply {
                setImageResource(R.drawable.ic_chevron_up)
                imageTintList = ColorStateList.valueOf(PaafektColors.accentText)
                background = GradientDrawable().apply {
                    shape = GradientDrawable.OVAL
                    setColor(PaafektColors.accent)
                }
                contentDescription = getString(R.string.room_viewer_immersive_show_controls)
                setPadding(dpToPx(12), dpToPx(12), dpToPx(12), dpToPx(12))
                setOnClickListener { immersiveChrome.summon() }
            }
            immersiveSummonButton = summonGold
            addView(
                summonGold,
                FrameLayout.LayoutParams(dpToPx(46), dpToPx(46)).apply {
                    gravity = Gravity.END or Gravity.BOTTOM
                    bottomMargin = PaafektSpace.viewerBottomInset(this@GLBRoomActivity) + dpToPx(8)
                    marginEnd = PaafektSpace.lg(this@GLBRoomActivity)
                },
            )
        }
    }

    private fun restingMeasurementPillText(): String {
        return String.format("%.1f m × %.1f m", roomWidth, roomDepth)
    }

    private fun refreshImmersiveChromeVisibility(animate: Boolean = true) {
        if (!::immersiveRestingChrome.isInitialized) return
        immersiveChrome.applyPhase(
            this,
            restingViews = listOf(immersiveRestingChrome),
            summonedViews = listOf(topBar, bottomControls),
            animate = animate,
        )
        updateInlineBrainSegmentButton()
        ensureNavigationChromeOnTop()
    }

    /** Keep back / title / recenter / bottom brain+camera above the WebView and brain overlay. */
    private fun ensureNavigationChromeOnTop() {
        topBar.elevation = 40f
        brainFullVideoButton?.elevation = 38f
        bottomControls.elevation = 37f
        if (::immersiveRestingChrome.isInitialized) {
            immersiveRestingChrome.elevation = 36f
        }
        rootLayout.bringChildToFront(bottomControls)
        if (::immersiveRestingChrome.isInitialized) {
            rootLayout.bringChildToFront(immersiveRestingChrome)
        }
        if (::brainProgressOverlay.isInitialized && brainProgressOverlay.visibility == View.VISIBLE) {
            rootLayout.bringChildToFront(brainProgressOverlay)
        }
        brainFullVideoButton?.takeIf { it.visibility == View.VISIBLE }?.let { rootLayout.bringChildToFront(it) }
        rootLayout.bringChildToFront(topBar)
        hintController.bringToFront()
        firstRunCoachController.bringToFront()
    }

    private fun dpToPx(dp: Int): Int {
        return (dp * resources.displayMetrics.density).toInt()
    }

    private fun createDpadCircleButton(label: String, onClick: () -> Unit): TextView {
        return TextView(this).apply {
            text = label
            textSize = 20f
            setTypeface(null, Typeface.BOLD)
            setTextColor(Color.WHITE)
            gravity = Gravity.CENTER
            background = GradientDrawable().apply {
                shape = GradientDrawable.OVAL
                setColor(Color.parseColor("#80000000"))
            }
            val size = dpToPx(44)
            layoutParams = LinearLayout.LayoutParams(size, size)
            setOnClickListener { onClick() }
        }
    }

    /** Top-left arrow cluster — posts the same JS calls as iOS `WebGLCameraMove*`. */
    private fun createCameraDPadOverlay(): FrameLayout {
        val topInset = if (photoOrientation == "landscape") dpToPx(12) else dpToPx(110)
        return FrameLayout(this).apply {
            val cluster = LinearLayout(this@GLBRoomActivity).apply {
                orientation = LinearLayout.HORIZONTAL
                gravity = Gravity.CENTER_VERTICAL
            }

            cluster.addView(createDpadCircleButton("\u2190") { nudgeCameraLeft() })

            val verticalPad = LinearLayout(this@GLBRoomActivity).apply {
                orientation = LinearLayout.VERTICAL
                val pad = dpToPx(8)
                setPadding(pad, 0, pad, 0)
            }
            verticalPad.addView(createDpadCircleButton("\u2191") { nudgeCameraUp() })
            verticalPad.addView(createDpadCircleButton("\u2193") { nudgeCameraDown() })
            cluster.addView(verticalPad)

            cluster.addView(createDpadCircleButton("\u2192") { nudgeCameraRight() })

            addView(
                cluster,
                FrameLayout.LayoutParams(
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                ).apply {
                    gravity = Gravity.START or Gravity.TOP
                    topMargin = topInset
                    marginStart = dpToPx(12)
                },
            )
        }
    }

    private fun createTopBar(): FrameLayout {
        return PaafektViewerToolbar.createTopChromeRow(this).apply {
            val backBtn = PaafektViewerToolbar.createFloatingBackButton(this@GLBRoomActivity) {
                handleBackNavigation()
            }.apply {
                contentDescription = getString(R.string.photo_room_back)
            }
            addView(
                backBtn,
                FrameLayout.LayoutParams(
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                ).apply { gravity = Gravity.START or Gravity.TOP },
            )

            val principalControls = PaafektViewerToolbar.createToolbarCapsule(this@GLBRoomActivity)
            principalControls.addView(
                PaafektViewerToolbar.createCapsuleIconButton(
                    this@GLBRoomActivity,
                    R.drawable.ic_ruler,
                    contentDescription = getString(R.string.faq_measurement_pill),
                ) { showRoomDimensionsHint() },
            )
            principalControls.addView(
                PaafektViewerToolbar.createCapsuleIconButton(
                    this@GLBRoomActivity,
                    R.drawable.ic_gesture_pinch,
                    contentDescription = getString(R.string.room_viewer_navigation_teaching_hint),
                ) {
                    if (hintController.isVisible) {
                        hintController.hide()
                    } else {
                        hintController.showBottomCentered(
                            this@GLBRoomActivity,
                            R.drawable.ic_gesture_pinch,
                            R.string.room_viewer_navigation_teaching_hint,
                        )
                    }
                },
            )
            principalControls.addView(
                PaafektViewerToolbar.createCapsuleIconButton(
                    this@GLBRoomActivity,
                    R.drawable.ic_gesture_tap,
                    contentDescription = getString(R.string.room_viewer_display_all_helpers),
                ) { showAllGestureHelpers() },
            )

            addView(
                principalControls,
                FrameLayout.LayoutParams(
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                ).apply { gravity = Gravity.TOP or Gravity.CENTER_HORIZONTAL },
            )

            val trailingControls = LinearLayout(this@GLBRoomActivity).apply {
                orientation = LinearLayout.HORIZONTAL
                gravity = Gravity.CENTER_VERTICAL
            }
            trailingControls.addView(
                PaafektViewerToolbar.createFloatingIconButton(
                    this@GLBRoomActivity,
                    R.drawable.ic_viewfinder,
                    contentDescription = getString(R.string.room_viewer_recenter),
                ) { recenterCamera() },
            )
            if (isPreviewMode) {
                trailingControls.addView(
                    PaafektViewerToolbar.createFloatingIconButton(
                        this@GLBRoomActivity,
                        R.drawable.ic_download,
                        contentDescription = getString(R.string.common_save),
                    ) { showSaveDialog() },
                )
            }
            trailingArSizingButton = PaafektViewerToolbar.createFloatingIconButton(
                this@GLBRoomActivity,
                R.drawable.ic_square_resize,
                contentDescription = getString(R.string.room_viewer_ar_sizing_enable),
                isActive = inlineBrainArAssistedSizingEnabled,
                activeFillColor = 0xE634C759.toInt(),
            ) {
                toggleInlineBrainArAssistedSizing()
            }.also { trailingControls.addView(it) }
            updateTrailingArSizingVisibility()

            addView(
                trailingControls,
                FrameLayout.LayoutParams(
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                ).apply { gravity = Gravity.END or Gravity.TOP },
            )

            titleView = TextView(this@GLBRoomActivity).apply {
                visibility = View.GONE
                text = roomName
            }
        }
    }

    private fun updateTrailingArSizingVisibility() {
        val brainActive = ::brainDetectionOverlay.isInitialized &&
            brainDetectionOverlay.visibility == View.VISIBLE
        trailingArSizingButton?.visibility = if (brainActive) View.VISIBLE else View.GONE
    }

    private fun toggleInlineBrainArAssistedSizing() {
        inlineBrainArAssistedSizingEnabled = !inlineBrainArAssistedSizingEnabled
        trailingArSizingButton?.background = if (inlineBrainArAssistedSizingEnabled) {
            android.graphics.drawable.GradientDrawable().apply {
                shape = android.graphics.drawable.GradientDrawable.OVAL
                setColor(0xE634C759.toInt())
            }
        } else {
            PaafektDrawables.toolbarCircle()
        }
    }

    private fun showAllGestureHelpers() {
        hintController.showBottomCentered(
            this,
            R.drawable.ic_gesture_tap,
            R.string.room_viewer_hero_actions_teaching_hint,
            bottomMarginDp = 188,
        )
    }

    private fun showRoomDimensionsHint() {
        val heightLabel = getString(R.string.approximate_room_height, roomHeight)
        hintController.showText(
            this,
            R.drawable.ic_ruler,
            heightLabel,
            topMarginDp = 52,
        )
    }

    private fun toolbarCapsuleDrawable(): GradientDrawable = PaafektDrawables.toolbarCapsule()

    private fun toolbarCircleDrawable(): GradientDrawable = PaafektDrawables.toolbarCircle()

    private fun createBottomIconButton(
        iconResId: Int,
        isActive: Boolean = false,
        onClick: () -> Unit,
    ): ImageButton {
        return ImageButton(this).apply {
            setImageResource(iconResId)
            imageTintList = ColorStateList.valueOf(
                if (isActive) com.furnit.android.theme.PaafektColors.accent else Color.WHITE,
            )
            background = toolbarCircleDrawable()
            scaleType = ImageView.ScaleType.CENTER_INSIDE
            setPadding(dpToPx(10), dpToPx(10), dpToPx(10), dpToPx(10))
            setOnClickListener { onClick() }
        }
    }

    private fun createToolbarIconButton(iconResId: Int, onClick: () -> Unit): ImageButton {
        return ImageButton(this).apply {
            setImageResource(iconResId)
            imageTintList = ColorStateList.valueOf(Color.WHITE)
            background = toolbarCircleDrawable()
            scaleType = ImageView.ScaleType.CENTER
            setPadding(dpToPx(7), dpToPx(7), dpToPx(7), dpToPx(7))
            layoutParams = LinearLayout.LayoutParams(dpToPx(36), dpToPx(36)).apply {
                setMargins(dpToPx(4), 0, dpToPx(4), 0)
            }
            setOnClickListener { onClick() }
        }
    }

    private fun createToolbarTextButton(label: String, onClick: () -> Unit): TextView {
        return TextView(this).apply {
            text = label
            setTextColor(Color.WHITE)
            gravity = Gravity.CENTER
            setTypeface(null, Typeface.BOLD)
            background = toolbarCircleDrawable()
            setOnClickListener { onClick() }
        }
    }

    private fun createBottomControls(): FrameLayout {
        return FrameLayout(this).apply {
            setPadding(PaafektSpace.lg(this@GLBRoomActivity), 0, PaafektSpace.lg(this@GLBRoomActivity), PaafektSpace.xl(this@GLBRoomActivity))

            val column = LinearLayout(this@GLBRoomActivity).apply {
                orientation = LinearLayout.VERTICAL
                gravity = Gravity.CENTER_HORIZONTAL
                layoutParams = FrameLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                ).apply {
                    gravity = Gravity.BOTTOM
                    bottomMargin = PaafektSpace.viewerBottomInset(this@GLBRoomActivity)
                }
            }
            bottomControlsInnerColumn = column

            val tapToHide = TextView(this@GLBRoomActivity).apply {
                text = getString(R.string.room_viewer_immersive_tap_to_hide)
                textSize = 11f
                setTextColor(PaafektColors.textSecondary)
                gravity = Gravity.CENTER
                setPadding(0, 0, 0, PaafektSpace.sm(this@GLBRoomActivity))
                setOnClickListener { immersiveChrome.immerse() }
            }
            column.addView(tapToHide)

            val heroRow = LinearLayout(this@GLBRoomActivity).apply {
                orientation = LinearLayout.HORIZONTAL
                gravity = Gravity.CENTER
                layoutParams = LinearLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                )
            }

            val fitBtn = PaafektHintViews.createHeroButton(
                this@GLBRoomActivity,
                R.drawable.ic_ai,
                getString(R.string.room_viewer_hero_fit_furniture),
                isActive = false,
            ) {
                immersiveChrome.noteChromeInteraction()
                toggleInlineBrainSegmentation()
            }
            fitBtn.layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f).apply {
                marginEnd = PaafektSpace.sm(this@GLBRoomActivity)
            }
            brainButton = fitBtn
            heroRow.addView(fitBtn)

            val captureBtn = PaafektHintViews.createHeroButton(
                this@GLBRoomActivity,
                R.drawable.ic_snapshot,
                getString(R.string.room_viewer_hero_capture),
            ) {
                immersiveChrome.noteChromeInteraction()
                takeScreenshot()
            }
            captureBtn.layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f).apply {
                marginStart = PaafektSpace.sm(this@GLBRoomActivity)
            }
            heroRow.addView(captureBtn)

            column.addView(heroRow)
            addView(column)
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
        intent.addFlags(Intent.FLAG_ACTIVITY_NO_ANIMATION)
        startActivity(intent)
        overridePendingTransition(0, 0)
    }

    private fun createBrainProgressOverlay(): FrameLayout {
        return FrameLayout(this).apply {
            setBackgroundColor(Color.TRANSPARENT)
            isClickable = false
            isFocusable = false
            val content = LinearLayout(this@GLBRoomActivity).apply {
                orientation = LinearLayout.VERTICAL
                gravity = Gravity.CENTER
                setPadding(0, 0, 0, 0)
                brainProgressLabel = TextView(this@GLBRoomActivity).apply {
                    text = getString(R.string.smartypants_detecting_furniture)
                    textSize = 14f
                    setTypeface(null, Typeface.BOLD)
                    setTextColor(Color.WHITE)
                    gravity = Gravity.CENTER
                    setPadding(dpToPx(14), dpToPx(8), dpToPx(14), dpToPx(8))
                    background = GradientDrawable().apply {
                        cornerRadius = dpToPx(10).toFloat()
                        setColor(0x99000000.toInt())
                    }
                }
                addView(brainProgressLabel)
                brainProgressBar = ProgressBar(
                    this@GLBRoomActivity,
                    null,
                    android.R.attr.progressBarStyleHorizontal,
                ).apply {
                    isIndeterminate = false
                    max = 100
                    progress = 55
                    progressTintList = ColorStateList.valueOf(Color.parseColor("#34C759"))
                    progressBackgroundTintList = ColorStateList.valueOf(0x4DFFFFFF)
                }
                addView(
                    brainProgressBar,
                    LinearLayout.LayoutParams(
                        ViewGroup.LayoutParams.MATCH_PARENT,
                        dpToPx(3),
                    ).apply { topMargin = dpToPx(8) },
                )
            }
            addView(
                content,
                FrameLayout.LayoutParams(
                    dpToPx(300),
                    dpToPx(62),
                ).apply {
                    gravity = Gravity.TOP or Gravity.CENTER_HORIZONTAL
                    topMargin = dpToPx(96)
                },
            )
        }
    }

    private fun showBrainProgress(message: CharSequence, progressPercent: Int = 55) {
        if (!::brainProgressOverlay.isInitialized) return
        brainProgressLabel.text = message
        brainProgressBar.progress = progressPercent.coerceIn(0, 100)
        brainProgressOverlay.visibility = View.VISIBLE
        ensureNavigationChromeOnTop()
    }

    private fun hideBrainProgress() {
        if (::brainProgressOverlay.isInitialized) {
            brainProgressOverlay.visibility = View.GONE
        }
    }

    private fun createBrainFullVideoButton(): ImageButton {
        return ImageButton(this).apply {
            setImageResource(R.drawable.ic_text_viewfinder)
            scaleType = ImageView.ScaleType.CENTER_INSIDE
            setPadding(dpToPx(6), dpToPx(6), dpToPx(6), dpToPx(6))
            background = GradientDrawable().apply {
                shape = GradientDrawable.OVAL
                setColor(0x9E000000.toInt())
                setStroke(dpToPx(1), Color.parseColor("#2EFFFFFF"))
            }
            contentDescription = getString(R.string.room_viewer_full_video_with_identifications)
            visibility = View.GONE
            setOnClickListener { toggleInlineBrainFullVideoMode() }
        }
    }

    private fun updateBrainFullVideoButtonAppearance() {
        val button = brainFullVideoButton ?: return
        val active = inlineBrainFullVideoEnabled
        val iconColor = if (active) Color.parseColor("#00FFFF") else Color.WHITE
        button.imageTintList = ColorStateList.valueOf(iconColor)
        (button.background as? GradientDrawable)?.setStroke(
            dpToPx(1),
            if (active) Color.parseColor("#E600FFFF") else Color.parseColor("#2EFFFFFF"),
        )
    }

    private fun toggleInlineBrainFullVideoMode() {
        if (brainDetectionOverlay.visibility != View.VISIBLE) return
        inlineBrainFullVideoEnabled = !inlineBrainFullVideoEnabled
        inlineBrainSelectedPins = emptyList()
        inlineBrainMode = if (inlineBrainFullVideoEnabled) {
            InlineBrainMode.IDENTIFY
        } else {
            InlineBrainMode.DEFAULT_SEGMENT
        }
        brainDetectionOverlayView.setMaskAndDetections(
            mask = null,
            dets = emptyList(),
            frameAlignedOverlay = inlineBrainFullVideoEnabled,
        )
        brainDetectionOverlayView.setDetectionBoxVisibility(inlineBrainFullVideoEnabled)
        brainDetectionOverlayView.setIdentifySelectionState(
            inlineBrainFullVideoEnabled,
            inlineBrainSelectedPins,
        )
        if (inlineBrainFullVideoEnabled) {
            hideBrainProgress()
        } else {
            showBrainProgress(getString(R.string.smartypants_detecting_furniture))
        }
        updateInlineBrainSegmentButton()
        updateBrainFullVideoButtonAppearance()
        ensureNavigationChromeOnTop()
        rebindInlineBrainCameraIfActive()
        LogUtil.d(
            TAG,
            "Inline brain full video toggled: enabled=$inlineBrainFullVideoEnabled mode=$inlineBrainMode",
        )
    }

    private fun shouldShowInlineBrainCameraPreview(): Boolean {
        return inlineBrainFullVideoEnabled &&
            inlineBrainMode == InlineBrainMode.IDENTIFY &&
            brainDetectionOverlay.visibility == View.VISIBLE
    }

    private fun updateInlineBrainCameraPreviewVisibility() {
        if (!::brainCameraPreview.isInitialized) return
        brainCameraPreview.visibility = if (shouldShowInlineBrainCameraPreview()) View.VISIBLE else View.GONE
    }

    private fun rebindInlineBrainCameraIfActive() {
        if (brainDetectionOverlay.visibility != View.VISIBLE) return
        val provider = cameraProvider
        if (provider != null) {
            applyInlineBrainCameraBinding(provider, brainSessionGeneration.get())
        } else {
            bindInlineBrainCamera(brainSessionGeneration.get())
        }
    }

    private fun createInlineBrainSegmentButton(): TextView {
        return TextView(this).apply {
            text = getString(R.string.segment_furniture_action)
            textSize = 16f
            setTypeface(null, Typeface.BOLD)
            setTextColor(Color.WHITE)
            gravity = Gravity.CENTER
            setPadding(dpToPx(24), dpToPx(12), dpToPx(24), dpToPx(12))
            background = GradientDrawable().apply {
                shape = GradientDrawable.RECTANGLE
                cornerRadius = dpToPx(24).toFloat()
                setColor(Color.parseColor("#34C759"))
            }
            setOnClickListener { toggleInlineBrainSegmentMode() }
        }
    }

    private fun toggleInlineBrainSegmentation() {
        if (::brainDetectionOverlay.isInitialized && brainDetectionOverlay.visibility == View.VISIBLE) {
            stopInlineBrainSegmentation()
            return
        }
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
            return
        }
        startInlineBrainSegmentation()
    }

    private fun startInlineBrainSegmentation() {
        LogUtil.d(TAG, "Inline brain: start")
        val generation = brainSessionGeneration.incrementAndGet()
        inlineBrainFullVideoEnabled = false
        inlineBrainMode = InlineBrainMode.DEFAULT_SEGMENT
        inlineBrainSelectedPins = emptyList()
        brainAcceptingUpdates = false
        isBrainInferenceRunning.set(false)
        brainDetectionOverlay.visibility = View.VISIBLE
        brainFullVideoButton?.visibility = View.VISIBLE
        updateBrainFullVideoButtonAppearance()
        ensureNavigationChromeOnTop()
        brainDetectionOverlayView.setMaskAndDetections(
            mask = null,
            dets = emptyList(),
            frameAlignedOverlay = inlineBrainFullVideoEnabled,
        )
        brainDetectionOverlayView.setDetectionBoxVisibility(inlineBrainFullVideoEnabled)
        brainDetectionOverlayView.setIdentifySelectionState(inlineBrainFullVideoEnabled, inlineBrainSelectedPins)
        showBrainProgress(getString(R.string.detector_loading_model), 20)
        setBrainButtonActive(true)
        updateTrailingArSizingVisibility()
        updateInlineBrainSegmentButton()
        ensureNavigationChromeOnTop()

        lifecycleScope.launch {
            val manager = furnitureFitManager ?: withContext(Dispatchers.IO) {
                FurnitureFitManager(this@GLBRoomActivity).takeIf { it.initializeAuto() }
            }
            if (manager == null) {
                hideBrainProgress()
                setBrainButtonActive(false)
                Toast.makeText(this@GLBRoomActivity, getString(R.string.detector_model_unavailable), Toast.LENGTH_SHORT).show()
                return@launch
            }
            furnitureFitManager = manager
            bindInlineBrainCamera(generation)
        }
    }

    private fun toggleInlineBrainSelectedCluster(detection: DetectionResult) {
        if (!inlineBrainFullVideoEnabled || inlineBrainMode != InlineBrainMode.IDENTIFY) return
        val current = inlineBrainSelectedPins.toMutableList()
        val existingIndex = current.indexOfFirst { pin ->
            pin.classId == detection.classId && detectionIoU(pin, detection) >= 0.50f
        }
        if (existingIndex >= 0) {
            current.removeAt(existingIndex)
        } else {
            current += detection
        }
        inlineBrainSelectedPins = current
        brainDetectionOverlayView.setIdentifySelectionState(true, inlineBrainSelectedPins)
        updateInlineBrainSegmentButton()
    }

    private fun toggleInlineBrainSegmentMode() {
        if (!inlineBrainFullVideoEnabled) return
        if (inlineBrainMode == InlineBrainMode.SEGMENT_SELECTED) {
            inlineBrainMode = InlineBrainMode.IDENTIFY
            brainDetectionOverlayView.setMaskAndDetections(
                mask = null,
                dets = emptyList(),
                frameAlignedOverlay = true,
            )
            brainDetectionOverlayView.setDetectionBoxVisibility(true)
            brainDetectionOverlayView.setIdentifySelectionState(true, inlineBrainSelectedPins)
        } else if (inlineBrainSelectedPins.isNotEmpty()) {
            inlineBrainMode = InlineBrainMode.SEGMENT_SELECTED
            brainDetectionOverlayView.setDetectionBoxVisibility(false)
            brainDetectionOverlayView.setIdentifySelectionState(false, inlineBrainSelectedPins)
            showBrainProgress(getString(R.string.detector_segmenting_selection))
        }
        updateInlineBrainSegmentButton()
        updateInlineBrainCameraPreviewVisibility()
        // Keep the existing CameraX Analysis pipeline alive. Rebinding here blocks the UI
        // thread and delays the first selected-segmentation frame; hiding PreviewView is enough.
    }

    private fun updateInlineBrainSegmentButton() {
        val button = brainSegmentButton ?: return
        if (!inlineBrainFullVideoEnabled || brainDetectionOverlay.visibility != View.VISIBLE) {
            button.visibility = View.GONE
            return
        }
        if (immersiveChrome.phase != PaafektImmersiveChromeController.Phase.SUMMONED) {
            button.visibility = View.GONE
            return
        }
        button.visibility = View.VISIBLE
        val segmenting = inlineBrainMode == InlineBrainMode.SEGMENT_SELECTED
        button.text = getString(if (segmenting) R.string.segment_stop_action else R.string.segment_furniture_action)
        button.alpha = if (segmenting || inlineBrainSelectedPins.isNotEmpty()) 1f else 0.55f
    }

    private fun detectionIoU(first: DetectionResult, second: DetectionResult): Float {
        val firstLeft = first.x - first.w / 2f
        val firstTop = first.y - first.h / 2f
        val firstRight = first.x + first.w / 2f
        val firstBottom = first.y + first.h / 2f
        val secondLeft = second.x - second.w / 2f
        val secondTop = second.y - second.h / 2f
        val secondRight = second.x + second.w / 2f
        val secondBottom = second.y + second.h / 2f
        val intersectionLeft = maxOf(firstLeft, secondLeft)
        val intersectionTop = maxOf(firstTop, secondTop)
        val intersectionRight = minOf(firstRight, secondRight)
        val intersectionBottom = minOf(firstBottom, secondBottom)
        val intersectionWidth = maxOf(0f, intersectionRight - intersectionLeft)
        val intersectionHeight = maxOf(0f, intersectionBottom - intersectionTop)
        val intersectionArea = intersectionWidth * intersectionHeight
        val unionArea = first.w * first.h + second.w * second.h - intersectionArea
        return if (unionArea > 0f) intersectionArea / unionArea else 0f
    }

    @SuppressLint("UnsafeOptInUsageError")
    private fun bindInlineBrainCamera(generation: Int) {
        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener({
            applyInlineBrainCameraBinding(providerFuture.get(), generation)
        }, ContextCompat.getMainExecutor(this))
    }

    @SuppressLint("UnsafeOptInUsageError")
    private fun applyInlineBrainCameraBinding(provider: ProcessCameraProvider, generation: Int) {
        cameraProvider = provider
        boundPreview?.setSurfaceProvider(null)
        boundPreview = null
        provider.unbindAll()
        updateInlineBrainCameraPreviewVisibility()

        val analysisSize =
            if (photoOrientation.equals("landscape", ignoreCase = true)) {
                android.util.Size(1280, 720)
            } else {
                android.util.Size(720, 1280)
            }
        val analysis = ImageAnalysis.Builder()
            .setTargetResolution(analysisSize)
            .setTargetRotation(displayRotationForCameraX())
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build()

        analysis.setAnalyzer(cameraExecutor) { imageProxy ->
            try {
                if (!brainAcceptingUpdates || brainSessionGeneration.get() != generation) return@setAnalyzer
                if (isBrainInferenceRunning.get()) return@setAnalyzer
                val rawBitmap = imageProxy.toBitmapSafe() ?: return@setAnalyzer
                val (bitmap, _) = rawBitmap.rotateToMatchLockedRoomPhoto(photoOrientation)
                if (bitmap !== rawBitmap) rawBitmap.recycle()
                isBrainInferenceRunning.set(true)
                val modeSnapshot = inlineBrainMode
                val fullVideoSnapshot = inlineBrainFullVideoEnabled
                val selectedPinsSnapshot = inlineBrainSelectedPins
                val callback: (SegmentationResult?) -> Unit = { result ->
                    bitmap.recycle()
                    runOnUiThread {
                        isBrainInferenceRunning.set(false)
                        if (!brainAcceptingUpdates || brainSessionGeneration.get() != generation) return@runOnUiThread
                        if (inlineBrainMode != modeSnapshot || inlineBrainFullVideoEnabled != fullVideoSnapshot) {
                            LogUtil.d(
                                TAG,
                                "Inline brain: dropping stale result mode=$modeSnapshot current=$inlineBrainMode " +
                                    "fullVideo=$fullVideoSnapshot currentFullVideo=$inlineBrainFullVideoEnabled",
                            )
                            return@runOnUiThread
                        }
                        applyInlineBrainResult(result)
                    }
                }
                when {
                    fullVideoSnapshot && modeSnapshot == InlineBrainMode.IDENTIFY ->
                        furnitureFitManager?.detectWithDetectionsAsync(bitmap, callback)
                    fullVideoSnapshot && modeSnapshot == InlineBrainMode.SEGMENT_SELECTED ->
                        furnitureFitManager?.segmentSelectedInstancesAsync(bitmap, selectedPinsSnapshot, callback)
                    else ->
                        furnitureFitManager?.segmentWithDetectionsAsync(bitmap, callback)
                }
            } finally {
                imageProxy.close()
            }
        }

        try {
            // Live preview is identify-only. During selected segmentation the camera keeps
            // analyzing but preview stays off so transparent mask pixels reveal the 3D room.
            if (shouldShowInlineBrainCameraPreview()) {
                val preview = Preview.Builder()
                    .setTargetResolution(analysisSize)
                    .setTargetRotation(displayRotationForCameraX())
                    .build()
                    .also { it.setSurfaceProvider(brainCameraPreview.surfaceProvider) }
                boundPreview = preview
                provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis)
            } else {
                provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, analysis)
            }
            brainAcceptingUpdates = true
            updateInlineBrainCameraPreviewVisibility()
            LogUtil.d(
                TAG,
                "Inline brain: CameraX bound preview=${shouldShowInlineBrainCameraPreview()} mode=$inlineBrainMode generation=$generation",
            )
        } catch (e: Exception) {
            brainAcceptingUpdates = false
            hideBrainProgress()
            setBrainButtonActive(false)
            updateInlineBrainCameraPreviewVisibility()
            LogUtil.e(TAG, "Inline brain camera bind failed", e)
            Toast.makeText(this, getString(R.string.smartypants_camera_error, e.message ?: ""), Toast.LENGTH_SHORT).show()
            CrashReporter.report(this, e, "GLB room inline brain camera bind")
        }
    }

    private fun applyInlineBrainResult(result: SegmentationResult?) {
        hideBrainProgress()
        val mask = result?.mask
        val detections = result?.detections ?: emptyList()
        if (inlineBrainFullVideoEnabled) {
            when (inlineBrainMode) {
                InlineBrainMode.IDENTIFY -> {
                    brainDetectionOverlayView.setMaskAndDetections(
                        mask = null,
                        dets = detections,
                        modelInputSize = result?.inputSize ?: 640,
                        clusters = result?.detectionClusters ?: emptyList(),
                        frameAlignedOverlay = true,
                        sourceWidth = result?.sourceWidth ?: 640,
                        sourceHeight = result?.sourceHeight ?: 640,
                    )
                    brainDetectionOverlayView.setDetectionBoxVisibility(true)
                    brainDetectionOverlayView.setIdentifySelectionState(true, inlineBrainSelectedPins)
                }
                InlineBrainMode.SEGMENT_SELECTED -> {
                    updateInlineBrainCameraPreviewVisibility()
                    brainDetectionOverlayView.setMaskAndDetections(
                        mask = mask,
                        dets = emptyList(),
                        modelInputSize = result?.inputSize ?: 640,
                        frameAlignedOverlay = true,
                        sourceWidth = result?.sourceWidth ?: mask?.width ?: 640,
                        sourceHeight = result?.sourceHeight ?: mask?.height ?: 640,
                    )
                    brainDetectionOverlayView.setDetectionBoxVisibility(false)
                    brainDetectionOverlayView.setIdentifySelectionState(false, inlineBrainSelectedPins)
                }
                InlineBrainMode.DEFAULT_SEGMENT -> Unit
            }
            updateInlineBrainSegmentButton()
        } else {
            brainDetectionOverlayView.setMaskAndDetections(
                mask,
                emptyList(),
                result?.inputSize ?: 640,
                1f,
                null,
                roomHeight,
            )
        }
        LogUtil.i(
            TAG,
            "Inline brain result: mask=${mask != null} alpha=${mask?.hasAlpha()} dets=${detections.size} " +
                "clusters=${result?.detectionClusters?.size ?: 0} mode=$inlineBrainMode " +
                "primary=${result?.primaryDetection?.label}:${result?.primaryDetection?.confidence}",
        )
    }

    private fun stopInlineBrainSegmentation() {
        LogUtil.d(TAG, "Inline brain: stop")
        brainSessionGeneration.incrementAndGet()
        brainAcceptingUpdates = false
        isBrainInferenceRunning.set(false)
        inlineBrainMode = InlineBrainMode.DEFAULT_SEGMENT
        inlineBrainFullVideoEnabled = false
        inlineBrainSelectedPins = emptyList()
        hideBrainProgress()
        brainDetectionOverlay.visibility = View.GONE
        brainDetectionOverlayView.setMaskAndDetections(null, emptyList())
        brainDetectionOverlayView.setDetectionBoxVisibility(false)
        brainDetectionOverlayView.setIdentifySelectionState(false, emptyList())
        brainSegmentButton?.visibility = View.GONE
        brainFullVideoButton?.visibility = View.GONE
        boundPreview?.setSurfaceProvider(null)
        boundPreview = null
        if (::brainCameraPreview.isInitialized) {
            brainCameraPreview.visibility = View.GONE
        }
        setBrainButtonActive(false)
        inlineBrainArAssistedSizingEnabled = false
        updateTrailingArSizingVisibility()
        try {
            cameraProvider?.unbindAll()
        } catch (_: Exception) {
        }
        cameraProvider = null
        ensureNavigationChromeOnTop()
    }

    private fun setBrainButtonActive(active: Boolean) {
        brainButton?.background = PaafektDrawables.heroButton(active)
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
                    text = getString(R.string.photo_room_loading)
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

    private fun nudgeCameraLeft() {
        webView.evaluateJavascript(
            "if(typeof moveCamera==='function')moveCamera(-8,0);",
            null,
        )
    }

    private fun nudgeCameraRight() {
        webView.evaluateJavascript(
            "if(typeof moveCamera==='function')moveCamera(8,0);",
            null,
        )
    }

    private fun nudgeCameraUp() {
        webView.evaluateJavascript(
            "if(typeof moveCameraUp==='function')moveCameraUp(0.2);",
            null,
        )
    }

    private fun nudgeCameraDown() {
        webView.evaluateJavascript(
            "if(typeof moveCameraUp==='function')moveCameraUp(-0.2);",
            null,
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

        val glbBytes = glbFile.length()
        validateGlbHeader(glbFile)?.let { validationError ->
            val detail = "Invalid room GLB ($validationError) path=${glbFile.absolutePath} size=$glbBytes"
            LogUtil.e(TAG, detail)
            reportWebGlError(detail)
            return
        }

        LogUtil.d(
            TAG,
            "Loading GLB file: path=${glbFile.absolutePath} name=${glbFile.name} size=$glbBytes " +
                "viewer=WebViewAssetLoader extensions=KHR_materials_unlit (no Draco in Android export)",
        )

        // Serve GLB from cache via WebViewAssetLoader instead of inlining base64 in HTML.
        // Portrait photos embed a full JPEG in the GLB; base64 in loadData() often truncates
        // on Android WebView and produces a corrupt buffer that GLTFLoader cannot parse.
        val viewerDir = File(cacheDir, "glb_web_viewer").apply {
            deleteRecursively()
            mkdirs()
        }
        glbViewerCacheDir = viewerDir
        glbFile.inputStream().use { input ->
            File(viewerDir, "room.glb").outputStream().use { output ->
                input.copyTo(output)
                output.flush()
            }
        }
        val cachedGlb = File(viewerDir, "room.glb")
        if (cachedGlb.length() != glbBytes) {
            val detail = "GLB cache copy size mismatch source=$glbBytes cached=${cachedGlb.length()} path=${glbFile.absolutePath}"
            LogUtil.e(TAG, detail)
            reportWebGlError(detail)
            return
        }

        File(viewerDir, "index.html").writeText(generateWebGLHTML())

        val assetLoader = WebViewAssetLoader.Builder()
            .addPathHandler("/assets/", WebViewAssetLoader.AssetsPathHandler(this))
            .addPathHandler("/glb/", WebViewAssetLoader.InternalStoragePathHandler(this, viewerDir))
            .build()

        webView.webViewClient = object : WebViewClient() {
            override fun shouldInterceptRequest(
                view: WebView,
                request: WebResourceRequest,
            ): WebResourceResponse? {
                return assetLoader.shouldInterceptRequest(request.url)
            }

            override fun onPageFinished(view: WebView?, url: String?) {
                super.onPageFinished(view, url)
                LogUtil.d(TAG, "WebView page loaded: $url")
                notifyWebViewViewportChanged()
            }

            override fun onReceivedError(
                view: WebView?,
                request: WebResourceRequest?,
                error: WebResourceError?,
            ) {
                val failingUrl = request?.url?.toString().orEmpty()
                LogUtil.e(TAG, "WebView resource error url=$failingUrl desc=${error?.description}")
            }
        }

        webView.loadUrl("https://appassets.androidplatform.net/glb/index.html")
    }

    /** Android room GLBs are plain glTF 2.0 + JPEG textures (GlbGenerator); no Draco/KTX2. */
    private fun validateGlbHeader(glbFile: File): String? {
        if (glbFile.length() < 12) {
            return "file too small (${glbFile.length()} bytes)"
        }
        glbFile.inputStream().use { input ->
            val header = ByteArray(12)
            if (input.read(header) != 12) {
                return "could not read GLB header"
            }
            val buffer = ByteBuffer.wrap(header).order(ByteOrder.LITTLE_ENDIAN)
            val magic = buffer.int
            val version = buffer.int
            val length = buffer.int
            if (magic != GLB_MAGIC) {
                return "bad magic 0x${magic.toUInt().toString(16)}"
            }
            if (version != GLB_VERSION) {
                return "unsupported version $version"
            }
            if (length.toLong() != glbFile.length()) {
                return "header length $length != file size ${glbFile.length()}"
            }
        }
        return null
    }

    private fun reportWebGlError(message: String) {
        loadingOverlay.visibility = View.GONE
        Toast.makeText(this, message, Toast.LENGTH_LONG).show()
        CrashReporter.report(this, RuntimeException(message), "GLB room — WebGL viewer")
    }

    private fun generateWebGLHTML(): String {
        val isPortrait = photoOrientation == "portrait"

        // Three.js GLB viewer matching iOS GLBRoomView exactly
        return """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        html, body {
            margin: 0;
            padding: 0;
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
            "three": "$VIEWER_THREE_BASE/build/three.module.js",
            "three/addons/": "$VIEWER_THREE_BASE/examples/jsm/"
        }
    }
    </script>
    <script type="module">
        import * as THREE from 'three';
        import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
        import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

        function reportError(prefix, error) {
            const detail = prefix + ': ' + (error?.message || error?.toString?.() || String(error));
            console.error('[GLBViewer]', detail, error);
            if (window.Android) {
                window.Android.onError(detail);
            }
        }

        window.addEventListener('error', (event) => {
            reportError('Viewer script error', event.error || event.message);
        });
        window.addEventListener('unhandledrejection', (event) => {
            reportError('Viewer promise rejection', event.reason);
        });

        const isPortrait = $isPortrait;
        const inferenceRoomWidth = $roomWidth;
        const inferenceRoomHeight = $roomHeight;

        console.log('[GLBViewer] Starting...');

        // Scene setup
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x808080);

        // Camera
        const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 1000);
        camera.position.set(0, 2, 5);

        // Renderer — sized in resizeViewer() once the WebView viewport is known.
        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.domElement.style.width = '100%';
        renderer.domElement.style.height = '100%';
        document.body.appendChild(renderer.domElement);

        // Orbit controls - matching iOS settings exactly
        const controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = false;
        controls.rotateSpeed = 0.7;
        controls.zoomSpeed = 2.5;       // Fast zoom
        controls.enableZoom = true;
        controls.enablePan = true;
        controls.panSpeed = 1.5;
        controls.minDistance = 0.5;
        controls.maxDistance = 20;
        controls.touches = {
            ONE: THREE.TOUCH.ROTATE,
            TWO: THREE.TOUCH.DOLLY_PAN
        };

        // D-pad removed from UI; moveCamera still used if needed — two-finger pan covers walk-through.
        let initialCameraPosition = null;
        let initialControlsTarget = null;
        let roomBoundsForClamping = null;
        let isFlatPhotoMesh = false;
        let flatPhotoWidth = 0;
        let flatPhotoHeight = 0;

        function viewportSize() {
            const visualViewport = window.visualViewport;
            return {
                width: Math.max(visualViewport?.width ?? window.innerWidth, 1),
                height: Math.max(visualViewport?.height ?? window.innerHeight, 1),
                dpr: window.devicePixelRatio || 1,
            };
        }

        function depthAnythingImagePlaneStandoff(width, height) {
            const halfFovRad = Math.PI / 6.0;
            const viewportAspect = Math.max(viewportSize().width / viewportSize().height, 0.01);
            const planeWidth = Math.max(width, 0.05);
            const planeHeight = Math.max(height, 0.05);
            let fitWidth;
            let fitHeight;
            if (!isPortrait) {
                fitWidth = planeWidth / (2 * Math.tan(halfFovRad));
                const verticalHalfFov = Math.atan(Math.tan(halfFovRad) / viewportAspect);
                fitHeight = planeHeight / (2 * Math.tan(verticalHalfFov));
            } else {
                fitHeight = planeHeight / (2 * Math.tan(halfFovRad));
                const horizontalHalfFov = Math.atan(Math.tan(halfFovRad) * viewportAspect);
                fitWidth = planeWidth / (2 * Math.tan(horizontalHalfFov));
            }
            // Cover framing for both orientations (Swift DepthAnythingFlatPhotoCameraFraming parity).
            const fitDistance = Math.min(fitWidth, fitHeight) * 0.98;
            return Math.max(fitDistance, 0.2);
        }

        function updateFlatPhotoProjection() {
            const viewportAspect = Math.max(viewportSize().width / viewportSize().height, 0.01);
            if (!isPortrait) {
                const verticalHalfFov = Math.atan(Math.tan(Math.PI / 6.0) / viewportAspect);
                camera.fov = THREE.MathUtils.radToDeg(2 * verticalHalfFov);
            } else {
                camera.fov = 60;
            }
            camera.aspect = viewportAspect;
            camera.updateProjectionMatrix();
        }

        function backCenterInsetFraction(depth) {
            const t = Math.min(1, Math.max(0, depth / 6.0));
            return 0.035 + 0.065 * t;
        }

        function applyBackCenterCamera(boxWorld) {
            const depth = Math.max(boxWorld.max.z - boxWorld.min.z, 0.1);
            const insetFromBack = Math.max(depth * backCenterInsetFraction(depth), 0.05);
            const centerY = (boxWorld.min.y + boxWorld.max.y) * 0.5;
            const cameraZ = boxWorld.max.z - insetFromBack;
            const targetZ = boxWorld.min.z;

            camera.position.set(0, centerY + 0.4, cameraZ);
            controls.target.set(0, centerY, targetZ);
            camera.lookAt(controls.target);
            controls.update();

            initialCameraPosition = camera.position.clone();
            initialControlsTarget = controls.target.clone();

            const roomWidth = boxWorld.max.x - boxWorld.min.x;
            controls.maxDistance = Math.max(roomWidth, depth) * 2.0;
            roomBoundsForClamping = {
                minX: boxWorld.min.x + 0.05,
                maxX: boxWorld.max.x - 0.05,
                minY: boxWorld.min.y + 0.05,
                maxY: boxWorld.max.y - 0.05,
                minZ: boxWorld.min.z + 0.05,
                maxZ: boxWorld.max.z - 0.02
            };
            console.log('[GLBViewer] Back-center camera inset=', insetFromBack.toFixed(2),
                'posZ=', cameraZ.toFixed(2), 'targetZ=', targetZ.toFixed(2));
        }

        function applyFlatPhotoCamera() {
            if (!isFlatPhotoMesh || flatPhotoWidth <= 0 || flatPhotoHeight <= 0) return;
            updateFlatPhotoProjection();
            const planeWidth = inferenceRoomWidth > 0.05 ? inferenceRoomWidth : flatPhotoWidth;
            const planeHeight = inferenceRoomHeight > 0.05 ? inferenceRoomHeight : flatPhotoHeight;
            const standoff = depthAnythingImagePlaneStandoff(planeWidth, planeHeight);
            const camY = planeHeight * 0.5;
            // Photographer viewpoint: plane at z≈0, camera on −Z (iOS getCameraForDepthAnythingImagePlane).
            camera.position.set(0, camY, -standoff);
            controls.target.set(0, camY, 0);
            camera.lookAt(controls.target);
            controls.update();
            initialCameraPosition = camera.position.clone();
            initialControlsTarget = controls.target.clone();
            roomBoundsForClamping = {
                minX: -planeWidth * 0.5 + 0.05,
                maxX: planeWidth * 0.5 - 0.05,
                minY: 0.05,
                maxY: planeHeight - 0.05,
                minZ: -Math.max(standoff * 1.5, 8.0),
                maxZ: -0.02
            };
            console.log('[GLBViewer] Flat photo camera standoff=', standoff.toFixed(2),
                'plane=', planeWidth.toFixed(2), 'x', planeHeight.toFixed(2),
                'posZ=', (-standoff).toFixed(2));
        }

        function scheduleCameraFraming() {
            requestAnimationFrame(() => {
                requestAnimationFrame(() => {
                    if (isFlatPhotoMesh) {
                        applyFlatPhotoCamera();
                    }
                    resizeViewer();
                });
            });
        }

        // D-pad / Splat parity: walk on XZ, vertical Y (same as iOS GLBRoomView).
        window.moveCamera = function(dx, dy) {
            const moveSpeed = 0.03;
            let newX = camera.position.x + dx * moveSpeed;
            let newZ = camera.position.z + dy * moveSpeed;
            if (roomBoundsForClamping) {
                const marginSide = 0.05;
                const marginBack = 0.02;
                newX = Math.max(roomBoundsForClamping.minX + marginSide,
                       Math.min(roomBoundsForClamping.maxX - marginSide, newX));
                newZ = Math.max(roomBoundsForClamping.minZ + marginSide,
                       Math.min(roomBoundsForClamping.maxZ - marginBack, newZ));
            }
            const actualDx = newX - camera.position.x;
            const actualDz = newZ - camera.position.z;
            camera.position.x = newX;
            camera.position.z = newZ;
            controls.target.x += actualDx;
            controls.target.z += actualDz;
            controls.update();
        };

        window.moveCameraUp = function(dy) {
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
        };

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
            if (isFlatPhotoMesh) {
                applyFlatPhotoCamera();
                console.log('[GLBViewer] Flat photo camera recentered');
                return;
            }
            if (initialCameraPosition && initialControlsTarget) {
                camera.position.copy(initialCameraPosition);
                controls.target.copy(initialControlsTarget);
                controls.update();
                console.log('[GLBViewer] Camera recentered');
            }
        };

        function resizeViewer() {
            const viewport = viewportSize();
            camera.aspect = viewport.width / viewport.height;
            if (isFlatPhotoMesh) {
                updateFlatPhotoProjection();
            } else {
                camera.updateProjectionMatrix();
            }
            renderer.setPixelRatio(viewport.dpr);
            renderer.setSize(viewport.width, viewport.height, false);
            if (isFlatPhotoMesh && flatPhotoWidth > 0 && flatPhotoHeight > 0) {
                applyFlatPhotoCamera();
            }
            console.log('[GLBViewer] resizeViewer', viewport.width, 'x', viewport.height, 'dpr', viewport.dpr);
        }

        window.resizeViewer = resizeViewer;
        resizeViewer();
        window.addEventListener('resize', resizeViewer);
        if (window.visualViewport) {
            window.visualViewport.addEventListener('resize', resizeViewer);
        }

        // Lighting
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        scene.add(ambientLight);

        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(5, 10, 5);
        scene.add(directionalLight);

        // Load GLB from same-origin cache URL (avoids huge inline base64 in HTML).
        const loader = new GLTFLoader();
        const glbUrl = 'room.glb';
        console.log('[GLBViewer] Loading GLB from', glbUrl);
        loader.load(glbUrl, function(gltf) {
            try {
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

                const roomWidth = size.x;
                const roomHeight = size.y;
                const roomDepth = size.z;
                isFlatPhotoMesh = roomDepth < 0.05;

                if (isFlatPhotoMesh) {
                    flatPhotoWidth = roomWidth;
                    flatPhotoHeight = roomHeight;
                } else {
                    const boxWorld = new THREE.Box3().setFromObject(model);
                    applyBackCenterCamera(boxWorld);
                }

                console.log('[GLBViewer] Room size:', roomWidth.toFixed(2), 'x', roomHeight.toFixed(2), 'x', roomDepth.toFixed(2));
                console.log('[GLBViewer] Camera:', camera.position.x.toFixed(2), camera.position.y.toFixed(2), camera.position.z.toFixed(2),
                    'lookAt', controls.target.x.toFixed(2), controls.target.y.toFixed(2), controls.target.z.toFixed(2));

                if (window.Android) {
                    window.Android.onLoaded();
                }
                scheduleCameraFraming();
            } catch (error) {
                reportError('GLB viewer setup failed', error);
            }

        }, undefined, function(error) {
            reportError('GLB load failed', error);
        });

        // Animation loop
        function animate() {
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }
        animate();

        console.log('[GLBViewer] Viewer ready');
    </script>
</body>
</html>
        """.trimIndent()
    }

    private fun showSaveDialog() {
        PaafektDialogs.showNameRoomDialog(this) { typedName, dismiss ->
            if (typedName.isNotEmpty() && !ModelManager.isRoomNameAvailable(this, typedName)) {
                Toast.makeText(this, getString(R.string.home_room_name_duplicate), Toast.LENGTH_SHORT).show()
                return@showNameRoomDialog
            }
            val name = if (typedName.isEmpty()) {
                ModelManager.findAvailableRoomName(this, RoomDisplayName.myRoomWithTimestamp())
            } else {
                typedName
            }
            saveRoom(name)
            dismiss()
        }
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
                val glbSnapshot = RoomFolderMetadata.snapshotPreservingCalibrationFields(
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

                PaafektSnackbar.showRoomSaved(rootLayout, name)
                LogUtil.d(TAG, "Room saved: $name at ${savedRoomFolder.absolutePath}")

                rootLayout.postDelayed({
                    val intent = Intent(this, ContentActivity::class.java)
                    intent.flags = Intent.FLAG_ACTIVITY_CLEAR_TOP or Intent.FLAG_ACTIVITY_NEW_TASK
                    startActivity(intent)
                    finish()
                }, 1200)
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
                ensureNavigationChromeOnTop()
                notifyWebViewViewportChanged()
                LogUtil.d(TAG, "WebGL viewer reported loaded")
            }
        }

        @JavascriptInterface
        fun onError(message: String) {
            runOnUiThread {
                loadingOverlay.visibility = View.GONE
                LogUtil.e(TAG, "WebGL error: $message")
                Toast.makeText(this@GLBRoomActivity, message, Toast.LENGTH_LONG).show()
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
        if (::brainDetectionOverlay.isInitialized && brainDetectionOverlay.visibility == View.VISIBLE) {
            stopInlineBrainSegmentation()
            return
        }
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

    override fun onResume() {
        super.onResume()
        enterImmersiveMode()
        notifyWebViewViewportChanged()
    }

    override fun onWindowFocusChanged(hasFocus: Boolean) {
        super.onWindowFocusChanged(hasFocus)
        if (hasFocus) {
            enterImmersiveMode()
            notifyWebViewViewportChanged()
        }
    }

    override fun onBackPressed() {
        handleBackNavigation()
    }

    override fun onDestroy() {
        immersiveChrome.destroy()
        stopInlineBrainSegmentation()
        furnitureFitManager?.close()
        furnitureFitManager = null
        cameraExecutor.shutdown()
        glbViewerCacheDir?.deleteRecursively()
        glbViewerCacheDir = null
        webView.destroy()
        requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_UNSPECIFIED
        super.onDestroy()
    }
}
