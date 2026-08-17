package com.furnit.android

import android.annotation.SuppressLint
import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.content.res.ColorStateList
import android.content.res.Configuration
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.net.Uri
import android.os.Bundle
import android.os.Environment
import com.furnit.android.utils.CrashReporter
import com.furnit.android.utils.FurnitureFitFrameUsability
import com.furnit.android.utils.FurnitureFitThermalCadence
import com.furnit.android.theme.PaafektColors
import com.furnit.android.theme.PaafektDialogs
import com.furnit.android.theme.PaafektDrawables
import com.furnit.android.theme.PaafektSavingRoomOverlay
import com.furnit.android.theme.PaafektSnackbar
import com.furnit.android.theme.PaafektFirstRunCoachMarkController
import com.furnit.android.theme.PaafektHintController
import com.furnit.android.theme.PaafektImmersiveChromeController
import com.furnit.android.theme.PaafektImmersiveSummonedToolbar
import com.furnit.android.theme.ImmersiveSummonedToolbarHolder
import com.furnit.android.theme.PaafektHintViews
import com.furnit.android.theme.PaafektSpace
import com.furnit.android.theme.PaafektViewerToolbar
import com.furnit.android.theme.PlacementIntelligenceCardView
import com.furnit.android.theme.PlacementIntelligenceCardMapper
import com.furnit.android.utils.LogUtil
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
import com.furnit.android.ar.ArSupportChecker
import com.furnit.android.ar.FurnitureFitArCameraController
import com.furnit.android.ar.rotateToMatchLockedRoomPhoto
import com.furnit.android.models.ModelManager
import com.furnit.android.models.roomintelligence.BitmapStraightSrgbExtractor
import com.furnit.android.models.roomintelligence.FurnitureAestheticProfile
import com.furnit.android.models.roomintelligence.FurnitureDimensions
import com.furnit.android.models.roomintelligence.RoomDimensions
import com.furnit.android.models.roomintelligence.RoomIntelligenceEngine
import com.furnit.android.models.roomintelligence.RoomIntelligenceStatus
import com.furnit.android.models.roomintelligence.RoomPaletteSampler
import com.furnit.android.models.roomintelligence.StraightSrgbColor
import com.furnit.android.models.roomintelligence.SurfacePalette
import com.furnit.android.services.DepthAnythingRoomMeasurer
import com.furnit.android.services.FurnitureFitManager
import com.furnit.android.services.RoomArtifactPromoter
import com.furnit.android.services.RoomMeasurementDisplay
import com.furnit.android.services.SegmentationResult
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
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
import java.util.concurrent.atomic.AtomicReference
import kotlin.math.roundToInt

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
        const val EXTRA_ROOM_DEPTH = "room_depth"
        const val EXTRA_IS_PREVIEW = "is_preview"
        const val EXTRA_PHOTO_ORIENTATION = "photo_orientation"
    }

    private lateinit var webView: WebView
    private lateinit var loadingOverlay: FrameLayout
    private lateinit var rootLayout: FrameLayout
    private lateinit var bottomControls: FrameLayout
    private lateinit var cameraDpadOverlay: FrameLayout
    private lateinit var immersiveRestingChrome: FrameLayout
    private lateinit var windowInsetsController: WindowInsetsControllerCompat
    private var immersiveBackButton: View? = null
    private var immersiveSummonButton: View? = null
    private var immersiveFitFab: LinearLayout? = null
    private var immersiveSaveFab: LinearLayout? = null
    private var immersivePersistentActions: PaafektViewerToolbar.PersistentPrimaryActionsHolder? = null
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
    private data class PendingInlineBrainFrame(
        val bitmap: Bitmap,
        val generation: Int,
    )
    /** Swift parity: at most one copied newest CameraX frame in full-video live modes. */
    private val pendingLatestInlineBrainFrame = AtomicReference<PendingInlineBrainFrame?>(null)
    private val brainSessionGeneration = AtomicInteger(0)
    private var brainAcceptingUpdates = false
    private val inlineBrainThermalCadence = FurnitureFitThermalCadence(logTag = "GLBRoomInlineBrainThermal")
    private var summonedToolbar: ImmersiveSummonedToolbarHolder? = null
    private var inlineBrainArAssistedSizingEnabled = false
    private var inlineBrainArCameraController: FurnitureFitArCameraController? = null
    private var inlineBrainFurnitureWidthMeters: Float? = null
    private var inlineBrainFurnitureHeightMeters: Float? = null
    private lateinit var placementIntelligenceCard: PlacementIntelligenceCardView
    private var inlineBrainRoomPalette: SurfacePalette = SurfacePalette.EMPTY
    private var inlineBrainFurnitureColor: StraightSrgbColor? = null
    private var inlineBrainFurnitureLabel: String? = null
    private var inlineBrainHasSegmentedFurniture = false
    private var lastInlineBrainColorSampleMs: Long = 0L
    private var roomPaletteLoadJob: Job? = null
    @Volatile private var inlineBrainMode: InlineBrainMode = InlineBrainMode.DEFAULT_SEGMENT
    @Volatile private var inlineBrainActionStartedAtNanos: Long? = null
    @Volatile private var inlineBrainFirstMaskLogged = false
    @Volatile private var inlineBrainFullVideoEnabled = false
    @Volatile private var inlineBrainSelectedPins: List<DetectionResult> = emptyList()
    private val inlineBrainPinMissingFrameCounts = mutableMapOf<String, Int>()
    private var glbPath: String? = null
    private var glbViewerCacheDir: File? = null
    private var roomName: String = ""
    private var roomId: String? = null
    private var isPreviewMode: Boolean = false
    private var photoOrientation: String = "portrait"

    private var roomWidth: Float = 4.0f
    private var roomHeight: Float = 3.0f
    private var roomDepth: Float = 4.5f
    private var hasRoomWidthSignal: Boolean = false
    private var hasRoomHeightSignal: Boolean = false
    private var hasRoomDepthSignal: Boolean = false
    private var isFlatPhotoRoomMesh: Boolean = false
    private var hasCalculatedRoomMeasurements: Boolean = false
    private var roomDimsApproach: String? = null
    private var measurementPillView: TextView? = null
    @Volatile private var isMeasuringRoomDimensions: Boolean = false
    private var saveRoomJob: Job? = null

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
        roomName = intent.getStringExtra(EXTRA_ROOM_NAME) ?: getString(R.string.room_viewer_title)
        roomId = intent.getStringExtra(EXTRA_ROOM_ID)
        isPreviewMode = intent.getBooleanExtra(EXTRA_IS_PREVIEW, false)
        hasRoomWidthSignal = intent.hasExtra(EXTRA_ROOM_WIDTH)
        hasRoomHeightSignal = intent.hasExtra(EXTRA_ROOM_HEIGHT)
        hasRoomDepthSignal = intent.hasExtra(EXTRA_ROOM_DEPTH)
        roomWidth = intent.getFloatExtra(EXTRA_ROOM_WIDTH, RoomDefaults.widthMeters(this))
        roomHeight = intent.getFloatExtra(EXTRA_ROOM_HEIGHT, RoomDefaults.heightMeters(this))
        roomDepth = intent.getFloatExtra(EXTRA_ROOM_DEPTH, RoomDefaults.depthMeters(this))
        photoOrientation = intent.getStringExtra(EXTRA_PHOTO_ORIENTATION) ?: "portrait"
        loadRoomMetadataFromFolder()

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
            id = R.id.saved_room_viewport
            contentDescription = getString(R.string.photo_room_loading)
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
                        if (immersiveChrome.phase == PaafektImmersiveChromeController.Phase.RESTING) {
                            if (inlineBrainFullVideoEnabled &&
                                inlineBrainMode == InlineBrainMode.SEGMENT_SELECTED
                            ) {
                                toggleInlineBrainSegmentMode()
                                return false
                            }
                            if (!(inlineBrainFullVideoEnabled && inlineBrainMode == InlineBrainMode.IDENTIFY)) {
                                immersiveChrome.summon()
                            }
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

        hintController = PaafektHintController(rootLayout)
        firstRunCoachController = PaafektFirstRunCoachMarkController(rootLayout)

        // Bottom controls (summoned chrome)
        bottomControls = createBottomControls()
        rootLayout.addView(bottomControls, FrameLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT
        ).apply { gravity = Gravity.BOTTOM })

        // Top-left camera D-pad, restored for iOS parity (PaafektViewerCameraDPad).
        // Deliberately NOT part of the immersive resting chrome and never auto-hidden:
        // iOS keeps it visible at all times because stepping the camera is a continuous
        // task that should not require summoning the toolbar first. It calls the same JS
        // entry points iOS drives through its WebGLCameraMove* notifications.
        cameraDpadOverlay = createCameraDPadOverlay()
        rootLayout.addView(
            cameraDpadOverlay,
            FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT,
            ),
        )

        immersiveRestingChrome = createImmersiveRestingChrome()
        rootLayout.addView(
            immersiveRestingChrome,
            FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT,
            ),
        )

        val persistentActions = PaafektViewerToolbar.createPersistentPrimaryActionsRow(
            this@GLBRoomActivity,
            showFit = true,
            showSave = isPreviewMode,
            fitLabel = getString(R.string.room_viewer_immersive_fit_short),
            saveLabel = getString(R.string.common_save),
            onFit = {
                onMorphingPrimaryFitPressed()
            },
            onSave = {
                immersiveChrome.noteChromeInteraction()
                showSaveDialog()
            },
        )
        immersivePersistentActions = persistentActions
        immersiveFitFab = persistentActions.fitButton
        immersiveSaveFab = persistentActions.saveButton
        persistentActions.container.elevation = 40f
        rootLayout.addView(persistentActions.container)

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
        rootLayout.addView(
            brainDetectionOverlay,
            FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT,
            ),
        )

        placementIntelligenceCard = PlacementIntelligenceCardView(this).apply {
            layoutParams = PlacementIntelligenceCardView.viewerLayoutParams(this@GLBRoomActivity)
            elevation = 45f
        }
        rootLayout.addView(placementIntelligenceCard)
        loadPlacementIntelligenceRoomPalette()

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
        if (!isPreviewMode) {
            warmRoomMeasurementInBackgroundIfNeeded()
        }
        // Match iOS: preload RTMDet when the room opens so the first Fit tap is camera+frame,
        // not LiteRT model/interpreter setup.
        preloadFurnitureFitModelInBackground()
    }

    /**
     * iOS hosts call `RTMDetModelService.ensureModelLoaded()` on room appear.
     * Load only here—not at application launch—then recheck when Fit is activated.
     */
    private fun preloadFurnitureFitModelInBackground() {
        lifecycleScope.launch(Dispatchers.IO) {
            val existingManager = furnitureFitManager
            val manager = existingManager ?: FurnitureFitManager(this@GLBRoomActivity)
            val ok = manager.initializeAuto()
            if (!ok) {
                if (existingManager == null) manager.close()
                LogUtil.w(TAG, "Inline brain: RTMDet preload failed")
                return@launch
            }
            if (isDestroyed) {
                if (existingManager == null) manager.close()
                releaseFurnitureFitResourcesForViewerIfNeeded()
                return@launch
            }
            withContext(Dispatchers.Main) {
                if (isDestroyed) {
                    if (furnitureFitManager !== manager) manager.close()
                    releaseFurnitureFitResourcesForViewerIfNeeded()
                    return@withContext
                }
                if (furnitureFitManager == null) {
                    furnitureFitManager = manager
                }
                LogUtil.d(TAG, "Inline brain: RTMDet preloaded while room open")
            }
        }
    }

    /** Swift's saved-room GLB viewer releases RTMDet; its generated-room preview retains it. */
    private fun releaseFurnitureFitResourcesForViewerIfNeeded() {
        if (!isPreviewMode) {
            FurnitureFitManager.releaseSharedResourcesAsync()
        }
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
        if (::bottomControls.isInitialized) {
            bottomControls.setPadding(side, 0, side, bars.bottom + PaafektSpace.lg(this))
        }
        immersiveBackButton?.layoutParams?.let { lp ->
            if (lp is FrameLayout.LayoutParams) {
                lp.topMargin = bars.top + PaafektSpace.md(this)
                lp.marginStart = bars.left + PaafektSpace.xl(this)
                immersiveBackButton?.layoutParams = lp
            }
        }
        immersiveFitFab?.layoutParams?.let { lp ->
            if (lp is FrameLayout.LayoutParams) {
                lp.bottomMargin = bars.bottom + PaafektSpace.lg(this)
                immersiveFitFab?.layoutParams = lp
            }
        }
        PaafektViewerToolbar.updatePersistentPrimaryActionsInsets(immersivePersistentActions, bars.bottom)
        if (::placementIntelligenceCard.isInitialized) {
            PlacementIntelligenceCardView.updateViewerInsets(placementIntelligenceCard, bars.bottom)
        }
        immersiveSummonButton?.layoutParams?.let { lp ->
            if (lp is FrameLayout.LayoutParams) {
                lp.bottomMargin = bars.bottom + PaafektSpace.lg(this)
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
                contentDescription = getString(R.string.common_back)
            }
            immersiveBackButton = back
            addView(
                back,
                FrameLayout.LayoutParams(
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                ).apply {
                    gravity = Gravity.START or Gravity.TOP
                    topMargin = PaafektSpace.md(this@GLBRoomActivity)
                    marginStart = PaafektSpace.xl(this@GLBRoomActivity)
                },
            )

            val measurementPill = TextView(this@GLBRoomActivity).apply {
                text = restingMeasurementPillText()
                textSize = 12f
                setTextColor(PaafektColors.textSecondary)
                setPadding(dpToPx(12), dpToPx(8), dpToPx(12), dpToPx(8))
                background = PaafektDrawables.hintChip()
                visibility = if (isPreviewMode) View.GONE else View.VISIBLE
            }
            measurementPillView = measurementPill
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

            val summonQuiet = PaafektViewerToolbar.createQuietSummonButton(
                this@GLBRoomActivity,
                getString(R.string.room_viewer_immersive_show_controls),
            ) {
                immersiveChrome.summon()
            }
            immersiveSummonButton = summonQuiet
            addView(
                summonQuiet,
                PaafektViewerToolbar.quietSummonButtonLayoutParams(this@GLBRoomActivity),
            )
        }
    }

    private fun isFlatPhotoRoom(): Boolean = isFlatPhotoRoomMesh

    private fun restingMeasurementPillText(): String {
        val text = RoomMeasurementDisplay.restingPillText(
            width = roomWidth.takeIf { hasRoomWidthSignal } ?: 0f,
            height = roomHeight.takeIf { hasRoomHeightSignal } ?: 0f,
            depth = roomDepth.takeIf { hasRoomDepthSignal } ?: 0f,
            emphasizeHeight = isFlatPhotoRoomMesh,
            approximateHeightFormatter = { heightMeters ->
                getString(R.string.approximate_room_height, heightMeters)
            },
            widthDepthFormatter = { widthMeters, depthMeters ->
                getString(R.string.room_dimensions_width_depth, widthMeters, depthMeters)
            },
        )
        return text ?: roomName.takeIf { it.isNotBlank() } ?: getString(R.string.room_viewer_title)
    }

    private fun refreshMeasurementPill() {
        measurementPillView?.text = restingMeasurementPillText()
    }

    private fun loadRoomMetadataFromFolder() {
        val folder = glbPath?.let { File(it).parentFile } ?: return
        val snapshot = RoomFolderMetadata.readFromFolder(folder) ?: return
        photoOrientation = snapshot.normalizedOrientation()
        snapshot.roomWidth?.takeIf { it > 0.05f && it.isFinite() }?.let {
            roomWidth = it
            hasRoomWidthSignal = true
        }
        snapshot.roomHeight?.takeIf { it > 0.05f && it.isFinite() }?.let {
            roomHeight = it
            hasRoomHeightSignal = true
        }
        snapshot.roomDepth?.takeIf { it > 0.05f && it.isFinite() }?.let {
            roomDepth = it
            hasRoomDepthSignal = true
        }
        roomDimsApproach = snapshot.roomDimsApproach
        isFlatPhotoRoomMesh = snapshot.type == "photo" ||
            snapshot.roomDimsApproach == "depth_anything_metric" ||
            roomDepth < 0.05f
        hasCalculatedRoomMeasurements = snapshot.roomDimsApproach == "depth_anything_metric"
        LogUtil.d(
            TAG,
            "Canonical room metadata orientation=$photoOrientation preview=$isPreviewMode " +
                "projection=${snapshot.depthMeshProjectionVersion} " +
                "completedBackground=${snapshot.depthMeshHasCompletedBackground} " +
                "continuous=${snapshot.depthMeshUsesContinuousSurface}",
        )
    }

    private fun warmRoomMeasurementInBackgroundIfNeeded() {
        if (isPreviewMode || hasCalculatedRoomMeasurements || isMeasuringRoomDimensions) return
        val sourcePhoto = resolveSourcePhotoFile() ?: return
        val sourcePhotoUri = resolveSourcePhotoUri()
        lifecycleScope.launch {
            isMeasuringRoomDimensions = true
            val measured = withContext(Dispatchers.Default) {
                DepthAnythingRoomMeasurer.measureFromFile(
                    this@GLBRoomActivity,
                    sourcePhoto,
                    sourcePhotoUri,
                )
            }
            isMeasuringRoomDimensions = false
            if (measured.measured) {
                applyMeasuredDimensions(measured, persist = true)
            }
        }
    }

    private fun resolveSourcePhotoFile(): File? {
        val folder = glbPath?.let { File(it).parentFile } ?: return null
        listOf("source_photo.jpg", "source_photo.png", "front_wall.png").forEach { name ->
            val candidate = File(folder, name)
            if (candidate.exists() && candidate.length() > 0L) return candidate
        }
        return null
    }

    private fun resolveSourcePhotoUri(): Uri? {
        val folder = glbPath?.let { File(it).parentFile } ?: return null
        val uriText = runCatching {
            File(folder, "source_photo_uri.txt").takeIf(File::isFile)?.readText()?.trim()
        }.getOrNull()
        return uriText?.takeIf(String::isNotBlank)?.let(Uri::parse)
    }

    private fun applyMeasuredDimensions(measured: DepthAnythingRoomMeasurer.Result, persist: Boolean) {
        roomWidth = measured.width
        roomHeight = measured.height
        roomDepth = measured.depth
        hasRoomWidthSignal = true
        hasRoomHeightSignal = true
        hasRoomDepthSignal = true
        hasCalculatedRoomMeasurements = measured.measured
        roomDimsApproach = measured.source
        isFlatPhotoRoomMesh = true
        refreshMeasurementPill()
        refreshPlacementIntelligence()
        if (persist) {
            persistMeasuredDimensionsToFolder(measured)
        }
    }

    private fun persistMeasuredDimensionsToFolder(measured: DepthAnythingRoomMeasurer.Result) {
        val folder = glbPath?.let { File(it).parentFile } ?: return
        val existing = RoomFolderMetadata.readFromFolder(folder)
        val snapshot = RoomFolderMetadata.snapshotPreservingCalibrationFields(
            folder,
            (existing ?: RoomFolderMetadata.Snapshot()).copy(
                roomWidth = measured.width,
                roomHeight = measured.height,
                roomDepth = measured.depth,
                roomDimsApproach = measured.source,
                roomSceneWidth = measured.width,
                roomSceneHeight = measured.height,
                roomSceneDepth = measured.depth,
            ),
        )
        RoomFolderMetadata.writeToFolder(folder, snapshot)
        val metadataFile = File(folder, "metadata.txt")
        if (metadataFile.exists()) {
            val lines = metadataFile.readLines().mapNotNull { line ->
                when {
                    line.startsWith("roomWidth=") -> "roomWidth=${measured.width}"
                    line.startsWith("roomHeight=") -> "roomHeight=${measured.height}"
                    line.startsWith("roomDepth=") -> "roomDepth=${measured.depth}"
                    else -> line
                }
            }.toMutableList()
            if (lines.none { it.startsWith("roomDepth=") }) {
                lines += "roomDepth=${measured.depth}"
            }
            metadataFile.writeText(lines.joinToString("\n") + "\n")
        }
    }

    private fun startAsyncRoomMeasurementForRuler(onComplete: () -> Unit) {
        if (isMeasuringRoomDimensions) return
        val sourcePhoto = resolveSourcePhotoFile()
        if (sourcePhoto == null) {
            onComplete()
            return
        }
        lifecycleScope.launch {
            isMeasuringRoomDimensions = true
            loadingOverlay.visibility = View.VISIBLE
            val measured = withContext(Dispatchers.Default) {
                DepthAnythingRoomMeasurer.measureFromFile(this@GLBRoomActivity, sourcePhoto)
            }
            isMeasuringRoomDimensions = false
            loadingOverlay.visibility = View.GONE
            if (measured.measured) {
                applyMeasuredDimensions(measured, persist = true)
            }
            onComplete()
        }
    }

    private fun refreshImmersiveChromeVisibility(animate: Boolean = true) {
        if (!::immersiveRestingChrome.isInitialized) return
        immersiveChrome.applyPhase(
            this,
            restingViews = listOf(immersiveRestingChrome),
            summonedViews = listOf(bottomControls),
            animate = animate,
        )
        updateSummonedToolbarState()
        ensureNavigationChromeOnTop()
    }

    /** Keep back / title / recenter / bottom brain+camera above the WebView and brain overlay. */
    private fun ensureNavigationChromeOnTop() {
        bottomControls.elevation = 37f
        if (::cameraDpadOverlay.isInitialized) {
            cameraDpadOverlay.elevation = 39f
        }
        if (::immersiveRestingChrome.isInitialized) {
            immersiveRestingChrome.elevation = 36f
        }
        immersiveFitFab?.elevation = 40f
        rootLayout.bringChildToFront(bottomControls)
        if (::immersiveRestingChrome.isInitialized) {
            rootLayout.bringChildToFront(immersiveRestingChrome)
        }
        if (::cameraDpadOverlay.isInitialized) {
            rootLayout.bringChildToFront(cameraDpadOverlay)
        }
        immersivePersistentActions?.container?.let { rootLayout.bringChildToFront(it) }
        if (::brainProgressOverlay.isInitialized && brainProgressOverlay.visibility == View.VISIBLE) {
            rootLayout.bringChildToFront(brainProgressOverlay)
        }
        hintController.bringToFront()
        firstRunCoachController.bringToFront()
    }

    private fun dpToPx(dp: Int): Int {
        return (dp * resources.displayMetrics.density).toInt()
    }

    private fun createDpadCircleButton(
        label: String,
        testName: String,
        onClick: () -> Unit,
    ): TextView {
        return TextView(this).apply {
            text = label
            contentDescription = testName
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

    /**
     * Top-left arrow cluster. Mirrors iOS `PaafektViewerCameraDPad`: left / (up over down) /
     * right, 44dp circles at 50% black, anchored top-start. The JS calls below are the exact
     * ones iOS issues from `GLBRoomView.nudgeGLBCamera*`, so both platforms step the camera
     * by the same amount.
     */
    private fun createCameraDPadOverlay(): FrameLayout {
        val topInset = if (photoOrientation == "landscape") dpToPx(12) else dpToPx(110)
        return FrameLayout(this).apply {
            // The overlay fills the screen so the cluster can be positioned inside it, but it
            // must not swallow drags meant for OrbitControls — only the buttons take touches.
            isClickable = false
            isFocusable = false

            val cluster = LinearLayout(this@GLBRoomActivity).apply {
                orientation = LinearLayout.HORIZONTAL
                gravity = Gravity.CENTER_VERTICAL
            }

            cluster.addView(createDpadCircleButton("←", "camera_dpad_left") { nudgeCameraLeft() })

            val verticalPad = LinearLayout(this@GLBRoomActivity).apply {
                orientation = LinearLayout.VERTICAL
                val pad = dpToPx(8)
                setPadding(pad, 0, pad, 0)
            }
            verticalPad.addView(createDpadCircleButton("↑", "camera_dpad_up") { nudgeCameraUp() })
            verticalPad.addView(createDpadCircleButton("↓", "camera_dpad_down") { nudgeCameraDown() })
            cluster.addView(verticalPad)

            cluster.addView(createDpadCircleButton("→", "camera_dpad_right") { nudgeCameraRight() })

            addView(
                cluster,
                FrameLayout.LayoutParams(
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                ).apply {
                    // Top-END, not START: the back button owns the top-start corner and the
                    // D-pad was overlapping it. Matches iOS PaafektViewerCameraDPadOverlay.
                    gravity = Gravity.END or Gravity.TOP
                    topMargin = topInset
                    marginEnd = dpToPx(12)
                },
            )
        }
    }

    private fun nudgeCameraLeft() {
        webView.evaluateJavascript("if(typeof moveCamera==='function')moveCamera(-8,0);", null)
    }

    private fun nudgeCameraRight() {
        webView.evaluateJavascript("if(typeof moveCamera==='function')moveCamera(8,0);", null)
    }

    private fun nudgeCameraUp() {
        webView.evaluateJavascript("if(typeof moveCameraUp==='function')moveCameraUp(0.2);", null)
    }

    private fun nudgeCameraDown() {
        webView.evaluateJavascript("if(typeof moveCameraUp==='function')moveCameraUp(-0.2);", null)
    }

    private fun toggleInlineBrainArAssistedSizing() {
        val requestedEnabled = !inlineBrainArAssistedSizingEnabled
        if (requestedEnabled && !ArSupportChecker.isArCoreSupported(this)) {
            Toast.makeText(this, getString(R.string.furniture_fit_ar_not_supported), Toast.LENGTH_SHORT).show()
            return
        }
        inlineBrainArAssistedSizingEnabled = requestedEnabled
        brainSessionGeneration.incrementAndGet()
        clearInlineBrainMetricDimensions()
        placementIntelligenceCard.clear()
        updateSummonedToolbarState()
        rebindInlineBrainCameraIfActive()
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
        if (isPreviewMode) return
        val showHint = {
            val heightLabel = if (
                isFlatPhotoRoom() &&
                hasRoomWidthSignal &&
                hasRoomHeightSignal &&
                hasRoomDepthSignal
            ) {
                getString(R.string.room_dimensions_whd_near_accurate, roomWidth, roomHeight, roomDepth)
            } else if (hasRoomHeightSignal) {
                getString(R.string.approximate_room_height, roomHeight)
            } else {
                roomName.takeIf { it.isNotBlank() } ?: getString(R.string.room_viewer_title)
            }
            hintController.showText(
                this,
                R.drawable.ic_ruler,
                heightLabel,
                topMarginDp = 52,
            )
        }
        if (hasCalculatedRoomMeasurements) {
            showHint()
            return
        }
        startAsyncRoomMeasurementForRuler(onComplete = showHint)
    }

    private fun updateSummonedToolbarState() {
        val toolbar = summonedToolbar ?: return
        val brainActive = ::brainDetectionOverlay.isInitialized &&
            brainDetectionOverlay.visibility == View.VISIBLE
        immersiveFitFab?.let {
            PaafektViewerToolbar.updateMorphingPrimaryFitButton(
                it,
                resolveMorphingPrimaryAction(),
                inlineBrainSelectedPins.isNotEmpty(),
            )
        }
        toolbar.setFullVideoVisible(brainActive)
        toolbar.setFullVideoActive(inlineBrainFullVideoEnabled)
        toolbar.setArSizingVisible(brainActive)
        toolbar.setArSizingActive(inlineBrainArAssistedSizingEnabled)
        updateInlineBrainSegmentButton()
    }

    private fun createBottomControls(): FrameLayout {
        val holder = PaafektImmersiveSummonedToolbar.createBottomChrome(
            this@GLBRoomActivity,
            onRecenter = {
                immersiveChrome.noteChromeInteraction()
                recenterCamera()
            },
            onRuler = {
                immersiveChrome.noteChromeInteraction()
                showRoomDimensionsHint()
            },
            onPinchHint = {
                immersiveChrome.noteChromeInteraction()
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
            onDisplayAllHelpers = {
                immersiveChrome.noteChromeInteraction()
                showAllGestureHelpers()
            },
            onFullVideo = {
                immersiveChrome.noteChromeInteraction()
                toggleInlineBrainFullVideoMode()
            },
            onArSizing = {
                immersiveChrome.noteChromeInteraction()
                toggleInlineBrainArAssistedSizing()
            },
            onCapture = {
                immersiveChrome.noteChromeInteraction()
                takeScreenshot()
            },
            persistentActionCount = if (isPreviewMode) 2 else 1,
        )
        summonedToolbar = holder
        return holder.root
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
        intent.putExtra("ROOM_DEPTH", roomDepth)
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

    private fun toggleInlineBrainFullVideoMode() {
        if (brainDetectionOverlay.visibility != View.VISIBLE) return
        clearPendingLatestInlineBrainFrame()
        inlineBrainFullVideoEnabled = !inlineBrainFullVideoEnabled
        inlineBrainSelectedPins = emptyList()
        inlineBrainPinMissingFrameCounts.clear()
        inlineBrainHasSegmentedFurniture = false
        inlineBrainFurnitureColor = null
        inlineBrainFurnitureLabel = null
        clearInlineBrainMetricDimensions()
        placementIntelligenceCard.clear()
        brainDetectionOverlayView.resetTransform()
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
            presentFullVideoFurnitureTapHint()
        } else {
            dismissFullVideoFurnitureTapHint()
            showBrainProgress(getString(R.string.smartypants_detecting_furniture))
            presentFullVideoSelectionHelper()
        }
        updateInlineBrainSegmentButton()
        updateSummonedToolbarState()
        ensureNavigationChromeOnTop()
        rebindInlineBrainCameraIfActive()
        LogUtil.d(
            TAG,
            "Inline brain full video toggled: enabled=$inlineBrainFullVideoEnabled mode=$inlineBrainMode",
        )
    }

    private fun shouldShowInlineBrainCameraPreview(): Boolean {
        return !inlineBrainArAssistedSizingEnabled &&
            inlineBrainFullVideoEnabled &&
            inlineBrainMode == InlineBrainMode.IDENTIFY &&
            brainDetectionOverlay.visibility == View.VISIBLE
    }

    private fun updateInlineBrainCameraPreviewVisibility() {
        if (!::brainCameraPreview.isInitialized) return
        brainCameraPreview.visibility = if (shouldShowInlineBrainCameraPreview()) View.VISIBLE else View.GONE
    }

    private fun rebindInlineBrainCameraIfActive() {
        if (brainDetectionOverlay.visibility != View.VISIBLE) return
        if (inlineBrainArAssistedSizingEnabled) {
            startInlineBrainArCamera(brainSessionGeneration.get())
            return
        }
        releaseInlineBrainArCamera()
        val provider = cameraProvider
        if (provider != null) {
            applyInlineBrainCameraBinding(provider, brainSessionGeneration.get())
        } else {
            bindInlineBrainCamera(brainSessionGeneration.get())
        }
    }

    private fun onSegmentationDonePressed() {
        if (inlineBrainFullVideoEnabled && inlineBrainMode == InlineBrainMode.SEGMENT_SELECTED) {
            toggleInlineBrainSegmentMode()
        } else {
            stopInlineBrainSegmentation()
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
        inlineBrainActionStartedAtNanos = android.os.SystemClock.elapsedRealtimeNanos()
        inlineBrainFirstMaskLogged = false
        LogUtil.i(TAG, "Inline brain timing: FIT_START mode=DEFAULT_SEGMENT")
        LogUtil.d(TAG, "Inline brain: start")
        val generation = brainSessionGeneration.incrementAndGet()
        clearPendingLatestInlineBrainFrame()
        inlineBrainFullVideoEnabled = false
        inlineBrainMode = InlineBrainMode.DEFAULT_SEGMENT
        inlineBrainSelectedPins = emptyList()
        inlineBrainPinMissingFrameCounts.clear()
        brainAcceptingUpdates = false
        isBrainInferenceRunning.set(false)
        inlineBrainHasSegmentedFurniture = false
        inlineBrainFurnitureColor = null
        inlineBrainFurnitureLabel = null
        lastInlineBrainColorSampleMs = 0L
        if (::placementIntelligenceCard.isInitialized) {
            placementIntelligenceCard.clear()
        }
        inlineBrainThermalCadence.start(this)
        brainDetectionOverlay.visibility = View.VISIBLE
        updateSummonedToolbarState()
        ensureNavigationChromeOnTop()
        brainDetectionOverlayView.resetTransform()
        brainDetectionOverlayView.setMaskAndDetections(
            mask = null,
            dets = emptyList(),
            frameAlignedOverlay = inlineBrainFullVideoEnabled,
        )
        brainDetectionOverlayView.setDetectionBoxVisibility(inlineBrainFullVideoEnabled)
        brainDetectionOverlayView.setIdentifySelectionState(inlineBrainFullVideoEnabled, inlineBrainSelectedPins)
        val modelAlreadyReady = furnitureFitManager != null || FurnitureFitManager.isSharedBackendReady()
        // When preloaded (iOS parity), skip the multi-second "Loading detection model…" wait UX.
        if (modelAlreadyReady) {
            showBrainProgress(getString(R.string.smartypants_detecting_furniture), 55)
        } else {
            showBrainProgress(getString(R.string.detector_loading_model), 20)
        }
        refreshMorphingPrimaryFitButton()
        updateSummonedToolbarState()
        updateInlineBrainSegmentButton()
        ensureNavigationChromeOnTop()
        presentFullVideoSelectionHelper()

        lifecycleScope.launch {
            val manager = furnitureFitManager ?: withContext(Dispatchers.IO) {
                FurnitureFitManager(this@GLBRoomActivity).takeIf { manager ->
                    manager.initializeAuto()
                }
            }
            if (manager == null) {
                hideBrainProgress()
                refreshMorphingPrimaryFitButton()
                Toast.makeText(this@GLBRoomActivity, getString(R.string.detector_model_unavailable), Toast.LENGTH_SHORT).show()
                return@launch
            }
            furnitureFitManager = manager
            if (!modelAlreadyReady) {
                showBrainProgress(getString(R.string.smartypants_detecting_furniture), 55)
            }
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
            inlineBrainPinMissingFrameCounts.remove(inlineBrainPinKey(current.removeAt(existingIndex)))
        } else {
            current += detection
        }
        inlineBrainSelectedPins = current
        val activeKeys = current.mapTo(mutableSetOf(), ::inlineBrainPinKey)
        inlineBrainPinMissingFrameCounts.keys.retainAll(activeKeys)
        brainDetectionOverlayView.setIdentifySelectionState(true, inlineBrainSelectedPins)
        updateInlineBrainSegmentButton()
    }

    private fun toggleInlineBrainSegmentMode() {
        if (!inlineBrainFullVideoEnabled) return
        if (inlineBrainMode == InlineBrainMode.SEGMENT_SELECTED) {
            inlineBrainMode = InlineBrainMode.IDENTIFY
            inlineBrainHasSegmentedFurniture = false
            inlineBrainFurnitureColor = null
            inlineBrainFurnitureLabel = null
            clearInlineBrainMetricDimensions()
            placementIntelligenceCard.clear()
            brainDetectionOverlayView.resetTransform()
            brainDetectionOverlayView.setMaskAndDetections(
                mask = null,
                dets = emptyList(),
                frameAlignedOverlay = true,
            )
            brainDetectionOverlayView.setDetectionBoxVisibility(true)
            brainDetectionOverlayView.setIdentifySelectionState(true, inlineBrainSelectedPins)
            presentFullVideoFurnitureTapHint()
        } else if (inlineBrainSelectedPins.isNotEmpty()) {
            inlineBrainActionStartedAtNanos = android.os.SystemClock.elapsedRealtimeNanos()
            inlineBrainFirstMaskLogged = false
            LogUtil.i(TAG, "Inline brain timing: SEGMENT_SELECTED_START selected=${inlineBrainSelectedPins.size}")
            inlineBrainMode = InlineBrainMode.SEGMENT_SELECTED
            // Begin every newly selected cutout at its exact CameraX-aligned 1x pose. Incoming
            // segmentation frames preserve subsequent drag and pinch changes.
            brainDetectionOverlayView.resetTransform()
            brainDetectionOverlayView.setDetectionBoxVisibility(false)
            brainDetectionOverlayView.setIdentifySelectionState(false, inlineBrainSelectedPins)
            showBrainProgress(getString(R.string.detector_segmenting_selection))
            // Keep sticky helper visible during segment (same as iOS full-video mode pill).
            presentFullVideoFurnitureTapHint()
        }
        updateInlineBrainSegmentButton()
        updateInlineBrainCameraPreviewVisibility()
        // Keep the existing CameraX Analysis pipeline alive. Rebinding here blocks the UI
        // thread and delays the first selected-segmentation frame; hiding PreviewView is enough.
    }

    private fun resolveMorphingPrimaryAction(): PaafektViewerToolbar.MorphingPrimaryAction {
        val showingFurnitureFit = ::brainDetectionOverlay.isInitialized &&
            brainDetectionOverlay.visibility == View.VISIBLE
        return PaafektViewerToolbar.MorphingPrimaryActionResolver.resolve(
            showingFurnitureFit = showingFurnitureFit,
            showFullVideoWithIdentifications = inlineBrainFullVideoEnabled,
            segmentationModeSegmentSelected = inlineBrainMode == InlineBrainMode.SEGMENT_SELECTED,
            hasSelectedObject = inlineBrainSelectedPins.isNotEmpty(),
        )
    }

    private fun onMorphingPrimaryFitPressed() {
        immersiveChrome.noteChromeInteraction()
        when (resolveMorphingPrimaryAction()) {
            PaafektViewerToolbar.MorphingPrimaryAction.FIT_ENTER ->
                toggleInlineBrainSegmentation()
            PaafektViewerToolbar.MorphingPrimaryAction.FIT_EXIT_ACTIVE ->
                stopInlineBrainSegmentation()
            PaafektViewerToolbar.MorphingPrimaryAction.SEGMENT ->
                toggleInlineBrainSegmentMode()
            PaafektViewerToolbar.MorphingPrimaryAction.DONE ->
                onSegmentationDonePressed()
        }
    }

    private fun updateMorphingPrimaryFitState() {
        val action = resolveMorphingPrimaryAction()
        val segmentEnabled = inlineBrainSelectedPins.isNotEmpty()
        PaafektViewerToolbar.updateMorphingPrimaryFitButton(
            immersiveFitFab,
            action,
            segmentEnabled,
        )
    }

    private fun updateInlineBrainSegmentButton() {
        updateMorphingPrimaryFitState()
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

    private fun inlineBrainPinKey(detection: DetectionResult): String = buildString {
        append(detection.classId)
        append(':').append((detection.x * 1000f).roundToInt())
        append(':').append((detection.y * 1000f).roundToInt())
        append(':').append((detection.w * 1000f).roundToInt())
        append(':').append((detection.h * 1000f).roundToInt())
    }

    /** Swift keeps a selected instance for three missed frames, then replaces/removes pins by IoU. */
    private fun refreshInlineBrainPinsFromCandidates(candidates: List<DetectionResult>) {
        val pins = inlineBrainSelectedPins
        if (pins.isEmpty()) {
            inlineBrainPinMissingFrameCounts.clear()
            return
        }

        val usedCandidateIndices = mutableSetOf<Int>()
        val updatedPins = mutableListOf<DetectionResult>()
        val updatedMissingCounts = inlineBrainPinMissingFrameCounts.toMutableMap()
        var didRemove = false
        for (pin in pins) {
            val pinKey = inlineBrainPinKey(pin)
            var bestIndex = -1
            var bestIou = 0f
            for ((index, candidate) in candidates.withIndex()) {
                if (index in usedCandidateIndices || candidate.classId != pin.classId) continue
                val iou = detectionIoU(candidate, pin)
                if (iou > bestIou) {
                    bestIou = iou
                    bestIndex = index
                }
            }
            if (bestIndex >= 0 && bestIou >= 0.45f) {
                val matched = candidates[bestIndex]
                usedCandidateIndices += bestIndex
                updatedMissingCounts.remove(pinKey)
                updatedMissingCounts.remove(inlineBrainPinKey(matched))
                updatedPins += matched
                continue
            }

            val nextMissCount = (updatedMissingCounts[pinKey] ?: 0) + 1
            if (nextMissCount > 3) {
                updatedMissingCounts.remove(pinKey)
                didRemove = true
            } else {
                updatedMissingCounts[pinKey] = nextMissCount
                updatedPins += pin
            }
        }

        inlineBrainSelectedPins = updatedPins
        val remainingKeys = updatedPins.mapTo(mutableSetOf(), ::inlineBrainPinKey)
        inlineBrainPinMissingFrameCounts.clear()
        inlineBrainPinMissingFrameCounts.putAll(updatedMissingCounts.filterKeys(remainingKeys::contains))
        if (didRemove) updateInlineBrainSegmentButton()
    }

    @SuppressLint("UnsafeOptInUsageError")
    private fun bindInlineBrainCamera(generation: Int) {
        if (inlineBrainArAssistedSizingEnabled) {
            startInlineBrainArCamera(generation)
            return
        }
        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener({
            applyInlineBrainCameraBinding(providerFuture.get(), generation)
        }, ContextCompat.getMainExecutor(this))
    }

    private fun startInlineBrainArCamera(generation: Int) {
        if (!brainAcceptingUpdates && brainDetectionOverlay.visibility != View.VISIBLE) return
        releaseInlineBrainArCamera()
        try {
            boundPreview?.setSurfaceProvider(null)
            boundPreview = null
            cameraProvider?.unbindAll()
            cameraProvider = null

            val controller = FurnitureFitArCameraController(this, cameraExecutor).apply {
                lockedPhotoOrientation = photoOrientation
                roomHeightMetersForFallback = roomHeight.coerceAtLeast(0.1f)
                shouldPostBitmapFrame = {
                        brainAcceptingUpdates &&
                        brainSessionGeneration.get() == generation &&
                        !isBrainInferenceRunning.get() &&
                        !inlineBrainThermalCadence.isPausedForThermalCritical
                }
                onAssistedMeasurementUpdated = {
                    runOnUiThread {
                        if (brainAcceptingUpdates && brainSessionGeneration.get() == generation) {
                            updateInlineBrainMetricDimensions(this)
                        }
                    }
                }
                onBitmapFrame = { bitmap ->
                    processInlineBrainBitmap(bitmap, generation, bitmapAlreadyOriented = true)
                }
            }
            inlineBrainArCameraController = controller
            val showArPreview = inlineBrainFullVideoEnabled &&
                inlineBrainMode == InlineBrainMode.IDENTIFY
            controller.glSurfaceView.layoutParams = if (showArPreview) {
                FrameLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.MATCH_PARENT,
                )
            } else {
                FrameLayout.LayoutParams(1, 1)
            }
            controller.glSurfaceView.alpha = if (showArPreview) 1f else 0.01f
            rootLayout.addView(controller.glSurfaceView, 2.coerceAtMost(rootLayout.childCount))
            controller.onHostResume()
            brainAcceptingUpdates = true
            updateInlineBrainCameraPreviewVisibility()
            LogUtil.d(TAG, "Inline brain: ARCore sizing camera bound generation=$generation")
        } catch (exception: Exception) {
            inlineBrainArAssistedSizingEnabled = false
            releaseInlineBrainArCamera()
            clearInlineBrainMetricDimensions()
            updateSummonedToolbarState()
            LogUtil.e(TAG, "Inline brain ARCore camera bind failed", exception)
            Toast.makeText(
                this,
                getString(R.string.smartypants_camera_error_generic),
                Toast.LENGTH_SHORT,
            ).show()
            bindInlineBrainCamera(generation)
        }
    }

    private fun updateInlineBrainMetricDimensions(controller: FurnitureFitArCameraController) {
        inlineBrainFurnitureWidthMeters = controller.getProvisionalWidthMeters()
            ?.takeIf { it.isFinite() && it > 0f }
            ?: controller.getLastEstimatedWidthMeters()?.takeIf { it.isFinite() && it > 0f }
        inlineBrainFurnitureHeightMeters = controller.getProvisionalHeightMeters()
            ?.takeIf { it.isFinite() && it > 0f }
            ?: controller.getLastEstimatedHeightMeters()?.takeIf { it.isFinite() && it > 0f }
        refreshPlacementIntelligence()
    }

    private fun clearInlineBrainMetricDimensions() {
        inlineBrainFurnitureWidthMeters = null
        inlineBrainFurnitureHeightMeters = null
        refreshPlacementIntelligence()
    }

    private fun loadPlacementIntelligenceRoomPalette() {
        val roomFolder = glbPath?.let { File(it).parentFile } ?: return
        roomPaletteLoadJob?.cancel()
        roomPaletteLoadJob = lifecycleScope.launch {
            val sampledPalette = withContext(Dispatchers.IO) {
                RoomPaletteSampler.sample(roomFolder)
            }
            inlineBrainRoomPalette = sampledPalette
            refreshPlacementIntelligence()
        }
    }

    private fun sampleInlineBrainFurnitureColor(mask: Bitmap) {
        val now = android.os.SystemClock.elapsedRealtime()
        if (now - lastInlineBrainColorSampleMs < 500L) return
        lastInlineBrainColorSampleMs = now
        val generation = brainSessionGeneration.get()
        lifecycleScope.launch {
            val sampledColor = withContext(Dispatchers.Default) {
                BitmapStraightSrgbExtractor.mean(mask)
            }
            if (brainAcceptingUpdates && brainSessionGeneration.get() == generation) {
                inlineBrainFurnitureColor = sampledColor
                refreshPlacementIntelligence()
            }
        }
    }

    private fun refreshPlacementIntelligence() {
        if (!::placementIntelligenceCard.isInitialized || !inlineBrainHasSegmentedFurniture) {
            if (::placementIntelligenceCard.isInitialized) {
                placementIntelligenceCard.clear()
            }
            return
        }
        val roomDimensions = if (
            hasRoomWidthSignal &&
            hasRoomHeightSignal &&
            hasRoomDepthSignal &&
            roomWidth > 0.05f &&
            roomHeight > 0.05f &&
            roomDepth > 0.05f
        ) {
            RoomDimensions(roomWidth, roomHeight, roomDepth)
        } else {
            null
        }
        val hasSingleFurnitureSelection =
            !inlineBrainFullVideoEnabled || inlineBrainSelectedPins.size == 1
        val evaluatedWidthMeters = inlineBrainFurnitureWidthMeters.takeIf {
            hasSingleFurnitureSelection
        }
        val evaluatedHeightMeters = inlineBrainFurnitureHeightMeters.takeIf {
            hasSingleFurnitureSelection
        }
        val furnitureDimensions = if (
            evaluatedWidthMeters != null &&
            evaluatedHeightMeters != null
        ) {
            FurnitureDimensions.fromMeasuredWidthAndHeight(
                evaluatedWidthMeters,
                evaluatedHeightMeters,
            )
        } else {
            null
        }
        val furnitureProfile = inlineBrainFurnitureColor?.let { furnitureColor ->
            FurnitureAestheticProfile(
                primaryColor = furnitureColor,
                styleTags = furnitureStyleTags(inlineBrainFurnitureLabel),
            )
        }
        var result = RoomIntelligenceEngine.evaluateMeasuredFurniture(
            roomDimensions = roomDimensions,
            furnitureWidthMeters = evaluatedWidthMeters,
            furnitureHeightMeters = evaluatedHeightMeters,
            roomPalette = inlineBrainRoomPalette,
            furnitureAestheticProfile = furnitureProfile,
        )
        if (furnitureDimensions == null && furnitureProfile == null) {
            result = result.copy(status = RoomIntelligenceStatus.MEASURING)
        }
        var cardState = PlacementIntelligenceCardMapper.map(
            context = this,
            result = result,
            roomDimensions = roomDimensions,
            furnitureDimensions = furnitureDimensions,
        )
        if (!hasSingleFurnitureSelection) {
            val arMeasurementPrompt = getString(R.string.placement_intelligence_measure_with_ar)
            cardState = cardState.copy(
                notes = listOf(getString(R.string.placement_intelligence_single_item)) +
                    cardState.notes.filterNot { it == arMeasurementPrompt },
            )
        }
        placementIntelligenceCard.render(cardState)
        placementIntelligenceCard.bringToFront()
        ensureNavigationChromeOnTop()
    }

    private fun furnitureStyleTags(label: String?): List<String> = when (label?.lowercase(Locale.US)) {
        "couch", "sofa", "chair", "dining table", "bed" -> listOf("modern", "minimalist")
        else -> listOf("modern")
    }

    private fun inlineBrainSelectionKey(detection: DetectionResult): String {
        val selectedPin = inlineBrainSelectedPins.singleOrNull()
        return if (inlineBrainFullVideoEnabled && selectedPin != null) {
            "${selectedPin.classId}:${selectedPin.x.roundToInt()}:${selectedPin.y.roundToInt()}"
        } else {
            detection.label
        }
    }

    private fun releaseInlineBrainArCamera() {
        val controller = inlineBrainArCameraController ?: return
        try {
            rootLayout.removeView(controller.glSurfaceView)
        } catch (_: Exception) {
        }
        controller.destroy()
        inlineBrainArCameraController = null
    }

    @SuppressLint("UnsafeOptInUsageError")
    private fun applyInlineBrainCameraBinding(provider: ProcessCameraProvider, generation: Int) {
        if (inlineBrainArAssistedSizingEnabled) {
            provider.unbindAll()
            startInlineBrainArCamera(generation)
            return
        }
        releaseInlineBrainArCamera()
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
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .setOutputImageRotationEnabled(true)
            .build()

        analysis.setAnalyzer(cameraExecutor) { imageProxy ->
            try {
                if (!brainAcceptingUpdates || brainSessionGeneration.get() != generation) return@setAnalyzer
                if (isBrainInferenceRunning.get()) {
                    retainLatestDroppedInlineBrainFrame(imageProxy, generation)
                    return@setAnalyzer
                }
                // Covered lens: skip decode + detector; stop thrashing the progress overlay.
                if (FurnitureFitFrameUsability.isFullyDark(imageProxy)) {
                    runOnUiThread {
                        if (brainAcceptingUpdates && brainSessionGeneration.get() == generation) {
                            hideBrainProgress()
                        }
                    }
                    return@setAnalyzer
                }
                // Match iOS RTMDet live cadence (200/400ms) + thermal-critical pause for both Fit modes.
                if (!inlineBrainThermalCadence.tryBeginInference()) return@setAnalyzer
                val rawBitmap = imageProxy.toBitmapSafe() ?: return@setAnalyzer
                val (bitmap, _) = rawBitmap.rotateToMatchLockedRoomPhoto(photoOrientation)
                if (bitmap !== rawBitmap) rawBitmap.recycle()
                if (FurnitureFitFrameUsability.isFullyDark(bitmap)) {
                    bitmap.recycle()
                    runOnUiThread {
                        if (brainAcceptingUpdates && brainSessionGeneration.get() == generation) {
                            hideBrainProgress()
                        }
                    }
                    return@setAnalyzer
                }
                isBrainInferenceRunning.set(true)
                submitInlineBrainInference(bitmap, generation)
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
            refreshMorphingPrimaryFitButton()
            updateInlineBrainCameraPreviewVisibility()
            LogUtil.e(TAG, "Inline brain camera bind failed", e)
            Toast.makeText(this, R.string.smartypants_camera_error_generic, Toast.LENGTH_SHORT).show()
            CrashReporter.report(this, e, "GLB room inline brain camera bind")
        }
    }

    private fun processInlineBrainBitmap(
        sourceBitmap: Bitmap,
        generation: Int,
        bitmapAlreadyOriented: Boolean,
    ) {
        if (!brainAcceptingUpdates || brainSessionGeneration.get() != generation) {
            sourceBitmap.recycle()
            inlineBrainArCameraController?.onInferenceFinished()
            return
        }
        if (isBrainInferenceRunning.get()) {
            sourceBitmap.recycle()
            inlineBrainArCameraController?.onInferenceFinished()
            return
        }
        val bitmap = if (bitmapAlreadyOriented) {
            sourceBitmap
        } else {
            val (orientedBitmap, _) = sourceBitmap.rotateToMatchLockedRoomPhoto(photoOrientation)
            if (orientedBitmap !== sourceBitmap) sourceBitmap.recycle()
            orientedBitmap
        }
        if (
            FurnitureFitFrameUsability.isFullyDark(bitmap) ||
            inlineBrainThermalCadence.isPausedForThermalCritical
        ) {
            bitmap.recycle()
            inlineBrainArCameraController?.onInferenceFinished()
            return
        }
        isBrainInferenceRunning.set(true)
        submitInlineBrainInference(bitmap, generation)
    }

    private fun submitInlineBrainInference(bitmap: Bitmap, generation: Int) {
        // A deferred newest frame re-enters Swift's complete processFrame path, including its
        // all-black camera guard. Keep that guard here as well; pending frames bypass the outer
        // CameraX admission method by design.
        if (FurnitureFitFrameUsability.isFullyDark(bitmap)) {
            bitmap.recycle()
            val pendingNext = takePendingLatestInlineBrainFrame(generation)
            if (pendingNext != null) {
                submitInlineBrainInference(pendingNext.bitmap, generation)
            } else if (brainSessionGeneration.get() == generation) {
                isBrainInferenceRunning.set(false)
                inlineBrainArCameraController?.onInferenceFinished()
            }
            return
        }
        val modeSnapshot = inlineBrainMode
        val fullVideoSnapshot = inlineBrainFullVideoEnabled
        val selectedPinsSnapshot = inlineBrainSelectedPins
        val callback: (SegmentationResult?) -> Unit = callback@{ result ->
            bitmap.recycle()
            if (!brainAcceptingUpdates || brainSessionGeneration.get() != generation) {
                result?.mask?.takeIf { !it.isRecycled }?.recycle()
                return@callback
            }

            // Swift posts the presentation update, then releases the inference gate on its worker
            // queue. Do not wait for Android's main thread (or color/overlay work) before accepting
            // the newest camera pose.
            runOnUiThread {
                if (!brainAcceptingUpdates || brainSessionGeneration.get() != generation) {
                    result?.mask?.takeIf { !it.isRecycled }?.recycle()
                    return@runOnUiThread
                }
                if (inlineBrainMode != modeSnapshot || inlineBrainFullVideoEnabled != fullVideoSnapshot) {
                    LogUtil.d(
                        TAG,
                        "Inline brain: dropping stale result mode=$modeSnapshot current=$inlineBrainMode " +
                            "fullVideo=$fullVideoSnapshot currentFullVideo=$inlineBrainFullVideoEnabled",
                    )
                    result?.mask?.takeIf { !it.isRecycled }?.recycle()
                } else {
                    applyInlineBrainResult(result)
                }
            }

            // Keep the gate closed only when transferring the single newest deferred frame. There
            // is no FIFO; this mirrors processPendingLatestSegmentationFrameIfNeeded in Swift.
            val pendingNext = takePendingLatestInlineBrainFrame(generation)
            if (pendingNext != null) {
                submitInlineBrainInference(pendingNext.bitmap, generation)
            } else {
                isBrainInferenceRunning.set(false)
                inlineBrainArCameraController?.onInferenceFinished()
            }
        }

        val manager = furnitureFitManager
        if (manager == null) {
            bitmap.recycle()
            if (brainSessionGeneration.get() == generation) {
                isBrainInferenceRunning.set(false)
                inlineBrainArCameraController?.onInferenceFinished()
            }
            return
        }
        when {
            fullVideoSnapshot && modeSnapshot == InlineBrainMode.IDENTIFY ->
                manager.detectWithDetectionsOnInferenceThreadAsync(bitmap, requireClusters = true, callback = callback)
            fullVideoSnapshot && modeSnapshot == InlineBrainMode.SEGMENT_SELECTED ->
                manager.segmentSelectedInstancesOnInferenceThreadAsync(bitmap, selectedPinsSnapshot, callback)
            else ->
                manager.segmentWithDetectionsOnInferenceThreadAsync(bitmap, callback)
        }
    }

    /** Same mode policy as Swift `shouldKeepLatestDroppedCameraFrame` for RTMDet. */
    private fun shouldKeepLatestDroppedInlineBrainFrame(): Boolean {
        return inlineBrainFullVideoEnabled &&
            (inlineBrainMode == InlineBrainMode.IDENTIFY ||
                inlineBrainMode == InlineBrainMode.SEGMENT_SELECTED)
    }

    /**
     * Copies and replaces one newest dropped CameraX frame. Default RTMDet segmentation and the
     * ARCore path continue dropping every busy frame exactly as Swift does.
     */
    private fun retainLatestDroppedInlineBrainFrame(imageProxy: androidx.camera.core.ImageProxy, generation: Int) {
        if (!shouldKeepLatestDroppedInlineBrainFrame()) return
        val rawBitmap = imageProxy.toBitmapSafe() ?: return
        val (orientedBitmap, _) = rawBitmap.rotateToMatchLockedRoomPhoto(photoOrientation)
        if (orientedBitmap !== rawBitmap) rawBitmap.recycle()

        if (
            !brainAcceptingUpdates ||
            brainSessionGeneration.get() != generation ||
            !shouldKeepLatestDroppedInlineBrainFrame()
        ) {
            orientedBitmap.recycle()
            return
        }

        pendingLatestInlineBrainFrame
            .getAndSet(PendingInlineBrainFrame(orientedBitmap, generation))
            ?.bitmap
            ?.takeIf { !it.isRecycled }
            ?.recycle()
    }

    private fun takePendingLatestInlineBrainFrame(generation: Int): PendingInlineBrainFrame? {
        val pending = pendingLatestInlineBrainFrame.getAndSet(null) ?: return null
        if (
            pending.generation != generation ||
            !brainAcceptingUpdates ||
            brainSessionGeneration.get() != generation ||
            !shouldKeepLatestDroppedInlineBrainFrame()
        ) {
            pending.bitmap.takeIf { !it.isRecycled }?.recycle()
            return null
        }
        return pending
    }

    private fun clearPendingLatestInlineBrainFrame() {
        pendingLatestInlineBrainFrame
            .getAndSet(null)
            ?.bitmap
            ?.takeIf { !it.isRecycled }
            ?.recycle()
    }

    private fun applyInlineBrainResult(result: SegmentationResult?) {
        hideBrainProgress()
        val mask = result?.mask
        val detections = result?.detections ?: emptyList()
        if (
            result != null &&
            inlineBrainFullVideoEnabled &&
            (detections.isNotEmpty() || inlineBrainMode == InlineBrainMode.SEGMENT_SELECTED)
        ) {
            refreshInlineBrainPinsFromCandidates(detections)
        }
        val primaryDetection = result?.primaryDetection ?: detections.firstOrNull()
        if (mask != null) {
            if (!inlineBrainFirstMaskLogged) {
                inlineBrainFirstMaskLogged = true
                val startedAt = inlineBrainActionStartedAtNanos
                val elapsedMs = startedAt?.let {
                    (android.os.SystemClock.elapsedRealtimeNanos() - it) / 1_000_000L
                }
                LogUtil.i(TAG, "Inline brain timing: FIRST_MASK_ARRIVAL elapsedMs=${elapsedMs ?: -1} mode=$inlineBrainMode")
            }
            inlineBrainHasSegmentedFurniture = true
            inlineBrainFurnitureLabel = primaryDetection?.label
            sampleInlineBrainFurnitureColor(mask)
        } else if (inlineBrainMode != InlineBrainMode.IDENTIFY) {
            inlineBrainHasSegmentedFurniture = false
            inlineBrainFurnitureColor = null
            inlineBrainFurnitureLabel = null
        }
        if (inlineBrainArAssistedSizingEnabled && mask != null && primaryDetection != null) {
            val inputSize = result.inputSize.coerceAtLeast(1).toFloat()
            val sourceWidth = result.sourceWidth.takeIf { it > 0 } ?: mask.width
            val sourceHeight = result.sourceHeight.takeIf { it > 0 } ?: mask.height
            inlineBrainArCameraController?.setBboxHint(
                primaryDetection.x * sourceWidth / inputSize,
                primaryDetection.y * sourceHeight / inputSize,
                primaryDetection.w * sourceWidth / inputSize,
                primaryDetection.h * sourceHeight / inputSize,
                primaryDetection.label,
                inlineBrainSelectionKey(primaryDetection),
            )
            inlineBrainArCameraController?.let(::updateInlineBrainMetricDimensions)
        } else if (inlineBrainMode != InlineBrainMode.IDENTIFY) {
            inlineBrainArCameraController?.clearBboxHint()
            clearInlineBrainMetricDimensions()
        }
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
                        primaryDetection = primaryDetection,
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
                        primaryDetection = primaryDetection,
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
                primaryDetection = primaryDetection,
            )
        }
        LogUtil.i(
            TAG,
            "Inline brain result: mask=${mask != null} alpha=${mask?.hasAlpha()} dets=${detections.size} " +
                "clusters=${result?.detectionClusters?.size ?: 0} mode=$inlineBrainMode " +
                "primary=${result?.primaryDetection?.label}:${result?.primaryDetection?.confidence}",
        )
        refreshPlacementIntelligence()
    }

    private fun stopInlineBrainSegmentation() {
        LogUtil.d(TAG, "Inline brain: stop")
        brainSessionGeneration.incrementAndGet()
        brainAcceptingUpdates = false
        clearPendingLatestInlineBrainFrame()
        furnitureFitManager?.rotateInferenceQueueForNewSession()
        isBrainInferenceRunning.set(false)
        inlineBrainThermalCadence.stop()
        inlineBrainMode = InlineBrainMode.DEFAULT_SEGMENT
        inlineBrainFullVideoEnabled = false
        inlineBrainSelectedPins = emptyList()
        inlineBrainPinMissingFrameCounts.clear()
        inlineBrainHasSegmentedFurniture = false
        inlineBrainFurnitureColor = null
        inlineBrainFurnitureLabel = null
        hideBrainProgress()
        brainDetectionOverlay.visibility = View.GONE
        brainDetectionOverlayView.setMaskAndDetections(null, emptyList())
        brainDetectionOverlayView.setDetectionBoxVisibility(false)
        brainDetectionOverlayView.setIdentifySelectionState(false, emptyList())
        releaseInlineBrainArCamera()
        clearInlineBrainMetricDimensions()
        if (::placementIntelligenceCard.isInitialized) {
            placementIntelligenceCard.clear()
        }
        boundPreview?.setSurfaceProvider(null)
        boundPreview = null
        if (::brainCameraPreview.isInitialized) {
            brainCameraPreview.visibility = View.GONE
        }
        refreshMorphingPrimaryFitButton()
        inlineBrainArAssistedSizingEnabled = false
        updateSummonedToolbarState()
        try {
            cameraProvider?.unbindAll()
        } catch (_: Exception) {
        }
        cameraProvider = null
        dismissFullVideoFurnitureTapHint()
        hintController.hide(animated = false)
        ensureNavigationChromeOnTop()
    }

    private fun presentFullVideoFurnitureTapHint() {
        if (!inlineBrainFullVideoEnabled) return
        if (!::brainDetectionOverlay.isInitialized || brainDetectionOverlay.visibility != View.VISIBLE) return
        hintController.showStickyTop(
            this,
            R.drawable.ic_gesture_tap,
            R.string.room_viewer_full_video_furniture_tap_hint,
            topMarginDp = 52,
        )
    }

    private fun dismissFullVideoFurnitureTapHint() {
        hintController.hideSticky(animated = true)
    }

    /** Matches iOS `presentFullVideoSelectionHelperIfNeeded` — transient when Fit is on, full video off. */
    private fun presentFullVideoSelectionHelper() {
        val fitActive = ::brainDetectionOverlay.isInitialized &&
            brainDetectionOverlay.visibility == View.VISIBLE
        if (!fitActive || inlineBrainFullVideoEnabled) return
        if (inlineBrainMode != InlineBrainMode.DEFAULT_SEGMENT) return
        hintController.show(
            this,
            R.drawable.ic_text_viewfinder,
            R.string.room_viewer_full_video_selection_helper,
            durationMs = 3000L,
        )
    }

    private fun refreshMorphingPrimaryFitButton() {
        updateMorphingPrimaryFitState()
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
        val viewerDir = File(
            cacheDir,
            "glb_web_viewer_${Integer.toHexString(System.identityHashCode(this))}",
        ).apply {
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
        Toast.makeText(this, R.string.room_viewer_preview_unavailable, Toast.LENGTH_LONG).show()
        CrashReporter.report(this, RuntimeException(message), "GLB room — WebGL viewer")
    }

    private fun generateWebGLHTML(): String {
        val isPortrait = photoOrientation == "portrait"
        // Settings > Auto Orbit (settings_auto_orbit). Default OFF, matching iOS
        // @AppStorage("roomViewer.oscillation") and the original 4fd456a8 behaviour.
        val autoOrbitEnabled = getSharedPreferences("furnit_prefs", MODE_PRIVATE)
            .getBoolean("auto_orbit_enabled", false)
        // Settings > Infinite Zoom. Default OFF, matching iOS. The normal room/photo
        // navigation contract stays bounded unless the user explicitly enables it.
        // @AppStorage("roomViewer.infiniteZoom").
        val infiniteZoomEnabled = getSharedPreferences("furnit_prefs", MODE_PRIVATE)
            .getBoolean("infinite_zoom_enabled", false)

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
        // Settings > Infinite Zoom: lift the dolly clamps instead of the room-sized
        // default. Mirrors the Metal splat viewer, which widens 0.5..3.0 to 0.1..50.
        const INFINITE_ZOOM_ENABLED = $infiniteZoomEnabled;
        controls.minDistance = INFINITE_ZOOM_ENABLED ? 0.05 : 0.5;
        controls.maxDistance = INFINITE_ZOOM_ENABLED ? 1000 : 20;
        controls.touches = {
            ONE: THREE.TOUCH.ROTATE,
            TWO: THREE.TOUCH.DOLLY_PAN
        };

        // --- Auto Orbit (Settings > Auto Orbit) -------------------------------------
        // Restored from 4fd456a8; the implementation was lost when the WebGL splat viewer
        // was replaced by MetalSplatter in 71cba782, leaving the toggle wired to nothing.
        // Same constants as the original: speed 0.35, amplitude PI/6 (+/-30 degrees),
        // portrait sweeps a circular arc, landscape sweeps horizontally, 30fps throttle.
        const AUTO_ORBIT_ENABLED = $autoOrbitEnabled;
        const AUTO_ORBIT_IS_PORTRAIT = $isPortrait;
        const AUTO_ORBIT_SPEED = 0.35;
        const AUTO_ORBIT_AMPLITUDE = Math.PI / 6;
        const AUTO_ORBIT_IDLE_DELAY_MS = 2000;
        const AUTO_ORBIT_FRAME_MS = 1000 / 30;

        let autoOrbitTime = 0;
        let autoOrbitRadius = 0;
        let autoOrbitBaseAngle = 0;
        let autoOrbitOrigin = null;
        let autoOrbitLastFrameMs = 0;
        let userInteracting = false;
        let lastInteractionMs = Date.now();

        // Any camera change counts as interaction: pinch, drag, and the D-pad nudges all
        // move the camera, and none of them should fight an idle animation.
        function noteAutoOrbitInteraction() {
            lastInteractionMs = Date.now();
            autoOrbitOrigin = null;
        }
        controls.addEventListener('start', function () {
            userInteracting = true;
            noteAutoOrbitInteraction();
        });
        controls.addEventListener('end', function () {
            userInteracting = false;
            noteAutoOrbitInteraction();
        });

        // Capture the orbit basis when idling begins, so the sweep starts from wherever the
        // user left the camera rather than snapping to the initial framing.
        function beginAutoOrbitFromCurrentCamera() {
            const t = controls.target;
            const dx = camera.position.x - t.x;
            const dz = camera.position.z - t.z;
            autoOrbitRadius = Math.sqrt(dx * dx + dz * dz);
            autoOrbitBaseAngle = Math.atan2(dx, dz);
            autoOrbitOrigin = { x: camera.position.x, z: camera.position.z };
            autoOrbitTime = 0;
        }

        function stepAutoOrbit(nowMs) {
            if (!AUTO_ORBIT_ENABLED || userInteracting || navigationMode !== 'orbit') return false;
            if (nowMs - lastInteractionMs < AUTO_ORBIT_IDLE_DELAY_MS) return false;
            if (nowMs - autoOrbitLastFrameMs < AUTO_ORBIT_FRAME_MS) return false;

            if (autoOrbitOrigin === null) beginAutoOrbitFromCurrentCamera();
            if (autoOrbitRadius <= 0.1) return false;

            const dt = (nowMs - autoOrbitLastFrameMs) / 1000;
            autoOrbitLastFrameMs = nowMs;
            autoOrbitTime += Math.min(dt, 0.1);

            const t = controls.target;
            if (AUTO_ORBIT_IS_PORTRAIT) {
                const angle = autoOrbitBaseAngle
                    + AUTO_ORBIT_AMPLITUDE * Math.sin(autoOrbitTime * AUTO_ORBIT_SPEED);
                camera.position.x = t.x + autoOrbitRadius * Math.sin(angle);
                camera.position.z = t.z + autoOrbitRadius * Math.cos(angle);
            } else {
                const sweep = autoOrbitRadius * 0.3
                    * Math.sin(autoOrbitTime * AUTO_ORBIT_SPEED);
                camera.position.x = autoOrbitOrigin.x + sweep;
                camera.position.z = autoOrbitOrigin.z;
            }
            camera.lookAt(t);
            return true;
        }

        let initialCameraPosition = null;
        let initialControlsTarget = null;
        let roomBoundsForClamping = null;
        let isFlatPhotoMesh = false;
        let isDepthPhotoMesh = false;
        let flatPhotoWidth = 0;
        let flatPhotoHeight = 0;
        let photoSurfaceFrontZ = 0;
        const depthPhotoCaptureOrigin = new THREE.Vector3();
        let depthPhotoVerticalFovDegrees = 60;
        let depthPhotoTargetDistance = 2;
        let depthPhotoZoom = 1;
        let isLayeredDepthPhoto = false;
        let depthPhotoTranslationLimit = 0;
        let depthPhotoLateralLimit = 0;
        let depthPhotoBackwardLimit = 0;
        let depthPhotoYawLimit = Math.PI / 6;
        let depthPhotoPitchLimit = Math.PI / 5;
        let depthPhotoSourceAspect = null;
        let depthPhotoNearestReliableDepth = null;
        const cameraOrbitRadiansPerPixel = 0.012;
        const depthPhotoDpadCoveragePadding = THREE.MathUtils.degToRad(0.25);

        function isPhotoSurfaceMesh() {
            return isFlatPhotoMesh || isDepthPhotoMesh;
        }

        // A room needs two different camera models. OrbitControls is appropriate while viewing a
        // room from outside, but once the eye is inside a real room volume it must turn in place.
        // Auto Orbit is only an idle animation setting and deliberately does not choose this mode.
        let navigationMode = 'orbit';
        const interiorPointers = new Map();
        const interiorEuler = new THREE.Euler(0, 0, 0, 'YXZ');
        const interiorForward = new THREE.Vector3();
        const interiorRight = new THREE.Vector3();
        const interiorMove = new THREE.Vector3();
        let previousInteriorCentroid = null;
        let previousInteriorDistance = null;

        function hasNavigableRoomVolume() {
            if (!roomBoundsForClamping || isPhotoSurfaceMesh()) return false;
            const width = roomBoundsForClamping.maxX - roomBoundsForClamping.minX;
            const height = roomBoundsForClamping.maxY - roomBoundsForClamping.minY;
            const depth = roomBoundsForClamping.maxZ - roomBoundsForClamping.minZ;
            return width > 0.2 && height > 0.2 && depth > 0.2;
        }

        function isInsideRoom(position) {
            if (!hasNavigableRoomVolume()) return false;
            const b = roomBoundsForClamping;
            return position.x >= b.minX && position.x <= b.maxX &&
                position.y >= b.minY && position.y <= b.maxY &&
                position.z >= b.minZ && position.z <= b.maxZ;
        }

        function constrainToRoom(position) {
            if (INFINITE_ZOOM_ENABLED) return position;
            if (!roomBoundsForClamping) return position;
            const b = roomBoundsForClamping;
            if (isPhotoSurfaceMesh()) {
                position.x = THREE.MathUtils.clamp(position.x, b.minX, b.maxX);
                position.y = THREE.MathUtils.clamp(position.y, b.minY, b.maxY);
                position.z = THREE.MathUtils.clamp(position.z, b.minZ, b.maxZ);
                return position;
            }
            if (!hasNavigableRoomVolume()) return position;
            position.x = THREE.MathUtils.clamp(position.x, b.minX, b.maxX);
            position.y = THREE.MathUtils.clamp(position.y, b.minY, b.maxY);
            position.z = THREE.MathUtils.clamp(position.z, b.minZ, b.maxZ);
            return position;
        }

        function interiorLookDistance() {
            if (!hasNavigableRoomVolume()) return 2.0;
            const b = roomBoundsForClamping;
            return Math.max(1.0, Math.hypot(b.maxX - b.minX, b.maxZ - b.minZ) * 0.5);
        }

        function syncFirstPersonTarget() {
            camera.getWorldDirection(interiorForward);
            controls.target.copy(camera.position).addScaledVector(interiorForward, interiorLookDistance());
        }

        function resetInteriorGestureBaseline() {
            const points = Array.from(interiorPointers.values());
            if (points.length === 0) {
                previousInteriorCentroid = null;
                previousInteriorDistance = null;
                return;
            }
            const centroid = points.reduce(
                (sum, point) => ({ x: sum.x + point.x, y: sum.y + point.y }),
                { x: 0, y: 0 },
            );
            previousInteriorCentroid = {
                x: centroid.x / points.length,
                y: centroid.y / points.length,
            };
            previousInteriorDistance = points.length === 2
                ? Math.hypot(points[0].x - points[1].x, points[0].y - points[1].y)
                : null;
        }

        function updateNavigationMode() {
            const nextMode = isDepthPhotoMesh
                ? 'photoDepth'
                : (isInsideRoom(camera.position) ? 'firstPerson' : 'orbit');
            if (nextMode === navigationMode) {
                if (navigationMode === 'firstPerson' || navigationMode === 'photoDepth') syncFirstPersonTarget();
                return;
            }
            navigationMode = nextMode;
            controls.enabled = navigationMode === 'orbit';
            interiorPointers.clear();
            resetInteriorGestureBaseline();
            if (navigationMode === 'firstPerson' || navigationMode === 'photoDepth') syncFirstPersonTarget();
            console.log('[GLBViewer] Navigation mode:', navigationMode);
        }

        // OrbitControls owns gestures outside. Inside, one pointer changes yaw/pitch without
        // translating the eye. Two pointers move volumetric rooms, but only change projection
        // zoom for single-photo depth rooms so their authored image rays remain aligned.
        renderer.domElement.addEventListener('pointerdown', (event) => {
            if (navigationMode !== 'firstPerson' && navigationMode !== 'photoDepth') return;
            event.preventDefault();
            interiorPointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
            renderer.domElement.setPointerCapture?.(event.pointerId);
            resetInteriorGestureBaseline();
            noteAutoOrbitInteraction();
        }, { passive: false });

        renderer.domElement.addEventListener('pointermove', (event) => {
            if ((navigationMode !== 'firstPerson' && navigationMode !== 'photoDepth') ||
                    !interiorPointers.has(event.pointerId)) return;
            event.preventDefault();
            const previousPoint = interiorPointers.get(event.pointerId);
            interiorPointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
            const points = Array.from(interiorPointers.values());

            if (points.length === 1) {
                interiorEuler.setFromQuaternion(camera.quaternion, 'YXZ');
                interiorEuler.y -= (event.clientX - previousPoint.x) * 0.005;
                interiorEuler.x -= (event.clientY - previousPoint.y) * 0.005;
                if (navigationMode === 'photoDepth' && isLayeredDepthPhoto) {
                    const coverage = currentDepthPhotoLookLimits();
                    interiorEuler.y = THREE.MathUtils.clamp(interiorEuler.y, -coverage.yaw, coverage.yaw);
                    interiorEuler.x = THREE.MathUtils.clamp(interiorEuler.x, -coverage.pitch, coverage.pitch);
                } else {
                    interiorEuler.x = THREE.MathUtils.clamp(
                        interiorEuler.x,
                        -(Math.PI / 2 - 0.05),
                        Math.PI / 2 - 0.05,
                    );
                }
                interiorEuler.z = 0;
                camera.quaternion.setFromEuler(interiorEuler);
            } else if (points.length === 2 && previousInteriorCentroid) {
                const centroid = {
                    x: (points[0].x + points[1].x) * 0.5,
                    y: (points[0].y + points[1].y) * 0.5,
                };
                const distance = Math.hypot(
                    points[0].x - points[1].x,
                    points[0].y - points[1].y,
                );
                if (navigationMode === 'photoDepth') {
                    if (previousInteriorDistance !== null && previousInteriorDistance > 0) {
                        if (isLayeredDepthPhoto) {
                            const viewportSpan = Math.max(
                                Math.min(renderer.domElement.clientWidth, renderer.domElement.clientHeight),
                                1,
                            );
                            const gestureFraction = (distance - previousInteriorDistance) / viewportSpan;
                            const delta = gestureFraction * depthPhotoTranslationLimit * 2.25;
                            camera.getWorldDirection(interiorForward);
                            constrainToRoom(camera.position.addScaledVector(interiorForward, delta));
                            const coverage = currentDepthPhotoLookLimits();
                            interiorEuler.setFromQuaternion(camera.quaternion, 'YXZ');
                            interiorEuler.y = THREE.MathUtils.clamp(interiorEuler.y, -coverage.yaw, coverage.yaw);
                            interiorEuler.x = THREE.MathUtils.clamp(interiorEuler.x, -coverage.pitch, coverage.pitch);
                            interiorEuler.z = 0;
                            camera.quaternion.setFromEuler(interiorEuler);
                        } else {
                            const pinchRatio = distance / previousInteriorDistance;
                            if (Number.isFinite(pinchRatio) && pinchRatio > 0) {
                                const minimumPhotoZoom = INFINITE_ZOOM_ENABLED ? 0.05 : 0.5;
                                const maximumPhotoZoom = INFINITE_ZOOM_ENABLED ? 1000 : 4.0;
                                depthPhotoZoom = THREE.MathUtils.clamp(
                                    depthPhotoZoom * pinchRatio,
                                    minimumPhotoZoom,
                                    maximumPhotoZoom,
                                );
                                updatePhotoProjection();
                            }
                        }
                    }
                    previousInteriorCentroid = centroid;
                    previousInteriorDistance = distance;
                    syncFirstPersonTarget();
                    return;
                }
                const b = roomBoundsForClamping;
                const roomScale = Math.max(b.maxX - b.minX, b.maxZ - b.minZ, 1.0);
                camera.getWorldDirection(interiorForward);
                interiorForward.y = 0;
                if (interiorForward.lengthSq() > 1e-6) interiorForward.normalize();
                interiorRight.set(interiorForward.z, 0, -interiorForward.x);
                interiorMove.set(0, 0, 0)
                    .addScaledVector(interiorRight, -(centroid.x - previousInteriorCentroid.x) * roomScale * 0.0015)
                    .addScaledVector(THREE.Object3D.DEFAULT_UP, -(centroid.y - previousInteriorCentroid.y) * roomScale * 0.0015);
                if (previousInteriorDistance !== null) {
                    interiorMove.addScaledVector(
                        interiorForward,
                        (distance - previousInteriorDistance) * roomScale * 0.0025,
                    );
                }
                constrainToRoom(camera.position.add(interiorMove));
                previousInteriorCentroid = centroid;
                previousInteriorDistance = distance;
            }
            syncFirstPersonTarget();
        }, { passive: false });

        function finishInteriorPointer(event) {
            if (!interiorPointers.has(event.pointerId)) return;
            interiorPointers.delete(event.pointerId);
            resetInteriorGestureBaseline();
        }
        renderer.domElement.addEventListener('pointerup', finishInteriorPointer);
        renderer.domElement.addEventListener('pointercancel', finishInteriorPointer);
        controls.addEventListener('end', updateNavigationMode);

        function viewportSize() {
            const visualViewport = window.visualViewport;
            return {
                width: Math.max(visualViewport?.width ?? window.innerWidth, 1),
                height: Math.max(visualViewport?.height ?? window.innerHeight, 1),
                dpr: window.devicePixelRatio || 1,
            };
        }

        // Keep cover-framing stable while Android is between configuration and WebView resize
        // callbacks. This mirrors the iOS orientation-locked photo-room viewport fallback.
        function photoProjectionAspect() {
            const viewport = viewportSize();
            const liveAspect = Math.max(viewport.width / viewport.height, 0.01);
            if (!isPortrait && liveAspect < 1.05) return 19.5 / 9.0;
            if (isPortrait && liveAspect > 0.95) return 9.0 / 19.5;
            return liveAspect;
        }

        function depthAnythingImagePlaneStandoff(width, height) {
            const halfFovRad = Math.PI / 6.0;
            const viewportAspect = photoProjectionAspect();
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

        function updatePhotoProjection() {
            const viewportAspect = photoProjectionAspect();
            if (isDepthPhotoMesh) {
                const sourceHalfY = THREE.MathUtils.degToRad(depthPhotoVerticalFovDegrees) * 0.5;
                const sourceHalfX = Number.isFinite(depthPhotoSourceAspect)
                    ? Math.atan(Math.tan(sourceHalfY) * depthPhotoSourceAspect)
                    : sourceHalfY;
                // Aspect-fill parity with preview. Never reveal pixels beyond the captured frame
                // just because the saved-room viewport is wider than the source photograph.
                const coverHalfY = Math.min(
                    sourceHalfY,
                    Math.atan(Math.tan(sourceHalfX) / viewportAspect),
                );
                camera.fov = THREE.MathUtils.radToDeg(
                    2 * Math.atan(Math.tan(coverHalfY) / depthPhotoZoom)
                );
            } else if (!isPortrait) {
                const verticalHalfFov = Math.atan(Math.tan(Math.PI / 6.0) / viewportAspect);
                camera.fov = THREE.MathUtils.radToDeg(2 * verticalHalfFov);
            } else {
                camera.fov = 60;
            }
            const viewport = viewportSize();
            camera.aspect = Math.max(viewport.width / viewport.height, 0.01);
            camera.updateProjectionMatrix();
        }

        function currentDepthPhotoLookLimits() {
            if (!isLayeredDepthPhoto || !Number.isFinite(depthPhotoNearestReliableDepth) ||
                    !Number.isFinite(depthPhotoSourceAspect)) {
                return { yaw: depthPhotoYawLimit, pitch: depthPhotoPitchLimit };
            }
            const viewportAspect = photoProjectionAspect();
            const sourceHalfY = THREE.MathUtils.degToRad(depthPhotoVerticalFovDegrees) * 0.5;
            const sourceHalfX = Math.atan(Math.tan(sourceHalfY) * depthPhotoSourceAspect);
            const cameraHalfY = THREE.MathUtils.degToRad(camera.fov) * 0.5;
            const cameraHalfX = Math.atan(Math.tan(cameraHalfY) * viewportAspect);
            const depth = Math.max(depthPhotoNearestReliableDepth, 0.2);
            const forward = Math.max(0, depthPhotoCaptureOrigin.z - camera.position.z);
            const remaining = Math.max(depth - forward, depth * 0.25);
            const lateral = Math.abs(camera.position.x - depthPhotoCaptureOrigin.x);
            const vertical = Math.abs(camera.position.y - depthPhotoCaptureOrigin.y);
            const coveredHalfX = Math.atan(Math.max(Math.tan(sourceHalfX) * depth - lateral, 0.001) / remaining);
            const coveredHalfY = Math.atan(Math.max(Math.tan(sourceHalfY) * depth - vertical, 0.001) / remaining);
            return {
                yaw: Math.max(0, Math.min(depthPhotoYawLimit, coveredHalfX - cameraHalfX - 0.01)),
                pitch: Math.max(0, Math.min(depthPhotoPitchLimit, coveredHalfY - cameraHalfY - 0.01)),
            };
        }

        // Aspect-fill consumes the full captured-image coverage on one axis: vertical for a
        // portrait viewport and horizontal for a landscape viewport. A D-pad step on that axis
        // would therefore be clamped to zero at the capture pose. Create only the projection
        // overscan required by the requested step before turning, so every arrow responds while
        // the bounded path still keeps the renderer inside the photographed pixels.
        function ensureDepthPhotoDpadCoverage(axis, requiredLook) {
            if (navigationMode !== 'photoDepth' || !isLayeredDepthPhoto ||
                    !Number.isFinite(requiredLook) || requiredLook <= 0) return;

            const coverageForAxis = () => {
                const coverage = currentDepthPhotoLookLimits();
                return axis === 'yaw' ? coverage.yaw : coverage.pitch;
            };
            const currentCoverage = coverageForAxis();
            if (currentCoverage >= requiredLook) return;

            const originalZoom = depthPhotoZoom;
            const maximumPhotoZoom = INFINITE_ZOOM_ENABLED ? 1000 : 4.0;
            if (originalZoom >= maximumPhotoZoom) return;

            let lowZoom = originalZoom;
            let highZoom = maximumPhotoZoom;
            depthPhotoZoom = highZoom;
            updatePhotoProjection();
            const maximumCoverage = coverageForAxis();
            if (maximumCoverage <= currentCoverage) {
                depthPhotoZoom = originalZoom;
                updatePhotoProjection();
                return;
            }

            // If the authored envelope or zoom ceiling cannot provide a full step, use the
            // largest safe partial step. Otherwise binary-search the smallest necessary crop.
            const targetCoverage = Math.min(requiredLook, maximumCoverage);
            for (let iteration = 0; iteration < 14; iteration++) {
                const candidateZoom = (lowZoom + highZoom) * 0.5;
                depthPhotoZoom = candidateZoom;
                updatePhotoProjection();
                if (coverageForAxis() >= targetCoverage) {
                    highZoom = candidateZoom;
                } else {
                    lowZoom = candidateZoom;
                }
            }
            depthPhotoZoom = highZoom;
            updatePhotoProjection();
            console.log('[GLBViewer] D-pad safe overscan axis=', axis,
                'zoom=', depthPhotoZoom.toFixed(3));
        }

        function prepareDepthPhotoDpadLook(deltaX, deltaY) {
            if (navigationMode !== 'photoDepth' || !isLayeredDepthPhoto) return;
            interiorEuler.setFromQuaternion(camera.quaternion, 'YXZ');
            if (deltaX !== 0) {
                const targetYaw = THREE.MathUtils.clamp(
                    interiorEuler.y - deltaX * cameraOrbitRadiansPerPixel,
                    -depthPhotoYawLimit,
                    depthPhotoYawLimit,
                );
                if (Math.abs(targetYaw - interiorEuler.y) > 1e-6) {
                    ensureDepthPhotoDpadCoverage(
                        'yaw',
                        Math.abs(targetYaw) + depthPhotoDpadCoveragePadding,
                    );
                }
            }
            if (deltaY !== 0) {
                const targetPitch = THREE.MathUtils.clamp(
                    interiorEuler.x - deltaY * cameraOrbitRadiansPerPixel,
                    -depthPhotoPitchLimit,
                    depthPhotoPitchLimit,
                );
                if (Math.abs(targetPitch - interiorEuler.x) > 1e-6) {
                    ensureDepthPhotoDpadCoverage(
                        'pitch',
                        Math.abs(targetPitch) + depthPhotoDpadCoveragePadding,
                    );
                }
            }
        }

        function backCenterInsetFraction(depth) {
            const t = Math.min(1, Math.max(0, depth / 6.0));
            return 0.035 + 0.065 * t;
        }

        function applyBackCenterCamera(boxWorld) {
            const depth = Math.max(boxWorld.max.z - boxWorld.min.z, 0.1);
            const insetFromBack = Math.max(depth * backCenterInsetFraction(depth), 0.05);
            const roomHeight = Math.max(boxWorld.max.y - boxWorld.min.y, 0.1);
            const centerY = (boxWorld.min.y + boxWorld.max.y) * 0.5;
            const lookAtY = centerY - roomHeight * 0.06;
            const cameraZ = boxWorld.max.z - insetFromBack;
            const targetZ = boxWorld.min.z;

            camera.position.set(0, lookAtY + Math.max(roomHeight * 0.14, 0.35), cameraZ);
            controls.target.set(0, lookAtY, targetZ);
            camera.lookAt(controls.target);
            controls.update();

            initialCameraPosition = camera.position.clone();
            initialControlsTarget = controls.target.clone();

            const roomWidth = boxWorld.max.x - boxWorld.min.x;
            // Framing recomputes a room-relative cap; leave it alone when the user asked
            // for infinite zoom, or the setting is silently undone a frame later.
            if (!INFINITE_ZOOM_ENABLED) controls.maxDistance = Math.max(roomWidth, depth) * 2.0;
            roomBoundsForClamping = {
                minX: boxWorld.min.x + 0.05,
                maxX: boxWorld.max.x - 0.05,
                minY: boxWorld.min.y + 0.05,
                maxY: boxWorld.max.y - 0.05,
                minZ: boxWorld.min.z + 0.05,
                maxZ: boxWorld.max.z - 0.02
            };
            updateNavigationMode();
            console.log('[GLBViewer] Back-center camera inset=', insetFromBack.toFixed(2),
                'posZ=', cameraZ.toFixed(2), 'targetZ=', targetZ.toFixed(2));
        }

        function applyFlatPhotoCamera() {
            if (!isFlatPhotoMesh || flatPhotoWidth <= 0 || flatPhotoHeight <= 0) return;
            updatePhotoProjection();
            const planeWidth = flatPhotoWidth;
            const planeHeight = flatPhotoHeight;
            const standoff = depthAnythingImagePlaneStandoff(planeWidth, planeHeight);
            // Bias the crop slightly below wall center so cover framing leaves headroom, but keep
            // camera and target at the same height. Tilting a camera toward an already-perspective
            // photograph applies a second perspective transform and makes straight pixels look
            // warped/soft.
            const lookAtY = planeHeight * 0.43;
            const camY = lookAtY;
            // glTF plane normal is +Z; WebGL/Three.js must view from +Z (SceneKit preview parity).
            // Camera on −Z shows the back face and the photo appears horizontally mirrored.
            camera.position.set(0, camY, photoSurfaceFrontZ + standoff);
            controls.target.set(0, lookAtY, 0);
            camera.lookAt(controls.target);
            controls.update();
            initialCameraPosition = camera.position.clone();
            initialControlsTarget = controls.target.clone();
            roomBoundsForClamping = {
                minX: -planeWidth * 0.5 + 0.05,
                maxX: planeWidth * 0.5 - 0.05,
                minY: 0.05,
                maxY: planeHeight - 0.05,
                minZ: photoSurfaceFrontZ + 0.02,
                maxZ: photoSurfaceFrontZ + Math.max(standoff * 1.5, 8.0)
            };
            updateNavigationMode();
            console.log('[GLBViewer] Flat photo camera standoff=', standoff.toFixed(2),
                'plane=', planeWidth.toFixed(2), 'x', planeHeight.toFixed(2),
                'posZ=', standoff.toFixed(2));
        }

        function applyDepthPhotoCamera() {
            if (!isDepthPhotoMesh || flatPhotoWidth <= 0 || flatPhotoHeight <= 0) return;
            updatePhotoProjection();
            camera.position.copy(depthPhotoCaptureOrigin);
            controls.target.set(
                depthPhotoCaptureOrigin.x,
                depthPhotoCaptureOrigin.y,
                depthPhotoCaptureOrigin.z - depthPhotoTargetDistance
            );
            camera.lookAt(controls.target);
            controls.update();
            initialCameraPosition = camera.position.clone();
            initialControlsTarget = controls.target.clone();

            const lateralAllowance = isLayeredDepthPhoto ? depthPhotoLateralLimit : 0.001;
            const verticalAllowance = isLayeredDepthPhoto
                ? THREE.MathUtils.clamp(depthPhotoTranslationLimit * 0.18, 0.10, 0.24)
                : 0.001;
            const forwardAllowance = isLayeredDepthPhoto ? depthPhotoTranslationLimit : 0.001;
            const backwardAllowance = isLayeredDepthPhoto ? depthPhotoBackwardLimit : 0.001;
            roomBoundsForClamping = {
                minX: depthPhotoCaptureOrigin.x - lateralAllowance,
                maxX: depthPhotoCaptureOrigin.x + lateralAllowance,
                minY: depthPhotoCaptureOrigin.y - verticalAllowance,
                maxY: depthPhotoCaptureOrigin.y + verticalAllowance,
                // The GLB camera looks down -Z. The capture point is the rear edge, not the
                // centre of a tiny symmetric cage.
                minZ: depthPhotoCaptureOrigin.z - forwardAllowance,
                maxZ: depthPhotoCaptureOrigin.z + backwardAllowance
            };
            updateNavigationMode();
            console.log('[GLBViewer] Depth photo capture camera origin=',
                depthPhotoCaptureOrigin.x.toFixed(2), depthPhotoCaptureOrigin.y.toFixed(2),
                depthPhotoCaptureOrigin.z.toFixed(2), 'verticalFov=', depthPhotoVerticalFovDegrees.toFixed(2));
        }

        function scheduleCameraFraming() {
            requestAnimationFrame(() => {
                requestAnimationFrame(() => {
                    if (isDepthPhotoMesh) {
                        applyDepthPhotoCamera();
                    } else if (isFlatPhotoMesh) {
                        applyFlatPhotoCamera();
                    }
                    resizeViewer();
                });
            });
        }

        // D-pad / Splat parity: walk on XZ, vertical Y (same as iOS GLBRoomView).
        window.moveCamera = function(dx, dy) {
            noteAutoOrbitInteraction();
            if (navigationMode === 'photoDepth') {
                prepareDepthPhotoDpadLook(dx, dy);
                window.orbitCamera(dx, dy);
                return;
            }
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
            updateNavigationMode();
        };

        window.moveCameraUp = function(dy) {
            noteAutoOrbitInteraction();
            if (typeof dy !== 'number' || !isFinite(dy)) return;
            if (navigationMode === 'photoDepth') {
                const pitchPixels = -dy * 40;
                prepareDepthPhotoDpadLook(0, pitchPixels);
                window.orbitCamera(0, pitchPixels);
                return;
            }
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
            updateNavigationMode();
        };

        // Camera orbit function (called from Android)
        window.orbitCamera = function(deltaX, deltaY) {
            noteAutoOrbitInteraction();
            if (navigationMode === 'firstPerson' || navigationMode === 'photoDepth') {
                interiorEuler.setFromQuaternion(camera.quaternion, 'YXZ');
                interiorEuler.y -= deltaX * cameraOrbitRadiansPerPixel;
                if (navigationMode === 'photoDepth' && isLayeredDepthPhoto) {
                    const coverage = currentDepthPhotoLookLimits();
                    interiorEuler.y = THREE.MathUtils.clamp(interiorEuler.y, -coverage.yaw, coverage.yaw);
                    interiorEuler.x = THREE.MathUtils.clamp(
                        interiorEuler.x - deltaY * cameraOrbitRadiansPerPixel,
                        -coverage.pitch,
                        coverage.pitch,
                    );
                } else {
                    const pitchLimit = Math.PI / 2 - 0.05;
                    interiorEuler.x = THREE.MathUtils.clamp(
                        interiorEuler.x - deltaY * cameraOrbitRadiansPerPixel,
                        -pitchLimit,
                        pitchLimit,
                    );
                }
                interiorEuler.z = 0;
                camera.quaternion.setFromEuler(interiorEuler);
                syncFirstPersonTarget();
                return;
            }
            const spherical = new THREE.Spherical();
            const offset = new THREE.Vector3();
            offset.copy(camera.position).sub(controls.target);
            spherical.setFromVector3(offset);

            spherical.theta -= deltaX * cameraOrbitRadiansPerPixel;
            spherical.phi -= deltaY * cameraOrbitRadiansPerPixel;
            spherical.phi = Math.max(0.1, Math.min(Math.PI - 0.1, spherical.phi));

            offset.setFromSpherical(spherical);
            camera.position.copy(controls.target).add(offset);
            camera.lookAt(controls.target);
        };

        // Recenter function
        window.recenterCamera = function() {
            noteAutoOrbitInteraction();
            if (isDepthPhotoMesh) {
                depthPhotoZoom = 1;
                applyDepthPhotoCamera();
                console.log('[GLBViewer] Depth photo camera recentered');
                return;
            }
            if (isFlatPhotoMesh) {
                applyFlatPhotoCamera();
                console.log('[GLBViewer] Flat photo camera recentered');
                return;
            }
            if (initialCameraPosition && initialControlsTarget) {
                camera.position.copy(initialCameraPosition);
                controls.target.copy(initialControlsTarget);
                controls.update();
                updateNavigationMode();
                console.log('[GLBViewer] Camera recentered');
            }
        };

        function resizeViewer() {
            const viewport = viewportSize();
            camera.aspect = viewport.width / viewport.height;
            if (isPhotoSurfaceMesh()) {
                updatePhotoProjection();
            } else {
                camera.updateProjectionMatrix();
            }
            renderer.setPixelRatio(viewport.dpr);
            renderer.setSize(viewport.width, viewport.height, false);
            if (isDepthPhotoMesh && flatPhotoWidth > 0 && flatPhotoHeight > 0) {
                applyDepthPhotoCamera();
            } else if (isFlatPhotoMesh && flatPhotoWidth > 0 && flatPhotoHeight > 0) {
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
                const depthPhotoNode = model.getObjectByName('photo_room_depth');
                isDepthPhotoMesh = Boolean(depthPhotoNode);
                isFlatPhotoMesh = roomDepth < 0.05;

                if (isPhotoSurfaceMesh()) {
                    flatPhotoWidth = roomWidth;
                    flatPhotoHeight = roomHeight;
                    const boxWorld = new THREE.Box3().setFromObject(model);
                    if (isDepthPhotoMesh) {
                        // The depth GLB is authored in its capture-camera coordinate frame. The
                        // model translation applied above must also be applied to that origin.
                        depthPhotoCaptureOrigin.copy(model.position);
                        depthPhotoTargetDistance = Math.max(-boxWorld.min.z + depthPhotoCaptureOrigin.z, 0.2);
                        const authoredFov = Number(depthPhotoNode?.userData?.cameraVerticalFovDegrees);
                        depthPhotoVerticalFovDegrees = Number.isFinite(authoredFov)
                            ? THREE.MathUtils.clamp(authoredFov, 10, 140)
                            : 60;
                        const layeredVersion = Number(depthPhotoNode?.userData?.layeredDepthVersion);
                        const usesContinuousSurface =
                            depthPhotoNode?.userData?.usesContinuousSurface === true;
                        isLayeredDepthPhoto = usesContinuousSurface ||
                            (Number.isFinite(layeredVersion) && layeredVersion >= 4);
                        const authoredTranslation = Number(depthPhotoNode?.userData?.cameraTranslationLimitMeters);
                        const authoredForward = Number(depthPhotoNode?.userData?.cameraForwardTranslationLimitMeters);
                        const requestedForward = Number.isFinite(authoredForward) ? authoredForward : authoredTranslation;
                        depthPhotoTranslationLimit = isLayeredDepthPhoto && Number.isFinite(requestedForward)
                            ? THREE.MathUtils.clamp(requestedForward, 0.75, 1.40)
                            : (isLayeredDepthPhoto ? 0.75 : 0);
                        const authoredLateral = Number(depthPhotoNode?.userData?.cameraLateralTranslationLimitMeters);
                        depthPhotoLateralLimit = isLayeredDepthPhoto && Number.isFinite(authoredLateral)
                            ? THREE.MathUtils.clamp(authoredLateral, 0.24, 0.48)
                            : (isLayeredDepthPhoto ? Math.min(depthPhotoTranslationLimit * 0.34, 0.48) : 0);
                        const authoredBackward = Number(depthPhotoNode?.userData?.cameraBackwardTranslationLimitMeters);
                        depthPhotoBackwardLimit = isLayeredDepthPhoto && Number.isFinite(authoredBackward)
                            ? THREE.MathUtils.clamp(authoredBackward, 0.18, 0.32)
                            : (isLayeredDepthPhoto ? Math.min(depthPhotoTranslationLimit * 0.25, 0.32) : 0);
                        const authoredYaw = Number(depthPhotoNode?.userData?.cameraYawLimitRadians);
                        if (Number.isFinite(authoredYaw)) depthPhotoYawLimit = THREE.MathUtils.clamp(authoredYaw, 0.1, Math.PI);
                        const authoredPitch = Number(depthPhotoNode?.userData?.cameraPitchLimitRadians);
                        if (Number.isFinite(authoredPitch)) depthPhotoPitchLimit = THREE.MathUtils.clamp(authoredPitch, 0.1, Math.PI / 2 - 0.05);
                        const sourceTextureImage = depthPhotoNode?.material?.map?.image;
                        const authoredSourceWidth = Number(depthPhotoNode?.userData?.cameraSourceImageWidth);
                        const authoredSourceHeight = Number(depthPhotoNode?.userData?.cameraSourceImageHeight);
                        const sourceWidth = Number.isFinite(authoredSourceWidth)
                            ? authoredSourceWidth
                            : Number(sourceTextureImage?.naturalWidth ?? sourceTextureImage?.width);
                        const sourceHeight = Number.isFinite(authoredSourceHeight)
                            ? authoredSourceHeight
                            : Number(sourceTextureImage?.naturalHeight ?? sourceTextureImage?.height);
                        depthPhotoSourceAspect = Number.isFinite(sourceWidth) && Number.isFinite(sourceHeight) && sourceHeight > 0
                            ? sourceWidth / sourceHeight
                            : null;
                        const authoredNearestReliableDepth = Number(
                            depthPhotoNode?.userData?.cameraNearestReliableDepthMeters
                        );
                        // Version-4 assets predate the explicit source/depth coverage metadata.
                        // Their nearest visible point is still available from the authored mesh,
                        // so use it to keep the old edge-extension triangles outside the frustum.
                        const inferredNearestReliableDepth = depthPhotoCaptureOrigin.z - boxWorld.max.z;
                        const nearestReliableDepth = Number.isFinite(authoredNearestReliableDepth)
                            ? authoredNearestReliableDepth
                            : inferredNearestReliableDepth;
                        depthPhotoNearestReliableDepth = Number.isFinite(nearestReliableDepth) && nearestReliableDepth > 0.2
                            ? nearestReliableDepth
                            : null;
                        depthPhotoZoom = 1;
                    } else {
                        photoSurfaceFrontZ = boxWorld.max.z;
                    }
                } else {
                    const boxWorld = new THREE.Box3().setFromObject(model);
                    applyBackCenterCamera(boxWorld);
                }
                updateNavigationMode();

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
            stepAutoOrbit(Date.now());
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
            if (!ModelManager.isRoomNameAvailable(this, typedName)) {
                Toast.makeText(this, getString(R.string.home_room_name_duplicate), Toast.LENGTH_SHORT).show()
                return@showNameRoomDialog
            }
            saveRoom(typedName)
            dismiss()
        }
    }

    private fun saveRoom(name: String) {
        val path = glbPath ?: return
        if (!ModelManager.isRoomNameAvailable(this, name)) {
            Toast.makeText(this, getString(R.string.home_room_name_duplicate), Toast.LENGTH_SHORT).show()
            return
        }

        val overlay = PaafektSavingRoomOverlay.show(rootLayout)
        overlay.setTitle(getString(R.string.room_viewer_saving_room))
        overlay.setProgress(0.02f, getString(R.string.room_viewer_saving_room_ellipsis))

        saveRoomJob?.cancel()
        saveRoomJob = lifecycleScope.launch {
            try {
                delay(220)

                val glbFile = File(path)
                val previewRoomFolder = glbFile.parentFile
                    ?: throw IllegalStateException("Missing room folder")
                overlay.setProgress(0.35f, getString(R.string.room_viewer_saving_room_ellipsis))

                withContext(Dispatchers.IO) {
                    val roomsDir = File(filesDir, "rooms").apply { mkdirs() }
                    val savedRoomFolder = File(roomsDir, previewRoomFolder.name)
                    val destinationAlreadyExisted = savedRoomFolder.exists()
                    try {
                        val savedGlb = RoomArtifactPromoter.copyPreviewArtifact(
                            previewRoomFolder = previewRoomFolder,
                            savedRoomFolder = savedRoomFolder,
                            glbFileName = glbFile.name,
                        )

                        val createdAtMillis = System.currentTimeMillis()
                        val previewMetadata = RoomFolderMetadata.readFromFolder(savedRoomFolder)
                            ?: RoomFolderMetadata.Snapshot()
                        val committedType = previewMetadata.type
                            ?: if (isFlatPhotoRoomMesh) "photo" else "manual"
                        val metadataFile = File(savedRoomFolder, "metadata.txt")
                        metadataFile.writeText(
                            buildString {
                                append("name=$name\n")
                                append("created=$createdAtMillis\n")
                                append("type=$committedType\n")
                                append("glb=${savedGlb.name}\n")
                                append("roomWidth=$roomWidth\n")
                                append("roomHeight=$roomHeight\n")
                                append("roomDepth=$roomDepth\n")
                                append("photoOrientation=${previewMetadata.normalizedOrientation()}\n")
                                append("photoWideAngle=${previewMetadata.photoWideAngle}\n")
                                previewMetadata.roomDimsApproach?.let {
                                    append("roomDimsApproach=$it\n")
                                }
                                append("previewOnly=false\n")
                                previewMetadata.depthMeshProjectionVersion?.let {
                                    append("depthMeshProjectionVersion=$it\n")
                                }
                                previewMetadata.depthMeshHasCompletedBackground?.let {
                                    append("depthMeshHasCompletedBackground=$it\n")
                                }
                                previewMetadata.depthMeshUsesContinuousSurface?.let {
                                    append("depthMeshUsesContinuousSurface=$it\n")
                                }
                            },
                        )
                        val glbSnapshot = RoomFolderMetadata.snapshotPreservingCalibrationFields(
                            savedRoomFolder,
                            previewMetadata.copy(
                                name = name,
                                createdAt = createdAtMillis,
                                type = committedType,
                                roomWidth = roomWidth,
                                roomHeight = roomHeight,
                                roomDepth = roomDepth,
                                roomDimsApproach = previewMetadata.roomDimsApproach ?: roomDimsApproach,
                                roomSceneWidth = previewMetadata.roomSceneWidth ?: roomWidth,
                                roomSceneHeight = previewMetadata.roomSceneHeight ?: roomHeight,
                                roomSceneDepth = previewMetadata.roomSceneDepth ?: roomDepth,
                                previewOnly = false,
                            ),
                        )
                        RoomFolderMetadata.writeToFolder(savedRoomFolder, glbSnapshot)
                        LogUtil.i(
                            TAG,
                            "Promoted exact preview GLB bytes=${savedGlb.length()} " +
                                "projection=${glbSnapshot.depthMeshProjectionVersion} " +
                                "completedBackground=${glbSnapshot.depthMeshHasCompletedBackground} " +
                                "continuous=${glbSnapshot.depthMeshUsesContinuousSurface}",
                        )
                        previewRoomFolder.parentFile?.deleteRecursively()
                    } catch (error: Exception) {
                        if (!destinationAlreadyExisted) savedRoomFolder.deleteRecursively()
                        throw error
                    }
                }

                overlay.setProgress(1f, getString(R.string.room_viewer_saving_room_ellipsis))
                delay(280)
                PaafektSavingRoomOverlay.hide(rootLayout)

                PaafektSnackbar.showRoomSaved(rootLayout, name)
                LogUtil.d(TAG, "Room saved: $name")

                rootLayout.postDelayed({
                    val intent = Intent(this@GLBRoomActivity, ContentActivity::class.java)
                    intent.flags = Intent.FLAG_ACTIVITY_CLEAR_TOP or Intent.FLAG_ACTIVITY_NEW_TASK
                    startActivity(intent)
                    finish()
                }, 1200)
            } catch (e: Exception) {
                PaafektSavingRoomOverlay.hide(rootLayout)
                LogUtil.e(TAG, "Failed to save room", e)
                Toast.makeText(
                    this@GLBRoomActivity,
                    getString(R.string.glb_room_save_failed_generic),
                    Toast.LENGTH_SHORT,
                ).show()
                CrashReporter.report(this@GLBRoomActivity, e, "GLB room — save room")
            }
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
            startActivity(Intent.createChooser(shareIntent, getString(R.string.share_screenshot_chooser)))

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
                webView.contentDescription = getString(R.string.room_viewer_title)
                ensureNavigationChromeOnTop()
                notifyWebViewViewportChanged()
                LogUtil.d(TAG, "WebGL viewer reported loaded")
            }
        }

        @JavascriptInterface
        fun onError(message: String) {
            runOnUiThread {
                loadingOverlay.visibility = View.GONE
                webView.contentDescription = getString(R.string.room_viewer_preview_unavailable)
                LogUtil.e(TAG, "WebGL error: $message")
                Toast.makeText(
                    this@GLBRoomActivity,
                    R.string.room_viewer_preview_unavailable,
                    Toast.LENGTH_LONG,
                ).show()
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
        inlineBrainArCameraController?.onHostResume()
    }

    override fun onConfigurationChanged(newConfig: Configuration) {
        super.onConfigurationChanged(newConfig)
        LogUtil.d(
            TAG,
            "Viewer configuration changed orientation=${newConfig.orientation} " +
                "viewport=${resources.displayMetrics.widthPixels}x${resources.displayMetrics.heightPixels}",
        )
        rootLayout.post {
            enterImmersiveMode()
            notifyWebViewViewportChanged()
            ensureNavigationChromeOnTop()
        }
    }

    override fun onPause() {
        inlineBrainArCameraController?.onHostPause()
        super.onPause()
    }

    override fun onWindowFocusChanged(hasFocus: Boolean) {
        super.onWindowFocusChanged(hasFocus)
        if (hasFocus) {
            enterImmersiveMode()
            notifyWebViewViewportChanged()
        }
    }

    @SuppressLint("MissingSuperCall")
    override fun onBackPressed() {
        handleBackNavigation()
    }

    override fun onDestroy() {
        saveRoomJob?.cancel()
        roomPaletteLoadJob?.cancel()
        immersiveChrome.destroy()
        stopInlineBrainSegmentation()
        releaseInlineBrainArCamera()
        furnitureFitManager?.close()
        furnitureFitManager = null
        releaseFurnitureFitResourcesForViewerIfNeeded()
        cameraExecutor.shutdown()
        glbViewerCacheDir?.deleteRecursively()
        glbViewerCacheDir = null
        webView.destroy()
        requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_UNSPECIFIED
        super.onDestroy()
    }
}
