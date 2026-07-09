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
import android.util.Base64
import com.furnit.android.utils.CrashReporter
import com.furnit.android.theme.PaafektDrawables
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
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsControllerCompat
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
    private lateinit var brainDetectionOverlay: FrameLayout
    private lateinit var brainDetectionOverlayView: FurnitureFitOverlayView
    private lateinit var brainCameraPreview: PreviewView
    private lateinit var brainProgressOverlay: FrameLayout
    private lateinit var brainProgressLabel: TextView
    private lateinit var brainProgressBar: ProgressBar
    private lateinit var cameraExecutor: ExecutorService
    private var cameraProvider: ProcessCameraProvider? = null
    private var boundPreview: Preview? = null
    private var furnitureFitManager: FurnitureFitManager? = null
    private val isBrainInferenceRunning = AtomicBoolean(false)
    private val brainSessionGeneration = AtomicInteger(0)
    private var brainAcceptingUpdates = false
    private var brainButton: TextView? = null
    private var brainSegmentButton: TextView? = null
    private var brainFullVideoButton: ImageButton? = null
    @Volatile private var inlineBrainMode: InlineBrainMode = InlineBrainMode.DEFAULT_SEGMENT
    @Volatile private var inlineBrainFullVideoEnabled = false
    @Volatile private var inlineBrainSelectedPins: List<DetectionResult> = emptyList()
    private var glbPath: String? = null
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

        rootLayout = FrameLayout(this)
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

        // Top-left camera D-pad (same handlers as iOS GLBRoomView / ModelViewerView).
        cameraDpadOverlay = createCameraDPadOverlay()
        rootLayout.addView(
            cameraDpadOverlay,
            FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT,
            ),
        )

        // Bottom controls
        bottomControls = createBottomControls()
        rootLayout.addView(bottomControls, FrameLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT
        ).apply { gravity = Gravity.BOTTOM })

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
                bottomMargin = dpToPx(100)
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
        ensureNavigationChromeOnTop()

        // Load the WebGL viewer
        loadWebGLViewer()
    }

    /** Keep back / title / recenter / D-pad / bottom brain+camera above the WebView and brain overlay. */
    private fun ensureNavigationChromeOnTop() {
        topBar.elevation = 40f
        cameraDpadOverlay.elevation = 39f
        brainFullVideoButton?.elevation = 38f
        bottomControls.elevation = 37f
        rootLayout.bringChildToFront(bottomControls)
        rootLayout.bringChildToFront(cameraDpadOverlay)
        if (::brainProgressOverlay.isInitialized && brainProgressOverlay.visibility == View.VISIBLE) {
            rootLayout.bringChildToFront(brainProgressOverlay)
        }
        brainFullVideoButton?.takeIf { it.visibility == View.VISIBLE }?.let { rootLayout.bringChildToFront(it) }
        rootLayout.bringChildToFront(topBar)
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
        return FrameLayout(this).apply {
            setPadding(dpToPx(16), dpToPx(48), dpToPx(16), 0)

            val backBtn = createToolbarTextButton("‹") { handleBackNavigation() }.apply {
                textSize = 24f
                contentDescription = getString(R.string.photo_room_back)
            }
            addView(
                backBtn,
                FrameLayout.LayoutParams(dpToPx(36), dpToPx(36)).apply {
                    gravity = Gravity.START or Gravity.TOP
                },
            )

            val principalControls = LinearLayout(this@GLBRoomActivity).apply {
                orientation = LinearLayout.HORIZONTAL
                gravity = Gravity.CENTER_VERTICAL
                background = toolbarCapsuleDrawable()
                setPadding(dpToPx(8), dpToPx(4), dpToPx(8), dpToPx(4))
            }

            principalControls.addView(createToolbarIconButton(R.drawable.ic_ruler) { showRoomDimensionsDialog() })
            principalControls.addView(createToolbarIconButton(R.drawable.ic_gesture_pinch) {
                Toast.makeText(this@GLBRoomActivity, R.string.room_viewer_pinch_hint, Toast.LENGTH_SHORT).show()
            })
            principalControls.addView(createToolbarIconButton(R.drawable.ic_gesture_tap) {
                Toast.makeText(this@GLBRoomActivity, R.string.room_viewer_brain_gesture_hint_explanation, Toast.LENGTH_SHORT).show()
            })

            addView(
                principalControls,
                FrameLayout.LayoutParams(
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                ).apply {
                    gravity = Gravity.TOP or Gravity.CENTER_HORIZONTAL
                },
            )

            val trailingControls = LinearLayout(this@GLBRoomActivity).apply {
                orientation = LinearLayout.HORIZONTAL
                gravity = Gravity.CENTER_VERTICAL
            }
            trailingControls.addView(createToolbarIconButton(R.drawable.ic_viewfinder) { recenterCamera() })
            if (isPreviewMode) {
                trailingControls.addView(createToolbarIconButton(R.drawable.ic_download) { showSaveDialog() })
            }
            trailingControls.addView(createToolbarIconButton(R.drawable.ic_square_resize) {
                openFurnitureFit(enableArAssistedSizing = true)
            })
            addView(
                trailingControls,
                FrameLayout.LayoutParams(
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                ).apply {
                    gravity = Gravity.END or Gravity.TOP
                },
            )

            titleView = TextView(this@GLBRoomActivity).apply {
                visibility = View.GONE
                text = roomName
            }
        }
    }

    private fun toolbarCapsuleDrawable(): GradientDrawable = PaafektDrawables.toolbarCapsule()

    private fun toolbarCircleDrawable(): GradientDrawable = PaafektDrawables.toolbarCircle()

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

    private fun showRoomDimensionsDialog() {
        val dimensions = String.format(
            Locale.US,
            "%.2f m × %.2f m × %.2f m",
            roomWidth,
            roomHeight,
            roomDepth,
        )
        AlertDialog.Builder(this)
            .setTitle(R.string.faq_measurement_pill)
            .setMessage(getString(R.string.room_viewer_dimensions, dimensions))
            .setPositiveButton(android.R.string.ok, null)
            .show()
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
                setOnClickListener { toggleInlineBrainSegmentation() }
            }
            brainButton = brainBtn
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
        try {
            cameraProvider?.unbindAll()
        } catch (_: Exception) {
        }
        cameraProvider = null
        ensureNavigationChromeOnTop()
    }

    private fun setBrainButtonActive(active: Boolean) {
        val color = if (active) "#34C759" else "#007AFF"
        (brainButton?.background as? GradientDrawable)?.setColor(Color.parseColor(color))
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
        const inferenceRoomWidth = $roomWidth;
        const inferenceRoomHeight = $roomHeight;

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
        let roomBoundsForClamping = null;
        let isFlatPhotoMesh = false;
        let flatPhotoWidth = 0;
        let flatPhotoHeight = 0;

        function depthAnythingImagePlaneStandoff(width, height) {
            const halfFovRad = Math.PI / 6.0;
            const viewportAspect = Math.max(window.innerWidth / window.innerHeight, 0.01);
            let fitWidth;
            let fitHeight;
            if (!isPortrait) {
                fitWidth = width / (2 * Math.tan(halfFovRad));
                const verticalHalfFov = Math.atan(Math.tan(halfFovRad) / viewportAspect);
                fitHeight = height / (2 * Math.tan(verticalHalfFov));
            } else {
                fitHeight = height / (2 * Math.tan(halfFovRad));
                const horizontalHalfFov = Math.atan(Math.tan(halfFovRad) * viewportAspect);
                fitWidth = width / (2 * Math.tan(horizontalHalfFov));
            }
            const useCoverFraming = !isPortrait;
            const fitDistance = useCoverFraming
                ? Math.min(fitWidth, fitHeight) * 0.98
                : Math.max(fitWidth, fitHeight) * 1.02;
            return Math.max(fitDistance, 0.85);
        }

        function updateFlatPhotoProjection() {
            const viewportAspect = Math.max(window.innerWidth / window.innerHeight, 0.01);
            if (!isPortrait) {
                const verticalHalfFov = Math.atan(Math.tan(Math.PI / 6.0) / viewportAspect);
                camera.fov = THREE.MathUtils.radToDeg(2 * verticalHalfFov);
            } else {
                camera.fov = 60;
            }
            camera.aspect = viewportAspect;
            camera.updateProjectionMatrix();
        }

        function applyFlatPhotoCamera() {
            if (!isFlatPhotoMesh || flatPhotoWidth <= 0 || flatPhotoHeight <= 0) return;
            updateFlatPhotoProjection();
            const standoff = depthAnythingImagePlaneStandoff(flatPhotoWidth, flatPhotoHeight);
            const camY = flatPhotoHeight * 0.5;
            camera.position.set(0, camY, standoff);
            controls.target.set(0, camY, 0);
            controls.update();
            initialCameraPosition = camera.position.clone();
            initialControlsTarget = controls.target.clone();
            roomBoundsForClamping = {
                minX: -flatPhotoWidth * 0.5 + 0.05,
                maxX: flatPhotoWidth * 0.5 - 0.05,
                minY: 0.05,
                maxY: flatPhotoHeight - 0.05,
                minZ: 0.5,
                maxZ: Math.max(standoff * 1.5, 8.0)
            };
            console.log('[GLBViewer] Flat photo camera standoff=', standoff.toFixed(2),
                'plane=', flatPhotoWidth.toFixed(2), 'x', flatPhotoHeight.toFixed(2));
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

                const roomWidth = size.x;
                const roomHeight = size.y;
                const roomDepth = size.z;
                isFlatPhotoMesh = roomDepth < 0.05;
                const halfDepth = roomDepth * 0.5;

                if (isFlatPhotoMesh) {
                    flatPhotoWidth = Math.max(roomWidth, inferenceRoomWidth);
                    flatPhotoHeight = Math.max(roomHeight, inferenceRoomHeight);
                    applyFlatPhotoCamera();
                } else {
                    const camX = 0;
                    const camY = roomHeight * 0.5;
                    const camZ = 0;
                    const targetX = 0;
                    const targetY = camY;
                    const targetZ = -halfDepth;

                    camera.position.set(camX, camY, camZ);
                    controls.target.set(targetX, targetY, targetZ);
                    controls.update();

                    initialCameraPosition = camera.position.clone();
                    initialControlsTarget = controls.target.clone();

                    const boxWorld = new THREE.Box3().setFromObject(model);
                    roomBoundsForClamping = {
                        minX: boxWorld.min.x + 0.05,
                        maxX: boxWorld.max.x - 0.05,
                        minY: boxWorld.min.y + 0.05,
                        maxY: boxWorld.max.y - 0.05,
                        minZ: boxWorld.min.z + 0.05,
                        maxZ: boxWorld.max.z - 0.02
                    };
                }

                console.log('[GLBViewer] Room size:', roomWidth.toFixed(2), 'x', roomHeight.toFixed(2), 'x', roomDepth.toFixed(2));
                console.log('[GLBViewer] Camera:', camera.position.x.toFixed(2), camera.position.y.toFixed(2), camera.position.z.toFixed(2),
                    'lookAt', controls.target.x.toFixed(2), controls.target.y.toFixed(2), controls.target.z.toFixed(2));

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

        // Handle resize — reframe flat photo rooms (iOS shouldReframeForViewportChange parity).
        window.addEventListener('resize', () => {
            camera.aspect = window.innerWidth / window.innerHeight;
            if (isFlatPhotoMesh) {
                applyFlatPhotoCamera();
            } else {
                camera.updateProjectionMatrix();
            }
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
                ensureNavigationChromeOnTop()
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

    override fun onBackPressed() {
        handleBackNavigation()
    }

    override fun onDestroy() {
        stopInlineBrainSegmentation()
        furnitureFitManager?.close()
        furnitureFitManager = null
        cameraExecutor.shutdown()
        webView.destroy()
        requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_UNSPECIFIED
        super.onDestroy()
    }
}
