package com.furnit.android

import android.Manifest
import android.animation.AnimatorSet
import android.animation.ObjectAnimator
import android.animation.ValueAnimator
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import com.furnit.android.utils.CrashReporter
import com.furnit.android.utils.DebugLogger
import com.furnit.android.utils.LogUtil
import com.furnit.android.utils.RoomFolderMetadata
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import android.view.animation.AccelerateDecelerateInterpolator
import android.widget.*
import androidx.activity.OnBackPressedCallback
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.launch
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import androidx.core.content.FileProvider
import com.google.android.material.button.MaterialButton
import com.google.android.material.progressindicator.CircularProgressIndicator
import com.furnit.android.ar.ArSupportChecker
import com.furnit.android.ar.MetricAnchor
import com.furnit.android.models.PhotoOrientation
import com.furnit.android.models.RoomStructure
import com.furnit.android.services.SharpGenerationUiState
import com.furnit.android.services.SharpService
import com.furnit.android.services.FurnitureFitManager
import org.json.JSONObject
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * SinglePhotoRoomActivity - Image picker with Manual/AI room creation options
 * (Matches Swift's SinglePhotoRoomView)
 *
 * Flow:
 * 1. User picks a photo from gallery
 * 2. Shows preview with two options: Manual Setup or AI Room (Sharp)
 * 3. Manual Setup: boundary adjustment for room creation
 * 4. AI Room: Sharp-based 3D generation
 */
class SinglePhotoRoomActivity : AppCompatActivity() {

    private lateinit var rootLayout: FrameLayout
    private lateinit var initialView: LinearLayout
    private lateinit var methodPickerView: LinearLayout
    private lateinit var progressOverlay: FrameLayout
    /** Bottom bar when AI job is active and the full-screen progress modal is hidden. */
    private lateinit var globalAiProgressBar: FrameLayout
    private lateinit var globalAiProgressLabel: TextView
    private lateinit var progressRing: CircularProgressIndicator
    private lateinit var progressText: TextView
    private lateinit var progressPercent: TextView
    private lateinit var runInBackgroundButton: MaterialButton
    /** Host of the ring — subtle pulse animation. */
    private var progressRingHost: View? = null
    private var progressOverlayPulse: AnimatorSet? = null
    private var phaseStripViews: Array<TextView> = emptyArray()
    private lateinit var selectedImageView: ImageView
    private lateinit var singleImageOverlayView: FurnitureFitOverlayView
    private lateinit var singleImageScanStatusView: TextView
    private lateinit var orientationIndicator: LinearLayout
    private lateinit var orientationIcon: TextView
    private lateinit var orientationText: TextView
    private var selectedBitmap: Bitmap? = null
    private var selectedImageUri: Uri? = null
    private var cameraPhotoUri: Uri? = null
    private var detectedOrientation: PhotoOrientation = PhotoOrientation.PORTRAIT
    /** True after user tapped the orientation row — keeps true landscape for 0.5× shots when needed. */
    private var orientationUserOverridden: Boolean = false
    /** True when the user indicates the photo was taken with the wide-angle (0.5x) lens; fixes camera position in the 3D viewer. */
    private var photoWideAngle: Boolean = false

    /** AI generation started on photo select; cancel and release when user picks Manual/Back/Change. */
    private var aiGenerationHandle: SharpService.GenerationHandle? = null
    private var aiGenerationResult: SharpService.GenerationResult? = null
    private var aiGenerationRunning = false
    /** Set when user taps AI Room while generation is running - callback will show overlay and open on complete. */
    private var aiRoomOverlayRequested = false
    /** After "Run in background", AI option line shows percent; otherwise only friendly text (no %). */
    private var aiOptionShowPercent = false
    /** Bumped on cancel/restart so stale [SharpService.ProgressCallback] completions are ignored and folders deleted. */
    private var aiSessionId: Int = 0
    private var pendingMetricAnchors: ArrayList<MetricAnchor>? = null
    private val furnitureFitManager by lazy { FurnitureFitManager(this) }
    private var furnitureFitInitialized = false
    private var singleImageScanRequestId = 0

    private val imagePickerLauncher = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        if (uri != null) {
            DebugLogger.d("SinglePhotoRoom", "Image selected: $uri")
            selectedImageUri = uri
            pendingMetricAnchors = null
            loadImageFromUri(uri)
        } else {
            DebugLogger.d("SinglePhotoRoom", "No image selected")
        }
    }

    private val cameraLauncher = registerForActivityResult(
        ActivityResultContracts.TakePicture()
    ) { success: Boolean ->
        if (success && cameraPhotoUri != null) {
            DebugLogger.d("SinglePhotoRoom", "Photo captured: $cameraPhotoUri")
            selectedImageUri = cameraPhotoUri
            loadImageFromUri(cameraPhotoUri!!)
        } else {
            DebugLogger.d("SinglePhotoRoom", "Camera capture cancelled or failed")
        }
    }

    private val cameraPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted: Boolean ->
        if (isGranted) {
            DebugLogger.d("SinglePhotoRoom", "Camera permission granted")
            launchCamera()
        } else {
            DebugLogger.d("SinglePhotoRoom", "Camera permission denied")
            Toast.makeText(this, "Camera permission is required to take photos", Toast.LENGTH_SHORT).show()
        }
    }

    private val arPhotoCaptureLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        if (result.resultCode != RESULT_OK) {
            DebugLogger.d("SinglePhotoRoom", "AR photo capture cancelled")
            return@registerForActivityResult
        }
        val data = result.data
        val imageUriString = data?.getStringExtra(ArDepthPhotoCaptureActivity.EXTRA_CAPTURED_IMAGE_URI)
        val anchors: ArrayList<MetricAnchor>? = data?.extras?.let { bundle ->
            @Suppress("UNCHECKED_CAST", "DEPRECATION")
            (if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                bundle.getSerializable(ArDepthPhotoCaptureActivity.EXTRA_METRIC_ANCHORS, ArrayList::class.java)
            } else {
                bundle.getSerializable(ArDepthPhotoCaptureActivity.EXTRA_METRIC_ANCHORS)
            }) as? ArrayList<MetricAnchor>
        }
        if (imageUriString.isNullOrBlank()) {
            DebugLogger.d("SinglePhotoRoom", "AR photo capture missing image uri")
            Toast.makeText(this, "AR photo capture failed", Toast.LENGTH_SHORT).show()
            return@registerForActivityResult
        }
        pendingMetricAnchors = anchors
        selectedImageUri = Uri.parse(imageUriString)
        DebugLogger.d("SinglePhotoRoom", "AR photo captured with anchors=${anchors?.size ?: 0}")
        loadImageFromUri(selectedImageUri!!)
    }

    private val boundaryActivityLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        if (result.resultCode == RESULT_OK) {
            DebugLogger.d("SinglePhotoRoom", "Boundary adjustment completed")
            // TODO: Get boundaries from result and process room
            // val boundaries = result.data?.getSerializableExtra(RoomBoundaryActivity.RESULT_BOUNDARIES) as? RoomStructure
        }
    }

    private val sharpRoomLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult(),
    ) {
        if (!isDestroyed) {
            showInitialView()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        rootLayout = FrameLayout(this)
        rootLayout.setBackgroundColor(Color.parseColor("#F5F5F5"))

        // Initial view - photo selection
        initialView = createInitialView()
        rootLayout.addView(initialView)

        // Method picker view - hidden initially
        methodPickerView = createMethodPickerView()
        methodPickerView.visibility = View.GONE
        rootLayout.addView(methodPickerView)

        // Full-screen modal first; global bar added last so it stays on top (progress + Stop always visible).
        progressOverlay = createProgressOverlay()
        progressOverlay.visibility = View.GONE
        rootLayout.addView(progressOverlay)

        setContentView(rootLayout)

        // Sibling of rootLayout under android.R.id.content so the strip stays above both
        // Create 3D Room and method-picker full-screen views (not buried under them).
        val contentRoot = findViewById<FrameLayout>(android.R.id.content)
        globalAiProgressBar = createGlobalAiProgressBar()
        contentRoot.addView(
            globalAiProgressBar,
            FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT,
                Gravity.BOTTOM,
            ),
        )

        onBackPressedDispatcher.addCallback(
            this,
            object : OnBackPressedCallback(true) {
                override fun handleOnBackPressed() {
                    if (progressOverlay.visibility == View.VISIBLE) {
                        aiRoomOverlayRequested = false
                        hideProgressOverlay()
                        updateAIOptionProgress(lastAIGenerationProgress, lastAIGenerationRawMessage)
                        return
                    }
                    if (methodPickerView.visibility == View.VISIBLE) {
                        showMethodPickerBackConfirmation()
                        return
                    } else {
                        finish()
                    }
                }
            },
        )

        // Preload ExecuTorch Part1 when backend is ExecuTorch (hides "stuck at 5%" stall at Generate)
        lifecycleScope.launch {
            SharpService.getInstance(this@SinglePhotoRoomActivity).preloadSharpModels()
        }
    }

    override fun onResume() {
        super.onResume()
        refreshGlobalAiProgressUi()
    }

    override fun onDestroy() {
        if (furnitureFitInitialized) {
            furnitureFitManager.close()
        }
        super.onDestroy()
    }

    /** Keeps the bottom strip visible and on top after navigation or window insets change. */
    private fun refreshGlobalAiProgressUi() {
        updateGlobalAiProgressOverlay()
    }

    private fun showMethodPickerBackConfirmation() {
        AlertDialog.Builder(this)
            .setTitle(R.string.photo_room_back_confirm_title)
            .setMessage(
                getString(
                    R.string.photo_room_back_confirm_message,
                    getString(R.string.photo_room_ai_room),
                    getString(R.string.photo_room_manual_setup),
                ),
            )
            .setNegativeButton(R.string.photo_room_back_stay, null)
            .setPositiveButton(R.string.photo_room_back_leave) { _, _ ->
                if (aiGenerationRunning || aiGenerationResult != null) {
                    finish()
                } else {
                    showInitialView()
                }
            }
            .show()
    }

    private fun createInitialView(): LinearLayout {
        return LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            layoutParams = FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT
            )
            gravity = Gravity.CENTER_HORIZONTAL
            setPadding(48, 80, 48, 48)

            // Back button
            val backBtn = TextView(this@SinglePhotoRoomActivity).apply {
                text = "< Back"
                textSize = 16f
                setTextColor(Color.parseColor("#007AFF"))
                setPadding(0, 0, 0, 32)
                setOnClickListener { onBackPressedDispatcher.onBackPressed() }
            }
            addView(backBtn, LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            ))

            // Title
            val title = TextView(this@SinglePhotoRoomActivity).apply {
                text = "Create 3D Room"
                textSize = 24f
                setTypeface(null, Typeface.BOLD)
                setTextColor(Color.parseColor("#333333"))
                gravity = Gravity.CENTER
            }
            addView(title)

            // Subtitle
            val subtitle = TextView(this@SinglePhotoRoomActivity).apply {
                text = "Capture or select a photo of your room"
                textSize = 16f
                setTextColor(Color.parseColor("#666666"))
                gravity = Gravity.CENTER
                setPadding(0, 16, 0, 32)
            }
            addView(subtitle)

            // Take Photo button (Camera)
            val takePhotoBtn = LinearLayout(this@SinglePhotoRoomActivity).apply {
                orientation = LinearLayout.VERTICAL
                gravity = Gravity.CENTER
                setPadding(48, 36, 48, 36)
                val bg = GradientDrawable().apply {
                    shape = GradientDrawable.RECTANGLE
                    cornerRadius = dpToPx(16).toFloat()
                    setColor(Color.parseColor("#E3F2FD"))
                    setStroke(dpToPx(2), Color.parseColor("#2196F3"))
                }
                background = bg

                val icon = TextView(this@SinglePhotoRoomActivity).apply {
                    text = "\uD83D\uDCF7" // Camera icon
                    textSize = 48f
                    gravity = Gravity.CENTER
                }
                addView(icon)

                val btnText = TextView(this@SinglePhotoRoomActivity).apply {
                    text = "Take a Photo"
                    textSize = 18f
                    setTypeface(null, Typeface.BOLD)
                    setTextColor(Color.parseColor("#2196F3"))
                    gravity = Gravity.CENTER
                    setPadding(0, 16, 0, 0)
                }
                addView(btnText)

                val btnHint = TextView(this@SinglePhotoRoomActivity).apply {
                    text = "Use your camera"
                    textSize = 14f
                    setTextColor(Color.parseColor("#666666"))
                    gravity = Gravity.CENTER
                    setPadding(0, 8, 0, 0)
                }
                addView(btnHint)

                setOnClickListener {
                    checkCameraPermissionAndLaunch()
                }
            }
            addView(takePhotoBtn, LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            ).apply { setMargins(0, 0, 0, 24) })

            // "or" divider
            val dividerRow = LinearLayout(this@SinglePhotoRoomActivity).apply {
                orientation = LinearLayout.HORIZONTAL
                gravity = Gravity.CENTER_VERTICAL
                setPadding(32, 0, 32, 0)

                val leftLine = View(this@SinglePhotoRoomActivity).apply {
                    setBackgroundColor(Color.parseColor("#CCCCCC"))
                }
                addView(leftLine, LinearLayout.LayoutParams(0, dpToPx(1), 1f))

                val orText = TextView(this@SinglePhotoRoomActivity).apply {
                    text = "or"
                    textSize = 14f
                    setTextColor(Color.parseColor("#999999"))
                    setPadding(dpToPx(16), 0, dpToPx(16), 0)
                }
                addView(orText)

                val rightLine = View(this@SinglePhotoRoomActivity).apply {
                    setBackgroundColor(Color.parseColor("#CCCCCC"))
                }
                addView(rightLine, LinearLayout.LayoutParams(0, dpToPx(1), 1f))
            }
            addView(dividerRow, LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            ).apply { setMargins(0, 0, 0, 24) })

            // Photo selection button (Library)
            val selectPhotoBtn = LinearLayout(this@SinglePhotoRoomActivity).apply {
                orientation = LinearLayout.VERTICAL
                gravity = Gravity.CENTER
                setPadding(48, 36, 48, 36)
                val bg = GradientDrawable().apply {
                    shape = GradientDrawable.RECTANGLE
                    cornerRadius = dpToPx(16).toFloat()
                    setColor(Color.parseColor("#E8F5E9"))
                    setStroke(dpToPx(2), Color.parseColor("#4CAF50"))
                }
                background = bg

                val icon = TextView(this@SinglePhotoRoomActivity).apply {
                    text = "\uD83D\uDDBC️" // Frame icon
                    textSize = 48f
                    gravity = Gravity.CENTER
                }
                addView(icon)

                val btnText = TextView(this@SinglePhotoRoomActivity).apply {
                    text = "Select Photo"
                    textSize = 18f
                    setTypeface(null, Typeface.BOLD)
                    setTextColor(Color.parseColor("#4CAF50"))
                    gravity = Gravity.CENTER
                    setPadding(0, 16, 0, 0)
                }
                addView(btnText)

                val btnHint = TextView(this@SinglePhotoRoomActivity).apply {
                    text = "From your photo library"
                    textSize = 14f
                    setTextColor(Color.parseColor("#666666"))
                    gravity = Gravity.CENTER
                    setPadding(0, 8, 0, 0)
                }
                addView(btnHint)

                setOnClickListener {
                    openImagePicker()
                }
            }
            addView(selectPhotoBtn, LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            ).apply { setMargins(0, 0, 0, 24) })

            // Warning
            val warning = TextView(this@SinglePhotoRoomActivity).apply {
                text = "⚠️ Do not use screenshots - use actual photos"
                textSize = 14f
                setTextColor(Color.parseColor("#F44336"))
                gravity = Gravity.CENTER
            }
            addView(warning)
        }
    }

    private fun createMethodPickerView(): LinearLayout {
        return LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            layoutParams = FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT
            )
            setBackgroundColor(Color.BLACK)

            val previewContainer = FrameLayout(this@SinglePhotoRoomActivity).apply {
                layoutParams = LinearLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    0,
                    1f,
                )
                setBackgroundColor(Color.BLACK)
            }

            selectedImageView = ImageView(this@SinglePhotoRoomActivity).apply {
                scaleType = ImageView.ScaleType.FIT_CENTER
                setBackgroundColor(Color.BLACK)
                adjustViewBounds = true
            }
            previewContainer.addView(
                selectedImageView,
                FrameLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.MATCH_PARENT,
                ),
            )

            singleImageOverlayView = FurnitureFitOverlayView(this@SinglePhotoRoomActivity).apply {
                setBackgroundColor(Color.TRANSPARENT)
                setDetectionBoxVisibility(true)
            }
            previewContainer.addView(
                singleImageOverlayView,
                FrameLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.MATCH_PARENT,
                ),
            )

            val topBar = LinearLayout(this@SinglePhotoRoomActivity).apply {
                orientation = LinearLayout.HORIZONTAL
                gravity = Gravity.CENTER_VERTICAL
                setPadding(dpToPx(16), dpToPx(18), dpToPx(16), dpToPx(12))
                background = GradientDrawable(
                    GradientDrawable.Orientation.TOP_BOTTOM,
                    intArrayOf(Color.argb(180, 0, 0, 0), Color.TRANSPARENT),
                )
            }

            val backBtn = TextView(this@SinglePhotoRoomActivity).apply {
                text = getString(R.string.photo_room_back)
                textSize = 16f
                setTypeface(null, Typeface.BOLD)
                setTextColor(Color.WHITE)
                setPadding(dpToPx(12), dpToPx(10), dpToPx(12), dpToPx(10))
                background = GradientDrawable().apply {
                    shape = GradientDrawable.RECTANGLE
                    cornerRadius = dpToPx(20).toFloat()
                    setColor(Color.argb(150, 0, 0, 0))
                }
                setOnClickListener { showMethodPickerBackConfirmation() }
            }
            topBar.addView(backBtn)

            topBar.addView(
                Space(this@SinglePhotoRoomActivity),
                LinearLayout.LayoutParams(0, 0, 1f),
            )

            val changePhotoBtn = TextView(this@SinglePhotoRoomActivity).apply {
                text = "Choose Different Photo"
                textSize = 14f
                setTextColor(Color.WHITE)
                setPadding(dpToPx(12), dpToPx(10), dpToPx(12), dpToPx(10))
                background = GradientDrawable().apply {
                    shape = GradientDrawable.RECTANGLE
                    cornerRadius = dpToPx(20).toFloat()
                    setColor(Color.argb(150, 0, 0, 0))
                }
                setOnClickListener { openImagePicker() }
            }
            topBar.addView(changePhotoBtn)

            previewContainer.addView(
                topBar,
                FrameLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                    Gravity.TOP,
                ),
            )

            singleImageScanStatusView = TextView(this@SinglePhotoRoomActivity).apply {
                text = "Scanning selected image…"
                textSize = 14f
                setTextColor(Color.WHITE)
                setPadding(dpToPx(14), dpToPx(10), dpToPx(14), dpToPx(10))
                background = GradientDrawable().apply {
                    shape = GradientDrawable.RECTANGLE
                    cornerRadius = dpToPx(18).toFloat()
                    setColor(Color.argb(170, 0, 0, 0))
                }
                visibility = View.GONE
            }
            previewContainer.addView(
                singleImageScanStatusView,
                FrameLayout.LayoutParams(
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                    Gravity.BOTTOM or Gravity.CENTER_HORIZONTAL,
                ).apply {
                    bottomMargin = dpToPx(18)
                },
            )

            addView(previewContainer)

            val controlsContainer = ScrollView(this@SinglePhotoRoomActivity).apply {
                setBackgroundColor(Color.parseColor("#F5F5F5"))
                overScrollMode = View.OVER_SCROLL_NEVER
            }

            val controlsContent = LinearLayout(this@SinglePhotoRoomActivity).apply {
                orientation = LinearLayout.VERTICAL
                setPadding(24, 20, 24, 24)
                background = GradientDrawable().apply {
                    shape = GradientDrawable.RECTANGLE
                    setColor(Color.parseColor("#F5F5F5"))
                    cornerRadii = floatArrayOf(
                        dpToPx(24).toFloat(), dpToPx(24).toFloat(),
                        dpToPx(24).toFloat(), dpToPx(24).toFloat(),
                        0f, 0f, 0f, 0f,
                    )
                }
            }

            // Orientation indicator (tap to override auto-detection)
            orientationIndicator = LinearLayout(this@SinglePhotoRoomActivity).apply {
                orientation = LinearLayout.HORIZONTAL
                gravity = Gravity.CENTER
                setPadding(16, 8, 16, 8)
                setBackgroundColor(Color.parseColor("#F0F0F0"))
                isClickable = true
                isFocusable = true
                setOnClickListener {
                    orientationUserOverridden = true
                    detectedOrientation = if (detectedOrientation.isLandscape) PhotoOrientation.PORTRAIT else PhotoOrientation.LANDSCAPE
                    updateOrientationIndicator()
                    DebugLogger.d("SinglePhotoRoom", "User overrode orientation to: ${detectedOrientation.value}")
                    if (aiGenerationRunning && selectedBitmap != null) {
                        startAIGenerationInBackground(selectedBitmap!!)
                    }
                }

                orientationIcon = TextView(this@SinglePhotoRoomActivity).apply {
                    text = "\uD83D\uDCF1"
                    textSize = 16f
                    setPadding(0, 0, 8, 0)
                }
                addView(orientationIcon)

                orientationText = TextView(this@SinglePhotoRoomActivity).apply {
                    text = "Portrait - held vertically"
                    textSize = 13f
                    setTextColor(Color.parseColor("#666666"))
                }
                addView(orientationText)
            }
            controlsContent.addView(
                orientationIndicator,
                LinearLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                ).apply { setMargins(0, 0, 0, 8) },
            )

            val wideAngleRow = LinearLayout(this@SinglePhotoRoomActivity).apply {
                orientation = LinearLayout.HORIZONTAL
                gravity = Gravity.CENTER_VERTICAL
                setPadding(16, 8, 16, 8)
                setBackgroundColor(Color.parseColor("#F5F5F5"))
                isClickable = true
                isFocusable = true
                setOnClickListener {
                    photoWideAngle = !photoWideAngle
                    if (photoWideAngle && !orientationUserOverridden) {
                        detectedOrientation = PhotoOrientation.coercePortraitForUltraWide(detectedOrientation, true)
                        updateOrientationIndicator()
                    }
                    updateWideAngleIndicator(this)
                    DebugLogger.d("SinglePhotoRoom", "Wide angle (0.5x): $photoWideAngle")
                    if (aiGenerationRunning && selectedBitmap != null) {
                        startAIGenerationInBackground(selectedBitmap!!)
                    }
                }
                val wideIcon = TextView(this@SinglePhotoRoomActivity).apply {
                    text = "\uD83D\uDCF8"
                    textSize = 16f
                    setPadding(0, 0, 8, 0)
                }
                addView(wideIcon)
                val wideText = TextView(this@SinglePhotoRoomActivity).apply {
                    text = getString(R.string.camera_wide_angle_desc)
                    textSize = 13f
                    setTextColor(Color.parseColor("#666666"))
                }
                addView(wideText)
                val wideCheck = TextView(this@SinglePhotoRoomActivity).apply {
                    setTag("wide_check")
                    text = if (photoWideAngle) " \u2713" else " "
                    textSize = 16f
                    setTextColor(Color.parseColor("#4CAF50"))
                    setPadding(dpToPx(8), 0, 0, 0)
                }
                addView(wideCheck)
            }
            controlsContent.addView(
                wideAngleRow,
                LinearLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                ).apply { setMargins(0, 0, 0, 16) },
            )

            val title = TextView(this@SinglePhotoRoomActivity).apply {
                text = "How would you like to create your room?"
                textSize = 18f
                setTypeface(null, Typeface.BOLD)
                setTextColor(Color.parseColor("#333333"))
                gravity = Gravity.CENTER
            }
            controlsContent.addView(title)

            val subtitle = TextView(this@SinglePhotoRoomActivity).apply {
                text = "Tap an option below"
                textSize = 14f
                setTextColor(Color.parseColor("#666666"))
                gravity = Gravity.CENTER
                setPadding(0, 8, 0, 24)
            }
            controlsContent.addView(subtitle)

            val aiOption = createOptionCard(
                icon = "\uD83E\uDE84",
                title = "AI Room",
                subtitle = getString(R.string.single_photo_ai_room_subtitle_idle),
                bgColor = "#F3E5F5",
                footnote = getString(R.string.single_photo_ai_room_async_footnote),
                onSubtitleCreated = { view -> aiOptionSubtitleView = view },
                onStopButtonCreated = { view -> aiStopButtonView = view },
            ) {
                onAIRoomSelected()
            }
            controlsContent.addView(aiOption)

            val manualOption = createOptionCard(
                icon = "\uD83D\uDCCF",
                title = "Manual Setup",
                subtitle = "Adjust room boundaries manually",
                bgColor = "#FFF3E0",
            ) {
                onManualSetupSelected()
            }
            controlsContent.addView(manualOption)

            controlsContainer.addView(
                controlsContent,
                FrameLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                ),
            )
            addView(
                controlsContainer,
                LinearLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                ),
            )
        }
    }

    private fun createOptionCard(
        icon: String,
        title: String,
        subtitle: String,
        bgColor: String,
        footnote: String? = null,
        onSubtitleCreated: ((TextView) -> Unit)? = null,
        onStopButtonCreated: ((TextView) -> Unit)? = null,
        onClick: () -> Unit
    ): LinearLayout {
        return LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            setBackgroundColor(Color.parseColor(bgColor))
            setPadding(24, 24, 24, 24)
            gravity = Gravity.CENTER_VERTICAL
            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            ).apply { setMargins(0, 0, 0, 16) }

            // Icon
            val iconView = TextView(this@SinglePhotoRoomActivity).apply {
                text = icon
                textSize = 32f
            }
            addView(iconView, LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            ).apply { setMargins(0, 0, 24, 0) })

            // Text container
            val textContainer = LinearLayout(this@SinglePhotoRoomActivity).apply {
                orientation = LinearLayout.VERTICAL
                layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)

                val titleView = TextView(this@SinglePhotoRoomActivity).apply {
                    text = title
                    textSize = 16f
                    setTypeface(null, Typeface.BOLD)
                    setTextColor(Color.parseColor("#333333"))
                }
                addView(titleView)

                val subtitleView = TextView(this@SinglePhotoRoomActivity).apply {
                    text = subtitle
                    textSize = 12f
                    setTextColor(Color.parseColor("#666666"))
                }
                addView(subtitleView)
                onSubtitleCreated?.invoke(subtitleView)
                if (footnote != null) {
                    addView(TextView(this@SinglePhotoRoomActivity).apply {
                        text = footnote
                        textSize = 11f
                        setTextColor(Color.parseColor("#888888"))
                        setPadding(0, dpToPx(6), 0, 0)
                    })
                }
            }
            addView(textContainer)

            if (onStopButtonCreated != null) {
                val stopBtn = TextView(this@SinglePhotoRoomActivity).apply {
                    text = "⏹"
                    textSize = 11f
                    setTextColor(Color.WHITE)
                    gravity = Gravity.CENTER
                    setPadding(dpToPx(6), dpToPx(4), dpToPx(6), dpToPx(4))
                    background = GradientDrawable().apply {
                        shape = GradientDrawable.RECTANGLE
                        cornerRadius = dpToPx(4).toFloat()
                        setColor(Color.parseColor("#E53935"))
                    }
                    visibility = View.GONE
                    contentDescription = getString(R.string.single_photo_ai_stop)
                    setOnClickListener { onAIStopClicked() }
                }
                onStopButtonCreated(stopBtn)
                addView(
                    stopBtn,
                    LinearLayout.LayoutParams(
                        ViewGroup.LayoutParams.WRAP_CONTENT,
                        ViewGroup.LayoutParams.WRAP_CONTENT,
                    ).apply {
                        gravity = Gravity.CENTER_VERTICAL
                        setMargins(0, 0, dpToPx(8), 0)
                    },
                )
            }

            // Chevron
            val chevron = TextView(this@SinglePhotoRoomActivity).apply {
                text = ">"
                textSize = 18f
                setTextColor(Color.parseColor("#999999"))
            }
            addView(chevron)

            setOnClickListener { onClick() }
        }
    }

    private fun openImagePicker() {
        DebugLogger.d("SinglePhotoRoom", "Opening image picker")
        imagePickerLauncher.launch("image/*")
    }

    private fun checkCameraPermissionAndLaunch() {
        when {
            ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED -> {
                DebugLogger.d("SinglePhotoRoom", "Camera permission already granted")
                launchCamera()
            }
            else -> {
                DebugLogger.d("SinglePhotoRoom", "Requesting camera permission")
                cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
            }
        }
    }

    private fun launchCamera() {
        if (ArSupportChecker.isArCoreSupported(this)) {
            arPhotoCaptureLauncher.launch(Intent(this, ArDepthPhotoCaptureActivity::class.java))
            return
        }
        try {
            val photoFile = createImageFile()
            cameraPhotoUri = FileProvider.getUriForFile(
                this,
                "${packageName}.fileprovider",
                photoFile
            )
            DebugLogger.d("SinglePhotoRoom", "Launching camera with URI: $cameraPhotoUri")
            cameraLauncher.launch(cameraPhotoUri)
        } catch (e: Exception) {
            DebugLogger.eDebugMode("SinglePhotoRoom", "Error launching camera", e)
            Toast.makeText(this, "Error opening camera: ${e.message}", Toast.LENGTH_SHORT).show()
            CrashReporter.report(this, e, "Single photo room — launch camera")
        }
    }

    private fun createImageFile(): File {
        val timeStamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())
        val storageDir = getExternalFilesDir(Environment.DIRECTORY_PICTURES)
        return File.createTempFile(
            "ROOM_${timeStamp}_",
            ".jpg",
            storageDir
        ).also {
            DebugLogger.d("SinglePhotoRoom", "Created temp file: ${it.absolutePath}")
        }
    }

    private fun dpToPx(dp: Int): Int {
        return (dp * resources.displayMetrics.density).toInt()
    }

    private fun loadImageFromUri(uri: Uri) {
        try {
            val bitmap = PhotoOrientation.loadBitmapApplyingExif(this, uri)

            if (bitmap != null) {
                selectedBitmap = bitmap
                selectedImageView.setImageBitmap(bitmap)
                resetSingleImageOverlay()
                orientationUserOverridden = false
                photoWideAngle = false

                // Must match bitmap pixels fed to SHARP (see PhotoOrientation.fromBitmapDimensions KDoc).
                detectedOrientation = PhotoOrientation.fromBitmapDimensions(bitmap)
                DebugLogger.d(
                    "SinglePhotoRoom",
                    "Orientation from bitmap ${bitmap.width}x${bitmap.height}: ${detectedOrientation.value}"
                )
                updateOrientationIndicator()

                // Start AI generation in background immediately (ONNX or Native Pt)
                startAIGenerationInBackground(bitmap)
                startSingleImageOverlayScan(bitmap)

                showMethodPicker()
                DebugLogger.d("SinglePhotoRoom", "Image loaded: ${bitmap.width}x${bitmap.height}, AI started in background")
            } else {
                DebugLogger.eDebugMode("SinglePhotoRoom", "Failed to decode image")
                Toast.makeText(this, "Failed to load image", Toast.LENGTH_SHORT).show()
                CrashReporter.report(this, IllegalStateException("Bitmap decode returned null"), "Single photo room — decode image")
            }
        } catch (e: Exception) {
            DebugLogger.eDebugMode("SinglePhotoRoom", "Error loading image", e)
            Toast.makeText(this, "Error: ${e.message}", Toast.LENGTH_SHORT).show()
            CrashReporter.report(this, e, "Single photo room — load image")
        }
    }

    /** Matches room metadata / SharpService (ultra-wide portrait bias unless user locked orientation). */
    private fun metadataOrientationStringForViewer(): String {
        val o = if (orientationUserOverridden) {
            detectedOrientation
        } else {
            PhotoOrientation.coercePortraitForUltraWide(detectedOrientation, photoWideAngle)
        }
        return if (o.isLandscape) "landscape" else "portrait"
    }

    private fun updateOrientationIndicator() {
        orientationIcon.text = "\uD83D\uDCF1" // Phone icon
        orientationIcon.rotation = if (detectedOrientation.isLandscape) 90f else 0f

        val orientationLabel = if (detectedOrientation.isLandscape) "Landscape" else "Portrait"
        val heldLabel = if (detectedOrientation.isLandscape) "held horizontally" else "held vertically"
        orientationText.text = getString(R.string.orientation_tap_to_change, orientationLabel, heldLabel)
    }

    private fun updateWideAngleIndicator(wideAngleRow: ViewGroup) {
        (wideAngleRow.findViewWithTag("wide_check") as? TextView)?.text = if (photoWideAngle) " \u2713" else " "
    }

    private fun showMethodPicker() {
        initialView.visibility = View.GONE
        methodPickerView.visibility = View.VISIBLE
        rootLayout.post { refreshGlobalAiProgressUi() }
    }

    private fun resetSingleImageOverlay() {
        if (::singleImageOverlayView.isInitialized) {
            singleImageOverlayView.setMaskAndDetections(null, emptyList(), 640)
            singleImageOverlayView.resetTransform()
        }
        if (::singleImageScanStatusView.isInitialized) {
            singleImageScanStatusView.visibility = View.GONE
            singleImageScanStatusView.text = "Scanning selected image…"
        }
    }

    private fun startSingleImageOverlayScan(bitmap: Bitmap) {
        if (!::singleImageOverlayView.isInitialized || !::singleImageScanStatusView.isInitialized) return
        val requestId = ++singleImageScanRequestId
        singleImageScanStatusView.text = "Scanning selected image…"
        singleImageScanStatusView.visibility = View.VISIBLE
        lifecycleScope.launch {
            val initialized = if (furnitureFitInitialized) {
                true
            } else {
                withContext(Dispatchers.IO) {
                    furnitureFitManager.initializeAuto()
                }.also { success ->
                    furnitureFitInitialized = success
                }
            }

            if (requestId != singleImageScanRequestId || isDestroyed) return@launch

            if (!initialized) {
                singleImageScanStatusView.text = "Overlay unavailable"
                return@launch
            }

            furnitureFitManager.segmentWithDetectionsAsync(bitmap) { result ->
                if (requestId != singleImageScanRequestId || isDestroyed) return@segmentWithDetectionsAsync
                runOnUiThread {
                    if (result == null) {
                        singleImageOverlayView.setMaskAndDetections(null, emptyList(), 640)
                        singleImageScanStatusView.text = "No overlay detected"
                        return@runOnUiThread
                    }
                    singleImageOverlayView.setMaskAndDetections(
                        result.mask,
                        result.detections,
                        result.inputSize,
                    )
                    singleImageScanStatusView.text =
                        if (result.detections.isEmpty()) "No objects detected"
                        else "Overlay ready: ${result.detections.first().label}"
                    singleImageScanStatusView.postDelayed({
                        if (requestId == singleImageScanRequestId && !isDestroyed) {
                            singleImageScanStatusView.visibility = View.GONE
                        }
                    }, 1800L)
                }
            }
        }
    }

    private fun deleteSharpRoomFolder(result: SharpService.GenerationResult?) {
        val parent = result?.plyFile?.parentFile ?: return
        val disk = runCatching { RoomFolderMetadata.readFromFolder(parent) }.getOrNull()
        if (disk?.previewOnly == false) {
            DebugLogger.d("SinglePhotoRoom", "Skip delete — room already on Home list: ${parent.absolutePath}")
            return
        }
        try {
            if (parent.exists()) parent.deleteRecursively()
            DebugLogger.d("SinglePhotoRoom", "Deleted SHARP room folder: ${parent.absolutePath}")
        } catch (e: Exception) {
            DebugLogger.eDebugMode("SinglePhotoRoom", "Failed to delete room folder", e)
        }
    }

    /** User tapped Stop — remove folder even if it was promoted to Home ([deleteSharpRoomFolder] would skip). */
    private fun deleteSharpRoomFolderUnconditional(result: SharpService.GenerationResult?) {
        val parent = result?.plyFile?.parentFile ?: return
        try {
            if (parent.exists()) parent.deleteRecursively()
            DebugLogger.d("SinglePhotoRoom", "Deleted SHARP room folder (stop): ${parent.absolutePath}")
        } catch (e: Exception) {
            DebugLogger.eDebugMode("SinglePhotoRoom", "Failed to delete room folder (stop)", e)
        }
    }

    private fun updateAiStopButtonVisibility() {
        val show = aiGenerationRunning || aiGenerationResult != null
        aiStopButtonView?.visibility = if (show) View.VISIBLE else View.GONE
    }

    private fun onAIStopClicked() {
        if (!aiGenerationRunning && aiGenerationResult == null) return
        DebugLogger.d("SinglePhotoRoom", "AI Stop tapped")
        hideProgressOverlay()
        aiRoomOverlayRequested = false
        aiGenerationHandle?.cancel()
        aiGenerationHandle = null
        aiGenerationResult?.let { deleteSharpRoomFolderUnconditional(it) }
        aiGenerationResult = null
        aiGenerationRunning = false
        aiOptionShowPercent = false
        lastAIGenerationRawMessage = ""
        lastAIGenerationProgress = 0f
        lastAIGenerationMessage = "Getting started…"
        aiSessionId++
        SharpService.getInstance(this).release()
        aiOptionSubtitleView?.text = getString(R.string.single_photo_ai_room_subtitle_idle)
        updateAiStopButtonVisibility()
        updateGlobalAiProgressOverlay()
    }

    /** After \"Run in background\", mark folder so [ModelManager] shows it on Home (not preview-only). */
    private fun promoteSharpRoomToLibrary(result: SharpService.GenerationResult) {
        val folder = result.plyFile.parentFile ?: return
        try {
            val prev = RoomFolderMetadata.readFromFolder(folder) ?: return
            val updated = RoomFolderMetadata.snapshotPreservingYoloFields(
                folder,
                prev.copy(previewOnly = false),
            )
            RoomFolderMetadata.writeToFolder(folder, updated)
            val metaTxt = File(folder, "metadata.txt")
            if (metaTxt.exists()) {
                val lines = metaTxt.readLines().map { line ->
                    if (line.trimStart().startsWith("previewOnly=")) "previewOnly=false" else line
                }
                metaTxt.writeText(lines.joinToString("\n").trimEnd() + "\n")
            }
            LogUtil.i("SinglePhotoRoom", "Promoted SHARP room to Home list: ${folder.absolutePath}")
        } catch (e: Exception) {
            DebugLogger.eDebugMode("SinglePhotoRoom", "promoteSharpRoomToLibrary failed", e)
        }
    }

    /** Start AI generation in background when photo is selected. Cancel on Manual/Back/Change. */
    private fun startAIGenerationInBackground(bitmap: Bitmap) {
        cancelAndReleaseAI()
        val session = aiSessionId
        aiGenerationResult = null
        aiGenerationRunning = true
        val sharpService = SharpService.getInstance(this)
        val orientationForMetadata = metadataOrientationStringForViewer()
        aiGenerationHandle = sharpService.startGenerationInBackground(
            bitmap,
            object : SharpService.ProgressCallback {
            override fun onProgress(progress: Float, message: String) {
                runOnUiThread {
                    if (session != aiSessionId) return@runOnUiThread
                    logProgress0("SinglePhotoRoomActivity.kt:onProgress", "callback", mapOf(
                        "progress" to progress, "message" to message, "aiGenerationRunning" to aiGenerationRunning,
                        "aiRoomOverlayRequested" to aiRoomOverlayRequested
                    ))
                    if (aiGenerationRunning) {
                        updateAIOptionProgress(progress, message)
                        if (aiRoomOverlayRequested && !isDestroyed) updateProgressOverlay(progress, message)
                    }
                    if (!isDestroyed) updateAiStopButtonVisibility()
                }
            }
            override fun onComplete(result: SharpService.GenerationResult) {
                runOnUiThread {
                    if (session != aiSessionId) {
                        deleteSharpRoomFolder(result)
                        DebugLogger.d("SinglePhotoRoom", "Discarded stale AI completion (session mismatch)")
                        return@runOnUiThread
                    }
                    aiGenerationRunning = false
                    aiGenerationResult = result
                    aiGenerationHandle = null
                    if (aiOptionShowPercent) {
                        promoteSharpRoomToLibrary(result)
                    }
                    if (!isDestroyed) {
                        updateAIOptionProgress(1f, "Ready")
                        hideProgressOverlay()
                        if (aiRoomOverlayRequested) {
                            aiRoomOverlayRequested = false
                            openSharpRoomWithResult(result)
                        }
                        updateAiStopButtonVisibility()
                    } else {
                        lastAIGenerationProgress = 1f
                        lastAIGenerationRawMessage = "Ready"
                        lastAIGenerationMessage = toFriendlyMessage(1f, "Ready")
                        updateGlobalAiProgressOverlay()
                    }
                    DebugLogger.d("SinglePhotoRoom", "AI generation completed in background")
                }
            }
            override fun onError(message: String) {
                runOnUiThread {
                    if (session != aiSessionId) return@runOnUiThread
                    if (message == "SHARP_CANCELLED") {
                        aiGenerationRunning = false
                        aiGenerationResult = null
                        aiGenerationHandle = null
                        aiRoomOverlayRequested = false
                        aiOptionShowPercent = false
                        lastAIGenerationRawMessage = ""
                        lastAIGenerationProgress = 0f
                        lastAIGenerationMessage = "Getting started…"
                        if (!isDestroyed) {
                            hideProgressOverlay()
                            aiOptionSubtitleView?.text = getString(R.string.single_photo_ai_room_subtitle_idle)
                            updateAiStopButtonVisibility()
                        }
                        updateGlobalAiProgressOverlay()
                        return@runOnUiThread
                    }
                    aiGenerationRunning = false
                    aiGenerationResult = null
                    aiGenerationHandle = null
                    aiRoomOverlayRequested = false
                    aiOptionShowPercent = false
                    if (!isDestroyed) {
                        updateAIOptionProgress(0f, "Failed")
                        hideProgressOverlay()
                        Toast.makeText(this@SinglePhotoRoomActivity, message, Toast.LENGTH_LONG).show()
                        DebugLogger.eDebugMode("SinglePhotoRoom", "AI generation failed: $message")
                        CrashReporter.report(
                            this@SinglePhotoRoomActivity,
                            RuntimeException(message),
                            "Single photo room — AI / SHARP generation",
                        )
                        updateAiStopButtonVisibility()
                    } else {
                        SharpGenerationUiState.clear()
                    }
                }
            }
        },
            viewerPhotoOrientation = orientationForMetadata,
            viewerPhotoWideAngle = photoWideAngle,
            orientationLockedByUser = orientationUserOverridden,
            sourcePhotoUri = selectedImageUri,
            metricAnchors = pendingMetricAnchors,
        )
        updateAiStopButtonVisibility()
        updateGlobalAiProgressOverlay()
    }

    /** Cancel AI generation, delete any room folder on disk, and release model memory. */
    private fun cancelAndReleaseAI() {
        aiGenerationHandle?.cancel()
        aiGenerationHandle = null
        deleteSharpRoomFolder(aiGenerationResult)
        aiGenerationResult = null
        aiGenerationRunning = false
        aiRoomOverlayRequested = false
        aiOptionShowPercent = false
        lastAIGenerationRawMessage = ""
        aiSessionId++
        hideProgressOverlay()
        aiOptionSubtitleView?.text = getString(R.string.single_photo_ai_room_subtitle_idle)
        SharpService.getInstance(this).release()
        DebugLogger.d("SinglePhotoRoom", "AI cancelled and memory released (session=$aiSessionId)")
        updateAiStopButtonVisibility()
        updateGlobalAiProgressOverlay()
    }

    private var aiOptionSubtitleView: TextView? = null
    private var aiStopButtonView: TextView? = null

    /** Last progress from generation callback — used when showing overlay for already-running gen. */
    private var lastAIGenerationProgress: Float = 0f
    private var lastAIGenerationMessage: String = "Getting started…"
    private var lastAIGenerationRawMessage: String = ""

    private fun updateAIOptionProgress(progress: Float, message: String) {
        lastAIGenerationProgress = progress
        lastAIGenerationRawMessage = message
        val friendly = toFriendlyMessage(progress, message)
        lastAIGenerationMessage = friendly
        val idle = getString(R.string.single_photo_ai_room_subtitle_idle)
        // Idle bluff only when nothing is running. While generating (or after Run in background), show % like before.
        if (!isDestroyed) {
            aiOptionSubtitleView?.text = when {
                progress >= 1f -> "Ready — tap to view"
                aiGenerationRunning || aiOptionShowPercent -> "$friendly (${(progress * 100).toInt()}%)"
                else -> idle
            }
        }
        updateGlobalAiProgressOverlay()
    }

    /** Dismiss full-screen progress; keep generation running and show percent on the AI Room row. */
    private fun onRunInBackgroundClicked() {
        aiRoomOverlayRequested = false
        aiOptionShowPercent = true
        hideProgressOverlay()
        updateAIOptionProgress(lastAIGenerationProgress, lastAIGenerationRawMessage)
    }

    private fun onAIRoomSelected() {
        DebugLogger.d("SinglePhotoRoom", "AI Room selected")
        if (selectedBitmap == null) {
            Toast.makeText(this, "No image selected", Toast.LENGTH_SHORT).show()
            return
        }

        // Use result from background generation if already done
        val result = aiGenerationResult
        if (result != null) {
            DebugLogger.d("SinglePhotoRoom", "Using cached AI result")
            openSharpRoomWithResult(result)
            return
        }

        // Generation still running: show overlay with current progress (don't reset to 0%)
        if (aiGenerationRunning) {
            logProgress0("SinglePhotoRoomActivity.kt:onAIRoomSelected", "gen running, show overlay", mapOf(
                "lastProgress" to lastAIGenerationProgress, "lastMessage" to lastAIGenerationMessage
            ))
            aiRoomOverlayRequested = true
            showProgressOverlay(preserveProgress = true)
            return
        }

        // Not running and no result (failed or cancelled): start fresh
        logProgress0("SinglePhotoRoomActivity.kt:onAIRoomSelected", "start fresh", mapOf())
        lastAIGenerationProgress = 0f
        lastAIGenerationMessage = "Getting started…"
        lastAIGenerationRawMessage = ""
        aiOptionShowPercent = false
        startAIGenerationInBackground(selectedBitmap!!)
        aiRoomOverlayRequested = true
        showProgressOverlay(preserveProgress = false)
    }

    private fun openSharpRoomWithResult(result: SharpService.GenerationResult) {
        val intent = Intent(this, SharpRoomActivity::class.java).apply {
            putExtra(SharpRoomActivity.EXTRA_PLY_PATH, result.classicPlyFile.absolutePath)
            putExtra(SharpRoomActivity.EXTRA_ROOM_FOLDER, result.plyFile.parentFile?.absolutePath)
            if (result.roomWidth > 0f) putExtra(SharpRoomActivity.EXTRA_ROOM_WIDTH, result.roomWidth)
            if (result.roomHeight > 0f) putExtra(SharpRoomActivity.EXTRA_ROOM_HEIGHT, result.roomHeight)
            if (result.roomDepth > 0f) putExtra(SharpRoomActivity.EXTRA_ROOM_DEPTH, result.roomDepth)
            result.roomCenterX?.let { putExtra(SharpRoomActivity.EXTRA_ROOM_CENTER_X, it) }
            result.roomCenterY?.let { putExtra(SharpRoomActivity.EXTRA_ROOM_CENTER_Y, it) }
            result.roomCenterZ?.let { putExtra(SharpRoomActivity.EXTRA_ROOM_CENTER_Z, it) }
            putExtra(SharpRoomActivity.EXTRA_ALLOW_SAVE, true)
            putExtra("photo_orientation", metadataOrientationStringForViewer())
            putExtra(SharpRoomActivity.EXTRA_PHOTO_WIDE_ANGLE, photoWideAngle)
            // Preview-only / silent builds: delete on exit if never saved. Already on Home (e.g. Run in background) → not temp.
            val folder = result.plyFile.parentFile
            val snap = folder?.let { RoomFolderMetadata.readFromFolder(it) }
            val isTempPreview = snap?.previewOnly != false
            putExtra(SharpRoomActivity.EXTRA_IS_TEMP_SHARP_ROOM, isTempPreview)
            putExtra(SharpRoomActivity.EXTRA_OPENED_FROM_SINGLE_PHOTO_ROOM, true)
        }
        DebugLogger.i(
            "SHARP_ROOM_MEAS",
            if (result.roomWidth > 0f && result.roomHeight > 0f && result.roomDepth > 0f) {
                "[open_sharp_viewer] W×H×D=${result.roomWidth}×${result.roomHeight}×${result.roomDepth} " +
                    "center=(${result.roomCenterX},${result.roomCenterY},${result.roomCenterZ}) " +
                    "folder=${result.plyFile.parentFile?.absolutePath} classic=${result.classicPlyFile.name}"
            } else {
                "[open_sharp_viewer] dims=deferred_async " +
                    "center=(${result.roomCenterX},${result.roomCenterY},${result.roomCenterZ}) " +
                    "folder=${result.plyFile.parentFile?.absolutePath} classic=${result.classicPlyFile.name}"
            },
        )
        sharpRoomLauncher.launch(intent)
    }

    private fun onManualSetupSelected() {
        DebugLogger.d("SinglePhotoRoom", "Manual Setup selected")
        cancelAndReleaseAI()
        val uri = selectedImageUri
        if (uri == null) {
            Toast.makeText(this, "No image selected", Toast.LENGTH_SHORT).show()
            return
        }

        val intent = Intent(this, RoomBoundaryActivity::class.java).apply {
            putExtra(RoomBoundaryActivity.EXTRA_IMAGE_URI, uri.toString())
            putExtra(RoomBoundaryActivity.EXTRA_PHOTO_ORIENTATION, metadataOrientationStringForViewer())
        }
        boundaryActivityLauncher.launch(intent)
    }

    /** Returns to Create 3D Room photo selection and cancels AI unless already finished (e.g. after Sharp viewer). */
    private fun showInitialView() {
        singleImageScanRequestId++
        resetSingleImageOverlay()
        cancelAndReleaseAI()
        methodPickerView.visibility = View.GONE
        initialView.visibility = View.VISIBLE
        selectedBitmap = null
        selectedImageUri = null
        orientationUserOverridden = false
        photoWideAngle = false
    }

    /**
     * Turns backend progress messages into short, friendly text for the user.
     * No technical terms — keeps people engaged during the ~2 minute wait.
     */
    private fun toFriendlyMessage(progress: Float, message: String): String {
        val m = message.lowercase()
        return when {
            progress >= 1f -> "Your room is ready!"
            m.contains("preprocess") || m.contains("1536") -> "Getting your photo ready…"
            m.contains("loading") && (m.contains("encoder") || m.contains("model")) -> "Warming up…"
            m.contains("part 1") || m.contains("part 2") || m.contains("patch") -> {
                val step = Regex("""(\d+)\s*/\s*35""").find(message)?.groupValues?.get(1)
                if (step != null) "Building your room… step $step of 35" else "Building your room…"
            }
            m.contains("part 3") || m.contains("image encoder") -> "Understanding the full picture…"
            m.contains("part 4a") || m.contains("tokens") -> "Adding depth and shape…"
            m.contains("part 4b") || m.contains("decoder") -> "Adding the finishing touches…"
            m.contains("writing") || m.contains("room file") || m.contains("ply") || m.contains("gaussian") ->
                "Preparing your preview…"
            m.contains("done") -> "Your room is ready!"
            m.contains("error") || m.contains("failed") -> message
            progress < 0.15f -> "Getting started…"
            progress < 0.45f -> "Working on it…"
            progress < 0.75f -> "Almost there…"
            else -> "Finishing up…"
        }
    }

    private fun phasePillDrawable(active: Boolean): GradientDrawable {
        val density = resources.displayMetrics.density
        val strokeW = (1.5f * density).toInt().coerceAtLeast(1)
        return GradientDrawable().apply {
            shape = GradientDrawable.RECTANGLE
            cornerRadius = 999f
            if (active) {
                setColor(Color.parseColor("#6A1B9A"))
            } else {
                setColor(Color.WHITE)
                setStroke(strokeW, Color.parseColor("#E1BEE7"))
            }
        }
    }

    private fun setPhaseStripForPercent(percent: Int) {
        if (phaseStripViews.isEmpty()) return
        val activeIdx = when {
            percent < 26 -> 0
            percent < 88 -> 1
            else -> 2
        }
        phaseStripViews.forEachIndexed { index, textView ->
            val active = index == activeIdx
            textView.background = phasePillDrawable(active)
            textView.setTextColor(if (active) Color.WHITE else Color.parseColor("#6A1B9A"))
        }
    }

    private fun startProgressOverlayPulse() {
        val host = progressRingHost ?: return
        progressOverlayPulse?.cancel()
        val scaleX = ObjectAnimator.ofFloat(host, View.SCALE_X, 1f, 1.045f).apply {
            duration = 1400
            repeatCount = ValueAnimator.INFINITE
            repeatMode = ValueAnimator.REVERSE
            interpolator = AccelerateDecelerateInterpolator()
        }
        val scaleY = ObjectAnimator.ofFloat(host, View.SCALE_Y, 1f, 1.045f).apply {
            duration = 1400
            repeatCount = ValueAnimator.INFINITE
            repeatMode = ValueAnimator.REVERSE
            interpolator = AccelerateDecelerateInterpolator()
        }
        progressOverlayPulse = AnimatorSet().apply {
            playTogether(scaleX, scaleY)
            start()
        }
    }

    private fun stopProgressOverlayPulse() {
        progressOverlayPulse?.cancel()
        progressOverlayPulse = null
        progressRingHost?.apply {
            scaleX = 1f
            scaleY = 1f
        }
    }

    private fun createProgressOverlay(): FrameLayout {
        val density = resources.displayMetrics.density
        val padH = (40 * density).toInt()
        val padVTop = (44 * density).toInt()
        val padVBottom = (48 * density).toInt()
        val ringSize = (196 * density).toInt()

        val screenH = resources.displayMetrics.heightPixels
        val screenW = resources.displayMetrics.widthPixels
        val marginOuter = (20 * density).toInt()
        val maxPanelHeight = (screenH * 0.88f).toInt().coerceAtLeast((280 * density).toInt())

        return FrameLayout(this).apply {
            setBackgroundColor(Color.parseColor("#CC000000"))
            layoutParams = FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT
            )
            clipChildren = false

            val content = LinearLayout(this@SinglePhotoRoomActivity).apply {
                orientation = LinearLayout.VERTICAL
                gravity = Gravity.CENTER_HORIZONTAL
                setPadding(padH, padVTop, padH, padVBottom)
                background = GradientDrawable().apply {
                    shape = GradientDrawable.RECTANGLE
                    cornerRadius = 28f * density
                    setColor(Color.WHITE)
                }
                if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.LOLLIPOP) {
                    elevation = 14f * density
                }

                addView(TextView(this@SinglePhotoRoomActivity).apply {
                    text = "✨  3D room"
                    textSize = 14f
                    setTypeface(null, Typeface.BOLD)
                    setTextColor(Color.parseColor("#7B1FA2"))
                    gravity = Gravity.CENTER
                })

                addView(TextView(this@SinglePhotoRoomActivity).apply {
                    text = "Neural reconstruction"
                    textSize = 11f
                    setTextColor(Color.parseColor("#9E9E9E"))
                    gravity = Gravity.CENTER
                    setPadding(0, (6 * density).toInt(), 0, 0)
                })

                val ringFrame = FrameLayout(this@SinglePhotoRoomActivity).apply {
                    layoutParams = LinearLayout.LayoutParams(ringSize, ringSize).apply {
                        gravity = Gravity.CENTER_HORIZONTAL
                        topMargin = (18 * density).toInt()
                    }
                }
                progressRingHost = ringFrame

                progressRing = CircularProgressIndicator(this@SinglePhotoRoomActivity).apply {
                    max = 100
                    isIndeterminate = false
                    indicatorSize = ringSize
                    trackThickness = (9 * density).toInt()
                    setIndicatorColor(
                        Color.parseColor("#AB47BC"),
                        Color.parseColor("#8E24AA"),
                        Color.parseColor("#6A1B9A"),
                    )
                    setTrackColor(Color.parseColor("#F3E5F5"))
                    layoutParams = FrameLayout.LayoutParams(ringSize, ringSize).apply {
                        gravity = Gravity.CENTER
                    }
                    setProgress(0, false)
                }
                ringFrame.addView(progressRing)

                progressPercent = TextView(this@SinglePhotoRoomActivity).apply {
                    text = "0%"
                    textSize = 36f
                    setTypeface(null, Typeface.BOLD)
                    setTextColor(Color.parseColor("#4A148C"))
                    gravity = Gravity.CENTER
                    layoutParams = FrameLayout.LayoutParams(
                        ViewGroup.LayoutParams.MATCH_PARENT,
                        ViewGroup.LayoutParams.MATCH_PARENT
                    ).apply { gravity = Gravity.CENTER }
                }
                ringFrame.addView(progressPercent)
                addView(ringFrame)

                val phaseStrip = LinearLayout(this@SinglePhotoRoomActivity).apply {
                    orientation = LinearLayout.HORIZONTAL
                    gravity = Gravity.CENTER
                    layoutParams = LinearLayout.LayoutParams(
                        ViewGroup.LayoutParams.MATCH_PARENT,
                        ViewGroup.LayoutParams.WRAP_CONTENT
                    ).apply { topMargin = (22 * density).toInt() }
                }
                val phaseNames = listOf(
                    "Prepare",
                    "SHARP",
                    getString(R.string.single_photo_phase_finalize),
                )
                phaseStripViews = Array(phaseNames.size) { index ->
                    TextView(this@SinglePhotoRoomActivity).apply {
                        text = phaseNames[index]
                        textSize = 11f
                        setTypeface(null, Typeface.BOLD)
                        setPadding(
                            (14 * density).toInt(),
                            (8 * density).toInt(),
                            (14 * density).toInt(),
                            (8 * density).toInt(),
                        )
                        layoutParams = LinearLayout.LayoutParams(
                            ViewGroup.LayoutParams.WRAP_CONTENT,
                            ViewGroup.LayoutParams.WRAP_CONTENT
                        ).apply {
                            if (index < phaseNames.lastIndex) {
                                marginEnd = (8 * density).toInt()
                            }
                        }
                        background = phasePillDrawable(false)
                        setTextColor(Color.parseColor("#6A1B9A"))
                    }.also { phaseStrip.addView(it) }
                }
                setPhaseStripForPercent(0)
                addView(phaseStrip)

                progressText = TextView(this@SinglePhotoRoomActivity).apply {
                    text = "Creating your 3D room…"
                    textSize = 16f
                    setTypeface(null, Typeface.BOLD)
                    setTextColor(Color.parseColor("#424242"))
                    gravity = Gravity.CENTER
                    setPadding(0, (20 * density).toInt(), 0, (10 * density).toInt())
                    layoutParams = LinearLayout.LayoutParams(
                        ViewGroup.LayoutParams.MATCH_PARENT,
                        ViewGroup.LayoutParams.WRAP_CONTENT,
                    )
                }
                addView(progressText)

                runInBackgroundButton = MaterialButton(this@SinglePhotoRoomActivity).apply {
                    text = getString(R.string.single_photo_run_in_background)
                    textSize = 14f
                    setOnClickListener { onRunInBackgroundClicked() }
                    layoutParams = LinearLayout.LayoutParams(
                        ViewGroup.LayoutParams.MATCH_PARENT,
                        ViewGroup.LayoutParams.WRAP_CONTENT,
                    ).apply {
                        topMargin = (16 * density).toInt()
                    }
                }
                addView(runInBackgroundButton)

                addView(TextView(this@SinglePhotoRoomActivity).apply {
                    text = getString(R.string.single_photo_progress_overlay_background_hint)
                    textSize = 12f
                    setTextColor(Color.parseColor("#757575"))
                    gravity = Gravity.CENTER
                    setLineSpacing(3f * density, 1f)
                    layoutParams = LinearLayout.LayoutParams(
                        ViewGroup.LayoutParams.MATCH_PARENT,
                        ViewGroup.LayoutParams.WRAP_CONTENT,
                    ).apply {
                        topMargin = (12 * density).toInt()
                        bottomMargin = (16 * density).toInt()
                    }
                })
            }

            // Card was taller than many phones — bottom text was cut off. Scroll so "background" is always reachable.
            val scrollView = ScrollView(this@SinglePhotoRoomActivity).apply {
                layoutParams = FrameLayout.LayoutParams(
                    screenW - 2 * marginOuter,
                    maxPanelHeight,
                ).apply {
                    gravity = Gravity.CENTER
                    setMargins(marginOuter, (24 * density).toInt(), marginOuter, (24 * density).toInt())
                }
                isFillViewport = false
                isVerticalScrollBarEnabled = true
                scrollBarStyle = View.SCROLLBARS_INSIDE_OVERLAY
            }
            scrollView.addView(
                content,
                FrameLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                ),
            )
            addView(scrollView)
        }
    }

    // #region agent log
    private fun logProgress0(location: String, message: String, data: Map<String, Any?>) {
        val payload = JSONObject().apply {
            put("location", location)
            put("message", message)
            put("timestamp", System.currentTimeMillis())
            data.forEach { (k, v) -> if (v != null) put(k, v) }
        }
        DebugLogger.d("Progress0", payload.toString())
        try {
            val dir = getExternalFilesDir(null) ?: filesDir
            File(dir, "debug_progress.ndjson").appendText(payload.toString() + "\n")
        } catch (_: Throwable) {}
    }
    // #endregion

    private fun showProgressOverlay(preserveProgress: Boolean = false) {
        val displayProgress: Float
        val displayMessage: String
        if (preserveProgress && lastAIGenerationProgress > 0f) {
            displayProgress = lastAIGenerationProgress
            displayMessage = lastAIGenerationMessage
            logProgress0("SinglePhotoRoomActivity.kt:showProgressOverlay", "preserveProgress=true", mapOf(
                "preserveProgress" to true, "lastProgress" to lastAIGenerationProgress, "lastMessage" to lastAIGenerationMessage
            ))
        } else {
            displayProgress = 0f
            displayMessage = "Getting started…"
            logProgress0("SinglePhotoRoomActivity.kt:showProgressOverlay", "reset to 0%", mapOf(
                "preserveProgress" to preserveProgress, "reason" to if (preserveProgress) "lastProgress was 0" else "fresh start"
            ))
        }
        progressOverlay.visibility = View.VISIBLE
        val pct = (displayProgress * 100).toInt().coerceIn(0, 100)
        progressRing.setProgress(pct, false)
        progressPercent.text = "$pct%"
        progressText.text = displayMessage
        setPhaseStripForPercent(pct)
        startProgressOverlayPulse()
        updateGlobalAiProgressOverlay()
    }

    private fun hideProgressOverlay() {
        stopProgressOverlayPulse()
        progressOverlay.visibility = View.GONE
        updateGlobalAiProgressOverlay()
    }

    private fun createGlobalAiProgressBar(): FrameLayout {
        val density = resources.displayMetrics.density
        return FrameLayout(this).apply {
            visibility = View.GONE
            setBackgroundColor(Color.parseColor("#5E35B1"))
            if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.LOLLIPOP) {
                elevation = 28f * density
            }
            val row = LinearLayout(this@SinglePhotoRoomActivity).apply {
                orientation = LinearLayout.HORIZONTAL
                gravity = Gravity.CENTER_VERTICAL
                setPadding(dpToPx(14), dpToPx(10), dpToPx(14), dpToPx(14))
            }
            ViewCompat.setOnApplyWindowInsetsListener(this) { _, insets ->
                val nav = insets.getInsets(WindowInsetsCompat.Type.navigationBars())
                row.setPadding(dpToPx(14), dpToPx(10), dpToPx(14), dpToPx(14) + nav.bottom)
                insets
            }
            globalAiProgressLabel = TextView(this@SinglePhotoRoomActivity).apply {
                layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
                textSize = 13f
                setTextColor(Color.WHITE)
                maxLines = 2
            }
            row.addView(globalAiProgressLabel)
            val stopGlobal = TextView(this@SinglePhotoRoomActivity).apply {
                text = "⏹"
                textSize = 11f
                setTextColor(Color.WHITE)
                gravity = Gravity.CENTER
                setPadding(dpToPx(8), dpToPx(4), dpToPx(8), dpToPx(4))
                background = GradientDrawable().apply {
                    shape = GradientDrawable.RECTANGLE
                    cornerRadius = dpToPx(4).toFloat()
                    setColor(Color.parseColor("#E53935"))
                }
                contentDescription = getString(R.string.single_photo_ai_stop)
                setOnClickListener { onAIStopClicked() }
            }
            row.addView(stopGlobal)
            addView(row)
        }
    }

    /** Bottom strip while a job is active — drawn above the full-screen modal so it stays visible on every screen. */
    private fun updateGlobalAiProgressOverlay() {
        if (::globalAiProgressBar.isInitialized && !isDestroyed) {
            val active = aiGenerationRunning || aiGenerationResult != null
            globalAiProgressBar.visibility = if (active) View.VISIBLE else View.GONE
            if (active) {
                val pct = (lastAIGenerationProgress * 100).toInt().coerceIn(0, 100)
                globalAiProgressLabel.text = if (lastAIGenerationProgress >= 1f && aiGenerationResult != null) {
                    getString(R.string.single_photo_global_ai_ready, pct)
                } else {
                    "${lastAIGenerationMessage} · $pct%"
                }
            }
            (globalAiProgressBar.parent as? ViewGroup)?.bringChildToFront(globalAiProgressBar)
            ViewCompat.requestApplyInsets(globalAiProgressBar)
        }
        syncSharpGenerationUiStateForList()
    }

    /** Shared with [com.furnit.android.ContentActivity] when this activity is in the background or destroyed. */
    private fun syncSharpGenerationUiStateForList() {
        val active = aiGenerationRunning || aiGenerationResult != null
        if (!active) {
            SharpGenerationUiState.clear()
            return
        }
        val pct = (lastAIGenerationProgress * 100).toInt().coerceIn(0, 100)
        val ctx = if (isDestroyed) applicationContext else this
        val line = if (lastAIGenerationProgress >= 1f && aiGenerationResult != null) {
            ctx.getString(R.string.single_photo_global_ai_ready, pct)
        } else {
            "${lastAIGenerationMessage} · $pct%"
        }
        SharpGenerationUiState.update(true, lastAIGenerationProgress, line)
    }

    private fun updateProgressOverlay(progress: Float, message: String) {
        val percent = (progress * 100).toInt().coerceIn(0, 100)
        val friendly = toFriendlyMessage(progress, message)
        logProgress0("SinglePhotoRoomActivity.kt:updateProgressOverlay", "updating UI", mapOf(
            "progress" to progress, "percent" to percent, "message" to message
        ))
        progressRing.setProgress(percent, true)
        progressPercent.text = "$percent%"
        progressText.text = friendly
        setPhaseStripForPercent(percent)
        updateGlobalAiProgressOverlay()
    }
}
