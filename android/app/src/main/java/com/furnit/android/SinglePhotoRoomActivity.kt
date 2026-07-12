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
import com.furnit.android.services.FurnitureFitManager
import com.furnit.android.services.PhotoRoomGenerationService
import com.furnit.android.services.RoomGenerationUiState
import com.furnit.android.theme.PaafektColors
import com.furnit.android.theme.PaafektDrawables
import android.content.res.ColorStateList
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
 * 2. Shows preview with two options: Manual Setup or AI Room
 * 3. Manual Setup: boundary adjustment for room creation
 * 4. AI Room: generic photo-to-room GLB generation
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
    private var selectedBitmap: Bitmap? = null
    private var selectedImageUri: Uri? = null
    private var cameraPhotoUri: Uri? = null
    private var detectedOrientation: PhotoOrientation = PhotoOrientation.PORTRAIT
    /** True after user tapped the orientation row — keeps true landscape for 0.5× shots when needed. */
    private var orientationUserOverridden: Boolean = false
    /** True when the user indicates the photo was taken with the wide-angle (0.5x) lens; fixes camera position in the 3D viewer. */
    private var photoWideAngle: Boolean = false

    /** AI generation started on photo select; cancel and release when user picks Manual/Back/Change. */
    private var aiGenerationHandle: PhotoRoomGenerationService.GenerationHandle? = null
    private var aiGenerationResult: PhotoRoomGenerationService.GenerationResult? = null
    private var aiGenerationRunning = false
    /** Set when user taps AI Room while generation is running - callback will show overlay and open on complete. */
    private var aiRoomOverlayRequested = false
    /** After "Run in background", AI option line shows percent; otherwise only friendly text (no %). */
    private var aiOptionShowPercent = false
    /** Bumped on cancel/restart so stale generation callbacks are ignored and folders deleted. */
    private var aiSessionId: Int = 0
    private var pendingMetricAnchors: ArrayList<MetricAnchor>? = null
    private val furnitureFitManager by lazy { FurnitureFitManager(this) }
    private var furnitureFitInitialized = false
    private var singleImageScanRequestId = 0
    private var imageLoadRequestId = 0
    private val maxRoomPhotoDimensionPx = 2048

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

    private val generatedRoomLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult(),
    ) {
        if (!isDestroyed) {
            showInitialView()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        rootLayout = FrameLayout(this)
        rootLayout.setBackgroundColor(PaafektColors.background)

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

            val backBtn = TextView(this@SinglePhotoRoomActivity).apply {
                text = getString(R.string.photo_room_back)
                textSize = 16f
                setTextColor(PaafektColors.accent)
                setPadding(0, 0, 0, 32)
                setOnClickListener { onBackPressedDispatcher.onBackPressed() }
            }
            addView(backBtn, LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            ))

            val title = TextView(this@SinglePhotoRoomActivity).apply {
                text = getString(R.string.photo_room_create_title)
                textSize = 24f
                setTypeface(null, Typeface.BOLD)
                setTextColor(PaafektColors.textPrimary)
                gravity = Gravity.CENTER
            }
            addView(title)

            val subtitle = TextView(this@SinglePhotoRoomActivity).apply {
                text = getString(R.string.photo_room_capture_subtitle)
                textSize = 16f
                setTextColor(PaafektColors.textSecondary)
                gravity = Gravity.CENTER
                setPadding(0, 16, 0, 32)
            }
            addView(subtitle)

            val takePhotoBtn = LinearLayout(this@SinglePhotoRoomActivity).apply {
                orientation = LinearLayout.VERTICAL
                gravity = Gravity.CENTER
                setPadding(48, 36, 48, 36)
                background = PaafektDrawables.creationCardPrimary()

                val icon = ImageView(this@SinglePhotoRoomActivity).apply {
                    setImageResource(R.drawable.ic_camera)
                    imageTintList = ColorStateList.valueOf(PaafektColors.accent)
                    scaleType = ImageView.ScaleType.CENTER_INSIDE
                }
                addView(icon, LinearLayout.LayoutParams(dpToPx(48), dpToPx(48)))

                val btnText = TextView(this@SinglePhotoRoomActivity).apply {
                    text = getString(R.string.photo_room_take_photo)
                    textSize = 18f
                    setTypeface(null, Typeface.BOLD)
                    setTextColor(PaafektColors.textPrimary)
                    gravity = Gravity.CENTER
                    setPadding(0, 16, 0, 0)
                }
                addView(btnText)

                val btnHint = TextView(this@SinglePhotoRoomActivity).apply {
                    text = getString(R.string.photo_room_use_camera)
                    textSize = 14f
                    setTextColor(PaafektColors.textSecondary)
                    gravity = Gravity.CENTER
                    setPadding(0, 8, 0, 0)
                }
                addView(btnHint)

                setOnClickListener { checkCameraPermissionAndLaunch() }
            }
            addView(takePhotoBtn, LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            ).apply { setMargins(0, 0, 0, 24) })

            val dividerRow = LinearLayout(this@SinglePhotoRoomActivity).apply {
                orientation = LinearLayout.HORIZONTAL
                gravity = Gravity.CENTER_VERTICAL
                setPadding(32, 0, 32, 0)

                val leftLine = View(this@SinglePhotoRoomActivity).apply {
                    setBackgroundColor(PaafektColors.hairline)
                }
                addView(leftLine, LinearLayout.LayoutParams(0, dpToPx(1), 1f))

                val orText = TextView(this@SinglePhotoRoomActivity).apply {
                    text = getString(R.string.photo_room_or)
                    textSize = 14f
                    setTextColor(PaafektColors.textSecondary)
                    setPadding(dpToPx(16), 0, dpToPx(16), 0)
                }
                addView(orText)

                val rightLine = View(this@SinglePhotoRoomActivity).apply {
                    setBackgroundColor(PaafektColors.hairline)
                }
                addView(rightLine, LinearLayout.LayoutParams(0, dpToPx(1), 1f))
            }
            addView(dividerRow, LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            ).apply { setMargins(0, 0, 0, 24) })

            val selectPhotoBtn = LinearLayout(this@SinglePhotoRoomActivity).apply {
                orientation = LinearLayout.VERTICAL
                gravity = Gravity.CENTER
                setPadding(48, 36, 48, 36)
                background = PaafektDrawables.creationCardSecondary()

                val icon = ImageView(this@SinglePhotoRoomActivity).apply {
                    setImageResource(R.drawable.ic_grid_3x3)
                    imageTintList = ColorStateList.valueOf(PaafektColors.textPrimary)
                    scaleType = ImageView.ScaleType.CENTER_INSIDE
                }
                addView(icon, LinearLayout.LayoutParams(dpToPx(48), dpToPx(48)))

                val btnText = TextView(this@SinglePhotoRoomActivity).apply {
                    text = getString(R.string.photo_room_select_photo)
                    textSize = 18f
                    setTypeface(null, Typeface.BOLD)
                    setTextColor(PaafektColors.textPrimary)
                    gravity = Gravity.CENTER
                    setPadding(0, 16, 0, 0)
                }
                addView(btnText)

                val btnHint = TextView(this@SinglePhotoRoomActivity).apply {
                    text = getString(R.string.photo_room_from_library)
                    textSize = 14f
                    setTextColor(PaafektColors.textSecondary)
                    gravity = Gravity.CENTER
                    setPadding(0, 8, 0, 0)
                }
                addView(btnHint)

                setOnClickListener { openImagePicker() }
            }
            addView(selectPhotoBtn, LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            ).apply { setMargins(0, 0, 0, 24) })

            val warning = TextView(this@SinglePhotoRoomActivity).apply {
                text = getString(R.string.photo_room_screenshot_warning)
                textSize = 14f
                setTextColor(PaafektColors.danger)
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
                ViewGroup.LayoutParams.MATCH_PARENT,
            )
            setBackgroundColor(PaafektColors.background)

            val scroll = ScrollView(this@SinglePhotoRoomActivity).apply {
                overScrollMode = View.OVER_SCROLL_NEVER
                isFillViewport = true
            }

            val content = LinearLayout(this@SinglePhotoRoomActivity).apply {
                orientation = LinearLayout.VERTICAL
                setPadding(dpToPx(16), dpToPx(8), dpToPx(16), dpToPx(24))
            }

            val header = FrameLayout(this@SinglePhotoRoomActivity).apply {
                setPadding(0, dpToPx(8), 0, dpToPx(8))
            }
            val backBtn = TextView(this@SinglePhotoRoomActivity).apply {
                text = getString(R.string.photo_room_back)
                textSize = 16f
                setTextColor(PaafektColors.accent)
                setOnClickListener { showMethodPickerBackConfirmation() }
            }
            header.addView(
                backBtn,
                FrameLayout.LayoutParams(
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                    Gravity.START or Gravity.CENTER_VERTICAL,
                ),
            )
            val navTitle = TextView(this@SinglePhotoRoomActivity).apply {
                text = getString(R.string.photo_room_title)
                textSize = 16f
                setTypeface(null, Typeface.BOLD)
                setTextColor(PaafektColors.textPrimary)
                gravity = Gravity.CENTER
            }
            header.addView(
                navTitle,
                FrameLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                    Gravity.CENTER,
                ),
            )
            content.addView(
                header,
                LinearLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                ),
            )

            val previewFrame = FrameLayout(this@SinglePhotoRoomActivity).apply {
                background = GradientDrawable().apply {
                    shape = GradientDrawable.RECTANGLE
                    cornerRadius = dpToPx(12).toFloat()
                    setColor(PaafektColors.surface)
                }
                clipToOutline = true
            }

            selectedImageView = ImageView(this@SinglePhotoRoomActivity).apply {
                scaleType = ImageView.ScaleType.FIT_CENTER
                adjustViewBounds = true
            }
            previewFrame.addView(
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
            previewFrame.addView(
                singleImageOverlayView,
                FrameLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.MATCH_PARENT,
                ),
            )

            singleImageScanStatusView = TextView(this@SinglePhotoRoomActivity).apply {
                text = getString(R.string.single_image_scan_scanning)
                textSize = 14f
                setTextColor(PaafektColors.textPrimary)
                setPadding(dpToPx(14), dpToPx(10), dpToPx(14), dpToPx(10))
                background = PaafektDrawables.hintChip()
                visibility = View.GONE
            }
            previewFrame.addView(
                singleImageScanStatusView,
                FrameLayout.LayoutParams(
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                    Gravity.BOTTOM or Gravity.CENTER_HORIZONTAL,
                ).apply {
                    bottomMargin = dpToPx(12)
                },
            )

            content.addView(
                previewFrame,
                LinearLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    dpToPx(250),
                ).apply {
                    topMargin = dpToPx(8)
                },
            )

            val headingBlock = LinearLayout(this@SinglePhotoRoomActivity).apply {
                orientation = LinearLayout.VERTICAL
                gravity = Gravity.CENTER_HORIZONTAL
                setPadding(0, dpToPx(8), 0, dpToPx(8))
            }
            headingBlock.addView(
                TextView(this@SinglePhotoRoomActivity).apply {
                    text = getString(R.string.photo_room_how_to_create)
                    textSize = 17f
                    setTypeface(null, Typeface.BOLD)
                    setTextColor(PaafektColors.textPrimary)
                    gravity = Gravity.CENTER
                },
            )
            headingBlock.addView(
                TextView(this@SinglePhotoRoomActivity).apply {
                    text = getString(R.string.photo_room_tap_option)
                    textSize = 14f
                    setTextColor(PaafektColors.textSecondary)
                    gravity = Gravity.CENTER
                    setPadding(0, dpToPx(4), 0, 0)
                },
            )
            content.addView(headingBlock)

            val aiOption = createOptionCard(
                iconResId = R.drawable.ic_ai,
                title = getString(R.string.photo_room_title),
                subtitle = getString(R.string.photo_room_ai_powered),
                primary = true,
                onSubtitleCreated = { view -> aiOptionSubtitleView = view },
            ) {
                onAIRoomSelected()
            }
            content.addView(
                aiOption,
                LinearLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                ).apply {
                    topMargin = dpToPx(8)
                },
            )

            val manualOption = createOptionCard(
                iconResId = R.drawable.ic_square_resize,
                title = getString(R.string.photo_room_manual_setup),
                subtitle = getString(R.string.photo_room_manual_setup_desc),
                primary = false,
            ) {
                onManualSetupSelected()
            }
            content.addView(manualOption)

            val changePhotoLink = TextView(this@SinglePhotoRoomActivity).apply {
                text = getString(R.string.photo_room_choose_different)
                textSize = 14f
                setTextColor(PaafektColors.textSecondary)
                gravity = Gravity.CENTER
                setPadding(0, dpToPx(16), 0, 0)
                setOnClickListener { openImagePicker() }
            }
            content.addView(changePhotoLink)

            scroll.addView(
                content,
                FrameLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                ),
            )
            addView(
                scroll,
                LinearLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.MATCH_PARENT,
                ),
            )
        }
    }

    private fun createOptionCard(
        icon: String? = null,
        iconResId: Int? = null,
        title: String,
        subtitle: String,
        primary: Boolean = false,
        onSubtitleCreated: ((TextView) -> Unit)? = null,
        onClick: () -> Unit,
    ): LinearLayout {
        return LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            background = if (primary) PaafektDrawables.creationCardPrimary() else PaafektDrawables.creationCardSecondary()
            setPadding(dpToPx(24), dpToPx(24), dpToPx(24), dpToPx(24))
            gravity = Gravity.CENTER_VERTICAL
            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT,
            ).apply { setMargins(0, 0, 0, dpToPx(16)) }

            val iconView = if (iconResId != null) {
                ImageView(this@SinglePhotoRoomActivity).apply {
                    setImageResource(iconResId)
                    imageTintList = ColorStateList.valueOf(
                        if (primary) PaafektColors.accent else PaafektColors.textPrimary,
                    )
                    scaleType = ImageView.ScaleType.CENTER_INSIDE
                }
            } else {
                TextView(this@SinglePhotoRoomActivity).apply {
                    text = icon.orEmpty()
                    textSize = 30f
                }
            }
            addView(
                iconView,
                LinearLayout.LayoutParams(dpToPx(50), dpToPx(50)).apply {
                    setMargins(0, 0, dpToPx(16), 0)
                },
            )

            val textContainer = LinearLayout(this@SinglePhotoRoomActivity).apply {
                orientation = LinearLayout.VERTICAL
                layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)

                val titleView = TextView(this@SinglePhotoRoomActivity).apply {
                    text = title
                    textSize = 16f
                    setTypeface(null, Typeface.BOLD)
                    setTextColor(PaafektColors.textPrimary)
                }
                addView(titleView)

                val subtitleView = TextView(this@SinglePhotoRoomActivity).apply {
                    text = subtitle
                    textSize = 12f
                    setTextColor(PaafektColors.textSecondary)
                }
                addView(subtitleView)
                onSubtitleCreated?.invoke(subtitleView)
            }
            addView(textContainer)

            val chevron = TextView(this@SinglePhotoRoomActivity).apply {
                text = "\u203A"
                textSize = 22f
                setTextColor(PaafektColors.textSecondary)
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
        val requestId = ++imageLoadRequestId
        cancelAndReleaseAI()
        selectedBitmap = null
        resetSingleImageOverlay()
        selectedImageView.setImageDrawable(null)
        singleImageScanStatusView.text = getString(R.string.ai_progress_getting_photo_ready)
        singleImageScanStatusView.visibility = View.VISIBLE
        showMethodPicker()

        lifecycleScope.launch {
            val bitmap = try {
                withContext(Dispatchers.IO) {
                    PhotoOrientation.loadBitmapApplyingExif(
                        this@SinglePhotoRoomActivity,
                        uri,
                        maxRoomPhotoDimensionPx,
                    )
                }
            } catch (e: Exception) {
                DebugLogger.eDebugMode("SinglePhotoRoom", "Error loading image", e)
                CrashReporter.report(this@SinglePhotoRoomActivity, e, "Single photo room — load image")
                null
            }

            if (requestId != imageLoadRequestId || isDestroyed) {
                bitmap?.recycle()
                return@launch
            }

            if (bitmap == null) {
                DebugLogger.eDebugMode("SinglePhotoRoom", "Failed to decode image")
                Toast.makeText(this@SinglePhotoRoomActivity, "Failed to load image", Toast.LENGTH_SHORT).show()
                CrashReporter.report(
                    this@SinglePhotoRoomActivity,
                    IllegalStateException("Bitmap decode returned null"),
                    "Single photo room — decode image",
                )
                showInitialView()
                return@launch
            }

            selectedBitmap = bitmap
            selectedImageView.setImageBitmap(bitmap)
            singleImageScanStatusView.visibility = View.GONE
            orientationUserOverridden = false
            photoWideAngle = false

            // Must match bitmap pixels used for room generation (see PhotoOrientation.fromBitmapDimensions KDoc).
            detectedOrientation = PhotoOrientation.fromBitmapDimensions(bitmap)
            DebugLogger.d(
                "SinglePhotoRoom",
                "Orientation from sampled bitmap ${bitmap.width}x${bitmap.height}: ${detectedOrientation.value}",
            )
            updateOrientationIndicator()
            DebugLogger.d("SinglePhotoRoom", "Image loaded: ${bitmap.width}x${bitmap.height}, starting background room generation")

            methodPickerView.post {
                if (requestId == imageLoadRequestId && selectedBitmap === bitmap && !isDestroyed) {
                    startAIGenerationInBackground(bitmap)
                }
            }
        }
    }

    /** Matches room metadata (ultra-wide portrait bias unless user locked orientation). */
    private fun metadataOrientationStringForViewer(): String {
        val o = if (orientationUserOverridden) {
            detectedOrientation
        } else {
            PhotoOrientation.coercePortraitForUltraWide(detectedOrientation, photoWideAngle)
        }
        return if (o.isLandscape) "landscape" else "portrait"
    }

    private fun updateOrientationIndicator() {
        DebugLogger.d(
            "SinglePhotoRoom",
            "Detected orientation: ${detectedOrientation.value} (userOverridden=$orientationUserOverridden)",
        )
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
            singleImageScanStatusView.text = getString(R.string.single_image_scan_scanning)
        }
    }

    private fun startSingleImageOverlayScan(bitmap: Bitmap) {
        if (!::singleImageOverlayView.isInitialized || !::singleImageScanStatusView.isInitialized) return
        val requestId = ++singleImageScanRequestId
        singleImageScanStatusView.text = getString(R.string.single_image_scan_scanning)
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
                singleImageScanStatusView.text = getString(R.string.detector_model_unavailable)
                return@launch
            }

            furnitureFitManager.segmentWithDetectionsAsync(bitmap) { result ->
                if (requestId != singleImageScanRequestId || isDestroyed) return@segmentWithDetectionsAsync
                runOnUiThread {
                    if (result == null) {
                        singleImageOverlayView.setMaskAndDetections(null, emptyList(), 640)
                        singleImageScanStatusView.text = getString(R.string.single_image_scan_no_overlay)
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

    private fun deleteGeneratedRoomFolder(result: PhotoRoomGenerationService.GenerationResult?) {
        val parent = result?.roomFolder ?: return
        val disk = runCatching { RoomFolderMetadata.readFromFolder(parent) }.getOrNull()
        if (!result.previewOnly || disk?.previewOnly == false) {
            DebugLogger.d("SinglePhotoRoom", "Skip delete — room already on Home list: ${parent.absolutePath}")
            return
        }
        try {
            if (parent.exists()) parent.deleteRecursively()
            DebugLogger.d("SinglePhotoRoom", "Deleted generated room folder: ${parent.absolutePath}")
        } catch (e: Exception) {
            DebugLogger.eDebugMode("SinglePhotoRoom", "Failed to delete room folder", e)
        }
    }

    /** User tapped Stop — remove folder even if it was promoted to Home ([deleteGeneratedRoomFolder] would skip). */
    private fun deleteGeneratedRoomFolderUnconditional(result: PhotoRoomGenerationService.GenerationResult?) {
        val parent = result?.roomFolder ?: return
        try {
            if (parent.exists()) parent.deleteRecursively()
            DebugLogger.d("SinglePhotoRoom", "Deleted generated room folder (stop): ${parent.absolutePath}")
        } catch (e: Exception) {
            DebugLogger.eDebugMode("SinglePhotoRoom", "Failed to delete room folder (stop)", e)
        }
    }

    private fun updateAiStopButtonVisibility() {
        // Stop controls live on the full-screen progress overlay (iOS parity — not on method picker).
    }

    private fun onAIStopClicked() {
        if (!aiGenerationRunning && aiGenerationResult == null) return
        DebugLogger.d("SinglePhotoRoom", "AI Stop tapped")
        hideProgressOverlay()
        aiRoomOverlayRequested = false
        aiGenerationHandle?.cancel()
        aiGenerationHandle = null
        aiGenerationResult?.let { deleteGeneratedRoomFolderUnconditional(it) }
        aiGenerationResult = null
        aiGenerationRunning = false
        aiOptionShowPercent = false
        lastAIGenerationRawMessage = ""
        lastAIGenerationProgress = 0f
        lastAIGenerationMessage = getString(R.string.ai_progress_getting_started)
        aiSessionId++
        PhotoRoomGenerationService.getInstance(this).cancelGeneration()
        RoomGenerationUiState.clear()
        aiOptionSubtitleView?.text = getString(R.string.photo_room_ai_powered)
        updateAiStopButtonVisibility()
        updateGlobalAiProgressOverlay()
    }

    /** After "Run in background", copy the generated room to Home (not preview-only). */
    private fun promoteGeneratedRoomToLibrary(
        result: PhotoRoomGenerationService.GenerationResult,
    ): PhotoRoomGenerationService.GenerationResult {
        return try {
            val promoted = PhotoRoomGenerationService.getInstance(this).promoteToLibrary(result)
            LogUtil.i("SinglePhotoRoom", "Promoted generated room to Home list: ${promoted.roomFolder.absolutePath}")
            promoted
        } catch (e: Exception) {
            DebugLogger.eDebugMode("SinglePhotoRoom", "promoteGeneratedRoomToLibrary failed", e)
            result
        }
    }

    /** Start AI generation in background when photo is selected. Cancel on Manual/Back/Change. */
    private fun startAIGenerationInBackground(bitmap: Bitmap) {
        cancelAndReleaseAI()
        val session = aiSessionId
        aiGenerationResult = null
        aiGenerationRunning = true
        val generationService = PhotoRoomGenerationService.getInstance(this)
        val orientationForMetadata = metadataOrientationStringForViewer()
        aiGenerationHandle = generationService.startGenerationInBackground(
            bitmap,
            object : PhotoRoomGenerationService.ProgressCallback {
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
            override fun onComplete(result: PhotoRoomGenerationService.GenerationResult) {
                runOnUiThread {
                    if (session != aiSessionId) {
                        deleteGeneratedRoomFolder(result)
                        DebugLogger.d("SinglePhotoRoom", "Discarded stale AI completion (session mismatch)")
                        return@runOnUiThread
                    }
                    var finalResult = result
                    aiGenerationRunning = false
                    aiGenerationHandle = null
                    if (aiOptionShowPercent) {
                        finalResult = promoteGeneratedRoomToLibrary(result)
                    }
                    aiGenerationResult = finalResult
                    if (!isDestroyed) {
                        updateAIOptionProgress(1f, getString(R.string.detector_model_ready))
                        hideProgressOverlay()
                        if (aiRoomOverlayRequested) {
                            aiRoomOverlayRequested = false
                            openGeneratedRoomWithResult(finalResult)
                        }
                        updateAiStopButtonVisibility()
                    } else {
                        lastAIGenerationProgress = 1f
                        lastAIGenerationRawMessage = getString(R.string.detector_model_ready)
                        lastAIGenerationMessage = toFriendlyMessage(1f, getString(R.string.detector_model_ready))
                        updateGlobalAiProgressOverlay()
                    }
                    DebugLogger.d("SinglePhotoRoom", "AI generation completed in background")
                }
            }
            override fun onError(message: String) {
                runOnUiThread {
                    if (session != aiSessionId) return@runOnUiThread
                    if (message == "ROOM_CANCELLED") {
                        aiGenerationRunning = false
                        aiGenerationResult = null
                        aiGenerationHandle = null
                        aiRoomOverlayRequested = false
                        aiOptionShowPercent = false
                        lastAIGenerationRawMessage = ""
                        lastAIGenerationProgress = 0f
                        lastAIGenerationMessage = getString(R.string.ai_progress_getting_started)
                        if (!isDestroyed) {
                            hideProgressOverlay()
                            aiOptionSubtitleView?.text = getString(R.string.photo_room_ai_powered)
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
                            "Single photo room — AI generation",
                        )
                        updateAiStopButtonVisibility()
                    } else {
                        RoomGenerationUiState.clear()
                    }
                }
            }
        },
            viewerPhotoOrientation = orientationForMetadata,
            viewerPhotoWideAngle = photoWideAngle,
            sourcePhotoUri = selectedImageUri,
        )
        updateAiStopButtonVisibility()
        updateGlobalAiProgressOverlay()
    }

    /** Cancel AI generation and delete any preview room folder on disk. */
    private fun cancelAndReleaseAI() {
        aiGenerationHandle?.cancel()
        aiGenerationHandle = null
        deleteGeneratedRoomFolder(aiGenerationResult)
        aiGenerationResult = null
        aiGenerationRunning = false
        aiRoomOverlayRequested = false
        aiOptionShowPercent = false
        lastAIGenerationRawMessage = ""
        aiSessionId++
        hideProgressOverlay()
        aiOptionSubtitleView?.text = getString(R.string.photo_room_ai_powered)
        PhotoRoomGenerationService.getInstance(this).cancelGeneration()
        RoomGenerationUiState.clear()
        DebugLogger.d("SinglePhotoRoom", "AI cancelled (session=$aiSessionId)")
        updateAiStopButtonVisibility()
        updateGlobalAiProgressOverlay()
    }

    private var aiOptionSubtitleView: TextView? = null

    /** Last progress from generation callback — used when showing overlay for already-running gen. */
    private var lastAIGenerationProgress: Float = 0f
    private var lastAIGenerationMessage: String = ""
    private var lastAIGenerationRawMessage: String = ""

    private fun updateAIOptionProgress(progress: Float, message: String) {
        lastAIGenerationProgress = progress
        lastAIGenerationRawMessage = message
        val friendly = toFriendlyMessage(progress, message)
        lastAIGenerationMessage = friendly
        if (!isDestroyed && methodPickerView.visibility != View.VISIBLE) {
            val idle = getString(R.string.photo_room_ai_powered)
            aiOptionSubtitleView?.text = when {
                progress >= 1f -> getString(R.string.ai_progress_ready_tap_to_view)
                aiGenerationRunning || aiOptionShowPercent -> "$friendly (${(progress * 100).toInt()}%)"
                else -> idle
            }
        } else if (!isDestroyed) {
            aiOptionSubtitleView?.text = getString(R.string.photo_room_ai_powered)
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
            openGeneratedRoomWithResult(result)
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
        lastAIGenerationMessage = getString(R.string.ai_progress_getting_started)
        lastAIGenerationRawMessage = ""
        aiOptionShowPercent = false
        startAIGenerationInBackground(selectedBitmap!!)
        aiRoomOverlayRequested = true
        showProgressOverlay(preserveProgress = false)
    }

    private fun openGeneratedRoomWithResult(result: PhotoRoomGenerationService.GenerationResult) {
        val intent = Intent(this, GLBRoomActivity::class.java).apply {
            putExtra(GLBRoomActivity.EXTRA_GLB_PATH, result.glbFile.absolutePath)
            putExtra(GLBRoomActivity.EXTRA_ROOM_NAME, "Your Room")
            putExtra(GLBRoomActivity.EXTRA_ROOM_WIDTH, result.roomWidth)
            putExtra(GLBRoomActivity.EXTRA_ROOM_HEIGHT, result.roomHeight)
            putExtra("ROOM_DEPTH", result.roomDepth)
            putExtra(GLBRoomActivity.EXTRA_ROOM_DEPTH, result.roomDepth)
            putExtra(GLBRoomActivity.EXTRA_IS_PREVIEW, result.previewOnly)
            putExtra(GLBRoomActivity.EXTRA_PHOTO_ORIENTATION, result.photoOrientation)
            putExtra("ROOM_FOLDER", result.roomFolder.absolutePath)
        }
        DebugLogger.i(
            "ROOM_GENERATION",
            "[open_glb_viewer] W×H×D=${result.roomWidth}×${result.roomHeight}×${result.roomDepth} " +
                "folder=${result.roomFolder.absolutePath} glb=${result.glbFile.name} preview=${result.previewOnly}",
        )
        generatedRoomLauncher.launch(intent)
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

    /** Returns to Create 3D Room photo selection and cancels AI unless already finished. */
    private fun showInitialView() {
        imageLoadRequestId++
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
            progress >= 1f -> getString(R.string.room_generation_ready)
            m.contains("preprocess") || m.contains("preparing") || m.contains("image") ->
                getString(R.string.ai_progress_getting_photo_ready)
            m.contains("loading") && (m.contains("encoder") || m.contains("model")) -> getString(R.string.ai_progress_warming_up)
            m.contains("extracting") || m.contains("texture") -> getString(R.string.ai_progress_understanding_picture)
            m.contains("building") || m.contains("3d model") -> getString(R.string.ai_progress_building_room)
            m.contains("depth") || m.contains("shape") -> getString(R.string.ai_progress_adding_depth_shape)
            m.contains("final") -> getString(R.string.ai_progress_finishing_touches)
            m.contains("writing") || m.contains("room file") || m.contains("preview") ->
                getString(R.string.ai_progress_preparing_preview)
            m.contains("done") -> getString(R.string.room_generation_ready)
            m.contains("error") || m.contains("failed") -> message
            progress < 0.15f -> getString(R.string.ai_progress_getting_started)
            progress < 0.45f -> getString(R.string.ai_progress_working_on_it)
            progress < 0.75f -> getString(R.string.ai_progress_almost_there)
            else -> getString(R.string.ai_progress_finishing_up)
        }
    }

    private fun phasePillDrawable(active: Boolean): GradientDrawable {
        val density = resources.displayMetrics.density
        val strokeW = (1.5f * density).toInt().coerceAtLeast(1)
        return GradientDrawable().apply {
            shape = GradientDrawable.RECTANGLE
            cornerRadius = 999f
            if (active) {
                setColor(PaafektColors.accent)
            } else {
                setColor(PaafektColors.surface)
                setStroke(strokeW, PaafektColors.hairline)
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
            textView.setTextColor(if (active) PaafektColors.accentText else PaafektColors.textSecondary)
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
            setBackgroundColor(Color.argb(204, 0x0E, 0x0F, 0x12))
            layoutParams = FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT
            )
            clipChildren = false

            val content = LinearLayout(this@SinglePhotoRoomActivity).apply {
                orientation = LinearLayout.VERTICAL
                gravity = Gravity.CENTER_HORIZONTAL
                setPadding(padH, padVTop, padH, padVBottom)
                background = PaafektDrawables.secondaryButton().apply {
                    cornerRadius = 28f * density
                }
                if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.LOLLIPOP) {
                    elevation = 14f * density
                }

                addView(TextView(this@SinglePhotoRoomActivity).apply {
                    text = getString(R.string.photo_room_ai_generation)
                    textSize = 14f
                    setTypeface(null, Typeface.BOLD)
                    setTextColor(PaafektColors.accent)
                    gravity = Gravity.CENTER
                })

                addView(TextView(this@SinglePhotoRoomActivity).apply {
                    text = getString(R.string.photo_room_building_room)
                    textSize = 17f
                    setTypeface(null, Typeface.BOLD)
                    setTextColor(PaafektColors.textPrimary)
                    gravity = Gravity.CENTER
                    setPadding(0, (6 * density).toInt(), 0, 0)
                })

                addView(TextView(this@SinglePhotoRoomActivity).apply {
                    text = getString(R.string.photo_room_building_room_subtext)
                    textSize = 13f
                    setTextColor(PaafektColors.textSecondary)
                    gravity = Gravity.CENTER
                    setPadding(0, (8 * density).toInt(), 0, 0)
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
                    setIndicatorColor(PaafektColors.accent)
                    setTrackColor(PaafektColors.surfaceHi)
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
                    setTextColor(PaafektColors.textPrimary)
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
                    "Build",
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
                        setTextColor(PaafektColors.textSecondary)
                    }.also { phaseStrip.addView(it) }
                }
                setPhaseStripForPercent(0)
                addView(phaseStrip)

                progressText = TextView(this@SinglePhotoRoomActivity).apply {
                    text = getString(R.string.photo_room_creating)
                    textSize = 16f
                    setTypeface(null, Typeface.BOLD)
                    setTextColor(PaafektColors.textPrimary)
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
                    setTextColor(PaafektColors.accentText)
                    setBackgroundColor(PaafektColors.accent)
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
                    setTextColor(PaafektColors.textSecondary)
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
            displayMessage = getString(R.string.ai_progress_getting_started)
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
            setBackgroundColor(PaafektColors.surfaceHi)
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
                setTextColor(PaafektColors.textPrimary)
                maxLines = 2
            }
            row.addView(globalAiProgressLabel)
            val stopGlobal = TextView(this@SinglePhotoRoomActivity).apply {
                text = getString(R.string.single_photo_ai_stop)
                textSize = 11f
                setTextColor(PaafektColors.accentText)
                gravity = Gravity.CENTER
                setPadding(dpToPx(8), dpToPx(4), dpToPx(8), dpToPx(4))
                background = GradientDrawable().apply {
                    shape = GradientDrawable.RECTANGLE
                    cornerRadius = dpToPx(4).toFloat()
                    setColor(PaafektColors.danger)
                }
                contentDescription = getString(R.string.single_photo_ai_stop)
                setOnClickListener { onAIStopClicked() }
            }
            row.addView(stopGlobal)
            addView(row)
        }
    }

    /** Bottom strip while a job is active — hidden on method picker to match iOS. */
    private fun updateGlobalAiProgressOverlay() {
        if (::globalAiProgressBar.isInitialized && !isDestroyed) {
            val onMethodPicker = ::methodPickerView.isInitialized && methodPickerView.visibility == View.VISIBLE
            val active = (aiGenerationRunning || aiGenerationResult != null) && !onMethodPicker
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
        syncRoomGenerationUiStateForList()
    }

    /** Shared with [com.furnit.android.ContentActivity] when this activity is in the background or destroyed. */
    private fun syncRoomGenerationUiStateForList() {
        val active = aiGenerationRunning || aiGenerationResult != null
        if (!active) {
            RoomGenerationUiState.clear()
            return
        }
        val pct = (lastAIGenerationProgress * 100).toInt().coerceIn(0, 100)
        val ctx = if (isDestroyed) applicationContext else this
        val line = if (lastAIGenerationProgress >= 1f && aiGenerationResult != null) {
            ctx.getString(R.string.single_photo_global_ai_ready, pct)
        } else {
            "${lastAIGenerationMessage} · $pct%"
        }
        RoomGenerationUiState.update(true, lastAIGenerationProgress, line)
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
