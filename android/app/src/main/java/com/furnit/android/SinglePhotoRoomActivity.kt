package com.furnit.android

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.net.Uri
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import com.furnit.android.utils.CrashReporter
import com.furnit.android.utils.DebugLogger
import com.furnit.android.utils.FurnitureClassNames
import com.furnit.android.utils.LogUtil
import com.furnit.android.utils.WindowInsetsUtil
import com.furnit.android.utils.RoomFolderMetadata
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.widget.*
import androidx.activity.OnBackPressedCallback
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import androidx.core.content.FileProvider
import com.furnit.android.models.PhotoOrientation
import com.furnit.android.services.FurnitureFitManager
import com.furnit.android.services.PhotoRoomGenerationService
import com.furnit.android.theme.PaafektBuildingRoomOverlay
import com.furnit.android.theme.PaafektColors
import com.furnit.android.theme.PaafektDrawables
import com.furnit.android.theme.PaafektScreenViews
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

    private enum class CameraCaptureMode {
        STANDARD,
        WIDE_ANGLE,
    }

    private lateinit var rootLayout: FrameLayout
    private lateinit var initialView: LinearLayout
    private lateinit var cameraModeView: LinearLayout
    private lateinit var methodPickerView: LinearLayout
    private var buildingRoomOverlay: PaafektBuildingRoomOverlay? = null
    private lateinit var selectedImageView: ImageView
    private lateinit var singleImageOverlayView: FurnitureFitOverlayView
    private lateinit var singleImageScanStatusView: TextView
    private var selectedBitmap: Bitmap? = null
    private var selectedImageUri: Uri? = null
    private var cameraPhotoUri: Uri? = null
    private var detectedOrientation: PhotoOrientation = PhotoOrientation.PORTRAIT
    /** True when the selected/captured source uses the wide-angle camera. */
    private var photoWideAngle: Boolean = false
    private var selectedCameraMode: CameraCaptureMode = CameraCaptureMode.STANDARD

    /** AI generation started on photo select; cancel and release when user picks Manual/Back/Change. */
    private var aiGenerationHandle: PhotoRoomGenerationService.GenerationHandle? = null
    private var aiGenerationResult: PhotoRoomGenerationService.GenerationResult? = null
    private var aiGenerationRunning = false
    /** Bumped on cancel/restart so stale generation callbacks are ignored and folders deleted. */
    private var aiSessionId: Int = 0
    private val furnitureFitManager by lazy { FurnitureFitManager(this) }
    private var furnitureFitInitialized = false
    private var furnitureFitPreloadJob: Job? = null
    private var singleImageScanRequestId = 0
    private var imageLoadRequestId = 0
    private val maxRoomPhotoDimensionPx = 2048

    private val imagePickerLauncher = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        if (uri != null) {
            DebugLogger.d("SinglePhotoRoom", "Image selected: $uri")
            selectedImageUri = uri
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
            photoWideAngle = false
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
            Toast.makeText(this, getString(R.string.camera_permission_required), Toast.LENGTH_SHORT).show()
        }
    }

    private val wideAngleCaptureLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult(),
    ) { result ->
        if (result.resultCode != RESULT_OK) {
            DebugLogger.d("SinglePhotoRoom", "Wide-angle capture cancelled")
            return@registerForActivityResult
        }
        val imageUriString = result.data?.getStringExtra(WideAnglePhotoCaptureActivity.EXTRA_CAPTURED_IMAGE_URI)
        if (imageUriString.isNullOrBlank()) {
            DebugLogger.d("SinglePhotoRoom", "Wide-angle capture missing image uri")
            Toast.makeText(this, getString(R.string.camera_ar_capture_failed), Toast.LENGTH_SHORT).show()
            return@registerForActivityResult
        }
        photoWideAngle = true
        selectedImageUri = Uri.parse(imageUriString)
        DebugLogger.d("SinglePhotoRoom", "Wide-angle photo captured: $selectedImageUri")
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
        // Edge-to-edge (targetSdk 35+) draws behind the system bars; add the real
        // status/navigation bar insets so each mode's top content clears them.
        WindowInsetsUtil.applySystemBarInsetsAsPadding(rootLayout)

        // Initial view - photo selection
        initialView = createInitialView()
        rootLayout.addView(initialView)

        // Camera mode chooser (Standard / Wide Angle) — iOS CameraCaptureView parity
        cameraModeView = createCameraModeView()
        cameraModeView.visibility = View.GONE
        rootLayout.addView(cameraModeView)

        // Method picker view - hidden initially
        methodPickerView = createMethodPickerView()
        methodPickerView.visibility = View.GONE
        rootLayout.addView(methodPickerView)

        setContentView(rootLayout)

        onBackPressedDispatcher.addCallback(
            this,
            object : OnBackPressedCallback(true) {
                override fun handleOnBackPressed() {
                    if (buildingRoomOverlay != null) {
                        return
                    }
                    if (methodPickerView.visibility == View.VISIBLE) {
                        showMethodPickerBackConfirmation()
                        return
                    }
                    if (cameraModeView.visibility == View.VISIBLE) {
                        showInitialView()
                        return
                    }
                    finish()
                }
            },
        )
    }

    override fun onDestroy() {
        if (furnitureFitInitialized) {
            furnitureFitManager.close()
        }
        super.onDestroy()
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
            setPadding(dpToPx(24), dpToPx(24), dpToPx(24), dpToPx(32))

            val backBtn = TextView(this@SinglePhotoRoomActivity).apply {
                text = PaafektScreenViews.backLabel(this@SinglePhotoRoomActivity)
                textSize = 16f
                setTextColor(PaafektColors.accent)
                setPadding(0, 0, 0, dpToPx(16))
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
                setPadding(0, dpToPx(20), 0, 0)
            }
            addView(title)

            val subtitle = TextView(this@SinglePhotoRoomActivity).apply {
                text = getString(R.string.photo_room_capture_subtitle)
                textSize = 16f
                setTextColor(PaafektColors.textSecondary)
                gravity = Gravity.CENTER
                setPadding(0, dpToPx(12), 0, dpToPx(28))
            }
            addView(subtitle)

            val takePhotoBtn = LinearLayout(this@SinglePhotoRoomActivity).apply {
                orientation = LinearLayout.VERTICAL
                gravity = Gravity.CENTER
                setPadding(dpToPx(32), dpToPx(28), dpToPx(32), dpToPx(28))
                background = PaafektDrawables.creationCardPrimary()

                val icon = ImageView(this@SinglePhotoRoomActivity).apply {
                    setImageResource(R.drawable.ic_camera)
                    imageTintList = ColorStateList.valueOf(PaafektColors.accent)
                    scaleType = ImageView.ScaleType.CENTER_INSIDE
                }
                addView(icon, LinearLayout.LayoutParams(dpToPx(56), dpToPx(56)))

                val btnText = TextView(this@SinglePhotoRoomActivity).apply {
                    text = getString(R.string.photo_room_take_photo)
                    textSize = 18f
                    setTypeface(null, Typeface.BOLD)
                    setTextColor(PaafektColors.textPrimary)
                    gravity = Gravity.CENTER
                    setPadding(0, dpToPx(16), 0, 0)
                }
                addView(btnText)

                val btnHint = TextView(this@SinglePhotoRoomActivity).apply {
                    text = getString(R.string.photo_room_use_camera)
                    textSize = 14f
                    setTextColor(PaafektColors.textSecondary)
                    gravity = Gravity.CENTER
                    setPadding(0, dpToPx(8), 0, 0)
                }
                addView(btnHint)

                setOnClickListener { showCameraModeView() }
            }
            addView(takePhotoBtn, LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            ).apply { setMargins(0, 0, 0, dpToPx(20)) })

            val dividerRow = LinearLayout(this@SinglePhotoRoomActivity).apply {
                orientation = LinearLayout.HORIZONTAL
                gravity = Gravity.CENTER_VERTICAL
                setPadding(dpToPx(32), 0, dpToPx(32), 0)

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
            ).apply { setMargins(0, 0, 0, dpToPx(20)) })

            val selectPhotoBtn = LinearLayout(this@SinglePhotoRoomActivity).apply {
                orientation = LinearLayout.VERTICAL
                gravity = Gravity.CENTER
                setPadding(dpToPx(32), dpToPx(28), dpToPx(32), dpToPx(28))
                background = PaafektDrawables.creationCardSecondary()

                val icon = ImageView(this@SinglePhotoRoomActivity).apply {
                    setImageResource(R.drawable.ic_grid_3x3)
                    imageTintList = ColorStateList.valueOf(PaafektColors.textPrimary)
                    scaleType = ImageView.ScaleType.CENTER_INSIDE
                }
                addView(icon, LinearLayout.LayoutParams(dpToPx(56), dpToPx(56)))

                val btnText = TextView(this@SinglePhotoRoomActivity).apply {
                    text = getString(R.string.photo_room_select_photo)
                    textSize = 18f
                    setTypeface(null, Typeface.BOLD)
                    setTextColor(PaafektColors.textPrimary)
                    gravity = Gravity.CENTER
                    setPadding(0, dpToPx(16), 0, 0)
                }
                addView(btnText)

                val btnHint = TextView(this@SinglePhotoRoomActivity).apply {
                    text = getString(R.string.photo_room_from_library)
                    textSize = 14f
                    setTextColor(PaafektColors.textSecondary)
                    gravity = Gravity.CENTER
                    setPadding(0, dpToPx(8), 0, 0)
                }
                addView(btnHint)

                setOnClickListener { openImagePicker() }
            }
            addView(selectPhotoBtn, LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            ).apply { setMargins(0, 0, 0, dpToPx(20)) })

            val warning = LinearLayout(this@SinglePhotoRoomActivity).apply {
                orientation = LinearLayout.HORIZONTAL
                gravity = Gravity.CENTER_VERTICAL
                setPadding(dpToPx(16), dpToPx(14), dpToPx(16), dpToPx(14))
                background = GradientDrawable().apply {
                    shape = GradientDrawable.RECTANGLE
                    cornerRadius = dpToPx(12).toFloat()
                    setColor(Color.argb(34, 0xC8, 0x5A, 0x54))
                    setStroke(dpToPx(1), Color.argb(125, 0xC8, 0x5A, 0x54))
                }

                val warningIcon = TextView(this@SinglePhotoRoomActivity).apply {
                    text = "!"
                    textSize = 18f
                    setTypeface(null, Typeface.BOLD)
                    setTextColor(PaafektColors.danger)
                    gravity = Gravity.CENTER
                    background = GradientDrawable().apply {
                        shape = GradientDrawable.OVAL
                        setColor(Color.argb(36, 0xC8, 0x5A, 0x54))
                        setStroke(dpToPx(2), PaafektColors.danger)
                    }
                }
                addView(
                    warningIcon,
                    LinearLayout.LayoutParams(dpToPx(30), dpToPx(30)).apply {
                        marginEnd = dpToPx(12)
                    },
                )

                val warningText = TextView(this@SinglePhotoRoomActivity).apply {
                    text = getString(R.string.photo_room_screenshot_warning)
                    textSize = 16f
                    setTypeface(null, Typeface.BOLD)
                    setTextColor(PaafektColors.danger)
                    includeFontPadding = false
                    isSingleLine = false
                }
                addView(
                    warningText,
                    LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f),
                )
            }
            addView(
                warning,
                LinearLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                ).apply {
                    topMargin = dpToPx(4)
                },
            )
        }
    }

    private fun createCameraModeView(): LinearLayout {
        return LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            layoutParams = FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT,
            )
            setBackgroundColor(PaafektColors.background)
            setPadding(dpToPx(24), dpToPx(24), dpToPx(24), dpToPx(32))

            val backBtn = TextView(this@SinglePhotoRoomActivity).apply {
                text = PaafektScreenViews.backLabel(this@SinglePhotoRoomActivity)
                textSize = 16f
                setTextColor(PaafektColors.accent)
                setPadding(0, 0, 0, dpToPx(16))
                setOnClickListener { showInitialView() }
            }
            addView(
                backBtn,
                LinearLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                ),
            )

            val headerIcon = ImageView(this@SinglePhotoRoomActivity).apply {
                setImageResource(R.drawable.ic_camera)
                imageTintList = ColorStateList.valueOf(PaafektColors.accent)
                scaleType = ImageView.ScaleType.CENTER_INSIDE
            }
            addView(
                headerIcon,
                LinearLayout.LayoutParams(dpToPx(48), dpToPx(48)).apply {
                    gravity = Gravity.CENTER_HORIZONTAL
                    bottomMargin = dpToPx(12)
                },
            )

            val title = TextView(this@SinglePhotoRoomActivity).apply {
                text = getString(R.string.camera_choose_mode)
                textSize = 22f
                setTypeface(null, Typeface.BOLD)
                setTextColor(PaafektColors.textPrimary)
                gravity = Gravity.CENTER
            }
            addView(title)

            val hint = TextView(this@SinglePhotoRoomActivity).apply {
                text = getString(R.string.camera_choose_mode_hint)
                textSize = 15f
                setTextColor(PaafektColors.textSecondary)
                gravity = Gravity.CENTER
                setPadding(0, dpToPx(8), 0, dpToPx(24))
            }
            addView(hint)

            lateinit var standardRow: LinearLayout
            lateinit var wideRow: LinearLayout
            lateinit var wideInfoBanner: LinearLayout
            lateinit var primaryAction: TextView
            lateinit var secondaryAction: TextView

            fun refreshModeUi() {
                styleCameraModeRow(standardRow, selectedCameraMode == CameraCaptureMode.STANDARD)
                styleCameraModeRow(wideRow, selectedCameraMode == CameraCaptureMode.WIDE_ANGLE)
                wideInfoBanner.visibility =
                    if (selectedCameraMode == CameraCaptureMode.WIDE_ANGLE) View.VISIBLE else View.GONE
                if (selectedCameraMode == CameraCaptureMode.WIDE_ANGLE) {
                    primaryAction.text = getString(R.string.camera_capture_wide_angle)
                    primaryAction.background = GradientDrawable().apply {
                        shape = GradientDrawable.RECTANGLE
                        cornerRadius = dpToPx(12).toFloat()
                        setColor(Color.parseColor("#E67E22"))
                    }
                    secondaryAction.visibility = View.VISIBLE
                } else {
                    primaryAction.text = getString(R.string.camera_take_photo)
                    primaryAction.background = PaafektDrawables.primaryButton()
                    secondaryAction.visibility = View.GONE
                }
            }

            standardRow = createCameraModeOptionRow(
                titleRes = R.string.camera_standard,
                descRes = R.string.camera_standard_desc,
            ) {
                selectedCameraMode = CameraCaptureMode.STANDARD
                refreshModeUi()
            }
            addView(
                standardRow,
                LinearLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                ).apply { bottomMargin = dpToPx(12) },
            )

            wideRow = createCameraModeOptionRow(
                titleRes = R.string.camera_wide_angle,
                descRes = R.string.camera_wide_angle_desc,
            ) {
                selectedCameraMode = CameraCaptureMode.WIDE_ANGLE
                refreshModeUi()
            }
            addView(
                wideRow,
                LinearLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                ).apply { bottomMargin = dpToPx(12) },
            )

            wideInfoBanner = LinearLayout(this@SinglePhotoRoomActivity).apply {
                orientation = LinearLayout.HORIZONTAL
                gravity = Gravity.CENTER_VERTICAL
                setPadding(dpToPx(14), dpToPx(12), dpToPx(14), dpToPx(12))
                background = GradientDrawable().apply {
                    shape = GradientDrawable.RECTANGLE
                    cornerRadius = dpToPx(10).toFloat()
                    setColor(Color.parseColor("#26E67E22"))
                }
                visibility = View.GONE
                val infoIcon = TextView(this@SinglePhotoRoomActivity).apply {
                    text = "ⓘ"
                    textSize = 16f
                    setTextColor(Color.parseColor("#E67E22"))
                    setPadding(0, 0, dpToPx(10), 0)
                }
                addView(infoIcon)
                val infoText = TextView(this@SinglePhotoRoomActivity).apply {
                    text = getString(R.string.camera_wide_angle_info)
                    textSize = 13f
                    setTextColor(PaafektColors.textSecondary)
                }
                addView(
                    infoText,
                    LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f),
                )
            }
            addView(
                wideInfoBanner,
                LinearLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                ).apply { bottomMargin = dpToPx(16) },
            )

            addView(
                View(this@SinglePhotoRoomActivity),
                LinearLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    0,
                    1f,
                ),
            )

            primaryAction = TextView(this@SinglePhotoRoomActivity).apply {
                text = getString(R.string.camera_take_photo)
                textSize = 17f
                setTypeface(null, Typeface.BOLD)
                setTextColor(PaafektColors.accentText)
                gravity = Gravity.CENTER
                setPadding(0, dpToPx(16), 0, dpToPx(16))
                background = PaafektDrawables.primaryButton()
                setOnClickListener {
                    when (selectedCameraMode) {
                        CameraCaptureMode.STANDARD -> {
                            photoWideAngle = false
                            checkCameraPermissionAndLaunch()
                        }
                        CameraCaptureMode.WIDE_ANGLE -> launchWideAngleCapture()
                    }
                }
            }
            addView(
                primaryAction,
                LinearLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                ).apply { bottomMargin = dpToPx(12) },
            )

            secondaryAction = TextView(this@SinglePhotoRoomActivity).apply {
                text = getString(R.string.camera_select_wide_angle)
                textSize = 17f
                setTypeface(null, Typeface.BOLD)
                setTextColor(Color.parseColor("#E67E22"))
                gravity = Gravity.CENTER
                setPadding(0, dpToPx(16), 0, dpToPx(16))
                background = GradientDrawable().apply {
                    shape = GradientDrawable.RECTANGLE
                    cornerRadius = dpToPx(12).toFloat()
                    setColor(Color.parseColor("#26E67E22"))
                }
                visibility = View.GONE
                setOnClickListener { openWideAngleImagePicker() }
            }
            addView(
                secondaryAction,
                LinearLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                ),
            )

            refreshModeUi()
        }
    }

    private fun createCameraModeOptionRow(
        titleRes: Int,
        descRes: Int,
        onClick: () -> Unit,
    ): LinearLayout {
        return LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dpToPx(16), dpToPx(14), dpToPx(16), dpToPx(14))
            val titleView = TextView(this@SinglePhotoRoomActivity).apply {
                text = getString(titleRes)
                textSize = 17f
                setTypeface(null, Typeface.BOLD)
                setTextColor(PaafektColors.textPrimary)
                tag = "title"
            }
            addView(titleView)
            val descView = TextView(this@SinglePhotoRoomActivity).apply {
                text = getString(descRes)
                textSize = 13f
                setTextColor(PaafektColors.textSecondary)
                setPadding(0, dpToPx(4), 0, 0)
                tag = "desc"
            }
            addView(descView)
            setOnClickListener { onClick() }
        }
    }

    private fun styleCameraModeRow(row: LinearLayout, selected: Boolean) {
        row.background = GradientDrawable().apply {
            shape = GradientDrawable.RECTANGLE
            cornerRadius = dpToPx(12).toFloat()
            setColor(if (selected) PaafektColors.surfaceHi else PaafektColors.surface)
            setStroke(
                if (selected) dpToPx(2) else 1,
                if (selected) PaafektColors.accent else PaafektColors.hairline,
            )
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
                text = PaafektScreenViews.backLabel(this@SinglePhotoRoomActivity)
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
                text = PaafektScreenViews.forwardChevron(this@SinglePhotoRoomActivity)
                textSize = 22f
                setTextColor(PaafektColors.textSecondary)
            }
            addView(chevron)

            setOnClickListener { onClick() }
        }
    }

    private fun openImagePicker() {
        DebugLogger.d("SinglePhotoRoom", "Opening image picker")
        photoWideAngle = false
        imagePickerLauncher.launch("image/*")
    }

    private fun openWideAngleImagePicker() {
        DebugLogger.d("SinglePhotoRoom", "Opening wide-angle image picker")
        photoWideAngle = true
        imagePickerLauncher.launch("image/*")
    }

    private fun showCameraModeView() {
        selectedCameraMode = CameraCaptureMode.STANDARD
        photoWideAngle = false
        rootLayout.removeView(cameraModeView)
        cameraModeView = createCameraModeView()
        rootLayout.addView(cameraModeView)
        initialView.visibility = View.GONE
        methodPickerView.visibility = View.GONE
        cameraModeView.visibility = View.VISIBLE
    }

    private fun launchWideAngleCapture() {
        photoWideAngle = true
        DebugLogger.d("SinglePhotoRoom", "Launching wide-angle capture")
        wideAngleCaptureLauncher.launch(Intent(this, WideAnglePhotoCaptureActivity::class.java))
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
        photoWideAngle = false
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
            Toast.makeText(this, R.string.camera_error_opening_generic, Toast.LENGTH_SHORT).show()
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
                Toast.makeText(
                    this@SinglePhotoRoomActivity,
                    getString(R.string.photo_room_failed_load_image),
                    Toast.LENGTH_SHORT,
                ).show()
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
            // Keep photoWideAngle from the capture/picker path that selected this image.

            // Must match bitmap pixels used for room generation (see PhotoOrientation.fromBitmapDimensions KDoc).
            detectedOrientation = PhotoOrientation.fromBitmapDimensions(bitmap)
            DebugLogger.d(
                "SinglePhotoRoom",
                "Orientation from sampled bitmap ${bitmap.width}x${bitmap.height}: ${detectedOrientation.value}",
            )
            updateOrientationIndicator()
            DebugLogger.d("SinglePhotoRoom", "Image loaded: ${bitmap.width}x${bitmap.height}")
        }
    }

    /** Viewer orientation must follow the EXIF-normalized pixels exactly. */
    private fun metadataOrientationStringForViewer(): String {
        return if (detectedOrientation.isLandscape) "landscape" else "portrait"
    }

    private fun updateOrientationIndicator() {
        DebugLogger.d(
            "SinglePhotoRoom",
            "Detected orientation: ${detectedOrientation.value}",
        )
    }

    private fun showMethodPicker() {
        initialView.visibility = View.GONE
        cameraModeView.visibility = View.GONE
        methodPickerView.visibility = View.VISIBLE
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
            furnitureFitPreloadJob?.join()
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
                        primaryDetection = result.primaryDetection,
                    )
                    singleImageScanStatusView.text = if (result.detections.isEmpty()) {
                        getString(R.string.single_image_scan_no_objects)
                    } else {
                        val detection = result.primaryDetection ?: result.detections.first()
                        getString(
                            R.string.single_image_scan_overlay_ready,
                            FurnitureClassNames.localized(
                                this@SinglePhotoRoomActivity,
                                detection.classId,
                                detection.label,
                            ),
                        )
                    }
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
        // iOS parity — no stop / run-in-background controls on generation overlay.
    }

    /** Match Swift's photo-room flow: request RTMDet when AI generation begins, not at app launch. */
    private fun preloadRTMDetForPhotoRoomFlow() {
        if (furnitureFitInitialized || furnitureFitPreloadJob?.isActive == true) return
        furnitureFitPreloadJob = lifecycleScope.launch {
            val initialized = withContext(Dispatchers.IO) {
                furnitureFitManager.initializeAuto()
            }
            furnitureFitInitialized = initialized
            LogUtil.d(
                "SinglePhotoRoom",
                "RTMDet photo-room preload completed, initialized=$initialized",
            )
        }
    }

    /** Start AI generation when user taps AI Room (Swift parity — not on photo select). */
    private fun startAIGenerationInBackground(bitmap: Bitmap) {
        aiGenerationHandle?.cancel()
        aiGenerationHandle = null
        deleteGeneratedRoomFolder(aiGenerationResult)
        aiGenerationResult = null
        PhotoRoomGenerationService.getInstance(this).cancelGeneration()
        aiSessionId++
        val session = aiSessionId
        aiGenerationRunning = true
        preloadRTMDetForPhotoRoomFlow()
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
                    ))
                    if (aiGenerationRunning && !isDestroyed) {
                        updateAIOptionProgress(progress, message)
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
                    aiGenerationResult = finalResult
                    if (!isDestroyed) {
                        updateAIOptionProgress(1f, getString(R.string.detector_model_ready))
                        hideProgressOverlay()
                        openGeneratedRoomWithResult(finalResult)
                        updateAiStopButtonVisibility()
                    }
                    DebugLogger.d("SinglePhotoRoom", "AI generation completed")
                }
            }
            override fun onError(message: String) {
                runOnUiThread {
                    if (session != aiSessionId) return@runOnUiThread
                    if (message == "ROOM_CANCELLED") {
                        aiGenerationRunning = false
                        aiGenerationResult = null
                        aiGenerationHandle = null
                        lastAIGenerationRawMessage = ""
                        lastAIGenerationProgress = 0f
                        lastAIGenerationMessage = getString(R.string.ai_progress_getting_started)
                        if (!isDestroyed) {
                            hideProgressOverlay()
                            aiOptionSubtitleView?.text = getString(R.string.photo_room_ai_powered)
                            updateAiStopButtonVisibility()
                        }
                        return@runOnUiThread
                    }
                    aiGenerationRunning = false
                    aiGenerationResult = null
                    aiGenerationHandle = null
                    if (!isDestroyed) {
                        updateAIOptionProgress(0f, getString(R.string.boundary_failed_create))
                        hideProgressOverlay()
                        Toast.makeText(this@SinglePhotoRoomActivity, message, Toast.LENGTH_LONG).show()
                        DebugLogger.eDebugMode("SinglePhotoRoom", "AI generation failed: $message")
                        CrashReporter.report(
                            this@SinglePhotoRoomActivity,
                            RuntimeException(message),
                            "Single photo room — AI generation",
                        )
                        updateAiStopButtonVisibility()
                    }
                }
            }
        },
            viewerPhotoOrientation = orientationForMetadata,
            viewerPhotoWideAngle = photoWideAngle,
            sourcePhotoUri = selectedImageUri,
        )
        updateAiStopButtonVisibility()
    }

    /** Cancel AI generation and delete any preview room folder on disk. */
    private fun cancelAndReleaseAI() {
        aiGenerationHandle?.cancel()
        aiGenerationHandle = null
        deleteGeneratedRoomFolder(aiGenerationResult)
        aiGenerationResult = null
        aiGenerationRunning = false
        lastAIGenerationRawMessage = ""
        aiSessionId++
        hideProgressOverlay()
        aiOptionSubtitleView?.text = getString(R.string.photo_room_ai_powered)
        PhotoRoomGenerationService.getInstance(this).cancelGeneration()
        DebugLogger.d("SinglePhotoRoom", "AI cancelled (session=$aiSessionId)")
        updateAiStopButtonVisibility()
    }

    private var aiOptionSubtitleView: TextView? = null

    /** Last progress from generation callback — used when showing overlay for already-running gen. */
    private var lastAIGenerationProgress: Float = 0f
    private var lastAIGenerationMessage: String = ""
    private var lastAIGenerationRawMessage: String = ""

    private fun updateAIOptionProgress(progress: Float, message: String) {
        lastAIGenerationProgress = progress
        lastAIGenerationRawMessage = message
        lastAIGenerationMessage = toFriendlyMessage(progress, message)
        buildingRoomOverlay?.setProgress(progress, lastAIGenerationMessage)
        if (!isDestroyed && methodPickerView.visibility == View.VISIBLE) {
            aiOptionSubtitleView?.text = getString(R.string.photo_room_ai_powered)
        }
    }

    private fun onAIRoomSelected() {
        DebugLogger.d("SinglePhotoRoom", "AI Room selected")
        if (selectedBitmap == null) {
            Toast.makeText(this, getString(R.string.photo_room_no_image_selected), Toast.LENGTH_SHORT).show()
            return
        }

        val result = aiGenerationResult
        if (result != null) {
            DebugLogger.d("SinglePhotoRoom", "Using cached AI result")
            openGeneratedRoomWithResult(result)
            return
        }

        if (aiGenerationRunning) {
            showProgressOverlay(preserveProgress = true)
            return
        }

        lastAIGenerationProgress = 0f
        lastAIGenerationMessage = getString(R.string.ai_progress_getting_started)
        lastAIGenerationRawMessage = ""
        showProgressOverlay(preserveProgress = false)
        startAIGenerationInBackground(selectedBitmap!!)
    }

    private fun openGeneratedRoomWithResult(result: PhotoRoomGenerationService.GenerationResult) {
        val intent = Intent(this, GLBRoomActivity::class.java).apply {
            putExtra(GLBRoomActivity.EXTRA_GLB_PATH, result.glbFile.absolutePath)
            putExtra(GLBRoomActivity.EXTRA_ROOM_NAME, getString(R.string.room_viewer_your_room))
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
            Toast.makeText(this, getString(R.string.photo_room_no_image_selected), Toast.LENGTH_SHORT).show()
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
        cameraModeView.visibility = View.GONE
        initialView.visibility = View.VISIBLE
        selectedBitmap = null
        selectedImageUri = null
        photoWideAngle = false
        selectedCameraMode = CameraCaptureMode.STANDARD
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
        val displayProgress = if (preserveProgress && lastAIGenerationProgress > 0f) {
            lastAIGenerationProgress
        } else {
            0f
        }
        val displayMessage = if (preserveProgress && lastAIGenerationProgress > 0f) {
            lastAIGenerationMessage
        } else {
            getString(R.string.ai_progress_getting_started)
        }
        buildingRoomOverlay = PaafektBuildingRoomOverlay.show(rootLayout)
        buildingRoomOverlay?.setProgress(displayProgress, displayMessage)
    }

    private fun hideProgressOverlay() {
        PaafektBuildingRoomOverlay.hide(rootLayout)
        buildingRoomOverlay = null
    }
}
