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
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import com.furnit.android.utils.CrashReporter
import com.furnit.android.utils.DebugLogger
import com.furnit.android.utils.LogUtil
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.view.animation.AccelerateDecelerateInterpolator
import android.widget.*
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.launch
import androidx.core.content.FileProvider
import com.google.android.material.progressindicator.CircularProgressIndicator
import com.furnit.android.ar.ArSupportChecker
import com.furnit.android.ar.MetricAnchor
import com.furnit.android.models.PhotoOrientation
import com.furnit.android.models.RoomStructure
import com.furnit.android.services.SharpService
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
    private lateinit var progressRing: CircularProgressIndicator
    private lateinit var progressText: TextView
    private lateinit var progressPercent: TextView
    /** Host of the ring — subtle pulse animation. */
    private var progressRingHost: View? = null
    private var progressOverlayPulse: AnimatorSet? = null
    private var phaseStripViews: Array<TextView> = emptyArray()
    private lateinit var selectedImageView: ImageView
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
    private var pendingMetricAnchors: ArrayList<MetricAnchor>? = null

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
        val anchors = data?.getSerializableExtra(ArDepthPhotoCaptureActivity.EXTRA_METRIC_ANCHORS) as? ArrayList<MetricAnchor>
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

        // Progress overlay - hidden initially
        progressOverlay = createProgressOverlay()
        progressOverlay.visibility = View.GONE
        rootLayout.addView(progressOverlay)

        setContentView(rootLayout)

        // Preload ExecuTorch Part1 when backend is ExecuTorch (hides "stuck at 5%" stall at Generate)
        lifecycleScope.launch {
            SharpService.getInstance(this@SinglePhotoRoomActivity).preloadSharpModels()
        }
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
                setOnClickListener { finish() }
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
            setPadding(32, 48, 32, 32)

            // Back button
            val backBtn = TextView(this@SinglePhotoRoomActivity).apply {
                text = "< Back"
                textSize = 16f
                setTextColor(Color.parseColor("#007AFF"))
                setPadding(0, 0, 0, 16)
                setOnClickListener {
                    showInitialView()
                }
            }
            addView(backBtn)

            // Image preview
            selectedImageView = ImageView(this@SinglePhotoRoomActivity).apply {
                scaleType = ImageView.ScaleType.CENTER_CROP
                setBackgroundColor(Color.parseColor("#E0E0E0"))
            }
            addView(selectedImageView, LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                400
            ).apply { setMargins(0, 0, 0, 8) })

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
                    text = "\uD83D\uDCF1" // Phone icon
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
            addView(orientationIndicator, LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            ).apply { setMargins(0, 0, 0, 8) })

            // Wide angle (0.5x) toggle – tap to set if photo was taken with ultra-wide lens (fixes camera position in viewer)
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
            addView(wideAngleRow, LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            ).apply { setMargins(0, 0, 0, 16) })

            // Title
            val title = TextView(this@SinglePhotoRoomActivity).apply {
                text = "How would you like to create your room?"
                textSize = 18f
                setTypeface(null, Typeface.BOLD)
                setTextColor(Color.parseColor("#333333"))
                gravity = Gravity.CENTER
            }
            addView(title)

            val subtitle = TextView(this@SinglePhotoRoomActivity).apply {
                text = "Tap an option below"
                textSize = 14f
                setTextColor(Color.parseColor("#666666"))
                gravity = Gravity.CENTER
                setPadding(0, 8, 0, 24)
            }
            addView(subtitle)

            // AI Room option (Sharp) - subtitle shows live progress when generation runs in background
            val aiOption = createOptionCard(
                icon = "\uD83E\uDE84", // Magic wand
                title = "AI Room",
                subtitle = "AI-powered 3D generation",
                bgColor = "#F3E5F5",
                onSubtitleCreated = { view -> aiOptionSubtitleView = view }
            ) {
                onAIRoomSelected()
            }
            addView(aiOption)

            // Manual Setup option
            val manualOption = createOptionCard(
                icon = "\uD83D\uDCCF", // Ruler
                title = "Manual Setup",
                subtitle = "Adjust room boundaries manually",
                bgColor = "#FFF3E0"
            ) {
                onManualSetupSelected()
            }
            addView(manualOption)

            // Change photo button
            val changePhotoBtn = TextView(this@SinglePhotoRoomActivity).apply {
                text = "Choose Different Photo"
                textSize = 14f
                setTextColor(Color.parseColor("#666666"))
                gravity = Gravity.CENTER
                setPadding(0, 32, 0, 0)
                setOnClickListener {
                    openImagePicker()
                }
            }
            addView(changePhotoBtn)
        }
    }

    private fun createOptionCard(
        icon: String,
        title: String,
        subtitle: String,
        bgColor: String,
        onSubtitleCreated: ((TextView) -> Unit)? = null,
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
            }
            addView(textContainer)

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
    }

    /** Start AI generation in background when photo is selected. Cancel on Manual/Back/Change. */
    private fun startAIGenerationInBackground(bitmap: Bitmap) {
        cancelAndReleaseAI()
        aiGenerationResult = null
        aiGenerationRunning = true
        val sharpService = SharpService.getInstance(this)
        val orientationForMetadata = metadataOrientationStringForViewer()
        aiGenerationHandle = sharpService.startGenerationInBackground(
            bitmap,
            object : SharpService.ProgressCallback {
            override fun onProgress(progress: Float, message: String) {
                runOnUiThread {
                    logProgress0("SinglePhotoRoomActivity.kt:onProgress", "callback", mapOf(
                        "progress" to progress, "message" to message, "aiGenerationRunning" to aiGenerationRunning,
                        "aiRoomOverlayRequested" to aiRoomOverlayRequested
                    ))
                    if (aiGenerationRunning) {
                        updateAIOptionProgress(progress, message)
                        if (aiRoomOverlayRequested) updateProgressOverlay(progress, message)
                    }
                }
            }
            override fun onComplete(result: SharpService.GenerationResult) {
                runOnUiThread {
                    aiGenerationRunning = false
                    aiGenerationResult = result
                    aiGenerationHandle = null
                    updateAIOptionProgress(1f, "Ready")
                    hideProgressOverlay()
                    if (aiRoomOverlayRequested) {
                        aiRoomOverlayRequested = false
                        openSharpRoomWithResult(result)
                    }
                    DebugLogger.d("SinglePhotoRoom", "AI generation completed in background")
                }
            }
            override fun onError(message: String) {
                runOnUiThread {
                    aiGenerationRunning = false
                    aiGenerationResult = null
                    aiGenerationHandle = null
                    aiRoomOverlayRequested = false
                    updateAIOptionProgress(0f, "Failed")
                    hideProgressOverlay()
                    Toast.makeText(this@SinglePhotoRoomActivity, message, Toast.LENGTH_LONG).show()
                    DebugLogger.eDebugMode("SinglePhotoRoom", "AI generation failed: $message")
                    CrashReporter.report(
                        this@SinglePhotoRoomActivity,
                        RuntimeException(message),
                        "Single photo room — AI / SHARP generation",
                    )
                }
            }
        },
            viewerPhotoOrientation = orientationForMetadata,
            viewerPhotoWideAngle = photoWideAngle,
            orientationLockedByUser = orientationUserOverridden,
            sourcePhotoUri = selectedImageUri,
            metricAnchors = pendingMetricAnchors,
        )
    }

    /** Cancel AI generation and release model memory. Call when user chooses Manual/Back/Change. */
    private fun cancelAndReleaseAI() {
        aiGenerationHandle?.cancel()
        aiGenerationHandle = null
        aiGenerationRunning = false
        aiGenerationResult = null
        SharpService.getInstance(this).release()
        DebugLogger.d("SinglePhotoRoom", "AI cancelled and memory released")
    }

    private var aiOptionSubtitleView: TextView? = null

    /** Last progress from generation callback — used when showing overlay for already-running gen. */
    private var lastAIGenerationProgress: Float = 0f
    private var lastAIGenerationMessage: String = "Getting started…"

    private fun updateAIOptionProgress(progress: Float, message: String) {
        lastAIGenerationProgress = progress
        val friendly = toFriendlyMessage(progress, message)
        lastAIGenerationMessage = friendly
        aiOptionSubtitleView?.text = if (progress >= 1f) "Ready — tap to view" else if (progress > 0f) "$friendly (${(progress * 100).toInt()}%)" else "Create a 3D room from your photo"
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
        showProgressOverlay(preserveProgress = false)
        startAIGenerationInBackground(selectedBitmap!!)
        aiRoomOverlayRequested = true
    }

    private fun openSharpRoomWithResult(result: SharpService.GenerationResult) {
        val intent = Intent(this, SharpRoomActivity::class.java).apply {
            putExtra(SharpRoomActivity.EXTRA_PLY_PATH, result.classicPlyFile.absolutePath)
            putExtra(SharpRoomActivity.EXTRA_ROOM_FOLDER, result.plyFile.parentFile?.absolutePath)
            putExtra(SharpRoomActivity.EXTRA_ROOM_WIDTH, result.roomWidth)
            putExtra(SharpRoomActivity.EXTRA_ROOM_HEIGHT, result.roomHeight)
            putExtra(SharpRoomActivity.EXTRA_ROOM_DEPTH, result.roomDepth)
            result.roomCenterX?.let { putExtra(SharpRoomActivity.EXTRA_ROOM_CENTER_X, it) }
            result.roomCenterY?.let { putExtra(SharpRoomActivity.EXTRA_ROOM_CENTER_Y, it) }
            result.roomCenterZ?.let { putExtra(SharpRoomActivity.EXTRA_ROOM_CENTER_Z, it) }
            putExtra(SharpRoomActivity.EXTRA_ALLOW_SAVE, true)
            putExtra("photo_orientation", metadataOrientationStringForViewer())
            putExtra(SharpRoomActivity.EXTRA_PHOTO_WIDE_ANGLE, photoWideAngle)
            // Mark this as a temporary SHARP room: if the user backs out without saving,
            // SharpRoomActivity will delete the folder so it doesn't appear in the list.
            putExtra(SharpRoomActivity.EXTRA_IS_TEMP_SHARP_ROOM, true)
        }
        LogUtil.i(
            "SHARP_ROOM_MEAS",
            "[open_sharp_viewer] W×H×D=${result.roomWidth}×${result.roomHeight}×${result.roomDepth} " +
                "center=(${result.roomCenterX},${result.roomCenterY},${result.roomCenterZ}) " +
                "folder=${result.plyFile.parentFile?.absolutePath} classic=${result.classicPlyFile.name}",
        )
        startActivity(intent)
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

    private fun showInitialView() {
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
                val phaseNames = listOf("Prepare", "SHARP", "Save")
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

                addView(TextView(this@SinglePhotoRoomActivity).apply {
                    text = "Usually ~2 minutes. You can leave anytime.\nWork continues in the background."
                    textSize = 12f
                    setTextColor(Color.parseColor("#757575"))
                    gravity = Gravity.CENTER
                    setLineSpacing(3f * density, 1f)
                    layoutParams = LinearLayout.LayoutParams(
                        ViewGroup.LayoutParams.MATCH_PARENT,
                        ViewGroup.LayoutParams.WRAP_CONTENT,
                    ).apply {
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
    }

    private fun hideProgressOverlay() {
        stopProgressOverlayPulse()
        progressOverlay.visibility = View.GONE
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
    }
}
