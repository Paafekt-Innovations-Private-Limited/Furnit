package com.furnit.android

import android.content.Intent
import android.content.res.Configuration
import android.graphics.Bitmap
import android.graphics.Color
import android.os.Bundle
import android.os.Environment
import android.os.Handler
import android.os.Looper
import com.furnit.android.utils.CrashReporter
import com.furnit.android.utils.LogUtil
import com.furnit.android.utils.RoomDisplayName
import android.view.PixelCopy
import android.view.View
import android.widget.EditText
import android.widget.FrameLayout
import android.widget.ImageButton
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.activity.OnBackPressedCallback
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.FileProvider
import androidx.lifecycle.lifecycleScope
import android.view.ViewGroup
import android.view.Gravity
import android.view.MotionEvent
import android.graphics.drawable.GradientDrawable
import android.content.res.ColorStateList
import android.view.GestureDetector
import android.widget.ImageView
import com.furnit.android.theme.PaafektColors
import com.furnit.android.theme.PaafektDialogs
import com.furnit.android.theme.PaafektImmersiveChromeController
import com.furnit.android.theme.PaafektImmersiveSummonedToolbar
import com.furnit.android.theme.ImmersiveSummonedToolbarHolder
import com.furnit.android.theme.PaafektSavingRoomOverlay
import com.furnit.android.theme.PaafektSnackbar
import com.furnit.android.services.DepthAnythingRoomMeasurer
import com.furnit.android.models.ModelManager
import com.furnit.android.theme.PaafektDrawables
import com.furnit.android.theme.PaafektFirstRunCoachMarkController
import com.furnit.android.theme.PaafektHintController
import com.furnit.android.theme.PaafektHintViews
import com.furnit.android.theme.PaafektSpace
import com.furnit.android.theme.PaafektViewerToolbar
import com.furnit.android.utils.RoomBoundaryManager
import com.furnit.android.utils.RoomFolderMetadata
import com.furnit.android.utils.RoomSceneLighting
import io.github.sceneview.SceneView
import io.github.sceneview.node.CubeNode
import io.github.sceneview.node.ModelNode
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class ModelDetailActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "ModelDetailActivity"
        const val EXTRA_MODEL_ID = "MODEL_ID"
        const val EXTRA_GLB_PATH = "GLB_PATH"
        const val EXTRA_ROOM_NAME = "ROOM_NAME"
        const val EXTRA_IS_PREVIEW = "IS_PREVIEW"  // True if this is a preview before saving
    }

    private lateinit var sceneView: SceneView
    private lateinit var loadingIndicator: ProgressBar
    private lateinit var modelTitle: TextView
    private lateinit var modelManager: ModelManager
    private lateinit var saveButton: ImageButton
    private lateinit var shareButton: ImageButton
    private lateinit var helpButton: ImageButton
    private lateinit var brainButton: ImageButton
    private lateinit var screenshotButton: ImageButton
    private lateinit var orientationLabel: LinearLayout
    private lateinit var boundaryManager: RoomBoundaryManager
    private lateinit var viewerRootLayout: FrameLayout
    private lateinit var hintController: PaafektHintController
    private lateinit var firstRunCoachController: PaafektFirstRunCoachMarkController
    private var immersiveFitFab: LinearLayout? = null
    private var heroFitAction: () -> Unit = {}
    private val immersiveChrome = PaafektImmersiveChromeController()
    private lateinit var immersiveRestingChrome: FrameLayout
    private lateinit var summonedBottomChrome: View
    private var summonedToolbar: ImmersiveSummonedToolbarHolder? = null
    private var initialCameraPosition: io.github.sceneview.math.Position? = null
    private var initialCameraLookAt: io.github.sceneview.math.Position? = null
    private lateinit var immersiveTapDetector: GestureDetector
    private var measurementPillView: TextView? = null
    private var isPreviewMode = false
    /** True while showing an on-disk GLB preview that has not been saved to the library yet. */
    private var unsavedPreviewActive = false
    private var roomWidth: Float = RoomDefaults.DEFAULT_WIDTH_M
    private var roomHeight: Float = RoomDefaults.DEFAULT_HEIGHT_M
    private var roomDepth: Float = RoomDefaults.DEFAULT_DEPTH_M
    private var isFlatPhotoRoomMesh: Boolean = false
    private lateinit var previewBackCallback: OnBackPressedCallback

    private var glbPath: String? = null
    private var saveRoomJob: Job? = null
    private var currentModelId: String? = null
    private var currentModelNode: ModelNode? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Simple setup - let system handle insets normally
        // Edge-to-edge was causing SceneView rendering issues
        window.statusBarColor = Color.parseColor("#1C1C1E")
        window.navigationBarColor = Color.BLACK

        setContentView(R.layout.activity_model_detail)

        modelManager = ModelManager(this)
        boundaryManager = RoomBoundaryManager()

        sceneView = findViewById(R.id.sceneView)
        sceneView.post { RoomSceneLighting.applyIndoorPbrLighting(sceneView) }
        loadingIndicator = findViewById(R.id.loadingIndicator)
        modelTitle = findViewById(R.id.modelTitle)
        saveButton = findViewById(R.id.saveButton)
        shareButton = findViewById(R.id.shareButton)
        helpButton = findViewById(R.id.helpButton)
        brainButton = findViewById(R.id.brainButton)
        screenshotButton = findViewById(R.id.screenshotButton)
        orientationLabel = findViewById(R.id.orientationLabel)
        viewerRootLayout = findViewById(R.id.viewerRoot)
        hintController = PaafektHintController(viewerRootLayout)
        firstRunCoachController = PaafektFirstRunCoachMarkController(viewerRootLayout)

        findViewById<View>(R.id.topBarContainer).visibility = View.GONE
        orientationLabel.visibility = View.GONE

        isPreviewMode = intent.getBooleanExtra(EXTRA_IS_PREVIEW, false)
        installImmersiveViewerChrome()

        previewBackCallback = object : OnBackPressedCallback(false) {
            override fun handleOnBackPressed() {
                showUnsavedPreviewLeaveDialog()
            }
        }
        onBackPressedDispatcher.addCallback(this, previewBackCallback)

        val backButton: ImageButton = findViewById(R.id.backButton)
        backButton.visibility = View.GONE

        // Help — accessible from long-press on toolbar tap icon if needed; hide legacy bar.
        helpButton.visibility = View.GONE

        updateOrientationLabel()

        viewerRootLayout.post {
            firstRunCoachController.showIfNeeded(this) {
                hintController.showBottomCentered(
                    this,
                    R.drawable.ic_gesture_tap,
                    R.string.room_viewer_hero_actions_teaching_hint,
                    bottomMarginDp = 188,
                )
            }
        }

        // Check for direct GLB path first (for preview mode)
        val directGlbPath = intent.getStringExtra(EXTRA_GLB_PATH)

        if (directGlbPath != null) {
            // Direct GLB path mode (preview before save)
            unsavedPreviewActive = true
            previewBackCallback.isEnabled = true
            glbPath = directGlbPath
            roomWidth = intent.getFloatExtra("ROOM_WIDTH", RoomDefaults.widthMeters(this))
            roomHeight = intent.getFloatExtra("ROOM_HEIGHT", RoomDefaults.heightMeters(this))
            roomDepth = intent.getFloatExtra("ROOM_DEPTH", RoomDefaults.depthMeters(this))
            isFlatPhotoRoomMesh = intent.getBooleanExtra("FLAT_PHOTO_ROOM", true)
            modelTitle.text = getString(R.string.model_detail_preview)
            LogUtil.d(TAG, "Preview mode - GLB path: $directGlbPath")

            // In preview mode, show save button (down arrow), hide share button
            saveButton.visibility = View.VISIBLE
            saveButton.setOnClickListener { showSaveDialog() }
            shareButton.visibility = View.GONE

            // Brain button prompts to save first in preview mode
            brainButton.visibility = View.GONE
            heroFitAction = {
                LogUtil.d(TAG, "Brain button clicked in preview mode")
                Toast.makeText(this, getString(R.string.model_detail_save_first), Toast.LENGTH_SHORT).show()
            }

            // Screenshot works in preview mode
            screenshotButton.visibility = View.GONE

            // Verify file exists before loading
            val glbFile = File(directGlbPath)
            if (glbFile.exists()) {
                LogUtil.d(TAG, "Preview GLB exists: ${glbFile.length()} bytes")
                loadModel(directGlbPath, null, null, null)
            } else {
                LogUtil.e(TAG, "Preview GLB not found: $directGlbPath")
                Toast.makeText(this, getString(R.string.model_detail_room_not_found), Toast.LENGTH_SHORT).show()
            }
        } else {
            // Model ID mode (existing rooms - bundled vintage/cozy or from list)
            val modelId = intent.getStringExtra(EXTRA_MODEL_ID) ?: return
            val model = modelManager.getModel(modelId) ?: run {
                LogUtil.e(TAG, "Model not found for id=$modelId")
                return
            }

            LogUtil.d(TAG, "ModelDetail mode: id=$modelId name=${model.name} assetPath=${model.assetPath} isUserCreated=${model.isUserCreated}")

            currentModelId = modelId
            glbPath = model.assetPath
            modelTitle.text = getString(R.string.room_viewer_title)

            // In view mode, hide save button and show share button
            saveButton.visibility = View.GONE
            shareButton.visibility = View.VISIBLE
            shareButton.setOnClickListener { shareRoom() }

            // Brain button launches FurnitureFit segmentation with this room as background
            brainButton.visibility = View.GONE
            heroFitAction = {
                val roomFolder = java.io.File(model.assetPath).let { f ->
                    if (f.isFile) f.parent else f.absolutePath
                }
                val defaultRoomWidth = model.roomWidth ?: RoomDefaults.widthMeters(this)
                val defaultRoomHeight = model.roomHeight ?: RoomDefaults.heightMeters(this)
                val defaultRoomDepth = model.roomDepth ?: RoomDefaults.depthMeters(this)
                val absoluteFolder = if (roomFolder != null && java.io.File(roomFolder).isAbsolute) roomFolder else null
                LogUtil.d(TAG, "Brain click: ROOM_ID=${model.id} ROOM_FOLDER=$absoluteFolder (raw=$roomFolder)")
                val intent = Intent(this, FurnitureFitActivity::class.java)
                intent.putExtra("ROOM_ID", model.id)
                intent.putExtra("ROOM_NAME", model.name)
                if (absoluteFolder != null) intent.putExtra("ROOM_FOLDER", absoluteFolder)
                intent.putExtra("ROOM_WIDTH", defaultRoomWidth)
                intent.putExtra("ROOM_HEIGHT", defaultRoomHeight)
                intent.putExtra("ROOM_DEPTH", defaultRoomDepth)
                intent.putExtra("PHOTO_ORIENTATION", model.photoOrientation)
                startActivity(intent)
            }

            loadModel(model.assetPath, model.roomWidth, model.roomHeight, model.roomDepth)
            roomWidth = model.roomWidth ?: RoomDefaults.widthMeters(this)
            roomHeight = model.roomHeight ?: RoomDefaults.heightMeters(this)
            roomDepth = model.roomDepth ?: RoomDefaults.depthMeters(this)
            val roomFolder = java.io.File(model.assetPath).let { file ->
                if (file.isFile) file.parentFile else file
            }
            val metadata = roomFolder?.let { RoomFolderMetadata.readFromFolder(it) }
            isFlatPhotoRoomMesh = metadata?.type == "photo" ||
                metadata?.roomDimsApproach == "depth_anything_metric" ||
                (model.roomDepth ?: 1f) < 0.05f
            refreshMeasurementPill()
        }
    }

    /** Immersive resting ↔ summoned chrome (matches iOS ModelViewer + GLB Android). */
    private fun installImmersiveViewerChrome() {
        summonedBottomChrome = findViewById(R.id.bottomControlsContainer)
        installHeroBottomControls()
        summonedBottomChrome.visibility = View.GONE

        immersiveRestingChrome = createImmersiveRestingChrome()
        viewerRootLayout.addView(
            immersiveRestingChrome,
            FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT,
            ),
        )

        val fitFab = PaafektViewerToolbar.createPersistentFitFab(
            this@ModelDetailActivity,
            getString(R.string.room_viewer_immersive_fit_short),
        ) {
            immersiveChrome.noteChromeInteraction()
            heroFitAction()
        }
        immersiveFitFab = fitFab
        fitFab.elevation = 40f
        viewerRootLayout.addView(fitFab)

        immersiveTapDetector = GestureDetector(
            this,
            object : GestureDetector.SimpleOnGestureListener() {
                override fun onSingleTapConfirmed(e: MotionEvent): Boolean {
                    if (immersiveChrome.phase == PaafektImmersiveChromeController.Phase.RESTING) {
                        immersiveChrome.summon()
                    }
                    return false
                }
            },
        )
        installSceneViewTapToSummon()

        immersiveChrome.onPhaseChanged = { refreshImmersiveChromeVisibility() }
        refreshImmersiveChromeVisibility(animate = false)
    }

    private fun createImmersiveRestingChrome(): FrameLayout {
        return FrameLayout(this).apply {
            isClickable = false
            isFocusable = false
            val back = PaafektViewerToolbar.createFloatingBackButton(this@ModelDetailActivity) {
                handleViewerBack()
            }.apply {
                alpha = 0.55f
                contentDescription = getString(R.string.photo_room_back)
            }
            addView(
                back,
                FrameLayout.LayoutParams(
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                ).apply {
                    gravity = Gravity.START or Gravity.TOP
                    topMargin = PaafektSpace.viewerTopInset(this@ModelDetailActivity)
                    marginStart = PaafektSpace.lg(this@ModelDetailActivity)
                },
            )

            measurementPillView = TextView(this@ModelDetailActivity).apply {
                text = restingMeasurementPillText()
                textSize = 12f
                setTextColor(PaafektColors.textSecondary)
                setPadding(dpToPx(12), dpToPx(8), dpToPx(12), dpToPx(8))
                background = PaafektDrawables.hintChip()
                visibility = if (shouldHideRoomMeasurementChrome()) View.GONE else View.VISIBLE
            }
            addView(
                measurementPillView,
                FrameLayout.LayoutParams(
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                ).apply {
                    gravity = Gravity.BOTTOM or Gravity.CENTER_HORIZONTAL
                    bottomMargin = dpToPx(72)
                },
            )

            val summonQuiet = PaafektViewerToolbar.createQuietSummonButton(
                this@ModelDetailActivity,
                getString(R.string.room_viewer_immersive_show_controls),
            ) {
                immersiveChrome.summon()
            }
            addView(
                summonQuiet,
                PaafektViewerToolbar.quietSummonButtonLayoutParams(this@ModelDetailActivity),
            )
            installRestingChromeTouchPassthrough(this)
        }
    }

    /** SceneView/Filament: non-consuming tap detector; camera gestures stay on SceneView. */
    private fun installSceneViewTapToSummon() {
        sceneView.setOnTouchListener { _, event ->
            if (immersiveChrome.phase == PaafektImmersiveChromeController.Phase.RESTING) {
                immersiveTapDetector.onTouchEvent(event)
            }
            false
        }
    }

    /** Forward drags/pinches through resting chrome to SceneView; buttons keep their clicks. */
    private fun installRestingChromeTouchPassthrough(overlay: FrameLayout) {
        overlay.setOnTouchListener { _, event ->
            if (immersiveChrome.phase == PaafektImmersiveChromeController.Phase.RESTING) {
                immersiveTapDetector.onTouchEvent(event)
                if (!isTouchOnRestingChromeControl(overlay, event)) {
                    sceneView.dispatchTouchEvent(event)
                }
            }
            false
        }
    }

    private fun isTouchOnRestingChromeControl(overlay: FrameLayout, event: MotionEvent): Boolean {
        for (index in 0 until overlay.childCount) {
            val child = overlay.getChildAt(index)
            if (child.visibility != View.VISIBLE) continue
            if (!child.isClickable && !child.isFocusable) continue
            val locationOnScreen = IntArray(2)
            child.getLocationOnScreen(locationOnScreen)
            val touchX = event.rawX
            val touchY = event.rawY
            if (touchX >= locationOnScreen[0] &&
                touchX < locationOnScreen[0] + child.width &&
                touchY >= locationOnScreen[1] &&
                touchY < locationOnScreen[1] + child.height
            ) {
                return true
            }
        }
        return false
    }

    private fun shouldHideRoomMeasurementChrome(): Boolean = isPreviewMode || unsavedPreviewActive

    private fun restingMeasurementPillText(): String {
        val text = com.furnit.android.services.RoomMeasurementDisplay.restingPillText(
            width = roomWidth,
            height = roomHeight,
            depth = roomDepth,
            emphasizeHeight = isFlatPhotoRoomMesh,
        ) { heightMeters ->
            getString(R.string.approximate_room_height, heightMeters)
        }
        return text ?: getString(R.string.approximate_room_height, roomHeight)
    }

    private fun refreshMeasurementPill() {
        measurementPillView?.let { pill ->
            pill.text = restingMeasurementPillText()
            pill.visibility = if (shouldHideRoomMeasurementChrome()) View.GONE else View.VISIBLE
        }
    }

    private fun refreshImmersiveChromeVisibility(animate: Boolean = true) {
        if (!::immersiveRestingChrome.isInitialized) return
        immersiveChrome.applyPhase(
            this,
            restingViews = listOf(immersiveRestingChrome),
            summonedViews = listOf(summonedBottomChrome),
            animate = animate,
        )
        summonedBottomChrome.elevation = 40f
        immersiveFitFab?.elevation = 41f
        immersiveFitFab?.let { viewerRootLayout.bringChildToFront(it) }
    }

    private fun recenterCamera() {
        val position = initialCameraPosition ?: return
        val lookAt = initialCameraLookAt ?: return
        sceneView.cameraNode.apply {
            this.position = position
            lookAt(lookAt)
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
        if (shouldHideRoomMeasurementChrome()) return
        val heightLabel = if (isFlatPhotoRoomMesh) {
            getString(R.string.room_dimensions_whd_near_accurate, roomWidth, roomHeight, roomDepth)
        } else {
            getString(R.string.approximate_room_height, roomHeight)
        }
        hintController.showText(
            this,
            R.drawable.ic_ruler,
            heightLabel,
            topMarginDp = 52,
        )
    }

    private fun dpToPx(dp: Int): Int {
        return (dp * resources.displayMetrics.density).toInt()
    }

    private fun handleViewerBack() {
        if (unsavedPreviewActive) {
            showUnsavedPreviewLeaveDialog()
        } else {
            finish()
        }
    }

    private fun installHeroBottomControls() {
        brainButton.visibility = View.GONE
        screenshotButton.visibility = View.GONE

        val bottomContainer = findViewById<FrameLayout>(R.id.bottomControlsContainer)
        bottomContainer.removeAllViews()

        val holder = PaafektImmersiveSummonedToolbar.createBottomChrome(
            this,
            onTapToHide = { immersiveChrome.immerse() },
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
                        this,
                        R.drawable.ic_gesture_pinch,
                        R.string.room_viewer_navigation_teaching_hint,
                    )
                }
            },
            onDisplayAllHelpers = {
                immersiveChrome.noteChromeInteraction()
                showAllGestureHelpers()
            },
            onFullVideo = {},
            onArSizing = {},
            onCapture = {
                immersiveChrome.noteChromeInteraction()
                Toast.makeText(this, getString(R.string.model_detail_taking_screenshot), Toast.LENGTH_SHORT).show()
                takeScreenshot()
            },
            includeFurnitureFitExtras = false,
        )
        summonedToolbar = holder
        bottomContainer.addView(
            holder.root,
            FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT,
            ).apply { gravity = Gravity.BOTTOM },
        )
    }

    private fun showHelpDialog() {
        AlertDialog.Builder(this)
            .setTitle("3D Room Controls")
            .setMessage("• Drag on screen to look around\n\n• Pinch to zoom in/out\n\n• Tap the gold Fit button (bottom-right) to try furniture fit\n\n• Tap the room to summon controls, then Capture to save a screenshot")
            .setPositiveButton("OK", null)
            .show()
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

    private fun showUnsavedPreviewLeaveDialog() {
        AlertDialog.Builder(this, R.style.DarkDialogTheme)
            .setTitle(R.string.room_preview_leave_title)
            .setMessage(R.string.room_preview_leave_message)
            .setNegativeButton(R.string.room_preview_leave_stay, null)
            .setPositiveButton(R.string.room_preview_leave_confirm) { _, _ ->
                unsavedPreviewActive = false
                previewBackCallback.isEnabled = false
                finish()
            }
            .show()
    }

    private fun saveProgressSubtitle(progress: Float): String {
        return when {
            progress < 0.58f -> getString(R.string.room_viewer_measuring_room)
            progress < 0.85f -> getString(R.string.generation_progress_generating_3d_model)
            else -> getString(R.string.room_viewer_saving_room_ellipsis)
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

    private fun saveRoom(name: String) {
        val path = glbPath
        if (path == null) {
            Toast.makeText(this, getString(R.string.model_detail_no_room_data), Toast.LENGTH_SHORT).show()
            return
        }
        if (!ModelManager.isRoomNameAvailable(this, name)) {
            Toast.makeText(this, getString(R.string.home_room_name_duplicate), Toast.LENGTH_SHORT).show()
            return
        }

        val overlay = PaafektSavingRoomOverlay.show(viewerRootLayout)
        overlay.setTitle(getString(R.string.room_viewer_saving_room))
        overlay.setProgress(0.02f, getString(R.string.room_viewer_measuring_room))

        saveRoomJob?.cancel()
        saveRoomJob = lifecycleScope.launch {
            try {
                delay(220)

                val glbFile = File(path)
                val previewRoomFolder = glbFile.parentFile
                    ?: throw IllegalStateException("Missing room folder")

                val sourcePhoto = resolveSourcePhotoFile()
                if (sourcePhoto != null && isFlatPhotoRoomMesh) {
                    val progressJob = launch {
                        var progress = 0.05f
                        while (progress < 0.85f && isActive) {
                            overlay.setProgress(progress, saveProgressSubtitle(progress))
                            delay(100)
                            progress += 0.02f
                        }
                    }
                    val measured = withContext(Dispatchers.Default) {
                        DepthAnythingRoomMeasurer.measureFromFile(this@ModelDetailActivity, sourcePhoto)
                    }
                    progressJob.cancel()
                    if (measured.measured) {
                        roomWidth = measured.width
                        roomHeight = measured.height
                        roomDepth = measured.depth
                    }
                    overlay.setProgress(0.88f, getString(R.string.room_viewer_saving_room_ellipsis))
                } else {
                    overlay.setProgress(0.35f, getString(R.string.room_viewer_saving_room_ellipsis))
                }

                withContext(Dispatchers.IO) {
                    val roomsDir = File(filesDir, "rooms").apply { mkdirs() }
                    val savedRoomFolder = File(roomsDir, previewRoomFolder.name)
                    previewRoomFolder.copyRecursively(savedRoomFolder, overwrite = true)

                    val createdAtMillis = System.currentTimeMillis()
                    val metadataFile = File(savedRoomFolder, "metadata.txt")
                    metadataFile.writeText(
                        buildString {
                            append("name=$name\n")
                            append("created=$createdAtMillis\n")
                            append("type=manual\n")
                            append("roomWidth=$roomWidth\n")
                            append("roomHeight=$roomHeight\n")
                            append("roomDepth=$roomDepth\n")
                        },
                    )

                    previewRoomFolder.parentFile?.deleteRecursively()
                }

                overlay.setProgress(1f, getString(R.string.room_viewer_saving_room_ellipsis))
                delay(280)
                PaafektSavingRoomOverlay.hide(viewerRootLayout)

                PaafektSnackbar.showRoomSaved(viewerRootLayout, name)
                LogUtil.d(TAG, "Room saved: $name")

                viewerRootLayout.postDelayed({
                    val intent = Intent(this@ModelDetailActivity, ContentActivity::class.java)
                    intent.flags = Intent.FLAG_ACTIVITY_CLEAR_TOP or Intent.FLAG_ACTIVITY_NEW_TASK
                    startActivity(intent)
                    finish()
                }, 1200)
            } catch (e: Exception) {
                PaafektSavingRoomOverlay.hide(viewerRootLayout)
                LogUtil.e(TAG, "Failed to save room", e)
                Toast.makeText(
                    this@ModelDetailActivity,
                    getString(R.string.photo_room_error_load, e.message ?: ""),
                    Toast.LENGTH_SHORT,
                ).show()
                CrashReporter.report(this@ModelDetailActivity, e, "Model detail — save room")
            }
        }
    }

    private fun shareRoom() {
        val path = glbPath ?: return
        try {
            val glbFile = File(path)
            if (!glbFile.exists()) {
                Toast.makeText(this, getString(R.string.model_detail_room_not_found), Toast.LENGTH_SHORT).show()
                return
            }

            val uri = FileProvider.getUriForFile(this, "${packageName}.fileprovider", glbFile)
            val shareIntent = Intent(Intent.ACTION_SEND).apply {
                type = "model/gltf-binary"
                putExtra(Intent.EXTRA_STREAM, uri)
                addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            }
            startActivity(Intent.createChooser(shareIntent, "Share Room"))
        } catch (e: Exception) {
            LogUtil.e(TAG, "Failed to share room", e)
            Toast.makeText(this, getString(R.string.model_detail_failed_share), Toast.LENGTH_SHORT).show()
            CrashReporter.report(this, e, "Model detail — share room")
        }
    }

    private fun takeScreenshot() {
        try {
            val bitmap = Bitmap.createBitmap(sceneView.width, sceneView.height, Bitmap.Config.ARGB_8888)
            PixelCopy.request(
                sceneView,
                bitmap,
                { result ->
                    if (result == PixelCopy.SUCCESS) {
                        saveAndShareScreenshot(bitmap)
                    } else {
                        runOnUiThread {
                            Toast.makeText(this, getString(R.string.model_detail_failed_capture), Toast.LENGTH_SHORT).show()
                        }
                    }
                },
                Handler(Looper.getMainLooper())
            )
        } catch (e: Exception) {
            LogUtil.e(TAG, "Screenshot failed", e)
            Toast.makeText(this, getString(R.string.model_detail_screenshot_failed), Toast.LENGTH_SHORT).show()
            CrashReporter.report(this, e, "Model detail — screenshot capture")
        }
    }

    private fun saveAndShareScreenshot(bitmap: Bitmap) {
        try {
            val timeStamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())
            val fileName = "Room_$timeStamp.png"

            // Save to gallery using MediaStore (Android 10+)
            val contentValues = android.content.ContentValues().apply {
                put(android.provider.MediaStore.Images.Media.DISPLAY_NAME, fileName)
                put(android.provider.MediaStore.Images.Media.MIME_TYPE, "image/png")
                put(android.provider.MediaStore.Images.Media.RELATIVE_PATH, "${Environment.DIRECTORY_PICTURES}/Screenshots")
            }

            val resolver = contentResolver
            val uri = resolver.insert(android.provider.MediaStore.Images.Media.EXTERNAL_CONTENT_URI, contentValues)

            if (uri != null) {
                resolver.openOutputStream(uri)?.use { out ->
                    bitmap.compress(Bitmap.CompressFormat.PNG, 100, out)
                }

                runOnUiThread {
                    Toast.makeText(this, getString(R.string.smartypants_saved_screenshots), Toast.LENGTH_SHORT).show()
                }

                // Share the screenshot
                val shareIntent = Intent(Intent.ACTION_SEND).apply {
                    type = "image/png"
                    putExtra(Intent.EXTRA_STREAM, uri)
                    addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                }
                startActivity(Intent.createChooser(shareIntent, "Share Screenshot"))
            } else {
                runOnUiThread {
                    Toast.makeText(this, getString(R.string.smartypants_failed_save_screenshot), Toast.LENGTH_SHORT).show()
                }
            }

        } catch (e: Exception) {
            LogUtil.e(TAG, "Failed to save screenshot", e)
            runOnUiThread {
                Toast.makeText(this, getString(R.string.model_detail_failed_save_screenshot_message, e.message ?: ""), Toast.LENGTH_SHORT).show()
                CrashReporter.report(this@ModelDetailActivity, e, "Model detail — save/share screenshot")
            }
        }
    }

    private fun updateOrientationLabel() {
        val isPortrait = resources.configuration.orientation == Configuration.ORIENTATION_PORTRAIT
        val subtitleView = findViewById<TextView>(R.id.orientationSubtitle)
        val titleView = findViewById<TextView>(R.id.orientationTitle)

        if (isPortrait) {
            subtitleView.text = getString(R.string.orientation_held_vertically)
            titleView.text = getString(R.string.orientation_portrait)
        } else {
            subtitleView.text = getString(R.string.orientation_held_horizontally)
            titleView.text = getString(R.string.orientation_landscape)
        }
    }

    override fun onConfigurationChanged(newConfig: Configuration) {
        super.onConfigurationChanged(newConfig)
        updateOrientationLabel()
    }

    private fun loadModel(
        assetPath: String,
        roomWidth: Float?,
        roomHeight: Float?,
        roomDepth: Float?,
    ) {
        lifecycleScope.launch {
            try {
                val isFileSystemPath = assetPath.startsWith("/")
                LogUtil.d(TAG, "=== Loading Model ===")
                LogUtil.d(TAG, "  Path type: ${if (isFileSystemPath) "FILE SYSTEM" else "ASSETS"}")
                LogUtil.d(TAG, "  Path: $assetPath")
                LogUtil.d(TAG, "  Room dims: ${roomWidth ?: "null"} x ${roomHeight ?: "null"} x ${roomDepth ?: "null"}")

                val modelInstance = if (isFileSystemPath) {
                    // File system path - load from file (user-created rooms)
                    val file = File(assetPath)
                    LogUtil.d(TAG, "  File exists: ${file.exists()}, size: ${file.length()} bytes")
                    val bytes = file.readBytes()
                    val buffer = ByteBuffer.wrap(bytes)
                    sceneView.modelLoader.createModelInstance(buffer)
                } else {
                    // Asset path - load from assets (bundled rooms like vintage)
                    sceneView.modelLoader.createModelInstance(
                        assetFileLocation = assetPath
                    )
                }

                // Don't scale - keep original size
                val modelNode = ModelNode(
                    modelInstance = modelInstance,
                    scaleToUnits = null  // Keep original scale
                )

                sceneView.addChildNode(modelNode)
                currentModelNode = modelNode

                // Center the room at origin (match Swift: use actual model bounds so camera sees full room)
                val bboxCenter = modelNode.center
                val bboxExtents = modelNode.extents
                LogUtil.d(TAG, "  Model bbox center: (${bboxCenter.x}, ${bboxCenter.y}, ${bboxCenter.z}) extents: (${bboxExtents.x}, ${bboxExtents.y}, ${bboxExtents.z})")

                modelNode.position = io.github.sceneview.math.Position(
                    -bboxCenter.x,
                    -bboxCenter.y,
                    -bboxCenter.z
                )
                LogUtil.d(TAG, "  Model position set to center at origin: (${modelNode.position.x}, ${modelNode.position.y}, ${modelNode.position.z})")

                addDebugCuboid()

                // Use actual model extents so camera bounds match the geometry (not passed dims which may be wrong)
                val w = bboxExtents.x
                val h = bboxExtents.y
                val d = bboxExtents.z
                boundaryManager.initializeFromCenteredExtents(width = w, height = h, depth = d)
                val cameraSetup = if (isFileSystemPath) {
                    LogUtil.d(TAG, "[ModelDetail] getCameraAtBackCenter CALLED (bbox ${w}x${h}x${d})")
                    boundaryManager.getCameraAtBackCenter()
                } else {
                    LogUtil.d(TAG, "[ModelDetail] getCameraOutsideBackView CALLED (bundled bbox ${w}x${h}x${d})")
                    boundaryManager.getCameraOutsideBackView()
                }
                LogUtil.d(TAG, "[ModelDetail] camera SET pos=(${cameraSetup.position.x}, ${cameraSetup.position.y}, ${cameraSetup.position.z}) lookAt=(${cameraSetup.lookAt.x}, ${cameraSetup.lookAt.y}, ${cameraSetup.lookAt.z})")
                initialCameraPosition = cameraSetup.position
                initialCameraLookAt = cameraSetup.lookAt

                // Position camera IMMEDIATELY after adding model
                sceneView.cameraNode.apply {
                    position = cameraSetup.position
                    lookAt(cameraSetup.lookAt)
                }

                LogUtil.d(TAG, "  Camera position set: ${cameraSetup.position}")
                LogUtil.d(TAG, "  Camera lookAt: ${cameraSetup.lookAt}")

                // Re-apply camera position after a frame to override any manipulator reset
                sceneView.post {
                    sceneView.cameraNode.apply {
                        position = cameraSetup.position
                        lookAt(cameraSetup.lookAt)
                    }
                    LogUtil.d(TAG, "  Camera position re-applied (post)")
                }

                // Also re-apply after a short delay to handle async initialization
                sceneView.postDelayed({
                    sceneView.cameraNode.apply {
                        position = cameraSetup.position
                        lookAt(cameraSetup.lookAt)
                    }
                    LogUtil.d(TAG, "  Camera position re-applied (delayed)")
                    LogUtil.d(TAG, "  Final camera: ${sceneView.cameraNode.position}")
                }, 100)

                LogUtil.d(TAG, "=== Model Load Complete ===")

                loadingIndicator.visibility = View.GONE

            } catch (e: Exception) {
                LogUtil.e(TAG, "Failed to load model", e)
                e.printStackTrace()
                loadingIndicator.visibility = View.GONE
                modelTitle.text = getString(R.string.model_detail_failed_load, e.message ?: "")
                runOnUiThread {
                    CrashReporter.report(this@ModelDetailActivity, e, "Model detail — load 3D model")
                }
            }
        }
    }

    /**
     * Add wireframe outline of room bounds for debugging coordinates
     * Room dimensions: Width=4 (X: -2 to +2), Depth=4.5 (Z: -2.25 to +2.25), Height=2.8 (Y: 0 to 2.8)
     */
    private fun addDebugCuboid() {
        try {
            val bounds = boundaryManager.getBounds() ?: return
            val beamThickness = 0.08f  // Thicker beams for better visibility

            // Material for wireframe - bright green for visibility
            val wireMaterial = sceneView.materialLoader.createColorInstance(
                color = Color.parseColor("#00FF00"),  // Bright green
                metallic = 0.0f,
                roughness = 1.0f,
                reflectance = 0.0f
            )

            // Room corner coordinates
            val minX = bounds.minX
            val maxX = bounds.maxX
            val minY = bounds.minY  // Floor
            val maxY = bounds.maxY  // Ceiling
            val minZ = bounds.minZ  // Front wall
            val maxZ = bounds.maxZ  // Back wall

            LogUtil.d(TAG, "Room bounds: X[$minX to $maxX], Y[$minY to $maxY], Z[$minZ to $maxZ]")

            // Create 12 edge beams

            // 4 vertical edges (floor to ceiling)
            addBeam(minX, minY, minZ, beamThickness, bounds.height, beamThickness, wireMaterial) // Front-left
            addBeam(maxX, minY, minZ, beamThickness, bounds.height, beamThickness, wireMaterial) // Front-right
            addBeam(minX, minY, maxZ, beamThickness, bounds.height, beamThickness, wireMaterial) // Back-left
            addBeam(maxX, minY, maxZ, beamThickness, bounds.height, beamThickness, wireMaterial) // Back-right

            // 4 floor edges (horizontal on floor)
            addBeam(minX, minY, minZ, bounds.width, beamThickness, beamThickness, wireMaterial) // Front edge
            addBeam(minX, minY, maxZ, bounds.width, beamThickness, beamThickness, wireMaterial) // Back edge
            addBeam(minX, minY, minZ, beamThickness, beamThickness, bounds.depth, wireMaterial) // Left edge
            addBeam(maxX, minY, minZ, beamThickness, beamThickness, bounds.depth, wireMaterial) // Right edge

            // 4 ceiling edges (horizontal at ceiling)
            addBeam(minX, maxY, minZ, bounds.width, beamThickness, beamThickness, wireMaterial) // Front edge
            addBeam(minX, maxY, maxZ, bounds.width, beamThickness, beamThickness, wireMaterial) // Back edge
            addBeam(minX, maxY, minZ, beamThickness, beamThickness, bounds.depth, wireMaterial) // Left edge
            addBeam(maxX, maxY, minZ, beamThickness, beamThickness, bounds.depth, wireMaterial) // Right edge

            // Add corner markers with coordinate labels
            addCornerMarker(minX, minY, minZ, Color.RED, "Front-Left-Floor")
            addCornerMarker(maxX, minY, minZ, Color.BLUE, "Front-Right-Floor")
            addCornerMarker(minX, minY, maxZ, Color.YELLOW, "Back-Left-Floor")
            addCornerMarker(maxX, minY, maxZ, Color.MAGENTA, "Back-Right-Floor")

            LogUtil.d(TAG, "Room wireframe added with corner markers")
        } catch (e: Exception) {
            LogUtil.e(TAG, "Failed to add room wireframe", e)
        }
    }

    private fun addBeam(
        startX: Float, startY: Float, startZ: Float,
        width: Float, height: Float, depth: Float,
        material: com.google.android.filament.MaterialInstance
    ) {
        val beam = CubeNode(
            engine = sceneView.engine,
            size = dev.romainguy.kotlin.math.Float3(width, height, depth),
            materialInstance = material
        )
        // Position beam so it starts at the given corner
        beam.position = io.github.sceneview.math.Position(
            startX + width / 2f,
            startY + height / 2f,
            startZ + depth / 2f
        )
        sceneView.addChildNode(beam)
    }

    private fun addCornerMarker(x: Float, y: Float, z: Float, color: Int, label: String) {
        val marker = CubeNode(
            engine = sceneView.engine,
            size = dev.romainguy.kotlin.math.Float3(0.25f, 0.25f, 0.25f),  // Bigger markers
            materialInstance = sceneView.materialLoader.createColorInstance(
                color = color,
                metallic = 0.0f,
                roughness = 0.5f,
                reflectance = 0.3f
            )
        )
        marker.position = io.github.sceneview.math.Position(x, y + 0.125f, z)
        sceneView.addChildNode(marker)
        LogUtil.d(TAG, "Corner marker '$label' at ($x, $y, $z)")
    }

    override fun onDestroy() {
        saveRoomJob?.cancel()
        immersiveChrome.destroy()
        super.onDestroy()
    }

}
