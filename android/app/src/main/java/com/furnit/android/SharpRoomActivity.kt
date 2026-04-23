package com.furnit.android

import android.Manifest
import android.annotation.SuppressLint
import android.content.Intent
import android.content.pm.PackageManager
import android.content.res.ColorStateList
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.widget.ImageView
import android.widget.Space
import android.os.Bundle
import android.os.Environment
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import com.furnit.android.utils.DebugLogger
import com.furnit.android.utils.LogUtil
import android.view.Gravity
import android.view.Menu
import android.view.View
import android.view.ViewGroup
import android.view.ViewTreeObserver
import android.content.pm.ActivityInfo
import android.view.WindowManager
import android.webkit.*
import android.widget.PopupMenu
import android.widget.*
import androidx.activity.result.contract.ActivityResultContracts
import com.furnit.android.ar.ArSupportChecker
import com.furnit.android.ar.FurnitureFitArCameraController
import com.furnit.android.ar.rotateToMatchLockedRoomPhoto
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.widget.AppCompatImageButton
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.core.ImageProxy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import androidx.core.view.ViewCompat
import androidx.core.view.WindowCompat
import kotlin.math.max
import kotlin.math.min
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.WindowInsetsControllerCompat
import androidx.core.widget.ImageViewCompat
import androidx.lifecycle.lifecycleScope
import androidx.webkit.WebViewAssetLoader
import com.furnit.android.models.ModelManager
import com.furnit.android.models.roomintelligence.AestheticAdvisor
import com.furnit.android.models.roomintelligence.AestheticScore
import com.furnit.android.models.roomintelligence.CornerPlacement
import com.furnit.android.models.roomintelligence.CornerPlacementSuggestion
import com.furnit.android.models.roomintelligence.FitCheckEngine
import com.furnit.android.models.roomintelligence.FitCheckResult
import com.furnit.android.models.roomintelligence.FurnitureProfile
import com.furnit.android.models.roomintelligence.HarmonyType
import com.furnit.android.models.roomintelligence.RoomFurnitureDimensions
import com.furnit.android.models.roomintelligence.RoomIntelligenceLoader
import com.furnit.android.models.roomintelligence.RoomModel
import com.furnit.android.models.roomintelligence.SurfacePalette
import com.furnit.android.models.roomintelligence.Vec3f
import com.furnit.android.utils.CrashReporter
import com.furnit.android.utils.RoomDisplayName
import com.furnit.android.utils.FurnitureSegmentationMeanColor
import com.furnit.android.utils.RoomFolderMetadata
import com.furnit.android.utils.SharpRoomDimensionsV7
import com.furnit.android.utils.SharpRoomDimensionsV7Result
import com.furnit.android.utils.SharpRoomDimensionSanitizer
import com.furnit.android.utils.SplatLoadHint
import com.furnit.android.utils.SplatLoadHintVector3
import com.furnit.android.services.FurnitureFitManager
import kotlinx.coroutines.Deferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.ByteArrayInputStream
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
 * SharpRoomActivity - WebGL-based 3D Gaussian Splat viewer
 * (Matches Swift's SharpRoomView)
 *
 * Uses THREE.js and SparkJS to render PLY files in a WebView
 */
class SharpRoomActivity : AppCompatActivity() {
    private enum class BrainSegmentationMode {
        IDENTIFY_ONLY,
        SEGMENT_SELECTED,
    }

    companion object {
        private const val TAG = "SharpRoomActivity"
        private const val SCALE_LOG_TAG = "FURNIT_SCALE"
        private const val BRAIN_BUTTON_COLOR_IDLE = "#007AFF"
        private const val BRAIN_BUTTON_COLOR_SEGMENTING = "#34C759"
        const val EXTRA_PLY_PATH = "ply_path"
        const val EXTRA_ROOM_FOLDER = "room_folder"
        const val EXTRA_ROOM_WIDTH = "room_width"
        const val EXTRA_ROOM_HEIGHT = "room_height"
        const val EXTRA_ROOM_DEPTH = "room_depth"
        const val EXTRA_ROOM_CENTER_X = "room_center_x"
        const val EXTRA_ROOM_CENTER_Y = "room_center_y"
        const val EXTRA_ROOM_CENTER_Z = "room_center_z"
        const val EXTRA_ALLOW_SAVE = "allow_save"
        /** True if the photo was taken with the wide-angle (0.5x) lens; used to adjust initial camera position. */
        const val EXTRA_PHOTO_WIDE_ANGLE = "photo_wide_angle"
        /** True when this Sharp room comes directly from a new SHARP generation (SinglePhotoRoom); delete if not saved. */
        const val EXTRA_IS_TEMP_SHARP_ROOM = "is_temp_sharp_room"
        /** When true, system back exits the viewer (to [SinglePhotoRoomActivity]) instead of walking WebView history. */
        const val EXTRA_OPENED_FROM_SINGLE_PHOTO_ROOM = "opened_from_single_photo_room"

        private const val OV_SHARE = 10001
        private const val OV_SAVE = 10002
        private const val OV_CALIBRATE = 10003
        private const val OV_RECENTER = 10004
        private const val OV_RESET_OVERLAY = 10005
        private const val OV_HELP = 10006
        private const val OV_FULL_VIDEO_IDENTIFICATIONS = 10007
    }

    /** Persist latest Spark/Box3 dimensions into room_meta.json so list screen shows accurate width/height. */
    private fun persistSparkBoxDimensionsDebounced() {
        if (hasPlausibleOpenSnapshotRoomDims() && !roomDimensionsLockedByTapeCalibration) {
            LogUtil.i(
                "SHARP_ROOM_MEAS",
                "[box3_persist] skipped; keeping saved SHARP/export dims ${openSnapshotRoomWidth}×${openSnapshotRoomHeight}×${openSnapshotRoomDepth}",
            )
            return
        }
        val folderPath = roomFolder ?: return
        val w = roomWidth
        val h = roomHeight
        if (w <= 0f || h <= 0f) return

        val folder = File(folderPath)
        sparkBoxPersistRunnable?.let { sparkBoxPersistHandler.removeCallbacks(it) }
        val runnable = Runnable {
            try {
                val prev = RoomFolderMetadata.readFromFolder(folder)
                val baseSnapshot = if (prev != null) {
                    prev.copy(
                        roomWidth = w,
                        roomHeight = h,
                        roomDepth = roomDepth
                    )
                } else {
                    RoomFolderMetadata.Snapshot(
                        name = null,
                        createdAt = folder.lastModified(),
                        type = "sharp",
                        photoOrientation = if (photoOrientation == "landscape") "landscape" else "portrait",
                        photoWideAngle = photoWideAngle,
                        roomWidth = w,
                        roomHeight = h,
                        roomDepth = roomDepth,
                        roomCenterX = null,
                        roomCenterY = null,
                        roomCenterZ = null,
                        arDisplayScale = arDisplayScale,
                        previewOnly = true,
                    )
                }
                val merged = RoomFolderMetadata.snapshotPreservingYoloFields(folder, baseSnapshot)
                RoomFolderMetadata.writeToFolder(folder, merged)
                DebugLogger.d(TAG, "Persisted Spark Box3 dimensions to room_meta.json: ${w}x${h}")
                LogUtil.i(
                    "SHARP_ROOM_MEAS",
                    "[box3_persist] room_meta.json W×H=$w×$h depth=$roomDepth arDisplayScale=$arDisplayScale folder=${folder.absolutePath}",
                )
            } catch (e: Exception) {
                DebugLogger.eDebugMode(TAG, "Failed to persist Spark Box3 dimensions", e)
            }
        }
        sparkBoxPersistRunnable = runnable
        sparkBoxPersistHandler.postDelayed(runnable, 1500L)
    }

    private fun logSharpLoadTiming(stage: String, detail: String = "") {
        val elapsedMs = SystemClock.elapsedRealtime() - sharpLoadStartMs
        val suffix = if (detail.isBlank()) "" else " $detail"
        LogUtil.i("SPLAT_LOAD", "[android_open] stage=$stage elapsed_ms=$elapsedMs$suffix")
    }

    private fun loadPersistedSplatLoadHint(roomPlyFile: File): SplatLoadHint? {
        val sidecarFile = SplatLoadHint.sidecarFileFor(roomPlyFile)
        val hint = SplatLoadHint.readFrom(sidecarFile)
        if (hint == null) {
            logSharpLoadTiming("hint_miss", "reason=missing sidecar=${sidecarFile.name}")
            return null
        }
        if (!hint.matches(roomPlyFile)) {
            logSharpLoadTiming("hint_stale", "sidecar=${sidecarFile.name}")
            return null
        }
        logSharpLoadTiming("hint_hit", "sidecar=${sidecarFile.name} splats=${hint.splatCount}")
        return hint
    }

    private fun persistSplatLoadHintFromBounds(
        fullBoundsMin: SplatLoadHintVector3,
        fullBoundsMax: SplatLoadHintVector3,
        framingBoundsMin: SplatLoadHintVector3,
        framingBoundsMax: SplatLoadHintVector3,
        centroid: SplatLoadHintVector3,
        source: String,
    ) {
        val originalPlyFile = plyPath?.let { File(it) } ?: return
        val nextHint = SplatLoadHint.createForFile(
            roomPlyFile = originalPlyFile,
            splatCount = persistedSplatLoadHint?.splatCount ?: 0,
            fullBoundsMin = fullBoundsMin,
            fullBoundsMax = fullBoundsMax,
            framingBoundsMin = framingBoundsMin,
            framingBoundsMax = framingBoundsMax,
            centroid = centroid,
        ) ?: return
        persistedSplatLoadHint = nextHint
        try {
            SplatLoadHint.writeTo(SplatLoadHint.sidecarFileFor(originalPlyFile), nextHint)
            logSharpLoadTiming("hint_saved", "source=$source sidecar=${SplatLoadHint.sidecarFileFor(originalPlyFile).name}")
        } catch (exception: Exception) {
            DebugLogger.eDebugMode(TAG, "Failed to save splat load hint", exception)
        }
    }

    private lateinit var webView: WebView
    private lateinit var loadingOverlay: FrameLayout
    private lateinit var brainProgressOverlay: FrameLayout
    private lateinit var brainDetectionOverlay: FrameLayout
    private lateinit var brainDetectionOverlayView: FurnitureFitOverlayView
    private lateinit var brainCameraPreviewView: PreviewView
    /** Bottom-left brain control; blue when idle, green while live segmentation is active. */
    private lateinit var brainModeButton: AppCompatImageButton
    private var brainActionButton: TextView? = null
    /** Top-right AR sizing control; active when the current brain session requested AR-assisted sizing. */
    private var brainArAssistButton: AppCompatImageButton? = null
    private var fullVideoIdentificationsButton: AppCompatImageButton? = null
    private lateinit var roomRulerButton: AppCompatImageButton
    /** Top chrome (back pill + ruler); used to position the room-dimensions hint below the bar. */
    private lateinit var sharpRoomTopBar: FrameLayout
    private var plyPath: String? = null
    private var roomFolder: String? = null
    private var allowSave: Boolean = true

    // Room dimensions (from intent or JS-measured)
    private var roomWidth: Float = 4.0f
    private var roomHeight: Float = 3.0f
    private var roomDepth: Float = 4.5f
    private var roomCenterX: Float = 0f
    private var roomCenterY: Float = 0f
    private var roomCenterZ: Float = 0f
    /** Isotropic scale for displayed dims vs raw SHARP bbox (ARCore calibration). */
    private var arDisplayScale: Float = 1f
    // Brain (SmartyPants) furniture calibration state (height and optional scale factor for display).
    private var brainLockedFurnitureWidthMeters: Float? = null
    private var brainLockedFurnitureHeightMeters: Float? = null
    private var brainRealFurnitureHeightMeters: Float? = null
    private var brainCalibrationScaleFactor: Float = 1.0f
    private var photoOrientation: String = "portrait"
    /** True when the photo was taken with wide-angle (0.5x) lens; viewer camera position is adjusted for wider FOV. */
    private var photoWideAngle: Boolean = false
    private var hasSavedDimensions: Boolean = false  // True if dimensions were passed from saved room (logging / open path)
    /** When the user applies tape (wall) calibration, do not let WebGL Box3 callbacks overwrite those numbers until recenter. */
    private var roomDimensionsLockedByTapeCalibration: Boolean = false
    /** True when dimensions came from the shared iOS/Android room_dims_v7 measurement path. */
    private var roomDimensionsFromRoomDimsV7: Boolean = false
    /** True once at least one `onBoxMetricsMeasured` or `onDimensionsMeasured` callback arrives from WebGL. */
    private var roomDimensionsReceivedFromWebGL: Boolean = false
    /** True while we are waiting for WebGL to report box3 dimensions (ruler tap before WebGL ready). */
    private var isMeasuringRoomDimensions: Boolean = false
    /**
     * W×H copied from intent + room_meta right after load (streaming AABB / saved metadata).
     * Used when WebGL Box3 reports a degenerate footprint for this viewer rotation but pipeline dims are sane.
     */
    private var openSnapshotRoomWidth: Float = 4f
    private var openSnapshotRoomHeight: Float = 3f
    private var openSnapshotRoomDepth: Float = 4.5f

    // Brain (SmartyPants) overlay: show progress in same Activity so room stays visible
    private var brainOverlayVisible = false
    private var furnitureFitManager: FurnitureFitManager? = null
    private var brainModelWarmupJob: Deferred<FurnitureFitManager?>? = null
    private var cameraProvider: ProcessCameraProvider? = null
    /** Brain flow: ARCore camera when ARCore is supported (metric overlay sizing). */
    private var brainArController: FurnitureFitArCameraController? = null
    /** [setContentView] root — used to insert/remove AR [GLSurfaceView] for brain mode. */
    private lateinit var sharpRoomContentRoot: FrameLayout
    // Brain overlay calibration pill (bottom overlay).
    private var brainCalibrationPillContainer: View? = null
    private var brainCalibrationPillLine1: TextView? = null
    private var brainCalibrationPillLine2: TextView? = null
    private var placementIntelligenceCard: View? = null
    private var placementIntelligenceExpandedPanel: LinearLayout? = null
    private var placementIntelligenceToggleRing: GradientDrawable? = null
    private var placementIntelligenceStatusView: TextView? = null
    private var placementIntelligenceBodyView: TextView? = null
    private var roomDimensionsHintView: TextView? = null
    private var pinchHintExplanationView: TextView? = null
    private var brainHintExplanationView: TextView? = null
    private var snapshotHintExplanationView: TextView? = null
    private val gestureHintHideHandler = Handler(Looper.getMainLooper())
    private val hidePinchHintRunnable = Runnable { pinchHintExplanationView?.visibility = View.GONE }
    private val hideBrainHintRunnable = Runnable { brainHintExplanationView?.visibility = View.GONE }
    private val hideSnapshotHintRunnable = Runnable { snapshotHintExplanationView?.visibility = View.GONE }
    private val hideRoomDimensionsHintRunnable = Runnable { roomDimensionsHintView?.visibility = View.GONE }
    private var isPlacementIntelligenceExpanded = false
    private var lastBrainOverlayScaleLogMs: Long = 0L
    private var lastBrainArBridgeLogMs: Long = 0L
    private var roomPlacementModel: RoomModel? = null
    private var latestFitCheckResult: FitCheckResult? = null
    private var latestCornerPlacementSuggestions: List<CornerPlacementSuggestion> = emptyList()
    private var latestAestheticScore: AestheticScore? = null
    private var latestEstimatedFurnitureDepthMeters: Float? = null
    private var segmentedFurnitureMeanSrgb: Vec3f? = null
    private var latestBrainPrimaryDetection: DetectionResult? = null
    private var latestBrainDetections: List<DetectionResult> = emptyList()
    private var latestBrainMask: Bitmap? = null
    private var latestBrainInputSize: Int = 640
    private var latestBrainOverlayScale: Float = 1f
    private var brainSegmentationMode: BrainSegmentationMode = BrainSegmentationMode.IDENTIFY_ONLY
    /** Tapped object instances (bbox snapshots); segmentation matches by class + IoU, not class id alone. */
    private val selectedBrainPins = mutableListOf<DetectionResult>()
    private var showIdentifyLivePreview: Boolean = true
    private var showFullVideoWithIdentifications: Boolean = false
    private var cameraPreviewUseCase: Preview? = null
    private val cameraExecutor: ExecutorService = Executors.newSingleThreadExecutor()
    /** True while one frame is in inference; drop new frames so overlay shows current view when camera moves. */
    private val isBrainInferenceRunning = AtomicBoolean(false)
    /** Monotonic generation for live brain runs so old callbacks cannot repaint after stop/restart. */
    private val brainSessionGeneration = AtomicInteger(0)
    /**
     * False as soon as brain segmentation is torn down. Inference callbacks may still be scheduled;
     * they must bail out so they do not repopulate height / the calibration pill after stop.
     */
    @Volatile
    private var brainSegmentationAcceptingUpdates: Boolean = false
    /** True after the first segmentation result arrives for the current brain session (CameraX or ARCore). */
    @Volatile
    private var brainFirstResultReceived: Boolean = false
    /** Once true, skip the full-screen "Detecting furniture…" progress on later brain taps this activity session. */
    private var brainSegmentationCompletedOnceThisSession: Boolean = false
    /** Used so we can fall back from ARCore brain path to CameraX if no result arrives. */
    private var disableArBrainThisSession: Boolean = false
    /** True when the current brain session was explicitly started from the AR button. */
    private var brainArAssistRequested: Boolean = false
    /** Preserves the requested mode across a camera permission prompt. */
    private var pendingBrainStartArAssist: Boolean = false
    private val brainTimeoutHandler = android.os.Handler(android.os.Looper.getMainLooper())
    private var brainTimeoutRunnable: Runnable? = null
    /** Progress bar inside the brain progress overlay (used only for simple animated feedback). */
    private var brainProgressBar: ProgressBar? = null
    private var brainProgressLabel: TextView? = null
    /** Status bar inset top (set from window insets) so arrow overlay can sit below top bar in portrait and landscape. */
    private var statusBarInsetTop = 0

    /** Debounced write of Spark/Box3 dimensions to [room_meta.json] (list screen reads same file). */
    private val sparkBoxPersistHandler = Handler(Looper.getMainLooper())
    private var sparkBoxPersistRunnable: Runnable? = null
    /** True when this viewer is showing a freshly-generated SHARP room that hasn't been saved with a name yet. */
    private var isTempSharpRoom: Boolean = false
    /** Launched from [SinglePhotoRoomActivity]; back returns to Create 3D Room (not WebView history). */
    private var openedFromSinglePhotoRoom: Boolean = false
    /** Set to true once the user explicitly saves the room from this viewer. */
    private var hasSavedRoom: Boolean = false
    private var internalPlyFile: File? = null
    private var persistedSplatLoadHint: SplatLoadHint? = null
    private var sharpLoadStartMs: Long = 0L
    private val cameraPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        DebugLogger.d(TAG, "Brain: camera permission result isGranted=$isGranted")
        if (isGranted) {
            resetBrainSessionUiState()
            showBrainProgressOverlayIfNeeded()
            startBrainDetection(pendingBrainStartArAssist)
        } else {
            pendingBrainStartArAssist = false
            DebugLogger.d(TAG, "Brain: camera permission denied")
            Toast.makeText(this, getString(R.string.camera_permission_required), Toast.LENGTH_LONG).show()
        }
    }

    @SuppressLint("SetJavaScriptEnabled", "ClickableViewAccessibility")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        sharpLoadStartMs = SystemClock.elapsedRealtime()
        showFullVideoWithIdentifications = FurnitureFitManager.isFullVideoWithIdentificationsEnabled(this)

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

        plyPath = intent.getStringExtra(EXTRA_PLY_PATH)
        roomFolder = intent.getStringExtra(EXTRA_ROOM_FOLDER)
        allowSave = intent.getBooleanExtra(EXTRA_ALLOW_SAVE, true)
        isTempSharpRoom = intent.getBooleanExtra(EXTRA_IS_TEMP_SHARP_ROOM, false)
        openedFromSinglePhotoRoom = intent.getBooleanExtra(EXTRA_OPENED_FROM_SINGLE_PHOTO_ROOM, false)

        // Load saved dimensions and orientation from intent (if available)
        var savedWidth = intent.getFloatExtra(EXTRA_ROOM_WIDTH, 0f)
        var savedHeight = intent.getFloatExtra(EXTRA_ROOM_HEIGHT, 0f)
        var savedDepth = intent.getFloatExtra(EXTRA_ROOM_DEPTH, 4.5f)
        roomDepth = savedDepth
        roomCenterX = intent.getFloatExtra(EXTRA_ROOM_CENTER_X, 0f)
        roomCenterY = intent.getFloatExtra(EXTRA_ROOM_CENTER_Y, 0f)
        roomCenterZ = intent.getFloatExtra(EXTRA_ROOM_CENTER_Z, 0f)
        var rawOrientation = intent.getStringExtra("photo_orientation")?.trim()?.lowercase()
        photoWideAngle = intent.getBooleanExtra(EXTRA_PHOTO_WIDE_ANGLE, false)

        // Single disk source: room_meta.json (or legacy metadata.txt via RoomFolderMetadata). Same parser as home list.
        roomFolder?.let { folderPath ->
            val disk = RoomFolderMetadata.readFromFolder(File(folderPath))
            if (disk != null) {
                disk.roomWidth?.takeIf { it > 0f }?.let { savedWidth = it }
                disk.roomHeight?.takeIf { it > 0f }?.let { savedHeight = it }
                disk.roomDepth?.takeIf { it > 0f }?.let { roomDepth = it }
                disk.roomCenterX?.let { roomCenterX = it }
                disk.roomCenterY?.let { roomCenterY = it }
                disk.roomCenterZ?.let { roomCenterZ = it }
                rawOrientation = disk.normalizedOrientation()
                photoWideAngle = disk.photoWideAngle
                disk.arDisplayScale?.takeIf { it > 0f }?.let { arDisplayScale = it }
                roomDimensionsFromRoomDimsV7 = disk.roomDimsApproach?.startsWith("room_dims_v7") == true
                DebugLogger.d(
                    TAG,
                    "RoomFolderMetadata: ${savedWidth}x${savedHeight}x${roomDepth} orientation=${disk.normalizedOrientation()} wide=$photoWideAngle arDisplayScale=$arDisplayScale roomDimsApproach=${disk.roomDimsApproach ?: "none"}"
                )
            }
        }

        photoOrientation = if (rawOrientation == "landscape") "landscape" else "portrait"

        // Lock orientation based on room's photo orientation (no auto-rotate)
        requestedOrientation = if (photoOrientation == "landscape") {
            ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE
        } else {
            ActivityInfo.SCREEN_ORIENTATION_PORTRAIT
        }

        // Use saved dimensions if provided, otherwise use defaults
        if (savedWidth > 0f && savedHeight > 0f) {
            roomWidth = savedWidth
            roomHeight = savedHeight
            hasSavedDimensions = true
        } else {
            roomWidth = 4.0f
            roomHeight = 3.0f
            hasSavedDimensions = false
        }
        val sanitizedOpen = SharpRoomDimensionSanitizer.sanitizeMeters(roomWidth, roomHeight, roomDepth)
        roomWidth = sanitizedOpen.first
        roomHeight = sanitizedOpen.second
        roomDepth = sanitizedOpen.third
        openSnapshotRoomWidth = roomWidth
        openSnapshotRoomHeight = roomHeight
        openSnapshotRoomDepth = roomDepth

        DebugLogger.d(
            TAG,
            "Opening SharpRoomActivity with PLY: $plyPath, dims: " +
                if (hasSavedDimensions) {
                    "${roomWidth}x${roomHeight}x${roomDepth}"
                } else {
                    "deferred_async"
                } +
                ", hasSaved: $hasSavedDimensions, photoOrientation: $photoOrientation, photoWideAngle: $photoWideAngle"
        )
        DebugLogger.d(
            TAG,
            if (hasSavedDimensions) {
                "SharpRoom intent roomWidth=$roomWidth roomHeight=$roomHeight roomDepth=$roomDepth isPortrait=${photoOrientation != "landscape"} wideAngle=$photoWideAngle"
            } else {
                "SharpRoom intent roomDims=deferred_async isPortrait=${photoOrientation != "landscape"} wideAngle=$photoWideAngle"
            }
        )
        val isPortraitReceived = photoOrientation != "landscape"
        DebugLogger.d(
            TAG,
            if (hasSavedDimensions) {
                "VIEWER_RECEIVED isPortrait=$isPortraitReceived roomWidth=$roomWidth roomHeight=$roomHeight roomDepth=$roomDepth path=$roomFolder"
            } else {
                "VIEWER_RECEIVED isPortrait=$isPortraitReceived roomDims=deferred_async path=$roomFolder"
            }
        )
        DebugLogger.i(
            "SHARP_ROOM_MEAS",
            if (hasSavedDimensions) {
                "[viewer_open] raw W×H×D=$roomWidth×$roomHeight×$roomDepth " +
                    "center=($roomCenterX,$roomCenterY,$roomCenterZ) arDisplayScale=$arDisplayScale " +
                    "eff_front_wall=${effRoomWidth()}×${effRoomHeight()} hasSavedMeta=$hasSavedDimensions folder=$roomFolder"
            } else {
                "[viewer_open] dims=deferred_async " +
                    "center=($roomCenterX,$roomCenterY,$roomCenterZ) arDisplayScale=$arDisplayScale " +
                    "hasSavedMeta=$hasSavedDimensions folder=$roomFolder"
            },
        )
        logSharpLoadTiming(
            stage = "metadata_ready",
            detail = "savedDims=$hasSavedDimensions orientation=$photoOrientation wide=$photoWideAngle",
        )

        if (plyPath == null) {
            Toast.makeText(this, getString(R.string.sharp_room_no_ply), Toast.LENGTH_SHORT).show()
            finish()
            return
        }
        persistedSplatLoadHint = loadPersistedSplatLoadHint(File(plyPath!!))
        val rootLayout = FrameLayout(this)
        rootLayout.setBackgroundColor(Color.parseColor("#808080"))

        // Copy PLY file to internal files dir for WebViewAssetLoader
        val plyFile = File(plyPath!!)
        val internalPlyDir = File(filesDir, "webview_assets")
        internalPlyDir.mkdirs()
        val internalPlyFile = File(internalPlyDir, "room.ply")
        this.internalPlyFile = internalPlyFile
        if (plyFile.exists()) {
            plyFile.copyTo(internalPlyFile, overwrite = true)
            DebugLogger.d(TAG, "Copied PLY to internal storage: ${internalPlyFile.absolutePath}")
            logSharpLoadTiming("ply_copied", "bytes=${plyFile.length()}")
        }

        // WebViewAssetLoader serves files from internal storage via https:// URL
        // This allows SparkJS fetch() to work properly
        val assetLoader = WebViewAssetLoader.Builder()
            .addPathHandler("/assets/", WebViewAssetLoader.AssetsPathHandler(this))
            .addPathHandler("/files/", WebViewAssetLoader.InternalStoragePathHandler(this, internalPlyDir))
            .build()

        // WebView for 3D rendering
        webView = WebView(this).apply {
            settings.javaScriptEnabled = true
            settings.domStorageEnabled = true
            settings.allowFileAccess = true
            settings.allowContentAccess = true
            settings.mixedContentMode = WebSettings.MIXED_CONTENT_ALWAYS_ALLOW
            settings.mediaPlaybackRequiresUserGesture = false
            setBackgroundColor(Color.TRANSPARENT)
            configureSharpRoomGpuWebView(this)

            webViewClient = object : WebViewClient() {
                override fun onPageFinished(view: WebView?, url: String?) {
                    super.onPageFinished(view, url)
                    DebugLogger.d(TAG, "WebView page loaded")
                    logSharpLoadTiming("webview_page_finished")
                    // Hide loading after a delay for splat rendering
                    postDelayed({
                        loadingOverlay.visibility = View.GONE
                    }, 2000)
                }

                override fun onReceivedError(view: WebView?, request: WebResourceRequest?, error: WebResourceError?) {
                    // Don't log favicon as error (we intercept it; this is a fallback if something else fails)
                    if (request?.url?.toString()?.contains("favicon") == true) return
                    DebugLogger.eDebugMode(TAG, "WebView error: ${error?.description}")
                }

                // Use WebViewAssetLoader to serve files
                override fun shouldInterceptRequest(view: WebView?, request: WebResourceRequest?): WebResourceResponse? {
                    val url = request?.url ?: return null
                    val urlString = url.toString()
                    // Suppress favicon request so we don't get net::ERR_NAME_NOT_RESOLVED (matches iOS: no favicon error)
                    if (urlString.endsWith("/favicon.ico") || urlString.contains("favicon.ico")) {
                        return WebResourceResponse("image/png", null, ByteArrayInputStream(ByteArray(0)))
                    }
                    DebugLogger.d(TAG, "shouldInterceptRequest: $url")
                    return assetLoader.shouldInterceptRequest(url)
                }
            }

            // WebGL console → logcat only when Settings → Debug Mode is ON (DebugLogger)
            webChromeClient = object : WebChromeClient() {
                override fun onConsoleMessage(message: ConsoleMessage?): Boolean {
                    message?.let { m ->
                        DebugLogger.d(TAG, "JSConsole: ${m.message()} -- ${m.sourceId()}:${m.lineNumber()}")
                    }
                    return true
                }
            }

            // Add JavaScript interface for communication
            addJavascriptInterface(WebAppInterface(), "Android")
        }
        rootLayout.addView(webView, FrameLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.MATCH_PARENT
        ))
        brainCameraPreviewView = PreviewView(this).apply {
            layoutParams = FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT,
            )
            implementationMode = PreviewView.ImplementationMode.COMPATIBLE
            scaleType = PreviewView.ScaleType.FIT_CENTER
            visibility = View.GONE
        }
        rootLayout.addView(brainCameraPreviewView, FrameLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.MATCH_PARENT
        ))

        // No gesture overlay - let WebView's OrbitControls handle all gestures
        // (rotation, zoom, pan) directly like iOS

        // Top bar
        val topBar = createTopBar()
        sharpRoomTopBar = topBar
        rootLayout.addView(topBar, FrameLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT
        ).apply { gravity = Gravity.TOP })

        val topHelperOverlay = createTopHelperOverlay()
        rootLayout.addView(topHelperOverlay, FrameLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.MATCH_PARENT
        ))

        roomDimensionsHintView = buildRoomDimensionsHintView()
        rootLayout.addView(
            roomDimensionsHintView,
            FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT,
                ViewGroup.LayoutParams.WRAP_CONTENT,
            ).apply { gravity = Gravity.TOP or Gravity.CENTER_HORIZONTAL },
        )

        // Bottom controls
        val bottomControls = createBottomControls()
        rootLayout.addView(bottomControls, FrameLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT
        ).apply { gravity = Gravity.BOTTOM })

        // Camera pan arrows (not in ⋮ menu)
        val cameraArrowOverlay = createCameraArrowOverlay()
        rootLayout.addView(cameraArrowOverlay, FrameLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.MATCH_PARENT
        ).apply { gravity = Gravity.TOP or Gravity.START })

        // Loading overlay
        loadingOverlay = createLoadingOverlay()
        rootLayout.addView(loadingOverlay)

        // Brain progress overlay (on top; room stays visible underneath)
        brainProgressOverlay = createBrainProgressOverlay()
        brainProgressOverlay.visibility = View.GONE
        brainProgressOverlay.elevation = 20f
        rootLayout.addView(brainProgressOverlay)

        // Brain detection overlay: live segmentation on top of room; tap green brain button to stop.
        brainDetectionOverlay = FrameLayout(this).apply {
            visibility = View.GONE
            elevation = 21f
            layoutParams = FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT
            )
        }
        brainDetectionOverlayView = FurnitureFitOverlayView(this).apply {
            layoutParams = FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT
            )
            onTouchOutsideFurniture = { ev -> webView.dispatchTouchEvent(ev) }
            onDetectionTapped = { detection -> handleBrainDetectionTapped(detection) }
        }
        brainDetectionOverlay.addView(brainDetectionOverlayView)
        rootLayout.addView(brainDetectionOverlay)

        setContentView(rootLayout)
        sharpRoomContentRoot = rootLayout
        refreshRoomDimensionsDisplay()
        rootLayout.post { restartTransientGestureHints() }
        reloadPlacementRoomModel()
        prewarmBrainSegmentationIfNeeded()

        // Apply status bar insets; pan overlay sits below top bar
        ViewCompat.setOnApplyWindowInsetsListener(rootLayout) { _, insets ->
            val statusBar = insets.getInsets(WindowInsetsCompat.Type.statusBars())
            statusBarInsetTop = statusBar.top
            topBar.setPadding(
                topBar.paddingLeft,
                statusBarInsetTop,
                topBar.paddingRight,
                topBar.paddingBottom
            )
            updateCameraArrowOverlayTop(topBar, cameraArrowOverlay)
            updateTopHelperOverlayTop(topBar, topHelperOverlay)
            updateRoomDimensionsHintPosition()
            cameraArrowOverlay.post { updateCameraArrowOverlayTop(topBar, cameraArrowOverlay) }
            topHelperOverlay.post { updateTopHelperOverlayTop(topBar, topHelperOverlay) }
            cameraArrowOverlay.post { updateRoomDimensionsHintPosition() }
            topBar.viewTreeObserver.addOnGlobalLayoutListener(object : ViewTreeObserver.OnGlobalLayoutListener {
                override fun onGlobalLayout() {
                    updateCameraArrowOverlayTop(topBar, cameraArrowOverlay)
                    updateTopHelperOverlayTop(topBar, topHelperOverlay)
                    updateRoomDimensionsHintPosition()
                }
            })
            insets
        }
        ViewCompat.requestApplyInsets(rootLayout)

        // Load the WebGL viewer
        loadWebGLViewer()
        scheduleRoomDimsV7BackfillIfNeeded()
    }

    private fun dpToPx(dp: Int): Int {
        return (dp * resources.displayMetrics.density).toInt()
    }

    private fun effRoomWidth(): Float = roomWidth * arDisplayScale
    private fun effRoomHeight(): Float = roomHeight * arDisplayScale
    private fun effRoomDepth(): Float = roomDepth * arDisplayScale
    private fun effRoomCenterX(): Float = roomCenterX * arDisplayScale
    private fun effRoomCenterY(): Float = roomCenterY * arDisplayScale
    private fun effRoomCenterZ(): Float = roomCenterZ * arDisplayScale
    private fun hasPlausibleOpenSnapshotRoomDims(): Boolean =
        hasSavedDimensions &&
            if (roomDimensionsFromRoomDimsV7) {
                openSnapshotRoomWidth > 0.05f && openSnapshotRoomHeight > 0.05f && openSnapshotRoomDepth > 0.05f
            } else {
                openSnapshotRoomWidth >= 2f && openSnapshotRoomHeight >= 2f && openSnapshotRoomDepth >= 1f
            }

    private fun scheduleRoomDimsV7BackfillIfNeeded() {
        if (roomDimensionsFromRoomDimsV7) return
        val folder = roomFolder?.let { File(it) }?.takeIf { it.isDirectory } ?: return
        val plyFile = plyPath?.let { File(it) }?.takeIf { it.isFile } ?: return
        lifecycleScope.launch {
            val measured = withContext(Dispatchers.Default) {
                val imageSize = readReferenceImageSize(folder)
                SharpRoomDimensionsV7.measureBest(
                    plyFile = plyFile,
                    sourceImageWidthPx = imageSize?.first ?: 0,
                    sourceImageHeightPx = imageSize?.second ?: 0,
                    cameraExifFile = File(folder, "camera_exif.json").takeIf { it.isFile },
                )
            }
            if (measured == null) {
                LogUtil.w(
                    "SHARP_ROOM_MEAS",
                    "[ROOM_DIMS_APP] viewer_backfill room_dims_v7 unavailable; keeping current dims $roomWidth×$roomHeight×$roomDepth folder=${folder.absolutePath}",
                )
                return@launch
            }

            val sanitized = SharpRoomDimensionSanitizer.sanitizeMeters(
                measured.width,
                measured.height,
                measured.depth,
            )
            if (roomDimensionsLockedByTapeCalibration) {
                LogUtil.i(
                    "SHARP_ROOM_MEAS",
                    "[ROOM_DIMS_APP] viewer_backfill skipped; tape calibration lock keeps $roomWidth×$roomHeight×$roomDepth",
                )
                return@launch
            }

            roomWidth = sanitized.first
            roomHeight = sanitized.second
            roomDepth = sanitized.third
            hasSavedDimensions = true
            roomDimensionsFromRoomDimsV7 = true
            openSnapshotRoomWidth = roomWidth
            openSnapshotRoomHeight = roomHeight
            openSnapshotRoomDepth = roomDepth
            isMeasuringRoomDimensions = false
            refreshRoomDimensionsDisplay()
            reloadPlacementRoomModel()

            withContext(Dispatchers.IO) {
                persistRoomDimsV7Backfill(folder, measured, sanitized)
            }
            LogUtil.i(
                "SHARP_ROOM_MEAS",
                "[ROOM_DIMS_APP] viewer_backfill SOURCE=ROOM_DIMS_V7 W=$roomWidth H=$roomHeight D=$roomDepth " +
                    "approach=${measured.approach} shot=${measured.shotType} folder=${folder.absolutePath}",
            )
        }
    }

    private fun readReferenceImageSize(folder: File): Pair<Int, Int>? {
        val imageFile = listOf("thumbnail.png", "front_wall.png")
            .map { File(folder, it) }
            .firstOrNull { it.isFile } ?: return null
        val options = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeFile(imageFile.absolutePath, options)
        val width = options.outWidth
        val height = options.outHeight
        return if (width > 0 && height > 0) width to height else null
    }

    private fun persistRoomDimsV7Backfill(
        folder: File,
        measured: SharpRoomDimensionsV7Result,
        sanitizedDimensions: Triple<Float, Float, Float>,
    ) {
        val previous = RoomFolderMetadata.readFromFolder(folder)
        val base = previous ?: RoomFolderMetadata.Snapshot(
            createdAt = folder.lastModified(),
            type = "sharp",
            photoOrientation = if (photoOrientation == "landscape") "landscape" else "portrait",
            photoWideAngle = photoWideAngle,
            previewOnly = isTempSharpRoom,
        )
        val next = base.copy(
            roomWidth = sanitizedDimensions.first,
            roomHeight = sanitizedDimensions.second,
            roomDepth = sanitizedDimensions.third,
            roomDimsApproach = measured.approach,
            roomSceneWidth = measured.sceneWidth,
            roomSceneHeight = measured.sceneHeight,
            roomSceneDepth = measured.sceneDepth,
        )
        RoomFolderMetadata.writeToFolder(
            folder,
            RoomFolderMetadata.snapshotPreservingYoloFields(folder, next),
        )

        val metadataFile = File(folder, "metadata.txt")
        val lines = linkedMapOf<String, String>()
        if (metadataFile.isFile) {
            metadataFile.readLines().forEach { line ->
                val idx = line.indexOf('=')
                if (idx > 0) lines[line.substring(0, idx).trim()] = line.substring(idx + 1).trim()
            }
        }
        lines["roomWidth"] = sanitizedDimensions.first.toString()
        lines["roomHeight"] = sanitizedDimensions.second.toString()
        lines["roomDepth"] = sanitizedDimensions.third.toString()
        lines["roomDimsApproach"] = measured.approach
        lines["roomSceneWidth"] = measured.sceneWidth.toString()
        lines["roomSceneHeight"] = measured.sceneHeight.toString()
        lines["roomSceneDepth"] = measured.sceneDepth.toString()
        metadataFile.writeText(lines.entries.joinToString(separator = "\n", postfix = "\n") { (key, value) -> "$key=$value" })
    }

    private fun configureSharpRoomGpuWebView(target: WebView) {
        target.setLayerType(View.LAYER_TYPE_HARDWARE, null)
        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.M) {
            target.settings.offscreenPreRaster = true
        }
        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.O) {
            target.setRendererPriorityPolicy(WebView.RENDERER_PRIORITY_IMPORTANT, true)
        }

        val hasVulkan = packageManager.hasSystemFeature(PackageManager.FEATURE_VULKAN_HARDWARE_VERSION)
        val webViewPackage = if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.O) {
            WebView.getCurrentWebViewPackage()?.packageName ?: "unknown"
        } else {
            "unknown"
        }
        LogUtil.i(
            TAG,
            "SharpRoom GPU WebView configured: hardwareLayer=true, optionalVulkanFeature=$hasVulkan, webViewPackage=$webViewPackage"
        )
    }

    private fun updateCameraArrowOverlayTop(topBar: View, arrowOverlay: View) {
        val top = statusBarInsetTop + topBar.height
        arrowOverlay.setPadding(0, top, 0, 0)
    }

    private fun updateTopHelperOverlayTop(topBar: View, helperOverlay: View) {
        helperOverlay.setPadding(0, statusBarInsetTop + topBar.height, 0, 0)
    }

    private fun buildGestureHintBubble(): TextView {
        return TextView(this).apply {
            visibility = View.GONE
            setTextColor(Color.WHITE)
            textSize = 11f
            maxWidth = dpToPx(220)
            gravity = Gravity.CENTER
            setPadding(dpToPx(8), dpToPx(8), dpToPx(8), dpToPx(8))
            background = GradientDrawable().apply {
                cornerRadius = dpToPx(8).toFloat()
                setColor(Color.argb((255 * 0.78f).toInt(), 0, 0, 0))
            }
        }
    }

    private fun buildHintIconButton(iconRes: Int, onClick: () -> Unit): AppCompatImageButton {
        val hintSize = dpToPx(40)
        val hintBg = GradientDrawable().apply {
            shape = GradientDrawable.OVAL
            setColor(Color.argb(128, 0, 0, 0))
        }
        return AppCompatImageButton(this).apply {
            setImageResource(iconRes)
            ImageViewCompat.setImageTintList(this, ColorStateList.valueOf(Color.WHITE))
            background = hintBg
            scaleType = ImageView.ScaleType.CENTER_INSIDE
            layoutParams = LinearLayout.LayoutParams(hintSize, hintSize).apply {
                gravity = Gravity.CENTER_HORIZONTAL
            }
            setOnClickListener { onClick() }
        }
    }

    private fun buildToolbarIconButton(
        iconRes: Int,
        contentDescriptionText: String,
        onClick: () -> Unit,
    ): AppCompatImageButton {
        return AppCompatImageButton(this).apply {
            setImageResource(iconRes)
            ImageViewCompat.setImageTintList(this, ColorStateList.valueOf(Color.WHITE))
            val typedArray = theme.obtainStyledAttributes(intArrayOf(android.R.attr.selectableItemBackgroundBorderless))
            val ripple = typedArray.getDrawable(0)
            typedArray.recycle()
            background = ripple
            scaleType = ImageView.ScaleType.CENTER_INSIDE
            contentDescription = contentDescriptionText
            setOnClickListener { onClick() }
        }
    }

    private fun buildCircularToolbarIconButton(
        iconRes: Int,
        contentDescriptionText: String,
        backgroundColor: String = "#3A3A3C",
        onClick: () -> Unit,
    ): AppCompatImageButton {
        return AppCompatImageButton(this).apply {
            setImageResource(iconRes)
            ImageViewCompat.setImageTintList(this, ColorStateList.valueOf(Color.WHITE))
            background = GradientDrawable().apply {
                shape = GradientDrawable.OVAL
                setColor(Color.parseColor(backgroundColor))
            }
            scaleType = ImageView.ScaleType.CENTER_INSIDE
            contentDescription = contentDescriptionText
            setPadding(dpToPx(8), dpToPx(8), dpToPx(8), dpToPx(8))
            setOnClickListener { onClick() }
        }
    }

    private fun buildRoomDimensionsHintView(): TextView {
        return TextView(this).apply {
            visibility = View.GONE
            setTextColor(Color.WHITE)
            textSize = 11f
            maxWidth = dpToPx(220)
            gravity = Gravity.CENTER
            setPadding(dpToPx(8), dpToPx(8), dpToPx(8), dpToPx(8))
            background = GradientDrawable().apply {
                cornerRadius = dpToPx(8).toFloat()
                setColor(Color.argb((255 * 0.78f).toInt(), 0, 0, 0))
            }
            elevation = dpToPx(4).toFloat()
        }
    }

    private fun updateRoomDimensionsHintPosition() {
        val hint = roomDimensionsHintView ?: return
        if (!::sharpRoomTopBar.isInitialized) return
        val lp = hint.layoutParams as FrameLayout.LayoutParams
        lp.topMargin = sharpRoomTopBar.bottom + dpToPx(12)
        hint.layoutParams = lp
    }

    private fun updateRoomDimensionsHintText() {
        val w = effRoomWidth()
        val h = effRoomHeight()
        val d = effRoomDepth()
        roomDimensionsHintView?.text = if (w > 0.05f && h > 0.05f && d > 0.05f) {
            String.format(
                Locale.US,
                "W × H × D\n%.2f × %.2f × %.2f m",
                w, h, d,
            )
        } else if (w > 0.05f && h > 0.05f) {
            String.format(Locale.US, "W × H\n%.2f × %.2f m", w, h)
        } else {
            getString(R.string.sharp_room_dimensions_unavailable)
        }
    }

    private fun refreshRoomDimensionsDisplay() {
        updateRoomDimensionsHintText()
        val hint = roomDimensionsHintView ?: return
        if (effRoomWidth() > 0.05f && effRoomHeight() > 0.05f) {
            hint.visibility = View.VISIBLE
            updateRoomDimensionsHintPosition()
            gestureHintHideHandler.removeCallbacks(hideRoomDimensionsHintRunnable)
            gestureHintHideHandler.postDelayed(hideRoomDimensionsHintRunnable, 3000L)
        }
    }

    private fun onRoomRulerTapped() {
        val hint = roomDimensionsHintView ?: return
        if (hint.visibility == View.VISIBLE) {
            hint.visibility = View.GONE
            gestureHintHideHandler.removeCallbacks(hideRoomDimensionsHintRunnable)
            return
        }
        if (roomDimensionsReceivedFromWebGL || hasSavedDimensions) {
            DebugLogger.d(TAG, "[ROOM_DIMS][RULER] USING_EXISTING webgl=$roomDimensionsReceivedFromWebGL saved=$hasSavedDimensions")
            updateRoomDimensionsHintText()
            hint.visibility = View.VISIBLE
            updateRoomDimensionsHintPosition()
            gestureHintHideHandler.removeCallbacks(hideRoomDimensionsHintRunnable)
            gestureHintHideHandler.postDelayed(hideRoomDimensionsHintRunnable, 3000L)
        } else {
            DebugLogger.d(TAG, "[ROOM_DIMS][RULER] FALLBACK=START_ASYNC_MEASURE")
            startAsyncRoomMeasurementFromJS()
        }
    }

    private fun startAsyncRoomMeasurementFromJS() {
        if (isMeasuringRoomDimensions) return
        isMeasuringRoomDimensions = true
        val hint = roomDimensionsHintView ?: return
        hint.text = getString(R.string.sharp_room_measuring)
        hint.visibility = View.VISIBLE
        updateRoomDimensionsHintPosition()
        webView.evaluateJavascript(
            "if(typeof sendDimensionsToAndroid==='function'){sendDimensionsToAndroid();}",
            null,
        )
        gestureHintHideHandler.postDelayed({
            if (isMeasuringRoomDimensions) {
                isMeasuringRoomDimensions = false
                if (!roomDimensionsReceivedFromWebGL) {
                    updateRoomDimensionsHintText()
                }
                gestureHintHideHandler.removeCallbacks(hideRoomDimensionsHintRunnable)
                gestureHintHideHandler.postDelayed(hideRoomDimensionsHintRunnable, 3000L)
            }
        }, 5000L)
    }

    private fun onPinchHintIconTapped() {
        val v = pinchHintExplanationView ?: return
        if (v.visibility == View.VISIBLE) {
            v.visibility = View.GONE
            gestureHintHideHandler.removeCallbacks(hidePinchHintRunnable)
        } else {
            v.visibility = View.VISIBLE
            gestureHintHideHandler.removeCallbacks(hidePinchHintRunnable)
            gestureHintHideHandler.postDelayed(hidePinchHintRunnable, 3000L)
        }
    }

    private fun onBrainHintIconTapped() {
        val v = brainHintExplanationView ?: return
        if (v.visibility == View.VISIBLE) {
            v.visibility = View.GONE
            gestureHintHideHandler.removeCallbacks(hideBrainHintRunnable)
        } else {
            v.visibility = View.VISIBLE
            gestureHintHideHandler.removeCallbacks(hideBrainHintRunnable)
            gestureHintHideHandler.postDelayed(hideBrainHintRunnable, 3000L)
        }
    }

    private fun onSnapshotHintIconTapped() {
        val v = snapshotHintExplanationView ?: return
        if (v.visibility == View.VISIBLE) {
            v.visibility = View.GONE
            gestureHintHideHandler.removeCallbacks(hideSnapshotHintRunnable)
        } else {
            v.visibility = View.VISIBLE
            gestureHintHideHandler.removeCallbacks(hideSnapshotHintRunnable)
            gestureHintHideHandler.postDelayed(hideSnapshotHintRunnable, 3000L)
        }
    }

    private fun restartPinchGestureHint() {
        pinchHintExplanationView?.let { v ->
            v.text = getString(R.string.sharp_room_pinch_gesture_hint)
            v.visibility = View.VISIBLE
            gestureHintHideHandler.removeCallbacks(hidePinchHintRunnable)
            gestureHintHideHandler.postDelayed(hidePinchHintRunnable, 3000L)
        }
    }

    private fun restartBrainGestureHint() {
        brainHintExplanationView?.let { v ->
            v.text = getString(R.string.sharp_room_brain_gesture_hint)
            v.translationY = 0f
            v.visibility = View.VISIBLE
            gestureHintHideHandler.removeCallbacks(hideBrainHintRunnable)
            gestureHintHideHandler.postDelayed(hideBrainHintRunnable, 3000L)
        }
    }

    private fun restartSnapshotGestureHint() {
        snapshotHintExplanationView?.let { v ->
            v.text = getString(R.string.sharp_room_snapshot_gesture_hint)
            v.visibility = View.VISIBLE
            gestureHintHideHandler.removeCallbacks(hideSnapshotHintRunnable)
            gestureHintHideHandler.postDelayed(hideSnapshotHintRunnable, 3000L)
        }
    }

    private fun restartTransientGestureHints() {
        restartPinchGestureHint()
        restartBrainGestureHint()
        restartSnapshotGestureHint()
    }

    private fun displayAllGestureHelpers() {
        restartTransientGestureHints()
        roomDimensionsHintView?.let { hint ->
            updateRoomDimensionsHintText()
            hint.visibility = View.VISIBLE
            updateRoomDimensionsHintPosition()
            gestureHintHideHandler.removeCallbacks(hideRoomDimensionsHintRunnable)
            gestureHintHideHandler.postDelayed(hideRoomDimensionsHintRunnable, 3000L)
        }
    }

    private fun createTopBar(): FrameLayout {
        return FrameLayout(this).apply {
            setPadding(dpToPx(16), dpToPx(48), dpToPx(16), dpToPx(12))

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

            val iconSize = dpToPx(40)

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
                setOnClickListener { onBackPressedDispatcher.onBackPressed() }
            }
            barContainer.addView(backBtn, LinearLayout.LayoutParams(iconSize, iconSize))

            barContainer.addView(
                Space(this@SharpRoomActivity),
                LinearLayout.LayoutParams(0, 1, 1f),
            )

            val fullVideoBtn = buildToolbarIconButton(
                R.drawable.ic_text_viewfinder,
                getString(R.string.settings_full_video_with_identifications),
            ) { toggleFullVideoIdentifications() }.apply {
                visibility = View.GONE
            }
            fullVideoIdentificationsButton = fullVideoBtn

            val arBtn = buildCircularToolbarIconButton(
                R.drawable.ic_square_resize,
                getString(R.string.sharp_room_ar_sizing_hint),
            ) { launchBrainMode(arAssistedRequested = true) }.apply {
                tooltipText = getString(R.string.sharp_room_ar_sizing_hint)
                visibility = View.GONE
            }
            brainArAssistButton = arBtn
            setBrainArAssistButtonActive(false)

            roomRulerButton = AppCompatImageButton(this@SharpRoomActivity).apply {
                setImageResource(R.drawable.ic_ruler)
                ImageViewCompat.setImageTintList(this, ColorStateList.valueOf(Color.WHITE))
                val typedArray = theme.obtainStyledAttributes(intArrayOf(android.R.attr.selectableItemBackgroundBorderless))
                val ripple = typedArray.getDrawable(0)
                typedArray.recycle()
                background = ripple
                scaleType = ImageView.ScaleType.CENTER_INSIDE
                contentDescription = getString(R.string.sharp_room_ruler_content_description)
                setOnClickListener { onRoomRulerTapped() }
            }

            val toolbarButtons = listOf(
                roomRulerButton,
                buildToolbarIconButton(
                    R.drawable.ic_gesture_pinch,
                    getString(R.string.sharp_room_pinch_gesture_hint),
                ) { onPinchHintIconTapped() },
                buildToolbarIconButton(
                    R.drawable.ic_gesture_tap,
                    getString(R.string.sharp_room_display_all_helpers_content_description),
                ) { displayAllGestureHelpers() },
                buildToolbarIconButton(
                    R.drawable.ic_viewfinder,
                    getString(R.string.sharp_room_menu_recenter),
                ) { recenterCamera() },
                fullVideoBtn,
                arBtn,
            )

            toolbarButtons.forEachIndexed { index, button ->
                barContainer.addView(
                    button,
                    LinearLayout.LayoutParams(dpToPx(40), dpToPx(40)).apply {
                        if (index > 0) marginStart = dpToPx(4)
                    },
                )
            }

            addView(barContainer, FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            ))

        }
    }

    private fun createTopHelperOverlay(): FrameLayout {
        pinchHintExplanationView = buildGestureHintBubble().apply {
            text = getString(R.string.sharp_room_pinch_gesture_hint)
            gravity = Gravity.END
        }
        return FrameLayout(this).apply {
            isClickable = false
            elevation = 19f
            clipChildren = false
            clipToPadding = false
            addView(
                pinchHintExplanationView,
                FrameLayout.LayoutParams(
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                ).apply {
                    gravity = Gravity.TOP or Gravity.END
                    topMargin = dpToPx(6)
                    rightMargin = dpToPx(16)
                },
            )
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

    private fun showHelpDialog() {
        AlertDialog.Builder(this)
            .setTitle("3D Room Controls")
            .setMessage("• Drag to rotate view\n• Pinch to zoom\n• Two-finger drag to pan\n• Top-left arrows nudge the view\n• ⋮ → Recenter view to reset camera")
            .setPositiveButton("OK", null)
            .show()
    }

    /** Overflow menu: matches iOS SharpRoomView ellipsis (share, save, calibrate, recenter, pan, …). */
    private fun showSharpRoomOverflowMenu(anchor: View) {
        val popup = PopupMenu(this, anchor)
        val menu = popup.menu
        menu.add(Menu.NONE, OV_SHARE, Menu.NONE, R.string.sharp_room_menu_share)
        if (allowSave) {
            menu.add(Menu.NONE, OV_SAVE, Menu.NONE, R.string.sharp_room_menu_save)
        }
        if (FurnitureFitManager.isRoomFurnitureCalibrateUiEnabled(this)) {
            menu.add(Menu.NONE, OV_CALIBRATE, Menu.NONE, R.string.sharp_room_menu_calibrate)
        }
        if (brainOverlayVisible) {
            menu.add(Menu.NONE, OV_FULL_VIDEO_IDENTIFICATIONS, Menu.NONE, R.string.settings_full_video_with_identifications)
                .apply {
                    isCheckable = true
                    isChecked = showFullVideoWithIdentifications
                }
        }
        menu.add(Menu.NONE, OV_RECENTER, Menu.NONE, R.string.sharp_room_menu_recenter)
        menu.add(Menu.NONE, OV_RESET_OVERLAY, Menu.NONE, R.string.sharp_room_menu_reset_overlay)
        menu.add(Menu.NONE, OV_HELP, Menu.NONE, R.string.sharp_room_menu_help)
        popup.setOnMenuItemClickListener { item ->
            when (item.itemId) {
                OV_SHARE -> sharePlyFile()
                OV_SAVE -> showSaveDialog()
                OV_CALIBRATE -> showRoomCalibrationDialog()
                OV_FULL_VIDEO_IDENTIFICATIONS -> toggleFullVideoIdentifications()
                OV_RECENTER -> recenterCamera()
                OV_RESET_OVERLAY -> brainDetectionOverlayView.resetTransform()
                OV_HELP -> showHelpDialog()
                else -> return@setOnMenuItemClickListener false
            }
            true
        }
        popup.show()
    }

    private fun toggleFullVideoIdentifications() {
        showFullVideoWithIdentifications = !showFullVideoWithIdentifications
        updateFullVideoToolbarButton()
        if (!showFullVideoWithIdentifications && brainSegmentationMode == BrainSegmentationMode.SEGMENT_SELECTED) {
            stopBrainSegmentationOnly()
        } else {
            if (showFullVideoWithIdentifications) {
                showIdentifyLivePreview = true
            }
            updateBrainLivePreviewVisibility()
            ensureCameraPreviewBoundForFullVideoIfNeeded()
            showBrainDetectionOverlay()
        }
        DebugLogger.d(TAG, "Full video with identifications toggled: $showFullVideoWithIdentifications")
    }

    private fun setBrainCalibrationPillVisible(visible: Boolean) {
        runOnUiThread {
            brainCalibrationPillContainer?.visibility = if (visible) View.VISIBLE else View.GONE
        }
    }

    /**
     * Manual calibration wins, then live AR (provisional → committed). GL thread already EMA-smooths
     * pinhole height; a one-shot [brainLockedFurnitureHeightMeters] must not freeze display when the user moves.
     * Locked height is only a fallback when AR tiers are null (e.g. stale provisional).
     */
    private fun effectiveBrainFurnitureHeightDisplayMeters(): Float? {
        return brainRealFurnitureHeightMeters?.takeIf { it.isFinite() && it > 0f }
            ?: brainArController?.getProvisionalHeightMeters()?.takeIf { it.isFinite() && it > 0f }
            ?: brainArController?.getLastEstimatedHeightMeters()?.takeIf { it.isFinite() && it > 0f }
            ?: latestBrainPrimaryDetection?.let { detection ->
                val input = latestBrainInputSize.takeIf { it > 0 }?.toFloat() ?: return@let null
                val roomH = effRoomHeight().takeIf { it.isFinite() && it > 0.1f } ?: return@let null
                (roomH * (detection.h / input)).coerceIn(0.05f, roomH * 0.95f)
            }
            ?: brainLockedFurnitureHeightMeters?.takeIf { it.isFinite() && it > 0f }
    }

    private fun effectiveBrainFurnitureWidthDisplayMeters(): Float? =
        brainLockedFurnitureWidthMeters?.takeIf { it.isFinite() && it > 0f }?.let { baseWidth ->
            val scale = brainCalibrationScaleFactor.takeIf { it.isFinite() && it > 0f } ?: 1f
            if (brainRealFurnitureHeightMeters != null) baseWidth * scale else baseWidth
        }
            ?: run {
                val detection = latestBrainPrimaryDetection ?: return@run null
                val displayHeight = effectiveBrainFurnitureHeightDisplayMeters() ?: return@run null
                if (detection.h <= 1e-3f) return@run null
                (displayHeight * (detection.w / detection.h)).coerceAtLeast(0.05f)
            }

    private fun brainOverlayScaleForDetection(
        det: DetectionResult?,
        modelInputSize: Int,
        targetHeightMeters: Float?,
    ): Float {
        if (!brainArAssistRequested) return 1f
        val detection = det ?: return 1f
        if (modelInputSize <= 0) return 1f
        val roomHeightMeters = effRoomHeight()
        val stableHeightMeters = targetHeightMeters?.takeIf { it.isFinite() && it > 0f }
            ?: com.furnit.android.ar.FurnitureFitStandardHeights.heightMetersForLabel(detection.label)
        if (roomHeightMeters <= 0.1f) return 1f
        val currentFraction = (detection.h / modelInputSize.toFloat()).coerceIn(0.06f, 0.92f)
        val targetFraction = (stableHeightMeters / roomHeightMeters).coerceIn(0.06f, 0.92f)
        val finalScale = (targetFraction / currentFraction).coerceIn(0.25f, 4f)
        val nowMs = SystemClock.elapsedRealtime()
        if (nowMs - lastBrainOverlayScaleLogMs >= 500L) {
            lastBrainOverlayScaleLogMs = nowMs
            val depthMeters = brainArController?.getLastMetricDistanceMeters()
            val depthSource = brainArController?.getLastMetricDistanceSource()
            val depthDiagnostic = brainArController?.getLastMetricDistanceDiagnostic()
            val provisionalH = brainArController?.getProvisionalHeightMeters()
            val committedH = brainArController?.getLastEstimatedHeightMeters()
            val lockedSnapH = brainLockedFurnitureHeightMeters
            val scaleSource = when {
                brainRealFurnitureHeightMeters != null -> "manual"
                provisionalH != null -> "provisional_ar"
                committedH != null -> "committed_ar"
                lockedSnapH != null -> "locked_fallback"
                else -> "fallback_std"
            }
            val provAge = brainArController?.getProvisionalHeightAgeMs() ?: -1L
            val commitAge = brainArController?.getCommittedHeightAgeMs() ?: -1L
            val driftPct = if (stableHeightMeters > 1e-6f && provisionalH != null) {
                String.format(
                    Locale.US,
                    "%.1f",
                    kotlin.math.abs(provisionalH - stableHeightMeters) / stableHeightMeters * 100f,
                )
            } else {
                "n/a"
            }
            LogUtil.i(
                SCALE_LOG_TAG,
                "screen=SharpRoomActivity roomH_m=${String.format(Locale.US, "%.3f", roomHeightMeters)} " +
                    "displayH_m=${String.format(Locale.US, "%.3f", stableHeightMeters)} " +
                    "bboxH_model=${String.format(Locale.US, "%.1f", detection.h)} input=$modelInputSize " +
                    "currentFrac=${String.format(Locale.US, "%.4f", currentFraction)} " +
                    "targetFrac=${String.format(Locale.US, "%.4f", targetFraction)} " +
                    "layerScale=${String.format(Locale.US, "%.4f", finalScale)} " +
                    "width_m=${effectiveBrainFurnitureWidthDisplayMeters()?.let { String.format(Locale.US, "%.3f", it) } ?: "null"} " +
                    "height_m=${effectiveBrainFurnitureHeightDisplayMeters()?.let { String.format(Locale.US, "%.3f", it) } ?: "null"} " +
                    "provisionalH_m=${provisionalH?.let { String.format(Locale.US, "%.3f", it) } ?: "null"} " +
                    "committedH_m=${committedH?.let { String.format(Locale.US, "%.3f", it) } ?: "null"} " +
                    "lockedSnap_m=${lockedSnapH?.let { String.format(Locale.US, "%.3f", it) } ?: "null"} " +
                    "source=$scaleSource drift_pct_vs_display=$driftPct " +
                    "provisionalAge_ms=$provAge committedAge_ms=$commitAge " +
                    "depth_m=${depthMeters?.let { String.format(Locale.US, "%.3f", it) } ?: "null"} " +
                    "depthSource=${depthSource ?: "none"} " +
                    "depthDiag=${depthDiagnostic ?: "none"}",
            )
        }
        return finalScale
    }

    private fun lockBrainFurnitureSizeIfNeeded(widthMeters: Float?, heightMeters: Float?) {
        if (brainLockedFurnitureWidthMeters == null) {
            brainLockedFurnitureWidthMeters = widthMeters?.takeIf { it.isFinite() && it > 0f }
        }
        if (brainLockedFurnitureHeightMeters == null) {
            brainLockedFurnitureHeightMeters = heightMeters?.takeIf { it.isFinite() && it > 0f }
        }
    }

    /** Update the brain (FurnitureFit) pill: always shows Furn/Room measurements; “Tap to calibrate” only when pref is on. */
    private fun updateBrainCalibrationPill() {
        runOnUiThread {
            val container = brainCalibrationPillContainer
            val line1 = brainCalibrationPillLine1
            val line2 = brainCalibrationPillLine2
            val calibrateUi = FurnitureFitManager.isRoomFurnitureCalibrateUiEnabled(this)
            val detectedWidth = effectiveBrainFurnitureWidthDisplayMeters()
            val detected = effectiveBrainFurnitureHeightDisplayMeters()
            if (container == null || line1 == null || line2 == null) return@runOnUiThread
            val roomH = effRoomHeight().takeIf { it.isFinite() && it > 0.05f }
            line1.text = roomH?.let { "Room: ${String.format(Locale.US, "%.2f", it)}m" } ?: "Room:"
            line1.setTextColor(0xFFFFFFFF.toInt())
            val realH = brainRealFurnitureHeightMeters
            line2.text = if (detectedWidth != null && detected != null) {
                "Furn: ${String.format(Locale.US, "%.2f", detectedWidth)}×${String.format(Locale.US, "%.2f", detected)}m"
            } else if (detected != null) {
                "Furn: H ${String.format(Locale.US, "%.2f", detected)}m"
            } else {
                getString(R.string.smartypants_tap_calibrate)
            }
            line2.setTextColor(
                when {
                    realH != null && realH > 0f -> 0xFF4CAF50.toInt()
                    detected != null -> 0xFFFFFFFF.toInt()
                    else -> 0xFFAAAAAA.toInt()
                },
            )
            if (calibrateUi && detected != null) {
                container.isClickable = true
                container.setOnClickListener { showBrainCalibrationDialog() }
            } else {
                container.setOnClickListener(null)
                container.isClickable = false
            }
            line2.visibility = View.VISIBLE
        }
    }

    private fun setPlacementIntelligenceVisible(visible: Boolean) {
        runOnUiThread {
            placementIntelligenceCard?.visibility = if (visible) View.VISIBLE else View.GONE
        }
    }

    private fun reloadPlacementRoomModel() {
        val roomFile = plyPath?.let { File(it) }?.takeIf { it.exists() }
        val roomFolderFile = roomFolder?.let { File(it) }?.takeIf { it.isDirectory }
        roomPlacementModel = RoomIntelligenceLoader.load(
            roomFile = roomFile,
            roomFolder = roomFolderFile,
            roomWidthMeters = effRoomWidth(),
            roomHeightMeters = effRoomHeight(),
            roomDepthMeters = roomDepth,
        )
        updateRoomPlacementIntelligence()
    }

    private fun derivedDetectedFurnitureDimensionsForRoomIntelligence(): RoomFurnitureDimensions? {
        val detection = latestBrainPrimaryDetection ?: return null
        val height = effectiveBrainFurnitureHeightDisplayMeters()
            ?.takeIf { it.isFinite() && it > 0.05f }
            ?: return null
        val aspect = if (detection.h > 1e-3f) {
            (detection.w / detection.h).coerceIn(0.25f, 4f)
        } else {
            0.9f
        }
        val width = (height * aspect).coerceIn(0.15f, max(0.25f, effRoomWidth() * 0.95f))
        val depth = (width * 0.72f).coerceIn(0.25f, 1.4f)
        return RoomFurnitureDimensions(widthM = width, heightM = height, depthM = depth)
    }

    private fun placementIntelligenceHasFurnitureSignal(): Boolean {
        return latestBrainPrimaryDetection != null ||
            segmentedFurnitureMeanSrgb != null ||
            derivedDetectedFurnitureDimensionsForRoomIntelligence() != null
    }

    private fun inferredRoomStyleTags(palette: SurfacePalette): List<String> {
        val tags = linkedSetOf<String>()
        listOfNotNull(palette.floor, palette.walls, palette.ceiling).forEach { layer ->
            when (layer.hint) {
                SurfacePalette.MaterialHint.WOOD -> {
                    tags += "rustic"
                    tags += "traditional"
                }
                SurfacePalette.MaterialHint.TILE -> tags += "modern"
                SurfacePalette.MaterialHint.CONCRETE -> {
                    tags += "industrial"
                    tags += "modern"
                }
                SurfacePalette.MaterialHint.CARPET -> {
                    tags += "traditional"
                    tags += "eclectic"
                }
                SurfacePalette.MaterialHint.PLASTER -> {
                    tags += "modern"
                    tags += "scandinavian"
                }
                SurfacePalette.MaterialHint.BRICK -> {
                    tags += "traditional"
                    tags += "industrial"
                }
                SurfacePalette.MaterialHint.MARBLE -> {
                    tags += "modern"
                    tags += "luxury"
                }
                SurfacePalette.MaterialHint.UNKNOWN -> Unit
            }
        }
        return if (tags.isEmpty()) listOf("modern", "minimalist") else tags.take(6)
    }

    private fun heuristicFurnitureProfileForAesthetic(
        roomModel: RoomModel,
        segmentedMeanSrgb: Vec3f?,
    ): FurnitureProfile {
        val palette = roomModel.surfacePalette
        val wall = palette.walls?.dominantColors?.firstOrNull()
        val floor = palette.floor?.dominantColors?.firstOrNull()
        val ceiling = palette.ceiling?.dominantColors?.firstOrNull()
        val primary = when {
            segmentedMeanSrgb != null -> segmentedMeanSrgb
            wall != null -> {
                Vec3f(
                    x = min(wall.x * 0.82f + 0.06f, 1f),
                    y = min(wall.y * 0.78f + 0.05f, 1f),
                    z = min(wall.z * 0.74f + 0.04f, 1f),
                )
            }
            floor != null -> {
                Vec3f(
                    x = 0.38f * 0.55f + floor.x * 0.45f,
                    y = 0.38f * 0.55f + floor.y * 0.45f,
                    z = 0.38f * 0.55f + floor.z * 0.45f,
                )
            }
            ceiling != null -> {
                Vec3f(
                    x = ceiling.x * 0.55f,
                    y = ceiling.y * 0.52f,
                    z = ceiling.z * 0.48f,
                )
            }
            else -> Vec3f(0.44f, 0.40f, 0.36f)
        }
        return FurnitureProfile(
            primaryColor = primary,
            accentColor = null,
            styleTags = listOf("modern", "minimalist", "contemporary"),
        )
    }

    private fun harmonyTypeDisplayName(type: HarmonyType): String {
        return when (type) {
            HarmonyType.ANALOGOUS -> getString(R.string.placement_harmony_analogous)
            HarmonyType.COMPLEMENTARY -> getString(R.string.placement_harmony_complementary)
            HarmonyType.TRIADIC -> getString(R.string.placement_harmony_triadic)
            HarmonyType.SPLIT_COMPLEMENTARY -> getString(R.string.placement_harmony_split_complementary)
            HarmonyType.NEUTRAL -> getString(R.string.placement_harmony_neutral)
            HarmonyType.CLASH -> getString(R.string.placement_harmony_clash)
        }
    }

    private fun updatePlacementIntelligenceCard() {
        runOnUiThread {
            val card = placementIntelligenceCard ?: return@runOnUiThread
            val statusView = placementIntelligenceStatusView ?: return@runOnUiThread
            val bodyView = placementIntelligenceBodyView ?: return@runOnUiThread
            val fit = latestFitCheckResult
            val aesthetic = latestAestheticScore
            val dimensions = derivedDetectedFurnitureDimensionsForRoomIntelligence()
            val shouldShow = brainDetectionOverlay.visibility == View.VISIBLE &&
                placementIntelligenceHasFurnitureSignal() &&
                (fit != null || aesthetic != null)
            card.visibility = if (shouldShow) View.VISIBLE else View.GONE
            if (!shouldShow) return@runOnUiThread

            statusView.text = when {
                fit == null -> getString(R.string.placement_badge_style_only)
                fit.fitsInRoom -> getString(R.string.placement_fit_count, max(fit.fitLocations.size, 1))
                else -> getString(R.string.placement_no_fit)
            }
            statusView.setTextColor(
                when {
                    fit == null -> Color.parseColor("#7FDBFF")
                    fit.fitsInRoom -> Color.parseColor("#4CAF50")
                    else -> Color.parseColor("#FF6B6B")
                },
            )

            val lines = mutableListOf<String>()
            if (dimensions == null) {
                lines += getString(R.string.placement_metric_unavailable_note)
            } else {
                lines += getString(
                    R.string.placement_detected_size,
                    dimensions.widthM.toDouble(),
                    dimensions.heightM.toDouble(),
                    dimensions.depthM.toDouble(),
                )
            }
            fit?.let {
                lines += if (it.fitsInRoom) {
                    if (it.fitLocations.isEmpty()) {
                        getString(R.string.placement_fit_no_region)
                    } else {
                        getString(R.string.placement_fit_regions, it.fitLocations.size)
                    }
                } else {
                    getString(R.string.placement_exceeds_room)
                }
                latestCornerPlacementSuggestions.firstOrNull()?.let { best ->
                    lines += getString(
                        R.string.placement_best_corner,
                        best.score.toDouble(),
                        (best.yRotationRad * 180f / Math.PI.toFloat()).toDouble(),
                    )
                }
                it.warnings.firstOrNull()?.let { warning -> lines += warning }
                    ?: if (latestEstimatedFurnitureDepthMeters != null) {
                        lines += getString(R.string.placement_depth_estimated)
                    } else {
                        Unit
                    }
            }
            aesthetic?.let {
                lines += getString(
                    R.string.placement_harmony_summary,
                    it.harmonyScore.toDouble(),
                    harmonyTypeDisplayName(it.harmonyType),
                    it.contrastScore.toDouble(),
                    it.styleCompatibilityScore.toDouble(),
                )
                it.recommendations.take(4).forEach { reco -> lines += "\u2022 $reco" }
            }
            bodyView.text = lines.joinToString("\n")
            placementIntelligenceExpandedPanel?.visibility =
                if (isPlacementIntelligenceExpanded) View.VISIBLE else View.GONE
            val ringColor = when {
                fit == null -> Color.parseColor("#00BCD4")
                fit.fitsInRoom -> Color.parseColor("#4CAF50")
                else -> Color.parseColor("#FF6B6B")
            }
            placementIntelligenceToggleRing?.setStroke(dpToPx(3), ringColor)
        }
    }

    private fun updateRoomPlacementIntelligence() {
        if (!::brainDetectionOverlay.isInitialized) {
            return
        }
        if (brainDetectionOverlay.visibility != View.VISIBLE || !placementIntelligenceHasFurnitureSignal()) {
            latestFitCheckResult = null
            latestCornerPlacementSuggestions = emptyList()
            latestEstimatedFurnitureDepthMeters = null
            latestAestheticScore = null
            updatePlacementIntelligenceCard()
            return
        }
        val roomModel = roomPlacementModel ?: run {
            updatePlacementIntelligenceCard()
            return
        }

        val dimensions = derivedDetectedFurnitureDimensionsForRoomIntelligence()
        if (dimensions != null) {
            latestEstimatedFurnitureDepthMeters = dimensions.depthM
            val fitEngine = FitCheckEngine(roomModel)
            latestFitCheckResult = fitEngine.checkFit(dimensions)
            latestCornerPlacementSuggestions = CornerPlacement(roomModel).suggestions(dimensions).take(3)
            DebugLogger.d(
                TAG,
                "Placement intelligence updated furniture=${String.format(Locale.US, "%.2f", dimensions.widthM)}x" +
                    "${String.format(Locale.US, "%.2f", dimensions.heightM)}x${String.format(Locale.US, "%.2f", dimensions.depthM)} " +
                    "fits=${latestFitCheckResult?.fitsInRoom} fitLocations=${latestFitCheckResult?.fitLocations?.size ?: 0}",
            )
        } else {
            latestEstimatedFurnitureDepthMeters = null
            latestFitCheckResult = null
            latestCornerPlacementSuggestions = emptyList()
        }

        val palette = roomModel.surfacePalette
        val roomStyleTags = inferredRoomStyleTags(palette)
        val furnitureProfile = heuristicFurnitureProfileForAesthetic(roomModel, segmentedFurnitureMeanSrgb)
        latestAestheticScore = AestheticAdvisor(palette, roomStyleTags).evaluate(furnitureProfile)
        updatePlacementIntelligenceCard()
    }

    /** Dialog for per-object furniture calibration in brain overlay. */
    private fun showBrainCalibrationDialog() {
        if (!FurnitureFitManager.isRoomFurnitureCalibrateUiEnabled(this)) return
        val detected = effectiveBrainFurnitureHeightDisplayMeters() ?: return
        val ctx = this
        val edit = EditText(ctx).apply {
            hint = getString(R.string.smartypants_real_height_hint)
            inputType = android.text.InputType.TYPE_CLASS_NUMBER or android.text.InputType.TYPE_NUMBER_FLAG_DECIMAL
            setText(String.format(Locale.US, "%.2f", detected))
            setSelection(text?.length ?: 0)
        }
        AlertDialog.Builder(ctx)
            .setTitle(getString(R.string.smartypants_calibrate_title))
            .setMessage(getString(R.string.smartypants_calibrate_message, String.format(Locale.US, "%.2f", detected)))
            .setView(edit)
            .setNegativeButton(android.R.string.cancel, null)
            .setPositiveButton(android.R.string.ok) { _, _ ->
                val raw = edit.text?.toString()?.trim() ?: return@setPositiveButton
                val real = raw.toFloatOrNull() ?: return@setPositiveButton
                if (real <= 0f) {
                    Toast.makeText(ctx, getString(R.string.smartypants_enter_positive_number), Toast.LENGTH_SHORT).show()
                    return@setPositiveButton
                }
                val factor = real / kotlin.math.abs(detected).coerceAtLeast(0.0001f)
                brainCalibrationScaleFactor = factor
                brainRealFurnitureHeightMeters = real
                // For now, reflect calibration in UI; viewer scaling can be added via JS hook later.
                updateBrainCalibrationPill()
                updateRoomPlacementIntelligence()
                Toast.makeText(ctx, getString(R.string.smartypants_room_scaled, String.format(Locale.US, "%.2f", factor)), Toast.LENGTH_SHORT).show()
            }
            .show()
    }

    /** Dialog for manual room calibration (front-wall height) – Android counterpart to iOS wall-calibration overlay. */
    private fun showRoomCalibrationDialog() {
        if (!FurnitureFitManager.isRoomFurnitureCalibrateUiEnabled(this)) return
        val ctx = this
        val edit = EditText(ctx).apply {
            hint = getString(R.string.smartypants_real_height_hint)
            inputType = android.text.InputType.TYPE_CLASS_NUMBER or android.text.InputType.TYPE_NUMBER_FLAG_DECIMAL
            setText(String.format(Locale.US, "%.2f", effRoomHeight()))
            setSelection(text?.length ?: 0)
        }
        AlertDialog.Builder(ctx)
            .setTitle(getString(R.string.smartypants_calibrate_title))
            .setMessage(getString(R.string.smartypants_calibrate_message, String.format(Locale.US, "%.2f", effRoomHeight())))
            .setView(edit)
            .setNegativeButton(android.R.string.cancel, null)
            .setPositiveButton(android.R.string.ok) { _, _ ->
                val raw = edit.text?.toString()?.trim() ?: return@setPositiveButton
                val real = raw.toFloatOrNull() ?: return@setPositiveButton
                if (real <= 0f) {
                    Toast.makeText(ctx, getString(R.string.smartypants_enter_positive_number), Toast.LENGTH_SHORT).show()
                    return@setPositiveButton
                }
                // Compute how much we need to scale the current room so that the *effective* front-wall
                // height (roomHeight * arDisplayScale) matches the user-entered real height.
                val currentEffH = effRoomHeight()
                if (currentEffH <= 0f) {
                    Toast.makeText(ctx, getString(R.string.smartypants_enter_positive_number), Toast.LENGTH_SHORT).show()
                    return@setPositiveButton
                }
                val factor = real / currentEffH
                // Scale the underlying SHARP dimensions; arDisplayScale stays the same so the new
                // effective dimensions become real numbers. Depth is scaled as well for consistency.
                roomWidth *= factor
                roomHeight *= factor
                roomDepth *= factor
                // Update title to show calibrated dimensions.
                refreshRoomDimensionsDisplay()
                DebugLogger.d(TAG, "Room calibration applied: factor=$factor newDims=${effRoomWidth()}x${effRoomHeight()} (real=$real)")
                LogUtil.i(
                    "SHARP_ROOM_MEAS",
                    "[wall_calibrate] factor=$factor real_wall_h_m=$real raw_after W×H×D=$roomWidth×$roomHeight×$roomDepth eff_front_wall=${effRoomWidth()}×${effRoomHeight()}",
                )
                // Persist calibrated dimensions so the home list and future viewer sessions match.
                roomDimensionsLockedByTapeCalibration = true
                reloadPlacementRoomModel()
                persistSparkBoxDimensionsDebounced()
            }
            .show()
    }

    private fun createBottomControls(): FrameLayout {
        return FrameLayout(this).apply {
            val horizontalPad = if (photoOrientation == "landscape") dpToPx(30) else dpToPx(16)
            setPadding(horizontalPad, 0, horizontalPad, dpToPx(20))

            val mainColumn = LinearLayout(this@SharpRoomActivity).apply {
                orientation = LinearLayout.VERTICAL
                layoutParams = FrameLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                ).apply {
                    gravity = Gravity.BOTTOM
                }
            }

            if (photoOrientation != "landscape") {
                val orientationLabel = TextView(this@SharpRoomActivity).apply {
                    text = getString(R.string.orientation_held_vertically)
                    setTextColor(Color.WHITE)
                    gravity = Gravity.CENTER
                    textSize = 11f
                    alpha = 0.85f
                    setPadding(0, 0, 0, dpToPx(4))
                }
                mainColumn.addView(
                    orientationLabel,
                    LinearLayout.LayoutParams(
                        ViewGroup.LayoutParams.MATCH_PARENT,
                        ViewGroup.LayoutParams.WRAP_CONTENT,
                    ),
                )
            }

            val bottomRow = LinearLayout(this@SharpRoomActivity).apply {
                orientation = LinearLayout.HORIZONTAL
                gravity = Gravity.BOTTOM
                layoutParams = LinearLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                )
            }

            brainHintExplanationView = buildGestureHintBubble().apply {
                text = getString(R.string.sharp_room_brain_gesture_hint)
            }
            snapshotHintExplanationView = buildGestureHintBubble().apply {
                text = getString(R.string.sharp_room_snapshot_gesture_hint)
            }

            val brainHintColumn = LinearLayout(this@SharpRoomActivity).apply {
                orientation = LinearLayout.VERTICAL
                gravity = Gravity.CENTER_HORIZONTAL
            }
            brainHintColumn.addView(
                brainHintExplanationView,
                LinearLayout.LayoutParams(
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                ).apply { bottomMargin = dpToPx(6) },
            )
            brainHintColumn.addView(buildHintIconButton(R.drawable.ic_gesture_tap) { onBrainHintIconTapped() })

            val brainSize = dpToPx(60)
            val brainBtn = AppCompatImageButton(this@SharpRoomActivity).apply {
                setImageResource(R.drawable.ic_brain)
                ImageViewCompat.setImageTintList(this, ColorStateList.valueOf(Color.WHITE))
                val bg = GradientDrawable().apply {
                    shape = GradientDrawable.OVAL
                    setColor(Color.parseColor(BRAIN_BUTTON_COLOR_IDLE))
                }
                background = bg
                scaleType = ImageView.ScaleType.CENTER_INSIDE
                setPadding(dpToPx(8), dpToPx(8), dpToPx(8), dpToPx(8))
                layoutParams = LinearLayout.LayoutParams(brainSize, brainSize).apply {
                    topMargin = dpToPx(6)
                }
                setOnClickListener {
                    val roomId = roomFolder?.let { File(it).name }
                    DebugLogger.d(TAG, "Brain click: ROOM_ID=$roomId ROOM_FOLDER=$roomFolder")
                    launchBrainMode(arAssistedRequested = false)
                }
            }
            brainModeButton = brainBtn
            val brainStack = LinearLayout(this@SharpRoomActivity).apply {
                orientation = LinearLayout.VERTICAL
                gravity = Gravity.CENTER_HORIZONTAL
            }
            brainStack.addView(brainHintColumn)
            brainStack.addView(brainBtn)
            bottomRow.addView(brainStack)

            val segmentActionBtn = TextView(this@SharpRoomActivity).apply {
                setTextColor(Color.WHITE)
                textSize = 15f
                gravity = Gravity.CENTER
                visibility = View.GONE
                setPadding(dpToPx(18), dpToPx(10), dpToPx(18), dpToPx(10))
                background = GradientDrawable().apply {
                    shape = GradientDrawable.RECTANGLE
                    cornerRadius = dpToPx(20).toFloat()
                    setColor(Color.parseColor("#E6333333"))
                }
                layoutParams = LinearLayout.LayoutParams(
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                ).apply {
                    marginStart = dpToPx(10)
                    gravity = Gravity.BOTTOM
                }
                setOnClickListener {
                    if (brainSegmentationMode == BrainSegmentationMode.SEGMENT_SELECTED) {
                        stopBrainSegmentationOnly()
                    } else {
                        activateSelectedBrainSegmentation()
                    }
                }
            }
            brainActionButton = segmentActionBtn
            bottomRow.addView(segmentActionBtn)

            bottomRow.addView(
                Space(this@SharpRoomActivity).apply {
                    layoutParams = LinearLayout.LayoutParams(0, 0, 1f)
                },
            )

            val placementExpandedPanelLocal = LinearLayout(this@SharpRoomActivity).apply {
                orientation = LinearLayout.VERTICAL
                visibility = View.GONE
                setPadding(dpToPx(16), dpToPx(12), dpToPx(16), dpToPx(12))
                background = GradientDrawable().apply {
                    shape = GradientDrawable.RECTANGLE
                    cornerRadius = dpToPx(14).toFloat()
                    setColor(Color.parseColor("#D91C1C1E"))
                }
                layoutParams = LinearLayout.LayoutParams(
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                ).apply {
                    gravity = Gravity.CENTER_HORIZONTAL
                }
            }
            val placementTitle = TextView(this@SharpRoomActivity).apply {
                text = getString(R.string.placement_title)
                setTextColor(Color.WHITE)
                textSize = 13f
                setTypeface(null, Typeface.BOLD)
            }
            val placementStatusView = TextView(this@SharpRoomActivity).apply {
                text = getString(R.string.placement_badge_style_only)
                setTextColor(Color.parseColor("#7FDBFF"))
                textSize = 12f
                setTypeface(null, Typeface.BOLD)
            }
            placementIntelligenceStatusView = placementStatusView
            val placementBodyView = TextView(this@SharpRoomActivity).apply {
                setTextColor(Color.WHITE)
                textSize = 12f
            }
            placementIntelligenceBodyView = placementBodyView
            placementExpandedPanelLocal.addView(placementTitle)
            placementExpandedPanelLocal.addView(placementStatusView)
            placementExpandedPanelLocal.addView(placementBodyView)
            placementIntelligenceExpandedPanel = placementExpandedPanelLocal

            val placementRing = GradientDrawable(
                GradientDrawable.Orientation.TL_BR,
                intArrayOf(Color.rgb(56, 56, 56), Color.rgb(31, 31, 31)),
            ).apply {
                shape = GradientDrawable.OVAL
                setStroke(dpToPx(3), Color.parseColor("#00BCD4"))
            }
            placementIntelligenceToggleRing = placementRing

            val placementToggleButton = FrameLayout(this@SharpRoomActivity).apply {
                layoutParams = LinearLayout.LayoutParams(dpToPx(46), dpToPx(46)).apply {
                    gravity = Gravity.CENTER_HORIZONTAL
                    topMargin = dpToPx(10)
                }
                background = placementRing
                isClickable = true
                setOnClickListener {
                    isPlacementIntelligenceExpanded = !isPlacementIntelligenceExpanded
                    updatePlacementIntelligenceCard()
                }
                addView(
                    ImageView(this@SharpRoomActivity).apply {
                        setImageResource(R.drawable.ic_square_split_2x2)
                        ImageViewCompat.setImageTintList(this, ColorStateList.valueOf(Color.WHITE))
                        scaleType = ImageView.ScaleType.FIT_CENTER
                        importantForAccessibility = View.IMPORTANT_FOR_ACCESSIBILITY_NO
                    },
                    FrameLayout.LayoutParams(dpToPx(22), dpToPx(22), Gravity.CENTER),
                )
            }

            val placementOuter = LinearLayout(this@SharpRoomActivity).apply {
                orientation = LinearLayout.VERTICAL
                gravity = Gravity.CENTER_HORIZONTAL
                visibility = View.GONE
                layoutParams = LinearLayout.LayoutParams(
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                ).apply { gravity = Gravity.BOTTOM }
            }
            placementOuter.addView(placementExpandedPanelLocal)
            placementOuter.addView(placementToggleButton)
            placementIntelligenceCard = placementOuter
            bottomRow.addView(placementOuter)

            bottomRow.addView(
                Space(this@SharpRoomActivity).apply {
                    layoutParams = LinearLayout.LayoutParams(0, 0, 1f)
                },
            )

            val pillContent = LinearLayout(this@SharpRoomActivity).apply {
                orientation = LinearLayout.VERTICAL
                gravity = Gravity.CENTER
                setPadding(dpToPx(24), dpToPx(12), dpToPx(24), dpToPx(12))
                background = GradientDrawable().apply {
                    shape = GradientDrawable.RECTANGLE
                    cornerRadius = dpToPx(24).toFloat()
                    setColor(Color.parseColor("#E6333333"))
                }
            }
            brainCalibrationPillLine1 = TextView(this@SharpRoomActivity).apply {
                text = "Furn:"
                setTextColor(Color.WHITE)
                textSize = 14f
                setShadowLayer(2f, 1f, 1f, Color.BLACK)
            }
            brainCalibrationPillLine2 = TextView(this@SharpRoomActivity).apply {
                text = getString(R.string.smartypants_tap_calibrate)
                setTextColor(Color.parseColor("#AAFFFFFF"))
                textSize = 12f
                setShadowLayer(2f, 1f, 1f, Color.BLACK)
            }
            pillContent.addView(brainCalibrationPillLine1)
            pillContent.addView(brainCalibrationPillLine2)
            val pillContainer = FrameLayout(this@SharpRoomActivity).apply {
                layoutParams = LinearLayout.LayoutParams(
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                ).apply { bottomMargin = dpToPx(8) }
                visibility = View.GONE
                isClickable = false
                addView(pillContent)
            }
            brainCalibrationPillContainer = pillContainer

            val snapshotHintColumn = LinearLayout(this@SharpRoomActivity).apply {
                orientation = LinearLayout.VERTICAL
                gravity = Gravity.CENTER_HORIZONTAL
            }
            snapshotHintColumn.addView(
                snapshotHintExplanationView,
                LinearLayout.LayoutParams(
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                ).apply { bottomMargin = dpToPx(6) },
            )
            snapshotHintColumn.addView(buildHintIconButton(R.drawable.ic_gesture_tap) { onSnapshotHintIconTapped() })

            val cameraBtn = AppCompatImageButton(this@SharpRoomActivity).apply {
                setImageResource(R.drawable.ic_camera)
                ImageViewCompat.setImageTintList(this, ColorStateList.valueOf(Color.WHITE))
                val bg = GradientDrawable().apply {
                    shape = GradientDrawable.OVAL
                    setColor(Color.parseColor("#007AFF"))
                }
                background = bg
                scaleType = ImageView.ScaleType.CENTER_INSIDE
                setPadding(dpToPx(10), dpToPx(10), dpToPx(10), dpToPx(10))
                layoutParams = LinearLayout.LayoutParams(brainSize, brainSize).apply {
                    topMargin = dpToPx(6)
                }
                setOnClickListener { takeScreenshot() }
            }

            val rightColumn = LinearLayout(this@SharpRoomActivity).apply {
                orientation = LinearLayout.VERTICAL
                gravity = Gravity.END or Gravity.BOTTOM
            }
            rightColumn.addView(pillContainer)
            rightColumn.addView(snapshotHintColumn)
            rightColumn.addView(cameraBtn)
            bottomRow.addView(rightColumn)

            mainColumn.addView(bottomRow)
            addView(mainColumn)
        }
    }

    /** Camera move arrows (up/down/left/right) — on-screen only, not in overflow menu. Matches iOS cameraDPadCluster. */
    private fun createCameraArrowOverlay(): FrameLayout {
        val paddingPx = dpToPx(12)
        val buttonSizePx = dpToPx(44)

        fun makeArrowButton(arrowText: String, onClick: () -> Unit): TextView {
            return TextView(this).apply {
                layoutParams = LinearLayout.LayoutParams(buttonSizePx, buttonSizePx)
                text = arrowText
                gravity = Gravity.CENTER
                textSize = 20f
                setTypeface(null, Typeface.BOLD)
                setTextColor(Color.WHITE)
                background = GradientDrawable().apply {
                    shape = GradientDrawable.OVAL
                    setColor(Color.argb(160, 0, 0, 0))
                }
                elevation = 2f
                setOnClickListener { onClick() }
            }
        }

        val container = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            setPadding(paddingPx, paddingPx, paddingPx, paddingPx)
            gravity = Gravity.CENTER_VERTICAL
        }

        container.addView(makeArrowButton("\u2190") { runMoveCamera(-8.0, 0.0) })
        val upDownColumn = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dpToPx(8), 0, dpToPx(8), 0)
        }
        upDownColumn.addView(makeArrowButton("\u2191") { runMoveCameraUp(0.2) })
        val downBtn = makeArrowButton("\u2193") { runMoveCameraUp(-0.2) }
        downBtn.layoutParams = LinearLayout.LayoutParams(buttonSizePx, buttonSizePx).apply { topMargin = dpToPx(8) }
        upDownColumn.addView(downBtn)
        container.addView(upDownColumn)
        container.addView(makeArrowButton("\u2192") { runMoveCamera(8.0, 0.0) })

        return FrameLayout(this).apply {
            isClickable = false
            elevation = 18f
            clipChildren = false
            clipToPadding = false
            addView(
                container,
                FrameLayout.LayoutParams(
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                ).apply {
                    gravity = Gravity.TOP or Gravity.START
                    leftMargin = dpToPx(12)
                    topMargin = dpToPx(12)
                },
            )
        }
    }

    private fun runMoveCamera(dx: Double, dy: Double) {
        val js = "if (typeof moveCamera === 'function') moveCamera($dx, $dy);"
        webView.evaluateJavascript(js, null)
    }

    private fun runMoveCameraUp(dy: Double) {
        val js = "if (typeof moveCameraUp === 'function') moveCameraUp($dy);"
        webView.evaluateJavascript(js, null)
    }

    private fun takeScreenshot() {
        try {
            val bitmap = captureSharpRoomSnapshotBitmap()

            // Save to Pictures folder
            val timeStamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())
            val fileName = "Room_$timeStamp.png"
            val picturesDir = getExternalFilesDir(Environment.DIRECTORY_PICTURES)
            val file = File(picturesDir, fileName)

            FileOutputStream(file).use { out ->
                bitmap.compress(Bitmap.CompressFormat.PNG, 100, out)
            }

            Toast.makeText(this, getString(R.string.sharp_room_screenshot_saved, fileName), Toast.LENGTH_SHORT).show()
            DebugLogger.d(TAG, "Screenshot saved: ${file.absolutePath}")

            // Share the screenshot
            val uri: android.net.Uri = FileProvider.getUriForFile(this, "${packageName}.fileprovider", file)
            val shareIntent = Intent(Intent.ACTION_SEND).apply {
                type = "image/png"
                putExtra(Intent.EXTRA_STREAM, uri)
                addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            }
            startActivity(Intent.createChooser(shareIntent, "Share Screenshot"))

        } catch (e: Exception) {
            DebugLogger.eDebugMode(TAG, "Failed to take screenshot", e)
            Toast.makeText(this, getString(R.string.sharp_room_screenshot_failed), Toast.LENGTH_SHORT).show()
        }
    }

    /**
     * WebView.draw() only captures the room renderer. When Brain/Furniture Fit is active, mirror
     * iOS by compositing the live segmentation/selection overlay into the exported snapshot too.
     */
    private fun captureSharpRoomSnapshotBitmap(): Bitmap {
        val captureWidth = webView.width.takeIf { it > 0 } ?: sharpRoomContentRoot.width
        val captureHeight = webView.height.takeIf { it > 0 } ?: sharpRoomContentRoot.height
        require(captureWidth > 0 && captureHeight > 0) { "Sharp room view is not laid out" }

        val bitmap = Bitmap.createBitmap(captureWidth, captureHeight, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bitmap)
        canvas.drawColor(Color.rgb(128, 128, 128))
        drawViewIntoSnapshot(webView, canvas)

        if (::brainDetectionOverlay.isInitialized && brainDetectionOverlay.visibility == View.VISIBLE) {
            drawViewIntoSnapshot(brainDetectionOverlay, canvas)
            DebugLogger.d(TAG, "Screenshot composited with Brain/Furniture Fit overlay")
        }

        return bitmap
    }

    private fun drawViewIntoSnapshot(view: View, canvas: Canvas) {
        if (view.width <= 0 || view.height <= 0 || view.visibility != View.VISIBLE) return
        val rootLocation = IntArray(2)
        val viewLocation = IntArray(2)
        webView.getLocationInWindow(rootLocation)
        view.getLocationInWindow(viewLocation)

        val saveCount = canvas.save()
        canvas.translate(
            (viewLocation[0] - rootLocation[0]).toFloat(),
            (viewLocation[1] - rootLocation[1]).toFloat(),
        )
        view.draw(canvas)
        canvas.restoreToCount(saveCount)
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

    /** Progress overlay when brain is tapped: room stays visible underneath (semi-transparent). */
    private fun createBrainProgressOverlay(): FrameLayout {
        return FrameLayout(this).apply {
            setBackgroundColor(Color.parseColor("#80000000"))
            layoutParams = FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT
            )
            val content = LinearLayout(this@SharpRoomActivity).apply {
                orientation = LinearLayout.VERTICAL
                gravity = Gravity.CENTER_HORIZONTAL
                setBackgroundColor(Color.parseColor("#99000000"))
                setPadding(dpToPx(32), dpToPx(16), dpToPx(32), dpToPx(16))
                val lp = FrameLayout.LayoutParams(
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT
                )
                lp.gravity = Gravity.TOP or Gravity.CENTER_HORIZONTAL
                lp.topMargin = dpToPx(120)
                layoutParams = lp

                val label = TextView(this@SharpRoomActivity).apply {
                    text = getString(R.string.smartypants_detecting_furniture)
                    setTextColor(Color.WHITE)
                    textSize = 14f
                    setPadding(dpToPx(8), dpToPx(4), dpToPx(8), dpToPx(8))
                }
                brainProgressLabel = label
                addView(label)
                val progress = ProgressBar(
                    this@SharpRoomActivity,
                    null,
                    android.R.attr.progressBarStyleHorizontal
                ).apply {
                    layoutParams = LinearLayout.LayoutParams(dpToPx(250), ViewGroup.LayoutParams.WRAP_CONTENT)
                    isIndeterminate = true
                    progressDrawable.colorFilter = android.graphics.PorterDuffColorFilter(
                        0xFF4CAF50.toInt(),
                        android.graphics.PorterDuff.Mode.SRC_IN
                    )
                }
                brainProgressBar = progress
                addView(progress)
            }
            addView(content)
        }
    }

    private fun activateBrainOverlayUi() {
        brainOverlayVisible = true
        setBrainSegmentationButtonActive(true)
        updateBrainActionButton()
        updateBrainLivePreviewVisibility()
        updateBrainSelectionHelperText()
        brainHintExplanationView?.post { updateBrainSelectionHelperText() }
        setBrainCalibrationPillVisible(true)
        updateBrainCalibrationPill()
        updatePlacementIntelligenceCard()
        updateFullVideoToolbarButton()
    }

    private fun showBrainProgressOverlay() {
        activateBrainOverlayUi()
        brainProgressOverlay.visibility = View.VISIBLE
    }

    /** Show startup progress only until the first successful detection run in this viewer session. */
    private fun showBrainProgressOverlayIfNeeded() {
        activateBrainOverlayUi()
        brainProgressOverlay.visibility = if (brainSegmentationCompletedOnceThisSession) {
            View.GONE
        } else {
            View.VISIBLE
        }
    }

    private fun hideBrainProgressOverlay() {
        brainProgressOverlay.visibility = View.GONE
    }

    private fun updateBrainActionButton() {
        val actionButton = brainActionButton ?: return
        if (!brainOverlayVisible) {
            actionButton.visibility = View.GONE
            return
        }
        if (brainSegmentationMode == BrainSegmentationMode.SEGMENT_SELECTED) {
            actionButton.text = getString(R.string.segment_stop_action)
            actionButton.visibility = View.VISIBLE
            return
        }
        if (selectedBrainPins.isNotEmpty()) {
            actionButton.text = getString(R.string.segment_furniture_action)
            actionButton.visibility = View.VISIBLE
            return
        }
        actionButton.visibility = View.GONE
    }

    private fun updateBrainSelectionHelperText() {
        brainHintExplanationView?.let { hint ->
            if (shouldShowIdentifyLivePreview() || selectedBrainPins.isNotEmpty()) {
                hint.text = getString(R.string.smartypants_pick_another_hint)
                hint.translationY = -dpToPx(34).toFloat()
                hint.visibility = View.VISIBLE
                gestureHintHideHandler.removeCallbacks(hideBrainHintRunnable)
                if (selectedBrainPins.isEmpty()) {
                    gestureHintHideHandler.postDelayed(hideBrainHintRunnable, 3000L)
                }
            } else {
                hint.text = getString(R.string.sharp_room_brain_gesture_hint)
                hint.translationY = 0f
            }
        }
    }

    private fun shouldShowIdentifyLivePreview(): Boolean {
        return showFullVideoWithIdentifications &&
            showIdentifyLivePreview &&
            brainSegmentationMode == BrainSegmentationMode.IDENTIFY_ONLY
    }

    private fun shouldShowIdentifyMaskOverlay(): Boolean {
        return brainSegmentationMode == BrainSegmentationMode.IDENTIFY_ONLY &&
            !shouldShowIdentifyLivePreview()
    }

    private fun updateBrainLivePreviewVisibility() {
        val shouldShowLivePreview = shouldShowIdentifyLivePreview()
        if (::brainCameraPreviewView.isInitialized) {
            brainCameraPreviewView.visibility =
                if (shouldShowLivePreview && brainArController == null && cameraPreviewUseCase != null) View.VISIBLE else View.GONE
        }
        brainArController?.let { controller ->
            val layoutParams = controller.glSurfaceView.layoutParams as? FrameLayout.LayoutParams
                ?: FrameLayout.LayoutParams(1, 1)
            if (shouldShowLivePreview) {
                layoutParams.width = ViewGroup.LayoutParams.MATCH_PARENT
                layoutParams.height = ViewGroup.LayoutParams.MATCH_PARENT
                controller.glSurfaceView.layoutParams = layoutParams
                controller.glSurfaceView.alpha = 1f
                controller.glSurfaceView.visibility = View.VISIBLE
            } else {
                layoutParams.width = 1
                layoutParams.height = 1
                controller.glSurfaceView.layoutParams = layoutParams
                controller.glSurfaceView.alpha = 0.01f
                controller.glSurfaceView.visibility = View.VISIBLE
            }
        }
    }

    private fun showBrainDetectionOverlay() {
        val maskForOverlay =
            if (brainSegmentationMode == BrainSegmentationMode.SEGMENT_SELECTED || shouldShowIdentifyMaskOverlay()) {
                latestBrainMask
            } else {
                null
            }
        val detectionsForOverlay =
            if (brainSegmentationMode == BrainSegmentationMode.IDENTIFY_ONLY) {
                latestBrainDetections
            } else {
                emptyList()
            }
        brainDetectionOverlayView.setMaskAndDetections(
            maskForOverlay,
            detectionsForOverlay,
            latestBrainInputSize,
            latestBrainOverlayScale,
            effectiveBrainFurnitureHeightDisplayMeters(),
            effRoomHeight(),
        )
        brainDetectionOverlayView.setDetectionBoxVisibility(brainSegmentationMode == BrainSegmentationMode.IDENTIFY_ONLY)
        brainDetectionOverlayView.setIdentifySelectionState(
            enabled = brainSegmentationMode == BrainSegmentationMode.IDENTIFY_ONLY,
            selectedPins = selectedBrainPins.toList(),
        )
        brainDetectionOverlay.visibility = if (brainOverlayVisible) View.VISIBLE else View.GONE
        updateBrainLivePreviewVisibility()
        setBrainSegmentationButtonActive(brainOverlayVisible)
        updateFullVideoToolbarButton()
        updateBrainActionButton()
        updateBrainSelectionHelperText()
        updatePlacementIntelligenceCard()
    }

    private fun clearBrainSelection() {
        selectedBrainPins.clear()
        updateBrainActionButton()
        updateBrainSelectionHelperText()
        brainDetectionOverlayView.setIdentifySelectionState(
            enabled = brainSegmentationMode == BrainSegmentationMode.IDENTIFY_ONLY,
            selectedPins = selectedBrainPins.toList(),
        )
    }

    private fun resetBrainSessionUiState() {
        brainSegmentationMode = BrainSegmentationMode.IDENTIFY_ONLY
        showIdentifyLivePreview = true
        latestBrainDetections = emptyList()
        latestBrainMask = null
        latestBrainInputSize = 640
        latestBrainOverlayScale = 1f
        latestBrainPrimaryDetection = null
        segmentedFurnitureMeanSrgb = null
        clearBrainSelection()
    }

    /** IoU in model input space (same as [DetectionResult] centers). Used to toggle the same instance vs another object of the same class. */
    private fun brainDetectionIoU(a: DetectionResult, b: DetectionResult): Float {
        val ax1 = a.x - a.w / 2f
        val ay1 = a.y - a.h / 2f
        val ax2 = a.x + a.w / 2f
        val ay2 = a.y + a.h / 2f
        val bx1 = b.x - b.w / 2f
        val by1 = b.y - b.h / 2f
        val bx2 = b.x + b.w / 2f
        val by2 = b.y + b.h / 2f
        val ix1 = max(ax1, bx1)
        val iy1 = max(ay1, by1)
        val ix2 = min(ax2, bx2)
        val iy2 = min(ay2, by2)
        val iw = max(0f, ix2 - ix1)
        val ih = max(0f, iy2 - iy1)
        val inter = iw * ih
        val ua = a.w * a.h + b.w * b.h - inter
        return if (ua > 0f) inter / ua else 0f
    }

    private fun handleBrainDetectionTapped(detection: DetectionResult) {
        if (brainSegmentationMode != BrainSegmentationMode.IDENTIFY_ONLY) return
        val idx = selectedBrainPins.indexOfFirst { brainDetectionIoU(it, detection) >= 0.5f }
        if (idx >= 0) {
            selectedBrainPins.removeAt(idx)
        } else {
            selectedBrainPins.add(detection)
        }
        brainDetectionOverlayView.setIdentifySelectionState(true, selectedBrainPins.toList())
        updateBrainSelectionHelperText()
        updateBrainActionButton()
    }

    private fun activateSelectedBrainSegmentation() {
        if (selectedBrainPins.isEmpty()) return
        brainSegmentationMode = BrainSegmentationMode.SEGMENT_SELECTED
        showIdentifyLivePreview = false
        latestBrainMask = null
        segmentedFurnitureMeanSrgb = null
        showBrainDetectionOverlay()
    }

    private fun stopBrainSegmentationOnly() {
        brainSegmentationMode = BrainSegmentationMode.IDENTIFY_ONLY
        showIdentifyLivePreview = true
        latestBrainMask = null
        segmentedFurnitureMeanSrgb = null
        showBrainDetectionOverlay()
    }

    private fun requestBrainInference(
        manager: FurnitureFitManager,
        bitmap: Bitmap,
        callback: (com.furnit.android.services.SegmentationResult?) -> Unit,
    ) {
        if (brainSegmentationMode == BrainSegmentationMode.SEGMENT_SELECTED && selectedBrainPins.isNotEmpty()) {
            manager.segmentSelectedInstancesAsync(bitmap, selectedBrainPins.toList(), callback)
        } else {
            manager.segmentWithDetectionsAsync(bitmap, callback)
        }
    }

    private fun applyBrainInferenceResult(
        result: com.furnit.android.services.SegmentationResult?,
        firstResultCallback: () -> Unit,
        sourceBitmap: Bitmap? = null,
    ) {
        if (brainSegmentationMode == BrainSegmentationMode.SEGMENT_SELECTED) {
            segmentedFurnitureMeanSrgb = result?.mask?.let { FurnitureSegmentationMeanColor.meanStraightSrgb(it) }
        } else {
            segmentedFurnitureMeanSrgb = null
        }
        latestBrainDetections = result?.detections ?: emptyList()
        latestBrainMask = result?.mask
        latestBrainInputSize = result?.inputSize ?: 640
        latestBrainPrimaryDetection = result?.primaryDetection ?: latestBrainDetections.firstOrNull()
        val currentHeightMeters =
            brainArController?.getProvisionalHeightMeters()?.takeIf { it.isFinite() && it > 0f }
                ?: brainArController?.getLastEstimatedHeightMeters()?.takeIf { it.isFinite() && it > 0f }
        lockBrainFurnitureSizeIfNeeded(null, currentHeightMeters)
        latestBrainOverlayScale = brainOverlayScaleForDetection(
            latestBrainPrimaryDetection,
            latestBrainInputSize,
            effectiveBrainFurnitureHeightDisplayMeters(),
        )
        firstResultCallback()
        showBrainDetectionOverlay()
        if (sourceBitmap != null && latestBrainPrimaryDetection != null && latestBrainInputSize > 0) {
            val det = latestBrainPrimaryDetection ?: return
            val inp = latestBrainInputSize.coerceAtLeast(1).toFloat()
            val scaleX = sourceBitmap.width / inp
            val scaleY = sourceBitmap.height / inp
            brainArController?.setBboxHint(
                det.x * scaleX,
                det.y * scaleY,
                det.h * scaleY,
                det.label,
            )
        } else if (latestBrainPrimaryDetection == null) {
            brainArController?.clearBboxHint()
        }
        updateBrainCalibrationPill()
        updateRoomPlacementIntelligence()
    }

    private fun setBrainSegmentationButtonActive(active: Boolean) {
        if (!::brainModeButton.isInitialized) return
        val color = if (active) BRAIN_BUTTON_COLOR_SEGMENTING else BRAIN_BUTTON_COLOR_IDLE
        (brainModeButton.background as? GradientDrawable)?.setColor(Color.parseColor(color))
    }

    private fun setBrainArAssistButtonActive(active: Boolean) {
        val color = if (active) BRAIN_BUTTON_COLOR_SEGMENTING else "#3A3A3C"
        (brainArAssistButton?.background as? GradientDrawable)?.setColor(Color.parseColor(color))
    }

    private fun updateFullVideoToolbarButton() {
        fullVideoIdentificationsButton?.let { button ->
            button.visibility = if (brainOverlayVisible) View.VISIBLE else View.GONE
            val tint = if (showFullVideoWithIdentifications) Color.parseColor("#34C759") else Color.WHITE
            ImageViewCompat.setImageTintList(button, ColorStateList.valueOf(tint))
        }
        brainArAssistButton?.let { button ->
            button.visibility = if (brainOverlayVisible) View.VISIBLE else View.GONE
            ImageViewCompat.setImageTintList(
                button,
                ColorStateList.valueOf(if (brainArAssistRequested) Color.parseColor("#34C759") else Color.WHITE),
            )
        }
    }

    private fun launchBrainMode(arAssistedRequested: Boolean) {
        if (!brainOverlayVisible && brainProgressOverlay.visibility != View.VISIBLE) {
            showFullVideoWithIdentifications = FurnitureFitManager.isFullVideoWithIdentificationsEnabled(this)
        }
        pendingBrainStartArAssist = arAssistedRequested
        if (brainDetectionOverlay.visibility == View.VISIBLE) {
            if (brainArAssistRequested == arAssistedRequested) {
                DebugLogger.d(TAG, "Brain: tap while same mode active — stopping")
                hideBrainDetectionOverlay()
                return
            }
            DebugLogger.d(TAG, "Brain: switching live mode arAssist=$arAssistedRequested")
            hideBrainDetectionOverlay()
        }
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            != PackageManager.PERMISSION_GRANTED) {
            DebugLogger.d(TAG, "Brain: requesting CAMERA permission arAssist=$arAssistedRequested")
            cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
        } else {
            DebugLogger.d(TAG, "Brain: permission OK, starting detection arAssist=$arAssistedRequested")
            resetBrainSessionUiState()
            showBrainProgressOverlayIfNeeded()
            startBrainDetection(arAssistedRequested)
        }
    }

    private fun ensureCameraPreviewBoundForFullVideoIfNeeded() {
        if (!brainOverlayVisible ||
            !showFullVideoWithIdentifications ||
            !showIdentifyLivePreview ||
            brainSegmentationMode != BrainSegmentationMode.IDENTIFY_ONLY ||
            brainArController != null ||
            cameraPreviewUseCase != null
        ) {
            return
        }
        val manager = furnitureFitManager ?: return
        val nextGeneration = brainSessionGeneration.incrementAndGet()
        brainSegmentationAcceptingUpdates = false
        isBrainInferenceRunning.set(false)
        bindBrainCamera(manager, nextGeneration)
    }

    private fun hideBrainDetectionOverlay() {
        DebugLogger.d(TAG, "Brain: hideBrainDetectionOverlay() - user stopped or Back, stopping camera")
        setBrainSegmentationButtonActive(false)
        setBrainArAssistButtonActive(false)
        brainOverlayVisible = false
        updateFullVideoToolbarButton()
        brainDetectionOverlay.visibility = View.GONE
        brainDetectionOverlayView.setMaskAndDetections(null, emptyList())
        brainDetectionOverlayView.setDetectionBoxVisibility(false)
        brainDetectionOverlayView.setIdentifySelectionState(false, emptyList())
        updateBrainLivePreviewVisibility()
        hideBrainProgressOverlay()
        stopBrainDetection()
        setBrainCalibrationPillVisible(false)
        setPlacementIntelligenceVisible(false)
        updateBrainActionButton()
    }

    private fun createInitializedFurnitureFitManager(): FurnitureFitManager? {
        setBrainProgressText(R.string.yoloe_loading_model)
        val initializedManager = FurnitureFitManager(this)
        if (initializedManager.initializeAuto()) {
            setBrainProgressText(R.string.smartypants_detecting_furniture)
            return initializedManager
        }
        setBrainProgressText(R.string.yoloe_model_unavailable)
        initializedManager.close()
        return null
    }

    private fun setBrainProgressText(resId: Int) {
        runOnUiThread {
            brainProgressLabel?.text = getString(resId)
        }
    }

    private fun prewarmBrainSegmentationIfNeeded() {
        if (furnitureFitManager != null) return
        val existingWarmupJob = brainModelWarmupJob
        if (existingWarmupJob != null && existingWarmupJob.isActive) return
        brainModelWarmupJob = lifecycleScope.async(Dispatchers.IO) {
            createInitializedFurnitureFitManager()
        }
    }

    private fun startBrainDetection(arAssistedRequested: Boolean = false) {
        DebugLogger.d(TAG, "Brain: startBrainDetection() - initializing SmartyPants on IO thread arAssist=$arAssistedRequested")
        val sessionGeneration = brainSessionGeneration.incrementAndGet()
        // Reset per-session state and any pending timeout from a previous brain run.
        resetBrainSessionUiState()
        brainFirstResultReceived = false
        brainTimeoutRunnable?.let { brainTimeoutHandler.removeCallbacks(it) }
        brainTimeoutRunnable = null
        disableArBrainThisSession = false
        brainArAssistRequested = arAssistedRequested
        pendingBrainStartArAssist = arAssistedRequested
        setBrainArAssistButtonActive(arAssistedRequested)
        prewarmBrainSegmentationIfNeeded()
        lifecycleScope.launch {
            val warmedManager = furnitureFitManager ?: brainModelWarmupJob?.await()
            if (furnitureFitManager == null && warmedManager != null) {
                furnitureFitManager = warmedManager
            }
            brainModelWarmupJob = null
            val manager = furnitureFitManager ?: withContext(Dispatchers.IO) {
                createInitializedFurnitureFitManager()
            }
            if (manager == null) {
                DebugLogger.eDebugMode(TAG, "Brain: SmartyPants failed to initialize")
                runOnUiThread {
                    hideBrainProgressOverlay()
                    setBrainCalibrationPillVisible(false)
                    Toast.makeText(this@SharpRoomActivity, getString(R.string.sharp_room_smartypants_failed), Toast.LENGTH_SHORT).show()
                }
                return@launch
            }
            DebugLogger.d(TAG, "Brain: SmartyPants OK, binding camera on UI thread")
            furnitureFitManager = manager
            runOnUiThread { bindBrainCamera(manager, sessionGeneration) }

            // If ARCore path fails to produce any segmentation result (e.g. camera not available or ARCore
            // session cannot be created), fall back to classic CameraX brain path instead of leaving the
            // user stuck on "Detecting furniture…".
            brainTimeoutRunnable = Runnable {
                if (!brainFirstResultReceived) {
                    DebugLogger.eDebugMode(TAG, "Brain: timeout waiting for first result, falling back to CameraX brain path")
                    teardownBrainArCoreController()
                    brainSegmentationAcceptingUpdates = false
                    setBrainSegmentationButtonActive(false)
                    hideBrainProgressOverlay()
                    disableArBrainThisSession = true
                    furnitureFitManager?.let { mgr ->
                        val fallbackGeneration = brainSessionGeneration.incrementAndGet()
                        bindBrainCamera(mgr, fallbackGeneration)
                    }
                }
            }.also { runnable ->
                brainTimeoutHandler.postDelayed(runnable, 7_000L)
            }
        }
    }

    private fun shouldUseArBrainCamera(): Boolean {
        if (disableArBrainThisSession) return false
        return brainArAssistRequested && ArSupportChecker.isArCoreSupported(this)
    }

    /** Single teardown path for brain ARCore controller (GL session close + view removal). */
    private fun teardownBrainArCoreController() {
        brainArController?.let { controller ->
            try {
                sharpRoomContentRoot.removeView(controller.glSurfaceView)
            } catch (_: Exception) {
            }
            controller.destroy()
        }
        brainArController = null
    }

    /** Match CameraX brain frames to [photoOrientation], same as ARCore path and Furniture Fit fragment. */
    private fun alignBrainCameraBitmapToLockedRoom(bitmap: Bitmap): Bitmap {
        val (oriented, _) = bitmap.rotateToMatchLockedRoomPhoto(photoOrientation)
        if (oriented !== bitmap) bitmap.recycle()
        return oriented
    }

    @SuppressLint("UnsafeOptInUsageError")
    private fun bindBrainCamera(manager: FurnitureFitManager, sessionGeneration: Int) {
        if (shouldUseArBrainCamera()) {
            bindBrainArCoreCamera(manager, sessionGeneration)
            return
        }
        // Switching from ARCore brain path to CameraX: remove GL surface or we keep AR frames while AR is off in prefs.
        teardownBrainArCoreController()
        DebugLogger.d(TAG, "Brain: bindBrainCamera() - getting ProcessCameraProvider")
        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener({
            val provider = providerFuture.get()
            cameraProvider = provider
            provider.unbindAll()
            DebugLogger.d(TAG, "Brain: building ImageAnalysis and binding to BACK_CAMERA")
            val brainAnalysisSize =
                if (photoOrientation.equals("landscape", ignoreCase = true)) {
                    android.util.Size(640, 480)
                } else {
                    android.util.Size(480, 640)
                }
            val analysis = ImageAnalysis.Builder()
                .setTargetResolution(brainAnalysisSize)
                // Match display so ImageProxy.rotationDegrees + toBitmapSafe() align mask with portrait/landscape UI
                .setTargetRotation(displayRotationForCameraX())
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
            val shouldBindPreview =
                showFullVideoWithIdentifications &&
                    showIdentifyLivePreview &&
                    brainSegmentationMode == BrainSegmentationMode.IDENTIFY_ONLY
            val preview =
                if (shouldBindPreview) {
                    Preview.Builder()
                        .setTargetResolution(brainAnalysisSize)
                        .setTargetRotation(displayRotationForCameraX())
                        .build().also { previewUseCase ->
                            previewUseCase.setSurfaceProvider(brainCameraPreviewView.surfaceProvider)
                        }
                } else {
                    null
                }
            if (::brainCameraPreviewView.isInitialized) {
                brainCameraPreviewView.visibility = View.GONE
            }
            cameraPreviewUseCase = preview
            var frameCount = 0
            val hasFirstResult = BooleanArray(1) { false }
            analysis.setAnalyzer(cameraExecutor) { imageProxy ->
                try {
                    val rawBitmap = imageProxy.toBitmapSafe() ?: return@setAnalyzer
                    val bitmap = alignBrainCameraBitmapToLockedRoom(rawBitmap)
                    // Only process one frame at a time; drop others so we show current view when camera moves (no "chair forever")
                    if (isBrainInferenceRunning.get()) {
                        return@setAnalyzer
                    }
                    isBrainInferenceRunning.set(true)
                    frameCount++
                    if (frameCount == 1 || frameCount % 30 == 0) {
                        DebugLogger.d(TAG, "Brain: analysis frame $frameCount (camera active)")
                    }
                    requestBrainInference(manager, bitmap) { result ->
                        runOnUiThread {
                            isBrainInferenceRunning.set(false)
                            if (!brainSegmentationAcceptingUpdates || brainSessionGeneration.get() != sessionGeneration) return@runOnUiThread
                            applyBrainInferenceResult(
                                result = result,
                                firstResultCallback = {
                                    if (!hasFirstResult[0]) {
                                        hasFirstResult[0] = true
                                        brainFirstResultReceived = true
                                        brainTimeoutRunnable?.let { brainTimeoutHandler.removeCallbacks(it) }
                                        brainTimeoutRunnable = null
                                        DebugLogger.d(TAG, "Brain: first result - hiding progress, showing detection overlay")
                                        brainSegmentationCompletedOnceThisSession = true
                                        hideBrainProgressOverlay()
                                    }
                                },
                            )
                        }
                    }
                } finally {
                    imageProxy.close()
                }
            }
            try {
                brainSegmentationAcceptingUpdates = true
                if (preview != null) {
                    provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis)
                } else {
                    provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, analysis)
                }
                updateBrainLivePreviewVisibility()
                DebugLogger.d(TAG, "Brain: camera bound successfully - live segmentation running")
            } catch (e: Exception) {
                brainSegmentationAcceptingUpdates = false
                cameraPreviewUseCase = null
                DebugLogger.eDebugMode(TAG, "Brain camera bind failed", e)
                runOnUiThread {
                    hideBrainProgressOverlay()
                    setBrainCalibrationPillVisible(false)
                    Toast.makeText(this@SharpRoomActivity, getString(R.string.sharp_room_camera_error, e.message ?: ""), Toast.LENGTH_SHORT).show()
                    CrashReporter.report(this@SharpRoomActivity, e, "Sharp room brain / camera bind")
                }
            }
        }, ContextCompat.getMainExecutor(this))
    }

    @SuppressLint("UnsafeOptInUsageError")
    private fun bindBrainArCoreCamera(manager: FurnitureFitManager, sessionGeneration: Int) {
        DebugLogger.d(TAG, "Brain: bindBrainArCoreCamera() - ARCore path")
        cameraProvider?.unbindAll()
        cameraProvider = null
        teardownBrainArCoreController()
        val controller = FurnitureFitArCameraController(this, cameraExecutor)
        brainArController = controller
        controller.lockedPhotoOrientation = photoOrientation
        controller.roomHeightMetersForFallback = effRoomHeight()
        controller.onAssistedMeasurementUpdated = {
            updateBrainCalibrationPill()
        }
        val lp = FrameLayout.LayoutParams(
            1,
            1,
        )
        sharpRoomContentRoot.addView(controller.glSurfaceView, 1, lp)
        controller.glSurfaceView.visibility = View.VISIBLE
        controller.glSurfaceView.alpha = 0.01f

        val hasFirstResult = BooleanArray(1) { false }
        controller.shouldPostBitmapFrame = { !isBrainInferenceRunning.get() }
        controller.onBitmapFrame = arBitmap@{ bitmap ->
            if (isBrainInferenceRunning.get()) {
                return@arBitmap
            }
            isBrainInferenceRunning.set(true)
            requestBrainInference(manager, bitmap) { result ->
                runOnUiThread {
                    isBrainInferenceRunning.set(false)
                    if (!brainSegmentationAcceptingUpdates || brainSessionGeneration.get() != sessionGeneration) return@runOnUiThread
                    val mask = result?.mask
                    val dets = result?.detections ?: emptyList()
                    val size = result?.inputSize ?: 640
                    val nowMs = SystemClock.elapsedRealtime()
                    if (nowMs - lastBrainArBridgeLogMs >= 500L) {
                        lastBrainArBridgeLogMs = nowMs
                        val firstDet = dets.firstOrNull()
                        LogUtil.furnitureFitAr(
                            "platform=android phase=brain_bridge " +
                                "controllerPresent=${brainArController != null} " +
                                "maskPresent=${mask != null} detCount=${dets.size} input=$size " +
                                "detLabel=${firstDet?.label?.take(48) ?: "none"} " +
                                "bboxModelH=${firstDet?.h?.let { String.format(Locale.US, "%.1f", it) } ?: "null"}",
                        )
                    }
                    applyBrainInferenceResult(
                        result = result,
                        firstResultCallback = {
                            if (!hasFirstResult[0]) {
                                hasFirstResult[0] = true
                                brainFirstResultReceived = true
                                brainTimeoutRunnable?.let { brainTimeoutHandler.removeCallbacks(it) }
                                brainTimeoutRunnable = null
                                DebugLogger.d(TAG, "Brain: first result (ARCore) - hiding progress, showing detection overlay")
                                brainSegmentationCompletedOnceThisSession = true
                                hideBrainProgressOverlay()
                            }
                        },
                        sourceBitmap = bitmap,
                    )
                    brainArController?.onInferenceFinished()
                }
            }
        }
        brainSegmentationAcceptingUpdates = true
        updateBrainLivePreviewVisibility()
        controller.onHostResume()
    }

    private fun stopBrainDetection() {
        DebugLogger.d(TAG, "Brain: stopBrainDetection() - unbinding camera / AR")
        brainSessionGeneration.incrementAndGet()
        brainSegmentationAcceptingUpdates = false
        // Must clear even if a pending segmentWithDetectionsAsync callback returns early (acceptingUpdates false),
        // or the next brain session never processes frames (CameraX analyzer gates on this flag).
        isBrainInferenceRunning.set(false)
        brainFirstResultReceived = false
        brainTimeoutRunnable?.let { brainTimeoutHandler.removeCallbacks(it) }
        brainTimeoutRunnable = null
        disableArBrainThisSession = false
        pendingBrainStartArAssist = false
        brainArAssistRequested = false
        setBrainArAssistButtonActive(false)
        cameraPreviewUseCase = null
        if (::brainCameraPreviewView.isInitialized) {
            brainCameraPreviewView.visibility = View.GONE
        }
        brainArController?.clearBboxHint()
        teardownBrainArCoreController()
        brainLockedFurnitureWidthMeters = null
        brainLockedFurnitureHeightMeters = null
        brainRealFurnitureHeightMeters = null
        brainCalibrationScaleFactor = 1.0f
        resetBrainSessionUiState()
        latestFitCheckResult = null
        latestCornerPlacementSuggestions = emptyList()
        latestAestheticScore = null
        latestEstimatedFurnitureDepthMeters = null
        updateBrainCalibrationPill()
        updatePlacementIntelligenceCard()
        try {
            cameraProvider?.unbindAll()
        } catch (_: Exception) { }
        cameraProvider = null
    }

    private fun loadWebGLViewer() {
        val plyFile = File(plyPath!!)
        if (!plyFile.exists()) {
            DebugLogger.eDebugMode(TAG, "PLY file not found: $plyPath")
            Toast.makeText(this, getString(R.string.sharp_room_ply_not_found), Toast.LENGTH_SHORT).show()
            finish()
            return
        }

        DebugLogger.d(TAG, "Loading PLY file: ${plyFile.name} (${plyFile.length()} bytes)")
        logSharpLoadTiming("webview_load_start", "file=${plyFile.name} bytes=${plyFile.length()}")

        // Load HTML using WebViewAssetLoader base URL
        // SparkJS will fetch PLY from https://appassets.androidplatform.net/files/room.ply
        val html = generateWebGLHTML()
        webView.loadDataWithBaseURL(
            "https://appassets.androidplatform.net/",
            html,
            "text/html",
            "UTF-8",
            null
        )
    }

    private fun generateWebGLHTML(): String {
        // Check auto-orbit + debug logging (same keys as Settings / DebugLogger)
        val prefs = getSharedPreferences("furnit_prefs", MODE_PRIVATE)
        val autoOrbitEnabled = prefs.getBoolean("auto_orbit_enabled", false)
        val sharpJsBool = if (prefs.getBoolean("debug_mode", false)) "true" else "false"
        // Use isPortrait like iOS for consistency
        val isPortrait = photoOrientation != "landscape"
        DebugLogger.d(TAG, "[SharpRoom] Building WebView HTML: photoOrientation=$photoOrientation isPortrait=$isPortrait photoWideAngle=$photoWideAngle (this activity = PLY/splat room)")
        // JS framing matches iOS SharpRoomView WebGL: camera starts outside the front wall at maxZ (+dist), target at maxZ.
        // (Legacy Android min-Z rail for landscape misaligned “start inside the room” vs iOS.)
        val defaultDisplayW = SharpRoomDimensionSanitizer.DEFAULT_DISPLAY_WIDTH_M.toDouble()
        val defaultDisplayH = SharpRoomDimensionSanitizer.DEFAULT_DISPLAY_HEIGHT_M.toDouble()
        val defaultDisplayD = SharpRoomDimensionSanitizer.DEFAULT_DISPLAY_DEPTH_M.toDouble()
        val fallbackW = effRoomWidth().toDouble()
        val fallbackH = effRoomHeight().toDouble()
        val fallbackD = effRoomDepth().toDouble()
        val fallbackCx = effRoomCenterX().toDouble()
        val fallbackCy = effRoomCenterY().toDouble()
        val fallbackCz = effRoomCenterZ().toDouble()
        val hint = persistedSplatLoadHint
        val hasSplatLoadHint = hint != null
        val hintFullMinX = hint?.fullBoundsMin?.x?.toDouble() ?: 0.0
        val hintFullMinY = hint?.fullBoundsMin?.y?.toDouble() ?: 0.0
        val hintFullMinZ = hint?.fullBoundsMin?.z?.toDouble() ?: 0.0
        val hintFullMaxX = hint?.fullBoundsMax?.x?.toDouble() ?: 0.0
        val hintFullMaxY = hint?.fullBoundsMax?.y?.toDouble() ?: 0.0
        val hintFullMaxZ = hint?.fullBoundsMax?.z?.toDouble() ?: 0.0
        val hintFramingMinX = hint?.framingBoundsMin?.x?.toDouble() ?: hintFullMinX
        val hintFramingMinY = hint?.framingBoundsMin?.y?.toDouble() ?: hintFullMinY
        val hintFramingMinZ = hint?.framingBoundsMin?.z?.toDouble() ?: hintFullMinZ
        val hintFramingMaxX = hint?.framingBoundsMax?.x?.toDouble() ?: hintFullMaxX
        val hintFramingMaxY = hint?.framingBoundsMax?.y?.toDouble() ?: hintFullMaxY
        val hintFramingMaxZ = hint?.framingBoundsMax?.z?.toDouble() ?: hintFullMaxZ
        val hintCenterX = hint?.centroid?.x?.toDouble() ?: 0.0
        val hintCenterY = hint?.centroid?.y?.toDouble() ?: 0.0
        val hintCenterZ = hint?.centroid?.z?.toDouble() ?: 0.0
        val usedWideLens = photoWideAngle

        // SparkJS implementation matching iOS exactly
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
            -webkit-touch-callout: none;
            -webkit-user-select: none;
        }
        canvas {
            width: 100%;
            height: 100%;
            display: block;
            touch-action: none;
        }
    </style>
    <script type="importmap">
    {
        "imports": {
            "three": "https://cdnjs.cloudflare.com/ajax/libs/three.js/0.170.0/three.module.min.js",
            "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.170.0/examples/jsm/",
            "@sparkjsdev/spark": "https://sparkjs.dev/releases/spark/0.1.10/spark.module.js"
        }
    }
    </script>
</head>
<body>
    <script type="module">
        import * as THREE from 'three';
        import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
        import { SplatMesh, SparkRenderer } from '@sparkjsdev/spark';

        const SHARP_ROOM_DEBUG = $sharpJsBool;
        const _sharpConsoleLog = console.log.bind(console);
        console.log = function() { if (SHARP_ROOM_DEBUG) _sharpConsoleLog.apply(console, arguments); };
        function sharpAndroidLog(msg) {
            if (SHARP_ROOM_DEBUG && window.Android && window.Android.log) window.Android.log(msg);
        }
        function reportStage(stage, detail) {
            if (window.Android && window.Android.onSplatLoadStage) {
                window.Android.onSplatLoadStage(stage, detail || '');
            }
        }

        console.log('[WebGL] SparkJS Gaussian Splat viewer initializing...');
        reportStage('html_init', '');
        // Orientation and fallback dimensions from Kotlin (module scope so autoFrameRoom can use them)
        const isPortrait = $isPortrait;
        const usedWideLens = $usedWideLens;
        const fallbackRoomWidth = $fallbackW;
        const fallbackRoomHeight = $fallbackH;
        const fallbackRoomDepth = $fallbackD;
        const fallbackRoomCenterX = $fallbackCx;
        const fallbackRoomCenterY = $fallbackCy;
        const fallbackRoomCenterZ = $fallbackCz;
        const hasSplatLoadHint = $hasSplatLoadHint;
        const hintFullMinX = $hintFullMinX;
        const hintFullMinY = $hintFullMinY;
        const hintFullMinZ = $hintFullMinZ;
        const hintFullMaxX = $hintFullMaxX;
        const hintFullMaxY = $hintFullMaxY;
        const hintFullMaxZ = $hintFullMaxZ;
        const hintFramingMinX = $hintFramingMinX;
        const hintFramingMinY = $hintFramingMinY;
        const hintFramingMinZ = $hintFramingMinZ;
        const hintFramingMaxX = $hintFramingMaxX;
        const hintFramingMaxY = $hintFramingMaxY;
        const hintFramingMaxZ = $hintFramingMaxZ;
        const hintCenterX = $hintCenterX;
        const hintCenterY = $hintCenterY;
        const hintCenterZ = $hintCenterZ;
        const reasonableDefaultW = $defaultDisplayW;
        const reasonableDefaultH = $defaultDisplayH;
        const reasonableDefaultD = $defaultDisplayD;
        console.log('[SharpRoom] orientation: ' + (isPortrait ? 'portrait' : 'landscape') + ' (isPortrait=' + isPortrait + '), wideAngle(0.5x): ' + usedWideLens + ', fallbackDims: ' + fallbackRoomWidth.toFixed(2) + 'x' + fallbackRoomHeight.toFixed(2) + 'x' + fallbackRoomDepth.toFixed(2) + ', center: ' + fallbackRoomCenterX.toFixed(2) + ',' + fallbackRoomCenterY.toFixed(2) + ',' + fallbackRoomCenterZ.toFixed(2));

        // Scene setup (matching iOS exactly)
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x808080);

        // Camera — landscape infinite pinch: smaller near (see iOS INFINITE_ZOOM) so close dollying still draws; portrait keeps 0.1.
        const cameraNear = isPortrait ? 0.1 : 0.001;
        const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, cameraNear, 1000);
        camera.position.set(0, 0, 5);
        camera.up.set(0, 1, 0);

        // THREE.js Renderer. Android cannot force WebView's Chromium backend to Vulkan from app code, but
        // powerPreference + hardware WebView composition lets ANGLE/WebView choose its fastest GPU path.
        const renderer = new THREE.WebGLRenderer({
            antialias: false,
            alpha: false,
            depth: true,
            stencil: false,
            preserveDrawingBuffer: false,
            failIfMajorPerformanceCaveat: false,
            powerPreference: 'high-performance'
        });
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.setPixelRatio(window.devicePixelRatio);
        document.body.appendChild(renderer.domElement);

        function reportWebGlBackend() {
            try {
                const gl = renderer.getContext();
                const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
                const vendor = debugInfo ? gl.getParameter(debugInfo.UNMASKED_VENDOR_WEBGL) : gl.getParameter(gl.VENDOR);
                const rendererName = debugInfo ? gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER);
                const version = gl.getParameter(gl.VERSION);
                const shadingLanguage = gl.getParameter(gl.SHADING_LANGUAGE_VERSION);
                const webglKind = (typeof WebGL2RenderingContext !== 'undefined' && gl instanceof WebGL2RenderingContext) ? 'WebGL2' : 'WebGL1';
                const line = '[SharpRoom_GPU] kind=' + webglKind +
                    ' vendor=' + vendor +
                    ' renderer=' + rendererName +
                    ' version=' + version +
                    ' shading=' + shadingLanguage +
                    ' dpr=' + window.devicePixelRatio;
                _sharpConsoleLog(line);
                sharpAndroidLog(line);
                if (window.Android && window.Android.onGpuRendererInfo) {
                    window.Android.onGpuRendererInfo(line);
                }
            } catch (e) {
                const line = '[SharpRoom_GPU] renderer probe failed: ' + e;
                _sharpConsoleLog(line);
                sharpAndroidLog(line);
            }
        }
        renderer.domElement.addEventListener('webglcontextlost', function(ev) {
            ev.preventDefault();
            const line = '[SharpRoom_GPU] WebGL context lost';
            _sharpConsoleLog(line);
            sharpAndroidLog(line);
            if (window.Android && window.Android.onGpuRendererInfo) window.Android.onGpuRendererInfo(line);
        }, false);
        renderer.domElement.addEventListener('webglcontextrestored', function() {
            const line = '[SharpRoom_GPU] WebGL context restored';
            _sharpConsoleLog(line);
            sharpAndroidLog(line);
            if (window.Android && window.Android.onGpuRendererInfo) window.Android.onGpuRendererInfo(line);
            needsRender = true;
        }, false);
        reportWebGlBackend();

        // SparkRenderer with settings matching iOS exactly
        const spark = new SparkRenderer({
            renderer: renderer,
            maxStdDev: 3.0,
            preBlurAmount: 0.5,
            blurAmount: 0.3,
            falloff: 0.8,
            focalAdjustment: 1.5
        });
        camera.add(spark);

        // Orbit controls (low sensitivity: higher damping = less oscillation, low rotateSpeed = slow, comfortable drag)
        const controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.25;   // Settle quickly so orbit does not oscillate
        controls.rotateSpeed = 0.25;     // Slow rotation for touch so room does not move too fast
        controls.screenSpacePanning = false;
        // Portrait: small minDistance + capped max (room stays framed). Landscape: infinite dolly — minDistance 0 lets
        // pinch-zoom pass through the target / walls along the view axis (matches iOS INFINITE_ZOOM idea).
        const roomMaxDim = Math.max(fallbackRoomWidth, fallbackRoomHeight, fallbackRoomDepth);
        if (isPortrait) {
            controls.minDistance = 0.001;
            controls.maxDistance = Math.max(6, Math.min(25, roomMaxDim * 2.5));
        } else {
            controls.minDistance = 0;
            controls.maxDistance = 1e6;
            controls.zoomSpeed = 2.0;
            // Built-in OrbitControls dolly only moves camera toward a fixed target → stalls ~0.02m and cannot pass
            // through walls. Same as iOS INFINITE_ZOOM zoomCamera: move camera + target together along view ray.
            controls.enableZoom = false;
        }
        controls.target.set(0, 0, 0);
        controls.minAzimuthAngle = -Infinity;
        controls.maxAzimuthAngle = Infinity;
        // Portrait: narrow polar cone (comfortable tilt). Landscape: full 0..π — tight min/max polar was still clamping
        // spherical phi when dollying through the target (pinch “infinite” felt stuck at the wall).
        if (isPortrait) {
            controls.minPolarAngle = 0.001;
            controls.maxPolarAngle = Math.PI - 0.001;
        } else {
            controls.minPolarAngle = 0;
            controls.maxPolarAngle = Math.PI;
        }

        let initialCameraPosition = camera.position.clone();
        let initialControlsTarget = controls.target.clone();
        let needsRender = true;

        // Auto-orbit settings (matches iOS)
        const OSCILLATION_ENABLED = $autoOrbitEnabled;
        let autoOrbitEnabled = OSCILLATION_ENABLED;
        let autoOrbitTime = 0;
        let autoOrbitBaseAngle = 0;
        let autoOrbitRadius = 0;

        // Warm-up rendering (matches iOS - 5 seconds)
        const WARMUP_DURATION = 5000;
        const animationStartTime = performance.now();

        // Room dimensions (will be set by autoFrameRoom)
        let measuredRoomWidth = 4.0;
        let measuredRoomHeight = 3.0;
        let cameraFramedAt = 0;
        // Current room dims for benchmark log (set when framing; used when user stops moving)
        let currentRoomW = fallbackRoomWidth, currentRoomH = fallbackRoomHeight, currentRoomD = fallbackRoomDepth;
        /** Tightened bounds after autoFrame (matches iOS joystick / zoom clamping). */
        let roomBoundsForClamping = null;

        let benchmarkLogTimeout = null;
        controls.addEventListener('change', function() {
            needsRender = true;
            if (benchmarkLogTimeout) clearTimeout(benchmarkLogTimeout);
            benchmarkLogTimeout = setTimeout(function() {
                benchmarkLogTimeout = null;
                const p = camera.position;
                const t = controls.target;
                const dist = p.distanceTo(t);
                console.log('[BENCHMARK_CAMERA] portrait=' + (isPortrait ? 1 : 0) + ' posX=' + p.x.toFixed(4) + ' posY=' + p.y.toFixed(4) + ' posZ=' + p.z.toFixed(4) + ' tgtX=' + t.x.toFixed(4) + ' tgtY=' + t.y.toFixed(4) + ' tgtZ=' + t.z.toFixed(4) + ' distance=' + dist.toFixed(4) + ' roomW=' + currentRoomW.toFixed(4) + ' roomH=' + currentRoomH.toFixed(4) + ' roomD=' + currentRoomD.toFixed(4));
            }, 500);
        });

        if (!isPortrait) {
            const LANDSCAPE_DOLLY_SENS = 4.0;
            const LANDSCAPE_DOLLY_STEP = 0.22;
            const PINCH_THRESHOLD = 0.012;
            function landscapeDollyAlongView(scale) {
                if (typeof scale !== 'number' || !isFinite(scale) || scale <= 0) return;
                let forward = new THREE.Vector3().subVectors(controls.target, camera.position);
                if (forward.lengthSq() < 1e-14) {
                    camera.getWorldDirection(forward);
                    forward.negate();
                } else {
                    forward.normalize();
                }
                const step = (scale - 1) * LANDSCAPE_DOLLY_STEP * LANDSCAPE_DOLLY_SENS;
                camera.position.addScaledVector(forward, step);
                controls.target.addScaledVector(forward, step);
                controls.update();
                needsRender = true;
                autoOrbitEnabled = false;
                const dist = camera.position.distanceTo(controls.target);
                const line = '[LANDSCAPE_DOLLY] scale=' + scale.toFixed(4) + ' step=' + step.toFixed(5) + ' dist=' + dist.toFixed(5) +
                    ' cam=' + camera.position.x.toFixed(3) + ',' + camera.position.y.toFixed(3) + ',' + camera.position.z.toFixed(3) +
                    ' tgt=' + controls.target.x.toFixed(3) + ',' + controls.target.y.toFixed(3) + ',' + controls.target.z.toFixed(3);
                _sharpConsoleLog(line);
                sharpAndroidLog(line);
            }
            let lastPinchDist = 0;
            const canvas = renderer.domElement;
            canvas.addEventListener('touchstart', function(ev) {
                if (ev.touches.length === 2) {
                    const dx = ev.touches[0].clientX - ev.touches[1].clientX;
                    const dy = ev.touches[0].clientY - ev.touches[1].clientY;
                    lastPinchDist = Math.hypot(dx, dy);
                }
            }, { passive: true });
            canvas.addEventListener('touchmove', function(ev) {
                if (ev.touches.length !== 2 || lastPinchDist <= 0) return;
                const dx = ev.touches[0].clientX - ev.touches[1].clientX;
                const dy = ev.touches[0].clientY - ev.touches[1].clientY;
                const dist = Math.hypot(dx, dy);
                const scale = dist / lastPinchDist;
                lastPinchDist = dist;
                if (Math.abs(scale - 1) < PINCH_THRESHOLD) return;
                ev.preventDefault();
                ev.stopPropagation();
                landscapeDollyAlongView(scale);
            }, { passive: false, capture: true });
            canvas.addEventListener('touchend', function(ev) {
                if (ev.touches.length < 2) lastPinchDist = 0;
            }, { passive: true });
            canvas.addEventListener('wheel', function(ev) {
                ev.preventDefault();
                const scale = ev.deltaY > 0 ? 1.07 : 0.93;
                landscapeDollyAlongView(scale);
            }, { passive: false });
        }

        let splatMesh = null;

        // Auto-frame when mesh has valid bounds (called from onLoad when PLY ready, or by polling)
        let frameAttempts = 0;
        const maxFrameAttempts = 150;  // 150 * 200ms = 30s for large PLY (e.g. 292MB)
        /** Stops duplicate work when multiple setTimeout(autoFrameRoom) chains run (onLoad + initial poll). */
        let framingComplete = false;

        // Load PLY using SparkJS SplatMesh (matching iOS exactly)
        // URL served by WebViewAssetLoader; onLoad runs when PLY is loaded and decoded
        const plyURL = 'https://appassets.androidplatform.net/files/room.ply';
        console.log('[WebGL] Loading splat from:', plyURL);

        try {
            splatMesh = new SplatMesh({
                url: plyURL,
                maxSh: 0,
                onLoad: function(mesh) {
                    console.log('[WebGL] SplatMesh onLoad — scheduling autoFrameRoom');
                    reportStage('spark_mesh_loaded', '');
                    setTimeout(autoFrameRoom, 600);
                }
            });
            // Keep landscape upright while correcting the sensor/viewer left-right flip.
            // Ry=π also flipped depth and Rx=π turned the room upside down, so use X scale only.
            scene.add(splatMesh);
            splatMesh.scale.set(isPortrait ? 1 : -1, 1, 1);
            splatMesh.rotation.set(0, 0, 0);
            console.log('[WebGL] SplatMesh: identity rotation; landscape scaleX=-1');
            if (!trySplatLoadHintBox()) {
                reportStage('hint_unavailable', '');
            }
            setTimeout(autoFrameRoom, 500);
        } catch (err) {
            console.error('[WebGL] Failed to create SplatMesh:', err);
        }

        function giveUpDefaultCamera() {
            framingComplete = true;
            camera.position.set(0, 0, 4);
            controls.target.set(0, 0, 0);
            controls.update();
            needsRender = true;
            reportStage('default_camera_fallback', '');
            if (window.Android) window.Android.onLoaded();
        }

        function trySplatLoadHintBox() {
            if (!hasSplatLoadHint) return false;
            const framingBox = new THREE.Box3(
                new THREE.Vector3(hintFramingMinX, hintFramingMinY, hintFramingMinZ),
                new THREE.Vector3(hintFramingMaxX, hintFramingMaxY, hintFramingMaxZ)
            );
            reportStage('hint_frame_requested', 'centroid=' + hintCenterX.toFixed(3) + ',' + hintCenterY.toFixed(3) + ',' + hintCenterZ.toFixed(3));
            frameFromWorldBox(framingBox, 'splat_load_hint');
            return true;
        }

        function tryMetadataFallbackBox() {
            const metaMax = Math.max(fallbackRoomWidth, fallbackRoomHeight, fallbackRoomDepth);
            if (metaMax < 0.05) return false;
            if (splatMesh) {
                splatMesh.position.set(0, 0, 0);
                splatMesh.updateMatrixWorld(true);
            }
            const cx = fallbackRoomCenterX, cy = fallbackRoomCenterY, cz = fallbackRoomCenterZ;
            const hw = fallbackRoomWidth * 0.5, hh = fallbackRoomHeight * 0.5, hd = fallbackRoomDepth * 0.5;
            const b = new THREE.Box3(
                new THREE.Vector3(cx - hw, cy - hh, cz - hd),
                new THREE.Vector3(cx + hw, cy + hh, cz + hd)
            );
            reportStage('metadata_fallback_box', '');
            frameFromWorldBox(b, 'metadata_fallback');
            return true;
        }

        /**
         * ROOM: splatMesh position stays (0,0,0) + load-time rotation only — we do NOT slide the room for framing.
         * CAMERA: same as iOS SharpRoomView WebGL — outside maxZ (+dist), target maxZ (front wall), portrait and landscape.
         * Zoom / step back along view: change distInFront (metres). Smaller = closer to wall; larger = farther in front.
         */
        function frameFromWorldBox(box, frameSource) {
            if (framingComplete) {
                sharpAndroidLog('[SharpRoom] frameFromWorldBox SKIPPED (framingComplete already true) source=' + frameSource);
                return;
            }
            framingComplete = true; // block duplicate onLoad + poll (recenter clears this first)

            const size = box.getSize(new THREE.Vector3());
            let roomWidth, roomHeight, roomDepth;
            if (isPortrait) {
                roomWidth = size.x;
                roomHeight = size.y;
            } else {
                roomWidth = size.y;
                roomHeight = size.x;
            }
            roomDepth = size.z;

            const maxRealisticWidth = isPortrait ? 5.0 : 8.0;
            const maxRealisticHeight = isPortrait ? 3.5 : 3.2;
            if (roomWidth > maxRealisticWidth) roomWidth = maxRealisticWidth;
            if (roomHeight > maxRealisticHeight) roomHeight = maxRealisticHeight;

            const rawMinX = box.min.x, rawMaxX = box.max.x;
            const rawMinY = box.min.y, rawMaxY = box.max.y;
            const rawMinZ = box.min.z, rawMaxZ = box.max.z;

            const fogFactor = 0.15;
            const shrinkX = roomWidth * fogFactor * 0.5;
            const shrinkY = roomHeight * fogFactor * 0.5;
            // Spark AABB can be paper-thin on Z after rotation (~9mm). Fixed 20mm back inset made minZ > maxZ.
            const zSpanRaw = Math.max(1e-6, rawMaxZ - rawMinZ);
            const shrinkZ = Math.min(roomDepth * fogFactor * 0.5, zSpanRaw * 0.2);
            const backWallInset = Math.min(0.02, zSpanRaw * 0.12);

            const minX = rawMinX + shrinkX;
            const maxX = rawMaxX - shrinkX;
            const minY = rawMinY + shrinkY;
            const maxY = rawMaxY - shrinkY;
            let minZ = rawMinZ + shrinkZ;
            let maxZ = rawMaxZ - backWallInset;
            if (minZ >= maxZ) {
                const midZ = (rawMinZ + rawMaxZ) * 0.5;
                const halfZ = zSpanRaw * 0.45;
                minZ = midZ - halfZ;
                maxZ = midZ + halfZ;
            }

            const innerCenterX = (minX + maxX) / 2;
            const innerCenterY = (minY + maxY) / 2;
            const innerCenterZ = (minZ + maxZ) / 2;

            roomBoundsForClamping = {
                minX, maxX, minY, maxY, minZ, maxZ,
                centerX: innerCenterX, centerY: innerCenterY, centerZ: innerCenterZ
            };

            currentRoomW = roomWidth;
            currentRoomH = roomHeight;
            currentRoomD = roomDepth;

            // Initial camera: thin Z slab (rotated splat AABB) needs distance from floor span, not 9mm depth.
            const thinZSlab = zSpanRaw < 0.08;
            let entranceZ, cameraZ;
            let wallSide;
            let distInFront;
            if (thinZSlab) {
                // Match SharpRoomView.swift when Z span is tiny: still put camera outside maxZ (front wall), any orientation.
                wallSide = isPortrait ? 'thinZ_portrait_maxZ_swift' : 'thinZ_landscape_maxZ_swift';
                const roomSpan = Math.max(roomWidth, roomHeight, fallbackRoomWidth, fallbackRoomHeight);
                distInFront = Math.max(0.75, Math.min(2.0, 0.56 * roomSpan));
                const frontWallZ = maxZ;
                entranceZ = frontWallZ;
                cameraZ = frontWallZ + distInFront;
            } else {
                const FRONT_DIST_K = 0.28;
                const FRONT_DIST_CAP = 1.2;
                const depthForStandoff = Math.max(roomDepth, fallbackRoomDepth, zSpanRaw, 0.15);
                const depthProduct = depthForStandoff * FRONT_DIST_K;
                distInFront = Math.max(0.012, Math.min(depthProduct, FRONT_DIST_CAP));
                wallSide = isPortrait ? 'maxZ_front_swift_portrait' : 'maxZ_front_swift_landscape';
                entranceZ = maxZ;
                cameraZ = maxZ + distInFront;
            }
            const distDbg = '[SharpRoom_DIST] src=' + frameSource + ' thinZ=' + (thinZSlab ? 1 : 0) + ' sizeXYZ=' + size.x.toFixed(3) + ',' + size.y.toFixed(3) + ',' + size.z.toFixed(3) + ' zSpanRaw=' + zSpanRaw.toFixed(4) + ' distInFront=' + distInFront.toFixed(4) + ' minZ=' + minZ.toFixed(4) + ' maxZ=' + maxZ.toFixed(4) + ' innerZ=' + innerCenterZ.toFixed(4) + ' wallSide=' + wallSide + ' targetZ=' + entranceZ.toFixed(4) + ' cameraZ=' + cameraZ.toFixed(4);
            console.log(distDbg);
            sharpAndroidLog(distDbg);

            const newCamPos = new THREE.Vector3(innerCenterX, innerCenterY, cameraZ);
            const newTarget = new THREE.Vector3(innerCenterX, innerCenterY, entranceZ);

            camera.position.copy(newCamPos);
            controls.target.copy(newTarget);
            controls.update();

            initialCameraPosition.copy(camera.position);
            initialControlsTarget.copy(controls.target);

            autoOrbitRadius = camera.position.distanceTo(controls.target);
            autoOrbitBaseAngle = Math.atan2(
                camera.position.x - controls.target.x,
                camera.position.z - controls.target.z
            );

            // Title / room_meta only: do not change mesh rotation. Just send the Box3 footprint to Android; do not
            // upsize “small” rooms or apply defaults here — SHARP / Box3 measurement is the source of truth.
            (function applyDisplayDimsForAndroid() {
                measuredRoomWidth = roomWidth;
                measuredRoomHeight = roomHeight;
                sharpAndroidLog('[SharpRoom] displayDimsForAndroid: raw box3 ' +
                    measuredRoomWidth.toFixed(2) + 'x' + measuredRoomHeight.toFixed(2) +
                    ' zSpan=' + zSpanRaw.toFixed(4));
            })();

            const camPos = camera.position;
            const tgt = controls.target;
            const dist = camPos.distanceTo(tgt);
            console.log('[SharpRoom] CAMERA_FRAME wallSide=' + wallSide + ' source=' + frameSource + ' isPortrait=' + (isPortrait ? 1 : 0) +
                ' roomW=' + roomWidth.toFixed(2) + ' roomH=' + roomHeight.toFixed(2) + ' roomD=' + roomDepth.toFixed(2) +
                ' distance=' + dist.toFixed(2) + ' camPos=' + camPos.x.toFixed(2) + ',' + camPos.y.toFixed(2) + ',' + camPos.z.toFixed(2) +
                ' target=' + tgt.x.toFixed(2) + ',' + tgt.y.toFixed(2) + ',' + tgt.z.toFixed(2) +
                ' innerZ=' + innerCenterZ.toFixed(3) + ' minZ=' + minZ.toFixed(3) + ' maxZ=' + maxZ.toFixed(3));

            console.log('[WebGL] Camera framed wallSide=' + wallSide + ' dist=' + distInFront.toFixed(3));
            cameraFramedAt = performance.now();
            needsRender = true;

            window.sendDimensionsToAndroid = sendDimensionsToAndroid;
            function sendDimensionsToAndroid() {
                if (window.Android && window.Android.onBoxMetricsMeasured) {
                    window.Android.onBoxMetricsMeasured(
                        measuredRoomWidth,
                        measuredRoomHeight,
                        currentRoomD,
                        size.x,
                        size.y,
                        size.z,
                        frameSource,
                        thinZSlab
                    );
                    console.log('[WebGL] Sent box metrics to Android:', measuredRoomWidth.toFixed(2), 'x', measuredRoomHeight.toFixed(2), 'x', currentRoomD.toFixed(2), 'raw=', size.x.toFixed(3) + ',' + size.y.toFixed(3) + ',' + size.z.toFixed(3), 'source=', frameSource, 'thinZ=', thinZSlab ? 1 : 0);
                } else if (window.Android && window.Android.onDimensionsMeasured) {
                    window.Android.onDimensionsMeasured(measuredRoomWidth, measuredRoomHeight);
                    console.log('[WebGL] Sent dimensions to Android:', measuredRoomWidth.toFixed(2), 'x', measuredRoomHeight.toFixed(2));
                }
            }
            sendDimensionsToAndroid();
            setTimeout(sendDimensionsToAndroid, 500);
            setTimeout(sendDimensionsToAndroid, 1500);
            setTimeout(sendDimensionsToAndroid, 3000);

            if (window.Android && window.Android.onSplatLoadHintMeasured) {
                window.Android.onSplatLoadHintMeasured(
                    rawMinX, rawMinY, rawMinZ,
                    rawMaxX, rawMaxY, rawMaxZ,
                    minX, minY, minZ,
                    maxX, maxY, maxZ,
                    innerCenterX, innerCenterY, innerCenterZ,
                    frameSource
                );
            }

            reportStage('frame_complete', frameSource);
            if (window.Android) window.Android.onLoaded();
        }

        function autoFrameRoom(fromRecenter) {
            if (fromRecenter) {
                framingComplete = false;
                frameAttempts = 0;
            }
            if (framingComplete) return;
            frameAttempts++;
            console.log('[WebGL] autoFrameRoom() attempt:', frameAttempts, 'fromRecenter=', !!fromRecenter);

            if (!splatMesh) {
                if (frameAttempts < maxFrameAttempts) {
                    setTimeout(autoFrameRoom, 200);
                } else {
                    console.error('[WebGL] Gave up waiting for splatMesh');
                    tryMetadataFallbackBox() || giveUpDefaultCamera();
                }
                return;
            }

            // Measure PLY at identity position; frameFromWorldBox applies mesh translation for viewer-centric framing.
            splatMesh.position.set(0, 0, 0);
            splatMesh.updateMatrixWorld(true);

            const metaMaxDim = Math.max(fallbackRoomWidth, fallbackRoomHeight, fallbackRoomDepth);

            if (typeof splatMesh.isInitialized === 'boolean' && !splatMesh.isInitialized) {
                if (frameAttempts >= 10 && metaMaxDim > 0.05 && tryMetadataFallbackBox()) return;
                if (frameAttempts < maxFrameAttempts) {
                    console.log('[WebGL] SplatMesh not initialized yet, retry...');
                    setTimeout(autoFrameRoom, 200);
                } else {
                    tryMetadataFallbackBox() || giveUpDefaultCamera();
                }
                return;
            }

            splatMesh.updateMatrixWorld(true);
            let box;
            try {
                const localBox = splatMesh.getBoundingBox(true);
                box = localBox.clone().applyMatrix4(splatMesh.matrixWorld);
            } catch (e) {
                console.warn('[WebGL] getBoundingBox failed:', e);
                if (frameAttempts >= 8 && metaMaxDim > 0.05 && tryMetadataFallbackBox()) return;
                if (frameAttempts < maxFrameAttempts) {
                    setTimeout(autoFrameRoom, 300);
                } else {
                    tryMetadataFallbackBox() || giveUpDefaultCamera();
                }
                return;
            }

            const size = box.getSize(new THREE.Vector3());
            if (size.length() < 0.01) {
                if (frameAttempts >= 8 && metaMaxDim > 0.05 && tryMetadataFallbackBox()) return;
                if (frameAttempts < maxFrameAttempts) {
                    console.log('[WebGL] Box3 size near zero after getBoundingBox, retry...');
                    setTimeout(autoFrameRoom, 200);
                } else {
                    tryMetadataFallbackBox() || giveUpDefaultCamera();
                }
                return;
            }

            reportStage('spark_box_ready', size.x.toFixed(3) + ',' + size.y.toFixed(3) + ',' + size.z.toFixed(3));
            frameFromWorldBox(box, 'spark_getBoundingBox');
        }

        window.autoFrameRoom = autoFrameRoom;

        // Camera controls (called from Android)
        window.orbitCamera = function(deltaX, deltaY) {
            autoOrbitEnabled = false;
            const rotateSpeed = 0.002;   // Slow rotation so touch does not move room too fast
            const offset = new THREE.Vector3().subVectors(camera.position, controls.target);
            const spherical = new THREE.Spherical().setFromVector3(offset);
            spherical.theta -= deltaX * rotateSpeed;
            spherical.phi += deltaY * rotateSpeed;
            spherical.phi = Math.max(0.1, Math.min(Math.PI - 0.1, spherical.phi));
            offset.setFromSpherical(spherical);
            camera.position.copy(controls.target).add(offset);
            controls.update();
            needsRender = true;
        };

        // Joystick: same mapping as SharpRoomView.swift (dx → world X, dy → world Z) + room clamping
        window.moveCamera = function(dx, dy) {
            autoOrbitEnabled = false;
            const moveSpeed = 0.03;
            let newX = camera.position.x + dx * moveSpeed;
            let newZ = camera.position.z + dy * moveSpeed;
            if (roomBoundsForClamping) {
                const marginSide = 0.05;
                const marginBack = 0.02;
                newX = Math.max(roomBoundsForClamping.minX + marginSide,
                    Math.min(roomBoundsForClamping.maxX - marginSide, newX));
                // Start outside maxZ (front-wall view, portrait and landscape); do not clamp Z down to maxZ.
                if (camera.position.z > roomBoundsForClamping.maxZ) {
                    newZ = Math.max(roomBoundsForClamping.minZ + marginSide, newZ);
                } else {
                    newZ = Math.max(roomBoundsForClamping.minZ + marginSide,
                        Math.min(roomBoundsForClamping.maxZ - marginBack, newZ));
                }
            }
            const actualDx = newX - camera.position.x;
            const actualDz = newZ - camera.position.z;
            camera.position.x = newX;
            camera.position.z = newZ;
            controls.target.x += actualDx;
            controls.target.z += actualDz;
            controls.update();
            needsRender = true;
        };

        window.moveCameraUp = function(dy) {
            autoOrbitEnabled = false;
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
            needsRender = true;
        };

        window.recenterCamera = function() {
            if (typeof autoFrameRoom === 'function') {
                autoFrameRoom(true);
            } else {
                camera.position.copy(initialCameraPosition);
                controls.target.copy(initialControlsTarget);
                controls.update();
            }
            needsRender = true;
            setTimeout(function() { needsRender = true; }, 0);
            setTimeout(function() { needsRender = true; }, 50);
        };

        window.addEventListener('resize', () => {
            // Camera aspect matches window (activity is locked to portrait/landscape). Uniform scaling only—no independent X/Y scale.
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
            needsRender = true;
        });

        // Animation loop with warm-up and auto-orbit (matching iOS exactly)
        const clock = new THREE.Clock();
        let lastRenderTime = 0;
        const IDLE_FPS = 30;
        const IDLE_FRAME_TIME = 1000 / IDLE_FPS;

        function animate(currentTime) {
            requestAnimationFrame(animate);

            const dt = clock.getDelta();
            let shouldRender = needsRender;
            needsRender = false;

            // Warm-up period: always render for first 5 seconds
            const elapsed = performance.now() - animationStartTime;
            const inWarmup = elapsed < WARMUP_DURATION;
            if (inWarmup) {
                shouldRender = true;
            }

            // Auto-orbit when enabled and not interacting (skip for 500ms after framing so initial position is visible)
            const orbitCooldown = 500;
            if (autoOrbitEnabled && autoOrbitRadius > 0.1 && (performance.now() - cameraFramedAt) > orbitCooldown) {
                autoOrbitTime += dt;
                const speed = 0.35;
                const t = controls.target;

                if (isPortrait) {
                    const amplitude = Math.PI / 6;
                    const angle = autoOrbitBaseAngle + amplitude * Math.sin(autoOrbitTime * speed);
                    camera.position.x = t.x + autoOrbitRadius * Math.sin(angle);
                    camera.position.z = t.z + autoOrbitRadius * Math.cos(angle);
                } else {
                    const sweepAmount = autoOrbitRadius * 0.3 * Math.sin(autoOrbitTime * speed);
                    camera.position.x = initialCameraPosition.x + sweepAmount;
                    camera.position.z = initialCameraPosition.z;
                }

                if (currentTime - lastRenderTime >= IDLE_FRAME_TIME) {
                    shouldRender = true;
                }
            }

            if (!shouldRender && !inWarmup) {
                return;
            }

            lastRenderTime = currentTime;
            controls.update();

            // Use SparkRenderer's update method for optimized Gaussian rendering
            spark.update({ scene });
            renderer.render(scene, camera);
        }
        animate(0);

        console.log('[WebGL] SparkJS viewer ready');
    </script>
</body>
</html>
        """.trimIndent()
    }

    private fun showSaveDialog() {
        val input = EditText(this).apply {
            hint = getString(R.string.room_viewer_enter_name)
            setPadding(48, 32, 48, 32)
            val folder = roomFolder?.let { File(it) }?.takeIf { it.isDirectory }
            val snapshot = folder?.let { RoomFolderMetadata.readFromFolder(it) }
            // Preview rooms carry an internal "AI Room …" name on disk for list/debug — don't pre-fill it here.
            val initialName = when {
                snapshot?.previewOnly == true -> ""
                else -> snapshot?.name?.takeIf { it.isNotBlank() }.orEmpty()
            }
            setText(initialName)
        }

        val dialog = AlertDialog.Builder(this)
            .setTitle(R.string.room_viewer_save_room)
            .setMessage(R.string.room_viewer_enter_name)
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
                    ModelManager.findAvailableRoomName(this, RoomDisplayName.aiRoomWithTimestamp())
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
        val folderPath = roomFolder
        if (folderPath == null) {
            Toast.makeText(this, getString(R.string.sharp_room_cannot_save), Toast.LENGTH_SHORT).show()
            return
        }
        if (!ModelManager.isRoomNameAvailable(this, name)) {
            Toast.makeText(this, getString(R.string.home_room_name_duplicate), Toast.LENGTH_SHORT).show()
            return
        }

        lifecycleScope.launch {
            var rw = roomWidth
            var rh = roomHeight
            var rd = roomDepth
            LogUtil.i("SHARP_ROOM_MEAS", "saveRoom start name=$name folder=$folderPath W×H×D=$rw×$rh×$rd")

            withContext(Dispatchers.Main) {
                try {
                    val metadataFile = File(folderPath, "metadata.txt")
                    val metadata = StringBuilder()
                    metadata.append("name=$name\n")
                    metadata.append("created=${System.currentTimeMillis()}\n")
                    metadata.append("type=sharp\n")
                    metadata.append("roomWidth=$rw\n")
                    metadata.append("roomHeight=$rh\n")
                    metadata.append("roomDepth=$rd\n")
                    metadata.append("roomCenterX=$roomCenterX\n")
                    metadata.append("roomCenterY=$roomCenterY\n")
                    metadata.append("roomCenterZ=$roomCenterZ\n")
                    metadata.append("photoOrientation=${if (photoOrientation == "landscape") "landscape" else "portrait"}\n")
                    metadata.append("photoWideAngle=$photoWideAngle\n")
                    metadata.append("arDisplayScale=$arDisplayScale\n")
                    metadata.append("previewOnly=false\n")
                    metadataFile.writeText(metadata.toString())
                    val folderFile = File(folderPath)
                    val snapshotToWrite = RoomFolderMetadata.snapshotPreservingYoloFields(
                        folderFile,
                        RoomFolderMetadata.Snapshot(
                            name = name,
                            createdAt = System.currentTimeMillis(),
                            type = "sharp",
                            photoOrientation = if (photoOrientation == "landscape") "landscape" else "portrait",
                            photoWideAngle = photoWideAngle,
                            roomWidth = rw,
                            roomHeight = rh,
                            roomDepth = rd,
                            roomCenterX = roomCenterX,
                            roomCenterY = roomCenterY,
                            roomCenterZ = roomCenterZ,
                            arDisplayScale = arDisplayScale,
                            previewOnly = false,
                        ),
                    )
                    RoomFolderMetadata.writeToFolder(folderFile, snapshotToWrite)

                    Toast.makeText(this@SharpRoomActivity, getString(R.string.sharp_room_saved, name), Toast.LENGTH_SHORT).show()
                    DebugLogger.d(TAG, "Room saved: $name at $folderPath with dims: ${rw}x${rh}x${rd}")
                    hasSavedRoom = true
                    isTempSharpRoom = false

                    val intent = Intent(this@SharpRoomActivity, ContentActivity::class.java)
                    intent.flags = Intent.FLAG_ACTIVITY_CLEAR_TOP or Intent.FLAG_ACTIVITY_NEW_TASK
                    startActivity(intent)
                    finish()
                } catch (e: Exception) {
                    DebugLogger.eDebugMode(TAG, "Failed to save room", e)
                    Toast.makeText(this@SharpRoomActivity, getString(R.string.sharp_room_save_failed, e.message ?: ""), Toast.LENGTH_SHORT).show()
                    CrashReporter.report(this@SharpRoomActivity, e, "Sharp room save")
                }
            }
        }
    }

    private fun recenterCamera() {
        roomDimensionsLockedByTapeCalibration = false
        webView.evaluateJavascript(
            "if(typeof recenterCamera==='function')recenterCamera();",
            null
        )
    }

    private fun sharePlyFile() {
        val plyFile = File(plyPath ?: return)
        if (!plyFile.exists()) {
            Toast.makeText(this, getString(R.string.sharp_room_ply_not_found), Toast.LENGTH_SHORT).show()
            return
        }

        try {
            val uri: android.net.Uri = FileProvider.getUriForFile(
                this,
                "${packageName}.fileprovider",
                plyFile
            )

            val shareIntent = Intent(Intent.ACTION_SEND).apply {
                type = "application/octet-stream"
                putExtra(Intent.EXTRA_STREAM, uri)
                putExtra(Intent.EXTRA_SUBJECT, "3D Room PLY File")
                putExtra(Intent.EXTRA_TEXT, "Check out this 3D room scan!")
                addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            }

            startActivity(Intent.createChooser(shareIntent, "Share PLY File"))
            DebugLogger.d(TAG, "Sharing PLY file: ${plyFile.name}")

        } catch (e: Exception) {
            DebugLogger.eDebugMode(TAG, "Failed to share PLY file", e)
            Toast.makeText(this, getString(R.string.sharp_room_share_failed, e.message ?: ""), Toast.LENGTH_SHORT).show()
            CrashReporter.report(this, e, "Sharp room share PLY")
        }
    }

    // JavaScript interface for communication from WebView
    inner class WebAppInterface {
        @JavascriptInterface
        fun onLoaded() {
            runOnUiThread {
                loadingOverlay.visibility = View.GONE
                DebugLogger.d(TAG, "WebGL viewer reported loaded")
                logSharpLoadTiming("interactive_ready")
            }
        }

        @JavascriptInterface
        fun onSplatLoadStage(stage: String?, detail: String?) {
            logSharpLoadTiming(stage ?: "unknown_stage", detail ?: "")
        }

        @JavascriptInterface
        fun onSplatLoadHintMeasured(
            fullMinX: Float,
            fullMinY: Float,
            fullMinZ: Float,
            fullMaxX: Float,
            fullMaxY: Float,
            fullMaxZ: Float,
            framingMinX: Float,
            framingMinY: Float,
            framingMinZ: Float,
            framingMaxX: Float,
            framingMaxY: Float,
            framingMaxZ: Float,
            centerX: Float,
            centerY: Float,
            centerZ: Float,
            source: String?,
        ) {
            runOnUiThread {
                persistSplatLoadHintFromBounds(
                    fullBoundsMin = SplatLoadHintVector3(fullMinX, fullMinY, fullMinZ),
                    fullBoundsMax = SplatLoadHintVector3(fullMaxX, fullMaxY, fullMaxZ),
                    framingBoundsMin = SplatLoadHintVector3(framingMinX, framingMinY, framingMinZ),
                    framingBoundsMax = SplatLoadHintVector3(framingMaxX, framingMaxY, framingMaxZ),
                    centroid = SplatLoadHintVector3(centerX, centerY, centerZ),
                    source = source ?: "unknown",
                )
            }
        }

        @JavascriptInterface
        fun onDimensionsMeasured(width: Float, height: Float) {
            runOnUiThread {
                if (width <= 0f || height <= 0f) {
                    DebugLogger.d(TAG, "WebGL dimensions measured but non-positive, ignoring: ${width}x${height}")
                    return@runOnUiThread
                }
                if (hasPlausibleOpenSnapshotRoomDims() && !roomDimensionsLockedByTapeCalibration) {
                    roomDimensionsReceivedFromWebGL = true
                    isMeasuringRoomDimensions = false
                    LogUtil.i(
                        "SHARP_ROOM_MEAS",
                        "[box3_measured] ignoring legacy Spark width/height; keeping saved SHARP/export dims ${openSnapshotRoomWidth}×${openSnapshotRoomHeight}×${openSnapshotRoomDepth}",
                    )
                    return@runOnUiThread
                }
                if (roomDimensionsLockedByTapeCalibration) {
                    roomDimensionsReceivedFromWebGL = true
                    isMeasuringRoomDimensions = false
                    LogUtil.i(
                        "SHARP_ROOM_MEAS",
                        "[box3_measured] skip overwrite (tape calibration lock); WebGL offered ${width}×${height}, keeping $roomWidth×$roomHeight",
                    )
                    return@runOnUiThread
                }
                val capW = if (photoOrientation == "landscape") 8f else 5f
                val capH = if (photoOrientation == "landscape") 3.2f else 3.5f
                val minPlausible = 2.0f
                var finalW = width
                var finalH = height
                val snapW = openSnapshotRoomWidth.coerceIn(0.01f, capW)
                val snapH = openSnapshotRoomHeight.coerceIn(0.01f, capH)
                if (width < minPlausible && height < minPlausible &&
                    snapW >= minPlausible && snapH >= minPlausible &&
                    (snapW > width + 0.05f || snapH > height + 0.05f)
                ) {
                    finalW = snapW
                    finalH = snapH
                    LogUtil.i(
                        "SHARP_ROOM_MEAS",
                        "[box3_measured] Kotlin fallback: JS ${width}×${height} -> open snapshot ${finalW}×${finalH} (caps ${capW}×${capH})",
                    )
                }
                val sanitized = SharpRoomDimensionSanitizer.sanitizeMeters(finalW, finalH, roomDepth)
                finalW = sanitized.first
                finalH = sanitized.second
                if (sanitized.third != roomDepth) {
                    roomDepth = sanitized.third
                }
                roomWidth = finalW
                roomHeight = finalH
                roomDimensionsReceivedFromWebGL = true
                isMeasuringRoomDimensions = false
                refreshRoomDimensionsDisplay()
                DebugLogger.d(TAG, "WebGL dimensions applied: ${roomWidth}x${roomHeight} (will persist)")
                LogUtil.i(
                    "SHARP_ROOM_MEAS",
                    "[box3_measured] final W×H=$roomWidth×$roomHeight title_m_label depth=$roomDepth arDisplayScale=$arDisplayScale " +
                        "hasSavedMeta=$hasSavedDimensions folder=$roomFolder",
                )
                reloadPlacementRoomModel()
                persistSparkBoxDimensionsDebounced()
            }
        }

        @JavascriptInterface
        fun onBoxMetricsMeasured(
            width: Float,
            height: Float,
            depth: Float,
            rawSpanX: Float,
            rawSpanY: Float,
            rawSpanZ: Float,
            source: String?,
            thinZ: Boolean
        ) {
            runOnUiThread {
                if (width <= 0f || height <= 0f) {
                    DebugLogger.d(TAG, "WebGL box metrics non-positive, ignoring: ${width}x${height}x${depth}")
                    return@runOnUiThread
                }
                if (hasPlausibleOpenSnapshotRoomDims() && !roomDimensionsLockedByTapeCalibration) {
                    roomDimensionsReceivedFromWebGL = true
                    isMeasuringRoomDimensions = false
                    LogUtil.i(
                        "SHARP_ROOM_MEAS",
                        "[box3_metrics] ignoring Spark metrics; keeping saved SHARP/export dims ${openSnapshotRoomWidth}×${openSnapshotRoomHeight}×${openSnapshotRoomDepth} raw=${rawSpanX}×${rawSpanY}×${rawSpanZ}",
                    )
                    return@runOnUiThread
                }
                if (roomDimensionsLockedByTapeCalibration) {
                    roomDimensionsReceivedFromWebGL = true
                    isMeasuringRoomDimensions = false
                    LogUtil.i(
                        "SHARP_ROOM_MEAS",
                        "[box3_metrics] skip overwrite (tape calibration lock); WebGL offered ${width}×${height}×${depth}, keeping $roomWidth×$roomHeight×$roomDepth",
                    )
                    return@runOnUiThread
                }

                val capW = if (photoOrientation == "landscape") 8f else 5f
                val capH = if (photoOrientation == "landscape") 3.2f else 3.5f
                val snapW = openSnapshotRoomWidth.coerceIn(0.01f, capW)
                val snapH = openSnapshotRoomHeight.coerceIn(0.01f, capH)
                val snapD = openSnapshotRoomDepth.coerceAtLeast(0.01f)
                val minPlausible = 2.0f
                val depthLooksCollapsed = thinZ || (depth in 0.001f..0.08f)
                val widthHeightLookCollapsed =
                    width < minPlausible && height < minPlausible &&
                        snapW >= minPlausible && snapH >= minPlausible &&
                        (snapW > width + 0.05f || snapH > height + 0.05f)

                var finalW = width
                var finalH = height
                var finalD = depth
                if ((depthLooksCollapsed && snapD >= 1.0f) || widthHeightLookCollapsed) {
                    finalW = snapW
                    finalH = snapH
                    finalD = snapD
                    LogUtil.i(
                        "SHARP_ROOM_MEAS",
                        "[box3_metrics] keeping open snapshot due to collapsed Spark AABB: js=${width}×${height}×${depth} raw=${rawSpanX}×${rawSpanY}×${rawSpanZ} thinZ=$thinZ source=${source ?: "unknown"} snapshot=${finalW}×${finalH}×${finalD}",
                    )
                }

                val sanitized = SharpRoomDimensionSanitizer.sanitizeMeters(finalW, finalH, finalD)
                roomWidth = sanitized.first
                roomHeight = sanitized.second
                roomDepth = sanitized.third
                roomDimensionsReceivedFromWebGL = true
                isMeasuringRoomDimensions = false
                refreshRoomDimensionsDisplay()
                DebugLogger.d(
                    TAG,
                    "WebGL box metrics applied: ${roomWidth}x${roomHeight}x${roomDepth} raw=${rawSpanX}x${rawSpanY}x${rawSpanZ} thinZ=$thinZ source=${source ?: "unknown"}"
                )
                LogUtil.i(
                    "SHARP_ROOM_MEAS",
                    "[box3_metrics] final W×H×D=$roomWidth×$roomHeight×$roomDepth rawXYZ=$rawSpanX×$rawSpanY×$rawSpanZ source=${source ?: "unknown"} thinZ=$thinZ arDisplayScale=$arDisplayScale folder=$roomFolder",
                )
                reloadPlacementRoomModel()
                persistSparkBoxDimensionsDebounced()
            }
        }

        @JavascriptInterface
        fun log(message: String) {
            DebugLogger.d(TAG, "WebGL: $message")
        }

        @JavascriptInterface
        fun onGpuRendererInfo(message: String) {
            LogUtil.i(TAG, message)
        }
    }

    override fun onResume() {
        super.onResume()
        brainArController?.onHostResume()
        sharpRoomContentRoot.post { restartTransientGestureHints() }
        val mgr = furnitureFitManager
        if (mgr != null && brainDetectionOverlay.visibility == View.VISIBLE) {
            val wantAr = shouldUseArBrainCamera()
            val hasAr = brainArController != null
            if (wantAr != hasAr) {
                bindBrainCamera(mgr, brainSessionGeneration.get())
            }
            setBrainCalibrationPillVisible(true)
            updateBrainCalibrationPill()
        }
    }

    override fun onPause() {
        brainArController?.onHostPause()
        super.onPause()
    }

    @Deprecated("Deprecated in Java")
    override fun onBackPressed() {
        if (brainDetectionOverlay.visibility == View.VISIBLE) {
            hideBrainDetectionOverlay()
            return
        }
        if (brainProgressOverlay.visibility == View.VISIBLE || brainDetectionOverlay.visibility == View.VISIBLE) {
            stopBrainDetection()
            hideBrainProgressOverlay()
            if (brainDetectionOverlay.visibility == View.VISIBLE) hideBrainDetectionOverlay()
            else {
                brainOverlayVisible = false
                setBrainCalibrationPillVisible(false)
            }
            return
        }
        if (allowSave && !hasSavedRoom) {
            showUnsavedPreviewLeaveDialog()
            return
        }
        if (webView.canGoBack()) {
            webView.goBack()
            return
        }
        super.onBackPressed()
    }

    override fun onDestroy() {
        if (isFinishing) {
            deleteTempSharpRoomIfNeeded()
        }
        stopBrainDetection()
        brainModelWarmupJob?.cancel()
        brainModelWarmupJob = null
        if (!cameraExecutor.isShutdown) {
            cameraExecutor.shutdown()
        }
        furnitureFitManager?.close()
        webView.destroy()
        super.onDestroy()
    }

    /** Delete SHARP room folder when viewer was opened as a temp room and user backed out without saving. */
    private fun deleteTempSharpRoomIfNeeded() {
        if (!isTempSharpRoom || hasSavedRoom) return
        val folderPath = roomFolder ?: return
        try {
            val folder = File(folderPath)
            if (folder.exists()) {
                val ok = folder.deleteRecursively()
                DebugLogger.d(TAG, "Temp Sharp room deleted on exit: $folderPath success=$ok")
            }
        } catch (e: Exception) {
            DebugLogger.eDebugMode(TAG, "Failed to delete temp Sharp room at $folderPath", e)
        }
    }
}
