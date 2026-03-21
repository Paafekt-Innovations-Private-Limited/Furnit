package com.furnit.android

import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.YuvImage
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.os.Handler
import android.os.Looper
import com.furnit.android.utils.LogUtil
import android.view.*
import android.widget.FrameLayout
import android.widget.ImageButton
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.EditText
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import androidx.fragment.app.Fragment
import androidx.lifecycle.lifecycleScope
import com.furnit.android.services.FurnitureFitManager
import com.furnit.android.utils.RoomBoundaryManager
import io.github.sceneview.SceneView
import io.github.sceneview.math.Position
import io.github.sceneview.math.Scale
import io.github.sceneview.node.ModelNode
import kotlinx.coroutines.launch
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileOutputStream
import android.webkit.ConsoleMessage
import android.webkit.WebChromeClient
import android.webkit.WebResourceError
import android.webkit.WebResourceRequest
import android.webkit.WebSettings
import android.webkit.WebView
import android.webkit.WebViewClient
import androidx.webkit.WebViewAssetLoader
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class FurnitureFitFragment : Fragment() {
    private lateinit var previewView: PreviewView
    private lateinit var roomSceneView: SceneView
    private lateinit var overlay: FurnitureFitOverlayView
    private lateinit var statusLabel: TextView
    private lateinit var progressContainer: LinearLayout
    private lateinit var progressBar: android.widget.ProgressBar
    private lateinit var progressLabel: TextView
    private lateinit var cameraExecutor: ExecutorService
    private var cameraProvider: ProcessCameraProvider? = null
    private lateinit var manager: FurnitureFitManager
    private var isProcessing = false
    private var hasFirstDetection = false
    private var showRoomBackground = true  // Show room behind segmented furniture
    private var selectedRoomId: String? = null
    private var selectedRoomName: String? = null
    private var selectedRoomFolder: String? = null  // Absolute path to room folder (so correct room is used as background)
    private var selectedRoomWidth: Float = 4f
    private var selectedRoomHeight: Float = 3f
    private var selectedRoomDepth: Float = 4.5f
    private var selectedPhotoOrientation: String = "portrait"
    private var usePlyBackground: Boolean = false  // True when background is PLY (WebView), false when GLB (SceneView) or none
    private var roomPlyWebView: WebView? = null

    // Tap to calibrate (from Swift): estimated furniture height from detections, user-entered real height, scale factor
    private val defaultRoomHeightMeters = 3.0f
    private var detectedFurnitureHeightMeters: Float? = null
    private var realFurnitureHeightMeters: Float? = null
    private var calibratedRoomHeightMeters: Float? = null
    private var calibrationScaleFactor: Float = 1.0f
    private var roomModelNode: ModelNode? = null
    private var roomBoundaryManager: RoomBoundaryManager? = null
    private var initialCameraSetup: RoomBoundaryManager.CameraSetup? = null
    private var calibrationPillContainer: View? = null
    private var calibrationPillLine1: TextView? = null
    private var calibrationPillLine2: TextView? = null

    // For camera drag (when touching outside furniture)
    private var lastCameraTouchX = 0f
    private var lastCameraTouchY = 0f

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Get room info from arguments (same as SharpRoom so PLY viewer can use same camera framing)
        selectedRoomId = arguments?.getString("ROOM_ID")
        selectedRoomName = arguments?.getString("ROOM_NAME")
        selectedRoomFolder = arguments?.getString("ROOM_FOLDER")
        selectedRoomWidth = arguments?.getFloat("ROOM_WIDTH") ?: 4f
        selectedRoomHeight = arguments?.getFloat("ROOM_HEIGHT") ?: 3f
        selectedRoomDepth = arguments?.getFloat("ROOM_DEPTH") ?: 4.5f
        selectedPhotoOrientation = if (arguments?.getString("PHOTO_ORIENTATION")?.trim()?.lowercase() == "landscape") "landscape" else "portrait"
        LogUtil.d("FurnitureFit", "Fragment onCreate - ROOM_NAME=$selectedRoomName ROOM_ID=$selectedRoomId ROOM_FOLDER=$selectedRoomFolder dims=${selectedRoomWidth}x${selectedRoomHeight}x${selectedRoomDepth} orientation=$selectedPhotoOrientation")
        cameraExecutor = Executors.newSingleThreadExecutor()
        manager = FurnitureFitManager(requireContext())
        // Initialize model - try NCNN first (1280x1280, more efficient), fall back to ONNX
        LogUtil.d("FurnitureFit", "Calling initializeAuto...")
        val success = manager.initializeAuto()
        LogUtil.d("FurnitureFit", "initializeAuto completed, success=$success")
    }

    override fun onCreateView(inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?): View {
        val root = FrameLayout(requireContext())

        previewView = PreviewView(requireContext()).apply {
            layoutParams = FrameLayout.LayoutParams(FrameLayout.LayoutParams.MATCH_PARENT, FrameLayout.LayoutParams.MATCH_PARENT)
            implementationMode = PreviewView.ImplementationMode.COMPATIBLE
        }

        // Room background layer - SceneView for GLB, or WebView for PLY (splat)
        roomSceneView = SceneView(requireContext()).apply {
            layoutParams = FrameLayout.LayoutParams(FrameLayout.LayoutParams.MATCH_PARENT, FrameLayout.LayoutParams.MATCH_PARENT)
            visibility = View.GONE
        }
        roomPlyWebView = WebView(requireContext()).apply {
            layoutParams = FrameLayout.LayoutParams(FrameLayout.LayoutParams.MATCH_PARENT, FrameLayout.LayoutParams.MATCH_PARENT)
            visibility = View.GONE
            settings.javaScriptEnabled = true
            settings.domStorageEnabled = true
            setBackgroundColor(Color.parseColor("#808080"))
            webChromeClient = object : WebChromeClient() {
                override fun onConsoleMessage(msg: ConsoleMessage?): Boolean {
                    msg?.let { LogUtil.d("FurnitureFit", "PLY WebView: ${it.message()}") }
                    return true
                }
            }
        }
        loadRoom3D()

        overlay = FurnitureFitOverlayView(requireContext()).apply {
            layoutParams = FrameLayout.LayoutParams(FrameLayout.LayoutParams.MATCH_PARENT, FrameLayout.LayoutParams.MATCH_PARENT)
            // Handle camera drag when touching outside furniture
            onTouchOutsideFurniture = { event ->
                handleCameraDrag(event)
            }
        }

        statusLabel = TextView(requireContext()).apply {
            text = getString(R.string.smartypants_initializing)
            setTextColor(0xFFFFFFFF.toInt())
            setShadowLayer(2f, 1f, 1f, 0xFF000000.toInt())
            setPadding(24, 48, 24, 24)
            textSize = 14f
        }

        // Progress container (like iOS progressContainer)
        progressContainer = LinearLayout(requireContext()).apply {
            orientation = LinearLayout.VERTICAL
            gravity = Gravity.CENTER
            setBackgroundColor(0x99000000.toInt())
            setPadding(32, 16, 32, 16)
            val lp = FrameLayout.LayoutParams(FrameLayout.LayoutParams.WRAP_CONTENT, FrameLayout.LayoutParams.WRAP_CONTENT)
            lp.gravity = Gravity.CENTER_HORIZONTAL or Gravity.TOP
            lp.setMargins(0, 120, 0, 0)
            layoutParams = lp
        }

        progressLabel = TextView(requireContext()).apply {
            text = getString(R.string.smartypants_starting)
            setTextColor(0xFFFFFFFF.toInt())
            textSize = 14f
            gravity = Gravity.CENTER
            setPadding(8, 4, 8, 8)
        }
        progressContainer.addView(progressLabel)

        progressBar = android.widget.ProgressBar(requireContext(), null, android.R.attr.progressBarStyleHorizontal).apply {
            val lp = LinearLayout.LayoutParams(250.dp, LinearLayout.LayoutParams.WRAP_CONTENT)
            layoutParams = lp
            max = 100
            progress = 5
            progressDrawable.setColorFilter(0xFF4CAF50.toInt(), android.graphics.PorterDuff.Mode.SRC_IN)
        }
        progressContainer.addView(progressBar)

        // Back button
        val backButton = ImageButton(requireContext()).apply {
            setImageResource(android.R.drawable.ic_menu_close_clear_cancel)
            setBackgroundColor(0x80000000.toInt())
            setPadding(16, 16, 16, 16)
            val lp = FrameLayout.LayoutParams(FrameLayout.LayoutParams.WRAP_CONTENT, FrameLayout.LayoutParams.WRAP_CONTENT)
            lp.gravity = Gravity.TOP or Gravity.START
            lp.setMargins(16, 48, 0, 0)
            layoutParams = lp
            setOnClickListener { activity?.finish() }
        }

        // Touch layer - passes all events to overlay for furniture manipulation
        // Single finger: drag furniture, Two fingers: pinch to scale furniture
        val touchLayer = View(requireContext()).apply {
            layoutParams = FrameLayout.LayoutParams(FrameLayout.LayoutParams.MATCH_PARENT, FrameLayout.LayoutParams.MATCH_PARENT)
            setOnTouchListener { _, event ->
                // Pass all events to overlay for furniture drag and pinch-to-zoom
                this@FurnitureFitFragment.overlay.handleExternalTouchEvent(event)
                true
            }
        }

        // Bottom controls container
        val bottomControls = FrameLayout(requireContext()).apply {
            val lp = FrameLayout.LayoutParams(FrameLayout.LayoutParams.MATCH_PARENT, FrameLayout.LayoutParams.WRAP_CONTENT)
            lp.gravity = Gravity.BOTTOM
            lp.setMargins(20, 0, 20, 24)
            layoutParams = lp
        }

        // Hint label (center) - shows drag instruction
        val hintLabel = TextView(requireContext()).apply {
            text = getString(R.string.smartypants_drag_hint)
            setTextColor(0xAAFFFFFF.toInt())
            textSize = 12f
            setShadowLayer(2f, 1f, 1f, 0xFF000000.toInt())
            val lp = FrameLayout.LayoutParams(FrameLayout.LayoutParams.WRAP_CONTENT, FrameLayout.LayoutParams.WRAP_CONTENT)
            lp.gravity = Gravity.CENTER_HORIZONTAL or Gravity.BOTTOM
            lp.setMargins(0, 0, 0, 8)
            layoutParams = lp
        }
        bottomControls.addView(hintLabel)

        // Tap to calibrate pill (center-bottom): Furn X.XXm / Tap to calibrate; Room X.XXm when calibrated
        val pillContent = LinearLayout(requireContext()).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(24.dp, 12.dp, 24.dp, 12.dp)
            gravity = Gravity.CENTER
            background = android.graphics.drawable.GradientDrawable().apply {
                setColor(0xE6333333.toInt())
                cornerRadius = (24 * resources.displayMetrics.density)
            }
        }
        calibrationPillLine1 = TextView(requireContext()).apply {
            setTextColor(0xFFFFFFFF.toInt())
            textSize = 14f
            setShadowLayer(2f, 1f, 1f, 0xFF000000.toInt())
        }
        calibrationPillLine2 = TextView(requireContext()).apply {
            text = getString(R.string.smartypants_tap_calibrate)
            setTextColor(0xAAFFFFFF.toInt())
            textSize = 12f
            setShadowLayer(2f, 1f, 1f, 0xFF000000.toInt())
        }
        pillContent.addView(calibrationPillLine1)
        pillContent.addView(calibrationPillLine2)
        val calibrationPill = FrameLayout(requireContext()).apply {
            layoutParams = FrameLayout.LayoutParams(FrameLayout.LayoutParams.WRAP_CONTENT, FrameLayout.LayoutParams.WRAP_CONTENT).apply {
                gravity = Gravity.CENTER_HORIZONTAL or Gravity.BOTTOM
                setMargins(0, 0, 0, 52)
            }
            addView(pillContent)
            visibility = View.GONE
            setOnClickListener { showCalibrationDialog() }
        }
        calibrationPillContainer = calibrationPill
        bottomControls.addView(calibrationPill)

        // Screenshot button (right)
        val screenshotButton = ImageButton(requireContext()).apply {
            setImageResource(android.R.drawable.ic_menu_camera)
            setBackgroundResource(android.R.drawable.btn_default)
            val size = (56 * resources.displayMetrics.density).toInt()
            val lp = FrameLayout.LayoutParams(size, size)
            lp.gravity = Gravity.END or Gravity.BOTTOM
            layoutParams = lp
            setOnClickListener { takeScreenshot(root) }
        }
        bottomControls.addView(screenshotButton)

        // When opened from brain (room background): hide camera feed and show room from start so progress bar overlays room (not grey)
        val hasRoomBackground = selectedRoomId != null || !selectedRoomFolder.isNullOrBlank()
        if (hasRoomBackground) {
            previewView.visibility = View.GONE
            // Show room layer immediately so progress bar overlays the room; PLY will load into this WebView
            roomPlyWebView?.visibility = View.VISIBLE
        }
        root.addView(previewView)
        root.addView(roomSceneView)
        roomPlyWebView?.let { root.addView(it) }  // PLY splat background (shown when no room.glb)
        root.addView(overlay)
        root.addView(touchLayer)     // Touch-anywhere drag layer
        root.addView(statusLabel)
        root.addView(progressContainer)
        root.addView(backButton)
        root.addView(bottomControls)

        // Initial progress
        setProgress(5, "Starting camera...")

        startCamera()
        return root
    }

    private fun handleCameraDrag(event: MotionEvent) {
        LogUtil.d("FurnitureFit", "handleCameraDrag called, action=${event.actionMasked}, roomVisible=${roomSceneView.visibility == View.VISIBLE}")
        if (roomSceneView.visibility != View.VISIBLE) return

        when (event.actionMasked) {
            MotionEvent.ACTION_DOWN -> {
                lastCameraTouchX = event.x
                lastCameraTouchY = event.y
                LogUtil.d("FurnitureFit", "Camera drag DOWN at (${event.x}, ${event.y})")
            }
            MotionEvent.ACTION_MOVE -> {
                val deltaX = event.x - lastCameraTouchX
                val deltaY = event.y - lastCameraTouchY

                // Convert screen pixels to camera movement
                val sensitivity = 0.01f
                val camera = roomSceneView.cameraNode
                val position = camera.position

                val newX = position.x - deltaX * sensitivity
                val newZ = position.z - deltaY * sensitivity
                LogUtil.d("FurnitureFit", "Camera drag MOVE: delta=($deltaX, $deltaY), newPos=($newX, ${position.y}, $newZ)")

                camera.position = Position(newX, position.y, newZ)

                lastCameraTouchX = event.x
                lastCameraTouchY = event.y
            }
        }
    }

    private fun takeScreenshot(rootView: View) {
        try {
            // First capture the 3D room using PixelCopy (for OpenGL content)
            val roomBitmap = Bitmap.createBitmap(roomSceneView.width, roomSceneView.height, Bitmap.Config.ARGB_8888)

            android.view.PixelCopy.request(
                roomSceneView,
                roomBitmap,
                { copyResult ->
                    if (copyResult == android.view.PixelCopy.SUCCESS) {
                        // Now composite the overlay on top of the room
                        val compositeBitmap = Bitmap.createBitmap(rootView.width, rootView.height, Bitmap.Config.ARGB_8888)
                        val canvas = Canvas(compositeBitmap)

                        // Draw room background (scaled to fit)
                        canvas.drawBitmap(roomBitmap, null, android.graphics.RectF(0f, 0f, rootView.width.toFloat(), rootView.height.toFloat()), null)

                        // Draw overlay on top
                        overlay.draw(canvas)

                        // Save the composite
                        saveScreenshotToGallery(compositeBitmap)
                    } else {
                        // Fallback: just capture the overlay with black background
                        activity?.runOnUiThread {
                            val fallbackBitmap = Bitmap.createBitmap(rootView.width, rootView.height, Bitmap.Config.ARGB_8888)
                            val canvas = Canvas(fallbackBitmap)
                            canvas.drawColor(Color.BLACK)
                            overlay.draw(canvas)
                            saveScreenshotToGallery(fallbackBitmap)
                        }
                    }
                },
                Handler(Looper.getMainLooper())
            )
        } catch (e: Exception) {
            LogUtil.e("FurnitureFit", "Screenshot failed", e)
            Toast.makeText(requireContext(), getString(R.string.smartypants_screenshot_failed, e.message ?: ""), Toast.LENGTH_SHORT).show()
        }
    }

    private fun saveScreenshotToGallery(bitmap: Bitmap) {
        try {
            val timeStamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())
            val fileName = "FurnitureFit_$timeStamp.png"

            // Save to gallery using MediaStore (Android 10+)
            val contentValues = android.content.ContentValues().apply {
                put(android.provider.MediaStore.Images.Media.DISPLAY_NAME, fileName)
                put(android.provider.MediaStore.Images.Media.MIME_TYPE, "image/png")
                put(android.provider.MediaStore.Images.Media.RELATIVE_PATH, "${Environment.DIRECTORY_PICTURES}/Screenshots")
            }

            val resolver = requireContext().contentResolver
            val uri = resolver.insert(android.provider.MediaStore.Images.Media.EXTERNAL_CONTENT_URI, contentValues)

            if (uri != null) {
                resolver.openOutputStream(uri)?.use { out ->
                    bitmap.compress(Bitmap.CompressFormat.PNG, 100, out)
                }

                activity?.runOnUiThread {
                    Toast.makeText(requireContext(), getString(R.string.smartypants_saved_screenshots), Toast.LENGTH_SHORT).show()
                }

                // Share the screenshot
                val shareIntent = Intent(Intent.ACTION_SEND).apply {
                    type = "image/png"
                    putExtra(Intent.EXTRA_STREAM, uri)
                    addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                }
                startActivity(Intent.createChooser(shareIntent, "Share Screenshot"))
            } else {
                activity?.runOnUiThread {
                    Toast.makeText(requireContext(), getString(R.string.smartypants_failed_save_screenshot), Toast.LENGTH_SHORT).show()
                }
            }
        } catch (e: Exception) {
            LogUtil.e("FurnitureFit", "Save screenshot failed", e)
            activity?.runOnUiThread {
                Toast.makeText(requireContext(), getString(R.string.smartypants_failed_save, e.message ?: ""), Toast.LENGTH_SHORT).show()
            }
        }
    }

    private fun startCamera() {
        val camProviderFuture = ProcessCameraProvider.getInstance(requireContext())
        camProviderFuture.addListener({
            cameraProvider = camProviderFuture.get()
            bindCameraUseCases()
        }, ContextCompat.getMainExecutor(requireContext()))
    }

    private fun bindCameraUseCases() {
        val cameraProvider = cameraProvider ?: return
        cameraProvider.unbindAll()

        val rotation = requireContext().displayRotationForCameraX()
        val analysis = ImageAnalysis.Builder()
            .setTargetResolution(android.util.Size(768, 768))
            .setTargetRotation(rotation)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build()

        analysis.setAnalyzer(cameraExecutor) { imageProxy ->
            processFrame(imageProxy)
        }

        val hasRoomBackground = selectedRoomId != null || !selectedRoomFolder.isNullOrBlank()

        try {
            if (hasRoomBackground) {
                // Brain flow: only analysis (segmentation), no Preview – user sees progress bar only
                cameraProvider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, analysis)
            } else {
                // No room: show live camera + analysis
                val preview = Preview.Builder()
                    .setTargetRotation(rotation)
                    .build()
                    .also {
                        it.setSurfaceProvider(previewView.surfaceProvider)
                    }
                cameraProvider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis)
            }
            activity?.runOnUiThread {
                statusLabel.text = if (hasRoomBackground) getString(R.string.smartypants_detecting_furniture) else getString(R.string.smartypants_camera_ready)
            }
        } catch (e: Exception) {
            LogUtil.e("FurnitureFit", "bindToLifecycle failed", e)
            activity?.runOnUiThread {
                statusLabel.text = getString(R.string.smartypants_camera_error, e.message ?: "")
            }
        }
    }

    private fun processFrame(imageProxy: ImageProxy) {
        if (isProcessing) {
            imageProxy.close()
            return
        }
        isProcessing = true

        val bitmap = imageProxy.toBitmapSafe()
        if (bitmap == null) {
            LogUtil.w("FurnitureFit", "Failed to convert imageProxy to bitmap")
            isProcessing = false
            imageProxy.close()
            return
        }

        // Update progress during processing
        if (!hasFirstDetection) {
            setProgress(15, getString(R.string.smartypants_preprocessing))
        }

        manager.segmentWithDetectionsAsync(bitmap) { result ->
            activity?.runOnUiThread {
                if (result != null && result.mask != null) {
                    // First detection - hide progress bar
                    if (!hasFirstDetection) {
                        hasFirstDetection = true
                        progressContainer.visibility = View.GONE
                    }

                    // Show what was detected
                    val labels = result.detections.take(3).joinToString(", ") {
                        "${it.label} ${(it.confidence * 100).toInt()}%"
                    }
                    statusLabel.text = if (labels.isNotEmpty()) labels else getString(R.string.smartypants_detected)
                    overlay.setMaskAndDetections(result.mask, result.detections, result.inputSize)

                    // Estimated furniture height in meters from largest detection (for Tap to calibrate)
                    val inputSize = result.inputSize
                    if (result.detections.isNotEmpty() && inputSize > 0) {
                        val maxH = result.detections.maxOf { it.h }
                        detectedFurnitureHeightMeters = (maxH / inputSize) * defaultRoomHeightMeters
                        updateCalibrationPill()
                    } else {
                        detectedFurnitureHeightMeters = null
                        updateCalibrationPill()
                    }

                    // Show 3D room (GLB or PLY) behind segmented furniture, hide camera preview.
                    // Show room on first valid segmentation result (mask present), even if no detections yet, so the room is visible and segmentation can continue.
                    if (showRoomBackground) {
                        if (usePlyBackground) {
                            roomPlyWebView?.visibility = View.VISIBLE
                            roomSceneView.visibility = View.GONE
                        } else {
                            roomSceneView.visibility = View.VISIBLE
                            roomPlyWebView?.visibility = View.GONE
                        }
                        previewView.visibility = View.GONE
                    }
                } else {
                    if (!hasFirstDetection) {
                        setProgress(40, getString(R.string.smartypants_scanning_for_furniture))
                    }
                    statusLabel.text = getString(R.string.smartypants_scanning)
                    overlay.setMaskAndDetections(null, emptyList())
                    roomSceneView.visibility = View.GONE
                    roomPlyWebView?.visibility = View.GONE
                    previewView.visibility = View.VISIBLE
                }
            }
            isProcessing = false
        }

        imageProxy.close()
    }

    private fun setProgress(value: Int, text: String) {
        if (hasFirstDetection) return
        activity?.runOnUiThread {
            progressContainer.visibility = View.VISIBLE
            progressBar.progress = value
            progressLabel.text = text
        }
    }

    private fun updateCalibrationPill() {
        activity?.runOnUiThread {
            val detected = detectedFurnitureHeightMeters
            calibrationPillContainer?.visibility = if (detected != null) View.VISIBLE else View.GONE
            if (detected != null) {
                val roomM = calibratedRoomHeightMeters
                calibrationPillLine1?.text = if (roomM != null) "Room: ${String.format(Locale.US, "%.2f", roomM)}m" else "Furn: ${String.format(Locale.US, "%.2f", detected)}m"
                calibrationPillLine1?.setTextColor(if (roomM != null) 0xFF4CAF50.toInt() else 0xFFFFFFFF.toInt())
                calibrationPillLine2?.text = getString(R.string.smartypants_tap_calibrate)
            }
        }
    }

    private fun showCalibrationDialog() {
        val detected = detectedFurnitureHeightMeters ?: return
        val ctx = requireContext()
        val edit = EditText(ctx).apply {
            setHint(getString(R.string.smartypants_real_height_hint))
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
                calibrationScaleFactor = real / detected
                realFurnitureHeightMeters = real
                calibratedRoomHeightMeters = defaultRoomHeightMeters * calibrationScaleFactor
                applyScaleToRoom()
                updateCalibrationPill()
                Toast.makeText(ctx, getString(R.string.smartypants_room_scaled, String.format(Locale.US, "%.2f", calibrationScaleFactor)), Toast.LENGTH_SHORT).show()
            }
            .show()
    }

    /**
     * Apply calibration scale to room and adjust camera position so the room stays framed
     * (same idea as Swift: scaleRoom then autoFrameRoom — scale mesh then re-frame camera).
     */
    private fun applyScaleToRoom() {
        val scale = calibrationScaleFactor
        roomModelNode?.let { node ->
            node.scale = Scale(scale, scale, scale)
        }
        val setup = initialCameraSetup
        if (setup != null) {
            roomSceneView.cameraNode.apply {
                position = Position(setup.position.x * scale, setup.position.y * scale, setup.position.z * scale)
                lookAt(Position(setup.lookAt.x * scale, setup.lookAt.y * scale, setup.lookAt.z * scale))
            }
        } else {
            val baseDistance = 4f
            roomSceneView.cameraNode.apply {
                position = Position(0f, 1.6f, baseDistance * scale)
                lookAt(Position(0f, 1.4f, -baseDistance * scale))
            }
        }
        // PLY WebView: scaling + camera reframe could be added via JS if the viewer supports it
    }

    private val Int.dp: Int
        get() = (this * resources.displayMetrics.density).toInt()

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        manager.close()
    }

    /**
     * Load 3D room background for furniture segmentation (brain icon).
     * Uses room.glb in SceneView when present; uses room.ply (Gaussian splat) in a WebView when the folder has no room.glb.
     * Bundled rooms (vintage, cozy_room) or rooms/ and sharp_rooms/ by ROOM_ID use room.glb when present.
     * No fallback: if there is no room.glb and no room.ply, no background is shown.
     */
    private fun loadRoom3D() {
        val roomId = selectedRoomId
        val roomFolderPath = selectedRoomFolder
        LogUtil.d("FurnitureFit", "loadRoom3D: ROOM_ID=$roomId ROOM_FOLDER=$roomFolderPath")

        lifecycleScope.launch {
            try {
                val modelInstance = when {
                    // Bundled rooms in assets
                    roomId == "vintage" || roomId == "cozy_room" -> {
                        val assetPath = "models/$roomId.glb"
                        LogUtil.d("FurnitureFit", "Using assets: $assetPath")
                        roomSceneView.modelLoader.createModelInstance(assetFileLocation = assetPath)
                    }
                    // Explicit room folder – use room.glb if present, else room.ply (splat) as WebView background
                    roomFolderPath != null && roomFolderPath.isNotBlank() -> {
                        val folder = java.io.File(roomFolderPath)
                        val glbFile = java.io.File(folder, "room.glb")
                        val plyFile = java.io.File(folder, "room.ply")
                        if (glbFile.exists()) {
                            LogUtil.d("FurnitureFit", "Using opened folder room.glb: ${glbFile.absolutePath}")
                            val bytes = glbFile.readBytes()
                            val buffer = java.nio.ByteBuffer.wrap(bytes)
                            roomSceneView.modelLoader.createModelInstance(buffer)
                        } else if (plyFile.exists()) {
                            LogUtil.d("FurnitureFit", "Using opened folder room.ply as PLY background: ${plyFile.absolutePath}")
                            usePlyBackground = true
                            loadPlyBackground(plyFile)
                            activity?.runOnUiThread {
                                roomPlyWebView?.visibility = View.VISIBLE
                            }
                            null
                        } else {
                            LogUtil.d("FurnitureFit", "No room.glb or room.ply in $roomFolderPath; no 3D background")
                            null
                        }
                    }
                    // Look up by ROOM_ID in rooms/ and sharp_rooms/
                    roomId != null -> {
                        val filesDir = requireContext().filesDir
                        val glbInRooms = java.io.File(java.io.File(filesDir, "rooms"), roomId).let { java.io.File(it, "room.glb") }
                        val glbInSharp = java.io.File(java.io.File(filesDir, "sharp_rooms"), roomId).let { java.io.File(it, "room.glb") }
                        val glbFile = when {
                            glbInRooms.exists() -> glbInRooms
                            glbInSharp.exists() -> glbInSharp
                            else -> null
                        }
                        if (glbFile != null) {
                            LogUtil.d("FurnitureFit", "Using rooms/sharp_rooms by id: ${glbFile.absolutePath}")
                            val bytes = glbFile.readBytes()
                            val buffer = java.nio.ByteBuffer.wrap(bytes)
                            roomSceneView.modelLoader.createModelInstance(buffer)
                        } else {
                            LogUtil.d("FurnitureFit", "No room.glb for id=$roomId; no 3D background")
                            null
                        }
                    }
                    else -> {
                        LogUtil.d("FurnitureFit", "No ROOM_ID or ROOM_FOLDER; no 3D background")
                        null
                    }
                }

                if (modelInstance != null) {
                    val modelNode = ModelNode(
                        modelInstance = modelInstance,
                        scaleToUnits = null
                    )
                    roomModelNode = modelNode
                    roomSceneView.addChildNode(modelNode)

                    // Center room at origin and position camera inside room facing front wall (match ModelDetailActivity / Swift)
                    val bboxCenter = modelNode.center
                    val bboxExtents = modelNode.extents
                    LogUtil.d("FurnitureFit", "Room bbox center=(${bboxCenter.x}, ${bboxCenter.y}, ${bboxCenter.z}) extents=(${bboxExtents.x}, ${bboxExtents.y}, ${bboxExtents.z})")
                    modelNode.position = Position(-bboxCenter.x, -bboxCenter.y, -bboxCenter.z)

                    val boundaryManager = RoomBoundaryManager()
                    roomBoundaryManager = boundaryManager
                    val w = bboxExtents.x
                    val h = bboxExtents.y
                    val d = bboxExtents.z
                    boundaryManager.initializeFromDimensions(width = w, depth = d, height = h)
                    LogUtil.d("FurnitureFit", "[FurnitureFit] getCameraCenteredView CALLED (bbox ${w}x${h}x${d})")
                    val cameraSetup = boundaryManager.getCameraCenteredView()
                    initialCameraSetup = cameraSetup
                    roomSceneView.cameraNode.apply {
                        position = cameraSetup.position
                        lookAt(cameraSetup.lookAt)
                    }
                    roomSceneView.post {
                        roomSceneView.cameraNode.apply {
                            position = cameraSetup.position
                            lookAt(cameraSetup.lookAt)
                        }
                    }
                    LogUtil.d("FurnitureFit", "[FurnitureFit] camera SET pos=(${cameraSetup.position.x}, ${cameraSetup.position.y}, ${cameraSetup.position.z}) lookAt=(${cameraSetup.lookAt.x}, ${cameraSetup.lookAt.y}, ${cameraSetup.lookAt.z})")
                    if (roomFolderPath != null || roomId != null) {
                        activity?.runOnUiThread {
                            roomSceneView.visibility = View.VISIBLE
                            roomPlyWebView?.visibility = View.GONE
                        }
                    }
                } else {
                    roomSceneView.cameraNode.apply {
                        position = Position(0f, 1.6f, 4f)
                        lookAt(Position(0f, 1.4f, -4f))
                    }
                    initialCameraSetup = null
                    LogUtil.d("FurnitureFit", "No 3D room; camera ready for segmentation")
                }
            } catch (e: Exception) {
                LogUtil.e("FurnitureFit", "Failed to load 3D room", e)
            }
        }
    }

    /** Load PLY (Gaussian splat) as background via WebView + SparkJS. Same approach as SharpRoomActivity. */
    private fun loadPlyBackground(plyFile: File) {
        val webView = roomPlyWebView ?: return
        val ctx = requireContext()
        val internalDir = File(ctx.filesDir, "webview_assets")
        internalDir.mkdirs()
        val destPly = File(internalDir, "room.ply")
        try {
            plyFile.copyTo(destPly, overwrite = true)
            LogUtil.d("FurnitureFit", "Copied PLY for background: ${destPly.absolutePath}")
        } catch (e: Exception) {
            LogUtil.e("FurnitureFit", "Failed to copy PLY for background", e)
            return
        }
        val assetLoader = WebViewAssetLoader.Builder()
            .addPathHandler("/assets/", WebViewAssetLoader.AssetsPathHandler(ctx))
            .addPathHandler("/files/", WebViewAssetLoader.InternalStoragePathHandler(ctx, internalDir))
            .build()
        webView.webViewClient = object : WebViewClient() {
            override fun shouldInterceptRequest(
                view: WebView?,
                request: android.webkit.WebResourceRequest?
            ): android.webkit.WebResourceResponse? {
                val url = request?.url ?: return null
                return assetLoader.shouldInterceptRequest(url)
            }
        }
        val isPortrait = selectedPhotoOrientation != "landscape"
        val html = buildPlyViewerHtml(
            isPortrait = isPortrait,
            roomWidth = selectedRoomWidth,
            roomHeight = selectedRoomHeight,
            roomDepth = selectedRoomDepth
        )
        webView.loadDataWithBaseURL(
            "https://appassets.androidplatform.net/",
            html,
            "text/html",
            "UTF-8",
            null
        )
    }

    /** Minimal SparkJS splat viewer for PLY background with a centered, straight-on front-wall camera. */
    private fun buildPlyViewerHtml(isPortrait: Boolean, roomWidth: Float, roomHeight: Float, roomDepth: Float): String {
        val w = roomWidth.toDouble()
        val h = roomHeight.toDouble()
        val d = roomDepth.toDouble()
        return """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>* { margin: 0; padding: 0; } html, body { width: 100%; height: 100%; overflow: hidden; background: #808080; } canvas { display: block; width: 100%; height: 100%; }</style>
    <script type="importmap">
    {"imports":{"three":"https://cdnjs.cloudflare.com/ajax/libs/three.js/0.170.0/three.module.min.js","three/addons/":"https://cdn.jsdelivr.net/npm/three@0.170.0/examples/jsm/","@sparkjsdev/spark":"https://sparkjs.dev/releases/spark/0.1.10/spark.module.js"}}
    </script>
</head>
<body>
<script type="module">
import * as THREE from 'three';
import { SplatMesh, SparkRenderer } from '@sparkjsdev/spark';
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x808080);
const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.up.set(0, 1, 0);
const renderer = new THREE.WebGLRenderer({ antialias: false });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
document.body.appendChild(renderer.domElement);
const spark = new SparkRenderer({ renderer: renderer, maxStdDev: 3.0, preBlurAmount: 0.5, blurAmount: 0.3, falloff: 0.8, focalAdjustment: 1.5 });
camera.add(spark);
const isPortrait = ${isPortrait};
const fallbackW = $w;
const fallbackH = $h;
const fallbackD = $d;
if (isPortrait) {
    const frontWall = -fallbackD * 0.5;
    camera.position.set(0, 0, 0);
    camera.lookAt(new THREE.Vector3(frontWall, 0, 0));
} else {
    const frontWallZ = -fallbackD * 0.5;
    camera.position.set(0, 0, 0);
    camera.lookAt(new THREE.Vector3(0, 0, frontWallZ));
}
const plyURL = 'https://appassets.androidplatform.net/files/room.ply';
const splatMesh = new SplatMesh({ url: plyURL, maxSh: 0 });
if (isPortrait) splatMesh.rotation.y = Math.PI / 2;
scene.add(splatMesh);
function animate() {
    requestAnimationFrame(animate);
    spark.update({ scene });
    renderer.render(scene, camera);
}
window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});
animate();
</script>
</body>
</html>
        """.trimIndent()
    }
}

/**
 * Current [android.view.Display.getRotation] as [Surface.ROTATION_*] for CameraX
 * [androidx.camera.core.ImageAnalysis.Builder.setTargetRotation] /
 * [androidx.camera.core.Preview.Builder.setTargetRotation].
 * Without this, buffers stay sensor-native (often landscape) while the UI is portrait → 90° tilt
 * after [ImageProxy.toBitmapSafe] rotation metadata can be wrong vs locked activity orientation.
 */
fun Context.displayRotationForCameraX(): Int {
    return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
        display?.rotation ?: Surface.ROTATION_0
    } else {
        @Suppress("DEPRECATION")
        (getSystemService(Context.WINDOW_SERVICE) as WindowManager).defaultDisplay.rotation
    }
}

// Convert ImageProxy to Bitmap - handles YUV_420_888 format with proper stride handling
fun ImageProxy.toBitmapSafe(): Bitmap? {
    return try {
        val yPlane = planes[0]
        val uPlane = planes[1]
        val vPlane = planes[2]

        val yBuffer = yPlane.buffer
        val uBuffer = uPlane.buffer
        val vBuffer = vPlane.buffer

        val yRowStride = yPlane.rowStride
        val uvRowStride = uPlane.rowStride
        val uvPixelStride = uPlane.pixelStride

        // Build NV21 byte array with proper stride handling
        val nv21 = ByteArray(width * height * 3 / 2)

        // Copy Y plane row by row (handle stride)
        var pos = 0
        for (row in 0 until height) {
            yBuffer.position(row * yRowStride)
            yBuffer.get(nv21, pos, width)
            pos += width
        }

        // Copy UV planes (interleaved as VU for NV21)
        val uvHeight = height / 2
        val uvWidth = width / 2
        for (row in 0 until uvHeight) {
            for (col in 0 until uvWidth) {
                val uvIndex = row * uvRowStride + col * uvPixelStride
                vBuffer.position(uvIndex)
                uBuffer.position(uvIndex)
                nv21[pos++] = vBuffer.get()
                nv21[pos++] = uBuffer.get()
            }
        }

        val yuvImage = YuvImage(nv21, ImageFormat.NV21, width, height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, width, height), 90, out)
        val imageBytes = out.toByteArray()

        var bitmap = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)

        // Rotate if needed based on image rotation
        val rotation = imageInfo.rotationDegrees
        if (rotation != 0) {
            val matrix = Matrix()
            matrix.postRotate(rotation.toFloat())
            bitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
        }
        bitmap
    } catch (e: Exception) {
        LogUtil.e("FurnitureFit", "toBitmap failed: ${e.message}")
        e.printStackTrace()
        null
    }
}
