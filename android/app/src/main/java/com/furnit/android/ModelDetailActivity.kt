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
import android.view.MotionEvent
import com.furnit.android.models.ModelManager
import com.furnit.android.utils.RoomBoundaryManager
import io.github.sceneview.SceneView
import io.github.sceneview.node.CubeNode
import io.github.sceneview.node.ModelNode
import kotlinx.coroutines.launch
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
    private var isPreviewMode = false
    /** True while showing an on-disk GLB preview that has not been saved to the library yet. */
    private var unsavedPreviewActive = false
    private lateinit var previewBackCallback: OnBackPressedCallback

    // Touch-anywhere drag for camera control
    private var lastTouchX = 0f
    private var lastTouchY = 0f
    private var isDragging = false
    private var glbPath: String? = null
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
        loadingIndicator = findViewById(R.id.loadingIndicator)
        modelTitle = findViewById(R.id.modelTitle)
        saveButton = findViewById(R.id.saveButton)
        shareButton = findViewById(R.id.shareButton)
        helpButton = findViewById(R.id.helpButton)
        brainButton = findViewById(R.id.brainButton)
        screenshotButton = findViewById(R.id.screenshotButton)
        orientationLabel = findViewById(R.id.orientationLabel)

        previewBackCallback = object : OnBackPressedCallback(false) {
            override fun handleOnBackPressed() {
                showUnsavedPreviewLeaveDialog()
            }
        }
        onBackPressedDispatcher.addCallback(this, previewBackCallback)

        val backButton: ImageButton = findViewById(R.id.backButton)
        backButton.setOnClickListener {
            if (unsavedPreviewActive) {
                showUnsavedPreviewLeaveDialog()
            } else {
                finish()
            }
        }

        // Help button
        helpButton.setOnClickListener { showHelpDialog() }

        // Screenshot button
        screenshotButton.setOnClickListener { takeScreenshot() }

        // Touch overlay is handled via dispatchTouchEvent override

        // Update orientation label based on device orientation
        updateOrientationLabel()

        isPreviewMode = intent.getBooleanExtra(EXTRA_IS_PREVIEW, false)

        // Check for direct GLB path first (for preview mode)
        val directGlbPath = intent.getStringExtra(EXTRA_GLB_PATH)

        if (directGlbPath != null) {
            // Direct GLB path mode (preview before save)
            unsavedPreviewActive = true
            previewBackCallback.isEnabled = true
            glbPath = directGlbPath
            modelTitle.text = getString(R.string.model_detail_preview)
            LogUtil.d(TAG, "Preview mode - GLB path: $directGlbPath")

            // In preview mode, show save button (down arrow), hide share button
            saveButton.visibility = View.VISIBLE
            saveButton.setOnClickListener { showSaveDialog() }
            shareButton.visibility = View.GONE

            // Brain button prompts to save first in preview mode
            brainButton.visibility = View.VISIBLE
            brainButton.setOnClickListener {
                LogUtil.d(TAG, "Brain button clicked in preview mode")
                Toast.makeText(this, getString(R.string.model_detail_save_first), Toast.LENGTH_SHORT).show()
            }

            // Screenshot works in preview mode
            screenshotButton.visibility = View.VISIBLE
            screenshotButton.setOnClickListener {
                LogUtil.d(TAG, "Screenshot button clicked in preview mode")
                Toast.makeText(this, getString(R.string.model_detail_taking_screenshot), Toast.LENGTH_SHORT).show()
                takeScreenshot()
            }

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
            brainButton.visibility = View.VISIBLE
            brainButton.setOnClickListener {
                val roomFolder = java.io.File(model.assetPath).let { f ->
                    if (f.isFile) f.parent else f.absolutePath
                }
                // Only pass ROOM_FOLDER when it's an absolute path to a real folder (user room).
                // Bundled assets (bundled_rooms/vintage.glb) use a relative asset path — no room folder; omit ROOM_FOLDER to use ROOM_ID.
                val absoluteFolder = if (roomFolder != null && java.io.File(roomFolder).isAbsolute) roomFolder else null
                LogUtil.d(TAG, "Brain click: ROOM_ID=${model.id} ROOM_FOLDER=$absoluteFolder (raw=$roomFolder)")
                val intent = Intent(this, FurnitureFitActivity::class.java)
                intent.putExtra("ROOM_ID", model.id)
                intent.putExtra("ROOM_NAME", model.name)
                if (absoluteFolder != null) intent.putExtra("ROOM_FOLDER", absoluteFolder)
                startActivity(intent)
            }

            loadModel(model.assetPath, model.roomWidth, model.roomHeight, model.roomDepth)
        }
    }

    private fun showHelpDialog() {
        AlertDialog.Builder(this)
            .setTitle("3D Room Controls")
            .setMessage("• Drag on screen to rotate the view\n\n• Pinch to zoom in/out\n\n• Tap the brain icon to detect furniture\n\n• Tap the camera icon to take a screenshot")
            .setPositiveButton("OK", null)
            .show()
    }

    private fun showSaveDialog() {
        val input = EditText(this).apply {
            hint = "Room Name"
            setHintTextColor(0x80FFFFFF.toInt())
            setTextColor(0xFFFFFFFF.toInt())
            setBackgroundResource(R.drawable.edittext_border)
            textSize = 16f
        }

        val container = LinearLayout(this@ModelDetailActivity).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(48, 24, 48, 0)
            addView(input, LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ))
        }

        AlertDialog.Builder(this, R.style.DarkDialogTheme)
            .setTitle("Save Room")
            .setMessage("Enter a name for your room")
            .setView(container)
            .setPositiveButton("Save") { _, _ ->
                val name = input.text.toString().ifEmpty { RoomDisplayName.myRoomWithTimestamp() }
                saveRoom(name)
            }
            .setNegativeButton("Cancel", null)
            .show()
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

    private fun saveRoom(name: String) {
        val path = glbPath
        if (path == null) {
            Toast.makeText(this, getString(R.string.model_detail_no_room_data), Toast.LENGTH_SHORT).show()
            return
        }

        try {
            val glbFile = File(path)
            val previewRoomFolder = glbFile.parentFile

            if (previewRoomFolder != null) {
                // Move room from preview directory to rooms directory
                val roomsDir = File(filesDir, "rooms")
                roomsDir.mkdirs()

                val savedRoomFolder = File(roomsDir, previewRoomFolder.name)

                // Copy all files from preview to rooms folder
                previewRoomFolder.copyRecursively(savedRoomFolder, overwrite = true)

                // Update metadata with user's name in the saved location
                val metadataFile = File(savedRoomFolder, "metadata.txt")
                metadataFile.writeText("name=$name\ncreated=${System.currentTimeMillis()}\ntype=manual")

                // Clean up preview directory
                previewRoomFolder.parentFile?.deleteRecursively()

                Toast.makeText(this, getString(R.string.room_viewer_save_success, name), Toast.LENGTH_SHORT).show()
                LogUtil.d(TAG, "Room saved: $name at ${savedRoomFolder.absolutePath}")

                // Go to rooms list screen (ContentActivity)
                val intent = Intent(this, ContentActivity::class.java)
                intent.flags = Intent.FLAG_ACTIVITY_CLEAR_TOP or Intent.FLAG_ACTIVITY_NEW_TASK
                startActivity(intent)
                finish()
            } else {
                Toast.makeText(this, getString(R.string.model_detail_failed_save), Toast.LENGTH_SHORT).show()
            }
        } catch (e: Exception) {
            LogUtil.e(TAG, "Failed to save room", e)
            Toast.makeText(this, getString(R.string.photo_room_error_load, e.message ?: ""), Toast.LENGTH_SHORT).show()
            CrashReporter.report(this, e, "Model detail — save room")
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

    private fun handleCameraDrag(event: MotionEvent) {
        when (event.action) {
            MotionEvent.ACTION_DOWN -> {
                lastTouchX = event.x
                lastTouchY = event.y
                isDragging = true
                LogUtil.d(TAG, "Touch DOWN at ($lastTouchX, $lastTouchY)")
            }
            MotionEvent.ACTION_MOVE -> {
                if (isDragging) {
                    val deltaX = event.x - lastTouchX
                    val deltaY = event.y - lastTouchY

                    // Convert screen pixels to camera movement
                    // Negative because dragging right should move camera left (pan effect)
                    val sensitivity = 0.02f
                    val camera = sceneView.cameraNode
                    val position = camera.position

                    val newX = position.x - deltaX * sensitivity
                    val newZ = position.z - deltaY * sensitivity

                    LogUtil.d(TAG, "Touch MOVE delta=($deltaX, $deltaY) -> camera ($newX, ${position.y}, $newZ)")

                    camera.position = io.github.sceneview.math.Position(
                        newX,
                        position.y,
                        newZ
                    )

                    lastTouchX = event.x
                    lastTouchY = event.y
                }
            }
            MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                isDragging = false
                LogUtil.d(TAG, "Touch UP")
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
        roomDepth: Float?
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
                boundaryManager.initializeFromDimensions(width = w, depth = d, height = h)
                LogUtil.d(TAG, "[ModelDetail] getCameraCenteredView CALLED (bbox ${w}x${h}x${d})")
                val cameraSetup = boundaryManager.getCameraCenteredView()
                LogUtil.d(TAG, "[ModelDetail] camera SET pos=(${cameraSetup.position.x}, ${cameraSetup.position.y}, ${cameraSetup.position.z}) lookAt=(${cameraSetup.lookAt.x}, ${cameraSetup.lookAt.y}, ${cameraSetup.lookAt.z})")

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
        super.onDestroy()
    }

    override fun dispatchTouchEvent(event: MotionEvent): Boolean {
        // Handle camera drag for touches not on buttons
        val touchOverlay = findViewById<View>(R.id.touchOverlay)
        if (touchOverlay != null) {
            // Check if touch is within the 3D view area (not on top bar or bottom controls)
            val topBar = findViewById<View>(R.id.topBarContainer)
            val bottomControls = findViewById<View>(R.id.bottomControlsContainer)

            val topBarBottom = topBar?.bottom ?: 0
            val bottomControlsTop = bottomControls?.top ?: Int.MAX_VALUE

            if (event.y > topBarBottom && event.y < bottomControlsTop) {
                handleCameraDrag(event)
            }
        }

        return super.dispatchTouchEvent(event)
    }
}
