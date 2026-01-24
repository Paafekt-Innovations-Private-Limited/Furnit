package com.furnit.android

import android.content.Intent
import android.graphics.Bitmap
import android.graphics.Color
import android.os.Bundle
import android.os.Environment
import android.util.Log
import android.view.View
import android.view.WindowManager
import android.widget.EditText
import android.widget.FrameLayout
import android.widget.ImageButton
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.FileProvider
import androidx.core.view.ViewCompat
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.WindowInsetsControllerCompat
import androidx.core.view.updatePadding
import androidx.lifecycle.lifecycleScope
import com.furnit.android.models.ModelManager
import com.google.android.filament.Camera
import io.github.sceneview.SceneView
import io.github.sceneview.math.Position
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
    private lateinit var brainButton: ImageButton
    private lateinit var saveButton: ImageButton
    private lateinit var shareButton: ImageButton
    private lateinit var helpButton: ImageButton
    private lateinit var screenshotButton: ImageButton
    private lateinit var orientationLabel: LinearLayout
    private var isPreviewMode = false
    private var glbPath: String? = null
    private var currentModelId: String? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Enable true edge-to-edge display (matching iOS ignoresSafeArea)
        WindowCompat.setDecorFitsSystemWindows(window, false)

        // Make status bar and navigation bar transparent so content draws behind them
        window.statusBarColor = Color.TRANSPARENT
        window.navigationBarColor = Color.TRANSPARENT

        // Configure window insets controller for proper edge-to-edge
        WindowInsetsControllerCompat(window, window.decorView).let { controller ->
            // Show system bars but make them transparent
            controller.isAppearanceLightStatusBars = false
            controller.isAppearanceLightNavigationBars = false
        }

        // Enable layout in display cutout area (notch)
        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.P) {
            window.attributes.layoutInDisplayCutoutMode =
                WindowManager.LayoutParams.LAYOUT_IN_DISPLAY_CUTOUT_MODE_SHORT_EDGES
        }

        setContentView(R.layout.activity_model_detail)

        modelManager = ModelManager(this)

        sceneView = findViewById(R.id.sceneView)
        loadingIndicator = findViewById(R.id.loadingIndicator)
        modelTitle = findViewById(R.id.modelTitle)
        brainButton = findViewById(R.id.brainButton)
        saveButton = findViewById(R.id.saveButton)
        shareButton = findViewById(R.id.shareButton)
        helpButton = findViewById(R.id.helpButton)
        screenshotButton = findViewById(R.id.screenshotButton)
        orientationLabel = findViewById(R.id.orientationLabel)

        // Handle window insets for overlay controls (like iOS safe area)
        val topBarContainer = findViewById<LinearLayout>(R.id.topBarContainer)
        val bottomControlsContainer = findViewById<FrameLayout>(R.id.bottomControlsContainer)

        ViewCompat.setOnApplyWindowInsetsListener(topBarContainer) { view, windowInsets ->
            val insets = windowInsets.getInsets(WindowInsetsCompat.Type.systemBars())
            view.updatePadding(top = insets.top + 16)
            windowInsets
        }

        ViewCompat.setOnApplyWindowInsetsListener(bottomControlsContainer) { view, windowInsets ->
            val insets = windowInsets.getInsets(WindowInsetsCompat.Type.systemBars())
            view.updatePadding(bottom = insets.bottom + 24)
            windowInsets
        }

        val backButton: ImageButton = findViewById(R.id.backButton)
        backButton.setOnClickListener { finish() }

        // Help button
        helpButton.setOnClickListener { showHelpDialog() }

        // Screenshot button
        screenshotButton.setOnClickListener { takeScreenshot() }

        isPreviewMode = intent.getBooleanExtra(EXTRA_IS_PREVIEW, false)

        // Check for direct GLB path first (for preview mode)
        val directGlbPath = intent.getStringExtra(EXTRA_GLB_PATH)
        val roomName = intent.getStringExtra(EXTRA_ROOM_NAME)

        if (directGlbPath != null) {
            // Direct GLB path mode (preview before save)
            glbPath = directGlbPath
            modelTitle.text = "3D Room View"

            // In preview mode, show save button (down arrow), hide share button
            saveButton.visibility = View.VISIBLE
            saveButton.setOnClickListener { showSaveDialog() }
            shareButton.visibility = View.GONE

            // Show brain button but prompt to save first
            brainButton.visibility = View.VISIBLE
            brainButton.setOnClickListener {
                Toast.makeText(this, "Please save the room first", Toast.LENGTH_SHORT).show()
            }

            loadModel(directGlbPath)
        } else {
            // Model ID mode (existing rooms)
            val modelId = intent.getStringExtra(EXTRA_MODEL_ID) ?: return
            val model = modelManager.getModel(modelId) ?: return

            currentModelId = modelId
            glbPath = model.assetPath
            modelTitle.text = "3D Room View"

            // In view mode, hide save button and show share button
            saveButton.visibility = View.GONE
            shareButton.visibility = View.VISIBLE
            shareButton.setOnClickListener { shareRoom() }
            brainButton.visibility = View.VISIBLE

            // Brain button launches FurnitureFit segmentation with this room as background
            brainButton.setOnClickListener {
                val intent = Intent(this, FurnitureFitActivity::class.java)
                intent.putExtra("ROOM_ID", model.id)
                intent.putExtra("ROOM_NAME", model.name)
                startActivity(intent)
            }

            loadModel(model.assetPath)
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
                val name = input.text.toString().ifEmpty { "My Room" }
                saveRoom(name)
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun saveRoom(name: String) {
        val path = glbPath
        if (path == null) {
            Toast.makeText(this, "No room data to save", Toast.LENGTH_SHORT).show()
            return
        }

        try {
            val glbFile = File(path)
            val roomFolder = glbFile.parentFile

            if (roomFolder != null) {
                // Update metadata with user's name
                val metadataFile = File(roomFolder, "metadata.txt")
                metadataFile.writeText("name=$name\ncreated=${System.currentTimeMillis()}\ntype=manual")

                Toast.makeText(this, "Room '$name' saved!", Toast.LENGTH_SHORT).show()
                Log.d(TAG, "Room saved: $name at ${roomFolder.absolutePath}")

                // Go to rooms list screen (ContentActivity)
                val intent = Intent(this, ContentActivity::class.java)
                intent.flags = Intent.FLAG_ACTIVITY_CLEAR_TOP or Intent.FLAG_ACTIVITY_NEW_TASK
                startActivity(intent)
                finish()
            } else {
                Toast.makeText(this, "Failed to save room", Toast.LENGTH_SHORT).show()
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to save room", e)
            Toast.makeText(this, "Error: ${e.message}", Toast.LENGTH_SHORT).show()
        }
    }

    private fun shareRoom() {
        val path = glbPath ?: return
        try {
            val glbFile = File(path)
            if (!glbFile.exists()) {
                Toast.makeText(this, "Room file not found", Toast.LENGTH_SHORT).show()
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
            Log.e(TAG, "Failed to share room", e)
            Toast.makeText(this, "Failed to share room", Toast.LENGTH_SHORT).show()
        }
    }

    private fun takeScreenshot() {
        try {
            // Capture the SceneView
            val bitmap = Bitmap.createBitmap(sceneView.width, sceneView.height, Bitmap.Config.ARGB_8888)
            val pixelCopy = android.view.PixelCopy.request(
                sceneView,
                bitmap,
                { result ->
                    if (result == android.view.PixelCopy.SUCCESS) {
                        saveAndShareScreenshot(bitmap)
                    } else {
                        runOnUiThread {
                            Toast.makeText(this, "Failed to capture screenshot", Toast.LENGTH_SHORT).show()
                        }
                    }
                },
                android.os.Handler(mainLooper)
            )
        } catch (e: Exception) {
            Log.e(TAG, "Screenshot failed", e)
            Toast.makeText(this, "Screenshot failed", Toast.LENGTH_SHORT).show()
        }
    }

    private fun saveAndShareScreenshot(bitmap: Bitmap) {
        try {
            val timeStamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())
            val fileName = "Room_$timeStamp.png"
            val picturesDir = getExternalFilesDir(Environment.DIRECTORY_PICTURES)
            val file = File(picturesDir, fileName)

            FileOutputStream(file).use { out ->
                bitmap.compress(Bitmap.CompressFormat.PNG, 100, out)
            }

            runOnUiThread {
                Toast.makeText(this, "Screenshot saved", Toast.LENGTH_SHORT).show()
            }

            // Share the screenshot
            val uri = FileProvider.getUriForFile(this, "${packageName}.fileprovider", file)
            val shareIntent = Intent(Intent.ACTION_SEND).apply {
                type = "image/png"
                putExtra(Intent.EXTRA_STREAM, uri)
                addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            }
            startActivity(Intent.createChooser(shareIntent, "Share Screenshot"))

        } catch (e: Exception) {
            Log.e(TAG, "Failed to save screenshot", e)
            runOnUiThread {
                Toast.makeText(this, "Failed to save screenshot", Toast.LENGTH_SHORT).show()
            }
        }
    }

    private fun loadModel(assetPath: String) {
        lifecycleScope.launch {
            try {
                val modelInstance = if (assetPath.startsWith("/")) {
                    // File system path - load from file
                    Log.d(TAG, "Loading GLB from file: $assetPath")
                    val file = File(assetPath)
                    val bytes = file.readBytes()
                    val buffer = ByteBuffer.wrap(bytes)
                    sceneView.modelLoader.createModelInstance(buffer)
                } else {
                    // Asset path - load from assets
                    Log.d(TAG, "Loading GLB from assets: $assetPath")
                    sceneView.modelLoader.createModelInstance(
                        assetFileLocation = assetPath
                    )
                }

                // Room dimensions from GlbGenerator: width=4, depth=4.5, height=2.8
                // Model Y goes from 0 (floor) to 2.8 (ceiling), so center Y is at 1.4
                // Scale room to fit in ~3 units
                val modelNode = ModelNode(
                    modelInstance = modelInstance,
                    scaleToUnits = 3.0f  // Scale room to fit in 3-unit cube
                )

                sceneView.addChildNode(modelNode)

                Log.d(TAG, "Model added with scaleToUnits=3.0")

                // Setup camera after view is ready
                sceneView.post {
                    setupCamera()
                }

                loadingIndicator.visibility = View.GONE

            } catch (e: Exception) {
                Log.e(TAG, "Failed to load model", e)
                e.printStackTrace()
                loadingIndicator.visibility = View.GONE
                modelTitle.text = "Failed to load: ${e.message}"
            }
        }
    }

    private fun setupCamera() {
        // Simple camera setup - position to view the room
        // Room model: floor at Y=0, ceiling at Y=2.8, scaled to fit ~3 units
        val roomCenterY = 1.0f  // Center of room after scaling

        sceneView.cameraNode.apply {
            position = Position(0f, roomCenterY, 4.5f)
            lookAt(Position(0f, roomCenterY, 0f))
        }

        Log.d(TAG, "Camera setup: pos=(0, $roomCenterY, 4.5), lookAt=(0, $roomCenterY, 0)")
    }

    override fun onDestroy() {
        super.onDestroy()
    }
}
