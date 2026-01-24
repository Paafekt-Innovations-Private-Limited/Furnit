package com.furnit.android

import android.content.Intent
import android.graphics.Color
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.EditText
import android.widget.ImageButton
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.FileProvider
import androidx.lifecycle.lifecycleScope
import com.furnit.android.models.ModelManager
import com.furnit.android.utils.RoomBoundaryManager
import io.github.sceneview.SceneView
import io.github.sceneview.node.ModelNode
import kotlinx.coroutines.launch
import java.io.File
import java.nio.ByteBuffer

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
    private lateinit var boundaryManager: RoomBoundaryManager
    private var isPreviewMode = false
    private var glbPath: String? = null
    private var currentModelId: String? = null

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

        val backButton: ImageButton = findViewById(R.id.backButton)
        backButton.setOnClickListener { finish() }

        // Help button
        helpButton.setOnClickListener { showHelpDialog() }

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

    private fun loadModel(assetPath: String) {
        lifecycleScope.launch {
            try {
                val isFileSystemPath = assetPath.startsWith("/")
                Log.d(TAG, "=== Loading Model ===")
                Log.d(TAG, "  Path type: ${if (isFileSystemPath) "FILE SYSTEM" else "ASSETS"}")
                Log.d(TAG, "  Path: $assetPath")

                val modelInstance = if (isFileSystemPath) {
                    // File system path - load from file (user-created rooms)
                    val file = File(assetPath)
                    Log.d(TAG, "  File exists: ${file.exists()}, size: ${file.length()} bytes")
                    val bytes = file.readBytes()
                    val buffer = ByteBuffer.wrap(bytes)
                    sceneView.modelLoader.createModelInstance(buffer)
                } else {
                    // Asset path - load from assets (bundled rooms like vintage)
                    sceneView.modelLoader.createModelInstance(
                        assetFileLocation = assetPath
                    )
                }

                // Room dimensions from GlbGenerator: width=4, depth=4.5, height=2.8
                // Model Y goes from 0 (floor) to 2.8 (ceiling)
                // Don't scale - keep original size for proper camera positioning
                val modelNode = ModelNode(
                    modelInstance = modelInstance,
                    scaleToUnits = null  // Keep original scale
                )

                sceneView.addChildNode(modelNode)

                // Log model position
                Log.d(TAG, "  Model added, position: ${modelNode.position}")

                // Use RoomBoundaryManager for camera positioning (like iOS)
                // Initialize with default room dimensions (matches GlbGenerator)
                boundaryManager.initializeFromDimensions()

                // Detect orientation - portrait needs camera further back
                val isPortrait = resources.configuration.orientation ==
                    android.content.res.Configuration.ORIENTATION_PORTRAIT

                // Get optimal camera position from boundary manager
                val cameraSetup = boundaryManager.getOptimalCameraPosition(isPortrait = isPortrait)

                // Position camera IMMEDIATELY after adding model
                sceneView.cameraNode.apply {
                    position = cameraSetup.position
                    lookAt(cameraSetup.lookAt)
                }

                Log.d(TAG, "  Camera position set: ${cameraSetup.position}")
                Log.d(TAG, "  Camera lookAt: ${cameraSetup.lookAt}")

                // Re-apply camera position after a frame to override any manipulator reset
                sceneView.post {
                    sceneView.cameraNode.apply {
                        position = cameraSetup.position
                        lookAt(cameraSetup.lookAt)
                    }
                    Log.d(TAG, "  Camera position re-applied (post)")
                }

                // Also re-apply after a short delay to handle async initialization
                sceneView.postDelayed({
                    sceneView.cameraNode.apply {
                        position = cameraSetup.position
                        lookAt(cameraSetup.lookAt)
                    }
                    Log.d(TAG, "  Camera position re-applied (delayed)")
                    Log.d(TAG, "  Final camera: ${sceneView.cameraNode.position}")
                }, 100)

                Log.d(TAG, "=== Model Load Complete ===")

                loadingIndicator.visibility = View.GONE

            } catch (e: Exception) {
                Log.e(TAG, "Failed to load model", e)
                e.printStackTrace()
                loadingIndicator.visibility = View.GONE
                modelTitle.text = "Failed to load: ${e.message}"
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
    }
}
