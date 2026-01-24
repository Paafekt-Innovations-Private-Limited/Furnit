package com.furnit.android

import android.content.Intent
import android.graphics.Bitmap
import android.os.Bundle
import android.os.Environment
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
    private lateinit var helpButton: ImageButton
    private lateinit var screenshotButton: ImageButton
    private lateinit var orientationLabel: LinearLayout
    private var isPreviewMode = false
    private var glbPath: String? = null
    private var currentModelId: String? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_model_detail)

        modelManager = ModelManager(this)

        sceneView = findViewById(R.id.sceneView)
        loadingIndicator = findViewById(R.id.loadingIndicator)
        modelTitle = findViewById(R.id.modelTitle)
        brainButton = findViewById(R.id.brainButton)
        saveButton = findViewById(R.id.saveButton)
        helpButton = findViewById(R.id.helpButton)
        screenshotButton = findViewById(R.id.screenshotButton)
        orientationLabel = findViewById(R.id.orientationLabel)

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

            // In preview mode, show save button and hide brain button
            saveButton.visibility = View.VISIBLE
            saveButton.setOnClickListener { showSaveDialog() }
            brainButton.visibility = View.GONE

            loadModel(directGlbPath)
        } else {
            // Model ID mode (existing rooms)
            val modelId = intent.getStringExtra(EXTRA_MODEL_ID) ?: return
            val model = modelManager.getModel(modelId) ?: return

            currentModelId = modelId
            glbPath = model.assetPath
            modelTitle.text = "3D Room View"

            // In view mode, hide save button and show brain button
            saveButton.visibility = View.GONE
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
            hint = "Enter room name"
            setPadding(48, 32, 48, 32)
        }

        AlertDialog.Builder(this)
            .setTitle("Save Room")
            .setMessage("Enter a name for your room")
            .setView(input)
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

                // Return to home
                setResult(RESULT_OK)
                finish()
            } else {
                Toast.makeText(this, "Failed to save room", Toast.LENGTH_SHORT).show()
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to save room", e)
            Toast.makeText(this, "Error: ${e.message}", Toast.LENGTH_SHORT).show()
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
                    android.util.Log.d("ModelDetail", "Loading GLB from file: $assetPath")
                    val file = File(assetPath)
                    val bytes = file.readBytes()
                    val buffer = ByteBuffer.wrap(bytes)
                    sceneView.modelLoader.createModelInstance(buffer)
                } else {
                    // Asset path - load from assets
                    android.util.Log.d("ModelDetail", "Loading GLB from assets: $assetPath")
                    sceneView.modelLoader.createModelInstance(
                        assetFileLocation = assetPath
                    )
                }

                val modelNode = ModelNode(
                    modelInstance = modelInstance,
                    scaleToUnits = null  // Keep original scale
                )

                // Position model at origin
                modelNode.position = Position(0f, 0f, 0f)

                sceneView.addChildNode(modelNode)

                // Position camera at back of room, eye level, looking toward front wall
                // Room geometry: front wall at z=-2.25, back at z=+2.25
                sceneView.cameraNode.apply {
                    position = Position(0f, 1.6f, 3.5f)   // Back of room, eye level
                    lookAt(Position(0f, 1.4f, -2.25f))    // Look at front wall center
                }

                android.util.Log.d("ModelDetail", "Camera at back of room looking at front wall")

                loadingIndicator.visibility = View.GONE

            } catch (e: Exception) {
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
