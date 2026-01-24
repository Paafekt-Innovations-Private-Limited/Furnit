package com.furnit.android

import android.content.Intent
import android.os.Bundle
import android.view.View
import android.widget.ImageButton
import android.widget.ProgressBar
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.furnit.android.models.ModelManager
import io.github.sceneview.SceneView
import io.github.sceneview.math.Position
import io.github.sceneview.node.ModelNode
import kotlinx.coroutines.launch
import java.io.File
import java.nio.ByteBuffer

class ModelDetailActivity : AppCompatActivity() {

    companion object {
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
    private var isPreviewMode = false
    private var glbPath: String? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_model_detail)

        modelManager = ModelManager(this)

        sceneView = findViewById(R.id.sceneView)
        loadingIndicator = findViewById(R.id.loadingIndicator)
        modelTitle = findViewById(R.id.modelTitle)
        brainButton = findViewById(R.id.brainButton)

        val backButton: ImageButton = findViewById(R.id.backButton)
        backButton.setOnClickListener { finish() }

        isPreviewMode = intent.getBooleanExtra(EXTRA_IS_PREVIEW, false)

        // Check for direct GLB path first (for preview mode)
        val directGlbPath = intent.getStringExtra(EXTRA_GLB_PATH)
        val roomName = intent.getStringExtra(EXTRA_ROOM_NAME)

        if (directGlbPath != null) {
            // Direct GLB path mode (preview before save)
            glbPath = directGlbPath
            modelTitle.text = roomName ?: "Room Preview"
            brainButton.visibility = View.GONE  // Hide brain button in preview mode
            loadModel(directGlbPath)
        } else {
            // Model ID mode (existing rooms)
            val modelId = intent.getStringExtra(EXTRA_MODEL_ID) ?: return
            val model = modelManager.getModel(modelId) ?: return

            glbPath = model.assetPath
            modelTitle.text = model.name

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
