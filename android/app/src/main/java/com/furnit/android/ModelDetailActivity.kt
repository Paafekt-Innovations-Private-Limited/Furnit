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

class ModelDetailActivity : AppCompatActivity() {

    private lateinit var sceneView: SceneView
    private lateinit var loadingIndicator: ProgressBar
    private lateinit var modelTitle: TextView
    private lateinit var modelManager: ModelManager
    private lateinit var brainButton: ImageButton

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

        val modelId = intent.getStringExtra("MODEL_ID") ?: return
        val model = modelManager.getModel(modelId) ?: return

        // Brain button launches SmartyPants segmentation with this room as background
        brainButton.setOnClickListener {
            val intent = Intent(this, SmartyPantsActivity::class.java)
            intent.putExtra("ROOM_ID", model.id)
            intent.putExtra("ROOM_NAME", model.name)
            startActivity(intent)
        }

        modelTitle.text = model.name

        loadModel(model.assetPath)
    }

    private fun loadModel(assetPath: String) {
        lifecycleScope.launch {
            try {
                val modelNode = ModelNode(
                    modelInstance = sceneView.modelLoader.createModelInstance(
                        assetFileLocation = assetPath
                    ),
                    scaleToUnits = null  // Keep original scale
                )

                // Position model at origin
                modelNode.position = Position(0f, 0f, 0f)

                sceneView.addChildNode(modelNode)

                // Position camera at back wall, middle, eye level, looking straight into room
                sceneView.cameraNode.apply {
                    position = Position(0f, 1.6f, -4f)  // Back wall center, eye level
                    lookAt(Position(0f, 1.6f, 4f))     // Look straight ahead (same Y)
                }

                android.util.Log.d("ModelDetail", "Camera at back wall looking into room")

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
