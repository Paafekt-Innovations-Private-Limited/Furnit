package com.furnit.android

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Color
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.os.Bundle
import android.util.Log
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import com.furnit.android.services.ExecutorchClassifier
import com.furnit.android.services.FrameAnalyzer
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/**
 * Full-bleed camera preview with live ExecuTorch MobileNetV3 classification overlay.
 * Shows top-5 predictions and FPS counter.
 */
class CameraClassifierActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "CameraClassifier"
    }

    private lateinit var previewView: PreviewView
    private lateinit var overlayLayout: LinearLayout
    private lateinit var fpsText: TextView
    private lateinit var statusText: TextView
    private val predictionViews = mutableListOf<PredictionRow>()

    private lateinit var classifier: ExecutorchClassifier
    private lateinit var cameraExecutor: ExecutorService

    private val cameraPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (isGranted) {
            startCamera()
        } else {
            Toast.makeText(this, "Camera permission required", Toast.LENGTH_LONG).show()
            finish()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        cameraExecutor = Executors.newSingleThreadExecutor()
        classifier = ExecutorchClassifier(this)

        setupUI()
        initializeClassifier()
    }

    private fun dpToPx(dp: Int): Int {
        return (dp * resources.displayMetrics.density).toInt()
    }

    private fun setupUI() {
        val rootLayout = FrameLayout(this).apply {
            setBackgroundColor(Color.BLACK)
        }

        // Full-bleed camera preview
        previewView = PreviewView(this).apply {
            layoutParams = FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT
            )
            implementationMode = PreviewView.ImplementationMode.PERFORMANCE
        }
        rootLayout.addView(previewView)

        // Back button at top-left
        val backButton = TextView(this).apply {
            text = "\u2190"
            textSize = 24f
            setTextColor(Color.WHITE)
            gravity = Gravity.CENTER
            val backgroundDrawable = GradientDrawable().apply {
                shape = GradientDrawable.OVAL
                setColor(Color.argb(128, 0, 0, 0))
            }
            background = backgroundDrawable
            val buttonParams = FrameLayout.LayoutParams(dpToPx(44), dpToPx(44))
            buttonParams.setMargins(dpToPx(16), dpToPx(48), 0, 0)
            layoutParams = buttonParams
            setOnClickListener { finish() }
        }
        rootLayout.addView(backButton)

        // Status text (shown during init)
        statusText = TextView(this).apply {
            text = "Initializing model..."
            textSize = 16f
            setTextColor(Color.WHITE)
            gravity = Gravity.CENTER
            val statusParams = FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT,
                ViewGroup.LayoutParams.WRAP_CONTENT,
                Gravity.CENTER
            )
            layoutParams = statusParams
        }
        rootLayout.addView(statusText)

        // FPS counter at top-right
        fpsText = TextView(this).apply {
            text = "-- FPS"
            textSize = 14f
            setTextColor(Color.WHITE)
            setTypeface(null, Typeface.BOLD)
            val fpsBackground = GradientDrawable().apply {
                shape = GradientDrawable.RECTANGLE
                cornerRadius = dpToPx(8).toFloat()
                setColor(Color.argb(160, 0, 0, 0))
            }
            background = fpsBackground
            setPadding(dpToPx(12), dpToPx(6), dpToPx(12), dpToPx(6))
            val fpsParams = FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT,
                ViewGroup.LayoutParams.WRAP_CONTENT,
                Gravity.TOP or Gravity.END
            )
            fpsParams.setMargins(0, dpToPx(52), dpToPx(16), 0)
            layoutParams = fpsParams
        }
        rootLayout.addView(fpsText)

        // Bottom overlay with predictions
        val overlayContainer = FrameLayout(this).apply {
            val overlayBackground = GradientDrawable().apply {
                shape = GradientDrawable.RECTANGLE
                cornerRadii = floatArrayOf(
                    dpToPx(20).toFloat(), dpToPx(20).toFloat(),
                    dpToPx(20).toFloat(), dpToPx(20).toFloat(),
                    0f, 0f, 0f, 0f
                )
                setColor(Color.argb(200, 0, 0, 0))
            }
            background = overlayBackground
            val overlayParams = FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT,
                Gravity.BOTTOM
            )
            layoutParams = overlayParams
        }

        overlayLayout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dpToPx(20), dpToPx(16), dpToPx(20), dpToPx(32))
        }

        // Title in overlay
        val overlayTitle = TextView(this).apply {
            text = "Classification"
            textSize = 16f
            setTypeface(null, Typeface.BOLD)
            setTextColor(Color.WHITE)
            setPadding(0, 0, 0, dpToPx(12))
        }
        overlayLayout.addView(overlayTitle)

        // Create 5 prediction rows
        for (i in 0 until 5) {
            val row = createPredictionRow()
            predictionViews.add(row)
            overlayLayout.addView(row.container)
        }

        overlayContainer.addView(overlayLayout)
        rootLayout.addView(overlayContainer)

        setContentView(rootLayout)
    }

    private data class PredictionRow(
        val container: LinearLayout,
        val labelText: TextView,
        val confidenceText: TextView,
        val confidenceBar: View,
        val barBackground: View
    )

    private fun createPredictionRow(): PredictionRow {
        val container = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(0, dpToPx(4), 0, dpToPx(4))
        }

        val topRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }

        val labelText = TextView(this).apply {
            text = "—"
            textSize = 14f
            setTextColor(Color.WHITE)
            layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
        }
        topRow.addView(labelText)

        val confidenceText = TextView(this).apply {
            text = "0%"
            textSize = 14f
            setTextColor(Color.parseColor("#34C759"))
            setTypeface(null, Typeface.BOLD)
        }
        topRow.addView(confidenceText)

        container.addView(topRow)

        // Confidence bar
        val barContainer = FrameLayout(this).apply {
            val barParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                dpToPx(4)
            )
            barParams.setMargins(0, dpToPx(4), 0, 0)
            layoutParams = barParams
        }

        val barBackground = View(this).apply {
            val barBg = GradientDrawable().apply {
                shape = GradientDrawable.RECTANGLE
                cornerRadius = dpToPx(2).toFloat()
                setColor(Color.argb(80, 255, 255, 255))
            }
            background = barBg
            layoutParams = FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT
            )
        }
        barContainer.addView(barBackground)

        val confidenceBar = View(this).apply {
            val barFg = GradientDrawable().apply {
                shape = GradientDrawable.RECTANGLE
                cornerRadius = dpToPx(2).toFloat()
                setColor(Color.parseColor("#34C759"))
            }
            background = barFg
            layoutParams = FrameLayout.LayoutParams(0, ViewGroup.LayoutParams.MATCH_PARENT)
        }
        barContainer.addView(confidenceBar)

        container.addView(barContainer)

        return PredictionRow(container, labelText, confidenceText, confidenceBar, barBackground)
    }

    private fun initializeClassifier() {
        if (!classifier.isModelAvailable()) {
            statusText.text = "Model not found.\nPush mobilenet_v3_small.pte to\n/data/local/tmp/furnit/"
            return
        }

        Thread {
            val success = classifier.initialize()
            runOnUiThread {
                if (success) {
                    statusText.visibility = View.GONE
                    checkCameraPermission()
                } else {
                    statusText.text = "Failed to load model"
                }
            }
        }.start()
    }

    private fun checkCameraPermission() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) {
            startCamera()
        } else {
            cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder()
                .build()
                .also { it.setSurfaceProvider(previewView.surfaceProvider) }

            val imageAnalysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also { analysis ->
                    analysis.setAnalyzer(cameraExecutor, FrameAnalyzer(classifier) { results, fps ->
                        runOnUiThread {
                            updatePredictions(results, fps)
                        }
                    })
                }

            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalysis)
                Log.d(TAG, "Camera started with classification analyzer")
            } catch (e: Exception) {
                Log.e(TAG, "Camera bind failed: ${e.message}", e)
                statusText.visibility = View.VISIBLE
                statusText.text = "Camera failed: ${e.message}"
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun updatePredictions(results: List<Pair<String, Float>>, fps: Float) {
        fpsText.text = String.format("%.1f FPS", fps)

        for (i in 0 until 5) {
            val row = predictionViews[i]
            if (i < results.size) {
                val (label, confidence) = results[i]
                row.labelText.text = label
                row.confidenceText.text = String.format("%.1f%%", confidence * 100)

                // Update bar width
                val barParams = row.confidenceBar.layoutParams as FrameLayout.LayoutParams
                row.container.post {
                    val parentWidth = row.container.width - dpToPx(40)
                    barParams.width = (parentWidth * confidence).toInt().coerceAtLeast(1)
                    row.confidenceBar.layoutParams = barParams
                }
            } else {
                row.labelText.text = "—"
                row.confidenceText.text = ""
                val barParams = row.confidenceBar.layoutParams as FrameLayout.LayoutParams
                barParams.width = 0
                row.confidenceBar.layoutParams = barParams
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        classifier.release()
    }
}
