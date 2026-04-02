package com.furnit.android

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Color
import android.os.Bundle
import android.view.Gravity
import android.view.ViewGroup
import android.widget.Button
import android.widget.FrameLayout
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.furnit.android.ar.ArPhotoCaptureResult
import com.furnit.android.ar.FurnitureFitArCameraController
import com.furnit.android.ar.MetricAnchor
import com.furnit.android.utils.LogUtil
import java.io.File
import java.io.FileOutputStream
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.Executors

class ArDepthPhotoCaptureActivity : AppCompatActivity() {
    companion object {
        const val EXTRA_CAPTURED_IMAGE_URI = "captured_image_uri"
        const val EXTRA_METRIC_ANCHORS = "metric_anchors"
    }

    private val inferenceExecutor = Executors.newSingleThreadExecutor()
    private lateinit var controller: FurnitureFitArCameraController
    private lateinit var previewImageView: ImageView
    private lateinit var captureButton: Button
    private lateinit var statusText: TextView
    @Volatile
    private var latestPreviewBitmap: Bitmap? = null

    private val cameraPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission(),
    ) { granted ->
        if (granted) {
            controller.onHostResume()
        } else {
            Toast.makeText(this, "Camera permission is required", Toast.LENGTH_LONG).show()
            finish()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        controller = FurnitureFitArCameraController(this, inferenceExecutor).apply {
            lockedPhotoOrientation = "portrait"
            minFrameIntervalMs = 120L
            shouldPostBitmapFrame = { true }
            onBitmapFrame = { bitmap ->
                val previousBitmap = latestPreviewBitmap
                latestPreviewBitmap = bitmap
                runOnUiThread {
                    previewImageView.setImageBitmap(bitmap)
                    previousBitmap?.takeIf { it !== bitmap && !it.isRecycled }?.recycle()
                }
            }
        }

        previewImageView = ImageView(this).apply {
            scaleType = ImageView.ScaleType.CENTER_CROP
            setBackgroundColor(Color.BLACK)
        }

        statusText = TextView(this).apply {
            text = "Move slowly for depth, then tap Capture"
            setTextColor(Color.WHITE)
            textSize = 16f
            setPadding(32, 32, 32, 32)
            setBackgroundColor(Color.parseColor("#66000000"))
        }
        captureButton = Button(this).apply {
            text = "Capture"
            setOnClickListener { capturePhotoWithAnchors() }
        }
        val cancelButton = Button(this).apply {
            text = "Cancel"
            setOnClickListener {
                setResult(RESULT_CANCELED)
                finish()
            }
        }

        val controls = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER
            setPadding(24, 16, 24, 40)
            setBackgroundColor(Color.parseColor("#66000000"))
            addView(cancelButton, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
            addView(captureButton, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
        }

        val root = FrameLayout(this).apply {
            addView(
                controller.glSurfaceView,
                FrameLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.MATCH_PARENT,
                ),
            )
            addView(
                previewImageView,
                FrameLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.MATCH_PARENT,
                ),
            )
            addView(
                statusText,
                FrameLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                ).apply { gravity = Gravity.TOP },
            )
            addView(
                controls,
                FrameLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                ).apply { gravity = Gravity.BOTTOM },
            )
        }
        setContentView(root)
    }

    override fun onResume() {
        super.onResume()
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            controller.onHostResume()
        } else {
            cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    override fun onPause() {
        controller.onHostPause()
        super.onPause()
    }

    override fun onDestroy() {
        latestPreviewBitmap?.takeIf { !it.isRecycled }?.recycle()
        controller.destroy()
        inferenceExecutor.shutdownNow()
        super.onDestroy()
    }

    private fun capturePhotoWithAnchors() {
        captureButton.isEnabled = false
        statusText.text = "Capturing photo and AR depth..."
        controller.requestPhotoCapture { result ->
            if (result == null) {
                captureButton.isEnabled = true
                statusText.text = "Capture failed. Try again."
                return@requestPhotoCapture
            }
            runCatching {
                val imageFile = saveCapturedBitmap(result)
                val data = Intent()
                    .putExtra(EXTRA_CAPTURED_IMAGE_URI, imageFile.toURI().toString())
                    .putExtra(EXTRA_METRIC_ANCHORS, ArrayList(result.metricAnchors))
                setResult(RESULT_OK, data)
                finish()
            }.onFailure { throwable ->
                LogUtil.e("ArDepthPhotoCapture", "Saving AR capture failed", throwable)
                captureButton.isEnabled = true
                statusText.text = "Could not save photo. Try again."
            }
        }
    }

    private fun saveCapturedBitmap(result: ArPhotoCaptureResult): File {
        val capturesDir = File(cacheDir, "ar_depth_captures").apply { mkdirs() }
        val filename = "ar_capture_${SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())}.jpg"
        val outFile = File(capturesDir, filename)
        FileOutputStream(outFile).use { output ->
            result.bitmap.compress(android.graphics.Bitmap.CompressFormat.JPEG, 95, output)
        }
        return outFile
    }
}
