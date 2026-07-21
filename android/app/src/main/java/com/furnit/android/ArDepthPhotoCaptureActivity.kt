package com.furnit.android

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.os.Bundle
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.widget.FrameLayout
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.furnit.android.ar.ArPhotoCaptureResult
import com.furnit.android.ar.FurnitureFitArCameraController
import com.furnit.android.utils.LogUtil
import com.furnit.android.utils.WindowInsetsUtil
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
    private lateinit var captureButton: FrameLayout
    private lateinit var cancelButton: TextView
    private lateinit var statusText: TextView
    @Volatile
    private var latestPreviewBitmap: Bitmap? = null

    private val cameraPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission(),
    ) { granted ->
        if (granted) {
            controller.onHostResume()
        } else {
            Toast.makeText(this, getString(R.string.camera_permission_required), Toast.LENGTH_LONG).show()
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
            scaleType = ImageView.ScaleType.FIT_CENTER
            setBackgroundColor(Color.BLACK)
        }

        statusText = TextView(this).apply {
            text = getString(R.string.camera_ar_hint)
            setTextColor(Color.WHITE)
            textSize = 14f
            gravity = Gravity.CENTER
            setTypeface(null, Typeface.BOLD)
            setPadding(dp(16), dp(12), dp(16), dp(12))
            background = roundedRect(Color.parseColor("#73000000"), dp(8).toFloat())
            // Edge-to-edge (targetSdk 35+) draws behind the status bar; add the real
            // status bar inset so the hint text clears the notification bar.
            WindowInsetsUtil.applyTopInsetAsPadding(this)
        }
        captureButton = createShutterButton().apply {
            contentDescription = getString(R.string.camera_capture)
            setOnClickListener { capturePhotoWithAnchors() }
        }
        cancelButton = TextView(this).apply {
            text = getString(R.string.common_cancel)
            setTextColor(Color.WHITE)
            textSize = 18f
            gravity = Gravity.CENTER
            setTypeface(null, Typeface.BOLD)
            setPadding(dp(20), dp(12), dp(20), dp(12))
            setOnClickListener {
                setResult(RESULT_CANCELED)
                finish()
            }
        }

        val controls = FrameLayout(this).apply {
            setPadding(dp(16), dp(8), dp(16), dp(28))
            background = GradientDrawable(
                GradientDrawable.Orientation.TOP_BOTTOM,
                intArrayOf(Color.TRANSPARENT, Color.parseColor("#8A000000")),
            )
            // Add the navigation bar inset so the buttons clear the gesture/nav bar.
            WindowInsetsUtil.applyBottomInsetAsPadding(this)
            addView(
                cancelButton,
                FrameLayout.LayoutParams(
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                    Gravity.START or Gravity.CENTER_VERTICAL,
                ),
            )
            addView(
                captureButton,
                FrameLayout.LayoutParams(dp(88), dp(88), Gravity.CENTER),
            )
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
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                    Gravity.TOP or Gravity.CENTER_HORIZONTAL,
                ).apply {
                    topMargin = dp(12)
                    marginStart = dp(16)
                    marginEnd = dp(16)
                },
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
        statusText.text = getString(R.string.camera_ar_capturing)
        controller.requestPhotoCapture { result ->
            if (result == null) {
                captureButton.isEnabled = true
                statusText.text = getString(R.string.camera_ar_capture_failed)
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
                statusText.text = getString(R.string.camera_ar_save_failed)
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

    private fun createShutterButton(): FrameLayout {
        val outerRing = View(this).apply {
            background = ovalStroke(Color.WHITE, dp(4))
        }
        val innerCircle = View(this).apply {
            background = ovalFill(Color.WHITE)
        }
        return FrameLayout(this).apply {
            foreground = selectableItemBackgroundBorderless()
            addView(
                outerRing,
                FrameLayout.LayoutParams(dp(78), dp(78), Gravity.CENTER),
            )
            addView(
                innerCircle,
                FrameLayout.LayoutParams(dp(58), dp(58), Gravity.CENTER),
            )
        }
    }

    private fun selectableItemBackgroundBorderless(): android.graphics.drawable.Drawable? {
        val attrs = intArrayOf(android.R.attr.selectableItemBackgroundBorderless)
        val typedArray = obtainStyledAttributes(attrs)
        return try {
            typedArray.getDrawable(0)
        } finally {
            typedArray.recycle()
        }
    }

    private fun roundedRect(color: Int, radius: Float): GradientDrawable {
        return GradientDrawable().apply {
            shape = GradientDrawable.RECTANGLE
            cornerRadius = radius
            setColor(color)
        }
    }

    private fun ovalFill(color: Int): GradientDrawable {
        return GradientDrawable().apply {
            shape = GradientDrawable.OVAL
            setColor(color)
        }
    }

    private fun ovalStroke(color: Int, width: Int): GradientDrawable {
        return GradientDrawable().apply {
            shape = GradientDrawable.OVAL
            setColor(Color.TRANSPARENT)
            setStroke(width, color)
        }
    }

    private fun dp(value: Int): Int = (value * resources.displayMetrics.density).toInt()
}
