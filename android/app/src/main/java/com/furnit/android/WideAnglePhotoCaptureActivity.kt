package com.furnit.android

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.hardware.camera2.CameraCharacteristics
import android.os.Bundle
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.widget.FrameLayout
import android.widget.ImageButton
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.camera2.interop.Camera2CameraInfo
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import com.furnit.android.utils.LogUtil
import com.furnit.android.utils.WindowInsetsUtil
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/**
 * Ultra-wide / wide back-camera capture for room creation (iOS `WideAngleCameraView` parity).
 * Landscape-locked, flash off, rule-of-thirds grid, lowest focal-length back camera when available.
 */
class WideAnglePhotoCaptureActivity : AppCompatActivity() {
    companion object {
        const val EXTRA_CAPTURED_IMAGE_URI = "captured_image_uri"
        private const val TAG = "WideAngleCapture"
    }

    private lateinit var previewView: PreviewView
    private lateinit var zoomLabel: TextView
    private lateinit var statusLabel: TextView
    private lateinit var captureButton: ImageButton
    private var imageCapture: ImageCapture? = null
    private var cameraExecutor: ExecutorService = Executors.newSingleThreadExecutor()
    private var usingUltraWide = false
    private var capturing = false

    private val cameraPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission(),
    ) { granted ->
        if (granted) {
            bindCamera()
        } else {
            Toast.makeText(this, getString(R.string.camera_permission_required), Toast.LENGTH_LONG).show()
            setResult(RESULT_CANCELED)
            finish()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        previewView = PreviewView(this).apply {
            implementationMode = PreviewView.ImplementationMode.COMPATIBLE
            scaleType = PreviewView.ScaleType.FILL_CENTER
        }

        statusLabel = TextView(this).apply {
            text = getString(R.string.camera_wide_angle_hold_steady)
            setTextColor(Color.WHITE)
            textSize = 15f
            gravity = Gravity.CENTER
            setPadding(dp(16), dp(12), dp(16), dp(12))
            setBackgroundColor(Color.parseColor("#99000000"))
            // Edge-to-edge (targetSdk 35+) draws behind the status bar; add the real
            // status bar inset so the hint clears the notification bar.
            WindowInsetsUtil.applyTopInsetAsPadding(this)
        }

        zoomLabel = TextView(this).apply {
            text = getString(R.string.camera_zoom_wide)
            setTextColor(Color.parseColor("#FFD60A"))
            textSize = 14f
            setTypeface(null, android.graphics.Typeface.BOLD)
            gravity = Gravity.CENTER
            setPadding(dp(12), dp(6), dp(12), dp(6))
        }

        val gridOverlay = RuleOfThirdsGridView(this)

        captureButton = ImageButton(this).apply {
            setImageResource(android.R.drawable.radiobutton_on_background)
            setBackgroundColor(Color.TRANSPARENT)
            imageTintList = android.content.res.ColorStateList.valueOf(Color.WHITE)
            contentDescription = getString(R.string.camera_capture_wide_angle)
            setOnClickListener { capturePhoto() }
        }

        val cancelButton = TextView(this).apply {
            text = getString(R.string.common_cancel)
            setTextColor(Color.WHITE)
            textSize = 16f
            setPadding(dp(20), dp(16), dp(20), dp(16))
            setOnClickListener {
                setResult(RESULT_CANCELED)
                finish()
            }
        }

        val bottomBar = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setBackgroundColor(Color.parseColor("#99000000"))
            setPadding(dp(8), dp(12), dp(8), dp(20))
            // Add the navigation bar inset so controls clear the gesture/nav bar.
            WindowInsetsUtil.applyBottomInsetAsPadding(this)
            addView(
                cancelButton,
                LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f),
            )
            addView(
                captureButton,
                LinearLayout.LayoutParams(dp(72), dp(72)).apply { gravity = Gravity.CENTER },
            )
            addView(
                View(this@WideAnglePhotoCaptureActivity),
                LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f),
            )
        }

        val root = FrameLayout(this).apply {
            setBackgroundColor(Color.BLACK)
            addView(
                previewView,
                FrameLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.MATCH_PARENT,
                ),
            )
            addView(
                gridOverlay,
                FrameLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.MATCH_PARENT,
                ),
            )
            addView(
                statusLabel,
                FrameLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                ).apply {
                    gravity = Gravity.TOP
                    topMargin = dp(8)
                },
            )
            addView(
                zoomLabel,
                FrameLayout.LayoutParams(
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                ).apply {
                    gravity = Gravity.TOP or Gravity.END
                    topMargin = dp(80)
                    marginEnd = dp(16)
                },
            )
            addView(
                bottomBar,
                FrameLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                ).apply { gravity = Gravity.BOTTOM },
            )
        }
        setContentView(root)

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) ==
            PackageManager.PERMISSION_GRANTED
        ) {
            bindCamera()
        } else {
            cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    @androidx.annotation.OptIn(androidx.camera.camera2.interop.ExperimentalCamera2Interop::class)
    private fun bindCamera() {
        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener({
            val provider = try {
                providerFuture.get()
            } catch (e: Exception) {
                LogUtil.e(TAG, "Camera provider failed", e)
                Toast.makeText(this, getString(R.string.camera_wide_unavailable), Toast.LENGTH_SHORT).show()
                finish()
                return@addListener
            }

            val selector = ultraWideOrBackSelector(provider)
            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }
            imageCapture = ImageCapture.Builder()
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY)
                .setFlashMode(ImageCapture.FLASH_MODE_OFF)
                .build()

            try {
                provider.unbindAll()
                provider.bindToLifecycle(this, selector, preview, imageCapture)
                zoomLabel.text = if (usingUltraWide) {
                    getString(R.string.camera_zoom_ultra_wide)
                } else {
                    getString(R.string.camera_zoom_wide)
                }
                LogUtil.d(TAG, "Camera bound ultraWide=$usingUltraWide")
            } catch (e: Exception) {
                LogUtil.e(TAG, "bindToLifecycle failed", e)
                Toast.makeText(this, getString(R.string.camera_wide_unavailable), Toast.LENGTH_SHORT).show()
                finish()
            }
        }, ContextCompat.getMainExecutor(this))
    }

    @androidx.annotation.OptIn(androidx.camera.camera2.interop.ExperimentalCamera2Interop::class)
    private fun ultraWideOrBackSelector(provider: ProcessCameraProvider): CameraSelector {
        var bestId: String? = null
        var bestFocal = Float.MAX_VALUE
        for (info in provider.availableCameraInfos) {
            val cam2 = Camera2CameraInfo.from(info)
            val facing = cam2.getCameraCharacteristic(CameraCharacteristics.LENS_FACING)
            if (facing == null || facing != CameraCharacteristics.LENS_FACING_BACK) continue
            val focals = cam2.getCameraCharacteristic(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)
                ?: continue
            val minFocal = focals.minOrNull() ?: continue
            if (minFocal < bestFocal) {
                bestFocal = minFocal
                bestId = cam2.cameraId
            }
        }

        // Heuristic: ultra-wide focal lengths are typically well below the main ~4mm class.
        usingUltraWide = bestFocal < 3.2f && bestId != null

        return if (bestId != null) {
            val targetId = bestId
            CameraSelector.Builder()
                .requireLensFacing(CameraSelector.LENS_FACING_BACK)
                .addCameraFilter { cameraInfos ->
                    val match = cameraInfos.filter {
                        Camera2CameraInfo.from(it).cameraId == targetId
                    }
                    match.ifEmpty { cameraInfos }
                }
                .build()
        } else {
            usingUltraWide = false
            CameraSelector.DEFAULT_BACK_CAMERA
        }
    }

    private fun capturePhoto() {
        if (capturing) return
        val capture = imageCapture ?: return
        capturing = true
        captureButton.isEnabled = false

        val photoFile = createImageFile()
        val outputOptions = ImageCapture.OutputFileOptions.Builder(photoFile).build()
        capture.takePicture(
            outputOptions,
            cameraExecutor,
            object : ImageCapture.OnImageSavedCallback {
                override fun onImageSaved(outputFileResults: ImageCapture.OutputFileResults) {
                    val uri = outputFileResults.savedUri
                        ?: FileProvider.getUriForFile(
                            this@WideAnglePhotoCaptureActivity,
                            "${packageName}.fileprovider",
                            photoFile,
                        )
                    runOnUiThread {
                        setResult(
                            RESULT_OK,
                            Intent().putExtra(EXTRA_CAPTURED_IMAGE_URI, uri.toString()),
                        )
                        finish()
                    }
                }

                override fun onError(exception: ImageCaptureException) {
                    LogUtil.e(TAG, "Capture failed", exception)
                    runOnUiThread {
                        capturing = false
                        captureButton.isEnabled = true
                        Toast.makeText(
                            this@WideAnglePhotoCaptureActivity,
                            getString(R.string.camera_ar_capture_failed),
                            Toast.LENGTH_SHORT,
                        ).show()
                    }
                }
            },
        )
    }

    private fun createImageFile(): File {
        val timeStamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val dir = File(cacheDir, "wide_angle_captures").apply { mkdirs() }
        return File(dir, "wide_${timeStamp}.jpg")
    }

    private fun dp(value: Int): Int = (value * resources.displayMetrics.density).toInt()

    override fun onDestroy() {
        cameraExecutor.shutdown()
        super.onDestroy()
    }

    private class RuleOfThirdsGridView(context: android.content.Context) : View(context) {
        private val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.parseColor("#66FFFFFF")
            strokeWidth = 1.5f * resources.displayMetrics.density
            style = Paint.Style.STROKE
        }

        override fun onDraw(canvas: Canvas) {
            super.onDraw(canvas)
            val w = width.toFloat()
            val h = height.toFloat()
            if (w <= 0f || h <= 0f) return
            canvas.drawLine(w / 3f, 0f, w / 3f, h, paint)
            canvas.drawLine(2f * w / 3f, 0f, 2f * w / 3f, h, paint)
            canvas.drawLine(0f, h / 3f, w, h / 3f, paint)
            canvas.drawLine(0f, 2f * h / 3f, w, 2f * h / 3f, paint)
        }
    }
}
