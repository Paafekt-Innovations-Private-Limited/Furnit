package com.furnit.android

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.ImageFormat
import android.graphics.Paint
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.hardware.camera2.CameraCharacteristics
import android.os.Bundle
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.widget.FrameLayout
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
import kotlin.math.atan

internal data class RoomCameraCandidate(
    val cameraId: String,
    val horizontalFovDegrees: Double,
    val maxJpegPixels: Long,
)

internal fun selectWidestCaptureQualityCamera(
    candidates: List<RoomCameraCandidate>,
    minimumJpegPixels: Long,
): RoomCameraCandidate? {
    val captureQualityCandidates = candidates.filter { it.maxJpegPixels >= minimumJpegPixels }
    return captureQualityCandidates.ifEmpty { candidates }
        .maxWithOrNull(
            compareBy<RoomCameraCandidate> { it.horizontalFovDegrees }
                .thenBy { it.maxJpegPixels },
        )
}

/**
 * Ultra-wide / wide back-camera capture for room creation (iOS `WideAngleCameraView` parity).
 * Landscape-locked, flash off, rule-of-thirds grid, widest capture-quality back camera available.
 */
class WideAnglePhotoCaptureActivity : AppCompatActivity() {
    companion object {
        const val EXTRA_CAPTURED_IMAGE_URI = "captured_image_uri"
        private const val TAG = "WideAngleCapture"
        private const val MIN_QUALITY_JPEG_PIXELS = 5_000_000L
    }

    private lateinit var previewView: PreviewView
    private lateinit var zoomLabel: TextView
    private lateinit var statusLabel: TextView
    private lateinit var captureButton: FrameLayout
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
            implementationMode = PreviewView.ImplementationMode.PERFORMANCE
            scaleType = PreviewView.ScaleType.FILL_CENTER
        }

        statusLabel = TextView(this).apply {
            text = getString(R.string.camera_wide_angle_hold_steady)
            setTextColor(Color.WHITE)
            textSize = 15f
            gravity = Gravity.CENTER
            setTypeface(null, Typeface.BOLD)
            setPadding(dp(16), dp(12), dp(16), dp(12))
            background = roundedRect(Color.parseColor("#73000000"), dp(8).toFloat())
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
            background = roundedRect(Color.parseColor("#66000000"), dp(8).toFloat())
        }

        val gridOverlay = RuleOfThirdsGridView(this)

        captureButton = createShutterButton().apply {
            contentDescription = getString(R.string.camera_capture_wide_angle)
            setOnClickListener { capturePhoto() }
        }

        val cancelButton = TextView(this).apply {
            text = getString(R.string.common_cancel)
            setTextColor(Color.WHITE)
            textSize = 18f
            setTypeface(null, Typeface.BOLD)
            gravity = Gravity.CENTER
            setPadding(dp(20), dp(12), dp(20), dp(12))
            setOnClickListener {
                setResult(RESULT_CANCELED)
                finish()
            }
        }

        val bottomBar = FrameLayout(this).apply {
            setPadding(dp(16), dp(8), dp(16), dp(28))
            background = GradientDrawable(
                GradientDrawable.Orientation.TOP_BOTTOM,
                intArrayOf(Color.TRANSPARENT, Color.parseColor("#8A000000")),
            )
            // Add the navigation bar inset so controls clear the gesture/nav bar.
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
                val camera = provider.bindToLifecycle(this, selector, preview, imageCapture)
                applyWidestLogicalZoom(camera)
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
        val candidates = mutableListOf<RoomCameraCandidate>()
        for (info in provider.availableCameraInfos) {
            val cam2 = Camera2CameraInfo.from(info)
            val facing = cam2.getCameraCharacteristic(CameraCharacteristics.LENS_FACING)
            if (facing == null || facing != CameraCharacteristics.LENS_FACING_BACK) continue
            val focals = cam2.getCameraCharacteristic(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)
                ?: continue
            val minFocal = focals.minOrNull() ?: continue
            val sensorSize = cam2.getCameraCharacteristic(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE)
                ?: continue
            if (sensorSize.width <= 0f || minFocal <= 0f) continue
            val horizontalFov = Math.toDegrees(
                2.0 * atan((sensorSize.width / (2f * minFocal)).toDouble()),
            )
            val streamMap = cam2.getCameraCharacteristic(
                CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP,
            )
            val maxJpegPixels = try {
                streamMap
                    ?.getOutputSizes(ImageFormat.JPEG)
                    ?.maxOfOrNull { size -> size.width.toLong() * size.height.toLong() }
                    ?: 0L
            } catch (_: RuntimeException) {
                0L
            }
            candidates += RoomCameraCandidate(cam2.cameraId, horizontalFov, maxJpegPixels)
        }

        // Raw focal length alone can select a tiny, low-resolution macro/depth sensor. Rank by
        // actual horizontal field of view and exclude sub-5 MP auxiliary cameras when a
        // capture-quality back camera is available.
        val best = selectWidestCaptureQualityCamera(candidates, MIN_QUALITY_JPEG_PIXELS)
        usingUltraWide = best != null && best.horizontalFovDegrees >= 80.0
        if (best != null) {
            LogUtil.d(
                TAG,
                "Selected back camera id=${best.cameraId} horizontalFov=${"%.1f".format(Locale.US, best.horizontalFovDegrees)} " +
                    "maxJpegPixels=${best.maxJpegPixels}",
            )
        }

        return if (best != null) {
            val targetId = best.cameraId
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

    private fun applyWidestLogicalZoom(camera: androidx.camera.core.Camera) {
        val zoomState = camera.cameraInfo.zoomState.value
        val minZoom = zoomState?.minZoomRatio ?: 1f
        if (minZoom < 0.95f) {
            camera.cameraControl.setZoomRatio(minZoom)
            usingUltraWide = true
            zoomLabel.text = getString(R.string.camera_zoom_ultra_wide)
        } else {
            usingUltraWide = usingUltraWide && minZoom <= 1f
            zoomLabel.text = if (usingUltraWide) {
                getString(R.string.camera_zoom_ultra_wide)
            } else {
                getString(R.string.camera_zoom_wide)
            }
        }
        LogUtil.d(TAG, "Camera zoom min=$minZoom ultraWide=$usingUltraWide")
    }

    private fun capturePhoto() {
        if (capturing) return
        val capture = imageCapture ?: return
        capturing = true
        captureButton.isEnabled = false
        previewView.display?.rotation?.let { capture.targetRotation = it }

        val photoFile = createImageFile()
        val outputOptions = ImageCapture.OutputFileOptions.Builder(photoFile).build()
        capture.takePicture(
            outputOptions,
            cameraExecutor,
            object : ImageCapture.OnImageSavedCallback {
                override fun onImageSaved(outputFileResults: ImageCapture.OutputFileResults) {
                    val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
                    BitmapFactory.decodeFile(photoFile.absolutePath, bounds)
                    LogUtil.d(
                        TAG,
                        "Saved wide capture ${bounds.outWidth}x${bounds.outHeight} " +
                            "bytes=${photoFile.length()} ultraWide=$usingUltraWide",
                    )
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
