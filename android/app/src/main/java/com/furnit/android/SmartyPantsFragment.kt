package com.furnit.android

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.YuvImage
import android.os.Bundle
import android.util.Log
import android.view.*
import android.widget.FrameLayout
import android.widget.ImageButton
import android.widget.ImageView
import android.widget.ProgressBar
import android.widget.TextView
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import com.furnit.android.services.SmartyPantsManager
import java.io.ByteArrayOutputStream
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class SmartyPantsFragment : Fragment() {
    private lateinit var previewView: PreviewView
    private lateinit var roomBackgroundView: ImageView
    private lateinit var overlay: SmartyPantsOverlayView
    private lateinit var statusLabel: TextView
    private lateinit var cameraExecutor: ExecutorService
    private var cameraProvider: ProcessCameraProvider? = null
    private lateinit var manager: SmartyPantsManager
    private var isProcessing = false
    private var showRoomBackground = true  // Show room behind segmented furniture

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        Log.d("SmartyPants", "Fragment onCreate - initializing...")
        cameraExecutor = Executors.newSingleThreadExecutor()
        manager = SmartyPantsManager(requireContext())
        // Initialize model - try NCNN first (1280x1280, more efficient), fall back to ONNX
        Log.d("SmartyPants", "Calling initializeAuto...")
        val success = manager.initializeAuto()
        Log.d("SmartyPants", "initializeAuto completed, success=$success")
    }

    override fun onCreateView(inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?): View {
        val root = FrameLayout(requireContext())

        previewView = PreviewView(requireContext()).apply {
            layoutParams = FrameLayout.LayoutParams(FrameLayout.LayoutParams.MATCH_PARENT, FrameLayout.LayoutParams.MATCH_PARENT)
            implementationMode = PreviewView.ImplementationMode.COMPATIBLE
        }

        // Room background layer - shows behind segmented furniture
        roomBackgroundView = ImageView(requireContext()).apply {
            layoutParams = FrameLayout.LayoutParams(FrameLayout.LayoutParams.MATCH_PARENT, FrameLayout.LayoutParams.MATCH_PARENT)
            scaleType = ImageView.ScaleType.CENTER_CROP
            visibility = View.GONE  // Hidden initially until segmentation starts
        }
        loadRoomBackground()

        overlay = SmartyPantsOverlayView(requireContext()).apply {
            layoutParams = FrameLayout.LayoutParams(FrameLayout.LayoutParams.MATCH_PARENT, FrameLayout.LayoutParams.MATCH_PARENT)
        }

        statusLabel = TextView(requireContext()).apply {
            text = "Initializing..."
            setTextColor(0xFFFFFFFF.toInt())
            setShadowLayer(2f, 1f, 1f, 0xFF000000.toInt())
            setPadding(24, 48, 24, 24)
            textSize = 14f
        }

        // Back button
        val backButton = ImageButton(requireContext()).apply {
            setImageResource(android.R.drawable.ic_menu_close_clear_cancel)
            setBackgroundColor(0x80000000.toInt())
            setPadding(16, 16, 16, 16)
            val lp = FrameLayout.LayoutParams(FrameLayout.LayoutParams.WRAP_CONTENT, FrameLayout.LayoutParams.WRAP_CONTENT)
            lp.gravity = Gravity.TOP or Gravity.START
            lp.setMargins(16, 48, 0, 0)
            layoutParams = lp
            setOnClickListener { activity?.finish() }
        }

        root.addView(previewView)
        root.addView(roomBackgroundView)  // Room background between camera and overlay
        root.addView(overlay)
        root.addView(statusLabel)
        root.addView(backButton)

        startCamera()
        return root
    }

    private fun startCamera() {
        val camProviderFuture = ProcessCameraProvider.getInstance(requireContext())
        camProviderFuture.addListener({
            cameraProvider = camProviderFuture.get()
            bindCameraUseCases()
        }, ContextCompat.getMainExecutor(requireContext()))
    }

    private fun bindCameraUseCases() {
        val cameraProvider = cameraProvider ?: return
        cameraProvider.unbindAll()

        val preview = Preview.Builder().build().also {
            it.setSurfaceProvider(previewView.surfaceProvider)
        }

        val analysis = ImageAnalysis.Builder()
            .setTargetResolution(android.util.Size(768, 768))
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build()

        analysis.setAnalyzer(cameraExecutor) { imageProxy ->
            processFrame(imageProxy)
        }

        try {
            cameraProvider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis)
            activity?.runOnUiThread {
                statusLabel.text = "Camera ready - detecting furniture..."
            }
        } catch (e: Exception) {
            Log.e("SmartyPants", "bindToLifecycle failed", e)
            activity?.runOnUiThread {
                statusLabel.text = "Camera error: ${e.message}"
            }
        }
    }

    private fun processFrame(imageProxy: ImageProxy) {
        if (isProcessing) {
            imageProxy.close()
            return
        }
        isProcessing = true

        val bitmap = imageProxy.toBitmapSafe()
        if (bitmap == null) {
            Log.w("SmartyPants", "Failed to convert imageProxy to bitmap")
            isProcessing = false
            imageProxy.close()
            return
        }

        manager.segmentWithDetectionsAsync(bitmap) { result ->
            activity?.runOnUiThread {
                if (result != null && result.mask != null) {
                    // Show what was detected
                    val labels = result.detections.take(3).joinToString(", ") {
                        "${it.label} ${(it.confidence * 100).toInt()}%"
                    }
                    statusLabel.text = if (labels.isNotEmpty()) labels else "Detected"
                    overlay.setMaskAndDetections(result.mask, result.detections, result.inputSize)

                    // Show room background behind segmented furniture
                    if (showRoomBackground && result.detections.isNotEmpty()) {
                        roomBackgroundView.visibility = View.VISIBLE
                    }
                } else {
                    statusLabel.text = "Scanning..."
                    overlay.setMaskAndDetections(null, emptyList())
                    roomBackgroundView.visibility = View.GONE
                }
            }
            isProcessing = false
        }

        imageProxy.close()
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        manager.close()
    }

    private fun loadRoomBackground() {
        // Try to load Vintage Living Room or first available room
        try {
            // First try assets
            val assetFiles = requireContext().assets.list("room_previews") ?: emptyArray()
            if (assetFiles.contains("vintage.png")) {
                val bitmap = requireContext().assets.open("room_previews/vintage.png").use {
                    BitmapFactory.decodeStream(it)
                }
                roomBackgroundView.setImageBitmap(bitmap)
                Log.d("SmartyPants", "Loaded vintage room from assets")
                return
            }

            // Try user-created rooms
            val roomsDir = java.io.File(requireContext().filesDir, "rooms")
            if (roomsDir.exists()) {
                val roomFolders = roomsDir.listFiles { f -> f.isDirectory }
                if (!roomFolders.isNullOrEmpty()) {
                    val frontWall = java.io.File(roomFolders[0], "front_wall.png")
                    if (frontWall.exists()) {
                        val bitmap = BitmapFactory.decodeFile(frontWall.absolutePath)
                        roomBackgroundView.setImageBitmap(bitmap)
                        Log.d("SmartyPants", "Loaded room: ${roomFolders[0].name}")
                        return
                    }
                }
            }

            // Fallback: create a simple room-like background
            val bitmap = createDefaultRoomBackground()
            roomBackgroundView.setImageBitmap(bitmap)
            Log.d("SmartyPants", "Using default room background")
        } catch (e: Exception) {
            Log.e("SmartyPants", "Failed to load room background", e)
            val bitmap = createDefaultRoomBackground()
            roomBackgroundView.setImageBitmap(bitmap)
        }
    }

    private fun createDefaultRoomBackground(): Bitmap {
        val width = 1080
        val height = 1920
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val canvas = android.graphics.Canvas(bitmap)

        // Wall color (warm beige like vintage room)
        val wallPaint = android.graphics.Paint().apply { color = 0xFFD4C4A8.toInt() }
        // Floor color (wood-like)
        val floorPaint = android.graphics.Paint().apply { color = 0xFF8B7355.toInt() }
        // Baseboard
        val baseboardPaint = android.graphics.Paint().apply { color = 0xFFE8DCC8.toInt() }

        // Draw wall (top 70%)
        canvas.drawRect(0f, 0f, width.toFloat(), height * 0.7f, wallPaint)

        // Draw baseboard
        canvas.drawRect(0f, height * 0.68f, width.toFloat(), height * 0.72f, baseboardPaint)

        // Draw floor (bottom 30%)
        canvas.drawRect(0f, height * 0.7f, width.toFloat(), height.toFloat(), floorPaint)

        return bitmap
    }
}

// Convert ImageProxy to Bitmap - handles YUV_420_888 format with proper stride handling
fun ImageProxy.toBitmapSafe(): Bitmap? {
    return try {
        val yPlane = planes[0]
        val uPlane = planes[1]
        val vPlane = planes[2]

        val yBuffer = yPlane.buffer
        val uBuffer = uPlane.buffer
        val vBuffer = vPlane.buffer

        val yRowStride = yPlane.rowStride
        val uvRowStride = uPlane.rowStride
        val uvPixelStride = uPlane.pixelStride

        // Build NV21 byte array with proper stride handling
        val nv21 = ByteArray(width * height * 3 / 2)

        // Copy Y plane row by row (handle stride)
        var pos = 0
        for (row in 0 until height) {
            yBuffer.position(row * yRowStride)
            yBuffer.get(nv21, pos, width)
            pos += width
        }

        // Copy UV planes (interleaved as VU for NV21)
        val uvHeight = height / 2
        val uvWidth = width / 2
        for (row in 0 until uvHeight) {
            for (col in 0 until uvWidth) {
                val uvIndex = row * uvRowStride + col * uvPixelStride
                vBuffer.position(uvIndex)
                uBuffer.position(uvIndex)
                nv21[pos++] = vBuffer.get()
                nv21[pos++] = uBuffer.get()
            }
        }

        val yuvImage = YuvImage(nv21, ImageFormat.NV21, width, height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, width, height), 90, out)
        val imageBytes = out.toByteArray()

        var bitmap = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)

        // Rotate if needed based on image rotation
        val rotation = imageInfo.rotationDegrees
        if (rotation != 0) {
            val matrix = Matrix()
            matrix.postRotate(rotation.toFloat())
            bitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
        }
        bitmap
    } catch (e: Exception) {
        Log.e("SmartyPants", "toBitmap failed: ${e.message}")
        e.printStackTrace()
        null
    }
}
