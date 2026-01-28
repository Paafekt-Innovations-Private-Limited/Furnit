package com.furnit.android

import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.YuvImage
import android.os.Bundle
import android.os.Environment
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.view.*
import android.widget.FrameLayout
import android.widget.ImageButton
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import androidx.fragment.app.Fragment
import androidx.lifecycle.lifecycleScope
import com.furnit.android.services.FurnitureFitManager
import com.furnit.android.views.JoystickView
import io.github.sceneview.SceneView
import io.github.sceneview.math.Position
import io.github.sceneview.node.ModelNode
import kotlinx.coroutines.launch
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileOutputStream
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class FurnitureFitFragment : Fragment() {
    private lateinit var previewView: PreviewView
    private lateinit var roomSceneView: SceneView
    private lateinit var overlay: FurnitureFitOverlayView
    private lateinit var statusLabel: TextView
    private lateinit var progressContainer: LinearLayout
    private lateinit var progressBar: android.widget.ProgressBar
    private lateinit var progressLabel: TextView
    private lateinit var cameraExecutor: ExecutorService
    private var cameraProvider: ProcessCameraProvider? = null
    private lateinit var manager: FurnitureFitManager
    private var isProcessing = false
    private var hasFirstDetection = false
    private var showRoomBackground = true  // Show room behind segmented furniture
    private var selectedRoomId: String? = null
    private var selectedRoomName: String? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Get room info from arguments
        selectedRoomId = arguments?.getString("ROOM_ID")
        selectedRoomName = arguments?.getString("ROOM_NAME")
        Log.d("FurnitureFit", "Fragment onCreate - room: $selectedRoomName (id=$selectedRoomId)")
        cameraExecutor = Executors.newSingleThreadExecutor()
        manager = FurnitureFitManager(requireContext())
        // Initialize model - try NCNN first (1280x1280, more efficient), fall back to ONNX
        Log.d("FurnitureFit", "Calling initializeAuto...")
        val success = manager.initializeAuto()
        Log.d("FurnitureFit", "initializeAuto completed, success=$success")
    }

    override fun onCreateView(inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?): View {
        val root = FrameLayout(requireContext())

        previewView = PreviewView(requireContext()).apply {
            layoutParams = FrameLayout.LayoutParams(FrameLayout.LayoutParams.MATCH_PARENT, FrameLayout.LayoutParams.MATCH_PARENT)
            implementationMode = PreviewView.ImplementationMode.COMPATIBLE
        }

        // Room background layer - 3D room rendered with SceneView
        roomSceneView = SceneView(requireContext()).apply {
            layoutParams = FrameLayout.LayoutParams(FrameLayout.LayoutParams.MATCH_PARENT, FrameLayout.LayoutParams.MATCH_PARENT)
            visibility = View.GONE  // Hidden initially until segmentation starts
        }
        loadRoom3D()

        overlay = FurnitureFitOverlayView(requireContext()).apply {
            layoutParams = FrameLayout.LayoutParams(FrameLayout.LayoutParams.MATCH_PARENT, FrameLayout.LayoutParams.MATCH_PARENT)
        }

        statusLabel = TextView(requireContext()).apply {
            text = "Initializing..."
            setTextColor(0xFFFFFFFF.toInt())
            setShadowLayer(2f, 1f, 1f, 0xFF000000.toInt())
            setPadding(24, 48, 24, 24)
            textSize = 14f
        }

        // Progress container (like iOS progressContainer)
        progressContainer = LinearLayout(requireContext()).apply {
            orientation = LinearLayout.VERTICAL
            gravity = Gravity.CENTER
            setBackgroundColor(0x99000000.toInt())
            setPadding(32, 16, 32, 16)
            val lp = FrameLayout.LayoutParams(FrameLayout.LayoutParams.WRAP_CONTENT, FrameLayout.LayoutParams.WRAP_CONTENT)
            lp.gravity = Gravity.CENTER_HORIZONTAL or Gravity.TOP
            lp.setMargins(0, 120, 0, 0)
            layoutParams = lp
        }

        progressLabel = TextView(requireContext()).apply {
            text = "Starting camera..."
            setTextColor(0xFFFFFFFF.toInt())
            textSize = 14f
            gravity = Gravity.CENTER
            setPadding(8, 4, 8, 8)
        }
        progressContainer.addView(progressLabel)

        progressBar = android.widget.ProgressBar(requireContext(), null, android.R.attr.progressBarStyleHorizontal).apply {
            val lp = LinearLayout.LayoutParams(250.dp, LinearLayout.LayoutParams.WRAP_CONTENT)
            layoutParams = lp
            max = 100
            progress = 5
            progressDrawable.setColorFilter(0xFF4CAF50.toInt(), android.graphics.PorterDuff.Mode.SRC_IN)
        }
        progressContainer.addView(progressBar)

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

        // Bottom controls container
        val bottomControls = FrameLayout(requireContext()).apply {
            val lp = FrameLayout.LayoutParams(FrameLayout.LayoutParams.MATCH_PARENT, FrameLayout.LayoutParams.WRAP_CONTENT)
            lp.gravity = Gravity.BOTTOM
            lp.setMargins(20, 0, 20, 24)
            layoutParams = lp
        }

        // Joystick (center) for camera control
        val joystickContainer = LinearLayout(requireContext()).apply {
            orientation = LinearLayout.VERTICAL
            gravity = Gravity.CENTER
            val lp = FrameLayout.LayoutParams(FrameLayout.LayoutParams.WRAP_CONTENT, FrameLayout.LayoutParams.WRAP_CONTENT)
            lp.gravity = Gravity.CENTER_HORIZONTAL or Gravity.BOTTOM
            layoutParams = lp
        }

        val joystick = JoystickView(requireContext()).apply {
            val size = (100 * resources.displayMetrics.density).toInt()
            layoutParams = LinearLayout.LayoutParams(size, size)
            onJoystickMove = { x, y ->
                moveRoomCamera(x, y)
            }
        }
        joystickContainer.addView(joystick)
        bottomControls.addView(joystickContainer)

        // Screenshot button (right)
        val screenshotButton = ImageButton(requireContext()).apply {
            setImageResource(android.R.drawable.ic_menu_camera)
            setBackgroundResource(android.R.drawable.btn_default)
            val size = (56 * resources.displayMetrics.density).toInt()
            val lp = FrameLayout.LayoutParams(size, size)
            lp.gravity = Gravity.END or Gravity.BOTTOM
            layoutParams = lp
            setOnClickListener { takeScreenshot(root) }
        }
        bottomControls.addView(screenshotButton)

        root.addView(previewView)
        root.addView(roomSceneView)  // 3D room background between camera and overlay
        root.addView(overlay)
        root.addView(statusLabel)
        root.addView(progressContainer)
        root.addView(backButton)
        root.addView(bottomControls)

        // Initial progress
        setProgress(5, "Starting camera...")

        startCamera()
        return root
    }

    private fun moveRoomCamera(normalizedX: Float, normalizedY: Float) {
        val moveSpeed = 0.1f
        val deadZone = 0.1f

        val magnitude = kotlin.math.sqrt(normalizedX * normalizedX + normalizedY * normalizedY)
        if (magnitude < deadZone) return

        val camera = roomSceneView.cameraNode
        val position = camera.position

        val deltaX = normalizedX * moveSpeed
        val deltaZ = normalizedY * moveSpeed

        camera.position = Position(
            position.x + deltaX,
            position.y,
            position.z + deltaZ
        )
    }

    private fun takeScreenshot(rootView: View) {
        try {
            // First capture the 3D room using PixelCopy (for OpenGL content)
            val roomBitmap = Bitmap.createBitmap(roomSceneView.width, roomSceneView.height, Bitmap.Config.ARGB_8888)

            android.view.PixelCopy.request(
                roomSceneView,
                roomBitmap,
                { copyResult ->
                    if (copyResult == android.view.PixelCopy.SUCCESS) {
                        // Now composite the overlay on top of the room
                        val compositeBitmap = Bitmap.createBitmap(rootView.width, rootView.height, Bitmap.Config.ARGB_8888)
                        val canvas = Canvas(compositeBitmap)

                        // Draw room background (scaled to fit)
                        canvas.drawBitmap(roomBitmap, null, android.graphics.RectF(0f, 0f, rootView.width.toFloat(), rootView.height.toFloat()), null)

                        // Draw overlay on top
                        overlay.draw(canvas)

                        // Save the composite
                        saveScreenshotToGallery(compositeBitmap)
                    } else {
                        // Fallback: just capture the overlay with black background
                        activity?.runOnUiThread {
                            val fallbackBitmap = Bitmap.createBitmap(rootView.width, rootView.height, Bitmap.Config.ARGB_8888)
                            val canvas = Canvas(fallbackBitmap)
                            canvas.drawColor(Color.BLACK)
                            overlay.draw(canvas)
                            saveScreenshotToGallery(fallbackBitmap)
                        }
                    }
                },
                Handler(Looper.getMainLooper())
            )
        } catch (e: Exception) {
            Log.e("FurnitureFit", "Screenshot failed", e)
            Toast.makeText(requireContext(), "Screenshot failed: ${e.message}", Toast.LENGTH_SHORT).show()
        }
    }

    private fun saveScreenshotToGallery(bitmap: Bitmap) {
        try {
            val timeStamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())
            val fileName = "FurnitureFit_$timeStamp.png"

            // Save to gallery using MediaStore (Android 10+)
            val contentValues = android.content.ContentValues().apply {
                put(android.provider.MediaStore.Images.Media.DISPLAY_NAME, fileName)
                put(android.provider.MediaStore.Images.Media.MIME_TYPE, "image/png")
                put(android.provider.MediaStore.Images.Media.RELATIVE_PATH, "${Environment.DIRECTORY_PICTURES}/Screenshots")
            }

            val resolver = requireContext().contentResolver
            val uri = resolver.insert(android.provider.MediaStore.Images.Media.EXTERNAL_CONTENT_URI, contentValues)

            if (uri != null) {
                resolver.openOutputStream(uri)?.use { out ->
                    bitmap.compress(Bitmap.CompressFormat.PNG, 100, out)
                }

                activity?.runOnUiThread {
                    Toast.makeText(requireContext(), "Saved to Screenshots", Toast.LENGTH_SHORT).show()
                }

                // Share the screenshot
                val shareIntent = Intent(Intent.ACTION_SEND).apply {
                    type = "image/png"
                    putExtra(Intent.EXTRA_STREAM, uri)
                    addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                }
                startActivity(Intent.createChooser(shareIntent, "Share Screenshot"))
            } else {
                activity?.runOnUiThread {
                    Toast.makeText(requireContext(), "Failed to save screenshot", Toast.LENGTH_SHORT).show()
                }
            }
        } catch (e: Exception) {
            Log.e("FurnitureFit", "Save screenshot failed", e)
            activity?.runOnUiThread {
                Toast.makeText(requireContext(), "Failed to save: ${e.message}", Toast.LENGTH_SHORT).show()
            }
        }
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
            Log.e("FurnitureFit", "bindToLifecycle failed", e)
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
            Log.w("FurnitureFit", "Failed to convert imageProxy to bitmap")
            isProcessing = false
            imageProxy.close()
            return
        }

        // Update progress during processing
        if (!hasFirstDetection) {
            setProgress(15, "Preprocessing...")
        }

        manager.segmentWithDetectionsAsync(bitmap) { result ->
            activity?.runOnUiThread {
                if (result != null && result.mask != null) {
                    // First detection - hide progress bar
                    if (!hasFirstDetection) {
                        hasFirstDetection = true
                        progressContainer.visibility = View.GONE
                    }

                    // Show what was detected
                    val labels = result.detections.take(3).joinToString(", ") {
                        "${it.label} ${(it.confidence * 100).toInt()}%"
                    }
                    statusLabel.text = if (labels.isNotEmpty()) labels else "Detected"
                    overlay.setMaskAndDetections(result.mask, result.detections, result.inputSize)

                    // Show 3D room behind segmented furniture
                    if (showRoomBackground && result.detections.isNotEmpty()) {
                        roomSceneView.visibility = View.VISIBLE
                    }
                } else {
                    if (!hasFirstDetection) {
                        setProgress(40, "Scanning for furniture...")
                    }
                    statusLabel.text = "Scanning..."
                    overlay.setMaskAndDetections(null, emptyList())
                    roomSceneView.visibility = View.GONE
                }
            }
            isProcessing = false
        }

        imageProxy.close()
    }

    private fun setProgress(value: Int, text: String) {
        if (hasFirstDetection) return
        activity?.runOnUiThread {
            progressContainer.visibility = View.VISIBLE
            progressBar.progress = value
            progressLabel.text = text
        }
    }

    private val Int.dp: Int
        get() = (this * resources.displayMetrics.density).toInt()

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        manager.close()
    }

    private fun loadRoom3D() {
        val roomId = selectedRoomId ?: "vintage"
        Log.d("FurnitureFit", "Loading 3D room: $roomId")

        lifecycleScope.launch {
            try {
                val modelInstance = when {
                    // Bundled rooms in assets
                    roomId == "vintage" || roomId == "cozy_room" -> {
                        val assetPath = "models/$roomId.glb"
                        Log.d("FurnitureFit", "Loading from assets: $assetPath")
                        roomSceneView.modelLoader.createModelInstance(assetFileLocation = assetPath)
                    }
                    // User-created rooms in filesDir
                    else -> {
                        val roomsDir = java.io.File(requireContext().filesDir, "rooms")
                        val roomFolder = java.io.File(roomsDir, roomId)
                        val glbFile = java.io.File(roomFolder, "room.glb")

                        if (glbFile.exists()) {
                            Log.d("FurnitureFit", "Loading from file: ${glbFile.absolutePath}")
                            val bytes = glbFile.readBytes()
                            val buffer = java.nio.ByteBuffer.wrap(bytes)
                            roomSceneView.modelLoader.createModelInstance(buffer)
                        } else {
                            // Fallback to vintage if room not found
                            Log.w("FurnitureFit", "Room not found: $roomId, falling back to vintage")
                            roomSceneView.modelLoader.createModelInstance(assetFileLocation = "models/vintage.glb")
                        }
                    }
                }

                val modelNode = ModelNode(
                    modelInstance = modelInstance,
                    scaleToUnits = null  // Keep original scale
                )

                modelNode.position = Position(0f, 0f, 0f)
                roomSceneView.addChildNode(modelNode)

                // Position camera at back wall, eye level, looking at front wall
                roomSceneView.cameraNode.apply {
                    position = Position(0f, 1.6f, 4f)   // At back of room
                    lookAt(Position(0f, 1.4f, -4f))     // Looking at front wall
                }

                Log.d("FurnitureFit", "3D room loaded successfully")
            } catch (e: Exception) {
                Log.e("FurnitureFit", "Failed to load 3D room", e)
            }
        }
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
        Log.e("FurnitureFit", "toBitmap failed: ${e.message}")
        e.printStackTrace()
        null
    }
}
