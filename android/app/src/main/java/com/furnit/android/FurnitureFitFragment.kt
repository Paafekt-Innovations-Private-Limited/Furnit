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
import android.widget.TextView
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import androidx.lifecycle.lifecycleScope
import com.furnit.android.services.FurnitureFitManager
import io.github.sceneview.SceneView
import io.github.sceneview.math.Position
import io.github.sceneview.node.ModelNode
import kotlinx.coroutines.launch
import java.io.ByteArrayOutputStream
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class FurnitureFitFragment : Fragment() {
    private lateinit var previewView: PreviewView
    private lateinit var roomSceneView: SceneView
    private lateinit var overlay: FurnitureFitOverlayView
    private lateinit var statusLabel: TextView
    private lateinit var cameraExecutor: ExecutorService
    private var cameraProvider: ProcessCameraProvider? = null
    private lateinit var manager: FurnitureFitManager
    private var isProcessing = false
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
        root.addView(roomSceneView)  // 3D room background between camera and overlay
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

        manager.segmentWithDetectionsAsync(bitmap) { result ->
            activity?.runOnUiThread {
                if (result != null && result.mask != null) {
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
                    statusLabel.text = "Scanning..."
                    overlay.setMaskAndDetections(null, emptyList())
                    roomSceneView.visibility = View.GONE
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

    private fun loadRoom3D() {
        val roomId = selectedRoomId ?: "vintage"
        Log.d("FurnitureFit", "Loading 3D room: $roomId")

        // Map room ID to GLB asset path
        val assetPath = when (roomId) {
            "vintage" -> "models/vintage.glb"
            "modern" -> "models/modern.glb"
            "minimalist" -> "models/minimalist.glb"
            else -> "models/$roomId.glb"
        }

        lifecycleScope.launch {
            try {
                val modelNode = ModelNode(
                    modelInstance = roomSceneView.modelLoader.createModelInstance(
                        assetFileLocation = assetPath
                    ),
                    scaleToUnits = null  // Keep original scale
                )

                modelNode.position = Position(0f, 0f, 0f)
                roomSceneView.addChildNode(modelNode)

                // Position camera at back wall, eye level, looking into room
                roomSceneView.cameraNode.apply {
                    position = Position(0f, 1.6f, -4f)
                    lookAt(Position(0f, 1.6f, 4f))
                }

                Log.d("FurnitureFit", "3D room loaded: $assetPath")
            } catch (e: Exception) {
                Log.e("FurnitureFit", "Failed to load 3D room: $assetPath", e)
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
