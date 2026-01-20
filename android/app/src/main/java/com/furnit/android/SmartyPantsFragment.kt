package com.furnit.android

import android.graphics.Bitmap
import android.os.Bundle
import android.util.Log
import android.view.*
import android.widget.FrameLayout
import android.widget.ProgressBar
import android.widget.TextView
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import com.furnit.android.services.SmartyPantsManager
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class SmartyPantsFragment : Fragment() {
    private lateinit var previewView: PreviewView
    private lateinit var overlay: SmartyPantsOverlayView
    private lateinit var progressBar: ProgressBar
    private lateinit var progressLabel: TextView
    private lateinit var cameraExecutor: ExecutorService
    private var cameraProvider: ProcessCameraProvider? = null
    private lateinit var manager: SmartyPantsManager

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        cameraExecutor = Executors.newSingleThreadExecutor()
        manager = SmartyPantsManager(requireContext())
    }

    override fun onCreateView(inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?): View {
        val root = FrameLayout(requireContext())
        previewView = PreviewView(requireContext()).apply {
            layoutParams = FrameLayout.LayoutParams(FrameLayout.LayoutParams.MATCH_PARENT, FrameLayout.LayoutParams.MATCH_PARENT)
            implementationMode = PreviewView.ImplementationMode.COMPATIBLE
        }
        overlay = SmartyPantsOverlayView(requireContext())
        progressBar = ProgressBar(requireContext(), null, android.R.attr.progressBarStyleHorizontal).apply {
            isIndeterminate = false
            max = 100
            progress = 0
            val lp = FrameLayout.LayoutParams(FrameLayout.LayoutParams.MATCH_PARENT, 8)
            lp.topMargin = 40
            layoutParams = lp
        }
        progressLabel = TextView(requireContext()).apply {
            text = ""
            setPadding(8,8,8,8)
        }

        root.addView(previewView)
        root.addView(overlay)
        root.addView(progressBar)
        root.addView(progressLabel)
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
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build()

        analysis.setAnalyzer(cameraExecutor) { imageProxy ->
            processFrame(imageProxy)
        }

        try {
            cameraProvider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis)
        } catch (e: Exception) {
            Log.e("SmartyPants", "bindToLifecycle failed", e)
        }
    }

    private fun processFrame(imageProxy: ImageProxy) {
        // Convert to Bitmap quickly (basic YUV->RGB conversion) and pass to manager
        val bitmap = imageProxy.toBitmap() // extension helper below
        // Update progress UI
        activity?.runOnUiThread {
            progressLabel.text = "Processing"
            progressBar.progress = 10
        }

        manager.segmentImageAsync(bitmap) { maskBitmap ->
            // maskBitmap may be null for stub
            activity?.runOnUiThread {
                progressBar.progress = 100
                progressLabel.text = "Done"
                overlay.setMask(maskBitmap)
            }
        }

        imageProxy.close()
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        manager.close()
    }
}

// Simple extension to convert ImageProxy (YUV) to Bitmap. This is an approximate conversion and
// sufficient for prototyping. For production use a more efficient pipeline or RenderScript.
fun ImageProxy.toBitmap(): Bitmap? {
    val plane = planes[0]
    val buffer = plane.buffer
    val bytes = ByteArray(buffer.capacity())
    buffer.get(bytes)
    // Not a correct conversion; return null if we can't convert quickly.
    return null
}
