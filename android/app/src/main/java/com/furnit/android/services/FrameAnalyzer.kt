package com.furnit.android.services

import android.util.Log
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import java.util.concurrent.atomic.AtomicBoolean

/**
 * CameraX ImageAnalysis.Analyzer that throttles to ~10 FPS
 * and runs ExecuTorch classification on each frame.
 */
class FrameAnalyzer(
    private val classifier: ExecutorchClassifier,
    private val onResults: (List<Pair<String, Float>>, Float) -> Unit
) : ImageAnalysis.Analyzer {

    companion object {
        private const val TAG = "FrameAnalyzer"
        private const val MIN_INTERVAL_MS = 100L // ~10 FPS max
    }

    private val isProcessing = AtomicBoolean(false)
    private var lastAnalysisTime = 0L

    override fun analyze(imageProxy: ImageProxy) {
        val currentTime = System.currentTimeMillis()

        // Throttle: skip frame if still processing or too soon
        if (isProcessing.get() || currentTime - lastAnalysisTime < MIN_INTERVAL_MS) {
            imageProxy.close()
            return
        }

        isProcessing.set(true)
        lastAnalysisTime = currentTime

        try {
            val bitmap = ImagePreprocessor.imageProxyToBitmap(imageProxy)
            if (bitmap == null) {
                imageProxy.close()
                isProcessing.set(false)
                return
            }

            val preprocessedData = ImagePreprocessor.preprocessBitmap(bitmap)
            bitmap.recycle()

            val inferenceStart = System.currentTimeMillis()
            val results = classifier.classify(preprocessedData)
            val inferenceTimeMs = System.currentTimeMillis() - inferenceStart

            val fps = if (inferenceTimeMs > 0) 1000f / inferenceTimeMs else 0f

            onResults(results, fps)
        } catch (e: Exception) {
            Log.e(TAG, "Frame analysis failed: ${e.message}", e)
        } finally {
            imageProxy.close()
            isProcessing.set(false)
        }
    }
}
