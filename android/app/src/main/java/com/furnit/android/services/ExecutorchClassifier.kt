package com.furnit.android.services

import android.content.Context
import android.util.Log
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor
import java.io.BufferedReader
import java.io.InputStreamReader

/**
 * Loads a MobileNetV3 Small .pte model via ExecuTorch and runs classification.
 */
class ExecutorchClassifier(private val context: Context) {

    companion object {
        private const val TAG = "ExecutorchClassifier"
        private const val MODEL_PATH = "/data/local/tmp/furnit/mobilenet_v3_small.pte"
        private const val LABELS_ASSET = "imagenet_classes.txt"
        private const val NUM_CLASSES = 1000
        private const val TOP_K = 5
    }

    private var module: Module? = null
    private var labels: List<String> = emptyList()

    /**
     * Load the model and labels. Call before classify().
     * Returns true if successful.
     */
    fun initialize(): Boolean {
        return try {
            labels = loadLabels()
            module = Module.load(MODEL_PATH)
            Log.d(TAG, "ExecuTorch model loaded from $MODEL_PATH")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load ExecuTorch model: ${e.message}", e)
            false
        }
    }

    /**
     * Check if the model file exists on device.
     */
    fun isModelAvailable(): Boolean {
        return java.io.File(MODEL_PATH).exists()
    }

    /**
     * Run classification on preprocessed input.
     * @param inputData FloatArray in NCHW format [1, 3, 224, 224]
     * @return Top-5 (label, confidence) pairs, or empty list on error
     */
    fun classify(inputData: FloatArray): List<Pair<String, Float>> {
        val currentModule = module ?: return emptyList()

        return try {
            val inputTensor = Tensor.fromBlob(inputData, longArrayOf(1, 3, 224, 224))
            val outputValues = currentModule.forward(EValue.from(inputTensor))
            val outputTensor = outputValues[0].toTensor()
            val scores = outputTensor.getDataAsFloatArray()

            // Apply softmax
            val probabilities = softmax(scores)

            // Get top-K indices
            val topIndices = probabilities.indices
                .sortedByDescending { probabilities[it] }
                .take(TOP_K)

            topIndices.map { idx ->
                val label = if (idx < labels.size) labels[idx] else "class_$idx"
                Pair(label, probabilities[idx])
            }
        } catch (e: Exception) {
            Log.e(TAG, "Classification failed: ${e.message}", e)
            emptyList()
        }
    }

    /**
     * Release the model resources.
     */
    fun release() {
        module?.destroy()
        module = null
        Log.d(TAG, "ExecuTorch model released")
    }

    private fun loadLabels(): List<String> {
        val labelsList = mutableListOf<String>()
        context.assets.open(LABELS_ASSET).use { inputStream ->
            BufferedReader(InputStreamReader(inputStream)).use { reader ->
                var line = reader.readLine()
                while (line != null) {
                    labelsList.add(line.trim())
                    line = reader.readLine()
                }
            }
        }
        Log.d(TAG, "Loaded ${labelsList.size} labels")
        return labelsList
    }

    private fun softmax(logits: FloatArray): FloatArray {
        val maxLogit = logits.max()
        val exps = FloatArray(logits.size) { kotlin.math.exp((logits[it] - maxLogit).toDouble()).toFloat() }
        val sumExp = exps.sum()
        return FloatArray(exps.size) { exps[it] / sumExp }
    }
}
