package com.furnit.android.services

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.pytorch.IValue
import org.pytorch.LiteModuleLoader
import org.pytorch.Tensor
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.text.SimpleDateFormat
import java.util.*
import kotlin.math.*

/**
 * SHARP inference via PyTorch Mobile Lite.
 *
 * Loads .ptl (TorchScript Lite) directly -- same weights as Python.
 * No ONNX/TFLite/ExecuTorch conversion needed.
 *
 * Model: sharp_mobile.ptl (~2.5GB)
 * Input: [1, 3, 1536, 1536] float32
 * Output: [N, 14] Gaussian parameters
 */
class TorchMobileSharp private constructor(private val context: Context) {

    companion object {
        private const val TAG = "TorchMobileSharp"
        private const val MODEL_FILENAME = "sharp_mobile.ptl"
        private const val IMAGE_SIZE = 1536
        private const val PARAMS_PER_GAUSSIAN = 14
        private const val SH_C0 = 0.28209479177387814f
        private const val BYTES_PER_VERTEX = 62 * 4

        private const val LOGIT_LUT_SIZE = 1024
        private val LOGIT_LUT = FloatArray(LOGIT_LUT_SIZE) { i ->
            val p = (i.toFloat() / (LOGIT_LUT_SIZE - 1)).coerceIn(1e-4f, 1f - 1e-4f)
            ln(p / (1f - p))
        }

        @Volatile
        private var instance: TorchMobileSharp? = null

        fun getInstance(context: Context): TorchMobileSharp {
            return instance ?: synchronized(this) {
                instance ?: TorchMobileSharp(context.applicationContext).also { instance = it }
            }
        }
    }

    private var module: org.pytorch.Module? = null
    private var isInitialized = false
    private val modelsDir: File by lazy {
        context.getExternalFilesDir("models") ?: File(context.filesDir, "models")
    }

    data class StreamingResult(
        val plyFile: File,
        val classicPlyFile: File,
        val gaussianCount: Int,
        val roomWidth: Float,
        val roomHeight: Float,
        val roomDepth: Float
    )

    private fun findModelFile(): File? {
        val f = File(modelsDir, MODEL_FILENAME)
        if (f.exists() && f.length() > 0) return f
        val alt = File("/data/local/tmp/furnit/", MODEL_FILENAME)
        if (alt.exists() && alt.length() > 0) return alt
        return null
    }

    fun isModelReady(): Boolean = findModelFile() != null

    /**
     * Initialize and PRE-LOAD model into memory.
     * Call early (e.g. when user enters Single Photo screen) so model
     * is ready when Generate is tapped. Spreads memory load over time.
     */
    fun initialize(): Boolean {
        val modelFile = findModelFile()
        if (modelFile == null) {
            Log.e(TAG, "Model not found: $MODEL_FILENAME")
            return false
        }
        val sizeMB = modelFile.length() / 1024 / 1024
        Log.d(TAG, "Model found: ${modelFile.name} (${sizeMB}MB)")

        // Pre-load model NOW so it's ready for inference later
        Log.d(TAG, "Pre-loading model into memory...")
        val loadStart = System.currentTimeMillis()
        try {
            module = LiteModuleLoader.load(modelFile.absolutePath)
            val loadTime = System.currentTimeMillis() - loadStart
            Log.d(TAG, "Model pre-loaded in ${loadTime}ms")
        } catch (e: Exception) {
            Log.e(TAG, "Model pre-load failed: ${e.message}")
            // Don't fail init -- will retry during inference
        }

        isInitialized = true
        return true
    }

    suspend fun inferStreaming(
        bitmap: Bitmap,
        progressCallback: ((Float, String) -> Unit)? = null
    ): StreamingResult? = withContext(Dispatchers.IO) {
        if (!isInitialized) {
            Log.e(TAG, "Not initialized")
            return@withContext null
        }

        try {
            val startTime = System.currentTimeMillis()

            // Preprocess image to [1, 3, 1536, 1536]
            progressCallback?.invoke(0.05f, "Preprocessing image...")
            val scaledBitmap = Bitmap.createScaledBitmap(bitmap, IMAGE_SIZE, IMAGE_SIZE, true)
            val pixels = IntArray(IMAGE_SIZE * IMAGE_SIZE)
            scaledBitmap.getPixels(pixels, 0, IMAGE_SIZE, 0, 0, IMAGE_SIZE, IMAGE_SIZE)
            scaledBitmap.recycle()

            val floatData = FloatArray(3 * IMAGE_SIZE * IMAGE_SIZE)
            val channelSize = IMAGE_SIZE * IMAGE_SIZE
            val inv255 = 1f / 255f
            for (i in pixels.indices) {
                floatData[i] = ((pixels[i] shr 16) and 0xFF) * inv255
                floatData[channelSize + i] = ((pixels[i] shr 8) and 0xFF) * inv255
                floatData[2 * channelSize + i] = (pixels[i] and 0xFF) * inv255
            }

            val inputTensor = Tensor.fromBlob(floatData, longArrayOf(1, 3, IMAGE_SIZE.toLong(), IMAGE_SIZE.toLong()))
            Log.d(TAG, "Image preprocessed")

            // Use pre-loaded model (or load now if not pre-loaded)
            val loadTime: Long
            if (module == null) {
                progressCallback?.invoke(0.10f, "Loading PyTorch model...")
                val modelFile = findModelFile()!!
                val loadStart = System.currentTimeMillis()
                module = LiteModuleLoader.load(modelFile.absolutePath)
                loadTime = System.currentTimeMillis() - loadStart
                Log.d(TAG, "Model loaded on-demand in ${loadTime}ms")
            } else {
                loadTime = 0
                Log.d(TAG, "Using pre-loaded model")
            }

            // Run inference -- single forward pass, same as Python
            progressCallback?.invoke(0.20f, "Running PyTorch inference...")
            val inferStart = System.currentTimeMillis()
            val output = module!!.forward(IValue.from(inputTensor)).toTensor()
            val inferTime = System.currentTimeMillis() - inferStart
            Log.d(TAG, "Inference completed in ${inferTime}ms")

            val outputData = output.dataAsFloatArray
            val gaussianCount = outputData.size / PARAMS_PER_GAUSSIAN
            Log.d(TAG, "Produced $gaussianCount Gaussians")

            // Write PLY
            progressCallback?.invoke(0.80f, "Writing PLY ($gaussianCount Gaussians)...")
            val roomsDir = File(context.filesDir, "sharp_rooms")
            val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
            val roomFolder = File(roomsDir, "room_$timestamp")
            roomFolder.mkdirs()

            val plyFile = File(roomFolder, "room.ply")
            val classicPlyFile = File(roomFolder, "room_classic.ply")

            writePly(plyFile, outputData, gaussianCount)
            plyFile.copyTo(classicPlyFile, overwrite = true)

            // Room bounds
            var minX = Float.MAX_VALUE; var maxX = -Float.MAX_VALUE
            var minY = Float.MAX_VALUE; var maxY = -Float.MAX_VALUE
            var minZ = Float.MAX_VALUE; var maxZ = -Float.MAX_VALUE
            for (i in 0 until gaussianCount) {
                val off = i * PARAMS_PER_GAUSSIAN
                val x = outputData[off]; val y = outputData[off + 1]; val z = outputData[off + 2]
                if (x < minX) minX = x; if (x > maxX) maxX = x
                if (y < minY) minY = y; if (y > maxY) maxY = y
                if (z < minZ) minZ = z; if (z > maxZ) maxZ = z
            }

            val elapsed = System.currentTimeMillis() - startTime
            Log.d(TAG, "PyTorch Mobile completed: $gaussianCount Gaussians in ${elapsed}ms")
            Log.d(TAG, "  load=${loadTime}ms infer=${inferTime}ms total=${elapsed}ms")
            Log.d(TAG, "  Room: ${maxX-minX}m x ${maxY-minY}m x ${maxZ-minZ}m")

            progressCallback?.invoke(1.0f, "Done!")

            return@withContext StreamingResult(
                plyFile = plyFile,
                classicPlyFile = classicPlyFile,
                gaussianCount = gaussianCount,
                roomWidth = maxX - minX,
                roomHeight = maxY - minY,
                roomDepth = maxZ - minZ
            )

        } catch (e: Exception) {
            Log.e(TAG, "PyTorch Mobile inference failed", e)
            progressCallback?.invoke(0f, "Error: ${e.message}")
            return@withContext null
        }
    }

    private fun writePly(file: File, params: FloatArray, gaussianCount: Int) {
        val header = buildString {
            append("ply\nformat binary_little_endian 1.0\n")
            append("element vertex $gaussianCount\n")
            append("property float x\nproperty float y\nproperty float z\n")
            append("property float nx\nproperty float ny\nproperty float nz\n")
            for (i in 0 until 3) append("property float f_dc_$i\n")
            for (i in 0 until 45) append("property float f_rest_$i\n")
            append("property float opacity\n")
            append("property float scale_0\nproperty float scale_1\nproperty float scale_2\n")
            append("property float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\n")
            append("end_header\n")
        }

        val zeroSH = ByteArray(45 * 4)
        val lutScale = (LOGIT_LUT_SIZE - 1).toFloat()

        FileOutputStream(file).use { fos ->
            val channel = fos.channel
            channel.write(ByteBuffer.wrap(header.toByteArray()))

            val batchSize = 512
            val buf = ByteBuffer.allocateDirect(BYTES_PER_VERTEX * batchSize).order(ByteOrder.LITTLE_ENDIAN)

            var processed = 0
            while (processed < gaussianCount) {
                val count = minOf(batchSize, gaussianCount - processed)
                buf.clear()

                for (j in 0 until count) {
                    val off = (processed + j) * PARAMS_PER_GAUSSIAN
                    // pos(3) + scale(3) + rot(4) + opacity(1) + color(3) from model output
                    buf.putFloat(params[off + 0])       // x
                    buf.putFloat(-params[off + 1])      // -y
                    buf.putFloat(-params[off + 2])      // -z
                    buf.putFloat(0f); buf.putFloat(0f); buf.putFloat(0f) // normals

                    // Color -> SH DC (indices 11,12,13 in output)
                    val r = params[off + 11].coerceIn(0f, 1f)
                    val g = params[off + 12].coerceIn(0f, 1f)
                    val b = params[off + 13].coerceIn(0f, 1f)
                    buf.putFloat((r - 0.5f) / SH_C0)
                    buf.putFloat((g - 0.5f) / SH_C0)
                    buf.putFloat((b - 0.5f) / SH_C0)

                    buf.put(zeroSH) // 45 zero SH

                    // Opacity (index 10) -> logit
                    val op = params[off + 10].coerceIn(0f, 1f)
                    val lutIdx = (op * lutScale).toInt().coerceIn(0, LOGIT_LUT_SIZE - 1)
                    buf.putFloat(LOGIT_LUT[lutIdx])

                    // Scale (indices 3,4,5) -> log
                    for (s in 3..5) {
                        buf.putFloat(ln(max(params[off + s], 0.001f)))
                    }

                    // Rotation (indices 6,7,8,9) -> normalize
                    val qw = params[off + 6]; val qx = params[off + 7]
                    val qy = params[off + 8]; val qz = params[off + 9]
                    val mag = sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
                    val inv = if (mag > 1e-8f) 1f / mag else 1f
                    buf.putFloat(qw * inv); buf.putFloat(qx * inv)
                    buf.putFloat(qy * inv); buf.putFloat(qz * inv)
                }

                buf.flip()
                buf.limit(count * BYTES_PER_VERTEX)
                while (buf.hasRemaining()) channel.write(buf)

                processed += count
            }
        }
    }

    fun release() {
        module?.destroy()
        module = null
        isInitialized = false
        Log.d(TAG, "TorchMobileSharp released")
    }
}
