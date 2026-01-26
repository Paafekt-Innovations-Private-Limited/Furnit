package com.furnit.android.services

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.sqrt

/**
 * SharpService - On-device 3D Gaussian Splat generation using NCNN.
 *
 * Singleton pattern matching iOS SHARPService.shared:
 * - Model loaded once and reused across views
 * - Lazy initialization on first use
 * - No fallbacks - NCNN only
 */
class SharpService private constructor(private val context: Context) {

    companion object {
        private const val TAG = "SharpService"
        private const val SH_C0 = 0.28209479177387814f
        private const val BYTES_PER_VERTEX = 62 * 4  // 62 floats per vertex

        @Volatile
        private var instance: SharpService? = null

        /**
         * Shared singleton instance (matches iOS SHARPService.shared)
         */
        fun getInstance(context: Context): SharpService {
            return instance ?: synchronized(this) {
                instance ?: SharpService(context.applicationContext).also {
                    instance = it
                    Log.d(TAG, "SharpService singleton created")
                }
            }
        }
    }

    // Try NCNN first, fallback to Split ONNX (memory-efficient), then regular ONNX
    private val ncnnSharp by lazy { NcnnSharp(context) }
    private val splitOnnxSharp by lazy { SplitOnnxSharp.getInstance(context) }
    private val onnxSharp by lazy { OnnxSharp.getInstance(context) }
    private var isInitialized = false
    private var useOnnx = false
    private var useSplitOnnx = false

    data class GenerationResult(
        val plyFile: File,
        val classicPlyFile: File,
        val roomWidth: Float,
        val roomHeight: Float,
        val roomDepth: Float
    )

    interface ProgressCallback {
        fun onProgress(progress: Float, message: String)
        fun onComplete(result: GenerationResult)
        fun onError(message: String)
    }

    /**
     * Check if SHARP model is available (NCNN, Split ONNX, or regular ONNX)
     */
    fun isModelReady(): Boolean = ncnnSharp.isModelReady() || splitOnnxSharp.isModelReady() || onnxSharp.isModelReady()

    /**
     * Initialize model - tries NCNN first, then Split ONNX (memory-efficient), then full ONNX
     */
    suspend fun initialize(): Boolean {
        if (isInitialized) return true

        // Try NCNN first
        if (ncnnSharp.ensureModelReady()) {
            try {
                ncnnSharp.init()
                isInitialized = true
                useOnnx = false
                useSplitOnnx = false
                Log.d(TAG, "NCNN initialized successfully")
                return true
            } catch (e: Exception) {
                Log.w(TAG, "NCNN init failed: ${e.message}, trying Split ONNX...")
            }
        }

        // Try Split ONNX (memory-efficient 4-part model - FP32 compatible)
        if (splitOnnxSharp.isModelReady()) {
            Log.d(TAG, "Split ONNX model ready - using memory-efficient 4-part inference")
            isInitialized = true
            useOnnx = false
            useSplitOnnx = true
            return true
        } else {
            Log.w(TAG, "Split ONNX not ready, missing: ${splitOnnxSharp.getMissingFiles()}")
        }

        // Fall back to regular ONNX with mmap (may cause OOM on some devices)
        if (onnxSharp.ensureModelDownloaded()) {
            try {
                if (onnxSharp.initialize()) {
                    isInitialized = true
                    useOnnx = true
                    useSplitOnnx = false
                    Log.d(TAG, "ONNX with mmap initialized successfully")
                    return true
                }
            } catch (e: Exception) {
                Log.e(TAG, "ONNX init failed: ${e.message}")
            }
        }

        Log.e(TAG, "No model backend available")
        return false
    }

    /**
     * Generate 3D Gaussian splats from an image
     */
    fun generateGaussians(image: Bitmap, callback: ProgressCallback) {
        Log.d(TAG, "Starting generation: ${image.width}x${image.height}")

        Thread {
            try {
                callback.onProgress(0.1f, "Preparing...")

                // Initialize if needed
                if (!isInitialized) {
                    callback.onProgress(0.15f, "Loading SHARP model...")
                    val initialized = kotlinx.coroutines.runBlocking { initialize() }
                    if (!initialized) {
                        callback.onError("SHARP model not available. Push model to device.")
                        return@Thread
                    }
                }

                if (useSplitOnnx) {
                    // Use Split ONNX backend (memory-efficient 4-part model)
                    callback.onProgress(0.2f, "Running SHARP (Split ONNX - memory efficient)...")
                    val result = kotlinx.coroutines.runBlocking {
                        splitOnnxSharp.inferStreaming(image) { progress, message ->
                            callback.onProgress(progress, message)
                        }
                    }

                    if (result == null) {
                        callback.onError("SHARP Split ONNX inference failed")
                        return@Thread
                    }

                    Log.d(TAG, "Generated ${result.gaussianCount} Gaussians (Split ONNX)")
                    Log.d(TAG, "Room: ${result.roomWidth}m x ${result.roomHeight}m x ${result.roomDepth}m")

                    // Save thumbnail and metadata
                    saveMetadata(result.plyFile.parentFile!!, image, "sharp_split_onnx")

                    callback.onProgress(1.0f, "Done!")
                    callback.onComplete(GenerationResult(
                        plyFile = result.plyFile,
                        classicPlyFile = result.classicPlyFile,
                        roomWidth = result.roomWidth,
                        roomHeight = result.roomHeight,
                        roomDepth = result.roomDepth
                    ))
                } else if (useOnnx) {
                    // Use ONNX backend (streaming)
                    callback.onProgress(0.2f, "Running SHARP (ONNX)...")
                    val result = kotlinx.coroutines.runBlocking {
                        onnxSharp.inferStreaming(image) { progress, message ->
                            callback.onProgress(progress, message)
                        }
                    }

                    if (result == null) {
                        callback.onError("SHARP ONNX inference failed")
                        return@Thread
                    }

                    Log.d(TAG, "Generated ${result.gaussianCount} Gaussians (ONNX)")
                    Log.d(TAG, "Room: ${result.roomWidth}m x ${result.roomHeight}m x ${result.roomDepth}m")

                    // Save thumbnail and metadata
                    saveMetadata(result.plyFile.parentFile!!, image, "sharp_onnx")

                    callback.onProgress(1.0f, "Done!")
                    callback.onComplete(GenerationResult(
                        plyFile = result.plyFile,
                        classicPlyFile = result.classicPlyFile,
                        roomWidth = result.roomWidth,
                        roomHeight = result.roomHeight,
                        roomDepth = result.roomDepth
                    ))
                } else {
                    // Use NCNN backend
                    callback.onProgress(0.2f, "Running SHARP (NCNN)...")

                    val gaussianResult = ncnnSharp.generateGaussians(image)

                    callback.onProgress(0.6f, "Writing PLY file...")

                    Log.d(TAG, "Generated ${gaussianResult.gaussianCount} Gaussians (NCNN)")
                    Log.d(TAG, "Room: ${gaussianResult.roomWidth}m x ${gaussianResult.roomHeight}m x ${gaussianResult.roomDepth}m")

                    // Create output directory
                    val roomsDir = File(context.filesDir, "sharp_rooms")
                    roomsDir.mkdirs()

                    val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
                    val roomFolder = File(roomsDir, "room_$timestamp")
                    roomFolder.mkdirs()

                    val plyFile = File(roomFolder, "room.ply")
                    val classicPlyFile = File(roomFolder, "room_classic.ply")

                    // Write PLY file from Gaussian parameters
                    writePlyFile(plyFile, gaussianResult)
                    plyFile.copyTo(classicPlyFile, overwrite = true)

                    // Save thumbnail and metadata
                    saveMetadata(roomFolder, image, "sharp_ncnn")

                    callback.onProgress(1.0f, "Done!")
                    callback.onComplete(GenerationResult(
                        plyFile = plyFile,
                        classicPlyFile = classicPlyFile,
                        roomWidth = gaussianResult.roomWidth,
                        roomHeight = gaussianResult.roomHeight,
                        roomDepth = gaussianResult.roomDepth
                    ))
                }

            } catch (e: Exception) {
                Log.e(TAG, "Generation failed", e)
                callback.onError("Failed: ${e.message}")
            }
        }.start()
    }

    private fun saveMetadata(roomFolder: File, image: Bitmap, modelType: String) {
        // Save thumbnail
        val thumbnailFile = File(roomFolder, "thumbnail.png")
        FileOutputStream(thumbnailFile).use { out ->
            image.compress(Bitmap.CompressFormat.PNG, 90, out)
        }

        // Save metadata
        val metadataFile = File(roomFolder, "metadata.txt")
        val dateFormat = SimpleDateFormat("MMM d", Locale.getDefault())
        val roomName = "AI Room ${dateFormat.format(Date())}"
        metadataFile.writeText("name=$roomName\ncreated=${System.currentTimeMillis()}\ntype=$modelType")
    }

    /**
     * Write Gaussian parameters to PLY file in standard 3DGS format.
     */
    private fun writePlyFile(file: File, result: NcnnSharp.GaussianResult) {
        val gaussianCount = result.gaussianCount
        val params = result.params

        // Build PLY header
        val header = buildString {
            append("ply\n")
            append("format binary_little_endian 1.0\n")
            append("element vertex $gaussianCount\n")
            append("property float x\n")
            append("property float y\n")
            append("property float z\n")
            append("property float nx\n")
            append("property float ny\n")
            append("property float nz\n")
            for (i in 0 until 3) append("property float f_dc_$i\n")
            for (i in 0 until 45) append("property float f_rest_$i\n")
            append("property float opacity\n")
            append("property float scale_0\n")
            append("property float scale_1\n")
            append("property float scale_2\n")
            append("property float rot_0\n")
            append("property float rot_1\n")
            append("property float rot_2\n")
            append("property float rot_3\n")
            append("end_header\n")
        }

        FileOutputStream(file).use { fos ->
            fos.write(header.toByteArray(Charsets.UTF_8))

            val vertexBuffer = ByteBuffer.allocate(BYTES_PER_VERTEX)
            vertexBuffer.order(ByteOrder.LITTLE_ENDIAN)

            for (i in 0 until gaussianCount) {
                vertexBuffer.clear()
                val offset = i * NcnnSharp.PARAMS_PER_GAUSSIAN

                // Position (flip Y and Z for coordinate system)
                val x = params[offset + 0]
                val y = -params[offset + 1]
                val z = -params[offset + 2]
                vertexBuffer.putFloat(x)
                vertexBuffer.putFloat(y)
                vertexBuffer.putFloat(z)

                // Normals (unused)
                vertexBuffer.putFloat(0f)
                vertexBuffer.putFloat(0f)
                vertexBuffer.putFloat(0f)

                // Colors -> SH DC coefficients
                val r = params[offset + 11].coerceIn(0f, 1f)
                val g = params[offset + 12].coerceIn(0f, 1f)
                val b = params[offset + 13].coerceIn(0f, 1f)
                vertexBuffer.putFloat((r - 0.5f) / SH_C0)
                vertexBuffer.putFloat((g - 0.5f) / SH_C0)
                vertexBuffer.putFloat((b - 0.5f) / SH_C0)

                // Higher order SH (45 zeros)
                for (j in 0 until 45) vertexBuffer.putFloat(0f)

                // Opacity -> logit transform
                val rawOpacity = params[offset + 10].coerceIn(1e-4f, 1f - 1e-4f)
                vertexBuffer.putFloat(ln(rawOpacity / (1f - rawOpacity)))

                // Scale -> log transform
                val scaleBoost = 1.3f
                val minScale = 0.001f
                vertexBuffer.putFloat(ln(max(params[offset + 3] * scaleBoost, minScale)))
                vertexBuffer.putFloat(ln(max(params[offset + 4] * scaleBoost, minScale)))
                vertexBuffer.putFloat(ln(max(params[offset + 5] * scaleBoost, minScale)))

                // Rotation quaternion (normalize)
                val rw = params[offset + 6]
                val rx = params[offset + 7]
                val ry = params[offset + 8]
                val rz = params[offset + 9]
                val mag = sqrt(rw * rw + rx * rx + ry * ry + rz * rz)
                val invMag = if (mag > 1e-8f) 1f / mag else 1f
                vertexBuffer.putFloat(rw * invMag)
                vertexBuffer.putFloat(rx * invMag)
                vertexBuffer.putFloat(ry * invMag)
                vertexBuffer.putFloat(rz * invMag)

                fos.write(vertexBuffer.array())
            }
        }

        Log.d(TAG, "Wrote PLY file: ${file.absolutePath} (${file.length()} bytes)")
    }

    /**
     * Release resources
     */
    fun release() {
        if (useOnnx) {
            onnxSharp.release()
        } else {
            ncnnSharp.release()
        }
        isInitialized = false
    }
}
