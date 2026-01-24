package com.furnit.android.services

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.text.SimpleDateFormat
import java.util.*
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.pow
import kotlin.math.sqrt

/**
 * SharpService - On-device 3D Gaussian Splat generation using NCNN
 *
 * Uses NcnnSharp for inference when native library is available,
 * falls back to depth-aware generation otherwise.
 *
 * Generates PLY files compatible with Gaussian splatting renderers.
 */
class SharpService(private val context: Context) {

    companion object {
        private const val TAG = "SharpService"

        // Parameters per Gaussian: pos(3) + scale(3) + rot(4) + opacity(1) + color(3)
        private const val PARAMS_PER_GAUSSIAN = 14
    }

    private val ncnnSharp = NcnnSharp(context)
    private var isInitialized = false

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
     * Initialize NCNN model (optional - will use fallback if not available)
     */
    fun initialize(): Boolean {
        isInitialized = ncnnSharp.init()
        Log.d(TAG, "NCNN initialized: $isInitialized (native: ${NcnnSharp.isNativeAvailable()})")
        return isInitialized
    }

    /**
     * Generate 3D Gaussian splats from an image
     */
    fun generateGaussians(image: Bitmap, callback: ProgressCallback) {
        Log.d(TAG, "Starting Gaussian generation from image: ${image.width}x${image.height}")

        Thread {
            try {
                callback.onProgress(0.1f, "Preparing photo...")

                // Initialize if not already done
                if (!isInitialized) {
                    initialize()
                }

                callback.onProgress(0.2f, "Analyzing room structure...")

                // Generate Gaussians using NCNN or fallback
                val gaussianResult = ncnnSharp.generateGaussians(image)

                callback.onProgress(0.5f, "Creating 3D room...")
                Log.d(TAG, "Generated ${gaussianResult.gaussianCount} Gaussians")

                // Create PLY files
                callback.onProgress(0.7f, "Saving model...")
                val result = writePLYFiles(image, gaussianResult)

                callback.onProgress(1.0f, "Done!")
                callback.onComplete(result)

            } catch (e: Exception) {
                Log.e(TAG, "Generation failed", e)
                callback.onError("Failed to generate room: ${e.message}")
            }
        }.start()
    }

    /**
     * Write Gaussian parameters to PLY files
     */
    private fun writePLYFiles(
        originalImage: Bitmap,
        gaussianResult: NcnnSharp.GaussianResult
    ): GenerationResult {
        // Create output directory
        val roomsDir = File(context.filesDir, "sharp_rooms")
        roomsDir.mkdirs()

        val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())
        val roomFolder = File(roomsDir, "room_$timestamp")
        roomFolder.mkdirs()

        val plyFile = File(roomFolder, "room.ply")
        val classicPlyFile = File(roomFolder, "room_classic.ply")

        val gaussianCount = gaussianResult.gaussianCount
        val params = gaussianResult.params

        // PLY header
        val header = """
ply
format binary_little_endian 1.0
element vertex $gaussianCount
property float x
property float y
property float z
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
property float opacity
property uchar red
property uchar green
property uchar blue
end_header
""".trimStart()

        // Write PLY files
        FileOutputStream(plyFile).use { fos ->
            fos.write(header.toByteArray(Charsets.UTF_8))

            val buffer = ByteBuffer.allocate(47)  // 11 floats (44 bytes) + 3 bytes RGB
            buffer.order(ByteOrder.LITTLE_ENDIAN)

            for (i in 0 until gaussianCount) {
                val offset = i * PARAMS_PER_GAUSSIAN
                buffer.clear()

                // Position (with Y/Z flip for standard coordinate system)
                val x = params[offset + 0]
                val y = -params[offset + 1]  // Flip Y
                val z = -params[offset + 2]  // Flip Z

                buffer.putFloat(x)
                buffer.putFloat(y)
                buffer.putFloat(z)

                // Scale (convert to log for renderer)
                val minScale = 0.001f
                val scaleBoost = 1.3f
                buffer.putFloat(ln(max(params[offset + 3] * scaleBoost, minScale)))
                buffer.putFloat(ln(max(params[offset + 4] * scaleBoost, minScale)))
                buffer.putFloat(ln(max(params[offset + 5] * scaleBoost, minScale)))

                // Rotation quaternion (normalize)
                val rw = params[offset + 6]
                val rx = params[offset + 7]
                val ry = params[offset + 8]
                val rz = params[offset + 9]
                val mag = sqrt(rw * rw + rx * rx + ry * ry + rz * rz)
                val invMag = if (mag > 1e-8f) 1f / mag else 1f
                buffer.putFloat(rw * invMag)
                buffer.putFloat(rx * invMag)
                buffer.putFloat(ry * invMag)
                buffer.putFloat(rz * invMag)

                // Opacity (convert to logit)
                val rawOpacity = params[offset + 10]
                val clampedOpacity = rawOpacity.coerceIn(1e-4f, 1f - 1e-4f)
                buffer.putFloat(ln(clampedOpacity / (1f - clampedOpacity)))

                // Color (with gamma correction)
                val gamma = 1.0 / 2.2
                val brightness = 1.1f
                val r = params[offset + 11] * brightness
                val g = params[offset + 12] * brightness
                val b = params[offset + 13] * brightness

                buffer.put((r.coerceIn(0f, 1f).toDouble().pow(gamma) * 255).toInt().coerceIn(0, 255).toByte())
                buffer.put((g.coerceIn(0f, 1f).toDouble().pow(gamma) * 255).toInt().coerceIn(0, 255).toByte())
                buffer.put((b.coerceIn(0f, 1f).toDouble().pow(gamma) * 255).toInt().coerceIn(0, 255).toByte())

                fos.write(buffer.array())
            }
        }

        // Write classic PLY (different coordinate transform for antimatter15 viewer)
        FileOutputStream(classicPlyFile).use { fos ->
            fos.write(header.toByteArray(Charsets.UTF_8))

            val buffer = ByteBuffer.allocate(47)
            buffer.order(ByteOrder.LITTLE_ENDIAN)

            for (i in 0 until gaussianCount) {
                val offset = i * PARAMS_PER_GAUSSIAN
                buffer.clear()

                // Classic transform: (x, -y, -z) → (x, y, z)
                val x = params[offset + 0]
                val y = params[offset + 1]
                val z = params[offset + 2]

                buffer.putFloat(x)
                buffer.putFloat(-y)
                buffer.putFloat(-z)

                // Scale
                val minScale = 0.001f
                val scaleBoost = 1.3f
                buffer.putFloat(ln(max(params[offset + 3] * scaleBoost, minScale)))
                buffer.putFloat(ln(max(params[offset + 4] * scaleBoost, minScale)))
                buffer.putFloat(ln(max(params[offset + 5] * scaleBoost, minScale)))

                // Rotation
                val rw = params[offset + 6]
                val rx = params[offset + 7]
                val ry = params[offset + 8]
                val rz = params[offset + 9]
                val mag = sqrt(rw * rw + rx * rx + ry * ry + rz * rz)
                val invMag = if (mag > 1e-8f) 1f / mag else 1f
                buffer.putFloat(rw * invMag)
                buffer.putFloat(rx * invMag)
                buffer.putFloat(ry * invMag)
                buffer.putFloat(rz * invMag)

                // Opacity
                val rawOpacity = params[offset + 10]
                val clampedOpacity = rawOpacity.coerceIn(1e-4f, 1f - 1e-4f)
                buffer.putFloat(ln(clampedOpacity / (1f - clampedOpacity)))

                // Color
                val gamma = 1.0 / 2.2
                val brightness = 1.1f
                val r = params[offset + 11] * brightness
                val g = params[offset + 12] * brightness
                val b = params[offset + 13] * brightness

                buffer.put((r.coerceIn(0f, 1f).toDouble().pow(gamma) * 255).toInt().coerceIn(0, 255).toByte())
                buffer.put((g.coerceIn(0f, 1f).toDouble().pow(gamma) * 255).toInt().coerceIn(0, 255).toByte())
                buffer.put((b.coerceIn(0f, 1f).toDouble().pow(gamma) * 255).toInt().coerceIn(0, 255).toByte())

                fos.write(buffer.array())
            }
        }

        Log.d(TAG, "PLY files saved: ${plyFile.name} (${plyFile.length()} bytes)")

        // Save thumbnail
        val thumbnailFile = File(roomFolder, "thumbnail.png")
        FileOutputStream(thumbnailFile).use { out ->
            originalImage.compress(Bitmap.CompressFormat.PNG, 90, out)
        }

        // Save metadata
        val metadataFile = File(roomFolder, "metadata.txt")
        val roomName = "AI Room ${SimpleDateFormat("MMM d", Locale.getDefault()).format(Date())}"
        metadataFile.writeText("name=$roomName\ncreated=${System.currentTimeMillis()}\ntype=sharp")

        return GenerationResult(
            plyFile = plyFile,
            classicPlyFile = classicPlyFile,
            roomWidth = gaussianResult.roomWidth,
            roomHeight = gaussianResult.roomHeight,
            roomDepth = gaussianResult.roomDepth
        )
    }

    /**
     * Release resources
     */
    fun release() {
        ncnnSharp.release()
        isInitialized = false
    }
}
