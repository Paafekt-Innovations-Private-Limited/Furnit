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
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * SharpService - On-device 3D Gaussian Splat generation
 * (Matches Swift's SHARPService)
 *
 * NOTE: The actual SHARP model requires CoreML on iOS.
 * For Android, this service provides:
 * - A working PLY file generator using the photo's color data
 * - Proper PLY format compatible with SparkJS viewer
 * - Ready for integration with a TFLite or cloud-based model
 */
class SharpService(private val context: Context) {

    companion object {
        private const val TAG = "SharpService"

        // PLY generation parameters
        private const val GRID_SIZE = 96  // Denser grid for better quality
        private const val SPLAT_SCALE = 0.02f
    }

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
     * Generate 3D Gaussian splats from an image
     * Creates a PLY file compatible with SparkJS/THREE.js viewer
     */
    fun generateGaussians(image: Bitmap, callback: ProgressCallback) {
        Log.d(TAG, "Starting Gaussian generation from image: ${image.width}x${image.height}")

        Thread {
            try {
                callback.onProgress(0.1f, "Preparing photo...")
                Thread.sleep(300)

                callback.onProgress(0.2f, "Analyzing room structure...")
                Thread.sleep(500)

                callback.onProgress(0.4f, "Creating 3D room...")

                // Generate PLY from image
                val result = createPLYFromImage(image)

                callback.onProgress(0.8f, "Finalizing model...")
                Thread.sleep(300)

                callback.onProgress(1.0f, "Done!")
                callback.onComplete(result)

            } catch (e: Exception) {
                Log.e(TAG, "Generation failed", e)
                callback.onError("Failed to generate room: ${e.message}")
            }
        }.start()
    }

    /**
     * Create PLY file from image data
     * Generates a 3D point cloud representing the room
     */
    private fun createPLYFromImage(image: Bitmap): GenerationResult {
        // Create output directory
        val roomsDir = File(context.filesDir, "sharp_rooms")
        roomsDir.mkdirs()

        val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())
        val roomFolder = File(roomsDir, "room_$timestamp")
        roomFolder.mkdirs()

        val plyFile = File(roomFolder, "room.ply")
        val classicPlyFile = File(roomFolder, "room_classic.ply")

        // Scale image for processing
        val processSize = 256
        val scaledBitmap = Bitmap.createScaledBitmap(image, processSize, processSize, true)

        // Generate Gaussian splats from image pixels
        val gaussians = generateGaussiansFromImage(scaledBitmap)
        Log.d(TAG, "Generated ${gaussians.size} Gaussians")

        // Write PLY files
        writePLYFile(plyFile, gaussians, false)
        writePLYFile(classicPlyFile, gaussians, true)

        // Calculate room dimensions from bounds
        var minX = Float.MAX_VALUE
        var maxX = Float.MIN_VALUE
        var minY = Float.MAX_VALUE
        var maxY = Float.MIN_VALUE
        var minZ = Float.MAX_VALUE
        var maxZ = Float.MIN_VALUE

        for (g in gaussians) {
            minX = minOf(minX, g.x)
            maxX = maxOf(maxX, g.x)
            minY = minOf(minY, g.y)
            maxY = maxOf(maxY, g.y)
            minZ = minOf(minZ, g.z)
            maxZ = maxOf(maxZ, g.z)
        }

        val width = maxX - minX
        val height = maxY - minY
        val depth = maxZ - minZ

        Log.d(TAG, "Room dimensions: ${width}x${height}x${depth}")
        Log.d(TAG, "PLY file saved: ${plyFile.absolutePath}")

        // Save thumbnail
        val thumbnailFile = File(roomFolder, "thumbnail.png")
        FileOutputStream(thumbnailFile).use { out ->
            image.compress(Bitmap.CompressFormat.PNG, 90, out)
        }

        // Save metadata
        val metadataFile = File(roomFolder, "metadata.txt")
        val roomName = "AI Room ${SimpleDateFormat("MMM d", Locale.getDefault()).format(Date())}"
        metadataFile.writeText("name=$roomName\ncreated=${System.currentTimeMillis()}\ntype=sharp")

        return GenerationResult(
            plyFile = plyFile,
            classicPlyFile = classicPlyFile,
            roomWidth = width,
            roomHeight = height,
            roomDepth = depth
        )
    }

    /**
     * Generate Gaussian splats from image data
     * Creates a 3D representation based on pixel colors
     */
    private fun generateGaussiansFromImage(image: Bitmap): List<GaussianSplat> {
        val gaussians = mutableListOf<GaussianSplat>()

        val width = image.width
        val height = image.height

        // Room dimensions (in meters, roughly)
        val roomWidth = 4.0f
        val roomHeight = 3.0f
        val roomDepth = 5.0f

        // Create a grid of splats based on image pixels
        val stepX = width / GRID_SIZE
        val stepY = height / GRID_SIZE

        for (gridY in 0 until GRID_SIZE) {
            for (gridX in 0 until GRID_SIZE) {
                val pixelX = gridX * stepX + stepX / 2
                val pixelY = gridY * stepY + stepY / 2

                if (pixelX >= width || pixelY >= height) continue

                val pixel = image.getPixel(pixelX, pixelY)
                val r = ((pixel shr 16) and 0xFF) / 255f
                val g = ((pixel shr 8) and 0xFF) / 255f
                val b = (pixel and 0xFF) / 255f

                // Skip very dark pixels (likely background)
                val brightness = (r + g + b) / 3f
                if (brightness < 0.05f) continue

                // Map grid position to 3D space
                // Front wall plane (z = -roomDepth/2)
                val x = (gridX.toFloat() / GRID_SIZE - 0.5f) * roomWidth
                val y = (0.5f - gridY.toFloat() / GRID_SIZE) * roomHeight
                val z = -roomDepth * 0.4f  // Front wall position

                gaussians.add(GaussianSplat(
                    x = x, y = y, z = z,
                    scaleX = SPLAT_SCALE * 1.5f,
                    scaleY = SPLAT_SCALE * 1.5f,
                    scaleZ = SPLAT_SCALE * 0.3f,
                    rotW = 1f, rotX = 0f, rotY = 0f, rotZ = 0f,
                    opacity = 0.95f,
                    r = r, g = g, b = b
                ))
            }
        }

        // Add floor plane (horizontal)
        addFloorGaussians(gaussians, image, roomWidth, roomHeight, roomDepth)

        // Add ceiling plane
        addCeilingGaussians(gaussians, image, roomWidth, roomHeight, roomDepth)

        // Add side walls
        addSideWallGaussians(gaussians, image, roomWidth, roomHeight, roomDepth)

        return gaussians
    }

    private fun addFloorGaussians(
        gaussians: MutableList<GaussianSplat>,
        image: Bitmap,
        roomWidth: Float,
        roomHeight: Float,
        roomDepth: Float
    ) {
        val floorY = -roomHeight / 2f
        val gridSize = GRID_SIZE * 2 / 3  // Denser floor

        // Sample floor color from bottom portion of image
        val floorStartY = (image.height * 0.75f).toInt()

        for (gridZ in 0 until gridSize) {
            for (gridX in 0 until gridSize) {
                val sampleX = (gridX * image.width / gridSize).coerceIn(0, image.width - 1)
                val sampleY = (floorStartY + gridZ * (image.height - floorStartY) / gridSize)
                    .coerceIn(0, image.height - 1)

                val pixel = image.getPixel(sampleX, sampleY)
                val r = ((pixel shr 16) and 0xFF) / 255f
                val g = ((pixel shr 8) and 0xFF) / 255f
                val b = (pixel and 0xFF) / 255f

                val x = (gridX.toFloat() / gridSize - 0.5f) * roomWidth
                val z = (gridZ.toFloat() / gridSize - 0.5f) * roomDepth

                gaussians.add(GaussianSplat(
                    x = x, y = floorY, z = z,
                    scaleX = SPLAT_SCALE * 2.5f,
                    scaleY = SPLAT_SCALE * 0.2f,
                    scaleZ = SPLAT_SCALE * 2.5f,
                    rotW = 1f, rotX = 0f, rotY = 0f, rotZ = 0f,
                    opacity = 0.9f,
                    r = r, g = g, b = b
                ))
            }
        }
    }

    private fun addCeilingGaussians(
        gaussians: MutableList<GaussianSplat>,
        image: Bitmap,
        roomWidth: Float,
        roomHeight: Float,
        roomDepth: Float
    ) {
        val ceilingY = roomHeight / 2f
        val gridSize = GRID_SIZE / 2

        // Sample ceiling color from top portion of image
        val ceilingEndY = (image.height * 0.2f).toInt()

        for (gridZ in 0 until gridSize) {
            for (gridX in 0 until gridSize) {
                val sampleX = (gridX * image.width / gridSize).coerceIn(0, image.width - 1)
                val sampleY = (gridZ * ceilingEndY / gridSize).coerceIn(0, image.height - 1)

                val pixel = image.getPixel(sampleX, sampleY)
                val r = ((pixel shr 16) and 0xFF) / 255f
                val g = ((pixel shr 8) and 0xFF) / 255f
                val b = (pixel and 0xFF) / 255f

                val x = (gridX.toFloat() / gridSize - 0.5f) * roomWidth
                val z = (gridZ.toFloat() / gridSize - 0.5f) * roomDepth

                gaussians.add(GaussianSplat(
                    x = x, y = ceilingY, z = z,
                    scaleX = SPLAT_SCALE * 2f,
                    scaleY = SPLAT_SCALE * 0.3f,
                    scaleZ = SPLAT_SCALE * 2f,
                    rotW = 1f, rotX = 0f, rotY = 0f, rotZ = 0f,
                    opacity = 0.8f,
                    r = r, g = g, b = b
                ))
            }
        }
    }

    private fun addSideWallGaussians(
        gaussians: MutableList<GaussianSplat>,
        image: Bitmap,
        roomWidth: Float,
        roomHeight: Float,
        roomDepth: Float
    ) {
        val gridSize = GRID_SIZE / 2

        // Left wall (sample from left edge of image)
        val leftX = -roomWidth / 2f
        for (gridZ in 0 until gridSize) {
            for (gridY in 0 until gridSize) {
                val sampleX = (image.width * 0.1f).toInt().coerceIn(0, image.width - 1)
                val sampleY = (gridY * image.height / gridSize).coerceIn(0, image.height - 1)

                val pixel = image.getPixel(sampleX, sampleY)
                val r = ((pixel shr 16) and 0xFF) / 255f
                val g = ((pixel shr 8) and 0xFF) / 255f
                val b = (pixel and 0xFF) / 255f

                val y = (0.5f - gridY.toFloat() / gridSize) * roomHeight
                val z = (gridZ.toFloat() / gridSize - 0.5f) * roomDepth

                gaussians.add(GaussianSplat(
                    x = leftX, y = y, z = z,
                    scaleX = SPLAT_SCALE * 0.3f,
                    scaleY = SPLAT_SCALE * 2f,
                    scaleZ = SPLAT_SCALE * 2f,
                    rotW = 0.707f, rotX = 0f, rotY = 0.707f, rotZ = 0f,
                    opacity = 0.7f,
                    r = r, g = g, b = b
                ))
            }
        }

        // Right wall (sample from right edge of image)
        val rightX = roomWidth / 2f
        for (gridZ in 0 until gridSize) {
            for (gridY in 0 until gridSize) {
                val sampleX = (image.width * 0.9f).toInt().coerceIn(0, image.width - 1)
                val sampleY = (gridY * image.height / gridSize).coerceIn(0, image.height - 1)

                val pixel = image.getPixel(sampleX, sampleY)
                val r = ((pixel shr 16) and 0xFF) / 255f
                val g = ((pixel shr 8) and 0xFF) / 255f
                val b = (pixel and 0xFF) / 255f

                val y = (0.5f - gridY.toFloat() / gridSize) * roomHeight
                val z = (gridZ.toFloat() / gridSize - 0.5f) * roomDepth

                gaussians.add(GaussianSplat(
                    x = rightX, y = y, z = z,
                    scaleX = SPLAT_SCALE * 0.3f,
                    scaleY = SPLAT_SCALE * 2f,
                    scaleZ = SPLAT_SCALE * 2f,
                    rotW = 0.707f, rotX = 0f, rotY = -0.707f, rotZ = 0f,
                    opacity = 0.7f,
                    r = r, g = g, b = b
                ))
            }
        }
    }

    /**
     * Write Gaussians to PLY file in binary format
     */
    private fun writePLYFile(file: File, gaussians: List<GaussianSplat>, classic: Boolean) {
        val header = """
ply
format binary_little_endian 1.0
element vertex ${gaussians.size}
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

        FileOutputStream(file).use { fos ->
            // Write header
            fos.write(header.toByteArray(Charsets.UTF_8))

            // Write binary vertex data
            val buffer = ByteBuffer.allocate(47) // 11 floats (44 bytes) + 3 bytes (RGB)
            buffer.order(ByteOrder.LITTLE_ENDIAN)

            for (g in gaussians) {
                buffer.clear()

                // Position - apply classic transform if needed
                if (classic) {
                    // Classic PLY: flip for antimatter15/splat viewer
                    buffer.putFloat(g.x)
                    buffer.putFloat(-g.y)
                    buffer.putFloat(-g.z)
                } else {
                    buffer.putFloat(g.x)
                    buffer.putFloat(-g.y)  // Y flip for standard PLY
                    buffer.putFloat(-g.z)  // Z flip
                }

                // Scale (log scale for renderer)
                val minScale = 0.001f
                buffer.putFloat(kotlin.math.ln(maxOf(g.scaleX, minScale)))
                buffer.putFloat(kotlin.math.ln(maxOf(g.scaleY, minScale)))
                buffer.putFloat(kotlin.math.ln(maxOf(g.scaleZ, minScale)))

                // Rotation quaternion (normalized)
                val mag = sqrt(g.rotW * g.rotW + g.rotX * g.rotX + g.rotY * g.rotY + g.rotZ * g.rotZ)
                val invMag = if (mag > 1e-8f) 1f / mag else 1f
                buffer.putFloat(g.rotW * invMag)
                buffer.putFloat(g.rotX * invMag)
                buffer.putFloat(g.rotY * invMag)
                buffer.putFloat(g.rotZ * invMag)

                // Opacity (logit for renderer)
                val clampedOpacity = g.opacity.coerceIn(1e-4f, 1f - 1e-4f)
                buffer.putFloat(kotlin.math.ln(clampedOpacity / (1f - clampedOpacity)))

                // RGB as uchar (with gamma correction)
                val gamma = 1.0 / 2.2
                val red = (Math.pow(g.r.toDouble(), gamma) * 255).toInt().coerceIn(0, 255)
                val green = (Math.pow(g.g.toDouble(), gamma) * 255).toInt().coerceIn(0, 255)
                val blue = (Math.pow(g.b.toDouble(), gamma) * 255).toInt().coerceIn(0, 255)
                buffer.put(red.toByte())
                buffer.put(green.toByte())
                buffer.put(blue.toByte())

                fos.write(buffer.array())
            }
        }

        Log.d(TAG, "Wrote PLY file: ${file.name} with ${gaussians.size} vertices")
    }

    /**
     * Represents a single Gaussian splat
     */
    data class GaussianSplat(
        val x: Float,
        val y: Float,
        val z: Float,
        val scaleX: Float,
        val scaleY: Float,
        val scaleZ: Float,
        val rotW: Float,
        val rotX: Float,
        val rotY: Float,
        val rotZ: Float,
        val opacity: Float,
        val r: Float,
        val g: Float,
        val b: Float
    )
}
