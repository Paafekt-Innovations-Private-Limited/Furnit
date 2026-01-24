package com.furnit.android.services

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.util.Log
import com.furnit.android.models.RoomStructure
import java.io.File
import java.io.FileOutputStream
import kotlin.math.max
import kotlin.math.min

/**
 * SinglePhotoRoomReconstructor - Creates 3D room from photo and boundaries
 * (Matches Swift's SinglePhotoRoomReconstructor - simplified version)
 *
 * Creates a GLB model file with:
 * - Floor, ceiling, and walls as textured planes
 * - Textures extracted from the photo based on boundary positions
 */
class SinglePhotoRoomReconstructor(private val context: Context) {

    companion object {
        private const val TAG = "RoomReconstructor"

        // Default room dimensions in meters
        const val DEFAULT_WIDTH = 4.0f
        const val DEFAULT_DEPTH = 4.5f
        const val DEFAULT_HEIGHT = 2.8f
    }

    data class RoomDimensions(
        var width: Float = DEFAULT_WIDTH,
        var depth: Float = DEFAULT_DEPTH,
        var height: Float = DEFAULT_HEIGHT
    )

    interface ProgressCallback {
        fun onProgress(progress: Float, message: String)
        fun onComplete(glbFile: File?)
        fun onError(message: String)
    }

    /**
     * Process photo with adjusted boundaries and create 3D room
     */
    fun processPhotoWithBoundaries(
        image: Bitmap,
        boundaries: RoomStructure,
        dimensions: RoomDimensions = RoomDimensions(),
        callback: ProgressCallback
    ) {
        Log.d(TAG, "Starting room reconstruction...")
        Log.d(TAG, "  Boundaries: floor=${boundaries.floorY}, ceiling=${boundaries.ceilingY}")
        Log.d(TAG, "  Boundaries: left=${boundaries.leftX}, right=${boundaries.rightX}")
        Log.d(TAG, "  Boundaries: vp=(${boundaries.vanishingX}, ${boundaries.vanishingY})")

        Thread {
            try {
                callback.onProgress(0.1f, "Preparing image...")
                Thread.sleep(200)

                // Extract textures from photo
                callback.onProgress(0.3f, "Extracting textures...")
                val frontWallTexture = extractFrontWallTexture(image, boundaries)
                val floorTexture = extractFloorTexture(image, boundaries)
                val ceilingTexture = extractCeilingTexture(image, boundaries)
                val leftWallTexture = extractLeftWallTexture(image, boundaries)
                val rightWallTexture = extractRightWallTexture(image, boundaries)
                Thread.sleep(200)

                callback.onProgress(0.5f, "Building 3D model...")
                Thread.sleep(200)

                // Create GLB file
                callback.onProgress(0.7f, "Creating room file...")
                val glbFile = createRoomGLB(
                    dimensions,
                    frontWallTexture,
                    floorTexture,
                    ceilingTexture,
                    leftWallTexture,
                    rightWallTexture
                )
                Thread.sleep(200)

                callback.onProgress(0.9f, "Finalizing...")
                Thread.sleep(200)

                callback.onProgress(1.0f, "Room ready!")
                callback.onComplete(glbFile)

            } catch (e: Exception) {
                Log.e(TAG, "Room reconstruction failed", e)
                callback.onError("Failed to create room: ${e.message}")
            }
        }.start()
    }

    private fun extractFrontWallTexture(image: Bitmap, boundaries: RoomStructure): Bitmap {
        val left = (boundaries.leftX * image.width).toInt()
        val right = (boundaries.rightX * image.width).toInt()
        val top = (boundaries.ceilingY * image.height).toInt()
        val bottom = (boundaries.floorY * image.height).toInt()

        val width = max(1, right - left)
        val height = max(1, bottom - top)

        return try {
            Bitmap.createBitmap(image, left, top, width, height)
        } catch (e: Exception) {
            Log.w(TAG, "Failed to extract front wall texture", e)
            createSolidColorBitmap(Color.LTGRAY, 256, 256)
        }
    }

    private fun extractFloorTexture(image: Bitmap, boundaries: RoomStructure): Bitmap {
        val left = (boundaries.leftX * image.width).toInt()
        val right = (boundaries.rightX * image.width).toInt()
        val top = (boundaries.floorY * image.height).toInt()
        val bottom = image.height

        val width = max(1, right - left)
        val height = max(1, bottom - top)

        return try {
            Bitmap.createBitmap(image, left, top, width, height)
        } catch (e: Exception) {
            Log.w(TAG, "Failed to extract floor texture", e)
            createSolidColorBitmap(Color.parseColor("#D7CCC8"), 256, 256)
        }
    }

    private fun extractCeilingTexture(image: Bitmap, boundaries: RoomStructure): Bitmap {
        val left = (boundaries.leftX * image.width).toInt()
        val right = (boundaries.rightX * image.width).toInt()
        val top = 0
        val bottom = (boundaries.ceilingY * image.height).toInt()

        val width = max(1, right - left)
        val height = max(1, bottom - top)

        return try {
            if (height > 0) {
                Bitmap.createBitmap(image, left, top, width, height)
            } else {
                createSolidColorBitmap(Color.WHITE, 256, 256)
            }
        } catch (e: Exception) {
            Log.w(TAG, "Failed to extract ceiling texture", e)
            createSolidColorBitmap(Color.WHITE, 256, 256)
        }
    }

    private fun extractLeftWallTexture(image: Bitmap, boundaries: RoomStructure): Bitmap {
        val stripWidth = (image.width * 0.1f).toInt()
        val left = max(0, (boundaries.leftX * image.width).toInt() - stripWidth / 2)
        val top = (boundaries.ceilingY * image.height).toInt()
        val bottom = (boundaries.floorY * image.height).toInt()

        val width = min(stripWidth, image.width - left)
        val height = max(1, bottom - top)

        return try {
            Bitmap.createBitmap(image, left, top, width, height)
        } catch (e: Exception) {
            Log.w(TAG, "Failed to extract left wall texture", e)
            createSolidColorBitmap(Color.parseColor("#E0E0E0"), 256, 256)
        }
    }

    private fun extractRightWallTexture(image: Bitmap, boundaries: RoomStructure): Bitmap {
        val stripWidth = (image.width * 0.1f).toInt()
        val right = (boundaries.rightX * image.width).toInt()
        val left = max(0, right - stripWidth / 2)
        val top = (boundaries.ceilingY * image.height).toInt()
        val bottom = (boundaries.floorY * image.height).toInt()

        val width = min(stripWidth, image.width - left)
        val height = max(1, bottom - top)

        return try {
            Bitmap.createBitmap(image, left, top, width, height)
        } catch (e: Exception) {
            Log.w(TAG, "Failed to extract right wall texture", e)
            createSolidColorBitmap(Color.parseColor("#E0E0E0"), 256, 256)
        }
    }

    private fun createSolidColorBitmap(color: Int, width: Int, height: Int): Bitmap {
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bitmap)
        canvas.drawColor(color)
        return bitmap
    }

    /**
     * Creates a GLB file for the room with 5 textured planes
     */
    private fun createRoomGLB(
        dimensions: RoomDimensions,
        frontWall: Bitmap,
        floor: Bitmap,
        ceiling: Bitmap,
        leftWall: Bitmap,
        rightWall: Bitmap
    ): File {
        // Create room directory
        val roomDir = File(context.filesDir, "rooms")
        roomDir.mkdirs()

        val timestamp = System.currentTimeMillis()
        val roomFolder = File(roomDir, "room_$timestamp")
        roomFolder.mkdirs()

        // Save front wall texture for thumbnail
        saveTexture(frontWall, File(roomFolder, "front_wall.png"))

        // Generate GLB file using GlbGenerator
        val glbFile = File(roomFolder, "room.glb")
        val generator = GlbGenerator()
        val glbDimensions = GlbGenerator.RoomDimensions(
            width = dimensions.width,
            depth = dimensions.depth,
            height = dimensions.height
        )

        val success = generator.generateGlb(
            outputFile = glbFile,
            dimensions = glbDimensions,
            frontWallTexture = frontWall,
            floorTexture = floor,
            ceilingTexture = ceiling,
            leftWallTexture = leftWall,
            rightWallTexture = rightWall
        )

        if (!success) {
            Log.e(TAG, "Failed to generate GLB, falling back to textures only")
        }

        // Save dimensions
        val dimensionsFile = File(roomFolder, "dimensions.txt")
        dimensionsFile.writeText("width=${dimensions.width}\ndepth=${dimensions.depth}\nheight=${dimensions.height}")

        // Save metadata for ModelManager
        val metadataFile = File(roomFolder, "metadata.txt")
        val roomName = "My Room ${java.text.SimpleDateFormat("MMM d", java.util.Locale.getDefault()).format(java.util.Date())}"
        metadataFile.writeText("name=$roomName\ncreated=${System.currentTimeMillis()}\nglb=room.glb")

        Log.d(TAG, "Room created at: ${roomFolder.absolutePath}")
        return if (success) glbFile else File(roomFolder, "front_wall.png")
    }

    private fun saveTexture(bitmap: Bitmap, file: File) {
        try {
            FileOutputStream(file).use { out ->
                bitmap.compress(Bitmap.CompressFormat.PNG, 90, out)
            }
            Log.d(TAG, "Saved texture: ${file.name}")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to save texture: ${file.name}", e)
        }
    }
}
