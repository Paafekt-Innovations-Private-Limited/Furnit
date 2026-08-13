package com.furnit.android.services

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.net.Uri
import com.furnit.android.R
import com.furnit.android.RoomDefaults
import com.furnit.android.models.RoomStructure
import com.furnit.android.utils.LogUtil
import com.furnit.android.utils.RoomDisplayName
import java.io.File
import java.io.FileOutputStream
import java.util.Date
import kotlin.math.max
import kotlin.math.min

/**
 * SinglePhotoRoomReconstructor - Creates 3D room from photo and boundaries
 * (Matches Swift's SinglePhotoRoomReconstructor - simplified version)
 *
 * Creates a GLB model file with:
 * - Floor, ceiling, and walls as textured planes
 * - Textures extracted from the photo based on boundary positions
 * - Or, for AI/photo preview, a pixel-correct flat full-photo surface
 */
class SinglePhotoRoomReconstructor(private val context: Context) {

    companion object {
        private const val TAG = "RoomReconstructor"

        // Default room dimensions in meters (Swift / Depth Anything fallback parity)
        const val DEFAULT_WIDTH = RoomDefaults.DEFAULT_WIDTH_M
        const val DEFAULT_DEPTH = RoomDefaults.DEFAULT_DEPTH_M
        const val DEFAULT_HEIGHT = RoomDefaults.DEFAULT_HEIGHT_M
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
        callback: ProgressCallback,
        flatPhotoMesh: Boolean = false,
        sourcePhotoUri: Uri? = null,
    ) {
        LogUtil.d(TAG, "Starting room reconstruction...")
        LogUtil.d(TAG, "  Boundaries: floor=${boundaries.floorY}, ceiling=${boundaries.ceilingY}")
        LogUtil.d(TAG, "  Boundaries: left=${boundaries.leftX}, right=${boundaries.rightX}")
        LogUtil.d(TAG, "  Boundaries: vp=(${boundaries.vanishingX}, ${boundaries.vanishingY})")

        Thread {
            try {
                callback.onProgress(0.1f, context.getString(R.string.ai_progress_getting_photo_ready))

                if (flatPhotoMesh) {
                    callback.onProgress(0.15f, context.getString(R.string.room_viewer_measuring_room))
                    val measured = DepthAnythingRoomMeasurer.measure(context, image, sourcePhotoUri)
                    if (measured.measured) {
                        dimensions.width = measured.width
                        dimensions.height = measured.height
                        dimensions.depth = measured.depth
                        LogUtil.i(
                            TAG,
                            "Depth Anything room dims W=${dimensions.width} H=${dimensions.height} D=${dimensions.depth}",
                        )
                    } else {
                        LogUtil.w(TAG, "Depth Anything measurement unavailable; using defaults")
                    }
                }

                callback.onProgress(0.3f, context.getString(R.string.ai_progress_understanding_picture))
                val frontWallTexture: Bitmap
                val floorTexture: Bitmap
                val ceilingTexture: Bitmap
                val leftWallTexture: Bitmap
                val rightWallTexture: Bitmap
                if (flatPhotoMesh) {
                    frontWallTexture = image
                    floorTexture = image
                    ceilingTexture = image
                    leftWallTexture = image
                    rightWallTexture = image
                } else {
                    frontWallTexture = extractFrontWallTexture(image, boundaries)
                    floorTexture = extractFloorTexture(image, boundaries)
                    ceilingTexture = extractCeilingTexture(image, boundaries)
                    leftWallTexture = extractLeftWallTexture(image, boundaries)
                    rightWallTexture = extractRightWallTexture(image, boundaries)
                }

                callback.onProgress(0.5f, context.getString(R.string.ai_progress_building_room))

                // Create GLB file
                callback.onProgress(0.7f, context.getString(R.string.ai_progress_preparing_preview))
                val glbFile = createRoomGLB(
                    dimensions,
                    frontWallTexture,
                    floorTexture,
                    ceilingTexture,
                    leftWallTexture,
                    rightWallTexture,
                    flatPhotoMesh,
                    sourcePhoto = if (flatPhotoMesh) image else null,
                )

                callback.onProgress(0.9f, context.getString(R.string.ai_progress_finishing_up))

                callback.onProgress(1.0f, context.getString(R.string.room_generation_ready))
                callback.onComplete(glbFile)

            } catch (e: Exception) {
                LogUtil.e(TAG, "Room reconstruction failed", e)
                callback.onError(context.getString(R.string.boundary_failed_create))
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
            LogUtil.w(TAG, "Failed to extract front wall texture", e)
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
            LogUtil.w(TAG, "Failed to extract floor texture", e)
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
            LogUtil.w(TAG, "Failed to extract ceiling texture", e)
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
            LogUtil.w(TAG, "Failed to extract left wall texture", e)
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
            LogUtil.w(TAG, "Failed to extract right wall texture", e)
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
        rightWall: Bitmap,
        flatPhotoMesh: Boolean = false,
        sourcePhoto: Bitmap? = null,
    ): File {
        // Create room in preview directory (NOT in rooms folder yet)
        // Room will be moved to rooms folder when user clicks Save in preview
        val previewDir = File(context.filesDir, "room_preview")
        // Clear any previous preview
        previewDir.deleteRecursively()
        previewDir.mkdirs()

        val timestamp = System.currentTimeMillis()
        val roomFolder = File(previewDir, "room_$timestamp")
        roomFolder.mkdirs()

        // Save front wall texture for thumbnail
        saveTexture(frontWall, File(roomFolder, "front_wall.png"))
        sourcePhoto?.let { saveTexture(it, File(roomFolder, "source_photo.jpg"), Bitmap.CompressFormat.JPEG) }

        // Generate GLB file using GlbGenerator
        val glbFile = File(roomFolder, "room.glb")
        val generator = GlbGenerator()
        val glbDimensions = GlbGenerator.RoomDimensions(
            width = dimensions.width,
            depth = dimensions.depth,
            height = dimensions.height
        )

        val success = if (flatPhotoMesh) {
            generator.generateFlatPhotoGlb(
                outputFile = glbFile,
                dimensions = glbDimensions,
                photoTexture = frontWall,
            )
        } else {
            generator.generateGlb(
                outputFile = glbFile,
                dimensions = glbDimensions,
                frontWallTexture = frontWall,
                floorTexture = floor,
                ceilingTexture = ceiling,
                leftWallTexture = leftWall,
                rightWallTexture = rightWall,
            )
        }

        if (!success) {
            LogUtil.e(TAG, "Failed to generate GLB, falling back to textures only")
        }

        // Save dimensions
        val dimensionsFile = File(roomFolder, "dimensions.txt")
        dimensionsFile.writeText("width=${dimensions.width}\ndepth=${dimensions.depth}\nheight=${dimensions.height}")

        // Save metadata for ModelManager
        val metadataFile = File(roomFolder, "metadata.txt")
        val createdAt = System.currentTimeMillis()
        val roomName = RoomDisplayName.myRoomWithTimestamp(context, Date(createdAt))
        metadataFile.writeText("name=$roomName\ncreated=$createdAt\nglb=room.glb")

        LogUtil.d(TAG, "Room created at: ${roomFolder.absolutePath}")
        return if (success) glbFile else File(roomFolder, "front_wall.png")
    }

    private fun saveTexture(
        bitmap: Bitmap,
        file: File,
        format: Bitmap.CompressFormat = Bitmap.CompressFormat.PNG,
    ) {
        try {
            FileOutputStream(file).use { out ->
                bitmap.compress(format, if (format == Bitmap.CompressFormat.JPEG) 92 else 90, out)
            }
            LogUtil.d(TAG, "Saved texture: ${file.name}")
        } catch (e: Exception) {
            LogUtil.e(TAG, "Failed to save texture: ${file.name}", e)
        }
    }
}
