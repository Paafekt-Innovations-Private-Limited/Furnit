package com.furnit.android.services

import android.content.Context
import android.graphics.Bitmap
import android.net.Uri
import com.furnit.android.models.RoomStructure
import com.furnit.android.utils.LogUtil
import com.furnit.android.utils.RoomDisplayName
import com.furnit.android.utils.RoomFolderMetadata
import java.io.File
import java.util.Date
import java.util.concurrent.atomic.AtomicInteger

/**
 * Generic Android photo-to-room generation path.
 *
 * Android now keeps the Swift-parity room-generation asset layout under
 * assets/room_generation. Until GeoCalib has an Android export and the full
 * metric reconstruction path is wired, this service keeps room creation on the
 * existing textured-GLB path and avoids the removed licensed model stack.
 */
class PhotoRoomGenerationService private constructor(private val context: Context) {

    data class GenerationResult(
        val glbFile: File,
        val roomFolder: File,
        val roomWidth: Float,
        val roomHeight: Float,
        val roomDepth: Float,
        val photoOrientation: String,
        val photoWideAngle: Boolean,
        val previewOnly: Boolean,
    )

    interface ProgressCallback {
        fun onProgress(progress: Float, message: String)
        fun onComplete(result: GenerationResult)
        fun onError(message: String)
    }

    class GenerationHandle internal constructor(
        private val service: PhotoRoomGenerationService,
        private val token: GenerationToken,
    ) {
        fun cancel() {
            service.cancel(token)
        }
    }

    internal data class GenerationToken(
        val id: Int,
        @Volatile var cancelled: Boolean = false,
    )

    private val nextId = AtomicInteger(0)
    private val lock = Any()
    private var activeToken: GenerationToken? = null

    fun startGenerationInBackground(
        bitmap: Bitmap,
        callback: ProgressCallback,
        viewerPhotoOrientation: String,
        viewerPhotoWideAngle: Boolean,
        sourcePhotoUri: Uri? = null,
    ): GenerationHandle {
        val token = GenerationToken(nextId.incrementAndGet())
        synchronized(lock) {
            activeToken?.cancelled = true
            activeToken = token
        }

        RoomGenerationAssets.logAvailability(context)

        val dimensions = SinglePhotoRoomReconstructor.RoomDimensions()
        val reconstructor = SinglePhotoRoomReconstructor(context)
        val boundaries = defaultBoundariesFor(bitmap)

        callback.onProgress(0.05f, "Getting your photo ready...")
        reconstructor.processPhotoWithBoundaries(
            bitmap,
            boundaries,
            dimensions,
            object : SinglePhotoRoomReconstructor.ProgressCallback {
                override fun onProgress(progress: Float, message: String) {
                    if (!isActive(token)) return
                    callback.onProgress(progress.coerceIn(0f, 0.95f), message)
                }

                override fun onComplete(glbFile: File?) {
                    if (!isActive(token)) {
                        glbFile?.parentFile?.deleteRecursively()
                        return
                    }
                    if (glbFile == null || !glbFile.exists() || !glbFile.name.endsWith(".glb", ignoreCase = true)) {
                        clearIfActive(token)
                        callback.onError("Failed to create room")
                        return
                    }

                    val result = writePreviewMetadata(
                        glbFile = glbFile,
                        dimensions = dimensions,
                        photoOrientation = normalizedOrientation(viewerPhotoOrientation),
                        photoWideAngle = viewerPhotoWideAngle,
                        sourcePhotoUri = sourcePhotoUri,
                    )
                    clearIfActive(token)
                    callback.onProgress(1f, "Your room is ready!")
                    callback.onComplete(result)
                }

                override fun onError(message: String) {
                    if (!isActive(token)) return
                    clearIfActive(token)
                    callback.onError(message)
                }
            },
        )

        return GenerationHandle(this, token)
    }

    fun cancelGeneration() {
        synchronized(lock) {
            activeToken?.cancelled = true
            activeToken = null
        }
    }

    fun promoteToLibrary(result: GenerationResult): GenerationResult {
        if (!result.previewOnly && result.roomFolder.parentFile?.name == ROOMS_DIR) return result

        val roomsDir = File(context.filesDir, ROOMS_DIR).apply { mkdirs() }
        val destination = uniqueRoomFolder(roomsDir, result.roomFolder.name)
        result.roomFolder.copyRecursively(destination, overwrite = false)

        val glbFile = File(destination, result.glbFile.name)
        writeMetadata(
            folder = destination,
            name = RoomDisplayName.myRoomWithTimestamp(),
            type = "photo",
            createdAt = System.currentTimeMillis(),
            roomWidth = result.roomWidth,
            roomHeight = result.roomHeight,
            roomDepth = result.roomDepth,
            photoOrientation = result.photoOrientation,
            photoWideAngle = result.photoWideAngle,
            previewOnly = false,
        )

        if (result.roomFolder.parentFile?.name == PREVIEW_DIR) {
            result.roomFolder.parentFile?.deleteRecursively()
        }

        return result.copy(
            glbFile = glbFile,
            roomFolder = destination,
            previewOnly = false,
        )
    }

    private fun cancel(token: GenerationToken) {
        synchronized(lock) {
            if (activeToken?.id == token.id) {
                token.cancelled = true
                activeToken = null
            } else {
                token.cancelled = true
            }
        }
    }

    private fun isActive(token: GenerationToken): Boolean {
        return synchronized(lock) {
            activeToken?.id == token.id && !token.cancelled
        }
    }

    private fun clearIfActive(token: GenerationToken) {
        synchronized(lock) {
            if (activeToken?.id == token.id) activeToken = null
        }
    }

    private fun defaultBoundariesFor(bitmap: Bitmap): RoomStructure {
        val landscape = bitmap.width > bitmap.height
        return RoomStructure(
            floorY = if (landscape) 0.80f else 0.84f,
            ceilingY = if (landscape) 0.18f else 0.16f,
            leftX = if (landscape) 0.15f else 0.12f,
            rightX = if (landscape) 0.85f else 0.88f,
            vanishingX = 0.5f,
            vanishingY = if (landscape) 0.44f else 0.45f,
        )
    }

    private fun writePreviewMetadata(
        glbFile: File,
        dimensions: SinglePhotoRoomReconstructor.RoomDimensions,
        photoOrientation: String,
        photoWideAngle: Boolean,
        sourcePhotoUri: Uri?,
    ): GenerationResult {
        val folder = glbFile.parentFile ?: error("Room GLB has no folder")
        val createdAt = System.currentTimeMillis()
        val roomName = RoomDisplayName.myRoomWithTimestamp(Date(createdAt))
        writeMetadata(
            folder = folder,
            name = roomName,
            type = "photo",
            createdAt = createdAt,
            roomWidth = dimensions.width,
            roomHeight = dimensions.height,
            roomDepth = dimensions.depth,
            photoOrientation = photoOrientation,
            photoWideAngle = photoWideAngle,
            previewOnly = true,
        )
        if (sourcePhotoUri != null) {
            runCatching {
                File(folder, "source_photo_uri.txt").writeText(sourcePhotoUri.toString())
            }.onFailure { LogUtil.w(TAG, "Could not write source photo uri", it) }
        }
        return GenerationResult(
            glbFile = glbFile,
            roomFolder = folder,
            roomWidth = dimensions.width,
            roomHeight = dimensions.height,
            roomDepth = dimensions.depth,
            photoOrientation = photoOrientation,
            photoWideAngle = photoWideAngle,
            previewOnly = true,
        )
    }

    private fun writeMetadata(
        folder: File,
        name: String,
        type: String,
        createdAt: Long,
        roomWidth: Float,
        roomHeight: Float,
        roomDepth: Float,
        photoOrientation: String,
        photoWideAngle: Boolean,
        previewOnly: Boolean,
    ) {
        val normalizedOrientation = normalizedOrientation(photoOrientation)
        File(folder, "metadata.txt").writeText(
            buildString {
                append("name=").append(name).append('\n')
                append("created=").append(createdAt).append('\n')
                append("type=").append(type).append('\n')
                append("glb=room.glb\n")
                append("roomWidth=").append(roomWidth).append('\n')
                append("roomHeight=").append(roomHeight).append('\n')
                append("roomDepth=").append(roomDepth).append('\n')
                append("photoOrientation=").append(normalizedOrientation).append('\n')
                append("photoWideAngle=").append(photoWideAngle).append('\n')
                append("previewOnly=").append(previewOnly).append('\n')
            },
        )
        RoomFolderMetadata.writeToFolder(
            folder,
            RoomFolderMetadata.Snapshot(
                name = name,
                createdAt = createdAt,
                type = type,
                photoOrientation = normalizedOrientation,
                photoWideAngle = photoWideAngle,
                roomWidth = roomWidth,
                roomHeight = roomHeight,
                roomDepth = roomDepth,
                roomDimsApproach = "photo-glb",
                roomSceneWidth = roomWidth,
                roomSceneHeight = roomHeight,
                roomSceneDepth = roomDepth,
                previewOnly = previewOnly,
            ),
        )
    }

    private fun uniqueRoomFolder(parent: File, preferredName: String): File {
        var candidate = File(parent, preferredName)
        var suffix = 2
        while (candidate.exists()) {
            candidate = File(parent, "${preferredName}_$suffix")
            suffix++
        }
        return candidate
    }

    private fun normalizedOrientation(value: String): String {
        return if (value.trim().lowercase() == "landscape") "landscape" else "portrait"
    }

    companion object {
        private const val TAG = "PhotoRoomGeneration"
        private const val PREVIEW_DIR = "room_preview"
        private const val ROOMS_DIR = "rooms"

        @Volatile
        private var instance: PhotoRoomGenerationService? = null

        fun getInstance(context: Context): PhotoRoomGenerationService {
            return instance ?: synchronized(this) {
                instance ?: PhotoRoomGenerationService(context.applicationContext).also { instance = it }
            }
        }
    }
}
